"""Framework-neutral tools for external GUI agents.

This module is the single seam between third-party planning frameworks and the
existing OmniFlow/OOB runtime.  It deliberately exposes neither ADB nor a
second device driver: callers can only observe through the configured host,
execute canonical actions through that host, or invoke registered OmniFlow
Functions through the initialized runtime.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
import inspect
import json
from threading import Lock
from typing import Any
import uuid

from jsonschema import validate

from omniflow.core.model import ActionResult, Function, Observation, RunResult
from omniflow.core.schemas import canonicalize_action, load_canonical_action_schema
from omniflow.runtime.control import (
    CURRENT_CONTROL,
    ExecutionControl,
    ExecutionStopped,
    invoke,
)
from omniflow.runtime.protocol import invocation_feedback, observation_payload


@dataclass(frozen=True)
class GuiAgentTool:
    """One model-visible tool at the external-agent seam."""

    name: str
    description: str
    input_schema: dict[str, Any]
    kind: str

    def to_mcp_tool(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": _json_copy(self.input_schema),
        }

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "strict": self.kind != "service",
                "parameters": _json_copy(self.input_schema),
            },
        }


@dataclass(frozen=True)
class GuiAgentToolResult:
    """Framework-neutral result returned for every tool invocation."""

    name: str
    kind: str
    success: bool
    output: dict[str, Any]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "success": self.success,
            "output": _json_copy(self.output),
            "error": self.error,
        }


class GuiAgentToolRuntime:
    """Expose canonical actions and OmniFlow Functions behind one interface."""

    def __init__(
        self,
        *,
        host: Any,
        flow: Any | None = None,
        experiment: str = "external_gui_agent",
        timeout_seconds: float = 600.0,
    ) -> None:
        if host is None:
            raise TypeError("gui_agent_host_required")
        self.host = host
        self.flow = flow
        if getattr(flow, "host", host) is not host:
            raise ValueError("gui_agent_host_must_match_function_host")
        self.timeout_seconds = timeout_seconds
        self._session_lock = Lock()
        self._control: ExecutionControl | None = None
        self._closed = False
        self.session_id = uuid.uuid4().hex
        self._checker_trigger_counts: dict[str, int] = {}
        self._requests: dict[str, tuple[str, GuiAgentToolResult | None]] = {}
        self._task_id: str | None = None
        self._retired_task_ids: set[str] = set()
        self.experiment = str(experiment or "").strip()
        if not self.experiment:
            raise ValueError("gui_agent_experiment_required")

    def list_tools(self) -> tuple[GuiAgentTool, ...]:
        tools = list(_canonical_action_tools())
        used_names = {tool.name for tool in tools}
        for function in self._visible_functions():
            name = str(function.id or "").strip()
            if name in used_names:
                raise ValueError(f"gui_agent_tool_name_collision:{name}")
            tools.append(
                GuiAgentTool(
                    name=name,
                    description=str(function.description or "").strip(),
                    input_schema=_json_copy(function.input_schema),
                    kind="function",
                )
            )
            used_names.add(name)
        return tuple(tools)

    def observe(self) -> dict[str, Any]:
        return asyncio.run(self.aobserve())

    async def aobserve(self) -> dict[str, Any]:
        with self._session():
            observation = await self._observe()
            self._control.check()
            return observation

    async def _observe(self) -> dict[str, Any]:
        observe = getattr(self.host, "observe", None)
        if not callable(observe):
            raise TypeError("gui_agent_host_observe_required")
        raw_observation = await invoke(observe, xml=True, screenshot=True, app_info=True)
        observation = Observation.from_value(raw_observation)
        self._control.observation = observation
        return observation.to_dict()

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        _registered_only: bool = False,
    ) -> GuiAgentToolResult:
        if not _registered_only and name in {tool.name for tool in self.function_tools()}:
            return await self.call_function_tool(name, arguments)
        action_offset = self._control.actions_executed if self._control else 0
        trace_offset = len(self._control.trace) if self._control else 0
        try:
            with self._session():
                result = await self._dispatch_tool(name, arguments, registered_only=_registered_only)
                self._control.check()
                if self._control.effect_unknown:
                    raise ExecutionStopped("effect_unknown")
                feedback = result.output.get("feedback") or {}
                if (feedback.get("control") or {}).get("next") == "stop":
                    self._closed = True
                return result
        except ExecutionStopped as error:
            self._closed = True
            return GuiAgentToolResult(name, "control", False,
                {"feedback": invocation_feedback(self._control.stopped_result(
                    error.reason, action_offset=action_offset, trace_offset=trace_offset))}, error.reason)

    async def _dispatch_tool(self, name: str, arguments: dict[str, Any], *, registered_only=False) -> GuiAgentToolResult:
        normalized_name = str(name or "").strip()
        if registered_only:
            function = self.flow.store.get_function(normalized_name)
            if function is None or not function.agent_visible:
                raise ValueError("function_not_registered")
            return await self._call_function(normalized_name, arguments, registered_only=True)
        tools = {tool.name: tool for tool in self.list_tools()}
        tool = tools.get(normalized_name)
        if tool is None:
            raise ValueError(f"gui_agent_tool_unknown:{normalized_name}")
        if tool.kind == "action":
            return await self._call_action(normalized_name, arguments)
        if tool.kind == "function":
            return await self._call_function(normalized_name, arguments, registered_only=registered_only)
        raise ValueError(f"gui_agent_tool_kind_unsupported:{tool.kind}")

    def call_tool_sync(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> GuiAgentToolResult:
        return asyncio.run(self.call_tool(name, arguments))

    async def execute_request(self, *, session_id: str, request_id: str,
                              tool_name: str, arguments: dict[str, Any], function_only=False) -> GuiAgentToolResult:
        """Deduplicate transport retries inside one live session.

        A restarted process has a different session id. It cannot claim that
        an interrupted side effect did or did not happen on the device.
        """
        if session_id != self.session_id:
            raise ValueError("gui_agent_session_mismatch")
        if not request_id or len(request_id) > 128:
            raise ValueError("gui_agent_request_id_required_max_128")
        signature = json.dumps([tool_name, arguments, function_only], sort_keys=True)
        if request_id in self._requests:
            previous, result = self._requests[request_id]
            if previous != signature:
                raise ValueError("gui_agent_request_id_conflict")
            if result is None:
                raise ValueError("gui_agent_request_in_flight_or_unknown")
            return result
        if len(self._requests) >= 128:
            raise ValueError("gui_agent_session_request_budget_exhausted")
        self._requests[request_id] = (signature, None)
        result = await self.call_tool(tool_name, arguments, _registered_only=function_only)
        self._requests[request_id] = (signature, result)
        # A session retains compact replies for transport deduplication, not a
        # second copy of the complete invocation history and XML trace.
        if self._control is not None:
            self._control.trace.clear()
        return result

    async def _call_action(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> GuiAgentToolResult:
        action = canonicalize_action(
            {"tool": name, "args": dict(arguments or {})},
            persisted_only=False,
        )
        if self.flow is not None:
            return await self._call_function(name, arguments, kind="action")
        act = getattr(self.host, "act", None)
        if not callable(act):
            raise TypeError("gui_agent_host_act_required")
        self._control.check()
        self._control.effect_unknown = True
        raw_result = await invoke(act, action)
        result = ActionResult.from_value(raw_result)
        self._control.effect_unknown = not result.success
        self._control.actions_executed += int(result.success)
        return GuiAgentToolResult(
            name=name,
            kind="action",
            success=result.success,
            output=result.to_dict(),
            error=result.error,
        )

    async def _call_function(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        kind: str = "function",
        registered_only: bool = False,
    ) -> GuiAgentToolResult:
        acall_tool = getattr(self.flow, "acall_tool", None)
        if not registered_only and not callable(acall_tool):
            raise TypeError("gui_agent_function_runtime_required")
        if registered_only:
            raw_result = self.flow.aexecute_function(name, dict(arguments or {}),
                checker_trigger_counts=self._checker_trigger_counts)
        else:
            raw_result = acall_tool(
                {"name": name, "arguments": dict(arguments or {})},
                experiment=self.experiment,
                **({"checker_trigger_counts": self._checker_trigger_counts}
                   if hasattr(self.flow, "checker_library") else {}),
            )
        if inspect.isawaitable(raw_result):
            raw_result = await raw_result
        if isinstance(raw_result, RunResult):
            output = {
                "function_id": raw_result.function_id,
                "actions_executed": raw_result.actions_executed,
                "model_calls": raw_result.model_calls,
                "fallback_steps": raw_result.fallback_steps,
                "detail": _json_copy({key: value for key, value in raw_result.detail.items()
                                      if key not in {"trace", "feedback"}}),
                "observation": observation_payload(raw_result.final_state),
                "feedback": invocation_feedback(raw_result),
            }
            return GuiAgentToolResult(
                name=name,
                kind=kind,
                success=raw_result.success,
                output=output,
                error=raw_result.error,
            )
        if isinstance(raw_result, dict) and isinstance(raw_result.get("success"), bool):
            success = raw_result["success"]
            error = str(raw_result.get("error") or "").strip() or None
            return GuiAgentToolResult(
                name=name,
                kind=kind,
                success=success,
                output=_json_copy(raw_result),
                error=error,
            )
        return GuiAgentToolResult(
            name=name,
            kind=kind,
            success=False,
            output={"content": "Function returned an invalid execution result."},
            error="gui_agent_function_result_invalid",
        )

    @contextmanager
    def _session(self, *, task_id: str | None = None):
        if not self._session_lock.acquire(blocking=False):
            raise ValueError("gui_agent_session_busy")
        token = None
        try:
            if task_id is not None:
                self._select_task(task_id)
            if self._closed:
                raise ValueError("gui_agent_session_closed")
            if self._control is None:
                self._control = ExecutionControl(self.timeout_seconds)
            token = CURRENT_CONTROL.set(self._control)
            self._control.check()
            yield
        finally:
            if token is not None:
                CURRENT_CONTROL.reset(token)
            self._session_lock.release()

    def cancel(self) -> dict[str, Any]:
        self._closed = True
        if self._control is not None:
            self._control.cancelled.set()
        return {"session_id": self.session_id, "status": "cancelled", "in_flight": self._session_lock.locked()}

    def start_session(self) -> dict[str, Any]:
        if not self._session_lock.acquire(blocking=False):
            raise ValueError("gui_agent_session_active")
        try:
            if self._control is not None and not self._closed:
                raise ValueError("gui_agent_session_active")
            self._control = ExecutionControl(self.timeout_seconds)
            self._checker_trigger_counts.clear()
            self._requests.clear()
            self._closed = False
            self.session_id = uuid.uuid4().hex
        finally:
            self._session_lock.release()
        return self.session_status()

    def session_status(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "status": "closed" if self._closed else "active" if self._control is not None else "idle",
            "in_flight": self._session_lock.locked(),
            "remaining_seconds": self._control.remaining if self._control else self.timeout_seconds,
            "actions_executed": self._control.actions_executed if self._control else 0,
            "effect_unknown": self._control.effect_unknown if self._control else False,
        }

    async def finish(self, content: str) -> dict[str, Any]:
        if not str(content).strip():
            raise ValueError("finished_content_required")
        with self._session():
            checker = getattr(self.flow, "completion_checker", None)
            status = "unknown"
            if checker is not None:
                try:
                    status = "verified_success" if float(await invoke(checker)) > 0.5 else "verified_incomplete"
                except Exception:  # noqa: BLE001 -- verifier errors are terminal facts
                    status = "verifier_error"
            self._closed = status != "verified_incomplete"
            self._control.check()
            return {"session_id": self.session_id, "task_status": status,
                    "control": "stop" if self._closed else "host", "content": str(content).strip()}

    def _visible_functions(self) -> tuple[Function, ...]:
        if self.flow is None:
            return ()
        store = getattr(self.flow, "store", None)
        list_functions = getattr(store, "list_functions", None)
        if not callable(list_functions):
            raise TypeError("gui_agent_function_store_required")
        functions = list_functions(include_hidden=False)
        if not all(isinstance(function, Function) for function in functions):
            raise TypeError("gui_agent_function_must_be_canonical")
        return tuple(function for function in functions if function.agent_visible)

    @staticmethod
    def function_tools() -> tuple[GuiAgentTool, ...]:
        """The entire model-visible OmniFlow service: recall and execute."""
        string = {"type": "string", "minLength": 1, "maxLength": 128}
        def schema(properties):
            return {"type": "object", "properties": properties,
                    "required": list(properties), "additionalProperties": False}
        return (
            GuiAgentTool("omniflow_recall",
                "Recall registered Functions for the current page and goal. Use one task_id per logical task; a new id explicitly starts a new task. Returns parameter schemas and session_id without executing device actions.",
                schema({"task_id": string, "goal": {"type": "string", "minLength": 1},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 32}}), "service"),
            GuiAgentTool("omniflow_execute",
                "Execute one registered Function using its input schema. Returns partial progress and current state to the host. Reuse request_id only for identical transport retries; never replay a successful prefix blindly.",
                schema({"session_id": string, "request_id": string, "function_id": string,
                        "arguments": {"type": "object"}}), "service"),
        )

    def harness_tools(self) -> tuple[GuiAgentTool, ...]:
        """Legacy native-action adapters add only the two Function service tools."""
        return (*_canonical_action_tools(), *self.function_tools())

    def _select_task(self, task_id: str) -> None:
        """Called only while holding the session lock through the whole recall."""
        if task_id == self._task_id:
            return
        if task_id in self._retired_task_ids:
            raise ValueError("gui_agent_task_id_retired")
        if len(self._retired_task_ids) >= 128:
            raise ValueError("gui_agent_connection_task_budget_exhausted")
        if self._task_id is not None:
            self._retired_task_ids.add(self._task_id)
        self._task_id = task_id
        self._control = ExecutionControl(self.timeout_seconds)
        self._closed = False
        self.session_id = uuid.uuid4().hex
        self._requests.clear()
        self._checker_trigger_counts.clear()

    async def call_function_tool(self, name: str, arguments: dict[str, Any]) -> GuiAgentToolResult:
        definitions = {tool.name: tool for tool in self.function_tools()}
        if name not in definitions:
            raise ValueError("function_service_tool_unknown")
        validate(arguments, definitions[name].input_schema)
        if self.flow is None:
            raise TypeError("gui_agent_function_runtime_required")
        if name == "omniflow_execute":
            return await self.execute_request(session_id=arguments["session_id"],
                request_id=arguments["request_id"], tool_name=arguments["function_id"],
                arguments=arguments["arguments"], function_only=True)
        with self._session(task_id=arguments["task_id"]):
            result = await self.flow.arecall(arguments["goal"], limit=arguments["limit"])
            feedback = invocation_feedback(result)
            if feedback["control"]["next"] == "stop":
                self._closed = True
            return GuiAgentToolResult(name, "service", result.success, {
                "session_id": self.session_id, "task_id": self._task_id,
                **result.detail.get("recall", {}), "feedback": feedback,
                "observation": observation_payload(result.final_state),
            }, result.error)


def _canonical_action_tools() -> tuple[GuiAgentTool, ...]:
    tools: list[GuiAgentTool] = []
    for raw_tool in load_canonical_action_schema().get("tools") or ():
        if (
            not isinstance(raw_tool, dict)
            or raw_tool.get("kind") != "action"
            or raw_tool.get("model_visible") is False
        ):
            continue
        properties: dict[str, Any] = {}
        required: list[str] = []
        for raw_argument in raw_tool.get("args") or ():
            if not isinstance(raw_argument, dict):
                continue
            name = str(raw_argument.get("name") or "").strip()
            if not name:
                continue
            argument_type = str(raw_argument.get("type") or "string")
            schema: dict[str, Any] = {
                "type": "array" if argument_type == "string_array" else argument_type
            }
            if schema["type"] == "array":
                schema["items"] = {"type": "string"}
            if raw_argument.get("enum_values"):
                schema["enum"] = list(raw_argument["enum_values"])
            for bound in ("minimum", "maximum"):
                if raw_argument.get(bound) is not None:
                    schema[bound] = raw_argument[bound]
            description = raw_argument.get("description")
            if isinstance(description, dict) and description.get("en_us"):
                schema["description"] = str(description["en_us"])
            properties[name] = schema
            if raw_argument.get("required"):
                required.append(name)
        description = raw_tool.get("description")
        tools.append(
            GuiAgentTool(
                name=str(raw_tool.get("name") or "").strip(),
                description=(
                    str(description.get("en_us") or "")
                    if isinstance(description, dict)
                    else str(description or "")
                ),
                input_schema={
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
                kind="action",
            )
        )
    return tuple(tools)


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


__all__ = ["GuiAgentTool", "GuiAgentToolResult", "GuiAgentToolRuntime"]
