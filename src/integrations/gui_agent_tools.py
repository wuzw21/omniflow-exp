"""Framework-neutral tools for external GUI agents.

This module is the single seam between third-party planning frameworks and the
existing OmniFlow/OOB runtime.  It deliberately exposes neither ADB nor a
second device driver: callers can only observe through the configured host,
execute canonical actions through that host, or invoke registered OmniFlow
Functions through the initialized runtime.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import inspect
import json
from typing import Any

from omniflow.core.model import ActionResult, Function, Observation, RunResult
from omniflow.core.schemas import canonicalize_action, load_canonical_action_schema


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
                "strict": True,
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
    ) -> None:
        if host is None:
            raise TypeError("gui_agent_host_required")
        self.host = host
        self.flow = flow
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
        observe = getattr(self.host, "observe", None)
        if not callable(observe):
            raise TypeError("gui_agent_host_observe_required")
        raw_observation = observe(xml=True, screenshot=True, app_info=True)
        if inspect.isawaitable(raw_observation):
            raise TypeError("gui_agent_host_observe_must_be_synchronous")
        return Observation.from_value(raw_observation).to_dict()

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> GuiAgentToolResult:
        normalized_name = str(name or "").strip()
        tools = {tool.name: tool for tool in self.list_tools()}
        tool = tools.get(normalized_name)
        if tool is None:
            raise ValueError(f"gui_agent_tool_unknown:{normalized_name}")
        if tool.kind == "action":
            return await self._call_action(normalized_name, arguments)
        if tool.kind == "function":
            return await self._call_function(normalized_name, arguments)
        raise ValueError(f"gui_agent_tool_kind_unsupported:{tool.kind}")

    def call_tool_sync(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> GuiAgentToolResult:
        return asyncio.run(self.call_tool(name, arguments))

    async def _call_action(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> GuiAgentToolResult:
        action = canonicalize_action(
            {"tool": name, "args": dict(arguments or {})},
            persisted_only=False,
        )
        act = getattr(self.host, "act", None)
        if not callable(act):
            raise TypeError("gui_agent_host_act_required")
        raw_result = act(action)
        if inspect.isawaitable(raw_result):
            raw_result = await raw_result
        result = ActionResult.from_value(raw_result)
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
    ) -> GuiAgentToolResult:
        acall_tool = getattr(self.flow, "acall_tool", None)
        if not callable(acall_tool):
            raise TypeError("gui_agent_function_runtime_required")
        raw_result = acall_tool(
            {"name": name, "arguments": dict(arguments or {})},
            experiment=self.experiment,
        )
        if inspect.isawaitable(raw_result):
            raw_result = await raw_result
        if isinstance(raw_result, RunResult):
            output = {
                "function_id": raw_result.function_id,
                "actions_executed": raw_result.actions_executed,
                "model_calls": raw_result.model_calls,
                "fallback_steps": raw_result.fallback_steps,
                "detail": _json_copy(raw_result.detail),
            }
            return GuiAgentToolResult(
                name=name,
                kind="function",
                success=raw_result.success,
                output=output,
                error=raw_result.error,
            )
        if isinstance(raw_result, dict):
            success = bool(raw_result.get("success", True))
            error = str(raw_result.get("error") or "").strip() or None
            return GuiAgentToolResult(
                name=name,
                kind="function",
                success=success,
                output=_json_copy(raw_result),
                error=error,
            )
        return GuiAgentToolResult(
            name=name,
            kind="function",
            success=True,
            output={"content": "Done" if raw_result is None else str(raw_result)},
        )

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
