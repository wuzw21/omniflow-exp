from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, TextIO

from omniflow.catalog import CatalogSnapshot, load_catalog, load_default_catalog
from omniflow.core.config import OmniFlowConfig, RuntimeSettings
from omniflow.core.model import (
    Action,
    ActionResult,
    Function,
    Observation,
    ToolCall,
)
from omniflow.core.trajectory import canonicalize_run_log
from omniflow.functions.artifact import parse_function_artifact
from omniflow.functions.compiler import compile_runlog_to_store
from omniflow.runlog import import_run_log_evidence
from omniflow.runtime.engine import InputRequired, OmniFlow
from omniflow.vlm.gui import (
    ModelToolCallError,
    build_model_turn_request,
    parse_model_turn_response,
)
from omniflow.vlm.guidance import resolve_step_guidance

PROTOCOL_VERSION = "2025-11-25"
_DEFAULT_GUI_MAX_STEPS = 20
_MAX_GUI_MAX_STEPS = 64
_MODEL_TOOL_CALL_ATTEMPTS = 2

_FUNCTION_CATALOG_ACTIONS = {
    "list_functions": "list",
    "get_function": "get",
    "delete_function": "delete",
    "clear_functions": "clear",
}

_MANAGEMENT_TOOL_NAMES = frozenset(
    {
        *_FUNCTION_CATALOG_ACTIONS,
        "save_function",
        "list_run_logs",
        "get_run_log",
        "get_run_log_state",
    }
)

_RUN_GUI_TOOL = "run_gui"


class JsonLineBridge:
    def __init__(
        self,
        store_path: str | Path,
        *,
        reader: TextIO = sys.stdin,
        writer: TextIO = sys.stdout,
        catalog: CatalogSnapshot | None = None,
    ):
        self.reader = reader
        self.writer = writer
        self.catalog = catalog
        self.flow = OmniFlow(store_path, catalog=catalog)
        self._host_call_index = 0

    def serve_forever(self) -> None:
        for line in self.reader:
            request = self._parse(line)
            if request is not None and self._serve_request(request):
                return

    def serve_once(self) -> None:
        for line in self.reader:
            request = self._parse(line)
            if request is not None:
                self._serve_request(request)
                return

    def _serve_request(self, request: dict[str, Any]) -> bool:
        request_id = request.get("id")
        try:
            if request.get("jsonrpc") != "2.0":
                raise ValueError("jsonrpc_2_required")
            method = str(request.get("method") or "")
            if not method:
                raise ValueError("jsonrpc_method_required")
            if request_id is None:
                if method == "notifications/initialized":
                    return False
                raise ValueError("jsonrpc_request_id_required")
            if not isinstance(request_id, (str, int)) or isinstance(request_id, bool):
                raise ValueError("jsonrpc_request_id_invalid")
            result = self._handle(str(request_id), method, request.get("params"))
            self._response(request_id, result)
        except Exception as error:  # noqa: BLE001
            if request_id is not None:
                self._error_response(request_id, error)
        return False

    def _handle(self, request_id: str, operation: str, payload: Any) -> Any:
        body = payload if isinstance(payload, dict) else {}
        if operation == "initialize":
            _require_contract(
                body,
                {"protocolVersion", "capabilities", "clientInfo"},
                {
                    "protocolVersion",
                    "capabilities",
                    "clientInfo",
                },
            )
            if body["protocolVersion"] != PROTOCOL_VERSION:
                raise ValueError("unsupported_protocol_version")
            self.flow.store.reload()
            return {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {
                    "name": "omniflow",
                    "version": str(_bridge_identity()["runtime_version"]),
                },
                "_meta": {
                    "omniflow/runtime": {
                        **_bridge_identity(),
                        "protocol_version": PROTOCOL_VERSION,
                    }
                },
            }
        if operation == "tools/list":
            _require_contract(body, set(), {"cursor"})
            return {
                "tools": [
                    _management_tool_definition(name)
                    for name in sorted({*_MANAGEMENT_TOOL_NAMES, _RUN_GUI_TOOL})
                ]
            }
        if operation == "tools/call":
            return self._call_tool(request_id, body)
        raise ValueError(f"unsupported_operation:{operation}")

    def _call_tool(
        self,
        request_id: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        _require_contract(body, {"name"}, {"name", "arguments", "_meta"})
        call = ToolCall.from_value(
            {
                "name": body.get("name"),
                "arguments": body.get("arguments") or {},
            }
        )
        tool = call.name
        args = call.arguments

        if tool == _RUN_GUI_TOOL:
            metadata = body.get("_meta")
            run_metadata = dict(metadata) if isinstance(metadata, dict) else {}
            return self._run(request_id, {**args, **run_metadata})
        if tool not in _MANAGEMENT_TOOL_NAMES:
            return self._execute_tool(request_id, call, body.get("_meta"))

        catalog_action = _FUNCTION_CATALOG_ACTIONS.get(tool)
        if catalog_action is not None:
            result = self._catalog({**args, "action": catalog_action})
            if tool == "get_function" and result.get("success") is True:
                function = result.get("function")
                return dict(function) if isinstance(function, dict) else {}
            return result
        if tool == "save_function":
            return self._save_function(request_id, args)
        if tool == "list_run_logs":
            response = self.host_call(
                request_id,
                "list_run_logs",
                {
                    "limit": max(1, min(_int_arg(args.get("limit"), 50), 200)),
                    "offset": max(0, _int_arg(args.get("offset"), 0)),
                    "source": str(args.get("source") or "").strip(),
                    "status": str(args.get("status") or "").strip(),
                    "model": str(args.get("model") or "").strip(),
                    "query": str(args.get("query") or "").strip(),
                },
            )
            if isinstance(response, dict):
                return response
            return _management_error(
                "RUN_LOG_LIST_INVALID",
                "RunLog list response must be an object",
            )
        if tool == "get_run_log":
            run_id = str(args.get("run_id") or "").strip()
            if not run_id:
                return _management_error("RUN_LOG_ID_EMPTY", "run_id is required")
            response = self.host_call(
                request_id,
                "get_run_log",
                {"run_id": run_id},
            )
            if isinstance(response, dict):
                return response
            return _management_error(
                "RUN_LOG_NOT_FOUND",
                f"RunLog not found: {run_id}",
                run_id=run_id,
            )
        if tool == "get_run_log_state":
            state_id = str(args.get("state_id") or "").strip()
            if not state_id:
                return _management_error(
                    "STATE_ID_EMPTY",
                    "state_id is required",
                )
            response = self.host_call(
                request_id,
                "get_state",
                {"state_id": state_id},
            )
            if isinstance(response, dict):
                return response
            return _management_error(
                "STATE_NOT_FOUND",
                f"RunLog state not found: {state_id}",
            )
        return _management_error(
            "UNKNOWN_FUNCTION_MANAGEMENT_TOOL",
            f"Unknown Function management tool: {tool}",
        )

    def _run(
        self,
        request_id: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        goal = str(body.get("goal") or "").strip()
        if not goal:
            return _run_error(
                body,
                code="RUN_REQUEST_INVALID",
                message="run_goal_required",
            )
        model = str(body.get("model") or "").strip()
        if not model:
            return _run_error(
                body,
                code="MODEL_REQUIRED",
                message="model_required",
            )
        max_steps = _gui_max_steps(body.get("max_steps"))
        host = _BridgeHost(
            self,
            request_id,
            defer_user_input=body.get("defer_user_input") is True,
        )
        installed_apps = host.installed_apps()
        planner = _BridgePlanner(
            self,
            request_id,
            host,
            model=model,
            target_package_name=str(body.get("target_package_name") or ""),
            step_skill_guidance=resolve_step_guidance(
                goal,
                str(body.get("step_skill_guidance") or ""),
            ),
            max_steps=max_steps,
        )
        flow = OmniFlow(
            self.flow.store.path,
            host=host,
            planner=planner,
            installed_apps=installed_apps,
            config=OmniFlowConfig(runtime=RuntimeSettings(max_steps=max_steps)),
            catalog=self.catalog,
        )
        return _run_result(flow.run(goal), body=body, function=None)

    def _execute_tool(
        self,
        request_id: str,
        tool_call: ToolCall,
        metadata: Any,
    ) -> dict[str, Any]:
        run_metadata = dict(metadata) if isinstance(metadata, dict) else {}
        max_steps = _gui_max_steps(run_metadata.get("max_steps"))
        host = _BridgeHost(
            self,
            request_id,
            defer_user_input=run_metadata.get("defer_user_input") is True,
        )
        model = str(run_metadata.get("model") or "").strip()
        planner = (
            _BridgePlanner(
                self,
                request_id,
                host,
                model=model,
                target_package_name=str(
                    run_metadata.get("target_package_name") or ""
                ),
                step_skill_guidance=str(
                    run_metadata.get("step_skill_guidance") or ""
                ),
                max_steps=max_steps,
            )
            if model
            else None
        )
        flow = OmniFlow(
            self.flow.store.path,
            host=host,
            planner=planner,
            installed_apps=host.installed_apps(),
            config=OmniFlowConfig(runtime=RuntimeSettings(max_steps=max_steps)),
            catalog=self.catalog,
        )
        function = flow.store.get_function(tool_call.name)
        result = flow.call_tool(tool_call)
        return _run_result(result, body=run_metadata, function=function)

    def _catalog(self, body: dict[str, Any]) -> dict[str, Any]:
        self.flow.store.reload()
        action = str(body.get("action") or "list")
        if action == "list":
            include_hidden = body.get("include_hidden") is True
            limit = max(1, min(int(body.get("limit") or 100), 500))
            offset = max(0, int(body.get("offset") or 0))
            functions = self.flow.store.list_functions(
                offset=offset,
                limit=limit,
                include_hidden=include_hidden,
            )
            total = sum(
                1
                for item in self.flow.store.functions.values()
                if include_hidden or item.agent_visible
            )
            return {
                "success": True,
                "functions": [item.to_dict() for item in functions],
                "count": len(functions),
                "total": total,
                "limit": limit,
                "offset": offset,
                "next_offset": offset + len(functions),
                "has_more": offset + len(functions) < total,
                "include_hidden": include_hidden,
            }
        if action == "get":
            function_id = str(body.get("function_id") or "").strip()
            if not function_id:
                return _management_error("FUNCTION_ID_EMPTY", "function_id is required")
            function = self.flow.store.get_function(function_id)
            if function is None:
                return _management_error(
                    "FUNCTION_NOT_FOUND",
                    f"Function not found: {function_id}",
                    function_id=function_id,
                )
            return {
                "success": True,
                "function_id": function_id,
                "function": function.to_dict(),
                "source": "omniflow_python",
            }
        if action == "delete":
            function_id = str(body.get("function_id") or "").strip()
            deleted = self.flow.store.delete_function(function_id)
            return {
                "success": deleted,
                "function_id": function_id,
                "deleted": deleted,
                "source": "omniflow_python",
            }
        if action == "clear":
            if body.get("confirm") is not True:
                return _management_error(
                    "FUNCTION_CLEAR_CONFIRMATION_REQUIRED",
                    "Set confirm=true to clear all registered Functions",
                )
            return {
                "success": True,
                "deleted": True,
                "deleted_count": self.flow.store.clear_functions(),
                "source": "omniflow_python",
            }
        raise ValueError(f"unsupported_catalog_action:{action}")

    def _save_function(
        self,
        request_id: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        _require_contract(
            body,
            set(),
            {
                "run_id",
                "run_log",
                "function",
                "arguments",
                "agent_visible",
            },
        )
        supplied_function = body.get("function")
        run_id = str(body.get("run_id") or "").strip()
        supplied_run_log = body.get("run_log")
        if supplied_function is not None and not isinstance(supplied_function, dict):
            return _save_error("FUNCTION_SCHEMA_INVALID", "function must be an object")
        supplied_arguments = body.get("arguments")
        if supplied_arguments is not None and not isinstance(supplied_arguments, dict):
            return _save_error("FUNCTION_ARGUMENTS_INVALID", "arguments must be an object")
        if supplied_run_log is not None:
            try:
                run_log = _load_run_log_input(supplied_run_log)
            except (OSError, ValueError) as error:
                return _save_error("RUN_LOG_INVALID", str(error))
            embedded_run_id = str(run_log.get("run_id") or "").strip()
            if run_id and run_id != embedded_run_id:
                return _save_error("RUN_LOG_ID_MISMATCH", "run_id does not match run_log")
            run_id = embedded_run_id
        else:
            run_log = None
        if not run_id and run_log is None:
            if supplied_function is None:
                return _save_error(
                    "FUNCTION_SAVE_INPUT_REQUIRED",
                    "run_id, run_log, or function is required",
                )
            try:
                function = parse_function_artifact(supplied_function)
                saved = self.flow.store.put_function(function)
            except ValueError as error:
                return _save_error("FUNCTION_SCHEMA_INVALID", str(error))
            return _save_success(saved)

        if run_log is None:
            run_log = self.host_call(request_id, "get_run_log", {"run_id": run_id})
        if not isinstance(run_log, dict):
            return _save_error(
                "RUN_LOG_NOT_FOUND",
                f"RunLog not found: {run_id}",
            )
        if run_log.get("error_code"):
            return _save_error(
                str(run_log["error_code"]),
                str(run_log.get("error_message") or f"RunLog not found: {run_id}"),
            )

        try:
            with tempfile.TemporaryDirectory(prefix="omniflow-compile-") as output_root:
                function_bundle = None
                if supplied_function is not None:
                    function_id = str(supplied_function.get("function_id") or "").strip()
                    function_bundle = {
                        "schema_version": "omniflow.function-bundle.v2",
                        "run_id": run_id,
                        "arguments": {
                            function_id: dict(supplied_arguments or {})
                        },
                        "functions": [supplied_function],
                    }
                compile_options: dict[str, Any] = {}
                if supplied_run_log is not None:
                    _, source_states = import_run_log_evidence(run_log)
                    compile_options["source_states"] = source_states
                else:
                    compile_options["state_loader"] = lambda state_id: self.host_call(
                        request_id,
                        "get_state",
                        {"state_id": state_id},
                    )
                report = compile_runlog_to_store(
                    run_log,
                    output_root,
                    function_bundle=function_bundle,
                    **compile_options,
                )
                compiled = OmniFlow(Path(output_root) / "store.json")
                function_id = next(iter(report["function_ids"]), "")
                function = compiled.store.get_function(function_id)
        except ValueError as error:
            return _save_compile_error(error)
        if function is None:
            return _save_error(
                "RUN_LOG_NO_REPLAYABLE_STEPS",
                "RunLog has no replayable steps",
            )

        value = function.to_dict()
        if "agent_visible" in body:
            value["agent_visible"] = body.get("agent_visible") is True
        try:
            value = parse_function_artifact(value).to_dict()
        except ValueError as error:
            return _save_error("FUNCTION_SCHEMA_INVALID", str(error))
        saved = self.flow.store.put_function(value)
        return _save_success(saved)

    def host_call(self, request_id: str, method: str, payload: dict[str, Any]) -> Any:
        self._host_call_index += 1
        call_id = f"{request_id}:host:{self._host_call_index}"
        self._write(
            {
                "jsonrpc": "2.0",
                "id": call_id,
                "method": f"omniflow/{method}",
                "params": payload,
            }
        )
        for line in self.reader:
            response = self._parse(line)
            if response is None:
                continue
            if response.get("jsonrpc") != "2.0":
                raise RuntimeError("host_response_jsonrpc_invalid")
            if str(response.get("id") or "") != call_id:
                raise RuntimeError("host_response_out_of_order")
            if "error" in response:
                error = response.get("error")
                raise RuntimeError(
                    str(
                        error.get("message")
                        if isinstance(error, dict)
                        else error or "host_call_failed"
                    )
                )
            return response.get("result")
        raise EOFError("host_response_missing")

    def _response(
        self,
        request_id: str | int,
        result: Any = None,
    ) -> None:
        self._write(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result,
            }
        )

    def _error_response(self, request_id: str | int, error: Exception) -> None:
        self._write(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": str(error) or type(error).__name__,
                    "data": {"type": type(error).__name__},
                },
            }
        )

    def _write(self, value: dict[str, Any]) -> None:
        self.writer.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        self.writer.write("\n")
        self.writer.flush()

    @staticmethod
    def _parse(line: str) -> dict[str, Any] | None:
        stripped = line.strip()
        if not stripped:
            return None
        value = json.loads(stripped)
        if not isinstance(value, dict):
            raise ValueError("bridge_message_must_be_object")
        return value


class _BridgeHost:
    def __init__(
        self,
        bridge: JsonLineBridge,
        request_id: str,
        *,
        defer_user_input: bool = False,
    ):
        self.bridge = bridge
        self.request_id = request_id
        self.defer_user_input = bool(defer_user_input)
        self.current_observation: Observation | None = None
        self.current_action_metadata: dict[str, Any] = {}

    def observe(self, **kwargs: Any) -> Observation:
        self.current_observation = _state_observation(
            self.bridge.host_call(self.request_id, "observe", kwargs)
        )
        return self.current_observation

    def installed_apps(self) -> dict[str, str]:
        response = self.bridge.host_call(
            self.request_id,
            "installed_apps",
            {},
        )
        if not isinstance(response, dict) or not isinstance(response.get("apps"), dict):
            raise ValueError("installed_apps_response_invalid")
        return {
            str(label): str(package)
            for label, package in response["apps"].items()
            if str(label).strip() and str(package).strip()
        }

    def act(self, action: Action) -> ActionResult:
        if self.current_observation is None:
            raise RuntimeError("host_action_state_required")
        return ActionResult.from_value(
            self.bridge.host_call(
                self.request_id,
                "act",
                {
                    "action": action.to_dict(),
                    "metadata": dict(self.current_action_metadata),
                    "state": _state_from_observation(
                        self.current_observation,
                        include_xml=True,
                    ),
                },
            )
        )

    def request_input(self, question: str) -> str:
        if self.defer_user_input:
            raise InputRequired(question)
        response = self.bridge.host_call(
            self.request_id,
            "request_input",
            {"question": str(question)},
        )
        if not isinstance(response, dict):
            raise ValueError("request_input_response_invalid")
        return str(response.get("value") or "")

    def get_state(self, source_state_id: str) -> Observation:
        return _state_observation(
            self.bridge.host_call(
                self.request_id, "get_state", {"state_id": source_state_id}
            )
        )

    def record_step(self, fact: dict[str, Any]) -> dict[str, Any]:
        response = self.bridge.host_call(
            self.request_id,
            "record_step",
            {"fact": fact},
        )
        if not isinstance(response, dict):
            raise ValueError("record_step_response_invalid")
        return response


class _BridgePlanner:
    def __init__(
        self,
        bridge: JsonLineBridge,
        request_id: str,
        host: _BridgeHost,
        *,
        model: str,
        target_package_name: str,
        step_skill_guidance: str,
        max_steps: int,
    ):
        self.bridge = bridge
        self.request_id = request_id
        self.host = host
        self.model = str(model).strip()
        self.target_package_name = str(target_package_name).strip()
        self.step_skill_guidance = str(step_skill_guidance)
        self.max_steps = int(max_steps)
        self._metadata: dict[str, Any] = {}
        self._rejected_tool_calls: list[dict[str, Any]] = []
        self._turn_index = 0

    def one_step_tool_call(
        self,
        goal: str,
        observation: Observation,
        functions: tuple[Function, ...] = (),
        installed_apps: dict[str, str] | None = None,
    ) -> ToolCall:
        state = _planner_state(observation)
        validation_error = ""
        retry_tool_name = ""
        rejected_tool_call: dict[str, Any] | None = None
        lightweight_retry = False
        self._rejected_tool_calls.clear()
        for attempt in range(_MODEL_TOOL_CALL_ATTEMPTS):
            self._turn_index += 1
            request = build_model_turn_request(
                goal=str(goal),
                model=self.model,
                state=state,
                target_package_name=self.target_package_name,
                step_skill_guidance=self.step_skill_guidance,
                installed_apps=installed_apps or {},
                functions=functions,
                max_steps=self.max_steps,
                turn_index=self._turn_index,
                validation_error=validation_error,
                retry_tool_name=retry_tool_name,
                rejected_tool_call=rejected_tool_call,
                lightweight_retry=lightweight_retry,
            )
            response = self.bridge.host_call(
                self.request_id,
                "model_turn",
                {
                    "goal": str(goal),
                    "model": self.model,
                    "state": state,
                    "target_package_name": self.target_package_name,
                    "step_skill_guidance": self.step_skill_guidance,
                    "max_steps": self.max_steps,
                    "request": request,
                },
            )
            try:
                tool_call, metadata = parse_model_turn_response(
                    response,
                    requested_model=self.model,
                    turn_index=self._turn_index,
                    functions=functions,
                    display=(
                        state.get("display")
                        if isinstance(state.get("display"), dict)
                        else None
                    ),
                )
                break
            except ModelToolCallError as error:
                rejected_entry = {
                    "turn_index": self._turn_index,
                    "tool": error.tool_name or None,
                    "error": str(error),
                }
                if error.arguments is not None:
                    rejected_entry["arguments"] = error.arguments
                self._rejected_tool_calls.append(rejected_entry)
                if attempt == _MODEL_TOOL_CALL_ATTEMPTS - 1:
                    self._metadata = {
                        "rejected_tool_calls": list(self._rejected_tool_calls)
                    }
                    raise
                validation_error = str(error)
                retry_tool_name = error.tool_name
                rejected_tool_call = {
                    "tool": error.tool_name or None,
                    "arguments": error.arguments,
                }
                lightweight_retry = error.code.endswith(
                    "expected_one_native_tool_call:got_0"
                )
        if self._rejected_tool_calls:
            metadata["rejected_tool_calls"] = list(self._rejected_tool_calls)
        self._metadata = metadata
        self.host.current_action_metadata = dict(self._metadata)
        return tool_call

    def take_metadata(self) -> dict[str, Any]:
        metadata = dict(self._metadata)
        self._metadata.clear()
        return metadata


def _planner_state(observation: Observation) -> dict[str, Any]:
    state = observation.to_dict()
    state["state_id"] = str(observation.extra.get("state_id") or "").strip()
    for key in ("display", "screenshot_path"):
        if observation.extra.get(key) is not None:
            state[key] = observation.extra[key]
    state["extra"] = {
        key: value
        for key, value in observation.extra.items()
        if key not in {"state_id", "display", "screenshot_path"}
    }
    return {key: value for key, value in state.items() if value is not None}


def _state_observation(value: Any) -> Observation:
    if not isinstance(value, dict):
        raise ValueError("state_must_be_object")
    allowed = {
        "state_id",
        "xml",
        "package_name",
        "activity_name",
        "display",
        "screenshot_path",
        "image_base64",
        "extra",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"state_unknown_fields:{','.join(unknown)}")
    state_id = str(value.get("state_id") or "").strip()
    if not state_id:
        raise ValueError("state_id_required")
    return Observation.from_value({**value, "state_id": state_id})


def _require_contract(
    value: dict[str, Any],
    required: set[str],
    allowed: set[str],
) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"request_missing_fields:{','.join(missing)}")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"request_unknown_fields:{','.join(unknown)}")


def _management_tool_definition(name: str) -> dict[str, Any]:
    if name != "save_function":
        return {"name": name, "inputSchema": {"type": "object"}}
    return {
        "name": name,
        "description": (
            "Save one Agent-authored reusable Function. Analyze the RunLog goal, "
            "ordered actions, and existing action metadata, then pass run_id or "
            "run_log together with one complete Function and its exact source "
            "arguments. The Function description must state its atomic effect, "
            "parameter meanings, fixed choices, and excluded work. Preserve recorded "
            "actions in order; do not invent actions, UI evidence, or checker rules."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "run_log": {
                    "anyOf": [{"type": "object"}, {"type": "string"}]
                },
                "function": {"type": "object"},
                "arguments": {"type": "object"},
                "agent_visible": {"type": "boolean"},
            },
            "additionalProperties": False,
            "anyOf": [
                {"required": ["run_id"]},
                {"required": ["run_log"]},
                {"required": ["function"]},
            ],
        },
    }


def _load_run_log_input(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return canonicalize_run_log(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("run_log must be an object or absolute JSON path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("run_log path must be absolute")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("run_log file must contain one object")
    return canonicalize_run_log(loaded)


def _int_arg(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _save_compile_error(error: ValueError) -> dict[str, Any]:
    message = str(error)
    code = {
        "successful_source_actions_required": "RUN_LOG_NO_REPLAYABLE_STEPS",
        "semantic_functions_required": "RUN_LOG_NO_REPLAYABLE_STEPS",
        "function_author_agent_required": "FUNCTION_AUTHOR_AGENT_REQUIRED",
        "successful_source_goal_required": "RUN_LOG_GOAL_EMPTY",
    }.get(message, "RUN_LOG_COMPILE_FAILED")
    user_message = {
        "RUN_LOG_NO_REPLAYABLE_STEPS": "RunLog has no replayable steps",
        "RUN_LOG_GOAL_EMPTY": "RunLog goal is required",
        "FUNCTION_AUTHOR_AGENT_REQUIRED": (
            "An Agent-authored Function and source arguments are required"
        ),
    }.get(code, message)
    return _save_error(code, user_message)


def _save_success(function: Function) -> dict[str, Any]:
    return {
        "success": True,
        "function_id": function.function_id,
        "function": function.to_dict(),
        "error": None,
    }


def _save_error(code: str, message: str) -> dict[str, Any]:
    return {
        "success": False,
        "function_id": "",
        "function": None,
        "error": {"code": code, "message": message},
    }


def _management_error(
    code: str,
    message: str,
    *,
    function_id: str = "",
    run_id: str = "",
) -> dict[str, Any]:
    return {
        "success": False,
        "accepted": False,
        "status": "rejected",
        "registered": False,
        "function_id": function_id,
        "run_id": run_id,
        "function": None,
        "error": message,
        "error_code": code,
        "error_message": message,
        "source": "omniflow_python",
    }


def _gui_max_steps(value: Any) -> int:
    return max(
        1,
        min(int(value or _DEFAULT_GUI_MAX_STEPS), _MAX_GUI_MAX_STEPS),
    )


def _run_result(
    result,
    *,
    body: dict[str, Any],
    function: Function | None,
) -> dict[str, Any]:
    finished_at_ms = int(time.time() * 1000)
    started_at_ms = int(body.get("started_at_ms") or finished_at_ms)
    trace = [
        step for step in (result.detail.get("trace") or []) if isinstance(step, dict)
    ]
    successful_steps = sum(
        1
        for step in trace
        if isinstance(step.get("result"), dict)
        and step["result"].get("success") is True
    )
    failed_step_index = next(
        (
            index
            for index, step in enumerate(trace)
            if not isinstance(step.get("result"), dict)
            or step["result"].get("success") is not True
        ),
        None,
    )
    current_step_index = (
        failed_step_index
        if failed_step_index is not None
        else len(trace) - 1
        if trace
        else None
    )
    error = str(result.error or "")
    error_code = None
    if not result.success:
        error_code = (
            "FUNCTION_ARGUMENTS_MISSING"
            if error.startswith("function_arguments_invalid:missing:")
            else "FUNCTION_CALL_FAILED"
        )
    function_resolution = result.detail.get("function_resolution") or None
    recalled_function_id = str(result.function_id or "").strip()
    recall_hit = bool(recalled_function_id)
    post_run_actions: list[dict[str, Any]] = []
    done_reason = result.detail.get("done_reason") or (
        "finished" if result.success else "error"
    )
    if (
        result.success
        and done_reason == "finished"
        and not recall_hit
        and int(result.actions_executed) > 0
    ):
        post_run_actions.append(
            {
                "name": "save_function",
                "arguments": {
                    "run_id": str(body.get("run_id") or ""),
                    "agent_visible": True,
                },
            }
        )
    payload: dict[str, Any] = {
        "success": result.success,
        "status": "succeeded" if result.success else "failed",
        "run_id": str(body.get("run_id") or ""),
        "function_id": str(result.function_id or ""),
        "name": function.name if function else "",
        "description": function.description if function else "",
        "source": "function" if function else "vlm",
        "step_count": len(function.steps) if function else 0,
        "success_step_count": successful_steps,
        "completed_step_count": len(trace),
        "actions_executed": int(result.actions_executed),
        "steps": trace,
        "failed_step_index": failed_step_index,
        "current_step_index": current_step_index,
        "started_at_ms": started_at_ms,
        "finished_at_ms": finished_at_ms,
        "duration_ms": max(0, finished_at_ms - started_at_ms),
        "error_code": error_code,
        "error_message": error or None,
        "done_reason": done_reason,
        "finished_content": result.detail.get("finished_content"),
        "model": str(body.get("model") or "") or None,
        "model_calls": int(result.model_calls),
        "fallback_steps": int(result.fallback_steps),
        "completion_review_calls": int(
            result.detail.get("completion_review_calls") or 0
        ),
        "planner_diagnostics": result.detail.get("planner_diagnostics") or None,
        "function_resolution": function_resolution,
        "recall_hit": recall_hit,
        "recalled_function_id": recalled_function_id or None,
        "post_run_actions": post_run_actions or None,
        "runtime_limits": result.detail.get("runtime_limits") or None,
        "missing_required_arguments": (
            [
                value
                for value in error.removeprefix(
                    "function_arguments_invalid:missing:"
                ).split(",")
                if value
            ]
            if error.startswith("function_arguments_invalid:missing:")
            else None
        ),
        "final_state": _state_from_observation(result.final_state),
    }
    return {key: value for key, value in payload.items() if value is not None}


def _run_error(
    body: dict[str, Any],
    *,
    code: str,
    message: str,
) -> dict[str, Any]:
    finished_at_ms = int(time.time() * 1000)
    started_at_ms = int(body.get("started_at_ms") or finished_at_ms)
    return {
        "success": False,
        "status": "failed",
        "run_id": str(body.get("run_id") or ""),
        "function_id": "",
        "source": "omniflow",
        "step_count": 0,
        "success_step_count": 0,
        "completed_step_count": 0,
        "actions_executed": 0,
        "steps": [],
        "started_at_ms": started_at_ms,
        "finished_at_ms": finished_at_ms,
        "duration_ms": max(0, finished_at_ms - started_at_ms),
        "error_code": code,
        "error_message": message,
        "done_reason": "error",
    }


def _state_from_observation(
    value: Observation | None,
    *,
    include_xml: bool = False,
) -> dict[str, Any] | None:
    if value is None:
        return None
    display = value.extra.get("display")
    canonical_display = dict(display) if isinstance(display, dict) else None
    identity = "\0".join(
        (
            str(value.package_name or ""),
            str(value.activity_name or ""),
            str(value.xml or ""),
            str((canonical_display or {}).get("width") or ""),
            str((canonical_display or {}).get("height") or ""),
        )
    )
    state: dict[str, Any] = {
        "state_id": str(value.extra.get("state_id") or "").strip()
        or "state_" + hashlib.sha256(identity.encode()).hexdigest()[:20],
    }
    if value.package_name:
        state["package_name"] = str(value.package_name)
    if value.activity_name:
        state["activity_name"] = str(value.activity_name)
    if canonical_display is not None:
        state["display"] = canonical_display
    canonical = _canonicalize_bridge_state(state)
    if include_xml and value.xml:
        canonical["xml"] = value.xml
    return canonical


def _canonicalize_bridge_state(value: dict[str, Any]) -> dict[str, Any]:
    state_id = str(value.get("state_id") or "").strip()
    if not state_id:
        raise ValueError("bridge_state_id_required")
    canonical: dict[str, Any] = {"state_id": state_id}
    for field in ("package_name", "activity_name"):
        item = value.get(field)
        if item is not None:
            canonical[field] = str(item)
    display = value.get("display")
    if display is not None:
        if not isinstance(display, dict) or set(display) != {"width", "height"}:
            raise ValueError("bridge_state_display_invalid")
        width = display.get("width")
        height = display.get("height")
        if (
            not isinstance(width, int)
            or isinstance(width, bool)
            or width <= 0
            or not isinstance(height, int)
            or isinstance(height, bool)
            or height <= 0
        ):
            raise ValueError("bridge_state_display_invalid")
        canonical["display"] = {"width": width, "height": height}
    return canonical


def _bridge_identity() -> dict[str, Any]:
    source_file = Path(__file__).resolve()
    contract = next(
        (
            parent / "schemas/oob/omniflow_android_bridge.v2.json"
            for parent in source_file.parents
            if (parent / "schemas/oob/omniflow_android_bridge.v2.json").is_file()
        ),
        None,
    )
    properties_file = source_file.parents[2] / "runtime.properties"
    properties: dict[str, str] = {}
    if properties_file.is_file():
        for raw_line in properties_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith(("#", "!")) or "=" not in line:
                continue
            key, value = line.split("=", 1)
            properties[key.strip()] = value.strip()
    transfer = {"ready": False, "backend": "unavailable"}
    try:
        from omniflow.transfer.runtime import preflight_omnitransfer

        transfer = preflight_omnitransfer()
    except (ImportError, RuntimeError, ValueError):
        pass
    operations: list[str] = []
    if contract is not None:
        value = json.loads(contract.read_text(encoding="utf-8"))
        if isinstance(value, dict) and isinstance(value.get("operations"), dict):
            operations = list(value["operations"])
    return {
        "capabilities": operations,
        "contract_sha256": (
            hashlib.sha256(contract.read_bytes()).hexdigest() if contract else ""
        ),
        "runtime_version": properties.get("runtime.version", ""),
        "omniflow_commit": properties.get("omniflow.commit", ""),
        "omniflow_source_sha256": properties.get("omniflow.source.sha256", ""),
        "omnitransfer_commit": properties.get("omnitransfer.commit", ""),
        "omnitransfer_source_sha256": properties.get("omnitransfer.source.sha256", ""),
        "omnitransfer_ready": transfer["ready"],
        "omnitransfer_backend": transfer["backend"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store")
    parser.add_argument(
        "--catalog",
        help="Catalog release directory, or 'default' for the packaged release.",
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.self_test:
        return self_test()
    if not arguments.store:
        parser.error("the following arguments are required: --store")
    catalog = None
    if arguments.catalog:
        catalog = (
            load_default_catalog()
            if arguments.catalog == "default"
            else load_catalog(arguments.catalog)
        )
    bridge = JsonLineBridge(arguments.store, catalog=catalog)
    if arguments.once:
        bridge.serve_once()
    else:
        bridge.serve_forever()
    return 0


def self_test() -> int:
    identity = _bridge_identity()
    if not identity.get("runtime_version"):
        raise RuntimeError("runtime_version_missing")
    if not identity.get("contract_sha256"):
        raise RuntimeError("bridge_contract_missing")
    if identity.get("omnitransfer_ready") is not True:
        raise RuntimeError("omnitransfer_not_ready")
    print(json.dumps({"ok": True, "runtime": identity}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
