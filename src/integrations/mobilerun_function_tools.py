"""Expose OmniFlow Functions as Mobilerun custom tools.

Mobilerun owns planning and custom-tool dispatch.  OmniFlow owns Function
binding, state transfer, checker recovery, and physical execution.  This
module is the seam between those two interfaces; it deliberately does not
turn Function steps into Mobilerun's atomic ADB actions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import inspect
import json
import re
from typing import Any

from omniflow.core.model import Function, RunResult
from src.integrations.gui_agent_tools import (
    GuiAgentTool,
    GuiAgentToolResult,
    GuiAgentToolRuntime,
)

MobilerunInvoker = Callable[
    [Function, dict[str, Any], Any],
    Any,
]

_MOBILERUN_NAME = re.compile(r"^[a-zA-Z0-9_-]+$")


def build_custom_tools(
    functions: Iterable[Function],
    *,
    invoker: MobilerunInvoker,
) -> dict[str, dict[str, Any]]:
    """Build the ``custom_tools`` dictionary accepted by Mobilerun.

    The public tool names remain Function IDs.  Function parameter names are
    normalized only at this transport seam when needed; the callback maps
    them back to the canonical Function schema before invoking ``invoker``.
    Function IDs are identity, so an invalid ID is rejected instead of being
    silently renamed.
    """

    if not callable(invoker):
        raise TypeError("mobilerun_function_invoker_required")

    tools: dict[str, dict[str, Any]] = {}
    for function in functions:
        if not isinstance(function, Function):
            raise TypeError("mobilerun_function_must_be_canonical")
        if not function.agent_visible:
            continue
        name = str(function.id or "").strip()
        if not _MOBILERUN_NAME.fullmatch(name):
            raise ValueError(f"mobilerun_function_name_invalid:{name}")
        if name in tools:
            raise ValueError(f"mobilerun_duplicate_function_name:{name}")

        public_parameters, reverse_parameter_names = _custom_parameters(function)

        async def call_tool(
            *,
            ctx: Any = None,
            _function: Function = function,
            _reverse: Mapping[str, str] = reverse_parameter_names,
            **arguments: Any,
        ) -> str:
            canonical_arguments = {
                _reverse.get(key, key): value for key, value in arguments.items()
            }
            result = invoker(_function, canonical_arguments, ctx)
            if inspect.isawaitable(result):
                result = await result
            return _result_text(result)

        tools[name] = {
            "parameters": public_parameters,
            "description": str(function.description).strip(),
            "function": call_tool,
        }
    return tools


def build_omniflow_custom_tools(flow: Any) -> dict[str, dict[str, Any]]:
    """Expose the visible Functions from an initialized ``OmniFlow``.

    ``flow`` must already own a configured Host, Function Store, Transfer
    plugin, and shared Checker library.  Mobilerun's ``ctx`` is intentionally
    not used as a replacement device host: the initialized OmniFlow runtime
    remains responsible for OOB-compatible physical execution.
    """

    store = getattr(flow, "store", None)
    list_functions = getattr(store, "list_functions", None)
    acall_tool = getattr(flow, "acall_tool", None)
    if not callable(list_functions) or not callable(acall_tool):
        raise TypeError("omniflow_runtime_with_store_required")

    async def invoke(function: Function, arguments: dict[str, Any], _ctx: Any) -> Any:
        return await acall_tool(
            {"name": function.id, "arguments": arguments},
            experiment="mobilerun",
        )

    return build_custom_tools(
        list_functions(include_hidden=False),
        invoker=invoke,
    )


def build_runtime_custom_tools(
    runtime: GuiAgentToolRuntime,
) -> dict[str, dict[str, Any]]:
    """Expose the complete OOB-owned tool surface to Mobilerun.

    These custom tools intentionally use the same names as Mobilerun's atomic
    actions.  Mobilerun registers custom tools last, so the OOB-backed tools
    replace its native device-driver actions instead of creating a second
    physical execution path.
    """

    if not isinstance(runtime, GuiAgentToolRuntime):
        raise TypeError("mobilerun_gui_agent_runtime_required")

    tools: dict[str, dict[str, Any]] = {}
    for tool in runtime.list_tools():
        name = str(tool.name or "").strip()
        if not _MOBILERUN_NAME.fullmatch(name):
            raise ValueError(f"mobilerun_tool_name_invalid:{name}")
        if name in tools:
            raise ValueError(f"mobilerun_duplicate_tool_name:{name}")
        public_parameters, reverse_parameter_names = _tool_parameters(tool)

        async def call_tool(
            *,
            ctx: Any = None,
            _name: str = name,
            _reverse: Mapping[str, str] = reverse_parameter_names,
            **arguments: Any,
        ) -> str:
            del ctx
            canonical_arguments = {
                _reverse.get(key, key): value for key, value in arguments.items()
            }
            result = await runtime.call_tool(_name, canonical_arguments)
            return _gui_agent_result_text(result)

        tools[name] = {
            "parameters": public_parameters,
            "description": tool.description,
            "function": call_tool,
        }
    return tools


def _custom_parameters(
    function: Function,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    schema = function.input_schema
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(properties, dict):
        raise TypeError(f"mobilerun_function_schema_invalid:{function.id}")
    required = {str(value) for value in schema.get("required") or ()}
    parameters: dict[str, dict[str, Any]] = {}
    reverse: dict[str, str] = {}
    used_names: set[str] = set()
    for canonical_name, raw_definition in properties.items():
        canonical_name = str(canonical_name)
        if not isinstance(raw_definition, dict):
            raise TypeError(
                f"mobilerun_function_parameter_schema_invalid:{function.id}:{canonical_name}"
            )
        public_name = _public_parameter_name(canonical_name, used_names)
        used_names.add(public_name)
        definition: dict[str, Any] = {
            "type": str(raw_definition.get("type") or "string"),
            "required": canonical_name in required,
        }
        description = str(raw_definition.get("description") or "").strip()
        if description:
            definition["description"] = description
        if "default" in raw_definition:
            definition["default"] = raw_definition["default"]
        parameters[public_name] = definition
        reverse[public_name] = canonical_name
    return parameters, reverse


def _public_parameter_name(name: str, used_names: set[str]) -> str:
    candidate = re.sub(r"[^a-zA-Z0-9_-]", "_", name).strip("_") or "parameter"
    if _MOBILERUN_NAME.fullmatch(candidate) is None:
        candidate = "parameter"
    base = candidate
    suffix = 2
    while candidate in used_names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _tool_parameters(
    tool: GuiAgentTool,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    schema = tool.input_schema
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(properties, dict):
        raise TypeError(f"mobilerun_tool_schema_invalid:{tool.name}")
    required = {str(value) for value in schema.get("required") or ()}
    parameters: dict[str, dict[str, Any]] = {}
    reverse: dict[str, str] = {}
    used_names: set[str] = set()
    for canonical_name, raw_definition in properties.items():
        canonical_name = str(canonical_name)
        if not isinstance(raw_definition, dict):
            raise TypeError(
                f"mobilerun_tool_parameter_schema_invalid:{tool.name}:{canonical_name}"
            )
        public_name = _public_parameter_name(canonical_name, used_names)
        used_names.add(public_name)
        definition: dict[str, Any] = {
            "type": str(raw_definition.get("type") or "string"),
            "required": canonical_name in required,
        }
        for field in ("description", "default", "enum", "minimum", "maximum", "items"):
            if field in raw_definition:
                definition[field] = raw_definition[field]
        parameters[public_name] = definition
        reverse[public_name] = canonical_name
    return parameters, reverse


def _result_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, RunResult):
        payload = {
            "success": result.success,
            "function_id": result.function_id,
            "actions_executed": result.actions_executed,
            "error": result.error,
        }
        prefix = "Completed" if result.success else "Failed"
        return f"{prefix}: " + json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    if result is None:
        return "Done"
    if isinstance(result, (dict, list, tuple, bool, int, float)):
        return json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    return str(result)


def _gui_agent_result_text(result: GuiAgentToolResult) -> str:
    prefix = "Completed" if result.success else "Failed"
    return f"{prefix}: " + json.dumps(
        result.to_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
    )


__all__ = [
    "MobilerunInvoker",
    "build_custom_tools",
    "build_omniflow_custom_tools",
    "build_runtime_custom_tools",
]
