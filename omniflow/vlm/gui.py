from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from omniflow.core.config import DEFAULT_PLANNER_SYSTEM_PROMPT
from omniflow.core.model import Function, ToolCall
from omniflow.core.schemas import canonicalize_action, vlm_action_tools
from omniflow.functions.artifact import validate_arguments
from omniflow.vlm.model_adapter import adapt_tool_arguments
from omniflow.vlm.tool_arguments import load_tool_arguments
from omniflow.vlm.ui_projection import (
    UIProjection,
    project_ui,
    projected_numeric_summary_center,
    projected_node_center,
)
from omniflow.vlm_coordinates import (
    display_size,
    relative_args_to_canonical,
    relative_coordinate_tools,
    screen_pixel_args_to_canonical,
)

SYSTEM_PROMPT = DEFAULT_PLANNER_SYSTEM_PROMPT


class ModelToolCallError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        tool_name: str = "",
        arguments: Any = None,
    ):
        self.code = str(message)
        self.tool_name = str(tool_name).strip()
        self.arguments = arguments
        super().__init__(message)


def build_model_turn_request(
    *,
    goal: str,
    model: str,
    state: dict[str, Any],
    max_steps: int,
    turn_index: int,
    target_package_name: str = "",
    step_skill_guidance: str = "",
    installed_apps: dict[str, str] | None = None,
    functions: list[Function] | tuple[Function, ...] = (),
    validation_error: str = "",
    retry_tool_name: str = "",
    rejected_tool_call: dict[str, Any] | None = None,
    lightweight_retry: bool = False,
) -> dict[str, Any]:
    text_only_model = str(model).strip().casefold() == "glm-5.1"
    global_functions = tuple(
        function
        for function in functions
        if function.agent_visible
        and function.steps
        and function.steps[0].action.tool == "open_app"
    )
    compact_global_startup = bool(global_functions) and not lightweight_retry
    projection = (
        UIProjection("<omitted>", 0, 0, 0)
        if lightweight_retry or compact_global_startup
        else project_ui(str(state.get("xml") or ""), goal)
    )
    text = _turn_text(
        goal=goal,
        state=state,
        max_steps=max_steps,
        turn_index=turn_index,
        target_package_name=target_package_name,
        step_skill_guidance=step_skill_guidance,
        validation_error=validation_error,
        rejected_tool_call=rejected_tool_call,
        lightweight_retry=lightweight_retry or compact_global_startup,
        projection=projection,
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    display = state.get("display") if isinstance(state.get("display"), dict) else None
    if global_functions:
        # A recalled global Function owns startup. Keep this as a normal tool
        # call, but remove every competing native action and lower-priority
        # Function from this turn. If execution fails, the runtime excludes the
        # failed Function on the next turn and the complete native tool set is
        # visible again for VLM fallback.
        tools = []
        visible_functions = global_functions
    else:
        tools = relative_coordinate_tools(
            vlm_action_tools(include_summary=True),
            display,
        )
        tools = constrain_open_app_tool(tools, installed_apps or {})
        visible_functions = functions
    tools.extend(function_tools(visible_functions, include_summary=True))
    if retry_tool_name:
        tools = [
            tool
            for tool in tools
            if tool.get("function", {}).get("name") == retry_tool_name
        ]
        if len(tools) != 1:
            raise ValueError(f"model_turn_retry_tool_not_visible:{retry_tool_name}")
    tool_choice: str | dict[str, Any] = "required"
    if len(global_functions) == 1:
        tool_choice = {
            "type": "function",
            "function": {"name": global_functions[0].id},
        }
    request = {
        "model": str(model),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "max_tokens": 512,
        "temperature": 0,
        "stream": False,
        "tools": tools,
        "tool_choice": tool_choice,
        "parallel_tool_calls": False,
        "enable_thinking": False,
        "thinking": {"type": "disabled"},
    }
    if not text_only_model:
        request["reasoning_effort"] = "none"
    return request


def parse_model_turn_response(
    value: Any,
    *,
    requested_model: str,
    turn_index: int,
    display: dict[str, Any] | None = None,
    functions: list[Function] | tuple[Function, ...] = (),
    installed_apps: dict[str, str] | None = None,
    state: dict[str, Any] | None = None,
    goal: str = "",
) -> tuple[ToolCall, dict[str, Any]]:
    if not isinstance(value, dict):
        raise ValueError("model_turn_response_invalid")
    if str(value.get("requested_model") or "").strip() != requested_model:
        raise ValueError("model_turn_requested_model_mismatch")
    tool_calls = value.get("tool_calls")
    if not isinstance(tool_calls, list):
        raise ModelToolCallError("model_turn_tool_calls_invalid")
    if len(tool_calls) != 1:
        raise ModelToolCallError(
            f"provider_tool_call_contract_violation:expected_one_native_tool_call:got_{len(tool_calls)}"
        )
    tool_call = tool_calls[0]
    if not isinstance(tool_call, dict):
        raise ModelToolCallError("model_turn_tool_call_invalid")
    function = tool_call.get("function")
    if not isinstance(function, dict):
        raise ModelToolCallError("model_turn_tool_call_function_invalid")
    tool = str(function.get("name") or "").strip()
    model_visible_tools = {
        str(item.get("function", {}).get("name") or "")
        for item in vlm_action_tools()
        if isinstance(item, dict) and isinstance(item.get("function"), dict)
    }
    function_catalog = {function.id: function for function in functions}
    if not tool and len(function_catalog) == 1:
        only_function = next(iter(function_catalog.values()))
        if (
            only_function.agent_visible
            and only_function.steps
            and only_function.steps[0].action.tool == "open_app"
        ):
            # Some OpenAI-compatible streaming gateways preserve the sole tool
            # call but omit its function name.  With exactly one visible global
            # Function, the identity is unambiguous; do not infer it when the
            # tool set contains multiple choices.
            tool = only_function.id
    model_visible_tools.update(function_catalog)
    if tool not in model_visible_tools:
        raise ModelToolCallError(
            f"model_turn_tool_not_visible:{tool}",
            tool_name=tool,
            arguments=function.get("arguments"),
        )
    raw_arguments = function.get("arguments")
    if not isinstance(raw_arguments, str):
        raise ModelToolCallError(
            "model_turn_tool_arguments_invalid",
            tool_name=tool,
            arguments=raw_arguments,
        )
    try:
        arguments, arguments_repaired = load_tool_arguments(raw_arguments)
    except json.JSONDecodeError as error:
        raise ModelToolCallError(
            "model_turn_tool_arguments_must_be_json",
            tool_name=tool,
            arguments=raw_arguments,
        ) from error
    if not isinstance(arguments, dict):
        raise ModelToolCallError(
            "model_turn_tool_arguments_must_be_object",
            tool_name=tool,
            arguments=arguments,
        )
    rejected_arguments = dict(arguments)
    summary = str(arguments.pop("summary", "") or "").strip()
    resolved_model = str(value.get("resolved_model") or requested_model).strip()
    adapter_metadata = None
    coordinate_metadata = None
    node_grounding_metadata = None
    try:
        if tool in function_catalog:
            validate_arguments(function_catalog[tool].input_schema, arguments)
        else:
            package_name = arguments.get("package_name")
            installed_packages = {
                str(package).strip()
                for package in (installed_apps or {}).values()
                if str(package).strip()
            }
            if (
                tool == "open_app"
                and isinstance(package_name, str)
                and package_name.strip() not in installed_packages
            ):
                allowed_packages = ",".join(sorted(installed_packages))
                raise ValueError(
                    "planner_open_app_package_not_installed:"
                    f"{package_name.strip()}:"
                    f"allowed_package_name={allowed_packages}"
                )
            arguments, adapter_metadata = adapt_tool_arguments(
                tool=tool,
                arguments=arguments,
                requested_model=requested_model,
                resolved_model=resolved_model,
                display=display,
            )
            if tool in {"click", "input_text"}:
                projection = project_ui(
                    str((state or {}).get("xml") or ""),
                    goal,
                )
                grounded = projected_node_center(
                    projection,
                    str(arguments.get("target_description") or ""),
                )
                grounding_source = "target_description"
                if grounded is None and not arguments.get("target_description"):
                    numeric_grounding = projected_numeric_summary_center(
                        projection,
                        summary,
                    )
                    if numeric_grounding is not None:
                        node, center, numeric_label = numeric_grounding
                        grounded = node, center
                        arguments = {
                            **arguments,
                            "target_description": numeric_label,
                        }
                        grounding_source = "numeric_summary"
                if grounded is not None:
                    node, (x, y) = grounded
                    original = {
                        "x": arguments.get("x"),
                        "y": arguments.get("y"),
                    }
                    arguments = {**arguments, "x": x, "y": y}
                    node_grounding_metadata = {
                        "name": "projected_node_center.v1",
                        "reference": node.reference,
                        "target_description": str(
                            arguments.get("target_description") or ""
                        ),
                        "bounds": list(node.bounds),
                        "original_relative_0_1000": original,
                        "grounded_raw_pixels": {"x": x, "y": y},
                    }
                    if grounding_source == "numeric_summary":
                        node_grounding_metadata["source"] = grounding_source
            if node_grounding_metadata is not None:
                arguments, coordinate_metadata = screen_pixel_args_to_canonical(
                    tool=tool,
                    args=arguments,
                    display=display,
                )
                node_grounding_metadata["grounded_relative_0_1000"] = {
                    "x": arguments.get("x"),
                    "y": arguments.get("y"),
                }
            else:
                arguments, coordinate_metadata = relative_args_to_canonical(
                    tool=tool,
                    args=arguments,
                )
            canonical = canonicalize_action(
                {"tool": tool, "args": arguments},
                persisted_only=False,
                allow_non_action=True,
            )
            arguments = dict(canonical["args"])
    except ValueError as error:
        raise ModelToolCallError(
            str(error),
            tool_name=tool,
            arguments=rejected_arguments,
        ) from error
    metadata: dict[str, Any] = {"summary": summary}
    if arguments_repaired:
        metadata["json_repair"] = {
            "name": "json_repair",
            "applied": True,
        }
    if adapter_metadata is not None:
        metadata["model_adapter"] = adapter_metadata
    if coordinate_metadata is not None:
        metadata["coordinate_conversion"] = coordinate_metadata
    if node_grounding_metadata is not None:
        metadata["node_grounding"] = node_grounding_metadata
    thinking = str(value.get("reasoning") or "").strip()
    if thinking:
        metadata["thinking"] = thinking
    usage = value.get("usage")
    if isinstance(usage, dict):
        metadata["token_usage"] = {
            **usage,
            "model": requested_model,
            "resolved_model": resolved_model,
            "turn_index": int(turn_index),
        }
    return ToolCall(tool, arguments), metadata


def function_tools(
    functions: list[Function] | tuple[Function, ...],
    *,
    include_summary: bool,
) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for function in functions:
        parameters = deepcopy(function.input_schema)
        properties = parameters.setdefault("properties", {})
        required = list(parameters.get("required") or ())
        if include_summary:
            properties = {
                "summary": {
                    "type": "string",
                    "description": (
                        "Immediate subgoal and expected progress of this Function, "
                        "in at most 20 Chinese characters or one short sentence. "
                        "This becomes short step memory on the next turn."
                    ),
                },
                **properties,
            }
            parameters["properties"] = properties
        parameters["required"] = required
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": function.id,
                    "description": function.description,
                    "strict": True,
                    "parameters": parameters,
                },
            }
        )
    return tools


def _has_global_startup_function(
    functions: list[Function] | tuple[Function, ...],
) -> bool:
    return any(
        function.agent_visible
        and function.steps
        and function.steps[0].action.tool == "open_app"
        for function in functions
    )


def _turn_text(
    *,
    goal: str,
    state: dict[str, Any],
    max_steps: int,
    turn_index: int,
    target_package_name: str,
    step_skill_guidance: str,
    validation_error: str,
    rejected_tool_call: dict[str, Any] | None,
    lightweight_retry: bool,
    projection: UIProjection,
) -> str:
    display = state.get("display") if isinstance(state.get("display"), dict) else {}
    width, height = display_size(display)
    center_x = 500
    center_y = 500
    upper_y = 700
    lower_y = 300
    lines = [
        f"Goal: {goal}",
        f"Progress: {turn_index}/{max_steps} model turns used",
        f"Current package: {state.get('package_name') or ''}",
        f"Current activity: {state.get('activity_name') or ''}",
        f"Display: {display.get('width') or ''}x{display.get('height') or ''}",
        (
            "Coordinate contract: every tool coordinate is one device-independent "
            "relative value from 0..1000 on each axis. XML b bounds are raw pixels "
            f"in the {int(width)}x{int(height)} Display; convert them before calling a tool."
        ),
        (
            'Relative-coordinate examples: click {"summary":"Tap center","x":'
            f'{center_x},"y":{center_y}'
            '}; swipe {"summary":"Scroll up","direction":"up","x1":'
            f'{center_x},"y1":{upper_y},"x2":{center_x},"y2":{lower_y}'
            "}."
        ),
    ]
    if target_package_name:
        lines.append(f"Target package: {target_package_name}")
    if step_skill_guidance.strip() and not lightweight_retry:
        lines.extend(("Task guidance:", step_skill_guidance.strip()))
    if validation_error.strip():
        lines.extend(
            (
                "Your previous native tool_call was rejected by the registered schema:",
                validation_error.strip(),
                "Return one corrected native tool_call using the schema exactly. Do not rename, wrap, combine, or infer fields.",
                (
                    "Coordinate fields such as x and y must each be one relative "
                    "JSON number from 0..1000 on its axis. Never use "
                    "[x, y], an object, string, or boolean."
                ),
            )
        )
        if validation_error.startswith("planner_open_app_package_not_installed:"):
            lines.append(
                "package_name is an opaque identifier: copy one complete "
                "allowed_package_name value byte-for-byte from the registered "
                "open_app enum. Never shorten it, remove vendor segments, or "
                "invent an Android package name."
            )
        if rejected_tool_call:
            lines.extend(
                (
                    "Rejected native tool_call from your immediately previous attempt (verbatim):",
                    json.dumps(
                        rejected_tool_call,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "Do not repeat that argument shape. Return a new tool_call; do not explain or repair it in text.",
                    (
                        'Valid relative-coordinate scalar shape: {"x":'
                        f'{center_x},"y":{center_y},"x1":{center_x},'
                        f'"y1":{upper_y},"x2":{center_x},"y2":{lower_y}'
                        '}. Invalid array shape: {"x":[500],"y":[464],"x1":[500,800]}.'
                    ),
                    "If your rejected call placed one intended point in x as [X,Y], choose the scalars yourself and emit x:X and y:Y in the new call. The runtime will not transform the array for you.",
                )
            )
    raw_extra = state.get("extra")
    extra = deepcopy(raw_extra) if isinstance(raw_extra, dict) else raw_extra
    if not lightweight_retry and isinstance(extra, dict) and extra:
        context = dict(extra)
        context.pop("installed_apps", None)
        execution_history = str(context.pop("execution_history", "") or "").strip()
        if has_successful_function_action(context):
            lines.append(
                "The previous recalled Function tool call finished all of its "
                "actions successfully. Those actions are already applied. Judge "
                "the complete user goal from the current accessibility state. "
                "Choose finished only if the whole goal is satisfied; otherwise "
                "choose exactly one next tool. Never repeat or toggle the last "
                "successful action merely to verify it, because that can undo the "
                "completed operation."
            )
        if context.get("previous_action_error") or context.get("recent_actions"):
            lines.append(
                "Inspect the action history, observed results, and any previous "
                "error before selecting again. The latest accessibility state is "
                "authoritative. Do not repeat the same action or no-progress "
                "sequence; choose a different visible control or path, finish, "
                "or abort."
            )
        if execution_history:
            lines.extend(("Completed tool-call history:", execution_history))
        if context:
            lines.extend(
                (
                    "Recent execution context:",
                    json.dumps(context, ensure_ascii=False, separators=(",", ":")),
                )
            )
    if not lightweight_retry:
        lines.extend(
            (
                f"Relevant UI elements (1-{projection.selected_count}); {projection.candidate_count} candidates:",
                projection.text,
            )
        )
    return "\n".join(lines)


def has_successful_function_action(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    recent_actions = value.get("recent_actions")
    return isinstance(recent_actions, list) and any(
        isinstance(item, dict)
        and item.get("success") is True
        and bool(str(item.get("function_id") or "").strip())
        for item in recent_actions
    )


def constrain_open_app_tool(
    tools: list[dict[str, Any]],
    installed_apps: dict[str, str],
) -> list[dict[str, Any]]:
    candidates = _installed_app_candidates(installed_apps)
    packages = list(
        dict.fromkeys(
            package for _label, package in candidates
        )
    )
    constrained: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict) or function.get("name") != "open_app":
            constrained.append(tool)
            continue
        if not packages:
            continue
        parameters = function.get("parameters")
        properties = (
            parameters.get("properties") if isinstance(parameters, dict) else None
        )
        package_schema = (
            properties.get("package_name") if isinstance(properties, dict) else None
        )
        if not isinstance(package_schema, dict):
            raise ValueError("open_app_package_schema_missing")
        package_schema["enum"] = packages
        label_mapping = ", ".join(
            f"{label}={package}" for label, package in candidates
        )
        package_schema["description"] = (
            "Exact installed launchable package. Runtime app mapping: "
            f"{label_mapping}"
        )
        constrained.append(tool)
    return constrained


def _installed_app_candidates(
    installed_apps: dict[str, str],
) -> list[tuple[str, str]]:
    candidates = {
        (str(label).strip(), str(package).strip())
        for label, package in installed_apps.items()
        if str(label).strip() and str(package).strip()
    }
    return sorted(candidates, key=lambda item: (item[0].casefold(), item[1]))


__all__ = [
    "ModelToolCallError",
    "SYSTEM_PROMPT",
    "build_model_turn_request",
    "has_successful_function_action",
    "parse_model_turn_response",
]
