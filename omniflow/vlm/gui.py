from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from omniflow.core.model import Function, ToolCall
from omniflow.core.schemas import canonicalize_action, vlm_action_tools
from omniflow.functions.artifact import validate_arguments
from omniflow.vlm.model_adapter import adapt_tool_arguments
from omniflow.vlm.tool_arguments import load_tool_arguments
from omniflow.vlm.ui_projection import (
    UIProjection,
    project_ui,
    projected_node_center,
)
from omniflow.vlm_coordinates import (
    display_size,
    screen_context_to_pixels,
    screen_pixel_args_to_canonical,
    screen_pixel_tools,
)

SYSTEM_PROMPT = """
You are an Android GUI agent. Complete the user goal from the compact relevant UI
elements and current screenshot. Treat the screenshot as primary evidence for icon
identity and spatial relationships, and XML as evidence for text and control state.
UI elements are grouped by priority; global controls come first, and goal_controls
are actionable visual elements adaptively associated with nearby goal text. The `v`
field is a stable visual reference for an actionable element at its XML bounds.
When you choose a projected native XML node, whether by `v` or its exact label,
default x and y to the center of that node's `b=[left,top][right,bottom]` bounds.
For example, a chosen node with bounds
[0,766][720,878] uses (360,822), not (360,640). This node-center rule does not apply
to WebView or screenshot-only visual targets without a reliable projected node; locate
those from the screenshot instead.
Return exactly one native tool_call each turn. Never put
action JSON or tool syntax in assistant text. Choose one action, wait for its
result, then inspect the fresh state before choosing another action. Coordinates
are raw pixels in the current original Display coordinate frame, never normalized
0..1000 values. XML bounds use that same raw-pixel frame. A screenshot may be
resized for transport, but its coordinates must still refer to the original
Display. If you include a summary, use a summary of at most 12 words naming only
the immediate action. The summary is optional metadata; never reject a valid tool
call because it is absent.
Do not emit analysis, chain-of-thought, reasoning, thinking, rationale, or prose.
Make the decision directly from current evidence and return only the tool call.
Never call a recalled Function merely because it matches the goal. A global
Function whose first action is `open_app` is an exception: call it directly from
the launcher or an unrelated starting page because it owns the startup and
navigation prefix; do not call `open_app` separately first. For every other
Function, call it only when the current UI already shows its described page or
control; finish onboarding and navigation, and reopen the requested content,
before calling it.
Every coordinate is one scalar raw-pixel number, never an array, object, string,
boolean, normalized value, or combined coordinate pair.
Use finished only when current evidence directly proves the goal is complete.
When calling finished, keep content to one short factual sentence describing only
the outcome directly supported by the current screen or previous tool result. Do
not claim that a RunLog or reusable Function was registered; the host reports the
real registration state after execution.
For switches and checkboxes, checked=false means off and checked=true means on.
Never toggle a switch when its checked state already matches the requested goal.
If a click leaves the state unchanged, do not repeat the same coordinates; ground the
next action in the exact projected bounds or choose a different visible control.
If the previous action succeeded and the state changed, reassess the fresh page before
repeating the same semantic target; repeat it only when the current evidence shows it
is still the required next action, never as timeout or recovery behavior.
Prefer stable, reusable navigation. When the current app or page provides search,
use search and type the requested text directly before browsing long menus or
swiping. Do not select history, recent, suggestion, or cached-value items when the
requested value can be entered directly. Swipe only when no usable search or input
path exists, or when search results still require browsing.
""".strip()


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
    previous_screenshot_path: str = "",
    validation_error: str = "",
    retry_tool_name: str = "",
    rejected_tool_call: dict[str, Any] | None = None,
    lightweight_retry: bool = False,
) -> dict[str, Any]:
    projection = (
        UIProjection("<omitted>", 0, 0, 0)
        if lightweight_retry
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
        lightweight_retry=lightweight_retry,
        projection=projection,
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    include_images = not lightweight_retry
    current_image = _state_image_data_uri(state) if include_images else ""
    if current_image:
        content.append({"type": "image_url", "image_url": {"url": current_image}})
    display = state.get("display") if isinstance(state.get("display"), dict) else None
    global_functions = tuple(
        function
        for function in functions
        if function.agent_visible
        and function.steps
        and function.steps[0].action.tool == "open_app"
    )
    if global_functions:
        # A recalled global Function owns startup. Keep this as a normal tool
        # call, but remove every competing native action and lower-priority
        # Function from this turn. If execution fails, the runtime excludes the
        # failed Function on the next turn and the complete native tool set is
        # visible again for VLM fallback.
        tools = []
        visible_functions = global_functions
    else:
        tools = screen_pixel_tools(
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
    return {
        "model": str(model),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "max_tokens": 512,
        "temperature": 0,
        "stream": True,
        "tools": tools,
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "reasoning_effort": "none",
        "enable_thinking": False,
        "thinking": {"type": "disabled"},
    }


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
                        "original_raw_pixels": original,
                        "grounded_raw_pixels": {"x": x, "y": y},
                    }
            arguments, coordinate_metadata = screen_pixel_args_to_canonical(
                tool=tool,
                args=arguments,
                display=display,
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
                        "Why this single tool is the best next step, in at most "
                        "20 Chinese characters or one short sentence."
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
    center_x = int(width / 2)
    center_y = int(height / 2)
    upper_y = int(height * 0.7)
    lower_y = int(height * 0.3)
    lines = [
        f"Goal: {goal}",
        f"Progress: {turn_index}/{max_steps} model turns used",
        f"Current package: {state.get('package_name') or ''}",
        f"Current activity: {state.get('activity_name') or ''}",
        f"Display: {display.get('width') or ''}x{display.get('height') or ''}",
        (
            "Coordinate contract: every tool coordinate is one raw pixel in the "
            f"current original Display frame (X 0..{int(width)}, Y 0..{int(height)}). "
            "Do not output normalized 0..1000 coordinates."
        ),
        (
            'Raw-pixel examples: click {"summary":"Tap center","x":'
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
                    "Coordinate fields such as x and y must each be one raw-pixel "
                    f"JSON number in the current Display (X 0..{int(width)}, "
                    f"Y 0..{int(height)}). Never use normalized 0..1000 values, "
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
                        'Valid raw-pixel scalar shape: {"x":'
                        f'{center_x},"y":{center_y},"x1":{center_x},'
                        f'"y1":{upper_y},"x2":{center_x},"y2":{lower_y}'
                        '}. Invalid array shape: {"x":[500],"y":[464],"x1":[500,800]}.'
                    ),
                    "If your rejected call placed one intended point in x as [X,Y], choose the scalars yourself and emit x:X and y:Y in the new call. The runtime will not transform the array for you.",
                )
            )
    raw_extra = state.get("extra")
    extra = (
        screen_context_to_pixels(raw_extra, display)
        if isinstance(raw_extra, dict)
        else raw_extra
    )
    if not lightweight_retry and isinstance(extra, dict) and extra:
        context = dict(extra)
        context.pop("installed_apps", None)
        execution_history = str(context.pop("execution_history", "") or "").strip()
        if has_successful_function_action(context):
            lines.append(
                "The previous recalled Function tool call finished all of its "
                "actions successfully. Those actions are already applied. Judge "
                "the complete user goal from the current screenshot and UI state. "
                "Choose finished only if the whole goal is satisfied; otherwise "
                "choose exactly one next tool. Never repeat or toggle the last "
                "successful action merely to verify it, because that can undo the "
                "completed operation."
            )
        if context.get("previous_action_error") or context.get("recent_actions"):
            lines.append(
                "Use the recent action history and error before selecting again. "
                "Do not repeat the same action when it already succeeded or made no "
                "progress; choose a different schema-valid action, finish, or abort."
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


def _state_image_data_uri(state: dict[str, Any]) -> str:
    image = str(state.get("image_base64") or "").strip()
    if image:
        return (
            image
            if image.startswith("data:image/")
            else f"data:image/jpeg;base64,{image}"
        )
    return _image_data_uri(str(state.get("screenshot_path") or ""))


def _image_data_uri(path: str) -> str:
    candidate = Path(str(path or "").strip())
    if not candidate.is_file():
        return ""
    try:
        payload = candidate.read_bytes()
    except OSError:
        return ""
    mime_type = _image_mime_type(payload)
    if not mime_type:
        return ""
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime_type};base64,{encoded}" if encoded else ""


def _image_mime_type(payload: bytes) -> str:
    if payload.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(payload) >= 12 and payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        return "image/webp"
    return ""


__all__ = [
    "ModelToolCallError",
    "SYSTEM_PROMPT",
    "build_model_turn_request",
    "has_successful_function_action",
    "parse_model_turn_response",
]
