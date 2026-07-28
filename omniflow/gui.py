from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from omniflow.artifact import validate_arguments
from omniflow.model import Function, ToolCall
from omniflow.model_adapter import adapt_tool_arguments
from omniflow.schemas import canonicalize_action, vlm_action_tools
from omniflow.tool_argument_repair import load_tool_arguments
from omniflow.ui_projection import UIProjection, project_ui
from omniflow.vlm_coordinates import (
    display_size,
    screen_context_to_pixels,
    screen_pixel_args_to_canonical,
    screen_pixel_tools,
)

SYSTEM_PROMPT = """
You are an Android GUI agent. Complete the user goal from the compact relevant UI
elements and optional current screenshot. Return exactly one native tool_call each turn. Never put
action JSON or tool syntax in assistant text. Choose one action, wait for its
result, then inspect the fresh state before choosing another action. Coordinates
are raw pixels in the current original Display coordinate frame, never normalized
0..1000 values. XML bounds use that same raw-pixel frame. A screenshot may be
resized for transport, but its coordinates must still refer to the original
Display. Every tool call
must include a concise summary explaining why that action is the best next step.
Every coordinate is one scalar raw-pixel number, never an array, object, string,
boolean, normalized value, or combined coordinate pair.
Use finished only when current evidence directly proves the goal is complete.
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
        installed_apps=installed_apps or {},
        validation_error=validation_error,
        rejected_tool_call=rejected_tool_call,
        lightweight_retry=lightweight_retry,
        projection=projection,
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    include_images = not validation_error.strip() and projection.requires_screenshot
    current_image = _state_image_data_uri(state) if include_images else ""
    if current_image:
        content.append({"type": "image_url", "image_url": {"url": current_image}})
    display = state.get("display") if isinstance(state.get("display"), dict) else None
    tools = screen_pixel_tools(
        vlm_action_tools(include_summary=True),
        display,
    )
    tools = constrain_open_app_tool(tools, installed_apps or {})
    tools.extend(function_tools(functions, include_summary=True))
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
        "max_completion_tokens": 4096,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": tools,
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "enable_thinking": False,
    }


def parse_model_turn_response(
    value: Any,
    *,
    requested_model: str,
    turn_index: int,
    display: dict[str, Any] | None = None,
    functions: list[Function] | tuple[Function, ...] = (),
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
    function_catalog = {
        function.id: function for function in functions if function.agent_visible
    }
    model_visible_tools.update(function_catalog)
    if tool not in model_visible_tools:
        raise ModelToolCallError(f"model_turn_tool_not_visible:{tool}")
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
    if not summary:
        raise ModelToolCallError(
            "model_turn_summary_required",
            tool_name=tool,
            arguments=rejected_arguments,
        )
    resolved_model = str(value.get("resolved_model") or requested_model).strip()
    adapter_metadata = None
    coordinate_metadata = None
    try:
        if tool in function_catalog:
            validate_arguments(function_catalog[tool].input_schema, arguments)
        else:
            arguments, adapter_metadata = adapt_tool_arguments(
                tool=tool,
                arguments=arguments,
                requested_model=requested_model,
                resolved_model=resolved_model,
                display=display,
            )
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
    for function in sorted(functions, key=lambda item: item.id):
        if not function.agent_visible:
            continue
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
            required = ["summary", *required]
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


def _turn_text(
    *,
    goal: str,
    state: dict[str, Any],
    max_steps: int,
    turn_index: int,
    target_package_name: str,
    step_skill_guidance: str,
    installed_apps: dict[str, str],
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
    app_candidates = _installed_app_candidates(installed_apps)
    if app_candidates and not lightweight_retry:
        lines.append(
            "Installed app candidates (open_app.package_name must be one exact "
            "package from this list; never guess a package):"
        )
        lines.append(
            json.dumps(
                [
                    {"label": label, "package_name": package}
                    for label, package in app_candidates
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
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
        execution_history = str(context.pop("execution_history", "") or "").strip()
        recent_actions = context.get("recent_actions")
        if isinstance(recent_actions, list) and any(
            isinstance(item, dict)
            and item.get("success") is True
            and str(item.get("function_id") or "").strip()
            for item in recent_actions
        ):
            lines.append(
                "A recalled Function completed successfully in the recent action "
                "history. If the goal asks to run, use, or execute that saved "
                "workflow once, the requested operation is already complete: "
                "choose finished now. Do not add extra GUI actions merely to verify it."
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


def constrain_open_app_tool(
    tools: list[dict[str, Any]],
    installed_apps: dict[str, str],
) -> list[dict[str, Any]]:
    packages = list(
        dict.fromkeys(
            package for _label, package in _installed_app_candidates(installed_apps)
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
    "parse_model_turn_response",
]
