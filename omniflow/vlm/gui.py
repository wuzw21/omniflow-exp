from __future__ import annotations

import base64
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
from typing import Any

from PIL import Image

from omniflow.core.config import DEFAULT_PLANNER_SYSTEM_PROMPT
from omniflow.core.model import Function, ToolCall
from omniflow.core.schemas import canonicalize_action, vlm_action_tools
from omniflow.functions.artifact import validate_arguments
from omniflow.vlm.tool_arguments import load_tool_arguments
from omniflow.vlm.ui_projection import (
    UIProjection,
    project_ui,
)
from omniflow.vlm_coordinates import (
    display_size,
    relative_args_to_canonical,
    relative_coordinate_tools,
)

SYSTEM_PROMPT = DEFAULT_PLANNER_SYSTEM_PROMPT

_PLANNER_CONTEXT_KEYS = (
    "previous_action_error",
    "previous_action",
    "recent_actions",
    "execution_history",
    "function_execution",
    "user_input",
    "transfer_candidates_hint",
)


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
    include_image = not lightweight_retry and _state_has_screenshot(state)
    current_image = _state_image_data_uri(state) if include_image else ""
    if current_image:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": _compact_image_data_uri(current_image),
                    "detail": "low",
                },
            }
        )
    display = state.get("display") if isinstance(state.get("display"), dict) else None
    tools = relative_coordinate_tools(
        vlm_action_tools(include_summary=True),
        display,
    )
    tools.extend(function_tools(functions, include_summary=True))
    if retry_tool_name:
        tools = [
            tool
            for tool in tools
            if tool.get("function", {}).get("name") == retry_tool_name
        ]
        if len(tools) != 1:
            raise ValueError(f"model_turn_retry_tool_not_visible:{retry_tool_name}")
    request: dict[str, Any] = {
        "model": str(model),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "max_tokens": 512,
        "temperature": 0,
        "stream": False,
        "tools": tools,
        "tool_choice": "required",
        "parallel_tool_calls": False,
    }
    request["enable_thinking"] = False
    request["thinking"] = {"type": "disabled"}
    request["reasoning_effort"] = "none"
    return request


def _state_has_screenshot(state: dict[str, Any]) -> bool:
    return bool(
        str(state.get("image_base64") or "").strip()
        or str(state.get("screenshot_path") or "").strip()
    )


def _state_image_data_uri(state: dict[str, Any]) -> str:
    image = str(state.get("image_base64") or "").strip()
    if image:
        return (
            image
            if image.startswith("data:image/")
            else f"data:image/jpeg;base64,{image}"
        )
    return _image_data_uri(str(state.get("screenshot_path") or ""))


def _compact_image_data_uri(value: str) -> str:
    prefix, separator, encoded = str(value or "").partition(",")
    if not separator or "base64" not in prefix.casefold():
        return value
    try:
        image = Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")
        thumbnail_box = (640, 360) if image.width > image.height else (360, 640)
        image.thumbnail(thumbnail_box, Image.Resampling.LANCZOS)
        output = BytesIO()
        image.save(output, format="JPEG", quality=60, optimize=True)
        compact = base64.b64encode(output.getvalue()).decode("ascii")
    except Exception:
        return value
    return f"data:image/jpeg;base64,{compact}"


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
    try:
        if tool in function_catalog:
            validate_arguments(function_catalog[tool].input_schema, arguments)
        else:
            package_name = arguments.get("package_name")
            if tool == "open_app" and isinstance(package_name, str):
                from src.integrations.android_world.apps import (
                    canonicalize_androidworld_package,
                )

                canonical_package = canonicalize_androidworld_package(package_name)
                if canonical_package != package_name.strip():
                    arguments["package_name"] = canonical_package
                    package_name = canonical_package
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
            # The ACP/VLM boundary uses one canonical action schema for every
            # provider.  Do not reinterpret malformed coordinate shapes based on
            # a provider/model name; invalid arguments must be rejected and sent
            # back to the planner for correction.
            adapter_metadata = None
            if tool in {"click", "input_text"}:
                projection = project_ui(
                    str((state or {}).get("xml") or ""),
                    goal,
                )
                adapter_metadata = {
                    **dict(adapter_metadata or {}),
                    "ui_projection": {
                        "candidate_count": projection.candidate_count,
                        "selected_count": projection.selected_count,
                        "visual_context_required": projection.visual_context_required,
                        "visual_candidate_count": projection.visual_candidate_count,
                    },
                }
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
            "Click/input contract: select one exact A-reference from a v=Axx line. "
            "Lines without v are evidence only. The runtime clicks the node center. Swipe coordinates "
            "alone use device-independent 0..1000 values. Example: swipe "
            '{"summary":"Scroll up","direction":"up","x1":'
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
    extra = (
        {
            key: deepcopy(raw_extra[key])
            for key in _PLANNER_CONTEXT_KEYS
            if key in raw_extra and raw_extra[key] is not None
        }
        if isinstance(raw_extra, dict)
        else None
    )
    execution_history = ""
    if not lightweight_retry and isinstance(extra, dict) and extra:
        context = dict(extra)
        context.pop("installed_apps", None)
        execution_history = str(context.pop("execution_history", "") or "").strip()
        context.pop("function_execution", None)
        context.pop("previous_action", None)
        recent_actions = context.pop("recent_actions", None)
        transfer_hint = context.pop("transfer_candidates_hint", None)
        if context.get("previous_action_error") or recent_actions:
            lines.append(
                "Inspect the action history, observed results, and any previous "
                "error before selecting again. The latest accessibility state is "
                "authoritative. Do not repeat the same action or no-progress "
                "sequence; choose a different visible control or path, finish, "
                "or abort."
            )
        if context:
            lines.extend(
                (
                    "Recent execution context:",
                    json.dumps(context, ensure_ascii=False, separators=(",", ":")),
                )
            )
        if transfer_hint:
            lines.extend(
                (
                    "OmniTransfer candidate hint (ranked; verify against the current screenshot):",
                    json.dumps(transfer_hint, ensure_ascii=False, separators=(",", ":")),
                )
            )
    if not lightweight_retry:
        lines.extend(("Past Actions:", execution_history or "0. No action yet."))
    if not lightweight_retry:
        lines.extend(
            (
                f"Relevant UI elements ({projection.selected_count}/{projection.candidate_count}):",
                projection.text,
            )
        )
        if projection.visual_context_required:
            lines.extend(
                (
                    "This screen contains a repeated or unlabeled action surface. "
                    "Use the current screenshot and accessibility bounds together, "
                    "then return the visible target's current-screen relative x/y "
                    "coordinates. Never reuse a point from an earlier screen.",
                )
            )
        if any(
            '"d":"Delete"' in line
            or '"d":"Save"' in line
            or '"d":"Send"' in line
            for line in projection.text.splitlines()
        ):
            lines.append(
                "When a labeled control directly performs the named goal effect "
                "(for example Delete, Save, or Send), select it before generic "
                "navigation such as More options."
            )
    lines.append(
        "Review the complete Past Actions and current UI. If they indicate that "
        "the Goal has been completed, choose `finished`; otherwise choose exactly "
        "one next Action."
    )
    return "\n".join(lines)


__all__ = [
    "ModelToolCallError",
    "SYSTEM_PROMPT",
    "build_model_turn_request",
    "parse_model_turn_response",
]
