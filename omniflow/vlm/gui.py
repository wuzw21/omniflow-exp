from __future__ import annotations

import base64
from collections.abc import Callable
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import re
from typing import Any
from xml.etree import ElementTree

from PIL import Image

from omniflow.core.config import DEFAULT_PLANNER_SYSTEM_PROMPT
from omniflow.core.model import Function, ToolCall
from omniflow.core.schemas import canonicalize_action, vlm_action_tools
from omniflow.functions.artifact import validate_arguments
from omniflow.vlm.tool_arguments import load_tool_arguments
from omniflow.vlm_coordinates import (
    display_size,
    relative_args_to_canonical,
    relative_coordinate_tools,
)

SYSTEM_PROMPT = DEFAULT_PLANNER_SYSTEM_PROMPT
_MAX_ACCESSIBILITY_ROWS = 50

_PLANNER_CONTEXT_KEYS = (
    "planner_feedback",
    "forbid_finished",
    "previous_action_error",
    "previous_action",
    "recent_actions",
    "execution_history",
    "user_input",
)

PlannerContextProjector = Callable[[dict[str, Any]], dict[str, Any]]


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
    installed_apps: dict[str, str] | None = None,
    functions: list[Function] | tuple[Function, ...] = (),
    context_projector: PlannerContextProjector | None = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict[str, Any]:
    projected_context = (context_projector or project_planner_context)(dict(state))
    if not isinstance(projected_context, dict):
        raise TypeError("planner_context_projector_result_invalid")
    projected_state = {**state, **projected_context}
    display = (
        state.get("display") if isinstance(state.get("display"), dict) else None
    )
    text = _turn_text(
        goal=goal,
        state=projected_state,
        max_steps=max_steps,
        turn_index=turn_index,
        target_package_name=target_package_name,
        xml_text=str(projected_state.get("xml") or ""),
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    current_image = _state_image_data_uri(state) if _state_has_screenshot(state) else ""
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
    native_tools = vlm_action_tools(include_summary=True)
    projected_extra = projected_state.get("extra")
    if isinstance(projected_extra, dict) and bool(
        projected_extra.get("forbid_finished")
    ):
        native_tools = [
            tool
            for tool in native_tools
            if tool.get("function", {}).get("name") != "finished"
        ]
    tools = relative_coordinate_tools(native_tools, display)
    tools.extend(function_tools(functions, include_summary=True))
    request: dict[str, Any] = {
        "model": str(model),
        "messages": [
            {"role": "system", "content": str(system_prompt).strip() or SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "max_tokens": 512,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": tools,
        "tool_choice": "required",
        "parallel_tool_calls": False,
    }
    request["enable_thinking"] = False
    request["thinking"] = {"type": "disabled"}
    request["reasoning_effort"] = "none"
    return request


def project_planner_context(state: dict[str, Any]) -> dict[str, Any]:
    projected = dict(state)
    display = projected.get("display")
    projected["xml"] = _compact_accessibility_observation(
        str(projected.get("xml") or ""),
        display=display if isinstance(display, dict) else None,
        current_package=str(projected.get("package_name") or "").strip(),
        max_rows=_MAX_ACCESSIBILITY_ROWS,
    )
    return projected


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
        thumbnail_box = (480, 270) if image.width > image.height else (270, 480)
        image.thumbnail(thumbnail_box, Image.Resampling.LANCZOS)
        output = BytesIO()
        image.save(output, format="WEBP", quality=45, method=6)
        compact = base64.b64encode(output.getvalue()).decode("ascii")
    except Exception:
        return value
    return f"data:image/webp;base64,{compact}"


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


def _compact_accessibility_observation(
    xml_text: str,
    *,
    display: dict[str, Any] | None = None,
    current_package: str = "",
    max_rows: int = _MAX_ACCESSIBILITY_ROWS,
) -> str:
    source = str(xml_text or "").strip()
    if not source:
        return "<none>"
    try:
        root = ElementTree.fromstring(source)
    except ElementTree.ParseError:
        return source

    width, height = _accessibility_dimensions(root, display)
    projected: list[
        tuple[dict[str, Any], tuple[str, tuple[str, ...]] | None, bool, bool]
    ] = []
    for node in root.iter("node"):
        attributes = node.attrib
        if attributes.get("visible-to-user") == "false":
            continue
        text = str(attributes.get("text") or "").strip()
        description = str(attributes.get("content-desc") or "").strip()
        hint = str(attributes.get("hint-text") or "").strip()
        resource_id = str(attributes.get("resource-id") or "").strip()
        draggable_control = _is_draggable_control(attributes)
        actions = [
            name
            for name, attribute in (
                ("click", "clickable"),
                ("long_press", "long-clickable"),
                ("scroll", "scrollable"),
                ("edit", "editable"),
                ("check", "checkable"),
            )
            if attributes.get(attribute) == "true"
        ]
        if draggable_control:
            actions.append("swipe")
        semantic_labels = list(
            dict.fromkeys(
                value for value in (text, description, hint) if value
            )
        )
        labels = list(semantic_labels)
        visual_region_key: tuple[str, tuple[str, ...]] | None = None
        if actions and not labels and resource_id:
            fallback_label = resource_id.rsplit("/", 1)[-1]
            labels.append(fallback_label)
            visual_region_key = (fallback_label, tuple(actions))
        if not labels and not actions:
            continue
        row: dict[str, Any] = {"label": " | ".join(labels)} if labels else {}
        if actions:
            bounds = str(attributes.get("bounds") or "").strip()
            normalized_bounds = _normalized_bounds(
                bounds,
                width=width,
                height=height,
            )
            if normalized_bounds:
                row["bounds_0_1000"] = normalized_bounds
            row["actions"] = actions
            if attributes.get("enabled") == "false":
                row["enabled"] = False
            for state_name in ("checked", "selected", "focused"):
                if attributes.get(state_name) == "true":
                    row[state_name] = True
        node_package = str(attributes.get("package") or "").strip()
        projected.append(
            (
                row,
                visual_region_key,
                bool(actions),
                not current_package or node_package == current_package,
            )
        )

    repeated_visual_regions: dict[tuple[str, tuple[str, ...]], int] = {}
    for _row, key, _interactive, _current_app in projected:
        if key is not None:
            repeated_visual_regions[key] = repeated_visual_regions.get(key, 0) + 1

    interactive_labels = {
        str(row.get("label") or "").strip().casefold()
        for row, _key, interactive, _current_app in projected
        if interactive and str(row.get("label") or "").strip()
    }
    ordered = sorted(
        projected,
        key=lambda item: (
            0 if item[3] else 1,
            0 if item[2] else 1,
        ),
    )
    rows: list[str] = []
    emitted_visual_groups: set[tuple[str, tuple[str, ...]]] = set()
    emitted_rows: set[str] = set()
    omitted = 0
    row_limit = max(1, int(max_rows))
    for row, key, interactive, _current_app in ordered:
        if key is not None and repeated_visual_regions.get(key, 0) > 4:
            if key in emitted_visual_groups:
                omitted += 1
                continue
            emitted_visual_groups.add(key)
            row = {
                "visual_region_group": key[0],
                "count": repeated_visual_regions[key],
                "grounding": "use current screenshot",
            }
        label = str(row.get("label") or "").strip().casefold()
        if not interactive and label and any(
            label in interactive_label for interactive_label in interactive_labels
        ):
            omitted += 1
            continue
        serialized = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        if serialized in emitted_rows:
            omitted += 1
            continue
        if len(rows) >= row_limit:
            omitted += 1
            continue
        emitted_rows.add(serialized)
        rows.append(serialized)
    if omitted:
        rows.append(
            json.dumps(
                {
                    "omitted_elements": omitted,
                    "grounding": "use current screenshot",
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    return "\n".join(rows) or "<none>"


def _is_draggable_control(attributes: dict[str, str]) -> bool:
    class_name = str(attributes.get("class") or "").rsplit(".", 1)[-1].casefold()
    resource_id = str(attributes.get("resource-id") or "").rsplit("/", 1)[-1].casefold()
    return class_name in {"seekbar", "slider"} or any(
        token in resource_id for token in ("seekbar", "slider")
    )


def _accessibility_dimensions(
    root: ElementTree.Element,
    display: dict[str, Any] | None,
) -> tuple[float | None, float | None]:
    try:
        width = float(root.attrib.get("width") or 0)
        height = float(root.attrib.get("height") or 0)
    except (TypeError, ValueError):
        width = height = 0
    if width > 0 and height > 0:
        return width, height
    try:
        return display_size(display)
    except ValueError:
        return None, None


def _normalized_bounds(
    bounds: str,
    *,
    width: float | None,
    height: float | None,
) -> str:
    if width is None or height is None:
        return ""
    values = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", bounds)]
    if len(values) != 4:
        return ""
    left, top, right, bottom = values
    normalized = (
        round(min(1000, max(0, left / width * 1000))),
        round(min(1000, max(0, top / height * 1000))),
        round(min(1000, max(0, right / width * 1000))),
        round(min(1000, max(0, bottom / height * 1000))),
    )
    return f"[{normalized[0]},{normalized[1]}][{normalized[2]},{normalized[3]}]"


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
    model_visible_tools = {
        str(item.get("function", {}).get("name") or "")
        for item in vlm_action_tools()
        if isinstance(item, dict) and isinstance(item.get("function"), dict)
    }
    function_catalog = {function.id: function for function in functions}
    if not tool_calls:
        raise ModelToolCallError("model_turn_tool_calls_empty")
    if len(tool_calls) > 1:
        # Some OpenAI-compatible gateways ignore parallel_tool_calls=false and
        # return several actions in one response.  A GUI turn is deliberately
        # sequential: consume one valid next action and let the next fresh
        # observation ground the following action.  Never replay the extra
        # calls or source coordinates from the same response.
        visible_names = model_visible_tools | set(function_catalog)
        selected_index = next(
            (
                index
                for index, candidate in enumerate(tool_calls)
                if isinstance(candidate, dict)
                and isinstance(candidate.get("function"), dict)
                and str(candidate["function"].get("name") or "").strip()
                in visible_names
            ),
            0,
        )
        discarded = len(tool_calls) - 1
        selected = tool_calls[selected_index]
        tool_calls = [selected]
    else:
        discarded = 0
    tool_call = tool_calls[0]
    if not isinstance(tool_call, dict):
        raise ModelToolCallError("model_turn_tool_call_invalid")
    function = tool_call.get("function")
    if not isinstance(function, dict):
        raise ModelToolCallError("model_turn_tool_call_function_invalid")
    tool = str(function.get("name") or "").strip()
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
    if discarded:
        metadata["discarded_extra_tool_calls"] = discarded
    if arguments_repaired:
        metadata["json_repair"] = {
            "name": "json_repair",
            "applied": True,
        }
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
    xml_text: str,
) -> str:
    display = state.get("display") if isinstance(state.get("display"), dict) else {}
    display_size(display)
    lines = [
        f"Goal: {goal}",
        f"Progress: {turn_index}/{max_steps} model turns used",
        f"Current package: {state.get('package_name') or ''}",
        f"Current activity: {state.get('activity_name') or ''}",
        f"Display: {display.get('width') or ''}x{display.get('height') or ''}",
        "Coordinates and bounds use the current-screen 0..1000 scale.",
    ]
    if target_package_name:
        lines.append(f"Target package: {target_package_name}")
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
    feedback: list[str] = []
    if isinstance(extra, dict) and extra:
        context = dict(extra)
        execution_history = str(context.pop("execution_history", "") or "").strip()
        planner_feedback = str(context.pop("planner_feedback", "") or "").strip()
        previous_action_error = str(
            context.pop("previous_action_error", "") or ""
        ).strip()
        context.pop("forbid_finished", None)
        user_input = str(context.pop("user_input", "") or "").strip()
        if planner_feedback:
            feedback.append(planner_feedback)
        if previous_action_error:
            feedback.append(f"Previous action error: {previous_action_error}")
        if user_input:
            feedback.append(f"User input: {user_input}")
    lines.extend(("Past Actions:", execution_history or "0. No action yet."))
    if feedback:
        lines.extend(("Feedback:", "\n".join(feedback)))
    lines.extend(
        (
            "Current accessibility elements (label-only rows are read-only; rows with actions are interactive):",
            xml_text or "<none>",
        )
    )
    lines.append(
        "Apply the decision policy from the system instructions to the current UI and Past Actions, then return exactly one provided tool call."
    )
    return "\n".join(lines)


__all__ = [
    "ModelToolCallError",
    "PlannerContextProjector",
    "SYSTEM_PROMPT",
    "build_model_turn_request",
    "parse_model_turn_response",
    "project_planner_context",
]
