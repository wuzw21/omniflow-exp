from __future__ import annotations

import copy
import math
from typing import Any

_TOOL_COORDINATE_AXES: dict[str, dict[str, str]] = {
    "click": {"x": "x", "y": "y"},
    "long_press": {"x": "x", "y": "y"},
    "input_text": {"x": "x", "y": "y"},
    "swipe": {"x1": "x", "y1": "y", "x2": "x", "y2": "y"},
}


def screen_pixel_tools(
    tools: list[dict[str, Any]],
    display: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    width, height = display_size(display)
    converted = copy.deepcopy(tools)
    for tool in converted:
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "")
        axes = _TOOL_COORDINATE_AXES.get(name, {})
        parameters = function.get("parameters")
        properties = parameters.get("properties") if isinstance(parameters, dict) else None
        if not isinstance(properties, dict):
            continue
        for field, axis in axes.items():
            schema = properties.get(field)
            if not isinstance(schema, dict):
                continue
            dimension = width if axis == "x" else height
            schema["minimum"] = 0
            schema["maximum"] = dimension
            schema["description"] = (
                f"Raw {axis.upper()} pixel coordinate in the current original "
                f"screen frame; valid range is 0..{dimension}."
            )
    return converted


def canonical_action_to_screen_pixels(
    action: dict[str, Any],
    display: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(action, dict):
        return copy.deepcopy(action)
    tool = str(action.get("tool") or "")
    args = action.get("args")
    if not isinstance(args, dict) or tool not in _TOOL_COORDINATE_AXES:
        return copy.deepcopy(action)
    converted, _metadata = _convert(
        tool=tool,
        args=args,
        display=display,
        to_screen_pixels=True,
    )
    return {**copy.deepcopy(action), "args": converted}


def screen_pixel_args_to_canonical(
    *,
    tool: str,
    args: dict[str, Any],
    display: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    return _convert(
        tool=tool,
        args=args,
        display=display,
        to_screen_pixels=False,
    )


def screen_context_to_pixels(
    extra: dict[str, Any],
    display: dict[str, Any] | None,
) -> dict[str, Any]:
    converted = copy.deepcopy(extra)
    previous = converted.get("previous_action")
    if isinstance(previous, dict):
        converted["previous_action"] = canonical_action_to_screen_pixels(
            previous,
            display,
        )
    recent = converted.get("recent_actions")
    if isinstance(recent, list):
        converted["recent_actions"] = [
            canonical_action_to_screen_pixels(item, display)
            if isinstance(item, dict)
            else copy.deepcopy(item)
            for item in recent
        ]
    return converted


def display_size(display: dict[str, Any] | None) -> tuple[float, float]:
    value = display or {}
    width = _positive_number(value.get("width"))
    height = _positive_number(value.get("height"))
    if width is None or height is None:
        raise ValueError("vlm_coordinate_display_required")
    return width, height


def _convert(
    *,
    tool: str,
    args: dict[str, Any],
    display: dict[str, Any] | None,
    to_screen_pixels: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    axes = _TOOL_COORDINATE_AXES.get(str(tool), {})
    converted = copy.deepcopy(args)
    present = [field for field in axes if field in converted]
    if not present:
        return converted, None
    width, height = display_size(display)
    changes: list[dict[str, Any]] = []
    for field in present:
        value = converted[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"canonical_action_arg_type_invalid:{field}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"canonical_action_arg_type_invalid:{field}")
        dimension = width if axes[field] == "x" else height
        if to_screen_pixels:
            if not 0 <= number <= 1000:
                raise ValueError(f"canonical_action_arg_range_invalid:{field}")
            result = number / 1000.0 * dimension
        else:
            if not 0 <= number <= dimension:
                raise ValueError(f"canonical_action_arg_range_invalid:{field}")
            result = number / dimension * 1000.0
        converted[field] = _compact_number(result)
        changes.append(
            {
                "field": field,
                "from": _compact_number(number),
                "to": converted[field],
            }
        )
    return converted, {
        "name": (
            "relative_0_1000_to_screen_pixels.v1"
            if to_screen_pixels
            else "screen_pixels_to_relative_0_1000.v1"
        ),
        "display": {
            "width": _compact_number(width),
            "height": _compact_number(height),
        },
        "changes": changes,
    }


def _positive_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0 else None


def _compact_number(value: float) -> int | float:
    return int(value) if value.is_integer() else value


__all__ = [
    "canonical_action_to_screen_pixels",
    "display_size",
    "screen_context_to_pixels",
    "screen_pixel_args_to_canonical",
    "screen_pixel_tools",
]
