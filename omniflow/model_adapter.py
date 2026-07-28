from __future__ import annotations

import math
import re
from typing import Any

_ADAPTER_NAME = "qwen_vl_coordinate_arrays.v1"
_QWEN_VL_MODEL = re.compile(
    r"(?:^|[^a-z0-9])qwen(?:\d+(?:\.\d+)?)?[-_.]?vl(?:[^a-z0-9]|$)",
    re.IGNORECASE,
)
_COORDINATE_PAIRS = {
    "click": (("x", "y"),),
    "long_press": (("x", "y"),),
    "input_text": (("x", "y"),),
    "swipe": (("x1", "y1"), ("x2", "y2")),
}
_MISSING = object()


def adapt_tool_arguments(
    *,
    tool: str,
    arguments: dict[str, Any],
    requested_model: str,
    resolved_model: str,
    display: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    model = _adapter_model(requested_model, resolved_model)
    coordinate_pairs = _COORDINATE_PAIRS.get(tool)
    if not model or coordinate_pairs is None:
        return dict(arguments), None

    adapted = dict(arguments)
    changes: list[dict[str, Any]] = []
    display_width = _positive_number((display or {}).get("width"))
    display_height = _positive_number((display or {}).get("height"))
    for x_field, y_field in coordinate_pairs:
        _adapt_coordinate_pair(
            adapted,
            x_field,
            y_field,
            changes,
            display_width=display_width,
            display_height=display_height,
        )
    if not changes:
        return adapted, None
    return adapted, {
        "name": _ADAPTER_NAME,
        "model": model,
        "tool": tool,
        "changes": changes,
    }


def _adapter_model(requested_model: str, resolved_model: str) -> str:
    for candidate in (resolved_model, requested_model):
        normalized = str(candidate or "").strip()
        if normalized and _QWEN_VL_MODEL.search(normalized):
            return normalized
    return ""


def _adapt_coordinate_pair(
    arguments: dict[str, Any],
    x_field: str,
    y_field: str,
    changes: list[dict[str, Any]],
    *,
    display_width: float | None,
    display_height: float | None,
) -> None:
    x_value = arguments.get(x_field, _MISSING)
    y_value = arguments.get(y_field, _MISSING)
    if (
        display_width is not None
        and display_height is not None
        and isinstance(x_value, list)
        and len(x_value) == 2
        and all(_is_number(value) for value in x_value)
        and isinstance(y_value, list)
        and len(y_value) == 1
        and _is_number(y_value[0])
        and 0 <= float(x_value[0]) <= display_width
        and 0 <= float(x_value[1]) <= display_height
        and 0 <= float(y_value[0]) <= display_height
        and (float(x_value[1]) > 1000 or float(y_value[0]) > 1000)
    ):
        arguments[x_field] = x_value[0]
        arguments[y_field] = x_value[1]
        changes.append(
            {
                "source_field": x_field,
                "source_shape": "pixel_point_with_trailing_y",
                "target_fields": [x_field, y_field],
            }
        )
        return
    if (
        isinstance(x_value, list)
        and len(x_value) == 2
        and all(_is_number(value) for value in x_value)
        and _matches_inferred_y(y_value, x_value[1])
    ):
        arguments[x_field] = x_value[0]
        arguments[y_field] = x_value[1]
        changes.append(
            {
                "source_field": x_field,
                "source_shape": "number_pair",
                "target_fields": [x_field, y_field],
            }
        )
        return

    for field in (x_field, y_field):
        value = arguments.get(field, _MISSING)
        if isinstance(value, list) and len(value) == 1 and _is_number(value[0]):
            arguments[field] = value[0]
            changes.append(
                {
                    "source_field": field,
                    "source_shape": "singleton_number_array",
                    "target_fields": [field],
                }
            )


def _matches_inferred_y(value: Any, inferred_y: int | float) -> bool:
    if value is _MISSING:
        return True
    if _is_number(value):
        return value == inferred_y
    return (
        isinstance(value, list)
        and len(value) == 1
        and _is_number(value[0])
        and value[0] == inferred_y
    )


def _is_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _positive_number(value: Any) -> float | None:
    return float(value) if _is_number(value) and float(value) > 0 else None


__all__ = ["adapt_tool_arguments"]
