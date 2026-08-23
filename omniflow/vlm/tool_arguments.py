from __future__ import annotations

import json
from typing import Any

from json_repair import loads as repair_json_loads


def load_tool_arguments(raw_arguments: str) -> tuple[Any, bool]:
    """Parse tool arguments, repairing only structurally truncated JSON."""
    try:
        return json.loads(raw_arguments), False
    except json.JSONDecodeError as original_error:
        if _ends_inside_json_string(raw_arguments):
            raise original_error
        try:
            return (
                repair_json_loads(raw_arguments, skip_json_loads=True),
                True,
            )
        except (TypeError, ValueError, RecursionError):
            raise original_error


def _ends_inside_json_string(value: str) -> bool:
    in_string = False
    escaped = False
    for character in value:
        if not in_string:
            if character == '"':
                in_string = True
            continue
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == '"':
            in_string = False
    return in_string


__all__ = ["load_tool_arguments"]
