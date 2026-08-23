from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

from omniflow.core.androidworld_accessibility import androidworld_forest_xml
from omniflow.core.schemas import load_omniflow_run_log_schema

OMNIFLOW_RUN_LOG_SCHEMA_VERSION = "omniflow.run_log.v1"
_ACTION_TYPES = {
    "answer",
    "click",
    "double_tap",
    "input_text",
    "keyboard_enter",
    "long_press",
    "navigate_back",
    "navigate_home",
    "open_app",
    "press_keyboard",
    "scroll",
    "status",
    "swipe",
    "unknown",
    "wait",
}
_ACTION_FIELDS = {
    "action_type",
    "index",
    "x",
    "y",
    "x1",
    "y1",
    "x2",
    "y2",
    "duration_ms",
    "text",
    "direction",
    "app_name",
    "goal_status",
    "keycode",
    "clear_text",
}


def canonicalize_run_log(value: dict[str, Any]) -> dict[str, Any]:
    """Validate and copy the only RunLog accepted by OmniFlow runtime code."""
    schema = load_omniflow_run_log_schema()
    prepared = _copy(value)
    _drop_screenshot_hashes(prepared)
    _validate_schema(prepared, schema, schema, "run_log")
    canonical = _copy(prepared)
    canonical["steps"] = [
        canonicalize_run_log_step(step) for step in prepared["steps"]
    ]
    for step in canonical["steps"]:
        _validate_screenshot_reference(observation_screenshot(step["observation"]))
        if "next_observation" in step:
            _validate_screenshot_reference(
                observation_screenshot(step["next_observation"])
            )
    if "final_observation" in canonical:
        _validate_screenshot_reference(
            observation_screenshot(canonical["final_observation"])
        )
    provenance = canonical["provenance"]
    if provenance["kind"] == "legacy_import":
        required = {"source_path", "source_sha256", "source_schema_version"}
        missing = sorted(required - set(provenance))
        if missing:
            raise ValueError(
                "run_log_provenance_required:" + ",".join(missing)
            )
        if not Path(provenance["source_path"]).is_absolute():
            raise ValueError("run_log_provenance_source_path_must_be_absolute")
    _validate_schema(canonical, schema, schema, "run_log")
    return canonical


def canonicalize_run_log_step(value: Any) -> dict[str, Any]:
    schema = load_omniflow_run_log_schema()
    step_schema = {"$ref": "#/$defs/step"}
    prepared = _copy(value)
    _drop_screenshot_hashes(prepared)
    _validate_schema(prepared, step_schema, schema, "run_log_step")
    canonical = _copy(prepared)
    canonical["action"] = canonicalize_androidworld_action(prepared["action"])
    _validate_schema(canonical, step_schema, schema, "run_log_step")
    return canonical


def canonicalize_androidworld_action(value: Any) -> dict[str, Any]:
    """Validate the serializable fields of AndroidWorld ``JSONAction``."""
    if not isinstance(value, dict):
        raise ValueError("androidworld_action_must_be_object")
    unknown = sorted(set(value) - _ACTION_FIELDS)
    if unknown:
        raise ValueError("androidworld_action_unknown_fields:" + ",".join(unknown))
    action = {key: item for key, item in value.items() if item is not None}
    action_type = action.get("action_type")
    if action_type not in _ACTION_TYPES:
        raise ValueError(f"androidworld_action_type_invalid:{action_type}")
    if "index" in action:
        _integer(action["index"], "androidworld_action_index_invalid")
        if "x" in action or "y" in action:
            raise ValueError("androidworld_action_index_or_coordinates_required")
    for key in ("x", "y", "x1", "y1", "x2", "y2"):
        if key in action:
            _number(
                action[key], f"androidworld_action_{key}_invalid"
            )
    swipe_coordinates = {"x1", "y1", "x2", "y2"}
    provided_swipe_coordinates = swipe_coordinates & set(action)
    if provided_swipe_coordinates:
        if action_type != "swipe":
            raise ValueError("androidworld_action_swipe_coordinates_require_swipe")
        if provided_swipe_coordinates != swipe_coordinates:
            raise ValueError("androidworld_action_swipe_coordinates_incomplete")
    if "duration_ms" in action:
        _integer(action["duration_ms"], "androidworld_action_duration_ms_invalid")
        if action["duration_ms"] < 0:
            raise ValueError("androidworld_action_duration_ms_invalid")
    if action.get("direction") is not None and action["direction"] not in {
        "left",
        "right",
        "down",
        "up",
    }:
        raise ValueError("androidworld_action_direction_invalid")
    if action.get("keycode") is not None and (
        not isinstance(action["keycode"], str)
        or not action["keycode"].startswith("KEYCODE_")
    ):
        raise ValueError("androidworld_action_keycode_required")
    return _copy(action)


def require_complete_source_run_log(value: dict[str, Any]) -> dict[str, Any]:
    """Require a successful official-validator source with executable steps."""
    run_log = canonicalize_run_log(value)
    if run_log["validator"]["official"] is not True:
        raise ValueError("androidworld_source_run_log_official_validator_required")
    if run_log["status"] != "succeeded" or run_log["success"] is not True:
        raise ValueError("androidworld_source_run_log_success_required")
    if not run_log["steps"]:
        raise ValueError("androidworld_source_run_log_steps_required")
    return run_log


def state_id(observation: dict[str, Any]) -> str:
    auxiliaries = observation.get("auxiliaries")
    if isinstance(auxiliaries, dict):
        explicit = str(auxiliaries.get("state_id") or "").strip()
        if explicit:
            return explicit
    identity = _copy(observation)
    if "screenshot" in identity:
        identity["screenshot"] = None
    if "pixels" in identity:
        identity["pixels"] = None
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "state_" + hashlib.sha256(encoded.encode()).hexdigest()[:20]


def observation_display(observation: dict[str, Any]) -> tuple[int, int] | None:
    pixels = observation_screenshot(observation)
    if isinstance(pixels, dict):
        return int(pixels["width"]), int(pixels["height"])
    auxiliaries = observation.get("auxiliaries")
    display = auxiliaries.get("display") if isinstance(auxiliaries, dict) else None
    if isinstance(display, dict):
        width = display.get("width")
        height = display.get("height")
        if (
            isinstance(width, int)
            and not isinstance(width, bool)
            and width > 0
            and isinstance(height, int)
            and not isinstance(height, bool)
            and height > 0
        ):
            return width, height
    return None


def observation_xml(observation: dict[str, Any]) -> str:
    xml = observation.get("xml")
    if isinstance(xml, str) and xml.strip():
        return xml.strip()
    value = observation.get("forest")
    if isinstance(value, str):
        return value
    display = observation_display(observation)
    if value is None or display is None:
        return ""
    return androidworld_forest_xml(value, screen_size=display)


def observation_screenshot(observation: dict[str, Any]) -> Any:
    """Return the compact screenshot reference, accepting legacy pixels."""
    screenshot = observation.get("screenshot")
    return screenshot if screenshot is not None else observation.get("pixels")


def _validate_screenshot_reference(value: Any) -> None:
    if value is None:
        return
    path = Path(str(value.get("path") or "")).expanduser()
    if not path.is_absolute():
        raise ValueError("run_log_screenshot_path_must_be_absolute")


def _integer(value: Any, error: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(error)
    return value


def _number(value: Any, error: str) -> int | float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(error)
    return value


def _validate_schema(
    value: Any,
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    path: str,
) -> None:
    if "$ref" in schema:
        _validate_schema(value, _resolve_ref(root_schema, schema["$ref"]), root_schema, path)
        return
    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        if not any(_schema_matches(value, item, root_schema) for item in any_of):
            raise ValueError(f"run_log_schema_invalid:{path}:anyOf")
        return
    _validate_type(value, schema.get("type"), path)
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"run_log_schema_invalid:{path}:const")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"run_log_schema_invalid:{path}:enum")
    if isinstance(value, dict):
        required = schema.get("required") or ()
        missing = [str(field) for field in required if field not in value]
        if missing:
            raise ValueError(
                f"run_log_schema_invalid:{path}:required:{','.join(missing)}"
            )
        properties = schema.get("properties") or {}
        additional = schema.get("additionalProperties", True)
        for field, item in value.items():
            child_path = f"{path}.{field}"
            if field in properties:
                _validate_schema(item, properties[field], root_schema, child_path)
            elif additional is False:
                raise ValueError(
                    f"run_log_schema_invalid:{path}:additionalProperties:{field}"
                )
            elif isinstance(additional, dict):
                _validate_schema(item, additional, root_schema, child_path)
    elif isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_schema(item, item_schema, root_schema, f"{path}[{index}]")
    elif isinstance(value, str):
        if len(value) < int(schema.get("minLength") or 0):
            raise ValueError(f"run_log_schema_invalid:{path}:minLength")
        pattern = schema.get("pattern")
        if pattern and re.search(str(pattern), value) is None:
            raise ValueError(f"run_log_schema_invalid:{path}:pattern")
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and "minimum" in schema
        and value < schema["minimum"]
    ):
        raise ValueError(f"run_log_schema_invalid:{path}:minimum")
    for item_schema in schema.get("allOf") or ():
        _validate_schema(value, item_schema, root_schema, path)
    condition = schema.get("if")
    if isinstance(condition, dict):
        branch = (
            schema.get("then")
            if _schema_matches(value, condition, root_schema)
            else schema.get("else")
        )
        if isinstance(branch, dict):
            _validate_schema(value, branch, root_schema, path)


def _validate_type(value: Any, expected: Any, path: str) -> None:
    if expected is None:
        return
    matches = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }.get(str(expected), False)
    if not matches:
        raise ValueError(f"run_log_schema_invalid:{path}:type:{expected}")


def _schema_matches(value: Any, schema: dict[str, Any], root_schema: dict[str, Any]) -> bool:
    try:
        _validate_schema(value, schema, root_schema, "condition")
        return True
    except ValueError:
        return False


def _resolve_ref(root_schema: dict[str, Any], ref: Any) -> dict[str, Any]:
    if not isinstance(ref, str) or not ref.startswith("#/"):
        raise ValueError(f"run_log_schema_ref_unsupported:{ref}")
    resolved: Any = root_schema
    for segment in ref[2:].split("/"):
        key = segment.replace("~1", "/").replace("~0", "~")
        if not isinstance(resolved, dict) or key not in resolved:
            raise ValueError(f"run_log_schema_ref_missing:{ref}")
        resolved = resolved[key]
    if not isinstance(resolved, dict):
        raise ValueError(f"run_log_schema_ref_invalid:{ref}")
    return resolved


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _drop_screenshot_hashes(value: Any) -> None:
    if isinstance(value, dict):
        for field in ("screenshot", "pixels"):
            screenshot = value.get(field)
            if isinstance(screenshot, dict):
                screenshot.pop("sha256", None)
        for item in value.values():
            _drop_screenshot_hashes(item)
    elif isinstance(value, list):
        for item in value:
            _drop_screenshot_hashes(item)


__all__ = [
    "OMNIFLOW_RUN_LOG_SCHEMA_VERSION",
    "canonicalize_androidworld_action",
    "canonicalize_run_log",
    "canonicalize_run_log_step",
    "observation_display",
    "observation_screenshot",
    "observation_xml",
    "require_complete_source_run_log",
    "state_id",
]
