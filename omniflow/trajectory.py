from __future__ import annotations

import json
from typing import Any

from omniflow.schemas import canonicalize_action, load_canonical_run_log_schema

CANONICAL_RUN_LOG_SCHEMA_VERSION = "omniflow.canonical_run_log.v1"
_STATE_FIELDS = {
    "state_id",
    "package_name",
    "activity_name",
    "display",
}


def canonicalize_run_log(value: dict[str, Any]) -> dict[str, Any]:
    """Validate and copy the single schema-owned RunLog representation."""
    schema = load_canonical_run_log_schema()
    _validate_schema(value, schema, schema, "run_log")
    canonical = _copy(value)
    canonical["steps"] = [canonicalize_run_log_step(step) for step in value["steps"]]
    _validate_schema(canonical, schema, schema, "run_log")
    return canonical


def canonicalize_run_log_step(
    value: Any,
    *,
    replayable_only: bool = False,
) -> dict[str, Any]:
    schema = load_canonical_run_log_schema()
    step_schema = {"$ref": "#/$defs/step"}
    _validate_schema(value, step_schema, schema, "run_log_step")
    canonical = _copy(value)
    canonical["action"] = canonicalize_action(
        value["action"],
        replayable_only=replayable_only,
        allow_non_action=True,
    )
    _validate_schema(canonical, step_schema, schema, "run_log_step")
    return canonical


def _validate_schema(
    value: Any,
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    path: str,
) -> None:
    if "$ref" in schema:
        _validate_schema(value, _resolve_ref(root_schema, schema["$ref"]), root_schema, path)
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
    elif isinstance(value, str) and len(value) < int(schema.get("minLength") or 0):
        raise ValueError(f"run_log_schema_invalid:{path}:minLength")
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
        branch = schema.get("then") if _schema_matches(value, condition, root_schema) else schema.get("else")
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


def canonicalize_state(value: Any) -> dict[str, Any]:
    """Validate and copy the one state representation used by every core boundary."""
    if not isinstance(value, dict):
        raise ValueError("run_log_state_must_be_object")
    _reject_unknown(value, _STATE_FIELDS, "run_log_state")
    state_id = _required_string(value.get("state_id"), "run_log_state_id_required")
    state = {"state_id": state_id}
    for key, item in value.items():
        if key == "state_id":
            continue
        if key == "display":
            if not isinstance(item, dict) or set(item) != {"width", "height"}:
                raise ValueError("run_log_state_display_invalid")
            for dimension in ("width", "height"):
                number = item.get(dimension)
                if not isinstance(number, int) or isinstance(number, bool) or number <= 0:
                    raise ValueError("run_log_state_display_invalid")
        elif not isinstance(item, str):
            raise ValueError(f"run_log_state_{key}_must_be_string")
        state[key] = item
    return state


def _reject_unknown(value: dict[str, Any], allowed: set[str], prefix: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{prefix}_unknown_fields:{','.join(unknown)}")


def _required_string(value: Any, error: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(error)
    return value


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


__all__ = [
    "CANONICAL_RUN_LOG_SCHEMA_VERSION",
    "canonicalize_run_log",
    "canonicalize_run_log_step",
]
