from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import sysconfig
from typing import Any

CANONICAL_ACTION_SCHEMA_FILENAME = "oob_canonical_actions.v1.json"
OMNIFLOW_RUN_LOG_SCHEMA_FILENAME = "omniflow_run_log.v1.json"
CHECKER_RULE_SCHEMA_FILENAME = "omniflow_checker_rule.v1.json"
VLM_ACTION_TOOL_NAMES = (
    "click",
    "input_text",
    "swipe",
    "open_app",
    "press_key",
    "wait",
    "finished",
)
_VLM_ACTION_ARGUMENT_NAMES = {
    "click": ("x", "y"),
    "input_text": ("text", "x", "y"),
    "swipe": ("direction", "x1", "y1", "x2", "y2"),
    "open_app": ("package_name",),
    "press_key": ("key",),
    "wait": ("duration_ms",),
    "finished": ("content",),
}


def canonical_action_schema_path() -> Path:
    return _schema_path(CANONICAL_ACTION_SCHEMA_FILENAME)


def omniflow_run_log_schema_path() -> Path:
    return _schema_path(OMNIFLOW_RUN_LOG_SCHEMA_FILENAME)


def checker_rule_schema_path() -> Path:
    return _schema_path(CHECKER_RULE_SCHEMA_FILENAME)


def _schema_path(filename: str) -> Path:
    source_path = Path(__file__).resolve()
    data_root = Path(sysconfig.get_path("data") or sys.prefix)
    candidates = tuple(
        parent / "schemas" / "oob" / filename for parent in source_path.parents
    ) + (data_root / "schemas" / "oob" / filename,)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(filename)


def load_canonical_action_schema() -> dict[str, Any]:
    return _load_schema(canonical_action_schema_path())


def load_omniflow_run_log_schema() -> dict[str, Any]:
    return _load_schema(omniflow_run_log_schema_path())


def load_checker_rule_schema() -> dict[str, Any]:
    return _load_schema(checker_rule_schema_path())


def _load_schema(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("canonical_action_schema_must_be_object")
    return payload


def openai_action_tools(*, include_summary: bool = False) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for action in load_canonical_action_schema().get("tools") or ():
        if not isinstance(action, dict) or action.get("model_visible") is False:
            continue
        properties: dict[str, Any] = {}
        required: list[str] = []
        if include_summary:
            properties["summary"] = {
                "type": "string",
                "description": (
                    "One concise sentence containing any accumulated goal-relevant "
                    "facts that must survive to the next turn, plus the immediate "
                    "purpose of this action. This becomes cross-turn memory."
                ),
            }
            required.append("summary")
        for argument in action.get("args") or ():
            if not isinstance(argument, dict) or not argument.get("name"):
                continue
            name = str(argument["name"])
            argument_type = str(argument.get("type") or "string")
            schema = {
                "type": "array" if argument_type == "string_array" else argument_type
            }
            if argument.get("enum_values"):
                schema["enum"] = list(argument["enum_values"])
            if argument.get("minimum") is not None:
                schema["minimum"] = argument["minimum"]
            if argument.get("maximum") is not None:
                schema["maximum"] = argument["maximum"]
            if schema["type"] == "array":
                schema["items"] = {"type": "string"}
            if schema["type"] == "object":
                schema["additionalProperties"] = bool(
                    argument.get("additional_properties")
                )
            description = argument.get("description") or {}
            if isinstance(description, dict) and description.get("en_us"):
                schema["description"] = description["en_us"]
            properties[name] = schema
            if argument.get("required"):
                required.append(name)
        description = action.get("description") or {}
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": str(action.get("name") or ""),
                    "description": description.get("en_us", "")
                    if isinstance(description, dict)
                    else str(description),
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                },
            }
        )
    return tools


def vlm_action_tools(*, include_summary: bool = False) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for tool in openai_action_tools(include_summary=include_summary):
        function = tool.get("function", {})
        name = str(function.get("name") or "")
        allowed_arguments = _VLM_ACTION_ARGUMENT_NAMES.get(name)
        if allowed_arguments is None:
            continue
        parameters = function.get("parameters", {})
        properties = parameters.get("properties", {})
        allowed = ({"summary"} if include_summary else set()).union(
            allowed_arguments
        )
        parameters["properties"] = {
            key: value for key, value in properties.items() if key in allowed
        }
        parameters["required"] = [
            key for key in parameters.get("required", ()) if key in allowed
        ]
        tools.append(tool)
    return tools


def canonicalize_action(
    value: Any,
    *,
    replayable_only: bool = False,
    persisted_only: bool = True,
    allow_non_action: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"tool", "args"}:
        raise ValueError("canonical_action_contract_invalid")
    tool = str(value.get("tool") or "").strip().lower()
    specs = {
        str(item.get("name") or "").strip().lower(): item
        for item in load_canonical_action_schema().get("tools") or ()
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    }
    tool_spec = specs.get(tool)
    if tool_spec is None:
        raise ValueError(f"canonical_action_tool_unsupported:{tool}")
    kind = str(tool_spec.get("kind") or "")
    if kind != "action" and not allow_non_action:
        raise ValueError(f"canonical_action_kind_invalid:{kind}:{tool}")
    if replayable_only and tool_spec.get("replayable") is False:
        raise ValueError(f"canonical_action_tool_not_replayable:{tool}")
    raw_args = value.get("args")
    if not isinstance(raw_args, dict):
        raise ValueError("canonical_action_args_must_be_object")
    all_arg_specs = [
        item for item in tool_spec.get("args") or () if isinstance(item, dict)
    ]
    known_arg_names = {str(item.get("name") or "") for item in all_arg_specs}
    unknown = sorted(set(raw_args) - known_arg_names)
    if unknown:
        raise ValueError(f"canonical_action_args_unknown:{tool}:{','.join(unknown)}")
    arg_specs = [
        item
        for item in all_arg_specs
        if not persisted_only or item.get("persisted") is not False
    ]
    args: dict[str, Any] = {}
    for spec in arg_specs:
        name = str(spec.get("name") or "")
        if name not in raw_args or raw_args[name] is None:
            continue
        args[name] = _canonical_arg(raw_args[name], spec)
    missing = [
        str(spec.get("name") or "")
        for spec in arg_specs
        if spec.get("required") and str(spec.get("name") or "") not in args
    ]
    if missing:
        raise ValueError(
            f"canonical_action_required_args_missing:{tool}:{','.join(missing)}"
        )
    return {"tool": tool, "args": args}


def _canonical_arg(value: Any, spec: dict[str, Any]) -> Any:
    name = str(spec.get("name") or "")
    kind = str(spec.get("type") or "string")
    if kind == "string":
        if not isinstance(value, str):
            raise ValueError(f"canonical_action_arg_type_invalid:{name}")
        converted: Any = value
    elif kind in {"number", "integer"}:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"canonical_action_arg_type_invalid:{name}")
        number = float(value)
        if not math.isfinite(number) or (kind == "integer" and not number.is_integer()):
            raise ValueError(f"canonical_action_arg_type_invalid:{name}")
        converted = int(number) if number.is_integer() else number
    elif kind == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"canonical_action_arg_type_invalid:{name}")
        converted = value
    elif kind == "object":
        if not isinstance(value, dict):
            raise ValueError(f"canonical_action_arg_type_invalid:{name}")
        converted = json.loads(json.dumps(value, ensure_ascii=False))
    elif kind == "string_array":
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"canonical_action_arg_type_invalid:{name}")
        converted = list(value)
    else:
        raise ValueError(f"canonical_action_arg_type_invalid:{name}")
    if spec.get("enum_values") and converted not in spec["enum_values"]:
        raise ValueError(f"canonical_action_arg_enum_invalid:{name}")
    if isinstance(converted, (int, float)) and not isinstance(converted, bool):
        if spec.get("minimum") is not None and converted < spec["minimum"]:
            raise ValueError(f"canonical_action_arg_range_invalid:{name}")
        if spec.get("maximum") is not None and converted > spec["maximum"]:
            raise ValueError(f"canonical_action_arg_range_invalid:{name}")
    return converted


__all__ = [
    "CANONICAL_ACTION_SCHEMA_FILENAME",
    "OMNIFLOW_RUN_LOG_SCHEMA_FILENAME",
    "CHECKER_RULE_SCHEMA_FILENAME",
    "VLM_ACTION_TOOL_NAMES",
    "canonical_action_schema_path",
    "omniflow_run_log_schema_path",
    "canonicalize_action",
    "checker_rule_schema_path",
    "load_canonical_action_schema",
    "load_omniflow_run_log_schema",
    "load_checker_rule_schema",
    "openai_action_tools",
    "vlm_action_tools",
]
