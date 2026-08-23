from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Iterable
import xml.etree.ElementTree as ET

from omniflow.core.model import Action, Function, FunctionStep
from omniflow.core.schemas import (
    canonicalize_action,
    load_canonical_action_schema,
)
from omniflow.core.trajectory import (
    observation_display,
    observation_xml,
    state_id,
)
from omniflow.runlog import import_run_log_evidence, project_androidworld_step_actions
from omniflow.runtime.checker import validate_checker_rule
from omniflow.runtime.semantic_grounding import semantic_target_at_point

FUNCTION_ARTIFACT_VERSION = "omniflow.function.v3"
STORE_VERSION = "omniflow.store.v2"
_AUTHORING_STAGE_ATTEMPTS = 3

_TOP_LEVEL_FIELDS = {
    "schema_version",
    "function_id",
    "name",
    "description",
    "input_schema",
    "bindings",
    "transfer_states",
    "steps",
    "checker_rules",
    "model_handoffs",
    "agent_visible",
}
_REQUIRED_TOP_LEVEL_FIELDS = _TOP_LEVEL_FIELDS - {"model_handoffs"}
_SOURCE_PATH = re.compile(
    r"^\$\.arguments(?P<tail>(?:\.[A-Za-z_][A-Za-z0-9_]*|\[\d+\])+)$"
)
_TARGET_PATH = re.compile(
    r"^\$\.steps\[(?P<action_index>\d+)]\.action\.args"
    r"(?P<tail>(?:\.[A-Za-z_][A-Za-z0-9_]*|\[\d+\])+)$"
)
_PATH_TOKEN = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)]")


def function_authoring_tool() -> dict[str, Any]:
    """Return the single v3 Function authoring contract."""

    return {
        "type": "function",
        "function": {
            "name": "author_functions",
            "description": (
                "Author one or more independent Functions. Every step references "
                "one or more available RunLog observations as transfer states."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["functions"],
                "properties": {
                    "functions": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "function_id",
                                "name",
                                "description",
                                "steps",
                            ],
                            "properties": {
                                "function_id": {
                                    "type": "string",
                                    "pattern": "^[a-z][a-z0-9_]{0,63}$",
                                },
                                "name": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 120,
                                },
                                "description": {
                                    "type": "string",
                                    "minLength": 1,
                                },
                                "steps": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "required": [
                                            "transfer_state_ids",
                                            "action",
                                        ],
                                        "properties": {
                                            "transfer_state_ids": {
                                                "type": "array",
                                                "minItems": 1,
                                                "uniqueItems": True,
                                                "items": {
                                                    "type": "string",
                                                    "minLength": 1,
                                                },
                                            },
                                            "action": {
                                                "type": "object",
                                                "additionalProperties": False,
                                                "required": ["tool", "args"],
                                                "properties": {
                                                    "tool": {
                                                        "type": "string",
                                                        "minLength": 1,
                                                    },
                                                    "args": {"type": "object"},
                                                },
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    }
                },
            },
        },
    }


def _canonical_function_observation(value: Any) -> dict[str, Any]:
    """Keep the exact RunLog observation shape used as a transfer state."""

    if not isinstance(value, dict):
        raise ValueError("function_transfer_state_must_be_observation")
    fields = set(value)
    if fields == {"screenshot", "xml"}:
        xml = value.get("xml")
        screenshot = value.get("screenshot")
        if not isinstance(xml, str) or not xml:
            raise ValueError("function_transfer_state_xml_required")
        if screenshot is not None:
            if not isinstance(screenshot, dict) or set(screenshot) != {
                "path",
                "width",
                "height",
                "mime_type",
            }:
                raise ValueError("function_transfer_state_screenshot_invalid")
            if (
                not isinstance(screenshot.get("path"), str)
                or not screenshot["path"]
                or not all(
                    isinstance(screenshot.get(key), int)
                    and not isinstance(screenshot.get(key), bool)
                    and screenshot[key] > 0
                    for key in ("width", "height")
                )
                or screenshot.get("mime_type")
                not in {"image/jpeg", "image/png", "image/webp"}
            ):
                raise ValueError("function_transfer_state_screenshot_invalid")
        return _copy_value(value)
    if fields == {"pixels", "forest", "ui_elements", "auxiliaries"}:
        if not isinstance(value.get("ui_elements"), list):
            raise ValueError("function_transfer_state_ui_elements_invalid")
        if value.get("auxiliaries") is not None and not isinstance(
            value.get("auxiliaries"), dict
        ):
            raise ValueError("function_transfer_state_auxiliaries_invalid")
        return _copy_value(value)
    raise ValueError("function_transfer_state_runlog_schema_required")


def _transfer_state_as_runlog_observation(value: dict[str, Any]) -> dict[str, Any]:
    xml = str(value.get("xml") or "")
    if not xml:
        raise ValueError("function_transfer_state_xml_required")
    screenshot_path = str(value.get("screenshot_path") or "").strip()
    display = value.get("display") if isinstance(value.get("display"), dict) else {}
    screenshot = None
    if screenshot_path:
        suffix = Path(screenshot_path).suffix.casefold()
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(suffix, "image/png")
        screenshot = {
            "path": screenshot_path,
            "width": int(display.get("width") or 1),
            "height": int(display.get("height") or 1),
            "mime_type": mime_type,
        }
    return {"screenshot": screenshot, "xml": xml}


def parse_function_artifact(value: dict[str, Any]) -> Function:
    if not isinstance(value, dict):
        raise ValueError("function_artifact_must_be_object")
    missing = sorted(_REQUIRED_TOP_LEVEL_FIELDS - set(value))
    if missing:
        raise ValueError(f"function_artifact_missing_fields:{','.join(missing)}")
    unknown = sorted(set(value) - _TOP_LEVEL_FIELDS)
    if unknown:
        raise ValueError(f"function_artifact_unknown_fields:{','.join(unknown)}")
    if value.get("schema_version") != FUNCTION_ARTIFACT_VERSION:
        raise ValueError("unsupported_function_artifact_version")
    raw_transfer_states = value.get("transfer_states")
    if not isinstance(raw_transfer_states, dict):
        raise ValueError("function_transfer_states_must_be_object")
    transfer_states = {
        str(state_id): _canonical_function_observation(observation)
        for state_id, observation in raw_transfer_states.items()
        if str(state_id).strip()
    }
    if len(transfer_states) != len(raw_transfer_states):
        raise ValueError("function_transfer_state_id_required")
    raw_steps = value.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("function_steps_required")
    canonical_steps: list[dict[str, Any]] = []
    for index, step in enumerate(raw_steps):
        if not isinstance(step, dict) or set(step) != {
            "step_index",
            "transfer_state_ids",
            "action",
        }:
            raise ValueError("function_step_contract_invalid")
        if step.get("step_index") != index:
            raise ValueError("function_step_index_invalid")
        state_ids = step.get("transfer_state_ids")
        if (
            not isinstance(state_ids, list)
            or not state_ids
            or any(not isinstance(state_id, str) or not state_id.strip() for state_id in state_ids)
            or len(state_ids) != len(set(state_ids))
        ):
            raise ValueError("function_step_transfer_state_ids_invalid")
        missing_state_ids = [state_id for state_id in state_ids if state_id not in transfer_states]
        if missing_state_ids:
            raise ValueError(
                "function_transfer_states_missing:" + ",".join(missing_state_ids)
            )
        action = step.get("action")
        if not isinstance(action, dict) or set(action) != {"tool", "args"}:
            raise ValueError("function_action_contract_invalid")
        if not str(action.get("tool") or "").strip():
            raise ValueError("function_action_tool_required")
        if not isinstance(action.get("args"), dict):
            raise ValueError("function_action_args_must_be_object")
        if "target" in action["args"]:
            raise ValueError(f"function_action_target_forbidden:{index}")
        canonical_steps.append(
            {
                "step_index": index,
                "transfer_state_ids": list(state_ids),
                "action": canonicalize_action(action, replayable_only=True),
            }
        )
    canonical_value = dict(value)
    canonical_value["transfer_states"] = transfer_states
    canonical_value["steps"] = canonical_steps
    canonical_value["checker_rules"] = _canonical_checker_rules(
        value.get("checker_rules")
    )
    canonical_value["model_handoffs"] = _canonical_model_handoffs(
        value.get("model_handoffs"),
        step_count=len(canonical_steps),
    )
    function = Function.from_dict(canonical_value)
    validate_function_artifact(function)
    return function


def validate_function_artifact(function: Function) -> None:
    if function.schema_version != FUNCTION_ARTIFACT_VERSION:
        raise ValueError("unsupported_function_artifact_version")
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", function.function_id) is None:
        raise ValueError("function_id_invalid")
    builtin_tool_names = {
        str(tool.get("name") or "").strip()
        for tool in load_canonical_action_schema().get("tools") or ()
        if isinstance(tool, dict)
    }
    if function.id in builtin_tool_names:
        raise ValueError(f"tool_name_reserved:{function.id}")
    if not function.name.strip():
        raise ValueError("function_name_required")
    if not function.description.strip():
        raise ValueError("function_description_required")
    if not function.steps:
        raise ValueError("function_steps_required")
    _canonical_model_handoffs(
        list(function.model_handoffs),
        step_count=len(function.steps),
    )
    for step in function.steps:
        if (
            canonicalize_action(step.action.to_dict(), replayable_only=True)
            != step.action.to_dict()
        ):
            raise ValueError("function_action_not_canonical")
    schema = function.input_schema
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError("function_parameters_must_be_object_schema")
    unknown_schema_fields = sorted(
        set(schema) - {"type", "properties", "required", "additionalProperties"}
    )
    if unknown_schema_fields:
        raise ValueError(
            "function_parameter_schema_unknown_fields:"
            + ",".join(unknown_schema_fields)
        )
    if not isinstance(schema.get("properties"), dict):
        raise ValueError("function_parameter_properties_required")
    if schema.get("additionalProperties") is not False:
        raise ValueError("function_parameters_must_forbid_additional_properties")
    required = schema.get("required") or []
    if not isinstance(required, list) or not all(
        isinstance(name, str) for name in required
    ):
        raise ValueError("function_parameter_required_invalid")
    unknown_required = sorted(set(required) - set(schema["properties"]))
    if unknown_required:
        raise ValueError(
            f"function_parameter_required_unknown:{','.join(unknown_required)}"
        )
    _validate_bindings(function)


def bind_function(function: Function, arguments: dict[str, Any]) -> Function:
    validate_function_artifact(function)
    validate_arguments(function.input_schema, arguments)
    steps = [
        FunctionStep(
            step_index=step.step_index,
            transfer_state_ids=step.transfer_state_ids,
            action=Action(step.action.tool, _copy_value(step.action.args)),
        )
        for step in function.steps
    ]
    source_root = {"arguments": arguments}
    for binding in function.bindings:
        source_match = _SOURCE_PATH.fullmatch(binding["source"])
        target_match = _TARGET_PATH.fullmatch(binding["target"])
        if source_match is None or target_match is None:
            raise ValueError("function_binding_path_invalid")
        value = _read_path(
            source_root, _tokens(".arguments" + source_match.group("tail"))
        )
        action_index = int(target_match.group("action_index"))
        params = _copy_value(steps[action_index].action.args)
        _write_path(params, _tokens(target_match.group("tail")), _copy_value(value))
        steps[action_index] = replace(
            steps[action_index],
            action=Action(steps[action_index].action.tool, params),
        )
    return replace(function, steps=tuple(steps))


def validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> None:
    if not isinstance(arguments, dict):
        raise ValueError("function_arguments_must_be_object")
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(properties, dict):
        raise ValueError("function_parameter_properties_required")
    required = [str(name) for name in schema.get("required") or ()]
    missing = [name for name in required if name not in arguments]
    if missing:
        raise ValueError(f"function_arguments_invalid:missing:{','.join(missing)}")
    if schema.get("additionalProperties") is False:
        unknown = sorted(set(arguments) - set(properties))
        if unknown:
            raise ValueError(f"function_arguments_invalid:unknown:{','.join(unknown)}")
    for name, value in arguments.items():
        definition = properties.get(name)
        if not isinstance(definition, dict):
            continue
        expected_type = str(definition.get("type") or "")
        if expected_type and not _matches_json_type(value, expected_type):
            raise ValueError(f"function_arguments_invalid:type:{name}")
        if (
            isinstance(value, str)
            and "minLength" in definition
            and len(value.strip()) < int(definition["minLength"])
        ):
            raise ValueError(f"function_arguments_invalid:minLength:{name}")
        enum_values = definition.get("enum")
        if isinstance(enum_values, list) and value not in enum_values:
            raise ValueError(f"function_arguments_invalid:enum:{name}")


def _validate_bindings(function: Function) -> None:
    properties = function.input_schema["properties"]
    targets: set[str] = set()
    bound_properties: set[str] = set()
    for binding in function.bindings:
        if set(binding) != {"source", "target"}:
            raise ValueError("function_binding_contract_invalid")
        source = str(binding.get("source") or "")
        target = str(binding.get("target") or "")
        source_match = _SOURCE_PATH.fullmatch(source)
        target_match = _TARGET_PATH.fullmatch(target)
        if source_match is None or target_match is None:
            raise ValueError(f"function_binding_path_invalid:{source}->{target}")
        source_tokens = _tokens(".arguments" + source_match.group("tail"))
        property_name = source_tokens[1] if len(source_tokens) > 1 else ""
        if property_name not in properties:
            raise ValueError(f"function_binding_source_unknown:{source}")
        bound_properties.add(str(property_name))
        action_index = int(target_match.group("action_index"))
        if action_index not in range(len(function.steps)):
            raise ValueError(f"function_binding_target_invalid:{target}")
        if target in targets:
            raise ValueError(f"function_binding_target_duplicate:{target}")
        targets.add(target)
        try:
            _read_path(
                function.steps[action_index].action.args,
                _tokens(target_match.group("tail")),
            )
        except (IndexError, KeyError, TypeError) as error:
            raise ValueError(f"function_binding_target_missing:{target}") from error
    unbound = sorted(
        set(function.input_schema.get("required") or ()) - bound_properties
    )
    if unbound:
        raise ValueError(f"function_required_parameters_unbound:{','.join(unbound)}")


def _canonical_checker_rules(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("function_checker_rules_must_be_array")
    return [validate_checker_rule(rule) for rule in value]


def _canonical_model_handoffs(
    value: Any,
    *,
    step_count: int,
) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("function_model_handoffs_must_be_array")
    handoffs: list[dict[str, Any]] = []
    previous_step_index = -1
    for item in value:
        if not isinstance(item, dict) or set(item) != {"step_index", "reason"}:
            raise ValueError("function_model_handoff_contract_invalid")
        step_index = item.get("step_index")
        if isinstance(step_index, bool) or not isinstance(step_index, int):
            raise ValueError("function_model_handoff_step_index_invalid")
        if step_index < 0 or step_index >= step_count:
            raise ValueError("function_model_handoff_step_index_invalid")
        if step_index <= previous_step_index:
            raise ValueError("function_model_handoff_step_index_not_unique_sorted")
        reason = str(item.get("reason") or "").strip()
        if not reason or len(reason) > 500:
            raise ValueError("function_model_handoff_reason_invalid")
        handoffs.append({"step_index": step_index, "reason": reason})
        previous_step_index = step_index
    return handoffs


def _tokens(tail: str) -> list[str | int]:
    return [name if name else int(index) for name, index in _PATH_TOKEN.findall(tail)]


def _read_path(value: Any, tokens: list[str | int]) -> Any:
    current = value
    for token in tokens:
        if isinstance(token, str) and isinstance(current, dict):
            if token not in current:
                raise KeyError(token)
            current = current[token]
        elif isinstance(token, int) and isinstance(current, list):
            current = current[token]
        else:
            raise TypeError("path_type_mismatch")
    return current


def _write_path(value: Any, tokens: list[str | int], replacement: Any) -> None:
    if not tokens:
        raise ValueError("function_binding_target_root_forbidden")
    current = value
    for token in tokens[:-1]:
        if isinstance(token, str) and isinstance(current, dict):
            if token not in current:
                raise KeyError(token)
            current = current[token]
        elif isinstance(token, int) and isinstance(current, list):
            current = current[token]
        else:
            raise TypeError("path_type_mismatch")
    final = tokens[-1]
    if isinstance(final, str) and isinstance(current, dict) and final in current:
        current[final] = replacement
        return
    if isinstance(final, int) and isinstance(current, list):
        current[final] = replacement
        return
    raise KeyError(final)


def _copy_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _copy_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    return value


def _matches_json_type(value: Any, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "null":
        return value is None
    return False


class FunctionStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.functions: dict[str, Function] = {}
        self.source_calls: list[dict[str, Any]] = []
        self.load_errors: dict[str, str] = {}
        self._load()

    def list_functions(
        self,
        *,
        offset: int = 0,
        limit: int = 100,
        include_hidden: bool = True,
    ) -> list[Function]:
        start = max(0, int(offset))
        end = start + max(1, min(int(limit), 500))
        functions = (
            self.functions.values()
            if include_hidden
            else (item for item in self.functions.values() if item.agent_visible)
        )
        return sorted(functions, key=lambda item: item.id)[start:end]

    def get_function(self, function_id: str) -> Function | None:
        return self.functions.get(str(function_id or "").strip())

    def delete_function(self, function_id: str) -> bool:
        normalized = str(function_id or "").strip()
        if normalized not in self.functions:
            return False
        del self.functions[normalized]
        self.load_errors.clear()
        self.source_calls = [
            call
            for call in self.source_calls
            if call["function_id"] != normalized
        ]
        _write_store(self.path, self.functions, self.source_calls)
        return True

    def clear_functions(self) -> int:
        deleted = len(self.functions)
        self.functions.clear()
        self.source_calls.clear()
        self.load_errors.clear()
        _write_store(self.path, self.functions, self.source_calls)
        return deleted

    def reload(self) -> None:
        self.functions = {}
        self.source_calls = []
        self.load_errors = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != STORE_VERSION:
            raise ValueError("unsupported_store_version")
        raw_functions = payload.get("functions")
        if not isinstance(raw_functions, dict):
            raise ValueError("function_store_functions_must_be_object")
        raw_source_calls = payload.get("source_calls") or []
        if not isinstance(raw_source_calls, list) or any(
            not isinstance(call, dict)
            or set(call) != {"function_id", "arguments"}
            or not str(call.get("function_id") or "").strip()
            or not isinstance(call.get("arguments"), dict)
            for call in raw_source_calls
        ):
            raise ValueError("function_store_source_calls_invalid")
        loaded: dict[str, Function] = {}
        load_errors: dict[str, str] = {}
        for key, value in raw_functions.items():
            try:
                function = parse_function_artifact(value)
                if str(key) != function.id:
                    raise ValueError("function_store_key_mismatch")
            except (TypeError, ValueError) as error:
                load_errors[str(key)] = str(error) or type(error).__name__
                continue
            loaded[function.id] = function
        self.functions = loaded
        self.source_calls = [
            {
                "function_id": str(call["function_id"]),
                "arguments": _copy_value(call["arguments"]),
            }
            for call in raw_source_calls
            if str(call["function_id"]) in loaded
        ]
        self.load_errors = load_errors

def _write_store(
    path: Path,
    functions: dict[str, Function],
    source_calls: Iterable[dict[str, Any]] = (),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": STORE_VERSION,
        "functions": {
            key: value.to_dict() for key, value in sorted(functions.items())
        },
        "source_calls": list(source_calls),
    }
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_function_store(
    path: str | Path,
    functions: Iterable[Function | dict[str, Any]],
    source_calls: Iterable[dict[str, Any]] = (),
) -> Path:
    """Validate and write one current-version Function Store."""

    parsed: dict[str, Function] = {}
    for value in functions:
        function = (
            value if isinstance(value, Function) else parse_function_artifact(value)
        )
        validate_function_artifact(function)
        if function.id in parsed:
            raise ValueError(f"function_store_duplicate_function:{function.id}")
        parsed[function.id] = function
    normalized_calls: list[dict[str, Any]] = []
    for call in source_calls:
        if (
            not isinstance(call, dict)
            or set(call) != {"function_id", "arguments"}
            or not str(call.get("function_id") or "").strip()
            or not isinstance(call.get("arguments"), dict)
        ):
            raise ValueError("function_store_source_calls_invalid")
        if str(call["function_id"]) not in parsed:
            raise ValueError("function_store_source_call_function_missing")
        normalized_calls.append(
            {
                "function_id": str(call["function_id"]),
                "arguments": _copy_value(call["arguments"]),
            }
        )
    destination = Path(path).expanduser().resolve()
    _write_store(destination, parsed, normalized_calls)
    return destination


def save_function(
    run_log: str | Path | dict[str, Any] | None,
    store_path: str | Path,
    *,
    functions: list[dict[str, Any]] | None = None,
    arguments: dict[str, Any] | None = None,
    enhance: bool = False,
    complete_json: Callable[[str, dict[str, Any]], str] | None = None,
    instruction: str = "",
    authoring_trace: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write v3 Functions; RunLogs are optional authoring input only."""
    destination = Path(store_path).expanduser().resolve()
    if functions is not None and not isinstance(functions, list):
        raise ValueError("functions_must_be_array")
    if arguments is not None and not isinstance(arguments, dict):
        raise ValueError("function_arguments_invalid")
    if run_log is None:
        if enhance:
            raise ValueError("function_enhancement_run_log_required")
        if not functions:
            raise ValueError("function_required")
        return _write_function_artifacts(
            destination,
            functions,
            arguments_by_function=dict(arguments or {}),
            enhanced=False,
        )

    evidence_root: Path | None = None
    source_run_log_path: Path | None = None
    if isinstance(run_log, dict):
        raw = dict(run_log)
    else:
        run_log_path = Path(run_log).expanduser().resolve()
        source_run_log_path = run_log_path
        value = json.loads(run_log_path.read_text())
        if not isinstance(value, dict):
            raise ValueError("source_runlog_must_be_object")
        raw = value
        evidence_root = run_log_path.parent
    if (
        raw.get("schema_version") == "omniflow.run_log.v1"
        and isinstance(raw.get("steps"), list)
        and any(
            isinstance(step, dict) and "before_state_id" in step
            for step in raw["steps"]
        )
    ):
        if source_run_log_path is None or evidence_root is None:
            raise ValueError("legacy_run_log_path_required")
        raw = json.loads(json.dumps(raw, ensure_ascii=False))
        for step in raw["steps"]:
            if isinstance(step, dict):
                step.pop("before_state_id", None)
                step.pop("after_state_id", None)
    raw_steps = raw.get("steps")
    compact_trace = (
        raw.get("schema_version") == "omniflow.canonical_run_log.v1"
        and isinstance(raw_steps, list)
        and bool(raw_steps)
        and all(
            isinstance(step, dict)
            and "before_state_id" in step
            and isinstance(step.get("action"), dict)
            and "tool" in step["action"]
            for step in raw_steps
        )
    )
    if raw.get("schema_version") != "omniflow.run_log.v1" and not compact_trace:
        if evidence_root is None:
            raise ValueError("legacy_run_log_path_required")
        from src.integrations.runlog import adapt_source_run_log

        legacy_payload = (
            raw.get("payload")
            if isinstance(raw.get("payload"), dict)
            else raw
        )
        if isinstance(legacy_payload.get("run_log"), dict):
            legacy_payload = legacy_payload["run_log"]
        extra = raw.get("extra") if isinstance(raw.get("extra"), dict) else {}
        raw = adapt_source_run_log(
            raw,
            task_name=str(
                legacy_payload.get("androidworld_task")
                or extra.get("androidworld_task")
                or evidence_root.name
            ),
            task_parameters=(
                dict(legacy_payload.get("androidworld_params"))
                if isinstance(legacy_payload.get("androidworld_params"), dict)
                else dict(extra.get("androidworld_params"))
                if isinstance(extra.get("androidworld_params"), dict)
                else {}
            ),
            seed=(
                int(extra["seed"])
                if isinstance(extra.get("seed"), int)
                and not isinstance(extra.get("seed"), bool)
                else None
            ),
            source_path=evidence_root / run_log_path.name,
            screenshot_roots=(evidence_root, evidence_root.parent),
            require_screenshots=True,
        )
        evidence_root = None
    payload, source_catalog = import_run_log_evidence(
        raw,
        evidence_root=evidence_root,
    )
    goal = str(payload.get("goal") or "").strip()
    if not goal:
        raise ValueError("successful_source_goal_required")

    steps: list[dict[str, Any]] = []
    for step in payload["steps"]:
        if not isinstance(step, dict):
            continue
        metadata = step.get("metadata") if isinstance(step.get("metadata"), dict) else {}
        result = step.get("result") if isinstance(step.get("result"), dict) else {}
        if result.get("success") is not True:
            continue
        observation = step.get("observation")
        if isinstance(observation, dict):
            before_state_id = state_id(observation)
            next_observation = step.get("next_observation")
            after_state_id = state_id(
                next_observation
                if isinstance(next_observation, dict)
                else observation
            )
            action_type = str(step.get("action", {}).get("action_type") or "")
            if action_type in {"answer", "status", "unknown"}:
                continue
            projected_actions = project_androidworld_step_actions(step)
            source_page = _source_page_summary(observation)
            after_page = _source_page_summary(
                next_observation
                if isinstance(next_observation, dict)
                else observation
            )
        else:
            before_state_id = str(step.get("before_state_id") or "").strip()
            after_state_id = str(step.get("after_state_id") or "").strip()
            try:
                projected_actions = [
                    canonicalize_action(
                        step.get("action"),
                        replayable_only=True,
                        allow_non_action=True,
                    )
                ]
            except ValueError as error:
                if str(error).startswith("canonical_action_tool_not_replayable:"):
                    continue
                raise
            before_state = source_catalog["states"].get(before_state_id)
            after_state = source_catalog["states"].get(after_state_id)
            source_page = _source_transfer_state_summary(before_state)
            after_page = _source_transfer_state_summary(after_state)
        action_metadata = {
            key: metadata[key]
            for key in ("summary", "thinking", "action_description")
            if str(metadata.get(key) or "").strip()
        }
        action_metadata["source_page"] = source_page
        action_metadata["after_page"] = after_page
        for action in projected_actions:
            step_metadata = dict(action_metadata)
            semantic_target = (
                _projected_semantic_target(action, observation)
                if isinstance(observation, dict)
                else _transfer_state_semantic_target(action, before_state)
            )
            if semantic_target:
                step_metadata["semantic_target"] = semantic_target
            steps.append(
                {
                    "step_index": len(steps),
                    "before_state_id": before_state_id,
                    "action": action,
                    "result": {"success": True},
                    "after_state_id": after_state_id,
                    "metadata": step_metadata,
                }
            )
    if not steps:
        raise ValueError("successful_source_actions_required")
    facts = {
        "schema_version": "omniflow.function-compilation-facts.v1",
        "run_id": str(payload.get("run_id") or "successful-source"),
        "task_name": str(
            payload.get("task_name") or payload.get("task") or ""
        ).strip(),
        "goal": goal,
        "status": "succeeded",
        "success": True,
        "steps": steps,
        "transfer_states": {
            str(source_state_id): _transfer_state_as_runlog_observation(
                _normalize_source_state(value, str(source_state_id))
            )
            for source_state_id, value in source_catalog["states"].items()
        },
    }
    if enhance and complete_json is None:
        raise ValueError("function_enhancer_required")
    if enhance:
        raw_functions, generated_arguments = _author_functions(
            facts,
            complete_json,
            instruction=instruction,
            existing_functions=functions or [],
            authoring_trace=authoring_trace,
        )
        arguments_by_function = {
            **generated_arguments,
            **dict(arguments or {}),
        }
    else:
        if functions:
            raw_functions = json.loads(json.dumps(functions, ensure_ascii=False))
            arguments_by_function = dict(arguments or {})
        else:
            raw_function, generated_arguments = _compile_deterministic_function(facts)
            raw_functions = [raw_function]
            arguments_by_function = generated_arguments
            if arguments:
                arguments_by_function.update(dict(arguments))
    return _write_function_artifacts(
        destination,
        raw_functions,
        arguments_by_function=arguments_by_function,
        enhanced=enhance,
    )


def _write_function_artifacts(
    destination: Path,
    raw_functions: list[dict[str, Any]],
    *,
    arguments_by_function: dict[str, Any],
    enhanced: bool,
) -> dict[str, Any]:
    parsed_functions = [parse_function_artifact(value) for value in raw_functions]
    if not parsed_functions:
        raise ValueError("function_required")
    function_ids = [function.id for function in parsed_functions]
    if len(function_ids) != len(set(function_ids)):
        raise ValueError("duplicate_function_id")
    if set(arguments_by_function) - set(function_ids):
        raise ValueError("function_arguments_unknown_function")
    saved_source_calls: list[dict[str, Any]] = []
    for function in parsed_functions:
        raw_arguments = arguments_by_function.get(function.id, {})
        if isinstance(raw_arguments, list):
            raise ValueError("function_single_source_call_required")
        calls = [raw_arguments]
        if not isinstance(raw_arguments, dict):
            raise ValueError("function_arguments_invalid")
        for arguments in calls:
            bind_function(function, arguments)
            saved_source_calls.append(
                {
                    "function_id": function.id,
                    "arguments": _copy_value(arguments),
                }
            )
    store = FunctionStore(destination)
    store.functions = {function.id: function for function in parsed_functions}
    store.source_calls = list(saved_source_calls)
    _write_store(destination, store.functions, store.source_calls)
    transfer_state_ids = {
        state_id
        for function in parsed_functions
        for state_id in function.transfer_states
    }
    report = {
        "schema_version": "omniflow.function-save.v2",
        "success": True,
        "store_path": str(destination),
        "transfer_state_count": len(transfer_state_ids),
        "function_ids": function_ids,
        "function_count": len(function_ids),
        "source_arguments": json.loads(
            json.dumps(arguments_by_function, ensure_ascii=False)
        ),
        "enhanced": bool(enhanced),
    }
    return report


def _normalize_source_state(value: Any, expected_state_id: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"function_source_state_invalid:{expected_state_id}")
    extra = value.get("extra") if isinstance(value.get("extra"), dict) else {}
    state_id = str(
        value.get("state_id") or extra.get("state_id") or expected_state_id
    ).strip()
    if state_id != expected_state_id:
        raise ValueError(f"function_source_state_id_mismatch:{expected_state_id}")
    state: dict[str, Any] = {"state_id": state_id}
    aliases = {
        "xml": ("xml", "page", "observation_xml"),
        "package_name": ("package_name", "packageName"),
        "activity_name": ("activity_name", "activityName"),
        "screenshot_path": ("screenshot_path",),
    }
    for output, names in aliases.items():
        item = next(
            (
                source[name]
                for source in (value, extra)
                for name in names
                if source.get(name) is not None
            ),
            None,
        )
        if item is not None:
            if not isinstance(item, str):
                raise ValueError(
                    f"function_source_state_{output}_invalid:{state_id}"
                )
            state[output] = item
    display = value.get("display") or extra.get("display")
    if not isinstance(display, dict):
        width = value.get("width") or value.get("display_width")
        height = value.get("height") or value.get("display_height")
        if width is None:
            width = extra.get("width") or extra.get("display_width")
        if height is None:
            height = extra.get("height") or extra.get("display_height")
        if width is not None or height is not None:
            display = {"width": width, "height": height}
    if display is not None:
        if not isinstance(display, dict) or set(display) != {"width", "height"}:
            raise ValueError(f"function_source_state_display_invalid:{state_id}")
        try:
            width = int(display["width"])
            height = int(display["height"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"function_source_state_display_invalid:{state_id}"
            ) from error
        if width <= 0 or height <= 0:
            raise ValueError(f"function_source_state_display_invalid:{state_id}")
        state["display"] = {"width": width, "height": height}
    if "display" not in state and state.get("xml"):
        try:
            root = ET.fromstring(state["xml"])
            width = int(root.attrib.get("width") or 0)
            height = int(root.attrib.get("height") or 0)
        except (ET.ParseError, TypeError, ValueError):
            width = height = 0
        if width > 0 and height > 0:
            state["display"] = {"width": width, "height": height}
    return state


def _author_functions(
    facts: dict[str, Any],
    complete_json: Callable[[str, dict[str, Any]], str],
    *,
    instruction: str | None,
    existing_functions: list[dict[str, Any]],
    authoring_trace: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Author independent v3 Functions from the observations available to the writer."""

    transfer_states = facts.get("transfer_states")
    if not isinstance(transfer_states, dict) or not transfer_states:
        raise ValueError("function_transfer_states_required")
    available_state_ids = tuple(str(value) for value in transfer_states)
    prompt_payload = {
        "goal": str(facts.get("goal") or "").strip(),
        "instruction": str(instruction or "").strip(),
        "available_transfer_state_ids": list(available_state_ids),
        "recorded_examples": [
            {
                "transfer_state_id": str(step.get("before_state_id") or ""),
                "action": _copy_value(step.get("action")),
                "metadata": _copy_value(step.get("metadata") or {}),
            }
            for step in facts.get("steps") or ()
            if isinstance(step, dict)
        ],
        "existing_function_hints": [
            {
                "function_id": str(value.get("function_id") or ""),
                "name": str(value.get("name") or ""),
                "description": str(value.get("description") or ""),
            }
            for value in existing_functions
            if isinstance(value, dict)
        ],
    }
    prompt = (
        "Author one or more independent OmniFlow Function v3 assets. "
        "The recorded actions are examples, not a required sequence: you may omit, "
        "reorder, or replace actions. Each output step must reference one or more IDs "
        "from available_transfer_state_ids. Transfer states are observations used by "
        "OmniTransfer; do not invent IDs. Return only the author_functions tool call.\n\n"
        "Authoring input:\n"
        + json.dumps(prompt_payload, ensure_ascii=False)
    )
    raw = complete_json(prompt, function_authoring_tool())
    authored = _json_object(raw)
    if set(authored) != {"functions"}:
        raise ValueError("function_authoring_result_contract_invalid")
    drafts = authored.get("functions")
    if not isinstance(drafts, list) or not drafts:
        raise ValueError("function_authoring_functions_required")

    available = set(available_state_ids)
    functions: list[dict[str, Any]] = []
    arguments_by_function: dict[str, dict[str, Any]] = {}
    for draft in drafts:
        if not isinstance(draft, dict) or set(draft) != {
            "function_id",
            "name",
            "description",
            "steps",
        }:
            raise ValueError("function_authoring_function_contract_invalid")
        function_id = str(draft.get("function_id") or "").strip()
        name = str(draft.get("name") or "").strip()
        description = str(draft.get("description") or "").strip()
        if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", function_id) is None:
            raise ValueError("function_id_invalid")
        if not name or len(name) > 120:
            raise ValueError("function_name_invalid")
        if not description:
            raise ValueError("function_description_required")
        draft_steps = draft.get("steps")
        if not isinstance(draft_steps, list) or not draft_steps:
            raise ValueError("function_steps_required")

        used_state_ids: set[str] = set()
        steps: list[dict[str, Any]] = []
        for step_index, step in enumerate(draft_steps):
            if not isinstance(step, dict) or set(step) != {
                "transfer_state_ids",
                "action",
            }:
                raise ValueError("function_authoring_step_contract_invalid")
            raw_state_ids = step.get("transfer_state_ids")
            if (
                not isinstance(raw_state_ids, list)
                or not raw_state_ids
                or any(
                    not isinstance(state_id, str) or not state_id.strip()
                    for state_id in raw_state_ids
                )
            ):
                raise ValueError("function_step_transfer_state_ids_invalid")
            state_ids = [state_id.strip() for state_id in raw_state_ids]
            if len(state_ids) != len(set(state_ids)):
                raise ValueError("function_step_transfer_state_ids_invalid")
            missing = [state_id for state_id in state_ids if state_id not in available]
            if missing:
                raise ValueError(
                    "function_transfer_states_missing:" + ",".join(missing)
                )
            used_state_ids.update(state_ids)
            steps.append(
                {
                    "step_index": step_index,
                    "transfer_state_ids": state_ids,
                    "action": canonicalize_action(
                        step.get("action"), replayable_only=True
                    ),
                }
            )

        function = {
            "schema_version": FUNCTION_ARTIFACT_VERSION,
            "function_id": function_id,
            "name": name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "bindings": [],
            "transfer_states": {
                state_id: _copy_value(transfer_states[state_id])
                for state_id in available_state_ids
                if state_id in used_state_ids
            },
            "steps": steps,
            "checker_rules": [],
            "agent_visible": True,
        }
        parse_function_artifact(function)
        functions.append(function)
        arguments_by_function[function_id] = {}

    if len(arguments_by_function) != len(functions):
        raise ValueError("duplicate_function_id")
    if authoring_trace is not None:
        authoring_trace.append(
            {
                "tool": "author_functions",
                "function_ids": list(arguments_by_function),
                "function_count": len(functions),
            }
        )
    return functions, arguments_by_function


def _compile_deterministic_function(
    facts: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the minimal default v3 Function when model authoring is disabled."""

    task_name = str(facts.get("task_name") or "").strip()
    label = task_name or str(facts.get("goal") or "successful_task").strip()
    normalized = re.sub(r"[^a-z0-9]+", "_", label.casefold()).strip("_")
    if not normalized:
        normalized = "successful_task"
    if not normalized[0].isalpha():
        normalized = f"task_{normalized}"
    function_id = f"replay_{normalized[:56]}"
    display_label = task_name or "successful task"
    metadata = {
        "function_id": function_id,
        "name": f"Replay {display_label}"[:120],
        "description": str(facts.get("goal") or display_label).strip(),
    }
    function, source_arguments = _compile_draft_function(
        facts,
        metadata=metadata,
        source_indices=tuple(range(len(facts["steps"]))),
        action_edits=[],
        action_overrides=[],
        parameter_bindings=[],
        checker_indices=[],
    )
    return function, {function_id: source_arguments}


def _compile_draft_function(
    facts: dict[str, Any],
    *,
    metadata: dict[str, str],
    source_indices: tuple[int, ...],
    action_edits: list[dict[str, Any]],
    action_overrides: list[dict[str, Any]],
    parameter_bindings: list[dict[str, Any]],
    checker_indices: list[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    formal_indices = [index for index in source_indices if index not in checker_indices]
    if not formal_indices:
        raise ValueError("function_steps_required")
    local_index = {
        source_index: index for index, source_index in enumerate(formal_indices)
    }
    steps = [
        {
            "step_index": local_index[source_index],
            "transfer_state_ids": [
                facts["steps"][source_index]["before_state_id"]
            ],
            "action": _draft_action(
                facts,
                action_edits,
                action_overrides=action_overrides,
                function_id=metadata["function_id"],
                step_index=source_index,
            ),
        }
        for source_index in formal_indices
    ]
    properties: dict[str, dict[str, str]] = {}
    source_arguments: dict[str, Any] = {}
    bindings: list[dict[str, str]] = []
    for binding in parameter_bindings:
        source_index = binding["step_index"]
        name = binding["name"]
        path = binding["argument_path"]
        value = _read_path(
            _draft_action(
            facts,
            action_edits,
            action_overrides=action_overrides,
            function_id=metadata["function_id"],
                step_index=source_index,
            )["args"],
            _tokens("." + path),
        )
        value_type = _json_type(value)
        assert value_type is not None
        if name in source_arguments and source_arguments[name] != value:
            raise ValueError(f"function_parameter_source_value_conflict:{name}")
        if name in properties and properties[name]["type"] != value_type:
            raise ValueError(f"function_parameter_type_conflict:{name}")
        properties.setdefault(
            name,
            {"type": value_type, "description": binding["description"]},
        )
        source_arguments[name] = _copy_value(value)
        target = f"$.steps[{local_index[source_index]}].action.args.{path}"
        bindings.append({"source": f"$.arguments.{name}", "target": target})
        _write_path(
            steps[local_index[source_index]]["action"]["args"],
            _tokens("." + path),
            _empty_json_value(value_type),
        )
    checker_rules = [
        {
            "transfer_state_ids": [
                facts["steps"][index]["before_state_id"]
            ],
            "action": _copy_value(facts["steps"][index]["action"]),
        }
        for index in checker_indices
    ]
    return (
        {
            "schema_version": FUNCTION_ARTIFACT_VERSION,
            **metadata,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
            "bindings": bindings,
            "transfer_states": {
                state_id: _copy_value(facts["transfer_states"][state_id])
                for state_id in dict.fromkeys(
                    facts["steps"][index]["before_state_id"]
                    for index in (*formal_indices, *checker_indices)
                )
            },
            "steps": steps,
            "checker_rules": checker_rules,
            "agent_visible": True,
        },
        source_arguments,
    )


def _function_hints(functions: list[dict[str, Any]]) -> list[dict[str, str]]:
    hints: list[dict[str, str]] = []
    for value in functions:
        if not isinstance(value, dict):
            continue
        hint = {
            key: str(value.get(key) or "").strip()
            for key in ("function_id", "name", "description")
        }
        if hint["function_id"]:
            hints.append(hint)
    return hints


def _json_type(value: Any) -> str | None:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return None


def _empty_json_value(value_type: str) -> Any:
    return {
        "string": "",
        "integer": 0,
        "number": 0.0,
        "boolean": False,
        "array": [],
        "object": {},
    }[value_type]


def _source_page_summary(observation: dict[str, Any]) -> dict[str, Any]:
    auxiliaries = (
        observation.get("auxiliaries")
        if isinstance(observation.get("auxiliaries"), dict)
        else {}
    )
    labels: list[str] = []
    xml = observation_xml(observation)
    if xml:
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            root = None
        if root is not None:
            for node in root.iter():
                for field in ("text", "content-desc", "content_description"):
                    label = " ".join(str(node.attrib.get(field) or "").split())
                    if label and label not in labels:
                        labels.append(label[:160])
                        if len(labels) == 20:
                            break
                if len(labels) == 20:
                    break
    package = str(auxiliaries.get("package_name") or "")
    return {
        "package": package,
        "activity": str(auxiliaries.get("activity_name") or ""),
        "is_launcher": "launcher" in package.casefold(),
        "visible_labels": labels,
        "screenshot": (
            observation.get("pixels", {}).get("path")
            if isinstance(observation.get("pixels"), dict)
            else None
        ),
    }


def _source_transfer_state_summary(value: Any) -> dict[str, Any]:
    state = value if isinstance(value, dict) else {}
    labels: list[str] = []
    xml = str(state.get("xml") or "")
    if xml:
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            root = None
        if root is not None:
            for node in root.iter():
                for field in ("text", "content-desc", "content_description"):
                    label = " ".join(str(node.attrib.get(field) or "").split())
                    if label and label not in labels:
                        labels.append(label[:160])
                        if len(labels) == 20:
                            break
                if len(labels) == 20:
                    break
    package = str(state.get("package_name") or "")
    return {
        "package": package,
        "activity": str(state.get("activity_name") or ""),
        "is_launcher": "launcher" in package.casefold(),
        "visible_labels": labels,
        "screenshot": state.get("screenshot_path"),
    }


def _enhancement_step_actions(
    step: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    observation = step.get("observation")
    if isinstance(observation, dict):
        return state_id(observation), project_androidworld_step_actions(step)
    action = step.get("action")
    if not isinstance(action, dict):
        return "", []
    try:
        canonical = canonicalize_action(action, replayable_only=True)
    except (TypeError, ValueError):
        return "", []
    return str(step.get("before_state_id") or "").strip(), [canonical]




def _projected_semantic_target(
    action: dict[str, Any],
    observation: dict[str, Any],
) -> str:
    if action.get("tool") != "click":
        return ""
    args = action.get("args")
    display = observation_display(observation)
    if not isinstance(args, dict) or display is None:
        return ""
    try:
        x = float(args["x"]) / 1000.0 * display[0]
        y = float(args["y"]) / 1000.0 * display[1]
    except (KeyError, TypeError, ValueError):
        return ""
    return semantic_target_at_point(observation_xml(observation), x, y)


def _transfer_state_semantic_target(
    action: dict[str, Any],
    value: Any,
) -> str:
    if action.get("tool") != "click" or not isinstance(value, dict):
        return ""
    args = action.get("args")
    display = value.get("display")
    if not isinstance(args, dict) or not isinstance(display, dict):
        return ""
    try:
        width = float(display["width"])
        height = float(display["height"])
        x = float(args["x"]) / 1000.0 * width
        y = float(args["y"]) / 1000.0 * height
    except (KeyError, TypeError, ValueError):
        return ""
    if width <= 0 or height <= 0:
        return ""
    return semantic_target_at_point(str(value.get("xml") or ""), x, y)


def _json_object(raw: str) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("function_enhancement_json_missing")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("function_enhancement_json_invalid") from error
    if not isinstance(value, dict):
        raise ValueError("function_enhancement_json_invalid")
    return value
