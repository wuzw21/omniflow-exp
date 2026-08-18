from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable
import xml.etree.ElementTree as ET

from omniflow.core.model import Action, Function, FunctionStep
from omniflow.core.schemas import (
    canonicalize_action,
    load_canonical_action_schema,
    load_checker_rule_schema,
    load_function_schema,
)
from omniflow.core.trajectory import (
    observation_display,
    observation_xml,
    state_id,
)
from omniflow.runlog import import_run_log_evidence, project_androidworld_step_actions
from omniflow.runtime.checker import validate_checker_rule
from omniflow.runtime.semantic_grounding import semantic_target_at_point

FUNCTION_ARTIFACT_VERSION = "omniflow.function.v2"
STORE_VERSION = "omniflow.store.v2"

_TOP_LEVEL_FIELDS = {
    "schema_version",
    "function_id",
    "name",
    "description",
    "input_schema",
    "bindings",
    "steps",
    "checker_rules",
    "agent_visible",
}
_SOURCE_PATH = re.compile(
    r"^\$\.arguments(?P<tail>(?:\.[A-Za-z_][A-Za-z0-9_]*|\[\d+\])+)$"
)
_TARGET_PATH = re.compile(
    r"^\$\.steps\[(?P<action_index>\d+)]\.action\.args"
    r"(?P<tail>(?:\.[A-Za-z_][A-Za-z0-9_]*|\[\d+\])+)$"
)
_PATH_TOKEN = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)]")


def function_authoring_tool(
    *,
    stage: str,
    current_bundle: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return the complete bundle contract narrowed to one authoring stage."""

    if stage not in {"split", "parameters", "checkers"}:
        raise ValueError(f"function_authoring_stage_invalid:{stage}")
    if stage == "split" and current_bundle is not None:
        raise ValueError("function_authoring_split_bundle_forbidden")
    if stage != "split" and not isinstance(current_bundle, dict):
        raise ValueError(f"function_authoring_{stage}_bundle_required")

    function_schema = load_function_schema()
    definitions = function_schema.pop("$defs")
    action_schema = definitions["action"]
    step_schema = definitions["step"]
    step_schema["properties"]["action"] = action_schema
    checker_schema = load_checker_rule_schema()
    function_schema["properties"]["input_schema"] = definitions["input_schema"]
    function_schema["properties"]["bindings"]["items"] = definitions["binding"]
    function_schema["properties"]["steps"]["items"] = step_schema
    function_schema["properties"]["checker_rules"]["items"] = checker_schema
    for key in ("$schema", "$id", "title"):
        function_schema.pop(key, None)
        checker_schema.pop(key, None)
    functions_schema: dict[str, Any] = {
        "type": "array",
        "minItems": 1,
        "items": function_schema,
    }
    arguments_schema: dict[str, Any] = {
        "type": "object",
        "description": (
            "Map every function_id to one source argument object or a non-empty "
            "list of source argument objects."
        ),
        "additionalProperties": {
            "oneOf": [
                {"type": "object"},
                {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "object"},
                },
            ]
        },
    }
    if stage == "split":
        function_schema["properties"]["input_schema"] = {
            "const": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            }
        }
        function_schema["properties"]["bindings"] = {"const": []}
        function_schema["properties"]["checker_rules"] = {"const": []}
    else:
        assert current_bundle is not None
        current_functions = list(current_bundle.get("functions") or ())
        if not current_functions:
            raise ValueError(f"function_authoring_{stage}_functions_required")
        functions_schema = {
            "type": "array",
            "minItems": len(current_functions),
            "maxItems": len(current_functions),
            "items": {
                "oneOf": [
                    _stage_function_schema(function_schema, value, stage=stage)
                    for value in current_functions
                ]
            },
        }
        function_ids = [str(value["function_id"]) for value in current_functions]
        if stage == "checkers":
            arguments_schema = {"const": current_bundle.get("arguments") or {}}
        else:
            arguments_schema = {
                "type": "object",
                "additionalProperties": False,
                "required": function_ids,
                "properties": {
                    function_id: {
                        "oneOf": [
                            {"type": "object"},
                            {
                                "type": "array",
                                "minItems": 1,
                                "items": {"type": "object"},
                            },
                        ]
                    }
                    for function_id in function_ids
                },
            }
    return {
        "type": "function",
        "function": {
            "name": "submit_function_bundle",
            "description": (
                f"Return the complete Function bundle for save_function stage {stage}. "
                "Include one full-trajectory Function and every reusable semantic "
                "subsegment, with source arguments for every Function."
            ),
            "strict": False,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["functions", "arguments"],
                "properties": {
                    "functions": {
                        **functions_schema,
                    },
                    "arguments": arguments_schema,
                },
            },
        },
    }


def _stage_function_schema(
    base_schema: dict[str, Any],
    previous: dict[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    schema = json.loads(json.dumps(base_schema, ensure_ascii=False))
    properties = schema["properties"]
    immutable_fields = {
        "schema_version",
        "function_id",
        "name",
        "description",
        "agent_visible",
    }
    if stage == "checkers":
        immutable_fields.add("input_schema")
    for field in immutable_fields:
        properties[field] = {"const": previous[field]}
    previous_steps = list(previous["steps"])
    if stage == "parameters":
        properties["checker_rules"] = {"const": []}
        step_schemas = [
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["step_index", "source_state_id", "action"],
                "properties": {
                    "step_index": {"const": step["step_index"]},
                    "source_state_id": {"const": step["source_state_id"]},
                    "action": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["tool", "args"],
                        "properties": {
                            "tool": {"const": step["action"]["tool"]},
                            "args": _parameter_arguments_schema(
                                step["action"]["args"]
                            ),
                        },
                    },
                },
            }
            for step in previous_steps
        ]
        properties["steps"] = {
            "type": "array",
            "minItems": len(previous_steps),
            "maxItems": len(previous_steps),
            "items": {"oneOf": step_schemas},
        }
        return schema
    properties["steps"] = {
        "type": "array",
        "minItems": 1,
        "maxItems": len(previous_steps),
        "items": {
            "oneOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["step_index", "source_state_id", "action"],
                    "properties": {
                        "step_index": {"type": "integer", "minimum": 0},
                        "source_state_id": {"const": step["source_state_id"]},
                        "action": {"const": step["action"]},
                    },
                }
                for step in previous_steps
            ]
        },
    }
    properties["checker_rules"] = {
        "type": "array",
        "items": {
            "enum": [
                {
                    "source_state_id": step["source_state_id"],
                    "action": step["action"],
                }
                for step in previous_steps
            ]
        },
    }
    return schema


def _parameter_arguments_schema(arguments: dict[str, Any]) -> dict[str, Any]:
    json_types = {
        str: "string",
        bool: "boolean",
        int: "integer",
        float: "number",
        list: "array",
        dict: "object",
        type(None): "null",
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(arguments),
        "properties": {
            key: {"type": json_types[type(value)]}
            for key, value in arguments.items()
        },
    }


def parse_function_artifact(value: dict[str, Any]) -> Function:
    if not isinstance(value, dict):
        raise ValueError("function_artifact_must_be_object")
    unknown = sorted(set(value) - _TOP_LEVEL_FIELDS)
    if unknown:
        raise ValueError(f"function_artifact_unknown_fields:{','.join(unknown)}")
    if value.get("schema_version") != FUNCTION_ARTIFACT_VERSION:
        raise ValueError("unsupported_function_artifact_version")
    raw_steps = value.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("function_steps_required")
    canonical_steps: list[dict[str, Any]] = []
    for index, step in enumerate(raw_steps):
        if not isinstance(step, dict) or set(step) != {
            "step_index",
            "source_state_id",
            "action",
        }:
            raise ValueError("function_step_contract_invalid")
        if step.get("step_index") != index:
            raise ValueError("function_step_index_invalid")
        if not str(step.get("source_state_id") or "").strip():
            raise ValueError("function_step_source_state_id_required")
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
                "source_state_id": str(step["source_state_id"]),
                "action": canonicalize_action(action, replayable_only=True),
            }
        )
    canonical_value = dict(value)
    canonical_value["steps"] = canonical_steps
    canonical_value["checker_rules"] = _canonical_checker_rules(
        value.get("checker_rules")
    )
    function = Function.from_dict(canonical_value)
    validate_function_artifact(function)
    return function


def validate_function_artifact(function: Function) -> None:
    if function.schema_version != FUNCTION_ARTIFACT_VERSION:
        raise ValueError("unsupported_function_artifact_version")
    if not function.function_id.strip():
        raise ValueError("function_id_required")
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
    for step in function.steps:
        if (
            canonicalize_action(step.action.to_dict(), replayable_only=True)
            != step.action.to_dict()
        ):
            raise ValueError("function_action_not_canonical")
    formal_actions = {
        (
            step.source_state_id,
            json.dumps(
                step.action.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        for step in function.steps
    }
    for rule in function.checker_rules:
        checker = (
            str(rule.get("source_state_id") or ""),
            json.dumps(
                canonicalize_action(rule.get("action"), replayable_only=True),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        if checker in formal_actions:
            raise ValueError(
                f"function_checker_duplicates_formal_action:{function.id}"
            )
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
            source_state_id=step.source_state_id,
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


def save_function(
    run_log: str | Path | dict[str, Any],
    store_path: str | Path,
    *,
    functions: list[dict[str, Any]] | None = None,
    arguments: dict[str, Any] | None = None,
    enhance: bool = False,
    complete_json: Callable[[str, dict[str, Any]], str] | None = None,
    instruction: str = "",
) -> dict[str, Any]:
    """Create or replace RunLog-grounded Functions through the only writer."""
    from omniflow.transfer.runtime import (
        TRANSFER_STATE_CATALOG_FILENAME,
        TRANSFER_STATE_CATALOG_VERSION,
        load_transfer_state_catalog,
    )

    destination = Path(store_path).expanduser().resolve()
    root = destination.parent

    evidence_root: Path | None = None
    if isinstance(run_log, dict):
        raw = dict(run_log)
    else:
        run_log_path = Path(run_log).expanduser().resolve()
        value = json.loads(run_log_path.read_text())
        if not isinstance(value, dict):
            raise ValueError("source_runlog_must_be_object")
        raw = value
        evidence_root = run_log_path.parent
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
            source_page = _source_transfer_state_summary(
                source_catalog["states"].get(before_state_id)
            )
        action_metadata = {
            key: metadata[key]
            for key in ("summary", "thinking", "action_description")
            if str(metadata.get(key) or "").strip()
        }
        action_metadata["source_page"] = source_page
        for action in projected_actions:
            step_metadata = dict(action_metadata)
            semantic_target = (
                _projected_semantic_target(action, observation)
                if isinstance(observation, dict)
                else ""
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
        "goal": goal,
        "status": "succeeded",
        "success": True,
        "steps": steps,
    }
    if enhance and complete_json is None:
        raise ValueError("function_enhancer_required")
    if functions is not None and not isinstance(functions, list):
        raise ValueError("functions_invalid")
    if arguments is not None and not isinstance(arguments, dict):
        raise ValueError("function_arguments_invalid")
    if enhance:
        raw_functions, generated_arguments = _author_functions(
            facts,
            complete_json,
            instruction=instruction,
            existing_functions=functions or [],
        )
        arguments_by_function = {
            **generated_arguments,
            **dict(arguments or {}),
        }
    else:
        if not functions:
            raise ValueError("functions_required")
        raw_functions = json.loads(json.dumps(functions, ensure_ascii=False))
        arguments_by_function = dict(arguments or {})
    parsed_functions = [parse_function_artifact(value) for value in raw_functions]
    _validate_checker_evidence(parsed_functions, payload)
    function_ids = [function.id for function in parsed_functions]
    if len(function_ids) != len(set(function_ids)):
        raise ValueError("duplicate_function_id")
    if set(arguments_by_function) - set(function_ids):
        raise ValueError("function_arguments_unknown_function")
    saved_source_calls: list[dict[str, Any]] = []
    exact_source_coverages: list[tuple[int, ...]] = []
    for function in parsed_functions:
        raw_arguments = arguments_by_function.get(function.id, {})
        calls = raw_arguments if isinstance(raw_arguments, list) else [raw_arguments]
        if not calls or any(not isinstance(arguments, dict) for arguments in calls):
            raise ValueError("function_arguments_invalid")
        exact_fallback_grounding = False
        for arguments in calls:
            bound = bind_function(function, arguments)
            _validate_action_grounding(
                bound,
                steps,
                allow_semantic_relocation=True,
            )
            exact_fallback_grounding = exact_fallback_grounding or _action_grounded(
                bound,
                steps,
                allow_semantic_relocation=False,
            )
            exact_coverage = _function_source_indices(
                bound,
                steps,
                allow_semantic_relocation=False,
            )
            if exact_coverage is not None:
                source_index_groups = _function_source_index_groups(
                    bound,
                    steps,
                    allow_semantic_relocation=False,
                )
                if source_index_groups is None:
                    raise AssertionError("exact_source_index_groups_missing")
                _validate_checker_checkpoints(*source_index_groups)
                exact_source_coverages.append(exact_coverage)
            saved_source_calls.append(
                {
                    "function_id": function.id,
                    "arguments": _copy_value(arguments),
                }
            )
        if not exact_fallback_grounding:
            raise ValueError(
                "function_action_not_grounded:"
                f"{function.id}:{function.steps[0].step_index}"
            )
    if enhance and tuple(range(len(steps))) not in exact_source_coverages:
        raise ValueError("function_enhancement_full_trajectory_required")

    referenced_state_ids = _referenced_source_state_ids(parsed_functions)
    raw_states = source_catalog["states"]
    states = {
        str(source_state_id): _normalize_source_state(value, str(source_state_id))
        for source_state_id, value in raw_states.items()
    }
    missing_state_ids = [
        state_id for state_id in referenced_state_ids if state_id not in states
    ]
    if missing_state_ids:
        raise ValueError(
            "function_source_states_missing:" + ",".join(missing_state_ids)
        )
    frozen_states = {
        state_id: states[state_id] for state_id in referenced_state_ids
    }
    root.mkdir(parents=True, exist_ok=True)
    transfer_state_catalog_path = root / TRANSFER_STATE_CATALOG_FILENAME
    existing_states = load_transfer_state_catalog(transfer_state_catalog_path)
    merged_states = {**existing_states, **frozen_states}
    state_run_id = facts["run_id"]
    if transfer_state_catalog_path.is_file():
        existing_catalog = json.loads(
            transfer_state_catalog_path.read_text(encoding="utf-8")
        )
        existing_run_id = str(existing_catalog.get("run_id") or "").strip()
        if existing_run_id and existing_run_id != state_run_id:
            state_run_id = "multiple-runlogs"
    temporary_states = transfer_state_catalog_path.with_suffix(".json.tmp")
    temporary_states.write_text(
        json.dumps(
            {
                "schema_version": TRANSFER_STATE_CATALOG_VERSION,
                "run_id": state_run_id,
                "states": merged_states,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    store = FunctionStore(destination)
    for function in parsed_functions:
        store.functions[function.id] = function
    replaced_ids = set(function_ids)
    store.source_calls = [
        call
        for call in store.source_calls
        if call["function_id"] not in replaced_ids
    ] + saved_source_calls
    _write_store(destination, store.functions, store.source_calls)
    temporary_states.replace(transfer_state_catalog_path)
    report = {
        "schema_version": "omniflow.function-save.v1",
        "success": True,
        "store_path": str(destination),
        "transfer_state_catalog": str(transfer_state_catalog_path),
        "transfer_state_count": len(merged_states),
        "function_ids": function_ids,
        "function_count": len(function_ids),
        "source_arguments": json.loads(
            json.dumps(arguments_by_function, ensure_ascii=False)
        ),
        "enhanced": bool(enhance),
    }
    return report


def _referenced_source_state_ids(functions: list[Any]) -> list[str]:
    state_ids: list[str] = []
    for function in functions:
        for item in (*function.steps, *function.checker_rules):
            if isinstance(item, dict):
                state_id = str(item.get("source_state_id") or "").strip()
            else:
                state_id = str(getattr(item, "source_state_id", "") or "").strip()
            if state_id and state_id not in state_ids:
                state_ids.append(state_id)
    if not state_ids:
        raise ValueError("function_source_state_references_required")
    return state_ids


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
    return state


def _validate_action_grounding(
    function: Any,
    source_steps: list[dict[str, Any]],
    *,
    allow_semantic_relocation: bool = False,
) -> None:
    if _action_grounded(
        function,
        source_steps,
        allow_semantic_relocation=allow_semantic_relocation,
    ):
        return
    raise ValueError(
        "function_action_not_grounded:"
        f"{function.id}:{function.steps[0].step_index}"
    )


def _action_grounded(
    function: Any,
    source_steps: list[dict[str, Any]],
    *,
    allow_semantic_relocation: bool,
) -> bool:
    return _function_source_indices(
        function,
        source_steps,
        allow_semantic_relocation=allow_semantic_relocation,
    ) is not None


def _function_source_indices(
    function: Any,
    source_steps: list[dict[str, Any]],
    *,
    allow_semantic_relocation: bool,
) -> tuple[int, ...] | None:
    groups = _function_source_index_groups(
        function,
        source_steps,
        allow_semantic_relocation=allow_semantic_relocation,
    )
    if groups is None:
        return None
    formal_indices, checker_indices = groups
    return tuple(sorted((*formal_indices, *checker_indices)))


def _function_source_index_groups(
    function: Any,
    source_steps: list[dict[str, Any]],
    *,
    allow_semantic_relocation: bool,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    formal_indices: list[int] = []
    cursor = 0
    for expected_step in function.steps:
        matched_index = next(
            (
                index
                for index in range(cursor, len(source_steps))
                if (
                    allow_semantic_relocation
                    or str(source_steps[index].get("before_state_id") or "")
                    == expected_step.source_state_id
                )
                and _grounded_action_matches(
                    expected_step.action.to_dict(),
                    source_steps[index],
                    allow_semantic_relocation=allow_semantic_relocation,
                )
            ),
            None,
        )
        if matched_index is None:
            return None
        formal_indices.append(matched_index)
        cursor = matched_index + 1

    used = set(formal_indices)
    checker_indices: list[int] = []
    for rule in function.checker_rules:
        expected_action = canonicalize_action(rule.get("action"), replayable_only=True)
        source_state_id = str(rule.get("source_state_id") or "")
        matched_index = next(
            (
                index
                for index, source_step in enumerate(source_steps)
                if index not in used
                and str(source_step.get("before_state_id") or "") == source_state_id
                and _grounded_action_matches(
                    expected_action,
                    source_step,
                    allow_semantic_relocation=False,
                )
            ),
            None,
        )
        if matched_index is None:
            return None
        used.add(matched_index)
        checker_indices.append(matched_index)

    coverage = tuple(sorted((*formal_indices, *checker_indices)))
    if not coverage:
        return None
    if coverage != tuple(range(coverage[0], coverage[-1] + 1)):
        return None
    return tuple(formal_indices), tuple(checker_indices)


def _validate_checker_checkpoints(
    formal_indices: tuple[int, ...] | list[int],
    checker_indices: tuple[int, ...] | list[int],
) -> None:
    if any(
        not any(
            formal_index > checker_index for formal_index in formal_indices
        )
        for checker_index in checker_indices
    ):
        raise ValueError("checker_requires_later_formal_action")


def _grounded_action_matches(
    expected_action: dict[str, Any],
    source_step: dict[str, Any],
    *,
    allow_semantic_relocation: bool,
) -> bool:
    source_action = source_step["action"]
    if source_action == expected_action:
        return True
    if expected_action.get("tool") != "click" or source_action.get("tool") != "click":
        return False
    expected_args = expected_action.get("args")
    source_args = source_action.get("args")
    if not isinstance(expected_args, dict) or not isinstance(source_args, dict):
        return False
    target = str(expected_args.get("target_description") or "").strip()
    metadata = source_step.get("metadata")
    if (
        not target
        or not isinstance(metadata, dict)
        or str(metadata.get("semantic_target") or "").strip() != target
    ):
        return False
    fallback_action = {
        "tool": "click",
        "args": {
            key: value
            for key, value in expected_args.items()
            if key != "target_description"
        },
    }
    return allow_semantic_relocation or fallback_action == source_action


def _validate_checker_evidence(
    functions: list[Any],
    run_log: dict[str, Any],
) -> None:
    evidence: set[tuple[str, str]] = set()
    for raw_step in run_log.get("steps") or ():
        if not isinstance(raw_step, dict):
            continue
        result = raw_step.get("result")
        if not isinstance(result, dict) or result.get("success") is not True:
            continue
        source_state_id, actions = _enhancement_step_actions(raw_step)
        for action in actions:
            evidence.add(
                (
                    source_state_id,
                    json.dumps(
                        action,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
            )
    for function in functions:
        for rule in function.checker_rules:
            source_state_id = str(rule.get("source_state_id") or "")
            action = json.dumps(
                canonicalize_action(rule.get("action"), replayable_only=True),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if (source_state_id, action) not in evidence:
                raise ValueError("function_checker_rule_missing_recovery_evidence")
def _author_functions(
    facts: dict[str, Any],
    complete_json: Callable[[str, dict[str, Any]], str],
    *,
    instruction: str,
    existing_functions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bundle: dict[str, Any] | None = None
    for stage in ("split", "parameters", "checkers"):
        previous_bundle = bundle
        stage_prompt = _authoring_prompt(
            facts,
            stage=stage,
            current_bundle=bundle,
            existing_functions=existing_functions,
            instruction=instruction,
        )
        tool = function_authoring_tool(
            stage=stage,
            current_bundle=bundle,
        )
        validation_error: Exception | None = None
        for output_attempt in range(2):
            prompt = stage_prompt
            if output_attempt:
                assert validation_error is not None
                prompt = _authoring_correction_prompt(
                    stage_prompt,
                    validation_error,
                )
            try:
                raw_proposal = complete_json(prompt, tool)
            except Exception as error:
                raise ValueError(
                    f"function_enhancement_{stage}_model_failed:"
                    f"{type(error).__name__}:{error}"
                ) from error
            try:
                proposal = _json_object(raw_proposal)
            except ValueError as error:
                validation_error = ValueError(
                    f"function_enhancement_{stage}_output_invalid:{error}"
                )
            else:
                try:
                    candidate = _validate_agent_bundle(proposal, stage=stage)
                    _validate_agent_stage_contract(
                        candidate,
                        previous_bundle=previous_bundle,
                        stage=stage,
                    )
                    _validate_agent_trajectory(candidate, facts, stage=stage)
                except (TypeError, ValueError) as error:
                    validation_error = error
                else:
                    bundle = candidate
                    break
            assert validation_error is not None
            if output_attempt == 0:
                continue
            raise validation_error
    assert bundle is not None
    return bundle["functions"], bundle["arguments"]


def _authoring_correction_prompt(
    stage_prompt: str,
    error: Exception,
) -> str:
    return (
        f"{stage_prompt}\n\n"
        "The previous full bundle was rejected by the authoritative validator: "
        f"{type(error).__name__}: {error}. Correct that exact violation and return "
        "the complete functions and arguments bundle again. Preserve every field "
        "owned by earlier stages. This is the only correction opportunity for this "
        "stage."
    )


def _validate_agent_bundle(value: Any, *, stage: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"functions", "arguments"}:
        raise ValueError(f"function_enhancement_{stage}_output_invalid")
    functions = value.get("functions")
    arguments = value.get("arguments")
    if not isinstance(functions, list) or not functions:
        raise ValueError(f"function_enhancement_{stage}_functions_required")
    if not isinstance(arguments, dict):
        raise ValueError(f"function_enhancement_{stage}_arguments_invalid")
    parsed = [parse_function_artifact(function).to_dict() for function in functions]
    function_ids = [function["function_id"] for function in parsed]
    if len(function_ids) != len(set(function_ids)):
        raise ValueError("duplicate_function_id")
    if set(arguments) != set(function_ids):
        raise ValueError(f"function_enhancement_{stage}_arguments_incomplete")
    return {
        "functions": parsed,
        "arguments": json.loads(json.dumps(arguments, ensure_ascii=False)),
    }


def _validate_agent_stage_contract(
    bundle: dict[str, Any],
    *,
    previous_bundle: dict[str, Any] | None,
    stage: str,
) -> None:
    functions = [parse_function_artifact(value) for value in bundle["functions"]]
    previous_functions: list[dict[str, Any]] = []
    if previous_bundle is not None:
        previous_functions = list(previous_bundle["functions"])
        previous_ids = [str(value["function_id"]) for value in previous_functions]
        if [function.id for function in functions] != previous_ids:
            raise ValueError(f"function_enhancement_{stage}_function_set_changed")
    if stage in {"split", "parameters"} and any(
        function.checker_rules for function in functions
    ):
        raise ValueError(f"function_enhancement_{stage}_checker_rules_forbidden")
    if stage == "parameters":
        immutable_fields = {
            "schema_version",
            "function_id",
            "name",
            "description",
            "checker_rules",
            "agent_visible",
        }
        for function, previous in zip(
            functions,
            previous_functions,
            strict=True,
        ):
            current = function.to_dict()
            if any(current[field] != previous[field] for field in immutable_fields):
                raise ValueError("parameters_stage_changed_function_logic")
            raw_arguments = bundle["arguments"][function.id]
            calls = raw_arguments if isinstance(raw_arguments, list) else [raw_arguments]
            if any(
                bind_function(function, arguments).to_dict()["steps"]
                != previous["steps"]
                for arguments in calls
            ):
                raise ValueError("parameters_stage_changed_function_logic")
        return
    if stage == "checkers":
        if bundle["arguments"] != previous_bundle["arguments"]:
            raise ValueError("checkers_stage_changed_arguments")
        immutable_fields = {
            "schema_version",
            "function_id",
            "name",
            "description",
            "input_schema",
            "agent_visible",
        }
        for function, previous in zip(
            functions,
            previous_functions,
            strict=True,
        ):
            current = function.to_dict()
            if any(current[field] != previous[field] for field in immutable_fields):
                raise ValueError("checkers_stage_changed_function_logic")
            selected_indices: list[int] = []
            search_start = 0
            for rule in current["checker_rules"]:
                selected_index = next(
                    (
                        index
                        for index in range(search_start, len(previous["steps"]))
                        if previous["steps"][index]["source_state_id"]
                        == rule["source_state_id"]
                        and previous["steps"][index]["action"] == rule["action"]
                    ),
                    None,
                )
                if selected_index is None:
                    raise ValueError("checker_not_registered_on_function")
                selected_indices.append(selected_index)
                search_start = selected_index + 1
            selected = set(selected_indices)
            remaining_indices = [
                index
                for index in range(len(previous["steps"]))
                if index not in selected
            ]
            _validate_checker_checkpoints(remaining_indices, selected_indices)
            expected_steps = [
                {**previous["steps"][old_index], "step_index": new_index}
                for new_index, old_index in enumerate(remaining_indices)
            ]
            if current["steps"] != expected_steps:
                raise ValueError("checkers_stage_changed_function_logic")
            new_step_index = {
                old_index: new_index
                for new_index, old_index in enumerate(remaining_indices)
            }
            expected_bindings: list[dict[str, str]] = []
            for binding in previous["bindings"]:
                target_match = _TARGET_PATH.fullmatch(binding["target"])
                if target_match is None:
                    raise ValueError("function_binding_path_invalid")
                old_index = int(target_match.group("action_index"))
                if old_index in selected:
                    raise ValueError("checker_action_cannot_use_parameter_binding")
                expected_bindings.append(
                    {
                        "source": binding["source"],
                        "target": (
                            f"$.steps[{new_step_index[old_index]}].action.args"
                            f"{target_match.group('tail')}"
                        ),
                    }
                )
            if current["bindings"] != expected_bindings:
                raise ValueError("checkers_stage_changed_parameter_bindings")
        return
    if stage != "split":
        return
    for function in functions:
        if (
            function.input_schema
            != {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            }
            or function.bindings
            or bundle["arguments"][function.id] != {}
        ):
            raise ValueError("function_enhancement_split_parameters_forbidden")


def _validate_agent_trajectory(
    bundle: dict[str, Any],
    facts: dict[str, Any],
    *,
    stage: str,
) -> None:
    source_steps = list(facts["steps"])
    full_trajectory_present = False
    for raw_function in bundle["functions"]:
        function = parse_function_artifact(raw_function)
        raw_calls = bundle["arguments"].get(function.id, {})
        calls = raw_calls if isinstance(raw_calls, list) else [raw_calls]
        for arguments in calls:
            if not isinstance(arguments, dict):
                raise ValueError(
                    f"function_enhancement_{stage}_arguments_invalid"
                )
            bound = bind_function(function, arguments)
            if _function_source_indices(
                bound,
                source_steps,
                allow_semantic_relocation=False,
            ) == tuple(range(len(source_steps))):
                full_trajectory_present = True
                break
        if full_trajectory_present:
            break
    if not full_trajectory_present:
        raise ValueError(f"function_enhancement_full_trajectory_required:{stage}")
    if stage == "split" and len(source_steps) > 1:
        for raw_function in bundle["functions"]:
            function = parse_function_artifact(raw_function)
            if (
                len(function.steps) == 1
                and function.steps[0].action.tool in {"click", "long_press"}
            ):
                raise ValueError(
                    "function_enhancement_single_click_fragment_forbidden"
                )


def _authoring_prompt(
    facts: dict[str, Any],
    *,
    stage: str,
    current_bundle: dict[str, Any] | None,
    existing_functions: list[dict[str, Any]],
    instruction: str,
) -> str:
    stage_instruction = {
        "split": (
            "Split the successful trajectory into semantic operations. Keep one "
            "Function covering the full trajectory, and identify every reusable "
            "contiguous semantic subsegment without creating one-click fragments. "
            "Draft each complete Function. In this stage only, use "
            "an empty object input_schema, bindings=[], and empty arguments for every "
            "Function; parameterization belongs exclusively to the next stage."
        ),
        "parameters": (
            "Review the draft and expose only caller-varying text, numbers, dates, "
            "and choices through input_schema, bindings, and source arguments. "
            "Keep stable app packages and fixed navigation controls inside the Function. "
            "Return the same Functions in the same order with the same identity, "
            "description, visibility, source states, action tools, and step order."
        ),
        "checkers": (
            "Select only optional existing formal actions that should run when their "
            "RunLog source state and mapped target are both present and are safe to "
            "skip without breaking the remaining formal path. This may include optional "
            "setup, interruption dismissal, recovery, or alternate-path navigation. "
            "Move each selected action to checker_rules on that same Function. Every "
            "selected action must have a later unselected formal action, because rules "
            "are evaluated only before pending formal actions. Do not change Function "
            "meaning, identity, order, parameters, arguments, or any unselected action. "
            "Copy the complete Function set; checker_rules may contain only exact "
            "source-state/action pairs moved from that same Function."
        ),
    }[stage]
    example_steps = [
        {
            "step_index": 0,
            "source_state_id": "source-home",
            "action": {"tool": "click", "args": {"x": 700, "y": 800}},
        },
        {
            "step_index": 1,
            "source_state_id": "source-promo-dialog",
            "action": {"tool": "click", "args": {"x": 800, "y": 700}},
        },
        {
            "step_index": 2,
            "source_state_id": "source-search-page",
            "action": {"tool": "input_text", "args": {"text": ""}},
        },
        {
            "step_index": 3,
            "source_state_id": "source-search-filled",
            "action": {"tool": "click", "args": {"x": 900, "y": 900}},
        },
    ]
    if stage == "split":
        example_steps[2]["action"]["args"]["text"] = "museum"
        example_bindings: list[dict[str, str]] = []
        example_properties: dict[str, Any] = {}
        example_required: list[str] = []
        example_arguments: dict[str, Any] = {}
        example_checkers: list[dict[str, Any]] = []
    else:
        example_bindings = [
            {
                "source": "$.arguments.query",
                "target": "$.steps[2].action.args.text",
            }
        ]
        example_properties = {
            "query": {"type": "string", "description": "Text to search for"}
        }
        example_required = ["query"]
        example_arguments = {"query": "museum"}
        example_checkers = []
    if stage == "checkers":
        optional_step = example_steps.pop(1)
        for index, step in enumerate(example_steps):
            step["step_index"] = index
        example_bindings[0]["target"] = "$.steps[1].action.args.text"
        example_checkers = [
            {
                "source_state_id": optional_step["source_state_id"],
                "action": optional_step["action"],
            }
        ]
    subsegment_steps = json.loads(json.dumps(example_steps[-2:]))
    for index, step in enumerate(subsegment_steps):
        step["step_index"] = index
    subsegment_bindings = (
        []
        if stage == "split"
        else [
            {
                "source": "$.arguments.query",
                "target": "$.steps[0].action.args.text",
            }
        ]
    )
    subsegment_arguments = {} if stage == "split" else {"query": "museum"}
    example = {
        "functions": [
            {
                "schema_version": FUNCTION_ARTIFACT_VERSION,
                "function_id": "search_the_web",
                "name": "Search the web",
                "description": (
                    "Open the browser, enter a task-provided query, and submit it."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": example_properties,
                    "required": example_required,
                    "additionalProperties": False,
                },
                "bindings": example_bindings,
                "steps": example_steps,
                "checker_rules": example_checkers,
                "agent_visible": True,
            },
            {
                "schema_version": FUNCTION_ARTIFACT_VERSION,
                "function_id": "submit_web_search",
                "name": "Submit web search",
                "description": "Enter a task-provided query and submit the search.",
                "input_schema": {
                    "type": "object",
                    "properties": example_properties,
                    "required": example_required,
                    "additionalProperties": False,
                },
                "bindings": subsegment_bindings,
                "steps": subsegment_steps,
                "checker_rules": [],
                "agent_visible": True,
            },
        ],
        "arguments": {
            "search_the_web": example_arguments,
            "submit_web_search": subsegment_arguments,
        },
    }
    evidence = {
        "goal": facts["goal"],
        "actions": facts["steps"],
        "existing_functions": existing_functions,
        "current_bundle": current_bundle,
        "instruction": str(instruction or "").strip()[:2000],
    }
    return f"""
You are stage {stage} of one offline save_function pipeline.
{stage_instruction}

Return exactly one complete JSON object with keys functions and arguments. Every Function
must use omniflow.function.v2 and contain function_id, name, description, input_schema,
bindings, ordered steps, checker_rules, and agent_visible. You may write actions directly,
but every source_state_id and action must be supported by the supplied successful RunLog.
At every stage, include at least one large semantic Function that covers the complete
successful trajectory. Do not replace the complete Function with one Function per click.
Identify every reusable contiguous semantic subsegment and return it alongside the required
full-trajectory Function; do not create meaningless single-click fragments. Preserve source action order
and never invent target-device state, target coordinates,
validator logic, task-specific gates, or source-coordinate fallback. One Function is a
reusable semantic operation, not one click. A checker belongs only to its Function and
contains exactly source_state_id and action; never add a trigger expression, step number,
or condition object. During checker review, move only optional actions whose execution is
conditional on their RunLog source state and mapped target and that are safe to skip without
breaking the remaining formal path. This can include optional setup, interruption dismissal,
recovery, or alternate-path navigation. Every checker must have a later unselected formal
action that provides a runtime check point. Never move required navigation or the terminal
action into checker_rules, duplicate a formal action as a checker, or create a standalone
Function merely to hold a checker.
Checker actions must be transferable click, input_text, or long_press actions. Parameters must have
source arguments that reproduce the recorded source action after binding. Return the full
bundle even when this stage makes no change. Do not add commentary or extra keys.

Few-shot shape:
{json.dumps(example, ensure_ascii=False, separators=(",", ":"))}

RunLog evidence and current draft:
{json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))}
""".strip()


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
    return {
        "package": str(auxiliaries.get("package_name") or ""),
        "activity": str(auxiliaries.get("activity_name") or ""),
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
    return {
        "package": str(state.get("package_name") or ""),
        "activity": str(state.get("activity_name") or ""),
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
