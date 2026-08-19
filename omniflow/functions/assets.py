from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
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

FUNCTION_ARTIFACT_VERSION = "omniflow.function.v2"
STORE_VERSION = "omniflow.store.v2"
_AUTHORING_STAGE_ATTEMPTS = 3

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
) -> dict[str, Any]:
    """Return the small decision contract used by the offline enhancer."""

    metadata = {
        "type": "object",
        "additionalProperties": False,
        "required": ["function_id", "name", "description"],
        "properties": {
            "function_id": {
                "type": "string",
                "pattern": "^[a-z][a-z0-9_]{0,63}$",
            },
            "name": {"type": "string", "minLength": 1, "maxLength": 120},
            "description": {"type": "string", "minLength": 1},
        },
    }
    parameter = {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "description"],
        "properties": {
            "name": {
                "type": "string",
                "pattern": "^[a-z][a-z0-9_]{0,63}$",
            },
            "description": {"type": "string", "minLength": 1},
        },
    }
    contracts = {
        "split": {
            "description": (
                "Name the one complete Function covering the successful RunLog."
            ),
            "required": ["complete_function"],
            "properties": {
                "complete_function": metadata,
            },
        },
        "parameters": {
            "description": (
                "Author source-grounded actions and caller parameters. Direct actions "
                "are accepted only when the compiler can prove them from the RunLog."
            ),
            "required": ["action_edits", "bindings"],
            "properties": {
                "action_edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "function_id",
                            "step_index",
                            "operation",
                        ],
                        "properties": {
                            "function_id": metadata["properties"]["function_id"],
                            "step_index": {"type": "integer", "minimum": 0},
                            "operation": {
                                "type": "string",
                                "enum": ["open_app", "set_target"],
                            },
                        },
                    },
                },
                "bindings": {
                    "type": "array",
                    "items": {
                        **parameter,
                        "required": [
                            "function_id",
                            "step_index",
                            "name",
                            "description",
                        ],
                        "properties": {
                            "function_id": metadata["properties"]["function_id"],
                            "step_index": {"type": "integer", "minimum": 0},
                            **parameter["properties"],
                        },
                    },
                },
                "actions": {
                    "type": "array",
                    "description": (
                        "Optional direct action proposals. Use the original RunLog "
                        "step index; save_function rejects any ungrounded action."
                    ),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["function_id", "step_index", "action"],
                        "properties": {
                            "function_id": metadata["properties"]["function_id"],
                            "step_index": {"type": "integer", "minimum": 0},
                            "action": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["tool", "args"],
                                "properties": {
                                    "tool": {"type": "string", "minLength": 1},
                                    "args": {"type": "object"},
                                },
                            },
                        },
                    },
                },
            },
        },
        "checkers": {
            "description": "Register optional RunLog actions as Function checkers.",
            "required": ["checker_steps"],
            "properties": {
                "checker_steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["function_id", "step_index"],
                        "properties": {
                            "function_id": metadata["properties"]["function_id"],
                            "step_index": {"type": "integer", "minimum": 0},
                        },
                    },
                }
            },
        },
    }
    if stage not in contracts:
        raise ValueError(f"function_authoring_stage_invalid:{stage}")
    contract = contracts[stage]
    return {
        "type": "function",
        "function": {
            "name": "edit_function_draft",
            "description": contract["description"],
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": contract["required"],
                "properties": contract["properties"],
            },
        },
    }


def parse_function_artifact(value: dict[str, Any]) -> Function:
    if not isinstance(value, dict):
        raise ValueError("function_artifact_must_be_object")
    missing = sorted(_TOP_LEVEL_FIELDS - set(value))
    if missing:
        raise ValueError(f"function_artifact_missing_fields:{','.join(missing)}")
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
    if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", function.function_id) is None:
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
        if len(raw_functions) > 1:
            raise ValueError("function_store_single_function_required")
        raw_source_calls = payload.get("source_calls") or []
        if not isinstance(raw_source_calls, list) or any(
            not isinstance(call, dict)
            or set(call) != {"function_id", "arguments"}
            or not str(call.get("function_id") or "").strip()
            or not isinstance(call.get("arguments"), dict)
            for call in raw_source_calls
        ):
            raise ValueError("function_store_source_calls_invalid")
        if len(raw_source_calls) > 1:
            raise ValueError("function_store_single_source_call_required")
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
    if len(parsed) > 1:
        raise ValueError("function_store_single_function_required")

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
    if len(normalized_calls) > 1:
        raise ValueError("function_store_single_source_call_required")

    destination = Path(path).expanduser().resolve()
    _write_store(destination, parsed, normalized_calls)
    return destination


def _write_json_artifact(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_file_artifact(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _materialize_canonical_run_log(
    *,
    bundle_root: Path,
    payload: dict[str, Any],
    source_run_log_path: Path | None,
) -> Path:
    canonical = json.loads(json.dumps(payload, ensure_ascii=False))
    screenshots_root = bundle_root / "screenshots"
    suffixes = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }
    seen: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            pixels = value.get("pixels")
            if isinstance(pixels, dict) and pixels.get("path"):
                digest = str(pixels.get("sha256") or "").strip().lower()
                mime_type = str(pixels.get("mime_type") or "").strip()
                suffix = suffixes.get(mime_type)
                if len(digest) != 64 or suffix is None:
                    raise ValueError("run_log_screenshot_metadata_invalid")
                source = Path(str(pixels["path"])).expanduser().resolve()
                if not source.is_file() and source_run_log_path is not None:
                    source = (source_run_log_path.parent / source.name).resolve()
                if not source.is_file():
                    raise FileNotFoundError(f"run_log_screenshot_missing:{source}")
                if _sha256_file(source) != digest:
                    raise ValueError(f"run_log_screenshot_hash_mismatch:{source}")
                target = screenshots_root / f"{digest}{suffix}"
                if digest not in seen:
                    _copy_file_artifact(source, target)
                    seen.add(digest)
                pixels["path"] = str(target)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(canonical)
    run_log_path = bundle_root / "run_log.json"
    _write_json_artifact(run_log_path, canonical)
    return run_log_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bundle_identity(
    destination: Path,
    *,
    task: str,
) -> dict[str, str] | None:
    parts = destination.parts
    for index, component in enumerate(parts):
        if component != "data":
            continue
        remaining = parts[index + 1 :]
        if not remaining:
            continue
        if remaining[0] not in {"androidworld", "bmoca"}:
            raise ValueError("function_artifact_path_not_canonical")
        if len(remaining) != 7:
            raise ValueError("function_artifact_path_not_canonical")
        environment, path_task, device, category, method, attempt_id, filename = (
            remaining
        )
        if path_task != task or category != "function" or filename != destination.name:
            raise ValueError("function_artifact_path_not_canonical")
        if not all(
            str(value).strip()
            for value in (environment, path_task, device, method, attempt_id)
        ):
            raise ValueError("function_artifact_path_not_canonical")
        return {
            "environment": environment,
            "task": path_task,
            "device": device,
            "category": category,
            "method": method,
            "attempt_id": attempt_id,
        }
    return None


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
    )

    destination = Path(store_path).expanduser().resolve()
    root = destination.parent

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
    if source_run_log_path is None:
        provenance = payload.get("provenance")
        candidate = (
            Path(str(provenance.get("source_path"))).expanduser().resolve()
            if isinstance(provenance, dict)
            and str(provenance.get("source_path") or "").strip()
            and Path(str(provenance.get("source_path"))).expanduser().is_absolute()
            else None
        )
        expected_hash = (
            str(provenance.get("source_sha256") or "").strip().lower()
            if isinstance(provenance, dict)
            else ""
        )
        if candidate is not None and candidate.is_file():
            if expected_hash and _sha256_file(candidate) != expected_hash:
                raise ValueError("function_source_run_log_sha256_mismatch")
            source_run_log_path = candidate
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
    }
    if enhance and complete_json is None:
        raise ValueError("function_enhancer_required")
    if functions is not None and (
        not isinstance(functions, list) or len(functions) > 1
    ):
        raise ValueError("function_single_function_required")
    if arguments is not None and not isinstance(arguments, dict):
        raise ValueError("function_arguments_invalid")
    enhanced_source_function_id: str | None = None
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
        enhanced_source_function_id = str(
            raw_functions[0].get("function_id") or ""
        ).strip()
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
    parsed_functions = [parse_function_artifact(value) for value in raw_functions]
    if len(parsed_functions) != 1:
        raise ValueError("function_single_function_required")
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
        if isinstance(raw_arguments, list):
            raise ValueError("function_single_source_call_required")
        calls = [raw_arguments]
        if not isinstance(raw_arguments, dict):
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
            if not enhance or function.id == enhanced_source_function_id:
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
    if source_run_log_path is None:
        source_run_log_path = root / "run_log.json"
        _write_json_artifact(source_run_log_path, payload)
    elif _canonical_bundle_identity(
        destination,
        task=str(payload.get("task_name") or payload.get("task") or "").strip(),
    ) is not None:
        source_run_log_path = _materialize_canonical_run_log(
            bundle_root=root,
            payload=payload,
            source_run_log_path=source_run_log_path,
        )
    transfer_state_catalog_path = root / TRANSFER_STATE_CATALOG_FILENAME
    merged_states = frozen_states
    state_run_id = facts["run_id"]
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
    store.functions = {function.id: function for function in parsed_functions}
    store.source_calls = list(saved_source_calls)
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
    metadata = source_step.get("metadata")
    if (
        expected_action.get("tool") == "open_app"
        and source_action.get("tool") == "click"
        and isinstance(metadata, dict)
    ):
        source_page = metadata.get("source_page") or {}
        after_page = metadata.get("after_page") or {}
        expected_package = str(
            (expected_action.get("args") or {}).get("package_name") or ""
        ).strip()
        return bool(
            source_page.get("is_launcher") is True
            and expected_package
            and expected_package == str(after_page.get("package") or "").strip()
            and expected_package != str(source_page.get("package") or "").strip()
        )
    if expected_action.get("tool") != "click" or source_action.get("tool") != "click":
        return False
    expected_args = expected_action.get("args")
    source_args = source_action.get("args")
    if not isinstance(expected_args, dict) or not isinstance(source_args, dict):
        return False
    target = str(expected_args.get("target_description") or "").strip()
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
    split = _request_authoring_decision(
        complete_json,
        prompt=_draft_split_prompt(
            facts,
            existing_functions=existing_functions,
            instruction=instruction,
        ),
        tool=function_authoring_tool(stage="split"),
        label="split",
        validate=lambda value: _validate_split_draft(value, facts),
    )
    plan = split["complete_function"]
    parameters: dict[str, list[dict[str, Any]]] = {
        "action_edits": [],
        "bindings": [],
        "actions": [],
    }
    function_id = plan["function_id"]
    decision = _request_authoring_decision(
        complete_json,
        prompt=_draft_parameters_prompt(facts, plan),
        tool=function_authoring_tool(stage="parameters"),
        label=f"parameters:{function_id}",
        validate=lambda value: _validate_parameter_draft_for_function(
            value,
            facts,
            split,
            function_id=function_id,
        ),
    )
    parameters["action_edits"].extend(decision["action_edits"])
    parameters["bindings"].extend(decision["bindings"])
    parameters["actions"].extend(decision["actions"])
    checkers = _request_authoring_decision(
        complete_json,
        prompt=_draft_checkers_prompt(facts, plan, parameters),
        tool=function_authoring_tool(stage="checkers"),
        label=f"checkers:{function_id}",
        validate=lambda value: _validate_checker_draft_for_function(
            value,
            facts,
            split,
            parameters,
            function_id=function_id,
        ),
    )
    return _compile_function_draft(facts, split, parameters, checkers)


def _authoring_correction_prompt(
    stage_prompt: str,
    error: Exception,
) -> str:
    correction = ""
    if str(error).startswith("function_parameter_value_not_requested:"):
        correction = (
            " Remove that binding. A source-state value absent from the goal may "
            "keep a source-proven action edit, but it cannot be an input parameter."
        )
    elif str(error).startswith("function_parameter_source_value_conflict:"):
        correction = (
            " The same parameter name was assigned to different source values. "
            "Keep a binding only for a caller-varying value requested by the goal; "
            "use distinct semantic parameter names for distinct values, or remove "
            "the binding when the value is source state."
        )
    elif str(error).startswith("function_parameter_type_conflict:"):
        correction = (
            " The same parameter name was assigned to different JSON types. "
            "Use distinct semantic parameter names for the different values, or "
            "remove the binding that is not caller input."
        )
    return (
        f"{stage_prompt}\n\n"
        "The previous small decision was rejected: "
        f"{type(error).__name__}: {error}.{correction} Correct only this decision and return "
        "the same small schema once."
    )


def _request_authoring_decision(
    complete_json: Callable[[str, dict[str, Any]], str],
    *,
    prompt: str,
    tool: dict[str, Any],
    label: str,
    validate: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    validation_error: Exception | None = None
    for attempt in range(_AUTHORING_STAGE_ATTEMPTS):
        request = (
            prompt
            if attempt == 0
            else _authoring_correction_prompt(prompt, validation_error)
        )
        try:
            raw = complete_json(request, tool)
        except Exception as error:
            raise ValueError(
                f"function_enhancement_{label}_model_failed:"
                f"{type(error).__name__}:{error}"
            ) from error
        try:
            return validate(_json_object(raw))
        except (TypeError, ValueError) as error:
            validation_error = error
    assert validation_error is not None
    raise validation_error


def _validate_function_metadata(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {
        "function_id",
        "name",
        "description",
    }:
        raise ValueError("function_metadata_contract_invalid")
    function_id = str(value.get("function_id") or "").strip()
    name = str(value.get("name") or "").strip()
    description = str(value.get("description") or "").strip()
    if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", function_id) is None:
        raise ValueError("function_id_invalid")
    if not name or len(name) > 120:
        raise ValueError("function_name_invalid")
    if not description:
        raise ValueError("function_description_required")
    return {
        "function_id": function_id,
        "name": name,
        "description": description,
    }


def _function_indices(value: dict[str, Any], step_count: int) -> tuple[int, ...]:
    if "start_step_index" not in value:
        return tuple(range(step_count))
    return tuple(range(value["start_step_index"], value["end_step_index"]))


def _validate_split_draft(value: Any, facts: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"complete_function"}:
        raise ValueError("function_split_contract_invalid")
    return {"complete_function": _validate_function_metadata(value["complete_function"])}


def _validate_parameter_draft(
    value: Any,
    facts: dict[str, Any],
    split: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict) or not set(value).issubset(
        {"action_edits", "bindings", "actions"}
    ) or not {"action_edits", "bindings"}.issubset(value):
        raise ValueError("function_parameters_contract_invalid")
    raw_edits = value["action_edits"]
    raw_bindings = value["bindings"]
    raw_actions = value.get("actions", [])
    if (
        not isinstance(raw_edits, list)
        or not isinstance(raw_bindings, list)
        or not isinstance(raw_actions, list)
    ):
        raise ValueError("function_parameters_invalid")
    functions = {
        item["function_id"]: _function_indices(item, len(facts["steps"]))
        for item in (split["complete_function"],)
    }
    action_edits: list[dict[str, Any]] = []
    edited_steps: set[tuple[str, int]] = set()
    for raw in raw_edits:
        if not isinstance(raw, dict) or set(raw) != {
            "function_id",
            "step_index",
            "operation",
        }:
            raise ValueError("function_action_edit_contract_invalid")
        function_id = str(raw.get("function_id") or "").strip()
        step_index = raw.get("step_index")
        operation = str(raw.get("operation") or "").strip()
        if function_id not in functions:
            raise ValueError("function_action_edit_unknown_function")
        if (
            not isinstance(step_index, int)
            or isinstance(step_index, bool)
            or step_index not in functions[function_id]
        ):
            raise ValueError("function_action_edit_step_not_in_function")
        key = (function_id, step_index)
        if key in edited_steps:
            raise ValueError("function_action_edit_duplicate")
        source_step = facts["steps"][step_index]
        source_action = source_step["action"]
        metadata = source_step.get("metadata") or {}
        if operation == "set_target":
            source_target = str(metadata.get("semantic_target") or "").strip()
            if source_action["tool"] != "click" or not source_target:
                raise ValueError("function_action_target_not_source_proven")
            edit_value = source_target
        elif operation == "open_app":
            source_page = metadata.get("source_page") or {}
            after_page = metadata.get("after_page") or {}
            after_package = str(after_page.get("package") or "").strip()
            if (
                source_action["tool"] != "click"
                or source_page.get("is_launcher") is not True
                or not after_package
                or after_package == str(source_page.get("package") or "").strip()
            ):
                raise ValueError("function_open_app_not_source_proven")
            edit_value = after_package
        else:
            raise ValueError("function_action_edit_operation_invalid")
        edited_steps.add(key)
        action_edits.append(
            {
                "function_id": function_id,
                "step_index": step_index,
                "operation": operation,
                "value": edit_value,
            }
        )
    action_overrides: list[dict[str, Any]] = []
    override_steps: set[tuple[str, int]] = set()
    for raw in raw_actions:
        if not isinstance(raw, dict) or set(raw) != {
            "function_id",
            "step_index",
            "action",
        }:
            raise ValueError("function_direct_action_contract_invalid")
        function_id = str(raw.get("function_id") or "").strip()
        step_index = raw.get("step_index")
        if function_id not in functions:
            raise ValueError("function_direct_action_unknown_function")
        if (
            not isinstance(step_index, int)
            or isinstance(step_index, bool)
            or step_index not in functions[function_id]
        ):
            raise ValueError("function_direct_action_step_not_in_function")
        key = (function_id, int(step_index))
        if key in override_steps or key in edited_steps:
            raise ValueError("function_direct_action_duplicate")
        try:
            proposed = canonicalize_action(raw["action"], replayable_only=True)
        except (TypeError, ValueError) as error:
            raise ValueError("function_direct_action_invalid") from error
        proposed = _validate_direct_action(
            facts,
            proposed,
            step_index=int(step_index),
        )
        override_steps.add(key)
        action_overrides.append(
            {
                "function_id": function_id,
                "step_index": int(step_index),
                "action": proposed,
            }
        )
    bindings: list[dict[str, Any]] = []
    targets: set[tuple[str, int, str]] = set()
    for raw in raw_bindings:
        expected = {
            "function_id",
            "step_index",
            "name",
            "description",
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError("function_parameter_contract_invalid")
        function_id = str(raw.get("function_id") or "").strip()
        step_index = raw.get("step_index")
        name = str(raw.get("name") or "").strip()
        description = str(raw.get("description") or "").strip()
        if function_id not in functions:
            raise ValueError("function_parameter_unknown_function")
        if (
            not isinstance(step_index, int)
            or isinstance(step_index, bool)
            or step_index not in functions[function_id]
        ):
            raise ValueError("function_parameter_step_not_in_function")
        if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name) is None:
            raise ValueError("function_parameter_name_invalid")
        if not description:
            raise ValueError("function_parameter_description_required")
        draft_action = _draft_action(
            facts,
            action_edits,
            action_overrides=action_overrides,
            function_id=function_id,
            step_index=step_index,
        )
        args = draft_action["args"]
        if draft_action["tool"] == "input_text" and "text" in args:
            path = "text"
        elif draft_action["tool"] == "click" and "target_description" in args:
            path = "target_description"
        else:
            raise ValueError("function_parameter_target_unavailable")
        target = (function_id, int(step_index), path)
        if target in targets:
            raise ValueError("function_parameter_target_duplicate")
        try:
            source_value = _read_path(
                args,
                _tokens("." + path),
            )
        except (IndexError, KeyError, TypeError) as error:
            raise ValueError(f"function_parameter_path_missing:{path}") from error
        if _json_type(source_value) is None:
            raise ValueError(f"function_parameter_type_invalid:{path}")
        requested_value = str(source_value).strip().casefold()
        if requested_value and requested_value not in str(facts["goal"]).casefold():
            raise ValueError(f"function_parameter_value_not_requested:{name}")
        targets.add(target)
        bindings.append(
            {
                "function_id": function_id,
                "step_index": int(step_index),
                "name": name,
                "description": description,
                "argument_path": path,
            }
        )
    return {
        "action_edits": action_edits,
        "bindings": bindings,
        "actions": action_overrides,
    }


def _validate_parameter_draft_for_function(
    value: Any,
    facts: dict[str, Any],
    split: dict[str, Any],
    *,
    function_id: str,
) -> dict[str, Any]:
    decision = _validate_parameter_draft(value, facts, split)
    if any(
        item["function_id"] != function_id
        for key in ("action_edits", "bindings", "actions")
        for item in decision[key]
    ):
        raise ValueError("function_stage_must_edit_one_function")
    return decision


def _draft_action(
    facts: dict[str, Any],
    action_edits: list[dict[str, Any]],
    *,
    action_overrides: list[dict[str, Any]] | None = None,
    function_id: str,
    step_index: int,
) -> dict[str, Any]:
    direct_action = next(
        (
            item["action"]
            for item in action_overrides or ()
            if item["function_id"] == function_id
            and item["step_index"] == step_index
        ),
        None,
    )
    if direct_action is not None:
        return _copy_value(direct_action)
    action = _copy_value(facts["steps"][step_index]["action"])
    edit = next(
        (
            item
            for item in action_edits
            if item["function_id"] == function_id
            and item["step_index"] == step_index
        ),
        None,
    )
    if edit is None:
        return action
    if edit["operation"] == "open_app":
        return {"tool": "open_app", "args": {"package_name": edit["value"]}}
    action["args"]["target_description"] = edit["value"]
    return action


def _validate_direct_action(
    facts: dict[str, Any],
    proposed: dict[str, Any],
    *,
    step_index: int,
) -> dict[str, Any]:
    """Allow exploration-style authoring while keeping actions RunLog-grounded."""

    source = facts["steps"][step_index]
    source_action = source["action"]
    if proposed == source_action:
        return proposed
    metadata = source.get("metadata") or {}
    source_target = str(metadata.get("semantic_target") or "").strip()
    source_page = metadata.get("source_page") or {}
    after_page = metadata.get("after_page") or {}
    after_package = str(after_page.get("package") or "").strip()
    if (
        proposed.get("tool") == "open_app"
        and source_action.get("tool") == "click"
        and source_page.get("is_launcher") is True
        and after_package
        and after_package != str(source_page.get("package") or "").strip()
        and proposed.get("args") == {"package_name": after_package}
    ):
        return proposed
    if (
        proposed.get("tool") == "click"
        and source_action.get("tool") == "click"
        and source_target
        and proposed.get("args")
        == {
            **dict(source_action.get("args") or {}),
            "target_description": source_target,
        }
    ):
        return proposed
    raise ValueError(f"function_direct_action_not_runlog_grounded:{step_index}")


def _validate_checker_draft(
    value: Any,
    facts: dict[str, Any],
    split: dict[str, Any],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"checker_steps"}:
        raise ValueError("function_checkers_contract_invalid")
    raw_checkers = value["checker_steps"]
    if not isinstance(raw_checkers, list):
        raise ValueError("function_checkers_invalid")
    plans = {split["complete_function"]["function_id"]: split["complete_function"]}
    functions = {
        function_id: _function_indices(item, len(facts["steps"]))
        for function_id, item in plans.items()
    }
    parameter_steps = {
        (item["function_id"], item["step_index"])
        for item in parameters["bindings"]
    }
    parameter_steps.update(
        (item["function_id"], item["step_index"])
        for item in parameters["actions"]
    )
    edited_steps = {
        (item["function_id"], item["step_index"])
        for item in parameters["action_edits"]
    }
    edited_steps.update(
        (item["function_id"], item["step_index"])
        for item in parameters["actions"]
    )
    checker_steps: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for raw in raw_checkers:
        if not isinstance(raw, dict) or set(raw) != {"function_id", "step_index"}:
            raise ValueError("function_checker_selection_contract_invalid")
        function_id = str(raw.get("function_id") or "").strip()
        step_index = raw.get("step_index")
        if function_id not in functions:
            raise ValueError("checker_not_registered_on_function")
        if step_index not in functions[function_id]:
            raise ValueError("checker_not_registered_on_function")
        key = (function_id, int(step_index))
        if key in seen:
            raise ValueError("function_checker_selection_duplicate")
        if key in parameter_steps:
            raise ValueError("checker_action_cannot_use_parameter_binding")
        if key in edited_steps:
            raise ValueError("checker_action_cannot_use_action_edit")
        source_step = facts["steps"][step_index]
        action = source_step["action"]
        if action["tool"] not in {"click", "input_text", "long_press"}:
            raise ValueError(f"checker_action_not_transferable:{step_index}")
        source_target = str(
            (source_step.get("metadata") or {}).get("semantic_target") or ""
        ).strip()
        task_progress = " ".join(
            (
                str(facts["goal"]),
                str(plans[function_id]["name"]),
                str(plans[function_id]["description"]),
            )
        ).casefold()
        if source_target and source_target.casefold() in task_progress:
            raise ValueError("checker_action_is_task_progress")
        formal = [index for index in functions[function_id] if index != step_index]
        _validate_checker_checkpoints(formal, [int(step_index)])
        seen.add(key)
        checker_steps.append(
            {"function_id": function_id, "step_index": int(step_index)}
        )
    return {"checker_steps": checker_steps}


def _validate_checker_draft_for_function(
    value: Any,
    facts: dict[str, Any],
    split: dict[str, Any],
    parameters: dict[str, Any],
    *,
    function_id: str,
) -> dict[str, Any]:
    decision = _validate_checker_draft(value, facts, split, parameters)
    if any(
        item["function_id"] != function_id
        for item in decision["checker_steps"]
    ):
        raise ValueError("function_stage_must_edit_one_function")
    return decision


def _compact_source_actions(facts: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "step_index": index,
            "action": step["action"],
            "source_page": step.get("metadata", {}).get("source_page", {}),
            "after_page": step.get("metadata", {}).get("after_page", {}),
            "source_target": str(
                step.get("metadata", {}).get("semantic_target") or ""
            ),
        }
        for index, step in enumerate(facts["steps"])
    ]


def _draft_split_prompt(
    facts: dict[str, Any],
    *,
    existing_functions: list[dict[str, Any]],
    instruction: str,
) -> str:
    evidence = {
        "goal": facts["goal"],
        "source_actions": _compact_source_actions(facts),
        "existing_function_hints": _function_hints(existing_functions),
        "instruction": str(instruction or "").strip()[:1000],
    }
    return (
        "Edit only the semantic structure of one Function draft. Name and describe "
        "the mandatory complete-RunLog Function. Use a stable lower_snake_case "
        "function_id and a short human-readable name with normal word spacing. The "
        "Function must cover every successful source action in order. Preserve the "
        "full trajectory even when some actions can also act as optional checkers. "
        "Do not split the RunLog or return complete Functions, actions, parameters, "
        "or checker registrations in this stage.\n\nDraft input:\n"
        + json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
    )


def _draft_parameters_prompt(
    facts: dict[str, Any],
    plan: dict[str, Any],
) -> str:
    source_indices = _function_indices(plan, len(facts["steps"]))
    source_actions = [
        _compact_source_actions(facts)[index] for index in source_indices
    ]
    goal = str(facts["goal"]).casefold()
    eligible_parameter_indices: list[int] = []
    eligible_open_app_indices: list[int] = []
    eligible_target_indices: list[int] = []
    for action in source_actions:
        source_action = action["action"]
        source_page = action.get("source_page") or {}
        after_page = action.get("after_page") or {}
        after_package = str(after_page.get("package") or "").strip()
        open_app_candidate = (
            source_action.get("tool") == "click"
            and source_page.get("is_launcher") is True
            and bool(after_package)
            and after_package != str(source_page.get("package") or "").strip()
        )
        if open_app_candidate:
            eligible_open_app_indices.append(action["step_index"])
        elif (
            source_action.get("tool") == "click"
            and str(action.get("source_target") or "").strip()
        ):
            eligible_target_indices.append(action["step_index"])
        value = (
            (source_action.get("args") or {}).get("text")
            if source_action.get("tool") == "input_text"
            else action.get("source_target")
            if source_action.get("tool") == "click"
            else ""
        )
        normalized_value = str(value or "").strip().casefold()
        if normalized_value and normalized_value in goal:
            eligible_parameter_indices.append(action["step_index"])
    evidence = {
        "goal": facts["goal"],
        "function": plan,
        "eligible_open_app_step_indices": eligible_open_app_indices,
        "eligible_set_target_step_indices": eligible_target_indices,
        "eligible_parameter_step_indices": eligible_parameter_indices,
        "source_actions": source_actions,
    }
    return (
        "Edit only source-proven action semantics and parameter declarations on the "
        "one Function shown below. Every returned function_id must exactly match the "
        "shown function_id. step_index is the original RunLog source index and must be "
        f"one of {list(source_indices)}; it is not a local Function index. For a "
        "launcher click whose after_page.package is a "
        "different app, use operation=open_app. Return "
        "open_app edits only for eligible_open_app_step_indices "
        f"{eligible_open_app_indices}. An action already represented as open_app needs "
        "no edit. For a stable visible source_target, use operation=set_target only "
        "for eligible_set_target_step_indices "
        f"{eligible_target_indices}. "
        "Do not return a package, label, target, or value: the compiler copies the "
        "exact package or source target from the listed RunLog evidence after it "
        "validates your operation choice. Bind caller-varying values already "
        "present after those edits only when the value is requested by the goal and "
        "replacing it changes the requested outcome. The compiler derives the binding "
        "target from the validated action: input_text binds text, while a click with a "
        "source-proven set_target edit binds target_description. You may also return "
        "direct actions for an exploration-style draft, but include only the original "
        "step index and a canonical action copied from the evidence; the compiler "
        "rejects any action it cannot prove from that source step. Do not return a path. "
        "A current source-state value clicked only to open a picker or "
        "menu is not a caller parameter. For example, clicking the currently displayed "
        "minute before selecting the requested minute may receive set_target grounding, "
        "but only the requested minute is bound. Every bound source value must appear "
        "directly in the shown goal; a value absent from the goal is source state, not "
        "caller input. Return bindings only for eligible_parameter_step_indices "
        f"{eligible_parameter_indices}; an empty list means no binding is allowed. A "
        "step outside that list may still receive a source-proven action edit. Reusing "
        "one parameter name on multiple "
        "steps is valid only when every bound source value is equal. A time, query, "
        "contact, quantity, or selected visible label explicitly varying with the goal "
        "must be a parameter. Coordinates, packages, waits, and directions are not "
        "parameters. Example: a launcher click from package containing 'launcher' to "
        "after_page.package='com.example.app' becomes an open_app edit; a source_target "
        "'6' selected for an alarm can become set_target plus a binding from "
        "target_description to parameter hour. Return empty lists only when source "
        "evidence proves no edit and the operation has no caller-varying value.\n\n"
        "Draft input:\n"
        + json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
    )


def _draft_checkers_prompt(
    facts: dict[str, Any],
    plan: dict[str, Any],
    parameters: dict[str, Any],
) -> str:
    source_indices = _function_indices(plan, len(facts["steps"]))
    function_id = plan["function_id"]
    unavailable_indices = {
        item["step_index"]
        for key in ("bindings", "action_edits", "actions")
        for item in parameters[key]
        if item["function_id"] == function_id
    }
    task_progress = " ".join(
        (str(facts["goal"]), str(plan["name"]), str(plan["description"]))
    ).casefold()
    eligible_indices = []
    for index in source_indices:
        source_step = facts["steps"][index]
        source_target = str(
            (source_step.get("metadata") or {}).get("semantic_target") or ""
        ).strip()
        if (
            index not in unavailable_indices
            and source_step["action"]["tool"]
            in {"click", "input_text", "long_press"}
            and any(later > index for later in source_indices)
            and not (source_target and source_target.casefold() in task_progress)
        ):
            eligible_indices.append(index)
    evidence = {
        "function": plan,
        "bindings": [
            item
            for item in parameters["bindings"]
            if item["function_id"] == function_id
        ],
        "action_edits": [
            item
            for item in parameters["action_edits"]
            if item["function_id"] == function_id
        ],
        "actions": [
            item
            for item in parameters["actions"]
            if item["function_id"] == function_id
        ],
        "eligible_checker_step_indices": eligible_indices,
        "source_actions": [
            _compact_source_actions(facts)[index] for index in source_indices
        ],
    }
    return (
        "Edit only checker registrations for the one Function shown below. Every "
        "returned function_id must exactly match the shown function_id. step_index is "
        "the original RunLog source index and must be one of "
        f"{list(source_indices)}. Select only from eligible_checker_step_indices "
        f"{eligible_indices}; bound actions, edited actions, unsupported actions, and "
        "actions without a later formal step or whose target names task progress have "
        "already been excluded. Select "
        "an existing source step only when it is optional, safe to skip, has a later "
        "formal action in that Function, and is a transferable click, input_text, or "
        "long_press. Required navigation and terminal actions are not checkers. Do not "
        "write rules or triggers; return only Function and source-step relationships. "
        "Return an empty checker_steps list when none are safe.\n\nDraft input:\n"
        + json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
    )


def _compile_function_draft(
    facts: dict[str, Any],
    split: dict[str, Any],
    parameters: dict[str, Any],
    checkers: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan = split["complete_function"]
    function_id = plan["function_id"]
    function, source_arguments = _compile_draft_function(
        facts,
        metadata={
            key: plan[key] for key in ("function_id", "name", "description")
        },
        source_indices=_function_indices(plan, len(facts["steps"])),
        action_edits=[
            item
            for item in parameters["action_edits"]
            if item["function_id"] == function_id
        ],
        action_overrides=[
            item
            for item in parameters["actions"]
            if item["function_id"] == function_id
        ],
        parameter_bindings=[
            item
            for item in parameters["bindings"]
            if item["function_id"] == function_id
        ],
        checker_indices=[
            item["step_index"]
            for item in checkers["checker_steps"]
            if item["function_id"] == function_id
        ],
    )
    return [function], {function_id: source_arguments}


def _compile_deterministic_function(
    facts: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compile one complete source-grounded Function without model authoring."""

    task_name = str(facts.get("task_name") or "").strip()
    label = task_name or str(facts.get("goal") or "successful_task").strip()
    normalized = re.sub(r"[^a-z0-9]+", "_", label.casefold()).strip("_")
    if not normalized:
        normalized = "successful_task"
    if not normalized[0].isalpha():
        normalized = f"task_{normalized}"
    function_id = f"replay_{normalized}"
    if len(function_id) > 64:
        digest = hashlib.sha256(label.encode("utf-8")).hexdigest()[:10]
        function_id = f"replay_{normalized[:64 - len(digest) - 8]}_{digest}"
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
    _validate_checker_checkpoints(formal_indices, checker_indices)
    local_index = {
        source_index: index for index, source_index in enumerate(formal_indices)
    }
    steps = [
        {
            "step_index": local_index[source_index],
            "source_state_id": facts["steps"][source_index]["before_state_id"],
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
            "source_state_id": facts["steps"][index]["before_state_id"],
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
