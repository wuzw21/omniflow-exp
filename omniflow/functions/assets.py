from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable

from omniflow.core.model import Action, Function, FunctionStep
from omniflow.core.schemas import canonicalize_action, load_canonical_action_schema
from omniflow.core.trajectory import canonicalize_run_log, state_id
from omniflow.runlog import project_androidworld_step_actions
from omniflow.runtime.checker import validate_checker_rule

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
_PARAMETER_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_PARAMETERIZABLE_ACTION_ARGS = {
    "input_text": frozenset({"text"}),
    "open_app": frozenset({"package_name"}),
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
        if not isinstance(step, dict) or set(step) - {
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
    schema = function.input_schema
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError("function_parameters_must_be_object_schema")
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
    def __init__(
        self,
        path: str | Path,
        *,
        seed_functions: Iterable[Function] = (),
        replace_seeded: bool = False,
    ):
        self.path = Path(path)
        self.functions: dict[str, Function] = {}
        self.load_errors: dict[str, str] = {}
        self._load()
        self._seed(seed_functions, replace=replace_seeded)

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

    def put_function(self, value: Function | dict) -> Function:
        function = value if isinstance(value, Function) else parse_function_artifact(value)
        validate_function_artifact(function)
        self.functions[function.id] = function
        self.load_errors.clear()
        self.save()
        return function

    def delete_function(self, function_id: str) -> bool:
        normalized = str(function_id or "").strip()
        if normalized not in self.functions:
            return False
        del self.functions[normalized]
        self.load_errors.clear()
        self.save()
        return True

    def clear_functions(self) -> int:
        deleted = len(self.functions)
        self.functions.clear()
        self.load_errors.clear()
        self.save()
        return deleted

    def reload(self) -> None:
        self.functions = {}
        self.load_errors = {}
        self._load()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": STORE_VERSION,
            "functions": {
                key: value.to_dict() for key, value in sorted(self.functions.items())
            },
        }
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def _load(self) -> None:
        if not self.path.exists():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != STORE_VERSION:
            raise ValueError("unsupported_store_version")
        raw_functions = payload.get("functions")
        if not isinstance(raw_functions, dict):
            raise ValueError("function_store_functions_must_be_object")
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
        self.load_errors = load_errors

    def _seed(
        self,
        seed_functions: Iterable[Function],
        *,
        replace: bool,
    ) -> None:
        changed = False
        for function in seed_functions:
            validate_function_artifact(function)
            if function.id in self.functions and not replace:
                continue
            if self.functions.get(function.id) == function:
                continue
            self.functions[function.id] = function
            changed = True
        if changed:
            self.save()


def compile_runlog_to_store(
    run_log: str | Path | dict[str, Any],
    output_root: str | Path,
    *,
    function_bundle: dict[str, Any] | None = None,
    model: str | None = None,
    client: Any | None = None,
    prompt: str | None = None,
    timeout: float = 120.0,
    source_states: str | Path | dict[str, Any] | None = None,
    state_loader: Any | None = None,
) -> dict[str, Any]:
    """Register strict v2 Functions and their referenced source states."""
    from omniflow.transfer.runtime import (
        TRANSFER_STATE_CATALOG_FILENAME,
        TRANSFER_STATE_CATALOG_VERSION,
        load_transfer_state_catalog,
    )

    root = Path(output_root).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"immutable output version already exists: {root}")

    if isinstance(run_log, dict):
        raw = dict(run_log)
    else:
        value = json.loads(Path(run_log).expanduser().resolve().read_text())
        if not isinstance(value, dict):
            raise ValueError("source_runlog_must_be_object")
        raw = value
    payload = canonicalize_run_log(raw)
    goal = str(payload.get("goal") or "").strip()
    if not goal:
        raise ValueError("successful_source_goal_required")

    steps: list[dict[str, Any]] = []
    recovery_examples: list[dict[str, Any]] = []
    for step in payload["steps"]:
        if not isinstance(step, dict):
            continue
        metadata = step.get("metadata") if isinstance(step.get("metadata"), dict) else {}
        result = step.get("result") if isinstance(step.get("result"), dict) else {}
        if result.get("success") is not True:
            continue
        observation = step["observation"]
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
        if metadata.get("origin") == "checker":
            for action in projected_actions:
                example = {
                    "source_state_id": before_state_id,
                    "action": action,
                    "metadata": {
                        key: metadata[key]
                        for key in ("thinking", "summary")
                        if str(metadata.get(key) or "").strip()
                    },
                }
                trigger = str(metadata.get("checker_trigger") or "").strip()
                if trigger:
                    example["trigger"] = trigger
                recovery_examples.append(example)
            continue
        action_metadata = {
            key: metadata[key]
            for key in ("summary", "thinking", "action_description")
            if str(metadata.get(key) or "").strip()
        }
        for action in projected_actions:
            steps.append(
                {
                    "step_index": len(steps),
                    "before_state_id": before_state_id,
                    "action": action,
                    "result": {"success": True},
                    "after_state_id": after_state_id,
                    "metadata": action_metadata,
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
    usage = {
        "model_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    if model is not None or client is not None or prompt is not None:
        raise ValueError("runtime_function_authoring_removed_use_skill_bundle")
    _ = timeout
    if function_bundle is None:
        raise ValueError("function_bundle_required_from_authoring_skill")
    authored = {
        "reason": "Registered Function bundle produced by the authoring skill.",
        "bundle": json.loads(json.dumps(function_bundle, ensure_ascii=False)),
    }
    if not isinstance(authored, dict) or set(authored) != {"reason", "bundle"}:
        raise ValueError("function_author_response_contract_invalid")
    if not isinstance(authored["reason"], str):
        raise ValueError("function_author_reason_must_be_string")

    bundle = authored["bundle"]
    if bundle is None:
        raise ValueError("functions_required")
    if not isinstance(bundle, dict):
        raise ValueError("function_author_bundle_must_be_object_or_null")
    if set(bundle) != {
        "schema_version",
        "run_id",
        "arguments",
        "functions",
    }:
        raise ValueError("function_bundle_contract_invalid")
    if bundle.get("schema_version") != "omniflow.function-bundle.v2":
        raise ValueError("unsupported_function_bundle_version")
    if str(bundle.get("run_id") or "") != facts["run_id"]:
        raise ValueError("function_bundle_run_id_mismatch")
    raw_functions = bundle.get("functions")
    arguments_by_function = bundle.get("arguments")
    if not isinstance(raw_functions, list) or not raw_functions:
        raise ValueError("function_bundle_functions_required")
    if not isinstance(arguments_by_function, dict):
        raise ValueError("function_bundle_source_arguments_invalid")
    functions = [parse_function_artifact(value) for value in raw_functions]
    _validate_checker_evidence(functions, recovery_examples)
    function_ids = [function.id for function in functions]
    if len(function_ids) != len(set(function_ids)):
        raise ValueError("function_bundle_duplicate_function_id")
    if set(arguments_by_function) - set(function_ids):
        raise ValueError("function_bundle_source_arguments_unknown_function")
    for function in functions:
        raw_arguments = arguments_by_function.get(function.id, {})
        calls = raw_arguments if isinstance(raw_arguments, list) else [raw_arguments]
        if not calls or any(not isinstance(arguments, dict) for arguments in calls):
            raise ValueError("function_bundle_source_arguments_invalid")
        for arguments in calls:
            bound = bind_function(function, arguments)
            _validate_action_grounding(bound, steps)

    if source_states is not None and state_loader is not None:
        raise ValueError("function_source_state_provider_ambiguous")
    referenced_state_ids = _referenced_source_state_ids(functions)
    states: dict[str, dict[str, Any]]
    source_catalog_run_id = facts["run_id"]
    if source_states is not None:
        if isinstance(source_states, (str, Path)):
            source_catalog_path = Path(source_states).expanduser().resolve()
            raw_catalog = json.loads(source_catalog_path.read_text(encoding="utf-8"))
            if not isinstance(raw_catalog, dict):
                raise ValueError("function_source_state_catalog_invalid")
            source_catalog_run_id = str(raw_catalog.get("run_id") or "").strip()
            states = load_transfer_state_catalog(source_catalog_path)
        elif isinstance(source_states, dict):
            raw_states = source_states.get("states")
            if raw_states is None:
                raw_states = source_states
            elif source_states.get("schema_version") != (
                TRANSFER_STATE_CATALOG_VERSION
            ):
                raise ValueError("function_source_state_catalog_invalid")
            if not isinstance(raw_states, dict):
                raise ValueError("function_source_state_catalog_invalid")
            source_catalog_run_id = str(
                source_states.get("run_id") or facts["run_id"]
            ).strip()
            states = {
                str(state_id): _normalize_source_state(value, str(state_id))
                for state_id, value in raw_states.items()
            }
        else:
            raise ValueError("function_source_state_catalog_invalid")
    elif callable(state_loader):
        states = {
            state_id: _normalize_source_state(state_loader(state_id), state_id)
            for state_id in referenced_state_ids
        }
    else:
        raise ValueError("function_source_states_required")
    if source_catalog_run_id != facts["run_id"]:
        raise ValueError("function_source_state_run_id_mismatch")
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
    store_path = root / "store.json"
    store = FunctionStore(store_path)
    for function in functions:
        store.put_function(function)
    transfer_state_catalog_path = root / TRANSFER_STATE_CATALOG_FILENAME
    transfer_state_catalog_path.write_text(
        json.dumps(
            {
                "schema_version": TRANSFER_STATE_CATALOG_VERSION,
                "run_id": facts["run_id"],
                "states": frozen_states,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": "omniflow.androidworld.function-gate.v2",
        "success": True,
        "live_probe_allowed": True,
        "classification": "ready_for_live_probe",
        "reason": authored["reason"],
        "model": None,
        "prompt_sha256": None,
        "store_path": str(store_path),
        "transfer_state_catalog": str(transfer_state_catalog_path),
        "transfer_state_count": len(frozen_states),
        "function_ids": function_ids,
        "function_count": len(function_ids),
        "source_arguments": json.loads(
            json.dumps(arguments_by_function, ensure_ascii=False)
        ),
        **usage,
    }
    (root / "compile_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
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
) -> None:
    expected = [step.action.to_dict() for step in function.steps]
    width = len(expected)
    if any(
        [step["action"] for step in source_steps[start : start + width]] == expected
        for start in range(len(source_steps) - width + 1)
    ):
        return
    raise ValueError(
        "function_action_not_grounded:"
        f"{function.id}:{function.steps[0].step_index}"
    )


def _validate_checker_evidence(
    functions: list[Any],
    recovery_examples: list[dict[str, Any]],
) -> None:
    evidence = [
        {
            "source_state_id": str(example.get("source_state_id") or ""),
            "action": json.dumps(
                canonicalize_action(example.get("action"), replayable_only=True),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "trigger": str(example.get("trigger") or "").strip(),
        }
        for example in recovery_examples
    ]
    for function in functions:
        for rule in function.checker_rules:
            source_state_id = str(rule.get("source_state_id") or "")
            action = json.dumps(
                canonicalize_action(rule.get("action"), replayable_only=True),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            matches = [
                example
                for example in evidence
                if example["source_state_id"] == source_state_id
                and example["action"] == action
            ]
            if not matches:
                raise ValueError("function_checker_rule_missing_recovery_evidence")
            captured_triggers = {
                example["trigger"] for example in matches if example["trigger"]
            }
            if captured_triggers and rule.get("trigger") not in captured_triggers:
                raise ValueError("function_checker_rule_trigger_mismatch")


def edit_function(
    value: dict[str, Any],
    edits: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    original = parse_function_artifact(value).to_dict()
    updated = json.loads(json.dumps(original, ensure_ascii=False))
    steps = updated["steps"]
    changes: list[dict[str, Any]] = []
    deletes: set[int] = set()
    for edit in edits:
        if not isinstance(edit, dict):
            continue
        index = _integer(edit.get("index"), -1)
        if index not in range(len(steps)):
            continue
        action = steps[index]["action"]
        tool = action["tool"]
        expected_tool = str(edit.get("expected_tool") or "").strip()
        if expected_tool and expected_tool != tool:
            continue
        operation = str(edit.get("op") or "").strip().lower()
        if operation == "delete":
            deletes.add(index)
        elif operation == "replace_args":
            patch = edit.get("args")
            if not isinstance(patch, dict) or not patch:
                continue
            canonical = canonicalize_action(
                {"tool": tool, "args": {**action["args"], **patch}},
                replayable_only=True,
                persisted_only=True,
            )
            if canonical["args"] == action["args"]:
                continue
            action["args"] = canonical["args"]
            changes.append(_change("replace_args", index, tool, edit.get("reason")))
    if deletes and (len(deletes) >= len(steps) or updated["bindings"]):
        deletes.clear()
    for index in sorted(deletes, reverse=True):
        tool = steps[index]["action"]["tool"]
        del steps[index]
        reason = next(
            (
                edit.get("reason")
                for edit in edits
                if isinstance(edit, dict)
                and str(edit.get("op") or "").strip().lower() == "delete"
                and _integer(edit.get("index"), -1) == index
            ),
            None,
        )
        changes.append(_change("delete", index, tool, reason))
    for index, step in enumerate(steps):
        step["step_index"] = index
    return parse_function_artifact(updated).to_dict(), changes


def enhance_function(
    value: dict[str, Any],
    run_log: dict[str, Any],
    complete_json: Callable[[str], str],
    *,
    instruction: str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    original = parse_function_artifact(value).to_dict()
    proposal = _json_object(
        complete_json(_enhancement_prompt(original, run_log, instruction=instruction))
    )
    _require_checker_evidence(proposal, run_log)
    updated = json.loads(json.dumps(original, ensure_ascii=False))
    changes: list[dict[str, Any]] = []
    for field, limit in (("name", 80), ("description", 2000)):
        replacement = str(proposal.get(field) or "").strip()[:limit]
        if replacement and replacement != updated[field]:
            updated[field] = replacement
            changes.append({"part": "function", "field": field})
    if "steps" in proposal:
        replacement_steps = _grounded_replacement_steps(proposal["steps"], run_log)
        if replacement_steps != updated["steps"]:
            updated["steps"] = replacement_steps
            updated["input_schema"] = {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            }
            updated["bindings"] = []
            changes.append({"part": "function", "field": "steps"})
    if "parameters" in proposal and _apply_parameters(
        updated,
        proposal["parameters"],
        run_log,
    ):
        changes.append({"part": "function", "field": "parameters"})
    if "checker_rules" in proposal:
        candidate = dict(updated)
        candidate["checker_rules"] = proposal["checker_rules"]
        canonical_rules = parse_function_artifact(candidate).to_dict()["checker_rules"]
        if canonical_rules != updated["checker_rules"]:
            updated["checker_rules"] = canonical_rules
            changes.append({"part": "function", "field": "checker_rules"})
    canonical = parse_function_artifact(updated).to_dict()
    return canonical, changes, "enhanced" if changes else "unchanged"


def _enhancement_prompt(
    function: dict[str, Any],
    run_log: dict[str, Any],
    *,
    instruction: str = "",
) -> str:
    steps = [
        {
            "index": index,
            "tool": step["action"]["tool"],
            "target": str(step["action"]["args"].get("target_description") or "")[:120],
        }
        for index, step in enumerate(function["steps"])
    ]
    run_log_facts = {
        "run_id": str(run_log.get("run_id") or ""),
        "goal": str(run_log.get("goal") or ""),
        "successful_function_steps": _successful_source_steps(run_log),
        "raw_steps": [
            {
                key: step.get(key)
                for key in (
                    "step_index",
                    "before_state_id",
                    "action",
                    "result",
                    "after_state_id",
                    "metadata",
                )
            }
            for step in run_log.get("steps") or ()
            if isinstance(step, dict)
        ],
    }
    brief = {
        "function_id": function["function_id"],
        "name": function["name"],
        "description": function["description"],
        "steps": steps,
        "parameter_candidates": _parameter_candidates(function),
        "run_log": run_log_facts,
        "user_instruction": str(instruction or "").strip()[:2000],
    }
    return f"""
Improve the reusable Android automation Function below for future recall.
Return one JSON object with optional keys: name, description, steps, parameters, and checker_rules.
Describe when to reuse the Function, visible operations, inputs, success signal, and avoid cases.
You may add, remove, modify, or reorder actions when needed to recover the complete reusable
semantic operation. The final steps must be one exact contiguous sequence of successful actions
from the supplied source RunLog. Copy every source_state_id, tool, and argument exactly; never
invent an action or state. Keep function_id unchanged. Use the same language as the current
name/description.
Treat user_instruction as optional enhancement guidance. It may refine semantic naming,
description, action selection, parameter selection, and evidence-backed checker rules, but it
cannot override the RunLog evidence requirements above.

steps is the complete replacement action sequence. Each item has exactly:
{{"source_state_id":"state-id","action":{{"tool":"click","args":{{"x":10,"y":20}}}}}}.
Omit steps only when the current action sequence is already complete.

parameters is an array of semantic input bindings. Each item has exactly:
{{"name":"query","description":"Text to search for","step_index":1,"arg_name":"text"}}.
Only select entries present in the final steps and copy step_index and arg_name exactly.
Choose a stable identifier name and a concise user-facing description. Return parameters=[]
when the recorded value is intentionally fixed. Do not return input_schema or bindings; the
runtime derives them and verifies the successful RunLog evidence.

checker_rules is an ordered array. Each rule has exactly:
{{"schema_version":"omniflow.checker_rule.v1","trigger":"text_contains(\\"跳过广告\\")","source_state_id":"state-id","action":{{"tool":"click","args":{{"x":900,"y":100}}}}}}.
Create a checker only when RunLog metadata explicitly identifies a successful recovery step
(metadata.origin == "checker" and result.success == true). Copy its action and before_state_id.
When metadata.checker_trigger exists, copy it exactly. Otherwise return checker_rules=[].

Function:
{json.dumps(brief, ensure_ascii=False, separators=(",", ":"))}
""".strip()


def _grounded_replacement_steps(
    proposal: Any,
    run_log: dict[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(proposal, list) or not proposal:
        raise ValueError("function_enhancement_steps_invalid")
    steps: list[dict[str, Any]] = []
    for index, raw_step in enumerate(proposal):
        if not isinstance(raw_step, dict) or set(raw_step) - {
            "step_index",
            "source_state_id",
            "action",
        }:
            raise ValueError("function_enhancement_step_contract_invalid")
        steps.append(
            {
                "step_index": index,
                "source_state_id": str(raw_step.get("source_state_id") or "").strip(),
                "action": raw_step.get("action"),
            }
        )
    candidate = {
        "schema_version": FUNCTION_ARTIFACT_VERSION,
        "function_id": "EnhancedFunctionEvidenceCheck",
        "name": "Enhanced Function evidence check",
        "description": "Validate replacement actions against source evidence.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "bindings": [],
        "steps": steps,
        "checker_rules": [],
        "agent_visible": False,
    }
    function = parse_function_artifact(candidate)
    evidence = _successful_source_steps(run_log)
    expected = [
        (step.source_state_id, step.action.to_dict())
        for step in function.steps
    ]
    width = len(expected)
    if not any(
        [
            (str(step.get("source_state_id") or ""), step["action"])
            for step in evidence[start : start + width]
        ]
        == expected
        for start in range(len(evidence) - width + 1)
    ):
        raise ValueError(
            "function_action_not_grounded:"
            f"{function.id}:{function.steps[0].step_index}"
        )
    return [step.to_dict() for step in function.steps]


def _successful_source_steps(run_log: dict[str, Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for raw_step in run_log.get("steps") or ():
        if not isinstance(raw_step, dict):
            continue
        result = raw_step.get("result")
        metadata = raw_step.get("metadata")
        if not isinstance(result, dict) or result.get("success") is not True:
            continue
        if isinstance(metadata, dict) and metadata.get("origin") == "checker":
            continue
        observation = raw_step.get("observation")
        action = raw_step.get("action")
        if isinstance(observation, dict):
            source_state_id = state_id(observation)
            projected = project_androidworld_step_actions(raw_step)
        elif isinstance(action, dict) and set(action) == {"tool", "args"}:
            source_state_id = str(raw_step.get("before_state_id") or "").strip()
            projected = [canonicalize_action(action, replayable_only=True)]
        else:
            continue
        evidence.extend(
            {
                "source_state_id": source_state_id,
                "action": projected_action,
            }
            for projected_action in projected
        )
    return evidence


def _parameter_candidates(function: dict[str, Any]) -> list[dict[str, Any]]:
    bound_targets = {
        str(binding.get("target") or "")
        for binding in function.get("bindings") or ()
        if isinstance(binding, dict)
    }
    candidates: list[dict[str, Any]] = []
    for step_index, step in enumerate(function.get("steps") or ()):
        if not isinstance(step, dict):
            continue
        action = step.get("action")
        if not isinstance(action, dict):
            continue
        tool = str(action.get("tool") or "")
        args = action.get("args")
        if not isinstance(args, dict):
            continue
        for arg_name in _PARAMETERIZABLE_ACTION_ARGS.get(tool, ()):
            target = f"$.steps[{step_index}].action.args.{arg_name}"
            value = args.get(arg_name)
            if target in bound_targets or not isinstance(value, str) or not value:
                continue
            candidates.append(
                {
                    "step_index": step_index,
                    "tool": tool,
                    "arg_name": arg_name,
                    "recorded_value": value,
                }
            )
    return candidates


def _apply_parameters(
    function: dict[str, Any],
    proposal: Any,
    run_log: dict[str, Any],
) -> bool:
    if not isinstance(proposal, list):
        raise ValueError("function_enhancement_parameters_invalid")
    candidates = {
        (candidate["step_index"], candidate["arg_name"]): candidate
        for candidate in _parameter_candidates(function)
    }
    schema = function["input_schema"]
    properties = schema["properties"]
    required = schema["required"]
    bindings = function["bindings"]
    existing_names = set(properties)
    changed = False
    for parameter in proposal:
        if not isinstance(parameter, dict) or set(parameter) - {
            "name",
            "description",
            "step_index",
            "arg_name",
        }:
            raise ValueError("function_enhancement_parameter_contract_invalid")
        name = str(parameter.get("name") or "").strip()
        description = str(parameter.get("description") or "").strip()[:240]
        step_index = _integer(parameter.get("step_index"), -1)
        arg_name = str(parameter.get("arg_name") or "").strip()
        candidate = candidates.get((step_index, arg_name))
        if candidate is None:
            raise ValueError("function_enhancement_parameter_target_invalid")
        if _PARAMETER_NAME.fullmatch(name) is None or name in existing_names:
            raise ValueError("function_enhancement_parameter_name_invalid")
        step = function["steps"][step_index]
        value = step["action"]["args"][arg_name]
        if not _has_parameter_evidence(step, arg_name, value, run_log):
            raise ValueError("function_enhancement_parameter_evidence_missing")
        definition: dict[str, Any] = {"type": "string"}
        if description:
            definition["description"] = description
        properties[name] = definition
        required.append(name)
        bindings.append(
            {
                "source": f"$.arguments.{name}",
                "target": f"$.steps[{step_index}].action.args.{arg_name}",
            }
        )
        step["action"]["args"][arg_name] = ""
        existing_names.add(name)
        changed = True
    return changed


def _has_parameter_evidence(
    function_step: dict[str, Any],
    arg_name: str,
    value: Any,
    run_log: dict[str, Any],
) -> bool:
    for raw_step in _successful_source_steps(run_log):
        action = raw_step["action"]
        args = action.get("args")
        if not isinstance(args, dict):
            continue
        if (
            str(raw_step.get("source_state_id") or "")
            == str(function_step.get("source_state_id") or "")
            and str(action.get("tool") or "")
            == str(function_step.get("action", {}).get("tool") or "")
            and args.get(arg_name) == value
        ):
            return True
    return False


def _require_checker_evidence(
    proposal: dict[str, Any],
    run_log: dict[str, Any],
) -> None:
    if "checker_rules" not in proposal:
        return
    evidence: list[tuple[str, dict[str, Any], str]] = []
    for step in run_log.get("steps") or ():
        if not isinstance(step, dict):
            continue
        metadata = step.get("metadata")
        result = step.get("result")
        if not isinstance(metadata, dict) or not isinstance(result, dict):
            continue
        if metadata.get("origin") != "checker" or result.get("success") is not True:
            continue
        state_id = str(step.get("before_state_id") or "").strip()
        if not state_id:
            continue
        evidence.append(
            (
                state_id,
                canonicalize_action(step.get("action"), replayable_only=True),
                str(metadata.get("checker_trigger") or "").strip(),
            )
        )
    candidate = {
        "schema_version": "omniflow.function.v2",
        "function_id": "EvidenceCheck",
        "name": "Evidence check",
        "description": "Validate proposed checker rules.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "bindings": [],
        "steps": [
            {
                "step_index": 0,
                "source_state_id": "evidence",
                "action": {"tool": "wait", "args": {"duration_ms": 1}},
            }
        ],
        "checker_rules": proposal.get("checker_rules"),
        "agent_visible": False,
    }
    rules = parse_function_artifact(candidate).to_dict()["checker_rules"]
    for rule in rules:
        matches = [
            item
            for item in evidence
            if item[0] == rule["source_state_id"] and item[1] == rule["action"]
        ]
        if not matches:
            raise ValueError("checker_rule_missing_recovery_evidence")
        captured = {item[2] for item in matches if item[2]}
        if captured and rule["trigger"] not in captured:
            raise ValueError("checker_rule_trigger_mismatch")


def _json_object(raw: str) -> dict[str, Any]:
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("function_enhancement_json_missing")
    value = json.loads(raw[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("function_enhancement_json_invalid")
    return value


def _change(operation: str, index: int, tool: str, reason: Any) -> dict[str, Any]:
    value = {
        "part": "action",
        "field": operation,
        "op": operation,
        "step_index": index,
        "tool": tool,
    }
    text = str(reason or "").strip()
    if text:
        value["reason"] = text
    return value


def _integer(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
