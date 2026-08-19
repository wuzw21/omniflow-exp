"""Convert historical Function JSON into the current one-Function Store."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Iterable

from omniflow.core.trajectory import state_id
from omniflow.functions.assets import (
    FUNCTION_ARTIFACT_VERSION,
    STORE_VERSION,
    parse_function_artifact,
    save_function,
    write_function_store,
)
from omniflow.runlog import (
    import_run_log_evidence,
    project_androidworld_step_actions,
)

LEGACY_BUNDLE_VERSION = "omniflow.function-bundle.v2"
LEGACY_CATALOG_VERSION = "omniflow.function-asset-catalog.v1"
_SOURCE_PATH = re.compile(
    r"^\$\.arguments(?P<tail>(?:\.[A-Za-z_][A-Za-z0-9_]*|\[\d+\])+)$"
)
_TARGET_PATH = re.compile(
    r"^\$\.actions\[(?P<index>\d+)]\.arguments"
    r"(?P<tail>(?:\.[A-Za-z_][A-Za-z0-9_]*|\[\d+\])+)$"
)
_PATH_TOKEN = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)]")
_IGNORED_ARGS = {
    "app_name",
    "clear_text",
    "post_action_wait_s",
    "post_wait_s",
    "wait_after_s",
}
_SOURCE_CENTER_TOLERANCE = 5.0


def migrate_function_store(
    input_path: str | Path,
    output: str | Path,
    *,
    source_run_log: str | Path | None = None,
    transfer_states: str | Path | None = None,
    force: bool = False,
    dry_run: bool = False,
    function_id: str | None = None,
) -> dict[str, Any]:
    """Migrate one old Store or bundle without changing the input file.

    A historical bundle needs its successful source RunLog because old actions
    did not carry source state IDs. A historical Store with multiple Functions
    is split into one current Store per Function when every Function has one
    source call; incomplete evidence is rejected instead of guessed.
    """

    input_file = Path(input_path).expanduser().resolve()
    if not input_file.is_file():
        raise FileNotFoundError(f"function_migration_input_missing:{input_file}")
    payload = _read_object(input_file)
    schema_version = str(payload.get("schema_version") or "")
    if schema_version == STORE_VERSION:
        functions, source_calls = _read_current_store(payload)
        if function_id is not None:
            selected = [
                function
                for function in functions
                if function["function_id"] == str(function_id)
            ]
            if len(selected) != 1:
                raise ValueError(f"function_migration_function_missing:{function_id}")
            functions = selected
            source_calls = [
                call
                for call in source_calls
                if call["function_id"] == str(function_id)
            ]
        return _write_current_stores(
            input_file=input_file,
            functions=functions,
            source_calls=source_calls,
            output=output,
            source_run_log=source_run_log,
            transfer_states=transfer_states,
            force=force,
            dry_run=dry_run,
        )
    if schema_version == LEGACY_BUNDLE_VERSION:
        if source_run_log is None:
            raise ValueError("legacy_bundle_source_run_log_required")
        return _migrate_legacy_bundle(
            input_file=input_file,
            bundle=payload,
            source_run_log=Path(source_run_log).expanduser().resolve(),
            output=output,
            force=force,
            dry_run=dry_run,
        )
    raise ValueError(
        "unsupported_function_json_version:"
        f"{schema_version or 'missing'}:expected={STORE_VERSION}|{LEGACY_BUNDLE_VERSION}"
    )


def _read_current_store(
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_functions = payload.get("functions")
    if not isinstance(raw_functions, dict) or not raw_functions:
        raise ValueError("function_store_functions_must_be_object")
    functions: list[dict[str, Any]] = []
    for key, value in sorted(raw_functions.items()):
        if not isinstance(value, dict):
            raise ValueError(f"function_store_function_invalid:{key}")
        function = parse_function_artifact(value)
        if str(key) != function.id:
            raise ValueError(f"function_store_key_mismatch:{key}")
        functions.append(function.to_dict())

    raw_calls = payload.get("source_calls") or []
    if not isinstance(raw_calls, list):
        raise ValueError("function_store_source_calls_invalid")
    source_calls = []
    for call in raw_calls:
        if (
            not isinstance(call, dict)
            or set(call) != {"function_id", "arguments"}
            or not isinstance(call.get("arguments"), dict)
        ):
            raise ValueError("function_store_source_calls_invalid")
        source_calls.append(
            {
                "function_id": str(call.get("function_id") or ""),
                "arguments": deepcopy(call["arguments"]),
            }
        )
    function_ids = {function["function_id"] for function in functions}
    if any(call["function_id"] not in function_ids for call in source_calls):
        raise ValueError("function_store_source_call_function_missing")
    return functions, source_calls


def _write_current_stores(
    *,
    input_file: Path,
    functions: list[dict[str, Any]],
    source_calls: list[dict[str, Any]],
    output: str | Path,
    source_run_log: str | Path | None,
    transfer_states: str | Path | None,
    force: bool,
    dry_run: bool,
) -> dict[str, Any]:
    output_path = Path(output).expanduser().resolve()
    if len(functions) == 1 and output_path.suffix.lower() == ".json":
        destinations = [(functions[0], output_path)]
    else:
        if output_path.suffix.lower() == ".json":
            raise ValueError("function_migration_multiple_functions_need_directory")
        destinations = [
            (function, output_path / function["function_id"] / "function_store.json")
            for function in functions
        ]

    state_source = (
        Path(transfer_states).expanduser().resolve()
        if transfer_states is not None
        else input_file.with_name("transfer_states.json")
    )
    state_payload = _read_transfer_states(state_source) if state_source.is_file() else None
    source_log = (
        Path(source_run_log).expanduser().resolve()
        if source_run_log is not None
        else input_file.with_name("run_log.json")
    )
    if not source_log.is_file():
        raise FileNotFoundError(f"function_migration_source_run_log_missing:{source_log}")
    reports: list[dict[str, Any]] = []
    for function, destination in destinations:
        matching_calls = [
            call
            for call in source_calls
            if call.get("function_id") == function["function_id"]
        ]
        if len(matching_calls) != 1:
            raise ValueError(
                "function_migration_source_call_required:"
                f"{function['function_id']}:count={len(matching_calls)}"
            )
        _refuse_existing(destination, force)
        if state_payload is None and function["steps"]:
            raise FileNotFoundError(
                f"function_migration_transfer_states_missing:{state_source}"
            )
        if state_payload is not None:
            referenced = {
                str(item.get("source_state_id") or "")
                for item in (*function.get("steps", []), *function.get("checker_rules", []))
                if isinstance(item, dict) and str(item.get("source_state_id") or "")
            }
            missing = sorted(referenced - set(state_payload["states"]))
            if missing:
                raise ValueError(
                    "function_migration_transfer_states_missing:" + ",".join(missing)
                )
        _refuse_existing(destination.with_name("run_log.json"), force)
        if state_payload is not None:
            _refuse_existing(destination.with_name("transfer_states.json"), force)
    if dry_run:
        return {
            "schema_version": "omniflow.function-store-migration.v1",
            "input": str(input_file),
            "source_schema_version": STORE_VERSION,
            "dry_run": True,
            "stores": [
                {
                    "function_id": function["function_id"],
                    "store_path": str(destination),
                    "run_log_path": str(destination.with_name("run_log.json")),
                    "transfer_states_path": str(destination.with_name("transfer_states.json")),
                }
                for function, destination in destinations
            ],
        }
    for function, destination in destinations:
        matching_calls = [
            call
            for call in source_calls
            if call.get("function_id") == function["function_id"]
        ]
        write_function_store(
            destination,
            [function],
            matching_calls,
        )
        if state_payload is not None:
            _write_filtered_transfer_states(destination, function, state_payload, force)
        _copy_artifact(source_log, destination.with_name("run_log.json"), force)
        reports.append(
            {
                "function_id": function["function_id"],
                "store_path": str(destination),
                "run_log_path": str(destination.with_name("run_log.json")),
                "transfer_states_path": str(destination.with_name("transfer_states.json")),
            }
        )
    return {
        "schema_version": "omniflow.function-store-migration.v1",
        "input": str(input_file),
        "source_schema_version": STORE_VERSION,
        "stores": reports,
    }


def _migrate_legacy_bundle(
    *,
    input_file: Path,
    bundle: dict[str, Any],
    source_run_log: Path,
    output: str | Path,
    force: bool,
    dry_run: bool,
) -> dict[str, Any]:
    if not source_run_log.is_file():
        raise FileNotFoundError(f"legacy_bundle_source_run_log_missing:{source_run_log}")
    if bundle.get("source_success") is not True:
        raise ValueError("legacy_bundle_source_not_successful")
    source_payload = _read_object(source_run_log)
    run_log, _ = import_run_log_evidence(
        source_payload,
        evidence_root=source_run_log.parent,
    )
    if run_log.get("status") != "succeeded" or run_log.get("success") is not True:
        raise ValueError("legacy_bundle_source_run_log_not_successful")
    if str(bundle.get("source_run_id") or "") != str(run_log.get("run_id") or ""):
        raise ValueError("legacy_bundle_source_run_id_mismatch")

    source_steps = _source_steps(run_log)
    source_arguments = bundle.get("source_arguments") or {}
    if not isinstance(source_arguments, dict):
        raise ValueError("legacy_bundle_source_arguments_invalid")
    raw_functions = bundle.get("functions")
    if not isinstance(raw_functions, list) or not raw_functions:
        raise ValueError("legacy_bundle_functions_required")

    converted: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for raw_function in raw_functions:
        converted.append(
            _convert_legacy_function(
                raw_function,
                source_arguments=source_arguments,
                source_steps=source_steps,
            )
        )

    output_path = Path(output).expanduser().resolve()
    if len(converted) == 1 and output_path.suffix.lower() == ".json":
        destinations = [(converted[0], output_path)]
    else:
        if output_path.suffix.lower() == ".json":
            raise ValueError("function_migration_multiple_functions_need_directory")
        destinations = [
            (item, output_path / item[0]["function_id"] / "function_store.json")
            for item in converted
        ]

    reports: list[dict[str, Any]] = []
    for (function, arguments), destination in destinations:
        _refuse_existing(destination, force)
        _refuse_existing(destination.with_name("run_log.json"), force)
        _refuse_existing(destination.with_name("transfer_states.json"), force)
        if dry_run:
            reports.append(
                {
                    "function_id": function["function_id"],
                    "store_path": str(destination),
                    "run_log_path": str(destination.with_name("run_log.json")),
                    "transfer_states_path": str(destination.with_name("transfer_states.json")),
                }
            )
            continue
        save_function(
            source_run_log,
            destination,
            functions=[function],
            arguments={function["function_id"]: arguments},
        )
        _copy_artifact(source_run_log, destination.with_name("run_log.json"), force)
        reports.append(
            {
                "function_id": function["function_id"],
                "store_path": str(destination),
                "run_log_path": str(destination.with_name("run_log.json")),
                "transfer_states_path": str(destination.with_name("transfer_states.json")),
            }
        )
    return {
        "schema_version": "omniflow.function-store-migration.v1",
        "input": str(input_file),
        "source_schema_version": LEGACY_BUNDLE_VERSION,
        "source_run_log": str(source_run_log),
        "dry_run": dry_run,
        "stores": reports,
    }


def _source_steps(run_log: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw_step in run_log.get("steps") or ():
        if not isinstance(raw_step, dict):
            continue
        if (raw_step.get("result") or {}).get("success") is not True:
            continue
        observation = raw_step.get("observation")
        if isinstance(observation, dict):
            actions = project_androidworld_step_actions(raw_step)
            source_state_id = state_id(observation)
        else:
            actions = [raw_step.get("action")]
            source_state_id = str(raw_step.get("before_state_id") or "").strip()
        if not source_state_id:
            continue
        for action in actions:
            if isinstance(action, dict):
                result.append(
                    {
                        "source_state_id": source_state_id,
                        "action": deepcopy(action),
                    }
                )
    if not result:
        raise ValueError("legacy_bundle_source_actions_required")
    return result


def _convert_legacy_function(
    raw_function: Any,
    *,
    source_arguments: dict[str, Any],
    source_steps: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(raw_function, dict):
        raise ValueError("legacy_function_invalid")
    function_id = str(raw_function.get("function_id") or raw_function.get("id") or "").strip()
    if not function_id:
        raise ValueError("legacy_function_id_required")
    arguments = deepcopy(source_arguments.get(function_id, {}))
    if not isinstance(arguments, dict):
        raise ValueError(f"legacy_function_source_arguments_invalid:{function_id}")
    raw_actions = raw_function.get("actions")
    bindings = raw_function.get("bindings")
    if not isinstance(raw_actions, list) or not raw_actions:
        raise ValueError(f"legacy_function_actions_required:{function_id}")
    if not isinstance(bindings, list):
        raise ValueError(f"legacy_function_bindings_invalid:{function_id}")
    actions = deepcopy(raw_actions)
    _materialize_bindings(actions, arguments, bindings)
    expected, mapping = _expanded_actions(actions)
    start = _unique_alignment(expected, source_steps, function_id)
    matched = source_steps[start : start + len(expected)]
    steps = [
        {
            "step_index": index,
            "source_state_id": item["source_state_id"],
            "action": deepcopy(item["action"]),
        }
        for index, item in enumerate(matched)
    ]
    current_bindings = []
    for binding in bindings:
        source = str(binding.get("source") or "")
        target = str(binding.get("target") or "")
        source_match = _SOURCE_PATH.fullmatch(source)
        target_match = _TARGET_PATH.fullmatch(target)
        if source_match is None or target_match is None:
            raise ValueError(f"legacy_function_binding_path_invalid:{function_id}")
        legacy_index = int(target_match.group("index"))
        candidates = mapping.get(legacy_index, [])
        target_tokens = _path_tokens(target_match.group("tail"))
        matching = []
        for relative_index in candidates:
            try:
                _read_path(steps[relative_index]["action"]["args"], target_tokens)
            except (IndexError, KeyError, TypeError):
                continue
            matching.append(relative_index)
        if len(matching) != 1:
            raise ValueError(f"legacy_function_binding_target_ambiguous:{function_id}")
        relative_index = matching[0]
        _write_path(
            steps[relative_index]["action"]["args"],
            target_tokens,
            _read_path({"arguments": arguments}, _path_tokens(".arguments" + source_match.group("tail"))),
        )
        current_bindings.append(
            {
                "source": source,
                "target": f"$.steps[{relative_index}].action.args{target_match.group('tail')}",
            }
        )
    name = str(raw_function.get("name") or raw_function.get("description") or "").strip()
    description = str(raw_function.get("description") or "").strip()
    input_schema = raw_function.get("input_schema")
    if input_schema is None:
        input_schema = raw_function.get("parameters")
    if not name or not description or not isinstance(input_schema, dict):
        raise ValueError(f"legacy_function_semantics_invalid:{function_id}")
    checker_rules = raw_function.get("checker_rules") or []
    if not isinstance(checker_rules, list):
        raise ValueError(f"legacy_function_checker_rules_invalid:{function_id}")
    return (
        {
            "schema_version": FUNCTION_ARTIFACT_VERSION,
            "function_id": function_id,
            "name": name,
            "description": description,
            "input_schema": deepcopy(input_schema),
            "bindings": current_bindings,
            "steps": steps,
            "checker_rules": deepcopy(checker_rules),
            "agent_visible": raw_function.get("agent_visible") is not False,
        },
        arguments,
    )


def _materialize_bindings(actions: list[Any], arguments: dict[str, Any], bindings: list[Any]) -> None:
    for binding in bindings:
        if not isinstance(binding, dict):
            raise ValueError("legacy_function_binding_invalid")
        source = str(binding.get("source") or "")
        target = str(binding.get("target") or "")
        source_match = _SOURCE_PATH.fullmatch(source)
        target_match = _TARGET_PATH.fullmatch(target)
        if source_match is None or target_match is None:
            raise ValueError("legacy_function_binding_path_invalid")
        action_index = int(target_match.group("index"))
        try:
            value = _read_path({"arguments": arguments}, _path_tokens(".arguments" + source_match.group("tail")))
            _write_path(actions[action_index]["arguments"], _path_tokens(target_match.group("tail")), deepcopy(value))
        except (IndexError, KeyError, TypeError) as error:
            raise ValueError("legacy_function_binding_evidence_missing") from error


def _expanded_actions(actions: list[Any]) -> tuple[list[dict[str, Any]], dict[int, list[int]]]:
    expanded: list[dict[str, Any]] = []
    mapping: dict[int, list[int]] = {}
    for index, raw_action in enumerate(actions):
        if not isinstance(raw_action, dict):
            raise ValueError("legacy_function_action_invalid")
        tool = str(raw_action.get("tool") or "").strip()
        args = raw_action.get("arguments")
        if not tool or not isinstance(args, dict):
            raise ValueError("legacy_function_action_invalid")
        if tool == "input_text" and "x" in args and "y" in args:
            converted = [
                {"tool": "click", "args": {}},
                {"tool": "input_text", "args": {key: value for key, value in args.items() if key not in _IGNORED_ARGS | {"x", "y"}}},
            ]
        elif tool == "press_back":
            converted = [{"tool": "press_key", "args": {"key": "back"}}]
        elif tool == "wait" and "time_s" in args:
            converted = [{"tool": "wait", "args": {"duration_ms": round(float(args["time_s"]) * 1000)}}]
        else:
            converted = [{"tool": tool, "args": {key: value for key, value in args.items() if key not in _IGNORED_ARGS}}]
        mapping[index] = list(range(len(expanded), len(expanded) + len(converted)))
        expanded.extend(converted)
    return expanded, mapping


def _unique_alignment(expected: list[dict[str, Any]], source_steps: list[dict[str, Any]], function_id: str) -> int:
    matches = [
        start
        for start in range(len(source_steps) - len(expected) + 1)
        if all(_actions_match(item, source_steps[start + offset]["action"]) for offset, item in enumerate(expected))
    ]
    if len(matches) != 1:
        raise ValueError(f"legacy_function_source_alignment_invalid:{function_id}:{matches}")
    return matches[0]


def _actions_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    if expected.get("tool") != actual.get("tool"):
        return False
    expected_args = expected.get("args")
    actual_args = actual.get("args")
    if not isinstance(expected_args, dict) or not isinstance(actual_args, dict):
        return False
    for key, value in expected_args.items():
        if key == "app_name":
            continue
        if key not in actual_args:
            if expected["tool"] == "open_app" and key == "package_name":
                continue
            return False
        actual_value = actual_args[key]
        if isinstance(value, (int, float)) and isinstance(actual_value, (int, float)):
            if abs(float(value) - float(actual_value)) > _SOURCE_CENTER_TOLERANCE:
                return False
        elif value != actual_value:
            return False
    return True


def _read_transfer_states(path: Path) -> dict[str, Any]:
    payload = _read_object(path)
    if payload.get("schema_version") != "omniflow.transfer-state-catalog.v1":
        raise ValueError(f"function_migration_transfer_states_invalid:{path}")
    if not isinstance(payload.get("states"), dict):
        raise ValueError(f"function_migration_transfer_states_invalid:{path}")
    return payload


def _write_filtered_transfer_states(
    store_path: Path,
    function: dict[str, Any],
    payload: dict[str, Any],
    force: bool,
) -> None:
    destination = store_path.with_name("transfer_states.json")
    _refuse_existing(destination, force)
    referenced = {
        str(item.get("source_state_id") or "")
        for item in (*function.get("steps", []), *function.get("checker_rules", []))
        if isinstance(item, dict) and str(item.get("source_state_id") or "")
    }
    states = payload["states"]
    missing = sorted(referenced - set(states))
    if missing:
        raise ValueError("function_migration_transfer_states_missing:" + ",".join(missing))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": payload["schema_version"],
                "run_id": payload.get("run_id"),
                "states": {key: states[key] for key in sorted(referenced)},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def _copy_artifact(source: Path, destination: Path, force: bool) -> None:
    """Copy evidence atomically so a failed migration cannot leave a partial file."""

    if not source.is_file():
        raise FileNotFoundError(f"function_migration_artifact_missing:{source}")
    _refuse_existing(destination, force)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:8]
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{suffix}.tmp")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _migrate_canonical_store(
    input_path: Path,
    output_root: Path,
    *,
    task: str,
    source_run_log: Path | None,
    transfer_states: Path | None,
    environment: str,
    device: str,
    method: str,
    attempt: str,
    force: bool,
    dry_run: bool,
) -> list[dict[str, Any]]:
    """Write one canonical attempt per Function, including split old Stores."""

    payload = _read_object(input_path)
    is_current_store = payload.get("schema_version") == STORE_VERSION
    functions = payload.get("functions") if is_current_store else None
    if not isinstance(functions, dict) or len(functions) <= 1:
        result = migrate_function_store(
            input_path,
            _canonical_store_path(
                output_root, environment, task, device, method, attempt
            ),
            source_run_log=source_run_log,
            transfer_states=transfer_states,
            force=force,
            dry_run=dry_run,
        )
        return result.get("stores", [])

    stores: list[dict[str, Any]] = []
    for function_id in sorted(functions):
        function_attempt = f"{attempt}_{_safe_path_name(function_id)}"
        result = migrate_function_store(
            input_path,
            _canonical_store_path(
                output_root, environment, task, device, method, function_attempt
            ),
            source_run_log=source_run_log,
            transfer_states=transfer_states,
            force=force,
            dry_run=dry_run,
            function_id=function_id,
        )
        stores.extend(result.get("stores", []))
    return stores


def _canonical_store_path(
    output_root: Path,
    environment: str,
    task: str,
    device: str,
    method: str,
    attempt: str,
) -> Path:
    return (
        output_root / environment / task / device / "function" / method / attempt
        / "function_store.json"
    )


def _safe_path_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return normalized or "function"


def migrate_function_catalog(
    input_path: str | Path,
    output_root: str | Path,
    *,
    environment: str = "androidworld",
    device: str = "source5554",
    method: str = "function_authoring",
    attempt_id: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    seen_store_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Migrate Stores referenced by a historical catalog, one task at a time."""

    catalog_path = Path(input_path).expanduser().resolve()
    catalog = _read_object(catalog_path)
    if catalog.get("schema_version") != LEGACY_CATALOG_VERSION:
        raise ValueError(f"function_migration_catalog_invalid:{catalog_path}")
    tasks = catalog.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError(f"function_migration_catalog_tasks_invalid:{catalog_path}")
    root = Path(output_root).expanduser().resolve()
    seen = seen_store_ids if seen_store_ids is not None else set()
    reports: list[dict[str, Any]] = []
    for task_name, raw_item in sorted(tasks.items()):
        if not isinstance(raw_item, dict):
            reports.append({
                "task": str(task_name),
                "status": "blocked",
                "reason": "catalog_task_invalid",
            })
            continue
        task = str(raw_item.get("task") or task_name).strip()
        try:
            store_path = _catalog_path(catalog_path, raw_item, "store_path")
            store_identity = (
                f"sha256:{_sha256(store_path)}"
                if store_path.is_file()
                else f"path:{store_path}"
            )
            if store_identity in seen:
                reports.append({
                    "task": task,
                    "status": "duplicate",
                    "source": str(store_path),
                })
                continue
            source_run_log = _catalog_path(catalog_path, raw_item, "source_run_log")
            transfer_states = _catalog_path(
                catalog_path, raw_item, "transfer_states_path"
            )
            attempt = str(
                raw_item.get("attempt_id")
                or attempt_id
                or f"migration_{store_identity.split(':', 1)[1][:12]}"
            )
            stores = _migrate_canonical_store(
                store_path,
                root,
                task=task,
                source_run_log=source_run_log,
                transfer_states=transfer_states,
                environment=environment,
                device=device,
                method=method,
                attempt=attempt,
                force=force,
                dry_run=dry_run,
            )
            seen.add(store_identity)
            reports.append({
                "task": task,
                "status": "converted",
                "source": str(store_path),
                "attempt_id": attempt,
                "stores": stores,
            })
        except (FileExistsError, FileNotFoundError, TypeError, ValueError) as error:
            reports.append({
                "task": task,
                "status": "blocked",
                "reason": str(error) or type(error).__name__,
            })
    return {
        "schema_version": "omniflow.function-store-batch-migration.v1",
        "source_schema_version": LEGACY_CATALOG_VERSION,
        "input": str(catalog_path),
        "output_root": str(root),
        "dry_run": dry_run,
        "counts": _migration_counts(reports),
        "tasks": reports,
    }


def scan_function_json_root(
    input_root: str | Path,
    output_root: str | Path,
    *,
    environment: str = "androidworld",
    device: str = "source5554",
    method: str = "function_authoring",
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Classify and migrate every known historical Function JSON below a root."""

    source_root = Path(input_root).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"function_migration_input_root_missing:{source_root}")
    candidates: list[tuple[Path, str]] = []
    for path in sorted(source_root.rglob("*.json")):
        try:
            payload = _read_object(path)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            continue
        schema_version = str(payload.get("schema_version") or "")
        if schema_version in {
            STORE_VERSION,
            LEGACY_BUNDLE_VERSION,
            LEGACY_CATALOG_VERSION,
        }:
            candidates.append((path, schema_version))

    reports: list[dict[str, Any]] = []
    referenced_stores: set[Path] = set()
    seen_catalog_digests: set[str] = set()
    seen_store_ids: set[str] = set()
    for path, schema_version in candidates:
        if schema_version != LEGACY_CATALOG_VERSION:
            continue
        catalog_digest = _sha256(path)
        if catalog_digest in seen_catalog_digests:
            continue
        seen_catalog_digests.add(catalog_digest)
        try:
            catalog = _read_object(path)
            tasks = catalog.get("tasks")
            if isinstance(tasks, dict):
                for item in tasks.values():
                    if isinstance(item, dict) and item.get("store_path"):
                        referenced_stores.add(_catalog_path(path, item, "store_path"))
            reports.append(
                migrate_function_catalog(
                    path,
                    output_root,
                    environment=environment,
                    device=device,
                    method=method,
                    force=force,
                    dry_run=dry_run,
                    seen_store_ids=seen_store_ids,
                )
            )
        except (FileExistsError, FileNotFoundError, TypeError, ValueError) as error:
            reports.append({
                "schema_version": "omniflow.function-store-batch-migration.v1",
                "source_schema_version": schema_version,
                "input": str(path),
                "dry_run": dry_run,
                "counts": {"converted": 0, "blocked": 1, "stores": 0},
                "tasks": [{"status": "blocked", "reason": str(error)}],
            })

    direct_reports: list[dict[str, Any]] = []
    seen_direct_ids: set[str] = set()
    for path, schema_version in candidates:
        if schema_version == LEGACY_CATALOG_VERSION or path.resolve() in referenced_stores:
            continue
        direct_identity = _sha256(path)
        if direct_identity in seen_direct_ids:
            direct_reports.append({
                "input": str(path),
                "source_schema_version": schema_version,
                "status": "duplicate",
            })
            continue
        task = _task_name_for_path(path)
        attempt = f"migration_{direct_identity[:12]}"
        source_run_log = path.with_name("run_log.json")
        try:
            stores = _migrate_canonical_store(
                path,
                Path(output_root).expanduser().resolve(),
                task=task,
                source_run_log=source_run_log if source_run_log.is_file() else None,
                transfer_states=(
                    path.with_name("transfer_states.json")
                    if path.with_name("transfer_states.json").is_file()
                    else None
                ),
                environment=environment,
                device=device,
                method=method,
                attempt=attempt,
                force=force,
                dry_run=dry_run,
            )
            seen_direct_ids.add(direct_identity)
            direct_reports.append({
                "input": str(path),
                "source_schema_version": schema_version,
                "status": "converted",
                "task": task,
                "stores": stores,
            })
        except (FileExistsError, FileNotFoundError, TypeError, ValueError) as error:
            direct_reports.append({
                "input": str(path),
                "source_schema_version": schema_version,
                "status": "blocked",
                "task": task,
                "reason": str(error) or type(error).__name__,
            })

    return {
        "schema_version": "omniflow.function-store-root-migration.v1",
        "input_root": str(source_root),
        "output_root": str(Path(output_root).expanduser().resolve()),
        "dry_run": dry_run,
        "catalogs": reports,
        "direct": direct_reports,
        "counts": _migration_counts([*reports, *direct_reports]),
    }


def _catalog_path(catalog_path: Path, item: dict[str, Any], field: str) -> Path:
    value = str(item.get(field) or "").strip()
    if not value:
        raise ValueError(f"catalog_{field}_required")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else catalog_path.parent / path).resolve()


def _task_name_for_path(path: Path) -> str:
    parts = path.resolve().parts
    for marker in ("androidworld", "bmoca"):
        if marker in parts and parts.index(marker) + 1 < len(parts):
            return parts[parts.index(marker) + 1]
    return path.parent.name or "unknown_task"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _migration_counts(reports: Iterable[dict[str, Any]]) -> dict[str, int]:
    converted = blocked = stores = 0
    for report in reports:
        if report.get("status") == "converted":
            converted += 1
        if report.get("status") == "blocked":
            blocked += 1
        nested = report.get("tasks")
        if isinstance(nested, list):
            for item in nested:
                if item.get("status") == "converted":
                    converted += 1
                if item.get("status") == "blocked":
                    blocked += 1
                item_stores = item.get("stores")
                if isinstance(item_stores, list):
                    stores += len(item_stores)
        stores_value = report.get("stores")
        if isinstance(stores_value, list):
            stores += len(stores_value)
    return {"converted": converted, "blocked": blocked, "stores": stores}


def _path_tokens(value: str) -> list[str | int]:
    return [name if name else int(index) for name, index in _PATH_TOKEN.findall(value)]


def _read_path(root: Any, tokens: list[str | int]) -> Any:
    value = root
    for token in tokens:
        value = value[token]
    return value


def _write_path(root: Any, tokens: list[str | int], value: Any) -> None:
    if not tokens:
        raise ValueError("function_migration_empty_path")
    target = root
    for token in tokens[:-1]:
        target = target[token]
    target[tokens[-1]] = value


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"function_migration_json_object_required:{path}")
    return value


def _refuse_existing(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"function_migration_output_exists:{path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Old Store, bundle, or catalog JSON.")
    input_group.add_argument("--input-root", help="Root containing historical Function JSON.")
    parser.add_argument("--output", required=True, help="New JSON path or migration output root.")
    parser.add_argument("--source-run-log", help="Required for omniflow.function-bundle.v2.")
    parser.add_argument("--transfer-states", help="Optional old transfer_states.json for a Store.")
    parser.add_argument("--environment", default="androidworld")
    parser.add_argument("--device", default="source5554")
    parser.add_argument("--method", default="function_authoring")
    parser.add_argument("--attempt-id")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report without writing output.")
    parser.add_argument("--report", help="Write the JSON migration report to this path.")
    parser.add_argument("--force", action="store_true", help="Allow replacing migration outputs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.input_root:
        report = scan_function_json_root(
            args.input_root,
            args.output,
            environment=args.environment,
            device=args.device,
            method=args.method,
            force=args.force,
            dry_run=args.dry_run,
        )
    else:
        input_path = Path(args.input).expanduser().resolve()
        payload = _read_object(input_path)
        if payload.get("schema_version") == LEGACY_CATALOG_VERSION:
            report = migrate_function_catalog(
                input_path,
                args.output,
                environment=args.environment,
                device=args.device,
                method=args.method,
                attempt_id=args.attempt_id,
                force=args.force,
                dry_run=args.dry_run,
            )
        else:
            report = migrate_function_store(
                input_path,
                args.output,
                source_run_log=args.source_run_log,
                transfer_states=args.transfer_states,
                force=args.force,
                dry_run=args.dry_run,
            )
    if args.report:
        report_path = Path(args.report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
