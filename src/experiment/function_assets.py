"""Convert legacy authored Function bundles into current frozen assets."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from omniflow import compile_runlog_to_store
from omniflow.functions.store import FunctionStore
from omniflow.transfer.runtime import (
    audit_transfer_action_sources,
    load_transfer_state_catalog,
    transfer_state_coverage,
)
from src.integrations.runlog import import_run_log_evidence

CATALOG_SCHEMA = "omniflow.function-asset-catalog.v1"
_LEGACY_BUNDLE_NAME = "codex_function_bundle.json"
_TASK_DIRECTORY = re.compile(r"^(?:[0-9]+_)?(?P<task>[A-Za-z0-9][A-Za-z0-9._-]*)$")
_SOURCE_PATH = re.compile(r"^\$\.arguments(?P<tail>(?:\.[A-Za-z_][A-Za-z0-9_]*|\[\d+\])+)$")
_TARGET_PATH = re.compile(
    r"^\$\.actions\[(?P<index>\d+)]\.arguments"
    r"(?P<tail>(?:\.[A-Za-z_][A-Za-z0-9_]*|\[\d+\])+)$"
)
_PATH_TOKEN = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)]")
_IGNORED_LEGACY_ARGS = {
    "app_name",
    "clear_text",
    "post_action_wait_s",
    "post_wait_s",
    "wait_after_s",
}
_SOURCE_CENTER_TOLERANCE = 5.0


def convert_function_assets(
    *,
    legacy_roots: Iterable[str | Path],
    source_asset_index: str | Path,
    output_root: str | Path,
    task_names: Iterable[str] | None = None,
    exclude_task_names: Iterable[str] = (),
) -> dict[str, Any]:
    """Deduplicate authored bundles by task and convert available source assets.

    Legacy bundles and frozen source evidence are read-only. The output root is
    a new immutable version directory containing one task directory per unique
    authored bundle and one runtime Store per task with complete current source
    evidence.
    """

    roots = tuple(
        sorted({Path(value).expanduser().resolve() for value in legacy_roots})
    )
    if not roots:
        raise ValueError("legacy_function_roots_required")
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"legacy_function_root_missing:{root}")

    destination = Path(output_root).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"immutable_function_asset_root_exists:{destination}")

    source_index_path = Path(source_asset_index).expanduser().resolve()
    source_index = _read_object(source_index_path)
    raw_source_rows = source_index.get("assets", source_index)
    if not isinstance(raw_source_rows, dict):
        raise ValueError("source_asset_index_tasks_required")

    bundles = _deduplicated_bundles(roots)
    if task_names is not None:
        requested = tuple(str(value).strip() for value in task_names)
        if not requested or any(not value for value in requested):
            raise ValueError("function_asset_task_names_required")
        if len(set(requested)) != len(requested):
            raise ValueError("function_asset_task_names_duplicate")
        missing = sorted(set(requested) - set(bundles))
        if missing:
            raise ValueError(
                f"legacy_function_task_missing:{','.join(missing)}"
            )
        bundles = {
            task_name: bundles[task_name]
            for task_name in sorted(requested)
        }
    excluded = {
        str(value).strip()
        for value in exclude_task_names
        if str(value).strip()
    }
    excluded_present = sorted(set(bundles) & excluded)
    bundles = {
        task_name: bundle
        for task_name, bundle in bundles.items()
        if task_name not in excluded
    }
    task_reports: dict[str, dict[str, Any]] = {}
    store_index: dict[str, dict[str, Any]] = {}
    destination.mkdir(parents=True, exist_ok=True)

    for task_name, legacy in bundles.items():
        task_root = destination / "tasks" / task_name
        task_root.mkdir(parents=True, exist_ok=False)
        source_row = raw_source_rows.get(task_name)
        report: dict[str, Any] = {
            "task": task_name,
            "legacy_bundle_sha256": legacy["sha256"],
            "legacy_bundle_path": legacy["canonical_path"],
            "legacy_bundle_sources": legacy["sources"],
            "deduplicated_source_count": len(legacy["sources"]),
            "target_inputs_read": False,
            "target_observations_read": False,
        }
        if not isinstance(source_row, dict):
            report["status"] = "catalogued_source_evidence_missing"
        else:
            source_path = _source_run_log_path(
                source_row,
                source_index_path=source_index_path,
                task_name=task_name,
            )
            converted = _convert_runlog_to_function_asset(
                task_name=task_name,
                legacy_bundle=legacy["payload"],
                legacy_bundle_sha256=legacy["sha256"],
                source_run_log=source_path,
                source_index_path=source_index_path,
                output_root=task_root / "function_store",
            )
            report.update(converted)
            report["status"] = "converted"
            store_index[task_name] = {
                "store_path": converted["store_path"],
                "store_sha256": converted["store_sha256"],
                "transfer_states_path": converted["transfer_states_path"],
                "transfer_states_sha256": converted["transfer_states_sha256"],
                "provenance_path": converted["provenance_path"],
                "provenance_sha256": converted["provenance_sha256"],
            }
        _write_json(task_root / "task_manifest.json", report)
        task_reports[task_name] = report

    report = {
        "schema_version": CATALOG_SCHEMA,
        "legacy_roots": [str(root) for root in roots],
        "source_asset_index": str(source_index_path),
        "source_asset_index_sha256": _sha256(source_index_path),
        "task_count": len(task_reports),
        "excluded_existing_task_count": len(excluded_present),
        "excluded_existing_tasks": excluded_present,
        "converted_task_count": len(store_index),
        "catalogued_task_count": len(task_reports) - len(store_index),
        "target_inputs_read": False,
        "target_observations_read": False,
        "tasks": task_reports,
    }
    _write_json(destination / "store_index.json", store_index)
    _write_json(destination / "catalog.json", report)
    return report


def _deduplicated_bundles(
    roots: tuple[Path, ...],
) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, list[tuple[Path, dict[str, Any]]]]] = {}
    for root in roots:
        for path in sorted(root.rglob(_LEGACY_BUNDLE_NAME)):
            match = _TASK_DIRECTORY.fullmatch(path.parent.name)
            if match is None:
                raise ValueError(f"legacy_function_task_directory_invalid:{path}")
            task_name = match.group("task")
            payload = _read_object(path)
            _validate_legacy_bundle(payload, path)
            digest = _json_sha256(payload)
            candidates.setdefault(task_name, {}).setdefault(digest, []).append(
                (path.resolve(), payload)
            )

    result: dict[str, dict[str, Any]] = {}
    for task_name, by_hash in sorted(candidates.items()):
        if len(by_hash) != 1:
            raise ValueError(
                "legacy_function_bundle_conflict:"
                f"{task_name}:{','.join(sorted(by_hash))}"
            )
        digest, values = next(iter(by_hash.items()))
        sources = sorted(str(path) for path, _payload in values)
        result[task_name] = {
            "sha256": digest,
            "canonical_path": sources[0],
            "sources": sources,
            "payload": values[0][1],
        }
    if not result:
        raise ValueError("legacy_function_bundles_missing")
    return result


def _validate_legacy_bundle(payload: dict[str, Any], path: Path) -> None:
    if payload.get("schema_version") != "omniflow.function-bundle.v2":
        raise ValueError(f"legacy_function_bundle_version_invalid:{path}")
    if payload.get("source_success") is not True:
        raise ValueError(f"legacy_function_bundle_source_not_successful:{path}")
    if not str(payload.get("source_run_id") or "").strip():
        raise ValueError(f"legacy_function_bundle_source_run_id_required:{path}")
    if not isinstance(payload.get("source_arguments"), dict):
        raise ValueError(f"legacy_function_bundle_arguments_invalid:{path}")
    functions = payload.get("functions")
    if not isinstance(functions, list) or not functions:
        raise ValueError(f"legacy_function_bundle_functions_required:{path}")


def _convert_runlog_to_function_asset(
    *,
    task_name: str,
    legacy_bundle: dict[str, Any],
    source_run_log: str | Path,
    output_root: str | Path,
    legacy_bundle_sha256: str | None = None,
    source_index_path: str | Path | None = None,
) -> dict[str, Any]:
    """Adapt one historical source RunLog to the shared Function compiler."""
    source_path = Path(source_run_log).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(
            f"function_asset_source_run_log_missing:{task_name}:{source_path}"
        )
    output_path = Path(output_root).expanduser().resolve()
    bundle_sha256 = legacy_bundle_sha256 or _json_sha256(legacy_bundle)
    index_path = (
        Path(source_index_path).expanduser().resolve()
        if source_index_path is not None
        else None
    )
    source_run_log, source_states = import_run_log_evidence(
        _read_object(source_path),
        evidence_root=source_path.parent,
    )
    legacy_source_run_id = str(legacy_bundle.get("source_run_id") or "").strip()
    if legacy_source_run_id != str(source_run_log["run_id"]):
        raise ValueError(
            "legacy_function_bundle_source_run_id_mismatch:"
            f"{task_name}:legacy={legacy_source_run_id}:"
            f"source={source_run_log['run_id']}"
        )
    source_steps = [
        step
        for step in source_run_log["steps"]
        if isinstance(step, dict)
        and isinstance(step.get("result"), dict)
        and step["result"].get("success") is True
        and isinstance(step.get("action"), dict)
    ]
    if not source_steps:
        raise ValueError(f"function_asset_source_actions_required:{task_name}")

    source_arguments = legacy_bundle["source_arguments"]
    functions: list[dict[str, Any]] = []
    current_arguments: dict[str, dict[str, Any]] = {}
    for legacy_function in legacy_bundle["functions"]:
        function, arguments = _convert_function(
            task_name=task_name,
            legacy_function=legacy_function,
            source_arguments=source_arguments,
            source_steps=source_steps,
        )
        functions.append(function)
        current_arguments[function["function_id"]] = arguments

    current_bundle = {
        "schema_version": "omniflow.function-bundle.v2",
        "run_id": source_run_log["run_id"],
        "arguments": current_arguments,
        "functions": functions,
    }
    compile_result = compile_runlog_to_store(
        source_run_log,
        output_path,
        function_bundle=current_bundle,
        source_states=source_states,
    )
    store_path = Path(compile_result["store_path"]).resolve()
    transfer_path = Path(compile_result["transfer_state_catalog"]).resolve()
    store = FunctionStore(store_path)
    if store.load_errors:
        raise ValueError(
            "converted_function_store_invalid:"
            f"{task_name}:{','.join(sorted(store.load_errors))}"
        )
    compiled_states = load_transfer_state_catalog(transfer_path)
    coverage = transfer_state_coverage(store.functions, compiled_states)
    if not coverage["complete"]:
        raise ValueError(
            "converted_function_transfer_states_incomplete:"
            f"{task_name}:{','.join(coverage['missing_state_ids'])}"
        )
    try:
        source_target_audit = audit_transfer_action_sources(
            store.functions,
            compiled_states,
        )
    except ValueError as error:
        reason = str(error)
        if not reason.startswith(
            (
                "transfer_action_source_target_unresolved:",
                "transfer_action_source_state_not_raw:",
            )
        ):
            raise
        source_target_audit = {
            "source_target_audit_complete": False,
            "source_target_count": 0,
            "source_targets": [],
            "fallback_required": True,
            "failure": reason,
        }
    provenance_path = output_path.parent / "provenance_manifest.json"
    provenance = {
        "schema_version": "omniflow.function-asset-conversion-provenance.v1",
        "task": task_name,
        "source_run_log": str(source_path),
        "source_run_log_sha256": _sha256(source_path),
        "source_run_id": source_run_log["run_id"],
        "source_asset_index": str(index_path) if index_path is not None else "",
        "source_asset_index_sha256": (
            _sha256(index_path) if index_path is not None else ""
        ),
        "legacy_bundle_sha256": bundle_sha256,
        "legacy_source_run_id": legacy_source_run_id,
        "legacy_source_run_id_match": (
            legacy_source_run_id == str(source_run_log["run_id"])
        ),
        "semantic_enhancement": {
            "mode": "frozen_authored_function_bundle",
            "model_calls": 0,
            "function_ids": list(compile_result["function_ids"]),
        },
        "store_path": str(store_path),
        "store_sha256": _sha256(store_path),
        "transfer_states_path": str(transfer_path),
        "transfer_states_sha256": _sha256(transfer_path),
        "source_state_count": len(compiled_states),
        "transfer_state_coverage": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in coverage.items()
        },
        "source_target_audit": source_target_audit,
        "target_inputs_read": False,
        "target_observations_read": False,
        "validator_state_read": False,
    }
    _write_json(provenance_path, provenance)
    return {
        "function_ids": list(compile_result["function_ids"]),
        "function_count": int(compile_result["function_count"]),
        "legacy_bundle_sha256": bundle_sha256,
        "legacy_source_run_id": legacy_source_run_id,
        "legacy_source_run_id_match": provenance["legacy_source_run_id_match"],
        "indexed_source_run_log": str(source_path),
        "indexed_source_run_log_sha256": _sha256(source_path),
        "source_run_log": str(source_path),
        "source_run_log_sha256": _sha256(source_path),
        "compiled_source_run_id": source_run_log["run_id"],
        "store_path": str(store_path),
        "store_sha256": _sha256(store_path),
        "transfer_states_path": str(transfer_path),
        "transfer_states_sha256": _sha256(transfer_path),
        "provenance_path": str(provenance_path),
        "provenance_sha256": _sha256(provenance_path),
        "source_target_audit": source_target_audit,
    }


def _source_run_log_path(
    row: dict[str, Any],
    *,
    source_index_path: Path,
    task_name: str,
) -> Path:
    path_value = next(
        (
            str(row.get(field) or "").strip()
            for field in (
                "retained_source_run_log",
                "source_run_log",
                "object_path",
            )
            if row.get(field)
        ),
        "",
    )
    if not path_value:
        raise ValueError(f"source_index_run_log_required:{task_name}")
    raw_path = Path(path_value).expanduser()
    candidates = (
        (raw_path,)
        if raw_path.is_absolute()
        else tuple(
            parent / raw_path
            for parent in (source_index_path.parent, *source_index_path.parents)
        )
    )
    source_path = next(
        (candidate.resolve() for candidate in candidates if candidate.is_file()),
        candidates[0].resolve(),
    )
    if not source_path.is_file():
        raise FileNotFoundError(
            f"function_asset_source_run_log_missing:{task_name}:{source_path}"
        )
    expected_hash = next(
        (
            str(row.get(field) or "").strip()
            for field in (
                "retained_source_run_log_sha256",
                "source_run_log_sha256",
                "sha256",
            )
            if row.get(field)
        ),
        "",
    )
    actual_hash = _sha256(source_path)
    if expected_hash and expected_hash != actual_hash:
        raise ValueError(
            "function_asset_source_run_log_hash_mismatch:"
            f"{task_name}:expected={expected_hash}:actual={actual_hash}"
        )
    return source_path


def _convert_function(
    *,
    task_name: str,
    legacy_function: Any,
    source_arguments: dict[str, Any],
    source_steps: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(legacy_function, dict):
        raise ValueError(f"legacy_function_invalid:{task_name}")
    function_id = str(
        legacy_function.get("function_id") or legacy_function.get("id") or ""
    ).strip()
    if not function_id:
        raise ValueError(f"legacy_function_id_required:{task_name}")
    raw_arguments = source_arguments.get(function_id, {})
    if not isinstance(raw_arguments, dict):
        raise ValueError(f"legacy_function_source_arguments_invalid:{function_id}")
    arguments = deepcopy(raw_arguments)
    raw_actions = legacy_function.get("actions")
    if not isinstance(raw_actions, list) or not raw_actions:
        raise ValueError(f"legacy_function_actions_required:{function_id}")
    actions = deepcopy(raw_actions)
    raw_bindings = legacy_function.get("bindings")
    if not isinstance(raw_bindings, list):
        raise ValueError(f"legacy_function_bindings_invalid:{function_id}")
    _materialize_legacy_actions(
        actions=actions,
        arguments=arguments,
        bindings=raw_bindings,
        function_id=function_id,
    )

    expected, legacy_to_expected = _expanded_actions(actions, function_id)
    start = _source_alignment(
        expected,
        source_steps=source_steps,
        function_id=function_id,
    )
    matched_source_steps = source_steps[start : start + len(expected)]
    current_steps = [
        {
            "step_index": index,
            "source_state_id": str(step.get("before_state_id") or ""),
            "action": deepcopy(step["action"]),
        }
        for index, step in enumerate(matched_source_steps)
    ]
    current_bindings: list[dict[str, str]] = []
    for binding in raw_bindings:
        source = str(binding.get("source") or "")
        target = str(binding.get("target") or "")
        source_match = _SOURCE_PATH.fullmatch(source)
        target_match = _TARGET_PATH.fullmatch(target)
        if source_match is None or target_match is None:
            raise ValueError(
                f"legacy_function_binding_path_invalid:{function_id}:{source}->{target}"
            )
        legacy_index = int(target_match.group("index"))
        if legacy_index not in legacy_to_expected:
            raise ValueError(
                f"legacy_function_binding_action_invalid:{function_id}:{legacy_index}"
            )
        target_tokens = _path_tokens(target_match.group("tail"))
        relative_candidates = legacy_to_expected[legacy_index]
        matching_targets: list[int] = []
        for relative_index in relative_candidates:
            try:
                _read_path(
                    current_steps[relative_index]["action"]["args"],
                    target_tokens,
                )
            except (IndexError, KeyError, TypeError):
                continue
            matching_targets.append(relative_index)
        if len(matching_targets) != 1:
            raise ValueError(
                "legacy_function_binding_target_ambiguous:"
                f"{function_id}:{target}:{matching_targets}"
            )
        relative_index = matching_targets[0]
        legacy_placeholder = _read_path(
            raw_actions[legacy_index]["arguments"],
            target_tokens,
        )
        _write_path(
            current_steps[relative_index]["action"]["args"],
            target_tokens,
            deepcopy(legacy_placeholder),
        )
        current_bindings.append(
            {
                "source": source,
                "target": (
                    f"$.steps[{relative_index}].action.args"
                    f"{target_match.group('tail')}"
                ),
            }
        )

    description = str(legacy_function.get("description") or "").strip()
    name = str(legacy_function.get("name") or description).strip()
    if not name or not description:
        raise ValueError(f"legacy_function_semantics_required:{function_id}")
    input_schema = legacy_function.get("input_schema")
    if input_schema is None:
        input_schema = legacy_function.get("parameters")
    if not isinstance(input_schema, dict):
        raise ValueError(f"legacy_function_parameters_invalid:{function_id}")
    checker_rules = legacy_function.get("checker_rules")
    if not isinstance(checker_rules, list):
        raise ValueError(f"legacy_function_checker_rules_invalid:{function_id}")
    return (
        {
            "schema_version": "omniflow.function.v2",
            "function_id": function_id,
            "name": name,
            "description": description,
            "input_schema": deepcopy(input_schema),
            "bindings": current_bindings,
            "steps": current_steps,
            "checker_rules": deepcopy(checker_rules),
            "agent_visible": legacy_function.get("agent_visible") is not False,
        },
        arguments,
    )


def _materialize_legacy_actions(
    *,
    actions: list[Any],
    arguments: dict[str, Any],
    bindings: list[Any],
    function_id: str,
) -> None:
    for binding in bindings:
        if not isinstance(binding, dict):
            raise ValueError(f"legacy_function_binding_invalid:{function_id}")
        source = str(binding.get("source") or "")
        target = str(binding.get("target") or "")
        source_match = _SOURCE_PATH.fullmatch(source)
        target_match = _TARGET_PATH.fullmatch(target)
        if source_match is None or target_match is None:
            raise ValueError(
                f"legacy_function_binding_path_invalid:{function_id}:{source}->{target}"
            )
        action_index = int(target_match.group("index"))
        try:
            action_args = actions[action_index]["arguments"]
            value = _read_path(
                {"arguments": arguments},
                _path_tokens(".arguments" + source_match.group("tail")),
            )
            _write_path(
                action_args,
                _path_tokens(target_match.group("tail")),
                deepcopy(value),
            )
        except (IndexError, KeyError, TypeError) as error:
            raise ValueError(
                f"legacy_function_binding_evidence_missing:{function_id}:{target}"
            ) from error


def _expanded_actions(
    actions: list[Any],
    function_id: str,
) -> tuple[list[dict[str, Any]], dict[int, list[int]]]:
    expanded: list[dict[str, Any]] = []
    mapping: dict[int, list[int]] = {}
    for legacy_index, raw_action in enumerate(actions):
        if not isinstance(raw_action, dict):
            raise ValueError(f"legacy_function_action_invalid:{function_id}")
        tool = str(raw_action.get("tool") or "").strip()
        args = raw_action.get("arguments")
        if not tool or not isinstance(args, dict):
            raise ValueError(f"legacy_function_action_invalid:{function_id}")
        converted: list[dict[str, Any]]
        if tool == "input_text" and "x" in args and "y" in args:
            converted = [
                {"tool": "click", "args": {}},
                {
                    "tool": "input_text",
                    "args": {
                        key: value
                        for key, value in args.items()
                        if key not in _IGNORED_LEGACY_ARGS | {"x", "y"}
                    },
                },
            ]
        elif tool == "press_back":
            converted = [{"tool": "press_key", "args": {"key": "back"}}]
        elif tool == "press_key":
            raw_key = str(args.get("key") or args.get("keycode") or "").strip()
            key = raw_key.lower().removeprefix("keycode_")
            if key == "del":
                key = "delete"
            converted = [{"tool": "press_key", "args": {"key": key}}]
        elif tool == "start_activity":
            converted = [{"tool": "open_app", "args": {}}]
        elif tool in {"answer", "finished"}:
            converted = []
        elif tool == "wait" and "time_s" in args:
            converted = [
                {
                    "tool": "wait",
                    "args": {"duration_ms": round(float(args["time_s"]) * 1000)},
                }
            ]
        else:
            normalized_args = {
                key: value
                for key, value in args.items()
                if key not in _IGNORED_LEGACY_ARGS
            }
            if tool == "swipe":
                aliases = {
                    "start_x": "x1",
                    "start_y": "y1",
                    "end_x": "x2",
                    "end_y": "y2",
                }
                for old, new in aliases.items():
                    if new not in normalized_args and old in normalized_args:
                        normalized_args[new] = normalized_args.pop(old)
                if "direction" not in normalized_args and all(
                    key in normalized_args
                    for key in ("x1", "y1", "x2", "y2")
                ):
                    dx = float(normalized_args["x2"]) - float(
                        normalized_args["x1"]
                    )
                    dy = float(normalized_args["y2"]) - float(
                        normalized_args["y1"]
                    )
                    normalized_args["direction"] = (
                        ("right" if dx > 0 else "left")
                        if abs(dx) >= abs(dy)
                        else ("down" if dy > 0 else "up")
                    )
            converted = [
                {
                    "tool": tool,
                    "args": normalized_args,
                }
            ]
        mapping[legacy_index] = list(
            range(len(expanded), len(expanded) + len(converted))
        )
        expanded.extend(converted)
    return expanded, mapping


def _source_alignment(
    expected: list[dict[str, Any]],
    *,
    source_steps: list[dict[str, Any]],
    function_id: str,
) -> int:
    matches = [
        start
        for start in range(len(source_steps) - len(expected) + 1)
        if all(
            _actions_match(item, source_steps[start + offset]["action"])
            for offset, item in enumerate(expected)
        )
    ]
    if not matches:
        raise ValueError(
            f"legacy_function_source_alignment_invalid:{function_id}:{matches}"
        )
    return matches[0]


def _actions_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    if str(expected.get("tool") or "") != str(actual.get("tool") or ""):
        return False
    expected_args = expected.get("args")
    actual_args = actual.get("args")
    if not isinstance(expected_args, dict) or not isinstance(actual_args, dict):
        return False
    for key, value in expected_args.items():
        if expected["tool"] == "swipe" and key == "direction":
            continue
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


def _path_tokens(value: str) -> list[str | int]:
    return [
        name if name else int(index)
        for name, index in _PATH_TOKEN.findall(value)
    ]


def _read_path(root: Any, tokens: list[str | int]) -> Any:
    value = root
    for token in tokens:
        value = value[token]
    return value


def _write_path(root: Any, tokens: list[str | int], value: Any) -> None:
    if not tokens:
        raise ValueError("function_asset_binding_path_empty")
    target = root
    for token in tokens[:-1]:
        target = target[token]
    target[tokens[-1]] = value


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _freeze_tree(root: Path) -> None:
    paths = [root, *root.rglob("*")]
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"function_asset_symlink_forbidden:{path}")
        path.chmod(0o555 if path.is_dir() else 0o444)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Deduplicate legacy authored Function bundles by task and convert "
            "those with current frozen source evidence."
        )
    )
    parser.add_argument(
        "--legacy-root",
        action="append",
        required=True,
        help="Read-only root containing task/codex_function_bundle.json assets.",
    )
    parser.add_argument("--source-asset-index", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--task",
        action="append",
        help="Convert only this exact task; repeat for multiple tasks.",
    )
    parser.add_argument(
        "--memory-index",
        required=True,
        help=(
            "Existing AndroidWorld artifact-memory current.json. The frozen "
            "conversion catalog is registered here before success is reported."
        ),
    )
    args = parser.parse_args(argv)
    from src.experiment.artifact_memory import (
        load_artifact_memory,
        refresh_artifact_memory_from_pointer,
    )

    existing_memory = load_artifact_memory(args.memory_index)
    existing_function_stores = set(
        existing_memory["canonical"]["function_stores"]
    )
    report = convert_function_assets(
        legacy_roots=args.legacy_root,
        source_asset_index=args.source_asset_index,
        output_root=args.output_root,
        task_names=args.task,
        exclude_task_names=existing_function_stores,
    )
    output_root = Path(args.output_root).expanduser().resolve()
    _freeze_tree(output_root)
    memory = refresh_artifact_memory_from_pointer(
        memory_index=args.memory_index,
        additional_function_catalogs=(output_root / "catalog.json",),
    )
    print(
        json.dumps(
            {
                "catalog": str(output_root / "catalog.json"),
                "store_index": str(output_root / "store_index.json"),
                "memory_index": str(
                    Path(args.memory_index).expanduser().resolve()
                ),
                "memory_function_store_tasks": memory["counts"][
                    "function_store_tasks"
                ],
                "tasks": report["task_count"],
                "reused": report["excluded_existing_task_count"],
                "converted": report["converted_task_count"],
                "catalogued": report["catalogued_task_count"],
                "frozen": True,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


__all__ = ["CATALOG_SCHEMA", "convert_function_assets"]


if __name__ == "__main__":
    raise SystemExit(main())
