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
from omniflow.core.trajectory import canonicalize_run_log

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
            converted = _convert_task(
                task_name=task_name,
                legacy_bundle=legacy["payload"],
                legacy_bundle_sha256=legacy["sha256"],
                source_row=source_row,
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


def _convert_task(
    *,
    task_name: str,
    legacy_bundle: dict[str, Any],
    legacy_bundle_sha256: str,
    source_row: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    indexed_source_path = _frozen_path(
        source_row,
        path_fields=("source_run_log",),
        hash_fields=("source_run_log_sha256",),
        label=f"{task_name}:source_run_log",
    )
    states_path = _frozen_path(
        source_row,
        path_fields=("transfer_state_catalog", "transfer_states_path"),
        hash_fields=(
            "transfer_state_catalog_sha256",
            "transfer_states_sha256",
        ),
        label=f"{task_name}:transfer_states",
    )
    provenance_path = _frozen_path(
        source_row,
        path_fields=("provenance_manifest", "provenance_path"),
        hash_fields=("provenance_manifest_sha256", "provenance_sha256"),
        label=f"{task_name}:provenance",
    )
    provenance = _read_object(provenance_path)
    recorded_indexed_hash = str(
        provenance.get("source_run_log_sha256") or ""
    ).strip()
    if recorded_indexed_hash and recorded_indexed_hash != _sha256(
        indexed_source_path
    ):
        raise ValueError(
            f"function_asset_provenance_source_mismatch:{task_name}"
        )
    raw_source_states = _read_object(states_path)
    source_path = _paired_source_run_log(
        task_name=task_name,
        indexed_source_path=indexed_source_path,
        provenance=provenance,
        provenance_path=provenance_path,
        source_states=raw_source_states,
    )
    source_run_log = canonicalize_run_log(_read_object(source_path))
    legacy_source_run_id = str(legacy_bundle.get("source_run_id") or "").strip()
    if legacy_source_run_id != str(source_run_log["run_id"]):
        raise ValueError(
            "legacy_function_bundle_source_run_id_mismatch:"
            f"{task_name}:legacy={legacy_source_run_id}:"
            f"current={source_run_log['run_id']}"
        )
    source_states = deepcopy(raw_source_states)
    original_state_catalog_run_id = str(source_states.get("run_id") or "")
    source_states["run_id"] = source_run_log["run_id"]
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
        output_root,
        function_bundle=current_bundle,
        source_states=source_states,
    )
    store_path = Path(compile_result["store_path"]).resolve()
    transfer_path = Path(compile_result["transfer_state_catalog"]).resolve()
    return {
        "function_ids": list(compile_result["function_ids"]),
        "function_count": int(compile_result["function_count"]),
        "legacy_bundle_sha256": legacy_bundle_sha256,
        "indexed_source_run_log": str(indexed_source_path),
        "indexed_source_run_log_sha256": _sha256(indexed_source_path),
        "source_run_log": str(source_path),
        "source_run_log_sha256": _sha256(source_path),
        "source_transfer_states": str(states_path),
        "source_transfer_states_sha256": _sha256(states_path),
        "source_transfer_states_run_id": original_state_catalog_run_id,
        "compiled_source_run_id": source_run_log["run_id"],
        "store_path": str(store_path),
        "store_sha256": _sha256(store_path),
        "transfer_states_path": str(transfer_path),
        "transfer_states_sha256": _sha256(transfer_path),
        "provenance_path": str(provenance_path),
        "provenance_sha256": _sha256(provenance_path),
    }


def _paired_source_run_log(
    *,
    task_name: str,
    indexed_source_path: Path,
    provenance: dict[str, Any],
    provenance_path: Path,
    source_states: dict[str, Any],
) -> Path:
    raw_states = source_states.get("states")
    if not isinstance(raw_states, dict):
        raise ValueError(f"function_asset_source_states_invalid:{task_name}")
    available = set(raw_states)
    indexed = canonicalize_run_log(_read_object(indexed_source_path))
    if _required_state_ids(indexed).issubset(available):
        return indexed_source_path

    paired_value = str(provenance.get("output_source_run_log") or "").strip()
    paired_hash = str(
        provenance.get("output_source_run_log_sha256") or ""
    ).strip()
    if not paired_value or not paired_hash:
        raise ValueError(
            f"function_asset_paired_source_run_log_required:{task_name}"
        )
    paired_path = Path(paired_value).expanduser()
    if not paired_path.is_absolute():
        paired_path = provenance_path.parent / paired_path
    paired_path = paired_path.resolve()
    if not paired_path.is_file():
        raise FileNotFoundError(
            f"function_asset_paired_source_run_log_missing:{task_name}:{paired_path}"
        )
    actual_hash = _sha256(paired_path)
    if actual_hash != paired_hash:
        raise ValueError(
            "function_asset_paired_source_run_log_hash_mismatch:"
            f"{task_name}:expected={paired_hash}:actual={actual_hash}"
        )
    paired = canonicalize_run_log(_read_object(paired_path))
    missing = sorted(_required_state_ids(paired) - available)
    if missing:
        raise ValueError(
            f"function_asset_source_states_missing:{task_name}:{','.join(missing)}"
        )
    return paired_path


def _required_state_ids(run_log: dict[str, Any]) -> set[str]:
    return {
        str(step.get("before_state_id") or "").strip()
        for step in run_log.get("steps") or ()
        if isinstance(step, dict)
        and isinstance(step.get("result"), dict)
        and step["result"].get("success") is True
        and isinstance(step.get("action"), dict)
        and str(step.get("before_state_id") or "").strip()
    }


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
    start = _unique_alignment(
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
        elif tool == "wait" and "time_s" in args:
            converted = [
                {
                    "tool": "wait",
                    "args": {"duration_ms": round(float(args["time_s"]) * 1000)},
                }
            ]
        else:
            converted = [
                {
                    "tool": tool,
                    "args": {
                        key: value
                        for key, value in args.items()
                        if key not in _IGNORED_LEGACY_ARGS
                    },
                }
            ]
        mapping[legacy_index] = list(
            range(len(expanded), len(expanded) + len(converted))
        )
        expanded.extend(converted)
    return expanded, mapping


def _unique_alignment(
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
    if len(matches) != 1:
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


def _frozen_path(
    row: dict[str, Any],
    *,
    path_fields: tuple[str, ...],
    hash_fields: tuple[str, ...],
    label: str,
) -> Path:
    path_value = next(
        (str(row.get(field) or "").strip() for field in path_fields if row.get(field)),
        "",
    )
    expected = next(
        (str(row.get(field) or "").strip() for field in hash_fields if row.get(field)),
        "",
    )
    path = Path(path_value).expanduser()
    if not path.is_absolute() or not path.is_file():
        raise FileNotFoundError(f"function_asset_evidence_missing:{label}:{path}")
    actual = _sha256(path)
    if not expected or actual != expected:
        raise ValueError(
            f"function_asset_evidence_hash_mismatch:{label}:"
            f"expected={expected or 'missing'}:actual={actual}"
        )
    return path.resolve()


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
        "--memory-index",
        required=True,
        help=(
            "Existing AndroidWorld artifact-memory current.json. The frozen "
            "conversion catalog is registered here before success is reported."
        ),
    )
    args = parser.parse_args(argv)
    report = convert_function_assets(
        legacy_roots=args.legacy_root,
        source_asset_index=args.source_asset_index,
        output_root=args.output_root,
    )
    output_root = Path(args.output_root).expanduser().resolve()
    _freeze_tree(output_root)
    from src.experiment.artifact_memory import (
        refresh_artifact_memory_from_pointer,
    )

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
