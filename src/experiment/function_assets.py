"""Compile human-recorded source RunLogs into frozen Function assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from omniflow import compile_runlog_to_store
from omniflow.functions.assets import FunctionStore
from omniflow.transfer.runtime import (
    audit_transfer_action_sources,
    load_transfer_state_catalog,
    transfer_state_coverage,
)
from src.integrations.android_world.apps import resolve_androidworld_package
from src.integrations.runlog import import_run_log_evidence

CATALOG_SCHEMA = "omniflow.function-asset-catalog.v1"
AUTHORING_SCHEMA = "omniflow.function-agent-skill-manifest.v1"


def convert_function_assets(
    *,
    source_asset_index: str | Path,
    authoring_manifest: str | Path,
    output_root: str | Path,
    task_names: Iterable[str] | None = None,
    exclude_task_names: Iterable[str] = (),
) -> dict[str, Any]:
    """Compile selected source RunLogs from an immutable offline manifest."""

    destination = Path(output_root).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"immutable_function_asset_root_exists:{destination}")

    source_index_path = Path(source_asset_index).expanduser().resolve()
    source_index = _read_object(source_index_path)
    authoring_path = Path(authoring_manifest).expanduser().resolve()
    authoring = _read_object(authoring_path)
    _validate_authoring_manifest(
        authoring,
        source_index_path=source_index_path,
    )
    raw_authoring_tasks = authoring["tasks"]
    producer = authoring["producer"]
    raw_source_rows = source_index.get("assets", source_index)
    if not isinstance(raw_source_rows, dict):
        raise ValueError("source_asset_index_tasks_required")

    available: dict[str, dict[str, Any]] = {}
    for task_name, row in raw_source_rows.items():
        if not isinstance(row, dict):
            raise ValueError(f"source_asset_index_task_invalid:{task_name}")
        available[str(task_name)] = row
    if task_names is None:
        requested = tuple(sorted(available))
    else:
        requested = tuple(str(value).strip() for value in task_names)
        if not requested or any(not value for value in requested):
            raise ValueError("function_asset_task_names_required")
        if len(set(requested)) != len(requested):
            raise ValueError("function_asset_task_names_duplicate")
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"source_function_task_missing:{','.join(missing)}")

    excluded = {
        str(value).strip()
        for value in exclude_task_names
        if str(value).strip()
    }
    excluded_present = sorted(set(requested) & excluded)
    selected = tuple(task for task in requested if task not in excluded)
    missing_authoring = sorted(set(selected) - set(raw_authoring_tasks))
    if missing_authoring:
        raise ValueError(
            "function_authoring_task_missing:" + ",".join(missing_authoring)
        )

    destination.mkdir(parents=True, exist_ok=True)
    task_reports: dict[str, dict[str, Any]] = {}
    store_index: dict[str, dict[str, Any]] = {}
    for task_name in selected:
        source_path = _source_run_log_path(
            available[task_name],
            source_index_path=source_index_path,
            task_name=task_name,
        )
        task_root = destination / "tasks" / task_name
        task_root.mkdir(parents=True, exist_ok=False)
        converted = _convert_runlog_to_function_asset(
            task_name=task_name,
            source_run_log=source_path,
            source_index_path=source_index_path,
            authoring_manifest_path=authoring_path,
            producer=producer,
            authoring_task=raw_authoring_tasks[task_name],
            output_root=task_root / "function_store",
        )
        report = {
            "task": task_name,
            "status": "converted",
            "target_inputs_read": False,
            "target_observations_read": False,
            **converted,
        }
        _write_json(task_root / "task_manifest.json", report)
        task_reports[task_name] = report
        store_index[task_name] = {
            key: converted[key]
            for key in (
                "store_path",
                "store_sha256",
                "transfer_states_path",
                "transfer_states_sha256",
                "provenance_path",
                "provenance_sha256",
            )
        }

    catalog = {
        "schema_version": CATALOG_SCHEMA,
        "source_asset_index": str(source_index_path),
        "source_asset_index_sha256": _sha256(source_index_path),
        "authoring_manifest": str(authoring_path),
        "authoring_manifest_sha256": _sha256(authoring_path),
        "model": None,
        "task_count": len(task_reports),
        "excluded_existing_task_count": len(excluded_present),
        "excluded_existing_tasks": excluded_present,
        "converted_task_count": len(store_index),
        "catalogued_task_count": 0,
        "target_inputs_read": False,
        "target_observations_read": False,
        "tasks": task_reports,
    }
    _write_json(destination / "store_index.json", store_index)
    _write_json(destination / "catalog.json", catalog)
    return catalog


def _convert_runlog_to_function_asset(
    *,
    task_name: str,
    source_run_log: str | Path,
    source_index_path: str | Path,
    authoring_manifest_path: str | Path,
    producer: dict[str, Any],
    authoring_task: dict[str, Any],
    output_root: str | Path,
) -> dict[str, Any]:
    source_path = Path(source_run_log).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(
            f"function_asset_source_run_log_missing:{task_name}:{source_path}"
        )
    index_path = Path(source_index_path).expanduser().resolve()
    authoring_path = Path(authoring_manifest_path).expanduser().resolve()
    output_path = Path(output_root).expanduser().resolve()
    expected_source_hash = str(
        authoring_task.get("source_run_log_sha256") or ""
    ).strip()
    actual_source_hash = _sha256(source_path)
    if expected_source_hash != actual_source_hash:
        raise ValueError(
            "function_authoring_source_run_log_hash_mismatch:"
            f"{task_name}:expected={expected_source_hash or 'missing'}:"
            f"actual={actual_source_hash}"
        )
    run_log, source_states = import_run_log_evidence(
        _read_object(source_path),
        evidence_root=source_path.parent,
        package_resolver=resolve_androidworld_package,
    )

    author_response = authoring_task["author_response"]
    compile_result = compile_runlog_to_store(
        run_log,
        output_path,
        function_bundle=author_response["bundle"],
        source_states=source_states,
    )
    store_path = Path(compile_result["store_path"]).resolve()
    transfer_path = Path(compile_result["transfer_state_catalog"]).resolve()
    store = FunctionStore(store_path)
    if store.load_errors:
        raise ValueError(
            "compiled_function_store_invalid:"
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
        "source_run_id": run_log["run_id"],
        "source_asset_index": str(index_path),
        "source_asset_index_sha256": _sha256(index_path),
        "semantic_collection": {
            "function": "androidworld_runlog_harvester_skill",
            "manifest_path": str(authoring_path),
            "manifest_sha256": _sha256(authoring_path),
            "producer": json.loads(json.dumps(producer)),
            "reason": author_response["reason"],
            "model": None,
            "model_calls": 0,
        },
        "function_ids": [function.id for function in store.list_functions()],
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
        "function_ids": provenance["function_ids"],
        "function_count": len(provenance["function_ids"]),
        "indexed_source_run_log": str(source_path),
        "indexed_source_run_log_sha256": _sha256(source_path),
        "source_run_log": str(source_path),
        "source_run_log_sha256": _sha256(source_path),
        "compiled_source_run_id": run_log["run_id"],
        "store_path": str(store_path),
        "store_sha256": _sha256(store_path),
        "transfer_states_path": str(transfer_path),
        "transfer_states_sha256": _sha256(transfer_path),
        "provenance_path": str(provenance_path),
        "provenance_sha256": _sha256(provenance_path),
        "source_target_audit": source_target_audit,
    }


def _validate_authoring_manifest(
    value: dict[str, Any],
    *,
    source_index_path: Path,
) -> None:
    if set(value) != {"schema_version", "source_asset_index_sha256", "producer", "tasks"}:
        raise ValueError("function_authoring_manifest_contract_invalid")
    if value.get("schema_version") != AUTHORING_SCHEMA:
        raise ValueError("unsupported_function_authoring_manifest_version")
    expected_hash = str(value.get("source_asset_index_sha256") or "").strip()
    actual_hash = _sha256(source_index_path)
    if expected_hash != actual_hash:
        raise ValueError(
            "function_authoring_source_index_hash_mismatch:"
            f"expected={expected_hash or 'missing'}:actual={actual_hash}"
        )
    producer = value.get("producer")
    if not isinstance(producer, dict) or producer.get("kind") != "androidworld_runlog_harvester_skill":
        raise ValueError("function_authoring_skill_producer_required")
    tasks = value.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError("function_authoring_tasks_required")
    for task_name, task in tasks.items():
        if not str(task_name).strip() or not isinstance(task, dict):
            raise ValueError("function_authoring_task_invalid")
        if set(task) != {"source_run_log_sha256", "author_response"}:
            raise ValueError(f"function_authoring_task_contract_invalid:{task_name}")
        if not str(task.get("source_run_log_sha256") or "").strip():
            raise ValueError(f"function_authoring_source_hash_required:{task_name}")
        response = task.get("author_response")
        if not isinstance(response, dict) or set(response) != {"reason", "bundle"}:
            raise ValueError(f"function_authoring_response_invalid:{task_name}")
        if not str(response.get("reason") or "").strip():
            raise ValueError(f"function_authoring_reason_required:{task_name}")
        bundle = response.get("bundle")
        if not isinstance(bundle, dict):
            raise ValueError(f"function_authoring_bundle_required:{task_name}")


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


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


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
            "Compile human-recorded source RunLogs with an immutable offline "
            "authoring manifest, validate, freeze, and register the result."
        )
    )
    parser.add_argument("--source-asset-index", required=True)
    parser.add_argument("--authoring-manifest", required=True)
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
            "Existing AndroidWorld memory current.json. Completed Function "
            "assets are registered before success is reported."
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
        source_asset_index=args.source_asset_index,
        authoring_manifest=args.authoring_manifest,
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
