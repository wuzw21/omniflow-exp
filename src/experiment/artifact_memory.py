#!/usr/bin/env python3
"""Content-addressed long-term memory for AndroidWorld experiment evidence."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Sequence

MEMORY_SCHEMA = "omniflow.androidworld-artifact-memory.v1"
CURRENT_SCHEMA = "omniflow.androidworld-artifact-memory-pointer.v1"
RESULT_FILE_NAMES = (
    "one_task_commands.jsonl",
    "one_task_summary.json",
    "registered_result.json",
    "registration_manifest.json",
    "stats.jsonl",
    "summary.json",
    "task_results.jsonl",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_object(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_index_reference(index_path: Path, value: Any) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("source_index_run_log_required")
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    for parent in (index_path.parent, *index_path.parents):
        candidate = (parent / path).resolve()
        if candidate.is_file():
            return candidate
    return (index_path.parent / path).resolve()


def _materialize_object(memory_root: Path, source: Path, digest: str) -> Path:
    target = memory_root / "objects" / "sha256" / digest[:2] / f"{digest}.json"
    if target.exists():
        if not target.is_file() or _sha256(target) != digest:
            raise ValueError(f"memory_object_hash_mismatch:{target}")
        return target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        # The object store must not share an inode with external immutable
        # evidence: making the object read-only must never alter source modes.
        shutil.copyfile(source, temporary)
        if _sha256(temporary) != digest:
            raise ValueError(f"memory_object_copy_hash_mismatch:{source}")
        temporary.chmod(0o444)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target.resolve()


def _require_hashed_file(value: Any, expected: Any, *, label: str) -> Path:
    path = Path(str(value or "")).expanduser().resolve()
    expected_hash = str(expected or "").strip()
    if not path.is_file():
        raise FileNotFoundError(f"{label}_missing:{path}")
    actual = _sha256(path)
    if not expected_hash or actual != expected_hash:
        raise ValueError(
            f"{label}_hash_mismatch:"
            f"expected={expected_hash or 'missing'}:actual={actual}"
        )
    return path


def _link_object(source: Path, target: Path, expected_hash: str) -> None:
    if target.exists():
        if not target.is_file() or _sha256(target) != expected_hash:
            raise ValueError(f"memory_runtime_hash_mismatch:{target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
        if _sha256(temporary) != expected_hash:
            raise ValueError(f"memory_runtime_copy_hash_mismatch:{source}")
        temporary.chmod(0o444)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_index(memory_root: Path, name: str, value: Any) -> tuple[Path, str]:
    content = _json_bytes(value)
    digest = hashlib.sha256(content).hexdigest()
    path = memory_root / "indexes" / f"{name}.{digest}.json"
    if path.exists():
        if _sha256(path) != digest:
            raise ValueError(f"memory_index_hash_mismatch:{path}")
    else:
        _atomic_write(path, content)
        path.chmod(0o444)
    return path.resolve(), digest


def _materialize_function_store(
    memory_root: Path,
    *,
    store_object: Path,
    store_sha256: str,
    transfer_object: Path,
    transfer_sha256: str,
    provenance_object: Path,
    provenance_sha256: str,
) -> dict[str, str]:
    identity = hashlib.sha256(
        "\0".join(
            (store_sha256, transfer_sha256, provenance_sha256)
        ).encode("utf-8")
    ).hexdigest()
    runtime_root = memory_root / "runtime" / "function_stores" / identity
    store_path = runtime_root / "store.json"
    transfer_path = runtime_root / "transfer_states.json"
    provenance_path = runtime_root / "provenance_manifest.json"
    _link_object(store_object, store_path, store_sha256)
    _link_object(transfer_object, transfer_path, transfer_sha256)
    _link_object(provenance_object, provenance_path, provenance_sha256)
    runtime_root.chmod(0o555)
    return {
        "store_path": str(store_path.resolve()),
        "store_sha256": store_sha256,
        "transfer_states_path": str(transfer_path.resolve()),
        "transfer_states_sha256": transfer_sha256,
        "provenance_path": str(provenance_path.resolve()),
        "provenance_sha256": provenance_sha256,
    }


def _runlog_paths(roots: Iterable[Path]) -> list[Path]:
    paths: set[Path] = set()
    for root in roots:
        resolved = root.expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"runlog_root_missing:{resolved}")
        paths.update(path.resolve() for path in resolved.rglob("*.run_log.json"))
        paths.update(path.resolve() for path in resolved.rglob("run_log.json"))
    return sorted(paths)


def _result_paths(roots: Iterable[Path]) -> list[Path]:
    paths: set[Path] = set()
    for root in roots:
        resolved = root.expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"result_root_missing:{resolved}")
        for name in RESULT_FILE_NAMES:
            paths.update(path.resolve() for path in resolved.rglob(name))
    return sorted(paths)


def _task_from_path(path: Path, task_names: Sequence[str]) -> str:
    parts = set(path.parts)
    matches = [task for task in task_names if task in parts]
    if matches:
        return sorted(matches, key=lambda task: (-len(task), task))[0]
    tasks_by_normalized_name: dict[str, list[str]] = {}
    for task in task_names:
        normalized = re.sub(r"[^a-z0-9]", "", task.lower())
        tasks_by_normalized_name.setdefault(normalized, []).append(task)
    normalized_parts = {
        re.sub(r"[^a-z0-9]", "", part.lower()) for part in path.parts
    }
    normalized_matches = {
        tasks[0]
        for normalized, tasks in tasks_by_normalized_name.items()
        if normalized in normalized_parts and len(tasks) == 1
    }
    if len(normalized_matches) == 1:
        return next(iter(normalized_matches))
    snake_names = {
        task: re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", task).lower()
        for task in task_names
    }
    prefix_matches = {
        task
        for task, snake_name in snake_names.items()
        if any(
            part.lower() == snake_name
            or part.lower().startswith(f"{snake_name}_")
            for part in path.parts
        )
    }
    if not prefix_matches:
        return ""
    longest = max(len(snake_names[task]) for task in prefix_matches)
    longest_matches = {
        task for task in prefix_matches if len(snake_names[task]) == longest
    }
    return next(iter(longest_matches)) if len(longest_matches) == 1 else ""


def _first_result_row(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    rows = payload.get("rows")
    if isinstance(rows, list):
        return next((row for row in rows if isinstance(row, dict)), {})
    return payload


def _read_result_payload(path: Path) -> Any:
    if path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                return json.loads(line)
        return {}
    return _load_object(path)


def _verified_registered_result(path: Path) -> dict[str, Any]:
    payload = _load_object(path)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version")
        != "omniflow.androidworld_registered_result.v1"
    ):
        raise ValueError("registered_result_schema_invalid")
    manifest_path = path.with_name("registration_manifest.json").resolve()
    recorded_manifest = Path(
        str(payload.get("registration_manifest") or "")
    ).expanduser()
    if not recorded_manifest.is_absolute():
        recorded_manifest = (path.parent / recorded_manifest).resolve()
    else:
        recorded_manifest = recorded_manifest.resolve()
    if recorded_manifest != manifest_path:
        raise ValueError("registered_result_manifest_path_mismatch")
    manifest = _load_object(manifest_path)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version")
        != "omniflow.androidworld_result_registration.v1"
        or manifest.get("immutable") is not True
    ):
        raise ValueError("registration_manifest_invalid")
    if str(manifest.get("registered_result_sha256") or "") != _sha256(path):
        raise ValueError("registered_result_hash_mismatch")
    for field in ("registration_id", "attempt_id", "task_name"):
        if str(payload.get(field) or "") != str(manifest.get(field) or ""):
            raise ValueError(f"registered_result_{field}_mismatch")
    rows = [row for row in payload.get("rows") or [] if isinstance(row, dict)]
    if len(rows) != 1:
        raise ValueError("registered_result_row_count_invalid")
    row = rows[0]
    method = str(row.get("method") or "")
    device = str(row.get("device") or "")
    if method != str(manifest.get("method") or ""):
        raise ValueError("registered_result_method_mismatch")
    if device != str(manifest.get("device") or ""):
        raise ValueError("registered_result_device_mismatch")
    official_used = row.get("official_validator_used") is True
    official_success = row.get("official_validator_success")
    try:
        validator_count = float(
            row.get("official_validator_task_count") or 0
        )
        validator_coverage = float(
            row.get("official_validator_coverage_rate") or 0
        )
    except (TypeError, ValueError) as error:
        raise ValueError("registered_result_validator_coverage_invalid") from error
    conclusion = (
        official_used and isinstance(official_success, bool)
    ) or validator_count > 0 or validator_coverage > 0
    return {
        "payload": payload,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "row": row,
        "validator_conclusion": conclusion,
    }


def _formal_device_label(value: Any) -> str:
    label = str(value or "").strip()
    return {
        "target5554": "small5554",
        "target5564": "fold5564",
    }.get(label, label)


def _load_results(
    memory_root: Path,
    roots: Sequence[Path],
    task_names: Sequence[str],
) -> tuple[
    list[Path],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    paths = _result_paths(roots)
    records: dict[str, dict[str, Any]] = {}
    candidates: dict[str, list[tuple[str, str, str, dict[str, Any]]]] = {}
    for path in paths:
        digest = _sha256(path)
        record = records.setdefault(
            digest,
            {
                "sha256": digest,
                "object_path": str(_materialize_object(memory_root, path, digest)),
                "aliases": [],
                "file_names": [],
                "tasks": [],
                "methods": [],
                "devices": [],
            },
        )
        record["aliases"].append(str(path))
        record["file_names"].append(path.name)
        try:
            payload = _read_result_payload(path)
            row = _first_result_row(payload)
            task = str(
                (payload.get("task_name") if isinstance(payload, dict) else "")
                or row.get("task_name")
                or row.get("task")
                or _task_from_path(path, task_names)
                or ""
            )
            method = str(
                row.get("method")
                or (payload.get("method") if isinstance(payload, dict) else "")
                or ""
            )
            device = str(
                row.get("device")
                or (payload.get("device") if isinstance(payload, dict) else "")
                or ""
            )
            if task:
                record["tasks"].append(task)
            if method:
                record["methods"].append(method)
            if device:
                record["devices"].append(device)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            record["parse_errors"] = sorted(
                set(
                    [
                        *record.get("parse_errors", []),
                        f"{path}:{type(error).__name__}:{error}",
                    ]
                )
            )
            continue
        if path.name != "registered_result.json":
            continue
        try:
            verified = _verified_registered_result(path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            record["verification_errors"] = sorted(
                set(
                    [
                        *record.get("verification_errors", []),
                        f"{path}:{type(error).__name__}:{error}",
                    ]
                )
            )
            continue
        record["verified_registration"] = True
        if not verified["validator_conclusion"]:
            continue
        result_row = verified["row"]
        result_payload = verified["payload"]
        manifest = verified["manifest"]
        task = str(result_payload["task_name"])
        method = str(result_row["method"])
        registered_device_label = str(result_row["device"])
        device = _formal_device_label(registered_device_label)
        manifest_path = verified["manifest_path"]
        manifest_digest = _sha256(manifest_path)
        manifest_object = _materialize_object(
            memory_root,
            manifest_path,
            manifest_digest,
        )
        candidate = {
            "task": task,
            "method": method,
            "device": device,
            "registered_device_label": registered_device_label,
            "registration_id": str(result_payload.get("registration_id") or ""),
            "attempt_id": str(result_payload.get("attempt_id") or ""),
            "registered_at": str(manifest.get("registered_at") or ""),
            "official_validator_success": result_row.get(
                "official_validator_success"
            ),
            "registered_result_sha256": digest,
            "registered_result_object_path": record["object_path"],
            "registered_result_aliases": sorted(set(record["aliases"])),
            "registration_manifest_sha256": manifest_digest,
            "registration_manifest_object_path": str(manifest_object),
            "selection_reason": (
                "earliest_verified_official_validator_conclusion"
            ),
        }
        cell = f"{task}|{method}|{device}"
        candidates.setdefault(cell, []).append(
            (
                candidate["registered_at"],
                candidate["registration_id"],
                str(path),
                candidate,
            )
        )

    for record in records.values():
        for field in ("aliases", "file_names", "tasks", "methods", "devices"):
            record[field] = sorted(set(record[field]))
    canonical = {
        cell: sorted(cell_candidates, key=lambda value: value[:3])[0][3]
        for cell, cell_candidates in sorted(candidates.items())
    }
    return paths, records, canonical


def _load_function_stores(
    memory_root: Path,
    catalogs: Sequence[Path],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    records: dict[str, dict[str, Any]] = {}
    candidates: dict[str, list[tuple[tuple[int, int], str]]] = {}
    for catalog_path in catalogs:
        if not catalog_path.is_file():
            raise FileNotFoundError(f"function_catalog_missing:{catalog_path}")
        catalog_digest = _sha256(catalog_path)
        _materialize_object(memory_root, catalog_path, catalog_digest)
        payload = _load_object(catalog_path)
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version")
            != "omniflow.function-asset-catalog.v1"
            or not isinstance(payload.get("tasks"), dict)
        ):
            raise ValueError(f"function_catalog_invalid:{catalog_path}")
        for raw_task, raw_item in payload["tasks"].items():
            task = str(raw_task)
            if not isinstance(raw_item, dict) or raw_item.get("status") != "converted":
                continue
            if raw_item.get("target_inputs_read") is not False:
                raise ValueError(f"function_catalog_target_inputs_read:{task}")
            if raw_item.get("target_observations_read") is not False:
                raise ValueError(f"function_catalog_target_observations_read:{task}")
            store = _require_hashed_file(
                raw_item.get("store_path"),
                raw_item.get("store_sha256"),
                label=f"function_store:{task}",
            )
            transfer = _require_hashed_file(
                raw_item.get("transfer_states_path"),
                raw_item.get("transfer_states_sha256"),
                label=f"function_transfer_states:{task}",
            )
            provenance = _require_hashed_file(
                raw_item.get("provenance_path"),
                raw_item.get("provenance_sha256"),
                label=f"function_provenance:{task}",
            )
            store_payload = _load_object(store)
            if (
                not isinstance(store_payload, dict)
                or store_payload.get("schema_version") != "omniflow.store.v2"
                or not isinstance(store_payload.get("functions"), dict)
            ):
                raise ValueError(f"function_store_invalid:{task}:{store}")
            store_hash = _sha256(store)
            transfer_hash = _sha256(transfer)
            provenance_hash = _sha256(provenance)
            identity = hashlib.sha256(
                "\0".join(
                    (store_hash, transfer_hash, provenance_hash)
                ).encode("utf-8")
            ).hexdigest()
            store_object = _materialize_object(memory_root, store, store_hash)
            transfer_object = _materialize_object(
                memory_root,
                transfer,
                transfer_hash,
            )
            provenance_object = _materialize_object(
                memory_root,
                provenance,
                provenance_hash,
            )
            record = records.setdefault(
                identity,
                {
                    "identity_sha256": identity,
                    "tasks": [],
                    "catalog_aliases": [],
                    "function_count": len(store_payload["functions"]),
                    **_materialize_function_store(
                        memory_root,
                        store_object=store_object,
                        store_sha256=store_hash,
                        transfer_object=transfer_object,
                        transfer_sha256=transfer_hash,
                        provenance_object=provenance_object,
                        provenance_sha256=provenance_hash,
                    ),
                },
            )
            record["tasks"].append(task)
            record["catalog_aliases"].append(str(catalog_path))
            quality = (1, len(store_payload["functions"]))
            candidates.setdefault(task, []).append((quality, identity))

    for record in records.values():
        record["tasks"] = sorted(set(record["tasks"]))
        record["catalog_aliases"] = sorted(set(record["catalog_aliases"]))

    canonical: dict[str, dict[str, Any]] = {}
    for task, task_candidates in sorted(candidates.items()):
        best_quality = max(quality for quality, _ in task_candidates)
        best_ids = sorted(
            {
                identity
                for quality, identity in task_candidates
                if quality == best_quality
            }
        )
        if len(best_ids) != 1:
            raise ValueError(
                f"ambiguous_best_function_store:{task}:{','.join(best_ids)}"
            )
        canonical[task] = dict(records[best_ids[0]])
    return records, canonical


@contextmanager
def _memory_lock(memory_root: Path):
    memory_root.mkdir(parents=True, exist_ok=True)
    lock_path = memory_root / ".memory.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _refresh_artifact_memory_unlocked(
    *,
    memory_root: str | Path,
    source_index: str | Path,
    function_catalogs: Sequence[str | Path],
    runlog_roots: Sequence[str | Path],
    result_roots: Sequence[str | Path],
) -> dict[str, Any]:
    """Import immutable evidence and publish one deterministic canonical index."""

    root = Path(memory_root).expanduser().resolve()
    index_path = Path(source_index).expanduser().resolve()
    if not index_path.is_file():
        raise FileNotFoundError(f"source_index_missing:{index_path}")
    source_payload = _load_object(index_path)
    if not isinstance(source_payload, dict):
        raise ValueError("source_index_must_be_object")
    task_names = sorted(str(task) for task in source_payload)
    indexed_paths: dict[Path, str] = {}
    for task, item in source_payload.items():
        if not isinstance(item, dict):
            raise ValueError(f"source_index_item_invalid:{task}")
        path = _resolve_index_reference(
            index_path,
            item.get("retained_source_run_log") or item.get("source_run_log"),
        )
        if not path.is_file():
            raise FileNotFoundError(f"indexed_source_run_log_missing:{task}:{path}")
        indexed_paths[path] = str(task)

    paths = _runlog_paths(Path(value) for value in runlog_roots)
    for path in indexed_paths:
        if path not in paths:
            paths.append(path)
    paths.sort()

    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        digest = _sha256(path)
        payload = _load_object(path)
        if not isinstance(payload, dict):
            raise ValueError(f"run_log_must_be_object:{path}")
        task = indexed_paths.get(path) or _task_from_path(path, task_names)
        record = records.setdefault(
            digest,
            {
                "sha256": digest,
                "object_path": str(_materialize_object(root, path, digest)),
                "aliases": [],
                "tasks": [],
                "schema_version": str(payload.get("schema_version") or ""),
                "run_id": str(payload.get("run_id") or ""),
                "success": payload.get("success")
                if isinstance(payload.get("success"), bool)
                else None,
                "step_count": len(payload.get("steps") or []),
            },
        )
        record["aliases"].append(str(path))
        if task:
            record["tasks"].append(task)

    for record in records.values():
        record["aliases"] = sorted(set(record["aliases"]))
        record["tasks"] = sorted(set(record["tasks"]))

    canonical_sources: dict[str, dict[str, Any]] = {}
    for path, task in sorted(indexed_paths.items(), key=lambda item: item[1]):
        digest = _sha256(path)
        canonical_sources[task] = dict(records[digest])

    catalog_paths = sorted(
        {Path(value).expanduser().resolve() for value in function_catalogs}
    )
    resolved_result_roots = sorted(
        {Path(value).expanduser().resolve() for value in result_roots}
    )
    function_records, canonical_function_stores = _load_function_stores(
        root,
        catalog_paths,
    )
    result_paths, result_records, canonical_result_cells = _load_results(
        root,
        resolved_result_roots,
        task_names,
    )
    memory_source_index: dict[str, Any] = {}
    for task, raw_item in source_payload.items():
        item = dict(raw_item)
        item["retained_source_run_log"] = canonical_sources[str(task)]["object_path"]
        item["retained_source_run_log_sha256"] = canonical_sources[str(task)][
            "sha256"
        ]
        memory_source_index[str(task)] = item
    memory_source_index_path, memory_source_index_hash = _publish_index(
        root,
        "source_index",
        memory_source_index,
    )
    store_index = {
        task: {
            key: record[key]
            for key in (
                "store_path",
                "store_sha256",
                "transfer_states_path",
                "transfer_states_sha256",
                "provenance_path",
                "provenance_sha256",
            )
        }
        for task, record in canonical_function_stores.items()
    }
    store_index_path, store_index_hash = _publish_index(
        root,
        "ours_store_index",
        store_index,
    )
    result_cells_path, result_cells_hash = _publish_index(
        root,
        "result_cells",
        canonical_result_cells,
    )
    by_task: dict[str, dict[str, Any]] = {}
    for task in task_names:
        by_task[task] = {
            "task": task,
            "run_log_sha256s": sorted(
                digest
                for digest, record in records.items()
                if task in record["tasks"]
            ),
            "function_store_identity_sha256s": sorted(
                digest
                for digest, record in function_records.items()
                if task in record["tasks"]
            ),
            "result_sha256s": sorted(
                digest
                for digest, record in result_records.items()
                if task in record["tasks"]
            ),
            "canonical": {
                "source_run_log": canonical_sources.get(task),
                "function_store": canonical_function_stores.get(task),
                "result_cells": {
                    cell: value
                    for cell, value in canonical_result_cells.items()
                    if cell.split("|", 1)[0] == task
                },
            },
        }
    unclassified_result_hashes = sorted(
        digest
        for digest, record in result_records.items()
        if not record["tasks"]
    )
    unclassified_runlog_hashes = sorted(
        digest for digest, record in records.items() if not record["tasks"]
    )

    registry: dict[str, Any] = {
        "schema_version": MEMORY_SCHEMA,
        "policy": {
            "deduplication": "exact_sha256",
            "source_run_log": "source_index_authoritative",
            "result": "earliest_verified_official_validator_conclusion",
            "success_cherry_picking": False,
        },
        "inputs": {
            "source_index": str(index_path),
            "source_index_sha256": _sha256(index_path),
            "function_catalogs": [str(path) for path in catalog_paths],
            "runlog_roots": sorted(
                str(Path(value).expanduser().resolve()) for value in runlog_roots
            ),
            "result_roots": [str(path) for path in resolved_result_roots],
        },
        "counts": {
            "task_count": len(task_names),
            "run_log_paths": len(paths),
            "unique_run_logs": len(records),
            "function_catalog_paths": len(catalog_paths),
            "unique_function_stores": len(function_records),
            "function_store_tasks": len(canonical_function_stores),
            "result_paths": len(result_paths),
            "unique_results": len(result_records),
            "canonical_result_cells": len(canonical_result_cells),
        },
        "indexes": {
            "source_index": str(memory_source_index_path),
            "source_index_sha256": memory_source_index_hash,
            "ours_store_index": str(store_index_path),
            "ours_store_index_sha256": store_index_hash,
            "result_cells": str(result_cells_path),
            "result_cells_sha256": result_cells_hash,
        },
        "artifacts": {
            "run_logs": {digest: records[digest] for digest in sorted(records)},
            "function_stores": {
                digest: function_records[digest]
                for digest in sorted(function_records)
            },
            "results": {
                digest: result_records[digest] for digest in sorted(result_records)
            },
        },
        "canonical": {
            "source_run_logs": canonical_sources,
            "function_stores": canonical_function_stores,
            "result_cells": canonical_result_cells,
        },
        "by_task": by_task,
        "unclassified": {
            "run_log_sha256s": unclassified_runlog_hashes,
            "result_sha256s": unclassified_result_hashes,
        },
    }
    registry_bytes = _json_bytes(registry)
    registry_sha256 = hashlib.sha256(registry_bytes).hexdigest()
    snapshot_root = root / "snapshots" / registry_sha256
    registry_path = snapshot_root / "registry.json"
    if registry_path.exists():
        if _sha256(registry_path) != registry_sha256:
            raise ValueError(f"memory_snapshot_hash_mismatch:{registry_path}")
    else:
        snapshot_root.mkdir(parents=True, exist_ok=False)
        _atomic_write(registry_path, registry_bytes)
        registry_path.chmod(0o444)
        by_task_root = snapshot_root / "by_task"
        by_task_root.mkdir()
        for task, task_payload in by_task.items():
            if not task.isalnum():
                raise ValueError(f"memory_task_name_unsafe:{task}")
            task_path = by_task_root / f"{task}.json"
            _atomic_write(task_path, _json_bytes(task_payload))
            task_path.chmod(0o444)
        if unclassified_runlog_hashes or unclassified_result_hashes:
            unclassified_path = by_task_root / "_unclassified.json"
            _atomic_write(
                unclassified_path,
                _json_bytes(registry["unclassified"]),
            )
            unclassified_path.chmod(0o444)
        by_task_root.chmod(0o555)
        snapshot_root.chmod(0o555)
    by_task_root = snapshot_root / "by_task"
    if not by_task_root.is_dir():
        raise ValueError(f"memory_by_task_index_missing:{by_task_root}")
    pointer = {
        "schema_version": CURRENT_SCHEMA,
        "registry_path": str(registry_path.resolve()),
        "registry_sha256": registry_sha256,
        "source_index": str(memory_source_index_path),
        "source_index_sha256": memory_source_index_hash,
        "ours_store_index": str(store_index_path),
        "ours_store_index_sha256": store_index_hash,
        "result_cells": str(result_cells_path),
        "result_cells_sha256": result_cells_hash,
        "by_task_root": str(by_task_root.resolve()),
    }
    _atomic_write(root / "current.json", _json_bytes(pointer))
    return registry


def refresh_artifact_memory(
    *,
    memory_root: str | Path,
    source_index: str | Path,
    function_catalogs: Sequence[str | Path],
    runlog_roots: Sequence[str | Path],
    result_roots: Sequence[str | Path],
) -> dict[str, Any]:
    """Import immutable evidence and publish one deterministic canonical index."""

    root = Path(memory_root).expanduser().resolve()
    with _memory_lock(root):
        return _refresh_artifact_memory_unlocked(
            memory_root=root,
            source_index=source_index,
            function_catalogs=function_catalogs,
            runlog_roots=runlog_roots,
            result_roots=result_roots,
        )


def load_artifact_memory(memory_index: str | Path) -> dict[str, Any]:
    """Load and verify the registry selected by ``current.json``."""

    pointer_path = Path(memory_index).expanduser().resolve()
    pointer = _load_object(pointer_path)
    if (
        not isinstance(pointer, dict)
        or pointer.get("schema_version") != CURRENT_SCHEMA
    ):
        raise ValueError(f"artifact_memory_pointer_invalid:{pointer_path}")
    registry_path = Path(str(pointer.get("registry_path") or "")).expanduser()
    if not registry_path.is_absolute() or not registry_path.is_file():
        raise FileNotFoundError(
            f"artifact_memory_registry_missing:{registry_path}"
        )
    expected = str(pointer.get("registry_sha256") or "")
    actual = _sha256(registry_path)
    if not expected or actual != expected:
        raise ValueError(
            "artifact_memory_registry_hash_mismatch:"
            f"expected={expected or 'missing'}:actual={actual}"
        )
    registry = _load_object(registry_path)
    if (
        not isinstance(registry, dict)
        or registry.get("schema_version") != MEMORY_SCHEMA
    ):
        raise ValueError(f"artifact_memory_registry_invalid:{registry_path}")
    for path_field, hash_field in (
        ("source_index", "source_index_sha256"),
        ("ours_store_index", "ours_store_index_sha256"),
        ("result_cells", "result_cells_sha256"),
    ):
        path = Path(str(pointer.get(path_field) or "")).expanduser()
        expected_hash = str(pointer.get(hash_field) or "")
        if not path.is_absolute() or not path.is_file():
            raise FileNotFoundError(
                f"artifact_memory_index_missing:{path_field}:{path}"
            )
        if not expected_hash or _sha256(path) != expected_hash:
            raise ValueError(
                f"artifact_memory_index_hash_mismatch:{path_field}:{path}"
            )
    return registry


def registered_cell_plan_from_memory(
    *,
    memory_index: str | Path,
    task_name: str,
    methods: Sequence[str],
    devices: Sequence[str],
) -> dict[str, list[tuple[str, str]]]:
    """Resolve completed formal cells without rescanning historical results."""

    registry = load_artifact_memory(memory_index)
    cells = registry["canonical"]["result_cells"]
    expected = [(method, device) for method in methods for device in devices]
    completed = [
        (method, device)
        for method, device in expected
        if f"{task_name}|{method}|{device}" in cells
    ]
    return {
        "completed": completed,
        "pending": [cell for cell in expected if cell not in completed],
    }


def refresh_artifact_memory_from_pointer(
    *,
    memory_index: str | Path,
    additional_function_catalogs: Sequence[str | Path] = (),
    additional_runlog_roots: Sequence[str | Path] = (),
    additional_result_roots: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Refresh a memory using its recorded inputs plus newly completed evidence."""

    pointer_path = Path(memory_index).expanduser().resolve()
    with _memory_lock(pointer_path.parent):
        registry = load_artifact_memory(pointer_path)
        inputs = registry["inputs"]
        function_catalogs = sorted(
            {
                *(str(value) for value in inputs.get("function_catalogs") or []),
                *(
                    str(Path(value).expanduser().resolve())
                    for value in additional_function_catalogs
                ),
            }
        )
        runlog_roots = sorted(
            {
                *(str(value) for value in inputs.get("runlog_roots") or []),
                *(
                    str(Path(value).expanduser().resolve())
                    for value in additional_runlog_roots
                ),
            }
        )
        result_roots = sorted(
            {
                *(str(value) for value in inputs.get("result_roots") or []),
                *(
                    str(Path(value).expanduser().resolve())
                    for value in additional_result_roots
                ),
            }
        )
        return _refresh_artifact_memory_unlocked(
            memory_root=pointer_path.parent,
            source_index=str(inputs["source_index"]),
            function_catalogs=function_catalogs,
            runlog_roots=runlog_roots,
            result_roots=result_roots,
        )


def _split_values(values: Sequence[str]) -> list[str]:
    return [
        item
        for value in values
        for item in value.split(":")
        if item.strip()
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Maintain content-addressed long-term memory for AndroidWorld "
            "RunLogs, Function assets, and registered results."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    refresh_parser = subparsers.add_parser("refresh")
    refresh_parser.add_argument("--memory-root", required=True)
    refresh_parser.add_argument("--source-index", required=True)
    refresh_parser.add_argument("--function-catalog", action="append", default=[])
    refresh_parser.add_argument("--runlog-root", action="append", required=True)
    refresh_parser.add_argument("--result-root", action="append", default=[])
    paths_parser = subparsers.add_parser("paths")
    paths_parser.add_argument("--memory-index", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--memory-index", required=True)
    plan_parser.add_argument("--task", required=True)
    plan_parser.add_argument("--methods", required=True)
    plan_parser.add_argument("--devices", required=True)
    args = parser.parse_args(argv)

    if args.command == "refresh":
        report = refresh_artifact_memory(
            memory_root=args.memory_root,
            source_index=args.source_index,
            function_catalogs=_split_values(args.function_catalog),
            runlog_roots=_split_values(args.runlog_root),
            result_roots=_split_values(args.result_root),
        )
        pointer = _load_object(
            Path(args.memory_root).expanduser().resolve() / "current.json"
        )
        output = {
            "current": str(
                Path(args.memory_root).expanduser().resolve() / "current.json"
            ),
            "registry": pointer["registry_path"],
            "counts": report["counts"],
        }
    elif args.command == "paths":
        load_artifact_memory(args.memory_index)
        output = _load_object(Path(args.memory_index).expanduser().resolve())
    else:
        output = registered_cell_plan_from_memory(
            memory_index=args.memory_index,
            task_name=args.task,
            methods=tuple(item for item in args.methods.split(",") if item),
            devices=tuple(item for item in args.devices.split(",") if item),
        )
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


__all__ = [
    "load_artifact_memory",
    "refresh_artifact_memory",
    "refresh_artifact_memory_from_pointer",
    "registered_cell_plan_from_memory",
]


if __name__ == "__main__":
    raise SystemExit(main())
