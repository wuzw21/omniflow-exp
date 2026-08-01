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

from omniflow.core.trajectory import require_complete_source_run_log
from omniflow.transfer.runtime import load_transfer_state_catalog
from src.integrations.runlog import adapt_source_run_log

MEMORY_SCHEMA = "omniflow.androidworld-artifact-memory.v2"
CURRENT_SCHEMA = "omniflow.androidworld-artifact-memory-pointer.v2"
SOURCE_SELECTION_SCHEMA = "omniflow.androidworld-source-selection.v1"
FUNCTION_SOURCE_LINEAGE_SCHEMA = "omniflow.function-store-source-lineage.v1"
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


def _materialize_content(memory_root: Path, content: bytes, digest: str) -> Path:
    target = memory_root / "objects" / "sha256" / digest[:2] / f"{digest}.json"
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError(f"memory_content_hash_mismatch:{digest}")
    if target.exists():
        if not target.is_file() or _sha256(target) != digest:
            raise ValueError(f"memory_object_hash_mismatch:{target}")
        return target.resolve()
    _atomic_write(target, content)
    target.chmod(0o444)
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
        "\0".join((store_sha256, transfer_sha256, provenance_sha256)).encode("utf-8")
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


def _function_source_seed(item: dict[str, Any]) -> int | None:
    for field in ("source_seed", "replay_seed", "collect_seed", "task_random_seed"):
        value = item.get(field)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _register_run_log_record(
    records: dict[str, dict[str, Any]],
    *,
    path: Path,
    digest: str,
    payload: dict[str, Any],
    task: str,
    alias: str = "",
) -> dict[str, Any]:
    record = records.setdefault(
        digest,
        {
            "sha256": digest,
            "object_path": str(path.resolve()),
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
    if alias:
        record["aliases"] = sorted(set(record["aliases"]) | {alias})
    if task:
        record["tasks"] = sorted(set(record["tasks"]) | {task})
    return record


def _canonicalize_function_source_run_log(
    memory_root: Path,
    *,
    task: str,
    source_run_log: Path,
    source_payload: dict[str, Any],
    source_metadata: dict[str, Any],
    canonical_source: dict[str, Any],
    screenshot_roots: Sequence[Path],
    records: dict[str, dict[str, Any]],
) -> tuple[Path, str, dict[str, Any]]:
    source_sha256 = _sha256(source_run_log)
    source_object = _materialize_object(
        memory_root,
        source_run_log,
        source_sha256,
    )
    _register_run_log_record(
        records,
        path=source_object,
        digest=source_sha256,
        payload=source_payload,
        task=task,
        alias=str(source_run_log),
    )
    try:
        canonical = require_complete_source_run_log(source_payload)
        conversion = "identity"
    except ValueError:
        canonical_object = _require_hashed_file(
            canonical_source.get("object_path"),
            canonical_source.get("sha256"),
            label=f"canonical_source_run_log:{task}",
        )
        canonical_candidate = require_complete_source_run_log(
            _load_object(canonical_object)
        )
        if canonical_candidate["task_name"] != task:
            raise ValueError(
                "canonical_function_source_task_mismatch:"
                f"{task}:{canonical_candidate['task_name']}"
            )
        canonical_provenance = canonical_candidate.get("provenance")
        canonical_provenance_source_sha256 = (
            str(canonical_provenance.get("source_sha256") or "").strip()
            if isinstance(canonical_provenance, dict)
            else ""
        )
        if canonical_provenance_source_sha256 == source_sha256:
            canonical = canonical_candidate
            conversion = "canonical_source_reuse"
        else:
            source_states = None
            source_catalog_value = source_metadata.get(
                "source_state_catalog"
            ) or source_metadata.get("transfer_state_catalog")
            if source_catalog_value:
                source_catalog_path = _require_hashed_file(
                    source_catalog_value,
                    source_metadata.get("source_state_catalog_sha256")
                    or source_metadata.get("transfer_state_catalog_sha256"),
                    label=f"function_source_state_catalog:{task}",
                )
                source_states = load_transfer_state_catalog(source_catalog_path)
            canonical = require_complete_source_run_log(
                adapt_source_run_log(
                    source_payload,
                    task_name=task,
                    task_parameters=dict(
                        source_metadata.get("params")
                        or source_metadata.get("task_parameters")
                        or {}
                    ),
                    seed=_function_source_seed(source_metadata),
                    source_path=source_object,
                    source_states=source_states,
                    screenshot_roots=screenshot_roots,
                    require_screenshots=False,
                )
            )
            conversion = "legacy_import"
    if conversion == "identity":
        canonical_sha256 = source_sha256
        canonical_object = source_object
    elif conversion == "canonical_source_reuse":
        canonical_sha256 = str(canonical_source["sha256"])
    else:
        canonical_content = _json_bytes(canonical)
        canonical_sha256 = hashlib.sha256(canonical_content).hexdigest()
        canonical_object = _materialize_content(
            memory_root,
            canonical_content,
            canonical_sha256,
        )
    _register_run_log_record(
        records,
        path=canonical_object,
        digest=canonical_sha256,
        payload=canonical,
        task=task,
    )
    lineage = {
        "schema_version": FUNCTION_SOURCE_LINEAGE_SCHEMA,
        "conversion": conversion,
        "source_path": str(source_object),
        "source_sha256": source_sha256,
        "source_schema_version": str(source_payload.get("schema_version") or ""),
        "output_path": str(canonical_object),
        "output_sha256": canonical_sha256,
    }
    return canonical_object, canonical_sha256, lineage


def _load_source_selections(
    manifest_path: str | Path | None,
    *,
    memory_root: Path,
    source_payload: dict[str, Any],
    index_path: Path,
    records: dict[str, dict[str, Any]],
    screenshot_roots: Sequence[Path],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    if manifest_path is None or not str(manifest_path).strip():
        return {}, {}
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"source_selection_manifest_missing:{path}")
    payload = _load_object(path)
    raw_selections = payload.get("selections") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != SOURCE_SELECTION_SCHEMA
        or not isinstance(raw_selections, dict)
        or not raw_selections
    ):
        raise ValueError(f"source_selection_manifest_invalid:{path}")
    unknown_tasks = sorted(set(raw_selections) - set(source_payload))
    if unknown_tasks:
        raise ValueError("source_selection_tasks_unknown:" + ",".join(unknown_tasks))
    selections: dict[str, dict[str, Any]] = {}
    for task, raw_selection in sorted(raw_selections.items()):
        if not isinstance(raw_selection, dict):
            raise ValueError(f"source_selection_invalid:{task}")
        expected = (
            str(raw_selection.get("expected_source_run_log_sha256") or "")
            .strip()
            .lower()
        )
        selected_run_log_sha256 = (
            str(raw_selection.get("selected_source_run_log_sha256") or "")
            .strip()
            .lower()
        )
        selected_evidence_sha256 = (
            str(raw_selection.get("selected_source_evidence_sha256") or "")
            .strip()
            .lower()
        )
        expected_converted_sha256 = (
            str(raw_selection.get("expected_converted_source_run_log_sha256") or "")
            .strip()
            .lower()
        )
        reason = str(raw_selection.get("reason") or "").strip()
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"source_selection_expected_sha256_invalid:{task}")
        if not reason:
            raise ValueError(f"source_selection_reason_required:{task}")
        direct_selection = bool(selected_run_log_sha256)
        conversion_selection = bool(
            selected_evidence_sha256 or expected_converted_sha256
        )
        if direct_selection == conversion_selection:
            raise ValueError(f"source_selection_mode_invalid:{task}")
        raw_item = source_payload[task]
        if not isinstance(raw_item, dict):
            raise ValueError(f"source_index_item_invalid:{task}")
        baseline_path = _resolve_index_reference(
            index_path,
            raw_item.get("retained_source_run_log") or raw_item.get("source_run_log"),
        )
        baseline = _sha256(baseline_path)
        if direct_selection:
            if not re.fullmatch(r"[0-9a-f]{64}", selected_run_log_sha256):
                raise ValueError(f"source_selection_selected_sha256_invalid:{task}")
            selected_sha256 = selected_run_log_sha256
            if expected == selected_sha256:
                raise ValueError(f"source_selection_noop:{task}")
            if baseline not in {expected, selected_sha256}:
                raise ValueError(
                    f"source_selection_stale:{task}:"
                    f"expected={expected}:selected={selected_sha256}:"
                    f"actual={baseline}"
                )
            candidate = records.get(selected_sha256)
            if candidate is None:
                raise ValueError(
                    f"source_selection_candidate_unregistered:{task}:{selected_sha256}"
                )
            selected_run_log = require_complete_source_run_log(
                _load_object(Path(candidate["object_path"]))
            )
            conversion: dict[str, Any] | None = None
        else:
            if not re.fullmatch(r"[0-9a-f]{64}", selected_evidence_sha256):
                raise ValueError(f"source_selection_evidence_sha256_invalid:{task}")
            if not re.fullmatch(r"[0-9a-f]{64}", expected_converted_sha256):
                raise ValueError(f"source_selection_converted_sha256_invalid:{task}")
            selected_sha256 = expected_converted_sha256
            if expected == selected_sha256:
                raise ValueError(f"source_selection_noop:{task}")
            if baseline not in {expected, selected_sha256}:
                raise ValueError(
                    f"source_selection_stale:{task}:"
                    f"expected={expected}:selected={selected_sha256}:"
                    f"actual={baseline}"
                )
            task_parameters = raw_selection.get("task_parameters")
            if not isinstance(task_parameters, dict):
                raise ValueError(f"source_selection_task_parameters_invalid:{task}")
            source_seed = raw_selection.get("source_seed")
            if (
                not isinstance(source_seed, int)
                or isinstance(source_seed, bool)
                or source_seed < 0
            ):
                raise ValueError(f"source_selection_source_seed_invalid:{task}")
            evidence = records.get(selected_evidence_sha256)
            if evidence is None:
                raise ValueError(
                    f"source_selection_evidence_unregistered:{task}:"
                    f"{selected_evidence_sha256}"
                )
            evidence_path = Path(evidence["object_path"])
            selected_run_log = require_complete_source_run_log(
                adapt_source_run_log(
                    _load_object(evidence_path),
                    task_name=task,
                    task_parameters=task_parameters,
                    seed=source_seed,
                    source_path=evidence_path,
                    screenshot_roots=screenshot_roots,
                    require_screenshots=False,
                )
            )
            converted_content = _json_bytes(selected_run_log)
            actual_converted_sha256 = hashlib.sha256(converted_content).hexdigest()
            if actual_converted_sha256 != expected_converted_sha256:
                raise ValueError(
                    "source_selection_converted_hash_mismatch:"
                    f"{task}:expected={expected_converted_sha256}:"
                    f"actual={actual_converted_sha256}"
                )
            converted_path = _materialize_content(
                memory_root,
                converted_content,
                expected_converted_sha256,
            )
            record = records.setdefault(
                expected_converted_sha256,
                {
                    "sha256": expected_converted_sha256,
                    "object_path": str(converted_path),
                    "aliases": [],
                    "tasks": [],
                    "schema_version": str(selected_run_log["schema_version"]),
                    "run_id": str(selected_run_log["run_id"]),
                    "success": selected_run_log["success"],
                    "step_count": len(selected_run_log["steps"]),
                },
            )
            record["tasks"] = sorted(set(record["tasks"]) | {task})
            conversion = {
                "kind": "legacy_evidence_to_official_run_log",
                "selected_source_evidence_sha256": selected_evidence_sha256,
                "expected_converted_source_run_log_sha256": (expected_converted_sha256),
                "source_seed": source_seed,
                "task_parameters": json.loads(
                    json.dumps(task_parameters, ensure_ascii=False)
                ),
                "screenshot_roots": [str(root) for root in screenshot_roots],
            }
        if selected_run_log["task_name"] != task:
            raise ValueError(
                "source_selection_task_mismatch:"
                f"{task}:{selected_run_log['task_name']}:{selected_sha256}"
            )
        selections[task] = {
            "expected_source_run_log_sha256": expected,
            "selected_source_run_log_sha256": selected_sha256,
            "reason": reason,
            "selected_run_log": selected_run_log,
        }
        if conversion is not None:
            selections[task]["conversion"] = conversion
    return selections, {
        "path": str(path),
        "sha256": _sha256(path),
    }


def _task_from_path(path: Path, task_names: Sequence[str]) -> str:
    parts = set(path.parts)
    matches = [task for task in task_names if task in parts]
    if matches:
        return sorted(matches, key=lambda task: (-len(task), task))[0]
    tasks_by_normalized_name: dict[str, list[str]] = {}
    for task in task_names:
        normalized = re.sub(r"[^a-z0-9]", "", task.lower())
        tasks_by_normalized_name.setdefault(normalized, []).append(task)
    normalized_parts = {re.sub(r"[^a-z0-9]", "", part.lower()) for part in path.parts}
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
            part.lower() == snake_name or part.lower().startswith(f"{snake_name}_")
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
        or payload.get("schema_version") != "omniflow.androidworld_registered_result.v1"
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
    for field in (
        "registration_id",
        "attempt_id",
        "task_name",
        "source_seed",
        "evaluation_seed",
    ):
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
        validator_count = float(row.get("official_validator_task_count") or 0)
        validator_coverage = float(row.get("official_validator_coverage_rate") or 0)
    except (TypeError, ValueError) as error:
        raise ValueError("registered_result_validator_coverage_invalid") from error
    validator_error = str(row.get("error") or "").strip()
    conclusion = not validator_error and (
        (official_used and isinstance(official_success, bool))
        or validator_count > 0
        or validator_coverage > 0
    )
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


def _formal_result_protocol_error(
    *,
    task_names: Sequence[str],
    task: str,
    method: str,
    device: str,
    source_seed: Any,
    evaluation_seed: Any,
    row: dict[str, Any],
) -> str | None:
    from src.experiment.result_registry import (
        FORMAL_DEVICE_TARGETS,
        FORMAL_EVALUATION_SEED,
        FORMAL_MAX_STEPS,
        FORMAL_METHODS,
        FORMAL_SOURCE_SEED,
        validate_formal_result_protocol,
    )

    violations: list[str] = []
    if task not in task_names:
        violations.append("task_not_indexed")
    if method not in FORMAL_METHODS:
        violations.append("unsupported_method")
    if device not in FORMAL_DEVICE_TARGETS:
        violations.append("unsupported_device")
    if source_seed != FORMAL_SOURCE_SEED:
        violations.append("source_seed")
    if evaluation_seed != FORMAL_EVALUATION_SEED:
        violations.append("evaluation_seed")
    if violations:
        return (
            "formal_result_protocol_mismatch:"
            f"{task}:{method}:{device}:{','.join(violations)}"
        )
    try:
        validate_formal_result_protocol(
            row,
            task_name=task,
            method=method,
            device=device,
            evaluation_seed=FORMAL_EVALUATION_SEED,
            max_steps=FORMAL_MAX_STEPS,
        )
    except ValueError as error:
        return str(error)
    return None


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
        source_seed = result_payload.get("source_seed")
        evaluation_seed = result_payload.get("evaluation_seed")
        protocol_error = _formal_result_protocol_error(
            task_names=task_names,
            task=task,
            method=method,
            device=device,
            source_seed=source_seed,
            evaluation_seed=evaluation_seed,
            row=result_row,
        )
        if protocol_error is not None:
            record["canonical_exclusion_errors"] = sorted(
                set(
                    [
                        *record.get("canonical_exclusion_errors", []),
                        f"{path}:{protocol_error}",
                    ]
                )
            )
            continue
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
            "source_seed": source_seed,
            "evaluation_seed": evaluation_seed,
            "registered_device_label": registered_device_label,
            "registration_id": str(result_payload.get("registration_id") or ""),
            "attempt_id": str(result_payload.get("attempt_id") or ""),
            "registered_at": str(manifest.get("registered_at") or ""),
            "official_validator_success": result_row.get("official_validator_success"),
            "registered_result_sha256": digest,
            "registered_result_object_path": record["object_path"],
            "registered_result_aliases": sorted(set(record["aliases"])),
            "registration_manifest_sha256": manifest_digest,
            "registration_manifest_object_path": str(manifest_object),
            "selection_reason": (
                "earliest_formal_protocol_compliant_validator_conclusion"
            ),
        }
        cell = f"{task}|{method}|{device}|{source_seed}|{evaluation_seed}"
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
    *,
    source_metadata: dict[str, Any],
    canonical_sources: dict[str, dict[str, Any]],
    screenshot_roots: Sequence[Path],
    run_log_records: dict[str, dict[str, Any]],
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
            or payload.get("schema_version") != "omniflow.function-asset-catalog.v1"
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
            source_run_log = _require_hashed_file(
                raw_item.get("source_run_log"),
                raw_item.get("source_run_log_sha256"),
                label=f"function_source_run_log:{task}",
            )
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
            source_run_log_hash = _sha256(source_run_log)
            raw_source_payload = _load_object(source_run_log)
            if not isinstance(raw_source_payload, dict):
                raise ValueError(f"function_source_run_log_invalid:{task}")
            task_source_metadata = source_metadata.get(task)
            if not isinstance(task_source_metadata, dict):
                raise ValueError(f"function_source_metadata_missing:{task}")
            function_source_metadata = dict(task_source_metadata)
            source_transfer_states_value = raw_item.get("source_transfer_states")
            if source_transfer_states_value:
                source_transfer_states = _require_hashed_file(
                    source_transfer_states_value,
                    raw_item.get("source_transfer_states_sha256"),
                    label=f"function_source_transfer_states:{task}",
                )
                source_transfer_states_payload = _load_object(source_transfer_states)
                expected_source_run_id = str(
                    raw_item.get("source_transfer_states_run_id") or ""
                ).strip()
                actual_source_run_id = str(
                    source_transfer_states_payload.get("run_id")
                    if isinstance(source_transfer_states_payload, dict)
                    else ""
                ).strip()
                function_source_run_id = str(
                    raw_source_payload.get("run_id") or ""
                ).strip()
                if (
                    not expected_source_run_id
                    or actual_source_run_id != expected_source_run_id
                    or function_source_run_id != expected_source_run_id
                ):
                    raise ValueError(
                        "function_source_transfer_states_run_id_mismatch:"
                        f"{task}:source={function_source_run_id}:"
                        f"catalog={actual_source_run_id}:"
                        f"expected={expected_source_run_id or 'missing'}"
                    )
                function_source_metadata["source_state_catalog"] = str(
                    source_transfer_states
                )
                function_source_metadata["source_state_catalog_sha256"] = _sha256(
                    source_transfer_states
                )
            canonical_source = canonical_sources.get(task)
            if not isinstance(canonical_source, dict):
                raise ValueError(f"canonical_function_source_missing:{task}")
            (
                canonical_source_run_log,
                canonical_source_run_log_hash,
                source_run_log_lineage,
            ) = _canonicalize_function_source_run_log(
                memory_root,
                task=task,
                source_run_log=source_run_log,
                source_payload=raw_source_payload,
                source_metadata=function_source_metadata,
                canonical_source=canonical_source,
                screenshot_roots=screenshot_roots,
                records=run_log_records,
            )
            identity = hashlib.sha256(
                "\0".join(
                    (
                        source_run_log_hash,
                        store_hash,
                        transfer_hash,
                        provenance_hash,
                    )
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
                    "source_run_log_path": str(canonical_source_run_log),
                    "source_run_log_sha256": canonical_source_run_log_hash,
                    "source_run_log_lineage": source_run_log_lineage,
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
    source_selection_manifest: str | Path | None = None,
    source_screenshot_roots: Sequence[str | Path] = (),
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
    screenshot_roots = tuple(
        sorted(
            {Path(value).expanduser().resolve() for value in source_screenshot_roots}
        )
    )
    for screenshot_root in screenshot_roots:
        if not screenshot_root.is_dir():
            raise FileNotFoundError(f"source_screenshot_root_missing:{screenshot_root}")
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
        indexed_task = indexed_paths.get(path)
        if indexed_task:
            source_run_log = require_complete_source_run_log(payload)
            if source_run_log["task_name"] != indexed_task:
                raise ValueError(
                    "indexed_source_run_log_task_mismatch:"
                    f"{indexed_task}:{source_run_log['task_name']}:{path}"
                )
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

    source_selections, source_selection_manifest_record = _load_source_selections(
        source_selection_manifest,
        memory_root=root,
        source_payload=source_payload,
        index_path=index_path,
        records=records,
        screenshot_roots=screenshot_roots,
    )
    if source_selection_manifest_record:
        selection_path = Path(source_selection_manifest_record["path"])
        source_selection_manifest_record["object_path"] = str(
            _materialize_object(
                root,
                selection_path,
                source_selection_manifest_record["sha256"],
            )
        )

    canonical_sources: dict[str, dict[str, Any]] = {}
    for path, task in sorted(indexed_paths.items(), key=lambda item: item[1]):
        baseline_digest = _sha256(path)
        selection = source_selections.get(task)
        digest = (
            str(selection["selected_source_run_log_sha256"])
            if selection is not None
            else baseline_digest
        )
        canonical_sources[task] = dict(records[digest])
        if selection is not None:
            canonical_selection = {
                "expected_source_run_log_sha256": selection[
                    "expected_source_run_log_sha256"
                ],
                "selected_source_run_log_sha256": digest,
                "reason": selection["reason"],
                "manifest_sha256": source_selection_manifest_record["sha256"],
                "manifest_object_path": source_selection_manifest_record["object_path"],
            }
            if "conversion" in selection:
                canonical_selection["conversion"] = selection["conversion"]
            canonical_sources[task]["selection"] = canonical_selection

    catalog_paths = sorted(
        {Path(value).expanduser().resolve() for value in function_catalogs}
    )
    resolved_result_roots = sorted(
        {Path(value).expanduser().resolve() for value in result_roots}
    )
    function_records, canonical_function_stores = _load_function_stores(
        root,
        catalog_paths,
        source_metadata=source_payload,
        canonical_sources=canonical_sources,
        screenshot_roots=screenshot_roots,
        run_log_records=records,
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
        item["retained_source_run_log_sha256"] = canonical_sources[str(task)]["sha256"]
        selection = source_selections.get(str(task))
        if selection is not None:
            selected_run_log = selection["selected_run_log"]
            item["goal"] = selected_run_log["goal"]
            item["params"] = selected_run_log["task_parameters"]
            item["source_seed"] = selected_run_log["seed"]
            item["step_count"] = len(selected_run_log["steps"])
            item["legacy_index_metadata"] = {
                "collect_seed": selected_run_log["seed"],
                "replay_seed": selected_run_log["seed"],
                "run_id": selected_run_log["run_id"],
                "task_random_seed": selected_run_log["seed"],
            }
            item["canonical_source_selection"] = canonical_sources[str(task)][
                "selection"
            ]
            for stale_key in (
                "source_state_catalog",
                "source_state_catalog_sha256",
                "transfer_state_catalog",
                "transfer_state_catalog_sha256",
                "store_provenance",
                "store_provenance_sha256",
            ):
                item.pop(stale_key, None)
        source_state_catalog = raw_item.get("source_state_catalog") or raw_item.get(
            "transfer_state_catalog"
        )
        if source_state_catalog and selection is None:
            catalog_path = _resolve_index_reference(
                index_path,
                source_state_catalog,
            )
            if not catalog_path.is_file():
                raise FileNotFoundError(
                    f"indexed_source_state_catalog_missing:{task}:{catalog_path}"
                )
            catalog_digest = _sha256(catalog_path)
            expected_catalog_digest = str(
                raw_item.get("source_state_catalog_sha256")
                or raw_item.get("transfer_state_catalog_sha256")
                or ""
            ).strip()
            if expected_catalog_digest and expected_catalog_digest != catalog_digest:
                raise ValueError(
                    "indexed_source_state_catalog_hash_mismatch:"
                    f"{task}:expected={expected_catalog_digest}:"
                    f"actual={catalog_digest}"
                )
            item.pop("transfer_state_catalog", None)
            item.pop("transfer_state_catalog_sha256", None)
            item["source_state_catalog"] = str(
                _materialize_object(root, catalog_path, catalog_digest)
            )
            item["source_state_catalog_sha256"] = catalog_digest
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
                "source_run_log_path",
                "source_run_log_sha256",
                "source_run_log_lineage",
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
                digest for digest, record in records.items() if task in record["tasks"]
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
        digest for digest, record in result_records.items() if not record["tasks"]
    )
    unclassified_runlog_hashes = sorted(
        digest for digest, record in records.items() if not record["tasks"]
    )

    registry: dict[str, Any] = {
        "schema_version": MEMORY_SCHEMA,
        "policy": {
            "deduplication": "exact_sha256",
            "source_run_log": (
                "source_index_with_explicit_sha256_selections"
                if source_selections
                else "source_index_authoritative"
            ),
            "result": ("earliest_formal_protocol_compliant_validator_conclusion"),
            "success_cherry_picking": False,
        },
        "inputs": {
            "source_index": str(index_path),
            "source_index_sha256": _sha256(index_path),
            "source_selection_manifest": source_selection_manifest_record,
            "source_screenshot_roots": [str(path) for path in screenshot_roots],
            "function_catalogs": [str(path) for path in catalog_paths],
            "runlog_roots": sorted(
                str(Path(value).expanduser().resolve()) for value in runlog_roots
            ),
            "result_roots": [str(path) for path in resolved_result_roots],
        },
        "counts": {
            "task_count": len(task_names),
            "source_selection_tasks": len(source_selections),
            "run_log_paths": len(paths),
            "unique_run_logs": len(records),
            "function_catalog_paths": len(catalog_paths),
            "unique_function_stores": len(function_records),
            "function_store_tasks": len(canonical_function_stores),
            "result_paths": len(result_paths),
            "unique_results": len(result_records),
            "formal_protocol_excluded_results": sum(
                "canonical_exclusion_errors" in record
                for record in result_records.values()
            ),
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
                digest: function_records[digest] for digest in sorted(function_records)
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
    source_selection_manifest: str | Path | None = None,
    source_screenshot_roots: Sequence[str | Path] = (),
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
            source_selection_manifest=source_selection_manifest,
            source_screenshot_roots=source_screenshot_roots,
        )


def load_artifact_memory(memory_index: str | Path) -> dict[str, Any]:
    """Load and verify the registry selected by ``current.json``."""

    pointer_path = Path(memory_index).expanduser().resolve()
    pointer = _load_object(pointer_path)
    if not isinstance(pointer, dict) or pointer.get("schema_version") != CURRENT_SCHEMA:
        raise ValueError(f"artifact_memory_pointer_invalid:{pointer_path}")
    registry_path = Path(str(pointer.get("registry_path") or "")).expanduser()
    if not registry_path.is_absolute() or not registry_path.is_file():
        raise FileNotFoundError(f"artifact_memory_registry_missing:{registry_path}")
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
            raise ValueError(f"artifact_memory_index_hash_mismatch:{path_field}:{path}")
    return registry


def registered_cell_plan_from_memory(
    *,
    memory_index: str | Path,
    task_name: str,
    methods: Sequence[str],
    devices: Sequence[str],
    source_seed: int,
    evaluation_seed: int,
    formal_max_steps: int | None = None,
) -> dict[str, list[tuple[str, str]]]:
    """Resolve completed formal cells without rescanning historical results."""

    registry = load_artifact_memory(memory_index)
    cells = registry["canonical"]["result_cells"]
    expected = [(method, device) for method in methods for device in devices]
    completed: list[tuple[str, str]] = []
    for method, device in expected:
        cell_key = f"{task_name}|{method}|{device}|{source_seed}|{evaluation_seed}"
        record = cells.get(cell_key)
        if not isinstance(record, dict):
            continue
        if formal_max_steps is not None:
            object_path = Path(
                str(record.get("registered_result_object_path") or "")
            ).expanduser()
            expected_hash = str(record.get("registered_result_sha256") or "")
            if (
                not object_path.is_absolute()
                or not object_path.is_file()
                or not expected_hash
                or _sha256(object_path) != expected_hash
            ):
                raise ValueError(
                    f"artifact_memory_result_object_invalid:{cell_key}:{object_path}"
                )
            payload = _load_object(object_path)
            rows = payload.get("rows") if isinstance(payload, dict) else None
            if (
                not isinstance(rows, list)
                or len(rows) != 1
                or not isinstance(rows[0], dict)
            ):
                raise ValueError(
                    f"artifact_memory_result_payload_invalid:{cell_key}:{object_path}"
                )
            from src.experiment.result_registry import (
                validate_formal_result_protocol,
            )

            validate_formal_result_protocol(
                rows[0],
                task_name=task_name,
                method=method,
                device=device,
                evaluation_seed=evaluation_seed,
                max_steps=formal_max_steps,
            )
        completed.append((method, device))
    return {
        "completed": completed,
        "pending": [cell for cell in expected if cell not in completed],
    }


def refresh_artifact_memory_from_pointer(
    *,
    memory_index: str | Path,
    source_index: str | Path | None = None,
    source_selection_manifest: str | Path | None = None,
    source_screenshot_roots: Sequence[str | Path] = (),
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
            source_index=source_index or str(inputs["source_index"]),
            function_catalogs=function_catalogs,
            runlog_roots=runlog_roots,
            result_roots=result_roots,
            source_selection_manifest=(
                source_selection_manifest
                or (inputs.get("source_selection_manifest") or {}).get("object_path")
                or (inputs.get("source_selection_manifest") or {}).get("path")
            ),
            source_screenshot_roots=(
                source_screenshot_roots
                or tuple(inputs.get("source_screenshot_roots") or ())
            ),
        )


def _split_values(values: Sequence[str]) -> list[str]:
    return [item for value in values for item in value.split(":") if item.strip()]


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
    refresh_parser.add_argument("--source-selection-manifest")
    refresh_parser.add_argument("--source-screenshot-root", action="append", default=[])
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
        memory_root = Path(args.memory_root).expanduser().resolve()
        pointer_path = memory_root / "current.json"
        if pointer_path.is_file():
            report = refresh_artifact_memory_from_pointer(
                memory_index=pointer_path,
                source_index=args.source_index,
                source_selection_manifest=args.source_selection_manifest,
                source_screenshot_roots=args.source_screenshot_root,
                additional_function_catalogs=_split_values(args.function_catalog),
                additional_runlog_roots=_split_values(args.runlog_root),
                additional_result_roots=_split_values(args.result_root),
            )
        else:
            report = refresh_artifact_memory(
                memory_root=memory_root,
                source_index=args.source_index,
                source_selection_manifest=args.source_selection_manifest,
                source_screenshot_roots=args.source_screenshot_root,
                function_catalogs=_split_values(args.function_catalog),
                runlog_roots=_split_values(args.runlog_root),
                result_roots=_split_values(args.result_root),
            )
        pointer = _load_object(pointer_path)
        output = {
            "current": str(pointer_path),
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
