#!/usr/bin/env python3
"""Content-addressed AndroidWorld RunLog, method-memory, and result evidence."""

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
from src.experiment.mobilegpt_contract import (
    MOBILEGPT_LEARNING_MODE,
    MOBILEGPT_MEMORY_MANIFEST,
    MOBILEGPT_MEMORY_SCHEMA,
    MOBILEGPT_PREP_TYPE,
    MOBILEGPT_SOURCE_METHOD,
    MOBILEGPT_SOURCE_METHOD_BY_SCHEMA,
    MOBILEGPT_SUPPORTED_MEMORY_SCHEMAS,
)
from src.experiment.protocol import (
    DEVICES,
    MAX_STEPS,
    METHODS,
    RESULT_COMMANDS_FILE,
    RESULT_SUMMARY_FILE,
    SOURCE_SEED,
    TASK_SEED,
)
from src.integrations.runlog import adapt_source_run_log

MEMORY_SCHEMA = "omniflow.androidworld-artifact-memory.v2"
CURRENT_SCHEMA = "omniflow.androidworld-artifact-memory-pointer.v2"
FUNCTION_SOURCE_LINEAGE_SCHEMA = "omniflow.function-store-source-lineage.v1"
RESULT_FILE_NAMES = (
    RESULT_COMMANDS_FILE,
    RESULT_SUMMARY_FILE,
    "one_task_commands.jsonl",  # immutable historical results only
    "one_task_summary.json",  # immutable historical results only
    "registered_result.json",
    "registration_manifest.json",
    "stats.jsonl",
    "summary.json",
    "task_results.jsonl",
)
BASELINE_BATCH_REPORT_SCHEMA = "omniflow.androidworld.batch_report.v1"
BASELINE_BATCH_REPORT_SELECTION = (
    "authoritative_immutable_batch_report_validator_conclusion"
)
_ARCHIVED_MOBILEGPT_RESULT_CONTRACTS = {
    "omniflow.mobilegpt-runlog-semantic-memory.v1": {
        "mode": "semantic",
        "source_method": "mobilegpt_runlog_semantic_memory",
        "prep_type": "mobilegpt_runlog_semantic_memory",
        "learning_mode": "mobilegpt_runlog_semantic_conversion",
    },
    MOBILEGPT_MEMORY_SCHEMA: {
        "mode": "direct",
        "source_method": MOBILEGPT_SOURCE_METHOD,
        "prep_type": MOBILEGPT_PREP_TYPE,
        "learning_mode": MOBILEGPT_LEARNING_MODE,
    },
    "omniflow.mobilegpt-runlog-native-derive-memory.v2": {
        "mode": "native_derive",
        "source_method": "mobilegpt_runlog_native_derive",
        "prep_type": "mobilegpt_runlog_native_derive_memory",
        "learning_mode": "mobilegpt_runlog_virtual_source",
    },
    "omniflow.mobilegpt-runlog-teacher-memory.v1": {
        "mode": "legacy",
        "source_method": "mobilegpt_runlog_teacher",
        "prep_type": "mobilegpt_runlog_teacher_memory",
        "learning_mode": "mobilegpt_runlog_teacher",
    },
}


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


def _materialize_binary_object(
    memory_root: Path,
    source: Path,
    digest: str,
    *,
    suffix: str,
) -> Path:
    target = memory_root / "objects" / "sha256" / digest[:2] / f"{digest}{suffix}"
    if target.exists():
        if not target.is_file() or _sha256(target) != digest:
            raise ValueError(f"memory_object_hash_mismatch:{target}")
        return target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(source, temporary)
        if _sha256(temporary) != digest:
            raise ValueError(f"memory_object_copy_hash_mismatch:{source}")
        temporary.chmod(0o444)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target.resolve()


def _materialize_run_log_dependencies(
    memory_root: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if payload.get("schema_version") != "omniflow.run_log.v1":
        return {"screenshots": []}
    references: list[dict[str, Any]] = []
    for step in payload.get("steps") or ():
        if not isinstance(step, dict):
            continue
        for phase in ("observation", "next_observation"):
            observation = step.get(phase)
            if isinstance(observation, dict):
                references.append(observation)
    final_observation = payload.get("final_observation")
    if isinstance(final_observation, dict):
        references.append(final_observation)

    screenshots: dict[str, dict[str, str]] = {}
    suffixes = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }
    for observation in references:
        pixels = observation.get("pixels")
        if not isinstance(pixels, dict):
            continue
        digest = str(pixels.get("sha256") or "").strip().lower()
        mime_type = str(pixels.get("mime_type") or "").strip()
        suffix = suffixes.get(mime_type)
        if suffix is None:
            raise ValueError(f"run_log_screenshot_mime_type_invalid:{mime_type}")
        source = Path(str(pixels.get("path") or "")).expanduser().resolve()
        stored = (
            memory_root
            / "objects"
            / "sha256"
            / digest[:2]
            / f"{digest}{suffix}"
        )
        if not source.is_file() and stored.is_file():
            source = stored
        if not source.is_file():
            raise FileNotFoundError(f"run_log_screenshot_missing:{source}")
        actual = _sha256(source)
        if actual != digest:
            raise ValueError(
                "run_log_screenshot_hash_mismatch:"
                f"expected={digest or 'missing'}:actual={actual}:path={source}"
            )
        object_path = _materialize_binary_object(
            memory_root,
            source,
            digest,
            suffix=suffix,
        )
        screenshots[digest] = {
            "sha256": digest,
            "mime_type": mime_type,
            "object_path": str(object_path),
        }
    return {"screenshots": [screenshots[key] for key in sorted(screenshots)]}


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
) -> dict[str, str]:
    identity = hashlib.sha256(
        "\0".join((store_sha256, transfer_sha256)).encode("utf-8")
    ).hexdigest()
    runtime_root = memory_root / "runtime" / "function_stores" / identity
    store_path = runtime_root / "store.json"
    transfer_path = runtime_root / "transfer_states.json"
    _link_object(store_object, store_path, store_sha256)
    _link_object(transfer_object, transfer_path, transfer_sha256)
    runtime_root.chmod(0o555)
    return {
        "store_path": str(store_path.resolve()),
        "store_sha256": store_sha256,
        "transfer_states_path": str(transfer_path.resolve()),
        "transfer_states_sha256": transfer_sha256,
    }


def _runlog_paths(roots: Iterable[Path]) -> list[Path]:
    paths: set[Path] = set()
    for root in roots:
        resolved = root.expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"runlog_root_missing:{resolved}")
        paths.update(
            path.resolve()
            for path in resolved.rglob("*.run_log.json")
            if not path.name.startswith("._")
        )
        paths.update(
            path.resolve()
            for path in resolved.rglob("run_log.json")
            if not path.name.startswith("._")
        )
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
    memory_root: Path,
    path: Path,
    digest: str,
    payload: dict[str, Any],
    task: str,
    alias: str = "",
    materialize_dependencies: bool = False,
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
            "dependencies": (
                _materialize_run_log_dependencies(memory_root, payload)
                if materialize_dependencies
                else {"screenshots": []}
            ),
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
        memory_root=memory_root,
        path=source_object,
        digest=source_sha256,
        payload=source_payload,
        task=task,
        alias=str(source_run_log),
        materialize_dependencies=True,
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
        memory_root=memory_root,
        path=canonical_object,
        digest=canonical_sha256,
        payload=canonical,
        task=task,
        materialize_dependencies=True,
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
    public_row = rows[0]
    detail_rows = [
        row for row in payload.get("details") or [] if isinstance(row, dict)
    ]
    row = next(
        (
            detail
            for detail in detail_rows
            if str(detail.get("method") or "")
            == str(public_row.get("method") or "")
            and str(detail.get("device") or "")
            == str(public_row.get("device") or "")
        ),
        public_row,
    )
    method = str(public_row.get("method") or "")
    device = str(public_row.get("device") or "")
    if method != str(manifest.get("method") or ""):
        raise ValueError("registered_result_method_mismatch")
    if device != str(manifest.get("device") or ""):
        raise ValueError("registered_result_device_mismatch")
    official_used = row.get("official_validator_used") is True or (
        "validator_success" in public_row
    )
    official_success = row.get("official_validator_success")
    if official_success is None and "validator_success" in public_row:
        official_success = public_row.get("validator_success")
    conclusion = official_used and isinstance(official_success, bool)
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


def _mobilegpt_result_protocol_error(
    *,
    task: str,
    source_seed: Any,
    row: dict[str, Any],
) -> str | None:
    prefix = f"formal_result_mobilegpt_memory_invalid:{task}"
    archived_prep_types = {
        str(contract["prep_type"])
        for contract in _ARCHIVED_MOBILEGPT_RESULT_CONTRACTS.values()
    }
    if str(row.get("prep_type") or "") not in archived_prep_types:
        return f"{prefix}:prep_type"
    manifest_path = Path(str(row.get("prep_manifest") or "")).expanduser()
    if not manifest_path.is_absolute() or not manifest_path.is_file():
        return f"{prefix}:manifest_missing:{manifest_path}"
    recorded_manifest_sha256 = str(row.get("prep_manifest_sha256") or "").strip()
    actual_manifest_sha256 = _sha256(manifest_path)
    if recorded_manifest_sha256 != actual_manifest_sha256:
        return (
            f"{prefix}:manifest_hash_mismatch:"
            f"recorded={recorded_manifest_sha256 or 'missing'}:"
            f"actual={actual_manifest_sha256}"
        )
    try:
        manifest = _load_object(manifest_path)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return f"{prefix}:manifest_unreadable:{type(error).__name__}:{error}"
    if not isinstance(manifest, dict):
        return f"{prefix}:manifest_type"
    schema_version = manifest.get("schema_version")
    contract = _ARCHIVED_MOBILEGPT_RESULT_CONTRACTS.get(str(schema_version or ""))
    if contract is None:
        return f"{prefix}:schema"
    if str(manifest.get("task_name") or "") != task:
        return f"{prefix}:task"
    if manifest.get("source_seed") != source_seed:
        return f"{prefix}:source_seed"
    mode = str(contract["mode"])
    legacy = mode == "legacy"
    semantic = mode == "semantic"
    native_derive = mode == "native_derive"
    expected_source_method = str(contract["source_method"])
    expected_prep_type = str(contract["prep_type"])
    if str(row.get("prep_type") or "") != expected_prep_type:
        return f"{prefix}:prep_type_schema_mismatch"
    if str(manifest.get("source_method") or "") != expected_source_method:
        return f"{prefix}:source_method"
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        return f"{prefix}:provenance"
    required_provenance = {
        "native_mobilegpt_learning": legacy or native_derive,
        "task_local_memory": True,
        "learning_mode": str(contract["learning_mode"]),
        "teacher_forcing": legacy,
        "synthetic_subtasks": not (legacy or semantic or native_derive),
        "actions_supplied_to_mobilegpt": not native_derive,
        "function_store_used": False,
        "function_conversion_enabled": False,
        "target_inputs_read": False,
        "target_observations_read": False,
        "validator_state_read": False,
        "coordinate_replay": False,
    }
    if legacy:
        required_provenance["complete_teacher_action_consumption"] = True
    elif semantic:
        required_provenance.update(
            {
                "semantic_subtasks": True,
                "original_mobilegpt_prompts": True,
                "source_transitions_supplied": True,
                "source_success_boundary_supplied": True,
                "runlog_transition_compilation": True,
                "complete_transition_mapping": True,
                "official_reader_validation": True,
                "source_emulator_used": False,
            }
        )
    elif native_derive:
        required_provenance.update(
            {
                "source_transitions_supplied": True,
                "source_success_boundary_supplied": True,
                "trajectory_action_validation": True,
                "complete_trajectory_validation": True,
                "source_emulator_used": False,
            }
        )
    else:
        required_provenance.update(
            {
                "source_transitions_supplied": True,
                "source_success_boundary_supplied": True,
                "runlog_transition_compilation": True,
                "complete_transition_mapping": True,
                "official_reader_validation": True,
                "source_emulator_used": False,
            }
        )
    for field, expected in required_provenance.items():
        if provenance.get(field) != expected:
            return f"{prefix}:provenance_{field}"
    if legacy:
        official_source_result = manifest.get("official_source_result")
        if (
            not isinstance(official_source_result, dict)
            or official_source_result.get("official_validator_used") is not True
            or official_source_result.get("official_validator_success") is not True
        ):
            return f"{prefix}:official_source_result"
    elif "official_source_result" in manifest:
        return f"{prefix}:official_source_result_forbidden"
    memory = manifest.get("memory")
    if not isinstance(memory, dict):
        return f"{prefix}:memory"
    manifest_memory_sha256 = str(memory.get("sha256") or "").strip()
    recorded_memory_sha256 = str(row.get("prep_memory_sha256") or "").strip()
    if (
        not manifest_memory_sha256
        or recorded_memory_sha256 != manifest_memory_sha256
    ):
        return f"{prefix}:memory_hash"
    return None


def _formal_result_protocol_error(
    *,
    task_names: Sequence[str],
    task: str,
    method: str,
    device: str,
    source_seed: Any,
    evaluation_seed: Any,
    row: dict[str, Any],
    canonical_source_sha256: str = "",
) -> str | None:
    from src.experiment.result_registry import (
        formal_result_environment_failure_reasons,
        has_official_validator_conclusion,
        validate_formal_result_protocol,
    )
    formal_device_targets = {
        label: (serial, port) for label, serial, port in DEVICES
    }

    violations: list[str] = []
    if task not in task_names:
        violations.append("task_not_indexed")
    if method not in METHODS:
        violations.append("unsupported_method")
    if device not in formal_device_targets:
        violations.append("unsupported_device")
    if source_seed != SOURCE_SEED:
        violations.append("source_seed")
    if evaluation_seed != TASK_SEED:
        violations.append("evaluation_seed")
    if violations:
        return (
            "formal_result_protocol_mismatch:"
            f"{task}:{method}:{device}:{','.join(violations)}"
        )
    environment_reasons = formal_result_environment_failure_reasons(row)
    if environment_reasons:
        return (
            "formal_result_environment_failure:"
            f"{task}:{method}:{device}:{','.join(environment_reasons)}"
        )
    if not has_official_validator_conclusion(row):
        return (
            "official_validator_conclusion_missing:"
            f"{task}:{method}:{device}"
        )
    try:
        validate_formal_result_protocol(
            row,
            task_name=task,
            method=method,
            device=device,
            evaluation_seed=TASK_SEED,
            max_steps=MAX_STEPS,
        )
    except ValueError as error:
        return str(error)
    if method == "fixed_replay":
        expected_source_sha256 = str(canonical_source_sha256 or "").strip()
        for field in ("source_run_log", "replay_run_log"):
            path = Path(str(row.get(field) or "")).expanduser()
            if not path.is_absolute() or not path.is_file():
                return (
                    "formal_result_fixed_replay_source_missing:"
                    f"{task}:{device}:{field}:{path}"
                )
            actual_sha256 = _sha256(path)
            recorded_sha256 = str(row.get(f"{field}_sha256") or "").strip()
            if recorded_sha256 and recorded_sha256 != actual_sha256:
                return (
                    "formal_result_fixed_replay_recorded_hash_mismatch:"
                    f"{task}:{device}:{field}:"
                    f"recorded={recorded_sha256}:actual={actual_sha256}"
                )
            if actual_sha256 != expected_source_sha256:
                return (
                    "formal_result_fixed_replay_source_hash_mismatch:"
                    f"{task}:{device}:{field}:"
                    f"expected={expected_source_sha256}:actual={actual_sha256}"
                )
    if method == "mobilegpt_offline_retrieval":
        return _mobilegpt_result_protocol_error(
            task=task,
            source_seed=source_seed,
            row=row,
        )
    return None


def _load_results(
    memory_root: Path,
    roots: Sequence[Path],
    task_names: Sequence[str],
    canonical_sources: dict[str, dict[str, Any]],
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
            canonical_source_sha256=str(
                (canonical_sources.get(task) or {}).get("sha256") or ""
            ),
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
        if not verified["validator_conclusion"]:
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
        if method == "mobilegpt_offline_retrieval":
            prep_manifest_path = Path(
                str(result_row.get("prep_manifest") or "")
            ).expanduser()
            prep_manifest = _load_object(prep_manifest_path)
            candidate["mobilegpt_memory_schema"] = str(
                prep_manifest.get("schema_version") or ""
            )
            candidate["mobilegpt_source_method"] = str(
                prep_manifest.get("source_method") or ""
            )
            candidate["mobilegpt_prep_type"] = str(
                result_row.get("prep_type") or ""
            )
        if method == "fixed_replay":
            candidate["source_run_log_sha256"] = _sha256(
                Path(str(result_row["source_run_log"])).expanduser()
            )
        result = f"{task}|{method}|{device}|{source_seed}|{evaluation_seed}"
        candidates.setdefault(result, []).append(
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
    canonical: dict[str, dict[str, Any]] = {}
    for result, cell_candidates in sorted(candidates.items()):
        ordered = sorted(cell_candidates, key=lambda value: value[:3])
        current_mobilegpt = [
            value
            for value in ordered
            if value[3]["method"] == "mobilegpt_offline_retrieval"
            and value[3].get("mobilegpt_memory_schema")
            in MOBILEGPT_SUPPORTED_MEMORY_SCHEMAS
        ]
        canonical[result] = (current_mobilegpt or ordered)[0][3]
    return paths, records, canonical


def _load_function_stores(
    memory_root: Path,
    catalogs: Sequence[Path],
    *,
    source_metadata: dict[str, Any],
    canonical_sources: dict[str, dict[str, Any]],
    screenshot_roots: Sequence[Path],
    run_log_records: dict[str, dict[str, Any]],
    existing_canonical_identities: dict[str, str] | None = None,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
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
            provenance_payload: dict[str, Any] = {}
            if raw_item.get("provenance_path"):
                provenance = _require_hashed_file(
                    raw_item.get("provenance_path"),
                    raw_item.get("provenance_sha256"),
                    label=f"function_provenance:{task}",
                )
                loaded_provenance = _load_object(provenance)
                if isinstance(loaded_provenance, dict):
                    provenance_payload = loaded_provenance
            store_payload = _load_object(store)
            if (
                not isinstance(store_payload, dict)
                or store_payload.get("schema_version") != "omniflow.store.v2"
                or not isinstance(store_payload.get("functions"), dict)
            ):
                raise ValueError(f"function_store_invalid:{task}:{store}")
            store_hash = _sha256(store)
            transfer_hash = _sha256(transfer)
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
                    )
                ).encode("utf-8")
            ).hexdigest()
            store_object = _materialize_object(memory_root, store, store_hash)
            transfer_object = _materialize_object(
                memory_root,
                transfer,
                transfer_hash,
            )
            source_calls = store_payload.get("source_calls")
            if not isinstance(source_calls, list) or not source_calls:
                source_calls = provenance_payload.get("source_calls")
            if not isinstance(source_calls, list) or not source_calls:
                source_calls = [
                    {"function_id": function_id, "arguments": {}}
                    for function_id in store_payload["functions"]
                ]
            record = records.setdefault(
                identity,
                {
                    "identity_sha256": identity,
                    "tasks": [],
                    "catalog_aliases": [],
                    "function_count": len(store_payload["functions"]),
                    "source_calls": source_calls,
                    "source_run_log_path": str(canonical_source_run_log),
                    "source_run_log_sha256": canonical_source_run_log_hash,
                    "source_run_log_lineage": source_run_log_lineage,
                    **_materialize_function_store(
                        memory_root,
                        store_object=store_object,
                        store_sha256=store_hash,
                        transfer_object=transfer_object,
                        transfer_sha256=transfer_hash,
                    ),
                },
            )
            record["tasks"].append(task)
            record["catalog_aliases"].append(str(catalog_path))
            quality = (1 if store_payload.get("source_calls") else 0,)
            candidates.setdefault(task, []).append((quality, identity))

    for record in records.values():
        record["tasks"] = sorted(set(record["tasks"]))
        record["catalog_aliases"] = sorted(set(record["catalog_aliases"]))

    canonical: dict[str, dict[str, Any]] = {}
    existing_identities = existing_canonical_identities or {}
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
            existing_identity = str(existing_identities.get(task) or "")
            if existing_identity in best_ids:
                canonical[task] = dict(records[existing_identity])
                continue
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


def _load_mobilegpt_memories(
    memory_root: Path,
    roots: Sequence[Path],
    *,
    canonical_sources: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    manifest_paths = sorted(
        {
            path.resolve()
            for root in roots
            if root.is_dir()
            for path in root.rglob(MOBILEGPT_MEMORY_MANIFEST)
            if path.is_file()
        }
    )
    records: dict[str, dict[str, Any]] = {}
    candidates: dict[str, set[str]] = {}
    for manifest_path in manifest_paths:
        manifest = _load_object(manifest_path)
        if not isinstance(manifest, dict):
            continue
        schema_version = str(manifest.get("schema_version") or "")
        if schema_version not in MOBILEGPT_SUPPORTED_MEMORY_SCHEMAS:
            continue
        source_method = MOBILEGPT_SOURCE_METHOD_BY_SCHEMA[schema_version]
        task = str(manifest.get("task_name") or "").strip()
        source = canonical_sources.get(task)
        if source is None:
            raise ValueError(f"mobilegpt_memory_task_not_canonical:{task}")
        memory_record = manifest.get("memory")
        if not isinstance(memory_record, dict):
            raise ValueError(f"mobilegpt_memory_record_missing:{manifest_path}")
        memory_path = (
            manifest_path.parent
            / str(memory_record.get("relative_path") or "")
        ).resolve()
        from src.experiment.androidworld import validate_mobilegpt_adapted_memory

        validated = validate_mobilegpt_adapted_memory(
            memory_path,
            task_name=task,
            source_seed=SOURCE_SEED,
            source_run_log=str(source["object_path"]),
            expected_model=str(manifest.get("source_model") or ""),
            expected_source_method=source_method,
        )
        memory_sha256 = str(validated["memory_sha256"])
        manifest_sha256 = _sha256(manifest_path)
        record = records.setdefault(
            memory_sha256,
            {
                "task": task,
                "schema_version": schema_version,
                "source_seed": SOURCE_SEED,
                "source_method": source_method,
                "source_model": str(manifest.get("source_model") or ""),
                "source_run_log_sha256": str(source["sha256"]),
                "memory_sha256": memory_sha256,
                "memory_file_count": int(validated["memory_file_count"]),
                "memory_root_aliases": [],
                "manifest_aliases": [],
                "manifest_sha256s": [],
                "manifest_object_paths": [],
                "inventory": dict(validated["memory_inventory"]),
            },
        )
        identity = (
            record["task"],
            record["source_run_log_sha256"],
            record["source_model"],
        )
        candidate_identity = (
            task,
            str(source["sha256"]),
            str(manifest.get("source_model") or ""),
        )
        if identity != candidate_identity:
            raise ValueError(
                f"mobilegpt_memory_hash_identity_conflict:{memory_sha256}"
            )
        record["memory_root_aliases"].append(str(memory_path))
        record["manifest_aliases"].append(str(manifest_path))
        record["manifest_sha256s"].append(manifest_sha256)
        record["manifest_object_paths"].append(
            str(_materialize_object(memory_root, manifest_path, manifest_sha256))
        )
        candidates.setdefault(task, set()).add(memory_sha256)

    for record in records.values():
        for field in (
            "memory_root_aliases",
            "manifest_aliases",
            "manifest_sha256s",
            "manifest_object_paths",
        ):
            record[field] = sorted(set(record[field]))
        record["memory_root"] = record["memory_root_aliases"][0]
        record["manifest_path"] = record["manifest_aliases"][0]

    canonical: dict[str, dict[str, Any]] = {}
    for task, memory_sha256s in sorted(candidates.items()):
        canonical[task] = _select_canonical_mobilegpt_memory(
            task=task,
            memory_sha256s=memory_sha256s,
            records=records,
        )
    return records, canonical


def _select_canonical_mobilegpt_memory(
    *,
    task: str,
    memory_sha256s: set[str],
    records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    canonical_memory_sha256s = {
        memory_sha256
        for memory_sha256 in memory_sha256s
        if records[memory_sha256]["source_method"]
        == MOBILEGPT_SOURCE_METHOD
    }
    unsupported_memory_sha256s = memory_sha256s - canonical_memory_sha256s
    if unsupported_memory_sha256s:
        raise ValueError(
            f"unsupported_mobilegpt_memory:{task}:"
            + ",".join(sorted(unsupported_memory_sha256s))
        )
    if len(canonical_memory_sha256s) != 1:
        raise ValueError(
            f"ambiguous_mobilegpt_memory:{task}:"
            + ",".join(sorted(canonical_memory_sha256s))
        )
    memory_sha256 = next(iter(canonical_memory_sha256s))
    return {
        **records[memory_sha256],
        "selection_reason": "only_supported_mobilegpt_memory",
    }


def _load_baseline_batch_reports(
    memory_root: Path,
    report_paths: Sequence[Path],
    *,
    task_names: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load explicit immutable batch snapshots as read-only completed results."""

    known_tasks = set(task_names)
    formal_device_targets = {
        label: (serial, port) for label, serial, port in DEVICES
    }
    records: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {}
    for summary_path in report_paths:
        if not summary_path.is_file():
            raise FileNotFoundError(
                f"baseline_batch_report_missing:{summary_path}"
            )
        summary = _load_object(summary_path)
        if (
            not isinstance(summary, dict)
            or summary.get("schema_version") != BASELINE_BATCH_REPORT_SCHEMA
            or summary.get("immutable") is not True
        ):
            raise ValueError(f"baseline_batch_report_invalid:{summary_path}")
        results_path = Path(str(summary.get("results_jsonl") or "")).expanduser()
        if not results_path.is_absolute():
            results_path = (summary_path.parent / results_path).resolve()
        else:
            results_path = results_path.resolve()
        if not results_path.is_file():
            raise FileNotFoundError(
                f"baseline_batch_results_missing:{summary_path}:{results_path}"
            )
        rows = [
            json.loads(line)
            for line in results_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not all(isinstance(row, dict) for row in rows):
            raise ValueError(f"baseline_batch_results_invalid:{results_path}")
        expected_source_seed = summary.get("source_seed")
        expected_evaluation_seed = summary.get("evaluation_seed")
        actual_counts = {
            "planned": len(rows),
            "validator_success": 0,
            "validator_failure": 0,
            "non_validator_failure": 0,
            "pending": 0,
        }
        report_results: set[str] = set()
        validator_result_count = 0
        for row in rows:
            task = str(row.get("task_name") or "")
            method = str(row.get("method") or "")
            device = _formal_device_label(row.get("device"))
            source_seed = row.get("source_seed")
            evaluation_seed = row.get("evaluation_seed")
            if task not in known_tasks:
                raise ValueError(
                    f"baseline_batch_task_not_indexed:{summary_path}:{task}"
                )
            if method not in METHODS:
                raise ValueError(
                    f"baseline_batch_method_invalid:{summary_path}:{method}"
                )
            if device not in formal_device_targets:
                raise ValueError(
                    f"baseline_batch_device_invalid:{summary_path}:{device}"
                )
            if (
                source_seed != expected_source_seed
                or evaluation_seed != expected_evaluation_seed
            ):
                raise ValueError(
                    f"baseline_batch_seed_mismatch:{summary_path}:{task}:{device}"
                )
            result = f"{task}|{method}|{device}|{source_seed}|{evaluation_seed}"
            if result in report_results:
                raise ValueError(
                    f"baseline_batch_duplicate_result:{summary_path}:{result}"
                )
            report_results.add(result)
            conclusion = str(row.get("conclusion") or "")
            if conclusion not in actual_counts or conclusion == "planned":
                raise ValueError(
                    f"baseline_batch_conclusion_invalid:{summary_path}:{result}:"
                    f"{conclusion}"
                )
            actual_counts[conclusion] += 1
            if row.get("official_validator_used") is not True:
                continue
            success = row.get("official_validator_success")
            if not isinstance(success, bool):
                raise ValueError(
                    f"baseline_batch_validator_result_invalid:{summary_path}:{result}"
                )
            expected_conclusion = (
                "validator_success" if success else "validator_failure"
            )
            if conclusion != expected_conclusion:
                raise ValueError(
                    f"baseline_batch_validator_conclusion_mismatch:"
                    f"{summary_path}:{result}"
                )
            validator_result_count += 1
            row_payload = _json_bytes({"rows": [row]})
            row_digest = hashlib.sha256(row_payload).hexdigest()
            row_object = _materialize_content(
                memory_root,
                row_payload,
                row_digest,
            )
            candidate = {
                "task": task,
                "method": method,
                "device": device,
                "source_seed": source_seed,
                "evaluation_seed": evaluation_seed,
                "attempt_id": str(row.get("attempt_id") or ""),
                "official_validator_success": success,
                "registered_result_sha256": row_digest,
                "registered_result_object_path": str(row_object),
                "registered_result_aliases": [str(results_path)],
                "selection_reason": BASELINE_BATCH_REPORT_SELECTION,
                "baseline_batch_report": str(summary_path),
                "baseline_batch_results": str(results_path),
            }
            existing = results.get(result)
            if existing is not None and existing != candidate:
                raise ValueError(f"conflicting_baseline_batch_result:{result}")
            results[result] = candidate
        summary_counts = summary.get("counts")
        if not isinstance(summary_counts, dict) or any(
            int(summary_counts.get(key, -1)) != value
            for key, value in actual_counts.items()
        ):
            raise ValueError(
                f"baseline_batch_counts_mismatch:{summary_path}:"
                f"expected={summary_counts}:actual={actual_counts}"
            )
        summary_digest = _sha256(summary_path)
        results_digest = _sha256(results_path)
        records[summary_digest] = {
            "attempt_id": str(summary.get("attempt_id") or ""),
            "summary_sha256": summary_digest,
            "summary_alias": str(summary_path),
            "summary_object_path": str(
                _materialize_object(memory_root, summary_path, summary_digest)
            ),
            "results_sha256": results_digest,
            "results_alias": str(results_path),
            "results_object_path": str(
                _materialize_object(memory_root, results_path, results_digest)
            ),
            "planned_results": len(rows),
            "validator_results": validator_result_count,
        }
    return records, results


def _refresh_artifact_memory_unlocked(
    *,
    memory_root: str | Path,
    source_index: str | Path,
    function_catalogs: Sequence[str | Path],
    runlog_roots: Sequence[str | Path],
    result_roots: Sequence[str | Path],
    mobilegpt_memory_roots: Sequence[str | Path] = (),
    baseline_batch_reports: Sequence[str | Path] = (),
    source_screenshot_roots: Sequence[str | Path] = (),
    existing_function_store_identities: dict[str, str] | None = None,
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
    indexed_canonical_digests: dict[str, str] = {}
    for path in paths:
        digest = _sha256(path)
        payload = _load_object(path)
        if not isinstance(payload, dict):
            raise ValueError(f"run_log_must_be_object:{path}")
        indexed_task = indexed_paths.get(path)
        migrated_source: dict[str, Any] | None = None
        migrated_digest = ""
        migrated_object: Path | None = None
        if indexed_task:
            try:
                source_run_log = require_complete_source_run_log(payload)
            except (TypeError, ValueError) as error:
                source_metadata = source_payload[indexed_task]
                try:
                    migrated_source = require_complete_source_run_log(
                        adapt_source_run_log(
                            payload,
                            task_name=indexed_task,
                            task_parameters=dict(
                                source_metadata.get("params")
                                or source_metadata.get("task_parameters")
                                or {}
                            ),
                            seed=(
                                source_metadata.get("source_seed")
                                if source_metadata.get("source_seed") is not None
                                else source_metadata.get("task_random_seed")
                                if source_metadata.get("task_random_seed") is not None
                                else source_metadata.get("collect_seed")
                            ),
                            source_path=path,
                            screenshot_roots=screenshot_roots,
                            require_screenshots=False,
                        )
                    )
                except (TypeError, ValueError, FileNotFoundError) as migration_error:
                    raise ValueError(
                        f"indexed_source_run_log_invalid:{indexed_task}:{path}:"
                        f"{error}:legacy_migration_failed:{migration_error}"
                    ) from migration_error
                migrated_content = _json_bytes(migrated_source)
                migrated_digest = hashlib.sha256(migrated_content).hexdigest()
                migrated_object = _materialize_content(
                    root,
                    migrated_content,
                    migrated_digest,
                )
                source_run_log = migrated_source
            if source_run_log["task_name"] != indexed_task:
                raise ValueError(
                    "indexed_source_run_log_task_mismatch:"
                    f"{indexed_task}:{source_run_log['task_name']}:{path}"
                )
        task = indexed_paths.get(path) or _task_from_path(path, task_names)
        record = _register_run_log_record(
            records,
            memory_root=root,
            path=_materialize_object(root, path, digest),
            digest=digest,
            payload=payload,
            task=task,
            alias=str(path),
        )
        if indexed_task:
            indexed_canonical_digests[indexed_task] = migrated_digest or digest
        if migrated_source is not None and migrated_object is not None:
            migrated_record = _register_run_log_record(
                records,
                memory_root=root,
                path=migrated_object,
                digest=migrated_digest,
                payload=migrated_source,
                task=indexed_task,
            )
            migrated_record["migration"] = {
                "kind": "indexed_legacy_source_to_omniflow_run_log_v1",
                "source_path": str(path),
                "source_sha256": digest,
                "source_schema_version": str(payload.get("schema_version") or ""),
            }

    for record in records.values():
        record["aliases"] = sorted(set(record["aliases"]))
        record["tasks"] = sorted(set(record["tasks"]))

    canonical_sources: dict[str, dict[str, Any]] = {}
    for path, task in sorted(indexed_paths.items(), key=lambda item: item[1]):
        baseline_digest = _sha256(path)
        digest = indexed_canonical_digests.get(task, baseline_digest)
        canonical_sources[task] = dict(records[digest])
        canonical_payload = _load_object(
            Path(canonical_sources[task]["object_path"])
        )
        canonical_sources[task]["dependencies"] = (
            _materialize_run_log_dependencies(root, canonical_payload)
        )

    catalog_paths = sorted(
        {Path(value).expanduser().resolve() for value in function_catalogs}
    )
    resolved_result_roots = sorted(
        {Path(value).expanduser().resolve() for value in result_roots}
    )
    resolved_mobilegpt_memory_roots = sorted(
        {Path(value).expanduser().resolve() for value in mobilegpt_memory_roots}
    )
    resolved_baseline_batch_reports = sorted(
        {Path(value).expanduser().resolve() for value in baseline_batch_reports}
    )
    for mobilegpt_memory_root in resolved_mobilegpt_memory_roots:
        if not mobilegpt_memory_root.is_dir():
            raise FileNotFoundError(
                f"mobilegpt_memory_root_missing:{mobilegpt_memory_root}"
            )
    function_records, canonical_function_stores = _load_function_stores(
        root,
        catalog_paths,
        source_metadata=source_payload,
        canonical_sources=canonical_sources,
        screenshot_roots=screenshot_roots,
        run_log_records=records,
        existing_canonical_identities=existing_function_store_identities,
    )
    result_paths, result_records, canonical_result_cells = _load_results(
        root,
        resolved_result_roots,
        task_names,
        canonical_sources,
    )
    baseline_report_records, baseline_result_cells = _load_baseline_batch_reports(
        root,
        resolved_baseline_batch_reports,
        task_names=task_names,
    )
    canonical_result_cells.update(baseline_result_cells)
    mobilegpt_memory_records, canonical_mobilegpt_memories = (
        _load_mobilegpt_memories(
            root,
            resolved_mobilegpt_memory_roots,
            canonical_sources=canonical_sources,
        )
    )
    memory_source_index: dict[str, Any] = {}
    for task, raw_item in source_payload.items():
        item = dict(raw_item)
        item["retained_source_run_log"] = canonical_sources[str(task)]["object_path"]
        item["retained_source_run_log_sha256"] = canonical_sources[str(task)]["sha256"]
        source_state_catalog = raw_item.get("source_state_catalog") or raw_item.get(
            "transfer_state_catalog"
        )
        if source_state_catalog:
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
                "source_calls",
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
    mobilegpt_memory_index_path, mobilegpt_memory_index_hash = _publish_index(
        root,
        "mobilegpt_memory_index",
        canonical_mobilegpt_memories,
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
                "mobilegpt_memory": canonical_mobilegpt_memories.get(task),
                "result_cells": {
                    result: value
                    for result, value in canonical_result_cells.items()
                    if result.split("|", 1)[0] == task
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
            "source_run_log": "source_index_authoritative",
            "result": ("earliest_formal_protocol_compliant_validator_conclusion"),
            "success_cherry_picking": False,
        },
        "inputs": {
            "source_index": str(index_path),
            "source_index_sha256": _sha256(index_path),
            "source_screenshot_roots": [str(path) for path in screenshot_roots],
            "function_catalogs": [str(path) for path in catalog_paths],
            "runlog_roots": sorted(
                str(Path(value).expanduser().resolve()) for value in runlog_roots
            ),
            "result_roots": [str(path) for path in resolved_result_roots],
            "mobilegpt_memory_roots": [
                str(path) for path in resolved_mobilegpt_memory_roots
            ],
            "baseline_batch_reports": [
                str(path) for path in resolved_baseline_batch_reports
            ],
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
            "formal_protocol_excluded_results": sum(
                "canonical_exclusion_errors" in record
                for record in result_records.values()
            ),
            "canonical_result_cells": len(canonical_result_cells),
            "baseline_batch_reports": len(baseline_report_records),
            "baseline_validator_results": len(baseline_result_cells),
            "unique_mobilegpt_memories": len(mobilegpt_memory_records),
            "mobilegpt_memory_tasks": len(canonical_mobilegpt_memories),
        },
        "indexes": {
            "source_index": str(memory_source_index_path),
            "source_index_sha256": memory_source_index_hash,
            "ours_store_index": str(store_index_path),
            "ours_store_index_sha256": store_index_hash,
            "result_cells": str(result_cells_path),
            "result_cells_sha256": result_cells_hash,
            "mobilegpt_memory_index": str(mobilegpt_memory_index_path),
            "mobilegpt_memory_index_sha256": mobilegpt_memory_index_hash,
        },
        "artifacts": {
            "run_logs": {digest: records[digest] for digest in sorted(records)},
            "function_stores": {
                digest: function_records[digest] for digest in sorted(function_records)
            },
            "results": {
                digest: result_records[digest] for digest in sorted(result_records)
            },
            "mobilegpt_memories": {
                digest: mobilegpt_memory_records[digest]
                for digest in sorted(mobilegpt_memory_records)
            },
            "baseline_batch_reports": {
                digest: baseline_report_records[digest]
                for digest in sorted(baseline_report_records)
            },
        },
        "canonical": {
            "source_run_logs": canonical_sources,
            "function_stores": canonical_function_stores,
            "result_cells": canonical_result_cells,
            "mobilegpt_memories": canonical_mobilegpt_memories,
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
        "mobilegpt_memory_index": str(mobilegpt_memory_index_path),
        "mobilegpt_memory_index_sha256": mobilegpt_memory_index_hash,
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
    mobilegpt_memory_roots: Sequence[str | Path] = (),
    baseline_batch_reports: Sequence[str | Path] = (),
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
            mobilegpt_memory_roots=mobilegpt_memory_roots,
            baseline_batch_reports=baseline_batch_reports,
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
    mobilegpt_index = pointer.get("mobilegpt_memory_index")
    mobilegpt_index_hash = pointer.get("mobilegpt_memory_index_sha256")
    if mobilegpt_index is not None or mobilegpt_index_hash is not None:
        path = Path(str(mobilegpt_index or "")).expanduser()
        expected_hash = str(mobilegpt_index_hash or "")
        if not path.is_absolute() or not path.is_file():
            raise FileNotFoundError(
                f"artifact_memory_index_missing:mobilegpt_memory_index:{path}"
            )
        if not expected_hash or _sha256(path) != expected_hash:
            raise ValueError(
                "artifact_memory_index_hash_mismatch:mobilegpt_memory_index:"
                f"{path}"
            )
    return registry


def registered_result_plan_from_memory(
    *,
    memory_index: str | Path,
    task_name: str,
    methods: Sequence[str],
    devices: Sequence[str],
    source_seed: int,
    evaluation_seed: int,
    formal_max_steps: int | None = None,
    mobilegpt_memory_schemas: Sequence[str] | None = None,
) -> dict[str, list[tuple[str, str]]]:
    """Resolve completed formal results without rescanning historical results."""

    registry = load_artifact_memory(memory_index)
    results = registry["canonical"]["result_cells"]
    expected = [(method, device) for method in methods for device in devices]
    completed: list[tuple[str, str]] = []
    for method, device in expected:
        result_key = f"{task_name}|{method}|{device}|{source_seed}|{evaluation_seed}"
        record = results.get(result_key)
        if not isinstance(record, dict):
            continue
        if (
            method == "mobilegpt_offline_retrieval"
            and mobilegpt_memory_schemas is not None
            and record.get("selection_reason") != BASELINE_BATCH_REPORT_SELECTION
            and str(record.get("mobilegpt_memory_schema") or "")
            not in set(mobilegpt_memory_schemas)
        ):
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
                    f"artifact_memory_result_object_invalid:{result_key}:{object_path}"
                )
            payload = _load_object(object_path)
            rows = payload.get("rows") if isinstance(payload, dict) else None
            if (
                not isinstance(rows, list)
                or len(rows) != 1
                or not isinstance(rows[0], dict)
            ):
                raise ValueError(
                    f"artifact_memory_result_payload_invalid:{result_key}:{object_path}"
                )
            if record.get("selection_reason") != BASELINE_BATCH_REPORT_SELECTION:
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
        "pending": [result for result in expected if result not in completed],
    }


def canonical_mobilegpt_memory_from_memory(
    *,
    memory_index: str | Path,
    task_name: str,
) -> dict[str, Any] | None:
    """Resolve one validated task-local MobileGPT memory from current.json."""

    registry = load_artifact_memory(memory_index)
    record = registry.get("canonical", {}).get("mobilegpt_memories", {}).get(
        str(task_name)
    )
    if record is None:
        return None
    if not isinstance(record, dict):
        raise ValueError(f"mobilegpt_memory_index_record_invalid:{task_name}")
    memory_root = Path(str(record.get("memory_root") or "")).expanduser()
    if not memory_root.is_absolute() or not memory_root.is_dir():
        raise FileNotFoundError(
            f"mobilegpt_memory_index_root_missing:{task_name}:{memory_root}"
        )
    return dict(record)


def refresh_artifact_memory_from_pointer(
    *,
    memory_index: str | Path,
    source_screenshot_roots: Sequence[str | Path] = (),
    additional_function_catalogs: Sequence[str | Path] = (),
    additional_runlog_roots: Sequence[str | Path] = (),
    additional_result_roots: Sequence[str | Path] = (),
    additional_mobilegpt_memory_roots: Sequence[str | Path] = (),
    additional_baseline_batch_reports: Sequence[str | Path] = (),
    replace_recorded_roots: bool = False,
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
        resolved_runlog_roots = {
            str(Path(value).expanduser().resolve())
            for value in additional_runlog_roots
        }
        resolved_result_roots = {
            str(Path(value).expanduser().resolve())
            for value in additional_result_roots
        }
        if replace_recorded_roots:
            runlog_roots = sorted(resolved_runlog_roots)
            result_roots = sorted(resolved_result_roots)
        else:
            runlog_roots = sorted(
                {
                    *(str(value) for value in inputs.get("runlog_roots") or []),
                    *resolved_runlog_roots,
                }
            )
            result_roots = sorted(
                {
                    *(str(value) for value in inputs.get("result_roots") or []),
                    *resolved_result_roots,
                }
            )
        resolved_mobilegpt_memory_roots = {
            str(Path(value).expanduser().resolve())
            for value in additional_mobilegpt_memory_roots
        }
        if replace_recorded_roots:
            # An explicit refresh may replace stale task-local MobileGPT
            # bundles just like run-log and result roots.  Keeping the old
            # roots in the scan would validate them against the newly
            # selected canonical source and fail lineage checks before the
            # replacement bundles can be selected.
            mobilegpt_memory_roots = sorted(resolved_mobilegpt_memory_roots)
        else:
            mobilegpt_memory_roots = sorted(
                {
                    *(
                        str(value)
                        for value in inputs.get("mobilegpt_memory_roots") or []
                    ),
                    *resolved_mobilegpt_memory_roots,
                }
            )
        baseline_batch_reports = sorted(
            {
                *(str(value) for value in inputs.get("baseline_batch_reports") or []),
                *(
                    str(Path(value).expanduser().resolve())
                    for value in additional_baseline_batch_reports
                ),
            }
        )
        return _refresh_artifact_memory_unlocked(
            memory_root=pointer_path.parent,
            source_index=str(inputs["source_index"]),
            function_catalogs=function_catalogs,
            runlog_roots=runlog_roots,
            result_roots=result_roots,
            mobilegpt_memory_roots=mobilegpt_memory_roots,
            baseline_batch_reports=baseline_batch_reports,
            source_screenshot_roots=(
                source_screenshot_roots
                or tuple(inputs.get("source_screenshot_roots") or ())
            ),
            existing_function_store_identities={
                str(task): str(record.get("identity_sha256") or "")
                for task, record in (
                    registry.get("canonical", {}).get("function_stores", {}).items()
                )
                if isinstance(record, dict)
            },
        )


def _split_values(values: Sequence[str]) -> list[str]:
    return [item for value in values for item in value.split(":") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Maintain content-addressed long-term memory for AndroidWorld "
            "RunLogs, method assets, and registered results."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    refresh_parser = subparsers.add_parser("refresh")
    refresh_parser.add_argument("--memory-root", required=True)
    refresh_parser.add_argument("--source-index")
    refresh_parser.add_argument("--source-screenshot-root", action="append", default=[])
    refresh_parser.add_argument("--function-catalog", action="append", default=[])
    refresh_parser.add_argument("--runlog-root", action="append", required=True)
    refresh_parser.add_argument("--result-root", action="append", default=[])
    refresh_parser.add_argument(
        "--mobilegpt-memory-root",
        action="append",
        default=[],
    )
    refresh_parser.add_argument(
        "--baseline-batch-report",
        action="append",
        default=[],
    )
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
                source_screenshot_roots=args.source_screenshot_root,
                additional_function_catalogs=_split_values(args.function_catalog),
                additional_runlog_roots=_split_values(args.runlog_root),
                additional_result_roots=_split_values(args.result_root),
                additional_mobilegpt_memory_roots=_split_values(
                    args.mobilegpt_memory_root
                ),
                additional_baseline_batch_reports=_split_values(
                    args.baseline_batch_report
                ),
                replace_recorded_roots=True,
            )
        else:
            if not args.source_index:
                refresh_parser.error(
                    "--source-index is required when initializing memory"
                )
            report = refresh_artifact_memory(
                memory_root=memory_root,
                source_index=args.source_index,
                source_screenshot_roots=args.source_screenshot_root,
                function_catalogs=_split_values(args.function_catalog),
                runlog_roots=_split_values(args.runlog_root),
                result_roots=_split_values(args.result_root),
                mobilegpt_memory_roots=_split_values(
                    args.mobilegpt_memory_root
                ),
                baseline_batch_reports=_split_values(
                    args.baseline_batch_report
                ),
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
        output = registered_result_plan_from_memory(
            memory_index=args.memory_index,
            task_name=args.task,
            methods=tuple(item for item in args.methods.split(",") if item),
            devices=tuple(item for item in args.devices.split(",") if item),
            source_seed=SOURCE_SEED,
            evaluation_seed=TASK_SEED,
            formal_max_steps=MAX_STEPS,
        )
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


__all__ = [
    "load_artifact_memory",
    "canonical_mobilegpt_memory_from_memory",
    "refresh_artifact_memory",
    "refresh_artifact_memory_from_pointer",
    "registered_result_plan_from_memory",
]


if __name__ == "__main__":
    raise SystemExit(main())
