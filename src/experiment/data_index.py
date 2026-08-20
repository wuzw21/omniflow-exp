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
from omniflow.runlog import import_run_log_evidence
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
from src.experiment.paths import sha256_file
from src.integrations.runlog import adapt_source_run_log

MEMORY_SCHEMA = "omniflow.data.v4"
CURRENT_SCHEMA = "omniflow.data-index.v2"
_MISSING = object()
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


def _object_store_root(memory_root: Path) -> Path:
    """Legacy object-store location, accepted only while reading old indexes."""

    return memory_root / "androidworld" / ".archive" / "object_store"


def _index_record_root(memory_root: Path) -> Path:
    """Stable non-hash cache for index-generated compatibility records."""

    return memory_root / "androidworld" / ".archive" / "index_records"


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


def _localize_persisted_paths(value: Any, root: Path, key: str = "") -> Any:
    if isinstance(value, dict):
        localized: dict[str, Any] = {}
        for name, item in value.items():
            result = _localize_persisted_paths(item, root, str(name))
            if result is not _MISSING:
                localized[name] = result
        return localized
    if isinstance(value, list):
        localized_items = [
            result
            for item in value
            for result in (_localize_persisted_paths(item, root, key),)
            if result is not _MISSING
        ]
        return localized_items
    if not isinstance(value, str) or not value.startswith("/"):
        return value
    resolved = Path(value).expanduser().resolve()
    try:
        resolved.relative_to(root)
        return str(resolved)
    except ValueError:
        pass
    match = re.search(
        r"/objects/sha256/([0-9a-f]{2})/([0-9a-f]{64})(\.[^/]+)?$",
        value,
    )
    if match:
        suffix = match.group(3) or ".json"
        candidate = _object_store_root(root) / "sha256" / match.group(1) / (
            match.group(2) + suffix
        )
        if candidate.is_file():
            return str(candidate.resolve())
    if key.endswith("aliases") or key in {
        "source_path",
        "manifest_path",
        "provenance_path",
        "manifest_object_path",
        "provenance_object_path",
        "catalog_path",
    }:
        return _MISSING
    return value


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


def _load_source_index(path: Path) -> dict[str, Any]:
    payload = _load_object(path)
    if not isinstance(payload, dict):
        raise ValueError("source_index_must_be_object")
    if payload.get("schema_version") == CURRENT_SCHEMA:
        source_index = payload.get("source_index")
        if not isinstance(source_index, dict):
            raise ValueError("current_source_index_must_be_object")
        return source_index
    return payload


def _require_qualified_source_run_log(
    value: dict[str, Any],
    *,
    task: str,
    source_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        run_log = require_complete_source_run_log(value)
        if run_log.get("task_name") != task:
            raise ValueError("androidworld_source_run_log_task_mismatch")
        validator = run_log.get("validator")
        if not isinstance(validator, dict) or validator.get("success") is not True:
            raise ValueError("androidworld_source_run_log_validator_success_required")
        for step in run_log["steps"]:
            metadata = step.get("metadata")
            if not isinstance(metadata, dict) or not str(
                metadata.get("reasoning") or ""
            ).strip():
                raise ValueError("androidworld_source_run_log_reasoning_required")
            screenshot_path = str(metadata.get("screenshot_path") or "").strip()
            if not screenshot_path or not Path(screenshot_path).expanduser().is_file():
                raise ValueError("androidworld_source_run_log_screenshot_required")
            for observation_key in ("observation", "next_observation"):
                observation = step.get(observation_key)
                pixels = None
                if isinstance(observation, dict):
                    pixels = observation.get("screenshot")
                    if pixels is None:
                        pixels = observation.get("pixels")
                pixels_path = (
                    str(pixels.get("path") or "").strip()
                    if isinstance(pixels, dict)
                    else ""
                )
                if not pixels_path or not Path(pixels_path).expanduser().is_file():
                    raise ValueError(
                        f"androidworld_source_run_log_{observation_key}_screenshot_required"
                    )
        return run_log
    except (TypeError, ValueError) as strict_error:
        # The historical AndroidWorld source archive contains official,
        # successful v1 RunLogs whose screenshot binaries were stored in the
        # neighbouring observation object store and whose canonical JSON was
        # intentionally reduced to XML plus action evidence.  Keep
        # those sources usable when the authoritative source index says they
        # were successful.  This is deliberately limited to indexed sources;
        # newly collected sources still use the complete contract above.
        metadata = source_metadata if isinstance(source_metadata, dict) else {}
        historical = (
            metadata.get("latest_official_success_source") is True
            and str(metadata.get("source_kind") or "")
            in {
                "androidworld_validator_success_source_runlog",
                "one_time_canonicalized_seed111_screenshot_source",
            }
        )
        if not historical:
            raise
        payload_task = str(value.get("task_name") or "").strip()
        if payload_task and payload_task != task:
            raise ValueError("androidworld_source_run_log_task_mismatch") from strict_error
        if not _historical_source_success(value):
            raise ValueError("androidworld_source_run_log_success_required") from strict_error
        if not _historical_source_has_xml(value):
            raise ValueError("androidworld_source_run_log_xml_required") from strict_error
        return value


def _historical_source_success(value: dict[str, Any]) -> bool:
    if value.get("success") is True:
        return True
    if str(value.get("status") or "").strip().lower() in {"success", "succeeded"}:
        return True
    validator = value.get("validator")
    return isinstance(validator, dict) and validator.get("success") is True


def _historical_source_has_xml(value: dict[str, Any]) -> bool:
    def walk(item: Any) -> Iterable[Any]:
        if isinstance(item, dict):
            yield item
            for child in item.values():
                yield from walk(child)
        elif isinstance(item, list):
            for child in item:
                yield from walk(child)

    for item in walk(value):
        if isinstance(item.get("forest"), (str, dict)) and item.get("forest"):
            return True
        if isinstance(item.get("xml"), str) and item["xml"].strip():
            return True
    return False


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
    del memory_root
    resolved = source.expanduser().resolve()
    if not resolved.is_file() or sha256_file(resolved) != digest:
        raise ValueError(f"direct_object_invalid:{resolved}")
    return resolved


def _materialize_binary_object(
    memory_root: Path,
    source: Path,
    digest: str,
    *,
    suffix: str,
) -> Path:
    del memory_root, suffix
    resolved = source.expanduser().resolve()
    if not resolved.is_file() or sha256_file(resolved) != digest:
        raise ValueError(f"direct_binary_object_invalid:{resolved}")
    return resolved


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
        pixels = observation.get("screenshot")
        if pixels is None:
            pixels = observation.get("pixels")
        if not isinstance(pixels, dict):
            continue
        digest = str(pixels.get("sha256") or "").strip().lower()
        mime_type = str(pixels.get("mime_type") or "").strip()
        suffix = suffixes.get(mime_type)
        if suffix is None:
            raise ValueError(f"run_log_screenshot_mime_type_invalid:{mime_type}")
        source = Path(str(pixels.get("path") or "")).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"run_log_screenshot_missing:{source}")
        actual = sha256_file(source)
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
            "path": str(object_path),
        }
    return {"screenshots": [screenshots[key] for key in sorted(screenshots)]}


def _materialize_content(memory_root: Path, content: bytes, digest: str) -> Path:
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError(f"memory_content_hash_mismatch:{digest}")
    root = _index_record_root(memory_root)
    root.mkdir(parents=True, exist_ok=True)
    existing = sorted(root.glob("record_*.json"))
    for target in existing:
        if target.is_file() and target.read_bytes() == content:
            return target.resolve()
    numbers = [
        int(path.stem.removeprefix("record_"))
        for path in existing
        if path.stem.removeprefix("record_").isdigit()
    ]
    target = root / f"record_{max(numbers, default=0) + 1:03d}.json"
    _atomic_write(target, content)
    target.chmod(0o444)
    return target.resolve()


def _require_hashed_file(value: Any, expected: Any, *, label: str) -> Path:
    path = Path(str(value or "")).expanduser().resolve()
    expected_hash = str(expected or "").strip()
    if not path.is_file():
        raise FileNotFoundError(f"{label}_missing:{path}")
    actual = sha256_file(path)
    if not expected_hash or actual != expected_hash:
        raise ValueError(
            f"{label}_hash_mismatch:"
            f"expected={expected_hash or 'missing'}:actual={actual}"
        )
    return path


def _link_object(source: Path, target: Path, expected_hash: str) -> None:
    if target.exists():
        if not target.is_file() or sha256_file(target) != expected_hash:
            raise ValueError(f"memory_runtime_hash_mismatch:{target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
        if sha256_file(temporary) != expected_hash:
            raise ValueError(f"memory_runtime_copy_hash_mismatch:{source}")
        temporary.chmod(0o444)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


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
            and ".archive" not in path.relative_to(resolved).parts
        )
        paths.update(
            path.resolve()
            for path in resolved.rglob("run_log.json")
            if not path.name.startswith("._")
            and ".archive" not in path.relative_to(resolved).parts
        )
    return sorted(paths)


def _result_paths(roots: Iterable[Path]) -> list[Path]:
    paths: set[Path] = set()
    for root in roots:
        resolved = root.expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"result_root_missing:{resolved}")
        for name in RESULT_FILE_NAMES:
            paths.update(
                path.resolve()
                for path in resolved.rglob(name)
                if ".archive" not in path.relative_to(resolved).parts
            )
    return sorted(paths)


def _canonical_function_store_paths(memory_root: Path) -> list[Path]:
    paths: list[Path] = []
    for environment in ("androidworld", "bmoca"):
        environment_root = memory_root / environment
        if not environment_root.is_dir():
            continue
        for path in environment_root.rglob("function_store.json"):
            relative = path.relative_to(environment_root).parts
            androidworld_layout = (
                environment == "androidworld"
                and len(relative) == 6
                and relative[3] == "memory"
                and relative[5] == "function_store.json"
                and not relative[0].startswith(".")
            )
            bmoca_layout = (
                environment == "bmoca"
                and len(relative) == 6
                and relative[2] == "function"
                and relative[5] == "function_store.json"
            )
            if androidworld_layout or bmoca_layout:
                paths.append(path.resolve())
    return sorted(set(paths))


def _function_bundle_identity(
    memory_root: Path,
    store_path: Path,
) -> dict[str, str]:
    for environment in ("androidworld", "bmoca"):
        environment_root = memory_root / environment
        try:
            relative = store_path.resolve().relative_to(environment_root.resolve())
        except ValueError:
            continue
        parts = relative.parts
        if environment == "androidworld" and (
            len(parts) == 6
            and parts[3] == "memory"
            and parts[5] == "function_store.json"
        ):
            return {
                "environment": environment,
                "task": parts[0],
                "device": parts[2],
                "category": "function",
                "method": parts[1],
                "attempt_id": parts[4],
            }
        if environment != "bmoca" or (
            len(parts) != 6
            or parts[2] != "function"
            or parts[5] != "function_store.json"
        ):
            raise ValueError(f"function_store_path_not_canonical:{store_path}")
        return {
            "environment": environment,
            "task": parts[0],
            "device": parts[1],
            "category": parts[2],
            "method": parts[3],
            "attempt_id": parts[4],
        }
    raise ValueError(f"function_store_path_not_canonical:{store_path}")


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
    source_sha256 = sha256_file(source_run_log)
    if source_payload.get("schema_version") == "omniflow.run_log.v1":
        source_payload, _source_catalog = import_run_log_evidence(
            source_payload,
            evidence_root=source_run_log.parent,
        )
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
    if str(manifest.get("registered_result_sha256") or "") != sha256_file(path):
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
    actual_manifest_sha256 = sha256_file(manifest_path)
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
            actual_sha256 = sha256_file(path)
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
    if method == "mobilegpt":
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
        digest = sha256_file(path)
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
        manifest_digest = sha256_file(manifest_path)
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
        if method == "mobilegpt":
            prep_manifest_path = Path(
                str(result_row.get("prep_manifest") or "")
            ).expanduser()
            prep_manifest = _load_object(prep_manifest_path)
            candidate["memory_schema"] = str(
                prep_manifest.get("schema_version") or ""
            )
            candidate["memory_source_method"] = str(
                prep_manifest.get("source_method") or ""
            )
            candidate["memory_prep_type"] = str(
                result_row.get("prep_type") or ""
            )
        if method == "fixed_replay":
            candidate["source_run_log_sha256"] = sha256_file(
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
            if value[3]["method"] == "mobilegpt"
            and value[3].get("memory_schema")
            in MOBILEGPT_SUPPORTED_MEMORY_SCHEMAS
        ]
        canonical[result] = (current_mobilegpt or ordered)[0][3]
    return paths, records, canonical


def _load_function_stores(
    memory_root: Path,
    stores: Sequence[Path],
    *,
    source_metadata: dict[str, Any],
    canonical_sources: dict[str, dict[str, Any]],
    screenshot_roots: Sequence[Path],
    run_log_records: dict[str, dict[str, Any]],
    existing_canonical_identities: dict[str, str] | None = None,
    existing_canonical_stores: dict[str, dict[str, Any]] | None = None,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    records: dict[str, dict[str, Any]] = {}
    candidates: dict[str, list[tuple[tuple[int, int], str]]] = {}
    previous_stores = existing_canonical_stores or {}
    for store in stores:
        if not store.is_file():
            raise FileNotFoundError(f"function_store_missing:{store}")
        bundle_identity = _function_bundle_identity(memory_root, store)
        task = bundle_identity["task"]
        task_source_metadata = source_metadata.get(task)
        if not isinstance(task_source_metadata, dict):
            raise ValueError(f"function_source_metadata_missing:{task}")
        if task_source_metadata.get("latest_official_success_source") is not True:
            continue
        source_run_log = store.with_name("run_log.json")
        transfer = store.with_name("transfer_states.json")
        if not source_run_log.is_file():
            continue
        if not transfer.is_file():
            continue
        store_payload = _load_object(store)
        if (
            not isinstance(store_payload, dict)
            or store_payload.get("schema_version") != "omniflow.store.v2"
            or not isinstance(store_payload.get("functions"), dict)
        ):
            raise ValueError(f"function_store_invalid:{task}:{store}")
        if len(store_payload["functions"]) != 1:
            raise ValueError(f"function_store_single_function_required:{task}")
        source_calls = store_payload.get("source_calls")
        if (
            not isinstance(source_calls, list)
            or len(source_calls) != 1
            or not isinstance(source_calls[0], dict)
            or not isinstance(source_calls[0].get("arguments"), dict)
            or str(source_calls[0].get("function_id") or "")
            != next(iter(store_payload["functions"]))
        ):
            raise ValueError(f"function_store_single_source_call_required:{task}")
        store_hash = sha256_file(store)
        transfer_hash = sha256_file(transfer)
        source_run_log_hash = sha256_file(source_run_log)
        raw_source_payload = _load_object(source_run_log)
        if not isinstance(raw_source_payload, dict):
            raise ValueError(f"function_source_run_log_invalid:{task}")
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
            source_metadata=dict(task_source_metadata),
            canonical_source=canonical_source,
            screenshot_roots=screenshot_roots,
            records=run_log_records,
        )
        identity = hashlib.sha256(
            "\0".join(
                (source_run_log_hash, store_hash, transfer_hash)
            ).encode("utf-8")
        ).hexdigest()
        record = records.setdefault(
            identity,
            {
                "identity_sha256": identity,
                "tasks": [],
                "function_count": len(store_payload["functions"]),
                "source_calls": source_calls,
                "source_run_log_path": str(canonical_source_run_log),
                "source_run_log_sha256": canonical_source_run_log_hash,
                "source_run_log_lineage": source_run_log_lineage,
                "store_path": str(store),
                "store_sha256": store_hash,
                "transfer_states_path": str(transfer),
                "transfer_states_sha256": transfer_hash,
                "environment": bundle_identity["environment"],
                "device": bundle_identity["device"],
                "category": bundle_identity["category"],
                "method": bundle_identity["method"],
                "attempt_id": bundle_identity["attempt_id"],
            },
        )
        record["tasks"].append(task)
        candidates.setdefault(task, []).append(((1,), identity))

    for record in records.values():
        record["tasks"] = sorted(set(record["tasks"]))

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
    for task, previous in sorted(previous_stores.items()):
        if task in canonical or not isinstance(previous, dict):
            continue
        canonical[task] = dict(previous)
        identity = str(previous.get("identity_sha256") or "").strip()
        if identity:
            records.setdefault(identity, dict(previous))
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


def _load_prepared_memories(
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
        from src.integrations.mobilegpt import validate_prepared_memory

        validated = validate_prepared_memory(
            memory_path,
            task_name=task,
            source_seed=SOURCE_SEED,
            source_run_log=str(source["object_path"]),
            expected_model=str(manifest.get("source_model") or ""),
            expected_source_method=source_method,
        )
        memory_sha256 = str(validated["memory_sha256"])
        manifest_sha256 = sha256_file(manifest_path)
        record = records.setdefault(
            memory_sha256,
            {
                "task": task,
                "provider": "mobilegpt",
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
        canonical[task] = _select_canonical_prepared_memory(
            task=task,
            memory_sha256s=memory_sha256s,
            records=records,
        )
    return records, canonical


def _select_canonical_prepared_memory(
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
        summary_digest = sha256_file(summary_path)
        results_digest = sha256_file(results_path)
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


def _refresh_data_index_unlocked(
    *,
    memory_root: str | Path,
    source_index: str | Path,
    runlog_roots: Sequence[str | Path],
    result_roots: Sequence[str | Path],
    prepared_memory_roots: Sequence[str | Path] = (),
    baseline_batch_reports: Sequence[str | Path] = (),
    source_screenshot_roots: Sequence[str | Path] = (),
    existing_function_store_identities: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Import immutable evidence and publish one deterministic canonical index."""

    root = Path(memory_root).expanduser().resolve()
    index_path = Path(source_index).expanduser().resolve()
    if not index_path.is_file():
        raise FileNotFoundError(f"source_index_missing:{index_path}")
    source_payload = _load_source_index(index_path)
    previous_registry = load_data_index(index_path)
    previous_sources = previous_registry.get("canonical", {}).get(
        "source_run_logs", {}
    )
    if not isinstance(previous_sources, dict):
        previous_sources = {}
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
        # A source index may retain an explicitly unqualified/pending source
        # for provenance.  It is not an input to the canonical run, Function
        # store, or result registry, and must not make an otherwise valid
        # refresh fail merely because that historical artifact lacks the
        # strict source-evidence contract.
        if item.get("latest_official_success_source") is not True:
            continue
        reference = item.get("retained_source_run_log") or item.get("source_run_log")
        if not reference:
            raise ValueError(f"source_index_run_log_required:{task}")
        path = _resolve_index_reference(index_path, reference)
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
        digest = sha256_file(path)
        payload = _load_object(path)
        if not isinstance(payload, dict):
            raise ValueError(f"run_log_must_be_object:{path}")
        indexed_task = indexed_paths.get(path)
        migrated_source: dict[str, Any] | None = None
        migrated_digest = ""
        migrated_object: Path | None = None
        if indexed_task:
            try:
                source_run_log = _require_qualified_source_run_log(
                    payload,
                    task=indexed_task,
                    source_metadata=source_payload[indexed_task],
                )
            except (TypeError, ValueError) as error:
                source_metadata = source_payload[indexed_task]
                try:
                    migrated_source = _require_qualified_source_run_log(
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
                        ),
                        task=indexed_task,
                        source_metadata=source_payload[indexed_task],
                    )
                except (TypeError, ValueError, FileNotFoundError) as migration_error:
                    # Keep an already-published canonical source for an
                    # unrelated task when its retained historical artifact
                    # no longer satisfies the strict collection contract.
                    # The invalid artifact remains visible in provenance, but
                    # must not block registration of a new valid Function or
                    # result cell for another task.
                    previous = previous_sources.get(indexed_task)
                    if isinstance(previous, dict):
                        indexed_paths.pop(path, None)
                        continue
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
        baseline_digest = sha256_file(path)
        digest = indexed_canonical_digests.get(task, baseline_digest)
        canonical_sources[task] = dict(records[digest])
        canonical_payload = _load_object(
            Path(canonical_sources[task]["object_path"])
        )
        try:
            canonical_sources[task]["dependencies"] = (
                _materialize_run_log_dependencies(root, canonical_payload)
            )
        except (FileNotFoundError, TypeError, ValueError):
            previous = previous_sources.get(task)
            if not isinstance(previous, dict):
                raise
            # Older canonical records already point to content-addressed
            # screenshot dependencies.  Preserve those verified references
            # when the raw RunLog still contains a stale workstation path.
            canonical_sources[task] = dict(previous)

    for task, previous in previous_sources.items():
        if task in canonical_sources or task not in source_payload:
            continue
        if not isinstance(previous, dict):
            continue
        previous_path = Path(str(previous.get("object_path") or "")).expanduser()
        previous_digest = str(previous.get("sha256") or "").strip().lower()
        if (
            not previous_path.is_file()
            or not previous_digest
            or sha256_file(previous_path) != previous_digest
        ):
            continue
        preserved = dict(previous)
        preserved["object_path"] = str(previous_path.resolve())
        canonical_sources[task] = preserved
        records.setdefault(previous_digest, preserved)

    function_store_paths = _canonical_function_store_paths(root)
    resolved_result_roots = sorted(
        {Path(value).expanduser().resolve() for value in result_roots}
    )
    resolved_prepared_memory_roots = sorted(
        {Path(value).expanduser().resolve() for value in prepared_memory_roots}
    )
    resolved_baseline_batch_reports = sorted(
        {Path(value).expanduser().resolve() for value in baseline_batch_reports}
    )
    for prepared_memory_root in resolved_prepared_memory_roots:
        if not prepared_memory_root.is_dir():
            raise FileNotFoundError(
                f"prepared_memory_root_missing:{prepared_memory_root}"
            )
    function_records, canonical_function_stores = _load_function_stores(
        root,
        function_store_paths,
        source_metadata=source_payload,
        canonical_sources=canonical_sources,
        screenshot_roots=screenshot_roots,
        run_log_records=records,
        existing_canonical_identities=existing_function_store_identities,
        existing_canonical_stores=previous_registry.get("canonical", {}).get(
            "function_stores", {}
        ),
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
    prepared_memory_records, canonical_prepared_memories = (
        _load_prepared_memories(
            root,
            resolved_prepared_memory_roots,
            canonical_sources=canonical_sources,
        )
    )
    memory_source_index: dict[str, Any] = {}
    for task, raw_item in source_payload.items():
        item = dict(raw_item)
        item.pop("retained_source_run_log_sha256", None)
        item.pop("source_run_log_sha256", None)
        canonical_source = canonical_sources.get(str(task))
        if not isinstance(canonical_source, dict):
            item["retained_source_run_log"] = ""
            memory_source_index[str(task)] = item
            continue
        item["retained_source_run_log"] = canonical_source["object_path"]
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
            catalog_digest = sha256_file(catalog_path)
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
    registry: dict[str, Any] = {
        "schema_version": MEMORY_SCHEMA,
        "policy": {
            "deduplication": "direct_paths",
            "source_run_log": "source_index_authoritative",
            "result": ("earliest_formal_protocol_compliant_validator_conclusion"),
            "success_cherry_picking": False,
        },
        "inputs": {
            "source_index": str(root / "current.json"),
            "source_screenshot_roots": [str(path) for path in screenshot_roots],
            "runlog_roots": sorted(
                str(Path(value).expanduser().resolve()) for value in runlog_roots
            ),
            "result_roots": [str(path) for path in resolved_result_roots],
            "prepared_memory_roots": [
                str(path) for path in resolved_prepared_memory_roots
            ],
            "baseline_batch_reports": [
                str(path) for path in resolved_baseline_batch_reports
            ],
        },
        "counts": {
            "task_count": len(task_names),
            "run_log_paths": len(paths),
            "unique_run_logs": len(records),
            "function_store_paths": len(function_store_paths),
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
            "unique_prepared_memories": len(prepared_memory_records),
            "prepared_memory_tasks": len(canonical_prepared_memories),
        },
        "canonical": {
            "source_run_logs": canonical_sources,
            "function_stores": canonical_function_stores,
            "result_cells": canonical_result_cells,
            "prepared_memories": canonical_prepared_memories,
        },
    }
    registry["schema_version"] = CURRENT_SCHEMA
    registry["source_index"] = memory_source_index
    registry = _localize_persisted_paths(registry, root)
    _atomic_write(root / "current.json", _json_bytes(registry))
    return registry


def refresh_data_index(
    *,
    memory_root: str | Path,
    source_index: str | Path,
    runlog_roots: Sequence[str | Path],
    result_roots: Sequence[str | Path],
    prepared_memory_roots: Sequence[str | Path] = (),
    baseline_batch_reports: Sequence[str | Path] = (),
    source_screenshot_roots: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Import immutable evidence and publish one deterministic canonical index."""

    root = Path(memory_root).expanduser().resolve()
    with _memory_lock(root):
        return _refresh_data_index_unlocked(
            memory_root=root,
            source_index=source_index,
            runlog_roots=runlog_roots,
            result_roots=result_roots,
            prepared_memory_roots=prepared_memory_roots,
            baseline_batch_reports=baseline_batch_reports,
            source_screenshot_roots=source_screenshot_roots,
        )


def load_data_index(memory_index: str | Path) -> dict[str, Any]:
    """Load and verify the single canonical data index."""

    pointer_path = Path(memory_index).expanduser().resolve()
    current = _load_object(pointer_path)
    if not isinstance(current, dict) or current.get("schema_version") != CURRENT_SCHEMA:
        raise ValueError(f"local_data_index_invalid:{pointer_path}")
    if not isinstance(current.get("canonical"), dict):
        raise ValueError(f"local_data_index_canonical_missing:{pointer_path}")
    if not isinstance(current.get("source_index"), dict):
        raise ValueError(f"local_data_index_source_missing:{pointer_path}")
    return current


def registered_result_plan_from_memory(
    *,
    memory_index: str | Path,
    task_name: str,
    methods: Sequence[str],
    devices: Sequence[str],
    source_seed: int,
    evaluation_seed: int,
    formal_max_steps: int | None = None,
    prepared_memory_schemas: Sequence[str] | None = None,
) -> dict[str, list[tuple[str, str]]]:
    """Resolve completed formal results without rescanning historical results."""

    registry = load_data_index(memory_index)
    results = registry["canonical"]["result_cells"]
    expected = [(method, device) for method in methods for device in devices]
    completed: list[tuple[str, str]] = []
    for method, device in expected:
        result_key = f"{task_name}|{method}|{device}|{source_seed}|{evaluation_seed}"
        record = results.get(result_key)
        if not isinstance(record, dict):
            continue
        if record.get("official_validator_success") is not True:
            continue
        if (
            method == "mobilegpt"
            and prepared_memory_schemas is not None
            and record.get("selection_reason") != BASELINE_BATCH_REPORT_SELECTION
            and str(record.get("memory_schema") or "")
            not in set(prepared_memory_schemas)
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
                or sha256_file(object_path) != expected_hash
            ):
                raise ValueError(
                    f"local_data_result_object_invalid:{result_key}:{object_path}"
                )
            payload = _load_object(object_path)
            rows = payload.get("rows") if isinstance(payload, dict) else None
            if (
                not isinstance(rows, list)
                or len(rows) != 1
                or not isinstance(rows[0], dict)
            ):
                raise ValueError(
                    f"local_data_result_payload_invalid:{result_key}:{object_path}"
                )
            if record.get("selection_reason") != BASELINE_BATCH_REPORT_SELECTION:
                details = payload.get("details") if isinstance(payload, dict) else None
                detail_row = next(
                    (
                        detail
                        for detail in details or ()
                        if isinstance(detail, dict)
                        and str(detail.get("method") or "") == method
                        and str(detail.get("device") or "") == device
                    ),
                    rows[0],
                )
                from src.experiment.result_registry import (
                    validate_formal_result_protocol,
                )

                validate_formal_result_protocol(
                    detail_row,
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


def canonical_prepared_memory_from_index(
    *,
    memory_index: str | Path,
    task_name: str,
    provider: str = "mobilegpt",
) -> dict[str, Any] | None:
    """Resolve one validated prepared memory from the local data index."""

    registry = load_data_index(memory_index)
    record = registry.get("canonical", {}).get("prepared_memories", {}).get(str(task_name))
    if record is None:
        return None
    if not isinstance(record, dict):
        raise ValueError(f"prepared_memory_index_record_invalid:{task_name}")
    if str(record.get("provider") or "") != str(provider or "").strip():
        raise ValueError(
            f"prepared_memory_provider_mismatch:{task_name}:"
            f"expected={provider}:actual={record.get('provider')}"
        )
    memory_root = Path(str(record.get("memory_root") or "")).expanduser()
    if not memory_root.is_absolute() or not memory_root.is_dir():
        raise FileNotFoundError(
            f"prepared_memory_index_root_missing:{task_name}:{memory_root}"
        )
    return dict(record)


def refresh_data_index_from_pointer(
    *,
    memory_index: str | Path,
    source_screenshot_roots: Sequence[str | Path] = (),
    additional_runlog_roots: Sequence[str | Path] = (),
    additional_result_roots: Sequence[str | Path] = (),
    additional_prepared_memory_roots: Sequence[str | Path] = (),
    additional_baseline_batch_reports: Sequence[str | Path] = (),
    replace_recorded_roots: bool = False,
) -> dict[str, Any]:
    """Refresh a memory using its recorded inputs plus newly completed evidence."""

    pointer_path = Path(memory_index).expanduser().resolve()
    with _memory_lock(pointer_path.parent):
        registry = load_data_index(pointer_path)
        inputs = registry["inputs"]
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
        resolved_prepared_memory_roots = {
            str(Path(value).expanduser().resolve())
            for value in additional_prepared_memory_roots
        }
        if replace_recorded_roots:
            # An explicit refresh may replace stale task-local MobileGPT
            # bundles just like run-log and result roots.  Keeping the old
            # roots in the scan would validate them against the newly
            # selected canonical source and fail lineage checks before the
            # replacement bundles can be selected.
            prepared_memory_roots = sorted(resolved_prepared_memory_roots)
        else:
            prepared_memory_roots = sorted(
                {
                    *(
                        str(value)
                        for value in inputs.get("prepared_memory_roots") or []
                    ),
                    *resolved_prepared_memory_roots,
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
        return _refresh_data_index_unlocked(
            memory_root=pointer_path.parent,
            source_index=str(inputs["source_index"]),
            runlog_roots=runlog_roots,
            result_roots=result_roots,
            prepared_memory_roots=prepared_memory_roots,
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
            report = refresh_data_index_from_pointer(
                memory_index=pointer_path,
                source_screenshot_roots=args.source_screenshot_root,
                additional_runlog_roots=_split_values(args.runlog_root),
                additional_result_roots=_split_values(args.result_root),
                additional_prepared_memory_roots=_split_values(
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
            report = refresh_data_index(
                memory_root=memory_root,
                source_index=args.source_index,
                source_screenshot_roots=args.source_screenshot_root,
                runlog_roots=_split_values(args.runlog_root),
                result_roots=_split_values(args.result_root),
                prepared_memory_roots=_split_values(
                    args.mobilegpt_memory_root
                ),
                baseline_batch_reports=_split_values(
                    args.baseline_batch_report
                ),
            )
        output = {
            "current": str(pointer_path),
            "counts": report["counts"],
        }
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
    "load_data_index",
    "canonical_prepared_memory_from_index",
    "refresh_data_index",
    "refresh_data_index_from_pointer",
    "registered_result_plan_from_memory",
]


if __name__ == "__main__":
    raise SystemExit(main())
