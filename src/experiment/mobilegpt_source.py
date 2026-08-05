"""Prepare immutable MobileGPT memory from canonical RunLogs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import datetime
import json
from pathlib import Path
import time
from typing import Any

from src.experiment import androidworld as pipeline
from src.experiment.mobilegpt_contract import (
    MOBILEGPT_DIRECT_LEARNING_MODE,
    MOBILEGPT_DIRECT_MEMORY_SCHEMA,
    MOBILEGPT_DIRECT_SOURCE_METHOD,
    MOBILEGPT_MEMORY_MANIFEST,
    MOBILEGPT_SOURCE_METHOD_BY_SCHEMA,
)
from src.integrations.mobilegpt_converter import (
    MobileGPTConversionError,
    convert_runlog_to_mobilegpt_memory,
    preflight_runlog_conversion,
    write_conversion_failure_audit,
)
from src.integrations.runlog import import_run_log

SOURCE_SEED = 111
_IGNORED_SOURCE_PACKAGES = {
    "com.android.systemui",
    "com.example.MobileGPT",
    "com.google.android.apps.nexuslauncher",
}


def load_canonical_source_item(
    index_path: str | Path,
    *,
    task_name: str,
) -> pipeline.ArchivedRunLog:
    matches = [
        item
        for item in pipeline.load_archive_index(index_path)
        if item.task == str(task_name)
    ]
    if len(matches) != 1:
        raise ValueError(
            "mobilegpt_source_task_resolution_failed:"
            f"task={task_name}:matches={len(matches)}"
        )
    item = matches[0]
    source_kind = str(item.meta.get("source_kind") or "").strip()
    if item.meta.get("latest_official_success_source") is not True:
        raise ValueError(
            f"mobilegpt_source_official_success_required:task={task_name}"
        )
    if source_kind and source_kind != "androidworld_validator_success_source_runlog":
        raise ValueError(
            "mobilegpt_source_kind_invalid:"
            f"task={task_name}:actual={source_kind}"
        )
    if not item.source_run_log.is_file():
        raise FileNotFoundError(
            f"mobilegpt_source_runlog_missing:{item.source_run_log}"
        )
    expected_sha256 = str(
        item.meta.get("retained_source_run_log_sha256")
        or item.meta.get("source_run_log_sha256")
        or ""
    ).strip()
    actual_sha256 = pipeline._file_sha256(item.source_run_log)
    if not expected_sha256 or expected_sha256 != actual_sha256:
        raise ValueError(
            f"mobilegpt_source_runlog_hash_mismatch:task={task_name}"
        )
    canonical = import_run_log(
        json.loads(item.source_run_log.read_text(encoding="utf-8"))
    )
    if (
        canonical.get("status") != "succeeded"
        or canonical.get("success") is not True
        or not canonical.get("steps")
    ):
        raise ValueError(
            f"mobilegpt_source_runlog_not_successful:task={task_name}"
        )
    return item


def _mobilegpt_source_target(
    *,
    item: pipeline.ArchivedRunLog,
    source: dict[str, Any],
) -> dict[str, str]:
    inferred = pipeline._infer_mobilegpt_target_from_source_run_log(item)
    package_name = str(inferred.get("target_package") or "").strip()
    if package_name:
        return {
            key: str(value)
            for key, value in inferred.items()
            if value is not None
        }
    source_packages: set[str] = set()
    for step in source.get("steps") or []:
        observation = step.get("observation") if isinstance(step, dict) else None
        auxiliaries = (
            observation.get("auxiliaries")
            if isinstance(observation, dict)
            else None
        )
        package = str(
            auxiliaries.get("package_name")
            if isinstance(auxiliaries, dict)
            else ""
        ).strip()
        if package and package not in _IGNORED_SOURCE_PACKAGES:
            source_packages.add(package)
    if len(source_packages) != 1:
        label = "unresolved" if not source_packages else "ambiguous"
        raise ValueError(
            f"mobilegpt_source_target_package_{label}:"
            + ",".join(sorted(source_packages))
        )
    package_name = next(iter(source_packages))
    return {
        "target_package": package_name,
        "target_app": package_name,
        "target_source": "canonical_source_runlog_observation",
    }


def _source_preflight(
    item: pipeline.ArchivedRunLog,
) -> tuple[Path, tuple[str, ...], dict[str, Any], dict[str, str]]:
    source_run_log = item.source_run_log
    source_sha256 = pipeline._file_sha256(source_run_log)
    source = import_run_log(
        json.loads(source_run_log.read_text(encoding="utf-8"))
    )
    target_info = _mobilegpt_source_target(item=item, source=source)
    report = preflight_runlog_conversion(
        source_run_log,
        target_package=str(target_info.get("target_package") or ""),
        target_app=str(target_info.get("target_app") or ""),
    )
    if report.get("ready") is not True:
        raise MobileGPTConversionError(
            str(report.get("failure_code") or "mobilegpt_conversion_preflight_failed"),
            **dict(report.get("failure_details") or {}),
        )
    audit = {
        "schema_version": "omniflow.mobilegpt-conversion-preflight.v1",
        "grounding_source": "canonical_androidworld_run_log",
        "source_run_log": str(source_run_log),
        "source_run_log_sha256": source_sha256,
        "actions_supplied_to_mobilegpt": True,
        "function_store_used": False,
        "report": report,
    }
    return source_run_log, (source_sha256,), audit, target_info


def preflight_mobilegpt_source(
    *,
    index_path: str | Path,
    task_name: str,
) -> dict[str, Any]:
    item = load_canonical_source_item(index_path, task_name=task_name)
    _, _, source_audit, target_info = _source_preflight(item)
    report = dict(source_audit["report"])
    return {
        "schema_version": "omniflow.mobilegpt-source-preflight.v4",
        "task_name": item.task,
        "source_seed": SOURCE_SEED,
        "source_method": MOBILEGPT_DIRECT_SOURCE_METHOD,
        "source_run_log": source_audit["source_run_log"],
        "source_run_log_sha256": source_audit["source_run_log_sha256"],
        "learning_mode": MOBILEGPT_DIRECT_LEARNING_MODE,
        "teacher_forcing": False,
        "synthetic_subtasks": True,
        "semantic_subtasks": False,
        "original_mobilegpt_prompts": False,
        "actions_supplied_to_mobilegpt": True,
        "source_transitions_supplied": True,
        "source_success_boundary_supplied": True,
        "function_store_used": False,
        "transition_count": int(report["transition_count"]),
        "action_type_counts": dict(report["action_type_counts"]),
        "skipped_actions": list(report["skipped_actions"]),
        "target_package": target_info["target_package"],
        "target_source": target_info["target_source"],
        "source_audit": source_audit,
        "ready": True,
    }


def validate_mobilegpt_source_memory(
    *,
    index_path: str | Path,
    task_name: str,
    memory_root: str | Path,
    model: str,
    memory_index: str | Path | None = None,
) -> dict[str, Any]:
    item = load_canonical_source_item(index_path, task_name=task_name)
    source_run_log, compatible_sha256s, _, _ = _source_preflight(item)
    manifest_path = Path(memory_root).expanduser().resolve().parent / MOBILEGPT_MEMORY_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = str(manifest.get("schema_version") or "")
    try:
        source_method = MOBILEGPT_SOURCE_METHOD_BY_SCHEMA[schema_version]
    except KeyError as error:
        raise ValueError("mobilegpt_source_memory_schema_invalid") from error
    validated = pipeline.validate_mobilegpt_adapted_memory(
        memory_root,
        task_name=item.task,
        source_seed=SOURCE_SEED,
        source_run_log=source_run_log,
        compatible_source_sha256s=compatible_sha256s,
        expected_model=str(model),
        expected_source_method=source_method,
    )
    result = {
        "schema_version": "omniflow.mobilegpt-source-validation.v4",
        "task_name": item.task,
        "source_seed": SOURCE_SEED,
        "source_method": source_method,
        "source_run_log": str(source_run_log),
        "model": str(model),
        "validated": validated,
    }
    if memory_index is not None:
        result["memory_registration"] = _register_mobilegpt_memory(
            memory_index=memory_index,
            bundle_root=Path(memory_root).expanduser().resolve().parent,
            task_name=item.task,
        )
    return result


def _register_mobilegpt_memory(
    *,
    memory_index: str | Path,
    bundle_root: str | Path,
    task_name: str,
) -> dict[str, Any]:
    from src.experiment.artifact_memory import refresh_artifact_memory_from_pointer

    report = refresh_artifact_memory_from_pointer(
        memory_index=memory_index,
        additional_mobilegpt_memory_roots=(bundle_root,),
    )
    registered = report.get("canonical", {}).get("mobilegpt_memories", {}).get(
        str(task_name)
    )
    if not isinstance(registered, dict):
        raise ValueError(f"mobilegpt_memory_registration_missing:{task_name}")
    return registered


def prepare_mobilegpt_source_memory(
    *,
    index_path: str | Path,
    task_name: str,
    mobilegpt_root: str | Path,
    output_root: str | Path,
    model: str,
    memory_index: str | Path | None = None,
) -> dict[str, Any]:
    normalized_model = str(model or "").strip()
    if not normalized_model:
        raise ValueError("mobilegpt_source_model_required")
    item = load_canonical_source_item(index_path, task_name=task_name)
    source_run_log, _, source_audit, target_info = _source_preflight(item)
    bundle_root = Path(output_root).expanduser().resolve()
    if bundle_root.exists():
        raise FileExistsError(
            f"immutable_mobilegpt_source_attempt_exists:{bundle_root}"
        )
    bundle_root.mkdir(parents=True)
    preflight_path = bundle_root / "conversion_preflight.json"
    preflight_path.write_text(
        json.dumps(source_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    memory_root = bundle_root / "memory"
    stats_path = bundle_root / "source_stats.jsonl"
    stats_summary_path = bundle_root / "source_stats_summary.json"
    audit_path = bundle_root / "trajectory_audit.json"
    started = time.monotonic()
    try:
        generated = convert_runlog_to_mobilegpt_memory(
            source_run_log=source_run_log,
            mobilegpt_root=mobilegpt_root,
            memory_root=memory_root,
            stats_path=stats_path,
            audit_path=audit_path,
            model=normalized_model,
            target_package=str(target_info.get("target_package") or ""),
            target_app=str(target_info.get("target_app") or ""),
        )
    except BaseException as error:
        write_conversion_failure_audit(
            source_run_log=source_run_log,
            stats_path=stats_path,
            audit_path=audit_path,
            error=error,
            wall_sec=time.monotonic() - started,
            target_package=str(target_info.get("target_package") or ""),
            target_app=str(target_info.get("target_app") or ""),
        )
        raise
    wall_sec = round(time.monotonic() - started, 6)
    stats_summary = pipeline.summarize_mobilegpt_stats(stats_path)
    stats_summary_path.write_text(
        json.dumps(stats_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    sealed = pipeline.seal_mobilegpt_source_memory(
        memory_root=memory_root,
        source_run_log=source_run_log,
        source_stats=stats_path,
        trajectory_audit=audit_path,
        task_name=item.task,
        source_seed=SOURCE_SEED,
        target_package=str(target_info.get("target_package") or ""),
        target_app=str(target_info.get("target_app") or ""),
        source_wall_sec=wall_sec,
        source_model=normalized_model,
        memory_schema=MOBILEGPT_DIRECT_MEMORY_SCHEMA,
    )
    result = {
        "schema_version": "omniflow.mobilegpt-source-prepare.v7",
        "task_name": item.task,
        "source_seed": SOURCE_SEED,
        "source_method": MOBILEGPT_DIRECT_SOURCE_METHOD,
        "source_run_log": str(source_run_log),
        "model": normalized_model,
        "memory_root": str(memory_root),
        "learning_mode": MOBILEGPT_DIRECT_LEARNING_MODE,
        "teacher_forcing": False,
        "synthetic_subtasks": True,
        "semantic_subtasks": False,
        "original_mobilegpt_prompts": False,
        "actions_supplied_to_mobilegpt": True,
        "source_transitions_supplied": True,
        "source_success_boundary_supplied": True,
        "function_store_used": False,
        "source_emulator_used": False,
        "source_stats": str(stats_path),
        "source_stats_summary": str(stats_summary_path),
        "trajectory_audit": str(audit_path),
        "source_wall_sec": wall_sec,
        "generated": generated,
        "sealed": sealed,
    }
    if memory_index is not None:
        result["memory_registration"] = _register_mobilegpt_memory(
            memory_index=memory_index,
            bundle_root=bundle_root,
            task_name=item.task,
        )
    return result


def _write_failure_marker(output_root: str | Path, error: BaseException) -> None:
    root = Path(output_root).expanduser().resolve()
    if not root.is_dir() or (root / MOBILEGPT_MEMORY_MANIFEST).exists():
        return
    marker = root / "prep_failure.json"
    if marker.exists():
        return
    stats_path = root / "source_stats.jsonl"
    stats_summary = (
        pipeline.summarize_mobilegpt_stats(stats_path)
        if stats_path.is_file()
        else {}
    )
    audit_path = root / "trajectory_audit.json"
    trajectory_audit: dict[str, Any] = {}
    if audit_path.is_file():
        try:
            loaded = json.loads(audit_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            loaded = {}
        if isinstance(loaded, dict):
            trajectory_audit = {
                key: loaded.get(key)
                for key in (
                    "transition_count",
                    "validated_transition_count",
                    "failure_code",
                    "failure_details",
                    "wall_sec",
                )
            }
    marker.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.mobilegpt-source-failure.v1",
                "failed_at": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
                "stats": stats_summary,
                "trajectory_audit": trajectory_audit,
                "retry_allowed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _selected_source_tasks(
    index_path: str | Path,
    task_names: Sequence[str] = (),
) -> list[str]:
    path = Path(index_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"mobilegpt_source_index_invalid:{path}")
    available = [str(name) for name in payload]
    if not task_names:
        return available
    requested = [str(name).strip() for name in task_names]
    if any(not name for name in requested):
        raise ValueError("mobilegpt_source_task_filter_empty")
    if len(set(requested)) != len(requested):
        raise ValueError("mobilegpt_source_task_filter_duplicate")
    unknown = [name for name in requested if name not in payload]
    if unknown:
        raise ValueError("mobilegpt_source_task_unknown:" + ",".join(unknown))
    return requested


def preflight_mobilegpt_source_batch(
    *,
    index_path: str | Path,
    task_names: Sequence[str] = (),
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for task_name in _selected_source_tasks(index_path, task_names):
        try:
            result = preflight_mobilegpt_source(
                index_path=index_path,
                task_name=task_name,
            )
        except BaseException as error:
            rows.append(
                {
                    "task_name": task_name,
                    "ready": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "failure_code": (
                        error.code
                        if isinstance(error, MobileGPTConversionError)
                        else type(error).__name__
                    ),
                    "failure_details": (
                        dict(error.details)
                        if isinstance(error, MobileGPTConversionError)
                        else {}
                    ),
                }
            )
            continue
        rows.append(
            {
                "task_name": task_name,
                "ready": True,
                "transition_count": int(result["transition_count"]),
                "action_type_counts": dict(result["action_type_counts"]),
                "target_package": str(result["target_package"]),
            }
        )
    ready = sum(row["ready"] is True for row in rows)
    return {
        "schema_version": "omniflow.mobilegpt-source-batch-preflight.v2",
        "planned": len(rows),
        "ready": ready,
        "blocked": len(rows) - ready,
        "model_calls": 0,
        "source_emulator_used": False,
        "rows": rows,
    }


def _batch_task_evidence(task_root: Path) -> dict[str, Any]:
    manifest_path = task_root / MOBILEGPT_MEMORY_MANIFEST
    failure_path = task_root / "prep_failure.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_stats = dict(manifest.get("source_stats") or {})
        return {
            "status": "sealed",
            "manifest": str(manifest_path),
            "schema_version": str(manifest.get("schema_version") or ""),
            "model_calls": int(source_stats.get("model_calls") or 0),
            "prompt_tokens": int(source_stats.get("prompt_tokens") or 0),
            "completion_tokens": int(source_stats.get("completion_tokens") or 0),
            "total_tokens": int(source_stats.get("total_tokens") or 0),
            "task_elapsed_sec": float(source_stats.get("task_elapsed_sec") or 0.0),
            "wall_sec": float(source_stats.get("wall_sec") or 0.0),
            "memory_inventory": dict(
                (manifest.get("memory") or {}).get("inventory") or {}
            ),
        }
    if failure_path.is_file():
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        stats = dict(failure.get("stats") or {})
        trajectory = dict(failure.get("trajectory_audit") or {})
        return {
            "status": "failed",
            "failure": str(failure_path),
            "error_type": str(failure.get("error_type") or ""),
            "error": str(failure.get("error") or ""),
            "model_calls": int(stats.get("model_calls") or 0),
            "prompt_tokens": int(stats.get("prompt_tokens") or 0),
            "completion_tokens": int(stats.get("completion_tokens") or 0),
            "total_tokens": int(stats.get("total_tokens") or 0),
            "task_elapsed_sec": float(stats.get("task_elapsed_sec") or 0.0),
            "wall_sec": float(trajectory.get("wall_sec") or 0.0),
            "transition_count": int(trajectory.get("transition_count") or 0),
            "validated_transition_count": int(
                trajectory.get("validated_transition_count") or 0
            ),
            "failure_code": str(trajectory.get("failure_code") or ""),
            "failure_details": dict(trajectory.get("failure_details") or {}),
        }
    return {"status": "incomplete"}


def _write_batch_report(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare_mobilegpt_source_batch(
    *,
    index_path: str | Path,
    mobilegpt_root: str | Path,
    output_root: str | Path,
    model: str,
    memory_index: str | Path,
    task_names: Sequence[str] = (),
) -> dict[str, Any]:
    from src.experiment.artifact_memory import canonical_mobilegpt_memory_from_memory

    tasks = _selected_source_tasks(index_path, task_names)
    batch_root = Path(output_root).expanduser().resolve()
    batch_root.mkdir(parents=True, exist_ok=True)
    manifest_path = batch_root / "batch_manifest.json"
    expected_manifest = {
        "schema_version": "omniflow.mobilegpt-source-batch.v3",
        "source_memory_schema": MOBILEGPT_DIRECT_MEMORY_SCHEMA,
        "source_method": MOBILEGPT_DIRECT_SOURCE_METHOD,
        "model": str(model),
        "index_path": str(Path(index_path).expanduser().resolve()),
        "memory_index": str(Path(memory_index).expanduser().resolve()),
        "tasks": tasks,
        "model_max_attempts": 1,
        "source_emulator_used": False,
    }
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != expected_manifest:
            raise ValueError(f"mobilegpt_source_batch_manifest_mismatch:{manifest_path}")
    else:
        manifest_path.write_text(
            json.dumps(expected_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    rows: list[dict[str, Any]] = []
    report_path = batch_root / "batch_report.json"
    for ordinal, task_name in enumerate(tasks, start=1):
        canonical = canonical_mobilegpt_memory_from_memory(
            memory_index=memory_index,
            task_name=task_name,
        )
        task_root = batch_root / task_name
        if (
            canonical is not None
            and canonical.get("schema_version") == MOBILEGPT_DIRECT_MEMORY_SCHEMA
            and canonical.get("source_method") == MOBILEGPT_DIRECT_SOURCE_METHOD
        ):
            row = {
                "task_name": task_name,
                "ordinal": ordinal,
                "status": "canonical_skipped",
                "memory_root": str(canonical["memory_root"]),
                "memory_sha256": str(canonical["memory_sha256"]),
            }
        elif task_root.exists():
            evidence = _batch_task_evidence(task_root)
            if evidence["status"] == "sealed":
                validate_mobilegpt_source_memory(
                    index_path=index_path,
                    task_name=task_name,
                    memory_root=task_root / "memory",
                    model=model,
                    memory_index=memory_index,
                )
            elif evidence["status"] == "incomplete":
                _write_failure_marker(
                    task_root,
                    RuntimeError("immutable_mobilegpt_source_attempt_incomplete"),
                )
                evidence = _batch_task_evidence(task_root)
            row = {"task_name": task_name, "ordinal": ordinal, **evidence}
        else:
            try:
                preflight_mobilegpt_source(
                    index_path=index_path,
                    task_name=task_name,
                )
            except BaseException as error:
                task_root.mkdir(parents=True, exist_ok=False)
                _write_failure_marker(task_root, error)
            else:
                try:
                    prepare_mobilegpt_source_memory(
                        index_path=index_path,
                        task_name=task_name,
                        mobilegpt_root=mobilegpt_root,
                        output_root=task_root,
                        model=model,
                        memory_index=memory_index,
                    )
                except BaseException as error:
                    _write_failure_marker(task_root, error)
            row = {
                "task_name": task_name,
                "ordinal": ordinal,
                **_batch_task_evidence(task_root),
            }
        rows.append(row)
        counts = {
            "planned": len(tasks),
            "processed": len(rows),
            "pending": len(tasks) - len(rows),
            "sealed": sum(row["status"] == "sealed" for row in rows),
            "canonical_skipped": sum(
                row["status"] == "canonical_skipped" for row in rows
            ),
            "failed": sum(row["status"] == "failed" for row in rows),
        }
        _write_batch_report(
            report_path,
            {
                "schema_version": "omniflow.mobilegpt-source-batch-report.v3",
                "batch_root": str(batch_root),
                "complete": counts["pending"] == 0,
                "counts": counts,
                "rows": rows,
            },
        )
    return json.loads(report_path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--index", required=True)
    prepare.add_argument("--task", required=True)
    prepare.add_argument("--mobilegpt-root", required=True)
    prepare.add_argument("--output-root", required=True)
    prepare.add_argument("--model", required=True)
    prepare.add_argument("--memory-index", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--index", required=True)
    validate.add_argument("--task", required=True)
    validate.add_argument("--memory-root", required=True)
    validate.add_argument("--model", required=True)
    validate.add_argument("--memory-index", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--index", required=True)
    preflight.add_argument("--task", required=True)
    preflight_batch = subparsers.add_parser("preflight-batch")
    preflight_batch.add_argument("--index", required=True)
    preflight_batch.add_argument("--task", action="append", default=[])
    batch = subparsers.add_parser("batch")
    batch.add_argument("--index", required=True)
    batch.add_argument("--mobilegpt-root", required=True)
    batch.add_argument("--output-root", required=True)
    batch.add_argument("--model", required=True)
    batch.add_argument("--memory-index", required=True)
    batch.add_argument("--task", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "prepare":
            result = prepare_mobilegpt_source_memory(
                index_path=args.index,
                task_name=args.task,
                mobilegpt_root=args.mobilegpt_root,
                output_root=args.output_root,
                model=args.model,
                memory_index=args.memory_index,
            )
        elif args.command == "validate":
            result = validate_mobilegpt_source_memory(
                index_path=args.index,
                task_name=args.task,
                memory_root=args.memory_root,
                model=args.model,
                memory_index=args.memory_index,
            )
        elif args.command == "preflight":
            result = preflight_mobilegpt_source(
                index_path=args.index,
                task_name=args.task,
            )
        elif args.command == "preflight-batch":
            result = preflight_mobilegpt_source_batch(
                index_path=args.index,
                task_names=args.task,
            )
        else:
            result = prepare_mobilegpt_source_batch(
                index_path=args.index,
                mobilegpt_root=args.mobilegpt_root,
                output_root=args.output_root,
                model=args.model,
                memory_index=args.memory_index,
                task_names=args.task,
            )
    except BaseException as error:
        if args.command == "prepare":
            _write_failure_marker(args.output_root, error)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
