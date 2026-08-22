"""Prepare immutable MobileGPT memory from canonical RunLogs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import datetime
import json
from pathlib import Path
import time
from typing import Any

from src.experiment.mobilegpt_contract import (
    MOBILEGPT_EMBEDDING_MODEL,
    MOBILEGPT_LEARNING_MODE,
    MOBILEGPT_MEMORY_MANIFEST,
    MOBILEGPT_MEMORY_SCHEMA,
    MOBILEGPT_SOURCE_METHOD,
)
from src.experiment import run_task as pipeline
from src.experiment.paths import sha256_file
from src.experiment.source_records import CanonicalRunLog
from src.experiment.protocol import SOURCE_SEED
from src.integrations.mobilegpt import (
    MobileGPTConversionError,
    convert_runlog_to_mobilegpt_memory,
    preflight_runlog_conversion,
    validate_prepared_memory,
    write_conversion_failure_audit,
)
from src.integrations import mobilegpt_memory
from src.integrations.runlog import import_run_log

_IGNORED_SOURCE_PACKAGES = {
    "com.android.systemui",
    "com.example.MobileGPT",
    "com.google.android.apps.nexuslauncher",
}


def load_canonical_source_item(
    index_path: str | Path,
    *,
    task_name: str,
) -> CanonicalRunLog:
    matches = [
        item
        for item in pipeline.load_canonical_source_index(index_path)
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
    if source_kind and source_kind not in {
        "androidworld_validator_success_source_runlog",
        "one_time_canonicalized_seed111_screenshot_source",
    }:
        raise ValueError(
            "mobilegpt_source_kind_invalid:"
            f"task={task_name}:actual={source_kind}"
        )
    if not item.source_run_log.is_file():
        raise FileNotFoundError(
            f"mobilegpt_source_runlog_missing:{item.source_run_log}"
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
    item: CanonicalRunLog,
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
        package = pipeline._mobilegpt_observation_package(observation)
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
    item: CanonicalRunLog,
) -> tuple[Path, tuple[str, ...], dict[str, Any], dict[str, str]]:
    source_run_log = item.source_run_log
    source_sha256 = sha256_file(source_run_log)
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
        "schema_version": "omniflow.mobilegpt.source-check.v2",
        "grounding_source": "canonical_androidworld_run_log",
        "source_run_log": str(source_run_log),
        "source_run_log_sha256": source_sha256,
        "actions_supplied_to_mobilegpt": False,
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
        "schema_version": "omniflow.mobilegpt.source-check.v2",
        "task_name": item.task,
        "source_seed": SOURCE_SEED,
        "source_method": MOBILEGPT_SOURCE_METHOD,
        "source_run_log": source_audit["source_run_log"],
        "source_run_log_sha256": source_audit["source_run_log_sha256"],
        "learning_mode": MOBILEGPT_LEARNING_MODE,
        "teacher_forcing": False,
        "synthetic_subtasks": False,
        "semantic_subtasks": True,
        "original_mobilegpt_prompts": True,
        "actions_supplied_to_mobilegpt": False,
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
    if schema_version != MOBILEGPT_MEMORY_SCHEMA:
        raise ValueError("mobilegpt_source_memory_schema_invalid")
    validated = validate_prepared_memory(
        memory_root,
        task_name=item.task,
        source_seed=SOURCE_SEED,
        source_run_log=source_run_log,
        compatible_source_sha256s=compatible_sha256s,
        expected_model=str(model),
        expected_source_method=MOBILEGPT_SOURCE_METHOD,
    )
    result = {
        "schema_version": "omniflow.mobilegpt.memory-check.v2",
        "task_name": item.task,
        "source_seed": SOURCE_SEED,
        "source_method": MOBILEGPT_SOURCE_METHOD,
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
    from src.experiment.data_index import refresh_data_index_from_pointer

    report = refresh_data_index_from_pointer(
        memory_index=memory_index,
        additional_prepared_memory_roots=(bundle_root,),
    )
    registered = report.get("canonical", {}).get("prepared_memories", {}).get(
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
    embedding_model: str = MOBILEGPT_EMBEDDING_MODEL,
    memory_index: str | Path | None = None,
) -> dict[str, Any]:
    normalized_model = str(model or "").strip()
    if not normalized_model:
        raise ValueError("mobilegpt_source_model_required")
    normalized_embedding_model = (
        str(embedding_model or "").strip() or MOBILEGPT_EMBEDDING_MODEL
    )
    item = load_canonical_source_item(index_path, task_name=task_name)
    source_run_log, _, source_audit, target_info = _source_preflight(item)
    result = convert_runlog_to_mobilegpt_bundle(
        source_run_log=source_run_log,
        mobilegpt_root=mobilegpt_root,
        output_root=output_root,
        model=normalized_model,
        embedding_model=normalized_embedding_model,
        target_package=str(target_info.get("target_package") or ""),
        target_app=str(target_info.get("target_app") or ""),
        preflight_audit=source_audit,
    )
    result.update(
        {
            "schema_version": "omniflow.mobilegpt.memory-prepare.v2",
            "source_method": MOBILEGPT_SOURCE_METHOD,
            "learning_mode": MOBILEGPT_LEARNING_MODE,
            "teacher_forcing": False,
            "synthetic_subtasks": False,
            "semantic_subtasks": True,
            "original_mobilegpt_prompts": True,
            "actions_supplied_to_mobilegpt": False,
            "source_transitions_supplied": True,
            "source_success_boundary_supplied": True,
            "function_store_used": False,
            "source_emulator_used": False,
        }
    )
    bundle_root = Path(output_root).expanduser().resolve()
    if memory_index is not None:
        result["memory_registration"] = _register_mobilegpt_memory(
            memory_index=memory_index,
            bundle_root=bundle_root,
            task_name=item.task,
        )
    return result


def convert_runlog_to_mobilegpt_bundle(
    *,
    source_run_log: str | Path,
    mobilegpt_root: str | Path,
    output_root: str | Path,
    model: str,
    embedding_model: str = MOBILEGPT_EMBEDDING_MODEL,
    target_package: str = "",
    target_app: str = "",
    preflight_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert one valid RunLog and seal one native MobileGPT bundle."""

    normalized_model = str(model or "").strip()
    if not normalized_model:
        raise ValueError("mobilegpt_source_model_required")
    normalized_embedding_model = (
        str(embedding_model or "").strip() or MOBILEGPT_EMBEDDING_MODEL
    )
    source_path = Path(source_run_log).expanduser().resolve()
    source = import_run_log(json.loads(source_path.read_text(encoding="utf-8")))
    if (
        source.get("status") != "succeeded"
        or source.get("success") is not True
        or (source.get("validator") or {}).get("official") is not True
        or (source.get("validator") or {}).get("success") is not True
    ):
        raise ValueError("mobilegpt_source_runlog_not_successful")
    report = preflight_runlog_conversion(
        source_path,
        target_package=target_package,
        target_app=target_app,
    )
    if report.get("ready") is not True:
        raise MobileGPTConversionError(
            str(report.get("failure_code") or "mobilegpt_conversion_preflight_failed"),
            **dict(report.get("failure_details") or {}),
        )
    source_audit = preflight_audit or {
        "schema_version": "omniflow.mobilegpt.source-check.v2",
        "grounding_source": "canonical_androidworld_run_log",
        "source_run_log": str(source_path),
        "source_run_log_sha256": sha256_file(source_path),
        "actions_supplied_to_mobilegpt": False,
        "function_store_used": False,
        "report": report,
    }
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
            source_run_log=source_path,
            mobilegpt_root=mobilegpt_root,
            memory_root=memory_root,
            stats_path=stats_path,
            audit_path=audit_path,
            model=normalized_model,
            embedding_model=normalized_embedding_model,
            target_package=str(target_package or ""),
            target_app=str(target_app or ""),
        )
    except BaseException as error:
        write_conversion_failure_audit(
            source_run_log=source_path,
            stats_path=stats_path,
            audit_path=audit_path,
            error=error,
            wall_sec=time.monotonic() - started,
            target_package=str(target_package or ""),
            target_app=str(target_app or ""),
            conversion_mode="official_mobilegpt_learning",
        )
        raise
    wall_sec = round(time.monotonic() - started, 6)
    stats_summary = mobilegpt_memory.summarize_mobilegpt_stats(stats_path)
    stats_summary_path.write_text(
        json.dumps(stats_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    sealed = pipeline.seal_mobilegpt_source_memory(
        memory_root=memory_root,
        source_run_log=source_path,
        source_stats=stats_path,
        trajectory_audit=audit_path,
        task_name=str(source["task_name"]),
        source_seed=SOURCE_SEED,
        target_package=str(target_package or generated.get("target_package") or ""),
        target_app=str(target_app or generated.get("target_app") or ""),
        source_wall_sec=wall_sec,
        source_model=normalized_model,
        memory_schema=MOBILEGPT_MEMORY_SCHEMA,
    )
    return {
        "schema_version": "omniflow.mobilegpt.memory-prepare.v2",
        "method": "mobilegpt",
        "task_name": str(source["task_name"]),
        "source_seed": SOURCE_SEED,
        "source_run_log": str(source_path),
        "model": normalized_model,
        "embedding_model": normalized_embedding_model,
        "memory_root": str(memory_root),
        "source_stats": str(stats_path),
        "source_stats_summary": str(stats_summary_path),
        "trajectory_audit": str(audit_path),
        "source_wall_sec": wall_sec,
        "generated": generated,
        "sealed": sealed,
        "manifest": sealed,
    }


def _write_failure_marker(output_root: str | Path, error: BaseException) -> None:
    root = Path(output_root).expanduser().resolve()
    if not root.is_dir() or (root / MOBILEGPT_MEMORY_MANIFEST).exists():
        return
    marker = root / "prep_failure.json"
    if marker.exists():
        return
    stats_path = root / "source_stats.jsonl"
    stats_summary = (
        mobilegpt_memory.summarize_mobilegpt_stats(stats_path)
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
                "schema_version": "omniflow.mobilegpt.memory-failure.v2",
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
        else:
            raise ValueError(f"unsupported_mobilegpt_source_command:{args.command}")
    except BaseException as error:
        if args.command == "prepare":
            _write_failure_marker(args.output_root, error)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
