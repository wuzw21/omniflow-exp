"""Prepare one immutable MobileGPT source memory from a canonical RunLog."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
import datetime
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from src.experiment import androidworld as pipeline
from src.experiment.source_assets import (
    build_grounded_teacher_run_log_from_item,
)
from src.integrations.runlog import import_run_log

SOURCE_SEED = 111
_IGNORED_SOURCE_PACKAGES = {
    "com.android.systemui",
    "com.example.MobileGPT",
    "com.google.android.apps.nexuslauncher",
}


def source_method_label(item: pipeline.ArchivedRunLog) -> str:
    """Preserve recorded provenance without inventing a generating method."""

    return str(item.meta.get("method") or "").strip() or "unrecorded"


def load_canonical_source_item(
    index_path: str | Path,
    *,
    task_name: str,
) -> pipeline.ArchivedRunLog:
    """Resolve and audit the frozen source shared by every comparison method."""

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
    if item.replay_seed != SOURCE_SEED:
        raise ValueError(
            "mobilegpt_source_seed_mismatch:"
            f"task={task_name}:expected={SOURCE_SEED}:actual={item.replay_seed}"
        )
    if item.meta.get("latest_official_success_source") is not True:
        raise ValueError(
            f"mobilegpt_source_official_success_required:task={task_name}"
        )
    if (
        source_kind
        and source_kind != "androidworld_validator_success_source_runlog"
    ):
        raise ValueError(
            "mobilegpt_source_kind_invalid:"
            f"task={task_name}:actual={source_kind or 'missing'}"
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


def validate_mobilegpt_source_memory(
    *,
    index_path: str | Path,
    task_name: str,
    memory_root: str | Path,
    model: str,
) -> dict[str, Any]:
    item = load_canonical_source_item(index_path, task_name=task_name)
    source_method = source_method_label(item)
    validated = pipeline.validate_mobilegpt_adapted_memory(
        memory_root,
        task_name=item.task,
        source_seed=SOURCE_SEED,
        source_run_log=item.source_run_log,
        expected_model=str(model),
        expected_source_method=source_method,
    )
    return {
        "schema_version": "omniflow.mobilegpt-source-validation.v1",
        "task_name": item.task,
        "source_seed": SOURCE_SEED,
        "source_method": source_method,
        "source_run_log": str(item.source_run_log),
        "model": str(model),
        "validated": validated,
    }


def _grounded_source_payload(
    *,
    index_path: str | Path,
    item: pipeline.ArchivedRunLog,
    store_index_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return build_grounded_teacher_run_log_from_item(
        index_path=index_path,
        item=item,
        store_index_path=store_index_path,
    )


def _preflight_mobilegpt_teacher(
    *,
    index_path: str | Path,
    item: pipeline.ArchivedRunLog,
    store_index_path: str | Path | None = None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
]:
    grounded, grounding_audit = _grounded_source_payload(
        index_path=index_path,
        item=item,
        store_index_path=store_index_path,
    )
    with tempfile.TemporaryDirectory(
        prefix="omniflow-mobilegpt-preflight-"
    ) as temporary:
        grounded_path = Path(temporary) / "grounded.run_log.json"
        grounded_path.write_text(
            json.dumps(grounded, ensure_ascii=False),
            encoding="utf-8",
        )
        teacher_payload = pipeline.build_mobilegpt_teacher_source(
            grounded_path,
            task_name=item.task,
            source_seed=SOURCE_SEED,
            provenance_source_run_log=item.source_run_log,
        )
    target_info = _mobilegpt_source_target(item=item, grounded=grounded)
    return grounded, grounding_audit, teacher_payload, target_info


def _mobilegpt_source_target(
    *,
    item: pipeline.ArchivedRunLog,
    grounded: dict[str, Any],
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
    for step in grounded.get("steps") or []:
        observation = (
            step.get("observation_before_act")
            if isinstance(step, dict)
            else None
        )
        package = str(
            observation.get("package_name")
            if isinstance(observation, dict)
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
        "target_source": "frozen_source_states",
    }


def preflight_mobilegpt_source(
    *,
    index_path: str | Path,
    task_name: str,
    store_index_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate one source asset without creating a persistent output."""

    item = load_canonical_source_item(index_path, task_name=task_name)
    _, grounding_audit, teacher_payload, target_info = (
        _preflight_mobilegpt_teacher(
        index_path=index_path,
        item=item,
        store_index_path=store_index_path,
        )
    )
    return {
        "schema_version": "omniflow.mobilegpt-source-preflight.v1",
        "task_name": item.task,
        "source_seed": SOURCE_SEED,
        "source_method": source_method_label(item),
        "source_run_log": str(item.source_run_log),
        "action_count": int(teacher_payload["action_count"]),
        "target_package": target_info["target_package"],
        "target_source": target_info["target_source"],
        "grounding": grounding_audit,
        "ready": True,
    }


def prepare_mobilegpt_source_memory(
    *,
    index_path: str | Path,
    task_name: str,
    mobilegpt_root: str | Path,
    android_world_root: str | Path,
    output_root: str | Path,
    model: str,
    store_index_path: str | Path | None = None,
    serial: str = "emulator-5560",
    console_port: int = 5560,
    adb_path: str = "",
    max_steps: int = 20,
    server_host: str = "0.0.0.0",
    port: int = 12345,
    server_warmup_sec: float = 5.0,
    wait_start_timeout_sec: float = 60.0,
    wait_finish_timeout_sec: float = 180.0,
    perform_emulator_setup: bool = True,
) -> dict[str, Any]:
    """Run one source episode, seal its memory, and never retry it."""

    normalized_model = str(model or "").strip()
    if not normalized_model:
        raise ValueError("mobilegpt_source_model_required")
    item = load_canonical_source_item(index_path, task_name=task_name)
    source_method = source_method_label(item)
    bundle_root = Path(output_root).expanduser().resolve()
    if bundle_root.exists():
        raise FileExistsError(
            f"immutable_mobilegpt_source_attempt_exists:{bundle_root}"
        )
    grounded_payload, grounding_audit, teacher_payload, target_info = (
        _preflight_mobilegpt_teacher(
            index_path=index_path,
            item=item,
            store_index_path=store_index_path,
        )
    )

    target_package = str(target_info.get("target_package") or "").strip()
    if not target_package:
        raise ValueError("mobilegpt_source_target_package_unresolved")
    target_app = str(target_info.get("target_app") or target_package).strip()

    bundle_root.mkdir(parents=True)
    grounded_source_path = bundle_root / "grounded_teacher_run_log.json"
    grounded_source_path.write_text(
        json.dumps(grounded_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    teacher_source_path = bundle_root / "teacher_source.json"
    teacher_source_path.write_text(
        json.dumps(teacher_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    memory_root = bundle_root / "memory"
    memory_root.mkdir()
    stats_path = bundle_root / "source_stats.jsonl"
    stats_summary_path = bundle_root / "source_stats_summary.json"
    runtime_serial_file = bundle_root / "active_serial.txt"
    pipeline._write_mobilegpt_runtime_serial(runtime_serial_file, serial)

    target = pipeline.DeviceTarget(
        label=f"source{int(console_port)}",
        serial=str(serial),
        console_port=int(console_port),
    )

    for patched in pipeline._patch_mobilegpt_stats(
        mobilegpt_root=mobilegpt_root,
    ):
        print(f"[mobilegpt:patch-stats] {patched}", flush=True)
    for patched in pipeline._patch_mobilegpt_server_runtime_context(
        mobilegpt_root=mobilegpt_root,
    ):
        print(f"[mobilegpt:patch-server-runtime] {patched}", flush=True)

    server_spec = pipeline.build_mobilegpt_command(
        "teach-server",
        mobilegpt_root=mobilegpt_root,
        mobilegpt_memory_root=memory_root,
        serial=serial,
        adb_path=adb_path,
        server_host=server_host,
        port=int(port),
        stats_jsonl=stats_path,
        source_run_log=grounded_source_path,
        fallback_to_vlm_on_teacher_miss=False,
        target_package=target_package,
        target_app=target_app,
        runtime_serial_file=runtime_serial_file,
    )
    server_spec = replace(
        server_spec,
        env={
            **server_spec.env,
            "MOBILEGPT_CHAT_MODEL": normalized_model,
            "MOBILEGPT_CHAT_MAX_ATTEMPTS": "1",
            "MOBILEGPT_OOB_OBSERVE_RETRIES": "1",
            "MOBILEGPT_TEACHER_FAIL_ON_ACTION_ERROR": "1",
        },
        metadata={
            **server_spec.metadata,
            "model": normalized_model,
            "model_max_attempts": 1,
            "episode_retries": 0,
            "source_method": source_method,
        },
    )

    browser_prepare, browser_server = pipeline._start_mobilegpt_browser_task_server(
        item=item,
        memory_root=bundle_root,
        android_world_root=android_world_root,
        task_params_override=dict(item.params),
        dry_run=False,
    )
    browser_url = str(browser_prepare.get("url") or "").strip()
    if browser_url:
        server_spec = replace(
            server_spec,
            env={
                **server_spec.env,
                "OMNIFLOW_MOBILEGPT_BROWSER_TASK_URL": browser_url,
                "OMNIFLOW_MOBILEGPT_ADB_PATH": str(adb_path or "").strip(),
            },
            metadata={
                **server_spec.metadata,
                "browser_task_url": browser_url,
                "browser_task_serial": target.serial,
            },
        )

    episode_spec = pipeline.build_mobilegpt_androidworld_command(
        item,
        method_name="mobilegpt_native_source_cold",
        target=target,
        android_world_root=android_world_root,
        output_root=bundle_root / "_source_episode",
        stats_jsonl=stats_path,
        runtime_serial_file=runtime_serial_file,
        max_steps=max(int(max_steps), int(teacher_payload["action_count"]) + 3),
        task_random_seed=SOURCE_SEED,
        fixed_task_seed=True,
        fixed_task_params=True,
        task_params_override=dict(item.params),
        perform_emulator_setup=bool(perform_emulator_setup),
        adb_path=str(adb_path),
        start_timeout_sec=float(wait_start_timeout_sec),
        finish_timeout_sec=float(wait_finish_timeout_sec),
        rebroadcast_limit=0,
    )
    episode_spec.metadata.update(
        {
            "model": normalized_model,
            "source_method": source_method,
            "prep_type": "mobilegpt_native_teacher_source_memory",
            "rebroadcast_limit": 0,
        }
    )
    command_manifest_path = bundle_root / "source_episode_command.json"
    command_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.mobilegpt-source-command.v2",
                "source_seed": SOURCE_SEED,
                "source_method": source_method,
                "task_name": item.task,
                "task_params": item.params,
                "serial": str(serial),
                "model": normalized_model,
                "model_max_attempts": 1,
                "episode_retries": 0,
                "server_command": pipeline._command_line(server_spec),
                "episode_command": pipeline._command_line(episode_spec),
                "source_run_log": str(item.source_run_log),
                "source_run_log_sha256": pipeline._file_sha256(
                    item.source_run_log
                ),
                "teacher_source": str(teacher_source_path),
                "teacher_source_sha256": pipeline._file_sha256(
                    teacher_source_path
                ),
                "grounded_teacher_run_log": str(grounded_source_path),
                "grounded_teacher_run_log_sha256": pipeline._file_sha256(
                    grounded_source_path
                ),
                "grounding_audit": grounding_audit,
                "target_inputs_read": False,
                "target_observations_read": False,
                "validator_state_read": False,
                "function_conversion_enabled": False,
                "coordinate_replay": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    server = None
    started = time.monotonic()
    try:
        server, server_returncode = pipeline._start_background_command(
            server_spec,
            warmup_sec=float(server_warmup_sec),
        )
        if server_returncode != 0:
            raise RuntimeError(
                f"mobilegpt_teacher_server_failed:{server_returncode}"
            )
        episode_returncode = pipeline.run_command(episode_spec)
        if episode_returncode != 0:
            raise RuntimeError(
                f"mobilegpt_source_episode_failed:{episode_returncode}"
            )
    finally:
        pipeline._stop_background_command(server)
        pipeline._stop_background_command(browser_server)
    wall_sec = round(time.monotonic() - started, 6)

    if episode_spec.output_path is None:
        raise RuntimeError("mobilegpt_source_episode_output_missing")
    result_path = episode_spec.output_path / "task_results.jsonl"
    stats_summary = pipeline.summarize_mobilegpt_stats(stats_path)
    stats_summary_path.write_text(
        json.dumps(stats_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    actual_models = {
        str(value or "").strip()
        for value in stats_summary.get("chat_models") or []
        if str(value or "").strip()
    }
    if actual_models != {normalized_model}:
        raise ValueError(
            "mobilegpt_source_model_mismatch:"
            f"expected={normalized_model}:actual={sorted(actual_models)}"
        )

    sealed = pipeline.seal_mobilegpt_adapted_memory(
        memory_root=memory_root,
        teacher_source=teacher_source_path,
        source_run_log=item.source_run_log,
        source_stats=stats_path,
        official_source_result=result_path,
        task_name=item.task,
        source_seed=SOURCE_SEED,
        target_package=target_package,
        target_app=target_app,
        source_wall_sec=wall_sec,
        source_method=source_method,
        source_model=normalized_model,
    )
    return {
        "schema_version": "omniflow.mobilegpt-source-prepare.v2",
        "task_name": item.task,
        "source_seed": SOURCE_SEED,
        "source_method": source_method,
        "source_run_log": str(item.source_run_log),
        "model": normalized_model,
        "memory_root": str(memory_root),
        "teacher_source": str(teacher_source_path),
        "source_stats": str(stats_path),
        "source_stats_summary": str(stats_summary_path),
        "official_source_result": str(result_path),
        "source_wall_sec": wall_sec,
        "sealed": sealed,
    }


def _write_failure_marker(output_root: str | Path, error: BaseException) -> None:
    root = Path(output_root).expanduser().resolve()
    if not root.is_dir() or (root / "cold_memory_manifest.json").exists():
        return
    marker = root / "prep_failure.json"
    if marker.exists():
        return
    marker.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.mobilegpt-source-failure.v1",
                "failed_at": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
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
    prepare.add_argument("--android-world-root", required=True)
    prepare.add_argument("--output-root", required=True)
    prepare.add_argument("--model", required=True)
    prepare.add_argument("--store-index", default="")
    prepare.add_argument("--serial", default="emulator-5560")
    prepare.add_argument("--console-port", type=int, default=5560)
    prepare.add_argument("--adb-path", default="")
    prepare.add_argument("--max-steps", type=int, default=20)
    prepare.add_argument("--server-host", default="0.0.0.0")
    prepare.add_argument("--port", type=int, default=12345)
    prepare.add_argument("--server-warmup-sec", type=float, default=5.0)
    prepare.add_argument("--wait-start-timeout-sec", type=float, default=60.0)
    prepare.add_argument("--wait-finish-timeout-sec", type=float, default=180.0)
    prepare.add_argument("--no-emulator-setup", action="store_true")

    validate = subparsers.add_parser("validate")
    validate.add_argument("--index", required=True)
    validate.add_argument("--task", required=True)
    validate.add_argument("--memory-root", required=True)
    validate.add_argument("--model", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--index", required=True)
    preflight.add_argument("--task", required=True)
    preflight.add_argument("--store-index", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "prepare":
            result = prepare_mobilegpt_source_memory(
                index_path=args.index,
                task_name=args.task,
                mobilegpt_root=args.mobilegpt_root,
                android_world_root=args.android_world_root,
                output_root=args.output_root,
                model=args.model,
                store_index_path=args.store_index or None,
                serial=args.serial,
                console_port=args.console_port,
                adb_path=args.adb_path,
                max_steps=args.max_steps,
                server_host=args.server_host,
                port=args.port,
                server_warmup_sec=args.server_warmup_sec,
                wait_start_timeout_sec=args.wait_start_timeout_sec,
                wait_finish_timeout_sec=args.wait_finish_timeout_sec,
                perform_emulator_setup=not args.no_emulator_setup,
            )
        elif args.command == "validate":
            result = validate_mobilegpt_source_memory(
                index_path=args.index,
                task_name=args.task,
                memory_root=args.memory_root,
                model=args.model,
            )
        else:
            result = preflight_mobilegpt_source(
                index_path=args.index,
                task_name=args.task,
                store_index_path=args.store_index or None,
            )
    except BaseException as error:
        if args.command == "prepare":
            _write_failure_marker(args.output_root, error)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
