"""Build reusable memory with MobileGPT's original source-device cold start."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
import datetime
import json
import os
from pathlib import Path
import time
from typing import Any

from src.experiment import run_task as pipeline
from src.experiment.mobilegpt_contract import (
    MOBILEGPT_EMBEDDING_MODEL,
    MOBILEGPT_LEARNING_MODE,
    MOBILEGPT_MEMORY_MANIFEST,
    MOBILEGPT_SOURCE_METHOD,
)
from src.experiment.paths import sha256_file
from src.experiment.protocol import SOURCE_SEED
from src.experiment.source_records import CanonicalRunLog
from src.integrations import mobilegpt_memory
from src.integrations.runlog import adapt_source_run_log, import_run_log

_IGNORED_SOURCE_PACKAGES = {
    "com.android.systemui",
    "com.example.MobileGPT",
    "com.android.documentsui",
    "com.android.permissioncontroller",
    "com.google.android.documentsui",
    "com.google.android.apps.nexuslauncher",
    "com.google.android.permissioncontroller",
    "com.google.android.inputmethod.latin",
    "com.android.inputmethod.latin",
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
    if item.meta.get("latest_official_success_source") is not True:
        raise ValueError(f"mobilegpt_source_official_success_required:task={task_name}")
    if not item.source_run_log.is_file():
        raise FileNotFoundError(f"mobilegpt_source_runlog_missing:{item.source_run_log}")
    source = _load_mobilegpt_source_payload(item)
    if (
        source.get("status") != "succeeded"
        or source.get("success") is not True
        or not source.get("steps")
    ):
        raise ValueError(f"mobilegpt_source_runlog_not_successful:task={task_name}")
    return item


def _load_mobilegpt_source_payload(item: CanonicalRunLog) -> dict[str, Any]:
    raw = json.loads(item.source_run_log.read_text(encoding="utf-8"))
    steps = raw.get("steps")
    legacy_steps = isinstance(steps, list) and any(
        isinstance(step, dict)
        and any(
            key in step
            for key in ("before_state_id", "after_state_id", "observation_before_act")
        )
        for step in steps
    )
    if raw.get("schema_version") == "omniflow.run_log.v1" and not legacy_steps:
        return import_run_log(raw)
    return adapt_source_run_log(
        raw,
        task_name=item.task,
        task_parameters=dict(item.params),
        seed=int(item.replay_seed),
        source_path=item.source_run_log,
        screenshot_roots=(
            item.source_run_log.parent / "observations" / "objects",
            item.source_run_log.parent,
        ),
        require_screenshots=True,
    )


def _mobilegpt_source_target(
    *,
    item: CanonicalRunLog,
    source: dict[str, Any],
) -> dict[str, str]:
    inferred = pipeline._infer_mobilegpt_target_from_source_run_log(item)
    inferred_package = str(inferred.get("target_package") or "").strip()
    if inferred_package:
        return {
            "target_package": inferred_package,
            "target_app": str(inferred.get("target_app") or inferred_package),
            "target_source": (
                "canonical_source_package"
                if "." in inferred_package
                else "canonical_source_open_app_alias"
            ),
        }

    packages: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            package = str(
                value.get("package_name") or value.get("packageName") or ""
            ).strip()
            if package and package not in _IGNORED_SOURCE_PACKAGES:
                packages.add(package)
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    for step in source.get("steps") or []:
        if not isinstance(step, dict):
            continue
        collect(step.get("observation"))
        collect(step.get("next_observation"))
    collect(source.get("final_observation"))
    if len(packages) != 1:
        label = "unresolved" if not packages else "ambiguous"
        raise ValueError(
            f"mobilegpt_source_target_package_{label}:"
            + ",".join(sorted(packages))
        )
    package = next(iter(packages))
    return {
        "target_package": package,
        "target_app": package,
        "target_source": "canonical_source_observation",
    }


def _source_preflight(
    item: CanonicalRunLog,
) -> tuple[Path, tuple[str, ...], dict[str, Any], dict[str, str]]:
    source = _load_mobilegpt_source_payload(item)
    target = _mobilegpt_source_target(item=item, source=source)
    source_path = item.source_run_log.resolve()
    source_sha256 = sha256_file(source_path)
    audit = {
        "schema_version": "omniflow.mobilegpt-native-source-check.v1",
        "task_name": item.task,
        "source_run_log": str(source_path),
        "source_run_log_sha256": source_sha256,
        "learning_mode": MOBILEGPT_LEARNING_MODE,
        "teacher_forcing": False,
        "actions_supplied_to_mobilegpt": False,
        "runlog_conversion_used": False,
        "source_emulator_required": True,
        "target_package": target["target_package"],
    }
    return source_path, (source_sha256,), audit, target


def preflight_mobilegpt_source(
    *,
    index_path: str | Path,
    task_name: str,
) -> dict[str, Any]:
    item = load_canonical_source_item(index_path, task_name=task_name)
    source_path, _, audit, target = _source_preflight(item)
    return {
        "schema_version": "omniflow.mobilegpt-native-source-preflight.v1",
        "task_name": item.task,
        "source_seed": SOURCE_SEED,
        "source_method": MOBILEGPT_SOURCE_METHOD,
        "source_run_log": str(source_path),
        "learning_mode": MOBILEGPT_LEARNING_MODE,
        "teacher_forcing": False,
        "actions_supplied_to_mobilegpt": False,
        "runlog_conversion_used": False,
        "source_emulator_required": True,
        "target_package": target["target_package"],
        "source_audit": audit,
        "ready": True,
    }


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


def validate_mobilegpt_source_memory(
    *,
    index_path: str | Path,
    task_name: str,
    memory_root: str | Path,
    model: str,
    memory_index: str | Path | None = None,
) -> dict[str, Any]:
    item = load_canonical_source_item(index_path, task_name=task_name)
    source_path, compatible_sha256s, _, _ = _source_preflight(item)
    validated = mobilegpt_memory.validate_mobilegpt_adapted_memory(
        memory_root,
        task_name=item.task,
        source_seed=SOURCE_SEED,
        source_run_log=source_path,
        compatible_source_sha256s=compatible_sha256s,
        expected_model=str(model or "").strip(),
        expected_source_method=MOBILEGPT_SOURCE_METHOD,
    )
    result: dict[str, Any] = {
        "schema_version": "omniflow.mobilegpt-native-memory-check.v1",
        "task_name": item.task,
        "source_seed": SOURCE_SEED,
        "source_method": MOBILEGPT_SOURCE_METHOD,
        "model": str(model or "").strip(),
        "validated": validated,
    }
    if memory_index is not None:
        result["memory_registration"] = _register_mobilegpt_memory(
            memory_index=memory_index,
            bundle_root=Path(memory_root).expanduser().resolve().parent,
            task_name=item.task,
        )
    return result


def prepare_mobilegpt_source_memory(
    *,
    index_path: str | Path,
    task_name: str,
    mobilegpt_root: str | Path,
    android_world_root: str | Path,
    output_root: str | Path,
    model: str,
    embedding_model: str = MOBILEGPT_EMBEDDING_MODEL,
    memory_index: str | Path | None = None,
    serial: str = "emulator-5560",
    console_port: int = 5560,
    adb_path: str = "",
    max_steps: int = 20,
    server_host: str = "0.0.0.0",
    port: int = 12345,
    server_warmup_sec: float = 5.0,
    wait_start_timeout_sec: float = 60.0,
    wait_finish_timeout_sec: float = 600.0,
    timeout_sec: float = 900.0,
    perform_emulator_setup: bool = True,
) -> dict[str, Any]:
    """Run one unassisted official cold episode and seal only validator success."""

    normalized_model = str(model or "").strip()
    if not normalized_model:
        raise ValueError("mobilegpt_source_model_required")
    normalized_embedding_model = (
        str(embedding_model or "").strip() or MOBILEGPT_EMBEDDING_MODEL
    )
    item = load_canonical_source_item(index_path, task_name=task_name)
    source_path, _, source_audit, target = _source_preflight(item)
    resolved_target_package = pipeline._resolve_mobilegpt_target_package(
        target["target_package"],
        adb_path=str(adb_path),
        serial=str(serial),
    )
    if "." not in resolved_target_package:
        raise ValueError(
            "mobilegpt_source_target_package_unresolved:"
            f"alias={target['target_package']}:resolved={resolved_target_package}"
        )
    target = {
        **target,
        "target_package": resolved_target_package,
    }
    bundle_root = Path(output_root).expanduser().resolve()
    if bundle_root.exists():
        raise FileExistsError(f"immutable_mobilegpt_source_attempt_exists:{bundle_root}")
    bundle_root.mkdir(parents=True)
    memory_root = bundle_root / "memory"
    memory_root.mkdir()
    stats_path = bundle_root / "source_stats.jsonl"
    stats_summary_path = bundle_root / "source_stats_summary.json"
    target_device = pipeline.DeviceTarget(
        label=f"source{int(console_port)}",
        serial=str(serial),
        console_port=int(console_port),
    )

    server_spec = pipeline.build_mobilegpt_server_command(
        "server",
        mobilegpt_root=mobilegpt_root,
        mobilegpt_memory_root=memory_root,
        embedding_model=normalized_embedding_model,
        write_through_memory=True,
        serial=serial,
        adb_path=adb_path,
        server_host=server_host,
        port=int(port),
        stats_jsonl=stats_path,
        target_package=target["target_package"],
        target_app=target["target_app"],
        target_task_name=item.task,
    )
    server_spec = pipeline._configure_mobilegpt_formal_server(
        server_spec,
        model=normalized_model,
    )
    server_spec = replace(
        server_spec,
        env={
            **server_spec.env,
            "MOBILEGPT_TEACHER_RUNLOG": "",
            "MOBILEGPT_TEACHER_ARTIFACT_DIR": "",
            "MOBILEGPT_TEACHER_FALLBACK_TO_VLM_ON_MISS": "",
        },
        metadata={
            **server_spec.metadata,
            "source_method": MOBILEGPT_SOURCE_METHOD,
            "learning_mode": MOBILEGPT_LEARNING_MODE,
            "teacher_forcing": False,
            "runlog_conversion_used": False,
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
        )

    episode_spec = pipeline.build_mobilegpt_command(
        item,
        method_name=MOBILEGPT_SOURCE_METHOD,
        target=target_device,
        android_world_root=android_world_root,
        output_root=bundle_root / "_source_episode",
        stats_jsonl=stats_path,
        mobilegpt_root=mobilegpt_root,
        server_host=server_host,
        server_port=int(port),
        target_package=target["target_package"],
        max_steps=int(max_steps),
        task_random_seed=SOURCE_SEED,
        fixed_task_seed=True,
        fixed_task_params=True,
        task_params_override=dict(item.params),
        perform_emulator_setup=bool(perform_emulator_setup),
        adb_path=str(adb_path),
        start_timeout_sec=float(wait_start_timeout_sec),
        finish_timeout_sec=float(wait_finish_timeout_sec),
        timeout_sec=float(timeout_sec),
        server_log_path=str(server_spec.metadata.get("log_path") or ""),
    )
    episode_spec.metadata.update(
        {
            "source_method": MOBILEGPT_SOURCE_METHOD,
            "prep_type": "mobilegpt_native_source_cold_memory",
            "model": normalized_model,
        }
    )
    (bundle_root / "source_episode_command.json").write_text(
        json.dumps(
            {
                "schema_version": "omniflow.mobilegpt-native-source-command.v1",
                "task_name": item.task,
                "source_seed": SOURCE_SEED,
                "source_method": MOBILEGPT_SOURCE_METHOD,
                "learning_mode": MOBILEGPT_LEARNING_MODE,
                "serial": str(serial),
                "model": normalized_model,
                "embedding_model": normalized_embedding_model,
                "server_command": pipeline._command_line(server_spec),
                "episode_command": pipeline._command_line(episode_spec),
                "source_audit": source_audit,
                "teacher_forcing": False,
                "actions_supplied_to_mobilegpt": False,
                "runlog_conversion_used": False,
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
            raise RuntimeError(f"mobilegpt_native_server_failed:{server_returncode}")
        episode_returncode = pipeline.run_command(episode_spec)
        if episode_returncode != 0:
            raise RuntimeError(f"mobilegpt_source_episode_failed:{episode_returncode}")
    finally:
        pipeline._stop_background_command(server)
        pipeline._stop_background_command(browser_server)
    wall_sec = round(time.monotonic() - started, 6)

    if episode_spec.output_path is None:
        raise RuntimeError("mobilegpt_source_episode_output_missing")
    result_path = episode_spec.output_path / "task_results.jsonl"
    stats_summary = mobilegpt_memory.summarize_mobilegpt_stats(stats_path)
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

    sealed = pipeline.seal_mobilegpt_source_memory(
        memory_root=memory_root,
        source_run_log=source_path,
        source_stats=stats_path,
        official_source_result=result_path,
        task_name=item.task,
        source_seed=SOURCE_SEED,
        target_package=target["target_package"],
        target_app=target["target_app"],
        source_wall_sec=wall_sec,
        source_model=normalized_model,
    )
    result: dict[str, Any] = {
        "schema_version": "omniflow.mobilegpt-native-source-prepare.v1",
        "task_name": item.task,
        "source_seed": SOURCE_SEED,
        "source_method": MOBILEGPT_SOURCE_METHOD,
        "model": normalized_model,
        "embedding_model": normalized_embedding_model,
        "memory_root": str(memory_root),
        "learning_mode": MOBILEGPT_LEARNING_MODE,
        "teacher_forcing": False,
        "actions_supplied_to_mobilegpt": False,
        "runlog_conversion_used": False,
        "source_emulator_used": True,
        "source_stats": str(stats_path),
        "source_stats_summary": str(stats_summary_path),
        "official_source_result": str(result_path),
        "source_wall_sec": wall_sec,
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
    marker.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.mobilegpt-native-source-failure.v1",
                "failed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
                "retry_allowed": True,
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
    prepare.add_argument(
        "--embedding-model",
        default=os.environ.get("MOBILEGPT_EMBEDDING_MODEL")
        or MOBILEGPT_EMBEDDING_MODEL,
    )
    prepare.add_argument("--memory-index", required=True)
    prepare.add_argument("--serial", default="emulator-5560")
    prepare.add_argument("--console-port", type=int, default=5560)
    prepare.add_argument("--adb-path", default="")
    prepare.add_argument("--max-steps", type=int, default=20)
    prepare.add_argument("--server-host", default="0.0.0.0")
    prepare.add_argument("--port", type=int, default=12345)
    prepare.add_argument("--server-warmup-sec", type=float, default=5.0)
    prepare.add_argument("--wait-start-timeout-sec", type=float, default=60.0)
    prepare.add_argument("--wait-finish-timeout-sec", type=float, default=600.0)
    prepare.add_argument("--timeout-sec", type=float, default=900.0)
    prepare.add_argument("--no-emulator-setup", action="store_true")

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
                android_world_root=args.android_world_root,
                output_root=args.output_root,
                model=args.model,
                embedding_model=args.embedding_model,
                memory_index=args.memory_index,
                serial=args.serial,
                console_port=args.console_port,
                adb_path=args.adb_path,
                max_steps=args.max_steps,
                server_host=args.server_host,
                port=args.port,
                server_warmup_sec=args.server_warmup_sec,
                wait_start_timeout_sec=args.wait_start_timeout_sec,
                wait_finish_timeout_sec=args.wait_finish_timeout_sec,
                timeout_sec=args.timeout_sec,
                perform_emulator_setup=not args.no_emulator_setup,
            )
        elif args.command == "validate":
            result = validate_mobilegpt_source_memory(
                index_path=args.index,
                task_name=args.task,
                memory_root=args.memory_root,
                model=args.model,
                memory_index=args.memory_index,
            )
        else:
            result = preflight_mobilegpt_source(
                index_path=args.index,
                task_name=args.task,
            )
    except BaseException as error:
        if args.command == "prepare":
            _write_failure_marker(args.output_root, error)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
