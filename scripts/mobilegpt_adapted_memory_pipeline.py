#!/usr/bin/env python3
"""Prepare and seal native MobileGPT memory from one source-seed RunLog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import androidworld_replay_pipeline as pipeline


def prepare_mobilegpt_adapted_memory(
    *,
    mobilegpt_root: str | Path,
    android_world_root: str | Path,
    output_root: str | Path,
    source_run_log: str | Path,
    task_name: str,
    task_params: dict[str, Any],
    serial: str = "emulator-5560",
    console_port: int = 5560,
    adb_path: str = "",
    max_steps: int = 20,
    server_host: str = "0.0.0.0",
    port: int = 12345,
    server_warmup_sec: float = 2.0,
    wait_start_timeout_sec: float = 60.0,
    wait_finish_timeout_sec: float = 180.0,
    perform_emulator_setup: bool = True,
) -> dict[str, Any]:
    bundle_root = Path(output_root).expanduser().resolve()
    if bundle_root.exists():
        raise FileExistsError(
            f"immutable_mobilegpt_adapted_memory_exists:{bundle_root}"
        )
    bundle_root.mkdir(parents=True)
    source_path = Path(source_run_log).expanduser().resolve()
    teacher_payload = pipeline.build_mobilegpt_teacher_source(
        source_path,
        task_name=task_name,
        source_seed=111,
    )
    teacher_source_path = bundle_root / "teacher_source.json"
    teacher_source_path.write_text(
        json.dumps(teacher_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    memory_root = bundle_root / "memory"
    memory_root.mkdir()
    stats_path = bundle_root / "source_stats.jsonl"
    runtime_serial_file = bundle_root / "active_serial.txt"
    pipeline._write_mobilegpt_runtime_serial(runtime_serial_file, serial)

    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    goal = str(
        source_payload.get("goal")
        or source_payload.get("operation_description")
        or task_name
    )
    item = pipeline.ArchivedRunLog(
        task=str(task_name),
        goal=goal,
        params=dict(task_params),
        source_run_log=source_path,
        replay_seed=111,
        step_count=int(teacher_payload["action_count"]),
        meta={"latest_official_success_source": True, "source_seed": 111},
    )
    target_info = pipeline._infer_mobilegpt_target_from_source_run_log(item)
    target_package = str(target_info.get("target_package") or "").strip()
    if not target_package:
        raise ValueError("mobilegpt_source_target_package_unresolved")
    target_app = str(target_info.get("target_app") or target_package).strip()
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
        source_run_log=source_path,
        fallback_to_vlm_on_teacher_miss=False,
        target_package=target_package,
        target_app=target_app,
        runtime_serial_file=runtime_serial_file,
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
        task_random_seed=111,
        fixed_task_seed=True,
        fixed_task_params=True,
        task_params_override=dict(task_params),
        perform_emulator_setup=bool(perform_emulator_setup),
        adb_path=str(adb_path),
        start_timeout_sec=float(wait_start_timeout_sec),
        finish_timeout_sec=float(wait_finish_timeout_sec),
        rebroadcast_limit=1,
    )
    command_manifest_path = bundle_root / "source_episode_command.json"
    command_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.mobilegpt-source-command.v1",
                "source_seed": 111,
                "task_name": str(task_name),
                "task_params": task_params,
                "serial": str(serial),
                "server_command": pipeline._command_line(server_spec),
                "episode_command": pipeline._command_line(episode_spec),
                "source_run_log": str(source_path),
                "source_run_log_sha256": pipeline._file_sha256(source_path),
                "teacher_source": str(teacher_source_path),
                "teacher_source_sha256": pipeline._file_sha256(
                    teacher_source_path
                ),
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
    wall_sec = round(time.monotonic() - started, 6)

    if episode_spec.output_path is None:
        raise RuntimeError("mobilegpt_source_episode_output_missing")
    result_path = episode_spec.output_path / "task_results.jsonl"
    sealed = pipeline.seal_mobilegpt_adapted_memory(
        memory_root=memory_root,
        teacher_source=teacher_source_path,
        source_run_log=source_path,
        source_stats=stats_path,
        official_source_result=result_path,
        task_name=task_name,
        source_seed=111,
        target_package=target_package,
        target_app=target_app,
        source_wall_sec=wall_sec,
    )
    return {
        "schema_version": "omniflow.mobilegpt-adapted-memory-prepare.v1",
        "task_name": str(task_name),
        "source_seed": 111,
        "memory_root": str(memory_root),
        "teacher_source": str(teacher_source_path),
        "source_stats": str(stats_path),
        "official_source_result": str(result_path),
        "source_wall_sec": wall_sec,
        "sealed": sealed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--mobilegpt-root", required=True)
    prepare.add_argument("--android-world-root", required=True)
    prepare.add_argument("--output-root", required=True)
    prepare.add_argument("--source-run-log", required=True)
    prepare.add_argument("--task", required=True)
    prepare.add_argument("--task-params-json", required=True)
    prepare.add_argument("--serial", default="emulator-5560")
    prepare.add_argument("--console-port", type=int, default=5560)
    prepare.add_argument("--adb-path", default="")
    prepare.add_argument("--max-steps", type=int, default=20)
    prepare.add_argument("--server-host", default="0.0.0.0")
    prepare.add_argument("--port", type=int, default=12345)
    prepare.add_argument("--server-warmup-sec", type=float, default=2.0)
    prepare.add_argument("--wait-start-timeout-sec", type=float, default=60.0)
    prepare.add_argument("--wait-finish-timeout-sec", type=float, default=180.0)
    prepare.add_argument("--no-emulator-setup", action="store_true")
    validate = subparsers.add_parser("validate")
    validate.add_argument("--memory-root", required=True)
    validate.add_argument("--source-run-log", required=True)
    validate.add_argument("--task", required=True)
    validate.add_argument("--source-seed", type=int, default=111)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "prepare":
        task_params = json.loads(args.task_params_json)
        if not isinstance(task_params, dict):
            raise ValueError("--task-params-json must decode to an object")
        result = prepare_mobilegpt_adapted_memory(
            mobilegpt_root=args.mobilegpt_root,
            android_world_root=args.android_world_root,
            output_root=args.output_root,
            source_run_log=args.source_run_log,
            task_name=args.task,
            task_params=task_params,
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
    else:
        result = pipeline.validate_mobilegpt_adapted_memory(
            args.memory_root,
            task_name=args.task,
            source_seed=args.source_seed,
            source_run_log=args.source_run_log,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
