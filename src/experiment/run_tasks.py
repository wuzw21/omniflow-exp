"""Minimal AndroidWorld memory conversion and task runner."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence

from src.experiment.appagent_source import convert_runlog_to_appagent_memory
from src.experiment.function_v2 import compile_function_v2
from src.experiment.protocol import (
    DEFAULT_DEVICE,
    DEFAULT_METHOD,
    DEFAULT_TASK,
    DEVICES,
    FORMAL_MODEL,
    FORMAL_MODEL_BASE_URL,
    FORMAL_MODEL_ENDPOINT_PROFILE,
    MAX_FALLBACK_STEPS,
    MAX_STEPS,
    METHODS,
    SOURCE_SEED,
    TASK_DEADLINE_SEC,
    TASK_SEED,
)
from src.integrations.mobilegpt import convert_runlog_to_mobilegpt_bundle


def _methods(value: str) -> tuple[str, ...]:
    if not value or value == "all":
        return METHODS
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = tuple(item for item in selected if item not in METHODS)
    if unknown:
        raise ValueError("unknown_method:" + ",".join(unknown))
    return selected


def _devices(value: str) -> tuple[tuple[str, str, int], ...]:
    if not value or value == "all":
        return DEVICES
    catalog = {item[0]: item for item in DEVICES}
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = tuple(item for item in selected if item not in catalog)
    if unknown:
        raise ValueError("unknown_device:" + ",".join(unknown))
    return tuple(catalog[item] for item in selected)


def _convert_memory(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source_run_log).expanduser()
    output = Path(args.memory).expanduser()
    if args.method == "omniflow":
        report = compile_function_v2(
            source,
            output,
            enhance=bool(args.model),
            model=args.model,
            model_endpoint_profile=FORMAL_MODEL_ENDPOINT_PROFILE,
            model_base_url=FORMAL_MODEL_BASE_URL,
        )
        memory = Path(str(report["store_path"]))
    elif args.method == "mobilegpt":
        report = convert_runlog_to_mobilegpt_bundle(
            source_run_log=source,
            mobilegpt_root=args.mobilegpt_root,
            output_root=output,
            model=args.model,
            source_seed=args.source_seed,
        )
        memory = Path(str(report["memory_root"]))
    elif args.method == "appagent":
        convert_runlog_to_appagent_memory(
            source_run_log=source,
            appagent_root=args.appagent_root,
            memory_root=output,
            model=args.model,
        )
        memory = output
    else:
        memory = source
    return {
        "action": "convert-memory",
        "task": args.task,
        "method": args.method,
        "memory": str(memory),
    }


def _run_command(
    args: argparse.Namespace,
    method: str,
    device: tuple[str, str, int],
    temporary_root: Path,
) -> tuple[str, str, int]:
    label, serial, port = device
    command = [
        sys.executable,
        "-m",
        "src.experiment.run_task",
        "result",
        "--task",
        args.task,
        "--method",
        method,
        "--device",
        f"{label}:{serial}:{port}",
        "--output-path",
        str(temporary_root / method / label),
        "--source-seed",
        str(args.source_seed),
        "--task-random-seed",
        str(args.evaluation_seed),
        "--max-steps",
        str(args.max_steps),
        "--max-fallback-steps",
        str(args.max_fallback_steps),
        "--timeout-sec",
        str(args.deadline),
        "--mobilegpt-wait-start-timeout-sec",
        str(args.deadline),
        "--model",
        args.model,
    ]
    if args.mobilegpt_root:
        command.extend(("--mobilegpt-root", args.mobilegpt_root))
    if args.appagent_root:
        command.extend(("--appagent-root", args.appagent_root))
    if args.source_run_log:
        command.extend(("--source-run-log", args.source_run_log))
    if args.memory:
        command.extend(("--memory", args.memory))
    if args.dry_run:
        command.append("--dry-run")
    environment = dict(os.environ)
    environment["OMNIFLOW_ANDROIDWORLD_ARCHIVE_ROOT"] = str(args.output)
    return method, label, subprocess.run(
        command,
        env=environment,
        check=False,
    ).returncode


def _run(args: argparse.Namespace) -> dict[str, Any]:
    methods = _methods(args.method)
    devices = _devices(args.device)
    if args.dry_run:
        return {
            "action": "run",
            "task": args.task,
            "methods": list(methods),
            "devices": [item[0] for item in devices],
            "memory": args.memory or None,
            "source_run_log": args.source_run_log or None,
        }

    started = time.monotonic()
    with tempfile.TemporaryDirectory(
        prefix="omniflow-androidworld-"
    ) as temporary:
        root = Path(temporary)
        jobs = tuple((method, device) for device in devices for method in methods)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, len(devices))
        ) as executor:
            rows = list(
                executor.map(
                    lambda job: _run_command(args, job[0], job[1], root),
                    jobs,
                )
            )
    return {
        "action": "run",
        "task": args.task,
        "status": "completed" if all(row[2] == 0 for row in rows) else "failed",
        "wall_sec": round(time.monotonic() - started, 3),
        "runs": [
            {"method": method, "device": device, "returncode": returncode}
            for method, device, returncode in rows
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        nargs="?",
        choices=("run", "convert-memory"),
        default="run",
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--memory", default="")
    parser.add_argument("--source-run-log", default="")
    parser.add_argument("--output", default=str(repo / "data" / "androidworld"))
    parser.add_argument("--source-seed", type=int, default=SOURCE_SEED)
    parser.add_argument("--evaluation-seed", type=int, default=TASK_SEED)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--max-fallback-steps", type=int, default=MAX_FALLBACK_STEPS)
    parser.add_argument("--deadline", type=int, default=TASK_DEADLINE_SEC)
    parser.add_argument("--model", default=FORMAL_MODEL)
    parser.add_argument(
        "--mobilegpt-root",
        default=os.environ.get("OMNIFLOW_MOBILEGPT_ROOT", ""),
    )
    parser.add_argument(
        "--appagent-root",
        default=os.environ.get("OMNIFLOW_APPAGENT_ROOT", ""),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(repo=repo)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "convert-memory":
        if not args.source_run_log or not args.memory:
            raise ValueError("convert-memory requires --source-run-log and --memory")
        if args.method not in METHODS:
            raise ValueError(f"unknown_method:{args.method}")
        result = (
            {
                "action": "convert-memory",
                "task": args.task,
                "method": args.method,
                "source_run_log": args.source_run_log,
                "memory": args.memory,
            }
            if args.dry_run
            else _convert_memory(args)
        )
    else:
        result = _run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status", "completed") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
