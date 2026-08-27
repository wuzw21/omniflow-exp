"""Minimal AndroidWorld memory conversion and task runner."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence

from src.experiment.function_v2 import compile_function_v2
from src.experiment.appagent_source import convert_runlog_to_appagent_memory
from src.experiment.protocol import (
    DEFAULT_DEVICE,
    DEFAULT_METHOD,
    DEFAULT_TASK,
    DEVICE_AVDS,
    DEVICES,
    FORMAL_MODEL,
    FORMAL_MODEL_BASE_URL,
    FORMAL_MODEL_ENDPOINT_PROFILE,
    MAX_FALLBACK_STEPS,
    MAX_STEPS,
    METHODS,
    SOURCE_DEVICE,
    SOURCE_METHOD,
    SOURCE_SEED,
    TASK_DEADLINE_SEC,
    TASK_SEED,
    require_formal_model,
)
from src.integrations.mobilegpt import convert_runlog_to_mobilegpt_bundle


def _methods(value: str) -> tuple[str, ...]:
    if not value or value == "all":
        return METHODS
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = tuple(
        item for item in selected if item not in (*METHODS, SOURCE_METHOD)
    )
    if unknown:
        raise ValueError("unknown_method:" + ",".join(unknown))
    return selected


def _devices(value: str) -> tuple[tuple[str, str, int], ...]:
    if not value or value == "all":
        return DEVICES
    configured_devices = {item[0]: item for item in (*DEVICES, SOURCE_DEVICE)}
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    devices: list[tuple[str, str, int]] = []
    for item in selected:
        configured = configured_devices.get(item)
        if configured is None:
            raise ValueError(f"unknown_configured_device:{item}")
        devices.append(configured)
    return tuple(devices)


def _sdk_tool(name: str) -> str:
    discovered = shutil.which(name)
    if discovered:
        return discovered
    adb = shutil.which("adb")
    if adb:
        candidate = Path(adb).resolve().parent.parent / "emulator" / name
        if candidate.is_file():
            return str(candidate)
    for variable in ("ANDROID_SDK_ROOT", "ANDROID_HOME"):
        root = str(os.environ.get(variable) or "").strip()
        if root:
            tool_directory = "platform-tools" if name == "adb" else "emulator"
            candidate = Path(root).expanduser() / tool_directory / name
            if candidate.is_file():
                return str(candidate)
    tool_directory = "platform-tools" if name == "adb" else "emulator"
    for root in (Path.home() / "Library/Android/sdk", Path.home() / "Android/Sdk"):
        candidate = root / tool_directory / name
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError(f"android_sdk_tool_missing:{name}")


def _mobilegpt_server_port(console_port: int) -> int:
    """Derive a deterministic local TCP port for one AVD.

    MobileGPT's official Server is one process per device.  The upstream
    checkout defaults to 12345, so parallel AVD runs would contend for the
    same listener even though their Android serials are distinct.  Reuse the
    configured emulator console port as the stable source for an isolated,
    unprivileged server port.
    """

    # The original source5560 APK is the canonical MobileGPT collector and
    # embeds the upstream server port 12345.  Keep that source-only contract;
    # formal target AVDs use deterministic isolated ports derived from their
    # own console ports below.
    if int(console_port) == 5560:
        return 12345
    return 12000 + int(console_port) % 40000


def _device_booted(adb: str, serial: str) -> bool:
    try:
        state = subprocess.run(
            (adb, "-s", serial, "get-state"),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if state.returncode != 0 or state.stdout.strip() != "device":
            return False
        booted = subprocess.run(
            (adb, "-s", serial, "shell", "getprop", "sys.boot_completed"),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return booted.returncode == 0 and booted.stdout.strip() == "1"
    except (OSError, subprocess.SubprocessError):
        return False


def _wait_for_device(adb: str, serial: str, timeout: int) -> None:
    deadline = time.monotonic() + max(1, timeout)
    while time.monotonic() < deadline:
        if _device_booted(adb, serial):
            return
        time.sleep(1)
    raise RuntimeError(f"android_emulator_boot_timeout:{serial}")


def _running_avd_name(adb: str, serial: str) -> str:
    try:
        completed = subprocess.run(
            (adb, "-s", serial, "emu", "avd", "name"),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0:
        return ""
    return next(
        (
            line.strip()
            for line in completed.stdout.splitlines()
            if line.strip() and line.strip().casefold() != "ok"
        ),
        "",
    )


def _validate_configured_avd_identity(
    adb: str,
    devices: tuple[tuple[str, str, int], ...],
    avds: dict[str, str],
) -> None:
    for _label, serial, _port in devices:
        expected = str(avds.get(serial) or "").strip()
        if not expected:
            continue
        actual = _running_avd_name(adb, serial)
        if actual != expected:
            raise RuntimeError(
                "android_emulator_avd_mismatch:"
                f"{serial}:expected={expected}:actual={actual or 'unknown'}"
            )


def _ensure_devices_started(
    devices: tuple[tuple[str, str, int], ...],
    *,
    timeout: int,
) -> None:
    """Reuse online configured AVDs and start only the missing ones."""

    adb = _sdk_tool("adb")
    avds = dict(DEVICE_AVDS)
    missing = tuple(device for device in devices if not _device_booted(adb, device[1]))
    if missing:
        emulator = _sdk_tool("emulator")
        for _label, serial, port in missing:
            avd = avds.get(serial)
            if not avd:
                continue
            subprocess.Popen(
                (
                    emulator,
                    "-avd",
                    avd,
                    "-port",
                    str(port),
                    "-grpc",
                    str(port + 3000),
                    "-no-window",
                    "-no-audio",
                    "-no-boot-anim",
                    "-no-snapshot-save",
                    "-gpu",
                    "swiftshader_indirect",
                ),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, len(devices))
    ) as executor:
        tuple(
            executor.map(
                lambda device: _wait_for_device(adb, device[1], timeout),
                devices,
            )
        )
    _validate_configured_avd_identity(adb, devices, avds)


def _convert_memory(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source_run_log).expanduser()
    output = Path(args.memory).expanduser()
    if args.method == "omniflow":
        report = compile_function_v2(
            source,
            output,
            enhance=True,
            model=FORMAL_MODEL,
            model_endpoint_profile=FORMAL_MODEL_ENDPOINT_PROFILE,
            model_base_url=FORMAL_MODEL_BASE_URL,
        )
        memory = Path(str(report["store_path"]))
    elif args.method == "mobilegpt":
        report = convert_runlog_to_mobilegpt_bundle(
            source_run_log=source,
            mobilegpt_root=args.mobilegpt_root,
            output_root=output,
            model=FORMAL_MODEL,
            source_seed=SOURCE_SEED,
        )
        memory = Path(str(report["memory_root"]))
    elif args.method == "appagent":
        report = convert_runlog_to_appagent_memory(
            source_run_log=source,
            appagent_root=args.appagent_root,
            memory_root=output,
            model=FORMAL_MODEL,
        )
        memory = Path(str(report["memory_root"]))
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
    ]
    if args.mobilegpt_root:
        command.extend(("--mobilegpt-root", args.mobilegpt_root))
    if args.appagent_root:
        command.extend(("--appagent-root", args.appagent_root))
    if method == "mobilegpt":
        command.extend(
            (
                "--mobilegpt-port",
                str(_mobilegpt_server_port(port)),
                # The official Server imports its vision/memory stack before
                # opening the per-device socket.  Five seconds is too short
                # on a cold Python process and is reported as an environment
                # failure before an episode starts; keep this startup wait
                # outside the episode wall-clock measurement.
                "--mobilegpt-server-warmup-sec",
                "30",
            )
        )
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
    if SOURCE_METHOD in methods and devices != (SOURCE_DEVICE,):
        raise ValueError("source_method_requires_device:source5560")
    if args.dry_run:
        return {
            "action": "run",
            "task": args.task,
            "methods": list(methods),
            "devices": [item[0] for item in devices],
            "memory": args.memory or None,
            "source_run_log": args.source_run_log or None,
        }

    _ensure_devices_started(devices, timeout=min(TASK_DEADLINE_SEC, 180))
    started = time.monotonic()
    with tempfile.TemporaryDirectory(
        prefix="omniflow-androidworld-"
    ) as temporary:
        root = Path(temporary)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, len(devices))
        ) as executor:
            rows_by_device = list(
                executor.map(
                    lambda device: tuple(
                        _run_command(args, method, device, root)
                        for method in methods
                    ),
                    devices,
                )
            )
        rows = [row for device_rows in rows_by_device for row in device_rows]
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
    parser.add_argument(
        "--mobilegpt-root",
        default=os.environ.get("OMNIFLOW_MOBILEGPT_ROOT", ""),
    )
    parser.add_argument(
        "--appagent-root",
        default=os.environ.get(
            "OMNIFLOW_APPAGENT_ROOT",
            str(Path.home() / "Projects" / "Omni" / "OmniFlow" / "runtime" / "external" / "appagent"),
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(repo=repo)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_formal_model()
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
