from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from typing import Sequence

from src.experiment.emulator_processes import find_managed_emulator_pids


def _console_port(serial: str) -> int:
    prefix = "emulator-"
    if not serial.startswith(prefix) or not serial.removeprefix(prefix).isdigit():
        raise ValueError(f"invalid emulator serial: {serial}")
    return int(serial.removeprefix(prefix))


def _grpc_ready(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def _adb_output(adb: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            [str(adb), *arguments],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        # An emulator transitioning between offline, booting, and ready can
        # leave an individual adb probe hanging.  Readiness owns retry and
        # relaunch; a single timed-out probe must not abort the whole task.
        return ""
    return completed.stdout.replace("\r", "").strip()


def _device_ready(adb: Path, serial: str, avd: str, grpc_port: int) -> bool:
    avd_lines = _adb_output(adb, "-s", serial, "emu", "avd", "name").splitlines()
    active_avd = avd_lines[0].strip() if avd_lines else ""
    return (
        _adb_output(adb, "-s", serial, "get-state") == "device"
        and active_avd == avd
        and _adb_output(adb, "-s", serial, "shell", "getprop", "sys.boot_completed")
        == "1"
        and _grpc_ready(grpc_port)
    )


def _wait_until_stopped(adb: Path, serial: str, grpc_port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (
            _adb_output(adb, "-s", serial, "get-state") != "device"
            and not _grpc_ready(grpc_port)
        ):
            return True
        time.sleep(1)
    return False


def _stop_exact_emulator(adb: Path, serial: str, grpc_port: int) -> None:
    _adb_output(adb, "-s", serial, "emu", "kill")
    if _wait_until_stopped(adb, serial, grpc_port, 20):
        return
    process_ids = find_managed_emulator_pids(serial=serial, avd=None)
    if len(process_ids) > 1:
        raise RuntimeError(f"ambiguous emulator processes for {serial}: {process_ids}")
    if process_ids:
        os.kill(process_ids[0], signal.SIGTERM)
    if not _wait_until_stopped(adb, serial, grpc_port, 10):
        raise RuntimeError(f"emulator did not stop: {serial}")


def ensure_development_emulator(
    *,
    adb: Path,
    emulator: Path,
    serial: str,
    avd: str,
    gpu: str,
    log_path: Path,
    boot_timeout: float,
) -> str:
    console_port = _console_port(serial)
    grpc_port = console_port + 3000
    if _device_ready(adb, serial, avd, grpc_port):
        return "reused"
    _stop_exact_emulator(adb, serial, grpc_port)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as log_file:
        subprocess.Popen(
            [
                str(emulator),
                "-avd",
                avd,
                "-port",
                str(console_port),
                "-grpc",
                str(grpc_port),
                "-no-window",
                "-no-audio",
                "-no-boot-anim",
                "-read-only",
                "-no-snapshot-load",
                "-no-snapshot-save",
                "-gpu",
                gpu,
            ],
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    deadline = time.monotonic() + boot_timeout
    while time.monotonic() < deadline:
        if _device_ready(adb, serial, avd, grpc_port):
            return "launched"
        time.sleep(1)
    raise RuntimeError(
        f"development emulator not ready: serial={serial} avd={avd} "
        f"grpc={grpc_port} log={log_path}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adb", type=Path, required=True)
    parser.add_argument("--emulator", type=Path, required=True)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--avd", required=True)
    parser.add_argument("--gpu", default="swiftshader_indirect")
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--boot-timeout", type=float, default=240)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    state = ensure_development_emulator(
        adb=args.adb.resolve(),
        emulator=args.emulator.resolve(),
        serial=args.serial,
        avd=args.avd,
        gpu=args.gpu,
        log_path=args.log_path.resolve(),
        boot_timeout=max(1.0, args.boot_timeout),
    )
    print(f"[emulator] {state} serial={args.serial} avd={args.avd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
