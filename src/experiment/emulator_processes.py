from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
from typing import Sequence


def _flag_values(arguments: Sequence[str], flag: str) -> tuple[str, ...]:
    return tuple(
        arguments[index + 1]
        for index, argument in enumerate(arguments[:-1])
        if argument == flag
    )


def is_managed_emulator_process(
    arguments: Sequence[str],
    *,
    serial: str,
    avd: str | None,
) -> bool:
    prefix = "emulator-"
    if not serial.startswith(prefix) or not serial.removeprefix(prefix).isdigit():
        raise ValueError(f"invalid emulator serial: {serial}")
    if not arguments:
        return False
    executable = Path(arguments[0]).name
    if executable != "emulator" and not executable.startswith("qemu-system-"):
        return False
    console_port = serial.removeprefix(prefix)
    if _flag_values(arguments, "-port") != (console_port,):
        return False
    return avd is None or _flag_values(arguments, "-avd") == (avd,)


def _system_process_arguments() -> tuple[tuple[int, tuple[str, ...]], ...]:
    completed = subprocess.run(
        ["ps", "-axo", "pid=,command="],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    rows: list[tuple[int, tuple[str, ...]]] = []
    for raw_line in completed.stdout.splitlines():
        fields = raw_line.strip().split(maxsplit=1)
        if len(fields) != 2 or not fields[0].isdigit():
            continue
        try:
            arguments = tuple(shlex.split(fields[1]))
        except ValueError:
            continue
        rows.append((int(fields[0]), arguments))
    return tuple(rows)


def find_managed_emulator_pids(
    *,
    serial: str,
    avd: str | None,
    proc_root: Path = Path("/proc"),
) -> tuple[int, ...]:
    if not proc_root.is_dir():
        return tuple(
            process_id
            for process_id, arguments in _system_process_arguments()
            if is_managed_emulator_process(arguments, serial=serial, avd=avd)
        )
    matches: list[int] = []
    for process_root in sorted(
        (path for path in proc_root.iterdir() if path.name.isdigit()),
        key=lambda path: int(path.name),
    ):
        try:
            raw_arguments = (process_root / "cmdline").read_bytes()
        except OSError:
            continue
        arguments = tuple(
            raw.decode(errors="surrogateescape")
            for raw in raw_arguments.split(b"\0")
            if raw
        )
        if is_managed_emulator_process(arguments, serial=serial, avd=avd):
            matches.append(int(process_root.name))
    return tuple(matches)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", required=True)
    parser.add_argument("--avd")
    parser.add_argument("--any-avd", action="store_true")
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if bool(args.avd) == bool(args.any_avd):
        raise SystemExit("exactly one of --avd and --any-avd is required")
    for process_id in find_managed_emulator_pids(
        serial=args.serial,
        avd=None if args.any_avd else args.avd,
        proc_root=args.proc_root,
    ):
        print(process_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
