from __future__ import annotations

import argparse
from pathlib import Path
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
    avd: str,
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
    return _flag_values(arguments, "-avd") == (avd,) and _flag_values(
        arguments, "-port"
    ) == (console_port,)


def find_managed_emulator_pids(
    *,
    serial: str,
    avd: str,
    proc_root: Path = Path("/proc"),
) -> tuple[int, ...]:
    if not proc_root.is_dir():
        return ()
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
    parser.add_argument("--avd", required=True)
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    for process_id in find_managed_emulator_pids(
        serial=args.serial,
        avd=args.avd,
        proc_root=args.proc_root,
    ):
        print(process_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
