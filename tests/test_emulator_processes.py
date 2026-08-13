from __future__ import annotations

from pathlib import Path

import pytest

from src.experiment.emulator_processes import (
    find_managed_emulator_pids,
    is_managed_emulator_process,
)


def _write_process(
    proc_root: Path,
    process_id: int,
    arguments: tuple[str, ...],
) -> None:
    process_root = proc_root / str(process_id)
    process_root.mkdir()
    (process_root / "cmdline").write_bytes(
        b"\0".join(argument.encode() for argument in arguments) + b"\0"
    )


def test_find_managed_emulator_pids_requires_exact_avd_and_port(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    executable = "/sdk/qemu/qemu-system-x86_64-headless"
    _write_process(
        proc_root,
        101,
        (executable, "-avd", "SmallPhone", "-port", "5554", "-no-window"),
    )
    _write_process(
        proc_root,
        102,
        (executable, "-avd", "OtherPhone", "-port", "5554"),
    )
    _write_process(
        proc_root,
        103,
        (executable, "-avd", "SmallPhone", "-port", "5564"),
    )
    _write_process(
        proc_root,
        104,
        ("bash", "-avd", "SmallPhone", "-port", "5554"),
    )

    assert find_managed_emulator_pids(
        serial="emulator-5554",
        avd="SmallPhone",
        proc_root=proc_root,
    ) == (101,)


def test_find_managed_emulator_pids_accepts_emulator_launcher(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _write_process(
        proc_root,
        201,
        ("/sdk/emulator/emulator", "-port", "5554", "-avd", "SmallPhone"),
    )

    assert find_managed_emulator_pids(
        serial="emulator-5554",
        avd="SmallPhone",
        proc_root=proc_root,
    ) == (201,)


def test_managed_emulator_process_rejects_invalid_serial() -> None:
    with pytest.raises(ValueError, match="invalid emulator serial"):
        is_managed_emulator_process(
            ("emulator", "-avd", "SmallPhone", "-port", "5554"),
            serial="phone-5554",
            avd="SmallPhone",
        )


def test_managed_emulator_process_can_match_unique_console_port() -> None:
    arguments = (
        "/sdk/emulator/qemu/darwin-aarch64/qemu-system-aarch64-headless",
        "-avd",
        "OldAvdName",
        "-port",
        "5560",
    )

    assert is_managed_emulator_process(
        arguments,
        serial="emulator-5560",
        avd=None,
    )
    assert not is_managed_emulator_process(
        arguments,
        serial="emulator-5564",
        avd=None,
    )
