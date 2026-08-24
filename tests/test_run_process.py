from __future__ import annotations

from pathlib import Path
import signal
import time

import pytest

from src.experiment.run_process import (
    managed_mobilegpt_orphan_pids,
    run_process,
    start_process,
    stop_managed_mobilegpt_orphans,
    stop_process,
)


def test_run_process_preserves_logged_command_result(tmp_path: Path) -> None:
    log_path = tmp_path / "command.log"

    result = run_process(
        ["/bin/sh", "-c", "printf 'episode-output\\n'"],
        cwd=tmp_path,
        environment={},
        log_path=log_path,
        stdin_devnull=True,
        timeout_sec=5,
    )

    assert result["returncode"] == 0
    assert result["timed_out"] is False
    assert log_path.read_text(encoding="utf-8") == "episode-output\n"


def test_run_process_does_not_launch_after_deadline(tmp_path: Path) -> None:
    log_path = tmp_path / "deadline.log"

    result = run_process(
        ["this-command-must-not-run"],
        cwd=tmp_path,
        environment={},
        log_path=log_path,
        timeout_sec=0,
    )

    assert result["returncode"] == 124
    assert result["timed_out"] is True
    assert "deadline exceeded" in log_path.read_text(encoding="utf-8")


def test_run_process_allows_unbounded_command_when_timeout_is_none(
    tmp_path: Path,
) -> None:
    result = run_process(
        ["/bin/sh", "-c", "exit 7"],
        cwd=tmp_path,
        environment={},
        timeout_sec=None,
    )

    assert result["returncode"] == 7
    assert result["timed_out"] is False


def test_run_process_can_forward_one_interactive_input(tmp_path: Path) -> None:
    result = run_process(
        ["/bin/sh", "-c", "read value; printf 'received=%s\\n' \"$value\""],
        cwd=tmp_path,
        environment={},
        timeout_sec=5,
        log_path=tmp_path / "stdin.log",
        stdin_text="official-task\n",
    )

    assert result["returncode"] == 0
    assert (tmp_path / "stdin.log").read_text(encoding="utf-8") == (
        "received=official-task\n"
    )


def test_background_process_uses_shared_group_lifecycle(tmp_path: Path) -> None:
    log_path = tmp_path / "background.log"
    process = start_process(
        ["/bin/sh", "-c", "printf 'background-output\\n'; sleep 30"],
        cwd=tmp_path,
        environment={},
        log_path=log_path,
        log_mode="x",
    )
    try:
        assert process.poll() is None
        time.sleep(0.1)
    finally:
        stop_process(process, timeout_sec=1)
    assert process.poll() is not None
    assert "background-output" in log_path.read_text(encoding="utf-8")


def _write_proc_entry(
    proc_root: Path,
    *,
    pid: int,
    parent_pid: int,
    argv: list[str],
) -> None:
    entry = proc_root / str(pid)
    entry.mkdir(parents=True)
    (entry / "cmdline").write_bytes(b"\0".join(value.encode() for value in argv) + b"\0")
    (entry / "status").write_text(
        f"Name:\tpython\nPid:\t{pid}\nPPid:\t{parent_pid}\n",
        encoding="utf-8",
    )


def test_managed_mobilegpt_orphans_are_limited_to_results_workspace(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    results_root = tmp_path / "data"
    managed_server = (
        results_root
        / "androidworld"
        / "Task"
        / "mobilegpt"
        / "Device_seed111_eval113"
        / "memory"
        / "attempt_001.mobilegpt.device"
        / "_episodes"
        / "device"
        / "official_server_workspace"
        / "Server"
        / "main.py"
    )
    unrelated_server = (
        tmp_path / "other" / "official_server_workspace" / "Server" / "main.py"
    )
    _write_proc_entry(
        proc_root,
        pid=101,
        parent_pid=1,
        argv=["python", str(managed_server)],
    )
    _write_proc_entry(
        proc_root,
        pid=102,
        parent_pid=77,
        argv=["python", str(managed_server)],
    )
    _write_proc_entry(
        proc_root,
        pid=103,
        parent_pid=1,
        argv=["python", str(unrelated_server)],
    )

    assert managed_mobilegpt_orphan_pids(
        results_root=results_root,
        proc_root=proc_root,
    ) == [101]


def test_managed_mobilegpt_cleanup_stops_only_detached_group_leaders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc_root = tmp_path / "proc"
    results_root = tmp_path / "data"
    managed_server = (
        results_root / "Task" / "official_server_workspace" / "Server" / "main.py"
    )
    _write_proc_entry(
        proc_root,
        pid=201,
        parent_pid=1,
        argv=["python", str(managed_server)],
    )
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr("src.experiment.run_process.os.getpgid", lambda pid: pid)
    monkeypatch.setattr(
        "src.experiment.run_process.os.killpg",
        lambda pid, value: signals.append((pid, value)),
    )

    stopped = stop_managed_mobilegpt_orphans(
        results_root=results_root,
        timeout_sec=0,
        proc_root=proc_root,
    )

    assert stopped == [201]
    assert signals == [(201, signal.SIGTERM), (201, signal.SIGKILL)]
