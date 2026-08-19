from __future__ import annotations

from pathlib import Path
import time

from src.experiment.run_process import run_process, start_process, stop_process


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
