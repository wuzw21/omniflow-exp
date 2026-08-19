"""One process-lifecycle seam for experiment commands.

Command builders describe an episode; schedulers decide when to launch it;
this module owns only child-process lifecycle, timeout, and optional immutable
log capture.  Keeping that policy here prevents AndroidWorld and B-MoCA from
drifting into subtly different timeout behavior.
"""

from __future__ import annotations

from contextlib import nullcontext
import datetime as dt
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Sequence


def start_process(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log_path: Path | None = None,
    log_mode: str = "a",
    stdin_devnull: bool = False,
) -> subprocess.Popen[Any]:
    """Start one background experiment process under the shared policy."""

    command_list = list(command)
    resolved_log_path = Path(log_path).expanduser().resolve() if log_path else None
    log_file = None
    try:
        if resolved_log_path is not None:
            resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = resolved_log_path.open(log_mode, encoding="utf-8")
        return subprocess.Popen(
            command_list,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL if stdin_devnull else None,
            stdout=log_file,
            stderr=subprocess.STDOUT if log_file is not None else None,
            text=True,
            start_new_session=True,
        )
    finally:
        if log_file is not None:
            log_file.close()


def stop_process(
    process: subprocess.Popen[Any] | None,
    *,
    timeout_sec: float = 10.0,
) -> None:
    """Stop one background process group and escalate after a bounded wait."""

    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=max(0.1, float(timeout_sec)))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_process(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    timeout_sec: float | None,
    log_path: Path | None = None,
    stdin_devnull: bool = False,
    stdin_text: str | None = None,
) -> dict[str, Any]:
    """Run one command with one timeout and one process-group policy.

    A non-positive timeout is a scheduler decision that the child must not be
    launched.  When ``log_path`` is supplied, the file is opened exclusively
    so an existing immutable attempt cannot be overwritten.
    """

    command_list = list(command)
    resolved_log_path = Path(log_path).expanduser().resolve() if log_path else None
    if timeout_sec is not None and timeout_sec <= 0:
        if resolved_log_path is not None:
            resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
            resolved_log_path.write_text(
                "global task deadline exceeded before launch\n",
                encoding="utf-8",
            )
        return {
            "command": command_list,
            "returncode": 124,
            "timed_out": True,
            "wall_sec": 0.0,
            "log_path": str(resolved_log_path) if resolved_log_path else "",
        }

    if resolved_log_path is not None:
        resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    log_context = (
        resolved_log_path.open("x", encoding="utf-8")
        if resolved_log_path is not None
        else nullcontext(None)
    )
    try:
        with log_context as log_file:
            process = subprocess.Popen(
                command_list,
                cwd=cwd,
                env=environment,
                stdin=(
                    subprocess.PIPE
                    if stdin_text is not None
                    else subprocess.DEVNULL
                    if stdin_devnull
                    else None
                ),
                stdout=log_file,
                stderr=subprocess.STDOUT if log_file is not None else None,
                text=True,
                start_new_session=True,
            )
            try:
                if stdin_text is not None:
                    process.communicate(
                        input=stdin_text,
                        timeout=(
                            None
                            if timeout_sec is None
                            else max(0.1, float(timeout_sec))
                        ),
                    )
                else:
                    process.wait(
                        timeout=(
                            None
                            if timeout_sec is None
                            else max(0.1, float(timeout_sec))
                        )
                    )
                timed_out = False
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate_process_group(process)
    except BaseException:
        if "process" in locals() and process.poll() is None:
            _terminate_process_group(process)
        raise

    return {
        "command": command_list,
        "returncode": 124 if timed_out else int(process.returncode or 0),
        "timed_out": timed_out,
        "wall_sec": round(time.monotonic() - started, 6),
        "log_path": str(resolved_log_path) if resolved_log_path else "",
        "process_pid": int(process.pid),
        "started_at": started_at,
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    stop_process(process)
