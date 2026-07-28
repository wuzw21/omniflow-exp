from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any

from src.integrations.android_world.host import make_agent_result


MOBILEGPT_PACKAGE = "com.example.MobileGPT"
MOBILEGPT_ACTIVITY = "com.example.MobileGPT/.MainActivity"
MOBILEGPT_SERVICE = (
    "com.example.MobileGPT/com.example.MobileGPT.MobileGPTAccessibilityService"
)
MOBILEGPT_ACTION = "com.example.MobileGPT.STRING_ACTION"
MOBILEGPT_INSTRUCTION_EXTRA = "com.example.MobileGPT.INSTRUCTION_EXTRA"
ANDROIDWORLD_CANONICAL_EMULATOR_MODEL = "sdk_gphone_x86_64"


def _device_compatible_goal(goal: str, device_model: str) -> str:
    model = str(device_model or "").strip()
    if not model.startswith("sdk_gphone"):
        return goal
    return str(goal).replace(ANDROIDWORLD_CANONICAL_EMULATOR_MODEL, model)


def _event_count(path: Path, event: str) -> int:
    if not path.is_file():
        return 0
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        count += payload.get("event") == event
    return count


def _event_text_after(
    path: Path,
    event: str,
    previous_count: int,
) -> str:
    if not path.is_file():
        return ""
    seen = 0
    latest = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event") != event:
            continue
        seen += 1
        if seen > previous_count:
            text = str(payload.get("text") or "").strip()
            if text:
                latest = text
    return latest


def _wait_for_event(
    path: Path,
    event: str,
    previous_count: int,
    timeout_sec: float,
) -> bool:
    unbounded = timeout_sec < 0
    deadline = None if unbounded else time.monotonic() + max(0.0, timeout_sec)
    while True:
        if _event_count(path, event) > previous_count:
            return True
        if deadline is not None and time.monotonic() >= deadline:
            return False
        time.sleep(0.5)


def build_mobilegpt_agent(*, env: Any, adb_serial: str, adb_path: str = "") -> Any:
    adb = str(adb_path or "adb").strip() or "adb"
    serial = str(adb_serial or os.environ.get("ANDROID_SERIAL") or "").strip()
    stats_path = Path(os.environ["MOBILEGPT_STATS_JSONL"]).expanduser().resolve()
    serial_file_text = str(os.environ.get("MOBILEGPT_OOB_SERIAL_FILE") or "").strip()
    start_timeout = float(os.environ.get("MOBILEGPT_WAIT_START_TIMEOUT_SEC") or 60.0)
    finish_timeout = float(os.environ.get("MOBILEGPT_WAIT_FINISH_TIMEOUT_SEC") or 120.0)
    rebroadcast_limit = int(os.environ.get("MOBILEGPT_REBROADCAST_LIMIT") or 1)

    class MobileGPTAndroidWorldAgent:
        name = "external:mobilegpt"
        transition_pause = 0.0

        def __init__(self) -> None:
            self.env = env
            self.attempted = False

        def reset(self, go_home: bool = False) -> None:
            self.attempted = False
            self.env.reset(go_home=go_home)

        def set_max_steps(self, max_steps: int) -> None:
            del max_steps

        def _adb(self, *args: str) -> subprocess.CompletedProcess[str]:
            command = [adb]
            if serial:
                command.extend(["-s", serial])
            command.extend(args)
            return subprocess.run(
                command,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        def _prepare(self) -> None:
            enabled = self._adb(
                "shell", "settings", "get", "secure", "enabled_accessibility_services"
            ).stdout.strip()
            services = [
                value
                for value in enabled.split(":")
                if value and value.lower() != "null" and value != MOBILEGPT_SERVICE
            ]
            self._adb(
                "shell",
                "settings",
                "put",
                "secure",
                "enabled_accessibility_services",
                ":".join(services),
            )
            self._adb("shell", "am", "force-stop", MOBILEGPT_PACKAGE)
            stop_deadline = time.monotonic() + 5.0
            while time.monotonic() < stop_deadline:
                accessibility = self._adb(
                    "shell",
                    "dumpsys",
                    "accessibility",
                ).stdout
                if "Service[label=MobileGPT Accessibility" not in accessibility:
                    break
                time.sleep(0.25)
            time.sleep(0.5)
            services.append(MOBILEGPT_SERVICE)
            self._adb(
                "shell",
                "settings",
                "put",
                "secure",
                "enabled_accessibility_services",
                ":".join(services),
            )
            self._adb("shell", "settings", "put", "secure", "accessibility_enabled", "1")
            self._adb("shell", "am", "start", "-n", MOBILEGPT_ACTIVITY)
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                accessibility = self._adb(
                    "shell",
                    "dumpsys",
                    "accessibility",
                ).stdout
                if "Service[label=MobileGPT Accessibility" in accessibility:
                    break
                time.sleep(0.25)
            time.sleep(1.0)
            if serial_file_text:
                serial_file = Path(serial_file_text).expanduser().resolve()
                serial_file.parent.mkdir(parents=True, exist_ok=True)
                serial_file.write_text(serial + "\n", encoding="utf-8")

        def _broadcast(self, goal: str) -> subprocess.CompletedProcess[str]:
            self._adb("shell", "am", "start", "-n", MOBILEGPT_ACTIVITY)
            device_model = self._adb(
                "shell", "getprop", "ro.product.model"
            ).stdout.strip()
            wire_goal = _device_compatible_goal(goal, device_model)
            shell_command = shlex.join(
                [
                    "am",
                    "broadcast",
                    "-a",
                    MOBILEGPT_ACTION,
                    "--es",
                    MOBILEGPT_INSTRUCTION_EXTRA,
                    wire_goal,
                ]
            )
            return self._adb("shell", shell_command)

        def step(self, goal: str):
            if self.attempted:
                return make_agent_result(
                    done=True,
                    data={"summary": "MobileGPT did not finish", "source": self.name},
                )
            self.attempted = True
            self._prepare()
            started_before = _event_count(stats_path, "task_started")
            finished_before = _event_count(stats_path, "task_finished")
            answers_before = _event_count(stats_path, "agent_answer")
            broadcast = self._broadcast(goal)
            started = _wait_for_event(
                stats_path,
                "task_started",
                started_before,
                start_timeout,
            )
            for _ in range(max(0, rebroadcast_limit)):
                if started:
                    break
                self._prepare()
                broadcast = self._broadcast(goal)
                started = _wait_for_event(
                    stats_path,
                    "task_started",
                    started_before,
                    start_timeout,
                )
            finished = started and _wait_for_event(
                stats_path,
                "task_finished",
                finished_before,
                finish_timeout,
            )
            answer = (
                _event_text_after(
                    stats_path,
                    "agent_answer",
                    answers_before,
                )
                if finished
                else ""
            )
            if answer:
                self.env.interaction_cache = answer
            error = "" if finished else "task_finished_timeout" if started else "task_started_timeout"
            return make_agent_result(
                done=True,
                data={
                    "summary": "MobileGPT finished" if finished else error,
                    "source": self.name,
                    "error": error or None,
                    "answer": answer or None,
                    "broadcast_returncode": int(broadcast.returncode),
                    "actions_executed": 0,
                },
            )

    return MobileGPTAndroidWorldAgent()


__all__ = ["build_mobilegpt_agent"]
