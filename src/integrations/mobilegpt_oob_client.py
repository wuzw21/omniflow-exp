"""MobileGPT socket client using OmniFlow's canonical OOB physical layer.

The MobileGPT Server and planner remain upstream-compatible.  This client
implements only the small wire protocol expected by that Server and routes
every observation and device action through :class:`OobControlClient`.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from typing import Any
import xml.etree.ElementTree as ET

from PIL import Image

from src.integrations.android_world.oob_control import OobControlClient


def _require_oob_backend() -> None:
    backend = str(
        os.environ.get("OMNIFLOW_ANDROIDWORLD_CONTROL_BACKEND") or "oob"
    ).strip().lower()
    if backend != "oob":
        raise RuntimeError(f"mobilegpt_oob_backend_required:{backend or 'unset'}")


def _decode_image(value: Any) -> bytes:
    encoded = str(value or "").strip()
    if encoded.startswith("data:image/") and "," in encoded:
        encoded = encoded.split(",", 1)[1]
    if not encoded:
        return b""
    try:
        raw = base64.b64decode(encoded, validate=False)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=95)
        return output.getvalue()
    except (binascii.Error, OSError, ValueError):
        return b""


def _ensure_mobilegpt_indices(xml: str) -> str:
    """Add the node indices expected by the official MobileGPT protocol."""

    root = ET.fromstring(xml)
    for index, node in enumerate(root.iter("node")):
        node.set("index", str(index))
    return ET.tostring(root, encoding="unicode")


def _run_adb(adb_path: str, serial: str, args: list[str]) -> str:
    result = subprocess.run(
        [adb_path, "-s", serial, *args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=30.0,
    )
    return str(result.stdout or "")


def _installed_packages(adb_path: str, serial: str) -> list[str]:
    packages: list[str] = []
    for line in _run_adb(adb_path, serial, ["shell", "pm", "list", "packages"]).splitlines():
        package = line.strip().removeprefix("package:").strip()
        if package:
            packages.append(package)
    return packages


def _wire_packages(adb_path: str, serial: str) -> list[str]:
    target_package = str(os.environ.get("MOBILEGPT_TARGET_PACKAGE") or "").strip()
    return [target_package] if target_package else _installed_packages(adb_path, serial)


def _launch_selected_package(
    oob: OobControlClient,
    adb_path: str,
    serial: str,
    selected: str,
    *,
    timeout_sec: float,
) -> str:
    target_package = str(os.environ.get("MOBILEGPT_TARGET_PACKAGE") or "").strip()
    installed = set(_installed_packages(adb_path, serial))
    candidate = str(selected or "").strip()
    if candidate not in installed:
        candidate = target_package
    if not candidate or candidate not in installed:
        raise RuntimeError("mobilegpt_target_app_package_unresolved")
    # The OOB resident keeps the most recent observation as the action
    # precondition.  Prime that state before the first open_app action; the
    # native AndroidWorld host normally does this in its observation/action
    # lifecycle, while this socket adapter owns that lifecycle directly.
    oob.observe(wait_to_stabilize=True)
    oob.act({"tool": "open_app", "args": {"package_name": candidate}})
    launch_budget = max(1.0, float(timeout_sec))
    started = time.monotonic()
    deadline = started + launch_budget
    relaunch_at = started + launch_budget / 2.0
    relaunched = False
    while True:
        now = time.monotonic()
        if now >= deadline:
            break
        snapshot = oob.observe(wait_to_stabilize=True)
        observed = str(snapshot.get("package_name") or "").strip()
        if observed == candidate:
            return candidate
        if _dismiss_oob_permission_dialog(oob, snapshot):
            time.sleep(0.25)
            continue
        # A freshly booted emulator can finish its own launcher transition
        # after the first OOB open_app action and overwrite that launch.  Retry
        # the same OOB action once inside the bounded startup handshake.  This
        # never falls back to adb, AndroidWorld actions, or MobileGPT's client.
        if not relaunched and now >= relaunch_at:
            oob.act({"tool": "open_app", "args": {"package_name": candidate}})
            relaunched = True
        time.sleep(0.25)
    raise RuntimeError(
        "mobilegpt_target_app_not_ready:"
        + candidate
        + ":foreground="
        + (str(oob.observe(wait_to_stabilize=True).get("package_name") or "unknown"))
    )


def _prelaunch_target_package(
    oob: OobControlClient,
    adb_path: str,
    serial: str,
    *,
    timeout_sec: float,
) -> str:
    """Launch the benchmark target through OOB before Planner handshaking."""

    target_package = str(os.environ.get("MOBILEGPT_TARGET_PACKAGE") or "").strip()
    if not target_package:
        raise RuntimeError("mobilegpt_target_app_package_unresolved")
    return _launch_selected_package(
        oob,
        adb_path,
        serial,
        target_package,
        timeout_sec=timeout_sec,
    )


def _normalised_point(bounds: str, display: dict[str, Any]) -> tuple[int, int]:
    values = bounds.replace("][", ",").strip("[]").split(",")
    if len(values) != 4:
        raise ValueError("mobilegpt_oob_bounds_invalid")
    left, top, right, bottom = (int(value) for value in values)
    width = max(1, int(display.get("width") or right or 1))
    height = max(1, int(display.get("height") or bottom or 1))
    return (
        max(0, min(1000, round(((left + right) / 2) / width * 1000))),
        max(0, min(1000, round(((top + bottom) / 2) / height * 1000))),
    )


def _dismiss_oob_permission_dialog(
    oob: OobControlClient,
    snapshot: dict[str, Any],
) -> bool:
    """Dismiss Android permission prompts through the OOB executor."""

    if not str(snapshot.get("package_name") or "").strip().endswith(
        ".permissioncontroller"
    ):
        return False
    try:
        root = ET.fromstring(str(snapshot.get("xml") or ""))
    except ET.ParseError:
        return False
    for element in root.iter():
        resource_id = str(element.attrib.get("resource-id") or "").strip()
        label = " ".join(
            (
                str(element.attrib.get("text") or "").strip(),
                str(element.attrib.get("content-desc") or "").strip(),
            )
        ).casefold()
        is_deny = "permission_deny" in resource_id.casefold() or label in {
            "deny",
            "don't allow",
            "don’t allow",
            "do not allow",
        }
        bounds = str(element.attrib.get("bounds") or "").strip()
        if not is_deny or not bounds:
            continue
        point = _normalised_point(bounds, snapshot.get("display") or {})
        oob.act({"tool": "click", "args": {"x": point[0], "y": point[1]}})
        return True
    return False


def _stats_terminal_reason(stats_path: Path | None) -> str:
    """Read explicit provider telemetry without interpreting prompt text."""

    if stats_path is None or not stats_path.is_file():
        return ""
    for line in reversed(
        stats_path.read_text(encoding="utf-8", errors="replace").splitlines()
    ):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("event") == "chat_empty_or_invalid":
            return "mobilegpt_server_no_action"
        if event.get("event") in {"chat_call", "action_sent", "task_finished"}:
            return ""
    return ""


def _is_oob_environment_failure(reason: str) -> bool:
    """Classify physical OOB/setup failures separately from planner errors."""

    value = str(reason or "").strip()
    if value.startswith("mobilegpt_oob_action_"):
        # These failures occur after a valid OOB observation when the official
        # Planner response cannot be represented as an executable action.
        # They are method/protocol conclusions, not device or OOB failures.
        return False
    return value.startswith((
        "oob_",
        "mobilegpt_oob_",
        "mobilegpt_target_app_",
    ))


def _official_task_instruction(
    task: Any,
    *,
    requested_instruction: str,
    task_name: str,
) -> str:
    """Use the goal belonging to the evaluated AndroidWorld task instance."""

    return str(
        getattr(task, "goal", "")
        or requested_instruction
        or task_name
    ).strip()


def _action_with_bounds(action: dict[str, Any], xml: str) -> dict[str, Any]:
    """Attach the current OOB node bounds to an official index action."""

    name = str(action.get("name") or "").strip().lower()
    if name not in {"click", "long-click", "long_click", "input"}:
        return action
    parameters = action.get("parameters")
    if not isinstance(parameters, dict) or parameters.get("oob_bounds"):
        return action
    wanted = str(
        parameters.get("index")
        or parameters.get("node_index")
        or parameters.get("target_index")
        or ""
    ).strip()
    if not wanted:
        return action
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return action
    for element in root.iter():
        if str(element.attrib.get("index") or "").strip() != wanted:
            continue
        bounds = str(element.attrib.get("bounds") or "").strip()
        if not bounds:
            continue
        enriched = dict(action)
        enriched["parameters"] = dict(parameters)
        enriched["parameters"]["oob_bounds"] = bounds
        return enriched
    return action


def _oob_action(
    oob: OobControlClient,
    action: dict[str, Any],
    display: dict[str, Any],
    xml: str,
) -> None:
    action = _action_with_bounds(action, xml)
    name = str(action.get("name") or "").strip().lower()
    args = action.get("parameters")
    if not isinstance(args, dict):
        args = {}
    if name in {"click", "long-click", "long_click"}:
        bounds = str(args.get("oob_bounds") or "").strip()
        if bounds:
            point = _normalised_point(bounds, display)
        elif args.get("x") is not None and args.get("y") is not None:
            point = (int(float(args["x"])), int(float(args["y"])))
        else:
            raise RuntimeError("mobilegpt_oob_action_target_missing")
        tool = "long_press" if name in {"long-click", "long_click"} else "click"
        payload: dict[str, Any] = {"x": point[0], "y": point[1]}
        if tool == "long_press":
            payload["duration_ms"] = 1000
        oob.act({"tool": tool, "args": payload})
        return
    if name == "input":
        bounds = str(args.get("oob_bounds") or "").strip()
        if bounds:
            point = _normalised_point(bounds, display)
            oob.act({"tool": "click", "args": {"x": point[0], "y": point[1]}})
            # Refresh the OOB action precondition after focusing the field.
            # This is required by the resident executor before the following
            # input_text action and does not alter the official planner flow.
            observe = getattr(oob, "observe", None)
            if callable(observe):
                observe(wait_to_stabilize=True)
        oob.act(
            {
                "tool": "input_text",
                "args": {
                    "text": str(
                        args.get("input_text")
                        or args.get("text")
                        or args.get("value")
                        or ""
                    ),
                    "clear_text": True,
                },
            }
        )
        return
    if name in {"back", "go-back", "navigate_back", "press_back"}:
        oob.act({"tool": "press_key", "args": {"key": "back"}})
        return
    if name in {"home", "navigate_home", "press_home"}:
        oob.act({"tool": "press_key", "args": {"key": "home"}})
        return
    if name in {"enter", "navigate_enter", "press_enter"}:
        oob.act({"tool": "press_key", "args": {"key": "enter"}})
        return
    if name == "scroll":
        direction = str(args.get("direction") or "down").strip().lower()
        y1, y2 = (750, 250) if direction in {"up", "forward"} else (250, 750)
        oob.act(
            {
                "tool": "swipe",
                "args": {"x1": 500, "y1": y1, "x2": 500, "y2": y2, "duration_ms": 400},
            }
        )
        return
    if name in {"speak", "read_screen"}:
        return
    if name in {"wait", "sleep"}:
        oob.act({"tool": "wait", "args": {"seconds": float(args.get("seconds") or 1.0)}})
        return
    raise RuntimeError(f"mobilegpt_oob_action_unsupported:{name}")


def _run_mobilegpt_oob_transport(
    *,
    serial: str,
    adb_path: str,
    server_host: str,
    server_port: int,
    instruction: str,
    timeout_sec: float,
    max_steps: int,
    output_root: str | Path,
    server_log_path: str | Path = "",
) -> dict[str, Any]:
    """Run one MobileGPT episode over the OOB observe/act transport."""

    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    lines = ["[omniflow] MobileGPT OOB client starting"]
    started = time.monotonic()
    actions = 0
    planner_steps = 0
    reason = ""
    task_finished = False
    oob = OobControlClient(None, adb_serial=serial, adb_path=adb_path)
    server_log = Path(server_log_path).expanduser() if str(server_log_path).strip() else None
    stats_value = str(os.environ.get("MOBILEGPT_STATS_JSONL") or "").strip()
    stats_path = Path(stats_value).expanduser() if stats_value else None
    app_ready_timeout_sec = float(os.environ.get("MOBILEGPT_APP_READY_TIMEOUT_SEC", "20") or 20)
    finish_stall_timeout_sec = float(os.environ.get("MOBILEGPT_FINISH_STALL_TIMEOUT_SEC", "8") or 8)
    finish_stall_started: float | None = None
    last_finish_count = 0
    last_action_count = 0

    connect_host = str(os.environ.get("MOBILEGPT_OOB_SERVER_HOST") or "127.0.0.1")

    try:
        _require_oob_backend()
        launched_package = _prelaunch_target_package(
            oob,
            adb_path,
            serial,
            timeout_sec=app_ready_timeout_sec,
        )
        lines.append(f"[omniflow] OOB target ready={launched_package}")
        with socket.create_connection((connect_host, int(server_port)), timeout=30.0) as sock:
            sock.settimeout(1.0)

            def send(payload: bytes) -> None:
                sock.sendall(payload)

            def send_line(prefix: str, text: str) -> None:
                send(prefix.encode("utf-8") + text.encode("utf-8") + b"\n")

            def receive_line() -> str:
                nonlocal finish_stall_started, last_finish_count
                value = bytearray()
                deadline = time.monotonic() + 90.0
                while time.monotonic() < deadline:
                    try:
                        chunk = sock.recv(1)
                    except socket.timeout:
                        if finish_stall_started is not None and time.monotonic() - finish_stall_started >= max(1.0, finish_stall_timeout_sec):
                            return "$$$$$"
                        if server_log is not None and server_log.is_file():
                            server_text = server_log.read_text(encoding="utf-8", errors="replace")[-12000:]
                            if "Traceback (most recent call last)" in server_text:
                                raise RuntimeError("mobilegpt_server_handler_failed")
                            terminal_reason = _stats_terminal_reason(stats_path)
                            if terminal_reason:
                                raise RuntimeError(terminal_reason)
                            # The pinned Server logs ``finish subtask!!`` when
                            # its official agent has completed the request,
                            # but some versions do not emit the final socket
                            # sentinel afterwards.  Do not change the
                            # Planner/Executor decision; close only this
                            # transport stall after the server has reported
                            # completion and no new action arrived.
                            finish_count = server_text.count("finish subtask!!")
                            if finish_count > last_finish_count:
                                last_finish_count = finish_count
                                if finish_stall_started is None:
                                    finish_stall_started = time.monotonic()
                        continue
                    if not chunk:
                        raise RuntimeError("mobilegpt_oob_server_closed")
                    value.extend(chunk)
                    if chunk == b"\n":
                        return value.decode("utf-8", errors="replace").strip()
                raise RuntimeError("mobilegpt_oob_server_response_timeout")

            send_line("L", "##".join(_wire_packages(adb_path, serial)))
            send_line("I", instruction)
            selected = receive_line()
            lines.append(f"[omniflow] server selected app frame={selected[:200]}")
            if not selected.startswith("##$$##"):
                raise RuntimeError("mobilegpt_oob_app_selection_missing")
            selected_package = selected[6:].strip()
            if selected_package != launched_package:
                raise RuntimeError(
                    "mobilegpt_oob_selected_package_mismatch:"
                    f"expected={launched_package}:selected={selected_package}"
                )

            while time.monotonic() - started < max(1.0, float(timeout_sec)):
                snapshot = oob.observe(wait_to_stabilize=True)
                xml = str(snapshot.get("xml") or "")
                if not xml.strip():
                    raise RuntimeError("mobilegpt_oob_observation_xml_missing")
                xml = _ensure_mobilegpt_indices(xml)
                image = _decode_image(snapshot.get("image_base64"))
                send(b"S" + str(len(image)).encode("ascii") + b"\n" + image)
                encoded_xml = xml.encode("utf-8")
                send(b"X" + str(len(encoded_xml)).encode("ascii") + b"\n" + encoded_xml)
                response = receive_line()
                if response == "$$$$$":
                    task_finished = True
                    lines.append("[omniflow] Task finished")
                    break
                if response.startswith("##$$##"):
                    _launch_selected_package(
                        oob,
                        adb_path,
                        serial,
                        response[6:].strip(),
                        timeout_sec=app_ready_timeout_sec,
                    )
                    continue
                if response.startswith("$$##$$"):
                    if server_log is not None and server_log.is_file():
                        server_text = server_log.read_text(encoding="utf-8", errors="replace")[-12000:]
                        finish_count = server_text.count("finish subtask!!")
                        if finish_count > last_finish_count:
                            last_finish_count = finish_count
                            if finish_stall_started is None:
                                finish_stall_started = time.monotonic()
                    if finish_stall_started is not None and time.monotonic() - finish_stall_started >= max(1.0, finish_stall_timeout_sec):
                        task_finished = True
                        lines.append("[omniflow] Task finished (official server completion keep-alive)")
                        break
                    continue
                try:
                    action = json.loads(response)
                except json.JSONDecodeError as error:
                    raise RuntimeError("mobilegpt_oob_action_json_invalid") from error
                if not isinstance(action, dict):
                    raise RuntimeError("mobilegpt_oob_action_not_object")
                planner_steps += 1
                if str(action.get("name") or "").strip().lower() == "finish":
                    task_finished = True
                    lines.append("[omniflow] Task finished (finish action)")
                    break
                lines.append(
                    "[omniflow] OOB planned action="
                    + json.dumps(action, ensure_ascii=False, sort_keys=True)
                )
                _oob_action(
                    oob,
                    action,
                    snapshot.get("display") or {},
                    xml,
                )
                if str(action.get("name") or "").strip().lower() not in {"speak", "read_screen"}:
                    actions += 1
                lines.append(f"[omniflow] OOB action={action.get('name')}")
                finish_stall_started = None
                last_action_count = actions
                if max_steps > 0 and planner_steps >= max_steps:
                    reason = "mobilegpt_step_budget_exhausted"
                    break
            else:
                reason = "mobilegpt_episode_timeout"
    except (OSError, RuntimeError, ValueError) as error:
        reason = str(error)
        lines.append(f"[omniflow] {reason}")

    if not task_finished and not reason:
        reason = "mobilegpt_oob_episode_incomplete"
    client_log = "\n".join(lines) + "\n"
    (output / "client_log.txt").write_text(client_log, encoding="utf-8")
    return {
        "returncode": 0 if task_finished else 124 if reason == "mobilegpt_episode_timeout" else 1,
        "reason": reason,
        "task_finished": task_finished,
        "actions": actions,
        "planner_steps": planner_steps,
        "log": client_log,
        "server_host": connect_host,
        "server_port": int(server_port),
    }


def _mobilegpt_stats(stats_path: Path) -> dict[str, int]:
    values = {
        "model_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    if not stats_path.is_file():
        return values
    for line in stats_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("event") not in {
            "chat_call",
            "embedding_call",
        }:
            continue
        values["model_calls"] += 1
        values["prompt_tokens"] += int(event.get("prompt_tokens") or 0)
        values["completion_tokens"] += int(event.get("completion_tokens") or 0)
        values["total_tokens"] += int(event.get("total_tokens") or 0)
    return values


def run_mobilegpt_oob_client(
    *,
    serial: str,
    adb_path: str,
    server_host: str,
    server_port: int,
    instruction: str,
    timeout_sec: float,
    max_steps: int,
    output_root: str | Path,
    server_log_path: str | Path = "",
    android_world_root: str | Path = "",
    task_name: str = "",
    task_params_json: str = "{}",
    task_seed: int = 113,
    console_port: int = 5560,
    grpc_port: int = 8560,
    perform_emulator_setup: bool = True,
) -> dict[str, Any]:
    """Run official MobileGPT planning with the canonical OOB lifecycle."""

    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    def run_episode(episode_instruction: str = instruction) -> dict[str, Any]:
        return _run_mobilegpt_oob_transport(
            serial=serial,
            adb_path=adb_path,
            server_host=server_host,
            server_port=server_port,
            instruction=episode_instruction,
            timeout_sec=timeout_sec,
            max_steps=max_steps,
            output_root=output,
            server_log_path=server_log_path,
        )

    if android_world_root and task_name:
        from src.integrations.official_forward import _androidworld_task_startup

        startup = _androidworld_task_startup(
            android_world_root=android_world_root,
            task_name=task_name,
            task_params_json=task_params_json,
            task_seed=task_seed,
            console_port=console_port,
            grpc_port=grpc_port,
            adb_path=adb_path,
            perform_emulator_setup=perform_emulator_setup,
            use_uiautomator=False,
        )
        with startup as (env, task):
            official_instruction = _official_task_instruction(
                task,
                requested_instruction=instruction,
                task_name=task_name,
            )
            result = run_episode(official_instruction)
            params = json.loads(str(task_params_json or "{}"))
            reward = float(task.is_successful(env))
            stats = _mobilegpt_stats(
                Path(os.environ.get("MOBILEGPT_STATS_JSONL", "")).expanduser()
            )
            result_row = {
                "schema_version": "omniflow.androidworld.result.v1",
                "task_name": task_name,
                "task": task_name,
                "goal": official_instruction,
                "requested_instruction": instruction,
                "official_task_instruction": official_instruction,
                "task_params": params,
                "task_params_sha256": hashlib.sha256(
                    json.dumps(params, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "method": "mobilegpt",
                "device": serial,
                "task_random_seed": int(task_seed),
                "fixed_task_seed": True,
                "fixed_task_params": True,
                "official_validator_used": True,
                "official_validator_success": reward > 0.5,
                "official_validator_coverage_rate": 1.0,
                "androidworld_validator_result": {
                    "validator": "androidworld_official",
                    "success": reward > 0.5,
                    "reward": reward,
                },
                "process_returncode": int(result.get("returncode") or 1),
                "classification": (
                    "success"
                    if reward > 0.5
                    else "environment_failure"
                    if _is_oob_environment_failure(str(result.get("reason") or ""))
                    else "method_failure"
                ),
                "actions_executed": int(result.get("actions") or 0),
                "planner_steps": int(result.get("planner_steps") or 0),
                **stats,
                "token_usage_status": "tracked" if stats["model_calls"] else "unavailable",
                "fallback_steps": 0,
                "mobilegpt_stats_jsonl": os.environ.get("MOBILEGPT_STATS_JSONL", ""),
                "mobilegpt_protocol": {
                    "transport": "oob_control",
                    "server_host": str(server_host),
                    "server_port": int(server_port),
                    "task_finished": bool(result.get("task_finished")),
                },
                "environment_failure": reward <= 0.5
                and _is_oob_environment_failure(str(result.get("reason") or "")),
                "failure_reason": str(result.get("reason") or ""),
                "runtime_integrity_error": str(result.get("reason") or ""),
            }
            (output / "task_results.jsonl").write_text(
                json.dumps(result_row, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            result["validator_success"] = reward > 0.5
            result["task_results"] = str(output / "task_results.jsonl")
            return result
    return run_episode()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="MobileGPT official planner over OmniFlow OOB.")
    parser.add_argument("--serial", required=True)
    parser.add_argument("--adb", required=True)
    parser.add_argument("--server-host", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--server-log", default="")
    parser.add_argument("--android-world-root", default="")
    parser.add_argument("--task", default="")
    parser.add_argument("--task-params-json", default="{}")
    parser.add_argument("--task-seed", type=int, default=113)
    parser.add_argument("--console-port", type=int, default=5560)
    parser.add_argument("--grpc-port", type=int, default=8560)
    parser.add_argument("--no-perform-emulator-setup", action="store_true")
    args = parser.parse_args(argv)
    result = run_mobilegpt_oob_client(
        serial=args.serial,
        adb_path=args.adb,
        server_host=args.server_host,
        server_port=args.server_port,
        instruction=args.instruction,
        timeout_sec=args.timeout,
        max_steps=args.max_steps,
        output_root=args.output,
        server_log_path=args.server_log,
        android_world_root=args.android_world_root,
        task_name=args.task,
        task_params_json=args.task_params_json,
        task_seed=args.task_seed,
        console_port=args.console_port,
        grpc_port=args.grpc_port,
        perform_emulator_setup=not args.no_perform_emulator_setup,
    )
    return 0 if result.get("validator_success") or result.get("task_finished") else int(result.get("returncode") or 1)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_mobilegpt_oob_client"]
