from __future__ import annotations

import base64
import importlib
import io
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from typing import Any, Callable
import xml.etree.ElementTree as ET

from omniflow.core.model import Action, ActionResult, RunResult
from src.experiment.performance_metrics import PerformanceMetrics
from src.experiment.protocol import MAX_STEPS
from src.integrations.android_world.host import AndroidWorldHost, make_agent_result
from src.integrations.mobilegpt_runtime import (
    mobilegpt_compatible_xml,
    normalize_mobilegpt_action,
)

_BOUNDS_PATTERN = re.compile(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]")


def _wire_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _socket_timeout(value: float) -> float | None:
    return None if value < 0 else max(0.1, value)


def _connect_with_retry(host: str, port: int, timeout: float) -> socket.socket:
    if timeout < 0:
        return socket.create_connection((host, port), timeout=None)
    deadline = time.monotonic() + max(0.1, timeout)
    last_error: OSError | None = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if last_error is not None:
                raise last_error
            raise TimeoutError("mobilegpt_server_start_timeout")
        try:
            return socket.create_connection(
                (host, port),
                timeout=min(1.0, max(0.1, remaining)),
            )
        except OSError as error:
            last_error = error
            time.sleep(min(0.1, remaining))


def _jpeg_payload(image_base64: str | None) -> bytes:
    if not str(image_base64 or "").strip():
        raise ValueError("mobilegpt_androidworld_state_screenshot_empty")
    try:
        raw = base64.b64decode(str(image_base64), validate=True)
        image_module = importlib.import_module("PIL.Image")
        image = image_module.open(io.BytesIO(raw))
        image.load()
        if image.mode != "RGB":
            image = image.convert("RGB")
        output = io.BytesIO()
        image.save(output, format="JPEG")
        return output.getvalue()
    except Exception as error:
        raise ValueError("mobilegpt_androidworld_state_screenshot_invalid") from error


def _bounds_for_index(xml_text: str, index: Any) -> tuple[int, int, int, int]:
    target_index = str(index).strip()
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as error:
        raise ValueError(f"mobilegpt_native_xml_invalid:{error}") from error
    target = next(
        (
            element
            for element in root.iter()
            if str(element.attrib.get("index") or "").strip() == target_index
        ),
        None,
    )
    if target is None:
        raise ValueError(f"mobilegpt_action_index_missing:{target_index}")
    match = _BOUNDS_PATTERN.fullmatch(
        str(target.attrib.get("bounds") or "").strip()
    )
    if match is None:
        raise ValueError(f"mobilegpt_action_bounds_missing:{target_index}")
    return tuple(int(value) for value in match.groups())


def _center(bounds: tuple[int, int, int, int]) -> tuple[int, int]:
    left, top, right, bottom = bounds
    return (left + right) // 2, (top + bottom) // 2


def _display_from_xml(xml_text: str) -> tuple[int, int]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as error:
        raise ValueError(f"mobilegpt_native_xml_invalid:{error}") from error
    parsed = [
        match
        for element in root.iter()
        if (
            match := _BOUNDS_PATTERN.fullmatch(
                str(element.attrib.get("bounds") or "").strip()
            )
        )
    ]
    if not parsed:
        raise ValueError("mobilegpt_native_display_missing")
    width = max(int(match.group(3)) for match in parsed)
    height = max(int(match.group(4)) for match in parsed)
    if width <= 0 or height <= 0:
        raise ValueError("mobilegpt_native_display_invalid")
    return width, height


def _indexed_app_ui_count(xml_text: str, package_name: str = "") -> int:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return 0
    count = 0
    for element in root.iter():
        attributes = element.attrib
        if not str(attributes.get("index") or "").strip():
            continue
        element_package = str(attributes.get("package") or "").strip()
        if package_name and element_package != package_name:
            continue
        if element_package == "com.android.systemui":
            continue
        has_identity = any(
            str(attributes.get(key) or "").strip()
            for key in ("resource-id", "text", "content-desc")
        )
        has_action = any(
            str(attributes.get(key) or "").strip().lower() == "true"
            for key in ("clickable", "editable", "scrollable", "long-clickable")
        )
        if has_identity or has_action:
            count += 1
    return count


def _write_stats_event(event: dict[str, Any]) -> None:
    stats_path = str(os.environ.get("MOBILEGPT_STATS_JSONL") or "").strip()
    if not stats_path:
        return
    try:
        output = Path(stats_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(event)
        payload.setdefault("ts_unix", time.time())
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def build_mobilegpt_agent(
    *,
    env: Any | None = None,
    host: Any | None = None,
    evidence_root: str | Path | None = None,
    action_factory: Callable[..., Any] | None = None,
    performance_metrics: PerformanceMetrics | None = None,
) -> Any:
    server_host = str(os.environ.get("MOBILEGPT_SERVER_HOST") or "127.0.0.1").strip()
    server_port = int(os.environ.get("MOBILEGPT_SERVER_PORT") or 12345)
    connect_timeout = float(os.environ.get("MOBILEGPT_WAIT_START_TIMEOUT_SEC") or 60.0)
    response_timeout = float(os.environ.get("MOBILEGPT_WAIT_FINISH_TIMEOUT_SEC") or 120.0)
    post_action_wait = max(
        0.0,
        float(os.environ.get("MOBILEGPT_POST_ACTION_WAIT_SEC") or 1.0),
    )
    app_ready_timeout = max(
        0.0,
        float(os.environ.get("MOBILEGPT_APP_READY_TIMEOUT_SEC") or 15.0),
    )
    app_ready_poll = max(
        0.0,
        float(os.environ.get("MOBILEGPT_APP_READY_POLL_SEC") or 0.25),
    )
    target_package = str(os.environ.get("MOBILEGPT_TARGET_PACKAGE") or "").strip()
    packages = [
        package.strip()
        for package in re.split(
            r"##|,",
            str(os.environ.get("MOBILEGPT_APP_PACKAGES") or target_package),
        )
        if package.strip()
    ]
    if env is None and host is None:
        raise ValueError("mobilegpt_environment_or_host_required")
    canonical_host = host is not None
    runtime_host = host or AndroidWorldHost(
        env,
        evidence_root=evidence_root,
        performance_metrics=performance_metrics,
    )
    upstream_mode = str(
        os.environ.get("MOBILEGPT_UPSTREAM_MODE") or ""
    ).strip().lower() in {"1", "true", "yes", "on"}

    class MobileGPTAndroidWorldAgent:
        name = "mobilegpt"
        transition_pause = 0.0

        def __init__(self) -> None:
            self.env = env
            self.max_steps = MAX_STEPS
            self.attempted = False
            self.last_result_data: dict[str, Any] = {}

        def reset(self, go_home: bool = False) -> None:
            self.attempted = False
            if self.env is not None:
                self.env.reset(go_home=go_home)

        def set_max_steps(self, max_steps: int) -> None:
            self.max_steps = max(1, int(max_steps))

        def _new_action(self, **payload: Any) -> Any:
            if action_factory is not None:
                return action_factory(**payload)
            json_action = importlib.import_module("android_world.env.json_action")
            return json_action.JSONAction(**payload)

        def _execute(self, *, xml_text: str = "", **payload: Any) -> None:
            if not canonical_host:
                self.env.execute_action(self._new_action(**payload))
                return
            action_type = str(payload.get("action_type") or "").strip()
            if action_type == "open_app":
                action = Action(
                    "open_app",
                    {"package_name": str(payload.get("app_name") or "")},
                )
            elif action_type in {
                "click",
                "double_tap",
                "long_press",
                "input_text",
            }:
                width, height = _display_from_xml(xml_text)
                x = round(float(payload.get("x")) * 1000.0 / width)
                y = round(float(payload.get("y")) * 1000.0 / height)
                args: dict[str, Any] = {"x": x, "y": y}
                if action_type == "input_text":
                    args.update(
                        text=str(payload.get("text") or ""),
                        clear_text=bool(payload.get("clear_text", True)),
                    )
                action = Action(action_type, args)
            elif action_type == "scroll":
                direction = str(payload.get("direction") or "").strip().lower()
                gestures = {
                    "up": {"x1": 500, "y1": 800, "x2": 500, "y2": 200},
                    "down": {"x1": 500, "y1": 200, "x2": 500, "y2": 800},
                    "left": {"x1": 800, "y1": 500, "x2": 200, "y2": 500},
                    "right": {"x1": 200, "y1": 500, "x2": 800, "y2": 500},
                }
                if direction not in gestures:
                    raise ValueError(
                        f"mobilegpt_scroll_direction_invalid:{direction or 'missing'}"
                    )
                action = Action("swipe", gestures[direction])
            elif action_type == "navigate_back":
                action = Action("press_key", {"key": "back"})
            elif action_type == "answer":
                action = Action("answer", {"text": str(payload.get("text") or "")})
            else:
                raise ValueError(
                    f"mobilegpt_canonical_action_unsupported:{action_type or 'missing'}"
                )
            result = ActionResult.from_value(runtime_host.act(action))
            if not result.success:
                raise RuntimeError(result.error or "mobilegpt_canonical_action_failed")

        def _send_line(self, stream: Any, prefix: str, value: str) -> None:
            stream.write(f"{prefix}{_wire_text(value)}\n".encode())

        def _send_xml(self, stream: Any, xml_text: str) -> None:
            payload = xml_text.encode("utf-8")
            stream.write(f"X{len(payload)}\n".encode())
            stream.write(payload)

        def _send_screenshot(self, stream: Any, image_base64: str | None) -> None:
            payload = _jpeg_payload(image_base64)
            stream.write(f"S{len(payload)}\n".encode())
            stream.write(payload)

        def _read_line(self, stream: Any) -> str:
            raw = stream.readline()
            if not raw:
                raise ConnectionError("mobilegpt_server_closed_connection")
            return raw.decode("utf-8").strip()

        def _observe_ready_app(self, package: str) -> tuple[Any, str]:
            started_at = time.monotonic()
            deadline = started_at + app_ready_timeout
            expected_package = package
            attempts = 0
            observed_package = ""
            indexed_app_nodes = 0
            while True:
                attempts += 1
                observation = runtime_host.observe(
                    xml=True,
                    screenshot=True,
                    app_info=True,
                )
                observed_package = str(observation.package_name or "").strip()
                raw_xml = str(observation.xml or "").strip()
                xml_text = mobilegpt_compatible_xml(raw_xml) if raw_xml else ""
                indexed_app_nodes = _indexed_app_ui_count(
                    xml_text,
                    expected_package if "." in expected_package else "",
                )
                package_matches = (
                    not observed_package
                    or "." not in expected_package
                    or observed_package == expected_package
                )
                if xml_text and indexed_app_nodes > 0 and package_matches:
                    _write_stats_event(
                        {
                            "event": "mobilegpt_app_ui_ready",
                            "expected_package": expected_package,
                            "observed_package": observed_package,
                            "attempts": attempts,
                            "indexed_app_nodes": indexed_app_nodes,
                            "wait_seconds": time.monotonic() - started_at,
                        }
                    )
                    return observation, xml_text
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    waited = time.monotonic() - started_at
                    _write_stats_event(
                        {
                            "event": "mobilegpt_app_ui_ready_timeout",
                            "expected_package": expected_package,
                            "observed_package": observed_package,
                            "attempts": attempts,
                            "indexed_app_nodes": indexed_app_nodes,
                            "wait_seconds": waited,
                        }
                    )
                    raise RuntimeError(
                        "mobilegpt_app_ui_not_ready:"
                        f"expected={expected_package}:"
                        f"observed={observed_package or 'unknown'}:"
                        f"indexed_app_nodes={indexed_app_nodes}:"
                        f"attempts={attempts}:"
                        f"wait_seconds={waited:.3f}"
                    )
                time.sleep(min(app_ready_poll, remaining))

        def _execute_server_action(
            self,
            action: Any,
            *,
            xml_text: str,
        ) -> tuple[bool, str]:
            normalized = action if upstream_mode else normalize_mobilegpt_action(action)
            if not isinstance(normalized, dict):
                raise ValueError("mobilegpt_action_not_object")
            name = str(normalized.get("name") or "").strip()
            parameters = normalized.get("parameters")
            parameters = dict(parameters) if isinstance(parameters, dict) else {}
            if name in {"click", "long-click", "repeat-click", "input"}:
                if parameters.get("index") is None:
                    raise ValueError(f"mobilegpt_action_index_required:{name}")
                x, y = _center(
                    _bounds_for_index(xml_text, parameters.get("index"))
                )
                if name == "click":
                    self._execute(
                        action_type="click",
                        x=x,
                        y=y,
                        xml_text=xml_text,
                    )
                elif name == "repeat-click":
                    try:
                        count = int(parameters.get("number") or 0)
                    except (TypeError, ValueError) as error:
                        raise ValueError("mobilegpt_repeat_click_count_invalid") from error
                    if count <= 0:
                        raise ValueError("mobilegpt_repeat_click_count_invalid")
                    for _ in range(count):
                        self._execute(
                            action_type="click",
                            x=x,
                            y=y,
                            xml_text=xml_text,
                        )
                elif name == "long-click":
                    self._execute(
                        action_type="long_press",
                        x=x,
                        y=y,
                        xml_text=xml_text,
                    )
                else:
                    self._execute(
                        action_type="input_text",
                        x=x,
                        y=y,
                        text=str(
                            parameters.get("input_text")
                            or parameters.get("text")
                            or ""
                        ),
                        clear_text=True,
                        xml_text=xml_text,
                    )
                return True, ""
            if name == "scroll":
                direction = str(parameters.get("direction") or "").strip().lower()
                if direction not in {"up", "down", "left", "right"}:
                    raise ValueError(
                        f"mobilegpt_scroll_direction_invalid:{direction or 'missing'}"
                    )
                self._execute(action_type="scroll", direction=direction)
                return True, ""
            if name in {"back", "go-back"}:
                self._execute(action_type="navigate_back")
                return True, ""
            if name == "speak":
                message = str(parameters.get("message") or "").strip()
                self._execute(action_type="answer", text=message)
                if self.env is not None:
                    self.env.interaction_cache = message
                return False, message
            if name == "ask":
                raise RuntimeError("mobilegpt_ask_has_no_androidworld_answer_source")
            raise ValueError(f"mobilegpt_action_unsupported:{name or 'missing'}")

        def _result(
            self,
            *,
            actions_executed: int,
            answer: str,
            error: str,
        ) -> Any:
            data = {
                "summary": (
                    "MobileGPT native AndroidWorld episode finished"
                    if not error
                    else error
                ),
                "source": self.name,
                "error": error or None,
                "answer": answer or None,
                "actions_executed": int(actions_executed),
                "state_backend": (
                    "canonical_host" if canonical_host else "androidworld"
                ),
                "action_backend": (
                    "canonical_host" if canonical_host else "androidworld"
                ),
                "native_androidworld_agent_io": not canonical_host,
            }
            self.last_result_data = dict(data)
            return make_agent_result(
                done=True,
                data=data,
            )

        def step(self, goal: str) -> Any:
            if self.attempted:
                return self._result(
                    actions_executed=0,
                    answer="",
                    error="mobilegpt_episode_already_attempted",
                )
            self.attempted = True
            actions_executed = 0
            answer = ""
            try:
                with _connect_with_retry(
                    server_host,
                    server_port,
                    connect_timeout,
                ) as connection:
                    connection.settimeout(_socket_timeout(response_timeout))
                    with connection.makefile("rwb", buffering=0) as stream:
                        self._send_line(stream, "L", "##".join(packages))
                        self._send_line(stream, "I", goal)
                        launch = self._read_line(stream)
                        if not launch.startswith("##$$##"):
                            raise RuntimeError(
                                f"mobilegpt_launch_response_invalid:{launch}"
                            )
                        launch_package = launch.removeprefix("##$$##").strip()
                        if not launch_package:
                            raise RuntimeError("mobilegpt_launch_package_missing")
                        self._execute(
                            action_type="open_app",
                            app_name=launch_package,
                        )
                        if post_action_wait:
                            time.sleep(post_action_wait)

                        ready_observation: tuple[Any, str] | None = (
                            self._observe_ready_app(launch_package)
                        )

                        while actions_executed < self.max_steps:
                            if ready_observation is not None:
                                observation, xml_text = ready_observation
                                ready_observation = None
                            else:
                                observation = runtime_host.observe(
                                    xml=True,
                                    screenshot=True,
                                    app_info=True,
                                )
                                xml_text = mobilegpt_compatible_xml(
                                    str(observation.xml or "").strip()
                                )
                            if not xml_text:
                                raise RuntimeError("mobilegpt_androidworld_state_xml_empty")
                            self._send_screenshot(stream, observation.image_base64)
                            self._send_xml(stream, xml_text)
                            while True:
                                response = self._read_line(stream)
                                if response == "$$$$$":
                                    return self._result(
                                        actions_executed=actions_executed,
                                        answer=answer,
                                        error="",
                                    )
                                if response.startswith("##$$##"):
                                    package = response.removeprefix("##$$##").strip()
                                    if not package:
                                        raise RuntimeError(
                                            "mobilegpt_launch_package_missing"
                                        )
                                    self._execute(
                                        action_type="open_app",
                                        app_name=package,
                                    )
                                    if post_action_wait:
                                        time.sleep(post_action_wait)
                                    ready_observation = self._observe_ready_app(package)
                                    break
                                try:
                                    action = json.loads(response)
                                    device_action, spoken = self._execute_server_action(
                                        action,
                                        xml_text=xml_text,
                                    )
                                except Exception as action_error:
                                    if upstream_mode:
                                        raise RuntimeError(
                                            "mobilegpt_upstream_action_error:"
                                            f"{type(action_error).__name__}: {action_error}"
                                        ) from action_error
                                    self._send_line(
                                        stream,
                                        "E",
                                        f"{type(action_error).__name__}: {action_error}",
                                    )
                                    continue
                                if spoken:
                                    answer = spoken
                                if not device_action:
                                    continue
                                actions_executed += 1
                                if post_action_wait:
                                    time.sleep(post_action_wait)
                                break
                        return self._result(
                            actions_executed=actions_executed,
                            answer=answer,
                            error=f"mobilegpt_step_budget_exhausted:{self.max_steps}",
                        )
            except Exception as error:
                return self._result(
                    actions_executed=actions_executed,
                    answer=answer,
                    error=f"{type(error).__name__}: {error}",
                )

    return MobileGPTAndroidWorldAgent()


def run_mobilegpt_replay(
    *,
    host: Any,
    goal: str,
    memory_root: str | Path,
    mobilegpt_root: str | Path,
    server_port: int,
    max_steps: int,
    stats_path: str | Path,
    server_log_path: str | Path,
) -> RunResult:
    """Run MobileGPT's selected-memory path with exploration fallbacks disabled."""

    from src.integrations.mobilegpt_converter import validate_mobilegpt_memory

    memory = Path(memory_root).expanduser().resolve()
    validated = validate_mobilegpt_memory(memory)
    upstream = Path(mobilegpt_root).expanduser().resolve()
    stats = Path(stats_path).expanduser().resolve()
    server_log = Path(server_log_path).expanduser().resolve()
    stats.parent.mkdir(parents=True, exist_ok=True)
    server_log.parent.mkdir(parents=True, exist_ok=True)
    if stats.exists() or server_log.exists():
        raise FileExistsError("mobilegpt_replay_evidence_already_exists")
    port = int(server_port)
    if port <= 0:
        raise ValueError("mobilegpt_server_port_invalid")
    environment = dict(os.environ)
    environment.update(
        {
            "MOBILEGPT_MEMORY_ROOT": str(memory),
            "MOBILEGPT_MEMORY_ONLY": "1",
            "MOBILEGPT_UPSTREAM_MODE": "0",
            "MOBILEGPT_SERVER_HOST": "127.0.0.1",
            "MOBILEGPT_SERVER_PORT": str(port),
            "MOBILEGPT_STATS_JSONL": str(stats),
            "MOBILEGPT_TARGET_APP": str(validated["app"]),
            "MOBILEGPT_TARGET_PACKAGE": str(validated["app"]),
            "MOBILEGPT_APP_PACKAGES": str(validated["app"]),
            "MOBILEGPT_CHAT_MAX_ATTEMPTS": "1",
        }
    )
    command = [
        sys.executable,
        "-m",
        "src.integrations.mobilegpt_runtime",
        "--mobilegpt-root",
        str(upstream),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    process: subprocess.Popen[Any] | None = None
    agent_result: Any | None = None
    try:
        with server_log.open("w", encoding="utf-8") as log_stream:
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
            original = {
                key: os.environ.get(key)
                for key in (
                    "MOBILEGPT_SERVER_HOST",
                    "MOBILEGPT_SERVER_PORT",
                    "MOBILEGPT_TARGET_PACKAGE",
                    "MOBILEGPT_APP_PACKAGES",
                )
            }
            os.environ.update(
                {
                    "MOBILEGPT_SERVER_HOST": "127.0.0.1",
                    "MOBILEGPT_SERVER_PORT": str(port),
                    "MOBILEGPT_TARGET_PACKAGE": str(validated["app"]),
                    "MOBILEGPT_APP_PACKAGES": str(validated["app"]),
                }
            )
            try:
                agent = build_mobilegpt_agent(host=host)
                agent.set_max_steps(max_steps)
                agent_result = agent.step(goal)
            finally:
                for key, value in original.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)

    events = _mobilegpt_stats(stats)
    chat_calls = [event for event in events if event.get("event") == "chat_call"]
    embedding_calls = [
        event for event in events if event.get("event") == "embedding_call"
    ]
    memory_misses = [
        event
        for event in events
        if event.get("event") == "mobilegpt_memory_only_miss"
    ]
    data = dict(getattr(agent_result, "data", {}) or {})
    error = str(data.get("error") or "").strip()
    if memory_misses and not error:
        error = "mobilegpt_memory_only_miss:" + str(
            memory_misses[0].get("stage") or "unknown"
        )
    usage = {
        key: sum(int(event.get(key) or 0) for event in chat_calls)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    return RunResult(
        not error,
        function_id="mobilegpt_replay",
        actions_executed=int(data.get("actions_executed") or 0),
        model_calls=len(chat_calls),
        fallback_steps=0,
        error=error or None,
        detail={
            "memory_hit": not memory_misses,
            "mobilegpt_result": data,
            "mobilegpt_stats": events,
            "embedding_calls": len(embedding_calls),
            "llm_usage": {
                **usage,
                "model_calls": len(chat_calls),
                "token_usage_status": (
                    "reported" if usage["total_tokens"] > 0 else "unavailable"
                ),
            },
        },
    )


def _mobilegpt_stats(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


__all__ = ["build_mobilegpt_agent", "run_mobilegpt_replay"]
