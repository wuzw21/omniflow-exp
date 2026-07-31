from __future__ import annotations

import base64
import importlib
import io
import json
import os
from pathlib import Path
import re
import socket
import time
from typing import Any, Callable
import xml.etree.ElementTree as ET

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


def build_mobilegpt_agent(
    *,
    env: Any,
    evidence_root: str | Path | None = None,
    action_factory: Callable[..., Any] | None = None,
) -> Any:
    server_host = str(os.environ.get("MOBILEGPT_SERVER_HOST") or "127.0.0.1").strip()
    server_port = int(os.environ.get("MOBILEGPT_SERVER_PORT") or 12345)
    connect_timeout = float(os.environ.get("MOBILEGPT_WAIT_START_TIMEOUT_SEC") or 60.0)
    response_timeout = float(os.environ.get("MOBILEGPT_WAIT_FINISH_TIMEOUT_SEC") or 120.0)
    post_action_wait = max(
        0.0,
        float(os.environ.get("MOBILEGPT_POST_ACTION_WAIT_SEC") or 1.0),
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
    host = AndroidWorldHost(env, evidence_root=evidence_root)

    class MobileGPTAndroidWorldAgent:
        name = "external:mobilegpt"
        transition_pause = 0.0

        def __init__(self) -> None:
            self.env = env
            self.max_steps = 20
            self.attempted = False

        def reset(self, go_home: bool = False) -> None:
            self.attempted = False
            self.env.reset(go_home=go_home)

        def set_max_steps(self, max_steps: int) -> None:
            self.max_steps = max(1, int(max_steps))

        def _new_action(self, **payload: Any) -> Any:
            if action_factory is not None:
                return action_factory(**payload)
            json_action = importlib.import_module("android_world.env.json_action")
            return json_action.JSONAction(**payload)

        def _execute(self, **payload: Any) -> None:
            self.env.execute_action(self._new_action(**payload))

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

        def _execute_server_action(
            self,
            action: Any,
            *,
            xml_text: str,
        ) -> tuple[bool, str]:
            normalized = normalize_mobilegpt_action(action)
            if not isinstance(normalized, dict):
                raise ValueError("mobilegpt_action_not_object")
            name = str(normalized.get("name") or "").strip()
            parameters = normalized.get("parameters")
            parameters = dict(parameters) if isinstance(parameters, dict) else {}
            if name in {"click", "long-click", "input"}:
                if parameters.get("index") is None:
                    raise ValueError(f"mobilegpt_action_index_required:{name}")
                x, y = _center(
                    _bounds_for_index(xml_text, parameters.get("index"))
                )
                if name == "click":
                    self._execute(action_type="click", x=x, y=y)
                elif name == "long-click":
                    self._execute(action_type="long_press", x=x, y=y)
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
            return make_agent_result(
                done=True,
                data={
                    "summary": (
                        "MobileGPT native AndroidWorld episode finished"
                        if not error
                        else error
                    ),
                    "source": self.name,
                    "error": error or None,
                    "answer": answer or None,
                    "actions_executed": int(actions_executed),
                    "state_backend": "androidworld",
                    "action_backend": "androidworld",
                    "native_androidworld_agent_io": True,
                },
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
                with socket.create_connection(
                    (server_host, server_port),
                    timeout=_socket_timeout(connect_timeout),
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

                        while actions_executed < self.max_steps:
                            observation = host.observe(
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
                                    continue
                                try:
                                    action = json.loads(response)
                                    device_action, spoken = self._execute_server_action(
                                        action,
                                        xml_text=xml_text,
                                    )
                                except Exception as action_error:
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


__all__ = ["build_mobilegpt_agent"]
