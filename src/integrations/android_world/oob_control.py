from __future__ import annotations

import base64
import binascii
import io
import json
import os
import subprocess
import time
from types import SimpleNamespace
from typing import Any, Callable
import uuid

import numpy as np
from PIL import Image

CONTROL_ACTION = "cn.com.omnimind.bot.debug.CONTROL_OMNIFLOW"
CONTROL_PACKAGE = "cn.com.omnimind.bot.debug"
CONTROL_ACCESSIBILITY_SERVICE = (
    f"{CONTROL_PACKAGE}/"
    "com.google.android.accessibility.selecttospeak.SelectToSpeakService"
)
CONTROL_RECEIVER = ".DebugOmniFlowControlReceiver"
CONTROL_RESULT_PATH = "files/debug-omniflow-control-result.json"
OBSERVE_ACTION = "cn.com.omnimind.bot.debug.OBSERVE_OMNIFLOW"
OBSERVE_RECEIVER = ".DebugOmniFlowObserveReceiver"
OBSERVE_RESULT_PATH = "files/debug-omniflow-observe-result.json"


class OobControlClient:
    def __init__(
        self,
        env: Any,
        *,
        adb_serial: str = "",
        adb_path: str = "",
        package_name: str = CONTROL_PACKAGE,
        receiver: str = CONTROL_RECEIVER,
        timeout_seconds: float = 30.0,
        run: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self.env = env
        self.adb_serial = self._resolve_serial(adb_serial)
        self.adb_path = str(adb_path or os.environ.get("ADB_PATH") or "adb")
        self.package_name = str(package_name or CONTROL_PACKAGE).strip()
        self.receiver = str(receiver or CONTROL_RECEIVER).strip()
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self._run_command = run or subprocess.run

    def observe(self, *, wait_to_stabilize: bool = False) -> dict[str, Any]:
        result = self._observe_request(wait_to_stabilize=wait_to_stabilize)
        if not isinstance(result, dict):
            raise RuntimeError("oob_control_observe_result_invalid")
        return result

    def act(self, action: dict[str, Any]) -> dict[str, Any]:
        result = self._request(
            "act",
            # The resident OOB executor otherwise returns immediately after
            # dispatch.  That leaves the next Function step and the official
            # validator observing the pre-action UI state.  Stabilization is
            # part of the OOB act contract; wait actions remain cheap because
            # the Android side handles them as already completed.
            {"action": action, "await_stabilization": True},
        )
        if not isinstance(result, dict):
            raise RuntimeError("oob_control_act_result_invalid")
        return result

    def _observe_request(self, *, wait_to_stabilize: bool) -> dict[str, Any]:
        self._run(
            [
                "shell",
                "run-as",
                self.package_name,
                "rm",
                "-f",
                OBSERVE_RESULT_PATH,
            ],
            timeout=10.0,
        )
        component = OBSERVE_RECEIVER
        if component.startswith("."):
            component = f"{self.package_name}/{component}"
        broadcast = self._run(
            [
                "shell",
                "am",
                "broadcast",
                "-a",
                OBSERVE_ACTION,
                "-n",
                component,
                "--ez",
                "includeScreenshot",
                "true",
                "--ez",
                "waitToStabilize",
                "true" if wait_to_stabilize else "false",
            ],
            timeout=self.timeout_seconds,
        )
        if broadcast.returncode != 0:
            raise RuntimeError(
                "oob_observe_broadcast_failed:"
                + (broadcast.stderr or broadcast.stdout or "").strip()
            )
        deadline = time.monotonic() + self.timeout_seconds
        last_error = ""
        while time.monotonic() < deadline:
            result = self._run(
                [
                    "shell",
                    "run-as",
                    self.package_name,
                    "cat",
                    OBSERVE_RESULT_PATH,
                ],
                timeout=10.0,
            )
            text = str(result.stdout or "").strip()
            if result.returncode == 0 and text:
                try:
                    response = json.loads(text)
                except json.JSONDecodeError as error:
                    raise RuntimeError("oob_observe_result_invalid_json") from error
                if not isinstance(response, dict):
                    raise RuntimeError("oob_observe_result_not_object")
                if response.get("success") is not True:
                    raise RuntimeError(
                        "oob_observe_failed:" + str(response.get("error") or "unknown")
                    )
                state = response.get("state")
                return {
                    **response,
                    **(state if isinstance(state, dict) else {}),
                }
            else:
                last_error = (result.stderr or result.stdout or "").strip()
            time.sleep(0.05)
        raise RuntimeError("oob_observe_result_timeout:" + last_error[-500:])

    def reset(self) -> None:
        self._request("reset", {})

    def _request(self, operation: str, payload: dict[str, Any]) -> Any:
        request_id = uuid.uuid4().hex
        encoded = base64.b64encode(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        ).decode("ascii")
        self._run(
            [
                "shell",
                "run-as",
                self.package_name,
                "rm",
                "-f",
                CONTROL_RESULT_PATH,
            ],
            timeout=10.0,
        )
        component = self.receiver
        if component.startswith("."):
            component = f"{self.package_name}/{component}"
        elif "/" not in component:
            component = f"{self.package_name}/.{component}"
        broadcast = self._run(
            [
                "shell",
                "am",
                "broadcast",
                "-a",
                CONTROL_ACTION,
                "-n",
                component,
                "--es",
                "requestId",
                request_id,
                "--es",
                "operation",
                operation,
                "--es",
                "requestBase64",
                encoded,
            ],
            timeout=self.timeout_seconds,
        )
        if broadcast.returncode != 0:
            raise RuntimeError(
                "oob_control_broadcast_failed:"
                + (broadcast.stderr or broadcast.stdout or "").strip()
            )
        deadline = time.monotonic() + self.timeout_seconds
        last_error = ""
        while time.monotonic() < deadline:
            result = self._run(
                [
                    "shell",
                    "run-as",
                    self.package_name,
                    "cat",
                    CONTROL_RESULT_PATH,
                ],
                timeout=10.0,
            )
            text = str(result.stdout or "").strip()
            if result.returncode == 0 and text:
                try:
                    response = json.loads(text)
                except json.JSONDecodeError as error:
                    raise RuntimeError("oob_control_result_invalid_json") from error
                if not isinstance(response, dict):
                    raise RuntimeError("oob_control_result_not_object")
                if response.get("request_id") != request_id:
                    last_error = "oob_control_result_request_id_mismatch"
                elif response.get("success") is not True:
                    raise RuntimeError(
                        "oob_control_failed:" + str(response.get("error") or "unknown")
                    )
                else:
                    return response.get("result")
            else:
                last_error = (result.stderr or result.stdout or "").strip()
            time.sleep(0.05)
        raise RuntimeError("oob_control_result_timeout:" + last_error[-500:])

    def _run(self, args: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
        command = [self.adb_path]
        if self.adb_serial:
            command.extend(["-s", self.adb_serial])
        command.extend(args)
        return self._run_command(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )

    def _resolve_serial(self, value: str) -> str:
        serial = str(value or os.environ.get("ANDROID_SERIAL") or "").strip()
        if serial:
            return serial
        controller = getattr(self.env, "controller", None)
        try:
            port = controller.env._coordinator._simulator._config.emulator_launcher.emulator_console_port
            if port:
                return f"emulator-{int(port)}"
        except Exception:
            pass
        return ""


def oob_state_from_payload(
    payload: dict[str, Any],
    *,
    fallback_screen_size: tuple[int, int] = (1, 1),
) -> Any:
    nested_state = payload.get("state")
    if isinstance(nested_state, dict):
        payload = {**payload, **nested_state}
    xml = str(payload.get("xml") or "")
    if not xml.strip():
        raise ValueError("oob_control_xml_missing")
    display = payload.get("display")
    if isinstance(display, dict):
        width = _positive_int(display.get("width"), fallback_screen_size[0])
        height = _positive_int(display.get("height"), fallback_screen_size[1])
    else:
        width, height = fallback_screen_size
    pixels = _decode_pixels(payload.get("image_base64"), width, height)
    representation_utils = __import__(
        "android_world.env.representation_utils",
        fromlist=["xml_dump_to_ui_elements"],
    )
    ui_elements = representation_utils.xml_dump_to_ui_elements(xml)
    package_name = str(payload.get("package_name") or "")
    activity_name = str(payload.get("activity_name") or "")
    auxiliaries = {
        "observe_backend": "oob_control",
        "package_name": package_name,
        "activity_name": activity_name,
        "display": {"width": width, "height": height},
    }
    stabilization = payload.get("stabilization")
    if isinstance(stabilization, dict):
        auxiliaries["stabilization"] = dict(stabilization)
    return SimpleNamespace(
        pixels=pixels,
        forest=xml,
        ui_elements=ui_elements,
        auxiliaries=auxiliaries,
    )


def _decode_pixels(value: Any, width: int, height: int) -> np.ndarray:
    encoded = str(value or "").strip()
    if encoded.startswith("data:image/") and "," in encoded:
        encoded = encoded.split(",", 1)[1]
    if encoded:
        try:
            image = Image.open(io.BytesIO(base64.b64decode(encoded, validate=False))).convert(
                "RGB"
            )
            return np.asarray(image).copy()
        except (binascii.Error, OSError, ValueError) as error:
            raise ValueError("oob_control_image_invalid") from error
    return np.zeros((max(1, height), max(1, width), 3), dtype=np.uint8)


def _positive_int(value: Any, fallback: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = 0
    return number if number > 0 else max(1, int(fallback))


__all__ = ["OobControlClient", "oob_state_from_payload"]
