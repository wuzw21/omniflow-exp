from __future__ import annotations

import base64
import importlib
import inspect
import io
import json
import os
import subprocess
import time
from types import SimpleNamespace
from typing import Any
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from omniflow import Action, ActionResult, Observation

_OOB_FIELDS_OUTSIDE_EXTRA = {
    "image",
    "image_base64",
    "indexed_context",
    "pixels",
    "screenshot",
    "screenshot_base64",
    "xml",
}


def normalize_oob_get_state_payload(payload: dict[str, Any]) -> dict[str, Any]:
    state = payload.get("state")
    if not isinstance(state, dict):
        return payload
    normalized = dict(payload)
    for key, value in state.items():
        if value is not None or key not in normalized:
            normalized[key] = value
    return normalized


def _read(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _xml_escape(value: Any) -> str:
    return (
        str(value or "")
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _bbox(value: Any) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    try:
        return tuple(
            int(float(_read(value, key)))
            for key in ("x_min", "y_min", "x_max", "y_max")
        )
    except (TypeError, ValueError):
        return None


def _elements_xml(elements: list[Any]) -> str:
    nodes: list[str] = []
    width = height = 1
    package = ""
    for index, element in enumerate(elements):
        bounds = _bbox(_read(element, "bbox_pixels") or _read(element, "bbox"))
        if bounds is None:
            bounds = (0, 0, 1, 1)
        width = max(width, bounds[2])
        height = max(height, bounds[3])
        package = package or str(_read(element, "package_name", "") or "")
        attrs = {
            "id": index,
            "class": _read(element, "class_name", ""),
            "text": _read(element, "text", ""),
            "content-desc": _read(element, "content_description", ""),
            "resource-id": _read(element, "resource_name", ""),
            "package": _read(element, "package_name", ""),
            "bounds": f"[{bounds[0]},{bounds[1]}][{bounds[2]},{bounds[3]}]",
            "clickable": str(bool(_read(element, "is_clickable", False))).lower(),
            "editable": str(bool(_read(element, "is_editable", False))).lower(),
            "scrollable": str(bool(_read(element, "is_scrollable", False))).lower(),
        }
        rendered = " ".join(
            f'{key}="{_xml_escape(value)}"' for key, value in attrs.items()
        )
        nodes.append(f"    <node {rendered} />")
    return "\n".join(
        [
            "<hierarchy>",
            f'  <node package="{_xml_escape(package)}" bounds="[0,0][{width},{height}]">',
            *nodes,
            "  </node>",
            "</hierarchy>",
        ]
    )


def androidworld_elements_xml(elements: list[Any]) -> str:
    return _elements_xml(elements)


def _native_androidworld_xml(env: Any) -> str:
    controller = getattr(env, "controller", None)
    return str(getattr(controller, "_omniflow_last_ui_xml", "") or "").strip()


def _image_base64(pixels: Any) -> str | None:
    if pixels is None:
        return None
    if isinstance(pixels, str):
        return pixels.split(",", 1)[-1]
    if isinstance(pixels, (bytes, bytearray)):
        return base64.b64encode(bytes(pixels)).decode("ascii")
    try:
        from PIL import Image

        image = pixels if isinstance(pixels, Image.Image) else Image.fromarray(pixels)
        output = io.BytesIO()
        image.save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("ascii")
    except Exception:
        return None


def _package_from_xml(xml_text: str) -> str:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return ""
    packages = [
        str(element.attrib.get("package") or "")
        for element in root.iter()
        if str(element.attrib.get("package") or "")
    ]
    non_system = [item for item in packages if item != "com.android.systemui"]
    return (non_system or packages or [""])[-1]


def capture_adapter_state(
    *,
    env: Any,
    adb_serial: str = "",
    include_pixels: bool = True,
) -> Any | None:
    del adb_serial
    controller = getattr(env, "controller", None)
    if controller is None:
        return None
    elements: list[Any] = []
    get_elements = getattr(controller, "get_ui_elements", None)
    if callable(get_elements):
        try:
            elements = list(get_elements() or ())
        except Exception:
            elements = []
    xml_text = _native_androidworld_xml(env)
    pixels = None
    if include_pixels:
        screenshot = getattr(controller, "get_screenshot", None)
        if callable(screenshot):
            try:
                pixels = screenshot()
            except Exception:
                pixels = None
    activity = str(getattr(env, "foreground_activity_name", "") or "")
    package = activity.split("/", 1)[0] if activity else ""
    return SimpleNamespace(
        pixels=pixels,
        ui_elements=elements,
        xml=xml_text or (_elements_xml(elements) if elements else ""),
        activity_name=activity,
        package_name=package,
    )


def _agent_result_class() -> Any | None:
    for module_name in (
        "android_world.agents.base_agent",
        "android_world.agents.env_interacting_agent",
    ):
        try:
            result_class = getattr(importlib.import_module(module_name), "AgentInteractionResult")
            return result_class
        except (ImportError, AttributeError):
            continue
    return None


def make_agent_result(done: bool, data: dict[str, Any]) -> Any:
    result_class = _agent_result_class()
    if result_class is None:
        return SimpleNamespace(done=bool(done), data=dict(data))
    try:
        return result_class(done=bool(done), data=dict(data))
    except TypeError:
        return result_class(bool(done), dict(data))


class AndroidWorldHost:
    def __init__(
        self,
        env: Any,
        *,
        adb_serial: str = "",
        adb_path: str = "",
        post_action_wait_seconds: float = 0.0,
        open_app_ready_timeout_seconds: float | None = None,
    ):
        self.env = env
        self.adb_serial = str(adb_serial or "")
        self.adb_path = str(adb_path or os.environ.get("ADB_PATH") or "adb")
        self.post_action_wait_seconds = max(0.0, float(post_action_wait_seconds))
        ready_timeout = (
            float(os.environ.get("OMNIFLOW_OPEN_APP_READY_TIMEOUT_SEC") or 5.0)
            if open_app_ready_timeout_seconds is None
            else float(open_app_ready_timeout_seconds)
        )
        self.open_app_ready_timeout_seconds = max(0.0, ready_timeout)
        self.oob_url = str(os.environ.get("OMNIFLOW_OOB_DEVICE_URL") or "").rstrip("/")
        self.oob_device_url = self.oob_url
        self.observe_backend = str(
            os.environ.get("OMNIFLOW_OBSERVE_BACKEND") or "androidworld"
        ).strip().lower()
        self.act_backend = str(
            os.environ.get("OMNIFLOW_ACT_BACKEND") or "androidworld"
        ).lower()

    def _adb(self, *args: str, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
        command = [self.adb_path]
        if self.adb_serial:
            command.extend(["-s", self.adb_serial])
        command.extend(args)
        return subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )

    def installed_packages(self) -> set[str]:
        if not self.adb_serial:
            return set()
        result = self._adb("shell", "pm", "list", "packages", timeout=30)
        if result.returncode != 0:
            raise RuntimeError(
                f"installed_packages_failed:{str(result.stderr or '').strip()}"
            )
        return {
            line.removeprefix("package:").strip()
            for line in str(result.stdout or "").splitlines()
            if line.startswith("package:") and line.removeprefix("package:").strip()
        }

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.oob_url:
            raise RuntimeError("OMNIFLOW_OOB_DEVICE_URL is not configured")
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.oob_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"} if data else {},
            method=method,
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=45) as response:
            return json.loads(response.read().decode("utf-8"))

    def _fresh_uiautomator_xml(self) -> str:
        if not self.adb_serial:
            return ""
        remote_path = f"/sdcard/omniflow-window-{os.getpid()}-{id(self):x}.xml"
        try:
            self._adb("shell", "rm", "-f", remote_path, timeout=5)
            dumped = self._adb(
                "shell",
                "uiautomator",
                "dump",
                remote_path,
                timeout=20,
            )
            if dumped.returncode != 0:
                return ""
            read = self._adb("exec-out", "cat", remote_path, timeout=10)
            if read.returncode != 0:
                return ""
            xml_text = str(read.stdout or "").strip()
            if not xml_text:
                return ""
            ET.fromstring(xml_text)
            return xml_text
        except (ET.ParseError, OSError, subprocess.SubprocessError):
            return ""
        finally:
            try:
                self._adb("shell", "rm", "-f", remote_path, timeout=5)
            except (OSError, subprocess.SubprocessError):
                pass

    def _oob_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._request(method, path, payload)

    def _read_oob_debug_receiver_state(
        self,
        *,
        xml: bool,
        screenshot: bool,
    ) -> dict[str, Any]:
        package = str(
            os.environ.get("OMNIFLOW_OOB_PACKAGE") or "cn.com.omnimind.bot.debug"
        )
        receiver = str(
            os.environ.get("OMNIFLOW_OOB_GET_STATE_RECEIVER")
            or ".DebugGetStateReceiver"
        )
        component = f"{package}/{receiver}" if receiver.startswith(".") else receiver
        result_path = "files/debug-get-state-result.json"
        self._adb("shell", "run-as", package, "rm", "-f", result_path, timeout=10)
        broadcast = self._adb(
            "shell",
            "am",
            "broadcast",
            "-a",
            f"{package}.RUN_GET_STATE",
            "-n",
            component,
            "--ez",
            "includeXml",
            str(bool(xml)).lower(),
            "--ez",
            "includeScreenshot",
            str(bool(screenshot)).lower(),
            "--ez",
            "includeIndexedContext",
            "false",
            "--ei",
            "maxXmlChars",
            "0",
            timeout=30,
        )
        if broadcast.returncode != 0:
            return {"success": False, "error": broadcast.stderr.strip()}
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            result = self._adb(
                "shell", "run-as", package, "cat", result_path, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout)
            time.sleep(0.5)
        return {"success": False, "error": "OOB get_state result timeout"}

    def _observe_oob(self, *, xml: bool, screenshot: bool, app_info: bool) -> Observation:
        include_indexed = str(
            os.environ.get("OMNIFLOW_OOB_INCLUDE_INDEXED_CONTEXT") or ""
        ).lower() in {"1", "true", "yes", "on"}
        query = urllib.parse.urlencode(
            {
                "includeXml": str(bool(xml)).lower(),
                "includeScreenshot": str(bool(screenshot)).lower(),
                "includeIndexedContext": str(include_indexed).lower(),
                "maxXmlChars": "0",
                "filterOverlay": "true",
            }
        )
        raw = (
            self._oob_request("GET", f"/get_state?{query}")
            if self.oob_url
            else self._read_oob_debug_receiver_state(xml=xml, screenshot=screenshot)
        )
        raw = normalize_oob_get_state_payload(raw)
        if raw.get("success") is False:
            raise RuntimeError(str(raw.get("error") or "OOB get_state failed"))
        image = raw.get("screenshot") if isinstance(raw.get("screenshot"), dict) else {}
        indexed = raw.get("indexed_context")
        raw_state = {
            key: value
            for key, value in raw.items()
            if key not in _OOB_FIELDS_OUTSIDE_EXTRA
        }
        display_width, display_height = self._screen_size()
        extra: dict[str, Any] = {
            "observe_backend": "oob",
            "oob_device_url": self.oob_url,
            "include_indexed_context": include_indexed,
            "display": {
                "width": int(display_width),
                "height": int(display_height),
            },
            "raw_state": raw_state,
        }
        if include_indexed:
            extra["indexed_context"] = indexed
        return Observation(
            xml=str(raw.get("xml") or "") or None if xml else None,
            package_name=str(raw.get("package_name") or "") or None if app_info else None,
            activity_name=str(raw.get("activity_name") or "") or None if app_info else None,
            image_base64=str(image.get("data") or image.get("data_uri") or "") or None,
            extra=extra,
        )

    def observe(
        self,
        *,
        xml: bool = True,
        screenshot: bool = False,
        app_info: bool = True,
        **_: Any,
    ) -> Observation:
        if self.observe_backend.startswith("oob"):
            return self._observe_oob(xml=xml, screenshot=screenshot, app_info=app_info)
        try:
            state = self.env.get_state()
        except Exception:
            state = capture_adapter_state(
                env=self.env,
                adb_serial=self.adb_serial,
                include_pixels=screenshot,
            )
            if state is None:
                raise
        elements = list(getattr(state, "ui_elements", ()) or ())
        xml_text = str(getattr(state, "xml", "") or "")
        auxiliaries = getattr(state, "auxiliaries", None)
        if not xml_text and isinstance(auxiliaries, dict):
            xml_text = str(auxiliaries.get("xml") or "")
        if xml and not xml_text:
            xml_text = self._fresh_uiautomator_xml()
        if not xml_text:
            xml_text = _native_androidworld_xml(self.env)
        if xml and not xml_text:
            xml_text = _elements_xml(elements)
        activity = str(
            getattr(state, "activity_name", "")
            or getattr(self.env, "foreground_activity_name", "")
            or ""
        )
        package = str(getattr(state, "package_name", "") or "")
        package = package or _package_from_xml(xml_text)
        package = package or (activity.split("/", 1)[0] if activity else "")
        display_width, display_height = self._screen_size()
        return Observation(
            xml=xml_text or None if xml else None,
            package_name=package or None if app_info else None,
            activity_name=activity or None if app_info else None,
            image_base64=_image_base64(getattr(state, "pixels", None)) if screenshot else None,
            extra={
                "observe_backend": "androidworld",
                "ui_element_count": len(elements),
                "display": {
                    "width": int(display_width),
                    "height": int(display_height),
                },
            },
        )

    def _screen_size(self) -> tuple[float, float]:
        value = tuple(getattr(self.env, "logical_screen_size", ()) or ())
        if len(value) == 2 and float(value[0]) > 0 and float(value[1]) > 0:
            return float(value[0]), float(value[1])
        return 1000.0, 1000.0

    def _wait_for_package(self, package_name: str) -> tuple[bool, str]:
        deadline = time.monotonic() + self.open_app_ready_timeout_seconds
        observed_package = ""
        while True:
            try:
                observation = self.observe(xml=True, screenshot=False, app_info=True)
                observed_package = str(observation.package_name or "").strip()
                if observed_package == package_name and str(observation.xml or "").strip():
                    return True, observed_package
            except Exception:
                pass
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False, observed_package
            time.sleep(min(0.1, remaining))

    def _json_action(self, action: Action) -> Any:
        if action.tool == "android_world_raw":
            return dict(action.args.get("payload") or {})
        module = importlib.import_module("android_world.env.json_action")
        action_class = getattr(module, "JSONAction")
        action_types = getattr(module, "ActionType", None)
        directions = getattr(module, "ScrollDirection", None)
        width, height = self._screen_size()
        params = dict(action.args)
        payload: dict[str, Any] = {}
        action_name = action.tool
        enum_name = {
            "click": "CLICK",
            "long_press": "LONG_PRESS",
            "input_text": "INPUT_TEXT",
            "swipe": "SCROLL",
            "press_key": "PRESS_KEYBOARD",
            "press_back": "NAVIGATE_BACK",
            "press_home": "NAVIGATE_HOME",
            "press_enter": "KEYBOARD_ENTER",
            "wait": "WAIT",
            "open_app": "OPEN_APP",
        }.get(action_name)
        if action_name == "press_key":
            key = str(params.get("keycode") or params.get("key") or "").strip().upper()
            if key in {"BACK", "NAVIGATE_BACK", "PRESS_BACK", "KEYCODE_BACK"}:
                enum_name = "NAVIGATE_BACK"
            elif key in {"HOME", "NAVIGATE_HOME", "PRESS_HOME", "KEYCODE_HOME"}:
                enum_name = "NAVIGATE_HOME"
            elif key in {
                "ENTER",
                "KEYBOARD_ENTER",
                "PRESS_ENTER",
                "KEYCODE_ENTER",
            }:
                enum_name = "KEYBOARD_ENTER"
            elif key in {"DEL", "DELETE", "KEYCODE_DEL"}:
                payload["keycode"] = "KEYCODE_DEL"
            elif key:
                payload["keycode"] = key if key.startswith("KEYCODE_") else f"KEYCODE_{key}"
            else:
                raise ValueError("press_key missing keycode")
        if enum_name is None:
            raise ValueError(f"unsupported AndroidWorld action: {action_name}")
        payload["action_type"] = getattr(action_types, enum_name, enum_name.lower())
        if action_name in {"click", "long_press", "swipe", "input_text"}:
            if params.get("x") is not None:
                payload["x"] = float(params["x"]) / 1000.0 * width
            if params.get("y") is not None:
                payload["y"] = float(params["y"]) / 1000.0 * height
        if action_name == "input_text":
            payload["text"] = str(params.get("text") or "")
            payload["clear_text"] = True
        elif action_name == "swipe":
            direction = str(params.get("direction") or "down").upper()
            payload["direction"] = getattr(directions, direction, direction.lower())
        elif action_name == "wait":
            payload["seconds"] = float(params.get("duration_ms", 1000)) / 1000.0
        elif action_name == "open_app":
            package = str(params.get("package_name") or params.get("app_name") or "")
            payload.update(app_name=package, package_name=package)
        signature = inspect.signature(action_class)
        return action_class(
            **{key: value for key, value in payload.items() if key in signature.parameters}
        )

    def act(self, value: Action | dict[str, Any], **_: Any) -> ActionResult:
        action = Action.from_value(value)
        if action.tool == "finished":
            return ActionResult(True)
        try:
            settled_by_oob = False
            open_app_package = (
                str(action.args.get("package_name") or "").strip()
                if action.tool == "open_app"
                else ""
            )
            if action.tool == "wait":
                duration_ms = max(0.0, float(action.args.get("duration_ms", 1000)))
                time.sleep(duration_ms / 1000.0)
                return ActionResult(True)
            if action.tool == "set_clipboard":
                text = str(action.args.get("text") or "")
                adb_utils = importlib.import_module("android_world.env.adb_utils")
                controller = getattr(self.env, "controller", self.env)
                adb_utils.set_clipboard_contents(text, controller)
                return ActionResult(True)
            if self.act_backend.startswith("oob") and self.oob_url:
                raw = self._request(
                    "POST",
                    "/act",
                    {
                        "action": action.to_dict(),
                        "settle_delay_ms": int(self.post_action_wait_seconds * 1000),
                    },
                )
                if raw.get("success") is not True:
                    return ActionResult(False, str(raw.get("error") or "OOB act failed"))
                settled_by_oob = True
            elif action.tool in {"click", "long_press"} and self.adb_serial and all(
                action.args.get(key) is not None for key in ("x", "y")
            ):
                width, height = self._screen_size()
                x = str(_relative_pixel(action.args["x"], width))
                y = str(_relative_pixel(action.args["y"], height))
                command = (
                    ("shell", "input", "tap", x, y)
                    if action.tool == "click"
                    else (
                        "shell",
                        "input",
                        "swipe",
                        x,
                        y,
                        x,
                        y,
                        str(int(float(action.args.get("duration_ms") or 1000))),
                    )
                )
                result = self._adb(*command, timeout=15)
                if result.returncode != 0:
                    return ActionResult(
                        False,
                        result.stderr.strip() or f"coordinate {action.tool} failed",
                    )
            elif action.tool == "swipe" and self.adb_serial and all(
                action.args.get(key) is not None
                for key in ("x1", "y1", "x2", "y2")
            ):
                width, height = self._screen_size()
                input_source = str(
                    action.args.get("input_source") or ""
                ).strip().lower()
                if input_source and input_source not in {
                    "touchscreen",
                    "mouse",
                    "stylus",
                    "touchpad",
                    "trackball",
                    "touchnavigation",
                    "joystick",
                }:
                    return ActionResult(False, f"unsupported input source: {input_source}")
                command = ["shell", "input"]
                if input_source:
                    command.append(input_source)
                command.extend(
                    [
                        "swipe",
                        str(_relative_pixel(action.args["x1"], width)),
                        str(_relative_pixel(action.args["y1"], height)),
                        str(_relative_pixel(action.args["x2"], width)),
                        str(_relative_pixel(action.args["y2"], height)),
                        str(int(float(action.args.get("duration_ms") or 500))),
                    ]
                )
                result = self._adb(
                    *command,
                    timeout=15,
                )
                if result.returncode != 0:
                    return ActionResult(
                        False,
                        result.stderr.strip() or "coordinate swipe failed",
                    )
            elif action.tool == "open_app" and self.adb_serial:
                package = str(action.args.get("package_name") or "").strip()
                app_name = str(action.args.get("app_name") or "").strip()
                if package:
                    result = self._adb(
                        "shell",
                        "monkey",
                        "-p",
                        package,
                        "-c",
                        "android.intent.category.LAUNCHER",
                        "1",
                        timeout=15,
                    )
                    if result.returncode != 0:
                        return ActionResult(
                            False,
                            result.stderr.strip() or "open_app failed",
                        )
                elif app_name:
                    adb_utils = importlib.import_module("android_world.env.adb_utils")
                    adb_utils.launch_app(app_name, self.env.controller)
                else:
                    return ActionResult(False, "open_app missing app identifier")
            else:
                self.env.execute_action(self._json_action(action))
            if (
                open_app_package
                and self.adb_serial
                and self.open_app_ready_timeout_seconds > 0
            ):
                ready, observed_package = self._wait_for_package(open_app_package)
                if not ready:
                    return ActionResult(
                        False,
                        "open_app_target_not_ready:"
                        f"expected={open_app_package}:observed={observed_package or 'unknown'}",
                    )
            wait_after_s = float(
                action.args.get("wait_after_s", self.post_action_wait_seconds) or 0.0
            )
            if wait_after_s > 0 and not settled_by_oob:
                time.sleep(wait_after_s)
            return ActionResult(True)
        except Exception as error:
            return ActionResult(False, str(error))

    def reset(self, go_home: bool = False) -> None:
        reset = getattr(self.env, "reset", None)
        if callable(reset):
            reset(go_home=go_home)


def _relative_pixel(value: object, extent: float) -> int:
    maximum = max(0, int(round(float(extent))) - 1)
    return max(0, min(maximum, int(round(float(value) / 1000.0 * extent))))


__all__ = [
    "AndroidWorldHost",
    "androidworld_elements_xml",
    "capture_adapter_state",
    "make_agent_result",
    "normalize_oob_get_state_payload",
]
