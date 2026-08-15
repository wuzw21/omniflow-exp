from __future__ import annotations

import base64
import importlib
import inspect
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import xml.etree.ElementTree as ET

from omniflow import Action, ActionResult, Observation
from src.integrations.android_world.accessibility import (
    androidworld_forest_xml,
    forest_has_complete_active_application_window,
    xml_covers_screen,
    xml_with_screen_size,
)
from src.integrations.android_world.state import snapshot_androidworld_state


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


def _xml_semantic_score(xml_text: str) -> tuple[int, int, int, int]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return (0, 0, 0, 0)
    nodes = [element for element in root.iter() if element.tag == "node"]
    identity_count = sum(
        bool(
            str(element.attrib.get("text") or "").strip()
            or str(element.attrib.get("content-desc") or "").strip()
            or str(element.attrib.get("resource-id") or "").strip()
        )
        for element in nodes
    )
    editable_count = sum(
        str(element.attrib.get("editable") or "").lower() == "true"
        for element in nodes
    )
    actionable_count = sum(
        any(
            str(element.attrib.get(attribute) or "").lower() == "true"
            for attribute in ("clickable", "editable", "scrollable", "long-clickable")
        )
        for element in nodes
    )
    semantic_value_count = sum(
        bool(str(element.attrib.get(attribute) or "").strip())
        for element in nodes
        for attribute in ("text", "content-desc", "resource-id", "class", "package")
    )
    return identity_count, editable_count, actionable_count, semantic_value_count


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
        evidence_root: str | Path | None = None,
    ):
        self.env = env
        self.evidence_root = (
            Path(evidence_root).expanduser().resolve()
            if evidence_root is not None
            else None
        )
        self.observe_backend = "androidworld"
        self.act_backend = "androidworld"

    def installed_packages(self) -> set[str]:
        setup = importlib.import_module("android_world.env.setup_device.setup")
        return set(setup.get_installed_packages(self.env))

    def observe(
        self,
        *,
        xml: bool = True,
        screenshot: bool = False,
        app_info: bool = True,
        **_: Any,
    ) -> Observation:
        state = self.env.get_state(wait_to_stabilize=True)
        official_state = snapshot_androidworld_state(
            state,
            evidence_root=self.evidence_root,
        )
        elements = list(getattr(state, "ui_elements", ()) or ())
        auxiliaries = getattr(state, "auxiliaries", None)
        activity = str(
            (
                auxiliaries.get("activity_name")
                if isinstance(auxiliaries, dict)
                else ""
            )
            or getattr(self.env, "foreground_activity_name", "")
            or ""
        )
        package = str(
            (
                auxiliaries.get("package_name")
                if isinstance(auxiliaries, dict)
                else ""
            )
            or ""
        )
        package = package or (activity.split("/", 1)[0] if activity else "")
        display_width, display_height = self._screen_size()
        xml_text = ""
        graph_source = ""
        forest = getattr(state, "forest", None)
        forest_xml = ""
        if xml and forest is not None:
            forest_xml = androidworld_forest_xml(
                forest,
                screen_size=(display_width, display_height),
            )
        elements_xml = _elements_xml(elements) if xml and elements else ""
        if forest_xml and (
            not elements_xml
            or _xml_semantic_score(forest_xml) >= _xml_semantic_score(elements_xml)
        ):
            xml_text = forest_xml
            graph_source = "androidworld_state_forest"
        elif elements_xml:
            xml_text = elements_xml
            graph_source = "androidworld_state_ui_elements"
        package = package or _package_from_xml(xml_text)
        graph_complete = bool(xml_text) and (
            xml_covers_screen(
                xml_text,
                package_name=package,
                screen_size=(display_width, display_height),
            )
            or forest_has_complete_active_application_window(
                forest,
                package_name=package,
            )
        )
        if xml and xml_text and not graph_complete:
            xml_text = xml_with_screen_size(
                xml_text,
                screen_size=(display_width, display_height),
            )
            graph_source = f"{graph_source}_partial"
        return Observation(
            xml=xml_text or None if xml else None,
            package_name=package or None if app_info else None,
            activity_name=activity or None if app_info else None,
            image_base64=_image_base64(getattr(state, "pixels", None)) if screenshot else None,
            extra={
                "observe_backend": "androidworld",
                "androidworld_state": official_state,
                "ui_element_count": len(elements),
                "ui_graph_source": graph_source,
                "ui_graph_complete": bool(xml_text) and graph_complete,
                "display": {
                    "width": int(display_width),
                    "height": int(display_height),
                },
            },
        )

    def _screen_size(self) -> tuple[float, float]:
        for attribute in ("device_screen_size", "logical_screen_size"):
            value = tuple(getattr(self.env, attribute, ()) or ())
            if len(value) == 2 and float(value[0]) > 0 and float(value[1]) > 0:
                return float(value[0]), float(value[1])
        return 1000.0, 1000.0

    def _json_action(self, action: Action | dict[str, Any]) -> Any:
        action = Action.from_value(action)
        module = importlib.import_module("android_world.env.json_action")
        action_class = getattr(module, "JSONAction")
        if action.tool == "android_world_raw":
            payload = dict(action.args.get("payload") or {})
            signature = inspect.signature(action_class)
            return action_class(
                **{
                    key: value
                    for key, value in payload.items()
                    if key in signature.parameters
                }
            )
        width, height = self._screen_size()
        params = dict(action.args)
        payload: dict[str, Any] = {}
        action_name = action.tool
        action_type = {
            "click": "click",
            "long_press": "long_press",
            "input_text": "input_text",
            "swipe": "scroll",
            "press_back": "navigate_back",
            "press_home": "navigate_home",
            "press_enter": "keyboard_enter",
            "wait": "wait",
            "open_app": "open_app",
        }.get(action_name)
        if action_name == "press_key":
            key = str(params.get("keycode") or params.get("key") or "").strip().upper()
            if key in {"BACK", "NAVIGATE_BACK", "PRESS_BACK", "KEYCODE_BACK"}:
                action_type = "navigate_back"
            elif key in {"HOME", "NAVIGATE_HOME", "PRESS_HOME", "KEYCODE_HOME"}:
                action_type = "navigate_home"
            elif key in {
                "ENTER",
                "KEYBOARD_ENTER",
                "PRESS_ENTER",
                "KEYCODE_ENTER",
            }:
                action_type = "keyboard_enter"
            else:
                raise ValueError(f"unsupported AndroidWorld key: {key or 'missing'}")
        if action_type is None:
            raise ValueError(f"unsupported AndroidWorld action: {action_name}")
        payload["action_type"] = action_type
        if action_name in {"click", "long_press", "swipe", "input_text"}:
            if params.get("x") is not None:
                payload["x"] = float(params["x"]) / 1000.0 * width
            if params.get("y") is not None:
                payload["y"] = float(params["y"]) / 1000.0 * height
        if action_name == "input_text":
            payload["text"] = str(params.get("text") or "")
            payload["clear_text"] = True
        elif action_name == "swipe":
            payload["direction"] = _official_swipe_direction(params)
            if all(params.get(key) is not None for key in ("x1", "y1", "x2", "y2")):
                payload["action_type"] = "swipe"
        elif action_name == "open_app":
            package = str(params.get("package_name") or params.get("app_name") or "")
            payload["app_name"] = package
        signature = inspect.signature(action_class)
        return action_class(
            **{key: value for key, value in payload.items() if key in signature.parameters}
        )

    def act(self, value: Action | dict[str, Any], **_: Any) -> ActionResult:
        action = Action.from_value(value)
        if action.tool == "finished":
            return ActionResult(True)
        try:
            self.env.execute_action(self._json_action(action))
            return ActionResult(True)
        except Exception as error:
            return ActionResult(False, str(error))

    def reset(self, go_home: bool = False) -> None:
        reset = getattr(self.env, "reset", None)
        if callable(reset):
            reset(go_home=go_home)


def _official_swipe_direction(params: dict[str, Any]) -> str:
    direction = str(params.get("direction") or "").strip().lower()
    if direction in {"left", "right", "up", "down"}:
        return direction
    if not all(params.get(key) is not None for key in ("x1", "y1", "x2", "y2")):
        return "down"
    delta_x = float(params["x2"]) - float(params["x1"])
    delta_y = float(params["y2"]) - float(params["y1"])
    if abs(delta_x) > abs(delta_y):
        return "right" if delta_x > 0 else "left"
    return "down" if delta_y > 0 else "up"


__all__ = [
    "AndroidWorldHost",
    "androidworld_elements_xml",
    "make_agent_result",
]
