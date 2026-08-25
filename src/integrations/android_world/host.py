from __future__ import annotations

import base64
from contextlib import nullcontext
import importlib
import inspect
import io
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
from typing import Any
import xml.etree.ElementTree as ET

from omniflow import Action, ActionResult, Observation
from omniflow.core.androidworld_accessibility import (
    androidworld_forest_xml,
    forest_has_complete_active_application_window,
    xml_covers_screen,
    xml_has_complete_application_modal,
    xml_with_screen_size,
)
from src.experiment.performance_metrics import PerformanceMetrics
from src.integrations.android_world.apps import (
    launchable_androidworld_apps,
    launcher_package_label,
    resolve_androidworld_app_name,
    resolve_androidworld_package,
)
from src.integrations.android_world.oob_control import (
    OobControlClient,
    oob_state_from_payload,
)
from src.integrations.android_world.state import snapshot_androidworld_state


_ANDROID_SYSTEM_OPEN_APP_PACKAGES = frozenset({"com.android.settings"})
_ANDROIDWORLD_NON_TASK_LAUNCHER_PACKAGES = frozenset(
    {
        "cn.com.omnimind.bot.debug",
        "com.example.MobileGPT",
        "com.google.androidenv.accessibilityforwarder",
    }
)


def _read(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _xml_escape(value: Any) -> str:
    return (
        str("" if value is None else value)
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
            "resource-id": _read(element, "resource_name", "")
            or _read(element, "resource_id", ""),
            "package": _read(element, "package_name", ""),
            "bounds": f"[{bounds[0]},{bounds[1]}][{bounds[2]},{bounds[3]}]",
            "clickable": str(bool(_read(element, "is_clickable", False))).lower(),
            "editable": str(
                bool(_read(element, "is_editable", False))
                or str(_read(element, "class_name", "")).endswith("EditText")
            ).lower(),
            "focusable": str(bool(_read(element, "is_focusable", False))).lower(),
            "focused": str(bool(_read(element, "is_focused", False))).lower(),
            "long-clickable": str(
                bool(_read(element, "is_long_clickable", False))
            ).lower(),
            "enabled": str(bool(_read(element, "is_enabled", True))).lower(),
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


def androidworld_observation_xml(observation: dict[str, Any]) -> str:
    xml = observation.get("xml")
    if isinstance(xml, str) and xml.strip():
        return xml.strip()
    forest = observation.get("forest")
    if isinstance(forest, str) and forest.strip():
        return forest.strip()
    elements = observation.get("ui_elements")
    if isinstance(elements, list) and elements:
        return androidworld_elements_xml(elements).strip()
    return ""


def androidworld_observation_package(observation: dict[str, Any]) -> str:
    auxiliaries = observation.get("auxiliaries")
    for value in (
        observation.get("package_name"),
        observation.get("packageName"),
        auxiliaries.get("package_name") if isinstance(auxiliaries, dict) else None,
        auxiliaries.get("packageName") if isinstance(auxiliaries, dict) else None,
    ):
        package = str(value or "").strip()
        if package:
            return package
    packages = [
        str(_read(element, "package_name", "") or "").strip()
        for element in observation.get("ui_elements") or ()
    ]
    packages = [package for package in packages if package]
    non_system = [package for package in packages if package != "com.android.systemui"]
    package = (non_system or packages or [""])[-1]
    if package:
        return package
    return _package_from_xml(androidworld_observation_xml(observation))


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
        performance_metrics: PerformanceMetrics | None = None,
        control_backend: str = "androidworld",
    ):
        self.env = env
        self.adb_serial = str(
            adb_serial or os.environ.get("ANDROID_SERIAL") or ""
        ).strip()
        self.adb_path = str(adb_path or os.environ.get("ADB_PATH") or "adb")
        self.recorder = getattr(env, "_recorder", None)
        self.evidence_root = (
            Path(evidence_root).expanduser().resolve()
            if evidence_root is not None
            else None
        )
        normalized_backend = str(control_backend or "androidworld").strip().lower()
        if normalized_backend in {"oob", "omniflow", "oob_control"}:
            self.observe_backend = "oob_control"
            self.act_backend = "oob_control"
            self.control_client = OobControlClient(
                env,
                adb_serial=adb_serial,
                adb_path=adb_path,
                package_name=str(
                    os.environ.get("OMNIFLOW_OOB_PACKAGE", "")
                    or "cn.com.omnimind.bot.debug"
                ),
                receiver=str(
                    os.environ.get("OMNIFLOW_OOB_CONTROL_RECEIVER", "")
                    or ".DebugOmniFlowControlReceiver"
                ),
            )
        elif normalized_backend in {"androidworld", "native"}:
            self.observe_backend = "androidworld"
            self.act_backend = "androidworld"
            self.control_client = None
        else:
            raise ValueError(f"androidworld_control_backend_invalid:{control_backend}")
        self.performance_metrics = performance_metrics
        self.open_app_ready_timeout_seconds = max(
            0.0, float(open_app_ready_timeout_seconds or 0.0)
        )

    def installed_packages(self) -> set[str]:
        setup = importlib.import_module("android_world.env.setup_device.setup")
        packages = set(setup.get_installed_packages(self.env))
        if not packages:
            command = [self.adb_path]
            if self.adb_serial:
                command.extend(["-s", self.adb_serial])
            command.extend(["shell", "pm", "list", "packages"])
            completed = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=15.0,
            )
            if completed.returncode == 0:
                packages.update(
                    line.removeprefix("package:").strip()
                    for line in completed.stdout.splitlines()
                    if line.strip().startswith("package:")
                )
        return packages.union(_ANDROID_SYSTEM_OPEN_APP_PACKAGES)

    def installed_apps(self) -> dict[str, str]:
        packages = self.installed_packages()
        controller = getattr(self.env, "controller", self.env)
        launcher_packages = self._launchable_packages()
        if launcher_packages:
            exposed_packages = (
                packages.intersection(launcher_packages)
                - _ANDROIDWORLD_NON_TASK_LAUNCHER_PACKAGES
            )
        else:
            # PackageManager can be unavailable during emulator startup. Keep
            # the official AndroidWorld registry as the narrow fallback.
            exposed_packages = packages
        catalog = launchable_androidworld_apps(exposed_packages, controller)
        if launcher_packages:
            catalog_values = set(catalog.values())
            for package in sorted(exposed_packages):
                if package in catalog_values:
                    continue
                registered_name = resolve_androidworld_app_name(package, controller)
                label = launcher_package_label(registered_name)
                if label in catalog and catalog[label] != package:
                    label = f"{label} ({package})"
                catalog[label] = package
                catalog_values.add(package)
        if "com.android.settings" in exposed_packages:
            catalog.setdefault("Settings", "com.android.settings")
        return dict(
            sorted(catalog.items(), key=lambda item: (item[0].casefold(), item[1]))
        )

    def _launchable_packages(self) -> set[str]:
        command = [self.adb_path]
        if self.adb_serial:
            command.extend(["-s", self.adb_serial])
        command.extend(
            [
                "shell",
                "cmd",
                "package",
                "query-activities",
                "--brief",
                "-a",
                "android.intent.action.MAIN",
                "-c",
                "android.intent.category.LAUNCHER",
            ]
        )
        try:
            completed = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=15.0,
            )
        except (OSError, subprocess.SubprocessError):
            return set()
        if completed.returncode != 0:
            return set()
        packages: set[str] = set()
        for line in completed.stdout.splitlines():
            component = line.strip()
            if "/" not in component:
                continue
            package = component.split("/", 1)[0].strip()
            if package and all(
                part.replace("_", "").isalnum() for part in package.split(".")
            ):
                packages.add(package)
        return packages

    def observe(
        self,
        *,
        xml: bool = True,
        screenshot: bool = False,
        app_info: bool = True,
        **_: Any,
    ) -> Observation:
        if self.performance_metrics is None:
            return self._observe_impl(
                xml=xml,
                screenshot=screenshot,
                app_info=app_info,
            )
        started_ns = time.perf_counter_ns()
        success = False
        try:
            observation = self._observe_impl(
                xml=xml,
                screenshot=screenshot,
                app_info=app_info,
            )
            success = True
            return observation
        finally:
            self.performance_metrics.record(
                "observe",
                (time.perf_counter_ns() - started_ns) / 1_000_000.0,
                success=success,
            )

    def _observe_impl(
        self,
        *,
        xml: bool,
        screenshot: bool,
        app_info: bool,
    ) -> Observation:
        metrics = self.performance_metrics
        wait_to_stabilize = (
            str(
                os.environ.get(
                    "OMNIFLOW_ANDROIDWORLD_WAIT_TO_STABILIZE", "1"
                )
            )
            .strip()
            .lower()
            not in {"0", "false", "no", "off"}
        )
        with (
            metrics.timed("observe_get_state")
            if metrics is not None
            else nullcontext()
        ):
            if self.control_client is not None:
                state = oob_state_from_payload(
                    self.control_client.observe(
                        wait_to_stabilize=wait_to_stabilize,
                    ),
                    fallback_screen_size=tuple(
                        int(value) for value in self._screen_size()
                    ),
                )
            else:
                state = self.env.get_state(
                    wait_to_stabilize=wait_to_stabilize
                )
        if self.control_client is not None:
            record_observation = getattr(
                self.recorder, "record_host_observation", None
            )
            if callable(record_observation):
                record_observation(state)
        with (
            metrics.timed("observe_snapshot")
            if metrics is not None
            else nullcontext()
        ):
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
        with (
            metrics.timed("observe_xml_transform")
            if metrics is not None
            else nullcontext()
        ):
            display_width, display_height = self._screen_size()
            xml_text = ""
            graph_source = ""
            forest = getattr(state, "forest", None)
            forest_xml = ""
            if xml and isinstance(forest, str):
                forest_xml = forest
            elif xml and forest is not None:
                forest_xml = androidworld_forest_xml(
                    forest,
                    screen_size=(display_width, display_height),
                )
            elements_xml = _elements_xml(elements) if xml and elements else ""
            graph_package = (
                package
                or _package_from_xml(forest_xml)
                or _package_from_xml(elements_xml)
            )
            forest_complete = bool(forest_xml) and (
                xml_covers_screen(
                    forest_xml,
                    package_name=graph_package,
                    screen_size=(display_width, display_height),
                )
                or (
                    not isinstance(forest, str)
                    and forest_has_complete_active_application_window(
                        forest,
                        package_name=graph_package,
                    )
                )
            )
            elements_complete = bool(elements_xml) and xml_covers_screen(
                elements_xml,
                package_name=graph_package,
                screen_size=(display_width, display_height),
            )
            if self.control_client is not None and forest_xml:
                # OOB already returns the complete, ordered accessibility
                # forest.  Re-serializing its parsed ui_elements flattens the
                # hierarchy and destroys the parent/sibling relations used by
                # OmniTransfer, even when the visible labels and bounds stay
                # identical.
                xml_text = forest_xml
                graph_source = "oob_control_forest"
            elif forest_xml and (
                not elements_xml
                or (forest_complete and not elements_complete)
                or (
                    forest_complete == elements_complete
                    and _xml_semantic_score(forest_xml)
                    >= _xml_semantic_score(elements_xml)
                )
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
                or xml_has_complete_application_modal(
                    xml_text,
                    package_name=package,
                )
                or (
                    not isinstance(forest, str)
                    and forest_has_complete_active_application_window(
                        forest,
                        package_name=package,
                    )
                )
            )
            if xml and xml_text and not graph_complete:
                xml_text = xml_with_screen_size(
                    xml_text,
                    screen_size=(display_width, display_height),
                )
                graph_source = f"{graph_source}_partial"
        with (
            metrics.timed("observe_image_encode")
            if metrics is not None
            else nullcontext()
        ):
            image_base64 = (
                _image_base64(getattr(state, "pixels", None)) if screenshot else None
            )
        return Observation(
            xml=xml_text or None if xml else None,
            package_name=package or None if app_info else None,
            activity_name=activity or None if app_info else None,
            image_base64=image_base64,
            extra={
                "observe_backend": self.observe_backend,
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
        # AndroidWorld coordinates and accessibility bounds are expressed in
        # the logical application display.  On Fold profiles the physical
        # size can be a rotated, different resolution (for example
        # 1768x2208 while the app window is 2208x1840).  Using the physical
        # size here corrupts XML completeness checks and pixel/canonical
        # coordinate conversion, so it is only a fallback when the logical
        # display is unavailable.
        for attribute in ("logical_screen_size", "device_screen_size"):
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
            elif key in {
                "DEL",
                "DELETE",
                "KEYCODE_DEL",
                "KEYCODE_DELETE",
            }:
                action_type = "press_keyboard"
                payload["keycode"] = "KEYCODE_DEL"
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
            controller = getattr(self.env, "controller", self.env)
            payload["app_name"] = resolve_androidworld_app_name(package, controller)
        signature = inspect.signature(action_class)
        return action_class(
            **{key: value for key, value in payload.items() if key in signature.parameters}
        )

    def act(self, value: Action | dict[str, Any], **_: Any) -> ActionResult:
        if self.performance_metrics is None:
            return self._act_impl(value)
        started_ns = time.perf_counter_ns()
        result: ActionResult | None = None
        try:
            result = self._act_impl(value)
            return result
        finally:
            self.performance_metrics.record(
                "act",
                (time.perf_counter_ns() - started_ns) / 1_000_000.0,
                success=result is not None and result.success,
            )

    def _act_impl(self, value: Action | dict[str, Any]) -> ActionResult:
        action = Action.from_value(value)
        if action.tool == "finished":
            return ActionResult(True)
        try:
            if self.control_client is not None:
                def execute() -> dict[str, Any]:
                    if action.tool == "press_key":
                        # OOB's ENTER is an IME action for focused text fields;
                        # AndroidWorld's native key event also handles system
                        # dialogs. Keep global key semantics compatible across
                        # the old and new collectors.
                        self.env.execute_action(self._json_action(action))
                        return {"success": True}
                    payload = action.to_dict()
                    if action.tool == "open_app":
                        args = dict(payload.get("args") or {})
                        identifier = str(
                            args.get("package_name") or args.get("app_name") or ""
                        ).strip()
                        resolved_package = resolve_androidworld_package(identifier)
                        if resolved_package:
                            args["package_name"] = resolved_package
                            args.pop("app_name", None)
                            payload["args"] = args
                    return self.control_client.act(payload)

                execute_host_action = getattr(
                    self.recorder, "execute_host_action", None
                )
                if callable(execute_host_action):
                    def after_observation() -> Any:
                        if (
                            action.tool == "open_app"
                            and self.open_app_ready_timeout_seconds > 0.0
                        ):
                            identifier = str(
                                action.args.get("package_name")
                                or action.args.get("app_name")
                                or ""
                            ).strip()
                            return self._observe_open_app_ready(identifier)
                        return oob_state_from_payload(
                            self.control_client.observe(wait_to_stabilize=True),
                            fallback_screen_size=tuple(
                                int(value) for value in self._screen_size()
                            ),
                        )

                    return ActionResult.from_value(
                        execute_host_action(
                            action,
                            execute=execute,
                            project=self._json_action,
                            after_observation=after_observation,
                        )
                    )
                return ActionResult.from_value(execute())
            self.env.execute_action(self._json_action(action))
            return ActionResult(True)
        except Exception as error:
            return ActionResult(False, str(error))

    def _observe_open_app_ready(self, identifier: str) -> Any:
        """Return the first post-launch OOB state owned by the target app.

        The recorder asks for an after-action observation immediately.  App
        launch is asynchronous, so that sample can still be Launcher even
        though the dispatch itself succeeded.  OmniFlow needs the ready state
        in the same action record; otherwise the next Function step can be
        grounded against stale Launcher XML.
        """

        expected_package = resolve_androidworld_package(identifier) or str(
            identifier or ""
        ).strip()
        deadline = time.monotonic() + self.open_app_ready_timeout_seconds
        last_state: Any = None
        while True:
            last_state = oob_state_from_payload(
                self.control_client.observe(wait_to_stabilize=True),
                fallback_screen_size=tuple(
                    int(value) for value in self._screen_size()
                ),
            )
            auxiliaries = getattr(last_state, "auxiliaries", None)
            observed_package = str(
                auxiliaries.get("package_name")
                if isinstance(auxiliaries, dict)
                else ""
            ).strip()
            if expected_package and observed_package == expected_package:
                return last_state
            if time.monotonic() >= deadline:
                return last_state
            time.sleep(0.25)

    def reset(self, go_home: bool = False) -> None:
        if self.control_client is not None:
            self.control_client.reset()
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
    "androidworld_observation_xml",
    "androidworld_observation_package",
    "make_agent_result",
]
