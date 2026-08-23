"""AppAgent controller backed by OmniFlow's canonical OOB observe/act API.

This module is copied into the disposable AppAgent workspace.  It deliberately
keeps AppAgent's XML traversal and planner intact; only the device I/O is
provided by the same OOB bridge used by the AndroidWorld methods.
"""

from __future__ import annotations

import base64
import binascii
import io
import os
import subprocess
import xml.etree.ElementTree as ET
from typing import Any

from PIL import Image

from src.integrations.android_world.oob_control import OobControlClient


class _Configs:
    # The official implementation uses this only for duplicate element
    # suppression.  Keep the upstream default without importing AppAgent's
    # config module, which would make this adapter depend on cwd state.
    MIN_DIST = 10


configs = {"MIN_DIST": 10}


class AndroidElement:
    def __init__(self, uid: str, bbox: tuple[tuple[int, int], tuple[int, int]], attrib: str):
        self.uid = uid
        self.bbox = bbox
        self.attrib = attrib


def _adb_path() -> str:
    return str(os.environ.get("ADB_PATH") or os.environ.get("OMNIFLOW_ADB_PATH") or "adb")


def _serial() -> str:
    return str(os.environ.get("OMNIFLOW_APPA_AGENT_SERIAL") or os.environ.get("ANDROID_SERIAL") or "").strip()


def _run_adb_without_agent_stdin(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    """Keep OOB's adb polling from consuming AppAgent's task input line."""

    kwargs.setdefault("stdin", subprocess.DEVNULL)
    return subprocess.run(command, **kwargs)


def execute_adb(adb_command: str) -> str:
    """Compatibility helper for imports from the untouched executor."""

    result = subprocess.run(
        adb_command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "ERROR"


def list_all_devices() -> list[str]:
    serial = _serial()
    return [serial] if serial else []


def _bounds(elem: ET.Element) -> tuple[int, int, int, int]:
    raw = elem.attrib.get("bounds", "[0,0][0,0]")[1:-1].split("][")
    x1, y1 = (int(value) for value in raw[0].split(","))
    x2, y2 = (int(value) for value in raw[1].split(","))
    return x1, y1, x2, y2


def get_id_from_element(elem: ET.Element) -> str:
    x1, y1, x2, y2 = _bounds(elem)
    if elem.attrib.get("resource-id"):
        elem_id = elem.attrib["resource-id"].replace(":", ".").replace("/", "_")
    else:
        elem_id = f"{elem.attrib.get('class', '')}_{x2 - x1}_{y2 - y1}"
    content_desc = elem.attrib.get("content-desc", "")
    if content_desc and len(content_desc) < 20:
        elem_id += "_" + content_desc.replace("/", "_").replace(" ", "").replace(":", "_")
    return elem_id


def traverse_tree(xml_path: str, elem_list: list[AndroidElement], attrib: str, add_index: bool = False) -> None:
    path: list[ET.Element] = []
    for event, elem in ET.iterparse(xml_path, ["start", "end"]):
        if event == "start":
            path.append(elem)
            if elem.attrib.get(attrib) == "true":
                x1, y1, x2, y2 = _bounds(elem)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                elem_id = get_id_from_element(elem)
                if len(path) > 1:
                    elem_id = get_id_from_element(path[-2]) + "_" + elem_id
                if add_index:
                    elem_id += "_" + elem.attrib.get("index", "0")
                duplicate = False
                for existing in elem_list:
                    bbox = existing.bbox
                    old_center = (
                        (bbox[0][0] + bbox[1][0]) // 2,
                        (bbox[0][1] + bbox[1][1]) // 2,
                    )
                    if ((center[0] - old_center[0]) ** 2 + (center[1] - old_center[1]) ** 2) ** 0.5 <= configs["MIN_DIST"]:
                        duplicate = True
                        break
                if not duplicate:
                    elem_list.append(AndroidElement(elem_id, ((x1, y1), (x2, y2)), attrib))
        else:
            path.pop()


class AndroidController:
    """Drop-in AppAgent controller whose every observation/action is OOB."""

    def __init__(self, device: str):
        self.device = str(device)
        self._oob = OobControlClient(
            None,
            adb_serial=self.device,
            adb_path=_adb_path(),
            run=_run_adb_without_agent_stdin,
        )
        self.screenshot_dir = "/sdcard"
        self.xml_dir = "/sdcard"
        self.width, self.height = self.get_device_size()
        self.backslash = "\\"

    def _snapshot(self) -> dict[str, Any]:
        snapshot = self._oob.observe(wait_to_stabilize=True)
        if not isinstance(snapshot, dict) or not str(snapshot.get("xml") or "").strip():
            raise RuntimeError("omniflow_oob_appagent_observation_missing")
        display = snapshot.get("display")
        if isinstance(display, dict):
            self.width = int(display.get("width") or self.width or 1)
            self.height = int(display.get("height") or self.height or 1)
        return snapshot

    def get_device_size(self) -> tuple[int, int]:
        snapshot = self._snapshot()
        display = snapshot.get("display") or {}
        return int(display.get("width") or 1), int(display.get("height") or 1)

    def _normalized(self, x: int | float, y: int | float) -> tuple[int, int]:
        return (
            max(0, min(1000, round(float(x) / max(1, self.width) * 1000))),
            max(0, min(1000, round(float(y) / max(1, self.height) * 1000))),
        )

    def _act(self, action: dict[str, Any]) -> str:
        self._oob.act(action)
        return "OK"

    def get_screenshot(self, prefix: str, save_dir: str) -> str:
        snapshot = self._snapshot()
        encoded = str(snapshot.get("image_base64") or "")
        if encoded.startswith("data:image/") and "," in encoded:
            encoded = encoded.split(",", 1)[1]
        try:
            image = Image.open(io.BytesIO(base64.b64decode(encoded, validate=False))).convert("RGB")
        except (binascii.Error, OSError, ValueError) as error:
            raise RuntimeError("omniflow_oob_appagent_screenshot_missing") from error
        path = os.path.join(save_dir, prefix + ".png")
        image.save(path)
        return path

    def get_xml(self, prefix: str, save_dir: str) -> str:
        snapshot = self._snapshot()
        path = os.path.join(save_dir, prefix + ".xml")
        with open(path, "w", encoding="utf-8") as output:
            output.write(str(snapshot["xml"]))
        return path

    def back(self) -> str:
        return self._act({"tool": "press_key", "args": {"key": "back"}})

    def tap(self, x: int | float, y: int | float) -> str:
        nx, ny = self._normalized(x, y)
        return self._act({"tool": "click", "args": {"x": nx, "y": ny}})

    def text(self, input_str: str) -> str:
        return self._act({"tool": "input_text", "args": {"text": str(input_str), "clear_text": True}})

    def long_press(self, x: int | float, y: int | float, duration: int = 1000) -> str:
        nx, ny = self._normalized(x, y)
        return self._act({"tool": "long_press", "args": {"x": nx, "y": ny, "duration_ms": int(duration)}})

    def swipe(self, x: int | float, y: int | float, direction: str, dist: str = "medium", quick: bool = False) -> str:
        unit = self.width / 10
        if dist == "long":
            unit *= 3
        elif dist == "medium":
            unit *= 2
        dx, dy = {"up": (0, -2 * unit), "down": (0, 2 * unit), "left": (-unit, 0), "right": (unit, 0)}.get(direction, (0, 0))
        if not (dx or dy):
            return "ERROR"
        return self.swipe_precise((x, y), (x + dx, y + dy), 100 if quick else 400)

    def swipe_precise(self, start: tuple[int | float, int | float], end: tuple[int | float, int | float], duration: int = 400) -> str:
        x1, y1 = self._normalized(*start)
        x2, y2 = self._normalized(*end)
        return self._act({"tool": "swipe", "args": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "duration_ms": int(duration)}})
