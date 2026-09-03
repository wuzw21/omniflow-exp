"""Minitap device-controller adapter backed by the generic OOB tool runtime."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
import re
from typing import Any
import xml.etree.ElementTree as ET

from PIL import Image

from src.integrations.gui_agent_tools import GuiAgentToolRuntime


@dataclass(frozen=True)
class MinitapCoordinates:
    x: int
    y: int


@dataclass(frozen=True)
class MinitapBounds:
    x1: int
    y1: int
    x2: int
    y2: int

    def get_center(self) -> MinitapCoordinates:
        return MinitapCoordinates(
            x=(self.x1 + self.x2) // 2,
            y=(self.y1 + self.y2) // 2,
        )


@dataclass(frozen=True)
class MinitapTapOutput:
    error: str | None = None


@dataclass(frozen=True)
class MinitapScreenData:
    base64: str
    elements: list[dict[str, Any]]
    width: int
    height: int
    platform: str = "android"


class MinitapOobController:
    """Implement Minitap's ``MobileDeviceController`` structural protocol.

    The adapter has no ADB/UIAutomator client.  Supported operations are
    translated to canonical 0..1000 actions and sent through
    :class:`GuiAgentToolRuntime`; unsupported operations fail closed.
    """

    def __init__(
        self,
        runtime: GuiAgentToolRuntime,
        *,
        width: int,
        height: int,
    ) -> None:
        if not isinstance(runtime, GuiAgentToolRuntime):
            raise TypeError("minitap_gui_agent_runtime_required")
        self.runtime = runtime
        self.device_width = _positive_dimension(width, "width")
        self.device_height = _positive_dimension(height, "height")
        self._last_point: tuple[int, int] | None = None

    async def tap(
        self,
        coords: Any,
        long_press: bool = False,
        long_press_duration: int = 1000,
    ) -> MinitapTapOutput:
        try:
            point = self._normalized_point(coords)
            result = await self.runtime.call_tool(
                "long_press" if long_press else "click",
                {
                    "x": point[0],
                    "y": point[1],
                    **({"duration_ms": int(long_press_duration)} if long_press else {}),
                },
            )
            if result.success:
                self._last_point = point
                return MinitapTapOutput()
            return MinitapTapOutput(error=result.error or "oob_action_failed")
        except (TypeError, ValueError, RuntimeError) as error:
            return MinitapTapOutput(error=str(error))

    async def swipe(
        self,
        start: Any,
        end: Any,
        duration: int = 400,
    ) -> str | None:
        try:
            x1, y1 = self._normalized_point(start)
            x2, y2 = self._normalized_point(end)
            direction = _swipe_direction(x1, y1, x2, y2)
            result = await self.runtime.call_tool(
                "swipe",
                {
                    "direction": direction,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "duration_ms": int(duration),
                },
            )
            return None if result.success else result.error or "oob_action_failed"
        except (TypeError, ValueError, RuntimeError) as error:
            return str(error)

    async def screenshot(self) -> str:
        return (await self.get_screen_data()).base64

    async def input_text(self, text: str) -> bool:
        if self._last_point is None:
            return False
        result = await self.runtime.call_tool(
            "input_text",
            {"text": str(text), "x": self._last_point[0], "y": self._last_point[1]},
        )
        return result.success

    async def launch_app(self, package_or_bundle_id: str) -> bool:
        result = await self.runtime.call_tool(
            "open_app",
            {"package_name": str(package_or_bundle_id)},
        )
        return result.success

    async def terminate_app(self, package_or_bundle_id: str | None) -> bool:
        del package_or_bundle_id
        return False

    async def open_url(self, url: str) -> bool:
        del url
        return False

    async def press_back(self) -> bool:
        return await self._press_key("back")

    async def press_home(self) -> bool:
        return await self._press_key("home")

    async def press_enter(self) -> bool:
        return await self._press_key("enter")

    async def get_ui_hierarchy(self) -> list[dict[str, Any]]:
        return (await self.get_screen_data()).elements

    def find_element(
        self,
        ui_hierarchy: list[dict[str, Any]],
        resource_id: str | None = None,
        text: str | None = None,
        index: int = 0,
    ) -> tuple[dict[str, Any] | None, MinitapBounds | None, str | None]:
        if not resource_id and not text:
            return None, None, "No resource_id or text provided"
        matches = [
            element
            for element in ui_hierarchy
            if (resource_id and str(element.get("resource-id") or "") == resource_id)
            or (
                text
                and text
                in {
                    str(element.get("text") or ""),
                    str(element.get("accessibilityText") or ""),
                }
            )
        ]
        if not matches:
            return None, None, "No matching element found"
        if index < 0 or index >= len(matches):
            return None, None, f"Index {index} out of range"
        element = matches[index]
        return element, _parse_bounds(element.get("bounds")), None

    async def cleanup(self) -> None:
        return None

    async def erase_text(self, nb_chars: int | None = None) -> bool:
        del nb_chars
        return False

    async def get_screen_data(self) -> MinitapScreenData:
        observation = self.runtime.observe()
        display = (observation.get("extra") or {}).get("display")
        if isinstance(display, dict):
            self.device_width = _positive_dimension(
                display.get("width") or self.device_width,
                "width",
            )
            self.device_height = _positive_dimension(
                display.get("height") or self.device_height,
                "height",
            )
        image = _bare_base64(observation.get("image_base64"))
        if not image:
            raise RuntimeError("minitap_oob_screenshot_missing")
        return MinitapScreenData(
            base64=image,
            elements=_xml_elements(str(observation.get("xml") or "")),
            width=self.device_width,
            height=self.device_height,
        )

    def get_compressed_b64_screenshot(
        self,
        image_base64: str,
        quality: int = 50,
    ) -> str:
        image = Image.open(
            BytesIO(base64.b64decode(_bare_base64(image_base64)))
        ).convert("RGB")
        output = BytesIO()
        image.save(output, format="JPEG", quality=int(quality), optimize=True)
        return base64.b64encode(output.getvalue()).decode("ascii")

    async def start_video_recording(self, max_duration_seconds: int = 900) -> Any:
        del max_duration_seconds
        raise RuntimeError("minitap_oob_video_recording_unsupported")

    async def stop_video_recording(self) -> Any:
        raise RuntimeError("minitap_oob_video_recording_unsupported")

    async def _press_key(self, key: str) -> bool:
        result = await self.runtime.call_tool("press_key", {"key": key})
        return result.success

    def _normalized_point(self, coords: Any) -> tuple[int, int]:
        try:
            x = float(coords.x)
            y = float(coords.y)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("minitap_coordinates_invalid") from error
        if not (0 <= x <= self.device_width and 0 <= y <= self.device_height):
            raise ValueError("minitap_coordinates_out_of_bounds")
        return (
            round(x / self.device_width * 1000),
            round(y / self.device_height * 1000),
        )


def _positive_dimension(value: Any, name: str) -> int:
    try:
        dimension = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"minitap_{name}_invalid") from error
    if dimension <= 0:
        raise ValueError(f"minitap_{name}_invalid")
    return dimension


def _swipe_direction(x1: int, y1: int, x2: int, y2: int) -> str:
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        raise ValueError("minitap_swipe_zero_distance")
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def _xml_elements(xml: str) -> list[dict[str, Any]]:
    if not xml.strip():
        raise RuntimeError("minitap_oob_xml_missing")
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as error:
        raise RuntimeError("minitap_oob_xml_invalid") from error
    elements: list[dict[str, Any]] = []
    for node in root.iter():
        if not node.attrib:
            continue
        element = dict(node.attrib)
        content_description = str(element.get("content-desc") or "")
        if content_description:
            element["accessibilityText"] = content_description
        elements.append(element)
    return elements


def _parse_bounds(value: Any) -> MinitapBounds | None:
    match = re.fullmatch(
        r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]",
        str(value or "").strip(),
    )
    if match is None:
        return None
    return MinitapBounds(*(int(part) for part in match.groups()))


def _bare_base64(value: Any) -> str:
    encoded = str(value or "").strip()
    if encoded.startswith("data:image/") and "," in encoded:
        return encoded.split(",", 1)[1]
    return encoded


__all__ = [
    "MinitapBounds",
    "MinitapCoordinates",
    "MinitapOobController",
    "MinitapScreenData",
    "MinitapTapOutput",
]
