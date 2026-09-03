"""V-Droid verifier output adapter for the generic OOB tool runtime."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
import json
import math
import re
from typing import Any
import xml.etree.ElementTree as ET

from src.integrations.gui_agent_tools import GuiAgentToolResult, GuiAgentToolRuntime


@dataclass(frozen=True)
class VDroidOutcome:
    success: bool
    finished: bool = False
    status: str = "acted"
    message: str = ""
    selected_index: int | None = None
    tool_result: GuiAgentToolResult | None = None


class VDroidAdapter:
    """Keep V-Droid's candidate verifier while replacing physical execution."""

    def __init__(
        self,
        runtime: GuiAgentToolRuntime,
        *,
        app_resolver: Callable[[str], str | None] | None = None,
        sleeper: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        if not isinstance(runtime, GuiAgentToolRuntime):
            raise TypeError("vdroid_gui_agent_runtime_required")
        self.runtime = runtime
        self.app_resolver = app_resolver
        self.sleeper = sleeper

    async def execute_selected(
        self,
        candidates: Sequence[str | dict[str, Any]],
        scores: Sequence[float],
        observation: dict[str, Any],
    ) -> VDroidOutcome:
        if not candidates or len(candidates) != len(scores):
            raise ValueError("vdroid_candidate_score_shape_invalid")
        numeric_scores = [float(score) for score in scores]
        if not all(math.isfinite(score) for score in numeric_scores):
            raise ValueError("vdroid_candidate_score_invalid")
        selected_index = max(range(len(candidates)), key=numeric_scores.__getitem__)
        outcome = await self.execute_action(candidates[selected_index], observation)
        return VDroidOutcome(
            success=outcome.success,
            finished=outcome.finished,
            status=outcome.status,
            message=outcome.message,
            selected_index=selected_index,
            tool_result=outcome.tool_result,
        )

    async def execute_action(
        self,
        action: str | dict[str, Any],
        observation: dict[str, Any],
    ) -> VDroidOutcome:
        payload = _action_payload(action)
        action_type = str(payload.get("action_type") or "").strip().lower()
        if action_type in {"status", "answer"}:
            goal_status = str(payload.get("goal_status") or "complete").strip().lower()
            success = action_type == "answer" or goal_status in {
                "complete",
                "completed",
                "success",
            }
            return VDroidOutcome(
                success=success,
                finished=True,
                status="finished",
                message=str(payload.get("text") or goal_status),
            )
        if action_type == "wait":
            await self.sleeper(1.0)
            return VDroidOutcome(success=True, status="waited")
        if action_type in {"tool", "omniflow_function"}:
            name = str(payload.get("name") or "").strip()
            arguments = payload.get("arguments")
            if not name or not isinstance(arguments, dict):
                raise ValueError("vdroid_function_call_invalid")
            return _tool_outcome(await self.runtime.call_tool(name, arguments))
        if action_type in {"navigate_back", "navigate_home", "keyboard_enter"}:
            key = {
                "navigate_back": "back",
                "navigate_home": "home",
                "keyboard_enter": "enter",
            }[action_type]
            return _tool_outcome(
                await self.runtime.call_tool("press_key", {"key": key})
            )
        if action_type == "open_app":
            requested = str(payload.get("app_name") or "").strip()
            package = self.app_resolver(requested) if self.app_resolver else None
            if not package:
                return _failed(f"vdroid_app_package_unresolved:{requested}")
            return _tool_outcome(
                await self.runtime.call_tool("open_app", {"package_name": str(package)})
            )

        display, bounds_by_id = _live_geometry(observation)
        if action_type in {"click", "long_press", "input_text"}:
            index = _action_index(payload)
            bounds = bounds_by_id.get(index)
            if bounds is None:
                return _failed(f"vdroid_live_index_unresolved:{index}")
            x, y = _normalized_center(bounds, display)
            if action_type == "input_text":
                result = await self.runtime.call_tool(
                    "input_text",
                    {"text": str(payload.get("text") or ""), "x": x, "y": y},
                )
            else:
                arguments: dict[str, Any] = {"x": x, "y": y}
                if action_type == "long_press":
                    arguments["duration_ms"] = 1000
                result = await self.runtime.call_tool(action_type, arguments)
            return _tool_outcome(result)
        if action_type == "scroll":
            direction = str(payload.get("direction") or "").strip().lower()
            if direction not in {"up", "down", "left", "right"}:
                raise ValueError(f"vdroid_scroll_direction_invalid:{direction}")
            if payload.get("index") is None:
                bounds = (0, 0, display[0], display[1])
            else:
                index = _action_index(payload)
                bounds = bounds_by_id.get(index)
                if bounds is None:
                    return _failed(f"vdroid_live_index_unresolved:{index}")
            arguments = _scroll_arguments(direction, bounds, display)
            return _tool_outcome(await self.runtime.call_tool("swipe", arguments))
        return _failed(f"vdroid_action_unsupported:{action_type or 'missing'}")


def _action_payload(value: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("vdroid_action_json_invalid") from error
    if not isinstance(value, dict):
        raise TypeError("vdroid_action_contract_invalid")
    return dict(value)


def _live_geometry(
    observation: dict[str, Any],
) -> tuple[tuple[int, int], dict[int, tuple[int, int, int, int]]]:
    extra = observation.get("extra")
    display = extra.get("display") if isinstance(extra, dict) else None
    try:
        width = int(display["width"])
        height = int(display["height"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("vdroid_display_invalid") from error
    if width <= 0 or height <= 0:
        raise ValueError("vdroid_display_invalid")
    try:
        root = ET.fromstring(str(observation.get("xml") or ""))
    except ET.ParseError as error:
        raise ValueError("vdroid_live_xml_invalid") from error
    bounds_by_id: dict[int, tuple[int, int, int, int]] = {}
    for node in root.iter("node"):
        node_id = str(node.attrib.get("id") or "").strip()
        if not node_id.isdigit():
            continue
        bounds = _parse_bounds(node.attrib.get("bounds"))
        if bounds is not None:
            bounds_by_id[int(node_id)] = bounds
    return (width, height), bounds_by_id


def _action_index(payload: dict[str, Any]) -> int:
    try:
        index = int(payload["index"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("vdroid_action_index_invalid") from error
    if index < 0:
        raise ValueError("vdroid_action_index_invalid")
    return index


def _parse_bounds(value: Any) -> tuple[int, int, int, int] | None:
    match = re.fullmatch(
        r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]",
        str(value or "").strip(),
    )
    return tuple(int(part) for part in match.groups()) if match else None


def _normalized_center(
    bounds: tuple[int, int, int, int],
    display: tuple[int, int],
) -> tuple[int, int]:
    left, top, right, bottom = bounds
    width, height = display
    return (
        round(((left + right) / 2) / width * 1000),
        round(((top + bottom) / 2) / height * 1000),
    )


def _scroll_arguments(
    direction: str,
    bounds: tuple[int, int, int, int],
    display: tuple[int, int],
) -> dict[str, Any]:
    left, top, right, bottom = bounds
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    endpoints = {
        "down": (center_x, top),
        "up": (center_x, bottom),
        "right": (left, center_y),
        "left": (right, center_y),
    }
    end_x, end_y = endpoints[direction]
    width, height = display
    x1 = round(center_x / width * 1000)
    y1 = round(center_y / height * 1000)
    x2 = round(end_x / width * 1000)
    y2 = round(end_y / height * 1000)
    physical_direction = (
        "right" if x2 > x1 else "left" if x2 != x1 else "down" if y2 > y1 else "up"
    )
    return {
        "direction": physical_direction,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "duration_ms": 400,
    }


def _tool_outcome(result: GuiAgentToolResult) -> VDroidOutcome:
    return VDroidOutcome(
        success=result.success,
        status="acted" if result.success else "failed",
        message=result.error or "",
        tool_result=result,
    )


def _failed(message: str) -> VDroidOutcome:
    return VDroidOutcome(success=False, status="failed", message=message)


__all__ = ["VDroidAdapter", "VDroidOutcome"]
