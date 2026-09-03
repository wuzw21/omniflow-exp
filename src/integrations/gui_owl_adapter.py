"""GUI-Owl 1.5 output adapter for OmniFlow's generic GUI-agent runtime."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import json
import re
from typing import Any

from src.integrations.gui_agent_tools import GuiAgentToolResult, GuiAgentToolRuntime


@dataclass(frozen=True)
class GuiOwlOutcome:
    success: bool
    finished: bool = False
    status: str = "acted"
    message: str = ""
    tool_result: GuiAgentToolResult | None = None


class GuiOwlAdapter:
    """Translate upstream GUI-Owl tool calls without using its ADB runner."""

    def __init__(
        self,
        runtime: GuiAgentToolRuntime,
        *,
        app_resolver: Callable[[str], str | None] | None = None,
        sleeper: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        if not isinstance(runtime, GuiAgentToolRuntime):
            raise TypeError("gui_owl_gui_agent_runtime_required")
        self.runtime = runtime
        self.app_resolver = app_resolver
        self.sleeper = sleeper
        self._last_point: tuple[float, float] | None = None

    def tool_definitions(self) -> list[dict[str, Any]]:
        """Return the canonical MCP/OpenAI-compatible tools for prompting."""

        return [tool.to_openai_tool() for tool in self.runtime.list_tools()]

    async def execute_output(self, output_text: str) -> GuiOwlOutcome:
        tool_call = parse_gui_owl_tool_call(output_text)
        name = tool_call["name"]
        arguments = tool_call["arguments"]
        if name != "mobile_use":
            result = await self.runtime.call_tool(name, arguments)
            return _tool_outcome(result)
        return await self._execute_mobile_use(arguments)

    async def _execute_mobile_use(self, arguments: dict[str, Any]) -> GuiOwlOutcome:
        action = (
            str(arguments.get("action") or "").strip().lower().replace("tap", "click")
        )
        if action in {"terminate", "answer"}:
            status = str(arguments.get("status") or "success").strip().lower()
            success = action == "answer" or status in {
                "success",
                "complete",
                "completed",
            }
            return GuiOwlOutcome(
                success=success,
                finished=True,
                status="finished",
                message=str(arguments.get("text") or status),
            )
        if action == "wait":
            seconds = float(arguments.get("time") or 2.0)
            if not 0 <= seconds <= 60:
                raise ValueError("gui_owl_wait_out_of_bounds")
            await self.sleeper(seconds)
            return GuiOwlOutcome(success=True, status="waited")
        if action in {"click", "long_press"}:
            x, y = _coordinate(arguments, "coordinate")
            payload: dict[str, Any] = {"x": x, "y": y}
            if action == "long_press":
                payload["duration_ms"] = round(
                    float(arguments.get("time") or 1.0) * 1000
                )
            result = await self.runtime.call_tool(action, payload)
            if result.success:
                self._last_point = (x, y)
            return _tool_outcome(result)
        if action == "type":
            if self._last_point is None:
                return GuiOwlOutcome(
                    success=False,
                    status="failed",
                    message="gui_owl_input_target_missing",
                )
            result = await self.runtime.call_tool(
                "input_text",
                {
                    "text": str(arguments.get("text") or ""),
                    "x": self._last_point[0],
                    "y": self._last_point[1],
                },
            )
            return _tool_outcome(result)
        if action in {"scroll", "swipe"}:
            x1, y1 = _coordinate(arguments, "coordinate")
            x2, y2 = _coordinate(arguments, "coordinate2")
            result = await self.runtime.call_tool(
                "swipe",
                {
                    "direction": _swipe_direction(x1, y1, x2, y2),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "duration_ms": round(float(arguments.get("time") or 0.4) * 1000),
                },
            )
            return _tool_outcome(result)
        if action == "system_button":
            key = str(arguments.get("button") or "").strip().lower()
            if key not in {"back", "home", "enter"}:
                raise ValueError(f"gui_owl_system_button_unsupported:{key}")
            return _tool_outcome(
                await self.runtime.call_tool("press_key", {"key": key})
            )
        if action in {"open", "open_app"}:
            requested = str(arguments.get("text") or "").strip()
            package_name = self.app_resolver(requested) if self.app_resolver else None
            if not package_name:
                return GuiOwlOutcome(
                    success=False,
                    status="failed",
                    message=f"gui_owl_app_package_unresolved:{requested}",
                )
            return _tool_outcome(
                await self.runtime.call_tool(
                    "open_app",
                    {"package_name": str(package_name)},
                )
            )
        raise ValueError(f"gui_owl_action_unsupported:{action or 'missing'}")


def parse_gui_owl_tool_call(output_text: str) -> dict[str, Any]:
    blocks = re.findall(
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        str(output_text or ""),
        flags=re.DOTALL,
    )
    if len(blocks) != 1:
        raise ValueError("gui_owl_tool_call_count_invalid")
    try:
        payload = json.loads(blocks[0])
    except json.JSONDecodeError as error:
        raise ValueError("gui_owl_tool_call_json_invalid") from error
    if not isinstance(payload, dict) or set(payload) != {"name", "arguments"}:
        raise ValueError("gui_owl_tool_call_contract_invalid")
    name = str(payload.get("name") or "").strip()
    arguments = payload.get("arguments")
    if not name or not isinstance(arguments, dict):
        raise ValueError("gui_owl_tool_call_contract_invalid")
    return {"name": name, "arguments": dict(arguments)}


def _coordinate(arguments: dict[str, Any], name: str) -> tuple[float, float]:
    value = arguments.get(name)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"gui_owl_{name}_invalid")
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError) as error:
        raise ValueError(f"gui_owl_{name}_invalid") from error
    return x, y


def _swipe_direction(x1: float, y1: float, x2: float, y2: float) -> str:
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        raise ValueError("gui_owl_swipe_zero_distance")
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def _tool_outcome(result: GuiAgentToolResult) -> GuiOwlOutcome:
    return GuiOwlOutcome(
        success=result.success,
        status="acted" if result.success else "failed",
        message=result.error or "",
        tool_result=result,
    )


__all__ = ["GuiOwlAdapter", "GuiOwlOutcome", "parse_gui_owl_tool_call"]
