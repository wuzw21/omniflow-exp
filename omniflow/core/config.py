from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path

from omniflow.core.model import (
    Checker,
    Transfer,
)

_ANDROIDWORLD_CONFIG_PATH = Path(
    os.environ.get("OMNIFLOW_ANDROIDWORLD_CONFIG")
    or Path(__file__).resolve().parents[2] / "config" / "paper_androidworld.json"
).expanduser()
_ANDROIDWORLD_CONFIG = json.loads(
    _ANDROIDWORLD_CONFIG_PATH.read_text(encoding="utf-8")
)
ANDROIDWORLD_PROTOCOL = dict(_ANDROIDWORLD_CONFIG["protocol"])
DEFAULT_MAX_STEPS = int(ANDROIDWORLD_PROTOCOL["max_steps"])
DEFAULT_MAX_FALLBACK_STEPS = int(ANDROIDWORLD_PROTOCOL["max_fallback_steps"])
DEFAULT_MAX_FUNCTION_TOOLS = int(ANDROIDWORLD_PROTOCOL["max_function_tools"])

GUI_AGENT_RULES = (
    "You are an Android GUI agent. Each turn, analyze the goal, the fresh current UI, and the complete action history, then return exactly one provided tool call.",
    "History is factual: every tool call, including a Function, is one Action and already happened, so preserve its effects and do not repeat it. Read the unified Action history and each Action result as completion evidence. Only the Planner model may return `finished`; when an Action's complete action_list succeeds and its description covers the goal, return `finished` immediately. A local Function Action is only progress, so continue the missing part of the Task.",
    "Return `finished` when the full goal is established by the current UI or by a completed Action's action_list and description. For a completed whole-task Function Action that covers the goal, return `finished` before taking another action; do not swipe, search, or click to look for redundant proof.",
    "A completed local Function does not by itself answer the Task. For a scrollable list without a completed whole-task Function, read title/condition pairs and swipe through unseen rows; do not click read-only rows or per-item condition fields to filter.",
    "Treat `state_changed=false` in an Action result as evidence that the action did not advance the goal. Re-observe the current UI and choose a different visible control or a different action; never repeat the same ineffective click or gesture without new evidence.",
    "`swipe` is a physical drag as well as a scroll: use it for draggable controls such as native SeekBar or Slider widgets, with both endpoints along the control track. A click on a draggable control is not a substitute for changing its value; if a click leaves the requested value unchanged, switch to the matching swipe gesture instead of repeating the click.",
    "For value-setting or state-setting goals without a completed whole-task Function, do not return `finished` after only navigation, scrolling, or an action reported as successful. First verify the requested value or state in the fresh accessibility observation or screenshot.",
    "If the current package is already the target app, do not reopen it or repeat completed setup actions. If a Function stopped, compare its next-step direction with the current UI and recover or resume only when aligned. Otherwise choose one action or Function that advances the missing part.",
    "Use the screenshot for visual identity and accessibility XML for text, state, and bounds. Use current-screen normalized 0..1000 coordinates and never reuse source-device coordinates or filenames.",
    "After every tool result, inspect the fresh observation before deciding again.",
)

DEFAULT_PLANNER_SYSTEM_PROMPT = (
    "You are a GUI agent. You are given a task, your action history, and the current "
    "accessibility observation. "
    + " ".join(GUI_AGENT_RULES)
)


@dataclass(frozen=True)
class Experiment:
    name: str = "ours"

    @classmethod
    def for_method(cls, name: str) -> "Experiment":
        return cls(name=str(name or "ours"))


@dataclass(frozen=True)
class PromptSet:
    planner_system: str = DEFAULT_PLANNER_SYSTEM_PROMPT


@dataclass(frozen=True)
class PluginSet:
    checker: Checker | None = None
    transfer: Transfer | None = None


@dataclass(frozen=True)
class RuntimeSettings:
    max_steps: int = DEFAULT_MAX_STEPS
    max_fallback_steps: int | None = None
    max_function_tools: int = 8


@dataclass(frozen=True)
class OmniFlowConfig:
    prompts: PromptSet = field(default_factory=PromptSet)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    plugins: PluginSet = field(default_factory=PluginSet)

    def resolved_plugins(self) -> PluginSet:
        from omniflow.runtime.execution import default_transfer
        from omniflow.runtime.checker import default_checker

        configured = self.plugins
        return PluginSet(
            checker=configured.checker or default_checker,
            transfer=configured.transfer or default_transfer,
        )
