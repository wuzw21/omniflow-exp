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
    "Accessibility XML is the authoritative evidence for visible controls, state, and bounds.",
    "Choose exactly one provided tool call for the immediate next action on the latest observed screen; if the final target is absent, use a visible navigation control and never guess future layout.",
    "For click and input_text, select an exact A-reference from the encoded accessibility observation; lines without an A-reference are evidence only and are never clickable. The runtime grounds the node center. Only swipe uses normalized 0..1000 coordinates.",
    "Functions are verified multi-step action paths in the same action space as native GUI tools. When a Function matches the task, prefer it because it can complete several actions quickly; if it fails, inspect the result and continue with another Function or native GUI action.",
    "Before every action, inspect the action history or RunLog for a repeated action or alternating action sequence; if it made no progress on the latest screen, do not repeat the loop—choose a different visible control or path, or stop if none remains.",
    "Correct previous action errors from the latest screen; after OmniTransfer failure, choose a fresh action and never reuse source-device coordinates.",
    "For switches and checkboxes, checked=false means off and checked=true means on; never toggle a control that already matches the goal.",
    "Prefer direct search or text input over browsing long menus, history, suggestions, or repeated swipes when a visible search path exists.",
    "When several visible controls mention the goal, prefer the direct control whose label or summary explicitly says it will cause the requested state change; avoid browse-only controls such as history, saved items, or See all unless the direct control is unavailable.",
    "If the previous successful action explicitly intended to complete the goal and its observed effect confirms a state change without contrary evidence, choose finished immediately instead of navigating for redundant verification.",
    "For a state-changing goal, never use answer as a substitute for the required physical change. Choose answer only when the goal explicitly asks for a factual response; otherwise continue until the current UI or a tool result shows the requested mutation, and do not claim success while the target item is still present.",
    "Use finished only when current evidence or a previous tool result proves the full task is complete; report one factual outcome and never claim RunLog or Function registration that the host has not confirmed.",
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

        configured = self.plugins
        return PluginSet(
            checker=configured.checker,
            transfer=configured.transfer or default_transfer,
        )
