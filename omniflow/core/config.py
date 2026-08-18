from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

from omniflow.core.model import Transfer

_ANDROIDWORLD_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "paper_androidworld.json"
)
_ANDROIDWORLD_CONFIG = json.loads(
    _ANDROIDWORLD_CONFIG_PATH.read_text(encoding="utf-8")
)
ANDROIDWORLD_PROTOCOL = dict(_ANDROIDWORLD_CONFIG["protocol"])

GUI_AGENT_RULES = (
    "Accessibility XML is primary evidence for visible controls, state, and bounds; vision only supplements missing XML details.",
    "Choose exactly one provided tool call for the immediate next action on the latest observed screen; if the final target is absent, use a visible navigation control and never guess future layout.",
    "Use normalized 0..1000 coordinates, click centers of XML bounds, and use visual coordinates only when XML is unreliable. For click and input_text, always name the intended visible control in target_description so the Harness can verify and ground it against Accessibility XML.",
    "Functions are verified multi-step action paths in the same action space as native GUI tools. When a Function matches the task, prefer it because it can complete several actions quickly; if it fails, inspect the result and continue with another Function or native GUI action.",
    "Before every action, inspect the action history or RunLog for a repeated action or alternating action sequence; if it made no progress on the latest screen, do not repeat the loop—choose a different visible control or path, or stop if none remains.",
    "Correct previous action errors from the latest screen; after OmniTransfer failure, choose a fresh action and never reuse source-device coordinates.",
    "For switches and checkboxes, checked=false means off and checked=true means on; never toggle a control that already matches the goal.",
    "Prefer direct search or text input over browsing long menus, history, suggestions, or repeated swipes when a visible search path exists.",
    "Use finished only when current evidence or a previous tool result proves the full task is complete; report one factual outcome and never claim RunLog or Function registration that the host has not confirmed.",
)

DEFAULT_MAX_STEPS = int(ANDROIDWORLD_PROTOCOL["max_steps"])
DEFAULT_MAX_FALLBACK_STEPS = int(ANDROIDWORLD_PROTOCOL["max_fallback_steps"])
DEFAULT_MAX_FUNCTION_TOOLS = int(ANDROIDWORLD_PROTOCOL["max_function_tools"])
_CHECKER_CONFIG = dict(ANDROIDWORLD_PROTOCOL["checker"])
DEFAULT_CHECKER_TARGET_THRESHOLD = float(
    _CHECKER_CONFIG["target_probability_threshold"]
)

DEFAULT_PLANNER_SYSTEM_PROMPT = (
    "You are a GUI agent. You are given a task and your action history, with screenshots. "
    + " ".join(GUI_AGENT_RULES)
)


@dataclass(frozen=True)
class Experiment:
    name: str = "ours"

    @classmethod
    def for_method(cls, name: str) -> "Experiment":
        return cls(name=str(name or "ours"))


@dataclass(frozen=True)
class PluginSet:
    transfer: Transfer | None = None


@dataclass(frozen=True)
class RuntimeSettings:
    max_steps: int = DEFAULT_MAX_STEPS
    max_fallback_steps: int | None = DEFAULT_MAX_FALLBACK_STEPS
    max_function_tools: int = DEFAULT_MAX_FUNCTION_TOOLS
    checker_target_threshold: float = DEFAULT_CHECKER_TARGET_THRESHOLD


@dataclass(frozen=True)
class OmniFlowConfig:
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    plugins: PluginSet = field(default_factory=PluginSet)

    def resolved_plugins(self) -> PluginSet:
        from omniflow.runtime.execution import default_transfer

        configured = self.plugins
        return PluginSet(
            transfer=configured.transfer or default_transfer,
        )
