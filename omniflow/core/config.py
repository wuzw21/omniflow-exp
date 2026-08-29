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
    "History is factual: completed actions and Function steps already happened, so preserve their effects and do not repeat them. Function completion is progress, not proof that the whole Task is complete.",
    "Call `finished` only when the current UI proves the full goal is complete; for list or information tasks, do not treat a partial viewport or a few matching rows as the complete answer. If a Function stopped, compare its next-step direction with the current UI and recover or resume only when aligned. Otherwise choose one action or Function that advances the missing part.",
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

        configured = self.plugins
        return PluginSet(
            checker=configured.checker,
            transfer=configured.transfer or default_transfer,
        )
