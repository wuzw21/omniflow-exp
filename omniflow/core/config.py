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
    "Observe the current screenshot and accessibility state before acting.",
    "Choose exactly one provided tool for the next action and use current-screen normalized 0..1000 coordinates.",
    "A Function is a normal multi-action tool; use it when it matches the goal and continue from the latest observation if it stops.",
    "Treat successful past actions as already completed. When a completed Function's described result satisfies the goal and the current UI does not contradict it, choose finished immediately; do not repeat visible or disabled save, confirm, or navigation controls.",
    "Use the complete action history and its observed UI facts as memory; for collection or range questions, retain requested values across visited states and answer after the requested scope is covered.",
    "Choose finished with a short result only when the goal is complete; otherwise act.",
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
