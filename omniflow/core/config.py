from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

from omniflow.core.model import (
    Checker,
    Transfer,
)

_ANDROIDWORLD_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "paper_androidworld.json"
)
_ANDROIDWORLD_CONFIG = json.loads(
    _ANDROIDWORLD_CONFIG_PATH.read_text(encoding="utf-8")
)
ANDROIDWORLD_PROTOCOL = dict(_ANDROIDWORLD_CONFIG["protocol"])
DEFAULT_MAX_STEPS = int(ANDROIDWORLD_PROTOCOL["max_steps"])
DEFAULT_MAX_FALLBACK_STEPS = int(ANDROIDWORLD_PROTOCOL["max_fallback_steps"])
DEFAULT_MAX_FUNCTION_TOOLS = int(ANDROIDWORLD_PROTOCOL["max_function_tools"])

DEFAULT_PLANNER_SYSTEM_PROMPT = (
    "Continue the user's complete goal from the current screen by choosing exactly "
    "one provided GUI tool. Use device-independent relative 0..1000 coordinates on "
    "each axis. XML bounds remain raw pixels in the current original Display frame, "
    "so convert their centers to the relative frame. Transport image resizing does "
    "not change the relative frame. Call finished only "
    "after the complete goal is visibly satisfied. Every coordinate is one scalar "
    "number, never an array, object, or combined coordinate pair. "
    "For open_app, use the exact "
    "package_name supplied by the runtime and never guess one. If "
    "screen_context contains previous_action_error, correct that action through "
    "the same normal tool path. Treat execution_history as the shared history of "
    "all canonical actions, regardless of whether they came from Function replay "
    "or the planner. If an error starts with `omnitransfer_` or says low "
    "confidence, continue from the current screen with a fresh action; do not "
    "abort only because replay mapping failed, and do not reuse source-device "
    "coordinates as target coordinates. Use recent_actions to advance the goal "
    "and never repeat an already successful action on an unchanged screen. Treat "
    "checked=false as an off switch or checkbox and checked=true as on. When "
    "calling finished, keep content to one short factual sentence describing only "
    "the outcome directly supported by the current screen or previous tool result. "
    "Do not claim that a RunLog or reusable Function was registered; the host reports "
    "registration state after execution."
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
    max_steps: int = 20
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
