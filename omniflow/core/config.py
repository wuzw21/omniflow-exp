from __future__ import annotations

from dataclasses import dataclass, field

from omniflow.core.model import (
    Checker,
    Transfer,
)

DEFAULT_PLANNER_SYSTEM_PROMPT = (
    "You are a GUI agent. You are given a task and your action history, with screenshots. "
    "Accessibility XML is primary evidence for visible controls and bounds; vision only supplements "
    "missing XML details. Choose exactly one provided tool call for the immediate next action. Act "
    "only on latest observed targets. If the final target is absent, use a visible navigation control; "
    "never guess future layout. Click centers of XML bounds; use visual coordinates only without "
    "reliable XML. Functions are actions in the same action space. Inspect results and avoid "
    "no-progress repeats. Use normalized 0..1000 "
    "coordinates. Use finished only when the full task is complete."
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
        from omniflow.runtime.checker import default_checker
        from omniflow.runtime.execution import default_transfer

        configured = self.plugins
        return PluginSet(
            checker=configured.checker or default_checker,
            transfer=configured.transfer or default_transfer,
        )
