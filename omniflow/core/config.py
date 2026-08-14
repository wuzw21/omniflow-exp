from __future__ import annotations

from dataclasses import dataclass, field

from omniflow.core.model import (
    Checker,
    Transfer,
)

DEFAULT_PLANNER_SYSTEM_PROMPT = (
    "You are a GUI agent. Given the task, history, results, and latest screenshot, choose exactly "
    "one provided tool call. Tool schemas define tools and arguments. Functions are peer action "
    "APIs like click and swipe; prefer one when it "
    "directly covers the next subgoal. Function success returns control and does not finish the "
    "task. Check whether the previous action produced the expected state. Do not repeat a failed "
    "strategy or loop without progress. Use normalized 0..1000 coordinates. Keep needed facts and "
    "unfinished work in summary. Use finished only when the full task is complete."
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
