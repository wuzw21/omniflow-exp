from __future__ import annotations

from dataclasses import dataclass, field

from omniflow.core.model import (
    Checker,
    Transfer,
)

DEFAULT_PLANNER_SYSTEM_PROMPT = (
    "You are an Android GUI agent. You are given a task, your action history, and "
    "the current screenshot. Choose the next action to complete the task. Return "
    "exactly one provided tool call. A recalled Function is an action API like click "
    "or swipe. Prefer it when it directly performs the next part of the task. A "
    "Function result returns control to you, so you may call another action next. "
    "Use current-screen raw-pixel coordinates. Put a brief plan and reason for the "
    "next action in summary. Use finished only when the complete task is done."
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
