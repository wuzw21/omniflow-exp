from __future__ import annotations

from dataclasses import dataclass, field

from omniflow.core.model import (
    Checker,
    Transfer,
)

GUI_AGENT_RULES = (
    "Accessibility XML is primary evidence for visible controls, state, and bounds; vision only supplements missing XML details.",
    "If the current app does not match the task, use open_app before in-app actions; do not search for target controls inside the wrong app.",
    "Choose exactly one provided tool call for the immediate next action on the latest observed screen; if the final target is absent, use a visible navigation control and never guess future layout.",
    "Use normalized 0..1000 coordinates, click centers of XML bounds, and use visual coordinates only when XML is unreliable.",
    "Functions are actions in the same action space as native GUI tools, and both belong to one shared action history.",
    "Before every action, inspect the action history or RunLog for a repeated action or alternating action sequence; changed=false means no progress, so do not repeat the loop—choose a different visible control or path, or stop if none remains.",
    "Correct previous action errors from the latest screen; after OmniTransfer failure, choose a fresh action and never reuse source-device coordinates.",
    "For switches and checkboxes, checked=false means off and checked=true means on; never toggle a control that already matches the goal.",
    "Prefer direct search or text input over browsing long menus, history, suggestions, or repeated swipes when a visible search path exists.",
    "Use finished immediately when the full task result is already visible in current XML or a previous tool result; report one factual outcome and never claim RunLog or Function registration that the host has not confirmed.",
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
