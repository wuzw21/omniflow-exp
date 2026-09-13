from __future__ import annotations

from dataclasses import dataclass, field

from omniflow.core.model import (
    Checker,
    Transfer,
)

# Kernel defaults have no benchmark filesystem dependency. The AndroidWorld
# harness reads its own protocol and passes the configured limits explicitly.
DEFAULT_MAX_STEPS = 30
DEFAULT_MAX_FUNCTION_TOOLS = 8

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
    def for_method(cls, name: str) -> Experiment:
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
    max_function_tools: int = DEFAULT_MAX_FUNCTION_TOOLS
    checker_enabled: bool = True
    function_memory_enabled: bool = True
    function_reentry_enabled: bool = True
    planner_error_retries: int = 0

    def __post_init__(self):
        if type(self.function_memory_enabled) is not bool:
            raise ValueError("function_memory_enabled_must_be_boolean")
        if type(self.function_reentry_enabled) is not bool:
            raise ValueError("function_reentry_enabled_must_be_boolean")
        if type(self.planner_error_retries) is not int or not 0 <= self.planner_error_retries <= 3:
            raise ValueError("planner_error_retries_must_be_integer_0_to_3")


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
            checker=(configured.checker or default_checker) if self.runtime.checker_enabled else None,
            transfer=configured.transfer or default_transfer,
        )
