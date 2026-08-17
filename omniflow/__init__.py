"""Adaptive replay for GUI agents."""

from typing import Any

from omniflow.core.config import (
    Experiment,
    OmniFlowConfig,
    PluginSet,
    PromptSet,
    RuntimeSettings,
)
from omniflow.core.model import (
    Action,
    ActionResult,
    CheckerContext,
    Function,
    FunctionRouter,
    Host,
    Observation,
    Planner,
    RunResult,
    StepResult,
    ToolCall,
)
from omniflow.core.trajectory import (
    OMNIFLOW_RUN_LOG_SCHEMA_VERSION,
    canonicalize_run_log,
    canonicalize_run_log_step,
)
from omniflow.functions.assets import FUNCTION_ARTIFACT_VERSION
from omniflow.runtime.engine import OmniFlow
from omniflow.transfer.page_embedding import OmniTransferPageEncoder, PageEmbedding


def __getattr__(name: str) -> Any:
    if name == "compile_runlog_to_store":
        from omniflow.functions.assets import compile_runlog_to_store

        globals()[name] = compile_runlog_to_store
        return compile_runlog_to_store
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Action",
    "ActionResult",
    "OMNIFLOW_RUN_LOG_SCHEMA_VERSION",
    "CheckerContext",
    "Experiment",
    "FUNCTION_ARTIFACT_VERSION",
    "Function",
    "FunctionRouter",
    "Host",
    "Observation",
    "OmniFlowConfig",
    "OmniFlow",
    "OmniTransferPageEncoder",
    "PageEmbedding",
    "Planner",
    "PluginSet",
    "PromptSet",
    "RunResult",
    "RuntimeSettings",
    "StepResult",
    "ToolCall",
    "compile_runlog_to_store",
    "canonicalize_run_log",
    "canonicalize_run_log_step",
]
