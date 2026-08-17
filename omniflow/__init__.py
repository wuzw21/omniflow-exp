"""Adaptive replay for GUI agents."""

from typing import Any

from omniflow.core.config import (
    Experiment,
    OmniFlowConfig,
    PluginSet,
    RuntimeSettings,
)
from omniflow.core.model import (
    Action,
    ActionResult,
    Function,
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
    if name == "save_function":
        from omniflow.functions.assets import save_function

        globals()[name] = save_function
        return save_function
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Action",
    "ActionResult",
    "OMNIFLOW_RUN_LOG_SCHEMA_VERSION",
    "Experiment",
    "FUNCTION_ARTIFACT_VERSION",
    "Function",
    "Host",
    "Observation",
    "OmniFlowConfig",
    "OmniFlow",
    "OmniTransferPageEncoder",
    "PageEmbedding",
    "Planner",
    "PluginSet",
    "RunResult",
    "RuntimeSettings",
    "StepResult",
    "ToolCall",
    "save_function",
    "canonicalize_run_log",
    "canonicalize_run_log_step",
]
