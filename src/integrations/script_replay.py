"""B-MoCA direct-Function baseline using the shared OmniFlow runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omniflow.core.config import Experiment, OmniFlowConfig, RuntimeSettings
from omniflow.core.model import RunResult, ToolCall
from omniflow.functions.assets import FunctionStore
from omniflow.runtime.engine import OmniFlow


def run_script_replay(
    *,
    store_path: str | Path,
    host: Any,
) -> RunResult:
    """Select the complete Function and execute it through OmniFlow once."""

    store = FunctionStore(Path(store_path).expanduser().resolve())
    if store.load_errors:
        raise ValueError(
            "script_replay_function_store_invalid:"
            + ";".join(
                f"{function_id}={error}"
                for function_id, error in sorted(store.load_errors.items())
            )
        )
    visible = store.list_functions(include_hidden=False, limit=500)
    if len(visible) != 1 or len(store.source_calls) != 1:
        raise ValueError("script_replay_single_function_required")
    complete = visible[0]
    source_call = store.source_calls[0]
    if source_call["function_id"] != complete.id:
        raise ValueError("script_replay_source_call_function_mismatch")
    arguments = source_call["arguments"]

    flow = OmniFlow(
        store_path,
        host=host,
        installed_apps={},
        config=OmniFlowConfig(
            runtime=RuntimeSettings(max_fallback_steps=0),
        ),
    )
    return flow.call_tool(
        ToolCall(complete.id, arguments),
        experiment=Experiment(name="bmoca"),
    )


__all__ = ["run_script_replay"]
