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
    if not visible:
        raise ValueError("script_replay_full_function_missing")
    largest_step_count = max(len(function.steps) for function in visible)
    complete = [
        function
        for function in visible
        if len(function.steps) == largest_step_count
    ]
    if len(complete) != 1:
        ids = ",".join(sorted(function.id for function in complete))
        raise ValueError(f"script_replay_full_function_ambiguous:{ids}")

    flow = OmniFlow(
        store_path,
        host=host,
        installed_apps={},
        config=OmniFlowConfig(
            runtime=RuntimeSettings(max_fallback_steps=0),
        ),
    )
    return flow.call_tool(
        ToolCall(complete[0].id, {}),
        experiment=Experiment(name="bmoca"),
    )


__all__ = ["run_script_replay"]
