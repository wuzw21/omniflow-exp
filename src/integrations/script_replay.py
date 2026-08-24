"""B-MoCA direct-Function baseline using the shared OmniFlow runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omniflow.core.config import Experiment, OmniFlowConfig, RuntimeSettings
from omniflow.core.model import RunResult
from omniflow.functions.store import FunctionStore
from omniflow.runtime.engine import OmniFlow
from omniflow.runtime.sequence import run_function_sequence
from src.experiment.function_v2 import load_v2_source_calls


def run_script_replay(
    *,
    store_path: str | Path,
    host: Any,
) -> RunResult:
    """Execute the compiler-authored Function calls in order through OmniFlow."""

    store = FunctionStore(Path(store_path).expanduser().resolve())
    if store.load_errors:
        raise ValueError(
            "script_replay_function_store_invalid:"
            + ";".join(
                f"{function_id}={error}"
                for function_id, error in sorted(store.load_errors.items())
            )
        )
    source_calls = load_v2_source_calls(store_path)
    if not source_calls:
        raise ValueError("script_replay_function_calls_required")
    for source_call in source_calls:
        function_id = str(source_call.get("function_id") or "").strip()
        function = store.get_function(function_id)
        if function is None or not function.agent_visible:
            raise ValueError(
                f"script_replay_source_call_function_missing:{function_id}"
            )

    flow = OmniFlow(
        store_path,
        host=host,
        installed_apps={},
        config=OmniFlowConfig(
            runtime=RuntimeSettings(max_fallback_steps=0),
        ),
    )
    return run_function_sequence(
        flow,
        source_calls,
        experiment=Experiment(name="bmoca"),
    )


__all__ = ["run_script_replay"]
