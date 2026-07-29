from __future__ import annotations

import json
from pathlib import Path

import pytest

from omniflow.functions.compiler import compile_runlog_to_store


def _run_log(step_count: int) -> dict:
    return {
        "schema_version": "omniflow.canonical_run_log.v1",
        "run_id": "source-run",
        "goal": "Open Settings and wait.",
        "status": "succeeded",
        "success": True,
        "steps": [
            {
                "step_index": index,
                "before_state_id": f"state_{index}",
                "action": (
                    {
                        "tool": "open_app",
                        "args": {"package_name": "com.android.settings"},
                    }
                    if index == 0
                    else {"tool": "wait", "args": {"duration_ms": 100}}
                ),
                "result": {"success": True},
                "after_state_id": f"state_{index + 1}",
            }
            for index in range(step_count)
        ],
    }


def test_default_compiler_rejects_one_action_atomic_function(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="default_bundle_actions_required"):
        compile_runlog_to_store(
            _run_log(1),
            tmp_path / "output",
            source_states={"state_0": {"state_id": "state_0"}},
        )


def test_compiler_freezes_only_function_referenced_states(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    result = compile_runlog_to_store(
        _run_log(2),
        output,
        source_states={
            "state_0": {"state_id": "state_0"},
            "state_1": {"state_id": "state_1"},
            "unused": {"state_id": "unused"},
        },
    )

    catalog = json.loads(
        Path(result["transfer_state_catalog"]).read_text(encoding="utf-8")
    )
    assert set(catalog["states"]) == {"state_0", "state_1"}
    assert result["transfer_state_count"] == 2
