from __future__ import annotations

import json
from pathlib import Path

import pytest
from runlog_fixtures import androidworld_run_log, androidworld_state

from omniflow.functions.compiler import compile_runlog_to_store


def _run_log(step_count: int) -> dict:
    actions = [
        (
            {"action_type": "open_app", "app_name": "com.android.settings"}
            if index == 0
            else {"action_type": "wait"}
        )
        for index in range(step_count)
    ]
    return androidworld_run_log(
        actions,
        observations=[
            androidworld_state(f"state_{index}")
            for index in range(step_count)
        ],
        goal="Open Settings and wait.",
    )


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
    assert result["source_arguments"] == {result["function_ids"][0]: {}}
    assert result["source_calls"] == [
        {"function_id": result["function_ids"][0], "arguments": {}}
    ]
    assert result["model_calls"] == 0
    assert result["prompt_tokens"] == 0
    assert result["completion_tokens"] == 0
    assert result["total_tokens"] == 0


def test_compiler_registers_source_call_for_argumentless_authored_function(
    tmp_path: Path,
) -> None:
    result = compile_runlog_to_store(
        _run_log(2),
        tmp_path / "output",
        function_bundle={
            "schema_version": "omniflow.function-bundle.v2",
            "run_id": "source-run",
            "checker_rules": [],
            "arguments": {},
            "functions": [
                {
                    "schema_version": "omniflow.function.v2",
                    "function_id": "open_settings",
                    "name": "Open Settings",
                    "description": "Open Settings and wait for the page.",
                    "input_schema": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                    "bindings": [],
                    "steps": [
                        {
                            "step_index": 0,
                            "source_state_id": "state_0",
                            "action": {
                                "tool": "open_app",
                                "args": {
                                    "package_name": "com.android.settings"
                                },
                            },
                        },
                        {
                            "step_index": 1,
                            "source_state_id": "state_1",
                            "action": {
                                "tool": "wait",
                                "args": {"duration_ms": 1000},
                            },
                        },
                    ],
                    "agent_visible": True,
                }
            ],
        },
        source_states={
            "state_0": {"state_id": "state_0"},
            "state_1": {"state_id": "state_1"},
        },
    )

    assert result["source_arguments"] == {"open_settings": {}}
    assert result["source_calls"] == [
        {"function_id": "open_settings", "arguments": {}}
    ]
