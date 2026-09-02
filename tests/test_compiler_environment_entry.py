from __future__ import annotations

import json

from omniflow.functions.artifact import parse_function_artifact
from omniflow.functions.compiler import (
    _validate_materialized_function_artifacts,
    compile_runlog_to_store,
)
from omniflow.runlog import import_run_log_evidence


def _state(xml: str) -> dict:
    return {
        "pixels": None,
        "xml": xml,
        "auxiliaries": {"display": {"width": 1000, "height": 1000}},
    }


def _run_log(goal: str, steps: list[dict]) -> dict:
    return {
        "schema_version": "omniflow.run_log.v1",
        "run_id": "environment-entry-test",
        "task_name": "environment-entry-test",
        "goal": goal,
        "task_parameters": {},
        "seed": None,
        "status": "succeeded",
        "success": True,
        "validator": {"official": True, "success": True, "reward": 1},
        "provenance": {"kind": "runtime"},
        "steps": steps,
    }


def _compile(run_log: dict, output_dir) -> tuple[dict, object]:
    report = compile_runlog_to_store(
        run_log,
        output_dir,
        source_states=import_run_log_evidence(run_log)[1],
    )
    function = parse_function_artifact(
        json.loads((output_dir / "store.json").read_text())["functions"][
            report["function_ids"][0]
        ]
    )
    return report, function


def test_launcher_click_opening_initialized_task_app_is_runtime_recovery(
    tmp_path,
) -> None:
    launcher = _state(
        '<hierarchy><node package="com.google.android.apps.nexuslauncher" '
        'text="Camera" /></hierarchy>'
    )
    camera = _state(
        '<hierarchy><node package="com.android.camera2" '
        'content-desc="Shutter" /></hierarchy>'
    )
    recorded = _state(
        '<hierarchy><node package="com.android.camera2" '
        'content-desc="Stop video" /></hierarchy>'
    )
    run_log = _run_log(
        "Take one video.",
        [
            {
                "step_index": 0,
                "observation": launcher,
                "action": {"action_type": "click", "x": 600, "y": 500},
                "result": {"success": True},
                "next_observation": camera,
            },
            {
                "step_index": 1,
                "observation": camera,
                "action": {"action_type": "click", "x": 500, "y": 900},
                "result": {"success": True},
                "next_observation": recorded,
            },
        ],
    )

    report, function = _compile(run_log, tmp_path / "compiled")

    assert [step.action.to_dict() for step in function.steps] == [
        {"tool": "click", "args": {"x": 500, "y": 900}}
    ]
    assert report["optional_checker_actions"] == [
        {
            "source_step_index": 0,
            "checker_id": "restore_target_app",
            "reason": "recorded_launcher_app_entry_is_environment_recovery",
        }
    ]


def test_launcher_systemui_action_remains_business_action(tmp_path) -> None:
    launcher = _state(
        '<hierarchy><node package="com.google.android.apps.nexuslauncher" /></hierarchy>'
    )
    system_ui = _state(
        '<hierarchy><node package="com.android.systemui" '
        'content-desc="Brightness" /></hierarchy>'
    )
    run_log = _run_log(
        "Set brightness to maximum.",
        [
            {
                "step_index": 0,
                "observation": launcher,
                "action": {"action_type": "click", "x": 500, "y": 100},
                "result": {"success": True},
                "next_observation": system_ui,
            },
            {
                "step_index": 1,
                "observation": system_ui,
                "action": {"action_type": "click", "x": 900, "y": 250},
                "result": {"success": True},
                "next_observation": system_ui,
            },
        ],
    )

    _, function = _compile(run_log, tmp_path / "compiled")

    assert [step.action.tool for step in function.steps] == ["click", "click"]


def test_materialized_author_attempt_checks_duplicate_render_targets() -> None:
    function = {
        "schema_version": "omniflow.function.v2",
        "function_id": "edit_note",
        "name": "Edit note",
        "description": "Edit a requested note.",
        "input_schema": {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
            "additionalProperties": False,
        },
        "bindings": [],
        "render_bindings": [
            {
                "source": "$.arguments.title",
                "step_index": 0,
                "node_id": "title",
                "attribute": "text",
                "recorded_value": "Old title",
            },
            {
                "source": "$.arguments.title",
                "step_index": 0,
                "node_id": "title",
                "attribute": "text",
                "recorded_value": "Old title",
            },
        ],
        "steps": [
            {
                "step_index": 0,
                "source_state_id": "state-1",
                "action": {"tool": "click", "args": {"x": 500, "y": 500}},
            }
        ],
        "agent_visible": True,
    }
    authored = {
        "bundle": {
            "schema_version": "omniflow.function-bundle.v2",
            "run_id": "environment-entry-test",
            "arguments": {"edit_note": {"title": "Old title"}},
            "functions": [function],
            "checker_rules": [],
        }
    }
    raw_payload = _run_log(
        "Edit a note.",
        [
            {
                "step_index": 0,
                "observation": _state(
                    '<hierarchy><node id="title" package="app.notes" '
                    'text="Old title" /></hierarchy>'
                ),
                "action": {"action_type": "click", "x": 500, "y": 500},
                "result": {"success": True},
                "next_observation": _state(
                    '<hierarchy><node id="title" package="app.notes" '
                    'text="Old title" /></hierarchy>'
                ),
            }
        ],
    )

    import pytest

    with pytest.raises(ValueError, match="function_render_binding_target_duplicate"):
        _validate_materialized_function_artifacts(
            authored,
            raw_payload=raw_payload,
        )
