from __future__ import annotations

import json
from io import StringIO

from omniflow.functions.artifact import parse_function_artifact
from omniflow.functions.compiler import compile_runlog_to_store
from omniflow.runlog import import_run_log_evidence
from omniflow.bridge import JsonLineBridge


def _state(xml: str = "<hierarchy />") -> dict:
    return {
        "pixels": None,
        "xml": xml,
        "auxiliaries": {"display": {"width": 1000, "height": 1000}},
    }


def _run_log() -> dict:
    before = _state()
    after = _state('<hierarchy><node text="Done" /></hierarchy>')
    return {
        "schema_version": "omniflow.run_log.v1",
        "run_id": "device-registration-test",
        "task_name": "device-registration-test",
        "goal": "点击完成",
        "task_parameters": {},
        "seed": None,
        "status": "succeeded",
        "success": True,
        "validator": {"official": True, "success": True, "reward": 1},
        "provenance": {"kind": "runtime"},
        "steps": [
            {
                "step_index": 0,
                "observation": before,
                "action": {"action_type": "click", "x": 500, "y": 500},
                "result": {"success": True},
                "next_observation": after,
            }
        ],
    }


def test_device_registration_uses_official_python_projection_without_model(
    tmp_path,
) -> None:
    run_log = _run_log()
    report = compile_runlog_to_store(
        run_log,
        tmp_path / "compiled",
        source_states=import_run_log_evidence(run_log)[1],
    )

    assert report["success"] is True
    assert report["model_calls"] == 0
    assert report["function_ids"] == ["complete_source_workflow"]
    function = parse_function_artifact(
        json.loads((tmp_path / "compiled" / "store.json").read_text())[
            "functions"
        ]["complete_source_workflow"]
    )
    assert function.steps[0].action.to_dict() == {
        "tool": "click",
        "args": {"x": 500, "y": 500},
    }


def test_input_text_keeps_source_point_for_omnitransfer_replay(tmp_path) -> None:
    run_log = _run_log()
    run_log["goal"] = "输入完成"
    run_log["steps"][0]["action"] = {
        "action_type": "input_text",
        "text": "完成",
        "x": 500,
        "y": 500,
    }

    report = compile_runlog_to_store(
        run_log,
        tmp_path / "compiled",
        source_states=import_run_log_evidence(run_log)[1],
    )
    function = parse_function_artifact(
        json.loads((tmp_path / "compiled" / "store.json").read_text())["functions"][
            report["function_ids"][0]
        ]
    )

    assert function.steps[0].action.to_dict() == {
        "tool": "input_text",
        "args": {"text": "", "x": 500, "y": 500},
    }
    assert function.bindings[0]["target"] == "$.steps[0].action.args.text"


def test_answer_only_source_retains_app_entry_as_reusable_progress(tmp_path) -> None:
    run_log = _run_log()
    run_log["goal"] = "Open the clock app and report what is visible."
    run_log["steps"] = [
        {
            "step_index": 0,
            "observation": _state(),
            "action": {"action_type": "open_app", "app_name": "clock"},
            "result": {"success": True},
            "next_observation": _state(
                '<hierarchy><node package="com.google.android.deskclock" '
                'text="Clock" /></hierarchy>'
            ),
        },
        {
            "step_index": 1,
            "observation": _state(
                '<hierarchy><node package="com.google.android.deskclock" '
                'text="Clock" /></hierarchy>'
            ),
            "action": {"action_type": "answer", "text": "Clock is open."},
            "result": {"success": True},
            "next_observation": _state(
                '<hierarchy><node package="com.google.android.deskclock" '
                'text="Clock" /></hierarchy>'
            ),
        },
    ]

    report = compile_runlog_to_store(
        run_log,
        tmp_path / "compiled",
        source_states=import_run_log_evidence(run_log)[1],
    )

    function = parse_function_artifact(
        json.loads((tmp_path / "compiled" / "store.json").read_text())["functions"][
            report["function_ids"][0]
        ]
    )
    assert [step.action.to_dict() for step in function.steps] == [
        {
            "tool": "open_app",
            "args": {"package_name": "com.google.android.deskclock"},
        }
    ]


def test_save_function_bridge_accepts_canonical_runlog_without_function_draft(
    tmp_path,
) -> None:
    bridge = JsonLineBridge(
        tmp_path / "store.json",
        reader=StringIO(),
        writer=StringIO(),
    )

    result = bridge._save_function(
        "request-1",
        {
            "run_id": "device-registration-test",
            "run_log": _run_log(),
            "agent_visible": False,
        },
    )

    assert result["success"] is True
    assert result["function_ids"] == ["complete_source_workflow"]
    assert result["functions"][0]["agent_visible"] is False
