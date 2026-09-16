from __future__ import annotations

import json
from io import StringIO

from omniflow.functions.artifact import parse_function_artifact
from omniflow.functions.compiler import compile_runlog_to_store
from omniflow.runlog import import_run_log_evidence
from omniflow.bridge import JsonLineBridge, _BridgeHost
from omniflow.transfer.runtime import load_transfer_state_catalog
import pytest


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
        "args": {"text": "完成", "x": 500, "y": 500},
    }
    assert function.bindings == ()


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
            "enhance": False,
        },
    )

    assert result["success"] is True
    assert result["registered"] is True
    assert result["function_ids"] == ["complete_source_workflow"]
    assert result["functions"][0]["agent_visible"] is False
    assert result["function"]["function_id"] == "complete_source_workflow"


def test_save_function_bridge_enhances_and_registers_the_same_function(
    tmp_path,
) -> None:
    run_log = _run_log()
    compiled_root = tmp_path / "source-compiled"
    compile_runlog_to_store(
        run_log,
        compiled_root,
        source_states=import_run_log_evidence(run_log)[1],
    )
    function = json.loads(
        (compiled_root / "store.json").read_text()
    )["functions"]["complete_source_workflow"]
    bridge = JsonLineBridge(
        tmp_path / "store.json",
        reader=StringIO(),
        writer=StringIO(),
    )
    model_calls: list[dict] = []

    def fake_host_call(request_id: str, method: str, payload: dict) -> dict:
        assert method == "complete_json"
        model_calls.append(payload)
        return {
            "content": json.dumps(
                {
                    "name": "增强后的点击流程",
                    "description": "增强后的可复用点击流程",
                    "parameters": [],
                },
                ensure_ascii=False,
            )
        }

    bridge.host_call = fake_host_call  # type: ignore[method-assign]
    result = bridge._save_function(
        "request-enhance",
        {
            "run_id": run_log["run_id"],
            "run_log": run_log,
            "functions": [function],
            "enhance": True,
            "instruction": "让名称更清楚",
        },
    )

    assert result["success"] is True
    assert result["registered"] is True
    assert result["function_id"] == "complete_source_workflow"
    assert result["function"]["function_id"] == "complete_source_workflow"
    assert result["function"]["name"] == "增强后的点击流程"
    assert result["functions"] == [result["function"]]
    assert len(model_calls) == 1
    assert model_calls[0]["max_tokens"] == 512


@pytest.mark.parametrize("options", [{}, {"enhance": True}])
def test_runlog_authoring_uses_host_model_and_compiler_feedback(tmp_path, options) -> None:
    bridge = JsonLineBridge(tmp_path / "store.json", reader=StringIO(), writer=StringIO())
    prompts = []
    proposal = {
        "binding_owner": "agent",
        "reason": "The recorded navigation is stable.",
        "semantic_analysis": {"steps": [{
            "source_step_index": 0, "semantic_kind": "stable",
            "parameter_names": [], "reason": "Fixed navigation target.",
        }]},
        "functions": [],
        "complete_function": {
            "function_id": "open_done", "name": "Open done",
            "description": "Open the completion page.",
            "source_step_indices": [0], "execution_mode": "direct_replay",
            "parameters": [],
        },
    }

    def host(request_id, method, payload):
        assert method == "complete_json"
        assert payload["max_tokens"] == 8192
        assert payload["enable_thinking"] is False
        assert payload["thinking"] == {"type": "disabled"}
        prompts.append(payload["prompt"])
        return {"content": json.dumps({} if len(prompts) == 1 else proposal)}

    bridge.host_call = host
    result = bridge._save_function("author", {"run_log": _run_log(), **options})
    assert result["success"] is True
    assert len(prompts) == 2
    assert '"harness_feedback"' in prompts[1]
    function = parse_function_artifact(result["function"])
    assert function.function_id == "open_done"
    assert function.steps[0].action.to_dict() == {"tool": "click", "args": {"x": 500, "y": 500}}
    assert function.agent_visible is True


def test_rejected_host_authoring_does_not_register_hidden_fallback_as_success(tmp_path) -> None:
    bridge = JsonLineBridge(tmp_path / "store.json", reader=StringIO(), writer=StringIO())
    calls = []

    def host(request_id, method, payload):
        calls.append(payload)
        return {"content": "{}"}

    bridge.host_call = host
    result = bridge._save_function("author", {"run_log": _run_log(), "enhance": True})
    assert result["success"] is False
    assert result["error"]["code"] == "FUNCTION_AUTHORING_REJECTED"
    assert len(calls) == 3
    assert bridge.flow.store.get_function("complete_source_workflow") is None


def test_package_commits_evidence_before_auto_registration(tmp_path) -> None:
    bridge = JsonLineBridge(tmp_path / "store.json", reader=StringIO(), writer=StringIO())
    calls = []
    bridge.host_call = lambda request_id, method, payload: calls.append(method) or {"finished": True}
    def save(request_id, body):
        assert calls == ["finish_run"]
        assert body == {"run_id": "run-1"}
        return {"success": True, "registered": True, "function_id": "registered"}
    bridge._save_function = save
    result = bridge._complete_run("call", {
        "success": True, "run_id": "run-1",
        "post_run_actions": [{"name": "save_function", "arguments": {"run_id": "run-1"}}],
    })
    assert result["auto_registered"] is True
    assert result["registered_function_id"] == "registered"
    assert "post_run_actions" not in result


def test_registration_failure_preserves_successful_device_result(tmp_path) -> None:
    bridge = JsonLineBridge(tmp_path / "store.json", reader=StringIO(), writer=StringIO())
    bridge.host_call = lambda *args: {"finished": True}
    def fail(*args):
        raise ValueError("authoring unavailable")
    bridge._save_function = fail
    result = bridge._complete_run("call", {
        "success": True, "run_id": "run-1",
        "post_run_actions": [{"name": "save_function", "arguments": {"run_id": "run-1"}}],
    })
    assert result["success"] is True
    assert result["auto_registered"] is False
    assert result["registration_error"] == "authoring unavailable"


def test_registered_source_survives_screenshot_cleanup_and_bridge_restart(tmp_path) -> None:
    screenshot = tmp_path / "recording.png"
    screenshot.write_bytes(b"immutable screenshot evidence")
    run_log = _run_log()
    run_log["steps"][0]["observation"]["pixels"] = {
        "path": str(screenshot), "width": 1000, "height": 1000, "mime_type": "image/png",
    }
    path = tmp_path / "registered" / "store.json"
    bridge = JsonLineBridge(path, reader=StringIO(), writer=StringIO())
    result = bridge._save_function("save", {"run_log": run_log, "enhance": False})
    assert result["success"] is True, result
    state_id = result["function"]["steps"][0]["source_state_id"]
    screenshot.unlink()

    restarted = JsonLineBridge(path, reader=StringIO(), writer=StringIO())
    def unavailable_host(*args):
        raise AssertionError("Registered evidence must not depend on old RunLog storage")
    restarted.host_call = unavailable_host
    observation = _BridgeHost(restarted, "replay").get_state(state_id)
    catalog = load_transfer_state_catalog(path.parent / "transfer_states.json")
    assert observation.xml == run_log["steps"][0]["observation"]["xml"]
    from pathlib import Path
    assert Path(catalog[state_id]["screenshot_path"]).read_bytes() == b"immutable screenshot evidence"


def test_state_import_keeps_earlier_evidence_and_rejects_identity_conflict(tmp_path) -> None:
    path = tmp_path / "registered" / "store.json"
    bridge = JsonLineBridge(path, reader=StringIO(), writer=StringIO())
    first = bridge._save_function("first", {"run_log": _run_log(), "enhance": False})
    first_id = first["function"]["steps"][0]["source_state_id"]
    second_log = _run_log()
    second_log["run_id"] = "second-run"
    second_log["steps"][0]["observation"]["xml"] = '<hierarchy page="second" />'
    second = bridge._save_function("second", {"run_log": second_log, "enhance": False})
    second_id = second["function"]["steps"][0]["source_state_id"]
    catalog_path = path.parent / "transfer_states.json"
    before = catalog_path.read_bytes()
    assert {first_id, second_id} <= load_transfer_state_catalog(catalog_path).keys()
    bridge.flow.store.import_transfer_states(catalog_path)
    assert catalog_path.read_bytes() == before
    conflict = json.loads(before)
    conflict["states"][first_id]["xml"] = '<hierarchy overwritten="true" />'
    conflict_path = tmp_path / "conflict.json"
    conflict_path.write_text(json.dumps(conflict))
    with pytest.raises(ValueError, match="function_source_state_conflict"):
        bridge.flow.store.import_transfer_states(conflict_path)
    assert catalog_path.read_bytes() == before
