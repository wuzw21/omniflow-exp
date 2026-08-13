from __future__ import annotations

import json

from runlog_fixtures import androidworld_run_log, androidworld_state

from omniflow.bridge import JsonLineBridge
from omniflow.functions.management import enhance_function


def _function() -> dict:
    return {
        "schema_version": "omniflow.function.v2",
        "function_id": "open_settings",
        "name": "Open settings",
        "description": "Open Android settings.",
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
                "source_state_id": "state-1",
                "action": {
                    "tool": "open_app",
                    "args": {"package_name": "com.android.settings"},
                },
            }
        ],
        "checker_rules": [],
        "agent_visible": True,
    }


def test_enhancement_instruction_is_included_in_prompt() -> None:
    prompts: list[str] = []

    enhance_function(
        _function(),
        {},
        lambda prompt: prompts.append(prompt) or "{}",
        instruction="Prefer a reusable search-first workflow.",
    )

    assert len(prompts) == 1
    assert "Prefer a reusable search-first workflow." in prompts[0]


def test_enhancement_uses_default_guidance_when_instruction_is_empty() -> None:
    prompts: list[str] = []

    enhance_function(
        _function(),
        {},
        lambda prompt: prompts.append(prompt) or "{}",
    )

    assert len(prompts) == 1
    assert '"user_instruction":""' in prompts[0]


def test_tools_expose_one_function_save_interface(tmp_path) -> None:
    bridge = JsonLineBridge(tmp_path / "functions.json")

    definitions = bridge._handle("request-1", "tools/list", {})["tools"]
    tools = {item["name"] for item in definitions}

    assert "save_function" in tools
    assert "create_function" not in tools
    assert "update_function" not in tools
    assert "convert_run_log" not in tools
    save = next(item for item in definitions if item["name"] == "save_function")
    assert "Agent-authored reusable Function" in save["description"]
    assert "atomic effect" in save["description"]
    assert "fixed choices" in save["description"]
    assert set(save["inputSchema"]["properties"]) == {
        "run_id",
        "run_log",
        "function",
        "arguments",
        "agent_visible",
    }


def test_save_function_accepts_complete_function(tmp_path) -> None:
    bridge = JsonLineBridge(tmp_path / "functions.json")

    result = bridge._handle(
        "request-1",
        "tools/call",
        {"name": "save_function", "arguments": {"function": _function()}},
    )

    assert result == {
        "success": True,
        "function_id": "open_settings",
        "function": _function(),
        "error": None,
    }
    assert bridge.flow.store.get_function("open_settings") is not None


def test_save_function_rejects_run_log_without_agent_authored_function(tmp_path) -> None:
    run_log = androidworld_run_log(
        [
            {"action_type": "open_app", "app_name": "com.android.settings"},
            {"action_type": "wait"},
        ],
        observations=[androidworld_state("state-1"), androidworld_state("state-2")],
        goal="Open Settings and wait.",
    )

    class Bridge(JsonLineBridge):
        def host_call(self, request_id, method, payload):
            if method == "get_run_log":
                return run_log
            if method == "get_state":
                return {"state_id": payload["state_id"]}
            raise AssertionError(method)

    bridge = Bridge(tmp_path / "functions.json")
    result = bridge._save_function(
        "request-1",
        {
            "run_id": run_log["run_id"],
        },
    )

    assert set(result) == {"success", "function_id", "function", "error"}
    assert result["success"] is False
    assert result["error"] == {
        "code": "FUNCTION_SKILL_BUNDLE_REQUIRED",
        "message": "A Function bundle produced by the authoring skill is required",
    }


def test_save_function_requires_agent_for_run_log_object_or_file(tmp_path) -> None:
    run_log = androidworld_run_log(
        [
            {"action_type": "open_app", "app_name": "com.android.settings"},
            {"action_type": "wait"},
        ],
        observations=[androidworld_state("state-1"), androidworld_state("state-2")],
        goal="Open Settings and wait.",
    )
    run_log_path = tmp_path / "source.run_log.json"
    run_log_path.write_text(json.dumps(run_log), encoding="utf-8")

    class Bridge(JsonLineBridge):
        def host_call(self, request_id, method, payload):
            raise AssertionError(method)

    for index, source in enumerate((run_log, str(run_log_path))):
        bridge = Bridge(tmp_path / f"functions-{index}.json")
        result = bridge._save_function("request-1", {"run_log": source})
        assert result["success"] is False
        assert result["error"]["code"] == "FUNCTION_SKILL_BUNDLE_REQUIRED"


def test_save_function_accepts_agent_authored_semantic_function(tmp_path) -> None:
    run_log = androidworld_run_log(
        [
            {"action_type": "open_app", "app_name": "com.android.settings"},
            {"action_type": "wait"},
        ],
        observations=[androidworld_state("state-1"), androidworld_state("state-2")],
        goal="Open Settings and wait.",
    )
    semantic = {
        **_function(),
        "name": "Open Settings and wait",
        "description": "Open Android Settings, then wait for its page to settle.",
        "steps": [
            {
                "step_index": 0,
                "source_state_id": "state-1",
                "action": {
                    "tool": "open_app",
                    "args": {"package_name": "com.android.settings"},
                },
            },
            {
                "step_index": 1,
                "source_state_id": "state-2",
                "action": {"tool": "wait", "args": {"duration_ms": 1000}},
            },
        ],
    }

    class Bridge(JsonLineBridge):
        def host_call(self, request_id, method, payload):
            if method == "get_run_log":
                return run_log
            if method == "get_state":
                return {"state_id": payload["state_id"]}
            raise AssertionError(method)

    bridge = Bridge(tmp_path / "functions.json")

    result = bridge._save_function(
        "request-1",
        {"run_id": run_log["run_id"], "function": semantic, "arguments": {}},
    )

    assert result == {
        "success": True,
        "function_id": "open_settings",
        "function": semantic,
        "error": None,
    }


def test_agent_enhancement_generates_checker_from_runlog_evidence() -> None:
    checker_action = {
        "tool": "click",
        "args": {
            "target_description": "Dismiss the temporary prompt",
            "x": 500,
            "y": 500,
        },
    }
    checker_rule = {
        "schema_version": "omniflow.checker_rule.v1",
        "trigger": 'xml_contains("not now")',
        "source_state_id": "state-checker",
        "action": checker_action,
    }
    run_log = {
        "run_id": "run-with-checker",
        "steps": [
            {
                "before_state_id": "state-checker",
                "action": checker_action,
                "result": {"success": True},
                "metadata": {
                    "origin": "checker",
                    "checker_trigger": 'xml_contains("not now")',
                },
            }
        ],
    }

    enhanced, changes, status = enhance_function(
        _function(),
        run_log,
        lambda _prompt: json.dumps({"checker_rules": [checker_rule]}),
        instruction="Add only evidence-backed recovery conditions.",
    )

    assert enhanced["checker_rules"] == [
        {
            **checker_rule,
            "action": {"tool": "click", "args": {"x": 500, "y": 500}},
        }
    ]
    assert {"part": "function", "field": "checker_rules"} in changes
    assert status == "enhanced"
