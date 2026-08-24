from __future__ import annotations

import json

import pytest
from runlog_fixtures import androidworld_run_log, androidworld_state

from omniflow.bridge import JsonLineBridge
from omniflow.core.model import Function
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
    assert "optional semantic authoring" in save["description"]
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


def test_bridge_rejects_hidden_direct_function_execution(tmp_path) -> None:
    bridge = JsonLineBridge(tmp_path / "functions.json")
    bridge.flow.store.put_function(Function.from_dict(_function()))

    with pytest.raises(ValueError, match="tool_not_exposed:open_settings"):
        bridge._handle(
            "request-1",
            "tools/call",
            {"name": "open_settings", "arguments": {}},
        )


def test_save_function_compiles_run_log_without_model_and_saves_once(tmp_path) -> None:
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
    writes: list[dict] = []
    original_put = bridge.flow.store.put_function

    def put_once(value):
        writes.append(dict(value))
        return original_put(value)

    bridge.flow.store.put_function = put_once

    result = bridge._save_function(
        "request-1",
        {
            "run_id": run_log["run_id"],
        },
    )

    assert set(result) == {"success", "function_ids", "functions", "error"}
    assert result["success"] is True
    assert result["functions"][0]["name"] == "Open Settings and wait."
    assert writes == result["functions"]


def test_save_function_accepts_run_log_object_or_file(tmp_path) -> None:
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
        assert result["success"] is True


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
        "function_ids": ["open_settings"],
        "functions": [semantic],
        "error": None,
    }


def test_agent_enhancement_parameterizes_click_target_description() -> None:
    function = _function()
    function["steps"] = [
        {
            "step_index": 0,
            "source_state_id": "state-number",
            "action": {
                "tool": "click",
                "args": {
                    "target_description": "2",
                    "x": 500,
                    "y": 500,
                },
            },
        }
    ]
    run_log = {
        "run_id": "run-with-number",
        "steps": [
            {
                "before_state_id": "state-number",
                "action": function["steps"][0]["action"],
                "result": {"success": True},
            }
        ],
    }

    enhanced, changes, status = enhance_function(
        function,
        run_log,
        lambda _prompt: json.dumps(
            {
                "parameters": [
                    {
                        "name": "number",
                        "description": "Visible number to click",
                        "step_index": 0,
                        "arg_name": "target_description",
                    }
                ]
            }
        ),
    )

    assert enhanced["input_schema"]["properties"]["number"] == {
        "type": "string",
        "description": "Visible number to click",
    }
    assert enhanced["bindings"] == [
        {
            "source": "$.arguments.number",
            "target": "$.steps[0].action.args.target_description",
        }
    ]
    assert enhanced["steps"][0]["action"]["args"]["target_description"] == ""
    assert {"part": "function", "field": "parameters"} in changes
    assert status == "enhanced"
