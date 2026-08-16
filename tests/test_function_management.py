from __future__ import annotations

import json

from runlog_fixtures import androidworld_run_log, androidworld_state

from omniflow.bridge import JsonLineBridge
from omniflow.functions.assets import enhance_function


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


def test_enhancement_prompt_projects_androidworld_actions_with_state_ids() -> None:
    prompts: list[str] = []
    run_log = androidworld_run_log(
        [{"action_type": "open_app", "app_name": "com.android.settings"}],
        observations=[androidworld_state("source-state")],
        goal="Open Settings.",
    )

    enhance_function(
        _function(),
        run_log,
        lambda prompt: prompts.append(prompt) or "{}",
    )

    assert '"source_state_id":"source-state"' in prompts[0]
    assert '"tool":"open_app"' in prompts[0]
    assert '"package_name":"com.android.settings"' in prompts[0]


def test_enhancement_parameterizes_projected_androidworld_input_text() -> None:
    run_log = androidworld_run_log(
        [{"action_type": "input_text", "text": "Paid by card"}],
        observations=[androidworld_state("note-state")],
        goal="Enter an expense note.",
    )
    function = {
        **_function(),
        "function_id": "enter_note",
        "name": "Enter note",
        "description": "Enter one expense note.",
        "steps": [
            {
                "step_index": 0,
                "source_state_id": "note-state",
                "action": {
                    "tool": "input_text",
                    "args": {"text": "Paid by card"},
                },
            }
        ],
    }

    enhanced, _, status = enhance_function(
        function,
        run_log,
        lambda _prompt: json.dumps(
            {
                "parameters": [
                    {
                        "name": "note",
                        "description": "Expense note",
                        "step_index": 0,
                        "arg_name": "text",
                    }
                ]
            }
        ),
    )

    assert enhanced["steps"][0]["action"]["args"] == {"text": ""}
    assert enhanced["bindings"][0]["source"] == "$.arguments.note"
    assert status == "enhanced"


def test_enhancement_parameterizes_projected_androidworld_click_target() -> None:
    run_log = androidworld_run_log(
        [{"action_type": "click", "x": 107, "y": 400}],
        observations=[
            androidworld_state(
                "category-state",
                forest=(
                    '<hierarchy><node bounds="[56,358][158,442]" clickable="true">'
                    '<node text="Food" bounds="[70,370][140,430]"/>'
                    "</node></hierarchy>"
                ),
                width=720,
                height=1280,
            )
        ],
        goal="Add a Food expense.",
    )
    function = {
        **_function(),
        "function_id": "select_expense_category",
        "name": "Select expense category",
        "description": "Select the requested expense category.",
        "steps": [
            {
                "step_index": 0,
                "source_state_id": "category-state",
                "action": {
                    "tool": "click",
                    "args": {
                        "target_description": "Food",
                        "x": 148.61111111111111,
                        "y": 312.5,
                    },
                },
            }
        ],
    }

    enhanced, _, status = enhance_function(
        function,
        run_log,
        lambda _prompt: json.dumps(
            {
                "parameters": [
                    {
                        "name": "category",
                        "description": "Expense category to select",
                        "step_index": 0,
                        "arg_name": "target_description",
                    }
                ]
            }
        ),
    )

    assert enhanced["steps"][0]["action"]["args"]["target_description"] == ""
    assert enhanced["bindings"] == [
        {
            "source": "$.arguments.category",
            "target": "$.steps[0].action.args.target_description",
        }
    ]
    assert status == "enhanced"


def test_enhancement_parameterizes_task_varying_open_app_package() -> None:
    run_log = {
        "run_id": "source-run",
        "steps": [
            {
                "before_state_id": "state-1",
                "action": {
                    "tool": "open_app",
                    "args": {"package_name": "com.android.settings"},
                },
                "result": {"success": True},
            }
        ],
    }

    enhanced, changes, status = enhance_function(
        _function(),
        run_log,
        lambda _prompt: json.dumps(
            {
                "name": "Open requested app",
                "description": "Open the requested installed Android app.",
                "parameters": [
                    {
                        "name": "package_name",
                        "description": "Installed Android package to open",
                        "step_index": 0,
                        "arg_name": "package_name",
                    }
                ],
            }
        ),
    )

    assert enhanced["input_schema"] == {
        "type": "object",
        "properties": {
            "package_name": {
                "type": "string",
                "description": "Installed Android package to open",
            }
        },
        "required": ["package_name"],
        "additionalProperties": False,
    }
    assert enhanced["bindings"] == [
        {
            "source": "$.arguments.package_name",
            "target": "$.steps[0].action.args.package_name",
        }
    ]
    assert enhanced["steps"][0]["action"]["args"] == {"package_name": ""}
    assert {"part": "function", "field": "parameters"} in changes
    assert status == "enhanced"


def test_enhancement_replaces_steps_with_successful_runlog_segment() -> None:
    run_log = {
        "run_id": "source-run",
        "steps": [
            {
                "before_state_id": "state-1",
                "action": {
                    "tool": "open_app",
                    "args": {"package_name": "com.example.expense"},
                },
                "result": {"success": True},
            },
            {
                "before_state_id": "state-2",
                "action": {"tool": "click", "args": {"x": 90, "y": 120}},
                "result": {"success": True},
            },
            {
                "before_state_id": "state-3",
                "action": {
                    "tool": "input_text",
                    "args": {"text": "Paid by card"},
                },
                "result": {"success": True},
            },
        ],
    }

    enhanced, changes, status = enhance_function(
        _function(),
        run_log,
        lambda _prompt: json.dumps(
            {
                "name": "Add expense note",
                "description": "Open the expense app and enter one expense note.",
                "steps": [
                    {
                        "source_state_id": "state-1",
                        "action": {
                            "tool": "open_app",
                            "args": {"package_name": "com.example.expense"},
                        },
                    },
                    {
                        "source_state_id": "state-2",
                        "action": {
                            "tool": "click",
                            "args": {"x": 90, "y": 120},
                        },
                    },
                    {
                        "source_state_id": "state-3",
                        "action": {
                            "tool": "input_text",
                            "args": {"text": "Paid by card"},
                        },
                    },
                ],
                "parameters": [
                    {
                        "name": "note",
                        "description": "Expense note",
                        "step_index": 2,
                        "arg_name": "text",
                    }
                ],
            }
        ),
    )

    assert [step["source_state_id"] for step in enhanced["steps"]] == [
        "state-1",
        "state-2",
        "state-3",
    ]
    assert [step["action"]["tool"] for step in enhanced["steps"]] == [
        "open_app",
        "click",
        "input_text",
    ]
    assert enhanced["steps"][2]["action"]["args"] == {"text": ""}
    assert enhanced["bindings"] == [
        {
            "source": "$.arguments.note",
            "target": "$.steps[2].action.args.text",
        }
    ]
    assert {"part": "function", "field": "steps"} in changes
    assert status == "enhanced"


def test_enhancement_rejects_action_not_grounded_in_runlog() -> None:
    run_log = {
        "run_id": "source-run",
        "steps": [
            {
                "before_state_id": "state-1",
                "action": {"tool": "click", "args": {"x": 90, "y": 120}},
                "result": {"success": True},
            }
        ],
    }

    try:
        enhance_function(
            _function(),
            run_log,
            lambda _prompt: json.dumps(
                {
                    "steps": [
                        {
                            "source_state_id": "state-1",
                            "action": {
                                "tool": "click",
                                "args": {"x": 900, "y": 800},
                            },
                        }
                    ]
                }
            ),
        )
    except ValueError as error:
        assert str(error).startswith("function_action_not_grounded:")
    else:
        raise AssertionError("invented action must be rejected")


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

    assert enhanced["checker_rules"] == [checker_rule]
    assert {"part": "function", "field": "checker_rules"} in changes
    assert status == "enhanced"
