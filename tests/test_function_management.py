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


def _source_step(
    source_state_id: str,
    tool: str,
    args: dict,
    *,
    success: bool = True,
) -> dict:
    return {
        "before_state_id": source_state_id,
        "action": {"tool": tool, "args": args},
        "result": {"success": success},
    }


def _function_step(source_step: dict, step_index: int = 0) -> dict:
    return {
        "step_index": step_index,
        "source_state_id": source_step["before_state_id"],
        "action": source_step["action"],
    }


def _proposal(
    run_log: dict,
    updates: dict | None = None,
    *,
    roles: list[str] | None = None,
) -> str:
    source_steps = [
        step for step in run_log.get("steps") or () if isinstance(step, dict)
    ]
    selected_roles = roles or ["function"] * len(source_steps)
    return json.dumps(
        {
            "step_decisions": [
                {
                    "step": index,
                    "role": role,
                    "reason": f"Step {index} is {role} evidence.",
                }
                for index, role in enumerate(selected_roles)
            ],
            **dict(updates or {}),
        }
    )


def test_enhancement_instruction_is_included_in_prompt() -> None:
    prompts: list[str] = []

    enhance_function(
        _function(),
        {},
        lambda prompt: prompts.append(prompt) or _proposal({}),
        instruction="Prefer a reusable search-first workflow.",
    )

    assert len(prompts) == 1
    assert "Prefer a reusable search-first workflow." in prompts[0]


def test_enhancement_uses_default_guidance_when_instruction_is_empty() -> None:
    prompts: list[str] = []

    enhance_function(
        _function(),
        {},
        lambda prompt: prompts.append(prompt) or _proposal({}),
    )

    assert len(prompts) == 1
    assert '"user_instruction":""' in prompts[0]
    assert (
        '"step_decisions":[{"step":0,"role":"function",'
        '"reason":"short semantic reason"}]' in prompts[0]
    )
    assert "one compact string" in prompts[0]


def test_enhancement_rejects_out_of_order_step_decisions() -> None:
    run_log = {
        "steps": [
            _source_step("state-1", "click", {"x": 10, "y": 20}),
            _source_step("state-2", "click", {"x": 30, "y": 40}),
        ]
    }

    try:
        enhance_function(
            _function(),
            run_log,
            lambda _prompt: json.dumps(
                {
                    "step_decisions": [
                        {"step": 1, "role": "function", "reason": "Task action."},
                        {"step": 0, "role": "checker", "reason": "Setup action."},
                    ]
                }
            ),
        )
    except ValueError as error:
        assert str(error) == "function_enhancement_step_decisions_out_of_order"
    else:
        raise AssertionError("out-of-order Step decisions must be rejected")


def test_enhancement_persists_semantic_checker_as_step_role() -> None:
    run_log = {
        "steps": [
            _source_step(
                "state-1",
                "open_app",
                {"package_name": "com.android.settings"},
            )
        ]
    }

    enhanced, changes, status = enhance_function(
        _function(),
        run_log,
        lambda _prompt: _proposal(run_log, roles=["checker"]),
    )

    assert enhanced["steps"][0]["role"] == "checker"
    assert enhanced["checker_rules"] == []
    assert {"part": "function", "field": "step_roles"} in changes
    assert status == "enhanced"


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
        lambda prompt: prompts.append(prompt) or _proposal(run_log),
    )

    assert '"source_state_id":"source-state"' in prompts[0]
    assert '"tool":"open_app"' in prompts[0]
    assert '"package_name":"com.android.settings"' in prompts[0]


def test_enhancement_prompt_uses_compact_source_page_semantics() -> None:
    prompts: list[str] = []
    run_log = {
        "goal": "open a new tab in Chrome",
        "steps": [
            {
                **_source_step("chrome-promo", "click", {"x": 806, "y": 695}),
                "step_index": 0,
                "after_state_id": "chrome-home",
                "metadata": {"origin": "action"},
            }
        ],
    }
    states = {
        "chrome-promo": {
            "package_name": "com.android.chrome",
            "activity_name": "com.google.android.apps.chrome.Main",
            "xml": (
                '<hierarchy><node text="Search with Sogou"/>'
                '<node text="OK" content-desc="Confirm search engine"/></hierarchy>'
            ),
        }
    }

    enhance_function(
        _function(),
        run_log,
        lambda prompt: prompts.append(prompt) or _proposal(run_log),
        state_loader=states.get,
    )

    assert '"page_semantics":{"package":"com.android.chrome"' in prompts[0]
    assert '"visible_labels":["Search with Sogou","OK","Confirm search engine"]' in prompts[0]
    assert "<hierarchy>" not in prompts[0]
    assert "does not depend on metadata.origin" in prompts[0]


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
        lambda _prompt: _proposal(
            run_log,
            {
                "parameters": [
                    {
                        "name": "note",
                        "description": "Expense note",
                        "step_index": 0,
                        "arg_name": "text",
                    }
                ]
            },
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
        lambda _prompt: _proposal(
            run_log,
            {
                "parameters": [
                    {
                        "name": "category",
                        "description": "Expense category to select",
                        "step_index": 0,
                        "arg_name": "target_description",
                    }
                ]
            },
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
        lambda _prompt: _proposal(
            run_log,
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
            },
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
        lambda _prompt: _proposal(
            run_log,
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
            },
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


def test_enhancement_can_delete_actions_at_segment_edges() -> None:
    source_steps = [
        _source_step(
            "state-1",
            "open_app",
            {"package_name": "com.example.expense"},
        ),
        _source_step("state-2", "click", {"x": 90, "y": 120}),
        _source_step("state-3", "input_text", {"text": "Paid by card"}),
    ]
    run_log = {"run_id": "source-run", "steps": source_steps}
    current = {
        **_function(),
        "steps": [
            _function_step(step, index) for index, step in enumerate(source_steps)
        ],
    }

    enhanced, _, status = enhance_function(
        current,
        run_log,
        lambda _prompt: _proposal(
            run_log,
            {
                "steps": [
                    _function_step(source_steps[1]),
                    _function_step(source_steps[2]),
                ]
            },
        ),
    )

    assert enhanced["function_id"] == current["function_id"]
    assert [step["source_state_id"] for step in enhanced["steps"]] == [
        "state-2",
        "state-3",
    ]
    assert status == "enhanced"


def test_enhancement_can_reorder_only_to_recorded_source_order() -> None:
    first = _source_step("state-a", "click", {"x": 10, "y": 20})
    second = _source_step("state-b", "click", {"x": 30, "y": 40})
    current = {
        **_function(),
        "steps": [
            _function_step(first),
            _function_step(second, 1),
        ],
    }
    run_log = {"run_id": "source-run", "steps": [second, first]}

    enhanced, _, status = enhance_function(
        current,
        run_log,
        lambda _prompt: _proposal(
            run_log,
            {
                "steps": [
                    _function_step(second),
                    _function_step(first),
                ]
            },
        ),
    )

    assert enhanced["function_id"] == current["function_id"]
    assert [step["source_state_id"] for step in enhanced["steps"]] == [
        "state-b",
        "state-a",
    ]
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
            lambda _prompt: _proposal(
                run_log,
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
                },
            ),
        )
    except ValueError as error:
        assert str(error).startswith("function_action_not_grounded:")
    else:
        raise AssertionError("invented action must be rejected")


def test_enhancement_rejects_successful_actions_separated_by_failure() -> None:
    first = _source_step(
        "state-1",
        "open_app",
        {"package_name": "com.example.expense"},
    )
    last = _source_step("state-3", "input_text", {"text": "Paid by card"})
    run_log = {
        "run_id": "source-run",
        "steps": [
            first,
            _source_step("state-2", "click", {"x": 10, "y": 20}, success=False),
            last,
        ],
    }

    try:
        enhance_function(
            _function(),
            run_log,
            lambda _prompt: _proposal(
                run_log,
                {
                    "steps": [
                        _function_step(first),
                        _function_step(last),
                    ]
                },
            ),
        )
    except ValueError as error:
        assert str(error).startswith("function_action_not_grounded:")
    else:
        raise AssertionError("failed RunLog steps must split successful evidence")


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
        lambda _prompt: _proposal(
            run_log,
            {"checker_rules": [checker_rule]},
            roles=["checker"],
        ),
        instruction="Add only evidence-backed recovery conditions.",
    )

    assert enhanced["checker_rules"] == [checker_rule]
    assert {"part": "function", "field": "checker_rules"} in changes
    assert status == "enhanced"
