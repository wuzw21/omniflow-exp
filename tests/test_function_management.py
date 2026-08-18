from __future__ import annotations

import json

import pytest
from runlog_fixtures import androidworld_run_log, androidworld_state

from omniflow.bridge import JsonLineBridge
from omniflow.functions.assets import FunctionStore, function_authoring_tool, save_function


def _function(function_id: str = "open_settings") -> dict:
    return {
        "schema_version": "omniflow.function.v2",
        "function_id": function_id,
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


def _authoring_run_log() -> dict:
    return androidworld_run_log(
        [
            {"action_type": "click", "x": 500, "y": 500},
            {"action_type": "input_text", "text": "meeting notes"},
            {"action_type": "wait"},
        ],
        observations=[
            androidworld_state(
                "state-checker",
                forest=(
                    '<hierarchy><node text="Dismiss" clickable="true" '
                    'bounds="[400,400][600,600]" /></hierarchy>'
                ),
            ),
            androidworld_state("state-input"),
            androidworld_state("state-ready"),
        ],
        goal="Enter meeting notes.",
    )


def _draft_input(prompt: str) -> dict:
    return json.loads(prompt.split("Draft input:\n", 1)[1].split("\n\n", 1)[0])


def _draft_enhancer(prompt: str, tool: dict) -> str:
    required = tool["function"]["parameters"]["required"]
    if required == ["complete_function", "subsegments"]:
        return json.dumps(
            {
                "complete_function": {
                    "function_id": "complete_note_entry",
                    "name": "Complete note entry",
                    "description": "Enter text and wait.",
                },
                "subsegments": [],
            }
        )

    if required == ["action_edits", "bindings"]:
        return json.dumps(
            {
                "action_edits": [],
                "bindings": [
                    {
                        "function_id": "complete_note_entry",
                        "step_index": 1,
                        "name": "note",
                        "description": "Note text to enter",
                    }
                ]
            }
        )
    assert required == ["checker_steps"]
    return json.dumps(
        {
            "checker_steps": [
                {"function_id": "complete_note_entry", "step_index": 0}
            ]
        }
    )


def test_enhancer_edits_one_draft_in_three_small_stages(tmp_path) -> None:
    prompts: list[str] = []
    result = save_function(
        _authoring_run_log(),
        tmp_path / "store.json",
        enhance=True,
        complete_json=lambda prompt, tool: prompts.append(prompt)
        or _draft_enhancer(prompt, tool),
        instruction="Prefer one reusable text-entry operation.",
    )

    assert result["function_ids"] == ["complete_note_entry"]
    saved = FunctionStore(tmp_path / "store.json").get_function(
        "complete_note_entry"
    )
    assert saved is not None
    assert [step.action.tool for step in saved.steps] == ["input_text", "wait"]
    assert saved.checker_rules[0]["source_state_id"] == "state-checker"
    assert saved.bindings == (
        {
            "source": "$.arguments.note",
            "target": "$.steps[0].action.args.text",
        },
    )
    assert FunctionStore(tmp_path / "store.json").source_calls == [
        {
            "function_id": "complete_note_entry",
            "arguments": {"note": "meeting notes"},
        }
    ]
    assert len(prompts) == 3
    assert "Prefer one reusable text-entry operation." in prompts[0]
    assert "Subsegments are optional" in prompts[0]
    assert "Omit any uncertain candidate" in prompts[0]
    assert "stable precondition and repeatable semantic effect" in prompts[0]
    assert any(
        "current source-state value clicked only to open a picker" in prompt
        for prompt in prompts
    )
    assert any(
        '"goal":"Enter meeting notes."' in prompt
        for prompt in prompts
    )
    assert any(
        '"eligible_parameter_step_indices":[1]' in prompt for prompt in prompts
    )
    assert all('"schema_version":"omniflow.function.v2"' not in p for p in prompts)


def test_enhancer_compiles_large_function_and_reusable_subsegments(tmp_path) -> None:
    run_log = androidworld_run_log(
        [
            {"action_type": "click", "x": 500, "y": 500},
            {"action_type": "input_text", "text": "museum"},
            {"action_type": "click", "x": 700, "y": 700},
            {"action_type": "wait"},
        ],
        observations=[
            androidworld_state("optional-dialog"),
            androidworld_state("search-input"),
            androidworld_state("search-filled"),
            androidworld_state("results"),
        ],
        goal="Dismiss an optional dialog, search for a museum, and show results.",
    )

    stage_calls: list[
        tuple[tuple[str, ...], str, tuple[int, ...], tuple[int, ...]]
    ] = []

    def complete(prompt: str, tool: dict) -> str:
        required = tool["function"]["parameters"]["required"]
        if required == ["complete_function", "subsegments"]:
            return json.dumps(
                {
                    "complete_function": {
                        "function_id": "search_for_a_place",
                        "name": "Search for a place",
                        "description": "Enter a place query and show its results.",
                    },
                    "subsegments": [
                        {
                            "function_id": "enter_search_query",
                            "name": "Enter a search query",
                            "description": "Enter and submit a search query.",
                            "stability_reason": (
                                "A visible search field is the stable precondition; "
                                "entering and submitting text has a repeatable effect, "
                                "and the query is caller-provided."
                            ),
                            "start_step_index": 1,
                            "end_step_index": 3,
                        },
                    ],
                }
            )
        draft = _draft_input(prompt)
        function_id = draft["function"]["function_id"]
        source_indices = tuple(
            action["step_index"] for action in draft["source_actions"]
        )
        eligible_checker_indices = tuple(
            draft.get("eligible_checker_step_indices", [])
        )
        stage_calls.append(
            (tuple(required), function_id, source_indices, eligible_checker_indices)
        )
        if required == ["action_edits", "bindings"]:
            return json.dumps(
                {
                    "action_edits": [],
                    "bindings": (
                        [
                            {
                                "function_id": function_id,
                                "step_index": 1,
                                "name": "query",
                                "description": "Place query to enter",
                            }
                        ]
                        if function_id in {
                            "search_for_a_place",
                            "enter_search_query",
                        }
                        else []
                    ),
                }
            )
        return json.dumps(
            {
                "checker_steps": (
                    [{"function_id": function_id, "step_index": 0}]
                    if function_id == "search_for_a_place"
                    else []
                )
            }
        )

    store_path = tmp_path / "store.json"
    result = save_function(run_log, store_path, enhance=True, complete_json=complete)
    assert result["function_ids"] == [
        "search_for_a_place",
        "enter_search_query",
    ]
    store = FunctionStore(store_path)
    assert len(store.get_function("search_for_a_place").checker_rules) == 1
    assert store.get_function("enter_search_query").checker_rules == ()
    assert store.get_function("enter_search_query").input_schema["required"] == [
        "query"
    ]
    assert stage_calls == [
        (("action_edits", "bindings"), "search_for_a_place", (0, 1, 2, 3), ()),
        (("action_edits", "bindings"), "enter_search_query", (1, 2), ()),
        (("checker_steps",), "search_for_a_place", (0, 1, 2, 3), (0,)),
        (("checker_steps",), "enter_search_query", (1, 2), ()),
    ]


def test_stage_validation_gets_one_small_correction(tmp_path) -> None:
    prompts: list[str] = []
    split_calls = 0

    def complete(prompt: str, tool: dict) -> str:
        nonlocal split_calls
        prompts.append(prompt)
        if tool["function"]["parameters"]["required"] == [
            "complete_function",
            "subsegments",
        ]:
            split_calls += 1
            if split_calls == 1:
                return '{"unexpected":true}'
        return _draft_enhancer(prompt, tool)

    save_function(
        _authoring_run_log(),
        tmp_path / "store.json",
        enhance=True,
        complete_json=complete,
    )
    assert split_calls == 2
    assert "previous small decision was rejected" in prompts[1]
    assert "function_split_contract_invalid" in prompts[1]


def test_stage_validation_allows_three_bounded_attempts(tmp_path) -> None:
    split_calls = 0

    def complete(prompt: str, tool: dict) -> str:
        nonlocal split_calls
        if tool["function"]["parameters"]["required"] == [
            "complete_function",
            "subsegments",
        ]:
            split_calls += 1
            if split_calls < 3:
                return '{"unexpected":true}'
        return _draft_enhancer(prompt, tool)

    save_function(
        _authoring_run_log(),
        tmp_path / "store.json",
        enhance=True,
        complete_json=complete,
    )
    assert split_calls == 3


def test_single_click_subsegment_correction_requests_omission(tmp_path) -> None:
    prompts: list[str] = []
    split_calls = 0

    def complete(prompt: str, tool: dict) -> str:
        nonlocal split_calls
        required = tool["function"]["parameters"]["required"]
        if required == ["complete_function", "subsegments"]:
            prompts.append(prompt)
            split_calls += 1
            if split_calls == 1:
                return json.dumps(
                    {
                        "complete_function": {
                            "function_id": "complete_note_entry",
                            "name": "Complete note entry",
                            "description": "Enter text and wait.",
                        },
                        "subsegments": [
                            {
                                "function_id": "dismiss_dialog",
                                "name": "Dismiss dialog",
                                "description": "Dismiss a dialog.",
                                "stability_reason": (
                                    "A dialog is visible and clicking dismiss closes it."
                                ),
                                "start_step_index": 0,
                                "end_step_index": 1,
                            }
                        ],
                    }
                )
        return _draft_enhancer(prompt, tool)

    save_function(
        _authoring_run_log(),
        tmp_path / "store.json",
        enhance=True,
        complete_json=complete,
    )
    assert split_calls == 2
    assert "Remove every one-click subsegment" in prompts[1]


def test_enhancer_rejects_subsegment_without_stability_reason(tmp_path) -> None:
    def complete(_prompt: str, tool: dict) -> str:
        required = tool["function"]["parameters"]["required"]
        assert required == ["complete_function", "subsegments"]
        return json.dumps(
            {
                "complete_function": {
                    "function_id": "complete_note_entry",
                    "name": "Complete note entry",
                    "description": "Enter note text and wait for the result.",
                },
                "subsegments": [
                    {
                        "function_id": "enter_note",
                        "name": "Enter a note",
                        "description": "Enter note text.",
                        "stability_reason": "",
                        "start_step_index": 1,
                        "end_step_index": 3,
                    }
                ],
            }
        )

    with pytest.raises(
        ValueError, match="function_subsegment_stability_reason_required"
    ):
        save_function(
            _authoring_run_log(),
            tmp_path / "store.json",
            enhance=True,
            complete_json=complete,
        )


def test_enhancer_compiles_source_proven_launcher_click_to_open_app(tmp_path) -> None:
    run_log = androidworld_run_log(
        [{"action_type": "click", "x": 500, "y": 500}],
        observations=[
            androidworld_state(
                "launcher",
                package_name="com.example.launcher",
                forest=(
                    '<hierarchy><node text="Clock" clickable="true" '
                    'bounds="[400,400][600,600]" /></hierarchy>'
                ),
            )
        ],
        goal="Open Clock.",
    )
    run_log["steps"][0]["next_observation"] = androidworld_state(
        "clock",
        package_name="com.example.clock",
    )

    def complete(_prompt: str, tool: dict) -> str:
        required = tool["function"]["parameters"]["required"]
        if required == ["complete_function", "subsegments"]:
            return json.dumps(
                {
                    "complete_function": {
                        "function_id": "open_clock",
                        "name": "Open Clock",
                        "description": "Open the Clock application.",
                    },
                    "subsegments": [],
                }
            )
        if required == ["action_edits", "bindings"]:
            return json.dumps(
                {
                    "action_edits": [
                        {
                            "function_id": "open_clock",
                            "step_index": 0,
                            "operation": "open_app",
                            "value": "com.example.clock",
                        }
                    ],
                    "bindings": [],
                }
            )
        return json.dumps({"checker_steps": []})

    store_path = tmp_path / "store.json"
    save_function(run_log, store_path, enhance=True, complete_json=complete)
    function = FunctionStore(store_path).get_function("open_clock")
    assert function is not None
    assert function.steps[0].action.to_dict() == {
        "tool": "open_app",
        "args": {"package_name": "com.example.clock"},
    }


def test_enhancer_does_not_reedit_an_existing_open_app_action(tmp_path) -> None:
    run_log = androidworld_run_log(
        [{"action_type": "open_app", "app_name": "com.example.contacts"}],
        observations=[
            androidworld_state(
                "launcher",
                package_name="com.example.launcher",
            )
        ],
        goal="Open Contacts.",
    )

    def complete(prompt: str, tool: dict) -> str:
        required = tool["function"]["parameters"]["required"]
        if required == ["complete_function", "subsegments"]:
            return json.dumps(
                {
                    "complete_function": {
                        "function_id": "open_contacts",
                        "name": "Open Contacts",
                        "description": "Open the Contacts application.",
                    },
                    "subsegments": [],
                }
            )
        if required == ["action_edits", "bindings"]:
            draft = _draft_input(prompt)
            eligible = draft.get("eligible_open_app_step_indices")
            return json.dumps(
                {
                    "action_edits": (
                        []
                        if eligible == []
                        else [
                            {
                                "function_id": "open_contacts",
                                "step_index": 0,
                                "operation": "open_app",
                                "value": "com.example.contacts",
                            }
                        ]
                    ),
                    "bindings": [],
                }
            )
        return json.dumps({"checker_steps": []})

    store_path = tmp_path / "store.json"
    save_function(run_log, store_path, enhance=True, complete_json=complete)
    function = FunctionStore(store_path).get_function("open_contacts")
    assert function is not None
    assert function.steps[0].action.to_dict() == {
        "tool": "open_app",
        "args": {"package_name": "com.example.contacts"},
    }


def test_enhancer_binds_source_proven_semantic_target(tmp_path) -> None:
    run_log = androidworld_run_log(
        [{"action_type": "click", "x": 500, "y": 500}],
        observations=[
            androidworld_state(
                "picker",
                forest=(
                    '<hierarchy><node text="6" clickable="true" '
                    'bounds="[400,400][600,600]" /></hierarchy>'
                ),
            )
        ],
        goal="Select hour 6.",
    )

    def complete(_prompt: str, tool: dict) -> str:
        required = tool["function"]["parameters"]["required"]
        if required == ["complete_function", "subsegments"]:
            return json.dumps(
                {
                    "complete_function": {
                        "function_id": "select_hour",
                        "name": "Select an hour",
                        "description": "Select a caller-provided visible hour.",
                    },
                    "subsegments": [],
                }
            )
        if required == ["action_edits", "bindings"]:
            return json.dumps(
                {
                    "action_edits": [
                        {
                            "function_id": "select_hour",
                            "step_index": 0,
                            "operation": "set_target",
                            "value": "6",
                        }
                    ],
                    "bindings": [
                        {
                            "function_id": "select_hour",
                            "step_index": 0,
                            "name": "hour",
                            "description": "Visible hour to select",
                        }
                    ],
                }
            )
        return json.dumps({"checker_steps": []})

    store_path = tmp_path / "store.json"
    save_function(run_log, store_path, enhance=True, complete_json=complete)
    store = FunctionStore(store_path)
    function = store.get_function("select_hour")
    assert function is not None
    assert function.input_schema["required"] == ["hour"]
    assert function.bindings == (
        {
            "source": "$.arguments.hour",
            "target": "$.steps[0].action.args.target_description",
        },
    )
    assert store.source_calls == [
        {"function_id": "select_hour", "arguments": {"hour": "6"}}
    ]


def test_enhancer_rejects_source_state_value_as_parameter(tmp_path) -> None:
    run_log = androidworld_run_log(
        [{"action_type": "click", "x": 500, "y": 500}],
        observations=[
            androidworld_state(
                "current-minute",
                forest=(
                    '<hierarchy><node text="27" clickable="true" '
                    'bounds="[400,400][600,600]" /></hierarchy>'
                ),
            )
        ],
        goal="Create an alarm at 06:30.",
    )

    def complete(_prompt: str, tool: dict) -> str:
        required = tool["function"]["parameters"]["required"]
        if required == ["complete_function", "subsegments"]:
            return json.dumps(
                {
                    "complete_function": {
                        "function_id": "open_minute_picker",
                        "name": "Open minute picker",
                        "description": "Open the minute picker from its current value.",
                    },
                    "subsegments": [],
                }
            )
        if required == ["action_edits", "bindings"]:
            return json.dumps(
                {
                    "action_edits": [
                        {
                            "function_id": "open_minute_picker",
                            "step_index": 0,
                            "operation": "set_target",
                            "value": "27",
                        }
                    ],
                    "bindings": [
                        {
                            "function_id": "open_minute_picker",
                            "step_index": 0,
                            "name": "minute",
                            "description": "Requested alarm minute",
                        }
                    ],
                }
            )
        return json.dumps({"checker_steps": []})

    with pytest.raises(
        ValueError, match="function_parameter_value_not_requested:minute"
    ):
        save_function(
            run_log,
            tmp_path / "store.json",
            enhance=True,
            complete_json=complete,
        )


def test_enhancer_rejects_invented_action_semantics(tmp_path) -> None:
    def invalid(prompt: str, tool: dict) -> str:
        value = json.loads(_draft_enhancer(prompt, tool))
        if "action_edits" in value:
            value["action_edits"] = [
                {
                    "function_id": "complete_note_entry",
                    "step_index": 0,
                    "operation": "set_target",
                    "value": "Invented target",
                }
            ]
        return json.dumps(value)

    with pytest.raises(ValueError, match="function_action_target_not_source_proven"):
        save_function(
            _authoring_run_log(),
            tmp_path / "store.json",
            enhance=True,
            complete_json=invalid,
        )


def test_parameter_binding_must_point_into_its_function_action(tmp_path) -> None:
    def invalid(prompt: str, tool: dict) -> str:
        value = json.loads(_draft_enhancer(prompt, tool))
        if "bindings" in value:
            value["bindings"][0]["step_index"] = 99
        return json.dumps(value)

    with pytest.raises(ValueError, match="function_parameter_step_not_in_function"):
        save_function(
            _authoring_run_log(),
            tmp_path / "store.json",
            enhance=True,
            complete_json=invalid,
        )


def test_checker_registration_is_function_local(tmp_path) -> None:
    def invalid(prompt: str, tool: dict) -> str:
        value = json.loads(_draft_enhancer(prompt, tool))
        if "checker_steps" in value:
            value["checker_steps"][0]["function_id"] = "unknown"
        return json.dumps(value)

    with pytest.raises(ValueError, match="checker_not_registered_on_function"):
        save_function(
            _authoring_run_log(),
            tmp_path / "store.json",
            enhance=True,
            complete_json=invalid,
        )


def test_checker_cannot_replace_a_task_progress_action(tmp_path) -> None:
    run_log = androidworld_run_log(
        [
            {"action_type": "click", "x": 500, "y": 500},
            {"action_type": "wait"},
        ],
        observations=[
            androidworld_state(
                "alarm-page",
                forest=(
                    '<hierarchy><node text="Add alarm" clickable="true" '
                    'bounds="[400,400][600,600]" /></hierarchy>'
                ),
            ),
            androidworld_state("alarm-picker"),
        ],
        goal="Add an alarm.",
    )

    def complete(_prompt: str, tool: dict) -> str:
        required = tool["function"]["parameters"]["required"]
        if required == ["complete_function", "subsegments"]:
            return json.dumps(
                {
                    "complete_function": {
                        "function_id": "add_alarm",
                        "name": "Add alarm",
                        "description": "Open the Add alarm picker.",
                    },
                    "subsegments": [],
                }
            )
        if required == ["action_edits", "bindings"]:
            return json.dumps({"action_edits": [], "bindings": []})
        return json.dumps(
            {
                "checker_steps": [
                    {"function_id": "add_alarm", "step_index": 0}
                ]
            }
        )

    with pytest.raises(ValueError, match="checker_action_is_task_progress"):
        save_function(
            run_log,
            tmp_path / "store.json",
            enhance=True,
            complete_json=complete,
        )


def test_same_source_action_cannot_be_checker_and_formal_across_functions(
    tmp_path,
) -> None:
    def complete(prompt: str, tool: dict) -> str:
        required = tool["function"]["parameters"]["required"]
        if required == ["complete_function", "subsegments"]:
            return json.dumps(
                {
                    "complete_function": {
                        "function_id": "complete_note_entry",
                        "name": "Complete note entry",
                        "description": "Enter meeting notes.",
                    },
                    "subsegments": [
                        {
                            "function_id": "enter_note",
                            "name": "Enter note",
                            "description": "Enter meeting notes.",
                            "stability_reason": (
                                "The note field is stable and the text is parameterized."
                            ),
                            "start_step_index": 0,
                            "end_step_index": 2,
                        }
                    ],
                }
            )
        draft = _draft_input(prompt)
        function_id = draft["function"]["function_id"]
        if required == ["action_edits", "bindings"]:
            return json.dumps({"action_edits": [], "bindings": []})
        return json.dumps(
            {
                "checker_steps": (
                    [{"function_id": function_id, "step_index": 0}]
                    if function_id == "complete_note_entry"
                    else []
                )
            }
        )

    with pytest.raises(
        ValueError, match="checker_action_role_inconsistent_across_functions"
    ):
        save_function(
            _authoring_run_log(),
            tmp_path / "store.json",
            enhance=True,
            complete_json=complete,
        )


def test_tools_expose_one_function_save_interface(tmp_path) -> None:
    bridge = JsonLineBridge(tmp_path / "functions.json")
    definitions = bridge._handle("request-1", "tools/list", {})["tools"]
    assert {item["name"] for item in definitions} == {
        "save_function",
        "list_functions",
        "get_function",
        "delete_function",
        "clear_functions",
        "list_run_logs",
        "get_run_log",
        "get_run_log_state",
        "run_gui",
    }
    save = next(item for item in definitions if item["name"] == "save_function")
    assert "Function draft" in save["description"]


def test_save_function_requires_functions_without_enhance(tmp_path) -> None:
    bridge = JsonLineBridge(tmp_path / "functions.json")
    result = bridge._save_function("request-1", {"run_log": _authoring_run_log()})
    assert result["success"] is False
    assert result["error"]["code"] == "FUNCTIONS_REQUIRED"


def test_save_function_accepts_one_runlog_and_multiple_functions(tmp_path) -> None:
    run_log = androidworld_run_log(
        [{"action_type": "open_app", "app_name": "com.android.settings"}],
        observations=[androidworld_state("state-1")],
        goal="Open Settings.",
    )
    bridge = JsonLineBridge(tmp_path / "functions.json")
    result = bridge._save_function(
        "request-1",
        {
            "run_log": run_log,
            "functions": [
                _function("open_settings"),
                _function("open_system_settings"),
            ],
        },
    )
    assert result["success"] is True
    assert result["function_ids"] == ["open_settings", "open_system_settings"]


def test_bridge_enhancement_edits_one_draft_in_three_stages(tmp_path) -> None:
    required_fields: list[list[str]] = []

    class Bridge(JsonLineBridge):
        def host_call(self, request_id, method, payload):
            assert method == "model_turn"
            request = payload["request"]
            tool = request["tools"][0]
            required_fields.append(tool["function"]["parameters"]["required"])
            assert request["tool_choice"]["function"]["name"] == "edit_function_draft"
            return {
                "tool_calls": [
                    {
                        "function": {
                            "name": "edit_function_draft",
                            "arguments": _draft_enhancer(
                                request["messages"][0]["content"], tool
                            ),
                        }
                    }
                ]
            }

    result = Bridge(tmp_path / "functions.json")._save_function(
        "request-1", {"run_log": _authoring_run_log(), "enhance": True}
    )
    assert result["success"] is True
    assert required_fields == [
        ["complete_function", "subsegments"],
        ["action_edits", "bindings"],
        ["checker_steps"],
    ]


def test_function_authoring_tool_is_three_small_draft_edits() -> None:
    assert function_authoring_tool(stage="split")["function"]["parameters"][
        "required"
    ] == ["complete_function", "subsegments"]
    assert function_authoring_tool(stage="parameters")["function"]["parameters"][
        "required"
    ] == ["action_edits", "bindings"]
    assert function_authoring_tool(stage="checkers")["function"]["parameters"][
        "required"
    ] == ["checker_steps"]
    assert {
        function_authoring_tool(stage=stage)["function"]["name"]
        for stage in ("split", "parameters", "checkers")
    } == {"edit_function_draft"}
    binding_schema = function_authoring_tool(stage="parameters")["function"][
        "parameters"
    ]["properties"]["bindings"]["items"]
    assert binding_schema["required"] == [
        "function_id",
        "step_index",
        "name",
        "description",
    ]
    assert "argument_path" not in binding_schema["properties"]


def test_save_function_reports_the_failed_stage(tmp_path) -> None:
    calls = 0

    def timeout(_prompt: str, _tool: dict) -> str:
        nonlocal calls
        calls += 1
        raise TimeoutError("endpoint did not answer")

    with pytest.raises(
        ValueError,
        match="function_enhancement_split_model_failed:TimeoutError:endpoint did not answer",
    ):
        save_function(
            _authoring_run_log(),
            tmp_path / "store.json",
            enhance=True,
            complete_json=timeout,
        )
    assert calls == 1


def test_save_function_rejects_model_commentary_around_a_draft_edit(tmp_path) -> None:
    with pytest.raises(ValueError, match="function_enhancement_json_invalid"):
        save_function(
            _authoring_run_log(),
            tmp_path / "store.json",
            enhance=True,
            complete_json=lambda prompt, tool: (
                "Here is the result: " + _draft_enhancer(prompt, tool)
            ),
        )
