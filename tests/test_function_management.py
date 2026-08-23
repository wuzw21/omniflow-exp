from __future__ import annotations

import json
from pathlib import Path

import pytest
from runlog_fixtures import androidworld_run_log, androidworld_state

from omniflow.bridge import JsonLineBridge, _BridgeHost
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
    if required == ["complete_function"]:
        return json.dumps(
            {
                "complete_function": {
                    "function_id": "complete_note_entry",
                    "name": "Complete note entry",
                    "description": "Enter text and wait.",
                }
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
    assert "complete semantic Function" in prompts[0]
    assert "must state the goal's final requested state change" in prompts[0]
    assert "not an open-brightness-settings Function" in prompts[0]
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
    assert any(
        "never add set_target for an input_text step" in prompt
        and "fixed Settings route is not caller-varying" in prompt
        for prompt in prompts
    )
    assert all('"schema_version":"omniflow.function.v2"' not in p for p in prompts)


def test_enhancer_can_omit_irrelevant_source_actions(tmp_path) -> None:
    def complete(prompt: str, tool: dict) -> str:
        required = tool["function"]["parameters"]["required"]
        if required == ["complete_function"]:
            return json.dumps(
                {
                    "complete_function": {
                        "function_id": "enter_note_only",
                        "name": "Enter note only",
                        "description": "Enter the requested note without unrelated UI actions.",
                    }
                }
            )
        if required == ["action_edits", "bindings"]:
            return json.dumps(
                {
                    "action_edits": [
                        {
                            "function_id": "enter_note_only",
                            "step_index": 0,
                            "operation": "omit",
                        },
                        {
                            "function_id": "enter_note_only",
                            "step_index": 2,
                            "operation": "omit",
                        },
                    ],
                    "bindings": [
                        {
                            "function_id": "enter_note_only",
                            "step_index": 1,
                            "name": "note",
                            "description": "Note text to enter",
                        }
                    ],
                }
            )
        return json.dumps({"checker_steps": []})

    result = save_function(
        _authoring_run_log(),
        tmp_path / "store.json",
        enhance=True,
        complete_json=complete,
    )

    store = FunctionStore(result["store_path"])
    function = store.functions["enter_note_only"]
    assert len(function.steps) == 1
    assert function.steps[0].action.tool == "input_text"
    assert store.source_calls == [
        {"function_id": "enter_note_only", "arguments": {"note": "meeting notes"}}
    ]


def test_enhancer_does_not_parameterize_fixed_settings_search_route(tmp_path) -> None:
    run_log = androidworld_run_log(
        [{"action_type": "input_text", "text": "brightness"}],
        observations=[
            androidworld_state(
                "settings-search",
                forest=(
                    '<hierarchy><node text="Search settings" '
                    'class="android.widget.EditText" editable="true" '
                    'bounds="[0,0][1000,200]" /></hierarchy>'
                ),
            )
        ],
        goal="Turn brightness to the max value.",
    )
    parameter_draft: dict = {}

    def complete(prompt: str, tool: dict) -> str:
        required = tool["function"]["parameters"]["required"]
        if required == ["complete_function"]:
            return json.dumps(
                {
                    "complete_function": {
                        "function_id": "set_brightness_to_max",
                        "name": "Set brightness to max",
                        "description": "Use the fixed Settings route for brightness.",
                    }
                }
            )
        if required == ["action_edits", "bindings"]:
            parameter_draft.update(_draft_input(prompt))
            return json.dumps({"action_edits": [], "bindings": []})
        return json.dumps({"checker_steps": []})

    store_path = tmp_path / "store.json"
    save_function(run_log, store_path, enhance=True, complete_json=complete)

    assert parameter_draft["eligible_parameter_step_indices"] == []
    function = FunctionStore(store_path).get_function("set_brightness_to_max")
    assert function is not None
    assert function.input_schema["properties"] == {}
    assert function.steps[0].action.to_dict() == {
        "tool": "input_text",
        "args": {"text": "brightness"},
    }


def test_enhancer_accepts_runlog_grounded_direct_actions(tmp_path) -> None:
    def complete(prompt: str, tool: dict) -> str:
        required = tool["function"]["parameters"]["required"]
        if required == ["complete_function"]:
            return json.dumps(
                {
                    "complete_function": {
                        "function_id": "complete_note_entry",
                        "name": "Complete note entry",
                        "description": "Enter text and wait.",
                    }
                }
            )
        if required == ["action_edits", "bindings"]:
            return json.dumps(
                {
                    "action_edits": [],
                    "bindings": [],
                    "actions": [
                        {
                            "function_id": "complete_note_entry",
                            "step_index": 1,
                            "action": {
                                "tool": "input_text",
                                "args": {"text": "meeting notes"},
                            },
                        }
                    ],
                }
            )
        return json.dumps({"checker_steps": []})

    save_function(
        _authoring_run_log(),
        tmp_path / "store.json",
        enhance=True,
        complete_json=complete,
    )
    saved = FunctionStore(tmp_path / "store.json").get_function(
        "complete_note_entry"
    )
    assert saved is not None
    assert saved.steps[1].action.tool == "input_text"
    assert saved.steps[1].action.args["text"] == "meeting notes"


def test_enhancer_compiles_one_large_function(tmp_path) -> None:
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
        if required == ["complete_function"]:
            return json.dumps(
                {
                    "complete_function": {
                        "function_id": "search_for_a_place",
                        "name": "Search for a place",
                        "description": "Enter a place query and show its results.",
                    }
                }
            )
        draft = _draft_input(prompt)
        function_id = "search_for_a_place"
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
                        if function_id == "search_for_a_place"
                        else []
                    ),
                }
            )
        return json.dumps(
            {
                "checker_steps": (
                    [{"function_id": function_id, "step_index": 0}]
                    if function_id == "search_for_a_place" else []
                )
            }
        )

    store_path = tmp_path / "store.json"
    result = save_function(run_log, store_path, enhance=True, complete_json=complete)
    assert result["function_ids"] == ["search_for_a_place"]
    store = FunctionStore(store_path)
    assert len(store.get_function("search_for_a_place").checker_rules) == 1
    assert store.source_calls == [
        {
            "function_id": "search_for_a_place",
            "arguments": {"query": "museum"},
        }
    ]
    assert stage_calls == [
        (("action_edits", "bindings"), "search_for_a_place", (0, 1, 2, 3), ()),
        (("checker_steps",), "search_for_a_place", (0, 1, 2, 3), (0, 2)),
    ]


def test_stage_validation_can_make_multiple_small_corrections(tmp_path) -> None:
    prompts: list[str] = []
    split_calls = 0

    def complete(prompt: str, tool: dict) -> str:
        nonlocal split_calls
        prompts.append(prompt)
        if tool["function"]["parameters"]["required"] == [
            "complete_function",
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
    assert "previous small decision was rejected" in prompts[1]
    assert "function_split_contract_invalid" in prompts[1]
    assert "previous small decision was rejected" in prompts[2]
    assert "function_split_contract_invalid" in prompts[2]


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
        if required == ["complete_function"]:
            return json.dumps(
                {
                    "complete_function": {
                        "function_id": "open_clock",
                        "name": "Open Clock",
                        "description": "Open the Clock application.",
                    }
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
        if required == ["complete_function"]:
            return json.dumps(
                {
                    "complete_function": {
                        "function_id": "open_contacts",
                        "name": "Open Contacts",
                        "description": "Open the Contacts application.",
                    }
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


def test_function_compiler_resolves_native_androidworld_app_labels(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "src.integrations.android_world.apps.resolve_androidworld_package",
        lambda name: (
            "com.google.android.deskclock" if name == "Clock" else name
        ),
    )
    run_log = androidworld_run_log(
        [{"action_type": "open_app", "app_name": "Clock"}],
        observations=[androidworld_state("launcher")],
        goal="Open Clock.",
    )

    store_path = tmp_path / "store.json"
    save_function(run_log, store_path)
    function = FunctionStore(store_path).get_function(
        "replay_task"
    )
    assert function is not None
    assert function.steps[0].action.to_dict() == {
        "tool": "open_app",
        "args": {"package_name": "com.google.android.deskclock"},
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
        if required == ["complete_function"]:
            return json.dumps(
                {
                    "complete_function": {
                        "function_id": "select_hour",
                        "name": "Select an hour",
                        "description": "Select a caller-provided visible hour.",
                    }
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
        if required == ["complete_function"]:
            return json.dumps(
                {
                    "complete_function": {
                        "function_id": "open_minute_picker",
                        "name": "Open minute picker",
                        "description": "Open the minute picker from its current value.",
                    }
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


def test_enhancer_rejects_agent_authored_action_value(tmp_path) -> None:
    def invalid(prompt: str, tool: dict) -> str:
        value = json.loads(_draft_enhancer(prompt, tool))
        if "action_edits" in value:
            value["action_edits"] = [
                {
                    "function_id": "complete_note_entry",
                    "step_index": 0,
                    "operation": "set_target",
                    "value": "Agent must not copy or invent evidence",
                }
            ]
        return json.dumps(value)

    with pytest.raises(ValueError, match="function_action_edit_contract_invalid"):
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
        if required == ["complete_function"]:
            return json.dumps(
                {
                    "complete_function": {
                        "function_id": "add_alarm",
                        "name": "Add alarm",
                        "description": "Open the Add alarm picker.",
                    }
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


def test_save_function_compiles_without_supplied_functions(tmp_path) -> None:
    bridge = JsonLineBridge(tmp_path / "functions.json")
    result = bridge._save_function("request-1", {"run_log": _authoring_run_log()})
    assert result["success"] is True, result
    assert result["function_ids"] == ["replay_task"]
    assert result["transfer_state_count"] == 3
    assert result["transfer_state_catalog"].endswith("transfer_states.json")


def test_save_function_accepts_compact_xml_screenshot_runlog(tmp_path) -> None:
    run_log = _authoring_run_log()
    source_root = tmp_path / "source"
    source_root.mkdir()
    for index, step in enumerate(run_log["steps"]):
        native_observation = step["observation"]
        screenshot_path = source_root / f"screen-{index}.png"
        screenshot_path.write_bytes(b"compact-screenshot")
        step["observation"] = {
            "screenshot": {
                "path": str(screenshot_path.resolve()),
                "width": 1000,
                "height": 1000,
                "mime_type": "image/png",
            },
            "xml": native_observation["forest"],
        }

    source_run_log = source_root / "run_log.json"
    source_run_log.write_text(json.dumps(run_log), encoding="utf-8")
    bundle_root = tmp_path / "memory"
    bridge = JsonLineBridge(bundle_root / "functions.json")
    result = bridge._save_function(
        "request-1", {"run_log": str(source_run_log)}
    )

    assert result["success"] is True, result
    assert result["function_ids"] == ["replay_task"]
    assert result["transfer_state_count"] == 2
    bundled_run_log = json.loads((bundle_root / "run_log.json").read_text())
    bundled_screenshots = [
        Path(step["observation"]["screenshot"]["path"])
        for step in bundled_run_log["steps"]
    ]
    assert all(path.parent == bundle_root / "screenshots" for path in bundled_screenshots)
    assert all(path.is_file() for path in bundled_screenshots)


def test_function_replay_reads_frozen_bundle_state_before_android_host(tmp_path) -> None:
    class Bridge(JsonLineBridge):
        def host_call(self, request_id, method, payload):
            raise AssertionError(f"legacy host lookup should not run: {method}")

    store_path = tmp_path / "functions.json"
    (tmp_path / "transfer_states.json").write_text(
        json.dumps(
            {
                "schema_version": "omniflow.transfer-state-catalog.v1",
                "run_id": "run-1",
                "states": {
                    "state-1": {
                        "state_id": "state-1",
                        "xml": "<hierarchy />",
                        "package_name": "com.example.app",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    observation = _BridgeHost(Bridge(store_path), "request-1").get_state("state-1")

    assert observation.extra["state_id"] == "state-1"
    assert observation.package_name == "com.example.app"


def test_old_function_store_still_uses_android_host_state_compatibility(tmp_path) -> None:
    class Bridge(JsonLineBridge):
        def host_call(self, request_id, method, payload):
            assert method == "get_state"
            assert payload == {"state_id": "state-old"}
            return {"state_id": "state-old", "xml": "<hierarchy />"}

    observation = _BridgeHost(
        Bridge(tmp_path / "functions.json"), "request-1"
    ).get_state("state-old")

    assert observation.extra["state_id"] == "state-old"


def test_save_function_rejects_multiple_functions(tmp_path) -> None:
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
    assert result["success"] is False
    assert result["error"]["code"] == "RUN_LOG_COMPILE_FAILED"
    assert "function_single_function_required" in result["error"]["message"]


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
        ["complete_function"],
        ["action_edits", "bindings"],
        ["checker_steps"],
    ]


def test_bridge_enhancement_accepts_submit_json_transport_alias(tmp_path) -> None:
    class Bridge(JsonLineBridge):
        def host_call(self, request_id, method, payload):
            assert method == "model_turn"
            request = payload["request"]
            tool = request["tools"][0]
            return {
                "tool_calls": [
                    {
                        "function": {
                            "name": "submit_json",
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


def test_bridge_enhancement_retries_tools_search_transport_misroute(tmp_path) -> None:
    calls = 0

    class Bridge(JsonLineBridge):
        def host_call(self, request_id, method, payload):
            nonlocal calls
            calls += 1
            assert method == "model_turn"
            request = payload["request"]
            tool = request["tools"][0]
            if calls == 1:
                return {
                    "tool_calls": [
                        {"function": {"name": "tools_search", "arguments": ""}}
                    ]
                }
            if calls == 2:
                assert request["messages"][-1]["content"].startswith(
                    "Do not call tools_search."
                )
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
    assert calls == 4


def test_function_authoring_tool_is_three_small_draft_edits() -> None:
    assert function_authoring_tool(stage="split")["function"]["parameters"][
        "required"
    ] == ["complete_function"]
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
    action_edit_schema = function_authoring_tool(stage="parameters")["function"][
        "parameters"
    ]["properties"]["action_edits"]["items"]
    assert action_edit_schema["required"] == [
        "function_id",
        "step_index",
        "operation",
    ]
    assert "value" not in action_edit_schema["properties"]
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
