from __future__ import annotations

import json

import pytest
from runlog_fixtures import androidworld_run_log, androidworld_state

from omniflow.bridge import JsonLineBridge
from omniflow.functions.assets import (
    FunctionStore,
    function_authoring_tool,
    save_function,
)


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
        goal="Dismiss an optional prompt and enter meeting notes.",
    )


def _semantic_plan(stage: str = "checkers") -> str:
    function = {
        "schema_version": "omniflow.function.v2",
        "function_id": "enter_note",
        "name": "Enter a note",
        "description": (
            "Dismiss an optional prompt, enter task-provided text, and wait "
            "for the page to settle."
        ),
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
                "source_state_id": "state-checker",
                "action": {"tool": "click", "args": {"x": 500, "y": 500}},
            },
            {
                "step_index": 1,
                "source_state_id": "state-input",
                "action": {
                    "tool": "input_text",
                    "args": {"text": "meeting notes"},
                },
            },
            {
                "step_index": 2,
                "source_state_id": "state-ready",
                "action": {"tool": "wait", "args": {"duration_ms": 1000}},
            },
        ],
        "checker_rules": [],
        "agent_visible": True,
    }
    arguments: dict[str, object] = {}
    if stage in {"parameters", "checkers"}:
        function["input_schema"] = {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "Note text to enter",
                }
            },
            "required": ["note"],
            "additionalProperties": False,
        }
        function["bindings"] = [
            {
                "source": "$.arguments.note",
                "target": "$.steps[1].action.args.text",
            }
        ]
        function["steps"][1]["action"]["args"]["text"] = ""
        arguments = {"note": "meeting notes"}
    if stage == "checkers":
        checker_step = function["steps"].pop(0)
        for index, step in enumerate(function["steps"]):
            step["step_index"] = index
        function["bindings"][0]["target"] = "$.steps[0].action.args.text"
        function["checker_rules"] = [
            {
                "source_state_id": checker_step["source_state_id"],
                "action": checker_step["action"],
            }
        ]
    return json.dumps(
        {
            "functions": [
                function
            ],
            "arguments": {"enter_note": arguments},
        }
    )


def _stage_from_prompt(prompt: str) -> str:
    return next(
        stage for stage in ("split", "parameters", "checkers")
        if f"stage {stage}" in prompt
    )


def test_enhance_creates_semantics_parameters_and_checker(tmp_path) -> None:
    prompts: list[str] = []
    store_path = tmp_path / "store.json"

    result = save_function(
        _authoring_run_log(),
        store_path,
        enhance=True,
        complete_json=lambda prompt, _tool: prompts.append(prompt)
        or _semantic_plan(_stage_from_prompt(prompt)),
        instruction="Prefer one reusable text-entry operation.",
    )

    assert result["function_ids"] == ["enter_note"]
    saved = FunctionStore(store_path).get_function("enter_note")
    assert saved is not None
    assert "enter task-provided text" in saved.description
    assert saved.input_schema["required"] == ["note"]
    assert saved.bindings == (
        {
            "source": "$.arguments.note",
            "target": "$.steps[0].action.args.text",
        },
    )
    assert saved.checker_rules[0]["source_state_id"] == "state-checker"
    assert FunctionStore(store_path).source_calls == [
        {"function_id": "enter_note", "arguments": {"note": "meeting notes"}}
    ]
    assert len(prompts) == 3
    assert "stage split" in prompts[0]
    assert "stage parameters" in prompts[1]
    assert "stage checkers" in prompts[2]
    assert "Prefer one reusable text-entry operation." in prompts[0]
    assert '"function_id":"submit_web_search"' in prompts[0]
    assert "Identify every reusable contiguous semantic subsegment" in prompts[0]


def test_enhance_rejects_extra_function_fields(tmp_path) -> None:
    plan = json.loads(_semantic_plan())
    plan["functions"][0]["actions"] = [{"tool": "click", "args": {}}]
    try:
        save_function(
            _authoring_run_log(),
            tmp_path / "store.json",
            enhance=True,
            complete_json=lambda _prompt, _tool: json.dumps(plan),
        )
    except ValueError as error:
        assert str(error) == "function_artifact_unknown_fields:actions"
    else:
        raise AssertionError("Unknown Agent output must be rejected")


def test_enhance_rejects_extra_parameter_schema_fields(tmp_path) -> None:
    def complete_json(prompt: str, _tool: dict) -> str:
        stage = _stage_from_prompt(prompt)
        plan = json.loads(_semantic_plan(stage))
        if stage == "parameters":
            plan["functions"][0]["input_schema"]["title"] = "Uncontrolled schema"
        return json.dumps(plan)

    with pytest.raises(
        ValueError,
        match="function_parameter_schema_unknown_fields:title",
    ):
        save_function(
            _authoring_run_log(),
            tmp_path / "store.json",
            enhance=True,
            complete_json=complete_json,
        )


@pytest.mark.parametrize(
    ("stage_to_corrupt", "expected_error"),
    [
        ("parameters", "parameters_stage_changed_function_logic"),
        ("checkers", "checkers_stage_changed_function_logic"),
    ],
)
def test_each_enhancement_stage_has_one_narrow_responsibility(
    tmp_path,
    stage_to_corrupt,
    expected_error,
) -> None:
    def complete_json(prompt: str, _tool: dict) -> str:
        stage = _stage_from_prompt(prompt)
        plan = json.loads(_semantic_plan(stage))
        if stage == stage_to_corrupt:
            plan["functions"][0]["description"] = "Rewritten by the wrong stage."
        return json.dumps(plan)

    with pytest.raises(ValueError, match=expected_error):
        save_function(
            _authoring_run_log(),
            tmp_path / "store.json",
            enhance=True,
            complete_json=complete_json,
        )


def test_checker_stage_cannot_register_another_functions_action(tmp_path) -> None:
    def complete_json(prompt: str, _tool: dict) -> str:
        stage = _stage_from_prompt(prompt)
        plan = json.loads(_semantic_plan(stage))
        wait_function = _function("wait_for_note")
        wait_function["name"] = "Wait for note"
        wait_function["description"] = "Wait for the note page to settle."
        wait_function["steps"] = [
            {
                "step_index": 0,
                "source_state_id": "state-ready",
                "action": {"tool": "wait", "args": {"duration_ms": 1000}},
            }
        ]
        if stage == "checkers":
            wait_function["checker_rules"] = [
                {
                    "source_state_id": "state-checker",
                    "action": {"tool": "click", "args": {"x": 500, "y": 500}},
                }
            ]
        plan["functions"].append(wait_function)
        plan["arguments"]["wait_for_note"] = {}
        return json.dumps(plan)

    with pytest.raises(ValueError, match="checker_not_registered_on_function"):
        save_function(
            _authoring_run_log(),
            tmp_path / "store.json",
            enhance=True,
            complete_json=complete_json,
        )


def test_checker_registration_is_function_local(tmp_path) -> None:
    store_path = tmp_path / "store.json"

    def complete_json(prompt: str, _tool: dict) -> str:
        plan = json.loads(_semantic_plan(_stage_from_prompt(prompt)))
        wait_function = _function("wait_for_note")
        wait_function["name"] = "Wait for note"
        wait_function["description"] = "Wait for the note page to settle."
        wait_function["steps"] = [
            {
                "step_index": 0,
                "source_state_id": "state-ready",
                "action": {"tool": "wait", "args": {"duration_ms": 1000}},
            }
        ]
        plan["functions"].append(wait_function)
        plan["arguments"]["wait_for_note"] = {}
        return json.dumps(plan)

    save_function(
        _authoring_run_log(),
        store_path,
        enhance=True,
        complete_json=complete_json,
    )

    store = FunctionStore(store_path)
    assert len(store.get_function("enter_note").checker_rules) == 1
    assert store.get_function("wait_for_note").checker_rules == ()


def test_tools_expose_one_function_save_interface(tmp_path) -> None:
    bridge = JsonLineBridge(tmp_path / "functions.json")
    definitions = bridge._handle("request-1", "tools/list", {})["tools"]
    tools = {item["name"] for item in definitions}

    assert tools == {
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
    assert "complete Function bundle" in save["description"]
    assert "functions" not in save["inputSchema"].get("required", [])


def test_save_function_requires_functions_without_enhance(tmp_path) -> None:
    bridge = JsonLineBridge(tmp_path / "functions.json")
    result = bridge._save_function(
        "request-1",
        {"run_log": _authoring_run_log()},
    )

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


def test_bridge_enhance_requires_a_complete_bundle_at_every_stage(tmp_path) -> None:
    class Bridge(JsonLineBridge):
        def host_call(self, request_id, method, payload):
            assert method == "model_turn"
            prompt = payload["request"]["messages"][0]["content"]
            stage = _stage_from_prompt(prompt)
            previous_stage = {"parameters": "split", "checkers": "parameters"}.get(
                stage
            )
            previous_bundle = (
                json.loads(_semantic_plan(previous_stage))
                if previous_stage is not None
                else None
            )
            function_schema = payload["request"]["tools"][0]["function"]
            assert function_schema == function_authoring_tool(
                stage=stage,
                current_bundle=previous_bundle,
            )["function"]
            return {
                "tool_calls": [
                    {
                        "function": {
                            "name": "submit_function_bundle",
                                "arguments": _semantic_plan(
                                    _stage_from_prompt(
                                        prompt
                                    )
                                ),
                        }
                    }
                ]
            }

    bridge = Bridge(tmp_path / "functions.json")
    result = bridge._save_function(
        "request-1",
        {"run_log": _authoring_run_log(), "enhance": True},
    )

    assert result["success"] is True
    assert result["function_ids"] == ["enter_note"]


def test_function_authoring_tool_requires_the_complete_function_contract() -> None:
    tool = function_authoring_tool(stage="split", current_bundle=None)
    function = tool["function"]
    parameters = function["parameters"]
    function_schema = parameters["properties"]["functions"]["items"]

    assert function["name"] == "submit_function_bundle"
    assert parameters["required"] == ["functions", "arguments"]
    assert parameters["additionalProperties"] is False
    assert function_schema["required"] == [
        "schema_version",
        "function_id",
        "name",
        "description",
        "input_schema",
        "bindings",
        "steps",
        "checker_rules",
        "agent_visible",
    ]
    assert function_schema["properties"]["steps"]["items"]["required"] == [
        "step_index",
        "source_state_id",
        "action",
    ]
    assert function_schema["properties"]["checker_rules"] == {"const": []}


def test_function_authoring_tool_locks_each_stage_to_its_responsibility() -> None:
    split_tool = function_authoring_tool(stage="split", current_bundle=None)
    split_function = split_tool["function"]["parameters"]["properties"][
        "functions"
    ]["items"]
    assert split_function["properties"]["checker_rules"] == {"const": []}
    assert split_function["properties"]["bindings"] == {"const": []}

    split_bundle = json.loads(_semantic_plan("split"))
    parameter_tool = function_authoring_tool(
        stage="parameters",
        current_bundle=split_bundle,
    )
    parameter_functions = parameter_tool["function"]["parameters"]["properties"][
        "functions"
    ]
    parameter_function = parameter_functions["items"]["oneOf"][0]
    assert parameter_functions["minItems"] == 1
    assert parameter_functions["maxItems"] == 1
    assert parameter_function["properties"]["function_id"] == {
        "const": "enter_note"
    }
    assert parameter_function["properties"]["name"] == {"const": "Enter a note"}
    assert parameter_function["properties"]["checker_rules"] == {"const": []}
    assert parameter_function["properties"]["steps"]["minItems"] == 3
    assert parameter_function["properties"]["steps"]["maxItems"] == 3

    parameter_bundle = json.loads(_semantic_plan("parameters"))
    checker_tool = function_authoring_tool(
        stage="checkers",
        current_bundle=parameter_bundle,
    )
    checker_parameters = checker_tool["function"]["parameters"]
    checker_function = checker_parameters["properties"]["functions"]["items"][
        "oneOf"
    ][0]
    assert checker_parameters["properties"]["arguments"] == {
        "const": parameter_bundle["arguments"]
    }
    assert checker_function["properties"]["checker_rules"]["items"]["enum"] == [
        {
            "source_state_id": step["source_state_id"],
            "action": step["action"],
        }
        for step in parameter_bundle["functions"][0]["steps"]
    ]


def test_save_function_reports_the_failed_agent_stage(tmp_path) -> None:
    def timeout(_prompt: str, _tool: dict) -> str:
        raise TimeoutError("endpoint did not answer")

    with pytest.raises(
        ValueError,
        match=(
            "function_enhancement_split_model_failed:"
            "TimeoutError:endpoint did not answer"
        ),
    ):
        save_function(
            _authoring_run_log(),
            tmp_path / "store.json",
            enhance=True,
            complete_json=timeout,
        )


def test_save_function_rejects_model_commentary_around_the_bundle(
    tmp_path,
) -> None:
    valid = _semantic_plan("split")

    with pytest.raises(ValueError, match="function_enhancement_json_invalid"):
        save_function(
            _authoring_run_log(),
            tmp_path / "store.json",
            enhance=True,
            complete_json=lambda _prompt, _tool: f"Here is the result: {valid}",
        )
