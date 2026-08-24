from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from runlog_fixtures import androidworld_run_log, androidworld_state

from omniflow.functions.artifact import parse_function_artifact
from omniflow.functions.compiler import (
    _atomicize_repeated_click_function,
    compile_runlog_to_store,
)
from omniflow.runlog import import_run_log_evidence


def _run_log(step_count: int) -> dict:
    actions = [
        (
            {"action_type": "open_app", "app_name": "com.android.settings"}
            if index == 0
            else {"action_type": "wait"}
        )
        for index in range(step_count)
    ]
    return androidworld_run_log(
        actions,
        observations=[
            androidworld_state(f"state_{index}")
            for index in range(step_count)
        ],
        goal="Open Settings and wait.",
    )


def test_default_compiler_registers_one_action_complete_function(
    tmp_path: Path,
) -> None:
    result = compile_runlog_to_store(
        _run_log(1),
        tmp_path / "output",
        source_states={"state_0": {"state_id": "state_0"}},
    )

    assert result["function_count"] == 1
    assert result["function_ids"][0].startswith("complete_recorded_")


def test_compiler_promotes_launcher_app_click_to_global_open_app(
    tmp_path: Path,
) -> None:
    launcher = androidworld_state(
        "launcher",
        package_name="com.google.android.apps.nexuslauncher",
        forest=(
            '<hierarchy><node package="com.google.android.apps.nexuslauncher" '
            'resource-id="com.google.android.apps.nexuslauncher:id/icon" '
            'text="Camera" clickable="true" /></hierarchy>'
        ),
    )
    camera = androidworld_state(
        "camera",
        package_name="com.android.camera2",
        forest='<hierarchy><node package="com.android.camera2" /></hierarchy>',
    )
    payload = androidworld_run_log(
        [{"action_type": "click", "x": 624, "y": 560}],
        observations=[launcher],
        goal="Take one video.",
    )
    payload["steps"][0]["next_observation"] = camera

    result = compile_runlog_to_store(
        payload,
        tmp_path / "output",
        source_states={"launcher": launcher},
    )

    store = json.loads(Path(result["store_path"]).read_text())
    function = next(iter(store["functions"].values()))
    assert function["steps"][0]["action"] == {
        "tool": "open_app",
        "args": {"package_name": "com.android.camera2"},
    }


def test_compiler_marks_answer_as_planner_handoff(tmp_path: Path) -> None:
    payload = androidworld_run_log(
        [
            {"action_type": "open_app", "app_name": "com.example.calendar"},
            {"action_type": "answer", "text": "Meeting"},
        ],
        observations=[
            androidworld_state("state_0"),
            androidworld_state("state_1"),
        ],
        goal="Find the event and answer with its title.",
    )

    result = compile_runlog_to_store(
        payload,
        tmp_path / "output",
        source_states={"state_0": {"state_id": "state_0"}},
    )

    store = json.loads(Path(result["store_path"]).read_text())
    function = next(iter(store["functions"].values()))
    assert "Planner must inspect it and provide the task answer" in function[
        "description"
    ]


def test_compiler_restores_omitted_essential_complete_action(
    tmp_path: Path,
) -> None:
    payload = androidworld_run_log(
        [
            {"action_type": "open_app", "app_name": "com.example.settings"},
            {"action_type": "click", "x": 300, "y": 400},
            {"action_type": "navigate_back"},
        ],
        observations=[
            androidworld_state("state_0"),
            androidworld_state("state_1"),
            androidworld_state("state_2"),
        ],
        goal="Open settings, change the setting, and return.",
    )
    _, source_states = import_run_log_evidence(payload)
    proposal = {
        "reason": "Keep the startup and return actions.",
        "plan": {
            "functions": [],
            "complete_function": {
                "function_id": "complete_settings",
                "name": "Complete settings",
                "description": "Open settings and return.",
                "source_step_indices": [0, 2],
                "parameters": [],
            },
        },
    }

    class Completions:
        def create(self, **_kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps(proposal))
                    )
                ],
                usage=None,
            )

    result = compile_runlog_to_store(
        payload,
        tmp_path / "output",
        source_states=source_states,
        model="test-model",
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
    )
    store = json.loads(Path(result["store_path"]).read_text())
    function = store["functions"]["complete_settings"]
    assert [step["action"]["tool"] for step in function["steps"]] == [
        "open_app",
        "click",
        "press_key",
    ]
    assert "restored omitted executable source steps" in result["reason"]


def test_compiler_freezes_only_function_referenced_states(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    result = compile_runlog_to_store(
        _run_log(2),
        output,
        source_states={
            "state_0": {"state_id": "state_0"},
            "state_1": {"state_id": "state_1"},
            "unused": {"state_id": "unused"},
        },
    )

    catalog = json.loads(
        Path(result["transfer_state_catalog"]).read_text(encoding="utf-8")
    )
    assert set(catalog["states"]) == {"state_0", "state_1"}
    assert result["transfer_state_count"] == 2
    assert result["source_arguments"] == {result["function_ids"][0]: {}}
    assert result["source_calls"] == [
        {"function_id": result["function_ids"][0], "arguments": {}}
    ]
    assert result["model_calls"] == 0
    assert result["prompt_tokens"] == 0
    assert result["completion_tokens"] == 0
    assert result["total_tokens"] == 0


def test_compiler_hands_off_before_observation_dependent_input(tmp_path: Path) -> None:
    payload = androidworld_run_log(
        [
            {"action_type": "input_text", "text": "3125"},
            {"action_type": "wait"},
        ],
        observations=[
            {
                "pixels": None,
                "forest": (
                    '<hierarchy><node class="android.widget.EditText" '
                    'text="Enter the product" resource-id="answer" '
                    'bounds="[216,278][615,331]" editable="true" '
                    'focused="true" /></hierarchy>'
                ),
                "ui_elements": [],
                "auxiliaries": {
                    "state_id": "product-form",
                    "display": {"width": 720, "height": 1280},
                },
            },
            androidworld_state("submitted", width=720, height=1280),
        ],
        goal="Enter the product and submit it.",
    )
    _, source_states = import_run_log_evidence(payload)

    result = compile_runlog_to_store(
        payload,
        tmp_path / "output",
        source_states=source_states,
    )

    store = json.loads(Path(result["store_path"]).read_text(encoding="utf-8"))
    function = next(iter(store["functions"].values()))
    assert [step["action"]["tool"] for step in function["steps"]] == ["wait"]
    assert function["input_schema"] == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    assert "observation-dependent handoff" in result["reason"]


def test_compiler_restores_omitted_post_input_commit(tmp_path: Path) -> None:
    form_state = {
        "pixels": None,
        "forest": (
            '<hierarchy><node class="android.widget.EditText" '
            'text="Enter the product" resource-id="answer" '
            'bounds="[216,278][615,331]" editable="true" '
            'focused="true" /></hierarchy>'
        ),
        "ui_elements": [],
        "auxiliaries": {
            "state_id": "product-form",
            "display": {"width": 720, "height": 1280},
        },
    }
    payload = androidworld_run_log(
        [
            {"action_type": "input_text", "text": "3125"},
            {"action_type": "click", "x": 600, "y": 900},
        ],
        observations=[
            form_state,
            androidworld_state("product-entered", width=720, height=1280),
        ],
        goal="Enter the product and submit it.",
    )
    _, source_states = import_run_log_evidence(payload)

    result = compile_runlog_to_store(
        payload,
        tmp_path / "output",
        function_bundle={
            "schema_version": "omniflow.function-bundle.v2",
            "run_id": "source-run",
            "checker_rules": [],
            "arguments": {"enter_product": {}},
            "functions": [
                {
                    "schema_version": "omniflow.function.v2",
                    "function_id": "enter_product",
                    "name": "Enter product",
                    "description": "Enter and submit the product.",
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
                            "source_state_id": "product-form",
                            "action": {
                                "tool": "click",
                                "args": {"x": 577.083333, "y": 237.890625},
                            },
                        },
                        {
                            "step_index": 1,
                            "source_state_id": "product-form",
                            "action": {
                                "tool": "input_text",
                                "args": {
                                    "target_description": "Enter the product",
                                    "text": "3125",
                                },
                            },
                        },
                    ],
                    "agent_visible": True,
                }
            ],
        },
        source_states=source_states,
    )

    store = json.loads(Path(result["store_path"]).read_text(encoding="utf-8"))
    function = store["functions"]["enter_product"]
    assert [step["action"]["tool"] for step in function["steps"]] == [
        "click",
        "input_text",
        "click",
    ]
    assert function["steps"][2]["source_state_id"] == "product-entered"
    assert "restored 1 successful post-input commit action" in result["reason"]


def test_function_rejects_ungrounded_input_text() -> None:
    with pytest.raises(
        ValueError,
        match="function_input_target_description_required:0",
    ):
        parse_function_artifact(
            {
                "schema_version": "omniflow.function.v2",
                "function_id": "enter_product",
                "name": "Enter product",
                "description": "Enter the computed product.",
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
                        "source_state_id": "product-form",
                        "action": {
                            "tool": "input_text",
                            "args": {"text": "3125"},
                        },
                    }
                ],
                "agent_visible": True,
            }
        )


def test_function_rejects_coordinate_parameter_bindings() -> None:
    with pytest.raises(
        ValueError,
        match="function_binding_target_non_parameterizable",
    ):
        parse_function_artifact(
            {
                "schema_version": "omniflow.function.v2",
                "function_id": "click_button",
                "name": "Click button",
                "description": "Click the visible button once.",
                "input_schema": {
                    "type": "object",
                    "properties": {"x": {"type": "number"}},
                    "required": ["x"],
                    "additionalProperties": False,
                },
                "bindings": [
                    {
                        "source": "$.arguments.x",
                        "target": "$.steps[0].action.args.x",
                    }
                ],
                "steps": [
                    {
                        "step_index": 0,
                        "source_state_id": "button-page",
                        "action": {
                            "tool": "click",
                            "args": {"x": 0, "y": 250},
                        },
                    }
                ],
                "agent_visible": True,
            }
        )


def test_compiler_registers_source_call_for_argumentless_authored_function(
    tmp_path: Path,
) -> None:
    result = compile_runlog_to_store(
        _run_log(2),
        tmp_path / "output",
        function_bundle={
            "schema_version": "omniflow.function-bundle.v2",
            "run_id": "source-run",
            "checker_rules": [],
            "arguments": {},
            "functions": [
                {
                    "schema_version": "omniflow.function.v2",
                    "function_id": "open_settings",
                    "name": "Open Settings",
                    "description": "Open Settings and wait for the page.",
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
                            "source_state_id": "state_0",
                            "action": {
                                "tool": "open_app",
                                "args": {
                                    "package_name": "com.android.settings"
                                },
                            },
                        },
                        {
                            "step_index": 1,
                            "source_state_id": "state_1",
                            "action": {
                                "tool": "wait",
                                "args": {"duration_ms": 1000},
                            },
                        },
                    ],
                    "agent_visible": True,
                }
            ],
        },
        source_states={
            "state_0": {"state_id": "state_0"},
            "state_1": {"state_id": "state_1"},
        },
    )

    assert result["source_arguments"] == {"open_settings": {}}
    assert result["source_calls"] == [
        {"function_id": "open_settings", "arguments": {}}
    ]


def test_authoring_prompt_forbids_hiding_observation_dependent_repeats(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    proposal = {
        "reason": "Keep the recorded navigation and wait together.",
        "plan": {
            "functions": [],
            "complete_function": {
                "function_id": "open_settings",
                "name": "Open Settings",
                "description": "Open Settings and wait for the page.",
                "source_step_indices": [0, 1],
                "parameters": [],
            },
        },
    }

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps(proposal))
                    )
                ],
                usage=None,
            )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )
    compile_runlog_to_store(
        _run_log(2),
        tmp_path / "output",
        source_states={
            "state_0": {"state_id": "state_0"},
            "state_1": {"state_id": "state_1"},
        },
        model="test-model",
        client=client,
    )

    system_prompt = captured["messages"][0]["content"]
    assert "encode repetition count" in system_prompt
    assert "call it repeatedly" in system_prompt
    assert "complete_function must start with" in system_prompt
    assert (
        "the first open_app and end with the terminal successful task action"
        in system_prompt
    )
    assert "never merely hard-code\nthe successful instance values" in system_prompt
    assert "Do not invent a nesting or parent/child schema" in system_prompt
    assert "Do not output input_schema, bindings, steps, actions" in system_prompt
    assert captured["max_tokens"] == 512
    assert captured["stream"] is False
    assert captured["reasoning_effort"] == "none"
    assert captured["extra_body"] == {
        "enable_thinking": False,
        "thinking": {"type": "disabled"},
    }
    request = json.loads(captured["messages"][1]["content"])
    facts = request["source_run"]
    assert facts["schema_version"] == "omniflow.function-compilation-facts.v2"
    assert [step["source_step_index"] for step in facts["steps"]] == [0, 1]
    assert all("step_index" not in step for step in facts["steps"])
    assert request["parameter_candidates"] == [
        {
            "source_step_index": 0,
            "tool": "open_app",
            "arg_name": "package_name",
            "recorded_value": "com.android.settings",
        }
    ]

    store = json.loads((tmp_path / "output" / "store.json").read_text())
    function = store["functions"]["open_settings"]
    assert [step["step_index"] for step in function["steps"]] == [0, 1]
    assert function["input_schema"] == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }


def test_model_plan_exposes_global_open_app_as_function_input(
    tmp_path: Path,
) -> None:
    payload = _run_log(1)
    _, source_states = import_run_log_evidence(payload)
    proposal = {
        "reason": "Launch the requested app through the global startup Function.",
        "plan": {
            "functions": [],
            "complete_function": {
                "function_id": "open_requested_app",
                "name": "Open requested app",
                "description": "Launch the app requested by the task.",
                "source_step_indices": [0],
                "parameters": [
                    {
                        "name": "package_name",
                        "description": "Package of the app to launch",
                        "source_step_index": 0,
                        "arg_name": "package_name",
                    }
                ],
            },
        },
    }

    class Completions:
        def create(self, **_kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps(proposal))
                    )
                ],
                usage=None,
            )

    result = compile_runlog_to_store(
        payload,
        tmp_path / "output",
        source_states=source_states,
        model="test-model",
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
    )

    store = json.loads(Path(result["store_path"]).read_text())
    function = store["functions"]["open_requested_app"]
    assert function["input_schema"] == {
        "type": "object",
        "properties": {
            "package_name": {
                "type": "string",
                "description": "Package of the app to launch",
            }
        },
        "required": ["package_name"],
        "additionalProperties": False,
    }
    assert function["bindings"] == [
        {
            "source": "$.arguments.package_name",
            "target": "$.steps[0].action.args.package_name",
        }
    ]
    assert function["steps"][0]["action"]["args"]["package_name"] == ""
    assert result["source_arguments"] == {
        "open_requested_app": {"package_name": "com.android.settings"}
    }


def test_model_plan_materializes_schema_binding_and_source_arguments(
    tmp_path: Path,
) -> None:
    form_state = {
        "pixels": None,
        "forest": (
            '<hierarchy><node class="android.widget.EditText" '
            'text="Enter the product" bounds="[100,100][600,200]" '
            'editable="true" focused="true" /></hierarchy>'
        ),
        "ui_elements": [],
        "auxiliaries": {
            "state_id": "product-form",
            "display": {"width": 720, "height": 1280},
        },
    }
    payload = androidworld_run_log(
        [
            {"action_type": "input_text", "text": "3125"},
            {"action_type": "click", "x": 600, "y": 900},
        ],
        observations=[
            form_state,
            androidworld_state("product-entered", width=720, height=1280),
        ],
        goal="Enter the product and submit it.",
    )
    _, source_states = import_run_log_evidence(payload)
    proposal = {
        "reason": "Keep input and submit together; parameterize the product.",
        "plan": {
            "functions": [
                {
                    "function_id": "enter_product",
                    "name": "Enter product",
                    "description": "Enter the requested product.",
                    "source_step_indices": [0],
                    "parameters": [
                        {
                            "name": "product",
                            "description": "Computed product to enter",
                            "source_step_index": 0,
                            "arg_name": "text",
                        }
                    ],
                }
            ],
            "complete_function": {
                "function_id": "complete_product_form",
                "name": "Enter and submit product",
                "description": "Enter the requested product and submit the form.",
                "source_step_indices": [0, 1],
                "parameters": [
                    {
                        "name": "product",
                        "description": "Computed product to enter",
                        "source_step_index": 0,
                        "arg_name": "text",
                    }
                ],
            },
        },
    }

    class Completions:
        def create(self, **_kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps(proposal))
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=100,
                    completion_tokens=40,
                    total_tokens=140,
                ),
            )

    result = compile_runlog_to_store(
        payload,
        tmp_path / "output",
        source_states=source_states,
        model="test-model",
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
    )

    store = json.loads(Path(result["store_path"]).read_text())
    assert len(store["functions"]) == 1
    function = next(iter(store["functions"].values()))
    assert [step["action"]["tool"] for step in function["steps"]] == ["click"]
    assert function["input_schema"] == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    assert function["bindings"] == []
    assert result["source_arguments"] == {function["function_id"]: {}}
    assert result["total_tokens"] == 140


def test_model_plan_copies_semantic_parameters_to_complete_function(
    tmp_path: Path,
) -> None:
    form_state = {
        "pixels": None,
        "forest": (
            '<hierarchy><node class="android.widget.EditText" '
            'text="Enter the product" bounds="[100,100][600,200]" '
            'editable="true" focused="true" /></hierarchy>'
        ),
        "ui_elements": [],
        "auxiliaries": {
            "state_id": "product-form",
            "display": {"width": 720, "height": 1280},
        },
    }
    payload = androidworld_run_log(
        [
            {"action_type": "input_text", "text": "3125"},
            {"action_type": "click", "x": 600, "y": 900},
        ],
        observations=[
            form_state,
            androidworld_state("product-entered", width=720, height=1280),
        ],
        goal="Enter product 3125 and submit it.",
    )
    _, source_states = import_run_log_evidence(payload)
    proposal = {
        "reason": "Create a reusable input Function and complete envelope.",
        "plan": {
            "functions": [
                {
                    "function_id": "enter_product",
                    "name": "Enter product",
                    "description": "Enter the requested product.",
                    "source_step_indices": [0],
                    "parameters": [
                        {
                            "name": "product",
                            "description": "Product to enter",
                            "source_step_index": 0,
                            "arg_name": "text",
                        }
                    ],
                }
            ],
            "complete_function": {
                "function_id": "complete_product_form",
                "name": "Complete product form",
                "description": "Enter the product and submit the form.",
                "source_step_indices": [0, 1],
                "parameters": [],
            },
        },
    }

    class Completions:
        def create(self, **_kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps(proposal))
                    )
                ],
                usage=None,
            )

    result = compile_runlog_to_store(
        payload,
        tmp_path / "output",
        source_states=source_states,
        model="test-model",
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
    )

    store = json.loads(Path(result["store_path"]).read_text())
    complete = store["functions"]["complete_product_form"]
    assert complete["input_schema"]["required"] == ["product"]
    assert complete["bindings"] == [
        {
            "source": "$.arguments.product",
            "target": "$.steps[0].action.args.text",
        }
    ]
    assert complete["steps"][0]["action"]["args"]["text"] == ""
    assert "copied a validated semantic parameter" in result["reason"]


def test_model_plan_atomicizes_observation_dependent_repeated_clicks(
    tmp_path: Path,
) -> None:
    payload = androidworld_run_log(
        [{"action_type": "click", "x": 500, "y": 500} for _ in range(5)],
        observations=[androidworld_state(f"number-{index}") for index in range(5)],
        goal="Click five times, read each number, and multiply them.",
    )
    _, source_states = import_run_log_evidence(payload)
    proposal = {
        "reason": "Group source steps 0-4 as five clicks.",
        "plan": {
            "functions": [
                {
                    "function_id": "click_button_5_times",
                    "name": "Click the button 5 times",
                    "description": "Click the button 5 times to display numbers.",
                    "source_step_indices": [0, 1, 2, 3, 4],
                    "parameters": [],
                }
            ],
            "complete_function": {
                "function_id": "complete_multiply_workflow",
                "name": "Complete multiplication workflow",
                "description": "Click five times and compute the product.",
                "source_step_indices": [0, 1, 2, 3, 4],
                "parameters": [],
            },
        },
    }

    class Completions:
        def create(self, **_kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps(proposal))
                    )
                ],
                usage=None,
            )

    result = compile_runlog_to_store(
        payload,
        tmp_path / "output",
        source_states=source_states,
        model="test-model",
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
    )

    store = json.loads(Path(result["store_path"]).read_text())
    assert result["function_count"] == 2
    assert result["function_ids"][0] == "click_button"
    function = store["functions"]["click_button"]
    assert function["name"] == "Click the button"
    assert function["description"] == "Click the button to display numbers."
    assert len(function["steps"]) == 1
    assert function["steps"][0]["step_index"] == 0
    complete_id = result["function_ids"][1]
    assert complete_id == "complete_multiply_workflow"
    assert len(store["functions"][complete_id]["steps"]) == 1
    assert "Planner observes after every click" in result["reason"]


def test_global_function_preserves_repeat_boundary_for_runtime_handoff(
    tmp_path: Path,
) -> None:
    actions = [
        {"action_type": "open_app", "app_name": "com.android.chrome"},
        *[
            {"action_type": "click", "x": 500, "y": 500}
            for _ in range(3)
        ],
        {"action_type": "wait"},
    ]
    payload = androidworld_run_log(
        actions,
        observations=[
            androidworld_state(f"state-{index}")
            for index in range(len(actions))
        ],
        goal="Open the task, click three times, and finish.",
    )
    _, source_states = import_run_log_evidence(payload)
    proposal = {
        "reason": "Keep an atomic click and the complete task envelope.",
        "plan": {
            "functions": [
                {
                    "function_id": "click_button_3_times",
                    "name": "Click button 3 times",
                    "description": "Click the button three times.",
                    "source_step_indices": [1, 2, 3],
                    "parameters": [],
                }
            ],
            "complete_function": {
                "function_id": "complete_task",
                "name": "Complete task",
                "description": "Open the app and complete the whole task.",
                "source_step_indices": [0, 1, 2, 3, 4],
                "parameters": [],
            },
        },
    }

    class Completions:
        def create(self, **_kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps(proposal))
                    )
                ],
                usage=None,
            )

    result = compile_runlog_to_store(
        payload,
        tmp_path / "output",
        source_states=source_states,
        model="test-model",
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
    )

    store = json.loads(Path(result["store_path"]).read_text())["functions"]
    assert len(store["click_button"]["steps"]) == 1
    assert [step["action"]["tool"] for step in store["complete_task"]["steps"]] == [
        "open_app",
        "click",
        "click",
        "click",
        "wait",
    ]


def test_model_plan_falls_back_when_global_function_drops_terminal_action(
    tmp_path: Path,
) -> None:
    proposal = {
        "reason": "Keep opening and waiting as separate reusable actions.",
        "plan": {
            "functions": [
                {
                    "function_id": "open_settings",
                    "name": "Open Settings",
                    "description": "Open the Settings app.",
                    "source_step_indices": [0],
                    "parameters": [],
                },
                {
                    "function_id": "wait_for_settings",
                    "name": "Wait for Settings",
                    "description": "Wait for the Settings page.",
                    "source_step_indices": [1],
                    "parameters": [],
                },
            ],
            "complete_function": {
                "function_id": "complete_settings_workflow",
                "name": "Open Settings",
                "description": "Open Settings as one safe reusable action.",
                "source_step_indices": [0],
                "parameters": [],
            },
        },
    }

    class Completions:
        def create(self, **_kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps(proposal))
                    )
                ],
                usage=None,
            )

    result = compile_runlog_to_store(
        _run_log(2),
        tmp_path / "output",
        source_states={
            "state_0": {"state_id": "state_0"},
            "state_1": {"state_id": "state_1"},
        },
        model="test-model",
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
    )

    assert result["success"] is True
    assert result["function_count"] == 3
    assert "complete_settings_workflow" in result["function_ids"]


def test_model_plan_allows_global_function_to_omit_unsafe_middle_actions(
    tmp_path: Path,
) -> None:
    proposal = {
        "reason": "Keep the stable task envelope and omit the unsafe middle wait.",
        "plan": {
            "functions": [],
            "complete_function": {
                "function_id": "complete_settings_workflow",
                "name": "Complete Settings workflow",
                "description": "Open Settings and finish the recorded workflow.",
                "source_step_indices": [0, 2],
                "parameters": [],
            },
        },
    }

    class Completions:
        def create(self, **_kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps(proposal))
                    )
                ],
                usage=None,
            )

    result = compile_runlog_to_store(
        _run_log(3),
        tmp_path / "output",
        source_states={
            f"state_{index}": {"state_id": f"state_{index}"}
            for index in range(3)
        },
        model="test-model",
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
    )

    function = json.loads(Path(result["store_path"]).read_text())["functions"][
        "complete_settings_workflow"
    ]
    assert [step["action"]["tool"] for step in function["steps"]] == [
        "open_app",
        "wait",
    ]


def test_same_state_click_retry_is_not_treated_as_observation_output() -> None:
    click = {"tool": "click", "args": {"x": 500, "y": 500}}
    source_steps = [
        {
            "before_state_id": "launcher",
            "action": {
                "tool": "open_app",
                "args": {"package_name": "com.android.documentsui"},
            },
        },
        {"before_state_id": "onboarding", "action": click},
        {"before_state_id": "onboarding", "action": click},
    ]

    result = _atomicize_repeated_click_function(
        [0, 1, 2],
        source_steps,
        function_id="open_file",
        name="Open file",
        description="Open the file through onboarding.",
    )

    assert result == (
        [0, 1, 2],
        "open_file",
        "Open file",
        "Open the file through onboarding.",
        0,
    )


def test_invalid_model_plan_preserves_evidence_and_uses_complete_fallback(
    tmp_path: Path,
) -> None:
    invalid = {"reason": "Old full bundle shape.", "bundle": {}}

    class Completions:
        def create(self, **_kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps(invalid))
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=12,
                    completion_tokens=8,
                    total_tokens=20,
                ),
            )

    output = tmp_path / "rejected"
    result = compile_runlog_to_store(
        _run_log(2),
        output,
        source_states={
            "state_0": {"state_id": "state_0"},
            "state_1": {"state_id": "state_1"},
        },
        model="test-model",
        client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
    )

    failure = json.loads((output / "authoring_failure.json").read_text())
    assert failure["classification"] == "authoring_rejected"
    assert failure["total_tokens"] == 20
    assert json.loads(failure["raw_response"]) == invalid
    assert result["success"] is True
    assert result["total_tokens"] == 20
    assert result["function_count"] == 1
    assert "complete schema-valid recorded Function" in result["reason"]
