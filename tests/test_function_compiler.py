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


def test_compiler_preserves_focused_input_source_semantics(tmp_path: Path) -> None:
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
    assert function["steps"][0]["action"] == {
        "tool": "input_text",
        "args": {
            "target_description": "Enter the product",
            "text": "3125",
        },
    }


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
    assert "compiler assigns every\nsource step exactly once" in system_prompt
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
    assert request["parameter_candidates"] == []

    store = json.loads((tmp_path / "output" / "store.json").read_text())
    function = store["functions"]["open_settings"]
    assert [step["step_index"] for step in function["steps"]] == [0, 1]
    assert function["input_schema"] == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
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
    function = store["functions"]["enter_product"]
    assert function["input_schema"] == {
        "type": "object",
        "properties": {
            "product": {
                "type": "string",
                "description": "Computed product to enter",
            }
        },
        "required": ["product"],
        "additionalProperties": False,
    }
    assert function["bindings"] == [
        {
            "source": "$.arguments.product",
            "target": "$.steps[0].action.args.text",
        }
    ]
    assert function["steps"][0]["action"]["args"]["text"] == ""
    complete = store["functions"]["complete_product_form"]
    assert complete["input_schema"] == function["input_schema"]
    assert complete["bindings"] == function["bindings"]
    assert [step["action"]["tool"] for step in complete["steps"]] == [
        "input_text",
        "click",
    ]
    assert result["source_arguments"] == {
        "enter_product": {"product": "3125"},
        "complete_product_form": {"product": "3125"},
    }
    assert result["total_tokens"] == 140


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
    assert len(store["functions"][complete_id]["steps"]) == 5
    assert "Planner observes after every click" in result["reason"]


def test_model_plan_keeps_fragments_and_agent_authored_complete_function(
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
                "name": "Open Settings and wait",
                "description": "Open Settings and wait for the page.",
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

    assert result["function_count"] == 3
    assert result["function_ids"][:2] == ["open_settings", "wait_for_settings"]
    complete_id = result["function_ids"][2]
    assert complete_id == "complete_settings_workflow"
    store = json.loads(Path(result["store_path"]).read_text())
    assert [
        step["action"]["tool"]
        for step in store["functions"][complete_id]["steps"]
    ] == ["open_app", "wait"]


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


def test_invalid_model_plan_preserves_failure_response_and_usage(
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
    with pytest.raises(
        ValueError,
        match="function_author_plan_response_contract_invalid",
    ):
        compile_runlog_to_store(
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
