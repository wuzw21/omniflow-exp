from __future__ import annotations

import json
from pathlib import Path

import pytest
from runlog_fixtures import androidworld_run_log, androidworld_state

from omniflow.functions.assets import compile_runlog_to_store


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


def test_compiler_requires_skill_bundle(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="function_bundle_required_from_authoring_skill"):
        compile_runlog_to_store(
            _run_log(1),
            tmp_path / "output",
            source_states={"state_0": {"state_id": "state_0"}},
        )

def test_compiler_freezes_only_function_referenced_states(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    bundle = {
        "schema_version": "omniflow.function-bundle.v2",
        "run_id": "source-run",
        "arguments": {"open_settings_and_wait": {}},
        "functions": [
            {
                "schema_version": "omniflow.function.v2",
                "function_id": "open_settings_and_wait",
                "name": "Open Settings and wait once",
                "description": (
                    "Open the recorded Settings package and wait once for its "
                    "initial page. This does not change any setting or verify a task."
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
                        "source_state_id": "state_0",
                        "action": {
                            "tool": "open_app",
                            "args": {"package_name": "com.android.settings"},
                        },
                    },
                    {
                        "step_index": 1,
                        "source_state_id": "state_1",
                        "action": {"tool": "wait", "args": {"duration_ms": 1000}},
                    },
                ],
                "checker_rules": [],
                "agent_visible": True,
            }
        ],
    }
    result = compile_runlog_to_store(
        _run_log(2),
        output,
        function_bundle=bundle,
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
    assert result["source_arguments"] == {"open_settings_and_wait": {}}
    assert result["model_calls"] == 0
    assert result["prompt_tokens"] == 0
    assert result["completion_tokens"] == 0
    assert result["total_tokens"] == 0


def test_compiler_binds_parameterized_open_app_to_source_evidence(
    tmp_path: Path,
) -> None:
    bundle = {
        "schema_version": "omniflow.function-bundle.v2",
        "run_id": "source-run",
        "arguments": {
            "open_requested_app": {
                "package_name": "com.android.settings",
            }
        },
        "functions": [
            {
                "schema_version": "omniflow.function.v2",
                "function_id": "open_requested_app",
                "name": "Open requested app",
                "description": "Open the requested installed Android app.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "package_name": {
                            "type": "string",
                            "description": "Installed Android package to open",
                        }
                    },
                    "required": ["package_name"],
                    "additionalProperties": False,
                },
                "bindings": [
                    {
                        "source": "$.arguments.package_name",
                        "target": "$.steps[0].action.args.package_name",
                    }
                ],
                "steps": [
                    {
                        "step_index": 0,
                        "source_state_id": "state_0",
                        "action": {
                            "tool": "open_app",
                            "args": {"package_name": ""},
                        },
                    }
                ],
                "checker_rules": [],
                "agent_visible": True,
            }
        ],
    }

    result = compile_runlog_to_store(
        _run_log(1),
        tmp_path / "output",
        function_bundle=bundle,
        source_states={"state_0": {"state_id": "state_0"}},
    )

    assert result["function_ids"] == ["open_requested_app"]
    assert result["source_arguments"] == bundle["arguments"]


def test_compiler_binds_semantic_click_target_to_source_evidence(
    tmp_path: Path,
) -> None:
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
        goal="Select an expense category.",
    )
    bundle = {
        "schema_version": "omniflow.function-bundle.v2",
        "run_id": "source-run",
        "arguments": {"select_category": {"category": "Food"}},
        "functions": [
            {
                "schema_version": "omniflow.function.v2",
                "function_id": "select_category",
                "name": "Select category",
                "description": "Select the requested visible category.",
                "input_schema": {
                    "type": "object",
                    "properties": {"category": {"type": "string"}},
                    "required": ["category"],
                    "additionalProperties": False,
                },
                "bindings": [
                    {
                        "source": "$.arguments.category",
                        "target": "$.steps[0].action.args.target_description",
                    }
                ],
                "steps": [
                    {
                        "step_index": 0,
                        "source_state_id": "category-state",
                        "action": {
                            "tool": "click",
                            "args": {
                                "target_description": "",
                                "x": 148.61111111111111,
                                "y": 312.5,
                            },
                        },
                    }
                ],
                "checker_rules": [],
                "agent_visible": True,
            }
        ],
    }

    result = compile_runlog_to_store(
        run_log,
        tmp_path / "output",
        function_bundle=bundle,
        source_states={
            "category-state": {
                "state_id": "category-state",
                "xml": run_log["steps"][0]["observation"]["forest"],
                "display": {"width": 720, "height": 1280},
            }
        },
    )

    assert result["function_ids"] == ["select_category"]


def test_compiler_rejects_semantic_click_target_without_source_evidence(
    tmp_path: Path,
) -> None:
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
        goal="Select an expense category.",
    )
    bundle = {
        "schema_version": "omniflow.function-bundle.v2",
        "run_id": "source-run",
        "arguments": {"select_category": {"category": "Income"}},
        "functions": [
            {
                "schema_version": "omniflow.function.v2",
                "function_id": "select_category",
                "name": "Select category",
                "description": "Select the requested visible category.",
                "input_schema": {
                    "type": "object",
                    "properties": {"category": {"type": "string"}},
                    "required": ["category"],
                    "additionalProperties": False,
                },
                "bindings": [
                    {
                        "source": "$.arguments.category",
                        "target": "$.steps[0].action.args.target_description",
                    }
                ],
                "steps": [
                    {
                        "step_index": 0,
                        "source_state_id": "category-state",
                        "action": {
                            "tool": "click",
                            "args": {
                                "target_description": "",
                                "x": 148.61111111111111,
                                "y": 312.5,
                            },
                        },
                    }
                ],
                "checker_rules": [],
                "agent_visible": True,
            }
        ],
    }

    with pytest.raises(
        ValueError,
        match="function_action_not_grounded:select_category:0",
    ):
        compile_runlog_to_store(
            run_log,
            tmp_path / "output",
            function_bundle=bundle,
            source_states={
                "category-state": {
                    "state_id": "category-state",
                    "xml": run_log["steps"][0]["observation"]["forest"],
                    "display": {"width": 720, "height": 1280},
                }
            },
        )


def _single_input_bundle(parameter_name: str, *, text: str = "Paid by card") -> dict:
    return {
        "schema_version": "omniflow.function-bundle.v2",
        "run_id": "source-run",
        "arguments": {"enter_expense_field": {parameter_name: text}},
        "functions": [
            {
                "schema_version": "omniflow.function.v2",
                "function_id": "enter_expense_field",
                "name": "Enter expense field",
                "description": "Enter a value in the expense form.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        parameter_name: {
                            "type": "string",
                            "description": f"Expense {parameter_name.replace('_', ' ')}",
                        }
                    },
                    "required": [parameter_name],
                    "additionalProperties": False,
                },
                "bindings": [
                    {
                        "source": f"$.arguments.{parameter_name}",
                        "target": "$.steps[0].action.args.text",
                    }
                ],
                "steps": [
                    {
                        "step_index": 0,
                        "source_state_id": "note-state",
                        "action": {"tool": "input_text", "args": {"text": ""}},
                    }
                ],
                "checker_rules": [],
                "agent_visible": True,
            }
        ],
    }


def _note_input_run_log() -> dict:
    return androidworld_run_log(
        [{"action_type": "input_text", "text": "Paid by card"}],
        observations=[
            androidworld_state(
                "note-state",
                forest=(
                    '<hierarchy><node text="Note" resource-id="app:id/note" '
                    'class="android.widget.EditText" editable="true" '
                    'focused="true" bounds="[0,0][100,100]" /></hierarchy>'
                ),
            )
        ],
        goal="Enter an expense note.",
    )


def test_compiler_rejects_agent_action_not_grounded_in_runlog(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="function_action_not_grounded:enter_expense_field:0",
    ):
        compile_runlog_to_store(
            _note_input_run_log(),
            tmp_path / "output",
            function_bundle=_single_input_bundle("note", text="invented"),
            source_states={
                "note-state": {
                    "state_id": "note-state",
                    "xml": (
                        '<hierarchy><node text="Note" resource-id="app:id/note" '
                        'class="android.widget.EditText" editable="true" '
                        'focused="true" bounds="[0,0][100,100]" /></hierarchy>'
                    ),
                }
            },
        )


def test_compiler_preserves_multiple_semantic_functions_and_source_calls(
    tmp_path: Path,
) -> None:
    run_log = androidworld_run_log(
        [
            {"action_type": "input_text", "text": "Theater Show"},
            {"action_type": "input_text", "text": "Museum Tickets"},
            {"action_type": "input_text", "text": "Household Items"},
        ],
        observations=[
            androidworld_state("category-six"),
            androidworld_state("category-six-after-first-expense"),
            androidworld_state("category-one"),
        ],
        goal="Add three expenses.",
    )
    bundle = {
        "schema_version": "omniflow.function-bundle.v2",
        "run_id": "source-run",
        "arguments": {
            "add_category_six_expense": [
                {"name": "Theater Show"},
                {"name": "Museum Tickets"},
            ],
            "add_category_one_expense": [
                {"name": "Household Items"},
            ],
        },
        "functions": [
            _single_input_bundle("name", text="Theater Show")["functions"][0]
            | {
                "function_id": "add_category_six_expense",
                "name": "Add one category-six expense",
                "description": "Add one expense using the category-six path.",
                "steps": [
                    {
                        "step_index": 0,
                        "source_state_id": "category-six",
                        "action": {"tool": "input_text", "args": {"text": ""}},
                    }
                ],
            },
            _single_input_bundle("name", text="Household Items")["functions"][0]
            | {
                "function_id": "add_category_one_expense",
                "name": "Add one category-one expense",
                "description": "Add one expense using the category-one path.",
                "steps": [
                    {
                        "step_index": 0,
                        "source_state_id": "category-one",
                        "action": {"tool": "input_text", "args": {"text": ""}},
                    }
                ],
            },
        ],
    }

    result = compile_runlog_to_store(
        run_log,
        tmp_path / "output",
        function_bundle=bundle,
        source_states={
            "category-six": {"state_id": "category-six"},
            "category-one": {"state_id": "category-one"},
        },
    )

    assert result["function_ids"] == [
        "add_category_six_expense",
        "add_category_one_expense",
    ]
    assert result["source_arguments"] == bundle["arguments"]
