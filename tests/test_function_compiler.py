from __future__ import annotations

import json
from pathlib import Path

import pytest
from runlog_fixtures import androidworld_run_log, androidworld_state

from omniflow.functions.assets import FunctionStore, save_function


def _save(run_log: dict, output: Path, *, function_bundle=None, **kwargs):
    bundle = function_bundle or {"functions": [], "arguments": {}}
    kwargs.pop("source_states", None)
    return save_function(
        run_log,
        output / "store.json",
        functions=bundle["functions"],
        arguments=bundle["arguments"],
        **kwargs,
    )


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
    with pytest.raises(ValueError, match="functions_required"):
        _save(
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
    result = _save(
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


def test_save_function_optionally_derives_checker_rules_with_agent(tmp_path: Path) -> None:
    run_log = androidworld_run_log(
        [
            {"action_type": "click", "x": 500, "y": 500},
            {"action_type": "wait"},
        ],
        observations=[androidworld_state("prompt"), androidworld_state("ready")],
        goal="Dismiss the prompt and continue.",
    )
    function = {
        "schema_version": "omniflow.function.v2",
        "function_id": "continue_after_prompt",
        "name": "Continue after an optional prompt",
        "description": "Dismiss the optional prompt when present, then continue.",
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
                "source_state_id": "prompt",
                "action": {"tool": "click", "args": {"x": 500, "y": 500}},
            },
            {
                "step_index": 1,
                "source_state_id": "ready",
                "action": {"tool": "wait", "args": {"duration_ms": 1000}},
            },
        ],
        "checker_rules": [],
        "agent_visible": True,
    }
    store_path = tmp_path / "store.json"
    authored = {
        **function,
        "steps": [
            {
                "step_index": 0,
                "source_state_id": "ready",
                "action": {"tool": "wait", "args": {"duration_ms": 1000}},
            }
        ],
        "checker_rules": [
            {
                "source_state_id": "prompt",
                "action": {"tool": "click", "args": {"x": 500, "y": 500}},
            }
        ],
    }

    result = save_function(
        run_log,
        store_path,
        functions=[function],
        enhance=True,
        complete_json=lambda prompt, _tool: json.dumps(
            {
                "functions": [
                    authored if "stage checkers" in prompt else function
                ],
                "arguments": {"continue_after_prompt": {}},
            }
        ),
    )

    saved = FunctionStore(store_path).get_function("continue_after_prompt")
    assert result["enhanced"] is True
    assert saved is not None
    assert [step.action.tool for step in saved.steps] == ["wait"]
    assert saved.checker_rules[0]["source_state_id"] == "prompt"


def test_enhance_requires_one_function_covering_the_complete_runlog(
    tmp_path: Path,
) -> None:
    run_log = androidworld_run_log(
        [
            {"action_type": "click", "x": 250, "y": 250},
            {"action_type": "click", "x": 750, "y": 750},
        ],
        observations=[androidworld_state("first"), androidworld_state("second")],
        goal="Open the menu and select the item.",
    )
    partial = {
        "schema_version": "omniflow.function.v2",
        "function_id": "open_menu",
        "name": "Open menu",
        "description": "Open the visible menu.",
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
                "source_state_id": "first",
                "action": {"tool": "click", "args": {"x": 250, "y": 250}},
            }
        ],
        "checker_rules": [],
        "agent_visible": True,
    }

    with pytest.raises(
        ValueError,
        match="function_enhancement_full_trajectory_required",
    ):
        save_function(
            run_log,
            tmp_path / "store.json",
            enhance=True,
            complete_json=lambda _prompt, _tool: json.dumps(
                {"functions": [partial], "arguments": {"open_menu": {}}}
            ),
        )


def test_checker_action_cannot_duplicate_a_formal_action_in_the_same_function(
    tmp_path: Path,
) -> None:
    run_log = androidworld_run_log(
        [{"action_type": "click", "x": 500, "y": 500}],
        observations=[androidworld_state("dialog")],
        goal="Dismiss the dialog.",
    )
    duplicated = {
        "schema_version": "omniflow.function.v2",
        "function_id": "dismiss_dialog",
        "name": "Dismiss dialog",
        "description": "Dismiss the visible dialog.",
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
                "source_state_id": "dialog",
                "action": {"tool": "click", "args": {"x": 500, "y": 500}},
            }
        ],
        "checker_rules": [
            {
                "source_state_id": "dialog",
                "action": {"tool": "click", "args": {"x": 500, "y": 500}},
            }
        ],
        "agent_visible": True,
    }

    with pytest.raises(ValueError, match="function_checker_duplicates_formal_action"):
        save_function(
            run_log,
            tmp_path / "store.json",
            enhance=True,
            complete_json=lambda prompt, _tool: json.dumps(
                {
                    "functions": [
                        duplicated
                        if "stage checkers" in prompt
                        else {**duplicated, "checker_rules": []}
                    ],
                    "arguments": {"dismiss_dialog": {}},
                }
            ),
        )


def test_checker_action_requires_a_later_formal_action(tmp_path: Path) -> None:
    run_log = androidworld_run_log(
        [
            {"action_type": "click", "x": 250, "y": 250},
            {"action_type": "click", "x": 750, "y": 750},
        ],
        observations=[androidworld_state("menu"), androidworld_state("item")],
        goal="Open the menu and select the item.",
    )
    function = {
        "schema_version": "omniflow.function.v2",
        "function_id": "open_menu_item",
        "name": "Open menu item",
        "description": "Open the menu and select the item.",
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
                "source_state_id": "menu",
                "action": {"tool": "click", "args": {"x": 250, "y": 250}},
            },
            {
                "step_index": 1,
                "source_state_id": "item",
                "action": {"tool": "click", "args": {"x": 750, "y": 750}},
            },
        ],
        "checker_rules": [],
        "agent_visible": True,
    }
    terminal_as_checker = {
        **function,
        "steps": [function["steps"][0]],
        "checker_rules": [
            {
                "source_state_id": "item",
                "action": {"tool": "click", "args": {"x": 750, "y": 750}},
            }
        ],
    }

    with pytest.raises(ValueError, match="checker_requires_later_formal_action"):
        save_function(
            run_log,
            tmp_path / "store.json",
            enhance=True,
            complete_json=lambda prompt, _tool: json.dumps(
                {
                    "functions": [
                        terminal_as_checker
                        if "stage checkers" in prompt
                        else function
                    ],
                    "arguments": {"open_menu_item": {}},
                }
            ),
        )

    with pytest.raises(ValueError, match="checker_requires_later_formal_action"):
        save_function(
            run_log,
            tmp_path / "direct-store.json",
            functions=[terminal_as_checker],
            arguments={"open_menu_item": {}},
        )


def test_save_function_accepts_one_complete_bmoca_runlog_path(tmp_path: Path) -> None:
    trace = tmp_path / "traces" / "trace-demo"
    screenshots = trace / "screenshots"
    screenshots.mkdir(parents=True)
    before_image = screenshots / "state-before.png"
    after_image = screenshots / "state-after.png"
    before_image.write_bytes(b"before")
    after_image.write_bytes(b"after")
    run_log_path = trace / "runlog.json"
    run_log_path.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.canonical_run_log.v1",
                "run_id": "bmoca-demo",
                "goal": "Open the visible item.",
                "status": "succeeded",
                "success": True,
                "steps": [
                    {
                        "step_index": 0,
                        "before_state_id": "state-before",
                        "action": {
                            "tool": "click",
                            "args": {"x": 500, "y": 500},
                        },
                        "result": {"success": True},
                        "after_state_id": "state-after",
                    }
                ],
                "diagnostics": {
                    "benchmark": "b-moca",
                    "task_id": "demo/open_item",
                    "official_success": True,
                },
                "final_state_id": "state-after",
            }
        ),
        encoding="utf-8",
    )
    (trace / "transfer_states.json").write_text(
        json.dumps(
            {
                "schema_version": "omniflow.transfer-state-catalog.v1",
                "run_id": "bmoca-demo",
                "states": {
                    state_id: {
                        "state_id": state_id,
                        "xml": "<hierarchy />",
                        "package_name": "com.example",
                        "activity_name": ".MainActivity",
                        "display": {"width": 1000, "height": 1000},
                    }
                    for state_id in ("state-before", "state-after")
                },
            }
        ),
        encoding="utf-8",
    )
    (trace / "screenshot_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "omniflow.bmoca-screenshot-manifest.v1",
                "run_id": "bmoca-demo",
                "screenshots": {
                    "state-before": "screenshots/state-before.png",
                    "state-after": "screenshots/state-after.png",
                },
                "referenced_state_ids": ["state-before", "state-after"],
                "missing_referenced_state_ids": [],
                "capture_events": [],
                "complete": True,
            }
        ),
        encoding="utf-8",
    )
    function = {
        "schema_version": "omniflow.function.v2",
        "function_id": "open_visible_item",
        "name": "Open the visible item",
        "description": "Open the item visible at the recorded source target.",
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
                "source_state_id": "state-before",
                "action": {"tool": "click", "args": {"x": 500, "y": 500}},
            }
        ],
        "checker_rules": [],
        "agent_visible": True,
    }

    result = save_function(
        run_log_path,
        tmp_path / "store" / "store.json",
        functions=[function],
        arguments={"open_visible_item": {}},
    )

    assert result["function_ids"] == ["open_visible_item"]
    states = json.loads(
        Path(result["transfer_state_catalog"]).read_text(encoding="utf-8")
    )["states"]
    assert states["state-before"]["screenshot_path"] == str(before_image)


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

    result = _save(
        _run_log(1),
        tmp_path / "output",
        function_bundle=bundle,
        source_states={"state_0": {"state_id": "state_0"}},
    )

    assert result["function_ids"] == ["open_requested_app"]
    assert result["source_arguments"] == bundle["arguments"]
    assert FunctionStore(tmp_path / "output" / "store.json").source_calls == [
        {
            "function_id": "open_requested_app",
            "arguments": {"package_name": "com.android.settings"},
        }
    ]


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

    result = _save(
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
        _save(
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


def test_compiler_accepts_semantic_click_target_at_different_source_positions(
    tmp_path: Path,
) -> None:
    run_log = androidworld_run_log(
        [
            {"action_type": "input_text", "text": "Fast Food"},
            {"action_type": "click", "x": 107, "y": 400},
            {"action_type": "input_text", "text": "Rental Income"},
            {"action_type": "click", "x": 241.5, "y": 400},
        ],
        observations=[
            androidworld_state("food-name", width=720, height=1280),
            androidworld_state(
                "food-category",
                forest=(
                    '<hierarchy><node bounds="[56,358][158,442]" clickable="true">'
                    '<node text="Food" bounds="[70,370][140,430]"/>'
                    "</node></hierarchy>"
                ),
                width=720,
                height=1280,
            ),
            androidworld_state("income-name", width=720, height=1280),
            androidworld_state(
                "income-category",
                forest=(
                    '<hierarchy><node bounds="[174,358][309,442]" clickable="true">'
                    '<node text="Income" bounds="[190,370][290,430]"/>'
                    "</node></hierarchy>"
                ),
                width=720,
                height=1280,
            ),
        ],
        goal="Add two categorized records.",
    )
    bundle = {
        "schema_version": "omniflow.function-bundle.v2",
        "run_id": "source-run",
        "arguments": {
            "add_record": [
                {"name": "Fast Food", "category": "Food"},
                {"name": "Rental Income", "category": "Income"},
            ]
        },
        "functions": [
            {
                "schema_version": "omniflow.function.v2",
                "function_id": "add_record",
                "name": "Add categorized record",
                "description": "Add one record with the requested category.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "category": {"type": "string"},
                    },
                    "required": ["name", "category"],
                    "additionalProperties": False,
                },
                "bindings": [
                    {
                        "source": "$.arguments.name",
                        "target": "$.steps[0].action.args.text",
                    },
                    {
                        "source": "$.arguments.category",
                        "target": "$.steps[1].action.args.target_description",
                    },
                ],
                "steps": [
                    {
                        "step_index": 0,
                        "source_state_id": "food-name",
                        "action": {"tool": "input_text", "args": {"text": ""}},
                    },
                    {
                        "step_index": 1,
                        "source_state_id": "food-category",
                        "action": {
                            "tool": "click",
                            "args": {
                                "target_description": "",
                                "x": 148.61111111111111,
                                "y": 312.5,
                            },
                        },
                    },
                ],
                "checker_rules": [],
                "agent_visible": True,
            }
        ],
    }

    result = _save(
        run_log,
        tmp_path / "output",
        function_bundle=bundle,
        source_states={
            state: {"state_id": state}
            for state in ("food-name", "food-category")
        },
    )

    assert result["source_arguments"] == bundle["arguments"]
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
        _save(
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

    result = _save(
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
    assert FunctionStore(tmp_path / "output" / "store.json").source_calls == [
        {
            "function_id": "add_category_six_expense",
            "arguments": {"name": "Theater Show"},
        },
        {
            "function_id": "add_category_six_expense",
            "arguments": {"name": "Museum Tickets"},
        },
        {
            "function_id": "add_category_one_expense",
            "arguments": {"name": "Household Items"},
        },
    ]
