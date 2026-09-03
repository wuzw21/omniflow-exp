from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from omniflow.core.config import PluginSet
from omniflow.core.model import (
    Action,
    ActionResult,
    Function,
    Observation,
    TransferResult,
)
from omniflow.functions.artifact import (
    bind_function,
    parse_function_artifact,
    render_bound_source_state,
    render_bound_target_state,
)
from omniflow.functions.compiler import (
    _authoring_candidate_catalog,
    _default_authoring_workflow_prompt,
    _direct_source_authoring_plan,
    _materialize_authoring_plan,
    _materialize_authoring_response,
    _materialize_authoring_workflow,
    _source_parameter_candidates,
    _source_node_parameter_evidence,
    compile_runlog_to_store,
)
from omniflow.functions.management import semantic_parameter_evidence
from omniflow.runtime.execution import execute_function
from src.experiment.function_v2 import load_v2_source_calls

SOURCE_XML = (
    '<hierarchy width="720" height="1280">'
    '<node id="row-1" class="android.widget.LinearLayout" '
    'content-desc="File yMkm_calm_umbrella" clickable="true" '
    'bounds="[0,898][720,991]" />'
    '</hierarchy>'
)


def _function() -> Function:
    return parse_function_artifact(
        {
            "schema_version": "omniflow.function.v2",
            "function_id": "delete_note",
            "name": "Delete note",
            "description": "Delete the requested <file_name>.",
            "input_schema": {
                "type": "object",
                "properties": {"file_name": {"type": "string"}},
                "required": ["file_name"],
                "additionalProperties": False,
            },
            "bindings": [],
            "render_bindings": [
                {
                    "source": "$.arguments.file_name",
                    "step_index": 0,
                    "node_id": "row-1",
                    "attribute": "content-desc",
                    "recorded_value": "yMkm_calm_umbrella",
                }
            ],
            "steps": [
                {
                    "step_index": 0,
                    "source_state_id": "state-1",
                    "action": {
                        "tool": "long_press",
                        "args": {"x": 486.111, "y": 726.562},
                    },
                }
            ],
            "agent_visible": True,
        }
    )


def test_bind_renders_node_without_changing_action_coordinates() -> None:
    function = bind_function(_function(), {"file_name": "happy_pig_backup"})

    assert function.steps[0].action.args == {"x": 486.111, "y": 726.562}
    assert "replacement" not in function.to_dict()["render_bindings"][0]
    rendered = render_bound_source_state(
        Observation(xml=SOURCE_XML),
        function.render_bindings,
        step_index=0,
    )
    assert "File &lt;file_name&gt;" in (rendered.xml or "")
    assert "File yMkm_calm_umbrella" in SOURCE_XML


def test_render_binding_masks_target_value_with_the_same_parameter() -> None:
    function = bind_function(_function(), {"file_name": "garden_layout_plan_backup"})
    target = Observation(
        xml=SOURCE_XML.replace("yMkm_calm_umbrella", "garden_layout_plan_backup")
    )

    rendered = render_bound_target_state(
        target,
        function.render_bindings,
        step_index=0,
    )

    assert "File &lt;file_name&gt;" in (rendered.xml or "")
    assert "garden_layout_plan_backup" in (target.xml or "")


def test_render_binding_fails_closed_when_source_literal_is_missing() -> None:
    function = bind_function(_function(), {"file_name": "happy_pig_backup"})

    with pytest.raises(ValueError, match="literal_missing"):
        render_bound_source_state(
            Observation(xml=SOURCE_XML.replace("yMkm_calm_umbrella", "other")),
            function.render_bindings,
            step_index=0,
        )


def test_compiler_extracts_task_value_from_clicked_node() -> None:
    evidence = _source_node_parameter_evidence(
        source_step={"observation": {"xml": SOURCE_XML}},
        action={
            "tool": "long_press",
            "args": {"x": 486.111, "y": 726.562},
        },
        source_step_index=2,
        task_parameters={"file_name": "yMkm_calm_umbrella", "seed": 111},
    )

    assert evidence == [
        {
            "source_step_index": 2,
            "tool": "long_press",
            "parameter_name": "file_name",
            "suggested_name": "file_name",
            "task_parameter_value": "yMkm_calm_umbrella",
            "recorded_value": "yMkm_calm_umbrella",
            "node_id": "row-1",
            "attribute": "content-desc",
            "node_label": "File yMkm_calm_umbrella",
        }
    ]


def test_compiler_extracts_task_value_from_clicked_card_sibling_label() -> None:
    xml = (
        '<hierarchy width="1000" height="1000">'
        '<node id="list" bounds="[0,0][1000,1000]">'
        '<node id="card-1" bounds="[0,0][1000,400]">'
        '<node id="image-1" content-desc="Recipe photo" bounds="[0,0][300,400]" />'
        '<node id="title-1" text="First Recipe" bounds="[600,0][1000,200]" />'
        '<node id="description-1" text="A nearby description" bounds="[300,200][1000,400]" />'
        '</node>'
        '<node id="card-2" bounds="[0,400][1000,800]">'
        '<node id="image-2" content-desc="Recipe photo" bounds="[0,400][300,800]" />'
        '<node id="title-2" text="Second Recipe" bounds="[300,400][1000,800]" />'
        '</node>'
        '</node>'
        '</hierarchy>'
    )

    evidence = _source_node_parameter_evidence(
        source_step={"observation": {"xml": xml}},
        action={"tool": "click", "args": {"x": 100, "y": 200}},
        source_step_index=1,
        task_parameters={
            "row_objects": [
                {"title": "First Recipe", "description": "A nearby description"},
                {"title": "Second Recipe"},
            ]
        },
    )

    assert [(item["node_id"], item["recorded_value"]) for item in evidence] == [
        ("title-1", "First Recipe")
    ]


def test_compiler_extracts_value_from_serialized_row_object() -> None:
    expense_xml = (
        '<hierarchy width="720" height="1280">'
        '<node id="expense-row" text="Bike Repairs" '
        'resource-id="com.arduia.expense:id/tv_name" '
        'bounds="[206,986][475,1029]" />'
        '</hierarchy>'
    )
    evidence = _source_node_parameter_evidence(
        source_step={"observation": {"xml": expense_xml}},
        action={"tool": "click", "args": {"x": 500, "y": 800}},
        source_step_index=2,
        task_parameters={
            "row_objects": [
                "Expense(name='Bike Repairs', amount=32155, note='Urgent')"
            ]
        },
    )

    assert evidence == [
        {
            "source_step_index": 2,
            "tool": "click",
            "parameter_name": "expense_name",
            "suggested_name": "expense_name",
            "task_parameter_value": "Bike Repairs",
            "recorded_value": "Bike Repairs",
            "node_id": "expense-row",
            "attribute": "text",
            "node_label": "Bike Repairs",
        }
    ]


def test_compiler_deduplicates_equivalent_parameters_for_one_render_target() -> None:
    calendar_xml = (
        '<hierarchy width="720" height="1280">'
        '<node id="event-row" text="Workshop on Annual Report" '
        'bounds="[300,350][540,430]" />'
        '</hierarchy>'
    )
    evidence = _source_node_parameter_evidence(
        source_step={"observation": {"xml": calendar_xml}},
        action={"tool": "click", "args": {"x": 583.333, "y": 304.688}},
        source_step_index=2,
        task_parameters={
            "event_title": "Workshop on Annual Report",
            "row_objects": [
                "CalendarEvent(title='Workshop on Annual Report')"
            ],
        },
    )

    assert len(evidence) == 1
    assert evidence[0]["parameter_name"] == "event_title"
    assert evidence[0]["node_id"] == "event-row"
    assert evidence[0]["attribute"] == "text"


def test_filename_stem_is_a_distinct_parameter_contract() -> None:
    evidence = semantic_parameter_evidence(
        {"action": {"tool": "input_text", "args": {"text": "source_note"}}},
        "text",
        "source_note",
        {
            "task_parameters": {"file_name": "source_note.txt"},
            "goal": "Create source_note.txt.",
        },
    )

    assert evidence == {
        "evidence": "task_parameter_filename_stem",
        "suggested_name": "file_stem",
        "fixed_suffix": ".txt",
    }


def test_compiler_materializes_filename_stem_schema() -> None:
    facts = {
        "run_id": "run-1",
        "goal": "Create source_note.txt.",
        "task_parameters": {"file_name": "source_note.txt"},
        "steps": [
            {
                "source_step_index": 4,
                "before_state_id": "state-1",
                "action": {
                    "tool": "input_text",
                    "args": {"text": "source_note", "x": 500, "y": 500},
                },
                "metadata": {},
            }
        ],
        "node_parameter_evidence": [],
    }

    candidates = _source_parameter_candidates(facts)
    assert candidates[0]["suggested_name"] == "file_stem"
    result = _materialize_authoring_plan(
        {
            "reason": "Enter the requested filename.",
            "plan": {
                "functions": [],
                "complete_function": {
                    "function_id": "create_file",
                    "name": "Create file",
                    "description": "Create the requested file.",
                    "source_step_indices": [4],
                    "parameters": [
                        {
                            "name": "file_name",
                            "description": "The requested filename",
                            "source_step_index": 4,
                            "arg_name": "text",
                        }
                    ],
                },
            },
        },
        facts,
    )

    function = result["bundle"]["functions"][0]
    assert function["input_schema"]["required"] == ["file_stem"]
    assert ".txt extension" in function["input_schema"]["properties"]["file_stem"]["description"]


def test_compiler_disambiguates_same_parameter_name_with_different_values() -> None:
    facts = {
        "run_id": "run-1",
        "goal": "Enter Pasta and Soup.",
        "task_parameters": {"titles": ["Pasta", "Soup"]},
        "steps": [
            {
                "source_step_index": index,
                "before_state_id": f"state-{index}",
                "action": {
                    "tool": "input_text",
                    "args": {"text": value, "x": 500, "y": 500},
                },
                "metadata": {},
            }
            for index, value in enumerate(("Pasta", "Soup"))
        ],
        "node_parameter_evidence": [],
    }

    result = _materialize_authoring_plan(
        {
            "reason": "Enter two requested titles.",
            "plan": {
                "functions": [],
                "complete_function": {
                    "function_id": "enter_titles",
                    "name": "Enter titles",
                    "description": "Enter the requested titles.",
                    "source_step_indices": [0, 1],
                    "parameters": [
                        {
                            "name": "title",
                            "description": "Requested title",
                            "source_step_index": index,
                            "arg_name": "text",
                        }
                        for index in range(2)
                    ],
                },
            },
        },
        facts,
    )

    function = next(
        item
        for item in result["bundle"]["functions"]
        if item["function_id"] == "enter_titles"
    )
    assert function["input_schema"]["required"] == ["titles", "titles_2"]
    assert result["bundle"]["arguments"]["enter_titles"] == {
        "titles": "Pasta",
        "titles_2": "Soup",
    }


def test_compiler_drops_parameter_already_bound_by_repeated_literal() -> None:
    facts = {
        "run_id": "run-1",
        "goal": "Enter the same title twice.",
        "task_parameters": {},
        "steps": [
            {
                "source_step_index": index,
                "before_state_id": f"state-{index}",
                "action": {
                    "tool": "input_text",
                    "args": {"text": "Pasta", "x": 500, "y": 500},
                },
                "metadata": {},
            }
            for index in range(2)
        ],
        "node_parameter_evidence": [],
    }

    result = _materialize_authoring_plan(
        {
            "reason": "Enter the repeated requested title.",
            "plan": {
                "functions": [],
                "complete_function": {
                    "function_id": "enter_title_twice",
                    "name": "Enter title twice",
                    "description": "Enter the requested title twice.",
                    "source_step_indices": [0, 1],
                    "parameters": [
                        {
                            "name": "title",
                            "description": "Requested title",
                            "source_step_index": 0,
                            "arg_name": "text",
                        },
                        {
                            "name": "input_text",
                            "description": "Repeated title",
                            "source_step_index": 1,
                            "arg_name": "text",
                        },
                    ],
                },
            },
        },
        facts,
    )

    function = next(
        item
        for item in result["bundle"]["functions"]
        if item["function_id"] == "enter_title_twice"
    )
    assert function["input_schema"]["required"] == ["title"]
    assert result["bundle"]["arguments"]["enter_title_twice"] == {
        "title": "Pasta"
    }
    assert len(function["bindings"]) == 2


def test_direct_source_fallback_derives_action_and_render_bindings() -> None:
    facts = {
        "run_id": "run-1",
        "goal": "Create a recipe titled Pasta.",
        "task_parameters": {"title": "Pasta"},
        "steps": [
            {
                "source_step_index": 0,
                "before_state_id": "state-0",
                "action": {
                    "tool": "input_text",
                    "args": {"text": "Pasta", "x": 500, "y": 500},
                },
                "metadata": {},
            }
        ],
        "node_parameter_evidence": [
            {
                "source_step_index": 0,
                "tool": "input_text",
                "parameter_name": "title",
                "suggested_name": "title",
                "task_parameter_value": "Pasta",
                "recorded_value": "Pasta",
                "node_id": "title-field",
                "attribute": "text",
                "node_label": "Pasta",
            }
        ],
    }

    result = _direct_source_authoring_plan(facts)

    function = result["bundle"]["functions"][0]
    assert function["function_id"] == "complete_source_workflow"
    assert function["input_schema"]["required"] == ["title"]
    assert result["bundle"]["arguments"]["complete_source_workflow"] == {
        "title": "Pasta"
    }
    assert function["bindings"] == [
        {
            "source": "$.arguments.title",
            "target": "$.steps[0].action.args.text",
        }
    ]
    assert function["render_bindings"][0]["node_id"] == "title-field"


def test_execution_sends_both_masked_endpoints_to_transfer() -> None:
    function = bind_function(_function(), {"file_name": "happy_pig_backup"})
    captured: list[tuple[str, str]] = []
    target_xml = SOURCE_XML.replace("yMkm_calm_umbrella", "happy_pig_backup")

    def transfer(action: Action, target: Observation, source: Observation) -> TransferResult:
        captured.append((source.xml or "", target.xml or ""))
        return TransferResult(action)

    class Host:
        async def observe(self, **_kwargs):
            return Observation(xml=target_xml)

        async def act(self, _action):
            return ActionResult(True)

    result = asyncio.run(
        execute_function(
            function,
            host=Host(),
            plugins=PluginSet(transfer=transfer),
            observation=Observation(xml=target_xml),
            state_loader=lambda _state_id: Observation(xml=SOURCE_XML),
        )
    )

    assert result.success is True
    assert len(captured) == 1
    assert "File &lt;file_name&gt;" in captured[0][0]
    assert "File &lt;file_name&gt;" in captured[0][1]


def test_compiler_materializes_node_binding_and_redacts_description() -> None:
    facts = {
        "run_id": "run-1",
        "goal": "Delete the note yMkm_calm_umbrella.",
        "steps": [
            {
                "source_step_index": 2,
                "before_state_id": "state-1",
                "action": {
                    "tool": "long_press",
                    "args": {"x": 486.111, "y": 726.562},
                },
                "metadata": {},
            }
        ],
        "node_parameter_evidence": _source_node_parameter_evidence(
            source_step={"observation": {"xml": SOURCE_XML}},
            action={
                "tool": "long_press",
                "args": {"x": 486.111, "y": 726.562},
            },
            source_step_index=2,
            task_parameters={"file_name": "yMkm_calm_umbrella"},
        ),
    }
    result = _materialize_authoring_plan(
        {
            "reason": "Use the selected node.",
            "plan": {
                "functions": [],
                "complete_function": {
                    "function_id": "delete_note",
                    "name": "Delete yMkm_calm_umbrella",
                    "description": "Delete yMkm_calm_umbrella.",
                    "source_step_indices": [2],
                    "parameters": [],
                },
            },
        },
        facts,
    )

    function = result["bundle"]["functions"][0]
    assert function["input_schema"]["required"] == ["file_name"]
    assert function["render_bindings"][0]["node_id"] == "row-1"
    assert "yMkm_calm_umbrella" not in function["name"]
    assert "yMkm_calm_umbrella" not in function["description"]


def test_compiler_disambiguates_repeated_structured_field_parameters() -> None:
    titles = ["First Recipe", "Second Recipe", "Third Recipe"]
    facts = {
        "run_id": "run-1",
        "goal": "Delete First Recipe, Second Recipe, and Third Recipe.",
        "steps": [
            {
                "source_step_index": index,
                "before_state_id": f"state-{index}",
                "action": {"tool": "click", "args": {"x": 500, "y": 500}},
                "metadata": {},
            }
            for index in range(3)
        ],
        "node_parameter_evidence": [
            {
                "source_step_index": index,
                "tool": "click",
                "parameter_name": "recipe_title",
                "suggested_name": "recipe_title",
                "task_parameter_value": title,
                "recorded_value": title,
                "node_id": f"recipe-{index}",
                "attribute": "text",
                "node_label": title,
            }
            for index, title in enumerate(titles)
        ],
    }

    result = _materialize_authoring_plan(
        {
            "reason": "Delete the requested recipes in source order.",
            "plan": {
                "functions": [],
                "complete_function": {
                    "function_id": "delete_recipes",
                    "name": "Delete requested recipes",
                    "description": "Delete First Recipe, Second Recipe, and Third Recipe.",
                    "source_step_indices": [0, 1, 2],
                    "parameters": [],
                },
            },
        },
        facts,
    )

    function = next(
        item
        for item in result["bundle"]["functions"]
        if item["function_id"] == "delete_recipes"
    )
    assert function["input_schema"]["required"] == [
        "recipe_title",
        "recipe_title_2",
        "recipe_title_3",
    ]
    assert [binding["source"] for binding in function["render_bindings"]] == [
        "$.arguments.recipe_title",
        "$.arguments.recipe_title_2",
        "$.arguments.recipe_title_3",
    ]
    assert result["bundle"]["arguments"]["delete_recipes"] == {
        "recipe_title": "First Recipe",
        "recipe_title_2": "Second Recipe",
        "recipe_title_3": "Third Recipe",
    }


def test_authoring_harness_registers_one_repeated_function_with_three_calls() -> None:
    titles = ["First Recipe", "Second Recipe", "Third Recipe"]
    facts = {
        "run_id": "run-1",
        "goal": "Delete First Recipe, Second Recipe, and Third Recipe.",
        "task_parameters": {},
        "parameter_evidence": [],
        "steps": [
            {
                "source_step_index": index,
                "before_state_id": f"state-{index}",
                "action": {"tool": "click", "args": {"x": 500, "y": 500}},
                "metadata": {},
            }
            for index in range(3)
        ],
        "node_parameter_evidence": [
            {
                "source_step_index": index,
                "tool": "click",
                "parameter_name": "recipe_title",
                "suggested_name": "recipe_title",
                "task_parameter_value": title,
                "recorded_value": title,
                "node_id": f"recipe-{index}",
                "attribute": "text",
                "node_label": title,
            }
            for index, title in enumerate(titles)
        ],
    }
    _public_candidates, candidate_map = _authoring_candidate_catalog(facts)
    result = _materialize_authoring_workflow(
        {
            "reason": "Reuse one stable delete operation for three recipes.",
            "functions": [
                {
                    "function_id": "delete_recipe",
                    "name": "Delete requested recipe",
                    "description": "Delete one requested recipe.",
                    "occurrences": [
                        {"source_step_indices": [0]},
                        {"source_step_indices": [1]},
                        {"source_step_indices": [2]},
                    ],
                    "parameters": [
                        {
                            "name": "recipe_title",
                            "description": "Recipe title requested by the goal",
                            "bindings": [
                                {
                                    "occurrence_index": index,
                                    "candidate_id": f"render_parameter_{index:03d}",
                                }
                                for index in range(3)
                            ],
                        }
                    ],
                }
            ],
            "complete_function": {
                "function_id": "delete_requested_recipes",
                "name": "Delete requested recipes",
                "description": "Delete all recipes requested by the goal.",
                "source_step_indices": [0, 1, 2],
            },
        },
        facts,
        candidate_map=candidate_map,
    )

    assert result["authoring_workflow"]["definition_count"] == 1
    assert result["authoring_workflow"]["invocation_count"] == 3
    assert [call["function_id"] for call in result["source_calls"]] == [
        "delete_recipe",
        "delete_recipe",
        "delete_recipe",
    ]
    assert [call["arguments"] for call in result["source_calls"]] == [
        {"recipe_title": title} for title in titles
    ]
    functions = result["bundle"]["functions"]
    assert [function["function_id"] for function in functions].count(
        "delete_recipe"
    ) == 1
    repeated = next(
        function
        for function in functions
        if function["function_id"] == "delete_recipe"
    )
    assert len(repeated["steps"]) == 1
    assert repeated["input_schema"]["required"] == ["recipe_title"]


def test_authoring_harness_derives_invocation_order_from_source_steps() -> None:
    facts = {
        "run_id": "run-1",
        "goal": "Perform the first and second operation.",
        "task_parameters": {},
        "parameter_evidence": [],
        "node_parameter_evidence": [],
        "steps": [
            {
                "source_step_index": index,
                "before_state_id": f"state-{index}",
                "action": {"tool": "click", "args": {"x": 500, "y": 500}},
                "metadata": {},
            }
            for index in range(2)
        ],
    }

    result = _materialize_authoring_workflow(
        {
            "reason": "Identify two stable operations.",
            "functions": [
                {
                    "function_id": "second_operation",
                    "name": "Second operation",
                    "description": "Perform the second stable operation.",
                    "occurrences": [{"source_step_indices": [1]}],
                    "parameters": [],
                },
                {
                    "function_id": "first_operation",
                    "name": "First operation",
                    "description": "Perform the first stable operation.",
                    "occurrences": [{"source_step_indices": [0]}],
                    "parameters": [],
                },
            ],
            "complete_function": {
                "function_id": "complete_operations",
                "name": "Complete operations",
                "description": "Perform both requested operations.",
                "source_step_indices": [0, 1],
            },
        },
        facts,
        candidate_map={},
    )

    assert [call["function_id"] for call in result["source_calls"]] == [
        "first_operation",
        "second_operation",
    ]


def test_authoring_harness_accepts_unselected_render_candidates() -> None:
    facts = {
        "run_id": "run-1",
        "goal": "Create the Pasta recipe.",
        "task_parameters": {"title": "Pasta"},
        "parameter_evidence": [],
        "node_parameter_evidence": [
            {
                "source_step_index": 0,
                "tool": "input_text",
                "parameter_name": "title",
                "suggested_name": "title",
                "task_parameter_value": "Pasta",
                "recorded_value": "Pasta",
                "node_id": "title-field",
                "attribute": "text",
                "node_label": "Pasta",
            }
        ],
        "steps": [
            {
                "source_step_index": 0,
                "before_state_id": "state-0",
                "action": {
                    "tool": "input_text",
                    "args": {"text": "Pasta", "x": 500, "y": 500},
                },
                "metadata": {},
            }
        ],
    }
    _public_candidates, candidate_map = _authoring_candidate_catalog(facts)

    result = _materialize_authoring_workflow(
        {
            "reason": "Create a reusable recipe entry Function.",
            "functions": [
                {
                    "function_id": "create_recipe",
                    "name": "Create recipe",
                    "description": "Create the requested recipe.",
                    "occurrences": [{"source_step_indices": [0]}],
                    "parameters": [
                        {
                            "name": "title",
                            "description": "Recipe title",
                            "bindings": [
                                {
                                    "occurrence_index": 0,
                                    "candidate_id": "action_parameter_000",
                                }
                            ],
                        }
                    ],
                }
            ],
            "complete_function": {
                "function_id": "complete_recipe",
                "name": "Complete recipe",
                "description": "Create the requested recipe.",
                "source_step_indices": [0],
            },
        },
        facts,
        candidate_map=candidate_map,
    )

    assert result["authoring_workflow"]["unselected_candidate_ids"] == [
        "render_parameter_000"
    ]
    complete_recipe = next(
        function
        for function in result["bundle"]["functions"]
        if function["function_id"] == "complete_recipe"
    )
    assert complete_recipe["input_schema"]["required"] == ["title"]
    assert result["authoring_workflow"]["definitions"][0]["registered"] is False
    assert result["source_calls"] == [
        {"function_id": "complete_recipe", "arguments": {"title": "Pasta"}}
    ]


def test_authoring_harness_selects_modal_repeated_occurrence_shape() -> None:
    actions = [
        {"tool": "click", "args": {"x": 100, "y": 100}},
        {
            "tool": "swipe",
            "args": {"x1": 100, "y1": 800, "x2": 100, "y2": 200},
        },
        {"tool": "navigate_back", "args": {}},
        {"tool": "click", "args": {"x": 100, "y": 100}},
        {"tool": "navigate_back", "args": {}},
        {"tool": "click", "args": {"x": 100, "y": 100}},
    ]
    facts = {
        "run_id": "run-1",
        "goal": "Perform a repeated operation.",
        "task_parameters": {},
        "parameter_evidence": [],
        "node_parameter_evidence": [],
        "steps": [
            {
                "source_step_index": index,
                "before_state_id": f"state-{index}",
                "action": action,
                "metadata": {},
            }
            for index, action in enumerate(actions)
        ],
    }

    result = _materialize_authoring_workflow(
        {
            "reason": "The repeated operation has one longer setup variant.",
            "functions": [
                {
                    "function_id": "repeat_operation",
                    "name": "Repeat operation",
                    "description": "Perform the reusable operation.",
                    "occurrences": [
                        {"source_step_indices": [0, 1]},
                        {"source_step_indices": [3]},
                        {"source_step_indices": [5]},
                    ],
                    "parameters": [],
                }
            ],
            "complete_function": {
                "function_id": "complete_operation",
                "name": "Complete operation",
                "description": "Complete the recorded operation.",
                "source_step_indices": list(range(6)),
            },
        },
        facts,
        candidate_map={},
    )

    definition = result["authoring_workflow"]["definitions"][0]
    assert definition["representative_source_step_indices"] == [3]
    assert result["authoring_workflow"]["uncovered_local_source_step_indices"] == [
        2,
        4,
    ]
    assert len(result["source_calls"]) == 3


def test_authoring_harness_accepts_complete_function_without_locals() -> None:
    facts = {
        "run_id": "run-1",
        "goal": "Open one recipe.",
        "task_parameters": {},
        "parameter_evidence": [],
        "node_parameter_evidence": [],
        "steps": [
            {
                "source_step_index": 0,
                "before_state_id": "state-0",
                "action": {"tool": "click", "args": {"x": 500, "y": 500}},
                "metadata": {},
            }
        ],
    }

    result = _materialize_authoring_workflow(
        {
            "reason": "The complete capability is already atomic.",
            "functions": [],
            "complete_function": {
                "function_id": "open_recipe",
                "name": "Open recipe",
                "description": "Open the requested recipe.",
                "source_step_indices": [],
            },
        },
        facts,
        candidate_map={},
    )

    assert result["authoring_workflow"]["definition_count"] == 0
    assert result["authoring_workflow"]["complete_source_indices_normalized"] is True
    assert result["source_calls"] == [
        {"function_id": "open_recipe", "arguments": {}}
    ]
    assert [
        function["function_id"] for function in result["bundle"]["functions"]
    ] == ["open_recipe"]


def test_default_authoring_prompt_requires_semantic_classification() -> None:
    prompt = _default_authoring_workflow_prompt()

    assert "Stage 1 — classify semantics" in prompt
    assert "Stage 2 — discover Functions" in prompt
    assert "Stage 3 — author every binding on the A side" in prompt
    assert "Stage 4 — convert and register" in prompt
    assert "stable" in prompt
    assert "task_parameter" in prompt
    assert "online_observation" in prompt
    assert "planner_handoff" in prompt
    assert '"validation"' not in prompt
    assert '"registration"' not in prompt
    assert "Any undeclared stable value keeps the" in prompt
    assert '"binding_owner": "agent"' in prompt
    assert "occurrence_values" not in prompt
    assert "other fixed control" in prompt
    assert "binding_evidence" not in prompt
    assert "candidate_id" not in prompt


def test_agent_owned_bindings_are_converted_without_compiler_candidates() -> None:
    facts = {
        "run_id": "agent-owned",
        "goal": "Create the Pasta recipe.",
        "task_parameters": {"title": "Pasta"},
        "steps": [
            {
                "source_step_index": 0,
                "before_state_id": "state-0",
                "action": {
                    "tool": "input_text",
                    "args": {"text": "Pasta", "x": 500, "y": 500},
                },
            },
            {
                "source_step_index": 1,
                "before_state_id": "state-1",
                "action": {"tool": "click", "args": {"x": 500, "y": 500}},
            },
            {
                "source_step_index": 2,
                "before_state_id": "state-2",
                "action": {
                    "tool": "input_text",
                    "args": {"text": ".txt", "x": 500, "y": 500},
                },
            },
        ],
    }
    result = _materialize_authoring_response(
        {
            "binding_owner": "agent",
            "reason": "The A-side Agent authored the title binding.",
            "semantic_analysis": {
                "steps": [
                    {
                        "source_step_index": 0,
                        "semantic_kind": "task_parameter",
                        "parameter_names": ["title"],
                        "reason": "The input text comes from the task goal.",
                    },
                    {
                        "source_step_index": 1,
                        "semantic_kind": "task_parameter",
                        "parameter_names": ["title"],
                        "reason": "The selected row is named by the task goal.",
                    },
                    {
                        "source_step_index": 2,
                        "semantic_kind": "stable",
                        "parameter_names": [],
                        "reason": "The suffix is invariant.",
                    },
                ]
            },
            "functions": [],
            "complete_function": {
                "function_id": "create_recipe",
                "name": "Create recipe",
                "description": "Create the requested recipe.",
                "source_step_indices": [0, 1, 2],
                "execution_mode": "direct_replay",
                "parameters": [
                    {
                        "name": "title",
                        "description": "Requested recipe title",
                        "bindings": [
                            {
                                "occurrence_index": 0,
                                "source_step_index": 0,
                                "binding_kind": "action_arg",
                                "arg_name": "text",
                                "recorded_value": "Pasta",
                            },
                            {
                                "occurrence_index": 0,
                                "source_step_index": 1,
                                "binding_kind": "render_node",
                                "node_id": "recipe-row",
                                "attribute": "text",
                                "recorded_value": "Pasta",
                            },
                        ],
                    }
                ],
            },
        },
        facts,
        candidate_map={},
    )

    function = result["bundle"]["functions"][0]
    assert function["bindings"] == [
        {
            "source": "$.arguments.title",
            "target": "$.steps[0].action.args.text",
        }
    ]
    assert function["render_bindings"] == [
        {
            "source": "$.arguments.title",
            "step_index": 1,
            "node_id": "recipe-row",
            "attribute": "text",
            "recorded_value": "Pasta",
        }
    ]
    assert function["steps"][2]["action"]["args"]["text"] == ".txt"
    assert "input_text" not in function["input_schema"]["properties"]


def test_agent_cannot_label_task_parameter_then_emit_empty_schema() -> None:
    facts = {
        "run_id": "calendar-event",
        "goal": "Create an event titled Meeting with HR.",
        "task_parameters": {"title": "Meeting with HR"},
        "steps": [
            {
                "source_step_index": 0,
                "before_state_id": "calendar-editor",
                "action": {
                    "tool": "input_text",
                    "args": {"text": "Call with Dr. Smith", "x": 500, "y": 500},
                },
            }
        ],
    }

    with pytest.raises(
        ValueError,
        match="function_author_task_parameter_binding_incomplete",
    ):
        _materialize_authoring_response(
            {
                "binding_owner": "agent",
                "reason": "The event title varies with the task.",
                "semantic_analysis": {
                    "steps": [
                        {
                            "source_step_index": 0,
                            "semantic_kind": "task_parameter",
                            "parameter_names": ["event_title"],
                            "reason": "The title is supplied by the current goal.",
                        }
                    ]
                },
                "functions": [],
                "complete_function": {
                    "function_id": "create_calendar_event",
                    "name": "Create calendar event",
                    "description": "Create the requested calendar event.",
                    "source_step_indices": [0],
                    "execution_mode": "direct_replay",
                    "parameters": [],
                },
            },
            facts,
            candidate_map={},
        )


def test_online_observation_hides_complete_replay_and_keeps_safe_local() -> None:
    facts = {
        "run_id": "browser-multiply",
        "goal": "Multiply the two numbers shown on the page.",
        "task_parameters": {},
        "steps": [
            {
                "source_step_index": 0,
                "before_state_id": "browser-page",
                "action": {"tool": "click", "args": {"x": 500, "y": 500}},
            },
            {
                "source_step_index": 1,
                "before_state_id": "answer-field",
                "action": {
                    "tool": "input_text",
                    "args": {"text": "15750", "x": 500, "y": 500},
                },
            },
        ],
    }
    proposal = {
        "binding_owner": "agent",
        "reason": "Navigation is reusable; the product must be recomputed live.",
        "semantic_analysis": {
            "steps": [
                {
                    "source_step_index": 0,
                    "semantic_kind": "stable",
                    "parameter_names": [],
                    "reason": "Focusing the answer field is stable.",
                },
                {
                    "source_step_index": 1,
                    "semantic_kind": "online_observation",
                    "parameter_names": [],
                    "reason": "The answer depends on the numbers visible now.",
                },
            ]
        },
        "functions": [
            {
                "function_id": "focus_answer_field",
                "name": "Focus answer field",
                "description": "Focus the visible answer field.",
                "occurrences": [{"source_step_indices": [0]}],
                "parameters": [],
            }
        ],
        "complete_function": {
            "function_id": "complete_multiplication",
            "name": "Complete multiplication task",
            "description": "Historical evidence for the full source workflow.",
            "source_step_indices": [0, 1],
            "execution_mode": "planner_handoff",
            "parameters": [],
        },
    }

    result = _materialize_authoring_response(proposal, facts, candidate_map={})
    functions = {
        function["function_id"]: function
        for function in result["bundle"]["functions"]
    }

    assert functions["focus_answer_field"]["agent_visible"] is True
    assert functions["complete_multiplication"]["agent_visible"] is False
    assert result["authoring_workflow"]["complete_execution_mode"] == (
        "planner_handoff"
    )
    assert result["authoring_workflow"]["semantic_analysis"]["counts"] == {
        "stable": 1,
        "task_parameter": 0,
        "online_observation": 1,
    }

    unsafe = json.loads(json.dumps(proposal))
    unsafe["complete_function"]["execution_mode"] = "direct_replay"
    with pytest.raises(
        ValueError,
        match="function_author_online_observation_requires_planner_handoff",
    ):
        _materialize_authoring_response(unsafe, facts, candidate_map={})


def test_compile_request_gives_raw_ui_to_agent_and_no_binding_candidates(
    tmp_path,
) -> None:
    state = {
        "pixels": None,
        "xml": (
            '<hierarchy width="1000" height="1000">'
            '<node id="title-field" text="Pasta" bounds="[0,0][500,100]" />'
            "</hierarchy>"
        ),
        "auxiliaries": {"display": {"width": 1000, "height": 1000}},
    }
    proposal = {
        "binding_owner": "agent",
        "reason": "Bind the requested title.",
        "semantic_analysis": {
            "steps": [
                {
                    "source_step_index": 0,
                    "semantic_kind": "task_parameter",
                    "parameter_names": ["title"],
                    "reason": "The text is supplied by the task goal.",
                }
            ]
        },
        "functions": [],
        "complete_function": {
            "function_id": "create_recipe",
            "name": "Create recipe",
            "description": "Create the requested recipe.",
            "source_step_indices": [0],
            "execution_mode": "direct_replay",
            "parameters": [
                {
                    "name": "title",
                    "description": "Requested recipe title",
                    "bindings": [
                        {
                            "occurrence_index": 0,
                            "source_step_index": 0,
                            "binding_kind": "action_arg",
                            "arg_name": "text",
                        }
                    ],
                }
            ],
        },
    }

    class CapturingCompletions:
        request = None

        def create(self, **kwargs):
            self.request = kwargs
            return SimpleNamespace(
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=2,
                    total_tokens=12,
                ),
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps(proposal))
                    )
                ],
            )

    completions = CapturingCompletions()
    run_log = {
        "schema_version": "omniflow.run_log.v1",
        "run_id": "agent-owned-request",
        "task_name": "agent-owned-request",
        "goal": "Create the Pasta recipe.",
        "task_parameters": {"title": "Pasta"},
        "seed": 111,
        "status": "succeeded",
        "success": True,
        "validator": {"official": True, "success": True, "reward": 1},
        "provenance": {"kind": "runtime"},
        "steps": [
            {
                "step_index": 0,
                "observation": state,
                "action": {
                    "action_type": "input_text",
                    "text": "Pasta",
                    "x": 500,
                    "y": 500,
                },
                "result": {"success": True},
                "next_observation": state,
            }
        ],
    }
    report = compile_runlog_to_store(
        run_log,
        tmp_path / "memory",
        model="test-author",
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        ),
        state_loader=lambda _state_id: state,
    )

    request = json.loads(completions.request["messages"][1]["content"])
    assert "binding_evidence" not in request
    assert "parameter_evidence" not in request["source_run"]
    assert request["source_run"]["steps"][0]["source_ui"]["nodes"][0][
        "id"
    ] == "title-field"
    assert report["authoring_workflow"]["binding_owner"] == "agent"
    assert report["authoring_workflow"]["agent_proposal_accepted"] is True


def test_agent_authors_semantic_binding_requests_and_harness_materializes_them() -> None:
    facts = {
        "run_id": "run-1",
        "goal": "Create the Pasta recipe.",
        "task_parameters": {"title": "Pasta"},
        "parameter_evidence": [],
        "node_parameter_evidence": [
            {
                "source_step_index": 1,
                "tool": "click",
                "parameter_name": "title",
                "suggested_name": "title",
                "task_parameter_value": "Pasta",
                "recorded_value": "Pasta",
                "node_id": "title-field",
                "attribute": "text",
                "node_label": "Pasta",
            }
        ],
        "steps": [
            {
                "source_step_index": 0,
                "before_state_id": "state-0",
                "action": {
                    "tool": "input_text",
                    "args": {"text": "Pasta", "x": 500, "y": 500},
                },
                "metadata": {},
            },
            {
                "source_step_index": 1,
                "before_state_id": "state-1",
                "action": {"tool": "click", "args": {"x": 500, "y": 500}},
                "metadata": {},
            },
        ],
    }
    public_evidence, candidate_map = _authoring_candidate_catalog(facts)

    assert all("candidate_id" not in item for item in public_evidence)
    result = _materialize_authoring_workflow(
        {
            "reason": "Bind the requested title to the entry action and its node.",
            "functions": [
                {
                    "function_id": "create_recipe",
                    "name": "Create recipe",
                    "description": "Create the requested recipe.",
                    "occurrences": [{"source_step_indices": [0, 1]}],
                    "parameters": [
                        {
                            "name": "title",
                            "description": "Requested recipe title",
                            "bindings": [
                                {
                                    "occurrence_index": 0,
                                    "source_step_index": 0,
                                    "binding_kind": "action_arg",
                                },
                                {
                                    "occurrence_index": 0,
                                    "source_step_index": 1,
                                    "binding_kind": "render_node",
                                },
                            ],
                        }
                    ],
                }
            ],
            "complete_function": {
                "function_id": "complete_recipe",
                "name": "Complete recipe",
                "description": "Create the requested recipe.",
                "source_step_indices": [0, 1],
            },
        },
        facts,
        candidate_map=candidate_map,
    )

    function = next(
        item
        for item in result["bundle"]["functions"]
        if item["function_id"] == "complete_recipe"
    )
    assert function["bindings"] == [
        {"source": "$.arguments.title", "target": "$.steps[0].action.args.text"}
    ]
    assert function["render_bindings"][0]["source"] == "$.arguments.title"
    assert result["source_calls"] == [
        {"function_id": "complete_recipe", "arguments": {"title": "Pasta"}}
    ]
    assert (
        result["authoring_workflow"]["schema_version"]
        == "omniflow.function-authoring-workflow.v2"
    )


def test_agent_semantic_binding_request_fails_closed_without_evidence() -> None:
    facts = {
        "run_id": "run-1",
        "goal": "Select swimming.",
        "task_parameters": {"category": "swimming"},
        "parameter_evidence": [],
        "node_parameter_evidence": [],
        "steps": [
            {
                "source_step_index": 0,
                "before_state_id": "state-0",
                "action": {"tool": "click", "args": {"x": 500, "y": 500}},
                "metadata": {},
            }
        ],
    }

    with pytest.raises(
        ValueError,
        match="function_author_parameter_binding_evidence_missing",
    ):
        _materialize_authoring_workflow(
            {
                "reason": "Bind the requested activity category.",
                "functions": [
                    {
                        "function_id": "select_category",
                        "name": "Select category",
                        "description": "Select the requested activity category.",
                        "occurrences": [{"source_step_indices": [0]}],
                        "parameters": [
                            {
                                "name": "category",
                                "description": "Requested activity category",
                                "bindings": [
                                    {
                                        "occurrence_index": 0,
                                        "source_step_index": 0,
                                        "binding_kind": "render_node",
                                    }
                                ],
                            }
                        ],
                    }
                ],
                "complete_function": {
                    "function_id": "complete_selection",
                    "name": "Complete selection",
                    "description": "Select the requested activity category.",
                    "source_step_indices": [0],
                },
            },
            facts,
            candidate_map={},
        )


def test_authoring_agent_rejection_falls_back_only_to_raw_source_replay(
    tmp_path,
) -> None:
    class InvalidCompletions:
        calls = 0

        def create(self, **_kwargs):
            self.calls += 1
            return SimpleNamespace(
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=2,
                    total_tokens=12,
                ),
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content="{}"))
                ],
            )

    completions = InvalidCompletions()
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    state = {
        "pixels": None,
        "xml": '<hierarchy width="1000" height="1000" />',
        "auxiliaries": {"display": {"width": 1000, "height": 1000}},
    }
    run_log = {
        "schema_version": "omniflow.run_log.v1",
        "run_id": "agent-rejection-test",
        "task_name": "agent-rejection-test",
        "goal": "Tap the requested item.",
        "task_parameters": {"item": "Pasta"},
        "seed": 111,
        "status": "succeeded",
        "success": True,
        "validator": {"official": True, "success": True, "reward": 1},
        "provenance": {"kind": "runtime"},
        "steps": [
            {
                "step_index": 0,
                "observation": state,
                "action": {"action_type": "click", "x": 500, "y": 500},
                "result": {"success": True},
                "next_observation": state,
            }
        ],
    }
    output = tmp_path / "memory"

    report = compile_runlog_to_store(
        run_log,
        output,
        model="test-author",
        client=client,
        state_loader=lambda _state_id: state,
    )

    assert completions.calls == 3
    failure = json.loads((output / "authoring_failure.json").read_text())
    assert failure["success"] is False
    assert (
        report["authoring_workflow"]["fallback_mode"]
        == "hidden_raw_source_evidence"
    )
    function = parse_function_artifact(
        json.loads((output / "store.json").read_text())["functions"][
            "complete_source_workflow"
        ]
    )
    assert function.input_schema["required"] == []
    assert function.bindings == ()
    assert function.render_bindings == ()
    assert function.agent_visible is False
    assert function.steps[0].action.to_dict() == {
        "tool": "click",
        "args": {"x": 500, "y": 500},
    }


def test_node_parameter_evidence_allows_distinct_values_in_one_node() -> None:
    evidence = _source_node_parameter_evidence(
        source_step={
            "observation": {
                "xml": (
                    '<hierarchy width="720" height="1280">'
                    '<node id="editor" class="android.widget.EditText" '
                    'text="Meeting notes\n\nExisting body" clickable="true" '
                    'bounds="[0,200][720,1100]" />'
                    "</hierarchy>"
                )
            }
        },
        action={"tool": "click", "args": {"x": 500, "y": 500}},
        source_step_index=4,
        task_parameters={
            "header": "Meeting notes",
            "original_content": "Existing body",
        },
    )

    assert {
        (item["parameter_name"], item["recorded_value"])
        for item in evidence
    } == {
        ("header", "Meeting notes"),
        ("original_content", "Existing body"),
    }


def test_authoring_harness_explains_complete_parameter_placement() -> None:
    facts = {
        "run_id": "run-1",
        "goal": "Save the recording as meeting_audio.",
        "task_parameters": {},
        "parameter_evidence": [],
        "node_parameter_evidence": [],
        "steps": [
            {
                "source_step_index": 0,
                "before_state_id": "state-0",
                "action": {"tool": "input_text", "args": {"text": "meeting_audio"}},
                "metadata": {},
            }
        ],
    }

    with pytest.raises(
        ValueError,
        match="parameters_must_be_declared_on_a_function",
    ):
        _materialize_authoring_workflow(
            {
                "reason": "The complete workflow needs one file-name parameter.",
                "functions": [],
                "complete_function": {
                    "function_id": "save_recording",
                    "name": "Save recording",
                    "description": "Save a recording under the requested name.",
                    "source_step_indices": [0],
                    "parameters": [],
                },
            },
            facts,
            candidate_map={},
        )


def test_source_call_loader_preserves_repeated_function_references(tmp_path) -> None:
    calls = [
        {
            "function_id": "delete_recipe",
            "arguments": {"recipe_title": title},
        }
        for title in ("First Recipe", "Second Recipe", "Third Recipe")
    ]
    (tmp_path / "compile_report.json").write_text(
        json.dumps({"source_calls": calls}),
        encoding="utf-8",
    )

    assert load_v2_source_calls(tmp_path / "store.json") == calls
