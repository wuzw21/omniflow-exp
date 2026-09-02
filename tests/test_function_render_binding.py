from __future__ import annotations

import asyncio
import json

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
    _materialize_authoring_plan,
    _materialize_authoring_workflow,
    _source_parameter_candidates,
    _source_node_parameter_evidence,
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


def test_authoring_workflow_registers_one_repeated_function_with_three_calls() -> None:
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
            "workflow": {
                "inventory": {
                    "definitions": [
                        {
                            "function_id": "delete_recipe",
                            "name": "Delete requested recipe",
                            "description": "Delete one requested recipe.",
                            "occurrences": [
                                {"source_step_indices": [0]},
                                {"source_step_indices": [1]},
                                {"source_step_indices": [2]},
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
                "parameterization": [
                    {
                        "function_id": "delete_recipe",
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
                "validation": {
                    "accepted": True,
                    "notes": "Each call deletes one visible recipe and then re-observes.",
                },
                "registration": {
                    "function_ids": [
                        "delete_recipe",
                        "delete_requested_recipes",
                    ],
                    "invocations": [
                        {
                            "function_id": "delete_recipe",
                            "occurrence_index": index,
                        }
                        for index in range(3)
                    ],
                },
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
