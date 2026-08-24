from __future__ import annotations

import copy

import pytest

from omniflow.core.schemas import load_function_schema
from omniflow.functions.artifact import parse_function_artifact


def _function() -> dict:
    return {
        "schema_version": "omniflow.function.v2",
        "function_id": "enter_product",
        "name": "Enter product",
        "description": "Enter the requested product.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        "bindings": [
            {
                "source": "$.arguments.text",
                "target": "$.steps[0].action.args.text",
            }
        ],
        "steps": [
            {
                "step_index": 0,
                "source_state_id": "source-form",
                "action": {
                    "tool": "input_text",
                    "args": {
                        "target_description": "Product field",
                        "text": "",
                    },
                },
            }
        ],
        "agent_visible": True,
    }


def test_shared_function_schema_is_the_parser_structure_source() -> None:
    schema = load_function_schema()

    assert schema["properties"]["schema_version"]["const"] == (
        "omniflow.function.v2"
    )
    assert set(schema["required"]) == set(_function())
    assert parse_function_artifact(_function()).function_id == "enter_product"


def test_shared_function_schema_closes_input_schema_shape() -> None:
    value = _function()
    value["input_schema"]["legacy_parameters"] = {}

    with pytest.raises(ValueError, match=r"function_schema_invalid:.*additionalProperties"):
        parse_function_artifact(value)


def test_shared_function_schema_forbids_coordinate_bindings() -> None:
    value = copy.deepcopy(_function())
    value["input_schema"] = {
        "type": "object",
        "properties": {"x": {"type": "number"}},
        "required": ["x"],
        "additionalProperties": False,
    }
    value["bindings"] = [
        {
            "source": "$.arguments.x",
            "target": "$.steps[0].action.args.x",
        }
    ]
    value["steps"][0]["action"] = {
        "tool": "click",
        "args": {"x": 500, "y": 500},
    }

    with pytest.raises(ValueError, match="function_binding_target_non_parameterizable"):
        parse_function_artifact(value)


def test_step_index_schema_documents_function_local_semantics() -> None:
    schema = load_function_schema()
    description = schema["$defs"]["step"]["properties"]["step_index"][
        "description"
    ]

    assert "Function-local" in description
    assert "MUST NOT copy source RunLog step indices" in description
