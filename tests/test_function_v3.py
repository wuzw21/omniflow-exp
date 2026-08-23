from __future__ import annotations

import asyncio

from omniflow.core.config import PluginSet
from omniflow.core.model import (
    Action,
    ActionResult,
    Function,
    Observation,
    TransferResult,
)
from omniflow.functions.assets import FunctionStore, save_function
from omniflow.runtime.execution import execute_function


def _observation(label: str) -> dict:
    return {
        "screenshot": None,
        "xml": f'<hierarchy width="100" height="100" text="{label}" />',
    }


def _function(function_id: str, state_ids: list[str]) -> dict:
    return {
        "schema_version": "omniflow.function.v3",
        "function_id": function_id,
        "name": function_id.replace("_", " ").title(),
        "description": f"Execute {function_id}.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "bindings": [],
        "transfer_states": {
            state_id: _observation(state_id) for state_id in state_ids
        },
        "steps": [
            {
                "step_index": 0,
                "transfer_state_ids": list(state_ids),
                "action": {"tool": "click", "args": {"x": 50, "y": 50}},
            }
        ],
        "checker_rules": [],
        "agent_visible": True,
    }


def test_save_function_writes_multiple_v3_functions_without_runlog(tmp_path) -> None:
    store_path = tmp_path / "functions.json"

    report = save_function(
        None,
        store_path,
        functions=[_function("first", ["a"]), _function("second", ["b"])],
        arguments={"first": {}, "second": {}},
    )

    store = FunctionStore(store_path)
    assert report["function_ids"] == ["first", "second"]
    assert report["transfer_state_count"] == 2
    assert set(store.functions) == {"first", "second"}
    assert not (tmp_path / "transfer_states.json").exists()


def test_function_step_can_reference_multiple_observations_and_uses_best() -> None:
    function = Function.from_dict(_function("choose_best", ["low", "high"]))

    class Host:
        def observe(self, **_kwargs):
            return Observation(xml="<hierarchy />", package_name="example")

        def act(self, _action):
            return ActionResult(True)

    def transfer(action, _target, source):
        score = 0.9 if "high" in str(source.xml) else 0.6
        return TransferResult(
            Action(action.tool, {**action.args, "x": 100, "y": 100}),
            detail={"candidates": [{"score": score}]},
        )

    result = asyncio.run(
        execute_function(
            function,
            host=Host(),
            plugins=PluginSet(transfer=transfer),
        )
    )

    transfer_detail = result.detail["trace"][0]["metadata"]["transfer"]
    assert result.success is True
    assert transfer_detail["transfer_state_candidate_index"] == 1
    assert transfer_detail["transfer_state_candidate_count"] == 2


def test_function_action_is_not_compared_with_runlog_order(tmp_path) -> None:
    function = _function("free_order", ["second_page"])
    function["steps"] = [
        {
            "step_index": 0,
            "transfer_state_ids": ["second_page"],
            "action": {"tool": "press_key", "args": {"key": "back"}},
        }
    ]

    report = save_function(
        None,
        tmp_path / "functions.json",
        functions=[function],
        arguments={"free_order": {}},
    )

    assert report["success"] is True
