from __future__ import annotations

from omniflow.bridge import _run_result
from omniflow.core.model import Function, RunResult
from omniflow.vlm.planner import (
    DEFAULT_STEP_GUIDANCE,
    build_model_turn_request,
    resolve_step_guidance,
)


def _function() -> Function:
    return Function.from_dict(
        {
            "schema_version": "omniflow.function.v2",
            "function_id": "order_beverage",
            "name": "Order a beverage",
            "description": "Order the requested beverage and stop before payment.",
            "input_schema": {
                "type": "object",
                "properties": {"beverage": {"type": "string"}},
                "required": ["beverage"],
                "additionalProperties": False,
            },
            "bindings": [],
            "steps": [],
            "checker_rules": [],
            "agent_visible": True,
        }
    )


def test_planner_guidance_is_explicit_only() -> None:
    assert resolve_step_guidance("find a contact") == DEFAULT_STEP_GUIDANCE
    assert DEFAULT_STEP_GUIDANCE == ""
    assert resolve_step_guidance("order coffee", "custom") == "custom"


def test_bridge_planner_exposes_function_with_native_actions() -> None:
    function = _function()
    request = build_model_turn_request(
        goal="order me a latte",
        model="scene.vlm.operation.primary",
        state={"state_id": "state-1", "display": {"width": 100, "height": 200}},
        functions=(function,),
        max_steps=20,
        turn_index=1,
    )

    tool_names = [tool["function"]["name"] for tool in request["tools"]]
    assert "click" in tool_names
    assert "swipe" in tool_names
    assert function.id in tool_names


def test_core_has_one_planner_implementation() -> None:
    import omniflow.bridge as bridge
    from omniflow.vlm.planner import VLMPlanner

    assert not hasattr(bridge, "_BridgePlanner")
    assert VLMPlanner.__name__ == "VLMPlanner"


def test_successful_online_run_requests_registration_after_run() -> None:
    payload = _run_result(
        RunResult(
            True,
            actions_executed=3,
            detail={
                "trace": [],
                "done_reason": "finished",
                "function_resolution": {
                    "selected_function_id": None,
                },
            },
        ),
        body={"run_id": "run-1", "goal": "order coffee"},
        function=None,
    )

    assert payload["recall_hit"] is False
    assert payload["post_run_actions"] == [
        {
            "name": "save_function",
            "arguments": {
                "run_id": "run-1",
                "agent_visible": True,
            },
        }
    ]


def test_recalled_run_is_not_registered_again() -> None:
    payload = _run_result(
        RunResult(
            True,
            function_id="order_beverage",
            actions_executed=3,
            detail={
                "trace": [],
                "done_reason": "finished",
                "function_resolution": {
                    "selected_function_id": "order_beverage",
                },
            },
        ),
        body={"run_id": "run-2", "goal": "order latte"},
        function=None,
    )

    assert payload["recall_hit"] is True
    assert payload["recalled_function_id"] == "order_beverage"
    assert "post_run_actions" not in payload
