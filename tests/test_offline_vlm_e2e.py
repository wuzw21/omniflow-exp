from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from omniflow.core.model import Action, ActionResult, Function, Observation, ToolCall
from omniflow.runtime import core
from omniflow.runtime.engine import OmniFlow
from omniflow.vlm.planner import VLMPlanner, normalize_openai_model_turn_response


def test_shared_planner_exposes_and_fills_function_input_schema() -> None:
    function = Function.from_dict(
        {
            "schema_version": "omniflow.function.v2",
            "function_id": "enter_product",
            "name": "Enter product",
            "description": "Enter the requested product.",
            "input_schema": {
                "type": "object",
                "properties": {"product": {"type": "string"}},
                "required": ["product"],
                "additionalProperties": False,
            },
            "bindings": [],
            "steps": [],
            "agent_visible": True,
        }
    )
    requests: list[dict[str, object]] = []

    def transport(envelope: dict[str, object]) -> dict[str, object]:
        request = envelope["request"]
        assert isinstance(request, dict)
        requests.append(request)
        return _response("enter_product", {"product": "37"})

    planned = asyncio.run(
        VLMPlanner(
            model="scene.vlm.operation.primary",
            transport=transport,
        ).one_step_tool_call(
            "Enter 37",
            Observation(extra={"display": {"width": 720, "height": 1280}}),
            (function,),
        )
    )

    assert planned == ToolCall("enter_product", {"product": "37"})
    function_tool = next(
        tool
        for tool in requests[0]["tools"]
        if tool["function"]["name"] == "enter_product"
    )
    assert function_tool["function"]["parameters"]["required"] == ["product"]


def test_shared_planner_accepts_tool_call_without_optional_summary() -> None:
    def transport(envelope: dict[str, object]) -> dict[str, object]:
        return {
            "requested_model": "test-model",
            "resolved_model": "test-model",
            "tool_calls": [
                {
                    "function": {
                        "name": "click",
                        "arguments": '{"x":120,"y":240}',
                    }
                }
            ],
        }

    planned = asyncio.run(
        VLMPlanner(model="test-model", transport=transport).one_step_tool_call(
            "Click the visible control",
            Observation(extra={"display": {"width": 720, "height": 1280}}),
        )
    )

    assert planned.name == "click"
    assert planned.arguments["x"] == pytest.approx(166.6666666667)
    assert planned.arguments["y"] == pytest.approx(187.5)


def test_openai_stream_is_normalized_to_shared_model_turn_response() -> None:
    chunks = [
        SimpleNamespace(
            model="resolved-model",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                function=SimpleNamespace(
                                    name="click",
                                    arguments='{"summary":"Tap","x":',
                                ),
                            )
                        ]
                    )
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            model="resolved-model",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                function=SimpleNamespace(
                                    name=None,
                                    arguments='150,"y":100}',
                                ),
                            )
                        ]
                    )
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=20,
                completion_tokens=5,
                total_tokens=25,
            ),
        ),
    ]

    normalized = normalize_openai_model_turn_response(
        chunks,
        requested_model="requested-model",
    )

    assert normalized == {
        "requested_model": "requested-model",
        "resolved_model": "resolved-model",
        "tool_calls": [
            {
                "function": {
                    "name": "click",
                    "arguments": '{"summary":"Tap","x":150,"y":100}',
                }
            }
        ],
        "reasoning": "",
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 5,
            "total_tokens": 25,
        },
    }


class OfflineVlmHost:
    def __init__(self) -> None:
        self.actions: list[Action] = []

    def observe(self, **_kwargs: object) -> Observation:
        xml = (
            '<hierarchy><node text="Settings" clickable="true" '
            'bounds="[0,0][300,200]" /></hierarchy>'
            if not self.actions
            else '<hierarchy><node text="Settings" /></hierarchy>'
        )
        return Observation(
            xml=xml,
            package_name="com.example.launcher",
            activity_name="MainActivity",
            extra={
                "state_id": f"state-{len(self.actions)}",
                "display": {"width": 1080, "height": 2400},
            },
        )

    def act(self, action: Action) -> ActionResult:
        self.actions.append(action)
        return ActionResult(True)


def _response(tool: str, arguments: dict[str, object]) -> dict[str, object]:
    arguments = {"summary": f"Use {tool}", **arguments}
    return {
        "requested_model": "scene.vlm.operation.primary",
        "resolved_model": "scene.vlm.operation.primary",
        "tool_calls": [
            {
                "function": {
                    "name": tool,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                }
            }
        ],
    }


def test_offline_vlm_task_runs_action_and_completion_without_retry(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    responses = iter(
        [
            _response(
                "click",
                {
                    "target_description": "Settings",
                    "x": 150,
                    "y": 100,
                },
            ),
            _response("finished", {"content": "已打开设置"}),
        ]
    )
    requests: list[dict[str, object]] = []

    def transport(envelope: dict[str, object]) -> dict[str, object]:
        request = envelope["request"]
        assert isinstance(request, dict)
        requests.append(request)
        return next(responses)

    host = OfflineVlmHost()
    planner = VLMPlanner(
        model="scene.vlm.operation.primary",
        transport=transport,
    )
    result = OmniFlow(
        tmp_path / "store.json",
        host=host,
        planner=planner,
    ).run("打开设置")

    assert result.success is True
    assert result.actions_executed == 1
    assert [action.tool for action in host.actions] == ["click"]
    assert result.detail["done_reason"] == "finished"
    assert len(requests) == 2
    assert all(request["max_tokens"] == 512 for request in requests)
    assert all("max_completion_tokens" not in request for request in requests)
    assert all(request["reasoning_effort"] == "none" for request in requests)
    assert all(request["enable_thinking"] is False for request in requests)
    assert all(
        request["thinking"] == {"type": "disabled"} for request in requests
    )
    assert all(request["parallel_tool_calls"] is False for request in requests)
    assert all(
        "Task guidance:" in request["messages"][1]["content"][0]["text"]
        for request in requests
    )
    assert all(
        "tools_search"
        not in {
            tool["function"]["name"] for tool in request["tools"]
        }
        for request in requests
    )
    assert "planner_diagnostics" not in result.detail
