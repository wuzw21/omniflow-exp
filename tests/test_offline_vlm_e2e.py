from __future__ import annotations

import asyncio
import json

from omniflow.core.model import Action, ActionResult, Observation
from omniflow.runtime.engine import OmniFlow
from omniflow.runtime import core
from omniflow.vlm.planner import VLMPlanner


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


def test_offline_vlm_task_runs_unknown_tool_recovery_action_and_completion(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    responses = iter(
        [
            _response("tools_search", {"query": "手机操作"}),
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
    assert len(requests) == 3
    assert all(
        "tools_search"
        not in {
            tool["function"]["name"] for tool in request["tools"]
        }
        for request in requests
    )
    assert (
        result.detail["planner_diagnostics"]["rejected_tool_calls"][0]["tool"]
        == "tools_search"
    )
