from __future__ import annotations

import asyncio

from omniflow.core.config import OmniFlowConfig, RuntimeSettings
from omniflow.core.model import ActionResult, Observation, ToolCall
from omniflow.functions.artifact import parse_function_artifact
from omniflow.functions.store import FunctionStore
from omniflow.runtime.engine import OmniFlow
from src.integrations.android_world.agent import _emit_official_completion
from src.integrations.android_world.run_episode import _raw_replay_action_to_payload


def test_androidworld_completion_uses_official_status_action() -> None:
    class Environment:
        def __init__(self) -> None:
            self.actions = []

        def execute_action(self, action: object) -> None:
            self.actions.append(action)

    environment = Environment()

    _emit_official_completion(environment)

    assert len(environment.actions) == 1
    action = environment.actions[0]
    assert action.action_type == "status"
    assert action.goal_status == "complete"
    assert action.text is None


def test_raw_replay_finished_uses_official_status_without_answer() -> None:
    finished_payload, finished_error = _raw_replay_action_to_payload(
        {"action_type": "finished", "content": "done"},
        source_size=(720, 1280),
        target_size=(1440, 2560),
        resolution={},
    )
    answer_payload, answer_error = _raw_replay_action_to_payload(
        {"action_type": "answer", "args": {"text": "the answer"}},
        source_size=(720, 1280),
        target_size=(1440, 2560),
        resolution={},
    )

    assert finished_error is None
    assert finished_payload == {
        "action_type": "status",
        "goal_status": "complete",
    }
    assert answer_error is None
    assert answer_payload == {"action_type": "answer", "text": "the answer"}


class _Host:
    def __init__(self) -> None:
        self.observations = 0

    def observe(self, **_: object) -> Observation:
        self.observations += 1
        return Observation(
            xml='<hierarchy width="720" height="1280" />',
            extra={"display": {"width": 720, "height": 1280}},
        )

    def act(self, _action: object) -> ActionResult:
        return ActionResult(True)


class _Planner:
    def __init__(self) -> None:
        self.calls = 0

    async def one_step_tool_call(self, *_: object, **__: object) -> ToolCall:
        self.calls += 1
        return ToolCall("finished", {"content": "亮度已设置完成"})


def test_planner_finished_closes_without_internal_completion_check() -> None:
    planner = _Planner()
    host = _Host()

    store_path = "/tmp/omniflow-completion-verification-store.json"
    FunctionStore(store_path).save()
    flow = OmniFlow(
        store_path,
        host=host,
        planner=planner,
        config=OmniFlowConfig(runtime=RuntimeSettings(max_steps=3)),
    )

    result = asyncio.run(flow.arun("Turn brightness to the max value."))

    assert result.success is True
    assert result.error is None
    assert result.detail["done_reason"] == "finished"
    assert result.detail["completion_review_calls"] == 0
    assert planner.calls == 1


def test_successful_complete_function_returns_to_planner_for_finish(tmp_path) -> None:
    class FunctionPlanner:
        def __init__(self) -> None:
            self.calls = 0
            self.observations: list[Observation] = []

        async def one_step_tool_call(
            self,
            _goal: object,
            observation: Observation,
            *_: object,
            **__: object,
        ) -> ToolCall:
            self.calls += 1
            self.observations.append(observation)
            if self.calls == 1:
                return ToolCall("complete_source_workflow", {})
            return ToolCall("finished", {"content": "亮度已设置完成"})

    class FunctionHost:
        def __init__(self) -> None:
            self.actions = 0

        async def observe(self, **_: object) -> Observation:
            return Observation(
                xml='<hierarchy width="720" height="1280" />',
                extra={"display": {"width": 720, "height": 1280}},
            )

        async def get_state(self, _state_id: str) -> Observation:
            return await self.observe()

        async def act(self, _action: object) -> ActionResult:
            self.actions += 1
            return ActionResult(True)

    store_path = tmp_path / "store.json"
    store = FunctionStore(store_path)
    store.put_function(
        parse_function_artifact(
            {
                "schema_version": "omniflow.function.v2",
                "function_id": "complete_source_workflow",
                "name": "Set brightness",
                "description": "Set the requested brightness.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                "bindings": [],
                "render_bindings": [],
                "steps": [
                    {
                        "step_index": 0,
                        "source_state_id": "state-1",
                        "action": {
                            "tool": "wait",
                            "args": {"duration_ms": 1},
                        },
                    }
                ],
                "agent_visible": True,
            }
        )
    )
    planner = FunctionPlanner()
    host = FunctionHost()

    flow = OmniFlow(
        store_path,
        host=host,
        planner=planner,
        config=OmniFlowConfig(runtime=RuntimeSettings(max_steps=3)),
    )

    result = asyncio.run(flow.arun("Set brightness to the maximum."))

    assert result.success is True
    assert result.error is None
    assert result.detail["done_reason"] == "finished"
    assert result.detail["completion_review_calls"] == 0
    assert result.detail["function_execution"]["task_completion_status"] == "unverified"
    assert "completion_gate" not in result.detail["function_execution"]
    assert planner.calls == 2
    assert host.actions == 1
    action_history = str(
        planner.observations[1].extra.get("execution_history") or ""
    )
    assert planner.observations[1].extra.get("completion_only") is not True
    assert "Action: complete_source_workflow" in action_history
    assert "action_list:" in action_history
    assert "Last internal action outcome: executed=yes" in action_history
    assert '"state_changed":false' in action_history
    assert "[Function]" not in action_history


def test_successful_local_function_returns_to_planner_mainline(tmp_path) -> None:
    class LocalPlanner:
        def __init__(self) -> None:
            self.calls = 0

        async def one_step_tool_call(self, *_: object, **__: object) -> ToolCall:
            self.calls += 1
            if self.calls == 1:
                return ToolCall("state_transition_001", {})
            return ToolCall("finished", {"content": "后续主线已完成"})

    class LocalHost:
        def __init__(self) -> None:
            self.actions = 0

        async def observe(self, **_: object) -> Observation:
            return Observation(
                xml='<hierarchy width="720" height="1280" />',
                extra={"display": {"width": 720, "height": 1280}},
            )

        async def get_state(self, _state_id: str) -> Observation:
            return await self.observe()

        async def act(self, _action: object) -> ActionResult:
            self.actions += 1
            return ActionResult(True)

    store_path = tmp_path / "store.json"
    store = FunctionStore(store_path)
    store.put_function(
        parse_function_artifact(
            {
                "schema_version": "omniflow.function.v2",
                "function_id": "state_transition_001",
                "name": "Recorded transition",
                "description": "Perform one reusable transition.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                "bindings": [],
                "render_bindings": [],
                "steps": [
                    {
                        "step_index": 0,
                        "source_state_id": "state-1",
                        "action": {
                            "tool": "wait",
                            "args": {"duration_ms": 1},
                        },
                    }
                ],
                "agent_visible": True,
            }
        )
    )
    planner = LocalPlanner()
    host = LocalHost()
    flow = OmniFlow(
        store_path,
        host=host,
        planner=planner,
        config=OmniFlowConfig(runtime=RuntimeSettings(max_steps=3)),
    )

    result = asyncio.run(flow.arun("Finish the remaining task."))

    assert result.success is True
    assert result.detail["done_reason"] == "finished"
    assert planner.calls == 2
    assert host.actions == 1
