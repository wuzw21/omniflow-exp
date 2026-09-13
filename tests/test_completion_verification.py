from __future__ import annotations

import asyncio

import pytest

from omniflow.core.config import OmniFlowConfig, RuntimeSettings
from omniflow.core.model import ActionResult, Observation, ToolCall
from omniflow.functions.artifact import parse_function_artifact
from omniflow.functions.store import FunctionStore
from omniflow.runtime.engine import OmniFlow
from src.integrations.android_world.agent import (
    _emit_official_completion,
    _install_official_completion_checker,
)
from src.integrations.android_world.run_episode import _raw_replay_action_to_payload


def _recovery_scenario(tmp_path, monkeypatch, *, replacement=False, still_blocked=False,
                       checker_accepts=True, runtime=None):
    """Deterministic control-flow regression; never a device/Transfer accuracy result."""
    from omniflow.core.config import PluginSet
    from omniflow.core.model import Action, Function, FunctionStep, TransferResult
    from omniflow.functions.recall import RecallResult

    class Host:
        def __init__(self):
            self.actions = []
            self.recovered = False
            self.done = False

        async def observe(self, **kwargs):
            return Observation(xml=f'<page recovered="{self.recovered}" done="{self.done}"/>',
                               image_base64='test-image',
                               extra={'display': {'width': 1000, 'height': 1000}})

        async def get_state(self, state_id):
            return Observation(xml='<source/>', extra={'display': {'width': 1000, 'height': 1000}})

        async def act(self, action):
            self.actions.append(action)
            if action.tool == 'press_key':
                self.recovered = True
            if action.tool == 'click':
                assert action.args['x'] == 800  # never dispatch source x=100
                self.done = True
            return ActionResult(True)

    host = Host()
    def transfer(action, target, source):
        if action.tool != 'click':
            return TransferResult(action)
        if not host.recovered or still_blocked:
            return TransferResult(None, reason='test_layout_unresolved')
        return TransferResult(Action('click', {'x': 800, 'y': 800}))

    store = FunctionStore(tmp_path / 'store.json')
    for function_id, prefix in [('original', True), ('replacement', False)]:
        steps = ([FunctionStep(0, Action('wait', {'duration_ms': 1}), 's0')] if prefix else [])
        steps.append(FunctionStep(len(steps), Action('click', {'x': 100, 'y': 100}), 's1'))
        store.put_function(Function(function_id, function_id, 'Finish the operation', tuple(steps),
                                    schema_version='omniflow.function.v2',
                                    input_schema={'type': 'object', 'properties': {},
                                                  'required': [], 'additionalProperties': False}))

    class Planner:
        def __init__(self):
            self.catalogs = []
            self.calls = 0

        async def one_step_tool_call(self, goal, observation, functions, *args):
            self.catalogs.append([f.id for f in functions])
            self.calls += 1
            if self.calls == 1:
                return ToolCall('original', {})
            if self.calls == 2:
                return ToolCall('press_key', {'key': 'back'})
            if self.calls == 3:
                if not functions:
                    return ToolCall('click', {'x': 800, 'y': 800})
                return ToolCall('replacement' if replacement else 'original', {})
            return ToolCall('finished', {'content': 'The current result is ready.'})

    planner = Planner()
    flow = OmniFlow(store.path, host=host, planner=planner,
                    completion_checker=lambda: checker_accepts and host.done,
                    config=OmniFlowConfig(runtime=runtime or RuntimeSettings(max_steps=4, checker_enabled=False),
                                         plugins=PluginSet(transfer=transfer)))
    async def recall(*args, **kwargs):
        return RecallResult((), {'reason': 'test_planner_selection'})
    monkeypatch.setattr(flow, '_recall', recall)
    return flow, host, planner


@pytest.mark.parametrize('replacement', [False, True], ids=['resume_failed_step', 'select_other_function'])
def test_recovery_preserves_prefix_and_records_actual_reuse(tmp_path, monkeypatch, replacement):
    import json
    from pathlib import Path
    from jsonschema import validate
    flow, host, planner = _recovery_scenario(tmp_path, monkeypatch, replacement=replacement)
    result = asyncio.run(flow.arun('Finish the operation'))
    assert result.success and result.detail['done_reason'] == 'function_completed_verified'
    assert [a.tool for a in host.actions] == ['wait', 'press_key', 'click']
    assert planner.calls == 3
    evidence = result.detail['function_resume']
    assert evidence['attempt_count'] == evidence['success_count'] == int(not replacement)
    assert [e['success'] for e in evidence['events']] == [False, True]
    assert evidence['events'][0]['failed_action_dispatched'] is False
    assert evidence['events'][1]['after_failure'] is True
    assert evidence['events'][1]['start_step_index'] == int(not replacement)
    schema = json.loads((Path(__file__).parents[1] / 'schemas/oob/omniflow_run_log.v1.json').read_text())
    validate({'function_resume': evidence}, schema['properties']['diagnostics'])


def test_failed_resume_stays_failed_and_never_dispatches_source_point(tmp_path, monkeypatch):
    flow, host, _ = _recovery_scenario(tmp_path, monkeypatch, still_blocked=True)
    result = asyncio.run(flow.arun('Finish the operation'))
    assert not result.success
    assert [a.tool for a in host.actions] == ['wait', 'press_key']
    evidence = result.detail['function_resume']
    assert evidence['attempt_count'] == 1 and evidence['success_count'] == 0
    assert evidence['events'][-1]['failed_action_dispatched'] is False


def test_successful_resume_does_not_override_official_rejection(tmp_path, monkeypatch):
    flow, host, _ = _recovery_scenario(tmp_path, monkeypatch, checker_accepts=False)
    result = asyncio.run(flow.arun('Finish the operation'))
    assert host.done and not result.success
    assert result.detail['function_resume']['success_count'] == 1


def test_missing_resume_evidence_remains_unknown():
    from src.experiment.run_task import _execution_audit
    audit = _execution_audit({})
    assert audit['resume_attempts'] is None and audit['resume_success'] is None
    audit = _execution_audit({'function_resume': {'attempt_count': 0, 'success_count': 0}})
    assert audit['resume_attempts'] == audit['resume_success'] == 0


@pytest.mark.parametrize('enabled', [True, False], ids=['adaptive_full', 'no_reentry'])
def test_reentry_ablation_keeps_same_planner_actions_and_completion_gate(tmp_path, monkeypatch, enabled):
    flow, host, planner = _recovery_scenario(tmp_path, monkeypatch,
        runtime=RuntimeSettings(max_steps=4, checker_enabled=False, function_reentry_enabled=enabled))
    result = asyncio.run(flow.arun('Finish the operation'))
    assert result.success
    assert [a.tool for a in host.actions] == ['wait', 'press_key', 'click']
    assert result.detail['runtime_policy']['function_reentry_enabled'] == enabled
    evidence = result.detail['function_resume']
    assert evidence['attempt_count'] == int(enabled)
    assert len(evidence['events']) == (2 if enabled else 1)
    assert bool(planner.catalogs[2]) == enabled


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


def test_androidworld_answer_populates_official_interaction_cache() -> None:
    class Environment:
        def __init__(self) -> None:
            self.actions = []

        def execute_action(self, action: object) -> None:
            self.actions.append(action)

    environment = Environment()

    _emit_official_completion(environment, "75")

    assert len(environment.actions) == 1
    action = environment.actions[0]
    assert action.action_type == "answer"
    assert action.text == "75"
    assert action.goal_status is None


def test_androidworld_installs_current_task_completion_checker() -> None:
    class Flow:
        def __init__(self) -> None:
            self.checker = None

        def set_completion_checker(self, checker: object) -> None:
            self.checker = checker

    class Task:
        def __init__(self) -> None:
            self.seen_env = None

        def is_successful(self, env: object) -> float:
            self.seen_env = env
            return 1.0

    flow = Flow()
    task = Task()
    env = object()

    _install_official_completion_checker(flow, task, env)  # type: ignore[arg-type]

    assert callable(flow.checker)
    assert flow.checker() == 1.0
    assert task.seen_env is env


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


def test_successful_function_stops_when_official_completion_checker_passes(
    tmp_path,
) -> None:
    class FunctionPlanner:
        def __init__(self) -> None:
            self.calls = 0

        async def one_step_tool_call(self, *_: object, **__: object) -> ToolCall:
            self.calls += 1
            if self.calls > 1:
                raise AssertionError("verified Function must bypass the Planner")
            return ToolCall("complete_source_workflow", {})

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

    class CompletionChecker:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self) -> float:
            self.calls += 1
            return 1.0

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
    checker = CompletionChecker()
    host = FunctionHost()
    flow = OmniFlow(
        store_path,
        host=host,
        planner=planner,
        completion_checker=checker,
        config=OmniFlowConfig(runtime=RuntimeSettings(max_steps=3)),
    )

    result = asyncio.run(flow.arun("Set brightness to the maximum."))

    assert result.success is True
    assert result.error is None
    assert result.detail["done_reason"] == "function_completed_verified"
    assert result.detail["completion_review_calls"] == 1
    assert result.detail["function_execution"]["task_completion_status"] == "verified"
    assert planner.calls == 1
    assert checker.calls == 1
    assert host.actions == 1


def test_rejected_function_completion_returns_checker_feedback_to_planner(
    tmp_path,
) -> None:
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
            if self.calls == 2:
                return ToolCall("click", {"x": 500, "y": 500})
            return ToolCall("finished", {"content": "done"})

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

    class CompletionChecker:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self) -> float:
            self.calls += 1
            return 0.0 if self.calls == 1 else 1.0

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
    checker = CompletionChecker()
    host = FunctionHost()
    flow = OmniFlow(
        store_path,
        host=host,
        planner=planner,
        completion_checker=checker,
        config=OmniFlowConfig(runtime=RuntimeSettings(max_steps=4)),
    )

    result = asyncio.run(flow.arun("Set brightness to the maximum."))

    assert result.success is True
    assert result.detail["done_reason"] == "function_completed_verified"
    assert result.detail["completion_review_calls"] == 2
    assert planner.calls == 3
    assert checker.calls == 2
    assert host.actions == 2
    planner_feedback = str(
        planner.observations[1].extra.get("planner_feedback") or ""
    )
    assert "official completion checker rejected" in planner_feedback
    assert planner.observations[1].extra["forbid_finished"] is True


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


def test_incomplete_local_function_can_route_to_second_local_function(
    tmp_path,
) -> None:
    class CompositionRouter:
        def __init__(self) -> None:
            self.calls = 0

        async def route_function(
            self,
            _goal: object,
            _functions: object,
        ) -> ToolCall:
            self.calls += 1
            return ToolCall(
                "first_local" if self.calls == 1 else "second_local",
                {},
            )

    class CompositionPlanner:
        async def one_step_tool_call(self, *_: object, **__: object) -> ToolCall:
            raise AssertionError(
                "the second page-matched local Function should use the Router"
            )

    class CompositionHost:
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

    class CompositionChecker:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self) -> float:
            self.calls += 1
            return 1.0 if self.calls == 2 else 0.0

    store_path = tmp_path / "store.json"
    store = FunctionStore(store_path)
    for function_id, name in (
        ("first_local", "Complete the first subtask"),
        ("second_local", "Complete the second subtask"),
    ):
        store.put_function(
            parse_function_artifact(
                {
                    "schema_version": "omniflow.function.v2",
                    "function_id": function_id,
                    "name": name,
                    "description": f"{name} for the combined task.",
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
                            "source_state_id": f"{function_id}-state",
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

    router = CompositionRouter()
    checker = CompositionChecker()
    host = CompositionHost()
    flow = OmniFlow(
        store_path,
        host=host,
        planner=CompositionPlanner(),
        function_router=router,
        completion_checker=checker,
        config=OmniFlowConfig(runtime=RuntimeSettings(max_steps=3)),
    )

    result = asyncio.run(flow.arun("Complete the first and second subtasks."))

    assert result.success is True
    assert result.detail["done_reason"] == "function_completed_verified"
    assert result.detail["completion_review_calls"] == 2
    assert result.fallback_steps == 0
    assert result.function_id == "second_local"
    assert router.calls == 2
    assert checker.calls == 2
    assert host.actions == 2


def test_router_does_not_repeat_same_function_from_same_gui_state(
    tmp_path,
) -> None:
    class RepeatingRouter:
        def __init__(self) -> None:
            self.calls = 0

        async def route_function(
            self,
            _goal: object,
            _functions: object,
        ) -> ToolCall:
            self.calls += 1
            return ToolCall("search_note", {})

    class FinishingPlanner:
        def __init__(self) -> None:
            self.calls = 0

        async def one_step_tool_call(self, *_: object, **__: object) -> ToolCall:
            self.calls += 1
            return ToolCall("finished", {"content": "The note was inspected."})

    class StaticStateHost:
        def __init__(self) -> None:
            self.actions = 0

        async def observe(self, **_: object) -> Observation:
            return Observation(
                xml='<hierarchy width="720" height="1280" />',
                extra={
                    "display": {"width": 720, "height": 1280},
                    "state_id": "same-search-page",
                },
            )

        async def get_state(self, _state_id: str) -> Observation:
            return await self.observe()

        async def act(self, _action: object) -> ActionResult:
            self.actions += 1
            return ActionResult(True)

    class CompletionChecker:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self) -> float:
            self.calls += 1
            return 1.0 if self.calls == 2 else 0.0

    store_path = tmp_path / "store.json"
    store = FunctionStore(store_path)
    store.put_function(
        parse_function_artifact(
            {
                "schema_version": "omniflow.function.v2",
                "function_id": "search_note",
                "name": "Search for a note",
                "description": "Search for the requested note title.",
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
                        "source_state_id": "search-state",
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

    router = RepeatingRouter()
    planner = FinishingPlanner()
    checker = CompletionChecker()
    host = StaticStateHost()
    flow = OmniFlow(
        store_path,
        host=host,
        planner=planner,
        function_router=router,
        completion_checker=checker,
        config=OmniFlowConfig(runtime=RuntimeSettings(max_steps=3)),
    )

    result = asyncio.run(flow.arun("Read the Team Weekly Sync note."))

    assert result.success is True
    assert result.detail["done_reason"] == "function_completed_verified"
    assert result.fallback_steps == 0
    assert router.calls == 2
    assert planner.calls == 1
    assert checker.calls == 2
    assert host.actions == 1
    events = result.detail["function_resolution"]["recall"]["events"]
    assert events[1]["function_cache"]["status"] == (
        "repeated_invocation_without_progress"
    )


@pytest.mark.parametrize("success_after, expected_success", [(2, True), (4, False)])
def test_router_progress_and_task_budget_are_independent(tmp_path, success_after, expected_success) -> None:
    class RepeatingRouter:
        def __init__(self) -> None:
            self.calls = 0

        async def route_function(
            self,
            _goal: object,
            _functions: object,
        ) -> ToolCall:
            self.calls += 1
            return ToolCall("delete_item", {})

    class UnusedPlanner:
        async def one_step_tool_call(self, *_: object, **__: object) -> ToolCall:
            raise AssertionError("changed GUI state should admit the Function again")

    class ChangingStateHost:
        def __init__(self) -> None:
            self.actions = 0

        async def observe(self, **_: object) -> Observation:
            return Observation(
                xml='<hierarchy width="720" height="1280" />',
                extra={
                    "display": {"width": 720, "height": 1280},
                    "state_id": f"item-list-{self.actions}",
                },
            )

        async def get_state(self, _state_id: str) -> Observation:
            return await self.observe()

        async def act(self, _action: object) -> ActionResult:
            self.actions += 1
            return ActionResult(True)

    class CompletionChecker:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self) -> float:
            self.calls += 1
            return 1.0 if self.calls == success_after else 0.0

    store_path = tmp_path / "store.json"
    store = FunctionStore(store_path)
    store.put_function(
        parse_function_artifact(
            {
                "schema_version": "omniflow.function.v2",
                "function_id": "delete_item",
                "name": "Delete one item",
                "description": "Delete the requested item from the visible list.",
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
                        "source_state_id": "item-list",
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

    router = RepeatingRouter()
    checker = CompletionChecker()
    host = ChangingStateHost()
    flow = OmniFlow(
        store_path,
        host=host,
        planner=UnusedPlanner(),
        function_router=router,
        completion_checker=checker,
        config=OmniFlowConfig(runtime=RuntimeSettings(max_steps=3)),
    )

    result = asyncio.run(flow.arun("Delete two items."))

    assert result.success is expected_success
    assert result.detail["done_reason"] == ("function_completed_verified" if expected_success else "step_budget_exceeded")
    assert result.fallback_steps == 0
    assert router.calls == min(success_after, 3)
    assert checker.calls == min(success_after, 3)
    assert host.actions == min(success_after, 3)
