import asyncio
import json
from pathlib import Path

import pytest

from omniflow.runtime.timing import TimingLedger, measure


def test_androidworld_common_step_owner_accounts_failures_and_reset():
    from types import SimpleNamespace
    from src.integrations.android_world.run_episode import _ExperimentAgentAdapter
    class Agent:
        def reset(self):
            pass
        def step(self, goal):
            with measure('host.test'):
                raise RuntimeError('failed step')
    adapter = _ExperimentAgentAdapter(Agent(), recording_session=SimpleNamespace(
        env=SimpleNamespace(), start_episode=lambda: None))
    with pytest.raises(RuntimeError, match='failed step'):
        adapter.step('goal')
    report = adapter.execution_timing.report()
    assert report['components']['execution.other']['failed_calls'] == 1
    assert report['components']['host.test']['calls'] == 1
    assert 0 < report['covered_wall_ms'] <= adapter.execution_duration_ms
    adapter.reset()
    assert adapter.execution_timing.report()['covered_wall_ms'] == 0


def test_nested_wall_time_reconciles_without_double_counting():
    clock = [0]
    ledger = TimingLedger(clock=lambda: clock[0] * 1_000_000)
    with ledger.span('execution.other'):
        clock[0] = 2
        with measure('transfer'):
            clock[0] = 7
        clock[0] = 10
    report = ledger.report()
    from jsonschema import validate
    validate(report, json.loads((Path(__file__).parents[1]/'schemas/omniflow_wall_accounting.v1.json').read_text()))
    assert report['covered_wall_ms'] == report['accounted_wall_ms'] == 10
    assert report['components']['transfer']['exclusive_ms'] == 5
    assert report['components']['execution.other']['exclusive_ms'] == 5


def test_lifecycle_and_execution_are_separate_accounting_scopes():
    clock = [0]
    lifecycle = TimingLedger(clock=lambda: clock[0] * 1_000_000)
    execution = TimingLedger(clock=lambda: clock[0] * 1_000_000)
    with lifecycle.span('lifecycle.other'):
        with measure('setup'):
            clock[0] = 2
        with measure('lifecycle.execution'):
            with execution.span('execution.other'):
                with measure('completion_checker'):
                    clock[0] = 5
                clock[0] = 7
        with measure('official_validator'):
            clock[0] = 9
        clock[0] = 10
    report = lifecycle.report(owner_wall_ms=10.1)
    assert report['covered_wall_ms'] == report['accounted_wall_ms'] == 10
    assert report['owner_delta_ms'] == pytest.approx(.1)
    assert report['components']['lifecycle.execution']['exclusive_ms'] == 5
    assert 'completion_checker' not in report['components']
    assert execution.report()['covered_wall_ms'] == 5


def test_parallel_tasks_and_cancelled_span_remain_accounted():
    async def scenario():
        clock = [0]
        ledger = TimingLedger(clock=lambda: clock[0] * 1_000_000)
        entered, finish = asyncio.Event(), asyncio.Event()
        async def work():
            with measure('device'):
                entered.set()
                await finish.wait()
        with ledger.span('execution.other'):
            task = asyncio.create_task(work())
            await entered.wait()
            clock[0] = 2
            with measure('logging'):
                clock[0] = 5
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            clock[0] = 6
        report = ledger.report()
        assert report['covered_wall_ms'] == report['accounted_wall_ms'] == 6
        assert report['components']['concurrent']['exclusive_ms'] == 3
        assert report['components']['device']['failed_calls'] == 1
    asyncio.run(scenario())


def test_async_cached_observation_is_awaited_and_included_in_post_action_timing():
    from omniflow.runtime.core import execute_action
    from omniflow.core.config import PluginSet
    from omniflow.core.model import Action, ActionResult, Observation
    clock = [0]
    ledger = TimingLedger(clock=lambda: clock[0] * 1_000_000)
    after = Observation(xml='<after/>')
    class Host:
        async def act(self, action):
            return ActionResult(True)
        async def take_after_action_observation(self):
            clock[0] = 5
            return after
        async def observe(self, **kwargs):
            raise AssertionError('cached state must avoid another observe')
    with ledger.span('execution.other'):
        result = asyncio.run(execute_action(Action('wait', {'duration_ms': 1}),
            observation=Observation(xml='<before/>'), host=Host(), plugins=PluginSet()))
    assert result.after is after
    assert ledger.report()['components']['observe.post_action']['exclusive_ms'] == 5


@pytest.mark.parametrize('fails', [False, True])
def test_model_stream_first_event_and_later_processing_are_distinct(fails):
    from omniflow.vlm.planner import normalize_openai_model_turn_response
    clock = [0]
    ledger = TimingLedger(clock=lambda: clock[0] * 1_000_000)
    def stream():
        clock[0] = 5
        if fails:
            raise TimeoutError('first event timeout')
        yield {'model': 'test', 'choices': []}
        clock[0] = 9
    with ledger.span('model.planner'):
        if fails:
            with pytest.raises(TimeoutError):
                normalize_openai_model_turn_response(stream(), requested_model='test')
        else:
            assert normalize_openai_model_turn_response(stream(), requested_model='test')['tool_calls'] == []
    report = ledger.report()
    components = report['components']
    assert components['model.planner.first_stream_event']['exclusive_ms'] == 5
    assert components['model.planner.first_stream_event']['failed_calls'] == int(fails)
    assert components['model.planner.stream']['exclusive_ms'] == (0 if fails else 4)
    assert report['accounted_wall_ms'] == report['covered_wall_ms']
