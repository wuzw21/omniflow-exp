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
