"""The benchmark validator is an evaluation oracle, never an in-loop signal by default."""

import asyncio

import pytest

from omniflow.core.config import OmniFlowConfig, PluginSet, RuntimeSettings
from omniflow.core.model import ActionResult, Observation, ToolCall, TransferResult
from omniflow.functions.store import FunctionStore
from omniflow.runtime.engine import OmniFlow


class Host:
    def __init__(self):
        self.actions = []

    async def observe(self, **kwargs):
        return Observation(
            xml='<hierarchy width="720" height="1280" />',
            extra={'display': {'width': 720, 'height': 1280}},
        )

    async def get_state(self, state_id):
        return await self.observe()

    async def act(self, action):
        self.actions.append(action)
        return ActionResult(True)


class FinishingPlanner:
    def __init__(self):
        self.calls = 0

    async def one_step_tool_call(self, *args, **kwargs):
        self.calls += 1
        return ToolCall('finished', {'content': 'Done.'})


def _flow(tmp_path, *, oracle: bool, checker, planner=None):
    return OmniFlow(
        FunctionStore(tmp_path / 'store.json').path,
        host=Host(),
        planner=planner,
        completion_checker=checker,
        config=OmniFlowConfig(
            # No Store: keep these policy tests independent of the OmniTransfer checkout.
            runtime=RuntimeSettings(max_steps=3, completion_oracle_enabled=oracle,
                                    function_memory_enabled=False),
            plugins=PluginSet(
                checker=lambda ctx: None,
                transfer=lambda *args: TransferResult(None, reason='transfer_unreliable'),
            ),
        ),
    )


def test_oracle_is_disabled_by_default():
    assert RuntimeSettings().completion_oracle_enabled is False


def test_constructor_and_setter_drop_validator_unless_oracle_ablation(tmp_path):
    flow = _flow(tmp_path, oracle=False, checker=lambda: 1.0)
    assert flow.completion_checker is None
    flow.set_completion_checker(lambda: 1.0)
    assert flow.completion_checker is None

    ablation = _flow(tmp_path, oracle=True, checker=lambda: 1.0)
    assert callable(ablation.completion_checker)


def test_planner_finished_is_trusted_without_consulting_validator(tmp_path):
    calls = []
    planner = FinishingPlanner()
    flow = _flow(tmp_path, oracle=False, checker=lambda: calls.append(True) or 0.0, planner=planner)
    result = asyncio.run(flow.arun('Complete the task'))
    # A validator that would reject must not be able to steer the formal loop.
    assert calls == []
    assert result.success
    assert result.detail['done_reason'] == 'finished'
    assert result.detail['feedback']['task']['status'] == 'unknown'
    assert result.detail['runtime_policy']['completion_oracle_enabled'] is False


def test_oracle_ablation_still_consults_validator(tmp_path):
    calls = []
    flow = _flow(tmp_path, oracle=True, checker=lambda: calls.append(True) or 1.0,
                 planner=FinishingPlanner())
    result = asyncio.run(flow.arun('Complete the task'))
    assert calls == [True]
    assert result.detail['done_reason'] == 'function_completed_verified'
    assert result.detail['runtime_policy']['completion_oracle_enabled'] is True


@pytest.mark.parametrize('value', ['on', 1, None])
def test_oracle_flag_must_be_boolean(value):
    with pytest.raises(ValueError, match='completion_oracle_enabled_must_be_boolean'):
        RuntimeSettings(completion_oracle_enabled=value)
