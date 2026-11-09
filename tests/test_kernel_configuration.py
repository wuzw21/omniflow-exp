import json
import os
from pathlib import Path
import subprocess
import sys
import asyncio

import pytest
from omniflow.core.config import OmniFlowConfig, PluginSet, RuntimeSettings
from omniflow.core.model import Action, ActionResult, Function, FunctionStep, Observation, ToolCall
from omniflow.functions.store import FunctionStore
from omniflow.runtime.engine import OmniFlow
from src.integrations.gui_agent_tools import GuiAgentToolRuntime


def test_standalone_kernel_and_mcp_do_not_load_benchmark_configuration(tmp_path):
    env = {**os.environ, 'OMNIFLOW_ANDROIDWORLD_CONFIG': str(tmp_path/'absent.json')}
    result = subprocess.run([sys.executable, '-c',
        'from omniflow import OmniFlow; from src.integrations.gui_agent_mcp import create_server; '
        'from omniflow.core.config import RuntimeSettings; assert RuntimeSettings().max_steps == 30'],
        env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_androidworld_harness_still_reads_explicit_protocol(tmp_path):
    payload = json.loads((Path(__file__).parents[1]/'config/paper_androidworld.json').read_text())
    payload['protocol']['max_steps'] = 17
    config = tmp_path/'experiment.json'
    config.write_text(json.dumps(payload))
    result = subprocess.run([sys.executable, '-c',
        'from src.experiment.protocol import MAX_STEPS; assert MAX_STEPS == 17'],
        env={**os.environ, 'OMNIFLOW_ANDROIDWORLD_CONFIG': str(config)},
        capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('enabled', [True, False])
def test_checker_switch_preserves_function_execution_and_official_checker(tmp_path, enabled):
    checked, completed, actions = [], [], []
    class Host:
        async def observe(self, **kwargs):
            return Observation(xml='<hierarchy/>')
        async def act(self, action):
            actions.append(action)
            return ActionResult(True)
    def checker(context):
        checked.append(context)
    def completion():
        completed.append(True)
        return 1
    store = FunctionStore(tmp_path/'store.json')
    store.put_function(Function('pause', 'Pause', 'Wait',
        (FunctionStep(0, Action('wait', {'duration_ms': 1}), 'source'),),
        schema_version='omniflow.function.v2',
        input_schema={'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False}))
    host = Host()
    flow = OmniFlow(store.path, host=host, completion_checker=completion,
        config=OmniFlowConfig(runtime=RuntimeSettings(completion_oracle_enabled=True, checker_enabled=enabled),plugins=PluginSet(checker=checker)))
    runtime = GuiAgentToolRuntime(host=host, flow=flow)
    result = asyncio.run(runtime.call_function_tool('omniflow_execute', {
        'session_id': runtime.session_id, 'request_id': 'one', 'function_id': 'pause', 'arguments': {}}))
    assert result.success and len(actions) == 1 and completed == [True]
    assert bool(checked) == enabled
    assert result.output['detail']['runtime_policy']['checker_enabled'] == enabled
    timing = result.output['detail']['wall_accounting']
    assert timing['covered_wall_ms'] > 0
    assert timing['accounted_wall_ms'] == pytest.approx(timing['covered_wall_ms'])
    assert timing['components']['host.act']['calls'] == 1
    assert result.output['feedback']['task']['status'] == 'verified_success'
    if not enabled:
        assert flow.checker_library.rules == () and flow.plugins.checker is None


@pytest.mark.parametrize('budget,error_type,always_fail,calls,success', [
    (0, TimeoutError, False, 1, False),
    (1, TimeoutError, False, 2, True),
    (2, TimeoutError, True, 3, False),
    (1, ValueError, False, 1, False),
])
def test_native_request_retry_is_bounded_and_refreshes_observation(
        tmp_path, budget, error_type, always_fail, calls, success):
    observations, requests = [], []
    class Host:
        async def observe(self, **kwargs):
            value = Observation(xml=f'<page n="{len(observations)}"/>', image_base64='present')
            observations.append(value)
            return value
        async def act(self, action):
            raise AssertionError('failed model calls must not dispatch actions')
    class Planner:
        async def one_step_tool_call(self, goal, observation, *args):
            requests.append(observation.xml)
            if always_fail or len(requests) == 1:
                raise error_type('model probe')
            return ToolCall('finished', {'content': 'done'})
    store = FunctionStore(tmp_path/'empty.json');store.save()
    flow = OmniFlow(store.path, host=Host(), planner=Planner(), completion_checker=lambda: True,
        config=OmniFlowConfig(runtime=RuntimeSettings(completion_oracle_enabled=True, checker_enabled=False, planner_error_retries=budget)))
    result = asyncio.run(flow.arun('goal'))
    assert result.success == success and len(requests) == calls
    assert len(set(requests)) == calls
    diagnostics = result.detail['planner_diagnostics']
    assert diagnostics['request_retry_budget'] == budget
    assert len(diagnostics['request_retries']) == calls - 1
    assert result.actions_executed == 0 and result.model_calls == calls


@pytest.mark.parametrize('budget', [-1, 4, True, 1.5])
def test_request_retry_policy_rejects_unbounded_or_ambiguous_budgets(budget):
    with pytest.raises(ValueError, match='planner_error_retries'):
        RuntimeSettings(planner_error_retries=budget)


def test_kernel_default_retries_are_nonzero_so_transport_noise_is_not_a_policy_failure():
    # Every real failure in data/androidworld/.archive was a request timeout that
    # ended the episode because the default budget was 0.
    assert RuntimeSettings().planner_error_retries >= 1


@pytest.mark.parametrize('value', [0, -1.0, float('inf'), float('nan'), 1800.1])
def test_execution_deadline_is_bounded_and_finite(value):
    with pytest.raises(ValueError, match='execution_timeout_seconds'):
        RuntimeSettings(execution_timeout_seconds=value)


def test_kernel_invocation_uses_the_configured_deadline_not_its_own_default(tmp_path):
    # ExecutionControl used to be built with its own 600s default, so raising the
    # protocol deadline had no effect on when an episode was sealed.
    seen = []
    class Host:
        async def observe(self, **kwargs):
            return Observation(xml='<page/>', image_base64='present')
        async def act(self, action):
            raise AssertionError('no action expected')
    class Planner:
        async def one_step_tool_call(self, goal, observation, *args):
            return ToolCall('finished', {'content': 'done'})
    store = FunctionStore(tmp_path / 'empty.json')
    store.save()
    flow = OmniFlow(
        store.path, host=Host(), planner=Planner(), completion_checker=lambda: True,
        config=OmniFlowConfig(runtime=RuntimeSettings(
            completion_oracle_enabled=True, checker_enabled=False,
            execution_timeout_seconds=1200.0,
        )),
    )
    # ``engine.py`` imports the owner directly, so patch the symbol at the
    # lookup site used by ``OmniFlow._controlled``.
    from omniflow.runtime import engine as engine_module
    original = engine_module.ExecutionControl

    class Recording(original):
        def __post_init__(self):
            seen.append(self.timeout_seconds)
            super().__post_init__()

    engine_module.ExecutionControl = Recording
    try:
        asyncio.run(flow.arun('goal'))
    finally:
        engine_module.ExecutionControl = original
    assert seen == [1200.0]


def test_androidworld_protocol_budget_is_reachable_and_shared():
    from src.experiment import protocol

    assert protocol.TASK_DEADLINE_SEC == 600
    assert protocol.EPISODE_TIMEOUT_SEC <= protocol.TASK_DEADLINE_SEC
    assert protocol.EPISODE_TIMEOUT_SEC == protocol.TASK_DEADLINE_SEC
    assert protocol.PLANNER_TIMEOUT_SEC == 30.0
    assert protocol.VALIDATOR_FLUSH_GRACE_SEC <= protocol.TASK_DEADLINE_SEC
    # Retries are a protocol constant, so no baseline row can carry a different
    # error tolerance than the OmniFlow row it is compared against.
    assert isinstance(protocol.PLANNER_ERROR_RETRIES, int)
    assert 0 <= protocol.PLANNER_ERROR_RETRIES <= 3
