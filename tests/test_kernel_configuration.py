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
        config=OmniFlowConfig(runtime=RuntimeSettings(checker_enabled=enabled),plugins=PluginSet(checker=checker)))
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
        config=OmniFlowConfig(runtime=RuntimeSettings(checker_enabled=False, planner_error_retries=budget)))
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
