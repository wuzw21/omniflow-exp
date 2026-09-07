import asyncio
from dataclasses import replace

import pytest

from omniflow.core.config import OmniFlowConfig, PluginSet, RuntimeSettings
from omniflow.core.model import (
    Action,
    ActionResult,
    Function,
    FunctionStep,
    Observation,
    ToolCall,
    TransferResult,
)
from omniflow.functions.store import FunctionStore
from omniflow.runtime.control import ExecutionControl
from omniflow.runtime.engine import OmniFlow
from src.integrations.gui_agent_tools import GuiAgentToolRuntime


class Host:
    def __init__(self):
        self.actions = []

    async def observe(self, **kwargs):
        return Observation(xml='<hierarchy width="720" height="1280" />', extra={'display': {'width': 720, 'height': 1280}})

    async def get_state(self, state_id):
        return await self.observe()

    async def act(self, action):
        self.actions.append(Action.from_value(action))
        return ActionResult(True)


def flow_with_function(tmp_path, planner):
    store = FunctionStore(tmp_path/'store.json')
    store.put_function(Function(
        'example', 'Example', 'A two-step operation',
        (FunctionStep(0, Action('wait', {'duration_ms': 1}), 'first'),
         FunctionStep(1, Action('click', {'x': 500, 'y': 500}), 'second')),
        schema_version='omniflow.function.v2',
        input_schema={'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False},
    ))
    return OmniFlow(store.path, host=Host(), planner=planner, config=OmniFlowConfig(
        runtime=RuntimeSettings(max_steps=3),
        plugins=PluginSet(checker=lambda ctx: None, transfer=lambda *args: TransferResult(None, reason='transfer_unreliable')),
    ))


def test_external_function_yields_partial_progress_without_nested_planner(tmp_path):
    class ForbiddenPlanner:
        async def one_step_tool_call(self, *args, **kwargs):
            raise AssertionError('External tool invocation entered the Planner')
    flow = flow_with_function(tmp_path, ForbiddenPlanner())
    result = asyncio.run(flow.acall_tool(ToolCall('example', {})))
    assert not result.success
    assert result.actions_executed == 1
    assert [action.tool for action in flow.host.actions] == ['wait']
    assert result.final_state.xml
    feedback = result.detail['feedback']
    assert feedback['execution']['failed_step_index'] == 1
    assert feedback['execution']['failed_action_dispatched'] is False
    assert feedback['task']['status'] == 'unknown'
    assert feedback['control']['next'] == 'host'
    assert feedback['control']['automatic_retry'] is False
    import json
    from pathlib import Path

    from jsonschema import validate
    validate(feedback, json.loads((Path(__file__).parents[1]/'schemas/omniflow_invocation.v1.json').read_text()))


def test_invalid_function_result_cannot_be_reported_as_success(tmp_path):
    flow = flow_with_function(tmp_path, None)
    async def invalid(*args, **kwargs):
        return None
    flow.acall_tool = invalid
    runtime = GuiAgentToolRuntime(host=flow.host, flow=flow)
    result = runtime.call_tool_sync('example', {})
    assert result.success is False
    assert result.error == 'gui_agent_function_result_invalid'


def test_verifier_error_stops_without_corrective_device_action(tmp_path):
    class Planner:
        async def one_step_tool_call(self, *args, **kwargs):
            return ToolCall('finished', {'content': 'The requested operation is complete.'})
    flow = flow_with_function(tmp_path, Planner())
    def broken_verifier():
        raise OSError('verifier unavailable')
    flow.set_completion_checker(broken_verifier)
    result = asyncio.run(flow.arun('Complete the task'))
    assert not result.success
    assert result.detail['done_reason'] == 'verifier_error'
    assert result.detail['feedback']['task']['status'] == 'verifier_error'
    assert flow.host.actions == []


def test_primitive_only_finish_is_checked_and_does_not_dispatch_status(tmp_path):
    class Planner:
        async def one_step_tool_call(self, *args, **kwargs):
            return ToolCall('finished', {'content': 'Verified task result.'})
    flow = flow_with_function(tmp_path, Planner())
    calls = []
    flow.set_completion_checker(lambda: calls.append(True) or 1)
    result = asyncio.run(flow.arun('Complete the task'))
    assert result.success and len(calls) == 1
    assert result.detail['feedback']['task']['status'] == 'verified_success'
    assert result.detail['feedback']['control']['next'] == 'stop'
    assert flow.host.actions == []


def test_cancel_drains_inflight_action_and_forbids_next_function_step(tmp_path):
    async def scenario():
        flow = flow_with_function(tmp_path, None)
        entered, release = asyncio.Event(), asyncio.Event()
        async def slow_action(action):
            flow.host.actions.append(action)
            entered.set()
            await release.wait()
            return ActionResult(True)
        flow.host.act = slow_action
        runtime = GuiAgentToolRuntime(host=flow.host, flow=flow)
        task = asyncio.create_task(runtime.call_tool('example', {}))
        await entered.wait()
        assert runtime.cancel()['in_flight']
        with pytest.raises(ValueError, match='session_active'):
            runtime.start_session()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()  # repeated cancellation must not release an active device
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        result = await task
        assert result.error == 'cancelled'
        assert len(flow.host.actions) == 1
        assert result.output['feedback']['execution']['actions_executed'] == 1
        with pytest.raises(ValueError, match='session_closed'):
            await runtime.call_tool('wait', {'duration_ms': 1})
    asyncio.run(scenario())


def test_external_request_retry_never_repeats_action_and_reconnect_is_explicit(tmp_path):
    async def scenario():
        flow = flow_with_function(tmp_path, None)
        runtime = GuiAgentToolRuntime(host=flow.host, flow=flow)
        request = dict(session_id=runtime.session_id, request_id='operation-1',
                       tool_name='wait', arguments={'duration_ms': 1})
        result = await runtime.execute_request(**request)
        assert result.success
        assert await runtime.execute_request(**request) is result
        assert len(flow.host.actions) == 1
        with pytest.raises(ValueError, match='request_id_conflict'):
            await runtime.execute_request(**{**request, 'arguments': {'duration_ms': 2}})
        runtime.cancel()
        runtime.start_session()
        with pytest.raises(ValueError, match='session_mismatch'):
            await runtime.execute_request(**request)
    asyncio.run(scenario())


def test_shared_session_deadline_and_finish_close_dispatch(tmp_path):
    flow = flow_with_function(tmp_path, None)
    runtime = GuiAgentToolRuntime(host=flow.host, flow=flow)
    clock = [0.0]
    runtime._control = ExecutionControl(10, clock=lambda: clock[0])
    assert runtime.call_tool_sync('wait', {'duration_ms': 1}).success
    clock[0] = 11
    result = runtime.call_tool_sync('wait', {'duration_ms': 1})
    assert result.error == 'deadline_exceeded'
    assert len(flow.host.actions) == 1
    runtime.start_session()
    flow.set_completion_checker(lambda: True)
    assert asyncio.run(runtime.finish('Done'))['task_status'] == 'verified_success'
    with pytest.raises(ValueError, match='session_closed'):
        runtime.call_tool_sync('wait', {'duration_ms': 1})


def test_unknown_action_effect_stops_instead_of_retrying(tmp_path):
    flow = flow_with_function(tmp_path, None)
    async def rejected(action):
        flow.host.actions.append(action)
        return ActionResult(False, error='transport_lost_after_dispatch')
    flow.host.act = rejected
    runtime = GuiAgentToolRuntime(host=flow.host, flow=flow)
    result = runtime.call_tool_sync('wait', {'duration_ms': 1})
    assert result.output['feedback']['control']['reason'] == 'effect_unknown'
    with pytest.raises(ValueError, match='session_closed'):
        runtime.call_tool_sync('wait', {'duration_ms': 1})
    assert len(flow.host.actions) == 1


def test_repeated_planner_decision_on_unchanged_state_is_bounded(tmp_path):
    class Planner:
        async def one_step_tool_call(self, *args, **kwargs):
            return ToolCall('wait', {'duration_ms': 1})
    flow = flow_with_function(tmp_path, Planner())
    flow.config = replace(flow.config, runtime=RuntimeSettings(max_steps=20))
    result = asyncio.run(flow.arun('Make progress'))
    assert result.detail['done_reason'] == 'no_progress'
    assert len(flow.host.actions) == 3


def test_sync_planner_is_interruptible_and_busy_flow_cannot_dispatch(tmp_path):
    from threading import Event
    entered, release = Event(), Event()
    class Planner:
        def one_step_tool_call(self, *args, **kwargs):
            entered.set()
            assert release.wait(timeout=3)
            return ToolCall('wait', {'duration_ms': 1})
    async def scenario():
        flow = flow_with_function(tmp_path, Planner())
        task = asyncio.create_task(flow.arun('Do the task'))
        assert await asyncio.to_thread(entered.wait, 2)
        busy = await flow.acall_tool(ToolCall('wait', {'duration_ms': 1}))
        assert busy.error == 'execution_busy' and 'feedback' in busy.detail
        assert flow.cancel()
        release.set()
        result = await task
        assert result.detail['done_reason'] == 'cancelled'
        assert not flow.host.actions
    asyncio.run(scenario())
