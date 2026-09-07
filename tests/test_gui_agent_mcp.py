import asyncio
import base64
from io import BytesIO
import json
from pathlib import Path
import sys

import pytest

pytest.importorskip('mcp')
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from omniflow.core.model import (
    Action,
    ActionResult,
    Function,
    FunctionStep,
    Observation,
)
from omniflow.functions.store import FunctionStore
from omniflow.runtime.engine import OmniFlow
from src.integrations.gui_agent_mcp import GuiAgentMcp, create_server, device_lease, runtime_socket
from src.integrations.gui_agent_tools import GuiAgentToolRuntime


def test_mcp_invocation_schema_and_retry_without_second_dispatch(tmp_path):
    from PIL import Image
    image = BytesIO()
    Image.new('RGB', (1440, 3120), 'white').save(image, format='PNG')
    class Host:
        def __init__(self):
            self.actions = []
        async def observe(self, **kwargs):
            return Observation(xml='<hierarchy/>', image_base64=base64.b64encode(image.getvalue()).decode())
        async def act(self, action):
            self.actions.append(action)
            return ActionResult(True)
    async def scenario():
        host = Host()
        store = FunctionStore(tmp_path/'store.json')
        store.put_function(Function('pause', 'Pause', 'Pause briefly',
            (FunctionStep(0, Action('wait', {'duration_ms': 1}), 'source'),),
            schema_version='omniflow.function.v2',
            input_schema={'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False}))
        runtime = GuiAgentToolRuntime(host=host, flow=OmniFlow(tmp_path/'store.json', host=host))
        transport = GuiAgentMcp(runtime)
        args = {'session_id': runtime.session_id, 'request_id': 'one', 'function_id': 'pause', 'arguments': {}}
        request = types.CallToolRequestParams(name='omniflow_execute', arguments=args)
        first = await transport.call_tool(None, request)
        second = await transport.call_tool(None, request)
        assert not first.is_error and first == second and len(host.actions) == 1
        body = json.loads(first.content[0].text)
        assert body['feedback']['observation']['image_base64'] is None
        wire = json.loads(first.model_dump_json(by_alias=True, exclude_unset=True))
        assert wire['content'][0]['type'] == 'text'
        assert wire['content'][1]['type'] == 'image'
        preview = Image.open(BytesIO(base64.b64decode(wire['content'][1]['data'])))
        assert max(preview.size) <= 1280 and wire['content'][1]['mimeType'] == 'image/jpeg'
        assert Image.open(BytesIO(image.getvalue())).size == (1440, 3120)
        from jsonschema import validate
        schema = json.loads((Path(__file__).parents[1]/'schemas/omniflow_invocation.v1.json').read_text())
        validate(body['feedback'], schema)
        invalid = await transport.call_tool(None, types.CallToolRequestParams(
            name='omniflow_execute', arguments={**args, 'request_id': 'two', 'arguments': {'duration_ms': 'bad'}}))
        assert invalid.is_error and len(host.actions) == 1
    asyncio.run(scenario())


def test_stdio_exposes_only_recall_and_function_execution(tmp_path):
    async def scenario():
        params = StdioServerParameters(command=sys.executable,
            args=['-m', 'src.integrations.gui_agent_mcp', '--device', 'protocol-test-no-device',
                  '--evidence-root', str(tmp_path)], cwd=str(Path(__file__).parents[1]))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                names = {tool.name for tool in (await client.list_tools()).tools}
                assert names == {'omniflow_recall', 'omniflow_execute'}
                rejected = await client.call_tool('omniflow_execute', {
                    'session_id': 'stale', 'request_id': 'one', 'function_id': 'wait', 'arguments': {}})
                assert rejected.is_error
    asyncio.run(scenario())
    assert not list(tmp_path.glob('*.json'))  # no implicit Memory compilation


def test_mcp_device_lease_excludes_second_owner():
    with device_lease('protocol-lease-test'):
        with pytest.raises(RuntimeError, match='already_owned'):
            with device_lease('protocol-lease-test'):
                raise AssertionError('two owners admitted')


@pytest.mark.parametrize('reward', [0, 1, 'checker error'])
def test_existing_runtime_stdio_bridge_and_completion_gate(tmp_path, reward):
    async def scenario():
        class Host:
            actions = []
            async def observe(self, **kwargs):
                return Observation(xml='<hierarchy/>')
            async def act(self, action):
                self.actions.append(action)
                return ActionResult(True)
        host = Host()
        store = FunctionStore(tmp_path/'store.json')
        store.put_function(Function('pause', 'Pause', 'Pause briefly',
            (FunctionStep(0, Action('wait', {'duration_ms': 1}), 'source'),),
            schema_version='omniflow.function.v2',
            input_schema={'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False}))
        flow = OmniFlow(store.path, host=host)
        checks = []
        def checker():
            checks.append(len(host.actions))
            if isinstance(reward, str):
                raise RuntimeError(reward)
            return reward
        flow.set_completion_checker(checker)
        runtime = GuiAgentToolRuntime(host=host, flow=flow, allow_task_switch=False)
        async with runtime_socket(runtime) as socket:
            params = StdioServerParameters(command=sys.executable,
                args=['-m', 'src.integrations.gui_agent_mcp', '--connect', socket],
                cwd=str(Path(__file__).parents[1]))
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as client:
                    await client.initialize()
                    assert {t.name for t in (await client.list_tools()).tools} == {'omniflow_recall', 'omniflow_execute'}
                    runtime._select_task('one')
                    args = {'session_id': runtime.session_id, 'request_id': 'one',
                            'function_id': 'pause', 'arguments': {}}
                    result = await client.call_tool('omniflow_execute', args)
                    feedback = json.loads(result.content[0].text)['feedback']
                    assert checks == [1] and len(host.actions) == 1
                    assert feedback['control']['next'] == ('host' if reward == 0 else 'stop')
                    assert await client.call_tool('omniflow_execute', args) == result
                    switched = await client.call_tool('omniflow_recall', {'task_id': 'two', 'goal': 'again', 'limit': 1})
                    assert switched.is_error and 'episode_task_fixed' in switched.content[0].text
                    assert len(host.actions) == 1
        assert runtime.execution_result().actions_executed == 1
        trace = runtime.execution_result().detail['trace']
        assert len(trace) == 1 and trace[0]['metadata']['function_id'] == 'pause'
        assert 'observation' not in trace[0]
    asyncio.run(scenario())


def test_real_mcp_cancellation_drains_action_and_closes_function_session(tmp_path):
    import anyio

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        actions = []
        class Host:
            async def observe(self, **kwargs):
                return Observation(xml='<hierarchy/>')
            async def get_state(self, state_id):
                return await self.observe()
            async def act(self, action):
                actions.append(action)
                entered.set()
                await release.wait()
                return ActionResult(True)
        host = Host()
        store = FunctionStore(tmp_path/'store.json')
        store.put_function(Function('pause_twice', 'Pause twice', 'Two waits',
            tuple(FunctionStep(i, Action('wait', {'duration_ms': 1}), 'source') for i in range(2)),
            schema_version='omniflow.function.v2',
            input_schema={'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False}))
        runtime = GuiAgentToolRuntime(host=host, flow=OmniFlow(store.path, host=host))
        server = create_server(runtime)
        client_write, server_read = anyio.create_memory_object_stream(16)
        server_write, client_read = anyio.create_memory_object_stream(16)
        async with anyio.create_task_group() as group:
            group.start_soon(server.run, server_read, server_write, server.create_initialization_options())
            async with ClientSession(client_read, client_write) as client:
                await client.initialize()
                args = {'session_id': runtime.session_id, 'request_id': 'cancelled-call',
                        'function_id': 'pause_twice', 'arguments': {}}
                task = asyncio.create_task(client.call_tool('omniflow_execute', args))
                await asyncio.wait_for(entered.wait(), 3)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                with anyio.fail_after(3):
                    while not runtime._control.cancelled.is_set():
                        await asyncio.sleep(.01)
                assert runtime.flow._operation_lock.locked()
                release.set()
                with anyio.fail_after(3):
                    while runtime.flow._operation_lock.locked():
                        await asyncio.sleep(.01)
                repeated = await client.call_tool('omniflow_execute', args)
                feedback = json.loads(repeated.content[0].text)['feedback']
                assert feedback['control']['reason'] == 'cancelled'
                assert feedback['execution']['actions_executed'] == 1
                rejected = await client.call_tool('omniflow_execute', {**args, 'request_id': 'new'})
                assert rejected.is_error and 'session_closed' in rejected.content[0].text
                assert len(actions) == 1
            group.cancel_scope.cancel()
    asyncio.run(scenario())
