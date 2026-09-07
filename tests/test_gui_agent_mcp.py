import asyncio
import json
from pathlib import Path
import sys

import pytest

pytest.importorskip('mcp')
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from omniflow.core.model import ActionResult, Observation
from omniflow.runtime.engine import OmniFlow
from src.integrations.gui_agent_mcp import GuiAgentMcp, device_lease
from src.integrations.gui_agent_tools import GuiAgentToolRuntime


def test_mcp_invocation_schema_and_retry_without_second_dispatch(tmp_path):
    class Host:
        def __init__(self):
            self.actions = []
        async def observe(self, **kwargs):
            return Observation(xml='<hierarchy/>')
        async def act(self, action):
            self.actions.append(action)
            return ActionResult(True)
    async def scenario():
        host = Host()
        runtime = GuiAgentToolRuntime(host=host, flow=OmniFlow(tmp_path/'store.json', host=host))
        transport = GuiAgentMcp(runtime)
        args = dict(session_id=runtime.session_id, request_id='one', tool_name='wait', arguments={'duration_ms': 1})
        request = types.CallToolRequestParams(name='omniflow_execute', arguments=args)
        first = await transport.call_tool(None, request)
        second = await transport.call_tool(None, request)
        assert not first.is_error and first == second and len(host.actions) == 1
        body = json.loads(first.content[0].text)
        from jsonschema import validate
        schema = json.loads((Path(__file__).parents[1]/'schemas/omniflow_invocation.v1.json').read_text())
        validate(body['feedback'], schema)
        invalid = await transport.call_tool(None, types.CallToolRequestParams(
            name='omniflow_execute', arguments={**args, 'request_id': 'two', 'arguments': {'duration_ms': 'bad'}}))
        assert invalid.is_error and len(host.actions) == 1
    asyncio.run(scenario())


def test_stdio_initialize_discover_cancel_and_explicit_new_session(tmp_path):
    async def scenario():
        params = StdioServerParameters(command=sys.executable,
            args=['-m', 'src.integrations.gui_agent_mcp', '--device', 'protocol-test-no-device',
                  '--evidence-root', str(tmp_path)], cwd=str(Path(__file__).parents[1]))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                names = {tool.name for tool in (await client.list_tools()).tools}
                assert {'omniflow_execute', 'omniflow_cancel', 'omniflow_finish'} <= names
                state = json.loads((await client.call_tool('omniflow_status', {})).content[0].text)
                old_id = state['session_id']
                cancelled = await client.call_tool('omniflow_cancel', {'session_id': old_id})
                assert not cancelled.is_error
                started = await client.call_tool('omniflow_start', {'session_id': old_id})
                assert not started.is_error
                assert json.loads(started.content[0].text)['session_id'] != old_id
                stale = await client.call_tool('omniflow_cancel', {'session_id': old_id})
                assert stale.is_error
    asyncio.run(scenario())
    assert not list(tmp_path.glob('*.json'))  # no implicit Memory compilation


def test_mcp_device_lease_excludes_second_owner():
    with device_lease('protocol-lease-test'):
        with pytest.raises(RuntimeError, match='already_owned'):
            with device_lease('protocol-lease-test'):
                raise AssertionError('two owners admitted')
