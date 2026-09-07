"""Transport/backend conformance, not a learned-mapping or device benchmark."""

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest

from omniflow.core.config import OmniFlowConfig, PluginSet
from omniflow.core.model import (
    Action,
    ActionResult,
    Function,
    FunctionStep,
    Observation,
    TransferResult,
)
from omniflow.functions.store import FunctionStore
from omniflow.runtime.engine import OmniFlow
from src.integrations.gui_agent_mcp import GuiAgentMcp, build_runtime
from src.integrations.gui_agent_oob_host import OobGuiAgentHost
from src.integrations.gui_agent_tools import GuiAgentToolRuntime
from src.integrations.gui_owl_adapter import GuiOwlAdapter
from src.integrations.mobilerun_function_tools import build_runtime_custom_tools
from src.integrations.vdroid_adapter import VDroidAdapter


class ContractEncoder:
    """Test double only. Production still uses canonical OmniTransfer V10."""
    name, version, dimension = 'contract-fixture', 'test', 1024
    def embed(self, observation):
        return SimpleNamespace(vector=np.ones(1024), node_count=1, elements=())


def make_runtime(tmp_path, backend, partial_failure=False):
    calls = []
    observation = Observation(xml='<hierarchy width="720" height="1280"/>', package_name='com.android.settings',
                              extra={'display': {'width': 720, 'height': 1280}})
    class SyncHost:
        def observe(self, **kwargs): return observation
        def get_state(self, state_id): return observation
        def act(self, action):
            calls.append(Action.from_value(action))
            return ActionResult(True)
    class AsyncHost:
        async def observe(self, **kwargs): return observation
        async def get_state(self, state_id): return observation
        async def act(self, action):
            calls.append(Action.from_value(action))
            return ActionResult(True)
    class OobClient:
        def observe(self, **kwargs):
            return {'xml': observation.xml, 'display': observation.extra['display'], 'package_name': observation.package_name}
        def act(self, action):
            calls.append(Action.from_value(action))
            return {'success': True}
    hosts = {'sync': SyncHost, 'async': AsyncHost,
             'oob': lambda: OobGuiAgentHost(OobClient(), source_states={'source': observation.to_dict()})}
    store = FunctionStore(tmp_path/'store.json')
    steps = [FunctionStep(0, Action('wait', {'duration_ms': 1}), 'source')]
    if partial_failure:
        steps.append(FunctionStep(1, Action('click', {'x': 300, 'y': 400}), 'source'))
    function = Function('pause', 'Pause', 'Pause briefly', tuple(steps), schema_version='omniflow.function.v2',
                        input_schema={'type':'object','properties':{},'required':[],'additionalProperties':False})
    store.put_function(function)
    class ForbiddenPlanner:
        async def one_step_tool_call(self, *args, **kwargs):
            raise AssertionError('external Function service started a Planner')
    flow = OmniFlow(store.path, host=hosts[backend](), planner=ForbiddenPlanner(),
        config=OmniFlowConfig(plugins=PluginSet(checker=lambda ctx: None,
            transfer=lambda *args: TransferResult(None, reason='test_mapping_rejected'))))
    flow._page_encoder = ContractEncoder()
    return GuiAgentToolRuntime(host=flow.host, flow=flow), calls


async def dispatch(runtime, harness, name, arguments):
    if harness == 'mcp':
        from mcp import types
        response = await GuiAgentMcp(runtime).call_tool(None,
            types.CallToolRequestParams(name=name, arguments=arguments))
        return json.loads(response.content[0].text)
    if harness == 'gui_owl':
        result = await GuiOwlAdapter(runtime).execute_output(
            '<tool_call>'+json.dumps({'name':name,'arguments':arguments})+'</tool_call>')
        response = result.tool_result
    elif harness == 'vdroid':
        result = await VDroidAdapter(runtime).execute_action(
            {'action_type':'tool','name':name,'arguments':arguments}, {})
        response = result.tool_result
    else:
        text = await build_runtime_custom_tools(runtime)[name]['function'](**arguments)
        result = json.loads(text.split(': ',1)[1])
        return {'success':result['success'], **result['output']}
    return {'success':response.success, **response.output}


@pytest.mark.parametrize('harness', ['mcp','gui_owl','vdroid','mobilerun'])
@pytest.mark.parametrize('backend', ['sync','async','oob'])
@pytest.mark.parametrize('partial_failure', [False,True])
def test_two_function_operations_share_kernel_across_harnesses_and_hosts(tmp_path,harness,backend,partial_failure):
    if harness == 'mcp': pytest.importorskip('mcp')
    runtime, calls = make_runtime(tmp_path, backend, partial_failure)
    async def scenario():
        recalled = await dispatch(runtime,harness,'omniflow_recall',
                                  {'task_id':'task-one','goal':'Pause briefly','limit':8})
        assert recalled['success'], recalled
        assert [f['function_id'] for f in recalled['functions']] == ['pause']
        assert not calls, 'recall dispatched a physical action'
        request = {'session_id':recalled['session_id'],'request_id':'execute-once',
                   'function_id':'pause','arguments':{}}
        result = await dispatch(runtime,harness,'omniflow_execute',request)
        assert result['success'] is not partial_failure
        feedback = result['feedback']
        assert feedback['execution']['actions_executed'] == 1
        assert [action.tool for action in calls] == ['wait']
        assert feedback['task']['status'] == 'unknown'
        assert feedback['control']['next'] == 'host'
        if partial_failure:
            assert feedback['execution']['failed_step_index'] == 1
            assert feedback['execution']['failed_action_dispatched'] is False
        repeated = await dispatch(runtime,harness,'omniflow_execute',request)
        assert repeated['feedback'] == feedback and len(calls)==1
    asyncio.run(scenario())


def test_new_task_identity_retires_previous_session_and_primitives_are_not_functions(tmp_path):
    runtime,calls=make_runtime(tmp_path,'async')
    async def scenario():
        a=await runtime.call_function_tool('omniflow_recall',{'task_id':'one','goal':'Pause','limit':8})
        sid=a.output['session_id']
        await runtime.call_function_tool('omniflow_recall',{'task_id':'two','goal':'Pause','limit':8})
        with pytest.raises(ValueError,match='session_mismatch'):
            await runtime.call_function_tool('omniflow_execute',{
                'session_id':sid,'request_id':'x','function_id':'pause','arguments':{}})
        with pytest.raises(ValueError,match='task_id_retired'):
            await runtime.call_function_tool('omniflow_recall',{'task_id':'one','goal':'Pause','limit':8})
        with pytest.raises(ValueError,match='function_not_registered'):
            await runtime.call_function_tool('omniflow_execute',{
                'session_id':runtime.session_id,'request_id':'y','function_id':'click','arguments':{'x':1,'y':1}})
        assert not calls
    asyncio.run(scenario())


def test_backend_factory_receives_explicit_configuration_and_uses_shared_runtime(tmp_path):
    seen=[]
    def factory(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(observe=lambda **kw:Observation(xml='<hierarchy/>'),
                               act=lambda action:ActionResult(True), get_state=lambda sid:None)
    runtime=build_runtime(device='configured-device',evidence_root=tmp_path,host_factory=factory)
    result=asyncio.run(runtime.call_function_tool('omniflow_recall',{
        'task_id':'empty','goal':'Find an operation','limit':8}))
    assert result.success and result.output['functions']==[]
    assert seen==[{'device':'configured-device','evidence_root':tmp_path,'source_states':{}}]
    assert not list(tmp_path.glob('*.json'))


def test_recall_cancellation_and_concurrent_task_switch_do_not_dispatch(tmp_path):
    from threading import Event
    entered, release = Event(), Event()
    runtime, calls = make_runtime(tmp_path, 'async')
    class SlowEncoder(ContractEncoder):
        def embed(self, observation):
            entered.set()
            assert release.wait(3)
            return super().embed(observation)
    runtime.flow._page_encoder = SlowEncoder()
    async def scenario():
        task=asyncio.create_task(runtime.call_function_tool('omniflow_recall',{
            'task_id':'one','goal':'Pause','limit':8}))
        assert await asyncio.to_thread(entered.wait,2)
        sid=runtime.session_id
        with pytest.raises(ValueError,match='session_busy'):
            await runtime.call_function_tool('omniflow_recall',{'task_id':'two','goal':'Pause','limit':8})
        assert runtime.session_id==sid
        runtime.cancel()
        release.set()
        result=await task
        assert not result.success
        assert result.output['feedback']['control']['reason']=='cancelled'
        assert not calls
    asyncio.run(scenario())


def test_execute_does_not_depend_on_legacy_primitive_tool_catalog(tmp_path):
    runtime,calls=make_runtime(tmp_path,'sync')
    def forbidden_catalog():
        raise AssertionError('Function service enumerated legacy primitive tools')
    runtime.list_tools=forbidden_catalog
    async def scenario():
        result=await runtime.call_function_tool('omniflow_execute',{
            'session_id':runtime.session_id,'request_id':'one','function_id':'pause','arguments':{}})
        assert result.success
        assert [a.tool for a in calls]==['wait']
    asyncio.run(scenario())


@pytest.mark.parametrize('backend', ['sync', 'async', 'oob'])
@pytest.mark.parametrize('inventory', [None, {}, {'Settings': 'com.android.settings'}])
def test_open_app_uses_host_inventory_without_bypassing_missing_packages(tmp_path, backend, inventory):
    runtime, calls = make_runtime(tmp_path, backend)
    reads = []
    def inventory_sync():
        reads.append(True)
        return inventory
    async def inventory_async():
        return inventory_sync()
    owner = runtime.host.control_client if backend == 'oob' else runtime.host
    owner.installed_apps = inventory_async if backend == 'async' else inventory_sync
    runtime.flow.store.put_function(Function(
        'open_settings', 'Settings', 'Open settings',
        (FunctionStep(0, Action('open_app', {'package_name': 'com.android.settings'}), 'source'),),
        schema_version='omniflow.function.v2',
        input_schema={'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False}))
    result = asyncio.run(runtime.flow.aexecute_function('open_settings', {}))
    assert result.success is bool(inventory)
    assert len(calls) == int(bool(inventory))
    assert reads == [True]
    if inventory is None:
        assert result.error == 'open_app_installed_packages_unavailable'
    elif not inventory:
        assert 'not_installed' in result.error


def test_oob_inventory_queries_actual_device_and_propagates_failure():
    import subprocess

    from src.integrations.android_world.oob_control import OobControlClient
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, 'package:com.android.settings\npackage:app.example\n', '')
    client = OobControlClient(None, adb_serial='target', run=run)
    assert client.installed_apps() == {'com.android.settings': 'com.android.settings', 'app.example': 'app.example'}
    assert calls == [['adb', '-s', 'target', 'shell', 'pm', 'list', 'packages']]
    client._run_command = lambda command, **kwargs: subprocess.CompletedProcess(command, 1, '', 'offline')
    with pytest.raises(RuntimeError, match='query_failed'):
        client.installed_apps()


@pytest.mark.parametrize('partial_failure', [False, True])
def test_installed_droidrun_registry_preserves_function_feedback(tmp_path, partial_failure):
    pytest.importorskip('mobilerun')
    registry_module = pytest.importorskip('droidrun.agent.tool_registry')
    runtime, calls = make_runtime(tmp_path, 'async', partial_failure)
    registry = registry_module.ToolRegistry()
    registry.register_from_dict(build_runtime_custom_tools(runtime, include_actions=False))
    assert set(registry.get_signatures()) == {'omniflow_recall', 'omniflow_execute'}
    async def scenario():
        recalled = await registry.execute('omniflow_recall', {
            'task_id': 'sdk', 'goal': 'Pause', 'limit': 8}, ctx=None)
        assert recalled.success and not calls
        output = json.loads(recalled.summary.split(': ', 1)[1])['output']
        executed = await registry.execute('omniflow_execute', {
            'session_id': output['session_id'], 'request_id': 'one',
            'function_id': 'pause', 'arguments': {}}, ctx=None)
        assert executed.success is not partial_failure
        feedback = json.loads(executed.summary.split(': ', 1)[1])['output']['feedback']
        assert feedback['control']['next'] == 'host'
        assert feedback['execution']['actions_executed'] == 1
    asyncio.run(scenario())
