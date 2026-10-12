import asyncio
import json
from pathlib import Path

import pytest

from omniflow.core.model import RunResult
from src.integrations.gui_agent_harness import (
    HarnessContext, _host_completion_claim, _host_report, build_harness)
from src.experiment.run_tasks import build_parser, _experimental_omniflow_enabled
from src.integrations.android_world.agent import _goal_with_task_parameters


def test_builtin_keeps_same_flow_and_goal(tmp_path):
    class Flow:
        async def arun(self, goal, *, experiment):
            assert goal == 'task goal' and experiment.name == 'androidworld'
            return result
    result = RunResult(True)
    context = HarnessContext(Flow(), 'task goal', 30, tmp_path)
    assert asyncio.run(build_harness().arun(context)) is result


def test_unified_harness_selection(monkeypatch):
    monkeypatch.delenv('OMNIFLOW_HARNESS', raising=False)
    monkeypatch.delenv('OMNIFLOW_EXPERIMENTAL_MODEL', raising=False)
    assert build_parser().parse_args(['run']).harness == 'builtin'
    assert build_parser().parse_args(['run', '--harness', 'codex']).harness == 'codex'
    monkeypatch.setenv('OMNIFLOW_HARNESS', 'codex')
    assert _experimental_omniflow_enabled('omniflow')
    assert not _experimental_omniflow_enabled('mobilegpt')
    with pytest.raises(ValueError, match='unknown_harness'):
        build_harness('unknown')


def test_reentry_ablation_is_explicit_and_cannot_promote_formal_results(monkeypatch):
    monkeypatch.setenv('OMNIFLOW_FUNCTION_REENTRY', 'off')
    assert build_parser().parse_args(['run']).function_reentry == 'off'
    assert build_parser().parse_args(['run', '--function-reentry', 'on']).function_reentry == 'on'
    assert _experimental_omniflow_enabled('omniflow')
    assert not _experimental_omniflow_enabled('mobilegpt')


def test_matched_memory_ablation_requires_store_and_is_isolated(monkeypatch):
    from src.experiment.run_tasks import main
    monkeypatch.setenv('OMNIFLOW_CHECKER_MODE', 'on')
    monkeypatch.setenv('OMNIFLOW_FUNCTION_REENTRY', 'on')
    monkeypatch.setenv('OMNIFLOW_HARNESS', 'builtin')
    with pytest.raises(ValueError, match='function_memory_ablation_requires_builtin_omniflow_run_with_store'):
        main(['run', '--method', 'omniflow', '--function-memory', 'off'])
    monkeypatch.setenv('OMNIFLOW_FUNCTION_MEMORY', 'off')
    assert _experimental_omniflow_enabled('omniflow')
    assert build_parser().parse_args(['run', '--function-memory', 'on']).function_memory == 'on'


@pytest.mark.parametrize('args', [
    ['run', '--method', 'mobilegpt'],
    ['run', '--method', 'omniflow', '--harness', 'codex'],
    ['convert-memory', '--method', 'omniflow'],
])
def test_reentry_ablation_rejects_inapplicable_execution_owner(monkeypatch, args):
    from src.experiment.run_tasks import main
    monkeypatch.setenv('OMNIFLOW_CHECKER_MODE', 'on')
    with pytest.raises(ValueError, match='function_reentry_ablation_requires_builtin_omniflow_run'):
        main([*args, '--function-reentry', 'off'])


def test_host_usage_does_not_invent_api_request_count(tmp_path):
    events = tmp_path/'events.jsonl'
    events.write_text(json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': 12, 'output_tokens': 3}}))
    report = _host_report('codex', events)
    assert report['model_calls'] is None and report['model'] is None
    assert report['total_tokens'] == 15 and report['token_usage_status'] == 'reported'
    from jsonschema import validate, ValidationError
    schema = json.loads((Path(__file__).parents[1]/'schemas/oob/omniflow_run_log.v1.json').read_text())
    validate({'harness': report}, schema['properties']['diagnostics'])
    from omniflow.core.trajectory import _validate_schema
    _validate_schema({'harness': report}, schema['properties']['diagnostics'], schema, 'diagnostics')
    with pytest.raises(ValueError, match='type:'):
        _validate_schema({'harness': {**report, 'model': 123}},
                         schema['properties']['diagnostics'], schema, 'diagnostics')
    with pytest.raises(ValidationError):
        validate({'harness': {**report, 'model_calls': -1}}, schema['properties']['diagnostics'])


def test_task_parameters_do_not_override_function_schema():
    prompt = _goal_with_task_parameters('goal literal', {'seed': 1, 'value': 'literal'})
    assert 'omit fields absent from that schema' in prompt
    assert '"value": "literal"' in prompt and '"seed"' not in prompt


def test_task_parameters_absent_from_goal_are_never_disclosed():
    params = {'file_name': 'notes.md', 'clipboard': 'hidden text', 'rows': [{'a': 1}], 'answer': 42}
    prompt = _goal_with_task_parameters('Create notes.md with the clipboard content', params)
    assert '"file_name": "notes.md"' in prompt
    assert 'hidden text' not in prompt and '42' not in prompt and 'rows' not in prompt
    assert _goal_with_task_parameters('Do it', {'secret': 'x1'}) == 'Do it'


@pytest.mark.parametrize('requires_planner', [True, False])
def test_factory_harness_declares_planner_dependency(monkeypatch, tmp_path, requires_planner):
    import sys
    from types import SimpleNamespace
    from src.integrations.android_world import methods
    from omniflow.vlm import planner, function_router
    harness = SimpleNamespace(arun=lambda context: None, requires_builtin_planner=requires_planner)
    monkeypatch.setitem(sys.modules, 'test_harness_factory', SimpleNamespace(build=lambda: harness))
    monkeypatch.setenv('OMNIFLOW_HARNESS', 'test_harness_factory:build')
    monkeypatch.setattr(methods, 'resolve_openai_compatible_config', lambda **kwargs: ('key', 'http://localhost'))
    monkeypatch.setattr(planner, 'VLMPlanner', lambda **kwargs: 'planner')
    monkeypatch.setattr(function_router, 'VLMFunctionRouter', lambda **kwargs: 'router')
    captured = {}
    store = tmp_path/'store.json'
    def build(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(store=SimpleNamespace(path=store))
    methods._build_omniflow(methods.MethodAdapterContext(selector='omniflow', env=None,
        store_path=str(store), adb_serial='test', planner_model='test', build_omniflow_agent=build))
    assert captured['harness'] is harness
    assert ('planner' in captured) == requires_planner
    assert ('function_router' in captured) == requires_planner


def _events(tmp_path, *events):
    path = tmp_path / 'events.jsonl'
    path.write_text('\n'.join(json.dumps(event) for event in events) + '\nnot-json\n')
    return path


def test_host_completion_claim_reads_explicit_final_status(tmp_path):
    claude = _events(tmp_path, {'type': 'result', 'result': 'done\nANSWER: 42\nTASK_STATUS: complete'})
    assert _host_completion_claim('claude', claude) == (True, '42')
    codex = _events(tmp_path,
        {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'TASK_STATUS: complete'}},
        {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': '`TASK_STATUS: incomplete`'}})
    assert _host_completion_claim('codex', codex) == (False, '')


def test_host_completion_claim_absent_status_is_incomplete(tmp_path):
    path = _events(tmp_path, {'type': 'result', 'result': 'I turned Bluetooth on.'})
    assert _host_completion_claim('claude', path) == (False, '')
    assert _host_completion_claim('codex', path) == (False, '')


@pytest.mark.parametrize('variable', [
    'OMNIFLOW_CHECKER_MODE', 'OMNIFLOW_COMPLETION_ORACLE',
    'OMNIFLOW_FUNCTION_REENTRY', 'OMNIFLOW_FUNCTION_MEMORY'])
def test_environment_ablation_switches_must_be_on_or_off(monkeypatch, variable):
    from src.experiment.run_tasks import main
    for name in ('OMNIFLOW_CHECKER_MODE', 'OMNIFLOW_FUNCTION_REENTRY', 'OMNIFLOW_FUNCTION_MEMORY'):
        monkeypatch.setenv(name, 'on')
    monkeypatch.setenv('OMNIFLOW_COMPLETION_ORACLE', 'off')
    monkeypatch.setenv(variable, 'yes')
    with pytest.raises(ValueError, match='_must_be_on_or_off'):
        main(['run', '--method', 'omniflow'])
