from __future__ import annotations

import json
from io import StringIO

from omniflow.functions.artifact import parse_function_artifact
from omniflow.functions.compiler import compile_runlog_to_store
from omniflow.runlog import import_run_log_evidence
from omniflow.bridge import JsonLineBridge, _BridgeHost
from omniflow.transfer.runtime import load_transfer_state_catalog
import pytest


def _state(xml: str = "<hierarchy />") -> dict:
    return {
        "pixels": None,
        "xml": xml,
        "auxiliaries": {"display": {"width": 1000, "height": 1000}},
    }


def _run_log() -> dict:
    before = _state()
    after = _state('<hierarchy><node text="Done" /></hierarchy>')
    return {
        "schema_version": "omniflow.run_log.v1",
        "run_id": "device-registration-test",
        "task_name": "device-registration-test",
        "goal": "点击完成",
        "task_parameters": {},
        "seed": None,
        "status": "succeeded",
        "success": True,
        "validator": {"official": True, "success": True, "reward": 1},
        "provenance": {"kind": "runtime"},
        "steps": [
            {
                "step_index": 0,
                "observation": before,
                "action": {"action_type": "click", "x": 500, "y": 500},
                "result": {"success": True},
                "next_observation": after,
            }
        ],
    }


def test_device_registration_uses_official_python_projection_without_model(
    tmp_path,
) -> None:
    run_log = _run_log()
    report = compile_runlog_to_store(
        run_log,
        tmp_path / "compiled",
        source_states=import_run_log_evidence(run_log)[1],
    )

    assert report["success"] is True
    assert report["model_calls"] == 0
    assert report["function_ids"] == ["complete_source_workflow"]
    function = parse_function_artifact(
        json.loads((tmp_path / "compiled" / "store.json").read_text())[
            "functions"
        ]["complete_source_workflow"]
    )
    assert function.steps[0].action.to_dict() == {
        "tool": "click",
        "args": {"x": 500, "y": 500},
    }


def test_input_text_keeps_source_point_for_omnitransfer_replay(tmp_path) -> None:
    run_log = _run_log()
    run_log["goal"] = "输入完成"
    run_log["steps"][0]["action"] = {
        "action_type": "input_text",
        "text": "完成",
        "x": 500,
        "y": 500,
    }

    report = compile_runlog_to_store(
        run_log,
        tmp_path / "compiled",
        source_states=import_run_log_evidence(run_log)[1],
    )
    function = parse_function_artifact(
        json.loads((tmp_path / "compiled" / "store.json").read_text())["functions"][
            report["function_ids"][0]
        ]
    )

    assert function.steps[0].action.to_dict() == {
        "tool": "input_text",
        "args": {"text": "完成", "x": 500, "y": 500},
    }
    assert function.bindings == ()


def test_answer_only_source_retains_app_entry_as_reusable_progress(tmp_path) -> None:
    run_log = _run_log()
    run_log["goal"] = "Open the clock app and report what is visible."
    run_log["steps"] = [
        {
            "step_index": 0,
            "observation": _state(),
            "action": {"action_type": "open_app", "app_name": "clock"},
            "result": {"success": True},
            "next_observation": _state(
                '<hierarchy><node package="com.google.android.deskclock" '
                'text="Clock" /></hierarchy>'
            ),
        },
        {
            "step_index": 1,
            "observation": _state(
                '<hierarchy><node package="com.google.android.deskclock" '
                'text="Clock" /></hierarchy>'
            ),
            "action": {"action_type": "answer", "text": "Clock is open."},
            "result": {"success": True},
            "next_observation": _state(
                '<hierarchy><node package="com.google.android.deskclock" '
                'text="Clock" /></hierarchy>'
            ),
        },
    ]

    report = compile_runlog_to_store(
        run_log,
        tmp_path / "compiled",
        source_states=import_run_log_evidence(run_log)[1],
    )

    function = parse_function_artifact(
        json.loads((tmp_path / "compiled" / "store.json").read_text())["functions"][
            report["function_ids"][0]
        ]
    )
    assert [step.action.to_dict() for step in function.steps] == [
        {
            "tool": "open_app",
            "args": {"package_name": "com.google.android.deskclock"},
        }
    ]


def test_save_function_bridge_accepts_canonical_runlog_without_function_draft(
    tmp_path,
) -> None:
    bridge = JsonLineBridge(
        tmp_path / "store.json",
        reader=StringIO(),
        writer=StringIO(),
    )

    result = bridge._save_function(
        "request-1",
        {
            "run_id": "device-registration-test",
            "run_log": _run_log(),
            "agent_visible": False,
            "enhance": False,
        },
    )

    assert result["success"] is True
    assert result["registered"] is True
    assert result["function_ids"] == ["complete_source_workflow"]
    assert result["functions"][0]["agent_visible"] is False
    assert result["function"]["function_id"] == "complete_source_workflow"


def test_save_function_bridge_enhances_and_registers_the_same_function(
    tmp_path,
) -> None:
    run_log = _run_log()
    compiled_root = tmp_path / "source-compiled"
    compile_runlog_to_store(
        run_log,
        compiled_root,
        source_states=import_run_log_evidence(run_log)[1],
    )
    function = json.loads(
        (compiled_root / "store.json").read_text()
    )["functions"]["complete_source_workflow"]
    bridge = JsonLineBridge(
        tmp_path / "store.json",
        reader=StringIO(),
        writer=StringIO(),
    )
    model_calls: list[dict] = []

    def fake_host_call(request_id: str, method: str, payload: dict) -> dict:
        assert method == "complete_json"
        model_calls.append(payload)
        return {
            "content": json.dumps(
                {
                    "name": "增强后的点击流程",
                    "description": "增强后的可复用点击流程",
                    "parameters": [],
                },
                ensure_ascii=False,
            )
        }

    bridge.host_call = fake_host_call  # type: ignore[method-assign]
    result = bridge._save_function(
        "request-enhance",
        {
            "run_id": run_log["run_id"],
            "run_log": run_log,
            "functions": [function],
            "enhance": True,
            "instruction": "让名称更清楚",
        },
    )

    assert result["success"] is True
    assert result["registered"] is True
    assert result["function_id"] == "complete_source_workflow"
    assert result["function"]["function_id"] == "complete_source_workflow"
    assert result["function"]["name"] == "增强后的点击流程"
    assert result["functions"] == [result["function"]]
    assert len(model_calls) == 1
    assert model_calls[0]["max_tokens"] == 8192


@pytest.mark.parametrize("invalid_patch", [7, "wrapper"])
def test_runlog_authoring_uses_host_model_and_compiler_feedback(tmp_path, invalid_patch) -> None:
    bridge = JsonLineBridge(tmp_path / "store.json", reader=StringIO(), writer=StringIO())
    run_log = _run_log()
    before = run_log['steps'][0]['observation']
    before['auxiliaries']['display'] = {'width': 1080, 'height': 2400}
    run_log['steps'][0]['next_observation']['auxiliaries']['display'] = {'width': 1080, 'height': 2400}
    before['xml'] = '<hierarchy><node id="done" text="Done" bounds="[480,1100][600,1300]" /></hierarchy>'
    run_log['steps'][0]['action'].update(x=540, y=1200)
    prompts = []
    accepted = {"patches": ["span 0 0 open_done"]}
    rejected = (
        {"Root": accepted}
        if invalid_patch == "wrapper"
        else {"patches": ["step 7 stable"]}
    )

    def host(request_id, method, payload):
        assert method == "complete_json"
        assert payload["max_tokens"] == 8192
        prompts.append(payload["prompt"])
        request = json.loads(payload['prompt'].rsplit('\n\n', 1)[1])
        card = request['judgment_card']['steps'][0]
        assert 'source_ui' not in card
        assert 'nodes' not in card
        assert card['hit']['node_id'] == 'done'
        assert card['hit']['text'] == 'Done'
        return {"content": json.dumps(rejected if len(prompts) == 1 else accepted)}

    bridge.host_call = host
    result = bridge._save_function("author", {"run_log": run_log, "enhance": True})
    assert result["success"] is True, result
    assert len(prompts) == 2
    feedback = json.loads(prompts[1].rsplit('\n\n', 1)[1])
    assert 'previous_response' not in feedback
    assert feedback['applied_patches'] == []
    failed = feedback['failed_patches'][0]
    if invalid_patch == 7:
        assert 'unknown_source_step_index=7' in failed['error']
        assert 'allowed_source_step_indices=[0]' in failed['error']
    else:
        assert "received_fields=['Root']" in failed['error']
        assert 'expected_top_level_fields=patches' in failed['error']
    functions = {item['function_id']: item for item in result['functions']}
    assert 'open_done' in functions
    function = parse_function_artifact({
        key: value for key, value in functions['open_done'].items() if key != 'source_run_id'
    })
    assert function.steps[0].action.to_dict() == {"tool": "click", "args": {"x": 500, "y": 500}}
    assert function.agent_visible is True


def test_host_authoring_keeps_span_when_one_argument_patch_fails(tmp_path) -> None:
    bridge = JsonLineBridge(tmp_path / 'store.json', reader=StringIO(), writer=StringIO())
    run = _run_log()
    run['steps'][0]['action'] = {'action_type': 'input_text', 'text': 'battery', 'x': 500, 'y': 500}
    requests = []

    def host(request_id, method, payload):
        request = json.loads(payload['prompt'].rsplit('\n\n', 1)[1])
        requests.append(request)
        if len(requests) == 1:
            return {'content': json.dumps({'patches': [
                'span 0 0 search',
                'step 0 task_parameter name=search_term arg=search_term',
            ]})}
        error = request['failed_patches'][0]['error']
        assert "arg_name='search_term'" in error
        assert "available_action_args=['text', 'x', 'y']" in error
        assert request['applied_patches'] == ['span 0 0 search']
        return {'content': json.dumps({'patches': [
            'step 0 task_parameter name=search_term arg=text',
        ]})}

    bridge.host_call = host
    result = bridge._save_function('author', {'run_log': run, 'enhance': True})
    assert result['success'] is True, json.dumps(result)
    assert len(requests) == 2
    functions = {item['function_id']: item for item in result['functions']}
    assert functions['search']['bindings'] == [{
        'source': '$.arguments.search_term',
        'target': '$.steps[0].action.args.text',
    }]


def test_parameter_feedback_identifies_missing_invocation_binding() -> None:
    from omniflow.functions.compiler import _agent_owned_parameters
    with pytest.raises(ValueError, match=r'parameter=query:missing_occurrence_indices=\[0\]'):
        _agent_owned_parameters(
            [{'name': 'query', 'description': 'Search query', 'bindings': [
                {'occurrence_index': 1, 'source_step_index': 1, 'binding_kind': 'action_arg', 'arg_name': 'text'}
            ]}], occurrence_count=2, spans=[[0], [1]], facts={},
        )


def test_rejected_host_authoring_does_not_register_hidden_fallback_as_success(tmp_path) -> None:
    bridge = JsonLineBridge(tmp_path / "store.json", reader=StringIO(), writer=StringIO())
    calls = []

    def host(request_id, method, payload):
        calls.append(payload)
        return {"content": "{}"}

    bridge.host_call = host
    result = bridge._save_function("author", {"run_log": _run_log(), "enhance": True})
    assert result["success"] is False
    assert result["error"]["code"] == "FUNCTION_AUTHORING_REJECTED"
    assert len(calls) == 3
    assert bridge.flow.store.get_function("complete_source_workflow") is None
    assert result["diagnostics"]["authoring_workflow"]["rejected_error"]
    assert result["error"]["message"] == result["diagnostics"]["authoring_workflow"]["rejected_error"]
    assert result["diagnostic_persisted"] is True
    persisted = tmp_path / "authoring_reports" / result["diagnostic_path"].rsplit("/", 1)[-1]
    assert persisted.parent == tmp_path / "authoring_reports"
    assert json.loads(persisted.read_text()) == result["diagnostics"]
    attempts = json.loads(persisted.read_text())['authoring_attempts']
    assert len(attempts) == 3
    assert [attempt['attempt'] for attempt in attempts] == [1, 2, 3]
    assert all(attempt['raw_response'] == '{}' and attempt['error'] for attempt in attempts)
    reopened = JsonLineBridge(tmp_path / "store.json", reader=StringIO(), writer=StringIO())
    assert reopened.flow.store.get_function("complete_source_workflow") is None
    assert persisted.is_file()


def test_rejected_authoring_survives_diagnostic_storage_failure(tmp_path) -> None:
    bridge = JsonLineBridge(tmp_path / "store.json", reader=StringIO(), writer=StringIO())
    bridge.host_call = lambda *_: {"content": "{}"}
    (tmp_path / "authoring_reports").write_text("not a directory")
    result = bridge._save_function("author", {"run_log": _run_log(), "enhance": True})
    assert result["success"] is False
    assert result["error"]["code"] == "FUNCTION_AUTHORING_REJECTED"
    assert result["diagnostics"]["authoring_workflow"]["rejected_error"]
    assert result["diagnostic_persisted"] is False
    assert "diagnostic_path" not in result


def test_package_commits_evidence_before_auto_registration(tmp_path) -> None:
    bridge = JsonLineBridge(tmp_path / "store.json", reader=StringIO(), writer=StringIO())
    calls = []
    bridge.host_call = lambda request_id, method, payload: calls.append(method) or {"finished": True}
    def save(request_id, body):
        assert calls == ["finish_run"]
        assert body == {"run_id": "run-1"}
        return {"success": True, "registered": True, "function_id": "registered"}
    bridge._save_function = save
    result = bridge._complete_run("call", {
        "success": True, "run_id": "run-1",
        "post_run_actions": [{"name": "save_function", "arguments": {"run_id": "run-1"}}],
    })
    assert result["auto_registered"] is True
    assert result["registered_function_id"] == "registered"
    assert "post_run_actions" not in result


def test_registration_failure_preserves_successful_device_result(tmp_path) -> None:
    bridge = JsonLineBridge(tmp_path / "store.json", reader=StringIO(), writer=StringIO())
    bridge.host_call = lambda *args: {"finished": True}
    def fail(*args):
        raise ValueError("authoring unavailable")
    bridge._save_function = fail
    result = bridge._complete_run("call", {
        "success": True, "run_id": "run-1",
        "post_run_actions": [{"name": "save_function", "arguments": {"run_id": "run-1"}}],
    })
    assert result["success"] is True
    assert result["auto_registered"] is False
    assert result["registration_error"] == "authoring unavailable"


def test_registered_source_survives_screenshot_cleanup_and_bridge_restart(tmp_path) -> None:
    screenshot = tmp_path / "recording.png"
    screenshot.write_bytes(b"immutable screenshot evidence")
    run_log = _run_log()
    run_log["steps"][0]["observation"]["pixels"] = {
        "path": str(screenshot), "width": 1000, "height": 1000, "mime_type": "image/png",
    }
    path = tmp_path / "registered" / "store.json"
    bridge = JsonLineBridge(path, reader=StringIO(), writer=StringIO())
    result = bridge._save_function("save", {"run_log": run_log, "enhance": False})
    assert result["success"] is True, result
    state_id = result["function"]["steps"][0]["source_state_id"]
    screenshot.unlink()

    restarted = JsonLineBridge(path, reader=StringIO(), writer=StringIO())
    def unavailable_host(*args):
        raise AssertionError("Registered evidence must not depend on old RunLog storage")
    restarted.host_call = unavailable_host
    observation = _BridgeHost(restarted, "replay").get_state(state_id)
    catalog = load_transfer_state_catalog(path.parent / "transfer_states.json")
    assert observation.xml == run_log["steps"][0]["observation"]["xml"]
    from pathlib import Path
    assert Path(catalog[state_id]["screenshot_path"]).read_bytes() == b"immutable screenshot evidence"


def test_state_import_keeps_earlier_evidence_and_rejects_identity_conflict(tmp_path) -> None:
    path = tmp_path / "registered" / "store.json"
    bridge = JsonLineBridge(path, reader=StringIO(), writer=StringIO())
    first = bridge._save_function("first", {"run_log": _run_log(), "enhance": False})
    first_id = first["function"]["steps"][0]["source_state_id"]
    second_log = _run_log()
    second_log["run_id"] = "second-run"
    second_log["steps"][0]["observation"]["xml"] = '<hierarchy page="second" />'
    second = bridge._save_function("second", {"run_log": second_log, "enhance": False})
    second_id = second["function"]["steps"][0]["source_state_id"]
    catalog_path = path.parent / "transfer_states.json"
    before = catalog_path.read_bytes()
    assert {first_id, second_id} <= load_transfer_state_catalog(catalog_path).keys()
    bridge.flow.store.import_transfer_states(catalog_path)
    assert catalog_path.read_bytes() == before
    conflict = json.loads(before)
    conflict["states"][first_id]["xml"] = '<hierarchy overwritten="true" />'
    conflict_path = tmp_path / "conflict.json"
    conflict_path.write_text(json.dumps(conflict))
    with pytest.raises(ValueError, match="function_source_state_conflict"):
        bridge.flow.store.import_transfer_states(conflict_path)
    assert catalog_path.read_bytes() == before


def test_registration_keeps_provenance_after_restart_and_does_not_overwrite(tmp_path):
    path = tmp_path / 'store.json'
    bridge = JsonLineBridge(path, reader=StringIO(), writer=StringIO())
    first = bridge._save_function('one', {'run_log': _run_log(), 'enhance': False})
    second_log = _run_log()
    second_log['run_id'] = 'another-recording'
    second_log['steps'][0]['action']['x'] = 600
    second = bridge._save_function('two', {'run_log': second_log, 'enhance': False})
    assert first['function_id'] != second['function_id']
    restarted = JsonLineBridge(path, reader=StringIO(), writer=StringIO())
    loaded = restarted._catalog({'action': 'get', 'function_id': first['function_id']})['function']
    assert loaded['steps'] == first['function']['steps']
    assert loaded['source_run_id'] == _run_log()['run_id']
    again = restarted._save_function('again', {'run_log': second_log, 'enhance': False})
    assert again['function_id'] == second['function_id']
    assert len(restarted._catalog({'include_hidden': True})['functions']) == 2


def test_automatic_save_never_waits_for_semantic_model(tmp_path):
    bridge = JsonLineBridge(tmp_path / 'store.json', reader=StringIO(), writer=StringIO())
    calls = []
    def host(request_id, method, body):
        calls.append(method)
        if method == 'finish_run':
            return {'finished': True}
        if method == 'get_run_log':
            return _run_log()
        if method == 'get_state':
            return import_run_log_evidence(_run_log())[1][body['state_id']]
        raise AssertionError(f'Automatic registration must not call a model: {method}')
    bridge.host_call = host
    result = bridge._complete_run('auto', {
        'success': True, 'run_id': _run_log()['run_id'],
        'post_run_actions': [{'name': 'save_function', 'arguments': {'run_log': _run_log()}}],
    })
    assert result['auto_registered'] is True, result
    assert 'complete_json' not in calls


def test_cancelled_run_keeps_terminal_identity_and_cannot_register(tmp_path):
    from omniflow.core.trajectory import canonicalize_run_log
    run = _run_log()
    run.update(status='cancelled', success=False)
    run['validator'].update(success=False, reward=0)
    assert canonicalize_run_log(run)['status'] == 'cancelled'
    bridge = JsonLineBridge(tmp_path / 'store.json', reader=StringIO(), writer=StringIO())
    result = bridge._save_function('cancelled', {'run_log': run})
    assert result['success'] is False
    assert not bridge.flow.store.functions


def test_enhancement_can_use_reloaded_management_payload(tmp_path):
    path = tmp_path / 'store.json'
    bridge = JsonLineBridge(path, reader=StringIO(), writer=StringIO())
    saved = bridge._save_function('save', {'run_log': _run_log()})
    bridge = JsonLineBridge(path, reader=StringIO(), writer=StringIO())
    function = bridge._catalog({'action': 'get', 'function_id': saved['function_id']})['function']
    def host(request_id, method, body):
        if method == 'get_run_log':
            assert body['run_id'] == function['source_run_id']
            return _run_log()
        if method == 'get_state':
            return import_run_log_evidence(_run_log())[1]['states'][body['state_id']]
        assert method == 'complete_json'
        return {'content': json.dumps({'name': 'Enhanced after restart', 'description': 'Open completion', 'parameters': []})}
    bridge.host_call = host
    enhanced = bridge._save_function('enhance', {'run_id': function['source_run_id'], 'function': function, 'enhance': True})
    assert enhanced['success'] is True, enhanced
    assert enhanced['function_id'] == function['function_id']
    assert enhanced['function']['steps'] == function['steps']
    assert enhanced['function']['name'] == 'Enhanced after restart'
    assert enhanced['function']['source_run_id'] == function['source_run_id']


@pytest.mark.parametrize("full_authoring", [False, True])
def test_enhancement_uses_frozen_screenshot_after_android_recaptures_state(tmp_path, full_authoring):
    screenshot = tmp_path / 'transient.png'
    screenshot.write_bytes(b'original source image')
    run = _run_log()
    run['steps'][0]['observation']['pixels'] = {
        'path': str(screenshot), 'width': 1000, 'height': 1000, 'mime_type': 'image/png',
    }
    bridge = JsonLineBridge(tmp_path / 'store.json', reader=StringIO(), writer=StringIO())
    saved = bridge._save_function('save', {'run_log': run})
    snapshot = (tmp_path / 'transfer_states.json').read_bytes()
    screenshot.write_bytes(b'later screenshot with same structural state id')
    def host(request_id, method, body):
        if method == 'get_run_log':
            return run
        if method == 'get_state':
            return import_run_log_evidence(run)[1]['states'][body['state_id']]
        assert method == 'complete_json'
        if full_authoring:
            return {'content': json.dumps({
                'patches': ['span 0 0 authored_done'],
            })}
        return {'content': json.dumps({'name': 'Enhanced', 'parameters': []})}
    bridge.host_call = host
    body = {'run_id': run['run_id'], 'enhance': True}
    if not full_authoring:
        body['function'] = saved['function']
    result = bridge._save_function('enhance', body)
    assert result['success'] is True, result
    assert (tmp_path / 'transfer_states.json').read_bytes() == snapshot


def test_agent_authors_proposal_without_hidden_model_or_second_writer(tmp_path):
    bridge = JsonLineBridge(tmp_path / 'store.json', reader=StringIO(), writer=StringIO())
    def forbidden_model(*args):
        raise AssertionError('The host Agent owns authoring; compiler must not call a model')
    bridge.host_call = forbidden_model
    prepared = bridge._save_function('prepare', {'run_log': _run_log(), 'prepare_authoring': True})
    assert prepared['success'] is True
    assert prepared['source_run']['steps'][0]['source_step_index'] == 0
    assert 'patches' in prepared['authoring_contract']
    assert not bridge.flow.store.functions
    proposal = {
        'binding_owner': 'agent', 'reason': 'Visible Done control is stable.',
        'semantic_analysis': {'steps': [{'source_step_index': 0, 'semantic_kind': 'stable',
            'parameter_names': [], 'reason': 'Fixed Done control.'}]},
        'functions': [], 'complete_function': {'function_id': 'click_done', 'name': 'Click Done',
            'description': 'Click the visible Done control.', 'source_step_indices': [0],
            'execution_mode': 'direct_replay', 'parameters': []},
    }
    result = bridge._save_function('compile', {'run_log': _run_log(), 'authoring_proposal': proposal})
    assert result['success'] is True
    assert result['function']['function_id'] == 'click_done'
    assert result['function']['source_run_id'] == _run_log()['run_id']


def test_invalid_agent_proposal_returns_feedback_without_registering_fallback(tmp_path):
    bridge = JsonLineBridge(tmp_path / 'store.json', reader=StringIO(), writer=StringIO())
    bridge.host_call = lambda *args: pytest.fail('Unexpected hidden model call')
    result = bridge._save_function('invalid', {'run_log': _run_log(), 'authoring_proposal': {'reason': 'missing evidence'}})
    assert result['success'] is False
    assert not bridge.flow.store.functions
    assert 'error' in result


def test_agent_authoring_schema_and_mutually_exclusive_inputs(tmp_path):
    from omniflow.bridge import _management_tool_definition
    properties = _management_tool_definition('save_function')['inputSchema']['properties']
    assert properties['prepare_authoring']['type'] == 'boolean'
    assert properties['authoring_proposal']['type'] == 'object'
    bridge = JsonLineBridge(tmp_path / 'store.json', reader=StringIO(), writer=StringIO())
    bridge.host_call = lambda *args: pytest.fail('Unexpected hidden model call')
    for flags in ({'enhance': True, 'authoring_proposal': {}},
                  {'prepare_authoring': True, 'authoring_proposal': {}},
                  {'prepare_authoring': True, 'enhance': True}):
        assert bridge._save_function('invalid', {'run_log': _run_log(), **flags})['success'] is False
    assert not bridge.flow.store.functions
