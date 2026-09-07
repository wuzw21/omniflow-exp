import asyncio
import json
from pathlib import Path
import pytest

from omniflow.core.config import PluginSet
from omniflow.core.model import Action, Observation, TransferResult
from omniflow.runtime.core import prepare_action
from omniflow.transfer.errors import attempt_transfer
from omniflow.runtime.timing import TimingLedger
from omniflow.transfer.failure_inputs import load_failure_inputs, replay_failure_inputs


def test_failed_mapping_and_stable_retry_share_pair_owner(tmp_path, monkeypatch):
    path = tmp_path/'pairs.jsonl'
    monkeypatch.setenv('OMNIFLOW_TRANSFER_ERROR_POOL', str(path))
    monkeypatch.setenv('OMNIFLOW_TRANSFER_ERROR_CONTEXT', json.dumps({
        'task': 'test', 'payload': 'PRIVATE_CONTEXT'*10000}))
    source = Observation(xml='<hierarchy/>', extra={'display': {'width': 100, 'height': 200}})
    target = Observation(xml='<hierarchy><node/></hierarchy>')
    action = Action('click', {'x': 20, 'y': 30})
    class Host:
        async def observe_stable(self, **kwargs):
            return target
    def mapper(*args):
        return TransferResult(None, reason='target_candidate_not_executable', detail={
            'raw_xml': 'PRIVATE_XML'*10000, 'image_base64': 'PRIVATE_IMAGE'*10000,
            'candidates': [{'score': .1, 'executable': False, 'raw_xml': 'private'}]*100})
    async def scenario():
        decision = await prepare_action(action, observation=target, source_state=source,
                                        host=Host(), plugins=PluginSet(transfer=mapper))
        assert decision.action is None
        await attempt_transfer(mapper, action, target, source, phase='recall.entry')
    ledger = TimingLedger()
    with ledger.span('execution.other'):
        asyncio.run(scenario())
    components = ledger.report()['components']
    assert components['transfer']['calls'] == 3
    assert components['observe.stable']['calls'] == 1
    records = [json.loads(line) for line in path.read_text().splitlines()]
    from jsonschema import validate
    schema = json.loads((Path(__file__).parents[1]/'schemas/omniflow_failure_pair.v1.json').read_text())
    for record in records:
        validate(record, schema)
    assert [r['phase'] for r in records] == ['execute.fast', 'execute.stable', 'recall.entry']
    assert len({r['pair_id'] for r in records}) == 1
    assert all(r['context']['task'] == 'test' and 'context_sha256' in r['context'] for r in records)
    assert all(len(r['transfer_detail']['candidates']) == 8 for r in records)
    assert 'PRIVATE_' not in path.read_text() and '<hierarchy' not in path.read_text()
    assert path.stat().st_size < 12000
    assert all(r['replay']['status'] == 'ready' for r in records)
    assert len({r['replay']['path'] for r in records}) == 1


def test_failure_input_replay_preserves_full_pair_and_deduplicates_assets(tmp_path, monkeypatch):
    pool = tmp_path/'pairs.jsonl'
    monkeypatch.setenv('OMNIFLOW_TRANSFER_ERROR_POOL', str(pool))
    screenshot = tmp_path/'screen.png'
    screenshot.write_bytes(b'original-image-bytes')
    source = Observation(xml='<source/>', package_name='app', image_base64='exact-base64',
        extra={'display': {'width': 100, 'height': 200}, 'screenshot_path': str(screenshot),
               'adapter_context': {'stable': True},
               'androidworld_state': {'screenshot': {'path': str(screenshot)}, 'xml': '<source/>'}})
    target = Observation(xml='<target/>', extra={'androidworld_state': {'pixels': {'path': str(screenshot)}}})
    action = Action('click', {'x': 5, 'y': 7})
    def reject(action, target, source):
        return TransferResult(None, reason='low_confidence')
    asyncio.run(attempt_transfer(reject, action, target, source, phase='execute.fast'))
    record = json.loads(pool.read_text())
    path = pool.parent / record['replay']['path']
    import gzip
    from jsonschema import validate
    validate(json.loads(gzip.decompress(path.read_bytes())), json.loads(
        (Path(__file__).parents[1]/'schemas/omniflow_failure_inputs.v1.json').read_text()))
    assets_before = set(path.parent.iterdir())
    screenshot.unlink()
    loaded_action, loaded_target, loaded_source = load_failure_inputs(path)
    assert loaded_action == action
    assert loaded_source.xml == source.xml and loaded_target.xml == target.xml
    assert loaded_source.image_base64 == source.image_base64
    assert loaded_source.extra['adapter_context'] == {'stable': True}
    source_image = Path(loaded_source.extra['screenshot_path'])
    assert source_image.read_bytes() == b'original-image-bytes'
    assert loaded_target.extra['androidworld_state']['pixels']['path'] == str(source_image)
    assert loaded_source.extra['androidworld_state']['screenshot']['path'] == str(source_image)
    result = asyncio.run(replay_failure_inputs(path, transfer=reject))
    assert result.action is None
    assert set(path.parent.iterdir()) == assets_before
    records = [json.loads(line) for line in pool.read_text().splitlines()]
    assert records[-1]['phase'] == 'replay'
    source_image.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='hash_mismatch'):
        load_failure_inputs(path)


def test_missing_pair_asset_keeps_failure_but_does_not_claim_replay_ready(tmp_path, monkeypatch):
    pool = tmp_path/'pairs.jsonl'
    monkeypatch.setenv('OMNIFLOW_TRANSFER_ERROR_POOL', str(pool))
    source = Observation(xml='<source/>', extra={'screenshot_path': str(tmp_path/'missing.png')})
    result = asyncio.run(attempt_transfer(lambda *args: TransferResult(None, reason='rejected'),
        Action('click', {'x': 1, 'y': 2}), Observation(xml='<target/>'), source, phase='execute.fast'))
    assert result.reason == 'rejected'
    assert json.loads(pool.read_text())['replay']['status'] == 'unavailable'


def test_large_pair_input_does_not_change_mapping_failure(tmp_path, monkeypatch):
    from omniflow.transfer import failure_inputs
    monkeypatch.setattr(failure_inputs, 'MAX_BLOB_BYTES', 128)
    pool = tmp_path/'pairs.jsonl'
    monkeypatch.setenv('OMNIFLOW_TRANSFER_ERROR_POOL', str(pool))
    result = asyncio.run(attempt_transfer(lambda *args: TransferResult(None, reason='rejected'),
        Action('click', {'x': 1, 'y': 2}), Observation(xml='x'*1000), None, phase='execute.fast'))
    assert result.reason == 'rejected'
    assert json.loads(pool.read_text())['replay']['status'] == 'unavailable'


def test_transfer_audit_counts_attempt_results_not_error_prefixes():
    from src.experiment.run_task import _execution_audit
    def row(tool, success, attempts):
        return {'action': {'tool': tool, 'args': {}},
                'result': {'success': success, 'error': 'custom_failure' if not success else None},
                'metadata': {'function_id': 'example', 'transfer': {'transfer_attempts': attempts}}}
    report = _execution_audit({'execution_trace': [
        row('open_app', True, []),
        row('click', False, [{'path': 'fast', 'mapped': False}, {'path': 'stable', 'mapped': False}]),
        row('click', False, [{'path': 'fast', 'mapped': True}]),
    ]})
    assert report['transfer_attempts'] == 3
    assert report['transfer_success'] == 1 and report['transfer_failure'] == 2


@pytest.mark.parametrize('accepted', [True, False])
@pytest.mark.parametrize('entry', ['core', 'robust'])
def test_stable_mapping_state_is_used_for_dispatch_evidence_and_recovery(accepted, entry):
    from omniflow.runtime.core import execute_action
    from omniflow.runtime.execution import execute_robust_action
    from omniflow.core.model import ActionResult
    original = Observation(xml='<old/>')
    stable = Observation(xml='<stable/>')
    after = Observation(xml='<after/>')
    mapped = Action('click', {'x': 10, 'y': 20})
    seen, dispatched = [], []
    class Host:
        async def observe_stable(self, **kwargs):
            return stable
        async def observe(self, **kwargs):
            return after
        async def act(self, action):
            dispatched.append(action)
            return ActionResult(True)
    def transfer(action, target, source):
        seen.append(target)
        return TransferResult(mapped if accepted and target is stable else None, reason='probe')
    callback = execute_action if entry == 'core' else execute_robust_action
    result = asyncio.run(callback(Action('click', {'x': 900, 'y': 900}), observation=original,
        host=Host(), plugins=PluginSet(transfer=transfer), source_state=Observation(xml='<source/>')))
    assert seen == [original, stable]
    assert result.before is stable
    assert result.success == accepted
    assert dispatched == ([mapped] if accepted else [])
    assert result.after is (after if accepted else None)


def test_mapping_exception_is_failure_with_pair_and_no_source_coordinate_replay(tmp_path, monkeypatch):
    path = tmp_path/'pairs.jsonl'
    monkeypatch.setenv('OMNIFLOW_TRANSFER_ERROR_POOL', str(path))
    def mapper(*args):
        raise ValueError('internal large request or payload')
    result = asyncio.run(attempt_transfer(mapper, Action('click', {'x': 1, 'y': 2}),
        Observation(xml='<target/>'), Observation(xml='<source/>'), phase='execute.fast'))
    assert result.action is None and result.reason == 'transfer_exception:ValueError'
    record = json.loads(path.read_text())
    assert record['page_pair']['complete'] and record['transfer_detail']['exception_type'] == 'ValueError'
    assert 'internal large' not in path.read_text()
