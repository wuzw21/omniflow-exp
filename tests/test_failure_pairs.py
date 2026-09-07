import asyncio
import json
from pathlib import Path

from omniflow.core.config import PluginSet
from omniflow.core.model import Action, Observation, TransferResult
from omniflow.runtime.core import prepare_action
from omniflow.transfer.errors import attempt_transfer
from omniflow.runtime.timing import TimingLedger


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
