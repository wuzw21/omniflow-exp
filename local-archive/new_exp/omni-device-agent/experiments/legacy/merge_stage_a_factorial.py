"""Merge completed disjoint question shards; never promote partial output to success."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def merge_reports(reports):
    if not reports:
        raise ValueError('No shards')
    base = reports[0]
    common = {k: v for k, v in base['protocol'].items() if k != 'shard'}
    count = base['protocol']['shard']['count']
    if len(reports) != count:
        raise ValueError('Missing shards')
    indices, rows = set(), {}
    for report in reports:
        if report['status'] != 'completed' or any(r['status'] != 'completed' for r in report['cases']):
            raise ValueError('Incomplete shard')
        p = report['protocol']
        if {k: v for k, v in p.items() if k != 'shard'} != common:
            raise ValueError('Different protocols')
        shard = p['shard']
        index = shard['index']
        if shard['count'] != count or type(index) is not int or not 0 <= index < count or index in indices:
            raise ValueError('Invalid or duplicate shard index')
        indices.add(index)
        if [r['id'] for r in report['cases']] != common['question_ids'][index::count]:
            raise ValueError('Shard order or coverage changed')
        for field in ('models', 'source_sha256', 'data_manifest_sha256'):
            if report[field] != base[field]:
                raise ValueError(f'Different {field}')
        for row in report['cases']:
            if row['id'] in rows:
                raise ValueError('Repeated question')
            rows[row['id']] = row
    if len(rows) != common['limit'] or set(rows) != set(common['question_ids']):
        raise ValueError('Full question coverage not achieved')
    result = {k: copy.deepcopy(base[k]) for k in ('models', 'source_sha256', 'data_manifest_sha256')}
    result.update(status='completed', scope='Merged full P2 development diagnostics; not the full P0/P1/P2 or held-out gate',
                  protocol=common, runtime={'shards': [r['runtime'] for r in reports]},
                  settings={'shards': [r['settings'] for r in reports]},
                  cases=[copy.deepcopy(rows[q]) for q in common['question_ids']])
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', nargs='+', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Never overwrite merged evidence')
    data = [Path(f).read_bytes() for f in args.inputs]
    result = merge_reports([json.loads(d) for d in data])
    result['shard_input_sha256'] = {name: hashlib.sha256(d).hexdigest() for name, d in zip(args.inputs, data)}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'merged_cases': len(result['cases']), 'output': str(output)}))
