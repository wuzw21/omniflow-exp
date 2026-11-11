"""Account for a complete or partial real trace; never equate protocol return with quality."""
import argparse
import json
from pathlib import Path
from probe_minions_collaboration import SOURCE, lcp


def analyze(directory):
    e = json.loads((directory / 'evidence.json').read_text())
    rows = [json.loads(s) for s in (directory / 'calls.jsonl').read_text().splitlines()]
    last, cases, calls = {}, {}, []
    for i, row in enumerate(rows):
        case, mode = row['phase'].split('/')
        endpoint = row['endpoint']
        key = case, mode, endpoint
        out = row['result']
        prefix = lcp(last.get(key, []), out['input_ids'])
        assert prefix == row['eligible_own_prefix_tokens']
        assert row['measured_cache_hits'] == 0
        last[key] = out['input_ids'] + out['generated_ids'][:-1]
        chunks = out['prefill_ranges']
        cursor = 0
        for start, end in chunks:
            assert start == cursor and end > start
            cursor = end
        assert cursor + out['first_forward_input_tokens'] == len(out['input_ids'])
        if out['first_forward_seconds'] is not None:
            assert abs(out['total_prefill_seconds']-out['preceding_prefill_seconds']-out['first_forward_seconds']) < 1e-6
        doc = (SOURCE / f'minions/examples/{case}/sample.txt').read_text()
        c = {'index': i, 'case': case, 'mode': mode, 'endpoint': endpoint,
             'input_tokens': len(out['input_ids']), 'output_tokens': len(out['generated_ids']),
             'eligible_own_prefix_tokens': prefix,
             'fresh_tokens_with_ideal_own_cache': len(out['input_ids'])-prefix,
             'prefill_seconds': out['total_prefill_seconds'],
             'generation_seconds': out['generation_seconds'],
             'remaining_generation_seconds': out['generation_seconds']-out['total_prefill_seconds'],
             'request_wall_seconds': row['request_wall_seconds'],
             'finish_reason': out['finish_reason'],
             'contains_full_file': any(doc in m['content'] for m in row['messages']),
             'output_utf8_bytes': len(out['text'].encode())}
        calls.append(c)
        t = cases.setdefault(case, {}).setdefault(mode + '/' + endpoint,
             {f: 0 for f in ['calls', 'input_tokens', 'output_tokens', 'eligible_own_prefix_tokens',
                            'prefill_seconds', 'generation_seconds', 'request_wall_seconds']})
        t['calls'] += 1
        for f in t:
            if f != 'calls':
                t[f] += c[f]
    answers = {}
    for case, record in e['cases'].items():
        answers[case] = {'protocol_status': record['status'],
                        'collaboration': record.get('minion_result', {}).get('final_answer'),
                        'direct': record['direct'], 'no_document': record.get('no_document', {}),
                        'upstream_reference': json.loads((SOURCE / f'minions/examples/{case}/task.json').read_text())['answer']}
    return {'execution_status': e['status'], 'calls': calls, 'totals_by_case': cases,
            'answers_for_adjudication': answers,
            'limits': ['Three predefined upstream examples, not benchmark accuracy',
                       'No prefix retention across requests: eligibility is not measured cache benefit',
                       'Chunked prefill and single-run instrumentation are not optimized serving performance',
                       'Different local model and execution mode from 0.5B run: not isolated size ablation',
                       'Cloud 7B is a controlled proxy, not a frontier model API',
                       'Public files do not establish local deployment necessity or privacy']}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    args = p.parse_args()
    result = analyze(args.directory)
    (args.directory / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
