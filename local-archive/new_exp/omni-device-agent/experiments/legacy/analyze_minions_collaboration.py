"""Trace accounting only: eligibility is not a measured cache hit or speedup."""
import argparse
import ast
import json
from pathlib import Path
from probe_minions_collaboration import SOURCE, lcp, load_upstream


def analyze(directory):
    evidence = json.loads((directory / 'evidence.json').read_text())
    rows = [json.loads(s) for s in (directory / 'calls.jsonl').read_text().splitlines()]
    document = (SOURCE / 'minions/examples/code/sample.txt').read_text()
    last = {}
    totals = {}
    calls = []
    for i, row in enumerate(rows):
        key = row['phase'], row['endpoint']
        out = row['result']
        prefix = lcp(last.get(key, []), out['input_ids'])
        assert prefix == row['eligible_own_prefix_tokens']
        assert row['measured_cache_hits'] == 0
        assert out['first_forward_input_tokens'] == len(out['input_ids'])
        last[key] = out['input_ids'] + out['generated_ids'][:-1]
        record = {'index': i, 'phase': row['phase'], 'endpoint': row['endpoint'],
                  'input_tokens': len(out['input_ids']), 'output_tokens': len(out['generated_ids']),
                  'eligible_own_prefix_tokens': prefix,
                  'new_input_if_exact_own_cache_retained': len(out['input_ids']) - prefix,
                  'first_forward_seconds': out['first_forward_seconds'],
                  'generation_seconds': out['generation_seconds'],
                  'request_wall_seconds': row['request_wall_seconds'],
                  'finish_reason': out['finish_reason'],
                  'contains_entire_file': any(document in m['content'] for m in row['messages'])}
        calls.append(record)
        total = totals.setdefault('/'.join(key), {f: 0 for f in
                                  ('calls', 'input_tokens', 'output_tokens', 'eligible_own_prefix_tokens',
                                   'first_forward_seconds', 'generation_seconds', 'request_wall_seconds')})
        total['calls'] += 1
        for f in total:
            if f != 'calls':
                total[f] += record[f]
    # Literal transfers in the protocol, preserving call ordering and repetitions.
    transfers = []
    intra_endpoint_copies = []
    Minion, _, _ = load_upstream()
    parse_json = Minion.__call__.__globals__['_extract_json']
    prior = {'local': [], 'remote': []}
    for i, row in enumerate(rows):
        if row['phase'] != 'minion':
            continue
        endpoint = row['endpoint']
        opposite = 'local' if endpoint == 'remote' else 'remote'
        for source_index, source_text in prior[opposite]:
            occurrences = sum(m['content'].count(source_text) for m in row['messages']) if source_text else 0
            if occurrences:
                transfers.append({'source_call': source_index, 'target_call': i,
                                  'kind': 'whole_generated_response',
                                  'text_utf8_bytes': len(source_text.encode()),
                                  'literal_occurrences_in_target_request': occurrences})
            if endpoint == 'local':
                # Supervisor sends the message field, not its whole JSON response.
                parsed = parse_json(source_text)
                message = parsed.get('message', '')
                occurrences = sum(m['content'].count(message) for m in row['messages']) if message else 0
                if occurrences:
                    transfers.append({'source_call': source_index, 'target_call': i,
                                      'kind': 'supervisor_message_field',
                                      'text_utf8_bytes': len(message.encode()),
                                      'literal_occurrences_in_target_request': occurrences})
        if prior[endpoint]:
            source_index, source_text = prior[endpoint][-1]
            if source_text and source_text in row['messages'][-1]['content']:
                intra_endpoint_copies.append({'source_call': source_index, 'target_call': i,
                                             'text_utf8_bytes': len(source_text.encode())})
        prior[endpoint].append((i, row['result']['text']))
    fixture = json.loads((SOURCE / 'minions/examples/code/task.json').read_text())
    methods = [n.name for n in ast.walk(ast.parse(document)) if isinstance(n, ast.FunctionDef)]
    answers = {'minion': evidence.get('minion_result', {}).get('final_answer', '')}
    answers.update(evidence.get('direct', {}))
    return {'status': evidence['status'], 'calls': calls, 'totals': totals,
            'literal_cross_endpoint_text': transfers,
            'previous_output_copied_into_last_input_message': intra_endpoint_copies,
            'fixture_bytes': len(document.encode()), 'reference_methods': fixture['answer'],
            'answer_method_mentions': {name: [m for m in methods if m in answer] for name, answer in answers.items()},
            'answers': answers,
            'limits': ['One public upstream example; local model necessity not established',
                       'Method mentions are not a task accuracy metric; inspect claims',
                       'Own-prefix counts are ideal eligibility, not measured serving cache savings',
                       'Fresh-prefill reference adapters, no serving-engine comparison',
                       'First-forward timing is diagnostic, includes cold-start effects; no repeated benchmark',
                       'No privacy guarantee, cross-model state transfer, or end-to-end speedup measured']}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    args = p.parse_args()
    result = analyze(args.directory)
    (args.directory / 'analysis.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))
