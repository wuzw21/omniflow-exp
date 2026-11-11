"""Dataset-wide offline prompt/token audit; stores aggregates, never raw requests.

Run with the tokenizer environment; --formatter-python supplies the isolated
DSPy environment. No model generation, optimized prompt, or timing is claimed.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess


def stats(values):
    values = sorted(values)
    assert values
    return {'count': len(values), 'sum': sum(values), 'mean': statistics.fmean(values),
            'min': values[0], 'median': statistics.median(values),
            'p90': values[math.ceil(.90 * len(values)) - 1],
            'p95': values[math.ceil(.95 * len(values)) - 1], 'max': values[-1]}


def lcp(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def query_span(tokenizer, messages, query):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    start = text.index(query, text.index('<|im_start|>user'))
    end = start + len(query)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded['input_ids']
    assert ids == tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                return_dict=False)
    # Exclude tokens crossing the query boundary; this is an exact interior-token count.
    positions = [i for i, (s, e) in enumerate(encoded['offset_mapping'])
                 if s >= start and e <= end and e > s]
    assert positions and positions == list(range(positions[0], positions[-1] + 1))
    return ids, positions[0], ids[positions[0]:positions[-1] + 1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--formatter-python', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from transformers import AutoTokenizer, __version__ as transformers_version
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    data_root = args.source / 'pupa'
    manifest = json.loads((data_root / 'manifest.json').read_text())
    worker = Path(__file__).with_name('papillon_format_worker.py')
    proc = subprocess.Popen([args.formatter_python, '-u', str(worker), '--source', str(args.source)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        ready = json.loads(proc.stdout.readline())
        assert ready['ready']
        result = {'scope': 'Official dataset requests, unoptimized upstream signatures, DSPy ChatAdapter; empty cloud response only for layout. No inference, cache hit, latency or quality claim.',
                  'source': ready['source'], 'dataset': manifest,
                  'versions': {'dspy': ready['dspy'], 'transformers': transformers_version},
                  'tokenizer': str(args.tokenizer),
                  'tokenizer_sha256': hashlib.sha256((args.tokenizer / 'tokenizer.json').read_bytes()).hexdigest(),
                  'scripts_sha256': {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                                     for f in [Path(__file__), worker]}, 'datasets': {}}
        for item in manifest['files']:
            if not item['name'].endswith('.csv'):
                continue
            path = data_root / item['name']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
            measures = {k: [] for k in ['query_tokens', 'creation_prompt_tokens', 'aggregation_min_prompt_tokens',
                                        'cross_operator_prefix_tokens', 'repeated_query_interior_tokens',
                                        'creation_query_start', 'aggregation_query_start']}
            categories = Counter()
            with path.open() as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == item['rows']
            for row in rows:
                q = row['user_query']
                assert q.strip(), 'Do not silently drop empty requests'
                proc.stdin.write(json.dumps({'query': q}) + '\n'); proc.stdin.flush()
                rendered = json.loads(proc.stdout.readline())
                a, apos, aq = query_span(tokenizer, rendered['creation'], q)
                b, bpos, bq = query_span(tokenizer, rendered['aggregation_empty_response'], q)
                assert aq == bq, 'Repeated query interior differs after formatting'
                common = lcp(a, b)
                assert common < min(apos, bpos), 'Prefix unexpectedly reaches the query'
                values = [len(tokenizer.encode(q, add_special_tokens=False)), len(a), len(b),
                          common, len(aq), apos, bpos]
                for key, value in zip(measures, values, strict=True):
                    measures[key].append(value)
                categories[row['predicted_category']] += 1
            query = measures['query_tokens']
            result['datasets'][item['name']] = {
                'stats': {k: stats(v) for k, v in measures.items()},
                'query_length_counts': {str(n): sum(v >= n for v in query) for n in [512, 1024, 2048, 4096, 8192]},
                'categories': dict(categories),
                'hypothetical_second_query_reuse_fraction_of_two_local_inputs':
                    sum(measures['repeated_query_interior_tokens']) /
                    (sum(measures['creation_prompt_tokens']) + sum(measures['aggregation_min_prompt_tokens']))}
        proc.stdin.close()
        proc.wait(timeout=30)
        assert proc.returncode == 0
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
        print(json.dumps(result['datasets'], ensure_ascii=False, indent=2))
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=15)


if __name__ == '__main__':
    main()
