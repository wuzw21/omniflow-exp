"""Sanitized format audit of real local replay; never emit request/response text."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re

os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', type=Path, required=True)
    args = p.parse_args()
    report = json.loads((args.result / 'summary.json').read_text())
    assert report['status'] == 'completed'
    trace = Path(report['trace'])
    raw_bytes = (trace / 'output.json').read_bytes()
    assert hashlib.sha256(raw_bytes).hexdigest() == report['raw_output_sha256']
    raw = json.loads(raw_bytes)
    source = Path.home() / '.agent-reach/research/papillon-594e7eb'
    for item in report['source']['files']:
        assert hashlib.sha256((source / item['path']).read_bytes()).hexdigest() == item['sha256']
    import dspy
    import json_repair
    spec = importlib.util.spec_from_file_location('pinned_papillon', source / 'papillon/run_llama_dspy.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    signature = module.PAPILLON(None).info_aggregator.signature
    parsed, rows = {}, []
    for r in raw['runs']:
        adapter = dspy.JSONAdapter() if r['mode'] == 'json' else dspy.ChatAdapter()
        final = adapter.parse(signature, r['text'])['finalOutput']
        parsed[r['case'], r['mode']] = final
        row = {'case': r['case'], 'mode': r['mode'], 'raw_chars': len(r['text']),
               'final_chars': len(final), 'output_tokens': r['output_tokens']}
        if r['mode'] == 'json':
            value = json_repair.loads(r['text'])
            row['json_top_level_keys'] = sorted(value) if isinstance(value, dict) else None
        rows.append(row)
    comparisons = []
    for case in ['p95', 'median']:
        a, b = parsed[case, 'full_before'], parsed[case, 'json']
        normalize = lambda text: re.findall(r'\w+', text.lower())
        comparisons.append({'case': case, 'same_parsed_final': a == b,
                            'same_casefold_word_sequence': normalize(a) == normalize(b),
                            'quality_evaluated': False})
    out = {'scope': 'Format/content identity only, not task quality', 'rows': rows,
           'comparisons': comparisons, 'audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.result / 'format_audit.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
