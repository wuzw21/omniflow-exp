"""Measure exact copied source token runs; not semantic attribution or privacy scoring."""
import argparse
import difflib
import json
from pathlib import Path
from probe_minions_collaboration import SOURCE


def spans(source, output, minimum=16):
    return [{'source_start': b.a, 'output_start': b.b, 'tokens': b.size}
            for b in difflib.SequenceMatcher(None, source, output, autojunk=False).get_matching_blocks()
            if b.size >= minimum]


def main():
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    evidence = json.loads((args.directory / 'evidence.json').read_text())
    tok = AutoTokenizer.from_pretrained(evidence['endpoints']['local']['model']['path'], local_files_only=True)
    rows = [json.loads(s) for s in (args.directory / 'calls.jsonl').read_text().splitlines()]
    results = []
    for i, row in enumerate(rows):
        if row['endpoint'] != 'local' or not row['phase'].endswith('/minion'):
            continue
        case = row['phase'].split('/')[0]
        source = tok.encode((SOURCE / f'minions/examples/{case}/sample.txt').read_text(), add_special_tokens=False)
        output = row['result']['generated_ids']
        matches = spans(source, output)
        results.append({'call_index': i, 'case': case, 'minimum_contiguous_match_tokens': 16,
                        'source_tokens': len(source), 'generated_tokens': len(output),
                        'matched_tokens': sum(m['tokens'] for m in matches), 'spans': matches,
                        'limits': 'Ordered exact matches only; repeated text may align to any occurrence; excludes matches shorter than threshold; not semantic-copy or leakage metric'})
    (args.directory / 'copy_spans.json').write_text(json.dumps(results, indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k not in ('spans','limits')} for r in results], indent=2))


if __name__ == '__main__':
    main()
