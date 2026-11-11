"""Summarize exact-state interventions without equating oracle layers with runtime."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean


LABELS = ('source_both', 'target_lower', 'target_upper', 'target_both')


def summarize(report):
    rows, p = report['cases'], report['protocol']
    if report['status'] != 'completed' or any(r['status'] != 'completed' for r in rows):
        raise ValueError('Incomplete run')
    if [r['id'] for r in rows] != p['question_ids'] or len(rows) != p['limit']:
        raise ValueError('Question coverage changed')
    if len(set(p['question_ids'])) != len(rows):
        raise ValueError('Duplicate questions')
    for r in rows:
        if r['split'] != 'dev' or r['condition'] != 'P2':
            raise ValueError('Unexpected split/condition')
        if [i['start'] for i in r['intervals']] != p['starts']:
            raise ValueError('Interval coverage changed')
        if not all(r['baseline_reproduction'].values()):
            raise ValueError('Previous baselines did not reproduce; cross-run comparison needs review')
        for key, s in r['scored_masks'].items():
            if len(key) != r['layers'] or not set(key) <= {'S', 'T'} or s['target_layers'] != key.count('T'):
                raise ValueError('Invalid provenance mask')
            if s['tokens'] != len(s['generated_ids']):
                raise ValueError('Invalid token count')
            for m in ('f1', 'em', 'argmax_agreement'):
                if not math.isfinite(s[m]) or not 0 <= s[m] <= 1:
                    raise ValueError('Invalid score')
        for i in r['intervals']:
            if i['width'] != p['width'] or set(i['variants']) != set(LABELS):
                raise ValueError('Interventions changed')
            for label, key in i['variants'].items():
                if key not in r['scored_masks']:
                    raise ValueError('Missing mask result')
                start, end = i['start'], i['start'] + i['width']
                expected = (('T' if label in ('target_lower', 'target_both') else 'S') * start
                            + 'T' * i['width']
                            + ('T' if label in ('target_upper', 'target_both') else 'S') * (r['layers'] - end))
                if key != expected:
                    raise ValueError('Mask does not implement named intervention')
    result = {'scope': 'Exploratory exact-state swaps on eight fixed development questions; no new quality gate or speed result',
              'question_count': len(rows), 'unique_cache_masks_scored': sum(len(r['scored_masks']) for r in rows),
              'baseline_generated_sequences_reproduced': True,
              'exact_vs_recomputed_same_sequences': sum(i['exact_vs_previous_recomputed']['same_generated_ids'] for r in rows for i in r['intervals']),
              'exact_vs_recomputed_comparisons': sum(len(r['intervals']) for r in rows),
              'native_f1': fmean(r['scored_masks'][r['native_key']]['f1'] for r in rows),
              'direct_source_f1': fmean(r['scored_masks'][r['source_key']]['f1'] for r in rows),
              'intervals': [], 'changed_sequences_from_recomputation': [],
              'interpretation': 'More target layers use additional privileged full-forward state. No equal-budget, deployability or timing claim.'}
    for start in p['starts']:
        intervals = [next(i for i in r['intervals'] if i['start'] == start) for r in rows]
        entry = {'start': start, 'width': p['width'], 'variants': {}}
        for label in LABELS:
            scores = [r['scored_masks'][i['variants'][label]] for r, i in zip(rows, intervals)]
            baseline = [r['scored_masks'][i['variants']['source_both']] for r, i in zip(rows, intervals)]
            entry['variants'][label] = {
                'target_layers': sorted(set(s['target_layers'] for s in scores)),
                'f1': fmean(s['f1'] for s in scores), 'em': fmean(s['em'] for s in scores),
                'kl': fmean(s['kl_reference_to_variant'] for s in scores),
                'f1_improved_vs_interval_only': sum(s['f1'] > b['f1'] for s, b in zip(scores, baseline)),
                'f1_worsened_vs_interval_only': sum(s['f1'] < b['f1'] for s, b in zip(scores, baseline))}
        result['intervals'].append(entry)
        for r, i in zip(rows, intervals):
            if not i['exact_vs_previous_recomputed']['same_generated_ids']:
                result['changed_sequences_from_recomputation'].append({
                    'id': r['id'], 'start': start, 'gold': r['gold_answer'],
                    'exact': r['scored_masks'][i['variants']['source_both']],
                    'previous': i['exact_vs_previous_recomputed']})
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError('Do not overwrite analysis')
    data = Path(args.input).read_bytes()
    result = summarize(json.loads(data))
    result['input_sha256'] = hashlib.sha256(data).hexdigest()
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('intervals', 'changed_sequences_from_recomputation')}))
