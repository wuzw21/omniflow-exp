"""Descriptive fixed-target diagnostics, never select an interval from this pilot."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean

from analyze_stage_a_development import paired_delta, recovery_fraction

ENTRY_NAMES = ('source', 'context_oracle', 'model_oracle', 'additive_oracle', 'joint_oracle')
METRICS = ('f1', 'em', 'kl_reference_to_variant', 'argmax_agreement')


def summarize(report, draws=10000):
    p, rows = report['protocol'], report['cases']
    if report['status'] != 'completed' or any(r['status'] != 'completed' for r in rows):
        raise ValueError('Incomplete execution')
    if [r['id'] for r in rows] != p['question_ids'] or len(rows) != p['limit']:
        raise ValueError('Frozen question coverage differs')
    if len(set(p['question_ids'])) != len(rows):
        raise ValueError('Duplicate question')
    for r in rows:
        if r['split'] != 'dev' or r['condition'] != 'P2':
            raise ValueError('Held-out or unconfigured condition')
        if set(r['cache_factorial']) != {'As', 'At', 'Bs', 'Bt'}:
            raise ValueError('Missing factorial cell')
        if [i['start'] for i in r['intervals']] != p['starts']:
            raise ValueError('Incomplete or reordered interval coverage')
        all_scores = [*r['cache_factorial'].values(), r['native_source_history']]
        for i in r['intervals']:
            if i['width'] != p['width'] or set(i['scores']) != set(ENTRY_NAMES):
                raise ValueError('Missing or changed intervention')
            expected_diagnostics = {'source_native_outside', 'joint_oracle_native_outside'}
            if p.get('exact_joint_diagnostic', False):
                expected_diagnostics.add('exact_joint_source_outside')
            if set(i['diagnostics']) != expected_diagnostics:
                raise ValueError('Missing native-outside diagnostic')
            all_scores += list(i['scores'].values()) + list(i['diagnostics'].values())
        for s in all_scores:
            if any(not math.isfinite(s[m]) for m in METRICS):
                raise ValueError('Nonfinite metric')
            if any(not 0 <= s[m] <= 1 for m in ('f1', 'em', 'argmax_agreement')):
                raise ValueError('Invalid quality metric')
            if s['tokens'] != len(s['generated_ids']) or s['finish_reason'] not in ('stop', 'length'):
                raise ValueError('Generation accounting mismatch')

    def means(scores):
        return {m: fmean(s[m] for s in scores) for m in METRICS}

    def contrast(left, right, metric='f1'):
        a, b = [s[metric] for s in left], [s[metric] for s in right]
        return {**paired_delta(a, b, draws=draws),
                'improved': sum(y > x for x, y in zip(a, b)),
                'worsened': sum(y < x for x, y in zip(a, b)),
                'unchanged': sum(y == x for x, y in zip(a, b))}

    native = [r['cache_factorial']['Bt'] for r in rows]
    history = [r['native_source_history'] for r in rows]
    result = {
        'scope': 'Fixed-target development diagnostic, no model mapping training or speed claim',
        'question_count': len(rows), 'interval_candidate': None, 'held_out_gate_passed': False,
        'factorial_mean': {k: means([r['cache_factorial'][k] for r in rows]) for k in ('As', 'At', 'Bs', 'Bt')},
        'native_source_history_mean': means(history), 'target_minus_source_history': contrast(history, native),
        'signed_factorial_interaction_mean': {m: fmean(r['factorial_interaction'][m] for r in rows)
                                             for m in ('f1', 'em', 'kl_reference_to_variant')},
        'numerical_calibration': {
            'max_split_dense_logit_error': max(r['calibration']['split_dense_relative_logit_error'] for r in rows),
            'max_split_dense_kl': max(r['calibration']['kl_reference_to_variant'] for r in rows),
            'split_dense_cases_with_argmax_disagreement': sum(r['calibration']['argmax_agreement'] < 1 for r in rows)},
        'intervals': [], 'changed_answers': [],
        'statistical_limit': 'Per-interval bootstrap intervals are descriptive, unadjusted for seven intervals, on development questions. Prior pilot cases retained. No confirmatory test or layer selection.'}
    for start in p['starts']:
        intervals = [next(i for i in r['intervals'] if i['start'] == start) for r in rows]
        groups = {k: [i['scores'][k] for i in intervals] for k in ENTRY_NAMES}
        diag = [i['diagnostics']['joint_oracle_native_outside'] for i in intervals]
        result['intervals'].append({
            'start': start, 'width': p['width'], 'mean': {k: means(v) for k, v in groups.items()},
            'joint_minus_source_f1': contrast(groups['source'], groups['joint_oracle']),
            'joint_minus_source_em': contrast(groups['source'], groups['joint_oracle'], 'em'),
            'joint_minus_additive_f1': contrast(groups['additive_oracle'], groups['joint_oracle']),
            'joint_minus_model_f1': contrast(groups['model_oracle'], groups['joint_oracle']),
            'recovery': recovery_fraction(fmean(s['f1'] for s in native), fmean(s['f1'] for s in groups['source']),
                                          fmean(s['f1'] for s in groups['joint_oracle'])),
            'joint_native_outside': {**means(diag),
                                    'identical_native_generated_sequences': sum(d['generated_ids'] == n['generated_ids'] for d, n in zip(diag, native))},
            'entry_error_mean': {k: fmean(i['entry_error'][k] for i in intervals) for k in ENTRY_NAMES},
            'max_joint_interval_kv_error': max(i['scores']['joint_oracle']['interval_kv_error_max'] for i in intervals)})
        if p.get('exact_joint_diagnostic', False):
            exact = [i['diagnostics']['exact_joint_source_outside'] for i in intervals]
            result['intervals'][-1]['exact_joint_diagnostic'] = {
                'mean': means(exact),
                'exact_minus_recomputed_f1': contrast(groups['joint_oracle'], exact),
                'exact_minus_source_f1': contrast(groups['source'], exact),
                'exact_minus_additive_f1': contrast(groups['additive_oracle'], exact),
                'same_generated_sequences': sum(e['generated_ids'] == j['generated_ids']
                                                for e, j in zip(exact, groups['joint_oracle']))}
        for row, i in zip(rows, intervals):
            scores = i['scores']
            if scores['joint_oracle']['generated_ids'] != scores['source']['generated_ids'] or scores['joint_oracle']['generated_ids'] != scores['additive_oracle']['generated_ids']:
                result['changed_answers'].append({'id': row['id'], 'start': start, 'gold': row['gold_answer'],
                    'scores': {k: {m: scores[k][m] for m in ('prediction', 'f1', 'em', 'finish_reason')}
                               for k in ('source', 'additive_oracle', 'joint_oracle')}})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Do not overwrite analysis')
    payload = Path(args.input).read_bytes()
    result = summarize(json.loads(payload))
    result['input_sha256'] = hashlib.sha256(payload).hexdigest()
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'questions': result['question_count'], 'interval_candidate': None}))
