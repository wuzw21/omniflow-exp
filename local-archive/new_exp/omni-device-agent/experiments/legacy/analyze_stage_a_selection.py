"""Gold-selected per-question headroom is an unattainable diagnostic, not an online policy."""
import argparse
import json
from pathlib import Path
from statistics import fmean

from analyze_stage_a_factorial import summarize


def selection_headroom(report, label):
    rows, starts = report['cases'], report['protocol']['starts']
    if not rows or report['status'] != 'completed' or any(r['status'] != 'completed' for r in rows):
        raise ValueError('Need completed cases')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate cases')
    scores = []
    for r in rows:
        if [i['start'] for i in r['intervals']] != starts:
            raise ValueError('Incomplete or reordered intervals')
        values = []
        for i in r['intervals']:
            s = i['diagnostics']['exact_joint_source_outside'] if label == 'exact_joint' else i['scores'][label]
            if not 0 <= s['f1'] <= 1:
                raise ValueError('Invalid F1')
            values.append(s['f1'])
        scores.append(values)
    means = [fmean(row[j] for row in scores) for j in range(len(starts))]
    best = max(range(len(starts)), key=lambda j: (means[j], -starts[j]))
    oracle = [max(row) for row in scores]
    return {
        'label': label, 'question_count': len(rows),
        'development_static_start': starts[best], 'development_static_mean_f1': means[best],
        'per_question_gold_oracle_mean_f1': fmean(oracle),
        'oracle_minus_development_static': fmean(oracle) - means[best],
        'questions_with_strict_improvement_over_static': sum(v > row[best] for v, row in zip(oracle, scores)),
        'means_by_start': dict(zip(map(str, starts), means)),
        'interpretation': 'Both selected and evaluated on the same development questions; optimistic descriptive comparison, not generalization or an online algorithm. Gold oracle uses unavailable answer labels.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Never overwrite an analysis')
    report = json.loads(Path(args.input).read_text())
    summarize(report, draws=10)  # Reuse strict full-coverage/metric validation; no claims from these draws.
    labels = ['source', 'model_oracle', 'additive_oracle', 'joint_oracle']
    if report['protocol'].get('exact_joint_diagnostic', False):
        labels.append('exact_joint')
    result = {'scope': 'Development-only layer-choice headroom diagnostic; no policy, no frozen interval',
              'comparisons': [selection_headroom(report, label) for label in labels]}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))
