"""Summarize the pinned development screen without claiming a held-out result.

P2 alone determines the source-entry interval candidate. Bootstrap samples
questions, keeping each source/oracle pair together. A candidate is emitted
only after all pinned development IDs have results for every configured
condition and the pre-registered seven four-layer intervals. Numerical review
is still required before freezing it. No rows are filtered by model accuracy.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


def paired_delta(reference, variant, draws=10000, seed=20261024):
    if not reference or len(reference)!=len(variant) or draws<1:
        raise ValueError('Need aligned, nonempty question pairs and positive draws')
    delta=[b-a for a,b in zip(reference,variant)]
    if any(not math.isfinite(v) for v in reference+variant):
        raise ValueError('Nonfinite metric')
    rng=random.Random(seed)
    means=sorted(statistics.fmean(rng.choices(delta,k=len(delta))) for _ in range(draws))
    return {'mean':statistics.fmean(delta),
            'paired_percentile_95ci':[means[int(.025*(draws-1))],means[int(.975*(draws-1))]],
            'question_count':len(delta),'bootstrap_draws':draws,'seed':seed}


def recovery_fraction(native, source, oracle):
    gap=native-source
    return {'native_minus_source':gap,'oracle_minus_source':oracle-source,
            'fraction':(oracle-source)/gap if gap>0 else None,
            'interpretation':'Descriptive ratio only; native superiority requires separate uncertainty assessment.'
                if gap>0 else 'Undefined: native quality does not exceed source-entry quality.'}


def summarize(report, manifest, draws=10000):
    if report['status']!='completed':
        raise ValueError('Cannot analyze an incomplete or failed run')
    dev_ids={q['id'] for q in manifest['questions'] if q['split']=='dev'}
    settings=report['settings']
    conditions=settings['conditions']
    starts=settings['starts']
    width=settings['width']
    if len(set(conditions))!=len(conditions) or len(set(starts))!=len(starts):
        raise ValueError('Duplicate configured conditions or intervals')
    if not conditions or not set(conditions)<=set(('P0','P1','P2')) or not starts or width<1:
        raise ValueError('Invalid condition/interval configuration')
    seen=set()
    for row in report['cases']:
        pair=(row['id'],row['condition'])
        if row['id'] not in dev_ids or row['split']!='dev' or row['condition'] not in conditions or pair in seen:
            raise ValueError('Held-out, duplicate, or unconfigured case')
        seen.add(pair)
        intervals=row['intervals']
        if len(intervals)!=len(starts) or {i['start'] for i in intervals}!=set(starts) or any(i['width']!=width for i in intervals):
            raise ValueError('Interval coverage differs across questions')
        scores=[row['native'],row['direct'],*row['history_controls'].values(),
                *(score for i in intervals for score in i['scores'].values())]
        if any(not math.isfinite(s[k]) or not 0<=s[k]<=1 for s in scores for k in ('f1','em')):
            raise ValueError('Invalid answer quality metric')
    ids={q for q,c in seen}
    if not ids or seen!={(q,c) for q in ids for c in conditions}:
        raise ValueError('Question conditions are incomplete')
    if len(ids)!=settings['limit']:
        raise ValueError('Run limit and completed question count differ')
    full=ids==dev_ids and set(conditions)=={'P0','P1','P2'} and sorted(starts)==list(range(0,28,4)) and width==4
    result={'scope':'Development-only descriptive analysis; no held-out gate or speed claim',
            'question_count':len(ids),'expected_development_questions':len(dev_ids),
            'complete_preregistered_development_screen':full,'conditions':{},
            'interval_candidate':None,'freeze_status':'not frozen; numerical review and explicit record required'}
    for condition in conditions:
        rows=sorted((r for r in report['cases'] if r['condition']==condition),key=lambda r:r['id'])
        mean=lambda scores,key:statistics.fmean(s[key] for s in scores)
        native=[r['native'] for r in rows]
        history=[r['history_controls']['native_source_history'] for r in rows]
        graft=[r['history_controls']['source_history_shared_with_target_prefix'] for r in rows]
        summary={'native_f1':mean(native,'f1'),'direct_f1':mean([r['direct'] for r in rows],'f1'),
                 'source_history_f1':mean(history,'f1'),'history_graft_f1':mean(graft,'f1'),
                 'target_minus_source_history_f1':paired_delta([s['f1'] for s in history],[s['f1'] for s in native],draws),
                 'calibration':{'max_split_dense_relative_logit_error':max(r['calibration']['split_dense_relative_logit_error'] for r in rows),
                                'max_split_dense_kl':max(r['calibration']['kl_reference_to_variant'] for r in rows),
                                'cases_with_argmax_disagreement':sum(r['calibration']['argmax_agreement']<1 for r in rows)},
                 'intervals':[]}
        for start in sorted(starts):
            scores=[next(i['scores'] for i in r['intervals'] if i['start']==start) for r in rows]
            source=[s['source'] for s in scores]
            oracle=[s['oracle'] for s in scores]
            summary['intervals'].append({'start':start,'width':width,'source_f1':mean(source,'f1'),
                'target_entry_availability':'Exact target embedding lookup; no target X prefill needed for E.' if start==0
                    else 'Oracle from full target forward; online construction cost not established.',
                'oracle_f1':mean(oracle,'f1'),
                'oracle_minus_source_f1':paired_delta([s['f1'] for s in source],[s['f1'] for s in oracle],draws),
                'oracle_minus_source_em':paired_delta([s['em'] for s in source],[s['em'] for s in oracle],draws),
                'recovery':recovery_fraction(mean(native,'f1'),mean(source,'f1'),mean(oracle,'f1')),
                'source_kl':mean(source,'kl_reference_to_variant'),'oracle_kl':mean(oracle,'kl_reference_to_variant')})
        result['conditions'][condition]=summary
    if full:
        best=min(result['conditions']['P2']['intervals'],key=lambda i:(-i['source_f1'],i['start']))
        result['interval_candidate']={'start':best['start'],'width':width,
            'selection':'Highest mean P2 source-entry F1 across all development questions; ties use lower start.'}
    result['statistical_limit']='Per-interval descriptive intervals are not multiplicity-adjusted. Development selection is not a confirmatory test.'
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--manifest',default='results/stage_a_data_manifest.json')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():
        raise FileExistsError('Never overwrite an analysis record')
    report=json.loads(Path(args.input).read_text())
    manifest_bytes=Path(args.manifest).read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest()!=report['data_manifest_sha256']:
        raise ValueError('Run and analysis refer to different data manifests')
    result=summarize(report,json.loads(manifest_bytes))
    result['input_sha256']=hashlib.sha256(Path(args.input).read_bytes()).hexdigest()
    result['data_manifest_sha256']=report['data_manifest_sha256']
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'output':str(output),'interval_candidate':result['interval_candidate']}))


if __name__=='__main__':
    main()
