"""Post-hoc measurement audit, without retuning gates or replacing raw results."""
import json
from pathlib import Path
from policies import textset,evidence_events,score

ROOT=Path(__file__).resolve().parent

def main():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    analysis=json.loads((ROOT/'analysis.json').read_text())
    ocr={r['path']:r for r in map(json.loads,(ROOT/'ocr_reference.jsonl').read_text().splitlines())}
    records=[]
    for row in manifest['records']:
        texts=[textset([r for r in ocr[f['path']]['rows'] if r['bbox'][1]+r['bbox'][3]<.90]) for f in row['frames']]
        events=evidence_events(texts)
        for prior in [r for r in analysis['records'] if r['id']==row['id']]:
            records.append({k:v for k,v in prior.items() if k not in ('events','hits','missed')} | score(events,prior['states']))
    aggregates=[]
    for a in analysis['aggregate']:
        rr=[r for r in records if (r['split'],r['kind'],r['param'])==(a['split'],a['kind'],a['param'])]
        n=sum(r['events'] for r in rr); h=sum(r['hits'] for r in rr)
        aggregates.append({**a,'events':n,'hits':h,'recall':h/n if n else None})
    audit={'status':'posthoc_measurement_sensitivity_not_retuning',
        'reason':'Manual image inspection found Android pointer-location debug overlays inside the original top-5% exclusion. Exclude the top 10% for this sensitivity only.',
        'decision_policy':'Use originally frozen dev-selected configurations; do not reselect using cleaned reference.',
        'aggregate':aggregates,'records':records,
        'manual_cases':[
            {'id':'clocktimerentry-0-0a8e4277','evidence_step':6,'query_step':7,
             'visually_verified_text':'00h 00m 16s',
             'note':'Screen image manually inspected; OCR renders h as n. This is a real numeric field change, not a debug overlay.'},
            {'id':'expenseaddsingle-0-ca36846c','evidence_step':9,'query_step':11,
             'visually_verified_text':['Transportation','Clothes','Health Care'],
             'note':'Screen image manually inspected. Horizontal category scrolling reveals these choices. Baseline earlier retained frame shows Food/Income/Housing/Social instead.'} ]}
    (ROOT/'measurement_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps([r for r in aggregates if r['split']=='test' and r['param'] in (0,.01,2)],indent=2))

if __name__=='__main__': main()
