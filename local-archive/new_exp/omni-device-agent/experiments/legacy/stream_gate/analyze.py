import hashlib
import json
from pathlib import Path
from PIL import Image
from policies import configurations,thumbnail,textset,evidence_events,replay,score

ROOT=Path(__file__).resolve().parent

def main():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    ocr={r['path']:r for r in map(json.loads,(ROOT/'ocr_reference.jsonl').read_text().splitlines())}
    records=[]
    for row in manifest['records']:
        features=[]; texts=[]
        for frame in row['frames']:
            p=Path(frame['path'])
            assert hashlib.sha256(p.read_bytes()).hexdigest()==frame['sha256']
            with Image.open(p) as img: features.append(thumbnail(img))
            texts.append(textset(ocr[str(p)]['rows']))
        events=evidence_events(texts)
        for kind,param in configurations():
            selected,states=replay(features,kind,param)
            records.append({'id':row['id'],'split':row['split'],'kind':kind,'param':param,
                'frames':len(features),'selected':selected,'states':states,**score(events,states)})
    aggregate=[]
    for split in ('dev','test'):
        for kind,param in configurations():
            rows=[r for r in records if (r['split'],r['kind'],r['param'])==(split,kind,param)]
            total=sum(r['events'] for r in rows); hits=sum(r['hits'] for r in rows)
            calls=sum(len(r['selected']) for r in rows); frames=sum(r['frames'] for r in rows)
            aggregate.append({'split':split,'kind':kind,'param':param,'events':total,'hits':hits,
                'recall':hits/total if total else None,'calls':calls,'frames':frames,'call_fraction':calls/frames})
    chosen=[]
    for kind in ('uniform','last','memory'):
        eligible=[r for r in aggregate if r['split']=='dev' and r['kind']==kind and r['recall'] is not None and r['recall']>=.95]
        if eligible: chosen.append(min(eligible,key=lambda r:(r['calls'],-r['recall'],r['param'])))
        else: chosen.append({'kind':kind,'status':'no_configuration_meets_dev_recall_95pct'})
    result={'reference':'Apple Vision OCR proxy, not human ground truth or QA accuracy',
        'threshold_selection':'dev only >=95% reference recall, minimize calls',
        'chosen':chosen,'aggregate':aggregate,'records':records}
    (ROOT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'chosen':chosen,'aggregate':aggregate},indent=2))

if __name__=='__main__': main()
