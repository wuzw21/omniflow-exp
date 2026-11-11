"""Pinned public screenshots only; never export source reasoning or private text."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
REV = 'ede2a362c99ddba0d03f439c47a0b8fea6abae22'
CACHE = Path.home() / '.agent-reach/research/stream-gate-pilot'
SOURCE = Path.home() / '.agent-reach/research/hcompany-trajectories'

def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    chosen, seen = [], set()
    index = json.loads((SOURCE / 'index.json').read_text())
    for entry in index['trajectories']:
        if entry['benchmark'] != 'AndroidWorld':
            continue
        row = json.loads((SOURCE / (entry['id'] + '.json')).read_text())
        task = row['task'].rsplit('-', 1)[0]
        if task in seen or not 5 <= len(row['steps']) <= 24:
            continue
        seen.add(task)
        chosen.append({'id': row['id'], 'task': task, 'split': 'dev' if len(chosen)<6 else 'test',
                       'frames': [{'source': s['image'], 'step': i,
                                   'path': str(CACHE / row['id'] / f'{i:03d}.webp')}
                                  for i,s in enumerate(row['steps'])]})
        if len(chosen)==12:
            break
    def fetch(frame):
        p=Path(frame['path']); p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            temp=p.with_suffix('.download')
            subprocess.run(['curl','-fsSL','--retry','2','--connect-timeout','15','--max-time','90',
                '--max-filesize','10000000',
                f'https://huggingface.co/datasets/Hcompany/trajectories/resolve/{REV}/{frame["source"]}',
                '-o',str(temp)],check=True)
            temp.replace(p)
        frame['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
        return frame
    frames=[f for r in chosen for f in r['frames']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for i,_ in enumerate(pool.map(fetch,frames)):
            if i%10==0: print(f'downloaded {i+1}/{len(frames)}',flush=True)
    result={'dataset':'Hcompany/trajectories','revision':REV,'records':chosen,
            'selection':'first 12 distinct task names with 5-24 steps in published AndroidWorld index',
            'index_sha256':hashlib.sha256((SOURCE/'index.json').read_bytes()).hexdigest()}
    (ROOT/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    binary=CACHE/'ocr'
    subprocess.run(['swiftc','-O',str(ROOT/'ocr.swift'),'-o',str(binary)],check=True)
    with (ROOT/'ocr_reference.jsonl').open('w') as out:
        subprocess.run([str(binary),*[f['path'] for f in frames]],stdout=out,check=True)
    print(json.dumps({'trajectories':len(chosen),'frames':len(frames),'ocr':'completed'}),flush=True)

if __name__=='__main__': main()
