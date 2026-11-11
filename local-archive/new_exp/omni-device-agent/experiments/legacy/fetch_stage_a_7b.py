"""Download only the manifest-pinned 7B pair, with resumable verified shards."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(path, git_blob=False):
    h=hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        h.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):
            h.update(chunk)
    return h.hexdigest()


def valid(path, entry):
    if not path.is_file() or path.stat().st_size!=entry['bytes']:
        return False
    expected=entry['sha256'] or entry['git_blob_sha1']
    return digest(path,git_blob=not bool(entry['sha256']))==expected


def fetch_model(root, role, model, endpoint='https://huggingface.co'):
    folder=root/model['revision']
    folder.mkdir(parents=True,exist_ok=True)
    for entry in model['files']:
        name=entry['name']
        if Path(name).name!=name:
            raise ValueError('Manifest filenames must be plain basenames')
        target=folder/name
        if valid(target,entry):
            print(f'verified cached {role}/{name}',flush=True)
            continue
        partial=folder/(name+'.download')
        if partial.exists() and partial.stat().st_size==entry['bytes']:
            if valid(partial,entry):
                partial.replace(target)
                continue
            raise ValueError(f'Existing full partial has a bad checksum: {partial}')
        if partial.exists() and partial.stat().st_size>entry['bytes']:
            raise ValueError(f'Partial exceeds expected size: {partial}')
        url=f"{endpoint}/{model['repo']}/resolve/{model['revision']}/{name}?download=true"
        print(f'downloading {role}/{name}',flush=True)
        subprocess.run(['curl','--fail','--location','--silent','--show-error',
                        '--retry','3','--connect-timeout','20','--max-time','7200',
                        '--continue-at','-','--max-filesize',str(max(entry['bytes'],1024**2)),
                        '--output',str(partial),url],check=True)
        if not valid(partial,entry):
            raise ValueError(f'Download integrity failure: {role}/{name}')
        partial.replace(target)
        print(f'verified {role}/{name}: {entry["bytes"]} bytes',flush=True)
    (folder/'verified_manifest.json').write_text(json.dumps(model,indent=2)+'\n')
    return str(folder)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',default='stage_a_7b_models.json')
    parser.add_argument('--root',required=True)
    parser.add_argument('--endpoint',choices=['https://huggingface.co','https://hf-mirror.com'],default='https://huggingface.co')
    parser.add_argument('--roles',nargs='+',choices=['base','instruct'],default=['base','instruct'])
    args=parser.parse_args()
    models=json.loads(Path(args.manifest).read_text())['models']
    models={role:models[role] for role in args.roles}
    root=Path(args.root).expanduser()
    root.mkdir(parents=True,exist_ok=True)
    missing=sum(max(0,e['bytes']-min(e['bytes'],(root/m['revision']/(e['name']+'.download')).stat().st_size
                                    if (root/m['revision']/(e['name']+'.download')).exists() else 0))
                for m in models.values() for e in m['files'] if not (root/m['revision']/e['name']).exists())
    if shutil.disk_usage(root).free<missing+2*1024**3:
        raise RuntimeError('Insufficient free disk space for pinned weights and margin')
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={role:pool.submit(fetch_model,root,role,m,args.endpoint) for role,m in models.items()}
        result={role:f.result() for role,f in futures.items()}
    print(json.dumps({'status':'verified_complete','model_paths':result}),flush=True)
