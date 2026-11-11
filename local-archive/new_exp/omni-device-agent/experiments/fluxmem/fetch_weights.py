"""Reacquire exactly the weight files recorded by the completed SHA-256 audit."""
import hashlib,json,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
manifest=json.loads(Path(__file__).with_name('evidence').joinpath('weights_verified.json').read_text())
root=Path.home()/'.agent-reach/research/fluxmem-qwen25vl3b'/manifest['revision'];root.mkdir(parents=True,exist_ok=True)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while block:=f.read(8*1024*1024):h.update(block)
 return h.hexdigest()
def fetch(row):
 name=row['path'];p=root/name
 if not p.exists():
  tmp=p.with_suffix(p.suffix+'.partial')
  url=f"https://huggingface.co/{manifest['repo']}/resolve/{manifest['revision']}/{name}"
  subprocess.run(['curl','-fL','--retry','3','--max-time','1800','-o',str(tmp),url],check=True)
  if digest(tmp)!=row['sha256']:raise RuntimeError('Invalid checksum: '+name)
  tmp.replace(p)
 if digest(p)!=row['sha256']:raise RuntimeError('Invalid existing file: '+name)
 print('VERIFIED',name,flush=True)
with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(fetch,manifest['files']))
(root/'verified_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
