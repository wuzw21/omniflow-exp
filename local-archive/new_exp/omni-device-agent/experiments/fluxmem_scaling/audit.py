"""Exact prefix audit on 94 real visual-feature blocks."""
import argparse, json, sys, hashlib, subprocess, traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
from common import setup, policy, sha, freeze, full_memory, compare, Counter
from compact import CompactMemory
import torch
from qwen2_5_vl_fluxmem.fluxmem import FluxMem
assert torch.backends.mps.is_available()
features=torch.load(json.loads((a.features/'results.json').read_text())['features_path'],weights_only=True).to('mps')
cfg,markers,rotary,_=setup(False)
pol=policy(cfg); inc=CompactMemory(pol); counter=Counter(pol)
queries=list(range(1,95))
r={'status':'running','device':'mps','chip':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),'features_sha256':sha(json.loads((a.features/'results.json').read_text())['features_path']),'queries':queries,'records':[],'source_hashes':{}}
for name in ['audit.py','common.py','PROTOCOL.md']:
 f=Path(__file__).with_name(name);(a.output/name).write_bytes(f.read_bytes());r['source_hashes'][name]=sha(f)
def save():(a.output/'results.json').write_text(json.dumps(r,indent=2)+'\n')
try:
 with torch.inference_mode():
  for i in range(94):
   f=features[i*1:i*1+1].reshape(120,features.shape[-1])
   inc.append(f)
   if i+1 in queries:
    oracle=full_memory(policy(cfg),features[:i+1].reshape(-1,features.shape[-1]),cfg,rotary,markers)
    out=inc.snapshot(); eq=compare(oracle,out)
    assert eq['features_bitwise_equal'],eq
    r['records'].append({'blocks':i+1,'seconds':2*(i+1),'equivalence':eq,'retained_tokens':len(out['indices']),'state':inc.state_report(),'counts':dict(counter.counts)})
    save(); print('audit',i+1,eq['tokens_a'],flush=True)
  r['status']='completed'
except BaseException:r.update(status='failed',error=traceback.format_exc());raise
finally:save()
