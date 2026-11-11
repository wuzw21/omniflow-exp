"""Real MPS equivalence, lifecycle, and copy audit; no synthetic features."""
import argparse,hashlib,json,sys,traceback,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
base=Path(__file__).resolve().parent;root=base.parents[1]
up=Path.home()/'.agent-reach/research/fluxmem-repro';modeldir=Path.home()/'.agent-reach/research/fluxmem-qwen25vl3b/66285546d2b821cf421d4f5eb2576359d3770cd3'
sys.path[:0]=[str(root/'experiments/fluxmem_incremental'),str(up/'models/qwen2-5-vl/src')]
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from safetensors import safe_open
from qwen2_5_vl_fluxmem import Qwen2_5_VLConfig
from qwen2_5_vl_fluxmem.fluxmem import FluxMem
from qwen2_5_vl_fluxmem.modeling_qwen2_5_vl_fluxmem import Qwen2_5_VLRotaryEmbedding
from state import Counter,IncrementalMemory,full_memory,compare
from compact import CompactMemory
assert torch.backends.mps.is_available();torch.set_num_threads(4);torch.manual_seed(1234)
old=json.loads(a.audit.read_text());assert old['status']=='completed'
src=next(s for s in old['snapshots'] if s['blocks']==41 and s['condition']=='A')
assert hashlib.sha256(Path(src['path']).read_bytes()).hexdigest()==src['sha256']
raw=torch.load(src['path'],weights_only=True)['raw'].to('mps')
cfg=Qwen2_5_VLConfig.from_pretrained(modeldir,local_files_only=True)
index=json.loads((modeldir/'model.safetensors.index.json').read_text())['weight_map']
with safe_open(modeldir/index['model.embed_tokens.weight'],framework='pt',device='cpu') as f:
 sl=f.get_slice('model.embed_tokens.weight');markers=torch.cat([sl[t:t+1] for t in [cfg.vision_start_token_id,cfg.vision_end_token_id]]).to('mps')
rotary=Qwen2_5_VLRotaryEmbedding(cfg).to('mps')
def pol():return FluxMem(cfg.vision_start_token_id,cfg.vision_end_token_id,short_term_frames=8,mid_term_frames=64)
class Copies(TorchDispatchMode):
 def __init__(self):super().__init__();self.calls={};self.bytes={}
 def __torch_dispatch__(self,func,types,args=(),kwargs=None):
  out=func(*args,**(kwargs or {}));name=str(func)
  if name in ['aten.cat.default','aten.clone.default','aten.copy_.default']:
   self.calls[name]=self.calls.get(name,0)+1;self.bytes[name]=self.bytes.get(name,0)+out.numel()*out.element_size()
  return out
r={'status':'running','device':'mps','torch':torch.__version__,'chip':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),'upstream_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=up,text=True).strip(),'upstream_diff':subprocess.check_output(['git','diff'],cwd=up,text=True),'input_manifest_sha256':old['input_manifest_sha256'],'parent_audit_sha256':hashlib.sha256(a.audit.read_bytes()).hexdigest(),'snapshots':[],'records':[],'source_hashes':{}}
for name in ['validate.py','compact.py','check_append.py','PROTOCOL.md']:
 f=base/name;(a.output/name).write_bytes(f.read_bytes());r['source_hashes'][name]=hashlib.sha256(f.read_bytes()).hexdigest()
def save():(a.output/'results.json').write_text(json.dumps(r,indent=2)+'\n')
try:
 with torch.inference_mode():
  oldpol=pol();newpol=pol();oldcounter=Counter(oldpol);newcounter=Counter(newpol)
  oldstate=IncrementalMemory(oldpol);newstate=CompactMemory(newpol);copies={'C1':Copies(),'C2':Copies()}
  for i in range(41):
   feature=raw[i*120:(i+1)*120]
   with copies['C1']:oldstate.append(feature)
   with copies['C2']:newstate.append(feature)
   reference=full_memory(pol(),raw[:(i+1)*120],cfg,rotary,markers)
   out=newstate.snapshot();eq=compare(reference,out);legacy=compare(oldstate.snapshot(),out)
   counts=dict(newcounter.counts);before=newstate.state_report();again=compare(out,newstate.snapshot())
   assert eq['features_bitwise_equal'] and legacy['features_bitwise_equal'] and again['features_bitwise_equal']
   assert counts==newcounter.counts==oldcounter.counts and before==newstate.state_report()
   r['records'].append({'blocks':i+1,'vs_original':eq,'vs_C1':legacy,'repeat_read':again,'counts':counts,'C1_state':oldstate.state_report(),'C2_state':before})
   if i+1 in [2,26,41]:
    for cond in 'AB':r['snapshots'].append(next(s for s in old['snapshots'] if s['blocks']==i+1 and s['condition']==cond))
    path=a.output/f'question_{i+1}_C.pt';torch.save({'raw':raw[:(i+1)*120].cpu(),'indices':out['indices'].cpu(),'features':out['features'].cpu()},path)
    r['snapshots'].append({'blocks':i+1,'condition':'C','path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
   print('prefix',i+1,'equal',eq['features_bitwise_equal'],'C2 bytes',before['tensor_storage_bytes'],flush=True);save()
  r['append_copies']={k:{'calls':v.calls,'logical_bytes':v.bytes} for k,v in copies.items()}
  assert copies['C2'].calls.get('aten.cat.default',0)==0
  assert newstate.state_report()['tensor_storage_bytes'] < raw.numel()*raw.element_size()
  newstate.reset();assert newstate.state_report()['tensor_storage_bytes']==0
  for i in range(5):newstate.append(raw[i*120:(i+1)*120])
  r['reset_replay']=compare(newstate.snapshot(),full_memory(pol(),raw[:600],cfg,rotary,markers))
  assert r['reset_replay']['features_bitwise_equal']
  # A returned snapshot is owned by the reader; mutating it cannot change state.
  out=newstate.snapshot();out['features'].zero_();out['indices'].zero_()
  r['snapshot_mutation_isolated']=compare(newstate.snapshot(),full_memory(pol(),raw[:600],cfg,rotary,markers))
  assert r['snapshot_mutation_isolated']['features_bitwise_equal']
  torch.mps.synchronize()
 r['status']='completed'
except BaseException:r.update(status='failed',error=traceback.format_exc());raise
finally:save()
