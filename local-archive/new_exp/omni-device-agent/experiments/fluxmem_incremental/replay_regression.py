"""Real MPS state replay, copy accounting, reset and a fault-injection regression."""
import argparse,json,sys,hashlib,traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
up=Path.home()/'.agent-reach/research/fluxmem-repro';modeldir=Path.home()/'.agent-reach/research/fluxmem-qwen25vl3b/66285546d2b821cf421d4f5eb2576359d3770cd3'
sys.path.insert(0,str(up/'models/qwen2-5-vl/src'))
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from qwen2_5_vl_fluxmem import Qwen2_5_VLConfig
from qwen2_5_vl_fluxmem.fluxmem import FluxMem
from qwen2_5_vl_fluxmem.modeling_qwen2_5_vl_fluxmem import Qwen2_5_VLRotaryEmbedding
from safetensors import safe_open
from state import Counter,IncrementalMemory,full_memory,compare
x=json.loads(a.audit.read_text());assert x['status']=='completed'
refs={}
for n in [2,26,41]:
 src=next(s for s in x['snapshots'] if s['blocks']==n and s['condition']=='A')
 assert hashlib.sha256(Path(src['path']).read_bytes()).hexdigest()==src['sha256']
 refs[n]={k:v.to('mps') for k,v in torch.load(src['path'],weights_only=True).items()}
raw=refs[41]['raw'];cfg=Qwen2_5_VLConfig.from_pretrained(modeldir,local_files_only=True)
def pol():return FluxMem(cfg.vision_start_token_id,cfg.vision_end_token_id,short_term_frames=8,mid_term_frames=64)
class Copies(TorchDispatchMode):
 def __init__(self):super().__init__();self.calls={};self.bytes={}
 def __torch_dispatch__(self,func,types,args=(),kwargs=None):
  out=func(*args,**(kwargs or {}));name=str(func)
  if name in ['aten.cat.default','aten.clone.default']:
   self.calls[name]=self.calls.get(name,0)+1;self.bytes[name]=self.bytes.get(name,0)+out.numel()*out.element_size()
  return out
r={'status':'running','device':'mps','torch':torch.__version__,'audit_sha256':hashlib.sha256(a.audit.read_bytes()).hexdigest(),'state_sha256':hashlib.sha256(Path(__file__).with_name('state.py').read_bytes()).hexdigest(),'checks':[]}
a.output.parent.mkdir(parents=True,exist_ok=True)
try:
 with torch.inference_mode():
  policy=pol();counter=Counter(policy);state=IncrementalMemory(policy);copies=Copies()
  for i in range(41):
   with copies:state.append(raw[i*120:(i+1)*120])
   if i+1 in refs:
    eq=compare(refs[i+1],state.snapshot());assert eq['features_bitwise_equal'],eq
    r['checks'].append({'blocks':i+1,'equivalence':eq,'cat_clone_calls':dict(copies.calls),'cat_clone_logical_output_bytes':dict(copies.bytes),'state':state.state_report()})
  before=dict(counter.counts);snap0=state.snapshot();snap1=state.snapshot()
  assert compare(snap0,snap1)['features_bitwise_equal'] and before==counter.counts
  state.reset();assert state.state_report()['tensor_storage_bytes']==0
  for i in range(2):state.append(raw[i*120:(i+1)*120])
  assert compare(refs[2],state.snapshot())['features_bitwise_equal']
  r['reset_and_replay_passed']=True
  # Deliberately inject the historical/read-mask bug into an isolated object.
  good=IncrementalMemory(pol());bad=IncrementalMemory(pol())
  for i in range(5):
   good.append(raw[i*120:(i+1)*120]);bad.append(raw[i*120:(i+1)*120])
   if i==3:
    for frame in bad.short:bad.keep[bad.indices[frame]]=True
  fault=compare(good.snapshot(),bad.snapshot());assert not fault['indices_equal'], 'Regression is insensitive to polluted short-window keep mask'
  r['injected_read_mask_fault_detected']=fault
  # Original full method at the first pruning boundary remains the oracle.
  index=json.loads((modeldir/'model.safetensors.index.json').read_text())['weight_map']
  with safe_open(modeldir/index['model.embed_tokens.weight'],framework='pt',device='cpu') as f:
   sl=f.get_slice('model.embed_tokens.weight');markers=torch.cat([sl[t:t+1] for t in [cfg.vision_start_token_id,cfg.vision_end_token_id]]).to('mps')
  oracle=full_memory(pol(),raw[:600],cfg,Qwen2_5_VLRotaryEmbedding(cfg).to('mps'),markers)
  eq=compare(oracle,good.snapshot());assert eq['features_bitwise_equal']
  r['first_prune_boundary_vs_original']=eq
  torch.mps.synchronize()
 r['status']='passed'
except Exception:r.update(status='failed',error=traceback.format_exc());raise
finally:a.output.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'status':r['status'],'copy_calls':copies.calls,'copy_logical_bytes':copies.bytes,'fault_detected':not fault['indices_equal']}),flush=True)
