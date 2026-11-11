"""Paired on-device B/C2 timing, with fresh state, balanced order, exact outputs."""
import argparse,gc,hashlib,json,statistics,subprocess,sys,time,traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--frontend',action='store_true');p.add_argument('--pairs',type=int,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
base=Path(__file__).resolve().parent;root=base.parents[1]
up=Path.home()/'.agent-reach/research/fluxmem-repro';modeldir=Path.home()/'.agent-reach/research/fluxmem-qwen25vl3b/66285546d2b821cf421d4f5eb2576359d3770cd3'
sys.path[:0]=[str(root/'experiments/fluxmem_incremental'),str(up/'models/qwen2-5-vl/src'),str(root/'experiments/fluxmem')]
import torch
from safetensors import safe_open
from qwen2_5_vl_fluxmem import Qwen2_5_VLConfig
from qwen2_5_vl_fluxmem.fluxmem import FluxMem
from qwen2_5_vl_fluxmem.modeling_qwen2_5_vl_fluxmem import Qwen2_5_VLRotaryEmbedding,Qwen2_5_VisionTransformerPretrainedModel
from state import Counter,full_memory,compare,storage_report
from compact import CompactMemory
from measure import Meter
assert torch.backends.mps.is_available();torch.set_num_threads(4);torch.manual_seed(1234)
prior=json.loads(a.audit.read_text());meta=json.loads(a.inputs.read_text())
assert prior['status']=='completed' and prior['input_manifest_sha256']==hashlib.sha256(a.inputs.read_bytes()).hexdigest()
refs={}
for n in [2,26,41]:
 src=next(s for s in prior['snapshots'] if s['blocks']==n and s['condition']=='A')
 assert hashlib.sha256(Path(src['path']).read_bytes()).hexdigest()==src['sha256']
 refs[n]={k:v.to('mps') for k,v in torch.load(src['path'],weights_only=True).items()}
raw=refs[41]['raw'];cfg=Qwen2_5_VLConfig.from_pretrained(modeldir,local_files_only=True)
index=json.loads((modeldir/'model.safetensors.index.json').read_text())['weight_map']
with safe_open(modeldir/index['model.embed_tokens.weight'],framework='pt',device='cpu') as f:
 sl=f.get_slice('model.embed_tokens.weight');markers=torch.cat([sl[t:t+1] for t in [cfg.vision_start_token_id,cfg.vision_end_token_id]]).to('mps')
rotary=Qwen2_5_VLRotaryEmbedding(cfg).to('mps')
vision=None
if a.frontend:
 assert hashlib.sha256(Path(meta['blocks_path']).read_bytes()).hexdigest()==meta['blocks_sha256']
 pixels=torch.load(meta['blocks_path'],weights_only=True)
 vision=Qwen2_5_VisionTransformerPretrainedModel._from_config(cfg.vision_config,torch_dtype=torch.bfloat16,attn_implementation='sdpa')
 weights={}
 for filename in sorted(set(index.values())):
  with safe_open(modeldir/filename,framework='pt',device='cpu') as f:
   for key in f.keys():
    if key.startswith('visual.'):weights[key.removeprefix('visual.')]=f.get_tensor(key)
 info=vision.load_state_dict(weights,strict=True,assign=True);assert not info.missing_keys and not info.unexpected_keys
 vision=vision.to('mps').eval();del weights
 grid=torch.tensor([[1,16,30]],device='mps')
gc.collect();torch.mps.empty_cache();torch.mps.synchronize()
r={'status':'running','mode':'frontend' if a.frontend else 'memory_only','pairs_requested':a.pairs,'torch':torch.__version__,'chip':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),'ram_bytes':int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True)),'input_manifest_sha256':hashlib.sha256(a.inputs.read_bytes()).hexdigest(),'audit_sha256':hashlib.sha256(a.audit.read_bytes()).hexdigest(),'source_hashes':{},'runs':[],'vision_parameter_bytes':sum(v.numel()*v.element_size() for v in vision.parameters()) if vision else 0,'source_feature_bank_bytes':raw.numel()*raw.element_size(),'process_peaks_are_not_single_condition_deployment':True}
for name in ['benchmark.py','compact.py','PROTOCOL.md']:
 f=base/name;(a.output/name).write_bytes(f.read_bytes());r['source_hashes'][name]=hashlib.sha256(f.read_bytes()).hexdigest()
meter=Meter(torch)
def save():(a.output/'results.json').write_text(json.dumps(r,indent=2)+'\n')
def run(cond,pair,warmup=False):
 policy=FluxMem(cfg.vision_start_token_id,cfg.vision_end_token_id,short_term_frames=8,mid_term_frames=64)
 counter=Counter(policy);state=CompactMemory(policy) if cond=='C2' else None;cache=[];cursor=0;events=[];vision_calls=0
 gc.collect();torch.mps.synchronize()
 for n in [2,26,41]:
  new=[]
  with meter.phase('incoming_visual') as visual_event:
   for i in range(cursor,n):
    if a.frontend:
     feature=vision(pixels[i].to('mps',dtype=torch.bfloat16),grid_thw=grid);vision_calls+=1
    else:feature=raw[i*120:(i+1)*120]
    new.append(feature)
  with meter.phase('memory_insertion_and_readout') as memory_event:
   if cond=='B':
    cache.extend(f.detach().clone() for f in new)
    features=torch.cat(cache)
    out=full_memory(policy,features,cfg,rotary,markers)
   else:
    for feature in new:state.append(feature)
    out=state.snapshot()
  # Outside all measured stages: validate features and output; report owned storage.
  assert all(torch.equal(f,raw[(cursor+j)*120:(cursor+j+1)*120]) for j,f in enumerate(new))
  eq=compare(refs[n],out);assert eq['features_bitwise_equal'],eq
  events.append({'blocks':n,'visual':visual_event,'memory':memory_event,'equivalence':eq,'state':state.state_report() if state else storage_report(cache)})
  cursor=n
  if cond=='B':del features
  del out,new,feature
 result={'condition':cond,'pair':pair,'warmup':warmup,'events':events,'counts':dict(counter.counts),'vision_calls':vision_calls,'visual_seconds':sum(e['visual']['seconds'] for e in events),'memory_seconds':sum(e['memory']['seconds'] for e in events)}
 result['total_seconds']=result['visual_seconds']+result['memory_seconds']
 result['final_state_bytes']=events[-1]['state']['tensor_storage_bytes']
 del state,cache;gc.collect();torch.mps.synchronize()
 return result
try:
 with torch.inference_mode():
  for c in ['B','C2']:
   result=run(c,-1,True);r['runs'].append(result);save();print('warmup',c,result['total_seconds'],flush=True)
  for pair in range(a.pairs):
   for c in (['B','C2'] if pair%2==0 else ['C2','B']):
    result=run(c,pair);r['runs'].append(result);save();print('pair',pair,c,'seconds',result['total_seconds'],'memory',result['memory_seconds'],'bytes',result['final_state_bytes'],flush=True)
 # Bootstrap over complete pairs, not queries; CI captures only timing repetitions.
 import random
 rng=random.Random(1234);measured=[v for v in r['runs'] if not v['warmup']];summary={}
 for metric in ['total_seconds','memory_seconds','visual_seconds']:
  vals={c:[v[metric] for v in measured if v['condition']==c] for c in ['B','C2']}
  deltas=[vals['B'][i]-vals['C2'][i] for i in range(a.pairs)]
  means=sorted(statistics.mean(rng.choices(deltas,k=len(deltas))) for _ in range(10000))
  summary[metric]={'conditions':{c:{'median':statistics.median(v),'mean':statistics.mean(v),'min':min(v),'max':max(v),'values':v} for c,v in vals.items()},'paired_B_minus_C2_seconds':deltas,'mean_paired_saving':statistics.mean(deltas),'mean_paired_saving_bootstrap_95':[means[249],means[9749]],'C2_faster_pairs':sum(d>0 for d in deltas)}
 r.update(status='completed',summary=summary)
except BaseException:r.update(status='failed',error=traceback.format_exc());raise
finally:meter.close();save()
