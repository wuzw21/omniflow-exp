"""Execute three independent controls on real MPS visual features at each prefix."""
import argparse,json,sys,hashlib,subprocess,time,traceback,gc
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=41);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
up=Path.home()/'.agent-reach/research/fluxmem-repro';modeldir=Path.home()/'.agent-reach/research/fluxmem-qwen25vl3b/66285546d2b821cf421d4f5eb2576359d3770cd3'
sys.path[:0]=[str(up/'models/qwen2-5-vl/src'),str(Path(__file__).resolve().parent.parent/'fluxmem')]
import torch
from safetensors import safe_open
from qwen2_5_vl_fluxmem import Qwen2_5_VLConfig
from qwen2_5_vl_fluxmem.modeling_qwen2_5_vl_fluxmem import Qwen2_5_VisionTransformerPretrainedModel,Qwen2_5_VLRotaryEmbedding
from qwen2_5_vl_fluxmem.fluxmem import FluxMem
from measure import Meter
from state import Counter,IncrementalMemory,full_memory,compare,storage_report
assert torch.backends.mps.is_available();torch.set_num_threads(4);torch.manual_seed(1234)
meta=json.loads(a.inputs.read_text());assert meta['status']=='completed'
assert hashlib.sha256(Path(meta['blocks_path']).read_bytes()).hexdigest()==meta['blocks_sha256']
blocks=torch.load(meta['blocks_path'],weights_only=True)
meter=Meter(torch,trace_path=a.output/'phases.jsonl')
r={'status':'running','torch':torch.__version__,'hardware':{'chip':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),'ram_bytes':int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True))},'upstream_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=up,text=True).strip(),'upstream_diff':subprocess.check_output(['git','diff'],cwd=up,text=True),'input_manifest_sha256':hashlib.sha256(a.inputs.read_bytes()).hexdigest(),'records':[],'snapshots':[],'source_hashes':{}}
for name in ['audit.py','state.py','prepare.py','PROTOCOL.md']:
 f=Path(__file__).with_name(name);(a.output/name).write_bytes(f.read_bytes());r['source_hashes'][name]=hashlib.sha256(f.read_bytes()).hexdigest()
def save():
 t=a.output/'results.tmp';t.write_text(json.dumps(r,indent=2)+'\n');t.replace(a.output/'results.json')
save()
try:
 cfg=Qwen2_5_VLConfig.from_pretrained(modeldir,local_files_only=True)
 with meter.phase('load_original_vision_weights') as load:
  vision=Qwen2_5_VisionTransformerPretrainedModel._from_config(cfg.vision_config,torch_dtype=torch.bfloat16,attn_implementation='sdpa')
  index=json.loads((modeldir/'model.safetensors.index.json').read_text())['weight_map'];weights={};markers=[]
  for filename in sorted(set(index.values())):
   with safe_open(modeldir/filename,framework='pt',device='cpu') as f:
    for key in f.keys():
     if key.startswith('visual.'):weights[key.removeprefix('visual.')]=f.get_tensor(key)
     if key=='model.embed_tokens.weight':
      sl=f.get_slice(key);markers=[sl[t:t+1].clone() for t in [cfg.vision_start_token_id,cfg.vision_end_token_id]]
  info=vision.load_state_dict(weights,strict=True,assign=True);vision=vision.to('mps').eval();del weights
  markers=torch.cat(markers).to('mps');rotary=Qwen2_5_VLRotaryEmbedding(cfg).to('mps');gc.collect()
 r.update(vision_parameter_bytes=sum(x.numel()*x.element_size() for x in vision.parameters()),vision_loading_info={'missing_keys':info.missing_keys,'unexpected_keys':info.unexpected_keys},load=load)
 grid=torch.tensor([[1,16,30]],device='mps')
 def policy():return FluxMem(cfg.vision_start_token_id,cfg.vision_end_token_id,short_term_frames=8,mid_term_frames=64)
 with torch.inference_mode():
  # One common untimed block warmup, not a condition's reusable cache.
  warm=vision(blocks[0].to('mps',dtype=torch.bfloat16),grid_thw=grid);torch.mps.synchronize();del warm
  for schedule,prefixes in [('dense',list(range(1,a.limit+1))),('questions',[n for n in [2,26,41] if n<=a.limit])]:
   policies={name:policy() for name in 'ABC'};counts={k:Counter(v) for k,v in policies.items()}
   inc=IncrementalMemory(policies['C']);cache_b=[];vision_counts={k:0 for k in 'ABC'}
   def encode(i,cond):
    vision_counts[cond]+=1
    return vision(blocks[i].to('mps',dtype=torch.bfloat16),grid_thw=grid)
   for n in prefixes:
    outputs={};raws={};states={};elapsed={};before={k:dict(v.counts) for k,v in counts.items()};vbefore=dict(vision_counts)
    # Rotate condition order; each still consumes only its own previous state.
    for cond in ['ABC','BCA','CAB'][(n-1)%3]:
     meter.context={'schedule':schedule,'blocks':n,'condition':cond}
     with meter.phase('visual_update') as ve:
      if cond=='A':raw=torch.cat([encode(i,cond) for i in range(n)])
      elif cond=='B':
       for i in range(len(cache_b),n):cache_b.append(encode(i,cond).detach().clone())
       raw=torch.cat(cache_b)
      else:
       # Stage new embeddings separately; append/maintenance is timed below.
       new=[encode(i,cond) for i in range(len(inc.raw),n)]
       raw=torch.cat(inc.raw+new)
     with meter.phase('memory_update_and_read') as me:
      if cond in 'AB':out=full_memory(policies[cond],raw,cfg,rotary,markers)
      else:
       for f in new:inc.append(f)
       out=inc.snapshot()
     outputs[cond]=out;raws[cond]=raw
     states[cond]=storage_report([]) if cond=='A' else (storage_report(cache_b) if cond=='B' else inc.state_report())
     elapsed[cond]={'visual_seconds':ve['seconds'],'memory_seconds':me['seconds']}
    equal={k:compare(outputs['A'],outputs[k]) for k in 'BC'}
    visual_equal={k:torch.equal(raws['A'],raws[k]) for k in 'BC'}
    # Read twice at the same prefix: must not change queues, masks or counters.
    keep_before=inc.keep.clone();counter_before=dict(counts['C'].counts)
    repeated=compare(inc.snapshot(),inc.snapshot())
    read_pure=torch.equal(keep_before,inc.keep) and counter_before==counts['C'].counts
    row={'schedule':schedule,'blocks':n,'prefix_seconds':n*2,'visual_bitwise_equal':visual_equal,'memory_equality':equal,'repeat_read_equal':repeated,'read_did_not_mutate_execution_state':read_pure,'state':states,'elapsed':elapsed,'counts':{k:dict(v.counts) for k,v in counts.items()},'counts_this_step':{k:{j:counts[k].counts[j]-before[k][j] for j in before[k]} for k in 'ABC'},'vision_blocks':dict(vision_counts),'vision_blocks_this_step':{k:vision_counts[k]-vbefore[k] for k in 'ABC'}}
    r['records'].append(row);save()
    print(schedule,n,'eq',equal,'vision',vision_counts,'state', {k:v['tensor_storage_bytes'] for k,v in states.items()},flush=True)
    if not (all(visual_equal.values()) and all(v['features_bitwise_equal'] for v in equal.values()) and read_pure):raise RuntimeError('Equivalence failure; inspect saved prefix, do not tune thresholds')
    if schedule=='questions':
     for cond in 'ABC':
      path=a.output/f'question_{n}_{cond}.pt'
      payload={'raw':raws[cond].cpu(),'indices':outputs[cond]['indices'].cpu(),'features':outputs[cond]['features'].cpu()}
      torch.save(payload,path)
      r['snapshots'].append({'blocks':n,'condition':cond,'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
     save()
    del outputs,raws,raw,out,new,keep_before
   # Explicit lifecycle boundary: resetting releases state, then first block replays cleanly.
   inc.reset();assert inc.state_report()['tensor_storage_bytes']==0
   r.setdefault('resets',[]).append({'schedule':schedule,'state_bytes_after_reset':0});save()
 r['status']='completed'
except Exception:
 r.update(status='failed',error=traceback.format_exc());raise
finally:
 meter.close();save()
