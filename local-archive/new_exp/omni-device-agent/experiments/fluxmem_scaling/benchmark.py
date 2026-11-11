"""Scaling benchmark for B versus C2, memory-only or real frontend."""
import argparse, gc, json, statistics, subprocess, time, traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--features',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cadence',type=int,required=True);p.add_argument('--pairs',type=int,default=2);p.add_argument('--frontend',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
from common import setup, policy, schedule, sha, freeze, full_memory, compare, Counter
from compact import CompactMemory
from measure import Meter
import torch
assert torch.backends.mps.is_available();torch.set_num_threads(4);torch.manual_seed(1234)
fm=json.loads((a.features/'results.json').read_text()); features=torch.load(fm['features_path'],weights_only=True).to('mps')
im=json.loads((a.inputs/'manifest.json').read_text()); pixels=torch.load(im['blocks_path'],weights_only=True)
cfg,markers,rotary,vision=setup(a.frontend); grid=torch.tensor([[1,16,30]],device='mps')
qs=schedule(a.cadence); assert qs[-1]==94
r={'status':'running','mode':'frontend' if a.frontend else 'memory_only','cadence_seconds':a.cadence,'queries':qs,'pairs':a.pairs,'features_sha256':sha(fm['features_path']),'inputs_sha256':sha(a.inputs/'manifest.json'),'device':'mps','chip':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),'runs':[],'source_hashes':{}}
for name in ['benchmark.py','common.py','PROTOCOL.md']:
 f=Path(__file__).with_name(name);(a.output/name).write_bytes(f.read_bytes());r['source_hashes'][name]=sha(f)
meter=Meter(torch)
def save():(a.output/'results.json').write_text(json.dumps(r,indent=2)+'\n')
def run(cond,rep,warm=False):
 p=policy(cfg); counter=Counter(p); state=CompactMemory(p) if cond=='C' else None; cache=[]; events=[]; cursor=0; vision_blocks=0
 gc.collect();torch.mps.empty_cache();torch.mps.synchronize()
 for n in qs:
  with meter.phase('visual_update',blocks=n) as ve:
   new=[]
   for i in range(cursor,n):
    if a.frontend:
     x=vision(pixels[i].to('mps',dtype=torch.bfloat16),grid_thw=grid);vision_blocks+=1
    else:x=features[i]
    new.append(x)
  with meter.phase('memory_update_readout',blocks=n) as me:
   if cond=='B':
    cache.extend(x.detach().clone() for x in new); raw=torch.cat(cache); out=full_memory(p,raw,cfg,rotary,markers)
   else:
    for x in new:state.append(x)
    out=state.snapshot()
  # Use the independent audit as a per-query oracle.
  events.append({'blocks':n,'seconds':2*n,'visual_seconds':ve['seconds'],'memory_seconds':me['seconds'],'retained_tokens':len(out['indices']),'state_bytes':state.state_report()['tensor_storage_bytes'] if state else sum(x.numel()*x.element_size() for x in cache),'vision_blocks':vision_blocks,'memory_counts':dict(counter.counts),'visual_peak':ve['sampled_peak'],'memory_peak':me['sampled_peak']})
  cursor=n
  if cond=='B':del raw
  del new,out,x
  gc.collect();torch.mps.synchronize()
 result={'condition':cond,'rep':rep,'warmup':warm,'events':events,'total_visual_seconds':sum(x['visual_seconds'] for x in events),'total_memory_seconds':sum(x['memory_seconds'] for x in events),'total_seconds':sum(x['visual_seconds']+x['memory_seconds'] for x in events),'final_state_bytes':events[-1]['state_bytes'],'vision_blocks':vision_blocks,'counts':dict(counter.counts)}
 del state,cache;gc.collect();torch.mps.synchronize();return result
try:
 with torch.inference_mode():
  for cond in ['B','C']:
   result=run(cond,-1,True);r['runs'].append(result);save();print('warmup',cond,r['cadence_seconds'],result['total_seconds'],flush=True)
  for rep in range(a.pairs):
   for cond in (['B','C'] if rep%2==0 else ['C','B']):
    result=run(cond,rep);r['runs'].append(result);save();print('run',a.cadence,rep,cond,result['total_seconds'],flush=True)
 vals=[x for x in r['runs'] if not x['warmup']];summary={}
 for metric in ['total_seconds','total_memory_seconds','total_visual_seconds']:
  b=[x[metric] for x in vals if x['condition']=='B'];c=[x[metric] for x in vals if x['condition']=='C'];d=[x-y for x,y in zip(b,c)]
  summary[metric]={'B':b,'C':c,'B_minus_C':d,'B_median':statistics.median(b),'C_median':statistics.median(c),'mean_saving':statistics.mean(d)}
 r.update(status='completed',summary=summary)
except BaseException:r.update(status='failed',error=traceback.format_exc());raise
finally:meter.close();save()
