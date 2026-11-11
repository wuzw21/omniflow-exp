"""Full author model: exact QA-input/logit/token checks for all three controls."""
import argparse,json,sys,hashlib,subprocess,traceback,functools,gc
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
meta=json.loads(a.inputs.read_text());audit=json.loads(a.audit.read_text())
assert audit['status']=='completed' and len(audit['snapshots'])==9
assert audit['input_manifest_sha256']==hashlib.sha256(a.inputs.read_bytes()).hexdigest()
a.output.mkdir(parents=True,exist_ok=False)
up=Path.home()/'.agent-reach/research/fluxmem-repro';modeldir=Path.home()/'.agent-reach/research/fluxmem-qwen25vl3b/66285546d2b821cf421d4f5eb2576359d3770cd3'
sys.path[:0]=[str(up/'models/qwen2-5-vl/src'),str(Path(__file__).resolve().parent.parent/'fluxmem')]
import torch
from qwen2_5_vl_fluxmem import Qwen2_5_VLForConditionalGeneration,Qwen2_5_VLProcessor
from qwen2_5_vl_fluxmem.utils import scan_visual_indices,right_pad_and_stack
from measure import Meter
from state import Counter
assert torch.backends.mps.is_available();torch.set_num_threads(4);torch.manual_seed(1234)
meter=Meter(torch,trace_path=a.output/'phases.jsonl')
r={'status':'running','torch':torch.__version__,'device':'mps','chip':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),'input_manifest_sha256':hashlib.sha256(a.inputs.read_bytes()).hexdigest(),'audit_sha256':hashlib.sha256(a.audit.read_bytes()).hexdigest(),'records':[],'source_hashes':{}}
for name in ['qa.py','state.py','PROTOCOL.md']:
 f=Path(__file__).with_name(name);(a.output/name).write_bytes(f.read_bytes());r['source_hashes'][name]=hashlib.sha256(f.read_bytes()).hexdigest()
def save():
 t=a.output/'results.tmp';t.write_text(json.dumps(r,indent=2)+'\n');t.replace(a.output/'results.json')
def flatten_out(out):return [out[0],out[1][0],out[1][1],out[2],out[3],out[4][0]]
def equal_out(a,b):
 names=['features','rope_cos','rope_sin','position_ids','attention_mask','kept_indices']
 return {k:{'bitwise_equal':torch.equal(x,y),'shape_a':list(x.shape),'shape_b':list(y.shape),'max_abs_error':float((x.float()-y.float()).abs().max()) if x.shape==y.shape else None} for k,x,y in zip(names,a,b)}
save()
try:
 with meter.phase('load_model') as load:
  model,info=Qwen2_5_VLForConditionalGeneration.from_pretrained(modeldir,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'mps'},low_cpu_mem_usage=True,output_loading_info=True)
  model.eval()
 assert all(not v for v in info.values()),info
 r.update(loading_info=info,load=load,parameter_devices=sorted({str(v.device) for v in model.parameters()}));save()
 proc=Qwen2_5_VLProcessor.from_pretrained(modeldir,local_files_only=True)
 policy=model.model.fluxmem;counter=Counter(policy)
 orig_visual=model.visual.forward;orig_memory=policy.process_memory_streaming
 for qi,row in enumerate(meta['queries']):
  n=row['blocks'];reference=None;reference_ids=None;reference_logits=None
  for cond in 'ABC':
   src=next(s for s in audit['snapshots'] if s['blocks']==n and s['condition']==cond)
   assert hashlib.sha256(Path(src['path']).read_bytes()).hexdigest()==src['sha256']
   snap_cpu=torch.load(src['path'],weights_only=True)
   snap={k:v.to('mps') for k,v in snap_cpu.items()};del snap_cpu
   assert hashlib.sha256(Path(row['input_path']).read_bytes()).hexdigest()==row['sha256']
   inp={k:v.to('mps') if isinstance(v,torch.Tensor) else v for k,v in torch.load(row['input_path'],weights_only=True).items()}
   vision_calls=0;cache_reads=0;cursor=0;captured=[];logits=[];counts_before=dict(counter.counts)
   @functools.wraps(orig_visual)
   def visual_forward(pixels,grid_thw):
    global vision_calls,cache_reads,cursor
    if cond=='A':vision_calls+=1;return orig_visual(pixels,grid_thw=grid_thw)
    assert grid_thw.tolist()==[[1,16,30]]
    cache_reads+=1;out=snap['raw'][cursor:cursor+120];cursor+=120;return out
   @functools.wraps(orig_memory)
   def memory_forward(*args,**kw):
    if cond!='C':out=orig_memory(*args,**kw)
    else:
     visual=scan_visual_indices(kw['input_ids'][0],policy.vision_start_token_id,policy.vision_end_token_id)
     assert torch.equal(kw['hidden_states'][0,visual],snap['raw'])
     selected=visual[snap['indices']]
     hidden=kw['hidden_states'].clone();hidden[0,selected]=snap['features']
     mask=torch.ones(hidden.shape[1],device='mps',dtype=torch.bool);mask[visual]=False;mask[selected]=True
     kept=mask.nonzero(as_tuple=True)[0]
     sample=policy._pack_sample(hidden,kw['position_ids'],kw['position_embeddings'],0,kept)
     out=(*right_pad_and_stack([sample],pos_key1='pos_emb1',pos_key2='pos_emb2'),[kept])
    captured.append([t.detach().cpu().clone() for t in flatten_out(out)])
    return out
   def logits_hook(module,args,kw,out):
    if not logits:logits.append(out.logits[:,-1].detach().float().cpu().clone())
   model.visual.forward=visual_forward;policy.process_memory_streaming=memory_forward
   hook=model.register_forward_hook(logits_hook,with_kwargs=True)
   meter.context={'query_index':qi,'condition':cond,'blocks':n}
   gc.collect();torch.mps.empty_cache()
   with torch.inference_mode(),meter.phase('generate_equivalence_check') as event:
    output=model.generate(**inp,max_new_tokens=32,do_sample=False,use_fluxmem=True,short_frames=8,medium_frames=64,video_chunk_t=1)
   hook.remove();model.visual.forward=orig_visual;policy.process_memory_streaming=orig_memory
   ids=output[0,inp['input_ids'].shape[1]:].detach().cpu();answer=proc.decode(ids,skip_special_tokens=True)
   assert len(captured)==1
   if cond=='A':reference=captured[0];reference_ids=ids;reference_logits=logits[0]
   check=equal_out(reference,captured[0]);logitdiff=(reference_logits-logits[0]).abs()
   record={'query_index':qi,'blocks':n,'condition':cond,'answer':answer,'gold':row['query']['answer'],'generated_ids':ids.tolist(),'generated_ids_equal_to_A':torch.equal(ids,reference_ids),'memory_outputs_vs_A':check,'first_logits_bitwise_equal':torch.equal(reference_logits,logits[0]),'first_logits_max_abs_error':logitdiff.max().item(),'first_logits_argmax_equal':torch.equal(reference_logits.argmax(-1),logits[0].argmax(-1)),'actual_vision_calls':vision_calls,'cached_vision_block_reads':cache_reads,'memory_counts':{k:counter.counts[k]-counts_before[k] for k in counter.counts},'measurement':event}
   r['records'].append(record);save()
   print('QA',qi,cond,repr(answer),'memory',all(v['bitwise_equal'] for v in check.values()),'logits',record['first_logits_bitwise_equal'],'vision',vision_calls,flush=True)
   torch.save({'first_logits':logits[0],'generated_ids':ids},a.output/f'query_{qi}_{cond}_outputs.pt')
   assert all(v['bitwise_equal'] for v in check.values()),'Memory input differs'
   assert record['first_logits_bitwise_equal'] and record['generated_ids_equal_to_A'],'QA differs; record numerical discrepancy'
   del inp,snap,captured,output,logits
 r['status']='completed'
except Exception:
 r.update(status='failed',error=traceback.format_exc());raise
finally:
 meter.close();save()
