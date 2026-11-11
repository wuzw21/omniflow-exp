"""Execute the author's unmodified model; instrument, never replace its memory policy."""
import argparse
import functools
import gc
import hashlib
import json
from pathlib import Path
import platform
import os
import resource
import subprocess
import sys
import traceback

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError('Use a new output file.')
    if not (args.model/'verified_manifest.json').exists():
        raise RuntimeError('Complete and verify the official weight download first.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for name in ['run_profile.py','measure.py','PROTOCOL.md']:
        (args.output.parent/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    sys.path.insert(0,str(args.upstream/'models/qwen2-5-vl/src'))
    import torch
    import transformers
    from qwen2_5_vl_fluxmem import Qwen2_5_VLForConditionalGeneration,Qwen2_5_VLProcessor
    from measure import Meter
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    assert torch.backends.mps.is_available(), 'This run must execute on the physical Mac MPS device.'
    meter=Meter(torch,trace_path=args.output.parent/"phases.jsonl")
    prepared=json.loads(args.prepared.read_text())
    result={'status':'running','mode':'smoke' if args.smoke else 'profile',
        'hardware':{'chip':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
                    'ram_bytes':int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True)),
                    'macos':platform.mac_ver()[0]},
        'mps_fallback_env':os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),'torch':torch.__version__,'transformers':transformers.__version__,
        'upstream_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.upstream,text=True).strip(),
        'upstream_diff':subprocess.check_output(['git','diff','--stat'],cwd=args.upstream,text=True),
        'model':json.loads((args.model/'verified_manifest.json').read_text()),
        'config':{'dtype':'bfloat16','device':'mps','attention':'sdpa','short_frames':8,'medium_frames':64,
                  'video_chunk_t':2,'max_new_tokens':32,'do_sample':False},
        'memory_measurement':'10ms sampled peaks plus synchronized boundaries; RSS and MPS overlap on UMA',
        'prepared_manifest_sha256':hashlib.sha256(args.prepared.read_bytes()).hexdigest(),
        'source_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest()
                         for f in [Path(__file__),Path(__file__).with_name('measure.py'),Path(__file__).with_name('PROTOCOL.md')]},
        'records':[]}
    def save():
        temp=args.output.with_suffix('.tmp')
        temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(args.output)
    save()
    try:
        with meter.phase('cold_model_load') as loading:
            model,info=Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model,local_files_only=True,torch_dtype=torch.bfloat16,
                attn_implementation='sdpa',device_map={'':'mps'},low_cpu_mem_usage=True,
                output_loading_info=True)
            model.eval()
        result['loading_info']=info
        if info.get('missing_keys') or info.get('unexpected_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
            raise RuntimeError('Weight load did not match the author model: '+str(info))
        result['load']=loading
        result['parameter_bytes']=sum(p.numel()*p.element_size() for p in model.parameters())
        result['parameter_devices']=sorted({str(p.device) for p in model.parameters()})
        result['parameter_dtypes']=sorted({str(p.dtype) for p in model.parameters()})
        result['resident_after_load']=meter.snapshot()
        processor=Qwen2_5_VLProcessor.from_pretrained(args.model,local_files_only=True)
        print('MODEL_LOADED',result['resident_after_load'],flush=True)
        save()
        forward_stack=[];vision_stack=[]
        def kv_bytes(cache):
            if cache is None:return 0
            return sum(t.numel()*t.element_size() for pair in zip(cache.key_cache,cache.value_cache)
                       for t in pair if isinstance(t,torch.Tensor))
        def before_forward(module,inputs,kwargs):
            ids=kwargs.get('input_ids',inputs[0] if inputs else None)
            n=ids.shape[1] if ids is not None else kwargs['inputs_embeds'].shape[1]
            forward_stack.append(meter.begin('prefill_forward' if n>1 else 'decode_forward',input_tokens=n))
        def after_forward(module,inputs,kwargs,out):
            event=forward_stack.pop()
            meter.end(event,logit_positions=out.logits.shape[1],kv_bytes=kv_bytes(out.past_key_values),
                      kv_seq_length=out.past_key_values.get_seq_length())
        def before_vision(module,inputs,kwargs):
            vision_stack.append(meter.begin('vision',patch_tokens=inputs[0].shape[0],grid=kwargs['grid_thw'].tolist()))
        def after_vision(module,inputs,kwargs,out):
            meter.end(vision_stack.pop(),output_tokens=out.shape[0])
        hooks=[model.register_forward_pre_hook(before_forward,with_kwargs=True),
               model.register_forward_hook(after_forward,with_kwargs=True),
               model.visual.register_forward_pre_hook(before_vision,with_kwargs=True),
               model.visual.register_forward_hook(after_vision,with_kwargs=True)]
        original=model.model.fluxmem.process_memory_streaming
        @functools.wraps(original)
        def traced_memory(*a,**kw):
            event=meter.begin('memory_compression',input_tokens=kw['hidden_states'].shape[1])
            out=original(*a,**kw)
            kept=out[-1][0].detach().cpu().tolist()
            meter.end(event,output_tokens=out[0].shape[1],kept_indices=kept)
            return out
        model.model.fluxmem.process_memory_streaming=traced_memory
        def run_one(repeat,index,enabled,warmup=False):
            meter.context={'repeat':repeat,'query_index':index,'fluxmem':enabled,'warmup':warmup}
            row=prepared['records'][index]
            input_path=Path(row['input_path'])
            if hashlib.sha256(input_path.read_bytes()).hexdigest()!=row['input_sha256']:
                raise RuntimeError('Changed prepared input')
            tensors=torch.load(input_path,map_location='cpu',weights_only=True)
            gc.collect();torch.mps.empty_cache();meter.sync()
            start=len(meter.events)
            with meter.phase('input_transfer'):
                inputs={k:(v.to('mps') if isinstance(v,torch.Tensor) else v) for k,v in tensors.items()}
            with torch.inference_mode(),meter.phase('generation') as generation:
                output=model.generate(**inputs,max_new_tokens=32,do_sample=False,use_fluxmem=enabled,
                                      short_frames=8,medium_frames=64,video_chunk_t=2)
            answer_ids=output[0,inputs['input_ids'].shape[1]:].detach().cpu().tolist()
            text=processor.decode(answer_ids,skip_special_tokens=True)
            events=meter.events[start:]
            pre=[e for e in events if e['stage']=='prefill_forward']
            if len(pre)!=1:raise RuntimeError('Unexpected number of prefill forwards')
            vision=sum(e['seconds'] for e in events if e['stage']=='vision')
            memory=sum(e['seconds'] for e in events if e['stage']=='memory_compression')
            record={'repeat':repeat,'query_index':index,'query_id':row['query']['question_id'],
                    'timestamp':row['query']['time_stamp'],'fluxmem':enabled,'warmup':warmup,
                    'answer':text,'answer_ids':answer_ids,'gold':row['query']['answer'],
                    'exact_letter_correct':text.strip()==row['query']['answer'],
                    'generated_tokens':len(answer_ids),'hit_length_limit':len(answer_ids)>=32,
                    'vision_seconds':vision,'memory_seconds':memory,
                    'first_forward_seconds':pre[0]['seconds'],
                    'language_prefill_and_other_seconds':pre[0]['seconds']-vision-memory,
                    'generation_seconds':generation['seconds'],
                    'qa_after_visual_memory_seconds':generation['seconds']-vision-memory,
                    'decode_forward_seconds':sum(e['seconds'] for e in events if e['stage']=='decode_forward'),
                    'first_forward_kv_bytes':pre[0]['kv_bytes'],'events':events}
            result['records'].append(record);save()
            print('DONE',repeat,index,enabled,round(generation['seconds'],3),repr(text),flush=True)
            del inputs,tensors,output
        if args.smoke:
            run_one(0,0,True)
            run_one(0,2,True)
        else:
            for enabled in [False,True]:run_one(-1,0,enabled,warmup=True)
            for repeat in range(3):
                for index in range(len(prepared['records'])):
                    for enabled in ([False,True] if repeat%2==0 else [True,False]):
                        run_one(repeat,index,enabled)
        for h in hooks:h.remove()
        model.model.fluxmem.process_memory_streaming=original
        result['status']='completed'
    except Exception:
        result.update(status='failed',error=traceback.format_exc())
        raise
    finally:
        result['process_maxrss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['remaining_active_stages']=[{k:v for k,v in e.items() if k!='_start'} for e in meter.active]
        meter.close();save()

if __name__=='__main__':main()
