"""Native 7B prefill timing on fixed development inputs, without KV migration.

Measure both cold-prefix and already-cached-prefix requests using upstream
Transformers. Cache preparation, token transfer and model loading are outside
timing. These are isolated warm-model timings, not production service latency.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch
import transformers
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from run_stage_a_development import load_cases
from stage_a_core import relative_error


@torch.inference_mode()
def main(args):
    output=Path(args.output)
    if output.exists():
        raise FileExistsError('Do not overwrite or restart a timing run')
    if args.repeats<3 or args.warmup<1 or not args.device.startswith('cuda:'):
        raise ValueError('Require CUDA, warmup and at least three measured repetitions')
    manifest,cases=load_cases(args.data_manifest,args.cases,args.limit,['P2'])
    item=json.loads(Path(args.model_manifest).read_text())['models']['instruct']
    folder=Path(args.model_root)/item['revision']
    if json.loads((folder/'verified_manifest.json').read_text())!=item:
        raise ValueError('Pinned Instruct model is not verified')
    index=int(args.device.split(':')[1])
    probe=subprocess.run(['nvidia-smi',f'--id={index}','--query-gpu=uuid,name,memory.free,utilization.gpu',
                          '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=30)
    fields=[s.strip() for s in probe.stdout.strip().split(',')]
    if int(fields[2])<20000 or int(fields[3])>=10:
        raise RuntimeError('Selected GPU is not sufficiently idle/free; no model loaded')
    torch.manual_seed(20261024)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(folder,local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,attn_implementation='sdpa').to(args.device).eval()
    result={'status':'running','purpose':__doc__,'settings':vars(args),'model':item,
            'gpu_before_run':{'uuid':fields[0],'name':fields[1],'free_MiB':int(fields[2]),'utilization_percent':int(fields[3])},
            'runtime':{'torch':torch.__version__,'transformers':transformers.__version__,'cuda':torch.version.cuda},
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'data_manifest_sha256':hashlib.sha256(Path(args.data_manifest).read_bytes()).hexdigest(),'cases':[],
            'limits':['No network transfer, serialization, migration, repair, queuing, or load time is measured.',
                      'Native prefix cache is cloned outside the timed region; weights and request tensors are already on GPU.',
                      'TTFT component only: forward through suffix and final logits; no autoregressive answer generation.',
                      'Network numbers are analytic no-prefetch/no-overlap sensitivities, not measurements.',
                      'A bandwidth crossover ignoring repair cost is an optimistic necessary condition, not a speedup guarantee.']}
    def save():
        temporary=output.with_suffix('.tmp')
        temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(output)
    output.parent.mkdir(parents=True,exist_ok=True);save()
    for case in cases:
        h,x,q=(case[k] for k in ('prefix_target_ids','shared_ids','suffix_ids'))
        full=torch.tensor([h+x+q],device=args.device)
        tail=torch.tensor([x+q],device=args.device)
        prefix=model.model(torch.tensor([h],device=args.device),use_cache=True).past_key_values.to_legacy_cache()
        prefix=tuple((k.clone(),v.clone()) for k,v in prefix)
        last_logits={};timing={}
        for mode in ('cold_prefix','cached_prefix'):
            measured=[]
            for repeat in range(args.warmup+args.repeats):
                cache=None if mode=='cold_prefix' else DynamicCache.from_legacy_cache(tuple((k.clone(),v.clone()) for k,v in prefix))
                if cache is not None and cache.get_seq_length()!=len(h):
                    raise AssertionError('A prior timed request mutated the prefix')
                torch.cuda.synchronize(args.device)
                began=time.perf_counter()
                out=model(full if cache is None else tail,past_key_values=cache,use_cache=True,logits_to_keep=1)
                torch.cuda.synchronize(args.device)
                elapsed=time.perf_counter()-began
                if out.past_key_values.get_seq_length()!=len(h+x+q):
                    raise AssertionError('Request accidentally reused a previous request continuation')
                last_logits[mode]=out.logits[0].float().cpu()
                if repeat>=args.warmup:measured.append(elapsed)
                del out,cache
            timing[mode]={'wall_seconds':measured,'median_seconds':statistics.median(measured),
                          'min_seconds':min(measured),'max_seconds':max(measured)}
        config=model.config;n=len(x);d=config.hidden_size//config.num_attention_heads
        layer_bytes=n*2*config.num_key_value_heads*d*2
        entry_bytes=n*config.hidden_size*2
        payload=(config.num_hidden_layers-4)*layer_bytes+entry_bytes
        # Common-tokenizer uint32 token IDs are a controlled transport baseline,
        # not a claim about an existing provider's wire format.
        token_bytes=4*len(x+q)
        native=timing['cached_prefix']['median_seconds']
        row={'id':case['id'],'split':'dev','condition':'P2','prefix_tokens':len(h),'shared_tokens':n,'suffix_tokens':len(q),
             'timing':timing,'cached_vs_cold_last_logit_relative_error':relative_error(last_logits['cached_prefix'],last_logits['cold_prefix']),
             'analytic_payload':{'kv_outside_four_layers_plus_one_E_bytes':payload,'uint32_input_token_bytes':token_bytes,
                 'state_serialization_seconds_at_100Mbps':payload*8/100e6,
                 'optimistic_no_overlap_crossover_Mbps':max(0,payload-token_bytes)*8/(native*1e6)}}
        result['cases'].append(row);save()
        print(json.dumps(row),flush=True)
    result['status']='completed';save()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data-manifest',default='results/stage_a_data_manifest.json')
    p.add_argument('--cases',required=True)
    p.add_argument('--model-manifest',default='stage_a_7b_models.json')
    p.add_argument('--model-root',required=True)
    p.add_argument('--device',default='cuda:2')
    p.add_argument('--limit',type=int,default=2)
    p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--repeats',type=int,default=7)
    p.add_argument('--output',required=True)
    main(p.parse_args())
