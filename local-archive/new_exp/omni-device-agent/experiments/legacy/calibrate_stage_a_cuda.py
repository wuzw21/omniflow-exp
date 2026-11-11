"""Native-to-native interval reconstruction on the actual 7B CUDA model.

No source model, model mismatch, context mismatch, or learned correction.
All outside KV and the selected interval entry come from the same native
target snapshot. This measures the numerical floor of interval rebuilding.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch
import transformers
from transformers import AutoModelForCausalLM
from run_stage_a_development import load_cases
from stage_a_core import assemble,capture,continuation_logits,distribution_metrics,greedy,rebuild_interval,relative_error


def main(args):
    output=Path(args.output)
    if output.exists():raise FileExistsError('Do not overwrite a calibration run')
    _,cases=load_cases('results/stage_a_data_manifest.json',args.cases,2,['P0','P2'])
    item=json.loads(Path('stage_a_7b_models.json').read_text())['models']['instruct']
    folder=Path(args.model_root)/item['revision']
    if json.loads((folder/'verified_manifest.json').read_text())!=item:
        raise ValueError('Pinned target weights are not verified')
    gpu=subprocess.run(['nvidia-smi','--id=2','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],
                       capture_output=True,text=True,check=True,timeout=30)
    free,util=(int(s.strip()) for s in gpu.stdout.strip().split(','))
    if free<20000 or util>=10:raise RuntimeError('GPU 2 is not idle/free')
    torch.set_num_threads(4);torch.manual_seed(20261024)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(folder,local_files_only=True,trust_remote_code=False,
                dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda:2').eval()
    stop=model.generation_config.eos_token_id;stop=set(stop if isinstance(stop,list) else [stop])
    starts=list(range(0,28,4))
    report={'status':'running','scope':__doc__,'model_revision':item['revision'],
            'device':'cuda:2','gpu_name':torch.cuda.get_device_name(2),
            'torch':torch.__version__,'transformers':transformers.__version__,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'data_manifest_sha256':hashlib.sha256(Path('results/stage_a_data_manifest.json').read_bytes()).hexdigest(),
            'cases':[]}
    def save():
        tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(output)
    save()
    for case in cases:
        h,x,q=(case[k] for k in ('prefix_target_ids','shared_ids','suffix_ids'))
        t=capture(model,h,x,starts)
        native=greedy(model,t.kv,q,64,stop)
        reference=continuation_logits(model,t.kv,q,native)
        rows=[]
        for start in starts:
            rep=rebuild_interval(model,t,t.entries[start],start,4)
            kv=assemble(t,t,rep)
            ids=greedy(model,kv,q,64,stop)
            logits=continuation_logits(model,kv,q,native)
            errors=[relative_error(got,full[:,:,len(h):]) for layer,pair in rep.items() for got,full in zip(pair,t.kv[layer])]
            rows.append({'start':start,'width':4,'generated_token_ids_identical':ids==native,
                         'interval_kv_error_max':max(errors),'relative_logit_error':relative_error(logits,reference),
                         **distribution_metrics(logits,reference,native)})
        row={'id':case['id'],'condition':case['condition'],'split':'dev','native_generated_token_ids':native,'intervals':rows}
        report['cases'].append(row);save()
        print(json.dumps(row),flush=True)
    unchanged=all(i['generated_token_ids_identical'] and i['argmax_agreement']==1 for c in report['cases'] for i in c['intervals'])
    report['status']='completed_outputs_unchanged_on_this_pilot' if unchanged else 'completed_with_output_changes'
    report['interpretation']='Measured numerical floor, not exact KV equivalence and not calibration of the full development/test population.'
    save()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cases',required=True)
    p.add_argument('--model-root',required=True)
    p.add_argument('--output',required=True)
    main(p.parse_args())
