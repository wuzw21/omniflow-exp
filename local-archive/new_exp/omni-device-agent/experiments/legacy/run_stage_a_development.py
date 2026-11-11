"""7B development-set screen: same-budget source versus oracle entry.

Only development IDs are accepted. No test-set run, learned corrector, or
serving benchmark is implemented here. Model A and B can use separate GPUs.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import time

import torch
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from fetch_stage_a_data import EVAL_SHA
from fetch_stage_a_models import sha256
from stage_a_core import (assemble,capture,continuation_logits,distribution_metrics,
                         greedy,rebuild_interval,relative_error,require_compatible)


def load_cases(manifest_path,cases_path,limit,conditions):
    manifest=json.loads(Path(manifest_path).read_text())
    if not 1<=limit<=manifest['split_counts']['dev']:
        raise ValueError('Development limit must be between 1 and the development-set size')
    if sha256(cases_path)!=manifest['cases_sha256']:
        raise ValueError('Derived cases changed')
    ids=[q['id'] for q in manifest['questions'] if q['split']=='dev'][:limit]
    cases=[c for line in Path(cases_path).read_text().splitlines()
           if (c:=json.loads(line))['id'] in ids and c['condition'] in conditions]
    if len(cases)!=limit*len(set(conditions)) or any(c['split']!='dev' for c in cases):
        raise ValueError('Incomplete or invalid development cases')
    return manifest,cases


def main(args):
    if Path(args.output).exists():
        raise FileExistsError('Choose a new output; never silently overwrite or restart a run')
    torch.manual_seed(20261024)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    manifest,cases=load_cases(args.data_manifest,args.cases,args.limit,args.conditions)
    model_manifest=json.loads(Path(args.model_manifest).read_text())['models']
    models,tokenizers={},{}
    for role,device in [('base',args.source_device),('instruct',args.target_device)]:
        item=model_manifest[role]
        folder=Path(args.model_root)/item['revision']
        if json.loads((folder/'verified_manifest.json').read_text())!=item:
            raise ValueError('Model download manifest is missing or differs')
        models[role]=AutoModelForCausalLM.from_pretrained(folder,local_files_only=True,
            trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval()
        tokenizers[role]=AutoTokenizer.from_pretrained(folder,local_files_only=True)
        print(json.dumps({'loaded':role,'device':device,'revision':item['revision']}),flush=True)
    a,b=models['base'],models['instruct']
    require_compatible(a,b)
    tokenizer=tokenizers['instruct']
    vocab_hash=hashlib.sha256(json.dumps(tokenizer.get_vocab(),sort_keys=True).encode()).hexdigest()
    if tokenizer.get_vocab()!=tokenizers['base'].get_vocab() or vocab_hash!=manifest['tokenizer_vocab_sha256']:
        raise ValueError('Model and dataset token vocabularies differ')
    if any(s<0 or s+args.width>b.config.num_hidden_layers for s in args.starts):
        raise ValueError('Invalid interval')
    if len(set(args.starts))!=len(args.starts):
        raise ValueError('Duplicate interval starts')
    if sha256(args.evaluator)!=EVAL_SHA:
        raise ValueError('Official answer evaluator changed')
    spec=importlib.util.spec_from_file_location('hotpot_evaluator',args.evaluator)
    evaluation=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    stop=b.generation_config.eos_token_id
    stop=set(stop if isinstance(stop,list) else [stop])
    report={'status':'running','phase':'7B development screen; no held-out conclusions',
            'settings':vars(args),'models':model_manifest,
            'runtime':{'platform':platform.platform(),'torch':torch.__version__,
                       'transformers':transformers.__version__,'cuda':torch.version.cuda,
                       'gpu_names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]},
            'source_sha256':{f:sha256(f) for f in ['stage_a_core.py','run_stage_a_development.py']},
            'data_manifest_sha256':sha256(args.data_manifest),'cases':[]}
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        temporary=output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        temporary.replace(output)
    save()
    for case in cases:
        begin=time.monotonic()
        hs,ht,x,q=[case[k] for k in ['prefix_source_ids','prefix_target_ids','shared_ids','suffix_ids']]
        s=capture(a,hs,x,args.starts)
        t=capture(b,ht,x,args.starts)
        with torch.inference_mode():
            embedding_entry=b.get_input_embeddings()(torch.tensor([x],device=b.device)).cpu()
        if 0 in args.starts and not torch.equal(embedding_entry,t.entries[0]):
            raise AssertionError('Qwen2 layer-zero entry must equal target token embeddings')
        # This control must retain H_s in its prefix as well as its X cache.
        # Giving it H_t at decode would no longer remove the target evidence.
        bs=t if hs==ht else capture(b,hs,x,[])
        native_ids=greedy(b,t.kv,q,args.max_new_tokens,stop)
        reference=continuation_logits(b,t.kv,q,native_ids)
        repeated=continuation_logits(b,t.kv,q,native_ids)
        if not torch.equal(reference,repeated):
            raise AssertionError('Repeated native continuation changed')
        with torch.inference_mode():
            dense=b(torch.tensor([ht+x+q+native_ids[:-1]],device=b.device),
                    logits_to_keep=len(native_ids)).logits[0].float().cpu()
        calibration={'split_dense_relative_logit_error':relative_error(reference,dense),
                     **distribution_metrics(reference,dense,native_ids)}
        def score(kv,use_native=False):
            answer_ids=native_ids if use_native else greedy(b,kv,q,args.max_new_tokens,stop)
            logits=reference if use_native else continuation_logits(b,kv,q,native_ids)
            prediction=tokenizer.decode(answer_ids,skip_special_tokens=True).strip()
            return {'prediction':prediction,'tokens':len(answer_ids),
                    'f1':evaluation.f1_score(prediction,case['answer'])[0],
                    'em':float(evaluation.exact_match_score(prediction,case['answer'])),
                    **distribution_metrics(logits,reference,native_ids)}
        native=score(t.kv,True)
        direct=score(assemble(s,t))
        history_controls={
            'native_source_history':score(bs.kv),
            'source_history_shared_with_target_prefix':score(assemble(bs,t)),
        }
        if hs==ht and any(value!=native for value in history_controls.values()):
            raise AssertionError('Identical histories must give identical history-control outcomes')
        intervals=[]
        for start in args.starts:
            results={}
            target_entry=embedding_entry if start==0 else t.entries[start]
            for label,entry in [('source',s.entries[start]),('oracle',target_entry)]:
                rep=rebuild_interval(b,t,entry,start,args.width)
                errors=[relative_error(got,full[:,:,len(ht):]) for layer,pair in rep.items()
                        for got,full in zip(pair,t.kv[layer])]
                results[label]=score(assemble(s,t,rep))
                results[label]['interval_kv_error_max']=max(errors)
                if label=='oracle' and max(errors)>0.03:
                    raise AssertionError('Native-entry interval reconstruction error exceeds 3%')
            intervals.append({'start':start,'width':args.width,'scores':results,
                'target_entry_provenance':'target_embedding_lookup' if start==0 else 'full_target_forward_oracle'})
        row={'id':case['id'],'condition':case['condition'],'split':'dev','type':case['type'],
             'prefix_tokens':len(ht),'shared_tokens':len(x),'gold_answer':case['answer'],
             'calibration':calibration,'native':native,'direct':direct,'intervals':intervals,
             'history_controls':history_controls,
             'wall_seconds_not_a_benchmark':time.monotonic()-begin}
        report['cases'].append(row)
        save()
        print(json.dumps({'completed_cases':len(report['cases']),'id':case['id'],
                          'condition':case['condition'],'calibration':calibration,
                          'wall_seconds':row['wall_seconds_not_a_benchmark']}),flush=True)
    report['status']='completed'
    report['interpretation']='Development results only. Numerical drift must be assessed before selecting intervals.'
    save()
    print('DEVELOPMENT_SCREEN_COMPLETED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data-manifest',default='results/stage_a_data_manifest.json')
    p.add_argument('--cases',required=True)
    p.add_argument('--model-manifest',default='stage_a_7b_models.json')
    p.add_argument('--model-root',required=True)
    p.add_argument('--evaluator',required=True)
    p.add_argument('--source-device',default='cuda:0')
    p.add_argument('--target-device',default='cuda:1')
    p.add_argument('--limit',type=int,default=2)
    p.add_argument('--conditions',nargs='+',choices=['P0','P1','P2'],default=['P0','P2'])
    p.add_argument('--starts',nargs='+',type=int,default=[0,4,8,12,16,20,24])
    p.add_argument('--width',type=int,default=4)
    p.add_argument('--max-new-tokens',type=int,default=64)
    p.add_argument('--output',required=True)
    main(p.parse_args())
