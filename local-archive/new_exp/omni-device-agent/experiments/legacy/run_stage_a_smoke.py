"""Run pretrained-weight intervention checks on DEVELOPMENT questions only.

This 0.5B run is deliberately labeled an implementation smoke test. It must not
be used for the pre-registered 7B quality go/no-go decision, layer selection,
latency claims, or any assertion about real edge/cloud workloads.
"""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import platform
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from fetch_stage_a_data import DATA_REVISION, EVAL_SHA
from fetch_stage_a_models import CACHE, MODELS, sha256
from stage_a_core import (additive_entry, assemble, capture, continuation_logits, distribution_metrics,
                          greedy, rebuild_interval, relative_error, require_compatible)


def evaluator():
    path = CACHE/'data'/DATA_REVISION/'hotpot_evaluate_v1.py'
    if sha256(path) != EVAL_SHA:
        raise ValueError('Upstream evaluator checksum mismatch')
    spec = importlib.util.spec_from_file_location('upstream_hotpot_evaluator',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synchronize(device):
    if device=='mps':
        torch.mps.synchronize()
    elif device=='cuda':
        torch.cuda.synchronize()


def main(args):
    if Path(args.output).exists():
        raise FileExistsError('Choose a new output; never silently overwrite an existing run')
    torch.manual_seed(20261024)
    torch.set_num_threads(4)
    dtype = getattr(torch,args.dtype)
    metadata = json.loads(Path(args.manifest).read_text())
    case_path = Path(metadata['cases_file'])
    if sha256(case_path)!=metadata['cases_sha256']:
        raise ValueError('Derived dataset checksum mismatch')
    cases = [json.loads(line) for line in case_path.read_text().splitlines()]
    ids = [q['id'] for q in metadata['questions'] if q['split']=='dev'][:args.limit]
    cases = [c for c in cases if c['id'] in ids and c['condition'] in args.conditions]
    if not cases or any(c['split']!='dev' for c in cases):
        raise ValueError('Smoke may use development questions only')
    models,tokenizers = {},{}
    for key,(_,revision,_) in MODELS.items():
        path = CACHE/'models'/revision
        models[key] = AutoModelForCausalLM.from_pretrained(
            path,local_files_only=True,trust_remote_code=False,
            dtype=dtype,attn_implementation='sdpa').to(args.device).eval()
        tokenizers[key] = AutoTokenizer.from_pretrained(path,local_files_only=True)
        print(f'loaded {key}: {revision}',flush=True)
    a,b = models['base'],models['instruct']
    require_compatible(a,b)
    if tokenizers['base'].get_vocab()!=tokenizers['instruct'].get_vocab():
        raise ValueError('Token vocabulary differs')
    tokenizer=tokenizers['instruct']
    eval_module=evaluator()
    stop_ids=set(b.generation_config.eos_token_id if isinstance(b.generation_config.eos_token_id,list)
                 else [b.generation_config.eos_token_id])
    report={'status':'running','purpose':'0.5B pretrained implementation validation; not main experiment',
            'hardware':{'platform':platform.platform(),'machine':platform.machine(),'device':args.device},
            'versions':{'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__},
            'models':{k:{'repo':v[0],'revision':v[1]} for k,v in MODELS.items()},
            'source_sha256':{name:sha256(name) for name in ['stage_a_core.py','run_stage_a_smoke.py','stage_a_data.py']},
            'settings':vars(args),'data_manifest_sha256':sha256(args.manifest),'cases':[]}
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        temporary=output.with_suffix(output.suffix+'.tmp')
        temporary.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        temporary.replace(output)
    save()
    for case in cases:
        began=time.monotonic()
        hs,ht,x,q=(case[k] for k in ('prefix_source_ids','prefix_target_ids','shared_ids','suffix_ids'))
        if len(hs)!=len(ht):
            raise ValueError('Prefix positions differ')
        # All variants use these exact token IDs, rather than independently templating the models.
        entries=sorted({0,args.start})
        states={}
        for key,model,prefix in [('As',a,hs),('At',a,ht),('Bs',b,hs),('Bt',b,ht)]:
            states[key]=capture(model,prefix,x,entries)
        s,t=states['As'],states['Bt']
        with torch.inference_mode():
            embedding_entry=b.get_input_embeddings()(torch.tensor([x],device=b.device)).cpu()
        if not torch.equal(embedding_entry,t.entries[0]):
            raise AssertionError('Layer-zero target entry must equal token embedding lookup')
        native_ids=greedy(b,t.kv,q,args.max_new_tokens,stop_ids)
        reference=continuation_logits(b,t.kv,q,native_ids)
        repeated=continuation_logits(b,t.kv,q,native_ids)
        with torch.inference_mode():
            dense=b(torch.tensor([ht+x+q+native_ids[:-1]],device=b.device),
                    logits_to_keep=len(native_ids)).logits[0].float().cpu()
        tol=2e-5 if dtype==torch.float32 else 0.03
        checks={'split_dense_logits_relative_error':relative_error(reference,dense),
                'repeated_continuation_relative_error':relative_error(reference,repeated),
                'B_layer0_context_relative_error':relative_error(states['Bs'].entries[0],t.entries[0]),
                'equal_prefix_positions':len(hs)==len(ht)}
        checks['split_dense_distribution']=distribution_metrics(reference,dense,native_ids)
        checks['strict_numerical_calibration_pass']=(
            checks['split_dense_logits_relative_error']<=2e-5 and
            checks['split_dense_distribution']['argmax_agreement']==1.0)
        if checks['split_dense_logits_relative_error']>tol or checks['repeated_continuation_relative_error']>tol:
            raise AssertionError(f'Continuation check failed: {checks}')
        if checks['B_layer0_context_relative_error']!=0:
            raise AssertionError('Layer-zero entry must not depend on preceding tokens')
        start=args.start
        e_as,e_at,e_bs,e_bt=[states[k].entries[start] for k in ['As','At','Bs','Bt']]
        if start==0:
            e_bt=embedding_entry
        e_add=additive_entry(e_as,e_at,e_bs)
        if case['condition']=='P0' and not torch.equal(e_add,e_bt):
            raise AssertionError('In P0, additive oracle must collapse to the joint oracle')
        entry_variants={'R2_source':e_as,'R3_context_oracle':e_at,'R4_model_oracle':e_bs,
                        'R5_additive_oracle':e_add,'R6_joint_oracle':e_bt}
        rebuilt={name:rebuild_interval(b,t,e,start,args.width) for name,e in entry_variants.items()}
        oracle_errors=[relative_error(got,full[:,:,len(ht):])
                       for layer,values in rebuilt['R6_joint_oracle'].items()
                       for got,full in zip(values,t.kv[layer])]
        checks['oracle_interval_kv_relative_error_max']=max(oracle_errors)
        if max(oracle_errors)>tol:
            raise AssertionError(f'Oracle reconstruction failed: {max(oracle_errors)}')
        variants={'R0_native':t.kv,'R1_direct':assemble(s,t)}
        # Unlike S1 below, this removes target-only history from both prefill
        # and decode. It measures history sensitivity, not KV repair quality.
        variants['H0_native_source_history']=states['Bs'].kv
        variants.update({name:assemble(s,t,rep) for name,rep in rebuilt.items()})
        variants['D_source_native_outside']=assemble(s,t,rebuilt['R2_source'],native_outside=True)
        variants['D_joint_native_outside']=assemble(s,t,rebuilt['R6_joint_oracle'],native_outside=True)
        # B->B controls isolate context mismatch with no cross-model difference.
        variants['S1_same_model_direct']=assemble(states['Bs'],t)
        variants['S2_same_model_source']=assemble(states['Bs'],t,rebuilt['R4_model_oracle'])
        variants['S6_same_model_joint']=assemble(states['Bs'],t,rebuilt['R6_joint_oracle'])
        scores={}
        for name,kv in variants.items():
            answer_ids=native_ids if name=='R0_native' else greedy(b,kv,q,args.max_new_tokens,stop_ids)
            prediction=tokenizer.decode(answer_ids,skip_special_tokens=True).strip()
            logits=reference if name=='R0_native' else continuation_logits(b,kv,q,native_ids)
            score=distribution_metrics(logits,reference,native_ids)
            score.update({'prediction':prediction,'generated_tokens':len(answer_ids),
                          'answer_f1':eval_module.f1_score(prediction,case['answer'])[0],
                          'answer_em':float(eval_module.exact_match_score(prediction,case['answer']))})
            if name in rebuilt:
                errors={label:[] for label in ['K','V']}
                for layer,values in rebuilt[name].items():
                    for label,got,full in zip(['K','V'],values,t.kv[layer]):
                        errors[label].append(relative_error(got,full[:,:,len(ht):]))
                score['interval_kv_relative_error_mean']={k:sum(v)/len(v) for k,v in errors.items()}
            scores[name]=score
        if case['condition']=='P0' and scores['S1_same_model_direct']['kl_reference_to_variant']>1e-6:
            raise AssertionError('Same-model same-prefix reuse must recover native continuation')
        if hs==ht and scores['H0_native_source_history']!=scores['R0_native']:
            raise AssertionError('Identical histories must give identical native outcomes')
        if case['condition']=='P0' and scores['R5_additive_oracle']!=scores['R6_joint_oracle']:
            raise AssertionError('Identical oracle entries must produce identical recorded outcomes')
        if start==0:
            if scores['R2_source']!=scores['R3_context_oracle']:
                raise AssertionError('Layer-zero source embedding cannot depend on source history')
            if not scores['R4_model_oracle']==scores['R5_additive_oracle']==scores['R6_joint_oracle']:
                raise AssertionError('Layer-zero target/context/additive entries must coincide')
        synchronize(args.device)
        result={'id':case['id'],'type':case['type'],'condition':case['condition'],'split':'dev',
                'prefix_tokens':len(ht),'shared_tokens':len(x),'gold_answer':case['answer'],
                'checks':checks,'joint_interaction_relative_norm':relative_error(e_add,e_bt),
                'target_entry_provenance':'target_embedding_lookup' if start==0 else 'full_target_forward_oracle',
                'scores':scores,'wall_seconds_not_a_benchmark':time.monotonic()-began}
        report['cases'].append(result)
        save()
        print(json.dumps({'case':len(report['cases']),'id':case['id'],'condition':case['condition'],
                          'checks':checks,'seconds':result['wall_seconds_not_a_benchmark']},ensure_ascii=False),flush=True)
        del states,variants,rebuilt,s,t,e_as,e_at,e_bs,e_bt,entry_variants
        gc.collect()
        if args.device=='mps':
            torch.mps.empty_cache()
    calibrated=all(c['checks']['strict_numerical_calibration_pass'] for c in report['cases'])
    report['status']='completed' if calibrated else 'completed_with_numerical_caveat'
    report['interpretation']='Structural checks passed; numerical calibration is reported separately. Not the 7B main experiment.'
    save()
    print(f'completed: {output}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',default='results/stage_a_data_manifest.json')
    parser.add_argument('--output',default='results/stage_a_smoke.json')
    parser.add_argument('--device',choices=['cpu','mps','cuda'],default='mps')
    parser.add_argument('--dtype',choices=['float32','bfloat16'],default='float32')
    parser.add_argument('--limit',type=int,default=2)
    parser.add_argument('--conditions',nargs='+',choices=['P0','P1','P2'],default=['P0','P1','P2'])
    parser.add_argument('--start',type=int,default=8)
    parser.add_argument('--width',type=int,default=4)
    parser.add_argument('--max-new-tokens',type=int,default=24)
    main(parser.parse_args())
