"""Measure no-intervention prefill chunking drift before interpreting KV repair."""
import gc
import json
from pathlib import Path
import platform

import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from fetch_stage_a_models import CACHE,MODELS,sha256
from stage_a_core import capture,continuation_logits,distribution_metrics,relative_error


def main():
    manifest=json.loads(Path('results/stage_a_data_manifest.json').read_text())
    case_path=Path(manifest['cases_file'])
    if sha256(case_path)!=manifest['cases_sha256']:
        raise ValueError('Dataset changed')
    ids=[q['id'] for q in manifest['questions'] if q['split']=='dev'][:4]
    cases=[c for line in case_path.read_text().splitlines()
           if (c:=json.loads(line))['id'] in ids and c['condition']=='P2']
    path=CACHE/'models'/MODELS['instruct'][1]
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    result={'purpose':'No KV transfer; compare segmented and unsplit prefill on a fixed gold continuation',
            'platform':platform.platform(),'device':'mps','model_revision':MODELS['instruct'][1],
            'torch':torch.__version__,'cases':[]}
    for dtype in [torch.float32,torch.bfloat16]:
        model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,trust_remote_code=False,
               dtype=dtype,attn_implementation='sdpa').to('mps').eval()
        for c in cases:
            h,x,q=[c[k] for k in ['prefix_target_ids','shared_ids','suffix_ids']]
            y=tokenizer.encode(c['answer'],add_special_tokens=False)+[tokenizer.eos_token_id]
            snapshot=capture(model,h,x,[])
            split=continuation_logits(model,snapshot.kv,q,y)
            with torch.inference_mode():
                dense=model(torch.tensor([h+x+q+y[:-1]],device='mps'),logits_to_keep=len(y)).logits[0].float().cpu()
            record={'id':c['id'],'dtype':str(dtype),'tokens':len(y),
                    'relative_logit_error':relative_error(split,dense),
                    **distribution_metrics(split,dense,y)}
            result['cases'].append(record)
            print(json.dumps(record),flush=True)
        del model,snapshot
        gc.collect()
        torch.mps.empty_cache()
    Path('results/stage_a_numerical_calibration.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
