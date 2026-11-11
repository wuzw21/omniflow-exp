"""Post-hoc failure check using real Qwen decoding; not a QA benchmark."""
import gc
import hashlib
import json
from pathlib import Path
import time
import torch
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration

ROOT=Path(__file__).resolve().parent
ASSETS=Path.home()/'.agent-reach/research/mobile-prefill-replay'

def main():
    out=ROOT/'case_qa.json'
    if out.exists(): raise FileExistsError(out)
    timing=json.loads((ROOT/'timing.json').read_text())
    if timing['status']!='completed': raise RuntimeError('Do not contaminate the ongoing encoder benchmark')
    path=(ASSETS/'model_path.txt').read_text().strip()
    torch.set_num_threads(4)
    processor=AutoProcessor.from_pretrained(path,local_files_only=True)
    model=Qwen3VLForConditionalGeneration.from_pretrained(path,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='sdpa').to('mps').eval()
    manifest=json.loads((ROOT/'manifest.json').read_text())
    analysis=json.loads((ROOT/'analysis.json').read_text())
    ident='expenseaddsingle-0-ca36846c'; query_step=11
    row=next(r for r in manifest['records'] if r['id']==ident)
    result={'status':'running','scope':'one manually inspected post-hoc failure; selected after gate evaluation, not an unbiased QA success estimate',
        'model':timing['model'],'revision':timing['revision'],'device':'M4 MPS','id':ident,'query_step':query_step,
        'human_verified_answer':'Health Care',
        'question':"These are screenshots retained from one expense-entry session, in chronological order. What category label is immediately to the right of 'Clothes'? Answer only the exact label visible in the screenshots, or UNKNOWN if it is not visible. Do not guess from general knowledge.",
        'records':[]}
    with torch.inference_mode():
        for kind,param in [('all',0),('last',.01),('memory',.01)]:
            policy=next(r for r in analysis['records'] if r['id']==ident and r['kind']==kind and r['param']==param)
            states=policy['states'][query_step]
            content=[]
            for i in states:
                content.append({'type':'text','text':f'Screenshot step {i}:'})
                content.append({'type':'image','image':row['frames'][i]['path']})
            content.append({'type':'text','text':result['question']})
            inputs=processor.apply_chat_template([{'role':'user','content':content}],tokenize=True,
                add_generation_prompt=True,return_dict=True,return_tensors='pt',
                images_kwargs={'size':{'shortest_edge':65536,'longest_edge':786432}}).to('mps')
            torch.mps.synchronize(); t=time.perf_counter()
            generated=model.generate(**inputs,max_new_tokens=32,do_sample=False)
            torch.mps.synchronize(); seconds=time.perf_counter()-t
            answer=processor.batch_decode(generated[:,inputs['input_ids'].shape[-1]:],skip_special_tokens=True)[0]
            result['records'].append({'kind':kind,'param':param,'retained_steps':states,'answer':answer,
                'seconds':seconds,'input_tokens':inputs['input_ids'].shape[-1]})
            out.write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(result['records'][-1]),flush=True)
            del generated,inputs; gc.collect(); torch.mps.empty_cache()
    result['status']='completed';out.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
