"""Actual native MPS vision execution. Not a latency simulator or full agent."""
import gc
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time
import numpy as np
from PIL import Image
import torch
import transformers
from transformers import AutoProcessor
from safetensors.torch import load_file
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionConfig,Qwen3VLVisionModel
from policies import Gate,thumbnail

ROOT=Path(__file__).resolve().parent
ASSETS=Path.home()/'.agent-reach/research/mobile-prefill-replay'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()

def main():
    target=ROOT/'timing.json'
    if target.exists(): raise FileExistsError('Preserve prior timing; use an explicitly new run directory.')
    manifest=json.loads((ROOT/'manifest.json').read_text())
    analysis=json.loads((ROOT/'analysis.json').read_text())
    conditions=[{'kind':'all','param':0,'status':'reference'}]
    for row in analysis['chosen']:
        conditions.append({'kind':row['kind'],'param':row.get('param',2 if row['kind']=='uniform' else .01),
             'status':'dev_qualified' if 'param' in row else 'diagnostic_dev_quality_failed'})
    result={'status':'running','hardware':{
        'machine':subprocess.check_output(['sysctl','-n','hw.model'],text=True).strip(),
        'chip':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
        'ram_bytes':int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True)),
        'macos':platform.mac_ver()[0]},'torch':torch.__version__,'transformers':transformers.__version__,
        'model':'Qwen/Qwen3-VL-2B-Instruct','revision':'89644892e4d85e24eaac8bacfd4f463576704203',
        'device':'mps','dtype':'bfloat16','attention':'sdpa','capacity':4,
        'conditions':conditions,'records':[],
        'scope':'actual screenshot open + cheap gate + official image processing + native vision + MPS sync; no LLM decoder, OCR, camera, or power measurement',
        'source_sha256':{p.name:digest(p) for p in (ROOT/'benchmark.py',ROOT/'policies.py',ROOT/'PROTOCOL.md')},
        'manifest_sha256':digest(ROOT/'manifest.json'),'analysis_sha256':digest(ROOT/'analysis.json')}
    def save():
        temp=target.with_suffix('.tmp'); temp.write_text(json.dumps(result,indent=2)+'\n'); temp.replace(target)
    save()
    try:
        assert torch.backends.mps.is_available()
        torch.set_num_threads(4)
        assets=ASSETS/'vision-only'
        cfg=Qwen3VLVisionConfig(**json.loads((assets/'config.json').read_text())); cfg._attn_implementation='sdpa'
        model=Qwen3VLVisionModel(cfg)
        model.load_state_dict(load_file(assets/'vision.safetensors'),strict=True)
        model=model.to('mps',torch.bfloat16).eval()
        processor=AutoProcessor.from_pretrained((ASSETS/'model_path.txt').read_text().strip(),local_files_only=True)
        result['weights_sha256']=digest(assets/'vision.safetensors')
        rows=[r for r in manifest['records'] if r['split']=='test']
        def prep(image):
            return processor.image_processor(images=[image],return_tensors='pt',
                size={'shortest_edge':65536,'longest_edge':262144})
        def forward(inputs):
            return model(hidden_states=inputs['pixel_values'].to('mps',torch.bfloat16),
                grid_thw=inputs['image_grid_thw'].to('mps'))
        with torch.inference_mode():
            with Image.open(rows[0]['frames'][0]['path']) as image:
                inputs=prep(image.convert('RGB'))
            for _ in range(2): out=forward(inputs); torch.mps.synchronize(); del out
            for repeat in range(3):
                order=conditions[repeat:]+conditions[:repeat]
                for condition in order:
                    for row in rows:
                        gate=Gate(condition['kind'],condition['param']); cache={}; calls=[]
                        record={'repeat':repeat,'id':row['id'],**condition,'frames':len(row['frames']),
                            'gate_seconds':0.,'prepare_seconds':0.,'encode_seconds':0.,'visual_tokens':0}
                        torch.mps.synchronize(); started=time.perf_counter()
                        for frame in row['frames']:
                            image=Image.open(frame['path']).convert('RGB')
                            t=time.perf_counter()
                            feature=thumbnail(image) if condition['kind'] in ('last','memory') else np.zeros(1,dtype=np.float32)
                            accept,state=gate.push(frame['step'],feature)
                            record['gate_seconds']+=time.perf_counter()-t
                            if accept:
                                t=time.perf_counter(); inputs=prep(image)
                                record['prepare_seconds']+=time.perf_counter()-t
                                torch.mps.synchronize(); t=time.perf_counter()
                                output=forward(inputs); torch.mps.synchronize()
                                record['encode_seconds']+=time.perf_counter()-t
                                record['visual_tokens']+=int(inputs['image_grid_thw'].prod())//(cfg.spatial_merge_size**2)
                                cache[frame['step']]=output; del output
                                calls.append(frame['step'])
                                for key in list(cache):
                                    if key not in state: del cache[key]
                        torch.mps.synchronize()
                        record['total_seconds']=time.perf_counter()-started
                        record['calls']=len(calls); record['selected']=calls
                        expected=next(r for r in analysis['records'] if r['id']==row['id'] and r['kind']==condition['kind'] and r['param']==condition['param'])
                        if calls!=expected['selected']: raise AssertionError('live gate differs from evaluated policy')
                        record['resident_mps_bytes']=torch.mps.current_allocated_memory()
                        record['driver_mps_bytes']=torch.mps.driver_allocated_memory()
                        result['records'].append(record); save()
                        print(json.dumps({k:record[k] for k in ('repeat','id','kind','param','calls','frames','total_seconds')}),flush=True)
                        del cache; gc.collect(); torch.mps.empty_cache()
        result['summaries']=[]
        for c in conditions:
            times=[]; enc=[]; gates=[]
            for repeat in range(3):
                rr=[r for r in result['records'] if r['repeat']==repeat and r['kind']==c['kind']]
                times.append(sum(r['total_seconds'] for r in rr)); enc.append(sum(r['encode_seconds'] for r in rr)); gates.append(sum(r['gate_seconds'] for r in rr))
            metrics=next(r for r in analysis['aggregate'] if r['split']=='test' and r['kind']==c['kind'] and r['param']==c['param'])
            result['summaries'].append({**c,'repeat_seconds':times,'median_seconds':statistics.median(times),
                'median_encode_seconds':statistics.median(enc),'median_gate_seconds':statistics.median(gates),
                'evidence_recall':metrics['recall'],'calls':metrics['calls'],'frames':metrics['frames']})
        result['status']='completed'; save()
        print(json.dumps(result['summaries'],indent=2),flush=True)
    except BaseException as e:
        result['status']='failed'; result['error']=repr(e); save(); raise

if __name__=='__main__': main()
