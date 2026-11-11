"""Run the real Qwen vision tower once for all 188 sampled seconds."""
import argparse, json, subprocess, sys, time
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--inputs',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
from common import setup, sha, freeze
import torch
assert torch.backends.mps.is_available()
meta=json.loads((a.inputs/'manifest.json').read_text()); pixels=torch.load(meta['blocks_path'],weights_only=True)
cfg,markers,rotary,vision=setup(True); grid=torch.tensor([[1,16,30]],device='mps')
features=[]; t0=time.perf_counter()
with torch.inference_mode():
    for i in range(pixels.shape[0]):
        x=vision(pixels[i].to('mps',dtype=torch.bfloat16),grid_thw=grid).detach().cpu()
        features.append(x)
        print('vision',i+1,'/',pixels.shape[0],flush=True)
features=torch.stack(features)
path=a.output/'features.pt'; torch.save(features,path)
r={'status':'completed','features_path':str(path.resolve()),'features_sha256':sha(path),'blocks':int(features.shape[0]),'tokens_per_block':int(features.shape[1]),'hidden_size':int(features.shape[2]),'vision_seconds':time.perf_counter()-t0,'vision_parameter_bytes':sum(x.numel()*x.element_size() for x in vision.parameters()),'inputs_sha256':sha(a.inputs/'manifest.json'),'source_hashes':freeze(a.output,['feature_prepare.py'])}
(a.output/'results.json').write_text(json.dumps(r,indent=2)+'\n')
