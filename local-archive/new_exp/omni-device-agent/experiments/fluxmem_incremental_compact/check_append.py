"""Two real visual blocks catch history concatenation in append; no model needed."""
import argparse,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--legacy',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(root/'experiments/fluxmem_incremental'),str(Path.home()/'.agent-reach/research/fluxmem-repro/models/qwen2-5-vl/src')]
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from qwen2_5_vl_fluxmem.fluxmem import FluxMem
if a.legacy:from state import IncrementalMemory
else:from compact import CompactMemory as IncrementalMemory
class Cats(TorchDispatchMode):
 def __init__(self):super().__init__();self.cats=0
 def __torch_dispatch__(self,func,types,args=(),kwargs=None):
  if str(func)=='aten.cat.default':self.cats+=1
  return func(*args,**(kwargs or {}))
raw=torch.load(root/'runs/fluxmem_incremental_audit_001/question_41_A.pt',weights_only=True)['raw'][:240].to('mps')
c=Cats();s=IncrementalMemory(FluxMem(151652,151653,short_term_frames=8,mid_term_frames=64))
with torch.inference_mode(),c:
 s.append(raw[:120]);s.append(raw[120:])
torch.mps.synchronize()
print({'device':'mps','legacy':a.legacy,'append_cat_calls':c.cats},flush=True)
assert c.cats==0,'append concatenates historical state'
