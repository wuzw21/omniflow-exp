"""Run the two previously failing operations on the actual MPS device."""
import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
import torch
assert torch.backends.mps.is_available()
torch.manual_seed(1234)
x=torch.randn(4,16,8,dtype=torch.bfloat16)
mask=torch.ones(1,16,16,dtype=torch.bool)
expected=torch.nn.functional.scaled_dot_product_attention(x.float(),x.float(),x.float(),mask)
y=torch.nn.functional.scaled_dot_product_attention(x.to('mps'),x.to('mps'),x.to('mps'),mask.to('mps'))
torch.mps.synchronize()
error=(y.cpu().float()-expected).abs().max().item()
assert torch.allclose(y.cpu().float(),expected,atol=.03,rtol=.03)
labels=torch.arange(4,device='mps');indices=torch.tensor([0,1],device='mps');src=torch.tensor([1,0],device='mps')
labels.scatter_reduce_(0,indices,src,reduce='amin',include_self=True);torch.mps.synchronize()
assert labels.cpu().tolist()==[0,0,2,3]
r={'status':'passed','torch':torch.__version__,'device':'mps','chip':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),'sdpa_3d_max_absolute_error_vs_cpu_fp32':error,'scatter_reduce_int64_output':labels.cpu().tolist()}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
