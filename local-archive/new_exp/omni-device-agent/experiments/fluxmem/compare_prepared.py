"""Compare actual processor tensors across runtime versions, not serialized bytes."""
import argparse,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('old',type=Path);p.add_argument('new',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
old=json.loads(a.old.read_text());new=json.loads(a.new.read_text());checks=[]
assert old['status']==new['status']=='completed'
assert len(old['records'])==len(new['records'])==3
for xrow,yrow in zip(old['records'],new['records']):
 x=torch.load(xrow['input_path'],weights_only=True);y=torch.load(yrow['input_path'],weights_only=True)
 assert x.keys()==y.keys()
 fields={k:torch.equal(v,y[k]) if isinstance(v,torch.Tensor) else v==y[k] for k,v in x.items()}
 trace=xrow['sampling_trace']==yrow['sampling_trace']
 checks.append({'query_index':xrow['index'],'fields_equal':fields,'sampling_trace_equal':trace})
 assert all(fields.values()) and trace
r={'status':'passed','runtime_torch':torch.__version__,'old_manifest':str(a.old),'new_manifest':str(a.new),'checks':checks}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print('PASS: all three actual input tensors and sampling traces match')
