"""Physical-device regression against upstream pixels, fps and cutoff."""
import argparse, json, sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--upstream',type=Path,required=True);p.add_argument('--video',required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
sys.path.insert(0,str(a.upstream/'models/qwen-vl-utils/src'))
import torch
from qwen_vl_utils_fluxmem.vision_process import _read_video_torchvision
from bounded_reader import read_video_bounded
from measure import Meter
m=Meter(torch,device='cpu')
ele={'video':a.video,'video_start':0,'video_end':4,'min_frames':2,'max_frames':96,'fps':1.0,'anchor_end':False}
with m.phase('upstream_torchvision') as original_measure:original,fps0=_read_video_torchvision(ele)
with m.phase('bounded_reader') as bounded_measure:bounded,fps1=read_video_bounded(ele)
result={'equal_pixels':torch.equal(original,bounded),'original_shape':list(original.shape),'bounded_shape':list(bounded.shape),'original_fps':fps0,'bounded_fps':fps1,'trace':read_video_bounded.last_trace,'events':m.events}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');m.close()
assert result['equal_pixels'] and fps0==fps1
assert max(result['trace']['selected_pts_seconds'])<=4
print(json.dumps({k:v for k,v in result.items() if k!='events'},indent=2))
