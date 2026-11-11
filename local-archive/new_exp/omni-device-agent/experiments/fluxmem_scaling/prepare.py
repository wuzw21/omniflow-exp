"""Extend the exact immutable sample timeline to the end of the real source video."""
import argparse,hashlib,json,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
from common import ROOT,UP,MODEL,sha
sys.path.insert(0,str(UP/'models/qwen-vl-utils/src'))
import av,torch
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
from qwen2_5_vl_fluxmem import Qwen2_5_VLProcessor
from qwen_vl_utils_fluxmem.vision_process import smart_resize
old=json.loads((ROOT/'runs/fluxmem_fixed_inputs_002/manifest.json').read_text());meta=old['dataset']
assert sha(meta['video'])==meta['video_sha256']
frames=[];pts=[];t0=time.perf_counter()
with av.open(meta['video']) as c:
 s=c.streams.video[0];duration=float(s.duration*s.time_base)
 for f in c.decode(s):
  if f.time>=188:break
  if len(frames)<188 and abs(f.time-len(frames))<1e-6:
   rgb=torch.from_numpy(f.to_ndarray(format='rgb24')).permute(2,0,1)
   h,w=smart_resize(rgb.shape[1],rgb.shape[2],factor=28,min_pixels=16*28**2,max_pixels=128*28**2)
   frames.append(resize(rgb,[h,w],interpolation=InterpolationMode.BICUBIC,antialias=True).float());pts.append(f.time)
assert pts==list(range(188)),pts
video=torch.stack(frames);del frames
proc=Qwen2_5_VLProcessor.from_pretrained(MODEL,local_files_only=True,min_pixels=16*28**2,max_pixels=128*28**2)
image=proc.image_processor(images=None,videos=[video],return_tensors='pt');assert image['video_grid_thw'].tolist()==[[94,16,30]]
blocks=image['pixel_values_videos'].reshape(94,480,-1)
hashes=[hashlib.sha256(b.numpy().tobytes()).hexdigest() for b in blocks]
assert hashes[:41]==old['block_pixel_sha256']
path=a.output/'blocks.pt';torch.save(blocks,path)
manifest={'status':'completed','duration_seconds':duration,'dataset':meta,'pts_seconds':pts,'block_pts':[pts[i:i+2] for i in range(0,188,2)],'block_pixel_sha256':hashes,'first_41_equal_to_previous':True,'previous_manifest_sha256':sha(ROOT/'runs/fluxmem_fixed_inputs_002/manifest.json'),'blocks_path':str(path.resolve()),'blocks_sha256':sha(path),'input_tensor_bytes':blocks.numel()*blocks.element_size(),'preparation_seconds':time.perf_counter()-t0,'source_sha256':sha(__file__),'torch':torch.__version__}
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(a.output/'prepare.py').write_bytes(Path(__file__).read_bytes());print('PREPARED',len(pts),'real frames',manifest['preparation_seconds'],flush=True)
