"""One immutable sample timeline, original resize and processor, real video only."""
import argparse,ast,json,hashlib,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
up=Path.home()/'.agent-reach/research/fluxmem-repro'
model=Path.home()/'.agent-reach/research/fluxmem-qwen25vl3b/66285546d2b821cf421d4f5eb2576359d3770cd3'
sys.path[:0]=[str(up/'models/qwen2-5-vl/src'),str(up/'models/qwen-vl-utils/src')]
import av,torch
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
from qwen2_5_vl_fluxmem import Qwen2_5_VLProcessor
from qwen_vl_utils_fluxmem.vision_process import smart_resize
meta=json.loads((Path.home()/'.agent-reach/research/fluxmem-streamingbench-sample1/manifest.json').read_text())
assert hashlib.sha256(Path(meta['video']).read_bytes()).hexdigest()==meta['video_sha256']
frames=[];pts=[];t0=time.perf_counter()
with av.open(meta['video']) as c:
 for f in c.decode(c.streams.video[0]):
  if f.time>=82:break
  if len(frames)<82 and abs(f.time-len(frames))<1e-6:
   rgb=torch.from_numpy(f.to_ndarray(format='rgb24')).permute(2,0,1)
   h,w=smart_resize(rgb.shape[1],rgb.shape[2],factor=28,min_pixels=16*28**2,max_pixels=128*28**2)
   frames.append(resize(rgb,[h,w],interpolation=InterpolationMode.BICUBIC,antialias=True).float());pts.append(f.time)
assert pts==list(range(82)),pts
video=torch.stack(frames);del frames
proc=Qwen2_5_VLProcessor.from_pretrained(model,local_files_only=True,min_pixels=16*28**2,max_pixels=128*28**2)
image=proc.image_processor(images=None,videos=[video],return_tensors='pt')
grid=image['video_grid_thw'];assert grid.tolist()==[[41,16,30]],grid
blocks=image['pixel_values_videos'].reshape(41,480,-1)
torch.save(blocks,a.output/'blocks.pt')
# Original question formatter; all input tensors must concatenate the immutable blocks.
tree=ast.parse((up/'evaluation/streamingbench/streamingbench.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='StreamingBenchEvaluator')
template=ast.literal_eval(next(n.value for n in cls.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROMPT_TEMPLATE' for t in n.targets)))
records=[]
for i,(row,n) in enumerate(zip(meta['queries'],[4,52,82])):
 prompt=template.format(question=row['question'],options='\n'.join(ast.literal_eval(row['options'])))
 msg=[{'role':'user','content':[{'type':'video'},{'type':'text','text':prompt}]}]
 text=proc.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
 inp=dict(proc(text=[text],videos=[video[:n]],fps=[1.],return_tensors='pt'))
 assert torch.equal(inp['pixel_values_videos'],blocks[:n//2].flatten(0,1))
 path=a.output/f'query_{i}.pt';torch.save(inp,path)
 records.append({'query':row,'frames':n,'blocks':n//2,'input_path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'tokens':inp['input_ids'].shape[1]})
manifest={'status':'completed','dataset':meta,'pts_seconds':pts,'block_pts':[pts[i:i+2] for i in range(0,82,2)],'block_pixel_sha256':[hashlib.sha256(b.numpy().tobytes()).hexdigest() for b in blocks], 'blocks_path':str((a.output/'blocks.pt').resolve()),'blocks_sha256':hashlib.sha256((a.output/'blocks.pt').read_bytes()).hexdigest(),'grid_hw':[16,30],'queries':records,'preparation_seconds':time.perf_counter()-t0,'torch':torch.__version__}
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('PREPARED',len(pts),'fixed frames',manifest['preparation_seconds'],flush=True)
