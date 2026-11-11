"""Pinned upstream/runtime setup shared by real-device scaling runners."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
UP=Path.home()/'.agent-reach/research/fluxmem-repro'
MODEL=Path.home()/'.agent-reach/research/fluxmem-qwen25vl3b/66285546d2b821cf421d4f5eb2576359d3770cd3'
sys.path[:0]=[str(ROOT/'experiments/fluxmem_incremental'),str(ROOT/'experiments/fluxmem_incremental_compact'),str(ROOT/'experiments/fluxmem'),str(UP/'models/qwen2-5-vl/src')]
import torch
from safetensors import safe_open
from qwen2_5_vl_fluxmem import Qwen2_5_VLConfig
from qwen2_5_vl_fluxmem.fluxmem import FluxMem
from qwen2_5_vl_fluxmem.modeling_qwen2_5_vl_fluxmem import Qwen2_5_VLRotaryEmbedding,Qwen2_5_VisionTransformerPretrainedModel
from state import Counter,full_memory,compare,storage_report
from compact import CompactMemory
from measure import Meter
ENDPOINTS=[41,60,94]
def schedule(cadence):return sorted(set(range(cadence//2,95,cadence//2))|set(ENDPOINTS))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def policy(cfg):return FluxMem(cfg.vision_start_token_id,cfg.vision_end_token_id,short_term_frames=8,mid_term_frames=64)
def setup(vision=True):
 assert torch.backends.mps.is_available();torch.set_num_threads(4);torch.manual_seed(1234)
 cfg=Qwen2_5_VLConfig.from_pretrained(MODEL,local_files_only=True)
 index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map'];weights={}
 with safe_open(MODEL/index['model.embed_tokens.weight'],framework='pt',device='cpu') as f:
  sl=f.get_slice('model.embed_tokens.weight');markers=torch.cat([sl[t:t+1] for t in [cfg.vision_start_token_id,cfg.vision_end_token_id]]).to('mps')
 rotary=Qwen2_5_VLRotaryEmbedding(cfg).to('mps');v=None
 if vision:
  v=Qwen2_5_VisionTransformerPretrainedModel._from_config(cfg.vision_config,torch_dtype=torch.bfloat16,attn_implementation='sdpa')
  for filename in sorted(set(index.values())):
   with safe_open(MODEL/filename,framework='pt',device='cpu') as f:
    for key in f.keys():
     if key.startswith('visual.'):weights[key.removeprefix('visual.')]=f.get_tensor(key)
  info=v.load_state_dict(weights,strict=True,assign=True);assert not info.missing_keys and not info.unexpected_keys
  v=v.to('mps').eval()
 return cfg,markers,rotary,v
def freeze(output,names):
 hashes={}
 for f in [Path(__file__),Path(__file__).with_name('PROTOCOL.md')]+[Path(__file__).with_name(n) for n in names]+[ROOT/'experiments/fluxmem_incremental/state.py',ROOT/'experiments/fluxmem_incremental_compact/compact.py',ROOT/'experiments/fluxmem/measure.py']:
  (output/f.name).write_bytes(f.read_bytes());hashes[f.name]=sha(f)
 return hashes
