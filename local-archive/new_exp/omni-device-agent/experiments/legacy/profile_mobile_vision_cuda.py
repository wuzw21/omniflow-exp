"""Read-only module profiling for the unexpectedly slow CUDA vision baseline."""
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel, Qwen3VLVisionConfig

p = argparse.ArgumentParser()
p.add_argument("--assets", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
if args.output.exists():
    raise FileExistsError("preserve profiling output")
torch.set_num_threads(4)
cfg = Qwen3VLVisionConfig(**json.loads((args.assets / "config.json").read_text()))
cfg._attn_implementation = "sdpa"
model = Qwen3VLVisionModel(cfg)
model.load_state_dict(load_file(args.assets / "vision.safetensors"), strict=True)
model = model.to("cuda", torch.bfloat16).eval()
x = torch.load(args.assets / "pixels.pt", weights_only=True, map_location="cpu")
kw = {"hidden_states": x["pixel_values"].to("cuda", torch.bfloat16), "grid_thw": x["image_grid_thw"].to("cuda")}
with torch.inference_mode():
    model(**kw)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        model(**kw)
        torch.cuda.synchronize()
summary = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25)
args.output.write_text(summary + "\n")
print(summary)
