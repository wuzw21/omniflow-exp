"""Bounded native-model diagnostic: module timing and grid placement equality.

No upstream math is changed. Synchronizing each module is diagnostic overhead;
these times must not be selected as production encoder timings.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import transformers
from safetensors.torch import load_file
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel, Qwen3VLVisionConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve prior diagnostic")
    torch.set_num_threads(4)
    result = {"status": "running", "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "torch": torch.__version__, "transformers": transformers.__version__,
              "device": torch.cuda.get_device_name(0), "records": []}
    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    save()
    cfg = Qwen3VLVisionConfig(**json.loads((args.assets / "config.json").read_text()))
    cfg._attn_implementation = "sdpa"
    model = Qwen3VLVisionModel(cfg)
    model.load_state_dict(load_file(args.assets / "vision.safetensors"), strict=True)
    model = model.to("cuda", torch.bfloat16).eval()
    data = torch.load(args.assets / "pixels.pt", weights_only=True, map_location="cpu")
    pixels = data["pixel_values"].to("cuda", torch.bfloat16)
    grids = {d: data["image_grid_thw"].to(d) for d in ("cpu", "cuda")}
    phase = {}

    def wrap(obj, name, label):
        original = getattr(obj, name)
        def measured(*a, **kw):
            torch.cuda.synchronize()
            started = time.perf_counter()
            value = original(*a, **kw)
            torch.cuda.synchronize()
            phase[label] = time.perf_counter() - started
            return value
        setattr(obj, name, measured)

    for name in ("fast_pos_embed_interpolate", "rot_pos_emb"):
        wrap(model, name, name)
    for name, module in model.named_children():
        if name == "blocks":
            for i, block in enumerate(module):
                wrap(block, "forward", f"block_{i}")
        elif name in ("patch_embed", "merger"):
            wrap(module, "forward", name)
        elif name == "deepstack_merger_list":
            for i, merger in enumerate(module):
                wrap(merger, "forward", f"deepstack_{i}")

    reference = None
    try:
        with torch.inference_mode():
            for grid_device in ("cuda", "cpu", "cpu", "cuda"):
                phase = {}
                torch.cuda.synchronize()
                started = time.perf_counter()
                output = model(hidden_states=pixels, grid_thw=grids[grid_device])
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                tensors = [output[0], *output[1]]
                if reference is None:
                    reference = [x.detach().clone() for x in tensors]
                record = {"grid_device": grid_device, "seconds": elapsed, "phases": phase,
                          "finite": all(bool(torch.isfinite(x).all()) for x in tensors),
                          "shapes": [list(x.shape) for x in tensors],
                          "max_abs_errors": [float((a.float() - b.float()).abs().max()) for a, b in zip(reference, tensors)],
                          "bit_equal": [torch.equal(a, b) for a, b in zip(reference, tensors)]}
                result["records"].append(record)
                save()
                print(json.dumps(record), flush=True)
        result["status"] = "completed"
        save()
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        save()
        raise


if __name__ == "__main__":
    main()
