"""Native CUDA numerical reference; no agent/task quality claims.

All modes use the same BF16-rounded weights and pixels. FP32 reference disables
TF32. It is a numerical diagnostic, not a mathematical exactness oracle.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import transformers
from safetensors.torch import load_file
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel, Qwen3VLVisionConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve numerical evidence")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = Qwen3VLVisionConfig(**json.loads((args.assets / "config.json").read_text()))
    cfg._attn_implementation = "sdpa"
    model = Qwen3VLVisionModel(cfg)
    model.load_state_dict(load_file(args.assets / "vision.safetensors"), strict=True)
    model = model.to("cuda", torch.bfloat16).eval()
    data = torch.load(args.assets / "pixels.pt", weights_only=True, map_location="cpu")
    pixels = data["pixel_values"].to("cuda", torch.bfloat16)
    grid = data["image_grid_thw"].to("cuda")
    count = int(data["image_grid_thw"][-1].prod())
    batches = {1: (pixels[-count:], grid[-1:]), 3: (pixels, grid)}
    outputs, patches = {}, {}
    patch_output = []
    handle = model.patch_embed.register_forward_hook(lambda module, args, out: patch_output.append(out.detach().float().cpu()))
    with torch.inference_mode():
        for mode in ("bf16_contiguous", "bf16_channels_last", "fp32_reference"):
            if mode == "bf16_channels_last":
                torch.nn.utils.convert_conv3d_weight_memory_format(model.patch_embed.proj, torch.channels_last_3d)
            elif mode == "fp32_reference":
                model.float()
            for n, (x, g) in batches.items():
                patch_output.clear()
                out = model(hidden_states=x.float() if mode == "fp32_reference" else x, grid_thw=g)
                outputs[mode, n] = [t.detach().float().cpu() for t in [out[0], *out[1]]]
                patches[mode, n] = patch_output[0]
    handle.remove()
    def error(a, ref):
        return {"max_abs_error": float((a-ref).abs().max()),
                "relative_l2_error": float(torch.linalg.vector_norm(a-ref) / torch.linalg.vector_norm(ref).clamp_min(1e-12)),
                "finite": bool(torch.isfinite(a).all() and torch.isfinite(ref).all())}
    records = []
    for n in (1, 3):
        for mode in ("bf16_contiguous", "bf16_channels_last"):
            records.append({"images": n, "mode": mode,
                "patch_vs_fp32": error(patches[mode, n], patches["fp32_reference", n]),
                "features_vs_fp32": [error(a, b) for a, b in zip(outputs[mode, n], outputs["fp32_reference", n])]})
    result = {"status": "completed", "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "torch": torch.__version__, "transformers": transformers.__version__,
              "device": torch.cuda.get_device_name(0), "cudnn_version": torch.backends.cudnn.version(),
              "tf32_disabled": True, "reference": "FP32 native forward of the same BF16-rounded weights and pixels",
              "records": records,
              "limits": ["No decoder, generated answers, agent or task-quality validation", "FP32 is a numerical reference, not an exact oracle"]}
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
