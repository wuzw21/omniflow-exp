"""Native target vision encoder only, on an explicitly selected CUDA device.

Inputs and vision weights are losslessly extracted from the pinned MPS probe.
This does not launch an agent, a network transfer benchmark or a decoder.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch
import transformers
from safetensors.torch import load_file
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel, Qwen3VLVisionConfig


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--physical-gpu", type=int, required=True)
    p.add_argument("--conv-layout", choices=("contiguous", "channels_last_3d"), default="contiguous")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve prior results")
    gpu = subprocess.check_output(["nvidia-smi", "-i", str(args.physical_gpu),
        "--query-gpu=name,uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip()
    fields = [s.strip() for s in gpu.split(",")]
    if int(fields[2]) < 4000 or int(fields[3]) > 10:
        raise RuntimeError("selected GPU is busy or lacks memory; not launching")
    result = {"status": "running", "source_sha256": sha(__file__), "gpu_before": gpu,
        "model_revision": "89644892e4d85e24eaac8bacfd4f463576704203",
        "torch": torch.__version__, "transformers": transformers.__version__,
        "dtype": "bfloat16", "attention": "sdpa", "vision_only": True,
        "conv_layout": args.conv_layout, "cudnn_version": torch.backends.cudnn.version(),
        "weights_sha256": sha(args.assets / "vision.safetensors"),
        "pixels_sha256": sha(args.assets / "pixels.pt"), "records": []}
    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    save()
    try:
        torch.set_num_threads(4)
        cfg = Qwen3VLVisionConfig(**json.loads((args.assets / "config.json").read_text()))
        cfg._attn_implementation = "sdpa"
        model = Qwen3VLVisionModel(cfg)
        model.load_state_dict(load_file(args.assets / "vision.safetensors"), strict=True)
        model = model.to(device="cuda", dtype=torch.bfloat16).eval()
        result["strict_weight_load_passed"] = True
        result["device_name"] = torch.cuda.get_device_name(0)
        data = torch.load(args.assets / "pixels.pt", map_location="cpu", weights_only=True)
        counts = [int(g.prod()) for g in data["image_grid_thw"]]
        keys = []
        offset = 0
        for count in counts:
            keys.append(hashlib.sha256(data["pixel_values"][offset:offset + count].contiguous().numpy().tobytes()).hexdigest())
            offset += count
        result["processed_image_hashes"] = keys
        batches = {3: {"hidden_states": data["pixel_values"].to("cuda", torch.bfloat16),
                       "grid_thw": data["image_grid_thw"].to("cuda")},
                   1: {"hidden_states": data["pixel_values"][-counts[-1]:].to("cuda", torch.bfloat16),
                       "grid_thw": data["image_grid_thw"][-1:].to("cuda")}}
        def features(output):
            if hasattr(output, "pooler_output"):
                return [output.pooler_output, *output.deepstack_features]
            return [output[0], *output[1]]
        conv = model.patch_embed.proj
        def backend(n):
            x = batches[n]["hidden_states"].view(-1, cfg.in_channels,
                    cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size)
            return str(torch._C._select_conv_backend(x, conv.weight, conv.bias,
                list(conv.stride), list(conv.padding), list(conv.dilation), False, [0, 0, 0], conv.groups))
        result["conv_backend_before"] = {str(n): backend(n) for n in (1, 3)}
        if args.conv_layout == "channels_last_3d":
            # Official layout conversion only; preserve native forward and weight values.
            with torch.inference_mode():
                references = {n: [x.detach().clone() for x in features(model(**batches[n]))] for n in (1, 3)}
                weight_before = conv.weight.detach().clone()
                torch.nn.utils.convert_conv3d_weight_memory_format(conv, torch.channels_last_3d)
                result["layout_weight_values_equal"] = torch.equal(weight_before, conv.weight)
                if not result["layout_weight_values_equal"]:
                    raise RuntimeError("layout conversion changed weight values")
                result["layout_output_validation"] = {}
                for n in (1, 3):
                    actual = features(model(**batches[n]))
                    result["layout_output_validation"][str(n)] = {
                        "finite": all(bool(torch.isfinite(x).all()) for x in actual),
                        "bit_equal": [torch.equal(a, b) for a, b in zip(references[n], actual)],
                        "max_abs_errors": [float((a.float()-b.float()).abs().max()) for a, b in zip(references[n], actual)],
                        "relative_l2_errors": [float(torch.linalg.vector_norm(a.float()-b.float()) /
                            torch.linalg.vector_norm(a.float()).clamp_min(1e-12)) for a, b in zip(references[n], actual)]}
                del references, weight_before, actual
        result["conv_backend_selected"] = {str(n): backend(n) for n in (1, 3)}
        save()
        with torch.inference_mode():
            for repeat in range(12):
                for n in ([1, 3] if repeat % 2 == 0 else [3, 1]):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    output = model(**batches[n])
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    tensors = features(output)
                    result["records"].append({"images": n, "repeat": repeat, "warmup": repeat < 2,
                        "seconds": elapsed, "finite": all(bool(torch.isfinite(x).all()) for x in tensors),
                        "output_bytes": sum(x.numel() * x.element_size() for x in tensors),
                        "output_shapes": [list(x.shape) for x in tensors]})
                    del output, tensors
        result["median_seconds_by_images"] = {str(n): statistics.median(r["seconds"] for r in result["records"]
            if r["images"] == n and not r["warmup"]) for n in (1, 3)}
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["status"] = "completed"
        save()
        print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
    except BaseException as e:
        result["status"] = "failed"
        result["error"] = type(e).__name__ + ": " + str(e)
        save()
        raise


if __name__ == "__main__":
    main()
