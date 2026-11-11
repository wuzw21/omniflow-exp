"""Lossless target vision extraction and identical official pixel processing.

Reproduces assets prepared for the CUDA-only encoder probe. No new encoder,
image compressor, or learned mapping is introduced.
"""
import argparse
import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoProcessor
from run_mobile_prefill_replay import MODEL_REV, upstream_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model_path = Path((args.assets / "model_path.txt").read_text().strip())
    if model_path.name != MODEL_REV:
        raise ValueError("model revision mismatch")
    with safe_open(model_path / "model.safetensors", framework="pt", device="cpu") as f:
        tensors = {k.removeprefix("model.visual."): f.get_tensor(k) for k in f.keys() if k.startswith("model.visual.")}
    if not tensors:
        raise ValueError("vision weights absent")
    save_file(tensors, args.output / "vision.safetensors")
    (args.output / "config.json").write_text(json.dumps(json.loads((model_path / "config.json").read_text())["vision_config"]))
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    cls = upstream_history(args.assets.parent / "qwen-cua-8a295fd")
    history = cls("", 50, 3)
    for i in range(2, 5):
        history.add_screenshot((args.assets / f"screenshot_{i}.webp").read_bytes())
    images = [Image.open(BytesIO(base64.b64decode(s))).convert("RGB") for s in history.screenshots]
    pixel_inputs = processor.image_processor(images=images, return_tensors="pt",
        size={"shortest_edge": 65536, "longest_edge": 262144})
    torch.save(dict(pixel_inputs), args.output / "pixels.pt")
    keys = []
    offset = 0
    for grid in pixel_inputs["image_grid_thw"]:
        count = int(grid.prod())
        keys.append(hashlib.sha256(pixel_inputs["pixel_values"][offset:offset + count].contiguous().numpy().tobytes()).hexdigest())
        offset += count
    expected = json.loads((Path(__file__).parent / "results/mobile_operator_probe_mps.json").read_text())["conditions"]["trim3/sparse"]["encoder_batch_key"]
    if ":".join(keys) != expected:
        raise ValueError("pixel preprocessing differs from actual MPS timing")
    print(json.dumps({"vision_tensor_count": len(tensors), "processed_image_hashes": keys,
                      "model_revision": MODEL_REV, "pixel_identity_check": "passed"}, indent=2))


if __name__ == "__main__":
    main()
