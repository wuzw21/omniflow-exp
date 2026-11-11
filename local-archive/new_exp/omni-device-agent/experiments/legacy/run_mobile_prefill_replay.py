"""Fixed public screenshot replay; no agent loop, action execution, or compression.

Reuses pinned Qwen-CUA message construction by AST extraction. The experiment
changes only image_max (3 vs 4), not the upstream pruning algorithm. This is a
Qwen3-VL diagnostic, NOT a reproduction of Holo4's published serving setup.
"""
import argparse
import ast
import base64
import copy
from dataclasses import dataclass, field
from datetime import date
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
import types
from typing import Any

from PIL import Image
import torch
import transformers
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_REV = "89644892e4d85e24eaac8bacfd4f463576704203"
UPSTREAM_SHA = {"runner.py": "c55f6ea209c6c3cf96fd5f0dfadd888e95f01acdae48dae5e449616567dc960f",
                "protocol.py": "8d21a56420a8521302eb9b6ae9a3143a1a886961bae1698ffd52e7a429565f5a"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def upstream_history(root):
    module = types.ModuleType("_pinned_qcua_history")
    sys.modules[module.__name__] = module
    # Freeze only the upstream date dependency, keeping every selected AST body.
    class FrozenDate:
        @staticmethod
        def today():
            return date(2026, 10, 24)
    module.__dict__.update({"dataclass": dataclass, "field": field, "Any": Any,
                           "json": json, "date": FrozenDate, "Image": Image,
                           "BytesIO": BytesIO, "math": math, "base64": base64})
    for name, names in [("protocol.py", {"COLLAPSED_SCREENSHOT_TEXT", "build_tool_definition", "build_system_prompt"}),
                        ("runner.py", {"AgentHistory", "_process_image"})]:
        p = root / name
        if digest(p) != UPSTREAM_SHA[name]:
            raise ValueError("upstream source hash changed")
        parsed = ast.parse(p.read_text())
        nodes = [n for n in parsed.body if getattr(n, "name", None) in names or
                 (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))]
        if len(nodes) != len(names):
            raise ValueError("upstream extraction incomplete")
        nodes.insert(0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))
        unit = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
        exec(compile(unit, str(p), "exec"), module.__dict__)
    return module.AgentHistory


def image_spans(ids, image_token):
    spans = []
    for i, token in enumerate(ids):
        if token == image_token:
            if i == 0 or ids[i - 1] != image_token:
                spans.append([i, i + 1])
            else:
                spans[-1][1] = i + 1
    return spans


def common_prefix(a, b):
    """Match tokens, 3-D positions and actual processed image payload identities."""
    left, right = a["input_ids"][0].tolist(), b["input_ids"][0].tolist()
    n = 0
    for i, (x, y) in enumerate(zip(left, right)):
        if x != y or not torch.equal(a["position_ids"][:, :, i], b["position_ids"][:, :, i]):
            break
        n += 1
    for (start, end), ident in zip(a["spans"], a["image_hashes"]):
        if start >= n:
            continue
        other = next((j for j, span in enumerate(b["spans"]) if span == [start, end]), None)
        if other is None or ident != b["image_hashes"][other]:
            n = min(n, start)
    # Never split a visual block; prefix caching is conservative at this boundary.
    for start, end in a["spans"] + b["spans"]:
        if start < n < end:
            n = start
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing evidence; choose a new result path")
    manifest = json.loads((args.assets / "image_manifest.json").read_text())
    model_path = Path((args.assets / "model_path.txt").read_text().strip())
    if model_path.name != MODEL_REV:
        raise ValueError("unexpected model revision")
    data_path = args.assets.parent / "hcompany-trajectories" / (manifest["trajectory"] + ".json")
    data = json.loads(data_path.read_text())
    result = {"status": "running", "source_sha256": digest(__file__),
              "model": "Qwen/Qwen3-VL-2B-Instruct", "model_revision": MODEL_REV,
              "dataset_revision": manifest["dataset_revision"], "trajectory": data["id"],
              "selection": manifest["selection"], "record_sha256": digest(data_path),
              "upstream_commit": "8a295fd7aaf7c4164a4c56dfb7b2d52ba6040bac", "upstream_sha256": UPSTREAM_SHA,
              "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                          "transformers": transformers.__version__, "device": "mps", "dtype": "bfloat16"},
              "processor_max_pixels": 262144, "records": [],
              "limits": ["Single fixed trajectory transition, no population timing claim",
                         "Qwen-CUA prompt wrapper and Qwen3-VL model, not original Holo4 serving",
                         "Teacher-forced published actions; no agent execution or task success measurement",
                         "Retaining four images changes context; output agreement across policies is not required",
                         "Forward timings exclude cache cloning, preprocessing, network, decode and tool execution",
                         "No new compression algorithm or second agent loop"]}
    def save():
        temp = args.output.with_suffix(".tmp")
        temp.write_text(json.dumps(result, indent=2) + "\n")
        temp.replace(args.output)
    save()
    try:
        History = upstream_history(args.upstream)
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_path,
            local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").to("mps").eval()
        result["weights_sha256"] = digest(model_path / "model.safetensors")
        result["processor_class"] = type(processor.image_processor).__name__
        def prepare(turn, limit):
            h = History(prompt=data["instruction"], history_n=50, image_max=limit)
            for entry in manifest["images"][:turn]:
                if digest(entry["path"]) != entry["sha256"]:
                    raise ValueError("screenshot hash mismatch")
                h.add_screenshot(Path(entry["path"]).read_bytes())
            # Fixed observed action serialization; never replay private thought text.
            h.responses = [json.dumps({"content": s["note"], "tool_calls": s["calls"]}, sort_keys=True)
                           for s in data["steps"][:turn - 1]]
            h.action_summaries = [""] * len(h.responses)
            messages = h.build_messages()
            images = []
            for message in messages:
                for item in message["content"]:
                    if item["type"] == "image_url":
                        image_data = base64.b64decode(item["image_url"]["url"].split(",", 1)[1])
                        images.append(Image.open(BytesIO(image_data)).convert("RGB"))
                        item.clear()
                        item.update({"type": "image"})
            rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            tensors = processor(text=[rendered], images=images, return_tensors="pt",
                                images_kwargs={"size": {"shortest_edge": 65536, "longest_edge": 262144}})
            # Compute global mRoPE positions before cutting a prefix; do not reset suffix positions.
            positions, _ = model.model.get_rope_index(**tensors)
            tensors["position_ids"] = positions
            spans = image_spans(tensors["input_ids"][0].tolist(), model.config.image_token_id)
            image_hashes = []
            offset = 0
            for grid in tensors["image_grid_thw"]:
                count = int(grid.prod())
                payload = tensors["pixel_values"][offset:offset + count].contiguous().numpy().tobytes()
                image_hashes.append(hashlib.sha256(payload).hexdigest())
                offset += count
            if len(spans) != len(image_hashes):
                raise ValueError("image/token span mismatch")
            return {**dict(tensors), "spans": spans, "image_hashes": image_hashes}
        third = prepare(3, 3)
        cases = {"keep4": prepare(4, 4), "trim3": prepare(4, 3)}
        def inputs(item, cut=0):
            out = {"input_ids": item["input_ids"][:, cut:], "attention_mask": item["attention_mask"],
                   "position_ids": item["position_ids"][:, :, cut:]}
            if "mm_token_type_ids" in item:
                out["mm_token_type_ids"] = item["mm_token_type_ids"][:, cut:]
            kept = [i for i, span in enumerate(item["spans"]) if span[0] >= cut]
            if kept:
                start = sum(int(grid.prod()) for grid in item["image_grid_thw"][:kept[0]])
                out["pixel_values"] = item["pixel_values"][start:]
                out["image_grid_thw"] = item["image_grid_thw"][kept[0]:]
            return {k: v.to("mps") for k, v in out.items()}
        with torch.inference_mode():
            source = model(**inputs(third), use_cache=True, logits_to_keep=1)
            source_cache = source.past_key_values
            del source
            result["third_step_tokens"] = third["input_ids"].shape[-1]
            result["conditions"] = {}
            references = {}
            for name, item in cases.items():
                cut = common_prefix(third, item)
                result["conditions"][name] = {"total_tokens": item["input_ids"].shape[-1],
                    "reusable_prefix_tokens": cut, "suffix_tokens": item["input_ids"].shape[-1] - cut,
                    "images": len(item["spans"]), "image_grid_thw": item["image_grid_thw"].tolist(),
                    "images_in_suffix": sum(start >= cut for start, end in item["spans"])}
                references[name] = model(**inputs(item), use_cache=False, logits_to_keep=1).logits.detach().float().cpu()
                ref = references[name]
                probs = torch.softmax(ref, -1)
                result["conditions"][name]["reference_diagnostics"] = {
                    "finite": bool(torch.isfinite(ref).all()), "logit_min": ref.min().item(),
                    "logit_max": ref.max().item(), "logit_rms": ref.square().mean().sqrt().item(),
                    "next_token_id": ref.argmax(-1).item(),
                    "next_token_text": processor.tokenizer.decode([ref.argmax(-1).item()]),
                    "entropy_nats": -(probs * torch.log_softmax(ref, -1)).sum().item()}
            # Change only visual content at fixed image grids/token positions. This
            # checks image conditioning, not whether the resulting answer is right.
            zero_images = inputs(cases["trim3"])
            zero_images["pixel_values"] = torch.zeros_like(zero_images["pixel_values"])
            control = model(**zero_images, use_cache=False, logits_to_keep=1).logits.detach().float().cpu()
            result["conditioning_diagnostics"] = {
                "zero_images_logit_max_abs_change": (control - references["trim3"]).abs().max().item(),
                "policy_change_logit_max_abs_change": (references["keep4"] - references["trim3"]).abs().max().item(),
                "zero_images_finite": bool(torch.isfinite(control).all())}
            del zero_images, control
            save()
            # First repetition is warmup; rotate order to reduce fixed-order bias.
            order = [(name, mode) for name in cases for mode in ("full", "prefix")]
            for repeat in range(args.repeats + 1):
                for name, mode in order[repeat % 4:] + order[:repeat % 4]:
                    item = cases[name]
                    cut = result["conditions"][name]["reusable_prefix_tokens"] if mode == "prefix" else 0
                    kw = inputs(item, cut)
                    if mode == "prefix":
                        cache = copy.deepcopy(source_cache)
                        cache.crop(cut)
                        kw["past_key_values"] = cache
                    torch.mps.synchronize()
                    started = time.perf_counter()
                    output = model(**kw, use_cache=True, logits_to_keep=1)
                    torch.mps.synchronize()
                    elapsed = time.perf_counter() - started
                    logits = output.logits.detach().float().cpu()
                    ref = references[name]
                    difference = logits - ref
                    kl = torch.nn.functional.kl_div(torch.log_softmax(logits, -1), torch.softmax(ref, -1), reduction="sum")
                    row = {"condition": name, "mode": mode, "repeat": repeat, "warmup": repeat == 0,
                           "forward_seconds": elapsed, "logit_max_abs_error": difference.abs().max().item(),
                           "logit_rms_error": difference.square().mean().sqrt().item(),
                           "next_token_agrees_with_full": logits.argmax(-1).item() == ref.argmax(-1).item(),
                           "next_token_kl_reference_to_run": kl.item()}
                    result["records"].append(row)
                    del output, kw
                    if mode == "prefix":
                        del cache
                    save()
                    print(json.dumps(row), flush=True)
            result["timing_medians_seconds"] = {name + "/" + mode: statistics.median(
                r["forward_seconds"] for r in result["records"] if r["condition"] == name and r["mode"] == mode and not r["warmup"])
                for name, mode in order}
            result["status"] = "completed"
            save()
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = type(exc).__name__ + ": " + str(exc)
        save()
        raise


if __name__ == "__main__":
    main()
