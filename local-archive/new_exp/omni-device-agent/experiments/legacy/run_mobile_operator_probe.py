"""Sparse-target operator probe using native vision/text modules, not a new runtime.

Measures an optimistic resident encoder-cache accounting separately from text
prefill. Operator sums are estimates, not measured end-to-end cache service.
The local agent is NOT executed: target invocation schedules are counterfactual.
"""
import argparse
import ast
import copy
import json
from pathlib import Path
import statistics
import time

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
import run_mobile_prefill_replay as baseline


def load_prepare(**bindings):
    """Execute exactly the preparation function used by the completed baseline."""
    path = Path(baseline.__file__)
    prior = json.loads((path.parent / "results/mobile_prefill_replay_mps_checked.json").read_text())
    if baseline.digest(path) != prior["source_sha256"]:
        raise ValueError("baseline source no longer matches completed reference")
    functions = [n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.FunctionDef) and n.name == "prepare"]
    if len(functions) != 1:
        raise ValueError("ambiguous preparation function")
    env = {**vars(baseline), **bindings}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), str(path), "exec"), env)
    return env["prepare"]


def select_missing_images(item, prefix, known_hashes):
    """Images already covered by decoder prefix need no encoder work either."""
    return [i for i, ((start, _), key) in enumerate(zip(item["spans"], item["image_hashes"]))
            if start >= prefix and key not in known_hashes]


def text_suffix(captured, cut, cache):
    out = dict(captured)
    out["input_ids"] = None
    out["inputs_embeds"] = captured["inputs_embeds"][:, cut:]
    out["position_ids"] = captured["position_ids"][:, :, cut:]
    mask = captured["visual_pos_masks"]
    skipped_visual = int(mask[:, :cut].sum())
    out["visual_pos_masks"] = mask[:, cut:]
    out["deepstack_visual_embeds"] = [x[skipped_visual:] for x in captured["deepstack_visual_embeds"]]
    out["past_key_values"] = cache
    out["use_cache"] = True
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a new output file")
    manifest = json.loads((args.assets / "image_manifest.json").read_text())
    model_path = Path((args.assets / "model_path.txt").read_text().strip())
    if model_path.name != baseline.MODEL_REV:
        raise ValueError("model revision mismatch")
    data = json.loads((args.assets.parent / "hcompany-trajectories" / (manifest["trajectory"] + ".json")).read_text())
    result = {"status": "running", "source_sha256": baseline.digest(__file__),
              "baseline_source_sha256": baseline.digest(baseline.__file__),
              "model_revision": baseline.MODEL_REV, "trajectory": manifest["trajectory"],
              "dataset_revision": manifest["dataset_revision"], "runtime": {"device": "mps", "torch": torch.__version__, "dtype": "bfloat16"},
              "schedules": {"cold": [], "sparse": [1], "continuous": [1, 2, 3]},
              "records": [], "vision_records": [], "conditions": {},
              "limits": ["Counterfactual target calls; no actual local/cloud agents or network",
                         "Sum of separately timed operators; not end-to-end serving latency",
                         "Known encoder features assumed resident without lookup/assembly cost",
                         "Precomputed target embeddings used only to isolate text compute, not claimed free on cold requests",
                         "Same 2B target as baseline, not a large-cloud-model result",
                         "One fixed transition, three timing repetitions after warmup"]}
    def save():
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2) + "\n")
        tmp.replace(args.output)
    save()
    try:
        History = baseline.upstream_history(args.upstream)
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, local_files_only=True,
                    dtype=torch.bfloat16, attn_implementation="sdpa").to("mps").eval()
        result["weights_sha256"] = baseline.digest(model_path / "model.safetensors")
        prepare = load_prepare(History=History, processor=processor, model=model, data=data, manifest=manifest)
        old = {"sparse": prepare(1, 3), "continuous": prepare(3, 3)}
        targets = {"keep4": prepare(4, 4), "trim3": prepare(4, 3)}
        def native_inputs(item):
            return {k: v.to("mps") for k, v in item.items() if isinstance(v, torch.Tensor)}
        with torch.inference_mode():
            source_caches = {}
            for schedule, item in old.items():
                output = model(**native_inputs(item), use_cache=True, logits_to_keep=1)
                source_caches[schedule] = output.past_key_values
                del output
            captured = {}
            refs = {}
            for policy, item in targets.items():
                def capture(module, positional, kw):
                    captured[policy] = dict(kw)
                handle = model.model.language_model.register_forward_pre_hook(capture, with_kwargs=True)
                try:
                    output = model(**native_inputs(item), use_cache=False, logits_to_keep=1)
                    refs[policy] = output.logits.detach().float().cpu()
                    del output
                finally:
                    handle.remove()
            jobs = []
            vision_jobs = {}
            for policy, item in targets.items():
                for schedule in ("cold", "sparse", "continuous"):
                    cut = 0 if schedule == "cold" else baseline.common_prefix(old[schedule], item)
                    known = set() if schedule == "cold" else set(old[schedule]["image_hashes"])
                    missing = select_missing_images(item, cut, known)
                    # Key by actual processed image identities, so equivalent batches
                    # are timed once and shared across their accounting cases.
                    key = ":".join(item["image_hashes"][i] for i in missing) or "empty"
                    offsets = [0]
                    for grid in item["image_grid_thw"]:
                        offsets.append(offsets[-1] + int(grid.prod()))
                    if missing and key not in vision_jobs:
                        vision_jobs[key] = {
                            "pixel_values": torch.cat([item["pixel_values"][offsets[i]:offsets[i + 1]] for i in missing]).to("mps"),
                            "image_grid_thw": item["image_grid_thw"][missing].to("mps")}
                    name = policy + "/" + schedule
                    result["conditions"][name] = {"prefix_tokens": cut,
                        "suffix_tokens": item["input_ids"].shape[-1] - cut,
                        "missing_encoder_images": missing, "encoder_batch_key": key,
                        "known_image_count": len(known),
                        "encoder_cache_hits_in_suffix": sum(start >= cut and key_ in known for (start, _), key_ in zip(item["spans"], item["image_hashes"]))}
                    jobs.append((name, policy, schedule, cut))
            save()
            for repeat in range(4):
                for key, kw in vision_jobs.items():
                    torch.mps.synchronize()
                    start = time.perf_counter()
                    features = model.model.get_image_features(**kw, return_dict=True)
                    torch.mps.synchronize()
                    elapsed = time.perf_counter() - start
                    tensors = list(features.pooler_output) + list(features.deepstack_features)
                    result["vision_records"].append({"key": key, "repeat": repeat, "warmup": repeat == 0,
                        "seconds": elapsed, "images": kw["image_grid_thw"].shape[0],
                        "pooled_plus_deepstack_bytes": sum(t.numel() * t.element_size() for t in tensors)})
                    del features, tensors
                for name, policy, schedule, cut in jobs[repeat % 6:] + jobs[:repeat % 6]:
                    cache = None if schedule == "cold" else copy.deepcopy(source_caches[schedule])
                    if cache is not None:
                        cache.crop(cut)
                    kw = text_suffix(captured[policy], cut, cache)
                    torch.mps.synchronize()
                    start = time.perf_counter()
                    text_output = model.model.language_model(**kw)
                    logits = model.lm_head(text_output.last_hidden_state[:, -1:, :])
                    torch.mps.synchronize()
                    elapsed = time.perf_counter() - start
                    current = logits.detach().float().cpu()
                    reference = refs[policy]
                    result["records"].append({"condition": name, "repeat": repeat, "warmup": repeat == 0,
                        "text_and_head_seconds": elapsed,
                        "last_logit_max_abs_error": (current - reference).abs().max().item(),
                        "last_logit_finite": bool(torch.isfinite(current).all()),
                        "next_token_agrees": current.argmax(-1).item() == reference.argmax(-1).item()})
                    del text_output, logits, kw, cache
                save()
                print("completed repetition", repeat, flush=True)
            for name, condition in result["conditions"].items():
                text_time = statistics.median(r["text_and_head_seconds"] for r in result["records"] if r["condition"] == name and not r["warmup"])
                vr = [r for r in result["vision_records"] if r["key"] == condition["encoder_batch_key"] and not r["warmup"]]
                vision_time = statistics.median(r["seconds"] for r in vr) if vr else 0
                condition.update({"median_text_seconds": text_time, "median_missing_encoder_seconds": vision_time,
                                  "sum_of_operator_medians_seconds": text_time + vision_time,
                                  "missing_encoder_output_bytes": vr[0]["pooled_plus_deepstack_bytes"] if vr else 0})
            result["status"] = "completed"
            save()
            print(json.dumps(result["conditions"], indent=2), flush=True)
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = type(exc).__name__ + ": " + str(exc)
        save()
        raise


if __name__ == "__main__":
    main()
