"""Real-model diagnostic using unchanged upstream kvloom modules.

No new compressor, agent loop or training. The split function only changes
operator placement and explicitly checks equality against the upstream forward.
This same-model pilot does NOT establish heterogeneous or WAN performance.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import pyarrow.parquet as pq
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvloom.adapters import adapter_for
from kvloom.cache.ops import add_residual, stack_layers, take_common_suffix
from kvloom.eval.datasets.ropes import RopesDataset
from kvloom.eval.metrics import exact_match, f1_score
from kvloom.session.channel import LatentChannel
from kvloom.training.checkpoint import CheckpointMetadata, load_translator
from kvloom.training.data import PromptTemplate, address_for_tokenizer
from kvloom.translator.protocol import Residual
from latent_question_pairs import select_pairs

COMMIT = "146ae229d3a1d5f0c2aa972aeb3096cb409b9b09"
MODEL = "Qwen/Qwen3-0.6B"
MODEL_REV = "c1899de289a04d12100db370d81485cdf75e47ca"
DATA_REV = "d59f1e2ee2b423d7c6ba71edd47fceb4158b07dd"
ADAPTER_REV = "259e1a4e9502113978c9b4af516f49451c1b8e6c"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def source_packet(translator, cache):
    keys, values = stack_layers(take_common_suffix(cache))
    dtype = next(translator.parameters()).dtype
    return translator.sharer_pooler(keys.to(dtype), values.to(dtype))


def receive_packet(translator, packet, cache):
    """Exact upstream forward after its sender pooling cut; no padding, batch=1."""
    keys, values = stack_layers(take_common_suffix(cache))
    native_dtype = keys.dtype
    dtype = next(translator.parameters()).dtype
    keys, values = keys.to(dtype), values.to(dtype)
    aligned = tuple(translator.layer_map(p) for p in packet)
    receiver_pool = translator.receiver_pooler(keys, values)
    memory = translator.memory(*aligned, *receiver_pool)
    delta_keys, delta_values = translator.retrieval(keys, values, memory)
    return Residual.from_stacked(delta_keys=delta_keys.to(native_dtype),
                                delta_values=delta_values.to(native_dtype))


def inject(cache, residual):
    return add_residual(cache, delta_keys=residual.delta_keys,
                        delta_values=residual.delta_values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        help="Optional device-remapped checkpoint, verified by upstream state hash")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--pairs", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior results; choose a new output path")
    actual = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if actual != COMMIT or subprocess.check_output(
            ["git", "-C", str(args.upstream), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Upstream must be the pinned, unchanged checkout")
    torch.manual_seed(0)
    data_path = args.assets / "ropes-validation.parquet"
    examples = [RopesDataset().to_example(r) for r in pq.read_table(data_path).to_pylist()]
    pairs = select_pairs(examples, args.pairs)
    result = {
        "status": "running", "kind": "same_model_question_change_diagnostic",
        "upstream_commit": COMMIT, "model": MODEL, "model_revision": MODEL_REV,
        "adapter_revision": ADAPTER_REV, "dataset_revision": DATA_REV,
        "dataset_sha256": sha(data_path), "source_sha256": sha(__file__),
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "transformers": transformers.__version__, "device": args.device},
        "selection": "first validation-order same-assigned-evidence groups with two different questions and disjoint gold strings; no prediction filtering",
        "pairs": [{"old": asdict(a), "current": asdict(b)} for a, b in pairs],
        "records": [],
        "limits": ["Same-model Qwen/Qwen control, not heterogeneous evidence.",
                   "Eight questions at most by default; no population-quality claim.",
                   "Question change only; no persistent generated/tool history or real WAN.",
                   "Published adapter did not pin base-weight revision; this run pins current weights.",
                   "No adapter training or new compression algorithm.",
                   "Timing is one-shot diagnostic, not a performance benchmark."],
    }
    def save():
        temp = args.output.with_suffix(".tmp")
        temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        temp.replace(args.output)
    save()  # Freeze selection and record failures before inference.
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                attn_implementation="sdpa", local_files_only=True).to(args.device).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        geometry = adapter_for(model.config).geometry(model.config)
        checkpoint = args.checkpoint or args.assets / "ropes-translator.pt"
        meta = CheckpointMetadata.read(checkpoint)
        if (meta.sharer_model, meta.receiver_model) != (MODEL, MODEL):
            raise RuntimeError("Pair identity mismatch")
        translator = load_translator(checkpoint, sharer_geometry=geometry,
                        receiver_geometry=geometry).to(args.device).eval()
        result["adapter_sha256"] = sha(checkpoint)
        result["model_weights_sha256"] = sha(args.model / "model.safetensors")
        result["adapter_metadata"] = asdict(meta)
        channel = LatentChannel(sharer=model, receiver=model, translator=translator)
        template = PromptTemplate()

        def encode(context, question):
            text = address_for_tokenizer(template.render(context=context, question=question,
                choices=None), tokenizer=tokenizer, template=template)
            return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(args.device)

        def answer(cache, held):
            ids = channel.greedy_from_cache(cache, held, 20, tokenizer.eos_token_id)
            full = tokenizer.decode(ids[0], skip_special_tokens=True)
            return {"text": full.strip().split("\n")[0].strip(),
                    "full_text": full, "token_ids": ids[0].tolist()}

        with torch.no_grad():
            for pair_index, (old, current) in enumerate(pairs):
                old_source = channel.prefill_sharer(encode(old.sharer_context, old.question))
                packet_old = source_packet(translator, old_source)
                for turn, example in enumerate((old, current)):
                    started = time.monotonic()
                    source_ids = encode(example.sharer_context, example.question)
                    receiver_ids = encode(example.receiver_context, example.question)
                    source = old_source if turn == 0 else channel.prefill_sharer(source_ids)
                    receiver = channel.prefill_receiver(receiver_ids[:, :-1])
                    before = receiver.content_hash()
                    packet = source_packet(translator, source)
                    # Lossless CPU tensor round-trip emulates the serialization cut,
                    # without claiming a network measurement.
                    wire_copy = tuple(p.cpu().contiguous().clone().to(args.device) for p in packet)
                    split = receive_packet(translator, wire_copy, receiver)
                    whole = translator(source, receiver)
                    equal = all(torch.equal(a, b) for a, b in zip(
                        split.delta_keys + split.delta_values,
                        whole.delta_keys + whole.delta_values, strict=True))
                    if not equal:
                        raise RuntimeError("Split placement changed upstream residuals")
                    stale = receive_packet(translator, packet_old, receiver)
                    held = receiver_ids[:, -1:]
                    conditions = {
                        "receiver_only": answer(receiver, held),
                        "fresh_latent": answer(inject(receiver, split), held),
                        "reused_old_question_packet": answer(inject(receiver, stale), held),
                    }
                    full_ids = encode(example.sharer_context + "\n\n" + example.receiver_context,
                                      example.question)
                    full_cache = channel.prefill_receiver(full_ids[:, :-1])
                    conditions["all_evidence_text"] = answer(full_cache, full_ids[:, -1:])
                    for prediction in conditions.values():
                        prediction["f1"] = f1_score(prediction["text"], example.answers)
                        prediction["exact_match"] = exact_match(prediction["text"], example.answers)
                    if receiver.content_hash() != before:
                        raise RuntimeError("Target input cache was mutated")
                    diff = sum((a.float() - b.float()).square().sum().item()
                               for a, b in zip(packet, packet_old, strict=True))
                    denom = sum(a.float().square().sum().item() for a in packet)
                    row = {"pair_index": pair_index, "turn": turn,
                           "id": example.example_id, "question": example.question,
                           "gold": list(example.answers), "conditions": conditions,
                           "packet_bytes": sum(p.numel() * p.element_size() for p in packet),
                           "packet_dtype": str(packet[0].dtype),
                           "packet_relative_l2_change": (diff / max(denom, 1e-30))**.5,
                           "split_residual_bit_identical": equal,
                           "receiver_unchanged": True,
                           "elapsed_seconds_all_conditions": time.monotonic() - started}
                    result["records"].append(row)
                    save()
                    print(json.dumps({"id": row["id"], "turn": turn,
                                      "answers": {k: v["text"] for k, v in conditions.items()}},
                                     ensure_ascii=False), flush=True)
        result["status"] = "completed"
        save()
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        save()
        raise


if __name__ == "__main__":
    main()
