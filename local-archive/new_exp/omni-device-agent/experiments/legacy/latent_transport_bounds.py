"""Analytical wire-cut audit, not a compressor or an end-to-end benchmark.

Reuses the existing byte/serialization arithmetic. A compact intermediate
computed jointly from both caches is not assumed to be available at the sender.
Run: python3 latent_transport_bounds.py --output results/latent_transport_bounds.json
"""
import argparse
import hashlib
import json
from pathlib import Path

from feasibility import kv_bytes, serialization_seconds


def compute_budget_seconds(native_seconds, raw_request_bytes, latent_bytes, mbps,
                           extra_round_trips=0, rtt_ms=0):
    """Maximum *additional* compute cost before losing to raw upload + prefill.

    Common network propagation cancels only for equal round-trip counts. No
    overlap, queues, serialization CPU cost, or quality equivalence is assumed.
    A negative result means the wire cost alone loses under these assumptions.
    """
    if native_seconds < 0 or extra_round_trips < 0 or rtt_ms < 0:
        raise ValueError("Costs must be nonnegative")
    return (native_seconds + serialization_seconds(raw_request_bytes, mbps)
            - serialization_seconds(latent_bytes, mbps)
            - extra_round_trips * rtt_ms / 1000)


def report(native_path):
    raw = Path(native_path).read_bytes()
    native = json.loads(raw)
    if native.get("status") != "completed":
        raise ValueError("A completed native measurement is required")
    # Official Qwen3-0.6B config, read 2026-10-24. This is geometry only;
    # no Qwen3 weights or latent adapter have been run by this script.
    geometry = {"layers": 28, "kv_heads": 8, "head_dim": 128}
    candidates = []
    for slots in (1, 3):
        for dtype, width in (("bfloat16", 2), ("float32", 4)):
            size = kv_bytes(slots, **geometry, bytes_per_value=width)
            candidates.append({
                "slots": slots, "dtype": dtype, "bytes": size,
                "KiB": size / 1024,
                "serialization_ms": {
                    str(bw): 1000 * serialization_seconds(size, bw)
                    for bw in (20, 100, 1000)
                },
                "uint32_token_count_with_same_bytes": size // 4,
            })
    budgets = []
    for case in native["cases"]:
        elapsed = case["timing"]["cached_prefix"]["median_seconds"]
        request = case["analytic_payload"]["uint32_input_token_bytes"]
        budgets.append({
            "case_id": case["id"], "native_seconds": elapsed,
            "native_model": native["model"]["repo"],
            "request_encoding": "uint32 token IDs; not measured UTF-8 or compressed text",
            "raw_request_bytes": request,
            "hypothetical_qwen3_sender_pool_cuts": [
                dict(c, additional_compute_budget_ms_at_100Mbps=
                     1000 * compute_budget_seconds(elapsed, request, c["bytes"], 100))
                for c in candidates
            ],
        })
    return {
        "status": "ANALYTICAL_SENSITIVITY_ONLY_NOT_LATENT_MODEL_MEASUREMENTS",
        "native_measurement_sha256": hashlib.sha256(raw).hexdigest(),
        "geometry_source": "https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json",
        "geometry": geometry,
        "cut": "sender pooled K and V BEFORE layer map and joint receiver memory",
        "candidates": candidates,
        "sensitivity_against_existing_7b_native_measurements": budgets,
        "limits": [
            "Qwen3-0.6B to the measured Qwen2.5-7B receiver is NOT a tested adapter pair.",
            "The FP32 cut matches kvloom's translator working dtype if its weights are FP32.",
            "BF16 wire encoding is hypothetical and requires its own quality test.",
            "Headers, protocol framing, CPU serialization, RTT differences and failures add cost.",
            "Pool/translation/local question processing costs and quality have NOT been measured.",
            "Pooling reduces token positions, not adapter weights; joint latent cannot be assumed sender-local.",
            "A positive budget is a necessary opportunity, NOT proof of speedup or quality.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path,
                        default=Path(__file__).parent / "results/stage_a_7b_native_prefill.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(report(args.native), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(result)
    else:
        print(result)
