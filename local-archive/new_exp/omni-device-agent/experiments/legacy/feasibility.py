"""Analytic bounds only: not a model benchmark or measured network trace.

Run: python3 feasibility.py --output results/analytic_bounds.json
The optional trace evaluator consumes observed routes and fixed expert residency;
it does not generate or predict a model's routing decisions.
"""
import argparse
import json
from pathlib import Path

MODEL = {
    "name": "Qwen/Qwen3-30B-A3B",
    "source": "https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json",
    "layers": 48, "hidden": 2048, "kv_heads": 4, "head_dim": 128,
    "experts": 128, "top_k": 8, "expert_intermediate": 768,
}


def kv_bytes(tokens, layers, kv_heads, head_dim, bytes_per_value=2):
    return tokens * layers * 2 * kv_heads * head_dim * bytes_per_value


def serialization_seconds(payload_bytes, mbps):
    if payload_bytes < 0 or mbps <= 0:
        raise ValueError("Nonnegative payload and positive Mbps required")
    return payload_bytes * 8 / (mbps * 1_000_000)


def iid_miss_estimate(expert_hit_rate, layers, top_k, rtt_ms):
    """Independent hits, no successful prefetch, one batched RPC per missed layer.

    This is a counterfactual sensitivity model, NOT an empirical assumption.
    Includes propagation only, omits compute, serialization, queues and retries.
    """
    if not 0 <= expert_hit_rate <= 1 or layers < 1 or top_k < 1 or rtt_ms < 0:
        raise ValueError("Invalid sensitivity parameters")
    expected_blocking_layers = layers * (1 - expert_hit_rate ** top_k)
    return {
        "expert_hit_rate": expert_hit_rate,
        "expected_blocking_layers_per_token": expected_blocking_layers,
        "propagation_seconds_per_token": expected_blocking_layers * rtt_ms / 1000,
        "probability_all_layers_local": expert_hit_rate ** (layers * top_k),
    }


def evaluate_fixed_residency(routes, resident, rtt_ms, expert_count=128):
    """routes[token][layer] = routed expert IDs; resident[layer] = local IDs.

    Model: exact synchronous single-sequence execution, fixed residency, all
    remote experts in a layer batched into one RPC. No overlap across dependent
    layers, no lookahead prefetch or delayed approximate expert execution.
    """
    if not routes or not resident or rtt_ms < 0:
        raise ValueError("Nonempty trace/residency and nonnegative RTT required")
    resident = [set(x) for x in resident]
    if any(e < 0 or e >= expert_count for s in resident for e in s):
        raise ValueError("Resident expert outside model range")
    calls = misses = selections = 0
    token_calls = []
    for token in routes:
        if len(token) != len(resident):
            raise ValueError("Trace layer count differs from residency")
        blocked = 0
        for chosen, local in zip(token, resident):
            if not chosen or len(set(chosen)) != len(chosen):
                raise ValueError("Each layer needs distinct routed experts")
            if any(e < 0 or e >= expert_count for e in chosen):
                raise ValueError("Routed expert outside model range")
            missing = set(chosen) - local
            selections += len(chosen)
            misses += len(missing)
            blocked += bool(missing)
        token_calls.append(blocked)
        calls += blocked
    return {
        "expert_hit_rate": 1 - misses / selections,
        "remote_expert_selections": misses,
        "blocking_rpc_per_token": token_calls,
        "propagation_seconds_total": calls * rtt_ms / 1000,
    }


def report():
    m = MODEL
    # Ideal 4-bit payload; quantization scales, alignment and metadata omitted.
    expert_payload = 3 * m["hidden"] * m["expert_intermediate"] // 2
    cache = kv_bytes(8192, m["layers"], m["kv_heads"], m["head_dim"])
    routes = [[list(range(8)) for _ in range(48)]]
    concentrated = [set(range(8)) for _ in range(48)]
    concentrated[0] = set()
    distributed = [set(range(8)) for _ in range(48)]
    for layer in range(8):
        distributed[layer].remove(0)
    return {
        "status": "ANALYTIC_ONLY_NO_MODEL_OR_NETWORK_MEASUREMENT",
        "model_config_fields": m,
        "assumptions": [
            "One autoregressive sequence; per-layer synchronous remote dependency",
            "Network settings are chosen sensitivity inputs, not measurements",
            "KV = uncompressed BF16 GQA, excludes allocator/metadata overhead",
            "Ideal 4-bit expert contains three dense matrices, excludes scales",
            "IID hit scenarios must be replaced by measured routing traces",
            "Synthetic traces below are counterexamples, not benchmark data",
        ],
        "router_only_propagation_floor_seconds_per_token": {
            str(rtt): m["layers"] * rtt / 1000 for rtt in (10, 30, 80)
        },
        "kv_8192_tokens_bytes": cache,
        "kv_serialization_floor_seconds": {
            str(bw): serialization_seconds(cache, bw) for bw in (20, 100, 1000)
        },
        "ideal_one_expert_q4_bytes": expert_payload,
        "ideal_eight_experts_each_layer_q4_bytes": expert_payload * 8 * 48,
        "iid_at_30ms_rtt": [iid_miss_estimate(h, 48, 8, 30) for h in (.95, .99, .999, 1)],
        "same_hit_rate_synthetic_counterexample": {
            "all_eight_misses_one_layer": evaluate_fixed_residency(routes, concentrated, 30),
            "one_miss_each_of_eight_layers": evaluate_fixed_residency(routes, distributed, 30),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(report(), ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
    else:
        print(result)
