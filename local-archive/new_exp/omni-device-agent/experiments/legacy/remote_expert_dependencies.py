"""Fixed-placement trace replay: expert hit rate versus dependent RPC layers.

Not a serving system or model/network benchmark. Uses only calibration segments
to construct placements. The coverage DP is a standard multiple-choice knapsack
over per-layer frequency-prefix candidate sets, not an optimum over all sets.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fetch_public_trace import CACHE, PART, REPO, REVISION
from streaming_expert_bounds import EXPERT_BYTES, load_verified


def count_frequencies(routes, experts):
    return np.stack([np.bincount(routes[:, layer, :].reshape(-1), minlength=experts)
                     for layer in range(routes.shape[1])])


def ranked_experts(counts):
    # Stable ties preserve ascending expert ID; no evaluation-data dependence.
    return np.argsort(-counts, axis=1, kind="stable")


def coverage_curves(routes, rankings, step):
    layers, experts = rankings.shape
    if experts % step or step <= 0:
        raise ValueError("Step must divide experts")
    curves = np.zeros((layers, experts // step + 1), dtype=np.int64)
    for layer in range(layers):
        rank = np.empty(experts, dtype=np.int64)
        rank[rankings[layer]] = np.arange(1, experts + 1)
        required = rank[routes[:, layer, :]].max(axis=-1)
        histogram = np.bincount(required, minlength=experts + 1).cumsum()
        curves[layer] = histogram[np.arange(0, experts + 1, step)]
    return curves


def allocate_coverage(curves, budget_units):
    """Maximize calibration fully-local layer events in this candidate family."""
    if budget_units < 0 or curves.ndim != 2 or np.any(np.diff(curves, axis=1) < 0):
        raise ValueError("Invalid budget/curves")
    layers, options = curves.shape
    negative = -(10**15)
    dp = np.full(budget_units + 1, negative, dtype=np.int64)
    dp[0] = 0
    choices = np.full((layers, budget_units + 1), -1, dtype=np.int16)
    for layer in range(layers):
        updated = np.full_like(dp, negative)
        for units in range(min(options-1, budget_units) + 1):
            candidate = dp[:budget_units+1-units] + curves[layer, units]
            valid = dp[:budget_units+1-units] != negative
            better = valid & (candidate > updated[units:])
            updated[units:][better] = candidate[better]
            choices[layer, units:][better] = units
        dp = updated
    # Use the entire available quantized budget; all families get equal capacity.
    if dp[budget_units] == negative:
        raise ValueError("Budget exceeds total candidate capacity")
    allocation = []
    remaining = budget_units
    for layer in range(layers-1, -1, -1):
        chosen = int(choices[layer, remaining])
        if chosen < 0:
            raise AssertionError("Unreachable DP backtrack")
        allocation.append(chosen)
        remaining -= chosen
    assert remaining == 0
    return list(reversed(allocation)), int(dp[budget_units])


def mask_from_counts(rankings, allocation):
    layers, experts = rankings.shape
    if len(allocation) != layers or any(k < 0 or k > experts for k in allocation):
        raise ValueError("Invalid per-layer allocation")
    mask = np.zeros((layers, experts), dtype=bool)
    for layer, count in enumerate(allocation):
        mask[layer, rankings[layer, :count]] = True
    return mask


def placements(calibration, experts, slots, step=16):
    counts = count_frequencies(calibration, experts)
    ranking = ranked_experts(counts)
    layers = counts.shape[0]
    if slots < 0 or slots > counts.size or slots % step:
        raise ValueError("Invalid slot budget")
    global_mask = np.zeros(counts.size, dtype=bool)
    global_mask[np.argsort(-counts.reshape(-1), kind="stable")[:slots]] = True
    equal = [slots // layers + (i < slots % layers) for i in range(layers)]
    packed = [max(0, min(experts, slots-i*experts)) for i in range(layers)]
    curves = coverage_curves(calibration, ranking, step)
    allocation, objective = allocate_coverage(curves, slots // step)
    masks = {
        "global_frequency": global_mask.reshape(counts.shape),
        "uniform_layer_frequency": mask_from_counts(ranking, equal),
        "packed_full_layers": mask_from_counts(ranking, packed),
        "coverage_candidate_DP": mask_from_counts(ranking, [x*step for x in allocation]),
    }
    assert all(int(mask.sum()) == slots for mask in masks.values())
    return masks, {"coverage_calibration_objective": objective, "step_experts": step,
                   "frequency_counts_sha256": hashlib.sha256(counts.tobytes()).hexdigest()}


def evaluate(routes, resident):
    if routes.ndim != 3 or routes.shape[1] != len(resident) or len(routes) == 0:
        raise ValueError("Invalid routes/residency")
    if routes.min() < 0 or routes.max() >= resident.shape[1]:
        raise ValueError("Invalid expert IDs")
    hit = resident[np.arange(routes.shape[1])[None, :, None], routes]
    remote_layers = (~hit.all(axis=-1)).sum(axis=-1)
    return {
        "tokens": len(routes), "local_expert_hit_rate": float(hit.mean()),
        "fully_local_token_fraction": float(np.mean(remote_layers == 0)),
        "remote_layers_per_token": {"mean": float(remote_layers.mean()),
             "p50": float(np.median(remote_layers)), "p95": float(np.quantile(remote_layers,.95)),
             "histogram": np.bincount(remote_layers, minlength=routes.shape[1]+1).tolist()},
        "required_rpc_propagation_floor_ms_per_token": {
             str(rtt): float(remote_layers.mean())*rtt for rtt in (2,10,30,80)},
        "optimistic_bf16_activation_bytes_per_token": float(remote_layers.mean()) * 2 * 2560 * 2,
        "resident_experts_per_layer": resident.sum(axis=1).tolist(),
        "resident_expert_ids": [np.flatnonzero(x).tolist() for x in resident]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing evidence")
    manifest, routes, segments = load_verified(args.trace_dir)
    # Whole segments in recorded order: parity split fixed before seeing metrics.
    train_ranges, test_ranges = segments[::2], segments[1::2]
    calibration = np.concatenate([routes[a:b] for a,b in train_ranges])
    evaluation = np.concatenate([routes[a:b] for a,b in test_ranges])
    result = {
        "status": "OFFLINE_FIXED_PLACEMENT_TRACE_REPLAY_NOT_MODEL_OR_WAN_MEASUREMENT",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source": {"repo": REPO, "revision": REVISION, "part": PART,
                   "license": "CC-BY-4.0", "arrays": manifest["arrays"]},
        "selection": "even recorded contiguous segments calibrate; odd segments evaluate",
        "calibration_ranges": train_ranges, "evaluation_ranges": test_ranges,
        "calibration_tokens": len(calibration), "evaluation_tokens": len(evaluation),
        "bytes_per_expert": EXPERT_BYTES, "results": {},
        "limits": [
            "One public shard/model; published capture not independently reproduced",
            "Segment holdout, not disjoint user/task/domain generalization",
            "No private context, agent loop, network, GPU inference or quality evaluation",
            "All attention/routing stays local; one aggregated remote call per missed layer",
            "One dependent decode sequence; no speculative changes or routing substitutions",
            "Fixed resident weights; no online prefetch/loading or extra SSD/CPU backing",
            "Expert-only ideal 4-bit budget excludes other weights/KV/runtime",
            "RTTs are sensitivity inputs; floor is part of critical path, not added idle time",
            "Coverage DP optimizes frequency-prefix candidates, not arbitrary expert subsets",
            "Not a reproduction or performance comparison of Prism/StateFlow",
            "No demonstration of a new mechanism or usable system speedup"]}
    for gib in (8,12,16,24,32):
        slots = (gib * 2**30 // EXPERT_BYTES // 16) * 16
        masks, meta = placements(calibration, 512, slots)
        row = {"slots": slots, "unused_bytes": gib*2**30-slots*EXPERT_BYTES,
               "construction": meta, "policies": {}}
        for name, mask in masks.items():
            measured = evaluate(evaluation, mask)
            # Per-request equal weighting without huge intermediate route copies.
            measured["request_macro_remote_layers_mean"] = float(np.mean([
                evaluate(routes[a:b], mask)["remote_layers_per_token"]["mean"] for a,b in test_ranges]))
            row["policies"][name] = measured
        result["results"][str(gib)] = row
        print(json.dumps({"budget_GiB": gib, "policies": {
            k: {"hit": v["local_expert_hit_rate"], "remote_layers": v["remote_layers_per_token"]["mean"]}
            for k,v in row["policies"].items()}}), flush=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == '__main__':
    main()
