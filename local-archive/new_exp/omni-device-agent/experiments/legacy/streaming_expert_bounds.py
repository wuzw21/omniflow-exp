"""Offline bounds for WAN-sourced expert weights, not an inference benchmark.

For a disjoint window with U unique layer/expert pairs of B bytes each,
at most M useful bytes can already be local at its start. Therefore at
least max(0, U*B-M) weight bytes must arrive during that window. Summing
disjoint windows is valid even with oracle prefetch and arbitrary eviction.
Taking the maximum over partitions is valid; summing partitions is not.

Assumes all local weight storage is charged to M, no extra SSD/CPU backing
store, unchanged routes, and fixed-size 4-bit dense expert matrices. This
does NOT bound remote expert execution, changed routing or other compression.
"""
import argparse
import hashlib
import heapq
import json
import platform
from pathlib import Path

import numpy as np

from fetch_public_trace import CACHE, FILES, PART, REPO, REVISION
from window_oracle import contiguous_segments, slots_in_window


EXPERT_BYTES = 3 * 2560 * 640 // 2


def partition_bound(routes, width, memory_bytes, expert_bytes, experts_per_layer):
    if width <= 0 or memory_bytes < 0 or expert_bytes <= 0:
        raise ValueError("Invalid capacity/window/weight size")
    total = 0
    # Retain the partial tail; windows cover this segment exactly once.
    for start in range(0, len(routes), width):
        unique = len(slots_in_window(routes[start:start + width], experts_per_layer))
        total += max(0, unique * expert_bytes - memory_bytes)
    return total


def belady_misses(sequence, capacity):
    """Exact MIN for standard demand paging of a fixed scalar reference order.

    Cache starts empty; each requested page must occupy a slot. This is an
    auxiliary oracle for this model, not a bound on arbitrary top-k execution.
    """
    if capacity <= 0:
        raise ValueError("Capacity must be positive")
    count = len(sequence)
    next_use = np.full(count, count, dtype=np.int64)
    future = {}
    for i in range(count - 1, -1, -1):
        page = int(sequence[i])
        next_use[i] = future.get(page, count)
        future[page] = i
    resident = {}
    queue = []
    misses = 0
    for i, raw in enumerate(sequence):
        page = int(raw)
        if page not in resident:
            misses += 1
            if len(resident) == capacity:
                while queue:
                    neg_next, victim = heapq.heappop(queue)
                    if resident.get(victim) == -neg_next:
                        del resident[victim]
                        break
                else:
                    raise AssertionError("Missing eviction candidate")
        resident[page] = int(next_use[i])
        heapq.heappush(queue, (-int(next_use[i]), page))
        if len(queue) > 4 * capacity:
            queue = [(-n, p) for p, n in resident.items()]
            heapq.heapify(queue)
    return misses


def load_verified(root):
    manifest = json.loads((root / "manifest.json").read_text())
    arrays = {}
    for name in FILES:
        path = root / name
        meta = manifest["arrays"][name]
        if path.stat().st_size != meta["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
            raise ValueError(f"Integrity mismatch: {name}")
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(arr.shape) != meta["shape"] or str(arr.dtype) != meta["dtype"]:
            raise ValueError(f"Schema mismatch: {name}")
        arrays[name] = arr
    routes = arrays["experts.npy"]
    if routes.shape[1:] != (48, 10) or routes.min() < 0 or routes.max() >= 512:
        raise ValueError("Unexpected routing geometry")
    if np.any(np.diff(np.sort(routes, axis=-1), axis=-1) == 0):
        raise ValueError("Duplicate experts within top-k")
    if any(len(v) != len(routes) for v in arrays.values()):
        raise ValueError("Unequal row counts")
    segments = contiguous_segments(*(arrays[n] for n in FILES[1:]))
    return manifest, routes, segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing results")
    manifest, routes, segments = load_verified(args.trace_dir)
    widths = (16, 32, 64, 128, 256, 512, 1024)
    budgets = (4, 8, 12, 16, 24, 32)
    rows = []
    # Count once per window, then reuse across capacity assumptions.
    for start, end in segments:
        counts = {w: [len(slots_in_window(routes[i:min(i+w, end)], 512))
                      for i in range(start, end, w)] for w in widths}
        bounds = {str(g): {str(w): sum(max(0, u * EXPERT_BYTES - g * 2**30)
                                    for u in counts[w]) for w in widths} for g in budgets}
        rows.append({"start": start, "end": end, "tokens": end-start,
                     "partition_bounds_bytes": bounds,
                     "best_bound_bytes": {g: max(v.values()) for g, v in bounds.items()}})
    summary = {}
    for g in budgets:
        lower = sum(r["best_bound_bytes"][str(g)] for r in rows)
        per_token = lower / len(routes)
        summary[str(g)] = {
            "total_received_weight_bytes_lower_bound": lower,
            "bytes_per_token_lower_bound": per_token,
            "request_macro_mean_bytes_per_token_lower_bound": float(np.mean([
                r["best_bound_bytes"][str(g)] / r["tokens"] for r in rows])),
            "positive_bound_segments": sum(r["best_bound_bytes"][str(g)] > 0 for r in rows),
            "aggregate_bandwidth_only_tokens_per_second_upper_bound": {
                str(mbps): mbps * 1e6 / 8 / per_token if per_token else None
                for mbps in (100, 1000, 10000)}}
    print(json.dumps({"window_bound_summary": summary}), flush=True)
    # Outcome-independent selection: first eight contiguous segments, first 256
    # tokens each, original scalar order. No timing or representative claim.
    oracle = []
    offsets = np.arange(48, dtype=np.int32)[None, :, None] * 512
    for start, end in segments[:8]:
        stop = min(end, start + 256)
        sequence = (routes[start:stop].astype(np.int32) + offsets).reshape(-1)
        oracle.append({"start": start, "end": stop, "tokens": stop-start,
                       "cold_MIN_misses": {str(g): belady_misses(sequence, g * 2**30 // EXPERT_BYTES)
                                           for g in (8, 12, 16)}})
    result = {
        "status": "OFFLINE_TRACE_BOUND_NOT_DEVICE_OR_NETWORK_MEASUREMENT",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source": {"repo": REPO, "revision": REVISION, "part": PART,
                   "license": "CC-BY-4.0", "arrays": {n: manifest["arrays"][n] for n in FILES}},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "rows": len(routes), "segments": len(segments), "bytes_per_expert": EXPERT_BYTES,
        "widths": widths, "expert_storage_budgets_GiB": budgets,
        "summary": summary, "per_segment": rows,
        "fixed_order_cold_paging_oracle": oracle,
        "limits": [
            "Publisher trace, capture not independently reproduced; one model/shard",
            "No task/agent quality, model inference, network, device or energy test",
            "Every local weight store must fit budget: extra SSD/CPU storage invalidates the WAN bound",
            "Fixed 4-bit dense payload excluding scales; no additional weight compression",
            "No shared weights, KV or activations charged: budgets are optimistic",
            "Routes unchanged; remote expert execution is outside this weight-transfer model",
            "Arbitrary initial resident bytes allowed in every window; no cold-start charge",
            "Disjoint partition sums only, then maximum; overlapping partitions never added",
            "Bandwidth caps are whole-trace amortized, not per-token latency or measured throughput",
            "MIN oracle is auxiliary: fixed scalar order, empty cache, whole-page demand paging",
            "A zero lower bound is inconclusive and does not prove a feasible schedule"]}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"oracle_segments": len(oracle), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
