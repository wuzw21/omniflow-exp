"""Oracle fixed-window residency analysis of externally recorded decode routes.

This is an offline trace analysis, not measured inference or a cache simulator.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from fetch_public_trace import CACHE, REPO, REVISION, PART, FILES


def contiguous_segments(request, position, session):
    if len(request) != len(position) or len(request) != len(session):
        raise ValueError("Boundary arrays have unequal lengths")
    if len(request) == 0:
        return []
    breaks = np.flatnonzero((request[1:] != request[:-1]) |
                           (session[1:] != session[:-1]) |
                           (position[1:] != position[:-1] + 1)) + 1
    edges = np.concatenate(([0], breaks, [len(request)]))
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def slots_in_window(routes, experts_per_layer):
    """Expert IDs are layer-local; never merge equal IDs from different layers."""
    if routes.ndim != 3 or len(routes) == 0:
        raise ValueError("Expected nonempty [token,layer,top_k] array")
    if routes.min() < 0 or routes.max() >= experts_per_layer:
        raise ValueError("Expert index outside model range")
    offsets = np.arange(routes.shape[1], dtype=np.int32)[None, :, None] * experts_per_layer
    return np.unique(routes.astype(np.int32) + offsets)


def summary(values):
    a = np.asarray(values)
    if not len(a):
        return {"count": 0}
    return {"count": len(a), "mean": float(a.mean()),
            "p10": float(np.quantile(a, .1)), "p50": float(np.median(a)),
            "p90": float(np.quantile(a, .9)), "min": int(a.min()), "max": int(a.max())}


def analyze(routes, segments, horizons, experts_per_layer, ideal_expert_bytes, session, session_names):
    output = {}
    for width in horizons:
        if width <= 0:
            raise ValueError("Window width must be positive")
        counts, delta_counts, request_medians = [], [], []
        per_session = {}
        for start, end in segments:
            previous = None
            request_counts = []
            for i in range(start, end - width + 1, width):
                slots = slots_in_window(routes[i:i + width], experts_per_layer)
                counts.append(len(slots))
                request_counts.append(len(slots))
                sid = int(session[i])
                name = session_names[sid] if 0 <= sid < len(session_names) else f"unmapped:{sid}"
                per_session.setdefault(name, []).append(len(slots))
                if previous is not None:
                    delta_counts.append(len(np.setdiff1d(slots, previous, assume_unique=True)))
                previous = slots
            if request_counts:
                request_medians.append(float(np.median(request_counts)))
        counts_array = np.asarray(counts)
        item = {"oracle_resident_slots": summary(counts),
                "request_macro_median_slots": summary(request_medians),
                "covered_rows": len(counts) * width,
                "slots_new_to_previous_disjoint_window": summary(delta_counts),
                "per_session_slots": {k: summary(v) for k, v in per_session.items()}}
        if len(counts):
            sizes = counts_array * ideal_expert_bytes
            item["ideal_expert_payload_GiB"] = {"p50": float(np.median(sizes)/2**30),
                                                  "p90": float(np.quantile(sizes,.9)/2**30)}
            item["fraction_windows_fitting_experts_only"] = {
                str(gib): float(np.mean(sizes <= gib * 2**30)) for gib in (4,8,12,16,24,32)}
            item["cold_payload_seconds_at_assumed_100Mbps_p50"] = float(np.median(sizes)*8/100_000_000)
        output[str(width)] = item
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.trace_dir / "manifest.json").read_text())
    arrays = {}
    for name in FILES:
        path = args.trace_dir / name
        meta = manifest["arrays"][name]
        if path.stat().st_size != meta["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
            raise ValueError(f"Source integrity mismatch: {name}")
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(arr.shape) != meta["shape"] or str(arr.dtype) != meta["dtype"]:
            raise ValueError(f"Source schema mismatch: {name}")
        arrays[name] = arr
    routes = arrays["experts.npy"]
    if routes.shape[1:] != (48, 10) or routes.min() < 0 or routes.max() >= 512:
        raise ValueError("Unexpected model routing shape/range")
    if np.any(np.diff(np.sort(routes, axis=-1), axis=-1) == 0):
        raise ValueError("Duplicate top-k expert selections")
    request, position, session = (arrays[n] for n in FILES[1:])
    if any(len(arr) != len(routes) for arr in (request, position, session)):
        raise ValueError("Routing and boundary row counts differ")
    segments = contiguous_segments(request, position, session)
    names = json.loads((args.trace_dir / "sessions.json").read_text())["names_by_session_index"]
    # From publisher's model card; 4-bit matrix payload excludes block scales.
    expert_bytes = 3 * 2560 * 640 // 2
    result = {
        "status": "OFFLINE_ANALYSIS_OF_THIRD_PARTY_REPORTED_DECODE_TRACE",
        "provenance": {"repository": REPO, "revision": REVISION, "part": PART,
                       "license": "CC-BY-4.0", "source_manifest": manifest},
        "limitations": [
            "Source capture not independently reproduced; publisher-reported model/engine",
            "One convenience-selected shard, not a representative agent benchmark",
            "Windows never cross request/session/position gaps; tail windows discarded",
            "Dataset session names are categories, not validated tool/agent phase labels",
            "No prediction training, timing measurement, model quality or energy evaluation",
            "Ideal fixed-window union assumes no reloads within window, ignores prefill",
            "Bytes exclude shared weights, quantization scales, KV and temporary buffers",
            "Previous-window set difference is not an optimal cache transfer lower bound",
            "Network bandwidth is an assumed sensitivity input, not measured",
        ],
        "validation": {"rows": len(routes), "contiguous_segments": len(segments),
                       "unique_request_session_pairs": len(set(zip(request.tolist(),session.tolist()))),
                       "segment_length": summary([b-a for a,b in segments]),
                       "expert_range": [int(routes.min()),int(routes.max())],
                       "unique_topk_per_row_layer": True},
        "ideal_bytes_per_expert": expert_bytes,
        "windows": analyze(routes, segments, (16,64,256),512,expert_bytes,session,names),
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({"validation": result["validation"], "summary": {
        k:{f:v for f,v in row.items() if f!='per_session_slots'} for k,row in result["windows"].items()
    }},indent=2))


if __name__ == "__main__":
    main()
