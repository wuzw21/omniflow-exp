"""Audit published token counters, not replayed inference or causal attribution.

No generated reasoning or task-private text is copied into the output.
Uncached input includes new observations: it is NOT redundant computation.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

REVISION = "ede2a362c99ddba0d03f439c47a0b8fea6abae22"
FIELDS = ("prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens")


def usage(value):
    if not isinstance(value, dict):
        raise ValueError("missing usage object")
    for key in FIELDS:
        if type(value.get(key)) is not int or value[key] < 0:
            raise ValueError(f"missing/invalid counter: {key}")
    if value["cached_tokens"] > value["prompt_tokens"]:
        raise ValueError("cached exceeds prompt")
    return {key: value[key] for key in FIELDS}


def summarize(records):
    totals = Counter()
    first = Counter()
    late = Counter()
    per_step = defaultdict(list)
    drops = 0
    transitions = 0
    fourth = []
    mismatches = []
    lengths = []
    root_duration = []
    first_cached = 0
    for row in records:
        steps = row["steps"]
        if not steps:
            raise ValueError("trajectory has no steps")
        lengths.append(len(steps))
        root_duration.append(row["duration_s"])
        aggregate = usage(row["usage"])
        summed = Counter()
        previous = None
        for i, step in enumerate(steps):
            u = usage(step.get("usage"))
            summed.update(u)
            totals.update(u)
            (first if i == 0 else late).update(u)
            if i == 0 and u["cached_tokens"] > 0:
                first_cached += 1
            if previous is not None:
                transitions += 1
                drops += u["cached_tokens"] < previous["cached_tokens"]
                if i == 3:
                    fourth.append({"cache_change": u["cached_tokens"] - previous["cached_tokens"],
                                   "prompt_change": u["prompt_tokens"] - previous["prompt_tokens"]})
            previous = u
            per_step[i + 1].append(u)
        for key in FIELDS:
            if aggregate[key] != summed[key]:
                mismatches.append({"id": row["id"], "field": key,
                                   "root": aggregate[key], "step_sum": summed[key]})
    def ratio(counter):
        return counter["cached_tokens"] / counter["prompt_tokens"] if counter["prompt_tokens"] else None
    return {
        "trajectories": len(records), "steps": sum(lengths),
        "steps_median": statistics.median(lengths) if lengths else None,
        "totals_from_steps": dict(totals),
        "uncached_prompt_tokens": totals["prompt_tokens"] - totals["cached_tokens"],
        "token_weighted_cache_fraction": ratio(totals),
        "first_step_cache_fraction": ratio(first),
        "later_steps_cache_fraction": ratio(late),
        "trajectories_with_first_step_cache_hits": first_cached,
        "adjacent_transitions": transitions, "cached_token_drop_transitions": drops,
        "fourth_step_eligible": len(fourth),
        "fourth_step_cache_drops": sum(x["cache_change"] < 0 for x in fourth),
        "fourth_step_cache_change_median": statistics.median(x["cache_change"] for x in fourth) if fourth else None,
        "fourth_step_prompt_change_median": statistics.median(x["prompt_change"] for x in fourth) if fourth else None,
        "published_total_duration_s_median": statistics.median(root_duration) if root_duration else None,
        "aggregate_counter_mismatches": mismatches,
        "by_step_first_12": [{"step": i, "n": len(us),
                              "prompt_median": statistics.median(u["prompt_tokens"] for u in us),
                              "cached_median": statistics.median(u["cached_tokens"] for u in us),
                              "uncached_median": statistics.median(u["prompt_tokens"] - u["cached_tokens"] for u in us)}
                             for i, us in sorted(per_step.items()) if i <= 12],
    }


def analyze(root):
    manifest_path = root / "android_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["revision"] != REVISION or manifest["errors"]:
        raise ValueError("unexpected revision or incomplete download")
    index_path = root / "index.json"
    rows = [r for r in json.loads(index_path.read_text())["trajectories"] if r["benchmark"] == "AndroidWorld"]
    expected = {r["id"] for r in rows}
    if len(expected) != len(rows):
        raise ValueError("duplicate index IDs")
    ids = [r["id"] for r in manifest["records"]]
    if len(ids) != len(set(ids)) or set(ids) != expected or len(ids) != manifest["expected_count"]:
        raise ValueError("manifest must cover every indexed AndroidWorld trajectory exactly once")
    records = []
    for entry in manifest["records"]:
        blob = (root / (entry["id"] + ".json")).read_bytes()
        if hashlib.sha256(blob).hexdigest() != entry["sha256"]:
            raise ValueError("record hash mismatch")
        row = json.loads(blob)
        if row["id"] != entry["id"] or row["benchmark"] != "AndroidWorld":
            raise ValueError("record identity mismatch")
        records.append(row)
    groups = defaultdict(list)
    for r in records:
        groups[r["model"]].append(r)
    # Example fixed by dataset index order, never by cache statistics or success.
    example = next(r for r in records if r["id"] == rows[0]["id"])
    return {"dataset_revision": REVISION,
            "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "evidence_type": "analysis_of_publisher_records_not_independent_agent_run",
            "limitations": ["uncached tokens are not necessarily repeated tokens",
                            "cache counters do not identify invalidation causes",
                            "first step is not necessarily a cold cache",
                            "no per-step prefill/decode timings or edge/cloud handoff labels",
                            "no latency, network or quality counterfactual is measured"],
            "all": summarize(records),
            "by_model": {k: summarize(v) for k, v in sorted(groups.items())},
            "example": {"id": example["id"], "selection": "first AndroidWorld row in pinned index",
                        "steps": [{"step": i + 1, **usage(s["usage"]),
                                   "tools": [c["name"] for c in s["calls"]]}
                                  for i, s in enumerate(example["steps"])]}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.data)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"all": {k:v for k,v in result["all"].items() if k != "by_step_first_12"},
                      "models": list(result["by_model"])}, indent=2))
