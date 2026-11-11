"""Three memory designs scored on one workload, from recorded RunLogs only.

This is the analysis behind `docs/experiments/MOBIMEM_REUSE_20261106.md`.

MobiMem publishes action reuse rates for ActTree (37.5%) and ActChain (59.7% /
77.3%).  Those come from ~454 parameterised variants inside 8 task families.  This
repository's workload is one task migrating across devices.  Putting the numbers
side by side would compare two different workloads, so instead this script scores
all three designs under one definition on our own 155 traces:

* **ActTree** — shared prefix with the historical trace, then zero.
* **ActChain** — invariant steps reuse anywhere, variable steps only in place.
* **Function** — OmniFlow's own unit: an executed step attributed to a Function.

Both ActTree and ActChain are computed twice: once with the task's own source
trace as history, and once with every completed trace in the same app, because
MobiMem keeps Action Memory per application.

Everything here is derived from sealed RunLogs.  No device action is dispatched,
so these are analysis numbers, not task results, and they say nothing about task
success — only about how much of a target rollout some memory could substitute
for a model call.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mobimem_reuse_rate import (  # noqa: E402
    _chain_reuse,
    _load_steps,
    _pooled_chain_reuse,
    _pooled_prefix_reuse,
    _prefix_reuse,
    _task_app,
    step_signature,
)


def _function_flags(payload: dict) -> tuple[list[bool], list[bool], float | None]:
    """Per-``steps`` flags for Function attribution and Transfer evidence.

    The two traces carry different data, so neither can be substituted for the
    other:

    * ``steps`` carries the observation XML, which is what an action signature
      needs, and every source and target has it.
    * ``diagnostics.execution_trace`` carries the ``function_id`` and
      ``transfer`` metadata, but its observations are empty, so signatures
      computed from it are blank and unusable for matching.

    Attribution is therefore taken **by position** from the trace, which is safe
    because both are the same executed sequence.  The RunLog may append terminal
    rows (``answer``) past the end of the trace; those simply get no attribution
    rather than borrowing the previous step's.
    """

    steps = [s for s in payload.get("steps") or [] if isinstance(s, dict)]
    diagnostics = payload.get("diagnostics") or {}
    trace = [s for s in diagnostics.get("execution_trace") or [] if isinstance(s, dict)]

    attributed: list[bool] = [False] * len(steps)
    transferred: list[bool] = [False] * len(steps)
    confidences: list[float] = []
    for index in range(min(len(steps), len(trace))):
        metadata = trace[index].get("metadata") or {}
        if not isinstance(metadata, dict):
            continue
        if str(metadata.get("function_id") or "").strip():
            attributed[index] = True
            transfer = metadata.get("transfer")
            if isinstance(transfer, dict):
                transferred[index] = True
                confidence = transfer.get("absolute_contextual_confidence")
                if isinstance(confidence, (int, float)):
                    confidences.append(float(confidence))
    mean = round(statistics.mean(confidences), 4) if confidences else None
    return attributed, transferred, mean


def build(root: Path) -> dict:
    source_paths = sorted(glob.glob(str(root / "*" / "source" / "*" / "runlog" / "current" / "run_log.json")))
    target_paths = sorted(glob.glob(str(root / "*" / "omniflow" / "*" / "runlog" / "current" / "run_log.json")))

    app_pool: dict[str, list[tuple[str, list[tuple]]]] = collections.defaultdict(list)
    for path in source_paths:
        try:
            steps = _load_steps(path)
        except (OSError, json.JSONDecodeError):
            continue
        app_pool[_task_app(steps)].append((str(path), [step_signature(s) for s in steps]))

    by_task: dict[str, list[str]] = collections.defaultdict(list)
    for path in target_paths:
        by_task[Path(path).parts[-6]].append(path)

    rows = []
    for source_path in source_paths:
        task = Path(source_path).parts[-6]
        if task not in by_task:
            continue
        try:
            source_steps = _load_steps(source_path)
        except (OSError, json.JSONDecodeError):
            continue
        source_signatures = [step_signature(s) for s in source_steps]
        app = _task_app(source_steps)
        history = [trace for origin, trace in app_pool.get(app, []) if origin != str(source_path)]

        for target_path in by_task[task]:
            try:
                payload = json.loads(Path(target_path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            steps = [s for s in payload.get("steps") or [] if isinstance(s, dict)]
            if not steps:
                continue
            signatures = [step_signature(s) for s in steps]
            attributed, transferred, confidence = _function_flags(payload)
            total = len(steps)
            if total == 0:
                continue
            rows.append({
                "task": task,
                "app": app,
                "device": Path(target_path).parts[-4],
                "steps": total,
                "acttree": _prefix_reuse(source_signatures, signatures),
                "actchain": _chain_reuse(source_signatures, signatures),
                "pooled_acttree": _pooled_prefix_reuse(history, signatures),
                "pooled_actchain": _pooled_chain_reuse(history, signatures),
                "function_steps": sum(attributed),
                "transfer_steps": sum(transferred),
                "transfer_confidence_mean": confidence,
                "validated": bool((payload.get("validator") or {}).get("success")),
            })
    return {"rows": rows}


def summarize(rows: list[dict]) -> dict:
    total = sum(r["steps"] for r in rows) or 1

    def rate(key: str) -> float:
        return round(sum(r[key] for r in rows) / total, 4)

    per_run = [r["function_steps"] / r["steps"] for r in rows if r["steps"]]
    return {
        "runs": len(rows),
        "steps": total,
        "validated_runs": sum(1 for r in rows if r["validated"]),
        "acttree_reuse_rate": rate("acttree"),
        "actchain_reuse_rate": rate("actchain"),
        "pooled_acttree_reuse_rate": rate("pooled_acttree"),
        "pooled_actchain_reuse_rate": rate("pooled_actchain"),
        "function_step_rate": rate("function_steps"),
        "transfer_step_rate": rate("transfer_steps"),
        "function_step_rate_per_run_median": (
            round(statistics.median(per_run), 4) if per_run else None
        ),
        "runs_with_zero_function_steps": sum(1 for r in rows if r["function_steps"] == 0),
        "runs_fully_function_covered": sum(
            1 for r in rows if r["steps"] and r["function_steps"] == r["steps"]
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/androidworld")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args(argv)
    payload = build(Path(args.root).expanduser().resolve())
    rows = payload["rows"]
    summary = summarize(rows)
    by_task = collections.defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    per_task = [
        {"task": task, **summarize(by_task[task])} for task in sorted(by_task)
    ]
    output = {"summary": summary, "per_task": per_task, "rows": rows}
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
