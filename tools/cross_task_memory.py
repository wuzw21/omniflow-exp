"""Cross-task memory transfer: the workload MobiMem actually measures.

The main comparison (`memory_design_comparison.py`) uses **the same task** as the
memory source, which is this repository's own question: one task migrating across
devices.  MobiMem measures something else — ~454 parameterised variants inside 8
task families, where a memory pool spans *different tasks in the same app*.

This script builds that second workload from the same sealed RunLogs, so the two
questions can be reported separately and the difference between them stops being
a confound:

* **same-task pool** — the task's own source trace.  High task overlap, low
  memorisation value: the memory knows the task, not just the app.
* **cross-task pool** — every *other* task in the same app.  This is the honest
  analogue of MobiMem's setup, and it is strictly harder: a memory drawn from a
  different task in the same app shares the app's navigation but not its goal.

Reporting both is the point.  If a design looks strong only under the same-task
pool, its advantage came from task identity rather than from a better memory
primitive.

Reads sealed RunLogs only; dispatches no device action.
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


def _split_pools(root: Path) -> tuple[dict, dict[str, list]]:
    sources = sorted(glob.glob(str(root / "*" / "source" / "*" / "runlog" / "current" / "run_log.json")))
    by_task: dict[str, list[tuple]] = {}
    app_pool: dict[str, list[tuple[str, str, list[tuple]]]] = collections.defaultdict(list)
    for path in sources:
        task = Path(path).parts[-6]
        try:
            steps = _load_steps(path)
        except (OSError, json.JSONDecodeError):
            continue
        app = _task_app(steps)
        signatures = [step_signature(s) for s in steps]
        by_task[task] = (path, app, signatures)
        app_pool[app].append((task, str(path), signatures))
    return by_task, app_pool


def build(root: Path) -> dict:
    by_task, app_pool = _split_pools(root)
    targets = sorted(glob.glob(str(root / "*" / "omniflow" / "*" / "runlog" / "current" / "run_log.json")))
    rows = []
    for target_path in targets:
        task = Path(target_path).parts[-6]
        source = by_task.get(task)
        if source is None:
            continue
        _, app, source_signatures = source
        try:
            payload = json.loads(Path(target_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        steps = [s for s in payload.get("steps") or [] if isinstance(s, dict)]
        if not steps:
            continue
        signatures = [step_signature(s) for s in steps]
        # Same-task pool: the task's own source trace.
        same_task = [source_signatures]
        # Cross-task pool: other tasks in the same app.  Never the same task --
        # that is what makes this the MobiMem-style condition.
        cross_task = [trace for other, _origin, trace in app_pool.get(app, []) if other != task]
        rows.append({
            "task": task,
            "app": app,
            "device": Path(target_path).parts[-4],
            "steps": len(steps),
            "same_acttree": _prefix_reuse(same_task[0], signatures),
            "same_actchain": _chain_reuse(same_task[0], signatures),
            "cross_pool_tasks": len(cross_task),
            "cross_acttree": _pooled_prefix_reuse(cross_task, signatures) if cross_task else 0,
            "cross_actchain": _pooled_chain_reuse(cross_task, signatures) if cross_task else 0,
        })
    return {"rows": rows}


def summarize(rows: list[dict]) -> dict:
    total = sum(r["steps"] for r in rows) or 1

    def rate(key: str) -> float:
        return round(sum(r[key] for r in rows) / total, 4)

    covered = [r for r in rows if r["cross_pool_tasks"] > 0]
    cross_total = sum(r["steps"] for r in covered) or 1
    return {
        "runs": len(rows),
        "steps": total,
        "same_task_acttree": rate("same_acttree"),
        "same_task_actchain": rate("same_actchain"),
        "runs_with_cross_task_history": len(covered),
        "cross_task_steps": cross_total,
        "cross_task_acttree": (
            round(sum(r["cross_acttree"] for r in covered) / cross_total, 4)
            if covered else None
        ),
        "cross_task_actchain": (
            round(sum(r["cross_actchain"] for r in covered) / cross_total, 4)
            if covered else None
        ),
        "median_cross_pool_tasks": (
            round(statistics.median([r["cross_pool_tasks"] for r in covered]), 2)
            if covered else None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/androidworld")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args(argv)
    payload = build(Path(args.root).expanduser().resolve())
    summary = summarize(payload["rows"])
    output = {"summary": summary, "rows": payload["rows"]}
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
