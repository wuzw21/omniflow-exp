"""Function coverage on the failure side, as a control for the success side.

The main comparison shows Function coverage beats both replay structures on the
155 successful target runs.  That number alone cannot support a claim about task
success: `current/` only keeps successes, so every run it scores passed the
official validator.  The control is the failure side — the runs archived as real
failures.

The test: if Function coverage is a cause of success, it should be **lower** among
failed runs.  If it is not lower, coverage is uncorrelated with outcome and the
paper claim must be restated around "steps that avoid a model call", not "tasks
that succeed".

This script refuses to overstate its own sample.  The archived failures are few
and concentrated on one task, which is reported rather than smoothed over.
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

from memory_design_comparison import _function_flags  # noqa: E402
from mobimem_reuse_rate import _load_steps, step_signature  # noqa: E402


def _coverage(payload: dict) -> dict:
    steps = [s for s in payload.get("steps") or [] if isinstance(s, dict)]
    attributed, transferred, confidence = _function_flags(payload)
    total = len(steps)
    diagnostics = payload.get("diagnostics") or {}
    summary = diagnostics.get("execution_summary") or {}
    return {
        "task": payload.get("task_name"),
        "steps": total,
        "function_steps": sum(attributed),
        "transfer_steps": sum(transferred),
        "transfer_confidence_mean": confidence,
        "coverage": round(sum(attributed) / total, 4) if total else None,
        "failure_reason": str(
            summary.get("failure_reason") or diagnostics.get("done_reason") or ""
        )[:80],
    }


def _archived_paths(root: Path) -> tuple[list[str], str]:
    """Return archived run_log paths and where the list came from.

    Preferred source is ``archive_index.jsonl``, which names the archive kind and
    the outcome for every archived result.  The fallback walks ``.archive/`` — that
    is what older batches only have, and it cannot tell a real failure from a
    success that merely lost promotion, so the caller is told which path was used.
    """

    index_path = root / "archive_index.jsonl"
    if index_path.is_file():
        paths: list[str] = []
        try:
            for line in index_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                relative = str(record.get("path") or "").strip()
                if relative:
                    candidate = root / relative / "run_log.json"
                    if candidate.is_file():
                        paths.append(str(candidate))
        except OSError:
            paths = []
        if paths:
            return paths, "archive_index"
    return sorted(glob.glob(str(root / ".archive" / "**" / "run_log.json"), recursive=True)), "directory_walk"


def build(root: Path) -> dict:
    archived, index_source = _archived_paths(root)
    broken = []
    successes = []
    for path in archived:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        row = _coverage(payload)
        row["path"] = path
        if payload.get("success") is True:
            successes.append(row)
        else:
            broken.append(row)
    return {
        "failures": broken,
        "archived_successes": successes,
        "archive_index_source": index_source,
    }


def summarize(payload: dict) -> dict:
    failures = payload["failures"]
    executed = [r for r in failures if r["steps"] > 0]
    empty = [r for r in failures if r["steps"] == 0]
    return {
        "archive_index_source": payload["archive_index_source"],
        "archived_failures": len(failures),
        "failures_with_zero_steps": len(empty),
        "failures_with_executed_steps": len(executed),
        "archived_successes_excluded": len(payload["archived_successes"]),
        "distinct_failed_tasks": sorted({str(r["task"]) for r in failures}),
        "failure_coverage_on_executed": (
            round(sum(r["function_steps"] for r in executed)
                  / sum(r["steps"] for r in executed), 4)
            if executed else None
        ),
        "failure_coverage_median": (
            round(statistics.median([r["coverage"] for r in executed if r["coverage"] is not None]), 4)
            if executed else None
        ),
        # A control is only usable when the failure side is big enough and is not
        # one task.  Say so out loud instead of reporting a ratio from 8 rows.
        "control_is_conclusive": (
            len(executed) >= 30 and len({str(r["task"]) for r in executed}) > 1
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/androidworld")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args(argv)
    payload = build(Path(args.root).expanduser().resolve())
    summary = summarize(payload)
    output = {"summary": summary, **payload}
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
