"""MobiMem action reuse rate, measured on this repository's own workload.

MobiMem (arXiv:2512.15784 §6.3) reports action reuse rate for three Action
Memory structures:

* **ActTree** — prefix reuse only.  Within one application, tasks share a common
  prefix; once the sequence branches, later actions are no longer assumed
  reusable.  Reported: 37.5% average reuse.
* **ActChain (LLM templates)** — prefix + suffix reuse driven by templates that
  do not strictly separate invariant from variable steps.  Reported: 59.7%.
* **ActChain (human templates)** — same, but templates place invariant and
  variable actions in adjacent steps to maximise invariant reuse.  Reported:
  77.3%.

Those numbers come from a *different workload*: ~454 parameterised variants
generated within 8 task families (email, train ticketing, food delivery, hotel
booking, shopping, browser, media playback, maps).  This repository's workload is
one task migrating across devices (Standard / Fold / Tablet), where the goal and
parameters are held fixed and only the screen changes.

So the numbers must not be put side by side.  What this script does instead is
apply MobiMem's *definitions* to this repository's workload, so the same quantity
is measured on our traces rather than borrowed from theirs.  A comparison against
MobiMem's figures requires a separate variant-generation workload, which this
script does not construct and must not pretend to.

The reuse decision here is computed from recorded RunLogs, not from a live agent:
an action counts as reused when the recorded source step it would replay is
matched on the target trace by the same structural signature.  Nothing here
dispatches a device action, so these are analysis numbers, not task results.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

# Actions whose target is a point on screen; these are what a replay mechanism
# can actually substitute for a model call.
_POINT_TOOLS = frozenset({"click", "long_press", "input_text"})
# Actions that are replayed verbatim without a target search.
_PASSTHROUGH_TOOLS = frozenset({"open_app", "press_key", "swipe", "back", "home"})


def _norm_action(raw: dict) -> tuple[str, dict]:
    tool = str(raw.get("action_type") or raw.get("tool") or "").strip()
    args = {k: v for k, v in raw.items() if k not in {"action_type", "tool"}}
    return tool, args


def _elements(xml_text: str) -> list[dict]:
    try:
        root = ET.fromstring(xml_text or "")
    except ET.ParseError:
        return []
    out = []
    for node in root.iter():
        bounds = _bounds(node.attrib.get("bounds"))
        if bounds is None:
            continue
        out.append({
            "bounds": bounds,
            "resource_id": str(node.attrib.get("resource-id") or "").rsplit("/", 1)[-1],
            "text": str(node.attrib.get("text") or "").strip(),
            "description": str(node.attrib.get("content-desc") or "").strip(),
            "class": str(node.attrib.get("class") or node.tag).rsplit(".", 1)[-1],
            "package": str(node.attrib.get("package") or "").strip(),
        })
    return out


def _bounds(value) -> tuple[int, int, int, int] | None:
    numbers = [int(x) for x in re.findall(r"-?\d+", str(value or ""))]
    if len(numbers) != 4 or numbers[2] <= numbers[0] or numbers[3] <= numbers[1]:
        return None
    return numbers[0], numbers[1], numbers[2], numbers[3]


def _display(elements: list[dict]) -> tuple[float, float] | None:
    if not elements:
        return None
    w = max(float(e["bounds"][2]) for e in elements)
    h = max(float(e["bounds"][3]) for e in elements)
    return (w, h) if w > 0 and h > 0 else None


def _element_at(elements: list[dict], x: float, y: float) -> dict | None:
    hits = [e for e in elements
            if e["bounds"][0] <= x <= e["bounds"][2] and e["bounds"][1] <= y <= e["bounds"][3]]
    labelled = [e for e in hits if e["resource_id"] or e["text"] or e["description"]]
    pool = labelled or hits
    if not pool:
        return None
    return min(pool, key=lambda e: (e["bounds"][2] - e["bounds"][0]) * (e["bounds"][3] - e["bounds"][1]))


def _signature(element: dict | None) -> tuple[str, str, str]:
    if element is None:
        return ("", "", "")
    return (element["resource_id"], element["text"], element["class"])


def _element_identity(element: dict | None) -> str:
    """The identity a replay predictor matches on.

    resource-id alone is not usable across devices: ``Settings`` exposes
    ``title`` on a phone and a different id on a tablet, and AndroidWorld's OOB
    observation reports the leaf id, so the same control can look different while
    being the same control.  MobiMem's Replay Predictor decides reuse from a
    semantic match, so the identity here is the *text* first and the
    resource-id/description only as a fallback — coordinates, bounds and class are
    excluded because they are exactly what a transfer has to re-derive.
    """

    if element is None:
        return ""
    text = element["text"] or ""
    description = element["description"] or ""
    rid = element["resource_id"] or ""
    if text:
        return f"text:{text}"
    if description:
        return f"desc:{description}"
    if rid:
        return f"rid:{rid}"
    return ""


def _is_replayable(element: dict | None) -> bool:
    """Only a labelled, interactive control is a replay candidate.

    A bare container (``FrameLayout`` / ``RecyclerView`` hit at a random point)
    carries no identity, so counting it as reusable would inflate the rate with
    matches that a real replay mechanism could never verify.
    """

    if element is None:
        return False
    return bool(_element_identity(element))


def _app_key(value: str) -> str:
    """Normalise an app identifier to its last segment.

    The same app is recorded as ``com.android.settings`` on one device and
    ``settings`` on another, depending on whether the launcher resolved a label
    or a package.  Comparing the raw strings would score an identical
    ``open_app`` as a miss.
    """

    text = str(value or "").strip().lower()
    if not text:
        return ""
    return text.rsplit(".", 1)[-1].replace(" ", "_")


def step_signature(step: dict) -> tuple:
    """Structural identity of one recorded action, independent of coordinates.

    This is what a replay mechanism matches on: the tool plus, for point
    actions, the identity of the control under the recorded point.  Coordinates
    and screen geometry are deliberately excluded — they are what transfer has
    to re-derive.  A point action whose control has no identity gets the empty
    signature and is never counted as reusable.
    """

    tool, args = _norm_action(step.get("action") or {})
    if tool in _PASSTHROUGH_TOOLS:
        payload = []
        for key, value in args.items():
            if key == "app_name":
                payload.append((key, _app_key(str(value))))
            elif key in {"key"}:
                payload.append((key, str(value)))
        return (tool, tuple(sorted(payload)), "")
    if tool not in _POINT_TOOLS:
        return (tool, (), "")
    observation = step.get("observation") or {}
    elements = _elements(str(observation.get("xml") or ""))
    display = _display(elements)
    if display is None:
        return (tool, (), "")
    try:
        x = float(args.get("x")) / 1000.0 * display[0]
        y = float(args.get("y")) / 1000.0 * display[1]
    except (TypeError, ValueError):
        return (tool, (), "")
    element = _element_at(elements, x, y)
    if not _is_replayable(element):
        return (tool, (), "")
    return (tool, (_element_identity(element),), "")


def _load_steps(path: Path | str) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [s for s in payload.get("steps") or [] if isinstance(s, dict)]


def _prefix_reuse(source: list[tuple], target: list[tuple]) -> int:
    """ActTree: the longest shared prefix, then zero.

    Once the sequences diverge the environment can no longer be assumed to
    agree, so nothing after the first mismatch is counted as reusable.
    """

    n = 0
    for a, b in zip(source, target):
        if a != b:
            break
        n += 1
    return n


def _chain_reuse(source: list[tuple], target: list[tuple]) -> int:
    """ActChain: invariant steps reusable anywhere; variable ones only in place.

    A step whose signature appears exactly once in the source trace is treated
    as *invariant* (it is a fixed control in the workflow).  A step that repeats
    is treated as *variable* and only counts when it matches at the same
    position.  This is the paper's invariant/variable split, applied without a
    template author.
    """

    counts = collections.Counter(source)
    reused = 0
    for index, signature in enumerate(target):
        if counts.get(signature, 0) == 0:
            continue
        if counts[signature] == 1:
            reused += 1  # invariant: reusable regardless of position
        elif index < len(source) and source[index] == signature:
            reused += 1  # variable: must match in place
    return reused


def _task_app(steps: list[dict]) -> str:
    """The app a trace operates in, from its first ``open_app``.

    MobiMem keeps Action Memory **per application**, so the history pool for a
    task is every completed trace in the same app, not only the same task.
    """

    for step in steps:
        tool, args = _norm_action(step.get("action") or {})
        if tool == "open_app":
            return _app_key(str(args.get("app_name") or ""))
    return ""


def _pooled_chain_reuse(history: list[list[tuple]], target: list[tuple]) -> int:
    """ActChain over an app-level pool of completed traces.

    Each target step is reusable when some historical trace contains the same
    invariant step, or contains it at the same position (variable step).  This is
    the paper's structure applied to the whole app pool rather than one trace.
    """

    position_counts: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    for trace in history:
        for index, signature in enumerate(trace):
            position_counts[index][signature] += 1
    totals = collections.Counter()
    for trace in history:
        for signature in trace:
            totals[signature] += 1
    reused = 0
    for index, signature in enumerate(target):
        if not signature[1] and not signature[0]:
            continue
        if totals.get(signature, 0) == 0:
            continue
        # Invariant when the step occurs in at most one position across history;
        # otherwise it is parameter-dependent and must match in place.
        positions = sum(1 for counter in position_counts.values() if counter.get(signature))
        if positions <= 1:
            reused += 1
        elif position_counts[index].get(signature):
            reused += 1
    return reused


def _pooled_prefix_reuse(history: list[list[tuple]], target: list[tuple]) -> int:
    """ActTree over an app-level pool: the best shared prefix any trace offers."""

    best = 0
    for trace in history:
        n = 0
        for a, b in zip(trace, target):
            if a != b:
                break
            n += 1
        best = max(best, n)
    return best


def collect(root: Path) -> dict:
    source_paths = sorted(glob.glob(str(root / "*" / "source" / "*" / "runlog" / "current" / "run_log.json")))
    targets = sorted(glob.glob(str(root / "*" / "omniflow" / "*" / "runlog" / "current" / "run_log.json")))
    by_task: dict[str, list[Path]] = collections.defaultdict(list)
    for path in targets:
        by_task[Path(path).parts[-6]].append(Path(path))

    # MobiMem keeps Action Memory per application, so the history pool for a
    # task is every completed trace in the same app -- not only the same task.
    # Build it before scoring, and never let a trace see itself.
    app_pool: dict[str, list[tuple[str, list[tuple]]]] = collections.defaultdict(list)
    for source_path in source_paths:
        try:
            steps = _load_steps(source_path)
        except (OSError, json.JSONDecodeError):
            continue
        app_pool[_task_app(steps)].append((str(source_path), [step_signature(s) for s in steps]))

    rows = []
    for source_path in source_paths:
        task = Path(source_path).parts[-6]
        target_paths = by_task.get(task)
        if not target_paths:
            continue
        try:
            source_steps = [_s for _s in (_load_steps(source_path))]
        except (OSError, json.JSONDecodeError):
            continue
        source_signatures = [step_signature(s) for s in source_steps]
        app = _task_app(source_steps)
        history = [trace for origin, trace in app_pool.get(app, [])
                   if origin != str(source_path)]
        for target_path in target_paths:
            try:
                target_steps = _load_steps(target_path)
            except (OSError, json.JSONDecodeError):
                continue
            target_signatures = [step_signature(s) for s in target_steps]
            device = Path(target_path).parts[-4]
            total = len(target_steps)
            if total == 0:
                continue
            rows.append({
                "task": task,
                "app": app,
                "device": device,
                "source_steps": len(source_steps),
                "target_steps": total,
                # History = the same task's own source trace.
                "acttree_reused": _prefix_reuse(source_signatures, target_signatures),
                "actchain_reused": _chain_reuse(source_signatures, target_signatures),
                # History = every completed trace in the same app (MobiMem's
                # per-application Action Memory).
                "pooled_app_traces": len(history),
                "pooled_acttree_reused": _pooled_prefix_reuse(history, target_signatures),
                "pooled_actchain_reused": _pooled_chain_reuse(history, target_signatures),
            })
    return {"rows": rows}


def summarize(rows: list[dict]) -> dict:
    def rate(key: str) -> float | None:
        total = sum(r["target_steps"] for r in rows)
        if total == 0:
            return None
        return round(sum(r[key] for r in rows) / total, 4)

    return {
        "runs": len(rows),
        "target_steps": sum(r["target_steps"] for r in rows),
        "acttree_reuse_rate": rate("acttree_reused"),
        "actchain_reuse_rate": rate("actchain_reused"),
        "pooled_acttree_reuse_rate": rate("pooled_acttree_reused"),
        "pooled_actchain_reuse_rate": rate("pooled_actchain_reused"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/androidworld")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    payload = collect(root)
    rows = payload["rows"]
    summary = summarize(rows)
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))
    by_task = collections.defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    task_rates = []
    for task in sorted(by_task):
        task_summary = summarize(by_task[task])
        task_rates.append({"task": task, **task_summary})
    payload = {"summary": summary, "rows": rows, "per_task": task_rates}
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
