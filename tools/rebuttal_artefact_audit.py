#!/usr/bin/env python3
"""Which experimental artefacts on this machine can actually support the
rebuttal, and which cannot. Read-only audit; no runlogs are modified.

Run from the repository root:
    /usr/bin/python3 -I tools/rebuttal_artefact_audit.py
"""
import os, json, collections, datetime

ROOT = "data/androidworld"
PAPER_TARGET_RUNS = 348  # 116 tasks x 3 target configs, per the submitted paper
CANON_TARGET = {
    "OmniFlowTargetPixel6Pro_seed111_eval113": "standard",
    "OmniFlowTargetFold_seed111_eval113": "fold",
    "WXGA_Tablet_test_00_seed111_eval113": "tablet",
    "OmniFlowTargetSmall_seed111_eval113": "standard",
}


def xlsx_serial_to_iso(x):
    return (datetime.datetime(1899, 12, 30) + datetime.timedelta(days=float(x))).isoformat()


def audit_disk():
    """Canonical, non-archived omniflow target runs."""
    per_task = collections.defaultdict(set)
    n = 0
    succ = 0
    callers = collections.Counter()
    for dp, _, fns in os.walk(ROOT):
        if "run_log.json" not in fns:
            continue
        if os.sep + "omniflow" + os.sep not in dp + os.sep:
            continue
        if "/attempt_" in dp or ".archive" in dp:
            continue
        parts = dp.split(os.sep)
        setting = parts[parts.index("omniflow") + 1]
        if setting not in CANON_TARGET:
            continue
        task = parts[parts.index("omniflow") - 1]
        n += 1
        per_task[task].add(setting)
        try:
            d = json.load(open(os.path.join(dp, "run_log.json")))
        except Exception:
            continue
        v = d.get("validator") or {}
        succ += bool(v.get("success", d.get("success")))
        # provenance tells us which machine produced it
        prov = json.dumps(d.get("provenance") or {})
        for host in ("/home/wuzewen", "/Users/wuzewen", "/home/zewen"):
            if host in prov:
                callers[host] += 1
    return {
        "target_runs": n,
        "distinct_tasks": len(per_task),
        "tasks_with_all_3_targets": sum(1 for s in per_task.values() if len(s) == 3),
        "success": succ,
        "provenance_hosts": dict(callers),
    }


def audit_workbook(path):
    """The 116x12 workbook is the only run-receipt registry available."""
    ins = path + ".inspect.ndjson"
    if not os.path.exists(ins):
        return {"error": "inspect.ndjson missing"}
    tabs = {}
    for line in open(ins):
        o = json.loads(line)
        if o.get("kind") == "table":
            tabs[o["sheet"]] = o
    out = {"sheets": list(tabs)}
    t = tabs.get("Run Receipts")
    if t:
        hdr = t["values"][0]
        idx = {h: i for i, h in enumerate(hdr)}
        rows = t["values"][1:]

        def g(r, k):
            i = idx.get(k)
            return r[i] if i is not None and i < len(r) else None

        ts = [g(r, "Timestamp UTC") for r in rows if isinstance(g(r, "Timestamp UTC"), (int, float))]
        counted = [r for r in rows if str(g(r, "Counted in 116×12")).lower() == "true"]
        by = collections.defaultdict(lambda: [0, 0])
        for r in counted:
            key = (str(g(r, "Method")), str(g(r, "Device Label")))
            by[key][1] += 1
            by[key][0] += str(g(r, "Official Validator")).lower() in ("true", "1")
        out["receipts"] = len(rows)
        out["counted"] = len(counted)
        out["newest_utc"] = xlsx_serial_to_iso(max(ts)) if ts else None
        out["oldest_utc"] = xlsx_serial_to_iso(min(ts)) if ts else None
        out["counted_by_method_device"] = {
            f"{m} | {d}": {"n": v[1], "succ": v[0]} for (m, d), v in sorted(by.items())
        }
    return out


def audit_ablations():
    root = os.path.join(ROOT, ".archive", "adaptive-v2-20260913")
    out = {}
    if not os.path.isdir(root):
        return {"error": "no adaptive-v2 archive"}
    for cond in sorted(os.listdir(root)):
        p = os.path.join(root, cond)
        if not os.path.isdir(p):
            continue
        runs = 0
        tasks = set()
        for dp, _, fns in os.walk(p):
            if "run_log.json" in fns:
                runs += 1
                parts = dp.split(os.sep)
                if "omniflow" in parts:
                    tasks.add(parts[parts.index("omniflow") - 1])
        out[cond] = {"runs": runs, "tasks": len(tasks)}
    return out


def main():
    print("=" * 68)
    print("REBUTTAL ARTEFACT AUDIT")
    print("=" * 68)
    a = audit_disk()
    print("\n[1] canonical omniflow TARGET runs on disk")
    for k, v in a.items():
        print(f"    {k:28s} {v}")
    print(f"    paper requires                {PAPER_TARGET_RUNS} target runs (116 x 3)")
    print(f"    coverage                      {100*a['target_runs']/PAPER_TARGET_RUNS:.1f}% of the paper's set")

    b = audit_workbook("outputs/script_replay_matrix/OmniFlow_AndroidWorld_116Tasks_12cell.xlsx")
    print("\n[2] 116x12 workbook (run-receipt registry)")
    if "error" in b:
        print("    ", b["error"])
    else:
        print(f"    sheets                        {b['sheets']}")
        print(f"    receipts                      {b['receipts']}")
        print(f"    counted in 116x12             {b['counted']}")
        print(f"    receipt window                {b['oldest_utc']} .. {b['newest_utc']}")
        print("    counted runs by method|device:")
        for k, v in b["counted_by_method_device"].items():
            sr = 100 * v["succ"] / v["n"] if v["n"] else 0
            print(f"      {k:28s} n={v['n']:3d} succ={v['succ']:3d} SR={sr:5.1f}%")

    c = audit_ablations()
    print("\n[3] adaptive-v2-20260913 ablation archive")
    if "error" in c:
        print("    ", c["error"])
    else:
        for k, v in c.items():
            print(f"    {k:24s} runs={v['runs']:3d} tasks={v['tasks']:2d}")

    print("\n" + "=" * 68)
    print("VERDICT")
    print("=" * 68)
    print("""
What the on-disk runlogs CAN support (this machine, no 4090 needed):
  - mechanism-level characterization: admission decisions, transfer errors,
    fallback incidence, per-step reuse evidence. These are per-invocation
    facts that do not need the full 116x3 matrix.
  - the ablation archive already contains memory_off / no_reentry /
    no_checker / no_reentry_no_checker conditions.

What they CANNOT support:
  - any absolute success rate comparable to the paper's 53.2% (185/348).
    On-disk coverage of the target matrix is far below the paper's set, and
    the surviving canonical directories are biased towards successful
    retries.
  - the granuality ladder's trajectory-level replay arm: only Fixed Replay
    exists (16/100/0 cells), and Script Replay on the small phone is empty.

Therefore: absolute-SR claims must come from the 4090 matrix. Mechanism and
threshold claims can be written now, with the subset stated explicitly.
""")


if __name__ == "__main__":
    main()
