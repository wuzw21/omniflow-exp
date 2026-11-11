#!/usr/bin/env python3
"""Mechanism-level statistics for the rebuttal.

These answer B4 / C4 / D4 with per-invocation facts, which are valid on the
on-disk subset because they are conditional on "a function was admitted" and
do not depend on the full 116x3 success matrix.

Scope warning is printed and must be carried into any text using these numbers.
"""
import os, json, collections, statistics

ROOT = "data/androidworld"
SETTINGS = {
    "OmniFlowTargetPixel6Pro_seed111_eval113": "standard",
    "OmniFlowTargetFold_seed111_eval113": "fold",
    "WXGA_Tablet_test_00_seed111_eval113": "tablet",
    "OmniFlowTargetSmall_seed111_eval113": "standard",
}

# an admitted action that reached a wrong/non-executable target
WRONG = {
    "omnitransfer_target_semantics_mismatch",
    "omnitransfer_target_identity_mismatch",
    "omnitransfer_target_page_identity_mismatch",
    "omnitransfer_target_not_executable",
    "omnitransfer_target_candidate_not_executable",
    "omnitransfer_ambiguous_target_identity",
    "omnitransfer_invalid_root_candidate",
    "omnitransfer_swipe_target_not_executable",
    "omnitransfer_swipe_source_container_missing",
    "validation_injected_mapping_rejection",
}
# rejected by the gate before anything was dispatched
GATE = {
    "omnitransfer_low_confidence",
    "omnitransfer_target_graph_incomplete",
    "omnitransfer_omniflow_no_capability_compatible_candidate",
    "validation_model_phase_rejection",
    "validation_native_recovery_rejection",
}


def load():
    out = []
    for dp, _, fns in os.walk(ROOT):
        if "run_log.json" not in fns:
            continue
        if os.sep + "omniflow" + os.sep not in dp + os.sep:
            continue
        if "/attempt_" in dp or ".archive" in dp:
            continue
        parts = dp.split(os.sep)
        st = parts[parts.index("omniflow") + 1]
        if st not in SETTINGS:
            continue
        try:
            d = json.load(open(os.path.join(dp, "run_log.json")))
        except Exception:
            continue
        out.append((dp, parts[parts.index("omniflow") - 1], SETTINGS[st], d))
    return out


def main():
    rs = load()
    print(f"target runs on disk: {len(rs)}  (partial subset; NOT the paper's 348)\n")

    inv = []
    outer = []
    resume_runs = 0
    fp_runs = 0
    for dp, task, dev, d in rs:
        dg = d.get("diagnostics") or {}
        fr = dg.get("function_resolution") or {}
        fe = dg.get("function_execution") or {}
        if dg.get("function_resume"):
            resume_runs += 1
            # a recovery event is the deployed stop/fallback/resume path
        for ev in ((fr.get("recall") or {}).get("events") or []):
            for dec in (ev.get("decisions") or []):
                outer.append({
                    "device": dev,
                    "selected": bool(dec.get("selected")),
                    "page_similarity": dec.get("page_similarity"),
                    "observed": dec.get("observed_page_similarity"),
                    "score": dec.get("score"),
                    "rejection": dec.get("rejection_reason"),
                })
            fc = ev.get("function_cache") or {}
            for dec in (fc.get("decisions") or []):
                if not dec.get("admitted"):
                    continue
                errs = [s.get("error") for s in (fe.get("steps") or []) if s.get("error")]
                if fc.get("error"):
                    errs.append(fc["error"])
                if fe.get("replay_status") == "actions_succeeded" or fc.get("status") == "executed":
                    oc = "executed_ok"
                elif any(e in WRONG for e in errs):
                    oc = "wrong_target"
                elif any(e in GATE for e in errs):
                    oc = "gate_reject"
                else:
                    oc = "other"
                inv.append({
                    "task": task, "device": dev, "outcome": oc,
                    "confidence": dec.get("confidence"),
                    "page_similarity": dec.get("page_similarity"),
                    "run_ok": bool((d.get("validator") or {}).get("success", d.get("success"))),
                    "errors": errs,
                })

    n = len(inv)
    print(f"=== admitted function invocations: {n} ===")
    print(f"{'device':10s} {'n':>4} {'ok':>4} {'wrong':>6} {'gate':>5} {'other':>6} {'wrong%':>7}")
    by = collections.defaultdict(list)
    for i in inv:
        by[i["device"]].append(i)
    for dev in sorted(by):
        g = by[dev]
        c = collections.Counter(x["outcome"] for x in g)
        print(f"{dev:10s} {len(g):4d} {c['executed_ok']:4d} {c['wrong_target']:6d} "
              f"{c['gate_reject']:5d} {c['other']:6d} {100*c['wrong_target']/len(g):6.2f}%")
    c = collections.Counter(x["outcome"] for x in inv)
    print(f"{'ALL':10s} {n:4d} {c['executed_ok']:4d} {c['wrong_target']:6d} "
          f"{c['gate_reject']:5d} {c['other']:6d} {100*c['wrong_target']/n:6.2f}%")

    print("\n=== which errors account for the wrong-target dispatches ===")
    errs = collections.Counter()
    for i in inv:
        if i["outcome"] == "wrong_target":
            for e in i["errors"]:
                if e:
                    errs[e] += 1
    for k, v in errs.most_common():
        print(f"  {k:52s} {v:3d}")

    print("\n=== per-step reuse evidence ===")
    fu = fp = 0
    for _, _, _, d in rs:
        for s in d.get("steps") or []:
            md = s.get("metadata") or {}
            ea = md.get("execution_action")
            if isinstance(ea, dict):
                if md.get("source"):
                    fu += 1
                else:
                    fp += 1
    print(f"  steps carrying an execution_action: {fu+fp}  (source-tagged: {fu})")

    print("\n=== recall-level gate signal ===")
    ps = [o["page_similarity"] for o in outer if isinstance(o["page_similarity"], (int, float))]
    ob = [o["observed"] for o in outer if isinstance(o["observed"], (int, float))]
    print(f"  recall decisions: {len(outer)}, selected: {sum(1 for o in outer if o['selected'])}")
    if ob:
        print(f"  observed_page_similarity: n={len(ob)} min={min(ob):.3f} "
              f"med={statistics.median(ob):.3f} p90={sorted(ob)[int(.9*len(ob))]:.3f} max={max(ob):.3f}")
    if ps:
        print(f"  page_similarity (source-vs-source): n={len(ps)} distinct={len(set(round(x,4) for x in ps))}")

    print(f"\n=== stop/fallback/resume path ===\n  runs with function_resume evidence: {resume_runs}/{len(rs)}")

    # operating point table
    print("\n=== deployed operating point (subset) ===")
    print(f"  false accept (wrong target dispatched): {c['wrong_target']}/{n} = {100*c['wrong_target']/n:.2f}%")
    print(f"  clean replay                          : {c['executed_ok']}/{n} = {100*c['executed_ok']/n:.2f}%")
    print(f"  gate rejected after admission         : {c['gate_reject']}/{n} = {100*c['gate_reject']/n:.2f}%")
    print(f"  other failure                         : {c['other']}/{n} = {100*c['other']/n:.2f}%")


if __name__ == "__main__":
    main()
