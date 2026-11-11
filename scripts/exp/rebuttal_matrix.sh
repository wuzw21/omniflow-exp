#!/usr/bin/env bash
# Rebuttal experiment launcher (to be run on the 4090 box).
#
# Every experiment here answers a specific reviewer question and maps onto
# switches that already exist in run_tasks.py. Nothing new is implemented.
#
#   L1  no memory                     -> answers "is the gain from memory at all"
#   L2  trajectory-level fixed replay -> the real "script replay" arm
#   L3  L2 + fallback (native)        -> isolates fallback from granularity
#   L4  function-level (OmniFlow)     -> the submitted configuration
#   plus memory_off / no_reentry / no_checker at scale, and the subset sweep.
#
# Usage:
#   bash scripts/exp/rebuttal_matrix.sh list
#   bash scripts/exp/rebuttal_matrix.sh <experiment> [--dry-run]
#   bash scripts/exp/rebuttal_matrix.sh all
#
# The only thing the caller must provide is a task list and the on-box paths;
# resolution below is deliberately explicit so nothing runs silently against
# the wrong Store.

set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"

RUN="bash scripts/exp/run_androidworld.sh"
DEVICES=("standard45562" "fold45564" "tablet45554")
SOURCE_SEED=111
EVAL_SEED=113

# --- task list -------------------------------------------------------------
# Default: the 55 tasks that already have canonical target assets on disk.
# Override with TASKS="A B C" or TASK_FILE=/path/to/list.txt.
tasks_from_disk() {
  /usr/bin/env python3 - <<'PY'
import os, json
root = "data/androidworld"
canon = {
    "OmniFlowTargetPixel6Pro_seed111_eval113",
    "OmniFlowTargetFold_seed111_eval113",
    "WXGA_Tablet_test_00_seed111_eval113",
    "OmniFlowTargetSmall_seed111_eval113",
}
seen = []
for t in sorted(os.listdir(root)):
    p = os.path.join(root, t, "omniflow")
    if not os.path.isdir(p):
        continue
    ok = any(os.path.isdir(os.path.join(p, s)) for s in canon)
    if ok:
        seen.append(t)
print("\n".join(seen))
PY
}

if [[ -n "${TASK_FILE:-}" ]]; then
  mapfile -t TASKS < "$TASK_FILE"
elif [[ -n "${TASKS:-}" ]]; then
  read -r -a TASKS <<< "$TASKS"
else
  mapfile -t TASKS < <(tasks_from_disk)
fi

if [[ ${#TASKS[@]} -eq 0 ]]; then
  echo "no tasks resolved; set TASKS or TASK_FILE" >&2
  exit 2
fi

source_runlog() { echo "data/androidworld/$1/source/OmniFlowSourceSmall_seed${SOURCE_SEED}/runlog/current/run_log.json"; }
store()        { echo "data/androidworld/$1/omniflow/OmniFlowSourceSmall_seed${SOURCE_SEED}/memory/current/store.json"; }

need() { # need <path> <what>
  [[ -e "$1" ]] || { echo "MISSING $2: $1" >&2; return 1; }
}

run_one() { # run_one <label> <task> <device> <extra args...>
  local label="$1" task="$2" device="$3"; shift 3
  local out="data/androidworld/.archive/rebuttal-$(date +%Y%m%d)/$label"
  local cmd=("$RUN" run --task "$task" --device "$device" --evaluation-seed "$EVAL_SEED" --output "$out" "$@")
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '  %s\n' "${cmd[*]}"
  else
    echo ">>> [$label] $task @ $device"
    "${cmd[@]}"
  fi
}

# --- experiment definitions ------------------------------------------------
# L2: raw source-coordinate replay, no Function memory, no transfer.
#     This IS "trajectory-level replay with fixed binding and no re-grounding".
exp_l2() {
  for task in "${TASKS[@]}"; do
    need "$(source_runlog "$task")" "source runlog" || continue
    for dev in "${DEVICES[@]}"; do
      run_one L2_fixed_replay "$task" "$dev" \
        --method fixed_replay --raw-replay-run-log "$(source_runlog "$task")"
    done
  done
}

# L3: NOT available as a switch. The Function Store has no "whole-trajectory
#     unit" knob, so "trajectory-level unit + OmniTransfer + fallback" would
#     require implementing a new replay unit. Do not fake it. Until that exists,
#     the ladder reports three rungs (L1, L2, L4) and states the gap explicitly:
#     L2 -> L4 conflates (i) granularity, (ii) learned re-grounding, (iii)
#     fallback. Reviewer B asked for exactly this confound to be removed; a
#     faked L3 would be worse than an acknowledged missing rung.
exp_l3() {
  cat >&2 <<'MSG'
L3 is not implemented: there is no switch that turns a recorded trajectory
into a single replay unit while keeping OmniTransfer and fallback. Building it
is a code change (a whole-trajectory Function plus transfer admission), not a
flag. Options:
  (a) implement `--replay-unit trajectory` (est. a day; then L3 exists),
  (b) report the 3-rung ladder and state the confound in the rebuttal.
Choose one; this launcher will not fabricate the cell.
MSG
  return 3
}

# L4: the submitted configuration.
exp_l4() {
  for task in "${TASKS[@]}"; do
    need "$(store "$task")" "store" || continue
    for dev in "${DEVICES[@]}"; do
      run_one L4_omniflow "$task" "$dev" \
        --method omniflow --memory "$(store "$task")" \
        --function-memory on --function-reentry on --checker on
    done
  done
}

# L1: no memory at all, same planner/host/model. Pairs with L4.
exp_l1() {
  for task in "${TASKS[@]}"; do
    need "$(store "$task")" "store" || continue
    for dev in "${DEVICES[@]}"; do
      run_one L1_memory_off "$task" "$dev" \
        --method omniflow --memory "$(store "$task")" \
        --function-memory off --function-reentry on --checker on
    done
  done
}

# Component ablations at scale (currently only 2-3 tasks in the archive).
exp_no_reentry() {
  for task in "${TASKS[@]}"; do
    need "$(store "$task")" "store" || continue
    for dev in "${DEVICES[@]}"; do
      run_one no_reentry "$task" "$dev" \
        --method omniflow --memory "$(store "$task")" \
        --function-memory on --function-reentry off --checker on
    done
  done
}

exp_no_checker() {
  for task in "${TASKS[@]}"; do
    need "$(store "$task")" "store" || continue
    for dev in "${DEVICES[@]}"; do
      run_one no_checker "$task" "$dev" \
        --method omniflow --memory "$(store "$task")" \
        --function-memory on --function-reentry on --checker off
    done
  done
}

# Accept-threshold sweep. Both gates are read from the environment at import
# time (OMNIFLOW_FUNCTION_CACHE_THRESHOLD, default 0.95 deployed;
# OMNIFLOW_FUNCTION_PAGE_SIMILARITY_THRESHOLD, default 0.80 deployed). Each run
# records the effective value in its RunLog, so a swept run cannot be mistaken
# for the deployed one.
exp_threshold_sweep() {
  for t in 0.80 0.90 0.95 0.99; do
    for task in "${TASKS[@]}"; do
      need "$(store "$task")" "store" || continue
      for dev in "${DEVICES[@]}"; do
        OMNIFLOW_FUNCTION_CACHE_THRESHOLD="$t" \
        OMNIFLOW_FUNCTION_PAGE_SIMILARITY_THRESHOLD="${PAGE_THR:-0.80}" \
          run_one "thr_cache_${t}" "$task" "$dev" \
          --method omniflow --memory "$(store "$task")" \
          --function-memory on --function-reentry on --checker on
      done
    done
  done
}

# Page-similarity gate sweep (the entry gate that actually varies; the cache
# confidence is ~1.0 at admission in every recorded run).
exp_page_sweep() {
  for t in 0.60 0.70 0.80 0.90; do
    for task in "${TASKS[@]}"; do
      need "$(store "$task")" "store" || continue
      for dev in "${DEVICES[@]}"; do
        OMNIFLOW_FUNCTION_PAGE_SIMILARITY_THRESHOLD="$t" \
          run_one "page_${t}" "$task" "$dev" \
          --method omniflow --memory "$(store "$task")" \
          --function-memory on --function-reentry on --checker on
      done
    done
  done
}

usage() {
  cat <<'EOF'
usage: bash scripts/exp/rebuttal_matrix.sh <experiment> [--dry-run]

experiments:
  l1          memory OFF               (pairs with l4)
  l2          trajectory-level fixed replay  -- the real script-replay arm
  l3          trajectory-level + fallback (approximation, see comments)
  l4          OmniFlow (submitted config)
  ladder      l1 then l2 then l3 then l4
  no_reentry  reentry OFF at scale
  no_checker  checker OFF at scale
  threshold   cache-confidence sweep {0.80,0.90,0.95,0.99}
  page        page-similarity gate sweep {0.60,0.70,0.80,0.90}
  list        print the resolved task list and exit
  all         ladder + no_reentry + no_checker

env:
  TASKS="A B C"    explicit task list
  TASK_FILE=path   one task per line
  DRY_RUN=1        print commands only

Reminder: l1/l4 must use the SAME --memory store; that is what makes them a
single-factor pair. Do not compare l1 against a run that omitted --memory.
EOF
}

cmd="${1:-}"; shift || true
[[ "${1:-}" == "--dry-run" ]] && { DRY_RUN=1; shift || true; }

case "$cmd" in
  list) printf '%s\n' "${TASKS[@]}"; echo "--- ${#TASKS[@]} tasks x ${#DEVICES[@]} devices" ;;
  l1) exp_l1 ;;
  l2) exp_l2 ;;
  l3) exp_l3 ;;
  l4) exp_l4 ;;
  ladder) exp_l1; exp_l2; exp_l3; exp_l4 ;;
  no_reentry) exp_no_reentry ;;
  no_checker) exp_no_checker ;;
  threshold) exp_threshold_sweep ;;
  page) exp_page_sweep ;;
  all) exp_l1; exp_l2; exp_l3; exp_l4; exp_no_reentry; exp_no_checker ;;
  *) usage; exit 1 ;;
esac
