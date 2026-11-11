#!/usr/bin/env bash
# Rebuttal 实验启动器 — 在 4090 canonical checkout 根目录运行。
# 用法: bash run_rebuttal.sh <l0|a|b|c|all> [--go]
#   默认 dry-run（只打印命令）；加 --go 才真正执行。
#   L0 未全绿时，a/b/c 一律拒绝执行。
set -euo pipefail

REPO="${REPO:-$(pwd)}"
DATE="$(date +%Y%m%d)"
ARCHIVE="data/androidworld/.archive/rebuttal-$DATE"
RUNS="/Users/zewenwu/Projects/OmniFlow/new_exp/runs"
LAYER="${1:-l0}"
GO="${2:-}"

[[ -f "$REPO/scripts/exp/rebuttal_matrix.sh" ]] || { echo "不是 canonical checkout: $REPO" >&2; exit 2; }
cd "$REPO"

run() { if [[ "$GO" == "--go" ]]; then echo ">>> $*"; "$@"; else echo "[dry-run] $*"; fi; }

case "$LAYER" in
  l0)
    echo "== L0 门禁 =="
    run bash scripts/exp/preflight.sh
    run /usr/bin/python3 -I tools/rebuttal_artefact_audit.py
    run bash scripts/exp/rebuttal_matrix.sh list
    echo "[L0] 输出目录：mkdir -p $ARCHIVE/l0"
    if [[ "$GO" == "--go" ]]; then
      mkdir -p "$ARCHIVE/l0"
      bash scripts/exp/preflight.sh > "$ARCHIVE/l0/preflight.txt" 2>&1 || { echo "L0 FAILED: 见 $ARCHIVE/l0/preflight.txt" >&2; exit 1; }
      /usr/bin/python3 -I tools/rebuttal_artefact_audit.py > "$ARCHIVE/l0/artefact_audit.txt" 2>&1
      bash scripts/exp/rebuttal_matrix.sh list | grep -v '^---' > "$ARCHIVE/l0/s55_tasks.txt"
      echo "OMNIFLOW_COMPLETION_ORACLE=${OMNIFLOW_COMPLETION_ORACLE:-unset}" > "$ARCHIVE/l0/oracle.txt"
      echo "L0 OK — $(wc -l < "$ARCHIVE/l0/s55_tasks.txt") tasks"
    fi
    ;;
  a|b|c|all)
    [[ -f "$ARCHIVE/l0/preflight.txt" ]] || { echo "先跑 l0 --go" >&2; exit 2; }
    grep -q "ready: run the cell" "$ARCHIVE/l0/preflight.txt" || { echo "L0 未全绿，拒绝开跑" >&2; exit 1; }
    TASK_FILE="$ARCHIVE/l0/s55_tasks.txt"
    ;;
  *) echo "用法: $0 <l0|a|b|c|all> [--go]" >&2; exit 2 ;;
esac

# A 层：同 harness 主表（承诺 1/6）。--baseline all 由注册表一次跑完 5 行。
# APPAGENT_DOCS_ROOT 必须指向 AppAgent 官方 demo_docs 根目录（<appagent>/apps/<app>/demo_docs）；
# 这是资产路径，不猜测、不由 task 名推导。缺失时 appagent_style 一行会失败，故显式拒绝。
if [[ "$LAYER" == "a" || "$LAYER" == "all" ]]; then
  echo "== A 层: baseline all =="
  if [[ -z "${APPAGENT_DOCS_ROOT:-}" ]]; then
    echo "A 层需要 APPAGENT_DOCS_ROOT=<appagent>/apps/<app>/demo_docs（未设置，禁止猜测路径）" >&2
    exit 2
  fi
  while read -r task; do
    for dev in standard45562 fold45564 tablet45554; do
      run bash scripts/exp/run_androidworld.sh run \
        --task "$task" --method omniflow --device "$dev" \
        --baseline all --completion-oracle off --evaluation-seed 113 \
        --memory "data/androidworld/$task/omniflow/OmniFlowSourceSmall_seed111/memory/current/store.json" \
        --source-run-log "data/androidworld/$task/source/OmniFlowSourceSmall_seed111/runlog/current/run_log.json" \
        --element-docs "$APPAGENT_DOCS_ROOT" \
        --output "$ARCHIVE/a/$task/$dev"
    done
  done < "$TASK_FILE"
fi

# B 层：阶梯与消融（承诺 4）。注意 ladder/all 含 L3，L3 未实现会 exit 3，故逐项跑。
if [[ "$LAYER" == "b" || "$LAYER" == "all" ]]; then
  echo "== B 层: ladder + ablation + sweeps =="
  for exp in l1 l2 l4 no_reentry no_checker threshold page; do
    run env TASK_FILE="$TASK_FILE" bash scripts/exp/rebuttal_matrix.sh "$exp"
  done
  echo "[B] L3 缺口按约定走 (b)：报告三档阶梯并承认 confound，不伪造该格"
fi

# C 层：真机与人工标注（承诺 2/3/5），无法脚本化，只列清单。
if [[ "$LAYER" == "c" || "$LAYER" == "all" ]]; then
  echo "== C 层: 需人工/真机，无自动命令 =="
  echo "  - 真机配对 full-task success / execution_duration_ms / energy + device identity + APK hash"
  echo "  - app 更新 / 重复弹窗 / 登录中断 / 网络间歇；每个 trigger 记 not_triggered"
  echo "  - PAPT 128 组 + ASE reviewed correspondence label → false-positive/negative"
  echo "  - Function 成功但 validator 失败的语义失败表"
fi

echo "[done] 结果回流：rsync -av <4090>:$ARCHIVE/ $RUNS/$DATE/"
