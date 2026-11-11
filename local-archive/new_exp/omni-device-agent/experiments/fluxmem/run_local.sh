#!/bin/bash
set -euo pipefail
# Usage: run_local.sh smoke|profile NEW_RUN_NAME
mode=${1:?smoke or profile}; run_name=${2:?new run name}
cd "$(dirname "$0")/../.."
task_root="$PWD"
upstream="$HOME/.agent-reach/research/fluxmem-repro"
model_dir="$HOME/.agent-reach/research/fluxmem-qwen25vl3b/66285546d2b821cf421d4f5eb2576359d3770cd3"
prepared="$task_root/runs/fluxmem_prepared_001/prepared.json"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
export FORCE_QWENVL_VIDEO_READER=torchvision
command_args=(.venv-fluxmem/bin/python experiments/fluxmem/run_profile.py --upstream "$upstream"
 --model "$model_dir" --prepared "$prepared" --output "$task_root/runs/$run_name/results.json")
case "$mode" in
  smoke) command_args+=(--smoke) ;;
  profile) ;;
  *) exit 2 ;;
esac
mkdir "runs/$run_name"
"${command_args[@]}" > "runs/$run_name/stdout.log" 2> "runs/$run_name/stderr.log"
