#!/usr/bin/env bash

# AndroidWorld paper-condition recipe. Every real run still enters through
# scripts/exp/run_androidworld.sh; this file only fixes the shared comparison
# parameters and selects one explicit stage.

set -eu

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
runner="$repo/scripts/exp/run_androidworld.sh"
stage="${1:?usage: run_condition.sh STAGE}"

task="${TASK:-SystemBluetoothTurnOn}"
device="${DEVICE:-standard45562}"
source_seed="${SOURCE_SEED:-111}"
evaluation_seed="${EVALUATION_SEED:-113}"
max_steps="${MAX_STEPS:-20}"
max_fallback_steps="${MAX_FALLBACK_STEPS:-5}"
deadline="${DEADLINE:-600}"
model="${MODEL:-Qwen3.6-Plus}"
mobilegpt_root="${MOBILEGPT_ROOT:-$repo/data/runtime/external/mobilegpt}"

run_args=(
  run
  --task "$task"
  --device "$device"
  --source-seed "$source_seed"
  --evaluation-seed "$evaluation_seed"
  --max-steps "$max_steps"
  --max-fallback-steps "$max_fallback_steps"
  --deadline "$deadline"
  --model "$model"
)

case "$stage" in
  mobilegpt-cold)
    exec bash "$runner" "${run_args[@]}" --method mobilegpt
    ;;
  mobilegpt-convert)
    source_run_log="${SOURCE_RUN_LOG:?set SOURCE_RUN_LOG to the successful cold run_log.json}"
    memory="${MEMORY:?set MEMORY to a new MobileGPT bundle directory}"
    exec bash "$runner" convert-memory \
      --task "$task" \
      --method mobilegpt \
      --source-run-log "$source_run_log" \
      --memory "$memory" \
      --source-seed "$source_seed" \
      --model "$model" \
      --mobilegpt-root "$mobilegpt_root"
    ;;
  mobilegpt-hot)
    memory="${MEMORY:?set MEMORY to the converted bundle memory directory}"
    exec bash "$runner" "${run_args[@]}" --method mobilegpt --memory "$memory"
    ;;
  omniflow-cold)
    exec bash "$runner" "${run_args[@]}" --method omniflow
    ;;
  omniflow-convert)
    source_run_log="${SOURCE_RUN_LOG:?set SOURCE_RUN_LOG to the successful cold run_log.json}"
    memory="${MEMORY:?set MEMORY to a new OmniFlow memory directory}"
    exec bash "$runner" convert-memory \
      --task "$task" \
      --method omniflow \
      --source-run-log "$source_run_log" \
      --memory "$memory" \
      --source-seed "$source_seed" \
      --model "$model"
    ;;
  omniflow-hot)
    memory="${MEMORY:?set MEMORY to the converted store.json}"
    exec bash "$runner" "${run_args[@]}" --method omniflow --memory "$memory"
    ;;
  fixed-replay)
    source_run_log="${SOURCE_RUN_LOG:?set SOURCE_RUN_LOG to the successful source run_log.json}"
    exec bash "$runner" "${run_args[@]}" \
      --method fixed_replay \
      --source-run-log "$source_run_log"
    ;;
  t3a-hint)
    source_run_log="${SOURCE_RUN_LOG:?set SOURCE_RUN_LOG to the successful source run_log.json}"
    exec bash "$runner" "${run_args[@]}" \
      --method t3a_hint \
      --source-run-log "$source_run_log"
    ;;
  *)
    echo "unknown stage: $stage" >&2
    echo "stages: mobilegpt-cold mobilegpt-convert mobilegpt-hot omniflow-cold omniflow-convert omniflow-hot fixed-replay t3a-hint" >&2
    exit 2
    ;;
esac
