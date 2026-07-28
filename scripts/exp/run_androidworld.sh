#!/usr/bin/env bash

set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
asset_root="${OMNIFLOW_EXP_ASSET_ROOT:-}"
results_root="${OMNIFLOW_EXP_RESULTS_ROOT:-}"
python_bin="${PYTHON_BIN:-python3}"
env_file="${OMNIFLOW_ENV_FILE:-${asset_root:+$asset_root/.env}}"
master_source_index="${OMNIFLOW_MASTER_SOURCE_INDEX:-${asset_root:+$asset_root/runtime/evals/androidworld_validator/core_archive/success_source_runlogs/index_by_task.json}}"
source_index="${OMNIFLOW_SINGLE_TASK_SOURCE_INDEX:-$master_source_index}"
expected_source_seed="${OMNIFLOW_SINGLE_TASK_SOURCE_SEED:-111}"
omnitransfer_root="${OMNITRANSFER_ROOT:-}"
android_world_root="${OMNIFLOW_ANDROID_WORLD_ROOT:-${asset_root:+$asset_root/runtime/external/droidrun-android-world/android_world}}"
config="$repo/config/paper_androidworld.json"
preflight="$repo/skills/androidworld-runtime-preflight/scripts/preflight.py"
task="${OMNIFLOW_SINGLE_TASK_TASK:-SystemBluetoothTurnOn}"
task_iteration="${OMNIFLOW_SINGLE_TASK_ITERATION:-1}"
all_methods="fixed_replay,ours,mobilegpt_offline_retrieval,appagent_demo,mobile_agent_v3"
if [[ "$task_iteration" == "1" ]]; then
  default_methods="$all_methods"
else
  default_methods="ours"
fi
methods="${OMNIFLOW_SINGLE_TASK_METHODS:-$default_methods}"
baseline_environment_repair="${OMNIFLOW_BASELINE_ENVIRONMENT_REPAIR_REASON:-}"
device_targets="${OMNIFLOW_SINGLE_TASK_DEVICE_TARGETS:-small5554:emulator-5554:5554,fold5564:emulator-5564:5564}"
fixed_task_params="${OMNIFLOW_SINGLE_TASK_FIXED_TASK_PARAMS:-0}"
timeout_sec="${OMNIFLOW_SINGLE_TASK_TIMEOUT_SEC:-600}"
max_fallback_steps="${OMNIFLOW_SINGLE_TASK_MAX_FALLBACK_STEPS:-5}"
store_path="${OMNIFLOW_SINGLE_TASK_STORE_PATH:-${asset_root:+$asset_root/runtime/evals/androidworld_single_task_assets/source_seed_111/$task/ours/store_r1/store.json}}"
mobilegpt_root="${OMNIFLOW_MOBILEGPT_ROOT:-${asset_root:+$asset_root/runtime/external/mobilegpt}}"
mobilegpt_source_memory_root="${OMNIFLOW_MOBILEGPT_SOURCE_MEMORY_ROOT:-${asset_root:+$asset_root/runtime/evals/androidworld_single_task_assets/source_seed_111/$task/mobilegpt_offline_retrieval/native_source_r2/memory}}"
appagent_root="${OMNIFLOW_APPAGENT_ROOT:-${asset_root:+$asset_root/runtime/external/appagent}}"
appagent_demo_memory_root="${OMNIFLOW_APPAGENT_DEMO_MEMORY_ROOT:-${asset_root:+$asset_root/runtime/evals/androidworld_single_task_assets/source_seed_111/$task/appagent_demo/native_source_r2}}"
mobile_agent_v3_root="${OMNIFLOW_MOBILE_AGENT_V3_ROOT:-${asset_root:+$asset_root/runtime/external/mobileagent}}"
mobile_agent_v3_official_revision="${OMNIFLOW_MOBILE_AGENT_V3_OFFICIAL_REVISION:-11cea575561fb7800b5fb6b6cafa56f7a91de11f}"
mobile_agent_v3_model_root="${OMNIFLOW_MOBILE_AGENT_V3_MODEL_ROOT:-/home/wuzewen/models/GUI-Owl-7B-7c1644c0288da07435a485701d0fea0ac353f38a}"
mobile_agent_v3_model_revision="${OMNIFLOW_MOBILE_AGENT_V3_MODEL_REVISION:-7c1644c0288da07435a485701d0fea0ac353f38a}"
mobile_agent_v3_model="${OMNIFLOW_MOBILE_AGENT_V3_MODEL:-GUI-Owl-7B}"
mobile_agent_v3_base_url="${OMNIFLOW_MOBILE_AGENT_V3_BASE_URL:-http://127.0.0.1:4243/v1}"
mobile_agent_v3_api_key="${OMNIFLOW_MOBILE_AGENT_V3_API_KEY:-local-vllm}"
preflight_profile="${OMNIFLOW_SINGLE_TASK_PREFLIGHT_PROFILE:-}"
preflight_serials="${OMNIFLOW_SINGLE_TASK_PREFLIGHT_SERIALS:-}"
if [[ ! "$task_iteration" =~ ^[1-3]$ ]]; then
  echo "OMNIFLOW_SINGLE_TASK_ITERATION must be an integer from 1 through 3." >&2
  exit 2
fi
if [[ ! "$max_fallback_steps" =~ ^[0-5]$ ]]; then
  echo "OMNIFLOW_SINGLE_TASK_MAX_FALLBACK_STEPS must be an integer from 0 through 5." >&2
  exit 2
fi
printf -v iteration_label '%02d' "$task_iteration"
attempt_id="iteration_${iteration_label}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
attempt_series_root="${results_root:+$results_root/androidworld_single_task_attempts/$task}"
output_root="${OMNIFLOW_SINGLE_TASK_OUTPUT_ROOT:-$attempt_series_root/$attempt_id}"
dry_run=0
requires_mobilegpt_source_memory=0
requires_omnitransfer=0
need_native_preflight=0
need_mobilegpt_preflight=0
need_appagent_preflight=0
need_mobile_agent_v3_preflight=0
contains_baseline_method=0

for method in ${methods//,/ }; do
  case "$method" in
    ours)
      requires_omnitransfer=1
      need_native_preflight=1
      ;;
    fixed_replay)
      need_native_preflight=1
      contains_baseline_method=1
      ;;
    mobilegpt_offline_retrieval)
      need_mobilegpt_preflight=1
      requires_mobilegpt_source_memory=1
      contains_baseline_method=1
      ;;
    appagent_demo)
      need_appagent_preflight=1
      contains_baseline_method=1
      ;;
    mobile_agent_v3)
      need_mobile_agent_v3_preflight=1
      contains_baseline_method=1
      ;;
    *)
      echo "Unsupported paper method: $method" >&2
      exit 2
      ;;
  esac
done
if [[ "$task_iteration" != "1" && "$contains_baseline_method" -eq 1 && -z "$baseline_environment_repair" ]]; then
  echo "Baseline methods are frozen after iteration 1. Set OMNIFLOW_BASELINE_ENVIRONMENT_REPAIR_REASON only for an audited environment-only retry." >&2
  exit 2
fi

if [[ $# -gt 1 || ($# -eq 1 && "$1" != "--dry-run") ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi
if [[ $# -eq 1 ]]; then
  dry_run=1
fi
if [[ -z "$asset_root" || -z "$results_root" ]]; then
  echo "Set OMNIFLOW_EXP_ASSET_ROOT and OMNIFLOW_EXP_RESULTS_ROOT to external absolute paths." >&2
  exit 2
fi
if [[ "$asset_root" != /* || "$results_root" != /* ]]; then
  echo "Experiment asset and result roots must be absolute paths." >&2
  exit 2
fi
if ! python_bin="$(command -v "$python_bin")"; then
  echo "Python runtime missing: ${PYTHON_BIN:-python3}" >&2
  exit 1
fi
"$python_bin" - "$task_iteration" "$attempt_series_root" "$(dirname "$output_root")" <<'PY'
import json
import sys
from pathlib import Path

iteration = int(sys.argv[1])
roots = {Path(value).expanduser().resolve() for value in sys.argv[2:]}
matches = []
for root in roots:
    if not root.is_dir():
        continue
    for manifest_path in root.glob("*/attempt_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            manifest.get("dry_run") is not True
            and int(manifest.get("task_iteration") or 0) == iteration
        ):
            matches.append(str(manifest_path))
if matches:
    raise SystemExit(
        "task_iteration_already_executed:"
        f"iteration={iteration}:manifests={','.join(sorted(matches))}"
    )
PY
if [[ ! -f "$env_file" ]]; then
  echo "Model environment file missing: $env_file" >&2
  exit 1
fi
if [[ ! -f "$source_index" ]]; then
  echo "Canonical source index missing: $source_index" >&2
  exit 1
fi
if [[ ! -f "$master_source_index" ]]; then
  echo "Canonical master source index missing: $master_source_index" >&2
  exit 1
fi
"$python_bin" - "$source_index" "$task" "$expected_source_seed" <<'PY'
import json
import sys
from pathlib import Path

index_path = Path(sys.argv[1])
task_name = sys.argv[2]
expected_seed = int(sys.argv[3])
payload = json.loads(index_path.read_text(encoding="utf-8"))
row = payload.get(task_name) if isinstance(payload, dict) else None
if not isinstance(row, dict):
    raise SystemExit(f"canonical_source_task_missing:{task_name}")
actual_seed = row.get("source_seed", row.get("replay_seed"))
if actual_seed != expected_seed:
    raise SystemExit(
        f"formal_source_seed_mismatch:{task_name}:"
        f"expected={expected_seed}:actual={actual_seed}"
    )
print(f"[source] task={task_name} seed={actual_seed} index={index_path}")
PY
if [[ "$requires_omnitransfer" -eq 1 && ( -z "$omnitransfer_root" || ! -f "$omnitransfer_root/src/omnitransfer/runtime.py" ) ]]; then
  echo "Set OMNITRANSFER_ROOT to the exact versioned OmniTransfer release." >&2
  exit 1
fi
if [[ -e "$output_root" ]]; then
  echo "Immutable attempt already exists: $output_root" >&2
  exit 1
fi

set -a
source "$env_file"
set +a
export OMNITRANSFER_ROOT="$omnitransfer_root"
unset OMNIFLOW_OOB_DEVICE_URL
export OMNIFLOW_OBSERVE_BACKEND="androidworld"
export OMNIFLOW_ACT_BACKEND="androidworld"
export PATH="/home/wuzewen/.local/bin:/home/wuzewen/Android/Sdk/platform-tools:$PATH"
export PYTHONPATH="$repo:$repo/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "$preflight_profile" ]]; then
  preflight_profiles="$preflight_profile"
else
  preflight_profiles=""
  if [[ "$need_native_preflight" -eq 1 ]]; then
    preflight_profiles+=" androidworld_native"
  fi
  if [[ "$need_mobilegpt_preflight" -eq 1 ]]; then
    preflight_profiles+=" mobilegpt"
  fi
  if [[ "$need_appagent_preflight" -eq 1 ]]; then
    preflight_profiles+=" appagent"
  fi
  if [[ "$need_mobile_agent_v3_preflight" -eq 1 ]]; then
    preflight_profiles+=" mobile_agent_v3"
  fi
fi
if [[ -z "$preflight_serials" ]]; then
  preflight_serials="emulator-5554 emulator-5564"
fi

mkdir -p "$output_root"
for profile in $preflight_profiles; do
for serial in $preflight_serials; do
  preflight_args=(
    --repo "$asset_root"
    --code-root "$repo"
    --profile "$profile"
    --serial "$serial"
    --require-kvm
    --require-device
    --oob-url ""
    --json-out "$output_root/runtime_preflight_${profile}_${serial#emulator-}.json"
  )
  if [[ "$profile" == "mobilegpt" ]]; then
    preflight_args+=(--expected-tasks 116)
    if [[ "$requires_mobilegpt_source_memory" -eq 1 ]]; then
      preflight_args+=(
        --source-memory-root "$mobilegpt_source_memory_root"
        --expected-memory-tasks 1
      )
    fi
  fi
  if [[ "$profile" == "androidworld_native" ]]; then
    preflight_args+=(
      --expected-tasks 116
      --source-index "$master_source_index"
    )
  fi
  if [[ "$profile" == "appagent" ]]; then
    preflight_args+=(--appagent-root "$appagent_root")
    if [[ -n "$appagent_demo_memory_root" ]]; then
      preflight_args+=(--appagent-demo-memory-root "$appagent_demo_memory_root")
    fi
  fi
  if [[ "$profile" == "mobile_agent_v3" ]]; then
    if [[ -z "$mobile_agent_v3_model_root" ]]; then
      echo "Set OMNIFLOW_MOBILE_AGENT_V3_MODEL_ROOT to the pinned GUI-Owl snapshot." >&2
      exit 1
    fi
    preflight_args+=(
      --expected-tasks 116
      --source-index "$master_source_index"
      --mobile-agent-v3-root "$mobile_agent_v3_root"
      --mobile-agent-v3-official-revision "$mobile_agent_v3_official_revision"
      --mobile-agent-v3-model-root "$mobile_agent_v3_model_root"
      --mobile-agent-v3-model-revision "$mobile_agent_v3_model_revision"
      --mobile-agent-v3-model "$mobile_agent_v3_model"
      --mobile-agent-v3-base-url "$mobile_agent_v3_base_url"
      --mobile-agent-v3-api-key "$mobile_agent_v3_api_key"
    )
  fi
  if [[ "$profile" == "mobilegpt" ]]; then
    preflight_args+=(--require-contacts-ready)
  fi
  "$python_bin" "$preflight" "${preflight_args[@]}"
done
done

command=(
  "$python_bin"
  -m
  src.experiment.androidworld
  one-task
  --experiment-config "$config"
  --index "$source_index"
  --android-world-root "$android_world_root"
  --tasks "$task"
  --task-iteration "$task_iteration"
  --output-root "$output_root"
  --master-source-index "$master_source_index"
  --result-registry-root "$results_root/androidworld_validator/runs"
  --master-progress-root "$results_root/androidworld_validator/master_progress"
  --omnitransfer-root "$omnitransfer_root"
  --store-path "$store_path"
  --mobilegpt-root "$mobilegpt_root"
  --mobilegpt-source-memory-root "$mobilegpt_source_memory_root"
  --appagent-root "$appagent_root"
  --mobile-agent-v3-root "$mobile_agent_v3_root"
  --mobile-agent-v3-official-revision "$mobile_agent_v3_official_revision"
  --mobile-agent-v3-model-root "$mobile_agent_v3_model_root"
  --mobile-agent-v3-model-revision "$mobile_agent_v3_model_revision"
  --mobile-agent-v3-model "$mobile_agent_v3_model"
  --mobile-agent-v3-base-url "$mobile_agent_v3_base_url"
  --mobile-agent-v3-api-key "$mobile_agent_v3_api_key"
  --timeout-sec "$timeout_sec"
  --max-fallback-steps "$max_fallback_steps"
)
if [[ -n "$baseline_environment_repair" ]]; then
  command+=(--baseline-environment-repair "$baseline_environment_repair")
fi
if [[ -n "$methods" ]]; then
  command+=(--methods "$methods")
fi
if [[ "$fixed_task_params" != "1" ]]; then
  command+=(--no-fixed-task-params --task-params-json "")
fi
if [[ -n "$device_targets" ]]; then
  command+=(--device-targets "$device_targets")
fi
if [[ -n "$appagent_demo_memory_root" ]]; then
  command+=(--appagent-demo-memory-root "$appagent_demo_memory_root")
fi
if [[ "$dry_run" -eq 1 ]]; then
  command+=(--dry-run)
fi

exec "${command[@]}"
