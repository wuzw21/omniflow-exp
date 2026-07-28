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
manage_emulators="${OMNIFLOW_SINGLE_TASK_MANAGE_EMULATORS:-1}"
emulator_avds="${OMNIFLOW_SINGLE_TASK_EMULATOR_AVDS:-emulator-5554=SmallPhone,emulator-5564=OmniFlowTargetPixelFoldApi34}"
emulator_gpu="${OMNIFLOW_SINGLE_TASK_EMULATOR_GPU:-swiftshader_indirect}"
emulator_boot_timeout_sec="${OMNIFLOW_SINGLE_TASK_EMULATOR_BOOT_TIMEOUT_SEC:-240}"
fold_serial="${OMNIFLOW_SINGLE_TASK_FOLD_SERIAL:-emulator-5564}"
fold_state="${OMNIFLOW_SINGLE_TASK_FOLD_STATE:-2}"
fold_size="${OMNIFLOW_SINGLE_TASK_FOLD_SIZE:-2208x1840}"
if [[ ! "$task_iteration" =~ ^[1-3]$ ]]; then
  echo "OMNIFLOW_SINGLE_TASK_ITERATION must be an integer from 1 through 3." >&2
  exit 2
fi
if [[ ! "$max_fallback_steps" =~ ^[0-5]$ ]]; then
  echo "OMNIFLOW_SINGLE_TASK_MAX_FALLBACK_STEPS must be an integer from 0 through 5." >&2
  exit 2
fi
if [[ ! "$manage_emulators" =~ ^[01]$ ]]; then
  echo "OMNIFLOW_SINGLE_TASK_MANAGE_EMULATORS must be 0 or 1." >&2
  exit 2
fi
if [[ ! "$emulator_boot_timeout_sec" =~ ^[1-9][0-9]*$ ]]; then
  echo "OMNIFLOW_SINGLE_TASK_EMULATOR_BOOT_TIMEOUT_SEC must be a positive integer." >&2
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
android_sdk_root="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-/home/wuzewen/Android/Sdk}}"
adb_bin="${OMNIFLOW_ADB_PATH:-$android_sdk_root/platform-tools/adb}"
emulator_bin="${OMNIFLOW_EMULATOR_BIN:-$android_sdk_root/emulator/emulator}"
export PATH="/home/wuzewen/.local/bin:$android_sdk_root/platform-tools:$PATH"
export PYTHONPATH="$repo:$repo/src${PYTHONPATH:+:$PYTHONPATH}"

missing_assets=()
require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    missing_assets+=("$label=$path")
  fi
}
require_directory() {
  local label="$1"
  local path="$2"
  if [[ ! -d "$path" ]]; then
    missing_assets+=("$label=$path")
  fi
}
require_command() {
  local label="$1"
  local command_name="$2"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    missing_assets+=("$label=command:$command_name")
  fi
}

require_file "experiment_config" "$config"
require_file "runtime_preflight" "$preflight"
require_file "androidworld_setup" "$android_world_root/android_world/env/setup_device/apps.py"
require_file "adb" "$adb_bin"
require_command "java" "java"
if [[ "$manage_emulators" -eq 1 ]]; then
  require_file "emulator" "$emulator_bin"
fi
if [[ "$requires_omnitransfer" -eq 1 ]]; then
  require_file "omnitransfer_runtime" "$omnitransfer_root/src/omnitransfer/runtime.py"
  require_file "ours_store" "$store_path"
fi
if [[ "$requires_mobilegpt_source_memory" -eq 1 ]]; then
  require_command "jq" "jq"
  require_file "mobilegpt_server" "$mobilegpt_root/Server/main.py"
  require_file "mobilegpt_tasks" "$mobilegpt_source_memory_root/tasks.csv"
  require_file "mobilegpt_cold_manifest" "$(dirname "$mobilegpt_source_memory_root")/cold_memory_manifest.json"
fi
if [[ "$need_appagent_preflight" -eq 1 ]]; then
  require_directory "appagent_root" "$appagent_root"
  require_file "appagent_demo_manifest" "$appagent_demo_memory_root/appagent_demo_manifest.json"
fi
if [[ "$need_mobile_agent_v3_preflight" -eq 1 ]]; then
  require_directory "mobile_agent_v3_root" "$mobile_agent_v3_root"
  require_directory "mobile_agent_v3_model_root" "$mobile_agent_v3_model_root"
fi
if [[ ${#missing_assets[@]} -gt 0 ]]; then
  echo "Static experiment asset preflight failed before device startup:" >&2
  printf '  - %s\n' "${missing_assets[@]}" >&2
  exit 1
fi

target_serials=()
IFS=',' read -r -a target_specs <<< "$device_targets"
for target_spec in "${target_specs[@]}"; do
  IFS=':' read -r target_label target_serial target_console_port target_extra <<< "$target_spec"
  if [[ -z "$target_label" || -z "$target_serial" || ! "$target_console_port" =~ ^[0-9]+$ || -n "${target_extra:-}" ]]; then
    echo "Invalid device target: $target_spec" >&2
    exit 2
  fi
  if [[ "$target_serial" != "emulator-$target_console_port" ]]; then
    echo "Device target serial/console mismatch: $target_spec" >&2
    exit 2
  fi
  target_serials+=("$target_serial")
done
if [[ ${#target_serials[@]} -eq 0 ]]; then
  echo "At least one device target is required." >&2
  exit 2
fi

avd_for_serial() {
  local wanted_serial="$1"
  local mapping mapping_serial mapping_avd
  IFS=',' read -r -a mappings <<< "$emulator_avds"
  for mapping in "${mappings[@]}"; do
    mapping_serial="${mapping%%=*}"
    mapping_avd="${mapping#*=}"
    if [[ "$mapping_serial" == "$wanted_serial" && "$mapping_avd" != "$mapping" ]]; then
      printf '%s\n' "$mapping_avd"
      return 0
    fi
  done
  return 1
}

device_state() {
  local serial="$1"
  local devices
  devices="$("$adb_bin" devices 2>/dev/null || true)"
  awk -v wanted="$serial" '$1 == wanted {print $2; exit}' <<< "$devices"
}

grpc_ready() {
  local port="$1"
  "$python_bin" - "$port" <<'PY'
import socket
import sys

try:
    with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=0.5):
        pass
except OSError:
    raise SystemExit(1)
PY
}

wait_for_emulator() {
  local serial="$1"
  local grpc_port="$2"
  local log_path="$3"
  local deadline now boot_completed
  deadline="$(( $(date +%s) + emulator_boot_timeout_sec ))"
  while true; do
    boot_completed=""
    if [[ "$(device_state "$serial")" == "device" ]]; then
      boot_completed="$("$adb_bin" -s "$serial" shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')"
    fi
    if [[ "$boot_completed" == "1" ]] && grpc_ready "$grpc_port"; then
      echo "[emulator] ready serial=$serial grpc=$grpc_port"
      return 0
    fi
    now="$(date +%s)"
    if (( now >= deadline )); then
      echo "Emulator did not become ready: serial=$serial grpc=$grpc_port log=$log_path" >&2
      tail -n 80 "$log_path" >&2 || true
      return 1
    fi
    sleep 1
  done
}

ensure_emulator() {
  local serial="$1"
  local console_port="${serial#emulator-}"
  local grpc_port="$(( console_port + 3000 ))"
  local avd log_path current_state stop_deadline
  current_state="$(device_state "$serial")"
  if [[ "$current_state" == "device" ]] && grpc_ready "$grpc_port"; then
    echo "[emulator] reuse serial=$serial grpc=$grpc_port"
    return 0
  fi
  if [[ "$manage_emulators" -ne 1 ]]; then
    echo "Emulator is not ready and automatic management is disabled: serial=$serial grpc=$grpc_port" >&2
    return 1
  fi
  if [[ -n "$current_state" ]]; then
    echo "[emulator] restart serial=$serial state=$current_state grpc=$grpc_port"
    "$adb_bin" -s "$serial" emu kill >/dev/null 2>&1 || true
    stop_deadline="$(( $(date +%s) + 30 ))"
    while [[ -n "$(device_state "$serial")" ]]; do
      if (( $(date +%s) >= stop_deadline )); then
        echo "Existing emulator could not be stopped safely: $serial" >&2
        return 1
      fi
      sleep 1
    done
  elif grpc_ready "$grpc_port"; then
    echo "gRPC port is occupied without its emulator: 127.0.0.1:$grpc_port" >&2
    return 1
  fi
  if ! avd="$(avd_for_serial "$serial")"; then
    echo "No AVD mapping configured for $serial in OMNIFLOW_SINGLE_TASK_EMULATOR_AVDS." >&2
    return 1
  fi
  if ! "$emulator_bin" -list-avds | grep -Fqx "$avd"; then
    echo "Configured AVD is not installed: serial=$serial avd=$avd" >&2
    return 1
  fi
  log_path="$output_root/emulator_${serial#emulator-}.log"
  echo "[emulator] launch serial=$serial avd=$avd grpc=$grpc_port"
  nohup "$emulator_bin" \
    -avd "$avd" \
    -port "$console_port" \
    -grpc "$grpc_port" \
    -no-window \
    -no-audio \
    -no-boot-anim \
    -no-snapshot-save \
    -gpu "$emulator_gpu" \
    >"$log_path" 2>&1 </dev/null &
  wait_for_emulator "$serial" "$grpc_port" "$log_path"
}

ensure_fold_state() {
  local selected=0 serial
  if [[ -z "$fold_serial" ]]; then
    return 0
  fi
  for serial in "${target_serials[@]}"; do
    if [[ "$serial" == "$fold_serial" ]]; then
      selected=1
      break
    fi
  done
  if [[ "$selected" -ne 1 ]]; then
    return 0
  fi
  "$adb_bin" -s "$fold_serial" shell cmd device_state state "$fold_state" >/dev/null
  local deadline current_state current_size
  deadline="$(( $(date +%s) + 30 ))"
  while true; do
    current_state="$("$adb_bin" -s "$fold_serial" shell cmd device_state print-state 2>/dev/null | tr -d '\r')"
    current_size="$("$adb_bin" -s "$fold_serial" shell wm size 2>/dev/null | tr -d '\r')"
    if [[ "$current_state" == "$fold_state" && "$current_size" == *"$fold_size"* ]]; then
      echo "[emulator] fold-ready serial=$fold_serial state=$current_state size=$fold_size"
      return 0
    fi
    if (( $(date +%s) >= deadline )); then
      echo "Pixel Fold did not reach required state/size: serial=$fold_serial expected_state=$fold_state expected_size=$fold_size actual_state=$current_state actual_size=$current_size" >&2
      return 1
    fi
    sleep 1
  done
}

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
  preflight_serials="${target_serials[*]}"
fi

mkdir -p "$output_root"
for serial in "${target_serials[@]}"; do
  ensure_emulator "$serial"
done
ensure_fold_state
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
