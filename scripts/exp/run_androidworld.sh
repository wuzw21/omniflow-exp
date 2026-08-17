#!/usr/bin/env bash

set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
workspace_root="$(cd "$repo/.." && pwd)"
default_asset_root="$workspace_root/OmniFlow"
default_memory_root="$workspace_root/assets/androidworld-experiment-memory-v1"
asset_root="${OMNIFLOW_EXP_ASSET_ROOT:-$default_asset_root}"
results_root="${OMNIFLOW_EXP_RESULTS_ROOT:-$asset_root/runtime/evals}"
account_root="$(cd && pwd)"
unified_python="$account_root/miniconda3/envs/omniflow-py31113/bin/python"
if [[ ! -x "$unified_python" ]]; then
  unified_python="$account_root/miniconda3/bin/python"
fi
if [[ ! -x "$unified_python" ]]; then
  unified_python="python3"
fi
python_bin="${PYTHON_BIN:-$unified_python}"
env_file="${OMNIFLOW_ENV_FILE:-${asset_root:+$asset_root/.env}}"
master_source_index="${OMNIFLOW_MASTER_SOURCE_INDEX:-${asset_root:+$asset_root/runtime/evals/androidworld_validator/core_archive/success_source_runlogs/index_by_task.json}}"
source_index="${OMNIFLOW_ANDROIDWORLD_SOURCE_INDEX:-$master_source_index}"
source_index_expected_tasks=""
formal_source_seed=""
formal_task_seed=""
formal_max_steps=""
formal_max_fallback_steps=""
formal_step_timeout_sec=""
official_validator_flush_grace_sec=""
formal_fixed_task_params=""
formal_fold_state=""
formal_fold_size=""
formal_model=""
formal_model_endpoint_profile=""
formal_model_base_url=""
formal_bmoca_revision="de06497ae51464dd06fe4dbd2e5f59f27bcd9250"
execution_environment="androidworld"
bmoca_root="${OMNIFLOW_BMOCA_ROOT:-}"
bmoca_environment_ids="${OMNIFLOW_BMOCA_ENVIRONMENT_IDS:-100,101,102,103,104,105,106,107,108,109}"
bmoca_avd_home="${OMNIFLOW_BMOCA_AVD_HOME:-${ANDROID_AVD_HOME:-}}"
bmoca_avd_template_home="${OMNIFLOW_BMOCA_AVD_TEMPLATE_HOME:-}"
bmoca_output_path="${OMNIFLOW_BMOCA_OUTPUT_PATH:-}"
bmoca_show_emulator="${OMNIFLOW_BMOCA_SHOW_EMULATOR:-0}"
android_world_revision="$(PYTHONPATH="$repo:$repo/src${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PY'
from src.experiment.protocol import ANDROIDWORLD_REVISION

print(ANDROIDWORLD_REVISION)
PY
)"
mobilegpt_source_schema="omniflow.mobilegpt-runlog-direct-memory.v1"
mobilegpt_source_method="mobilegpt_runlog_direct_memory"
mobilegpt_source_manifest_name="mobilegpt_memory_manifest.json"
expected_source_seed=""
task_seed=""
omnitransfer_root="${OMNITRANSFER_ROOT:-$workspace_root/OmniTransfer}"
android_world_release_root="${OMNIFLOW_ANDROIDWORLD_RELEASE_ROOT:-$(dirname "$asset_root")/releases/android-world-$android_world_revision}"
android_world_root="${OMNIFLOW_ANDROID_WORLD_ROOT:-$android_world_release_root}"
export PYTHONPATH="$repo:$repo/src${android_world_root:+:$android_world_root}${PYTHONPATH:+:$PYTHONPATH}"
protocol_values="$(python3 - <<'PY'
from src.experiment.protocol import (
    DEFAULT_DEVICE,
    DEFAULT_METHOD,
    DEVICES,
    FORMAL_MODEL,
    FORMAL_MODEL_BASE_URL,
    FORMAL_MODEL_ENDPOINT_PROFILE,
    FIXED_TASK_PARAMS,
    FOLD_SIZE,
    FOLD_STATE,
    MAX_FALLBACK_STEPS,
    MAX_STEPS,
    METHODS,
    SOURCE_DEVICE,
    SOURCE_SEED,
    STEP_TIMEOUT_SEC,
    TASK_DEADLINE_SEC,
    TASK_SEED,
    VALIDATOR_FLUSH_GRACE_SEC,
)

print(
    SOURCE_SEED,
    TASK_SEED,
    MAX_STEPS,
    MAX_FALLBACK_STEPS,
    STEP_TIMEOUT_SEC,
    VALIDATOR_FLUSH_GRACE_SEC,
    TASK_DEADLINE_SEC,
    FORMAL_MODEL,
    FORMAL_MODEL_ENDPOINT_PROFILE,
    FORMAL_MODEL_BASE_URL,
    int(FIXED_TASK_PARAMS),
    FOLD_STATE,
    FOLD_SIZE,
    DEFAULT_METHOD,
    ",".join(METHODS),
    ":".join(str(value) for value in SOURCE_DEVICE),
    DEFAULT_DEVICE,
    next(serial for label, serial, _ in DEVICES if label.startswith("fold")),
)
PY
)"
read -r formal_source_seed formal_task_seed formal_max_steps \
  formal_max_fallback_steps formal_step_timeout_sec \
  official_validator_flush_grace_sec \
  formal_task_deadline_sec formal_model formal_model_endpoint_profile \
  formal_model_base_url formal_fixed_task_params formal_fold_state formal_fold_size \
  formal_default_method all_methods \
  source_device default_device fold_serial <<< "$protocol_values"
expected_source_seed="${OMNIFLOW_ANDROIDWORLD_SOURCE_SEED:-$formal_source_seed}"
task_seed="${OMNIFLOW_ANDROIDWORLD_TASK_SEED:-$formal_task_seed}"
preflight="$repo/src/experiment/preflight.py"
selected_method_arg=""
selected_device_arg=""
task="${OMNIFLOW_ANDROIDWORLD_TASK:-SystemBluetoothTurnOn}"
task_iteration="${OMNIFLOW_ANDROIDWORLD_TASK_ITERATION:-1}"
baseline_environment_repair="${OMNIFLOW_BASELINE_ENVIRONMENT_REPAIR_REASON:-}"
mobilegpt_source_environment_repair="${OMNIFLOW_MOBILEGPT_SOURCE_ENVIRONMENT_REPAIR_REASON:-}"
appagent_source_environment_repair="${OMNIFLOW_APPAGENT_SOURCE_ENVIRONMENT_REPAIR_REASON:-}"
batch_attempt_id="${OMNIFLOW_BATCH_ATTEMPT_ID:-}"
device_target="${OMNIFLOW_ANDROIDWORLD_DEVICE:-$default_device}"
fixed_task_params="$formal_fixed_task_params"
timeout_sec=""
preflight_minimum_free_gb="${OMNIFLOW_PREFLIGHT_MINIMUM_FREE_GB:-20}"
max_steps="${OMNIFLOW_ANDROIDWORLD_MAX_STEPS:-$formal_max_steps}"
max_fallback_steps="${OMNIFLOW_ANDROIDWORLD_MAX_FALLBACK_STEPS:-$formal_max_fallback_steps}"
store_path="${OMNIFLOW_ANDROIDWORLD_STORE_PATH:-}"
ours_store_index="${OMNIFLOW_OURS_STORE_INDEX:-}"
ours_source_asset_index="${OMNIFLOW_OURS_SOURCE_ASSET_INDEX:-$master_source_index}"
ours_converted_asset_root="${OMNIFLOW_OURS_CONVERTED_ASSET_ROOT:-}"
ours_authoring_manifest="${OMNIFLOW_OURS_AUTHORING_MANIFEST:-}"
ours_revision_reason="${OMNIFLOW_OURS_REVISION_REASON:-}"
memory_root="${OMNIFLOW_EXP_MEMORY_ROOT:-$default_memory_root}"
memory_index="${OMNIFLOW_EXP_MEMORY_INDEX:-${memory_root:+$memory_root/current.json}}"
memory_function_catalogs="${OMNIFLOW_MEMORY_FUNCTION_CATALOGS:-}"
memory_runlog_roots="${OMNIFLOW_MEMORY_RUNLOG_ROOTS:-${asset_root:+$asset_root/runtime/evals}}"
memory_result_roots="${OMNIFLOW_MEMORY_RESULT_ROOTS:-${asset_root:+$asset_root/runtime/evals}}"
memory_mobilegpt_roots="${OMNIFLOW_MEMORY_MOBILEGPT_ROOTS:-}"
memory_baseline_batch_reports="${OMNIFLOW_MEMORY_BASELINE_BATCH_REPORTS:-}"
source_selection_manifest="${OMNIFLOW_SOURCE_SELECTION_MANIFEST:-}"
function_store_selection_manifest="${OMNIFLOW_FUNCTION_STORE_SELECTION_MANIFEST:-}"
if [[ -n "$results_root" && ":$memory_result_roots:" != *":$results_root:"* ]]; then
  memory_result_roots="${memory_result_roots:+$memory_result_roots:}$results_root"
fi
mobilegpt_root="${OMNIFLOW_MOBILEGPT_ROOT:-${asset_root:+$asset_root/runtime/external/mobilegpt}}"
mobilegpt_source_memory_root="${OMNIFLOW_MOBILEGPT_SOURCE_MEMORY_ROOT:-}"
appagent_root="${OMNIFLOW_APPAGENT_ROOT:-${asset_root:+$asset_root/runtime/external/appagent}}"
appagent_demo_memory_root="${OMNIFLOW_APPAGENT_DEMO_MEMORY_ROOT:-}"
preflight_profile=""
preflight_serials=""
manage_emulators="${OMNIFLOW_ANDROIDWORLD_MANAGE_EMULATORS:-1}"
emulator_avds="emulator-5554=OmniFlowTargetSmall,emulator-5560=SmallPhone,emulator-5564=OmniFlowTargetFold"
host_machine="$(uname -m)"
case "$host_machine" in
  x86_64|amd64)
    default_emulator_system_image_abi="x86_64"
    ;;
  arm64|aarch64)
    default_emulator_system_image_abi="arm64-v8a"
    ;;
  *)
    default_emulator_system_image_abi="arm64-v8a"
    ;;
esac
resolve_default_android_sdk_root() {
  local user_root="${1:-$account_root}"
  local candidate
  for candidate in \
    "$user_root/Library/Android/sdk" \
    "$user_root/Android/Sdk"; do
    if [[ -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  printf '%s\n' "$user_root/Android/Sdk"
}
ensure_androidworld_sqlite_fts4() {
  if "$python_bin" - <<'PY' >/dev/null 2>&1
import sqlite3

connection = sqlite3.connect(":memory:")
connection.execute("CREATE VIRTUAL TABLE androidworld_fts4_probe USING fts4(value)")
PY
  then
    return 0
  fi

  local candidate
  local configured_library="${OMNIFLOW_SQLITE_FTS4_LIBRARY:-}"
  local candidates=("")
  if [[ -n "$configured_library" ]]; then
    candidates+=("$configured_library")
  elif [[ "$(uname -s)" == "Linux" ]]; then
    if command -v ldconfig >/dev/null 2>&1; then
      while IFS= read -r candidate; do
        [[ -n "$candidate" ]] && candidates+=("$candidate")
      done < <(ldconfig -p 2>/dev/null | awk '$1 == "libsqlite3.so.0" {print $NF}')
    fi
    candidates+=(
      "/usr/lib/x86_64-linux-gnu/libsqlite3.so.0"
      "/lib/x86_64-linux-gnu/libsqlite3.so.0"
      "/usr/lib/aarch64-linux-gnu/libsqlite3.so.0"
      "/lib/aarch64-linux-gnu/libsqlite3.so.0"
    )
  fi

  for candidate in "${candidates[@]}"; do
    [[ "$candidate" == /* && -f "$candidate" ]] || continue
    local candidate_preload="$candidate${LD_PRELOAD:+:$LD_PRELOAD}"
    if LD_PRELOAD="$candidate_preload" "$python_bin" - <<'PY' >/dev/null 2>&1
import sqlite3

connection = sqlite3.connect(":memory:")
connection.execute("CREATE VIRTUAL TABLE androidworld_fts4_probe USING fts4(value)")
PY
    then
      export LD_PRELOAD="$candidate_preload"
      echo "[sqlite] AndroidWorld FTS4 enabled library=$candidate"
      return 0
    fi
  done

  echo "AndroidWorld setup requires Python SQLite with FTS4 support. Set OMNIFLOW_SQLITE_FTS4_LIBRARY to an absolute compatible libsqlite3 path." >&2
  return 1
}
default_emulator_avd_specs="SmallPhone|system-images;android-33;google_apis;$default_emulator_system_image_abi|small_phone,OmniFlowTargetSmall|system-images;android-33;google_apis;$default_emulator_system_image_abi|small_phone,OmniFlowTargetFold|system-images;android-34;google_apis;$default_emulator_system_image_abi|pixel_fold"
emulator_avd_specs="$default_emulator_avd_specs"
emulator_gpu="swiftshader_indirect"
emulator_boot_timeout_sec="240"
emulator_graceful_shutdown_timeout_sec="30"
emulator_forced_shutdown_timeout_sec="10"
androidworld_adb_file_transfer_timeout_sec="300"
export OMNIFLOW_ANDROIDWORLD_ADB_FILE_TRANSFER_TIMEOUT_SEC="$androidworld_adb_file_transfer_timeout_sec"
androidworld_setup_timeout_sec="300"
export OMNIFLOW_ANDROIDWORLD_SETUP_TIMEOUT_SEC="$androidworld_setup_timeout_sec"
fold_state="$formal_fold_state"
fold_size="$formal_fold_size"
dry_run=0
check_only=0
development_run=0
source_qualification_only=0
source_collection=0
all_tasks=0
batch_task_filter=""
convert_ours_assets=0
refresh_memory=0
convert_source_runlogs=0
prepare_mobilegpt_memory=0
convert_runlog_memory_method=""
runlog_memory_output_root="${OMNIFLOW_RUNLOG_MEMORY_OUTPUT_ROOT:-}"
e2e_task=""
  e2e_task_deadline_sec="${OMNIFLOW_E2E_TASK_DEADLINE_SEC:-$formal_task_deadline_sec}"
mobilegpt_memory_output_root="${OMNIFLOW_MOBILEGPT_MEMORY_OUTPUT_ROOT:-}"
source_runlog_output_root="${OMNIFLOW_SOURCE_RUNLOG_OUTPUT_ROOT:-${memory_root:+$memory_root/source_runlogs}}"
source_screenshot_roots="${OMNIFLOW_SOURCE_SCREENSHOT_ROOTS:-}"
runlog_memory_source_runlog="${OMNIFLOW_RUNLOG_MEMORY_SOURCE_RUNLOG:-}"

select_model_endpoint() {
  local profile="$1"
  local selected_model_config
  if ! selected_model_config="$($python_bin - "$profile" <<'PY'
import sys

from omniflow.vlm.model_config import resolve_openai_compatible_config

profile = sys.argv[1]
try:
    api_key, base_url = resolve_openai_compatible_config(profile=profile)
except ValueError as error:
    raise SystemExit(str(error)) from error
if not api_key or not base_url:
    raise SystemExit(f"model_endpoint_profile_incomplete:{profile}")
print(api_key)
print(base_url)
PY
  )"; then
    exit 2
  fi
  selected_model_api_key="${selected_model_config%%$'\n'*}"
  selected_model_base_url="${selected_model_config#*$'\n'}"
  if [[ -z "$selected_model_api_key" || -z "$selected_model_base_url" ]]; then
    echo "model_endpoint_profile_incomplete:$profile" >&2
    exit 2
  fi
  export OPENAI_API_KEY="$selected_model_api_key"
  export OPENAI_BASE_URL="$selected_model_base_url"
  export OMNIFLOW_MODEL_ENDPOINT_PROFILE="$profile"
}

validate_model_endpoint_auth() {
  if [[ "$dry_run" -eq 1 || "$check_only" -eq 1 ]]; then
    return 0
  fi
  local probe_status
  if ! probe_status="$(
    MODEL_ENDPOINT_API_KEY="$selected_model_api_key" \
    MODEL_ENDPOINT_BASE_URL="$selected_model_base_url" \
      "$python_bin" - <<'PY'
import os
import urllib.error
import urllib.request

base_url = os.environ["MODEL_ENDPOINT_BASE_URL"].rstrip("/")
request = urllib.request.Request(
    f"{base_url}/models",
    headers={"Authorization": f"Bearer {os.environ['MODEL_ENDPOINT_API_KEY']}"},
    method="GET",
)
try:
    with urllib.request.urlopen(request, timeout=10) as response:
        status = int(response.status)
except urllib.error.HTTPError as error:
    status = int(error.code)
except Exception as error:
    print(f"unavailable:{error}")
    raise SystemExit(1)
print(status)
raise SystemExit(0 if 200 <= status < 300 else 1)
PY
  )"; then
    echo "model_endpoint_auth_failed:profile=$formal_model_endpoint_profile status=$probe_status" >&2
    exit 2
  fi
}

validate_experiment_model() {
  local model="$1"
  local profile="$2"
  local normalized_model
  normalized_model="$(printf '%s' "$model" | tr '[:upper:]' '[:lower:]')"
  if [[ "$normalized_model" == "qwen3-vl-plus" ]]; then
    echo "qwen3-vl-plus is prohibited for AndroidWorld experiments." >&2
    exit 2
  fi
  if [[ "$normalized_model" != "glm-5.1" ]]; then
    echo "AndroidWorld experiments require GLM-5.1, got: $model" >&2
    exit 2
  fi
  if [[ "$profile" != "$formal_model_endpoint_profile" ]]; then
    echo "GLM-5.1 requires model endpoint profile $formal_model_endpoint_profile, got: $profile" >&2
    exit 2
  fi
  if [[ "$selected_model_base_url" != "$formal_model_base_url" ]]; then
    echo "GLM-5.1 requires model endpoint $formal_model_base_url, got: $selected_model_base_url" >&2
    exit 2
  fi
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/exp/run_androidworld.sh [OPTIONS]

Options:
  --environment NAME        Select androidworld (default) or bmoca. This changes
                            only the official environment, never the method.
  --check-only              Validate the complete selected run without creating
                            assets, attempts, result directories, or emulators.
  --development-run         Run one unregistered `ours` episode through this
                            script for bounded method debugging.
  --dry-run                 Build one task command without executing it.
  --all-tasks               Run the selected task set in task-major order.
  --method METHOD           Run one method in the single-result runner.
  --device LABEL:SERIAL:PORT
                            Run one target in the single-result runner.
  --tasks TASK1,TASK2,...   Select an ordered task-major subset, or scope
                            --convert-ours-assets. Implies --all-tasks during
                            experiment execution.
  --convert-ours-assets     Validate and freeze Function bundles from an
                            immutable skill manifest, then
                            freeze, and register the assets.
  --convert-source-runlogs  Convert the indexed legacy source RunLogs once to
                            omniflow.run_log.v1.
  --prepare-mobilegpt-memory
                            Build task-local MobileGPT memory from canonical
                            RunLogs only. With --check-only, run zero-model
                            preflight and create nothing.
  --convert-runlog-memory METHOD
                            Convert one successful RunLog into native
                            mobilegpt_offline_retrieval or appagent_demo memory.
  --e2e-task TASK           Run one bounded source-to-matrix task pipeline.
  --source-qualification-only
                            Stop that pipeline after immutable seed-111 Function
                            qualification; create no target result results.
  --collect-source         Re-run one task on the source device only and save
                            screenshot-backed native RunLog evidence.
  --task-deadline-sec SEC   Whole-task wall deadline; maximum/default is 1800.
  --refresh-memory          Deduplicate and index all configured RunLogs,
                            method assets, and existing results.
  -h, --help                Show this help and exit.

Required external roots:
  OMNIFLOW_EXP_ASSET_ROOT   Absolute root containing frozen experiment assets.
  OMNIFLOW_EXP_RESULTS_ROOT Absolute root for immutable results.
  OMNIFLOW_EXP_MEMORY_ROOT  Absolute content-addressed long-term-memory root.
  OMNITRANSFER_ROOT         Canonical/versioned OmniTransfer checkout.

Optional runtime overrides:
  PYTHON_BIN, OMNIFLOW_ENV_FILE, OMNIFLOW_ANDROIDWORLD_SOURCE_INDEX,
  OMNIFLOW_MASTER_SOURCE_INDEX, OMNIFLOW_OURS_STORE_INDEX,
  OMNIFLOW_MEMORY_MOBILEGPT_ROOTS,
  OMNIFLOW_MEMORY_BASELINE_BATCH_REPORTS,
  OMNIFLOW_ANDROID_SDK_ROOT, OMNIFLOW_JAVA_HOME,
  OMNIFLOW_MOBILEGPT_SOURCE_ENVIRONMENT_REPAIR_REASON,
  OMNIFLOW_APPAGENT_SOURCE_ENVIRONMENT_REPAIR_REASON,
  OMNIFLOW_DEVELOPMENT_OUTPUT_PATH, OMNIFLOW_DEVELOPMENT_MODEL,
  OMNIFLOW_DEVELOPMENT_MODEL_ENDPOINT_PROFILE (default: llmthu),
  OMNIFLOW_DEVELOPMENT_CONSOLE_PORT,
  OMNIFLOW_DEVELOPMENT_AVD (default: OmniFlowTargetSmall),
  OMNIFLOW_ANDROIDWORLD_LLM_MAX_TOKENS (online T3A response budget),
  OMNIFLOW_ANDROIDWORLD_PERFORM_EMULATOR_SETUP (0 reuses prior app snapshots),
  OMNIFLOW_BATCH_ATTEMPT_ID (resume one interrupted immutable batch).
  OMNIFLOW_E2E_OUTPUT_ROOT, OMNIFLOW_E2E_ATTEMPT_ID.
  OMNIFLOW_ADB_PATH.
  Managed emulators are cold-restarted before every pending result.

Asset conversion inputs:
  OMNIFLOW_OURS_SOURCE_ASSET_INDEX Source RunLog index; defaults to the master
                                   source index.
  OMNIFLOW_OURS_AUTHORING_MANIFEST Immutable Function bundle skill manifest.
  OMNIFLOW_OURS_CONVERTED_ASSET_ROOT New immutable conversion output root.
  OMNIFLOW_OURS_REVISION_REASON      Non-empty reason that replaces an existing
                                     canonical Store using the supplied manifest.
  OMNIFLOW_EXP_MEMORY_INDEX          Existing memory current.json.

Long-term-memory refresh inputs:
  OMNIFLOW_MEMORY_RUNLOG_ROOTS       Colon-separated evidence roots.
  OMNIFLOW_MEMORY_RESULT_ROOTS       Colon-separated result roots.
  OMNIFLOW_MEMORY_FUNCTION_CATALOGS  Colon-separated Function catalogs.
  OMNIFLOW_MEMORY_BASELINE_BATCH_REPORTS
                                     Colon-separated immutable batch summaries
                                     whose validator results must remain frozen.
  OMNIFLOW_SOURCE_SELECTION_MANIFEST Optional audited exact-SHA source repairs.
  OMNIFLOW_FUNCTION_STORE_SELECTION_MANIFEST
                                     Optional audited exact-SHA Function Store
                                     conflict selection.
  OMNIFLOW_SOURCE_SCREENSHOT_ROOTS   Optional screenshot roots for legacy repairs.

Source RunLog conversion inputs:
  OMNIFLOW_SOURCE_RUNLOG_OUTPUT_ROOT Absolute immutable output root.
  OMNIFLOW_SOURCE_SCREENSHOT_ROOTS   Optional colon-separated screenshot roots.
  OMNIFLOW_MOBILEGPT_MEMORY_OUTPUT_ROOT
                                     Absolute immutable batch-attempt root.
  OMNIFLOW_RUNLOG_MEMORY_OUTPUT_ROOT
                                     Absolute immutable output for one native
                                     baseline-memory conversion.
  --source-runlog PATH              Input RunLog for --convert-runlog-memory.
  OMNIFLOW_BMOCA_ROOT, OMNIFLOW_BMOCA_ENVIRONMENT_IDS (default: 100..109),
  OMNIFLOW_BMOCA_AVD_HOME, OMNIFLOW_BMOCA_AVD_TEMPLATE_HOME,
  OMNIFLOW_BMOCA_OUTPUT_PATH, OMNIFLOW_BMOCA_SHOW_EMULATOR.

Examples:
  bash scripts/exp/run_androidworld.sh --tasks AudioRecorderRecordAudio
  bash scripts/exp/run_androidworld.sh --convert-ours-assets \
    --tasks AudioRecorderRecordAudio
  bash scripts/exp/run_androidworld.sh --refresh-memory
  bash scripts/exp/run_androidworld.sh --convert-source-runlogs
  bash scripts/exp/run_androidworld.sh --check-only \
    --prepare-mobilegpt-memory
  bash scripts/exp/run_androidworld.sh --prepare-mobilegpt-memory \
    --tasks ContactsAddContact
  OMNIFLOW_RUNLOG_MEMORY_OUTPUT_ROOT=/abs/new-memory \
    bash scripts/exp/run_androidworld.sh \
      --convert-runlog-memory mobilegpt_offline_retrieval \
      --source-runlog /abs/success.run_log.json
  bash scripts/exp/run_androidworld.sh --check-only --all-tasks
  bash scripts/exp/run_androidworld.sh --all-tasks \
    --tasks AudioRecorderRecordAudioWithFileName,SystemCopyToClipboard
  bash scripts/exp/run_androidworld.sh \
    --e2e-task AudioRecorderRecordAudioWithFileName \
    --task-deadline-sec 1800
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --check-only)
      check_only=1
      ;;
    --environment)
      shift
      if [[ "$#" -eq 0 || ( "$1" != "androidworld" && "$1" != "bmoca" ) ]]; then
        echo "--environment requires androidworld or bmoca." >&2
        exit 2
      fi
      execution_environment="$1"
      ;;
    --development-run)
      development_run=1
      ;;
    --dry-run)
      dry_run=1
      ;;
    --all-tasks)
      all_tasks=1
      ;;
    --method)
      shift
      if [[ "$#" -eq 0 || -z "$1" ]]; then
        echo "--method requires one paper method." >&2
        exit 2
      fi
      selected_method_arg="$1"
      ;;
    --device)
      shift
      if [[ "$#" -eq 0 || -z "$1" ]]; then
        echo "--device requires LABEL:SERIAL:PORT." >&2
        exit 2
      fi
      selected_device_arg="$1"
      ;;
    --convert-ours-assets)
      convert_ours_assets=1
      ;;
    --refresh-memory)
      refresh_memory=1
      ;;
    --convert-source-runlogs)
      convert_source_runlogs=1
      ;;
    --prepare-mobilegpt-memory)
      prepare_mobilegpt_memory=1
      ;;
    --convert-runlog-memory)
      shift
      if [[ "$#" -eq 0 || ( "$1" != "mobilegpt_offline_retrieval" && "$1" != "appagent_demo" ) ]]; then
        echo "--convert-runlog-memory requires mobilegpt_offline_retrieval or appagent_demo." >&2
        exit 2
      fi
      convert_runlog_memory_method="$1"
      ;;
    --e2e-task)
      shift
      if [[ "$#" -eq 0 || -z "$1" ]]; then
        echo "--e2e-task requires one AndroidWorld task name." >&2
        exit 2
      fi
      e2e_task="$1"
      ;;
    --source-qualification-only)
      source_qualification_only=1
      ;;
    --collect-source)
      source_collection=1
      ;;
    --task-deadline-sec)
      shift
      if [[ "$#" -eq 0 || -z "$1" ]]; then
        echo "--task-deadline-sec requires a positive integer no greater than 1800." >&2
        exit 2
      fi
      e2e_task_deadline_sec="$1"
      ;;
    --source-runlog)
      shift
      if [[ "$#" -eq 0 || -z "$1" ]]; then
        echo "--source-runlog requires an absolute file path." >&2
        exit 2
      fi
      runlog_memory_source_runlog="$1"
      ;;
    --tasks)
      shift
      if [[ "$#" -eq 0 || -z "$1" ]]; then
        echo "--tasks requires a comma-separated task list." >&2
        exit 2
      fi
      batch_task_filter="$1"
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
  shift
done
if [[ -n "$selected_method_arg" || -n "$selected_device_arg" ]] && {
  [[ "$development_run" -eq 1 || "$source_collection" -eq 1 ||
    "$all_tasks" -eq 1 || -n "$e2e_task" || -n "$batch_task_filter" ||
    "$convert_ours_assets" -eq 1 || "$refresh_memory" -eq 1 ||
    "$convert_source_runlogs" -eq 1 || "$prepare_mobilegpt_memory" -eq 1 ||
    -n "$convert_runlog_memory_method" ]];
}; then
  echo "--method/--device are only valid for one direct AndroidWorld result." >&2
  exit 2
fi
if [[ "$execution_environment" != "bmoca" && "$source_collection" -eq 1 ]]; then
  if [[ -z "$batch_task_filter" || "$batch_task_filter" == *,* ]]; then
    echo "--collect-source requires exactly one task through --tasks." >&2
    exit 2
  fi
  e2e_task="$batch_task_filter"
  batch_task_filter=""
  source_qualification_only=0
fi
if [[ "$execution_environment" == "bmoca" ]]; then
  if [[ "$source_collection" -eq 1 || "$development_run" -eq 1 || "$check_only" -eq 1 || "$dry_run" -eq 1 || "$all_tasks" -eq 1 || -n "$e2e_task" || -n "$selected_method_arg" || -n "$selected_device_arg" ]]; then
    echo "--environment bmoca is one native OmniFlow E2E run and cannot be combined with AndroidWorld experiment modes." >&2
    exit 2
  fi
  if [[ -z "$batch_task_filter" || "$batch_task_filter" == *,* ]]; then
    echo "--environment bmoca requires exactly one task through --tasks." >&2
    exit 2
  fi
  if [[ -z "$store_path" || "$store_path" != /* || ! -f "$store_path" ]]; then
    echo "--environment bmoca requires an existing absolute OMNIFLOW_ANDROIDWORLD_STORE_PATH." >&2
    exit 2
  fi
  if [[ -z "$bmoca_root" || "$bmoca_root" != /* || ! -d "$bmoca_root/asset" ]]; then
    echo "--environment bmoca requires an absolute OMNIFLOW_BMOCA_ROOT." >&2
    exit 2
  fi
  bmoca_actual_revision="$(git -C "$bmoca_root" rev-parse HEAD 2>/dev/null || true)"
  if [[ "$bmoca_actual_revision" != "$formal_bmoca_revision" ]]; then
    echo "B-MoCA revision mismatch: expected=$formal_bmoca_revision actual=${bmoca_actual_revision:-missing}" >&2
    exit 2
  fi
  if [[ -z "$bmoca_avd_home" || "$bmoca_avd_home" != /* || ! -d "$bmoca_avd_home" ]]; then
    echo "--environment bmoca requires an absolute OMNIFLOW_BMOCA_AVD_HOME." >&2
    exit 2
  fi
  if [[ -n "$bmoca_avd_template_home" && ( "$bmoca_avd_template_home" != /* || ! -d "$bmoca_avd_template_home" ) ]]; then
    echo "OMNIFLOW_BMOCA_AVD_TEMPLATE_HOME must be an existing absolute directory." >&2
    exit 2
  fi
  if [[ -z "$bmoca_output_path" || "$bmoca_output_path" != /* || -e "$bmoca_output_path" ]]; then
    echo "--environment bmoca requires a new absolute OMNIFLOW_BMOCA_OUTPUT_PATH." >&2
    exit 2
  fi
  if [[ "$bmoca_show_emulator" != "0" && "$bmoca_show_emulator" != "1" ]]; then
    echo "OMNIFLOW_BMOCA_SHOW_EMULATOR must be 0 or 1." >&2
    exit 2
  fi
  if [[ -z "$env_file" || "$env_file" != /* || ! -f "$env_file" ]]; then
    echo "--environment bmoca requires an existing absolute OMNIFLOW_ENV_FILE." >&2
    exit 2
  fi
  if ! python_bin="$(command -v "$python_bin")"; then
    echo "Python runtime missing: ${PYTHON_BIN:-python3}" >&2
    exit 1
  fi
  bmoca_android_sdk_root="${OMNIFLOW_ANDROID_SDK_ROOT:-${ANDROID_SDK_ROOT:-${ANDROID_HOME:-$(resolve_default_android_sdk_root)}}}"
  if [[ "$bmoca_android_sdk_root" != /* || ! -x "$bmoca_android_sdk_root/platform-tools/adb" || ! -x "$bmoca_android_sdk_root/emulator/emulator" ]]; then
    echo "--environment bmoca requires a complete absolute Android SDK root: $bmoca_android_sdk_root" >&2
    exit 2
  fi
  set -a
  source "$env_file"
  set +a
  select_model_endpoint "$formal_model_endpoint_profile"
  validate_experiment_model "$formal_model" "$formal_model_endpoint_profile"
  export OMNIFLOW_ANDROIDWORLD_MAX_FALLBACK_STEPS=0
  bmoca_command=(
    "$python_bin" -m src.integrations.android_world.launch
    --environment bmoca
    --bmoca-root "$bmoca_root"
    --environment-ids "$bmoca_environment_ids"
    --android-sdk-root "$bmoca_android_sdk_root"
    --android-avd-home "$bmoca_avd_home"
    --tasks "$batch_task_filter"
    --agent omniflow
    --store-path "$store_path"
    --output-path "$bmoca_output_path"
    --model "$formal_model"
    --model-endpoint-profile "$formal_model_endpoint_profile"
    --planner-provider openai_compatible
    --planner-timeout-sec "${OMNIFLOW_BMOCA_PLANNER_TIMEOUT_SEC:-60}"
  )
  if [[ -n "$bmoca_avd_template_home" ]]; then
    bmoca_command+=(--bmoca-avd-template-home "$bmoca_avd_template_home")
  fi
  if [[ "$bmoca_show_emulator" == "1" ]]; then
    bmoca_command+=(--show-emulator)
  fi
  cd "$repo"
  exec "${bmoca_command[@]}"
fi
if [[ -n "$convert_runlog_memory_method" ]]; then
  if [[ "$convert_source_runlogs" -eq 1 || "$refresh_memory" -eq 1 || "$convert_ours_assets" -eq 1 || "$prepare_mobilegpt_memory" -eq 1 || "$development_run" -eq 1 || "$check_only" -eq 1 || "$dry_run" -eq 1 || "$all_tasks" -eq 1 || -n "$e2e_task" || -n "$batch_task_filter" ]]; then
    echo "--convert-runlog-memory cannot be combined with another experiment mode." >&2
    exit 2
  fi
  if [[ -z "$runlog_memory_source_runlog" || "$runlog_memory_source_runlog" != /* || ! -f "$runlog_memory_source_runlog" ]]; then
    echo "--convert-runlog-memory requires an existing absolute --source-runlog." >&2
    exit 2
  fi
  if [[ -z "$runlog_memory_output_root" || "$runlog_memory_output_root" != /* ]]; then
    echo "--convert-runlog-memory requires absolute OMNIFLOW_RUNLOG_MEMORY_OUTPUT_ROOT." >&2
    exit 2
  fi
  if [[ -e "$runlog_memory_output_root" ]]; then
    echo "Immutable RunLog memory output already exists: $runlog_memory_output_root" >&2
    exit 2
  fi
  if ! python_bin="$(command -v "$python_bin")"; then
    echo "Python runtime missing: ${PYTHON_BIN:-python3}" >&2
    exit 1
  fi
  if [[ -z "$env_file" || "$env_file" != /* || ! -f "$env_file" ]]; then
    echo "RunLog memory conversion requires an existing absolute OMNIFLOW_ENV_FILE." >&2
    exit 2
  fi
  case "$convert_runlog_memory_method" in
    mobilegpt_offline_retrieval)
      runlog_memory_upstream_root="$mobilegpt_root"
      runlog_memory_model="$formal_model"
      runlog_memory_embedding_model="text-embedding-v4"
      ;;
    appagent_demo)
      runlog_memory_upstream_root="$appagent_root"
      runlog_memory_model="$formal_model"
      runlog_memory_embedding_model=""
      ;;
  esac
  if [[ "$runlog_memory_upstream_root" != /* || ! -d "$runlog_memory_upstream_root" ]]; then
    echo "Native baseline root missing: $runlog_memory_upstream_root" >&2
    exit 2
  fi
  set -a
  source "$env_file"
  set +a
  if [[ "$convert_runlog_memory_method" == "mobilegpt_offline_retrieval" ]]; then
    if [[ -z "${OPENAI_API_KEY:-}" || -z "${OPENAI_BASE_URL:-}" ]]; then
      echo "MobileGPT conversion requires OPENAI_API_KEY/OPENAI_BASE_URL for its embedding model." >&2
      exit 2
    fi
    export MOBILEGPT_EMBEDDING_MODEL="$runlog_memory_embedding_model"
  elif [[ -z "$runlog_memory_model" ]]; then
    echo "AppAgent document model is required." >&2
    exit 2
  else
    select_model_endpoint "$formal_model_endpoint_profile"
    validate_experiment_model "$runlog_memory_model" "$formal_model_endpoint_profile"
  fi
  cd "$repo"
  "$python_bin" - \
    "$convert_runlog_memory_method" \
    "$runlog_memory_source_runlog" \
    "$runlog_memory_output_root" \
    "$runlog_memory_upstream_root" \
    "$runlog_memory_model" \
    "$runlog_memory_embedding_model" <<'PY'
import json
import sys

from src.experiment.source_assets import convert_runlog_memory

result = convert_runlog_memory(
    sys.argv[1],
    source_run_log=sys.argv[2],
    output_root=sys.argv[3],
    upstream_root=sys.argv[4],
    model=sys.argv[5],
    embedding_model=sys.argv[6],
)
print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
PY
  exit 0
fi
if [[ "$development_run" -eq 1 ]]; then
  if [[ "$convert_source_runlogs" -eq 1 || "$refresh_memory" -eq 1 || "$convert_ours_assets" -eq 1 || "$prepare_mobilegpt_memory" -eq 1 || "$check_only" -eq 1 || "$all_tasks" -eq 1 || -n "$e2e_task" ]]; then
    echo "--development-run cannot be combined with maintenance, formal matrix, or E2E options." >&2
    exit 2
  fi
  if [[ -n "$batch_task_filter" ]]; then
    if [[ "$batch_task_filter" == *,* ]]; then
      echo "--development-run accepts exactly one task." >&2
      exit 2
    fi
    task="$batch_task_filter"
  fi
  development_output_path="${OMNIFLOW_DEVELOPMENT_OUTPUT_PATH:-}"
  development_model="${OMNIFLOW_DEVELOPMENT_MODEL:-}"
  development_model_endpoint_profile="${OMNIFLOW_DEVELOPMENT_MODEL_ENDPOINT_PROFILE:-llmthu}"
  development_console_port="${OMNIFLOW_DEVELOPMENT_CONSOLE_PORT:-5554}"
  development_avd="${OMNIFLOW_DEVELOPMENT_AVD:-OmniFlowTargetSmall}"
  development_perform_setup="${OMNIFLOW_ANDROIDWORLD_PERFORM_EMULATOR_SETUP:-1}"
  development_planner_timeout="${OMNIFLOW_PLANNER_TIMEOUT_SEC:-60}"
  development_runtime_files=(
    "src/experiment/development_emulator.py"
    "src/integrations/android_world/launch.py"
  )
  missing_development_runtime_files=()
  for relative_path in "${development_runtime_files[@]}"; do
    if [[ ! -f "$repo/$relative_path" ]]; then
      missing_development_runtime_files+=("$relative_path")
    fi
  done
  if [[ ${#missing_development_runtime_files[@]} -gt 0 ]]; then
    echo "Development runtime deployment incomplete before device startup:" >&2
    printf '  - %s\n' "${missing_development_runtime_files[@]}" >&2
    exit 1
  fi
  if [[ -z "$task" || "$task" == *,* ]]; then
    echo "--development-run requires exactly one task through --tasks or OMNIFLOW_ANDROIDWORLD_TASK." >&2
    exit 2
  fi
  if [[ -z "$store_path" || "$store_path" != /* || ! -f "$store_path" ]]; then
    echo "--development-run requires an existing absolute OMNIFLOW_ANDROIDWORLD_STORE_PATH." >&2
    exit 2
  fi
  if [[ -z "$development_output_path" || "$development_output_path" != /* ]]; then
    echo "--development-run requires an absolute OMNIFLOW_DEVELOPMENT_OUTPUT_PATH." >&2
    exit 2
  fi
  if [[ -e "$development_output_path" ]]; then
    echo "Development attempt already exists: $development_output_path" >&2
    exit 1
  fi
  if [[ -z "$development_model" ]]; then
    echo "--development-run requires OMNIFLOW_DEVELOPMENT_MODEL." >&2
    exit 2
  fi
  if [[ ! "$development_console_port" =~ ^[0-9]+$ ]]; then
    echo "OMNIFLOW_DEVELOPMENT_CONSOLE_PORT must be a non-negative integer." >&2
    exit 2
  fi
  if [[ ! "$development_perform_setup" =~ ^[01]$ ]]; then
    echo "OMNIFLOW_ANDROIDWORLD_PERFORM_EMULATOR_SETUP must be 0 or 1." >&2
    exit 2
  fi
  if [[ -z "$android_world_root" || "$android_world_root" != /* || ! -d "$android_world_root/android_world" ]]; then
    echo "--development-run requires an absolute AndroidWorld checkout." >&2
    exit 2
  fi
  development_android_sdk_root="${OMNIFLOW_ANDROID_SDK_ROOT:-${ANDROID_SDK_ROOT:-${ANDROID_HOME:-$(resolve_default_android_sdk_root)}}}"
  development_adb_path="${OMNIFLOW_ADB_PATH:-$development_android_sdk_root/platform-tools/adb}"
  development_emulator_bin="${OMNIFLOW_EMULATOR_BIN:-$development_android_sdk_root/emulator/emulator}"
  if [[ "$development_adb_path" != /* || ! -x "$development_adb_path" ]]; then
    echo "--development-run requires an executable absolute ADB path: $development_adb_path" >&2
    exit 2
  fi
  if [[ "$development_emulator_bin" != /* || ! -x "$development_emulator_bin" ]]; then
    echo "--development-run requires an executable absolute emulator path: $development_emulator_bin" >&2
    exit 2
  fi
  if [[ -z "$env_file" || "$env_file" != /* || ! -f "$env_file" ]]; then
    echo "--development-run requires an existing absolute OMNIFLOW_ENV_FILE." >&2
    exit 2
  fi
  set -a
  source "$env_file"
  set +a
  select_model_endpoint "$development_model_endpoint_profile"
  validate_experiment_model "$development_model" "$development_model_endpoint_profile"
  echo "[model] model=$development_model model_endpoint_profile=$development_model_endpoint_profile model_endpoint=$selected_model_base_url"
  development_command=(
    "$python_bin" -m src.integrations.android_world.launch
    --android-world-root "$android_world_root"
    --tasks "$task"
    --task-random-seed "$task_seed"
    --n-task-combinations 1
    --console-port "$development_console_port"
    --agent omniflow
    --max-steps "$max_steps"
    --output-path "$development_output_path"
    --fixed-task-seed
    --store-path "$store_path"
    --planner-provider openai
    --model "$development_model"
    --model-endpoint-profile "$development_model_endpoint_profile"
    --planner-timeout-sec "$development_planner_timeout"
    --adb-path "$development_adb_path"
  )
  if [[ "$development_perform_setup" -eq 1 ]]; then
    development_command+=(--perform-emulator-setup)
  fi
  if [[ "$dry_run" -eq 1 ]]; then
    printf '%q ' "${development_command[@]}"
    printf '\n'
    exit 0
  fi
  cd "$repo"
  "$python_bin" -m src.experiment.development_emulator \
    --adb "$development_adb_path" \
    --emulator "$development_emulator_bin" \
    --serial "emulator-$development_console_port" \
    --avd "$development_avd" \
    --gpu "$emulator_gpu" \
    --log-path "${development_output_path}.emulator.log" \
    --boot-timeout "$emulator_boot_timeout_sec"
  exec "${development_command[@]}"
fi
if [[ -n "$e2e_task" ]]; then
  if [[ "$convert_source_runlogs" -eq 1 || "$refresh_memory" -eq 1 || "$convert_ours_assets" -eq 1 || "$prepare_mobilegpt_memory" -eq 1 || "$check_only" -eq 1 || "$all_tasks" -eq 1 || -n "$batch_task_filter" ]]; then
    echo "--e2e-task cannot be combined with maintenance or matrix-selection options." >&2
    exit 2
  fi
  if [[ ! "$e2e_task_deadline_sec" =~ ^[1-9][0-9]*$ ]] || (( e2e_task_deadline_sec > 1800 )); then
    echo "--task-deadline-sec must be a positive integer no greater than 1800." >&2
    exit 2
  fi
  if [[ -z "$asset_root" || "$asset_root" != /* || -z "$results_root" || "$results_root" != /* ]]; then
    echo "--e2e-task requires absolute OMNIFLOW_EXP_ASSET_ROOT and OMNIFLOW_EXP_RESULTS_ROOT." >&2
    exit 2
  fi
  if [[ -z "$memory_index" || "$memory_index" != /* || ! -f "$memory_index" ]]; then
    echo "--e2e-task requires an existing absolute OMNIFLOW_EXP_MEMORY_INDEX." >&2
    exit 2
  fi
  if [[ -z "$android_world_root" || "$android_world_root" != /* || ! -d "$android_world_root/android_world" ]]; then
    echo "--e2e-task requires an absolute AndroidWorld checkout." >&2
    exit 2
  fi
  canonical_omnitransfer_root="$account_root/Projects/Omni/OmniTransfer"
  if [[ ! -d "$canonical_omnitransfer_root" || -z "$omnitransfer_root" || ! -d "$omnitransfer_root" ]]; then
    echo "--e2e-task requires OMNITRANSFER_ROOT=$canonical_omnitransfer_root." >&2
    exit 2
  fi
  resolved_omnitransfer_root="$(cd "$omnitransfer_root" && pwd -P)"
  if [[ "$resolved_omnitransfer_root" != "$(cd "$canonical_omnitransfer_root" && pwd -P)" ]]; then
    echo "--e2e-task requires canonical OmniTransfer: $canonical_omnitransfer_root." >&2
    exit 2
  fi
  if [[ -z "$mobilegpt_root" || "$mobilegpt_root" != /* || ! -d "$mobilegpt_root" ]]; then
    echo "--e2e-task requires an absolute native MobileGPT root." >&2
    exit 2
  fi
  if [[ -z "$appagent_root" || "$appagent_root" != /* || ! -d "$appagent_root" ]]; then
    echo "--e2e-task requires an absolute native AppAgent root." >&2
    exit 2
  fi
  if ! python_bin="$(command -v "$python_bin")"; then
    echo "Python runtime missing: ${PYTHON_BIN:-python3}" >&2
    exit 1
  fi
  e2e_android_sdk_root="${OMNIFLOW_ANDROID_SDK_ROOT:-${ANDROID_SDK_ROOT:-${ANDROID_HOME:-$(resolve_default_android_sdk_root)}}}"
  e2e_adb_path="${OMNIFLOW_ADB_PATH:-$e2e_android_sdk_root/platform-tools/adb}"
  e2e_emulator_bin="${OMNIFLOW_EMULATOR_BIN:-$e2e_android_sdk_root/emulator/emulator}"
  if [[ "$e2e_adb_path" != /* || ! -x "$e2e_adb_path" ]]; then
    echo "--e2e-task requires an executable absolute ADB path: $e2e_adb_path" >&2
    exit 2
  fi
  if [[ "$e2e_emulator_bin" != /* || ! -x "$e2e_emulator_bin" ]]; then
    echo "--e2e-task requires an executable absolute emulator path: $e2e_emulator_bin" >&2
    exit 2
  fi
  if [[ -z "$env_file" || "$env_file" != /* || ! -f "$env_file" ]]; then
    echo "--e2e-task requires an existing absolute OMNIFLOW_ENV_FILE." >&2
    exit 2
  fi
  set -a
  source "$env_file"
  set +a
  normalized_e2e_model="$(printf '%s' "$formal_model" | tr '[:upper:]' '[:lower:]')"
  if [[ "$normalized_e2e_model" != "glm-5.1" ]]; then
    echo "AndroidWorld E2E requires GLM-5.1 for the formal model, got: $formal_model" >&2
    exit 2
  fi
  e2e_output_root="${OMNIFLOW_E2E_OUTPUT_ROOT:-$results_root/androidworld_e2e_task_attempts}"
  if [[ "$e2e_output_root" != /* ]]; then
    echo "OMNIFLOW_E2E_OUTPUT_ROOT must be absolute." >&2
    exit 2
  fi
  e2e_args=(
    -m src.experiment.e2e_task_pipeline
    --repo "$repo"
    --script "$repo/scripts/exp/run_androidworld.sh"
    --task "$e2e_task"
    --task-deadline-sec "$e2e_task_deadline_sec"
    --max-steps "$max_steps"
    --max-fallback-steps "$max_fallback_steps"
    --memory-index "$memory_index"
    --asset-root "$asset_root"
    --results-root "$results_root"
    --output-root "$e2e_output_root"
    --android-world-root "$android_world_root"
    --omnitransfer-root "$resolved_omnitransfer_root"
    --mobilegpt-root "$mobilegpt_root"
    --appagent-root "$appagent_root"
    --python-bin "$python_bin"
    --adb-path "$e2e_adb_path"
    --emulator-bin "$e2e_emulator_bin"
    --source-device "$source_device"
    --source-avd "SmallPhone"
    --emulator-gpu "$emulator_gpu"
    --runtime-preflight "$repo/src/experiment/preflight.py"
    --formal-model "$formal_model"
  )
  if [[ -n "${OMNIFLOW_E2E_ATTEMPT_ID:-}" ]]; then
    e2e_args+=(--attempt-id "$OMNIFLOW_E2E_ATTEMPT_ID")
  fi
  if [[ "$source_qualification_only" -eq 1 ]]; then
    e2e_args+=(--source-qualification-only)
  fi
  if [[ "$source_collection" -eq 1 ]]; then
    e2e_args+=(--source-only)
  fi
  if [[ -n "$appagent_demo_memory_root" ]]; then
    e2e_args+=(--appagent-memory-root "$appagent_demo_memory_root")
  fi
  if [[ "$dry_run" -eq 1 ]]; then
    e2e_args+=(--dry-run)
  fi
  # Source-only E2E collection exits through this branch before the formal
  # matrix path reaches the shared AndroidWorld capability gate.  Keep the
  # same FTS4 contract for every AndroidWorld execution entry.
  ensure_androidworld_sqlite_fts4
  cd "$repo"
  exec "$python_bin" "${e2e_args[@]}"
fi
if [[ "$convert_source_runlogs" -eq 1 ]]; then
  if [[ "$refresh_memory" -eq 1 || "$convert_ours_assets" -eq 1 || "$prepare_mobilegpt_memory" -eq 1 || "$check_only" -eq 1 || "$dry_run" -eq 1 || "$all_tasks" -eq 1 ]]; then
    echo "--convert-source-runlogs cannot be combined with experiment or other maintenance options." >&2
    exit 2
  fi
  if [[ -z "$master_source_index" || "$master_source_index" != /* || ! -f "$master_source_index" ]]; then
    echo "Source RunLog conversion requires an existing absolute master source index." >&2
    exit 2
  fi
  if [[ -z "$source_runlog_output_root" || "$source_runlog_output_root" != /* ]]; then
    echo "OMNIFLOW_SOURCE_RUNLOG_OUTPUT_ROOT must be absolute." >&2
    exit 2
  fi
  if ! python_bin="$(command -v "$python_bin")"; then
    echo "Python runtime missing: ${PYTHON_BIN:-python3}" >&2
    exit 1
  fi
  source_conversion_args=(
    -m src.experiment.source_runlogs
    --source-index "$master_source_index"
    --output-root "$source_runlog_output_root"
  )
  if [[ -n "$source_screenshot_roots" ]]; then
    IFS=':' read -r -a configured_screenshot_roots <<< "$source_screenshot_roots"
    for configured_root in "${configured_screenshot_roots[@]}"; do
      if [[ "$configured_root" != /* || ! -d "$configured_root" ]]; then
        echo "Screenshot root must be an existing absolute directory: $configured_root" >&2
        exit 2
      fi
      source_conversion_args+=(--screenshot-root "$configured_root")
    done
  fi
  if [[ -n "$batch_task_filter" ]]; then
    IFS=',' read -r -a conversion_tasks <<< "$batch_task_filter"
    for conversion_task in "${conversion_tasks[@]}"; do
      source_conversion_args+=(--task "$conversion_task")
    done
  fi
  cd "$repo"
  exec "$python_bin" "${source_conversion_args[@]}"
fi
if [[ "$refresh_memory" -eq 1 ]]; then
  if [[ "$convert_ours_assets" -eq 1 || "$prepare_mobilegpt_memory" -eq 1 || "$check_only" -eq 1 || "$dry_run" -eq 1 || "$all_tasks" -eq 1 || -n "$batch_task_filter" ]]; then
    echo "--refresh-memory cannot be combined with conversion or experiment run options." >&2
    exit 2
  fi
  if [[ -z "$memory_root" || "$memory_root" != /* ]]; then
    echo "Set OMNIFLOW_EXP_MEMORY_ROOT to an absolute path." >&2
    exit 2
  fi
  memory_pointer="$memory_root/current.json"
  if [[ ! -f "$memory_pointer" && ! -f "$master_source_index" ]]; then
    echo "Canonical master source index missing: $master_source_index" >&2
    exit 2
  fi
  if [[ -z "$memory_runlog_roots" ]]; then
    echo "OMNIFLOW_MEMORY_RUNLOG_ROOTS must contain at least one root." >&2
    exit 2
  fi
  if ! python_bin="$(command -v "$python_bin")"; then
    echo "Python runtime missing: ${PYTHON_BIN:-python3}" >&2
    exit 1
  fi
  memory_args=(
    -m src.experiment.artifact_memory
    refresh
    --memory-root "$memory_root"
  )
  if [[ ! -f "$memory_pointer" ]]; then
    memory_args+=(--source-index "$master_source_index")
  fi
  if [[ -n "$source_selection_manifest" ]]; then
    if [[ "$source_selection_manifest" != /* || ! -f "$source_selection_manifest" ]]; then
      echo "Source selection manifest must be an existing absolute file: $source_selection_manifest" >&2
      exit 2
    fi
    memory_args+=(--source-selection-manifest "$source_selection_manifest")
  fi
  if [[ -n "$function_store_selection_manifest" ]]; then
    if [[ "$function_store_selection_manifest" != /* || ! -f "$function_store_selection_manifest" ]]; then
      echo "Function Store selection manifest must be an existing absolute file: $function_store_selection_manifest" >&2
      exit 2
    fi
    memory_args+=(--function-store-selection-manifest "$function_store_selection_manifest")
  fi
  if [[ -n "$source_screenshot_roots" ]]; then
    IFS=':' read -r -a configured_screenshot_roots <<< "$source_screenshot_roots"
    for configured_root in "${configured_screenshot_roots[@]}"; do
      if [[ "$configured_root" != /* || ! -d "$configured_root" ]]; then
        echo "Screenshot root must be an existing absolute directory: $configured_root" >&2
        exit 2
      fi
      memory_args+=(--source-screenshot-root "$configured_root")
    done
  fi
  IFS=':' read -r -a configured_runlog_roots <<< "$memory_runlog_roots"
  for configured_root in "${configured_runlog_roots[@]}"; do
    if [[ "$configured_root" != /* || ! -d "$configured_root" ]]; then
      echo "Memory RunLog root must be an existing absolute directory: $configured_root" >&2
      exit 2
    fi
    memory_args+=(--runlog-root "$configured_root")
  done
  IFS=':' read -r -a configured_result_roots <<< "$memory_result_roots"
  for configured_root in "${configured_result_roots[@]}"; do
    if [[ -z "$configured_root" ]]; then
      continue
    fi
    if [[ "$configured_root" != /* || ! -d "$configured_root" ]]; then
      echo "Memory result root must be an existing absolute directory: $configured_root" >&2
      exit 2
    fi
    memory_args+=(--result-root "$configured_root")
  done
  if [[ -n "$memory_mobilegpt_roots" ]]; then
    IFS=':' read -r -a configured_mobilegpt_roots <<< "$memory_mobilegpt_roots"
    for configured_root in "${configured_mobilegpt_roots[@]}"; do
      if [[ -z "$configured_root" ]]; then
        continue
      fi
      if [[ "$configured_root" != /* || ! -d "$configured_root" ]]; then
        echo "MobileGPT memory root must be an existing absolute directory: $configured_root" >&2
        exit 2
      fi
      memory_args+=(--mobilegpt-memory-root "$configured_root")
    done
  fi
  if [[ -n "$memory_baseline_batch_reports" ]]; then
    IFS=':' read -r -a configured_baseline_reports <<< "$memory_baseline_batch_reports"
    for configured_report in "${configured_baseline_reports[@]}"; do
      if [[ -z "$configured_report" ]]; then
        continue
      fi
      if [[ "$configured_report" != /* || ! -f "$configured_report" ]]; then
        echo "Baseline batch report must be an existing absolute file: $configured_report" >&2
        exit 2
      fi
      memory_args+=(--baseline-batch-report "$configured_report")
    done
  fi
  if [[ -n "$memory_function_catalogs" ]]; then
    IFS=':' read -r -a configured_function_catalogs <<< "$memory_function_catalogs"
    for configured_catalog in "${configured_function_catalogs[@]}"; do
      if [[ -z "$configured_catalog" ]]; then
        continue
      fi
      if [[ "$configured_catalog" != /* || ! -f "$configured_catalog" ]]; then
        echo "Memory Function catalog must be an existing absolute file: $configured_catalog" >&2
        exit 2
      fi
      memory_args+=(--function-catalog "$configured_catalog")
    done
  fi
  cd "$repo"
  exec "$python_bin" "${memory_args[@]}"
fi
load_memory_paths() {
  "$python_bin" - "$repo" "$memory_index" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
from src.experiment.artifact_memory import load_artifact_memory

index_path = Path(sys.argv[2]).expanduser().resolve()
load_artifact_memory(index_path)
pointer = json.loads(index_path.read_text(encoding="utf-8"))
print(
    "\t".join(
        (
            str(pointer["source_index"]),
            str(pointer["ours_store_index"]),
        )
    )
)
PY
}
if [[ "$convert_ours_assets" -eq 1 ]]; then
  if [[ "$prepare_mobilegpt_memory" -eq 1 || "$check_only" -eq 1 || "$dry_run" -eq 1 || "$all_tasks" -eq 1 ]]; then
    echo "--convert-ours-assets cannot be combined with experiment run options." >&2
    exit 2
  fi
  if [[ -z "$ours_source_asset_index" || -z "$ours_authoring_manifest" || -z "$ours_converted_asset_root" || -z "$memory_index" ]]; then
    echo "Asset conversion requires a source index, authoring manifest, output root, and OMNIFLOW_EXP_MEMORY_INDEX." >&2
    exit 2
  fi
  if [[ "$ours_source_asset_index" != /* || "$ours_authoring_manifest" != /* || "$ours_converted_asset_root" != /* ]]; then
    echo "Asset conversion index, authoring manifest, and output root must be absolute paths." >&2
    exit 2
  fi
  if [[ ! -f "$ours_source_asset_index" || ! -f "$ours_authoring_manifest" ]]; then
    echo "Asset conversion index and authoring manifest must exist." >&2
    exit 2
  fi
  if ! python_bin="$(command -v "$python_bin")"; then
    echo "Python runtime missing: ${PYTHON_BIN:-python3}" >&2
    exit 1
  fi
  if [[ "$memory_index" != /* || ! -f "$memory_index" ]]; then
    echo "Long-term-memory index must be an existing absolute file: $memory_index" >&2
    exit 2
  fi
  if [[ -z "${OMNIFLOW_OURS_SOURCE_ASSET_INDEX:-}" ]]; then
    memory_paths="$(load_memory_paths)"
    IFS=$'\t' read -r ours_source_asset_index _ <<< "$memory_paths"
  fi
  conversion_args=(
    -m src.experiment.function_assets
    --source-asset-index "$ours_source_asset_index"
    --authoring-manifest "$ours_authoring_manifest"
    --output-root "$ours_converted_asset_root"
    --memory-index "$memory_index"
  )
  if [[ -n "$ours_revision_reason" ]]; then
    conversion_args+=(--revision-reason "$ours_revision_reason")
  fi
  if [[ -n "$batch_task_filter" ]]; then
    IFS=',' read -r -a conversion_tasks <<< "$batch_task_filter"
    for conversion_task in "${conversion_tasks[@]}"; do
      if [[ -z "$conversion_task" ]]; then
        echo "Conversion task list contains an empty task name." >&2
        exit 2
      fi
      conversion_args+=(--task "$conversion_task")
    done
  fi
  cd "$repo"
  exec "$python_bin" "${conversion_args[@]}"
fi
if [[ -n "$batch_task_filter" && "$all_tasks" -eq 0 && "$prepare_mobilegpt_memory" -eq 0 ]]; then
  all_tasks=1
fi
if [[ "$prepare_mobilegpt_memory" -eq 1 ]]; then
  if [[ "$dry_run" -eq 1 || "$all_tasks" -eq 1 ]]; then
    echo "--prepare-mobilegpt-memory cannot be combined with formal experiment axes or --dry-run." >&2
    exit 2
  fi
fi
if [[ "$task_iteration" == "1" ]]; then
  default_method="$formal_default_method"
else
  default_method="ours"
fi
method="${selected_method_arg:-${OMNIFLOW_ANDROIDWORLD_METHOD:-$default_method}}"
case ",$all_methods," in
  *",$method,"*)
    ;;
  *)
    echo "Unsupported paper method: $method" >&2
    exit 2
    ;;
esac
device_target="${selected_device_arg:-${OMNIFLOW_ANDROIDWORLD_DEVICE:-$default_device}}"
target_serials=()
target_labels_seen=","
target_serials_seen=","
target_console_ports_seen=","
IFS=',' read -r -a target_specs <<< "$device_target"
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
  if [[ "$target_labels_seen" == *",$target_label,"* ]]; then
    echo "Duplicate target label: $target_label" >&2
    exit 2
  fi
  if [[ "$target_serials_seen" == *",$target_serial,"* ]]; then
    echo "Duplicate target serial: $target_serial" >&2
    exit 2
  fi
  if [[ "$target_console_ports_seen" == *",$target_console_port,"* ]]; then
    echo "Duplicate target console port: $target_console_port" >&2
    exit 2
  fi
  target_labels_seen+="$target_label,"
  target_serials_seen+="$target_serial,"
  target_console_ports_seen+="$target_console_port,"
  target_serials+=("$target_serial")
done
if [[ ${#target_serials[@]} -eq 0 ]]; then
  echo "At least one device target is required." >&2
  exit 2
fi

IFS=':' read -r source_label source_serial source_console_port source_extra <<< "$source_device"
if [[ -z "$source_label" || -z "$source_serial" || ! "$source_console_port" =~ ^[0-9]+$ || -n "${source_extra:-}" ]]; then
  echo "Invalid source device: $source_device" >&2
  exit 2
fi
if [[ "$source_serial" != "emulator-$source_console_port" ]]; then
  echo "Source serial/console mismatch: $source_device" >&2
  exit 2
fi
if [[ "$target_labels_seen" == *",$source_label,"* ]]; then
  echo "Source label must be separate from target labels: $source_label" >&2
  exit 2
fi
if [[ "$target_serials_seen" == *",$source_serial,"* ]]; then
  echo "Source serial must be separate from target serials: $source_serial" >&2
  exit 2
fi
if [[ "$target_console_ports_seen" == *",$source_console_port,"* ]]; then
  echo "Source console port must be separate from target console ports: $source_console_port" >&2
  exit 2
fi
if [[ -z "$memory_index" || "$memory_index" != /* || ! -f "$memory_index" ]]; then
  echo "Long-term-memory index missing; run --refresh-memory first: $memory_index" >&2
  exit 2
fi
if ! python_bin="$(command -v "$python_bin")"; then
  echo "Python runtime missing: ${PYTHON_BIN:-python3}" >&2
  exit 1
fi
indexed_store_path_for_task() {
  if [[ -z "$ours_store_index" ]]; then
    return 1
  fi
  "$python_bin" - "$ours_store_index" "$1" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

index_path = Path(sys.argv[1]).expanduser().resolve()
task_name = sys.argv[2]
payload = json.loads(index_path.read_text(encoding="utf-8"))
row = payload.get(task_name) if isinstance(payload, dict) else None
if not isinstance(row, dict):
    raise SystemExit(3)
fields = (
    ("store_path", "store_sha256"),
    ("source_run_log_path", "source_run_log_sha256"),
    ("transfer_states_path", "transfer_states_sha256"),
    ("provenance_path", "provenance_sha256"),
)
for path_field, hash_field in fields:
    path = Path(str(row.get(path_field) or "")).expanduser()
    expected = str(row.get(hash_field) or "").strip()
    if not path.is_absolute() or not path.is_file():
        raise SystemExit(
            f"ours_store_index_file_missing:{task_name}:{path_field}:{path}"
        )
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if not expected or actual != expected:
        raise SystemExit(
            f"ours_store_index_hash_mismatch:{task_name}:{path_field}:"
            f"expected={expected or 'missing'}:actual={actual}"
        )
provenance_path = Path(str(row.get("provenance_path") or "")).expanduser()
if not provenance_path.is_absolute() or not provenance_path.is_file():
    raise SystemExit(
        f"ours_store_index_file_missing:{task_name}:provenance_path:{provenance_path}"
    )
provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
collection = provenance.get("semantic_collection")
semantic_function = str(collection.get("function") or "") if isinstance(collection, dict) else ""
skill_authored = (
    semantic_function == "androidworld_runlog_harvester_skill"
    and isinstance(collection.get("producer"), dict)
    and collection["producer"].get("kind") == "androidworld_runlog_harvester_skill"
)
if not skill_authored:
    raise SystemExit(
        f"ours_store_index_mechanical_asset:{task_name}:"
        "create a Function registration from the androidworld-runlog-harvester skill"
    )
store_path = Path(str(row["store_path"])).resolve()
transfer_path = Path(str(row["transfer_states_path"])).resolve()
if transfer_path != store_path.with_name("transfer_states.json"):
    raise SystemExit(f"ours_store_index_catalog_mismatch:{task_name}")
print(store_path)
PY
}
memory_paths="$(load_memory_paths)"
IFS=$'\t' read -r memory_source_index memory_store_index <<< "$memory_paths"
master_source_index="$memory_source_index"
source_index="$memory_source_index"
ours_store_index="$memory_store_index"
export OMNIFLOW_EXP_MEMORY_INDEX="$memory_index"
if [[ "$prepare_mobilegpt_memory" -eq 1 ]]; then
  if [[ "$mobilegpt_root" != /* || ! -d "$mobilegpt_root/Server" ]]; then
    echo "MobileGPT source-only generation requires an absolute native MobileGPT root: $mobilegpt_root" >&2
    exit 2
  fi
  mobilegpt_batch_args=(
    --index "$source_index"
  )
  if [[ -n "$batch_task_filter" ]]; then
    IFS=',' read -r -a mobilegpt_batch_tasks <<< "$batch_task_filter"
    for mobilegpt_batch_task in "${mobilegpt_batch_tasks[@]}"; do
      if [[ -z "$mobilegpt_batch_task" ]]; then
        echo "MobileGPT task filter contains an empty task name." >&2
        exit 2
      fi
      mobilegpt_batch_args+=(--task "$mobilegpt_batch_task")
    done
  fi
  cd "$repo"
  if [[ "$check_only" -eq 1 ]]; then
    exec "$python_bin" -m src.experiment.mobilegpt_source \
      preflight-batch "${mobilegpt_batch_args[@]}"
  fi
  if [[ -z "$mobilegpt_memory_output_root" ]]; then
    if [[ -z "$asset_root" || "$asset_root" != /* ]]; then
      echo "Set OMNIFLOW_MOBILEGPT_MEMORY_OUTPUT_ROOT or OMNIFLOW_EXP_ASSET_ROOT." >&2
      exit 2
    fi
    mobilegpt_memory_output_root="$asset_root/runtime/evals/androidworld_mobilegpt_runlog_direct_memory/attempt-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  fi
  if [[ "$mobilegpt_memory_output_root" != /* ]]; then
    echo "OMNIFLOW_MOBILEGPT_MEMORY_OUTPUT_ROOT must be absolute." >&2
    exit 2
  fi
  if [[ -z "$env_file" || "$env_file" != /* || ! -f "$env_file" ]]; then
    echo "MobileGPT model generation requires an existing absolute OMNIFLOW_ENV_FILE." >&2
    exit 2
  fi
  set -a
  source "$env_file"
  set +a
  if ! "$python_bin" - "$mobilegpt_root/Server" <<'PY'
import importlib
import sys

sys.path.insert(0, sys.argv[1])
for module_name in (
    "pandas",
    "agents.task_agent",
    "mobilegpt",
    "screenParser.Encoder",
    "utils.parsing_utils",
):
    importlib.import_module(module_name)
PY
  then
    echo "MobileGPT source-only generation Python dependencies are incomplete: $python_bin" >&2
    exit 2
  fi
  echo "[mobilegpt-memory] start output=$mobilegpt_memory_output_root model=$formal_model"
  exec "$python_bin" -m src.experiment.mobilegpt_source batch \
    "${mobilegpt_batch_args[@]}" \
    --mobilegpt-root "$mobilegpt_root" \
    --output-root "$mobilegpt_memory_output_root" \
    --model "$formal_model" \
    --memory-index "$memory_index"
fi
requires_function_asset=0
if [[ "$method" == "ours" ]]; then
  requires_function_asset=1
fi
prepare_function_asset_for_task() {
  local requested_task="$1"
  local conversion_root resolved_store_path store_status revision_reason
  if resolved_store_path="$(indexed_store_path_for_task "$requested_task")"; then
    if [[ -z "$ours_revision_reason" ]]; then
      prepared_store_path="$resolved_store_path"
      return 0
    fi
    store_status=0
  else
    store_status="$?"
  fi
  if [[ "$check_only" -eq 1 || "$dry_run" -eq 1 ]]; then
    echo "Canonical Function asset requires creation or revision for task=$requested_task; a read-only check cannot create it." >&2
    return 1
  fi
  if [[ -z "$asset_root" || "$asset_root" != /* ]]; then
    echo "Set OMNIFLOW_EXP_ASSET_ROOT to an absolute path before source adaptation." >&2
    return 2
  fi
  if [[ -z "$ours_authoring_manifest" || "$ours_authoring_manifest" != /* || ! -f "$ours_authoring_manifest" ]]; then
    echo "OMNIFLOW_OURS_AUTHORING_MANIFEST must be an existing absolute file before source adaptation." >&2
    return 1
  fi
  conversion_root="$ours_converted_asset_root"
  if [[ -z "$conversion_root" ]]; then
    conversion_root="$asset_root/runtime/evals/androidworld_single_task_assets/source_seed_${formal_source_seed}/$requested_task/ours/from_canonical_runlog"
  elif [[ "$all_tasks" -eq 1 ]]; then
    conversion_root="$conversion_root/$requested_task"
  fi
  if [[ "$conversion_root" != /* ]]; then
    echo "OMNIFLOW_OURS_CONVERTED_ASSET_ROOT must be absolute." >&2
    return 2
  fi
  echo "[source-adapter] create method=ours task=$requested_task"
  conversion_args=(
    -m src.experiment.function_assets
    --source-asset-index "$source_index"
    --authoring-manifest "$ours_authoring_manifest"
    --output-root "$conversion_root"
    --memory-index "$memory_index"
    --task "$requested_task"
  )
  revision_reason="$ours_revision_reason"
  if [[ -n "$revision_reason" ]]; then
    conversion_args+=(--revision-reason "$revision_reason")
  fi
  "$python_bin" "${conversion_args[@]}"
  memory_paths="$(load_memory_paths)"
  IFS=$'\t' read -r memory_source_index memory_store_index <<< "$memory_paths"
  master_source_index="$memory_source_index"
  source_index="$memory_source_index"
  ours_store_index="$memory_store_index"
  if ! prepared_store_path="$(
    indexed_store_path_for_task "$requested_task"
  )"; then
    echo "Function conversion completed without a registered Store: task=$requested_task" >&2
    return 1
  fi
}
if [[ "$all_tasks" -eq 0 && "$requires_function_asset" -eq 1 && -z "$store_path" ]]; then
  prepared_store_path=""
  prepare_function_asset_for_task "$task"
  store_path="$prepared_store_path"
fi
if [[ "$check_only" -eq 1 && "$dry_run" -eq 1 ]]; then
  echo "--check-only cannot be combined with --dry-run." >&2
  exit 2
fi
if [[ ! "$task_iteration" =~ ^[1-3]$ ]]; then
  echo "OMNIFLOW_ANDROIDWORLD_TASK_ITERATION must be an integer from 1 through 3." >&2
  exit 2
fi
if [[ ! "$max_fallback_steps" =~ ^[0-5]$ ]]; then
  echo "OMNIFLOW_ANDROIDWORLD_MAX_FALLBACK_STEPS must be an integer from 0 through 5." >&2
  exit 2
fi
if [[ ! "$max_steps" =~ ^[1-9][0-9]*$ ]]; then
  echo "OMNIFLOW_ANDROIDWORLD_MAX_STEPS must be a positive integer." >&2
  exit 2
fi
timeout_sec="$((max_steps * formal_step_timeout_sec + official_validator_flush_grace_sec))"
if [[ ! "$expected_source_seed" =~ ^[0-9]+$ ]]; then
  echo "OMNIFLOW_ANDROIDWORLD_SOURCE_SEED must be a non-negative integer." >&2
  exit 2
fi
if [[ ! "$task_seed" =~ ^[0-9]+$ ]]; then
  echo "OMNIFLOW_ANDROIDWORLD_TASK_SEED must be a non-negative integer." >&2
  exit 2
fi
if [[ ! "$manage_emulators" =~ ^[01]$ ]]; then
  echo "OMNIFLOW_ANDROIDWORLD_MANAGE_EMULATORS must be 0 or 1." >&2
  exit 2
fi
printf -v iteration_label '%02d' "$task_iteration"
if [[ -n "$batch_attempt_id" ]]; then
  if [[ ! "$batch_attempt_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "OMNIFLOW_BATCH_ATTEMPT_ID must be one safe path component." >&2
    exit 2
  fi
  attempt_id="$batch_attempt_id"
else
  attempt_id="iteration_${iteration_label}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
fi
attempt_series_root="${results_root:+$results_root/androidworld_single_task_attempts/$task}"
output_root="${OMNIFLOW_ANDROIDWORLD_OUTPUT_PATH:-$attempt_series_root/$attempt_id}"
preflight_output_root="${results_root:+$results_root/preflight/$task/$attempt_id}"
requires_mobilegpt_source_memory=0
requires_appagent_source_memory=0
requires_omnitransfer=0
need_native_preflight=0
need_mobilegpt_preflight=0
need_appagent_preflight=0
contains_baseline_method=0

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
      requires_appagent_source_memory=1
      contains_baseline_method=1
      ;;
    t3a_hint)
      need_native_preflight=1
      contains_baseline_method=1
      ;;
    *)
      echo "Unsupported paper method: $method" >&2
      exit 2
      ;;
esac
if [[ "$task_iteration" != "1" && "$contains_baseline_method" -eq 1 && -z "$baseline_environment_repair" ]]; then
  echo "Baseline methods are frozen after iteration 1. Set OMNIFLOW_BASELINE_ENVIRONMENT_REPAIR_REASON only for an audited environment-only retry." >&2
  exit 2
fi

if [[ -z "$asset_root" || -z "$results_root" ]]; then
  echo "Set OMNIFLOW_EXP_ASSET_ROOT and OMNIFLOW_EXP_RESULTS_ROOT to external absolute paths." >&2
  exit 2
fi
if [[ "$asset_root" != /* || "$results_root" != /* ]]; then
  echo "Experiment asset and result roots must be absolute paths." >&2
  exit 2
fi
if [[ -n "$ours_store_index" && "$ours_store_index" != /* ]]; then
  echo "OMNIFLOW_OURS_STORE_INDEX must be an absolute path." >&2
  exit 2
fi
if ! python_bin="$(command -v "$python_bin")"; then
  echo "Python runtime missing: ${PYTHON_BIN:-python3}" >&2
  exit 1
fi
android_sdk_root="${OMNIFLOW_ANDROID_SDK_ROOT:-${ANDROID_SDK_ROOT:-${ANDROID_HOME:-$(resolve_default_android_sdk_root)}}}"
if [[ "$android_sdk_root" != /* ]]; then
  echo "Android SDK root must be an absolute path: $android_sdk_root" >&2
  exit 2
fi
export ANDROID_SDK_ROOT="$android_sdk_root"
export ANDROID_HOME="$android_sdk_root"
adb_bin="${OMNIFLOW_ADB_PATH:-$android_sdk_root/platform-tools/adb}"
export OMNIFLOW_ANDROIDWORLD_A11Y_APK="${OMNIFLOW_ANDROIDWORLD_A11Y_APK:-$repo/runtime/cache/androidworld/accessibility_forwarder.apk}"
emulator_bin="${OMNIFLOW_EMULATOR_BIN:-$android_sdk_root/emulator/emulator}"
avdmanager_bin="${OMNIFLOW_AVDMANAGER_BIN:-$android_sdk_root/cmdline-tools/latest/bin/avdmanager}"
export PATH="$account_root/.local/bin:$android_sdk_root/platform-tools:$PATH"
java_home="${OMNIFLOW_JAVA_HOME:-}"
if [[ -z "$java_home" ]]; then
  for java_candidate in \
    /home/wuzewen/Android/jdk17 \
    /home/wuzewen/.local/jdks/temurin-17 \
    /home/wuzewen/.local/jdks/corretto-17 \
    "/Applications/Android Studio.app/Contents/jbr/Contents/Home" \
    "$account_root/Applications/Android Studio.app/Contents/jbr/Contents/Home"; do
    if [[ -x "$java_candidate/bin/java" ]]; then
      java_home="$java_candidate"
      break
    fi
  done
fi
if [[ -n "$java_home" ]]; then
  if [[ "$java_home" != /* ]]; then
    echo "OMNIFLOW_JAVA_HOME must be an absolute path: $java_home" >&2
    exit 2
  fi
  java_bin="$java_home/bin/java"
else
  java_bin="$(command -v java || true)"
  if [[ -n "$java_bin" ]]; then
    java_home="$(cd "$(dirname "$java_bin")/.." && pwd)"
  fi
fi
if [[ -z "$java_bin" || ! -x "$java_bin" ]]; then
  echo "Java runtime missing; set OMNIFLOW_JAVA_HOME to JDK 17 or newer." >&2
  exit 1
fi
java_version_line="$({ "$java_bin" -version 2>&1 || true; } | head -1)"
if [[ "$java_version_line" =~ version\ \"([0-9]+) ]]; then
  java_major="${BASH_REMATCH[1]}"
else
  echo "Unable to determine Java version from $java_bin: $java_version_line" >&2
  exit 1
fi
if (( java_major < 17 )); then
  echo "Java 17 or newer is required: java=$java_bin version=$java_version_line" >&2
  exit 1
fi
export JAVA_HOME="$java_home"
export PATH="$java_home/bin:$PATH"
echo "[java] home=$java_home major=$java_major version=$java_version_line"
select_source_asset_revision() {
  local hash_index="${5:-$source_index}"
  local expected_source_model="${6:-}"
  local expected_schema_version="${7:-}"
  local expected_source_method="${8:-}"
  "$python_bin" - "$repo" "$1" "$2" "$hash_index" "$3" "${4:-}" "$expected_source_model" "$expected_schema_version" "$expected_source_method" "$memory_index" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
from src.experiment.source_assets import select_source_asset_revision
from src.experiment.protocol import SOURCE_SEED

source_index = json.loads(
    Path(sys.argv[4]).read_text(encoding="utf-8")
)
source_row = source_index.get(sys.argv[5])
if not isinstance(source_row, dict):
    raise SystemExit(f"canonical_source_task_missing:{sys.argv[5]}")
source_sha256 = str(
    source_row.get("retained_source_run_log_sha256")
    or source_row.get("source_run_log_sha256")
    or ""
).strip()
if not source_sha256:
    raise SystemExit(f"canonical_source_run_log_hash_missing:{sys.argv[5]}")
compatible_source_sha256s = []
lineage = source_row.get("source_run_log_lineage")
if lineage is not None:
    if (
        not isinstance(lineage, dict)
        or lineage.get("schema_version")
        != "omniflow.function-store-source-lineage.v1"
        or str(lineage.get("output_sha256") or "") != source_sha256
    ):
        raise SystemExit(f"canonical_source_run_log_lineage_invalid:{sys.argv[5]}")
    compatible_source_sha256s.append(str(lineage.get("source_sha256") or ""))
candidate_validator = None
if sys.argv[8] == "omniflow.mobilegpt-runlog-direct-memory.v1":
    from src.experiment.androidworld import validate_mobilegpt_adapted_memory
    from src.experiment.artifact_memory import (
        canonical_mobilegpt_memory_from_memory,
    )
    from src.experiment.mobilegpt_contract import (
        MOBILEGPT_SOURCE_METHOD_BY_SCHEMA,
    )

    source_run_log = str(
        source_row.get("retained_source_run_log")
        or source_row.get("source_run_log")
        or ""
    ).strip()
    memory_index = Path(sys.argv[10]).expanduser().resolve()
    if memory_index.is_file():
        indexed_memory = canonical_mobilegpt_memory_from_memory(
            memory_index=memory_index,
            task_name=sys.argv[5],
        )
        if indexed_memory is not None:
            indexed_schema = str(indexed_memory.get("schema_version") or "")
            indexed_source_method = MOBILEGPT_SOURCE_METHOD_BY_SCHEMA.get(
                indexed_schema
            )
            if (
                indexed_source_method
                and indexed_memory.get("source_method") == indexed_source_method
            ):
                try:
                    validate_mobilegpt_adapted_memory(
                        indexed_memory["memory_root"],
                        task_name=sys.argv[5],
                        source_seed=SOURCE_SEED,
                        source_run_log=source_run_log,
                        compatible_source_sha256s=compatible_source_sha256s,
                        expected_model=sys.argv[7],
                        expected_source_method=indexed_source_method,
                    )
                except (OSError, TypeError, ValueError):
                    pass
                else:
                    print(Path(indexed_memory["memory_root"]).resolve().parent)
                    raise SystemExit(0)

    def candidate_validator(candidate, _payload):
        try:
            validate_mobilegpt_adapted_memory(
                candidate / "memory",
                task_name=sys.argv[5],
                source_seed=SOURCE_SEED,
                source_run_log=source_run_log,
                compatible_source_sha256s=compatible_source_sha256s,
                expected_model=sys.argv[7],
                expected_source_method=sys.argv[9],
            )
        except (OSError, TypeError, ValueError):
            return False
        return True
elif sys.argv[8] == "omniflow.appagent-demo-memory.v2":
    def candidate_validator(_candidate, payload):
        return (
            payload.get("conversion_mode") == "canonical_runlog_offline"
            and payload.get("source_emulator_used") is False
        )
try:
    selected = select_source_asset_revision(
        sys.argv[2],
        manifest_name=sys.argv[3],
        expected_source_sha256=source_sha256,
        compatible_source_sha256s=compatible_source_sha256s,
        expected_source_model=sys.argv[7],
        expected_schema_version=sys.argv[8],
        expected_source_method=sys.argv[9],
        environment_repair_reason=sys.argv[6],
        candidate_validator=candidate_validator,
    )
except ValueError as error:
    message = str(error)
    if message.startswith("source_asset_retry_forbidden:"):
        print(message, file=sys.stderr)
        raise SystemExit(75) from error
    raise
print(selected)
PY
}
terminal_source_failure_marker() {
  local marker="$1"
  [[ -f "$marker" ]] || return 1
  "$python_bin" - "$marker" <<'PY'
import json
import sys
from pathlib import Path

try:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if payload.get("retry_allowed") is False else 1)
PY
}
if [[ "$all_tasks" -eq 0 && "$requires_mobilegpt_source_memory" -eq 1 && -z "$mobilegpt_source_memory_root" ]]; then
  mobilegpt_source_base="$asset_root/runtime/evals/androidworld_single_task_assets/source_seed_${formal_source_seed}/$task/mobilegpt_offline_retrieval"
  mobilegpt_source_attempt_root="$(
    select_source_asset_revision \
      "$mobilegpt_source_base" \
      "$mobilegpt_source_manifest_name" \
      "$task" \
      "$mobilegpt_source_environment_repair" \
      "$source_index" \
      "$formal_model" \
      "$mobilegpt_source_schema" \
      "$mobilegpt_source_method"
  )"
  mobilegpt_source_memory_root="$mobilegpt_source_attempt_root/memory"
fi
if [[ "$all_tasks" -eq 0 && "$requires_appagent_source_memory" -eq 1 && -z "$appagent_demo_memory_root" ]]; then
  appagent_source_base="$asset_root/runtime/evals/androidworld_single_task_assets/source_seed_${formal_source_seed}/$task/appagent_demo"
  appagent_demo_memory_root="$(
    select_source_asset_revision \
      "$appagent_source_base" \
      "appagent_demo_manifest.json" \
      "$task" \
      "$appagent_source_environment_repair" \
      "$source_index" \
      "" \
      "omniflow.appagent-demo-memory.v2"
  )"
fi
for external_path in \
  "$env_file" \
  "$master_source_index" \
  "$source_index" \
  "$android_world_root" \
  "$output_root" \
  "$preflight_output_root" \
  "$mobilegpt_root" \
  "$appagent_root"; do
  if [[ "$external_path" != /* ]]; then
    echo "Experiment runtime paths must be absolute: $external_path" >&2
    exit 2
  fi
done
if [[ "$all_tasks" -eq 0 ]]; then
  if [[ "$requires_mobilegpt_source_memory" -eq 1 && "$mobilegpt_source_memory_root" != /* ]]; then
    echo "Experiment task asset paths must be absolute: $mobilegpt_source_memory_root" >&2
    exit 2
  fi
  if [[ "$requires_appagent_source_memory" -eq 1 && "$appagent_demo_memory_root" != /* ]]; then
    echo "Experiment task asset paths must be absolute: $appagent_demo_memory_root" >&2
    exit 2
  fi
fi
if [[ "$requires_omnitransfer" -eq 1 && "$omnitransfer_root" != /* ]]; then
  echo "OmniTransfer root must be an absolute path." >&2
  exit 2
fi
if [[ "$requires_omnitransfer" -eq 1 && "$all_tasks" -eq 0 && "$store_path" != /* ]]; then
  echo "Ours Store must be an absolute path." >&2
  exit 2
fi
if [[ "$all_tasks" -eq 1 ]]; then
  if [[ "$expected_source_seed" != "$formal_source_seed" \
    || "$task_seed" != "$formal_task_seed" \
    || "$max_steps" != "$formal_max_steps" \
    || "$max_fallback_steps" != "$formal_max_fallback_steps" \
    || "$fixed_task_params" != "$formal_fixed_task_params" \
    || "$fold_state" != "$formal_fold_state" \
    || "$fold_size" != "$formal_fold_size" ]]; then
    echo "--all-tasks requires the frozen formal protocol: source_seed=$formal_source_seed task_seed=$formal_task_seed max_steps=$formal_max_steps max_fallback_steps=$formal_max_fallback_steps fixed_task_params=$formal_fixed_task_params fold_state=$formal_fold_state fold_size=$formal_fold_size" >&2
    exit 2
  fi
  if [[ "$task_iteration" != "1" && -z "$baseline_environment_repair" ]]; then
    echo "--all-tasks iterations after 1 require an audited environment repair reason." >&2
    exit 2
  fi
  if [[ ! -f "$source_index" ]]; then
    echo "Canonical source index missing: $source_index" >&2
    exit 1
  fi
  formal_tasks=()
  while IFS= read -r batch_task_name; do
    formal_tasks+=("$batch_task_name")
  done < <(
    "$python_bin" - "$source_index" "$batch_task_filter" <<'PY'
import json
import sys
from pathlib import Path

index_path = Path(sys.argv[1]).expanduser().resolve()
task_filter = sys.argv[2].strip()
payload = json.loads(index_path.read_text(encoding="utf-8"))
if not isinstance(payload, dict) or (not task_filter and len(payload) != 116):
    raise SystemExit(
        f"formal_task_index_invalid:expected="
        f"{'selected slice' if task_filter else 116}:actual="
        f"{len(payload) if isinstance(payload, dict) else 'not_object'}"
    )
for task_name in payload:
    print(task_name)
PY
  )
  if [[ -z "$batch_task_filter" && "${#formal_tasks[@]}" -ne 116 ]]; then
    echo "Formal task enumeration failed: expected 116, got ${#formal_tasks[@]}." >&2
    exit 1
  fi
  batch_tasks=()
  if [[ -n "$batch_task_filter" ]]; then
    for requested_task in ${batch_task_filter//,/ }; do
      if [[ -z "$requested_task" ]]; then
        echo "Batch task list contains an empty task name." >&2
        exit 2
      fi
      matched=0
      for formal_task in "${formal_tasks[@]}"; do
        if [[ "$formal_task" == "$requested_task" ]]; then
          matched=1
          break
        fi
      done
      if [[ "$matched" -ne 1 ]]; then
        echo "Unknown formal task in --tasks: $requested_task" >&2
        exit 2
      fi
      if ((${#batch_tasks[@]} > 0)); then
        for selected_task in "${batch_tasks[@]}"; do
          if [[ "$selected_task" == "$requested_task" ]]; then
            echo "Duplicate task in --tasks: $requested_task" >&2
            exit 2
          fi
        done
      fi
      batch_tasks+=("$requested_task")
    done
  else
    batch_tasks=("${formal_tasks[@]}")
  fi
  if [[ "${#batch_tasks[@]}" -eq 0 ]]; then
    echo "Batch task selection is empty." >&2
    exit 2
  fi

  # The E2E pipeline owns method/device scheduling and immutable result
  # accounting. This shell entry only enumerates tasks and dispatches one
  # bounded task pipeline at a time.
  batch_status=0
  for batch_task in "${batch_tasks[@]}"; do
    child_args=(--e2e-task "$batch_task" --task-deadline-sec "$e2e_task_deadline_sec")
    if [[ "$check_only" -eq 1 || "$dry_run" -eq 1 ]]; then
      child_args+=(--dry-run)
    fi
    if [[ -n "${OMNIFLOW_E2E_ATTEMPT_ID:-}" ]]; then
      child_args+=(--attempt-id "${OMNIFLOW_E2E_ATTEMPT_ID}")
    fi
    echo "[batch] dispatch task=$batch_task"
    if ! bash "$0" "${child_args[@]}"; then
      batch_status=1
      echo "[batch] task failed task=$batch_task" >&2
    fi
  done
  exit "$batch_status"
fi
if [[ "${OMNIFLOW_BATCH_CHILD:-0}" != "1" ]]; then
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
fi
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
"$python_bin" - \
  "$source_index" \
  "$task" \
  "$expected_source_seed" \
  "$repo" \
  "$asset_root" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

index_path = Path(sys.argv[1]).expanduser().resolve()
task_name = sys.argv[2]
expected_seed = int(sys.argv[3])
sys.path.insert(0, str(Path(sys.argv[4]).resolve()))
asset_root = Path(sys.argv[5]).expanduser().resolve()
from src.integrations.android_world.apps import resolve_androidworld_package
from src.integrations.runlog import import_run_log

payload = json.loads(index_path.read_text(encoding="utf-8"))
row = payload.get(task_name) if isinstance(payload, dict) else None
if not isinstance(row, dict):
    raise SystemExit(f"canonical_source_task_missing:{task_name}")
actual_seed = row.get("source_seed", row.get("replay_seed"))
actual_method = str(row.get("method") or "").strip()
if row.get("latest_official_success_source") is not True:
    raise SystemExit(f"formal_source_official_success_missing:{task_name}")
actual_kind = str(row.get("source_kind") or "").strip()
if (
    actual_kind
    and actual_kind != "androidworld_validator_success_source_runlog"
):
    raise SystemExit(
        f"formal_source_kind_mismatch:{task_name}:actual={actual_kind or 'missing'}"
    )
run_log_value = str(
    row.get("retained_source_run_log") or row.get("source_run_log") or ""
).strip()
run_log = Path(run_log_value).expanduser()
if not run_log.is_absolute():
    index_relative = (index_path.parent / run_log).resolve()
    asset_relative = (asset_root / run_log).resolve()
    run_log = index_relative if index_relative.is_file() else asset_relative
if not run_log.is_file():
    raise SystemExit(f"formal_source_runlog_not_found:{task_name}:{run_log}")
expected_sha256 = str(
    row.get("retained_source_run_log_sha256")
    or row.get("source_run_log_sha256")
    or ""
).strip()
actual_sha256 = hashlib.sha256(run_log.read_bytes()).hexdigest()
if not expected_sha256 or expected_sha256 != actual_sha256:
    raise SystemExit(
        f"formal_source_runlog_hash_mismatch:{task_name}:"
        f"expected={expected_sha256 or 'missing'}:actual={actual_sha256}"
    )
try:
    canonical = import_run_log(
        json.loads(run_log.read_text(encoding="utf-8")),
        package_resolver=resolve_androidworld_package,
    )
except (OSError, ValueError, json.JSONDecodeError) as error:
    raise SystemExit(
        f"formal_source_runlog_schema_invalid:{task_name}:{error}"
    ) from error
if (
    canonical.get("status") != "succeeded"
    or canonical.get("success") is not True
    or not canonical.get("steps")
):
    raise SystemExit(f"formal_source_runlog_not_successful:{task_name}")
print(
    f"[source] task={task_name} protocol_seed={expected_seed} "
    f"recorded_seed={actual_seed if actual_seed is not None else 'unrecorded'} "
    f"method={actual_method or 'unrecorded'} "
    f"index={index_path}"
)
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
mobilegpt_embedding_api_key="${OPENAI_API_KEY:-}"
mobilegpt_embedding_base_url="${OPENAI_BASE_URL:-}"
paper_model="$formal_model"
export OPENAI_MODEL="$paper_model"
export OMNIFLOW_PLANNER_MODEL="$paper_model"
export MOBILEGPT_CHAT_MODEL="$paper_model"
export OMNITRANSFER_ROOT="$omnitransfer_root"
unset MOBILEGPT_MEMORY_ONLY
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
require_file "runtime_preflight" "$preflight"
require_file "androidworld_setup" "$android_world_root/android_world/env/setup_device/apps.py"
require_file "adb" "$adb_bin"
require_file "java" "$java_bin"
if [[ "$manage_emulators" -eq 1 ]]; then
  require_file "emulator" "$emulator_bin"
fi
if [[ "$requires_omnitransfer" -eq 1 ]]; then
  require_file "omnitransfer_runtime" "$omnitransfer_root/src/omnitransfer/runtime.py"
  require_file "ours_store" "$store_path"
fi
if [[ "$need_mobilegpt_preflight" -eq 1 ]]; then
  require_file "mobilegpt_server" "$mobilegpt_root/Server/main.py"
fi
if [[ "$need_appagent_preflight" -eq 1 ]]; then
  require_directory "appagent_root" "$appagent_root"
fi
if [[ ${#missing_assets[@]} -gt 0 ]]; then
  echo "Static experiment asset preflight failed before device startup:" >&2
  printf '  - %s\n' "${missing_assets[@]}" >&2
  exit 1
fi
if [[ "$requires_omnitransfer" -eq 1 ]]; then
  "$python_bin" - "$store_path" <<'PY'
import json
import sys

from src.experiment.androidworld import validate_ours_transfer_assets

audit = validate_ours_transfer_assets(
    sys.argv[1],
    require_action_transfer=True,
)
print("[ours-assets] " + json.dumps(audit, sort_keys=True))
PY
fi

mobilegpt_source_attempt_root="$(dirname "$mobilegpt_source_memory_root")"
mobilegpt_source_manifest="$mobilegpt_source_attempt_root/$mobilegpt_source_manifest_name"
mobilegpt_source_generation_required=0
if [[ "$requires_mobilegpt_source_memory" -eq 1 ]]; then
  if [[ -f "$mobilegpt_source_manifest" ]]; then
    :
  elif [[ -e "$mobilegpt_source_attempt_root" ]]; then
    echo "Immutable MobileGPT conversion attempt is incomplete and cannot be sealed in place: $mobilegpt_source_attempt_root" >&2
    exit 1
  else
    mobilegpt_source_generation_required=1
  fi
fi
appagent_source_manifest="$appagent_demo_memory_root/appagent_demo_manifest.json"
appagent_source_generation_required=0
if [[ "$requires_appagent_source_memory" -eq 1 ]]; then
  if [[ -f "$appagent_source_manifest" ]]; then
    :
  elif [[ -e "$appagent_demo_memory_root" ]]; then
    echo "Immutable AppAgent source attempt is incomplete and cannot be retried: $appagent_demo_memory_root" >&2
    exit 1
  else
    appagent_source_generation_required=1
  fi
fi
if [[ "$dry_run" -eq 1 && ( "$mobilegpt_source_generation_required" -eq 1 || "$appagent_source_generation_required" -eq 1 ) ]]; then
  echo "Dry-run cannot create frozen source assets; prepare them with a real one-command run first." >&2
  exit 1
fi
if [[ "$requires_mobilegpt_source_memory" -eq 1 ]]; then
  if [[ "$mobilegpt_source_generation_required" -eq 1 ]]; then
    "$python_bin" -m src.experiment.mobilegpt_source preflight \
      --index "$source_index" \
      --task "$task"
  else
    "$python_bin" -m src.experiment.mobilegpt_source validate \
      --index "$source_index" \
      --task "$task" \
      --memory-root "$mobilegpt_source_memory_root" \
      --model "$paper_model" \
      --memory-index "$memory_index"
  fi
fi
if [[ "$requires_appagent_source_memory" -eq 1 ]]; then
  if [[ "$appagent_source_generation_required" -eq 1 ]]; then
    "$python_bin" -m src.experiment.appagent_source preflight \
      --index "$source_index" \
      --task "$task" \
      --model "$paper_model"
  else
    "$python_bin" -m src.experiment.appagent_source validate \
      --index "$source_index" \
      --task "$task" \
      --memory-root "$appagent_demo_memory_root" \
      --model "$paper_model"
  fi
fi
if [[ "$check_only" -eq 1 ]]; then
  echo "[static] ready task=$task method=$method; no persistent output created"
  exit 0
fi
if [[ "$mobilegpt_source_generation_required" -eq 1 ]]; then
  "$python_bin" -m src.experiment.mobilegpt_source prepare \
    --index "$source_index" \
    --task "$task" \
    --mobilegpt-root "$mobilegpt_root" \
    --output-root "$mobilegpt_source_attempt_root" \
    --model "$paper_model" \
    --memory-index "$memory_index"
fi
select_model_endpoint "$formal_model_endpoint_profile"
validate_experiment_model "$paper_model" "$formal_model_endpoint_profile"
validate_model_endpoint_auth
export MOBILEGPT_CHAT_API_KEY="$selected_model_api_key"
export MOBILEGPT_CHAT_BASE_URL="$selected_model_base_url"
if [[ "$need_mobilegpt_preflight" -eq 1 ]]; then
  if [[ -z "$mobilegpt_embedding_api_key" || -z "$mobilegpt_embedding_base_url" ]]; then
    echo "MobileGPT embedding endpoint is missing from OPENAI_API_KEY/OPENAI_BASE_URL in OMNIFLOW_ENV_FILE." >&2
    exit 2
  fi
  export MOBILEGPT_EMBEDDING_API_KEY="$mobilegpt_embedding_api_key"
  export MOBILEGPT_EMBEDDING_BASE_URL="$mobilegpt_embedding_base_url"
  if [[ "$requires_mobilegpt_source_memory" -eq 1 ]]; then
    mobilegpt_embedding_contract="$($python_bin -m src.integrations.mobilegpt_runtime \
      preflight-endpoints \
      --manifest "$mobilegpt_source_manifest" \
      --memory-root "$mobilegpt_source_memory_root" \
      --chat-model "$paper_model")"
    IFS=$'\t' read -r mobilegpt_runtime_embedding_model mobilegpt_runtime_embedding_dimension <<< "$mobilegpt_embedding_contract"
    export MOBILEGPT_EMBEDDING_MODEL="$mobilegpt_runtime_embedding_model"
    echo "[mobilegpt-endpoints] chat_model=$paper_model embedding_model=$mobilegpt_runtime_embedding_model embedding_dimension=$mobilegpt_runtime_embedding_dimension"
  else
    export MOBILEGPT_EMBEDDING_MODEL="text-embedding-v4"
    echo "[mobilegpt-endpoints] chat_model=$paper_model embedding_model=text-embedding-v4"
  fi
fi
echo "[model] model=$paper_model model_endpoint_profile=$formal_model_endpoint_profile model_endpoint=$selected_model_base_url"
if [[ "$dry_run" -eq 1 ]]; then
  echo "[dry-run] ready task=$task method=$method device=$device_target; no device or persistent output created"
  exit 0
fi
ensure_androidworld_sqlite_fts4

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

avd_spec_for_name() {
  local wanted_avd="$1"
  local spec spec_avd system_image device_profile extra
  IFS=',' read -r -a configured_avd_specs <<< "$emulator_avd_specs"
  for spec in "${configured_avd_specs[@]}"; do
    IFS='|' read -r spec_avd system_image device_profile extra <<< "$spec"
    if [[ "$spec_avd" == "$wanted_avd" && -n "$system_image" && -n "$device_profile" && -z "${extra:-}" ]]; then
      printf '%s\t%s\n' "$system_image" "$device_profile"
      return 0
    fi
  done
  return 1
}

ensure_avd_installed() {
  local avd="$1"
  local avd_spec system_image device_profile image_dir
  if "$emulator_bin" -list-avds | grep -Fqx "$avd"; then
    return 0
  fi
  if ! avd_spec="$(avd_spec_for_name "$avd")"; then
    echo "Configured AVD is not installed and has no provisioning spec: avd=$avd" >&2
    return 1
  fi
  IFS=$'\t' read -r system_image device_profile <<< "$avd_spec"
  image_dir="$android_sdk_root/${system_image//;/\/}"
  if [[ ! -d "$image_dir" ]]; then
    echo "Configured AVD system image is not installed: avd=$avd image=$system_image" >&2
    return 1
  fi
  if [[ ! -x "$avdmanager_bin" ]]; then
    echo "Android avdmanager missing: $avdmanager_bin" >&2
    return 1
  fi
  echo "[emulator] create-avd avd=$avd image=$system_image device=$device_profile"
  printf 'no\n' | "$avdmanager_bin" create avd \
    --name "$avd" \
    --package "$system_image" \
    --device "$device_profile"
  if ! "$emulator_bin" -list-avds | grep -Fqx "$avd"; then
    echo "AVD provisioning completed without the configured AVD: avd=$avd" >&2
    return 1
  fi
}

device_state() {
  local serial="$1"
  local devices
  devices="$("$adb_bin" devices 2>/dev/null || true)"
  awk -v wanted="$serial" '$1 == wanted {print $2; exit}' <<< "$devices"
}

managed_emulator_pids() {
  local serial="$1"
  local avd="$2"
  "$python_bin" -m src.experiment.emulator_processes \
    --serial "$serial" \
    --avd "$avd"
}

managed_emulator_pids_by_port() {
  local serial="$1"
  "$python_bin" -m src.experiment.emulator_processes \
    --serial "$serial" \
    --any-avd
}

force_stop_managed_emulator() {
  local serial="$1"
  local avd console_port grpc_port process_ids process_id stop_deadline
  console_port="${serial#emulator-}"
  grpc_port="$(( console_port + 3000 ))"
  if ! avd="$(avd_for_serial "$serial")"; then
    echo "Cannot identify managed emulator AVD for forced stop: $serial" >&2
    return 1
  fi
  if ! process_ids="$(managed_emulator_pids "$serial" "$avd")"; then
    echo "Cannot inspect managed emulator process: serial=$serial avd=$avd" >&2
    return 1
  fi
  if [[ -z "$process_ids" ]]; then
    if ! process_ids="$(managed_emulator_pids_by_port "$serial")"; then
      echo "Cannot inspect emulator process by console port: serial=$serial" >&2
      return 1
    fi
    if [[ -n "$process_ids" ]]; then
      echo "[emulator] recover mismatched-avd serial=$serial expected_avd=$avd"
    fi
  fi
  if [[ -z "$process_ids" ]]; then
    stop_deadline="$(( $(date +%s) + emulator_forced_shutdown_timeout_sec ))"
    while [[ -n "$(device_state "$serial")" ]] || grpc_ready "$grpc_port"; do
      if (( $(date +%s) >= stop_deadline )); then
        echo "No emulator process found while device remained visible: serial=$serial grpc=$grpc_port" >&2
        return 1
      fi
      sleep 1
    done
    echo "[emulator] already stopped serial=$serial"
    return 0
  fi
  if [[ "$process_ids" == *$'\n'* ]]; then
    echo "Ambiguous managed emulator processes: serial=$serial avd=$avd pids=${process_ids//$'\n'/,}" >&2
    return 1
  fi
  process_id="$process_ids"
  echo "[emulator] terminate serial=$serial avd=$avd pid=$process_id"
  if ! kill -TERM "$process_id" 2>/dev/null && kill -0 "$process_id" 2>/dev/null; then
    echo "Managed emulator rejected SIGTERM: serial=$serial avd=$avd pid=$process_id" >&2
    return 1
  fi
  stop_deadline="$(( $(date +%s) + emulator_forced_shutdown_timeout_sec ))"
  while kill -0 "$process_id" 2>/dev/null; do
    if (( $(date +%s) >= stop_deadline )); then
      echo "[emulator] kill serial=$serial avd=$avd pid=$process_id"
      if ! kill -KILL "$process_id" 2>/dev/null && kill -0 "$process_id" 2>/dev/null; then
        echo "Managed emulator rejected SIGKILL: serial=$serial avd=$avd pid=$process_id" >&2
        return 1
      fi
      break
    fi
    sleep 1
  done
}

stop_emulator() {
  local serial="$1"
  local reason="$2"
  local console_port grpc_port stop_deadline
  console_port="${serial#emulator-}"
  grpc_port="$(( console_port + 3000 ))"
  echo "[emulator] stop serial=$serial reason=$reason"
  "$adb_bin" -s "$serial" emu kill >/dev/null 2>&1 || true
  stop_deadline="$(( $(date +%s) + emulator_graceful_shutdown_timeout_sec ))"
  while [[ -n "$(device_state "$serial")" ]] || grpc_ready "$grpc_port"; do
    if (( $(date +%s) >= stop_deadline )); then
      force_stop_managed_emulator "$serial"
      stop_deadline="$(( $(date +%s) + emulator_forced_shutdown_timeout_sec ))"
      while [[ -n "$(device_state "$serial")" ]] || grpc_ready "$grpc_port"; do
        if (( $(date +%s) >= stop_deadline )); then
          echo "Managed emulator remained visible after exact process stop: serial=$serial grpc=$grpc_port" >&2
          return 1
        fi
        sleep 1
      done
      return 0
    fi
    sleep 1
  done
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
  local avd log_path current_state
  current_state="$(device_state "$serial")"
  if [[ "$manage_emulators" -ne 1 ]]; then
    if [[ "$current_state" == "device" ]] && grpc_ready "$grpc_port"; then
      echo "[emulator] reuse unmanaged serial=$serial grpc=$grpc_port"
      return 0
    fi
    echo "Emulator is not ready and automatic management is disabled: serial=$serial grpc=$grpc_port" >&2
    return 1
  fi
  if [[ -n "$current_state" ]] || grpc_ready "$grpc_port"; then
    echo "[emulator] cold-restart serial=$serial state=${current_state:-absent} grpc=$grpc_port"
    stop_emulator "$serial" "cold-restart"
  fi
  if ! avd="$(avd_for_serial "$serial")"; then
    echo "No managed AVD mapping exists for $serial in the canonical emulator topology." >&2
    return 1
  fi
  if ! ensure_avd_installed "$avd"; then
    echo "Configured AVD is unavailable: serial=$serial avd=$avd" >&2
    return 1
  fi
  log_path="$preflight_output_root/emulator_${serial#emulator-}.log"
  echo "[emulator] launch serial=$serial avd=$avd grpc=$grpc_port"
  nohup "$emulator_bin" \
    -avd "$avd" \
    -port "$console_port" \
    -grpc "$grpc_port" \
    -no-window \
    -no-audio \
    -no-boot-anim \
    -read-only \
    -no-snapshot-load \
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
fi
if [[ -z "$preflight_serials" ]]; then
  preflight_serials="${target_serials[*]}"
fi
source_index_expected_tasks="$($python_bin - "$source_index" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not isinstance(payload, dict) or not payload:
    raise SystemExit("source_index_empty")
print(len(payload))
PY
)"

mkdir -p "$preflight_output_root"
if [[ "$appagent_source_generation_required" -eq 1 ]]; then
  "$python_bin" -m src.experiment.appagent_source prepare \
    --index "$source_index" \
    --task "$task" \
    --appagent-root "$appagent_root" \
    --memory-root "$appagent_demo_memory_root" \
    --model "$paper_model"
fi
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
    --minimum-free-gb "$preflight_minimum_free_gb"
    --json-out "$preflight_output_root/runtime_preflight_${profile}_${serial#emulator-}.json"
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
      --expected-tasks "$source_index_expected_tasks"
      --source-index "$source_index"
      --source-task "$task"
    )
    if [[ "$task" == Contacts* ]]; then
      preflight_args+=(--require-contacts-ready)
    fi
  fi
  if [[ "$profile" == "appagent" ]]; then
    preflight_args+=(--appagent-root "$appagent_root")
    if [[ -n "$appagent_demo_memory_root" ]]; then
      preflight_args+=(--appagent-demo-memory-root "$appagent_demo_memory_root")
    fi
  fi
  if [[ "$profile" == "mobilegpt" && "$task" == Contacts* ]]; then
    preflight_args+=(--require-contacts-ready)
  fi
  "$python_bin" "$preflight" "${preflight_args[@]}"
done
done

command=(
  "$python_bin"
  -m
  src.experiment.androidworld
  result
  --index "$source_index"
  --android-world-root "$android_world_root"
  --adb-path "$adb_bin"
  --task "$task"
  --source-seed "$expected_source_seed"
  --task-iteration "$task_iteration"
  --output-path "$output_root"
  --master-source-index "$master_source_index"
  --result-registry-root "$results_root/androidworld_validator/runs"
  --master-progress-root "$results_root/androidworld_validator/master_progress"
  --omnitransfer-root "$omnitransfer_root"
  --store-path "$store_path"
  --store-index "$ours_store_index"
  --mobilegpt-root "$mobilegpt_root"
  --mobilegpt-source-memory-root "$mobilegpt_source_memory_root"
  --appagent-root "$appagent_root"
  --timeout-sec "$timeout_sec"
  --max-steps "$max_steps"
  --max-fallback-steps "$max_fallback_steps"
  --task-random-seed "$task_seed"
  --model "$paper_model"
  --planner-provider openai
)
if [[ -n "$baseline_environment_repair" ]]; then
  command+=(--baseline-environment-repair "$baseline_environment_repair")
fi
command+=(--method "$method")
if [[ "$fixed_task_params" != "1" ]]; then
  command+=(--no-fixed-task-params --task-params-json "")
fi
command+=(--device "$device_target")
if [[ -n "$appagent_demo_memory_root" ]]; then
  command+=(--appagent-demo-memory-root "$appagent_demo_memory_root")
fi
exec "${command[@]}"
