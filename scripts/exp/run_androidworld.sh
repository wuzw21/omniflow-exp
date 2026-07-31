#!/usr/bin/env bash

set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
asset_root="${OMNIFLOW_EXP_ASSET_ROOT:-}"
results_root="${OMNIFLOW_EXP_RESULTS_ROOT:-}"
python_bin="${PYTHON_BIN:-python3}"
env_file="${OMNIFLOW_ENV_FILE:-${asset_root:+$asset_root/.env}}"
master_source_index="${OMNIFLOW_MASTER_SOURCE_INDEX:-${asset_root:+$asset_root/runtime/evals/androidworld_validator/core_archive/success_source_runlogs/index_by_task.json}}"
source_index="${OMNIFLOW_SINGLE_TASK_SOURCE_INDEX:-$master_source_index}"
source_index_expected_tasks="${OMNIFLOW_SOURCE_INDEX_EXPECTED_TASKS:-116}"
formal_source_seed=111
formal_evaluation_seed=113
formal_max_steps=20
formal_max_fallback_steps=5
formal_fixed_task_params=0
formal_fold_state=2
formal_fold_size="2208x1840"
expected_source_seed="${OMNIFLOW_SINGLE_TASK_SOURCE_SEED:-$formal_source_seed}"
evaluation_seed="${OMNIFLOW_SINGLE_TASK_EVALUATION_SEED:-$formal_evaluation_seed}"
omnitransfer_root="${OMNITRANSFER_ROOT:-}"
android_world_root="${OMNIFLOW_ANDROID_WORLD_ROOT:-${asset_root:+$asset_root/runtime/external/droidrun-android-world/android_world}}"
export PYTHONPATH="$repo:$repo/src${android_world_root:+:$android_world_root}${PYTHONPATH:+:$PYTHONPATH}"
config="$repo/config/paper_androidworld.json"
preflight="$repo/src/experiment/preflight.py"
task="${OMNIFLOW_SINGLE_TASK_TASK:-SystemBluetoothTurnOn}"
task_iteration="${OMNIFLOW_SINGLE_TASK_ITERATION:-1}"
all_methods="fixed_replay,ours,mobilegpt_offline_retrieval,appagent_demo,t3a_hint"
eight_cell_methods="fixed_replay,ours,mobilegpt_offline_retrieval,appagent_demo"
baseline_environment_repair="${OMNIFLOW_BASELINE_ENVIRONMENT_REPAIR_REASON:-}"
mobilegpt_source_environment_repair="${OMNIFLOW_MOBILEGPT_SOURCE_ENVIRONMENT_REPAIR_REASON:-}"
appagent_source_environment_repair="${OMNIFLOW_APPAGENT_SOURCE_ENVIRONMENT_REPAIR_REASON:-}"
formal_device_targets="small5554:emulator-5554:5554,fold5564:emulator-5564:5564"
device_targets="${OMNIFLOW_SINGLE_TASK_DEVICE_TARGETS:-$formal_device_targets}"
fixed_task_params="${OMNIFLOW_SINGLE_TASK_FIXED_TASK_PARAMS:-$formal_fixed_task_params}"
timeout_sec="${OMNIFLOW_SINGLE_TASK_TIMEOUT_SEC:-600}"
max_steps="${OMNIFLOW_SINGLE_TASK_MAX_STEPS:-$formal_max_steps}"
max_fallback_steps="${OMNIFLOW_SINGLE_TASK_MAX_FALLBACK_STEPS:-$formal_max_fallback_steps}"
store_path="${OMNIFLOW_SINGLE_TASK_STORE_PATH:-}"
ours_store_index="${OMNIFLOW_OURS_STORE_INDEX:-}"
ours_source_asset_index="${OMNIFLOW_OURS_SOURCE_ASSET_INDEX:-$master_source_index}"
ours_converted_asset_root="${OMNIFLOW_OURS_CONVERTED_ASSET_ROOT:-}"
ours_authoring_manifest="${OMNIFLOW_OURS_AUTHORING_MANIFEST:-}"
memory_root="${OMNIFLOW_EXP_MEMORY_ROOT:-${asset_root:+$asset_root/androidworld_memory}}"
memory_index="${OMNIFLOW_EXP_MEMORY_INDEX:-${memory_root:+$memory_root/current.json}}"
memory_function_catalogs="${OMNIFLOW_MEMORY_FUNCTION_CATALOGS:-}"
memory_runlog_roots="${OMNIFLOW_MEMORY_RUNLOG_ROOTS:-${asset_root:+$asset_root/runtime/evals}}"
memory_result_roots="${OMNIFLOW_MEMORY_RESULT_ROOTS:-${asset_root:+$asset_root/runtime/evals}}"
source_selection_manifest="${OMNIFLOW_SOURCE_SELECTION_MANIFEST:-}"
if [[ -n "$results_root" && ":$memory_result_roots:" != *":$results_root:"* ]]; then
  memory_result_roots="${memory_result_roots:+$memory_result_roots:}$results_root"
fi
mobilegpt_root="${OMNIFLOW_MOBILEGPT_ROOT:-${asset_root:+$asset_root/runtime/external/mobilegpt}}"
mobilegpt_source_memory_root="${OMNIFLOW_MOBILEGPT_SOURCE_MEMORY_ROOT:-}"
appagent_root="${OMNIFLOW_APPAGENT_ROOT:-${asset_root:+$asset_root/runtime/external/appagent}}"
appagent_demo_memory_root="${OMNIFLOW_APPAGENT_DEMO_MEMORY_ROOT:-}"
source_device="${OMNIFLOW_SOURCE_DEVICE:-source5560:emulator-5560:5560}"
preflight_profile="${OMNIFLOW_SINGLE_TASK_PREFLIGHT_PROFILE:-}"
preflight_serials="${OMNIFLOW_SINGLE_TASK_PREFLIGHT_SERIALS:-}"
manage_emulators="${OMNIFLOW_SINGLE_TASK_MANAGE_EMULATORS:-1}"
emulator_avds="${OMNIFLOW_SINGLE_TASK_EMULATOR_AVDS:-emulator-5554=SmallPhone,emulator-5560=AndroidWorldAvd,emulator-5564=OmniFlowTargetPixelFoldApi34}"
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
default_emulator_avd_specs="SmallPhone|system-images;android-33;google_apis;$default_emulator_system_image_abi|small_phone,AndroidWorldAvd|system-images;android-33;google_apis;$default_emulator_system_image_abi|pixel_6,OmniFlowTargetPixelFoldApi34|system-images;android-34;google_apis;$default_emulator_system_image_abi|pixel_fold"
emulator_avd_specs="${OMNIFLOW_SINGLE_TASK_EMULATOR_AVD_SPECS:-$default_emulator_avd_specs}"
emulator_gpu="${OMNIFLOW_SINGLE_TASK_EMULATOR_GPU:-swiftshader_indirect}"
emulator_boot_timeout_sec="${OMNIFLOW_SINGLE_TASK_EMULATOR_BOOT_TIMEOUT_SEC:-240}"
fold_serial="${OMNIFLOW_SINGLE_TASK_FOLD_SERIAL:-emulator-5564}"
fold_state="${OMNIFLOW_SINGLE_TASK_FOLD_STATE:-$formal_fold_state}"
fold_size="${OMNIFLOW_SINGLE_TASK_FOLD_SIZE:-$formal_fold_size}"
dry_run=0
check_only=0
all_tasks=0
eight_cells=0
batch_task_filter=""
convert_ours_assets=0
refresh_memory=0
convert_source_runlogs=0
selected_methods_arg=""
selected_devices_arg=""
source_runlog_output_root="${OMNIFLOW_SOURCE_RUNLOG_OUTPUT_ROOT:-${memory_root:+$memory_root/source_runlogs}}"
source_screenshot_roots="${OMNIFLOW_SOURCE_SCREENSHOT_ROOTS:-}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/exp/run_androidworld.sh [OPTIONS]

Options:
  --check-only              Validate the complete selected run without creating
                            assets, attempts, result directories, or emulators.
  --dry-run                 Build one task command without executing it.
  --all-tasks               Run the selected task set in task-major order.
  --eight-cells             Select the four non-T3A methods (legacy shorthand).
  --methods METHOD1,...     Select an ordered subset of the five paper methods.
  --devices DEVICE1,...     Select small5554 and/or fold5564 independently.
  --tasks TASK1,TASK2,...   Run an ordered task-major subset, or limit
                            --convert-ours-assets. Implies --all-tasks during
                            experiment execution.
  --convert-ours-assets     Compile human-recorded source RunLogs with an
                            immutable offline authoring manifest, then validate,
                            freeze, and register the assets.
  --convert-source-runlogs  Convert the indexed legacy source RunLogs once to
                            omniflow.run_log.v1.
  --refresh-memory          Deduplicate and index all configured RunLogs,
                            Function assets, and existing results.
  -h, --help                Show this help and exit.

Required external roots:
  OMNIFLOW_EXP_ASSET_ROOT   Absolute root containing frozen experiment assets.
  OMNIFLOW_EXP_RESULTS_ROOT Absolute root for immutable results.
  OMNIFLOW_EXP_MEMORY_ROOT  Absolute content-addressed long-term-memory root.
  OMNITRANSFER_ROOT         Canonical/versioned OmniTransfer checkout.

Optional runtime overrides:
  PYTHON_BIN, OMNIFLOW_ENV_FILE, OMNIFLOW_SINGLE_TASK_SOURCE_INDEX,
  OMNIFLOW_MASTER_SOURCE_INDEX, OMNIFLOW_OURS_STORE_INDEX,
  OMNIFLOW_ANDROID_SDK_ROOT, OMNIFLOW_JAVA_HOME,
  OMNIFLOW_MOBILEGPT_SOURCE_ENVIRONMENT_REPAIR_REASON,
  OMNIFLOW_APPAGENT_SOURCE_ENVIRONMENT_REPAIR_REASON.
  Managed emulators are cold-restarted before every pending cell.

Asset conversion inputs:
  OMNIFLOW_OURS_SOURCE_ASSET_INDEX Source RunLog index; defaults to the master
                                   source index.
  OMNIFLOW_OURS_AUTHORING_MANIFEST Immutable offline Function authoring manifest.
  OMNIFLOW_OURS_CONVERTED_ASSET_ROOT New immutable conversion output root.
  OMNIFLOW_EXP_MEMORY_INDEX          Existing memory current.json.

Long-term-memory refresh inputs:
  OMNIFLOW_MEMORY_RUNLOG_ROOTS       Colon-separated evidence roots.
  OMNIFLOW_MEMORY_RESULT_ROOTS       Colon-separated result roots.
  OMNIFLOW_MEMORY_FUNCTION_CATALOGS  Colon-separated Function catalogs.
  OMNIFLOW_SOURCE_SELECTION_MANIFEST Optional audited exact-SHA source repairs.
  OMNIFLOW_SOURCE_SCREENSHOT_ROOTS   Optional screenshot roots for legacy repairs.

Source RunLog conversion inputs:
  OMNIFLOW_SOURCE_RUNLOG_OUTPUT_ROOT Absolute immutable output root.
  OMNIFLOW_SOURCE_SCREENSHOT_ROOTS   Optional colon-separated screenshot roots.

Examples:
  bash scripts/exp/run_androidworld.sh --tasks AudioRecorderRecordAudio
  bash scripts/exp/run_androidworld.sh --convert-ours-assets \
    --tasks AudioRecorderRecordAudio
  bash scripts/exp/run_androidworld.sh --refresh-memory
  bash scripts/exp/run_androidworld.sh --convert-source-runlogs
  bash scripts/exp/run_androidworld.sh --check-only --all-tasks \
    --methods mobilegpt_offline_retrieval
  bash scripts/exp/run_androidworld.sh --all-tasks \
    --methods mobilegpt_offline_retrieval
  bash scripts/exp/run_androidworld.sh --check-only --all-tasks --eight-cells
  bash scripts/exp/run_androidworld.sh --all-tasks --eight-cells \
    --tasks AudioRecorderRecordAudioWithFileName,SystemCopyToClipboard
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
    --dry-run)
      dry_run=1
      ;;
    --all-tasks)
      all_tasks=1
      ;;
    --eight-cells)
      eight_cells=1
      ;;
    --methods)
      shift
      if [[ "$#" -eq 0 || -z "$1" ]]; then
        echo "--methods requires a comma-separated method list." >&2
        exit 2
      fi
      selected_methods_arg="$1"
      ;;
    --devices)
      shift
      if [[ "$#" -eq 0 || -z "$1" ]]; then
        echo "--devices requires a comma-separated device list." >&2
        exit 2
      fi
      selected_devices_arg="$1"
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
if [[ "$convert_source_runlogs" -eq 1 ]]; then
  if [[ "$refresh_memory" -eq 1 || "$convert_ours_assets" -eq 1 || "$check_only" -eq 1 || "$dry_run" -eq 1 || "$all_tasks" -eq 1 || "$eight_cells" -eq 1 || -n "$selected_methods_arg" || -n "$selected_devices_arg" ]]; then
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
  if [[ "$convert_ours_assets" -eq 1 || "$check_only" -eq 1 || "$dry_run" -eq 1 || "$all_tasks" -eq 1 || "$eight_cells" -eq 1 || -n "$selected_methods_arg" || -n "$selected_devices_arg" || -n "$batch_task_filter" ]]; then
    echo "--refresh-memory cannot be combined with conversion or experiment run options." >&2
    exit 2
  fi
  if [[ -z "$memory_root" || "$memory_root" != /* ]]; then
    echo "Set OMNIFLOW_EXP_MEMORY_ROOT to an absolute path." >&2
    exit 2
  fi
  if [[ ! -f "$master_source_index" ]]; then
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
    --source-index "$master_source_index"
  )
  if [[ -n "$source_selection_manifest" ]]; then
    if [[ "$source_selection_manifest" != /* || ! -f "$source_selection_manifest" ]]; then
      echo "Source selection manifest must be an existing absolute file: $source_selection_manifest" >&2
      exit 2
    fi
    memory_args+=(--source-selection-manifest "$source_selection_manifest")
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
if [[ "$convert_ours_assets" -eq 1 ]]; then
  if [[ "$check_only" -eq 1 || "$dry_run" -eq 1 || "$all_tasks" -eq 1 || "$eight_cells" -eq 1 || -n "$selected_methods_arg" || -n "$selected_devices_arg" ]]; then
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
  conversion_args=(
    -m src.experiment.function_assets
    --source-asset-index "$ours_source_asset_index"
    --authoring-manifest "$ours_authoring_manifest"
    --output-root "$ours_converted_asset_root"
    --memory-index "$memory_index"
  )
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
if [[ -n "$batch_task_filter" && "$all_tasks" -eq 0 ]]; then
  all_tasks=1
fi
if [[ "$task_iteration" == "1" ]]; then
  if [[ "$eight_cells" -eq 1 ]]; then
    default_methods="$eight_cell_methods"
  else
    default_methods="$all_methods"
  fi
else
  default_methods="ours"
fi
methods="${selected_methods_arg:-${OMNIFLOW_SINGLE_TASK_METHODS:-$default_methods}}"
validate_method_subset() {
  local raw_methods="$1"
  local selected_method canonical_method
  local seen_methods="," canonical_match
  local -a selected_method_array=()
  IFS=',' read -r -a selected_method_array <<< "$raw_methods"
  if [[ "${#selected_method_array[@]}" -eq 0 ]]; then
    echo "Method selection is empty." >&2
    return 2
  fi
  for selected_method in "${selected_method_array[@]}"; do
    if [[ -z "$selected_method" ]]; then
      echo "Method selection contains an empty name: $raw_methods" >&2
      return 2
    fi
    canonical_match=0
    for canonical_method in ${all_methods//,/ }; do
      if [[ "$selected_method" == "$canonical_method" ]]; then
        canonical_match=1
        break
      fi
    done
    if [[ "$canonical_match" -ne 1 ]]; then
      echo "Unsupported paper method: $selected_method" >&2
      return 2
    fi
    if [[ "$seen_methods" == *",$selected_method,"* ]]; then
      echo "Duplicate method in --methods: $selected_method" >&2
      return 2
    fi
    seen_methods+="$selected_method,"
  done
}
validate_method_subset "$methods" || exit "$?"
if [[ "$eight_cells" -eq 1 && "$methods" != "$eight_cell_methods" ]]; then
  echo "--eight-cells requires exactly: $eight_cell_methods" >&2
  exit 2
fi
if [[ -n "$selected_devices_arg" ]]; then
  device_targets=""
  selected_device_seen=","
  selected_device_array=()
  IFS=',' read -r -a selected_device_array <<< "$selected_devices_arg"
  for selected_device in "${selected_device_array[@]}"; do
    if [[ "$selected_device_seen" == *",$selected_device,"* ]]; then
      echo "Duplicate device in --devices: $selected_device" >&2
      exit 2
    fi
    case "$selected_device" in
      small5554)
        selected_device_target="small5554:emulator-5554:5554"
        ;;
      fold5564)
        selected_device_target="fold5564:emulator-5564:5564"
        ;;
      *)
        echo "Unsupported formal device: $selected_device" >&2
        exit 2
        ;;
    esac
    device_targets="${device_targets:+$device_targets,}$selected_device_target"
    selected_device_seen+="$selected_device,"
  done
fi
if [[ -z "$memory_index" || "$memory_index" != /* || ! -f "$memory_index" ]]; then
  echo "Long-term-memory index missing; run --refresh-memory first: $memory_index" >&2
  exit 2
fi
if ! python_bin="$(command -v "$python_bin")"; then
  echo "Python runtime missing: ${PYTHON_BIN:-python3}" >&2
  exit 1
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
requires_function_asset=0
for selected_method in ${methods//,/ }; do
  case "$selected_method" in
    ours)
      requires_function_asset=1
      ;;
  esac
done
prepare_function_asset_for_task() {
  local requested_task="$1"
  local conversion_root resolved_store_path store_status
  if resolved_store_path="$(indexed_store_path_for_task "$requested_task")"; then
    prepared_store_path="$resolved_store_path"
    return 0
  else
    store_status="$?"
  fi
  if [[ "$store_status" -ne 3 ]]; then
    echo "Canonical Function asset is invalid for task=$requested_task." >&2
    return "$store_status"
  fi
  if [[ "$check_only" -eq 1 || "$dry_run" -eq 1 ]]; then
    echo "Canonical Function asset missing for task=$requested_task; a read-only check cannot create it." >&2
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
    conversion_root="$asset_root/runtime/evals/androidworld_single_task_assets/source_seed_111/$requested_task/ours/from_canonical_runlog"
  elif [[ "$all_tasks" -eq 1 ]]; then
    conversion_root="$conversion_root/$requested_task"
  fi
  if [[ "$conversion_root" != /* ]]; then
    echo "OMNIFLOW_OURS_CONVERTED_ASSET_ROOT must be absolute." >&2
    return 2
  fi
  echo "[source-adapter] create method=ours task=$requested_task"
  "$python_bin" -m src.experiment.function_assets \
    --source-asset-index "$source_index" \
    --authoring-manifest "$ours_authoring_manifest" \
    --output-root "$conversion_root" \
    --memory-index "$memory_index" \
    --task "$requested_task"
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
if [[ "$all_tasks" -eq 1 && "$dry_run" -eq 1 ]]; then
  echo "--dry-run cannot be combined with --all-tasks." >&2
  exit 2
fi
if [[ "$check_only" -eq 1 && "$dry_run" -eq 1 ]]; then
  echo "--check-only cannot be combined with --dry-run." >&2
  exit 2
fi
if [[ ! "$task_iteration" =~ ^[1-3]$ ]]; then
  echo "OMNIFLOW_SINGLE_TASK_ITERATION must be an integer from 1 through 3." >&2
  exit 2
fi
if [[ ! "$max_fallback_steps" =~ ^[0-5]$ ]]; then
  echo "OMNIFLOW_SINGLE_TASK_MAX_FALLBACK_STEPS must be an integer from 0 through 5." >&2
  exit 2
fi
if [[ ! "$max_steps" =~ ^[1-9][0-9]*$ ]]; then
  echo "OMNIFLOW_SINGLE_TASK_MAX_STEPS must be a positive integer." >&2
  exit 2
fi
if [[ ! "$source_index_expected_tasks" =~ ^[1-9][0-9]*$ ]]; then
  echo "OMNIFLOW_SOURCE_INDEX_EXPECTED_TASKS must be a positive integer." >&2
  exit 2
fi
if [[ ! "$expected_source_seed" =~ ^[0-9]+$ ]]; then
  echo "OMNIFLOW_SINGLE_TASK_SOURCE_SEED must be a non-negative integer." >&2
  exit 2
fi
if [[ ! "$evaluation_seed" =~ ^[0-9]+$ ]]; then
  echo "OMNIFLOW_SINGLE_TASK_EVALUATION_SEED must be a non-negative integer." >&2
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
preflight_output_root="${OMNIFLOW_SINGLE_TASK_PREFLIGHT_OUTPUT_ROOT:-${results_root:+$results_root/preflight/$task/$attempt_id}}"
requires_mobilegpt_source_memory=0
requires_appagent_source_memory=0
requires_omnitransfer=0
need_native_preflight=0
need_mobilegpt_preflight=0
need_appagent_preflight=0
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
done
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
android_sdk_root="${OMNIFLOW_ANDROID_SDK_ROOT:-${ANDROID_SDK_ROOT:-${ANDROID_HOME:-/home/wuzewen/Android/Sdk}}}"
if [[ "$android_sdk_root" != /* ]]; then
  echo "Android SDK root must be an absolute path: $android_sdk_root" >&2
  exit 2
fi
export ANDROID_SDK_ROOT="$android_sdk_root"
export ANDROID_HOME="$android_sdk_root"
adb_bin="${OMNIFLOW_ADB_PATH:-$android_sdk_root/platform-tools/adb}"
emulator_bin="${OMNIFLOW_EMULATOR_BIN:-$android_sdk_root/emulator/emulator}"
avdmanager_bin="${OMNIFLOW_AVDMANAGER_BIN:-$android_sdk_root/cmdline-tools/latest/bin/avdmanager}"
export PATH="/home/wuzewen/.local/bin:$android_sdk_root/platform-tools:$PATH"
java_home="${OMNIFLOW_JAVA_HOME:-}"
if [[ -z "$java_home" ]]; then
  for java_candidate in \
    /home/wuzewen/Android/jdk17 \
    /home/wuzewen/.local/jdks/temurin-17 \
    /home/wuzewen/.local/jdks/corretto-17; do
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
  "$python_bin" - "$repo" "$1" "$2" "$source_index" "$3" "${4:-}" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
from src.experiment.source_assets import select_source_asset_revision

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
try:
    selected = select_source_asset_revision(
        sys.argv[2],
        manifest_name=sys.argv[3],
        expected_source_sha256=source_sha256,
        environment_repair_reason=sys.argv[6],
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
if [[ "$all_tasks" -eq 0 && "$requires_mobilegpt_source_memory" -eq 1 && -z "$mobilegpt_source_memory_root" ]]; then
  mobilegpt_source_base="$asset_root/runtime/evals/androidworld_single_task_assets/source_seed_111/$task/mobilegpt_offline_retrieval"
  mobilegpt_source_attempt_root="$(
    select_source_asset_revision \
      "$mobilegpt_source_base" \
      "cold_memory_manifest.json" \
      "$task" \
      "$mobilegpt_source_environment_repair"
  )"
  mobilegpt_source_memory_root="$mobilegpt_source_attempt_root/memory"
fi
if [[ "$all_tasks" -eq 0 && "$requires_appagent_source_memory" -eq 1 && -z "$appagent_demo_memory_root" ]]; then
  appagent_source_base="$asset_root/runtime/evals/androidworld_single_task_assets/source_seed_111/$task/appagent_demo"
  appagent_demo_memory_root="$(
    select_source_asset_revision \
      "$appagent_source_base" \
      "appagent_demo_manifest.json" \
      "$task" \
      "$appagent_source_environment_repair"
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
  if [[ "$eight_cells" -eq 1 ]]; then
    batch_methods="$eight_cell_methods"
  else
    batch_methods="$methods"
  fi
  batch_method_array=()
  IFS=',' read -r -a batch_method_array <<< "$batch_methods"
  batch_method_count="${#batch_method_array[@]}"
  batch_device_array=()
  IFS=',' read -r -a batch_device_array <<< "$device_targets"
  batch_device_count="${#batch_device_array[@]}"
  seen_batch_devices=","
  for batch_device_target in "${batch_device_array[@]}"; do
    case "$batch_device_target" in
      small5554:emulator-5554:5554|fold5564:emulator-5564:5564)
        ;;
      *)
        echo "--all-tasks device selection must use formal targets: $batch_device_target" >&2
        exit 2
        ;;
    esac
    batch_device_label="${batch_device_target%%:*}"
    if [[ "$seen_batch_devices" == *",$batch_device_label,"* ]]; then
      echo "Duplicate formal device target: $batch_device_label" >&2
      exit 2
    fi
    seen_batch_devices+="$batch_device_label,"
  done
  if [[ "$batch_device_count" -eq 0 ]]; then
    echo "--all-tasks device selection is empty." >&2
    exit 2
  fi
  batch_cell_count="$((batch_method_count * batch_device_count))"
  if [[ "$expected_source_seed" != "$formal_source_seed" \
    || "$evaluation_seed" != "$formal_evaluation_seed" \
    || "$max_steps" != "$formal_max_steps" \
    || "$max_fallback_steps" != "$formal_max_fallback_steps" \
    || "$fixed_task_params" != "$formal_fixed_task_params" \
    || "$fold_state" != "$formal_fold_state" \
    || "$fold_size" != "$formal_fold_size" ]]; then
    echo "--all-tasks requires the frozen formal protocol: source_seed=$formal_source_seed evaluation_seed=$formal_evaluation_seed max_steps=$formal_max_steps max_fallback_steps=$formal_max_fallback_steps fixed_task_params=$formal_fixed_task_params fold_state=$formal_fold_state fold_size=$formal_fold_size" >&2
    exit 2
  fi
  if [[ "$task_iteration" != "1" && -z "$baseline_environment_repair" ]]; then
    echo "--all-tasks iterations after 1 require an audited environment repair reason; resume skips registered cells." >&2
    exit 2
  fi
  if [[ ! -f "$source_index" ]]; then
    echo "Canonical source index missing: $source_index" >&2
    exit 1
  fi
  if [[ -n "$ours_store_index" && ! -f "$ours_store_index" ]]; then
    echo "OmniFlow store index missing: $ours_store_index" >&2
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
  if [[ -n "$batch_task_filter" && "${#formal_tasks[@]}" -eq 0 ]]; then
    echo "Selected source index contains no tasks." >&2
    exit 1
  fi
  batch_tasks=()
  if [[ -n "$batch_task_filter" ]]; then
    requested_tasks=()
    IFS=',' read -r -a requested_tasks <<< "$batch_task_filter"
    for requested_task in "${requested_tasks[@]}"; do
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
      if [[ "${#batch_tasks[@]}" -gt 0 ]]; then
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
  batch_task_count="${#batch_tasks[@]}"
  source_index_task_count="${#formal_tasks[@]}"
  if [[ "$batch_task_count" -eq 0 ]]; then
    echo "Batch task selection is empty." >&2
    exit 2
  fi
  batch_output_root="${OMNIFLOW_BATCH_OUTPUT_ROOT:-$results_root/attempts}"
  batch_log_root="${OMNIFLOW_BATCH_LOG_ROOT:-$results_root/logs}"
  registration_plan_for_task() {
    "$python_bin" - \
      "$repo" \
      "$results_root/androidworld_validator/runs" \
      "$memory_index" \
      "$1" \
      "$batch_methods" \
      "$device_targets" \
      "$expected_source_seed" \
      "$evaluation_seed" \
      "$max_steps" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
from src.experiment.artifact_memory import registered_cell_plan_from_memory

memory_index = Path(sys.argv[3]).expanduser().resolve()
task = sys.argv[4]
methods = tuple(sys.argv[5].split(","))
device_specs = {}
for raw in sys.argv[6].split(","):
    fields = raw.split(":")
    if len(fields) != 3:
        raise SystemExit(f"invalid_device_target:{raw}")
    device_specs[fields[0]] = fields
devices = tuple(device_specs)
source_seed = int(sys.argv[7])
evaluation_seed = int(sys.argv[8])
max_steps = int(sys.argv[9])
plan = registered_cell_plan_from_memory(
    memory_index=memory_index,
    task_name=task,
    methods=methods,
    devices=devices,
    source_seed=source_seed,
    evaluation_seed=evaluation_seed,
    formal_max_steps=max_steps,
)
print(f"summary\t{len(plan['completed'])}\t{len(plan['pending'])}")
for method, device in plan["pending"]:
    label, serial, port = device_specs[device]
    print(f"pending\t{method}\t{label}\t{serial}\t{port}")
PY
  }
  batch_registration_plans=()
  batch_runnable_plans=()
  batch_terminal_plans=()
  batch_store_paths=()
  batch_mobilegpt_source_roots=()
  batch_appagent_source_roots=()
  pending_cell_count=0
  terminal_cell_count=0
  for batch_index in "${!batch_tasks[@]}"; do
    batch_task="${batch_tasks[$batch_index]}"
    registration_plan="$(registration_plan_for_task "$batch_task")"
    batch_registration_plans[$batch_index]="$registration_plan"
    registration_header="${registration_plan%%$'\n'*}"
    IFS=$'\t' read -r header_kind completed_cells pending_cells <<< "$registration_header"
    if [[ "$header_kind" != "summary" || ! "$completed_cells" =~ ^[0-9]+$ || ! "$pending_cells" =~ ^[0-9]+$ ]]; then
      echo "Invalid registration plan: task=$batch_task header=$registration_header" >&2
      exit 1
    fi
    if [[ "$pending_cells" -eq 0 ]]; then
      batch_store_paths[$batch_index]=""
      echo "[batch:static] already-complete task=$batch_task cells=$completed_cells/$batch_cell_count"
      continue
    fi
    pending_cell_count="$((pending_cell_count + pending_cells))"
    pending_task_methods=""
    for candidate_method in ${batch_methods//,/ }; do
      if grep -Fq $'pending\t'"$candidate_method"$'\t' <<< "$registration_plan"; then
        pending_task_methods="${pending_task_methods:+$pending_task_methods,}$candidate_method"
      fi
    done
    if [[ -z "$pending_task_methods" ]]; then
      echo "Registration plan contains no pending methods: task=$batch_task" >&2
      exit 1
    fi
    terminal_task_methods=""
    selected_mobilegpt_source_root=""
    selected_appagent_source_root=""
    for source_method in mobilegpt_offline_retrieval appagent_demo; do
      case ",$pending_task_methods," in
        *,$source_method,*)
          ;;
        *)
          continue
          ;;
      esac
      case "$source_method" in
        mobilegpt_offline_retrieval)
          source_base="$asset_root/runtime/evals/androidworld_single_task_assets/source_seed_${expected_source_seed}/$batch_task/$source_method"
          source_manifest="cold_memory_manifest.json"
          source_repair_reason="$mobilegpt_source_environment_repair"
          ;;
        appagent_demo)
          source_base="$asset_root/runtime/evals/androidworld_single_task_assets/source_seed_${expected_source_seed}/$batch_task/$source_method"
          source_manifest="appagent_demo_manifest.json"
          source_repair_reason="$appagent_source_environment_repair"
          ;;
      esac
      if selected_source_root="$(
        select_source_asset_revision \
          "$source_base" \
          "$source_manifest" \
          "$batch_task" \
          "$source_repair_reason"
      )"; then
        case "$source_method" in
          mobilegpt_offline_retrieval)
            selected_mobilegpt_source_root="$selected_source_root/memory"
            ;;
          appagent_demo)
            selected_appagent_source_root="$selected_source_root"
            ;;
        esac
      else
        source_status="$?"
        if [[ "$source_status" -ne 75 ]]; then
          echo "Source asset selection failed: task=$batch_task method=$source_method status=$source_status" >&2
          exit "$source_status"
        fi
        terminal_task_methods="${terminal_task_methods:+$terminal_task_methods,}$source_method"
        method_pending_count="$(
          awk -F '\t' -v method="$source_method" \
            '$1 == "pending" && $2 == method {count += 1} END {print count + 0}' \
            <<< "$registration_plan"
        )"
        terminal_cell_count="$((terminal_cell_count + method_pending_count))"
        echo "[batch:static] terminal task=$batch_task method=$source_method pending=$method_pending_count"
      fi
    done
    batch_mobilegpt_source_roots[$batch_index]="$selected_mobilegpt_source_root"
    batch_appagent_source_roots[$batch_index]="$selected_appagent_source_root"
    runnable_task_methods=""
    for candidate_method in ${batch_methods//,/ }; do
      if ! grep -Fq $'pending\t'"$candidate_method"$'\t' <<< "$registration_plan"; then
        continue
      fi
      case ",$terminal_task_methods," in
        *,$candidate_method,*)
          continue
          ;;
      esac
      runnable_task_methods="${runnable_task_methods:+$runnable_task_methods,}$candidate_method"
    done
    runnable_plan=""
    terminal_plan=""
    while IFS=$'\t' read -r row_kind cell_method cell_device cell_serial cell_port; do
      if [[ "$row_kind" != "pending" ]]; then
        continue
      fi
      case ",$terminal_task_methods," in
        *,$cell_method,*)
          terminal_plan+="${terminal_plan:+$'\n'}terminal"$'\t'"$cell_method"$'\t'"$cell_device"$'\t'"$cell_serial"$'\t'"$cell_port"
          ;;
        *)
          runnable_plan+="${runnable_plan:+$'\n'}pending"$'\t'"$cell_method"$'\t'"$cell_device"$'\t'"$cell_serial"$'\t'"$cell_port"
          ;;
      esac
    done <<< "$registration_plan"
    batch_runnable_plans[$batch_index]="$runnable_plan"
    batch_terminal_plans[$batch_index]="$terminal_plan"
    task_requires_function_asset=0
    case ",$runnable_task_methods," in
      *,ours,*)
        task_requires_function_asset=1
        ;;
    esac
    if [[ "$check_only" -eq 0 && "$task_requires_function_asset" -eq 1 ]]; then
      prepared_store_path=""
      prepare_function_asset_for_task "$batch_task"
    fi
    indexed_store_path=""
    if [[ "$task_requires_function_asset" -eq 1 ]]; then
      if indexed_store_path="$(indexed_store_path_for_task "$batch_task")"; then
        :
      else
        store_status="$?"
        if [[ "$store_status" -eq 3 ]]; then
          echo "Canonical Function asset missing for task=$batch_task; run --convert-ours-assets for this task before experiment execution." >&2
        else
          echo "Canonical Function asset is invalid for task=$batch_task." >&2
        fi
        exit "$store_status"
      fi
    fi
    batch_store_paths[$batch_index]="$indexed_store_path"
    if [[ -z "$runnable_task_methods" ]]; then
      echo "[batch:static] no-runnable-cells task=$batch_task completed=$completed_cells pending=$pending_cells"
      continue
    fi
    task_output_root="$batch_output_root/$batch_task/$attempt_id/static"
    child_static_args=(--check-only)
    echo "[batch:static] check task=$batch_task methods=$runnable_task_methods completed=$completed_cells pending=$pending_cells"
    (
      export OMNIFLOW_BATCH_CHILD=1
      export OMNIFLOW_SINGLE_TASK_TASK="$batch_task"
      export OMNIFLOW_SINGLE_TASK_METHODS="$runnable_task_methods"
      export OMNIFLOW_SINGLE_TASK_OUTPUT_ROOT="$task_output_root"
      export OMNIFLOW_SOURCE_INDEX_EXPECTED_TASKS="$source_index_task_count"
      unset OMNIFLOW_MOBILEGPT_SOURCE_MEMORY_ROOT
      unset OMNIFLOW_APPAGENT_DEMO_MEMORY_ROOT
      if [[ -n "$selected_mobilegpt_source_root" ]]; then
        export OMNIFLOW_MOBILEGPT_SOURCE_MEMORY_ROOT="$selected_mobilegpt_source_root"
      fi
      if [[ -n "$selected_appagent_source_root" ]]; then
        export OMNIFLOW_APPAGENT_DEMO_MEMORY_ROOT="$selected_appagent_source_root"
      fi
      if [[ -n "$indexed_store_path" ]]; then
        export OMNIFLOW_SINGLE_TASK_STORE_PATH="$indexed_store_path"
      fi
      bash "$0" "${child_static_args[@]}"
    )
  done
  echo "[batch:static] ready tasks=$batch_task_count; no persistent output created"
  if [[ "$check_only" -eq 1 ]]; then
    if [[ "$terminal_cell_count" -ne 0 ]]; then
      echo "[batch:static] incomplete terminal=$terminal_cell_count pending=$pending_cell_count total=$((batch_task_count * batch_cell_count))" >&2
      exit 1
    fi
    exit 0
  fi
  if [[ "$pending_cell_count" -eq 0 ]]; then
    echo "[batch] complete completed=0 skipped=$((batch_task_count * batch_cell_count)) total=$((batch_task_count * batch_cell_count))"
    exit 0
  fi

  mkdir -p "$batch_output_root" "$batch_log_root"
  completed=0
  skipped="$((batch_task_count * batch_cell_count - pending_cell_count))"
  for batch_index in "${!batch_tasks[@]}"; do
    batch_task="${batch_tasks[$batch_index]}"
    registration_plan="${batch_registration_plans[$batch_index]}"
    registration_header="${registration_plan%%$'\n'*}"
    IFS=$'\t' read -r _ completed_cells pending_cells <<< "$registration_header"
    if [[ "$pending_cells" -eq 0 ]]; then
      echo "[batch] skip complete task=$batch_task cells=$batch_cell_count/$batch_cell_count"
      continue
    fi
    runnable_plan="${batch_runnable_plans[$batch_index]}"
    terminal_plan="${batch_terminal_plans[$batch_index]}"
    indexed_store_path="${batch_store_paths[$batch_index]}"
    selected_mobilegpt_source_root="${batch_mobilegpt_source_roots[$batch_index]}"
    selected_appagent_source_root="${batch_appagent_source_roots[$batch_index]}"
    while IFS=$'\t' read -r row_kind cell_method cell_device cell_serial cell_port; do
      if [[ "$row_kind" != "pending" ]]; then
        continue
      fi
      terminal_line=$'terminal\t'"$cell_method"$'\t'"$cell_device"$'\t'"$cell_serial"$'\t'"$cell_port"
      if grep -Fqx "$terminal_line" <<< "$terminal_plan"; then
        continue
      fi
      task_output_root="$batch_output_root/$batch_task/$attempt_id/$cell_method/$cell_device/$attempt_id"
      task_log="$batch_log_root/$batch_task/$attempt_id/$cell_method-$cell_device.log"
      mkdir -p "$(dirname "$task_log")"
      echo "[batch] start task=$batch_task method=$cell_method device=$cell_device completed=$completed skipped=$skipped"
      if (
        export OMNIFLOW_BATCH_CHILD=1
        export OMNIFLOW_SINGLE_TASK_TASK="$batch_task"
        export OMNIFLOW_SINGLE_TASK_METHODS="$cell_method"
        export OMNIFLOW_SINGLE_TASK_DEVICE_TARGETS="$cell_device:$cell_serial:$cell_port"
        export OMNIFLOW_SINGLE_TASK_OUTPUT_ROOT="$task_output_root"
        export OMNIFLOW_SOURCE_INDEX_EXPECTED_TASKS="$source_index_task_count"
        unset OMNIFLOW_MOBILEGPT_SOURCE_MEMORY_ROOT
        unset OMNIFLOW_APPAGENT_DEMO_MEMORY_ROOT
        if [[ -n "$selected_mobilegpt_source_root" ]]; then
          export OMNIFLOW_MOBILEGPT_SOURCE_MEMORY_ROOT="$selected_mobilegpt_source_root"
        fi
        if [[ -n "$selected_appagent_source_root" ]]; then
          export OMNIFLOW_APPAGENT_DEMO_MEMORY_ROOT="$selected_appagent_source_root"
        fi
        if [[ -n "$indexed_store_path" ]]; then
          export OMNIFLOW_SINGLE_TASK_STORE_PATH="$indexed_store_path"
        fi
        bash "$0" </dev/null
      ) 2>&1 | tee "$task_log"; then
        status=0
      else
        status="$?"
      fi
      updated_plan="$(registration_plan_for_task "$batch_task")"
      pending_line=$'pending\t'"$cell_method"$'\t'"$cell_device"$'\t'"$cell_serial"$'\t'"$cell_port"
      if grep -Fqx "$pending_line" <<< "$updated_plan"; then
        runtime_terminal=0
        if [[ "$status" -ne 0 ]]; then
          case "$cell_method" in
            mobilegpt_offline_retrieval)
              source_base="$asset_root/runtime/evals/androidworld_single_task_assets/source_seed_${expected_source_seed}/$batch_task/$cell_method"
              source_manifest="cold_memory_manifest.json"
              source_repair_reason="$mobilegpt_source_environment_repair"
              ;;
            appagent_demo)
              source_base="$asset_root/runtime/evals/androidworld_single_task_assets/source_seed_${expected_source_seed}/$batch_task/$cell_method"
              source_manifest="appagent_demo_manifest.json"
              source_repair_reason="$appagent_source_environment_repair"
              ;;
            *)
              source_base=""
              source_manifest=""
              source_repair_reason=""
              ;;
          esac
          if [[ -n "$source_base" ]]; then
            if select_source_asset_revision \
              "$source_base" \
              "$source_manifest" \
              "$batch_task" \
              "$source_repair_reason" \
              >/dev/null; then
              :
            else
              source_status="$?"
              if [[ "$source_status" -eq 75 ]]; then
                runtime_terminal=1
              fi
            fi
          fi
        fi
        if [[ "$runtime_terminal" -eq 1 ]]; then
          newly_terminal=0
          while IFS=$'\t' read -r pending_kind pending_method pending_device pending_serial pending_port; do
            if [[ "$pending_kind" != "pending" || "$pending_method" != "$cell_method" ]]; then
              continue
            fi
            terminal_line=$'terminal\t'"$pending_method"$'\t'"$pending_device"$'\t'"$pending_serial"$'\t'"$pending_port"
            if grep -Fqx "$terminal_line" <<< "$terminal_plan"; then
              continue
            fi
            terminal_plan+="${terminal_plan:+$'\n'}$terminal_line"
            newly_terminal="$((newly_terminal + 1))"
          done <<< "$updated_plan"
          batch_terminal_plans[$batch_index]="$terminal_plan"
          terminal_cell_count="$((terminal_cell_count + newly_terminal))"
          echo "[batch] terminal task=$batch_task method=$cell_method pending=$newly_terminal"
          continue
        fi
        echo "[batch] stopped task=$batch_task method=$cell_method device=$cell_device status=$status log=$task_log" >&2
        if [[ "$status" -eq 0 ]]; then
          exit 1
        fi
        exit "$status"
      fi
      completed="$((completed + 1))"
      echo "[batch] registered task=$batch_task method=$cell_method device=$cell_device status=$status completed=$completed skipped=$skipped total=$((batch_task_count * batch_cell_count))"
    done <<< "$runnable_plan"
    final_plan="$(registration_plan_for_task "$batch_task")"
    final_header="${final_plan%%$'\n'*}"
    IFS=$'\t' read -r _ final_completed final_pending <<< "$final_header"
    if [[ "$final_pending" -ne 0 ]]; then
      while IFS=$'\t' read -r row_kind cell_method cell_device cell_serial cell_port; do
        if [[ "$row_kind" != "pending" ]]; then
          continue
        fi
        terminal_line=$'terminal\t'"$cell_method"$'\t'"$cell_device"$'\t'"$cell_serial"$'\t'"$cell_port"
        if ! grep -Fqx "$terminal_line" <<< "$terminal_plan"; then
          echo "Task resume ended with unexpected pending cell: task=$batch_task method=$cell_method device=$cell_device" >&2
          exit 1
        fi
      done <<< "$final_plan"
    fi
  done
  if [[ "$terminal_cell_count" -ne 0 ]]; then
    for batch_index in "${!batch_tasks[@]}"; do
      batch_task="${batch_tasks[$batch_index]}"
      terminal_plan="${batch_terminal_plans[$batch_index]}"
      while IFS=$'\t' read -r row_kind cell_method cell_device _; do
        if [[ "$row_kind" == "terminal" ]]; then
          echo "[batch] unresolved task=$batch_task method=$cell_method device=$cell_device reason=source_asset_retry_forbidden" >&2
        fi
      done <<< "$terminal_plan"
    done
    echo "[batch] incomplete completed=$completed skipped=$skipped terminal=$terminal_cell_count pending=$terminal_cell_count total=$((batch_task_count * batch_cell_count))" >&2
    exit 1
  fi
  echo "[batch] complete completed=$completed skipped=$skipped total=$((batch_task_count * batch_cell_count))"
  exit 0
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
paper_model="$("$python_bin" - "$config" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
model = str((payload.get("one_task") or {}).get("model") or "").strip()
if not model:
    raise SystemExit("paper_model_missing")
print(model)
PY
)"
if [[ "$paper_model" != "qwen3-vl-plus" ]]; then
  echo "Formal model must remain qwen3-vl-plus, got: $paper_model" >&2
  exit 1
fi
export OPENAI_MODEL="$paper_model"
export OMNIFLOW_PLANNER_MODEL="$paper_model"
export MOBILEGPT_CHAT_MODEL="$paper_model"
export OMNITRANSFER_ROOT="$omnitransfer_root"
unset OMNIFLOW_OOB_DEVICE_URL
export OMNIFLOW_OBSERVE_BACKEND="androidworld"
export OMNIFLOW_ACT_BACKEND="androidworld"
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
require_file "experiment_config" "$config"
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
if [[ "$requires_mobilegpt_source_memory" -eq 1 ]]; then
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
mobilegpt_source_manifest="$mobilegpt_source_attempt_root/cold_memory_manifest.json"
mobilegpt_source_generation_required=0
if [[ "$requires_mobilegpt_source_memory" -eq 1 ]]; then
  if [[ -f "$mobilegpt_source_manifest" ]]; then
    :
  elif [[ -e "$mobilegpt_source_attempt_root" ]]; then
    echo "Immutable MobileGPT source attempt is incomplete and cannot be retried: $mobilegpt_source_attempt_root" >&2
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
      --model "$paper_model"
  fi
fi
if [[ "$requires_appagent_source_memory" -eq 1 ]]; then
  if [[ "$appagent_source_generation_required" -eq 1 ]]; then
    "$python_bin" -m src.experiment.appagent_source preflight \
      --index "$source_index" \
      --store-index "$ours_store_index" \
      --task "$task"
  else
    "$python_bin" -m src.experiment.appagent_source validate \
      --index "$source_index" \
      --task "$task" \
      --memory-root "$appagent_demo_memory_root" \
      --model "$paper_model" \
      --store-index "$ours_store_index"
  fi
fi
if [[ "$check_only" -eq 1 ]]; then
  echo "[static] ready task=$task methods=$methods; no persistent output created"
  exit 0
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

source_label=""
source_serial=""
source_console_port=""
if [[ "$mobilegpt_source_generation_required" -eq 1 || "$appagent_source_generation_required" -eq 1 ]]; then
  IFS=':' read -r source_label source_serial source_console_port source_extra <<< "$source_device"
  if [[ -z "$source_label" || -z "$source_serial" || ! "$source_console_port" =~ ^[0-9]+$ || -n "${source_extra:-}" ]]; then
    echo "Invalid source device: $source_device" >&2
    exit 2
  fi
  if [[ "$source_serial" != "emulator-$source_console_port" ]]; then
    echo "Source serial/console mismatch: $source_device" >&2
    exit 2
  fi
  for target_serial in "${target_serials[@]}"; do
    if [[ "$target_serial" == "$source_serial" ]]; then
      echo "Source device must be separate from target devices: $source_serial" >&2
      exit 2
    fi
  done
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

stop_emulator() {
  local serial="$1"
  local reason="$2"
  local stop_deadline
  echo "[emulator] stop serial=$serial reason=$reason"
  "$adb_bin" -s "$serial" emu kill >/dev/null 2>&1 || true
  stop_deadline="$(( $(date +%s) + 30 ))"
  while [[ -n "$(device_state "$serial")" ]]; do
    if (( $(date +%s) >= stop_deadline )); then
      echo "Existing emulator could not be stopped safely: $serial" >&2
      return 1
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
  if [[ -n "$current_state" ]]; then
    echo "[emulator] cold-restart serial=$serial state=$current_state grpc=$grpc_port"
    stop_emulator "$serial" "cold-restart"
  elif grpc_ready "$grpc_port"; then
    echo "gRPC port is occupied without its emulator: 127.0.0.1:$grpc_port" >&2
    return 1
  fi
  if ! avd="$(avd_for_serial "$serial")"; then
    echo "No AVD mapping configured for $serial in OMNIFLOW_SINGLE_TASK_EMULATOR_AVDS." >&2
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

mkdir -p "$preflight_output_root"
if [[ "$mobilegpt_source_generation_required" -eq 1 || "$appagent_source_generation_required" -eq 1 ]]; then
  ensure_emulator "$source_serial"
fi
if [[ "$mobilegpt_source_generation_required" -eq 1 ]]; then
  "$python_bin" "$preflight" \
    --repo "$asset_root" \
    --code-root "$repo" \
    --profile mobilegpt \
    --serial "$source_serial" \
    --require-kvm \
    --require-device \
    --json-out "$preflight_output_root/runtime_preflight_mobilegpt_source_${source_serial#emulator-}.json" \
    --expected-tasks 116 \
    --require-contacts-ready
  "$python_bin" -m src.experiment.mobilegpt_source prepare \
    --index "$source_index" \
    --task "$task" \
    --mobilegpt-root "$mobilegpt_root" \
    --android-world-root "$android_world_root" \
    --output-root "$mobilegpt_source_attempt_root" \
    --model "$paper_model" \
    --serial "$source_serial" \
    --console-port "$source_console_port" \
    --adb-path "$adb_bin" \
    --max-steps "$max_steps"
fi
if [[ "$appagent_source_generation_required" -eq 1 ]]; then
  "$python_bin" "$preflight" \
    --repo "$asset_root" \
    --code-root "$repo" \
    --profile appagent \
    --serial "$source_serial" \
    --require-kvm \
    --require-device \
    --json-out "$preflight_output_root/runtime_preflight_appagent_source_${source_serial#emulator-}.json" \
    --appagent-root "$appagent_root"
  "$python_bin" -m src.experiment.appagent_source prepare \
    --index "$source_index" \
    --store-index "$ours_store_index" \
    --task "$task" \
    --appagent-root "$appagent_root" \
    --android-world-root "$android_world_root" \
    --memory-root "$appagent_demo_memory_root" \
    --model "$paper_model" \
    --serial "$source_serial" \
    --console-port "$source_console_port" \
    --adb-path "$adb_bin" \
    --timeout-sec "$timeout_sec"
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
  fi
  if [[ "$profile" == "appagent" ]]; then
    preflight_args+=(--appagent-root "$appagent_root")
    if [[ -n "$appagent_demo_memory_root" ]]; then
      preflight_args+=(--appagent-demo-memory-root "$appagent_demo_memory_root")
    fi
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
  --store-index "$ours_store_index"
  --mobilegpt-root "$mobilegpt_root"
  --mobilegpt-source-memory-root "$mobilegpt_source_memory_root"
  --appagent-root "$appagent_root"
  --timeout-sec "$timeout_sec"
  --max-steps "$max_steps"
  --max-fallback-steps "$max_fallback_steps"
  --task-random-seed "$evaluation_seed"
  --model "$paper_model"
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
