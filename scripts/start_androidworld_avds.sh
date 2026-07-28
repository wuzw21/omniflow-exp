#!/usr/bin/env bash

set -euo pipefail

AVD_NAME="${ANDROIDWORLD_AVD:-AndroidWorldAvd}"
CONSOLE_PORT="${ANDROIDWORLD_PORT:-5556}"
GRPC_PORT="${ANDROIDWORLD_GRPC_PORT:-8556}"
LOG_DIR="${ANDROIDWORLD_EMULATOR_LOG_DIR:-/tmp/omniflow_androidworld_avds}"
WAIT_FOR_DEVICE="${ANDROIDWORLD_WAIT_FOR_DEVICE:-1}"
BOOT_TIMEOUT_SEC="${ANDROIDWORLD_BOOT_TIMEOUT_SEC:-180}"
POST_BOOT_STABLE_SEC="${ANDROIDWORLD_POST_BOOT_STABLE_SEC:-5}"
RESTART_EXISTING="${ANDROIDWORLD_RESTART_EXISTING:-1}"
SNAPSHOT_MODE="${ANDROIDWORLD_SNAPSHOT_MODE:-no-snapshot-load}"
HEADLESS="${ANDROIDWORLD_HEADLESS:-0}"
GPU_MODE="${ANDROIDWORLD_GPU_MODE:-}"
READ_ONLY="${ANDROIDWORLD_READ_ONLY:-0}"
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: scripts/start_androidworld_avds.sh [options]

Start the AndroidWorld AVD used by OmniFlow AndroidWorld evaluation.

Defaults:
  AVD:  AndroidWorldAvd
  port: 5556
  grpc: 8556

Options:
  --avd NAME              AVD name.
  --port PORT             Emulator console port. Device serial is emulator-PORT.
  --grpc-port PORT        Emulator gRPC port. Use 0 to disable.
  --snapshot MODE         One of: load, no-snapshot-load, no-snapshot, wipe-data.
  --read-only            Run as a read-only multi-instance AVD.
  --no-wait               Start emulator and return immediately.
  --no-restart            Do not stop an existing emulator on the same port.
  --dry-run               Print the launch command without starting.
  --list                  List installed AVDs.
  -h, --help              Show this help.

Environment overrides:
  ANDROID_EMULATOR_BIN, ANDROID_ADB_BIN
  ANDROIDWORLD_AVD, ANDROIDWORLD_PORT, ANDROIDWORLD_GRPC_PORT
  ANDROIDWORLD_SNAPSHOT_MODE, ANDROIDWORLD_WAIT_FOR_DEVICE
  ANDROIDWORLD_BOOT_TIMEOUT_SEC, ANDROIDWORLD_POST_BOOT_STABLE_SEC
  ANDROIDWORLD_RESTART_EXISTING, ANDROIDWORLD_HEADLESS
  ANDROIDWORLD_GPU_MODE, ANDROIDWORLD_READ_ONLY
USAGE
}

resolve_android_tool() {
  local subdir="$1"
  local binary_name="$2"
  local override_var="$3"
  local override="${!override_var:-}"
  local candidate

  if [ -n "$override" ]; then
    if [ -x "$override" ]; then
      printf '%s\n' "$override"
      return 0
    fi
    echo "[error] ${override_var} is not executable: ${override}" >&2
    return 1
  fi

  if command -v "$binary_name" >/dev/null 2>&1; then
    command -v "$binary_name"
    return 0
  fi

  for candidate in \
    "$HOME/Library/Android/sdk/${subdir}/${binary_name}" \
    "$HOME/Android/Sdk/${subdir}/${binary_name}" \
    "${ANDROID_HOME:-}/${subdir}/${binary_name}" \
    "${ANDROID_SDK_ROOT:-}/${subdir}/${binary_name}"
  do
    if [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  echo "[error] Android ${binary_name} binary not found. Set ${override_var}." >&2
  return 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --avd)
      AVD_NAME="${2:-}"
      shift 2
      ;;
    --port)
      CONSOLE_PORT="${2:-}"
      shift 2
      ;;
    --grpc-port)
      GRPC_PORT="${2:-}"
      shift 2
      ;;
    --snapshot)
      SNAPSHOT_MODE="${2:-}"
      shift 2
      ;;
    --read-only)
      READ_ONLY=1
      shift
      ;;
    --no-wait)
      WAIT_FOR_DEVICE=0
      shift
      ;;
    --no-restart)
      RESTART_EXISTING=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --list)
      EMULATOR_BIN="$(resolve_android_tool emulator emulator ANDROID_EMULATOR_BIN)"
      "$EMULATOR_BIN" -list-avds
      exit 0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[error] Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ -z "$AVD_NAME" ]; then
  echo "[error] AVD name is required." >&2
  exit 2
fi
if ! [[ "$CONSOLE_PORT" =~ ^[0-9]+$ ]] || [ "$CONSOLE_PORT" -le 0 ]; then
  echo "[error] Invalid emulator port: ${CONSOLE_PORT}" >&2
  exit 2
fi
if ! [[ "$GRPC_PORT" =~ ^[0-9]+$ ]]; then
  echo "[error] Invalid gRPC port: ${GRPC_PORT}" >&2
  exit 2
fi

EMULATOR_BIN="$(resolve_android_tool emulator emulator ANDROID_EMULATOR_BIN)"
ADB_BIN="$(resolve_android_tool platform-tools adb ANDROID_ADB_BIN)"
SERIAL="emulator-${CONSOLE_PORT}"
LOG_FILE="${LOG_DIR}/${AVD_NAME}-${CONSOLE_PORT}.log"
PID_FILE="${LOG_DIR}/${AVD_NAME}-${CONSOLE_PORT}.pid"
LAUNCH_LABEL="com.omniflow.androidworld.${AVD_NAME}.${CONSOLE_PORT}"
AVD_HOME="${ANDROID_AVD_HOME:-${HOME}/.android/avd}"
AVD_DIR="${AVD_HOME}/${AVD_NAME}.avd"
LOCK_FILE="${AVD_DIR}/multiinstance.lock"

if ! "$EMULATOR_BIN" -list-avds | grep -Fxq "$AVD_NAME"; then
  echo "[error] AVD not found: ${AVD_NAME}" >&2
  echo "[hint] Available AVDs:" >&2
  "$EMULATOR_BIN" -list-avds >&2 || true
  exit 1
fi

mkdir -p "$LOG_DIR"

stop_existing() {
  if command -v launchctl >/dev/null 2>&1; then
    launchctl remove "$LAUNCH_LABEL" >/dev/null 2>&1 || true
  fi
  if "$ADB_BIN" devices | grep -q "^${SERIAL}[[:space:]]"; then
    echo "[stop] ${SERIAL} via adb emu kill"
    "$ADB_BIN" -s "$SERIAL" emu kill >/dev/null 2>&1 || true
    sleep 2
  fi
  if [ -f "$PID_FILE" ]; then
    old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "$old_pid" ] && kill -0 "$old_pid" >/dev/null 2>&1; then
      echo "[stop] old emulator pid ${old_pid}"
      kill "$old_pid" >/dev/null 2>&1 || true
      sleep 2
    fi
  fi
  pkill -f "qemu-system.*-avd ${AVD_NAME}.*-port ${CONSOLE_PORT}" >/dev/null 2>&1 || true
  pkill -f "emulator.*-avd ${AVD_NAME}.*-port ${CONSOLE_PORT}" >/dev/null 2>&1 || true
  rm -f "$PID_FILE"
}

emulator_process_alive() {
  if [ -f "$PID_FILE" ]; then
    old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "$old_pid" ] && kill -0 "$old_pid" >/dev/null 2>&1; then
      return 0
    fi
  fi
  pgrep -f "qemu-system.*-avd ${AVD_NAME}.*-port ${CONSOLE_PORT}" >/dev/null 2>&1 && return 0
  pgrep -f "emulator.*-avd ${AVD_NAME}.*-port ${CONSOLE_PORT}" >/dev/null 2>&1 && return 0
  if command -v launchctl >/dev/null 2>&1; then
    launchctl list "$LAUNCH_LABEL" >/dev/null 2>&1 && return 0
  fi
  return 1
}

record_emulator_pid() {
  local pid=""
  pid="$(pgrep -f "qemu-system.*-avd ${AVD_NAME}.*-port ${CONSOLE_PORT}" | head -n 1 || true)"
  if [ -z "$pid" ]; then
    pid="$(pgrep -f "emulator.*-avd ${AVD_NAME}.*-port ${CONSOLE_PORT}" | head -n 1 || true)"
  fi
  if [ -n "$pid" ]; then
    printf '%s\n' "$pid" >"$PID_FILE"
    echo "[start] pid=${pid}"
  fi
}

snapshot_args=()
case "$SNAPSHOT_MODE" in
  load|"")
    ;;
  no-snapshot-load)
    snapshot_args=(-no-snapshot-load)
    ;;
  no-snapshot)
    snapshot_args=(-no-snapshot)
    ;;
  wipe-data)
    snapshot_args=(-wipe-data)
    ;;
  *)
    echo "[error] Unsupported snapshot mode: ${SNAPSHOT_MODE}" >&2
    exit 2
    ;;
esac

cmd=("$EMULATOR_BIN" -avd "$AVD_NAME" -port "$CONSOLE_PORT")
if [ "$GRPC_PORT" -gt 0 ]; then
  cmd+=(-grpc "$GRPC_PORT")
fi
if [ "$HEADLESS" = "1" ]; then
  cmd+=(-no-window -no-audio -no-boot-anim)
fi
if [ -n "$GPU_MODE" ]; then
  cmd+=(-gpu "$GPU_MODE")
fi
if [ "$READ_ONLY" = "1" ]; then
  cmd+=(-read-only)
fi
cmd+=("${snapshot_args[@]}")

if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ] && command -v arch >/dev/null 2>&1; then
  cmd=(arch -arm64 "${cmd[@]}")
fi

echo "[config] avd=${AVD_NAME} serial=${SERIAL} grpc=${GRPC_PORT} snapshot=${SNAPSHOT_MODE} headless=${HEADLESS} gpu=${GPU_MODE:-default} read_only=${READ_ONLY}"
echo "[config] emulator=${EMULATOR_BIN}"
echo "[config] adb=${ADB_BIN}"
echo "[config] log=${LOG_FILE}"
echo "[config] pid=${PID_FILE}"
echo "[config] launch_label=${LAUNCH_LABEL}"

if [ "$DRY_RUN" = "1" ]; then
  printf '[dry-run]'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

if [ "$RESTART_EXISTING" = "1" ]; then
  stop_existing
fi

if [ "$READ_ONLY" != "1" ] && [ -e "$LOCK_FILE" ]; then
  echo "[clean] ${LOCK_FILE}"
  rm -f "$LOCK_FILE"
fi

: >"$LOG_FILE"
echo "[start] ${AVD_NAME} -> ${SERIAL}"
if [ "$(uname -s)" = "Darwin" ] && command -v launchctl >/dev/null 2>&1; then
  launchctl submit -l "$LAUNCH_LABEL" -o "$LOG_FILE" -e "$LOG_FILE" -- "${cmd[@]}"
  sleep 1
  record_emulator_pid
else
  nohup "${cmd[@]}" >"$LOG_FILE" 2>&1 < /dev/null &
  EMULATOR_PID=$!
  printf '%s\n' "$EMULATOR_PID" >"$PID_FILE"
  echo "[start] pid=${EMULATOR_PID}"
fi

if [ "$WAIT_FOR_DEVICE" != "1" ]; then
  echo "[ok] emulator start requested"
  echo "[hint] $ADB_BIN devices -l"
  exit 0
fi

echo "[wait] waiting for ${SERIAL}"
deadline=$((SECONDS + BOOT_TIMEOUT_SEC))
while [ "$SECONDS" -lt "$deadline" ]; do
  if ! emulator_process_alive; then
    echo "[error] emulator process exited before boot completed" >&2
    echo "[hint] tail -n 80 ${LOG_FILE}" >&2
    exit 1
  fi
  if ! "$ADB_BIN" devices | grep -q "^${SERIAL}[[:space:]]device"; then
    sleep 2
    continue
  fi
  boot_completed="$("$ADB_BIN" -s "$SERIAL" shell getprop sys.boot_completed 2>/dev/null | tr -d '\r' || true)"
  if [ "$boot_completed" = "1" ]; then
    echo "[wait] ${SERIAL} boot completed; checking stability for ${POST_BOOT_STABLE_SEC}s"
    sleep "$POST_BOOT_STABLE_SEC"
    record_emulator_pid
    if ! emulator_process_alive; then
      echo "[error] emulator process exited after boot completed" >&2
      echo "[hint] tail -n 80 ${LOG_FILE}" >&2
      exit 1
    fi
    if ! "$ADB_BIN" devices | grep -q "^${SERIAL}[[:space:]]device"; then
      echo "[error] ${SERIAL} disappeared after boot completed" >&2
      echo "[hint] tail -n 80 ${LOG_FILE}" >&2
      exit 1
    fi
    echo "[ok] ${SERIAL} boot completed and stable"
    "$ADB_BIN" devices -l
    exit 0
  fi
  sleep 2
done

echo "[error] timed out waiting for ${SERIAL} boot completion" >&2
echo "[hint] tail -n 80 ${LOG_FILE}" >&2
exit 1
