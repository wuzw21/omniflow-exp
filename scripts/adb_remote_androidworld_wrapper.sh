#!/usr/bin/env bash

set -euo pipefail

REAL_ADB="${OMNIFLOW_REMOTE_ADB_REAL:-$HOME/.cache/omniflow/android-platform-tools/platform-tools/adb}"
REMOTE_ADB_PORT="${OMNIFLOW_REMOTE_ADB_PORT:-15050}"
CONSOLE_AUTH_TOKEN_FILE="${OMNIFLOW_EMULATOR_AUTH_TOKEN_FILE:-$HOME/.cache/omniflow/emulator-console-auth-token}"

raw_args=("$@")
console_port=""
emu_index=""
for ((index = 0; index < ${#raw_args[@]}; index++)); do
  if [[ "${raw_args[$index]}" == "-s" && "$((index + 1))" -lt "${#raw_args[@]}" ]]; then
    case "${raw_args[$((index + 1))]}" in
      emulator-16554|emulator-5554) console_port="16554" ;;
      emulator-16556|emulator-5556) console_port="16556" ;;
    esac
  elif [[ "${raw_args[$index]}" == "emu" ]]; then
    emu_index="$index"
  fi
done

if [[ -n "$console_port" && -n "$emu_index" && -f "$CONSOLE_AUTH_TOKEN_FILE" ]]; then
  console_command="${raw_args[*]:$((emu_index + 1))}"
  console_output="$({
    printf 'auth %s\n' "$(<"$CONSOLE_AUTH_TOKEN_FILE")"
    printf '%s\n' "$console_command"
    printf 'quit\n'
  } | nc -w 10 127.0.0.1 "$console_port")"
  printf '%s\n' "$console_output"
  if grep -Eq '(^|[[:space:]])KO([[:space:]]|$)' <<<"$console_output"; then
    exit 1
  fi
  exit 0
fi

args=()
is_devices=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -P)
      shift 2
      ;;
    -P*)
      shift
      ;;
    emulator-16554)
      args+=("emulator-5554")
      shift
      ;;
    emulator-16556)
      args+=("emulator-5556")
      shift
      ;;
    devices)
      args+=("$1")
      is_devices=1
      shift
      ;;
    kill-server)
      exit 0
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done

args=("-P" "$REMOTE_ADB_PORT" "${args[@]}")
if [[ "$is_devices" == "1" ]]; then
  "$REAL_ADB" "${args[@]}" \
    | sed -e 's/^emulator-5554/emulator-16554/' -e 's/^emulator-5556/emulator-16556/'
  exit "${PIPESTATUS[0]}"
fi

exec "$REAL_ADB" "${args[@]}"
