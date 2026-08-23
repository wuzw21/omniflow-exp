#!/usr/bin/env bash
set -euo pipefail

real_adb="${OMNIFLOW_REAL_ADB_PATH:-${ANDROID_SDK_ROOT:-}/platform-tools/adb}"
if [[ ! -x "$real_adb" ]]; then
  echo "autodroid_isolated_adb_real_binary_missing:$real_adb" >&2
  exit 127
fi

args=("$@")
server_port="${ANDROID_ADB_SERVER_PORT:-5038}"
has_server_port=0
for index in "${!args[@]}"; do
  if [[ "${args[$index]}" == "-P" && $((index + 1)) -lt ${#args[@]} ]]; then
    args[$((index + 1))]="$server_port"
    has_server_port=1
  elif [[ "${args[$index]}" == "emulator-5590" ]]; then
    args[$index]="127.0.0.1:5595"
  fi
done

if [[ "$has_server_port" -eq 0 ]]; then
  args=("-P" "$server_port" "${args[@]}")
fi

exec "$real_adb" "${args[@]}"
