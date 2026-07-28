#!/usr/bin/env bash

set -euo pipefail

ADB_BIN="${OMNIFLOW_DOCKER_ADB_BIN:-${ANDROID_HOME:-$HOME/Android/Sdk}/platform-tools/adb}"
SOURCE_ALIAS="${OMNIFLOW_DOCKER_SOURCE_ALIAS:-emulator-16554}"
SOURCE_ENDPOINT="${OMNIFLOW_DOCKER_SOURCE_ENDPOINT:-127.0.0.1:16555}"
TARGET_ALIAS="${OMNIFLOW_DOCKER_TARGET_ALIAS:-emulator-16556}"
TARGET_ENDPOINT="${OMNIFLOW_DOCKER_TARGET_ENDPOINT:-127.0.0.1:16557}"

if [ ! -x "$ADB_BIN" ]; then
  ADB_BIN="$(command -v adb || true)"
fi
if [ -z "$ADB_BIN" ] || [ ! -x "$ADB_BIN" ]; then
  echo "[error] adb is not executable; set OMNIFLOW_DOCKER_ADB_BIN" >&2
  exit 127
fi

rewrite_serial() {
  case "$1" in
    "$SOURCE_ALIAS") printf '%s\n' "$SOURCE_ENDPOINT" ;;
    "$TARGET_ALIAS") printf '%s\n' "$TARGET_ENDPOINT" ;;
    *) printf '%s\n' "$1" ;;
  esac
}

if [ -n "${ANDROID_SERIAL:-}" ]; then
  ANDROID_SERIAL="$(rewrite_serial "$ANDROID_SERIAL")"
  export ANDROID_SERIAL
fi

args=("$@")
for ((index = 0; index < ${#args[@]}; index += 1)); do
  if [ "${args[$index]}" = "-s" ] && [ $((index + 1)) -lt ${#args[@]} ]; then
    args[$((index + 1))]="$(rewrite_serial "${args[$((index + 1))]}")"
    index=$((index + 1))
  fi
done

if [ "${args[0]:-}" = "devices" ]; then
  output="$($ADB_BIN "${args[@]}")"
  output="${output//$SOURCE_ENDPOINT/$SOURCE_ALIAS}"
  output="${output//$TARGET_ENDPOINT/$TARGET_ALIAS}"
  printf '%s\n' "$output"
  exit 0
fi

exec "$ADB_BIN" "${args[@]}"
