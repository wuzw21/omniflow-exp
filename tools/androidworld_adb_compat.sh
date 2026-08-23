#!/usr/bin/env bash
set -euo pipefail

# AndroidWorld's current adb_utils always adds
# --bypass-low-target-sdk-block to install.  API 33 system images used by the
# formal 4090 devices reject that option; keep the real adb contract and only
# remove this unsupported install flag.
real_adb="${OMNIFLOW_REAL_ADB_PATH:-${ANDROID_SDK_ROOT:-}/platform-tools/adb}"
if [[ ! -x "$real_adb" ]]; then
  # The unified runner normally exports the SDK-selected adb explicitly.
  # Keep the wrapper usable for child processes that only inherit PATH; this
  # is still the real adb binary and does not require sudo or a second backend.
  real_adb="$(command -v adb || true)"
fi
if [[ ! -x "$real_adb" ]]; then
  echo "androidworld_adb_real_binary_missing:$real_adb" >&2
  exit 127
fi

args=("$@")
has_install=0
for arg in "${args[@]}"; do
  if [[ "$arg" == "install" ]]; then
    has_install=1
    break
  fi
done
if [[ "$has_install" == "1" ]]; then
  filtered=()
  for arg in "${args[@]}"; do
    [[ "$arg" == "--bypass-low-target-sdk-block" ]] && continue
    filtered+=("$arg")
  done
  args=("${filtered[@]}")
fi

exec "$real_adb" "${args[@]}"
