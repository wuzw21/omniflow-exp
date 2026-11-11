#!/usr/bin/env bash

# One-time host preparation for the pinned official MobileGPT client/server.
# Normal experiments continue to use run_androidworld.sh and never rebuild this
# environment.  Secrets are read from config/runtime.secrets.env only.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
for runtime_env in "$repo/config/runtime.env" "$repo/config/runtime.secrets.env"; do
  if [[ -f "$runtime_env" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$runtime_env"
    set +a
  fi
done

mobilegpt_root="${OMNIFLOW_MOBILEGPT_ROOT:-$repo/../mobilegpt-official-clean}"
for _mobilegpt_sibling in \
  "$repo/../mobilegpt-official-clean" \
  "$repo/../Omni/mobilegpt-official-clean"; do
  if [[ -d "$_mobilegpt_sibling/App" && -d "$_mobilegpt_sibling/Server" ]]; then
    mobilegpt_root="$_mobilegpt_sibling"
    break
  fi
done
runtime_root="${OMNIFLOW_MOBILEGPT_RUNTIME_ROOT:-$repo/data/runtime/mobilegpt}"
apk_path="${OMNIFLOW_MOBILEGPT_APK:-$runtime_root/client.apk}"
python_bin="${PYTHON_BIN:-$repo/.venv/bin/python}"
gradle_bin="${OMNIFLOW_GRADLE_BIN:-}"

if [[ ! -d "$mobilegpt_root/App" || ! -d "$mobilegpt_root/Server" ]]; then
  echo "official MobileGPT checkout missing: $mobilegpt_root" >&2
  exit 2
fi
if [[ ! -x "$python_bin" ]]; then
  echo "canonical Python missing: $python_bin" >&2
  exit 2
fi

export PYTHONPATH="$repo:$repo/src${PYTHONPATH:+:$PYTHONPATH}"
export ANDROID_HOME="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-}}"
export ANDROID_SDK_ROOT="${ANDROID_SDK_ROOT:-$ANDROID_HOME}"
export OMNIFLOW_MOBILEGPT_RUNTIME_ROOT="$runtime_root"
export MOBILEGPT_ROOT_FOR_SETUP="$mobilegpt_root"
export MOBILEGPT_APK_FOR_SETUP="$apk_path"
export MOBILEGPT_GRADLE_FOR_SETUP="$gradle_bin"

"$python_bin" - <<'PY'
import os
from pathlib import Path

from src.integrations.official_forward import (
    prepare_mobilegpt_client_apk,
    prepare_mobilegpt_runtime,
)

official_root = Path(os.environ["MOBILEGPT_ROOT_FOR_SETUP"])
runtime_root = Path(os.environ["OMNIFLOW_MOBILEGPT_RUNTIME_ROOT"])
prepare_mobilegpt_runtime(
    official_root=official_root,
    runtime_root=runtime_root,
    embedding_model="GLM-Embedding-2",
    chat_model="Qwen3.6-Plus",
)
for label, port in (
    ("source5560", 5560),
    ("standard45562", 45562),
    ("fold45564", 45564),
    ("tablet45554", 45554),
):
    prepare_mobilegpt_client_apk(
        official_root=official_root,
        output_apk=runtime_root / f"client_{label}.apk",
        host="10.0.2.2",
        port=12345 if port == 5560 else 12000 + port % 40000,
        gradle_bin=os.environ.get("MOBILEGPT_GRADLE_FOR_SETUP", ""),
    )
source_apk = runtime_root / "client_source5560.apk"
fallback_apk = Path(os.environ["MOBILEGPT_APK_FOR_SETUP"])
if not fallback_apk.is_file():
    fallback_apk.parent.mkdir(parents=True, exist_ok=True)
    fallback_apk.write_bytes(source_apk.read_bytes())
print(f"prepared MobileGPT runtime: {runtime_root}")
PY

adb_bin="${ADB_PATH:-${ANDROID_SDK_ROOT:-$ANDROID_HOME}/platform-tools/adb}"
if [[ -x "$adb_bin" ]]; then
  for label_serial in \
    source5560:emulator-5560 \
    standard45562:emulator-45562 \
    fold45564:emulator-45564 \
    tablet45554:emulator-45554; do
    label="${label_serial%%:*}"
    serial="${label_serial#*:}"
    if [[ "$("$adb_bin" -s "$serial" get-state 2>/dev/null || true)" != "device" ]]; then
      continue
    fi
    if [[ "${MOBILEGPT_FORCE_INSTALL:-0}" != "1" ]] && \
      "$adb_bin" -s "$serial" shell pm path com.example.MobileGPT >/dev/null 2>&1; then
      echo "reusing MobileGPT client: $label ($serial)"
      continue
    fi
    # MobileGPT is a pipeline-owned package. Replace only this package so the
    # installed client matches the one-time port-specific APK above.
    "$adb_bin" -s "$serial" shell pm uninstall --user 0 com.example.MobileGPT >/dev/null 2>&1 || true
    "$adb_bin" -s "$serial" install -r "$runtime_root/client_${label}.apk" >/dev/null
    echo "installed MobileGPT client: $label ($serial)"
  done
fi
