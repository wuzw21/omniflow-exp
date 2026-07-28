#!/usr/bin/env bash

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
FAILURES=0
WARNINGS=0

section() {
  printf '\n== %s ==\n' "$1"
}

ok() {
  printf '[ok] %s\n' "$1"
}

warn() {
  WARNINGS=$((WARNINGS + 1))
  printf '[warn] %s\n' "$1"
}

fail() {
  FAILURES=$((FAILURES + 1))
  printf '[fail] %s\n' "$1"
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

check_file() {
  if [ -f "$ROOT_DIR/$1" ]; then
    ok "$1"
  else
    fail "missing $1"
  fi
}

check_dir() {
  if [ -d "$ROOT_DIR/$1" ]; then
    ok "$1"
  else
    fail "missing $1"
  fi
}

resolve_tool() {
  local binary="$1"
  local subdir="$2"
  local candidate=""

  if have_cmd "$binary"; then
    command -v "$binary"
    return 0
  fi

  for candidate in \
    "${ANDROID_HOME:-}/$subdir/$binary" \
    "${ANDROID_SDK_ROOT:-}/$subdir/$binary" \
    "$HOME/Android/Sdk/$subdir/$binary" \
    "$HOME/Library/Android/sdk/$subdir/$binary"
  do
    if [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

section "Repository"
cd "$ROOT_DIR" || exit 2
check_file "pyproject.toml"
check_file "androidworld.md"
check_file "scripts/androidworld_replay_pipeline.py"
check_file "scripts/start_androidworld_avds.sh"

section "Python"
if have_cmd "$PYTHON_BIN"; then
  ok "$PYTHON_BIN: $("$PYTHON_BIN" --version 2>&1)"
  if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import sys
if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11+ is recommended for AndroidWorld experiments")
PY
  then
    ok "Python version is suitable"
  else
    warn "Python is older than the recommended AndroidWorld version"
  fi

  for module in omniflow dotenv openai uiautomator2; do
    if "$PYTHON_BIN" - <<PY >/dev/null 2>&1
import importlib
importlib.import_module("$module")
PY
    then
      ok "python module: $module"
    else
      warn "python module not importable: $module"
    fi
  done
else
  fail "python executable not found: $PYTHON_BIN"
fi

section "Model Env"
if [ -f "$ROOT_DIR/.env" ]; then
  ok ".env exists"
  grep -Eq '^[[:space:]]*OPENAI_API_KEY[[:space:]]*=' "$ROOT_DIR/.env" \
    && ok ".env has OPENAI_API_KEY" \
    || warn ".env missing OPENAI_API_KEY"
  grep -Eq '^[[:space:]]*OPENAI_BASE_URL[[:space:]]*=' "$ROOT_DIR/.env" \
    && ok ".env has OPENAI_BASE_URL" \
    || warn ".env missing OPENAI_BASE_URL"
  grep -Eq '^[[:space:]]*(OPENAI_MODEL|OMNIFLOW_PLANNER_MODEL)[[:space:]]*=' "$ROOT_DIR/.env" \
    && ok ".env has model name" \
    || warn ".env missing OPENAI_MODEL or OMNIFLOW_PLANNER_MODEL"
else
  warn ".env not found; model-backed methods need OPENAI_API_KEY and model config"
fi

section "Android SDK"
ADB_BIN="$(resolve_tool adb platform-tools || true)"
EMU_BIN="$(resolve_tool emulator emulator || true)"

if [ -n "$ADB_BIN" ]; then
  ok "adb: $ADB_BIN"
  "$ADB_BIN" devices || warn "adb devices failed"
else
  fail "adb not found; set ANDROID_HOME/ANDROID_SDK_ROOT or PATH"
fi

if [ -n "$EMU_BIN" ]; then
  ok "emulator: $EMU_BIN"
  AVD_LIST="$("$EMU_BIN" -list-avds 2>/dev/null || true)"
  if [ -n "$AVD_LIST" ]; then
    printf '%s\n' "$AVD_LIST" | sed 's/^/[avd] /'
    printf '%s\n' "$AVD_LIST" | grep -Fxq "AndroidWorldAvd" \
      && ok "AVD present: AndroidWorldAvd" \
      || warn "AVD missing: AndroidWorldAvd"
    printf '%s\n' "$AVD_LIST" | grep -Fxq "SmallPhone" \
      && ok "AVD present: SmallPhone" \
      || warn "AVD missing: SmallPhone"
  else
    warn "no AVDs listed by emulator"
  fi
else
  fail "emulator not found; set ANDROID_HOME/ANDROID_SDK_ROOT or PATH"
fi

if [ "$(uname -s)" = "Linux" ]; then
  if [ -e /dev/kvm ]; then
    if [ -r /dev/kvm ] && [ -w /dev/kvm ]; then
      ok "/dev/kvm is readable/writable"
    else
      fail "/dev/kvm exists but is not readable/writable by this user"
    fi
  else
    fail "/dev/kvm missing; Android emulator acceleration is not available"
  fi
fi

if have_cmd java; then
  JAVA_VERSION="$(java -version 2>&1 | head -n 1 || true)"
  if java -version >/dev/null 2>&1; then
    ok "java: $JAVA_VERSION"
  else
    warn "java command exists but no usable runtime: $JAVA_VERSION"
  fi
else
  warn "java not found; MobileGPT Android app build may fail"
fi

section "External Frameworks"
check_dir "runtime/external/droidrun-android-world"
check_file "runtime/external/droidrun-android-world/android_world/requirements.txt"
check_dir "runtime/external/mobilegpt"
check_file "runtime/external/mobilegpt/Server/requirements.txt"
check_file "runtime/external/mobilegpt/App/gradlew"

if [ -d "$ROOT_DIR/runtime/external/autodroid_official_newbranch" ]; then
  ok "runtime/external/autodroid_official_newbranch"
else
  warn "AutoDroid tree missing; this is optional for the current plan"
fi

section "Experiment Data"
check_file "runtime/evals/androidworld_validator/core_archive/success_source_runlogs/index_by_task.json"
check_file "runtime/evals/androidworld_validator/master_progress/androidworld_method_matrix.csv"
if [ -d "$ROOT_DIR/runtime/evals/androidworld_validator/runs" ]; then
  ok "runtime/evals/androidworld_validator/runs"
else
  warn "runtime/evals/androidworld_validator/runs missing; it will be created by runs"
fi

section "OOB Device Host"
OOB_URL="${OMNIFLOW_OOB_DEVICE_URL:-http://127.0.0.1:8910}"
if have_cmd curl; then
  if curl -fsS --max-time 3 "$OOB_URL/health" >/dev/null 2>&1; then
    ok "OOB health: $OOB_URL/health"
  else
    warn "OOB health not reachable at $OOB_URL/health"
  fi
else
  warn "curl not found; skipping OOB health check"
fi

section "Summary"
printf 'failures=%s warnings=%s\n' "$FAILURES" "$WARNINGS"

if [ "$FAILURES" -gt 0 ]; then
  exit 1
fi
exit 0
