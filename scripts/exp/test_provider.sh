#!/usr/bin/env bash

set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
default_python="$repo/.venv/bin/python"
python_bin="${PYTHON_BIN:-$default_python}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/exp/test_provider.sh mobilegpt
  bash scripts/exp/test_provider.sh appagent
  bash scripts/exp/test_provider.sh all

Run the offline contract tests for one provider. This command does not start
an emulator, call a model, create memory, or replace the formal launcher.
Set PYTHON_BIN only when using an explicitly provisioned equivalent runtime.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

provider="${1:-}"
if [[ -z "$provider" ]]; then
  usage >&2
  exit 2
fi

if [[ "$python_bin" != /* || ! -x "$python_bin" ]]; then
  echo "Python runtime missing: set PYTHON_BIN to one absolute executable (default: $default_python)." >&2
  exit 2
fi

run_provider() {
  local name="$1"
  local module
  local keyword
  local -a tests

  case "$name" in
    mobilegpt)
      module="src.experiment.mobilegpt_source"
      keyword="mobilegpt"
      if [[ -z "${MOBILEGPT_TEST_ROOT:-}" && -n "${OMNIFLOW_MOBILEGPT_ROOT:-}" ]]; then
        export MOBILEGPT_TEST_ROOT="$OMNIFLOW_MOBILEGPT_ROOT"
      fi
      tests=(
        "$repo/tests/test_mobilegpt_source.py"
        "$repo/tests/test_mobilegpt_converter.py"
      )
      ;;
    appagent)
      module="src.experiment.appagent_source"
      keyword="appagent"
      tests=("$repo/tests/test_appagent_source.py")
      ;;
    *)
      echo "Unknown provider: $name (choose mobilegpt, appagent, or all)." >&2
      exit 2
      ;;
  esac

  echo "[provider:$name] focused tests"
  PYTHONPATH="$repo:$repo/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$python_bin" -m pytest -q "${tests[@]}"

  echo "[provider:$name] shell integration tests"
  PYTHONPATH="$repo:$repo/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$python_bin" -m pytest -q "$repo/tests/test_exp_script.py" -k "$keyword"

  echo "[provider:$name] CLI entry"
  PYTHONPATH="$repo:$repo/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$python_bin" -m "$module" --help >/dev/null
}

case "$provider" in
  mobilegpt|appagent)
    run_provider "$provider"
    ;;
  all)
    run_provider mobilegpt
    run_provider appagent
    ;;
  *)
    echo "Unknown provider: $provider (choose mobilegpt, appagent, or all)." >&2
    usage >&2
    exit 2
    ;;
esac
