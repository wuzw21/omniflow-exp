#!/usr/bin/env bash

set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
env_file="${OMNIFLOW_ENV_FILE:-$repo/config/9207_mobilegpt.env}"
if [[ -n "$env_file" && "$env_file" != /* ]]; then
  echo "OMNIFLOW_ENV_FILE must be an absolute path: $env_file" >&2
  exit 2
fi
if [[ -n "$env_file" && -f "$env_file" ]]; then
  set -a
  source "$env_file"
  set +a
fi

# AndroidWorld physical I/O is a hard OOB contract for this checkout.
export OMNIFLOW_ANDROIDWORLD_CONTROL_BACKEND="oob"

python_bin="${PYTHON_BIN:-$repo/.venv/bin/python}"
if [[ "$python_bin" != /* || ! -x "$python_bin" ]]; then
  echo "Python runtime missing: $python_bin" >&2
  exit 2
fi

args=()
while (($#)); do
  case "$1" in
    --config)
      export OMNIFLOW_ANDROIDWORLD_CONFIG="$2"
      shift 2
      ;;
    --e2e-task)
      args+=(--task "$2")
      shift 2
      ;;
    --e2e-method)
      args+=(--method "$2")
      shift 2
      ;;
    --e2e-device)
      args+=(--device "$2")
      shift 2
      ;;
    --e2e-source-seed)
      args+=(--source-seed "$2")
      shift 2
      ;;
    --e2e-evaluation-seed)
      args+=(--evaluation-seed "$2")
      shift 2
      ;;
    --control-backend)
      if [[ "$2" != "oob" ]]; then
        echo "AndroidWorld requires the OOB control backend: $2" >&2
        exit 2
      fi
      shift 2
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done

export PYTHONPATH="$repo:$repo/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" -m src.experiment.run_tasks "${args[@]}"
