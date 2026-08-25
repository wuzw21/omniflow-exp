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
if [[ -z "${OPENAI_API_KEY:-}" && -n "${LLMTHU_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="$LLMTHU_API_KEY"
fi

export OMNIFLOW_ANDROIDWORLD_CONTROL_BACKEND="oob"
export PYTHONPATH="$repo:$repo/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON_BIN:-$repo/.venv/bin/python}" -m src.experiment.run_tasks "$@"
