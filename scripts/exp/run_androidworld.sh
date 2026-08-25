#!/usr/bin/env bash

set -e

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

for runtime_env in "$repo/config/runtime.env" "$repo/config/runtime.secrets.env"; do
  if [[ -f "$runtime_env" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$runtime_env"
    set +a
  fi
done

export OMNIFLOW_ANDROIDWORLD_CONTROL_BACKEND="oob"
export PYTHONPATH="$repo:$repo/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON_BIN:-$repo/.venv/bin/python}" -m src.experiment.run_tasks "$@"
