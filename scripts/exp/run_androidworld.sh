#!/usr/bin/env bash

set -e

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OMNIFLOW_ANDROIDWORLD_CONTROL_BACKEND="oob"
export PYTHONPATH="$repo:$repo/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON_BIN:-$repo/.venv/bin/python}" -m src.experiment.run_tasks "$@"
