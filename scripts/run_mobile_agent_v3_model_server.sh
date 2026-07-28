#!/usr/bin/env bash

set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vllm_bin="${OMNIFLOW_MOBILE_AGENT_V3_VLLM_BIN:-/home/wuzewen/miniconda3/envs/quant/bin/vllm}"
audit_python="${PYTHON_BIN:-/home/wuzewen/Projects/Omni/OmniFlow/.venv/bin/python}"
model_revision="${OMNIFLOW_MOBILE_AGENT_V3_MODEL_REVISION:-7c1644c0288da07435a485701d0fea0ac353f38a}"
model_root="${OMNIFLOW_MOBILE_AGENT_V3_MODEL_ROOT:-/home/wuzewen/models/GUI-Owl-7B-$model_revision}"
served_model="${OMNIFLOW_MOBILE_AGENT_V3_MODEL:-GUI-Owl-7B}"
host="${OMNIFLOW_MOBILE_AGENT_V3_HOST:-127.0.0.1}"
port="${OMNIFLOW_MOBILE_AGENT_V3_PORT:-4243}"
cuda_devices="${OMNIFLOW_MOBILE_AGENT_V3_CUDA_VISIBLE_DEVICES:-0,1}"
minimum_free_mib="${OMNIFLOW_MOBILE_AGENT_V3_MINIMUM_FREE_MIB:-20000}"

if [[ ! -x "$vllm_bin" ]]; then
  echo "vLLM executable missing: $vllm_bin" >&2
  exit 1
fi
if [[ ! -x "$audit_python" ]]; then
  echo "Audit Python missing: $audit_python" >&2
  exit 1
fi

PYTHONPATH="$repo:$repo/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$audit_python" - "$model_root" "$model_revision" <<'PY'
import json
import sys
from src.integrations.mobile_agent_v3_adapter import inspect_gui_owl_model

print(json.dumps(inspect_gui_owl_model(sys.argv[1], revision=sys.argv[2]), sort_keys=True))
PY

"$audit_python" - "$host" "$port" <<'PY'
import socket
import sys

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind((sys.argv[1], int(sys.argv[2])))
except OSError as error:
    raise SystemExit(f"GUI-Owl server port is already occupied: {error}")
finally:
    sock.close()
PY

IFS=',' read -r -a requested_gpus <<< "$cuda_devices"
mapfile -t gpu_free < <(
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits
)
for requested_gpu in "${requested_gpus[@]}"; do
  free_mib=""
  for row in "${gpu_free[@]}"; do
    index="${row%%,*}"
    value="${row#*,}"
    index="${index//[[:space:]]/}"
    value="${value//[[:space:]]/}"
    if [[ "$index" == "$requested_gpu" ]]; then
      free_mib="$value"
      break
    fi
  done
  if [[ -z "$free_mib" || "$free_mib" -lt "$minimum_free_mib" ]]; then
    echo "GPU $requested_gpu has ${free_mib:-unknown} MiB free; require $minimum_free_mib MiB." >&2
    exit 1
  fi
  echo "[gpu] index=$requested_gpu free_mib=$free_mib"
done

echo "[model] GUI-Owl-7B revision=$model_revision endpoint=http://$host:$port/v1 model=$served_model"
exec env CUDA_VISIBLE_DEVICES="$cuda_devices" "$vllm_bin" serve "$model_root" \
  --host "$host" \
  --port "$port" \
  --served-model-name "$served_model" \
  --max-model-len 32768 \
  --tensor-parallel-size 2 \
  --allowed-local-media-path / \
  --mm-processor-kwargs '{"min_pixels":3136,"max_pixels":10035200}' \
  --limit-mm-per-prompt '{"image":2}'
