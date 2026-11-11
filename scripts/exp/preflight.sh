#!/usr/bin/env bash
# Pre-flight check before running any AndroidWorld cell.
#
# Every gate below is a defect that was actually hit on 2026-11-06: the default
# --device value was unusable, the kernel sealed episodes at 600s regardless of
# the protocol, and a missing OmniTransfer checkout surfaced only as a mid-run
# exception.  Running this first turns those into one clear line each instead of
# a wasted batch.
#
# Exit code 0 means: every remaining risk is a device/AVD risk, not a config one.
set -u

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo" || exit 2
python="${PYTHON_BIN:-$repo/.venv/bin/python}"
fail=0

note() { printf '%-34s %s\n' "$1" "$2"; }
ok()   { note "$1" "OK   $2"; }
bad()  { note "$1" "FAIL $2"; fail=1; }

# 1. Interpreter
if [[ -x "$python" ]] && "$python" -c 'import sys' 2>/dev/null; then
  ok "interpreter" "$python ($("$python" -V 2>&1 | cut -d' ' -f2))"
else
  bad "interpreter" "missing: $python (set PYTHON_BIN to override)"
fi

# 2. Imports the launcher cannot start without
if "$python" - <<'PY' >/dev/null 2>&1
import openai, PIL, numpy, pandas, cv2, absl, requests, json_repair
from android_world import registry
PY
then
  ok "python deps" "openai PIL numpy pandas cv2 absl json_repair"
else
  missing=$("$python" - <<'PY' 2>&1 | tail -1
import importlib
for name in ("openai","PIL","numpy","pandas","cv2","absl","requests","json_repair"):
    try: importlib.import_module(name)
    except ImportError as e: print(e); break
else:
    try:
        import android_world  # noqa: F401
    except ImportError as e:
        print(e)
PY
)
  bad "python deps" "$missing"
fi

# 3. android_world must be the integrated build, not a bare release
if "$python" -c 'import android_world, pathlib; print(pathlib.Path(android_world.__file__).parent)' >/dev/null 2>&1; then
  task_count=$("$python" - <<'PY' 2>/dev/null | tail -1
from android_world import registry
print(len(registry.TaskRegistry().get_registry("android_world")))
PY
)
  if [[ "${task_count:-0}" == "116" ]]; then
    ok "android_world" "116 tasks"
  else
    bad "android_world" "expected 116 tasks, got ${task_count:-none}"
  fi
else
  bad "android_world" "not importable (install data/runtime/releases/experiment-ready/*.whl)"
fi

# 4. OmniTransfer is a separate checkout; without it every Function transfer
#    fails mid-episode with omnitransfer_root_missing.
transfer_root="${OMNIFLOW_OMNITRANSFER_ROOT:-$HOME/Projects/Omni/OmniTransfer}"
if [[ -d "$transfer_root" ]]; then
  ok "OmniTransfer" "$transfer_root"
else
  bad "OmniTransfer" "missing: $transfer_root (set OMNIFLOW_OMNITRANSFER_ROOT)"
fi

# 5. adb + a target device
adb="${OMNIFLOW_ADB_PATH:-$(command -v adb || true)}"
if [[ -n "$adb" && -x "$adb" ]]; then
  devices="$("$adb" devices 2>/dev/null | awk 'NR>1 && $2=="device"{print $1}' | paste -sd, -)"
  if [[ -n "$devices" ]]; then
    ok "adb devices" "$devices"
  else
    bad "adb devices" "none online (start the AVD first)"
  fi
else
  bad "adb" "not found (set OMNIFLOW_ADB_PATH)"
fi

# 6. Protocol budget must be reachable
if budget=$("$python" - <<'PY' 2>&1 | tail -1
from src.experiment.protocol import EPISODE_TIMEOUT_SEC, TASK_DEADLINE_SEC, DEFAULT_DEVICE_LABEL
assert EPISODE_TIMEOUT_SEC <= TASK_DEADLINE_SEC, "episode budget exceeds deadline"
print(f"{int(EPISODE_TIMEOUT_SEC)}s<={TASK_DEADLINE_SEC}s, device={DEFAULT_DEVICE_LABEL}")
PY
); then
  ok "protocol budget" "$budget"
else
  bad "protocol budget" "$budget"
fi

# 7. The baseline under test must resolve to a real mapper
if mapper=$("$python" - <<'PY' 2>&1 | tail -1
import os
os.environ.setdefault("OMNIFLOW_BASELINE", "mobimem_style")
from src.integrations.android_world.agent import same_harness_baseline, _baseline_transfer
name = same_harness_baseline()
transfer = _baseline_transfer(name)
assert name == "none" or callable(transfer), f"{name} has no mapper"
print(f"{name} -> {'builtin' if transfer is None else transfer.__qualname__}")
PY
); then
  ok "baseline mapper" "$mapper"
else
  bad "baseline mapper" "$mapper"
fi

echo
if [[ $fail -eq 0 ]]; then
  echo "ready: run the cell"
  exit 0
fi
echo "not ready: fix the FAIL lines above before spending device time"
exit 1
