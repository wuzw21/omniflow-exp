#!/usr/bin/env bash

set -e

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

for runtime_env in \
  "$repo/config/.env" \
  "$repo/config/runtime.env" \
  "$repo/config/runtime.secrets.env"; do
  if [[ -f "$runtime_env" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$runtime_env"
    set +a
  fi
done

if [[ -z "${OPENAI_API_KEY:-}" && -n "${LLMTHU_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="$LLMTHU_API_KEY"
fi

control_backend="${OMNIFLOW_ANDROIDWORLD_CONTROL_BACKEND:-oob}"
case "$control_backend" in
  oob|omniflow|oob_control)
    export OMNIFLOW_ANDROIDWORLD_CONTROL_BACKEND="oob"
    ;;
  *)
    echo "AndroidWorld requires the canonical OOB control backend; got: $control_backend" >&2
    exit 2
    ;;
esac
# Canonical post-action policy: a 0.5-second optimistic transition window,
# followed by one generic stable observation only when Transfer admission
# rejects the fast state. Keep these overridable for controlled ablations.
export OMNIFLOW_FAST_PASS="${OMNIFLOW_FAST_PASS:-1}"
export OMNIFLOW_ANDROIDWORLD_WAIT_TO_STABILIZE="${OMNIFLOW_ANDROIDWORLD_WAIT_TO_STABILIZE:-0}"
export OMNIFLOW_ANDROIDWORLD_ACT_AWAIT_STABILIZATION="${OMNIFLOW_ANDROIDWORLD_ACT_AWAIT_STABILIZATION:-0}"
export OMNIFLOW_ANDROIDWORLD_POST_ACTION_TRANSITION_MS="${OMNIFLOW_ANDROIDWORLD_POST_ACTION_TRANSITION_MS:-500}"
# The experiment has one OOB release artifact.  Publish its repository-
# relative location at the only process boundary so AndroidWorld lifecycle
# code, baselines, and OmniFlow all install/use the same physical layer.
export OMNIFLOW_OOB_APK="$repo/data/runtime/oob/OOB-Experiment.apk"
# MobileGPT is prepared once outside the experiment runner.  Its checkout is
# supplied by the explicit runtime configuration; never guess a sibling
# checkout, because that can silently run a different version.
export OMNIFLOW_MOBILEGPT_RUNTIME_ROOT="${OMNIFLOW_MOBILEGPT_RUNTIME_ROOT:-$repo/data/runtime/mobilegpt}"
export OMNIFLOW_MOBILEGPT_APK="${OMNIFLOW_MOBILEGPT_APK:-$OMNIFLOW_MOBILEGPT_RUNTIME_ROOT/client.apk}"
export OMNIFLOW_MOBILEGPT_REBUILD_CLIENT="0"
export PYTHONPATH="$repo:$repo/src${PYTHONPATH:+:$PYTHONPATH}"
# Keep runtime helper binaries (notably ffmpeg used by audio/Retro tasks)
# discoverable alongside the canonical experiment Python environment.
export PATH="$repo/.venv/bin:${PATH}"
# AndroidWorld generates short test MP3s through pydub.  The canonical venv
# carries imageio-ffmpeg's pinned binary, while the host image may not expose a
# system ``ffmpeg`` command; point pydub at that binary when available.
if [[ -z "${FFMPEG_BINARY:-}" ]]; then
  FFMPEG_BINARY="$(${PYTHON_BIN:-$repo/.venv/bin/python} -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null || true)"
  if [[ -n "$FFMPEG_BINARY" ]]; then
    export FFMPEG_BINARY
    # pydub resolves the encoder through PATH (it does not honor
    # FFMPEG_BINARY in the versions used on 9207). Expose the bundled
    # imageio-ffmpeg binary under the expected name without mutating the
    # repository or the virtualenv.
    _ffmpeg_shim_dir="${TMPDIR:-/tmp}/omniflow-ffmpeg-bin"
    mkdir -p "$_ffmpeg_shim_dir"
    ln -sf "$FFMPEG_BINARY" "$_ffmpeg_shim_dir/ffmpeg"
    export PATH="$_ffmpeg_shim_dir:$PATH"
  fi
fi
exec "${PYTHON_BIN:-$repo/.venv/bin/python}" -m src.experiment.run_tasks "$@"
