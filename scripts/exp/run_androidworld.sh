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

if [[ -z "${OPENAI_API_KEY:-}" && -n "${LLMTHU_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="$LLMTHU_API_KEY"
fi

export OMNIFLOW_ANDROIDWORLD_CONTROL_BACKEND="oob"
# The experiment has one OOB release artifact.  Publish its repository-
# relative location at the only process boundary so AndroidWorld lifecycle
# code, baselines, and OmniFlow all install/use the same physical layer.
export OMNIFLOW_OOB_APK="$repo/data/runtime/oob/OOB-Experiment.apk"
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
