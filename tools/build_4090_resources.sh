#!/usr/bin/env bash

# Build a complete OmniFlow experiment machine over SSH, or continue the
# build directly on the remote machine with --remote.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash tools/build_4090_resources.sh --ssh user@host [options]
  bash tools/build_4090_resources.sh --remote [options]

The default mode is latest: external repositories are updated to their
default-branch HEAD and the exact deployed commits are recorded in
<remote-root>/deployment_manifest.json. Python dependencies remain locked by
uv.lock unless --upgrade-python-deps is supplied.

Options:
  --ssh HOST                 SSH destination, e.g. user@4090
  --remote                   Run the build on the current Linux host
  --remote-root DIR          Remote bundle root (default: /data/omniflow-4090)
  --source-data DIR          Local authoritative data directory (default: repo/data)
  --model-env FILE           Local model.env to copy to the remote host
  --mode latest              External repo policy (default: latest)
  --archive-data             Also transfer a complete data archive
  --skip-data                Do not transfer authoritative data
  --skip-system-bootstrap    Do not use apt/sudo or install uv/Node/SDK
  --user-bootstrap            Skip apt/sudo, but install missing user-local tools
  --skip-device-setup        Do not create/configure Android devices
  --skip-bmoca               Do not clone/install B-MoCA dependencies
  --upgrade-python-deps      Run uv lock --upgrade before uv sync
  --run-smoke                Run CameraTakePhoto omniflow after validation
  --reuse-remote             Permit an existing remote bundle root
  --help                     Show this help

Example:
  bash tools/build_4090_resources.sh \
    --ssh user@4090 --model-env /secure/model.env --run-smoke
EOF
}

die() { echo "ERROR: $*" >&2; exit 2; }
log() { printf '[4090] %s\n' "$*"; }

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ssh_host=""
remote=0
remote_root="/data/omniflow-4090"
source_data="$repo/data"
model_env=""
mode="latest"
archive_data=0
skip_data=0
skip_system_bootstrap=0
user_bootstrap=0
skip_device_setup=0
skip_bmoca=0
upgrade_python_deps=0
run_smoke=0
reuse_remote=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ssh) [[ $# -ge 2 ]] || die "--ssh needs HOST"; ssh_host="$2"; shift 2 ;;
    --remote) remote=1; shift ;;
    --remote-root) [[ $# -ge 2 ]] || die "--remote-root needs DIR"; remote_root="$2"; shift 2 ;;
    --source-data) [[ $# -ge 2 ]] || die "--source-data needs DIR"; source_data="$2"; shift 2 ;;
    --model-env) [[ $# -ge 2 ]] || die "--model-env needs FILE"; model_env="$2"; shift 2 ;;
    --mode) [[ $# -ge 2 ]] || die "--mode needs latest"; mode="$2"; shift 2 ;;
    --archive-data) archive_data=1; shift ;;
    --skip-data) skip_data=1; shift ;;
    --skip-system-bootstrap) skip_system_bootstrap=1; shift ;;
    --user-bootstrap) user_bootstrap=1; shift ;;
    --skip-device-setup) skip_device_setup=1; shift ;;
    --skip-bmoca) skip_bmoca=1; shift ;;
    --upgrade-python-deps) upgrade_python_deps=1; shift ;;
    --run-smoke) run_smoke=1; shift ;;
    --reuse-remote) reuse_remote=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

[[ "$mode" == latest ]] || die "--mode must be latest"
if [[ "$remote" == 1 && -n "$ssh_host" ]]; then
  die "use either --ssh or --remote, not both"
fi
if [[ "$remote" == 0 && -z "$ssh_host" ]]; then
  usage >&2
  exit 2
fi

if [[ "$remote" == 0 ]]; then
  command -v ssh >/dev/null || die "ssh is required"
  command -v rsync >/dev/null || die "rsync is required"
  [[ -d "$repo" ]] || die "repository not found: $repo"
  [[ "$skip_data" == 1 || -d "$source_data" ]] || die "source data not found: $source_data"
  if [[ -n "$model_env" ]]; then
    [[ -f "$model_env" ]] || die "model env not found: $model_env"
  fi

  ssh_opts=(-o BatchMode=yes -o ConnectTimeout=15)
  ssh "${ssh_opts[@]}" "$ssh_host" true || die "cannot connect to $ssh_host with non-interactive SSH"
  if [[ "$reuse_remote" != 1 ]]; then
    if ssh "${ssh_opts[@]}" "$ssh_host" "test -e '$remote_root'"; then
      die "remote root already exists: $remote_root (use --reuse-remote after checking it)"
    fi
  fi

  log "creating remote bundle at $ssh_host:$remote_root"
  ssh "${ssh_opts[@]}" "$ssh_host" "mkdir -p '$remote_root'"
  rsync_repo_args=(-az)
  [[ "$reuse_remote" == 1 ]] || rsync_repo_args+=(--delete)
  rsync "${rsync_repo_args[@]}" \
    --exclude '.git/' --exclude '.venv/' --exclude '__pycache__/' \
    --exclude '.pytest_cache/' --exclude 'outputs/' --exclude 'data/' \
    "$repo/" "$ssh_host:$remote_root/OmniFlow-exp/"
  if [[ -n "$model_env" ]]; then
    rsync -az "$model_env" "$ssh_host:$remote_root/model.env"
    ssh "${ssh_opts[@]}" "$ssh_host" "chmod 600 '$remote_root/model.env'"
  fi

  if [[ "$skip_data" != 1 ]] && ! { [[ "$reuse_remote" == 1 ]] && ssh "${ssh_opts[@]}" "$ssh_host" "test -s '$remote_root/OmniFlow-exp/data/current.json'"; }; then
    log "migrating authoritative data and registered assets"
    bash "$repo/scripts/exp/migrate_authoritative_data.sh" \
      --source-data "$source_data" \
      --target-host "$ssh_host" \
      --target-data "$remote_root/OmniFlow-exp/data" \
      --sync
    if [[ -d "$repo/vendor/androidworld" ]]; then
      rsync -az "$repo/vendor/androidworld/" \
        "$ssh_host:$remote_root/OmniFlow-exp/vendor/androidworld/"
    fi
    if [[ -d "$repo/vendor/autodroid" ]]; then
      rsync -az "$repo/vendor/autodroid/" \
        "$ssh_host:$remote_root/OmniFlow-exp/vendor/autodroid/"
    fi
  fi
  if [[ "$archive_data" == 1 && "$skip_data" != 1 ]]; then
    rsync -az "$source_data/" "$ssh_host:$remote_root/data-archive/"
  fi

  remote_args=(--remote --remote-root "$remote_root" --mode "$mode")
  [[ "$skip_system_bootstrap" == 1 ]] && remote_args+=(--skip-system-bootstrap)
  [[ "$skip_device_setup" == 1 ]] && remote_args+=(--skip-device-setup)
  [[ "$skip_bmoca" == 1 ]] && remote_args+=(--skip-bmoca)
  [[ "$upgrade_python_deps" == 1 ]] && remote_args+=(--upgrade-python-deps)
  [[ "$run_smoke" == 1 ]] && remote_args+=(--run-smoke)
  [[ "$reuse_remote" == 1 ]] && remote_args+=(--reuse-remote)
  [[ "$user_bootstrap" == 1 ]] && remote_args+=(--user-bootstrap)
  printf -v remote_cmd '%q ' "${remote_args[@]}"
  source_commit="$(git -C "$repo" rev-parse HEAD 2>/dev/null || true)"
  source_dirty="$(git -C "$repo" status --porcelain 2>/dev/null | head -n 1)"
  printf -v source_commit_env 'OMNIFLOW_SOURCE_COMMIT=%q' "$source_commit"
  printf -v source_dirty_env 'OMNIFLOW_SOURCE_DIRTY=%q' "$([[ -n "$source_dirty" ]] && printf true || printf false)"
  ssh "${ssh_opts[@]}" "$ssh_host" \
    "cd '$remote_root/OmniFlow-exp' && $source_commit_env $source_dirty_env bash tools/build_4090_resources.sh $remote_cmd"
  exit 0
fi

[[ "$(uname -s)" == Linux ]] || die "--remote must run on Linux"
[[ "$(uname -m)" == x86_64 ]] || die "4090 build expects Linux x86_64"
repo="$remote_root/OmniFlow-exp"
[[ -d "$repo" ]] || die "remote repository missing: $repo"

if [[ -z "$model_env" ]]; then
  model_env="$remote_root/model.env"
fi
if [[ -f "$model_env" ]]; then
  chmod 600 "$model_env"
else
  log "model.env not found; continuing, but model-backed runs will need OMNIFLOW_ENV_FILE"
fi

account_home="$(getent passwd "$(id -un)" | cut -d: -f6 || true)"
account_home="${account_home:-$HOME}"
deps_root="$remote_root/deps"
android_world_root="$deps_root/android_world"
appagent_root="$deps_root/AppAgent"
mobilegpt_root="$deps_root/MobileGPT"
bmoca_root="$deps_root/B-MoCA"
android_env_parent="$deps_root/android-env"
android_env_root="$android_env_parent/android_env"
auto_ui_root="$deps_root/auto_ui"
omnitransfer_root="${OMNITRANSFER_ROOT:-$account_home/Projects/Omni/OmniTransfer}"
sdk_root="${OMNIFLOW_ANDROID_SDK_ROOT:-$account_home/Android/Sdk}"
python_bin="$repo/.venv/bin/python"

if [[ "$skip_system_bootstrap" != 1 && "$user_bootstrap" != 1 ]] && ! sudo -n true 2>/dev/null; then
  log "sudo is unavailable; switching to user-local bootstrap"
  user_bootstrap=1
fi

if [[ "$skip_system_bootstrap" != 1 && "$user_bootstrap" != 1 ]]; then
  log "installing system packages"
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git git-lfs curl wget unzip rsync build-essential pkg-config \
    python3 python3-venv openjdk-17-jdk \
    libsqlite3-dev sqlite3 adb ffmpeg libgl1 libglib2.0-0 libsm6 \
    libxext6 libxrender1
  git lfs install --skip-repo >/dev/null 2>&1 || true
fi

if ! command -v uv >/dev/null; then
  [[ "$skip_system_bootstrap" != 1 ]] || die "uv missing; remove --skip-system-bootstrap or install uv"
  log "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null || die "uv is unavailable"

if [[ "$skip_system_bootstrap" != 1 && "$skip_bmoca" != 1 ]]; then
  if [[ ! -s "$HOME/.nvm/nvm.sh" ]]; then
    log "installing Node.js 18 and Appium"
    curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
  fi
  # shellcheck disable=SC1090
  source "$HOME/.nvm/nvm.sh"
  nvm install 18.20.4
  nvm alias default 18.20.4
  npm install -g appium@2.5.4
  appium driver list --installed 2>/dev/null | grep -q uiautomator2 || appium driver install uiautomator2
fi

install_android_sdk() {
  if [[ -x "$sdk_root/cmdline-tools/latest/bin/sdkmanager" ]]; then
    return
  fi
  [[ "$skip_system_bootstrap" != 1 ]] || die "Android SDK missing at $sdk_root"
  local tools_url="${ANDROID_CMDLINE_TOOLS_URL:-https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip}"
  local tmp_dir
  tmp_dir="$(mktemp -d)"
  trap 'rm -rf "$tmp_dir"' RETURN
  mkdir -p "$sdk_root/cmdline-tools"
  curl -fL "$tools_url" -o "$tmp_dir/cmdline-tools.zip"
  unzip -q "$tmp_dir/cmdline-tools.zip" -d "$tmp_dir/unpacked"
  rm -rf "$sdk_root/cmdline-tools/latest"
  mv "$tmp_dir/unpacked/cmdline-tools" "$sdk_root/cmdline-tools/latest"
}

install_android_sdk
export ANDROID_HOME="$sdk_root"
export ANDROID_SDK_ROOT="$sdk_root"
export PATH="$sdk_root/platform-tools:$sdk_root/emulator:$sdk_root/cmdline-tools/latest/bin:$PATH"
sdkmanager_bin="$sdk_root/cmdline-tools/latest/bin/sdkmanager"
if [[ "$skip_system_bootstrap" != 1 ]]; then
  yes | "$sdkmanager_bin" --licenses >/dev/null || true
  "$sdkmanager_bin" "platform-tools" "emulator" \
    "platforms;android-33" "platforms;android-34" \
    "system-images;android-33;google_apis;x86_64" \
    "system-images;android-34;google_apis;x86_64"
fi

clone_latest() {
  local url="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  if [[ -d "$dest/.git" ]]; then
    if [[ -n "$(git -C "$dest" status --porcelain)" ]]; then
      die "external checkout is dirty; clean it before latest update: $dest"
    fi
    git -C "$dest" fetch --tags --prune origin
    local branch
    branch="$(git -C "$dest" symbolic-ref --quiet --short refs/remotes/origin/HEAD 2>/dev/null | sed 's#^origin/##' || true)"
    branch="${branch:-main}"
    git -C "$dest" checkout -B "$branch" "origin/$branch"
    git -C "$dest" reset --hard "origin/$branch" >/dev/null
  elif [[ -e "$dest" ]]; then
    die "non-git external path exists: $dest"
  else
    git clone --recurse-submodules "$url" "$dest"
  fi
}

prepare_external_repositories() {
  log "updating external repositories to default-branch HEAD"
  clone_latest "https://github.com/google-research/android_world.git" "$android_world_root"
  clone_latest "https://github.com/TencentQQGYLab/AppAgent.git" "$appagent_root"
  clone_latest "https://github.com/mobilegptsys/MobileGPT.git" "$mobilegpt_root"
  clone_latest "https://github.com/wuzw21/OmniTransfer.git" "$omnitransfer_root"
  if [[ "$skip_bmoca" != 1 ]]; then
    clone_latest "https://github.com/wuzw21/b-moca.git" "$bmoca_root"
    clone_latest "https://github.com/gimme1dollar/android_env.git" "$android_env_root"
    clone_latest "https://github.com/gimme1dollar/auto_ui.git" "$auto_ui_root"
  fi
}

prepare_external_repositories

log "creating Python 3.12 environment"
uv venv --python 3.12.11 "$repo/.venv"
if [[ "$upgrade_python_deps" == 1 ]]; then
  (cd "$repo" && uv lock --upgrade)
fi
uv_sync_args=(sync --group dev)
[[ "$skip_bmoca" == 1 ]] || uv_sync_args+=(--extra bmoca)
(cd "$repo" && uv "${uv_sync_args[@]}")
[[ -f "$android_world_root/requirements.txt" ]] && "$python_bin" -m pip install -r "$android_world_root/requirements.txt"
[[ -f "$mobilegpt_root/Server/requirements.txt" ]] && "$python_bin" -m pip install -r "$mobilegpt_root/Server/requirements.txt"
[[ -f "$appagent_root/requirements.txt" ]] && "$python_bin" -m pip install -r "$appagent_root/requirements.txt"
if [[ "$skip_bmoca" != 1 ]]; then
  [[ -f "$bmoca_root/requirements.txt" ]] && "$python_bin" -m pip install -r "$bmoca_root/requirements.txt"
  [[ -f "$android_env_root/pyproject.toml" ]] && "$python_bin" -m pip install -e "$android_env_root"
  [[ -f "$auto_ui_root/pyproject.toml" ]] && "$python_bin" -m pip install -e "$auto_ui_root"
fi

a11y_apk="$repo/vendor/androidworld/2024.05.13-accessibility_forwarder.apk"
[[ -f "$a11y_apk" ]] || die "Accessibility forwarder APK missing: $a11y_apk"

env_file="$remote_root/4090.env"
cat > "$env_file" <<EOF
export OMNIFLOW_EXP_ASSET_ROOT="$repo/data"
export OMNIFLOW_EXP_RESULTS_ROOT="$repo/data"
export OMNIFLOW_EXP_MEMORY_ROOT="$repo/data"
export OMNIFLOW_ENV_FILE="$model_env"
export OMNIFLOW_ANDROID_WORLD_ROOT="$android_world_root"
export OMNIFLOW_ANDROID_SDK_ROOT="$sdk_root"
export OMNIFLOW_APPAGENT_ROOT="$appagent_root"
export OMNIFLOW_APPAGENT_REVISION="$(git -C "$appagent_root" rev-parse HEAD)"
export OMNIFLOW_MOBILEGPT_ROOT="$mobilegpt_root"
export OMNITRANSFER_ROOT="$omnitransfer_root"
export OMNIFLOW_ANDROIDWORLD_A11Y_APK="$a11y_apk"
export OMNIFLOW_ANDROIDWORLD_REVISION="$(git -C "$android_world_root" rev-parse HEAD)"
export OMNIFLOW_BMOCA_ROOT="$bmoca_root"
export OMNIFLOW_BMOCA_ANDROID_ENV_ROOT="$android_env_parent"
export OMNIFLOW_BMOCA_AVD_HOME="$remote_root/bmoca-avd"
export PYTHON_BIN="$python_bin"
export OMNIFLOW_SOURCE_COMMIT="${OMNIFLOW_SOURCE_COMMIT:-}"
export OMNIFLOW_SOURCE_DIRTY="${OMNIFLOW_SOURCE_DIRTY:-unknown}"
export ANDROID_HOME="$sdk_root"
export ANDROID_SDK_ROOT="$sdk_root"
export PATH="$sdk_root/platform-tools:$sdk_root/emulator:$sdk_root/cmdline-tools/latest/bin:$PATH"
EOF
if [[ "$skip_bmoca" != 1 ]]; then
  printf 'export OMNIFLOW_BMOCA_REVISION="%s"\n' "$(git -C "$bmoca_root" rev-parse HEAD)" >> "$env_file"
fi
chmod 600 "$env_file"
# shellcheck disable=SC1090
source "$env_file"

log "checking native page encoder"
(cd "$repo" && "$python_bin" - <<'PY'
from omniflow.transfer.embedding import PageEncoder

encoder = PageEncoder()
if encoder.dimension != 512:
    raise SystemExit(f"native_page_encoder_dimension_mismatch:{encoder.dimension}")
print("native_page_encoder=ok")
PY
)

manifest="$remote_root/deployment_manifest.json"
export OMNIFLOW_4090_ENV="$env_file"
MANIFEST_PATH="$manifest" MODE="$mode" SKIP_BMOCA="$skip_bmoca" \
  REPO_PATH="$repo" AW_PATH="$android_world_root" APPAGENT_PATH="$appagent_root" \
  MOBILEGPT_PATH="$mobilegpt_root" TRANSFER_PATH="$omnitransfer_root" \
  BMOCA_PATH="$bmoca_root" ANDROID_ENV_PATH="$android_env_root" \
  "$python_bin" - <<'PY'
import json, os, platform, subprocess
from pathlib import Path

def rev(path):
    p = Path(path)
    if not (p / ".git").exists():
        return None
    return subprocess.check_output(["git", "-C", str(p), "rev-parse", "HEAD"], text=True).strip()

paths = {
    "omniflow_exp": os.environ["REPO_PATH"],
    "android_world": os.environ["AW_PATH"],
    "appagent": os.environ["APPAGENT_PATH"],
    "mobilegpt": os.environ["MOBILEGPT_PATH"],
    "omnitransfer": os.environ["TRANSFER_PATH"],
}
if os.environ["SKIP_BMOCA"] != "1":
    paths.update({"bmoca": os.environ["BMOCA_PATH"], "android_env": os.environ["ANDROID_ENV_PATH"]})
manifest = {
    "schema": 1,
    "mode": os.environ["MODE"],
    "source_worktree_dirty": os.environ.get("OMNIFLOW_SOURCE_DIRTY", "unknown"),
    "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
    "python": subprocess.check_output([os.environ["REPO_PATH"] + "/.venv/bin/python", "--version"], text=True).strip(),
    "repositories": {
        name: {
            "path": path,
            "commit": (
                os.environ.get("OMNIFLOW_SOURCE_COMMIT")
                if name == "omniflow_exp" and os.environ.get("OMNIFLOW_SOURCE_COMMIT")
                else rev(path)
            ),
        }
        for name, path in paths.items()
    },
    "runtime_env": os.environ.get("OMNIFLOW_4090_ENV", ""),
}
Path(os.environ["MANIFEST_PATH"]).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
PY

if [[ "$skip_device_setup" != 1 ]]; then
  log "setting up source/target AndroidWorld devices"
  (cd "$repo" && bash scripts/exp/run_androidworld.sh --setup-device all)
fi

log "running deployment validation"
(cd "$repo" && bash scripts/exp/run_androidworld.sh \
  --check-only --e2e-task CameraTakePhoto --e2e-method all --e2e-device all \
  --e2e-source-seed 111 --e2e-evaluation-seed 113)
(cd "$repo" && "$python_bin" -m pytest -q tests/test_exp_script.py tests/test_run_tasks.py)

if [[ "$run_smoke" == 1 ]]; then
  log "running CameraTakePhoto omniflow smoke test"
  (cd "$repo" && bash scripts/exp/run_androidworld.sh \
    --e2e-task CameraTakePhoto --e2e-method omniflow --e2e-device small5554 \
    --e2e-source-seed 111 --e2e-evaluation-seed 113 --control-backend oob)
fi

log "ready"
log "source env with: source '$env_file'"
log "deployment manifest: $manifest"
