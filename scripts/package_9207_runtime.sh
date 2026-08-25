#!/usr/bin/env bash

set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
temporary=""
trap '[[ -z "$temporary" ]] || rm -rf "$temporary"' EXIT

usage() {
  echo "usage: $0 build [archive.tar.gz] | install <archive.tar.gz> [omni-root]" >&2
  exit 2
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

verify_checksums() {
  local directory="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    (cd "$directory" && sha256sum -c SHA256SUMS)
  else
    (cd "$directory" && shasum -a 256 -c SHA256SUMS)
  fi
}

build_release() {
  local omni_root home_root transfer_root checkpoint oob_apk
  local flow_commit transfer_commit release_id archive payload python_bin
  omni_root="${OMNI_ROOT:-$(cd "$repo/.." && pwd)}"
  home_root="${HOME_ROOT:-$(cd "$omni_root/../.." && pwd)}"
  transfer_root="${OMNITRANSFER_ROOT:-$omni_root/OmniTransfer}"
  checkpoint="${OMNITRANSFER_MATCHER_CHECKPOINT:-$transfer_root/output/point_sparse_graph_original_multimodal_v1/full_seed17/model.pt}"
  oob_apk="${OMNIFLOW_OOB_APK:-$repo/data/runtime/oob/OpenOmniBot-foolproof-debug.apk}"
  flow_commit="$(git -C "$repo" rev-parse HEAD)"
  transfer_commit="$(git -C "$transfer_root" rev-parse HEAD)"
  release_id="omniflow-9207-${flow_commit:0:12}-${transfer_commit:0:12}"
  archive="${1:-$repo/data/runtime/releases/$release_id.tar.gz}"
  temporary="$(mktemp -d)"
  payload="$temporary/$release_id"
  mkdir -p "$payload/git" "$payload/assets/oob" \
    "$payload/assets/omnitransfer" "$payload/config"

  git -C "$repo" bundle create "$payload/git/omniflow-exp.bundle" main
  git -C "$transfer_root" bundle create \
    "$payload/git/omnitransfer.bundle" codex/runtime-api
  cp "$oob_apk" "$payload/assets/oob/OpenOmniBot-foolproof-debug.apk"
  cp "$checkpoint" "$payload/assets/omnitransfer/model.pt"
  sed \
    -e "s|@HOME_ROOT@|$home_root|g" \
    -e "s|@OMNI_ROOT@|$omni_root|g" \
    -e "s|@OMNIFLOW_ROOT@|$repo|g" \
    -e "s|@OMNITRANSFER_ROOT@|$transfer_root|g" \
    "$repo/config/runtime.env.template" > "$payload/config/runtime.env"

  python_bin="${PYTHON_BIN:-$repo/.venv/bin/python}"
  RELEASE_ID="$release_id" FLOW_COMMIT="$flow_commit" \
  TRANSFER_COMMIT="$transfer_commit" OOB_SHA="$(sha256_file "$oob_apk")" \
  CHECKPOINT_SHA="$(sha256_file "$checkpoint")" \
  "$python_bin" - "$payload/manifest.json" <<'PY'
import json
import os
import sys

manifest = {
    "schema_version": "omniflow.9207-runtime-release.v1",
    "release_id": os.environ["RELEASE_ID"],
    "omniflow_commit": os.environ["FLOW_COMMIT"],
    "omnitransfer_commit": os.environ["TRANSFER_COMMIT"],
    "omnitransfer_architecture": "omnitransfer_point_conditioned_sparse_graph_v10",
    "omnitransfer_checkpoint_sha256": os.environ["CHECKPOINT_SHA"],
    "oob_apk_sha256": os.environ["OOB_SHA"],
    "androidworld_revision": "632ac95959ace58c8e2ed2db8e4209cc3d9c26ef",
    "model": "Qwen3.6-Plus",
    "contains_secrets": False,
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
    handle.write("\n")
PY

  (
    cd "$payload"
    find assets config git -type f -print | sort | while read -r file; do
      printf '%s  %s\n' "$(sha256_file "$payload/$file")" "$file"
    done
    printf '%s  manifest.json\n' "$(sha256_file "$payload/manifest.json")"
  ) > "$payload/SHA256SUMS"
  mkdir -p "$(dirname "$archive")"
  tar -czf "$archive" -C "$temporary" "$release_id"
  echo "$archive"
}

install_release() {
  local archive="$1" omni_root="$2"
  local flow_root="$omni_root/OmniFlow-exp"
  local transfer_root="$omni_root/OmniTransfer"
  local payload legacy_config secret_config
  temporary="$(mktemp -d)"
  tar -xzf "$archive" -C "$temporary"
  payload="$(find "$temporary" -mindepth 1 -maxdepth 1 -type d -print -quit)"
  verify_checksums "$payload"

  git -C "$flow_root" update-ref -d refs/remotes/runtime/release || true
  git -C "$flow_root" fetch "$payload/git/omniflow-exp.bundle" \
    main:refs/remotes/runtime/release
  git -C "$flow_root" merge --ff-only refs/remotes/runtime/release
  git -C "$transfer_root" update-ref -d refs/remotes/runtime/release || true
  git -C "$transfer_root" fetch "$payload/git/omnitransfer.bundle" \
    codex/runtime-api:refs/remotes/runtime/release
  git -C "$transfer_root" merge --ff-only refs/remotes/runtime/release

  install -D -m 0644 "$payload/assets/oob/OpenOmniBot-foolproof-debug.apk" \
    "$flow_root/data/runtime/oob/OpenOmniBot-foolproof-debug.apk"
  install -D -m 0644 "$payload/assets/omnitransfer/model.pt" \
    "$transfer_root/output/point_sparse_graph_original_multimodal_v1/full_seed17/model.pt"
  install -D -m 0644 "$payload/config/runtime.env" \
    "$flow_root/config/runtime.env"

  legacy_config="$flow_root/config/9207_mobilegpt.env"
  secret_config="$flow_root/config/runtime.secrets.env"
  if [[ ! -f "$secret_config" && -f "$legacy_config" ]]; then
    awk -F= 'toupper($1) ~ /(API_KEY|TOKEN|SECRET|PASSWORD)$/ {print}' \
      "$legacy_config" > "$secret_config"
    if [[ -s "$secret_config" ]]; then
      chmod 0600 "$secret_config"
      rm -f "$legacy_config"
    else
      rm -f "$secret_config"
    fi
  fi
  echo "installed $(basename "$payload")"
}

case "${1:-}" in
  build)
    build_release "${2:-}"
    ;;
  install)
    [[ -n "${2:-}" ]] || usage
    install_release "$2" "${3:-${OMNI_ROOT:-/home/wuzewen/Projects/Omni}}"
    ;;
  *)
    usage
    ;;
esac
