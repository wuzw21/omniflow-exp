#!/usr/bin/env bash

set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source_data="$repo/data"
target_host=""
target_data=""
plan_only=1

usage() {
  cat <<'EOF'
Usage: migrate_authoritative_data.sh [OPTIONS]

Stage files referenced by the local authoritative data/current.json and,
optionally, copy them to a fresh isolated host directory. Remote data is never
deleted or merged in place.

Options:
  --source-data PATH       Local authoritative data root (default: repo/data).
  --target-host HOST       SSH host used with --sync.
  --target-data PATH       Absolute target data root on the host.
  --sync                   Copy the staged files and rewritten current.json.
  --plan                   Print the migration plan without staging (default).
  -h, --help               Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-data)
      shift
      source_data="$1"
      ;;
    --target-host)
      shift
      target_host="$1"
      ;;
    --target-data)
      shift
      target_data="$1"
      ;;
    --sync)
      plan_only=0
      ;;
    --plan)
      plan_only=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      exit 2
      ;;
  esac
  shift
done

source_data="$(cd "$source_data" && pwd -P)"
if [[ "$plan_only" -eq 0 ]]; then
  if [[ -z "$target_host" || -z "$target_data" ]]; then
    echo "--sync requires --target-host and --target-data." >&2
    exit 2
  fi
  if [[ "$target_data" != /* || "$target_data" == *[[:space:]"]"* ]]; then
    echo "--target-data must be an absolute path without spaces or quotes." >&2
    exit 2
  fi
fi

stage_root="$(mktemp -d "${TMPDIR:-/tmp}/omniflow-authoritative.XXXXXX")"
plan_output="${stage_root}.json"
cleanup() {
  rm -rf "$stage_root" "$plan_output"
}
trap cleanup EXIT

PYTHONPATH="$repo:$repo/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$repo/.venv/bin/python" -m src.experiment.data_migration \
  --source-data "$source_data" \
  --target-data "${target_data:-/unused/target/data}" \
  --stage-root "$stage_root" > "$plan_output"
cat "$plan_output"

if [[ "$plan_only" -eq 1 ]]; then
  exit 0
fi

if ssh -o BatchMode=yes "$target_host" "test ! -e '$target_data'"; then
  ssh -o BatchMode=yes "$target_host" "mkdir -p '$target_data'"
else
  echo "target already exists; choose a fresh isolated --target-data: $target_data" >&2
  exit 1
fi

rsync -a --files-from="$stage_root/files.txt" \
  "$source_data/" "$target_host:$target_data/"
rsync -a "$stage_root/current.json" "$target_host:$target_data/current.json"
rsync -a "$stage_root/migration.json" "$target_host:$target_data/migration.json"
ssh -o BatchMode=yes "$target_host" \
  "test -s '$target_data/current.json' && test -s '$target_data/migration.json'"
echo "migration_status=staged target=$target_host:$target_data"
