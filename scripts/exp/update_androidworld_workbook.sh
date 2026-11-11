#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
node_bin="${OMNIFLOW_WORKBOOK_NODE:-/Users/wuzewen/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node}"
node_modules="${OMNIFLOW_WORKBOOK_NODE_MODULES:-/Users/wuzewen/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules}"
if [[ ! -x "$node_bin" || ! -d "$node_modules" ]]; then
  echo "spreadsheet_runtime_missing:set OMNIFLOW_WORKBOOK_NODE and OMNIFLOW_WORKBOOK_NODE_MODULES" >&2
  exit 2
fi
workdir="$(mktemp -d "${TMPDIR:-/tmp}/omniflow-workbook.XXXXXX")"
trap 'rm -rf "$workdir"' EXIT
ln -s "$node_modules" "$workdir/node_modules"
cp "$repo/scripts/exp/update_androidworld_workbook.mjs" "$workdir/update.mjs"
"$node_bin" "$workdir/update.mjs" --repo-root "$repo" "$@"
