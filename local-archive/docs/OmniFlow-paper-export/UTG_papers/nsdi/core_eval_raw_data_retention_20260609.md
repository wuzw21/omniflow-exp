# Core Eval Raw Data Retention

Last updated: 2026-06-11

This document defines what to keep for the current AndroidWorld evaluation
claim. The current policy is latest-version only: preserve artifacts referenced
by the latest index. Older snapshots are not part of the current paper-facing
dataset.

## Current Source Of Truth

Latest index directory:

`/Users/wuzewen/Projects/Omni/OmniFlow/runtime/evals/androidworld_validator/latest_results_index/`

Required files:

- `manifest.json`
- `summary.md`
- `all_latest.jsonl`
- `success.jsonl`
- `failure.jsonl`
- `referenced_paths.json`
- `by_task/*/results.jsonl`

Do not copy raw RunLogs, screenshots, XML, checkpoints, or baseline directories
into `omniflow-paper-export`. Keep them in their original `runtime/` locations
and reference them by absolute path.

## Must Keep

Keep the latest index and every file or directory referenced by
`referenced_paths.json`. That file is generated from these fields in
`all_latest.jsonl`:

- `source_trace`
- `canonical_runlog`
- `target_trace`
- `output_dir`
- `result_file`
- `summary_json`
- `androidworld_task_results_jsonl`
- `replay_stdout_log`
- `replay_stderr_log`
- native replay result fields under `target_trace`
- `androidworld_checkpoint`

Keep only exact paths referenced by `referenced_paths.json` or required to open
those referenced directories. Older sibling attempts in the same runtime roots
are outside the latest version.

## Excluded History

These are not the current paper result. They can be deleted or archived outside
the paper-facing dataset unless a path is explicitly referenced by
`all_latest.jsonl`:

- `/Users/wuzewen/Projects/Omni/OpenOmniBot/runtime/androidworld_accepted_by_task_1729.json`
- Older first-30 accepted replay outputs.
- Old real-device `45291FDAP0013Z` MobileGPT/AutoDroid outputs.
- Ad-hoc Codex-collected trajectories not referenced by `all_latest.jsonl`.
- Dryrun, smoke, probe, debug, retry, and pre-fix directories.

## Deletion Rule

Before deleting a runtime artifact, check whether its path appears in the latest
index:

```bash
rg -F "/path/to/delete" \
  /Users/wuzewen/Projects/Omni/OmniFlow/runtime/evals/androidworld_validator/latest_results_index/all_latest.jsonl \
  /Users/wuzewen/Projects/Omni/OmniFlow/runtime/evals/androidworld_validator/latest_results_index/referenced_paths.json \
  /Users/wuzewen/Projects/Omni/OmniFlow/runtime/evals/androidworld_validator/latest_results_index/summary.md \
  /Users/wuzewen/Projects/Omni/OmniFlow/runtime/evals/androidworld_validator/latest_results_index/manifest.json
```

If the path appears there, keep it. If it only appears in older manifests or
debug folders, archive/delete it only after confirming it is not needed for a
specific failure analysis.
