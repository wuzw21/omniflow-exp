# AndroidWorld experiment entry point

`run_androidworld.sh` is the only experiment script in this repository. It
owns source-asset conversion, long-term-memory refresh, static validation, and
real-time AndroidWorld execution.

## Pipeline

One normal single-task invocation performs the complete workflow:

1. Read the task's successful seed-111 source RunLog from the authoritative
   source index.
2. Resolve or create the native source asset for every selected method:
   - `fixed_replay`: use the canonical recorded actions;
   - `ours`: compile the RunLog, call the existing `enhance_function(...)`
     exactly once, validate, freeze, and register the Store;
   - `mobilegpt_offline_retrieval`: resolve or create native source memory;
   - `appagent_demo`: resolve or create the native demonstration;
   - `t3a_hint`: derive the semantic hint from the same Function and RunLog.
3. Reuse every already registered or frozen source asset without regeneration.
4. Prepare the selected target devices and replay each method.
5. Use the AndroidWorld official validator as the result, recording calls,
   tokens, actions, episode duration, and outer wall time for every cell.

Function schema and transfer-state checks are internal validation for `ours`;
they are not the experiment conclusion. Conversion never observes a target
device or executes source coordinates directly on a target.

## Common environment

All data paths are absolute and outside the repository.

| Variable | Meaning |
| --- | --- |
| `OMNIFLOW_EXP_ASSET_ROOT` | External immutable experiment assets |
| `OMNIFLOW_EXP_RESULTS_ROOT` | External immutable attempts and registered results |
| `OMNIFLOW_EXP_MEMORY_ROOT` | Content-addressed long-term-memory root |
| `PYTHON_BIN` | Python executable containing the project dependencies |
| `OMNITRANSFER_ROOT` | Canonical OmniTransfer checkout |
| `OMNIFLOW_SINGLE_TASK_SOURCE_SEED` | Source RunLog seed; formal value is `111` |
| `OMNIFLOW_SINGLE_TASK_EVALUATION_SEED` | Target evaluation seed; formal value is `113` |
| `OMNIFLOW_OURS_CONVERTED_ASSET_ROOT` | A new, empty conversion version directory |
| `OMNIFLOW_ENV_FILE` | Model credentials and OpenAI-compatible endpoint |
| `OMNIFLOW_OURS_AUTHORING_MANIFEST` | Immutable offline Function authoring manifest |

The source RunLog index defaults to
`$OMNIFLOW_EXP_ASSET_ROOT/runtime/evals/androidworld_validator/core_archive/success_source_runlogs/index_by_task.json`.
Set `OMNIFLOW_OURS_SOURCE_ASSET_INDEX` only when an explicit immutable index is
required.

## Run one RunLog through all methods

This is the complete one-command workflow:

```bash
OMNIFLOW_EXP_ASSET_ROOT=/absolute/assets \
OMNIFLOW_EXP_RESULTS_ROOT=/absolute/results \
OMNIFLOW_EXP_MEMORY_ROOT=/absolute/memory \
OMNIFLOW_ENV_FILE=/absolute/model.env \
PYTHON_BIN=/absolute/python \
OMNITRANSFER_ROOT=/absolute/OmniTransfer \
bash scripts/exp/run_androidworld.sh \
  --tasks AudioRecorderRecordAudio
```

The default method set contains all five methods. Set
`OMNIFLOW_SINGLE_TASK_METHODS` only for an intentional subset. When the
canonical `ours` Store is missing, the command creates and registers it before
preparing MobileGPT and AppAgent assets, provided the immutable authoring
manifest is configured; then the same process continues to
target replay. `--tasks` implies task-major scheduling and skips every result
cell with the same task, method, device, source seed, and evaluation seed that
is already registered with an official-validator conclusion. A result from a
different seed never causes a skip. There is no model retry. A failed target
execution never rebuilds or replaces frozen source assets.

`--check-only` is deliberately read-only. It validates existing assets but
will fail rather than create a missing method asset:

```bash
bash scripts/exp/run_androidworld.sh --check-only
```

`--convert-ours-assets` remains available for conversion-only maintenance. It
uses no external authoring model, does not replay a task, and is not the normal
experiment workflow.

## Full and bounded matrices

Run all five methods, both devices, and every indexed task:

```bash
bash scripts/exp/run_androidworld.sh --all-tasks
```

Run the four non-T3A methods on both devices for a bounded task slice:

```bash
bash scripts/exp/run_androidworld.sh \
  --all-tasks \
  --eight-cells \
  --tasks AudioRecorderRecordAudio,FilesMoveFile
```

Batch scheduling is task-major. Before execution it performs a read-only static
pass over every selected task. A cell with an existing official-validator
conclusion in long-term memory is skipped; it is never rerun merely because a
later attempt might succeed or be cheaper.

## Refresh long-term memory

Use the same script to ingest immutable RunLogs, Function catalogs, and result
registrations:

```bash
OMNIFLOW_EXP_ASSET_ROOT=/absolute/assets \
OMNIFLOW_EXP_MEMORY_ROOT=/absolute/memory \
OMNIFLOW_MEMORY_RUNLOG_ROOTS=/absolute/runlogs \
OMNIFLOW_MEMORY_RESULT_ROOTS=/absolute/results \
OMNIFLOW_MEMORY_FUNCTION_CATALOGS=/absolute/catalog.json \
PYTHON_BIN=/absolute/python \
bash scripts/exp/run_androidworld.sh --refresh-memory
```

`current.json` is the only runtime entry point. Identical files share one
content-addressed object; original attempts remain immutable.

## Useful diagnostics

```bash
bash scripts/exp/run_androidworld.sh --help
bash -n scripts/exp/run_androidworld.sh
```

Conversion prints a JSON summary containing the catalog, Store index, memory
index, converted task count, and frozen status. Real-time runs write logs and
attempt evidence only beneath `OMNIFLOW_EXP_RESULTS_ROOT`.
Every method cell records all AndroidWorld observations, including successful
cells. `observations/index.json` is written before result aggregation; the same
ordered records appear as `observation_evidence` in `task_results.jsonl`.
Entries point to screenshots under `observations/objects/` with exact SHA-256
and dimensions. Repeated observations keep separate indices while identical
images share one object. A missing or invalid screenshot is reported on that
observation instead of being silently omitted.

Each observation uses one AndroidWorld `get_state()` call for pixels and the
native accessibility forest. The Host converts a complete forest to
hierarchical XML locally and does not issue another UI dump. An incomplete
Fold hierarchy is marked explicitly and cannot enter OmniTransfer; normal VLM
fallback receives the saved screenshot instead.
