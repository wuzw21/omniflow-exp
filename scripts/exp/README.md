# AndroidWorld experiment entry point

`run_androidworld.sh` is the only experiment script in this repository. It
owns source-asset conversion, long-term-memory refresh, static validation, and
real-time AndroidWorld execution.

## Pipeline

For `ours`, one task follows exactly this path:

1. Read the task's successful seed-111 source RunLog from the authoritative
   source index.
2. Import its actions and source UI states using the recorded full display
   width and height.
3. Call the single RunLog-to-Function compiler to create the complete recorded
   Function.
4. Call the existing `enhance_function(...)` exactly once to collect its
   semantic name, description, supported parameters, and evidence-backed
   checker rules. The model cannot alter the recorded action sequence.
5. Verify Function schema, transfer-state coverage, provenance, and the
   no-target-input audit. Record source actions that OmniTransfer cannot ground
   so the normal real-time VLM fallback handles them.
6. Freeze the new asset directory and register it in
   `OMNIFLOW_EXP_MEMORY_ROOT/current.json`.
7. During a later real-time run, resolve that frozen Store from memory and
   execute it through the normal Function router, OmniTransfer, VLM fallback,
   and AndroidWorld official validator.

Conversion never observes a target device, reads target task parameters, or
executes source coordinates directly on a target. Its only model call is the
single existing Function semantic collector.

## Common environment

All data paths are absolute and outside the repository.

| Variable | Meaning |
| --- | --- |
| `OMNIFLOW_EXP_ASSET_ROOT` | External immutable experiment assets |
| `OMNIFLOW_EXP_RESULTS_ROOT` | External immutable attempts and registered results |
| `OMNIFLOW_EXP_MEMORY_ROOT` | Content-addressed long-term-memory root |
| `PYTHON_BIN` | Python executable containing the project dependencies |
| `OMNITRANSFER_ROOT` | Canonical OmniTransfer checkout |
| `OMNIFLOW_OURS_CONVERTED_ASSET_ROOT` | A new, empty conversion version directory |
| `OMNIFLOW_ENV_FILE` | Model credentials and OpenAI-compatible endpoint |
| `OMNIFLOW_OURS_CONVERSION_MODEL` | Fixed conversion model; defaults to the paper config |

The source RunLog index defaults to
`$OMNIFLOW_EXP_ASSET_ROOT/runtime/evals/androidworld_validator/core_archive/success_source_runlogs/index_by_task.json`.
Set `OMNIFLOW_OURS_SOURCE_ASSET_INDEX` only when an explicit immutable index is
required.

## Convert one RunLog

This is the one-click conversion command:

```bash
OMNIFLOW_EXP_ASSET_ROOT=/absolute/assets \
OMNIFLOW_EXP_MEMORY_ROOT=/absolute/memory \
OMNIFLOW_OURS_CONVERTED_ASSET_ROOT=/absolute/assets/ours-functions-v3 \
OMNIFLOW_ENV_FILE=/absolute/model.env \
PYTHON_BIN=/absolute/python \
OMNITRANSFER_ROOT=/absolute/OmniTransfer \
bash scripts/exp/run_androidworld.sh \
  --convert-ours-assets \
  --tasks AudioRecorderRecordAudio
```

`--tasks` accepts a comma-separated list. Omit it to convert every source
RunLog in the authoritative index that does not already have a canonical
Function Store in memory. The output root must be new or empty. After
successful conversion it becomes read-only and is immediately registered in
long-term memory.

There is no model retry. A timeout, connection failure, invalid JSON,
source/hash mismatch, schema error, or missing source UI state fails conversion.
When a present source element cannot be mapped, provenance marks the Function
as requiring normal real-time VLM fallback; it never converts that condition
into source-coordinate passthrough. A failed target execution never rebuilds
or replaces the frozen Function asset.

## Run the converted Function in real time

After conversion, run the same task through the normal `ours` runtime on the
configured target devices:

```bash
OMNIFLOW_EXP_ASSET_ROOT=/absolute/assets \
OMNIFLOW_EXP_RESULTS_ROOT=/absolute/results \
OMNIFLOW_EXP_MEMORY_ROOT=/absolute/memory \
OMNIFLOW_SINGLE_TASK_TASK=AudioRecorderRecordAudio \
OMNIFLOW_SINGLE_TASK_METHODS=ours \
PYTHON_BIN=/absolute/python \
OMNITRANSFER_ROOT=/absolute/OmniTransfer \
bash scripts/exp/run_androidworld.sh
```

The script resolves the Store from memory, validates its SHA-256-bound Store,
transfer-state catalog, and provenance, prepares SmallPhone and unfolded Pixel
Fold targets, and records the official validator result and accounting fields.
It does not call the conversion path.

Use `--check-only` first to validate all static dependencies without creating
attempts or starting emulators:

```bash
bash scripts/exp/run_androidworld.sh --check-only
```

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
