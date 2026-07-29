# OmniFlow-exp

Clean, paper-only AndroidWorld evaluation code for OmniFlow.

This repository contains code and orchestration only. RunLogs, screenshots,
models, APKs, AndroidWorld checkouts, baseline memories, and evaluation results
must live outside the repository and are supplied through environment paths.

## Paper methods

- `fixed_replay` (RPA)
- `ours` (OmniFlow)
- `mobilegpt_offline_retrieval` (MobileGPT)
- `appagent_demo` (AppAgent)
- `t3a_hint` (T3A + retrieved semantic trace)

The public entry point is:

```bash
OMNIFLOW_EXP_ASSET_ROOT=/absolute/path/to/external/assets \
OMNIFLOW_EXP_RESULTS_ROOT=/absolute/path/to/external/results \
PYTHON_BIN=/absolute/path/to/python \
OMNITRANSFER_ROOT=/absolute/path/to/versioned/omnitransfer \
bash scripts/exp/run_androidworld.sh
```

The complete command reference, including one-RunLog Function conversion,
freezing, memory registration, real-time execution, and resume behavior, is in
[scripts/exp/README.md](scripts/exp/README.md).

## Ours: source RunLog to real-time execution

The conversion path has one public shell entry and one shared single-RunLog
Python interface: `compile_runlog_to_store(...)`. OOB calls this interface
directly using its RunLog and state loader. The AndroidWorld experiment adapter
imports the human-recorded actions and source UI states, calls the same
compiler, and then calls the existing `enhance_function(...)` exactly once.
OOB uses that same semantic collector. Batch conversion repeats this same path
once per selected task; it does not implement a second compiler or a separate
semantic-generation subsystem.

Convert exactly one source RunLog:

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

The conversion produces a v2 Function Store, referenced source transfer
states, and source-only provenance; freezes the output; then registers it in
the canonical long-term memory. It makes one fixed-model call through
`enhance_function(...)` to improve the Function name, description, supported
parameters, and evidence-backed checker rules. SDK retries are disabled. The
model cannot add, delete, reorder, or alter recorded actions. Conversion reads
no target input or target observation. If real OmniTransfer cannot ground a
present source element, provenance records that the corresponding runtime step
requires the ordinary VLM fallback; the converter never substitutes source
coordinates.

Run that frozen Function in real time:

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

At runtime the Function competes only in the dedicated Function routing path.
Each action is mapped by the canonical OmniTransfer implementation; a mapping
failure returns to the normal VLM fallback. Source-device coordinates are never
executed directly on a target.

### Shared with OOB

OOB exposes the same compiler through its `compile` bridge operation. The
bridge loads the requested RunLog with `get_run_log`, supplies referenced
source states with `get_state`, and calls `compile_runlog_to_store(...)`.
`enhance=true` then calls the same existing `enhance_function(...)`; it does not
select a different compiler.

The AndroidWorld adapter has one experiment-only responsibility: convert its
historical `executed_actions` and `observation_before_act` records into the
canonical RunLog and source-state contracts OOB already uses. Semantic
collection happens after the shared compiler call. Store schema, binding
validation, state freezing, and runtime consumption are therefore identical
for OOB and this experiment.

To run the canonical 116-task matrix in formal task-major order, keep the same
environment and run:

```bash
bash scripts/exp/run_androidworld.sh --all-tasks
```

Before launching, the same command can validate the entire selected matrix
without starting an emulator or creating an asset, attempt, log, preflight, or
result directory:

```bash
bash scripts/exp/run_androidworld.sh --check-only --all-tasks
```

When frozen T3A results are imported separately, run the remaining four
methods on both devices as the eight-cell phase:

```bash
bash scripts/exp/run_androidworld.sh --all-tasks --eight-cells
```

To run a bounded task-major slice through the same entry point:

```bash
bash scripts/exp/run_androidworld.sh --all-tasks --eight-cells \
  --tasks AudioRecorderRecordAudio,AudioRecorderRecordAudioWithFileName,FilesMoveFile
```

For a bounded slice, set `OMNIFLOW_SINGLE_TASK_SOURCE_INDEX` to the immutable
index containing exactly those selected tasks. Keep
`OMNIFLOW_MASTER_SOURCE_INDEX` pointed at the full 116-task inventory used by
result registration. When the frozen Function Stores do not use the default
layout, set `OMNIFLOW_OURS_STORE_INDEX` to their immutable hash-bound JSON
index.

The same `--eight-cells` flag without `--all-tasks` runs the selected single
task. It excludes only `t3a_hint`; it does not change any of the four method
implementations.

Batch mode requires both devices and either the exact five-method set or the
exact four-method eight-cell set. It skips a task only when every expected
immutable cell is registered, and stops for audit if a task is only partially
registered or an execution/environment failure occurs.
It performs one read-only static pass over every remaining selected task before
creating any batch directory. If any task fails, the whole batch stops before
source generation or target execution. Before it creates an attempt or starts
an emulator, it requires:

- exactly 116 indexed canonical RunLogs from source seed `111`, each
  marked as an AndroidWorld official-validator success, non-empty, and bound to
  its retained RunLog by a matching SHA-256;
- one immutable `ours` Function Store and complete referenced
  `transfer_states.json` catalog for the selected task;
- the hash-bound source state catalog and complete source-target provenance
  needed to ground every MobileGPT and AppAgent teacher action; and
- valid frozen baseline assets when they already exist.

The entry point does not synthesize or relabel missing RunLogs or Functions.
It preserves the recorded generating method, including an explicit
`unrecorded` value for legacy records without that field. Source assets are
authored once and frozen. The entry point may create the method-native
MobileGPT and AppAgent assets once from the same valid source RunLog; failed or
partial creation is immutable and is never retried.
Their shared grounding check uses source-only UI evidence, does not mutate the
canonical RunLog, and verifies that every teacher action can be grounded before
the immutable asset directory is claimed.
Environment setup and preflight logs are written to a separate unique external
preflight directory. A device or dependency failure therefore does not create
or consume the formal immutable task attempt.

The scheduler is task-major: it completes all ten method/device cells for one
task before starting the next task. It does not launch a method-major campaign.
The same entry point validates the frozen ours assets, then starts or
repairs the configured AVDs, waits for adb and emulator gRPC, forces the Pixel
Fold to state `2`, and runs every required runtime preflight. If the versioned
MobileGPT or AppAgent source memory is absent, the entry point creates it once
on the source-only `emulator-5560` from the task's shared official-success
RunLog, audits the exact `qwen3-vl-plus` model, and freezes it before target
execution.

## Repository contents

- `omniflow/`: OmniFlow's public Python package.
  - `core/`: data models, configuration, schemas, and canonical RunLog handling.
  - `functions/`: Function artifacts, compilation, retrieval, storage, and management.
  - `runtime/`: runtime orchestration, action execution, and Checker recovery.
  - `transfer/`: OmniTransfer calls, page encoding, alignment, memory, and review.
  - `vlm/`: VLM planning, prompt construction, model adaptation, UI projection, and accounting.
  - `bridge.py`: external JSON-line bridge entry point.
  - `vlm_coordinates.py`: shared-contract owner for VLM coordinate conversion.
- OmniTransfer is loaded only from `OMNITRANSFER_ROOT` or the canonical
  `~/Projects/Omni/OmniTransfer` checkout; no transfer implementation is
  vendored here.
- `src/experiment/`: formal task-major orchestration and immutable result registration.
- `src/integrations/`: AndroidWorld, baseline adapters, and their runtime helpers.
- `scripts/exp/run_androidworld.sh`: the only experiment script and public one-command entry point.
- `skills/`: AndroidWorld preflight and source RunLog collection instructions.
- `config/paper_androidworld.json`: the five-method paper configuration.

No experiment is run as part of code migration.
