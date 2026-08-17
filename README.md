# OmniFlow-exp

Clean, paper-only AndroidWorld evaluation and B-MoCA environment validation
code for OmniFlow.

This repository contains code and orchestration only. RunLogs, screenshots,
models, APKs, AndroidWorld checkouts, baseline memories, and evaluation results
must live outside the repository and are supplied through environment paths.

## Paper methods

- `fixed_replay` (RPA)
- `ours` (OmniFlow)
- `mobilegpt_offline_retrieval` (MobileGPT)
- `appagent_demo` (AppAgent)
- `t3a_hint` (T3A + retrieved semantic trace)

`mobilegpt_offline_retrieval` is an adapted offline-retrieval baseline. The
same verified source RunLog is deterministically converted into MobileGPT's
native page/subtask/action memory; target execution keeps MobileGPT's native
app selection, page retrieval, subtask retrieval, and action reader. The
conversion does not claim to reproduce MobileGPT's optional online authoring
episodes.

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

B-MoCA is selected through the same entry point with `--environment bmoca`.
The default method remains the native OmniFlow E2E loop with Function, checker,
and OmniTransfer. Passing `--methods script-replay` runs the registered
zero-model MobileGPT-style semantic-selector comparison on the same official
environments. It never uses resource IDs: pointer actions require a unique
text/content-description or local child/parent structural locator on a stable
target page.
For one-task evaluation, pass the successful canonical source RunLog through
`--source-runlog`; the same entry automatically builds the single replay
Function from its sibling `transfer_states.json` before running env100–109.
All environment jobs are submitted together, with a per-AVD execution lock for
snapshots that share the same virtual device.
For the same `omniflow` method, setting
`OMNIFLOW_BMOCA_DIRECT_FUNCTION_REPLAY=1` bypasses only the initial Planner and
directly calls the sole visible zero-argument Function. Successful replay makes
zero Planner calls; failed replay can use at most three existing fallback steps.
This switch does not add a new method or change Checker/OmniTransfer semantics.
OmniFlow rejects a transferred action only when its Top-1 rank probability is
below `0.70`; candidate margin remains diagnostic-only.
The entry pins the maintained B-MoCA revision whose device builder completes
Chrome and Gboard first-run setup before sealing the shared base snapshot; an
unverified keyboard state fails environment construction.

## Bounded per-task E2E pipeline

Use one command to obtain one complete, auditable task result set. The command
resolves the canonical successful source RunLog, prepares method-native
memories, qualifies `ours` on the source device, and runs the ten target cells.

```bash
export OMNIFLOW_EXP_ASSET_ROOT=/absolute/external/assets
export OMNIFLOW_EXP_RESULTS_ROOT=/absolute/external/results
export OMNIFLOW_EXP_MEMORY_ROOT=/absolute/external/androidworld_memory
export OMNIFLOW_EXP_MEMORY_INDEX="$OMNIFLOW_EXP_MEMORY_ROOT/current.json"
export OMNIFLOW_ENV_FILE=/absolute/model.env
export OMNIFLOW_ANDROID_WORLD_ROOT=/absolute/AndroidWorld
export OMNIFLOW_MOBILEGPT_ROOT=/absolute/MobileGPT
export OMNIFLOW_APPAGENT_ROOT=/absolute/AppAgent
export OMNITRANSFER_ROOT="$HOME/Projects/Omni/OmniTransfer"

bash scripts/exp/run_androidworld.sh \
  --e2e-task AudioRecorderRecordAudioWithFileName \
  --task-deadline-sec 1800
```

The deadline is a hard whole-pipeline wall limit and cannot exceed 1800
seconds. Within that limit the output is either all ten validator
conclusions or an immutable partial/blocked result identifying the exact
failed phase. It is not a promise that every method succeeds. A method failure
is retained as evaluation evidence rather than retried with changed rules.

### Fixed flow

1. Start or reuse the official source `AndroidWorldAvd` on
   `emulator-5560`, then pass the native AndroidWorld runtime preflight.
2. Resolve the canonical successful source-seed-`111` RunLog by exact SHA-256.
   Missing or unusable canonical source memory blocks the pipeline; it never
   switches to an unregistered online source.
3. Have an offline Agent interpret the selected RunLog's goal, ordered actions,
   action metadata, and fixed OmniTransfer capability boundary, then emit complete Function
   descriptions, parameters, bindings, fixed choices, and exclusions. The
   compiler only audits the Agent's actions against the RunLog and freezes the
   Store; it does not send page trees or screenshots back through the Agent and
   does not mechanically invent Function semantics.
4. Invoke that exact Function and its source arguments directly on
   `emulator-5560`. Qualification requires full replay, official validator
   success, `model_calls=0`, and `fallback_steps=0`. A failed qualification
   blocks only the two `ours` target cells.
5. Resolve or create task-local MobileGPT memory and AppAgent demo memory from
   the same selected RunLog. A preparation failure blocks only that method's
   two cells.
6. Run `fixed_replay`, `ours`, `mobilegpt_offline_retrieval`, `appagent_demo`,
   and `t3a_hint` sequentially on each device. The SmallPhone worker
   (`emulator-5554`) and unfolded Pixel Fold worker (`emulator-5564`, state
   `2`) run concurrently. Target seed is always `113`; fallback budget remains
   the frozen maximum of `5`.
7. Register every official-validator result immediately in the canonical
   external memory. Reuse already registered cells and never rerun them merely
   to improve success or cost.

| phase | per-phase cap | model use | failure scope |
|---|---:|---|---|
| source device and preflight | 240 s | none | all ten cells |
| semantic Function compilation | 180 s | `glm-5.1`, only when needed | two `ours` cells |
| direct source qualification | 300 s | forbidden | two `ours` cells |
| MobileGPT memory preparation | 300 s | frozen baseline implementation | two MobileGPT cells |
| AppAgent memory preparation | 360 s | frozen baseline implementation | two AppAgent cells |
| each target cell | 240 s | frozen method implementation | that cell |

These are local caps, not additive reservations. Every child receives only the
remaining part of the 1800-second global deadline. When no time remains, the
coordinator does not launch another process and writes `deadline_exceeded` for
every unfinished cell.

The five target methods retain their frozen models and policies.
`OMNIFLOW_E2E_OUTPUT_ROOT` changes the external attempt root, and
`OMNIFLOW_E2E_ATTEMPT_ID` supplies a safe immutable attempt identifier.

### Schema and evidence

The only source contract is `omniflow.run_log.v1`. Formal AndroidWorld runs use
the native `State` fields `pixels`, `forest`, `ui_elements`, and `auxiliaries`.
OOB integrations may use the schema's `pixels`, `xml`, and `auxiliaries`
Observation variant. In both variants, `pixels` is an immutable screenshot
reference. Actions always contain only AndroidWorld `JSONAction` action types
and fields. The formal experiment pipeline uses the official AndroidWorld
validator and does not use OOB. A replacement source or Function Store is
selected through an explicit exact-SHA manifest; it is never silently
substituted.

The AndroidWorld Harness owns the episode recorder. It wraps the common State
and action seams after task setup and seals the RunLog with the official
validator conclusion, so every method that enters an episode uses the same
recording path. Agents own only policy, planner, and method-native memory;
OmniFlow transfer-state evidence is an optional sidecar and is not required for
baseline RunLogs.

Each immutable attempt contains:

```text
<OMNIFLOW_E2E_OUTPUT_ROOT>/<Task>/<attempt-id>/
  pipeline_manifest.json
  pipeline_summary.json
  pipeline.md                 # preparation/cost table
  source/                     # source collection and selection evidence
  source_qualification/       # direct Function replay evidence
  assets/                     # immutable task-local preparation artifacts
  target_attempts/            # method/device child evidence
  report/
    cells.jsonl               # exactly eight rows
    cells.csv                 # exactly eight rows
    cells.md                  # human-readable 8-cell table
    summary.json
```

The top-level summaries expose only aggregate `tool_calls` and `tokens`.
Detailed prompt/completion accounting, actions, validator result, episode
duration, outer wall time, error, and evidence path remain in each cell row or
phase record. RunLogs, screenshots, memories, results, and attempts stay
outside this repository.

## One source RunLog to method-native replay

The normal single-task command is the complete workflow. It reads one canonical
human-recorded source RunLog, resolves or creates each selected method's native
source asset, then continues to target replay and the AndroidWorld official
validator. It does not stop after Function validation.

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

For `fixed_replay`, the runtime resolves a recorded selector first and falls
back to the recorded point scaled to the target display whenever the selector
cannot produce one target. For `ours`, a missing canonical Store is created
from the RunLog and then frozen and registered in long-term memory. Semantic
authoring is optional and may be supplied as a reviewed Function bundle.
MobileGPT and AppAgent resolve or create their method-native source assets from
the same RunLog. T3A derives its hint from the same frozen Function and RunLog.
Existing valid assets are reused without rebuilding or re-authoring them.

After adaptation, every selected method is replayed on the configured targets
and evaluated by the official validator. The result cell records validator
success, model calls, prompt/completion/total tokens, actions, episode duration,
and outer wall time. These detailed accounting fields remain in immutable cell
evidence; top-level JSON, Markdown, CSV, and terminal summaries expose only
aggregate `tool_calls` and `tokens`. Every method and every validator outcome
records each AndroidWorld observation in order. Screenshots live under
`observations/objects/<sha256>.png`; the independent
`observations/index.json` is written before result aggregation.
`observation_evidence` in `task_results.jsonl` also maps every observation
index to its relative path, dimensions, and SHA-256. Repeated observations
remain separate ordered records while identical images share one immutable
object. Missing or unencodable images are explicit per-observation errors. For
`ours`, each Function action is mapped by the canonical OmniTransfer
implementation; a mapping failure returns to the normal VLM fallback.
Source-device coordinates are never executed directly on a target.
`--tasks` implies task-major execution and skips cells that already have a
registered official-validator conclusion.

`--convert-ours-assets` remains a conversion-only maintenance mode. It is not
the end-to-end experiment command. Conversion requires an immutable offline
Function authoring manifest whose source-index and source-RunLog hashes match;
it does not call an external authoring model.

### Shared with OOB

OOB exposes one Function write operation, `save_function`. It accepts `run_id`,
a complete RunLog object, or an absolute RunLog JSON path. A complete RunLog is
self-contained through inline XML, auxiliaries, and immutable screenshot
references, so conversion needs no additional state lookup. For optional
semantic authoring, the Agent follows the tool's conversion prompt and includes
one complete Function in the same call. Compilation, source-state checks, and
persistence remain internal and the response exposes no lifecycle stages.

The AndroidWorld adapter converts historical `executed_actions` and
`observation_before_act` records into the canonical RunLog and source-state
contracts OOB already uses. The experiment then passes the reviewed offline
`function_bundle` directly to the shared compiler. Store schema, binding
validation, state freezing, and runtime consumption remain identical for OOB
and this experiment.

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
  used to audit source assets, resolve MobileGPT's target package, and ground
  every AppAgent teacher action; and
- valid frozen baseline assets when they already exist.

The entry point does not synthesize or relabel missing RunLogs or Functions.
It preserves the recorded generating method, including an explicit
`unrecorded` value for legacy records without that field. Source assets are
authored once and frozen. The entry point may create method-native MobileGPT
and AppAgent assets once from the same valid source RunLog; failed or partial
creation is immutable and is never retried. MobileGPT consumes the canonical
source-seed-`111` RunLog through the offline teacher server and writes one
task-local memory through its stock `TaskAgent`, `Explore`, `Select`, `Derive`,
and `Memory.save_task()` flow. The teacher supplies only the recorded source
trajectory; MobileGPT still creates its own pages, subtasks, and action memory.
It never reads the Function Store or synthetic OmniFlow subtasks, and a teacher
miss returns to MobileGPT's native VLM planner. AppAgent separately uses source-only
UI evidence to ground its native human-demo capture without mutating the
canonical RunLog.
Environment setup and preflight logs are written to a separate unique external
preflight directory. A device or dependency failure therefore does not create
or consume the formal immutable task attempt.

The scheduler is task-major: it completes all ten method/device cells for one
task before starting the next task. It does not launch a method-major campaign.
The same entry point validates the frozen ours assets, then starts or
repairs the configured AVDs. Every pending cell cold-restarts its managed AVD
without loading or saving a Quick Boot snapshot, waits for adb and emulator
gRPC, forces the Pixel Fold to state `2`, and runs every required runtime
preflight. If the versioned
MobileGPT or AppAgent source memory is absent, the entry point creates it once
on the source-only `emulator-5560`, audits the exact `qwen3-vl-plus` model, and
freezes it before target execution. MobileGPT learns natively in the normal
AndroidWorld episode; AppAgent captures its source demo from the task's shared
official-success RunLog.

## Repository contents

### Page embedding contract

All active page retrieval uses `omniflow/transfer/page_embedding.py`, which
loads the latest page embedding implementation from the canonical
`~/Projects/Omni/OmniTransfer` checkout. The adapter records the checkpoint
path and SHA-256 in every embedding audit. OmniFlow-exp contains no legacy
page-encoder branch or embedding comparison baseline.

The current frozen page-embedding checkpoint is
`src/omnitransfer/checkpoints/omnitransfer_spatial_xml_alignment_v9_20260805/v9_spatial_xml_alignment_seed29.pt`.
This is the page representation; OmniTransfer's candidate-ranking release
remains a separate runtime responsibility.

- `omniflow/`: OmniFlow's public Python package.
  - `core/`: data models, configuration, schemas, and canonical RunLog handling.
  - `functions/`: Function artifacts, compilation, retrieval, storage, and management.
  - `runtime/`: runtime orchestration, action execution, and Checker recovery.
  - `transfer/`: OmniTransfer candidate calls and the canonical page-embedding adapter.
  - `vlm/`: VLM planning, tool-call parsing, model adaptation, and accounting.
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
