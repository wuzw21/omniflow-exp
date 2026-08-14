# AndroidWorld experiment entry point

`run_androidworld.sh` is the only experiment launcher in this repository. It
owns source-asset conversion, long-term-memory refresh, static validation, and
real-time AndroidWorld execution. The Python modules below are implementation
seams, not alternate launchers:

This rule also applies to development debugging. Agents must not manually call
the Python launcher or maintain a second runtime script.

| Component | Role | Used by the normal E2E command? |
| --- | --- | --- |
| `src/experiment/e2e_task_pipeline.py` | Source qualification and 10-cell orchestration | Yes |
| `omniflow/functions/assets.py` | Function validation, storage, editing, and freezing of a skill-produced bundle | Yes, for every `ours` Function asset |
| `src/experiment/function_assets.py` | Immutable skill-manifest conversion | Used when the configured skill manifest supplies the missing task |
| `src/experiment/direct_function_launch.py` | Seed-111 atomic Function qualification runner | Called by the E2E pipeline |
| `src/experiment/batch_outcomes.py` | Immutable cell and table accounting | Called by the E2E pipeline |

The E2E pipeline does not reuse a Store merely because its RunLog hash matches.
The Store provenance must identify the `androidworld-runlog-harvester` skill.
A default action wrapper or a legacy mechanical manifest is not a valid
Function asset.

## Pipeline

One normal single-task invocation performs the complete workflow:

1. Read the task's successful seed-111 source RunLog from the authoritative
   source index.
2. Resolve or create the native source asset for every selected method:
   - `fixed_replay`: use the canonical recorded actions;
   - `ours`: have an Agent interpret the RunLog goal, ordered actions, and
     existing action metadata without re-reading page representations, then
     validate, freeze, and register its Function bundle;
   - `mobilegpt_offline_retrieval`: teach MobileGPT from the canonical RunLog,
     then save its native task/page/subtask/action memory;
   - `appagent_demo`: resolve or create the native demonstration;
   - `t3a_hint`: derive the semantic hint from the same Function and RunLog.
3. Reuse every already registered or frozen source asset without regeneration.
4. Cold-restart each pending cell's managed AVD without Quick Boot snapshot
   load/save, run AndroidWorld setup and preflight, and replay the method.
5. Use the AndroidWorld official validator as the result, recording calls,
   tokens, actions, episode duration, and outer wall time for every cell.

Function schema and transfer-state checks are internal validation for `ours`;
they are not the experiment conclusion. Conversion never observes a target
device or executes source coordinates directly on a target.

Source qualification is scoped to one atomic Function replay. Its
`function_replay_success` is separate from the AndroidWorld task validator.
In each normal target E2E step, the planner selects exactly one peer tool from
native GUI actions, recalled Functions, and terminal tools. A Function may
expand into multiple recorded actions, then returns control to the next planner
step. A whole-task validator failure must not block a successful atomic
Function from entering that planner-based E2E test.

The planner follows the compact UI-TARS loop: current screenshot, prior action
tool calls, and one next action. Recalled Functions are prepended to the same
tool list as native GUI actions; the planner has no XML projection, full RunLog
prompt, or task-specific navigation policy.

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
| `OMNIFLOW_OURS_AUTHORING_MANIFEST` | Immutable bundle manifest produced by `androidworld-runlog-harvester` using `omniflow.function-agent-skill-manifest.v1` |
| `OMNIFLOW_MEMORY_MOBILEGPT_ROOTS` | Optional colon-separated roots containing sealed MobileGPT semantic memory |

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
`--methods`, `--devices`, and `--tasks` independently select ordered subsets;
their defaults remain all five methods, both formal devices, and every indexed
task. `OMNIFLOW_SINGLE_TASK_METHODS` remains an internal/runtime override. When
the canonical `ours` Store is missing, the command creates and registers it
before preparing MobileGPT and AppAgent assets, provided the immutable
authoring manifest is configured; then the same process continues to
target replay. `--tasks` implies task-major scheduling and skips every result
cell with the same task, method, device, source seed, and evaluation seed that
is already registered with an official-validator conclusion. A result from a
different seed never causes a skip. There is no model retry. A failed target
execution never rebuilds or replaces frozen source assets.
Completed cells are skipped before device startup. Pending cells use the same
script-owned cold-restart lifecycle on SmallPhone, Pixel Fold, and the
source-only emulator; no device is prepared manually.

## Bounded `ours` development run

Use the same script for an unregistered one-task diagnostic episode:

```bash
OMNIFLOW_ANDROID_WORLD_ROOT=/absolute/AndroidWorld \
OMNIFLOW_ENV_FILE=/absolute/model.env \
OMNIFLOW_SINGLE_TASK_STORE_PATH=/absolute/function_store/store.json \
OMNIFLOW_DEVELOPMENT_OUTPUT_PATH=/absolute/new/attempt \
OMNIFLOW_DEVELOPMENT_MODEL=GLM-4.6V \
bash scripts/exp/run_androidworld.sh \
  --development-run --tasks ExpenseAddMultipleFromGallery
```

The first run performs AndroidWorld app setup and saves the app snapshots used
by task initialization. To repeat against the same already initialized live
emulator without rerunning onboarding and permission setup, add:

```bash
OMNIFLOW_SINGLE_TASK_PERFORM_EMULATOR_SETUP=0
```

This override is development-only. Formal cells retain cold restart, setup,
memory resolution, immutable registration, and the paper model. MobileGPT is
owned by its adapter and does not participate in an `ours` development run.

`--check-only` is deliberately read-only. It validates existing assets but
will fail rather than create a missing method asset:

```bash
bash scripts/exp/run_androidworld.sh --check-only
```

`--convert-ours-assets` remains available for conversion-only maintenance. It
does not mechanically create Function semantics: it accepts a complete offline
Agent response, verifies its instruction version and source RunLog hash, audits
its actions against the RunLog, then freezes and registers it without replaying
a task.

## Full and bounded matrices

Run all five methods, both devices, and every indexed task:

```bash
bash scripts/exp/run_androidworld.sh --all-tasks
```

Run only MobileGPT on both formal devices for every indexed task:

```bash
bash scripts/exp/run_androidworld.sh \
  --all-tasks \
  --methods mobilegpt_offline_retrieval
```

The three axes are independent. For example, select one task, one method, and
one formal device with `--tasks AudioRecorderRecordAudio`,
`--methods mobilegpt_offline_retrieval`, and `--devices fold5564`.

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
