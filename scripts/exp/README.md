# AndroidWorld experiment entry point

`run_androidworld.sh` is the only experiment launcher in this repository. It
owns source-asset conversion, long-term-memory refresh, static validation, and
real-time AndroidWorld execution. The Python modules below are implementation
seams, not alternate launchers:

| Purpose | Unified invocation | Internal path |
| --- | --- | --- |
| Formal task matrix | `run_androidworld.sh --tasks TASK` | Frozen five-method evaluation |
| Bounded `ours` development | `run_androidworld.sh --development-run --tasks TASK` | OmniFlow online Planner |
| Source screenshot refresh | `run_androidworld.sh --collect-source --tasks TASK` | Existing `fixed_replay`; no Planner or model calls |
| Static validation | `run_androidworld.sh --check-only` | No emulator or method execution |
| B-MoCA E2E | `run_androidworld.sh --environment bmoca --tasks TASK` | The same OmniFlow method on official B-MoCA snapshots |
| B-MoCA exact selector | `run_androidworld.sh --environment bmoca --methods script-replay --tasks TASK` | Parallel zero-model exact-selector replay |

B-MoCA changes only the environment by default. The unified command enumerates
the requested official environment IDs (100 through 109 by default), runs the normal
`observe -> recall -> plan -> Function/checker/OmniTransfer -> act` loop, and
reports the official reward for each episode. GLM-5.1 is used only by the front
Planner; Function fallback is fixed at zero.

The explicitly registered B-MoCA comparison `script-replay` is the only
exception to the default method above. It replays the sole visible Function,
ignores Checker steps, and resolves each pointer action by the first unique
exact source selector in `resource-id`, `text`, `content-desc` order. Missing
or ambiguous matches fail closed; it never uses fuzzy matching, parent lookup,
source coordinates, OmniTransfer, DP, Planner, or VLM. All requested environments
are submitted together (`OMNIFLOW_BMOCA_WORKERS`, default 10 for this method).
Different AVDs run concurrently; environments sharing one physical AVD are
serialized so Android Emulator never has two writers for the same device disks.

The maintained B-MoCA coordinator exposes every display size as screenshot
`[height, width]`, including tablets. The Host therefore preserves canonical
normalized UI coordinates without a tablet-specific axis swap or rescaling.
Its Chrome History validator accepts either the original Activity log event or
the stable on-screen History title plus History content container, so the same
goal contract covers phone and tablet Chrome presentations for every method.

The B-MoCA source is the pinned maintained revision under
`$workspace_root/releases/b-moca-$formal_bmoca_revision` unless
`OMNIFLOW_BMOCA_ROOT` supplies another exact-revision checkout. Its device
builder dismisses Gboard's first-run prompt and verifies keyboard readiness
before saving the base AVD snapshot; initialization failure prevents the
snapshot from being created.

`--collect-source` resolves the canonical successful seed-111 RunLog from
`current.json`, replays its recorded coordinates through the official
AndroidWorld task loop on the source AVD, and records native screenshots/XML
before every action plus the final state. It performs no selector lookup,
transfer, planning, or model call.
Collection succeeds only when fixed replay completes, the official validator
passes, and `model_calls=0`.

This rule also applies to development debugging. Agents must not manually call
the Python launcher or maintain a second runtime script.

| Component | Role | Used by the normal E2E command? |
| --- | --- | --- |
| `src/experiment/e2e_task_pipeline.py` | Source qualification and 10-cell orchestration | Yes |
| `omniflow/functions/assets.py` | Function validation, storage, editing, and freezing of a skill-produced bundle | Yes, for every `ours` Function asset |
| `src/experiment/function_assets.py` | Immutable skill-manifest conversion | Used when the configured skill manifest supplies the missing task |
| `src/experiment/direct_function_launch.py` | Seed-111 atomic Function qualification runner | Called by the E2E pipeline |
| `src/experiment/batch_outcomes.py` | Immutable cell and table accounting | Called by the E2E pipeline |

## Page embedding

The active page representation is the canonical OmniTransfer page embedding,
loaded through `omniflow/transfer/page_embedding.py` from
`~/Projects/Omni/OmniTransfer`. Its checkpoint path and SHA-256 are part of
the embedding provenance. This repository deliberately has no legacy
page-encoder branch or embedding comparison baseline.

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
   - `mobilegpt_offline_retrieval`: deterministically adapt the canonical
     RunLog plan into MobileGPT's native task/page/subtask/action memory, then
     use MobileGPT's native app and page retrieval online;
   - `appagent_demo`: convert the canonical RunLog schema to the native demonstration format, reusing only exact lineage-matched screenshot evidence;
   - `t3a_hint`: derive a versioned semantic hint from the aligned Function and
     source RunLog. Hint v2 identifies each operated UI object with its semantic
     target, successful-run purpose, and source accessibility evidence
     (`node id`, class, text, content description, resource ID, and package)
     when present. Source coordinates and source task values are never exposed;
     an unidentified pointer action fails conversion instead of becoming a
     generic “relevant UI target” instruction.
3. Reuse every already registered or frozen source asset without regeneration.
4. Cold-restart each pending cell's managed AVD without Quick Boot snapshot
   load/save, run AndroidWorld setup and preflight, and replay the method.
5. Use the AndroidWorld official validator as the result, recording calls,
   tokens, actions, reuse utilization, episode duration, and outer wall time
   for every cell.

AppAgent online execution uses the pinned upstream prompt, parser, model request,
label ordering, document UID algorithm, grid behavior, request interval,
single-response control flow, screenshot/XML capture, and device actions. The
experiment adapter mounts exactly one converted native demo-docs directory and
prepares the task's declared app before the first upstream decision round.
AndroidWorld owns only the LiveTask lifecycle, step-budget shell, and official
validator for this baseline. Converted RunLog actions remain offline provenance;
they are never replayed or injected into AppAgent online.

The same native-baseline rule applies to MobileGPT: only an
official-validator-successful seed-111 RunLog may be converted, conversion must
produce MobileGPT's standard memory schema, and MobileGPT retains its upstream
online retrieval and execution behavior. The required AndroidWorld adapters for
both baselines translate lifecycle, observations, actions, and accounting only;
they do not introduce a second planner or alter the baseline policy.

Every result records `reuse_numerator`, `reuse_denominator`, `reuse_rate`,
`reuse_unit`, and `reuse_evidence_status`. The rate is the fraction of native
reuse opportunities actually served by the method's converted source asset:
replayed GUI actions for `fixed_replay`, Function-origin GUI actions for
`ours`, direct memory hits for MobileGPT, AppAgent decision rounds with native
demo documentation, and T3A executed actions planned with the source hint.
Rates with different `reuse_unit` values are reported separately and are not
silently pooled. Missing evidence produces an unavailable rate rather than an
assumed zero or one.

Function schema and transfer-state checks are internal validation for `ours`;
they are not the experiment conclusion. Conversion never observes a target
device or executes source coordinates directly on a target.

An atomic Function replay may be used as an independent diagnostic, so its
`function_replay_success` remains separate from the whole-task validator. The
asset freeze gate is the ordered bundle qualification: when one RunLog yields
multiple semantic Functions, it invokes their recorded source calls in order
inside one AndroidWorld episode so later Functions retain the page state
established by earlier Functions. The bundle is source-qualified only when all
calls replay successfully, the official seed-111 validator passes, and both
`model_calls` and `fallback_steps` are zero. Each call is still one
`flow.call_tool(...)`; the qualification adapter marks only the final successful
call as terminal.
In each normal target E2E step, the planner selects exactly one peer tool from
native GUI actions, recalled Functions, and terminal tools. A Function may
expand into multiple recorded actions, then returns control to the next planner
step. A whole-task validator failure may guide another offline Function
revision, but the failed bundle is not frozen for formal target execution.

The planner follows the compact UI-TARS loop: current screenshot, prior action
tool calls, and one next action. Recalled Functions are prepended to the same
tool list as native GUI actions; the planner has no XML projection, full RunLog
prompt, or task-specific navigation policy.

AndroidWorld owns environment creation, controller setup, task lifecycle,
native state and action calls, snapshots, and validation. The experiment layer
only supplies method adapters and a transparent accounting proxy; it does not
patch AndroidWorld objects or provide an alternate device runtime.

## Common environment

All data paths are absolute and outside the repository.

The unified script has one persistent workspace profile: for a checkout at
`<workspace>/OmniFlow-exp`, it resolves experiment assets and results from the
sibling `<workspace>/OmniFlow`, long-term memory from
`<workspace>/assets/androidworld-experiment-memory-v1`, OmniTransfer from
`<workspace>/OmniTransfer`, and AndroidWorld/MobileGPT/AppAgent from the
corresponding `OmniFlow/runtime/external` directories. Environment variables
remain explicit overrides. Model credentials are never copied into this
profile and continue to load only from `OMNIFLOW_ENV_FILE` or the sibling
`OmniFlow/.env`.

Configuration and environment repairs belong to this entry point or the
narrow shared AndroidWorld harness seam. A manual export, ad-hoc emulator
command, or task-local workaround is diagnostic evidence only. Before moving
to the next task, preserve the failed attempt, add a deterministic regression
test, encode the stable repair in the shared script/core harness, and document
the resulting contract here.

The shared AndroidWorld setup seam first uses the pinned official APK install
call. If the connected emulator rejects the legacy
`--bypass-low-target-sdk-block` option, it retries the same install without that
option and preserves all other setup behavior.

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
| `OMNIFLOW_ANDROIDWORLD_RELEASE_ROOT` | Optional immutable AndroidWorld checkout root containing the `android_world` package; defaults to the pinned `632ac95` release beside the asset root |
| `OMNIFLOW_BMOCA_ROOT` | Exact maintained B-MoCA checkout; defaults to the pinned revision-named release beside `OmniFlow-exp` |
| `OMNIFLOW_BMOCA_ANDROID_ENV_ROOT` | Clean checkout of the pinned B-MoCA AndroidEnv fork; defaults to the revision-named release beside `OmniFlow-exp` |
| `OMNIFLOW_SQLITE_FTS4_LIBRARY` | Optional absolute compatible `libsqlite3` path; the unified entry automatically selects a system library when the experiment Python lacks AndroidWorld's required FTS4 support |
| `OMNIFLOW_ANDROIDWORLD_ADB_FILE_TRANSFER_TIMEOUT_SEC` | Positive timeout for each official AndroidWorld adb file push/copy; defaults to 300 seconds and never permits the upstream `None`/`0` unbounded wait |
| `OMNIFLOW_ANDROIDWORLD_SETUP_TIMEOUT_SEC` | Positive hard deadline for the complete official per-cell app setup; defaults to 300 seconds so an AndroidEnv/adb coordinator stall cannot consume the full episode budget |
| `OMNIFLOW_APPAGENT_DOCUMENT_MODEL` | AppAgent offline documentation VLM; defaults to the paper model `GLM-5.1` |
| `OMNIFLOW_MOBILEGPT_EMBEDDING_MODEL` | MobileGPT offline embedding model; defaults to its native `text-embedding-v4` |
| `OMNIFLOW_DEVELOPMENT_MODEL_ENDPOINT_PROFILE` | Development endpoint profile; defaults to `llmthu` |
| `OMNIFLOW_FORMAL_MODEL_ENDPOINT_PROFILE` | Formal endpoint profile; fixed to `llmthu` for `GLM-5.1` |
| `OMNIFLOW_OURS_AUTHORING_MANIFEST` | Immutable bundle manifest produced by `androidworld-runlog-harvester` using `omniflow.function-agent-skill-manifest.v1` |
| `OMNIFLOW_OURS_REVISION_REASON` | Explicit reason for selecting one newly converted immutable Function revision over the existing canonical Store; requires one `--tasks` value |
| `OMNIFLOW_MEMORY_MOBILEGPT_ROOTS` | Optional colon-separated roots containing sealed MobileGPT semantic memory |

The source RunLog index defaults to
`$OMNIFLOW_EXP_ASSET_ROOT/runtime/evals/androidworld_validator/core_archive/success_source_runlogs/index_by_task.json`.
Set `OMNIFLOW_OURS_SOURCE_ASSET_INDEX` only when an explicit immutable index is
required.

MobileGPT and AppAgent memory preparation share one RunLog conversion API:
`src.experiment.source_assets.convert_runlog_memory(method, ...)`. It accepts an
official-validator-successful seed-111 `omniflow.run_log.v1` and writes the
selected baseline's native immutable memory. MobileGPT delegates encoding,
memory access, and action generalization to its pinned upstream modules.
The adapter invokes the upstream action generalizer once per semantic parameter
and accepts only its native `<parameter__index>` placeholder grammar, preventing
one generated placeholder from being generalized again as ordinary text.
AppAgent writes the official demo directory and runs its pinned official
document generator. AppAgent additionally requires immutable before/after
screenshot references and XML in the RunLog; missing evidence is an explicit
conversion failure and never triggers source-coordinate replay or an emulator.
Its offline document generator uses `OMNIFLOW_APPAGENT_DOCUMENT_MODEL`; this is
separate from the fixed `GLM-5.1` online AndroidWorld planner model.
MobileGPT's offline encoder uses `OMNIFLOW_MOBILEGPT_EMBEDDING_MODEL`; the
default remains its native `text-embedding-v4`, and an endpoint-compatible
override does not change the online planner or AppAgent document model.
At runtime, MobileGPT chat is routed to the selected LLMTHU `GLM-5.1` endpoint,
while query embeddings are routed to the original `OPENAI_*` endpoint from the
environment file. The sealed memory manifest supplies the only accepted
embedding model, and the script verifies the live embedding dimension against
the stored hierarchy before starting a device.

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
For MobileGPT, a result cell is identified by the frozen native-memory contract
as well as task, method, device, and seeds. Archived memory protocols remain
immutable evidence but cannot shadow the earliest result produced by the
currently supported memory schema.

## Bounded `ours` development run

Use the same script for an unregistered single-cell diagnostic episode:

```bash
OMNIFLOW_ANDROID_WORLD_ROOT=/absolute/AndroidWorld \
OMNIFLOW_ENV_FILE=/absolute/model.env \
OMNIFLOW_SINGLE_TASK_STORE_PATH=/absolute/function_store/store.json \
OMNIFLOW_DEVELOPMENT_OUTPUT_PATH=/absolute/new/attempt \
OMNIFLOW_DEVELOPMENT_MODEL=GLM-4.6V \
OMNIFLOW_DEVELOPMENT_MODEL_ENDPOINT_PROFILE=llmthu \
bash scripts/exp/run_androidworld.sh \
  --development-run --tasks ExpenseAddMultipleFromGallery
```

The development entry ensures the selected AVD is booted with AndroidWorld's
required gRPC endpoint before launching the episode. Override
`OMNIFLOW_DEVELOPMENT_AVD` only when the selected console port is intentionally
mapped to another installed AVD.

The first run performs AndroidWorld app setup and saves the app snapshots used
by task initialization. To repeat against the same already initialized live
emulator without rerunning onboarding and permission setup, add:

```bash
OMNIFLOW_SINGLE_TASK_PERFORM_EMULATOR_SETUP=0
```

This override is development-only. Formal cells retain cold restart, setup,
memory resolution, immutable registration, and the paper model. MobileGPT is
owned by its adapter and does not participate in an `ours` development run.
The endpoint profile is explicit and fail-closed: `llmthu` reads only
`LLMTHU_KEY`/`LLMTHU_BASE_URL`, while `openai` reads only
`OPENAI_API_KEY`/`OPENAI_BASE_URL`. It never switches accounts because another
credential happens to be present in the same environment file.
Before a formal device starts, the unified script performs one authenticated
`GET /models` probe against the selected profile. Invalid credentials or an
unreachable endpoint fail before emulator startup; dry-runs and static checks
remain network-free.

## Unregistered stock T3A/M3A capture for SkyMark

SkyMark may collect immutable stock AndroidWorld step requests without adding a
method to the frozen five-method matrix. This diagnostic mode still enters only
through the unified script, uses the official task lifecycle and validator, and
caps the episode at seven decisions:

```bash
OMNIFLOW_STOCK_CAPTURE_OUTPUT_PATH=/absolute/new/attempt \
OMNIFLOW_STOCK_CAPTURE_MODEL=GLM-5.1 \
OMNIFLOW_STOCK_CAPTURE_MODEL_ENDPOINT_PROFILE=openai \
bash scripts/exp/run_androidworld.sh \
  --stock-capture m3a --tasks ContactsAddContact
```

Use `--stock-capture t3a` for the upstream text-only T3A Harness. The capture
persists exact action prompts, responses, parser results, request timings, and,
for M3A, the exact JPEG payloads sent to the model. It never exposes a reference
action to the runtime and never registers a formal experiment result. The stock
Harness and upstream prompts remain unchanged.

`--check-only` is deliberately read-only. It validates existing assets but
will fail rather than create a missing method asset:

```bash
bash scripts/exp/run_androidworld.sh --check-only
```

`--dry-run` additionally resolves the selected model endpoint and validates
MobileGPT chat/embedding capabilities when that method is selected. It exits
before creating an attempt or preflight directory, starting or stopping an
emulator, or invoking a method runner.

`--convert-ours-assets` remains available for conversion-only maintenance. It
does not mechanically create Function semantics: it accepts a complete offline
Agent response, verifies its instruction version and source RunLog hash, audits
its actions against the RunLog, then freezes and registers it without replaying
a task. By default it resolves the canonical normalized source index from
`OMNIFLOW_EXP_MEMORY_ROOT/current.json`; `OMNIFLOW_OURS_SOURCE_ASSET_INDEX` is
only an explicit immutable-index override.

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

An official-validator boolean conclusion is the terminal result for a formal
cell, even when the attempt also records parser, runtime-integrity, or
environment error evidence after task start. Registration preserves those
fields without interpreting exception names. An attempt with no boolean
official-validator conclusion remains pending and is excluded from the
completed-cell index.

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

The runtime pins the AndroidWorld adapter to commit
`632ac95959ace58c8e2ed2db8e4209cc3d9c26ef`. An explicit
`OMNIFLOW_ANDROID_WORLD_ROOT` remains the only override for a checked external
copy; when the pinned release is absent, the unified script fails closed
instead of silently using an unversioned dirty checkout.

The pinned setup clears Android's package-level `REVIEW_REQUIRED` permission
flags before launching legacy applications such as Clipper. This prevents the
Android 14 compatibility review activity from intercepting official task
initialization without relying on accessibility-visible dialog text.

If native `get_state()` reports a stale accessibility/gRPC tree, the shared
adapter force-toggles AndroidWorld's AccessibilityForwarder, rebuilds the
official wrapper, and retries that same observation once. Healthy observations
take no extra read, and unrelated state errors still fail unchanged.
If the official forwarder restart is itself temporarily unable to bind, that
failure remains inside the same bounded readiness loop instead of aborting the
cell before task start.

Before any AndroidWorld formal or development path, the unified script verifies
that the selected Python SQLite can create an FTS4 virtual table. If the managed
Python lacks FTS4, it selects and validates a compatible system `libsqlite3`
through `LD_PRELOAD`; setup fails closed when no compatible library exists. This
keeps the official Joplin database setup unchanged while making the required
host capability persistent across every task and batch.

During official app setup, every AndroidWorld adb file push/copy has the
script-owned positive timeout above. The adapter preserves any explicit
positive upstream timeout, but replaces upstream `None` or `0` so a dead adb
transfer terminates as an environment failure instead of blocking a batch.
The complete official setup call also has a separate 300-second hard deadline.
This covers AndroidEnv coordinator stalls where an adb child has already exited
but the upstream future never resolves; the cell remains an environment failure
with no fabricated validator conclusion, and the batch can advance.
The pinned AndroidWorld setup retries an app once with UIAutomator when its
native accessibility-backed setup raises `ValueError`. It saves the app snapshot
only after setup succeeds; a failed retry propagates and cannot persist an
onboarding or permission-dialog snapshot.
If an app's optional notification permission has already been settled, the setup
seam treats the missing `Don't allow` button as success only when no permission
controller is visible and a real app package remains visible. Other missing setup
targets and all visible permission dialogs still fail normally.
Legacy-app setup clears package permissions marked `REVIEW_REQUIRED` before
launch, so Android 14 cannot divert setup into a compatibility-review activity.

During official app setup only, the shared AndroidWorld adapter normalizes
Unicode presentation variants in visible UI labels while preserving native
resource IDs, bounds, actions, and the original observation. After the official
setup returns, it reopens every declared app once and clears any remaining
Android permission dialog through the platform's stable deny resource IDs. It
requires the declared package to regain the foreground, closes the app, and
overwrites the official snapshot before evaluation. This handles permission
prompts that appear only after a second launch without changing any evaluated
method. AndroidWorld task initialization may restore a snapshot whose first
subsequent launch presents the same platform dialog again, so the adapter repeats
that readiness check immediately after the official agent reset, without saving
another snapshot. The first evaluated observation therefore reaches the declared
app rather than `com.google.android.permissioncontroller`.

Before each agent step, the launcher asks the pinned AndroidWorld controller to
check for the known AccessibilityForwarder crash window. Healthy steps do not
refresh the environment or read another page. When that exact crash is visible,
the controller closes the system dialog, rebuilds the official a11y wrapper,
and verifies that a native accessibility forest is available before the agent
continues.
