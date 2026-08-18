# AndroidWorld experiment entry point

`run_androidworld.sh` is the only public AndroidWorld/B-MoCA launcher. Python
modules are implementation seams and must not be invoked as alternate runners.

## Commands

| Purpose | Command |
| --- | --- |
| Task-major formal run | `run_androidworld.sh --tasks TASK` |
| Full indexed run | `run_androidworld.sh --all-tasks` |
| Read-only static gate | `run_androidworld.sh --check-only [--all-tasks]` |
| Bounded `ours` development | `run_androidworld.sh --development-run --tasks TASK` |
| Source refresh | `run_androidworld.sh --collect-source --tasks TASK` |
| B-MoCA one method | `run_androidworld.sh --environment bmoca --method ours\|script_replay --tasks TASK` |
| B-MoCA campaign | `run_androidworld.sh --environment bmoca --all-tasks [--tasks TASK1,TASK2]` |
| Memory refresh | `run_androidworld.sh --refresh-memory` |

Source refresh validates the selected task and its frozen source lineage; it
does not require unrelated tasks to exist in the source index.
The historical successful RunLog supplies only the fixed action template. The
new capture runs at source seed 111, records screenshots, makes zero model
calls, and must pass the official validator before it is reported as collected.
Its isolated source AVD uses the canonical API-33 `small_phone` profile so the
recorded 720x1280 action contract is replayed at its original geometry.
The shell provisions the configured source AVD before handing control to the
pipeline; an emulator process that exits during boot is an immediate failure.

Both B-MoCA methods use the same Function/checker/OmniTransfer executor. `ours`
uses the Planner to select a Function; `script_replay` directly selects the
single complete Function and makes no model call.

One formal result is one task, one method, and one device. The E2E pipeline is
the only method/device scheduler. Direct `--method` and `--device` options are
internal single-result controls and cannot be combined with matrix modes.

## Required external roots

```bash
export OMNIFLOW_EXP_ASSET_ROOT=/absolute/assets
export OMNIFLOW_EXP_RESULTS_ROOT=/absolute/results
export OMNIFLOW_EXP_MEMORY_ROOT=/absolute/memory
export OMNIFLOW_ENV_FILE=/absolute/model.env
export OMNITRANSFER_ROOT="$HOME/Projects/Omni/OmniTransfer"
```

The launcher uses the single runtime at `../OmniFlow/.venv/bin/python` by
default. Set `PYTHON_BIN` only when that canonical workspace runtime lives at a
different absolute path. Development preflight loads the canonical
OmniTransfer page encoder before starting an emulator, so a broken Torch or
checkpoint installation fails without consuming an episode.

AndroidWorld, MobileGPT, and AppAgent checkouts may be supplied through their
documented absolute root variables. Credentials remain in `OMNIFLOW_ENV_FILE`.
Formal protocol values are not environment configuration: they come only from
`config/paper_androidworld.json`.
The GLM-5.1 credential name is `LLMTHU_API_KEY`; do not duplicate the canonical
endpoint as `LLMTHU_BASE_URL` in the environment file.

## Function preparation

Function authoring does not run through a shell conversion mode. Use the bridge
`save_function` API with one successful RunLog. Callers may submit complete
Functions, or set `enhance=true` so the Agent edits one in-memory Function draft
in exactly three model calls: semantic Function ranges, parameter declarations,
and Function-local checker registrations. Every reusable subsegment must carry
a non-empty `stability_reason` explaining why it is a deterministic contiguous
source-state/action sequence rather than a transient-dialog, task-completion,
or one-click fragment. `save_function` copies exact evidence and
deterministically compiles the complete Function plus accepted subsegments,
bindings, and checker rules. Normal and enhanced saves share one validation and
Store writer. One invalid stage edit gets one correction opportunity; a timeout
or transport failure is not retried, and nothing is persisted until every
compiled Function passes the same validator.

The explicit B-MoCA campaign is the only launcher-owned preparation path: for
each corpus task it calls that same `save_function(enhance=true)` writer once.
It then runs only `script_replay` on env100 and requires official success,
method success, `model_calls=0`, and `fallback_steps=0`. A failed source gate
ends that task without launching env101--109. A passing gate unlocks
Planner-selected `ours` on env100--109 and the remaining zero-model
`script_replay` results on env101--109 through the same OmniFlow runtime. It
writes `progress.csv`,
`progress.jsonl`, per-attempt RunLogs, and the terminal
`campaign_summary.json` under the new output root.

After saving, ingest the external Function catalog with `--refresh-memory`.
Experiment execution resolves the task's Store from `current.json`. If no Store
is registered, the command fails with an explicit message; it never calls a
converter or model to create one.

## Runtime flow

For each unfinished task, the pipeline:

1. resolves the official-successful source-seed-111 RunLog and exact hash;
2. resolves the registered Function Store and transfer states;
3. qualifies `ours` on the source device with official validator success,
   `model_calls=0`, and `fallback_steps=0`;
4. resolves method-native MobileGPT and AppAgent memory from the same RunLog;
5. runs the fixed five methods on SmallPhone and unfolded Pixel Fold; and
6. registers each official-validator conclusion immediately.

The task deadline is shared by the whole pipeline. Existing immutable results
are skipped before emulator startup. Formal runs use cold restart and official
AndroidWorld setup; only repeated development runs may set
`OMNIFLOW_ANDROIDWORLD_PERFORM_EMULATOR_SETUP=0` for an already initialized
live emulator.

The shared lifecycle seam keeps AndroidWorld directory cleanup idempotent when
host gRPC diagnostics pollute ADB stdout. It accepts Markor's absent final `OK`
only after the Markor main activity is already foregrounded; every other setup
error remains a failure. When official Contacts setup is blocked by Android's
`Open with` chooser, the seam selects `Contacts` and `Just once`, then resumes
the official onboarding `Skip`; any other chooser state remains a failure. The
same seam removes only gRPC fork diagnostics from
AndroidWorld ADB response payloads before official task code parses them, and
retries APK installation without `--bypass-low-target-sdk-block` only when the
emulator explicitly reports that option as unknown.

`open_app` keeps the RunLog package as its stored contract. The adapters derive
the official launcher name from the pinned AndroidWorld registry when present;
otherwise the same official launcher receives the package and uses its package
fallback. AndroidWorld's own ADB helper closes stale tasks. There is no local
app registry, pre-launch gate, or second launcher.

For `ours`, the AndroidWorld Method Adapter invokes one complete
`OmniFlow.run()` cycle on the task. The official episode runner contributes the
native lifecycle and may lower the canonical step budget; it does not split the
Planner into repeated one-step OmniFlow runs or maintain separate resume and
fallback state.

## Checker execution

Only rules registered on the active Function are considered. Before each
pending formal Function action, the latest canonical OmniTransfer page
embedding first matches each unexecuted rule's source state to the current
page. OmniTransfer then maps that rule's source action onto the current
observation. It executes only when both configured thresholds pass. A rule
contains only `source_state_id` and `action`; a failed condition skips it and
keeps it eligible for a later formal action. An executed rule remains complete
if that Function invocation resumes. Pair confidence is not a trigger.
Source coordinates are evidence only and never execute on a target.

The only configurable checker choices are the rules registered in each
Function and the two global `protocol.checker` thresholds. Evaluation cadence
is fixed: all still-unexecuted rules are checked before every pending formal
action. A Function with no registered rules performs no checker evaluation,
and rules registered on another Function are never considered.

The same page threshold gates every source-state-dependent formal Function
action before transfer. `open_app` and `wait` are state-independent. A mismatch
fails the Function into the normal Planner fallback instead of ranking targets
on the wrong page.

## Configuration ownership

The `protocol` block in `config/paper_androidworld.json` owns methods, devices,
seeds, budgets, timeouts, model endpoint, fold state, AVD names, API levels,
emulator profiles, and pinned revisions.
`src/experiment/protocol.py` and this shell only read those values. Active user
configuration is limited to external roots, credentials, task selection, and
explicit development inputs.

Retired source backend, source format, accepted/first/limit filters, cell modes,
authoring manifests, conversion output roots, and revision-reason flags are not
accepted. Historical readers may recognize old evidence fields but no new
command or metadata emits them.

## Long-term memory

`OMNIFLOW_EXP_MEMORY_ROOT/current.json` is the sole runtime index. Refresh it
with:

```bash
OMNIFLOW_MEMORY_RUNLOG_ROOTS=/absolute/runlogs \
OMNIFLOW_MEMORY_RESULT_ROOTS=/absolute/results \
OMNIFLOW_MEMORY_FUNCTION_CATALOGS=/absolute/function-catalog.json \
bash scripts/exp/run_androidworld.sh --refresh-memory
```

Memory is content-addressed by exact SHA-256. Refresh updates canonical indexes;
it does not rewrite or delete original evidence. Ambiguous, missing, or corrupt
entries are preflight failures.

## Result records

New public rows use only the 16 fields defined in `AGENTS.md`. Preparation,
reuse, baseline conversion, and component-level details are stored once in the
attempt evidence. Registration writes those compact rows, their single details
block, and one immutable ledger. It does not maintain parallel master-progress
tables. Formal attempts and validator conclusions are immutable.

## Development example

```bash
OMNIFLOW_ANDROID_WORLD_ROOT=/absolute/AndroidWorld \
OMNIFLOW_ANDROIDWORLD_STORE_PATH=/absolute/store.json \
OMNIFLOW_DEVELOPMENT_OUTPUT_PATH=/absolute/new-attempt \
OMNIFLOW_DEVELOPMENT_MODEL=GLM-4.6V \
bash scripts/exp/run_androidworld.sh \
  --development-run --tasks ExpenseAddMultipleFromGallery
```

Development is unregistered and bounded. It must not change frozen prompts,
baseline policies, official validation, or the formal protocol.
