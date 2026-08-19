# AndroidWorld experiment entry point

`run_androidworld.sh` is the only public AndroidWorld/B-MoCA launcher. Python
modules are implementation seams and must not be invoked as alternate runners.

## Commands

| Purpose | Command |
| --- | --- |
| Task-major formal run | `run_androidworld.sh --tasks TASK` |
| Full indexed run | `run_androidworld.sh --all-tasks` |
| Read-only static gate | `run_androidworld.sh --check-only [--all-tasks]` |
| Bounded `omniflow` development | `run_androidworld.sh --development-run --tasks TASK` |
| Source refresh | `run_androidworld.sh --collect-source --tasks TASK` |
| Provider contract tests | `bash scripts/exp/test_provider.sh mobilegpt|appagent|all` |
| OOB development/source transport | `run_androidworld.sh --control-backend oob --development-run --tasks TASK` |
| B-MoCA one reuse method | `run_androidworld.sh --environment bmoca --method ours_replay\|mobilegpt_replay\|skilldroid_replay --tasks TASK` |
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

The B-MoCA campaign measures oracle-memory-hit reuse rather than retrieval.
`ours_replay` directly invokes the complete Function with its saved source-call
arguments and canonical checker/OmniTransfer runtime. `mobilegpt_replay` uses
MobileGPT's native task memory and parameter-filling reuse path with exploration
disabled. `skilldroid_replay` compiles the env100 RunLog into DroidRun v0.5.6's
official `macro.json` format and replays it through DroidRun's native
`MacroPlayer`, with every model fallback disabled. Its B-MoCA `DeviceDriver`
adapter preserves official benchmark action/reward recording while keeping the
macro's source absolute pixels unchanged; DroidRun macro replay performs no
locator lookup or state verification.

One formal result is one task, one method, and one device. The E2E pipeline is
the only method/device scheduler. Direct `--method` and `--device` options are
internal single-result controls and cannot be combined with matrix modes.

## Provider integration: one place to start, one command to test

MobileGPT and AppAgent are different upstream contracts, so their conversion
implementations remain separate. Each provider has one public preparation
owner:

| Provider | Start editing here | Provider-specific implementation | Test command |
| --- | --- | --- | --- |
| MobileGPT | `src/experiment/mobilegpt_source.py` | `src/integrations/mobilegpt.py` and `src/integrations/mobilegpt_memory.py` | `bash scripts/exp/test_provider.sh mobilegpt` |
| AppAgent | `src/experiment/appagent_source.py` | `src/integrations/appagent.py` | `bash scripts/exp/test_provider.sh appagent` |

The source file is the provider's public seam: it owns `prepare`, `validate`,
and `preflight`, and is the first place to change when the provider's input or
output contract changes. Only follow the integration file when the failure is
inside the upstream format or runtime mapping. Do not add a provider branch to
`androidworld.py`, `e2e_task_pipeline.py`, or the shell scheduler.

The shared harness runs the provider's focused tests, the matching shell
integration tests, and its CLI help check. It is offline: it does not start an
emulator, call a model, create memory, or write `data/current.json`. Formal
execution still has exactly one entry, `run_androidworld.sh`.

## Required roots

```bash
export OMNIFLOW_EXP_ASSET_ROOT=/Users/wuzewen/Projects/Omni/OmniFlow-exp/data
export OMNIFLOW_EXP_RESULTS_ROOT=/Users/wuzewen/Projects/Omni/OmniFlow-exp/data
export OMNIFLOW_EXP_MEMORY_ROOT=/Users/wuzewen/Projects/Omni/OmniFlow-exp/data
export OMNIFLOW_ENV_FILE=/absolute/model.env
export OMNITRANSFER_ROOT="$HOME/Projects/Omni/OmniTransfer"
```

The launcher uses the single runtime at `OmniFlow-exp/.venv/bin/python` by
default. Set `PYTHON_BIN` only for an explicitly provisioned equivalent test
runtime; the formal experiment must use the repository-local environment.
Install the B-MoCA baseline runtime with `uv sync --extra bmoca`; the launcher
checks the installed DroidRun version against the protocol-pinned v0.5.6 before
running a B-MoCA campaign or a direct `skilldroid_replay` result.
The campaign also requires the env100 source AVD produced by B-MoCA's official
environment installer before Function enhancement begins. This is a read-only
asset gate: the E2E scheduler still creates its isolated env100 clone only after
enhancement succeeds, and it does not clone env101--109 until env100 qualifies.
Development preflight loads the canonical
OmniTransfer page encoder before starting an emulator, so a broken Torch or
checkpoint installation fails without consuming an episode.

AndroidWorld, MobileGPT, and AppAgent checkouts may be supplied through their
documented absolute root variables. Credentials remain in `OMNIFLOW_ENV_FILE`.
Formal protocol values are not environment configuration: they come only from
`config/paper_androidworld.json`.
The GLM-5.1 credential name is `LLMTHU_API_KEY`; the endpoint is read only from
the protocol configuration.

## Function preparation

Function authoring does not run through a shell conversion mode. Use the bridge
`save_function` API with one successful RunLog. The Store contains exactly one
complete Function; set `enhance=true` so the Agent edits one in-memory draft
through three stages: name the complete trajectory, edit source actions plus
parameter declarations, then register Function-local checkers.
The middle edit may mark an eligible launcher click as `open_app`, an eligible
visible click as `set_target`, or return a direct action copied from the shown
RunLog source step for exploration. The compiler copies or validates the exact
RunLog package, target, coordinates, and action; ungrounded direct actions are
rejected. `save_function` preserves exact evidence and action order, rejects
invented semantic edits, and deterministically compiles the complete Function,
bindings, and checker rules. Normal and enhanced saves share one validation and
Store writer. Each stage gets at most three model attempts; a rejected decision
receives only that stage's deterministic validation error before revising the
same in-memory draft. A timeout or transport failure is not retried, and nothing
is persisted until the complete Function passes the same validator.
For an enhanced Store, `source_calls` contains exactly the complete Function
call that reproduces the successful RunLog.
Descriptions may claim only effects caused inside the selected source range.
A bound value must appear directly in the RunLog goal; an unrequested current
page value is source state rather than caller input.
The compiler derives parameter paths from validated actions: `input_text`
binds `text`, and a source-proven semantic click binds `target_description`.
The Agent does not author a path.

The explicit B-MoCA campaign is the only launcher-owned preparation path: for
each corpus task it calls that same `save_function(enhance=true)` writer once,
then qualifies `ours_replay` on env100 with official success, method success,
`model_calls=0`, and `fallback_steps=0`. A failed source gate ends that task
without launching env101--109. A passing gate compiles the MobileGPT and
DroidRun memories from the same env100 RunLog, then runs `ours_replay` on
env101--109 and both baselines on env100--109. All lineage and hashes are
registered in `data/current.json`; no Function or replay bundle writes a
provenance sidecar. The scheduler creates no AVD
clone before enhancement succeeds, clones env100 for the source gate, and
clones env101--109 only after that gate passes. It writes
`progress.csv`,
`progress.jsonl`, per-attempt RunLogs, and the terminal
`campaign_summary.json` under the new output root.

After saving, refresh the local data index with `--refresh-memory`.
Experiment execution resolves the task's Store from `current.json`. If no Store
is registered, the command fails with an explicit message; it never calls a
converter or model to create one.

## Runtime flow

For each unfinished task, the pipeline:

1. resolves the official-successful source-seed-111 RunLog and exact hash;
2. resolves the registered Function Store and transfer states;
3. qualifies `omniflow` on the source device with official validator success,
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
chooser, the seam requires `Just once` and accepts either `Open with` plus a
visible `Contacts` choice or the already-selected title `Open with Contacts`.
It selects only the missing choice, confirms `Just once`, then resumes the
official onboarding `Skip`; any other chooser state remains a failure. The
same seam removes only gRPC fork diagnostics from
AndroidWorld ADB response payloads before official task code parses them, and
retries APK installation without `--bypass-low-target-sdk-block` only when the
emulator explicitly reports that option as unknown.

`open_app` keeps the RunLog package as its stored contract. The adapters derive
the official launcher name from the pinned AndroidWorld registry when present;
otherwise the same official launcher receives the package and uses its package
fallback. AndroidWorld's own ADB helper closes stale tasks. There is no local
app registry, pre-launch gate, or second launcher.

For `omniflow`, the AndroidWorld Method Adapter invokes one complete
`OmniFlow.run()` cycle on the task. The official episode runner contributes the
native lifecycle and may lower the canonical step budget; it does not split the
Planner into repeated one-step OmniFlow runs or maintain separate resume and
fallback state.

## Checker execution

Only rules registered on the active Function are considered. Before each
pending formal Function action, OmniTransfer maps every unexecuted rule's source
action onto the current observation. It executes only when the selected target
passes the one configured high-probability threshold. A rule
contains only `source_state_id` and `action`; a failed condition skips it and
keeps it eligible for a later formal action. An executed rule remains complete
if that Function invocation resumes. Pair confidence is not a trigger.
Source coordinates are evidence only and never execute on a target.

AndroidWorld episode preparation resolves a visible system app chooser and
permission obstruction through the shared native adapter before the first
observation. This setup recovery is state-based experiment plumbing; it must
never be encoded into a generated Function or task-specific prompt.

The only configurable checker choices are the rules registered in each
Function and the one global `protocol.checker` target threshold. Evaluation
cadence is fixed: all still-unexecuted rules are checked before every pending
formal action. A Function with no registered rules performs no checker
evaluation, and rules registered on another Function are never considered.
An action whose source target is named by the task goal or Function semantics
is task progress and cannot be registered as a checker.
The same RunLog action cannot have checker and formal roles across two emitted
Functions.

Formal Function actions use canonical OmniTransfer target mapping directly.
A missing or rejected mapping fails the Function into the normal Planner
fallback. Page-embedding similarity is not a checker or Function-step trigger.

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

`OMNIFLOW_EXP_MEMORY_ROOT/current.json` is the sole runtime index. The default
root is `../OmniFlow-exp/data`; refresh it
with:

```bash
OMNIFLOW_MEMORY_RUNLOG_ROOTS=/absolute/runlogs \
OMNIFLOW_MEMORY_RESULT_ROOTS=/absolute/results \
bash scripts/exp/run_androidworld.sh --refresh-memory
```

Memory is content-addressed by exact SHA-256. Refresh atomically rewrites the
single `data/current.json` index and does not rewrite original evidence.
Function bundles are discovered from their canonical directory and contain the
RunLog, Function Store, and transfer states together. Ambiguous, missing, or
corrupt entries are preflight failures.

## Result records

New public rows use only the 16 fields defined in `AGENTS.md`. Preparation,
reuse, baseline conversion, and component-level details are stored once in the
attempt evidence. Registration writes those compact rows, their single details
block, and one immutable ledger. It does not maintain parallel master-progress
tables. Formal attempts and validator conclusions are immutable.

Performance measurement is an explicit side channel. The launcher option
`--collect-performance` writes one `performance_sidecar.json` beside the
episode artifacts. It reports Host/native `observe`/`act` timing and optional
ADB energy diagnostics, but does not modify task results, batch details, or the
public result rows. ADB charge-counter estimates are diagnostic only and are
not a replacement for hardware power-analyzer measurements.

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
