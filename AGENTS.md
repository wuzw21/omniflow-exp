# OmniFlow-exp Rules

## Single-owner edit map

Change each concern only in its owner file; other files may call the owner but
must not redefine its interface or lifecycle:

- public experiment entry: `scripts/exp/run_androidworld.sh`
- experiment scheduler: `src/experiment/e2e_task_pipeline.py`
- one task/device runner: `src/experiment/androidworld.py`
- Function authoring and Store writer: `omniflow/functions/assets.py`
- canonical local index: `src/experiment/local_data.py`
- AndroidWorld native host: `src/integrations/android_world/host.py`
- method selection adapters: `src/integrations/android_world/methods.py`
- external JSON-line interface: `omniflow/bridge.py`

If a change appears to need a second owner, remove the duplicate or make it a
private adapter that forwards to the owner.

This repository contains only the paper's AndroidWorld experiment and the
B-MoCA validation of the same OmniFlow method. Do not add product features,
historical campaigns, ablations, raw assets, compatibility layers, or alternate
runners.

Before changing or running this repository, read this file, `README.md`, and
`scripts/exp/README.md`. Keep them consistent.

## One path only

- `scripts/exp/run_androidworld.sh` is the only public experiment entry.
- `src/experiment/e2e_task_pipeline.py` is the only method/device scheduler.
- `src/experiment/androidworld.py` runs exactly one `task + method + device`.
- `save_function` is the only Function write API and the only path from a
  successful RunLog to Function Store persistence.
- Transfer-state catalogs are immutable RunLog evidence. Runtime construction
  reads only the registered Function Store and never creates a replacement.
- Do not add an authoring manifest converter, automatic missing-Store builder,
  second compiler, second writer, checker plugin, diagnostic runner, or alias
  for a retired interface.
- `LLMTHU_API_KEY` is the only GLM-5.1 credential variable. The endpoint is
  canonical protocol configuration; do not restore `LLMTHU_KEY` or require
  operators to duplicate it as `LLMTHU_BASE_URL`.

The retained management tools are `list_functions`, `get_function`,
`save_function`, `delete_function`, `clear_functions`, `list_run_logs`,
`get_run_log`, and `get_run_log_state`. `run_gui` is the execution tool.

## Function and checker contract

One successful `omniflow.run_log.v1` saves exactly one semantic Function in one
`save_function` call. Enhancement is optional (`enhance=true`) and uses the same
validation and Store writer as a normal save.

Enhanced authoring uses one bounded three-stage workflow over one in-memory
draft, not one JSON call per RunLog action. Stage 1 names the one complete
Function; stages 2 and 3 edit that Function for source-proven action semantics
plus parameters, then checker registrations. The Agent never
returns complete Functions, source states,
bindings, checker rules, Stores, or authoring manifests. It may request only
the two action decisions defined by the shared schema: mark an eligible launcher
click as `open_app`, or mark an eligible visible click as `set_target`. The Agent
never returns the package or target value; the compiler copies it from validated
RunLog evidence.

Method improvement changes the shared authoring policy, evidence supplied to
the Agent, deterministic compiler, or runtime adapter; it never hand-edits a
generated Function or Store. Regenerate the draft from the same successful
RunLog through `save_function` after a policy repair. The enhancer may make
multiple bounded model calls, but every call edits only one small draft
decision and a rejected decision may receive only its own validation feedback.

`save_function` deterministically preserves source action order and states,
validates every requested action edit against the before/after RunLog states,
compiles parameter schemas and bindings, registers checkers, and emits one
large Function covering the complete successful trajectory. The Agent may not
split the trajectory or emit a second Function. A checker action may not also
remain a formal action in the same Function.

For enhanced authoring, `source_calls` contains exactly the complete Function
call that reproduces the successful RunLog.

The single model-facing tool is `edit_function_draft`. Its three strict stage
schemas return: `complete_function`, then `action_edits +
bindings + optional source-grounded actions`, then `checker_steps`. The latter two schemas are requested separately
for each Function and may refer only to that Function's listed source indices.
An action edit contains only `function_id`, `step_index`, and `operation`. Bridge
and experiment adapters import these schemas and select the supplied tool name
instead of defining another authoring contract. Each stage gets at most three
model attempts. A rejected decision receives only that stage's deterministic
validation error before the Agent revises the same in-memory draft; transport
failures and missing evidence fail immediately. Nothing is persisted
until all three edits compile and all Functions pass the same authoritative
validation and sole Store writer.

Action semantics are source evidence, not ungrounded generation. The action
stage may return direct canonical actions for exploration, but `save_function`
accepts them only when they exactly match the selected RunLog step or a
deterministic source-proven semantic edit. `open_app`
requires a source launcher page and a different non-empty package in the
RunLog after-state; the compiler copies that package. `set_target` requires an
exact visible label at the source action point; the compiler copies that label.
Caller-varying visible targets such as an hour or category bind
`target_description`; typed content binds `text`. The compiler derives this
path from the validated action; the Agent does not author it. Coordinates,
packages, waits, and directions are never parameters. A bound source value must occur directly
in the RunLog goal; a current page value absent from the goal is source state,
not caller input.

Before a saved B-MoCA bundle may run cross-environment, its complete Function
must pass env100 `script_replay` with official success, method success,
`model_calls=0`, and `fallback_steps=0`. A failed source gate ends that task;
never prepare or launch env101--109 or `omniflow` for an unqualified Function. Do
not clone any B-MoCA AVD before Function enhancement succeeds; prepare env100
for the source gate first and the remaining AVDs only after that gate passes.

Checker rules are registered on one Function through that Function's
`checker_rules`; there is no global checker pool. A rule belongs only to the
Function that saved it. Each rule contains a RunLog source state, its source
action, and no other fields. There is no step number trigger and no trigger DSL.

The entire configurable checker surface is exactly the Function-local
`checker_rules` registration plus the one global target probability threshold
in `protocol.checker`. Evaluation cadence is invariant: before every pending
formal action, check every still-unexecuted rule registered on that Function.
Frequent evaluation must not weaken execution: every fixed condition below
still has to pass, so an unrelated page or target can never trigger a checker.

Before every pending formal Function action, runtime checks each unexecuted rule
registered on that Function. A checker executes once only when all conditions
hold:

1. the rule is registered on the active Function;
2. OmniTransfer maps the source action onto a target on the current observation;
   and
3. the selected target's OmniTransfer rank probability reaches the configured
   high-confidence threshold.

A failed condition skips the checker and leaves it eligible before a later
formal action. A checker that executed stays complete if the same Function
invocation resumes after a formal-action failure. A new Function invocation
starts a new checker session. Allowed checker actions are `click`, `input_text`,
and `long_press` with source target coordinates used only as OmniTransfer evidence.
Never execute source-device coordinates on the target.
An action whose visible source target names progress stated in the RunLog goal
or Function name/description is a formal action and cannot be a checker.
One RunLog action may not be a checker in one emitted Function and a formal
step in another; conflicting roles reject the entire atomic save.

The global target-probability threshold is defined only in the
`protocol.checker` block of `config/paper_androidworld.json`. Pair confidence
and page-embedding similarity is not a trigger or recall signal. Per-rule thresholds
and condition switches are forbidden because they recreate a trigger language.
Formal Function actions go directly through canonical OmniTransfer target
mapping; a missing or rejected mapping returns control to the normal Planner
fallback without source-coordinate execution.

Function success is an ordinary Planner tool result, not AndroidWorld task
completion. The Planner may call more Functions or GUI actions before it
explicitly finishes.

### Function generation and success criteria

Function generation always starts from one complete, immutable, successful
RunLog. The RunLog must contain the real native observations, action sequence,
before/after states, screenshot references, task goal, device/source metadata,
and the official environment outcome. A RunLog that is incomplete, missing
screenshots or observations, has an invalid schema/hash, or fails the official
validator is evidence of a failed attempt only; it must never be converted into
an executable Function.

The only generation pipeline is:

1. `save_function` loads and validates the RunLog and checks its exact source
   lineage and immutable evidence.
2. With `enhance=false`, the deterministic compiler creates the complete
   source-grounded Function from the RunLog action/state sequence.
3. With `enhance=true`, `edit_function_draft` performs the bounded three-stage
   workflow: name the complete trajectory, edit only
   source-proven action semantics and bindings, then register Function-local
   checker steps. The Agent may propose decisions, but the compiler copies
   packages, targets, states, coordinates, and action evidence from the
   validated RunLog and rejects invented values.
4. The compiler preserves the complete ordered trajectory as the one large
   Function and never emits a split recall candidate.
5. The sole validator checks schemas, source indices, action/state transitions,
   parameter bindings, checker roles, transfer evidence, and Function/Store
   invariants. The sole Store writer then writes the Function Store atomically,
   records exact hashes and lineage in the canonical `data/current.json` index.

There are three distinct success levels and they must not be conflated:

- **Authoring success:** `save_function` completes without validation errors,
  writes a valid `omniflow.store.v2`, and the Store can be loaded through
  `get_function`/`list_functions` from the canonical index. This proves only
  that the Function was generated and persisted.
- **Function replay success:** the complete Function is selected and executed
  through the canonical OmniTransfer/runtime path, with no source-coordinate
  passthrough. Every formal action maps or falls back normally, the Function
  invocation returns success, and no evidence or checker invariant is violated.
  A successful tool result alone is not official task completion.
- **Official task/source-gate success:** the replayed task passes the official
  AndroidWorld validator or B-MoCA reward, and the required method/result
  contract also passes. For the B-MoCA env100 source gate, the required
  conditions are official success, method success, `model_calls=0`, and
  `fallback_steps=0`; only then may cross-environment replay proceed. For
  AndroidWorld source qualification, the canonical source must be seed 111,
  have screenshots and real RunLog evidence, pass the official validator, and
  use zero model calls. A Function is not labelled successful merely because
  authoring succeeded or because the Planner returned a tool result.

If any stage fails, the attempt remains a failure/trace bundle with its error
and evidence. No partial draft, invalid Function, checker-only fragment, or
missing-Store replacement may be persisted as a runnable Function. Repair the
shared policy/compiler/runtime seam and regenerate through `save_function` from
the same successful RunLog; never hand-edit a generated Function Store.

## OmniTransfer boundary

The canonical checkout is always `~/Projects/Omni/OmniTransfer`. Use its real
candidate mapper and latest page embedding. OmniTransfer returns ranked target
candidates and evidence; OmniFlow owns page checks, candidate selection,
execution, failure classification, and VLM fallback.

The only active page encoder is `omniflow/transfer/page_embedding.py`, backed by:

`src/omnitransfer/checkpoints/omnitransfer_spatial_xml_alignment_v9_20260805/v9_spatial_xml_alignment_seed29.pt`

Do not add native 512D, page-word, 1024D, local pooling, node/resource-id lookup,
coordinate passthrough, or another page encoder. Missing or invalid transfer
evidence is an explicit failure and returns to normal VLM fallback.

## Formal experiment contract

The atomic result is exactly one `task + method + device`. Do not reintroduce
cell protocols or names. Formal methods are exactly `fixed_replay`, `omniflow`,
`mobilegpt`, `appagent`, and `t3a_hint`.

The only formal configuration is the `protocol` block of
`config/paper_androidworld.json`. `src/experiment/protocol.py`, shell, runners,
and reports are derived views. Methods, devices, seeds, budgets, timeouts,
model endpoint, fold state, and pinned revisions must not be copied elsewhere.
Device serials, console ports, AVD names, API levels, and emulator profiles are
part of those protocol device records; shell and source collection must derive
them instead of naming an AVD locally.
The source AVD uses the API-33 `small_phone` geometry recorded by the retained
720x1280 source RunLogs; source collection must not scale those coordinates
onto a different display profile.
The shell provisions that configured source AVD before dispatching the E2E
pipeline, and the pipeline must report an exited emulator process immediately
instead of waiting for the full boot deadline.
Development overrides must be explicit. Retired source/format/accept/first/limit
selectors are historical reader fields, not active options.

Existing prompts and external baseline contracts are frozen. Do not add
task-specific prompts, accumulated planner history, guidance plumbing, hidden
retries, evaluator-aware completion, or baseline repairs.

New public result rows contain only: `task`, `method`, `device`, `source_seed`,
`evaluation_seed`, `status`, `validator_success`, `model_calls`,
`prompt_tokens`, `completion_tokens`, `total_tokens`, `actions_executed`,
`episode_duration_sec`, `outer_wall_sec`, `error`, and `evidence_paths`.
Preparation and component diagnostics belong once in a `details` evidence block.
Registration keeps that same compact-row/details split plus one immutable
ledger; do not recreate master matrix, run-record, or per-method summary tables.

## Execution and memory

### Canonical Python/Torch runtime

All OmniFlow-exp commands, tests, Function conversion, AndroidWorld setup, and
B-MoCA execution use the repository-local interpreter:

`~/Projects/Omni/OmniFlow-exp/.venv/bin/python`

This environment is the sole Python/Torch runtime for this repository and is
currently Python 3.12.11 with Torch 2.13.0. Do not use any neighboring
OmniFlow environment, Conda Python, system `python3`, or another Torch
installation. The public shell entry defaults to this interpreter and
rejects retired neighboring runtimes. `PYTHON_BIN` is reserved for an
explicitly provisioned equivalent test runtime, never for formal execution.

### Single artifact storage

All AndroidWorld RunLogs, screenshots, converted source evidence, Function
Stores, transfer-state catalogs, method memory, and registered result evidence
are stored under exactly one stable local root:

`~/Projects/Omni/OmniFlow-exp/data`

`data/current.json` is the only runtime entry point and contains the complete
`omniflow.local-artifact-index.v1` index inline: source rows, canonical
Function Stores, method memory, and result rows. There is no second registry,
`indexes/` directory, `snapshots/` directory, or per-task pointer. The
canonical local data is materialized one item at a time from selected evidence
and then treated as immutable. Do not create parallel `current.json` files,
scratch memory roots, compatibility stores, or duplicate converted copies.

Use this fixed `task_device_c` classification inside `data/` for every saved
bundle:

```text
data/<environment>/<task>/<device>/<category>/<method>/<attempt_id>/
  run_log.json                 # when the bundle contains a RunLog
  transfer_states.json         # when transfer evidence exists
  screenshots/                 # content-addressed screenshot files
  trace/                       # XML/native/action trace files
  function_store.json          # Function Store and Function-local checkers
  result.json                  # compact public result row
  details.json                 # detailed evidence companion
```

`environment` is `androidworld` or `bmoca`; `task` is the official task name;
`device` is the protocol device or B-MoCA environment (`source5560`,
`small5554`, `fold5564`, `env100`, and so on); `category` is one of `source`,
`function`, `checker`, `trace`, `replay`, `result`, or `memory`; and `method` is
the exact protocol/reuse method or `function_authoring` for shared authoring.
Source and shared Function evidence still use the real source device and
`function_authoring`, so every item is addressable as task + device + category
(`task_device_c`) and method-bearing evidence never loses its method dimension.
B-MoCA method selectors are a separate external contract documented only in
`scripts/exp/README.md`; do not copy them into AndroidWorld method code.

The inline `current.json` record carries the bundle classification, lineage,
and hashes; do not create a second manifest, provenance registry, checker file,
or per-task pointer for the same bundle. `save_function` keeps checker rules in
the Function Store and writes no checker/provenance/manifest sidecars.
Inner files must use the latest applicable repository schema: `omniflow.run_log.v1`,
`omniflow.function.v2`, `omniflow.store.v2`,
`omniflow.transfer-state-catalog.v1`, or the current method/result schema
defined by its loader. Convert legacy or malformed input in memory, validate
it, and write only the converted object plus its original hash in the inline
record; never retain an invalid runtime copy. Writes are one bundle at a time
and atomic; an incomplete bundle is not indexed or executable.

Save task-major, one bundle at a time, and resolve only `data/current.json`
during execution. Historical external locations and old task-local layouts are
read-only migration inputs. After conversion, runtime must not visit them and
must not create another task-only, method-only, or external artifact root.

Run task-major and complete one task before advancing. Resolve
`data/current.json` first and skip every formal result with
an existing official-validator conclusion. Formal results and original attempts
are immutable.

For each unfinished task:

1. run the static gate and verify the source-seed-111 RunLog, exact hashes,
   Function Store, transfer states, and canonical OmniTransfer checkout;
2. check Function recall, Planner selection, and offline replay;
3. qualify the Function on the source contract with official validator success,
   `model_calls=0`, and `fallback_steps=0`;
4. run at most three unregistered `omniflow` development iterations on SmallPhone,
   then Pixel Fold;
5. freeze the version and fill only missing formal results.

Every AndroidWorld/B-MoCA check, conversion, development episode, formal result,
or memory refresh enters through `scripts/exp/run_androidworld.sh`. Function
authoring itself enters through `save_function`; a missing Store blocks an
experiment and is never generated by the shell.
Source-only collection validates the selected task and its frozen source
lineage; it must not require an unrelated global source-index task count.
It may use an immutable official-successful historical RunLog as the fixed
action template, but the newly captured RunLog must execute at source seed 111,
contain screenshot references, use zero model calls, and pass the official
validator. Normal qualification and formal execution still reject a non-111
canonical source.

The shared AndroidWorld setup seam may resolve the system chooser for official
Contacts setup only when `Just once` is visible and the title is either
`Open with` with a visible `Contacts` choice, or `Open with Contacts` after that
choice is already selected. It selects only the missing choice, confirms
`Just once`, then resumes the official onboarding `Skip`. Other chooser states
remain setup failures.

For explicitly authorized source-data collection only, a single-task direct
AndroidWorld collector may use the pinned checkout's native emulator,
`env_launcher`, `TaskRegistry`, `get_state()`, `execute_action()`, and the
official task validator. This mode is limited to immutable successful seed-111
RunLogs with screenshots, native observations, and decision records; it must
not invoke experiment methods or register formal results.

Use AndroidWorld native state/action and its official validator. B-MoCA is an
environment adapter using the same OmniFlow Function/checker/OmniTransfer
runtime and official B-MoCA reward. `omniflow` lets the Planner select Functions;
`script_replay` selects the one complete Function directly with its saved
source-call arguments, but may not own a second action mapper or executor.

The AndroidWorld `omniflow` adapter runs exactly one persistent `OmniFlow.run()`
cycle per task. The official episode runner's outer `step()` call is only an
adapter invocation; it may not recreate OmniFlow with `max_steps=1`, accumulate
separate partial RunResults, or own Function resume/fallback state. The
official complexity budget may lower the canonical planner budget but never
raise it.

Local and host `9207` active checkouts are
`~/Projects/Omni/OmniFlow-exp` on `main`. Before remote execution, both full
commit SHAs must match and both tracked worktrees must be clean. Synchronize only
through Git.

## Code and data boundaries

- contracts and data types: `omniflow/core/`
- Function lifecycle: `omniflow/functions/`
- execution: `omniflow/runtime/`
- transfer: `omniflow/transfer/`
- VLM planning: `omniflow/vlm/`
- experiment code: `src/experiment/` and `src/integrations/`
- external bridge: `omniflow/bridge.py`

Do not commit RunLogs, screenshots, XML dumps, weights, APKs, emulator images,
baseline memories, credentials, attempts, or result tables. Assets are supplied
through explicit absolute paths and indexed by exact SHA-256 outside the repo.

The retired source-pool writer, `index_by_task` archive, and legacy source
converter are removed. Successful source RunLogs enter the canonical data
index and then `save_function`; no parallel source archive may be recreated.

`tools/manual_androidworld_harness.py` is human-only diagnosis. It cannot create
formal results, refresh canonical memory, or replace the unified shell entry.

RunLog `open_app` actions keep the canonical package as their stored contract.
At execution, the shared AndroidWorld host resolves a registered launcher name
when one exists and otherwise preserves the package for AndroidWorld's official
package-launch fallback. AndroidWorld's own ADB helper closes stale tasks. Do
not add a local app mapping, a pre-launch registry gate, or another launcher.
