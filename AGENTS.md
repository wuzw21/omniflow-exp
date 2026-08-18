# OmniFlow-exp Rules

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
- Catalog snapshots are read-only evidence. Runtime construction must never
  seed, replace, or persist catalog Functions into a Function Store.
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

One successful `omniflow.run_log.v1` may save one or more semantic Functions in
one `save_function` call. Enhancement is optional (`enhance=true`) and uses the
same validation and Store writer as a normal save.

Enhanced authoring uses one bounded three-stage workflow over one in-memory
draft, not one JSON call per RunLog action. Stage 1 identifies Function ranges
once; stages 2 and 3 edit one identified Function at a time for source-proven
action semantics plus parameters, then checker registrations. The Agent never
returns complete Functions, source states,
bindings, checker rules, Stores, or authoring manifests. It may request only
the two action edits defined by the shared schema: replace a launcher click
with the exact after-state package as `open_app`, or attach the exact visible
source target as `target_description`.

`save_function` deterministically preserves source action order and states,
validates every requested action edit against the before/after RunLog states,
compiles parameter schemas and bindings, registers checkers, and emits one
large Function covering the complete successful trajectory plus optional
reusable contiguous semantic subsegments identified by the Agent. The Agent
must omit uncertain, transient-dialog, task-ending, and task-specific candidates
instead of forcing a split. Every emitted subsegment must supply a non-empty
`stability_reason` naming its stable precondition, repeatable semantic effect,
and caller-varying content that must be parameterized. Subsegments never
replace the complete Function, and meaningless one-click fragments are not
saved. A checker action may not also
remain a formal action in the same Function.

The single model-facing tool is `edit_function_draft`. Its three strict stage
schemas return only: `complete_function + subsegments`, `action_edits +
bindings`, then `checker_steps`. The latter two schemas are requested separately
for each Function and may refer only to that Function's listed source indices.
An action edit contains only `function_id`,
`step_index`, `operation`, and `value`. A subsegment contains Function metadata,
`stability_reason`, and an inclusive/exclusive source step range. Bridge and
experiment adapters import these schemas and select the supplied tool name
instead of defining another authoring contract. One deterministically invalid
stage edit gets one correction opportunity. Transport failures, missing
evidence, or a second invalid edit fail immediately. Nothing is persisted
until all three edits compile and all Functions pass the same authoritative
validation and sole Store writer.

Action semantics are source evidence, not free-form generation. `open_app`
requires a source launcher page and a different non-empty package in the
RunLog after-state; its value must equal that package. `set_target` requires an
exact visible label at the source action point; its value must equal that label.
Caller-varying visible targets such as an hour or category bind
`target_description`; typed content binds `text`. The compiler derives this
path from the validated action; the Agent does not author it. Coordinates,
packages, waits, and directions are never parameters. An invented or
paraphrased target fails validation.

Before a saved B-MoCA bundle may run cross-environment, its complete Function
must pass env100 `script_replay` with official success, method success,
`model_calls=0`, and `fallback_steps=0`. A failed source gate ends that task;
never prepare or launch env101--109 or `ours` for an unqualified Function. Do
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

The global target-probability threshold is defined only in the
`protocol.checker` block of `config/paper_androidworld.json`. Pair confidence
and page-embedding similarity are evidence, not triggers. Per-rule thresholds
and condition switches are forbidden because they recreate a trigger language.
Formal Function actions go directly through canonical OmniTransfer target
mapping; a missing or rejected mapping returns control to the normal Planner
fallback without source-coordinate execution.

Function success is an ordinary Planner tool result, not AndroidWorld task
completion. The Planner may call more Functions or GUI actions before it
explicitly finishes.

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
cell protocols or names. Formal methods are exactly `fixed_replay`, `ours`,
`mobilegpt_offline_retrieval`, `appagent_demo`, and `t3a_hint`.

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

Run task-major and complete one task before advancing. Resolve
`OMNIFLOW_EXP_MEMORY_ROOT/current.json` first and skip every formal result with
an existing official-validator conclusion. Formal results and original attempts
are immutable.

For each unfinished task:

1. run the static gate and verify the source-seed-111 RunLog, exact hashes,
   Function Store, transfer states, and canonical OmniTransfer checkout;
2. check Function recall, Planner selection, and offline replay;
3. qualify the Function on the source contract with official validator success,
   `model_calls=0`, and `fallback_steps=0`;
4. run at most three unregistered `ours` development iterations on SmallPhone,
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

The shared AndroidWorld setup seam may resolve the system `Open with` chooser
for official Contacts setup only when `Open with`, `Contacts`, and `Just once`
are all visible; it selects `Contacts` and `Just once`, then resumes the
official onboarding `Skip`. Other chooser states remain setup failures.

For explicitly authorized source-data collection only, a single-task direct
AndroidWorld collector may use the pinned checkout's native emulator,
`env_launcher`, `TaskRegistry`, `get_state()`, `execute_action()`, and the
official task validator. This mode is limited to immutable successful seed-111
RunLogs with screenshots, native observations, and decision records; it must
not invoke experiment methods or register formal results.

Use AndroidWorld native state/action and its official validator. B-MoCA is an
environment adapter using the same OmniFlow Function/checker/OmniTransfer
runtime and official B-MoCA reward. `ours` lets the Planner select Functions;
`script_replay` selects the one complete Function directly with its saved
source-call arguments, but may not own a second action mapper or executor.

The AndroidWorld `ours` adapter runs exactly one persistent `OmniFlow.run()`
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

`tools/manual_androidworld_harness.py` is human-only diagnosis. It cannot create
formal results, refresh canonical memory, or replace the unified shell entry.

RunLog `open_app` actions keep the canonical package as their stored contract.
At execution, the shared AndroidWorld host resolves a registered launcher name
when one exists and otherwise preserves the package for AndroidWorld's official
package-launch fallback. AndroidWorld's own ADB helper closes stale tasks. Do
not add a local app mapping, a pre-launch registry gate, or another launcher.
