# OmniFlow-exp Rules

This repository exists only for the paper's AndroidWorld experiment and B-MoCA
cross-environment validation. B-MoCA additionally registers `script-replay` as
its deterministic MobileGPT-style semantic-selector comparison baseline. Do not add product code,
historical campaigns, other exploratory methods, raw data, or compatibility
layers.

Before changing this repository, read `README.md` and
`scripts/exp/README.md`. They are the user-facing description of the same
paper-only workflow and must stay consistent with this file.

## OmniFlow / OmniTransfer boundary

- OmniTransfer is responsible only for producing complete ranked target
  candidates and their evidence. It owns no harness, accept/reject decision,
  page gate, selected action, or fallback policy.
- OmniFlow owns OmniTransfer preflight, page checks, candidate selection,
  execution, failure classification, and fallback.
- OmniFlow-exp owns the experiment's core logic, but page identity is owned by
  the latest canonical OmniTransfer page embedding. The only page encoder in
  the active code path is `omniflow/transfer/page_embedding.py`; do not add a
  local legacy page encoder, learned pooling head, or comparison baseline.
- The canonical implementation is loaded from the real
  `~/Projects/Omni/OmniTransfer` checkout. Its frozen checkpoint provenance is
  recorded in every page embedding; a missing or mismatched canonical
  checkout/checkpoint is an explicit failure.
- The current page-embedding checkpoint is
  `src/omnitransfer/checkpoints/omnitransfer_spatial_xml_alignment_v9_20260805/v9_spatial_xml_alignment_seed29.pt`.
  This page embedding is separate from OmniTransfer's candidate-ranking
  release; do not silently substitute the candidate matcher or a local pooler.

## Current phase

The current phase is code migration and script maintenance only. Do not launch
formal experiments or revise method implementations unless the user explicitly
opens a new execution phase.

## Mandatory per-task testing memory

Before every AndroidWorld check, conversion, development run, formal cell, or
failure diagnosis, re-read this file and `scripts/exp/README.md`. This section
is the persistent testing protocol; do not replace it with task-local notes.

Run one task to a terminal conclusion before advancing to the next task. The
only task order is the persistent external difficulty-ordered experiment table.
For each task, resolve `current.json` first and skip every formal cell that
already has an official-validator conclusion.

Use this order for each unfinished task:

1. Run the unified static gate and validate the canonical seed-111 RunLog,
   exact SHA-256 lineage, Function Store provenance, and transfer states.
2. Check Function recall, one-step Planner tool choice, and Function replay
   offline before starting an emulator.
3. Qualify the Function directly on the seed-111 source contract. Require full
   replay, official validator success, `model_calls=0`, and `fallback_steps=0`.
4. Run bounded unregistered `ours` development episodes on SmallPhone and then
   Pixel Fold. A task gets at most three development iterations; preserve the
   failure and advance after the third unsuccessful iteration.
5. Freeze the development version, then fill only missing formal cells for the
   task. Formal results remain immutable and are never replaced by a retry.

Harness and script repairs are allowed only for reproducible task-independent
defects in environment resolution, dependency installation, official emulator
lifecycle, AndroidWorld setup, a11y/gRPC readiness, native `get_state()`, the
single `observe -> plan -> act` step boundary, schema-only adapter conversion,
coordinate-contract normalization, immutable accounting, result registration,
and completed-cell skipping. Add a deterministic regression test before the
repair whenever practical.

Every reproducible configuration, environment, setup, or runtime failure must
be converted into a permanent repair before advancing: put configuration
resolution in the canonical core configuration or `scripts/exp/run_androidworld.sh`,
put runtime recovery at the narrow shared harness seam, add a regression test,
and document the stable behavior here or in `scripts/exp/README.md`. Shell
history, manually exported variables, one-off remote commands, task-local
workarounds, and undocumented operator knowledge are not fixes. Preserve the
original immutable failed attempt and classify its exact failed stage.

During bounded `ours` development, the only method-level revisions allowed are
short task-independent Planner rules and offline Agent-authored Function
semantic improvements. Function improvements may clarify descriptions,
parameters, and reusable stable action segments or split a RunLog into multiple
semantic Functions. They may not contain target-device or task-instance
special cases. A Function success returns control to the Planner and never
terminates the AndroidWorld task by itself.

Never add task-specific prompts, coordinates, state gates, page thresholds,
evaluator-aware completion, source-coordinate fallback, screenshot cropping,
OOB observation, a second AndroidWorld runner, or hidden retries. Do not change
the formal model, parser, retry policy, step budget, action policy, or output
contract. A failure after `task_started` and an official validator conclusion
is a method result, not an environment failure.

`appagent_demo` and `mobilegpt_offline_retrieval` are native external baselines.
Keep their upstream online prompts, planners, parsers, policies, retries, step
budgets, models, and execution flow unchanged. Give each baseline only source
experience derived from an official-validator-successful seed-111 RunLog and
converted into that baseline's standard native memory or demonstration schema.
The AndroidWorld adapter is required, but its scope is limited to LiveTask
lifecycle, native observation/action bridging, setup, accounting, RunLog capture,
and official validation. Schema conversion and this adapter may be maintained to
preserve the native contract; they must not repair, reinterpret, or replace the
baseline method. If a successful source experience cannot be represented by the
native schema, record the exact conversion failure or unavailable status rather
than fabricating experience or changing the baseline.

## Mandatory execution entry

- Every AndroidWorld or B-MoCA check, development episode, formal cell, conversion, and
  memory refresh must enter through `scripts/exp/run_androidworld.sh`. Do not
  invoke `src.integrations.android_world.launch`, `src.experiment.androidworld`,
  or a skill-owned runner directly; they are implementation seams.
- Use `--development-run` for an unregistered bounded `ours` episode. A first
  run uses AndroidWorld setup. A repeated development run may set
  `OMNIFLOW_SINGLE_TASK_PERFORM_EMULATOR_SETUP=0` only when the same live
  emulator already has the required AndroidWorld app snapshots.
- Formal experiment execution keeps the default cold restart and setup. Never
  use the development reuse override to create or register a formal result.
- Treat `mobilegpt_offline_retrieval` and `appagent_demo` as frozen external
  baselines under the absolute no-repair rule above.
- B-MoCA is an environment adapter. Its default E2E path uses the same
  `OmniFlow.run()` Function/checker/OmniTransfer runtime, GLM-5.1 only for
  Planner tool selection, zero Function fallback steps, and the official
  B-MoCA reward as the success authority. The optional direct-Function diagnostic
  remains the `omniflow` method, calls the sole visible zero-argument Function,
  and permits at most three existing fallback steps after Function failure. Its
  registered `script-replay`
  comparison skips Checker steps and permits only unique text/content-description,
  semantic-child depth/rank, or semantic-parent/child-rank selection. Resource
  IDs are prohibited. It has no Planner, model, coordinate fallback, DP, or
  OmniTransfer call. Two consecutive semantic page observations must be stable
  and free of visible progress indicators before any pointer action.
  With `--source-runlog`, the unified entry first performs a schema-only,
  one-RunLog-to-one-Function conversion using the sibling transfer-state
  catalog; it does not split, rewrite, or semantically enhance the trajectory.

## Local and 9207 synchronization

- The only active OmniFlow-exp checkout on both the local machine and host
  `9207` is `~/Projects/Omni/OmniFlow-exp` on branch `main`.
- Before every 9207 check, conversion, development episode, or formal cell,
  compare `git rev-parse HEAD` in both active checkouts. The two full commit
  SHAs must be identical and both worktrees must contain no uncommitted tracked
  changes. A mismatch is a preflight failure; do not run an experiment.
- Synchronize source only through Git: commit and push local `main`, then
  fast-forward the 9207 `main` checkout to the same commit. Do not copy a source
  tree, run from an unversioned snapshot, or use a release directory as the
  active development checkout.
- Immutable commit-SHA releases may archive a formal checkpoint, but they must
  be created from the already synchronized `main` commit and never replace the
  two active checkouts.

## Formal protocol

- Run task-major: one task across every method and both devices before the next task.
- The exact method set is `fixed_replay`, `ours`,
  `mobilegpt_offline_retrieval`, `appagent_demo`, and `t3a_hint`.
- Source seed is `111`; target seed is `113`.
- Targets are SmallPhone and Pixel Fold in unfolded state `2`.
- Use AndroidWorld native observation/action and its official validator. Do not use OOB.
- Record validator result, model calls, prompt/completion/total tokens, actions,
  episode duration, and outer wall time for every cell.
- Results and attempts are immutable and live outside this repository.

## Method boundary

The migrated method implementations are frozen during this phase. Script,
environment, installation, preflight, accounting, and registration repairs may
not change a method's prompt, memory, demonstration, action policy, parser,
retry policy, step budget, model, or output.

OmniFlow is a general VLM-task runtime, not an AndroidWorld task terminator.
A successful Function execution returns a normal tool result to the task
planner; it does not imply that the task is done. The planner may call the same
or another recalled Function multiple times before it explicitly finishes the
task. Function resume is permitted only after a real Function execution
failure, never merely because a successful Function returned control to the
planner. Evaluation adapters must not convert Function success into task
completion or add evaluator-specific completion logic to the OmniFlow core.

## Code layout

- Put contracts and shared data types in `omniflow/core/`.
- Put Function lifecycle code in `omniflow/functions/`.
- Put replay orchestration and execution in `omniflow/runtime/`.
- Put transfer orchestration and evidence tooling in `omniflow/transfer/`.
- Put VLM-specific planning and adaptation in `omniflow/vlm/`.
- Keep `omniflow/bridge.py` as the external bridge entry point.
- Keep `omniflow/vlm_coordinates.py` at its shared-contract path.
- Do not recreate the retired flat module layout.
- Keep `scripts/exp/run_androidworld.sh` as the only repository script.
- Put experiment implementation in `src/experiment/` or `src/integrations/`.

## Data boundary

Never commit RunLogs, screenshots, XML dumps, model weights, APKs, emulator
images, baseline memory, credentials, runtime attempts, or result tables. All
such assets must be supplied through explicit absolute paths.

The only RunLog contract is `omniflow.run_log.v1`. Its observation fields are
the AndroidWorld `State` fields `pixels`, `forest`, `ui_elements`, and
`auxiliaries`; its actions use only fields and action types accepted by
AndroidWorld `JSONAction`. OmniFlow persists `pixels` as an immutable screenshot
reference instead of embedding the array. Never encode an OmniFlow action alias
or a private device command as an AndroidWorld action.
This contract is defined by OmniFlow source and the checked-in schema, never by
`current.json`; experiment memory only indexes immutable assets and results.

An empty or unusable OmniTransfer candidate ranking must return to the normal
OmniFlow fallback path. Low confidence alone is not a transfer failure when a
ranked target candidate with valid bounds exists. Never replay source-device
coordinates directly on a target device.

## Long-term experiment memory

- `OMNIFLOW_EXP_MEMORY_ROOT/current.json` is the only canonical entry point for
  existing AndroidWorld RunLogs, converted Function assets, and registered
  results. The experiment script must resolve source and Store indexes from it
  before preflight or execution.
- Maintain the memory only through
  `scripts/exp/run_androidworld.sh --refresh-memory`. Function conversion must
  update the same memory before reporting success, and result registration must
  update it immediately after writing an immutable formal result.
- Store immutable evidence by exact SHA-256. Identical content has one object
  and any additional locations are aliases; never copy it into another
  logical version or regenerate it.
- Keep every original attempt immutable. Deduplication changes only canonical
  indexes and content-addressed storage; it never deletes or rewrites evidence.
- Classify memory by AndroidWorld task. Preserve unclassified evidence in the
  registry instead of dropping it or guessing a task.
- The master source index is authoritative for the canonical source RunLog.
  A converted Function Store is canonical only when the v2 Store, transfer
  states, provenance hashes, and no-target-input audit all verify. Equal-quality
  conflicting Stores are an error and must not be selected silently.
- For a task/method/device result cell, use the earliest immutable registered
  result with an official-validator conclusion. Never rank or replace results
  by success, token count, duration, or a later retry.
- If `current.json` already contains a canonical asset or formal result, reuse
  it or skip the completed cell. Do not call a model, rebuild an asset, rerun a
  completed cell, or reason about which historical directory to use.
- A missing, corrupt, or ambiguous memory entry is a preflight failure. Report
  it explicitly; do not fall back to path guessing or automatic generation.
