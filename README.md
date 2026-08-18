# OmniFlow-exp

Paper-only AndroidWorld evaluation and B-MoCA validation for OmniFlow. This
repository contains code and orchestration only; RunLogs, screenshots, models,
APKs, emulator images, memories, and results stay outside the repository.

## Experiment contract

The five formal methods are:

- `fixed_replay`
- `ours`
- `mobilegpt_offline_retrieval`
- `appagent_demo`
- `t3a_hint`

One result is exactly `task + method + device`. The shell entry point dispatches
tasks, the E2E pipeline schedules methods/devices, and the Python runner executes
one result. Formal methods, devices, seeds, budgets, timeouts, endpoint, fold
state, AVD topology, and revisions have one source:
`config/paper_androidworld.json`.
For GLM-5.1, `model.env` contains only `LLMTHU_API_KEY`; the endpoint URL comes
from that canonical protocol and is exported internally.
The isolated source AVD uses the API-33 `small_phone` profile matching the
retained 720x1280 source trajectories; target AVDs remain separate instances.
The launcher provisions a missing configured source AVD before E2E dispatch and
fails immediately if its emulator process exits during boot.

Run a task-major slice:

```bash
OMNIFLOW_EXP_ASSET_ROOT=/absolute/assets \
OMNIFLOW_EXP_RESULTS_ROOT=/absolute/results \
OMNIFLOW_EXP_MEMORY_ROOT=/absolute/memory \
OMNIFLOW_ENV_FILE=/absolute/model.env \
PYTHON_BIN=/absolute/python \
OMNITRANSFER_ROOT="$HOME/Projects/Omni/OmniTransfer" \
bash scripts/exp/run_androidworld.sh --tasks AudioRecorderRecordAudio
```

Run all indexed tasks with `--all-tasks`; validate existing assets without
starting emulators with `--check-only --all-tasks`. B-MoCA uses the same public
launcher with `--environment bmoca --tasks TASK` for one method, or
`--environment bmoca --all-tasks` for the two-method corpus campaign. See
[`scripts/exp/README.md`](scripts/exp/README.md) for the command contract.

## One Function lifecycle

There is one write operation: `save_function`.

```text
successful RunLog
  -> optional three-stage Agent draft edit
  -> deterministic compilation
  -> validation
  -> save
  -> Function Store
```

One RunLog may save multiple semantic Functions in one call. `enhance=true` does
not open another path: the Agent edits one in-memory draft through three small
stages—semantic Function ranges once, then source-proven action semantics plus
parameter declarations and checker registrations separately for each Function.
The middle stage may request an action edit: a launcher click to the exact
after-state package becomes `open_app`, or a visible source target becomes
`target_description`. It never writes complete actions, states, bindings,
checker rules, or a Store.

The core validates those small edits against the RunLog, preserves action
order, compiles bindings and checkers, and emits the complete Function plus
reusable contiguous subsegments. Invented packages, paraphrased targets, and
ungrounded action changes are rejected.
Subsegments are optional and are emitted only when the Agent identifies the
source-state/action sequence as independently and stably replayable from its
own first state. Every emitted subsegment must include `stability_reason`, naming
its stable precondition, repeatable semantic effect, and any varying content
that must be parameterized. Uncertain, transient-dialog, task-ending, and
task-specific fragments are omitted; the complete RunLog Function remains the
fallback and is never replaced by forced segmentation.
Subsegment descriptions may claim only effects caused inside their source
range. A bound source value must appear directly in the RunLog goal; an
unrequested current page value is a precondition, not an input parameter.
Every output is grounded in the same successful RunLog and goes through the
same validator and Store writer. A rejected stage edit receives one bounded
correction; transport failures fail immediately and no partial Function is
saved.
Agent-authored parameters bind only `text` or `target_description`; the tool
schema cannot return coordinate, package, wait, direction, or arbitrary nested
bindings.

The retained bridge tools are:

- `save_function`
- `list_functions`, `get_function`, `delete_function`, `clear_functions`
- `list_run_logs`, `get_run_log`, `get_run_log_state`
- `run_gui`

The shell never auto-builds a missing Function Store. Save the Functions first,
refresh external memory, then run the experiment. Catalog snapshots are
read-only source evidence and never seed or rewrite the Store at runtime.

## Checker model

Checker rules are local registrations on one Function, not a global rule pool.
A Function with no checker rules receives none from another Function.
The only checker configuration is which rules that Function registers and the
one global target-probability threshold in `protocol.checker`. Check frequency
is fixed rather than configurable: every unexecuted registered rule is
evaluated before every pending formal action. This makes checking frequent
without making execution permissive.

Before every pending formal Function action, OmniFlow checks every unexecuted
rule registered on that Function. It executes the checker once only when all of
the following hold:

1. OmniTransfer finds a valid target for the source action on the current
   observation;
2. the selected target's OmniTransfer rank probability reaches the configured
   high-confidence threshold; and
3. the rule has not already executed in this Function invocation, including a
   resumed invocation after a later formal-action failure.

Each rule contains exactly `source_state_id` and `action`; registration on the
Function is the rule-to-Function relationship. Otherwise the rule is skipped
and may be checked again before a later formal action. The target threshold is
configured once in `config/paper_androidworld.json`. Pair confidence and
page-embedding similarity cannot compensate for an ambiguous target ranking.
There are no per-rule thresholds, step-number triggers, trigger DSLs, global
checker pool, or source-coordinate passthrough. Checker actions are limited to
transferable `click`, `input_text`, and `long_press` actions.

Formal Function actions use canonical OmniTransfer target mapping directly.
Missing or rejected mappings fail back to the Planner without source-coordinate
execution; no page-similarity gate runs before a Function step.

## OmniTransfer

All page identity and action transfer use the canonical
`~/Projects/Omni/OmniTransfer` checkout. The only active page adapter is
`omniflow/transfer/page_embedding.py`, using:

```text
src/omnitransfer/checkpoints/
  omnitransfer_spatial_xml_alignment_v9_20260805/
  v9_spatial_xml_alignment_seed29.pt
```

There is no native 512D, page-word, 1024D, local pooling, resource-id lookup, or
coordinate replay branch. Missing evidence is explicit and returns control to
normal VLM fallback.

## Results and memory

`OMNIFLOW_EXP_MEMORY_ROOT/current.json` is the canonical index for source
RunLogs, Function Stores, method-native memory, and registered results. Existing
official-validator conclusions are immutable and skipped.

Public result rows contain only:

```text
task, method, device, source_seed, evaluation_seed, status,
validator_success, model_calls, prompt_tokens, completion_tokens, total_tokens,
actions_executed, episode_duration_sec, outer_wall_sec, error, evidence_paths
```

Preparation, reuse, and component diagnostics are recorded once in the
attempt's `details` evidence instead of being repeated in every result row.
Registration preserves that same two-level shape and appends one immutable
registry ledger; it does not generate a second master matrix or run-record table.

## Repository layout

- `omniflow/core/`: contracts, models, and canonical configuration views
- `omniflow/functions/`: Function validation, enhancement, recall, and Store
- `omniflow/runtime/`: Planner loop and Function/checker execution
- `omniflow/transfer/`: OmniTransfer integration and page embedding
- `omniflow/vlm/`: existing frozen Planner prompt and model adaptation
- `omniflow/bridge.py`: external JSON-line API
- `src/experiment/`: task scheduling, accounting, and immutable registration
- `src/integrations/`: AndroidWorld and baseline adapters
- `scripts/exp/run_androidworld.sh`: only public experiment launcher
- `schemas/oob/`: shared RunLog, Function, checker, and bridge schemas

The AndroidWorld `ours` adapter invokes one complete, persistent OmniFlow cycle
per task. AndroidWorld's outer episode step does not split that cycle or own a
second planner budget, Function session, resume state, or fallback counter.
RunLog `open_app` stores only the package. At execution, the adapter uses the
pinned AndroidWorld registry when it has a launcher name and otherwise lets the
same official launcher use its package fallback, after closing any stale app
task. There is no adapter-owned app registry, pre-launch gate, or alternate
launcher.
Official Contacts setup may resolve Android's `Open with` chooser by selecting
`Contacts` and `Just once` before resuming the official onboarding `Skip`.

No formal experiment is launched during code migration.
