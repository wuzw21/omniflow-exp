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
  -> optional Agent split -> parameters -> checker review
  -> validation
  -> save
  -> Function Store
```

One RunLog may save multiple semantic Functions in one call. `enhance=true` does
not open another path: three internal Agent stages each return a complete
`{functions, arguments}` bundle for semantic splitting, parameter binding, and
checker review. The core validates every stage, grounds the final states and
actions in the same successful RunLog, and uses the same Store writer.

Every stage retains at least one large Function covering the complete successful
trajectory. The split stage also returns every reusable contiguous semantic
subsegment, without creating one-click fragments. All model transports use the
same Function-bundle tool generated from the checked-in contracts. Its schema
is narrowed for each stage: split cannot add parameters or checkers, parameter
binding cannot change Function identity or action order, and checker review can
select only exact actions already registered on that Function. Checker review
may move a safely optional source-state-dependent setup, interruption, recovery,
or alternate-path navigation action from the formal path into that Function's
checker list. It cannot move required navigation or a terminal action with no
later formal check point, rewrite Function meaning, parameters, arguments, or
unselected actions, duplicate formal actions, or replace the complete Function
with one-click fragments.

A deterministically rejected stage output receives one explicit correction
opportunity with the validator error. Model transport failures fail immediately;
no partial Function is saved, and corrected output passes through the same
validation and Store writer.

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

Before every pending formal Function action, OmniFlow checks every unexecuted
rule registered on that Function. It executes the checker once only when all of
the following hold:

1. the latest canonical OmniTransfer page embedding matches the current page
   to the rule's RunLog source state;
2. OmniTransfer finds a valid target for the source action on the current
   observation;
3. the selected target's OmniTransfer rank probability reaches the configured
   high-confidence threshold; and
4. the rule has not already executed in this Function call.

Each rule contains exactly `source_state_id` and `action`; registration on the
Function is the rule-to-Function relationship. Otherwise the rule is skipped
and may be checked again before a later formal action. The page and target
thresholds are configured once in `config/paper_androidworld.json`. Pair
confidence cannot compensate for a page mismatch or ambiguous target ranking.
There are no per-rule thresholds, step-number triggers, trigger DSLs, global
checker pool, or source-coordinate passthrough. Checker actions are limited to
transferable `click`, `input_text`, and `long_press` actions.

Formal Function actions are page-bound too: before any state-dependent action,
the same canonical page embedding threshold must match its RunLog source state.
On mismatch, the Function fails back to the Planner without attempting target
ranking. `open_app` and `wait` are the only state-independent actions.

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
pinned AndroidWorld registry and launcher, after closing any stale app task;
there is no adapter-owned app registry or alternate launcher.
Official Contacts setup may resolve Android's `Open with` chooser by selecting
`Contacts` and `Just once` before resuming the official onboarding `Skip`.

No formal experiment is launched during code migration.
