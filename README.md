# OmniFlow-exp

Paper-only AndroidWorld evaluation and B-MoCA validation for OmniFlow. This
repository contains code and orchestration only; RunLogs, screenshots, models,
APKs, emulator images, memories, and results stay outside the repository.

## Experiment contract

The five formal methods are:

- `fixed_replay`
- `omniflow`
- `mobilegpt`
- `appagent`
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
OMNIFLOW_EXP_ASSET_ROOT=/Users/wuzewen/Projects/Omni/OmniFlow-exp/data \
OMNIFLOW_EXP_RESULTS_ROOT=/Users/wuzewen/Projects/Omni/OmniFlow-exp/data \
OMNIFLOW_EXP_MEMORY_ROOT=/Users/wuzewen/Projects/Omni/OmniFlow-exp/data \
OMNIFLOW_ENV_FILE=/absolute/model.env \
OMNITRANSFER_ROOT="$HOME/Projects/Omni/OmniTransfer" \
bash scripts/exp/run_androidworld.sh --tasks AudioRecorderRecordAudio
```

Run all indexed tasks with `--all-tasks`; validate existing assets without
starting emulators with `--check-only --all-tasks`. B-MoCA uses the same public
launcher with `--environment bmoca --tasks TASK` for one method, or
`--environment bmoca --all-tasks` for the three-method reuse campaign. See
[`scripts/exp/README.md`](scripts/exp/README.md) for the command contract.

## Call path

The first code path to read is always:

```text
scripts/exp/run_androidworld.sh
  -> src/experiment/run_tasks.py       # task/method/device scheduling
  -> src/experiment/run_task.py            # one AndroidWorld result
  -> src/integrations/android_world/run_episode.py  # one native episode
```

The B-MoCA campaign uses the same shell and scheduler; each environment result
re-enters the shell's single-result B-MoCA branch and ends at the same native
launcher. Development and source collection are bounded modes of that
launcher, not alternate executors. Function creation has one route:
`save_function` validates and writes the Store, then `--refresh-memory` updates
`data/current.json`; runtime only reads that index and the registered Store.

There are two MobileGPT preparation adapters because their contracts differ:
AndroidWorld creates a sealed source bundle through
`src/experiment/mobilegpt_source.py`, while B-MoCA creates its native replay
memory through the scheduler's B-MoCA adapter. Both use the single converter in
`src/integrations/mobilegpt.py`; neither is a second Function or
episode runner. Do not add another preparation entry point.

Install the B-MoCA replay dependency with `uv sync --extra bmoca`. The protocol
pins DroidRun v0.5.6. `skilldroid_replay` converts the qualified env100 RunLog to
DroidRun's official `macro.json` and executes it with the native `MacroPlayer`
through a B-MoCA `DeviceDriver` adapter, preserving official reward and RunLog
recording. This baseline is absolute-coordinate macro replay; it does not add a
locator, state verification, or model fallback.

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

One RunLog saves exactly one semantic Function. `enhance=true` does not open
another path: the Agent edits one in-memory draft through three small stages—
name the complete Function, then source-proven actions plus parameter
declarations, then checker registrations. The middle
stage may request `open_app`/`set_target` semantics or return direct actions
indexed to the RunLog for exploration. The compiler copies or validates every
package, target, coordinate, state, binding, and checker against source
evidence; nothing is persisted outside the canonical writer.

The core validates those small edits against the RunLog, preserves action
order, compiles bindings and checkers, and emits the one complete Function. The action stage may also return direct
source-indexed actions for exploration; each must match the shown RunLog action
or a source-proven semantic edit, otherwise it is rejected. Packages, targets,
coordinates, and other evidence are copied or validated by the compiler rather
than trusted from free-form Agent output.
The complete RunLog Function is always the only recall candidate. The enhanced
Store's `source_calls` contains exactly one call to that Function.
An optional parameter must be grounded in the source RunLog goal; an
unrequested current page value is a precondition, not an input parameter.
Every output is grounded in the same successful RunLog and goes through the
same validator and Store writer. Each small stage gets at most three model
attempts; after a rejection the Agent sees only that stage's deterministic
validation error and revises the same in-memory draft. Transport failures fail
immediately and no partial Function is saved.
Agent-authored parameters bind only `text` or `target_description`; the tool
schema cannot return coordinate, package, wait, direction, or arbitrary nested
bindings.

Function quality fixes belong in this shared authoring policy, its evidence,
the deterministic compiler, or the runtime adapter. Do not patch a generated
Function or Store for one task. Re-run `save_function` from the same successful
RunLog after a policy fix; its bounded stages may use several small model calls.

The retained bridge tools are:

- `save_function`
- `list_functions`, `get_function`, `delete_function`, `clear_functions`
- `list_run_logs`, `get_run_log`, `get_run_log_state`
- `run_gui`

The shell never auto-builds a missing Function Store. Save the Functions first,
refresh the local data index, then run the experiment. Transfer-state evidence
is read-only and never seeds or rewrites the Store at runtime.

### Public bridge API

The bridge is a JSON-RPC-lines process. Start it with one Store:

```bash
.venv/bin/python -m omniflow.bridge \
  --store /absolute/path/function_store.json
```

The wire schema is [`schemas/oob/omniflow_android_bridge.v2.json`](schemas/oob/omniflow_android_bridge.v2.json).
The public `tools/call` names and their important inputs are:

| Tool | Inputs | Purpose |
| --- | --- | --- |
| `run_gui` | `goal`, `model`; optional `max_steps`, `defer_user_input` | Run the shared Planner/Function runtime through the host callbacks |
| `list_functions` | optional `limit`, `offset`, `include_hidden` | List registered Functions |
| `get_function` | `function_id` | Read one Function |
| `save_function` | `run_id` or `run_log`; `functions` or `enhance=true`; optional `arguments`, `instruction` | Compile, validate, and atomically save one RunLog-grounded Function |
| `delete_function` | `function_id` | Delete one registered Function |
| `clear_functions` | `confirm=true` | Delete all registered Functions explicitly |
| `list_run_logs` | optional `limit`, `offset`, `source`, `status`, `model`, `query` | Search host-owned RunLogs |
| `get_run_log` | `run_id` | Read one RunLog |
| `get_run_log_state` | `state_id` | Read one immutable XML/screenshot state |

The host integration must implement only the callbacks declared by the same
schema: `observe`, `act`, `model_turn`, `installed_apps`, `record_step`,
`request_input`, `list_run_logs`, `get_run_log`, and `get_state`. These are host
callbacks, not a second experiment runner. Function actions always use the
shared OmniTransfer mapping and fall back to the normal Planner on transfer
failure; source-device coordinates are never replayed directly.

### Configuration

There are two configuration layers:

1. Edit [`config/paper_androidworld.json`](config/paper_androidworld.json) for
   protocol values: formal methods, devices, seeds, step/time budgets,
   checker threshold, model endpoint profile, AVD profiles, and pinned
   revisions. Do not duplicate these values in shell scripts or Python.
2. Set paths and secrets in the environment. The minimum setup is:

```bash
export OMNIFLOW_EXP_ASSET_ROOT=/absolute/OmniFlow-exp/data
export OMNIFLOW_EXP_RESULTS_ROOT=/absolute/OmniFlow-exp/data
export OMNIFLOW_EXP_MEMORY_ROOT=/absolute/OmniFlow-exp/data
export OMNIFLOW_ENV_FILE=/absolute/model.env
export OMNITRANSFER_ROOT=$HOME/Projects/Omni/OmniTransfer
```

`model.env` contains `LLMTHU_API_KEY`; credentials are not committed. Use the
repository `.venv/bin/python` for formal runs. External AndroidWorld,
MobileGPT, AppAgent, B-MoCA, Android SDK, Java, and ADB locations are supplied
through the optional `OMNIFLOW_*_ROOT`, `OMNIFLOW_ANDROID_SDK_ROOT`,
`OMNIFLOW_JAVA_HOME`, and `OMNIFLOW_ADB_PATH` variables documented by
`bash scripts/exp/run_androidworld.sh --help`. The launcher is the only place
that turns these variables into scheduler arguments.

### Test and validation capabilities

Use the smallest check that answers the question:

```bash
# Full offline regression; no emulator or model is required by the tests.
.venv/bin/python -m pytest -q

# Provider-specific contract and shell integration checks.
bash scripts/exp/test_provider.sh mobilegpt
bash scripts/exp/test_provider.sh appagent
bash scripts/exp/test_provider.sh all

# Static experiment gate; validates the selected run without starting an emulator.
bash scripts/exp/run_androidworld.sh --check-only --all-tasks

# Build one command without executing it.
bash scripts/exp/run_androidworld.sh --dry-run --tasks TASK

# Validate and rebuild the single local data index.
bash scripts/exp/run_androidworld.sh --refresh-memory
```

Experiment execution capabilities are all routed through the same launcher:

- AndroidWorld formal methods: `fixed_replay`, `omniflow`, `mobilegpt`,
  `appagent`, and `t3a_hint`.
- Source qualification: `--collect-source` or
  `--source-qualification-only`.
- Bounded development: `--development-run`.
- B-MoCA: `ours_replay`, `mobilegpt_replay`, and `skilldroid_replay`.
- Performance side channel: `--collect-performance`, which writes a sidecar
  without changing the public result row.

For old Function JSON, run a dry migration before writing anything; the
converter classifies Stores, bundles, and catalogs and reports missing evidence
as `blocked`:

```bash
.venv/bin/python -m omniflow.functions.migrate_store \
  --input-root /absolute/old-data \
  --output /absolute/new-data \
  --dry-run --report /absolute/migration-report.json
```

See [`omniflow/functions/README.md`](omniflow/functions/README.md) for the
single-file migration form and the rules for rebuilding `data/current.json`.

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
configured once in `config/paper_androidworld.json`. Page-embedding similarity
is not used for Function recall or checker triggering; an ambiguous target
ranking is skipped.
There are no per-rule thresholds, step-number triggers, trigger DSLs, global
checker pool, or source-coordinate passthrough. Checker actions are limited to
transferable `click`, `input_text`, and `long_press` actions.
Any source target named by the task goal or Function semantics is task progress
and is rejected as a checker.
The same RunLog action cannot be a checker in one emitted Function and a formal
step in another.

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

`~/Projects/Omni/OmniFlow-exp/data/current.json`
is the only canonical index for source RunLogs, Function Stores, method-native
memory, and registered results. Every bundle uses the
`<environment>/<task>/<device>/<category>/<method>/<attempt_id>` classification
defined in `AGENTS.md`. Set `OMNIFLOW_EXP_MEMORY_ROOT` explicitly only for a
read-only migration input. Existing official-validator conclusions are
immutable and skipped. There are no parallel registry, snapshot, or index
files.

The former external memory locations are one-time migration inputs only; the
launcher does not select them after the local `data/current.json` is built.

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
Timing and energy measurement is an explicit side channel enabled with
`--collect-performance`; it writes `performance_sidecar.json` without changing
task results or public rows. Host/native AndroidWorld I/O expose mean/P50/P95
latency, while ADB battery estimates are diagnostic rather than hardware power
measurements.

## Current data state

The migrated local index contains 116 source tasks and four validated Function
Stores. Runtime reads only `data/current.json`; old indexes, snapshots, and
task-only scratch roots are outside the active repository path. Emulator E2E
acceptance remains a separate runtime validation and is not inferred from
authoring or index migration.

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

The AndroidWorld `omniflow` adapter invokes one complete, persistent OmniFlow cycle
per task. AndroidWorld's outer episode step does not split that cycle or own a
second planner budget, Function session, resume state, or fallback counter.
RunLog `open_app` stores only the package. At execution, the adapter uses the
pinned AndroidWorld registry when it has a launcher name and otherwise lets the
same official launcher use its package fallback, after closing any stale app
task. There is no adapter-owned app registry, pre-launch gate, or alternate
launcher.
Official Contacts setup may resolve Android's chooser when `Just once` is
visible and either `Open with` plus `Contacts`, or the already-selected title
`Open with Contacts`, is visible. It selects only the missing choice, confirms
`Just once`, then resumes the official onboarding `Skip`.

No formal experiment is launched during code migration.
