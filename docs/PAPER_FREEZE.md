# Paper Evaluation Freeze

This repository is an experiment implementation for one paper claim:

> A Function compiled from one successful source interaction can provide
> measurable experience reuse on a target environment that already satisfies
> the external observation and action contract.

The repository is not an AndroidWorld, OOB, OmniTransfer, or baseline product.
Those systems are external evaluation conditions.

## Canonical path

All AndroidWorld experiments use:

```text
scripts/exp/run_androidworld.sh
  -> src/experiment/run_tasks.py
  -> src/experiment/run_task.py
  -> src/integrations/android_world/run_episode.py
```

The only public operations are `convert-memory` and `run`.  B-MoCA is a
separate benchmark and is not an AndroidWorld entry path.

## Frozen contracts

The following are fixed for the paper and must not change in response to an
individual task failure:

- AndroidWorld task definitions and official validator.
- OOB observation and action transport.
- `schemas/oob/oob_canonical_actions.v1.json`.
- `schemas/oob/omniflow_function.v2.json`.
- `schemas/oob/omniflow_run_log.v1.json`.
- OmniTransfer V10 embedding, matcher, and fail-closed behavior.
- Planner one-action-per-turn contract.
- Formal model, endpoint, seeds, devices, and episode budgets.
- Source RunLog, Function Store, and transfer-state directory layout.
- The definition of success: official validator success.

### Planner budget pin (2026-11-06)

The AndroidWorld runtime uses one shared 600-second wall deadline. Planner
requests are individually bounded to 30 seconds and consume that same budget.
The validator flush grace is an internal component cap, not an additional wall
clock allowance.

| | before | after |
| --- | --- | --- |
| guaranteed planner budget | `30 steps x 60s step_timeout = 1800s` | shared 600s task budget |
| validator flush grace | 300s | 300s |
| derived episode ceiling | 2100s | 600s hard cap |
| `task_deadline_sec` | 600s | 600s |
| planner request clamp | hard `min(timeout, 30)` in `planner.py` | no extra clamp; `PLANNER_TIMEOUT_SEC` only |
| kernel invocation deadline | `ExecutionControl()` default 600s | `RuntimeSettings.execution_timeout_seconds` |
| `planner_error_retries` default | 0 | 2 (protocol constant for every row) |

Two independent defects made ordinary endpoint latency end an episode as if it
were a policy failure:

1. `omniflow/vlm/planner.py` clamped every request to 30s regardless of the
   configured budget, so the 60s CLI default never took effect.
2. `omniflow/runtime/engine.py` constructed `ExecutionControl()` with its own
   600s default, so the kernel sealed every episode at 600s no matter what the
   protocol deadline said. `omniflow/runtime/control.py` additionally rejected
   any deadline above 600s.

`EPISODE_TIMEOUT_SEC` is pinned to `TASK_DEADLINE_SEC` at import time, and the
protocol rejects values above 600 seconds. Retries re-observe
and re-ask the same turn, dispatch no action, and add no goal information, so a
retried episode is not a more informed agent. They spend the same episode deadline
through `bounded_timeout`, so a persistent endpoint hang ends as
`deadline_exceeded` rather than overrunning the wall clock.

**Effect on frozen results.** None of the 137 completed OmniFlow episodes touched
the 600s ceiling: observed episode durations ran 10.1s to 232.4s (median 53.3s).
Keeping the 600s ceiling therefore cannot change any recorded outcome. The retry budget
can only convert an `vlm_planner_failed:Request timed out.` epilogue into a re-ask;
it cannot rescue a wrong decision. Episodes produced before this date are not
comparable to episodes produced after it for *failure-rate* reporting, because the
retry budget differs.

### Action Schema pin

`schemas/oob/oob_canonical_actions.v1.json` is the fixed local mirror of the
external Android/OOB action wire contract. The runtime accepts only version
`oob.canonical_actions.v1`; the pinned file SHA-256 is
`5769667a576d82621a5a1b013df16fde6c905871f4c2805bb93da34435ef62a2`.

The current action vocabulary and argument meanings are frozen for the paper.
In particular, `swipe` must not be redesigned per task or widget. A change to
this contract requires a new protocol version and a separate experiment; it
is not a Function, Transfer, Checker, or Harness fix.

## Allowed changes

Only these changes are in scope:

1. A task's Function asset and its immutable provenance.
2. General Function exposure, argument binding, or execution bugs that affect
   all tasks and preserve the frozen contracts.
3. Evidence needed to prove injection, selection, binding, Transfer, execution,
   validator outcome, and model usage.
4. Paper tables, figures, and analysis.

The only per-task artifact that normally changes is the Function asset and its
provenance. Harness behavior, Action/Observation/RunLog schemas, device
profiles, and external validator behavior are evaluation conditions.

Every code fix must have a focused regression test and must not add a task,
device, action, Function-ID, or failure-type special case.

## Required evidence

Every paper result must identify:

```text
source RunLog + SHA-256
Function asset + SHA-256
code commit
method and target device
source/evaluation seed
model and OmniTransfer version
Function injected / selected / bound / executed
Transfer result
official validator result
model calls, tokens, fallback steps, duration
```

The source RunLog and Memory path are explicit inputs.  Runtime must not infer
them from sibling files, scan historical results, or select a different asset.

## Failure attribution

When a result fails, classify it before changing anything:

```text
function injection
function asset
transfer
external harness
policy
official validator
```

Only the first two categories can normally justify a change in this
repository.  A Transfer or Harness failure is evidence for the paper, not a
reason to redesign an Action, Function Record, or external interface.

## Iteration rule

Freeze the code commit and Function asset before each comparison.  Re-run the
same explicit source and target inputs when debugging.  Generate a new asset
or a new experiment only when the paper question requires it; never mutate a
previous result in place.

## Long-term Function-first rule

For every task, the first question is whether the Function asset is correct:
its source provenance, ordered actions, parameters, bindings, and transfer
states must agree with one successful source RunLog.  The same explicit Memory
must be used on every target device in a comparison.

Do not change the Harness, Action Schema, RunLog format, Validator, Planner
contract, or OmniTransfer to improve one task.  A failed Function injection or
an invalid asset is a repository issue; a valid Function with failed mapping,
external execution, or Planner recovery is recorded as evidence with its
failure category.  Never replace a Transfer failure with source-coordinate
replay or a task-specific fallback.

The only per-task correction loop is:

```text
qualify source -> validate Function asset -> run identical Memory on targets
  -> record injection/Transfer/execution/validator evidence -> classify failure
```

Function selection is not execution authorization. A Planner may propose a
Function because its name matches the goal, but the runtime must admit its
first contextual action on the current observation before dispatching it. A
returned coordinate alone is not proof of a valid Transfer: missing confidence,
an incompatible target capability, or a mismatched page must produce an
explicit Transfer failure and the normal fallback. This rule is shared across
all tasks and devices; it must not be implemented by adding task-specific
entry actions or by making a Function contain unrelated navigation steps.

On 4090, this loop runs only from `/home/zewen/Projects/OmniFlow-exp`, using
the repository `.venv`, the explicit Function Store, and the fixed experiment
protocol.  An offline AVD, stale checkout, or uncommitted code mismatch is an
environment/provenance failure and must be recorded before any task diagnosis.

## Automated workbook evidence update

The workbook is updated by the evidence writer, not by manually editing cells:

```bash
bash scripts/exp/update_androidworld_workbook.sh \
  --workbook /Users/wuzewen/Desktop/OmniFlow-AndroidWorld-Experiments/OmniFlow_AndroidWorld_116Tasks_12cell_autodroid.xlsx \
  --task SystemBrightnessMax \
  --source-run-log data/androidworld/SystemBrightnessMax/source/OmniFlowSourceSmall_seed111/runlog/current/run_log.json \
  --result standard=/absolute/path/to/standard/run_log.json \
  --result fold=/absolute/path/to/fold/run_log.json \
  --result tablet=/absolute/path/to/tablet/run_log.json
```

Each target evidence path is explicit. The writer does not scan attempts or
choose a historical result. It locates the task row, generates Source and
Standard/Fold/Tablet text from RunLog and `task_results.jsonl`, preserves the
existing workbook style, and updates `Run Receipts` idempotently. Official
validator success is recorded independently from Function replay; Transfer
failures and Planner/VLM fallback remain visible in the cell. The script is an
evidence/reporting utility only and is not an AndroidWorld execution path.
For remote evidence, pass the local staging file with `--result` and the
canonical provenance path with the matching optional `--result-display` entry;
the writer reads locally but records the supplied remote path.

The writer also accepts `--method appagent` and recognizes AppAgent's
`official_validator_success` and `classification` fields. `--batch-json`
supports incremental imports of already-finished target cells; it never
selects or starts an experiment.
