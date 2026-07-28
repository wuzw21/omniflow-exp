---
name: androidworld-runlog-harvester
description: Collect, author, validate, and test AndroidWorld OmniFlow Function memory one task at a time.
---

# AndroidWorld Runlog Harvester

Own one AndroidWorld task from successful exploration through accepted Function
evidence. The agent makes semantic decisions. Core and harness code validate
protocols, execute actions, initialize experiments, preserve logs, and run the
official validator.

## One Runtime Path

Normal execution is exactly:

```text
goal
  -> Function selection and argument extraction
  -> JSON Schema validation
  -> explicit JSON-path binding
  -> observe -> checker -> transfer -> act
  -> VLM fallback after any Function miss or failure
  -> official AndroidWorld validator
```

Every stored item is an ordinary Function. Complete trajectories, reusable
semantic subsequences, and learned one-action recoveries use the same schema,
selector, binder, checker, transfer, executor, and fallback path. Never add
runtime branches for Root, Child, recovery, exact-goal, task name, seed, or
device.

## Function Generation

Use one interface: `compile_runlog_to_store(...)`.

The compiler removes XML, screenshots, task parameters, validator state, source
context, and hidden runtime data from the author input and saved Functions.
It freezes only Function-referenced source states in `transfer_states.json`,
which Transfer loads by `state_id`; the resolver, author model, bindings, and
Function schema never receive that state. Without an author, `default_bundle`
saves exactly one Function containing the complete successful multi-action
sequence. It never registers every recorded action, and a one-action RunLog
produces no default Function.

An author model is optional. The active Codex agent may supply an offline
`function_bundle=`, and the deterministic no-author path may save the one
complete Function described above. Explicit Agent- or model-authored bundles
may additionally contain genuinely reusable semantic Functions. If an author
model returns no semantic Function, do not create a Store.

For batch authoring, never call Qwen or another external author model. The active
Codex agent reads each successful RunLog offline, authors semantic Functions,
and submits them through `function_bundle=`. The evaluation model may perform
normal Function recall, argument extraction, and VLM fallback, but it never
authors memory.

## Function Acceptance

Save a Function only when all of these hold:

- Its description states reusable intent rather than a benchmark task name.
- Its action sequence is a complete capability or a reusable semantic
  subsequence, not an arbitrary slice of a successful RunLog.
- The deterministic `default_bundle` contains one complete multi-action
  Function and no semantic subsequences.
- A one-action Function is saved only when the action itself is a named,
  reusable semantic capability or learned recovery explicitly supplied by an
  Agent or author model; `default_bundle` never creates one.
- Every parameter is inferable from the fresh goal and is already action-ready.
- Every binding is a direct field or fixed-index JSON path to an existing action
  field.
- Every coordinate action uses a source point inside the uniquely identified
  semantic source element. When a recorded point came from an arbitrary
  screenshot-relative fallback but `target_description` uniquely matches a
  source element, author the normalized center of that element. If no unique
  semantic source element exists, do not claim the sequence is transferable.
- The Function does not preserve a hardcoded source answer while claiming
  parameterized reuse.
- Dynamic UI discovery, visual transcription, hidden answers, and unbounded
  loops are not falsely represented as fixed replay.
- Its ordered actions fit the runtime action budget, or it is split into
  genuinely reusable semantic Functions.
- Checker rules are condition-only `omniflow.checker.v2` rules.

When a complete task cannot be expressed from fresh-goal values, mark the
complete capability unavailable and keep only genuinely reusable semantic
subsequences. Do not manufacture atomic Functions to avoid an empty result.

Never implement goal templates, lexical scoring, page filtering, task-parameter
injection, source hints, forced Function execution, binding transforms, lookup
tables, sidecar adapters, automatic Function writeback, or checker recovery IDs.

## Per-Task Loop

1. Select one unfinished task from the read-only inventory.
2. Explore in one normal AndroidWorld episode and preserve all evidence.
3. Require live official success and strict raw replay success.
4. Author semantic and complete Functions from the successful RunLog.
5. Call `compile_runlog_to_store(..., function_bundle=...)` once.
6. Accept the Store only when the strict v2 schema validator passes.
7. Freeze a new seed and task parameters.
8. Run normal goal-driven recall on source and target devices.
9. Preserve every failure and record the first divergence.
10. Fix only general Function semantics, bindings, actions, or checker memory.
11. Run the six-cell matrix only after both ours cells pass.

Do not retry an unchanged bundle. Each revision uses a new immutable output
directory and the same frozen evaluation tuple. Never change the seed to avoid a
failure. Never branch core, harness, resolver, checker, or Function content on a
task name, seed, generated value, or device.

## Compilation

```python
from omniflow import compile_runlog_to_store

result = compile_runlog_to_store(
    source_runlog,
    version_root,
    function_bundle=codex_authored_bundle,
)
```

The compiler validates each Function through `omniflow.artifact`, binds source
arguments once, and saves the referenced source states for OmniTransfer. It
saves exactly the authored Functions when a bundle/model is supplied; otherwise
it saves one complete multi-action Function. It does not repair invalid output
or append atomic actions.

## Honest Reporting

For every cell report the seed, task parameters, device, Store hash, resolver
model calls, selected Function ID, extracted arguments, actions executed, first
divergence, fallback steps, and official validator result. Keep source success,
ours source, ours target, and complete six-cell counts separate. Never count
direct replay, reused tuples, old Stores, or unvalidated runs as formal success.
