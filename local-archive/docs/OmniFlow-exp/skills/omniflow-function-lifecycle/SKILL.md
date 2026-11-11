---
name: omniflow-function-lifecycle
description: Design, author, compile, validate, debug, and improve reusable OmniFlow GUI Functions from RunLogs. Use this skill whenever work involves Function semantics, global versus local Functions, Function parameters or schema, RunLog-to-Function conversion, checker extraction, Page Embedding or OmniTransfer gating, Function recall, offline regression, source-device validation, or target-device end-to-end testing. It replaces task-specific harvesting and direct-replay workflows with one evidence-driven Function lifecycle.
compatibility: Requires the canonical OmniFlow repository, its pinned Python environment, and the canonical OmniTransfer checkout configured by that repository.
---

# OmniFlow Function Lifecycle

Use one lifecycle for every task:

```text
successful RunLog
  -> evidence qualification
  -> semantic authoring plan
  -> canonical compiler
  -> immutable Function bundle
  -> offline embedding/transfer regression
  -> source end-to-end validation
  -> target end-to-end validation
  -> success and failure pairs returned to regression data
```

The central idea is that a RunLog is evidence of a successful interaction, not a
script that must be copied. A Function represents a reusable semantic capability.
The compiler, not the authoring model, owns executable artifacts.

## Establish the canonical owners

Before changing or running anything, read the repository's `AGENTS.md`, `README.md`,
architecture, file-owner guide, and the focused tests for the owner being changed.
Resolve paths from the active canonical repository instead of using backup checkouts.

Keep these ownership boundaries:

- The public or agent-facing write operation is `save_function`.
- `save_function` and experiment preparation converge on
  `omniflow.functions.compiler.compile_runlog_to_store`; do not add a second writer.
- Runtime reads registered `omniflow.function.v2` Functions and their sibling
  transfer-state catalog. Do not hand-edit Store files.
- Use the repository's configured canonical Page Encoder and OmniTransfer. Do not
  add node-id lookup, resource-id lookup, source-coordinate passthrough, or a second
  mapper as a convenience fallback.
- Use the product or benchmark's normal `OmniFlow.run(goal)` loop for real
  validation. A launcher must not select a Function or fill its arguments.

If the repository exposes a wrapper around these owners, use that wrapper. Do not
copy their implementation into the Skill.

## 1. Qualify the RunLog

Compile only from a complete successful RunLog with a trustworthy evidence chain:

- a non-empty task goal and immutable run identity;
- successful terminal status or authoritative validator success;
- ordered successful actions with before and after observations;
- screenshots and native UI observations sufficient to reconstruct every referenced
  source state;
- actual action arguments and results, without fabricated target-device data;
- enough provenance to locate the app, device profile, source revision, and attempt.

Reject zero-step, running, validator-failed, screenshot-free, observation-free, or
partially overwritten logs as Function sources. A failed RunLog is still valuable
as regression evidence, but it is not positive Function-authoring evidence.

If the source RunLog is structurally incomplete, recollect it through the normal
observe/act lifecycle. Do not repair it by copying coordinates or inventing XML.

## 2. Design capabilities, not macros

Author two complementary kinds of Function from the qualified facts:

1. Zero or more local semantic Functions. Each captures one reusable action or a
   tightly coupled contiguous group.
2. Exactly one maximal safe complete Function. It expresses the task-level semantic
   envelope and provides the Planner with a strong entry capability.

The resulting bundle therefore contains at least one Function. Functions remain
flat; do not invent nested Function calls or parent/child schemas.

### Complete Function

- Give it a goal-level name and description, not a RunLog or task-instance name.
- If the source begins with `open_app`, begin the complete Function at that action
  and include the terminal successful task action.
- It may omit unsafe middle retries, duplicate actions, setup noise, checker actions,
  and steps that require a fresh Planner observation.
- Preserve selected source action order. The complete Function is a semantic task
  envelope, not a promise that every source action is replayed.
- A Function beginning with `open_app` is eligible at task entry because opening the
  app does not depend on the current app page. Later mapped actions still obey normal
  transfer admission and execution checks.

### Local Function

- Keep its selected source actions contiguous and semantically cohesive.
- Group input and the immediately following commit, submit, confirm, or advance
  action when the latter makes the input effective.
- End the Function wherever the next decision requires observing changed UI,
  reading a generated value, computing a result, or choosing a conditional branch.
- Let a successful local Function be recalled again when the current observation
  makes it applicable. Do not globally blacklist it merely because it ran once.

### Observation-dependent repetition

Never encode a changing-UI repetition as `click N times` or expose the count as a
Function parameter. Keep one representative action as a one-step local Function,
return control to the Planner, observe the new state, and let the Planner call the
same Function again. The complete Function may preserve the task envelope, but the
runtime must hand off at the first observation-dependent repeat instead of blindly
consuming the recorded sequence.

### Parameters and schema

- Parameterize semantic values that vary with the goal, such as text to enter.
- Select parameters only from compiler-provided candidates backed by the successful
  source action.
- Coordinates and repetition counts are never Function parameters.
- A stable UI label is normally grounding evidence, not a user parameter.
- The authoring model proposes parameter name, description, source step, and argument
  name. The compiler generates `input_schema`, `bindings`, and blank parameter slots.
- The Planner sees that generated schema and supplies arguments at call time.
- Do not change the shared action schema to make one Function easier to express.

### Checkers

Use a checker only for safe, optional, state-dependent recovery such as dismissing a
permission dialog or nuisance overlay. A checker must not carry required task
progress, replace the main workflow, or be inserted merely because an action was
hard to model. Keep checker rules in the shared checker library so multiple
Functions can use them with shared trigger budgets.

## 3. Convert the RunLog through the compiler

The authoring model outputs a compact plan, not executable actions. Its conceptual
shape is:

```json
{
  "reason": "Explain the semantic split and every omission.",
  "plan": {
    "functions": [
      {
        "function_id": "enter_query",
        "name": "Enter query",
        "description": "Enter the requested query in the visible field.",
        "source_step_indices": [3],
        "parameters": [
          {
            "name": "query",
            "description": "Query requested by the user",
            "source_step_index": 3,
            "arg_name": "text"
          }
        ]
      }
    ],
    "complete_function": {
      "function_id": "complete_search",
      "name": "Complete search",
      "description": "Open the app, enter the requested query, and submit it.",
      "source_step_indices": [0, 3, 4],
      "parameters": [
        {
          "name": "query",
          "description": "Query requested by the user",
          "source_step_index": 3,
          "arg_name": "text"
        }
      ]
    }
  }
}
```

Do not let the authoring model emit actions, coordinates, source state IDs,
`input_schema`, bindings, checker rules, schema versions, or Store JSON. Require it
to account for every source step: selected, omitted as noise/retry/checker, or split
at an observation boundary.

Run conversion through the repository's public preparation path. For direct Python
development inside OmniFlow-exp, call the existing experiment wrapper rather than
constructing a Store:

```python
from src.experiment.function_v2 import compile_function_v2

report = compile_function_v2(
    run_log=run_log_path,
    output_root=immutable_output_directory,
    enhance=True,
    model=protocol_model,
)
```

The output directory must be new and immutable. A successful compile produces the
canonical Store, `transfer_states.json`, checker Store, and `compile_report.json`.
Inspect the report and artifacts through their canonical readers; do not patch JSON
after compilation. If authoring is rejected, fix the semantic plan or source
evidence and compile into a new output directory.

## 4. Recall and execution contract

Function availability follows two stages:

1. Coarse retrieval uses the goal plus Function name and description. Page Embedding
   may rank candidates but does not alone prove executability.
2. Fine mapping calls real OmniTransfer on each candidate's first source action and
   current observation.

Admit a Function only when mapping returns a non-NULL target, absolute contextual
confidence is at least the repository threshold, the target is executable for the
action, and the observation has not changed between mapping and execution. Treat a
failed gate as ordinary Function unavailability and return to Planner/VLM behavior.
Never execute the recorded source coordinate on the target device.

Node-based grounding may default to the selected node's center. Do not force every
WebView or visually mapped target into a native-node-center rule. A semantic child
label is a valid source anchor only when its geometry can represent the source
action; otherwise prefer the containing actionable target or report transfer
failure.

## 5. Test from cheap evidence to real behavior

Use this order; each level answers a different question:

1. Compiler and schema tests: Does the plan become valid immutable artifacts with
   correct parameters, bindings, ordering, and observation boundaries?
2. Offline Page Embedding and OmniTransfer regression: Does the production mapping
   seam map known source/target observation pairs without ADB, a Planner, or a model?
3. Offline runtime scenario: With a fake Host and deterministic Planner, can the real
   recall/execution engine compose the complete Function, repeated locals,
   parameters, checker behavior, and fallback correctly?
4. Source-device E2E: Starting only from a goal and registered Store, can the normal
   Planner loop finish and satisfy the authoritative validator?
5. Target-device E2E: Run the identical entry and Planner path on each target. The
   launcher supplies only goal, Store, and environment; the Planner selects Functions
   and arguments.

Offline direct Function calls may be used only as focused unit diagnostics. They do
not qualify a formal result and must not become a benchmark execution path.

In OmniFlow-exp, use focused tests such as:

```bash
./.venv/bin/pytest -q \
  tests/test_function_compiler.py \
  tests/test_function_routing.py \
  tests/test_visual_transfer_pipeline.py \
  tests/test_offline_transfer_regression.py
```

Use the repository's offline regression CLI to add both successful and failed real
observation pairs, then rerun the whole dataset. Do not test only the latest error.

For formal AndroidWorld validation, use only the repository's documented public
launcher, for example:

```bash
bash scripts/exp/run_androidworld.sh \
  --e2e-task TASK \
  --e2e-method omniflow \
  --e2e-device DEVICE \
  --e2e-source-seed 111 \
  --e2e-evaluation-seed 113 \
  --control-backend oob
```

Read device labels, seeds, model, deadlines, and thresholds from the active protocol;
the example is not permission to duplicate them in a new runner.

## 6. Diagnose by layer

Classify a failure before changing code:

- Source evidence failure: recollect the RunLog; do not tune recall or Transfer.
- Authoring failure: fix capability boundaries, omissions, or parameter selection;
  do not weaken schema validation.
- Recall failure: inspect semantic candidates and first-step evidence separately.
- Transfer failure: save the exact observation pair, mapping result, confidence, and
  expected target; fix the canonical mapper only if the labeled pair proves it.
- Runtime composition failure: inspect handoff, repeated local recall, checker budget,
  binding, and resume alignment.
- Planner failure: inspect the actual tool list, screenshot/UI context, model response,
  and token/deadline contract; do not blame Transfer for an empty or malformed tool
  call.
- Environment failure: repair transport, emulator, app state, model service, or
  timeout ownership, then rerun without recording a method conclusion.
- Workflow or validator mismatch: record it explicitly; do not overfit a Function to
  make an invalid task instance pass.

## 7. Close the evidence loop

After every real attempt:

- preserve the immutable RunLog, screenshots, observations, Function Store revision,
  transfer states, compile report, runtime trace, and validator result;
- record Function candidates, selected Function, mapping gate reason/confidence,
  mapped target, parameter values, fallback steps, model calls, tokens, and timing;
- add successful source/target pairs to the offline regression set;
- add failed or ambiguous pairs too, marking missing truth as pending annotation
  instead of treating the failed action as a label;
- rerun the complete offline dataset after a fix;
- write the official result through the benchmark's normal result owner immediately.

Do not optimize for one replay. Improve a Function only when the evidence shows a
general semantic-authoring problem; improve Transfer only when a correctly labeled
pair shows a general mapping problem; improve the Planner only when its tool/context
trace shows a planning problem.

## Completion report

Report these items concisely:

- qualified source RunLog and evidence status;
- generated complete and local Function semantics and parameters;
- omitted steps and why;
- compile artifact paths and report status;
- offline mapping totals and unresolved cases;
- source and target E2E validator outcomes;
- Function hits, covered actions, fallback/model calls, and failure classification;
- regression data and official result locations updated by the run.
