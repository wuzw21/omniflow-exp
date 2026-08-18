---
name: androidworld-runlog-harvester
description: "Create reusable OmniFlow Functions from one official-successful AndroidWorld or B-MoCA RunLog through save_function."
---

# Function authoring

Use `save_function` as the only Function creation or update path. One
successful RunLog may produce the mandatory complete Function and zero or more
reusable subsegment Functions atomically.

## Procedure

1. Call `tools/list`.
2. Inspect source evidence only when needed with `get_run_log` and
   `get_run_log_state`.
3. Call `save_function` once. Set `enhance=true` for Agent authoring.
4. Do not call or create another converter, compiler, enhancer, manifest, or
   Store writer.

## Enhancement contract

The Agent edits one in-memory draft in exactly three stages. It does not make
one model call per action and never writes complete Function artifacts, source
states, source actions, bindings, checker rules, or Store entries.

Return only the strict schema supplied for the current
`edit_function_draft` call.

### Stage 1: semantic ranges

Return:

```json
{
  "complete_function": {
    "function_id": "search_for_a_place",
    "name": "Search for a place",
    "description": "Enter a place query and show its results."
  },
  "subsegments": [
    {
      "function_id": "enter_search_query",
      "name": "Enter a search query",
      "description": "Enter caller-provided text in the search field.",
      "stability_reason": "The same visible search field and input action form a deterministic sequence; the query is parameterized.",
      "start_step_index": 1,
      "end_step_index": 3
    }
  ]
}
```

Rules:

- `complete_function` is mandatory and covers the entire successful RunLog.
- A subsegment is an independently useful contiguous semantic operation.
- `start_step_index` is inclusive and `end_step_index` is exclusive.
- Select a subsegment only after identifying it as stably reproducible.
- `stability_reason` is mandatory. It must explain why the source-state/action
  sequence is deterministic across environments and remains replayable after
  caller-varying content is parameterized.
- Do not select a range whose behavior depends on a transient dialog, task
  completion, validator state, target device, or task-specific coincidence.
- Do not create isolated click or long-press fragments.
- Return an empty `subsegments` list when no stable reusable range exists.

### Stage 2: parameters

Return:

```json
{
  "bindings": [
    {
      "function_id": "enter_search_query",
      "step_index": 2,
      "name": "query",
      "description": "Place query to enter",
      "argument_path": "text"
    }
  ]
}
```

Rules:

- Declare only caller-varying values already present in that source action.
- `argument_path` is relative to `action.args`.
- The source step must be inside the named Function range.
- Never parameterize coordinates, package names, waits, directions,
  `target_description`, source states, or transfer evidence.
- Return an empty `bindings` list when no parameter is needed.

### Stage 3: checkers

Return:

```json
{
  "checker_steps": [
    {
      "function_id": "search_for_a_place",
      "step_index": 0
    }
  ]
}
```

Rules:

- Register a checker only on a Function whose range contains that source step.
- Select only an optional source-state-dependent action that is safe to skip and
  has a later formal action in the same Function.
- Allowed checker actions are `click`, `input_text`, and `long_press`.
- A parameterized action cannot be a checker.
- Required navigation, terminal actions, waits, app launches, key presses, and
  swipes are not checkers.
- Return an empty `checker_steps` list when no checker is safe.

## Deterministic compilation

`save_function` copies exact actions and source states from the successful
RunLog, removes registered checker actions from formal steps, creates parameter
schemas and bindings, validates every Function, and writes the Store atomically.
The Agent supplies decisions only; it cannot modify source evidence.

A rejected stage edit receives at most one correction of that same stage.
Transport failure, missing source evidence, or a second invalid edit fails the
save without partial persistence.

Runtime evaluates every unexecuted Function-local checker before every pending
formal action. The canonical OmniTransfer page embedding must match the source
state and OmniTransfer must find a target above the configured probability
threshold. A failed match skips the checker without executing source
coordinates.

Before cross-environment evaluation, the complete Function must pass source
environment `script_replay` with official success, `model_calls=0`, and
`fallback_steps=0`.
