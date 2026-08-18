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

The Agent edits one in-memory draft through three stages. Stage 1 runs once;
stages 2 and 3 run separately for each identified Function. It does not make
one model call per action and never writes complete Function artifacts, source
states, complete source actions, bindings, checker rules, or Store entries.

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
- Describe only effects caused by actions inside the selected range. Treat a
  condition already true in the first state as a precondition, not an effect.
- Do not select a range whose behavior depends on a transient dialog, task
  completion, validator state, target device, or task-specific coincidence.
- Do not create isolated click or long-press fragments.
- Return an empty `subsegments` list when no stable reusable range exists.

### Stage 2: source-proven actions and parameters

This stage receives exactly one Function and only the RunLog source actions in
that Function's range. Return entries only for the shown `function_id` and use
the listed original RunLog `step_index`; never use a local Function index.

Return:

```json
{
  "action_edits": [
    {
      "function_id": "search_for_a_place",
      "step_index": 0,
      "operation": "open_app",
      "value": "com.example.maps"
    },
    {
      "function_id": "enter_search_query",
      "step_index": 2,
      "operation": "set_target",
      "value": "Search"
    }
  ],
  "bindings": [
    {
      "function_id": "enter_search_query",
      "step_index": 2,
      "name": "query",
      "description": "Place query to enter"
    }
  ]
}
```

Rules:

- `action_edits` may contain only `open_app` and `set_target`.
- Use `open_app` only for a launcher click whose `after_page.package` is a
  different non-empty package. Copy that package exactly into `value`.
- Use `set_target` only when `source_target` is non-empty. Copy that label
  exactly into `value`; do not paraphrase it.
- Declare caller-varying values already present after the validated action edit.
- Bind only source values stated directly in the RunLog goal. A current page
  value absent from the goal is source state, not caller input.
- Do not return a binding path. The compiler derives `text` for `input_text`
  and `target_description` for a source-proven semantic click.
- The source step must be inside the named Function range.
- A varying visible selection such as an hour or category may bind
  `target_description` after `set_target`.
- Never parameterize coordinates, package names, waits, directions, source
  states, or transfer evidence.
- Return empty `action_edits` or `bindings` lists when that decision has no
  source-proven entry.

### Stage 3: checkers

This stage receives exactly one Function and only its source actions. Return
checker registrations only for the shown `function_id` and listed source
indices.

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
- A source target named by the RunLog goal or Function name/description is task
  progress and cannot be a checker.
- One source action cannot be a checker in one emitted Function and a formal
  step in another.
- Return an empty `checker_steps` list when no checker is safe.

## Deterministic compilation

`save_function` preserves source order and source states from the successful
RunLog, verifies action edits against exact before/after evidence, removes
registered checker actions from formal steps, creates parameter schemas and
bindings, validates every Function, and writes the Store atomically. The Agent
supplies decisions only; it cannot invent or modify source evidence.

A rejected stage edit receives at most one correction of that same stage.
Transport failure, missing source evidence, or a second invalid edit fails the
save without partial persistence.

Runtime evaluates every unexecuted Function-local checker before every pending
formal action. OmniTransfer must find a target above the one configured high
probability threshold. Page-embedding similarity is not a trigger. A missing or
low-confidence mapping skips the checker without executing source coordinates.

Before cross-environment evaluation, the complete Function must pass source
environment `script_replay` with official success, `model_calls=0`, and
`fallback_steps=0`.
