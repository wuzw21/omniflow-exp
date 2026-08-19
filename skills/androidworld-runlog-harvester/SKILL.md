---
name: androidworld-runlog-harvester
description: "Create reusable OmniFlow Functions from one official-successful AndroidWorld or B-MoCA RunLog through save_function."
---

# Function authoring

Use `save_function` as the only Function creation or update path. One successful
RunLog produces exactly one complete Function atomically.

## Procedure

1. Call `tools/list`.
2. Inspect source evidence only when needed with `get_run_log` and
   `get_run_log_state`.
3. Call `save_function` once. Set `enhance=true` for Agent authoring.
4. Do not call or create another converter, compiler, enhancer, manifest, or
   Store writer.

## Enhancement contract

The Agent edits one in-memory draft through three stages. Stage 1 runs once;
stages 2 and 3 run for that one Function. It does not make
one model call per action and never writes source states or Store entries. The
action stage may return source-indexed direct actions, but only when copied from
RunLog evidence and accepted by the deterministic `save_function` validator.

Return only the strict schema supplied for the current
`edit_function_draft` call. The stages still produce one atomic
`save_function` bundle: the complete Function is mandatory, while direct source
actions, parameters, and checker registrations are optional additions.

### Stage 1: semantic ranges

Return:

```json
{
  "complete_function": {
    "function_id": "search_for_a_place",
    "name": "Search for a place",
    "description": "Enter a place query and show its results."
  }
}
```

Rules:

- `complete_function` is mandatory and covers the entire successful RunLog.
- The saved source replay calls only `complete_function`.
- The complete Function preserves every successful source action in order.
- Describe only effects caused by actions inside the selected range. Treat a
  condition already true in the first state as a precondition, not an effect.
- Do not split the complete trajectory or return a second Function.

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
      "operation": "open_app"
    },
    {
      "function_id": "enter_search_query",
      "step_index": 2,
      "operation": "set_target"
    }
  ],
  "bindings": [
    {
      "function_id": "enter_search_query",
      "step_index": 2,
      "name": "query",
      "description": "Place query to enter"
    }
  ],
  "actions": [
    {
      "function_id": "enter_search_query",
      "step_index": 1,
      "action": {"tool": "input_text", "args": {"text": "museum"}}
    }
  ]
}
```

Rules:

- `action_edits` may contain only `open_app` and `set_target`.
- `actions` is optional for exploration-style authoring. Each action must be
  copied from the shown RunLog source step, using its original index;
  `save_function` rejects any changed coordinate, package, wait, direction, or
  other ungrounded action.
- Use `open_app` only for a listed eligible launcher click whose
  `after_page.package` is a different non-empty package.
- Use `set_target` only for a listed eligible click whose `source_target` is
  non-empty.
- Do not return a package, target, label, or `value` field. The compiler copies
  the exact value from validated RunLog evidence.
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

Each stage gets at most three model attempts. A rejected decision receives only
that stage's deterministic validation error before the Agent revises the same
in-memory draft. Transport failure, missing source evidence, or a third invalid
decision fails the save without partial persistence. Never edit a generated
Function or Store directly; repair this shared policy and regenerate from the
same successful RunLog through `save_function`.

Runtime evaluates every unexecuted Function-local checker before every pending
formal action. OmniTransfer must find a target above the one configured high
probability threshold. Page-embedding similarity is not a trigger. A missing or
low-confidence mapping skips the checker without executing source coordinates.

Before cross-environment evaluation, the complete Function must pass source
environment `script_replay` with official success, `model_calls=0`, and
`fallback_steps=0`.
