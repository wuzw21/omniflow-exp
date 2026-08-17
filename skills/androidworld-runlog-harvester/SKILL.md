---
name: androidworld-runlog-harvester
description: "Create or revise reusable OmniFlow Functions from an official-successful AndroidWorld or B-MoCA RunLog through the single save_function authoring pipeline."
---

# Function authoring from a successful RunLog

Use one public write operation: `save_function`. One RunLog may produce several
semantic Functions, but there is no converter, manifest writer, direct Store
write, or second enhancement interface.

## Workflow

1. Call `tools/list` and use `get_run_log` plus `get_run_log_state` only when
   more source evidence is needed.
2. Confirm the RunLog is successful and preserves ordered source actions,
   source states, and screenshot references.
3. Call `save_function` once. Submit complete Functions directly, or use
   `enhance=true` for the internal split, parameter-binding, and checker-review
   stages.
4. Return the saved Function IDs and save result. Function success is only a
   Planner tool result; it is not task completion.

## Required Agent output

Every internal enhancement stage must return exactly one JSON object with the
two top-level keys below and no commentary:

```json
{
  "functions": [
    {
      "schema_version": "omniflow.function.v2",
      "function_id": "search_the_web",
      "name": "Search the web",
      "description": "Open the browser, enter provided text, and submit it.",
      "input_schema": {
        "type": "object",
        "properties": {
          "query": {"type": "string"}
        },
        "required": ["query"],
        "additionalProperties": false
      },
      "bindings": [
        {
          "source": "$.arguments.query",
          "target": "$.steps[1].action.args.text"
        }
      ],
      "steps": [
        {
          "step_index": 0,
          "source_state_id": "source-home",
          "action": {"tool": "click", "args": {"x": 700, "y": 800}}
        },
        {
          "step_index": 1,
          "source_state_id": "source-search-page",
          "action": {"tool": "input_text", "args": {"text": ""}}
        },
        {
          "step_index": 2,
          "source_state_id": "source-search-filled",
          "action": {"tool": "click", "args": {"x": 900, "y": 900}}
        }
      ],
      "checker_rules": [
        {
          "source_state_id": "source-promo-dialog",
          "action": {"tool": "click", "args": {"x": 800, "y": 700}}
        }
      ],
      "agent_visible": true
    }
  ],
  "arguments": {
    "search_the_web": {"query": "museum"}
  }
}
```

Return a complete `omniflow.function.v2` for every Function at every stage,
even when that stage makes no changes. One Function is a reusable semantic
operation, not a single click. Every stage must include at least one large
Function covering the complete successful trajectory; reusable semantic
subsegments may be added, but they never replace that complete Function. Keep
source action order and continuity. Put task-varying values in `input_schema`,
`bindings`, and `arguments`.

The split stage chooses the complete Function and reusable subsegments. The
parameter stage binds task-varying values without losing trajectory coverage.
The checker stage may move optional setup, interruption, or recovery actions
from formal steps into the checker list of the Function that needs them. It may
not duplicate a formal action as a checker, mark ordinary navigation/final-task
actions as checkers, or create a one-click Function merely to hold a checker.

## Checker rules

A checker is registered only by being present in its Function's
`checker_rules`. A checker rule contains exactly `source_state_id` and `action`.
Do not return a step number, `when`, threshold, package switch, trigger DSL, or
global checker list.

Runtime checks an unexecuted registered rule before every pending Function
action. The latest OmniTransfer page embedding must first match the current
observation to the rule's source state, then OmniTransfer must find the source
action's target above the configured high-confidence threshold. A matching
checker executes once, while a nonmatching checker remains eligible before a
later action.

## Evidence and failure rules

Every state and action must be supported by the successful source RunLog.
Never invent target-device observations, target coordinates, validator logic,
task-specific gates, or source-coordinate fallback. Report the first validation
or save error. Do not retry through another interface or write the Store
directly.
