# OmniFlow shared schemas

These are the wire contracts shared by OpenOmniBot and OmniFlow:

- `oob_canonical_actions.v1.json`: executable `{tool, args}` actions.
- `omniflow_run_log.v1.json`: canonical AndroidWorld/OOB RunLog.
- `omniflow_function.v2.json`: reusable Function with ordered formal steps and
  Function-local `checker_rules`.
- `omniflow_checker_rule.v2.json`: exactly one RunLog source state and source
  action registered on a Function.
- `omniflow_android_bridge.v2.json`: JSON-line bridge API.

Functions and checker rules reference immutable source evidence by
`source_state_id`; they do not embed XML or screenshots. Source coordinates are
accepted only as OmniTransfer source-target evidence and never execute directly
on a target device.

A checker is registered only on the Function containing it. Before each pending
formal action, the latest OmniTransfer page embedding first matches each
unexecuted rule's source state to the current page, then OmniTransfer maps the
source action onto the current observation. Only a high-confidence page and
action-target match executes. Nonmatching rules remain eligible before later
formal actions. There is no per-rule condition, trigger DSL, step-index
trigger, global checker list, or default recovery rule.

The offline Agent may author complete semantic Functions through the internal
split, parameter-binding, and checker-review stages. Every stage returns one
complete bundle; the core verifies RunLog evidence, validates the schema, and
stores the final bundle through `save_function`.

Canonical actions use relative `0..1000` coordinates. The VLM boundary uses
pixels in the current display frame, with conversion owned only by
`omniflow.vlm_coordinates`. Unsupported actions and invalid persisted values
fail validation rather than entering a compatibility path.
