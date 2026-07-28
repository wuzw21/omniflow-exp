# OmniFlow Shared Schemas

These files are the wire contracts shared by OpenOmniBot and OmniFlow. Copies
in both repositories must remain byte-for-byte identical.

- `oob_canonical_actions.v1.json`: executable actions as `{tool, args}`.
- `omniflow_canonical_run_log.v1.json`: captured steps as
  `step_index/before_state_id/action/result/after_state_id/metadata`, plus one
  optional `final_state_id`.
- `omniflow_function.v2.json`: reusable Functions with `function_id`,
  `input_schema`, `bindings`, and `steps`; each step references
  `source_state_id`.
- `omniflow_checker_rule.v1.json`: optional offline-learned replay rules with
  exactly `schema_version/trigger/source_state_id/action`.
- `omniflow_android_bridge.v2.json`: the Android/Python Bridge API.

`state_id` is the only persisted UI-state reference. RunLogs and Functions do
not embed captured UI evidence. XML, screenshots, hashes, byte counts, and file
paths are Evidence Store implementation details and are resolved by `state_id`
only when transfer or diagnostics need them.
Function files never embed XML or screenshots. The Bridge resolves
`state_id` on demand and must block coordinate actions when state lookup or
OmniTransfer mapping fails; source coordinates must never pass through.

Production writers, compilers, stores, and replay code accept only these
contracts. Historical AndroidWorld data is normalized at its import adapter,
not inside the core runtime.

Canonical Actions use `0..1000` relative coordinates, but the VLM wire boundary
uses raw pixels in the current original device display frame so it matches XML
bounds. `omniflow.vlm_coordinates` is the only VLM conversion owner: it converts
canonical recent-action context to pixels before the call and converts validated
raw-pixel tool arguments back to canonical coordinates after the call. Manual
touch capture performs its raw-pixel-to-canonical conversion when the Action is
created. Screenshot transport resizing never changes the declared VLM coordinate
frame.

Checker rules are generated only during offline RunLog enhancement from an
explicit successful recovery step. The recovery step is omitted from the main
Function path; its `before_state_id` and canonical Action become the rule's
`source_state_id` and `action`. The Agent derives only the restricted Python
`trigger` justified by that evidence. A built-in deterministic recovery may
record `metadata.checker_trigger`; the fast compiler copies that verified
trigger and writes the Checker during the same conversion. Runtime executes at most one matching
recovery Action through OmniTransfer, observes again, then retries the original
Action. Missing evidence produces no Checker.

Android writers persist the canonical five truth fields plus optional
`metadata` directly. Kotlin storage validates the shared contract before every
append or upsert; OmniFlow consumes the same persisted schema offline.

RunLog step truth stays in the five required fields. Optional extensions use
only `metadata`; `step_id`, `status`, `thinking`, and `summary` are metadata,
never step-level aliases. Step success is always read from `result.success`.

Canonical action constraints include:

- `oob_canonical_actions.v1.json` is the only action-field rule source.
- Every RunLog and Function action passes through the same schema-driven
  canonical action converter before persistence.
- The converter keeps only arguments whose schema entry does not set
  `persisted: false`; runtime grounding hints such as `target_description`,
  node ids, resource ids, screenshots, and target evidence are never saved.
- There is no separate forbidden-field list or compiler cleanup list.
- Unsupported tools, invalid persisted values, and missing required persisted
  arguments fail conversion; all other non-persisted input is omitted.

The only saved arguments are:

- `click`: `x`, `y`.
- `long_press`: `x`, `y`, optional `duration_ms`.
- `input_text`: `text`.
- `swipe`: `direction`, `x1`, `y1`, `x2`, `y2`, optional `duration_ms`.
- `open_app`: `package_name`.
- `press_key`: `key`.
- `wait`: `duration_ms`.
