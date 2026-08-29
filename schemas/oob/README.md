# OmniFlow Shared Schemas

These files are the wire contracts shared by OpenOmniBot and OmniFlow. Copies
in both repositories must remain byte-for-byte identical.

- `oob_canonical_actions.v1.json`: executable actions as `{tool, args}`.
- `omniflow_run_log.v1.json`: AndroidWorld `JSONAction` values plus any of the
  current compact `screenshot/xml`, legacy compact `pixels/xml/auxiliaries`,
  or native AndroidWorld State observations. Legacy screenshot SHA-256 is
  accepted on input and removed from the canonical current representation.
- `omniflow_function.v2.json`: reusable Functions with `function_id`,
  `input_schema`, `bindings`, and `steps`; each step references
  `source_state_id`.
- `omniflow_checker_rule.v1.json`: one independently reusable shared checker
  rule with scope, phase, condition, action, budget, and priority.
- `omniflow_checker_store.v1.json`: the checker library stored separately from
  all Functions and shared across an ordered Function sequence.
- `omniflow_android_bridge.v2.json`: the Android/Python Bridge API.

Current collectors persist `screenshot/xml`; older AndroidWorld/OOB collectors
may persist native State or `pixels/xml/auxiliaries`. Functions never embed XML or screenshots;
their `source_state_id` resolves the evidence captured by the RunLog. The Bridge
must block coordinate actions when state lookup or OmniTransfer mapping fails;
source coordinates must never pass through.

Production writers, compilers, stores, and replay code accept only the
`omniflow.run_log.v1` RunLog. Historical AndroidWorld data is
converted once by the explicit offline converter, never inside the runtime.

RunLog actions preserve the official AndroidWorld action vocabulary: a native
`swipe`/`scroll` carries `direction` (and an optional `index`), not OOB endpoint
coordinates. The OOB adapter expands that direction into its private executable
representation only after the RunLog boundary. Canonical Actions use `0..1000` relative coordinates, but the VLM wire boundary
uses raw pixels in the current original device display frame so it matches XML
bounds. `omniflow.vlm_coordinates` is the only VLM conversion owner: it converts
canonical recent-action context to pixels before the call and converts validated
raw-pixel tool arguments back to canonical coordinates after the call. Manual
touch capture performs its raw-pixel-to-canonical conversion when the Action is
created. Screenshot transport resizing never changes the declared VLM coordinate
frame.

Checker rules live in an independent library, never inside a Function. They may
be global or scoped by Function, step, action type, or package. Runtime evaluates
the configured phase around every Function action and shares trigger budgets
across all Function calls in one sequence. Checker recovery uses the same
canonical action dispatcher as Function execution; configured `swipe` recovery
therefore carries the canonical direction and endpoint arguments rather than a
separate gesture implementation.

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
  `persisted: false`. Node ids, resource ids, screenshots, target descriptions,
  and target evidence are never saved.
- There is no separate forbidden-field list or compiler cleanup list.
- Unsupported tools, invalid persisted values, and missing required persisted
  arguments fail conversion; all other non-persisted input is omitted.

The only saved arguments are:

- `click`: `x`, `y`.
- `long_press`: `x`, `y`, optional `duration_ms`.
- `input_text`: `text`, `x`, `y`.
- `swipe`: `direction`, `x1`, `y1`, `x2`, `y2`, optional `duration_ms` (OOB
  canonical action only; these fields are not AndroidWorld RunLog fields).
- `open_app`: `package_name`.
- `press_key`: `key`.
- `wait`: `duration_ms`.
