# OmniFlow shared schemas

These are the wire contracts shared by OpenOmniBot and OmniFlow:

- `oob_canonical_actions.v1.json`: executable `{tool, args}` actions.
- `omniflow_run_log.v1.json`: canonical AndroidWorld/OOB RunLog.
- `omniflow_function.v3.json`: reusable Functions with embedded RunLog-shaped
  observations as transfer states.
- `omniflow_checker_rule.v1.json`: the unchanged canonical learned-recovery
  contract from the main OmniFlow repository.
- `omniflow_android_bridge.v2.json`: JSON-line bridge API.

Each Function owns a `transfer_states` object. Every formal step references one
or more observations with `transfer_state_ids`; the same observation may be
reused by several steps. Function actions and order are independent of any
RunLog trajectory. Source coordinates are accepted only as OmniTransfer source
evidence and never execute directly on a target device.

A checker remains Function-local and keeps the canonical v1 fields
`schema_version`, `trigger`, `source_state_id`, and `action`. Its
`source_state_id` resolves inside the owning Function's embedded
`transfer_states`; no sibling state catalog is used.

The offline Agent may author one or more semantic Functions. A RunLog may supply
observations during authoring, but the compiler does not require action coverage,
trajectory order, or semantic equality. `save_function` writes only the v3 Store.

Canonical actions use relative `0..1000` coordinates. The VLM boundary uses
pixels in the current display frame, with conversion owned only by
`omniflow.vlm_coordinates`. Unsupported actions and invalid persisted values
fail validation rather than entering a compatibility path.
