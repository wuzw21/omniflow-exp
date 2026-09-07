---
name: omniflow-gui
description: Use OmniFlow's configured MCP tools to operate an OOB Android device with reusable cross-device Functions. Use for Android GUI tasks requiring OmniFlow Memory, Function execution, recovery, or session cancellation.
---

# OmniFlow GUI execution

Use the configured OmniFlow MCP server. It supplies the device connection and
explicit Memory; this skill contains neither a driver nor a second Planner.
If its tools are unavailable, report that the OmniFlow MCP server must be
configured. Do not substitute another Android action backend.

## Discover and observe

- Call `omniflow_tools` for the available canonical action and Function schemas.
  Do not guess Function names, arguments, task bindings, or source coordinates.
- Read `omniflow_status` and retain the current `session_id`.
- Observe with `omniflow_observe` before deciding an action. Use the current
  image, UI state, original display, and coordinate contract from the schemas.

## Execute one decision

Call `omniflow_execute` with `session_id`, a unique `request_id`, `tool_name`, and
schema-valid `arguments`. Prefer a relevant registered Function when its inputs
and current entry state fit the task. The Function handles its own shared
Checker → OmniTransfer → Act → Observe loop. You own the outer task loop.

Read all returned feedback, even when the MCP result has `isError=true`:

- `execution` reports the successful prefix, failed step, and whether the
  failed action was dispatched. Successful Function execution is not task success.
- `observation` describes the latest state; reuse it rather than requesting
  another screenshot when it is sufficient for the next decision.
- `task.status` distinguishes unknown, verified success, verified incomplete,
  and verifier error.
- `control.next=stop` ends the task. Otherwise choose the next decision using
  the current observation and failure evidence.

On Function failure, do not blindly restart the Function or replay source
coordinates. Use current-state actions to recover or explain why progress is
blocked. A failed-step index is evidence, not a resume token. No public durable
resume API exists in v1.

## Retry, stop, and restart

- On a transport interruption, query status. To retrieve a prior result, reuse
  exactly the same request id and arguments. Never create a new id merely to
  repeat an operation with an unknown effect.
- `effect_unknown`, cancellation, verifier error, and exhausted budgets stop
  further dispatch. Do not reset the session to bypass these limits.
- When the task is complete, call `omniflow_finish(session_id, content)`.
  A service without an authoritative verifier returns unknown: report observed
  completion evidence without claiming official benchmark success.
- On a user stop request, call `omniflow_cancel(session_id)` immediately.
  An in-flight action may finish; do not send another action while it drains.
- Use `omniflow_start` only for an explicitly new task after the old session
  closes. Process restart creates a new session and cannot safely resume old ids.

## Boundaries

Keep Memory and original screenshot evidence on the server. Do not copy image
base64 into long-running notes or prompts. Do not modify Function source steps,
bindings, or the canonical OmniTransfer matcher. Mapping failure must return
to normal host recovery, never resource-id lookup or source-coordinate replay.
Codex native computer use may control its supported desktop surfaces; the
OmniFlow MCP server controls Android through OOB. Do not run competing device
controllers on the same session. Bug acceptance requires a physical-device
test with device, installed APK version, actions, and results recorded.
