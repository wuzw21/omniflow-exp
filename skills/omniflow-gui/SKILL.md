---
name: omniflow-gui
description: Recall and execute reusable OmniFlow Functions on a configured device. Use for GUI tasks that can benefit from cross-device Function Memory, while the host retains task planning and primitive actions.
---

# OmniFlow Functions

Use exactly two OmniFlow service tools: omniflow_recall and omniflow_execute.
The configured server supplies explicit Memory and its device backend. If these
tools are unavailable, report that the OmniFlow MCP service must be configured.

## Recall

Call omniflow_recall with a task_id, the current goal, and limit (normally 8).
Use the SAME task_id throughout a logical task. A new id explicitly starts a
new task and must not be used to evade cancellation, a deadline, or an unknown
side effect.

Read the returned candidate Function ids, descriptions, input schemas, current
observation, and session_id. Recall does not execute device actions. Empty
Memory or no suitable candidate is a valid result; do not invent a Function.

## Execute

Assess the current goal and the Function's starting conditions before invoking
it. A stored click on a toggle reverses its state; a name such as "turn on"
does not make it idempotent. If current evidence already satisfies the goal,
stop without executing. If a state-sensitive precondition is unknown, inspect
it through the host's device channel before choosing that Function.

Call omniflow_execute with the returned session_id, a unique request_id,
function_id, and arguments matching that Function's input schema.
This executes one registered Function through the shared
Checker → OmniTransfer → Act → Observe loop, then returns control to you.

Read feedback even if isError=true. Execution success is not verified task
success. A failure can include a successfully executed prefix, a failed step,
and a current observation. Never blindly replay the prefix, reuse source-device
coordinates, or interpret the step index as a resume token.

For a transport retry, reuse exactly the same request_id and arguments.
Do not generate a new request id to repeat an action with an unknown effect.
A restarted process has a different session; it cannot safely resume old ids.

If an interrupted tool call has no OmniFlow invocation feedback, treat its
effects as unknown. A host's generic "tool rejected" message or process exit
code does not prove that no device action ran. Inspect the current device
through the host's observation channel before considering recovery; if that
channel is unavailable, report the uncertainty and stop. Do not infer a safe
retry or begin a new task from the generic error alone.

## Host responsibilities

You own the task loop, completion judgment, user input, primitive actions, and
cancellation. Use the host's MCP cancellation channel or embedded runtime
cancel API on a user stop request. Drain an already-dispatched action before
any new operation; control.next=stop forbids continuing the old task.

If no Function fits, use the host's existing device action channel, or explain
that the task is blocked. Desktop computer use is not automatically an Android
action channel. Never restore a hidden internal Planner to mask that gap.

Use current observation evidence to assess completion. Without an authoritative
verifier, do not claim official benchmark success. Keep images on the server
and avoid copying base64 into notes. Do not modify Function source actions,
bindings, the canonical matcher, or benchmark protocols.
