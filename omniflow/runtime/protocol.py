"""Host-facing execution facts; independent of Planner and benchmark schemas."""

from __future__ import annotations

from omniflow.runtime.timing import timed

from typing import Any

from omniflow.core.model import Observation, RunResult

PROTOCOL_VERSION = "omniflow.invocation.v1"


@timed("completion_checker")
async def review_completion(checker) -> dict[str, Any] | None:
    """One completion judgment for built-in and external task harnesses."""
    if checker is None:
        return None
    from omniflow.runtime.control import invoke
    try:
        reward = float(await invoke(checker))
    except Exception as error:
        return {"status": "error", "error": f"{type(error).__name__}:{error}"}
    return {"status": "verified" if reward > .5 else "rejected", "reward": reward}


def observation_payload(observation: Observation | None) -> dict[str, Any] | None:
    if observation is None:
        return None
    payload = observation.to_dict()
    if observation.extra.get("screenshot_path"):
        payload["image_base64"] = None
    return payload


def invocation_feedback(result: RunResult) -> dict[str, Any]:
    """Do not infer task success from successful device actions."""
    detail = result.detail
    reason = str(detail.get("done_reason") or "")
    terminal = reason in {
        "finished", "function_completed_verified", "cancelled", "deadline_exceeded",
        "verifier_error", "abort", "waiting_input", "effect_unknown", "no_progress", "step_budget_exceeded",
    }
    gate = detail.get("completion_gate") or (detail.get("function_execution") or {}).get("completion_gate") or {}
    task_status = {
        "verified": "verified_success", "rejected": "verified_incomplete",
        "error": "verifier_error",
    }.get(gate.get("status"), "unknown")
    if reason == "function_completed_verified":
        task_status = "verified_success"
    if reason == "verifier_error":
        task_status = "verifier_error"
    failed_step = detail.get("failed_step_index")
    if failed_step is None:
        failed_step = (detail.get("function_resolution") or {}).get("failed_step_index")
    return {
        "schema_version": PROTOCOL_VERSION,
        "execution": {
            "status": "completed" if result.success else "yielded" if failed_step is not None else "failed",
            "function_id": result.function_id,
            "actions_executed": result.actions_executed,
            "next_step_index": detail.get("next_step_index", failed_step),
            "failed_step_index": failed_step,
            "failed_action_dispatched": detail.get("failed_action_dispatched"),
            "error": result.error,
        },
        "observation": observation_payload(result.final_state),
        "task": {"status": task_status},
        "control": {
            "next": "stop" if terminal else "host",
            "reason": reason or "invocation_returned",
            # Transfer failure is a yield, never implicit permission to resend.
            "automatic_retry": False,
        },
    }
