from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from omniflow.core.config import Experiment
from omniflow.core.model import RunResult, ToolCall
from omniflow.runtime.engine import OmniFlow


def run_function_sequence(
    flow: OmniFlow,
    calls: Iterable[dict[str, Any] | ToolCall],
    *,
    experiment: Experiment | str | None = None,
) -> RunResult:
    """Execute ordered Function calls as one run with one shared checker budget."""

    tool_calls = tuple(ToolCall.from_value(_tool_call_value(call)) for call in calls)
    if not tool_calls:
        raise ValueError("function_sequence_calls_required")

    actions_executed = 0
    model_calls = 0
    fallback_steps = 0
    trace: list[dict[str, Any]] = []
    sequence: list[dict[str, Any]] = []
    checker_trigger_counts: dict[str, int] = {}
    final_result: RunResult | None = None

    for index, tool_call in enumerate(tool_calls):
        result = flow.call_tool(
            tool_call,
            experiment=experiment,
            checker_trigger_counts=checker_trigger_counts,
        )
        final_result = result
        actions_executed += result.actions_executed
        model_calls += result.model_calls
        fallback_steps += result.fallback_steps
        function_trace = [
            dict(item)
            for item in result.detail.get("trace") or ()
            if isinstance(item, dict)
        ]
        trace.extend(function_trace)
        sequence.append(
            {
                "index": index,
                "function_id": tool_call.name,
                "arguments": dict(tool_call.arguments),
                "success": result.success,
                "actions_executed": result.actions_executed,
                "model_calls": result.model_calls,
                "fallback_steps": result.fallback_steps,
                "error": result.error,
            }
        )
        if not result.success:
            break

    assert final_result is not None
    completed = len(sequence) == len(tool_calls) and final_result.success
    detail = dict(final_result.detail)
    detail.update(
        {
            "trace": trace,
            "function_ids": [call.name for call in tool_calls],
            "function_sequence": sequence,
            "checker_trigger_counts": dict(checker_trigger_counts),
            "done_reason": (
                "function_sequence_completed"
                if completed
                else "function_sequence_failed"
            ),
        }
    )
    return RunResult(
        completed,
        final_result.function_id,
        actions_executed,
        model_calls,
        fallback_steps,
        None if completed else final_result.error or "function_sequence_failed",
        final_result.final_state,
        detail,
    )


def _tool_call_value(call: dict[str, Any] | ToolCall) -> dict[str, Any] | ToolCall:
    if isinstance(call, ToolCall):
        return call
    if not isinstance(call, dict):
        raise ValueError("function_sequence_call_invalid")
    if set(call) == {"function_id", "arguments"}:
        return {
            "name": call["function_id"],
            "arguments": call["arguments"],
        }
    return call


__all__ = ["run_function_sequence"]
