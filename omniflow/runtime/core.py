"""Common action path with one canonical transfer attempt and stable retry."""

from __future__ import annotations

import inspect
import os
import time
from typing import Any

from omniflow.core.config import PluginSet
from omniflow.core.model import (
    Action,
    ActionDecision,
    ActionResult,
    Host,
    Observation,
    StepResult,
)
from omniflow.transfer.runtime import requires_contextual_mapping

async def execute_action(
    action: Action,
    *,
    observation: Observation,
    host: Host,
    plugins: PluginSet,
    source_state: Observation | None = None,
    function_id: str | None = None,
) -> StepResult:
    """Transfer once, dispatch once, and consume one post-action observation."""

    decision = await prepare_action(
        action,
        observation=observation,
        host=host,
        plugins=plugins,
        source_state=source_state,
    )
    if decision.kind == "block" or decision.action is None:
        return StepResult(
            False,
            action=action,
            before=observation,
            error=decision.reason or "action_blocked",
            origin="core",
            function_id=function_id,
            detail=decision.detail,
        )
    action_started_ns = time.perf_counter_ns()
    act_started_ns = action_started_ns
    action_result = ActionResult.from_value(await _await(host.act(decision.action)))
    act_duration_ms = (time.perf_counter_ns() - act_started_ns) / 1_000_000.0
    if not action_result.success:
        detail = {
            **dict(decision.detail),
            "timing": {
                **dict(decision.detail.get("timing") or {}),
                "host_act_ms": round(act_duration_ms, 3),
                "action_dispatch_ms": round(
                    (time.perf_counter_ns() - action_started_ns) / 1_000_000.0,
                    3,
                ),
            },
        }
        return StepResult(
            False,
            action=decision.action,
            before=observation,
            result=action_result,
            error=action_result.error or "action_failed",
            origin="core",
            function_id=function_id,
            detail=detail,
        )
    take_after_action_observation = getattr(
        host, "take_after_action_observation", None
    )
    # Keep a single environment switch for the backend ablation: the
    # lightweight OOB path can be measured with or without reusing the
    # recorder's post-action observation.  The default preserves Fast Pass.
    fast_pass_enabled = (
        str(os.environ.get("OMNIFLOW_FAST_PASS", "1"))
        .strip()
        .lower()
        not in {"0", "false", "no", "off"}
    )
    cached_after = (
        take_after_action_observation()
        if fast_pass_enabled and callable(take_after_action_observation)
        else None
    )
    observation_started_ns = time.perf_counter_ns()
    after = Observation.from_value(
        cached_after
        if cached_after is not None
        else await _await(host.observe(xml=True, screenshot=True, app_info=True))
    )
    observation_duration_ms = (
        time.perf_counter_ns() - observation_started_ns
    ) / 1_000_000.0
    detail = {
        **dict(decision.detail),
        "timing": {
            **dict(decision.detail.get("timing") or {}),
            "host_act_ms": round(act_duration_ms, 3),
            "post_action_observation_ms": round(observation_duration_ms, 3),
            "action_dispatch_ms": round(
                (time.perf_counter_ns() - action_started_ns) / 1_000_000.0,
                3,
            ),
        },
    }
    return StepResult(
        True,
        action=decision.action,
        before=observation,
        after=after,
        result=action_result,
        actions_executed=1,
        origin="core",
        function_id=function_id,
        detail=detail,
    )


async def prepare_action(
    action: Action,
    *,
    observation: Observation,
    host: Host | None = None,
    plugins: PluginSet,
    source_state: Observation | None = None,
) -> ActionDecision:
    """Use the canonical transfer result, then use one stable retry on failure.

    The fallback is deliberately generic: it applies equally to Function
    actions and shared Checker actions, and never changes the action mapper or
    replays source coordinates.
    """

    if (
        source_state is None
        or not requires_contextual_mapping(action.tool, action.args)
    ):
        return ActionDecision("ready", action=action)
    if plugins.transfer is None:
        return ActionDecision("block", reason="transfer_not_configured")
    transfer_started_ns = time.perf_counter_ns()
    transfer = await _await(plugins.transfer(action, observation, source_state))
    transfer_fast_ms = (time.perf_counter_ns() - transfer_started_ns) / 1_000_000.0
    if transfer.action is not None:
        detail = {
            **dict(transfer.detail),
            "timing": {
                **dict(transfer.detail.get("timing") or {}),
                "transfer_ms": round(transfer_fast_ms, 3),
            },
        }
        return ActionDecision(
            "ready",
            action=transfer.action,
            reason=transfer.reason,
            detail=detail,
        )
    stable_observe = getattr(host, "observe_stable", None) if host is not None else None
    if callable(stable_observe):
        stable_observe_started_ns = time.perf_counter_ns()
        stable_observation = Observation.from_value(
            await _await(stable_observe(xml=True, screenshot=True, app_info=True))
        )
        stable_observe_ms = (
            time.perf_counter_ns() - stable_observe_started_ns
        ) / 1_000_000.0
        stable_transfer_started_ns = time.perf_counter_ns()
        stable_transfer = await _await(
            plugins.transfer(action, stable_observation, source_state)
        )
        stable_transfer_ms = (
            time.perf_counter_ns() - stable_transfer_started_ns
        ) / 1_000_000.0
        if stable_transfer.action is not None:
            detail = {
                **dict(stable_transfer.detail),
                "timing": {
                    **dict(stable_transfer.detail.get("timing") or {}),
                    "transfer_fast_ms": round(transfer_fast_ms, 3),
                    "stable_observation_ms": round(stable_observe_ms, 3),
                    "transfer_stable_ms": round(stable_transfer_ms, 3),
                    "transfer_ms": round(
                        transfer_fast_ms + stable_observe_ms + stable_transfer_ms,
                        3,
                    ),
                },
            }
            return ActionDecision(
                "ready",
                action=stable_transfer.action,
                reason=stable_transfer.reason,
                detail={
                    **detail,
                    "transfer_retry_path": "stable_observation",
                    "fast_transfer_failure": {
                        "reason": transfer.reason or "transfer_failed",
                    },
                },
            )
        fast_failure = {
            "path": "fast",
            "reason": transfer.reason or "transfer_failed",
        }
        stable_failure = {
            "path": "stable",
            "reason": stable_transfer.reason or "transfer_failed",
        }
        transfer = stable_transfer
    else:
        fast_failure = None
        stable_failure = None
    failure_detail = {
        **dict(transfer.detail),
        "timing": {
            **dict(transfer.detail.get("timing") or {}),
            "transfer_fast_ms": round(transfer_fast_ms, 3),
        },
    }
    if fast_failure is not None and stable_failure is not None:
        failure_detail["transfer_attempts"] = [fast_failure, stable_failure]
    return ActionDecision(
        "block",
        reason=transfer.reason or "transfer_failed",
        detail=failure_detail,
    )


async def _await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


__all__ = ["execute_action", "prepare_action"]
