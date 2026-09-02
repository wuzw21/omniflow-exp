"""Common action path with one canonical transfer attempt and stable retry."""

from __future__ import annotations

import inspect
import os
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
    action_result = ActionResult.from_value(await _await(host.act(decision.action)))
    if not action_result.success:
        return StepResult(
            False,
            action=decision.action,
            before=observation,
            result=action_result,
            error=action_result.error or "action_failed",
            origin="core",
            function_id=function_id,
            detail=decision.detail,
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
    after = Observation.from_value(
        cached_after
        if cached_after is not None
        else await _await(host.observe(xml=True, screenshot=True, app_info=True))
    )
    return StepResult(
        True,
        action=decision.action,
        before=observation,
        after=after,
        result=action_result,
        actions_executed=1,
        origin="core",
        function_id=function_id,
        detail=decision.detail,
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
    transfer = await _await(plugins.transfer(action, observation, source_state))
    if transfer.action is not None:
        return ActionDecision(
            "ready",
            action=transfer.action,
            reason=transfer.reason,
            detail=dict(transfer.detail),
        )
    stable_observe = getattr(host, "observe_stable", None) if host is not None else None
    if callable(stable_observe):
        stable_observation = Observation.from_value(
            await _await(stable_observe(xml=True, screenshot=True, app_info=True))
        )
        stable_transfer = await _await(
            plugins.transfer(action, stable_observation, source_state)
        )
        if stable_transfer.action is not None:
            return ActionDecision(
                "ready",
                action=stable_transfer.action,
                reason=stable_transfer.reason,
                detail={
                    **dict(stable_transfer.detail),
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
