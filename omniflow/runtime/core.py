"""Minimal recorded-action execution path.

This module deliberately contains no checker, retry, recovery, planner, or
payment policy. Robust runtimes wrap this interface instead of changing its
transfer decision.
"""

from __future__ import annotations

import asyncio
import inspect
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
from omniflow.transfer.admission import assess_transfer, requires_contextual_mapping

_ACTION_SETTLE_SECONDS = 1.0


async def execute_action(
    action: Action,
    *,
    observation: Observation,
    host: Host,
    plugins: PluginSet,
    source_state: Observation | None = None,
    function_id: str | None = None,
) -> StepResult:
    """Transfer once, dispatch once, settle once, and observe once."""

    decision = await prepare_action(
        action,
        observation=observation,
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
    if _ACTION_SETTLE_SECONDS > 0.0:
        await asyncio.sleep(_ACTION_SETTLE_SECONDS)
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
    after = Observation.from_value(
        await _await(host.observe(xml=True, screenshot=True, app_info=True))
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
    plugins: PluginSet,
    source_state: Observation | None = None,
) -> ActionDecision:
    """Return the real transfer adapter's decision without a second gate."""

    if (
        source_state is None
        or not requires_contextual_mapping(action.tool, action.args)
    ):
        return ActionDecision("ready", action=action)
    if plugins.transfer is None:
        return ActionDecision("block", reason="transfer_not_configured")
    transfer = await _await(plugins.transfer(action, observation, source_state))
    admission = assess_transfer(transfer, observation=observation)
    if not admission.accepted:
        return ActionDecision(
            "block",
            reason=admission.reason or transfer.reason or "transfer_failed",
            detail={
                **dict(transfer.detail),
                "mapping_confidence": admission.confidence,
            },
        )
    return ActionDecision(
        "ready",
        action=transfer.action,
        reason=transfer.reason,
        detail=transfer.detail,
    )


async def _await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


__all__ = ["execute_action", "prepare_action"]
