from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import inspect
import json
import math
import re
from typing import Any
import xml.etree.ElementTree as ET

from omniflow.core.config import PluginSet
from omniflow.core.model import (
    Action,
    ActionDecision,
    ActionResult,
    CheckerContext,
    Function,
    Host,
    Observation,
    RunResult,
    StepResult,
    TransferResult,
)
from omniflow.runtime.checker import default_checker_trigger, match_checker_rule
from omniflow.transfer.runtime import transfer_action

_ACTION_SETTLE_SECONDS = 1.0
_ALIGNMENT_MIN_PROBABILITY = 0.9
_ALIGNMENT_SOURCE_SKIP_PENALTY = math.log(3.0)


async def execute_function(
    function: Function,
    *,
    host: Host,
    plugins: PluginSet,
    observation: Observation | None = None,
    max_actions: int | None = None,
    start_step_index: int = 0,
    trace_start_index: int = 0,
    resume_metadata: dict[str, Any] | None = None,
    installed_packages: frozenset[str] | None = None,
) -> RunResult:
    current = observation or Observation.from_value(
        await _await(host.observe(xml=True, app_info=True))
    )
    steps = tuple(
        step for step in function.steps if step.step_index >= int(start_step_index)
    )
    if max_actions is not None and len(steps) > max_actions:
        return RunResult(
            False,
            function.id,
            0,
            error="function_exceeds_action_budget",
            final_state=current,
            detail={
                "trace": [],
                "required_actions": len(steps),
                "max_actions": max_actions,
                "next_step_index": int(start_step_index),
            },
        )
    executed = 0
    trace: list[dict[str, Any]] = []
    resume_metadata_pending = dict(resume_metadata or {})
    for function_step in steps:
        action = function_step.action
        if max_actions is not None and executed >= max_actions:
            return RunResult(
                False,
                function.id,
                executed,
                error="max_steps_exceeded",
                final_state=current,
                detail={
                    "trace": trace,
                    "next_step_index": function_step.step_index,
                },
            )
        source_state = await _load_state(host, function_step.source_state_id)
        step = await execute_action(
            action,
            observation=current,
            host=host,
            plugins=plugins,
            function=function,
            source_state=source_state,
            installed_packages=installed_packages,
        )
        executed += step.actions_executed
        trace.extend(
            await record_execution(
                host,
                step,
                trace_start_index=int(trace_start_index) + len(trace),
                metadata={"function_step_index": function_step.step_index},
                first_metadata=(
                    {"function_alignment": dict(resume_metadata_pending)}
                    if resume_metadata_pending
                    else None
                ),
            )
        )
        resume_metadata_pending.clear()
        current = step.after or step.before or current
        if not step.success:
            return RunResult(
                False,
                function.id,
                executed,
                error=step.error,
                final_state=current,
                detail={
                    "trace": trace,
                    "failed_step_index": function_step.step_index,
                    "next_step_index": function_step.step_index,
                },
            )
    return RunResult(
        True,
        function.id,
        executed,
        final_state=current,
        detail={
            "trace": trace,
            "next_step_index": (
                max((step.step_index for step in steps), default=start_step_index - 1)
                + 1
            ),
        },
    )


async def align_function_resume(
    function: Function,
    *,
    host: Host,
    plugins: PluginSet,
    observations: list[Observation],
    start_step_index: int,
) -> dict[str, Any] | None:
    transfer_fn = plugins.transfer
    if transfer_fn is None or len(observations) < 2:
        return None
    remaining = [
        step for step in function.steps if step.step_index >= int(start_step_index)
    ]
    candidate_steps = remaining[: len(observations)]
    if not candidate_steps:
        return None

    probabilities: list[list[float | None]] = []
    for function_step in candidate_steps:
        source_state = await _load_state(host, function_step.source_state_id)
        row: list[float | None] = []
        for observation in observations:
            if source_state is None:
                row.append(None)
                continue
            try:
                transfer = await _await(
                    transfer_fn(function_step.action, observation, source_state)
                )
            except Exception:  # noqa: BLE001
                row.append(None)
                continue
            probability = _alignment_probability(transfer.detail)
            row.append(
                probability
                if transfer.action is not None
                and probability is not None
                and probability >= _ALIGNMENT_MIN_PROBABILITY
                else None
            )
        probabilities.append(row)

    source_count = len(candidate_steps)
    target_count = len(observations)
    negative_infinity = float("-inf")
    scores = [
        [negative_infinity for _ in range(target_count + 1)]
        for _ in range(source_count + 1)
    ]
    back: list[list[str | None]] = [
        [None for _ in range(target_count + 1)] for _ in range(source_count + 1)
    ]
    scores[0][0] = 0.0
    for target_index in range(1, target_count + 1):
        scores[0][target_index] = scores[0][target_index - 1]
        back[0][target_index] = "target_gap"
    for source_index in range(1, source_count + 1):
        scores[source_index][0] = (
            scores[source_index - 1][0] - _ALIGNMENT_SOURCE_SKIP_PENALTY
        )
        back[source_index][0] = "source_gap"

    operation_priority = {"target_gap": 0, "source_gap": 1, "match": 2}
    for source_index in range(1, source_count + 1):
        for target_index in range(1, target_count + 1):
            options = [
                (scores[source_index][target_index - 1], "target_gap"),
                (
                    scores[source_index - 1][target_index]
                    - _ALIGNMENT_SOURCE_SKIP_PENALTY,
                    "source_gap",
                ),
            ]
            probability = probabilities[source_index - 1][target_index - 1]
            if probability is not None:
                options.append(
                    (
                        scores[source_index - 1][target_index - 1]
                        + _log_odds(probability),
                        "match",
                    )
                )
            score, operation = max(
                options,
                key=lambda item: (item[0], operation_priority[item[1]]),
            )
            scores[source_index][target_index] = score
            back[source_index][target_index] = operation

    candidates = [
        source_index
        for source_index in range(1, source_count + 1)
        if back[source_index][target_count] == "match"
        and scores[source_index][target_count] > 0.0
    ]
    if not candidates:
        return None
    best_source_index = max(
        candidates,
        key=lambda source_index: (scores[source_index][target_count], source_index),
    )
    matched_probability = probabilities[best_source_index - 1][target_count - 1]
    if matched_probability is None:
        return None

    path: list[dict[str, Any]] = []
    source_index = best_source_index
    target_index = target_count
    while source_index > 0 or target_index > 0:
        operation = back[source_index][target_index]
        if operation == "match":
            probability = probabilities[source_index - 1][target_index - 1]
            path.append(
                {
                    "function_step_index": candidate_steps[source_index - 1].step_index,
                    "target_observation_index": target_index - 1,
                    "probability": probability,
                }
            )
            source_index -= 1
            target_index -= 1
        elif operation == "source_gap":
            source_index -= 1
        elif operation == "target_gap":
            target_index -= 1
        else:
            break
    path.reverse()
    resume_step_index = candidate_steps[best_source_index - 1].step_index
    return {
        "protocol": "weighted_lcs_v1",
        "start_step_index": int(start_step_index),
        "resume_step_index": resume_step_index,
        "probability": matched_probability,
        "score": scores[best_source_index][target_count],
        "minimum_probability": _ALIGNMENT_MIN_PROBABILITY,
        "source_skip_penalty": _ALIGNMENT_SOURCE_SKIP_PENALTY,
        "target_observation_count": target_count,
        "path": path,
    }


async def execute_action(
    action: Action,
    *,
    observation: Observation,
    host: Host,
    plugins: PluginSet,
    function: Function | None = None,
    source_state: Observation | None = None,
    installed_packages: frozenset[str] | None = None,
) -> StepResult:
    function_id = function.id if function is not None else None
    executed_steps: list[StepResult] = []
    recovery_action: Action | None = None
    recovery_trigger: str | None = None
    try:
        recovery = match_checker_rule(
            CheckerContext(source_state, observation, action),
            function.checker_rules if function is not None else (),
        )
        if recovery is not None:
            recovery_trigger = recovery.trigger
            recovery_source_state = await _load_state(host, recovery.source_state_id)
            recovery_decision = await prepare_action(
                recovery.action,
                observation=observation,
                plugins=plugins,
                source_state=recovery_source_state,
            )
            if recovery_decision.kind == "block" or recovery_decision.action is None:
                return StepResult(
                    False,
                    action=action,
                    before=observation,
                    error=f"checker_recovery_failed:{recovery_decision.reason or 'blocked'}",
                    origin="blocked",
                    function_id=function_id,
                    detail=recovery_decision.detail,
                )
            recovery_action = recovery_decision.action
    except Exception as error:  # noqa: BLE001
        return StepResult(
            False,
            action=action,
            before=observation,
            error=f"checker_failed:{error}",
            origin="blocked",
            function_id=function_id,
        )
    checker = plugins.checker
    if recovery_action is None and checker is not None:
        try:
            recovery_value = await _await(
                checker(CheckerContext(source_state, observation, action))
            )
            recovery_action = (
                Action.from_value(recovery_value)
                if recovery_value is not None
                else None
            )
            if recovery_action is not None:
                recovery_trigger = default_checker_trigger(
                    CheckerContext(source_state, observation, action),
                    recovery_action,
                )
        except Exception as error:  # noqa: BLE001
            return StepResult(
                False,
                action=action,
                before=observation,
                error=f"checker_failed:{error}",
                origin="blocked",
                function_id=function_id,
            )
    if recovery_action is not None and not _recovery_action_available(
        recovery_action,
        installed_packages,
    ):
        recovery_action = None
        recovery_trigger = None
    if recovery_action is not None:
        recovery_step = replace(
            await _dispatch_prepared(
                recovery_action,
                observation=observation,
                host=host,
                installed_packages=installed_packages,
            ),
            origin="checker",
            function_id=function_id,
            checker_trigger=recovery_trigger,
        )
        executed_steps.append(recovery_step)
        if not recovery_step.success:
            return replace(
                recovery_step,
                executed_steps=tuple(executed_steps),
            )
        observation = recovery_step.after or observation
    decision = await prepare_action(
        action,
        observation=observation,
        plugins=plugins,
        source_state=source_state,
    )
    if decision.kind == "block" or decision.action is None:
        blocked = StepResult(
            False,
            action=action,
            before=observation,
            error=decision.reason or "action_blocked",
            origin="blocked",
            function_id=function_id,
            detail=decision.detail,
        )
        if not executed_steps:
            return blocked
        executed_steps.append(blocked)
        return replace(
            blocked,
            actions_executed=sum(item.actions_executed for item in executed_steps),
            executed_steps=tuple(executed_steps),
        )
    result = await _dispatch_prepared(
        decision.action,
        observation=observation,
        host=host,
        installed_packages=installed_packages,
    )
    result = replace(
        result,
        function_id=function_id,
        detail=decision.detail,
    )
    if not executed_steps:
        return result
    executed_steps.append(result)
    return replace(
        result,
        actions_executed=sum(item.actions_executed for item in executed_steps),
        executed_steps=tuple(executed_steps),
    )


async def prepare_action(
    action: Action,
    *,
    observation: Observation,
    plugins: PluginSet,
    source_state: Observation | None = None,
) -> ActionDecision:
    candidate = action
    if source_state is None or action.tool in {"open_app", "press_key"}:
        return ActionDecision("ready", action=candidate)
    transfer_fn = plugins.transfer
    if transfer_fn is None:
        return ActionDecision("block", reason="transfer_not_configured")
    transfer = await _await(transfer_fn(candidate, observation, source_state))
    if transfer.action is None:
        return ActionDecision(
            "block",
            reason=transfer.reason or "transfer_failed",
            detail=transfer.detail,
        )
    return ActionDecision(
        "ready",
        action=transfer.action,
        reason=transfer.reason,
        detail=transfer.detail,
    )


def _recovery_action_available(
    action: Action,
    installed_packages: frozenset[str] | None,
) -> bool:
    if action.tool != "open_app":
        return True
    package_name = str(action.args.get("package_name") or "").strip()
    return installed_packages is not None and package_name in installed_packages


async def _dispatch_prepared(
    action: Action,
    *,
    observation: Observation,
    host: Host,
    installed_packages: frozenset[str] | None,
) -> StepResult:
    if action.tool == "open_app":
        package_name = str(action.args.get("package_name") or "").strip()
        if installed_packages is None:
            return StepResult(
                False,
                action=action,
                before=observation,
                error="open_app_installed_packages_unavailable",
            )
        if package_name not in installed_packages:
            return StepResult(
                False,
                action=action,
                before=observation,
                error=f"open_app_package_not_installed:{package_name}",
            )
    action_result = ActionResult.from_value(await _await(host.act(action)))
    await asyncio.sleep(_ACTION_SETTLE_SECONDS)
    if not action_result.success:
        return StepResult(
            False,
            action=action,
            before=observation,
            result=action_result,
            error=action_result.error or "action_failed",
        )
    after = Observation.from_value(
        await _await(host.observe(xml=True, screenshot=True, app_info=True))
    )
    if action.tool == "open_app":
        expected_package = str(action.args.get("package_name") or "").strip()
        observed_package = str(after.package_name or "").strip()
        if expected_package and observed_package != expected_package:
            error = (
                "open_app_target_not_ready:"
                f"expected={expected_package}:"
                f"observed={observed_package or 'unknown'}"
            )
            return StepResult(
                False,
                action=action,
                before=observation,
                after=after,
                result=ActionResult(
                    False,
                    error=error,
                    extra={"dispatch_result": action_result.to_dict()},
                ),
                actions_executed=1,
                error=error,
            )
    return StepResult(
        True,
        action=action,
        before=observation,
        after=after,
        result=action_result,
        actions_executed=1,
    )


def step_fact(step: StepResult) -> dict[str, Any]:
    action = step.action or Action("")
    before = _state(step.before or Observation())
    after = _state(step.after or step.before or Observation())
    metadata: dict[str, Any] = {"origin": step.origin}
    if step.function_id:
        metadata["function_id"] = step.function_id
    if step.checker_trigger:
        metadata["checker_trigger"] = step.checker_trigger
    action_result = step.result or ActionResult(step.success, step.error)
    if action_result.extra:
        metadata["action_result"] = dict(action_result.extra)
    if step.detail:
        metadata["transfer"] = dict(step.detail)
    result: dict[str, Any] = {"success": step.success}
    if step.error:
        result["error"] = step.error
    return {
        "before_state_id": before["state_id"],
        "action": action.to_dict(),
        "result": result,
        "after_state_id": after["state_id"],
        "metadata": metadata,
    }


def _state(value: Observation) -> dict[str, Any]:
    state = {
        key: item
        for key, item in value.to_dict().items()
        if key in {"xml", "package_name", "activity_name"} and item not in {None, ""}
    }
    state.update(
        {
            key: item
            for key, item in value.extra.items()
            if key in {"display", "screenshot_path"} and item is not None and item != ""
        }
    )
    explicit_state_id = str(value.extra.get("state_id") or "").strip()
    identity = json.dumps(
        state, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return {
        "state_id": explicit_state_id
        or "state_" + hashlib.sha256(identity.encode()).hexdigest()[:20],
        **state,
    }


async def record_step(
    host: Host,
    fact: dict[str, Any],
    *,
    fallback_step_index: int,
) -> dict[str, Any]:
    recorder = getattr(host, "record_step", None)
    if callable(recorder):
        response = await _await(recorder(fact))
        if isinstance(response, dict) and isinstance(response.get("step"), dict):
            return dict(response["step"])
        raise RuntimeError("record_step_response_invalid")
    return {"step_index": int(fallback_step_index), **fact}


async def record_execution(
    host: Host,
    step: StepResult,
    *,
    trace_start_index: int,
    metadata: dict[str, Any] | None = None,
    first_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    recorded_steps: list[dict[str, Any]] = []
    for offset, executed_step in enumerate(step.executed_steps or (step,)):
        fact = step_fact(executed_step)
        fact["metadata"].update(metadata or {})
        if offset == 0:
            fact["metadata"].update(first_metadata or {})
        recorded_steps.append(
            await record_step(
                host,
                fact,
                fallback_step_index=int(trace_start_index) + offset,
            )
        )
    return recorded_steps


def default_transfer(
    action: Action,
    observation: Observation,
    source_state: Observation | None = None,
) -> TransferResult:
    if action.tool == "input_text" and not all(
        action.args.get(key) is not None for key in ("x", "y")
    ):
        return TransferResult(action)
    if action.tool == "swipe" and all(
        action.args.get(key) is not None for key in ("x1", "y1", "x2", "y2")
    ):
        return _transfer_swipe(action, observation, source_state)
    if action.tool not in {"click", "input_text", "long_press"}:
        return TransferResult(action)
    if not all(action.args.get(key) is not None for key in ("x", "y")):
        return TransferResult(None, reason="omnitransfer_invalid_source_point")
    target_xml = str(observation.xml or "")
    if not target_xml:
        return TransferResult(None, reason="omnitransfer_missing_target_page")
    if str(observation.extra.get("ui_graph_source") or "").endswith("_partial"):
        return TransferResult(None, reason="omnitransfer_target_graph_incomplete")
    elements = _elements(target_xml)
    display_size = _display_size(observation, elements)
    if display_size is None:
        return TransferResult(None, reason=_display_size_error(observation, elements))
    source_xml = str(source_state.xml or "") if source_state is not None else ""
    if not source_xml:
        return TransferResult(None, reason="omnitransfer_source_state_missing")
    request: dict[str, Any] = {
        "source_xml": source_xml,
        "target_xml": target_xml,
        "source_package_name": source_state.package_name,
        "target_package_name": observation.package_name,
        "source_activity_name": source_state.activity_name,
        "target_activity_name": observation.activity_name,
        "action_type": action.tool,
        "top_k": 3,
    }
    try:
        source_point = _relative_source_point(
            source_state,
            float(action.args["x"]),
            float(action.args["y"]),
        )
        request["source_point"] = source_point
    except (KeyError, TypeError, ValueError):
        return TransferResult(None, reason="omnitransfer_invalid_source_point")
    source_title = _source_semantic_title(source_xml, source_point)
    if source_title:
        request["source_element"] = {"text": source_title}
        target_titles = _document_titles(target_xml)
        if source_title not in target_titles:
            return TransferResult(
                None,
                reason="omnitransfer_target_semantic_missing",
                detail={
                    "source_title": source_title,
                    "target_titles": sorted(target_titles),
                },
            )
    try:
        result = transfer_action(**request)
    except Exception as exc:
        return TransferResult(None, reason=f"omnitransfer_error:{exc}")
    if result.get("mapped") is not True:
        reason = result.get("reason") or result.get("mapping_mode") or "failed"
        return TransferResult(
            None,
            reason=f"omnitransfer_{reason}",
            detail=_transfer_detail(result),
        )
    if _is_full_screen_candidate(result.get("target_bbox"), display_size):
        return TransferResult(
            None,
            reason="omnitransfer_invalid_root_candidate",
            detail=_transfer_detail(result),
        )
    semantic_conflict = _semantic_transfer_conflict(
        source_xml=source_xml,
        source_point=source_point,
        target_xml=target_xml,
        target_bbox=result.get("target_bbox"),
    )
    if semantic_conflict is not None:
        return TransferResult(
            None,
            reason="omnitransfer_semantic_conflict",
            detail={**_transfer_detail(result), **semantic_conflict},
        )
    try:
        target_x = float(result["new_x"])
        target_y = float(result["new_y"])
    except (KeyError, TypeError, ValueError):
        return TransferResult(None, reason="omnitransfer_invalid_target")
    if not math.isfinite(target_x) or not math.isfinite(target_y):
        return TransferResult(None, reason="omnitransfer_invalid_target")
    width, height = display_size
    params = dict(action.args)
    params.pop("node_id", None)
    params.pop("node_resource_id", None)
    params["x"] = target_x / width * 1000.0
    params["y"] = target_y / height * 1000.0
    return TransferResult(
        Action(action.tool, params),
        reason=str(result.get("mapping_mode") or "omnitransfer_mapped"),
        detail=_transfer_detail(result),
    )


def _transfer_swipe(
    action: Action,
    observation: Observation,
    source_state: Observation | None,
) -> TransferResult:
    target_xml = str(observation.xml or "")
    if not target_xml:
        return TransferResult(None, reason="omnitransfer_missing_target_page")
    source_xml = str(source_state.xml or "") if source_state is not None else ""
    if not source_xml:
        return TransferResult(None, reason="omnitransfer_source_state_missing")
    elements = _elements(target_xml)
    display_size = _display_size(observation, elements)
    if display_size is None:
        return TransferResult(None, reason=_display_size_error(observation, elements))
    width, height = display_size
    params = dict(action.args)
    mapping_modes: list[str] = []
    endpoint_details: list[dict[str, Any]] = []
    for index, (x_key, y_key) in enumerate((("x1", "y1"), ("x2", "y2"))):
        try:
            source_point = _relative_source_point(
                source_state,
                float(params[x_key]),
                float(params[y_key]),
            )
        except (KeyError, TypeError, ValueError):
            return TransferResult(None, reason="omnitransfer_invalid_source_point")
        try:
            request: dict[str, Any] = {
                "target_xml": target_xml,
                "source_xml": source_xml,
                "source_package_name": source_state.package_name,
                "target_package_name": observation.package_name,
                "source_activity_name": source_state.activity_name,
                "target_activity_name": observation.activity_name,
                "action_type": action.tool,
                "top_k": 3,
            }
            request["source_point"] = source_point
            result = transfer_action(
                **request,
            )
        except Exception as exc:
            return TransferResult(None, reason=f"omnitransfer_error:{exc}")
        if result.get("mapped") is not True:
            reason = result.get("reason") or result.get("mapping_mode") or "failed"
            return TransferResult(
                None,
                reason=f"omnitransfer_{reason}",
                detail=_transfer_detail(result),
            )
        try:
            target_x = float(result["new_x"])
            target_y = float(result["new_y"])
        except (KeyError, TypeError, ValueError):
            return TransferResult(None, reason="omnitransfer_invalid_target")
        if not math.isfinite(target_x) or not math.isfinite(target_y):
            return TransferResult(None, reason="omnitransfer_invalid_target")
        params[x_key] = target_x / width * 1000.0
        params[y_key] = target_y / height * 1000.0
        mapping_modes.append(str(result.get("mapping_mode") or "omnitransfer_mapped"))
        endpoint_details.append(_transfer_detail(result))
    reason = mapping_modes[0] if len(set(mapping_modes)) == 1 else "omnitransfer_mapped"
    detail: dict[str, Any] = {
        "mapping_mode": reason,
        "endpoints": endpoint_details,
    }
    probabilities = [
        probability
        for probability in (
            _alignment_probability(endpoint) for endpoint in endpoint_details
        )
        if probability is not None
    ]
    if probabilities:
        detail["score"] = min(probabilities)
    margins = [
        float(endpoint["margin"])
        for endpoint in endpoint_details
        if isinstance(endpoint.get("margin"), (int, float))
    ]
    if margins:
        detail["margin"] = min(margins)
    return TransferResult(
        Action(action.tool, params),
        reason=reason,
        detail=detail,
    )


def _transfer_detail(result: dict[str, Any]) -> dict[str, Any]:
    source = _element_detail(result.get("src_element"))
    source_display = _display_detail(result.get("source_size"))
    if source_display:
        source["display"] = source_display
    target: dict[str, Any] = {}
    target_display = _display_detail(result.get("target_size"))
    if target_display:
        target["display"] = target_display
    candidates = []
    for rank, raw in enumerate(result.get("top_candidates") or (), start=1):
        if not isinstance(raw, dict):
            continue
        candidate = _element_detail(raw)
        candidate["rank"] = rank
        candidate["bounds"] = list(raw.get("bbox") or ())
        candidate["score"] = raw.get("score")
        candidates.append(candidate)
    detail = {
        "mapping_mode": str(result.get("mapping_mode") or ""),
        "source": source,
        "target": target,
        "candidates": candidates,
    }
    if result.get("mapped") is True:
        for key in ("score", "margin"):
            try:
                detail[key] = float(result[key])
            except (KeyError, TypeError, ValueError):
                continue
        if "score" not in detail and candidates:
            try:
                detail["score"] = float(candidates[0]["score"])
            except (KeyError, TypeError, ValueError):
                pass
    return detail


def _alignment_probability(detail: dict[str, Any]) -> float | None:
    raw = detail.get("score")
    if raw is None:
        candidates = detail.get("candidates")
        if isinstance(candidates, list) and candidates:
            candidate = candidates[0]
            if isinstance(candidate, dict):
                raw = candidate.get("score")
    try:
        probability = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(probability):
        return None
    return min(1.0, max(0.0, probability))


def _log_odds(probability: float) -> float:
    bounded = min(1.0 - 1e-9, max(1e-9, probability))
    return math.log(bounded / (1.0 - bounded))


def _element_detail(value: Any) -> dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    return {
        key: raw[key]
        for key in (
            "resource_id",
            "text",
            "content_desc",
            "class",
            "bounds",
        )
        if raw.get(key) not in (None, "", [])
    }


def _display_detail(value: Any) -> dict[str, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return {}
    try:
        width, height = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return {}
    if width <= 0 or height <= 0:
        return {}
    return {"width": width, "height": height}


async def _load_state(host: Host, source_state_id: str | None) -> Observation | None:
    if not source_state_id:
        return None
    loader = getattr(host, "get_state", None)
    if not callable(loader):
        return None
    return Observation.from_value(await _await(loader(source_state_id)))


def _relative_source_point(
    source_state: Observation,
    x: float,
    y: float,
) -> tuple[float, float]:
    display = _display_size(source_state, _elements(str(source_state.xml or "")))
    if display is None:
        raise ValueError("source_display_size_missing")
    return x / 1000.0 * display[0], y / 1000.0 * display[1]


async def _await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _elements(xml_text: str) -> list[dict[str, Any]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    elements: list[dict[str, Any]] = []
    for element in root.iter():
        bounds = _bounds(element.attrib.get("bounds"))
        if bounds is None:
            continue
        elements.append(
            {
                "node_id": str(element.attrib.get("id") or ""),
                "bounds": bounds,
                "resource_id": str(element.attrib.get("resource-id") or "").rsplit(
                    "/", 1
                )[-1],
                "text": _text(element.attrib.get("text")),
                "description": _text(element.attrib.get("content-desc")),
                "class": str(element.attrib.get("class") or element.tag).rsplit(".", 1)[
                    -1
                ],
                "clickable": str(element.attrib.get("clickable") or "").lower()
                == "true",
            }
        )
    return elements


def _display_size(
    observation: Observation,
    elements: list[dict[str, Any]],
) -> tuple[float, float] | None:
    xml_size = _xml_size(elements)
    display = observation.extra.get("display")
    action_size = None
    if isinstance(display, dict) and set(display) == {"width", "height"}:
        try:
            width = float(display.get("width") or 0)
            height = float(display.get("height") or 0)
        except (TypeError, ValueError):
            width = height = 0.0
        if width > 0 and height > 0:
            action_size = (width, height)
    if xml_size is None:
        return action_size
    if action_size is None:
        return xml_size
    return max(xml_size[0], action_size[0]), max(xml_size[1], action_size[1])


def _xml_size(elements: list[dict[str, Any]]) -> tuple[float, float] | None:
    if not elements:
        return None
    width = max(float(element["bounds"][2]) for element in elements)
    height = max(float(element["bounds"][3]) for element in elements)
    return (width, height) if width > 0 and height > 0 else None


def _display_size_error(
    observation: Observation,
    elements: list[dict[str, Any]],
) -> str:
    return "omnitransfer_display_size_missing"


def _bounds(value: Any) -> tuple[int, int, int, int] | None:
    numbers = [int(item) for item in re.findall(r"-?\d+", str(value or ""))]
    if len(numbers) != 4 or numbers[2] <= numbers[0] or numbers[3] <= numbers[1]:
        return None
    return numbers[0], numbers[1], numbers[2], numbers[3]


def _semantic_transfer_conflict(
    *,
    source_xml: str,
    source_point: tuple[float, float],
    target_xml: str,
    target_bbox: Any,
) -> dict[str, Any] | None:
    source_root = _xml_root(source_xml)
    target_root = _xml_root(target_xml)
    mapped_bounds = _numeric_bounds(target_bbox)
    if source_root is None or target_root is None or mapped_bounds is None:
        return None
    source_title = _source_semantic_title(source_xml, source_point)
    target = _element_with_bounds(target_root, mapped_bounds)
    target_title = _element_title(target) if target is not None else ""
    available_titles = _document_titles(target_xml)
    if (
        not source_title
        or not target_title
        or source_title == target_title
        or source_title not in available_titles
    ):
        return None
    return {
        "source_title": source_title,
        "target_title": target_title,
        "target_bbox": list(mapped_bounds),
    }


def _source_semantic_title(
    source_xml: str,
    source_point: tuple[float, float],
) -> str:
    root = _xml_root(source_xml)
    if root is None:
        return ""
    source = _actionable_element_at_point(root, source_point)
    if source is None or _element_has_stable_identity(source):
        return ""
    title = _element_title(source)
    if title:
        return title
    return next(
        (
            label
            for descendant in source.iter()
            if (
                label := _text(
                    descendant.attrib.get("text")
                    or descendant.attrib.get("content-desc")
                )
            )
        ),
        "",
    )


def _document_titles(xml_text: str) -> set[str]:
    root = _xml_root(xml_text)
    if root is None:
        return set()
    return {title for element in root.iter() if (title := _element_title(element))}


def _xml_root(xml_text: str) -> ET.Element | None:
    try:
        return ET.fromstring(xml_text)
    except ET.ParseError:
        return None


def _actionable_element_at_point(
    root: ET.Element,
    point: tuple[float, float],
) -> ET.Element | None:
    x, y = point
    containing: list[tuple[ET.Element, tuple[int, int, int, int], int]] = []

    def visit(element: ET.Element, depth: int) -> None:
        bounds = _bounds(element.attrib.get("bounds"))
        if (
            bounds is not None
            and bounds[0] <= x <= bounds[2]
            and bounds[1] <= y <= bounds[3]
        ):
            containing.append((element, bounds, depth))
        for child in list(element):
            visit(child, depth + 1)

    visit(root, 0)
    actionable = [
        item
        for item in containing
        if str(item[0].attrib.get("enabled") or "true").lower() != "false"
        and any(
            str(item[0].attrib.get(attribute) or "").lower() == "true"
            for attribute in ("clickable", "editable", "scrollable")
        )
    ]
    candidates = actionable or containing
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            (item[1][2] - item[1][0]) * (item[1][3] - item[1][1]),
            not _element_has_stable_identity(item[0]),
            -item[2],
        ),
    )[0]


def _element_with_bounds(
    root: ET.Element,
    expected: tuple[float, float, float, float],
) -> ET.Element | None:
    matches = [
        element
        for element in root.iter()
        if (_bounds(element.attrib.get("bounds")) is not None)
        and tuple(float(value) for value in _bounds(element.attrib.get("bounds")) or ())
        == expected
    ]
    actionable = [
        element
        for element in matches
        if any(
            str(element.attrib.get(attribute) or "").lower() == "true"
            for attribute in ("clickable", "editable", "scrollable")
        )
    ]
    return (actionable or matches or [None])[0]


def _element_has_stable_identity(element: ET.Element) -> bool:
    return any(
        str(element.attrib.get(attribute) or "").strip()
        for attribute in ("resource-id", "text", "content-desc")
    )


def _element_title(element: ET.Element | None) -> str:
    if element is None:
        return ""
    titles = {
        _text(descendant.attrib.get("text") or descendant.attrib.get("content-desc"))
        for descendant in element.iter()
        if str(descendant.attrib.get("resource-id") or "").rsplit("/", 1)[-1] == "title"
        and _text(
            descendant.attrib.get("text") or descendant.attrib.get("content-desc")
        )
    }
    return next(iter(titles)) if len(titles) == 1 else ""


def _numeric_bounds(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        bounds = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return bounds if bounds[2] > bounds[0] and bounds[3] > bounds[1] else None


def _is_full_screen_candidate(
    value: Any,
    display_size: tuple[float, float],
) -> bool:
    bounds = _numeric_bounds(value)
    if bounds is None:
        return False
    width, height = display_size
    return (
        bounds[0] <= width * 0.01
        and bounds[1] <= height * 0.01
        and bounds[2] >= width * 0.95
        and bounds[3] >= height * 0.95
    )


def _text(value: Any) -> str:
    return " ".join(str(value or "").lower().split())
