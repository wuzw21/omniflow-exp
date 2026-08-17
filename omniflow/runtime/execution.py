from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import inspect
import json
import math
import re
from typing import Any, Callable
import unicodedata
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
from omniflow.runtime.core import (
    execute_action as execute_core_action,
)
from omniflow.runtime.core import (
    prepare_action as prepare_core_action,
)
from omniflow.runtime.semantic_grounding import resolve_semantic_action
from omniflow.transfer.runtime import transfer_action

_OPEN_APP_READY_POLL_SECONDS = 0.5
_OPEN_APP_READY_MAX_ATTEMPTS = 30
_OBSERVATION_READY_POLL_SECONDS = 0.25
_OBSERVATION_READY_MAX_ATTEMPTS = 20
_CHECKER_RECOVERY_MAX_ATTEMPTS = 8
# OmniTransfer already applies the deployment acceptance floor.  Keep only a
# minimal sanity floor here so OmniFlow does not reject a valid mapped target a
# second time merely because its confidence is below a conservative benchmark
# threshold.
_ALIGNMENT_MIN_PROBABILITY = 0.0
StateLoader = Callable[[str], Any]


async def execute_function(
    function: Function,
    *,
    host: Host,
    plugins: PluginSet,
    observation: Observation | None = None,
    start_step_index: int = 0,
    trace_start_index: int = 0,
    resume_metadata: dict[str, Any] | None = None,
    installed_packages: frozenset[str] | None = None,
    state_loader: StateLoader | None = None,
) -> RunResult:
    current = observation or Observation.from_value(
        await _await(host.observe(xml=True, app_info=True))
    )
    steps = tuple(
        step for step in function.steps if step.step_index >= int(start_step_index)
    )
    executed = 0
    trace: list[dict[str, Any]] = []
    checker_decisions: list[dict[str, Any]] = []
    resume_metadata_pending = dict(resume_metadata or {})
    for function_step in steps:
        action = function_step.action
        source_state = await _load_state(
            host,
            function_step.source_state_id,
            state_loader=state_loader,
        )
        if function_step.role == "checker":
            step, checker_decision = await execute_checker_step(
                action,
                observation=current,
                host=host,
                plugins=plugins,
                function=function,
                source_state=source_state,
                installed_packages=installed_packages,
            )
            checker_decisions.append(
                {
                    "function_step_index": function_step.step_index,
                    **checker_decision,
                }
            )
            if step.actions_executed == 0 and step.success:
                continue
        else:
            step = await execute_robust_action(
                action,
                observation=current,
                host=host,
                plugins=plugins,
                function=function,
                source_state=source_state,
                installed_packages=installed_packages,
                state_loader=state_loader,
            )
        executed += step.actions_executed
        trace.extend(
            await record_execution(
                host,
                step,
                trace_start_index=int(trace_start_index) + len(trace),
                metadata={
                    "function_step_index": function_step.step_index,
                    "function_step_role": function_step.role,
                },
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
                    "checker_decisions": checker_decisions,
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
            "checker_decisions": checker_decisions,
            "next_step_index": (
                max((step.step_index for step in steps), default=start_step_index - 1)
                + 1
            ),
        },
    )


async def execute_checker_step(
    action: Action,
    *,
    observation: Observation,
    host: Host,
    plugins: PluginSet,
    function: Function,
    source_state: Observation | None,
    installed_packages: frozenset[str] | None = None,
) -> tuple[StepResult, dict[str, Any]]:
    """Execute one optional checker Step only when its source target is present."""

    decision = await prepare_action(
        action,
        observation=observation,
        plugins=plugins,
        source_state=source_state,
    )
    if decision.kind == "block" or decision.action is None:
        return (
            StepResult(
                True,
                action=action,
                before=observation,
                after=observation,
                origin="checker",
                function_id=function.id,
                detail=dict(decision.detail or {}),
            ),
            {
                "status": "skipped",
                "reason": decision.reason or "transfer_not_applicable",
                "transfer": dict(decision.detail or {}),
            },
        )
    if not _checker_transfer_applicable(decision.detail):
        return (
            StepResult(
                True,
                action=action,
                before=observation,
                after=observation,
                origin="checker",
                function_id=function.id,
                detail=dict(decision.detail or {}),
            ),
            {
                "status": "skipped",
                "reason": "source_target_not_present",
                "transfer": dict(decision.detail or {}),
            },
        )
    step = replace(
        await _dispatch_prepared(
            decision.action,
            observation=observation,
            host=host,
            installed_packages=installed_packages,
        ),
        origin="checker",
        function_id=function.id,
        detail=dict(decision.detail or {}),
    )
    return (
        step,
        {
            "status": "executed" if step.success else "failed",
            "reason": decision.reason or "omnitransfer_mapped",
            "transfer": dict(decision.detail or {}),
        },
    )


def _checker_transfer_applicable(detail: dict[str, Any] | None) -> bool:
    """Use mapped target evidence instead of a handwritten page-trigger DSL."""

    evidence = detail if isinstance(detail, dict) else {}
    source = evidence.get("source")
    candidates = evidence.get("candidates")
    if not isinstance(source, dict) or not isinstance(candidates, list) or not candidates:
        return False
    target = candidates[0]
    if not isinstance(target, dict):
        return False
    source_resource_id = _normalized_semantic_value(source.get("resource_id"))
    if source_resource_id:
        return source_resource_id == _normalized_semantic_value(
            target.get("resource_id")
        )
    for field in ("text", "content_desc"):
        source_value = _normalized_semantic_value(source.get(field))
        if source_value:
            return source_value == _normalized_semantic_value(target.get(field))
    try:
        score = float(evidence.get("score"))
        margin = float(evidence.get("margin"))
    except (TypeError, ValueError):
        return False
    return score >= 0.75 and margin >= 0.50


def _normalized_semantic_value(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


async def execute_robust_action(
    action: Action,
    *,
    observation: Observation,
    host: Host,
    plugins: PluginSet,
    function: Function | None = None,
    source_state: Observation | None = None,
    installed_packages: frozenset[str] | None = None,
    state_loader: StateLoader | None = None,
    _checker_recovery_attempts_remaining: int = _CHECKER_RECOVERY_MAX_ATTEMPTS,
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
            recovery_source_state = await _load_state(
                host,
                recovery.source_state_id,
                state_loader=state_loader,
            )
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
        if _checker_recovery_attempts_remaining <= 0:
            return StepResult(
                False,
                action=action,
                before=observation,
                error="checker_recovery_limit_exceeded",
                origin="blocked",
                function_id=function_id,
            )
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
        retried = await execute_robust_action(
            action,
            observation=observation,
            host=host,
            plugins=plugins,
            function=function,
            source_state=source_state,
            installed_packages=installed_packages,
            state_loader=state_loader,
            _checker_recovery_attempts_remaining=(
                _checker_recovery_attempts_remaining - 1
            ),
        )
        retried_steps = tuple(retried.executed_steps or (retried,))
        all_steps = (recovery_step, *retried_steps)
        return replace(
            retried,
            actions_executed=sum(item.actions_executed for item in all_steps),
            executed_steps=all_steps,
        )
    if (
        _action_uses_transfer_target(action)
        and _observation_screenshot_path(source_state)
        and not _observation_screenshot_path(observation)
    ):
        observation = await _observe_ready(host)
    semantic = resolve_semantic_action(action, observation)
    action = semantic.action
    semantic_detail = semantic.detail
    transfer_source_state = source_state
    if semantic_detail is not None and semantic_detail.get("status") == "resolved":
        transfer_source_state = None
    decision = await prepare_action(
        action,
        observation=observation,
        plugins=plugins,
        source_state=transfer_source_state,
    )
    if decision.kind == "block" or decision.action is None:
        blocked = StepResult(
            False,
            action=action,
            before=observation,
            error=decision.reason or "action_blocked",
            origin="blocked",
            function_id=function_id,
            detail=_merge_action_detail(decision.detail, semantic_detail),
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
        detail=_merge_action_detail(decision.detail, semantic_detail),
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
    return await prepare_core_action(
        action,
        observation=observation,
        plugins=plugins,
        source_state=source_state,
    )


def _merge_action_detail(
    decision_detail: dict[str, Any] | None,
    semantic_detail: dict[str, object] | None,
) -> dict[str, Any] | None:
    if decision_detail is None and semantic_detail is None:
        return None
    detail = dict(decision_detail or {})
    if semantic_detail is not None:
        detail["semantic_grounding"] = dict(semantic_detail)
    return detail


def _recovery_action_available(
    action: Action,
    installed_packages: frozenset[str] | None,
) -> bool:
    if action.tool != "open_app":
        return True
    package_name = str(action.args.get("package_name") or "").strip()
    return bool(package_name)


async def _dispatch_prepared(
    action: Action,
    *,
    observation: Observation,
    host: Host,
    installed_packages: frozenset[str] | None,
) -> StepResult:
    core_step = await execute_core_action(
        action,
        observation=observation,
        host=host,
        plugins=PluginSet(),
    )
    if not core_step.success:
        return replace(core_step, origin="action")
    after = core_step.after or observation
    if _observation_window_outside_display(after):
        after = await _observe_ready(host)
    if action.tool == "open_app":
        expected_package = str(action.args.get("package_name") or "").strip()
        observed_package = str(after.package_name or "").strip()
        attempts = 1
        while (
            expected_package
            and observed_package != expected_package
            and attempts < _OPEN_APP_READY_MAX_ATTEMPTS
        ):
            await asyncio.sleep(_OPEN_APP_READY_POLL_SECONDS)
            after = await _observe_ready(host)
            observed_package = str(after.package_name or "").strip()
            attempts += 1
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
                    extra={
                        "dispatch_result": (
                            core_step.result.to_dict()
                            if core_step.result is not None
                            else {}
                        )
                    },
                ),
                actions_executed=1,
                error=error,
            )
    return replace(core_step, after=after, origin="action")


async def _observe_ready(host: Host) -> Observation:
    after = Observation()
    for attempt in range(_OBSERVATION_READY_MAX_ATTEMPTS):
        after = Observation.from_value(
            await _await(host.observe(xml=True, screenshot=True, app_info=True))
        )
        if not _observation_window_outside_display(after):
            return after
        if attempt + 1 < _OBSERVATION_READY_MAX_ATTEMPTS:
            await asyncio.sleep(_OBSERVATION_READY_POLL_SECONDS)
    return after


def _observation_window_outside_display(observation: Observation) -> bool:
    display = observation.extra.get("display")
    if not isinstance(display, dict):
        return False
    try:
        width = float(display.get("width") or 0)
        height = float(display.get("height") or 0)
    except (TypeError, ValueError):
        return False
    if width <= 0 or height <= 0:
        return False
    try:
        root = ET.fromstring(str(observation.xml or ""))
    except ET.ParseError:
        return False
    bounds = next(
        (
            parsed
            for element in root.iter()
            if (parsed := _bounds(element.attrib.get("bounds"))) is not None
        ),
        None,
    )
    if bounds is None:
        return False
    tolerance_x = max(2.0, width * 0.01)
    tolerance_y = max(2.0, height * 0.01)
    return (
        bounds[0] < -tolerance_x
        or bounds[1] < -tolerance_y
        or bounds[2] > width + tolerance_x
        or bounds[3] > height + tolerance_y
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
    _attach_visual_evidence(request, source_state, observation)
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
    try:
        result = transfer_action(**request)
    except Exception as exc:
        return TransferResult(None, reason=f"omnitransfer_error:{exc}")
    if result.get("mapped") is not True:
        reason = result.get("reason") or result.get("mapping_mode") or "failed"
        detail = _transfer_detail(result)
        if "low_confidence" in str(reason):
            return _recoverable_transfer_failure(
                "omnitransfer_low_confidence",
                detail,
            )
        return TransferResult(
            None,
            reason=f"omnitransfer_{reason}",
            detail=detail,
        )
    if _is_full_screen_candidate(result.get("target_bbox"), display_size):
        return TransferResult(
            None,
            reason="omnitransfer_invalid_root_candidate",
            detail=_transfer_detail(result),
        )
    transfer_detail = _transfer_detail(result)
    probability = _alignment_probability(transfer_detail)
    if probability is not None and probability < _ALIGNMENT_MIN_PROBABILITY:
        return _recoverable_transfer_failure(
            "omnitransfer_low_confidence",
            transfer_detail,
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
            _attach_visual_evidence(request, source_state, observation)
            request["source_point"] = source_point
            result = transfer_action(
                **request,
            )
        except Exception as exc:
            return TransferResult(None, reason=f"omnitransfer_error:{exc}")
        if result.get("mapped") is not True:
            reason = result.get("reason") or result.get("mapping_mode") or "failed"
            detail = _transfer_detail(result)
            if "low_confidence" in str(reason):
                return _recoverable_transfer_failure(
                    "omnitransfer_low_confidence",
                    detail,
                )
            return TransferResult(
                None,
                reason=f"omnitransfer_{reason}",
                detail=detail,
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
        endpoint_probability = _alignment_probability(endpoint_details[-1])
        if (
            endpoint_probability is not None
            and endpoint_probability < _ALIGNMENT_MIN_PROBABILITY
        ):
            return _recoverable_transfer_failure(
                "omnitransfer_low_confidence",
                {
                    "mapping_mode": str(
                        result.get("mapping_mode") or "omnitransfer_mapped"
                    ),
                    "endpoints": endpoint_details,
                    "score": endpoint_probability,
                },
            )
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


def _observation_screenshot_path(observation: Observation | None) -> str:
    if observation is None:
        return ""
    return str(observation.extra.get("screenshot_path") or "").strip()


def _attach_visual_evidence(
    request: dict[str, Any],
    source: Observation,
    target: Observation,
) -> None:
    evidence = {
        "source_screenshot_path": _observation_screenshot_path(source),
        "target_screenshot_path": _observation_screenshot_path(target),
        "source_visual_rgb": source.extra.get("visual_rgb"),
        "target_visual_rgb": target.extra.get("visual_rgb"),
    }
    request.update(
        {
            key: value
            for key, value in evidence.items()
            if value and (not key.endswith("visual_rgb") or isinstance(value, dict))
        }
    )


def _action_uses_transfer_target(action: Action) -> bool:
    if action.tool in {"click", "input_text", "long_press"}:
        return all(action.args.get(key) is not None for key in ("x", "y"))
    if action.tool == "swipe":
        return all(
            action.args.get(key) is not None
            for key in ("x1", "y1", "x2", "y2")
        )
    return False


def _recoverable_transfer_failure(
    reason: str,
    detail: dict[str, Any],
) -> TransferResult:
    """Make low-confidence transfer recoverable by the normal online VLM loop."""

    return TransferResult(
        None,
        reason=reason,
        detail={
            **dict(detail),
            "recoverable": True,
            "fallback": "online_vlm",
            "continue": True,
        },
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


async def _load_state(
    host: Host,
    source_state_id: str | None,
    *,
    state_loader: StateLoader | None = None,
) -> Observation | None:
    if not source_state_id:
        return None
    host_loader = getattr(host, "get_state", None)
    if callable(host_loader):
        try:
            loaded = Observation.from_value(
                await _await(host_loader(source_state_id))
            )
        except Exception:  # noqa: BLE001
            loaded = None
        if loaded is not None and (loaded.package_name or loaded.xml):
            return loaded
    if state_loader is not None:
        loaded = Observation.from_value(await _await(state_loader(source_state_id)))
        if loaded.package_name or loaded.xml:
            return loaded
    # A referenced-but-missing source state must remain a transfer failure.  It
    # must never become ``None`` because ``None`` means that replay can safely
    # skip transfer (only valid for actions recorded without a source state).
    return Observation(extra={"state_id": source_state_id})


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
                "package": str(element.attrib.get("package") or ""),
                "clickable": str(element.attrib.get("clickable") or "").lower()
                == "true",
            }
        )
    return elements


def _display_size(
    observation: Observation,
    elements: list[dict[str, Any]],
) -> tuple[float, float] | None:
    display = observation.extra.get("display")
    if isinstance(display, dict) and set(display) == {"width", "height"}:
        try:
            width = float(display.get("width") or 0)
            height = float(display.get("height") or 0)
        except (TypeError, ValueError):
            width = height = 0.0
        if width > 0 and height > 0:
            return width, height
    return _xml_size(elements)


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


def _source_semantic_title(
    source_xml: str,
    source_point: tuple[float, float],
) -> str:
    root = _xml_root(source_xml)
    if root is None:
        return ""
    source = _actionable_element_at_point(root, source_point)
    if source is None:
        return ""
    direct_label = _text(
        source.attrib.get("text") or source.attrib.get("content-desc")
    )
    if _is_semantic_label(direct_label):
        return direct_label
    title = _element_title(source)
    if _is_semantic_label(title):
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
            if _is_semantic_label(label)
        ),
        "",
    )


def _is_semantic_label(value: str) -> bool:
    normalized = _text(value)
    return bool(normalized) and any(
        unicodedata.category(character) != "Co"
        for character in normalized
        if not character.isspace()
    )


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
