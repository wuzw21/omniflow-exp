from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import inspect
import json
import math
import re
from typing import Any, Callable
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
from omniflow.runtime.checker import (
    checker_rule_action,
    checker_rule_matches,
    default_checker_trigger,
)
from omniflow.runtime.core import (
    execute_action as execute_core_action,
)
from omniflow.runtime.core import (
    prepare_action as prepare_core_action,
)
from omniflow.transfer.admission import assess_transfer
from omniflow.transfer.runtime import (
    transfer_action,
)

_OPEN_APP_READY_POLL_SECONDS = 0.5
_OPEN_APP_READY_MAX_ATTEMPTS = 30
_OBSERVATION_READY_POLL_SECONDS = 0.25
_OBSERVATION_READY_MAX_ATTEMPTS = 20
_CHECKER_RECOVERY_MAX_ATTEMPTS = 8
_ACTION_EFFECT_MAX_ITEMS = 8
_ACTION_EFFECT_VALUE_MAX_CHARS = 160
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
    checker_rules: tuple[dict[str, Any], ...] = (),
    checker_trigger_counts: dict[str, int] | None = None,
) -> RunResult:
    current = observation or Observation.from_value(
        await _await(host.observe(xml=True, app_info=True))
    )
    steps = tuple(
        step for step in function.steps if step.step_index >= int(start_step_index)
    )
    executed = 0
    trace: list[dict[str, Any]] = []
    checker_trigger_counts = (
        checker_trigger_counts if checker_trigger_counts is not None else {}
    )
    resume_metadata_pending = dict(resume_metadata or {})
    for step_offset, function_step in enumerate(steps):
        action = function_step.action
        source_state = await _load_state(
            host,
            function_step.source_state_id,
            state_loader=state_loader,
        )
        for checker_phase in ("pre_transfer", "pre_action"):
            checker_steps = await _run_shared_checker_phase(
                checker_phase,
                rules=checker_rules,
                trigger_counts=checker_trigger_counts,
                function=function,
                function_step_index=function_step.step_index,
                action=action,
                observation=current,
                source_state=source_state,
                host=host,
                installed_packages=installed_packages,
            )
            for checker_step in checker_steps:
                executed += checker_step.actions_executed
                trace.extend(
                    await record_execution(
                        host,
                        checker_step,
                        trace_start_index=int(trace_start_index) + len(trace),
                        metadata={"function_step_index": function_step.step_index},
                    )
                )
                current = checker_step.after or checker_step.before or current
                if not checker_step.success:
                    return RunResult(
                        False,
                        function.id,
                        executed,
                        error=checker_step.error,
                        final_state=current,
                        detail={
                            "trace": trace,
                            "failed_step_index": function_step.step_index,
                            "next_step_index": function_step.step_index,
                        },
                    )
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

        after_observation = step.after or step.before or current
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
        current = after_observation
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
        post_checker_steps = await _run_shared_checker_phase(
            "post_action",
            rules=checker_rules,
            trigger_counts=checker_trigger_counts,
            function=function,
            function_step_index=function_step.step_index,
            action=action,
            observation=current,
            source_state=source_state,
            host=host,
            installed_packages=installed_packages,
        )
        for checker_step in post_checker_steps:
            executed += checker_step.actions_executed
            trace.extend(
                await record_execution(
                    host,
                    checker_step,
                    trace_start_index=int(trace_start_index) + len(trace),
                    metadata={"function_step_index": function_step.step_index},
                )
            )
            current = checker_step.after or checker_step.before or current
            if not checker_step.success:
                return RunResult(
                    False,
                    function.id,
                    executed,
                    error=checker_step.error,
                    final_state=current,
                    detail={
                        "trace": trace,
                        "failed_step_index": function_step.step_index,
                        "next_step_index": function_step.step_index + 1,
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
    decision = await prepare_action(
        action,
        observation=observation,
        host=host,
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


async def _run_shared_checker_phase(
    phase: str,
    *,
    rules: tuple[dict[str, Any], ...],
    trigger_counts: dict[str, int],
    function: Function,
    function_step_index: int,
    action: Action,
    observation: Observation,
    source_state: Observation | None,
    host: Host,
    installed_packages: frozenset[str] | None,
    transfer_failed: bool = False,
) -> list[StepResult]:
    effects: list[StepResult] = []
    current = observation
    step_counts: dict[str, int] = {}
    ordered = sorted(
        (rule for rule in rules if str(rule.get("phase") or "pre_transfer") == phase),
        key=lambda rule: (-int(rule.get("priority") or 0), str(rule.get("id") or "")),
    )
    for _ in range(_CHECKER_RECOVERY_MAX_ATTEMPTS):
        selected: tuple[dict[str, Any], Action] | None = None
        for rule in ordered:
            rule_id = str(rule.get("id") or "")
            budget = rule.get("budget") if isinstance(rule.get("budget"), dict) else {}
            run_limit = int(budget.get("max_triggers_per_run", 1))
            step_limit = int(budget.get("max_triggers_per_step", run_limit))
            if trigger_counts.get(rule_id, 0) >= run_limit:
                continue
            if step_counts.get(rule_id, 0) >= step_limit:
                continue
            if not checker_rule_matches(
                rule,
                current=current,
                source=source_state,
                function_id=function.id,
                step_index=function_step_index,
                action=action,
                transfer_failed=transfer_failed,
            ):
                continue
            recovery_action = checker_rule_action(
                rule,
                current=current,
                source=source_state,
            )
            if recovery_action is not None and _recovery_action_available(
                recovery_action, installed_packages
            ):
                selected = rule, recovery_action
                break
        if selected is None:
            break
        rule, recovery_action = selected
        rule_id = str(rule["id"])
        checker_step = replace(
            await _dispatch_prepared(
                recovery_action,
                observation=current,
                host=host,
                installed_packages=installed_packages,
            ),
            origin="checker",
            function_id=function.id,
            checker_trigger=rule_id,
            detail={"checker_id": rule_id, "checker_phase": phase},
        )
        effects.append(checker_step)
        trigger_counts[rule_id] = trigger_counts.get(rule_id, 0) + 1
        step_counts[rule_id] = step_counts.get(rule_id, 0) + 1
        current = checker_step.after or current
        if not checker_step.success:
            break
    return effects


async def prepare_action(
    action: Action,
    *,
    observation: Observation,
    host: Host | None = None,
    plugins: PluginSet,
    source_state: Observation | None = None,
) -> ActionDecision:
    return await prepare_core_action(
        action,
        observation=observation,
        host=host,
        plugins=plugins,
        source_state=source_state,
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
        from src.integrations.android_world.apps import (
            canonicalize_androidworld_package,
        )

        package_name = canonicalize_androidworld_package(
            str(action.args.get("package_name") or "").strip()
        )
        if package_name != str(action.args.get("package_name") or "").strip():
            action = replace(
                action,
                args={**action.args, "package_name": package_name},
            )
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
    before_observation = step.before or Observation()
    after_observation = step.after or step.before or Observation()
    before = _state(before_observation)
    after = _state(after_observation)
    metadata: dict[str, Any] = {"origin": step.origin}
    if step.function_id:
        metadata["function_id"] = step.function_id
    if step.checker_trigger:
        metadata["checker_trigger"] = step.checker_trigger
        metadata["checker"] = {
            "id": step.checker_trigger,
            "phase": str(step.detail.get("checker_phase") or "")
            if isinstance(step.detail, dict)
            else "",
        }
    action_result = step.result or ActionResult(step.success, step.error)
    if action_result.extra:
        metadata["action_result"] = dict(action_result.extra)
    if step.detail and not step.checker_trigger:
        metadata["transfer"] = dict(step.detail)
    metadata["action_effect"] = _action_effect(
        before_observation,
        after_observation,
    )
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


def _action_effect(
    before: Observation,
    after: Observation,
) -> dict[str, Any]:
    """Describe the observed post-action change without another model call."""

    before_state = _state(before)
    after_state = _state(after)
    effect: dict[str, Any] = {
        "state_changed": before_state["state_id"] != after_state["state_id"],
    }
    if before.package_name != after.package_name:
        effect["package"] = {
            "before": str(before.package_name or ""),
            "after": str(after.package_name or ""),
        }
    if before.activity_name != after.activity_name:
        effect["activity"] = {
            "before": str(before.activity_name or ""),
            "after": str(after.activity_name or ""),
        }

    target_package = str(after.package_name or before.package_name or "").strip()
    if not target_package:
        target_package = _dominant_effect_package(after.xml or before.xml)
    before_nodes = _effect_nodes(before.xml, target_package=target_package)
    after_nodes = _effect_nodes(after.xml, target_package=target_package)
    changed: list[dict[str, str]] = []
    changed_before_labels: set[str] = set()
    changed_after_labels: set[str] = set()
    for identity in sorted(before_nodes.keys() & after_nodes.keys()):
        before_fields = before_nodes[identity]
        after_fields = after_nodes[identity]
        for field in ("text", "content_desc", "checked", "selected", "enabled"):
            old = before_fields.get(field, "")
            new = after_fields.get(field, "")
            if old == new:
                continue
            changed.append(
                {
                    "target": identity,
                    "field": field,
                    "before": _bounded_effect_value(old),
                    "after": _bounded_effect_value(new),
                }
            )
            if field in {"text", "content_desc"}:
                if old:
                    changed_before_labels.add(_bounded_effect_value(old))
                if new:
                    changed_after_labels.add(_bounded_effect_value(new))
            if len(changed) >= _ACTION_EFFECT_MAX_ITEMS:
                break
        if len(changed) >= _ACTION_EFFECT_MAX_ITEMS:
            break
    if changed:
        effect["changed"] = changed

    before_labels = _effect_labels(before_nodes)
    after_labels = _effect_labels(after_nodes)
    appeared = sorted(
        (after_labels - before_labels) - changed_after_labels
    )[:_ACTION_EFFECT_MAX_ITEMS]
    disappeared = sorted(
        (before_labels - after_labels) - changed_before_labels
    )[:_ACTION_EFFECT_MAX_ITEMS]
    if appeared:
        effect["appeared"] = appeared
    if disappeared:
        effect["disappeared"] = disappeared
    return effect


def _effect_nodes(xml: str, *, target_package: str) -> dict[str, dict[str, str]]:
    try:
        root = ET.fromstring(str(xml or ""))
    except ET.ParseError:
        return {}
    nodes: dict[str, dict[str, str]] = {}
    for element in root.iter():
        attributes = element.attrib
        package_name = str(attributes.get("package") or "").strip()
        if target_package and package_name and package_name != target_package:
            continue
        resource_id = str(attributes.get("resource-id") or "").strip()
        class_name = str(attributes.get("class") or "").strip()
        bounds = str(attributes.get("bounds") or "").strip()
        identity = resource_id or "@".join(
            part for part in (class_name, bounds) if part
        )
        if not identity:
            continue
        fields = {
            "text": str(attributes.get("text") or "").strip(),
            "content_desc": str(attributes.get("content-desc") or "").strip(),
            "checked": str(attributes.get("checked") or "").strip(),
            "selected": str(attributes.get("selected") or "").strip(),
            "enabled": str(attributes.get("enabled") or "").strip(),
        }
        if not any(fields.values()):
            continue
        # Resource ids are normally unique. Bounds disambiguate repeated rows
        # while keeping stable controls aligned across adjacent observations.
        if identity in nodes and bounds:
            identity = f"{identity}@{bounds}"
        nodes[identity] = fields
    return nodes


def _dominant_effect_package(xml: str) -> str:
    try:
        root = ET.fromstring(str(xml or ""))
    except ET.ParseError:
        return ""
    counts: dict[str, int] = {}
    for element in root.iter():
        package_name = str(element.attrib.get("package") or "").strip()
        if not package_name or package_name == "com.android.systemui":
            continue
        counts[package_name] = counts.get(package_name, 0) + 1
    return max(counts, key=counts.get) if counts else ""


def _effect_labels(nodes: dict[str, dict[str, str]]) -> set[str]:
    labels: set[str] = set()
    for fields in nodes.values():
        for field in ("text", "content_desc"):
            value = _bounded_effect_value(fields.get(field, ""))
            if value:
                labels.add(value)
    return labels


def _bounded_effect_value(value: Any) -> str:
    text = " ".join(str(value or "").split())
    return text[:_ACTION_EFFECT_VALUE_MAX_CHARS]


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
    source_screenshot_path = _observation_screenshot_path(source_state)
    target_screenshot_path = _observation_screenshot_path(observation)
    if source_screenshot_path:
        request["source_screenshot_path"] = source_screenshot_path
    if target_screenshot_path:
        request["target_screenshot_path"] = target_screenshot_path
    try:
        source_point = _relative_source_point(
            source_state,
            float(action.args["x"]),
            float(action.args["y"]),
        )
    except (KeyError, TypeError, ValueError):
        source_point = None
    if source_point is None:
        return TransferResult(None, reason="omnitransfer_invalid_source_point")
    request["source_point"] = source_point
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
    mapped_transfer = TransferResult(
        Action(action.tool, {}),
        reason=str(result.get("mapping_mode") or "omnitransfer_mapped"),
        detail=transfer_detail,
    )
    admission = assess_transfer(mapped_transfer)
    if not admission.accepted:
        return _recoverable_transfer_failure(
            admission.reason or "omnitransfer_low_confidence",
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
    source_elements = _elements(source_xml)
    source_display_size = _display_size(source_state, source_elements)
    if source_display_size is None:
        return TransferResult(None, reason="omnitransfer_source_display_size_missing")
    width, height = display_size
    params = dict(action.args)
    try:
        source_points = tuple(
            _relative_source_point(
                source_state,
                float(params[x_key]),
                float(params[y_key]),
            )
            for x_key, y_key in (("x1", "y1"), ("x2", "y2"))
        )
    except (KeyError, TypeError, ValueError):
        return TransferResult(None, reason="omnitransfer_invalid_source_point")
    source_container = _swipe_container(source_xml, source_points)
    if source_container is None:
        return _recoverable_transfer_failure(
            "omnitransfer_swipe_source_container_missing",
            {},
        )
    source_bounds = source_container["bounds"]
    source_center = (
        (source_bounds[0] + source_bounds[2]) / 2.0,
        (source_bounds[1] + source_bounds[3]) / 2.0,
    )
    request: dict[str, Any] = {
        "target_xml": target_xml,
        "source_xml": source_xml,
        "source_point": source_center,
        "source_package_name": source_state.package_name,
        "target_package_name": observation.package_name,
        "source_activity_name": source_state.activity_name,
        "target_activity_name": observation.activity_name,
        "action_type": action.tool,
        "top_k": 3,
    }
    source_screenshot_path = _observation_screenshot_path(source_state)
    target_screenshot_path = _observation_screenshot_path(observation)
    if source_screenshot_path:
        request["source_screenshot_path"] = source_screenshot_path
    if target_screenshot_path:
        request["target_screenshot_path"] = target_screenshot_path
    try:
        result = transfer_action(**request)
    except Exception as exc:
        return TransferResult(None, reason=f"omnitransfer_error:{exc}")
    detail = _transfer_detail(result)
    if result.get("mapped") is not True:
        reason = result.get("reason") or result.get("mapping_mode") or "failed"
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
    container_transfer = TransferResult(
        Action(action.tool, {}),
        reason=str(result.get("mapping_mode") or "omnitransfer_mapped"),
        detail=detail,
    )
    container_admission = assess_transfer(container_transfer)
    if not container_admission.accepted:
        return _recoverable_transfer_failure(
            container_admission.reason or "omnitransfer_low_confidence",
            detail,
        )
    target_container = _mapped_swipe_container(target_xml, result)
    if target_container is None:
        return _recoverable_transfer_failure(
            "omnitransfer_swipe_target_not_executable",
            detail,
        )
    target_bounds = target_container["bounds"]
    for source_point, (x_key, y_key) in zip(
        source_points,
        (("x1", "y1"), ("x2", "y2")),
        strict=True,
    ):
        offset_x = (source_point[0] - source_bounds[0]) / (
            source_bounds[2] - source_bounds[0]
        )
        offset_y = (source_point[1] - source_bounds[1]) / (
            source_bounds[3] - source_bounds[1]
        )
        if not all(0.0 <= value <= 1.0 for value in (offset_x, offset_y)):
            return _recoverable_transfer_failure(
                "omnitransfer_swipe_source_point_outside_container",
                detail,
            )
        target_x = target_bounds[0] + offset_x * (
            target_bounds[2] - target_bounds[0]
        )
        target_y = target_bounds[1] + offset_y * (
            target_bounds[3] - target_bounds[1]
        )
        params[x_key] = target_x / width * 1000.0
        params[y_key] = target_y / height * 1000.0
    if not _mapped_swipe_preserves_gesture(action.args, params):
        return _recoverable_transfer_failure(
            "omnitransfer_swipe_gesture_degenerate",
            detail,
        )
    detail["source_swipe_container"] = _swipe_container_detail(source_container)
    detail["target_swipe_container"] = _swipe_container_detail(target_container)
    detail["mapped_swipe"] = {
        key: params[key] for key in ("direction", "x1", "y1", "x2", "y2")
        if key in params
    }
    reason = str(result.get("mapping_mode") or "omnitransfer_mapped")
    return TransferResult(
        Action(action.tool, params),
        reason=reason,
        detail=detail,
    )


def _observation_screenshot_path(observation: Observation | None) -> str:
    if observation is None:
        return ""
    direct = str(observation.extra.get("screenshot_path") or "").strip()
    if direct:
        return direct
    androidworld_state = observation.extra.get("androidworld_state")
    if not isinstance(androidworld_state, dict):
        return ""
    pixels = androidworld_state.get("pixels")
    if not isinstance(pixels, dict):
        return ""
    return str(pixels.get("path") or "").strip()


def _swipe_container(
    xml_text: str,
    points: tuple[tuple[float, float], ...],
) -> dict[str, Any] | None:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None
    candidates: list[dict[str, Any]] = []
    resource_counts: dict[str, int] = {}
    for element in root.iter():
        resource_id = str(element.attrib.get("resource-id") or "").strip()
        if resource_id:
            resource_counts[resource_id] = resource_counts.get(resource_id, 0) + 1
    for element in root.iter():
        if str(element.attrib.get("scrollable") or "").lower() != "true":
            continue
        if str(element.attrib.get("enabled", "true")).lower() == "false":
            continue
        bounds = _bounds(element.attrib.get("bounds"))
        if bounds is None or not all(_point_in_bounds(point, bounds) for point in points):
            continue
        resource_id = str(element.attrib.get("resource-id") or "").strip()
        node_id = str(element.attrib.get("id") or "").strip()
        element_id = (
            resource_id
            if resource_id and resource_counts.get(resource_id) == 1
            else node_id
        )
        if not element_id:
            continue
        candidates.append(
            {
                "bounds": tuple(float(value) for value in bounds),
                "element_id": element_id,
                "resource_id": resource_id,
                "node_id": node_id,
                "class": str(element.attrib.get("class") or element.tag),
            }
        )
    return min(
        candidates,
        key=lambda candidate: (
            (candidate["bounds"][2] - candidate["bounds"][0])
            * (candidate["bounds"][3] - candidate["bounds"][1]),
            candidate["element_id"],
        ),
        default=None,
    )


def _mapped_swipe_container(
    target_xml: str,
    result: dict[str, Any],
) -> dict[str, Any] | None:
    target_bounds = _numeric_bounds(result.get("target_bbox"))
    if target_bounds is None:
        return None
    target_ids = {
        str(result.get(key) or "").strip()
        for key in ("target_candidate_id", "target_execution_candidate_id")
        if str(result.get(key) or "").strip()
    }
    try:
        root = ET.fromstring(target_xml)
    except ET.ParseError:
        return None
    candidates: list[dict[str, Any]] = []
    for element in root.iter():
        if str(element.attrib.get("scrollable") or "").lower() != "true":
            continue
        if str(element.attrib.get("enabled", "true")).lower() == "false":
            continue
        bounds = _bounds(element.attrib.get("bounds"))
        if bounds is None:
            continue
        numeric_bounds = tuple(float(value) for value in bounds)
        resource_id = str(element.attrib.get("resource-id") or "").strip()
        node_id = str(element.attrib.get("id") or "").strip()
        identifiers = {resource_id, resource_id.rsplit("/", 1)[-1], node_id} - {""}
        bounds_match = all(
            abs(left - right) <= 1.0
            for left, right in zip(numeric_bounds, target_bounds, strict=True)
        )
        if not bounds_match and not target_ids.intersection(identifiers):
            continue
        candidates.append(
            {
                "bounds": numeric_bounds,
                "element_id": resource_id or node_id,
                "resource_id": resource_id,
                "node_id": node_id,
                "class": str(element.attrib.get("class") or element.tag),
            }
        )
    return min(
        candidates,
        key=lambda candidate: (
            0 if candidate["bounds"] == target_bounds else 1,
            (candidate["bounds"][2] - candidate["bounds"][0])
            * (candidate["bounds"][3] - candidate["bounds"][1]),
        ),
        default=None,
    )


def _point_in_bounds(
    point: tuple[float, float],
    bounds: tuple[int, int, int, int],
) -> bool:
    return bounds[0] <= point[0] <= bounds[2] and bounds[1] <= point[1] <= bounds[3]


def _mapped_swipe_preserves_gesture(
    source_args: dict[str, Any],
    target_args: dict[str, Any],
) -> bool:
    try:
        source_dx = float(source_args["x2"]) - float(source_args["x1"])
        source_dy = float(source_args["y2"]) - float(source_args["y1"])
        target_dx = float(target_args["x2"]) - float(target_args["x1"])
        target_dy = float(target_args["y2"]) - float(target_args["y1"])
    except (KeyError, TypeError, ValueError):
        return False
    source_direction = _swipe_direction(source_args, source_dx, source_dy)
    target_direction = _swipe_direction({}, target_dx, target_dy)
    if source_direction != target_direction:
        return False
    source_distance = max(abs(source_dx), abs(source_dy))
    target_distance = max(abs(target_dx), abs(target_dy))
    return target_distance >= max(25.0, source_distance * 0.25)


def _swipe_direction(
    args: dict[str, Any],
    delta_x: float,
    delta_y: float,
) -> str:
    direction = str(args.get("direction") or "").strip().lower()
    if direction in {"left", "right", "up", "down"}:
        return direction
    if abs(delta_x) > abs(delta_y):
        return "right" if delta_x > 0 else "left"
    return "down" if delta_y > 0 else "up"


def _swipe_container_detail(container: dict[str, Any]) -> dict[str, Any]:
    return {
        key: container[key]
        for key in ("element_id", "resource_id", "node_id", "class", "bounds")
        if container.get(key) not in (None, "")
    }


def _action_uses_transfer_target(action: Action) -> bool:
    if action.tool == "input_text":
        return all(action.args.get(key) is not None for key in ("x", "y"))
    if action.tool in {"click", "long_press"}:
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
    target_bounds = result.get("target_bbox")
    if isinstance(target_bounds, (list, tuple)) and len(target_bounds) == 4:
        target["bounds"] = list(target_bounds)
    candidates = []
    for rank, raw in enumerate(result.get("top_candidates") or (), start=1):
        if not isinstance(raw, dict):
            continue
        candidate = _element_detail(raw)
        candidate["rank"] = rank
        candidate["bounds"] = list(raw.get("bbox") or ())
        candidate["execution_bounds"] = list(raw.get("execution_bbox") or ())
        candidate["execution_candidate_id"] = str(
            raw.get("execution_candidate_id") or ""
        )
        candidate["executable"] = raw.get("executable") is True
        candidate["score"] = raw.get("score")
        candidates.append(candidate)
    detail = {
        "mapping_mode": str(result.get("mapping_mode") or ""),
        "matcher_release": str(result.get("matcher_release") or ""),
        "matcher_backend": str(result.get("matcher_backend") or ""),
        "matcher_checkpoint": str(result.get("matcher_checkpoint") or ""),
        "matcher_checkpoint_sha256": str(
            result.get("matcher_checkpoint_sha256") or ""
        ),
        "source": source,
        "target": target,
        "candidates": candidates,
    }
    if result.get("mapped") is True:
        for key in (
            "score",
            "margin",
            "pair_confidence",
            "rank_probability",
            "null_probability",
        ):
            try:
                detail[key] = float(result[key])
            except (KeyError, TypeError, ValueError):
                continue
        if "score" not in detail and candidates:
            try:
                detail["score"] = float(candidates[0]["score"])
            except (KeyError, TypeError, ValueError):
                pass
        for key in (
            "absolute_contextual_confidence",
            "pair_confidence",
            "score",
        ):
            try:
                detail["absolute_contextual_confidence"] = float(result[key])
                break
            except (KeyError, TypeError, ValueError):
                continue
        target_candidate_id = str(result.get("target_candidate_id") or "").strip()
        if target_candidate_id:
            detail["target_candidate_id"] = target_candidate_id
        target_execution_candidate_id = str(
            result.get("target_execution_candidate_id") or ""
        ).strip()
        if target_execution_candidate_id:
            detail["target_execution_candidate_id"] = (
                target_execution_candidate_id
            )
    return detail


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
            "clickable",
            "long_clickable",
            "editable",
            "enabled",
            "visible",
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
