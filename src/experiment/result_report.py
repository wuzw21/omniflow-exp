"""Execution audit and OmniFlow result-block formatting (split out of run_task).

Moved verbatim from ``src.experiment.run_task``; that module re-exports these names.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from src.experiment.paths import sha256_file
from src.experiment.run_task_common import (
    DeviceTarget,
    _coerce_float,
    _coerce_int,
    _last_jsonl_object,
    _read_json,
)


def _result_device_name(target: DeviceTarget) -> str:
    label = str(target.label or "").casefold()
    if "standard" in label or "small" in label or "pixel6" in label:
        return "Standard / Pixel 6 Pro"
    if "fold" in label:
        return "Fold"
    if "tablet" in label or "wxga" in label:
        return "Tablet"
    return str(target.label or target.serial or "Unknown")


def _percentage_text(value: Any) -> str:
    try:
        percentage = float(value) * 100.0
    except (TypeError, ValueError):
        return "n/a"
    if percentage.is_integer():
        return f"{int(percentage)}%"
    return f"{percentage:.1f}%"


def _execution_audit(diagnostics: dict[str, Any]) -> dict[str, Any]:
    trace = diagnostics.get("execution_trace")
    trace_rows = (
        [row for row in trace if isinstance(row, dict)]
        if isinstance(trace, list)
        else []
    )
    function_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    planner_actions: list[str] = []
    planner_action_details: list[str] = []
    failures: list[str] = []
    for row in trace_rows:
        action = row.get("action")
        action_value = action if isinstance(action, dict) else {}
        tool = str(action_value.get("tool") or "unknown").strip()
        metadata = row.get("metadata")
        metadata_value = metadata if isinstance(metadata, dict) else {}
        result = row.get("result")
        result_value = result if isinstance(result, dict) else {}
        function_id = str(metadata_value.get("function_id") or "").strip()
        success = result_value.get("success") is True
        if function_id:
            function_rows.append(row)
        elif success:
            planner_actions.append(tool)
            step_index = row.get("step_index")
            try:
                step_text = str(int(step_index) + 1)
            except (TypeError, ValueError):
                step_text = "unknown"
            detail = f"step={step_text} tool={tool}"
            summary = str(metadata_value.get("summary") or "").strip()
            if summary:
                detail += f" intent={summary}"
            planner_action_details.append(detail)
        transfer = metadata_value.get("transfer")
        if isinstance(transfer, dict):
            attempts = transfer.get("transfer_attempts")
            if isinstance(attempts, list) and all(isinstance(item, dict)
                    and isinstance(item.get("mapped"), bool) for item in attempts):
                transfer_rows.extend(attempts)
            else:
                transfer_rows.append(row)
        if not function_id or success:
            continue
        step_index = metadata_value.get("function_step_index")
        try:
            step_text = str(int(step_index) + 1)
        except (TypeError, ValueError):
            step_text = "unknown"
        error = str(result_value.get("error") or "unknown_execution_error").strip()
        detail = f"step={step_text} tool={tool} error={error}"
        if isinstance(transfer, dict):
            score = _coerce_float(transfer.get("score"))
            detail += f" score={score:.3f}"
            if error.startswith("omnitransfer_"):
                detail += " canonical_transfer_result"
            candidates = transfer.get("candidates")
            candidate_rows: list[str] = []
            for candidate in candidates[:3] if isinstance(candidates, list) else ():
                if not isinstance(candidate, dict):
                    continue
                label = str(
                    candidate.get("text")
                    or candidate.get("content_desc")
                    or candidate.get("resource_id")
                    or candidate.get("execution_candidate_id")
                    or "unlabeled"
                ).strip()
                candidate_rows.append(
                    f"{label}({_coerce_float(candidate.get('score')):.3f})"
                )
            if candidate_rows:
                detail += " candidates=" + ", ".join(candidate_rows)
        failures.append(detail)
    function_success = sum(
        isinstance(row.get("result"), dict)
        and row["result"].get("success") is True
        for row in function_rows
    )
    transfer_success = 0
    for row in transfer_rows:
        if isinstance(row.get("mapped"), bool):
            transfer_success += int(row["mapped"])
            continue
        result = row.get("result")
        result_value = result if isinstance(result, dict) else {}
        error = str(result_value.get("error") or "").strip()
        if result_value.get("success") is True or not error.startswith("omnitransfer_"):
            transfer_success += 1
    resume = diagnostics.get("function_resume")
    resume_value = resume if isinstance(resume, dict) else {}
    usage = diagnostics.get("llm_usage")
    usage_value = usage if isinstance(usage, dict) else {}
    components = usage_value.get("by_component")
    component_values = components if isinstance(components, dict) else {}
    router = component_values.get("function_router")
    planner = component_values.get("planner")
    router_value = router if isinstance(router, dict) else {}
    planner_value = planner if isinstance(planner, dict) else {}
    planner_diagnostics = diagnostics.get("planner_diagnostics")
    planner_diagnostics_value = (
        planner_diagnostics if isinstance(planner_diagnostics, dict) else {}
    )
    planner_calls = planner_diagnostics_value.get("planner_calls")
    planner_call_rows = (
        [row for row in planner_calls if isinstance(row, dict)]
        if isinstance(planner_calls, list)
        else []
    )
    terminal_planner_call = dict(planner_call_rows[-1]) if planner_call_rows else {}
    function_resolution = diagnostics.get("function_resolution")
    resolution_value = (
        function_resolution if isinstance(function_resolution, dict) else {}
    )
    selected_function_id = str(
        resolution_value.get("selected_function_id") or ""
    ).strip()
    if not selected_function_id:
        for row in function_rows:
            metadata = row.get("metadata")
            metadata_value = metadata if isinstance(metadata, dict) else {}
            selected_function_id = str(
                metadata_value.get("function_id") or ""
            ).strip()
            if selected_function_id:
                break
    resolution_status = str(resolution_value.get("status") or "unknown").strip()
    binding_status = str(
        resolution_value.get("binding_status") or "not_attempted"
    ).strip()
    replay_status = str(
        resolution_value.get("replay_status") or "not_started"
    ).strip()
    if function_rows and resolution_status == "planner_tool_space":
        resolution_status = "planner_selected"
    if function_rows and replay_status == "not_started":
        replay_status = "failed" if failures else "executed_without_recorded_failure"
    resolution_failures: list[str] = []
    binding_error = str(resolution_value.get("binding_error") or "").strip()
    if binding_status == "failed":
        resolution_failures.append(f"binding_failed:{binding_error or 'unknown'}")
    replay_error = str(resolution_value.get("replay_error") or "").strip()
    if replay_status == "failed" and replay_error and not failures:
        resolution_failures.append(f"replay_failed:{replay_error}")
    recall = resolution_value.get("recall")
    recall_value = recall if isinstance(recall, dict) else {}
    events = recall_value.get("events")
    recall_events = (
        [event for event in events if isinstance(event, dict)]
        if isinstance(events, list)
        else []
    )
    router_statuses = [
        str(cache.get("status") or "unknown").strip()
        for event in recall_events
        if isinstance((cache := event.get("function_cache")), dict)
    ]
    registered_candidate_count = max(
        [_coerce_int(resolution_value.get("candidate_count"))]
        + [
            len(
                event.get("registered_candidate_function_ids")
                or event.get("candidate_function_ids")
                or ()
            )
            for event in recall_events
        ]
    )
    planner_candidate_count = max(
        [_coerce_int(resolution_value.get("planner_candidate_count"))]
        + [len(event.get("planner_function_ids") or ()) for event in recall_events]
    )
    if not selected_function_id:
        for event in recall_events:
            cache = event.get("function_cache")
            cache_value = cache if isinstance(cache, dict) else {}
            cache_status = str(cache_value.get("status") or "").strip()
            if cache_status in {
                "below_threshold",
                "router_unavailable",
                "rejected",
                "error",
                "unknown_selection",
                "binding_failed",
                "entry_state_changed",
            }:
                cache_error = str(cache_value.get("error") or "").strip()
                detail = f"cache_{cache_status}"
                if cache_error:
                    detail += f":{cache_error}"
                cache_decisions = cache_value.get("decisions")
                score_rows: list[str] = []
                for decision in (
                    cache_decisions if isinstance(cache_decisions, list) else ()
                ):
                    if not isinstance(decision, dict):
                        continue
                    function_id = str(
                        decision.get("function_id") or "unknown"
                    ).strip()
                    confidence = _coerce_float(decision.get("confidence"))
                    score_rows.append(f"{function_id}({confidence:.3f})")
                if score_rows:
                    if cache_status == "below_threshold":
                        threshold = _coerce_float(cache_value.get("threshold"))
                        detail += f" threshold={threshold:.3f}"
                    detail += " candidates=" + ", ".join(score_rows)
                resolution_failures.append(detail)
                break
        if not resolution_failures:
            rejection_rows: list[str] = []
            for event in recall_events:
                decisions = event.get("decisions")
                for decision in decisions if isinstance(decisions, list) else ():
                    if not isinstance(decision, dict):
                        continue
                    rejection_reason = str(
                        decision.get("rejection_reason") or ""
                    ).strip()
                    if not rejection_reason or rejection_reason == "function_excluded_for_run":
                        continue
                    function_id = str(decision.get("function_id") or "unknown").strip()
                    rejection_rows.append(f"{function_id}:{rejection_reason}")
            if rejection_rows:
                resolution_failures.append(
                    "recall_rejected:" + ", ".join(dict.fromkeys(rejection_rows))
                )
            elif recall_events and not any(
                event.get("candidate_function_ids") for event in recall_events
            ):
                resolution_failures.append("recall_returned_no_function_candidate")
    return {
        "function_attempts": len(function_rows),
        "function_success": function_success,
        "function_failure": len(function_rows) - function_success,
        "transfer_attempts": len(transfer_rows),
        "transfer_success": transfer_success,
        "transfer_failure": len(transfer_rows) - transfer_success,
        "failures": failures,
        "resolution_failures": resolution_failures,
        "selected_function_id": selected_function_id,
        "registered_candidate_count": registered_candidate_count,
        "planner_candidate_count": planner_candidate_count,
        "router_statuses": router_statuses,
        "resolution_status": resolution_status,
        "binding_status": binding_status,
        "replay_status": replay_status,
        "planner_actions": planner_actions,
        "planner_action_details": planner_action_details,
        "terminal_planner_call": terminal_planner_call,
        "resume_attempts": (resume_value.get("attempt_count")
                            if type(resume_value.get("attempt_count")) is int else None),
        "resume_success": (resume_value.get("success_count")
                           if type(resume_value.get("success_count")) is int else None),
        "router_tokens": _coerce_int(router_value.get("total_tokens")),
        "planner_tokens": _coerce_int(planner_value.get("total_tokens")),
        "router_calls": _coerce_int(router_value.get("model_calls")),
        "planner_calls": _coerce_int(planner_value.get("model_calls")),
    }


def _compact_report_value(value: Any, *, limit: int = 320) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _task_failure_attribution(
    *,
    completed: bool,
    official: bool,
    success: bool,
    execution_summary: dict[str, Any],
    audit: dict[str, Any],
    function_actions: int,
    function_denominator: int,
) -> str:
    if not completed:
        return "run_not_completed"
    if official and success:
        return "none"
    runtime_error = str(execution_summary.get("failure_reason") or "").strip()
    if runtime_error:
        return f"runtime_failure:{runtime_error}"
    if (
        not official
        and not audit.get("function_attempts")
        and not audit.get("planner_actions")
        and not audit.get("router_statuses")
    ):
        return "initialization_or_environment_failure:official_validator_not_reached"
    function_failures = list(audit.get("failures") or ()) + list(
        audit.get("resolution_failures") or ()
    )
    if function_failures:
        return "function_or_transfer_failure:" + " | ".join(
            str(value) for value in function_failures
        )
    terminal_call = audit.get("terminal_planner_call")
    terminal_call_value = terminal_call if isinstance(terminal_call, dict) else {}
    terminal_tool = str(terminal_call_value.get("tool") or "").strip()
    terminal_arguments = terminal_call_value.get("arguments")
    terminal_text = (
        f"; Planner terminal={terminal_tool}"
        f"({_compact_report_value(terminal_arguments)})"
        if terminal_tool
        else ""
    )
    if official and not success:
        if audit.get("replay_status") == "succeeded":
            planner_physical_actions = max(0, function_denominator - function_actions)
            if planner_physical_actions:
                return (
                    "official_validator_rejected_planner_completion:Function replay "
                    f"succeeded but covered {function_actions}/{function_denominator} "
                    "physical actions; Planner supplied "
                    f"{planner_physical_actions} remaining physical action(s)"
                    f"{terminal_text}"
                )
            return (
                "official_validator_rejected_function_result:Function supplied all "
                f"{function_denominator} physical action(s){terminal_text}"
            )
        return "official_validator_rejected_planner_result" + terminal_text
    return "non_official_or_unclassified_failure" + terminal_text


def format_omniflow_result_block(
    *,
    output_path: Path,
    target: DeviceTarget,
    source_seed: int,
    evaluation_seed: int,
    store_path: Path | None,
) -> str:
    """Render one copy-ready result block from the sealed official evidence."""

    run_log_path = output_path / "run_log.json"
    task_result = _last_jsonl_object(output_path / "task_results.jsonl")
    run_log = _read_json(run_log_path) if run_log_path.is_file() else {}
    validator = (
        dict(run_log.get("validator") or {})
        if isinstance(run_log.get("validator"), dict)
        else {}
    )
    official = (
        bool(task_result.get("official_validator_used"))
        if "official_validator_used" in task_result
        else bool(validator.get("official"))
    )
    success = (
        bool(task_result.get("success"))
        if "success" in task_result
        else bool(validator.get("success"))
    )
    completed = bool(task_result or run_log)
    actions = _coerce_int(
        task_result.get("actions_executed")
        or len(list(run_log.get("steps") or []))
    )
    reuse = task_result.get("reuse_metrics")
    if not isinstance(reuse, dict):
        from src.integrations.android_world.methods import reuse_metrics

        reuse = reuse_metrics(
            "omniflow",
            actions_executed=actions,
            canonical_run=run_log,
        )
    physical_actions = _coerce_int(reuse.get("physical_action_count"))
    state_changing_actions = _coerce_int(
        reuse.get("state_changing_physical_action_count")
    )
    function_actions = _coerce_int(reuse.get("reuse_numerator"))
    function_denominator = _coerce_int(reuse.get("reuse_denominator"))
    diagnostics = (
        dict(run_log.get("diagnostics") or {})
        if isinstance(run_log.get("diagnostics"), dict)
        else {}
    )
    execution_summary = (
        dict(diagnostics.get("execution_summary") or {})
        if isinstance(diagnostics.get("execution_summary"), dict)
        else {}
    )
    audit = _execution_audit(diagnostics)
    fallback_steps = _coerce_int(
        task_result.get("fallback_steps")
        if "fallback_steps" in task_result
        else execution_summary.get("fallback_steps")
    )
    model_calls = _coerce_int(
        task_result.get("model_calls")
        if "model_calls" in task_result
        else execution_summary.get("model_calls")
    )
    prompt_tokens = _coerce_int(
        task_result.get("prompt_tokens")
        if "prompt_tokens" in task_result
        else execution_summary.get("prompt_tokens")
    )
    completion_tokens = _coerce_int(
        task_result.get("completion_tokens")
        if "completion_tokens" in task_result
        else execution_summary.get("completion_tokens")
    )
    total_tokens = _coerce_int(
        task_result.get("total_tokens")
        if "total_tokens" in task_result
        else execution_summary.get("total_tokens")
        or execution_summary.get("tokens")
    )
    execution_ms = _coerce_float(
        task_result.get("execution_duration_ms")
        if "execution_duration_ms" in task_result
        else execution_summary.get("execution_duration_ms")
    )
    lifecycle_ms = _coerce_float(task_result.get("duration_ms"))
    action_types = {
        str((step.get("action") or {}).get("action_type") or "").strip()
        for step in run_log.get("steps") or []
        if isinstance(step, dict) and isinstance(step.get("action"), dict)
    }
    action_note = "（含非物理 answer）" if "answer" in action_types else ""
    validator_count = f"{1 if official and success else 0}/1" if official else "未覆盖"
    failure_attribution = _task_failure_attribution(
        completed=completed,
        official=official,
        success=success,
        execution_summary=execution_summary,
        audit=audit,
        function_actions=function_actions,
        function_denominator=function_denominator,
    )
    resolved_store = (
        str(store_path.expanduser().resolve()) if store_path is not None else "未提供"
    )
    digest = sha256_file(run_log_path) if run_log_path.is_file() else "未生成"
    return "\n".join(
        (
            "|   |",
            "| - |",
            "",
            f"完成：{'是' if completed else '否'}",
            "",
            f"通过：{'是' if official and success else '否'}（official validator {validator_count}）",
            "",
            f"设备：{_result_device_name(target)}",
            "",
            f"source_seed={int(source_seed)}；evaluation_seed={int(evaluation_seed)}",
            "",
            (
                f"actions={actions}{action_note}；physical={physical_actions}；"
                f"state-changing={state_changing_actions}"
            ),
            "",
            (
                f"Function successful physical reuse={function_actions}/"
                f"{function_denominator}；"
                f"reuse={_percentage_text(reuse.get('reuse_rate'))}；"
                f"fallback={fallback_steps}；model_calls={model_calls if task_result.get('model_calls') is not None else 'unknown'}"
            ),
            "",
            (
                f"Function attempts/success/failure="
                f"{audit['function_attempts']}/{audit['function_success']}/"
                f"{audit['function_failure']}；Transfer attempts/success/failure="
                f"{audit['transfer_attempts']}/{audit['transfer_success']}/"
                f"{audit['transfer_failure']}"
            ),
            "",
            (
                "Function outcome："
                f"selected={audit['selected_function_id'] or 'none'}；"
                f"resolution={audit['resolution_status']}；"
                f"binding={audit['binding_status']}；"
                f"replay={audit['replay_status']}"
            ),
            "",
            (
                "Function registration："
                f"registered={audit['registered_candidate_count']}；"
                f"planner_exposed={audit['planner_candidate_count']}；"
                "Router="
                + (" → ".join(audit["router_statuses"]) or "not_reached")
            ),
            "",
            (
                "Function failure reasons："
                + (
                    " | ".join(
                        str(value)
                        for value in (
                            list(audit["failures"])
                            + list(audit["resolution_failures"])
                        )
                    )
                    if audit["failures"] or audit["resolution_failures"]
                    else "none"
                )
            ),
            "",
            f"Task failure attribution：{failure_attribution}",
            "",
            (
                "Planner physical actions="
                + (", ".join(str(value) for value in audit["planner_actions"]) or "none")
                + f"；Function resume={audit['resume_success']}/{audit['resume_attempts']}"
            ),
            "",
            (
                "Planner action evidence："
                + (
                    " | ".join(
                        str(value) for value in audit["planner_action_details"]
                    )
                    or "none"
                )
            ),
            "",
            (
                "Reuse gap："
                + (
                    f"{function_denominator - function_actions} successful physical "
                    "action(s) came from Planner; this gap is not itself a Function "
                    "failure"
                    if function_denominator > function_actions
                    else "none"
                )
            ),
            "",
            f"tokens={prompt_tokens}/{completion_tokens}/{total_tokens}",
            "",
            (
                f"token components：router={audit['router_tokens']} "
                f"({audit['router_calls']} call(s))；planner={audit['planner_tokens']} "
                f"({audit['planner_calls']} call(s))"
            ),
            "",
            (
                f"execution={execution_ms / 1000.0:.3f}s（排除 setup/validator）；"
                f"lifecycle={lifecycle_ms / 1000.0:.3f}s"
            ),
            "",
            f"Store：{resolved_store}",
            "",
            f"RunLog：{run_log_path.resolve()}",
            "",
            f"SHA-256：{digest}",
        )
    )
