"""Small public result rows and their single detailed evidence companion."""

from __future__ import annotations

from typing import Any

RESULT_FIELDS = (
    "task", "method", "device", "source_seed", "evaluation_seed", "status",
    "validator_success", "official_validator_used",
    "official_validator_coverage_rate", "replay_completed", "model_calls",
    "model_calls_source", "prompt_tokens", "completion_tokens",
    "total_tokens", "actions_executed", "episode_duration_sec", "outer_wall_sec",
    "function_hit", "function_covered_steps", "function_total_steps",
    "function_step_coverage_rate", "vlm_calls", "vlm_latency_ms", "latency_sec",
    "energy_mwh", "energy_measurement_available", "attempt_id", "device_model",
    "device_serial", "model", "chat_model_calls", "embedding_model_calls",
    "prep_model_calls", "prep_prompt_tokens", "prep_completion_tokens",
    "prep_total_tokens", "model_calls_total", "total_tokens_including_prep",
    "token_usage_status",
    "memory_status", "memory_source", "memory_used",
    "memory_hit", "memory_covered_steps", "memory_total_steps", "memory_hit_rate",
    "memory_explore_count", "memory_action_recalled_count",
    "memory_action_use_rate",
    "fallback_steps", "fallback_steps_source", "max_fallback_steps",
    "fallback_measurement_status",
    "fallback_budget_exhausted", "method_outcome", "failure_stage",
    "environment_failure", "failure_reason", "error", "evidence_paths",
)


def function_metrics_from_result_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize Function reuse into task- and step-level public metrics."""

    reuse = row.get("reuse_metrics")
    method = str(row.get("method") or row.get("agent") or "").strip()
    function_id = str(row.get("function_id") or "").strip()
    # Only a real Function-backed trace counts. Fixed replay and hint injection
    # reuse artifacts, but they do not constitute a Function hit.
    function_backed = method == "omniflow" or bool(function_id)
    if isinstance(reuse, dict):
        artifact_used = bool(reuse.get("artifact_used"))
        covered_steps = max(0, int(float(reuse.get("reuse_numerator") or 0)))
        total_steps = max(0, int(float(reuse.get("reuse_denominator") or 0)))
        hit = bool(function_backed and artifact_used and covered_steps > 0)
    elif any(
        key in row
        for key in (
            "function_hit",
            "function_covered_steps",
            "function_total_steps",
            "function_step_coverage_rate",
        )
    ):
        # Registered public rows are already normalized. Preserve their
        # Function metrics when they are compacted again; recomputing from
        # reuse_metrics would erase them because that detail is intentionally
        # omitted from the public row.
        covered_steps = max(0, int(float(row.get("function_covered_steps") or 0)))
        total_steps = max(0, int(float(row.get("function_total_steps") or 0)))
        hit = bool(function_backed and row.get("function_hit"))
    else:
        covered_steps = 0
        total_steps = 0
        hit = False
    return {
        "function_hit": hit,
        "function_covered_steps": covered_steps if function_backed else 0,
        "function_total_steps": total_steps if function_backed else 0,
        "function_step_coverage_rate": (
            round(float(covered_steps) / float(total_steps), 6)
            if function_backed and total_steps > 0
            else None
        ),
    }


def performance_metrics_from_result_row(row: dict[str, Any]) -> dict[str, Any]:
    """Extract stable scalar latency/energy fields from a performance sidecar."""

    performance = row.get("performance_metrics")
    performance = performance if isinstance(performance, dict) else {}
    energy = performance.get("energy")
    energy = energy if isinstance(energy, dict) else {}
    usage = row.get("llm_usage")
    usage = usage if isinstance(usage, dict) else {}
    vlm_latency_ms = row.get("vlm_latency_ms")
    if vlm_latency_ms in (None, ""):
        vlm_latency_ms = usage.get("latency_ms") or usage.get("total_latency_ms")
    try:
        vlm_latency_ms = round(float(vlm_latency_ms or 0.0), 6)
    except (TypeError, ValueError):
        vlm_latency_ms = 0.0
    latency_sec = performance.get("method_wall_sec")
    if latency_sec in (None, ""):
        latency_sec = row.get("episode_duration_sec") or row.get("duration_sec")
    try:
        latency_sec = round(float(latency_sec or 0.0), 6)
    except (TypeError, ValueError):
        latency_sec = 0.0
    energy_mwh = energy.get("estimated_mwh")
    try:
        energy_mwh = round(float(energy_mwh), 6) if energy_mwh is not None else None
    except (TypeError, ValueError):
        energy_mwh = None
    return {
        "vlm_calls": int(float(row.get("vlm_calls") or row.get("model_calls") or 0)),
        "vlm_latency_ms": vlm_latency_ms,
        "latency_sec": latency_sec,
        "energy_mwh": energy_mwh,
        "energy_measurement_available": bool(energy.get("measurement_available")),
    }


def _optional_integer(row: dict[str, Any], *keys: str) -> int | None:
    """Read a count without turning an unreported metric into a measured zero."""

    for key in keys:
        if key not in row or row.get(key) in (None, ""):
            continue
        try:
            return int(float(row.get(key)))
        except (TypeError, ValueError):
            return None
    return None


def _optional_number(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key not in row or row.get(key) in (None, ""):
            continue
        try:
            return round(float(row.get(key)), 6)
        except (TypeError, ValueError):
            return None
    return None


def execution_metrics_from_result_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize metrics needed to audit one method/device cell.

    The public schema deliberately distinguishes a measured zero from a metric
    the method did not expose.  This is important for the official MobileGPT
    and AppAgent adapters: missing fallback accounting is not evidence of zero
    fallback, and missing memory-hit accounting is not evidence of a miss.
    """

    method = str(row.get("method") or row.get("agent") or "").strip()
    target = row.get("device_target")
    target = target if isinstance(target, dict) else {}
    function = function_metrics_from_result_row(row)

    chat_calls = _optional_integer(
        row, "chat_model_calls", "episode_chat_model_calls", "shared_chat_model_calls"
    )
    embedding_calls = _optional_integer(
        row,
        "embedding_model_calls",
        "episode_embedding_model_calls",
        "shared_embedding_model_calls",
    )
    model_calls = _optional_integer(row, "model_calls")
    if model_calls is None and chat_calls is not None and embedding_calls is not None:
        model_calls = chat_calls + embedding_calls

    prep_model_calls = _optional_integer(row, "prep_model_calls")
    prep_prompt_tokens = _optional_integer(row, "prep_prompt_tokens")
    prep_completion_tokens = _optional_integer(row, "prep_completion_tokens")
    prep_total_tokens = _optional_integer(row, "prep_total_tokens")
    model_calls_total = (
        model_calls + prep_model_calls
        if model_calls is not None and prep_model_calls is not None
        else model_calls
    )
    episode_total_tokens = _optional_integer(row, "total_tokens", "tokens")
    total_tokens_including_prep = (
        episode_total_tokens + prep_total_tokens
        if episode_total_tokens is not None and prep_total_tokens is not None
        else episode_total_tokens
    )

    explicit_memory_status = str(row.get("memory_status") or "").strip()
    explicit_memory_source = str(row.get("memory_source") or "").strip()
    reuse = row.get("reuse_metrics")
    artifact_used = bool(
        row.get("artifact_used")
        or (reuse.get("artifact_used") if isinstance(reuse, dict) else False)
    )
    memory_root = str(
        row.get("memory_root")
        or row.get("memory_path")
        or row.get("prep_manifest")
        or row.get("memory_manifest")
        or ""
    ).strip()
    if explicit_memory_status:
        memory_status = explicit_memory_status
    elif method == "omniflow" and artifact_used:
        memory_status = "used"
    elif method == "mobilegpt" and memory_root:
        memory_status = "used"
    elif method == "appagent" and memory_root:
        # AppAgent's preparation manifest proves the source memory was
        # prepared; unless the official executor reports retrieval, do not
        # claim that the episode consumed a memory hit.
        memory_status = "prepared"
    elif method in {"fixed_replay", "t3a_hint"}:
        memory_status = "not_applicable"
    else:
        memory_status = "unavailable"

    if explicit_memory_source:
        memory_source = explicit_memory_source
    elif method == "omniflow" and artifact_used:
        memory_source = "function_store"
    elif method == "mobilegpt" and memory_root:
        memory_source = "mobilegpt_memory"
    elif method == "appagent" and memory_root:
        memory_source = "appagent_preparation"
    elif memory_status == "not_applicable":
        memory_source = "none"
    else:
        memory_source = "unavailable"

    explicit_memory_used = row.get("memory_used")
    if isinstance(explicit_memory_used, bool):
        memory_used = explicit_memory_used
    else:
        memory_used = memory_status == "used"

    explicit_memory_hit = row.get("memory_hit")
    if isinstance(explicit_memory_hit, bool):
        memory_hit = explicit_memory_hit
    elif method == "omniflow":
        memory_hit = bool(function["function_hit"])
    else:
        memory_hit = None
    memory_covered = _optional_integer(
        row, "memory_covered_steps", "memory_hit_steps", "recall_hit_steps"
    )
    memory_total = _optional_integer(
        row, "memory_total_steps", "memory_query_steps", "recall_total_steps"
    )
    if method == "omniflow":
        memory_covered = function["function_covered_steps"]
        memory_total = function["function_total_steps"]
    memory_hit_rate = _optional_number(row, "memory_hit_rate", "recall_hit_rate")
    if memory_hit_rate is None and memory_total:
        memory_hit_rate = round(float(memory_covered or 0) / float(memory_total), 6)
    if memory_status in {"unavailable", "not_applicable"} and not (
        memory_covered or memory_total
    ):
        memory_hit_rate = None
    memory_explore_count = _optional_integer(row, "memory_explore_count")
    memory_action_recalled_count = _optional_integer(
        row, "memory_action_recalled_count"
    )
    memory_action_use_rate = _optional_number(row, "memory_action_use_rate")

    fallback_present = "fallback_steps" in row and row.get("fallback_steps") not in (
        None,
        "",
    )
    fallback_source = str(row.get("fallback_steps_source") or "").strip()
    measured_fallback = bool(
        fallback_source
        or "episode_fallback_count" in row
        or (method == "omniflow" and fallback_present)
    )
    fallback_steps = (
        _optional_integer(row, "fallback_steps")
        if fallback_present and measured_fallback
        else None
    )
    if measured_fallback:
        fallback_status = str(row.get("fallback_measurement_status") or "measured")
    elif method in {"fixed_replay", "t3a_hint"}:
        fallback_steps = 0
        fallback_status = "not_applicable"
    elif method in {"mobilegpt", "appagent"}:
        fallback_status = "not_exposed"
    else:
        fallback_status = "unavailable"
    max_fallback_steps = _optional_integer(row, "max_fallback_steps")
    if not fallback_source and measured_fallback:
        fallback_source = "result_row"
    fallback_budget_exhausted = bool(
        row.get("fallback_budget_exhausted")
        or "fallback_budget_exhausted" in str(row.get("error") or "")
        or "fallback_budget_exhausted" in str(row.get("failure_summary") or "")
    )

    validator = row.get("validator_success")
    if validator is None and isinstance(row.get("official_validator_success"), bool):
        validator = row.get("official_validator_success")
    if validator is True:
        method_outcome = "success"
    elif validator is False:
        method_outcome = "validator_failure"
    elif str(row.get("status") or "").strip() in {"pending", "running"}:
        method_outcome = str(row.get("status"))
    else:
        method_outcome = "execution_failure" if row.get("error") or row.get("failure_summary") else "unknown"
    failure_stage = str(row.get("failure_stage") or row.get("stage") or "").strip()
    failure_reason = str(
        row.get("failure_reason")
        or row.get("failure_summary")
        or row.get("error")
        or ""
    ).strip()
    environment_failure = row.get("environment_failure")
    if not isinstance(environment_failure, bool):
        environment_failure = None

    return {
        "attempt_id": str(row.get("attempt_id") or ""),
        "device_model": str(
            row.get("device_model") or target.get("device_model") or target.get("avd") or ""
        ),
        "device_serial": str(row.get("device_serial") or target.get("serial") or row.get("serial") or ""),
        "model": str(
            row.get("model")
            or row.get("planner_model")
            or row.get("llm_model")
            or ""
        ),
        "chat_model_calls": chat_calls,
        "embedding_model_calls": embedding_calls,
        "prep_model_calls": prep_model_calls,
        "prep_prompt_tokens": prep_prompt_tokens,
        "prep_completion_tokens": prep_completion_tokens,
        "prep_total_tokens": prep_total_tokens,
        "model_calls_total": model_calls_total,
        "total_tokens_including_prep": total_tokens_including_prep,
        "memory_status": memory_status,
        "memory_source": memory_source,
        "memory_used": memory_used,
        "memory_hit": memory_hit,
        "memory_covered_steps": memory_covered,
        "memory_total_steps": memory_total,
        "memory_hit_rate": memory_hit_rate,
        "memory_explore_count": memory_explore_count,
        "memory_action_recalled_count": memory_action_recalled_count,
        "memory_action_use_rate": memory_action_use_rate,
        "fallback_steps": fallback_steps,
        "fallback_steps_source": fallback_source,
        "max_fallback_steps": max_fallback_steps,
        "fallback_measurement_status": fallback_status,
        "fallback_budget_exhausted": fallback_budget_exhausted,
        "method_outcome": method_outcome,
        "failure_stage": failure_stage,
        "environment_failure": environment_failure,
        "failure_reason": failure_reason,
    }


def compact_result_row(
    row: dict[str, Any], *, source_seed: int, evaluation_seed: int
) -> dict[str, Any]:
    """Project one detailed result into the compact public result contract."""

    def number(value: Any) -> float:
        try:
            return float(value or 0)
        except (TypeError, ValueError):
            return 0.0

    def integer(value: Any) -> int:
        return int(number(value))

    evidence_paths = [
        str(value)
        for value in row.get("evidence_paths") or ()
        if str(value).strip()
    ]
    for key in (
        "source_run_log", "run_dir", "output_path", "task_results_jsonl",
        "run_log_path", "target_transfer_states_path", "prep_stats_summary",
        "stats_summary", "artifact_ref", "evidence_path", "memory_root",
        "memory_path", "memory_manifest", "prep_manifest", "store_path",
        "function_store_path",
    ):
        value = str(row.get(key) or "").strip()
        if value and value not in evidence_paths:
            evidence_paths.append(value)

    validator_success = row.get("validator_success")
    if "validator_success" not in row:
        validator_success = (
            row.get("official_validator_success")
            if row.get("official_validator_used") is True
            else None
        )
    episode_duration_sec = number(
        row.get("episode_duration_sec") or row.get("duration_sec")
    )
    if episode_duration_sec <= 0:
        episode_duration_sec = number(row.get("duration_ms")) / 1000.0
    result = {
        "task": str(row.get("task") or row.get("task_name") or ""),
        "method": str(row.get("method") or ""),
        "device": str(row.get("device") or ""),
        "source_seed": integer(row.get("source_seed") or source_seed),
        "evaluation_seed": integer(
            row.get("evaluation_seed") or row.get("task_random_seed") or evaluation_seed
        ),
        "status": str(row.get("status") or ""),
        "validator_success": validator_success,
        "official_validator_used": row.get("official_validator_used"),
        "official_validator_coverage_rate": _optional_number(
            row, "official_validator_coverage_rate"
        ),
        "replay_completed": row.get("replay_completed"),
        "model_calls": integer(row.get("model_calls")),
        "model_calls_source": str(row.get("model_calls_source") or ""),
        "prompt_tokens": integer(row.get("prompt_tokens")),
        "completion_tokens": integer(row.get("completion_tokens")),
        "total_tokens": integer(row.get("total_tokens") or row.get("tokens")),
        "actions_executed": integer(row.get("actions_executed")),
        "episode_duration_sec": round(episode_duration_sec, 6),
        "outer_wall_sec": round(number(row.get("outer_wall_sec") or row.get("wall_sec")), 6),
        "error": str(row.get("error") or row.get("failure_summary") or ""),
        "evidence_paths": evidence_paths,
    }
    result.update(function_metrics_from_result_row(row))
    result.update(performance_metrics_from_result_row(row))
    result.update(execution_metrics_from_result_row({**row, **result}))
    result["token_usage_status"] = str(row.get("token_usage_status") or "")
    result["model_calls_source"] = str(row.get("model_calls_source") or "")
    result["official_validator_used"] = row.get("official_validator_used")
    result["replay_completed"] = row.get("replay_completed")
    return {key: result[key] for key in RESULT_FIELDS}
