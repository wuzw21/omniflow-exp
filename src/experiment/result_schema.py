"""Small public result rows and their single detailed evidence companion."""

from __future__ import annotations

from typing import Any

RESULT_FIELDS = (
    "task", "method", "device", "source_seed", "evaluation_seed", "status",
    "validator_success", "model_calls", "prompt_tokens", "completion_tokens",
    "total_tokens", "actions_executed", "episode_duration_sec", "outer_wall_sec",
    "function_hit", "function_covered_steps", "function_total_steps",
    "function_step_coverage_rate", "vlm_calls", "vlm_latency_ms", "latency_sec",
    "energy_mwh", "energy_measurement_available", "error", "evidence_paths",
)


def function_metrics_from_result_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize Function reuse into task- and step-level public metrics."""

    reuse = row.get("reuse_metrics")
    reuse = reuse if isinstance(reuse, dict) else row
    method = str(row.get("method") or row.get("agent") or "").strip()
    function_id = str(row.get("function_id") or "").strip()
    artifact_used = bool(reuse.get("artifact_used"))
    covered_steps = max(0, int(float(reuse.get("reuse_numerator") or 0)))
    total_steps = max(0, int(float(reuse.get("reuse_denominator") or 0)))
    # Only a real Function-backed trace counts. Fixed replay and hint injection
    # reuse artifacts, but they do not constitute a Function hit.
    function_backed = method == "omniflow" or bool(function_id)
    hit = bool(function_backed and artifact_used and covered_steps > 0)
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
        "stats_summary", "artifact_ref", "evidence_path",
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
        "model_calls": integer(row.get("model_calls")),
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
    return {key: result[key] for key in RESULT_FIELDS}
