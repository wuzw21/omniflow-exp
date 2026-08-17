"""Small public result rows and their single detailed evidence companion."""

from __future__ import annotations

from typing import Any

RESULT_FIELDS = (
    "task", "method", "device", "source_seed", "evaluation_seed", "status",
    "validator_success", "model_calls", "prompt_tokens", "completion_tokens",
    "total_tokens", "actions_executed", "episode_duration_sec", "outer_wall_sec",
    "error", "evidence_paths",
)


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

    evidence_paths: list[str] = []
    for key in (
        "source_run_log", "run_dir", "output_path", "task_results_jsonl",
        "target_run_log_path", "target_transfer_states_path", "prep_stats_summary",
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
    return {
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
