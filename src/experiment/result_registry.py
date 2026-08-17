#!/usr/bin/env python3
"""Sync immutable AndroidWorld attempt summaries into master progress tables."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import tempfile
from typing import Any

from src.integrations.android_world.methods import reuse_metrics_from_result_row
from src.experiment.protocol import (
    DEVICES,
    FORMAL_METHODS,
    MAX_STEPS,
    SOURCE_SEED,
    TASK_SEED,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_ROOT = REPO_ROOT / "runtime/evals/androidworld_validator/runs"
DEFAULT_MASTER_ROOT = REPO_ROOT / "runtime/evals/androidworld_validator/master_progress"
DEFAULT_SOURCE_INDEX = (
    REPO_ROOT
    / "runtime/evals/androidworld_validator/core_archive/success_source_runlogs/index_by_task.json"
)
FORMAL_DEVICE_TARGETS = {
    label: (serial, port) for label, serial, port in DEVICES
}
FORMAL_SOURCE_SEED = SOURCE_SEED
FORMAL_TASK_SEED = TASK_SEED
FORMAL_MAX_STEPS = MAX_STEPS

METHOD_MATRIX_COLUMNS = [
    "task_index",
    "task_name",
    "method",
    "device_label",
    "official_validator_success",
    "model_calls",
    "total_tokens",
    "duration_sec",
    "task_seed",
    "task_params_json",
    "prep_model_calls",
    "prep_total_tokens",
    "prep_duration_sec",
    "status",
    "app_group",
    "difficulty_level",
    "source_run_log",
    "official_validator_accuracy",
    "prep_type",
    "prep_status",
    "prep_stats_summary",
    "run_label",
    "run_record_id",
    "duration_ms",
    "official_validator_success_count",
    "official_validator_task_count",
    "row_kind",
    "method_label",
    "device_serial",
    "is_latest_for_task_method",
    "planned",
    "completed",
    "source_goal",
    "source_params_json",
    "source_step_count",
    "record_root",
    "eval_summary",
    "stats_summary",
    "stats_jsonl",
    "task_results_jsonl",
    "commands_jsonl",
    "task_elapsed_sec",
    "avg_task_elapsed_sec",
    "actions_executed",
    "artifact_used",
    "reuse_numerator",
    "reuse_denominator",
    "reuse_rate",
    "reuse_unit",
    "reuse_evidence_status",
    "avg_actions_per_task",
    "avg_ms_per_action",
    "chat_model_calls",
    "embedding_model_calls",
    "prompt_tokens",
    "completion_tokens",
    "replay_completed",
    "replay_task_count",
    "replay_completed_count",
    "replay_completed_rate",
    "replay_coverage_rate",
    "replay_step_completed_count",
    "replay_step_total",
    "replay_step_completed_rate",
    "official_validator_coverage_rate",
    "artifact_kind",
    "artifact_ref",
    "error",
    "chat_models",
    "embedding_models",
    "run_delay_sec",
    "devices",
    "run_count",
    "last_run_at",
    "recorded_at",
    "attempt_id",
    "source_seed",
    "evaluation_seed",
    "registration_manifest",
    "source_summary_sha256",
    "notes",
]

RUN_RECORD_COLUMNS = [
    "run_record_id",
    "run_granularity",
    "task_index",
    "task_name",
    "app_group",
    "difficulty_level",
    "method",
    "run_label",
    "device_label",
    "device_serial",
    "status",
    "official_validator_success",
    "official_validator_success_count",
    "official_validator_task_count",
    "duration_ms",
    "actions_executed",
    "artifact_used",
    "reuse_numerator",
    "reuse_denominator",
    "reuse_rate",
    "reuse_unit",
    "reuse_evidence_status",
    "model_calls",
    "chat_model_calls",
    "embedding_model_calls",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "task_elapsed_sec",
    "prep_type",
    "prep_status",
    "prep_model_calls",
    "prep_total_tokens",
    "prep_duration_sec",
    "prep_stats_summary",
    "record_root",
    "eval_summary",
    "stats_summary",
    "stats_jsonl",
    "task_results_jsonl",
    "commands_jsonl",
    "source_run_log",
    "notes",
    "recorded_at",
    "avg_actions_per_task",
    "avg_ms_per_action",
    "replay_completed",
    "replay_task_count",
    "replay_completed_count",
    "replay_completed_rate",
    "replay_coverage_rate",
    "replay_step_completed_count",
    "replay_step_total",
    "replay_step_completed_rate",
    "official_validator_coverage_rate",
    "artifact_kind",
    "artifact_ref",
    "error",
    "official_validator_accuracy",
    "duration_sec",
    "row_kind",
    "method_label",
    "is_latest_for_task_method",
    "planned",
    "completed",
    "source_goal",
    "source_params_json",
    "source_step_count",
    "avg_task_elapsed_sec",
    "chat_models",
    "embedding_models",
    "run_delay_sec",
    "devices",
    "run_count",
    "last_run_at",
    "attempt_id",
    "source_seed",
    "evaluation_seed",
    "registration_manifest",
    "source_summary_sha256",
]

MASTER_PROGRESS_METHODS = FORMAL_METHODS
MASTER_PROGRESS_COMMON_FIELDS = (
    "sr",
    "status",
    "record_root",
    "eval_summary",
    "stats_summary",
    "tool_calls",
    "tokens",
)
MOBILEGPT_PROGRESS_METHODS = (
    "mobilegpt_offline_retrieval",
)
MOBILEGPT_PROGRESS_FIELDS = (
    "official_success_count",
    "official_task_count",
    "stats_jsonl",
    "task_elapsed_sec",
    "chat_models",
    "embedding_models",
    "run_delay_sec",
    "devices",
    "notes",
)

MASTER_PROGRESS_COLUMNS = [
    "task_index",
    "task_name",
    "app_group",
    "difficulty_level",
    "source_run_log",
    "source_goal",
    "source_params_json",
    "source_step_count",
    "latest_official_success_source",
    "accepted_first30",
    *[
        f"{method}_{field}"
        for method in MASTER_PROGRESS_METHODS
        for field in MASTER_PROGRESS_COMMON_FIELDS
    ],
    *[
        f"{method}_{field}"
        for method in MOBILEGPT_PROGRESS_METHODS
        for field in MOBILEGPT_PROGRESS_FIELDS
    ],
    "last_updated_at",
]

METHOD_LABELS = {
    "fixed_replay": "Fixed source-action replay / deterministic replay",
    "mobilegpt_offline_retrieval": (
        "MobileGPT task-local native source-cold memory + stock warm retrieval"
    ),
    "ours": "OmniFlow native E2E / persistent-function path",
    "t3a_hint": "AndroidWorld upstream T3A with source trace hint",
    "appagent_demo": "Pinned stock AppAgent with native human-demo UI docs",
}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    output = []
    with tempfile.SpooledTemporaryFile(
        mode="w+", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
        file.seek(0)
        output.append(file.read())
    _atomic_write_text(path, "".join(output))


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    _atomic_write_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_json(value: Any) -> str:
    if value in (None, "", {}, []):
        return ""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return _stable_json(value)
    return str(value)


def _number(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if number.is_integer():
        return str(int(number))
    return str(round(number, 6)).rstrip("0").rstrip(".")


def _bool_success(value: Any) -> str:
    if value is True or str(value).lower() == "true":
        return "true"
    if value is False or str(value).lower() == "false":
        return "false"
    return ""


def _status_from_row(row: dict[str, Any]) -> str:
    success = _bool_success(row.get("official_validator_success"))
    if success == "true":
        return "official_success"
    if success == "false":
        return "official_failed"
    return _cell(row.get("status")) or "completed"


def _as_repo_relative(path_value: Any) -> str:
    raw = str(path_value or "").strip()
    if not raw:
        return ""
    path = Path(raw)
    if path.is_absolute():
        try:
            return str(path.resolve().relative_to(REPO_ROOT))
        except ValueError:
            return str(path)
    return raw


def _path_cell(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _path_if_exists(base: str, name: str) -> str:
    if not base:
        return ""
    candidate = REPO_ROOT / base / name
    return _path_cell(candidate) if candidate.exists() else ""


def _summary_mtime(path: Path) -> str:
    return dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).isoformat()


def _parse_time(value: str) -> dt.datetime:
    if not value:
        return dt.datetime.min.replace(tzinfo=dt.timezone.utc)
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return dt.datetime.min.replace(tzinfo=dt.timezone.utc)


def _safe_component(value: str, *, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._-")
    return normalized or fallback


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_id(
    task: str,
    method: str,
    device: str,
    summary_path: Path,
    summary: dict[str, Any],
) -> str:
    attempt_id = str(summary.get("attempt_id") or "").strip()
    registration_id = str(summary.get("registration_id") or "").strip()
    if attempt_id or registration_id:
        identity = registration_id or attempt_id
        return (
            f"result.{task}.{method}.{device}."
            f"{_safe_component(identity, fallback=_sha256(summary_path)[:16])}"
        )
    return f"result.{task}.{method}.{device}.{summary_path.stat().st_mtime_ns}"


def _task_sort_key(task_name: str, source_index: dict[str, Any]) -> tuple[int, str]:
    task_names = list(source_index)
    try:
        return task_names.index(task_name) + 1, task_name
    except ValueError:
        return 10_000_000, task_name


def _task_index(task_name: str, source_index: dict[str, Any]) -> str:
    order, _ = _task_sort_key(task_name, source_index)
    return "" if order >= 10_000_000 else str(order)


def _app_group(task_name: str, existing_rows: list[dict[str, str]]) -> str:
    for row in existing_rows:
        if row.get("task_name") == task_name and row.get("app_group"):
            return row["app_group"]
    return ""


def _difficulty(task_name: str, existing_rows: list[dict[str, str]]) -> str:
    for row in existing_rows:
        if row.get("task_name") == task_name and row.get("difficulty_level"):
            return row["difficulty_level"]
    return ""


def _row_from_summary(
    *,
    summary_path: Path,
    summary: dict[str, Any],
    source_index: dict[str, Any],
    existing_rows: list[dict[str, str]],
    source_row: dict[str, Any],
) -> dict[str, str]:
    task_name = str(summary.get("task_name") or source_row.get("task") or "").strip()
    row = dict(source_row)
    method = str(row.get("method") or "").strip()
    device = str(row.get("device") or "").strip()
    record_root = _as_repo_relative(row.get("run_dir") or row.get("output_path"))
    success = _bool_success(row.get("official_validator_success"))
    source_meta = source_index.get(task_name, {})
    recorded_at = _summary_mtime(summary_path)
    duration_sec = _number(row.get("duration_sec") or row.get("wall_sec"))
    duration_ms = _number(row.get("duration_ms"))
    if not duration_ms and duration_sec:
        duration_ms = _number(float(duration_sec) * 1000)
    raw_task_count = row.get("official_validator_task_count")
    task_count = _number(
        raw_task_count
        if raw_task_count is not None
        else int(row.get("official_validator_used") is True)
    )
    success_count = _number(
        row.get("official_validator_success_count")
        if row.get("official_validator_success_count") is not None
        else (1 if success == "true" else 0 if success == "false" else "")
    )
    task_params = row.get("task_params")
    if task_params in (None, "", {}, []):
        task_params = source_meta.get("params")
    task_seed = row.get("task_random_seed") or row.get("task_seed")
    reuse = reuse_metrics_from_result_row(row, method=method)

    out = {
        "task_index": _task_index(task_name, source_index),
        "task_name": task_name,
        "method": method,
        "device_label": device,
        "official_validator_success": success,
        "model_calls": _number(row.get("model_calls")),
        "total_tokens": _number(row.get("total_tokens") or row.get("tokens")),
        "duration_sec": duration_sec,
        "task_seed": _cell(task_seed),
        "task_params_json": _stable_json(task_params),
        "prep_model_calls": _number(row.get("prep_model_calls")),
        "prep_total_tokens": _number(row.get("prep_total_tokens")),
        "prep_duration_sec": _number(row.get("prep_duration_sec")),
        "status": _status_from_row(row),
        "app_group": _app_group(task_name, existing_rows),
        "difficulty_level": _difficulty(task_name, existing_rows),
        "source_run_log": _as_repo_relative(
            row.get("source_run_log") or source_meta.get("retained_source_run_log")
        ),
        "official_validator_accuracy": "1.0"
        if success == "true"
        else "0.0"
        if success == "false"
        else "",
        "prep_type": _cell(row.get("prep_type")),
        "prep_status": _cell(row.get("prep_status")),
        "prep_stats_summary": _as_repo_relative(row.get("prep_stats_summary")),
        "run_label": "result_summary",
        "run_record_id": _record_id(
            task_name,
            method,
            device,
            summary_path,
            summary,
        ),
        "duration_ms": duration_ms,
        "official_validator_success_count": success_count,
        "official_validator_task_count": task_count,
        "row_kind": "device_run",
        "method_label": METHOD_LABELS.get(method, method),
        "device_serial": _cell(row.get("serial")),
        "is_latest_for_task_method": "true",
        "planned": "true",
        "completed": "true",
        "source_goal": _cell(source_meta.get("goal")),
        "source_params_json": _stable_json(source_meta.get("params")),
        "source_step_count": _number(source_meta.get("step_count")),
        "record_root": record_root,
        "eval_summary": _path_if_exists(record_root, "summary.json"),
        "stats_summary": _as_repo_relative(
            row.get("stats_summary") or row.get("mobilegpt_stats_summary")
        ),
        "stats_jsonl": _as_repo_relative(
            row.get("stats_jsonl") or row.get("mobilegpt_stats_jsonl")
        ),
        "task_results_jsonl": _path_if_exists(record_root, "task_results.jsonl"),
        "commands_jsonl": _path_cell(summary_path.with_name("one_task_commands.jsonl"))
        if summary_path.with_name("one_task_commands.jsonl").exists()
        else "",
        "task_elapsed_sec": _number(row.get("task_elapsed_sec") or row.get("wall_sec")),
        "avg_task_elapsed_sec": _number(row.get("avg_task_elapsed_sec")),
        "actions_executed": _number(row.get("actions_executed")),
        "artifact_used": _cell(reuse.get("artifact_used")),
        "reuse_numerator": _number(reuse.get("reuse_numerator")),
        "reuse_denominator": _number(reuse.get("reuse_denominator")),
        "reuse_rate": _number(reuse.get("reuse_rate")),
        "reuse_unit": _cell(reuse.get("reuse_unit")),
        "reuse_evidence_status": _cell(reuse.get("evidence_status")),
        "avg_actions_per_task": _number(row.get("avg_actions_per_task")),
        "avg_ms_per_action": _number(row.get("avg_ms_per_action")),
        "chat_model_calls": _number(row.get("chat_model_calls")),
        "embedding_model_calls": _number(row.get("embedding_model_calls")),
        "prompt_tokens": _number(row.get("prompt_tokens")),
        "completion_tokens": _number(row.get("completion_tokens")),
        "replay_completed": _cell(row.get("replay_completed")),
        "replay_task_count": _number(row.get("replay_task_count")),
        "replay_completed_count": _number(row.get("replay_completed_count")),
        "replay_completed_rate": _number(row.get("replay_completed_rate")),
        "replay_coverage_rate": _number(row.get("replay_coverage_rate")),
        "replay_step_completed_count": _number(row.get("replay_step_completed_count")),
        "replay_step_total": _number(row.get("replay_step_total")),
        "replay_step_completed_rate": _number(row.get("replay_step_completed_rate")),
        "official_validator_coverage_rate": _number(row.get("official_validator_coverage_rate")),
        "artifact_kind": _cell(row.get("artifact_kind")),
        "artifact_ref": _cell(row.get("artifact_ref")),
        "error": _cell(row.get("error")),
        "chat_models": _cell(row.get("chat_models")),
        "embedding_models": _cell(row.get("embedding_models")),
        "run_delay_sec": _number(row.get("run_delay_sec")),
        "devices": device,
        "run_count": "1",
        "last_run_at": recorded_at,
        "recorded_at": recorded_at,
        "attempt_id": _cell(summary.get("attempt_id")),
        "source_seed": _cell(summary.get("source_seed")),
        "evaluation_seed": _cell(summary.get("evaluation_seed") or task_seed),
        "registration_manifest": _as_repo_relative(
            summary.get("registration_manifest")
        ),
        "source_summary_sha256": _cell(summary.get("source_summary_sha256")),
        "notes": "Synced from task-local result summary (legacy one_task_summary.json).",
    }
    return {column: out.get(column, "") for column in METHOD_MATRIX_COLUMNS}


def _load_verified_registered_result(summary_path: Path) -> dict[str, Any]:
    summary = _load_json(summary_path)
    if (
        not isinstance(summary, dict)
        or summary.get("schema_version")
        != "omniflow.androidworld_registered_result.v1"
    ):
        raise ValueError(f"registered result schema invalid: {summary_path}")
    manifest_path = summary_path.with_name("registration_manifest.json")
    recorded_manifest = Path(
        str(summary.get("registration_manifest") or "")
    ).expanduser()
    if not recorded_manifest.is_absolute():
        recorded_manifest = (summary_path.parent / recorded_manifest).resolve()
    else:
        recorded_manifest = recorded_manifest.resolve()
    if recorded_manifest != manifest_path.resolve():
        raise ValueError(
            f"registered result manifest path mismatch: {summary_path}"
        )
    manifest = _load_json(manifest_path)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version")
        != "omniflow.androidworld_result_registration.v1"
        or manifest.get("immutable") is not True
    ):
        raise ValueError(f"registration manifest invalid: {manifest_path}")
    expected_sha256 = str(manifest.get("registered_result_sha256") or "")
    if not expected_sha256 or _sha256(summary_path) != expected_sha256:
        raise ValueError(f"registered result checksum mismatch: {summary_path}")
    for field in (
        "registration_id",
        "attempt_id",
        "task_name",
        "source_seed",
        "evaluation_seed",
    ):
        if str(summary.get(field) or "") != str(manifest.get(field) or ""):
            raise ValueError(
                f"registered result {field} mismatch: {summary_path}"
            )
    rows = [row for row in summary.get("rows") or [] if isinstance(row, dict)]
    if len(rows) != 1:
        raise ValueError(
            f"registered result must contain exactly one row: {summary_path}"
        )
    row = rows[0]
    if (
        str(row.get("method") or "") != str(manifest.get("method") or "")
        or str(row.get("device") or "") != str(manifest.get("device") or "")
    ):
        raise ValueError(f"registered result result mismatch: {summary_path}")
    return summary


def mobilegpt_runtime_integrity_error(value: Any) -> str | None:
    """Classify MobileGPT transport/runtime evidence for run accounting."""

    error = str(value or "").strip()
    if not error or "mobilegpt_step_budget_exhausted" in error:
        return None
    markers = (
        "ConnectionError:",
        "ConnectionRefusedError:",
        "TimeoutError:",
        "mobilegpt_androidworld_state_",
        "mobilegpt_app_ui_not_ready:",
        "mobilegpt_launch_response_invalid:",
        "mobilegpt_native_xml_invalid:",
        "mobilegpt_server_closed_connection",
    )
    return error if any(marker in error for marker in markers) else None


def formal_result_environment_failure_reasons(
    row: dict[str, Any],
) -> tuple[str, ...]:
    """Return explicit runtime/environment failures that forbid registration."""

    if has_official_validator_conclusion(row):
        return ()

    reasons: list[str] = []
    runtime_integrity_error = row.get("runtime_integrity_error")
    if (
        isinstance(runtime_integrity_error, str)
        and runtime_integrity_error.strip()
    ) or (
        runtime_integrity_error is not None
        and not isinstance(runtime_integrity_error, str)
        and bool(runtime_integrity_error)
    ):
        reasons.append("runtime_integrity_error")

    environment_failure = row.get("environment_failure")
    if environment_failure is True or (
        isinstance(environment_failure, str)
        and environment_failure.strip().lower()
        not in {"", "0", "false", "no", "none", "null"}
    ):
        reasons.append("environment_failure")

    for field in (
        "classification",
        "failure_classification",
        "result_classification",
        "status",
    ):
        if str(row.get(field) or "").strip().lower() == "environment_failure":
            reasons.append(field)
    return tuple(dict.fromkeys(reasons))


def has_official_validator_conclusion(row: dict[str, Any]) -> bool:
    return row.get("official_validator_used") is True and isinstance(
        row.get("official_validator_success"), bool
    )


def validate_formal_result_protocol(
    row: dict[str, Any],
    *,
    task_name: str,
    method: str,
    device: str,
    evaluation_seed: int,
    max_steps: int,
) -> None:
    """Reject a registered conclusion produced outside the frozen protocol."""

    violations: list[str] = []
    normalized_device = {
        "target5554": "small5554",
        "target5564": "fold5564",
    }.get(str(row.get("device") or ""), str(row.get("device") or ""))
    if str(row.get("task_name") or task_name) != task_name:
        violations.append("task")
    if str(row.get("method") or "") != method:
        violations.append("method")
    if normalized_device != device:
        violations.append("device")
    expected_device = FORMAL_DEVICE_TARGETS.get(device)
    if expected_device is None:
        violations.append("unsupported_device")
    elif (
        str(row.get("serial") or "") != expected_device[0]
        or row.get("console_port") != expected_device[1]
    ):
        violations.append("device_target")
    if row.get("task_random_seed") != evaluation_seed:
        violations.append("task_random_seed")
    if row.get("fixed_task_seed") is not True:
        violations.append("fixed_task_seed")
    if row.get("fixed_task_params") is not False:
        violations.append("fixed_task_params")
    if row.get("perform_emulator_setup") is not True:
        violations.append("perform_emulator_setup")
    if row.get("state_backend") != "androidworld":
        violations.append("state_backend")
    if method == "fixed_replay":
        if row.get("execution_backend") != "recorded_coordinate_replay_v1":
            violations.append("execution_backend")

    task_params = row.get("task_params")
    params_sha256 = str(row.get("task_params_sha256") or "")
    if not isinstance(task_params, dict) or not re.fullmatch(
        r"[0-9a-f]{64}", params_sha256
    ):
        violations.append("task_params_sha256")
    elif hashlib.sha256(
        json.dumps(
            task_params,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest() != params_sha256:
        violations.append("task_params_hash_mismatch")

    try:
        command = shlex.split(str(row.get("command") or ""))
    except ValueError:
        command = []
        violations.append("command")
    if any(
        token == "--oob-observe-backend"
        or token.startswith("--oob-observe-backend=")
        for token in command
    ):
        violations.append("oob_observe_backend")
    explicit_max_steps = row.get("max_steps")
    if explicit_max_steps is not None and explicit_max_steps != max_steps:
        violations.append("max_steps_metadata")
    recorded_max_steps_value: str | None = None
    for index, token in enumerate(command):
        if token == "--max-steps":
            if index + 1 < len(command):
                recorded_max_steps_value = command[index + 1]
            break
        if token.startswith("--max-steps="):
            recorded_max_steps_value = token.partition("=")[2]
            break
    try:
        recorded_max_steps = int(recorded_max_steps_value or "")
    except ValueError:
        violations.append("max_steps")
    else:
        if recorded_max_steps != max_steps:
            violations.append("max_steps")
    if violations:
        raise ValueError(
            "formal_result_protocol_mismatch:"
            f"{task_name}:{method}:{device}:{','.join(violations)}"
        )


def registered_result_plan(
    *,
    runs_root: Path,
    task_name: str,
    methods: tuple[str, ...],
    devices: tuple[str, ...],
    source_seed: int,
    evaluation_seed: int,
    formal_max_steps: int | None = None,
) -> dict[str, list[tuple[str, str]]]:
    """Return completed and pending formal results in protocol order.

    A result is completed once at least one immutable registered result verifies.
    Official success is deliberately not required: a verified failure is still
    a formal conclusion and must not be rerun by the batch launcher.
    """

    root = runs_root.expanduser().resolve()
    expected = [(method, device) for method in methods for device in devices]
    completed: list[tuple[str, str]] = []
    for method, device in expected:
        cell_root = root / task_name / method / device
        paths = sorted(cell_root.glob("*/registered_result.json"))
        cell_completed = False
        for path in paths:
            summary = _load_verified_registered_result(path)
            row = summary["rows"][0]
            if (
                str(summary.get("task_name") or "") != task_name
                or str(row.get("method") or "") != method
                or str(row.get("device") or "") != device
            ):
                raise ValueError(
                    f"registered result does not match expected result: {path}"
                )
            if (
                summary.get("source_seed") != source_seed
                or summary.get("evaluation_seed") != evaluation_seed
            ):
                continue
            if formal_result_environment_failure_reasons(row):
                continue
            cell_completed = has_official_validator_conclusion(row)
            if not cell_completed:
                continue
            if formal_max_steps is not None:
                validate_formal_result_protocol(
                    row,
                    task_name=task_name,
                    method=method,
                    device=device,
                    evaluation_seed=evaluation_seed,
                    max_steps=formal_max_steps,
                )
            break
        if cell_completed:
            completed.append((method, device))
    return {
        "completed": completed,
        "pending": [result for result in expected if result not in completed],
    }


def load_summary_rows(
    runs_root: Path,
    source_index: dict[str, Any],
    existing_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    registered_paths = sorted(runs_root.rglob("registered_result.json"))
    for summary_path in registered_paths:
        summary = _load_verified_registered_result(summary_path)
        for source_row in summary.get("rows") or []:
            if not isinstance(source_row, dict):
                continue
            if not source_row.get("method") or not source_row.get("device"):
                continue
            if formal_result_environment_failure_reasons(source_row):
                continue
            rows.append(
                _row_from_summary(
                    summary_path=summary_path,
                    summary=summary,
                    source_index=source_index,
                    existing_rows=existing_rows,
                    source_row=source_row,
                )
            )
    return rows


def merge_method_matrix(
    existing_rows: list[dict[str, str]],
    synced_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    synced_ids = {row["run_record_id"] for row in synced_rows}
    retained: list[dict[str, str]] = []
    removed_same_id = 0
    removed_planned = 0
    for row in existing_rows:
        if row.get("run_record_id") in synced_ids:
            removed_same_id += 1
            continue
        key = (row.get("task_name", ""), row.get("method", ""), row.get("device_label", ""))
        if row.get("row_kind") == "planned" and key[:2] in {
            (new_row["task_name"], new_row["method"]) for new_row in synced_rows
        }:
            removed_planned += 1
            continue
        retained.append(row)
    merged = retained + synced_rows
    global_latest_stats = normalize_latest_flags(merged)
    merged.sort(
        key=lambda row: (
            row.get("task_index") or "999999",
            row.get("task_name", ""),
            row.get("method", ""),
            row.get("device_label", ""),
            row.get("recorded_at", ""),
            row.get("run_record_id", ""),
        )
    )
    return merged, {
        "input_rows": len(existing_rows),
        "synced_rows": len(synced_rows),
        "removed_same_id": removed_same_id,
        "removed_planned": removed_planned,
        **global_latest_stats,
        "output_rows": len(merged),
    }


def normalize_latest_flags(rows: list[dict[str, str]]) -> dict[str, int]:
    concrete_by_key: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        if row.get("official_validator_success") not in {"true", "false"}:
            continue
        key = (
            row.get("task_name", ""),
            row.get("method", ""),
            row.get("device_label", ""),
        )
        if not all(key):
            continue
        concrete_by_key.setdefault(key, []).append(row)

    promoted = 0
    demoted = 0
    for key_rows in concrete_by_key.values():
        winner = max(
            key_rows,
            key=lambda row: (
                _parse_time(row.get("recorded_at", "")),
                _parse_time(row.get("last_run_at", "")),
                row.get("run_record_id", ""),
            ),
        )
        for row in key_rows:
            expected = "true" if row is winner else "false"
            if row.get("is_latest_for_task_method") == expected:
                continue
            if expected == "true":
                promoted += 1
            else:
                demoted += 1
            row["is_latest_for_task_method"] = expected
    return {
        "global_latest_promoted": promoted,
        "global_latest_demoted": demoted,
    }


def method_rows_to_run_records(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        if not str(row.get("run_record_id", "")).startswith(("result.", "one_task.")):
            continue
        record = {column: row.get(column, "") for column in RUN_RECORD_COLUMNS}
        record["run_granularity"] = "device"
        out.append(record)
    return out


def merge_run_records(
    existing_rows: list[dict[str, str]],
    synced_records: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    synced_ids = {row["run_record_id"] for row in synced_records}
    retained = [row for row in existing_rows if row.get("run_record_id") not in synced_ids]
    merged = retained + synced_records
    merged.sort(
        key=lambda row: (
            row.get("task_index") or "999999",
            row.get("task_name", ""),
            row.get("method", ""),
            row.get("device_label", ""),
            row.get("recorded_at", ""),
            row.get("run_record_id", ""),
        )
    )
    return merged, {
        "input_rows": len(existing_rows),
        "synced_rows": len(synced_records),
        "removed_same_id": len(existing_rows) - len(retained),
        "output_rows": len(merged),
    }


def _success_rate(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    success = sum(1 for row in rows if row.get("official_validator_success") == "true")
    return str(round(success / len(rows), 6))


def build_master_progress(
    matrix_rows: list[dict[str, str]],
    source_index: dict[str, Any],
    existing_master_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    task_names = list(source_index)
    existing_by_task = {row.get("task_name", ""): row for row in existing_master_rows}
    latest = [
        row
        for row in matrix_rows
        if row.get("official_validator_success") in {"true", "false"}
        and row.get("is_latest_for_task_method") != "false"
    ]
    by_task_method: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in latest:
        by_task_method.setdefault((row["task_name"], row["method"]), []).append(row)

    progress_rows: list[dict[str, str]] = []
    for task_name in task_names:
        source_meta = source_index.get(task_name, {})
        old = existing_by_task.get(task_name, {})
        out = {column: "" for column in MASTER_PROGRESS_COLUMNS}
        out.update(
            {
                "task_index": _task_index(task_name, source_index),
                "task_name": task_name,
                "app_group": old.get("app_group", ""),
                "difficulty_level": old.get("difficulty_level", ""),
                "source_run_log": _as_repo_relative(source_meta.get("retained_source_run_log")),
                "source_goal": _cell(source_meta.get("goal")),
                "source_params_json": _stable_json(source_meta.get("params")),
                "source_step_count": _number(source_meta.get("step_count")),
                "latest_official_success_source": _cell(
                    source_meta.get("latest_official_success_source")
                ),
                "accepted_first30": _cell(source_meta.get("accepted_first30")),
            }
        )
        for method in MASTER_PROGRESS_METHODS:
            rows = by_task_method.get((task_name, method), [])
            out[f"{method}_sr"] = _success_rate(rows)
            if rows:
                rows_sorted = sorted(
                    rows,
                    key=lambda row: (
                        row.get("recorded_at", ""),
                        row.get("device_label", ""),
                    ),
                )
                latest_row = rows_sorted[-1]
                out[f"{method}_status"] = latest_row.get("status", "")
                out[f"{method}_record_root"] = latest_row.get("record_root", "")
                out[f"{method}_eval_summary"] = latest_row.get("eval_summary", "")
                out[f"{method}_stats_summary"] = latest_row.get("stats_summary", "")
                out[f"{method}_tool_calls"] = str(
                    sum(int(float(row.get("model_calls") or 0)) for row in rows)
                )
                out[f"{method}_tokens"] = str(
                    sum(int(float(row.get("total_tokens") or 0)) for row in rows)
                )
            if method in MOBILEGPT_PROGRESS_METHODS and rows:
                out[f"{method}_official_success_count"] = str(
                    sum(
                        int(float(row.get("official_validator_success_count") or 0))
                        for row in rows
                    )
                )
                out[f"{method}_official_task_count"] = str(
                    sum(
                        int(float(row.get("official_validator_task_count") or 0))
                        for row in rows
                    )
                )
                out[f"{method}_stats_jsonl"] = rows[-1].get("stats_jsonl", "")
                out[f"{method}_task_elapsed_sec"] = str(
                    round(sum(float(row.get("task_elapsed_sec") or 0) for row in rows), 6)
                ).rstrip("0").rstrip(".")
                out[f"{method}_chat_models"] = rows[-1].get("chat_models", "")
                out[f"{method}_embedding_models"] = rows[-1].get("embedding_models", "")
                out[f"{method}_run_delay_sec"] = rows[-1].get("run_delay_sec", "")
                out[f"{method}_devices"] = ",".join(
                    sorted(row.get("device_label", "") for row in rows if row.get("device_label"))
                )
                out[f"{method}_notes"] = rows[-1].get("notes", "")
        out["last_updated_at"] = _utc_now()
        progress_rows.append(out)
    return progress_rows


def _validate_no_coverage_regression(
    *,
    source_index: dict[str, Any],
    matrix_existing: list[dict[str, str]],
    master_existing: list[dict[str, str]],
) -> None:
    indexed_tasks = {str(task).strip() for task in source_index if str(task).strip()}
    existing_tasks = {
        str(row.get("task_name") or "").strip()
        for row in [*matrix_existing, *master_existing]
        if str(row.get("task_name") or "").strip()
    }
    missing_tasks = sorted(existing_tasks - indexed_tasks)
    if missing_tasks:
        raise ValueError(
            "source index would drop existing master tasks: "
            + ", ".join(missing_tasks)
        )


def _validate_no_cell_regression(
    *,
    existing_rows: list[dict[str, str]],
    merged_rows: list[dict[str, str]],
) -> None:
    def results(rows: list[dict[str, str]]) -> set[tuple[str, str, str]]:
        return {
            (
                str(row.get("task_name") or "").strip(),
                str(row.get("method") or "").strip(),
                str(row.get("device_label") or "").strip(),
            )
            for row in rows
            if row.get("official_validator_success") in {"true", "false"}
            and str(row.get("task_name") or "").strip()
            and str(row.get("method") or "").strip()
            and str(row.get("device_label") or "").strip()
        }

    missing_cells = sorted(results(existing_rows) - results(merged_rows))
    if missing_cells:
        rendered = ", ".join("/".join(result) for result in missing_cells)
        raise ValueError(f"sync would drop existing result results: {rendered}")


@contextmanager
def _master_lock(master_root: Path):
    master_root.mkdir(parents=True, exist_ok=True)
    lock_path = master_root / ".result_registry.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def update_manifest(
    master_root: Path,
    matrix_rows: list[dict[str, str]],
    run_records: list[dict[str, str]],
    source_index: dict[str, Any],
) -> None:
    path = master_root / "manifest.json"
    manifest = _load_json(path) if path.exists() else {}
    methods = sorted({row.get("method", "") for row in matrix_rows if row.get("method")})
    manifest.update(
        {
            "updated_at": _utc_now(),
            "row_count": len(source_index),
            "method_count": len(methods),
            "method_matrix_row_count": len(matrix_rows),
            "run_record_count": len(run_records),
            "method_matrix_csv": _path_cell(master_root / "androidworld_method_matrix.csv"),
            "method_matrix_jsonl": _path_cell(master_root / "androidworld_method_matrix.jsonl"),
            "run_records_csv": _path_cell(master_root / "androidworld_run_records.csv"),
            "run_records_jsonl": _path_cell(master_root / "androidworld_run_records.jsonl"),
            "methods": [
                {"method": method, "label": METHOD_LABELS.get(method, method)}
                for method in methods
            ],
            "sync": {
                "script": "src/experiment/result_registry.py",
                "source": _path_cell(DEFAULT_RUNS_ROOT),
                "policy": (
                    "Only immutable registered_result.json rows are canonical for "
                    "synced task/method/device results; older registered rows for "
                    "the same key are kept with is_latest_for_task_method=false."
                ),
            },
        }
    )
    _atomic_write_text(
        path,
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    )


def _registration_fingerprint(
    *,
    attempt_manifest_sha256: str,
    source_summary_sha256: str,
    task_name: str,
    method: str,
    device: str,
) -> str:
    payload = "\0".join(
        (
            attempt_manifest_sha256,
            source_summary_sha256,
            task_name,
            method,
            device,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _append_ledger_records(
    path: Path,
    records: list[dict[str, Any]],
) -> int:
    existing_ids: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            registration_id = str(row.get("registration_id") or "").strip()
            if registration_id:
                existing_ids.add(registration_id)
    new_records = [
        row
        for row in records
        if str(row.get("registration_id") or "").strip() not in existing_ids
    ]
    if not new_records:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for row in new_records:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        file.flush()
        os.fsync(file.fileno())
    return len(new_records)


def register_attempt_summary(
    *,
    summary_path: Path,
    attempt_manifest_path: Path,
    runs_root: Path,
    master_root: Path,
    source_index_path: Path,
    artifact_memory_index: Path | None = None,
) -> dict[str, Any]:
    summary_path = summary_path.expanduser().resolve()
    attempt_manifest_path = attempt_manifest_path.expanduser().resolve()
    runs_root = runs_root.expanduser().resolve()
    master_root = master_root.expanduser().resolve()
    source_index_path = source_index_path.expanduser().resolve()
    summary = _load_json(summary_path)
    attempt_manifest = _load_json(attempt_manifest_path)
    if not isinstance(summary, dict):
        raise ValueError(f"summary must be a JSON object: {summary_path}")
    if not isinstance(attempt_manifest, dict) or attempt_manifest.get("immutable") is not True:
        raise ValueError(
            f"attempt manifest must declare immutable=true: {attempt_manifest_path}"
        )

    task_name = str(summary.get("task_name") or "").strip()
    attempt_id = str(attempt_manifest.get("attempt_id") or "").strip()
    if not task_name or not attempt_id:
        raise ValueError("task_name and attempt_id are required for result registration")
    source_summary_sha256 = _sha256(summary_path)
    attempt_manifest_sha256 = _sha256(attempt_manifest_path)
    commands_path = summary_path.with_name("one_task_commands.jsonl")
    registered_at = _utc_now()
    rows = [row for row in summary.get("rows") or [] if isinstance(row, dict)]
    if not rows:
        raise ValueError(f"summary contains no result rows: {summary_path}")
    for row in rows:
        reasons = formal_result_environment_failure_reasons(row)
        if reasons:
            raise ValueError(
                "formal_result_environment_failure:"
                f"{task_name}:{','.join(reasons)}"
            )
        if not has_official_validator_conclusion(row):
            raise ValueError(
                "official_validator_conclusion_missing:"
                f"{task_name}:{row.get('method')}:{row.get('device')}"
            )

    ledger_records: list[dict[str, Any]] = []
    registered_paths: list[str] = []
    with _master_lock(master_root):
        for row in rows:
            method = str(row.get("method") or "").strip()
            device = str(row.get("device") or "").strip()
            if not method or not device:
                raise ValueError(
                    f"result row must contain method and device: {summary_path}"
                )
            fingerprint = _registration_fingerprint(
                attempt_manifest_sha256=attempt_manifest_sha256,
                source_summary_sha256=source_summary_sha256,
                task_name=task_name,
                method=method,
                device=device,
            )
            registration_id = (
                f"{_safe_component(task_name, fallback='task')}."
                f"{_safe_component(method, fallback='method')}."
                f"{_safe_component(device, fallback='device')}."
                f"{_safe_component(attempt_id, fallback=fingerprint[:12])}."
                f"{fingerprint[:12]}"
            )
            destination = (
                runs_root
                / _safe_component(task_name, fallback="task")
                / _safe_component(method, fallback="method")
                / _safe_component(device, fallback="device")
                / _safe_component(attempt_id, fallback=fingerprint[:12])
            )
            registration_manifest_path = destination / "registration_manifest.json"
            registered_result_path = destination / "registered_result.json"
            registration_manifest = {
                "schema_version": "omniflow.androidworld_result_registration.v1",
                "registration_id": registration_id,
                "fingerprint_sha256": fingerprint,
                "immutable": True,
                "task_name": task_name,
                "method": method,
                "device": device,
                "attempt_id": attempt_id,
                "source_seed": attempt_manifest.get("source_seed"),
                "evaluation_seed": attempt_manifest.get("evaluation_seed"),
                "source_summary": str(summary_path),
                "source_summary_sha256": source_summary_sha256,
                "attempt_manifest": str(attempt_manifest_path),
                "attempt_manifest_sha256": attempt_manifest_sha256,
                "source_commands": str(commands_path) if commands_path.exists() else "",
                "source_commands_sha256": _sha256(commands_path)
                if commands_path.exists()
                else "",
                "registered_at": registered_at,
            }
            registered_summary = {
                "schema_version": "omniflow.androidworld_registered_result.v1",
                "registration_id": registration_id,
                "attempt_id": attempt_id,
                "task_name": task_name,
                "task_root": str(destination),
                "source_seed": attempt_manifest.get("source_seed"),
                "evaluation_seed": attempt_manifest.get("evaluation_seed"),
                "source_summary": str(summary_path),
                "source_summary_sha256": source_summary_sha256,
                "registration_manifest": str(registration_manifest_path),
                "rows": [row],
            }
            registered_summary_text = (
                json.dumps(registered_summary, indent=2, ensure_ascii=False) + "\n"
            )
            registered_result_sha256 = hashlib.sha256(
                registered_summary_text.encode("utf-8")
            ).hexdigest()
            registration_manifest["registered_result_sha256"] = (
                registered_result_sha256
            )

            if destination.exists():
                existing_manifest = _load_json(registration_manifest_path)
                if existing_manifest.get("fingerprint_sha256") != fingerprint:
                    raise FileExistsError(
                        f"immutable result registration conflict: {destination}"
                    )
                if _sha256(registered_result_path) != registered_result_sha256:
                    raise ValueError(
                        f"registered result checksum mismatch: {registered_result_path}"
                    )
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = Path(
                    tempfile.mkdtemp(
                        dir=destination.parent,
                        prefix=f".{destination.name}.",
                    )
                )
                try:
                    (temporary / "registration_manifest.json").write_text(
                        json.dumps(
                            registration_manifest,
                            indent=2,
                            ensure_ascii=False,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    (temporary / "registered_result.json").write_text(
                        registered_summary_text,
                        encoding="utf-8",
                    )
                    os.replace(temporary, destination)
                finally:
                    shutil.rmtree(temporary, ignore_errors=True)

            ledger_records.append(
                {
                    **registration_manifest,
                    "registered_result": str(registered_result_path),
                }
            )
            registered_paths.append(str(registered_result_path))

        appended = _append_ledger_records(
            master_root / "androidworld_run_ledger.jsonl",
            ledger_records,
        )
        sync_summary = sync(
            runs_root=runs_root,
            master_root=master_root,
            source_index_path=source_index_path,
            dry_run=False,
            _lock_held=True,
        )

    artifact_memory_updated = False
    if artifact_memory_index is not None:
        from src.experiment.artifact_memory import (
            refresh_artifact_memory_from_pointer,
        )

        experiment_results_root = runs_root.parent.parent
        refresh_artifact_memory_from_pointer(
            memory_index=artifact_memory_index,
            additional_runlog_roots=(experiment_results_root,),
            additional_result_roots=(experiment_results_root,),
        )
        artifact_memory_updated = True

    return {
        "task_name": task_name,
        "attempt_id": attempt_id,
        "registered_cells": len(registered_paths),
        "ledger_records_appended": appended,
        "registered_results": registered_paths,
        "artifact_memory_updated": artifact_memory_updated,
        "sync": sync_summary,
    }


def sync(
    *,
    runs_root: Path,
    master_root: Path,
    source_index_path: Path,
    dry_run: bool,
    _lock_held: bool = False,
) -> dict[str, Any]:
    if not _lock_held:
        with _master_lock(master_root):
            return sync(
                runs_root=runs_root,
                master_root=master_root,
                source_index_path=source_index_path,
                dry_run=dry_run,
                _lock_held=True,
            )

    source_index = _load_json(source_index_path)
    if not isinstance(source_index, dict):
        raise ValueError(f"source index must be a JSON object: {source_index_path}")

    matrix_csv = master_root / "androidworld_method_matrix.csv"
    run_records_csv = master_root / "androidworld_run_records.csv"
    master_progress_csv = master_root / "androidworld_master_progress.csv"

    matrix_columns, matrix_existing = _read_csv(matrix_csv)
    run_record_columns, run_record_existing = _read_csv(run_records_csv)
    master_columns, master_existing = _read_csv(master_progress_csv)

    _validate_no_coverage_regression(
        source_index=source_index,
        matrix_existing=matrix_existing,
        master_existing=master_existing,
    )

    matrix_columns = list(dict.fromkeys([*METHOD_MATRIX_COLUMNS, *matrix_columns]))
    run_record_columns = list(dict.fromkeys([*RUN_RECORD_COLUMNS, *run_record_columns]))
    retired_master_usage_columns = {
        f"{method}_{field}"
        for method in MASTER_PROGRESS_METHODS
        for field in (
            "model_calls",
            "total_tokens",
            "chat_model_calls",
            "embedding_model_calls",
            "prompt_tokens",
            "completion_tokens",
        )
    }
    master_columns = list(
        dict.fromkeys(
            [
                *MASTER_PROGRESS_COLUMNS,
                *(
                    column
                    for column in master_columns
                    if column not in retired_master_usage_columns
                ),
            ]
        )
    )

    synced_rows = load_summary_rows(runs_root, source_index, matrix_existing)
    matrix_rows, matrix_stats = merge_method_matrix(matrix_existing, synced_rows)
    _validate_no_cell_regression(
        existing_rows=matrix_existing,
        merged_rows=matrix_rows,
    )
    synced_records = method_rows_to_run_records(synced_rows)
    run_records, run_record_stats = merge_run_records(run_record_existing, synced_records)
    master_rows = build_master_progress(matrix_rows, source_index, master_existing)

    summary = {
        "summary_files": len(list(runs_root.rglob("one_task_summary.json")))
        + len(list(runs_root.rglob("registered_result.json"))),
        "synced_rows": len(synced_rows),
        "method_matrix": matrix_stats,
        "run_records": run_record_stats,
        "master_progress_rows": len(master_rows),
    }

    if not dry_run:
        _write_csv(matrix_csv, matrix_columns, matrix_rows)
        _write_jsonl(master_root / "androidworld_method_matrix.jsonl", matrix_rows)
        _write_csv(run_records_csv, run_record_columns, run_records)
        _write_jsonl(master_root / "androidworld_run_records.jsonl", run_records)
        _write_csv(master_progress_csv, master_columns, master_rows)
        _write_jsonl(master_root / "androidworld_master_progress.jsonl", master_rows)
        update_manifest(master_root, matrix_rows, run_records, source_index)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync AndroidWorld result summaries into master progress tables."
    )
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--master-root", default=str(DEFAULT_MASTER_ROOT))
    parser.add_argument("--source-index", default=str(DEFAULT_SOURCE_INDEX))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print sync stats without writing files.",
    )
    parser.add_argument(
        "--register-summary",
        default="",
        help="Register one completed result summary (legacy one_task_summary.json) before syncing.",
    )
    parser.add_argument(
        "--attempt-manifest",
        default="",
        help="Immutable attempt_manifest.json used with --register-summary.",
    )
    parser.add_argument(
        "--artifact-memory-index",
        default="",
        help="Optional long-term-memory current.json updated after registration.",
    )
    args = parser.parse_args()
    if args.register_summary:
        if args.dry_run:
            raise ValueError("--register-summary cannot be combined with --dry-run")
        if not args.attempt_manifest:
            raise ValueError("--attempt-manifest is required with --register-summary")
        summary = register_attempt_summary(
            summary_path=Path(args.register_summary),
            attempt_manifest_path=Path(args.attempt_manifest),
            runs_root=Path(args.runs_root),
            master_root=Path(args.master_root),
            source_index_path=Path(args.source_index),
            artifact_memory_index=Path(args.artifact_memory_index)
            if args.artifact_memory_index
            else None,
        )
    else:
        summary = sync(
            runs_root=Path(args.runs_root),
            master_root=Path(args.master_root),
            source_index_path=Path(args.source_index),
            dry_run=bool(args.dry_run),
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
