"""MobileGPT memory inventory, statistics, and validation."""

from __future__ import annotations

import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from omniflow.core.trajectory import canonicalize_run_log
from src.experiment.mobilegpt_contract import (
    MOBILEGPT_AUDIT_SCHEMA_BY_SCHEMA,
    MOBILEGPT_LEARNING_MODE_BY_SCHEMA,
    MOBILEGPT_MEMORY_MANIFEST,
    MOBILEGPT_MEMORY_SCHEMA,
    MOBILEGPT_SOURCE_METHOD_BY_SCHEMA,
)
from src.experiment.paths import resolve_path
from src.experiment.protocol import SOURCE_SEED
from src.integrations import mobilegpt


def _repo_path(value: str | Path) -> Path:
    return resolve_path(value, root=Path(__file__).resolve().parents[2])


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _rate(numerator: int | float, denominator: int | float) -> float:
    denominator = float(denominator or 0)
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / denominator, 6)


def _iter_jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            yield decoded


def _file_sha256(path: str | Path) -> str:
    resolved = _repo_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"provenance_artifact_missing:{resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "inspect_mobilegpt_memory",
    "mobilegpt_memory_digest",
    "mobilegpt_stats_row_fields",
    "summarize_mobilegpt_stats",
    "validate_mobilegpt_adapted_memory",
]

def summarize_mobilegpt_stats(path: str | Path) -> dict[str, Any]:
    stats_path = _repo_path(path)
    rows = list(_iter_jsonl_rows(stats_path))
    chat_rows = [row for row in rows if row.get("event") == "chat_call"]
    embedding_rows = [row for row in rows if row.get("event") == "embedding_call"]
    # The official learning bridge emits one wrapper lifecycle marker and one
    # MobileGPT task lifecycle marker.  Only the instruction-bearing marker is
    # the actual task boundary; counting both makes a valid session look like
    # two tasks and prevents memory sealing.
    finished_candidates = [
        row
        for row in rows
        if row.get("event") == "task_finished"
    ]
    started_candidates = [
        row
        for row in rows
        if row.get("event") == "task_started"
    ]
    # Newer official utility instrumentation emits an instruction-bearing
    # lifecycle row in addition to the adapter wrapper row.  A clean upstream
    # checkout emits only the wrapper row.  Prefer the richer row when present;
    # otherwise the single wrapper row is the valid official task boundary.
    finished_rows = [
        row
        for row in finished_candidates
        if str(row.get("instruction") or "").strip()
    ] or [
        row
        for row in finished_candidates
        if str(row.get("task_name") or "").strip()
    ] or finished_candidates
    started_rows = [
        row
        for row in started_candidates
        if str(row.get("instruction") or "").strip()
    ] or [
        row
        for row in started_candidates
        if str(row.get("task_name") or "").strip()
    ] or started_candidates
    teacher_rows = [
        row
        for row in rows
        if str(row.get("event") or "").startswith("mobilegpt_teacher_")
    ]
    teacher_preflight_rows = [
        row
        for row in teacher_rows
        if row.get("event") == "mobilegpt_teacher_source_preflight"
    ]
    teacher_miss_rows = [
        row for row in teacher_rows if row.get("event") == "mobilegpt_teacher_miss"
    ]
    memory_rows = [row for row in rows if row.get("event") == "memory_lookup"]
    memory_only_miss_rows = [
        row for row in rows if row.get("event") == "mobilegpt_memory_only_miss"
    ]
    action_rows = [row for row in rows if row.get("event") == "mobilegpt_action_sent"]
    device_action_rows = [
        row for row in action_rows if row.get("is_device_action") is True
    ]
    teacher_expected_action_count = max(
        [
            _coerce_int(row.get("teacher_action_count"))
            for row in teacher_rows
            if row.get("event") == "mobilegpt_teacher_source_preflight"
        ]
        or [0]
    )
    teacher_groundable_action_count = max(
        [
            _coerce_int(
                row.get("groundable_action_count"),
                _coerce_int(row.get("teacher_action_count")),
            )
            for row in teacher_preflight_rows
        ]
        or [teacher_expected_action_count]
    )
    teacher_action_count = sum(
        1 for row in teacher_rows if row.get("event") == "mobilegpt_teacher_action"
    )
    teacher_forced_select_rows = [
        row
        for row in teacher_rows
        if row.get("event") == "mobilegpt_teacher_forced_select"
    ]
    teacher_task_local_forced_select_count = sum(
        1
        for row in teacher_forced_select_rows
        if row.get("scope") == "task" and str(row.get("task_name") or "").strip()
    )
    teacher_skipped_noop_count = sum(
        _coerce_int(row.get("skipped_count"))
        for row in teacher_rows
        if row.get("event") == "mobilegpt_teacher_skipped_noop"
    )
    teacher_vlm_fallback_count = sum(
        1 for row in teacher_miss_rows if row.get("fallback_to_vlm") is True
    )
    teacher_unrecovered_miss_count = sum(
        1 for row in teacher_miss_rows if row.get("fallback_to_vlm") is not True
    )
    teacher_vlm_fallback_enabled = any(
        row.get("fallback_to_vlm_on_teacher_miss") is True
        for row in teacher_preflight_rows
    )
    native_vlm_fallback_only = any(
        row.get("native_vlm_fallback_only") is True
        for row in teacher_preflight_rows
    )
    memory_hit_count = sum(
        1 for row in memory_rows if row.get("result") == "direct_hit"
    )
    memory_explore_count = sum(
        1 for row in memory_rows if row.get("result") == "explore"
    )
    memory_action_recalled_rows = [
        row for row in rows if row.get("event") == "memory_action_recalled"
    ]
    in_context_fallback_count = sum(
        1 for row in memory_rows if row.get("result") == "in_context_fallback"
    )
    derive_fallback_count = sum(
        1 for row in memory_rows if row.get("result") == "derive_fallback"
    )
    prompt_tokens = sum(_coerce_int(row.get("prompt_tokens")) for row in rows)
    completion_tokens = sum(_coerce_int(row.get("completion_tokens")) for row in rows)
    total_tokens = sum(_coerce_int(row.get("total_tokens")) for row in rows)
    model_calls = len(chat_rows) + len(embedding_rows)
    return {
        "schema_version": "omniflow.mobilegpt_stats_summary.v1",
        "stats_path": str(stats_path),
        "event_count": len(rows),
        "task_started_count": len(started_rows),
        "task_finished_count": len(finished_rows),
        "teacher_event_count": len(teacher_rows),
        "teacher_source_preflight_count": len(teacher_preflight_rows),
        "teacher_action_count": teacher_action_count,
        "teacher_expected_action_count": teacher_expected_action_count,
        "teacher_groundable_action_count": teacher_groundable_action_count,
        "teacher_consumed_action_count": (
            teacher_action_count
            + teacher_skipped_noop_count
            + teacher_vlm_fallback_count
        ),
        "teacher_skipped_noop_count": teacher_skipped_noop_count,
        "teacher_miss_count": len(teacher_miss_rows),
        "teacher_vlm_fallback_count": teacher_vlm_fallback_count,
        "teacher_unrecovered_miss_count": teacher_unrecovered_miss_count,
        "teacher_vlm_fallback_enabled": teacher_vlm_fallback_enabled,
        "native_vlm_fallback_only": native_vlm_fallback_only,
        "teacher_failed_finish_count": sum(
            1
            for row in teacher_rows
            if row.get("event") == "mobilegpt_teacher_failed_finish"
        ),
        "teacher_forced_select_count": sum(1 for row in teacher_forced_select_rows),
        "teacher_task_local_forced_select_count": (
            teacher_task_local_forced_select_count
        ),
        "teacher_unsafe_forced_select_count": (
            len(teacher_forced_select_rows) - teacher_task_local_forced_select_count
        ),
        "teacher_action_error_count": sum(
            1
            for row in teacher_rows
            if row.get("event") == "mobilegpt_teacher_action_error"
        ),
        "chat_model_calls": len(chat_rows),
        "embedding_model_calls": len(embedding_rows),
        "model_calls": model_calls,
        "chat_models": sorted(
            {
                str(row.get("model") or "").strip()
                for row in chat_rows
                if str(row.get("model") or "").strip()
            }
        ),
        "chat_attempts": sorted(
            {
                _coerce_int(row.get("attempt"))
                for row in chat_rows
            }
        ),
        "embedding_models": sorted(
            {
                str(row.get("model") or "").strip()
                for row in embedding_rows
                if str(row.get("model") or "").strip()
            }
        ),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "token_usage_status": (
            "tracked"
            if model_calls > 0
            and total_tokens > 0
            and total_tokens == prompt_tokens + completion_tokens
            else "inconsistent"
            if model_calls > 0
            else "not_applicable"
        ),
        "chat_latency_sec": round(
            sum(_coerce_float(row.get("latency_sec")) for row in chat_rows),
            6,
        ),
        "embedding_latency_sec": round(
            sum(_coerce_float(row.get("latency_sec")) for row in embedding_rows),
            6,
        ),
        "task_elapsed_sec": round(
            sum(_coerce_float(row.get("elapsed_sec")) for row in finished_rows),
            6,
        ),
        "finished_tasks": [
            {
                "instruction": row.get("instruction"),
                "elapsed_sec": _coerce_float(row.get("elapsed_sec")),
                "subtask_count": _coerce_int(row.get("subtask_count")),
            }
            for row in finished_rows
        ],
        "memory_lookup_count": len(memory_rows),
        "memory_hit_count": memory_hit_count,
        "memory_hit_rate": _rate(memory_hit_count, len(memory_rows)),
        "memory_explore_count": memory_explore_count,
        "memory_action_recalled_count": len(memory_action_recalled_rows),
        "memory_action_use_rate": _rate(
            len(memory_action_recalled_rows), len(device_action_rows)
        ),
        "fallback_count": in_context_fallback_count + derive_fallback_count,
        "in_context_fallback_count": in_context_fallback_count,
        "derive_fallback_count": derive_fallback_count,
        "memory_only_miss_count": len(memory_only_miss_rows),
        "memory_only_stage_counts": dict(
            Counter(str(row.get("stage") or "unknown") for row in memory_only_miss_rows)
        ),
        "action_sent_count": len(action_rows),
        "actions_executed": len(device_action_rows),
        "action_name_counts": dict(
            Counter(str(row.get("action_name") or "") for row in action_rows)
        ),
    }


def _load_mobilegpt_stats_summary(
    *,
    summary_path: str | Path | None,
    stats_jsonl_path: str | Path | None,
) -> dict[str, Any]:
    summary_value = str(summary_path or "").strip()
    stats_value = str(stats_jsonl_path or "").strip()
    resolved_summary_path = _repo_path(summary_value) if summary_value else None
    resolved_stats_path = _repo_path(stats_value) if stats_value else None
    stats: dict[str, Any] = {}
    if resolved_summary_path is not None and resolved_summary_path.is_file():
        try:
            loaded = json.loads(resolved_summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            loaded = {}
        if isinstance(loaded, dict):
            stats = loaded
    elif resolved_stats_path is not None and resolved_stats_path.is_file():
        stats = summarize_mobilegpt_stats(resolved_stats_path)
    if not stats:
        return {}
    stats.setdefault("stats_path", str(resolved_stats_path or ""))
    stats["summary_path"] = str(resolved_summary_path or "")
    return stats


def mobilegpt_stats_row_fields(
    prefix: str,
    stats: dict[str, Any],
) -> dict[str, Any]:
    return {
        f"{prefix}_model_calls": _coerce_int(stats.get("model_calls")),
        f"{prefix}_chat_model_calls": _coerce_int(stats.get("chat_model_calls")),
        f"{prefix}_embedding_model_calls": _coerce_int(
            stats.get("embedding_model_calls")
        ),
        f"{prefix}_chat_models": list(stats.get("chat_models") or []),
        f"{prefix}_embedding_models": list(stats.get("embedding_models") or []),
        f"{prefix}_prompt_tokens": _coerce_int(stats.get("prompt_tokens")),
        f"{prefix}_completion_tokens": _coerce_int(stats.get("completion_tokens")),
        f"{prefix}_total_tokens": _coerce_int(stats.get("total_tokens")),
        f"{prefix}_token_usage_status": str(stats.get("token_usage_status") or ""),
        f"{prefix}_chat_latency_sec": _coerce_float(stats.get("chat_latency_sec")),
        f"{prefix}_embedding_latency_sec": _coerce_float(
            stats.get("embedding_latency_sec")
        ),
        f"{prefix}_task_elapsed_sec": _coerce_float(stats.get("task_elapsed_sec")),
        f"{prefix}_task_started_count": _coerce_int(stats.get("task_started_count")),
        f"{prefix}_task_finished_count": _coerce_int(stats.get("task_finished_count")),
        f"{prefix}_memory_lookup_count": _coerce_int(stats.get("memory_lookup_count")),
        f"{prefix}_memory_hit_count": _coerce_int(stats.get("memory_hit_count")),
        f"{prefix}_memory_hit_rate": _coerce_float(stats.get("memory_hit_rate")),
        f"{prefix}_memory_explore_count": _coerce_int(
            stats.get("memory_explore_count")
        ),
        f"{prefix}_memory_action_recalled_count": _coerce_int(
            stats.get("memory_action_recalled_count")
        ),
        f"{prefix}_memory_action_use_rate": _coerce_float(
            stats.get("memory_action_use_rate")
        ),
        f"{prefix}_fallback_count": _coerce_int(stats.get("fallback_count")),
        f"{prefix}_in_context_fallback_count": _coerce_int(
            stats.get("in_context_fallback_count")
        ),
        f"{prefix}_derive_fallback_count": _coerce_int(
            stats.get("derive_fallback_count")
        ),
        f"{prefix}_action_sent_count": _coerce_int(stats.get("action_sent_count")),
        f"{prefix}_actions_executed": _coerce_int(stats.get("actions_executed")),
        f"{prefix}_action_name_counts": dict(stats.get("action_name_counts") or {}),
        f"{prefix}_event_count": _coerce_int(stats.get("event_count")),
        f"{prefix}_stats_jsonl": str(stats.get("stats_path") or ""),
        f"{prefix}_stats_summary": str(stats.get("summary_path") or ""),
    }


def inspect_mobilegpt_memory(memory_root: str | Path) -> dict[str, Any]:
    root = _repo_path(memory_root)
    root_task_file = root / "tasks.csv"
    task_files = sorted(root.glob("*/tasks.csv")) if root.exists() else []
    page_files = sorted(root.glob("*/pages.csv")) if root.exists() else []
    hierarchy_files = sorted(root.glob("*/hierarchy.csv")) if root.exists() else []
    subtask_files = sorted(root.glob("*/pages/*/subtasks.csv")) if root.exists() else []
    available_subtask_files = (
        sorted(root.glob("*/pages/*/available_subtasks.csv"))
        if root.exists()
        else []
    )
    action_files = sorted(root.glob("*/pages/*/actions.csv")) if root.exists() else []
    screen_directories = (
        sorted(path for path in root.glob("*/pages/*/screen") if path.is_dir())
        if root.exists()
        else []
    )
    required_screen_files = {
        "screenshot.jpg",
        "raw.xml",
        "html.xml",
        "hierarchy.xml",
        "parsed.xml",
        "pretty.xml",
    }
    virtual_source_screen_files = required_screen_files - {"screenshot.jpg"}
    root_task_rows: list[dict[str, str]] = []
    app_task_rows: list[tuple[str, dict[str, str]]] = []
    task_names: list[str] = []
    root_task_apps: list[str] = []
    app_task_names: list[str] = []
    task_rows = 0
    page_rows = 0
    hierarchy_rows = 0
    page_indexes: set[str] = set()
    hierarchy_indexes: set[str] = set()
    subtask_rows = 0
    action_rows = 0
    non_finish_action_rows = 0
    action_file_rows: list[dict[str, Any]] = []
    subtask_names_by_page: dict[tuple[str, str], set[str]] = {}
    task_path_reference_count = 0
    recallable_task_path_reference_count = 0
    task_path_errors: list[dict[str, str]] = []
    missing_task_path_subtasks: list[dict[str, str]] = []

    if root_task_file.is_file():
        try:
            with root_task_file.open(newline="", encoding="utf-8") as handle:
                root_task_rows = list(csv.DictReader(handle))
        except Exception:
            root_task_rows = []
        task_names = [
            str(row.get("name") or "").strip()
            for row in root_task_rows
            if str(row.get("name") or "").strip()
        ]
        root_task_apps = [
            str(row.get("app") or "").strip()
            for row in root_task_rows
            if str(row.get("app") or "").strip()
        ]

    for task_file in task_files:
        try:
            with task_file.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except Exception:
            rows = []
        task_rows += len(rows)
        app_task_rows.extend((task_file.parent.name, row) for row in rows)
        app_task_names.extend(
            str(row.get("name") or "").strip()
            for row in rows
            if str(row.get("name") or "").strip()
        )

    for page_file in page_files:
        try:
            with page_file.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except Exception:
            rows = []
        page_rows += len(rows)
        page_indexes.update(
            str(row.get("index") or "").strip()
            for row in rows
            if str(row.get("index") or "").strip()
        )

    for hierarchy_file in hierarchy_files:
        try:
            with hierarchy_file.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except Exception:
            rows = []
        hierarchy_rows += len(rows)
        hierarchy_indexes.update(
            str(row.get("index") or "").strip()
            for row in rows
            if str(row.get("index") or "").strip()
        )

    for subtask_file in subtask_files:
        try:
            with subtask_file.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except Exception:
            rows = []
        subtask_rows += len(rows)
        subtask_names_by_page[
            (subtask_file.parents[2].name, subtask_file.parent.name)
        ] = {
            str(row.get("name") or "").strip()
            for row in rows
            if str(row.get("name") or "").strip()
        }

    native_primitive_subtasks = {"finish", "scroll_screen"}
    for app_name, row in app_task_rows:
        task_name = str(row.get("name") or "").strip()
        raw_path = str(row.get("path") or "").strip()
        try:
            task_path = json.loads(raw_path)
        except (TypeError, ValueError):
            task_path_errors.append(
                {
                    "app": app_name,
                    "task_name": task_name,
                    "reason": "invalid_json",
                }
            )
            continue
        if not isinstance(task_path, dict):
            task_path_errors.append(
                {
                    "app": app_name,
                    "task_name": task_name,
                    "reason": "path_not_object",
                }
            )
            continue
        for raw_page_index, raw_subtasks in task_path.items():
            page_index = str(raw_page_index).strip()
            if not isinstance(raw_subtasks, list):
                task_path_errors.append(
                    {
                        "app": app_name,
                        "task_name": task_name,
                        "page_index": page_index,
                        "reason": "subtasks_not_list",
                    }
                )
                continue
            page_subtasks = subtask_names_by_page.get((app_name, page_index), set())
            for raw_subtask_name in raw_subtasks:
                subtask_name = str(raw_subtask_name or "").strip()
                task_path_reference_count += 1
                if subtask_name in native_primitive_subtasks:
                    continue
                recallable_task_path_reference_count += 1
                if not subtask_name or subtask_name not in page_subtasks:
                    missing_task_path_subtasks.append(
                        {
                            "app": app_name,
                            "task_name": task_name,
                            "page_index": page_index,
                            "subtask_name": subtask_name,
                        }
                    )

    for action_file in action_files:
        file_action_rows = 0
        file_non_finish_rows = 0
        try:
            with action_file.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except Exception:
            rows = []
        for row in rows:
            if not any(str(value or "").strip() for value in row.values()):
                continue
            file_action_rows += 1
            action_text = str(row.get("action") or "")
            if (
                '"name": "finish"' not in action_text
                and "'name': 'finish'" not in action_text
            ):
                file_non_finish_rows += 1
        action_rows += file_action_rows
        non_finish_action_rows += file_non_finish_rows
        if file_action_rows:
            action_file_rows.append(
                {
                    "path": str(action_file),
                    "action_rows": file_action_rows,
                    "non_finish_action_rows": file_non_finish_rows,
                }
            )

    page_directories = {
        path.parent.name
        for path in subtask_files + available_subtask_files + action_files
    }
    complete_screen_directories = sum(
        1
        for directory in screen_directories
        if required_screen_files.issubset(
            {
                path.name
                for path in directory.iterdir()
                if path.is_file() and path.stat().st_size > 0
            }
        )
    )
    complete_virtual_source_screen_directories = sum(
        1
        for directory in screen_directories
        if virtual_source_screen_files.issubset(
            {
                path.name
                for path in directory.iterdir()
                if path.is_file() and path.stat().st_size > 0
            }
        )
    )
    task_local_memory = bool(
        root_task_file.is_file()
        and len(root_task_rows) == 1
        and len(task_files) == 1
        and task_rows == 1
        and task_names == app_task_names
        and root_task_apps == [task_files[0].parent.name]
    )
    native_memory_complete = bool(
        task_local_memory
        and len(page_files) == 1
        and len(hierarchy_files) == 1
        and page_rows > 0
        and hierarchy_rows > 0
        and page_indexes == hierarchy_indexes == page_directories
        and len(subtask_files) == page_rows
        and len(available_subtask_files) == page_rows
        and len(action_files) == page_rows
        and len(screen_directories) == page_rows
        and complete_screen_directories == page_rows
        and task_path_reference_count > 0
        and recallable_task_path_reference_count > 0
        and not task_path_errors
        and not missing_task_path_subtasks
    )
    virtual_source_memory_complete = bool(
        task_local_memory
        and len(page_files) == 1
        and len(hierarchy_files) == 1
        and page_rows > 0
        and hierarchy_rows > 0
        and page_indexes == hierarchy_indexes == page_directories
        and len(subtask_files) == page_rows
        and len(available_subtask_files) == page_rows
        and len(action_files) == page_rows
        and len(screen_directories) == page_rows
        and complete_virtual_source_screen_directories == page_rows
        and task_path_reference_count > 0
        and recallable_task_path_reference_count > 0
        and not task_path_errors
        and not missing_task_path_subtasks
    )

    return {
        "memory_root": str(root),
        "root_task_file_count": int(root_task_file.is_file()),
        "root_task_rows": len(root_task_rows),
        "root_task_names": task_names,
        "root_task_apps": root_task_apps,
        "task_file_count": len(task_files),
        "task_rows": task_rows,
        "app_task_names": app_task_names,
        "task_local_memory": task_local_memory,
        "task_path_reference_count": task_path_reference_count,
        "recallable_task_path_reference_count": recallable_task_path_reference_count,
        "task_path_errors": task_path_errors,
        "missing_task_path_subtasks": missing_task_path_subtasks,
        "page_file_count": len(page_files),
        "page_rows": page_rows,
        "page_indexes": sorted(page_indexes),
        "hierarchy_file_count": len(hierarchy_files),
        "hierarchy_rows": hierarchy_rows,
        "hierarchy_indexes": sorted(hierarchy_indexes),
        "native_memory_complete": native_memory_complete,
        "virtual_source_memory_complete": virtual_source_memory_complete,
        "subtask_file_count": len(subtask_files),
        "subtask_rows": subtask_rows,
        "available_subtask_file_count": len(available_subtask_files),
        "action_file_count": len(action_files),
        "action_rows": action_rows,
        "non_finish_action_rows": non_finish_action_rows,
        "screen_directory_count": len(screen_directories),
        "complete_screen_directory_count": complete_screen_directories,
        "complete_virtual_source_screen_directory_count": (
            complete_virtual_source_screen_directories
        ),
        "screen_file_count": sum(
            1
            for directory in screen_directories
            for path in directory.iterdir()
            if path.is_file()
        ),
        "has_recallable_subtasks": subtask_rows > 0,
        "has_useful_actions": non_finish_action_rows > 0 and subtask_rows > 0,
        "action_files": action_file_rows,
    }


def mobilegpt_memory_digest(memory_root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    file_count = 0
    for path in sorted(memory_root.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(memory_root).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
        file_count += 1
    return digest.hexdigest(), file_count


def _mobilegpt_manifest_evidence_path(
    bundle_root: Path,
    record: Any,
    *,
    label: str,
) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"mobilegpt_cold_memory_{label}_record_invalid")
    relative_value = str(record.get("relative_path") or "").strip()
    if not relative_value:
        raise ValueError(f"mobilegpt_cold_memory_{label}_path_missing")
    path = (bundle_root / relative_value).resolve()
    try:
        path.relative_to(bundle_root)
    except ValueError as error:
        raise ValueError(f"mobilegpt_cold_memory_{label}_outside_bundle") from error
    if not path.is_file():
        raise ValueError(f"mobilegpt_cold_memory_{label}_missing:{path}")
    expected_sha256 = str(record.get("sha256") or "").strip()
    if not expected_sha256 or _file_sha256(path) != expected_sha256:
        raise ValueError(f"mobilegpt_cold_memory_{label}_hash_mismatch")
    return path


def _official_source_result_summary(
    result_path: Path,
    *,
    task_name: str,
) -> dict[str, Any]:
    rows = [
        row
        for row in _iter_jsonl_rows(result_path)
        if str(row.get("task_name") or row.get("task") or "") == str(task_name)
    ]
    if not rows:
        raise ValueError("mobilegpt_source_result_task_missing")

    def official_used(row: dict[str, Any]) -> bool:
        if "official_validator_used" in row:
            return bool(row.get("official_validator_used"))
        if "uses_androidworld_official_validator" in row:
            return bool(row.get("uses_androidworld_official_validator"))
        validator = row.get("androidworld_validator_result")
        return bool(
            isinstance(validator, dict)
            and (
                validator.get("uses_androidworld_official_validator") is True
                or validator.get("validator") == "androidworld_official"
            )
        )

    official_rows = [row for row in rows if official_used(row)]
    if not official_rows:
        raise ValueError("mobilegpt_source_official_validator_required")

    def successful(row: dict[str, Any]) -> bool:
        validator = row.get("androidworld_validator_result")
        if isinstance(validator, dict) and "success" in validator:
            return bool(validator.get("success"))
        if "official_validator_success" in row:
            return bool(row.get("official_validator_success"))
        return bool(row.get("success"))

    return {
        "row_count": len(rows),
        "official_validator_used": True,
        "official_validator_success": any(successful(row) for row in official_rows),
    }


def _validate_mobilegpt_converted_memory(
    root: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    *,
    task_name: str,
    source_seed: int,
    source_run_log: str | Path,
    compatible_source_sha256s: Sequence[str],
    expected_model: str,
    expected_source_method: str,
) -> dict[str, Any]:
    schema_version = str(manifest.get("schema_version") or "")
    try:
        schema_source_method = MOBILEGPT_SOURCE_METHOD_BY_SCHEMA[schema_version]
        schema_learning_mode = MOBILEGPT_LEARNING_MODE_BY_SCHEMA[schema_version]
        schema_audit = MOBILEGPT_AUDIT_SCHEMA_BY_SCHEMA[schema_version]
    except KeyError as error:
        raise ValueError("mobilegpt_virtual_memory_schema_invalid") from error
    if str(manifest.get("task_name") or "") != str(task_name):
        raise ValueError("mobilegpt_virtual_memory_task_name_mismatch")
    if int(manifest.get("source_seed") or -1) != int(source_seed):
        raise ValueError("mobilegpt_virtual_memory_source_seed_mismatch")
    if int(source_seed) != SOURCE_SEED:
        raise ValueError("mobilegpt_virtual_memory_requires_source_seed_111")
    source_method = str(manifest.get("source_method") or "").strip()
    if source_method != schema_source_method:
        raise ValueError("mobilegpt_virtual_memory_source_method_invalid")
    normalized_expected_source_method = str(expected_source_method or "").strip()
    if (
        normalized_expected_source_method
        and source_method != normalized_expected_source_method
    ):
        raise ValueError(
            "mobilegpt_virtual_memory_source_method_mismatch:"
            f"expected={normalized_expected_source_method}:actual={source_method}"
        )

    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("mobilegpt_virtual_memory_provenance_missing")
    native_learning = bool(provenance.get("native_mobilegpt_learning"))
    required_provenance = {
        "native_mobilegpt_learning": native_learning,
        "task_local_memory": True,
        "learning_mode": schema_learning_mode,
        "teacher_forcing": False,
        "synthetic_subtasks": not native_learning,
        "semantic_subtasks": native_learning,
        "original_mobilegpt_prompts": native_learning,
        "actions_supplied_to_mobilegpt": not native_learning,
        "source_transitions_supplied": True,
        "source_success_boundary_supplied": True,
        "runlog_transition_compilation": not native_learning,
        "complete_transition_mapping": not native_learning,
        "official_reader_validation": True,
        "function_store_used": False,
        "function_conversion_enabled": False,
        "target_inputs_read": False,
        "target_observations_read": False,
        "validator_state_read": False,
        "coordinate_replay": False,
        "source_emulator_used": False,
    }
    for provenance_field, expected in required_provenance.items():
        if (
            provenance.get(provenance_field) is not expected
            and provenance.get(provenance_field) != expected
        ):
            raise ValueError(
                "mobilegpt_virtual_memory_provenance_invalid:"
                f"{provenance_field}"
            )
    if "official_source_result" in manifest:
        raise ValueError("mobilegpt_virtual_memory_official_source_result_forbidden")

    memory_record = manifest.get("memory")
    if not isinstance(memory_record, dict):
        raise ValueError("mobilegpt_virtual_memory_record_missing")
    expected_memory_path = (
        root.parent / str(memory_record.get("relative_path") or "")
    ).resolve()
    if expected_memory_path != root:
        raise ValueError("mobilegpt_virtual_memory_path_mismatch")
    actual_digest, actual_file_count = mobilegpt_memory_digest(root)
    if actual_file_count != int(memory_record.get("file_count") or -1):
        raise ValueError("mobilegpt_virtual_memory_file_count_mismatch")
    inventory = inspect_mobilegpt_memory(root)
    if inventory.get("task_local_memory") is not True:
        raise ValueError("mobilegpt_virtual_memory_not_task_local")
    if inventory.get("virtual_source_memory_complete") is not True:
        raise ValueError("mobilegpt_virtual_memory_graph_incomplete")
    if not inventory.get("has_recallable_subtasks"):
        raise ValueError("mobilegpt_virtual_memory_missing_recallable_subtasks")
    if not inventory.get("has_useful_actions"):
        raise ValueError("mobilegpt_virtual_memory_missing_useful_actions")

    bundle_root = root.parent.resolve()
    source_log_record = manifest.get("source_run_log")
    source_log_path = _mobilegpt_manifest_evidence_path(
        bundle_root,
        source_log_record,
        label="source_run_log",
    )
    try:
        source_payload = canonicalize_run_log(
            json.loads(source_log_path.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("mobilegpt_virtual_memory_source_run_log_invalid") from error
    recorded_source_seed = source_payload.get("seed")
    source_validator = source_payload.get("validator")
    if (
        str(source_payload.get("task_name") or "") != str(task_name)
        or type(recorded_source_seed) is not int
        or source_payload.get("status") != "succeeded"
        or source_payload.get("success") is not True
        or not isinstance(source_validator, dict)
        or source_validator.get("official") is not True
        or source_validator.get("success") is not True
        or source_log_record.get("recorded_seed") != recorded_source_seed
    ):
        raise ValueError("mobilegpt_virtual_memory_source_run_log_invalid")
    audit_record = manifest.get("trajectory_audit")
    audit_path = _mobilegpt_manifest_evidence_path(
        bundle_root,
        audit_record,
        label="trajectory_audit",
    )
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("mobilegpt_virtual_memory_audit_invalid_json") from error
    if not isinstance(audit, dict) or audit.get("schema_version") != schema_audit:
        raise ValueError("mobilegpt_virtual_memory_audit_invalid")
    transition_count = int(audit.get("transition_count") or 0)
    validated_count = int(audit.get("validated_transition_count") or 0)
    validation_rows = audit.get("validation_rows")
    direct_audit_valid = not (
        str(audit.get("task_name") or "") != str(task_name)
        or transition_count <= 0
        or validated_count != transition_count
        or not isinstance(validation_rows, list)
        or not validation_rows
        or any(row.get("matched") is not True for row in validation_rows if isinstance(row, dict))
        or any(not isinstance(row, dict) for row in validation_rows)
        or sum(int(row.get("consumed_transitions") or 0) for row in validation_rows)
        != transition_count
        or audit.get("actions_supplied_to_mobilegpt") is not True
        or audit.get("source_transitions_supplied") is not True
        or audit.get("source_success_boundary_supplied") is not True
        or audit.get("complete") is not True
    )
    official_audit_valid = (
        str(audit.get("task_name") or "") == str(task_name)
        and audit.get("conversion_mode") == "official_mobilegpt_learning"
        and audit.get("official_server_finished") is True
        and audit.get("actions_supplied_to_mobilegpt") is False
        and audit.get("source_transitions_supplied") is True
        and audit.get("source_success_boundary_supplied") is True
        and audit.get("complete") is True
    )
    if (native_learning and not official_audit_valid) or (
        not native_learning and not direct_audit_valid
    ):
        raise ValueError("mobilegpt_virtual_memory_trajectory_incomplete")
    official_reader = audit.get("official_reader_validation")
    if (
        not isinstance(official_reader, dict)
        or official_reader.get("loadable") is not True
        or int(official_reader.get("page_count") or 0) <= 0
        or (
            not native_learning
            and int(official_reader.get("task_path_pages") or 0) <= 0
        )
    ):
        raise ValueError("mobilegpt_virtual_memory_official_reader_invalid")
    success_boundary = audit.get("source_success_boundary")
    if (
        not isinstance(success_boundary, dict)
        or success_boundary.get("status") != "succeeded"
        or success_boundary.get("success") is not True
    ):
        raise ValueError("mobilegpt_virtual_memory_source_boundary_invalid")

    source_stats = manifest.get("source_stats")
    source_stats_path = _mobilegpt_manifest_evidence_path(
        bundle_root,
        source_stats,
        label="source_stats",
    )
    stats_summary = summarize_mobilegpt_stats(source_stats_path)
    if (
        int(stats_summary.get("task_started_count") or 0) != 1
        or int(stats_summary.get("task_finished_count") or 0) != 1
    ):
        raise ValueError("mobilegpt_virtual_memory_task_lifecycle_incomplete")
    if not native_learning and int(stats_summary.get("chat_model_calls") or 0) != 0:
        raise ValueError("mobilegpt_memory_chat_calls_forbidden")
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _file_sha256(manifest_path),
        "memory_root": str(root),
        "memory_sha256": actual_digest,
        "memory_file_count": actual_file_count,
        "memory_inventory": inventory,
        "source_stats_summary": stats_summary,
        "source_memory_write_status": {
            "memory_written": True,
            "trajectory_transition_count": transition_count,
            "trajectory_validated_transition_count": validated_count,
            "task_started_count": 1,
            "task_finished_count": 1,
        },
        "target_package": str(manifest.get("target_package") or ""),
        "target_app": str(manifest.get("target_app") or ""),
    }


def validate_mobilegpt_adapted_memory(
    memory_root: str | Path,
    *,
    task_name: str,
    source_seed: int,
    source_run_log: str | Path,
    compatible_source_sha256s: Sequence[str] = (),
    expected_model: str = "",
    expected_source_method: str = "",
) -> dict[str, Any]:
    """Validate one sealed RunLog-taught native MobileGPT memory tree."""

    root = _repo_path(memory_root)
    if not root.is_dir():
        raise FileNotFoundError(f"mobilegpt_source_memory_missing:{root}")
    manifest_path = root.parent / MOBILEGPT_MEMORY_MANIFEST
    if not manifest_path.is_file():
        raise ValueError(f"mobilegpt_memory_manifest_missing:{manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("mobilegpt_cold_memory_manifest_invalid_json") from error
    if not isinstance(manifest, dict):
        raise ValueError("mobilegpt_cold_memory_manifest_invalid")
    if manifest.get("schema_version") != MOBILEGPT_MEMORY_SCHEMA:
        raise ValueError("mobilegpt_cold_memory_manifest_schema_invalid")
    validated = _validate_mobilegpt_converted_memory(
        root,
        manifest,
        manifest_path,
        task_name=task_name,
        source_seed=source_seed,
        source_run_log=source_run_log,
        compatible_source_sha256s=compatible_source_sha256s,
        expected_model=expected_model,
        expected_source_method=expected_source_method,
    )
    from src.integrations.mobilegpt import validate_mobilegpt_memory

    validated["memory_validation"] = mobilegpt.validate_mobilegpt_memory(root)
    return validated
