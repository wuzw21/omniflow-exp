#!/usr/bin/env python3
"""Generate auditable per-task AndroidWorld execution tables.

The runner owns experiment execution.  This script is deliberately read-only:
it reads only ``runlog/current`` records and emits stable CSV/Markdown tables.
It does not scan attempts or change any experiment asset. A caller may pass
archived evidence explicitly with repeated ``--evidence`` arguments; those
files are validated and merged into one canonical row per method/device. One
task produces one summary table, one step table, and one Markdown view.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATA_ROOT = Path("data/androidworld")
# The current experiment contract reports the OmniFlow method only.  Other
# methods are separate experiments and must not appear as fabricated
# ``missing_current`` rows in an OmniFlow task report.
DEFAULT_METHODS = ("omniflow",)
SOURCE_METHOD = "source"
EXPECTED_TARGET_SETTINGS = (
    "OmniFlowTargetPixel6Pro_seed111_eval113",
    "OmniFlowTargetFold_seed111_eval113",
    "WXGA_Tablet_test_00_seed111_eval113",
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _last_jsonl(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _nested_number(value: Any, *keys: str) -> float:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return 0.0
        current = current.get(key)
    return _number(current)


def _device_from_setting(setting: Path) -> str:
    return setting.name


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _method_dirs(task_root: Path, methods: Iterable[str]) -> Iterable[tuple[str, Path]]:
    allowed = set(methods)
    for method_dir in sorted(task_root.iterdir() if task_root.is_dir() else ()):
        if not method_dir.is_dir() or method_dir.name not in allowed:
            continue
        yield method_dir.name, method_dir


def _first(*values: Any) -> Any:
    """Preserve measured zero/False; missing measurements stay missing."""
    return next((v for v in values if v is not None and v != ""), "")


def _seconds(value: Any) -> float | str:
    return round(float(value) / 1000, 6) if isinstance(value, (int, float)) else ""


def _evidence_payload(run_path: Path, result_path: Path) -> tuple[dict, dict]:
    result = _last_jsonl(result_path)
    run = _read_json(run_path) or _dict(result.get("canonical_run"))
    return run, result


def _record_from_paths(
    *, task: str, method: str, device: str, run_path: Path,
    result_path: Path, data_root: Path,
) -> dict[str, Any]:
    import hashlib

    run, result = _evidence_payload(run_path, result_path)
    reported_task = _first(result.get("task_name"), result.get("task"), run.get("task_name"))
    if reported_task and reported_task != task:
        raise ValueError(f"task mismatch: requested {task}, evidence {reported_task}")
    diag = _dict(run.get("diagnostics"))
    execution = _dict(diag.get("execution_summary"))
    usage = _dict(diag.get("llm_usage")) or _dict(result.get("llm_usage"))
    components = _dict(usage.get("by_component"))
    reuse = _dict(result.get("reuse_metrics"))
    validator = _dict(result.get("androidworld_validator_result")) or _dict(run.get("validator"))
    official = _first(result.get("official_validator_used"), validator.get("official"),
                      validator.get("uses_androidworld_official_validator"))
    valid = _first(validator.get("success"), result.get("official_validator_success"))
    success = _first(result.get("success"), valid if official is True else "", run.get("success"))
    status = _first(run.get("status"), result.get("classification"),
                    "succeeded" if success is True else "failed" if success is False else "unknown")
    function = _dict(diag.get("function_execution"))
    resolution = _dict(diag.get("function_resolution"))
    resume = _dict(result.get("function_resume")) or _dict(diag.get("function_resume"))
    planner = _dict(diag.get("planner_diagnostics"))
    trace = _list(diag.get("execution_trace"))
    steps = _list(run.get("steps"))
    func_steps = _list(function.get("steps"))
    evidence = result_path if result_path.is_file() else run_path
    reported_error = _first(
        result.get("error"), result.get("failure_reason"),
        result.get("runtime_integrity_error"), execution.get("failure_reason")
    )
    # A successful official validator result may retain a recoverable planner
    # warning in the raw receipt.  Keep that warning in raw_result_json and
    # diagnostics, but do not label the completed cell as failed evidence.
    if success is True and official is True:
        reported_error = ""
    row = {key: "" for key in SUMMARY_FIELDS}
    row.update({
        "task": task, "method": method, "device": device,
        "setting_dir": f"{task}/{method}/{device}",
        "run_log_path": _relative(run_path, data_root),
        "task_results_path": _relative(result_path, data_root),
        "evidence_path": _relative(evidence, data_root),
        "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        "evidence_kind": "archive" if ".archive" in evidence.parts else "current",
        "run_id": _first(result.get("artifact_ref"), run.get("run_id")),
        "started_at": _first(result.get("started_at"), run.get("started_at")),
        "success": success, "status": status,
        "official_validator_used": official, "validator_success": valid,
        "validator_reward": validator.get("reward", ""),
        "duration_sec": _seconds(result.get("duration_ms")),
        "execution_sec": _seconds(_first(result.get("execution_duration_ms"), execution.get("execution_duration_ms"))),
        "non_execution_sec": _seconds(result.get("non_execution_duration_ms")),
        "duration_scope": result.get("duration_scope", "reported_duration" if "duration_ms" in result else ""),
        "execution_scope": result.get("execution_duration_scope", ""),
        "model": _first(result.get("model"), execution.get("model")),
        "model_base_url": result.get("model_base_url", ""),
        "vlm_latency_sec": _seconds(result.get("vlm_latency_ms")),
        "performance_sidecar_path": result.get("performance_sidecar_path", ""),
        "step_count": _first(result.get("step_count"), len(steps) if steps else ""),
        "physical_action_count": reuse.get("physical_action_count", ""),
        "state_changing_action_count": reuse.get("state_changing_physical_action_count", ""),
        "function_id": _first(result.get("function_id"), diag.get("function_id")),
        "function_replay_status": _first(function.get("replay_status"), resolution.get("replay_status")),
        "function_step_count": len(func_steps) if function else "",
        "function_success_steps": sum(s.get("success") is True for s in func_steps) if function else "",
        "function_failed_steps": sum(s.get("success") is False for s in func_steps) if function else "",
        "function_resume_attempts": resume.get("attempt_count", ""),
        "function_resume_successes": resume.get("success_count", ""),
        "router_calls": _dict(components.get("function_router")).get("model_calls", ""),
        "planner_calls": _first(_dict(components.get("planner")).get("model_calls"),
                                len(planner["planner_calls"]) if "planner_calls" in planner else ""),
        "planner_turns_exposed": len(planner["planner_calls"]) if "planner_calls" in planner else "",
        "checker_trigger_counts_json": _json(diag.get("checker_trigger_counts", {})),
        "function_resolution_json": _json(resolution),
        "function_execution_json": _json(function),
        "function_resume_json": _json(resume), "llm_usage_json": _json(usage),
        "planner_diagnostics_json": _json(planner),
        "performance_metrics_json": _json(result.get("performance_metrics", {})),
        "task_params_json": _json(_first(result.get("task_params"), run.get("task_parameters")) or {}),
        "error": reported_error,
        "done_reason": diag.get("done_reason", ""),
        "execution_trace_count": len(trace) if "execution_trace" in diag else "",
        "raw_result_json": _json(result),
    })
    for key in ("model_calls", "prompt_tokens", "completion_tokens", "total_tokens",
                "fallback_steps", "token_usage_status"):
        row[key] = _first(result.get(key), execution.get(key), usage.get(key))
    for key in ("reuse_numerator", "reuse_denominator", "reuse_rate"):
        row[key] = reuse.get(key, "")
    for key in ("checker_trigger_total", "completion_review_calls"):
        row[key] = _first(diag.get(key), execution.get(key))
    for key in ("actions_executed", "observation_count", "cost_usd", "observed_cost_usd",
                "cost_status", "reasoning_tokens", "chat_model_calls", "embedding_model_calls",
                "memory_lookup_count", "memory_hit_count", "memory_hit_rate",
                "memory_explore_count", "memory_action_recalled_count", "memory_action_use_rate",
                "action_sent_count", "environment_failure", "classification",
                "action_backend", "observe_backend", "physical_backend", "control_backend",
                "state_backend", "task_params_sha256", "task_random_seed"):
        row[key] = _first(result.get(key))
    # Reuse is method-defined evidence, never inferred from a validator PASS.
    row["memory_direct_actions"] = _first(result.get("memory_action_recalled_count"),
                                          reuse.get("reuse_numerator"))
    row["reuse_unit"] = reuse.get("reuse_unit", "memory_action" if method == "mobilegpt" else "")
    row["reuse_evidence_status"] = reuse.get("evidence_status", "")
    backends = [str(row[k]) for k in ("action_backend", "observe_backend", "physical_backend", "control_backend") if row[k] != ""]
    row["backend_matches_oob_protocol"] = all("oob" in b.lower() for b in backends) if backends else ""
    expected_sizes = {
        EXPECTED_TARGET_SETTINGS[0]: (1440, 3120),
        # The Fold AVD reports 1768x2208 physical pixels but AndroidWorld's
        # logical app observation is the 2208x1840 window used by Host._screen_size.
        EXPECTED_TARGET_SETTINGS[1]: (2208, 1840),
        EXPECTED_TARGET_SETTINGS[2]: (1280, 800),
    }
    observed_sizes = set()
    for entry in _list(result.get("observation_evidence")):
        d = _dict(entry.get("display"))
        if d.get("width") and d.get("height"):
            observed_sizes.add((int(d["width"]), int(d["height"])))
    for step in steps:
        for key in ("observation", "next_observation"):
            shot = _dict(_dict(step.get(key)).get("screenshot"))
            if shot.get("width") and shot.get("height"):
                observed_sizes.add((int(shot["width"]), int(shot["height"])))
    row["observed_dimensions"] = _json(sorted(observed_sizes)) if observed_sizes else ""
    accepted_sizes = {expected_sizes[device]} if device in expected_sizes else set()
    if device == EXPECTED_TARGET_SETTINGS[1]:
        # Full-screen startup observations can expose the Fold physical
        # portrait surface; normal app observations use the logical window.
        accepted_sizes.add((1768, 2208))
    row["dimensions_match_protocol"] = bool(observed_sizes & accepted_sizes) if observed_sizes and accepted_sizes else ""
    return row


def _current_records(
    *, task_root: Path, data_root: Path, methods: Iterable[str]
) -> Iterable[dict[str, Any]]:
    for method, method_dir in _method_dirs(task_root, methods):
        for setting in sorted(method_dir.iterdir()):
            if not setting.is_dir():
                continue
            if method != SOURCE_METHOD and setting.name not in EXPECTED_TARGET_SETTINGS:
                continue
            current = setting / "runlog/current"
            if not (current / "run_log.json").is_file() and not (current / "task_results.jsonl").is_file():
                continue
            yield _record_from_paths(task=task_root.name, method=method, device=setting.name,
                run_path=current / "run_log.json", result_path=current / "task_results.jsonl",
                data_root=data_root)


def _explicit_record(task: str, path: Path, data_root: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".jsonl":
        result_path, run_path = path, path.parent / "run_log.json"
    elif path.name == "run_log.json":
        run_path, result_path = path, path.parent / "task_results.jsonl"
    else:
        raise ValueError(f"expected task_results.jsonl or run_log.json: {path}")
    run, result = _evidence_payload(run_path, result_path)
    if not run and not result:
        raise ValueError(f"empty or malformed evidence: {path}")
    parts = path.parts
    if task not in parts:
        raise ValueError(f"evidence path must include its task: {path}")
    index = parts.index(task)
    method, device = parts[index + 1:index + 3]
    if method not in DEFAULT_METHODS and method != SOURCE_METHOD:
        raise ValueError(f"unsupported method in evidence: {method}")
    aliases = dict(zip(("standard45562", "fold45564", "tablet45554"), EXPECTED_TARGET_SETTINGS))
    device = aliases.get(device, device)
    return _record_from_paths(task=task, method=method, device=device,
        run_path=run_path, result_path=result_path, data_root=data_root)


def _trace_rows(record: dict[str, Any], data_root: Path) -> list[dict[str, Any]]:
    run_log = _read_json(data_root / record["run_log_path"])
    if not run_log:
        try:
            raw_result = json.loads(record.get("raw_result_json", "{}"))
        except json.JSONDecodeError:
            raw_result = {}
        run_log = _dict(raw_result.get("canonical_run"))
    diagnostics = _dict(run_log.get("diagnostics"))
    trace = _list(diagnostics.get("execution_trace"))
    if not trace:
        trace = _list(run_log.get("steps"))
    else:
        # The diagnostic execution trace normally excludes the terminal
        # ``answer`` step.  Add any canonical RunLog steps absent from the
        # trace so the report exposes the complete interaction without
        # duplicating the physical action rows.
        seen_indices = {
            _integer(_dict(step).get("step_index"), -1) for step in trace
        }
        trace.extend(
            step for step in _list(run_log.get("steps"))
            if _integer(_dict(step).get("step_index"), -1) not in seen_indices
        )
        trace.sort(key=lambda step: _integer(_dict(step).get("step_index"), 0))
    rows: list[dict[str, Any]] = []
    for fallback_index, raw_step in enumerate(trace):
        step = _dict(raw_step)
        action = _dict(step.get("action"))
        metadata = _dict(step.get("metadata"))
        result = _dict(step.get("result"))
        token_usage = _dict(metadata.get("token_usage"))
        action_result = _dict(metadata.get("action_result"))
        action_result_diagnostics = _dict(action_result.get("diagnostics"))
        transfer = _dict(metadata.get("transfer"))
        if not transfer:
            transfer = _dict(metadata.get("mapping"))
        rows.append(
            {
                "task": record["task"],
                "method": record["method"],
                "device": record["device"],
                "summary_run_log_path": record["run_log_path"],
                "step_index": _integer(step.get("step_index"), fallback_index),
                "origin": metadata.get("origin") or "",
                "function_step_index": metadata.get("function_step_index", ""),
                "tool": action.get("tool") or action.get("action_type") or "",
                "action_args_json": _json(action.get("args") or action),
                "before_state_id": step.get("before_state_id") or "",
                "after_state_id": step.get("after_state_id") or "",
                "success": result.get("success"),
                "error": result.get("error") or metadata.get("error") or "",
                "action_summary": step.get("summary") or metadata.get("summary") or "",
                "token_prompt": _integer(token_usage.get("prompt_tokens")),
                "token_completion": _integer(token_usage.get("completion_tokens")),
                "token_total": _integer(token_usage.get("total_tokens")),
                "token_model": token_usage.get("model") or "",
                "transfer_confidence": _first(
                    transfer.get("mapping_confidence"), transfer.get("confidence")
                ),
                "transfer_margin": _first(transfer.get("margin")),
                "transfer_reason": transfer.get("reason") or transfer.get("error") or "",
                "transfer_fallback": transfer.get("fallback") or "",
                "transfer_json": _json(transfer),
                "coordinate_conversion_json": _json(metadata.get("coordinate_conversion", {})),
                "state_changed": _dict(metadata.get("action_effect")).get("state_changed", ""),
                "stabilization": action_result_diagnostics.get("state_stabilization") or "",
                "stabilization_result": action_result_diagnostics.get("state_stabilization_result") or "",
                "metadata_json": _json(metadata),
                "result_json": _json(result),
                "raw_step_json": _json(step),
            }
        )
    return rows


SUMMARY_FIELDS = (
    "task", "method", "device", "setting_dir", "run_log_path", "task_results_path", "run_id", "started_at",
    "success", "status", "official_validator_used", "validator_success", "validator_reward", "duration_sec",
    "execution_sec", "non_execution_sec", "duration_scope", "execution_scope", "model", "model_base_url",
    "model_calls", "router_calls", "planner_calls", "planner_turns_exposed", "completion_review_calls",
    "fallback_steps", "prompt_tokens", "completion_tokens", "total_tokens", "vlm_latency_sec", "performance_sidecar_path", "token_usage_status",
    "actions_executed", "step_count", "observation_count", "physical_action_count", "state_changing_action_count",
    "function_id", "function_replay_status", "function_step_count", "function_success_steps", "function_failed_steps",
    "function_resume_attempts", "function_resume_successes", "reuse_numerator", "reuse_denominator", "reuse_rate",
    "checker_trigger_total", "checker_trigger_counts_json", "function_resolution_json", "function_execution_json",
    "function_resume_json", "llm_usage_json", "planner_diagnostics_json", "performance_metrics_json", "task_params_json",
    "error", "done_reason", "execution_trace_count", "raw_result_json",
    "evidence_path", "evidence_sha256", "evidence_kind", "cost_usd", "observed_cost_usd", "cost_status",
    "reasoning_tokens", "chat_model_calls", "embedding_model_calls", "memory_lookup_count", "memory_hit_count",
    "memory_hit_rate", "memory_explore_count", "memory_action_recalled_count", "memory_action_use_rate",
    "memory_direct_actions", "reuse_unit", "reuse_evidence_status", "action_sent_count", "environment_failure",
    "classification", "action_backend", "observe_backend", "physical_backend", "control_backend", "state_backend",
    "backend_matches_oob_protocol", "observed_dimensions", "dimensions_match_protocol", "task_params_sha256",
    "task_random_seed",
)

STEP_FIELDS = (
    "task", "method", "device", "summary_run_log_path", "step_index", "origin", "function_step_index", "tool",
    "action_args_json", "before_state_id", "after_state_id", "success", "error", "action_summary", "token_prompt",
    "token_completion", "token_total", "token_model", "transfer_confidence", "transfer_margin", "transfer_reason",
    "transfer_fallback", "transfer_json", "coordinate_conversion_json", "state_changed", "stabilization",
    "stabilization_result", "metadata_json", "result_json", "raw_step_json",
)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _markdown_table(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> str:
    if not rows:
        return "（无 current 结果）\n"
    labels = {field: field for field in fields}
    lines = ["| " + " | ".join(labels[field] for field in fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        values = []
        for field in fields:
            value = str(row.get(field, ""))
            value = value.replace("|", "\\|").replace("\n", " ")
            values.append(value)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


DISPLAY_SUMMARY_FIELDS = tuple(
    field for field in SUMMARY_FIELDS if not field.endswith("_json") and field != "raw_result_json"
)
DISPLAY_STEP_FIELDS = tuple(
    field for field in STEP_FIELDS
    if field not in {"transfer_json", "coordinate_conversion_json", "metadata_json", "result_json", "raw_step_json"}
)


def generate_task(
    task: str, *, data_root: Path, output_root: Path, include_source: bool = False,
    result_paths: Iterable[Path] = (),
) -> dict[str, str]:
    task_root = data_root / task
    methods = list(DEFAULT_METHODS) + ([SOURCE_METHOD] if include_source else [])
    records = list(_current_records(task_root=task_root, data_root=data_root, methods=methods))
    explicit = [_explicit_record(task, Path(path), data_root) for path in result_paths]
    all_records = records + explicit
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in all_records:
        key = (candidate["method"], candidate["device"])
        if key not in selected:
            selected[key] = candidate
            continue
        current = selected[key]
        # A validator-confirmed result is the canonical cell evidence.  If
        # success is tied, prefer current over archive and then newest run.
        candidate_rank = (
            1 if candidate.get("validator_success") is True else 0,
            1 if candidate.get("evidence_kind") == "current" else 0,
            str(candidate.get("started_at") or ""),
            str(candidate.get("evidence_path") or ""),
        )
        current_rank = (
            1 if current.get("validator_success") is True else 0,
            1 if current.get("evidence_kind") == "current" else 0,
            str(current.get("started_at") or ""),
            str(current.get("evidence_path") or ""),
        )
        if candidate_rank > current_rank:
            selected[key] = candidate
    records = list(selected.values())
    present = {(str(row["method"]), str(row["device"])) for row in records}
    for method in DEFAULT_METHODS:
        for device in EXPECTED_TARGET_SETTINGS:
            if (method, device) in present:
                continue
            records.append(
                {
                    field: "" for field in SUMMARY_FIELDS
                } | {
                    "task": task,
                    "method": method,
                    "device": device,
                    "setting_dir": f"{task}/{method}/{device}",
                    "run_log_path": f"{task}/{method}/{device}/runlog/current/run_log.json",
                    "task_results_path": f"{task}/{method}/{device}/runlog/current/task_results.jsonl",
                    "success": "",
                    "status": "missing_current",
                    "error": "current runlog missing",
                }
            )
    records.sort(key=lambda row: (row["method"], row["device"]))
    steps: list[dict[str, Any]] = []
    for record in records:
        steps.extend(_trace_rows(record, data_root))
    task_output = output_root / task
    _write_csv(task_output / "summary.csv", records, SUMMARY_FIELDS)
    _write_csv(task_output / "steps.csv", steps, STEP_FIELDS)
    markdown = [
        f"# {task} execution table",
        "",
        "Methods included: `omniflow`.",
        "`fixed_replay` is intentionally excluded. `source` is included only with `--include-source`.",
        "",
        f"Summary rows: {len(records)}; step rows: {len(steps)}.",
        "",
        "## Summary",
        "",
        _markdown_table(records, DISPLAY_SUMMARY_FIELDS),
        "## Steps",
        "",
        _markdown_table(steps, DISPLAY_STEP_FIELDS),
    ]
    (task_output / "README.md").write_text("\n".join(markdown), encoding="utf-8")
    return {
        "task": task,
        "summary": str((task_output / "summary.csv").resolve()),
        "steps": str((task_output / "steps.csv").resolve()),
        "markdown": str((task_output / "README.md").resolve()),
        "summary_rows": str(len(records)),
        "step_rows": str(len(steps)),
    }


def _write_aggregate_index(path: Path, outputs: list[dict[str, str]]) -> None:
    """Write the machine-generated OmniFlow batch overview.

    The per-task CSV files remain the source of detail.  This compact index is
    deliberately generated from those files so the desktop handoff cannot
    drift from the task reports or accidentally include another method.
    """

    csv.field_size_limit(sys.maxsize)
    task_rows: list[tuple[str, list[dict[str, str]]]] = []
    total_cells = success_cells = official_cells = 0
    for output in outputs:
        summary_path = Path(output["summary"])
        with summary_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        task = output["task"]
        task_rows.append((task, rows))
        for row in rows:
            if row.get("method") != "omniflow":
                continue
            total_cells += 1
            official = row.get("official_validator_used") in {"True", "true", "1"}
            success = row.get("success") in {"True", "true", "1"}
            if official:
                official_cells += 1
            if official and success:
                success_cells += 1

    lines = [
        "# OmniFlow AndroidWorld 116-task batch overview",
        "",
        "Generated by scripts/exp/generate_task_table.py; method=omniflow only.",
        "Completed means official_validator_used=True and success=True.",
        "",
        f"tasks={len(task_rows)} cells={total_cells} official_validator_cells={official_cells} official_success_cells={success_cells}",
        "",
        "task\tstandard\tfold\ttablet\tfunction\tmodel_calls\tfallback_steps\tchecker_triggers\terror",
    ]
    device_names = {
        EXPECTED_TARGET_SETTINGS[0]: "standard",
        EXPECTED_TARGET_SETTINGS[1]: "fold",
        EXPECTED_TARGET_SETTINGS[2]: "tablet",
    }
    for task, rows in task_rows:
        by_device = {device_names.get(row.get("device", ""), row.get("device", "")): row
                     for row in rows if row.get("method") == "omniflow"}
        cells = []
        for device in ("standard", "fold", "tablet"):
            row = by_device.get(device, {})
            cells.append(
                f"{row.get('status', 'missing_current')}:{row.get('success', '')}"
            )
        representative = next((row for row in by_device.values() if row), {})
        lines.append("\t".join([
            task,
            *cells,
            representative.get("function_id", ""),
            representative.get("model_calls", ""),
            representative.get("fallback_steps", ""),
            representative.get("checker_trigger_total", ""),
            representative.get("error", "").replace("\t", " ").replace("\n", " "),
        ]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="", help="One AndroidWorld task name.")
    parser.add_argument("--all", action="store_true", help="Generate one table directory for every task.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-root", default="", help="Output directory; defaults to <data-root>/../task_tables.")
    parser.add_argument("--include-source", action="store_true")
    parser.add_argument(
        "--index-path",
        default="",
        help="Also write the generated aggregate overview to this text path.",
    )
    parser.add_argument("--evidence", action="append", default=[],
                        help="Explicit task_results.jsonl or run_log.json evidence; repeatable.")
    args = parser.parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    if not args.task and not args.all:
        parser.error("provide --task TASK or --all")
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else data_root.parent / "task_tables"
    )
    tasks = [args.task] if args.task else sorted(
        path.name
        for path in data_root.iterdir()
        if path.is_dir()
        and not path.name.startswith(".")
        and ((path / "omniflow").is_dir() or (path / "source").is_dir())
    )
    outputs = [
        generate_task(task, data_root=data_root, output_root=output_root,
                      include_source=args.include_source,
                      result_paths=[Path(path) for path in args.evidence] if args.task else [])
        for task in tasks
    ]
    if args.index_path:
        _write_aggregate_index(Path(args.index_path).expanduser().resolve(), outputs)
    print(json.dumps({"output_root": str(output_root), "tasks": outputs}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
