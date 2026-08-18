"""Immutable conclusions for AndroidWorld results without validator results."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable

from src.experiment.mobilegpt_contract import MOBILEGPT_SUPPORTED_SOURCE_METHODS
from src.integrations.android_world.methods import reuse_metrics_from_result_row

SCHEMA_VERSION = "omniflow.androidworld.result_outcome.v2"
LEGACY_SCHEMA_VERSION = "omniflow.androidworld.cell_outcome.v1"
_MOBILEGPT_SOURCE_STATS_PATTERN = re.compile(
    r"MOBILEGPT_STATS_JSONL=(?P<path>[^\s'\"]+source_stats\.jsonl)"
)


def _safe_component(value: str, *, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return normalized.strip("._") or fallback


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _jsonl_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _number(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _whole_or_float(value: int | float) -> int | float:
    numeric = float(value)
    return int(numeric) if numeric.is_integer() else round(numeric, 6)


def _stats_metrics(artifact_root: Path | None) -> dict[str, int | float]:
    if artifact_root is None or not artifact_root.exists():
        return {}
    stats_paths = sorted(artifact_root.rglob("*stats.jsonl"))
    rows = _jsonl_rows(stats_paths)
    prompt_tokens = sum(_number(row.get("prompt_tokens")) for row in rows)
    completion_tokens = sum(_number(row.get("completion_tokens")) for row in rows)
    model_rows = [
        row
        for row in rows
        if str(row.get("event") or "") in {"chat_call", "embedding_call"}
    ]
    total_tokens = sum(
        _number(row.get("total_tokens"))
        or _number(row.get("prompt_tokens"))
        + _number(row.get("completion_tokens"))
        for row in model_rows
    )
    task_finished = [
        row for row in rows if str(row.get("event") or "") == "task_finished"
    ]
    episode_duration_sec = max(
        (_number(row.get("elapsed_sec")) for row in task_finished),
        default=0.0,
    )
    return {
        "model_calls": sum(
            str(row.get("event") or "") in {"chat_call", "embedding_call"}
            for row in rows
        ),
        "chat_model_calls": sum(
            str(row.get("event") or "") == "chat_call" for row in rows
        ),
        "embedding_model_calls": sum(
            str(row.get("event") or "") == "embedding_call" for row in rows
        ),
        "prompt_tokens": _whole_or_float(prompt_tokens),
        "completion_tokens": _whole_or_float(completion_tokens),
        "total_tokens": _whole_or_float(total_tokens),
        "actions_executed": sum(
            str(row.get("event") or "") == "mobilegpt_action_sent" for row in rows
        ),
        "episode_duration_sec": round(episode_duration_sec, 6),
    }


def _recover_mobilegpt_source_accounting(
    outcome: dict[str, Any],
) -> tuple[dict[str, int | float], Path | None]:
    if str(outcome.get("method") or "") != "mobilegpt":
        return {}, None
    task_log = Path(str(outcome.get("task_log") or "")).expanduser()
    candidates: list[Path] = []

    artifact_root = Path(str(outcome.get("artifact_root") or "")).expanduser()
    roots = [artifact_root.resolve()] if artifact_root.is_dir() else []
    if task_log.is_file():
        text = task_log.read_text(encoding="utf-8", errors="replace")
        roots.extend(
            Path(match.group("path")).expanduser().resolve().parent
            for match in _MOBILEGPT_SOURCE_STATS_PATTERN.finditer(text)
        )
    for source_root in roots:
        if source_root in candidates or not list(source_root.glob("*stats.jsonl")):
            continue
        failure = _json_object(source_root / "prep_failure.json")
        command = _json_object(source_root / "source_episode_command.json")
        if (
            failure.get("retry_allowed") is not False
            or str(command.get("task_name") or "")
            != str(outcome.get("task_name") or "")
            or str(command.get("source_method") or "")
            not in MOBILEGPT_SUPPORTED_SOURCE_METHODS
        ):
            continue
        candidates.append(source_root)
    if len(candidates) != 1:
        return {}, None
    return _stats_metrics(candidates[0]), candidates[0]


def _failure_summary(task_log: Path | None, artifact_root: Path | None) -> str:
    if artifact_root is not None:
        marker = artifact_root / "prep_failure.json"
        failure = _json_object(marker)
        error = str(failure.get("error") or "").strip()
        if error:
            return error
    if task_log is not None and task_log.is_file():
        lines = [
            line.strip()
            for line in task_log.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            if line.strip()
        ]
        for line in reversed(lines):
            if any(
                token in line.lower()
                for token in ("error", "failed", "exception", "traceback")
            ):
                return line[-2000:]
        if lines:
            return lines[-1][-2000:]
    return "result_finished_without_registered_validator_result"


def record_result_outcome(
    *,
    outcomes_root: str | Path,
    task_name: str,
    method: str,
    device: str,
    device_serial: str,
    attempt_id: str,
    source_seed: int,
    evaluation_seed: int,
    status: str,
    stage: str,
    task_log: str | Path | None = None,
    artifact_root: str | Path | None = None,
    outer_wall_sec: float = 0.0,
) -> Path:
    """Write one immutable non-validator result conclusion."""

    root = Path(outcomes_root).expanduser().resolve()
    log_path = Path(task_log).expanduser().resolve() if task_log else None
    artifact_path = (
        Path(artifact_root).expanduser().resolve() if artifact_root else None
    )
    destination = (
        root
        / _safe_component(task_name, fallback="task")
        / _safe_component(method, fallback="method")
        / _safe_component(device, fallback="device")
        / _safe_component(attempt_id, fallback="attempt")
    )
    metrics = _stats_metrics(artifact_path)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "immutable": True,
        "task_name": str(task_name),
        "method": str(method),
        "device": str(device),
        "device_serial": str(device_serial),
        "attempt_id": str(attempt_id),
        "source_seed": int(source_seed),
        "evaluation_seed": int(evaluation_seed),
        "status": str(status),
        "stage": str(stage),
        "failure_summary": _failure_summary(log_path, artifact_path),
        "official_validator_used": False,
        "official_validator_success": None,
        "official_validator_coverage_rate": 0.0,
        "model_calls": 0,
        "chat_model_calls": 0,
        "embedding_model_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "actions_executed": 0,
        "episode_duration_sec": 0.0,
        "outer_wall_sec": round(float(outer_wall_sec or 0.0), 6),
        "retry_count": 0,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "task_log": str(log_path) if log_path is not None else "",
        "task_log_sha256": (
            _sha256(log_path) if log_path is not None and log_path.is_file() else ""
        ),
        "artifact_root": str(artifact_path) if artifact_path is not None else "",
    }
    payload.update(metrics)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    outcome_path = destination / "outcome.json"
    if destination.exists():
        existing = _json_object(outcome_path)
        comparable = {key: value for key, value in payload.items() if key != "recorded_at"}
        existing_comparable = {
            key: value for key, value in existing.items() if key != "recorded_at"
        }
        if comparable != existing_comparable:
            raise FileExistsError(f"immutable result outcome conflict: {destination}")
        return outcome_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(dir=destination.parent, prefix=f".{destination.name}.")
    )
    try:
        (temporary / "outcome.json").write_text(text, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return outcome_path


def concluded_result_keys(
    *,
    outcomes_root: str | Path,
    task_name: str,
    methods: Iterable[str],
    devices: Iterable[str],
    source_seed: int,
    evaluation_seed: int,
    attempt_id: str | None = None,
) -> set[tuple[str, str]]:
    """Return results concluded within the selected immutable attempt."""

    root = Path(outcomes_root).expanduser().resolve()
    accepted_methods = {str(value) for value in methods}
    accepted_devices = {str(value) for value in devices}
    concluded: set[tuple[str, str]] = set()
    task_root = root / _safe_component(task_name, fallback="task")
    if not task_root.is_dir():
        return concluded
    for outcome_path in sorted(task_root.rglob("outcome.json")):
        payload = _json_object(outcome_path)
        method = str(payload.get("method") or "")
        device = str(payload.get("device") or "")
        if (
            payload.get("schema_version") not in {SCHEMA_VERSION, LEGACY_SCHEMA_VERSION}
            or payload.get("immutable") is not True
            or str(payload.get("task_name") or "") != str(task_name)
            or method not in accepted_methods
            or device not in accepted_devices
            or int(payload.get("source_seed") or -1) != int(source_seed)
            or int(payload.get("evaluation_seed") or -1) != int(evaluation_seed)
            or (
                attempt_id is not None
                and str(payload.get("attempt_id") or "") != str(attempt_id)
            )
        ):
            continue
        concluded.add((method, device))
    return concluded


def _registered_result_rows(memory_index: Path) -> dict[str, dict[str, Any]]:
    current = _json_object(memory_index)
    result_cells = current.get("canonical", {}).get("result_cells", {})
    if not isinstance(result_cells, dict):
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for key, record in result_cells.items():
        if not isinstance(record, dict):
            continue
        object_path = Path(
            str(record.get("registered_result_object_path") or "")
        ).expanduser()
        registered = _json_object(object_path)
        candidates = [
            row
            for row in registered.get("details") or registered.get("rows") or []
            if isinstance(row, dict)
        ]
        if len(candidates) == 1:
            rows[str(key)] = candidates[0]
    return rows


def _outcome_rows(
    outcomes_root: Path,
    *,
    attempt_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not outcomes_root.is_dir():
        return rows
    for path in sorted(outcomes_root.rglob("outcome.json")):
        payload = _json_object(path)
        if (
            payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("immutable") is not True
            or (
                attempt_id is not None
                and str(payload.get("attempt_id") or "") != str(attempt_id)
            )
        ):
            continue
        key = "|".join(
            str(payload.get(field) or "")
            for field in (
                "task_name",
                "method",
                "device",
                "source_seed",
                "evaluation_seed",
            )
        )
        candidate = {**payload, "outcome_path": str(path)}
        recovered_metrics, accounting_root = _recover_mobilegpt_source_accounting(
            candidate
        )
        if recovered_metrics:
            candidate.update(recovered_metrics)
            candidate["accounting_recovered"] = True
            candidate["accounting_evidence_path"] = str(accounting_root)
        else:
            candidate["accounting_recovered"] = False
            candidate["accounting_evidence_path"] = str(
                candidate.get("artifact_root") or ""
            )
        existing = rows.get(key)
        if existing is None or str(candidate.get("recorded_at") or "") < str(
            existing.get("recorded_at") or ""
        ):
            rows[key] = candidate
    return rows


def _registered_report_row(
    *,
    task: str,
    method: str,
    device: str,
    source_seed: int,
    evaluation_seed: int,
    row: dict[str, Any],
) -> dict[str, Any]:
    success = bool(row.get("official_validator_success"))
    reuse = reuse_metrics_from_result_row(row, method=method)
    return {
        "task_name": task,
        "method": method,
        "device": device,
        "source_seed": source_seed,
        "evaluation_seed": evaluation_seed,
        "conclusion": "validator_success" if success else "validator_failure",
        "status": "completed",
        "failure_summary": "" if success else str(
            row.get("failure_summary")
            or row.get("error")
            or "official_validator_returned_false"
        ),
        "official_validator_used": bool(row.get("official_validator_used", True)),
        "official_validator_success": success,
        "official_validator_coverage_rate": _number(
            row.get("official_validator_coverage_rate") or 1.0
        ),
        "model_calls": int(
            _number(row.get("episode_model_calls") or row.get("model_calls"))
        ),
        "prompt_tokens": int(
            _number(row.get("episode_prompt_tokens") or row.get("prompt_tokens"))
        ),
        "completion_tokens": int(
            _number(
                row.get("episode_completion_tokens")
                or row.get("completion_tokens")
            )
        ),
        "total_tokens": int(
            _number(row.get("episode_total_tokens") or row.get("total_tokens"))
        ),
        "actions_executed": int(
            _number(
                row.get("episode_actions_executed") or row.get("actions_executed")
            )
        ),
        "artifact_used": reuse["artifact_used"],
        "reuse_numerator": reuse["reuse_numerator"],
        "reuse_denominator": reuse["reuse_denominator"],
        "reuse_rate": reuse["reuse_rate"],
        "reuse_unit": reuse["reuse_unit"],
        "reuse_evidence_status": reuse["evidence_status"],
        "episode_duration_sec": _number(
            row.get("episode_duration_sec")
            or row.get("duration_sec")
            or row.get("episode_task_elapsed_sec")
        ),
        "outer_wall_sec": _number(
            row.get("outer_wall_sec") or row.get("wall_sec")
        ),
        "attempt_id": str(row.get("attempt_id") or ""),
        "evidence_path": str(
            row.get("evidence_path")
            or row.get("run_dir")
            or row.get("output_path")
            or ""
        ),
        "accounting_recovered": False,
        "accounting_evidence_path": str(
            row.get("evidence_path")
            or row.get("run_dir")
            or row.get("output_path")
            or ""
        ),
    }


def _failure_report_row(
    *,
    task: str,
    method: str,
    device: str,
    source_seed: int,
    evaluation_seed: int,
    outcome: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task_name": task,
        "method": method,
        "device": device,
        "source_seed": source_seed,
        "evaluation_seed": evaluation_seed,
        "conclusion": "non_validator_failure",
        "status": str(outcome.get("status") or "execution_failed"),
        "failure_summary": str(outcome.get("failure_summary") or ""),
        "official_validator_used": False,
        "official_validator_success": None,
        "official_validator_coverage_rate": 0.0,
        "model_calls": int(_number(outcome.get("model_calls"))),
        "prompt_tokens": int(_number(outcome.get("prompt_tokens"))),
        "completion_tokens": int(_number(outcome.get("completion_tokens"))),
        "total_tokens": int(_number(outcome.get("total_tokens"))),
        "actions_executed": int(_number(outcome.get("actions_executed"))),
        "artifact_used": False,
        "reuse_numerator": 0,
        "reuse_denominator": 0,
        "reuse_rate": None,
        "reuse_unit": "",
        "reuse_evidence_status": "unavailable",
        "episode_duration_sec": _number(outcome.get("episode_duration_sec")),
        "outer_wall_sec": _number(outcome.get("outer_wall_sec")),
        "attempt_id": str(outcome.get("attempt_id") or ""),
        "evidence_path": str(outcome.get("outcome_path") or ""),
        "accounting_recovered": bool(outcome.get("accounting_recovered")),
        "accounting_evidence_path": str(
            outcome.get("accounting_evidence_path") or ""
        ),
    }


def summarize_results(
    *,
    memory_index: str | Path,
    outcomes_root: str | Path,
    tasks: Iterable[str],
    methods: Iterable[str],
    devices: Iterable[str],
    source_seed: int,
    evaluation_seed: int,
    attempt_id: str,
) -> dict[str, Any]:
    """Summarize immutable result conclusions without creating a second table."""

    memory_path = Path(memory_index).expanduser().resolve()
    registered = _registered_result_rows(memory_path)
    outcomes = _outcome_rows(
        Path(outcomes_root).expanduser().resolve(),
        attempt_id=attempt_id,
    )
    rows: list[dict[str, Any]] = []
    counts = {
        "planned": 0,
        "validator_success": 0,
        "validator_failure": 0,
        "non_validator_failure": 0,
        "pending": 0,
    }
    for task in tasks:
        for method in methods:
            for device in devices:
                counts["planned"] += 1
                key = f"{task}|{method}|{device}|{source_seed}|{evaluation_seed}"
                if key in outcomes:
                    row = _failure_report_row(
                        task=task,
                        method=method,
                        device=device,
                        source_seed=source_seed,
                        evaluation_seed=evaluation_seed,
                        outcome=outcomes[key],
                    )
                    counts["non_validator_failure"] += 1
                elif key in registered:
                    row = _registered_report_row(
                        task=task,
                        method=method,
                        device=device,
                        source_seed=source_seed,
                        evaluation_seed=evaluation_seed,
                        row=registered[key],
                    )
                    counts[str(row["conclusion"])] += 1
                else:
                    row = {
                        "task_name": task,
                        "method": method,
                        "device": device,
                        "source_seed": source_seed,
                        "evaluation_seed": evaluation_seed,
                        "conclusion": "pending",
                        "status": "pending",
                        "failure_summary": "",
                        "official_validator_used": False,
                        "official_validator_success": None,
                        "official_validator_coverage_rate": 0.0,
                        "model_calls": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "actions_executed": 0,
                        "artifact_used": False,
                        "reuse_numerator": 0,
                        "reuse_denominator": 0,
                        "reuse_rate": None,
                        "reuse_unit": "",
                        "reuse_evidence_status": "unavailable",
                        "episode_duration_sec": 0.0,
                        "outer_wall_sec": 0.0,
                        "attempt_id": "",
                        "evidence_path": "",
                        "accounting_recovered": False,
                        "accounting_evidence_path": "",
                    }
                    counts["pending"] += 1
                rows.append(row)
    model_calls = sum(int(row["model_calls"]) for row in rows)
    total_tokens = sum(int(row["total_tokens"]) for row in rows)
    summary = {
        "schema_version": "omniflow.androidworld.result-summary.v1",
        "attempt_id": str(attempt_id),
        "source_seed": int(source_seed),
        "evaluation_seed": int(evaluation_seed),
        "counts": counts,
        "model_calls": model_calls,
        "total_tokens": total_tokens,
        "episode_duration_sec": round(
            sum(_number(row["episode_duration_sec"]) for row in rows), 6
        ),
        "outer_wall_sec": round(
            sum(_number(row["outer_wall_sec"]) for row in rows), 6
        ),
    }
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    record = subparsers.add_parser("record")
    record.add_argument("--outcomes-root", required=True)
    record.add_argument("--task", required=True)
    record.add_argument("--method", required=True)
    record.add_argument("--device", required=True)
    record.add_argument("--device-serial", required=True)
    record.add_argument("--attempt-id", required=True)
    record.add_argument("--source-seed", type=int, required=True)
    record.add_argument("--evaluation-seed", type=int, required=True)
    record.add_argument("--status", required=True)
    record.add_argument("--stage", required=True)
    record.add_argument("--task-log", default="")
    record.add_argument("--artifact-root", default="")
    record.add_argument("--outer-wall-sec", type=float, default=0.0)

    concluded = subparsers.add_parser("concluded")
    concluded.add_argument("--outcomes-root", required=True)
    concluded.add_argument("--task", required=True)
    concluded.add_argument("--methods", required=True)
    concluded.add_argument("--devices", required=True)
    concluded.add_argument("--source-seed", type=int, required=True)
    concluded.add_argument("--evaluation-seed", type=int, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "record":
        result: Any = record_result_outcome(
            outcomes_root=args.outcomes_root,
            task_name=args.task,
            method=args.method,
            device=args.device,
            device_serial=args.device_serial,
            attempt_id=args.attempt_id,
            source_seed=args.source_seed,
            evaluation_seed=args.evaluation_seed,
            status=args.status,
            stage=args.stage,
            task_log=args.task_log or None,
            artifact_root=args.artifact_root or None,
            outer_wall_sec=args.outer_wall_sec,
        )
        print(result)
    elif args.command == "concluded":
        result = concluded_result_keys(
            outcomes_root=args.outcomes_root,
            task_name=args.task,
            methods=tuple(args.methods.split(",")),
            devices=tuple(args.devices.split(",")),
            source_seed=args.source_seed,
            evaluation_seed=args.evaluation_seed,
        )
        for method, device in sorted(result):
            print(f"{method}\t{device}")
    else:
        raise AssertionError(f"unsupported_batch_outcome_command:{args.command}")
    return 0


__all__ = [
    "concluded_result_keys",
    "record_result_outcome",
    "summarize_results",
]


if __name__ == "__main__":
    raise SystemExit(main())
