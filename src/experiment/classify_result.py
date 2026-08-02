#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

ENVIRONMENT_PATTERNS = {
    "app_setup_failed": r"Failed to automatically setup app|manually setup the app",
    "task_start_timeout": r"waited for task_started: timeout",
    "server_failed": r"episode_server_failed|server exited early",
    "adb_failed": r"device offline|no devices/emulators found|adb:.*not found|device .* not found",
    "oob_failed": r"OOB .*not ready|OOB health not reachable|OOB get_state failed",
    "port_conflict": r"Address already in use",
    "snapshot_failed": r"Snapshot not found|failed to restore .*snapshot",
    "model_service_failed": r"AuthenticationError|RateLimitError|APIConnectionError",
    "androidworld_accessibility_failed": (
        r"accessibilityforwarder keeps stopping|"
        r"Application Error: com\.google\.androidenv\.accessibilityforwarder"
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify one MobileGPT episode result.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument(
        "--initial-memory-condition",
        required=True,
        choices=["empty_memory", "native_memory", "function_transfer"],
    )
    parser.add_argument("--frozen-manifest")
    parser.add_argument("--json-out")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary_path = Path(args.summary).expanduser().resolve()
    log_path = Path(args.log).expanduser().resolve()
    frozen_manifest_path = (
        Path(args.frozen_manifest).expanduser().resolve()
        if str(args.frozen_manifest or "").strip()
        else None
    )
    summary = _read_json(summary_path)
    frozen_manifest = (
        _read_json(frozen_manifest_path)
        if frozen_manifest_path is not None
        else {}
    )
    rows = summary.get("rows") if isinstance(summary.get("rows"), list) else []
    row = rows[0] if rows and isinstance(rows[0], dict) else {}
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        log_text = ""

    environment_reasons: list[str] = []
    if not summary:
        environment_reasons.append("missing_summary")
    if not row:
        environment_reasons.append("missing_result_row")
    if not log_text:
        environment_reasons.append("missing_log")
    if args.initial_memory_condition != "empty_memory":
        if frozen_manifest_path is None or not frozen_manifest_path.is_file():
            environment_reasons.append("missing_frozen_memory")
        elif (
            not frozen_manifest
            or not bool(frozen_manifest.get("read_only"))
            or not str(frozen_manifest.get("digest") or "")
            or int(frozen_manifest.get("file_count") or 0) <= 0
        ):
            environment_reasons.append("invalid_frozen_memory_manifest")
    if row and not bool(row.get("official_validator_used")):
        environment_reasons.append("missing_official_validator")
    task_started_count = int(
        row.get("episode_task_started_count")
        or row.get("warm_task_started_count")
        or row.get("mobilegpt_task_started_count")
        or 0
    ) if row else 0
    if row and task_started_count != 1:
        environment_reasons.append("task_not_started_once")
    if row and str(row.get("runtime_integrity_error") or "").strip():
        environment_reasons.append("runtime_integrity_error")
    for reason, pattern in ENVIRONMENT_PATTERNS.items():
        if re.search(pattern, log_text, flags=re.IGNORECASE):
            environment_reasons.append(reason)

    official_success = bool(row.get("official_validator_success")) if row else False
    if environment_reasons:
        classification = "environment_failure"
        exit_code = 20
    elif official_success:
        classification = "success"
        exit_code = 0
    else:
        classification = "method_failure"
        exit_code = 10

    report = {
        "schema_version": "omniflow.mobilegpt_result_classification.v1",
        "classification": classification,
        "task": row.get("task_name") if row else None,
        "official_validator_success": official_success,
        "task_started_count": task_started_count,
        "task_finished_count": int(
            row.get("episode_task_finished_count")
            or row.get("warm_task_finished_count")
            or row.get("mobilegpt_task_finished_count")
            or 0
        )
        if row
        else 0,
        "actions_executed": int(row.get("actions_executed") or 0) if row else 0,
        "environment_reasons": sorted(set(environment_reasons)),
        "summary": str(summary_path),
        "log": str(log_path),
        "initial_memory_condition": args.initial_memory_condition,
        "frozen_manifest": str(frozen_manifest_path or ""),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json_out:
        output = Path(args.json_out).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
