#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any


SCHEMA_VERSION = "omniflow.androidworld.function-campaign.v1"
ENVIRONMENT_PATTERNS = {
    "adb_unavailable": r"device offline|no devices/emulators found|device .* not found|adb:.*not found|cannot connect to daemon",
    "app_setup_failed": r"failed to automatically setup app|manually setup the app|setup_apps|snapshot.*no such file or directory",
    "database_backend_failed": r"no such module:\s*fts[34]|sqlite.*fts[34]",
    "ffmpeg_unavailable": r"ffmpeg.*(?:not found|missing)|no such file.*ffmpeg",
    "grpc_failed": r"failed to connect to the emulator|grpc.*(?:unavailable|failed)|emulator stub has not been initialized",
    "oob_failed": r"oob .*not ready|oob accessibility service is not bound|oob_get_state_timeout|oob get_state failed|debug get_state did not become ready",
    "port_conflict": r"address already in use|port .* already in use",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run strict AndroidWorld Function probes one task at a time."
    )
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--android-world-root", type=Path, required=True)
    parser.add_argument("--function-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-device", required=True)
    parser.add_argument("--target-device", required=True)
    parser.add_argument("--adb-path", required=True)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--model", default="qwen3-vl-plus")
    parser.add_argument("--probe-script", type=Path)
    parser.add_argument("--manifest-loader", type=Path)
    return parser


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_tasks(loader_path: Path, manifest_path: Path, repo: Path) -> list[dict[str, Any]]:
    spec = importlib.util.spec_from_file_location("function_onepass_loader", loader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable_to_load_manifest_validator:{loader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tasks = module.load_function_manifest(manifest_path, repo)
    return sorted(tasks, key=lambda value: int(value["index"]), reverse=True)


def _last_jsonl(path: Path) -> dict[str, Any]:
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    except OSError:
        return {}
    if not lines:
        return {}
    try:
        value = json.loads(lines[-1])
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def classify_probe(probe: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in probe.get("rows") or () if isinstance(row, dict)]
    environment_reasons: set[str] = set()
    if len(rows) != 2:
        environment_reasons.add("missing_dual_device_rows")
    for row in rows:
        role = str(row.get("role") or row.get("device", {}).get("role") or "unknown")
        classification = str(row.get("classification") or "")
        result_path = Path(str(row.get("result_file") or ""))
        result = _last_jsonl(result_path) if result_path.is_file() else {}
        failure_text = "\n".join(
            str(value or "")
            for value in (
                row.get("failure_reason"),
                row.get("error"),
                result.get("error"),
                result.get("exception_info"),
            )
        )
        if classification == "setup_or_harness_failed":
            environment_reasons.add(f"{role}:setup_or_harness_failed")
        if not result:
            environment_reasons.add(f"{role}:missing_result")
        else:
            official = result.get("androidworld_validator_result")
            official = official if isinstance(official, dict) else {}
            official_used = bool(
                result.get("official_validator_used")
                or official.get("uses_androidworld_official_validator")
            )
            if not official_used:
                environment_reasons.add(f"{role}:missing_official_validator")
        for reason, pattern in ENVIRONMENT_PATTERNS.items():
            if re.search(pattern, failure_text, flags=re.IGNORECASE):
                environment_reasons.add(f"{role}:{reason}")
        if (
            int(row.get("returncode") or 0) == 124
            and int(row.get("model_calls") or 0) == 0
        ):
            environment_reasons.add(f"{role}:pre_resolver_timeout")

    if environment_reasons:
        classification = "environment_failure"
    elif probe.get("success") is True:
        classification = "success"
    else:
        classification = "method_failure"
    return {
        "schema_version": "omniflow.androidworld.function-task-classification.v1",
        "classification": classification,
        "task": probe.get("task"),
        "source_success": bool(
            next(
                (
                    row.get("success")
                    for row in rows
                    if str(row.get("role") or row.get("device", {}).get("role") or "")
                    == "source"
                ),
                False,
            )
        ),
        "target_success": bool(
            next(
                (
                    row.get("success")
                    for row in rows
                    if str(row.get("role") or row.get("device", {}).get("role") or "")
                    == "target"
                ),
                False,
            )
        ),
        "environment_reasons": sorted(environment_reasons),
    }


def _initial_state(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    tasks: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "running",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "model": str(args.model),
        "seed_base": int(args.seed_base),
        "source_device": str(args.source_device),
        "target_device": str(args.target_device),
        "task_count": len(tasks),
        "next_position": 0,
        "tasks": [
            {
                "position": position,
                "index": int(task["index"]),
                "task": str(task["task_name"]),
                "seed": int(args.seed_base) + int(task["index"]),
            }
            for position, task in enumerate(tasks)
        ],
        "attempts": [],
    }


def _verify_state(state: dict[str, Any], args: argparse.Namespace, digest: str) -> None:
    expected = {
        "manifest_sha256": digest,
        "model": str(args.model),
        "seed_base": int(args.seed_base),
        "source_device": str(args.source_device),
        "target_device": str(args.target_device),
    }
    mismatches = {
        key: {"expected": value, "actual": state.get(key)}
        for key, value in expected.items()
        if state.get(key) != value
    }
    if mismatches:
        raise ValueError(f"frozen_campaign_mismatch:{json.dumps(mismatches, sort_keys=True)}")


def _attempt_number(task_root: Path) -> int:
    values = []
    for path in task_root.glob("attempt_*"):
        try:
            values.append(int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(values, default=0) + 1


def _campaign_summary(state: dict[str, Any]) -> dict[str, Any]:
    latest: dict[int, dict[str, Any]] = {}
    for attempt in state.get("attempts") or ():
        if not isinstance(attempt, dict):
            continue
        latest[int(attempt.get("position") or 0)] = attempt
    counts: dict[str, int] = {}
    for attempt in latest.values():
        classification = str(attempt.get("classification") or "unknown")
        counts[classification] = counts.get(classification, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": state.get("status"),
        "task_count": state.get("task_count"),
        "completed_task_count": len(
            [
                attempt
                for attempt in latest.values()
                if attempt.get("classification") in {"success", "method_failure"}
            ]
        ),
        "next_position": state.get("next_position"),
        "counts": counts,
        "pending_task": state.get("pending_task"),
        "rows": [latest[index] for index in sorted(latest)],
    }


def main() -> int:
    args = _parser().parse_args()
    repo = args.repo.expanduser().resolve()
    android_world_root = args.android_world_root.expanduser().resolve()
    manifest_path = args.function_manifest.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    scripts = Path(__file__).resolve().parent
    probe_script = (
        args.probe_script.expanduser().resolve()
        if args.probe_script
        else scripts / "run_offline_function_probe.py"
    )
    manifest_loader = (
        args.manifest_loader.expanduser().resolve()
        if args.manifest_loader
        else scripts / "run_cloud_function_onepass.py"
    )
    for required in (manifest_path, probe_script, manifest_loader):
        if not required.is_file():
            raise FileNotFoundError(required)

    tasks = _load_tasks(manifest_loader, manifest_path, repo)
    digest = _sha256(manifest_path)
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "campaign_state.json"
    state = _read(state_path)
    if state:
        _verify_state(state, args, digest)
    else:
        state = _initial_state(
            manifest_path=manifest_path,
            manifest_sha256=digest,
            tasks=tasks,
            args=args,
        )
        _write(state_path, state)

    state["status"] = "running"
    state.pop("pending_task", None)
    start_position = int(state.get("next_position") or 0)
    for position in range(start_position, len(tasks)):
        task = tasks[position]
        index = int(task["index"])
        task_name = str(task["task_name"])
        seed = int(args.seed_base) + index
        task_root = output_root / "tasks" / f"{index:03d}_{task_name}"
        attempt_number = _attempt_number(task_root)
        attempt_root = task_root / f"attempt_{attempt_number:03d}"
        probe_root = attempt_root / "probe"
        attempt_root.mkdir(parents=True, exist_ok=False)
        command = [
            sys.executable,
            str(probe_script),
            "--repo",
            str(repo),
            "--android-world-root",
            str(android_world_root),
            "--enhancement-root",
            str(task["enhancement_root"]),
            "--store-path",
            str(task["store_path"]),
            "--task",
            task_name,
            "--eval-seed",
            str(seed),
            "--task-params-json",
            "{}",
            "--source-device",
            str(args.source_device),
            "--target-device",
            str(args.target_device),
            "--output-root",
            str(probe_root),
            "--adb-path",
            str(args.adb_path),
            "--max-steps",
            str(args.max_steps),
            "--timeout-sec",
            str(args.timeout_sec),
            "--model",
            str(args.model),
            "--recall-model",
            str(args.model),
            "--no-emulator-setup",
        ]
        started_at = dt.datetime.now(dt.timezone.utc).isoformat()
        completed = subprocess.run(
            command,
            cwd=repo,
            env={
                **os.environ,
                "OMNIFLOW_RECALL_MODEL": str(args.model),
                "OMNIFLOW_RECALL_PROVIDER": "openai",
                "OMNIFLOW_REPO_ROOT": str(repo),
            },
            check=False,
            capture_output=True,
            text=True,
        )
        (attempt_root / "runner.log").write_text(
            "$ "
            + " ".join(command)
            + "\n\nSTDOUT\n"
            + completed.stdout
            + "\nSTDERR\n"
            + completed.stderr,
            encoding="utf-8",
        )
        probe = _read(probe_root / "probe_summary.json")
        classification = classify_probe(probe)
        classification.update(
            {
                "position": position,
                "index": index,
                "task": task_name,
                "seed": seed,
                "attempt": attempt_number,
                "started_at": started_at,
                "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "probe_returncode": completed.returncode,
                "probe_summary": str(probe_root / "probe_summary.json"),
                "store_path": str(task["store_path"]),
                "store_sha256": _sha256(Path(task["store_path"])),
            }
        )
        _write(attempt_root / "classification.json", classification)
        state["attempts"].append(classification)
        state["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        if classification["classification"] == "environment_failure":
            state["status"] = "environment_blocked"
            state["next_position"] = position
            state["pending_task"] = {
                "position": position,
                "index": index,
                "task": task_name,
                "seed": seed,
                "environment_reasons": classification["environment_reasons"],
            }
            _write(state_path, state)
            _write(output_root / "campaign_summary.json", _campaign_summary(state))
            print(json.dumps(state["pending_task"], ensure_ascii=False, indent=2))
            return 20
        state["next_position"] = position + 1
        _write(state_path, state)
        _write(output_root / "campaign_summary.json", _campaign_summary(state))

    state["status"] = "complete"
    state["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    state.pop("pending_task", None)
    _write(state_path, state)
    summary = _campaign_summary(state)
    _write(output_root / "campaign_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
