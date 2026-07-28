from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omniflow import compile_runlog_to_store
from scripts.androidworld_replay_pipeline import validate_ours_transfer_assets
from scripts.migrate_androidworld_source_replay_runlog import (
    load_androidworld_app_name_to_package,
    migrate_source_replay_runlog,
)
from scripts.rebuild_transfer_store_from_source_replay import rebuild_transfer_store


SCHEMA_VERSION = "omniflow.androidworld-full-function-assets.v1"
SOURCE_DEVICE = {
    "label": "source5560",
    "serial": "emulator-5560",
    "console_port": 5560,
    "api_level": 33,
    "width": 720,
    "height": 1280,
    "density": 320,
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._") or "item"


def _absolute_executable(path: str | Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _directory_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"empty_directory:{root}")
    for path in files:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _omnitransfer_identity(root: Path) -> dict[str, Any]:
    package = root / "src/omnitransfer"
    if not package.is_dir():
        raise RuntimeError(f"omnitransfer_root_missing:{root}")
    try:
        revision = _git(root, "rev-parse", "HEAD")
    except subprocess.CalledProcessError:
        revision = ""
    if revision:
        if _git(root, "status", "--short"):
            raise RuntimeError("omnitransfer_repo_not_clean")
        kind = "clean_git_checkout"
    else:
        match = re.fullmatch(r"OmniTransfer-embedded-([0-9a-f]{8,40})", root.name)
        if match is None:
            raise RuntimeError("omnitransfer_immutable_release_name_required")
        revision = match.group(1)
        kind = "immutable_embedded_release"
    return {
        "root": str(root),
        "kind": kind,
        "revision": revision,
        "tree_sha256": _directory_sha256(root / "src"),
    }


def _next_attempt(root: Path, *, prefix: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", prefix):
        raise ValueError(f"invalid_attempt_prefix:{prefix}")
    pattern = re.compile(rf"{re.escape(prefix)}_(\d+)")
    existing = [
        int(match.group(1))
        for path in root.glob(f"{prefix}_*")
        if (match := pattern.fullmatch(path.name))
    ]
    return root / f"{prefix}_{max(existing, default=0) + 1:03d}"


def _adb(adb_path: Path, serial: str, *args: str) -> str:
    completed = subprocess.run(
        [str(adb_path), "-s", serial, *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    return completed.stdout.replace("\r", "").strip()


def _number_after_colon(value: str) -> int:
    match = re.search(r":\s*(\d+)", value)
    if match is None:
        raise ValueError(f"device_number_missing:{value}")
    return int(match.group(1))


def _device_snapshot(adb_path: Path) -> dict[str, Any]:
    serial = str(SOURCE_DEVICE["serial"])
    if _adb(adb_path, serial, "get-state") != "device":
        raise RuntimeError(f"source_device_offline:{serial}")
    if _adb(adb_path, serial, "shell", "getprop", "sys.boot_completed") != "1":
        raise RuntimeError(f"source_device_not_booted:{serial}")
    api_level = int(_adb(adb_path, serial, "shell", "getprop", "ro.build.version.sdk"))
    size_text = _adb(adb_path, serial, "shell", "wm", "size")
    density_text = _adb(adb_path, serial, "shell", "wm", "density")
    size_match = re.search(r"Physical size:\s*(\d+)x(\d+)", size_text)
    if size_match is None:
        raise RuntimeError(f"source_device_size_invalid:{size_text}")
    snapshot = {
        **SOURCE_DEVICE,
        "api_level": api_level,
        "width": int(size_match.group(1)),
        "height": int(size_match.group(2)),
        "density": _number_after_colon(density_text),
        "boot_completed": True,
    }
    expected = {
        key: SOURCE_DEVICE[key]
        for key in ("api_level", "width", "height", "density")
    }
    actual = {key: snapshot[key] for key in expected}
    if actual != expected:
        raise RuntimeError(f"source_device_profile_mismatch:{actual}:expected={expected}")
    return snapshot


def _source_path(canonical_repo: Path, row: dict[str, Any]) -> Path:
    retained = str(row.get("retained_source_run_log") or "").strip()
    if not retained:
        raise ValueError("retained_source_run_log_required")
    path = Path(retained).expanduser()
    if not path.is_absolute():
        path = canonical_repo / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _task_index(
    *,
    task: str,
    inventory_row: dict[str, Any],
    migrated_run_log: Path,
    output_path: Path,
) -> Path:
    row = json.loads(json.dumps(inventory_row))
    row["retained_source_run_log"] = str(migrated_run_log.resolve())
    row["source_asset_migration"] = "native-source-canonicalization"
    _write_json(output_path, {task: row})
    return output_path


def _run_source_replay(
    *,
    args: argparse.Namespace,
    task: str,
    task_index: Path,
    replay_root: Path,
    log_path: Path,
) -> tuple[int, float, list[str]]:
    command = [
        str(args.python),
        str(args.release_repo / "scripts/androidworld_replay_pipeline.py"),
        "one-task",
        "--index",
        str(task_index),
        "--master-source-index",
        str(args.inventory_index),
        "--android-world-root",
        str(args.android_world_root),
        "--tasks",
        task,
        "--methods",
        "fixed_replay",
        "--device-targets",
        "source5560:emulator-5560:5560",
        "--output-root",
        str(replay_root),
        "--result-registry-root",
        str(args.output_root / "isolated_registry/runs"),
        "--master-progress-root",
        str(args.output_root / "isolated_registry/master_progress"),
        "--task-random-seed",
        "111",
        "--perform-emulator-setup",
        "--adb-path",
        str(args.adb_path),
        "--timeout-sec",
        str(args.timeout_sec),
        "--fail-fast",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "OMNIFLOW_RAW_REPLAY_CAPTURE_OBSERVATIONS": "1",
            "OMNIFLOW_OBSERVE_BACKEND": "androidworld",
            "OMNITRANSFER_ROOT": str(args.omnitransfer_root),
        }
    )
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=args.release_repo,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=args.timeout_sec + 180,
        )
    return completed.returncode, time.monotonic() - started, command


def _audit_source_replay(
    *,
    task: str,
    migrated_run_log: Path,
    replay_root: Path,
    registry_root: Path,
) -> dict[str, Any]:
    task_root = replay_root / "source_seed_111" / _safe_name(task)
    summary_path = task_root / "one_task_summary.json"
    summary = _read_json(summary_path)
    rows = summary.get("rows")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError("source_replay_single_row_required")
    row = rows[0]
    if row.get("task_name") != task:
        raise ValueError("source_replay_task_mismatch")
    if row.get("method") != "fixed_replay" or row.get("device") != "source5560":
        raise ValueError("source_replay_identity_mismatch")
    if row.get("official_validator_used") is not True:
        raise ValueError("source_replay_official_validator_required")
    if row.get("official_validator_success") is not True:
        raise RuntimeError("source_replay_official_failure")
    if any(int(row.get(key) or 0) != 0 for key in (
        "model_calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    )):
        raise ValueError("source_replay_unexpected_model_usage")
    if row.get("token_usage_status") != "not_applicable":
        raise ValueError("source_replay_token_status_invalid")
    if float(row.get("wall_sec") or 0) < float(row.get("duration_sec") or 0):
        raise ValueError("source_replay_time_order_invalid")

    run_root = task_root / "fixed_replay/source5560"
    sidecar_path = run_root / "raw_replay_result.json"
    task_results_path = run_root / "task_results.jsonl"
    sidecar = _read_json(sidecar_path)
    if sidecar.get("completed") is not True or sidecar.get("replay_completed") is not True:
        raise ValueError("source_replay_incomplete")
    run_log = sidecar.get("run_log")
    if not isinstance(run_log, dict):
        raise ValueError("source_replay_run_log_required")
    if run_log.get("device_label") != "source5560":
        raise ValueError("source_replay_device_label_invalid")
    steps = run_log.get("steps")
    if not isinstance(steps, list) or len(steps) != 1 or not isinstance(steps[0], dict):
        raise ValueError("source_replay_provider_step_required")
    provider_detail = steps[0].get("provider_detail")
    detail = (
        provider_detail.get("raw_replay")
        if isinstance(provider_detail, dict)
        else None
    )
    if not isinstance(detail, dict):
        raise ValueError("source_replay_detail_required")
    replayed_source_run_log = Path(
        str(detail.get("source_run_log") or "")
    ).expanduser().resolve()
    if not replayed_source_run_log.is_file():
        raise FileNotFoundError(replayed_source_run_log)
    if _sha256(replayed_source_run_log) != _sha256(migrated_run_log):
        raise ValueError("source_replay_input_hash_mismatch")
    if detail.get("oob_prepare") is not None:
        raise ValueError("source_replay_oob_forbidden")
    step_results = detail.get("step_results")
    if not isinstance(step_results, list) or not step_results:
        raise ValueError("source_replay_step_results_required")
    observations = []
    for index, step in enumerate(step_results):
        if not isinstance(step, dict) or step.get("completed") is not True:
            raise ValueError(f"source_replay_step_incomplete:{index}")
        if step.get("skipped") is True:
            raise ValueError(f"source_replay_step_skipped:{index}")
        observations.append(step.get("observation_before_act"))
    observations.append(detail.get("final_observation"))
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            raise ValueError(f"source_replay_observation_missing:{index}")
        if observation.get("provider") != "androidworld":
            raise ValueError(f"source_replay_observation_not_native:{index}")
        if not str(observation.get("xml") or "").strip():
            raise ValueError(f"source_replay_observation_xml_missing:{index}")
        if int(observation.get("width") or 0) <= 0 or int(
            observation.get("height") or 0
        ) <= 0:
            raise ValueError(f"source_replay_observation_display_missing:{index}")

    registration = (
        registry_root
        / task
        / "fixed_replay"
        / "source5560"
        / replay_root.name
    )
    registration_manifest_path = registration / "registration_manifest.json"
    registered_result_path = registration / "registered_result.json"
    registration_manifest = _read_json(registration_manifest_path)
    if registration_manifest.get("immutable") is not True:
        raise ValueError("source_replay_registration_not_immutable")
    if registration_manifest.get("source_summary_sha256") != _sha256(summary_path):
        raise ValueError("source_replay_summary_hash_mismatch")
    if registration_manifest.get("registered_result_sha256") != _sha256(
        registered_result_path
    ):
        raise ValueError("source_replay_registered_result_hash_mismatch")
    return {
        "row": row,
        "summary_path": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": _sha256(sidecar_path),
        "task_results_path": str(task_results_path),
        "task_results_sha256": _sha256(task_results_path),
        "replayed_source_run_log": str(replayed_source_run_log),
        "replayed_source_run_log_sha256": _sha256(replayed_source_run_log),
        "step_count": len(step_results),
        "observation_count": len(observations),
        "registration": str(registration),
        "registration_manifest_sha256": _sha256(registration_manifest_path),
        "registered_result_sha256": _sha256(registered_result_path),
    }


def _asset_index(output_root: Path, tasks: list[str]) -> dict[str, Any]:
    assets: dict[str, Any] = {}
    for task in tasks:
        status_path = output_root / "tasks" / _safe_name(task) / "status.json"
        if not status_path.is_file():
            continue
        status = _read_json(status_path)
        result = status.get("result")
        if not isinstance(result, dict) or result.get("classification") != "completed":
            continue
        assets[task] = {
            key: result[key]
            for key in (
                "store_path",
                "store_sha256",
                "transfer_state_catalog",
                "transfer_state_catalog_sha256",
                "provenance_manifest",
                "provenance_manifest_sha256",
                "source_run_log",
                "source_run_log_sha256",
                "source_task_params",
                "source_task_params_sha256",
                "prep_total_wall_sec",
            )
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "task_count": len(tasks),
        "completed_task_count": len(assets),
        "assets": assets,
    }


def _write_progress(output_root: Path, tasks: list[str], *, current_task: str = "") -> None:
    index = _asset_index(output_root, tasks)
    _write_json(output_root / "asset_index.json", index)
    _write_json(
        output_root / "progress.json",
        {
            "schema_version": SCHEMA_VERSION,
            "completed_tasks": index["completed_task_count"],
            "total_tasks": len(tasks),
            "current_task": current_task,
        },
    )


def _build_one(
    *,
    args: argparse.Namespace,
    task: str,
    inventory_row: dict[str, Any],
    attempt_root: Path,
) -> dict[str, Any]:
    total_started = time.monotonic()
    source_path = _source_path(args.canonical_repo, inventory_row)
    app_mapping, app_mapping_source = load_androidworld_app_name_to_package(
        args.android_world_root
    )

    started = time.monotonic()
    migration = migrate_source_replay_runlog(
        source_path=source_path,
        output_root=attempt_root / "migrated_source",
        drop_clear_text=True,
        app_name_to_package=app_mapping,
        app_mapping_source=app_mapping_source,
    )
    migration_wall_sec = time.monotonic() - started
    migrated_run_log = Path(str(migration["output_run_log"])).resolve()

    task_index = _task_index(
        task=task,
        inventory_row=inventory_row,
        migrated_run_log=migrated_run_log,
        output_path=attempt_root / "task_index.json",
    )

    started = time.monotonic()
    compile_result = compile_runlog_to_store(
        migrated_run_log,
        attempt_root / "draft_store",
    )
    compile_wall_sec = time.monotonic() - started
    function_ids = list(compile_result.get("function_ids") or [])
    if len(function_ids) != 1:
        raise ValueError(f"single_complete_function_required:{len(function_ids)}")
    complete_function_id = str(function_ids[0])
    draft_store = Path(str(compile_result["store_path"])).resolve()

    replay_root = attempt_root / f"{attempt_root.name}_source_replay"
    returncode, source_replay_command_wall_sec, command = _run_source_replay(
        args=args,
        task=task,
        task_index=task_index,
        replay_root=replay_root,
        log_path=attempt_root / "source_replay.log",
    )
    _write_json(
        attempt_root / "source_replay_command.json",
        {"command": command, "returncode": returncode},
    )
    if returncode != 0:
        raise RuntimeError(f"source_replay_command_failed:{returncode}")
    source_audit = _audit_source_replay(
        task=task,
        migrated_run_log=migrated_run_log,
        replay_root=replay_root,
        registry_root=args.output_root / "isolated_registry/runs",
    )

    started = time.monotonic()
    rebuild = rebuild_transfer_store(
        store_path=draft_store,
        source_run_log_path=source_audit["replayed_source_run_log"],
        source_replay_path=source_audit["sidecar_path"],
        task_results_path=source_audit["task_results_path"],
        complete_function_id=complete_function_id,
        output_root=attempt_root / "ours_store",
    )
    rebuild_wall_sec = time.monotonic() - started

    started = time.monotonic()
    transfer_audit = validate_ours_transfer_assets(rebuild["output_store"])
    validation_wall_sec = time.monotonic() - started
    if transfer_audit.get("complete") is not True:
        raise ValueError("rebuilt_transfer_assets_incomplete")
    if transfer_audit.get("source_target_audit_complete") is not True:
        raise ValueError("rebuilt_source_target_audit_incomplete")

    provenance_manifest = attempt_root / "ours_store/provenance_manifest.json"
    row = source_audit["row"]
    result = {
        "schema_version": SCHEMA_VERSION,
        "classification": "completed",
        "task": task,
        "attempt_root": str(attempt_root),
        "source_seed": 111,
        "source_device": dict(SOURCE_DEVICE),
        "source_run_log_input": str(source_path),
        "source_run_log_input_sha256": _sha256(source_path),
        "migrated_source_run_log": str(migrated_run_log),
        "migrated_source_run_log_sha256": _sha256(migrated_run_log),
        "source_run_log": source_audit["replayed_source_run_log"],
        "source_run_log_sha256": source_audit[
            "replayed_source_run_log_sha256"
        ],
        "source_task_params": row.get("task_params"),
        "source_task_params_sha256": row.get("task_params_sha256"),
        "source_official_validator_success": True,
        "source_actions_executed": int(row.get("actions_executed") or 0),
        "source_episode_sec": float(row.get("duration_sec") or 0),
        "source_cell_e2e_sec": float(row.get("wall_sec") or 0),
        "source_replay_command_wall_sec": round(source_replay_command_wall_sec, 6),
        "source_observation_count": source_audit["observation_count"],
        "prep_model_calls": 0,
        "prep_prompt_tokens": 0,
        "prep_completion_tokens": 0,
        "prep_total_tokens": 0,
        "prep_token_usage_status": "not_applicable",
        "migration_wall_sec": round(migration_wall_sec, 6),
        "compile_wall_sec": round(compile_wall_sec, 6),
        "rebuild_wall_sec": round(rebuild_wall_sec, 6),
        "validation_wall_sec": round(validation_wall_sec, 6),
        "prep_total_wall_sec": round(time.monotonic() - total_started, 6),
        "complete_function_id": complete_function_id,
        "store_path": rebuild["output_store"],
        "store_sha256": rebuild["output_store_sha256"],
        "transfer_state_catalog": rebuild["output_transfer_state_catalog"],
        "transfer_state_catalog_sha256": rebuild[
            "output_transfer_state_catalog_sha256"
        ],
        "provenance_manifest": str(provenance_manifest),
        "provenance_manifest_sha256": _sha256(provenance_manifest),
        "migration_manifest": migration,
        "source_replay_audit": source_audit,
        "transfer_asset_audit": transfer_audit,
        "target_inputs_read": False,
    }
    _write_json(attempt_root / "asset_result.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build audited source-only Function transfer assets for all tasks."
    )
    parser.add_argument("--release-repo", required=True, type=Path)
    parser.add_argument("--canonical-repo", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--python", required=True)
    parser.add_argument("--android-world-root", required=True, type=Path)
    parser.add_argument("--inventory-index", required=True, type=Path)
    parser.add_argument("--omnitransfer-root", required=True, type=Path)
    parser.add_argument("--adb-path", required=True, type=Path)
    parser.add_argument("--expected-tasks", type=int, default=116)
    parser.add_argument("--tasks", default="")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.release_repo = args.release_repo.expanduser().resolve()
    args.canonical_repo = args.canonical_repo.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.python = _absolute_executable(args.python)
    args.android_world_root = args.android_world_root.expanduser().resolve()
    args.inventory_index = args.inventory_index.expanduser().resolve()
    args.omnitransfer_root = args.omnitransfer_root.expanduser().resolve()
    args.adb_path = args.adb_path.expanduser().resolve()

    inventory = _read_json(args.inventory_index)
    if len(inventory) != args.expected_tasks:
        raise ValueError(
            f"expected_{args.expected_tasks}_source_tasks:actual={len(inventory)}"
        )
    tasks = sorted(inventory)
    if args.tasks:
        selected = [item.strip() for item in args.tasks.split(",") if item.strip()]
        missing = sorted(set(selected) - set(tasks))
        if missing:
            raise ValueError(f"unknown_tasks:{','.join(missing)}")
        tasks = selected
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit_must_be_positive")
        tasks = tasks[: args.limit]

    release_revision = _git(args.release_repo, "rev-parse", "HEAD")
    if _git(args.release_repo, "status", "--short"):
        raise RuntimeError("release_repo_not_clean")
    transfer_identity = _omnitransfer_identity(args.omnitransfer_root)
    os.environ["OMNITRANSFER_ROOT"] = str(args.omnitransfer_root)
    snapshot = _device_snapshot(args.adb_path)

    args.output_root.mkdir(parents=True, exist_ok=True)
    frozen_inventory = args.output_root / "frozen_inputs/source_index_seed_111.json"
    if frozen_inventory.exists():
        if _sha256(frozen_inventory) != _sha256(args.inventory_index):
            raise ValueError("frozen_inventory_hash_mismatch")
    else:
        frozen_inventory.parent.mkdir(parents=True, exist_ok=True)
        frozen_inventory.write_bytes(args.inventory_index.read_bytes())
    args.inventory_index = frozen_inventory
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "release_repo": str(args.release_repo),
        "release_revision": release_revision,
        "canonical_repo": str(args.canonical_repo),
        "python": str(args.python),
        "android_world_root": str(args.android_world_root),
        "inventory_index": str(frozen_inventory),
        "inventory_sha256": _sha256(frozen_inventory),
        "omnitransfer_root": str(args.omnitransfer_root),
        "omnitransfer_identity": transfer_identity,
        "source_seed": 111,
        "source_device": snapshot,
        "task_count": len(tasks),
        "tasks": tasks,
        "uses_oob": False,
        "target_inputs_read": False,
    }
    manifest_path = args.output_root / "batch_manifest.json"
    if manifest_path.exists():
        if _read_json(manifest_path) != manifest:
            raise ValueError("batch_manifest_differs")
    else:
        _write_json(manifest_path, manifest)

    _write_progress(args.output_root, tasks)
    if args.dry_run:
        return 0
    for task in tasks:
        task_root = args.output_root / "tasks" / _safe_name(task)
        status_path = task_root / "status.json"
        if status_path.is_file():
            status = _read_json(status_path)
            result = status.get("result")
            if isinstance(result, dict) and result.get("classification") == "completed":
                continue
        _device_snapshot(args.adb_path)
        attempt = _next_attempt(
            task_root / "attempts",
            prefix=f"{_safe_name(args.output_root.name)}_attempt",
        )
        attempt.mkdir(parents=True, exist_ok=False)
        _write_progress(args.output_root, tasks, current_task=task)
        try:
            result = _build_one(
                args=args,
                task=task,
                inventory_row=dict(inventory[task]),
                attempt_root=attempt,
            )
        except Exception as error:
            result = {
                "schema_version": SCHEMA_VERSION,
                "classification": "asset_build_failure",
                "task": task,
                "attempt_root": str(attempt),
                "error_type": type(error).__name__,
                "error": str(error),
            }
            _write_json(attempt / "asset_failure.json", result)
            _write_json(status_path, {"schema_version": SCHEMA_VERSION, "result": result})
            _write_progress(args.output_root, tasks)
            raise
        _write_json(status_path, {"schema_version": SCHEMA_VERSION, "result": result})
        _write_progress(args.output_root, tasks)
        print(
            f"[asset] task={task} complete "
            f"episode_sec={result['source_episode_sec']:.3f} "
            f"prep_sec={result['prep_total_wall_sec']:.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
