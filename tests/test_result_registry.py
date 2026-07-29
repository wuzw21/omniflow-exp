from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.experiment.result_registry import (
    load_summary_rows,
    registered_cell_plan,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_registered_cell(
    runs_root: Path,
    *,
    task: str,
    method: str,
    device: str,
    success: bool,
    validator_task_count: int = 1,
    validator_used: bool = True,
) -> None:
    cell = runs_root / task / method / device / "iteration_01"
    result_path = cell / "registered_result.json"
    manifest_path = cell / "registration_manifest.json"
    result = {
        "schema_version": "omniflow.androidworld_registered_result.v1",
        "registration_id": f"{task}.{method}.{device}",
        "attempt_id": "iteration_01",
        "task_name": task,
        "registration_manifest": str(manifest_path),
        "rows": [
            {
                "method": method,
                "device": device,
                "official_validator_used": validator_used,
                "official_validator_success": success,
                "official_validator_task_count": validator_task_count,
                "official_validator_coverage_rate": float(
                    validator_task_count > 0
                ),
            }
        ],
    }
    _write_json(result_path, result)
    _write_json(
        manifest_path,
        {
            "schema_version": "omniflow.androidworld_result_registration.v1",
            "registration_id": result["registration_id"],
            "attempt_id": "iteration_01",
            "task_name": task,
            "method": method,
            "device": device,
            "immutable": True,
            "registered_result_sha256": hashlib.sha256(
                result_path.read_bytes()
            ).hexdigest(),
        },
    )


def test_unregistered_one_task_summary_is_not_loaded(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "attempt" / "one_task_summary.json",
        {
            "task_name": "SystemBluetoothTurnOn",
            "rows": [
                {
                    "method": "ours",
                    "device": "small5554",
                    "official_validator_success": True,
                }
            ],
        },
    )

    assert load_summary_rows(tmp_path, {}, []) == []


def test_registered_result_requires_matching_immutable_manifest(
    tmp_path: Path,
) -> None:
    cell = tmp_path / "runs" / "cell"
    result_path = cell / "registered_result.json"
    manifest_path = cell / "registration_manifest.json"
    result = {
        "schema_version": "omniflow.androidworld_registered_result.v1",
        "registration_id": "registration-1",
        "attempt_id": "attempt-1",
        "task_name": "SystemBluetoothTurnOn",
        "registration_manifest": str(manifest_path),
        "rows": [
            {
                "method": "ours",
                "device": "small5554",
                "official_validator_success": True,
            }
        ],
    }
    _write_json(result_path, result)
    _write_json(
        manifest_path,
        {
            "schema_version": "omniflow.androidworld_result_registration.v1",
            "registration_id": "registration-1",
            "attempt_id": "attempt-1",
            "task_name": "SystemBluetoothTurnOn",
            "method": "ours",
            "device": "small5554",
            "immutable": True,
            "registered_result_sha256": hashlib.sha256(
                result_path.read_bytes()
            ).hexdigest(),
        },
    )

    rows = load_summary_rows(
        tmp_path / "runs",
        {"SystemBluetoothTurnOn": {"goal": "Turn Bluetooth on."}},
        [],
    )

    assert len(rows) == 1
    assert rows[0]["method"] == "ours"
    assert rows[0]["device_label"] == "small5554"

    result["rows"][0]["official_validator_success"] = False
    _write_json(result_path, result)
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_summary_rows(tmp_path / "runs", {}, [])


def test_registered_cell_plan_skips_any_cell_with_a_verified_conclusion(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    task = "AudioRecorderRecordAudioWithFileName"
    _write_registered_cell(
        runs_root,
        task=task,
        method="fixed_replay",
        device="small5554",
        success=True,
    )
    _write_registered_cell(
        runs_root,
        task=task,
        method="ours",
        device="fold5564",
        success=False,
    )

    plan = registered_cell_plan(
        runs_root=runs_root,
        task_name=task,
        methods=("fixed_replay", "ours"),
        devices=("small5554", "fold5564"),
    )

    assert plan["completed"] == [
        ("fixed_replay", "small5554"),
        ("ours", "fold5564"),
    ]
    assert plan["pending"] == [
        ("fixed_replay", "fold5564"),
        ("ours", "small5554"),
    ]


def test_registered_cell_plan_retries_rows_without_validator_coverage(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    task = "AudioRecorderRecordAudioWithFileName"
    _write_registered_cell(
        runs_root,
        task=task,
        method="ours",
        device="small5554",
        success=False,
        validator_task_count=0,
        validator_used=False,
    )

    plan = registered_cell_plan(
        runs_root=runs_root,
        task_name=task,
        methods=("ours",),
        devices=("small5554",),
    )

    assert plan["completed"] == []
    assert plan["pending"] == [("ours", "small5554")]


def test_registered_cell_plan_accepts_per_episode_validator_conclusion(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    task = "AudioRecorderRecordAudioWithFileName"
    _write_registered_cell(
        runs_root,
        task=task,
        method="fixed_replay",
        device="small5554",
        success=False,
        validator_task_count=0,
        validator_used=True,
    )

    plan = registered_cell_plan(
        runs_root=runs_root,
        task_name=task,
        methods=("fixed_replay",),
        devices=("small5554",),
    )

    assert plan["completed"] == [("fixed_replay", "small5554")]
    assert plan["pending"] == []
