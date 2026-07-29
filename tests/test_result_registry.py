from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.experiment.result_registry import load_summary_rows


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
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
