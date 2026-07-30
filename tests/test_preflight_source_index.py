from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from src.experiment.preflight import (
    _dismiss_known_accessibility_crash_dialog,
    _validate_source_index,
)
from runlog_fixtures import androidworld_run_log


def test_preflight_dismisses_known_accessibility_crash_dialog(monkeypatch) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], timeout: float = 10.0):
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "Window launcher isVisible=true"
                if command[-4:] == ["shell", "dumpsys", "window", "windows"]
                else "Broadcast completed: result=0"
            ),
            stderr="",
        )

    monkeypatch.setattr("src.experiment.preflight._run", fake_run)
    monkeypatch.setattr("src.experiment.preflight.time.sleep", lambda _: None)

    refreshed = _dismiss_known_accessibility_crash_dialog(
        "/sdk/adb",
        "emulator-5560",
        "Window Application Error: com.google.androidenv.accessibilityforwarder",
    )

    assert refreshed == "Window launcher isVisible=true"
    assert commands[0] == [
        "/sdk/adb",
        "-s",
        "emulator-5560",
        "shell",
        "am",
        "broadcast",
        "-a",
        "android.intent.action.CLOSE_SYSTEM_DIALOGS",
    ]
    assert commands[1][-4:] == ["shell", "dumpsys", "window", "windows"]


def test_preflight_preserves_unknown_crash_dialog(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.experiment.preflight._run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unknown crash dialogs must not be dismissed")
        ),
    )
    focused_windows = "Window Application Error: com.example.app"

    assert (
        _dismiss_known_accessibility_crash_dialog(
            "/sdk/adb",
            "emulator-5560",
            focused_windows,
        )
        == focused_windows
    )


def _write_index(
    root: Path,
    *,
    method: str = "ours",
    official_success: bool = True,
    source_kind: str = "androidworld_validator_success_source_runlog",
) -> Path:
    root.mkdir(parents=True)
    run_log = root / "source.run_log.json"
    run_log.write_text(
        json.dumps(
            androidworld_run_log(
                [{"action_type": "wait"}],
                task_name="Task",
                goal="Complete Task.",
                with_pixels=True,
            )
        ),
        encoding="utf-8",
    )
    index = root / "index_by_task.json"
    index.write_text(
        json.dumps(
            {
                "Task": {
                    "source_seed": 111,
                    "method": method,
                    "latest_official_success_source": official_success,
                    "source_kind": source_kind,
                    "retained_source_run_log": run_log.name,
                    "source_run_log_sha256": hashlib.sha256(
                        run_log.read_bytes()
                    ).hexdigest(),
                }
            }
        ),
        encoding="utf-8",
    )
    return index


def test_source_index_requires_strict_frozen_provenance(tmp_path: Path) -> None:
    index = _write_index(tmp_path / "valid")

    result = _validate_source_index(
        index,
        source_root=tmp_path,
        expected_tasks=1,
    )

    assert result["task_count"] == 1
    assert result["run_log_count"] == 1


def test_source_index_accepts_recorded_seed_outside_protocol(tmp_path: Path) -> None:
    index = _write_index(tmp_path / "recorded-seed")
    payload = json.loads(index.read_text(encoding="utf-8"))
    payload["Task"]["source_seed"] = 3936510006
    index.write_text(json.dumps(payload), encoding="utf-8")

    result = _validate_source_index(
        index,
        source_root=tmp_path,
        expected_tasks=1,
    )

    assert result["run_log_count"] == 1


def test_source_index_rejects_runlog_changed_after_freeze(
    tmp_path: Path,
) -> None:
    index = _write_index(tmp_path / "tampered")
    run_log = index.with_name("source.run_log.json")
    payload = json.loads(run_log.read_text(encoding="utf-8"))
    payload["goal"] = "Changed after indexing."
    run_log.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="source_index_invalid_tasks"):
        _validate_source_index(
            index,
            source_root=tmp_path,
            expected_tasks=1,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("official_success", False),
        ("source_kind", "other"),
    ],
)
def test_source_index_rejects_invalid_frozen_source(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    kwargs = {field: value}
    index = _write_index(tmp_path / field, **kwargs)

    with pytest.raises(ValueError, match="source_index_invalid_tasks"):
        _validate_source_index(
            index,
            source_root=tmp_path,
            expected_tasks=1,
        )


def test_source_index_preserves_non_ours_source_method(tmp_path: Path) -> None:
    index = _write_index(tmp_path / "fixed", method="fixed_replay")

    result = _validate_source_index(
        index,
        source_root=tmp_path,
        expected_tasks=1,
    )

    assert result["run_log_count"] == 1


def test_source_index_rejects_registered_historical_runlog(
    tmp_path: Path,
) -> None:
    index = _write_index(tmp_path / "historical")
    run_log = index.with_name("source.run_log.json")
    run_log.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.run_log.v1",
                "run_id": "historical-source",
                "goal": "Complete Task.",
                "completed": True,
                "success": True,
                "steps": [
                    {
                        "step_index": 0,
                        "observation_before_act": {
                            "state_id": "state-0",
                            "width": 100,
                            "height": 100,
                        },
                        "executed_actions": [
                            {
                                "type": "wait",
                                "params": {"time_ms": 100},
                            }
                        ],
                        "success": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    payload = json.loads(index.read_text(encoding="utf-8"))
    row = payload["Task"]
    row.pop("source_kind")
    row.pop("source_run_log_sha256")
    row["retained_source_run_log_sha256"] = hashlib.sha256(
        run_log.read_bytes()
    ).hexdigest()
    index.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="source_index_invalid_tasks"):
        _validate_source_index(
            index,
            source_root=tmp_path,
            expected_tasks=1,
        )


def test_source_index_validates_only_selected_task_for_one_task_run(
    tmp_path: Path,
) -> None:
    index = _write_index(tmp_path / "selected")
    payload = json.loads(index.read_text(encoding="utf-8"))
    payload["UnrelatedInvalidTask"] = {
        "source_seed": 111,
        "latest_official_success_source": True,
        "retained_source_run_log": "missing.run_log.json",
        "retained_source_run_log_sha256": "0" * 64,
    }
    index.write_text(json.dumps(payload), encoding="utf-8")

    result = _validate_source_index(
        index,
        source_root=tmp_path,
        expected_tasks=2,
        task_names=("Task",),
    )

    assert result["task_count"] == 2
    assert result["run_log_count"] == 1
