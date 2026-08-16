from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from runlog_fixtures import androidworld_run_log

from src.experiment.artifact_memory import (
    load_artifact_memory,
    refresh_artifact_memory,
)
from src.experiment.result_registry import (
    build_master_progress,
    load_summary_rows,
    register_attempt_summary,
    registered_cell_plan,
)


def test_master_progress_usage_is_not_split_by_component() -> None:
    rows = build_master_progress(
        [
            {
                "task_name": "BrowserDraw",
                "method": "mobilegpt_offline_retrieval",
                "device_label": "small5554",
                "official_validator_success": "true",
                "is_latest_for_task_method": "true",
                "model_calls": "3",
                "total_tokens": "150",
                "chat_model_calls": "2",
                "embedding_model_calls": "1",
                "prompt_tokens": "120",
                "completion_tokens": "30",
            }
        ],
        {"BrowserDraw": {"task_index": 1}},
        [],
    )

    row = rows[0]
    assert row["mobilegpt_offline_retrieval_tool_calls"] == "3"
    assert row["mobilegpt_offline_retrieval_tokens"] == "150"
    for detailed_field in (
        "model_calls",
        "total_tokens",
        "chat_model_calls",
        "embedding_model_calls",
        "prompt_tokens",
        "completion_tokens",
    ):
        assert f"mobilegpt_offline_retrieval_{detailed_field}" not in row


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
    attempt: str = "iteration_01",
    validator_task_count: int = 1,
    validator_used: bool = True,
    source_seed: int = 111,
    evaluation_seed: int = 113,
    max_steps: int = 20,
    use_oob: bool = False,
    include_task_params: bool = True,
    legacy_fixed_replay: bool = False,
    include_uses_source_xml: bool = True,
    fixed_replay_backend: str = "recorded_coordinate_replay_v1",
    error: str = "",
) -> None:
    cell = runs_root / task / method / device / attempt
    result_path = cell / "registered_result.json"
    manifest_path = cell / "registration_manifest.json"
    task_params = {"seed": 1859998934}
    command = (
        "python -m src.integrations.android_world.launch "
        f"--task-random-seed {evaluation_seed} --max-steps {max_steps} "
        "--fixed-task-seed --perform-emulator-setup"
    )
    if use_oob:
        command += " --oob-observe-backend androidworld"
    result = {
        "schema_version": "omniflow.androidworld_registered_result.v1",
        "registration_id": f"{task}.{method}.{device}.{attempt}",
        "attempt_id": attempt,
        "task_name": task,
        "source_seed": source_seed,
        "evaluation_seed": evaluation_seed,
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
                "error": error,
                "task_random_seed": evaluation_seed,
                "max_steps": max_steps,
                "task_params": task_params,
                "task_params_sha256": (
                    hashlib.sha256(
                        json.dumps(task_params, sort_keys=True).encode("utf-8")
                    ).hexdigest()
                    if include_task_params
                    else None
                ),
                "state_backend": "androidworld",
                "execution_backend": (
                    "raw_coordinate_replay"
                    if method == "fixed_replay" and legacy_fixed_replay
                    else fixed_replay_backend
                    if method == "fixed_replay"
                    else None
                ),
                **(
                    {
                        "uses_source_xml": not legacy_fixed_replay,
                    }
                    if method == "fixed_replay" and include_uses_source_xml
                    else {}
                ),
                "fixed_task_seed": True,
                "fixed_task_params": False,
                "perform_emulator_setup": True,
                "serial": (
                    "emulator-5554"
                    if device in {"small5554", "target5554"}
                    else "emulator-5564"
                ),
                "console_port": (
                    5554
                    if device in {"small5554", "target5554"}
                    else 5564
                ),
                "command": command,
            }
        ],
    }
    _write_json(result_path, result)
    _write_json(
        manifest_path,
        {
            "schema_version": "omniflow.androidworld_result_registration.v1",
            "registration_id": result["registration_id"],
            "attempt_id": attempt,
            "task_name": task,
            "method": method,
            "device": device,
            "source_seed": source_seed,
            "evaluation_seed": evaluation_seed,
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
        source_seed=111,
        evaluation_seed=113,
        formal_max_steps=20,
    )

    assert plan["completed"] == [
        ("fixed_replay", "small5554"),
        ("ours", "fold5564"),
    ]
    assert plan["pending"] == [
        ("fixed_replay", "fold5564"),
        ("ours", "small5554"),
    ]


def test_registered_cell_plan_rejects_legacy_coordinate_fixed_replay(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    task = "AudioRecorderRecordAudio"
    _write_registered_cell(
        runs_root,
        task=task,
        method="fixed_replay",
        device="small5554",
        success=True,
        legacy_fixed_replay=True,
    )

    with pytest.raises(ValueError, match="formal_result_protocol_mismatch"):
        registered_cell_plan(
            runs_root=runs_root,
            task_name=task,
            methods=("fixed_replay",),
            devices=("small5554",),
            source_seed=111,
            evaluation_seed=113,
            formal_max_steps=20,
        )


def test_registered_cell_plan_rejects_previous_selector_stop_policy(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    task = "BrowserMaze"
    _write_registered_cell(
        runs_root,
        task=task,
        method="fixed_replay",
        device="small5554",
        success=False,
        fixed_replay_backend="selector_then_scaled_coordinate_replay",
    )

    with pytest.raises(ValueError, match="formal_result_protocol_mismatch"):
        registered_cell_plan(
            runs_root=runs_root,
            task_name=task,
            methods=("fixed_replay",),
            devices=("small5554",),
            source_seed=111,
            evaluation_seed=113,
            formal_max_steps=20,
        )


def test_registered_cell_plan_accepts_coordinate_replay_missing_redundant_audit_flag(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    task = "BrowserMaze"
    _write_registered_cell(
        runs_root,
        task=task,
        method="fixed_replay",
        device="small5554",
        success=False,
        include_uses_source_xml=False,
    )

    plan = registered_cell_plan(
        runs_root=runs_root,
        task_name=task,
        methods=("fixed_replay",),
        devices=("small5554",),
        source_seed=111,
        evaluation_seed=113,
        formal_max_steps=20,
    )

    assert plan["completed"] == [("fixed_replay", "small5554")]
    assert plan["pending"] == []


def test_registered_cell_plan_does_not_skip_a_different_evaluation_seed(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    task = "AudioRecorderRecordAudioWithFileName"
    _write_registered_cell(
        runs_root,
        task=task,
        method="ours",
        device="small5554",
        success=True,
        source_seed=111,
        evaluation_seed=113,
    )

    plan = registered_cell_plan(
        runs_root=runs_root,
        task_name=task,
        methods=("ours",),
        devices=("small5554",),
        source_seed=111,
        evaluation_seed=114,
    )

    assert plan["completed"] == []
    assert plan["pending"] == [("ours", "small5554")]


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
        source_seed=111,
        evaluation_seed=113,
    )

    assert plan["completed"] == []
    assert plan["pending"] == [("ours", "small5554")]


def test_registered_cell_plan_retries_validator_rows_with_environment_error(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    task = "NotesIsTodo"
    _write_registered_cell(
        runs_root,
        task=task,
        method="fixed_replay",
        device="fold5564",
        success=False,
        error="FileNotFoundError: app database missing",
    )

    plan = registered_cell_plan(
        runs_root=runs_root,
        task_name=task,
        methods=("fixed_replay",),
        devices=("fold5564",),
        source_seed=111,
        evaluation_seed=113,
    )

    assert plan["completed"] == []
    assert plan["pending"] == [("fixed_replay", "fold5564")]


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
        source_seed=111,
        evaluation_seed=113,
    )

    assert plan["completed"] == [("fixed_replay", "small5554")]
    assert plan["pending"] == []


def test_registered_cell_plan_rejects_incompatible_formal_protocol(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    task = "BrowserDraw"
    _write_registered_cell(
        runs_root,
        task=task,
        method="t3a_hint",
        device="small5554",
        success=False,
        max_steps=30,
        use_oob=True,
        include_task_params=False,
    )

    with pytest.raises(ValueError, match="formal_result_protocol_mismatch"):
        registered_cell_plan(
            runs_root=runs_root,
            task_name=task,
            methods=("t3a_hint",),
            devices=("small5554",),
            source_seed=111,
            evaluation_seed=113,
            formal_max_steps=20,
        )


def test_registered_cell_plan_uses_earliest_validator_conclusion(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    task = "BrowserDraw"
    for attempt, max_steps in (("iteration_01", 20), ("iteration_02", 30)):
        _write_registered_cell(
            runs_root,
            task=task,
            method="t3a_hint",
            device="small5554",
            success=False,
            attempt=attempt,
            max_steps=max_steps,
        )

    assert registered_cell_plan(
        runs_root=runs_root,
        task_name=task,
        methods=("t3a_hint",),
        devices=("small5554",),
        source_seed=111,
        evaluation_seed=113,
        formal_max_steps=20,
    )["completed"] == [("t3a_hint", "small5554")]


def test_result_registration_updates_long_term_memory(tmp_path: Path) -> None:
    source_run_log = tmp_path / "evidence" / "TaskOne" / "source.run_log.json"
    _write_json(
        source_run_log,
        androidworld_run_log(
            [{"action_type": "wait"}],
            task_name="TaskOne",
            goal="Complete task one.",
            with_pixels=True,
        ),
    )
    source_index = tmp_path / "source_index.json"
    _write_json(
        source_index,
        {
            "TaskOne": {
                "task": "TaskOne",
                "retained_source_run_log": str(source_run_log),
            }
        },
    )
    memory_root = tmp_path / "memory"
    refresh_artifact_memory(
        memory_root=memory_root,
        source_index=source_index,
        function_catalogs=(),
        runlog_roots=(tmp_path / "evidence",),
        result_roots=(),
    )
    results_root = tmp_path / "results"
    summary = results_root / "attempts" / "TaskOne" / "one_task_summary.json"
    _write_json(
        summary,
        {
            "task_name": "TaskOne",
            "rows": [
                {
                    "task_name": "TaskOne",
                    "method": "ours",
                    "device": "small5554",
                    "serial": "emulator-5554",
                    "console_port": 5554,
                    "official_validator_used": True,
                    "official_validator_success": False,
                    "official_validator_task_count": 1,
                    "task_random_seed": 113,
                    "max_steps": 20,
                    "task_params": {},
                    "task_params_sha256": hashlib.sha256(
                        json.dumps({}, sort_keys=True).encode("utf-8")
                    ).hexdigest(),
                    "state_backend": "androidworld",
                    "fixed_task_seed": True,
                    "fixed_task_params": False,
                    "perform_emulator_setup": True,
                    "command": (
                        "python -m src.integrations.android_world.launch "
                        "--task-random-seed 113 --max-steps 20 "
                        "--fixed-task-seed --perform-emulator-setup"
                    ),
                }
            ],
        },
    )
    attempt_manifest = summary.parent / "attempt_manifest.json"
    _write_json(
        attempt_manifest,
        {
            "immutable": True,
            "attempt_id": "iteration_01",
            "source_seed": 111,
            "evaluation_seed": 113,
        },
    )

    registration = register_attempt_summary(
        summary_path=summary,
        attempt_manifest_path=attempt_manifest,
        runs_root=results_root / "androidworld_validator" / "runs",
        master_root=results_root / "androidworld_validator" / "master_progress",
        source_index_path=source_index,
        artifact_memory_index=memory_root / "current.json",
    )

    assert registration["artifact_memory_updated"] is True
    memory = load_artifact_memory(memory_root / "current.json")
    cell = memory["canonical"]["result_cells"][
        "TaskOne|ours|small5554|111|113"
    ]
    assert cell["official_validator_success"] is False
