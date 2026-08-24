from __future__ import annotations

import json
from pathlib import Path

from src.experiment.batch_outcomes import (
    concluded_result_keys,
    record_result_outcome,
    summarize_results,
)
from src.experiment.mobilegpt_contract import MOBILEGPT_SOURCE_METHOD
from src.integrations.android_world.run_episode import _summarize_task_results


def test_record_prep_failure_preserves_reason_tokens_and_time(tmp_path: Path) -> None:
    source_attempt = tmp_path / "source_attempt"
    source_attempt.mkdir()
    (source_attempt / "prep_failure.json").write_text(
        json.dumps(
            {
                "schema_version": "omniflow.mobilegpt.memory-failure.v2",
                "error_type": "RuntimeError",
                "error": "mobilegpt_cold_memory_official_source_failed",
                "retry_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    (source_attempt / "source_stats.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "task_started",
                        "ts": "2026-08-01T00:00:00+00:00",
                    }
                ),
                json.dumps(
                    {
                        "event": "chat_call",
                        "prompt_tokens": 100,
                        "completion_tokens": 20,
                        "total_tokens": 120,
                        "ts": "2026-08-01T00:00:01+00:00",
                    }
                ),
                json.dumps(
                    {
                        "event": "mobilegpt_action_sent",
                        "ts": "2026-08-01T00:00:02+00:00",
                    }
                ),
                json.dumps(
                    {
                        "event": "task_finished",
                        "elapsed_sec": 12.5,
                        "ts": "2026-08-01T00:00:13+00:00",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    task_log = tmp_path / "task.log"
    task_log.write_text(
        "RuntimeError: mobilegpt_cold_memory_official_source_failed\n",
        encoding="utf-8",
    )

    outcome_path = record_result_outcome(
        outcomes_root=tmp_path / "outcomes",
        task_name="ExpenseAddSingle",
        method="mobilegpt",
        device="small5554",
        device_serial="emulator-5554",
        attempt_id="iteration_01-test",
        source_seed=111,
        evaluation_seed=113,
        status="prep_failed",
        stage="source_memory",
        task_log=task_log,
        artifact_root=source_attempt,
        outer_wall_sec=14.25,
    )

    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    assert outcome["schema_version"] == "omniflow.androidworld.result_outcome.v2"
    assert outcome["failure_summary"] == (
        "mobilegpt_cold_memory_official_source_failed"
    )
    assert outcome["prompt_tokens"] == 100
    assert outcome["completion_tokens"] == 20
    assert outcome["total_tokens"] == 120
    assert outcome["actions_executed"] == 1
    assert outcome["episode_duration_sec"] == 12.5
    assert outcome["outer_wall_sec"] == 14.25
    assert outcome["official_validator_success"] is None
    assert outcome["retry_count"] == 0


def test_record_outcome_accepts_empty_integer_token_totals(tmp_path: Path) -> None:
    artifact_root = tmp_path / "fixed_replay"
    artifact_root.mkdir()

    outcome_path = record_result_outcome(
        outcomes_root=tmp_path / "outcomes",
        task_name="BrowserMaze",
        method="fixed_replay",
        device="small5554",
        device_serial="emulator-5554",
        attempt_id="iteration_01-test",
        source_seed=111,
        evaluation_seed=113,
        status="completed",
        stage="target_episode",
        artifact_root=artifact_root,
    )

    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    assert outcome["prompt_tokens"] == 0
    assert outcome["completion_tokens"] == 0
    assert outcome["total_tokens"] == 0


def test_record_result_outcome_preserves_published_episode_accounting(
    tmp_path: Path,
) -> None:
    outcome_path = record_result_outcome(
        outcomes_root=tmp_path / "outcomes",
        task_name="CameraTakePhoto",
        method="omniflow",
        device="fold5564",
        device_serial="emulator-5564",
        attempt_id="attempt_001",
        source_seed=111,
        evaluation_seed=113,
        status="method_failed",
        stage="androidworld_validate",
        official_validator_used=True,
        official_validator_success=False,
        model_calls=3,
        prompt_tokens=14637,
        completion_tokens=560,
        total_tokens=15197,
    )

    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    assert outcome["model_calls"] == 3
    assert outcome["prompt_tokens"] == 14637
    assert outcome["completion_tokens"] == 560
    assert outcome["total_tokens"] == 15197


def test_record_result_outcome_preserves_function_metrics_from_result_summary(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifact"
    scheduler_root = artifact_root / "scheduler"
    scheduler_root.mkdir(parents=True)
    (scheduler_root / "result_summary.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "method": "omniflow",
                        "device": "small5554",
                        "function_hit": True,
                        "function_covered_steps": 2,
                        "function_total_steps": 2,
                        "function_step_coverage_rate": 1.0,
                        "vlm_calls": 0,
                        "vlm_latency_ms": 0.0,
                        "latency_sec": 17.8,
                        "energy_mwh": None,
                        "energy_measurement_available": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    outcome_path = record_result_outcome(
        outcomes_root=tmp_path / "outcomes",
        task_name="CameraTakePhoto",
        method="omniflow",
        device="small5554",
        device_serial="emulator-5554",
        attempt_id="attempt-registered-metrics",
        source_seed=111,
        evaluation_seed=113,
        status="completed",
        stage="androidworld_validate",
        artifact_root=artifact_root,
        official_validator_used=True,
        official_validator_success=True,
    )

    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    assert outcome["function_hit"] is True
    assert outcome["function_covered_steps"] == 2
    assert outcome["function_total_steps"] == 2
    assert outcome["function_step_coverage_rate"] == 1.0


def test_autodroid_validator_conclusion_is_not_non_validator_failure(
    tmp_path: Path,
) -> None:
    outcome_path = record_result_outcome(
        outcomes_root=tmp_path / "outcomes",
        task_name="CameraTakePhoto",
        method="autodroid",
        device="autodroid9207",
        device_serial="emulator-5590",
        attempt_id="autodroid-smoke",
        source_seed=111,
        evaluation_seed=113,
        status="method_failed",
        stage="androidworld_validate",
        official_validator_used=True,
        official_validator_success=False,
        official_validator_coverage_rate=1.0,
        actions_executed=7,
        episode_duration_sec=12.5,
    )

    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    assert outcome["official_validator_used"] is True
    assert outcome["official_validator_success"] is False

    report = summarize_results(
        memory_index=tmp_path / "current.json",
        outcomes_root=tmp_path / "outcomes",
        tasks=("CameraTakePhoto",),
        methods=("autodroid",),
        devices=("autodroid9207",),
        source_seed=111,
        evaluation_seed=113,
        attempt_id="autodroid-smoke",
    )

    assert report["counts"] == {
        "planned": 1,
        "validator_success": 0,
        "validator_failure": 1,
        "non_validator_failure": 0,
        "pending": 0,
    }
    assert report["total_tokens"] == 0
    assert report["episode_duration_sec"] == 12.5


def test_autodroid_explicit_metrics_survive_empty_stats_artifact(tmp_path: Path) -> None:
    artifact_root = tmp_path / "autodroid"
    artifact_root.mkdir()

    outcome_path = record_result_outcome(
        outcomes_root=tmp_path / "outcomes",
        task_name="SystemWifiTurnOn",
        method="autodroid",
        device="autodroid9207",
        device_serial="emulator-5590",
        attempt_id="autodroid-v8",
        source_seed=111,
        evaluation_seed=113,
        status="method_failed",
        stage="androidworld_validate",
        artifact_root=artifact_root,
        official_validator_used=True,
        official_validator_success=False,
        official_validator_coverage_rate=1.0,
        actions_executed=20,
        episode_duration_sec=38.907,
    )

    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))

    assert outcome["actions_executed"] == 20
    assert outcome["episode_duration_sec"] == 38.907


def test_summary_reads_registry_when_current_index_is_stale(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    memory_index = data_root / "current.json"
    memory_index.parent.mkdir(parents=True)
    memory_index.write_text(json.dumps({"canonical": {"result_cells": {}}}), encoding="utf-8")
    registered_path = (
        data_root
        / "androidworld"
        / ".archive"
        / "result_registry"
        / "CameraTakePhoto"
        / "omniflow"
        / "small5554"
        / "attempt_004.omniflow.small5554"
        / "registered_result.json"
    )
    registered_path.parent.mkdir(parents=True)
    row = {
        "task_name": "CameraTakePhoto",
        "method": "omniflow",
        "device": "small5554",
        "source_seed": 111,
        "evaluation_seed": 113,
        "status": "completed",
        "official_validator_used": True,
        "official_validator_success": True,
        "model_calls": 2,
        "total_tokens": 100,
        "actions_executed": 7,
    }
    registered_path.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.androidworld_registered_result.v1",
                "task_name": "CameraTakePhoto",
                "source_seed": 111,
                "evaluation_seed": 113,
                "rows": [row],
                "details": [row],
            }
        ),
        encoding="utf-8",
    )

    report = summarize_results(
        memory_index=memory_index,
        outcomes_root=data_root / "androidworld" / ".archive" / "outcomes",
        tasks=("CameraTakePhoto",),
        methods=("omniflow",),
        devices=("small5554",),
        source_seed=111,
        evaluation_seed=113,
        attempt_id="attempt_016",
    )

    assert report["counts"] == {
        "planned": 1,
        "validator_success": 1,
        "validator_failure": 0,
        "non_validator_failure": 0,
        "pending": 0,
    }


def test_concluded_result_keys_skip_immutable_failure_on_resume(tmp_path: Path) -> None:
    record_result_outcome(
        outcomes_root=tmp_path / "outcomes",
        task_name="BrowserDraw",
        method="mobilegpt",
        device="fold5564",
        device_serial="emulator-5564",
        attempt_id="iteration_01-test",
        source_seed=111,
        evaluation_seed=113,
        status="execution_failed",
        stage="target_episode",
    )

    concluded = concluded_result_keys(
        outcomes_root=tmp_path / "outcomes",
        task_name="BrowserDraw",
        methods=("mobilegpt",),
        devices=("small5554", "fold5564"),
        source_seed=111,
        evaluation_seed=113,
    )

    assert concluded == {("mobilegpt", "fold5564")}


def test_concluded_result_keys_maps_historical_label_to_current_device_model(
    tmp_path: Path,
) -> None:
    outcomes_root = tmp_path / "outcomes"
    record_result_outcome(
        outcomes_root=outcomes_root,
        task_name="CameraTakePhoto",
        method="mobilegpt",
        device="small5562",
        device_serial="emulator-5562",
        attempt_id="attempt_001",
        source_seed=111,
        evaluation_seed=113,
        status="method_failed",
        stage="androidworld_validate",
        official_validator_used=True,
        official_validator_success=False,
    )

    assert concluded_result_keys(
        outcomes_root=outcomes_root,
        task_name="CameraTakePhoto",
        methods=("mobilegpt",),
        devices=("standard45562",),
        source_seed=111,
        evaluation_seed=113,
        device_models={"standard45562": "OmniFlowTargetSmall"},
    ) == {("mobilegpt", "standard45562")}


def test_summary_maps_registered_historical_label_to_current_device_model(
    tmp_path: Path,
) -> None:
    memory_index = tmp_path / "data" / "current.json"
    memory_index.parent.mkdir(parents=True)
    memory_index.write_text("{}", encoding="utf-8")
    registered = (
        memory_index.parent
        / "androidworld"
        / ".archive"
        / "result_registry"
        / "CameraTakePhoto"
        / "mobilegpt"
        / "small5562"
        / "attempt_001.mobilegpt.small5562"
        / "registered_result.json"
    )
    registered.parent.mkdir(parents=True)
    registered.write_text(
        json.dumps(
            {
                "task_name": "CameraTakePhoto",
                "source_seed": 111,
                "evaluation_seed": 113,
                "details": [
                    {
                        "task_name": "CameraTakePhoto",
                        "method": "mobilegpt",
                        "device": "small5562",
                        "device_serial": "emulator-5562",
                        "source_seed": 111,
                        "evaluation_seed": 113,
                        "official_validator_used": True,
                        "official_validator_success": False,
                        "attempt_id": "attempt_001.mobilegpt.small5562",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = summarize_results(
        memory_index=memory_index,
        outcomes_root=tmp_path / "outcomes",
        tasks=("CameraTakePhoto",),
        methods=("mobilegpt",),
        devices=("standard45562",),
        source_seed=111,
        evaluation_seed=113,
        attempt_id="attempt_002",
        device_models={"standard45562": "OmniFlowTargetSmall"},
    )

    assert report["counts"]["validator_failure"] == 1
    assert report["counts"]["pending"] == 0


def test_environment_repair_retry_ignores_prior_attempt_outcomes(
    tmp_path: Path,
) -> None:
    outcomes_root = tmp_path / "outcomes"
    record_result_outcome(
        outcomes_root=outcomes_root,
        task_name="BrowserDraw",
        method="fixed_replay",
        device="fold5564",
        device_serial="emulator-5564",
        attempt_id="iteration_01-failed-environment",
        source_seed=111,
        evaluation_seed=113,
        status="execution_failed",
        stage="target_episode",
    )

    assert concluded_result_keys(
        outcomes_root=outcomes_root,
        task_name="BrowserDraw",
        methods=("fixed_replay",),
        devices=("fold5564",),
        source_seed=111,
        evaluation_seed=113,
        attempt_id="iteration_02-environment-repair",
    ) == set()

    record_result_outcome(
        outcomes_root=outcomes_root,
        task_name="BrowserDraw",
        method="fixed_replay",
        device="fold5564",
        device_serial="emulator-5564",
        attempt_id="iteration_02-environment-repair",
        source_seed=111,
        evaluation_seed=113,
        status="execution_failed",
        stage="target_episode",
    )

    assert concluded_result_keys(
        outcomes_root=outcomes_root,
        task_name="BrowserDraw",
        methods=("fixed_replay",),
        devices=("fold5564",),
        source_seed=111,
        evaluation_seed=113,
        attempt_id="iteration_02-environment-repair",
    ) == {("fixed_replay", "fold5564")}


def test_method_failed_without_validator_or_action_stays_retryable(
    tmp_path: Path,
) -> None:
    record_result_outcome(
        outcomes_root=tmp_path / "outcomes",
        task_name="BrowserMaze",
        method="mobilegpt",
        device="small5562",
        device_serial="emulator-5562",
        attempt_id="attempt_setup_abort",
        source_seed=111,
        evaluation_seed=113,
        status="method_failed",
        stage="androidworld_validate",
    )

    assert concluded_result_keys(
        outcomes_root=tmp_path / "outcomes",
        task_name="BrowserMaze",
        methods=("mobilegpt",),
        devices=("small5562",),
        source_seed=111,
        evaluation_seed=113,
    ) == set()


def test_batch_report_merges_validator_and_failure_outcomes(tmp_path: Path) -> None:
    source_index = tmp_path / "source_index.json"
    source_index.write_text(
        json.dumps({"BrowserDraw": {"task": "BrowserDraw", "source_seed": 111}}),
        encoding="utf-8",
    )
    registered_result = tmp_path / "registered_result.json"
    registered_result.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.androidworld_registered_result.v1",
                "attempt_id": "iteration_01-success",
                "rows": [
                    {
                        "task_name": "BrowserDraw",
                        "method": "mobilegpt",
                        "device": "small5554",
                        "official_validator_used": True,
                        "official_validator_success": True,
                        "official_validator_coverage_rate": 1.0,
                        "episode_model_calls": 4,
                        "episode_prompt_tokens": 80,
                        "episode_completion_tokens": 20,
                        "episode_total_tokens": 100,
                        "episode_actions_executed": 3,
                        "duration_sec": 9.5,
                        "wall_sec": 11.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result_cells = tmp_path / "result_cells.json"
    result_cells.write_text(
        json.dumps(
            {
                        "BrowserDraw|mobilegpt|small5554|111|113": {
                    "registered_result_object_path": str(registered_result),
                    "official_validator_success": True,
                }
            }
        ),
        encoding="utf-8",
    )
    memory_index = tmp_path / "current.json"
    memory_index.write_text(
        json.dumps({"canonical": {"result_cells": json.loads(result_cells.read_text())}}),
        encoding="utf-8",
    )
    record_result_outcome(
        outcomes_root=tmp_path / "outcomes",
        task_name="BrowserDraw",
        method="mobilegpt",
        device="fold5564",
        device_serial="emulator-5564",
        attempt_id="iteration_01-report",
        source_seed=111,
        evaluation_seed=113,
        status="execution_failed",
        stage="target_episode",
        outer_wall_sec=12.0,
    )

    report = summarize_results(
        memory_index=memory_index,
        outcomes_root=tmp_path / "outcomes",
        tasks=("BrowserDraw",),
        methods=("mobilegpt",),
        devices=("small5554", "fold5564"),
        source_seed=111,
        evaluation_seed=113,
        attempt_id="iteration_01-report",
    )

    assert report["counts"] == {
        "planned": 2,
        "validator_success": 1,
        "validator_failure": 0,
        "non_validator_failure": 1,
        "pending": 0,
    }
    assert report["model_calls"] == 4
    assert report["total_tokens"] == 100
    assert report["episode_duration_sec"] == 9.5
    assert report["outer_wall_sec"] == 23.0


def test_run_summary_uses_canonical_usage_fields(tmp_path: Path) -> None:
    task_results = tmp_path / "task_results.jsonl"
    task_results.write_text(
        json.dumps(
            {
                "task_name": "BrowserDraw",
                "official_validator_used": True,
                "official_validator_success": True,
                "model_calls": 3,
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "total_tokens": 150,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = _summarize_task_results(
        task_results_path=task_results,
        checkpoint_dir="checkpoint",
        agent="omniflow",
        tasks=("BrowserDraw",),
    )

    assert summary["tool_calls"] == 3
    assert summary["tokens"] == 150
    for detailed_field in (
        "model_calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ):
        assert detailed_field not in summary
    assert summary["per_task"][0]["prompt_tokens"] == 120
    assert summary["per_task"][0]["completion_tokens"] == 30


def test_batch_report_uses_current_attempt_failure_outcome(tmp_path: Path) -> None:
    source_index = tmp_path / "source_index.json"
    source_index.write_text(
        json.dumps({"BrowserDraw": {"task": "BrowserDraw", "source_seed": 111}}),
        encoding="utf-8",
    )
    result_cells = tmp_path / "result_cells.json"
    result_cells.write_text("{}", encoding="utf-8")
    memory_index = tmp_path / "current.json"
    memory_index.write_text(
        json.dumps({"canonical": {"result_cells": {}}}),
        encoding="utf-8",
    )
    outcomes_root = tmp_path / "outcomes"
    for attempt_id, outer_wall_sec in (
        ("iteration_01-environment-failure", 12.0),
        ("iteration_02-environment-repair", 34.0),
    ):
        record_result_outcome(
            outcomes_root=outcomes_root,
            task_name="BrowserDraw",
            method="fixed_replay",
            device="fold5564",
            device_serial="emulator-5564",
            attempt_id=attempt_id,
            source_seed=111,
            evaluation_seed=113,
            status="execution_failed",
            stage="target_episode",
            outer_wall_sec=outer_wall_sec,
        )

    report = summarize_results(
        memory_index=memory_index,
        outcomes_root=outcomes_root,
        tasks=("BrowserDraw",),
        methods=("fixed_replay",),
        devices=("fold5564",),
        source_seed=111,
        evaluation_seed=113,
        attempt_id="iteration_02-environment-repair",
    )

    assert report["outer_wall_sec"] == 34.0


def test_batch_report_current_attempt_failure_overrides_registered_result(
    tmp_path: Path,
) -> None:
    source_index = tmp_path / "source_index.json"
    source_index.write_text(
        json.dumps({"BrowserDraw": {"task": "BrowserDraw", "source_seed": 111}}),
        encoding="utf-8",
    )
    registered_result = tmp_path / "registered_result.json"
    registered_result.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.androidworld_registered_result.v1",
                "attempt_id": "iteration_01-success",
                "rows": [
                    {
                        "task_name": "BrowserDraw",
                        "method": "mobilegpt",
                        "device": "small5554",
                        "official_validator_used": True,
                        "official_validator_success": True,
                        "official_validator_coverage_rate": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result_cells = tmp_path / "result_cells.json"
    result_cells.write_text(
        json.dumps(
            {
                "BrowserDraw|mobilegpt|small5554|111|113": {
                    "registered_result_object_path": str(registered_result),
                    "official_validator_success": True,
                }
            }
        ),
        encoding="utf-8",
    )
    memory_index = tmp_path / "current.json"
    memory_index.write_text(
        json.dumps({"canonical": {"result_cells": json.loads(result_cells.read_text())}}),
        encoding="utf-8",
    )
    record_result_outcome(
        outcomes_root=tmp_path / "outcomes",
        task_name="BrowserDraw",
        method="mobilegpt",
        device="small5554",
        device_serial="emulator-5554",
        attempt_id="iteration_02-source-failure",
        source_seed=111,
        evaluation_seed=113,
        status="prep_failed",
        stage="source_memory",
        outer_wall_sec=4.0,
    )

    report = summarize_results(
        memory_index=memory_index,
        outcomes_root=tmp_path / "outcomes",
        tasks=("BrowserDraw",),
        methods=("mobilegpt",),
        devices=("small5554",),
        source_seed=111,
        evaluation_seed=113,
        attempt_id="iteration_02-source-failure",
    )

    assert report["counts"] == {
        "planned": 1,
        "validator_success": 0,
        "validator_failure": 0,
        "non_validator_failure": 1,
        "pending": 0,
    }
    assert report["outer_wall_sec"] == 4.0


def test_batch_report_recovers_runlog_teacher_source_failure_accounting(
    tmp_path: Path,
) -> None:
    source_index = tmp_path / "source_index.json"
    source_index.write_text(
        json.dumps({"BrowserDraw": {"task": "BrowserDraw", "source_seed": 111}}),
        encoding="utf-8",
    )
    result_cells = tmp_path / "result_cells.json"
    result_cells.write_text("{}", encoding="utf-8")
    memory_index = tmp_path / "current.json"
    memory_index.write_text(
        json.dumps({"canonical": {"result_cells": {}}}),
        encoding="utf-8",
    )
    source_attempt = tmp_path / "source_attempt"
    source_attempt.mkdir()
    stats = source_attempt / "source_stats.jsonl"
    stats.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "event": "chat_call",
                        "prompt_tokens": 90,
                        "completion_tokens": 10,
                        "total_tokens": 100,
                    }
                ),
                json.dumps({"event": "embedding_call", "prompt_tokens": 8}),
                json.dumps({"event": "mobilegpt_action_sent"}),
                json.dumps({"event": "task_finished", "elapsed_sec": 7.5}),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (source_attempt / "prep_failure.json").write_text(
        json.dumps(
            {
                "error": "mobilegpt_cold_memory_not_task_local",
                "retry_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    (source_attempt / "source_episode_command.json").write_text(
        json.dumps(
                {
                    "task_name": "BrowserDraw",
                    "source_method": MOBILEGPT_SOURCE_METHOD,
                }
        ),
        encoding="utf-8",
    )
    task_log = tmp_path / "task.log"
    task_log.write_text(
        f"env MOBILEGPT_STATS_JSONL={stats} python -m source\n"
        "ValueError: mobilegpt_cold_memory_not_task_local\n",
        encoding="utf-8",
    )
    outcome_path = record_result_outcome(
        outcomes_root=tmp_path / "outcomes",
        task_name="BrowserDraw",
        method="mobilegpt",
        device="small5554",
        device_serial="emulator-5554",
        attempt_id="iteration_01-report",
        source_seed=111,
        evaluation_seed=113,
        status="execution_failed",
        stage="target_episode",
        task_log=task_log,
        artifact_root=tmp_path / "unrelated_target_attempt",
    )
    legacy_outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    legacy_outcome.update(
        {
            "model_calls": 2,
            "prompt_tokens": 98,
            "completion_tokens": 10,
            "total_tokens": 100,
            "actions_executed": 1,
            "episode_duration_sec": 7.5,
        }
    )
    outcome_path.write_text(json.dumps(legacy_outcome), encoding="utf-8")

    report = summarize_results(
        memory_index=memory_index,
        outcomes_root=tmp_path / "outcomes",
        tasks=("BrowserDraw",),
        methods=("mobilegpt",),
        devices=("small5554",),
        source_seed=111,
        evaluation_seed=113,
        attempt_id="iteration_01-report",
    )

    assert report["model_calls"] == 2
    assert report["total_tokens"] == 108
    assert report["episode_duration_sec"] == 7.5
