from __future__ import annotations

import json
from pathlib import Path

from src.experiment.batch_outcomes import (
    concluded_cell_keys,
    record_cell_outcome,
    write_batch_report,
)


def test_record_prep_failure_preserves_reason_tokens_and_time(tmp_path: Path) -> None:
    source_attempt = tmp_path / "source_attempt"
    source_attempt.mkdir()
    (source_attempt / "prep_failure.json").write_text(
        json.dumps(
            {
                "schema_version": "omniflow.mobilegpt-source-failure.v1",
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

    outcome_path = record_cell_outcome(
        outcomes_root=tmp_path / "outcomes",
        task_name="ExpenseAddSingle",
        method="mobilegpt_offline_retrieval",
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
    assert outcome["schema_version"] == "omniflow.androidworld.cell_outcome.v1"
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

    outcome_path = record_cell_outcome(
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


def test_concluded_cell_keys_skip_immutable_failure_on_resume(tmp_path: Path) -> None:
    record_cell_outcome(
        outcomes_root=tmp_path / "outcomes",
        task_name="BrowserDraw",
        method="mobilegpt_offline_retrieval",
        device="fold5564",
        device_serial="emulator-5564",
        attempt_id="iteration_01-test",
        source_seed=111,
        evaluation_seed=113,
        status="execution_failed",
        stage="target_episode",
    )

    concluded = concluded_cell_keys(
        outcomes_root=tmp_path / "outcomes",
        task_name="BrowserDraw",
        methods=("mobilegpt_offline_retrieval",),
        devices=("small5554", "fold5564"),
        source_seed=111,
        evaluation_seed=113,
    )

    assert concluded == {("mobilegpt_offline_retrieval", "fold5564")}


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
                        "method": "mobilegpt_offline_retrieval",
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
                "BrowserDraw|mobilegpt_offline_retrieval|small5554|111|113": {
                    "registered_result_object_path": str(registered_result),
                    "official_validator_success": True,
                }
            }
        ),
        encoding="utf-8",
    )
    memory_index = tmp_path / "current.json"
    memory_index.write_text(
        json.dumps({"result_cells": str(result_cells)}),
        encoding="utf-8",
    )
    record_cell_outcome(
        outcomes_root=tmp_path / "outcomes",
        task_name="BrowserDraw",
        method="mobilegpt_offline_retrieval",
        device="fold5564",
        device_serial="emulator-5564",
        attempt_id="iteration_01-failure",
        source_seed=111,
        evaluation_seed=113,
        status="execution_failed",
        stage="target_episode",
        outer_wall_sec=12.0,
    )

    report = write_batch_report(
        report_root=tmp_path / "report",
        memory_index=memory_index,
        outcomes_root=tmp_path / "outcomes",
        source_index=source_index,
        tasks=("BrowserDraw",),
        methods=("mobilegpt_offline_retrieval",),
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
    rows = [
        json.loads(line)
        for line in Path(report["cells_jsonl"]).read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["total_tokens"] == 100
    assert rows[0]["episode_duration_sec"] == 9.5
    assert rows[1]["failure_summary"] == (
        "cell_finished_without_registered_validator_result"
    )
    assert rows[1]["outer_wall_sec"] == 12.0
