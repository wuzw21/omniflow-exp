from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
from runlog_fixtures import androidworld_run_log, androidworld_state

from src.experiment.run_task import (
    aggregate_task_results,
    write_metrics_summary,
)
from src.experiment.observation_evidence import (
    AndroidWorldEpisodeRecorder,
    androidworld_json_action_dict,
    persist_target_run_evidence,
)
from src.experiment.result_schema import RESULT_FIELDS, compact_result_row


def test_public_result_row_is_compact_and_keeps_details_out_of_the_row() -> None:
    row = compact_result_row(
        {
            "task_name": "Task",
            "method": "omniflow",
            "device": "small5554",
            "official_validator_used": True,
            "official_validator_success": True,
            "model_calls": 2,
            "prompt_tokens": 10,
            "completion_tokens": 3,
            "total_tokens": 13,
            "actions_executed": 4,
            "duration_ms": 2500,
            "run_dir": "/evidence/task",
            "reuse_rate": 1.0,
            "prep_model_calls": 3,
        },
        source_seed=111,
        evaluation_seed=113,
    )

    assert tuple(row) == RESULT_FIELDS
    assert row["episode_duration_sec"] == 2.5
    assert row["evidence_paths"] == ["/evidence/task"]
    assert "reuse_rate" not in row
    assert "prep_model_calls" not in row


def test_compact_result_row_is_idempotent_for_evidence_paths() -> None:
    original = compact_result_row(
        {
            "task": "TaskOne",
            "method": "omniflow",
            "device": "small5554",
            "validator_success": True,
            "evidence_paths": ["/evidence/result", "/evidence/details"],
        },
        source_seed=111,
        evaluation_seed=113,
    )

    assert compact_result_row(
        original,
        source_seed=111,
        evaluation_seed=113,
    ) == original


def test_episode_recorder_preserves_every_observation_with_sequential_images(
    tmp_path,
) -> None:
    states = [
        SimpleNamespace(
            pixels=Image.new("RGB", (4, 3), color="red"),
            forest="<hierarchy />",
            ui_elements=[],
            auxiliaries={
                "package_name": "com.android.settings",
                "activity_name": "com.android.settings/.Settings",
            },
        ),
        SimpleNamespace(
            pixels=Image.new("RGB", (4, 3), color="red"),
            forest="<hierarchy />",
            ui_elements=[],
            auxiliaries={
                "package_name": "com.android.settings",
                "activity_name": "com.android.settings/.Settings",
            },
        ),
    ]
    remaining = iter(states)
    recorder = AndroidWorldEpisodeRecorder(
        lambda: next(remaining),
        lambda action: action,
        evidence_root=tmp_path,
    )
    recorder.start_episode()

    assert recorder.get_state() is states[0]
    assert recorder.get_state() is states[1]
    records = recorder.persist_observations()

    assert [record["observation_index"] for record in records] == [0, 1]
    assert records[0]["path"] == "screenshots/screenshot_000001.png"
    assert records[1]["path"] == "screenshots/screenshot_000002.png"
    assert (tmp_path / records[0]["path"]).read_bytes() == (
        tmp_path / records[1]["path"]
    ).read_bytes()
    assert records[0]["display"] == {"width": 4, "height": 3}
    assert set(records[0]) == {
        "observation_index",
        "state_id",
        "display",
        "path",
    }
    assert len(list((tmp_path / "screenshots").glob("*.png"))) == 2
    assert not (tmp_path / "observations").exists()


def test_episode_recorder_accepts_an_observation_without_pixels(tmp_path) -> None:
    state = SimpleNamespace(
        pixels=None,
        forest="<hierarchy />",
        ui_elements=[],
        auxiliaries={
            "package_name": "com.android.settings",
            "activity_name": "com.android.settings/.Settings",
        },
    )
    recorder = AndroidWorldEpisodeRecorder(
        lambda: state,
        lambda action: action,
        evidence_root=tmp_path,
    )
    recorder.start_episode()

    assert recorder.get_state() is state

    records = recorder.persist_observations()
    assert records == [
        {
            "observation_index": 0,
            "state_id": records[0]["state_id"],
        }
    ]


def test_episode_recorder_refreshes_one_empty_accessibility_snapshot(tmp_path) -> None:
    empty = SimpleNamespace(
        pixels=Image.new("RGB", (4, 3), color="black"),
        forest="",
        ui_elements=[],
        auxiliaries={},
    )
    ready = SimpleNamespace(
        pixels=Image.new("RGB", (4, 3), color="white"),
        forest="<hierarchy />",
        ui_elements=[],
        auxiliaries={},
    )
    states = iter([empty, ready])
    recorder = AndroidWorldEpisodeRecorder(
        lambda: next(states),
        lambda action: action,
        evidence_root=tmp_path,
    )
    recorder.start_episode()

    assert recorder.get_state() is ready
    assert len(recorder.persist_observations()) == 1


def test_episode_recorder_records_action_and_official_validator_result(tmp_path) -> None:
    states = iter(
        [
            SimpleNamespace(
                pixels=Image.new("RGB", (4, 3), color="red"),
                forest="before",
                ui_elements=[],
                auxiliaries={},
            ),
            SimpleNamespace(
                pixels=Image.new("RGB", (4, 3), color="blue"),
                forest="after",
                ui_elements=[],
                auxiliaries={},
            ),
        ]
    )
    executed = []
    recorder = AndroidWorldEpisodeRecorder(
        lambda: next(states),
        lambda action: executed.append(action),
        evidence_root=tmp_path,
    )
    recorder.start_episode()
    recorder.execute_action(SimpleNamespace(action_type="click", x=1, y=2))
    run_log = recorder.seal_run_log(
        task_name="OpenSettings",
        goal="Open Settings.",
        task_parameters={},
        seed=111,
        validator_success=False,
        validator_reward=0.0,
    )

    assert len(executed) == 1
    assert run_log is not None
    assert run_log["status"] == "failed"
    assert run_log["steps"][0]["action"] == {
        "action_type": "click",
        "x": 1,
        "y": 2,
    }
    assert run_log["steps"][0]["observation"]["xml"] == "before"
    assert run_log["steps"][0]["next_observation"]["xml"] == "after"


def test_episode_recorder_records_an_action_exception(tmp_path) -> None:
    state = SimpleNamespace(
        pixels=Image.new("RGB", (4, 3), color="red"),
        forest="before",
        ui_elements=[],
        auxiliaries={},
    )

    def fail(_action):
        raise RuntimeError("device rejected action")

    recorder = AndroidWorldEpisodeRecorder(
        lambda: state,
        fail,
        evidence_root=tmp_path,
    )
    recorder.start_episode()

    try:
        recorder.execute_action(SimpleNamespace(action_type="wait"))
    except RuntimeError as error:
        assert str(error) == "device rejected action"
    else:
        raise AssertionError("action exception must propagate")

    run_log = recorder.seal_run_log(
        task_name="Task",
        goal="Goal",
        task_parameters={},
        seed=111,
        validator_official=False,
        validator_success=False,
        validator_reward=0.0,
    )
    assert run_log is not None
    assert run_log["validator"]["official"] is False
    assert run_log["steps"][0]["result"] == {
        "success": False,
        "error": "device rejected action",
    }


def test_episode_recorder_deduplicates_nested_host_and_env_actions(tmp_path) -> None:
    states = iter(
        [
            SimpleNamespace(
                pixels=Image.new("RGB", (4, 3), color="red"),
                forest="before",
                ui_elements=[],
                auxiliaries={},
            ),
            SimpleNamespace(
                pixels=Image.new("RGB", (4, 3), color="blue"),
                forest="after",
                ui_elements=[],
                auxiliaries={},
            ),
        ]
    )
    recorder = AndroidWorldEpisodeRecorder(
        lambda: next(states),
        lambda action: action,
        evidence_root=tmp_path,
    )
    recorder.start_episode()
    host_action = SimpleNamespace(action_type="click", x=1, y=2)

    recorder.execute_host_action(
        host_action,
        execute=lambda: recorder.execute_action(host_action),
        project=lambda action: action,
    )

    run_log = recorder.seal_run_log(
        task_name="Task",
        goal="Goal",
        task_parameters={},
        seed=111,
        validator_success=True,
        validator_reward=1.0,
    )
    assert run_log is not None
    assert len(run_log["steps"]) == 1


def test_episode_recorder_preserves_host_input_text_coordinates(tmp_path) -> None:
    states = iter(
        [
            SimpleNamespace(
                pixels=Image.new("RGB", (720, 1280), color="red"),
                forest="before",
                ui_elements=[],
                auxiliaries={},
            ),
            SimpleNamespace(
                pixels=Image.new("RGB", (720, 1280), color="blue"),
                forest="after",
                ui_elements=[],
                auxiliaries={},
            ),
        ]
    )
    recorder = AndroidWorldEpisodeRecorder(
        lambda: next(states),
        lambda action: action,
        evidence_root=tmp_path,
    )
    recorder.start_episode()
    projected = SimpleNamespace(
        action_type="input_text",
        x=360.0,
        y=761.5,
        text="I may repeat this",
        clear_text=True,
    )

    recorder.execute_host_action(
        {
            "tool": "input_text",
            "args": {
                "text": "I may repeat this",
                "x": 500.0,
                "y": 594.921875,
            },
        },
        execute=lambda: SimpleNamespace(success=True),
        project=lambda _action: projected,
    )

    run_log = recorder.seal_run_log(
        task_name="Task",
        goal="Goal",
        task_parameters={},
        seed=111,
        validator_success=True,
        validator_reward=1.0,
    )
    assert run_log is not None
    assert run_log["steps"][0]["action"] == {
        "action_type": "input_text",
        "x": 360.0,
        "y": 761.5,
        "text": "I may repeat this",
        "clear_text": True,
    }


def test_episode_recorder_projects_host_swipe_to_androidworld_direction(tmp_path) -> None:
    states = iter(
        [
            SimpleNamespace(
                pixels=Image.new("RGB", (4, 3), color="red"),
                forest="before",
                ui_elements=[],
                auxiliaries={},
            ),
            SimpleNamespace(
                pixels=Image.new("RGB", (4, 3), color="blue"),
                forest="after",
                ui_elements=[],
                auxiliaries={},
            ),
        ]
    )
    recorder = AndroidWorldEpisodeRecorder(
        lambda: next(states),
        lambda action: action,
        evidence_root=tmp_path,
    )
    recorder.start_episode()

    recorder.execute_host_action(
        {"tool": "swipe", "args": {"x1": 500, "y1": 800, "x2": 500, "y2": 200}},
        execute=lambda: SimpleNamespace(success=True),
        project=lambda _action: SimpleNamespace(
            action_type="scroll",
            direction="up",
        ),
    )

    run_log = recorder.seal_run_log(
        task_name="Task",
        goal="Goal",
        task_parameters={},
        seed=111,
        validator_success=True,
        validator_reward=1.0,
    )
    assert run_log is not None
    assert run_log["steps"][0]["action"] == {
        "action_type": "scroll",
        "direction": "up",
    }


def test_recorder_accepts_official_press_keyboard_action() -> None:
    assert androidworld_json_action_dict(
        SimpleNamespace(
            action_type="press_keyboard",
            keycode="KEYCODE_DPAD_DOWN",
        )
    ) == {
        "action_type": "press_keyboard",
        "keycode": "KEYCODE_DPAD_DOWN",
    }


def test_episode_recorder_returns_no_run_log_before_episode(tmp_path) -> None:
    recorder = AndroidWorldEpisodeRecorder(
        lambda: None,
        lambda action: action,
        evidence_root=tmp_path,
    )

    assert recorder.seal_run_log(
        task_name="Task",
        goal="Goal",
        task_parameters={},
        seed=111,
        validator_success=False,
        validator_reward=0.0,
    ) is None


def test_target_run_evidence_is_immutable_and_hash_addressable(tmp_path) -> None:
    run_log = androidworld_run_log(
        [{"action_type": "open_app", "app_name": "com.android.settings"}],
        observations=[androidworld_state("target-before")],
        task_name="OpenSettings",
        run_id="target-run",
        goal="Open Settings.",
    )
    run_log["steps"][0]["next_observation"] = androidworld_state("target-after")
    states = {
        "target-before": {
            "state_id": "target-before",
            "xml": "<hierarchy />",
        }
    }
    audit = {
        "referenced_state_ids": ["target-after", "target-before"],
        "captured_state_ids": ["target-before"],
        "missing_state_ids": ["target-after"],
        "referenced_state_count": 2,
        "captured_state_count": 1,
        "missing_state_count": 1,
        "complete": False,
    }

    first = persist_target_run_evidence(
        tmp_path,
        run_log=run_log,
        captured_transfer_states=states,
        transfer_state_audit=audit,
    )
    second = persist_target_run_evidence(
        tmp_path,
        run_log=run_log,
        captured_transfer_states=states,
        transfer_state_audit=audit,
    )

    assert first == second
    assert Path(first["run_log_path"]).name == "run_log.json"
    for path_key, sha_key in (
        ("target_transfer_states_path", "target_transfer_states_sha256"),
    ):
        path = Path(first[path_key])
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == first[sha_key]
    assert first["target_transfer_state_audit"]["missing_state_ids"] == [
        "target-after"
    ]


def test_baseline_target_run_evidence_does_not_require_transfer_states(tmp_path) -> None:
    run_log = androidworld_run_log(
        [{"action_type": "wait"}],
        task_name="Task",
        run_id="baseline-run",
    )

    evidence = persist_target_run_evidence(tmp_path, run_log=run_log)

    assert evidence == {"run_log_path": str(tmp_path / "run_log.json")}


def test_target_evidence_provenance_survives_metrics_aggregation(tmp_path) -> None:
    result_path = tmp_path / "Task" / "omniflow" / "small5554" / "task_results.jsonl"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "task_name": "Task",
                "official_validator_used": True,
                "success": True,
                "run_log_path": "/evidence/run_log.json",
                "target_transfer_states_path": "/evidence/target.transfer_states.json",
                "target_transfer_states_sha256": "states-sha",
                "target_transfer_state_audit": {"complete": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    row = aggregate_task_results([result_path])["per_task"][0]

    assert row["run_log_path"] == "/evidence/run_log.json"
    assert row["target_transfer_states_sha256"] == "states-sha"
    assert row["target_transfer_state_audit"] == {"complete": True}


def test_metrics_preserve_autodroid_replay_action_count(tmp_path) -> None:
    result_path = (
        tmp_path
        / "CameraTakePhoto"
        / "autodroid"
        / "autodroid9207"
        / "task_results.jsonl"
    )
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "task_name": "CameraTakePhoto",
                "method": "autodroid",
                "device": "autodroid9207",
                "agent": "autodroid_official_replay",
                "official_validator_used": True,
                "official_validator_success": False,
                "actions_executed": 20,
                "step_count": 20,
                "model_calls": 0,
                "fallback_steps": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = aggregate_task_results([result_path])

    assert summary["actions_executed"] == 20
    assert summary["per_task"][0]["actions_executed"] == 20
    assert summary["model_calls"] == 0
    assert summary["fallback_steps"] == 0


def test_metrics_preserve_autodroid_replay_completion_and_duration(tmp_path) -> None:
    result_path = (
        tmp_path
        / "SystemWifiTurnOn"
        / "autodroid"
        / "autodroid9207"
        / "task_results.jsonl"
    )
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "task_name": "SystemWifiTurnOn",
                "method": "autodroid",
                "device": "autodroid9207",
                "official_validator_used": True,
                "official_validator_success": False,
                "actions_executed": 20,
                "duration_ms": 38907.396,
                "replay_completed": True,
                "replay_step_completed_count": 20,
                "replay_step_total": 20,
                "model_calls": 0,
                "fallback_steps": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = aggregate_task_results([result_path])
    row = summary["per_task"][0]

    assert summary["replay_task_count"] == 1
    assert summary["replay_completed_count"] == 1
    assert summary["replay_step_completed_count"] == 20
    assert summary["replay_step_total"] == 20
    assert row["replay_completed"] is True
    assert row["duration_sec"] == 38.907


def test_metrics_preserve_missing_validator_as_unknown(tmp_path) -> None:
    result_path = tmp_path / "Task" / "fixed_replay" / "fold5564" / "task_results.jsonl"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "task_name": "Task",
                "official_validator_used": False,
                "success": False,
                "error": "FileNotFoundError: app database missing",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = aggregate_task_results([result_path])

    assert summary["official_validator_task_count"] == 0
    assert summary["official_validator_coverage_rate"] == 0.0
    assert summary["per_task"][0]["official_validator_success"] is None
    assert summary["per_task"][0]["official_validator_success"] is None


def test_metrics_preserve_runtime_environment_failure_markers(tmp_path) -> None:
    result_path = (
        tmp_path
        / "ContactsNewContactDraft"
        / "mobilegpt"
        / "small5554"
        / "task_results.jsonl"
    )
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "task_name": "ContactsNewContactDraft",
                "official_validator_used": True,
                "official_validator_success": None,
                "runtime_integrity_error": "mobilegpt_app_ui_not_ready",
                "environment_failure": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    row = aggregate_task_results([result_path])["per_task"][0]

    assert row["runtime_integrity_error"] == "mobilegpt_app_ui_not_ready"
    assert row["environment_failure"] is True


def test_metrics_report_only_aggregate_model_calls_and_total_tokens(tmp_path: Path) -> None:
    result_path = tmp_path / "Task" / "omniflow" / "small5554" / "task_results.jsonl"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "task_name": "Task",
                "official_validator_used": True,
                "official_validator_success": True,
                "model_calls": 2,
                "prompt_tokens": 90,
                "completion_tokens": 10,
                "total_tokens": 100,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = aggregate_task_results([result_path])

    assert summary["model_calls"] == 2
    assert summary["total_tokens"] == 100
    assert "tool_calls" not in summary
    assert "tokens" not in summary
    assert summary["per_task"][0]["prompt_tokens"] == 90
    assert summary["per_task"][0]["completion_tokens"] == 10

    output = tmp_path / "metrics.json"
    write_metrics_summary(summary, output)
    markdown = output.with_suffix(".md").read_text(encoding="utf-8")
    assert "- model_calls: `2`" in markdown
    assert "- total_tokens: `100`" in markdown
    assert "prompt_tokens" not in markdown
    assert "completion_tokens" not in markdown
