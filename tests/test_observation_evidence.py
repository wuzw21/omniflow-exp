from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
from runlog_fixtures import androidworld_run_log, androidworld_state

from src.experiment.androidworld import aggregate_task_results, write_metrics_summary
from src.experiment.observation_evidence import (
    AndroidWorldEpisodeRecorder,
    androidworld_json_action_dict,
    persist_target_run_evidence,
)


def test_episode_recorder_preserves_every_observation_and_deduplicates_images(
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
    assert records[0]["path"] == records[1]["path"]
    assert records[0]["sha256"] == records[1]["sha256"]
    assert records[0]["display"] == {"width": 4, "height": 3}
    assert records[0]["package_name"] == "com.android.settings"
    assert len(list((tmp_path / "observations" / "objects").glob("*.png"))) == 1
    assert json.loads((tmp_path / "observations" / "index.json").read_text()) == {
        "schema_version": "omniflow.androidworld-observations.v1",
        "observation_count": 2,
        "observations": records,
    }


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
            "package_name": "com.android.settings",
            "activity_name": "com.android.settings/.Settings",
        }
    ]


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
    assert run_log["steps"][0]["observation"]["forest"] == "before"
    assert run_log["steps"][0]["next_observation"]["forest"] == "after"


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
    for path_key, sha_key in (
        ("target_run_log_path", "target_run_log_sha256"),
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

    assert set(evidence) == {
        "target_run_log_path",
        "target_run_log_sha256",
    }


def test_target_evidence_provenance_survives_metrics_aggregation(tmp_path) -> None:
    result_path = tmp_path / "Task" / "ours" / "small5554" / "task_results.jsonl"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "task_name": "Task",
                "official_validator_used": True,
                "success": True,
                "target_run_log_path": "/evidence/target.run_log.json",
                "target_run_log_sha256": "run-sha",
                "target_transfer_states_path": "/evidence/target.transfer_states.json",
                "target_transfer_states_sha256": "states-sha",
                "target_transfer_state_audit": {"complete": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    row = aggregate_task_results([result_path])["per_task"][0]

    assert row["target_run_log_sha256"] == "run-sha"
    assert row["target_transfer_states_sha256"] == "states-sha"
    assert row["target_transfer_state_audit"] == {"complete": True}


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
    assert summary["per_task"][0]["success"] is None


def test_metrics_report_only_aggregate_tool_calls_and_tokens(tmp_path: Path) -> None:
    result_path = tmp_path / "Task" / "ours" / "small5554" / "task_results.jsonl"
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

    assert summary["tool_calls"] == 2
    assert summary["tokens"] == 100
    for detailed_field in (
        "model_calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ):
        assert detailed_field not in summary
    assert summary["per_task"][0]["prompt_tokens"] == 90
    assert summary["per_task"][0]["completion_tokens"] == 10

    output = tmp_path / "metrics.json"
    write_metrics_summary(summary, output)
    markdown = output.with_suffix(".md").read_text(encoding="utf-8")
    assert "- tool_calls: `2`" in markdown
    assert "- tokens: `100`" in markdown
    assert "prompt_tokens" not in markdown
    assert "completion_tokens" not in markdown
