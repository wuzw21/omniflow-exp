from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType
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
import src.experiment.observation_evidence as observation_evidence
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
    assert row["prep_model_calls"] == 3
    assert row["model_calls_total"] == 5
    assert row["total_tokens_including_prep"] == 13
    assert row["memory_status"] == "unavailable"
    assert row["fallback_steps"] is None
    assert row["fallback_measurement_status"] == "unavailable"


def test_public_result_row_distinguishes_measured_zero_from_unreported_external_metrics() -> None:
    row = compact_result_row(
        {
            "task_name": "CameraTakePhoto",
            "method": "mobilegpt",
            "device": "small5562",
            "official_validator_used": True,
            "official_validator_success": True,
            "model_calls": 13,
            "prompt_tokens": 9000,
            "completion_tokens": 2958,
            "total_tokens": 11958,
            "actions_executed": 1,
        },
        source_seed=111,
        evaluation_seed=113,
    )

    assert row["model_calls"] == 13
    assert row["fallback_steps"] is None
    assert row["fallback_measurement_status"] == "not_exposed"
    assert row["memory_status"] == "unavailable"
    assert row["memory_hit"] is None

    omniflow = compact_result_row(
        {
            "task_name": "CameraTakePhoto",
            "method": "omniflow",
            "device": "small5562",
            "official_validator_used": True,
            "official_validator_success": True,
            "model_calls": 0,
            "fallback_steps": 0,
            "max_fallback_steps": 5,
            "reuse_metrics": {
                "artifact_used": True,
                "reuse_numerator": 4,
                "reuse_denominator": 4,
            },
        },
        source_seed=111,
        evaluation_seed=113,
    )
    assert omniflow["fallback_steps"] == 0
    assert omniflow["fallback_measurement_status"] == "measured"
    assert omniflow["memory_status"] == "used"
    assert omniflow["memory_hit"] is True
    assert omniflow["memory_hit_rate"] == 1.0


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


def test_compact_result_row_preserves_normalized_function_metrics() -> None:
    original = compact_result_row(
        {
            "task": "TaskOne",
            "method": "omniflow",
            "device": "small5554",
            "validator_success": True,
            "function_hit": True,
            "function_covered_steps": 2,
            "function_total_steps": 2,
            "function_step_coverage_rate": 1.0,
        },
        source_seed=111,
        evaluation_seed=113,
    )

    assert compact_result_row(
        original,
        source_seed=111,
        evaluation_seed=113,
    ) == original


def test_public_row_exposes_function_vlm_latency_and_energy_metrics() -> None:
    row = compact_result_row(
        {
            "task_name": "Task",
            "method": "omniflow",
            "device": "small5554",
            "model_calls": 3,
            "reuse_metrics": {
                "artifact_used": True,
                "reuse_numerator": 2,
                "reuse_denominator": 4,
            },
            "performance_metrics": {
                "method_wall_sec": 12.5,
                "energy": {
                    "measurement_available": True,
                    "estimated_mwh": 1.25,
                },
            },
            "llm_usage": {"latency_ms": 345.6},
        },
        source_seed=111,
        evaluation_seed=113,
    )

    assert row["function_hit"] is True
    assert row["function_covered_steps"] == 2
    assert row["function_total_steps"] == 4
    assert row["function_step_coverage_rate"] == 0.5
    assert row["vlm_calls"] == 3
    assert row["vlm_latency_ms"] == 345.6
    assert row["latency_sec"] == 12.5
    assert row["energy_mwh"] == 1.25
    assert row["energy_measurement_available"] is True


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


def test_episode_recorder_retries_oob_host_observation(tmp_path) -> None:
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
    states = iter([ready])
    recorder = AndroidWorldEpisodeRecorder(
        lambda: next(states),
        lambda action: action,
        evidence_root=tmp_path,
    )
    recorder.start_episode()

    recorder.record_host_observation(empty)

    assert len(recorder.persist_observations()) == 1


def test_episode_recorder_exposes_latest_oob_state_to_official_validator(
    tmp_path,
) -> None:
    native_state = SimpleNamespace(
        pixels=Image.new("RGB", (4, 3), color="black"),
        forest="<hierarchy><node text=\"old\" /></hierarchy>",
        ui_elements=["old"],
        auxiliaries={"package_name": "com.example"},
    )
    oob_state = SimpleNamespace(
        pixels=Image.new("RGB", (4, 3), color="white"),
        forest="<hierarchy><node text=\"new\" /></hierarchy>",
        ui_elements=["new"],
        auxiliaries={"package_name": "com.example"},
    )
    recorder = AndroidWorldEpisodeRecorder(
        lambda: native_state,
        lambda action: action,
        evidence_root=tmp_path,
    )
    recorder.start_episode()

    recorder.record_host_observation(oob_state)

    assert recorder.get_state() is oob_state


def test_episode_recorder_retries_action_before_observation(tmp_path) -> None:
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
    states = iter([empty, ready, ready])
    recorder = AndroidWorldEpisodeRecorder(
        lambda: next(states),
        lambda action: {"success": True},
        evidence_root=tmp_path,
    )
    recorder.start_episode()

    recorder.execute_action(SimpleNamespace(action_type="click", x=1, y=2))

    assert len(recorder.persist_observations()) == 2


def test_adb_ui_xml_is_used_when_uiautomator_dump_reports_nonzero(
    tmp_path, monkeypatch
) -> None:
    xml = "<hierarchy><node package=\"com.google.android.deskclock\" /></hierarchy>"
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if "cat" in command:
            return SimpleNamespace(returncode=0, stdout=xml, stderr="")
        return SimpleNamespace(
            returncode=1,
            stdout="ERROR: could not get idle state.",
            stderr="",
        )

    monkeypatch.setattr(observation_evidence.subprocess, "run", fake_run)
    recorder = AndroidWorldEpisodeRecorder(
        lambda: None,
        lambda action: action,
        evidence_root=tmp_path,
        adb_path="adb",
        adb_serial="emulator-5560",
    )

    assert recorder._read_adb_ui_xml() == xml
    assert len(calls) == 2


def test_adb_ui_xml_fallback_rehydrates_androidworld_ui_elements(
    tmp_path, monkeypatch
) -> None:
    representation_utils = ModuleType("android_world.env.representation_utils")
    expected = [SimpleNamespace(text="Stopwatch", content_description="Stopwatch")]
    representation_utils.xml_dump_to_ui_elements = lambda _xml: expected
    android_world = ModuleType("android_world")
    android_world_env = ModuleType("android_world.env")
    android_world_env.representation_utils = representation_utils
    android_world.env = android_world_env
    monkeypatch.setitem(sys.modules, "android_world", android_world)
    monkeypatch.setitem(sys.modules, "android_world.env", android_world_env)
    monkeypatch.setitem(
        sys.modules,
        "android_world.env.representation_utils",
        representation_utils,
    )

    state = SimpleNamespace(
        pixels=None,
        forest="",
        ui_elements=[SimpleNamespace(text="Launcher", content_description="")],
        auxiliaries={},
    )
    recorder = AndroidWorldEpisodeRecorder(
        lambda: state,
        lambda action: action,
        evidence_root=tmp_path,
    )

    # Supply the XML through the method's ADB seam without touching the device.
    monkeypatch.setattr(recorder, "_read_adb_ui_xml", lambda: "<hierarchy />")
    recovered = recorder._with_adb_observation_xml(state)
    assert recovered.ui_elements == expected


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


def test_metrics_preserve_mobilegpt_oob_action_index_protocol(tmp_path) -> None:
    result_path = (
        tmp_path
        / "CameraTakePhoto"
        / "mobilegpt"
        / "small5554"
        / "task_results.jsonl"
    )
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "task_name": "CameraTakePhoto",
                "method": "mobilegpt",
                "device": "small5554",
                "official_validator_used": True,
                "official_validator_success": False,
                "oob_action_index_protocol": "mobilegpt_source_node_id_v1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    row = aggregate_task_results([result_path])["per_task"][0]

    assert row["oob_action_index_protocol"] == "mobilegpt_source_node_id_v1"


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


def test_metrics_aggregate_function_and_performance_rates(tmp_path: Path) -> None:
    result_path = tmp_path / "Task" / "omniflow" / "small5554" / "task_results.jsonl"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "task_name": "Task",
                "method": "omniflow",
                "actions_executed": 4,
                "model_calls": 2,
                "reuse_metrics": {
                    "artifact_used": True,
                    "reuse_numerator": 3,
                    "reuse_denominator": 4,
                },
                "performance_metrics": {
                    "method_wall_sec": 2.0,
                    "energy": {
                        "measurement_available": True,
                        "estimated_mwh": 0.5,
                    },
                },
                "llm_usage": {"latency_ms": 100.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = aggregate_task_results([result_path])

    assert summary["function_hit_task_count"] == 1
    assert summary["function_hit_task_rate"] == 1.0
    assert summary["function_step_coverage_rate"] == 0.75
    assert summary["per_task"][0]["vlm_latency_ms"] == 100.0
    assert summary["performance_metrics"]["energy"]["estimated_mwh_total"] == 0.5
