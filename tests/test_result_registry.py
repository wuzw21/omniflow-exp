from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from runlog_fixtures import androidworld_run_log, androidworld_state

from src.experiment.artifact_memory import (
    load_artifact_memory,
    refresh_artifact_memory,
)
from src.experiment.result_registry import (
    register_attempt_summary,
    registered_result_plan,
)
from src.experiment.result_schema import RESULT_FIELDS


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_registered_result(
    runs_root: Path,
    *,
    task: str,
    method: str,
    device: str,
    success: bool | None,
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
    runtime_integrity_error: str = "",
    environment_failure: bool = False,
) -> None:
    result = runs_root / task / method / device / attempt
    result_path = result / "registered_result.json"
    manifest_path = result / "registration_manifest.json"
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
                "runtime_integrity_error": runtime_integrity_error,
                "environment_failure": environment_failure,
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


def test_registered_result_plan_skips_any_result_with_a_verified_conclusion(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    task = "AudioRecorderRecordAudioWithFileName"
    _write_registered_result(
        runs_root,
        task=task,
        method="fixed_replay",
        device="small5554",
        success=True,
    )
    _write_registered_result(
        runs_root,
        task=task,
        method="ours",
        device="fold5564",
        success=False,
    )

    plan = registered_result_plan(
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


def test_registered_result_plan_rejects_legacy_coordinate_fixed_replay(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    task = "AudioRecorderRecordAudio"
    _write_registered_result(
        runs_root,
        task=task,
        method="fixed_replay",
        device="small5554",
        success=True,
        legacy_fixed_replay=True,
    )

    with pytest.raises(ValueError, match="formal_result_protocol_mismatch"):
        registered_result_plan(
            runs_root=runs_root,
            task_name=task,
            methods=("fixed_replay",),
            devices=("small5554",),
            source_seed=111,
            evaluation_seed=113,
            formal_max_steps=20,
        )


def test_registered_result_plan_rejects_previous_selector_stop_policy(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    task = "BrowserMaze"
    _write_registered_result(
        runs_root,
        task=task,
        method="fixed_replay",
        device="small5554",
        success=False,
        fixed_replay_backend="selector_then_scaled_coordinate_replay",
    )

    with pytest.raises(ValueError, match="formal_result_protocol_mismatch"):
        registered_result_plan(
            runs_root=runs_root,
            task_name=task,
            methods=("fixed_replay",),
            devices=("small5554",),
            source_seed=111,
            evaluation_seed=113,
            formal_max_steps=20,
        )


def test_registered_result_plan_accepts_coordinate_replay_missing_redundant_audit_flag(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    task = "BrowserMaze"
    _write_registered_result(
        runs_root,
        task=task,
        method="fixed_replay",
        device="small5554",
        success=False,
        include_uses_source_xml=False,
    )

    plan = registered_result_plan(
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


def test_registered_result_plan_does_not_skip_a_different_evaluation_seed(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    task = "AudioRecorderRecordAudioWithFileName"
    _write_registered_result(
        runs_root,
        task=task,
        method="ours",
        device="small5554",
        success=True,
        source_seed=111,
        evaluation_seed=113,
    )

    plan = registered_result_plan(
        runs_root=runs_root,
        task_name=task,
        methods=("ours",),
        devices=("small5554",),
        source_seed=111,
        evaluation_seed=114,
    )

    assert plan["completed"] == []
    assert plan["pending"] == [("ours", "small5554")]


def test_registered_result_plan_retries_rows_without_validator_coverage(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    task = "AudioRecorderRecordAudioWithFileName"
    _write_registered_result(
        runs_root,
        task=task,
        method="ours",
        device="small5554",
        success=False,
        validator_task_count=0,
        validator_used=False,
    )

    plan = registered_result_plan(
        runs_root=runs_root,
        task_name=task,
        methods=("ours",),
        devices=("small5554",),
        source_seed=111,
        evaluation_seed=113,
    )

    assert plan["completed"] == []
    assert plan["pending"] == [("ours", "small5554")]


def test_registered_result_plan_keeps_validator_rows_with_environment_error(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    task = "NotesIsTodo"
    _write_registered_result(
        runs_root,
        task=task,
        method="fixed_replay",
        device="fold5564",
        success=False,
        error="FileNotFoundError: app database missing",
    )

    plan = registered_result_plan(
        runs_root=runs_root,
        task_name=task,
        methods=("fixed_replay",),
        devices=("fold5564",),
        source_seed=111,
        evaluation_seed=113,
    )

    assert plan["completed"] == [("fixed_replay", "fold5564")]
    assert plan["pending"] == []


def test_registered_result_plan_accepts_per_episode_validator_conclusion(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    task = "AudioRecorderRecordAudioWithFileName"
    _write_registered_result(
        runs_root,
        task=task,
        method="fixed_replay",
        device="small5554",
        success=False,
        validator_task_count=0,
        validator_used=True,
    )

    plan = registered_result_plan(
        runs_root=runs_root,
        task_name=task,
        methods=("fixed_replay",),
        devices=("small5554",),
        source_seed=111,
        evaluation_seed=113,
    )

    assert plan["completed"] == [("fixed_replay", "small5554")]
    assert plan["pending"] == []


def test_registered_result_plan_does_not_treat_coverage_as_a_conclusion(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    task = "AudioRecorderRecordAudioWithFileName"
    _write_registered_result(
        runs_root,
        task=task,
        method="t3a_hint",
        device="fold5564",
        success=None,
        validator_task_count=1,
        validator_used=True,
    )

    plan = registered_result_plan(
        runs_root=runs_root,
        task_name=task,
        methods=("t3a_hint",),
        devices=("fold5564",),
        source_seed=111,
        evaluation_seed=113,
    )

    assert plan["completed"] == []
    assert plan["pending"] == [("t3a_hint", "fold5564")]


@pytest.mark.parametrize(
    "parser_error",
    (
        "TypeError: baseline action parser failed",
        "ValueError: baseline action parser failed",
        "unclassified parser failure",
    ),
)
def test_registered_result_plan_keeps_validator_failure_with_parser_error(
    tmp_path: Path,
    parser_error: str,
) -> None:
    runs_root = tmp_path / "runs"
    task = "AudioRecorderRecordAudioWithFileName"
    _write_registered_result(
        runs_root,
        task=task,
        method="t3a_hint",
        device="fold5564",
        success=False,
        error=parser_error,
        runtime_integrity_error=parser_error,
    )

    plan = registered_result_plan(
        runs_root=runs_root,
        task_name=task,
        methods=("t3a_hint",),
        devices=("fold5564",),
        source_seed=111,
        evaluation_seed=113,
    )

    assert plan["completed"] == [("t3a_hint", "fold5564")]
    assert plan["pending"] == []


@pytest.mark.parametrize(
    ("method", "runtime_integrity_error", "environment_failure", "error"),
    (
        ("ours", "mobilegpt_app_ui_not_ready", False, ""),
        ("ours", "", True, ""),
        (
            "mobilegpt_offline_retrieval",
            "",
            False,
            "RuntimeError: mobilegpt_app_ui_not_ready:wrong_app",
        ),
    ),
)
def test_registered_result_plan_keeps_validator_conclusions_with_error_evidence(
    tmp_path: Path,
    method: str,
    runtime_integrity_error: str,
    environment_failure: bool,
    error: str,
) -> None:
    runs_root = tmp_path / "runs"
    task = "ContactsNewContactDraft"
    _write_registered_result(
        runs_root,
        task=task,
        method=method,
        device="small5554",
        success=False,
        runtime_integrity_error=runtime_integrity_error,
        environment_failure=environment_failure,
        error=error,
    )

    plan = registered_result_plan(
        runs_root=runs_root,
        task_name=task,
        methods=(method,),
        devices=("small5554",),
        source_seed=111,
        evaluation_seed=113,
    )

    assert plan["completed"] == [(method, "small5554")]
    assert plan["pending"] == []


def test_registered_result_plan_rejects_incompatible_formal_protocol(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    task = "BrowserDraw"
    _write_registered_result(
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
        registered_result_plan(
            runs_root=runs_root,
            task_name=task,
            methods=("t3a_hint",),
            devices=("small5554",),
            source_seed=111,
            evaluation_seed=113,
            formal_max_steps=20,
        )


def test_registered_result_plan_uses_earliest_validator_conclusion(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    task = "BrowserDraw"
    for attempt, max_steps in (("iteration_01", 20), ("iteration_02", 30)):
        _write_registered_result(
            runs_root,
            task=task,
            method="t3a_hint",
            device="small5554",
            success=False,
            attempt=attempt,
            max_steps=max_steps,
        )

    assert registered_result_plan(
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
    screenshot = source_run_log.parent / "state-0.png"
    screenshot.parent.mkdir(parents=True, exist_ok=True)
    screenshot.write_bytes(b"test screenshot")
    observation = androidworld_state("state-0", with_pixels=True)
    observation["pixels"]["path"] = str(screenshot)
    observation["pixels"]["sha256"] = hashlib.sha256(
        screenshot.read_bytes()
    ).hexdigest()
    _write_json(
        source_run_log,
        androidworld_run_log(
            [{"action_type": "wait"}],
            observations=[observation],
            task_name="TaskOne",
            goal="Complete task one.",
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
        artifact_memory_index=memory_root / "current.json",
    )

    assert registration["artifact_memory_updated"] is True
    memory = load_artifact_memory(memory_root / "current.json")
    result = memory["canonical"]["result_cells"][
        "TaskOne|ours|small5554|111|113"
    ]
    assert result["official_validator_success"] is False


def test_result_registration_keeps_runtime_integrity_evidence_after_conclusion(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "attempt" / "one_task_summary.json"
    _write_json(
        summary,
        {
            "task_name": "ContactsNewContactDraft",
            "rows": [
                {
                    "method": "mobilegpt_offline_retrieval",
                    "device": "small5554",
                    "official_validator_used": True,
                    "official_validator_success": False,
                    "runtime_integrity_error": "mobilegpt_app_ui_not_ready",
                }
            ],
        },
    )
    attempt_manifest = summary.with_name("attempt_manifest.json")
    _write_json(
        attempt_manifest,
        {
            "immutable": True,
            "attempt_id": "iteration_01",
            "source_seed": 111,
            "evaluation_seed": 113,
        },
    )
    runs_root = tmp_path / "runs"
    source_index = tmp_path / "source_index.json"
    _write_json(source_index, {})

    registration = register_attempt_summary(
        summary_path=summary,
        attempt_manifest_path=attempt_manifest,
        runs_root=runs_root,
    )

    registered = json.loads(
        Path(registration["registered_results"][0]).read_text(encoding="utf-8")
    )
    assert tuple(registered["rows"][0]) == RESULT_FIELDS
    assert registered["rows"][0]["task"] == "ContactsNewContactDraft"
    assert registered["rows"][0]["validator_success"] is False
    assert registered["details"][0]["runtime_integrity_error"] == (
        "mobilegpt_app_ui_not_ready"
    )


def test_result_registration_rejects_missing_boolean_validator_conclusion(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "attempt" / "one_task_summary.json"
    _write_json(
        summary,
        {
            "task_name": "ContactsNewContactDraft",
            "rows": [
                {
                    "method": "t3a_hint",
                    "device": "fold5564",
                    "official_validator_used": True,
                    "official_validator_success": None,
                    "official_validator_task_count": 1,
                    "official_validator_coverage_rate": 1.0,
                }
            ],
        },
    )
    attempt_manifest = summary.with_name("attempt_manifest.json")
    _write_json(
        attempt_manifest,
        {
            "immutable": True,
            "attempt_id": "iteration_01",
            "source_seed": 111,
            "evaluation_seed": 113,
        },
    )
    runs_root = tmp_path / "runs"

    with pytest.raises(ValueError, match="official_validator_conclusion_missing"):
        register_attempt_summary(
            summary_path=summary,
            attempt_manifest_path=attempt_manifest,
            runs_root=runs_root,
        )

    assert not runs_root.exists()


@pytest.mark.parametrize(
    "parser_error",
    (
        "TypeError: baseline action parser failed",
        "ValueError: baseline action parser failed",
        "arbitrary parser failure",
    ),
)
def test_result_registration_keeps_parser_failure_after_validator_conclusion(
    tmp_path: Path,
    parser_error: str,
) -> None:
    summary = tmp_path / "attempt" / "one_task_summary.json"
    _write_json(
        summary,
        {
            "task_name": "ExpenseAddSingle",
            "rows": [
                {
                    "method": "mobilegpt_offline_retrieval",
                    "device": "small5554",
                    "official_validator_used": True,
                    "official_validator_success": False,
                    "error": parser_error,
                    "runtime_integrity_error": parser_error,
                }
            ],
        },
    )
    attempt_manifest = summary.with_name("attempt_manifest.json")
    _write_json(
        attempt_manifest,
        {
            "immutable": True,
            "attempt_id": "iteration_01",
            "source_seed": 111,
            "evaluation_seed": 113,
        },
    )
    source_index = tmp_path / "source_index.json"
    _write_json(source_index, {})

    registration = register_attempt_summary(
        summary_path=summary,
        attempt_manifest_path=attempt_manifest,
        runs_root=tmp_path / "runs",
    )

    registered = Path(registration["registered_results"][0])
    payload = json.loads(registered.read_text(encoding="utf-8"))
    assert tuple(payload["rows"][0]) == RESULT_FIELDS
    assert payload["rows"][0]["validator_success"] is False
    assert payload["details"][0]["runtime_integrity_error"] == parser_error
