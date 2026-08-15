from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import stat

import pytest
from runlog_fixtures import androidworld_run_log

from src.experiment.artifact_memory import (
    _runlog_paths,
    _select_canonical_mobilegpt_memory,
    load_artifact_memory,
    refresh_artifact_memory,
    refresh_artifact_memory_from_pointer,
    registered_cell_plan_from_memory,
)
from src.experiment.artifact_memory import main as artifact_memory_main
from src.experiment.mobilegpt_contract import (
    MOBILEGPT_LEARNING_MODE,
    MOBILEGPT_MEMORY_MANIFEST,
    MOBILEGPT_MEMORY_SCHEMA,
    MOBILEGPT_PREP_TYPE,
    MOBILEGPT_PREP_TYPE_BY_SCHEMA,
    MOBILEGPT_SOURCE_METHOD,
)
from src.experiment.source_assets import store_source_run_log_sha256s
from src.integrations.runlog import adapt_source_run_log


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


ARCHIVED_MOBILEGPT_MEMORY_SCHEMA = "omniflow.mobilegpt-runlog-teacher-memory.v1"
ARCHIVED_MOBILEGPT_SOURCE_METHOD = "mobilegpt_runlog_teacher"
ARCHIVED_MOBILEGPT_LEARNING_MODE = "mobilegpt_runlog_teacher"
ARCHIVED_MOBILEGPT_PREP_TYPE = "mobilegpt_runlog_teacher_memory"


def test_noncanonical_mobilegpt_memory_cannot_be_selected() -> None:
    archived_sha = "a" * 64
    canonical_sha = "b" * 64
    records = {
        archived_sha: {
            "memory_sha256": archived_sha,
            "source_method": ARCHIVED_MOBILEGPT_SOURCE_METHOD,
        },
        canonical_sha: {
            "memory_sha256": canonical_sha,
            "source_method": MOBILEGPT_SOURCE_METHOD,
        },
    }

    with pytest.raises(ValueError, match="unsupported_mobilegpt_memory"):
        _select_canonical_mobilegpt_memory(
            task="RecordWithName",
            memory_sha256s={archived_sha, canonical_sha},
            records=records,
        )


def test_multiple_direct_mobilegpt_memories_remain_ambiguous() -> None:
    first_sha = "a" * 64
    second_sha = "b" * 64
    records = {
        digest: {
            "memory_sha256": digest,
            "source_method": MOBILEGPT_SOURCE_METHOD,
        }
        for digest in (first_sha, second_sha)
    }

    with pytest.raises(ValueError, match="ambiguous_mobilegpt_memory"):
        _select_canonical_mobilegpt_memory(
            task="RecordWithName",
            memory_sha256s={first_sha, second_sha},
            records=records,
        )


def _write_source_run_log(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "evidence" / "RecordWithName" / "source.run_log.json",
        androidworld_run_log(
            [{"action_type": "wait"}],
            task_name="RecordWithName",
            goal="Record audio and save it.",
            with_pixels=True,
        ),
    )


def test_runlog_paths_ignore_appledouble_sidecars(tmp_path: Path) -> None:
    run_log = _write_json(tmp_path / "source.run_log.json", {})
    (tmp_path / "._source.run_log.json").write_bytes(b"\x00\x05\x16\x07binary")

    assert _runlog_paths([tmp_path]) == [run_log.resolve()]


def _write_registered_result(
    root: Path,
    *,
    attempt: str,
    registered_at: str,
    success: bool,
    device: str = "small5554",
    max_steps: int = 20,
    use_oob: bool = False,
    include_task_params: bool = True,
    error: str = "",
    task_started_count: int | None = None,
    method: str = "ours",
    source_run_log: Path | None = None,
    mobilegpt_manifest: Path | None = None,
) -> Path:
    cell_root = root / "RecordWithName" / method / device / attempt
    result_path = cell_root / "registered_result.json"
    manifest_path = cell_root / "registration_manifest.json"
    registration_id = f"RecordWithName.{method}.{device}.{attempt}"
    task_params = {"file_name": "meeting.m4a"}
    command = (
        "python -m src.integrations.android_world.launch "
        f"--task-random-seed 113 --max-steps {max_steps} "
        "--fixed-task-seed --perform-emulator-setup"
    )
    if use_oob:
        command += " --oob-observe-backend androidworld"
    if method == "fixed_replay":
        if source_run_log is None:
            raise ValueError("fixed_replay source_run_log is required")
        command += f" --raw-replay-run-log {source_run_log}"
    result = {
        "schema_version": "omniflow.androidworld_registered_result.v1",
        "registration_id": registration_id,
        "attempt_id": attempt,
        "task_name": "RecordWithName",
        "source_seed": 111,
        "evaluation_seed": 113,
        "registration_manifest": str(manifest_path),
        "rows": [
            {
                "task_name": "RecordWithName",
                "method": method,
                "device": device,
                "serial": (
                    "emulator-5554"
                    if device in {"small5554", "target5554"}
                    else "emulator-5564"
                ),
                "console_port": (
                    5554 if device in {"small5554", "target5554"} else 5564
                ),
                "official_validator_used": True,
                "official_validator_success": success,
                "official_validator_task_count": 1,
                "error": error,
                **(
                    {"episode_task_started_count": task_started_count}
                    if task_started_count is not None
                    else {}
                ),
                "task_random_seed": 113,
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
                "fixed_task_seed": True,
                "fixed_task_params": False,
                "perform_emulator_setup": True,
                "command": command,
                **(
                    {
                        "execution_backend": (
                            "selector_then_scaled_coordinate_fallback_v2"
                        ),
                        "source_run_log": str(source_run_log),
                        "source_run_log_sha256": _sha256(source_run_log),
                        "replay_run_log": str(source_run_log),
                        "replay_run_log_sha256": _sha256(source_run_log),
                    }
                    if method == "fixed_replay" and source_run_log is not None
                    else {}
                ),
                **(
                    {
                        "prep_type": MOBILEGPT_PREP_TYPE_BY_SCHEMA.get(
                            json.loads(mobilegpt_manifest.read_text(encoding="utf-8"))[
                                "schema_version"
                            ],
                            ARCHIVED_MOBILEGPT_PREP_TYPE,
                        ),
                        "prep_manifest": str(mobilegpt_manifest),
                        "prep_manifest_sha256": _sha256(mobilegpt_manifest),
                        "prep_memory_sha256": "a" * 64,
                    }
                    if method == "mobilegpt_offline_retrieval"
                    and mobilegpt_manifest is not None
                    else {}
                ),
            }
        ],
    }
    _write_json(result_path, result)
    _write_json(
        manifest_path,
        {
            "schema_version": "omniflow.androidworld_result_registration.v1",
            "registration_id": registration_id,
            "immutable": True,
            "task_name": "RecordWithName",
            "method": method,
            "device": device,
            "attempt_id": attempt,
            "source_seed": 111,
            "evaluation_seed": 113,
            "registered_at": registered_at,
            "registered_result_sha256": _sha256(result_path),
        },
    )
    return result_path


def _write_mobilegpt_manifest(
    tmp_path: Path,
    *,
    schema_version: str = MOBILEGPT_MEMORY_SCHEMA,
    source_method: str = MOBILEGPT_SOURCE_METHOD,
    teacher_forcing: bool = False,
    official_source_success: bool = True,
) -> Path:
    archived = schema_version == ARCHIVED_MOBILEGPT_MEMORY_SCHEMA
    provenance = {
        "native_mobilegpt_learning": archived,
        "task_local_memory": True,
        "learning_mode": (
            ARCHIVED_MOBILEGPT_LEARNING_MODE
            if archived
            else MOBILEGPT_LEARNING_MODE
        ),
        "teacher_forcing": teacher_forcing,
        "synthetic_subtasks": not archived,
        "actions_supplied_to_mobilegpt": True,
        "function_store_used": False,
        "function_conversion_enabled": False,
        "target_inputs_read": False,
        "target_observations_read": False,
        "validator_state_read": False,
        "coordinate_replay": False,
    }
    if archived:
        provenance["complete_teacher_action_consumption"] = True
    else:
        provenance.update(
            {
                "semantic_subtasks": False,
                "original_mobilegpt_prompts": False,
                "source_transitions_supplied": True,
                "source_success_boundary_supplied": True,
                "runlog_transition_compilation": True,
                "complete_transition_mapping": True,
                "official_reader_validation": True,
                "source_emulator_used": False,
            }
        )
    payload = {
        "schema_version": schema_version,
        "task_name": "RecordWithName",
        "source_seed": 111,
        "source_method": source_method,
        "memory": {"sha256": "a" * 64, "file_count": 12},
        "provenance": provenance,
    }
    if archived:
        payload["official_source_result"] = {
            "official_validator_used": True,
            "official_validator_success": official_source_success,
        }
    return _write_json(
        tmp_path / "mobilegpt" / MOBILEGPT_MEMORY_MANIFEST,
        payload,
    )


def _write_baseline_batch_report(tmp_path: Path) -> Path:
    report_root = tmp_path / "baseline" / "iteration_01"
    cells_path = report_root / "cells.jsonl"
    rows = [
        {
            "task_name": "RecordWithName",
            "method": "mobilegpt_offline_retrieval",
            "device": "small5554",
            "source_seed": 111,
            "evaluation_seed": 113,
            "conclusion": "validator_failure",
            "status": "completed",
            "failure_summary": "official_validator_returned_false",
            "official_validator_used": True,
            "official_validator_success": False,
            "official_validator_coverage_rate": 1.0,
            "model_calls": 9,
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "actions_executed": 4,
            "episode_duration_sec": 12.5,
            "outer_wall_sec": 15.0,
            "attempt_id": "iteration_01",
            "evidence_path": "/immutable/old/small5554",
        },
        {
            "task_name": "RecordWithName",
            "method": "mobilegpt_offline_retrieval",
            "device": "fold5564",
            "source_seed": 111,
            "evaluation_seed": 113,
            "conclusion": "non_validator_failure",
            "status": "prep_failed",
            "failure_summary": "old_source_memory_failed",
            "official_validator_used": False,
            "official_validator_success": None,
            "official_validator_coverage_rate": 0.0,
            "model_calls": 3,
            "prompt_tokens": 30,
            "completion_tokens": 5,
            "total_tokens": 35,
            "actions_executed": 0,
            "episode_duration_sec": 0.0,
            "outer_wall_sec": 0.0,
            "attempt_id": "iteration_01",
            "evidence_path": "/immutable/old/fold5564",
        },
    ]
    cells_path.parent.mkdir(parents=True)
    cells_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return _write_json(
        report_root / "summary.json",
        {
            "schema_version": "omniflow.androidworld.batch_report.v1",
            "immutable": True,
            "attempt_id": "iteration_01",
            "source_seed": 111,
            "evaluation_seed": 113,
            "counts": {
                "planned": 2,
                "validator_success": 0,
                "validator_failure": 1,
                "non_validator_failure": 1,
                "pending": 0,
            },
            "cells_jsonl": str(cells_path),
        },
    )


def test_refresh_deduplicates_runlogs_and_keeps_indexed_source_as_canonical(
    tmp_path: Path,
) -> None:
    source = _write_source_run_log(tmp_path)
    duplicate = tmp_path / "other" / "RecordWithName" / "copy.run_log.json"
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(source.read_bytes())
    source_index = _write_json(
        tmp_path / "source_index.json",
        {
            "RecordWithName": {
                "task": "RecordWithName",
                "collect_seed": 111,
                "androidworld_success": True,
                "retained_source_run_log": str(source),
            }
        },
    )

    report = refresh_artifact_memory(
        memory_root=tmp_path / "memory",
        source_index=source_index,
        function_catalogs=(),
        runlog_roots=(tmp_path / "evidence", tmp_path / "other"),
        result_roots=(),
    )

    assert report["counts"]["run_log_paths"] == 2
    assert report["counts"]["unique_run_logs"] == 1
    canonical = report["canonical"]["source_run_logs"]["RecordWithName"]
    assert canonical["sha256"] == _sha256(source)
    assert canonical["aliases"] == sorted([str(source), str(duplicate)])
    object_path = Path(canonical["object_path"])
    assert object_path.is_file()
    assert object_path.read_bytes() == source.read_bytes()
    assert source.stat().st_mode & stat.S_IWUSR
    assert not object_path.stat().st_mode & stat.S_IWUSR
    current = json.loads(
        (tmp_path / "memory" / "current.json").read_text(encoding="utf-8")
    )
    assert Path(current["registry_path"]).is_file()
    assert Path(current["by_task_root"], "RecordWithName.json").is_file()


def test_refresh_applies_explicit_sha256_source_selection(
    tmp_path: Path,
) -> None:
    source = _write_source_run_log(tmp_path)
    selected = _write_json(
        tmp_path / "other" / "RecordWithName" / "selected.run_log.json",
        androidworld_run_log(
            [
                {"action_type": "click", "x": 10, "y": 10},
                {"action_type": "wait"},
            ],
            task_name="RecordWithName",
            goal="Record selected audio.",
            seed=113,
            with_pixels=True,
        ),
    )
    source_index = _write_json(
        tmp_path / "source_index.json",
        {
            "RecordWithName": {
                "task": "RecordWithName",
                "goal": "Record original audio.",
                "params": {"file_name": "original.m4a"},
                "source_seed": 111,
                "step_count": 1,
                "retained_source_run_log": str(source),
            }
        },
    )
    selection_manifest = _write_json(
        tmp_path / "source_selection.json",
        {
            "schema_version": "omniflow.androidworld-source-selection.v1",
            "selections": {
                "RecordWithName": {
                    "expected_source_run_log_sha256": _sha256(source),
                    "selected_source_run_log_sha256": _sha256(selected),
                    "reason": "Selected trajectory preserves the complete UI path.",
                }
            },
        },
    )

    report = refresh_artifact_memory(
        memory_root=tmp_path / "memory",
        source_index=source_index,
        source_selection_manifest=selection_manifest,
        function_catalogs=(),
        runlog_roots=(source.parent, selected.parent),
        result_roots=(),
    )

    canonical = report["canonical"]["source_run_logs"]["RecordWithName"]
    assert canonical["sha256"] == _sha256(selected)
    assert canonical["selection"]["expected_source_run_log_sha256"] == _sha256(source)
    assert report["counts"]["source_selection_tasks"] == 1
    source_index_payload = json.loads(
        Path(report["indexes"]["source_index"]).read_text(encoding="utf-8")
    )
    row = source_index_payload["RecordWithName"]
    assert row["retained_source_run_log_sha256"] == _sha256(selected)
    assert row["goal"] == "Record selected audio."
    assert row["source_seed"] == 113
    assert row["step_count"] == 2
    assert row["canonical_source_selection"]["reason"].startswith("Selected")


def test_refresh_rejects_stale_source_selection(tmp_path: Path) -> None:
    source = _write_source_run_log(tmp_path)
    selected = _write_json(
        tmp_path / "other" / "RecordWithName" / "selected.run_log.json",
        androidworld_run_log(
            [{"action_type": "wait"}],
            task_name="RecordWithName",
        ),
    )
    source_index = _write_json(
        tmp_path / "source_index.json",
        {
            "RecordWithName": {
                "task": "RecordWithName",
                "retained_source_run_log": str(source),
            }
        },
    )
    selection_manifest = _write_json(
        tmp_path / "source_selection.json",
        {
            "schema_version": "omniflow.androidworld-source-selection.v1",
            "selections": {
                "RecordWithName": {
                    "expected_source_run_log_sha256": "f" * 64,
                    "selected_source_run_log_sha256": _sha256(selected),
                    "reason": "Stale selection must not silently apply.",
                }
            },
        },
    )

    with pytest.raises(ValueError, match="source_selection_stale"):
        refresh_artifact_memory(
            memory_root=tmp_path / "memory",
            source_index=source_index,
            source_selection_manifest=selection_manifest,
            function_catalogs=(),
            runlog_roots=(source.parent, selected.parent),
            result_roots=(),
        )


def _write_legacy_selection_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, str]:
    source = _write_source_run_log(tmp_path)
    legacy_payload = {
        "run_id": "legacy-selected-source",
        "goal": "Tap the selected control.",
        "success": True,
        "steps": [
            {
                "observation_before_act": {
                    "hierarchy_xml": (
                        '<hierarchy><node text="Selected" /></hierarchy>'
                    ),
                    "width": 100,
                    "height": 200,
                },
                "action": {"type": "click", "params": {"x": 50, "y": 100}},
                "success": True,
            }
        ],
    }
    legacy = _write_json(
        tmp_path / "legacy" / "RecordWithName" / "legacy.run_log.json",
        legacy_payload,
    )
    source_index = _write_json(
        tmp_path / "source_index.json",
        {
            "RecordWithName": {
                "task": "RecordWithName",
                "retained_source_run_log": str(source),
            }
        },
    )
    evidence_sha256 = _sha256(legacy)
    evidence_object = (
        tmp_path
        / "memory"
        / "objects"
        / "sha256"
        / evidence_sha256[:2]
        / f"{evidence_sha256}.json"
    )
    evidence_object.parent.mkdir(parents=True, exist_ok=True)
    evidence_object.write_bytes(legacy.read_bytes())
    converted = adapt_source_run_log(
        legacy_payload,
        task_name="RecordWithName",
        task_parameters={"file_name": "selected.m4a"},
        seed=111,
        source_path=evidence_object,
        require_screenshots=False,
    )
    converted_sha256 = hashlib.sha256(_canonical_json_bytes(converted)).hexdigest()
    return source, legacy, source_index, converted_sha256


def test_refresh_converts_selected_legacy_evidence_to_official_run_log(
    tmp_path: Path,
) -> None:
    source, legacy, source_index, converted_sha256 = _write_legacy_selection_fixture(
        tmp_path
    )
    selection_manifest = _write_json(
        tmp_path / "source_selection.json",
        {
            "schema_version": "omniflow.androidworld-source-selection.v1",
            "selections": {
                "RecordWithName": {
                    "expected_source_run_log_sha256": _sha256(source),
                    "selected_source_evidence_sha256": _sha256(legacy),
                    "expected_converted_source_run_log_sha256": converted_sha256,
                    "source_seed": 111,
                    "task_parameters": {"file_name": "selected.m4a"},
                    "reason": "Legacy evidence contains the complete UI path.",
                }
            },
        },
    )

    report = refresh_artifact_memory(
        memory_root=tmp_path / "memory",
        source_index=source_index,
        source_selection_manifest=selection_manifest,
        function_catalogs=(),
        runlog_roots=(source.parent, legacy.parent),
        result_roots=(),
    )

    canonical = report["canonical"]["source_run_logs"]["RecordWithName"]
    assert canonical["sha256"] == converted_sha256
    conversion = canonical["selection"]["conversion"]
    assert conversion["selected_source_evidence_sha256"] == _sha256(legacy)
    converted = json.loads(Path(canonical["object_path"]).read_text())
    assert converted["schema_version"] == "omniflow.run_log.v1"
    assert converted["steps"][0]["observation"]["forest"].startswith("<hierarchy>")
    assert converted["steps"][0]["observation"]["pixels"] is None
    assert converted["task_parameters"] == {"file_name": "selected.m4a"}
    assert converted["seed"] == 111


def test_refresh_reuses_registered_legacy_conversion_by_exact_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, legacy, source_index, converted_sha256 = _write_legacy_selection_fixture(
        tmp_path
    )
    selection_manifest = _write_json(
        tmp_path / "source_selection.json",
        {
            "schema_version": "omniflow.androidworld-source-selection.v1",
            "selections": {
                "RecordWithName": {
                    "expected_source_run_log_sha256": _sha256(source),
                    "selected_source_evidence_sha256": _sha256(legacy),
                    "expected_converted_source_run_log_sha256": converted_sha256,
                    "source_seed": 111,
                    "task_parameters": {"file_name": "selected.m4a"},
                    "reason": "Legacy evidence contains the complete UI path.",
                }
            },
        },
    )
    refresh_artifact_memory(
        memory_root=tmp_path / "memory",
        source_index=source_index,
        source_selection_manifest=selection_manifest,
        function_catalogs=(),
        runlog_roots=(source.parent, legacy.parent),
        result_roots=(),
    )
    monkeypatch.setattr(
        "src.experiment.artifact_memory.adapt_source_run_log",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("registered conversion must not be regenerated")
        ),
    )

    second = refresh_artifact_memory(
        memory_root=tmp_path / "memory",
        source_index=source_index,
        source_selection_manifest=selection_manifest,
        function_catalogs=(),
        runlog_roots=(source.parent, legacy.parent),
        result_roots=(),
    )

    assert second["canonical"]["source_run_logs"]["RecordWithName"]["sha256"] == (
        converted_sha256
    )


def test_refresh_rejects_wrong_expected_legacy_conversion_hash(
    tmp_path: Path,
) -> None:
    source, legacy, source_index, converted_sha256 = _write_legacy_selection_fixture(
        tmp_path
    )
    wrong_sha256 = "0" * 64 if converted_sha256 != "0" * 64 else "1" * 64
    selection_manifest = _write_json(
        tmp_path / "source_selection.json",
        {
            "schema_version": "omniflow.androidworld-source-selection.v1",
            "selections": {
                "RecordWithName": {
                    "expected_source_run_log_sha256": _sha256(source),
                    "selected_source_evidence_sha256": _sha256(legacy),
                    "expected_converted_source_run_log_sha256": wrong_sha256,
                    "source_seed": 111,
                    "task_parameters": {"file_name": "selected.m4a"},
                    "reason": "A wrong conversion hash must stop refresh.",
                }
            },
        },
    )

    with pytest.raises(
        ValueError,
        match="source_selection_converted_hash_mismatch",
    ):
        refresh_artifact_memory(
            memory_root=tmp_path / "memory",
            source_index=source_index,
            source_selection_manifest=selection_manifest,
            function_catalogs=(),
            runlog_roots=(source.parent, legacy.parent),
            result_roots=(),
        )


def test_refresh_materializes_indexed_source_state_catalog(
    tmp_path: Path,
) -> None:
    source = _write_source_run_log(tmp_path)
    states = _write_json(
        tmp_path / "evidence" / "RecordWithName" / "transfer_states.json",
        {
            "schema_version": "omniflow.transfer-state-catalog.v1",
            "run_id": "source-run",
            "states": {},
        },
    )
    source_index = _write_json(
        tmp_path / "source_index.json",
        {
            "RecordWithName": {
                "task": "RecordWithName",
                "retained_source_run_log": str(source),
                "transfer_state_catalog": str(states),
            }
        },
    )

    report = refresh_artifact_memory(
        memory_root=tmp_path / "memory",
        source_index=source_index,
        function_catalogs=(),
        runlog_roots=(source.parent,),
        result_roots=(),
    )

    memory_source_index = json.loads(
        Path(report["indexes"]["source_index"]).read_text(encoding="utf-8")
    )
    row = memory_source_index["RecordWithName"]
    assert "transfer_state_catalog" not in row
    assert row["source_state_catalog_sha256"] == _sha256(states)
    materialized = Path(row["source_state_catalog"])
    assert materialized.is_file()
    assert materialized.read_bytes() == states.read_bytes()
    assert not materialized.stat().st_mode & stat.S_IWUSR


def test_refresh_classifies_case_normalized_task_directory_exactly(
    tmp_path: Path,
) -> None:
    source = _write_source_run_log(tmp_path)
    historical = _write_json(
        tmp_path
        / "raw_source_artifacts"
        / "recordwithname"
        / "historical.run_log.json",
        {
            "schema_version": "omniflow.run_log.v1",
            "run_id": "historical-run",
            "success": False,
            "steps": [{"step_index": 0}],
        },
    )
    campaign = _write_json(
        tmp_path
        / "record_with_name_source_raw_20260723"
        / "source.replay.run_log.json",
        {
            "schema_version": "omniflow.run_log.v1",
            "run_id": "campaign-run",
            "success": True,
            "steps": [{"step_index": 0}],
        },
    )
    source_index = _write_json(
        tmp_path / "source_index.json",
        {
            "RecordWithName": {
                "task": "RecordWithName",
                "retained_source_run_log": str(source),
            }
        },
    )

    report = refresh_artifact_memory(
        memory_root=tmp_path / "memory",
        source_index=source_index,
        function_catalogs=(),
        runlog_roots=(
            tmp_path / "evidence",
            historical.parents[2],
            campaign.parent,
        ),
        result_roots=(),
    )

    task = report["by_task"]["RecordWithName"]
    assert len(task["run_log_sha256s"]) == 3
    assert report["unclassified"]["run_log_sha256s"] == []


def test_refresh_keeps_one_verified_runtime_store_from_duplicate_catalogs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _write_source_run_log(tmp_path)
    source_index = _write_json(
        tmp_path / "source_index.json",
        {
            "RecordWithName": {
                "task": "RecordWithName",
                "retained_source_run_log": str(source),
            }
        },
    )
    store = _write_json(
        tmp_path / "converted" / "function_store" / "store.json",
        {
            "schema_version": "omniflow.store.v2",
            "functions": {"record_with_name": {"function_id": "record_with_name"}},
        },
    )
    transfer = _write_json(
        store.with_name("transfer_states.json"),
        {
            "schema_version": "omniflow.transfer-state-catalog.v1",
            "states": {"state-1": {"state_id": "state-1"}},
        },
    )
    provenance = _write_json(
        tmp_path / "converted" / "provenance_manifest.json",
        {"schema_version": "test.provenance.v1"},
    )
    catalog_payload = {
        "schema_version": "omniflow.function-asset-catalog.v1",
        "task_count": 1,
        "converted_task_count": 1,
        "tasks": {
            "RecordWithName": {
                "task": "RecordWithName",
                "status": "converted",
                "function_count": 1,
                "source_run_log": str(source),
                "source_run_log_sha256": _sha256(source),
                "store_path": str(store),
                "store_sha256": _sha256(store),
                "transfer_states_path": str(transfer),
                "transfer_states_sha256": _sha256(transfer),
                "provenance_path": str(provenance),
                "provenance_sha256": _sha256(provenance),
                "target_inputs_read": False,
                "target_observations_read": False,
            }
        },
    }
    catalogs = [
        _write_json(tmp_path / revision / "catalog.json", catalog_payload)
        for revision in ("v4", "v4-copy")
    ]

    report = refresh_artifact_memory(
        memory_root=tmp_path / "memory",
        source_index=source_index,
        function_catalogs=catalogs,
        runlog_roots=(tmp_path / "evidence",),
        result_roots=(),
    )

    assert report["counts"]["function_catalog_paths"] == 2
    assert report["counts"]["unique_function_stores"] == 1
    assert report["counts"]["function_store_tasks"] == 1
    canonical = report["canonical"]["function_stores"]["RecordWithName"]
    assert canonical["store_sha256"] == _sha256(store)
    assert canonical["catalog_aliases"] == sorted(str(path) for path in catalogs)
    current = json.loads(
        (tmp_path / "memory" / "current.json").read_text(encoding="utf-8")
    )
    store_index = json.loads(
        Path(current["ours_store_index"]).read_text(encoding="utf-8")
    )
    row = store_index["RecordWithName"]
    assert Path(row["store_path"]).read_bytes() == store.read_bytes()
    assert Path(row["transfer_states_path"]) == Path(row["store_path"]).with_name(
        "transfer_states.json"
    )
    assert Path(row["provenance_path"]).read_bytes() == provenance.read_bytes()
    assert Path(row["source_run_log_path"]).read_bytes() == source.read_bytes()
    assert row["source_run_log_sha256"] == _sha256(source)
    assert row["source_run_log_lineage"]["conversion"] == "identity"

    assert (
        artifact_memory_main(
            [
                "refresh",
                "--memory-root",
                str(tmp_path / "memory"),
                "--runlog-root",
                str(tmp_path / "evidence"),
            ]
        )
        == 0
    )
    refreshed = load_artifact_memory(tmp_path / "memory" / "current.json")
    assert list(refreshed["canonical"]["function_stores"]) == ["RecordWithName"]
    capsys.readouterr()
    assert (
        artifact_memory_main(
            [
                "plan",
                "--memory-index",
                str(tmp_path / "memory" / "current.json"),
                "--task",
                "RecordWithName",
                "--methods",
                "ours",
                "--devices",
                "small5554",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {
        "completed": [],
        "pending": [["ours", "small5554"]],
    }


def test_refresh_requires_exact_sha_selection_for_conflicting_function_stores(
    tmp_path: Path,
) -> None:
    source = _write_source_run_log(tmp_path)
    source_index = _write_json(
        tmp_path / "source_index.json",
        {
            "RecordWithName": {
                "task": "RecordWithName",
                "retained_source_run_log": str(source),
            }
        },
    )
    catalogs: list[Path] = []
    identities: list[str] = []
    stores: list[Path] = []
    for revision in ("v1", "v2"):
        root = tmp_path / revision
        store = _write_json(
            root / "function_store" / "store.json",
            {
                "schema_version": "omniflow.store.v2",
                "functions": {
                    "record_with_name": {
                        "function_id": "record_with_name",
                        "description": revision,
                    }
                },
            },
        )
        transfer = _write_json(
            store.with_name("transfer_states.json"),
            {
                "schema_version": "omniflow.transfer-state-catalog.v1",
                "states": {},
            },
        )
        provenance = _write_json(
            root / "provenance_manifest.json",
            {"schema_version": "test.provenance.v1", "revision": revision},
        )
        catalog = _write_json(
            root / "catalog.json",
            {
                "schema_version": "omniflow.function-asset-catalog.v1",
                "tasks": {
                    "RecordWithName": {
                        "status": "converted",
                        "source_run_log": str(source),
                        "source_run_log_sha256": _sha256(source),
                        "store_path": str(store),
                        "store_sha256": _sha256(store),
                        "transfer_states_path": str(transfer),
                        "transfer_states_sha256": _sha256(transfer),
                        "provenance_path": str(provenance),
                        "provenance_sha256": _sha256(provenance),
                        "target_inputs_read": False,
                        "target_observations_read": False,
                    }
                },
            },
        )
        identities.append(
            hashlib.sha256(
                "\0".join(
                    (
                        _sha256(source),
                        _sha256(store),
                        _sha256(transfer),
                        _sha256(provenance),
                    )
                ).encode("utf-8")
            ).hexdigest()
        )
        catalogs.append(catalog)
        stores.append(store)

    with pytest.raises(ValueError, match="ambiguous_best_function_store"):
        refresh_artifact_memory(
            memory_root=tmp_path / "memory",
            source_index=source_index,
            function_catalogs=catalogs,
            runlog_roots=(tmp_path / "evidence",),
            result_roots=(),
        )

    sorted_identities = sorted(identities)
    selection = _write_json(
        tmp_path / "function_store_selection.json",
        {
            "schema_version": (
                "omniflow.androidworld-function-store-selection.v1"
            ),
            "selections": {
                "RecordWithName": {
                    "expected_candidate_identity_sha256s": sorted_identities,
                    "selected_identity_sha256": identities[1],
                    "reason": "The second Store passed the audited replay.",
                }
            },
        },
    )
    report = refresh_artifact_memory(
        memory_root=tmp_path / "memory",
        source_index=source_index,
        function_catalogs=catalogs,
        runlog_roots=(tmp_path / "evidence",),
        result_roots=(),
        function_store_selection_manifest=selection,
    )

    canonical = report["canonical"]["function_stores"]["RecordWithName"]
    assert canonical["identity_sha256"] == identities[1]
    assert canonical["store_sha256"] == _sha256(stores[1])
    assert canonical["selection"]["reason"] == (
        "The second Store passed the audited replay."
    )
    assert report["counts"]["function_store_selection_tasks"] == 1

    stale_payload = json.loads(selection.read_text(encoding="utf-8"))
    stale_payload["selections"]["RecordWithName"][
        "expected_candidate_identity_sha256s"
    ] = sorted((identities[1], "f" * 64))
    _write_json(selection, stale_payload)
    with pytest.raises(ValueError, match="function_store_selection_stale"):
        refresh_artifact_memory(
            memory_root=tmp_path / "memory",
            source_index=source_index,
            function_catalogs=catalogs,
            runlog_roots=(tmp_path / "evidence",),
            result_roots=(),
            function_store_selection_manifest=selection,
        )


def test_refresh_reports_invalid_indexed_runlog_task_and_path(
    tmp_path: Path,
) -> None:
    invalid_source = _write_json(
        tmp_path / "invalid.run_log.json",
        {"schema_version": "omniflow.run_log.v1", "steps": []},
    )
    source_index = _write_json(
        tmp_path / "source_index.json",
        {
            "RecordWithName": {
                "task": "RecordWithName",
                "retained_source_run_log": str(invalid_source),
            }
        },
    )

    with pytest.raises(
        ValueError,
        match=(
            "indexed_source_run_log_invalid:RecordWithName:"
            + re.escape(str(invalid_source))
        ),
    ):
        refresh_artifact_memory(
            memory_root=tmp_path / "memory",
            source_index=source_index,
            function_catalogs=(),
            runlog_roots=(tmp_path,),
            result_roots=(),
        )


def test_refresh_registers_legacy_function_source_as_canonical_run_log(
    tmp_path: Path,
) -> None:
    source = _write_source_run_log(tmp_path)
    source_index = _write_json(
        tmp_path / "source_index.json",
        {
            "RecordWithName": {
                "task": "RecordWithName",
                "params": {"file_name": "meeting.m4a"},
                "source_seed": 111,
                "retained_source_run_log": str(source),
            }
        },
    )
    legacy_source = _write_json(
        tmp_path / "converted" / "legacy.run_log.json",
        {
            "schema_version": "omniflow.canonical_run_log.v1",
            "run_id": "legacy-function-source",
            "goal": "Record audio and save it.",
            "success": True,
            "steps": [
                {
                    "step_index": 0,
                    "before_state_id": "legacy-before",
                    "after_state_id": "legacy-after",
                    "action": {"tool": "click", "args": {"x": 500, "y": 500}},
                    "result": {"success": True},
                }
            ],
        },
    )
    source_transfer = _write_json(
        tmp_path / "converted" / "source_transfer_states.json",
        {
            "schema_version": "omniflow.transfer-state-catalog.v1",
            "run_id": "legacy-function-source",
            "states": {
                state_id: {
                    "state_id": state_id,
                    "xml": "<hierarchy />",
                    "package_name": "com.example.recorder",
                    "activity_name": ".MainActivity",
                    "display": {"width": 100, "height": 200},
                }
                for state_id in ("legacy-before", "legacy-after")
            },
        },
    )
    store = _write_json(
        tmp_path / "converted" / "function_store" / "store.json",
        {
            "schema_version": "omniflow.store.v2",
            "functions": {"record_with_name": {"function_id": "record_with_name"}},
        },
    )
    transfer = _write_json(
        store.with_name("transfer_states.json"),
        {"schema_version": "omniflow.transfer-state-catalog.v1", "states": {}},
    )
    provenance = _write_json(
        tmp_path / "converted" / "provenance_manifest.json",
        {"schema_version": "test.provenance.v1"},
    )
    catalog = _write_json(
        tmp_path / "converted" / "catalog.json",
        {
            "schema_version": "omniflow.function-asset-catalog.v1",
            "tasks": {
                "RecordWithName": {
                    "status": "converted",
                    "source_run_log": str(legacy_source),
                    "source_run_log_sha256": _sha256(legacy_source),
                    "source_transfer_states": str(source_transfer),
                    "source_transfer_states_sha256": _sha256(source_transfer),
                    "source_transfer_states_run_id": "legacy-function-source",
                    "store_path": str(store),
                    "store_sha256": _sha256(store),
                    "transfer_states_path": str(transfer),
                    "transfer_states_sha256": _sha256(transfer),
                    "provenance_path": str(provenance),
                    "provenance_sha256": _sha256(provenance),
                    "target_inputs_read": False,
                    "target_observations_read": False,
                }
            },
        },
    )

    report = refresh_artifact_memory(
        memory_root=tmp_path / "memory",
        source_index=source_index,
        function_catalogs=(catalog,),
        runlog_roots=(tmp_path / "evidence",),
        result_roots=(),
    )

    row = report["canonical"]["function_stores"]["RecordWithName"]
    canonical_source = json.loads(Path(row["source_run_log_path"]).read_text())
    lineage = row["source_run_log_lineage"]
    assert canonical_source["schema_version"] == "omniflow.run_log.v1"
    assert canonical_source["task_name"] == "RecordWithName"
    assert canonical_source["task_parameters"] == {"file_name": "meeting.m4a"}
    assert canonical_source["seed"] == 111
    assert canonical_source["steps"][0]["action"] == {
        "action_type": "click",
        "x": 50,
        "y": 100,
    }
    assert canonical_source["steps"][0]["observation"]["auxiliaries"][
        "display"
    ] == {"width": 100, "height": 200}
    assert canonical_source["provenance"]["source_sha256"] == _sha256(legacy_source)
    assert lineage["conversion"] == "legacy_import"
    assert lineage["source_sha256"] == _sha256(legacy_source)
    assert lineage["output_sha256"] == row["source_run_log_sha256"]
    assert Path(lineage["source_path"]).read_bytes() == legacy_source.read_bytes()
    assert Path(row["store_path"]).read_bytes() == store.read_bytes()
    assert len(report["by_task"]["RecordWithName"]["run_log_sha256s"]) == 3
    current = json.loads(
        (tmp_path / "memory" / "current.json").read_text(encoding="utf-8")
    )
    assert store_source_run_log_sha256s(
        current["ours_store_index"],
        task_name="RecordWithName",
    ) == (row["source_run_log_sha256"], _sha256(legacy_source))


def test_refresh_reuses_master_source_for_matching_legacy_store_source(
    tmp_path: Path,
) -> None:
    legacy_source = _write_json(
        tmp_path / "converted" / "legacy.run_log.json",
        {
            "schema_version": "omniflow.canonical_run_log.v1",
            "run_id": "legacy-function-source",
            "goal": "Record audio and save it.",
            "completed": True,
            "success": True,
            "steps": [
                {
                    "step_index": 0,
                    "before_state_id": "missing-from-task-catalog",
                    "action": {"type": "tap", "x": 500, "y": 500},
                    "success": True,
                }
            ],
        },
    )
    source_payload = androidworld_run_log(
        [{"action_type": "click", "x": 50, "y": 100}],
        task_name="RecordWithName",
        goal="Record audio and save it.",
    )
    source_payload["provenance"] = {
        "kind": "legacy_import",
        "source_path": str(legacy_source),
        "source_schema_version": "omniflow.canonical_run_log.v1",
        "source_sha256": _sha256(legacy_source),
    }
    source = _write_json(
        tmp_path / "evidence" / "RecordWithName" / "source.run_log.json",
        source_payload,
    )
    source_index = _write_json(
        tmp_path / "source_index.json",
        {
            "RecordWithName": {
                "task": "RecordWithName",
                "params": {"file_name": "meeting.m4a"},
                "source_seed": 111,
                "retained_source_run_log": str(source),
            }
        },
    )
    store = _write_json(
        tmp_path / "converted" / "function_store" / "store.json",
        {
            "schema_version": "omniflow.store.v2",
            "functions": {"record_with_name": {"function_id": "record_with_name"}},
        },
    )
    transfer = _write_json(
        store.with_name("transfer_states.json"),
        {"schema_version": "omniflow.transfer-state-catalog.v1", "states": {}},
    )
    provenance = _write_json(
        tmp_path / "converted" / "provenance_manifest.json",
        {"schema_version": "test.provenance.v1"},
    )
    catalog = _write_json(
        tmp_path / "converted" / "catalog.json",
        {
            "schema_version": "omniflow.function-asset-catalog.v1",
            "tasks": {
                "RecordWithName": {
                    "status": "converted",
                    "source_run_log": str(legacy_source),
                    "source_run_log_sha256": _sha256(legacy_source),
                    "store_path": str(store),
                    "store_sha256": _sha256(store),
                    "transfer_states_path": str(transfer),
                    "transfer_states_sha256": _sha256(transfer),
                    "provenance_path": str(provenance),
                    "provenance_sha256": _sha256(provenance),
                    "target_inputs_read": False,
                    "target_observations_read": False,
                }
            },
        },
    )

    report = refresh_artifact_memory(
        memory_root=tmp_path / "memory",
        source_index=source_index,
        function_catalogs=(catalog,),
        runlog_roots=(tmp_path / "evidence",),
        result_roots=(),
    )

    row = report["canonical"]["function_stores"]["RecordWithName"]
    lineage = row["source_run_log_lineage"]
    assert row["source_run_log_sha256"] == _sha256(source)
    assert Path(row["source_run_log_path"]).read_bytes() == source.read_bytes()
    assert lineage["conversion"] == "canonical_source_reuse"
    assert lineage["source_sha256"] == _sha256(legacy_source)
    assert lineage["output_sha256"] == _sha256(source)


def test_refresh_keeps_earliest_formal_result_without_success_cherry_picking(
    tmp_path: Path,
) -> None:
    source = _write_source_run_log(tmp_path)
    source_index = _write_json(
        tmp_path / "source_index.json",
        {
            "RecordWithName": {
                "task": "RecordWithName",
                "retained_source_run_log": str(source),
            }
        },
    )
    runs = tmp_path / "runs"
    first = _write_registered_result(
        runs,
        attempt="attempt_001",
        registered_at="2026-07-20T00:00:00+00:00",
        success=False,
    )
    _write_registered_result(
        runs,
        attempt="attempt_002",
        registered_at="2026-07-21T00:00:00+00:00",
        success=True,
    )

    refresh_artifact_memory(
        memory_root=tmp_path / "memory",
        source_index=source_index,
        function_catalogs=(),
        runlog_roots=(tmp_path / "evidence",),
        result_roots=(),
    )
    pointer = tmp_path / "memory" / "current.json"
    report = refresh_artifact_memory_from_pointer(
        memory_index=pointer,
        additional_result_roots=(runs,),
    )

    assert report["counts"]["result_paths"] == 4
    assert report["counts"]["canonical_result_cells"] == 1
    canonical = report["canonical"]["result_cells"][
        "RecordWithName|ours|small5554|111|113"
    ]
    assert canonical["official_validator_success"] is False
    assert canonical["registered_result_aliases"] == [str(first)]
    assert (
        canonical["selection_reason"]
        == "earliest_formal_protocol_compliant_validator_conclusion"
    )
    assert registered_cell_plan_from_memory(
        memory_index=pointer,
        task_name="RecordWithName",
        methods=("ours",),
        devices=("small5554", "fold5564"),
        source_seed=111,
        evaluation_seed=113,
    ) == {
        "completed": [("ours", "small5554")],
        "pending": [("ours", "fold5564")],
    }
    assert registered_cell_plan_from_memory(
        memory_index=pointer,
        task_name="RecordWithName",
        methods=("ours",),
        devices=("small5554",),
        source_seed=111,
        evaluation_seed=114,
    ) == {
        "completed": [],
        "pending": [("ours", "small5554")],
    }

    first_pointer = json.loads(pointer.read_text(encoding="utf-8"))
    refresh_artifact_memory(
        memory_root=tmp_path / "memory",
        source_index=source_index,
        function_catalogs=(),
        runlog_roots=(tmp_path / "evidence",),
        result_roots=(runs,),
    )
    second_pointer = json.loads(pointer.read_text(encoding="utf-8"))
    assert second_pointer == first_pointer
    assert load_artifact_memory(pointer)["canonical"] == report["canonical"]


def test_explicit_refresh_replaces_stale_recorded_roots(tmp_path: Path) -> None:
    source = _write_json(
        tmp_path / "evidence" / "source.run_log.json",
        androidworld_run_log(
            [{"action_type": "wait"}],
            task_name="RecordWithName",
            goal="Record audio and save it.",
        ),
    )
    source_index = _write_json(
        tmp_path / "source_index.json",
        {
            "RecordWithName": {
                "task": "RecordWithName",
                "retained_source_run_log": str(source),
            }
        },
    )
    stale_root = tmp_path / "stale"
    stale_root.mkdir()
    refresh_artifact_memory(
        memory_root=tmp_path / "memory",
        source_index=source_index,
        function_catalogs=(),
        runlog_roots=(tmp_path / "evidence", stale_root),
        result_roots=(stale_root,),
    )
    stale_root.rmdir()

    report = refresh_artifact_memory_from_pointer(
        memory_index=tmp_path / "memory" / "current.json",
        additional_runlog_roots=(tmp_path / "evidence",),
        additional_result_roots=(),
        replace_recorded_roots=True,
    )

    assert report["inputs"]["runlog_roots"] == [str(tmp_path / "evidence")]
    assert report["inputs"]["result_roots"] == []


def test_refresh_keeps_validator_conclusion_with_method_error(
    tmp_path: Path,
) -> None:
    source = _write_source_run_log(tmp_path)
    source_index = _write_json(
        tmp_path / "source_index.json",
        {
            "RecordWithName": {
                "task": "RecordWithName",
                "retained_source_run_log": str(source),
            }
        },
    )
    runs = tmp_path / "runs"
    result = _write_registered_result(
        runs,
        attempt="attempt_001",
        registered_at="2026-07-20T00:00:00+00:00",
        success=False,
        error="TimeoutError: timed out",
        task_started_count=1,
    )

    report = refresh_artifact_memory(
        memory_root=tmp_path / "memory",
        source_index=source_index,
        function_catalogs=(),
        runlog_roots=(tmp_path / "evidence",),
        result_roots=(runs,),
    )

    canonical = report["canonical"]["result_cells"][
        "RecordWithName|ours|small5554|111|113"
    ]
    assert canonical["official_validator_success"] is False
    assert canonical["registered_result_aliases"] == [str(result)]
    assert registered_cell_plan_from_memory(
        memory_index=tmp_path / "memory" / "current.json",
        task_name="RecordWithName",
        methods=("ours",),
        devices=("small5554",),
        source_seed=111,
        evaluation_seed=113,
    ) == {
        "completed": [("ours", "small5554")],
        "pending": [],
    }


def test_refresh_freezes_only_validator_cells_from_authoritative_batch_report(
    tmp_path: Path,
) -> None:
    source = _write_source_run_log(tmp_path)
    source_index = _write_json(
        tmp_path / "source_index.json",
        {
            "RecordWithName": {
                "task": "RecordWithName",
                "retained_source_run_log": str(source),
            }
        },
    )
    baseline_report = _write_baseline_batch_report(tmp_path)

    report = refresh_artifact_memory(
        memory_root=tmp_path / "memory",
        source_index=source_index,
        function_catalogs=(),
        runlog_roots=(tmp_path / "evidence",),
        result_roots=(),
        baseline_batch_reports=(baseline_report,),
    )

    assert report["counts"]["baseline_batch_reports"] == 1
    assert report["counts"]["baseline_validator_cells"] == 1
    assert report["counts"]["canonical_result_cells"] == 1
    cell = report["canonical"]["result_cells"][
        "RecordWithName|mobilegpt_offline_retrieval|small5554|111|113"
    ]
    assert cell["official_validator_success"] is False
    assert cell["selection_reason"] == (
        "authoritative_immutable_batch_report_validator_conclusion"
    )
    assert registered_cell_plan_from_memory(
        memory_index=tmp_path / "memory" / "current.json",
        task_name="RecordWithName",
        methods=("mobilegpt_offline_retrieval",),
        devices=("small5554", "fold5564"),
        source_seed=111,
        evaluation_seed=113,
        formal_max_steps=20,
        mobilegpt_memory_schemas=(MOBILEGPT_MEMORY_SCHEMA,),
    ) == {
        "completed": [("mobilegpt_offline_retrieval", "small5554")],
        "pending": [("mobilegpt_offline_retrieval", "fold5564")],
    }


def test_refresh_reads_archived_mobilegpt_result_without_reusing_memory(
    tmp_path: Path,
) -> None:
    source = _write_source_run_log(tmp_path)
    source_index = _write_json(
        tmp_path / "source_index.json",
        {
            "RecordWithName": {
                "task": "RecordWithName",
                "retained_source_run_log": str(source),
            }
        },
    )
    manifest = _write_mobilegpt_manifest(
        tmp_path,
        schema_version=ARCHIVED_MOBILEGPT_MEMORY_SCHEMA,
        source_method=ARCHIVED_MOBILEGPT_SOURCE_METHOD,
        teacher_forcing=True,
    )
    result = _write_registered_result(
        tmp_path / "runs",
        attempt="attempt_001",
        registered_at="2026-07-20T00:00:00+00:00",
        success=False,
        method="mobilegpt_offline_retrieval",
        mobilegpt_manifest=manifest,
    )

    report = refresh_artifact_memory(
        memory_root=tmp_path / "memory",
        source_index=source_index,
        function_catalogs=(),
        runlog_roots=(tmp_path / "evidence",),
        result_roots=(tmp_path / "runs",),
    )

    cell = report["canonical"]["result_cells"][
        "RecordWithName|mobilegpt_offline_retrieval|small5554|111|113"
    ]
    assert cell["registered_result_aliases"] == [str(result)]
    assert cell["mobilegpt_memory_schema"] == ARCHIVED_MOBILEGPT_MEMORY_SCHEMA
    assert registered_cell_plan_from_memory(
        memory_index=tmp_path / "memory" / "current.json",
        task_name="RecordWithName",
        methods=("mobilegpt_offline_retrieval",),
        devices=("small5554",),
        source_seed=111,
        evaluation_seed=113,
        formal_max_steps=20,
        mobilegpt_memory_schemas=(MOBILEGPT_MEMORY_SCHEMA,),
    ) == {
        "completed": [],
        "pending": [("mobilegpt_offline_retrieval", "small5554")],
    }


def test_refresh_prefers_current_mobilegpt_contract_over_earlier_archived_result(
    tmp_path: Path,
) -> None:
    source = _write_source_run_log(tmp_path)
    source_index = _write_json(
        tmp_path / "source_index.json",
        {
            "RecordWithName": {
                "task": "RecordWithName",
                "retained_source_run_log": str(source),
            }
        },
    )
    archived_manifest = _write_mobilegpt_manifest(
        tmp_path / "archived",
        schema_version=ARCHIVED_MOBILEGPT_MEMORY_SCHEMA,
        source_method=ARCHIVED_MOBILEGPT_SOURCE_METHOD,
        teacher_forcing=True,
    )
    _write_registered_result(
        tmp_path / "runs",
        attempt="attempt_001",
        registered_at="2026-07-20T00:00:00+00:00",
        success=True,
        method="mobilegpt_offline_retrieval",
        mobilegpt_manifest=archived_manifest,
    )
    current_manifest = _write_mobilegpt_manifest(tmp_path / "current")
    current = _write_registered_result(
        tmp_path / "runs",
        attempt="attempt_002",
        registered_at="2026-07-21T00:00:00+00:00",
        success=False,
        method="mobilegpt_offline_retrieval",
        mobilegpt_manifest=current_manifest,
    )

    report = refresh_artifact_memory(
        memory_root=tmp_path / "memory",
        source_index=source_index,
        function_catalogs=(),
        runlog_roots=(tmp_path / "evidence",),
        result_roots=(tmp_path / "runs",),
    )

    cell = report["canonical"]["result_cells"][
        "RecordWithName|mobilegpt_offline_retrieval|small5554|111|113"
    ]
    assert cell["registered_result_aliases"] == [str(current)]
    assert cell["mobilegpt_memory_schema"] == MOBILEGPT_MEMORY_SCHEMA
    assert cell["official_validator_success"] is False


@pytest.mark.parametrize(
    ("schema_version", "source_method", "teacher_forcing", "expected_error"),
    (
        (
            "omniflow.mobilegpt-cold-memory.v1",
            MOBILEGPT_SOURCE_METHOD,
            True,
            "schema",
        ),
        (
            MOBILEGPT_MEMORY_SCHEMA,
            "fixed_replay",
            True,
            "source_method",
        ),
        (
            MOBILEGPT_MEMORY_SCHEMA,
            MOBILEGPT_SOURCE_METHOD,
            True,
            "provenance_teacher_forcing",
        ),
    ),
)
def test_refresh_preserves_but_excludes_non_teacher_mobilegpt_result(
    tmp_path: Path,
    schema_version: str,
    source_method: str,
    teacher_forcing: bool,
    expected_error: str,
) -> None:
    source = _write_source_run_log(tmp_path)
    source_index = _write_json(
        tmp_path / "source_index.json",
        {
            "RecordWithName": {
                "task": "RecordWithName",
                "retained_source_run_log": str(source),
            }
        },
    )
    manifest = _write_mobilegpt_manifest(
        tmp_path,
        schema_version=schema_version,
        source_method=source_method,
        teacher_forcing=teacher_forcing,
    )
    result = _write_registered_result(
        tmp_path / "runs",
        attempt="attempt_001",
        registered_at="2026-07-20T00:00:00+00:00",
        success=False,
        method="mobilegpt_offline_retrieval",
        mobilegpt_manifest=manifest,
    )

    report = refresh_artifact_memory(
        memory_root=tmp_path / "memory",
        source_index=source_index,
        function_catalogs=(),
        runlog_roots=(tmp_path / "evidence",),
        result_roots=(tmp_path / "runs",),
    )

    assert report["canonical"]["result_cells"] == {}
    errors = report["artifacts"]["results"][_sha256(result)][
        "canonical_exclusion_errors"
    ]
    assert any(expected_error in error for error in errors)


def test_refresh_selects_fixed_replay_result_for_canonical_source_only(
    tmp_path: Path,
) -> None:
    canonical_source = _write_source_run_log(tmp_path)
    stale_source = _write_json(
        tmp_path / "stale" / "RecordWithName" / "source.run_log.json",
        androidworld_run_log(
            [{"action_type": "wait"}, {"action_type": "wait"}],
            task_name="RecordWithName",
            goal="Record stale audio.",
            with_pixels=True,
        ),
    )
    source_index = _write_json(
        tmp_path / "source_index.json",
        {
            "RecordWithName": {
                "task": "RecordWithName",
                "retained_source_run_log": str(canonical_source),
            }
        },
    )
    runs = tmp_path / "runs"
    stale = _write_registered_result(
        runs,
        attempt="attempt_001",
        registered_at="2026-07-20T00:00:00+00:00",
        success=False,
        method="fixed_replay",
        source_run_log=stale_source,
    )
    current = _write_registered_result(
        runs,
        attempt="attempt_002",
        registered_at="2026-07-21T00:00:00+00:00",
        success=True,
        method="fixed_replay",
        source_run_log=canonical_source,
    )

    report = refresh_artifact_memory(
        memory_root=tmp_path / "memory",
        source_index=source_index,
        function_catalogs=(),
        runlog_roots=(tmp_path / "evidence", tmp_path / "stale"),
        result_roots=(runs,),
    )

    cell = report["canonical"]["result_cells"][
        "RecordWithName|fixed_replay|small5554|111|113"
    ]
    assert cell["registered_result_aliases"] == [str(current)]
    assert cell["source_run_log_sha256"] == _sha256(canonical_source)
    stale_record = report["artifacts"]["results"][_sha256(stale)]
    assert any(
        "formal_result_fixed_replay_source_hash_mismatch" in error
        for error in stale_record["canonical_exclusion_errors"]
    )


def test_refresh_preserves_but_does_not_select_environment_error_result(
    tmp_path: Path,
) -> None:
    source = _write_source_run_log(tmp_path)
    source_index = _write_json(
        tmp_path / "source_index.json",
        {
            "RecordWithName": {
                "task": "RecordWithName",
                "retained_source_run_log": str(source),
            }
        },
    )
    runs = tmp_path / "runs"
    invalid = _write_registered_result(
        runs,
        attempt="attempt_001",
        registered_at="2026-07-20T00:00:00+00:00",
        success=False,
        error="FileNotFoundError: app database missing",
    )

    report = refresh_artifact_memory(
        memory_root=tmp_path / "memory",
        source_index=source_index,
        function_catalogs=(),
        runlog_roots=(tmp_path / "evidence",),
        result_roots=(runs,),
    )

    assert report["counts"]["canonical_result_cells"] == 0
    assert report["canonical"]["result_cells"] == {}
    assert report["artifacts"]["results"][_sha256(invalid)][
        "verified_registration"
    ] is True


def test_refresh_normalizes_legacy_target_device_labels(
    tmp_path: Path,
) -> None:
    source = _write_source_run_log(tmp_path)
    source_index = _write_json(
        tmp_path / "source_index.json",
        {
            "RecordWithName": {
                "task": "RecordWithName",
                "retained_source_run_log": str(source),
            }
        },
    )
    runs = tmp_path / "runs"
    _write_registered_result(
        runs,
        attempt="attempt_001",
        registered_at="2026-07-20T00:00:00+00:00",
        success=True,
        device="target5554",
    )

    report = refresh_artifact_memory(
        memory_root=tmp_path / "memory",
        source_index=source_index,
        function_catalogs=(),
        runlog_roots=(tmp_path / "evidence",),
        result_roots=(runs,),
    )

    cell = report["canonical"]["result_cells"]["RecordWithName|ours|small5554|111|113"]
    assert cell["device"] == "small5554"
    assert cell["registered_device_label"] == "target5554"


def test_refresh_preserves_but_does_not_select_incompatible_formal_result(
    tmp_path: Path,
) -> None:
    source = _write_source_run_log(tmp_path)
    source_index = _write_json(
        tmp_path / "source_index.json",
        {
            "RecordWithName": {
                "task": "RecordWithName",
                "retained_source_run_log": str(source),
            }
        },
    )
    runs = tmp_path / "runs"
    incompatible = _write_registered_result(
        runs,
        attempt="attempt_001",
        registered_at="2026-07-20T00:00:00+00:00",
        success=False,
        max_steps=30,
        use_oob=True,
        include_task_params=False,
    )
    report = refresh_artifact_memory(
        memory_root=tmp_path / "memory",
        source_index=source_index,
        function_catalogs=(),
        runlog_roots=(tmp_path / "evidence",),
        result_roots=(runs,),
    )

    assert report["counts"]["canonical_result_cells"] == 0
    assert report["counts"]["formal_protocol_excluded_results"] == 1
    assert report["canonical"]["result_cells"] == {}
    record = report["artifacts"]["results"][_sha256(incompatible)]
    assert record["verified_registration"] is True
    assert any(
        "formal_result_protocol_mismatch" in error
        for error in record["canonical_exclusion_errors"]
    )
    assert registered_cell_plan_from_memory(
        memory_index=tmp_path / "memory" / "current.json",
        task_name="RecordWithName",
        methods=("ours",),
        devices=("small5554",),
        source_seed=111,
        evaluation_seed=113,
        formal_max_steps=20,
    ) == {
        "completed": [],
        "pending": [("ours", "small5554")],
    }


def test_memory_plan_rejects_incompatible_canonical_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incompatible = _write_registered_result(
        tmp_path / "runs",
        attempt="attempt_001",
        registered_at="2026-07-20T00:00:00+00:00",
        success=False,
        max_steps=30,
        use_oob=True,
        include_task_params=False,
    )
    cell_key = "RecordWithName|ours|small5554|111|113"
    monkeypatch.setattr(
        "src.experiment.artifact_memory.load_artifact_memory",
        lambda _: {
            "canonical": {
                "result_cells": {
                    cell_key: {
                        "registered_result_object_path": str(incompatible),
                        "registered_result_sha256": _sha256(incompatible),
                    }
                }
            }
        },
    )

    with pytest.raises(ValueError, match="formal_result_protocol_mismatch"):
        registered_cell_plan_from_memory(
            memory_index=tmp_path / "memory" / "current.json",
            task_name="RecordWithName",
            methods=("ours",),
            devices=("small5554",),
            source_seed=111,
            evaluation_seed=113,
            formal_max_steps=20,
        )
