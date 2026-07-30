from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat

import pytest

from src.experiment.artifact_memory import (
    load_artifact_memory,
    refresh_artifact_memory,
    refresh_artifact_memory_from_pointer,
    registered_cell_plan_from_memory,
)
from src.experiment.artifact_memory import main as artifact_memory_main


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
) -> Path:
    cell_root = root / "RecordWithName" / "ours" / device / attempt
    result_path = cell_root / "registered_result.json"
    manifest_path = cell_root / "registration_manifest.json"
    registration_id = f"RecordWithName.ours.{device}.{attempt}"
    task_params = {"file_name": "meeting.m4a"}
    command = (
        "python -m src.integrations.android_world.launch "
        f"--task-random-seed 113 --max-steps {max_steps} "
        "--fixed-task-seed --perform-emulator-setup"
    )
    if use_oob:
        command += " --oob-observe-backend androidworld"
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
                "method": "ours",
                "device": device,
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
                "official_validator_used": True,
                "official_validator_success": success,
                "official_validator_task_count": 1,
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
            "method": "ours",
            "device": device,
            "attempt_id": attempt,
            "source_seed": 111,
            "evaluation_seed": 113,
            "registered_at": registered_at,
            "registered_result_sha256": _sha256(result_path),
        },
    )
    return result_path


def test_refresh_deduplicates_runlogs_and_keeps_indexed_source_as_canonical(
    tmp_path: Path,
) -> None:
    source = _write_json(
        tmp_path / "evidence" / "RecordWithName" / "source.run_log.json",
        {
            "schema_version": "omniflow.run_log.v1",
            "run_id": "source-run",
            "goal": "Record audio and save it.",
            "success": True,
            "steps": [{"step_index": 0}],
        },
    )
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


def test_refresh_materializes_indexed_source_state_catalog(
    tmp_path: Path,
) -> None:
    source = _write_json(
        tmp_path / "evidence" / "RecordWithName" / "source.run_log.json",
        {
            "schema_version": "omniflow.canonical_run_log.v1",
            "run_id": "source-run",
            "goal": "Record audio and save it.",
            "status": "succeeded",
            "success": True,
            "steps": [{"step_index": 0}],
        },
    )
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
    source = _write_json(
        tmp_path / "evidence" / "RecordWithName" / "source.run_log.json",
        {
            "schema_version": "omniflow.run_log.v1",
            "run_id": "source-run",
            "success": True,
            "steps": [{"step_index": 0}],
        },
    )
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
) -> None:
    source = _write_json(
        tmp_path / "evidence" / "RecordWithName" / "source.run_log.json",
        {
            "schema_version": "omniflow.run_log.v1",
            "run_id": "source-run",
            "goal": "Record audio and save it.",
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
    assert Path(row["transfer_states_path"]) == Path(
        row["store_path"]
    ).with_name("transfer_states.json")
    assert Path(row["provenance_path"]).read_bytes() == provenance.read_bytes()
    assert Path(row["source_run_log_path"]).read_bytes() == source.read_bytes()
    assert row["source_run_log_sha256"] == _sha256(source)

    assert artifact_memory_main(
        [
            "refresh",
            "--memory-root",
            str(tmp_path / "memory"),
            "--source-index",
            str(source_index),
            "--runlog-root",
            str(tmp_path / "evidence"),
        ]
    ) == 0
    refreshed = load_artifact_memory(tmp_path / "memory" / "current.json")
    assert list(refreshed["canonical"]["function_stores"]) == [
        "RecordWithName"
    ]


def test_refresh_keeps_earliest_formal_result_without_success_cherry_picking(
    tmp_path: Path,
) -> None:
    source = _write_json(
        tmp_path / "evidence" / "RecordWithName" / "source.run_log.json",
        {
            "schema_version": "omniflow.run_log.v1",
            "run_id": "source-run",
            "goal": "Record audio and save it.",
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
        == "earliest_verified_official_validator_conclusion"
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


def test_refresh_normalizes_legacy_target_device_labels(
    tmp_path: Path,
) -> None:
    source = _write_json(
        tmp_path / "evidence" / "RecordWithName" / "source.run_log.json",
        {
            "schema_version": "omniflow.run_log.v1",
            "run_id": "source-run",
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

    cell = report["canonical"]["result_cells"][
        "RecordWithName|ours|small5554|111|113"
    ]
    assert cell["device"] == "small5554"
    assert cell["registered_device_label"] == "target5554"


def test_memory_plan_rejects_incompatible_formal_protocol(
    tmp_path: Path,
) -> None:
    source = _write_json(
        tmp_path / "evidence" / "RecordWithName" / "source.run_log.json",
        {
            "schema_version": "omniflow.run_log.v1",
            "run_id": "source-run",
            "goal": "Record audio and save it.",
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
    runs = tmp_path / "runs"
    _write_registered_result(
        runs,
        attempt="attempt_001",
        registered_at="2026-07-20T00:00:00+00:00",
        success=False,
        max_steps=30,
        use_oob=True,
        include_task_params=False,
    )
    refresh_artifact_memory(
        memory_root=tmp_path / "memory",
        source_index=source_index,
        function_catalogs=(),
        runlog_roots=(tmp_path / "evidence",),
        result_roots=(runs,),
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
