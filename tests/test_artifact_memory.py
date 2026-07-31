from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat

import pytest
from runlog_fixtures import androidworld_run_log

from src.experiment.artifact_memory import (
    load_artifact_memory,
    refresh_artifact_memory,
    refresh_artifact_memory_from_pointer,
    registered_cell_plan_from_memory,
)
from src.experiment.artifact_memory import main as artifact_memory_main
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
                    5554 if device in {"small5554", "target5554"} else 5564
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
                "--source-index",
                str(source_index),
                "--runlog-root",
                str(tmp_path / "evidence"),
            ]
        )
        == 0
    )
    refreshed = load_artifact_memory(tmp_path / "memory" / "current.json")
    assert list(refreshed["canonical"]["function_stores"]) == ["RecordWithName"]


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
