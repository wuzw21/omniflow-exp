from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat

import pytest

from omniflow.functions.store import FunctionStore
from src.experiment.artifact_memory import (
    load_artifact_memory,
    refresh_artifact_memory,
)
from src.experiment.function_assets import convert_function_assets, main


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _legacy_bundle() -> dict:
    return {
        "schema_version": "omniflow.function-bundle.v2",
        "source_run_id": "current-source",
        "source_success": True,
        "source_arguments": {
            "record_with_name": {"filename": "source_name.m4a"}
        },
        "functions": [
            {
                "schema_version": "omniflow.function.v2",
                "id": "record_with_name",
                "description": "Record and save audio with the requested filename.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Exact requested filename.",
                        }
                    },
                    "required": ["filename"],
                    "additionalProperties": False,
                },
                "bindings": [
                    {
                        "source": "$.arguments.filename",
                        "target": "$.actions[2].arguments.text",
                    }
                ],
                "actions": [
                    {
                        "tool": "open_app",
                        "arguments": {"app_name": "audio recorder"},
                    },
                    {
                        "tool": "click",
                        "arguments": {"x": 503, "y": 801},
                    },
                    {
                        "tool": "input_text",
                        "arguments": {
                            "text": "",
                            "x": 500,
                            "y": 200,
                            "clear_text": True,
                        },
                    },
                ],
                "checker_rules": [],
            }
        ],
    }


def _source_assets(root: Path) -> Path:
    menu_xml = (
        '<hierarchy><node text="Record" resource-id="record" '
        'class="android.widget.Button" package="com.example.recorder" '
        'clickable="true" enabled="true" '
        'bounds="[450,750][550,850]" /></hierarchy>'
    )
    input_xml = (
        '<hierarchy><node text="File name" resource-id="filename" '
        'class="android.widget.EditText" editable="true" '
        'package="com.example.recorder" clickable="true" enabled="true" '
        'bounds="[400,100][600,300]" /></hierarchy>'
    )
    run_log = _write_json(
        root / "RecordWithName" / "source.run_log.json",
        {
            "run_id": "current-source",
            "goal": "Record and save audio as source_name.m4a.",
            "success": True,
            "steps": [
                {
                    "observation_before_act": {
                        "state_id": "state-open",
                        "package_name": "com.android.launcher",
                        "display_width": 1000,
                        "display_height": 1000,
                    },
                    "action": {
                        "tool": "open_app",
                        "args": {"package_name": "com.example.recorder"},
                    },
                    "result": {"success": True},
                },
                {
                    "observation_before_act": {
                        "state_id": "state-menu",
                        "xml": menu_xml,
                        "package_name": "com.example.recorder",
                        "activity_name": ".MainActivity",
                        "display_width": 1000,
                        "display_height": 1000,
                    },
                    "action": {"tool": "click", "args": {"x": 500, "y": 800}},
                    "result": {"success": True},
                },
                {
                    "observation_before_act": {
                        "state_id": "state-input",
                        "xml": input_xml,
                        "package_name": "com.example.recorder",
                        "activity_name": ".MainActivity",
                        "display_width": 1000,
                        "display_height": 1000,
                    },
                    "action": {"tool": "click", "args": {"x": 500, "y": 200}},
                    "result": {"success": True},
                },
                {
                    "observation_before_act": {
                        "state_id": "state-input",
                        "xml": input_xml,
                        "package_name": "com.example.recorder",
                        "activity_name": ".MainActivity",
                        "display_width": 1000,
                        "display_height": 1000,
                    },
                    "action": {
                        "tool": "input_text",
                        "args": {"text": "source_name.m4a"},
                    },
                    "result": {"success": True},
                },
            ],
        },
    )
    return _write_json(
        root / "index_by_task.json",
        {
            "RecordWithName": {
                "task": "RecordWithName",
                "collect_seed": 111,
                "retained_source_run_log": str(run_log),
                "retained_source_run_log_sha256": _sha256(run_log),
            }
        },
    )


def test_conversion_deduplicates_task_bundles_and_rebinds_current_steps(
    tmp_path: Path,
) -> None:
    bundle = _legacy_bundle()
    legacy_roots = []
    for revision in ("v1", "v2"):
        root = tmp_path / revision
        _write_json(
            root / "001_RecordWithName" / "codex_function_bundle.json",
            bundle,
        )
        _write_json(
            root / "002_NoEvidenceTask" / "codex_function_bundle.json",
            bundle,
        )
        legacy_roots.append(root)
    source_index = _source_assets(tmp_path / "source")
    output_root = tmp_path / "converted"

    report = convert_function_assets(
        legacy_roots=legacy_roots,
        source_asset_index=source_index,
        output_root=output_root,
    )

    assert report["task_count"] == 2
    assert report["converted_task_count"] == 1
    assert report["catalogued_task_count"] == 1
    task = report["tasks"]["RecordWithName"]
    assert task["deduplicated_source_count"] == 2
    assert task["status"] == "converted"
    assert (
        report["tasks"]["NoEvidenceTask"]["status"]
        == "catalogued_source_evidence_missing"
    )

    store = FunctionStore(task["store_path"])
    assert store.load_errors == {}
    function = store.get_function("record_with_name")
    assert function is not None
    assert function.input_schema["required"] == ["filename"]
    assert function.bindings == (
        {
            "source": "$.arguments.filename",
            "target": "$.steps[3].action.args.text",
        },
    )
    assert [step.action.tool for step in function.steps] == [
        "open_app",
        "click",
        "click",
        "input_text",
    ]
    assert function.steps[3].action.args["text"] == ""

    store_index = json.loads(
        (output_root / "store_index.json").read_text(encoding="utf-8")
    )
    assert list(store_index) == ["RecordWithName"]
    assert store_index["RecordWithName"]["store_path"] == task["store_path"]
    provenance = json.loads(
        Path(task["provenance_path"]).read_text(encoding="utf-8")
    )
    assert provenance["source_run_log_sha256"] == task["source_run_log_sha256"]
    assert provenance["legacy_bundle_sha256"] == task["legacy_bundle_sha256"]
    assert provenance["target_inputs_read"] is False
    assert provenance["target_observations_read"] is False
    assert provenance["source_target_audit"]["source_target_audit_complete"] is True


def test_conversion_rejects_conflicting_bundles_for_one_task(
    tmp_path: Path,
) -> None:
    roots = []
    for revision, description in (
        ("v1", "First semantic meaning."),
        ("v2", "Different semantic meaning."),
    ):
        root = tmp_path / revision
        bundle = _legacy_bundle()
        bundle["functions"][0]["description"] = description
        _write_json(
            root / "001_RecordWithName" / "codex_function_bundle.json",
            bundle,
        )
        roots.append(root)
    source_index = _source_assets(tmp_path / "source")
    output_root = tmp_path / "converted"

    with pytest.raises(ValueError, match="legacy_function_bundle_conflict"):
        convert_function_assets(
            legacy_roots=roots,
            source_asset_index=source_index,
            output_root=output_root,
        )

    assert not output_root.exists()


def test_conversion_can_freeze_exactly_one_requested_runlog_task(
    tmp_path: Path,
) -> None:
    legacy_root = tmp_path / "legacy"
    for task_name in ("RecordWithName", "NoEvidenceTask"):
        _write_json(
            legacy_root / task_name / "codex_function_bundle.json",
            _legacy_bundle(),
        )

    report = convert_function_assets(
        legacy_roots=(legacy_root,),
        source_asset_index=_source_assets(tmp_path / "source"),
        output_root=tmp_path / "converted",
        task_names=("RecordWithName",),
    )

    assert list(report["tasks"]) == ["RecordWithName"]
    assert report["task_count"] == 1
    assert report["converted_task_count"] == 1


def test_conversion_skips_a_function_store_already_registered_in_memory(
    tmp_path: Path,
) -> None:
    legacy_root = tmp_path / "legacy"
    _write_json(
        legacy_root / "RecordWithName" / "codex_function_bundle.json",
        _legacy_bundle(),
    )

    report = convert_function_assets(
        legacy_roots=(legacy_root,),
        source_asset_index=_source_assets(tmp_path / "source"),
        output_root=tmp_path / "converted",
        exclude_task_names=("RecordWithName",),
    )

    assert report["task_count"] == 0
    assert report["converted_task_count"] == 0
    assert report["excluded_existing_tasks"] == ["RecordWithName"]


def test_conversion_rejects_coordinate_function_without_source_ui(
    tmp_path: Path,
) -> None:
    legacy_root = tmp_path / "legacy"
    _write_json(
        legacy_root / "RecordWithName" / "codex_function_bundle.json",
        _legacy_bundle(),
    )
    source_index = _source_assets(tmp_path / "source")
    index_payload = json.loads(source_index.read_text(encoding="utf-8"))
    source_row = index_payload["RecordWithName"]
    run_log_path = Path(source_row["retained_source_run_log"])
    run_log = json.loads(run_log_path.read_text(encoding="utf-8"))
    run_log["steps"][1]["observation_before_act"].pop("xml")
    _write_json(run_log_path, run_log)
    source_row["retained_source_run_log_sha256"] = _sha256(run_log_path)
    _write_json(source_index, index_payload)

    with pytest.raises(
        ValueError,
        match="transfer_action_source_state_xml_invalid",
    ):
        convert_function_assets(
            legacy_roots=(legacy_root,),
            source_asset_index=source_index,
            output_root=tmp_path / "converted",
        )


def test_conversion_rejects_semantics_authored_from_a_different_runlog(
    tmp_path: Path,
) -> None:
    legacy_root = tmp_path / "legacy"
    bundle = _legacy_bundle()
    bundle["source_run_id"] = "different-source"
    _write_json(
        legacy_root / "RecordWithName" / "codex_function_bundle.json",
        bundle,
    )

    with pytest.raises(
        ValueError,
        match="legacy_function_bundle_source_run_id_mismatch",
    ):
        convert_function_assets(
            legacy_roots=(legacy_root,),
            source_asset_index=_source_assets(tmp_path / "source"),
            output_root=tmp_path / "converted",
        )


def test_conversion_cli_freezes_the_completed_asset_root(
    tmp_path: Path,
) -> None:
    legacy_root = tmp_path / "legacy"
    _write_json(
        legacy_root / "001_RecordWithName" / "codex_function_bundle.json",
        _legacy_bundle(),
    )
    output_root = tmp_path / "converted"
    conversion_source_index = _source_assets(tmp_path / "source")
    memory_root = tmp_path / "memory"
    refresh_artifact_memory(
        memory_root=memory_root,
        source_index=conversion_source_index,
        function_catalogs=(),
        runlog_roots=(tmp_path / "source",),
        result_roots=(),
    )

    assert (
        main(
            [
                "--legacy-root",
                str(legacy_root),
                "--source-asset-index",
                str(conversion_source_index),
                "--output-root",
                str(output_root),
                "--memory-index",
                str(memory_root / "current.json"),
            ]
        )
        == 0
    )
    memory = load_artifact_memory(memory_root / "current.json")
    assert list(memory["canonical"]["function_stores"]) == ["RecordWithName"]

    paths = [output_root, *output_root.rglob("*")]
    assert all(
        not path.stat().st_mode & stat.S_IWUSR
        for path in paths
        if not path.is_symlink()
    )
    for path in paths:
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | stat.S_IWUSR)
