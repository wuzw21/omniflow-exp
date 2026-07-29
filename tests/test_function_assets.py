from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat

import pytest

from omniflow.functions.store import FunctionStore
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
    run_log = _write_json(
        root / "paired.source.run_log.json",
        {
            "schema_version": "omniflow.canonical_run_log.v1",
            "run_id": "current-source",
            "goal": "Record and save audio as source_name.m4a.",
            "status": "succeeded",
            "success": True,
            "steps": [
                {
                    "step_index": 0,
                    "before_state_id": "state-open",
                    "action": {
                        "tool": "open_app",
                        "args": {"package_name": "com.example.recorder"},
                    },
                    "result": {"success": True},
                    "after_state_id": "state-menu",
                },
                {
                    "step_index": 1,
                    "before_state_id": "state-menu",
                    "action": {"tool": "click", "args": {"x": 500, "y": 800}},
                    "result": {"success": True},
                    "after_state_id": "state-input",
                },
                {
                    "step_index": 2,
                    "before_state_id": "state-input",
                    "action": {"tool": "click", "args": {"x": 500, "y": 200}},
                    "result": {"success": True},
                    "after_state_id": "state-input",
                },
                {
                    "step_index": 3,
                    "before_state_id": "state-input",
                    "action": {
                        "tool": "input_text",
                        "args": {"text": "source_name.m4a"},
                    },
                    "result": {"success": True},
                    "after_state_id": "state-done",
                },
            ],
        },
    )
    indexed_payload = json.loads(run_log.read_text(encoding="utf-8"))
    indexed_payload["run_id"] = "indexed-source"
    for step in indexed_payload["steps"]:
        step["before_state_id"] = "old-" + step["before_state_id"]
        step["after_state_id"] = "old-" + step["after_state_id"]
    indexed_run_log = _write_json(
        root / "indexed.source.run_log.json",
        indexed_payload,
    )
    states = _write_json(
        root / "transfer_states.json",
        {
            "schema_version": "omniflow.transfer-state-catalog.v1",
            "run_id": "source-replay-wrapper",
            "states": {
                state_id: {"state_id": state_id}
                for state_id in ("state-open", "state-menu", "state-input")
            },
        },
    )
    provenance = _write_json(
        root / "provenance_manifest.json",
        {
            "schema_version": "test.provenance.v1",
            "source_run_log": str(indexed_run_log),
            "source_run_log_sha256": _sha256(indexed_run_log),
            "output_source_run_log": str(run_log),
            "output_source_run_log_sha256": _sha256(run_log),
        },
    )
    return _write_json(
        root / "asset_index.json",
        {
            "assets": {
                "RecordWithName": {
                    "source_run_log": str(indexed_run_log),
                    "source_run_log_sha256": _sha256(indexed_run_log),
                    "transfer_state_catalog": str(states),
                    "transfer_state_catalog_sha256": _sha256(states),
                    "provenance_manifest": str(provenance),
                    "provenance_manifest_sha256": _sha256(provenance),
                }
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


def test_conversion_cli_freezes_the_completed_asset_root(
    tmp_path: Path,
) -> None:
    legacy_root = tmp_path / "legacy"
    _write_json(
        legacy_root / "001_RecordWithName" / "codex_function_bundle.json",
        _legacy_bundle(),
    )
    output_root = tmp_path / "converted"

    assert (
        main(
            [
                "--legacy-root",
                str(legacy_root),
                "--source-asset-index",
                str(_source_assets(tmp_path / "source")),
                "--output-root",
                str(output_root),
            ]
        )
        == 0
    )

    paths = [output_root, *output_root.rglob("*")]
    assert all(
        not path.stat().st_mode & stat.S_IWUSR
        for path in paths
        if not path.is_symlink()
    )
    for path in paths:
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | stat.S_IWUSR)
