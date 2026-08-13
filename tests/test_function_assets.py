from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import stat

import pytest
from runlog_fixtures import androidworld_run_log, androidworld_state

from omniflow.functions.artifact import bind_function
from omniflow.functions.authoring import (
    FUNCTION_AUTHORING_INSTRUCTIONS_VERSION,
    function_authoring_instructions_sha256,
)
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
        androidworld_run_log(
            [
                {
                    "action_type": "open_app",
                    "app_name": "com.example.recorder",
                },
                {"action_type": "click", "x": 500, "y": 800},
                {"action_type": "click", "x": 500, "y": 200},
                {
                    "action_type": "input_text",
                    "text": "source_name.m4a",
                },
            ],
            observations=[
                androidworld_state(
                    "state-open",
                    package_name="com.android.launcher",
                    with_pixels=True,
                ),
                androidworld_state(
                    "state-menu",
                    forest=menu_xml,
                    package_name="com.example.recorder",
                    with_pixels=True,
                ),
                androidworld_state(
                    "state-input",
                    forest=input_xml,
                    package_name="com.example.recorder",
                    with_pixels=True,
                ),
                androidworld_state(
                    "state-input",
                    forest=input_xml,
                    package_name="com.example.recorder",
                    with_pixels=True,
                ),
            ],
            task_name="RecordWithName",
            run_id="human-source",
            goal="Record and save audio as source_name.m4a.",
        ),
    )
    return _write_json(
        root / "index_by_task.json",
        {
            "RecordWithName": {
                "task": "RecordWithName",
                "collect_seed": 111,
                "source_seed": 111,
                "latest_official_success_source": True,
                "retained_source_run_log": str(run_log),
                "retained_source_run_log_sha256": _sha256(run_log),
            }
        },
    )


def _authoring_manifest(root: Path, source_index: Path) -> Path:
    source_row = json.loads(source_index.read_text(encoding="utf-8"))[
        "RecordWithName"
    ]
    return _write_json(
        root / "authoring_manifest.json",
        {
            "schema_version": "omniflow.function-agent-authoring-manifest.v2",
            "source_asset_index_sha256": _sha256(source_index),
            "agent": {
                "kind": "offline_agent",
                "instructions_version": FUNCTION_AUTHORING_INSTRUCTIONS_VERSION,
                "instructions_sha256": function_authoring_instructions_sha256(),
            },
            "tasks": {
                "RecordWithName": {
                    "source_run_log_sha256": source_row[
                        "retained_source_run_log_sha256"
                    ],
                    "author_response": {
                        "reason": (
                            "Steps 0-3 form one recorded audio workflow. The Agent "
                            "parameterized only the text entered in the visible File "
                            "name field and kept the recorder controls fixed."
                        ),
                        "bundle": {
                            "schema_version": "omniflow.function-bundle.v2",
                            "run_id": "human-source",
                            "arguments": {
                                "record_audio_with_filename": {
                                    "filename": "source_name.m4a"
                                }
                            },
                            "functions": [
                                {
                                    "schema_version": "omniflow.function.v2",
                                    "function_id": "record_audio_with_filename",
                                    "name": "Record one audio clip with a filename",
                                    "description": (
                                        "Open the recorded audio recorder, use its "
                                        "record controls, and enter one caller-supplied "
                                        "filename in the visible File name field. The "
                                        "recording controls and save flow remain fixed; "
                                        "this Function does not choose another recorder, "
                                        "repeat clips, or verify the final task result."
                                    ),
                                    "input_schema": {
                                        "type": "object",
                                        "properties": {
                                            "filename": {
                                                "type": "string",
                                                "description": (
                                                    "Exact text to enter in the visible "
                                                    "File name field."
                                                ),
                                            }
                                        },
                                        "required": ["filename"],
                                        "additionalProperties": False,
                                    },
                                    "bindings": [
                                        {
                                            "source": "$.arguments.filename",
                                            "target": "$.steps[3].action.args.text",
                                        }
                                    ],
                                    "steps": [
                                        {
                                            "step_index": 0,
                                            "source_state_id": "state-open",
                                            "action": {
                                                "tool": "open_app",
                                                "args": {
                                                    "package_name": "com.example.recorder"
                                                },
                                            },
                                        },
                                        {
                                            "step_index": 1,
                                            "source_state_id": "state-menu",
                                            "action": {
                                                "tool": "click",
                                                "args": {"x": 500, "y": 800},
                                            },
                                        },
                                        {
                                            "step_index": 2,
                                            "source_state_id": "state-input",
                                            "action": {
                                                "tool": "click",
                                                "args": {"x": 500, "y": 200},
                                            },
                                        },
                                        {
                                            "step_index": 3,
                                            "source_state_id": "state-input",
                                            "action": {
                                                "tool": "input_text",
                                                "args": {"text": ""},
                                            },
                                        },
                                    ],
                                    "checker_rules": [],
                                    "agent_visible": True,
                                }
                            ],
                        },
                    },
                }
            },
        },
    )


def test_human_runlog_uses_offline_manifest_and_preserves_actions(
    tmp_path: Path,
) -> None:
    source_index = _source_assets(tmp_path / "source")
    authoring_manifest = _authoring_manifest(tmp_path, source_index)
    report = convert_function_assets(
        source_asset_index=source_index,
        authoring_manifest=authoring_manifest,
        output_root=tmp_path / "converted",
    )

    assert set(inspect.signature(convert_function_assets).parameters) == {
        "source_asset_index",
        "authoring_manifest",
        "output_root",
        "task_names",
        "exclude_task_names",
    }
    assert report["task_count"] == 1
    assert report["converted_task_count"] == 1
    assert report["authoring_manifest"] == str(authoring_manifest.resolve())
    assert report["authoring_manifest_sha256"] == _sha256(authoring_manifest)

    task = report["tasks"]["RecordWithName"]
    store = FunctionStore(task["store_path"])
    assert store.load_errors == {}
    functions = store.list_functions()
    assert len(functions) == 1
    function = functions[0]
    assert function.name == "Record one audio clip with a filename"
    assert function.input_schema["required"] == ["filename"]
    assert function.bindings == (
        {
            "source": "$.arguments.filename",
            "target": "$.steps[3].action.args.text",
        },
    )
    bound = bind_function(function, {"filename": "source_name.m4a"})
    assert [step.action.to_dict() for step in bound.steps] == [
        {
            "tool": "open_app",
            "args": {"package_name": "com.example.recorder"},
        },
        {"tool": "click", "args": {"x": 500, "y": 800}},
        {"tool": "click", "args": {"x": 500, "y": 200}},
        {"tool": "input_text", "args": {"text": "source_name.m4a"}},
    ]

    provenance = json.loads(
        Path(task["provenance_path"]).read_text(encoding="utf-8")
    )
    assert provenance["source_run_id"] == "human-source"
    assert provenance["semantic_collection"] == {
        "function": "offline_agent_function_authoring",
        "manifest_path": str(authoring_manifest.resolve()),
        "manifest_sha256": _sha256(authoring_manifest),
        "agent": {
            "kind": "offline_agent",
            "instructions_version": FUNCTION_AUTHORING_INSTRUCTIONS_VERSION,
            "instructions_sha256": function_authoring_instructions_sha256(),
        },
        "reason": (
            "Steps 0-3 form one recorded audio workflow. The Agent parameterized "
            "only the text entered in the visible File name field and kept the "
            "recorder controls fixed."
        ),
        "model": None,
        "model_calls": 0,
    }
    assert provenance["target_inputs_read"] is False
    assert provenance["target_observations_read"] is False
    assert provenance["source_target_audit"]["source_target_audit_complete"] is True


def test_conversion_rejects_manifest_for_different_source_index(
    tmp_path: Path,
) -> None:
    source_index = _source_assets(tmp_path / "source")
    manifest = _authoring_manifest(tmp_path, source_index)
    payload = json.loads(source_index.read_text(encoding="utf-8"))
    payload["extra"] = {}
    _write_json(source_index, payload)

    with pytest.raises(
        ValueError,
        match="function_authoring_source_index_hash_mismatch",
    ):
        convert_function_assets(
            source_asset_index=source_index,
            authoring_manifest=manifest,
            output_root=tmp_path / "converted",
        )


def test_conversion_can_select_tasks_and_skip_registered_tasks(
    tmp_path: Path,
) -> None:
    source_index = _source_assets(tmp_path / "source")
    authoring_manifest = _authoring_manifest(tmp_path, source_index)

    selected = convert_function_assets(
        source_asset_index=source_index,
        authoring_manifest=authoring_manifest,
        output_root=tmp_path / "selected",
        task_names=("RecordWithName",),
    )
    assert list(selected["tasks"]) == ["RecordWithName"]

    skipped = convert_function_assets(
        source_asset_index=source_index,
        authoring_manifest=authoring_manifest,
        output_root=tmp_path / "skipped",
        exclude_task_names=("RecordWithName",),
    )
    assert skipped["task_count"] == 0
    assert skipped["excluded_existing_tasks"] == ["RecordWithName"]


def test_conversion_rejects_coordinate_function_without_source_ui(
    tmp_path: Path,
) -> None:
    source_index = _source_assets(tmp_path / "source")
    index_payload = json.loads(source_index.read_text(encoding="utf-8"))
    source_row = index_payload["RecordWithName"]
    run_log_path = Path(source_row["retained_source_run_log"])
    run_log = json.loads(run_log_path.read_text(encoding="utf-8"))
    run_log["steps"][1]["observation"]["forest"] = None
    _write_json(run_log_path, run_log)
    source_row["retained_source_run_log_sha256"] = _sha256(run_log_path)
    _write_json(source_index, index_payload)

    with pytest.raises(
        ValueError,
        match="transfer_action_source_state_xml_invalid",
    ):
        convert_function_assets(
            source_asset_index=source_index,
            authoring_manifest=_authoring_manifest(tmp_path, source_index),
            output_root=tmp_path / "converted",
        )


def test_conversion_rejects_legacy_run_log_at_formal_boundary(
    tmp_path: Path,
) -> None:
    source_run_log = _write_json(
        tmp_path / "source" / "SystemCopyToClipboard" / "source.run_log.json",
        {
            "run_id": "clipboard-source",
            "goal": "Copy the requested text to the clipboard.",
            "success": True,
            "steps": [
                {
                    "observation_before_act": {
                        "state_id": "clipboard-home",
                        "package_name": "com.android.launcher",
                    },
                    "executed_actions": [
                        {
                            "type": "open_app",
                            "params": {"package_name": "ca.zgrs.clipper"},
                        }
                    ],
                    "success": True,
                },
                {
                    "observation_before_act": {
                        "state_id": "clipboard-app",
                        "package_name": "ca.zgrs.clipper",
                    },
                    "executed_actions": [
                        {
                            "type": "set_clipboard",
                            "params": {"text": "9876 Pine Ave"},
                        }
                    ],
                    "success": True,
                },
            ],
        },
    )
    source_index = _write_json(
        tmp_path / "source" / "index.json",
        {
            "SystemCopyToClipboard": {
                "retained_source_run_log": str(source_run_log),
                "retained_source_run_log_sha256": _sha256(source_run_log),
            }
        },
    )
    authoring_manifest = _write_json(
        tmp_path / "authoring.json",
        {
            "schema_version": "omniflow.function-agent-authoring-manifest.v2",
            "source_asset_index_sha256": _sha256(source_index),
            "agent": {
                "kind": "offline_agent",
                "instructions_version": FUNCTION_AUTHORING_INSTRUCTIONS_VERSION,
                "instructions_sha256": function_authoring_instructions_sha256(),
            },
            "tasks": {
                "SystemCopyToClipboard": {
                    "source_run_log_sha256": _sha256(source_run_log),
                    "author_response": {
                        "reason": "Keep the recorded open-app action.",
                        "bundle": {
                            "schema_version": "omniflow.function-bundle.v2",
                            "run_id": "legacy-source",
                            "arguments": {},
                            "functions": [],
                        },
                    },
                }
            },
        },
    )

    with pytest.raises(ValueError, match="run_log_schema_invalid"):
        convert_function_assets(
            source_asset_index=source_index,
            authoring_manifest=authoring_manifest,
            output_root=tmp_path / "converted",
        )


def test_conversion_cli_freezes_and_registers_completed_assets(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "converted"
    source_index = _source_assets(tmp_path / "source")
    memory_root = tmp_path / "memory"
    refresh_artifact_memory(
        memory_root=memory_root,
        source_index=source_index,
        function_catalogs=(),
        runlog_roots=(tmp_path / "source",),
        result_roots=(),
    )
    authoring_manifest = _authoring_manifest(tmp_path, source_index)

    assert (
        main(
            [
                "--source-asset-index",
                str(source_index),
                "--authoring-manifest",
                str(authoring_manifest),
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
