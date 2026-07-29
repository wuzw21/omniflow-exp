from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import pytest

from omniflow.functions.artifact import bind_function
from omniflow.functions.store import FunctionStore
from src.experiment import function_assets
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
        {
            "run_id": "human-source",
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


def _semantic_response(_prompt: str) -> str:
    return json.dumps(
        {
            "name": "Record audio with a chosen filename",
            "description": (
                "Open the recorder, start recording, and save the audio using "
                "the filename supplied by the user."
            ),
            "parameters": [
                {
                    "name": "filename",
                    "description": "Exact filename requested by the user.",
                    "step_index": 3,
                    "arg_name": "text",
                }
            ],
            "checker_rules": [],
        }
    )


def test_human_runlog_calls_existing_semantic_function_once_and_preserves_actions(
    tmp_path: Path,
) -> None:
    prompts: list[str] = []

    def complete_json(prompt: str) -> str:
        prompts.append(prompt)
        return _semantic_response(prompt)

    source_index = _source_assets(tmp_path / "source")
    report = convert_function_assets(
        source_asset_index=source_index,
        output_root=tmp_path / "converted",
        complete_json=complete_json,
        model="qwen3-vl-plus",
    )

    assert len(prompts) == 1
    assert "Record and save audio as source_name.m4a." in prompts[0]
    assert '"tool":"input_text"' in prompts[0]
    assert report["task_count"] == 1
    assert report["converted_task_count"] == 1

    task = report["tasks"]["RecordWithName"]
    store = FunctionStore(task["store_path"])
    assert store.load_errors == {}
    functions = store.list_functions()
    assert len(functions) == 1
    function = functions[0]
    assert function.name == "Record audio with a chosen filename"
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
    assert provenance["semantic_collection"]["function"] == "enhance_function"
    assert provenance["semantic_collection"]["model_calls"] == 1
    assert provenance["semantic_collection"]["model"] == "qwen3-vl-plus"
    assert provenance["target_inputs_read"] is False
    assert provenance["target_observations_read"] is False
    assert provenance["source_target_audit"]["source_target_audit_complete"] is True


def test_semantic_function_failure_is_not_retried(tmp_path: Path) -> None:
    calls = 0

    def fail(_prompt: str) -> str:
        nonlocal calls
        calls += 1
        raise TimeoutError("model timed out")

    with pytest.raises(TimeoutError, match="model timed out"):
        convert_function_assets(
            source_asset_index=_source_assets(tmp_path / "source"),
            output_root=tmp_path / "converted",
            complete_json=fail,
            model="qwen3-vl-plus",
        )

    assert calls == 1


def test_model_adapter_disables_sdk_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    for name in (
        "OPENAI_API_KEY",
        "DASHSCOPE_API_KEY",
        "OPENAI_BASE_URL",
        "OMNIFLOW_OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)

    class FakeCompletions:
        def create(self, **kwargs: object) -> object:
            captured["request"] = kwargs
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"parameters":[]}')
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=20,
                    completion_tokens=5,
                    total_tokens=25,
                ),
            )

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=FakeOpenAI),
    )
    complete_json = function_assets._build_complete_json(
        model="qwen3-vl-plus",
        timeout=60,
    )

    assert complete_json("prompt") == '{"parameters":[]}'
    assert captured["client"] == {
        "api_key": "not-required",
        "max_retries": 0,
        "timeout": 60.0,
    }
    request = captured["request"]
    assert isinstance(request, dict)
    assert request["model"] == "qwen3-vl-plus"
    assert request["max_tokens"] == 1800
    assert request["timeout"] == 60.0
    assert complete_json.last_usage == {
        "prompt_tokens": 20,
        "completion_tokens": 5,
        "total_tokens": 25,
    }


def test_conversion_can_select_tasks_and_skip_registered_tasks(
    tmp_path: Path,
) -> None:
    source_index = _source_assets(tmp_path / "source")

    selected = convert_function_assets(
        source_asset_index=source_index,
        output_root=tmp_path / "selected",
        complete_json=_semantic_response,
        model="qwen3-vl-plus",
        task_names=("RecordWithName",),
    )
    assert list(selected["tasks"]) == ["RecordWithName"]

    calls = 0

    def should_not_run(_prompt: str) -> str:
        nonlocal calls
        calls += 1
        return "{}"

    skipped = convert_function_assets(
        source_asset_index=source_index,
        output_root=tmp_path / "skipped",
        complete_json=should_not_run,
        model="qwen3-vl-plus",
        exclude_task_names=("RecordWithName",),
    )
    assert skipped["task_count"] == 0
    assert skipped["excluded_existing_tasks"] == ["RecordWithName"]
    assert calls == 0


def test_conversion_rejects_coordinate_function_without_source_ui(
    tmp_path: Path,
) -> None:
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
            source_asset_index=source_index,
            output_root=tmp_path / "converted",
            complete_json=_semantic_response,
            model="qwen3-vl-plus",
        )


def test_conversion_cli_freezes_and_registers_completed_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    monkeypatch.setattr(
        "src.experiment.function_assets._build_complete_json",
        lambda **_kwargs: _semantic_response,
    )

    assert (
        main(
            [
                "--source-asset-index",
                str(source_index),
                "--output-root",
                str(output_root),
                "--memory-index",
                str(memory_root / "current.json"),
                "--model",
                "qwen3-vl-plus",
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
