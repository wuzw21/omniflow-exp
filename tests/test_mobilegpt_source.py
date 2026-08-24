from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from PIL import Image
import pytest
from runlog_fixtures import androidworld_run_log, androidworld_state

from src.experiment import run_task as pipeline
from src.experiment import mobilegpt_source
from src.experiment import run_tasks as scheduler
from src.experiment.data_index import (
    canonical_prepared_memory_from_index,
    refresh_data_index,
)
from src.experiment.mobilegpt_contract import (
    MOBILEGPT_AUDIT_SCHEMA,
    MOBILEGPT_LEARNING_MODE,
    MOBILEGPT_MEMORY_MANIFEST,
    MOBILEGPT_MEMORY_SCHEMA,
    MOBILEGPT_SOURCE_METHOD,
    MOBILEGPT_SOURCE_METHOD_BY_SCHEMA,
    MOBILEGPT_SUPPORTED_MEMORY_SCHEMAS,
)
from src.integrations import mobilegpt_memory
from src.integrations.mobilegpt import (
    CONVERSION_MODE_DIRECT,
    convert_runlog_to_mobilegpt_memory,
    validate_memory_manifest,
)


def _write_source_index(
    root: Path,
    *,
    action: dict | None = None,
    forest: str | None = None,
    recorded_seed: int = 111,
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    screenshot = root / "state-0.png"
    Image.new("RGB", (100, 100), color="blue").save(screenshot)
    observation = androidworld_state(
        "state-0",
        package_name="com.android.settings",
        width=100,
        height=100,
        forest=(
            forest
            or '<hierarchy><node index="0" text="Bluetooth" '
            'clickable="true" bounds="[0,0][100,100]" /></hierarchy>'
        ),
    )
    observation["pixels"] = {
        "path": str(screenshot.resolve()),
        "sha256": hashlib.sha256(screenshot.read_bytes()).hexdigest(),
        "width": 100,
        "height": 100,
        "mime_type": "image/png",
    }
    source_run_log = root / "source.run_log.json"
    source_run_log.write_text(
        json.dumps(
            androidworld_run_log(
                [action or {"action_type": "click", "x": 50, "y": 50}],
                observations=[observation],
                task_name="SystemBluetoothTurnOn",
                goal="Turn Bluetooth on.",
                seed=recorded_seed,
            )
        ),
        encoding="utf-8",
    )
    index = root / "index.json"
    index.write_text(
        json.dumps(
            {
                "SystemBluetoothTurnOn": {
                    "goal": "Turn Bluetooth on.",
                    "params": {},
                    "replay_seed": 111,
                    "step_count": 1,
                    "retained_source_run_log": str(source_run_log),
                    "method": "omniflow",
                    "latest_official_success_source": True,
                    "source_kind": "androidworld_validator_success_source_runlog",
                    "source_run_log_sha256": hashlib.sha256(
                        source_run_log.read_bytes()
                    ).hexdigest(),
                }
            }
        ),
        encoding="utf-8",
    )
    return index, source_run_log


def _write_mobilegpt_memory(root: Path, *, include_screenshot: bool = False) -> None:
    app_root = root / "com.android.settings"
    page_root = app_root / "pages" / "0"
    screen_root = page_root / "screen"
    screen_root.mkdir(parents=True)
    (root / "tasks.csv").write_text(
        "name,description,parameters,app\n"
        "toggleBluetooth,Toggle Bluetooth,{},com.android.settings\n",
        encoding="utf-8",
    )
    (app_root / "tasks.csv").write_text(
        'name,path\ntoggleBluetooth,"{""0"": [""toggleBluetooth""]}"\n',
        encoding="utf-8",
    )
    (app_root / "pages.csv").write_text(
        "index,available_subtasks,trigger_uis,extra_uis,screen\n"
        '0,"[]","{}","[]",screen-0\n',
        encoding="utf-8",
    )
    (app_root / "hierarchy.csv").write_text(
        "index,screen,embedding\n0,screen-0,[0.0]\n",
        encoding="utf-8",
    )
    for name in ("available_subtasks.csv", "subtasks.csv"):
        (page_root / name).write_text(
            "name,description,parameters\n"
            "toggleBluetooth,Toggle Bluetooth,{}\n",
            encoding="utf-8",
        )
    (page_root / "actions.csv").write_text(
        "subtask_name,step,action,example\n"
        'toggleBluetooth,0,"{""name"": ""click"", '
        '""parameters"": {""index"": 0}}",{}\n',
        encoding="utf-8",
    )
    for name in ("raw.xml", "html.xml", "hierarchy.xml", "parsed.xml", "pretty.xml"):
        (screen_root / name).write_text("<hierarchy />\n", encoding="utf-8")
    if include_screenshot:
        (screen_root / "screenshot.jpg").write_bytes(b"jpeg")


def _write_stats(path: Path) -> None:
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"event": "task_started"},
                {
                    "event": "embedding_call",
                    "model": "text-embedding-v3",
                    "prompt_tokens": 10,
                    "completion_tokens": 0,
                    "total_tokens": 10,
                },
                {"event": "task_finished"},
            )
        )
        + "\n",
        encoding="utf-8",
    )


def test_mobilegpt_stats_report_memory_utilization(tmp_path: Path) -> None:
    stats = tmp_path / "mobilegpt_stats.jsonl"
    stats.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"event": "memory_lookup", "result": "direct_hit"},
                {"event": "memory_lookup", "result": "explore"},
                {"event": "memory_action_recalled", "action_name": "click"},
                {"event": "mobilegpt_action_sent", "is_device_action": True},
                {"event": "mobilegpt_action_sent", "is_device_action": True},
            )
        )
        + "\n",
        encoding="utf-8",
    )

    summary = mobilegpt_memory.summarize_mobilegpt_stats(stats)

    assert summary["memory_lookup_count"] == 2
    assert summary["memory_hit_count"] == 1
    assert summary["memory_hit_rate"] == 0.5
    assert summary["memory_explore_count"] == 1
    assert summary["memory_action_recalled_count"] == 1
    assert summary["memory_action_use_rate"] == 0.5


def _write_audit(path: Path, *, matched: bool = True) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": MOBILEGPT_AUDIT_SCHEMA,
                "conversion_mode": "runlog_direct",
                "task_name": "SystemBluetoothTurnOn",
                "original_mobilegpt_prompts": False,
                "explore_agent_used": False,
                "select_agent_used": False,
                "derive_agent_fallback_allowed": True,
                "derive_agent_fallback_count": 0,
                "source_example_fallback_count": 0,
                "generalize_action_used": True,
                "direct_subtasks_from_runlog": True,
                "source_direct_hit_validation": True,
                "source_reader_coverage_validation": True,
                "transition_count": 1,
                "validated_transition_count": 1,
                "validation_rows": [
                    {
                        "source_step_index": 0,
                        "matched": matched,
                        "consumed_transitions": 1,
                    }
                ],
                "actions_supplied_to_mobilegpt": True,
                "source_transitions_supplied": True,
                "source_success_boundary_supplied": True,
                "source_success_boundary": {
                    "status": "succeeded",
                    "success": True,
                },
                "official_reader_validation": {
                    "task_path_pages": 1,
                    "page_count": 1,
                    "action_row_count": 2,
                    "source_direct_hit_count": 1,
                    "source_example_fallback_count": 0,
                    "source_reader_coverage_count": 1,
                    "loadable": True,
                },
                "complete": matched,
            }
        ),
        encoding="utf-8",
    )


def test_converted_memory_seals_and_registers(tmp_path: Path) -> None:
    index, source_run_log = _write_source_index(tmp_path / "source")
    registry_root = tmp_path / "registry"
    refresh_data_index(
        memory_root=registry_root,
        source_index=index,
        runlog_roots=(source_run_log.parent,),
        result_roots=(),
    )
    bundle = tmp_path / "bundle"
    memory = bundle / "memory"
    _write_mobilegpt_memory(memory)
    stats = bundle / "source_stats.jsonl"
    audit = bundle / "trajectory_audit.json"
    _write_stats(stats)
    _write_audit(audit)

    sealed = pipeline.seal_mobilegpt_source_memory(
        memory_root=memory,
        source_run_log=source_run_log,
        source_stats=stats,
        trajectory_audit=audit,
        task_name="SystemBluetoothTurnOn",
        target_package="com.android.settings",
        target_app="Settings",
        source_model="qwen3-vl-plus",
    )
    registered = mobilegpt_source._register_mobilegpt_memory(
        memory_index=registry_root / "current.json",
        bundle_root=bundle,
        task_name="SystemBluetoothTurnOn",
    )
    resolved = canonical_prepared_memory_from_index(
        memory_index=registry_root / "current.json",
        task_name="SystemBluetoothTurnOn",
    )
    source_validation = mobilegpt_source.validate_mobilegpt_source_memory(
        index_path=index,
        task_name="SystemBluetoothTurnOn",
        memory_root=memory,
        model="",
    )

    assert sealed["manifest"]["schema_version"] == MOBILEGPT_MEMORY_SCHEMA
    assert sealed["manifest"]["schema_version"] == (
        "omniflow.mobilegpt.memory.v2"
    )
    assert sealed["manifest"]["source_method"] == MOBILEGPT_SOURCE_METHOD
    assert sealed["manifest"]["source_model"] == ""
    assert sealed["manifest"]["source_stats"]["model_calls"] == 1
    assert sealed["manifest"]["source_stats"]["chat_model_calls"] == 0
    assert sealed["manifest"]["source_stats"]["embedding_model_calls"] == 1
    assert sealed["manifest"]["source_stats"]["prompt_tokens"] == 10
    assert sealed["manifest"]["source_stats"]["completion_tokens"] == 0
    assert sealed["manifest"]["source_stats"]["total_tokens"] == 10
    assert sealed["manifest"]["source_stats"]["chat_attempts"] == []
    assert sealed["manifest"]["provenance"]["learning_mode"] == MOBILEGPT_LEARNING_MODE
    assert sealed["manifest"]["provenance"]["native_mobilegpt_learning"] is False
    assert sealed["manifest"]["provenance"]["teacher_forcing"] is False
    assert sealed["manifest"]["provenance"]["original_mobilegpt_prompts"] is False
    assert sealed["manifest"]["provenance"]["semantic_subtasks"] is False
    assert sealed["manifest"]["provenance"]["actions_supplied_to_mobilegpt"] is True
    assert sealed["memory_validation"]["native_memory_complete"] is True
    assert validate_memory_manifest(memory)["task_name"] == (
        "SystemBluetoothTurnOn"
    )
    assert resolved is not None
    assert registered["memory_sha256"] == resolved["memory_sha256"]
    assert source_validation["source_method"] == MOBILEGPT_SOURCE_METHOD


def test_converted_memory_rejects_incomplete_trajectory(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    memory = bundle / "memory"
    _write_mobilegpt_memory(memory)
    _, source_run_log = _write_source_index(tmp_path / "source")
    stats = bundle / "source_stats.jsonl"
    audit = bundle / "trajectory_audit.json"
    _write_stats(stats)
    _write_audit(audit, matched=False)

    with pytest.raises(ValueError, match="mobilegpt_virtual_memory_trajectory_incomplete"):
        pipeline.seal_mobilegpt_source_memory(
            memory_root=memory,
            source_run_log=source_run_log,
            source_stats=stats,
            trajectory_audit=audit,
            task_name="SystemBluetoothTurnOn",
            source_model="qwen3-vl-plus",
        )

    assert not (bundle / MOBILEGPT_MEMORY_MANIFEST).exists()


def test_only_one_mobilegpt_contract_is_active() -> None:
    assert MOBILEGPT_SUPPORTED_MEMORY_SCHEMAS == frozenset(
        {MOBILEGPT_MEMORY_SCHEMA}
    )
    assert MOBILEGPT_SOURCE_METHOD_BY_SCHEMA == {
        MOBILEGPT_MEMORY_SCHEMA: MOBILEGPT_SOURCE_METHOD
    }


def test_converted_memory_ignores_source_model(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    memory = bundle / "memory"
    _write_mobilegpt_memory(memory)
    _, source_run_log = _write_source_index(tmp_path / "source")
    stats = bundle / "source_stats.jsonl"
    audit = bundle / "trajectory_audit.json"
    _write_stats(stats)
    _write_audit(audit)

    sealed = pipeline.seal_mobilegpt_source_memory(
        memory_root=memory,
        source_run_log=source_run_log,
        source_stats=stats,
        trajectory_audit=audit,
        task_name="SystemBluetoothTurnOn",
        source_model="qwen3-vl-plus",
        memory_schema=MOBILEGPT_MEMORY_SCHEMA,
    )

    manifest = sealed["manifest"]
    assert manifest["schema_version"] == MOBILEGPT_MEMORY_SCHEMA
    assert manifest["source_method"] == MOBILEGPT_SOURCE_METHOD
    assert manifest["source_model"] == ""
    assert manifest["source_stats"]["chat_model_calls"] == 0
    assert manifest["source_stats"]["embedding_model_calls"] == 1
    assert manifest["provenance"]["learning_mode"] == (
        MOBILEGPT_LEARNING_MODE
    )
    assert manifest["provenance"]["synthetic_subtasks"] is True
    assert manifest["provenance"]["semantic_subtasks"] is False


def test_source_preflight_is_read_only_and_uses_no_function_store(
    tmp_path: Path,
) -> None:
    index, source_run_log = _write_source_index(tmp_path / "source")
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    result = mobilegpt_source.preflight_mobilegpt_source(
        index_path=index,
        task_name="SystemBluetoothTurnOn",
    )

    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert before == after
    assert Path(result["source_run_log"]) == source_run_log
    assert result["source_method"] == MOBILEGPT_SOURCE_METHOD
    assert result["teacher_forcing"] is False
    assert result["actions_supplied_to_mobilegpt"] is True
    assert result["function_store_used"] is False
    assert result["transition_count"] == 1


def test_source_conversion_calls_only_converter_and_sealer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index, _ = _write_source_index(tmp_path / "source")
    calls: list[str] = []

    def convert(**kwargs: object) -> dict[str, object]:
        calls.append("convert")
        memory = Path(str(kwargs["memory_root"]))
        _write_mobilegpt_memory(memory)
        _write_stats(Path(str(kwargs["stats_path"])))
        _write_audit(Path(str(kwargs["audit_path"])))
        return {"memory_root": str(memory)}

    def seal(**kwargs: object) -> dict[str, object]:
        calls.append("seal")
        return {"memory_root": str(kwargs["memory_root"])}

    monkeypatch.setattr(
        mobilegpt_source,
        "convert_runlog_to_mobilegpt_memory",
        convert,
    )
    monkeypatch.setattr(pipeline, "seal_mobilegpt_source_memory", seal)

    result = mobilegpt_source.prepare_mobilegpt_source_memory(
        index_path=index,
        task_name="SystemBluetoothTurnOn",
        mobilegpt_root=tmp_path / "mobilegpt",
        output_root=tmp_path / "bundle",
        model="qwen3-vl-plus",
    )

    assert calls == ["convert", "seal"]
    assert result["teacher_forcing"] is False
    assert result["actions_supplied_to_mobilegpt"] is True
    assert result["source_emulator_used"] is False


def test_runlog_conversion_official_reader_and_sealer_close_the_loop(
    tmp_path: Path,
) -> None:
    _, source_run_log = _write_source_index(tmp_path / "source")
    bundle = tmp_path / "bundle"
    memory = bundle / "memory"
    stats = bundle / "source_stats.jsonl"
    audit = bundle / "trajectory_audit.json"
    mobilegpt_root = Path(
        os.environ.get(
            "MOBILEGPT_TEST_ROOT",
            "/Users/wuzewen/Projects/Omni/OmniFlow/runtime/external/mobilegpt-official",
        )
    )

    generated = convert_runlog_to_mobilegpt_memory(
        source_run_log=source_run_log,
        mobilegpt_root=mobilegpt_root,
        memory_root=memory,
        stats_path=stats,
        audit_path=audit,
        model="unused-offline",
        embedding_provider=lambda _screen: [0.25, 0.75],
        conversion_mode=CONVERSION_MODE_DIRECT,
    )
    sealed = pipeline.seal_mobilegpt_source_memory(
        memory_root=memory,
        source_run_log=source_run_log,
        source_stats=stats,
        trajectory_audit=audit,
        task_name="SystemBluetoothTurnOn",
        target_package="com.android.settings",
        target_app="com.android.settings",
    )

    assert generated["validated_transition_count"] == 1
    assert generated["official_reader_validation"]["loadable"] is True
    assert sealed["manifest"]["source_method"] == MOBILEGPT_SOURCE_METHOD
    assert validate_memory_manifest(memory)["validated_transition_count"] == 1


def test_scheduler_accepts_only_runlog_aligned_direct_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = tmp_path / "bundle" / "memory"
    memory.mkdir(parents=True)
    (memory.parent / MOBILEGPT_MEMORY_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": MOBILEGPT_MEMORY_SCHEMA,
                "source_method": MOBILEGPT_SOURCE_METHOD,
                "task_name": "SystemBluetoothTurnOn",
                "memory": {"sha256": "digest", "file_count": 3},
                "provenance": {
                    "native_mobilegpt_learning": False,
                    "learning_mode": MOBILEGPT_LEARNING_MODE,
                    "teacher_forcing": False,
                    "actions_supplied_to_mobilegpt": True,
                    "runlog_transition_compilation": True,
                    "complete_transition_mapping": True,
                    "official_reader_validation": True,
                    "source_emulator_used": False,
                },
            }
        ),
        encoding="utf-8",
    )
    inventory = {
        "native_memory_complete": True,
        "has_useful_actions": True,
    }
    monkeypatch.setattr(
        scheduler.mobilegpt_memory_runtime,
        "mobilegpt_memory_digest",
        lambda _root: ("digest", 3),
    )
    monkeypatch.setattr(
        scheduler.mobilegpt_memory_runtime,
        "inspect_mobilegpt_memory",
        lambda _root: inventory,
    )
    monkeypatch.setattr(
        "src.integrations.mobilegpt.validate_mobilegpt_memory",
        lambda _root: {"native_memory_complete": True},
    )

    result = scheduler._validate_prepared_mobilegpt_memory(
        memory,
        task_name="SystemBluetoothTurnOn",
    )

    assert result["memory_sha256"] == "digest"
    assert result["memory_inventory"] == inventory


def test_mobilegpt_source_uses_native_converter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.run_log.json"
    source.write_text(
        json.dumps(
            androidworld_run_log(
                [{"action_type": "click", "x": 50, "y": 50}],
                observations=[
                    androidworld_state(
                        "state-0",
                        forest=(
                            '<hierarchy><node text="Bluetooth" clickable="true" '
                            'bounds="[0,0][100,100]" /></hierarchy>'
                        ),
                        package_name="com.android.settings",
                        width=100,
                        height=100,
                    )
                ],
                task_name="SystemBluetoothTurnOn",
            )
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def native(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"manifest": {"native_format": "mobilegpt.memory"}}

    monkeypatch.setattr(
        mobilegpt_source,
        "convert_runlog_to_mobilegpt_bundle",
        native,
        raising=False,
    )

    result = mobilegpt_source.convert_runlog_to_mobilegpt_bundle(
        source_run_log=source,
        output_root=tmp_path / "bundle",
        mobilegpt_root=tmp_path / "mobilegpt",
        model="qwen3-vl-plus",
        embedding_model="GLM-Embedding-3",
    )

    assert captured["source_run_log"] == source.resolve()
    assert captured["embedding_model"] == "GLM-Embedding-3"
    assert result["manifest"]["native_format"] == "mobilegpt.memory"


def test_source_cli_has_no_teacher_or_cold_learning_commands() -> None:
    parser = mobilegpt_source.build_parser()
    help_text = parser.format_help()

    assert "seal-existing" not in help_text
    assert "teacher" not in help_text.casefold()
    assert "cold" not in help_text.casefold()


def test_strict_reader_validates_canonical_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = tmp_path / "bundle" / "memory"
    memory.mkdir(parents=True)
    (memory.parent / MOBILEGPT_MEMORY_MANIFEST).write_text(
        json.dumps({"schema_version": MOBILEGPT_MEMORY_SCHEMA}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        mobilegpt_memory,
        "_validate_mobilegpt_converted_memory",
        lambda *args, **kwargs: {"schema_version": MOBILEGPT_MEMORY_SCHEMA},
    )
    strict_reader_calls: list[Path] = []

    def validate_strict(root: Path) -> dict[str, bool]:
        strict_reader_calls.append(root)
        return {"native_memory_complete": True}

    monkeypatch.setattr(
        "src.integrations.mobilegpt.validate_mobilegpt_memory",
        validate_strict,
    )

    result = mobilegpt_memory.validate_mobilegpt_adapted_memory(
        memory,
        task_name="SystemBluetoothTurnOn",
        source_seed=111,
        source_run_log=tmp_path / "source.json",
    )

    assert strict_reader_calls == [memory.resolve()]
    assert "memory_validation" in result


def test_runtime_rejects_archived_mobilegpt_schema(tmp_path: Path) -> None:
    memory = tmp_path / "bundle" / "memory"
    memory.mkdir(parents=True)
    (memory.parent / MOBILEGPT_MEMORY_MANIFEST).write_text(
        json.dumps(
            {"schema_version": "omniflow.mobilegpt-runlog-semantic-memory.v1"}
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="mobilegpt_cold_memory_manifest_schema_invalid",
    ):
        mobilegpt_memory.validate_mobilegpt_adapted_memory(
            memory,
            task_name="SystemBluetoothTurnOn",
            source_seed=111,
            source_run_log=tmp_path / "source.json",
        )
