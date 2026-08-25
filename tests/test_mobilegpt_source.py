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
    MOBILEGPT_RUNLOG_LEARNING_MODE,
    MOBILEGPT_RUNLOG_MEMORY_SCHEMA,
    MOBILEGPT_RUNLOG_SOURCE_METHOD,
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


def test_system_ui_only_source_bootstraps_mobilegpt_through_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = pipeline.CanonicalRunLog(
        task="SystemBrightnessMin",
        goal="Turn brightness to the min value.",
        params={"max_or_min": "min"},
        source_run_log=tmp_path / "source.json",
        replay_seed=111,
        step_count=1,
        meta={},
    )
    monkeypatch.setattr(
        pipeline,
        "_infer_mobilegpt_target_from_source_run_log",
        lambda _item: {
            "target_package": "",
            "target_app": "",
            "target_source": "unresolved",
        },
    )
    source = {
        "steps": [
            {
                "observation": {
                    "xml": (
                        '<hierarchy><node package="com.google.android.apps.nexuslauncher" '
                        'bounds="[0,0][720,1280]" /><node package="com.android.systemui" '
                        'class="android.widget.SeekBar" text="Display brightness" '
                        'bounds="[32,64][688,160]" /></hierarchy>'
                    )
                },
                "action": {"action_type": "click", "x": 32, "y": 320},
            }
        ]
    }

    target = mobilegpt_source._mobilegpt_source_target(item=item, source=source)

    assert target == {
        "target_package": "com.android.settings",
        "target_app": "com.android.settings",
        "target_source": "system_ui_source_bootstrap",
    }


def test_launcher_only_source_keeps_launcher_as_mobilegpt_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = pipeline.CanonicalRunLog(
        task="SystemCopyToClipboard",
        goal="Copy text to the clipboard.",
        params={},
        source_run_log=tmp_path / "source.json",
        replay_seed=111,
        step_count=1,
        meta={},
    )
    monkeypatch.setattr(
        pipeline,
        "_infer_mobilegpt_target_from_source_run_log",
        lambda _item: {
            "target_package": "",
            "target_app": "",
            "target_source": "unresolved",
        },
    )
    source = {
        "steps": [
            {
                "observation": {
                    "xml": (
                        '<hierarchy><node package="com.google.android.apps.nexuslauncher" '
                        'class="android.widget.EditText" text="" clickable="true" '
                        'bounds="[24,48][696,160]" /></hierarchy>'
                    )
                },
                "action": {"action_type": "input_text", "text": "example"},
            }
        ]
    }

    target = mobilegpt_source._mobilegpt_source_target(item=item, source=source)

    assert target == {
        "target_package": "com.google.android.apps.nexuslauncher",
        "target_app": "com.google.android.apps.nexuslauncher",
        "target_source": "launcher_source_bootstrap",
    }


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
                "schema_semantics_validation": True,
                "transition_count": 1,
                "validated_transition_count": 1,
                "validation_rows": [
                    {
                        "source_step_index": 0,
                        "matched": matched,
                        "semantic_alignment": matched,
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


def test_converted_memory_remains_read_only_evidence(tmp_path: Path) -> None:
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

    sealed = pipeline.seal_mobilegpt_converted_memory(
        memory_root=memory,
        source_run_log=source_run_log,
        source_stats=stats,
        trajectory_audit=audit,
        task_name="SystemBluetoothTurnOn",
        target_package="com.android.settings",
        target_app="Settings",
        source_model="qwen3-vl-plus",
    )
    source_validation = mobilegpt_memory.validate_mobilegpt_adapted_memory(
        memory,
        task_name="SystemBluetoothTurnOn",
        source_seed=111,
        source_run_log=source_run_log,
        expected_source_method=MOBILEGPT_RUNLOG_SOURCE_METHOD,
    )

    assert sealed["manifest"]["schema_version"] == MOBILEGPT_RUNLOG_MEMORY_SCHEMA
    assert sealed["manifest"]["schema_version"] == (
        "omniflow.mobilegpt.memory.v2"
    )
    assert sealed["manifest"]["source_method"] == MOBILEGPT_RUNLOG_SOURCE_METHOD
    assert sealed["manifest"]["source_model"] == ""
    assert sealed["manifest"]["source_stats"]["model_calls"] == 1
    assert sealed["manifest"]["source_stats"]["chat_model_calls"] == 0
    assert sealed["manifest"]["source_stats"]["embedding_model_calls"] == 1
    assert sealed["manifest"]["source_stats"]["prompt_tokens"] == 10
    assert sealed["manifest"]["source_stats"]["completion_tokens"] == 0
    assert sealed["manifest"]["source_stats"]["total_tokens"] == 10
    assert sealed["manifest"]["source_stats"]["chat_attempts"] == []
    assert sealed["manifest"]["provenance"]["learning_mode"] == MOBILEGPT_RUNLOG_LEARNING_MODE
    assert sealed["manifest"]["provenance"]["native_mobilegpt_learning"] is False
    assert sealed["manifest"]["provenance"]["teacher_forcing"] is False
    assert sealed["manifest"]["provenance"]["original_mobilegpt_prompts"] is False
    assert sealed["manifest"]["provenance"]["semantic_subtasks"] is False
    assert sealed["manifest"]["provenance"]["actions_supplied_to_mobilegpt"] is True
    assert sealed["memory_validation"]["native_memory_complete"] is True
    assert source_validation["manifest"]["task_name"] == "SystemBluetoothTurnOn"
    assert canonical_prepared_memory_from_index(
        memory_index=registry_root / "current.json",
        task_name="SystemBluetoothTurnOn",
    ) is None


def test_native_cold_memory_seals_as_formal_official_source(tmp_path: Path) -> None:
    _, source_run_log = _write_source_index(tmp_path / "source")
    bundle = tmp_path / "bundle"
    memory = bundle / "memory"
    _write_mobilegpt_memory(memory, include_screenshot=True)
    stats = bundle / "source_stats.jsonl"
    _write_stats(stats)
    result = bundle / "task_results.jsonl"
    result.write_text(
        json.dumps(
            {
                "task_name": "SystemBluetoothTurnOn",
                "method": "mobilegpt",
                "official_validator_used": True,
                "official_validator_success": True,
                "action_backend": "mobilegpt_official_accessibility",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    sealed = pipeline.seal_mobilegpt_source_memory(
        memory_root=memory,
        source_run_log=source_run_log,
        source_stats=stats,
        official_source_result=result,
        task_name="SystemBluetoothTurnOn",
        target_package="com.android.settings",
        target_app="Settings",
        source_model="qwen3-vl-plus",
    )
    validated = validate_memory_manifest(memory)

    assert sealed["manifest"]["schema_version"] == MOBILEGPT_MEMORY_SCHEMA
    assert sealed["manifest"]["source_method"] == MOBILEGPT_SOURCE_METHOD
    assert sealed["manifest"]["provenance"]["native_mobilegpt_learning"] is True
    assert sealed["manifest"]["provenance"]["physical_backend"] == "mobilegpt_official_accessibility"
    assert validated["native_mobilegpt_learning"] is True
    assert validated["physical_backend"] == "mobilegpt_official_accessibility"


def test_converted_memory_rejects_stale_target_package(tmp_path: Path) -> None:
    index, source_run_log = _write_source_index(tmp_path / "source")
    bundle = tmp_path / "bundle"
    memory = bundle / "memory"
    _write_mobilegpt_memory(memory)
    stats = bundle / "source_stats.jsonl"
    audit = bundle / "trajectory_audit.json"
    _write_stats(stats)
    _write_audit(audit)
    pipeline.seal_mobilegpt_converted_memory(
        memory_root=memory,
        source_run_log=source_run_log,
        source_stats=stats,
        trajectory_audit=audit,
        task_name="SystemBluetoothTurnOn",
        target_package="com.android.settings",
        target_app="Settings",
        source_model="",
    )
    validated = validate_memory_manifest(memory)
    assert validated["manifest"]["schema_version"] == MOBILEGPT_RUNLOG_MEMORY_SCHEMA
    assert validated["manifest"]["source_method"] == MOBILEGPT_RUNLOG_SOURCE_METHOD
    manifest_path = bundle / MOBILEGPT_MEMORY_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["target_package"] = "android"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="mobilegpt_source_memory_target_package_mismatch",
    ):
        mobilegpt_source.validate_mobilegpt_source_memory(
            index_path=index,
            task_name="SystemBluetoothTurnOn",
            memory_root=memory,
            model="",
        )


def test_registration_replaces_stale_prepared_memory_scan_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "new_bundle"
    calls: list[dict[str, object]] = []

    def fake_refresh(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {
            "canonical": {
                "prepared_memories": {
                    "SystemBluetoothTurnOn": {"memory_sha256": "new"}
                }
            }
        }

    monkeypatch.setattr(
        "src.experiment.data_index.refresh_data_index_from_pointer",
        fake_refresh,
    )

    registered = mobilegpt_source._register_mobilegpt_memory(
        memory_index=tmp_path / "current.json",
        bundle_root=bundle,
        task_name="SystemBluetoothTurnOn",
    )

    assert registered["memory_sha256"] == "new"
    assert calls == [
        {
            "memory_index": tmp_path / "current.json",
            "additional_prepared_memory_roots": (bundle,),
            "replace_prepared_memory_roots": True,
        }
    ]


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
        pipeline.seal_mobilegpt_converted_memory(
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
        {MOBILEGPT_MEMORY_SCHEMA, MOBILEGPT_RUNLOG_MEMORY_SCHEMA}
    )
    assert MOBILEGPT_SOURCE_METHOD_BY_SCHEMA[MOBILEGPT_MEMORY_SCHEMA] == (
        MOBILEGPT_SOURCE_METHOD
    )


def test_converted_memory_ignores_source_model(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    memory = bundle / "memory"
    _write_mobilegpt_memory(memory)
    _, source_run_log = _write_source_index(tmp_path / "source")
    stats = bundle / "source_stats.jsonl"
    audit = bundle / "trajectory_audit.json"
    _write_stats(stats)
    _write_audit(audit)

    sealed = pipeline.seal_mobilegpt_converted_memory(
        memory_root=memory,
        source_run_log=source_run_log,
        source_stats=stats,
        trajectory_audit=audit,
        task_name="SystemBluetoothTurnOn",
        source_model="qwen3-vl-plus",
        memory_schema=MOBILEGPT_RUNLOG_MEMORY_SCHEMA,
    )

    manifest = sealed["manifest"]
    assert manifest["schema_version"] == MOBILEGPT_RUNLOG_MEMORY_SCHEMA
    assert manifest["source_method"] == MOBILEGPT_RUNLOG_SOURCE_METHOD
    assert manifest["source_model"] == ""
    assert manifest["source_stats"]["chat_model_calls"] == 0
    assert manifest["source_stats"]["embedding_model_calls"] == 1
    assert manifest["provenance"]["learning_mode"] == (
        MOBILEGPT_RUNLOG_LEARNING_MODE
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
    assert result["actions_supplied_to_mobilegpt"] is False
    assert result["function_store_used"] is False
    assert result["runlog_conversion_used"] is False
    assert result["source_emulator_required"] is True


def test_source_prepare_runs_native_cold_episode_through_official_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index, _ = _write_source_index(
        tmp_path / "source",
        action={"action_type": "open_app", "app_name": "settings"},
    )
    calls: list[str] = []
    server_arguments: dict[str, object] = {}

    def seal(**kwargs: object) -> dict[str, object]:
        calls.append("seal")
        return {"memory_root": str(kwargs["memory_root"])}

    server = pipeline.CommandSpec(
        label="mobilegpt:official-server",
        argv=["server"],
        env={},
        cwd=tmp_path,
        output_path=None,
        metadata={"log_path": str(tmp_path / "server.log")},
    )
    episode_output = tmp_path / "bundle" / "_source_episode" / "episode"
    episode = pipeline.CommandSpec(
        label="mobilegpt:official-accessibility:source5560",
        argv=["official-client"],
        env={},
        cwd=tmp_path,
        output_path=episode_output,
        metadata={
            "observe_backend": "mobilegpt_official_accessibility",
            "action_backend": "mobilegpt_official_accessibility",
        },
    )
    monkeypatch.setattr(
        pipeline,
        "_resolve_mobilegpt_target_package",
        lambda candidate, **_kwargs: (
            "com.android.settings" if candidate == "settings" else candidate
        ),
    )

    def build_server(*_args: object, **kwargs: object) -> pipeline.CommandSpec:
        server_arguments.update(kwargs)
        return server

    monkeypatch.setattr(pipeline, "build_mobilegpt_server_command", build_server)
    monkeypatch.setattr(pipeline, "_configure_mobilegpt_formal_server", lambda spec, **k: spec)
    monkeypatch.setattr(pipeline, "build_mobilegpt_command", lambda *a, **k: episode)
    monkeypatch.setattr(pipeline, "_start_background_command", lambda *a, **k: (object(), 0))
    monkeypatch.setattr(pipeline, "_stop_background_command", lambda *a, **k: None)

    def run_episode(spec: object) -> int:
        calls.append("native_official_episode")
        episode_output.mkdir(parents=True)
        (episode_output / "task_results.jsonl").write_text("{}\n", encoding="utf-8")
        _write_stats(tmp_path / "bundle" / "source_stats.jsonl")
        return 0

    monkeypatch.setattr(pipeline, "run_command", run_episode)
    monkeypatch.setattr(pipeline, "seal_mobilegpt_source_memory", seal)

    result = mobilegpt_source.prepare_mobilegpt_source_memory(
        index_path=index,
        task_name="SystemBluetoothTurnOn",
        mobilegpt_root=tmp_path / "mobilegpt",
        android_world_root=tmp_path / "android_world",
        output_root=tmp_path / "bundle",
        model="qwen3-vl-plus",
    )

    assert calls == ["native_official_episode", "seal"]
    assert result["teacher_forcing"] is False
    assert result["actions_supplied_to_mobilegpt"] is False
    assert result["source_emulator_used"] is True
    assert result["physical_backend"] == "mobilegpt_official_accessibility"
    assert server_arguments["target_package"] == "com.android.settings"
    assert server_arguments["target_app"] == "settings"


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
    sealed = pipeline.seal_mobilegpt_converted_memory(
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
    assert sealed["manifest"]["source_method"] == MOBILEGPT_RUNLOG_SOURCE_METHOD
    validated = mobilegpt_memory.validate_mobilegpt_adapted_memory(
        memory,
        task_name="SystemBluetoothTurnOn",
        source_seed=111,
        source_run_log=source_run_log,
        expected_source_method=MOBILEGPT_RUNLOG_SOURCE_METHOD,
    )
    assert validated["source_memory_write_status"]["trajectory_validated_transition_count"] == 1


def test_scheduler_rejects_runlog_direct_memory_from_formal_schema(
    tmp_path: Path,
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
    with pytest.raises(ValueError, match="mobilegpt_source_memory_source_evidence_missing"):
        scheduler._validate_prepared_mobilegpt_memory(
            memory,
            task_name="SystemBluetoothTurnOn",
        )


def test_scheduler_rejects_runlog_direct_launch_only_memory(
    tmp_path: Path,
) -> None:
    memory = tmp_path / "bundle" / "memory"
    memory.mkdir(parents=True)
    (memory.parent / MOBILEGPT_MEMORY_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": MOBILEGPT_MEMORY_SCHEMA,
                "source_method": MOBILEGPT_SOURCE_METHOD,
                "task_name": "OpenAppTaskEval",
                "memory": {
                    "sha256": "digest",
                    "file_count": 3,
                    "validation": {"launch_only": True},
                },
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
    with pytest.raises(ValueError, match="mobilegpt_source_memory_source_evidence_missing"):
        scheduler._validate_prepared_mobilegpt_memory(
            memory,
            task_name="OpenAppTaskEval",
        )


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


def test_source_cli_has_one_native_prepare_command() -> None:
    parser = mobilegpt_source.build_parser()
    help_text = parser.format_help()

    assert "seal-existing" not in help_text
    assert "teacher" not in help_text.casefold()
    assert "{prepare,validate,preflight}" in help_text


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
        "_validate_mobilegpt_native_cold_memory",
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
