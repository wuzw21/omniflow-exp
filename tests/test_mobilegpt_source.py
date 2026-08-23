from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
from runlog_fixtures import androidworld_run_log, androidworld_state

from src.experiment import mobilegpt_source
from src.experiment import run_task as pipeline
from src.experiment.mobilegpt_contract import (
    MOBILEGPT_LEARNING_MODE,
    MOBILEGPT_MEMORY_MANIFEST,
    MOBILEGPT_MEMORY_SCHEMA,
    MOBILEGPT_SOURCE_METHOD,
    MOBILEGPT_SOURCE_METHOD_BY_SCHEMA,
    MOBILEGPT_SUPPORTED_MEMORY_SCHEMAS,
)
from src.experiment.data_index import refresh_data_index
from src.integrations import mobilegpt_memory


def _write_source_index(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    screenshot = root / "state-0.png"
    Image.new("RGB", (100, 100), color="blue").save(screenshot)
    observation = androidworld_state(
        "state-0",
        package_name="com.android.settings",
        width=100,
        height=100,
        forest=(
            '<hierarchy><node index="0" text="Bluetooth" '
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
                [{"action_type": "click", "x": 50, "y": 50}],
                observations=[observation],
                task_name="SystemBluetoothTurnOn",
                goal="Turn Bluetooth on.",
                seed=111,
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
                    "method": "source",
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


def _write_mobilegpt_memory(root: Path) -> None:
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
    (screen_root / "screenshot.jpg").write_bytes(b"jpeg")


def _write_native_stats(path: Path) -> None:
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "event": "task_started",
                    "task_name": "SystemBluetoothTurnOn",
                    "instruction": "Turn Bluetooth on.",
                },
                {
                    "event": "chat_call",
                    "model": "GLM-4.6V",
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
                {
                    "event": "embedding_call",
                    "model": "GLM-Embedding-2",
                    "prompt_tokens": 4,
                    "completion_tokens": 0,
                    "total_tokens": 4,
                },
                {
                    "event": "task_finished",
                    "task_name": "SystemBluetoothTurnOn",
                    "instruction": "Turn Bluetooth on.",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )


def _write_official_result(path: Path, *, success: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "task_name": "SystemBluetoothTurnOn",
                "official_validator_used": True,
                "official_validator_success": success,
                "success": success,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_mobilegpt_source_target_ignores_permission_controller_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline,
        "_infer_mobilegpt_target_from_source_run_log",
        lambda _item: {},
    )
    target = mobilegpt_source._mobilegpt_source_target(
        item=SimpleNamespace(task="CameraTakePhoto"),
        source={
            "steps": [
                {
                    "observation": {
                        "package_name": "com.android.camera2",
                        "forest": {
                            "package_name": "com.google.android.permissioncontroller"
                        },
                    }
                }
            ]
        },
    )

    assert target["target_package"] == "com.android.camera2"
    assert target["target_source"] == "canonical_source_observation"


def test_mobilegpt_observation_package_ignores_systemui_only_xml() -> None:
    assert pipeline._mobilegpt_observation_package(
        {
            "xml": (
                '<hierarchy><node package="com.android.systemui" '
                'text="MODE LIST" /></hierarchy>'
            )
        }
    ) == ""


def test_mobilegpt_source_target_preserves_open_app_alias_for_device_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline,
        "_infer_mobilegpt_target_from_source_run_log",
        lambda _item: {"target_package": "settings", "target_app": "settings"},
    )

    target = mobilegpt_source._mobilegpt_source_target(
        item=SimpleNamespace(task="SystemBluetoothTurnOn"),
        source={"steps": []},
    )

    assert target == {
        "target_package": "settings",
        "target_app": "settings",
        "target_source": "canonical_source_open_app_alias",
    }


def test_only_one_mobilegpt_contract_is_active() -> None:
    assert MOBILEGPT_MEMORY_SCHEMA == "omniflow.mobilegpt-native-cold-memory.v1"
    assert MOBILEGPT_SOURCE_METHOD == "mobilegpt_native_source_cold"
    assert MOBILEGPT_LEARNING_MODE == "mobilegpt_native_cold"
    assert MOBILEGPT_SUPPORTED_MEMORY_SCHEMAS == frozenset(
        {MOBILEGPT_MEMORY_SCHEMA}
    )
    assert MOBILEGPT_SOURCE_METHOD_BY_SCHEMA[MOBILEGPT_MEMORY_SCHEMA] == (
        MOBILEGPT_SOURCE_METHOD
    )


def test_source_preflight_is_read_only_and_does_not_supply_runlog_actions(
    tmp_path: Path,
) -> None:
    index, source_run_log = _write_source_index(tmp_path / "source")
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    result = mobilegpt_source.preflight_mobilegpt_source(
        index_path=index,
        task_name="SystemBluetoothTurnOn",
    )

    assert before == sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert Path(result["source_run_log"]) == source_run_log
    assert result["source_method"] == MOBILEGPT_SOURCE_METHOD
    assert result["teacher_forcing"] is False
    assert result["actions_supplied_to_mobilegpt"] is False
    assert result["runlog_conversion_used"] is False
    assert result["source_emulator_required"] is True


def test_native_cold_memory_seals_only_official_source_success(
    tmp_path: Path,
) -> None:
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
    _write_native_stats(stats)
    result_path = bundle / "source_result.jsonl"
    _write_official_result(result_path, success=True)

    sealed = pipeline.seal_mobilegpt_source_memory(
        memory_root=memory,
        source_run_log=source_run_log,
        source_stats=stats,
        official_source_result=result_path,
        task_name="SystemBluetoothTurnOn",
        target_package="com.android.settings",
        target_app="Settings",
        source_model="GLM-4.6V",
    )

    manifest = sealed["manifest"]
    assert manifest["schema_version"] == MOBILEGPT_MEMORY_SCHEMA
    assert manifest["source_method"] == MOBILEGPT_SOURCE_METHOD
    assert manifest["official_source_result"]["official_validator_success"] is True
    assert manifest["provenance"]["source_emulator_used"] is True
    assert manifest["provenance"]["teacher_forcing"] is False
    assert "trajectory_audit" not in manifest
    assert "teacher_source" not in manifest
    assert sealed["memory_validation"]["native_memory_complete"] is True
    registered = mobilegpt_source._register_mobilegpt_memory(
        memory_index=registry_root / "current.json",
        bundle_root=bundle,
        task_name="SystemBluetoothTurnOn",
    )
    assert registered["schema_version"] == MOBILEGPT_MEMORY_SCHEMA
    assert registered["source_method"] == MOBILEGPT_SOURCE_METHOD


def test_native_cold_memory_rejects_failed_official_validator(
    tmp_path: Path,
) -> None:
    _, source_run_log = _write_source_index(tmp_path / "source")
    bundle = tmp_path / "bundle"
    memory = bundle / "memory"
    _write_mobilegpt_memory(memory)
    stats = bundle / "source_stats.jsonl"
    _write_native_stats(stats)
    result_path = bundle / "source_result.jsonl"
    _write_official_result(result_path, success=False)

    with pytest.raises(ValueError, match="official_source_failed"):
        pipeline.seal_mobilegpt_source_memory(
            memory_root=memory,
            source_run_log=source_run_log,
            source_stats=stats,
            official_source_result=result_path,
            task_name="SystemBluetoothTurnOn",
            source_model="GLM-4.6V",
        )

    assert not (bundle / MOBILEGPT_MEMORY_MANIFEST).exists()


def test_mobilegpt_package_resolution_gives_adb_wrapper_real_sdk_binary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk_root = tmp_path / "android-sdk"
    real_adb = sdk_root / "platform-tools" / "adb"
    real_adb.parent.mkdir(parents=True)
    real_adb.write_text("#!/bin/sh\n", encoding="utf-8")
    real_adb.chmod(0o755)
    monkeypatch.setattr(
        pipeline,
        "_subprocess_env",
        lambda *_args, **_kwargs: {"ANDROID_SDK_ROOT": str(sdk_root)},
    )
    monkeypatch.setattr(pipeline, "resolve_androidworld_package", lambda _value: "")
    captured: dict[str, object] = {}

    def run(*args: object, **kwargs: object) -> SimpleNamespace:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return SimpleNamespace(stdout="package:com.android.settings\n", returncode=0)

    monkeypatch.setattr(pipeline.subprocess, "run", run)

    resolved = pipeline._resolve_mobilegpt_target_package(
        "settings",
        adb_path="/repo/tools/androidworld_adb_compat.sh",
        serial="emulator-5560",
    )

    assert resolved == "com.android.settings"
    env = captured["kwargs"]["env"]
    assert env["OMNIFLOW_REAL_ADB_PATH"] == str(real_adb)


def test_mobilegpt_package_resolution_uses_official_androidworld_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline,
        "resolve_androidworld_package",
        lambda value: "com.android.settings" if value == "settings" else "",
    )
    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("official mapping must win"),
    )

    assert pipeline._resolve_mobilegpt_target_package(
        "settings",
        adb_path="/repo/tools/androidworld_adb_compat.sh",
        serial="emulator-5560",
        android_world_root="/official/android_world",
    ) == "com.android.settings"


def test_prepare_runs_original_mobilegpt_on_source_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index, _ = _write_source_index(tmp_path / "source")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        pipeline,
        "_infer_mobilegpt_target_from_source_run_log",
        lambda _item: {
            "target_package": "settings",
            "target_app": "settings",
        },
    )
    package_resolution: list[tuple[str, str]] = []
    monkeypatch.setattr(
        pipeline,
        "_resolve_mobilegpt_target_package",
        lambda candidate, *, adb_path, serial, android_world_root: package_resolution.append(
            (candidate, serial)
        )
        or "com.android.settings",
    )

    def build_server(action: str, **kwargs: object) -> pipeline.CommandSpec:
        captured["server_action"] = action
        captured["server_kwargs"] = kwargs
        return pipeline.CommandSpec(
            label="mobilegpt-server",
            argv=["python", "server.py"],
            env={},
            cwd=tmp_path,
            metadata={"log_path": str(tmp_path / "server.log")},
        )

    monkeypatch.setattr(pipeline, "build_mobilegpt_server_command", build_server)
    monkeypatch.setattr(
        pipeline,
        "_start_mobilegpt_browser_task_server",
        lambda **_kwargs: ({}, None),
    )
    episode_output = tmp_path / "episode"

    def build_episode(*_args: object, **kwargs: object) -> pipeline.CommandSpec:
        captured["episode_kwargs"] = kwargs
        return pipeline.CommandSpec(
            label="mobilegpt-source",
            argv=["python", "episode.py"],
            env={},
            cwd=tmp_path,
            output_path=episode_output,
            metadata={},
        )

    monkeypatch.setattr(pipeline, "build_mobilegpt_command", build_episode)
    monkeypatch.setattr(
        pipeline,
        "_start_background_command",
        lambda _spec, **_kwargs: (object(), 0),
    )
    monkeypatch.setattr(pipeline, "_stop_background_command", lambda _process: None)

    def run_episode(_spec: pipeline.CommandSpec) -> int:
        _write_official_result(episode_output / "task_results.jsonl", success=True)
        _write_native_stats(tmp_path / "bundle" / "source_stats.jsonl")
        return 0

    monkeypatch.setattr(pipeline, "run_command", run_episode)
    seal_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        pipeline,
        "seal_mobilegpt_source_memory",
        lambda **kwargs: seal_calls.append(kwargs) or {"manifest": {}},
    )

    prepared = mobilegpt_source.prepare_mobilegpt_source_memory(
        index_path=index,
        task_name="SystemBluetoothTurnOn",
        mobilegpt_root=tmp_path / "mobilegpt",
        android_world_root=tmp_path / "android_world",
        output_root=tmp_path / "bundle",
        model="GLM-4.6V",
        serial="emulator-5560",
        console_port=5560,
        adb_path="/sdk/adb",
    )

    server_kwargs = captured["server_kwargs"]
    episode_kwargs = captured["episode_kwargs"]
    assert isinstance(server_kwargs, dict)
    assert isinstance(episode_kwargs, dict)
    assert captured["server_action"] == "server"
    assert package_resolution == [("settings", "emulator-5560")]
    assert "source_run_log" not in server_kwargs
    assert server_kwargs["target_package"] == "com.android.settings"
    assert server_kwargs["embedding_model"] == "GLM-Embedding-2"
    assert server_kwargs["write_through_memory"] is True
    assert episode_kwargs["task_random_seed"] == 111
    assert episode_kwargs["target"].serial == "emulator-5560"
    assert episode_kwargs["method_name"] == MOBILEGPT_SOURCE_METHOD
    assert episode_kwargs["perform_emulator_setup"] is True
    assert seal_calls[0]["official_source_result"] == (
        episode_output / "task_results.jsonl"
    )
    assert prepared["teacher_forcing"] is False
    assert prepared["actions_supplied_to_mobilegpt"] is False
    assert prepared["runlog_conversion_used"] is False
    assert prepared["source_emulator_used"] is True


def test_source_cli_exposes_no_runlog_converter_or_teacher_command() -> None:
    help_text = mobilegpt_source.build_parser().format_help().casefold()

    assert "convert" not in help_text
    assert "teacher" not in help_text
    assert "seal-existing" not in help_text


def test_public_script_has_source_cold_memory_only_mode() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "exp"
        / "run_androidworld.sh"
    ).read_text(encoding="utf-8")

    assert "--prepare-mobilegpt-memory-only" in script
    assert "original cold build validated; no target emulator started" in script
