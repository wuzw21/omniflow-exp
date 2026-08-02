from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from runlog_fixtures import (
    androidworld_run_log,
    androidworld_state,
    mobilegpt_native_fallback_run_log,
    mobilegpt_partial_grounding_run_log,
)

from src.experiment import androidworld as pipeline
from src.experiment import mobilegpt_source
from src.integrations.mobilegpt_runtime import _mobilegpt_chat_model


def _write_source_index(
    root: Path,
    *,
    method: str = "ours",
    official_success: bool = True,
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    source_run_log = root / "source.run_log.json"
    source_run_log.write_text(
        json.dumps(
            androidworld_run_log(
                [{"action_type": "click", "x": 50, "y": 50}],
                observations=[
                    androidworld_state(
                        "state-0",
                        package_name="com.android.settings",
                        width=100,
                        height=100,
                        with_pixels=True,
                    )
                ],
                task_name="SystemBluetoothTurnOn",
                goal="Turn Bluetooth on.",
            )
        ),
        encoding="utf-8",
    )
    state_catalog = root / "transfer_states.json"
    state_catalog.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.transfer-state-catalog.v1",
                "run_id": "source-run",
                "states": {
                    "state-0": {
                        "state_id": "state-0",
                        "package_name": "com.android.settings",
                        "xml": (
                            '<hierarchy><node text="Bluetooth" '
                            'resource-id="android:id/switch_widget" '
                            'clickable="true" bounds="[0,0][100,100]" />'
                            "</hierarchy>"
                        ),
                        "display": {"width": 100, "height": 100},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    provenance = root / "provenance_manifest.json"
    provenance.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.source-replay-transfer-store.v1",
                "source_target_audit": {
                    "source_target_audit_complete": True,
                    "source_targets": [
                        {
                            "step_index": 0,
                            "source_state_id": "state-0",
                            "target": {
                                "text": "Bluetooth",
                                "resource_id": "android:id/switch_widget",
                            },
                        }
                    ],
                },
            }
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
                    "method": method,
                    "latest_official_success_source": official_success,
                    "source_kind": (
                        "androidworld_validator_success_source_runlog"
                    ),
                    "source_run_log_sha256": hashlib.sha256(
                        source_run_log.read_bytes()
                    ).hexdigest(),
                    "source_state_catalog": str(state_catalog),
                    "source_state_catalog_sha256": hashlib.sha256(
                        state_catalog.read_bytes()
                    ).hexdigest(),
                    "store_provenance": str(provenance),
                    "store_provenance_sha256": hashlib.sha256(
                        provenance.read_bytes()
                    ).hexdigest(),
                }
            }
        ),
        encoding="utf-8",
    )
    return index, source_run_log


def _mobilegpt_write_status(
    stats_path: Path,
    rows: list[dict],
) -> tuple[dict, dict]:
    stats_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    summary = pipeline.summarize_mobilegpt_stats(stats_path)
    status = pipeline._mobilegpt_memory_write_status(
        stats_summary=summary,
        memory_inventory={
            "has_recallable_subtasks": True,
            "has_useful_actions": True,
            "native_memory_complete": True,
        },
    )
    return summary, status


def _write_mobilegpt_memory(
    root: Path,
    *,
    root_task_name: str = "toggleBluetooth",
    app_task_name: str = "toggleBluetooth",
) -> None:
    app_root = root / "com.android.settings"
    page_root = app_root / "pages" / "0"
    page_root.mkdir(parents=True)
    (root / "tasks.csv").write_text(
        "name,description,parameters,app\n"
        f"{root_task_name},Toggle Bluetooth,{{}},com.android.settings\n",
        encoding="utf-8",
    )
    (app_root / "tasks.csv").write_text(
        f'name,path\n{app_task_name},"{{""0"": [""toggleBluetooth""]}}"\n',
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
    (page_root / "available_subtasks.csv").write_text(
        "name,description,parameters\n"
        "toggleBluetooth,Toggle Bluetooth,{}\n",
        encoding="utf-8",
    )
    (page_root / "subtasks.csv").write_text(
        "name,description,parameters\ntoggleBluetooth,Toggle Bluetooth,{}\n",
        encoding="utf-8",
    )
    (page_root / "actions.csv").write_text(
        "subtask_name,step,action,example\n"
        'toggleBluetooth,0,"{""name"": ""click""}",{}\n',
        encoding="utf-8",
    )
    screen_root = page_root / "screen"
    screen_root.mkdir()
    (screen_root / "hierarchy.xml").write_text(
        "<hierarchy />\n",
        encoding="utf-8",
    )
    for name in ("raw.xml", "html.xml", "parsed.xml", "pretty.xml"):
        (screen_root / name).write_text("<hierarchy />\n", encoding="utf-8")
    (screen_root / "screenshot.jpg").write_bytes(b"jpeg")


def test_mobilegpt_memory_inventory_requires_matching_task_graph(
    tmp_path: Path,
) -> None:
    memory = tmp_path / "memory"
    _write_mobilegpt_memory(memory)

    inventory = pipeline.inspect_mobilegpt_memory(memory)

    assert inventory["root_task_file_count"] == 1
    assert inventory["root_task_rows"] == 1
    assert inventory["root_task_names"] == ["toggleBluetooth"]
    assert inventory["task_file_count"] == 1
    assert inventory["task_rows"] == 1
    assert inventory["app_task_names"] == ["toggleBluetooth"]
    assert inventory["page_file_count"] == 1
    assert inventory["page_rows"] == 1
    assert inventory["hierarchy_file_count"] == 1
    assert inventory["hierarchy_rows"] == 1
    assert inventory["available_subtask_file_count"] == 1
    assert inventory["screen_directory_count"] == 1
    assert inventory["screen_file_count"] == 6
    assert inventory["native_memory_complete"] is True
    assert inventory["task_local_memory"] is True


def test_mobilegpt_memory_inventory_rejects_cross_task_graph(
    tmp_path: Path,
) -> None:
    memory = tmp_path / "memory"
    _write_mobilegpt_memory(
        memory,
        root_task_name="toggleBluetooth",
        app_task_name="changeBrightness",
    )

    inventory = pipeline.inspect_mobilegpt_memory(memory)

    assert inventory["task_local_memory"] is False


def test_mobilegpt_memory_inventory_rejects_incomplete_native_page(
    tmp_path: Path,
) -> None:
    memory = tmp_path / "memory"
    _write_mobilegpt_memory(memory)
    (memory / "com.android.settings" / "pages" / "0" / "available_subtasks.csv").unlink()

    inventory = pipeline.inspect_mobilegpt_memory(memory)

    assert inventory["native_memory_complete"] is False


@pytest.mark.parametrize("official_success", (True, False))
def test_mobilegpt_native_memory_seal_contains_no_teacher_artifacts(
    tmp_path: Path,
    official_success: bool,
) -> None:
    bundle = tmp_path / "bundle"
    memory = bundle / "memory"
    _write_mobilegpt_memory(memory)
    source_run_log = tmp_path / "source.run_log.json"
    source_run_log.write_text('{"source": true}\n', encoding="utf-8")
    stats = tmp_path / "source_stats.jsonl"
    stats.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"event": "task_started"},
                {
                    "event": "chat_call",
                    "model": "qwen3-vl-plus",
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
                {"event": "task_finished"},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    official_result = tmp_path / "task_results.jsonl"
    official_result.write_text(
        json.dumps(
            {
                "task_name": "SystemBluetoothTurnOn",
                "official_validator_used": True,
                "official_validator_success": official_success,
                "success": official_success,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    sealed = pipeline.seal_mobilegpt_adapted_memory(
        memory_root=memory,
        source_run_log=source_run_log,
        source_stats=stats,
        official_source_result=official_result,
        task_name="SystemBluetoothTurnOn",
        target_package="com.android.settings",
        target_app="Settings",
        source_method=pipeline.MOBILEGPT_NATIVE_SOURCE_METHOD,
        source_model="qwen3-vl-plus",
    )

    manifest = sealed["manifest"]
    assert manifest["schema_version"] == (
        "omniflow.mobilegpt-native-cold-memory.v1"
    )
    assert "teacher_source" not in manifest
    assert manifest["provenance"]["learning_mode"] == "mobilegpt_native_cold"
    assert manifest["provenance"]["teacher_forcing"] is False
    assert manifest["provenance"]["synthetic_subtasks"] is False
    assert (
        manifest["official_source_result"]["official_validator_success"]
        is official_success
    )
    assert sealed["memory_inventory"]["task_local_memory"] is True


def _write_store_index(source_index: Path) -> tuple[Path, Path]:
    source_payload = json.loads(source_index.read_text(encoding="utf-8"))
    source_row = source_payload["SystemBluetoothTurnOn"]
    indexed_source_run_log = Path(source_row["retained_source_run_log"])
    source_run_log = source_index.with_name("store_source.run_log.json")
    source_run_log.write_bytes(indexed_source_run_log.read_bytes())
    store_index = source_index.with_name("store_index.json")
    store_index.write_text(
        json.dumps(
            {
                "SystemBluetoothTurnOn": {
                    "source_run_log_path": str(source_run_log),
                    "source_run_log_sha256": hashlib.sha256(
                        source_run_log.read_bytes()
                    ).hexdigest(),
                    "transfer_states_path": source_row["source_state_catalog"],
                    "transfer_states_sha256": source_row[
                        "source_state_catalog_sha256"
                    ],
                    "provenance_path": source_row["store_provenance"],
                    "provenance_sha256": source_row[
                        "store_provenance_sha256"
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    return store_index, source_run_log


def test_mobilegpt_teacher_source_records_partial_grounding_fallback(
    tmp_path: Path,
) -> None:
    source_run_log = tmp_path / "partial.run_log.json"
    source_run_log.write_text(
        json.dumps(
            mobilegpt_partial_grounding_run_log(
                task_name="PartialGroundingTask"
            )
        ),
        encoding="utf-8",
    )

    teacher_source = pipeline.build_mobilegpt_teacher_source(
        source_run_log,
        task_name="PartialGroundingTask",
        fallback_to_vlm_on_teacher_miss=True,
    )

    assert teacher_source["action_count"] == 2
    assert teacher_source["groundable_action_count"] == 1
    assert teacher_source["expected_vlm_fallback_action_count"] == 1
    assert teacher_source["fallback_to_vlm_on_teacher_miss"] is True


def test_mobilegpt_teacher_source_records_native_fallback_only(
    tmp_path: Path,
) -> None:
    source_run_log = tmp_path / "native-fallback.run_log.json"
    source_run_log.write_text(
        json.dumps(mobilegpt_native_fallback_run_log(task_name="QueryTask")),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="mobilegpt_teacher_source_has_no_supported_actions",
    ):
        pipeline.build_mobilegpt_teacher_source(
            source_run_log,
            task_name="QueryTask",
        )

    teacher_source = pipeline.build_mobilegpt_teacher_source(
        source_run_log,
        task_name="QueryTask",
        fallback_to_vlm_on_teacher_miss=True,
    )

    assert teacher_source["action_count"] == 0
    assert teacher_source["groundable_action_count"] == 0
    assert teacher_source["native_vlm_fallback_only"] is True
    assert teacher_source["fallback_to_vlm_on_teacher_miss"] is True


def test_mobilegpt_native_cold_learning_can_seal(tmp_path: Path) -> None:
    summary, status = _mobilegpt_write_status(
        tmp_path / "source_stats.jsonl",
        [
            {"event": "task_started"},
            {
                "event": "chat_call",
                "model": "qwen3-vl-plus",
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
            },
            {"event": "task_finished"},
        ],
    )

    assert summary["task_started_count"] == 1
    assert summary["task_finished_count"] == 1
    assert summary["teacher_event_count"] == 0
    assert status["memory_written"] is True


def test_mobilegpt_native_cold_learning_requires_a_model_call(tmp_path: Path) -> None:
    summary, _ = _mobilegpt_write_status(
        tmp_path / "source_stats.jsonl",
        [
            {"event": "task_started"},
            {"event": "task_finished"},
        ],
    )
    status = pipeline._mobilegpt_memory_write_status(
        stats_summary=summary,
        memory_inventory={
            "has_recallable_subtasks": True,
            "has_useful_actions": True,
            "native_memory_complete": True,
        },
    )

    assert status["memory_written"] is False
    assert "missing_native_model_calls" in status["reasons"]


def test_mobilegpt_native_cold_learning_rejects_teacher_events(
    tmp_path: Path,
) -> None:
    _, status = _mobilegpt_write_status(
        tmp_path / "source_stats.jsonl",
        [
            {"event": "task_started"},
            {"event": "mobilegpt_teacher_started"},
            {"event": "chat_call", "model": "qwen3-vl-plus"},
            {"event": "task_finished"},
        ],
    )

    assert status["memory_written"] is False
    assert "teacher_forcing_detected" in status["reasons"]


def test_mobilegpt_v1_stats_manifest_allows_absent_derived_counts() -> None:
    expected = {
        "teacher_action_count": 7,
        "teacher_groundable_action_count": 7,
        "teacher_vlm_fallback_count": 0,
    }

    assert pipeline._mobilegpt_stats_manifest_matches(
        {"teacher_action_count": 7},
        expected,
    )
    assert not pipeline._mobilegpt_stats_manifest_matches(
        {
            "teacher_action_count": 7,
            "teacher_groundable_action_count": 0,
        },
        expected,
    )


def test_mobilegpt_deterministic_preflight_does_not_claim_output(
    tmp_path: Path,
) -> None:
    index, _ = _write_source_index(tmp_path / "source")
    payload = json.loads(index.read_text(encoding="utf-8"))
    row = payload["SystemBluetoothTurnOn"]
    row["source_state_catalog_sha256"] = "0" * 64
    index.write_text(json.dumps(payload), encoding="utf-8")
    store_index, _ = _write_store_index(index)
    store_payload = json.loads(store_index.read_text(encoding="utf-8"))
    store_payload["SystemBluetoothTurnOn"]["transfer_states_sha256"] = "0" * 64
    store_index.write_text(json.dumps(store_payload), encoding="utf-8")
    output_root = tmp_path / "never-created"

    with pytest.raises(ValueError, match="source_state_catalog_hash_mismatch"):
        mobilegpt_source.prepare_mobilegpt_source_memory(
            index_path=index,
            store_index_path=store_index,
            task_name="SystemBluetoothTurnOn",
            mobilegpt_root=tmp_path / "mobilegpt",
            android_world_root=tmp_path / "android_world",
            output_root=output_root,
            model="qwen3-vl-plus",
        )

    assert not output_root.exists()


def test_mobilegpt_source_accepts_successful_canonical_recorded_seed(
    tmp_path: Path,
) -> None:
    accepted_index, source_run_log = _write_source_index(tmp_path / "accepted")
    payload = json.loads(accepted_index.read_text(encoding="utf-8"))
    payload["SystemBluetoothTurnOn"]["replay_seed"] = 3936510006
    accepted_index.write_text(json.dumps(payload), encoding="utf-8")
    item = mobilegpt_source.load_canonical_source_item(
        accepted_index,
        task_name="SystemBluetoothTurnOn",
    )
    assert item.source_run_log == source_run_log
    assert item.replay_seed == 3936510006
    item.meta.pop("method")
    assert mobilegpt_source.source_method_label(item) == (
        "mobilegpt_native_source_cold"
    )

    rejected_root = tmp_path / "rejected"
    rejected_root.mkdir()
    rejected_index, _ = _write_source_index(
        rejected_root,
        method="fixed_replay",
    )
    item = mobilegpt_source.load_canonical_source_item(
        rejected_index,
        task_name="SystemBluetoothTurnOn",
    )
    assert mobilegpt_source.source_method_label(item) == (
        "mobilegpt_native_source_cold"
    )


def test_mobilegpt_offline_runner_uses_protocol_source_method_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_memory_root = tmp_path / "source-memory"
    source_memory_root.mkdir()
    source_run_log = tmp_path / "source.run_log.json"
    source_run_log.write_text("{}", encoding="utf-8")
    adapted_source_run_log = tmp_path / "adapted-source.run_log.json"
    adapted_source_run_log.write_text('{"canonical": true}', encoding="utf-8")
    item = pipeline.ArchivedRunLog(
        task="SystemBluetoothTurnOn",
        goal="Turn Bluetooth on.",
        params={},
        source_run_log=source_run_log,
        replay_seed=111,
        step_count=1,
        meta={"latest_official_success_source": True},
    )

    class ValidationReached(RuntimeError):
        pass

    def validate_source_memory(*args: object, **kwargs: object) -> dict[str, object]:
        assert kwargs["expected_source_method"] == (
            "mobilegpt_native_source_cold"
        )
        assert kwargs["source_run_log"] == adapted_source_run_log
        assert kwargs["compatible_source_sha256s"] == ("1" * 64, "2" * 64)
        raise ValidationReached

    monkeypatch.setattr(
        pipeline,
        "validate_mobilegpt_adapted_memory",
        validate_source_memory,
    )

    with pytest.raises(ValidationReached):
        pipeline._run_one_task_mobilegpt(
            args=pipeline.argparse.Namespace(
                mobilegpt_source_memory_root=str(source_memory_root),
                model="qwen3-vl-plus",
            ),
            item=item,
            targets=[
                pipeline.DeviceTarget(
                    label="small5554",
                    serial="emulator-5554",
                    console_port=5554,
                )
            ],
            output_root=tmp_path / "results",
            task_params_override=None,
            task_seed=113,
            method="mobilegpt_offline_retrieval",
            attempt_id="attempt-1",
            source_run_log=adapted_source_run_log,
            compatible_source_sha256s=("1" * 64, "2" * 64),
        )


def test_mobilegpt_legacy_fixed_replay_source_requires_strict_evidence(
    tmp_path: Path,
) -> None:
    source_run_log = tmp_path / "source.run_log.json"
    payload = {
        "schema_version": "omniflow.androidworld_function_ready_runlog.v1",
        "completed": True,
        "success": True,
        "androidworld": {
            "task_name": "SystemBluetoothTurnOff",
            "seed": 111,
            "validator": {
                "success": True,
                "uses_androidworld_official_validator": True,
            },
        },
        "raw_replay_evidence": {
            "success": True,
            "official_validator_used": True,
            "actions_executed": 5,
            "source_action_count": 5,
            "execution_backend": "memory_adapter_exact_sequence",
            "model_calls": 0,
        },
    }
    source_run_log.write_text(json.dumps(payload), encoding="utf-8")

    assert pipeline._mobilegpt_legacy_fixed_replay_source(
        source_run_log,
        task_name="SystemBluetoothTurnOff",
        source_seed=111,
    )

    payload["raw_replay_evidence"]["model_calls"] = 1
    source_run_log.write_text(json.dumps(payload), encoding="utf-8")
    assert not pipeline._mobilegpt_legacy_fixed_replay_source(
        source_run_log,
        task_name="SystemBluetoothTurnOff",
        source_seed=111,
    )


def test_mobilegpt_source_rejects_registered_historical_runlog(
    tmp_path: Path,
) -> None:
    index, source_run_log = _write_source_index(tmp_path / "historical")
    source_run_log.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.run_log.v1",
                "run_id": "historical-source",
                "goal": "Turn Bluetooth on.",
                "completed": True,
                "success": True,
                "steps": [
                    {
                        "step_index": 0,
                        "observation_before_act": {
                            "state_id": "state-0",
                            "width": 100,
                            "height": 100,
                        },
                        "executed_actions": [
                            {
                                "type": "click",
                                "params": {"x": 50, "y": 50},
                            }
                        ],
                        "success": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    payload = json.loads(index.read_text(encoding="utf-8"))
    row = payload["SystemBluetoothTurnOn"]
    row.pop("source_kind")
    row.pop("source_run_log_sha256")
    row["retained_source_run_log_sha256"] = hashlib.sha256(
        source_run_log.read_bytes()
    ).hexdigest()
    index.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="run_log_schema_invalid"):
        pipeline.build_mobilegpt_teacher_source(
            source_run_log,
            task_name="SystemBluetoothTurnOn",
            provenance_source_run_log=source_run_log,
        )


def test_mobilegpt_preflight_resolves_target_from_frozen_source_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index, _ = _write_source_index(tmp_path / "source-package")
    store_index, _ = _write_store_index(index)
    monkeypatch.setattr(
        pipeline,
        "_infer_mobilegpt_target_from_source_run_log",
        lambda _item: {
            "target_package": "",
            "target_app": "",
            "target_source": "unresolved",
        },
    )

    result = mobilegpt_source.preflight_mobilegpt_source(
        index_path=index,
        store_index_path=store_index,
        task_name="SystemBluetoothTurnOn",
    )

    assert result["target_package"] == "com.android.settings"
    assert result["target_source"] == "frozen_source_states"


def test_mobilegpt_preflight_uses_canonical_store_source(
    tmp_path: Path,
) -> None:
    index, indexed_source_run_log = _write_source_index(tmp_path / "source-store")
    store_index, store_source_run_log = _write_store_index(index)
    indexed_payload = json.loads(indexed_source_run_log.read_text(encoding="utf-8"))
    indexed_payload["steps"][0]["observation"] = androidworld_state(
        "different-state",
        forest="",
        package_name="com.android.settings",
    )
    indexed_source_run_log.write_text(
        json.dumps(indexed_payload),
        encoding="utf-8",
    )
    source_index_payload = json.loads(index.read_text(encoding="utf-8"))
    source_index_payload["SystemBluetoothTurnOn"]["source_run_log_sha256"] = (
        hashlib.sha256(indexed_source_run_log.read_bytes()).hexdigest()
    )
    index.write_text(json.dumps(source_index_payload), encoding="utf-8")

    result = mobilegpt_source.preflight_mobilegpt_source(
        index_path=index,
        store_index_path=store_index,
        task_name="SystemBluetoothTurnOn",
    )

    assert Path(result["source_run_log"]) == store_source_run_log
    assert result["ready"] is True


def test_mobilegpt_preflight_grounds_complete_store_source_without_full_function_catalog(
    tmp_path: Path,
) -> None:
    index, indexed_source_run_log = _write_source_index(
        tmp_path / "partial-function-catalog"
    )
    indexed_source_run_log.write_text(
        json.dumps(
            androidworld_run_log(
                [
                    {"action_type": "click", "x": 50, "y": 50},
                    {"action_type": "click", "x": 50, "y": 50},
                ],
                observations=[
                    androidworld_state(
                        "state-0",
                        forest=(
                            '<hierarchy><node text="Bluetooth" clickable="true" '
                            'bounds="[0,0][100,100]" /></hierarchy>'
                        ),
                        package_name="com.android.settings",
                    ),
                    androidworld_state(
                        "state-1",
                        forest=(
                            '<hierarchy><node text="Continue" clickable="true" '
                            'bounds="[0,0][100,100]" /></hierarchy>'
                        ),
                        package_name="com.android.settings",
                    ),
                ],
                task_name="SystemBluetoothTurnOn",
                goal="Turn Bluetooth on.",
            )
        ),
        encoding="utf-8",
    )
    source_index_payload = json.loads(index.read_text(encoding="utf-8"))
    source_row = source_index_payload["SystemBluetoothTurnOn"]
    source_row["source_run_log_sha256"] = hashlib.sha256(
        indexed_source_run_log.read_bytes()
    ).hexdigest()
    source_row["step_count"] = 2
    index.write_text(json.dumps(source_index_payload), encoding="utf-8")
    store_index, store_source_run_log = _write_store_index(index)

    result = mobilegpt_source.preflight_mobilegpt_source(
        index_path=index,
        store_index_path=store_index,
        task_name="SystemBluetoothTurnOn",
    )

    assert Path(result["source_run_log"]) == store_source_run_log
    assert result["source_audit"]["source_state_count"] == 2
    assert result["source_audit"]["grounding_source"] == (
        "canonical_androidworld_run_log"
    )
    assert result["ready"] is True


def test_mobilegpt_preflight_resolves_target_from_official_open_app_action(
    tmp_path: Path,
) -> None:
    index, source_run_log = _write_source_index(tmp_path / "official-action")
    store_index, _ = _write_store_index(index)
    source_run_log.write_text(
        json.dumps(
            androidworld_run_log(
                [
                    {
                        "action_type": "open_app",
                        "app_name": "com.android.settings",
                    },
                    {"action_type": "click", "x": 50, "y": 50},
                ],
                observations=[
                    androidworld_state(
                        "state-0",
                        forest="",
                        package_name="com.google.android.apps.nexuslauncher",
                    ),
                    androidworld_state(
                        "state-1",
                        forest=(
                            '<hierarchy><node text="Bluetooth" '
                            'resource-id="android:id/switch_widget" '
                            'clickable="true" bounds="[0,0][100,100]" />'
                            "</hierarchy>"
                        ),
                        package_name="com.android.settings",
                        width=100,
                        height=100,
                    ),
                ],
                task_name="SystemBluetoothTurnOn",
                goal="Turn Bluetooth on.",
            )
        ),
        encoding="utf-8",
    )
    payload = json.loads(index.read_text(encoding="utf-8"))
    row = payload["SystemBluetoothTurnOn"]
    row["source_run_log_sha256"] = hashlib.sha256(
        source_run_log.read_bytes()
    ).hexdigest()
    row["step_count"] = 2
    for key in (
        "source_state_catalog",
        "source_state_catalog_sha256",
        "store_provenance",
        "store_provenance_sha256",
    ):
        row.pop(key)
    index.write_text(json.dumps(payload), encoding="utf-8")

    result = mobilegpt_source.preflight_mobilegpt_source(
        index_path=index,
        store_index_path=store_index,
        task_name="SystemBluetoothTurnOn",
    )

    assert result["target_package"] == "com.android.settings"
    assert result["target_source"] == "source_runlog_open_app"


def test_mobilegpt_source_reads_explicit_source_seed(tmp_path: Path) -> None:
    index, _ = _write_source_index(tmp_path / "source-seed")
    payload = json.loads(index.read_text(encoding="utf-8"))
    row = payload["SystemBluetoothTurnOn"]
    row["source_seed"] = row.pop("replay_seed")
    index.write_text(json.dumps(payload), encoding="utf-8")

    item = mobilegpt_source.load_canonical_source_item(
        index,
        task_name="SystemBluetoothTurnOn",
    )

    assert item.replay_seed == 111


def test_mobilegpt_configured_model_overrides_upstream_model_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MOBILEGPT_CHAT_MODEL", "qwen3-vl-plus")
    assert _mobilegpt_chat_model("gpt-4") == "qwen3-vl-plus"


def test_mobilegpt_source_generation_has_no_model_or_episode_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    index, _ = _write_source_index(source_root)
    store_index, _ = _write_store_index(index)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        pipeline,
        "_infer_mobilegpt_target_from_source_run_log",
        lambda _item: {
            "target_package": "com.android.settings",
            "target_app": "Settings",
        },
    )
    monkeypatch.setattr(
        pipeline,
        "_patch_mobilegpt_stats",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        pipeline,
        "_patch_mobilegpt_server_runtime_context",
        lambda **_kwargs: [],
    )

    def build_server(action: str, **kwargs: object) -> pipeline.CommandSpec:
        captured["server_action"] = action
        captured["server_kwargs"] = kwargs
        return pipeline.CommandSpec(
            label="mobilegpt-native",
            argv=["python", "server.py"],
            env={},
            cwd=tmp_path,
        )

    monkeypatch.setattr(pipeline, "build_mobilegpt_command", build_server)
    monkeypatch.setattr(
        pipeline,
        "_start_mobilegpt_browser_task_server",
        lambda **_kwargs: ({}, None),
    )

    episode_output = tmp_path / "episode"

    def build_episode(
        *_args: object,
        **kwargs: object,
    ) -> pipeline.CommandSpec:
        captured["episode_kwargs"] = kwargs
        return pipeline.CommandSpec(
            label="mobilegpt-source",
            argv=["python", "episode.py"],
            env={},
            cwd=tmp_path,
            output_path=episode_output,
            metadata={},
        )

    monkeypatch.setattr(
        pipeline,
        "build_mobilegpt_androidworld_command",
        build_episode,
    )

    def start_server(
        spec: pipeline.CommandSpec,
        *,
        warmup_sec: float,
    ) -> tuple[object, int]:
        del warmup_sec
        captured["server"] = spec
        return object(), 0

    monkeypatch.setattr(pipeline, "_start_background_command", start_server)
    monkeypatch.setattr(
        pipeline,
        "_stop_background_command",
        lambda _process: None,
    )

    def run_episode(spec: pipeline.CommandSpec) -> int:
        assert spec.output_path is not None
        spec.output_path.mkdir(parents=True)
        (spec.output_path / "task_results.jsonl").write_text(
            json.dumps(
                {
                    "task_name": "SystemBluetoothTurnOn",
                    "official_validator_used": True,
                    "official_validator_success": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        stats_path = (
            tmp_path / "bundle" / "source_stats.jsonl"
        )
        stats_path.write_text(
            json.dumps(
                {
                    "event": "chat_call",
                    "model": "qwen3-vl-plus",
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(pipeline, "run_command", run_episode)
    monkeypatch.setattr(
        pipeline,
        "seal_mobilegpt_adapted_memory",
        lambda **kwargs: {"source_method": kwargs["source_method"]},
    )

    result = mobilegpt_source.prepare_mobilegpt_source_memory(
        index_path=index,
        store_index_path=store_index,
        task_name="SystemBluetoothTurnOn",
        mobilegpt_root=tmp_path / "mobilegpt",
        android_world_root=tmp_path / "android_world",
        output_root=tmp_path / "bundle",
        model="qwen3-vl-plus",
    )

    server = captured["server"]
    server_kwargs = captured["server_kwargs"]
    episode_kwargs = captured["episode_kwargs"]
    assert isinstance(server, pipeline.CommandSpec)
    assert isinstance(server_kwargs, dict)
    assert isinstance(episode_kwargs, dict)
    assert captured["server_action"] == "server"
    assert "fallback_to_vlm_on_teacher_miss" not in server_kwargs
    assert "source_run_log" not in server_kwargs
    assert server_kwargs["runtime_observe_backend"] == "androidworld"
    assert server.env["MOBILEGPT_CHAT_MODEL"] == "qwen3-vl-plus"
    assert server.env["MOBILEGPT_CHAT_MAX_ATTEMPTS"] == "1"
    assert server.env["MOBILEGPT_TEACHER_RUNLOG"] == ""
    assert server.env["MOBILEGPT_TEACHER_ARTIFACT_DIR"] == ""
    assert server.env["MOBILEGPT_TEACHER_FALLBACK_TO_VLM_ON_MISS"] == ""
    assert "MOBILEGPT_OOB_OBSERVE_RETRIES" not in server.env
    assert server.metadata["episode_retries"] == 0
    assert episode_kwargs["server_host"] == "0.0.0.0"
    assert episode_kwargs["target_package"] == "com.android.settings"
    assert episode_kwargs["max_steps"] == 20
    assert episode_kwargs["timeout_sec"] == 600.0
    assert "rebroadcast_limit" not in episode_kwargs
    assert result["source_method"] == "mobilegpt_native_source_cold"
    assert result["learning_mode"] == "mobilegpt_native_cold"
    assert result["teacher_forcing"] is False
    assert result["synthetic_subtasks"] is False
