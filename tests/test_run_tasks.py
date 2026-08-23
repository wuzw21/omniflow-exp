from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from runlog_fixtures import androidworld_run_log

from src.experiment.run_task import (
    DeviceTarget,
    _canonical_function_source_call,
    _resolve_mobilegpt_target_package,
    _mobilegpt_server_task_app,
    bind_function_arguments_to_task_params,
    _read_object,
    _t3a_hint_source_node,
    build_mobilegpt_server_command,
    build_autodroid_command,
    build_task_command,
    _formal_result_paths,
    _result_summary_rows,
    _subprocess_env,
    build_parser as build_run_task_parser,
    load_canonical_source_index,
)
from src.experiment.source_records import CanonicalRunLog
from src.experiment.batch_outcomes import record_result_outcome
from src.experiment.run_tasks import (
    Deadline,
    PipelinePhaseError,
    _bmoca_source_replay_qualified,
    _cached_source_function_qualification,
    _concluded_results,
    _fixed_replay_source_step_width,
    _function_replay_success,
    _e2e_devices,
    _e2e_methods,
    _ensure_oob_release_installed,
    _autodroid_task_params_from_index,
    _supplemental_outcomes_root,
    _max_live_bmoca_results,
    _mobilegpt_registered_conclusion_is_reusable,
    _next_source_attempt_id,
    _next_pipeline_attempt_id,
    _parse_source_device,
    _published_official_result_row,
    _result_row_is_environment_failure,
    _report,
    _resolve_args,
    _run_bmoca_method_results,
    _save_bmoca_function_once,
    _source_device_ready,
    build_parser,
    collect_replayed_source,
    ensure_source_device,
    ensure_target_devices,
    prepare_function_asset,
    prepare_mobilegpt_memory,
    qualify_source_function,
    run_bmoca_pipeline,
    run_logged_command,
    run_pipeline,
    run_target_workers,
)
from src.experiment.protocol import (
    BMOCA_RESULT_TIMEOUT_SEC,
    DEVICES,
    EPISODE_TIMEOUT_SEC,
    FUNCTION_ENHANCEMENT_TIMEOUT_SEC,
    MAX_FALLBACK_STEPS,
    MAX_STEPS,
    METHODS,
    SOURCE_AVD,
    SOURCE_DEVICE,
    SOURCE_MAX_STEPS,
    SOURCE_SEED,
    SUPPLEMENTAL_DEVICES,
    SUPPLEMENTAL_METHODS,
    SUPPLEMENTAL_RESULTS_NAMESPACE,
    STEP_TIMEOUT_SEC,
    TASK_DEADLINE_SEC,
    TASK_SEED,
)


def test_mobilegpt_camera_alias_resolves_to_installed_camera2_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.experiment.run_task.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="package:com.android.camera2\n",
        ),
    )

    assert (
        _resolve_mobilegpt_target_package(
            "Camera",
            adb_path="adb",
            serial="emulator-5564",
        )
        == "com.android.camera2"
    )


def test_subprocess_env_aliases_shared_glm_endpoint(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("LLMTHU_API_KEY", "test-key")

    environment = _subprocess_env({})

    assert environment["OPENAI_API_KEY"] == "test-key"
    assert environment["OPENAI_BASE_URL"]


def test_source_index_skips_unmaterialized_unrelated_task(tmp_path: Path) -> None:
    source = tmp_path / "ready.run_log.json"
    source.write_text("{}\n", encoding="utf-8")
    index = tmp_path / "current.json"
    index.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.data-index.v2",
                "source_index": {
                    "ReadyTask": {
                        "goal": "ready",
                        "latest_official_success_source": True,
                        "retained_source_run_log": str(source),
                    },
                    "PendingTask": {
                        "goal": "pending",
                        "latest_official_success_source": True,
                        "retained_source_run_log": "",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = load_canonical_source_index(index)

    assert [item.task for item in loaded] == ["ReadyTask"]


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        task="BrowserDraw",
        task_deadline_sec=TASK_DEADLINE_SEC,
        max_steps=MAX_STEPS,
        max_fallback_steps=MAX_FALLBACK_STEPS,
        attempt_id="attempt-test",
        repo=tmp_path / "repo",
        output_root=tmp_path / "repo" / "data" / "output",
        results_root=tmp_path / "repo" / "data",
        memory_index=tmp_path / "repo" / "data" / "current.json",
        script=tmp_path / "repo" / "scripts" / "exp" / "run_androidworld.sh",
        asset_root=tmp_path / "repo" / "data",
        android_world_root=tmp_path / "android_world",
        omnitransfer_root=tmp_path / "OmniTransfer",
        mobilegpt_root=tmp_path / "MobileGPT",
        appagent_root=tmp_path / "AppAgent",
        python_bin=tmp_path / "python",
        adb_path=tmp_path / "adb",
        source_model="glm-5.1",
        source_device=SOURCE_DEVICE,
        source_avd=SOURCE_AVD,
        source_qualification_only=False,
        source_only=False,
        dry_run=False,
    )


def test_ensure_oob_release_installed_uses_one_canonical_apk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    apk = args.repo.parents[1] / "releases" / "OOB" / "OpenOmniBot-foolproof-debug.apk"
    apk.parent.mkdir(parents=True)
    apk.write_bytes(b"apk")
    calls: list[list[str]] = []
    monkeypatch.delenv("OMNIFLOW_OOB_APK", raising=False)
    monkeypatch.setattr(
        "src.experiment.run_tasks.run_logged_command",
        lambda command, **_kwargs: calls.append(command) or {"returncode": 0},
    )
    monkeypatch.setattr("src.experiment.run_tasks.time.sleep", lambda _seconds: None)

    result = _ensure_oob_release_installed(
        args=args,
        serial="emulator-5560",
        log_path=tmp_path / "oob.log",
        deadline=Deadline(120),
    )

    assert result["status"] == "installed"
    assert result["apk_path"] == str(apk.resolve())
    assert calls[0] == [
        str(args.adb_path),
        "-s",
        "emulator-5560",
        "install",
        "-r",
        "-t",
        str(apk.resolve()),
    ]
    assert calls[1][-2:] == [
        "enabled_accessibility_services",
        "cn.com.omnimind.bot.debug/cn.com.omnimind.accessibility.service.AssistsService",
    ]
    assert calls[2][-2:] == ["accessibility_enabled", "1"]


def test_next_source_attempt_id_uses_unified_monotonic_name(tmp_path: Path) -> None:
    args = _args(tmp_path)
    root = (
        args.results_root
        / "androidworld"
        / args.task
        / "source"
        / f"{args.source_avd}_seed{SOURCE_SEED}"
        / "runlog"
    )
    (root / "legacy_name").mkdir(parents=True)
    (root / "attempt_001").mkdir()
    (root / "attempt_007").mkdir()
    (root / "attempt_invalid").mkdir()

    assert _next_source_attempt_id(args) == "attempt_008"


def test_concluded_results_reuses_registered_formal_cell_across_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    args.task = "CameraTakePhoto"
    args.e2e_method = "appagent"
    args.e2e_device = DEVICES[0]
    registry_root = args.results_root / "androidworld" / ".archive" / "result_registry"
    registry_root.mkdir(parents=True)
    monkeypatch.setattr(
        "src.experiment.run_tasks.registered_result_plan",
        lambda **_kwargs: {
            "completed": [("appagent", "small5554")],
            "pending": [],
        },
    )

    assert _concluded_results(
        args,
        args.results_root / "androidworld" / ".archive" / "outcomes" / "formal",
        "attempt_002",
    ) == {("appagent", "small5554")}


def test_concluded_results_reruns_existing_omniflow_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    args.task = "CameraTakePhoto"
    args.e2e_method = "omniflow"
    args.e2e_device = DEVICES[0]
    registry_root = args.results_root / "androidworld" / ".archive" / "result_registry"
    registry_root.mkdir(parents=True)
    monkeypatch.setattr(
        "src.experiment.run_tasks.registered_result_plan",
        lambda **_kwargs: {
            "completed": [("omniflow", "small5554")],
            "pending": [],
        },
    )

    assert _concluded_results(
        args,
        args.results_root / "androidworld" / ".archive" / "outcomes" / "formal",
        "attempt_002",
    ) == set()


def test_concluded_results_reruns_mobilegpt_failure_with_legacy_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    args.task = "CameraTakePhoto"
    args.e2e_method = "mobilegpt"
    args.e2e_device = DEVICES[0]
    monkeypatch.setattr(
        "src.experiment.run_tasks.concluded_result_keys",
        lambda **_kwargs: {("mobilegpt", "small5554")},
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks.registered_result_plan",
        lambda **_kwargs: {
            "completed": [("mobilegpt", "small5554")],
            "pending": [],
        },
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks._mobilegpt_registered_conclusion_is_reusable",
        lambda **_kwargs: False,
    )

    assert _concluded_results(
        args,
        args.results_root / "androidworld" / ".archive" / "outcomes" / "formal",
        "attempt_002",
    ) == set()


def test_mobilegpt_failure_is_reusable_only_with_authoritative_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_root = tmp_path / "result_registry"
    result_root = (
        registry_root
        / "CameraTakePhoto"
        / "mobilegpt"
        / "small5554"
        / "attempt_001.mobilegpt.small5554"
    )
    result_root.mkdir(parents=True)
    target_memory = tmp_path / "target_memory"
    target_memory.mkdir()
    source_memory = tmp_path / "source_bundle" / "memory"
    source_memory.mkdir(parents=True)
    (target_memory / "memory_manifest.json").write_text(
        json.dumps({"artifacts": {"source_memory_root": str(source_memory)}}),
        encoding="utf-8",
    )
    (result_root / "registered_result.json").write_text(
        json.dumps(
            {
                "task_name": "CameraTakePhoto",
                "source_seed": SOURCE_SEED,
                "evaluation_seed": 113,
                "details": [
                    {
                        "method": "mobilegpt",
                        "device": "small5554",
                        "official_validator_success": False,
                        "memory_root": str(target_memory),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks._validate_prepared_mobilegpt_memory",
        lambda *_args, **_kwargs: {"runlog_teacher_alignment": True},
    )

    assert _mobilegpt_registered_conclusion_is_reusable(
        registry_root=registry_root,
        task_name="CameraTakePhoto",
        device="small5554",
        source_seed=SOURCE_SEED,
        evaluation_seed=113,
    )


def test_pipeline_attempt_id_grows_past_historical_outcome(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.attempt_id = "attempt_001"
    args.e2e_method = "appagent"
    args.e2e_device = DEVICES[0]
    outcomes_root = args.results_root / "androidworld" / ".archive" / "outcomes" / "formal"
    record_result_outcome(
        outcomes_root=outcomes_root,
        task_name=args.task,
        method="appagent",
        device="small5562",
        device_serial="emulator-5562",
        attempt_id="attempt_001",
        source_seed=111,
        evaluation_seed=113,
        status="completed",
        stage="androidworld_validate",
        official_validator_used=True,
        official_validator_success=True,
    )

    assert _next_pipeline_attempt_id(args, outcomes_root) == "attempt_002"


def test_result_summary_resolves_native_row_to_unique_command_device() -> None:
    rows = _result_summary_rows(
        task="CameraTakePhoto",
        command_records=[
            {
                "method": "fixed_replay",
                "device": "small5554",
                "status": "completed",
                "returncode": 0,
                "command": "run_episode",
                "output_path": "/tmp/attempt",
                "metadata": {},
            }
        ],
        aggregate_summary={
            "per_task": [
                {
                    "task_name": "CameraTakePhoto",
                    "method": "fixed_replay",
                    "device": "",
                    "official_validator_used": True,
                    "official_validator_success": True,
                    "duration_ms": 1000,
                    "actions_executed": 4,
                }
            ]
        },
    )

    assert rows[0]["device"] == "small5554"
    assert rows[0]["official_validator_used"] is True
    assert rows[0]["official_validator_success"] is True


def test_setup_command_failure_is_retryable_environment_failure(
    tmp_path: Path,
) -> None:
    log = tmp_path / "mobilegpt.log"
    log.write_text(
        "TimeoutError: AndroidWorld official app setup exceeded 300 seconds\n",
        encoding="utf-8",
    )

    assert _result_row_is_environment_failure(
        {"status": "command_failed", "validator_success": None},
        artifact_root=tmp_path / "artifact",
        task_log=log,
    ) is True


def test_plain_command_failure_remains_method_failure(tmp_path: Path) -> None:
    log = tmp_path / "mobilegpt.log"
    log.write_text(
        "mobilegpt_server_handler_failed\n",
        encoding="utf-8",
    )

    assert _result_row_is_environment_failure(
        {"status": "command_failed", "validator_success": None},
        artifact_root=tmp_path / "artifact",
        task_log=log,
    ) is False


@pytest.mark.skip(reason="retired lineage helper")
def test_t3a_hint_uses_function_store_source_lineage(tmp_path: Path) -> None:
    source = tmp_path / "lineage.run_log.json"
    source.write_text(
        json.dumps(
            androidworld_run_log(
                [{"action_type": "click", "x": 10, "y": 10}],
                task_name="BrowserDraw",
            )
        ),
        encoding="utf-8",
    )
    store = tmp_path / "function_store.json"
    store.write_text("{}", encoding="utf-8")
    index = tmp_path / "current.json"
    index.write_text(
        json.dumps(
            {
                "canonical": {
                    "function_stores": {
                        "BrowserDraw": {
                            "store_path": str(store),
                            "source_run_log_path": str(source),
                            "source_run_log_sha256": hashlib.sha256(
                                source.read_bytes()
                            ).hexdigest(),
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    item = CanonicalRunLog(
        task="BrowserDraw",
        goal="stale goal",
        params={},
        source_run_log=tmp_path / "stale.json",
        replay_seed=111,
        step_count=0,
        meta={},
    )

    aligned = _function_lineage_item(
        item,
        store_path=store,
        index_path=index,
    )

    assert aligned.source_run_log == source.resolve()
    assert aligned.step_count == 1
    assert aligned.goal == "Complete the task."


def test_t3a_hint_reads_native_ui_element_bounds() -> None:
    source_node = _t3a_hint_source_node(
        {
            "action": {"action_type": "click", "x": 360, "y": 1112},
            "observation": {
                "forest": {"windows": []},
                "ui_elements": [
                    {
                        "class_name": "android.view.View",
                        "resource_name": "com.android.camera2:id/preview_overlay",
                        "bbox_pixels": {
                            "x_min": 0,
                            "y_min": 0,
                            "x_max": 720,
                            "y_max": 1232,
                        },
                        "is_clickable": False,
                    },
                    {
                        "class_name": "android.widget.ImageView",
                        "content_description": "Shutter",
                        "resource_name": "com.android.camera2:id/shutter_button",
                        "bbox_pixels": {
                            "x_min": 0,
                            "y_min": 992,
                            "x_max": 720,
                            "y_max": 1232,
                        },
                        "is_clickable": True,
                    },
                ],
            },
        },
        forbidden_values=(),
    )

    assert source_node["content_description"] == "Shutter"
    assert source_node["resource_id"] == "com.android.camera2:id/shutter_button"


def test_e2e_command_exposes_direct_function_with_fallback_planner(
    tmp_path: Path,
) -> None:
    item = CanonicalRunLog(
        task="BrowserDraw",
        goal="Draw a shape",
        params={"seed": 111},
        source_run_log=tmp_path / "source.run_log.json",
        replay_seed=111,
        step_count=1,
        meta={},
    )

    spec = build_task_command(
        item,
        android_world_root=tmp_path / "android-world",
        output_root=tmp_path / "data",
        method_name="function_replay",
        device_label="source5560",
        serial="emulator-5560",
        console_port=5560,
        store_path=tmp_path / "function_store.json",
        omnitransfer_root=tmp_path / "OmniTransfer",
        function_id="complete_task",
        function_arguments={"target": "Alarm"},
        planner_provider="openai",
        model="GLM-4.6V",
        planner_timeout_sec=45,
    )

    assert spec.metadata["mode"] == "direct_function_e2e"
    assert spec.metadata["function_id"] == "complete_task"
    assert spec.argv[spec.argv.index("--function-id") + 1] == "complete_task"
    assert json.loads(
        spec.argv[spec.argv.index("--function-arguments-json") + 1]
    ) == {"target": "Alarm"}
    assert spec.argv[spec.argv.index("--planner-provider") + 1] == "openai"
    assert spec.argv[spec.argv.index("--model") + 1] == "GLM-4.6V"
    assert spec.argv[spec.argv.index("--planner-timeout-sec") + 1] == "45.0"


def test_result_runner_planner_timeout_defaults_to_formal_vision_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OMNIFLOW_ANDROIDWORLD_PLANNER_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("OMNIFLOW_PLANNER_TIMEOUT_SEC", raising=False)

    parser = build_run_task_parser()
    args = parser.parse_args(["result", "--task", "CameraTakeVideo"])

    assert args.planner_timeout_sec == 180.0


def test_canonical_function_source_call_selects_store_source_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeStore:
        load_errors = {}
        source_calls = [{"function_id": "take_photo", "arguments": {}}]

        def __init__(self, _path: Path) -> None:
            pass

        def get_function(self, function_id: str):
            return object() if function_id == "take_photo" else None

    monkeypatch.setattr("src.experiment.run_task.FunctionStore", FakeStore)

    assert _canonical_function_source_call(tmp_path / "function_store.json") == (
        "take_photo",
        {},
    )


def test_function_arguments_bind_only_declared_dynamic_task_params() -> None:
    assert bind_function_arguments_to_task_params(
        {"folder_name": "source-folder", "static": "keep"},
        {"folder_name": "evaluation-folder", "unrelated": "ignore"},
    ) == {
        "folder_name": "evaluation-folder",
        "static": "keep",
    }


def test_function_arguments_bind_semantic_alias_from_source_runlog_provenance() -> None:
    assert bind_function_arguments_to_task_params(
        {"clipboard_text": "1234 Elm St, Springfield, IL", "static": "keep"},
        {"clipboard_content": "Acme Corp, Suite 200", "seed": 113},
        {
            "clipboard_content": "1234 Elm St, Springfield, IL",
            "seed": 111,
        },
    ) == {
        "clipboard_text": "Acme Corp, Suite 200",
        "static": "keep",
    }


def test_function_arguments_do_not_guess_ambiguous_value_provenance() -> None:
    assert bind_function_arguments_to_task_params(
        {"query": "same"},
        {"first": "target-a", "second": "target-b"},
        {"first": "same", "second": "same"},
    ) == {"query": "same"}


def test_omniflow_e2e_command_forwards_oob_backend_to_child(monkeypatch, tmp_path: Path) -> None:
    item = CanonicalRunLog(
        task="BrowserDraw",
        goal="Draw a shape",
        params={},
        source_run_log=tmp_path / "source.run_log.json",
        replay_seed=111,
        step_count=1,
        meta={},
    )
    monkeypatch.setenv("OMNIFLOW_ANDROIDWORLD_CONTROL_BACKEND", "oob")

    spec = build_task_command(
        item,
        android_world_root=tmp_path / "android-world",
        output_root=tmp_path / "data",
        method_name="omniflow",
        agent_name="omniflow",
        device_label="fold5564",
        serial="emulator-5564",
        console_port=5564,
        store_path=tmp_path / "function_store.json",
        omnitransfer_root=tmp_path / "OmniTransfer",
    )

    assert spec.env["OMNIFLOW_ANDROIDWORLD_CONTROL_BACKEND"] == "oob"
    assert spec.metadata["control_backend"] == "oob_control"
    assert spec.metadata["action_backend"] == "oob_control"
    assert spec.metadata["native_androidworld_agent_io"] is False


def test_mobilegpt_server_uses_sealed_source_manifest_for_episode_memory(
    tmp_path: Path,
) -> None:
    mobilegpt_root = tmp_path / "MobileGPT"
    (mobilegpt_root / "Server" / "memory").mkdir(parents=True)
    (mobilegpt_root / "Server" / "main.py").write_text("print('server')\n")
    memory_root = tmp_path / "episode" / "mobilegpt_memory"
    memory_root.mkdir(parents=True)
    (memory_root / "tasks.csv").write_text("name,app\nCameraTakePhoto,camera\n")
    source_manifest = tmp_path / "source" / "mobilegpt_memory_manifest.json"
    source_manifest.parent.mkdir()
    source_manifest.write_text(
        json.dumps({"source_stats": {"embedding_models": ["GLM-Embedding-2"]}})
    )

    spec = build_mobilegpt_server_command(
        "server",
        mobilegpt_root=mobilegpt_root,
        mobilegpt_memory_root=memory_root,
        mobilegpt_memory_manifest=source_manifest,
        target_task_name="CameraTakePhoto",
        repo_root=tmp_path,
    )

    assert spec.env["MOBILEGPT_EMBEDDING_MODEL"] == "GLM-Embedding-2"
    assert spec.env["MOBILEGPT_TARGET_TASK_NAME"] == "CameraTakePhoto"
    assert "MOBILEGPT_MEMORY_SIMILARITY_THRESHOLD" not in spec.env
    assert "MOBILEGPT_TARGET_MEMORY_THRESHOLD" not in spec.env
    assert "MOBILEGPT_MEMORY_REUSE_STRICT" not in spec.env
    assert spec.env["MOBILEGPT_THINKING"] == "disabled"
    assert spec.env["MOBILEGPT_MAX_TOKENS"] == "512"
    assert spec.env["MOBILEGPT_LIST_MAX_TOKENS"] == "512"
    assert spec.env["MOBILEGPT_REQUEST_TIMEOUT_SEC"] == "60"


def test_mobilegpt_server_keeps_memory_app_separate_from_installed_package() -> None:
    assert _mobilegpt_server_task_app("markor", "net.gsantner.markor") == "markor"
    assert _mobilegpt_server_task_app("", "net.gsantner.markor") == "net.gsantner.markor"


def test_e2e_command_rejects_direct_function_for_non_omniflow_agent(
    tmp_path: Path,
) -> None:
    item = CanonicalRunLog(
        task="BrowserDraw",
        goal="Draw a shape",
        params={},
        source_run_log=tmp_path / "source.run_log.json",
        replay_seed=111,
        step_count=1,
        meta={},
    )

    with pytest.raises(ValueError, match="direct_function_requires_omniflow_agent"):
        build_task_command(
            item,
            agent_name="official:t3a_gpt4",
            function_id="complete_task",
        )


def test_dry_run_has_fixed_task_method_device_schedule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    args.dry_run = True
    monkeypatch.setattr(
        "src.experiment.run_tasks.load_data_index",
        lambda _: {"canonical": {"source_run_logs": {}, "function_stores": {}}},
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks.registered_result_plan_from_memory",
        lambda **_: {
            "completed": [],
            "pending": [
                (method, device[0]) for method in METHODS for device in DEVICES
            ],
        },
    )

    plan = run_pipeline(args)

    assert len(plan["pending"]) == len(METHODS) * len(DEVICES)
    assert plan["source_seed"] == SOURCE_SEED == 111
    assert plan["evaluation_seed"] == TASK_SEED == 113
    assert plan["methods"] == list(METHODS)
    assert plan["devices"] == [list(device) for device in DEVICES]
    assert plan["schedule"] == {
        device[0]: list(METHODS) for device in DEVICES
    }
    assert MAX_FALLBACK_STEPS == 5
    assert SOURCE_MAX_STEPS == 30
    assert MAX_STEPS == 20
    assert SOURCE_DEVICE == ("source5560", "emulator-5560", 5560)
    assert plan["writes"] is False


def test_e2e_selection_runs_only_one_method_and_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    args.e2e_method = "omniflow"
    args.e2e_device = DEVICES[1]
    args.e2e_source_seed = SOURCE_SEED
    args.e2e_evaluation_seed = TASK_SEED
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "src.experiment.run_tasks._concluded_results",
        lambda *_: set(),
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks.record_result_outcome",
        lambda **_: tmp_path / "outcome.json",
    )

    def runner(command: list[str], **kwargs: object) -> dict[str, object]:
        environment = kwargs["environment"]
        calls.append(
            (
                str(environment["OMNIFLOW_ANDROIDWORLD_METHOD"]),
                str(environment["OMNIFLOW_ANDROIDWORLD_DEVICE"]),
            )
        )
        return {"returncode": 0, "timed_out": False, "wall_sec": 0}

    run_target_workers(
        args=args,
        deadline=Deadline(10),
        attempt_id="attempt-test",
        attempt_root=tmp_path / "attempt",
        outcomes_root=tmp_path / "outcomes",
        store_path=tmp_path / "store.json",
        mobilegpt_memory=None,
        appagent_memory=None,
        blocked_methods={},
        command_runner=runner,
    )

    assert calls == [("omniflow", "fold5564:emulator-5564:5564")]


def test_e2e_selection_accepts_method_and_device_lists_or_all() -> None:
    selected = SimpleNamespace(
        e2e_method="omniflow,appagent",
        e2e_device=(
            "small5554:emulator-5554:5554,"
            "fold5564:emulator-5564:5564"
        ),
    )
    assert _e2e_methods(selected) == ("omniflow", "appagent")
    assert [device[0] for device in _e2e_devices(selected)] == [
        "small5554",
        "fold5564",
    ]
    all_selected = SimpleNamespace(e2e_method="all", e2e_device="all")
    assert _e2e_methods(all_selected) == METHODS
    assert _e2e_devices(all_selected) == DEVICES


def test_autodroid_is_explicit_supplemental_only() -> None:
    selected = SimpleNamespace(
        e2e_method="autodroid",
        e2e_device="all",
    )

    assert _e2e_methods(selected) == SUPPLEMENTAL_METHODS == ("autodroid",)
    assert _e2e_devices(selected) == SUPPLEMENTAL_DEVICES
    assert [device[0] for device in SUPPLEMENTAL_DEVICES] == [
        "autodroidsmall5554",
        "autodroidfold5564",
        "autodroidandroidworld5594",
    ]
    assert _supplemental_outcomes_root(
        SimpleNamespace(
            e2e_method="autodroid",
            results_root=Path("/tmp/omniflow-results"),
        )
    ) == Path("/tmp/omniflow-results") / SUPPLEMENTAL_RESULTS_NAMESPACE

    with pytest.raises(ValueError, match="supplemental_method_must_run_alone"):
        _e2e_methods(SimpleNamespace(e2e_method="omniflow,autodroid"))
    with pytest.raises(ValueError, match="device_invalid"):
        _e2e_devices(
            SimpleNamespace(
                e2e_method="autodroid",
                e2e_device="small5554:emulator-5554:5554",
            )
        )

    all_selected = SimpleNamespace(e2e_method="all", e2e_device="all")
    assert _e2e_methods(all_selected) == METHODS
    assert _e2e_devices(all_selected) == DEVICES


def test_autodroid_task_policy_uses_official_goal_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    official_root = tmp_path / "autodroid"
    memory_root = tmp_path / "memory"
    (official_root / "droidbot").mkdir(parents=True)
    (official_root / "droidbot" / "start.py").write_text("", encoding="utf-8")
    memory_root.mkdir()
    (memory_root / "memory_manifest.json").write_text(
        '{"format":"autodroid-droidbot-memory-manifest-v1","apps":[{"name":"camera"}],"device":{}}\n',
        encoding="utf-8",
    )
    item = CanonicalRunLog(
        task="CameraTakePhoto",
        goal="Take a photo",
        params={"camera": "rear"},
        source_run_log=tmp_path / "source.json",
        replay_seed=111,
        step_count=0,
        meta={},
    )
    monkeypatch.setenv("OMNIFLOW_ANDROID_SDK_ROOT", "/opt/android-sdk")
    monkeypatch.setenv("ANDROID_ADB_SERVER_PORT", "5038")

    spec = build_autodroid_command(
        item,
        method_name="autodroid",
        target=DeviceTarget("autodroid9207", "emulator-5590", 5590),
        android_world_root=tmp_path / "android-world",
        output_root=tmp_path / "output",
        autodroid_root=official_root,
        autodroid_memory_root=memory_root,
        autodroid_policy="task",
        max_steps=20,
        timeout_sec=100,
        task_random_seed=113,
        fixed_task_seed=True,
        fixed_task_params=True,
        task_params_override=item.params,
        perform_emulator_setup=False,
        adb_path="/usr/bin/adb",
        repo_root=tmp_path,
    )

    assert "--policy" in spec.argv
    assert spec.argv[spec.argv.index("--policy") + 1] == "task"
    assert spec.argv[spec.argv.index("--goal") + 1] == "Take a photo"
    assert spec.metadata["official_policy"] == "task"
    assert spec.env["ANDROID_SDK_ROOT"] == "/opt/android-sdk"
    assert spec.env["ANDROID_HOME"] == "/opt/android-sdk"
    assert spec.env["ANDROID_ADB_SERVER_PORT"] == "5038"


def test_e2e_function_check_creates_and_validates_missing_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    args.ensure_function = True
    args.formal_model = "GLM-5.1"
    source_path = tmp_path / "source.run_log.json"
    source_path.write_text(
        json.dumps(
            androidworld_run_log(
                [{"action_type": "click", "x": 1, "y": 2}],
                task_name=args.task,
            )
        ),
        encoding="utf-8",
    )
    indexed: dict[str, object] = {"value": None}
    calls: list[dict[str, object]] = []

    def load_index(_path: Path) -> dict[str, object]:
        value = indexed["value"]
        return {
            "canonical": {
                "function_stores": {args.task: value} if value else {},
            }
        }

    def writer(run_log: Path, store_path: Path, **kwargs: object) -> dict[str, object]:
        calls.append({"run_log": run_log, "store_path": store_path, **kwargs})
        store_path.parent.mkdir(parents=True)
        store_path.write_text("{}", encoding="utf-8")
        indexed["value"] = {
            "store_path": str(store_path),
            "source_run_log_path": str(source_path),
            "source_run_log_sha256": hashlib.sha256(
                source_path.read_bytes()
            ).hexdigest(),
            "source_calls": [
                {"function_id": "complete", "arguments": {}}
            ],
        }
        return {"enhanced": True, "function_ids": ["complete"]}

    monkeypatch.setattr("src.experiment.run_tasks.load_data_index", load_index)
    monkeypatch.setattr("src.experiment.run_tasks.save_function", writer)
    monkeypatch.setattr(
        "src.experiment.run_tasks._function_enhancement_transport",
        lambda **_: (lambda _prompt, _tool: "{}"),
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks.refresh_data_index_from_pointer",
        lambda **_: {},
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks.validate_omniflow_transfer_assets",
        lambda *_args, **_kwargs: {"complete": True, "required_state_count": 1},
    )

    function_store, phase = prepare_function_asset(
        args=args,
        source_path=source_path,
        run_log={},
        attempt_root=tmp_path / "attempt",
        deadline=Deadline(60),
    )

    assert len(calls) == 1
    assert calls[0]["run_log"] == source_path
    assert calls[0]["enhance"] is True
    assert function_store["store_path"] == str(calls[0]["store_path"])
    assert phase["status"] == "created"
    assert phase["enhanced"] is True
    assert phase["transfer_audit"]["complete"] is True


def test_function_store_reuses_its_own_valid_source_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    args.task = "BrowserDraw"
    canonical_source = tmp_path / "canonical.run_log.json"
    canonical_source.write_text(
        json.dumps(
            androidworld_run_log(
                [{"action_type": "click", "x": 1, "y": 2}],
                task_name=args.task,
                run_id="canonical",
            )
        ),
        encoding="utf-8",
    )
    function_source = tmp_path / "function-source.run_log.json"
    function_source.write_text(
        json.dumps(
            androidworld_run_log(
                [{"action_type": "click", "x": 3, "y": 4}],
                task_name=args.task,
                run_id="function-source",
            )
        ),
        encoding="utf-8",
    )
    store_path = tmp_path / "function_store.json"
    store_path.write_text("{}", encoding="utf-8")
    function_store = {
        "store_path": str(store_path),
        "source_run_log_path": str(function_source),
        "source_run_log_sha256": hashlib.sha256(
            function_source.read_bytes()
        ).hexdigest(),
        "source_calls": [{"function_id": "complete", "arguments": {}}],
    }
    monkeypatch.setattr(
        "src.experiment.run_tasks._canonical_function_store",
        lambda *_args, **_kwargs: function_store,
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks.validate_omniflow_transfer_assets",
        lambda *_args, **_kwargs: {"complete": True, "required_state_count": 1},
    )

    reused, phase = prepare_function_asset(
        args=args,
        source_path=canonical_source,
        run_log=json.loads(canonical_source.read_text(encoding="utf-8")),
        attempt_root=tmp_path / "attempt",
        deadline=Deadline(60),
    )

    assert reused is function_store
    assert phase["status"] == "reused"


def test_mobilegpt_preparation_is_an_internal_pipeline_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    args.formal_model = "GLM-5.1"
    args.memory_index.parent.mkdir(parents=True)
    source_index = args.memory_index.parent / "source.json"
    source_index.write_text("{}", encoding="utf-8")
    args.memory_index.write_text(
        json.dumps({"source_index": str(source_index)}),
        encoding="utf-8",
    )
    calls = 0

    def canonical(**_: object) -> dict[str, str] | None:
        nonlocal calls
        calls += 1
        if calls == 1:
            return None
        return {"memory_root": str(tmp_path / "registered-mobilegpt")}

    captured: list[str] = []

    def run(command: list[str], **_: object) -> dict[str, object]:
        captured.extend(command)
        return {"returncode": 0, "timed_out": False, "wall_sec": 0.1}

    monkeypatch.setattr(
        "src.experiment.run_tasks.canonical_prepared_memory_from_index",
        canonical,
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks.run_logged_command",
        run,
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks._validate_prepared_mobilegpt_memory",
        lambda *_args, **_kwargs: {
            "task_name": args.task,
            "runlog_teacher_alignment": True,
        },
    )

    memory_root, phase = prepare_mobilegpt_memory(
        args=args,
        attempt_root=tmp_path / "attempt",
        deadline=Deadline(60),
    )

    assert captured[:4] == [
        str(args.python_bin),
        "-m",
        "src.experiment.mobilegpt_source",
        "prepare",
    ]
    assert "bash" not in captured
    assert "--prepare-mobilegpt-memory" not in captured
    assert str(args.memory_index) in captured
    assert memory_root == tmp_path / "registered-mobilegpt"
    assert phase["status"] == "created"


def test_mobilegpt_preparation_rejects_legacy_unaligned_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    memory_root = tmp_path / "legacy" / "memory"
    memory_root.mkdir(parents=True)
    (memory_root.parent / "mobilegpt_memory_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "omniflow.mobilegpt.memory.v2",
                "source_method": "mobilegpt_official_learning_memory",
                "task_name": args.task,
                "provenance": {
                    "native_mobilegpt_learning": True,
                    "official_authoring_session": True,
                    "official_prompt_extension": False,
                    "runlog_teacher_alignment": False,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "OMNIFLOW_MOBILEGPT_SOURCE_MEMORY_ROOT",
        str(memory_root),
    )

    with pytest.raises(ValueError, match="mobilegpt_source_memory_not_authoritative"):
        prepare_mobilegpt_memory(
            args=args,
            attempt_root=tmp_path / "attempt",
            deadline=Deadline(60),
        )


def test_source_device_uses_protocol_avd() -> None:
    parser = build_parser()
    destinations = {action.dest for action in parser._actions}

    assert parser.get_default("source_avd") == SOURCE_AVD
    assert parser.get_default("source_device") == SOURCE_DEVICE
    assert "bmoca_cell_timeout_sec" not in destinations
    assert "enhancement_timeout_sec" not in destinations
    assert BMOCA_RESULT_TIMEOUT_SEC == 600
    assert FUNCTION_ENHANCEMENT_TIMEOUT_SEC == 300


def test_source_device_accepts_an_isolated_console_port() -> None:
    assert _parse_source_device("source5570:emulator-5570:5570") == (
        "source5570",
        "emulator-5570",
        5570,
    )


def test_bmoca_method_launches_ten_isolated_overlapping_results(
    tmp_path: Path,
) -> None:
    lock = threading.Lock()
    active = 0
    maximum = 0
    environments: list[dict[str, str]] = []
    commands: list[list[str]] = []

    def runner(command: list[str], **kwargs: object) -> dict[str, object]:
        nonlocal active, maximum
        environment = dict(kwargs["environment"])
        output = Path(environment["OMNIFLOW_BMOCA_OUTPUT_PATH"])
        with lock:
            active += 1
            maximum = max(maximum, active)
            environments.append(environment)
            commands.append(command)
        time.sleep(0.05)
        output.mkdir(parents=True)
        environment_id = environment["OMNIFLOW_BMOCA_SINGLE_ENVIRONMENT_ID"]
        (output / "summary.json").write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "environment_id": environment_id,
                            "emulator_serial": (
                                "emulator-"
                                + environment["OMNIFLOW_BMOCA_EMULATOR_CONSOLE_PORT"]
                            ),
                            "official_success": True,
                            "method_success": True,
                            "actions_executed": 1,
                            "model_calls": 0,
                            "fallback_steps": 0,
                            "run_log_evidence": {
                                "target_run_log_path": str(output / "target.run_log.json")
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        with lock:
            active -= 1
        return {
            "returncode": 0,
            "wall_sec": 0.05,
            "process_pid": 10000 + int(environment_id),
            "started_at": "2026-08-18T00:00:00+00:00",
            "finished_at": "2026-08-18T00:00:01+00:00",
        }

    args = SimpleNamespace(
        repo=tmp_path / "repo",
        script=tmp_path / "repo/scripts/exp/run_androidworld.sh",
        python_bin=tmp_path / "python",
        omnitransfer_root=tmp_path / "OmniTransfer",
        bmoca_root=tmp_path / "BMoCA",
        bmoca_android_env_root=tmp_path / "AndroidEnv",
        android_sdk_root=tmp_path / "sdk",
    )
    rows = _run_bmoca_method_results(
        args=args,
        task="clock/create_alarm_at_06:30_am",
        method="skilldroid_replay",
        store_path=tmp_path / "store.json",
        memory_path=tmp_path / "skilldroid-memory.json",
        task_root=tmp_path / "task",
        avd_homes={
            str(value): tmp_path / f"avd/env_{value}" for value in range(100, 110)
        },
        command_runner=runner,
    )

    assert len(rows) == 10
    assert maximum == 10
    assert _max_live_bmoca_results(rows) == 10
    assert len({row["process_pid"] for row in rows}) == 10
    assert len({env["OMNIFLOW_BMOCA_AVD_HOME"] for env in environments}) == 10
    assert len({env["OMNIFLOW_BMOCA_APPIUM_PORT"] for env in environments}) == 10
    assert len({env["OMNIFLOW_BMOCA_EMULATOR_CONSOLE_PORT"] for env in environments}) == 10
    assert all(
        command[:4]
        == [
            str(args.python_bin),
            "-m",
            "src.integrations.android_world.run_episode",
            "--environment",
        ]
        for command in commands
    )
    assert all("bash" not in command for command in commands)
    assert all("OPENAI_API_KEY" not in env for env in environments)
    assert all("OMNIFLOW_ENV_FILE" not in env for env in environments)


@pytest.mark.parametrize(
    ("change", "qualified"),
    [
        ({}, True),
        ({"status": "method_failure"}, False),
        ({"official_success": False}, False),
        ({"method_success": False}, False),
        ({"model_calls": 1}, False),
        ({"fallback_steps": 1}, False),
    ],
)
def test_bmoca_source_replay_is_a_hard_gate(change, qualified) -> None:
    row = {
        "status": "success",
        "official_success": True,
        "method_success": True,
        "model_calls": 0,
        "fallback_steps": 0,
        **change,
    }
    assert _bmoca_source_replay_qualified(row) is qualified


def test_bmoca_pipeline_stops_task_after_failed_omniflow_source_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = "clock/create_alarm_at_06:30_am"
    source = tmp_path / "source.run_log.json"
    source.write_text("{}", encoding="utf-8")
    calls: list[tuple[str, tuple[str, ...]]] = []
    cloned: list[str] = []
    monkeypatch.setattr(
        "src.experiment.run_tasks._bmoca_manifest_tasks",
        lambda *_: ([task], {task: source}),
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks._bmoca_avd_names",
        lambda *_: {str(value): f"env{value}" for value in range(100, 110)},
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks._clone_bmoca_avd_home",
        lambda **kwargs: cloned.append(kwargs["avd_name"])
        or kwargs["target_home"],
    )
    store = tmp_path / "store.json"
    monkeypatch.setattr(
        "src.experiment.run_tasks._save_bmoca_function_once",
        lambda **_: (store, {"enhanced": True}),
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks._prepare_bmoca_mobilegpt_memory",
        lambda **_: (tmp_path / "mobilegpt", {"status": "prepared"}),
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks._prepare_bmoca_skilldroid_memory",
        lambda **_: (tmp_path / "skilldroid.json", {"status": "prepared"}),
    )

    def run_results(**kwargs: object) -> list[dict[str, object]]:
        method = str(kwargs["method"])
        environments = tuple(kwargs["environment_ids"])
        calls.append((method, environments))
        if method != "ours_replay":
            return [
                {
                    "task": task,
                    "method": method,
                    "environment_id": environment_id,
                    "status": "success",
                    "official_success": True,
                    "method_success": True,
                    "actions_executed": 1,
                    "model_calls": 0,
                    "fallback_steps": 0,
                    "error": "",
                }
                for environment_id in environments
            ]
        return [
            {
                "task": task,
                "method": method,
                "environment_id": "100",
                "status": "method_failure",
                "official_success": False,
                "method_success": False,
                "actions_executed": 1,
                "model_calls": 0,
                "fallback_steps": 0,
                "error": "official_validator_failed",
            }
        ]

    monkeypatch.setattr(
        "src.experiment.run_tasks._run_bmoca_method_results",
        run_results,
    )
    summary = run_bmoca_pipeline(
        SimpleNamespace(
            bmoca_corpus_manifest=tmp_path / "manifest.json",
            task="all",
            output_root=tmp_path / "campaign",
            bmoca_root=tmp_path / "BMoCA",
            bmoca_avd_home=tmp_path / "avd",
        )
    )

    assert calls == [
        ("ours_replay", ("100",)),
    ]
    assert cloned == ["env100"]
    assert summary["status_counts"] == {
        "method_failure": 1,
        "prep_failed": 29,
    }


def test_bmoca_pipeline_does_not_clone_avds_before_enhancement_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = "clock/create_alarm_at_06:30_am"
    source = tmp_path / "source.run_log.json"
    source.write_text("{}", encoding="utf-8")
    cloned: list[str] = []
    monkeypatch.setattr(
        "src.experiment.run_tasks._bmoca_manifest_tasks",
        lambda *_: ([task], {task: source}),
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks._bmoca_avd_names",
        lambda *_: {str(value): f"env{value}" for value in range(100, 110)},
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks._clone_bmoca_avd_home",
        lambda **kwargs: cloned.append(kwargs["avd_name"]),
    )

    def fail_enhancement(**_: object) -> None:
        raise ValueError("invalid enhancement")

    monkeypatch.setattr(
        "src.experiment.run_tasks._save_bmoca_function_once",
        fail_enhancement,
    )

    summary = run_bmoca_pipeline(
        SimpleNamespace(
            bmoca_corpus_manifest=tmp_path / "manifest.json",
            task="all",
            output_root=tmp_path / "campaign",
            bmoca_root=tmp_path / "BMoCA",
            bmoca_avd_home=tmp_path / "avd",
        )
    )

    assert cloned == []
    assert summary["status_counts"] == {"prep_failed": 30}


def test_bmoca_pipeline_runs_remaining_results_only_after_source_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = "clock/create_alarm_at_06:30_am"
    source = tmp_path / "source.run_log.json"
    source.write_text("{}", encoding="utf-8")
    calls: list[tuple[str, tuple[str, ...]]] = []
    cloned: list[str] = []
    monkeypatch.setattr(
        "src.experiment.run_tasks._bmoca_manifest_tasks",
        lambda *_: ([task], {task: source}),
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks._bmoca_avd_names",
        lambda *_: {str(value): f"env{value}" for value in range(100, 110)},
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks._clone_bmoca_avd_home",
        lambda **kwargs: cloned.append(kwargs["avd_name"])
        or kwargs["target_home"],
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks._save_bmoca_function_once",
        lambda **_: (tmp_path / "store.json", {"enhanced": True}),
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks._prepare_bmoca_mobilegpt_memory",
        lambda **_: (tmp_path / "mobilegpt", {"status": "prepared"}),
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks._prepare_bmoca_skilldroid_memory",
        lambda **_: (tmp_path / "skilldroid.json", {"status": "prepared"}),
    )

    def run_results(**kwargs: object) -> list[dict[str, object]]:
        method = str(kwargs["method"])
        environments = tuple(kwargs["environment_ids"])
        calls.append((method, environments))
        return [
            {
                "task": task,
                "method": method,
                "environment_id": environment_id,
                "status": "success",
                "official_success": True,
                "method_success": True,
                "actions_executed": 1,
                "model_calls": 0,
                "fallback_steps": 0,
                "error": "",
            }
            for environment_id in environments
        ]

    monkeypatch.setattr(
        "src.experiment.run_tasks._run_bmoca_method_results",
        run_results,
    )
    summary = run_bmoca_pipeline(
        SimpleNamespace(
            bmoca_corpus_manifest=tmp_path / "manifest.json",
            task="all",
            output_root=tmp_path / "campaign",
            bmoca_root=tmp_path / "BMoCA",
            bmoca_avd_home=tmp_path / "avd",
        )
    )

    assert calls == [
        ("ours_replay", ("100",)),
        ("ours_replay", tuple(str(value) for value in range(101, 110))),
        ("mobilegpt_replay", tuple(str(value) for value in range(100, 110))),
        ("skilldroid_replay", tuple(str(value) for value in range(100, 110))),
    ]
    assert cloned == [f"env{value}" for value in range(100, 110)]
    assert summary["status_counts"] == {"success": 30}


def test_bmoca_offline_enhancement_calls_v2_compiler_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.run_log.json"
    source.write_text("{}", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def writer(run_log: Path, output_root: Path, **kwargs: object) -> dict[str, object]:
        calls.append({"run_log": run_log, "output_root": output_root, **kwargs})
        output_root.mkdir(parents=True)
        store_path = output_root / "store.json"
        store_path.write_text("{}", encoding="utf-8")
        transfer = store_path.with_name("transfer_states.json")
        transfer.write_text("{}", encoding="utf-8")
        return {
            "enhanced": True,
            "function_ids": ["complete"],
            "store_path": str(store_path),
            "transfer_state_catalog": str(transfer),
        }

    monkeypatch.setattr("src.experiment.run_tasks.compile_function_v2", writer)
    args = SimpleNamespace(formal_model="GLM-5.1")

    _, report = _save_bmoca_function_once(
        args=args,
        task="clock/create_alarm_at_06:30_am",
        source_run_log=source,
        task_root=tmp_path / "task",
    )

    assert len(calls) == 1
    assert calls[0]["enhance"] is True
    assert calls[0]["model"] == "GLM-5.1"
    assert report["compile_function_calls"] == 1


def test_bmoca_enhancement_failure_preserves_stage_and_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.run_log.json"
    source.write_text("{}", encoding="utf-8")

    def fail(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise TimeoutError("endpoint did not answer")

    monkeypatch.setattr("src.experiment.run_tasks.compile_function_v2", fail)
    args = SimpleNamespace(formal_model="GLM-5.1")
    task_root = tmp_path / "task"

    with pytest.raises(TimeoutError, match="endpoint did not answer"):
        _save_bmoca_function_once(
            args=args,
            task="clock/create_alarm_at_06:30_am",
            source_run_log=source,
            task_root=task_root,
        )

    failure = json.loads(
        (task_root / "enhancement_failure.json").read_text(encoding="utf-8")
    )
    assert failure["status"] == "failed"
    assert failure["compile_function_calls"] == 1
    assert failure["model_calls"] == 0
    assert failure["error"] == "TimeoutError: endpoint did not answer"


@pytest.mark.skip(reason="v3 draft-edit transport was removed with v2 restoration")
def test_bmoca_enhancement_uses_the_shared_draft_edit_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    endpoint: dict[str, object] = {}

    class Completions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                ),
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            tool_calls=[
                                SimpleNamespace(
                                    function=SimpleNamespace(
                                        name="edit_function_draft",
                                        arguments=(
                                            '{"complete_function":{'
                                            '"function_id":"open_item",'
                                            '"name":"Open item",'
                                            '"description":"Open the visible item."}'
                                        ),
                                    )
                                )
                            ]
                        )
                    )
                ],
            )

    class OpenAI:
        def __init__(self, **_: object) -> None:
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=OpenAI))

    def resolve_endpoint(**kwargs: object) -> tuple[str, str]:
        endpoint.update(kwargs)
        return "key", "https://example.invalid/v1"

    monkeypatch.setattr(
        "src.experiment.run_tasks.resolve_openai_compatible_config",
        resolve_endpoint,
    )
    usage = {
        "model_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    complete = _function_enhancement_transport(
        model="GLM-5.1",
        timeout_sec=180,
        usage=usage,
    )
    tool = function_authoring_tool(stage="split")

    assert '"function_id":"open_item"' in complete("Edit draft", tool)
    assert endpoint == {
        "profile": "llmthu",
        "base_url": "https://llmapi.paratera.com/v1",
    }
    assert captured["tools"] == [tool]
    assert captured["max_completion_tokens"] == 4096
    assert captured["reasoning_effort"] == "none"
    assert captured["tool_choice"] == {
        "type": "function",
        "function": {"name": "edit_function_draft"},
    }
    assert usage == {
        "model_calls": 1,
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
    }


@pytest.mark.skip(reason="v3 draft-edit transport was removed with v2 restoration")
def test_function_enhancement_accepts_json_message_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Completions:
        def create(self, **_: object) -> SimpleNamespace:
            return SimpleNamespace(
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                ),
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                "Here is the requested JSON:\n"
                                '{"complete_function":{'
                                '"function_id":"open_item",'
                                '"name":"Open item",'
                                '"description":"Open the visible item."}'
                                '}'
                            ),
                            tool_calls=[],
                        )
                    )
                ],
            )

    class OpenAI:
        def __init__(self, **_: object) -> None:
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=OpenAI))
    monkeypatch.setattr(
        "src.experiment.run_tasks.resolve_openai_compatible_config",
        lambda **_: ("key", "https://example.invalid/v1"),
    )
    usage = {
        "model_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    complete = _function_enhancement_transport(
        model="GLM-4.6V",
        timeout_sec=180,
        usage=usage,
    )

    assert complete("Edit draft", function_authoring_tool(stage="split"))
    assert usage["model_calls"] == 1
    assert usage["total_tokens"] == 30


def test_resolve_args_preserves_symlinked_virtualenv_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    args.emulator_bin = tmp_path / "emulator"
    args.runtime_preflight = tmp_path / "repo" / "src" / "experiment" / "checks.py"
    for path in (
        args.script,
        args.memory_index,
        args.adb_path,
        args.emulator_bin,
        args.runtime_preflight,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    real_python = tmp_path / "runtime" / "python"
    real_python.parent.mkdir(parents=True)
    real_python.touch()
    args.python_bin.parent.mkdir(parents=True, exist_ok=True)
    args.python_bin.symlink_to(real_python)
    for path in (
        args.asset_root,
        args.results_root,
        args.output_root,
        args.android_world_root,
        args.mobilegpt_root,
        args.appagent_root,
    ):
        path.mkdir(parents=True, exist_ok=True)
    canonical_transfer = tmp_path / "Projects" / "Omni" / "OmniTransfer"
    canonical_transfer.mkdir(parents=True)
    args.omnitransfer_root = canonical_transfer
    args.appagent_memory_root = None
    args.source_avd = "SmallPhone"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    resolved = _resolve_args(args)

    assert resolved.python_bin == args.python_bin.absolute()
    assert resolved.python_bin.is_symlink()


def test_result_environment_uses_orchestrator_budget_and_child_guard(
    tmp_path: Path,
) -> None:
    from src.experiment.run_tasks import _result_environment

    args = _args(tmp_path)
    args.max_steps = 7
    args.max_fallback_steps = 2
    environment = _result_environment(
        args=args,
        attempt_id="attempt-test",
        attempt_root=tmp_path / "attempt",
        method="t3a_hint",
        device=DEVICES[0],
        store_path=None,
        mobilegpt_memory=None,
        appagent_memory=None,
    )

    assert environment["OMNIFLOW_BATCH_CHILD"] == "1"
    assert environment["OMNIFLOW_BATCH_ATTEMPT_ID"] == "attempt_001"
    assert environment["OMNIFLOW_ANDROIDWORLD_MAX_STEPS"] == "7"
    assert environment["OMNIFLOW_ANDROIDWORLD_MAX_FALLBACK_STEPS"] == "2"
    assert environment["OMNIFLOW_ANDROIDWORLD_ARCHIVE_ROOT"] == str(
        args.results_root / "androidworld"
    )
    assert "OMNIFLOW_ANDROIDWORLD_STORE_PATH" not in environment


def test_formal_timeout_covers_frozen_steps_and_validator_flush() -> None:
    assert EPISODE_TIMEOUT_SEC == MAX_STEPS * STEP_TIMEOUT_SEC + 300
    assert EPISODE_TIMEOUT_SEC > 600


def test_source_device_ready_requires_exact_avd_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = SimpleNamespace(source_avd="SmallPhone", source_device=SOURCE_DEVICE)

    def adb_output(_args: object, *command: str) -> str:
        if command[-3:] == ("emu", "avd", "name"):
            return "AndroidWorldAvd\nOK"
        if command[-1] == "get-state":
            return "device"
        if command[-2:] == ("getprop", "sys.boot_completed"):
            return "1"
        return ""

    monkeypatch.setattr(
        "src.experiment.run_tasks._adb_output",
        adb_output,
    )

    assert _source_device_ready(args) is False


def test_source_device_is_cold_restarted_when_already_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    args.source_avd = "SmallPhone"
    args.emulator_bin = tmp_path / "emulator"
    args.emulator_gpu = "swiftshader_indirect"
    args.runtime_preflight = tmp_path / "checks.py"
    args.task = "ContactsAddContact"
    adb_calls: list[tuple[str, ...]] = []
    preflight_commands: list[list[str]] = []
    preflight_environments: list[dict[str, str]] = []

    def adb_output(_args: object, *command: str) -> str:
        adb_calls.append(command)
        if command == ("devices",):
            return "List of devices attached\nemulator-5560\tdevice"
        return ""

    monkeypatch.setattr(
        "src.experiment.run_tasks._adb_output",
        adb_output,
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks._source_device_ready",
        lambda _args: True,
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks._read_object",
        lambda _path: {"source_index": "source-index.json"},
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks.run_logged_command",
        lambda command, **kwargs: (
            preflight_commands.append(command)
            or preflight_environments.append(kwargs["environment"])
            or {"returncode": 0}
        ),
    )
    monkeypatch.setattr(
        "src.experiment.run_process.subprocess.Popen",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks.time.sleep",
        lambda _seconds: None,
    )

    result = ensure_source_device(
        args=args,
        attempt_root=tmp_path / "attempt",
        deadline=Deadline(120),
    )

    # One kill is the intentional cold restart of the pre-existing source;
    # ensure_source_device must not kill the newly launched source after
    # preflight because qualification follows in the same pipeline.
    assert adb_calls.count(("-s", "emulator-5560", "emu", "kill")) == 1
    assert result["launched"] is True
    assert result["kept_alive_for_pipeline"] is True
    assert "--require-contacts-ready" not in preflight_commands[0]
    assert "--expected-tasks" not in preflight_commands[0]
    assert preflight_commands[0][preflight_commands[0].index("--android-world-root") + 1] == str(
        args.android_world_root
    )
    assert str(args.android_world_root) in preflight_environments[0]["PYTHONPATH"]


def test_source_device_reports_emulator_process_exit_immediately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    args.source_avd = "MissingSourceAvd"
    args.emulator_bin = tmp_path / "emulator"
    args.emulator_gpu = "swiftshader_indirect"

    class FailedProcess:
        pid = 42

        def poll(self) -> int:
            return 2

    monkeypatch.setattr(
        "src.experiment.run_tasks._adb_output",
        lambda *_args: "",
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks._source_device_ready",
        lambda _args: False,
    )
    monkeypatch.setattr(
        "src.experiment.run_process.subprocess.Popen",
        lambda *_args, **_kwargs: FailedProcess(),
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks.time.sleep",
        lambda _seconds: None,
    )

    with pytest.raises(RuntimeError, match="source_emulator_exited:2"):
        ensure_source_device(
            args=args,
            attempt_root=tmp_path / "attempt",
            deadline=Deadline(1),
        )


def test_mobilegpt_target_preflight_prepares_contacts_before_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    args.task = "ContactsAddContact"
    args.e2e_method = "mobilegpt"
    args.e2e_device = DEVICES[2]
    args.emulator_bin = tmp_path / "emulator"
    args.emulator_gpu = "swiftshader_indirect"
    args.runtime_preflight = tmp_path / "checks.py"
    calls: list[tuple[list[str], dict[str, object]]] = []

    def runner(command: list[str], **kwargs: object) -> dict[str, object]:
        calls.append((command, kwargs))
        return {"returncode": 0, "timed_out": False, "wall_sec": 0.01}

    monkeypatch.setattr("src.experiment.run_tasks.run_logged_command", runner)

    result = ensure_target_devices(
        args=args,
        attempt_root=tmp_path / "attempt",
        deadline=Deadline(120),
    )

    assert result["status"] == "ready"
    assert len(calls) == 2
    preflight, preflight_kwargs = calls[1]
    assert preflight[:2] == [str(args.python_bin), str(args.runtime_preflight)]
    assert preflight[preflight.index("--profile") + 1] == "mobilegpt"
    assert preflight[preflight.index("--serial") + 1] == DEVICES[2][1]
    assert "--require-contacts-ready" in preflight
    assert preflight[preflight.index("--source-task") + 1] == args.task
    assert str(args.android_world_root) in str(
        preflight_kwargs["environment"]["PYTHONPATH"]
    )


def test_target_workers_parallelize_devices_and_serialize_methods(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    calls: list[tuple[str, str, float, float]] = []
    commands: list[list[str]] = []
    completed: set[tuple[str, str]] = set()
    monkeypatch.setattr(
        "src.experiment.run_tasks._concluded_results",
        lambda *_: set(completed),
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks.concluded_result_keys",
        lambda **_: set(),
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks.record_result_outcome",
        lambda **_: tmp_path / "outcome.json",
    )

    def runner(command: list[str], **kwargs: object) -> dict[str, object]:
        environment = kwargs["environment"]
        assert isinstance(environment, dict)
        commands.append(command)
        method = str(environment["OMNIFLOW_ANDROIDWORLD_METHOD"])
        device = str(environment["OMNIFLOW_ANDROIDWORLD_DEVICE"]).split(":")[0]
        started = time.monotonic()
        time.sleep(0.03)
        finished = time.monotonic()
        calls.append((device, method, started, finished))
        completed.add((method, device))
        return {"returncode": 0, "timed_out": False, "wall_sec": 0.03}

    run_target_workers(
        args=args,
        deadline=Deadline(10),
        attempt_id="attempt-test",
        attempt_root=tmp_path / "attempt",
        outcomes_root=tmp_path / "outcomes",
        store_path=tmp_path / "store.json",
        mobilegpt_memory=tmp_path / "mobilegpt",
        appagent_memory=tmp_path / "appagent",
        blocked_methods={},
        command_runner=runner,
    )

    assert len(calls) == len(METHODS) * len(DEVICES)
    assert len(commands) == len(METHODS) * len(DEVICES)
    assert all(
        command[:4]
        == [str(args.python_bin), "-m", "src.experiment.run_task", "result"]
        for command in commands
    )
    assert all("bash" not in command for command in commands)
    for device in ("small5554", "fold5564"):
        rows = sorted((row for row in calls if row[0] == device), key=lambda row: row[2])
        assert [row[1] for row in rows] == list(METHODS)
        assert all(current[3] <= following[2] for current, following in zip(rows, rows[1:]))
    device_windows = [
        (min(row[2] for row in calls if row[0] == device),
         max(row[3] for row in calls if row[0] == device))
        for device, _, _ in DEVICES
    ]
    assert any(
        left_start < right_end and right_start < left_end
        for index, (left_start, left_end) in enumerate(device_windows)
        for right_start, right_end in device_windows[index + 1 :]
    )


def test_target_workers_fail_stop_after_pending_environment_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    calls: list[tuple[str, str]] = []
    recorded: list[dict[str, object]] = []
    monkeypatch.setattr(
        "src.experiment.run_tasks._concluded_results",
        lambda *_: set(),
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks.concluded_result_keys",
        lambda **_: set(),
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks.record_result_outcome",
        lambda **kwargs: recorded.append(kwargs) or tmp_path / "outcome.json",
    )

    def runner(command: list[str], **kwargs: object) -> dict[str, object]:
        environment = kwargs["environment"]
        assert isinstance(environment, dict)
        calls.append(
            (
                str(environment["OMNIFLOW_ANDROIDWORLD_DEVICE"]).split(":")[0],
                str(environment["OMNIFLOW_ANDROIDWORLD_METHOD"]),
            )
        )
        return {"returncode": 1, "timed_out": False, "wall_sec": 0.01}

    with pytest.raises(PipelinePhaseError, match="target_episode_environment_failure"):
        run_target_workers(
            args=args,
            deadline=Deadline(10),
            attempt_id="attempt-test",
            attempt_root=tmp_path / "attempt",
            outcomes_root=tmp_path / "outcomes",
            store_path=tmp_path / "store.json",
            mobilegpt_memory=tmp_path / "mobilegpt",
            appagent_memory=tmp_path / "appagent",
            blocked_methods={},
            command_runner=runner,
        )

    assert recorded
    assert all(row["status"] == "environment_failure" for row in recorded)
    for device in ("small5554", "fold5564"):
        assert len([call for call in calls if call[0] == device]) <= 1


def test_target_workers_continue_after_method_result_conclusion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    calls: list[tuple[str, str]] = []
    recorded: list[dict[str, object]] = []
    monkeypatch.setattr(
        "src.experiment.run_tasks._concluded_results",
        lambda *_: set(),
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks.concluded_result_keys",
        lambda **_: set(),
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks.record_result_outcome",
        lambda **kwargs: recorded.append(kwargs) or tmp_path / "outcome.json",
    )

    def runner(command: list[str], **kwargs: object) -> dict[str, object]:
        environment = kwargs["environment"]
        assert isinstance(environment, dict)
        method = str(environment["OMNIFLOW_ANDROIDWORLD_METHOD"])
        device = str(environment["OMNIFLOW_ANDROIDWORLD_DEVICE"]).split(":")[0]
        calls.append((device, method))
        if method == "mobilegpt":
            output_root = Path(str(environment["OMNIFLOW_ANDROIDWORLD_OUTPUT_PATH"]))
            output_root.mkdir(parents=True, exist_ok=True)
            (output_root / "result_summary.json").write_text(
                json.dumps({"rows": []})
            )
            return {"returncode": 1, "timed_out": False, "wall_sec": 0.01}
        return {"returncode": 0, "timed_out": False, "wall_sec": 0.01}

    run_target_workers(
        args=args,
        deadline=Deadline(10),
        attempt_id="attempt-test",
        attempt_root=tmp_path / "attempt",
        outcomes_root=tmp_path / "outcomes",
        store_path=tmp_path / "store.json",
        mobilegpt_memory=tmp_path / "mobilegpt",
        appagent_memory=tmp_path / "appagent",
        blocked_methods={},
        command_runner=runner,
    )

    assert len(calls) == len(METHODS) * len(DEVICES)
    assert len([row for row in recorded if row["status"] == "method_failed"]) == len(
        DEVICES
    )


def test_published_result_row_reads_native_summary_and_raw_validator_result(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    args.task = "CameraTakePhoto"
    archive = args.results_root / "androidworld" / args.task / "fixed_replay" / "target"
    archive.mkdir(parents=True)
    marker = "attempt-test.fixed_replay.small5554"
    (archive / marker / "result_summary.json").parent.mkdir(parents=True)
    (archive / marker / "result_summary.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "task": args.task,
                        "method": "fixed_replay",
                        "device": "small5554",
                        "validator_success": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    row = _published_official_result_row(
        args=args,
        attempt_id="attempt-test",
        method="fixed_replay",
        device="small5554",
    )
    assert row["validator_success"] is True
    assert row["official_validator_success"] is True

    raw_archive = args.results_root / "androidworld" / args.task / "omniflow" / "target"
    raw_archive.mkdir(parents=True)
    raw_marker = "attempt-raw.omniflow.small5554"
    raw_path = raw_archive / raw_marker / "task_results.jsonl"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text(
        json.dumps(
            {
                "official_validator_used": True,
                "androidworld_validator_result": {"success": False},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    raw_row = _published_official_result_row(
        args=args,
        attempt_id="attempt-raw",
        method="omniflow",
        device="small5554",
    )
    assert raw_row["validator_success"] is False

    external_archive = (
        args.results_root
        / "androidworld"
        / args.task
        / "mobilegpt"
        / "target"
        / "runlog"
        / "attempt-external"
        / "official_client"
    )
    external_archive.mkdir(parents=True)
    (external_archive / "task_results.jsonl").write_text(
        json.dumps(
            {
                "official_validator_used": True,
                "official_validator_success": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    external_row = _published_official_result_row(
        args=args,
        attempt_id="attempt-external",
        method="mobilegpt",
        device="small5554",
    )
    assert external_row["validator_success"] is True


def test_published_result_row_does_not_cross_match_device_attempts(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    args.task = "CameraTakePhoto"
    archive = args.results_root / "androidworld" / args.task / "mobilegpt"
    old_device = archive / "old" / "runlog" / "attempt-shared" / "official_client"
    new_device = (
        archive
        / "OmniFlowTargetSmall_seed111_eval113"
        / "runlog"
        / "attempt-shared"
        / "official_client"
    )
    old_device.mkdir(parents=True)
    new_device.mkdir(parents=True)
    (old_device / "task_results.jsonl").write_text(
        json.dumps(
            {
                "device": "emulator-5554",
                "official_validator_used": True,
                "official_validator_success": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (new_device / "task_results.jsonl").write_text(
        json.dumps(
            {
                "device": "emulator-5562",
                "official_validator_used": True,
                "official_validator_success": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    row = _published_official_result_row(
        args=args,
        attempt_id="attempt-shared",
        method="mobilegpt",
        device="small5562",
    )

    assert row["device"] == "emulator-5562"
    assert row["validator_success"] is True


def test_published_result_row_ignores_replay_memory_inputs(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    args.task = "CameraTakePhoto"
    archive = (
        args.results_root
        / "androidworld"
        / args.task
        / "fixed_replay"
        / "OmniFlowTargetFold_seed111_eval113"
    )
    target = archive / "runlog" / "attempt_002"
    replay_memory = (
        archive
        / "memory"
        / "attempt_005.fixed_replay.fold5564"
        / "_replay_runlogs"
    )
    target.mkdir(parents=True)
    replay_memory.mkdir(parents=True)
    (target / "task_results.jsonl").write_text(
        json.dumps(
            {
                "official_validator_used": True,
                "androidworld_validator_result": {"success": False},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (replay_memory / "task_results.jsonl").write_text(
        json.dumps(
            {
                "official_validator_used": True,
                "androidworld_validator_result": {"success": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    row = _published_official_result_row(
        args=args,
        attempt_id="attempt_005",
        method="fixed_replay",
        device="fold5564",
    )

    assert row == {}


def test_published_result_row_uses_canonical_device_scope(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    args.task = "CameraTakePhoto"
    archive = args.results_root / "androidworld" / args.task / "fixed_replay"
    small = archive / "OmniFlowTargetSmall_seed111_eval113" / "runlog" / "attempt_003"
    fold = archive / "OmniFlowTargetFold_seed111_eval113" / "runlog" / "attempt_003"
    small.mkdir(parents=True)
    fold.mkdir(parents=True)
    (small / "result_summary.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "task": args.task,
                        "validator_success": True,
                        "actions_executed": 4,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (fold / "task_results.jsonl").write_text(
        json.dumps(
            {
                "official_validator_used": True,
                "androidworld_validator_result": {"success": False},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    row = _published_official_result_row(
        args=args,
        attempt_id="attempt_003",
        method="fixed_replay",
        device="fold5564",
    )

    assert row["validator_success"] is False


def test_rerun_concluded_override_keeps_old_results_runnable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    args.task = "CameraTakePhoto"
    args.e2e_method = "fixed_replay"
    args.e2e_device = "small5554:emulator-5554:5554"
    outcomes_root = tmp_path / "outcomes"
    monkeypatch.setenv("OMNIFLOW_ANDROIDWORLD_RERUN_CONCLUDED", "1")
    monkeypatch.setattr(
        "src.experiment.run_tasks.concluded_result_keys",
        lambda **_: {("fixed_replay", "small5554")},
    )

    assert _concluded_results(args, outcomes_root, "attempt_002") == set()


def test_formal_result_paths_include_published_native_runner_file(
    tmp_path: Path,
) -> None:
    result_file = tmp_path / "runlog" / "attempt_002" / "task_results.jsonl"
    result_file.parent.mkdir(parents=True)
    result_file.write_text("{}\n", encoding="utf-8")
    paths = _formal_result_paths(
        {
            "status": "completed",
            "output_path": str(tmp_path / "target_attempt"),
            "metadata": {"official_result_files": [str(result_file)]},
        }
    )
    assert paths == [result_file.resolve()]


def test_blocked_cells_do_not_duplicate_shared_prep_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    recorded: list[dict[str, object]] = []
    completed: set[tuple[str, str]] = set()
    monkeypatch.setattr(
        "src.experiment.run_tasks._concluded_results",
        lambda *_: set(completed),
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks.concluded_result_keys",
        lambda **_: set(),
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks.record_result_outcome",
        lambda **kwargs: recorded.append(kwargs) or tmp_path / "outcome.json",
    )

    run_target_workers(
        args=args,
        deadline=Deadline(10),
        attempt_id="attempt-test",
        attempt_root=tmp_path / "attempt",
        outcomes_root=tmp_path / "outcomes",
        store_path=tmp_path / "store.json",
        mobilegpt_memory=None,
        appagent_memory=None,
        blocked_methods={
            "omniflow": ("prep_failed", "function_asset", str(tmp_path / "failure.json"))
        },
        command_runner=lambda *args, **kwargs: (
            completed.add(
                (
                    str(kwargs["environment"]["OMNIFLOW_ANDROIDWORLD_METHOD"]),
                    str(
                        kwargs["environment"]["OMNIFLOW_ANDROIDWORLD_DEVICE"]
                    ).split(":")[0],
                )
            )
            or {"returncode": 0, "timed_out": False, "wall_sec": 0}
        ),
    )

    omniflow = [row for row in recorded if row["method"] == "omniflow"]
    assert len(omniflow) == len(DEVICES)
    assert all(row["artifact_root"] is None for row in omniflow)


def test_zero_remaining_deadline_does_not_launch_child(tmp_path: Path) -> None:
    log_path = tmp_path / "deadline.log"

    result = run_logged_command(
        ["this-command-must-not-run"],
        cwd=tmp_path,
        environment={},
        log_path=log_path,
        timeout_sec=0,
    )

    assert result["returncode"] == 124
    assert result["timed_out"] is True
    assert "deadline exceeded" in log_path.read_text(encoding="utf-8")


def test_collect_replayed_source_uses_fixed_replay_and_captures_screenshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    source_path = tmp_path / "source.run_log.json"
    source = androidworld_run_log(
        [{"action_type": "click", "x": 50, "y": 50}],
        task_name=args.task,
    )
    source["seed"] = 999
    source_path.write_text(json.dumps(source), encoding="utf-8")
    screenshot = tmp_path / "screen.png"
    screenshot.write_bytes(b"captured-screen")
    screenshot_hash = hashlib.sha256(screenshot.read_bytes()).hexdigest()
    state = {
        "pixels": {
            "path": str(screenshot.resolve()),
            "sha256": screenshot_hash,
            "width": 100,
            "height": 200,
            "mime_type": "image/png",
        },
        "forest": "<hierarchy />",
        "ui_elements": [],
        "auxiliaries": {
            "state_id": "captured-state",
            "display": {"width": 100, "height": 200},
        },
    }

    def runner(command: list[str], **kwargs: object) -> dict[str, object]:
        assert command[command.index("--agent") + 1] == "fixed_replay"
        assert command[command.index("--raw-replay-run-log") + 1] == str(source_path)
        assert "--model" not in command
        assert "--planner-provider" not in command
        assert "--store-path" not in command
        environment = kwargs["environment"]
        assert isinstance(environment, dict)
        assert environment["OMNIFLOW_RAW_REPLAY_CAPTURE_OBSERVATIONS"] == "1"
        output = Path(command[command.index("--output-path") + 1])
        output.mkdir(parents=True, exist_ok=True)
        (output / "task_results.jsonl").write_text(
            json.dumps(
                {
                    "official_validator_used": True,
                    "androidworld_validator_result": {
                        "success": True,
                        "reward": 1.0,
                        "uses_androidworld_official_validator": True,
                    },
                    "model_calls": 0,
                    "total_tokens": 0,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        raw_replay_result = Path(environment["OMNIFLOW_RAW_REPLAY_RESULT_JSON"])
        raw_replay_result.parent.mkdir(parents=True, exist_ok=True)
        raw_replay_result.write_text(
            json.dumps(
                {
                    "completed": True,
                    "replay_completed": True,
                    "run_id": "fixed-replay-capture",
                    "execution_trace": {
                        "steps": [
                            {
                                "provider_detail": {
                                    "raw_replay": {
                                        "step_results": [
                                            {
                                                "completed": True,
                                                "observation_before_act": {
                                                    "androidworld_state": state
                                                },
                                            }
                                        ],
                                        "final_observation": {
                                            "androidworld_state": state
                                        },
                                    }
                                }
                            }
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )
        return {
            "returncode": 0,
            "timed_out": False,
            "wall_sec": 0.1,
            "log_path": str(kwargs["log_path"]),
        }

    monkeypatch.setattr(
        "src.experiment.run_tasks.run_logged_command",
        runner,
    )
    captured_path, captured, phase = collect_replayed_source(
        args=args,
        deadline=Deadline(10),
        attempt_root=tmp_path / "attempt",
        source_path=source_path,
        source_run_log=source,
    )

    assert captured_path.is_file()
    assert captured["steps"][0]["action"] == source["steps"][0]["action"]
    assert captured["seed"] == SOURCE_SEED
    assert "sha256" not in captured["steps"][0]["observation"]["pixels"]
    assert captured["steps"][0]["observation"]["pixels"]["path"] == str(
        screenshot.resolve()
    )
    assert phase["model_calls"] == 0
    assert phase["total_tokens"] == 0
    assert phase["status"] == "collected"


def test_fixed_replay_groups_editable_input_text_raw_actions() -> None:
    source = androidworld_run_log(
        [{"action_type": "input_text", "x": 50, "y": 50, "text": "hello"}],
        observations=[
            {
                "pixels": None,
                "forest": (
                    '<hierarchy><node class="android.widget.EditText" '
                    'editable="true" bounds="[0,0][100,100]" /></hierarchy>'
                ),
                "ui_elements": [],
                "auxiliaries": {"display": {"width": 1000, "height": 1000}},
            }
        ],
    )

    assert _fixed_replay_source_step_width(source["steps"][0]) == 2


def test_collect_replayed_source_rejects_model_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    source_path = tmp_path / "source.run_log.json"
    source = androidworld_run_log(
        [{"action_type": "click", "x": 50, "y": 50}],
        task_name=args.task,
    )
    source_path.write_text(json.dumps(source), encoding="utf-8")

    def runner(command: list[str], **kwargs: object) -> dict[str, object]:
        output = Path(command[command.index("--output-path") + 1])
        output.mkdir(parents=True, exist_ok=True)
        (output / "task_results.jsonl").write_text(
            json.dumps(
                {
                    "official_validator_used": True,
                    "androidworld_validator_result": {
                        "success": True,
                        "reward": 1.0,
                        "uses_androidworld_official_validator": True,
                    },
                    "model_calls": 1,
                    "total_tokens": 10,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "returncode": 0,
            "timed_out": False,
            "wall_sec": 0.1,
            "log_path": str(kwargs["log_path"]),
        }

    monkeypatch.setattr(
        "src.experiment.run_tasks.run_logged_command",
        runner,
    )

    with pytest.raises(PipelinePhaseError) as raised:
        collect_replayed_source(
            args=args,
            deadline=Deadline(10),
            attempt_root=tmp_path / "attempt",
            source_path=source_path,
            source_run_log=source,
        )

    assert raised.value.phase["model_calls"] == 1
    assert raised.value.phase["total_tokens"] == 10


def test_pipeline_does_not_collect_missing_canonical_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    monkeypatch.setattr(
        "src.experiment.run_tasks.ensure_source_device",
        lambda **_: {"status": "ready", "model_calls": 0, "total_tokens": 0},
    )
    collected = False

    def collect(**_kwargs: object) -> object:
        nonlocal collected
        collected = True
        raise AssertionError("formal orchestration must not collect source data")

    monkeypatch.setattr(
        "src.experiment.run_tasks.collect_replayed_source",
        collect,
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks._blocked_all",
        lambda **_: None,
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks._report",
        lambda **kwargs: kwargs["phases"],
    )

    phases = run_pipeline(args)

    assert phases["source"]["status"] == "failed"
    assert phases["source"]["model_calls"] == 0
    assert phases["source"]["total_tokens"] == 0
    assert collected is False


def test_autodroid_pipeline_uses_task_reference_without_source_runlog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    args.e2e_method = "autodroid"
    args.e2e_device = "autodroidsmall5554:emulator-5554:5554"
    args.source_only = True
    args.memory_index.parent.mkdir(parents=True)
    args.memory_index.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.data-index.v2",
                "canonical": {
                    "source_run_logs": {
                        args.task: {
                            "goal": "Draw a shape",
                            "params": {"shape": "circle"},
                        }
                    }
                },
                "source_index": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks._report",
        lambda **kwargs: kwargs["phases"],
    )

    phases = run_pipeline(args)

    assert phases["source_device"]["status"] == "skipped"
    assert phases["source"]["status"] == "skipped"
    assert phases["source"]["task_params"] == {"shape": "circle"}
    assert phases["source"]["task_reference_index"] == str(args.memory_index)


def test_autodroid_task_params_fall_back_to_retained_source_runlog(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.run_log.json"
    source_path.write_text(
        json.dumps(
            {
                "task_parameters": {
                    "row_objects": [{"name": "Bike Repairs"}],
                    "seed": 111,
                }
            }
        ),
        encoding="utf-8",
    )
    index_path = tmp_path / "current.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.data-index.v2",
                "canonical": {},
                "source_index": {
                    "ExpenseDeleteSingle": {
                        "retained_source_run_log": str(source_path),
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    assert _autodroid_task_params_from_index(
        index_path,
        "ExpenseDeleteSingle",
    ) == {"row_objects": [{"name": "Bike Repairs"}]}


def test_autodroid_task_params_skip_missing_retained_source_runlog(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "current.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.data-index.v2",
                "canonical": {},
                "source_index": {
                    "MarkorDeleteNote": {
                        "params": {},
                        "retained_source_run_log": str(tmp_path / "missing.json"),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert _autodroid_task_params_from_index(index_path, "MarkorDeleteNote") == {}


def test_autodroid_task_params_do_not_fallback_for_explicit_empty_params(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.run_log.json"
    source_path.write_text(
        json.dumps({"task_parameters": {"should_not": "be_read"}}),
        encoding="utf-8",
    )
    index_path = tmp_path / "current.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.data-index.v2",
                "canonical": {},
                "source_index": {
                    "CameraTakePhoto": {
                        "params": {},
                        "retained_source_run_log": str(source_path),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert _autodroid_task_params_from_index(index_path, "CameraTakePhoto") == {}


def test_source_only_pipeline_collects_replayed_source_and_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    args.source_only = True
    source_path = tmp_path / "source.run_log.json"
    monkeypatch.setattr(
        "src.experiment.run_tasks.ensure_source_device",
        lambda **_: {"status": "ready", "model_calls": 0, "total_tokens": 0},
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks._canonical_source",
        lambda *_, **__: ({}, source_path, {"task_name": args.task}),
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks.collect_replayed_source",
        lambda **_: (
            source_path,
            {"task_name": args.task},
            {
                "status": "collected",
                "source_run_log": str(source_path),
                "model_calls": 0,
                "total_tokens": 0,
            },
        ),
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks.prepare_function_asset",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("source-only collection must not prepare Functions")
        ),
    )

    report = run_pipeline(args)

    assert report["schema_version"] == (
        "omniflow.androidworld.source-collection-report.v1"
    )
    assert report["status"] == "collected"
    assert report["phases"]["source"]["source_run_log"] == str(source_path)


@pytest.mark.parametrize(
    ("source_only", "source_qualification_only", "report_status"),
    ((True, False, "collected"), (False, True, "qualified")),
)
def test_source_mode_success_returns_zero_exit_status(
    monkeypatch: pytest.MonkeyPatch,
    source_only: bool,
    source_qualification_only: bool,
    report_status: str,
) -> None:
    from src.experiment import run_tasks

    args = SimpleNamespace(
        environment="androidworld",
        dry_run=False,
        source_only=source_only,
        source_qualification_only=source_qualification_only,
    )
    monkeypatch.setattr(
        run_tasks,
        "build_parser",
        lambda: SimpleNamespace(parse_args=lambda _argv: args),
    )
    monkeypatch.setattr(run_tasks, "_resolve_args", lambda value: value)
    monkeypatch.setattr(
        run_tasks,
        "run_pipeline",
        lambda _args: {"status": report_status},
    )

    assert run_tasks.main([]) == 0


def test_pipeline_blocks_only_function_when_canonical_store_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    source_path = tmp_path / "source.json"
    monkeypatch.setattr(
        "src.experiment.run_tasks.ensure_source_device",
        lambda **_: {"status": "ready", "model_calls": 0, "total_tokens": 0},
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks._canonical_source",
        lambda *_, **__: ({}, source_path, {}),
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks.prepare_function_asset",
        lambda **_: (_ for _ in ()).throw(RuntimeError("function failed")),
    )
    mobilegpt_called = False

    def prepare_mobilegpt(**_kwargs: object) -> object:
        nonlocal mobilegpt_called
        mobilegpt_called = True
        return tmp_path / "mobilegpt", {"status": "reused"}

    monkeypatch.setattr(
        "src.experiment.run_tasks.prepare_mobilegpt_memory",
        prepare_mobilegpt,
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks.prepare_appagent_memory",
        lambda **_: (tmp_path / "appagent", {"status": "reused"}),
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks.run_target_workers",
        lambda **_: [],
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks._report",
        lambda **kwargs: kwargs["phases"],
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks._blocked_all",
        lambda **_: None,
    )

    phases = run_pipeline(args)

    assert phases["function"]["status"] == "failed"
    assert mobilegpt_called is True
    assert phases["mobilegpt_memory"]["status"] == "reused"
    assert phases["appagent_memory"]["status"] == "reused"


def test_appagent_pipeline_does_not_use_or_refresh_function_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    args.e2e_method = "appagent"
    args.e2e_device = DEVICES[0]
    source_path = tmp_path / "source.json"
    source_path.write_text("{}", encoding="utf-8")
    refresh_called = False

    monkeypatch.setattr(
        "src.experiment.run_tasks.ensure_source_device",
        lambda **_: {"status": "ready", "model_calls": 0, "total_tokens": 0},
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks._canonical_source",
        lambda *_: ({}, source_path, {"task_parameters": {}}),
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks.prepare_appagent_memory",
        lambda **kwargs: (
            tmp_path / "appagent",
            {
                "status": "created",
                "source_run_log": str(kwargs["source_run_log"]),
            },
        ),
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks.run_target_workers",
        lambda **_: [],
    )

    def refresh(**_: object) -> None:
        nonlocal refresh_called
        refresh_called = True

    monkeypatch.setattr(
        "src.experiment.run_tasks.refresh_data_index_from_pointer",
        refresh,
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks._report",
        lambda **kwargs: kwargs["phases"],
    )

    phases = run_pipeline(args)

    assert phases["function"]["status"] == "skipped"
    assert phases["appagent_memory"]["source_run_log"] == str(source_path)
    assert refresh_called is False


def test_pipeline_qualifies_one_source_function_before_target_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    source_path = tmp_path / "source.json"
    source_path.write_text("{}", encoding="utf-8")
    store_path = tmp_path / "store.json"
    store_path.write_text("{}", encoding="utf-8")
    source_call = {"function_id": "create_note", "arguments": {"name": "note"}}
    events: list[str] = []
    monkeypatch.setattr(
        "src.experiment.run_tasks.ensure_source_device",
        lambda **_: {"status": "ready", "model_calls": 0, "total_tokens": 0},
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks._canonical_source",
        lambda *_: ({}, source_path, {"task_parameters": {}}),
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks.prepare_function_asset",
        lambda **_: (
            {"store_path": str(store_path)},
            {
                "status": "reused",
                "model_calls": 0,
                "total_tokens": 0,
                "source_calls": [source_call],
            },
        ),
    )

    def qualify(**kwargs: object) -> dict[str, object]:
        events.append("qualify")
        assert kwargs["source_call"] == source_call
        return {
            "status": "qualified",
            "qualified": True,
            "model_calls": 0,
            "total_tokens": 0,
        }

    monkeypatch.setattr(
        "src.experiment.run_tasks.qualify_source_function",
        qualify,
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks.prepare_mobilegpt_memory",
        lambda **_: (tmp_path / "mobilegpt", {"status": "reused"}),
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks.prepare_appagent_memory",
        lambda **_: (tmp_path / "appagent", {"status": "reused"}),
    )

    def workers(**kwargs: object) -> list[dict[str, object]]:
        events.append("targets")
        assert kwargs["blocked_methods"] == {}
        return []

    monkeypatch.setattr(
        "src.experiment.run_tasks.run_target_workers",
        workers,
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks._report",
        lambda **kwargs: kwargs["phases"],
    )

    phases = run_pipeline(args)

    assert events == ["qualify", "targets"]
    assert phases["source_qualification"]["qualified"] is True


def test_cached_source_function_qualification_requires_matching_function_identity(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    source_path = tmp_path / "source.json"
    source_path.write_text("source", encoding="utf-8")
    store_path = tmp_path / "store.json"
    store_path.write_text("store", encoding="utf-8")
    qualification_path = (
        args.output_root
        / args.task
        / "previous"
        / "source_qualification"
        / "CameraTakePhoto"
        / "function_replay"
        / "source5560"
        / "qualification.json"
    )
    qualification_path.parent.mkdir(parents=True)
    qualification_path.write_text(
        json.dumps(
            {
                "qualified": True,
                "source_run_log": str(source_path),
                "store_path": str(store_path),
                "function_id": "create_note",
                "source_call": {
                    "function_id": "create_note",
                    "arguments": {},
                },
                "model_calls": 0,
                "fallback_steps": 0,
            }
        ),
        encoding="utf-8",
    )

    cached = _cached_source_function_qualification(
        args=args,
        source_path=source_path,
        function_store={"store_path": str(store_path)},
        source_call={"function_id": "create_note", "arguments": {}},
    )

    assert cached is not None
    assert cached["status"] == "reused"
    assert cached["cached_from"] == str(qualification_path.resolve())


def test_cached_source_function_qualification_ignores_other_function_store(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    source_path = tmp_path / "source.json"
    source_path.write_text("new source", encoding="utf-8")
    store_path = tmp_path / "store.json"
    store_path.write_text("new store", encoding="utf-8")
    qualification_path = (
        args.output_root
        / args.task
        / "previous"
        / "source_qualification"
        / "CameraTakePhoto"
        / "function_replay"
        / "source5560"
        / "qualification.json"
    )
    qualification_path.parent.mkdir(parents=True)
    qualification_path.write_text(
        json.dumps(
            {
                "qualified": True,
                "source_run_log": str(source_path),
                "store_path": str(tmp_path / "old-store.json"),
                "function_id": "open_brightness_settings",
                "source_call": {
                    "function_id": "open_brightness_settings",
                    "arguments": {},
                },
                "model_calls": 0,
                "fallback_steps": 0,
            }
        ),
        encoding="utf-8",
    )

    assert _cached_source_function_qualification(
        args=args,
        source_path=source_path,
        function_store={"store_path": str(store_path)},
        source_call={
            "function_id": "set_brightness_to_minimum",
            "arguments": {},
        },
    ) is None


def test_run_task_reads_json_objects_strictly(tmp_path: Path) -> None:
    path = tmp_path / "object.json"
    path.write_text('{"ready": true}', encoding="utf-8")

    assert _read_object(path) == {"ready": True}


def test_source_qualification_only_stops_before_baselines_and_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    args.source_qualification_only = True
    source_path = tmp_path / "source.json"
    source_path.write_text("{}", encoding="utf-8")
    store_path = tmp_path / "store.json"
    store_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "src.experiment.run_tasks.ensure_source_device",
        lambda **_: {"status": "ready", "model_calls": 0, "total_tokens": 0},
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks._canonical_source",
        lambda *_: ({}, source_path, {"task_parameters": {}}),
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks.prepare_function_asset",
        lambda **_: (
            {"store_path": str(store_path)},
            {
                "status": "reused",
                "model_calls": 0,
                "total_tokens": 0,
                "source_calls": [
                    {"function_id": "create_note", "arguments": {}}
                ],
            },
        ),
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks.qualify_source_function",
        lambda **_: {
            "status": "qualified",
            "qualified": True,
            "model_calls": 0,
            "total_tokens": 0,
        },
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks.prepare_mobilegpt_memory",
        lambda **_: (_ for _ in ()).throw(AssertionError("baseline prepared")),
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks.prepare_appagent_memory",
        lambda **_: (_ for _ in ()).throw(AssertionError("baseline prepared")),
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks.run_target_workers",
        lambda **_: (_ for _ in ()).throw(AssertionError("targets started")),
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks._report",
        lambda **kwargs: kwargs["phases"],
    )

    phases = run_pipeline(args)

    assert phases["source_qualification"]["qualified"] is True


def test_mobilegpt_memory_only_starts_no_emulators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    args.task = "SystemBluetoothTurnOff"
    args.e2e_method = "mobilegpt"
    args.e2e_device = "all"
    args.mobilegpt_memory_only = True
    memory = tmp_path / "memory"
    memory.mkdir()
    source_calls: list[str] = []
    monkeypatch.setattr(
        "src.experiment.run_tasks.ensure_source_device",
        lambda **_: source_calls.append("source")
        or {
            "status": "ready",
            "model_calls": 0,
            "total_tokens": 0,
        },
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks.ensure_target_devices",
        lambda **_: (_ for _ in ()).throw(AssertionError("target emulator started")),
    )
    monkeypatch.setattr(
        "src.experiment.run_tasks.prepare_mobilegpt_memory",
        lambda **_: (
            memory,
            {
                "status": "prepared",
                "memory_root": str(memory),
                "memory_validation": {"runlog_teacher_alignment": True},
                "model_calls": 1,
                "total_tokens": 10,
            },
        ),
    )

    report = run_pipeline(args)

    assert report["status"] == "validated"
    assert source_calls == []
    assert report["phases"]["source_device"]["status"] == "skipped"
    assert report["phases"]["target_devices"]["reason"] == (
        "runlog_memory_conversion_only"
    )


@pytest.mark.parametrize(
    ("model_calls", "fallback_steps", "expected"),
    [(0, 0, True), (1, 0, False), (0, 1, False)],
)
def test_source_function_qualification_requires_zero_model_and_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_calls: int,
    fallback_steps: int,
    expected: bool,
) -> None:
    args = _args(tmp_path)
    store = tmp_path / "store.json"
    store.write_text("{}", encoding="utf-8")
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")

    def runner(command: list[str], **kwargs: object) -> dict[str, object]:
        output = Path(command[command.index("--output-path") + 1])
        output.mkdir(parents=True)
        (output / "task_results.jsonl").write_text(
            json.dumps(
                {
                    "official_validator_success": True,
                    "model_calls": model_calls,
                    "fallback_steps": fallback_steps,
                    "canonical_run": {
                        "status": "succeeded",
                        "diagnostics": {
                            "execution_summary": {"success": True, "steps": 1},
                            "execution_trace": [{"result": {"success": True}}],
                        },
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "returncode": 0,
            "timed_out": False,
            "wall_sec": 0.1,
            "log_path": str(kwargs["log_path"]),
        }

    monkeypatch.setattr(
        "src.experiment.run_tasks.run_logged_command",
        runner,
    )
    result = qualify_source_function(
        args=args,
        source_path=source,
        run_log={"task_parameters": {}},
        function_store={
            "store_path": str(store),
            "transfer_states_sha256": "a" * 64,
        },
        source_call={"function_id": "draw", "arguments": {}},
        attempt_root=tmp_path / "attempt",
        deadline=Deadline(10),
        round_index=1,
    )

    assert result["qualified"] is expected


def test_source_function_qualification_does_not_require_whole_task_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    store = tmp_path / "store.json"
    store.write_text("{}", encoding="utf-8")
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")

    def runner(command: list[str], **kwargs: object) -> dict[str, object]:
        output = Path(command[command.index("--output-path") + 1])
        output.mkdir(parents=True)
        (output / "task_results.jsonl").write_text(
            json.dumps(
                {
                    "official_validator_success": False,
                    "model_calls": 0,
                    "fallback_steps": 0,
                    "canonical_run": {
                        "status": "failed",
                        "diagnostics": {
                            "execution_summary": {"success": True, "steps": 2},
                            "execution_trace": [
                                {"result": {"success": True}},
                                {"result": {"success": True}},
                            ],
                        },
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "returncode": 0,
            "timed_out": False,
            "wall_sec": 0.1,
            "log_path": str(kwargs["log_path"]),
        }

    monkeypatch.setattr(
        "src.experiment.run_tasks.run_logged_command",
        runner,
    )
    result = qualify_source_function(
        args=args,
        source_path=source,
        run_log={"task_parameters": {}},
        function_store={"store_path": str(store), "transfer_states_sha256": "a" * 64},
        source_call={"function_id": "draw", "arguments": {}},
        attempt_root=tmp_path / "attempt",
        deadline=Deadline(10),
        round_index=1,
    )

    assert result["official_validator_success"] is False
    assert result["function_replay_success"] is True
    assert result["qualified"] is False


def test_source_function_qualification_uses_one_complete_function(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    store = tmp_path / "store.json"
    store.write_text("{}", encoding="utf-8")
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    captured: list[list[str]] = []

    def runner(command: list[str], **kwargs: object) -> dict[str, object]:
        captured.append(command)
        output = Path(command[command.index("--output-path") + 1])
        output.mkdir(parents=True)
        (output / "task_results.jsonl").write_text(
            json.dumps(
                {
                    "official_validator_success": True,
                    "model_calls": 0,
                    "fallback_steps": 0,
                    "canonical_run": {
                        "status": "succeeded",
                        "diagnostics": {
                            "execution_summary": {"success": True, "steps": 6},
                            "execution_trace": [
                                {"result": {"success": True}}
                                for _ in range(6)
                            ],
                        },
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "returncode": 0,
            "timed_out": False,
            "wall_sec": 0.1,
            "log_path": str(kwargs["log_path"]),
        }

    monkeypatch.setattr(
        "src.experiment.run_tasks.run_logged_command",
        runner,
    )
    source_call = {"function_id": "create_note", "arguments": {"name": "note"}}

    result = qualify_source_function(
        args=args,
        source_path=source,
        run_log={"task_parameters": {}},
        function_store={
            "store_path": str(store),
            "transfer_states_sha256": "a" * 64,
        },
        source_call=source_call,
        attempt_root=tmp_path / "attempt",
        deadline=Deadline(10),
        round_index=1,
    )

    assert result["qualified"] is True
    assert result["qualification_scope"] == "atomic_function_replay"
    assert result["source_call"] == source_call
    assert len(captured) == 1
    command = captured[0]
    call_index = command.index("--function-id") + 1
    assert command[call_index] == source_call["function_id"]


def test_source_qualification_requires_official_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    store = tmp_path / "store.json"
    store.write_text("{}\n", encoding="utf-8")
    args = _args(tmp_path)

    def runner(command, **kwargs):
        output_root = Path(command[command.index("--output-path") + 1])
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "task_results.jsonl").write_text(
            json.dumps(
                {
                    "official_validator_success": False,
                    "model_calls": 0,
                    "fallback_steps": 0,
                    "canonical_run": {
                        "status": "failed",
                        "diagnostics": {
                            "execution_summary": {"success": True, "steps": 1},
                            "execution_trace": [{"result": {"success": True}}],
                        },
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "returncode": 0,
            "timed_out": False,
            "wall_sec": 0.1,
            "log_path": str(kwargs["log_path"]),
        }

    monkeypatch.setattr(
        "src.experiment.run_tasks.run_logged_command",
        runner,
    )
    result = qualify_source_function(
        args=args,
        source_path=source,
        run_log={"task_parameters": {}},
        function_store={
            "store_path": str(store),
            "transfer_states_sha256": "a" * 64,
        },
        source_call={"function_id": "create_note", "arguments": {}},
        attempt_root=tmp_path / "attempt",
        deadline=Deadline(10),
        round_index=1,
    )

    assert result["function_replay_success"] is True
    assert result["official_validator_success"] is False
    assert result["qualified"] is False


def test_function_replay_success_is_independent_of_validator() -> None:
    row = {
        "official_validator_success": False,
        "canonical_run": {
            "diagnostics": {
                "execution_summary": {"success": True, "steps": 1},
                "execution_trace": [{"result": {"success": True}}],
            }
        },
    }
    assert _function_replay_success(row) is True


def test_function_replay_success_rejects_whole_task_status_without_runtime_evidence() -> None:
    row = {
        "official_validator_success": True,
        "canonical_run": {"status": "succeeded"},
    }
    assert _function_replay_success(row) is False


def test_pipeline_report_always_materializes_four_report_formats(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.memory_index.parent.mkdir(parents=True)
    source_index = tmp_path / "source_index.json"
    source_index.write_text(json.dumps({args.task: {}}), encoding="utf-8")
    result_cells = tmp_path / "result_cells.json"
    result_cells.write_text("{}", encoding="utf-8")
    args.memory_index.write_text(
        json.dumps(
            {
                "source_index": str(source_index),
                "result_cells": str(result_cells),
            }
        ),
        encoding="utf-8",
    )
    attempt_root = tmp_path / "attempt"
    attempt_root.mkdir()
    outcomes_root = tmp_path / "outcomes"
    for method in METHODS:
        for label, serial, _ in DEVICES:
            record_result_outcome(
                outcomes_root=outcomes_root,
                task_name=args.task,
                method=method,
                device=label,
                device_serial=serial,
                attempt_id="attempt-test",
                source_seed=SOURCE_SEED,
                evaluation_seed=TASK_SEED,
                status="prep_failed",
                stage="test",
            )

    summary = _report(
        args=args,
        attempt_id="attempt-test",
        attempt_root=attempt_root,
        outcomes_root=outcomes_root,
        deadline=Deadline(10),
        phases={"source": {"status": "failed", "model_calls": 1, "total_tokens": 7}},
    )

    assert summary["counts"]["planned"] == len(METHODS) * len(DEVICES)
    assert summary["counts"]["pending"] == 0
    assert summary["model_calls"] == 1
    assert summary["total_tokens"] == 7
    assert (attempt_root / "pipeline_summary.json").is_file()
    assert summary["result_summary"]["counts"]["non_validator_failure"] == len(METHODS) * len(DEVICES)
    assert "tool_calls" not in summary
    assert "tokens" not in summary
