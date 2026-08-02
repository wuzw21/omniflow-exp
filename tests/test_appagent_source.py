from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from src.experiment import androidworld as pipeline
from src.experiment import appagent_source
from src.integrations import appagent_adapter
from runlog_fixtures import androidworld_run_log, androidworld_state


def _write_appagent_teacher_source(
    root: Path,
    *,
    task_name: str,
    action: dict,
) -> Path:
    source_run_log = root / "source.run_log.json"
    source_run_log.write_text("{}", encoding="utf-8")
    teacher_source = root / "teacher_source.json"
    teacher_source.write_text(
        json.dumps(
            {
                "schema_version": appagent_adapter.APPAGENT_TEACHER_SOURCE_SCHEMA,
                "task_name": task_name,
                "source_seed": 111,
                "source_run_id": "source",
                "source_run_log": str(source_run_log),
                "source_run_log_sha256": hashlib.sha256(
                    source_run_log.read_bytes()
                ).hexdigest(),
                "official_appagent_revision": (
                    appagent_adapter.APPAGENT_OFFICIAL_REVISION
                ),
                "actions": [
                    {
                        "source_step_index": 1,
                        "source_action_index": 0,
                        "action": action,
                    }
                ],
                "action_count": 1,
                "demo_action_count": 1,
                "consumer": "appagent_official_human_demonstration",
                "adapter_scope": "native_androidworld_action_sequence",
                "uses_omniflow_function": False,
                "writes_appagent_docs": False,
                "requires_native_source_episode": True,
                "target_inputs_read": False,
                "coordinate_replay": False,
            }
        ),
        encoding="utf-8",
    )
    return teacher_source


def _browser_draw_teacher_agent(
    tmp_path: Path,
    env: SimpleNamespace,
) -> appagent_adapter.AppAgentTeacherAgent:
    teacher_source = _write_appagent_teacher_source(
        tmp_path,
        task_name="BrowserDraw",
        action={
            "type": "click",
            "params": {
                "target_description": "6.50 kB",
                "source_context": {"element": {"text": "6.50 kB"}},
            },
        },
    )
    agent = appagent_adapter.AppAgentTeacherAgent(
        env=env,
        official_runtime=SimpleNamespace(),
        teacher_source=teacher_source,
        workspace_root=tmp_path / "workspace",
        demo_name="browser_draw",
        action_factory=lambda **kwargs: kwargs,
    )
    agent.set_current_task(
        "BrowserDraw",
        "Open task.html and draw.",
        {"app_names": ["chrome"]},
    )
    return agent


def _forest_node(
    node_id: int,
    bounds: tuple[int, int, int, int],
    *,
    child_ids: tuple[int, ...] = (),
    text: str = "",
    clickable: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        unique_id=node_id,
        bounds_in_screen=SimpleNamespace(
            left=bounds[0],
            top=bounds[1],
            right=bounds[2],
            bottom=bounds[3],
        ),
        child_ids=child_ids,
        text=text,
        package_name="com.google.android.documentsui",
        is_clickable=clickable,
        is_visible_to_user=True,
    )


def _copy_labeled_screenshot(source, destination, *_args, **_kwargs) -> None:
    Path(destination).write_bytes(Path(source).read_bytes())


def _install_androidworld_app_registry(
    monkeypatch: pytest.MonkeyPatch,
    controller: object,
) -> None:
    adb_utils = SimpleNamespace(
        get_all_apps=lambda actual_controller: (
            ["chrome", "files"] if actual_controller is controller else []
        ),
        get_adb_activity=lambda app_name: {
            "chrome": "com.android.chrome/com.google.android.apps.chrome.Main",
            "files": (
                "com.google.android.documentsui/"
                "com.android.documentsui.files.FilesActivity"
            ),
        }.get(app_name),
    )
    android_world = ModuleType("android_world")
    android_world_env = ModuleType("android_world.env")
    android_world_env.adb_utils = adb_utils
    android_world.env = android_world_env
    monkeypatch.setitem(sys.modules, "android_world", android_world)
    monkeypatch.setitem(sys.modules, "android_world.env", android_world_env)


def _write_open_app_teacher_source(root: Path) -> Path:
    source_run_log = root / "source.run_log.json"
    run_log = androidworld_run_log(
        [
            {
                "action_type": "open_app",
                "app_name": "com.google.android.documentsui",
            },
            {"action_type": "click", "x": 50, "y": 50},
        ],
        observations=[
            androidworld_state("launcher", width=100, height=100),
            androidworld_state("files", width=100, height=100),
        ],
        task_name="BrowserDraw",
        run_id="browser-draw-source",
    )
    run_log["steps"][1]["metadata"] = {
        "source_context": {"element": {"text": "6.50 kB"}}
    }
    source_run_log.write_text(
        json.dumps(run_log),
        encoding="utf-8",
    )
    teacher_source = root / "teacher_source.json"
    teacher_source.write_text(
        json.dumps(
            appagent_adapter.build_appagent_teacher_source(
                source_run_log,
                task_name="BrowserDraw",
            )
        ),
        encoding="utf-8",
    )
    return teacher_source


def test_appagent_teacher_executes_open_app_through_androidworld(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = object()
    _install_androidworld_app_registry(monkeypatch, controller)
    teacher_source = _write_open_app_teacher_source(tmp_path)
    events: list[tuple[str, object]] = []
    xml = (
        '<hierarchy class="android.widget.FrameLayout" '
        'bounds="[0,0][100,100]"><node index="0" '
        'class="android.widget.TextView" text="6.50 kB" clickable="true" '
        'bounds="[0,0][100,100]" /></hierarchy>'
    )

    def get_state():
        events.append(("observe", None))
        return SimpleNamespace(
            xml=xml,
            pixels=np.zeros((100, 100, 3), dtype=np.uint8),
        )

    agent = appagent_adapter.AppAgentTeacherAgent(
        env=SimpleNamespace(
            controller=controller,
            execute_action=lambda action: events.append(("action", action)),
            get_state=get_state,
        ),
        official_runtime=SimpleNamespace(
            min_dist=0.0,
            request_interval=0.0,
            draw_elements=_copy_labeled_screenshot,
        ),
        teacher_source=teacher_source,
        workspace_root=tmp_path / "workspace",
        demo_name="browser_draw",
        action_factory=lambda **kwargs: kwargs,
    )
    agent.set_current_task(
        "BrowserDraw",
        "Open task.html and draw.",
        {"app_names": ["chrome"]},
    )

    launch_result = agent.step("Open task.html and draw.")

    assert launch_result.done is False
    assert launch_result.data["teacher_actions_consumed"] == 1
    assert events == [
        ("action", {"action_type": "open_app", "app_name": "files"})
    ]

    action_result = agent.step("Open task.html and draw.")
    final_result = agent.step("Open task.html and draw.")

    assert action_result.done is False
    assert final_result.done is True
    assert [event[0] for event in events] == [
        "action",
        "observe",
        "action",
        "observe",
    ]
    assert events[2] == (
        "action",
        {"action_type": "click", "x": 50, "y": 50},
    )
    appagent_adapter._validate_demo_artifacts(
        Path(final_result.data["demo_root"]),
        expected_teacher_action_count=2,
        expected_demo_action_count=1,
    )


def test_appagent_deployment_consumes_open_app_before_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = object()
    _install_androidworld_app_registry(monkeypatch, controller)
    action_source = _write_open_app_teacher_source(tmp_path)
    events: list[tuple[str, object]] = []
    xml = (
        '<hierarchy><node text="6.50 kB" clickable="true" '
        'bounds="[0,0][100,100]" /></hierarchy>'
    )

    def get_state():
        events.append(("observe", None))
        return SimpleNamespace(
            xml=xml,
            pixels=np.zeros((100, 100, 3), dtype=np.uint8),
        )

    def draw_elements(source, destination, *_args, **_kwargs):
        Path(destination).write_bytes(Path(source).read_bytes())

    agent = appagent_adapter.AppAgentAndroidWorldAgent(
        env=SimpleNamespace(
            controller=controller,
            execute_action=lambda action: events.append(("action", action)),
            get_state=get_state,
        ),
        official_runtime=SimpleNamespace(
            min_dist=0.0,
            request_interval=0.0,
            draw_elements=draw_elements,
            build_task_prompt=lambda **_kwargs: "prompt",
            parse_response=lambda *_args, **_kwargs: ["FINISH"],
        ),
        llm=SimpleNamespace(
            predict_mm=lambda *_args, **_kwargs: ("finish", None, {})
        ),
        output_root=tmp_path / "output",
        docs_root=None,
        action_source=action_source,
        action_factory=lambda **kwargs: kwargs,
    )

    result = agent.step("Open task.html and draw.")

    assert result.done is True
    assert events[:2] == [
        ("action", {"action_type": "open_app", "app_name": "files"}),
        ("observe", None),
    ]


def _write_source_index(root: Path) -> Path:
    root.mkdir(parents=True)
    source_run_log = root / "source.run_log.json"
    source_run_log.write_text(
        json.dumps(
            androidworld_run_log(
                [{"action_type": "click", "x": 50, "y": 50}],
                observations=[
                    androidworld_state(
                        "state-0",
                        forest=(
                            '<hierarchy><node class="android.widget.FrameLayout" '
                            'bounds="[0,0][100,100]"><node text="Bluetooth" '
                            'resource-id="android:id/switch_widget" '
                            'clickable="true" bounds="[0,0][100,100]" />'
                            "</node></hierarchy>"
                        ),
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
                        "xml": (
                            '<hierarchy><node class="android.widget.FrameLayout" '
                            'bounds="[0,0][100,100]"><node text="Bluetooth" '
                            'resource-id="android:id/switch_widget" '
                            'clickable="true" bounds="[0,0][100,100]" />'
                            "</node></hierarchy>"
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
                    "params": {"on_or_off": "on"},
                    "replay_seed": 111,
                    "step_count": 1,
                    "retained_source_run_log": str(source_run_log),
                    "method": "ours",
                    "latest_official_success_source": True,
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
    return index


def test_appagent_preflight_uses_canonical_runlog_not_store_provenance(
    tmp_path: Path,
) -> None:
    index = _write_source_index(tmp_path / "source")
    payload = json.loads(index.read_text(encoding="utf-8"))
    row = payload["SystemBluetoothTurnOn"]
    row["store_provenance_sha256"] = "0" * 64
    index.write_text(json.dumps(payload), encoding="utf-8")

    result = appagent_source.preflight_appagent_source(
        index_path=index,
        task_name="SystemBluetoothTurnOn",
    )

    assert result["ready"] is True
    assert result["source_run_log"] == str(
        (tmp_path / "source" / "source.run_log.json").resolve()
    )
    assert result["grounding"]["grounding_source"] == (
        "canonical_androidworld_run_log"
    )


def test_appagent_source_generation_runs_each_phase_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    index = _write_source_index(tmp_path / "source")
    bundle = tmp_path / "bundle"
    calls = {"episode": 0, "documents": 0, "seal": 0}
    captured: dict[str, object] = {}
    monkeypatch.setenv(
        "OMNIFLOW_APPAGENT_SOURCE_ENVIRONMENT_REPAIR_REASON",
        "launch_source_app_package_from_teacher_contract@7709f60",
    )

    monkeypatch.setattr(
        appagent_source,
        "build_appagent_teacher_source",
        lambda *_args, **_kwargs: {
            "action_count": 1,
            "demo_action_count": 1,
            "actions": [
                {
                    "source_step_index": 0,
                    "action": {
                        "type": "click",
                        "params": {
                            "source_context": {
                                "element": {
                                    "text": "Bluetooth",
                                    "resource_id": (
                                        "android:id/switch_widget"
                                    ),
                                }
                            }
                        },
                    },
                }
            ],
        },
    )
    episode_output = tmp_path / "episode-output"

    def build_episode(
        _item,
        **kwargs,
    ) -> pipeline.CommandSpec:
        captured["episode_kwargs"] = kwargs
        return pipeline.CommandSpec(
            label="appagent-source",
            argv=["python", "source.py"],
            env={},
            cwd=tmp_path,
            output_path=episode_output,
        )

    monkeypatch.setattr(
        pipeline,
        "build_appagent_androidworld_command",
        build_episode,
    )

    def run_episode(_spec: pipeline.CommandSpec) -> int:
        calls["episode"] += 1
        episode_output.mkdir()
        (episode_output / "task_results.jsonl").write_text("{}\n")
        (bundle / "apps" / "settings").mkdir(parents=True)
        return 0

    monkeypatch.setattr(pipeline, "run_command", run_episode)
    monkeypatch.setattr(
        appagent_source,
        "validate_appagent_source_demo",
        lambda **_kwargs: {},
    )

    def generate_documents(**kwargs):
        calls["documents"] += 1
        log_path = Path(kwargs["log_path"])
        usage_path = Path(kwargs["usage_path"])
        log_path.parent.mkdir(parents=True)
        log_path.write_text("ok\n")
        usage_path.write_text(
            json.dumps(
                {
                    "model": "qwen3-vl-plus",
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                }
            )
            + "\n"
        )
        return {
            "models": ["qwen3-vl-plus"],
            "log_path": str(log_path),
            "usage_path": str(usage_path),
            "wall_sec": 1.0,
        }

    monkeypatch.setattr(
        appagent_source,
        "run_official_document_generation",
        generate_documents,
    )

    def seal(**kwargs):
        calls["seal"] += 1
        captured["seal_kwargs"] = kwargs
        return {"source_method": kwargs["source_method"]}

    monkeypatch.setattr(appagent_source, "seal_appagent_demo_memory", seal)

    result = appagent_source.prepare_appagent_demo_memory(
        index_path=index,
        task_name="SystemBluetoothTurnOn",
        appagent_root=tmp_path / "appagent",
        android_world_root=tmp_path / "android_world",
        memory_root=bundle,
        model="qwen3-vl-plus",
    )

    assert calls == {"episode": 1, "documents": 1, "seal": 1}
    episode_kwargs = captured["episode_kwargs"]
    assert episode_kwargs["task_random_seed"] == 111
    assert episode_kwargs["fixed_task_seed"] is True
    assert episode_kwargs["fixed_task_params"] is True
    assert episode_kwargs["max_steps"] == 2
    seal_kwargs = captured["seal_kwargs"]
    assert seal_kwargs["source_method"] == "ours"
    assert seal_kwargs["document_generation_model"] == "qwen3-vl-plus"
    assert result["source_method"] == "ours"
    command = json.loads(
        (bundle / "source_episode_command.json").read_text(encoding="utf-8")
    )
    assert command["model_attempts"] == 1
    assert command["episode_retries"] == 0
    assert command["source_environment_repair_reason"] == (
        "launch_source_app_package_from_teacher_contract@7709f60"
    )


def test_appagent_source_failure_marker_forbids_retry(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    appagent_source._write_failure_marker(bundle, RuntimeError("failed"))
    marker = json.loads(
        (bundle / "prep_failure.json").read_text(encoding="utf-8")
    )
    assert marker["retry_allowed"] is False
    assert marker["error_type"] == "RuntimeError"


def test_appagent_teacher_input_replaces_existing_field_text(
    tmp_path: Path,
) -> None:
    teacher_source = _write_appagent_teacher_source(
        tmp_path,
        task_name="AudioRecorderRecordAudioWithFileName",
        action={
            "type": "input_text",
            "params": {
                "text": "G367_conference.m4a",
                "source_context": {
                    "element": {
                        "resource_id": (
                            "com.dimowner.audiorecorder:id/input_name"
                        ),
                        "text": "Record-1",
                    }
                },
            },
        },
    )
    actions: list[dict] = []
    xml = (
        '<hierarchy class="android.widget.FrameLayout" '
        'bounds="[0,0][220,100]"><node class="android.widget.EditText" '
        'text="Record-1" '
        'resource-id="com.dimowner.audiorecorder:id/input_name" '
        'editable="true" clickable="true" enabled="true" '
        'bounds="[10,10][200,80]" /></hierarchy>'
    )
    env = SimpleNamespace(
        execute_action=actions.append,
        get_state=lambda: SimpleNamespace(
            xml=xml,
            pixels=np.zeros((100, 220, 3), dtype=np.uint8),
        ),
    )

    def draw_elements(source, destination, *_args, **_kwargs):
        Path(destination).write_bytes(Path(source).read_bytes())

    agent = appagent_adapter.AppAgentTeacherAgent(
        env=env,
        official_runtime=SimpleNamespace(
            min_dist=0.0,
            request_interval=0.0,
            draw_elements=draw_elements,
        ),
        teacher_source=teacher_source,
        workspace_root=tmp_path / "workspace",
        demo_name="record_with_name",
        action_factory=lambda **kwargs: kwargs,
    )
    agent.set_current_task(
        "AudioRecorderRecordAudioWithFileName",
        "Record with a file name.",
        {"app_names": ["audio recorder"]},
    )

    result = agent.step("Record with a file name.")

    assert result.done is False
    assert actions[-1] == {
        "action_type": "input_text",
        "text": "G367_conference.m4a",
        "clear_text": True,
    }


def test_appagent_teacher_uses_native_androidworld_observation(
    tmp_path: Path,
) -> None:
    teacher_source = _write_appagent_teacher_source(
        tmp_path,
        task_name="BrowserDraw",
        action={
            "type": "click",
            "params": {
                "target_description": "6.50 kB",
                "source_context": {"element": {"text": "6.50 kB"}},
            },
        },
    )

    forest = SimpleNamespace(
        windows=[
            SimpleNamespace(
                id=1,
                title="Downloads",
                tree=SimpleNamespace(
                    nodes=[
                        _forest_node(1, (0, 0, 220, 100), child_ids=(2, 3)),
                        _forest_node(2, (0, 0, 40, 20), clickable=True),
                        _forest_node(
                            3,
                            (40, 20, 200, 90),
                            child_ids=(4,),
                            clickable=True,
                        ),
                        _forest_node(4, (70, 50, 130, 70), text="6.50 kB"),
                    ]
                ),
            )
        ]
    )
    ui_elements = [
        SimpleNamespace(
            text="Search",
            bbox_pixels=SimpleNamespace(
                x_min=0,
                y_min=0,
                x_max=40,
                y_max=20,
            ),
            is_clickable=True,
            package_name="com.google.android.documentsui",
        ),
        SimpleNamespace(
            text="6.50 kB",
            bbox_pixels=SimpleNamespace(
                x_min=70,
                y_min=50,
                x_max=130,
                y_max=70,
            ),
            package_name="com.google.android.documentsui",
        ),
    ]
    actions: list[dict] = []
    env = SimpleNamespace(
        execute_action=actions.append,
        get_state=lambda: SimpleNamespace(
            pixels=np.zeros((100, 220, 3), dtype=np.uint8),
            forest=forest,
            ui_elements=ui_elements,
        ),
    )

    agent = appagent_adapter.AppAgentTeacherAgent(
        env=env,
        official_runtime=SimpleNamespace(
            min_dist=0.0,
            request_interval=0.0,
            draw_elements=_copy_labeled_screenshot,
        ),
        teacher_source=teacher_source,
        workspace_root=tmp_path / "workspace",
        demo_name="browser_draw",
        action_factory=lambda **kwargs: kwargs,
    )
    agent.set_current_task(
        "BrowserDraw",
        "Open task.html and draw.",
        {"app_names": ["chrome"]},
    )

    result = agent.step("Open task.html and draw.")

    assert result.done is False
    assert actions == [{"action_type": "click", "x": 120, "y": 55}]


def test_appagent_demo_memory_rejects_manifest_teacher_count_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher_source = _write_appagent_teacher_source(
        tmp_path,
        task_name="BrowserDraw",
        action={
            "type": "click",
            "params": {
                "target_description": "6.50 kB",
                "source_context": {"element": {"text": "6.50 kB"}},
            },
        },
    )
    source_run_log = tmp_path / "source.run_log.json"
    monkeypatch.setattr(appagent_adapter, "_require_hash", lambda *_args: None)
    monkeypatch.setattr(appagent_adapter, "_tree_sha256", lambda *_args: "tree")
    monkeypatch.setattr(appagent_adapter, "_validate_demo_artifacts", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(appagent_adapter, "_validate_demo_docs", lambda *_args: 1)
    manifest = {
        "schema_version": appagent_adapter.APPAGENT_DEMO_MEMORY_SCHEMA,
        "official_appagent_revision": appagent_adapter.APPAGENT_OFFICIAL_REVISION,
        "task_name": "BrowserDraw",
        "source_seed": 111,
        "official_source_success": True,
        "source_episode_metrics": {
            "duration_sec": 1.0,
            "wall_sec": 1.0,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
        "doc_generation_usage": {
            "model_calls": 1,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
            "wall_sec": 1.0,
        },
        "prep_wall_sec": 1.0,
        "teacher_complete": True,
        "teacher_action_count": 2,
        "teacher_actions_consumed": 2,
        "demo_action_count": 1,
        "uses_omniflow_function": False,
        "target_inputs_read": False,
        "target_observations_read": False,
        "validator_state_read_for_memory": False,
        "teacher_source": str(teacher_source),
        "teacher_source_sha256": hashlib.sha256(
            teacher_source.read_bytes()
        ).hexdigest(),
        "source_result": "source_result.jsonl",
        "source_result_sha256": "unused",
        "document_generation_log": "document.log",
        "document_generation_log_sha256": "unused",
        "document_generation_usage_path": "document_usage.jsonl",
        "document_generation_usage_sha256": "unused",
        "demo_root": "demo",
        "demo_sha256": "tree",
        "demo_docs_root": "docs",
        "demo_docs_sha256": "tree",
        "demo_docs_file_count": 1,
        "source_run_log": str(source_run_log),
        "source_run_log_sha256": hashlib.sha256(
            source_run_log.read_bytes()
        ).hexdigest(),
    }
    (tmp_path / appagent_adapter.APPAGENT_DEMO_MANIFEST).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="appagent_demo_memory_teacher_action_count_mismatch",
    ):
        appagent_adapter.validate_appagent_demo_memory(
            tmp_path,
            task_name="BrowserDraw",
            source_run_log=source_run_log,
        )


def test_appagent_warm_command_carries_unified_action_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_run_log = tmp_path / "source.run_log.json"
    source_run_log.write_text("{}", encoding="utf-8")
    action_source = tmp_path / "teacher_source.json"
    action_source.write_text("{}", encoding="utf-8")
    base_spec = pipeline.CommandSpec(
        label="base",
        argv=["python", "launch.py"],
        env={},
        cwd=tmp_path,
        output_path=tmp_path / "output",
    )
    monkeypatch.setattr(
        pipeline,
        "build_e2e_command",
        lambda *_args, **_kwargs: base_spec,
    )
    item = pipeline.ArchivedRunLog(
        task="BrowserDraw",
        goal="Open task.html and draw.",
        params={},
        source_run_log=source_run_log,
        replay_seed=111,
        step_count=2,
        meta={},
    )

    spec = pipeline.build_appagent_androidworld_command(
        item,
        method_name="appagent_demo",
        target=pipeline.DeviceTarget("small5554", "emulator-5554", 5554),
        android_world_root=tmp_path / "android_world",
        output_root=tmp_path / "output",
        appagent_root=tmp_path / "AppAgent",
        docs_root=tmp_path / "docs",
        action_source=action_source,
        max_steps=20,
        timeout_sec=60,
        task_random_seed=113,
        fixed_task_seed=True,
        fixed_task_params=True,
        task_params_override={},
        perform_emulator_setup=False,
        adb_path="adb",
        repo_root=tmp_path,
    )

    action_source_index = spec.argv.index("--appagent-action-source")
    assert spec.argv[action_source_index + 1] == str(action_source.resolve())
    assert spec.metadata["appagent_action_source"] == str(action_source.resolve())
