from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.experiment import androidworld as pipeline
from src.experiment import appagent_source
from src.integrations import appagent_adapter


def _write_appagent_teacher_source(
    root: Path,
    *,
    task_name: str,
    source_app_package: str,
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
                "source_app_package": source_app_package,
                "actions": [
                    {
                        "source_step_index": 1,
                        "source_action_index": 0,
                        "action": action,
                    }
                ],
                "action_count": 1,
                "consumer": "appagent_official_human_demonstration",
                "adapter_scope": "human_demo_primitive_grounding_only",
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


def _write_source_index(root: Path) -> Path:
    root.mkdir(parents=True)
    source_run_log = root / "source.run_log.json"
    source_run_log.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.canonical_run_log.v1",
                "run_id": "source-run",
                "goal": "Turn Bluetooth on.",
                "status": "succeeded",
                "success": True,
                "steps": [
                    {
                        "step_index": 0,
                        "before_state_id": "state-0",
                        "action": {
                            "tool": "click",
                            "args": {"x": 500, "y": 500},
                        },
                        "result": {"success": True},
                        "after_state_id": "state-1",
                    }
                ],
            }
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


def test_appagent_deterministic_preflight_does_not_claim_output(
    tmp_path: Path,
) -> None:
    index = _write_source_index(tmp_path / "source")
    payload = json.loads(index.read_text(encoding="utf-8"))
    row = payload["SystemBluetoothTurnOn"]
    row["store_provenance_sha256"] = "0" * 64
    index.write_text(json.dumps(payload), encoding="utf-8")
    output_root = tmp_path / "never-created"

    with pytest.raises(ValueError, match="source_provenance_hash_mismatch"):
        appagent_source.prepare_appagent_demo_memory(
            index_path=index,
            task_name="SystemBluetoothTurnOn",
            appagent_root=tmp_path / "appagent",
            android_world_root=tmp_path / "android_world",
            memory_root=output_root,
            model="qwen3-vl-plus",
        )

    assert not output_root.exists()


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
        source_app_package="com.dimowner.audiorecorder",
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


def test_appagent_teacher_launches_source_app_package(tmp_path: Path) -> None:
    teacher_source = _write_appagent_teacher_source(
        tmp_path,
        task_name="BrowserDraw",
        source_app_package="com.google.android.documentsui",
        action={
            "type": "click",
            "params": {
                "target_description": "6.50 kB",
                "source_context": {"element": {"text": "6.50 kB"}},
            },
        },
    )
    launched: list[dict] = []
    agent = appagent_adapter.AppAgentTeacherAgent(
        env=SimpleNamespace(execute_action=launched.append),
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
    agent._ensure_app_started()

    assert agent.app_name == "chrome"
    assert launched == [
        {
            "action_type": "open_app",
            "app_name": "com.google.android.documentsui",
        }
    ]


def test_appagent_teacher_retries_partial_a11y_tree_after_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APPAGENT_APP_START_WAIT_SEC", "0")
    teacher_source = _write_appagent_teacher_source(
        tmp_path,
        task_name="BrowserDraw",
        source_app_package="com.google.android.documentsui",
        action={
            "type": "click",
            "params": {
                "target_description": "6.50 kB",
                "source_context": {"element": {"text": "6.50 kB"}},
            },
        },
    )
    partial_xml = (
        '<hierarchy class="android.widget.FrameLayout" '
        'bounds="[0,0][220,100]"><node package="com.android.systemui" '
        'class="android.widget.TextView" text="11:22" '
        'clickable="false" bounds="[0,0][100,20]" /></hierarchy>'
    )
    ready_xml = (
        '<hierarchy class="android.widget.FrameLayout" '
        'bounds="[0,0][220,100]"><node '
        'package="com.google.android.documentsui" '
        'class="android.widget.TextView" text="6.50 kB" '
        'clickable="true" bounds="[10,20][110,80]" /></hierarchy>'
    )
    state_xml = iter((partial_xml, ready_xml))
    actions: list[dict] = []
    env = SimpleNamespace(
        execute_action=actions.append,
        get_state=lambda: SimpleNamespace(
            xml=next(state_xml),
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
    assert result.data["teacher_actions_consumed"] == 1
    assert actions == [
        {
            "action_type": "open_app",
            "app_name": "com.google.android.documentsui",
        },
        {"action_type": "click", "x": 60, "y": 50},
    ]
    trace = json.loads(
        (
            tmp_path
            / "workspace/apps/chrome/demos/browser_draw/teacher_trace.jsonl"
        ).read_text(encoding="utf-8")
    )
    assert trace["observation_attempts"] == 2


def test_appagent_teacher_preserves_native_accessibility_hierarchy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APPAGENT_APP_START_WAIT_SEC", "0")
    teacher_source = _write_appagent_teacher_source(
        tmp_path,
        task_name="BrowserDraw",
        source_app_package="com.google.android.documentsui",
        action={
            "type": "click",
            "params": {
                "target_description": "6.50 kB",
                "source_context": {"element": {"text": "6.50 kB"}},
            },
        },
    )

    def forest_node(
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

    forest = SimpleNamespace(
        windows=[
            SimpleNamespace(
                id=1,
                title="Downloads",
                tree=SimpleNamespace(
                    nodes=[
                        forest_node(1, (0, 0, 220, 100), child_ids=(2, 3)),
                        forest_node(2, (0, 0, 40, 20), clickable=True),
                        forest_node(
                            3,
                            (40, 20, 200, 90),
                            child_ids=(4,),
                            clickable=True,
                        ),
                        forest_node(4, (70, 50, 130, 70), text="6.50 kB"),
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
    assert actions[-1] == {"action_type": "click", "x": 120, "y": 55}
