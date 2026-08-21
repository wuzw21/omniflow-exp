from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
from PIL import Image
import pytest
from runlog_fixtures import androidworld_run_log, androidworld_state

from src.experiment import run_task as pipeline
from src.experiment import appagent_source
from src.experiment.source_records import CanonicalRunLog
from src.integrations import appagent as appagent_adapter


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
                "schema_version": appagent_adapter.APPAGENT_SOURCE_SCHEMA,
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


def test_appagent_document_uid_matches_official_size_based_identity() -> None:
    source_xml = (
        '<hierarchy bounds="[0,0][1080,2400]"><node index="" '
        'class="android.widget.ImageView" content-desc="Shutter" '
        'clickable="true" bounds="[0,2085][1080,2400]" /></hierarchy>'
    )
    target_xml = (
        '<hierarchy bounds="[0,0][720,1280]"><node index="" '
        'class="android.widget.ImageView" content-desc="Shutter" '
        'clickable="true" bounds="[0,1070][720,1280]" /></hierarchy>'
    )

    source_elements = appagent_adapter.appagent_elements_from_xml(
        source_xml,
        min_dist=0.0,
    )
    target_elements = appagent_adapter.appagent_elements_from_xml(
        target_xml,
        min_dist=0.0,
    )

    assert source_elements[0].uid == (
        "_1080_2400_android.widget.ImageView_1080_315_Shutter_"
    )
    assert target_elements[0].uid == (
        "_720_1280_android.widget.ImageView_720_210_Shutter_"
    )


def test_appagent_focusable_uid_omits_index_like_official_executor() -> None:
    xml = (
        '<hierarchy bounds="[0,0][720,1280]">'
        '<node index="7" class="android.widget.EditText" focusable="true" '
        'bounds="[10,20][710,220]" />'
        "</hierarchy>"
    )

    elements = appagent_adapter.appagent_elements_from_xml(xml, min_dist=0.0)

    assert elements[0].uid == "_720_1280_android.widget.EditText_700_200"


def test_appagent_teacher_grounds_anonymous_source_fab_without_coordinates(
    tmp_path: Path,
) -> None:
    source_xml = (
        '<hierarchy bounds="[0,0][720,1280]">'
        '<node clickable="true" focusable="true" '
        'bounds="[0,48][112,160]" />'
        '<node focusable="true" scrollable="true" '
        'bounds="[0,160][720,1232]" />'
        '<node clickable="true" focusable="true" '
        'bounds="[576,1056][688,1168]" />'
        "</hierarchy>"
    )
    source_run_log = tmp_path / "source.run_log.json"
    source_run_log.write_text(
        json.dumps(
            androidworld_run_log(
                [{"action_type": "click", "x": 632, "y": 1112}],
                observations=[
                    androidworld_state(
                        "expense-home",
                        forest=source_xml,
                        width=720,
                        height=1280,
                    )
                ],
                task_name="ExpenseAddMultiple",
            )
        ),
        encoding="utf-8",
    )

    teacher_source = appagent_adapter.build_appagent_teacher_source(
        source_run_log,
        task_name="ExpenseAddMultiple",
    )

    params = teacher_source["actions"][0]["action"]["params"]
    assert params == {"source_appagent_tag": 2}
    assert appagent_adapter._contains_source_coordinates(params) is False
    grounded = appagent_adapter.ground_appagent_teacher_action(
        source_xml,
        teacher_source["actions"][0]["action"],
        min_dist=30.0,
    )
    assert grounded.tag == 2
    assert grounded.match_reason == "source_appagent_tag"


def _install_androidworld_app_registry(
    monkeypatch: pytest.MonkeyPatch,
    controller: object,
) -> None:
    adb_utils = SimpleNamespace(
        get_all_apps=lambda actual_controller: (
            ["chrome", "files"] if actual_controller is controller else []
        ),
        launch_app=lambda app_name, actual_controller: (
            app_name if actual_controller is controller else None
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
                "app_name": "files",
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

    def get_state(*, wait_to_stabilize: bool = False):
        assert wait_to_stabilize is True
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
        (
            "action",
            {
                "action_type": "open_app",
                "app_name": "files",
            },
        )
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


def test_appagent_teacher_executes_back_as_unrecorded_androidworld_control(
    tmp_path: Path,
) -> None:
    source_run_log = tmp_path / "source.run_log.json"
    run_log = androidworld_run_log(
        [
            {"action_type": "navigate_back"},
            {"action_type": "click", "x": 50, "y": 50},
        ],
        observations=[
            androidworld_state("keyboard", width=100, height=100),
            androidworld_state("form", width=100, height=100),
        ],
        task_name="BrowserDraw",
        run_id="browser-draw-back-source",
    )
    run_log["steps"][1]["metadata"] = {
        "source_context": {"element": {"text": "6.50 kB"}}
    }
    source_run_log.write_text(json.dumps(run_log), encoding="utf-8")
    teacher_source = tmp_path / "teacher_source.json"
    teacher_source.write_text(
        json.dumps(
            appagent_adapter.build_appagent_teacher_source(
                source_run_log,
                task_name="BrowserDraw",
            )
        ),
        encoding="utf-8",
    )
    events: list[tuple[str, object]] = []
    xml = (
        '<hierarchy><node class="android.widget.FrameLayout" '
        'bounds="[0,0][100,100]"><node text="6.50 kB" clickable="true" '
        'bounds="[0,0][100,100]" /></node></hierarchy>'
    )

    def get_state(*, wait_to_stabilize: bool = False):
        assert wait_to_stabilize is True
        events.append(("observe", None))
        return SimpleNamespace(
            xml=xml,
            pixels=np.zeros((100, 100, 3), dtype=np.uint8),
        )

    agent = appagent_adapter.AppAgentTeacherAgent(
        env=SimpleNamespace(
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

    back_result = agent.step("Open task.html and draw.")
    click_result = agent.step("Open task.html and draw.")

    assert back_result.done is False
    assert click_result.done is False
    assert events == [
        ("action", {"action_type": "navigate_back"}),
        ("observe", None),
        ("action", {"action_type": "click", "x": 50, "y": 50}),
    ]
    assert back_result.data["demo_actions_consumed"] == 0
    assert click_result.data["demo_actions_consumed"] == 1


def test_appagent_deployment_uses_native_controller_after_task_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = object()
    _install_androidworld_app_registry(monkeypatch, controller)
    events: list[tuple[str, object]] = []
    xml = (
        '<hierarchy bounds="[0,0][100,100]"><node text="6.50 kB" '
        'clickable="true" bounds="[0,0][100,100]" /></hierarchy>'
    )

    def get_screenshot(prefix: str, save_dir: str) -> str:
        path = Path(save_dir) / f"{prefix}.png"
        path.write_bytes(b"image")
        events.append(("screenshot", path))
        return str(path)

    def get_xml(prefix: str, save_dir: str) -> str:
        path = Path(save_dir) / f"{prefix}.xml"
        path.write_text(xml, encoding="utf-8")
        events.append(("xml", path))
        return str(path)

    native_controller = SimpleNamespace(
        width=100,
        height=100,
        get_screenshot=get_screenshot,
        get_xml=get_xml,
    )

    def draw_elements(source, destination, *_args, **_kwargs):
        Path(destination).write_bytes(Path(source).read_bytes())

    agent = appagent_adapter.AppAgentAndroidWorldAgent(
        env=SimpleNamespace(
            controller=controller,
        ),
        official_runtime=SimpleNamespace(
            min_dist=0.0,
            request_interval=0.0,
            collect_elements=lambda _path: [],
            draw_elements=draw_elements,
            build_task_prompt=lambda **_kwargs: "prompt",
            parse_response=lambda *_args, **_kwargs: ["FINISH"],
        ),
        controller=native_controller,
        llm=SimpleNamespace(
            get_model_response=lambda *_args, **_kwargs: (True, "finish")
        ),
        output_root=tmp_path / "output",
        docs_root=None,
    )
    agent.set_current_task(
        "BrowserDraw",
        "Open task.html and draw.",
        {"app_names": ["chrome"]},
    )

    result = agent.step("Open task.html and draw.")

    assert result.done is True
    assert result.data["error"] is None
    assert [event[0] for event in events] == ["screenshot", "xml"]
    assert result.data["startup_actions_executed"] == 1


def test_appagent_model_failure_does_not_parse_or_retry(tmp_path: Path) -> None:
    calls = {"model": 0, "parse": 0}

    def model_response(*_args, **_kwargs):
        calls["model"] += 1
        return False, "upstream error"

    def parse_response(*_args, **_kwargs):
        calls["parse"] += 1
        return ["ERROR"]

    def draw_elements(source, destination, *_args, **_kwargs):
        Path(destination).write_bytes(Path(source).read_bytes())

    xml = (
        '<hierarchy bounds="[0,0][100,100]">'
        '<node class="android.widget.Button" clickable="true" '
        'bounds="[0,0][100,100]" />'
        "</hierarchy>"
    )

    def get_screenshot(prefix: str, save_dir: str) -> str:
        path = Path(save_dir) / f"{prefix}.png"
        path.write_bytes(b"image")
        return str(path)

    def get_xml(prefix: str, save_dir: str) -> str:
        path = Path(save_dir) / f"{prefix}.xml"
        path.write_text(xml, encoding="utf-8")
        return str(path)

    agent = appagent_adapter.AppAgentAndroidWorldAgent(
        env=SimpleNamespace(),
        official_runtime=SimpleNamespace(
            min_dist=0.0,
            request_interval=0.0,
            collect_elements=lambda _path: [],
            draw_elements=draw_elements,
            build_task_prompt=lambda **_kwargs: "prompt",
            parse_response=parse_response,
        ),
        controller=SimpleNamespace(
            width=100,
            height=100,
            get_screenshot=get_screenshot,
            get_xml=get_xml,
        ),
        llm=SimpleNamespace(get_model_response=model_response),
        output_root=tmp_path / "output",
        docs_root=None,
    )

    result = agent.step("Do the task.")

    assert result.done is True
    assert result.data["error"] == "appagent_model_response_failed"
    assert calls == {"model": 1, "parse": 0}


def _write_source_index(root: Path) -> Path:
    root.mkdir(parents=True)
    before_image = root / "state-0.png"
    after_image = root / "state-1.png"
    Image.new("RGB", (100, 100), "white").save(before_image)
    Image.new("RGB", (100, 100), "black").save(after_image)
    forest = (
        '<hierarchy><node class="android.widget.FrameLayout" '
        'bounds="[0,0][100,100]"><node text="Bluetooth" '
        'resource-id="android:id/switch_widget" '
        'clickable="true" bounds="[0,0][100,100]" />'
        "</node></hierarchy>"
    )
    before_state = androidworld_state(
        "state-0",
        forest=forest,
        width=100,
        height=100,
    )
    before_state["pixels"] = {
        "path": str(before_image.resolve()),
        "sha256": hashlib.sha256(before_image.read_bytes()).hexdigest(),
        "width": 100,
        "height": 100,
        "mime_type": "image/png",
    }
    after_state = androidworld_state(
        "state-1",
        forest=forest,
        width=100,
        height=100,
    )
    after_state["pixels"] = {
        "path": str(after_image.resolve()),
        "sha256": hashlib.sha256(after_image.read_bytes()).hexdigest(),
        "width": 100,
        "height": 100,
        "mime_type": "image/png",
    }
    run_log = androidworld_run_log(
        [{"action_type": "click", "x": 50, "y": 50}],
        observations=[before_state],
        task_name="SystemBluetoothTurnOn",
        goal="Turn Bluetooth on.",
    )
    run_log["steps"][0]["next_observation"] = after_state
    source_run_log = root / "source.run_log.json"
    source_run_log.write_text(
        json.dumps(run_log),
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
                            forest
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
                    "method": "omniflow",
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
    assert result["source"]["run_log"] == str(
        (tmp_path / "source" / "source.run_log.json").resolve()
    )
    assert result["grounding"]["source"] == (
        "canonical_androidworld_run_log"
    )


def test_appagent_preflight_skips_back_control_grounding(
    tmp_path: Path,
) -> None:
    index = _write_source_index(tmp_path / "source")
    payload = json.loads(index.read_text(encoding="utf-8"))
    row = payload["SystemBluetoothTurnOn"]
    source_run_log = Path(row["retained_source_run_log"])
    source_payload = json.loads(source_run_log.read_text(encoding="utf-8"))
    click_step = dict(source_payload["steps"][0])
    back_step = dict(click_step)
    back_step["step_index"] = 0
    back_step["action"] = {"action_type": "navigate_back"}
    click_step["step_index"] = 1
    source_payload["steps"] = [back_step, click_step]
    source_run_log.write_text(json.dumps(source_payload), encoding="utf-8")
    row["step_count"] = 2
    row["source_run_log_sha256"] = hashlib.sha256(
        source_run_log.read_bytes()
    ).hexdigest()
    index.write_text(json.dumps(payload), encoding="utf-8")

    result = appagent_source.preflight_appagent_source(
        index_path=index,
        task_name="SystemBluetoothTurnOn",
    )

    assert result["ready"] is True
    assert result["action_count"] == 2
    assert result["demo_action_count"] == 1
    assert result["grounding"]["appagent_groundable_action_count"] == 1


def test_appagent_source_generation_is_offline_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = _write_source_index(tmp_path / "source")
    bundle = tmp_path / "bundle"
    captured: dict[str, object] = {}

    def convert(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "task_name": "SystemBluetoothTurnOn",
            "source_run_log": str(kwargs["source_run_log"]),
            "memory_root": str(kwargs["memory_root"]),
            "manifest": {"source_method": kwargs["source_method"]},
        }

    monkeypatch.setattr(
        "src.experiment.appagent_source.convert_runlog_to_appagent_memory",
        convert,
    )

    result = appagent_source.prepare_appagent_memory(
        index_path=index,
        task_name="SystemBluetoothTurnOn",
        appagent_root=tmp_path / "appagent",
        android_world_root=tmp_path / "android_world",
        memory_root=bundle,
        model="qwen3-vl-plus",
        evidence_roots=[tmp_path / "unused-old-evidence"],
    )

    assert captured["source_method"] == "omniflow"
    assert captured["appagent_root"] == tmp_path / "appagent"
    assert captured["memory_root"] == bundle
    assert result["source_method"] == "omniflow"
    assert result["source_emulator_used"] is False
    assert result["native_memory_evidence"] is None


def test_appagent_source_builds_native_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.run_log.json"
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"
    Image.new("RGB", (100, 100), "white").save(before)
    Image.new("RGB", (100, 100), "black").save(after)
    xml = (
        '<hierarchy><node text="Bluetooth" clickable="true" '
        'bounds="[0,0][100,100]" /></hierarchy>'
    )
    before_state = androidworld_state(
        "before",
        forest=xml,
        package_name="com.android.settings",
        width=100,
        height=100,
    )
    before_state["pixels"] = {
        "path": str(before.resolve()),
        "sha256": hashlib.sha256(before.read_bytes()).hexdigest(),
        "width": 100,
        "height": 100,
        "mime_type": "image/png",
    }
    after_state = androidworld_state(
        "after",
        forest=xml,
        package_name="com.android.settings",
        width=100,
        height=100,
    )
    after_state["pixels"] = {
        "path": str(after.resolve()),
        "sha256": hashlib.sha256(after.read_bytes()).hexdigest(),
        "width": 100,
        "height": 100,
        "mime_type": "image/png",
    }
    payload = androidworld_run_log(
        [{"action_type": "click", "x": 50, "y": 50}],
        observations=[before_state],
        task_name="SystemBluetoothTurnOn",
        goal="Turn Bluetooth on.",
    )
    payload["steps"][0]["next_observation"] = after_state
    source.write_text(json.dumps(payload), encoding="utf-8")

    class Runtime:
        min_dist = 30.0

        def __init__(self, _root: Path) -> None:
            pass

        def draw_elements(
            self,
            source_path: Path,
            target_path: Path,
            _elements: list[object],
            *,
            record_mode: bool,
        ) -> None:
            assert record_mode is True
            Image.open(source_path).save(target_path)

    def generate_docs(**kwargs: object) -> dict[str, object]:
        workspace = Path(str(kwargs["workspace_root"]))
        docs = workspace / "apps" / "settings" / "demo_docs"
        docs.mkdir(parents=True)
        (docs / "uid.txt").write_text(
            repr(
                {
                    "tap": "Turn Bluetooth on",
                    "text": "",
                    "v_swipe": "",
                    "h_swipe": "",
                    "long_press": "",
                }
            ),
            encoding="utf-8",
        )
        Path(str(kwargs["log_path"])).write_text("ok\n", encoding="utf-8")
        Path(str(kwargs["usage_path"])).write_text(
            json.dumps(
                {
                    "model": "qwen3-vl-plus",
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {"wall_sec": 0.1}

    monkeypatch.setattr(appagent_source, "OfficialAppAgentRuntime", Runtime)
    monkeypatch.setattr(
        appagent_source,
        "run_official_document_generation",
        generate_docs,
    )

    result = appagent_source.convert_runlog_to_appagent_memory(
        source_run_log=source,
        memory_root=tmp_path / "bundle",
        appagent_root=tmp_path / "appagent",
        model="qwen3-vl-plus",
    )

    manifest = result["manifest"]
    assert manifest["conversion_mode"] == "canonical_runlog_offline"
    assert manifest["native_memory_evidence"] is None
    demo = Path(manifest["demo_root"])
    record_lines = (demo / "record.txt").read_text(encoding="utf-8").splitlines()
    assert record_lines[0].startswith("tap(1):::")
    assert record_lines[1] == "stop"
    assert len(list((demo / "raw_screenshots").glob("*.png"))) == 2


def test_appagent_source_rejects_missing_screenshot(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.run_log.json"
    source.write_text(
        json.dumps(
            androidworld_run_log(
                [{"action_type": "click", "x": 50, "y": 50}],
                observations=[
                    androidworld_state(
                        "before",
                        forest=(
                            '<hierarchy><node text="Bluetooth" clickable="true" '
                            'bounds="[0,0][100,100]" /></hierarchy>'
                        ),
                        width=100,
                        height=100,
                    )
                ],
                task_name="SystemBluetoothTurnOn",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="appagent_source_screenshot_missing:0:before",
    ):
        appagent_source.convert_runlog_to_appagent_memory(
            source_run_log=source,
            memory_root=tmp_path / "bundle",
            appagent_root=tmp_path / "appagent",
            model="qwen3-vl-plus",
        )


def test_appagent_resolves_materialized_screenshot_by_exact_sha256(
    tmp_path: Path,
) -> None:
    image = tmp_path / "before.png"
    Image.new("RGB", (8, 6), "blue").save(image)
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    source = (
        tmp_path
        / "memory"
        / "objects"
        / "sha256"
        / "aa"
        / f"{'a' * 64}.json"
    )
    source.parent.mkdir(parents=True)
    source.write_text("{}", encoding="utf-8")
    materialized = (
        source.parent.parent / digest[:2] / f"{digest}.png"
    )
    materialized.parent.mkdir(parents=True)
    materialized.write_bytes(image.read_bytes())
    image.unlink()

    resolved = appagent_source._resolve_appagent_screenshot(
        {
            "path": str(image),
            "sha256": digest,
            "width": 8,
            "height": 6,
            "mime_type": "image/png",
        },
        source_run_log=source,
    )

    assert resolved == materialized.resolve()


def test_appagent_remaps_screenshot_from_copied_data_root(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "remote" / "data"
    source_run_log = (
        data_root
        / "androidworld"
        / "CameraTakePhoto"
        / "source5554"
        / "run_log.json"
    )
    source_run_log.parent.mkdir(parents=True)
    source_run_log.write_text("{}", encoding="utf-8")
    screenshot = (
        data_root
        / "androidworld"
        / "CameraTakePhoto"
        / "source5554"
        / "screenshots"
        / "before.png"
    )
    screenshot.parent.mkdir(parents=True)
    Image.new("RGB", (8, 6), "green").save(screenshot)

    resolved = appagent_source._resolve_appagent_screenshot(
        {
            "path": "/Users/wuzewen/Projects/Omni/OmniFlow-exp/data/"
            "androidworld/CameraTakePhoto/source5554/screenshots/before.png",
            "mime_type": "image/png",
        },
        source_run_log=source_run_log,
    )

    assert resolved == screenshot.resolve()


def test_appagent_projects_androidworld_ui_elements_to_xml() -> None:
    xml_text = appagent_source._appagent_observation_xml(
        {
            "pixels": None,
            "forest": None,
            "ui_elements": [
                {
                    "class_name": "android.widget.Button",
                    "text": "Save",
                    "content_description": "Save contact",
                    "resource_name": "com.example:id/save",
                    "package_name": "com.example",
                    "bbox_pixels": {
                        "x_min": 10,
                        "y_min": 20,
                        "x_max": 90,
                        "y_max": 60,
                    },
                    "is_clickable": True,
                    "is_editable": False,
                    "is_focusable": True,
                    "is_focused": True,
                    "is_scrollable": False,
                }
            ],
            "auxiliaries": None,
        }
    )

    assert 'text="Save"' in xml_text
    assert 'content-desc="Save contact"' in xml_text
    assert 'bounds="[10,20][90,60]"' in xml_text
    assert 'focused="true"' in xml_text


def test_appagent_marks_semantic_androidworld_target_interactive() -> None:
    xml_text = (
        '<hierarchy><node text="Network &amp; internet" '
        'resource-id="android:id/title" clickable="false" '
        'bounds="[144,579][475,633]" /></hierarchy>'
    )
    action = {
        "type": "click",
        "params": {
            "target_description": "Network & internet",
            "source_context": {
                "element": {
                    "text": "Network & internet",
                    "resource_id": "android:id/title",
                }
            },
        },
    }

    marked = appagent_adapter.mark_appagent_teacher_target_interactive(
        xml_text,
        action,
    )

    assert 'clickable="true"' in marked
    grounded = appagent_adapter.ground_appagent_teacher_action(
        marked,
        action,
        min_dist=30.0,
    )
    assert grounded.match_reason == "exact_visible_identity"


def test_androidworld_ui_elements_supply_appagent_package() -> None:
    assert appagent_source.androidworld_observation_package(
        {
            "pixels": None,
            "forest": None,
            "ui_elements": [
                {"package_name": "com.android.systemui"},
                {"package_name": "com.google.android.contacts"},
            ],
            "auxiliaries": {},
        }
    ) == "com.google.android.contacts"


def test_appagent_treats_source_app_ime_as_auxiliary_window() -> None:
    assert appagent_source._appagent_demo_package(
        {"package_name": "com.google.android.inputmethod.latin"},
        "com.dimowner.audiorecorder",
    ) == "com.dimowner.audiorecorder"


def test_appagent_source_failure_marker_forbids_retry(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    appagent_source._write_failure_marker(bundle, RuntimeError("failed"))
    marker = json.loads(
        (bundle / "prep_failure.json").read_text(encoding="utf-8")
    )
    assert marker["retry_allowed"] is False
    assert marker["error_type"] == "RuntimeError"


def test_native_memory_evidence_accepts_shared_runlog_provenance(
    tmp_path: Path,
) -> None:
    original_sha256 = "a" * 64
    canonical_runlog = tmp_path / "canonical.json"
    canonical_runlog.write_text(
        json.dumps(
            {
                "run_id": "shared-run",
                "provenance": {"source_sha256": original_sha256},
            }
        ),
        encoding="utf-8",
    )
    item = SimpleNamespace(
        task="ContactsAddContact",
        source_run_log=canonical_runlog,
    )
    evidence = tmp_path / "evidence"
    legacy_runlog = evidence / "grounded_teacher_run_log.json"
    legacy_runlog.parent.mkdir()
    legacy_runlog.write_text(
        json.dumps(
            {
                "run_id": "shared-run",
                "provenance": {"source_sha256": original_sha256},
            }
        ),
        encoding="utf-8",
    )
    demo_root = evidence / "apps" / "contacts" / "demos" / "demo"
    docs_root = evidence / "apps" / "contacts" / "demo_docs"
    document_root = evidence / "_document_generation"
    demo_root.mkdir(parents=True)
    docs_root.mkdir(parents=True)
    document_root.mkdir()
    (demo_root / "teacher_trace.jsonl").write_text(
        json.dumps({"source_step_index": 99, "action_type": "click"}) + "\n",
        encoding="utf-8",
    )
    (document_root / "document_generation.log").write_text("ok\n")
    (document_root / "document_generation_usage.jsonl").write_text(
        json.dumps({"model": "qwen3-vl-plus"}) + "\n",
        encoding="utf-8",
    )
    manifest = evidence / "appagent_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "official_appagent_revision": appagent_adapter.APPAGENT_OFFICIAL_REVISION,
                "task_name": "ContactsAddContact",
                "source_seed": 111,
                "source_run_id": "shared-run",
                "source_run_log": "/migrated/original/runlog.json",
                "source_run_log_sha256": hashlib.sha256(
                    legacy_runlog.read_bytes()
                ).hexdigest(),
                "app_name": "contacts",
                "demo_name": "demo",
                "demo_sha256": "demo",
                "demo_docs_sha256": "docs",
                "document_generation_usage_sha256": "usage",
            }
        ),
        encoding="utf-8",
    )

    selected = appagent_source._native_memory_evidence(
        item=item,
        teacher_source={
            "actions": [
                {"source_step_index": 1, "action": {"type": "click"}}
            ]
        },
        evidence_roots=[evidence],
        model="different-online-model",
    )

    assert selected["manifest"] == manifest.resolve()
    assert selected["document_model"] == "qwen3-vl-plus"


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
        get_state=lambda **_kwargs: SimpleNamespace(
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


def test_appagent_grounds_androidworld_edittext_with_false_editable_flag() -> None:
    xml = (
        '<hierarchy bounds="[0,0][720,1280]">'
        '<node class="android.widget.EditText" text="" editable="false" '
        'clickable="true" focusable="true" bounds="[0,160][720,590]" />'
        '</hierarchy>'
    )

    grounded = appagent_adapter.ground_appagent_teacher_action(
        xml,
        {"type": "input_text", "params": {"text": "Note body"}},
        min_dist=30.0,
    )

    assert grounded.tag == 1
    assert grounded.match_reason == "unique_current_editable"


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
        get_state=lambda **_kwargs: SimpleNamespace(
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


def test_appagent_teacher_waits_for_stable_androidworld_observation(
    tmp_path: Path,
) -> None:
    teacher_source = _write_appagent_teacher_source(
        tmp_path,
        task_name="BrowserDraw",
        action={
            "type": "click",
            "params": {
                "source_context": {"element": {"text": "Open"}},
            },
        },
    )
    get_state_calls: list[bool] = []
    xml = (
        '<hierarchy bounds="[0,0][220,100]">'
        '<node class="android.widget.Button" text="Open" clickable="true" '
        'bounds="[70,30][170,80]" />'
        '</hierarchy>'
    )

    def get_state(*, wait_to_stabilize: bool = False):
        get_state_calls.append(wait_to_stabilize)
        return SimpleNamespace(
            xml=xml if wait_to_stabilize else '<hierarchy />',
            pixels=np.zeros((100, 220, 3), dtype=np.uint8),
        )

    actions: list[dict] = []
    agent = appagent_adapter.AppAgentTeacherAgent(
        env=SimpleNamespace(get_state=get_state, execute_action=actions.append),
        official_runtime=SimpleNamespace(
            min_dist=0.0,
            request_interval=0.0,
            draw_elements=_copy_labeled_screenshot,
        ),
        teacher_source=teacher_source,
        workspace_root=tmp_path / "workspace",
        demo_name="stable_observation",
        action_factory=lambda **kwargs: kwargs,
    )
    agent.set_current_task(
        "BrowserDraw",
        "Open the item.",
        {"app_names": ["chrome"]},
    )

    result = agent.step("Open the item.")

    assert result.done is False
    assert get_state_calls == [True]
    assert actions == [{"action_type": "click", "x": 120, "y": 55}]


def test_appagent_memory_rejects_manifest_teacher_count_mismatch(
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
        "schema_version": appagent_adapter.APPAGENT_MEMORY_SCHEMA,
        "official_appagent_revision": appagent_adapter.APPAGENT_OFFICIAL_REVISION,
        "task_name": "BrowserDraw",
        "source_seed": 111,
        "conversion_mode": "source_episode",
        "source_emulator_used": True,
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
    (tmp_path / appagent_adapter.APPAGENT_MANIFEST).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="appagent_memory_teacher_action_count_mismatch",
    ):
        appagent_adapter.validate_appagent_memory(
            tmp_path,
            task_name="BrowserDraw",
            source_run_log=source_run_log,
        )


def test_appagent_warm_command_mounts_native_docs_memory(
    tmp_path: Path,
) -> None:
    source_run_log = tmp_path / "source.run_log.json"
    source_run_log.write_text("{}", encoding="utf-8")
    appagent_root = tmp_path / "AppAgent"
    (appagent_root / "scripts").mkdir(parents=True)
    (appagent_root / "run.py").write_text("print('official')\n", encoding="utf-8")
    (appagent_root / "scripts" / "task_executor.py").write_text(
        "import os\n"
        "doc_path = os.path.join(docs_dir, f\"{elem.uid}.txt\")\n",
        encoding="utf-8",
    )
    docs_root = tmp_path / "apps" / "audiorecorder" / "demo_docs"
    docs_root.mkdir(parents=True)
    (docs_root / "button.txt").write_text("{}", encoding="utf-8")
    item = CanonicalRunLog(
        task="BrowserDraw",
        goal="Open task.html and draw.",
        params={},
        source_run_log=source_run_log,
        replay_seed=111,
        step_count=2,
        meta={},
    )

    spec = pipeline.build_appagent_command(
        item,
        method_name="appagent",
        target=pipeline.DeviceTarget("small5554", "emulator-5554", 5554),
        android_world_root=tmp_path / "android_world",
        output_root=tmp_path / "output",
        appagent_root=appagent_root,
        docs_root=docs_root,
        max_steps=20,
        timeout_sec=60,
        model="GLM-4.6V",
        task_random_seed=113,
        fixed_task_seed=True,
        fixed_task_params=True,
        task_params_override={},
        perform_emulator_setup=False,
        adb_path="adb",
        repo_root=tmp_path,
    )

    assert spec.argv[1:5] == [
        "-m",
        "src.integrations.official_forward",
        "--baseline",
        "appagent",
    ]
    assert "--executor" in spec.argv
    assert spec.metadata["official_executor"].endswith(
        "/scripts/task_executor.py"
    )
    assert spec.metadata["appagent_docs_root"] == str(docs_root.resolve())
    assert spec.metadata["official_wrapper"].endswith("/run.py")
    assert spec.stdin_text == ""
    assert spec.metadata["external_forward_only"] is True
    assert spec.env["OPENAI_MODEL"] == "GLM-4.6V"
