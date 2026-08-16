from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from PIL import Image
import pytest

from omniflow import Action
from src.integrations.android_world.host import AndroidWorldHost
from src.integrations.android_world.launch import (
    _ExperimentAgentAdapter,
    _androidworld_a11y_forwarder_installed,
    _ensure_androidworld_a11y_forwarder,
    _androidworld_setup_apps_for_suite,
    _wait_for_androidworld_a11y,
    _result_has_official_validator_conclusion,
    _runtime_execution_trace,
)


def test_androidworld_setup_uses_task_declared_app_dependencies() -> None:
    expense_app = object()
    markor_app = object()
    mappings = {"pro expense": expense_app, "markor": markor_app}
    suite = {
        "ExpenseTask": [
            SimpleNamespace(app_names=("pro expense", "markor")),
            SimpleNamespace(app_names=("pro expense",)),
        ]
    }

    assert _androidworld_setup_apps_for_suite(
        suite,
        get_app_mapping=mappings.get,
    ) == (expense_app, markor_app)


def test_androidworld_setup_rejects_unmapped_task_dependency() -> None:
    with pytest.raises(RuntimeError, match="setup app mapping missing: unknown"):
        _androidworld_setup_apps_for_suite(
            {"Task": [SimpleNamespace(app_names=("unknown",))]},
            get_app_mapping=lambda _name: None,
        )


def test_androidworld_waits_for_native_a11y_before_setup(monkeypatch) -> None:
    calls = 0

    def get_state(*, wait_to_stabilize: bool = False):
        nonlocal calls
        assert wait_to_stabilize is False
        calls += 1
        if calls < 3:
            raise RuntimeError("not ready")
        return SimpleNamespace(forest=object())

    monkeypatch.setattr("src.integrations.android_world.launch.time.sleep", lambda _: None)
    _wait_for_androidworld_a11y(SimpleNamespace(get_state=get_state))
    assert calls == 3


def test_androidworld_a11y_readiness_allows_cold_boot_recovery(monkeypatch) -> None:
    calls = 0

    def get_state(*, wait_to_stabilize: bool = False):
        nonlocal calls
        assert wait_to_stabilize is False
        calls += 1
        if calls < 4:
            raise RuntimeError("forwarder still reconnecting")
        return SimpleNamespace(forest=object())

    monkeypatch.setattr("src.integrations.android_world.launch.time.sleep", lambda _: None)
    _wait_for_androidworld_a11y(SimpleNamespace(get_state=get_state))
    assert calls == 4


def test_androidworld_a11y_readiness_refreshes_official_controller_first() -> None:
    calls: list[str] = []

    class Env:
        def refresh_env(self) -> None:
            calls.append("refresh")

        def get_state(self, *, wait_to_stabilize: bool = False):
            assert wait_to_stabilize is False
            calls.append("state")
            return SimpleNamespace(forest=object())

    _wait_for_androidworld_a11y(Env())

    assert calls == ["refresh", "state"]


def test_androidworld_reuses_installed_a11y_forwarder(monkeypatch) -> None:
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout="package:/data/app/base.apk\n",
        )

    monkeypatch.setattr(
        "src.integrations.android_world.launch.subprocess.run",
        run,
    )

    assert _androidworld_a11y_forwarder_installed(
        console_port=5554,
        adb_path="/sdk/adb",
    )
    assert calls[0][0] == [
        "/sdk/adb",
        "-s",
        "emulator-5554",
        "shell",
        "pm",
        "path",
        "com.google.androidenv.accessibilityforwarder",
    ]


def test_androidworld_installs_a11y_forwarder_when_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.integrations.android_world.launch.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=""),
    )

    assert not _androidworld_a11y_forwarder_installed(
        console_port=5564,
        adb_path="adb",
    )


def test_androidworld_installs_cached_a11y_forwarder(monkeypatch, tmp_path) -> None:
    apk = tmp_path / "forwarder.apk"
    apk.write_bytes(b"official-apk")
    installed = iter((False, True))
    monkeypatch.setattr(
        "src.integrations.android_world.launch._androidworld_a11y_forwarder_installed",
        lambda **_kwargs: next(installed),
    )
    monkeypatch.setattr(
        "src.integrations.android_world.launch.ANDROIDWORLD_A11Y_FORWARDER_SHA256",
        hashlib.sha256(apk.read_bytes()).hexdigest(),
    )
    calls = []
    monkeypatch.setattr(
        "src.integrations.android_world.launch.subprocess.run",
        lambda argv, **_kwargs: calls.append(argv) or SimpleNamespace(returncode=0),
    )

    assert _ensure_androidworld_a11y_forwarder(
        console_port=5554, adb_path="/sdk/adb", apk_path=str(apk)
    )
    assert calls == [["/sdk/adb", "-s", "emulator-5554", "install", "-r", str(apk)]]


def _official_state(**overrides):
    values = {
        "pixels": None,
        "forest": None,
        "ui_elements": [],
        "auxiliaries": {},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _ui_element(
    text: str = "Settings",
    bounds: tuple[int, int, int, int] = (0, 0, 4, 3),
):
    return SimpleNamespace(
        text=text,
        package_name="com.android.settings",
        bbox_pixels=SimpleNamespace(
            x_min=bounds[0],
            y_min=bounds[1],
            x_max=bounds[2],
            y_max=bounds[3],
        ),
    )


def test_input_text_coordinates_reach_androidworld_json_action(monkeypatch) -> None:
    class JSONAction:
        def __init__(
            self,
            action_type=None,
            index=None,
            x=None,
            y=None,
            text=None,
            direction=None,
            goal_status=None,
            app_name=None,
            keycode=None,
            clear_text=None,
        ):
            self.action_type = action_type
            self.x = x
            self.y = y
            self.text = text
            self.clear_text = clear_text

    module = SimpleNamespace(JSONAction=JSONAction)
    monkeypatch.setattr(
        "src.integrations.android_world.host.importlib.import_module",
        lambda name: module if name == "android_world.env.json_action" else None,
    )
    env = SimpleNamespace(device_screen_size=(720, 1280))

    action = AndroidWorldHost(env)._json_action(
        Action(
            "input_text",
            {"text": "I may repeat this", "x": 500.0, "y": 594.921875},
        )
    )

    assert action.action_type == "input_text"
    assert action.x == 360.0
    assert action.y == 761.5
    assert action.text == "I may repeat this"
    assert action.clear_text is True


def test_runtime_execution_trace_preserves_prepared_function_action() -> None:
    trace = [
        {
            "before_state_id": "before",
            "action": {
                "tool": "input_text",
                "args": {
                    "text": "I may repeat this",
                    "x": 500.0,
                    "y": 594.921875,
                },
            },
            "result": {"success": True},
            "after_state_id": "after",
            "metadata": {
                "function_id": "expense_add_multiple_replay",
                "function_step_index": 6,
            },
        }
    ]

    assert _runtime_execution_trace(SimpleNamespace(detail={"trace": trace})) == trace


def test_androidworld_skipped_episode_is_not_validator_conclusion() -> None:
    assert not _result_has_official_validator_conclusion(
        {
            "is_successful": 0.0,
            "exception_info": "FileNotFoundError: app database missing",
        }
    )
    assert _result_has_official_validator_conclusion(
        {"is_successful": 0.0, "exception_info": None}
    )


def test_observe_preserves_one_official_androidworld_state(tmp_path) -> None:
    state = _official_state(
        pixels=Image.new("RGB", (4, 3), color="blue"),
        forest={"source": "official-forest"},
        ui_elements=[_ui_element()],
        auxiliaries={"source": "androidworld"},
    )

    class Env:
        device_screen_size = (4, 3)
        logical_screen_size = (4, 3)
        foreground_activity_name = "com.android.settings/.Settings"

        def __init__(self) -> None:
            self.calls = 0

        def get_state(self, wait_to_stabilize: bool = False):
            assert wait_to_stabilize is True
            self.calls += 1
            return state

    env = Env()
    observation = AndroidWorldHost(env, evidence_root=tmp_path).observe(
        xml=True,
        screenshot=True,
        app_info=True,
    )

    assert env.calls == 1
    saved = observation.extra["androidworld_state"]
    assert set(saved) == {"pixels", "forest", "ui_elements", "auxiliaries"}
    assert saved["forest"] == {"source": "official-forest"}
    assert saved["ui_elements"] == [
        {
            "text": "Settings",
            "package_name": "com.android.settings",
            "bbox_pixels": {
                "x_min": 0,
                "y_min": 0,
                "x_max": 4,
                "y_max": 3,
            },
        }
    ]
    assert saved["auxiliaries"] == {"source": "androidworld"}
    screenshot_path = Path(saved["pixels"]["path"])
    assert screenshot_path.is_file()
    assert screenshot_path.is_absolute()
    assert hashlib.sha256(screenshot_path.read_bytes()).hexdigest() == saved["pixels"][
        "sha256"
    ]
    assert saved["pixels"]["width"] == 4
    assert saved["pixels"]["height"] == 3
    assert observation.package_name == "com.android.settings"
    assert "Settings" in str(observation.xml)


def test_observe_uses_official_stable_state() -> None:
    calls: list[bool] = []

    class Env:
        device_screen_size = (4, 3)
        logical_screen_size = (4, 3)
        foreground_activity_name = "com.android.settings/.Settings"

        def get_state(self, wait_to_stabilize: bool = False):
            calls.append(wait_to_stabilize)
            return _official_state(ui_elements=[_ui_element()])

    observation = AndroidWorldHost(Env()).observe()

    assert observation.package_name == "com.android.settings"
    assert calls == [True]


def test_observe_does_not_replace_failed_official_state() -> None:
    class Env:
        controller = SimpleNamespace(
            get_ui_elements=lambda: [_ui_element("fallback")],
            get_screenshot=lambda: Image.new("RGB", (1, 1)),
        )

        def get_state(self, wait_to_stabilize: bool = False):
            raise RuntimeError("official state unavailable")

    with pytest.raises(RuntimeError, match="official state unavailable"):
        AndroidWorldHost(Env()).observe()


def test_observe_requires_all_official_state_fields() -> None:
    incomplete = SimpleNamespace(pixels=None, forest=None, ui_elements=[])

    with pytest.raises(
        ValueError,
        match="androidworld_state_fields_missing:auxiliaries",
    ):
        AndroidWorldHost(
            SimpleNamespace(get_state=lambda **_: incomplete)
        ).observe()


def test_observe_ignores_non_official_xml_field() -> None:
    state = _official_state(xml="<hierarchy><node text='custom'/></hierarchy>")
    observation = AndroidWorldHost(
        SimpleNamespace(get_state=lambda **_: state)
    ).observe()

    assert observation.xml is None
    assert observation.extra["ui_graph_source"] == ""
    assert observation.extra["ui_graph_complete"] is False


def test_experiment_agent_adapter_checks_androidworld_before_each_step() -> None:
    calls: list[str] = []
    recording_session = SimpleNamespace(
        env=SimpleNamespace(
            ensure_accessibility_forwarder_ready=lambda: calls.append("ready")
        ),
        start_episode=lambda: calls.append("record"),
    )
    agent = SimpleNamespace(step=lambda goal: calls.append(goal) or "result")

    adapted = _ExperimentAgentAdapter(
        agent,
        recording_session=recording_session,
        goal_hint="Reference action sequence",
    )

    assert adapted.step("Complete task") == "result"
    assert calls == [
        "ready",
        "record",
        "Complete task\n\nReference action sequence",
    ]


def test_experiment_agent_adapter_enforces_step_budget_without_extra_model_call() -> None:
    calls: list[str] = []
    recording_session = SimpleNamespace(
        env=SimpleNamespace(ensure_accessibility_forwarder_ready=lambda: None),
        start_episode=lambda: None,
    )
    result = SimpleNamespace(done=False, data={})
    agent = SimpleNamespace(step=lambda goal: calls.append(goal) or result)
    adapted = _ExperimentAgentAdapter(
        agent,
        recording_session=recording_session,
        max_steps=1,
    )

    first = adapted.step("Complete task")

    assert calls == ["Complete task"]
    assert first.done is True
    assert first.data["experiment_step_budget_reached"] is True


def test_observe_derives_internal_xml_from_official_forest() -> None:
    def node(unique_id, bounds, *, child_ids=(), text=""):
        return SimpleNamespace(
            unique_id=unique_id,
            bounds_in_screen=SimpleNamespace(
                left=bounds[0],
                top=bounds[1],
                right=bounds[2],
                bottom=bounds[3],
            ),
            child_ids=list(child_ids),
            text=text,
            content_description="",
            view_id_resource_name="",
            package_name="com.android.settings",
            class_name="android.widget.TextView",
            is_checkable=False,
            is_checked=False,
            is_clickable=False,
            is_editable=False,
            is_enabled=True,
            is_focusable=False,
            is_focused=False,
            is_long_clickable=False,
            is_password=False,
            is_scrollable=False,
            is_selected=False,
            is_visible_to_user=True,
        )

    forest = SimpleNamespace(
        windows=[
            SimpleNamespace(
                id=7,
                title="Settings",
                tree=SimpleNamespace(
                    nodes=[
                        node(1, (0, 0, 2208, 1840), child_ids=(2,)),
                        node(2, (991, 586, 1171, 657), text="Internet"),
                    ]
                ),
            )
        ]
    )
    env = SimpleNamespace(
        get_state=lambda **_: _official_state(forest=forest),
        device_screen_size=(2208, 1840),
        logical_screen_size=(1080, 2092),
        foreground_activity_name="com.android.settings/.Settings",
    )

    observation = AndroidWorldHost(env).observe()

    assert observation.extra["ui_graph_source"] == "androidworld_state_forest"
    assert observation.extra["ui_graph_complete"] is True
    root = ET.fromstring(observation.xml or "")
    internet = next(
        element for element in root.iter() if element.attrib.get("text") == "Internet"
    )
    assert internet.attrib["bounds"] == "[991,586][1171,657]"


def test_observe_treats_complete_active_modal_window_as_complete_graph() -> None:
    def bounds(left, top, right, bottom):
        return SimpleNamespace(left=left, top=top, right=right, bottom=bottom)

    root = SimpleNamespace(
        unique_id=1,
        bounds_in_screen=bounds(0, 330, 720, 950),
        child_ids=[2],
        package_name="net.gsantner.markor",
        class_name="android.widget.FrameLayout",
        is_visible_to_user=True,
    )
    confirm = SimpleNamespace(
        unique_id=2,
        bounds_in_screen=bounds(64, 806, 224, 918),
        child_ids=[],
        text="FOLDER",
        package_name="net.gsantner.markor",
        class_name="android.widget.Button",
        is_visible_to_user=True,
        is_clickable=True,
    )
    forest = SimpleNamespace(
        windows=[
            SimpleNamespace(
                id=7,
                window_type="TYPE_APPLICATION",
                is_active=True,
                is_focused=True,
                bounds_in_screen=bounds(0, 330, 720, 950),
                tree=SimpleNamespace(nodes=[root, confirm]),
            )
        ]
    )
    ui_element = SimpleNamespace(
        text="FOLDER",
        package_name="net.gsantner.markor",
        class_name="android.widget.Button",
        is_clickable=True,
        bbox_pixels=SimpleNamespace(x_min=64, y_min=806, x_max=224, y_max=918),
    )
    env = SimpleNamespace(
        get_state=lambda **_: _official_state(forest=forest, ui_elements=[ui_element]),
        device_screen_size=(720, 1280),
        logical_screen_size=(720, 1280),
        foreground_activity_name="net.gsantner.markor/.MainActivity",
    )

    observation = AndroidWorldHost(env).observe()

    assert observation.extra["ui_graph_complete"] is True
    assert not observation.extra["ui_graph_source"].endswith("_partial")
    assert "FOLDER" in str(observation.xml)


def test_observe_accepts_serialized_modal_bounds_with_omitted_zero_edges() -> None:
    forest = {
        "windows": [
            {
                "id": 7,
                "window_type": 1,
                "is_active": True,
                "is_focused": True,
                "bounds_in_screen": {"top": 330, "right": 720, "bottom": 950},
                "tree": {
                    "nodes": [
                        {
                            "unique_id": 1,
                            "bounds_in_screen": {
                                "top": 330,
                                "right": 720,
                                "bottom": 950,
                            },
                            "child_ids": [2],
                            "package_name": "net.gsantner.markor",
                            "class_name": "android.widget.FrameLayout",
                            "is_visible_to_user": True,
                        },
                        {
                            "unique_id": 2,
                            "bounds_in_screen": {
                                "left": 64,
                                "top": 806,
                                "right": 224,
                                "bottom": 918,
                            },
                            "child_ids": [],
                            "text": "FOLDER",
                            "package_name": "net.gsantner.markor",
                            "class_name": "android.widget.Button",
                            "is_visible_to_user": True,
                            "is_clickable": True,
                        },
                    ]
                },
            }
        ]
    }
    ui_element = SimpleNamespace(
        text="FOLDER",
        package_name="net.gsantner.markor",
        class_name="android.widget.Button",
        is_clickable=True,
        bbox_pixels=SimpleNamespace(x_min=64, y_min=806, x_max=224, y_max=918),
    )
    env = SimpleNamespace(
        get_state=lambda **_: _official_state(forest=forest, ui_elements=[ui_element]),
        device_screen_size=(720, 1280),
        logical_screen_size=(720, 1280),
        foreground_activity_name="net.gsantner.markor/.MainActivity",
    )

    observation = AndroidWorldHost(env).observe()

    assert observation.extra["ui_graph_complete"] is True
    assert not observation.extra["ui_graph_source"].endswith("_partial")


def test_observe_prefers_semantically_richer_official_ui_elements() -> None:
    sparse_node = SimpleNamespace(
        unique_id=1,
        bounds_in_screen=SimpleNamespace(left=0, top=0, right=1080, bottom=2092),
        child_ids=[],
        is_visible_to_user=True,
        is_clickable=True,
        is_scrollable=False,
    )
    forest = SimpleNamespace(
        windows=[
            SimpleNamespace(
                id=1,
                title="",
                tree=SimpleNamespace(nodes=[sparse_node]),
            )
        ]
    )
    rich_element = SimpleNamespace(
        text="Meeting attendees",
        content_description="Attendee count",
        resource_name="com.example:id/attendees",
        package_name="com.example",
        class_name="android.widget.EditText",
        is_clickable=True,
        is_editable=True,
        is_scrollable=False,
        bbox_pixels=SimpleNamespace(x_min=100, y_min=200, x_max=900, y_max=300),
    )
    env = SimpleNamespace(
        get_state=lambda **_: _official_state(
            forest=forest,
            ui_elements=[rich_element],
        ),
        device_screen_size=(1080, 2092),
        logical_screen_size=(1080, 2092),
        foreground_activity_name="com.example/.MainActivity",
    )

    observation = AndroidWorldHost(env).observe()

    assert observation.extra["ui_graph_source"] == "androidworld_state_ui_elements_partial"
    root = ET.fromstring(observation.xml or "")
    target = next(
        element
        for element in root.iter()
        if element.attrib.get("resource-id") == "com.example:id/attendees"
    )
    assert target.attrib["text"] == "Meeting attendees"
    assert target.attrib["content-desc"] == "Attendee count"
    assert target.attrib["editable"] == "true"


def test_observe_keeps_semantically_richer_official_forest() -> None:
    rich_node = SimpleNamespace(
        unique_id=1,
        bounds_in_screen=SimpleNamespace(left=0, top=0, right=1080, bottom=2092),
        child_ids=[],
        text="Open settings",
        content_description="Settings",
        view_id_resource_name="com.example:id/settings",
        package_name="com.example",
        class_name="android.widget.Button",
        is_visible_to_user=True,
        is_clickable=True,
        is_editable=False,
        is_scrollable=False,
    )
    forest = SimpleNamespace(
        windows=[
            SimpleNamespace(
                id=1,
                title="Settings",
                tree=SimpleNamespace(nodes=[rich_node]),
            )
        ]
    )
    sparse_element = SimpleNamespace(
        package_name="com.example",
        bbox_pixels=SimpleNamespace(x_min=0, y_min=0, x_max=1080, y_max=2092),
    )
    env = SimpleNamespace(
        get_state=lambda **_: _official_state(
            forest=forest,
            ui_elements=[sparse_element],
        ),
        device_screen_size=(1080, 2092),
        logical_screen_size=(1080, 2092),
        foreground_activity_name="com.example/.MainActivity",
    )

    observation = AndroidWorldHost(env).observe()

    assert observation.extra["ui_graph_source"] == "androidworld_state_forest"
    assert "Open settings" in str(observation.xml)


def test_actions_dispatch_only_through_official_androidworld_api(monkeypatch) -> None:
    class JSONAction:
        def __init__(self, action_type=None, x=None, y=None, app_name=None):
            self.action_type = action_type
            self.x = x
            self.y = y
            self.app_name = app_name

    module = SimpleNamespace(JSONAction=JSONAction)
    monkeypatch.setattr(
        "src.integrations.android_world.host.importlib.import_module",
        lambda name: module if name == "android_world.env.json_action" else None,
    )
    actions: list[JSONAction] = []
    env = SimpleNamespace(
        device_screen_size=(720, 1280),
        execute_action=actions.append,
    )
    host = AndroidWorldHost(env, adb_serial="emulator-5564")

    click_result = host.act(Action("click", {"x": 500, "y": 250}))
    open_result = host.act(
        Action("open_app", {"package_name": "com.android.settings"})
    )

    assert click_result.success is True
    assert open_result.success is True
    assert [(action.action_type, action.x, action.y, action.app_name) for action in actions] == [
        ("click", 360.0, 320.0, None),
        ("open_app", None, None, "com.android.settings"),
    ]
