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
    _native_androidworld_a11y_method,
    _result_has_official_validator_conclusion,
)


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

        def get_state(self):
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


def test_observe_does_not_replace_failed_official_state() -> None:
    class Env:
        controller = SimpleNamespace(
            get_ui_elements=lambda: [_ui_element("fallback")],
            get_screenshot=lambda: Image.new("RGB", (1, 1)),
        )

        def get_state(self):
            raise RuntimeError("official state unavailable")

    with pytest.raises(RuntimeError, match="official state unavailable"):
        AndroidWorldHost(Env()).observe()


def test_observe_requires_all_official_state_fields() -> None:
    incomplete = SimpleNamespace(pixels=None, forest=None, ui_elements=[])

    with pytest.raises(
        ValueError,
        match="androidworld_state_fields_missing:auxiliaries",
    ):
        AndroidWorldHost(SimpleNamespace(get_state=lambda: incomplete)).observe()


def test_observe_ignores_non_official_xml_field() -> None:
    state = _official_state(xml="<hierarchy><node text='custom'/></hierarchy>")
    observation = AndroidWorldHost(
        SimpleNamespace(get_state=lambda: state)
    ).observe()

    assert observation.xml is None
    assert observation.extra["ui_graph_source"] == ""
    assert observation.extra["ui_graph_complete"] is False


def test_androidworld_native_observation_uses_accessibility_forest() -> None:
    forwarder = object()
    controller_module = SimpleNamespace(
        A11yMethod=SimpleNamespace(
            A11Y_FORWARDER_APP=forwarder,
            UIAUTOMATOR=object(),
        )
    )

    assert _native_androidworld_a11y_method(controller_module) is forwarder


def test_androidworld_host_cannot_be_switched_to_oob(monkeypatch) -> None:
    monkeypatch.setenv("OMNIFLOW_OBSERVE_BACKEND", "oob")
    monkeypatch.setenv("OMNIFLOW_ACT_BACKEND", "oob")
    monkeypatch.setenv("OMNIFLOW_OOB_DEVICE_URL", "http://127.0.0.1:8910")

    host = AndroidWorldHost(SimpleNamespace())

    assert host.observe_backend == "androidworld"
    assert host.act_backend == "androidworld"


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
        get_state=lambda: _official_state(forest=forest),
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
        get_state=lambda: _official_state(
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
        get_state=lambda: _official_state(
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


def test_open_app_waits_on_the_official_state() -> None:
    state = _official_state(ui_elements=[_ui_element()])
    env = SimpleNamespace(
        get_state=lambda: state,
        device_screen_size=(4, 3),
        logical_screen_size=(4, 3),
        foreground_activity_name="com.android.settings/.Settings",
    )
    host = AndroidWorldHost(
        env,
        adb_serial="emulator-5564",
        open_app_ready_timeout_seconds=0.01,
    )
    host._adb = lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stderr="")

    result = host.act(Action("open_app", {"package_name": "com.android.settings"}))

    assert result.success is True
