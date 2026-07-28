from __future__ import annotations

from types import SimpleNamespace
import xml.etree.ElementTree as ET

from omniflow import Action
from src.integrations.android_world.host import AndroidWorldHost
from src.integrations.android_world.launch import _native_androidworld_a11y_method


def test_androidworld_native_observation_uses_uiautomator() -> None:
    uiautomator = object()
    controller_module = SimpleNamespace(
        A11yMethod=SimpleNamespace(
            A11Y_FORWARDER_APP=object(),
            UIAUTOMATOR=uiautomator,
        )
    )

    assert _native_androidworld_a11y_method(controller_module) is uiautomator


def test_observe_keeps_complete_uiautomator_hierarchy() -> None:
    uiautomator_xml = """\
<hierarchy>
  <node package="com.android.settings" bounds="[0,0][1080,2400]">
    <node class="android.widget.TextView" text="Bluetooth"
          resource-id="android:id/title" package="com.android.settings"
          bounds="[24,200][1000,280]" />
  </node>
</hierarchy>
"""
    state = SimpleNamespace(
        ui_elements=[],
        xml="",
        auxiliaries={},
        pixels=None,
        activity_name="com.android.settings/.Settings",
        package_name="com.android.settings",
    )
    env = SimpleNamespace(
        get_state=lambda: state,
        device_screen_size=(1080, 2400),
        logical_screen_size=(1080, 2400),
        foreground_activity_name="com.android.settings/.Settings",
    )
    host = AndroidWorldHost(env)
    host._fresh_uiautomator_xml = lambda: uiautomator_xml

    observation = host.observe(xml=True, screenshot=False, app_info=True)

    assert observation.xml == uiautomator_xml
    assert observation.extra["ui_graph_source"] == "uiautomator"


def test_observe_places_partial_window_xml_on_full_device_canvas() -> None:
    partial_xml = """\
<hierarchy>
  <node package="com.android.settings" bounds="[802,0][2208,1840]">
    <node class="android.widget.TextView" text="Bluetooth"
          package="com.android.settings" bounds="[850,500][2160,620]" />
  </node>
</hierarchy>
"""
    state = SimpleNamespace(
        ui_elements=[],
        xml="",
        auxiliaries={},
        pixels=None,
        activity_name="com.android.settings/.SubSettings",
        package_name="com.android.settings",
    )
    env = SimpleNamespace(
        get_state=lambda: state,
        device_screen_size=(2208, 1840),
        logical_screen_size=(1080, 2092),
        foreground_activity_name="com.android.settings/.SubSettings",
    )
    host = AndroidWorldHost(env)
    host._fresh_uiautomator_xml = lambda: partial_xml

    observation = host.observe(xml=True, screenshot=False, app_info=True)

    root = ET.fromstring(observation.xml or "")
    assert root.attrib["bounds"] == "[0,0][2208,1840]"
    assert root.attrib["width"] == "2208"
    assert root.attrib["height"] == "1840"
    assert observation.extra["ui_graph_source"] == "uiautomator_partial"


def test_open_app_accepts_matching_package_with_incomplete_fold_xml() -> None:
    partial_xml = """\
<hierarchy>
  <node package="com.android.settings" bounds="[802,0][2208,1840]" />
</hierarchy>
"""
    state = SimpleNamespace(
        ui_elements=[],
        xml="",
        auxiliaries={},
        pixels=None,
        activity_name="com.android.settings/.Settings",
        package_name="com.android.settings",
    )
    env = SimpleNamespace(
        get_state=lambda: state,
        device_screen_size=(2208, 1840),
        logical_screen_size=(1080, 2092),
        foreground_activity_name="com.android.settings/.Settings",
    )
    host = AndroidWorldHost(
        env,
        adb_serial="emulator-5564",
        open_app_ready_timeout_seconds=0.01,
    )
    host._fresh_uiautomator_xml = lambda: partial_xml
    host._adb = lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stderr="")

    result = host.act(Action("open_app", {"package_name": "com.android.settings"}))

    assert result.success is True
