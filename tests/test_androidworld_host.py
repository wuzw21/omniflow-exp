from __future__ import annotations

from types import SimpleNamespace
import xml.etree.ElementTree as ET

from omniflow import Action
from src.integrations.android_world.host import AndroidWorldHost
from src.integrations.android_world.launch import _native_androidworld_a11y_method


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


def test_observe_prefers_complete_androidworld_accessibility_forest() -> None:
    def node(
        unique_id,
        bounds,
        *,
        child_ids=(),
        text="",
        resource_id="",
        class_name="android.widget.LinearLayout",
        clickable=False,
    ):
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
            view_id_resource_name=resource_id,
            package_name="com.android.settings",
            class_name=class_name,
            is_checkable=False,
            is_checked=False,
            is_clickable=clickable,
            is_editable=False,
            is_enabled=True,
            is_focusable=clickable,
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
                        node(1, (0, 0, 2208, 1840), child_ids=(2, 3)),
                        node(2, (0, 0, 802, 1840)),
                        node(3, (802, 0, 2208, 1840), child_ids=(4, 5)),
                        node(
                            4,
                            (802, 544, 2208, 750),
                            child_ids=(6,),
                            clickable=True,
                        ),
                        node(
                            5,
                            (802, 750, 2208, 956),
                            child_ids=(7,),
                            clickable=True,
                        ),
                        node(
                            6,
                            (991, 586, 1171, 657),
                            text="Internet",
                            resource_id="android:id/title",
                            class_name="android.widget.TextView",
                        ),
                        node(
                            7,
                            (991, 792, 1275, 863),
                            text="Calls & SMS",
                            resource_id="android:id/title",
                            class_name="android.widget.TextView",
                        ),
                    ]
                ),
            )
        ]
    )
    state = SimpleNamespace(
        forest=forest,
        ui_elements=[],
        xml="",
        auxiliaries={},
        pixels=None,
        activity_name="com.android.settings/.Settings$NetworkDashboardActivity",
        package_name="com.android.settings",
    )
    env = SimpleNamespace(
        get_state=lambda: state,
        device_screen_size=(2208, 1840),
        logical_screen_size=(1080, 2092),
        foreground_activity_name=(
            "com.android.settings/.Settings$NetworkDashboardActivity"
        ),
    )
    host = AndroidWorldHost(env)
    host._fresh_uiautomator_xml = lambda: (_ for _ in ()).throw(
        AssertionError("complete forest must not trigger another UI dump")
    )

    observation = host.observe(xml=True, screenshot=False, app_info=True)

    assert observation.extra["ui_graph_source"] == "androidworld_accessibility_forest"
    assert observation.extra["ui_graph_complete"] is True
    root = ET.fromstring(observation.xml or "")
    internet = next(
        element for element in root.iter() if element.attrib.get("text") == "Internet"
    )
    calls = next(
        element
        for element in root.iter()
        if element.attrib.get("text") == "Calls & SMS"
    )
    assert internet.attrib["bounds"] == "[991,586][1171,657]"
    assert calls.attrib["bounds"] == "[991,792][1275,863]"


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
