from __future__ import annotations

from src.integrations import mobilegpt_oob_client as mobilegpt_oob
from src.integrations.mobilegpt_oob_client import (
    _action_with_bounds,
    _dismiss_oob_permission_dialog,
    _is_oob_environment_failure,
    _launch_selected_package,
    _oob_action,
    _prelaunch_target_package,
    _require_oob_backend,
    _server_reported_empty_response,
)


class _FakeOob:
    def __init__(self) -> None:
        self.actions: list[dict] = []

    def act(self, action: dict) -> dict:
        self.actions.append(action)
        return {"success": True}


class _PermissionOob(_FakeOob):
    pass


class _LaunchOob(_FakeOob):
    def observe(self, *, wait_to_stabilize: bool = False) -> dict:
        del wait_to_stabilize
        package = "com.android.camera2" if len(self.actions) >= 2 else "android"
        return {"package_name": package, "xml": "<hierarchy />"}


def test_official_index_action_is_mapped_to_current_oob_bounds() -> None:
    xml = (
        '<hierarchy><node index="0" bounds="[0,0][1280,800]">'
        '<node index="4" bounds="[100,200][300,400]" />'
        "</node></hierarchy>"
    )
    action = {"name": "click", "parameters": {"index": "4"}}

    mapped = _action_with_bounds(action, xml)

    assert mapped["parameters"]["oob_bounds"] == "[100,200][300,400]"


def test_oob_action_executes_official_click_and_input_schema() -> None:
    xml = '<hierarchy><node index="2" bounds="[100,200][300,400]" /></hierarchy>'
    oob = _FakeOob()

    _oob_action(
        oob,
        {"name": "click", "parameters": {"index": "2"}},
        {"width": 1000, "height": 1000},
        xml,
    )
    _oob_action(
        oob,
        {
            "name": "input",
            "parameters": {"index": "2", "input_text": "Sara Ahmed"},
        },
        {"width": 1000, "height": 1000},
        xml,
    )

    assert oob.actions == [
        {"tool": "click", "args": {"x": 200, "y": 300}},
        {"tool": "click", "args": {"x": 200, "y": 300}},
        {
            "tool": "input_text",
            "args": {"text": "Sara Ahmed", "clear_text": True},
        },
    ]


def test_permission_controller_is_dismissed_through_oob() -> None:
    oob = _PermissionOob()
    snapshot = {
        "package_name": "com.google.android.permissioncontroller",
        "display": {"width": 1000, "height": 1000},
        "xml": (
            '<hierarchy><node resource-id="com.android.permissioncontroller:id/'
            'permission_deny_and_dont_ask_again_button" '
            'bounds="[100,200][300,400]" /></hierarchy>'
        ),
    }

    assert _dismiss_oob_permission_dialog(oob, snapshot) is True
    assert oob.actions == [
        {"tool": "click", "args": {"x": 200, "y": 300}},
    ]


def test_target_launch_is_retried_once_when_fresh_boot_overwrites_it(
    monkeypatch,
) -> None:
    oob = _LaunchOob()
    ticks = iter((0.0, 0.1, 0.6, 0.7, 1.1, 1.2))
    monkeypatch.setattr(
        mobilegpt_oob,
        "_installed_packages",
        lambda _adb, _serial: ["com.android.camera2"],
    )
    monkeypatch.setattr(mobilegpt_oob.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(mobilegpt_oob.time, "sleep", lambda _seconds: None)

    selected = _launch_selected_package(
        oob,
        "adb",
        "emulator-45562",
        "com.android.camera2",
        timeout_sec=1.0,
    )

    assert selected == "com.android.camera2"
    assert oob.actions == [
        {"tool": "open_app", "args": {"package_name": "com.android.camera2"}},
        {"tool": "open_app", "args": {"package_name": "com.android.camera2"}},
    ]


def test_empty_official_server_response_is_detected() -> None:
    assert _server_reported_empty_response("Response:\n\n") is True
    assert _server_reported_empty_response(
        'Response:\n{"name":"click"}\n'
    ) is False


def test_server_planner_failure_is_not_misclassified_as_oob_environment() -> None:
    assert _is_oob_environment_failure("mobilegpt_server_no_action") is False
    assert _is_oob_environment_failure("mobilegpt_server_handler_failed") is False
    assert _is_oob_environment_failure("mobilegpt_target_app_not_ready:contacts") is True


def test_mobilegpt_client_rejects_every_non_oob_physical_backend(monkeypatch) -> None:
    monkeypatch.setenv("OMNIFLOW_ANDROIDWORLD_CONTROL_BACKEND", "androidworld")

    try:
        _require_oob_backend()
    except RuntimeError as error:
        assert str(error) == "mobilegpt_oob_backend_required:androidworld"
    else:
        raise AssertionError("non-OOB MobileGPT backend was accepted")


def test_mobilegpt_client_accepts_only_canonical_oob_backend(monkeypatch) -> None:
    monkeypatch.setenv("OMNIFLOW_ANDROIDWORLD_CONTROL_BACKEND", "oob")

    _require_oob_backend()


def test_target_app_is_launched_through_oob_before_planner_handshake(
    monkeypatch,
) -> None:
    oob = _FakeOob()
    calls: list[tuple[object, str, str, str, float]] = []
    monkeypatch.setenv("MOBILEGPT_TARGET_PACKAGE", "com.android.camera2")
    monkeypatch.setattr(
        mobilegpt_oob,
        "_launch_selected_package",
        lambda client, adb, serial, package, *, timeout_sec: calls.append(
            (client, adb, serial, package, timeout_sec)
        )
        or package,
    )

    selected = _prelaunch_target_package(
        oob,
        "adb",
        "emulator-45562",
        timeout_sec=20.0,
    )

    assert selected == "com.android.camera2"
    assert calls == [
        (oob, "adb", "emulator-45562", "com.android.camera2", 20.0),
    ]
