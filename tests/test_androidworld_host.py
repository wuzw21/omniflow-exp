from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from PIL import Image
import pytest

from omniflow import Action
from src.integrations.android_world.host import AndroidWorldHost
from src.integrations.android_world.launch import (
    ANDROID_PERMISSION_DENY_RESOURCE_IDS,
    _androidworld_a11y_forwarder_installed,
    _androidworld_adb_file_transfer_timeout_sec,
    _androidworld_setup_apps_for_suite,
    _androidworld_setup_timeout_sec,
    _bounded_androidworld_adb_file_transfer_timeout,
    _ensure_androidworld_a11y_forwarder,
    _ExperimentAgentAdapter,
    _patch_androidworld_adb_output_sanitizer,
    _patch_androidworld_apk_install_compat,
    _patch_androidworld_chcon_compat,
    _patch_androidworld_directory_clear,
    _patch_androidworld_optional_setup_click,
    _prepare_androidworld_episode_apps,
    _repair_androidworld_chrome_first_run,
    _result_has_official_validator_conclusion,
    _run_androidworld_setup_apps,
    _runtime_execution_trace,
    _wait_for_androidworld_a11y,
)


def test_androidworld_setup_skips_only_already_settled_notification_permission(
    monkeypatch,
) -> None:
    class AndroidToolController:
        def __init__(self, env) -> None:
            self._env = env

        def click_element(self, element_text: str) -> None:
            raise ValueError(f'Target text "{element_text}" not found.')

    tools_module = SimpleNamespace(AndroidToolController=AndroidToolController)
    monkeypatch.setattr(
        "src.integrations.android_world.launch.importlib.import_module",
        lambda name: tools_module
        if name == "android_world.env.tools"
        else pytest.fail(f"unexpected import: {name}"),
    )
    controller = AndroidToolController(
        SimpleNamespace(
            get_ui_elements=lambda: [
                SimpleNamespace(package_name="com.google.android.contacts")
            ]
        )
    )

    patch = _patch_androidworld_optional_setup_click()
    assert patch is not None
    controller_type, original = patch
    try:
        controller.click_element("Don't allow")
        with pytest.raises(ValueError, match="Skip"):
            controller.click_element("Skip")
        controller._env.get_ui_elements = lambda: [
            SimpleNamespace(
                package_name="com.google.android.permissioncontroller"
            )
        ]
        with pytest.raises(ValueError, match="Don't allow"):
            controller.click_element("Don't allow")
    finally:
        controller_type.click_element = original


def test_androidworld_setup_skips_only_absent_markor_final_ok(monkeypatch) -> None:
    class AndroidToolController:
        def __init__(self, env) -> None:
            self._env = env

        def click_element(self, element_text: str) -> None:
            raise ValueError(f'Target text "{element_text}" not found.')

    tools_module = SimpleNamespace(AndroidToolController=AndroidToolController)
    monkeypatch.setattr(
        "src.integrations.android_world.launch.importlib.import_module",
        lambda name: tools_module
        if name == "android_world.env.tools"
        else pytest.fail(f"unexpected import: {name}"),
    )
    controller = AndroidToolController(
        SimpleNamespace(
            foreground_activity_name="net.gsantner.markor/.MainActivity",
            get_ui_elements=lambda: [],
        )
    )

    patch = _patch_androidworld_optional_setup_click()
    assert patch is not None
    controller_type, original = patch
    try:
        controller.click_element("OK")
        with pytest.raises(ValueError, match="CANCEL"):
            controller.click_element("CANCEL")
        controller._env.foreground_activity_name = "com.example/.MainActivity"
        with pytest.raises(ValueError, match="OK"):
            controller.click_element("OK")
    finally:
        controller_type.click_element = original


def test_androidworld_prepares_markor_data_directory(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []
    response = SimpleNamespace(status=1)

    class App:
        app_name = "markor"

        @staticmethod
        def package_name() -> str:
            return "net.gsantner.markor"

    controller = SimpleNamespace(get_ui_elements=lambda: [])
    env = SimpleNamespace(controller=controller)

    def issue_generic_request(arguments, request_env):
        calls.append(("adb", tuple(arguments), request_env))
        return response

    def check_ok(actual_response, message):
        calls.append(("check", actual_response, message))

    setup_module = SimpleNamespace(
        adb_utils=SimpleNamespace(
            issue_generic_request=issue_generic_request,
            check_ok=check_ok,
            launch_app=lambda app_name, actual_controller: calls.append(
                ("launch", app_name, actual_controller)
            ),
            close_app=lambda app_name, actual_controller: calls.append(
                ("close", app_name, actual_controller)
            ),
        )
    )
    real_import_module = __import__("importlib").import_module
    monkeypatch.setattr(
        "src.integrations.android_world.launch.importlib.import_module",
        lambda name: SimpleNamespace(
            MARKOR_DATA="/storage/emulated/0/Documents/Markor"
        )
        if name == "android_world.env.device_constants"
        else SimpleNamespace()
        if name == "android_world.env.actuation"
        else real_import_module(name),
    )
    monkeypatch.setattr("src.integrations.android_world.launch.time.sleep", lambda _: None)

    _prepare_androidworld_episode_apps(
        env,
        setup_module=setup_module,
        setup_apps=(App,),
    )

    assert calls == [
        (
            "adb",
            (
                "shell",
                "mkdir",
                "-p",
                "/storage/emulated/0/Documents/Markor",
            ),
            controller,
        ),
        ("check", response, "Failed to prepare Markor data directory."),
        ("launch", "markor", controller),
        ("close", "markor", controller),
    ]


def test_androidworld_directory_clear_is_idempotent() -> None:
    calls: list[tuple[object, ...]] = []
    response = SimpleNamespace(status=1)

    def original_clear_directory(_path, _controller):
        raise AssertionError("official non-idempotent clear should be replaced")

    file_utils = SimpleNamespace(clear_directory=original_clear_directory)
    adb_utils = SimpleNamespace(
        issue_generic_request=lambda arguments, controller: (
            calls.append(("adb", tuple(arguments), controller)) or response
        ),
        check_ok=lambda actual_response, message: calls.append(
            ("check", actual_response, message)
        ),
    )
    controller = object()

    original = _patch_androidworld_directory_clear(file_utils, adb_utils)
    file_utils.clear_directory("/storage/emulated/0/Documents/Markor", controller)

    assert original is original_clear_directory
    assert calls == [
        (
            "adb",
            (
                "shell",
                "rm",
                "-rf",
                "/storage/emulated/0/Documents/Markor/*",
            ),
            controller,
        ),
        (
            "check",
            response,
            "Failed to clear directory /storage/emulated/0/Documents/Markor.",
        ),
    ]


def test_androidworld_apk_install_retries_without_unsupported_flag() -> None:
    calls: list[tuple[object, ...]] = []

    class AdbUtils:
        def install_apk(self, apk, _env):
            calls.append(("official", apk))
            raise subprocess.CalledProcessError(
                1,
                ["adb", "install"],
                output=(
                    b"Exception occurred while executing 'install':\n"
                    b"java.lang.IllegalArgumentException: Unknown option "
                    b"--bypass-low-target-sdk-block\n"
                ),
            )

        def issue_generic_request(self, args, _env, *, timeout_sec):
            calls.append(("compat", tuple(args), timeout_sec))
            return SimpleNamespace(status=1)

    setup_module = SimpleNamespace(adb_utils=AdbUtils())
    original = _patch_androidworld_apk_install_compat(setup_module)
    assert original is not None
    try:
        response = setup_module.adb_utils.install_apk("/tmp/app.apk", object())
    finally:
        setup_module.adb_utils.install_apk = original

    assert response.status == 1
    assert calls == [
        ("official", "/tmp/app.apk"),
        ("compat", ("install", "/tmp/app.apk"), 30.0),
    ]


def test_androidworld_adb_output_removes_grpc_fork_diagnostics() -> None:
    class Controller:
        def execute_adb_call(self, _request):
            return SimpleNamespace(
                generic=SimpleNamespace(
                    output=(
                        b"I0818 08:14:27.756116 11386938 "
                        b"ev_poll_posix.cc:593] FD from fork parent still in poll list\n"
                        b"I0818 08:14:27.756170 11386938 fork_posix.cc:71] "
                        b"Other threads are currently calling into gRPC\n"
                        b"1697412907\n"
                    )
                )
            )

    controller = Controller()
    controller_type, original = _patch_androidworld_adb_output_sanitizer(
        controller
    )
    try:
        response = controller.execute_adb_call(object())
    finally:
        controller_type.execute_adb_call = original

    assert response.generic.output == b"1697412907\n"
    assert int(response.generic.output.strip()) == 1697412907


def test_androidworld_chcon_compat_only_normalizes_transport_endpoint_failure() -> None:
    class Response:
        def __init__(self, status: int, output: bytes) -> None:
            self.status = status
            self.generic = SimpleNamespace(output=output)

        def CopyFrom(self, other) -> None:
            self.status = other.status
            self.generic = SimpleNamespace(output=other.generic.output)

    responses = [
        Response(
            2,
            b"chcon: Operation not supported on transport endpoint",
        ),
        Response(2, b"chcon: permission denied"),
    ]

    class AdbUtils:
        def issue_generic_request(self, args, _env, *, timeout_sec=None):
            return responses.pop(0)

    setup_module = SimpleNamespace(adb_utils=AdbUtils())
    original = _patch_androidworld_chcon_compat(setup_module)
    assert original is not None
    try:
        normalized = setup_module.adb_utils.issue_generic_request(
            ["shell", "chcon", "u:object_r:media_rw_data_file:s0", "/map.obf"],
            object(),
        )
        unchanged = setup_module.adb_utils.issue_generic_request(
            ["shell", "chcon", "u:object_r:media_rw_data_file:s0", "/map.obf"],
            object(),
        )
    finally:
        setup_module.adb_utils.issue_generic_request = original

    assert normalized.status == 1
    assert unchanged.status == 2


def test_androidworld_chcon_compat_normalizes_controller_exception() -> None:
    class Response:
        class Status:
            OK = 1

        def __init__(self) -> None:
            self.status = 1

    class AdbUtils:
        def issue_generic_request(self, args, _env, *, timeout_sec=None):
            raise RuntimeError(
                "Error executing adb command: shell chcon ...: "
                "Operation not supported on transport endpoint"
            )

    setup_module = SimpleNamespace(
        adb_utils=AdbUtils(),
        adb_pb2=SimpleNamespace(AdbResponse=Response),
    )
    original = _patch_androidworld_chcon_compat(setup_module)
    assert original is not None
    try:
        response = setup_module.adb_utils.issue_generic_request(
            ["shell", "chcon", "context", "/map.obf"],
            object(),
        )
    finally:
        setup_module.adb_utils.issue_generic_request = original

    assert response.status == 1


def test_androidworld_file_transfer_timeout_bounds_unset_and_zero(
    monkeypatch,
) -> None:
    assert _bounded_androidworld_adb_file_transfer_timeout(
        None,
        default_timeout_sec=300.0,
    ) == 300.0
    assert _bounded_androidworld_adb_file_transfer_timeout(
        0,
        default_timeout_sec=300.0,
    ) == 300.0
    assert _bounded_androidworld_adb_file_transfer_timeout(
        45,
        default_timeout_sec=300.0,
    ) == 45.0

    monkeypatch.setenv("OMNIFLOW_ANDROIDWORLD_ADB_FILE_TRANSFER_TIMEOUT_SEC", "0")
    with pytest.raises(RuntimeError, match="must be positive"):
        _androidworld_adb_file_transfer_timeout_sec()


def test_androidworld_setup_timeout_is_positive(monkeypatch) -> None:
    monkeypatch.setenv("OMNIFLOW_ANDROIDWORLD_SETUP_TIMEOUT_SEC", "0")
    with pytest.raises(RuntimeError, match="must be positive"):
        _androidworld_setup_timeout_sec()


def test_androidworld_setup_has_hard_deadline(monkeypatch) -> None:
    installed_handler = None
    timer_calls: list[tuple[float, float]] = []

    def fake_signal(_signal_number, handler):
        nonlocal installed_handler
        previous = installed_handler
        installed_handler = handler
        return previous

    def fake_setitimer(_timer, delay, interval=0.0):
        timer_calls.append((float(delay), float(interval)))
        return (0.0, 0.0)

    def setup_apps(_setup_env, *, app_list) -> None:
        assert app_list == ("osmand",)
        assert callable(installed_handler)
        installed_handler(0, None)

    monkeypatch.setenv("OMNIFLOW_ANDROIDWORLD_SETUP_TIMEOUT_SEC", "12")
    monkeypatch.setattr(
        "src.integrations.android_world.launch.signal.getsignal",
        lambda _signal_number: "previous-handler",
    )
    monkeypatch.setattr(
        "src.integrations.android_world.launch.signal.signal", fake_signal
    )
    monkeypatch.setattr(
        "src.integrations.android_world.launch.signal.setitimer", fake_setitimer
    )

    with pytest.raises(
        TimeoutError, match="official app setup exceeded 12 seconds"
    ):
        _run_androidworld_setup_apps(
            SimpleNamespace(controller=SimpleNamespace()),
            setup_module=SimpleNamespace(setup_apps=setup_apps),
            setup_apps=("osmand",),
        )

    assert timer_calls == [(0.0, 0.0), (12.0, 0.0), (0.0, 0.0)]
    assert installed_handler == "previous-handler"


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


def test_androidworld_setup_normalizes_permission_prompt_typography() -> None:
    permission_prompt = SimpleNamespace(
        text="Don’t allow",
        content_description=None,
        package_name="com.google.android.permissioncontroller",
        resource_id="com.android.permissioncontroller:id/permission_deny_button",
    )
    controller = SimpleNamespace(get_ui_elements=lambda: [permission_prompt])
    env = SimpleNamespace(controller=controller)
    observed: list[tuple[str, str]] = []

    def setup_apps(setup_env, *, app_list) -> None:
        element = setup_env.controller.get_ui_elements()[0]
        observed.append((element.text, element.resource_id))
        assert app_list == ("contacts",)

    _run_androidworld_setup_apps(
        env,
        setup_module=SimpleNamespace(setup_apps=setup_apps),
        setup_apps=("contacts",),
    )

    assert observed == [
        (
            "Don't allow",
            "com.android.permissioncontroller:id/permission_deny_button",
        )
    ]
    assert controller.get_ui_elements()[0].text == "Don’t allow"


def test_androidworld_chrome_setup_falls_back_to_semantic_labels(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    class ChromeApp:
        app_name = "chrome"

    class Controller:
        def click_resource_id(self, resource_ids, **_kwargs):
            calls.append(("resource", resource_ids))
            raise ValueError("resource id unavailable")

        def click_element(self, label):
            calls.append(("text", label))

    class AdbUtils:
        def launch_app(self, app_name, _controller):
            calls.append(("launch", app_name))

        def close_app(self, app_name, _controller):
            calls.append(("close", app_name))

    class Tools:
        @staticmethod
        def AndroidToolController(**_kwargs):
            return Controller()

    real_import_module = __import__("importlib").import_module
    monkeypatch.setattr(
        "src.integrations.android_world.launch.importlib.import_module",
        lambda name: Tools()
        if name == "android_world.env.tools"
        else real_import_module(name),
    )
    monkeypatch.setattr(
        "src.integrations.android_world.launch.time.sleep", lambda _seconds: None
    )

    _repair_androidworld_chrome_first_run(
        SimpleNamespace(controller=object()),
        setup_module=SimpleNamespace(adb_utils=AdbUtils()),
        setup_apps=(ChromeApp,),
    )

    assert calls[0] == ("launch", "chrome")
    assert ("text", "Accept & continue") in calls
    assert ("text", "No thanks") in calls
    assert calls[-1] == ("close", "chrome")


def test_androidworld_setup_clears_late_permission_dialog_before_resnapshot(
    monkeypatch,
) -> None:
    calls: list[object] = []

    class Controller:
        def __init__(self) -> None:
            self.app_open = False
            self.permission_denied = False

        def get_ui_elements(self):
            if not self.app_open:
                return []
            if not self.permission_denied:
                return [
                    SimpleNamespace(
                        text="Allow Contacts to send you notifications?",
                        content_description=None,
                        package_name="com.google.android.permissioncontroller",
                        resource_id=None,
                        resource_name="com.android.permissioncontroller:id/permission_message",
                    ),
                    SimpleNamespace(
                        text="Don’t allow",
                        content_description=None,
                        package_name="com.google.android.permissioncontroller",
                        resource_id=None,
                        resource_name="com.android.permissioncontroller:id/permission_deny_button",
                    ),
                ]
            return [
                SimpleNamespace(
                    text="Contacts",
                    content_description=None,
                    package_name="com.google.android.contacts",
                    resource_id=None,
                    resource_name="com.google.android.contacts:id/toolbar",
                )
            ]

    controller = Controller()
    env = SimpleNamespace(controller=controller)

    class ContactsApp:
        app_name = "contacts"

        @classmethod
        def package_name(cls) -> str:
            return "com.google.android.contacts"

        @classmethod
        def setup(cls, setup_env) -> None:
            calls.append("ContactsApp.setup")
            calls.append(("click_element", "Skip"))
            calls.append(("click_element", "Don't allow"))

    def save_snapshot(app_name, raw_controller) -> None:
        source = getattr(raw_controller, "_controller", raw_controller)
        calls.append(
            (
                "save_snapshot",
                app_name,
                source.permission_denied,
            )
        )

    def setup_app(app, setup_env) -> None:
        calls.append("setup_app")
        app.setup(setup_env)
        save_snapshot(app.app_name, setup_env.controller)

    def setup_apps(setup_env, *, app_list) -> None:
        calls.append("setup_apps")
        for app in app_list:
            setup_app(app, setup_env)

    def launch_app(app_name, raw_controller) -> None:
        calls.append(("launch_app", app_name))
        getattr(raw_controller, "_controller", raw_controller).app_open = True

    def close_app(app_name, raw_controller) -> None:
        calls.append(("close_app", app_name))
        getattr(raw_controller, "_controller", raw_controller).app_open = False

    def click_resource_id(resource_ids, raw_controller, timeout_sec=10.0) -> None:
        calls.append(("click_resource_id", tuple(resource_ids), timeout_sec))
        getattr(raw_controller, "_controller", raw_controller).permission_denied = True

    setup_module = SimpleNamespace(
        setup_apps=setup_apps,
        adb_utils=SimpleNamespace(launch_app=launch_app, close_app=close_app),
        app_snapshot=SimpleNamespace(save_snapshot=save_snapshot),
    )
    real_import_module = __import__("importlib").import_module
    monkeypatch.setattr(
        "src.integrations.android_world.launch.importlib.import_module",
        lambda name: SimpleNamespace(
            find_and_click_element_by_resource_id=click_resource_id
        )
        if name == "android_world.env.actuation"
        else real_import_module(name),
    )
    monkeypatch.setattr(
        "src.integrations.android_world.launch.time.sleep", lambda _: None
    )

    _run_androidworld_setup_apps(
        env,
        setup_module=setup_module,
        setup_apps=(ContactsApp,),
    )

    assert controller.permission_denied is True
    assert calls[:4] == [
        "setup_apps",
        "setup_app",
        "ContactsApp.setup",
        ("click_element", "Skip"),
    ]
    assert (
        "click_resource_id",
        (
            "com.android.permissioncontroller:id/permission_deny_button",
            "com.android.permissioncontroller:id/permission_deny_and_dont_ask_again_button",
        ),
        10.0,
    ) in calls
    assert (
        "save_snapshot",
        "contacts",
        True,
    ) in calls


def test_androidworld_official_setup_entry_clears_late_permission_dialog(
    monkeypatch,
) -> None:
    android_world_root_value = os.environ.get("OMNIFLOW_ANDROID_WORLD_ROOT", "")
    if not android_world_root_value:
        pytest.skip("requires the pinned AndroidWorld checkout")
    android_world_root = Path(android_world_root_value).expanduser()
    if not android_world_root.is_dir():
        pytest.skip("requires the pinned AndroidWorld checkout")
    monkeypatch.syspath_prepend(str(android_world_root))

    from android_world.env import actuation, tools
    from android_world.env.setup_device import apps
    from android_world.env.setup_device import setup as aw_setup

    class Controller:
        def __init__(self) -> None:
            self.app_open = False
            self.launch_count = 0
            self.permission_denied = False
            self.screen = "closed"

        def get_ui_elements(self):
            if self.screen == "skip":
                return [
                    SimpleNamespace(
                        text="Skip",
                        content_description=None,
                        package_name="com.google.android.contacts",
                        resource_id=None,
                        resource_name="com.google.android.contacts:id/skip",
                    )
                ]
            if self.screen == "permission":
                return [
                    SimpleNamespace(
                        text="Allow Contacts to send you notifications?",
                        content_description=None,
                        package_name="com.google.android.permissioncontroller",
                        resource_id=None,
                        resource_name="com.android.permissioncontroller:id/permission_message",
                    ),
                    SimpleNamespace(
                        text="Don’t allow",
                        content_description=None,
                        package_name="com.google.android.permissioncontroller",
                        resource_id=None,
                        resource_name="com.android.permissioncontroller:id/permission_deny_button",
                    ),
                ]
            if self.screen == "contacts":
                return [
                    SimpleNamespace(
                        text="Contacts",
                        content_description=None,
                        package_name="com.google.android.contacts",
                        resource_id=None,
                        resource_name="com.google.android.contacts:id/toolbar",
                    )
                ]
            return []

    controller = Controller()
    env = SimpleNamespace(controller=controller)
    click_targets: list[str] = []
    snapshots: list[bool] = []
    original_click_element = tools.AndroidToolController.click_element

    def launch_app(_app_name, raw_controller) -> None:
        source = getattr(raw_controller, "_controller", raw_controller)
        source.app_open = True
        source.launch_count += 1
        source.screen = "skip" if source.launch_count == 1 else "permission"

    def close_app(_app_name, raw_controller) -> None:
        source = getattr(raw_controller, "_controller", raw_controller)
        source.app_open = False
        source.screen = "closed"

    def execute_action(action, screen_elements, _screen_size, raw_controller) -> None:
        source = getattr(raw_controller, "_controller", raw_controller)
        selected = screen_elements[action.index]
        if selected.text == "Skip":
            source.screen = "contacts"
        elif selected.resource_name in ANDROID_PERMISSION_DENY_RESOURCE_IDS:
            source.permission_denied = True
            source.screen = "contacts"

    def click_element(tool_controller, target: str) -> None:
        click_targets.append(target)
        original_click_element(tool_controller, target)

    clock = iter((0.0, 0.0, 0.0, 0.0, 11.0))
    monkeypatch.setattr(aw_setup, "maybe_install_app", lambda *_args: None)
    monkeypatch.setattr(aw_setup.adb_utils, "press_home_button", lambda *_args: None)
    monkeypatch.setattr(aw_setup.adb_utils, "set_root_if_needed", lambda *_args: None)
    monkeypatch.setattr(aw_setup.adb_utils, "clear_app_data", lambda *_args: None)
    monkeypatch.setattr(aw_setup.adb_utils, "launch_app", launch_app)
    monkeypatch.setattr(aw_setup.adb_utils, "close_app", close_app)
    monkeypatch.setattr(
        aw_setup.app_snapshot,
        "save_snapshot",
        lambda _app_name, _controller: snapshots.append(controller.permission_denied),
    )
    monkeypatch.setattr(tools.AndroidToolController, "click_element", click_element)
    monkeypatch.setattr(actuation, "execute_adb_action", execute_action)
    monkeypatch.setattr(actuation.time, "time", lambda: next(clock, 11.0))
    monkeypatch.setattr(apps.time, "sleep", lambda _: None)
    monkeypatch.setattr("src.integrations.android_world.launch.time.sleep", lambda _: None)

    _run_androidworld_setup_apps(
        env,
        setup_module=aw_setup,
        setup_apps=(apps.ContactsApp,),
    )

    assert click_targets == ["Skip", "Don't allow"]
    assert controller.permission_denied is True
    assert snapshots == [False, True]
    assert controller.app_open is False


def test_androidworld_episode_reset_clears_permission_dialog_before_first_observe(
    monkeypatch,
) -> None:
    calls: list[object] = []

    class Controller:
        def __init__(self) -> None:
            self.screen = "closed"
            self.permission_denied = False

        def get_ui_elements(self):
            if self.screen == "permission":
                return [
                    SimpleNamespace(
                        package_name="com.google.android.permissioncontroller",
                        resource_name=(
                            "com.android.permissioncontroller:id/permission_deny_button"
                        ),
                    )
                ]
            if self.screen == "app":
                return [
                    SimpleNamespace(
                        package_name="com.example.target",
                        resource_name="com.example.target:id/root",
                    )
                ]
            return []

    controller = Controller()
    env = SimpleNamespace(controller=controller)

    class App:
        app_name = "target"

        @classmethod
        def package_name(cls) -> str:
            return "com.example.target"

    def launch_app(app_name, _controller) -> None:
        calls.append(("launch", app_name))
        controller.screen = "permission" if not controller.permission_denied else "app"

    def close_app(app_name, _controller) -> None:
        calls.append(("close", app_name))
        controller.screen = "closed"

    def click_resource_id(resource_ids, _controller, timeout_sec=10.0) -> None:
        calls.append(("deny", tuple(resource_ids), timeout_sec))
        controller.permission_denied = True
        controller.screen = "app"

    setup_module = SimpleNamespace(
        adb_utils=SimpleNamespace(launch_app=launch_app, close_app=close_app)
    )
    real_import_module = __import__("importlib").import_module
    monkeypatch.setattr(
        "src.integrations.android_world.launch.importlib.import_module",
        lambda name: SimpleNamespace(
            find_and_click_element_by_resource_id=click_resource_id
        )
        if name == "android_world.env.actuation"
        else real_import_module(name),
    )
    monkeypatch.setattr("src.integrations.android_world.launch.time.sleep", lambda _: None)

    class Agent:
        def reset(self, go_home: bool = False) -> None:
            calls.append(("reset", go_home))

    adapter = _ExperimentAgentAdapter(
        Agent(),
        recording_session=SimpleNamespace(),
        prepare_after_reset=lambda: _prepare_androidworld_episode_apps(
            env,
            setup_module=setup_module,
            setup_apps=(App,),
        ),
    )

    adapter.reset(go_home=False)
    launch_app("target", controller)
    observed_packages = {
        element.package_name for element in controller.get_ui_elements()
    }

    assert controller.permission_denied is True
    assert observed_packages == {"com.example.target"}
    assert calls == [
        ("reset", False),
        ("launch", "target"),
        (
            "deny",
            ANDROID_PERMISSION_DENY_RESOURCE_IDS,
            10.0,
        ),
        ("close", "target"),
        ("launch", "target"),
    ]


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


def test_androidworld_a11y_readiness_restarts_bound_stale_forwarder(
    monkeypatch,
) -> None:
    calls: list[str] = []

    class Env:
        controller = SimpleNamespace(
            restart_accessibility_forwarder=lambda: calls.append("restart")
        )

        def get_state(self, *, wait_to_stabilize: bool = False):
            assert wait_to_stabilize is False
            calls.append("state")
            if "restart" not in calls:
                raise RuntimeError("Could not get a11y tree after 5 attempts.")
            return SimpleNamespace(forest=object())

    monkeypatch.setattr("src.integrations.android_world.launch.time.sleep", lambda _: None)

    _wait_for_androidworld_a11y(Env())

    assert calls == ["state", "restart", "state"]


def test_androidworld_a11y_readiness_retries_when_forwarder_restart_is_unbound(
    monkeypatch,
) -> None:
    calls: list[str] = []

    class Env:
        class Controller:
            @staticmethod
            def restart_accessibility_forwarder() -> None:
                calls.append("restart")
                raise RuntimeError("Accessibility forwarder did not become bound.")

        controller = Controller()

        def get_state(self, *, wait_to_stabilize: bool = False):
            assert wait_to_stabilize is False
            calls.append("state")
            if calls.count("state") == 1:
                raise RuntimeError("Could not get a11y tree after 5 attempts.")
            return SimpleNamespace(forest=object())

    monkeypatch.setattr("src.integrations.android_world.launch.time.sleep", lambda _: None)

    _wait_for_androidworld_a11y(Env())

    assert calls == ["state", "restart", "state"]


def test_androidworld_a11y_readiness_rejects_empty_forests(monkeypatch) -> None:
    forests = iter(({}, [], {"window": object()}))
    monkeypatch.setattr("src.integrations.android_world.launch.time.sleep", lambda _: None)

    _wait_for_androidworld_a11y(
        SimpleNamespace(
            get_state=lambda **_kwargs: SimpleNamespace(forest=next(forests))
        )
    )


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
    assert _result_has_official_validator_conclusion(
        {
            "is_successful": False,
            "exception_info": "TypeError: baseline action parser failed",
        }
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


def test_experiment_agent_adapter_preserves_parameterless_reset_contract() -> None:
    calls: list[str] = []

    class Agent:
        def reset(self) -> None:
            calls.append("reset")

    adapted = _ExperimentAgentAdapter(
        Agent(),
        recording_session=SimpleNamespace(),
    )

    adapted.reset(go_home=True)

    assert calls == ["reset"]


def test_experiment_agent_adapter_passes_go_home_when_supported() -> None:
    calls: list[bool] = []

    class Agent:
        def reset(self, go_home: bool = False) -> None:
            calls.append(go_home)

    adapted = _ExperimentAgentAdapter(
        Agent(),
        recording_session=SimpleNamespace(),
    )

    adapted.reset(go_home=True)

    assert calls == [True]


def test_experiment_agent_adapter_does_not_mask_reset_type_error() -> None:
    class Agent:
        def reset(self) -> None:
            raise TypeError("reset implementation failed")

    adapted = _ExperimentAgentAdapter(
        Agent(),
        recording_session=SimpleNamespace(),
    )

    with pytest.raises(TypeError, match="reset implementation failed"):
        adapted.reset()


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
