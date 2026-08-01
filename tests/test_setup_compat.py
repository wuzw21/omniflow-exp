from __future__ import annotations

from enum import Enum
from types import SimpleNamespace

import pytest

from src.integrations.android_world import launch
from src.integrations.android_world.setup_compat import (
    patch_androidworld_legacy_apk_install,
    patch_androidworld_osmand_storage_setup,
    patch_androidworld_setup_click_retry,
    patch_androidworld_setup_fail_closed,
    patch_androidworld_special_storage_setup,
    restore_task_app_snapshots_after_initialize,
)


def _setup_element(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        package_name="com.android.chrome",
        text=text,
        content_description="",
    )


def test_androidworld_setup_clicks_visible_alias_before_legacy_label() -> None:
    click_calls: list[str] = []

    class Controller:
        def __init__(self) -> None:
            self._env = SimpleNamespace(
                get_ui_elements=lambda: [_setup_element("Use without an account")]
            )

        def click_element(self, element_text: str) -> None:
            click_calls.append(element_text)
            if element_text != "Use without an account":
                raise ValueError(f'Target text "{element_text}" not found.')

    tools_module = SimpleNamespace(AndroidToolController=Controller)
    patch_androidworld_setup_click_retry(
        tools_module,
        attempts=1,
        delay_seconds=0,
    )

    Controller().click_element("Accept & continue")

    assert click_calls == ["Use without an account"]


def test_androidworld_setup_falls_back_to_native_uiautomator_for_missing_label() -> None:
    click_calls: list[str] = []

    class A11yMethod(Enum):
        FORWARDER = "forwarder"
        UIAUTOMATOR = "uiautomator"

    class NativeController:
        _a11y_method = A11yMethod.FORWARDER

        def get_ui_elements(self):
            if self._a11y_method is A11yMethod.UIAUTOMATOR:
                return [_setup_element("Accept & continue")]
            return [_setup_element("Welcome to Chrome")]

    class Controller:
        def __init__(self) -> None:
            self._env = NativeController()

        def click_element(self, element_text: str) -> None:
            click_calls.append(element_text)
            if self._env._a11y_method is not A11yMethod.UIAUTOMATOR:
                raise ValueError(f'Target text "{element_text}" not found.')

    tools_module = SimpleNamespace(AndroidToolController=Controller)
    patch_androidworld_setup_click_retry(
        tools_module,
        attempts=1,
        delay_seconds=0,
    )
    controller = Controller()

    controller.click_element("Accept & continue")

    assert click_calls == ["Accept & continue"]
    assert controller._env._a11y_method is A11yMethod.FORWARDER


def test_androidworld_setup_skips_click_when_chrome_is_complete() -> None:
    click_calls: list[str] = []

    class Controller:
        def __init__(self) -> None:
            self._env = SimpleNamespace(
                get_ui_elements=lambda: [_setup_element("Search or type web address")]
            )

        def click_element(self, element_text: str) -> None:
            click_calls.append(element_text)
            raise ValueError(f'Target text "{element_text}" not found.')

    tools_module = SimpleNamespace(AndroidToolController=Controller)
    patch_androidworld_setup_click_retry(
        tools_module,
        attempts=1,
        delay_seconds=0,
    )

    assert Controller().click_element("No thanks") is None
    assert click_calls == []


def test_androidworld_setup_resolves_contacts_chooser_before_onboarding() -> None:
    click_calls: list[str] = []
    visible = ["Open with Contacts", "Always"]

    class Controller:
        def __init__(self) -> None:
            self._env = SimpleNamespace(
                get_ui_elements=lambda: [_setup_element(text) for text in visible]
            )

        def click_element(self, element_text: str) -> None:
            click_calls.append(element_text)
            if element_text == "Always":
                visible[:] = ["Skip"]

    tools_module = SimpleNamespace(AndroidToolController=Controller)
    patch_androidworld_setup_click_retry(
        tools_module,
        attempts=2,
        delay_seconds=0,
    )

    Controller().click_element("Skip")

    assert click_calls == ["Always", "Skip"]


def test_androidworld_setup_accepts_ready_contacts_without_notification_dialog() -> None:
    click_calls: list[str] = []

    class Controller:
        def __init__(self) -> None:
            self._env = SimpleNamespace(
                get_ui_elements=lambda: [_setup_element("Search contacts")]
            )

        def click_element(self, element_text: str) -> None:
            click_calls.append(element_text)

    tools_module = SimpleNamespace(AndroidToolController=Controller)
    patch_androidworld_setup_click_retry(
        tools_module,
        attempts=1,
        delay_seconds=0,
    )

    assert Controller().click_element("Don't allow") is None
    assert click_calls == []


def test_androidworld_legacy_apk_install_uses_platform_bypass() -> None:
    commands: list[list[str]] = []
    response = SimpleNamespace(status="ok", generic=SimpleNamespace(output=b""))
    setup_module = SimpleNamespace(
        download_and_install_apk=lambda apk_name, raw_env: (_ for _ in ()).throw(
            RuntimeError("INSTALL_FAILED_DEPRECATED_SDK_VERSION")
        ),
        apps=SimpleNamespace(download_app_data=lambda apk_name: f"/tmp/{apk_name}"),
        adb_utils=SimpleNamespace(
            issue_generic_request=lambda command, raw_env, timeout_sec: (
                commands.append(command) or response
            ),
            check_ok=lambda actual_response, message: None,
        ),
    )

    patch_androidworld_legacy_apk_install(setup_module)
    setup_module.download_and_install_apk("clipper.apk", object())

    assert commands == [
        [
            "install",
            "--bypass-low-target-sdk-block",
            "/tmp/clipper.apk",
        ]
    ]


def test_androidworld_osmand_chcon_requires_verified_map() -> None:
    commands: list[list[str]] = []
    response = SimpleNamespace(status="ok", generic=SimpleNamespace(output=b""))

    class OsmandApp:
        MAP_NAMES = ("map.obf",)
        DEVICE_MAPS_PATH = "/storage/maps/"

        @classmethod
        def setup(cls, env) -> None:
            raise RuntimeError(
                "chcon: Operation not supported on transport endpoint"
            )

    setup_module = SimpleNamespace(
        apps=SimpleNamespace(OsmandApp=OsmandApp),
        adb_utils=SimpleNamespace(
            issue_generic_request=lambda command, controller: (
                commands.append(command) or response
            ),
            check_ok=lambda actual_response, message: None,
        ),
    )

    patch_androidworld_osmand_storage_setup(setup_module)
    OsmandApp.setup(SimpleNamespace(controller=object()))

    assert commands == [["shell", "test", "-s", "/storage/maps/map.obf"]]


def test_androidworld_gallery_missing_dialog_requires_verified_appop() -> None:
    commands: list[list[str]] = []
    response = SimpleNamespace(
        status="ok",
        generic=SimpleNamespace(output=b"MANAGE_EXTERNAL_STORAGE: allow"),
    )

    class GalleryApp:
        app_name = "simple gallery pro"

        @classmethod
        def setup(cls, env) -> None:
            raise ValueError('AndroidWorld setup target "All files" not found')

    class VlcApp:
        app_name = "vlc"

        @classmethod
        def setup(cls, env) -> None:
            return None

    adb_utils = SimpleNamespace(
        extract_package_name=lambda activity: activity,
        get_adb_activity=lambda app_name: f"package.{app_name}",
        issue_generic_request=lambda command, controller: (
            commands.append(command) or response
        ),
        check_ok=lambda actual_response, message: None,
    )
    setup_module = SimpleNamespace(
        apps=SimpleNamespace(
            SimpleGalleryProApp=GalleryApp,
            VlcApp=VlcApp,
            adb_utils=adb_utils,
        )
    )

    patch_androidworld_special_storage_setup(setup_module)
    GalleryApp.setup(SimpleNamespace(controller=object()))

    assert commands.count(
        [
            "shell",
            "appops",
            "set",
            "package.simple gallery pro",
            "MANAGE_EXTERNAL_STORAGE",
            "allow",
        ]
    ) == 2


def test_androidworld_setup_retries_before_saving_snapshot() -> None:
    setup_calls: list[object] = []
    snapshot_calls: list[tuple[str, object]] = []
    env = SimpleNamespace(controller=object())

    class App:
        app_name = "chrome"

        @classmethod
        def setup(cls, actual_env) -> None:
            setup_calls.append(actual_env)
            if len(setup_calls) == 1:
                raise ValueError("onboarding not ready")

    setup_module = SimpleNamespace(
        setup_app=lambda app, actual_env: None,
        app_snapshot=SimpleNamespace(
            save_snapshot=lambda name, controller: snapshot_calls.append(
                (name, controller)
            )
        ),
    )

    patch_androidworld_setup_fail_closed(setup_module, attempts=2)
    setup_module.setup_app(App, env)

    assert setup_calls == [env, env]
    assert snapshot_calls == [("chrome", env.controller)]


def test_androidworld_setup_failure_does_not_save_snapshot() -> None:
    snapshot_calls: list[tuple[str, object]] = []
    env = SimpleNamespace(controller=object())

    class App:
        app_name = "chrome"

        @classmethod
        def setup(cls, actual_env) -> None:
            raise ValueError("onboarding never became ready")

    setup_module = SimpleNamespace(
        setup_app=lambda app, actual_env: None,
        app_snapshot=SimpleNamespace(
            save_snapshot=lambda name, controller: snapshot_calls.append(
                (name, controller)
            )
        ),
    )

    patch_androidworld_setup_fail_closed(setup_module, attempts=2)

    with pytest.raises(ValueError, match="onboarding never became ready"):
        setup_module.setup_app(App, env)

    assert snapshot_calls == []


def test_androidworld_restores_chrome_snapshot_after_task_initialize() -> None:
    restore_calls: list[tuple[str, object]] = []
    controller = object()

    def restore_snapshot(app_name: str, actual_controller: object) -> None:
        restore_calls.append((app_name, actual_controller))

    task = SimpleNamespace(app_names=["chrome"])
    env = SimpleNamespace(controller=controller)

    restore_task_app_snapshots_after_initialize(restore_snapshot, task, env)

    assert restore_calls == [("chrome", controller)]


def test_androidworld_does_not_restore_other_apps_after_task_initialize() -> None:
    restore_calls: list[str] = []
    task = SimpleNamespace(app_names=["contacts", "audio recorder"])
    env = SimpleNamespace(controller=object())

    restore_task_app_snapshots_after_initialize(
        lambda app_name, controller: restore_calls.append(app_name),
        task,
        env,
    )

    assert restore_calls == []


def test_androidworld_post_initialize_snapshot_restore_fails_closed() -> None:
    def missing_snapshot(app_name: str, controller: object) -> None:
        raise RuntimeError(f"Snapshot not found for {app_name}")

    with pytest.raises(RuntimeError, match="Snapshot not found for chrome"):
        restore_task_app_snapshots_after_initialize(
            missing_snapshot,
            SimpleNamespace(app_names=["chrome"]),
            SimpleNamespace(controller=object()),
        )


def test_androidworld_restores_chrome_before_agent_reads_task_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    env = SimpleNamespace(controller=object())
    task = SimpleNamespace(
        app_names=["chrome"],
        initialize_task=lambda actual_env: events.append("task_initialized"),
    )

    def restore_snapshot(app_name: str, controller: object) -> None:
        events.append(f"snapshot_restored:{app_name}")

    monkeypatch.setattr(
        launch,
        "_prepare_native_androidworld_a11y_runtime",
        lambda *args, **kwargs: events.append("a11y_ready"),
    )

    launch._wrap_task_initialize_for_observation_runtime(
        task,
        agent=SimpleNamespace(),
        adb_serial="emulator-5554",
        adb_path="adb",
        oob_url="",
        console_port=5554,
        restore_app_snapshot=restore_snapshot,
        after_initialized=lambda initialized_task: events.append("context_updated"),
    )
    task.initialize_task(env)

    assert events == [
        "task_initialized",
        "snapshot_restored:chrome",
        "context_updated",
        "a11y_ready",
    ]
