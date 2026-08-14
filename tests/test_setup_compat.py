from __future__ import annotations

from enum import Enum
import random
from types import SimpleNamespace

import pytest

from src.integrations.android_world import launch
from src.integrations.android_world.setup_compat import (
    _setup_click_is_already_complete,
    patch_androidworld_legacy_apk_install,
    patch_androidworld_open_tracks_setup,
    patch_androidworld_osmand_storage_setup,
    patch_androidworld_setup_click_retry,
    patch_androidworld_setup_fail_closed,
    patch_androidworld_special_storage_setup,
    patch_androidworld_vlc_apk_selection,
    resolve_androidworld_task_setup_apps,
    restore_task_app_snapshots_after_initialize,
)


def _sqlite_utils_with_reader(reader):
    return SimpleNamespace(
        delete_all_rows_from_table=lambda *args, **kwargs: None,
        insert_rows_to_remote_db=lambda *args, **kwargs: None,
        get_rows_from_remote_device=reader,
        sqlite3=SimpleNamespace(),
        os=SimpleNamespace(),
        time=SimpleNamespace(),
        file_utils=SimpleNamespace(),
        adb_utils=SimpleNamespace(),
        sqlite_schema_utils=SimpleNamespace(),
    )


def test_androidworld_sqlite_reader_retries_disappearing_sidecar() -> None:
    calls = 0

    def read_rows(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError(
                "adb pull /data/data/app/databases/accounting.db-journal: "
                "No such file or directory"
            )
        return ["row"]

    sqlite_utils = _sqlite_utils_with_reader(read_rows)
    launch._patch_androidworld_sqlite_writeback(sqlite_utils)

    assert sqlite_utils.get_rows_from_remote_device("table", "db", object()) == [
        "row"
    ]
    assert calls == 2


def test_androidworld_sqlite_reader_does_not_hide_missing_main_db() -> None:
    def read_rows(*args, **kwargs):
        raise RuntimeError(
            "adb pull /data/data/app/databases/accounting.db: No such file or directory"
        )

    sqlite_utils = _sqlite_utils_with_reader(read_rows)
    launch._patch_androidworld_sqlite_writeback(sqlite_utils)

    with pytest.raises(RuntimeError, match="accounting.db"):
        sqlite_utils.get_rows_from_remote_device("table", "db", object())


def test_androidworld_camera_setup_accepts_completed_onboarding() -> None:
    assert _setup_click_is_already_complete({"Options", "Shutter"}, "NEXT")


def test_androidworld_rehydrates_receipt_image_from_official_instance_seed() -> None:
    generated_image = object()

    class MarkorTranscribeReceipt:
        @classmethod
        def generate_random_params(cls) -> dict[str, object]:
            return {
                "img": generated_image,
                "file_name": "receipt.md",
                "text": f"receipt-{random.randint(1, 1_000_000)}",
            }

    random.seed(1217212335)
    expected_text = f"receipt-{random.randint(1, 1_000_000)}"
    random.seed(7)
    expected_next_random = random.random()
    random.seed(7)

    hydrated = launch._rehydrate_task_params(
        params={
            "file_name": "receipt.md",
            "text": expected_text,
            "seed": 1217212335,
        },
        task_type=MarkorTranscribeReceipt,
    )

    assert hydrated["img"] is generated_image
    assert hydrated["text"] == expected_text
    assert random.random() == expected_next_random


def test_androidworld_rejects_mismatched_generated_receipt_params() -> None:
    class MarkorTranscribeReceipt:
        @classmethod
        def generate_random_params(cls) -> dict[str, object]:
            return {
                "img": object(),
                "file_name": "receipt.md",
                "text": "official text",
            }

    with pytest.raises(ValueError, match="canonical source: text"):
        launch._rehydrate_task_params(
            params={
                "file_name": "receipt.md",
                "text": "different text",
                "seed": 1217212335,
            },
            task_type=MarkorTranscribeReceipt,
        )


def _setup_element(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        package_name="com.android.chrome",
        text=text,
        content_description="",
        bbox_pixels=SimpleNamespace(
            x_min=100,
            y_min=180,
            x_max=500,
            y_max=220,
        ),
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


def test_androidworld_setup_selects_contacts_before_resolving_chooser() -> None:
    click_calls: list[str] = []
    visible = ["Open with Omnibot", "Contacts", "Always"]

    class Controller:
        def __init__(self) -> None:
            self._env = SimpleNamespace(
                get_ui_elements=lambda: [_setup_element(text) for text in visible]
            )

        def click_element(self, element_text: str) -> None:
            click_calls.append(element_text)
            if element_text == "Contacts":
                visible[:] = ["Open with Contacts", "Always"]
            elif element_text == "Always":
                visible[:] = ["Skip"]

    tools_module = SimpleNamespace(AndroidToolController=Controller)
    patch_androidworld_setup_click_retry(
        tools_module,
        attempts=3,
        delay_seconds=0,
    )

    Controller().click_element("Skip")

    assert click_calls == ["Contacts", "Always", "Skip"]


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


def test_androidworld_vlc_install_selects_x86_64_apk() -> None:
    selected_apks: list[tuple[str, ...]] = []
    commands: list[list[str]] = []
    response = SimpleNamespace(
        status="ok",
        generic=SimpleNamespace(output=b"x86_64\n"),
    )

    class VlcApp:
        apk_names = (
            "org.videolan.vlc_13050407.apk",
            "org.videolan.vlc_13050408.apk",
        )

    setup_module = SimpleNamespace(
        maybe_install_app=lambda app, env: selected_apks.append(tuple(app.apk_names)),
        apps=SimpleNamespace(VlcApp=VlcApp),
        adb_utils=SimpleNamespace(
            issue_generic_request=lambda command, controller: (
                commands.append(command) or response
            ),
            check_ok=lambda actual_response, message: None,
        ),
    )

    patch_androidworld_vlc_apk_selection(setup_module)
    setup_module.maybe_install_app(VlcApp, SimpleNamespace(controller=object()))

    assert commands == [["shell", "getprop", "ro.product.cpu.abi"]]
    assert selected_apks == [("org.videolan.vlc_13050408.apk",)]
    assert VlcApp.apk_names == (
        "org.videolan.vlc_13050407.apk",
        "org.videolan.vlc_13050408.apk",
    )


def test_androidworld_vlc_install_rejects_unknown_abi() -> None:
    response = SimpleNamespace(
        status="ok",
        generic=SimpleNamespace(output=b"riscv64\n"),
    )

    class VlcApp:
        apk_names = (
            "org.videolan.vlc_13050407.apk",
            "org.videolan.vlc_13050408.apk",
        )

    setup_module = SimpleNamespace(
        maybe_install_app=lambda app, env: (_ for _ in ()).throw(
            AssertionError("unsupported ABI must not probe an APK")
        ),
        apps=SimpleNamespace(VlcApp=VlcApp),
        adb_utils=SimpleNamespace(
            issue_generic_request=lambda command, controller: response,
            check_ok=lambda actual_response, message: None,
        ),
    )

    patch_androidworld_vlc_apk_selection(setup_module)

    with pytest.raises(RuntimeError, match="riscv64"):
        setup_module.maybe_install_app(
            VlcApp,
            SimpleNamespace(controller=object()),
        )


def test_androidworld_osmand_chcon_requires_verified_map() -> None:
    commands: list[list[str]] = []
    response = SimpleNamespace(status="ok", generic=SimpleNamespace(output=b""))

    class OsmAndApp:
        MAP_NAMES = ("map.obf",)
        DEVICE_MAPS_PATH = "/storage/maps/"

        @classmethod
        def setup(cls, env) -> None:
            raise RuntimeError(
                "chcon: Operation not supported on transport endpoint"
            )

    setup_module = SimpleNamespace(
        apps=SimpleNamespace(OsmAndApp=OsmAndApp),
        adb_utils=SimpleNamespace(
            issue_generic_request=lambda command, controller: (
                commands.append(command) or response
            ),
            check_ok=lambda actual_response, message: None,
        ),
    )

    patch_androidworld_osmand_storage_setup(setup_module)
    OsmAndApp.setup(SimpleNamespace(controller=object()))

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


def test_androidworld_vlc_missing_dialog_requires_verified_appop() -> None:
    commands: list[list[str]] = []
    response = SimpleNamespace(
        status="ok",
        generic=SimpleNamespace(output=b"MANAGE_EXTERNAL_STORAGE: allow"),
    )

    class GalleryApp:
        app_name = "simple gallery pro"

        @classmethod
        def setup(cls, env) -> None:
            return None

    class VlcApp:
        app_name = "vlc"

        @classmethod
        def setup(cls, env) -> None:
            raise ValueError(
                'AndroidWorld setup target "GRANT PERMISSION" not found'
            )

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
    VlcApp.setup(SimpleNamespace(controller=object()))

    assert commands.count(
        [
            "shell",
            "appops",
            "set",
            "package.vlc",
            "MANAGE_EXTERNAL_STORAGE",
            "allow",
        ]
    ) == 2


def test_androidworld_open_tracks_missing_dialog_requires_bluetooth_grants() -> None:
    commands: list[list[str]] = []
    events: list[str] = []
    response = SimpleNamespace(
        status="ok",
        generic=SimpleNamespace(
            output=(
                b"android.permission.BLUETOOTH_SCAN: granted=true, flags=[]\n"
                b"android.permission.BLUETOOTH_CONNECT: granted=true, flags=[]"
            )
        ),
    )

    class OpenTracksApp:
        @classmethod
        def setup(cls, env) -> None:
            raise ValueError('AndroidWorld setup target "Allow" not found')

    adb_utils = SimpleNamespace(
        get_adb_activity=lambda app_name: "de.dennisguse.opentracks/.MainActivity",
        extract_package_name=lambda activity: activity.split("/")[0],
        issue_generic_request=lambda command, controller: (
            commands.append(command) or response
        ),
        check_ok=lambda actual_response, message: None,
        launch_app=lambda app_name, controller: events.append(f"launch:{app_name}"),
        close_app=lambda app_name, controller: events.append(f"close:{app_name}"),
    )
    setup_module = SimpleNamespace(
        apps=SimpleNamespace(OpenTracksApp=OpenTracksApp),
        adb_utils=adb_utils,
    )

    patch_androidworld_open_tracks_setup(setup_module)
    OpenTracksApp.setup(SimpleNamespace(controller=object()))

    assert commands == [
        ["shell", "dumpsys", "package", "de.dennisguse.opentracks"]
    ]
    assert events == ["launch:activity tracker", "close:activity tracker"]


def test_androidworld_open_tracks_missing_dialog_rejects_missing_grant() -> None:
    response = SimpleNamespace(
        status="ok",
        generic=SimpleNamespace(
            output=b"android.permission.BLUETOOTH_SCAN: granted=true, flags=[]"
        ),
    )

    class OpenTracksApp:
        @classmethod
        def setup(cls, env) -> None:
            raise ValueError('AndroidWorld setup target "Allow" not found')

    adb_utils = SimpleNamespace(
        get_adb_activity=lambda app_name: "de.dennisguse.opentracks/.MainActivity",
        extract_package_name=lambda activity: activity.split("/")[0],
        issue_generic_request=lambda command, controller: response,
        check_ok=lambda actual_response, message: None,
        launch_app=lambda app_name, controller: None,
        close_app=lambda app_name, controller: None,
    )
    setup_module = SimpleNamespace(
        apps=SimpleNamespace(OpenTracksApp=OpenTracksApp),
        adb_utils=adb_utils,
    )

    patch_androidworld_open_tracks_setup(setup_module)

    with pytest.raises(RuntimeError, match="BLUETOOTH_CONNECT"):
        OpenTracksApp.setup(SimpleNamespace(controller=object()))


def test_androidworld_setup_retries_before_saving_snapshot() -> None:
    setup_calls: list[object] = []
    snapshot_calls: list[tuple[str, object]] = []
    env = SimpleNamespace(controller=object())

    class App:
        app_name = "files"

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
    assert snapshot_calls == [("files", env.controller)]


def test_androidworld_setup_drains_ok_for_every_snapshot_app() -> None:
    events: list[str] = []
    visible = ["OK"]
    confirmation_count = 0

    class Controller:
        def get_ui_elements(self):
            return [_setup_element(label) for label in visible]

    class AndroidToolController:
        def __init__(self, env) -> None:
            self._env = env

        def click_element(self, element_text: str) -> None:
            raise AssertionError("bounded setup confirmations must use native tap")

    class App:
        app_name = "files"

        @classmethod
        def setup(cls, env) -> None:
            events.append("setup")

    def issue_generic_request(command, controller):
        nonlocal confirmation_count
        events.append(f"tap:{command[3]}:{command[4]}")
        confirmation_count += 1
        visible[:] = ["OK"] if confirmation_count == 1 else []
        return SimpleNamespace(status="ok")

    setup_module = SimpleNamespace(
        setup_app=lambda app, env: None,
        apps=SimpleNamespace(
            tools=SimpleNamespace(AndroidToolController=AndroidToolController)
        ),
        adb_utils=SimpleNamespace(
            issue_generic_request=issue_generic_request,
            check_ok=lambda response, message: None,
        ),
        app_snapshot=SimpleNamespace(
            save_snapshot=lambda name, controller: events.append(f"snapshot:{name}")
        ),
    )

    patch_androidworld_setup_fail_closed(
        setup_module,
        attempts=1,
        delay_seconds=0,
    )
    setup_module.setup_app(App, SimpleNamespace(controller=Controller()))

    assert events == [
        "setup",
        "tap:300:200",
        "tap:300:200",
        "snapshot:files",
    ]


def test_androidworld_setup_completes_visible_onboarding_before_snapshot() -> None:
    events: list[str] = []
    visible = ["Get started"]

    class Controller:
        def get_ui_elements(self):
            return [_setup_element(label) for label in visible]

    class AndroidToolController:
        def __init__(self, env) -> None:
            self._env = env

    class App:
        app_name = "audio recorder"

        @classmethod
        def setup(cls, env) -> None:
            events.append("setup")

    def issue_generic_request(command, controller):
        events.append(f"tap:{command[3]}:{command[4]}")
        if visible == ["Get started"]:
            visible[:] = ["Setup", "Apply"]
        else:
            visible.clear()
        return SimpleNamespace(status="ok")

    setup_module = SimpleNamespace(
        setup_app=lambda app, env: None,
        apps=SimpleNamespace(
            tools=SimpleNamespace(AndroidToolController=AndroidToolController)
        ),
        adb_utils=SimpleNamespace(
            launch_app=lambda name, controller: events.append(f"launch:{name}"),
            close_app=lambda name, controller: events.append(f"close:{name}"),
            issue_generic_request=issue_generic_request,
            check_ok=lambda response, message: None,
        ),
        app_snapshot=SimpleNamespace(
            save_snapshot=lambda name, controller: events.append(f"snapshot:{name}")
        ),
    )

    patch_androidworld_setup_fail_closed(
        setup_module,
        attempts=1,
        delay_seconds=0,
    )
    setup_module.setup_app(App, SimpleNamespace(controller=Controller()))

    assert events == [
        "setup",
        "launch:audio recorder",
        "tap:300:200",
        "tap:300:200",
        "close:audio recorder",
        "snapshot:audio recorder",
    ]


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


def test_androidworld_setup_fixes_contacts_notification_before_snapshot() -> None:
    events: list[str] = []
    commands: list[list[str]] = []
    permission_state = (
        "android.permission.POST_NOTIFICATIONS: granted=false, "
        "flags=[ USER_SET|USER_FIXED]"
    )

    def issue_generic_request(command, controller):
        commands.append(command)
        events.append(command[1] if command[1] == "dumpsys" else command[2])
        output = permission_state.encode() if command[1] == "dumpsys" else b""
        return SimpleNamespace(status="ok", generic=SimpleNamespace(output=output))

    class App:
        app_name = "contacts"

        @classmethod
        def setup(cls, env) -> None:
            events.append("setup")

    setup_module = SimpleNamespace(
        setup_app=lambda app, env: None,
        adb_utils=SimpleNamespace(
            issue_generic_request=issue_generic_request,
            check_ok=lambda response, message: None,
        ),
        app_snapshot=SimpleNamespace(
            save_snapshot=lambda name, controller: events.append("snapshot")
        ),
    )

    patch_androidworld_setup_fail_closed(setup_module, attempts=1)
    setup_module.setup_app(App, SimpleNamespace(controller=object()))

    assert events == [
        "setup",
        "revoke",
        "set-permission-flags",
        "dumpsys",
        "snapshot",
    ]
    assert commands == [
        [
            "shell",
            "pm",
            "revoke",
            "com.google.android.contacts",
            "android.permission.POST_NOTIFICATIONS",
        ],
        [
            "shell",
            "pm",
            "set-permission-flags",
            "com.google.android.contacts",
            "android.permission.POST_NOTIFICATIONS",
            "user-set",
            "user-fixed",
        ],
        ["shell", "dumpsys", "package", "com.google.android.contacts"],
    ]


def test_androidworld_setup_rejects_unfixed_contacts_notification_state() -> None:
    snapshot_calls: list[str] = []
    response = SimpleNamespace(
        status="ok",
        generic=SimpleNamespace(
            output=(
                b"android.permission.POST_NOTIFICATIONS: granted=false, flags=[]"
            )
        ),
    )

    class App:
        app_name = "contacts"

        @classmethod
        def setup(cls, env) -> None:
            return None

    setup_module = SimpleNamespace(
        setup_app=lambda app, env: None,
        adb_utils=SimpleNamespace(
            issue_generic_request=lambda command, controller: response,
            check_ok=lambda actual_response, message: None,
        ),
        app_snapshot=SimpleNamespace(
            save_snapshot=lambda name, controller: snapshot_calls.append(name)
        ),
    )

    patch_androidworld_setup_fail_closed(setup_module, attempts=1)

    with pytest.raises(RuntimeError, match="not denied and user-fixed"):
        setup_module.setup_app(App, SimpleNamespace(controller=object()))

    assert snapshot_calls == []


def test_androidworld_setup_uses_uiautomator_then_restores_forwarder() -> None:
    events: list[object] = []

    class A11yMethod(Enum):
        FORWARDER = "forwarder"
        UIAUTOMATOR = "uiautomator"

    controller = SimpleNamespace(_a11y_method=A11yMethod.FORWARDER)
    env = SimpleNamespace(controller=controller)

    class App:
        app_name = "contacts"

        @classmethod
        def setup(cls, actual_env) -> None:
            events.append(actual_env.controller._a11y_method)

    setup_module = SimpleNamespace(
        setup_app=lambda app, actual_env: None,
        adb_utils=SimpleNamespace(
            issue_generic_request=lambda command, actual_controller: SimpleNamespace(
                status="ok",
                generic=SimpleNamespace(
                    output=(
                        b"android.permission.POST_NOTIFICATIONS: granted=false, "
                        b"flags=[ USER_SET|USER_FIXED]"
                    )
                ),
            ),
            check_ok=lambda response, message: None,
        ),
        app_snapshot=SimpleNamespace(
            save_snapshot=lambda name, actual_controller: events.append(
                actual_controller._a11y_method
            )
        ),
    )

    patch_androidworld_setup_fail_closed(setup_module, attempts=1)
    setup_module.setup_app(App, env)

    assert events == [A11yMethod.UIAUTOMATOR, A11yMethod.FORWARDER]
    assert controller._a11y_method is A11yMethod.FORWARDER


def test_androidworld_restores_chrome_snapshot_after_task_initialize() -> None:
    restore_calls: list[tuple[str, object]] = []
    controller = object()

    def restore_snapshot(app_name: str, actual_controller: object) -> None:
        restore_calls.append((app_name, actual_controller))

    task = SimpleNamespace(app_names=["chrome"])
    env = SimpleNamespace(controller=controller)

    restore_task_app_snapshots_after_initialize(restore_snapshot, task, env)

    assert restore_calls == [("chrome", controller)]


def test_androidworld_setup_resolves_dynamic_task_instance_apps() -> None:
    class JoplinApp:
        app_name = "joplin"

    setup_module = SimpleNamespace(
        get_app_mapping=lambda app_name: JoplinApp if app_name == "joplin" else None,
        get_app_list_to_setup=lambda task_names: (),
    )

    setup_apps = resolve_androidworld_task_setup_apps(
        setup_module,
        task_types={"NotesIsTodo": SimpleNamespace(app_names=())},
        task_suite={
            "NotesIsTodo": [SimpleNamespace(app_names=("joplin",))],
        },
        selected_task_names=["NotesIsTodo"],
    )

    assert setup_apps == (JoplinApp,)


def test_androidworld_setup_deduplicates_class_instance_and_name_apps() -> None:
    class JoplinApp:
        app_name = "joplin"

    setup_module = SimpleNamespace(
        get_app_mapping=lambda app_name: JoplinApp if app_name == "joplin" else None,
        get_app_list_to_setup=lambda task_names: (JoplinApp,),
    )

    setup_apps = resolve_androidworld_task_setup_apps(
        setup_module,
        task_types={"JoplinTask": SimpleNamespace(app_names=("joplin",))},
        task_suite={"JoplinTask": [SimpleNamespace(app_names=("joplin",))]},
        selected_task_names=["JoplinTask"],
    )

    assert setup_apps == (JoplinApp,)


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


def test_androidworld_restores_chrome_before_agent_reads_task_context() -> None:
    events: list[str] = []
    env = SimpleNamespace(controller=object())
    task = SimpleNamespace(
        app_names=["chrome"],
        initialize_task=lambda actual_env: events.append("task_initialized"),
    )

    def restore_snapshot(app_name: str, controller: object) -> None:
        events.append(f"snapshot_restored:{app_name}")

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
    ]


def test_androidworld_closes_system_dialogs_with_async_broadcast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def run_adb_command(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"returncode": 0}

    monkeypatch.setattr(launch, "_run_adb_command", run_adb_command)

    launch._close_android_system_dialogs(
        adb_serial="emulator-5554",
        adb_path="/sdk/adb",
        failure="test_failure",
    )

    assert calls == [
        {
            "adb_serial": "emulator-5554",
            "adb_path": "/sdk/adb",
            "adb_args": [
                "shell",
                "am",
                "broadcast",
                "--async",
                "-a",
                "android.intent.action.CLOSE_SYSTEM_DIALOGS",
            ],
            "timeout_sec": 15,
            "capture_stdout": True,
        }
    ]
