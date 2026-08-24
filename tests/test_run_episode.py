from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import xml.etree.ElementTree as ET

from PIL import Image
import pytest

from omniflow import Action
from src.integrations.android_world.apps import resolve_androidworld_app_name
from src.integrations.android_world.host import AndroidWorldHost
from src.integrations.android_world.run_episode import (
    ANDROID_PERMISSION_DENY_RESOURCE_IDS,
    _androidworld_a11y_forwarder_installed,
    _androidworld_adb_file_transfer_timeout_sec,
    _androidworld_setup_apps_for_suite,
    _androidworld_setup_timeout_sec,
    _bounded_androidworld_adb_file_transfer_timeout,
    _ensure_androidworld_a11y_forwarder,
    _ExperimentAgentAdapter,
    _model_base_url_for_profile,
    _oob_control_accessibility_services,
    _OpenAICompatibleMultimodalWrapper,
    _patch_androidworld_adb_controller_install_compat,
    _patch_androidworld_adb_output_sanitizer,
    _patch_androidworld_apk_install_compat,
    _patch_androidworld_app_launch,
    _patch_androidworld_chcon_compat,
    _patch_androidworld_clipboard_read_compat,
    _patch_androidworld_current_activity,
    _patch_androidworld_directory_clear,
    _patch_androidworld_expense_setup_timeout,
    _patch_androidworld_optional_setup_click,
    _prepare_androidworld_episode_apps,
    _repair_androidworld_chrome_first_run,
    _reset_androidworld_file_picker_state,
    _result_has_official_validator_conclusion,
    _run_androidworld_setup_apps,
    _runtime_execution_trace,
    _wait_for_androidworld_a11y,
    build_parser,
)


def test_official_androidworld_model_request_uses_bounded_no_thinking_policy(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "model": "test-model",
                    "choices": [{"message": {"content": "done"}}],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            ).encode()

    def urlopen(request, **_kwargs: object) -> Response:
        captured.update(json.loads(request.data.decode()))
        return Response()

    monkeypatch.setattr(
        "src.integrations.android_world.run_episode.urllib.request.urlopen",
        urlopen,
    )
    wrapper = _OpenAICompatibleMultimodalWrapper(
        model_name="test-model",
        api_key="test-key",
        max_retry=1,
    )

    text, _safety, _metadata = wrapper.predict("Choose one action")

    assert text == "done"
    assert captured["max_tokens"] == 512
    assert captured["reasoning_effort"] == "none"
    assert captured["enable_thinking"] is False


def test_androidworld_episode_cli_has_no_direct_function_flags() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--function-id", "complete_task"])


def test_oob_control_uses_only_formal_accessibility_services() -> None:
    assert _oob_control_accessibility_services(
        [
            "com.google.androidenv.accessibilityforwarder/Service",
            "com.example.unrelated/LegacyService",
            "cn.com.omnimind.bot.debug/"
            "cn.com.omnimind.accessibility.service.AssistsService",
        ]
    ) == [
        "cn.com.omnimind.bot.debug/"
        "cn.com.omnimind.accessibility.service.AssistsService",
    ]


def test_formal_model_profile_uses_protocol_base_url_without_env_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omniflow.vlm.model_config import resolve_openai_compatible_config

    monkeypatch.setenv("OPENAI_API_KEY", "stale-openai-key")
    monkeypatch.setenv("LLMTHU_API_KEY", "test-key")
    api_key, base_url = resolve_openai_compatible_config(
        profile="llmthu",
        base_url=_model_base_url_for_profile("llmthu"),
    )

    assert api_key == "test-key"
    assert base_url == "https://llmapi.paratera.com/v1"


def test_formal_model_profile_rejects_retired_credential_aliases() -> None:
    from omniflow.vlm.model_config import resolve_openai_compatible_config

    with pytest.raises(ValueError, match="model_endpoint_profile_incomplete:llmthu"):
        resolve_openai_compatible_config(
            profile="llmthu",
            base_url="https://llmapi.paratera.com/v1",
            environment={
                "LLMTHU_KEY": "retired-key",
                "LLMTHU_BASE_URL": "https://retired.example/v1",
            },
        )


def test_system_package_resolves_from_official_registry_when_not_enumerated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adb_utils = SimpleNamespace(
        get_all_apps=lambda _controller: [],
        get_adb_activity=lambda app_name: (
            "com.android.settings/.Settings" if app_name == "settings" else None
        ),
        _PATTERN_TO_ACTIVITY={
            "settings|system settings": "com.android.settings/.Settings"
        },
    )
    android_world = ModuleType("android_world")
    android_world_env = ModuleType("android_world.env")
    android_world_env.adb_utils = adb_utils
    android_world.env = android_world_env
    monkeypatch.setitem(sys.modules, "android_world", android_world)
    monkeypatch.setitem(sys.modules, "android_world.env", android_world_env)

    assert (
        resolve_androidworld_app_name("com.android.settings", object())
        == "settings"
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
        "src.integrations.android_world.run_episode.importlib.import_module",
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


def test_androidworld_setup_resolves_contacts_open_with_before_skip(
    monkeypatch,
) -> None:
    calls: list[str] = []

    class Environment:
        screen = "chooser"

        def get_ui_elements(self):
            if self.screen == "chooser":
                labels = ("Open with", "Omnibot", "Contacts", "Just once", "Always")
            elif self.screen == "confirm":
                labels = ("Open with", "Contacts", "Just once", "Always")
            elif self.screen == "onboarding":
                labels = ("Skip",)
            else:
                labels = ("Contacts",)
            return [
                SimpleNamespace(text=label, content_description=None, package_name="android")
                for label in labels
            ]

    class AndroidToolController:
        def __init__(self, env) -> None:
            self._env = env

        def click_element(self, element_text: str) -> None:
            calls.append(element_text)
            if element_text == "Skip" and self._env.screen == "chooser":
                raise ValueError(
                    'Target text "Skip" not found. Visible labels: '
                    "['Open with', 'Omnibot', 'Contacts', 'Just once', 'Always']"
                )
            if element_text == "Contacts" and self._env.screen == "chooser":
                self._env.screen = "confirm"
                return
            if element_text == "Just once" and self._env.screen == "confirm":
                self._env.screen = "onboarding"
                return
            if element_text == "Skip" and self._env.screen == "onboarding":
                self._env.screen = "contacts"
                return
            raise AssertionError((self._env.screen, element_text))

    tools_module = SimpleNamespace(AndroidToolController=AndroidToolController)
    monkeypatch.setattr(
        "src.integrations.android_world.run_episode.importlib.import_module",
        lambda name: tools_module
        if name == "android_world.env.tools"
        else pytest.fail(f"unexpected import: {name}"),
    )
    controller = AndroidToolController(Environment())

    patch = _patch_androidworld_optional_setup_click()
    assert patch is not None
    controller_type, original = patch
    try:
        controller.click_element("Skip")
    finally:
        controller_type.click_element = original

    assert controller._env.screen == "contacts"
    assert calls == ["Skip", "Contacts", "Just once", "Skip"]


def test_androidworld_setup_skips_only_absent_markor_final_ok(monkeypatch) -> None:
    class AndroidToolController:
        def __init__(self, env) -> None:
            self._env = env

        def click_element(self, element_text: str) -> None:
            raise ValueError(f'Target text "{element_text}" not found.')

    tools_module = SimpleNamespace(AndroidToolController=AndroidToolController)
    monkeypatch.setattr(
        "src.integrations.android_world.run_episode.importlib.import_module",
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


def test_androidworld_setup_skips_absent_camera_onboarding_next(monkeypatch) -> None:
    class AndroidToolController:
        def __init__(self, env) -> None:
            self._env = env

        def click_element(self, element_text: str) -> None:
            raise ValueError(f'Target text "{element_text}" not found.')

    tools_module = SimpleNamespace(AndroidToolController=AndroidToolController)
    monkeypatch.setattr(
        "src.integrations.android_world.run_episode.importlib.import_module",
        lambda name: tools_module
        if name == "android_world.env.tools"
        else pytest.fail(f"unexpected import: {name}"),
    )
    controller = AndroidToolController(
        SimpleNamespace(
            foreground_activity_name="com.android.camera2/.CameraActivity",
            get_ui_elements=lambda: [
                SimpleNamespace(package_name="com.android.camera2")
            ],
        )
    )

    patch = _patch_androidworld_optional_setup_click()
    assert patch is not None
    controller_type, original = patch
    try:
        controller.click_element("NEXT")
        with pytest.raises(ValueError, match="NEXT"):
            controller._env.foreground_activity_name = "com.example/.MainActivity"
            controller._env.get_ui_elements = lambda: []
            controller.click_element("NEXT")
        controller._env.foreground_activity_name = (
            "com.google.android.apps.nexuslauncher/.NexusLauncherActivity"
        )
        controller.click_element("NEXT")
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
        "src.integrations.android_world.run_episode.importlib.import_module",
        lambda name: SimpleNamespace(
            MARKOR_DATA="/storage/emulated/0/Documents/Markor"
        )
        if name == "android_world.env.device_constants"
        else SimpleNamespace()
        if name == "android_world.env.actuation"
        else real_import_module(name),
    )
    monkeypatch.setattr("src.integrations.android_world.run_episode.time.sleep", lambda _: None)

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
        ("compat", ("install", "/tmp/app.apk"), 180.0),
    ]


def test_androidworld_adb_controller_retries_without_unsupported_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...], float | None]] = []

    class Controller:
        def execute_command(self, args, timeout=None, device_specific=True):
            del device_specific
            calls.append((tuple(args), timeout))
            if "--bypass-low-target-sdk-block" in args:
                raise RuntimeError(
                    "adb stdout: [Unknown option --bypass-low-target-sdk-block]"
                )
            return b"installed"

    original_import = importlib.import_module

    def import_module(name):
        if name == "android_env.components.adb_controller":
            return SimpleNamespace(AdbController=Controller)
        return original_import(name)

    monkeypatch.setattr(importlib, "import_module", import_module)
    original = _patch_androidworld_adb_controller_install_compat()
    assert original is not None
    controller_type, original_execute = original
    try:
        result = controller_type().execute_command(
            ["install", "--bypass-low-target-sdk-block", "/tmp/app.apk"]
        )
    finally:
        controller_type.execute_command = original_execute

    assert result == b"installed"
    assert calls == [
        (("install", "--bypass-low-target-sdk-block", "/tmp/app.apk"), 180.0),
        (("install", "/tmp/app.apk"), 180.0),
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
    patches = _patch_androidworld_adb_output_sanitizer(controller)
    try:
        response = controller.execute_adb_call(object())
    finally:
        for controller_type, original in patches:
            controller_type.execute_adb_call = original

    assert response.generic.output == b"1697412907\n"
    assert int(response.generic.output.strip()) == 1697412907


def test_androidworld_adb_output_sanitizes_a11y_refresh_original_env() -> None:
    class OriginalEnv:
        def execute_adb_call(self, _request):
            return SimpleNamespace(
                generic=SimpleNamespace(
                    output=(
                        b"I0818 08:14:27.756116 11386938 "
                        b"ev_poll_posix.cc:593] FD from fork parent still in poll list\n"
                        b"com.google.androidenv.accessibilityforwarder/Service\n"
                    )
                )
            )

    class Controller:
        def __init__(self) -> None:
            self._original_env = OriginalEnv()

        def execute_adb_call(self, request):
            return self._original_env.execute_adb_call(request)

    controller = Controller()
    patches = _patch_androidworld_adb_output_sanitizer(controller)
    try:
        response = controller._original_env.execute_adb_call(object())
    finally:
        for controller_type, original in patches:
            controller_type.execute_adb_call = original

    assert response.generic.output == (
        b"com.google.androidenv.accessibilityforwarder/Service\n"
    )


def test_androidworld_current_activity_recovers_from_dumpsys() -> None:
    calls: list[tuple[object, object, object]] = []

    def original(_controller, *, timeout_sec=None):
        return "com.google.android.deskclock", SimpleNamespace()

    def issue_generic_request(command, controller, *, timeout_sec=None):
        calls.append((tuple(command), controller, timeout_sec))
        return SimpleNamespace(
            generic=SimpleNamespace(
                output=(
                    b"mResumedActivity: ActivityRecord{u0 "
                    b"com.google.android.deskclock/com.android.deskclock.DeskClock}\n"
                )
            )
        )

    adb_utils = SimpleNamespace(
        get_current_activity=original,
        issue_generic_request=issue_generic_request,
    )
    controller = object()
    patched = _patch_androidworld_current_activity(adb_utils)
    try:
        activity, _ = adb_utils.get_current_activity(controller, timeout_sec=3.0)
    finally:
        adb_utils.get_current_activity = patched

    assert activity == "com.google.android.deskclock/com.android.deskclock.DeskClock"
    assert calls == [
        (("shell", "dumpsys", "activity", "activities"), controller, 3.0)
    ]


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


def test_androidworld_setup_ignores_unknown_optional_permission() -> None:
    class Response:
        def __init__(self, status: int, output: bytes) -> None:
            self.status = status
            self.generic = SimpleNamespace(output=output)

        def CopyFrom(self, other) -> None:
            self.status = other.status
            self.generic = SimpleNamespace(output=other.generic.output)

    class AdbUtils:
        def issue_generic_request(self, args, _env, *, timeout_sec=None):
            return Response(
                2,
                b"java.lang.IllegalArgumentException: Unknown permission: "
                b"android.permission.POST_NOTIFICATIONS",
            )

    setup_module = SimpleNamespace(adb_utils=AdbUtils())
    original = _patch_androidworld_chcon_compat(setup_module)
    assert original is not None
    try:
        response = setup_module.adb_utils.issue_generic_request(
            [
                "shell",
                "pm",
                "grant",
                "com.dimowner.audiorecorder",
                "android.permission.POST_NOTIFICATIONS",
            ],
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


def test_expense_setup_timeout_only_expands_official_next_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, float]] = []

    class Controller:
        def click_resource_id(self, resource_ids, timeout_sec=10.0):
            calls.append((resource_ids, float(timeout_sec)))

    module = SimpleNamespace(AndroidToolController=Controller)
    real_import_module = importlib.import_module
    monkeypatch.setattr(
        "src.integrations.android_world.run_episode.importlib.import_module",
        lambda name: module
        if name == "android_world.env.tools"
        else real_import_module(name),
    )

    patch = _patch_androidworld_expense_setup_timeout()
    assert patch is not None
    controller_type, original = patch
    try:
        controller_type.click_resource_id(
            Controller(), "com.arduia.expense:id/btn_continue", timeout_sec=5.0
        )
        controller_type.click_resource_id(Controller(), "other:id/button")
    finally:
        controller_type.click_resource_id = original

    assert calls == [
        ("com.arduia.expense:id/btn_continue", 30.0),
        ("other:id/button", 10.0),
    ]


def test_chrome_setup_skips_absent_onboarding_ids_after_first_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, float]] = []

    class Env:
        foreground_activity_name = "com.android.chrome/.Main"

        def get_ui_elements(self):
            return [
                SimpleNamespace(
                    package_name="com.android.chrome",
                    text="Chrome",
                    content_description="",
                )
            ]

    class Controller:
        _env = Env()

        def click_resource_id(self, resource_ids, timeout_sec=10.0):
            calls.append((resource_ids, float(timeout_sec)))
            raise ValueError("Target resource ID not found")

    module = SimpleNamespace(AndroidToolController=Controller)
    real_import_module = importlib.import_module
    monkeypatch.setattr(
        "src.integrations.android_world.run_episode.importlib.import_module",
        lambda name: module
        if name == "android_world.env.tools"
        else real_import_module(name),
    )

    patch = _patch_androidworld_expense_setup_timeout()
    assert patch is not None
    controller_type, original = patch
    try:
        assert (
            controller_type.click_resource_id(
                Controller(),
                (
                    "com.android.chrome:id/signin_fre_dismiss_button",
                    "com.android.chrome:id/terms_accept",
                ),
            )
            is None
        )
    finally:
        controller_type.click_resource_id = original

    assert calls


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
        "src.integrations.android_world.run_episode.signal.getsignal",
        lambda _signal_number: "previous-handler",
    )
    monkeypatch.setattr(
        "src.integrations.android_world.run_episode.signal.signal", fake_signal
    )
    monkeypatch.setattr(
        "src.integrations.android_world.run_episode.signal.setitimer", fake_setitimer
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
        "src.integrations.android_world.run_episode.importlib.import_module",
        lambda name: Tools()
        if name == "android_world.env.tools"
        else real_import_module(name),
    )
    monkeypatch.setattr(
        "src.integrations.android_world.run_episode.time.sleep", lambda _seconds: None
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


def test_androidworld_browser_setup_clears_file_picker_filters() -> None:
    calls: list[tuple[str, str]] = []

    class ChromeApp:
        app_name = "chrome"

    class DocumentsApp:
        app_name = "documents"

    class AdbUtils:
        def clear_app_data(self, package_name, _controller):
            calls.append(("clear", package_name))

    _reset_androidworld_file_picker_state(
        SimpleNamespace(controller=object()),
        setup_module=SimpleNamespace(adb_utils=AdbUtils()),
        setup_apps=(ChromeApp,),
    )
    _reset_androidworld_file_picker_state(
        SimpleNamespace(controller=object()),
        setup_module=SimpleNamespace(adb_utils=AdbUtils()),
        setup_apps=(DocumentsApp,),
    )

    assert calls == [("clear", "com.google.android.documentsui")]


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
        "src.integrations.android_world.run_episode.importlib.import_module",
        lambda name: SimpleNamespace(
            find_and_click_element_by_resource_id=click_resource_id
        )
        if name == "android_world.env.actuation"
        else real_import_module(name),
    )
    monkeypatch.setattr(
        "src.integrations.android_world.run_episode.time.sleep", lambda _: None
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


def test_androidworld_episode_setup_resolves_contacts_chooser(monkeypatch) -> None:
    calls: list[object] = []

    class Controller:
        screen = "chooser"

        def get_ui_elements(self):
            labels_by_screen = {
                "chooser": (
                    "Open with Contacts",
                    "Just once",
                    "Always",
                    "Use a different app",
                    "Omnibot",
                ),
                "permission": ("Allow Contacts to send you notifications?",),
                "contacts": ("Contacts",),
            }
            package = (
                "com.google.android.permissioncontroller"
                if self.screen == "permission"
                else "com.google.android.contacts"
                if self.screen == "contacts"
                else "android"
            )
            return [
                SimpleNamespace(
                    text=label,
                    content_description=None,
                    package_name=package,
                )
                for label in labels_by_screen[self.screen]
            ]

    controller = Controller()
    env = SimpleNamespace(controller=controller)

    class AndroidToolController:
        def __init__(self, raw_env) -> None:
            assert raw_env is controller
            self._env = raw_env

        def click_element(self, label: str) -> None:
            calls.append(("click_element", label))
            if label == "Just once" and controller.screen == "chooser":
                controller.screen = "permission"
                return
            raise AssertionError((controller.screen, label))

    def click_permission(_resource_ids, _controller, timeout_sec=10.0) -> None:
        calls.append(("click_permission", timeout_sec))
        controller.screen = "contacts"

    actuation = SimpleNamespace(
        find_and_click_element_by_resource_id=click_permission
    )
    tools = SimpleNamespace(AndroidToolController=AndroidToolController)
    monkeypatch.setattr(
        "src.integrations.android_world.run_episode.time.sleep", lambda _seconds: None
    )
    monkeypatch.setattr(
        "src.integrations.android_world.run_episode.importlib.import_module",
        lambda name: tools
        if name == "android_world.env.tools"
        else actuation
        if name == "android_world.env.actuation"
        else pytest.fail(f"unexpected import: {name}"),
    )

    class ContactsApp:
        app_name = "contacts"

        @classmethod
        def package_name(cls) -> str:
            return "com.google.android.contacts"

    setup_module = SimpleNamespace(
        adb_utils=SimpleNamespace(
            launch_app=lambda app, _controller: calls.append(("launch", app)),
            close_app=lambda app, _controller: calls.append(("close", app)),
        ),
        app_snapshot=SimpleNamespace(save_snapshot=lambda *_args: None),
    )

    _prepare_androidworld_episode_apps(
        env,
        setup_module=setup_module,
        setup_apps=(ContactsApp,),
    )

    assert calls == [
        ("launch", "contacts"),
        ("click_element", "Just once"),
        ("click_permission", 10.0),
        ("close", "contacts"),
    ]
    assert controller.screen == "contacts"


def test_androidworld_episode_setup_finishes_osmand_before_snapshot(
    monkeypatch,
) -> None:
    calls: list[object] = []

    class Controller:
        def get_ui_elements(self):
            return [
                SimpleNamespace(
                    text="SKIP DOWNLOAD",
                    content_description=None,
                    package_name="net.osmand",
                    resource_name="net.osmand:id/skip_button",
                )
            ]

    controller = Controller()
    env = SimpleNamespace(controller=controller)

    class OsmAndApp:
        app_name = "osmand"

        @classmethod
        def package_name(cls) -> str:
            return "net.osmand"

    class ToolController:
        def __init__(self, _controller) -> None:
            pass

        def click_element(self, label: str) -> None:
            calls.append(("click", label))

    setup_module = SimpleNamespace(
        adb_utils=SimpleNamespace(
            launch_app=lambda app, _controller: calls.append(("launch", app)),
            close_app=lambda app, _controller: calls.append(("close", app)),
        ),
        app_snapshot=SimpleNamespace(
            save_snapshot=lambda app, _controller: calls.append(("snapshot", app))
        ),
    )
    real_import_module = __import__("importlib").import_module

    def import_module(name: str):
        if name == "android_world.env.actuation":
            return SimpleNamespace()
        if name == "android_world.env.tools":
            return SimpleNamespace(AndroidToolController=ToolController)
        if name == "android_world.utils.file_utils":
            return SimpleNamespace(
                check_file_exists=lambda path, actual_controller: calls.append(
                    ("file_exists", path, actual_controller)
                )
                or True
            )
        return real_import_module(name)

    monkeypatch.setattr(
        "src.integrations.android_world.run_episode.importlib.import_module",
        import_module,
    )
    monkeypatch.setattr(
        "src.integrations.android_world.run_episode.time.sleep", lambda _: None
    )

    _prepare_androidworld_episode_apps(
        env,
        setup_module=setup_module,
        setup_apps=(OsmAndApp,),
        save_snapshots=True,
    )

    assert calls == [
        ("launch", "osmand"),
        ("click", "SKIP DOWNLOAD"),
        (
            "file_exists",
            "/data/data/net.osmand/databases/map_markers_db",
            controller,
        ),
        ("close", "osmand"),
        ("snapshot", "osmand"),
    ]


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
    monkeypatch.setattr("src.integrations.android_world.run_episode.time.sleep", lambda _: None)

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
        "src.integrations.android_world.run_episode.importlib.import_module",
        lambda name: SimpleNamespace(
            find_and_click_element_by_resource_id=click_resource_id
        )
        if name == "android_world.env.actuation"
        else real_import_module(name),
    )
    monkeypatch.setattr("src.integrations.android_world.run_episode.time.sleep", lambda _: None)

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

    monkeypatch.setattr("src.integrations.android_world.run_episode.time.sleep", lambda _: None)
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

    monkeypatch.setattr("src.integrations.android_world.run_episode.time.sleep", lambda _: None)
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

    monkeypatch.setattr("src.integrations.android_world.run_episode.time.sleep", lambda _: None)

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

    monkeypatch.setattr("src.integrations.android_world.run_episode.time.sleep", lambda _: None)

    _wait_for_androidworld_a11y(Env())

    assert calls == ["state", "restart", "state"]


def test_androidworld_a11y_readiness_rejects_empty_forests(monkeypatch) -> None:
    forests = iter(({}, [], {"window": object()}))
    monkeypatch.setattr("src.integrations.android_world.run_episode.time.sleep", lambda _: None)

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
        "src.integrations.android_world.run_episode.subprocess.run",
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
        "src.integrations.android_world.run_episode.subprocess.run",
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
        "src.integrations.android_world.run_episode._androidworld_a11y_forwarder_installed",
        lambda **_kwargs: next(installed),
    )
    monkeypatch.setattr(
        "src.integrations.android_world.run_episode.ANDROIDWORLD_A11Y_FORWARDER_SHA256",
        hashlib.sha256(apk.read_bytes()).hexdigest(),
    )
    calls = []
    monkeypatch.setattr(
        "src.integrations.android_world.run_episode.subprocess.run",
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


def test_fold_uses_logical_display_for_coordinate_conversion(monkeypatch) -> None:
    class JSONAction:
        def __init__(self, action_type=None, x=None, y=None):
            self.action_type = action_type
            self.x = x
            self.y = y

    module = SimpleNamespace(JSONAction=JSONAction)
    monkeypatch.setattr(
        "src.integrations.android_world.host.importlib.import_module",
        lambda name: module if name == "android_world.env.json_action" else None,
    )
    env = SimpleNamespace(
        # Fold physical size is rotated/different from the application
        # display used by the accessibility tree and screenshots.
        device_screen_size=(1768, 2208),
        logical_screen_size=(2208, 1840),
    )

    action = AndroidWorldHost(env)._json_action(
        Action("click", {"x": 1000.0, "y": 1000.0})
    )

    assert action.x == 2208.0
    assert action.y == 1840.0


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
    assert set(saved) == {"screenshot", "xml"}
    assert 'text="Settings"' in saved["xml"]
    screenshot_path = Path(saved["screenshot"]["path"])
    assert screenshot_path.is_file()
    assert screenshot_path.is_absolute()
    assert "sha256" not in saved["screenshot"]
    assert saved["screenshot"]["width"] == 4
    assert saved["screenshot"]["height"] == 3
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


def test_observe_treats_complete_application_modal_as_transfer_graph() -> None:
    modal_xml = """\
<hierarchy>
  <node package="org.videolan.vlc" bounds="[18,377][702,902]">
    <node package="org.videolan.vlc" resource-id="android:id/parentPanel"
          bounds="[50,409][670,870]">
      <node package="org.videolan.vlc" resource-id="android:id/alertTitle"
            class="android.widget.TextView" text="Allow VLC all file access"
            bounds="[178,452][622,501]" />
      <node package="org.videolan.vlc" resource-id="android:id/buttonPanel"
            bounds="[50,758][670,870]">
        <node package="org.videolan.vlc" resource-id="android:id/button1"
              class="android.widget.Button" text="OK" clickable="true"
              bounds="[518,766][646,862]" />
      </node>
    </node>
  </node>
</hierarchy>
"""

    class Env:
        device_screen_size = (720, 1280)
        logical_screen_size = (720, 1280)
        foreground_activity_name = "org.videolan.vlc/.gui.MainActivity"

        def get_state(self, wait_to_stabilize: bool = False):
            assert wait_to_stabilize is True
            return _official_state(forest=modal_xml)

    observation = AndroidWorldHost(Env()).observe()

    assert observation.extra["ui_graph_complete"] is True
    assert observation.extra["ui_graph_source"] == "androidworld_state_forest"


def test_observe_treats_custom_application_modal_with_ime_as_complete_graph() -> None:
    modal_xml = """\
<hierarchy>
  <node package="com.dimowner.audiorecorder" bounds="[18,94][702,636]">
    <node package="com.dimowner.audiorecorder"
          resource-id="android:id/parentPanel" bounds="[50,126][670,604]">
      <node package="com.dimowner.audiorecorder"
            resource-id="android:id/customPanel" bounds="[50,126][670,604]">
        <node package="com.dimowner.audiorecorder"
              resource-id="com.dimowner.audiorecorder:id/input_name"
              class="android.widget.EditText" text="G367_conference"
              bounds="[90,257][630,348]" />
        <node package="com.dimowner.audiorecorder"
              resource-id="com.dimowner.audiorecorder:id/dialog_positive_btn"
              class="android.widget.Button" text="Save" clickable="true"
              bounds="[438,452][614,548]" />
      </node>
    </node>
  </node>
  <node package="com.google.android.inputmethod.latin"
        resource-id="android:id/inputArea" bounds="[0,48][720,1184]" />
</hierarchy>
"""

    class Env:
        device_screen_size = (720, 1280)
        logical_screen_size = (720, 1280)
        foreground_activity_name = (
            "com.dimowner.audiorecorder/.app.main.MainActivity"
        )

        def get_state(self, wait_to_stabilize: bool = False):
            assert wait_to_stabilize is True
            return _official_state(forest=modal_xml)

    observation = AndroidWorldHost(Env()).observe()

    assert observation.extra["ui_graph_complete"] is True
    assert observation.extra["ui_graph_source"] == "androidworld_state_forest"
    assert "G367_conference" in str(observation.xml)
    assert "Save" in str(observation.xml)


def test_observe_can_disable_stabilization_for_diagnostic_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMNIFLOW_ANDROIDWORLD_WAIT_TO_STABILIZE", "false")
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
    assert calls == [False]


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


def test_observe_keeps_diagnostic_observation_without_xml() -> None:
    incomplete = SimpleNamespace(pixels=None, forest=None, ui_elements=[])

    observation = AndroidWorldHost(
        SimpleNamespace(get_state=lambda **_: incomplete)
    ).observe()
    assert set(observation.extra["androidworld_state"]) == {
        "pixels",
        "forest",
        "ui_elements",
        "auxiliaries",
    }


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


def test_observe_prefers_complete_ui_elements_over_partial_richer_forest() -> None:
    def forest_node(unique_id, bounds, *, child_ids=(), text=""):
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
            is_clickable=False,
            is_editable=False,
            is_scrollable=False,
            is_visible_to_user=True,
        )

    forest = SimpleNamespace(
        windows=[
            SimpleNamespace(
                id=1,
                title="Settings",
                tree=SimpleNamespace(
                    nodes=[
                        forest_node(1, (0, 0, 465, 800), child_ids=(2, 3, 4)),
                        forest_node(2, (32, 80, 400, 140), text="Network & internet"),
                        forest_node(3, (32, 160, 400, 220), text="Connected devices"),
                        forest_node(4, (32, 240, 400, 300), text="Apps"),
                    ]
                ),
            )
        ]
    )
    complete_elements = [
        SimpleNamespace(
            text="",
            package_name="com.android.settings",
            class_name="android.widget.FrameLayout",
            bbox_pixels=SimpleNamespace(x_min=0, y_min=0, x_max=1280, y_max=800),
        ),
        SimpleNamespace(
            text="Network & internet",
            package_name="com.android.settings",
            class_name="android.widget.TextView",
            bbox_pixels=SimpleNamespace(x_min=700, y_min=80, x_max=1100, y_max=140),
        ),
    ]
    env = SimpleNamespace(
        get_state=lambda **_: _official_state(
            forest=forest,
            ui_elements=complete_elements,
        ),
        device_screen_size=(1280, 800),
        logical_screen_size=(1280, 800),
        foreground_activity_name="com.android.settings/.Settings",
    )

    observation = AndroidWorldHost(env).observe()

    assert observation.extra["ui_graph_source"] == "androidworld_state_ui_elements"
    assert observation.extra["ui_graph_complete"] is True
    assert "Network &amp; internet" in str(observation.xml)


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
        "src.integrations.android_world.host.resolve_androidworld_app_name",
        lambda package_name, controller: (
            "settings" if package_name == "com.android.settings" else package_name
        ),
    )
    monkeypatch.setattr(
        "src.integrations.android_world.host.importlib.import_module",
        lambda name: module if name == "android_world.env.json_action" else None,
    )
    actions: list[JSONAction] = []
    env = SimpleNamespace(
        device_screen_size=(720, 1280),
        execute_action=actions.append,
    )
    host = AndroidWorldHost(env)

    click_result = host.act(Action("click", {"x": 500, "y": 250}))
    open_result = host.act(
        Action("open_app", {"package_name": "com.android.settings"})
    )

    assert click_result.success is True
    assert open_result.success is True
    assert [(action.action_type, action.x, action.y, action.app_name) for action in actions] == [
        ("click", 360.0, 320.0, None),
        ("open_app", None, None, "settings"),
    ]


def test_androidworld_host_exposes_system_settings_as_installed(monkeypatch) -> None:
    setup = SimpleNamespace(
        get_installed_packages=lambda _env: frozenset({"com.example.app"})
    )
    monkeypatch.setattr(
        "src.integrations.android_world.host.importlib.import_module",
        lambda name: (
            setup
            if name == "android_world.env.setup_device.setup"
            else None
        ),
    )

    packages = AndroidWorldHost(SimpleNamespace()).installed_packages()

    assert packages == {"com.example.app", "com.android.settings"}


def test_androidworld_host_exposes_only_launchable_apps_to_planner(
    monkeypatch,
) -> None:
    host = AndroidWorldHost(SimpleNamespace(controller=object()))
    monkeypatch.setattr(
        host,
        "installed_packages",
        lambda: {
            "com.android.settings",
            "com.google.android.documentsui",
            "com.google.android.documentsui.overlay",
            "cn.com.omnimind.bot.debug",
        },
    )
    monkeypatch.setattr(
        host,
        "_launchable_packages",
        lambda: {
            "com.android.settings",
            "com.google.android.documentsui",
            "cn.com.omnimind.bot.debug",
        },
    )
    monkeypatch.setattr(
        "src.integrations.android_world.host.launchable_androidworld_apps",
        lambda packages, _controller: (
            {"Settings": "com.android.settings"}
            if "com.android.settings" in packages
            else {}
        ),
    )

    assert host.installed_apps() == {
        "Documentsui": "com.google.android.documentsui",
        "Settings": "com.android.settings",
    }


def test_androidworld_host_queries_launcher_packages_from_package_manager(
    monkeypatch,
) -> None:
    host_module = sys.modules["src.integrations.android_world.host"]
    commands: list[list[str]] = []

    def run(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "2 activities found:\n"
                "  com.google.android.documentsui/com.android.documentsui.LauncherActivity\n"
                "com.android.settings/.Settings\n"
                "not-a-component\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(host_module.subprocess, "run", run)
    host = AndroidWorldHost(
        SimpleNamespace(),
        adb_serial="emulator-5560",
        adb_path="/tmp/androidworld-adb",
    )

    assert host._launchable_packages() == {
        "com.android.settings",
        "com.google.android.documentsui",
    }
    assert commands == [
        [
            "/tmp/androidworld-adb",
            "-s",
            "emulator-5560",
            "shell",
            "cmd",
            "package",
            "query-activities",
            "--brief",
            "-a",
            "android.intent.action.MAIN",
            "-c",
            "android.intent.category.LAUNCHER",
        ]
    ]


def test_androidworld_host_falls_back_to_adb_when_official_package_list_is_empty(
    monkeypatch,
) -> None:
    host_module = sys.modules["src.integrations.android_world.host"]
    setup = SimpleNamespace(get_installed_packages=lambda _env: frozenset())
    monkeypatch.setattr(
        host_module.importlib,
        "import_module",
        lambda name: (
            setup
            if name == "android_world.env.setup_device.setup"
            else None
        ),
    )
    commands: list[list[str]] = []

    def run(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="package:org.videolan.vlc\npackage:org.tasks\n",
            stderr="",
        )

    monkeypatch.setattr(host_module.subprocess, "run", run)

    packages = AndroidWorldHost(
        SimpleNamespace(),
        adb_serial="emulator-5560",
        adb_path="/tmp/androidworld-adb",
    ).installed_packages()

    assert commands == [
        [
            "/tmp/androidworld-adb",
            "-s",
            "emulator-5560",
            "shell",
            "pm",
            "list",
            "packages",
        ]
    ]
    assert packages == {
        "com.android.settings",
        "org.tasks",
        "org.videolan.vlc",
    }


def test_oob_open_app_resolves_androidworld_launcher_name(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class ControlClient:
        def act(self, payload: dict[str, object]) -> dict[str, object]:
            calls.append(payload)
            return {"success": True}

    monkeypatch.setattr(
        "src.integrations.android_world.host.resolve_androidworld_package",
        lambda identifier: (
            "com.dimowner.audiorecorder"
            if identifier == "audio recorder"
            else ""
        ),
    )
    host = AndroidWorldHost(
        SimpleNamespace(device_screen_size=(720, 1280)),
        control_backend="oob",
    )
    host.control_client = ControlClient()

    result = host.act(Action("open_app", {"package_name": "audio recorder"}))

    assert result.success is True
    assert calls == [
        {
            "tool": "open_app",
            "args": {"package_name": "com.dimowner.audiorecorder"},
        }
    ]


def test_androidworld_app_launch_restarts_mapped_app_before_opening() -> None:
    calls: list[tuple[str, str]] = []
    controller = object()
    adb_utils = SimpleNamespace(
        get_adb_activity=lambda app_name: (
            "com.android.settings/.Settings" if app_name == "settings" else None
        ),
        close_app=lambda app_name, actual_controller: calls.append(
            ("close", app_name)
        )
        if actual_controller is controller
        else pytest.fail("unexpected controller"),
        launch_app=lambda app_name, actual_controller: calls.append(
            ("launch", app_name)
        )
        if actual_controller is controller
        else pytest.fail("unexpected controller"),
    )

    original = _patch_androidworld_app_launch(adb_utils)
    try:
        adb_utils.launch_app("settings", controller)
        adb_utils.launch_app("com.example.app", controller)
    finally:
        adb_utils.launch_app = original

    assert calls == [
        ("close", "settings"),
        ("launch", "settings"),
        ("launch", "com.example.app"),
    ]


def test_androidworld_camera_launch_falls_back_to_public_capture_intent() -> None:
    calls: list[tuple[str, object]] = []
    controller = object()

    def launch_app(app_name: str, actual_controller: object) -> None:
        assert actual_controller is controller
        calls.append(("launch", app_name))
        raise RuntimeError("legacy_camera_component_missing")

    adb_utils = SimpleNamespace(
        get_adb_activity=lambda app_name: (
            "com.android.camera2/com.android.camera.CameraLauncher"
            if app_name == "camera"
            else None
        ),
        close_app=lambda app_name, actual_controller: calls.append(
            ("close", app_name)
        ),
        launch_app=launch_app,
        issue_generic_request=lambda command, actual_controller: calls.append(
            ("intent", tuple(command))
        ) or SimpleNamespace(ok=True),
        check_ok=lambda response, message: calls.append(("check", message)),
    )

    original = _patch_androidworld_app_launch(adb_utils)
    try:
        adb_utils.launch_app("camera", controller)
    finally:
        adb_utils.launch_app = original

    assert calls == [
        ("close", "camera"),
        ("launch", "camera"),
        (
            "intent",
            ("shell", "am", "start", "-a", "android.media.action.IMAGE_CAPTURE"),
        ),
        ("check", "Failed to launch the AndroidWorld Camera2 capture intent."),
    ]


def test_androidworld_clipboard_read_dismisses_legacy_dialog_from_live_xml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    controller = object()
    attempts = 0
    dialog_xml = b"""<?xml version='1.0' encoding='UTF-8'?>
<hierarchy>
  <node text="This app was built for an older version of Android and may not work properly." package="android" bounds="[40,300][680,700]" />
  <node text="OK" resource-id="android:id/button1" package="android" clickable="true" bounds="[520,730][680,810]" />
</hierarchy>
UI hierchary dumped to: /dev/tty
"""

    def get_clipboard_contents(actual_controller: object) -> str:
        nonlocal attempts
        assert actual_controller is controller
        attempts += 1
        if attempts == 1:
            raise RuntimeError(
                "Clipper app must be in the foreground to access clipboard. "
                "Additionally, app privileges must be granted manually."
            )
        return "1234 Elm St, Springfield, IL"

    def issue_generic_request(
        command: list[str], actual_controller: object
    ) -> SimpleNamespace:
        assert actual_controller is controller
        calls.append(("adb", tuple(command)))
        output = dialog_xml if "cat" in command else b""
        return SimpleNamespace(generic=SimpleNamespace(output=output))

    monkeypatch.setattr(
        "src.integrations.android_world.run_episode.time.sleep", lambda _: None
    )
    adb_utils = SimpleNamespace(
        get_clipboard_contents=get_clipboard_contents,
        issue_generic_request=issue_generic_request,
    )

    original = _patch_androidworld_clipboard_read_compat(adb_utils)
    try:
        result = adb_utils.get_clipboard_contents(controller)
    finally:
        adb_utils.get_clipboard_contents = original

    assert result == "1234 Elm St, Springfield, IL"
    assert attempts == 2
    assert calls == [
        (
            "adb",
            (
                "shell",
                "uiautomator",
                "dump",
                "/data/local/tmp/omniflow_clipboard_validator.xml",
            ),
        ),
        (
            "adb",
            (
                "shell",
                "cat",
                "/data/local/tmp/omniflow_clipboard_validator.xml",
            ),
        ),
        (
            "adb",
            (
                "shell",
                "rm",
                "-f",
                "/data/local/tmp/omniflow_clipboard_validator.xml",
            ),
        ),
        ("adb", ("shell", "input", "tap", "600", "770")),
    ]


def test_androidworld_clipboard_read_preserves_unrelated_failure() -> None:
    controller = object()
    adb_utils = SimpleNamespace(
        get_clipboard_contents=lambda actual_controller: (_ for _ in ()).throw(
            RuntimeError("Failed to get clipboard content.")
        ),
        issue_generic_request=lambda *_args, **_kwargs: pytest.fail(
            "unrelated failures must not inspect or modify the UI"
        ),
    )

    original = _patch_androidworld_clipboard_read_compat(adb_utils)
    try:
        with pytest.raises(RuntimeError, match="Failed to get clipboard content"):
            adb_utils.get_clipboard_contents(controller)
    finally:
        adb_utils.get_clipboard_contents = original
