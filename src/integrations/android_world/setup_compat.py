from __future__ import annotations

import json
import logging
import time
from typing import Any

_EQUIVALENT_SETUP_LABELS: dict[str, tuple[str, ...]] = {
    "Accept & continue": ("Use without an account",),
    "Don't allow": ("Don’t allow",),
    "No thanks": ("Keep Google",),
    "Skip": ("SKIP",),
}
_POST_INITIALIZE_SNAPSHOT_APPS = frozenset({"chrome"})
_CONTACTS_APP_NAME = "contacts"
_CONTACTS_PACKAGE = "com.google.android.contacts"
_POST_NOTIFICATIONS_PERMISSION = "android.permission.POST_NOTIFICATIONS"
_OPEN_TRACKS_ACTIVITY_NAME = "open tracks"
_OPEN_TRACKS_COMPANION_APP_NAME = "activity tracker"
_OPEN_TRACKS_BLUETOOTH_PERMISSIONS = (
    "android.permission.BLUETOOTH_SCAN",
    "android.permission.BLUETOOTH_CONNECT",
)
_VLC_APK_BY_ABI = {
    "arm64-v8a": "org.videolan.vlc_13050407.apk",
    "x86_64": "org.videolan.vlc_13050408.apk",
}
_CONTACTS_READY_LABELS = frozenset(
    {
        "Create contact",
        "Fix & manage",
        "Search contacts",
    }
)


def _deny_contacts_notification_permission(setup_module: Any, env: Any) -> None:
    adb_utils = setup_module.adb_utils
    commands = (
        [
            "shell",
            "pm",
            "revoke",
            _CONTACTS_PACKAGE,
            _POST_NOTIFICATIONS_PERMISSION,
        ],
        [
            "shell",
            "pm",
            "set-permission-flags",
            _CONTACTS_PACKAGE,
            _POST_NOTIFICATIONS_PERMISSION,
            "user-set",
            "user-fixed",
        ],
    )
    for command in commands:
        response = adb_utils.issue_generic_request(command, env.controller)
        adb_utils.check_ok(
            response,
            "Failed to fix the Contacts notification permission state.",
        )
    verification = adb_utils.issue_generic_request(
        ["shell", "dumpsys", "package", _CONTACTS_PACKAGE],
        env.controller,
    )
    adb_utils.check_ok(
        verification,
        "Failed to verify the Contacts notification permission state.",
    )
    output = bytes(getattr(verification.generic, "output", b"")).decode(
        "utf-8", errors="replace"
    )
    permission_lines = [
        line
        for line in output.splitlines()
        if f"{_POST_NOTIFICATIONS_PERMISSION}:" in line
    ]
    if not any(
        "granted=false" in line
        and "USER_SET" in line
        and "USER_FIXED" in line
        for line in permission_lines
    ):
        raise RuntimeError(
            "Contacts notification permission is not denied and user-fixed: "
            + repr(permission_lines)
        )


def _visible_setup_elements(controller: Any) -> list[dict[str, str]]:
    try:
        elements = tuple(controller._env.get_ui_elements() or ())
    except Exception:  # noqa: BLE001 - diagnostics must not mask setup retries
        return []
    return [
        {
            "package": str(getattr(element, "package_name", "") or ""),
            "text": str(getattr(element, "text", "") or ""),
            "content_description": str(
                getattr(element, "content_description", "") or ""
            ),
        }
        for element in elements
    ]


def _visible_setup_strings(controller: Any) -> set[str]:
    return {
        value.strip()
        for element in _visible_setup_elements(controller)
        for value in (element["text"], element["content_description"])
        if value.strip()
    }


def _setup_click_is_already_complete(
    visible: set[str],
    target_text: str,
) -> bool:
    if target_text == "NEXT" and "Shutter" in visible:
        return True
    if "Search or type web address" in visible:
        return target_text in {"Accept & continue", "No thanks"}
    if target_text == "Accept & continue":
        return "No thanks" in visible
    if target_text == "Don't allow":
        return bool(_CONTACTS_READY_LABELS.intersection(visible))
    return False


def _grant_manage_external_storage(apps_module: Any, app_name: str, env: Any) -> None:
    package_name = apps_module.adb_utils.extract_package_name(
        apps_module.adb_utils.get_adb_activity(app_name)
    )
    response = apps_module.adb_utils.issue_generic_request(
        [
            "shell",
            "appops",
            "set",
            package_name,
            "MANAGE_EXTERNAL_STORAGE",
            "allow",
        ],
        env.controller,
    )
    apps_module.adb_utils.check_ok(
        response,
        f"Failed to grant MANAGE_EXTERNAL_STORAGE to {package_name}.",
    )
    verification = apps_module.adb_utils.issue_generic_request(
        [
            "shell",
            "appops",
            "get",
            package_name,
            "MANAGE_EXTERNAL_STORAGE",
        ],
        env.controller,
    )
    apps_module.adb_utils.check_ok(
        verification,
        f"Failed to verify MANAGE_EXTERNAL_STORAGE for {package_name}.",
    )
    output = bytes(getattr(verification.generic, "output", b"")).decode(
        "utf-8", errors="replace"
    )
    if "allow" not in output.casefold():
        raise RuntimeError(
            f"MANAGE_EXTERNAL_STORAGE is not allowed for {package_name}: {output}"
        )


def patch_androidworld_setup_fail_closed(
    setup_module: Any,
    *,
    attempts: int = 2,
    delay_seconds: float = 1.0,
) -> None:
    """Retry official app setup and save snapshots only after success."""

    if getattr(setup_module, "_omniflow_setup_fail_closed_patch", False):
        return

    def setup_app_with_retry(app: Any, env: Any) -> None:
        attempt_count = max(1, int(attempts))
        native_controller = getattr(env, "controller", None)
        original_a11y_method = getattr(native_controller, "_a11y_method", None)
        uiautomator_method = getattr(
            type(original_a11y_method),
            "UIAUTOMATOR",
            None,
        )
        for attempt in range(1, attempt_count + 1):
            if uiautomator_method is not None:
                native_controller._a11y_method = uiautomator_method
            try:
                app.setup(env)
            except ValueError as error:
                if attempt >= attempt_count:
                    raise
                logging.warning(
                    "AndroidWorld app setup failed; retrying app=%s attempt=%d/%d error=%s",
                    app.app_name,
                    attempt,
                    attempt_count,
                    error,
                )
                time.sleep(max(0.0, float(delay_seconds)))
                continue
            finally:
                if uiautomator_method is not None:
                    native_controller._a11y_method = original_a11y_method
            if str(getattr(app, "app_name", "") or "") == _CONTACTS_APP_NAME:
                _deny_contacts_notification_permission(setup_module, env)
            setup_module.app_snapshot.save_snapshot(app.app_name, env.controller)
            return

    setup_module.setup_app = setup_app_with_retry
    setup_module._omniflow_setup_fail_closed_patch = True


def patch_androidworld_legacy_apk_install(setup_module: Any) -> None:
    """Retry only Android's explicit low-target-SDK rejection with its bypass."""

    if getattr(setup_module, "_omniflow_legacy_apk_install_patch", False):
        return
    original_download_and_install = setup_module.download_and_install_apk

    def download_and_install_apk(apk_name: str, raw_env: Any) -> None:
        try:
            original_download_and_install(apk_name, raw_env)
            return
        except Exception as error:  # noqa: BLE001 - match the ADB rejection exactly
            if "INSTALL_FAILED_DEPRECATED_SDK_VERSION" not in str(error):
                raise
        apk_path = setup_module.apps.download_app_data(apk_name)
        response = setup_module.adb_utils.issue_generic_request(
            [
                "install",
                "--bypass-low-target-sdk-block",
                apk_path,
            ],
            raw_env,
            timeout_sec=30.0,
        )
        setup_module.adb_utils.check_ok(
            response,
            f"Failed to install legacy AndroidWorld APK {apk_path}.",
        )
        logging.warning(
            "Installed AndroidWorld APK with the platform low-target-SDK bypass: %s",
            apk_name,
        )

    setup_module.download_and_install_apk = download_and_install_apk
    setup_module._omniflow_legacy_apk_install_patch = True


def patch_androidworld_vlc_apk_selection(setup_module: Any) -> None:
    """Select the official VLC APK matching the target device ABI."""

    if getattr(setup_module, "_omniflow_vlc_apk_selection_patch", False):
        return
    original_maybe_install_app = setup_module.maybe_install_app
    vlc_app = setup_module.apps.VlcApp

    def maybe_install_app(app: Any, env: Any) -> None:
        if app is not vlc_app:
            original_maybe_install_app(app, env)
            return
        response = setup_module.adb_utils.issue_generic_request(
            ["shell", "getprop", "ro.product.cpu.abi"],
            env.controller,
        )
        setup_module.adb_utils.check_ok(
            response,
            "Failed to read the Android device ABI for VLC installation.",
        )
        output = bytes(getattr(response.generic, "output", b"")).decode(
            "utf-8", errors="replace"
        )
        device_abi = output.strip().splitlines()[0] if output.strip() else ""
        selected_apk = _VLC_APK_BY_ABI.get(device_abi)
        available_apks = tuple(str(name) for name in app.apk_names)
        if selected_apk is None or selected_apk not in available_apks:
            raise RuntimeError(
                "No official AndroidWorld VLC APK matches the target device ABI: "
                f"abi={device_abi!r} available={available_apks!r}"
            )
        app.apk_names = (selected_apk,)
        try:
            original_maybe_install_app(app, env)
        finally:
            app.apk_names = available_apks

    setup_module.maybe_install_app = maybe_install_app
    setup_module._omniflow_vlc_apk_selection_patch = True


def patch_androidworld_osmand_storage_setup(setup_module: Any) -> None:
    """Accept unsupported shared-storage chcon only after every map verifies."""

    if getattr(setup_module, "_omniflow_osmand_storage_patch", False):
        return
    osmand_app = setup_module.apps.OsmAndApp
    original_setup = osmand_app.setup

    def setup(cls: Any, env: Any) -> None:
        try:
            original_setup(env)
            return
        except Exception as error:  # noqa: BLE001 - exact device-filesystem error
            message = str(error)
            if not (
                "chcon" in message
                and "Operation not supported on transport endpoint" in message
            ):
                raise
        for map_name in tuple(cls.MAP_NAMES):
            map_path = cls.DEVICE_MAPS_PATH.rstrip("/") + "/" + str(map_name)
            response = setup_module.adb_utils.issue_generic_request(
                ["shell", "test", "-s", map_path],
                env.controller,
            )
            setup_module.adb_utils.check_ok(
                response,
                f"OsmAnd map is missing after unsupported chcon: {map_path}",
            )
        logging.warning(
            "Skipped unsupported OsmAnd shared-storage chcon after map verification."
        )

    osmand_app.setup = classmethod(setup)
    setup_module._omniflow_osmand_storage_patch = True


def patch_androidworld_special_storage_setup(setup_module: Any) -> None:
    """Grant and verify special storage access for current Android system UI."""

    if getattr(setup_module, "_omniflow_special_storage_patch", False):
        return
    apps_module = setup_module.apps

    gallery_app = apps_module.SimpleGalleryProApp
    original_gallery_setup = gallery_app.setup

    def setup_gallery(cls: Any, env: Any) -> None:
        _grant_manage_external_storage(apps_module, cls.app_name, env)
        try:
            original_gallery_setup(env)
        except ValueError as error:
            if 'setup target "All files" not found' not in str(error):
                raise
            _grant_manage_external_storage(apps_module, cls.app_name, env)
            logging.warning(
                "Simple Gallery special storage UI was absent; verified app-op instead."
            )

    gallery_app.setup = classmethod(setup_gallery)

    vlc_app = apps_module.VlcApp
    original_vlc_setup = vlc_app.setup

    def setup_vlc(cls: Any, env: Any) -> None:
        _grant_manage_external_storage(apps_module, cls.app_name, env)
        try:
            original_vlc_setup(env)
        except ValueError as error:
            if 'setup target "GRANT PERMISSION" not found' not in str(error):
                raise
            _grant_manage_external_storage(apps_module, cls.app_name, env)
            logging.warning(
                "VLC special storage UI was absent; verified app-op instead."
            )

    vlc_app.setup = classmethod(setup_vlc)
    setup_module._omniflow_special_storage_patch = True


def patch_androidworld_open_tracks_setup(setup_module: Any) -> None:
    """Accept an absent Bluetooth dialog only after both grants verify."""

    if getattr(setup_module, "_omniflow_open_tracks_setup_patch", False):
        return
    open_tracks_app = setup_module.apps.OpenTracksApp
    original_setup = open_tracks_app.setup

    def setup_open_tracks(cls: Any, env: Any) -> None:
        try:
            original_setup(env)
            return
        except ValueError as error:
            if 'setup target "Allow" not found' not in str(error):
                raise

        package_name = setup_module.adb_utils.extract_package_name(
            setup_module.adb_utils.get_adb_activity(_OPEN_TRACKS_ACTIVITY_NAME)
        )
        verification = setup_module.adb_utils.issue_generic_request(
            ["shell", "dumpsys", "package", package_name],
            env.controller,
        )
        setup_module.adb_utils.check_ok(
            verification,
            f"Failed to verify OpenTracks permissions for {package_name}.",
        )
        output = bytes(getattr(verification.generic, "output", b"")).decode(
            "utf-8", errors="replace"
        )
        missing_permissions = [
            permission
            for permission in _OPEN_TRACKS_BLUETOOTH_PERMISSIONS
            if not any(
                permission in line and "granted=true" in line
                for line in output.splitlines()
            )
        ]
        if missing_permissions:
            raise RuntimeError(
                "OpenTracks Bluetooth dialog was absent without verified grants: "
                + repr(missing_permissions)
            )

        setup_module.adb_utils.launch_app(
            _OPEN_TRACKS_COMPANION_APP_NAME,
            env.controller,
        )
        setup_module.adb_utils.close_app(
            _OPEN_TRACKS_COMPANION_APP_NAME,
            env.controller,
        )
        logging.warning(
            "OpenTracks Bluetooth permission UI was absent; verified runtime grants instead."
        )

    open_tracks_app.setup = classmethod(setup_open_tracks)
    setup_module._omniflow_open_tracks_setup_patch = True


def restore_task_app_snapshots_after_initialize(
    restore_snapshot: Any,
    task: Any,
    env: Any,
) -> None:
    """Restore app setup that an upstream task clears after base initialization."""

    restored: set[str] = set()
    for value in tuple(getattr(task, "app_names", ()) or ()):
        app_name = str(value or "").strip()
        if app_name not in _POST_INITIALIZE_SNAPSHOT_APPS or app_name in restored:
            continue
        restore_snapshot(app_name, env.controller)
        restored.add(app_name)


def resolve_androidworld_task_setup_apps(
    setup_module: Any,
    *,
    task_types: dict[str, Any],
    task_suite: dict[str, Any],
    selected_task_names: list[str],
) -> tuple[type, ...]:
    """Resolve setup apps from both task classes and instantiated tasks."""

    setup_apps: list[type] = []
    seen_setup_apps: set[type] = set()
    for selected_task_name in selected_task_names:
        task_sources = [task_types.get(selected_task_name)]
        task_sources.extend(tuple(task_suite.get(selected_task_name, ()) or ()))
        for task_source in task_sources:
            for app_name in tuple(getattr(task_source, "app_names", ()) or ()):
                app_setup = setup_module.get_app_mapping(str(app_name))
                if app_setup is not None and app_setup not in seen_setup_apps:
                    setup_apps.append(app_setup)
                    seen_setup_apps.add(app_setup)
    for app_setup in setup_module.get_app_list_to_setup(selected_task_names) or ():
        if app_setup not in seen_setup_apps:
            setup_apps.append(app_setup)
            seen_setup_apps.add(app_setup)
    return tuple(setup_apps)


def patch_androidworld_setup_click_retry(
    tools_module: Any,
    *,
    attempts: int = 15,
    delay_seconds: float = 1.0,
) -> None:
    """Make official AndroidWorld app setup tolerate equivalent onboarding UI.

    AndroidWorld remains the owner of app setup.  This patch only retries its
    requested text click and maps labels whose semantics changed in newer
    system-app builds (notably Chrome's current first-run screen).
    """

    if getattr(tools_module, "_omniflow_setup_click_retry_patch", False):
        return
    controller_type = tools_module.AndroidToolController
    original_click_element = controller_type.click_element

    def click_label(
        controller: Any,
        label: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if args:
            return original_click_element(controller, label, *args[1:], **kwargs)
        call_kwargs = dict(kwargs)
        call_kwargs["element_text"] = label
        return original_click_element(controller, **call_kwargs)

    def click_element_with_retry(
        controller: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        target_text = str(args[0] if args else kwargs.get("element_text") or "")
        last_error: ValueError | None = None
        for attempt in range(1, max(1, int(attempts)) + 1):
            labels = (target_text, *_EQUIVALENT_SETUP_LABELS.get(target_text, ()))
            visible = _visible_setup_strings(controller)
            if _setup_click_is_already_complete(visible, target_text):
                return None
            if (
                target_text == "Skip"
                and "Always" in visible
                and any(label.startswith("Open with ") for label in visible)
            ):
                if "Open with Contacts" not in visible and "Contacts" in visible:
                    click_label(controller, "Contacts", args, kwargs)
                    if attempt < max(1, int(attempts)):
                        time.sleep(max(0.0, float(delay_seconds)))
                        continue
                click_label(controller, "Always", args, kwargs)
                if attempt < max(1, int(attempts)):
                    time.sleep(max(0.0, float(delay_seconds)))
                    continue
            visible_casefold = {value.casefold() for value in visible}
            for label in (
                candidate
                for candidate in labels
                if candidate.casefold() in visible_casefold
            ):
                try:
                    return click_label(controller, label, args, kwargs)
                except ValueError as error:
                    message = str(error or "")
                    if not (
                        ("Target text" in message and "not found" in message)
                        or "Invalid element index" in message
                    ):
                        raise
                    last_error = error

            native_controller = getattr(controller, "_env", None)
            original_method = getattr(native_controller, "_a11y_method", None)
            uiautomator_method = getattr(
                type(original_method),
                "UIAUTOMATOR",
                None,
            )
            if (
                native_controller is not None
                and uiautomator_method is not None
                and original_method is not uiautomator_method
            ):
                native_controller._a11y_method = uiautomator_method
                try:
                    native_visible = _visible_setup_strings(controller)
                    if _setup_click_is_already_complete(native_visible, target_text):
                        return None
                    native_visible_casefold = {
                        value.casefold() for value in native_visible
                    }
                    for label in (
                        candidate
                        for candidate in labels
                        if candidate.casefold() in native_visible_casefold
                    ):
                        try:
                            return click_label(controller, label, args, kwargs)
                        except ValueError as error:
                            message = str(error or "")
                            if not (
                                ("Target text" in message and "not found" in message)
                                or "Invalid element index" in message
                            ):
                                raise
                            last_error = error
                finally:
                    native_controller._a11y_method = original_method
            if attempt < max(1, int(attempts)):
                time.sleep(max(0.0, float(delay_seconds)))

        details = json.dumps(
            _visible_setup_elements(controller)[:80],
            ensure_ascii=False,
        )
        if last_error is None:
            raise ValueError(
                f'AndroidWorld setup target "{target_text}" not found; '
                f"visible_setup_elements={details}"
            )
        raise ValueError(
            f"{last_error}; visible_setup_elements={details}"
        ) from last_error

    controller_type.click_element = click_element_with_retry
    tools_module._omniflow_setup_click_retry_patch = True
