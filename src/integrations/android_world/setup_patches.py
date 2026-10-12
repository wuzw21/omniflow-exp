"""AndroidWorld task-setup compatibility patches (split out of run_episode).

Moved verbatim from ``src.integrations.android_world.run_episode``; that module re-exports these names.
"""

from __future__ import annotations

import logging
import importlib
import re
import time
from typing import Any, Sequence
from src.integrations.android_world.episode_utils import (
    _app_chooser_clicks,
    _normalize_androidworld_setup_label,
    logger,
)

logger = logging.getLogger("src.integrations.android_world.run_episode")


def _androidworld_runtime_permissions_granted(
    controller: Any,
    *,
    package_name: str,
    permissions: Sequence[str],
) -> bool:
    """Return whether Android already settled an optional setup prompt."""

    try:
        adb_utils = importlib.import_module("android_world.env.adb_utils")
        env = getattr(controller, "_env", None)
        if env is None:
            return False
        for permission in permissions:
            try:
                response = adb_utils.issue_generic_request(
                    [
                        "shell",
                        "pm",
                        "check-permission",
                        package_name,
                        permission,
                    ],
                    env,
                )
                output = getattr(getattr(response, "generic", None), "output", b"")
                if isinstance(output, bytes):
                    output = output.decode("utf-8", errors="replace")
                normalized = str(output or "").strip().casefold()
                if normalized == "granted":
                    continue
            except Exception as error:  # pragma: no cover - device-specific ADB shim
                logger.info(
                    "AndroidWorld permission probe is unsupported for %s: %s",
                    package_name,
                    error,
                )

            # Android 13 emulator images commonly omit the legacy
            # ``pm check-permission`` subcommand.  Keep the official probe as
            # the first path, then use the read-only dumpsys package output as
            # a compatibility adapter; never grant or mutate permissions here.
            fallback = adb_utils.issue_generic_request(
                ["shell", "dumpsys", "package", package_name],
                env,
            )
            fallback_output = getattr(
                getattr(fallback, "generic", None), "output", b""
            )
            if isinstance(fallback_output, bytes):
                fallback_output = fallback_output.decode("utf-8", errors="replace")
            permission_pattern = re.compile(
                rf"^\s*{re.escape(permission)}:\s+granted=true\b",
                flags=re.MULTILINE,
            )
            if permission_pattern.search(str(fallback_output or "")) is None:
                return False
    except (AttributeError, ModuleNotFoundError, RuntimeError, TypeError, ValueError):
        return False
    return True


def _patch_androidworld_optional_setup_click() -> tuple[Any, Any] | None:
    try:
        tools_module = importlib.import_module("android_world.env.tools")
    except ModuleNotFoundError:
        return None
    controller_type = getattr(tools_module, "AndroidToolController", None)
    original = getattr(controller_type, "click_element", None)
    if controller_type is None or not callable(original):
        return None

    def click_element(controller: Any, element_text: str) -> Any:
        try:
            return original(controller, element_text)
        except ValueError as error:
            message = str(error)
            normalized_label = _normalize_androidworld_setup_label(element_text)
            missing_target = "Target text" in message and "not found" in message
            empty_a11y_tree = (
                normalized_label in {"NEXT", "Skip", "Don't allow"}
                and "Invalid element index" in message
            )
            if missing_target:
                elements = controller._env.get_ui_elements() or []
                labels = {
                    _normalize_androidworld_setup_label(
                        getattr(element, "text", None)
                        or getattr(element, "content_description", None)
                        or ""
                    )
                    for element in elements
                }
                if "OK" in labels and any(
                    label.startswith("This app was built for an older version of Android")
                    for label in labels
                ):
                    logger.info(
                        "AndroidWorld setup is dismissing the legacy-app dialog "
                        "before retrying %s",
                        element_text,
                    )
                    original(controller, "OK")
                    for _ in range(6):
                        try:
                            return original(controller, element_text)
                        except ValueError:
                            time.sleep(0.5)
            if (
                normalized_label == "Allow"
                and missing_target
                and _androidworld_runtime_permissions_granted(
                    controller,
                    package_name="de.dennisguse.opentracks",
                    permissions=(
                        "android.permission.BLUETOOTH_CONNECT",
                        "android.permission.BLUETOOTH_SCAN",
                    ),
                )
            ):
                logger.info(
                    "AndroidWorld OpenTracks Bluetooth permissions are already "
                    "granted; skipping the stale Allow lookup"
                )
                return None
            if normalized_label == "NEXT" and missing_target:
                activity = str(
                    getattr(controller._env, "foreground_activity_name", "") or ""
                ).strip()
                packages = {
                    str(getattr(element, "package_name", "") or "").strip()
                    for element in controller._env.get_ui_elements() or ()
                }
                camera_is_foreground = activity.startswith(
                    "com.android.camera2/"
                )
                camera_is_visible = "com.android.camera2" in packages
                launcher_is_foreground = activity.startswith(
                    "com.google.android.apps.nexuslauncher/"
                )
                if camera_is_foreground or camera_is_visible or launcher_is_foreground:
                    logger.info(
                        "AndroidWorld Camera setup is already settled; skipping "
                        "stale NEXT lookup"
                    )
                    return None
            if normalized_label == "Skip" and (
                missing_target or empty_a11y_tree
            ):
                elements = controller._env.get_ui_elements() or []
                chooser_clicks = _app_chooser_clicks(
                    elements,
                    app_label="Contacts",
                )
                if chooser_clicks:
                    logger.info(
                        "AndroidWorld Contacts setup is resolving the system "
                        "app chooser before onboarding"
                    )
                    for label in chooser_clicks:
                        original(controller, label)
                    for _ in range(6):
                        try:
                            return original(controller, "Skip")
                        except ValueError:
                            time.sleep(0.5)
                    logger.info(
                        "AndroidWorld Contacts chooser completed without an "
                        "optional Skip control; continuing setup"
                    )
                    return None
            if missing_target or empty_a11y_tree:
                # Camera can still be publishing its accessibility tree while
                # AndroidWorld starts the app. Give the official setup a
                # short bounded chance to observe the real NEXT button before
                # applying the existing optional-click fallback.
                for _ in range(6):
                    time.sleep(0.5)
                    try:
                        return original(controller, element_text)
                    except ValueError as retry_error:
                        message = str(retry_error)
                        retry_missing_target = (
                            "Target text" in message and "not found" in message
                        )
                        retry_empty_a11y_tree = (
                            normalized_label in {"NEXT", "Skip", "Don't allow"}
                            and "Invalid element index" in message
                        )
                        if not retry_missing_target and not retry_empty_a11y_tree:
                            break
            if normalized_label == "NEXT" and empty_a11y_tree:
                activity = str(
                    getattr(controller._env, "foreground_activity_name", "") or ""
                ).strip()
                packages = {
                    str(getattr(element, "package_name", "") or "").strip()
                    for element in controller._env.get_ui_elements() or ()
                }
                if not activity and not packages:
                    logger.info(
                        "AndroidWorld Camera setup is still publishing an empty "
                        "tree; treating optional NEXT as non-blocking"
                    )
                    return None
            if normalized_label == "Skip" and missing_target:
                activity = str(
                    getattr(controller._env, "foreground_activity_name", "") or ""
                ).strip()
                packages = {
                    str(getattr(element, "package_name", "") or "").strip()
                    for element in controller._env.get_ui_elements() or ()
                }
                if not activity and not packages:
                    logger.info(
                        "AndroidWorld Contacts setup is still publishing an empty "
                        "tree; treating optional Skip as non-blocking"
                    )
                    return None
            if not missing_target and not empty_a11y_tree:
                raise
            if normalized_label == "OK":
                activity = str(
                    getattr(controller._env, "foreground_activity_name", "") or ""
                ).strip()
                if not activity.startswith("net.gsantner.markor/"):
                    raise
                logger.info(
                    "AndroidWorld Markor setup is already complete; skipping "
                    "absent OK button"
                )
                return None
            if normalized_label == "NEXT":
                activity = str(
                    getattr(controller._env, "foreground_activity_name", "") or ""
                ).strip()
                packages = {
                    str(getattr(element, "package_name", "") or "").strip()
                    for element in controller._env.get_ui_elements() or ()
                }
                camera_is_foreground = activity.startswith("com.android.camera2/")
                camera_is_visible = "com.android.camera2" in packages
                if camera_is_foreground or camera_is_visible:
                    logger.info(
                        "AndroidWorld Camera setup is already complete; "
                        "skipping absent NEXT button"
                    )
                    return None
                launcher_is_foreground = activity.startswith(
                    "com.google.android.apps.nexuslauncher/"
                )
                non_app_packages = {
                    "",
                    "android",
                    "com.android.systemui",
                    "com.google.android.apps.nexuslauncher",
                }
                if launcher_is_foreground or (
                    packages and packages.issubset(non_app_packages)
                ):
                    logger.info(
                        "AndroidWorld optional app setup did not expose NEXT "
                        "and remains on the launcher; continuing"
                    )
                    return None
            if normalized_label in {
                "All files",
                "Allow access to manage all files",
            }:
                activity = str(
                    getattr(controller._env, "foreground_activity_name", "") or ""
                ).strip()
                packages = {
                    str(getattr(element, "package_name", "") or "").strip()
                    for element in controller._env.get_ui_elements() or ()
                }
                gallery_visible = activity.startswith(
                    "com.simplemobiletools.gallery.pro/"
                ) or "com.simplemobiletools.gallery.pro" in packages
                if gallery_visible:
                    logger.info(
                        "AndroidWorld Gallery setup is already complete; "
                        "skipping absent %s button",
                        normalized_label,
                    )
                    return None
            if normalized_label != "Don't allow":
                raise
            elements = controller._env.get_ui_elements() or []
            packages = {
                str(getattr(element, "package_name", "") or "").strip()
                for element in elements
            }
            if not packages:
                logger.info(
                    "AndroidWorld setup is still publishing an empty tree; "
                    "treating optional %s as non-blocking",
                    element_text,
                )
                return None
            if any(package.endswith(".permissioncontroller") for package in packages):
                raise
            app_packages = {
                package
                for package in packages
                if package
                and package not in {"android", "com.android.systemui"}
                and not package.endswith(".permissioncontroller")
            }
            if not app_packages:
                raise
            logger.info(
                "AndroidWorld setup notification permission is already settled; "
                "skipping absent %s button on packages=%s",
                element_text,
                ",".join(sorted(app_packages)),
            )
            return None

    controller_type.click_element = click_element
    return controller_type, original
