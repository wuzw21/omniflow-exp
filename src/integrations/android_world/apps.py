from __future__ import annotations

import re
from typing import Iterable


# AndroidWorld's model-facing names and the packages exposed by the pinned
# emulator are not always byte-for-byte identical.  Keep this normalization
# in the shared app resolver so GUI validation, OOB dispatch, and replay use
# the same launchable package identity.
_PACKAGE_ALIASES = {
    "com.android.contacts": "com.google.android.contacts",
    "com.google.android.googlecamera": "com.android.camera2",
    "com.example.broccoli": "com.flauschcode.broccoli",
}


def canonicalize_androidworld_package(package_name: str) -> str:
    """Map a known official AndroidWorld package alias to its runtime package."""

    value = str(package_name or "").strip()
    if not value:
        return ""
    return _PACKAGE_ALIASES.get(value.casefold(), value)


def launchable_androidworld_apps(
    installed_packages: Iterable[str],
    controller: object,
) -> dict[str, str]:
    """Return only official AndroidWorld apps that have launchable activities."""

    from android_world.env import adb_utils

    installed = {
        str(package).strip() for package in installed_packages if str(package).strip()
    }
    catalog: dict[str, str] = {}
    for app_name in sorted(adb_utils.get_all_apps(controller)):
        activity = str(adb_utils.get_adb_activity(app_name) or "").strip()
        package = activity.split("/", 1)[0].strip()
        if not package or package not in installed:
            continue
        label = re.sub(r"[_-]+", " ", str(app_name)).strip().title()
        if label:
            catalog[label] = package
    return dict(sorted(catalog.items(), key=lambda item: (item[0].casefold(), item[1])))


def launcher_package_label(package_name: str) -> str:
    """Return a deterministic fallback label for a launcher package."""

    package = str(package_name or "").strip()
    segment = package.rsplit(".", 1)[-1] if package else ""
    words = re.sub(r"[^A-Za-z0-9]+", " ", segment).strip()
    return words.title() or package


def resolve_androidworld_app_name(package_name: str, controller: object) -> str:
    package = str(package_name or "").strip()
    if not package:
        return ""
    from android_world.env import adb_utils

    for app_name in sorted(adb_utils.get_all_apps(controller)):
        activity = str(adb_utils.get_adb_activity(app_name) or "").strip()
        if activity.split("/", 1)[0].strip() == package:
            return str(app_name)
    registry = getattr(adb_utils, "_PATTERN_TO_ACTIVITY", {})
    for pattern, activity in registry.items():
        if str(activity).split("/", 1)[0].strip() == package:
            return str(pattern).split("|", 1)[0].strip()
    return package


def resolve_androidworld_package(app_name: str) -> str:
    name = str(app_name or "").strip()
    if not name:
        return ""
    canonical_name = canonicalize_androidworld_package(name)
    from android_world.env import adb_utils

    activity = str(adb_utils.get_adb_activity(canonical_name) or "").strip()
    package_name = activity.split("/", 1)[0].strip()
    if package_name:
        return canonicalize_androidworld_package(package_name)
    if "." in canonical_name and " " not in canonical_name:
        return canonical_name
    return ""


__all__ = [
    "launchable_androidworld_apps",
    "launcher_package_label",
    "canonicalize_androidworld_package",
    "resolve_androidworld_app_name",
    "resolve_androidworld_package",
]
