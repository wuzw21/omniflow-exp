from __future__ import annotations

import re
from typing import Iterable


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
    from android_world.env import adb_utils

    activity = str(adb_utils.get_adb_activity(name) or "").strip()
    package_name = activity.split("/", 1)[0].strip()
    if package_name:
        return package_name
    if "." in name and " " not in name:
        return name
    return ""


__all__ = [
    "launchable_androidworld_apps",
    "launcher_package_label",
    "resolve_androidworld_app_name",
    "resolve_androidworld_package",
]
