from __future__ import annotations


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


__all__ = ["resolve_androidworld_app_name", "resolve_androidworld_package"]
