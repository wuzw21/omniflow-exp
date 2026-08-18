from __future__ import annotations


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


__all__ = ["resolve_androidworld_package"]
