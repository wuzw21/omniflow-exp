from __future__ import annotations


def resolve_androidworld_app_name(package_name: str, controller: object) -> str:
    package = str(package_name or "").strip()
    if not package:
        return ""
    from android_world.env import adb_utils

    matching_names: list[str] = []
    for app_name in sorted(adb_utils.get_all_apps(controller)):
        activity = str(adb_utils.get_adb_activity(app_name) or "").strip()
        if activity.split("/", 1)[0].strip() == package:
            matching_names.append(str(app_name))
    package_leaf = package.rsplit(".", 1)[-1].casefold()
    for app_name in matching_names:
        aliases = {alias.strip().casefold() for alias in app_name.split("|")}
        if package_leaf in aliases:
            return app_name
    if matching_names:
        return matching_names[0]
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
