from __future__ import annotations

import re
from typing import Any
import xml.etree.ElementTree as ET


def _read(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _forest_bounds(value: Any) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    try:
        bounds = (
            int(float(_read(value, "left", 0) or 0)),
            int(float(_read(value, "top", 0) or 0)),
            int(float(_read(value, "right"))),
            int(float(_read(value, "bottom"))),
        )
    except (TypeError, ValueError):
        return None
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        return None
    return bounds


def androidworld_forest_xml(
    forest: Any,
    *,
    screen_size: tuple[float, float],
) -> str:
    width, height = (max(1, int(value)) for value in screen_size)
    hierarchy = ET.Element(
        "hierarchy",
        {
            "width": str(width),
            "height": str(height),
        },
    )
    rendered_nodes = 0
    for window_index, window in enumerate(
        list(_read(forest, "windows", ()) or ())
    ):
        tree = _read(window, "tree")
        nodes = list(_read(tree, "nodes", ()) or ())
        if not nodes:
            continue
        node_by_id = {
            str(_read(node, "unique_id", index)): node
            for index, node in enumerate(nodes)
        }
        child_ids = {
            str(child_id)
            for node in nodes
            for child_id in list(_read(node, "child_ids", ()) or ())
        }
        roots = [
            node
            for index, node in enumerate(nodes)
            if str(_read(node, "unique_id", index)) not in child_ids
        ]
        window_id = str(_read(window, "id", window_index))
        window_element = ET.SubElement(
            hierarchy,
            "window",
            {
                "id": f"window-{window_id}",
                "title": str(_read(window, "title", "") or ""),
            },
        )
        visited: set[str] = set()

        def append_node(node: Any, parent: ET.Element) -> None:
            nonlocal rendered_nodes
            node_id = str(_read(node, "unique_id", ""))
            if not node_id or node_id in visited:
                return
            visited.add(node_id)
            children = [
                node_by_id[str(child_id)]
                for child_id in list(_read(node, "child_ids", ()) or ())
                if str(child_id) in node_by_id
            ]
            bounds = _forest_bounds(_read(node, "bounds_in_screen"))
            visible = bool(_read(node, "is_visible_to_user", True))
            if not visible or bounds is None:
                for child in children:
                    append_node(child, parent)
                return
            left, top, right, bottom = bounds
            attributes = {
                "id": f"{window_id}:{node_id}",
                "class": str(_read(node, "class_name", "") or ""),
                "text": str(_read(node, "text", "") or ""),
                "content-desc": str(
                    _read(node, "content_description", "") or ""
                ),
                "resource-id": str(
                    _read(node, "view_id_resource_name", "")
                    or _read(node, "resource_name", "")
                    or _read(node, "resource_id", "")
                    or ""
                ),
                "package": str(_read(node, "package_name", "") or ""),
                "bounds": f"[{left},{top}][{right},{bottom}]",
                "checkable": str(
                    bool(_read(node, "is_checkable", False))
                ).lower(),
                "checked": str(bool(_read(node, "is_checked", False))).lower(),
                "clickable": str(
                    bool(_read(node, "is_clickable", False))
                ).lower(),
                "editable": str(bool(_read(node, "is_editable", False))).lower(),
                "enabled": str(bool(_read(node, "is_enabled", True))).lower(),
                "focusable": str(
                    bool(_read(node, "is_focusable", False))
                ).lower(),
                "focused": str(bool(_read(node, "is_focused", False))).lower(),
                "long-clickable": str(
                    bool(_read(node, "is_long_clickable", False))
                ).lower(),
                "password": str(bool(_read(node, "is_password", False))).lower(),
                "scrollable": str(
                    bool(_read(node, "is_scrollable", False))
                ).lower(),
                "selected": str(bool(_read(node, "is_selected", False))).lower(),
                "visible": "true",
            }
            element = ET.SubElement(parent, "node", attributes)
            rendered_nodes += 1
            for child in children:
                append_node(child, element)

        for root in roots:
            append_node(root, window_element)
        for node in nodes:
            append_node(node, window_element)
        if not list(window_element):
            hierarchy.remove(window_element)
    if rendered_nodes == 0:
        return ""
    return ET.tostring(hierarchy, encoding="unicode")


def xml_covers_screen(
    xml_text: str,
    *,
    package_name: str,
    screen_size: tuple[float, float],
) -> bool:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return False
    bounds: list[tuple[int, int, int, int]] = []
    for element in root.iter():
        package = str(element.attrib.get("package") or "")
        if package_name and package != package_name:
            continue
        numbers = [
            int(item)
            for item in re.findall(
                r"-?\d+", str(element.attrib.get("bounds") or "")
            )
        ]
        if len(numbers) == 4:
            bounds.append((numbers[0], numbers[1], numbers[2], numbers[3]))
    if not bounds:
        return False
    width, height = (int(value) for value in screen_size)
    return (
        min(item[0] for item in bounds) <= 0
        and min(item[1] for item in bounds) <= 0
        and max(item[2] for item in bounds) >= width
        and max(item[3] for item in bounds) >= height
    )


def forest_has_complete_active_application_window(
    forest: Any,
    *,
    package_name: str,
) -> bool:
    """Returns whether the official forest fully represents an active app window.

    Android modal dialogs intentionally occupy only part of the physical screen.
    They are complete transfer pages when the active/focused application window
    has a package root covering that window and exposes semantic or actionable
    descendants.
    """
    if forest is None or not package_name:
        return False
    for window in list(_read(forest, "windows", ()) or ()):
        window_type = _read(window, "window_type", "")
        normalized_window_type = str(
            getattr(window_type, "name", "") or window_type or ""
        ).upper()
        if window_type != 1 and normalized_window_type not in {
            "1",
            "TYPE_APPLICATION",
        }:
            continue
        if not (
            bool(_read(window, "is_active", False))
            or bool(_read(window, "is_focused", False))
        ):
            continue
        window_bounds = _forest_bounds(_read(window, "bounds_in_screen"))
        if window_bounds is None:
            continue
        nodes = list(_read(_read(window, "tree"), "nodes", ()) or ())
        package_nodes = [
            node
            for node in nodes
            if str(_read(node, "package_name", "") or "") == package_name
            and bool(_read(node, "is_visible_to_user", True))
        ]
        roots_cover_window = any(
            (bounds := _forest_bounds(_read(node, "bounds_in_screen"))) is not None
            and bounds[0] <= window_bounds[0]
            and bounds[1] <= window_bounds[1]
            and bounds[2] >= window_bounds[2]
            and bounds[3] >= window_bounds[3]
            for node in package_nodes
        )
        has_semantic_or_actionable_node = any(
            str(_read(node, "text", "") or "").strip()
            or str(_read(node, "content_description", "") or "").strip()
            or str(_read(node, "view_id_resource_name", "") or "").strip()
            or bool(_read(node, "is_clickable", False))
            or bool(_read(node, "is_editable", False))
            for node in package_nodes
        )
        if roots_cover_window and has_semantic_or_actionable_node:
            return True
    return False


def xml_with_screen_size(
    xml_text: str,
    *,
    screen_size: tuple[float, float],
) -> str:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return xml_text
    width, height = (max(1, int(value)) for value in screen_size)
    root.attrib.update(
        bounds=f"[0,0][{width},{height}]",
        width=str(width),
        height=str(height),
    )
    return ET.tostring(root, encoding="unicode")


__all__ = [
    "androidworld_forest_xml",
    "forest_has_complete_active_application_window",
    "xml_covers_screen",
    "xml_with_screen_size",
]
