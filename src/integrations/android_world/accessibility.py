from __future__ import annotations

import re
import xml.etree.ElementTree as ET


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
    "xml_covers_screen",
    "xml_with_screen_size",
]
