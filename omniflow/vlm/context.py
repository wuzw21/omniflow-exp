from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any
import xml.etree.ElementTree as ET

_BOUNDS = re.compile(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]")
_TRUE = {"1", "true"}


@dataclass(frozen=True)
class PageContext:
    evidence: str = ""
    useful: bool = False
    clickable: bool = False
    editable: bool = False
    scrollable: bool = False
    focused: bool = False


def analyze_page_context(state: dict[str, Any], *, limit: int = 24) -> PageContext:
    xml_text = str(state.get("xml") or "").strip()
    if not xml_text:
        return PageContext()
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return PageContext()

    display = state.get("display") if isinstance(state.get("display"), dict) else {}
    width = _positive_number(display.get("width")) or _positive_number(root.get("width"))
    height = _positive_number(display.get("height")) or _positive_number(root.get("height"))
    records: list[tuple[bool, str]] = []
    seen: set[tuple[str, ...]] = set()
    clickable = editable = scrollable = focused = False
    for node in root.iter():
        attrs = node.attrib
        class_name = str(attrs.get("class") or "").strip()
        text = _clean(attrs.get("text"))
        description = _clean(attrs.get("content-desc") or attrs.get("contentDescription"))
        resource_id = _clean(attrs.get("resource-id") or attrs.get("resourceId"))
        bounds = _bounds(attrs.get("bounds"))
        node_clickable = _flag(attrs, "clickable") or _flag(attrs, "checkable")
        node_editable = _flag(attrs, "editable") or class_name.endswith("EditText")
        node_scrollable = _flag(attrs, "scrollable")
        node_focused = _flag(attrs, "focused")
        node_actionable = node_clickable or node_editable or node_scrollable
        clickable = clickable or node_clickable
        editable = editable or node_editable
        scrollable = scrollable or node_scrollable
        focused = focused or node_focused
        if not (text or description or resource_id or node_actionable):
            continue
        key = (text, description, resource_id, class_name, str(bounds))
        if key in seen:
            continue
        seen.add(key)
        parts: list[str] = []
        node_id = _clean(attrs.get("id") or attrs.get("index"))
        if node_id:
            parts.append(f"node_id={node_id}")
        if text:
            parts.append(f'text="{text}"')
        if description and description != text:
            parts.append(f'desc="{description}"')
        if resource_id:
            parts.append(f'id="{resource_id}"')
        if class_name:
            parts.append(f"class={class_name.rsplit('.', 1)[-1]}")
        if bounds is not None:
            parts.append(f"bounds={_normalized_bounds(bounds, width, height)}")
        flags = [
            name
            for name, enabled in (
                ("clickable", node_clickable),
                ("editable", node_editable),
                ("scrollable", node_scrollable),
                ("focused", node_focused),
                ("checked", _flag(attrs, "checked")),
                ("selected", _flag(attrs, "selected")),
            )
            if enabled
        ]
        if flags:
            parts.append("flags=" + ",".join(flags))
        records.append((node_actionable, " ".join(parts)))

    selected = sorted(
        enumerate(records),
        key=lambda item: (not item[1][0], item[0]),
    )[: max(1, int(limit))]
    visible_elements = [
        f"[{index}] {record}"
        for index, (_actionable, record) in enumerate(
            (item[1] for item in selected),
            start=1,
        )
    ]
    useful = bool(visible_elements)
    evidence = (
        "Primary UI grounding evidence from the latest accessibility XML "
        "(bounds use normalized 0..1000 coordinates):\n"
        + "\n".join(visible_elements)
    )
    return PageContext(
        evidence=evidence,
        useful=useful,
        clickable=clickable or any("bounds=" in record for record in visible_elements),
        editable=editable,
        scrollable=scrollable,
        focused=focused,
    )
def _flag(attrs: dict[str, str], name: str) -> bool:
    return str(attrs.get(name) or "").strip().casefold() in _TRUE


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())[:240]


def _positive_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _bounds(value: Any) -> tuple[int, int, int, int] | None:
    match = _BOUNDS.fullmatch(str(value or "").strip())
    if match is None:
        return None
    return tuple(int(item) for item in match.groups())


def _normalized_bounds(
    bounds: tuple[int, int, int, int],
    width: float | None,
    height: float | None,
) -> str:
    if width is None or height is None:
        return f"[{bounds[0]},{bounds[1]}][{bounds[2]},{bounds[3]}]"
    x1, y1, x2, y2 = bounds
    normalized = (
        round(max(0, min(1000, x1 * 1000 / width))),
        round(max(0, min(1000, y1 * 1000 / height))),
        round(max(0, min(1000, x2 * 1000 / width))),
        round(max(0, min(1000, y2 * 1000 / height))),
    )
    return f"[{normalized[0]},{normalized[1]}][{normalized[2]},{normalized[3]}]"


__all__ = [
    "PageContext",
    "analyze_page_context",
]
