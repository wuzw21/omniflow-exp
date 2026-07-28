from __future__ import annotations

from dataclasses import dataclass
import json
import re
import xml.etree.ElementTree as ET

_SEMANTIC_ATTRIBUTES = (
    ("text", "t"),
    ("content-desc", "d"),
    ("hint-text", "h"),
    ("resource-id", "r"),
)
_ACTION_ATTRIBUTES = (
    ("clickable", "click"),
    ("long-clickable", "long"),
    ("editable", "edit"),
    ("scrollable", "scroll"),
    ("checkable", "check"),
    ("focused", "focus"),
)
_ENGLISH_TOKEN = re.compile(r"[a-z0-9]+")
_CHINESE_TOKEN = re.compile(r"[\u4e00-\u9fff]+")
_VISUAL_GOAL_MARKERS = ("广告", "弹窗", "遮挡", "popup", "overlay", "close ad")


@dataclass(frozen=True)
class UIProjection:
    text: str
    candidate_count: int
    selected_count: int
    goal_match_count: int
    visual_context_required: bool = False

    @property
    def requires_screenshot(self) -> bool:
        return (
            self.visual_context_required
            or self.selected_count == 0
            or self.goal_match_count == 0
        )


def project_ui(xml_text: str, goal: str, *, max_nodes: int = 50) -> UIProjection:
    try:
        root = ET.fromstring(str(xml_text or ""))
    except ET.ParseError:
        return UIProjection("<none>", 0, 0, 0)
    goal_terms = _terms(goal)
    candidates: list[tuple[int, int, bool, dict[str, object]]] = []
    for order, element in enumerate(root.iter()):
        compact: dict[str, object] = {}
        semantic_values: list[str] = []
        node_id = str(element.attrib.get("id") or "").strip()
        if node_id:
            compact["i"] = node_id
        for attribute, output_key in _SEMANTIC_ATTRIBUTES:
            value = str(element.attrib.get(attribute) or "").strip()
            if value:
                compact[output_key] = value
                semantic_values.append(value)
        actions = [
            output_value
            for attribute, output_value in _ACTION_ATTRIBUTES
            if str(element.attrib.get(attribute) or "").strip().lower() == "true"
        ]
        if not semantic_values and not actions:
            continue
        bounds = str(element.attrib.get("bounds") or "").strip()
        if bounds:
            compact["b"] = bounds
        if actions:
            compact["a"] = actions
        candidate_terms = _terms(" ".join(semantic_values))
        overlap = goal_terms.intersection(candidate_terms)
        goal_match = bool(overlap)
        score = len(overlap) * 1000
        score += 40 if "edit" in actions or "focus" in actions else 0
        score += 20 if actions else 0
        score += min(10, len(semantic_values))
        candidates.append((score, order, goal_match, compact))
    selected = sorted(candidates, key=lambda item: (-item[0], item[1]))[:max_nodes]
    selected.sort(key=lambda item: item[1])
    text = "\n".join(
        json.dumps(item[3], ensure_ascii=False, separators=(",", ":"))
        for item in selected
    )
    return UIProjection(
        text=text or "<none>",
        candidate_count=len(candidates),
        selected_count=len(selected),
        goal_match_count=sum(1 for item in candidates if item[2]),
        visual_context_required=any(
            marker in str(goal or "").casefold() for marker in _VISUAL_GOAL_MARKERS
        ),
    )


def _terms(value: str) -> set[str]:
    normalized = str(value or "").casefold()
    terms = {
        token
        for token in _ENGLISH_TOKEN.findall(normalized)
        if len(token) >= 2
    }
    for segment in _CHINESE_TOKEN.findall(normalized):
        if len(segment) == 1:
            terms.add(segment)
        else:
            terms.update(segment[index : index + 2] for index in range(len(segment) - 1))
    return terms


__all__ = ["UIProjection", "project_ui"]
