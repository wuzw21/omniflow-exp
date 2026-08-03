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
_GLOBAL_CONTROL_MARKERS = (
    "back",
    "basket",
    "cart",
    "close",
    "input",
    "menu",
    "more options",
    "navigate up",
    "navigate_up",
    "search",
    "关闭",
    "输入",
    "返回",
    "搜索",
    "更多",
    "查找",
    "菜单",
    "购物车",
)
_GROUP_ORDER = ("global", "goal", "visual", "other")
_GROUP_HEADERS = {
    "global": "[global_controls]",
    "goal": "[goal_matches]",
    "visual": "[visual_controls]",
    "other": "[other_context]",
}
_GROUP_LIMITS = {
    "global": 8,
    "goal": 12,
    "visual": 6,
    "other": 4,
}


@dataclass(frozen=True)
class _Candidate:
    order: int
    score: int
    goal_match: bool
    group: str
    compact: dict[str, object]
    bounds: tuple[int, int, int, int] | None


@dataclass(frozen=True)
class UIProjection:
    text: str
    candidate_count: int
    selected_count: int
    goal_match_count: int
    visual_context_required: bool = False
    visual_candidate_count: int = 0

    @property
    def requires_screenshot(self) -> bool:
        return True


def project_ui(xml_text: str, goal: str, *, max_nodes: int = 30) -> UIProjection:
    try:
        root = ET.fromstring(str(xml_text or ""))
    except ET.ParseError:
        return UIProjection("<none>", 0, 0, 0)
    goal_terms = _terms(goal)
    candidates: list[_Candidate] = []
    for order, element in enumerate(root.iter()):
        compact: dict[str, object] = {}
        semantic_values: list[str] = []
        visible_semantic_values: list[str] = []
        node_id = str(element.attrib.get("id") or "").strip()
        if node_id:
            compact["i"] = node_id
        for attribute, output_key in _SEMANTIC_ATTRIBUTES:
            value = str(element.attrib.get(attribute) or "").strip()
            if value:
                compact[output_key] = value
                semantic_values.append(value)
                if attribute != "resource-id":
                    visible_semantic_values.append(value)
        actions = [
            output_value
            for attribute, output_value in _ACTION_ATTRIBUTES
            if str(element.attrib.get(attribute) or "").strip().lower() == "true"
        ]
        if not semantic_values and not actions:
            continue
        bounds = str(element.attrib.get("bounds") or "").strip()
        parsed_bounds = _parse_bounds(bounds)
        if bounds:
            compact["b"] = bounds
        if actions:
            compact["a"] = actions
        checked = str(element.attrib.get("checked") or "").strip().lower()
        if checked in {"true", "false"}:
            compact["checked"] = checked == "true"
        candidate_terms = _terms(" ".join(semantic_values))
        overlap = goal_terms.intersection(candidate_terms)
        goal_match = bool(overlap)
        score = len(overlap) * 1000
        global_control = _is_global_control(semantic_values, actions)
        visual_control = bool(actions and not visible_semantic_values)
        if global_control:
            group = "global"
            score += 5000
        elif goal_match:
            group = "goal"
        elif visual_control:
            group = "visual"
            score += 200
        else:
            group = "other"
        score += 400 if "edit" in actions or "focus" in actions else 0
        score += 50 if actions else 0
        score += min(10, len(semantic_values))
        candidates.append(
            _Candidate(
                order=order,
                score=score,
                goal_match=goal_match,
                group=group,
                compact=compact,
                bounds=parsed_bounds,
            )
        )
    selected = _select_candidates(candidates, max_nodes=max_nodes)
    text = _render_candidates(selected)
    return UIProjection(
        text=text or "<none>",
        candidate_count=len(candidates),
        selected_count=len(selected),
        goal_match_count=sum(1 for item in candidates if item.goal_match),
        visual_context_required=any(
            marker in str(goal or "").casefold() for marker in _VISUAL_GOAL_MARKERS
        ),
        visual_candidate_count=sum(
            1 for item in selected if item.group == "visual"
        ),
    )


def _select_candidates(
    candidates: list[_Candidate],
    *,
    max_nodes: int,
) -> list[_Candidate]:
    if max_nodes <= 0:
        return []
    selected: list[_Candidate] = []
    selected_orders: set[int] = set()
    for group in _GROUP_ORDER:
        remaining = max_nodes - len(selected)
        if remaining <= 0:
            break
        limit = min(_GROUP_LIMITS[group], remaining)
        group_candidates = sorted(
            (item for item in candidates if item.group == group),
            key=_candidate_rank,
        )
        for item in group_candidates[:limit]:
            selected.append(item)
            selected_orders.add(item.order)
    remaining = max_nodes - len(selected)
    if remaining > 0:
        overflow = sorted(
            (item for item in candidates if item.order not in selected_orders),
            key=lambda item: (_GROUP_ORDER.index(item.group), *_candidate_rank(item)),
        )
        selected.extend(overflow[:remaining])
    return sorted(
        selected,
        key=lambda item: (
            _GROUP_ORDER.index(item.group),
            _screen_order(item),
            item.order,
        ),
    )


def _render_candidates(candidates: list[_Candidate]) -> str:
    lines: list[str] = []
    visual_reference = 0
    for group in _GROUP_ORDER:
        group_candidates = [item for item in candidates if item.group == group]
        if not group_candidates:
            continue
        lines.append(_GROUP_HEADERS[group])
        for item in group_candidates:
            compact = dict(item.compact)
            if compact.get("a"):
                visual_reference += 1
                compact = {"v": f"A{visual_reference:02d}", **compact}
            lines.append(
                json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
            )
    return "\n".join(lines)


def _candidate_rank(item: _Candidate) -> tuple[int, tuple[int, int], int]:
    return (-item.score, _screen_order(item), item.order)


def _screen_order(item: _Candidate) -> tuple[int, int]:
    if item.bounds is None:
        return (10**9, 10**9)
    left, top, _right, _bottom = item.bounds
    return (top, left)


def _is_global_control(semantic_values: list[str], actions: list[str]) -> bool:
    if "edit" in actions or "focus" in actions:
        return True
    if not actions:
        return False
    semantic_text = " ".join(semantic_values).casefold()
    return any(marker in semantic_text for marker in _GLOBAL_CONTROL_MARKERS)


def _parse_bounds(value: str) -> tuple[int, int, int, int] | None:
    numbers = re.fullmatch(
        r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]",
        str(value or "").strip(),
    )
    if numbers is None:
        return None
    left, top, right, bottom = (int(item) for item in numbers.groups())
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


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
