from __future__ import annotations

from dataclasses import dataclass, replace
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
_NUMERIC_CONTROL_SUMMARY = re.compile(
    r"(?:"
    r"(?:digit|number|key|button|\u6570\u5b57|\u6309\u952e|\u6309\u94ae)\s*"
    r"|(?:tap|click|press|select|\u70b9\u51fb|\u6309\u4e0b|\u9009\u62e9)\s+(?:the\s+)?"
    r")['\"]?(\d{1,2})(?!\d)",
    re.IGNORECASE,
)
_VISUAL_GOAL_MARKERS = (
    "广告",
    "弹窗",
    "遮挡",
    "canvas",
    "close ad",
    "draw",
    "image",
    "maze",
    "overlay",
    "photo",
    "picture",
    "popup",
)
_VISUALLY_OPAQUE_ACTION_THRESHOLD = 8
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
_BROWSE_CONTROL_MARKERS = (
    "see all",
    "history",
    "recent",
    "saved",
    "view all",
    "历史",
    "已保存",
    "查看全部",
)
_GROUP_ORDER = ("global", "goal", "goal_control", "visual", "other")
_GROUP_HEADERS = {
    "global": "[global_controls]",
    "goal": "[goal_matches]",
    "goal_control": "[goal_controls]",
    "visual": "[visual_controls]",
    "other": "[other_context]",
}
_GROUP_LIMITS = {
    "global": 6,
    "goal": 10,
    "goal_control": 8,
    "visual": 4,
    "other": 6,
}


@dataclass(frozen=True)
class _Candidate:
    order: int
    score: int
    goal_match: bool
    group: str
    compact: dict[str, object]
    bounds: tuple[int, int, int, int] | None
    inside_webview: bool


@dataclass(frozen=True)
class UIProjection:
    text: str
    candidate_count: int
    selected_count: int
    goal_match_count: int
    visual_context_required: bool = False
    visual_candidate_count: int = 0
    nodes: tuple[ProjectedNode, ...] = ()

    @property
    def requires_screenshot(self) -> bool:
        return True


@dataclass(frozen=True)
class ProjectedNode:
    reference: str
    labels: tuple[str, ...]
    bounds: tuple[int, int, int, int]
    inside_webview: bool = False


def project_ui(
    xml_text: str,
    goal: str,
    *,
    max_nodes: int = 30,
    include_all_nodes: bool = False,
) -> UIProjection:
    try:
        root = ET.fromstring(str(xml_text or ""))
    except ET.ParseError:
        return UIProjection("<none>", 0, 0, 0)
    goal_terms = _terms(goal)
    candidates: list[_Candidate] = []
    webview_elements = _webview_elements(root)
    for order, element in enumerate(root.iter()):
        compact: dict[str, object] = {}
        semantic_values: list[str] = []
        visible_semantic_values: list[str] = []
        if include_all_nodes:
            compact["k"] = str(element.attrib.get("class") or element.tag).strip()
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
        if not include_all_nodes and not semantic_values and not actions:
            continue
        descendant_context = (
            _descendant_context(element) if actions and not visible_semantic_values else ""
        )
        if descendant_context:
            compact["c"] = descendant_context
        bounds = str(element.attrib.get("bounds") or "").strip()
        parsed_bounds = _parse_bounds(bounds)
        if bounds:
            compact["b"] = bounds
        if actions:
            compact["a"] = actions
        checked = str(element.attrib.get("checked") or "").strip().lower()
        if checked in {"true", "false"}:
            compact["checked"] = checked == "true"
        resource_context = str(compact.get("r") or "").rsplit("/", 1)[-1]
        candidate_terms = _terms(
            " ".join(
                (*visible_semantic_values, descendant_context, resource_context)
            )
        )
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
            score += 200 + _visual_specificity(parsed_bounds)
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
                inside_webview=id(element) in webview_elements,
            )
        )
    if not include_all_nodes:
        candidates = _prune_redundant_candidates(candidates, goal)
    candidates = _promote_goal_controls(candidates)
    visually_opaque_action_count = sum(
        1 for item in candidates if item.group in {"goal_control", "visual"}
    )
    visual_context_required = (
        any(
            marker in str(goal or "").casefold()
            for marker in _VISUAL_GOAL_MARKERS
        )
        or visually_opaque_action_count >= _VISUALLY_OPAQUE_ACTION_THRESHOLD
    )
    selected = (
        _order_all_candidates(candidates)
        if include_all_nodes or _has_repeated_action_surface(candidates)
        else _select_candidates(candidates, max_nodes=max_nodes)
    )
    text, nodes = _render_candidates(
        selected,
        suppress_unlabeled_action_references=visual_context_required,
    )
    return UIProjection(
        text=text or "<none>",
        candidate_count=len(candidates),
        selected_count=len(selected),
        goal_match_count=sum(1 for item in candidates if item.goal_match),
        visual_context_required=visual_context_required,
        visual_candidate_count=sum(
            1 for item in selected if item.group in {"goal_control", "visual"}
        ),
        nodes=nodes,
    )


def _order_all_candidates(candidates: list[_Candidate]) -> list[_Candidate]:
    """Order the complete tree for the model without dropping any XML node."""

    return sorted(
        candidates,
        key=lambda item: (
            _GROUP_ORDER.index(item.group),
            _screen_order(item),
            item.order,
        ),
    )


def _has_repeated_action_surface(candidates: list[_Candidate]) -> bool:
    """Detect a grid/list surface whose unlabeled controls need visual context.

    Android views such as calendar day cells often expose every cell as a
    clickable node with the same resource id and no text/content description.
    Relevance ranking cannot infer which cell the goal names, so retaining only
    a handful of visual candidates makes the target unreachable.  This is an
    adaptive UI-shape rule, not an app- or task-specific exception.
    """
    unlabeled_actions = [
        item
        for item in candidates
        if item.compact.get("a")
        and not any(
            str(item.compact.get(key) or "").strip()
            for key in ("t", "d", "h", "c")
        )
    ]
    if len(unlabeled_actions) < 12:
        return False
    repeated_keys: dict[tuple[str, tuple[str, ...]], int] = {}
    for item in unlabeled_actions:
        key = (
            str(item.compact.get("r") or ""),
            tuple(str(action) for action in (item.compact.get("a") or ())),
        )
        repeated_keys[key] = repeated_keys.get(key, 0) + 1
    return max(repeated_keys.values(), default=0) >= 8


def projected_node_center(
    projection: UIProjection,
    target_description: str,
) -> tuple[ProjectedNode, tuple[float, float]] | None:
    target = _normalized_label(target_description)
    if not target:
        return None
    reference_match = re.search(r"(?<![a-z0-9])a\d{2,}(?![a-z0-9])", target)
    if reference_match is None or reference_match.group(0) != target:
        return None
    reference = reference_match.group(0).upper()
    matches = [
        node
        for node in projection.nodes
        if node.reference == reference and not node.inside_webview
    ]
    if len(matches) != 1:
        return None
    node = matches[0]
    left, top, right, bottom = node.bounds
    return node, ((left + right) / 2, (top + bottom) / 2)


def projected_numeric_summary_center(
    projection: UIProjection,
    summary: str,
) -> tuple[ProjectedNode, tuple[float, float], str] | None:
    """Resolve an explicitly named numeric control from a model action summary."""
    label_match = _NUMERIC_CONTROL_SUMMARY.search(str(summary or ""))
    if label_match is None:
        return None
    label = label_match.group(1)
    matches = [
        node
        for node in projection.nodes
        if not node.inside_webview
        and label in {_normalized_label(value) for value in node.labels}
    ]
    if len(matches) != 1:
        return None
    node = matches[0]
    left, top, right, bottom = node.bounds
    return node, ((left + right) / 2, (top + bottom) / 2), label


def _promote_goal_controls(candidates: list[_Candidate]) -> list[_Candidate]:
    goal_bounds = [
        item.bounds
        for item in candidates
        if item.group == "goal" and item.bounds is not None
    ]
    if not goal_bounds:
        return candidates
    promoted: list[_Candidate] = []
    for item in candidates:
        if item.group != "visual" or item.bounds is None:
            promoted.append(item)
            continue
        proximity = min(_rectangle_gap(item.bounds, target) for target in goal_bounds)
        if proximity > 180:
            promoted.append(item)
            continue
        promoted.append(
            replace(
                item,
                group="goal_control",
                score=item.score + 2000 - proximity * 5,
            )
        )
    return promoted


def _prune_redundant_candidates(
    candidates: list[_Candidate],
    goal: str,
) -> list[_Candidate]:
    """Keep the encoded view action-centric without inventing a second encoder."""

    goal_text = _normalized_label(goal)
    browse_requested = any(marker in goal_text for marker in _BROWSE_CONTROL_MARKERS)
    actionable_goal = [
        item
        for item in candidates
        if item.group == "goal" and item.compact.get("a")
    ]
    direct_goal = [
        item
        for item in actionable_goal
        if not _is_browse_candidate(item)
        and set(item.compact.get("a") or ()) != {"scroll"}
    ]
    kept: list[_Candidate] = []
    for item in candidates:
        if _is_system_chrome_context(item):
            continue
        if (
            direct_goal
            and not browse_requested
            and item.group == "goal"
            and _is_browse_candidate(item)
        ):
            continue
        if _covered_scroll_container(item, direct_goal):
            continue
        if _covered_by_actionable_candidate(item, actionable_goal):
            continue
        kept.append(item)
    return kept


def _is_system_chrome_context(item: _Candidate) -> bool:
    return (
        not item.compact.get("a")
        and str(item.compact.get("r") or "").startswith("com.android.systemui:")
    )


def _covered_scroll_container(
    item: _Candidate,
    direct_goal: list[_Candidate],
) -> bool:
    if set(item.compact.get("a") or ()) != {"scroll"} or item.bounds is None:
        return False
    item_text = _normalized_label(str(item.compact.get("c") or ""))
    return any(
        action.bounds is not None
        and _bounds_contain(item.bounds, action.bounds)
        and item_text
        and _normalized_label(str(action.compact.get("c") or "")) in item_text
        for action in direct_goal
    )


def _is_browse_candidate(item: _Candidate) -> bool:
    text = _normalized_label(
        " ".join(
            str(item.compact.get(key) or "")
            for key in ("t", "d", "h", "c")
        )
    )
    return any(marker in text for marker in _BROWSE_CONTROL_MARKERS)


def _covered_by_actionable_candidate(
    item: _Candidate,
    actionable_goal: list[_Candidate],
) -> bool:
    if item.compact.get("a") or item.bounds is None:
        return False
    item_text = _normalized_label(
        " ".join(
            str(item.compact.get(key) or "") for key in ("t", "d", "h")
        )
    )
    if not item_text:
        return False
    for action in actionable_goal:
        if action.bounds is None or not _bounds_contain(action.bounds, item.bounds):
            continue
        action_text = _normalized_label(str(action.compact.get("c") or ""))
        if item_text in action_text:
            return True
    return False


def _bounds_contain(
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def _select_candidates(
    candidates: list[_Candidate],
    *,
    max_nodes: int,
) -> list[_Candidate]:
    if max_nodes <= 0:
        return []
    selected: list[_Candidate] = []
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
    return sorted(
        selected,
        key=lambda item: (
            _GROUP_ORDER.index(item.group),
            _screen_order(item),
            item.order,
        ),
    )


def _render_candidates(
    candidates: list[_Candidate],
    *,
    suppress_unlabeled_action_references: bool = False,
) -> tuple[str, tuple[ProjectedNode, ...]]:
    lines: list[str] = []
    nodes: list[ProjectedNode] = []
    visual_reference = 0
    for group in _GROUP_ORDER:
        group_candidates = [item for item in candidates if item.group == group]
        if not group_candidates:
            continue
        lines.append(_GROUP_HEADERS[group])
        for item in group_candidates:
            compact = dict(item.compact)
            reference = ""
            visible_label = any(
                str(compact.get(key) or "").strip()
                for key in ("t", "d", "h", "c")
            )
            if compact.get("a") and not (
                suppress_unlabeled_action_references
                and item.group in {"goal_control", "visual"}
                and not visible_label
            ):
                visual_reference += 1
                reference = f"A{visual_reference:02d}"
                compact = {"v": reference, **compact}
            labels = tuple(
                str(compact[key])
                for key in ("t", "d", "h", "c", "r")
                if str(compact.get(key) or "").strip()
            )
            if item.bounds is not None and (reference or labels):
                nodes.append(
                    ProjectedNode(
                        reference=reference,
                        labels=labels,
                        bounds=item.bounds,
                        inside_webview=item.inside_webview,
                    )
                )
            lines.append(
                json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
            )
    return "\n".join(lines), tuple(nodes)


def _webview_elements(root: ET.Element) -> set[int]:
    result: set[int] = set()

    def visit(element: ET.Element, inside_webview: bool) -> None:
        class_name = str(element.attrib.get("class") or "").casefold()
        current_inside = inside_webview or "webview" in class_name
        if current_inside:
            result.add(id(element))
        for child in element:
            visit(child, current_inside)

    visit(root, False)
    return result


def _normalized_label(value: str) -> str:
    return " ".join(str(value or "").casefold().split())


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
    semantic_tokens = _ENGLISH_TOKEN.findall(semantic_text)
    for marker in _GLOBAL_CONTROL_MARKERS:
        marker_tokens = _ENGLISH_TOKEN.findall(marker.casefold())
        if marker_tokens:
            width = len(marker_tokens)
            if any(
                semantic_tokens[index : index + width] == marker_tokens
                for index in range(len(semantic_tokens) - width + 1)
            ):
                return True
        elif marker in semantic_text:
            return True
    return False


def _descendant_context(element: ET.Element) -> str:
    values: list[str] = []
    for descendant in element.iter():
        if descendant is element:
            continue
        for attribute in ("text", "content-desc", "hint-text"):
            value = str(descendant.attrib.get(attribute) or "").strip()
            if value and value not in values:
                values.append(value)
                if len(values) == 2:
                    break
        if len(values) == 2:
            break
    return " | ".join(values)[:80]


def _visual_specificity(bounds: tuple[int, int, int, int] | None) -> int:
    if bounds is None:
        return 0
    left, top, right, bottom = bounds
    width = right - left
    height = bottom - top
    longest = max(width, height)
    score = 500 if longest <= 160 else 300 if longest <= 320 else 100 if longest <= 640 else 0
    if min(width, height) / longest >= 0.7:
        score += 100
    return score


def _rectangle_gap(
    left_bounds: tuple[int, int, int, int],
    right_bounds: tuple[int, int, int, int],
) -> int:
    left_left, left_top, left_right, left_bottom = left_bounds
    right_left, right_top, right_right, right_bottom = right_bounds
    horizontal = max(right_left - left_right, left_left - right_right, 0)
    vertical = max(right_top - left_bottom, left_top - right_bottom, 0)
    return horizontal + vertical


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


__all__ = [
    "ProjectedNode",
    "UIProjection",
    "project_ui",
    "projected_numeric_summary_center",
    "projected_node_center",
]
