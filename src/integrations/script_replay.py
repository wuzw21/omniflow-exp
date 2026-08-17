"""Deterministic MobileGPT-style semantic script replay baseline.

The baseline adapts a source click through text/content-description and local
tree structure.  Resource IDs, source-device coordinates, models, checkers,
and execution fallbacks are deliberately excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import time
from typing import Any, Mapping
from xml.etree import ElementTree

_BOUNDS = re.compile(
    r"^\[(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\]"
    r"\[(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\]$"
)
_SEMANTIC_FIELDS = ("text", "content-desc")
_LOCAL_DEPTH = 3


@dataclass(frozen=True)
class ScriptReplayResult:
    success: bool
    function_id: str
    actions_executed: int
    error: str | None
    trace: tuple[dict[str, Any], ...]

    @property
    def execution_summary(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "replay_completed": self.success,
            "actions_executed": self.actions_executed,
            "model_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "fallback_steps": 0,
            **({"failure_reason": self.error} if self.error else {}),
        }


def run_script_replay(
    *,
    store_path: str | Path,
    source_states: Mapping[str, Mapping[str, Any]],
    host: Any,
    post_action_wait_seconds: float = 0.0,
    stability_wait_seconds: float = 0.5,
) -> ScriptReplayResult:
    """Replay one Function through unique, stable semantic UI locators."""

    function = _only_visible_function(_read_store(store_path))
    function_id = str(function.get("function_id") or "").strip()
    trace: list[dict[str, Any]] = []
    actions_executed = 0
    steps = sorted(
        (step for step in function.get("steps") or () if isinstance(step, dict)),
        key=lambda step: int(step.get("step_index") or 0),
    )
    for step in steps:
        if str(step.get("role") or "").strip().lower() == "checker":
            trace.append(
                {
                    "step_index": int(step.get("step_index") or 0),
                    "status": "ignored_checker",
                }
            )
            continue
        action = step.get("action")
        action = dict(action) if isinstance(action, dict) else {}
        tool = str(action.get("tool") or "").strip()
        args = dict(action.get("args") or {}) if isinstance(action.get("args"), dict) else {}
        record: dict[str, Any] = {
            "step_index": int(step.get("step_index") or 0),
            "tool": tool,
        }
        try:
            if tool in {"click", "long_press"}:
                source_state_id = str(step.get("source_state_id") or "").strip()
                source_state = source_states.get(source_state_id)
                if not isinstance(source_state, Mapping):
                    raise ValueError(f"script_replay_source_state_missing:{source_state_id}")
                observation, target_xml = _stable_observation(
                    host, wait_seconds=stability_wait_seconds
                )
                source_root, source_node = _source_target_node(
                    str(source_state.get("xml") or ""),
                    x=args.get("x"),
                    y=args.get("y"),
                )
                matched, selector, candidates = _unique_semantic_target(
                    source_root,
                    source_node,
                    target_xml,
                )
                record["selector"] = selector
                record["selector_candidates"] = candidates
                left, top, right, bottom = _node_bounds(matched)
                width, height = _target_extent(observation, target_xml)
                target_action = {
                    "tool": tool,
                    "args": {
                        "x": ((left + right) / 2.0) / width * 1000.0,
                        "y": ((top + bottom) / 2.0) / height * 1000.0,
                    },
                }
            elif tool in {"press_back", "press_home", "press_key", "wait", "open_app"}:
                target_action = {"tool": tool, "args": args}
            else:
                raise ValueError(f"script_replay_unsupported_action:{tool or 'missing'}")
            action_result = host.act(target_action)
            if getattr(action_result, "success", False) is not True:
                raise RuntimeError(
                    str(getattr(action_result, "error", "") or "script_replay_action_failed")
                )
            actions_executed += 1
            record["status"] = "executed"
            record["action"] = target_action
            trace.append(record)
            if post_action_wait_seconds > 0:
                time.sleep(float(post_action_wait_seconds))
        except Exception as error:  # noqa: BLE001 - baseline must fail closed
            record["status"] = "failed"
            record["error"] = str(error)
            trace.append(record)
            return ScriptReplayResult(
                False,
                function_id,
                actions_executed,
                str(error),
                tuple(trace),
            )
    return ScriptReplayResult(True, function_id, actions_executed, None, tuple(trace))


def _read_store(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != "omniflow.store.v2":
        raise ValueError("script_replay_store_v2_required")
    return value


def _only_visible_function(store: Mapping[str, Any]) -> dict[str, Any]:
    functions = store.get("functions")
    functions = functions if isinstance(functions, Mapping) else {}
    visible = [
        dict(function)
        for function in functions.values()
        if isinstance(function, Mapping) and function.get("agent_visible") is not False
    ]
    if len(visible) != 1:
        raise ValueError(f"script_replay_requires_one_function:{len(visible)}")
    return visible[0]


def _source_target_node(
    xml: str, *, x: Any, y: Any
) -> tuple[ElementTree.Element, ElementTree.Element]:
    root = _xml_root(xml, error="script_replay_source_xml_invalid")
    width, height = _xml_extent(root)
    try:
        point_x = float(x) / 1000.0 * width
        point_y = float(y) / 1000.0 * height
    except (TypeError, ValueError) as error:
        raise ValueError("script_replay_source_point_invalid") from error
    parent_map = _parent_map(root)
    hits: list[tuple[float, int, ElementTree.Element]] = []
    for node in root.iter():
        if not _has_semantic_locator(node, parent_map):
            continue
        try:
            left, top, right, bottom = _node_bounds(node)
        except ValueError:
            continue
        if left <= point_x <= right and top <= point_y <= bottom:
            clickable_rank = 0 if node.attrib.get("clickable") == "true" else 1
            hits.append(((right - left) * (bottom - top), clickable_rank, node))
    if not hits:
        raise ValueError("script_replay_source_semantic_locator_missing")
    hits.sort(key=lambda item: (item[0], item[1]))
    return root, hits[0][2]


def _unique_semantic_target(
    source_root: ElementTree.Element,
    source_node: ElementTree.Element,
    target_xml: str,
) -> tuple[ElementTree.Element, dict[str, Any], dict[str, int]]:
    root = _xml_root(target_xml, error="script_replay_target_xml_invalid")
    source_package = str(source_node.attrib.get("package") or "").strip()
    source_class = str(source_node.attrib.get("class") or "").strip()
    candidate_counts: dict[str, int] = {}
    ambiguous = False

    # MobileGPT's first locator: direct semantic attribute.
    for field, value in _semantic_values(source_node):
        matches = [
            node
            for node in root.iter()
            if str(node.attrib.get(field) or "").strip() == value
            and _target_candidate(
                node,
                source_package=source_package,
                source_class=source_class,
                constrain_class=False,
            )
        ]
        candidate_counts[field] = len(matches)
        if len(matches) == 1:
            return matches[0], {"kind": field, "value": value}, candidate_counts
        ambiguous = ambiguous or len(matches) > 1

    # MobileGPT's second locator: semantic children at the same depth/rank.
    anchors = _descendant_anchors(source_node)
    if anchors:
        matches = [
            node
            for node in root.iter()
            if _target_candidate(
                node,
                source_package=source_package,
                source_class=source_class,
                constrain_class=True,
            )
            and _matches_descendant_anchors(node, anchors)
        ]
        candidate_counts["children"] = len(matches)
        if len(matches) == 1:
            selector = {
                "kind": "children",
                "anchors": [
                    {"path": list(path), "field": field, "value": value}
                    for path, field, value in anchors
                ],
            }
            return matches[0], selector, candidate_counts
        ambiguous = ambiguous or len(matches) > 1

    # MobileGPT's third locator: semantic parent plus the source child rank.
    source_parent = _parent_map(source_root).get(source_node)
    if source_parent is not None:
        source_siblings = list(source_parent)
        child_rank = source_siblings.index(source_node)
        for field, value in _semantic_values(source_parent):
            matches = []
            for target_parent in root.iter():
                if str(target_parent.attrib.get(field) or "").strip() != value:
                    continue
                children = list(target_parent)
                if child_rank >= len(children):
                    continue
                node = children[child_rank]
                if _target_candidate(
                    node,
                    source_package=source_package,
                    source_class=source_class,
                    constrain_class=True,
                ):
                    matches.append(node)
            key = f"parent_{field}"
            candidate_counts[key] = len(matches)
            if len(matches) == 1:
                return (
                    matches[0],
                    {
                        "kind": "parent",
                        "field": field,
                        "value": value,
                        "child_rank": child_rank,
                    },
                    candidate_counts,
                )
            ambiguous = ambiguous or len(matches) > 1

    failure = "ambiguous" if ambiguous else "absent"
    raise ValueError(f"script_replay_semantic_locator_{failure}:{candidate_counts}")


def _semantic_values(node: ElementTree.Element) -> tuple[tuple[str, str], ...]:
    return tuple(
        (field, value)
        for field in _SEMANTIC_FIELDS
        if (value := str(node.attrib.get(field) or "").strip())
    )


def _parent_map(root: ElementTree.Element) -> dict[ElementTree.Element, ElementTree.Element]:
    return {child: parent for parent in root.iter() for child in parent}


def _has_semantic_locator(
    node: ElementTree.Element,
    parent_map: Mapping[ElementTree.Element, ElementTree.Element],
) -> bool:
    if _semantic_values(node) or _descendant_anchors(node):
        return True
    parent = parent_map.get(node)
    return parent is not None and bool(_semantic_values(parent))


def _descendant_anchors(
    node: ElementTree.Element,
) -> tuple[tuple[tuple[int, ...], str, str], ...]:
    anchors: list[tuple[tuple[int, ...], str, str]] = []

    def visit(parent: ElementTree.Element, path: tuple[int, ...]) -> None:
        if len(path) >= _LOCAL_DEPTH:
            return
        for rank, child in enumerate(parent):
            child_path = (*path, rank)
            for field, value in _semantic_values(child):
                anchors.append((child_path, field, value))
            visit(child, child_path)

    visit(node, ())
    return tuple(anchors)


def _node_at_path(
    node: ElementTree.Element, path: tuple[int, ...]
) -> ElementTree.Element | None:
    current = node
    for rank in path:
        children = list(current)
        if rank >= len(children):
            return None
        current = children[rank]
    return current


def _matches_descendant_anchors(
    node: ElementTree.Element,
    anchors: tuple[tuple[tuple[int, ...], str, str], ...],
) -> bool:
    for path, field, value in anchors:
        descendant = _node_at_path(node, path)
        if descendant is None:
            return False
        if str(descendant.attrib.get(field) or "").strip() != value:
            return False
    return True


def _target_candidate(
    node: ElementTree.Element,
    *,
    source_package: str,
    source_class: str,
    constrain_class: bool,
) -> bool:
    if source_package and str(node.attrib.get("package") or "").strip() != source_package:
        return False
    if constrain_class and source_class:
        if str(node.attrib.get("class") or "").strip() != source_class:
            return False
    if node.attrib.get("enabled") == "false" or node.attrib.get("displayed") == "false":
        return False
    try:
        _node_bounds(node)
    except ValueError:
        return False
    return True


def _stable_observation(host: Any, *, wait_seconds: float) -> tuple[Any, str]:
    first = host.observe(xml=True, screenshot=False, app_info=True)
    first_xml = str(getattr(first, "xml", "") or "")
    first_root = _xml_root(first_xml, error="script_replay_target_xml_invalid")
    if _page_has_loading_indicator(first_root):
        raise ValueError("script_replay_page_loading")
    if wait_seconds > 0:
        time.sleep(float(wait_seconds))
    second = host.observe(xml=True, screenshot=False, app_info=True)
    second_xml = str(getattr(second, "xml", "") or "")
    second_root = _xml_root(second_xml, error="script_replay_target_xml_invalid")
    if _page_has_loading_indicator(second_root):
        raise ValueError("script_replay_page_loading")
    if _semantic_page_signature(first_root) != _semantic_page_signature(second_root):
        raise ValueError("script_replay_page_unstable")
    return second, second_xml


def _page_has_loading_indicator(root: ElementTree.Element) -> bool:
    for node in root.iter():
        if node.attrib.get("displayed") == "false":
            continue
        class_name = str(node.attrib.get("class") or "").lower()
        if "progressbar" in class_name or "progressindicator" in class_name:
            return True
    return False


def _semantic_page_signature(root: ElementTree.Element) -> tuple[Any, ...]:
    def signature(node: ElementTree.Element) -> tuple[Any, ...]:
        attributes = tuple(
            (field, str(node.attrib.get(field) or "").strip())
            for field in (
                "package",
                "class",
                "text",
                "content-desc",
                "clickable",
                "long-clickable",
                "scrollable",
                "enabled",
                "displayed",
            )
        )
        return node.tag, attributes, tuple(signature(child) for child in node)

    return signature(root)


def _xml_root(xml: str, *, error: str) -> ElementTree.Element:
    try:
        return ElementTree.fromstring(str(xml or ""))
    except ElementTree.ParseError as parse_error:
        raise ValueError(error) from parse_error


def _node_bounds(node: ElementTree.Element) -> tuple[float, float, float, float]:
    match = _BOUNDS.fullmatch(str(node.attrib.get("bounds") or "").strip())
    if match is None:
        raise ValueError("script_replay_bounds_missing")
    left, top, right, bottom = (float(value) for value in match.groups())
    if right <= left or bottom <= top:
        raise ValueError("script_replay_bounds_invalid")
    return left, top, right, bottom


def _xml_extent(root: ElementTree.Element) -> tuple[float, float]:
    try:
        width = float(root.attrib.get("width") or 0)
        height = float(root.attrib.get("height") or 0)
    except (TypeError, ValueError):
        width = height = 0
    if width > 0 and height > 0:
        return width, height
    bounds = []
    for node in root.iter():
        try:
            bounds.append(_node_bounds(node))
        except ValueError:
            continue
    if not bounds:
        raise ValueError("script_replay_xml_extent_missing")
    return max(item[2] for item in bounds), max(item[3] for item in bounds)


def _target_extent(observation: Any, xml: str) -> tuple[float, float]:
    extra = getattr(observation, "extra", None)
    extra = extra if isinstance(extra, Mapping) else {}
    display = extra.get("display")
    display = display if isinstance(display, Mapping) else {}
    try:
        width = float(display.get("width") or 0)
        height = float(display.get("height") or 0)
    except (TypeError, ValueError):
        width = height = 0
    return (width, height) if width > 0 and height > 0 else _xml_extent(_xml_root(xml, error="script_replay_target_xml_invalid"))
