from __future__ import annotations

import ast
from dataclasses import dataclass
import importlib
import json
import math
import re
from typing import Any
import xml.etree.ElementTree as ET

from omniflow.model import Action, CheckerContext
from omniflow.schemas import canonicalize_action
from omniflow.transfer import load_omnitransfer

_RULE_FIELDS = {"schema_version", "trigger", "source_state_id", "action"}
_TRIGGER_HELPERS = {
    "activity_is",
    "content_desc_contains",
    "content_desc_is",
    "package_is",
    "resource_id_contains",
    "resource_id_is",
    "text_contains",
    "text_is",
    "xml_contains",
}
_MAX_TRIGGER_LENGTH = 512
_MAX_TRIGGER_NODES = 64

_TRANSIENT_PACKAGES = {
    "com.android.systemui",
}
_TRANSIENT_PACKAGE_PARTS = (
    "packageinstaller",
    "permissioncontroller",
)
_EXPLICIT_LABELS = (
    "关闭广告",
    "跳过广告",
    "close ad",
    "close advertisement",
    "skip ad",
    "skip advertisement",
)
_EXACT_SKIP_LABELS = {"跳过", "skip"}
_GENERIC_CLOSE_LABELS = {"关闭", "close", "×", "✕"}
_AD_WORDS = {"ad", "ads", "advert", "advertisement", "sponsored"}
_CLOSE_WORDS = {"close", "dismiss", "skip"}


@dataclass(frozen=True)
class CheckerRecovery:
    action: Action
    source_state_id: str
    trigger: str


def validate_checker_rule(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _RULE_FIELDS:
        raise ValueError("checker_rule_contract_invalid")
    if value.get("schema_version") != "omniflow.checker_rule.v1":
        raise ValueError("unsupported_checker_rule_version")
    trigger = str(value.get("trigger") or "").strip()
    if not trigger:
        raise ValueError("checker_trigger_required")
    _parse_trigger(trigger)
    source_state_id = str(value.get("source_state_id") or "").strip()
    if not source_state_id:
        raise ValueError("checker_source_state_id_required")
    action = canonicalize_action(value.get("action"), replayable_only=True)
    return {
        "schema_version": "omniflow.checker_rule.v1",
        "trigger": trigger,
        "source_state_id": source_state_id,
        "action": action,
    }


def match_checker_rule(
    context: CheckerContext,
    rules: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> CheckerRecovery | None:
    facts = _TriggerFacts(context.current)
    for raw_rule in rules:
        rule = validate_checker_rule(raw_rule)
        if _evaluate_trigger(rule["trigger"], facts):
            return CheckerRecovery(
                Action.from_value(rule["action"]),
                rule["source_state_id"],
                rule["trigger"],
            )
    return None


def default_checker(context: CheckerContext) -> Action | None:
    if context.action.tool == "open_app":
        return None
    source_package = str(context.source.package_name or "") if context.source else ""
    current_package = str(context.current.package_name or "")
    if (
        source_package
        and current_package
        and source_package != current_package
        and not _is_transient_package(source_package)
        and not _is_transient_package(current_package)
    ):
        return Action("open_app", {"package_name": source_package})
    return _advertisement_recovery(context.current, context.action)


def default_checker_trigger(
    context: CheckerContext,
    recovery_action: Action,
) -> str | None:
    current_package = str(context.current.package_name or "").strip()
    if recovery_action.tool == "open_app" and current_package:
        return f"package_is({json.dumps(current_package, ensure_ascii=False)})"
    if recovery_action.tool != "click":
        return None
    normalized_xml = _normalize(context.current.xml)
    for marker in (*_EXPLICIT_LABELS, *_EXACT_SKIP_LABELS):
        if marker in normalized_xml:
            return f"xml_contains({json.dumps(marker, ensure_ascii=False)})"
    if "广告" in normalized_xml and any(
        marker in normalized_xml for marker in _GENERIC_CLOSE_LABELS
    ):
        return 'xml_contains("广告") and xml_contains("关闭")'
    if any(marker in normalized_xml for marker in _AD_WORDS) and any(
        marker in normalized_xml for marker in _CLOSE_WORDS
    ):
        return 'xml_contains("ad") and xml_contains("close")'
    return None


class _TriggerFacts:
    def __init__(self, observation: Any) -> None:
        self.package_name = _normalize(observation.package_name)
        self.activity_name = _normalize(observation.activity_name)
        self.xml = str(observation.xml or "")
        self.normalized_xml = _normalize(self.xml)
        self.values = {
            "text": [],
            "content-desc": [],
            "resource-id": [],
        }
        if self.xml:
            try:
                root = ET.fromstring(self.xml)
            except ET.ParseError:
                root = None
            if root is not None:
                for node in root.iter():
                    for field in self.values:
                        value = _normalize(node.attrib.get(field))
                        if value:
                            self.values[field].append(value)

    def call(self, name: str, needles: tuple[str, ...]) -> bool:
        normalized = tuple(_normalize(value) for value in needles if _normalize(value))
        if not normalized:
            return False
        if name == "package_is":
            return self.package_name in normalized
        if name == "activity_is":
            return self.activity_name in normalized
        if name == "xml_contains":
            return any(value in self.normalized_xml for value in normalized)
        field, operation = name.rsplit("_", 1)
        key = {"content_desc": "content-desc", "resource_id": "resource-id"}.get(
            field, field
        )
        candidates = self.values.get(key, ())
        if operation == "is":
            return any(candidate in normalized for candidate in candidates)
        return any(
            needle in candidate for candidate in candidates for needle in normalized
        )


def _parse_trigger(trigger: str) -> ast.Expression:
    if len(trigger) > _MAX_TRIGGER_LENGTH:
        raise ValueError("checker_trigger_too_long")
    try:
        tree = ast.parse(trigger, mode="eval")
    except SyntaxError as error:
        raise ValueError("checker_trigger_syntax_invalid") from error
    nodes = list(ast.walk(tree))
    if len(nodes) > _MAX_TRIGGER_NODES:
        raise ValueError("checker_trigger_too_complex")
    for node in nodes:
        if isinstance(node, (ast.Expression, ast.Load, ast.And, ast.Or, ast.Not)):
            continue
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            continue
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            continue
        if isinstance(node, ast.Call):
            if (
                not isinstance(node.func, ast.Name)
                or node.func.id not in _TRIGGER_HELPERS
                or node.keywords
                or not node.args
            ):
                raise ValueError("checker_trigger_call_invalid")
            continue
        if isinstance(node, ast.Name) and node.id in _TRIGGER_HELPERS:
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            continue
        raise ValueError("checker_trigger_expression_invalid")
    return tree


def _evaluate_trigger(trigger: str, facts: _TriggerFacts) -> bool:
    return _evaluate_trigger_node(_parse_trigger(trigger).body, facts)


def _evaluate_trigger_node(node: ast.AST, facts: _TriggerFacts) -> bool:
    if isinstance(node, ast.BoolOp):
        values = (_evaluate_trigger_node(value, facts) for value in node.values)
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.UnaryOp):
        return not _evaluate_trigger_node(node.operand, facts)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return facts.call(
            node.func.id,
            tuple(str(arg.value) for arg in node.args if isinstance(arg, ast.Constant)),
        )
    raise ValueError("checker_trigger_expression_invalid")


def _advertisement_recovery(
    observation: Any,
    original_action: Action,
) -> Action | None:
    xml = str(observation.xml or "")
    if not xml or not _might_contain_ad_recovery(xml):
        return None
    try:
        load_omnitransfer()
        graph_from_record = importlib.import_module(
            "omnitransfer.ui_graph"
        ).graph_from_record
        graph = graph_from_record({"xml": xml}, graph_id="checker-current")
    except (AttributeError, ImportError, RuntimeError, SyntaxError, TypeError, ValueError):
        return None
    nodes_by_id = {node.node_id: node for node in graph.nodes}
    page_has_ad_evidence = any(_node_has_ad_evidence(node) for node in graph.nodes)
    candidates: list[tuple[int, float, Any]] = []
    seen_targets: set[str] = set()
    for node in graph.nodes:
        priority = _close_signal_priority(node, page_has_ad_evidence)
        if priority is None:
            continue
        target = _clickable_target(node, nodes_by_id)
        if target is None or target.node_id in seen_targets:
            continue
        seen_targets.add(target.node_id)
        area = _area(target.bbox)
        candidates.append((priority, area, target))
    if not candidates:
        return None
    _, _, target = min(candidates, key=lambda item: (item[0], item[1], item[2].node_id))
    point = _relative_center(target.bbox, graph.width, graph.height)
    if point is None or _original_targets(original_action, point):
        return None
    return Action(
        "click",
        {
            "target_description": "关闭广告",
            "x": point[0],
            "y": point[1],
        },
    )


def _is_transient_package(package_name: str) -> bool:
    normalized = package_name.casefold()
    return normalized in _TRANSIENT_PACKAGES or any(
        part in normalized for part in _TRANSIENT_PACKAGE_PARTS
    )


def _might_contain_ad_recovery(xml: str) -> bool:
    normalized = xml.casefold()
    return any(
        marker in normalized
        for marker in (
            "广告",
            "advert",
            "sponsored",
            "ad_close",
            "close_ad",
            "ad_skip",
            "skip_ad",
            'text="跳过"',
            'content-desc="跳过"',
            'text="skip"',
            'content-desc="skip"',
        )
    )


def _close_signal_priority(node: Any, page_has_ad_evidence: bool) -> int | None:
    labels = tuple(
        normalized
        for normalized in (_normalize(node.text), _normalize(node.content_desc))
        if normalized
    )
    if any(
        label in _EXACT_SKIP_LABELS
        or any(explicit in label for explicit in _EXPLICIT_LABELS)
        for label in labels
    ):
        return 0
    if _resource_has_ad_close_signal(node.resource_id):
        return 1
    if page_has_ad_evidence and any(label in _GENERIC_CLOSE_LABELS for label in labels):
        return 2
    return None


def _node_has_ad_evidence(node: Any) -> bool:
    labels = (_normalize(node.text), _normalize(node.content_desc))
    return any(
        label == "ad"
        or "广告" in label
        or "advertisement" in label
        or "sponsored" in label
        for label in labels
    ) or _resource_has_ad_word(node.resource_id)


def _resource_has_ad_close_signal(resource_id: str) -> bool:
    words = _resource_words(resource_id)
    normalized = _normalize(resource_id)
    return (
        bool(words & _AD_WORDS)
        and bool(words & _CLOSE_WORDS)
    ) or any(
        marker in normalized
        for marker in ("adclose", "closead", "adskip", "skipad")
    )


def _resource_has_ad_word(resource_id: str) -> bool:
    words = _resource_words(resource_id)
    normalized = _normalize(resource_id)
    return bool(words & _AD_WORDS) or any(
        marker in normalized for marker in ("adclose", "closead", "adskip", "skipad")
    )


def _resource_words(value: str) -> set[str]:
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(value or ""))
    return set(re.findall(r"[a-z]+", separated.casefold()))


def _clickable_target(node: Any, nodes_by_id: dict[str, Any]) -> Any | None:
    current = node
    visited: set[str] = set()
    while current is not None and current.node_id not in visited:
        visited.add(current.node_id)
        if current.enabled and current.clickable and current.bbox is not None:
            return current
        current = nodes_by_id.get(current.parent_id)
    return None


def _relative_center(
    bounds: tuple[float, float, float, float] | None,
    width: float | None,
    height: float | None,
) -> tuple[float, float] | None:
    if bounds is None or not width or not height or width <= 0 or height <= 0:
        return None
    x = (bounds[0] + bounds[2]) / (2.0 * width) * 1000.0
    y = (bounds[1] + bounds[3]) / (2.0 * height) * 1000.0
    return max(0.0, min(1000.0, x)), max(0.0, min(1000.0, y))


def _original_targets(action: Action, point: tuple[float, float]) -> bool:
    if action.tool != "click":
        return False
    description = _normalize(action.args.get("target_description"))
    if (
        description in _EXACT_SKIP_LABELS
        or description in _GENERIC_CLOSE_LABELS
        or any(explicit in description for explicit in _EXPLICIT_LABELS)
    ):
        return True
    try:
        x = float(action.args["x"])
        y = float(action.args["y"])
    except (KeyError, TypeError, ValueError):
        return False
    return math.hypot(x - point[0], y - point[1]) <= 36.0


def _area(bounds: tuple[float, float, float, float] | None) -> float:
    if bounds is None:
        return math.inf
    return max(0.0, bounds[2] - bounds[0]) * max(0.0, bounds[3] - bounds[1])


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())
