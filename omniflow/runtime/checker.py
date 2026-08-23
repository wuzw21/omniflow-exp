from __future__ import annotations

import ast
from typing import Any
import xml.etree.ElementTree as ET

from omniflow.core.schemas import canonicalize_action

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


def validate_checker_rule(value: Any) -> dict[str, Any]:
    """Validate the canonical Checker v1 contract used by OmniFlow."""

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
    return {
        "schema_version": "omniflow.checker_rule.v1",
        "trigger": trigger,
        "source_state_id": source_state_id,
        "action": canonicalize_action(value.get("action"), replayable_only=True),
    }


def checker_rule_matches(rule: dict[str, Any], observation: Any) -> bool:
    canonical = validate_checker_rule(rule)
    return _evaluate_trigger_node(
        _parse_trigger(canonical["trigger"]).body,
        _TriggerFacts(observation),
    )


class _TriggerFacts:
    def __init__(self, observation: Any) -> None:
        self.package_name = _normalize(getattr(observation, "package_name", ""))
        self.activity_name = _normalize(getattr(observation, "activity_name", ""))
        self.xml = str(getattr(observation, "xml", "") or "")
        self.normalized_xml = _normalize(self.xml)
        self.values: dict[str, list[str]] = {
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


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


__all__ = ["checker_rule_matches", "validate_checker_rule"]
