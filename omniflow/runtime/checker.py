from __future__ import annotations

import ast
from dataclasses import dataclass
import importlib
import json
import math
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET

from omniflow.core.model import Action, CheckerContext
from omniflow.core.schemas import canonicalize_action
from omniflow.transfer.runtime import load_omnitransfer

CHECKER_STORE_VERSION = "omniflow.checker_store.v1"
DEFAULT_CHECKER_LIBRARY_PATH = (
    Path(__file__).resolve().parents[1] / "checkers" / "default.json"
)
SHARED_CHECKER_LIBRARY_REFERENCE = "omniflow/checkers/default.json"
_LEGACY_RULE_FIELDS = {"schema_version", "trigger", "source_state_id", "action"}
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
_TRANSIENT_DISMISS_LABELS = {
    "以后再说",
    "稍后",
    "暂不",
    "不了，谢谢",
    "not now",
    "no thanks",
}
_EXACT_SKIP_LABELS = {"跳过", "skip"}
_GENERIC_CLOSE_LABELS = {"关闭", "close", "×", "✕"}
_AD_WORDS = {"ad", "ads", "advert", "advertisement", "sponsored"}
_CLOSE_WORDS = {"close", "dismiss", "skip"}


@dataclass(frozen=True)
class CheckerRecovery:
    action: Action
    source_state_id: str
    trigger: str


@dataclass(frozen=True)
class CheckerLibrary:
    rules: tuple[dict[str, Any], ...] = ()

    @classmethod
    def load(cls, path: str | Path | None = None) -> "CheckerLibrary":
        merged: dict[str, dict[str, Any]] = {}
        for candidate in (DEFAULT_CHECKER_LIBRARY_PATH, Path(path) if path else None):
            if candidate is None or not candidate.is_file():
                continue
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("checker_store_must_be_object")
            version = payload.get("schema_version")
            if version not in (None, CHECKER_STORE_VERSION):
                raise ValueError("unsupported_checker_store_version")
            raw_rules = payload.get("checker_rules")
            if not isinstance(raw_rules, list):
                raise ValueError("checker_store_rules_must_be_array")
            for raw_rule in raw_rules:
                rule = validate_checker_rule(raw_rule)
                merged[rule["id"]] = rule
        return cls(tuple(merged.values()))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": CHECKER_STORE_VERSION,
            "checker_rules": [dict(rule) for rule in self.rules],
        }
        temporary = destination.with_suffix(f"{destination.suffix}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)


def validate_checker_rule(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("checker_rule_contract_invalid")
    rule_id = str(value.get("id") or "").strip()
    if not rule_id:
        raise ValueError("checker_rule_id_required")
    version = value.get("schema_version", "omniflow.checker_rule.v1")
    if version != "omniflow.checker_rule.v1":
        raise ValueError("unsupported_checker_rule_version")
    phase = str(value.get("phase") or "pre_transfer").strip()
    if phase not in {"pre_transfer", "pre_action", "post_action"}:
        raise ValueError("checker_rule_phase_invalid")
    condition = value.get("condition", value.get("when"))
    action = value.get("action", value.get("then"))
    condition = _normalize_library_condition(condition)
    action = _normalize_library_action(action)
    scope = value.get("scope") or {}
    budget = value.get("budget") or {}
    if not isinstance(scope, dict):
        raise ValueError("checker_rule_scope_invalid")
    if not isinstance(budget, dict):
        raise ValueError("checker_rule_budget_invalid")
    normalized_budget: dict[str, int] = {}
    for name in ("max_triggers_per_run", "max_triggers_per_step", "cooldown_ms"):
        if name not in budget:
            continue
        raw = budget[name]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ValueError(f"checker_rule_budget_invalid:{name}")
        normalized_budget[name] = raw
    return {
        "schema_version": "omniflow.checker_rule.v1",
        "id": rule_id,
        "enabled": value.get("enabled") is not False,
        "phase": phase,
        "scope": json.loads(json.dumps(scope, ensure_ascii=False)),
        "condition": condition,
        "action": action,
        "budget": normalized_budget,
        "priority": int(value.get("priority") or 0),
        "source": str(value.get("source") or "runtime_policy"),
    }


def validate_legacy_checker_rule(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _LEGACY_RULE_FIELDS:
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
        rule = validate_legacy_checker_rule(raw_rule)
        if _evaluate_trigger(rule["trigger"], facts):
            return CheckerRecovery(
                Action.from_value(rule["action"]),
                rule["source_state_id"],
                rule["trigger"],
            )
    return None


def checker_rule_matches(
    rule: dict[str, Any],
    *,
    current: Any,
    source: Any | None,
    function_id: str,
    step_index: int,
    action: Action,
    transfer_failed: bool = False,
) -> bool:
    normalized = validate_checker_rule(rule)
    if not normalized["enabled"] or not _scope_matches(
        normalized["scope"],
        current=current,
        function_id=function_id,
        step_index=step_index,
        action=action,
    ):
        return False
    condition = normalized["condition"]
    kind = condition["type"]
    if kind == "ui_unstable":
        return _ui_is_unstable(current)
    if kind == "package_mismatch":
        source_package = _observation_package(source)
        current_package = _observation_package(current)
        return bool(
            source_package
            and current_package
            and source_package != current_package
            and not is_transient_package(source_package)
            and not is_transient_package(current_package)
        )
    if kind == "keyboard_obscuring":
        extra = getattr(current, "extra", {}) or {}
        if extra.get("keyboard_visible") is True or extra.get("ime_visible") is True:
            keyboard_visible = True
        else:
            package_name = _normalize(getattr(current, "package_name", ""))
            activity_name = _normalize(getattr(current, "activity_name", ""))
            xml = _normalize(getattr(current, "xml", ""))
            # The IME is often embedded in the app's accessibility hierarchy while
            # the reported foreground package remains the app (for example Contacts
            # with Gboard open).  Package-only detection therefore misses exactly
            # the state in which a mapped form-field click can land on the keyboard.
            keyboard_visible = any(
                marker in value
                for value in (package_name, activity_name, xml)
                for marker in (
                    "inputmethod",
                    "softinputwindow",
                    "com.google.android.inputmethod",
                    "com.android.inputmethod",
                    "key_pos_",
                )
            )
        if not keyboard_visible:
            return False
        # Hiding the IME proactively can change the meaning of a later Back
        # action.  The shared recovery is therefore admitted only after this
        # exact action has failed Transfer; the action is then retried with
        # the same semantics.
        return bool(transfer_failed)
    xpath = str(condition.get("xpath") or "")
    return bool(_xpath_nodes(str(getattr(current, "xml", "") or ""), xpath))


def checker_rule_action(
    rule: dict[str, Any],
    *,
    current: Any,
    source: Any | None,
) -> Action | None:
    normalized = validate_checker_rule(rule)
    specification = normalized["action"]
    kind = specification["type"]
    if kind == "open_app":
        package_name = str(specification.get("package_name") or "").strip()
        if not package_name and source is not None:
            package_name = _observation_package(source)
        return Action("open_app", {"package_name": package_name}) if package_name else None
    if kind == "hide_keyboard":
        return Action("press_key", {"key": "back"})
    if kind == "wait":
        return Action("wait", {"duration_ms": int(specification.get("wait_ms") or 0)})
    if kind != "click":
        return None
    nodes = _xpath_nodes(
        str(getattr(current, "xml", "") or ""),
        str(specification.get("target_xpath") or ""),
    )
    if len(nodes) != 1:
        return None
    bounds = _parse_bounds(nodes[0].get("bounds"))
    display = (getattr(current, "extra", {}) or {}).get("display")
    if bounds is None or not isinstance(display, dict):
        return None
    try:
        width = float(display["width"])
        height = float(display["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return Action(
        "click",
        {
            "x": (bounds[0] + bounds[2]) / 2.0 / width * 1000.0,
            "y": (bounds[1] + bounds[3]) / 2.0 / height * 1000.0,
        },
    )


def _normalize_library_condition(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = {"type": value}
    if not isinstance(value, dict):
        raise ValueError("checker_rule_condition_invalid")
    if value.get("type"):
        kind = str(value["type"])
    elif value.get("xpath_exists"):
        kind = "xpath_exists"
    elif value.get("target_covered_by_xpath"):
        kind = "target_covered_by_xpath"
    elif value.get("keyboard_obscuring") is True or value.get("keyboard_obscures_target") is True:
        kind = "keyboard_obscuring"
    elif value.get("package_mismatch") is True:
        kind = "package_mismatch"
    elif value.get("ui_unstable") is True:
        kind = "ui_unstable"
    else:
        raise ValueError("checker_rule_condition_invalid")
    if kind not in {
        "xpath_exists",
        "target_covered_by_xpath",
        "keyboard_obscuring",
        "package_mismatch",
        "ui_unstable",
    }:
        raise ValueError("checker_rule_condition_invalid")
    result = {"type": kind}
    if kind in {"xpath_exists", "target_covered_by_xpath"}:
        xpath = str(
            value.get("xpath")
            or value.get("xpath_exists")
            or value.get("target_covered_by_xpath")
            or ""
        ).strip()
        if not xpath:
            raise ValueError("checker_rule_xpath_required")
        result["xpath"] = xpath
    return result


def _ui_is_unstable(observation: Any) -> bool:
    """Read only explicit stabilization signals; never infer instability from XML."""
    extra = getattr(observation, "extra", {}) or {}
    if extra.get("ui_stable") is False or extra.get("state_stable") is False:
        return True
    stabilization = extra.get("stabilization")
    if isinstance(stabilization, dict):
        status = str(stabilization.get("status") or stabilization.get("state") or "")
        return status.casefold() in {"pending", "unstable", "not_stable"}
    return False


def _normalize_library_action(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = {"type": value}
    if not isinstance(value, dict):
        raise ValueError("checker_rule_action_invalid")
    kind = str(value.get("type") or value.get("action") or "").strip()
    if kind not in {"click", "hide_keyboard", "open_app", "wait"}:
        raise ValueError("checker_rule_action_invalid")
    result: dict[str, Any] = {"type": kind}
    if kind == "click":
        target_xpath = str(value.get("target_xpath") or "").strip()
        if not target_xpath:
            raise ValueError("checker_rule_target_xpath_required")
        result["target_xpath"] = target_xpath
    if kind == "open_app" and str(value.get("package_name") or "").strip():
        result["package_name"] = str(value["package_name"]).strip()
    if kind == "wait":
        result["wait_ms"] = int(value.get("wait_ms") or 0)
    return result


def _scope_matches(
    scope: dict[str, Any],
    *,
    current: Any,
    function_id: str,
    step_index: int,
    action: Action,
) -> bool:
    checks = {
        "function_ids": function_id,
        "step_indexes": step_index,
        "action_types": action.tool,
        "package_names": str(getattr(current, "package_name", "") or ""),
    }
    for name, actual in checks.items():
        expected = scope.get(name)
        if expected is None:
            continue
        values = expected if isinstance(expected, list) else [expected]
        if actual not in values:
            return False
    return True


def _xpath_nodes(xml: str, xpath: str) -> list[Any]:
    if not xml or not xpath:
        return []
    try:
        from lxml import etree

        root = etree.fromstring(xml.encode("utf-8"))
        return [node for node in root.xpath(xpath) if hasattr(node, "get")]
    except Exception:
        return []


def _parse_bounds(value: Any) -> tuple[float, float, float, float] | None:
    match = re.fullmatch(
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
        str(value or ""),
    )
    if match is None:
        return None
    left, top, right, bottom = map(float, match.groups())
    return (left, top, right, bottom) if right > left and bottom > top else None


def default_checker(context: CheckerContext) -> Action | None:
    if context.action.tool in {"open_app", "press_key"}:
        return None
    source_package = str(context.source.package_name or "") if context.source else ""
    current_package = str(context.current.package_name or "")
    if (
        source_package
        and current_package
        and source_package != current_package
        and not is_transient_package(source_package)
        and not is_transient_package(current_package)
    ):
        return Action("open_app", {"package_name": source_package})
    return _advertisement_recovery(context.current, context.action)


def transient_obstruction_recovery(observation: Any) -> Action | None:
    return _transient_recovery(observation, original_action=None)


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
    return _transient_recovery(observation, original_action=original_action)


def _transient_recovery(
    observation: Any,
    *,
    original_action: Action | None,
) -> Action | None:
    xml = str(observation.xml or "")
    if not xml or not _might_contain_transient_recovery(xml):
        return None
    try:
        load_omnitransfer()
        graph_from_record = importlib.import_module(
            "omnitransfer.ui_graph"
        ).graph_from_record
        graph = graph_from_record({"xml": xml}, graph_id="checker-current")
    except (
        AttributeError,
        ImportError,
        RuntimeError,
        SyntaxError,
        TypeError,
        ValueError,
    ):
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
    if point is None or (
        original_action is not None and _original_targets(original_action, point)
    ):
        return None
    return Action(
        "click",
        {"x": point[0], "y": point[1]},
    )


def is_transient_package(package_name: str) -> bool:
    """Return whether a foreground package is a system obstruction layer."""

    normalized = str(package_name or "").casefold()
    return normalized in _TRANSIENT_PACKAGES or any(
        part in normalized for part in _TRANSIENT_PACKAGE_PARTS
    )


def _might_contain_transient_recovery(xml: str) -> bool:
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
            *_TRANSIENT_DISMISS_LABELS,
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
    if any(label in _TRANSIENT_DISMISS_LABELS for label in labels):
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
    return (bool(words & _AD_WORDS) and bool(words & _CLOSE_WORDS)) or any(
        marker in normalized for marker in ("adclose", "closead", "adskip", "skipad")
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


def _observation_package(observation: Any | None) -> str:
    """Return the reported package, falling back to the XML's main package."""

    if observation is None:
        return ""
    reported = _normalize(getattr(observation, "package_name", ""))
    if reported:
        return reported
    xml = str(getattr(observation, "xml", "") or "")
    if not xml:
        return ""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return ""
    counts: dict[str, int] = {}
    for node in root.iter():
        package = _normalize(node.attrib.get("package"))
        if not package or package == "com.android.systemui":
            continue
        counts[package] = counts.get(package, 0) + 1
    return max(counts, key=counts.get) if counts else ""
