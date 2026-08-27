from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any
import xml.etree.ElementTree as ET

from omniflow.core.model import Observation, TransferResult


MINIMUM_CONTEXTUAL_MAPPING_CONFIDENCE = 0.8
_CONTEXTUAL_MAPPING_TOOLS = frozenset({"click", "long_press", "input_text", "swipe"})
_SEMANTIC_SCROLL_GESTURES = {
    "down": (500.0, 500.0, 500.0, 0.0),
    "up": (500.0, 500.0, 500.0, 1000.0),
    "right": (500.0, 500.0, 0.0, 500.0),
    "left": (500.0, 500.0, 1000.0, 500.0),
}


@dataclass(frozen=True)
class TransferAdmission:
    accepted: bool
    reason: str | None
    confidence: float | None


def assess_transfer(
    transfer: TransferResult,
    *,
    observation: Observation | None = None,
    minimum_confidence: float = MINIMUM_CONTEXTUAL_MAPPING_CONFIDENCE,
) -> TransferAdmission:
    """Apply the one fail-closed admission policy for mapped actions."""

    if transfer.action is None:
        return TransferAdmission(
            False,
            transfer.reason or "omnitransfer_null_target",
            _mapping_confidence(transfer.detail),
        )
    if not requires_contextual_mapping(
        transfer.action.tool,
        transfer.action.args,
    ):
        return TransferAdmission(True, None, 1.0)
    confidence = _mapping_confidence(transfer.detail)
    if confidence is None:
        return TransferAdmission(False, "omnitransfer_confidence_missing", None)
    if confidence < float(minimum_confidence):
        if _exact_executable_identity_match(transfer.detail):
            return TransferAdmission(True, None, confidence)
        return TransferAdmission(False, "omnitransfer_low_confidence", confidence)
    if observation is not None and not _target_is_executable(
        transfer,
        observation,
    ):
        return TransferAdmission(
            False,
            "omnitransfer_target_not_executable",
            confidence,
        )
    if not _target_semantics_match(transfer.detail):
        return TransferAdmission(
            False,
            "omnitransfer_target_semantics_mismatch",
            confidence,
        )
    return TransferAdmission(True, None, confidence)


def requires_contextual_mapping(
    tool: str,
    args: dict[str, Any] | None = None,
) -> bool:
    normalized = str(tool).strip()
    if normalized == "swipe" and _is_semantic_scroll(args):
        return False
    return normalized in _CONTEXTUAL_MAPPING_TOOLS


def _is_semantic_scroll(args: dict[str, Any] | None) -> bool:
    params = args if isinstance(args, dict) else {}
    direction = str(params.get("direction") or "").strip().lower()
    return direction in _SEMANTIC_SCROLL_GESTURES


def _mapping_confidence(detail: dict[str, Any]) -> float | None:
    raw: Any = None
    for key in (
        "absolute_contextual_confidence",
        "pair_confidence",
        "score",
    ):
        if detail.get(key) is not None:
            raw = detail[key]
            break
    try:
        confidence = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(confidence):
        return None
    return min(1.0, max(0.0, confidence))


def _target_is_executable(
    transfer: TransferResult,
    observation: Observation,
) -> bool:
    action = transfer.action
    if action is None:
        return False
    if action.tool not in {"click", "long_press", "input_text", "swipe"}:
        return True
    if action.tool == "swipe":
        return _canonical_coordinates_valid(action.args, ("x1", "y1", "x2", "y2"))
    if not _canonical_coordinates_valid(action.args, ("x", "y")):
        return False
    point = _raw_point(action.args, observation)
    if point is None:
        return False
    try:
        root = ET.fromstring(str(observation.xml or ""))
    except ET.ParseError:
        return False
    matching = [
        element
        for element in root.iter()
        if _contains(_bounds(element.attrib.get("bounds")), point)
    ]
    if not matching:
        return False
    enabled = [
        element
        for element in matching
        if str(element.attrib.get("enabled", "true")).lower() != "false"
    ]
    if action.tool == "input_text":
        return any(
            str(element.attrib.get("editable", "false")).lower() == "true"
            or "edittext" in str(element.attrib.get("class") or "").lower()
            for element in enabled
        )
    attribute = "long-clickable" if action.tool == "long_press" else "clickable"
    return any(
        str(element.attrib.get(attribute, "false")).lower() == "true"
        for element in enabled
    )


def _target_semantics_match(detail: dict[str, Any]) -> bool:
    """Reject a clickable top candidate that loses a stable source identity."""

    source = detail.get("source")
    candidates = detail.get("candidates")
    if not isinstance(source, dict) or not isinstance(candidates, list) or not candidates:
        return True
    target = candidates[0]
    if not isinstance(target, dict):
        return True

    if _candidate_semantics_match(source, target):
        return True
    if _execution_wrapper_matches_source(source, target, detail):
        return True
    mapped_bounds = _candidate_bounds(detail.get("target"))
    if mapped_bounds is None:
        return False
    # OmniTransfer may return wrapper nodes before the actionable child when
    # they share the same bounds. Keep the hard semantic gate, but allow the
    # compatible child only for that exact mapped target rectangle.
    return any(
        isinstance(candidate, dict)
        and _candidate_bounds(candidate) == mapped_bounds
        and _candidate_semantics_match(source, candidate)
        for candidate in candidates[1:]
    )


def _execution_wrapper_matches_source(
    source: dict[str, Any],
    target: dict[str, Any],
    detail: dict[str, Any],
) -> bool:
    source_resource = _semantic_value(source.get("resource_id"))
    execution_resource = _semantic_value(target.get("execution_candidate_id"))
    if not source_resource or source_resource != execution_resource:
        return False
    if target.get("executable") is not True:
        return False
    execution_bounds = _candidate_bounds(
        {"bounds": target.get("execution_bounds")}
    )
    mapped_bounds = _candidate_bounds(detail.get("target"))
    return execution_bounds is not None and execution_bounds == mapped_bounds


def _exact_executable_identity_match(detail: dict[str, Any]) -> bool:
    """Allow a low-probability match only with an explicit safe identity."""

    source = detail.get("source")
    candidates = detail.get("candidates")
    if not isinstance(source, dict) or not isinstance(candidates, list) or not candidates:
        return False
    target = candidates[0]
    if not isinstance(target, dict) or target.get("executable") is not True:
        return False
    source_resource = _semantic_value(source.get("resource_id"))
    target_resource = _semantic_value(target.get("resource_id"))
    return bool(source_resource and source_resource == target_resource)


def _candidate_semantics_match(
    source: dict[str, Any],
    target: dict[str, Any],
) -> bool:
    source_resource = _semantic_value(source.get("resource_id"))
    target_resource = _semantic_value(target.get("resource_id"))
    if source_resource and target_resource:
        return source_resource == target_resource

    source_editable = str(source.get("editable") or "").casefold()
    target_editable = str(target.get("editable") or "").casefold()
    if source_editable == "false" and target_editable == "true":
        return False
    source_class = _semantic_value(source.get("class"))
    target_class = _semantic_value(target.get("class"))
    if "edittext" in target_class and "edittext" not in source_class:
        return False

    source_labels = {
        value
        for value in (
            _semantic_value(source.get("text")),
            _semantic_value(source.get("content_desc")),
        )
        if value
    }
    if not source_labels:
        return not source_resource
    target_labels = {
        value
        for value in (
            _semantic_value(target.get("text")),
            _semantic_value(target.get("content_desc")),
        )
        if value
    }
    return bool(source_labels & target_labels)


def _candidate_bounds(value: Any) -> tuple[float, float, float, float] | None:
    raw = value.get("bounds") if isinstance(value, dict) else None
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        result = tuple(float(item) for item in raw)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in result):
        return None
    return result  # type: ignore[return-value]


def _semantic_value(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _canonical_coordinates_valid(args: dict[str, Any], keys: tuple[str, ...]) -> bool:
    try:
        values = [float(args[key]) for key in keys]
    except (KeyError, TypeError, ValueError):
        return False
    return all(math.isfinite(value) and 0.0 <= value <= 1000.0 for value in values)


def _raw_point(
    args: dict[str, Any],
    observation: Observation,
) -> tuple[float, float] | None:
    display = observation.extra.get("display")
    if not isinstance(display, dict):
        return None
    try:
        width = float(display["width"])
        height = float(display["height"])
        x = float(args["x"]) / 1000.0 * width
        y = float(args["y"]) / 1000.0 * height
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0.0 or height <= 0.0:
        return None
    return x, y


def _bounds(value: Any) -> tuple[float, float, float, float] | None:
    values = re.findall(r"-?\d+(?:\.\d+)?", str(value or ""))
    if len(values) != 4:
        return None
    left, top, right, bottom = (float(item) for item in values)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _contains(
    bounds: tuple[float, float, float, float] | None,
    point: tuple[float, float],
) -> bool:
    if bounds is None:
        return False
    return bounds[0] <= point[0] <= bounds[2] and bounds[1] <= point[1] <= bounds[3]


__all__ = [
    "MINIMUM_CONTEXTUAL_MAPPING_CONFIDENCE",
    "TransferAdmission",
    "assess_transfer",
    "requires_contextual_mapping",
]
