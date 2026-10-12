"""Attribute-matching transfer: the MobileGPT-style same-harness baseline.

MobileGPT replays a remembered sub-task by re-locating each recorded UI element
on the new screen through its attributes (resource-id, text, content-desc) and
never re-plans the element choice.  This Transfer reproduces that policy inside
the OmniFlow harness so the only difference to OmniTransfer is the mapper.
"""

from __future__ import annotations

from typing import Any

from omniflow.core.model import Action, Observation, TransferResult
from omniflow.runtime.execution import _display_size, _elements, _relative_source_point

_POINTER_TOOLS = frozenset({"click", "input_text", "long_press"})


def attribute_transfer(
    action: Action,
    observation: Observation,
    source_state: Observation | None = None,
) -> TransferResult:
    if action.tool not in _POINTER_TOOLS:
        return TransferResult(action)
    if source_state is None or not source_state.xml:
        return TransferResult(None, reason="attribute_source_state_missing")
    try:
        point = _relative_source_point(source_state, float(action.args["x"]), float(action.args["y"]))
    except (KeyError, TypeError, ValueError):
        return TransferResult(None, reason="attribute_invalid_source_point")
    source = _element_at(_elements(str(source_state.xml)), point)
    if source is None:
        return TransferResult(None, reason="attribute_source_element_missing")
    targets = _elements(str(observation.xml or ""))
    display = _display_size(observation, targets)
    if display is None:
        return TransferResult(None, reason="attribute_display_size_missing")
    match, key = _match(source, targets, point)
    if match is None:
        return TransferResult(None, reason="attribute_no_match", detail={"source": _identity(source)})
    left, top, right, bottom = match["bounds"]
    params = {k: v for k, v in action.args.items() if k not in {"node_id", "node_resource_id"}}
    params["x"] = (left + right) / 2 / display[0] * 1000.0
    params["y"] = (top + bottom) / 2 / display[1] * 1000.0
    return TransferResult(
        Action(action.tool, params),
        reason=f"attribute_match_{key}",
        detail={"source": _identity(source), "target": _identity(match)},
    )


def _element_at(elements: list[dict[str, Any]], point: tuple[float, float]) -> dict[str, Any] | None:
    x, y = point
    hits = [e for e in elements if e["bounds"][0] <= x <= e["bounds"][2] and e["bounds"][1] <= y <= e["bounds"][3]]
    labelled = [e for e in hits if e["resource_id"] or e["text"] or e["description"]]
    pool = labelled or hits
    return min(pool, key=_area) if pool else None


def _match(
    source: dict[str, Any], targets: list[dict[str, Any]], point: tuple[float, float],
) -> tuple[dict[str, Any] | None, str]:
    rid, text, desc = source["resource_id"], source["text"], source["description"]
    rules = (
        ("resource_id_text", lambda e: rid and e["resource_id"] == rid and e["text"] == text),
        ("resource_id", lambda e: rid and e["resource_id"] == rid),
        ("text", lambda e: text and e["text"] == text),
        ("description", lambda e: desc and e["description"] == desc),
    )
    for key, rule in rules:
        hits = [e for e in targets if rule(e)]
        if hits:
            return min(hits, key=lambda e: (_distance(e, point), _area(e))), key
    return None, ""


def _area(element: dict[str, Any]) -> int:
    left, top, right, bottom = element["bounds"]
    return (right - left) * (bottom - top)


def _distance(element: dict[str, Any], point: tuple[float, float]) -> float:
    left, top, right, bottom = element["bounds"]
    return abs((left + right) / 2 - point[0]) + abs((top + bottom) / 2 - point[1])


def _identity(element: dict[str, Any]) -> dict[str, str]:
    return {k: element[k] for k in ("resource_id", "text", "description", "class")}


__all__ = ["attribute_transfer"]
