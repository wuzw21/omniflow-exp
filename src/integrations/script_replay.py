"""Deterministic exact-selector replay baseline.

The baseline reuses only stable selectors from source click targets.  It never
uses coordinates from the source device, fuzzy matching, a model, a checker,
or an execution fallback.
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
_SELECTOR_FIELDS = ("resource-id", "text", "content-desc")


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
) -> ScriptReplayResult:
    """Replay the only visible Function using unique exact UI selectors."""

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
                observation = host.observe(xml=True, screenshot=False, app_info=True)
                target_xml = str(getattr(observation, "xml", "") or "")
                source_node = _source_target_node(
                    str(source_state.get("xml") or ""),
                    x=args.get("x"),
                    y=args.get("y"),
                )
                matched, selector, candidates = _unique_exact_target(
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


def _source_target_node(xml: str, *, x: Any, y: Any) -> ElementTree.Element:
    root = _xml_root(xml, error="script_replay_source_xml_invalid")
    width, height = _xml_extent(root)
    try:
        point_x = float(x) / 1000.0 * width
        point_y = float(y) / 1000.0 * height
    except (TypeError, ValueError) as error:
        raise ValueError("script_replay_source_point_invalid") from error
    hits: list[tuple[float, int, ElementTree.Element]] = []
    for node in root.iter():
        if not _stable_values(node):
            continue
        try:
            left, top, right, bottom = _node_bounds(node)
        except ValueError:
            continue
        if left <= point_x <= right and top <= point_y <= bottom:
            clickable_rank = 0 if node.attrib.get("clickable") == "true" else 1
            hits.append(((right - left) * (bottom - top), clickable_rank, node))
    if not hits:
        raise ValueError("script_replay_source_selector_missing")
    hits.sort(key=lambda item: (item[0], item[1]))
    return hits[0][2]


def _unique_exact_target(
    source_node: ElementTree.Element,
    target_xml: str,
) -> tuple[ElementTree.Element, dict[str, str], dict[str, int]]:
    root = _xml_root(target_xml, error="script_replay_target_xml_invalid")
    source_package = str(source_node.attrib.get("package") or "").strip()
    candidate_counts: dict[str, int] = {}
    for field, value in _stable_values(source_node):
        matches = []
        for node in root.iter():
            if str(node.attrib.get(field) or "").strip() != value:
                continue
            if source_package and str(node.attrib.get("package") or "").strip() != source_package:
                continue
            if node.attrib.get("enabled") == "false" or node.attrib.get("displayed") == "false":
                continue
            try:
                _node_bounds(node)
            except ValueError:
                continue
            matches.append(node)
        candidate_counts[field] = len(matches)
        if len(matches) == 1:
            return matches[0], {"kind": field, "value": value}, candidate_counts
    if any(count > 1 for count in candidate_counts.values()):
        raise ValueError(f"script_replay_selector_ambiguous:{candidate_counts}")
    raise ValueError(f"script_replay_selector_absent:{candidate_counts}")


def _stable_values(node: ElementTree.Element) -> tuple[tuple[str, str], ...]:
    return tuple(
        (field, value)
        for field in _SELECTOR_FIELDS
        if (value := str(node.attrib.get(field) or "").strip())
    )


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
