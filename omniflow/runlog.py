from __future__ import annotations

import json
import re
from typing import Any
import xml.etree.ElementTree as ET

from omniflow.core.schemas import canonicalize_action
from omniflow.core.trajectory import (
    canonicalize_run_log,
    observation_display,
    observation_screenshot,
    observation_xml,
    state_id,
)


def import_run_log_evidence(
    value: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_log = _hydrate_run_log_display(canonicalize_run_log(value))
    states: dict[str, dict[str, Any]] = {}
    # Prefer screenshots captured immediately before an action over transition
    # aliases of the same structural state.  Stable observations can assign the
    # same state id to both records while persisting each image at a new path.
    for step in run_log["steps"]:
        _store_transfer_state(states, step["observation"])
    for step in run_log["steps"]:
        if isinstance(step.get("next_observation"), dict):
            _store_transfer_state(states, step["next_observation"])
    final_observation = run_log.get("final_observation")
    if isinstance(final_observation, dict):
        _store_transfer_state(states, final_observation)
    return run_log, {
        "schema_version": "omniflow.transfer-state-catalog.v1",
        "run_id": run_log["run_id"],
        "states": states,
    }


def project_androidworld_step_actions(
    value: dict[str, Any],
    *,
    previous_step: dict[str, Any] | None = None,
    next_observation: dict[str, Any] | None = None,
    execution_action: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or not isinstance(value.get("observation"), dict):
        raise ValueError("androidworld_run_log_step_required")
    action = dict(value.get("action") or {})
    observation = value["observation"]
    projected_action = _androidworld_action_to_omniflow(
        action,
        observation=observation,
        next_observation=next_observation,
        execution_action=execution_action,
    )
    return [projected_action]


def project_run_log_step_actions(
    run_log: dict[str, Any],
    source_step_index: int,
    *,
    previous_step: dict[str, Any] | None = None,
    next_observation: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Project one RunLog step through its canonical execution evidence."""

    steps = run_log.get("steps")
    if not isinstance(steps, list) or not 0 <= source_step_index < len(steps):
        raise ValueError("androidworld_run_log_step_index_invalid")
    step = steps[source_step_index]
    if not isinstance(step, dict):
        raise ValueError("androidworld_run_log_step_required")
    return project_androidworld_step_actions(
        step,
        previous_step=previous_step,
        next_observation=(
            next_observation
            if isinstance(next_observation, dict)
            else step.get("next_observation")
            if isinstance(step.get("next_observation"), dict)
            else None
        ),
        execution_action=_run_log_execution_action(run_log, source_step_index),
    )


def _run_log_execution_action(
    run_log: dict[str, Any],
    source_step_index: int,
) -> dict[str, Any] | None:
    source_step = run_log["steps"][source_step_index]
    step_metadata = source_step.get("metadata")
    if isinstance(step_metadata, dict):
        execution_action = step_metadata.get("execution_action")
        if (
            isinstance(execution_action, dict)
            and str(execution_action.get("tool") or "").strip()
            and isinstance(execution_action.get("args"), dict)
        ):
            # Interactive source collection records the exact action sent to
            # OOB.  Keep it as execution provenance so replay can preserve
            # real swipe/input geometry while the public AndroidWorld action
            # schema remains unchanged.
            return {
                "tool": str(execution_action["tool"]),
                "args": dict(execution_action["args"]),
            }
    diagnostics = run_log.get("diagnostics")
    trace = diagnostics.get("execution_trace") if isinstance(diagnostics, dict) else None
    if not isinstance(trace, list):
        return None
    indexed: dict[int, dict[str, Any]] = {}
    for fallback_index, item in enumerate(trace):
        if not isinstance(item, dict):
            continue
        raw_index = item.get("step_index", fallback_index)
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            continue
        if raw_index in indexed:
            raise ValueError(f"source_execution_trace_step_duplicate:{raw_index}")
        indexed[raw_index] = item
    trace_step = indexed.get(source_step_index)
    if trace_step is None:
        return None
    source_action = run_log["steps"][source_step_index].get("action")
    if not isinstance(source_action, dict):
        return None
    trace_action = trace_step.get("action")
    trace_result = trace_step.get("result")
    trace_args = trace_action.get("args") if isinstance(trace_action, dict) else None
    if source_action.get("action_type") == "open_app":
        if (
            isinstance(trace_action, dict)
            and trace_action.get("tool") == "open_app"
            and isinstance(trace_args, dict)
            and str(trace_args.get("package_name") or "").strip()
        ):
            return trace_action
        return None
    if source_action.get("action_type") not in {"scroll", "swipe"}:
        return None
    coordinate_keys = ("x1", "y1", "x2", "y2")
    if (
        isinstance(trace_action, dict)
        and trace_action.get("tool") == "swipe"
        and isinstance(trace_result, dict)
        and trace_result.get("success") is True
        and isinstance(trace_args, dict)
        and all(trace_args.get(key) is not None for key in coordinate_keys)
    ):
        return trace_action
    if all(source_action.get(key) is not None for key in coordinate_keys):
        return None
    raise ValueError(
        f"source_swipe_execution_evidence_required:{source_step_index}"
    )


def _store_transfer_state(
    states: dict[str, dict[str, Any]],
    observation: dict[str, Any],
) -> None:
    source_state = _transfer_state(observation)
    existing = states.get(source_state["state_id"])
    if existing is None:
        states[source_state["state_id"]] = source_state
        return
    comparable_existing = {
        key: value for key, value in existing.items() if key != "screenshot_path"
    }
    comparable_source = {
        key: value for key, value in source_state.items() if key != "screenshot_path"
    }
    if comparable_existing != comparable_source:
        raise ValueError(f"source_state_conflict:{source_state['state_id']}")
    if "screenshot_path" not in existing and source_state.get("screenshot_path"):
        existing["screenshot_path"] = source_state["screenshot_path"]


def _hydrate_run_log_display(run_log: dict[str, Any]) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    for step in run_log["steps"]:
        observations.append(step["observation"])
        next_observation = step.get("next_observation")
        if isinstance(next_observation, dict):
            observations.append(next_observation)
    final_observation = run_log.get("final_observation")
    if isinstance(final_observation, dict):
        observations.append(final_observation)

    explicit_displays = {
        display
        for observation in observations
        if (display := observation_display(observation)) is not None
    }
    if len(explicit_displays) > 1:
        raise ValueError("androidworld_run_log_display_conflict")
    if explicit_displays:
        width, height = next(iter(explicit_displays))
    else:
        xml_displays = {
            display
            for observation in observations
            if (display := _fullscreen_xml_display(observation_xml(observation)))
            is not None
        }
        if len(xml_displays) > 1:
            raise ValueError("androidworld_run_log_display_conflict")
        if not xml_displays:
            return run_log
        width, height = next(iter(xml_displays))

    for observation in observations:
        if observation_display(observation) is not None:
            continue
        auxiliaries = dict(observation.get("auxiliaries") or {})
        auxiliaries["display"] = {"width": width, "height": height}
        observation["auxiliaries"] = auxiliaries
    return canonicalize_run_log(run_log)


def _fullscreen_xml_display(xml: str) -> tuple[int, int] | None:
    if not xml.strip():
        return None
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None
    try:
        root_width = int(root.attrib.get("width") or 0)
        root_height = int(root.attrib.get("height") or 0)
    except (TypeError, ValueError):
        root_width, root_height = 0, 0
    if root_width > 0 and root_height > 0:
        return root_width, root_height
    first_node = next(root.iter("node"), None)
    if first_node is None:
        return None
    match = re.fullmatch(
        r"\[0,0\]\[(\d+),(\d+)\]",
        str(first_node.attrib.get("bounds") or ""),
    )
    if match is None:
        return None
    width, height = (int(item) for item in match.groups())
    return (width, height) if width > 0 and height > 0 else None


def _androidworld_input_target(
    action: dict[str, Any],
    observation: dict[str, Any],
    next_observation: dict[str, Any] | None = None,
) -> tuple[dict[str, float] | None, ET.Element | None]:
    display = _observation_display(observation)
    if display is None:
        return None, None
    try:
        root = ET.fromstring(observation_xml(observation))
    except ET.ParseError:
        return None, None
    editable_nodes = [
        node
        for node in root.iter()
        if _is_androidworld_input_node(node)
    ]
    transition_target = _changed_input_target(
        observation,
        next_observation,
        input_text=str(action.get("text") or ""),
    )
    if transition_target is not None:
        bounds = _parse_xml_bounds(transition_target.attrib.get("bounds"))
        if bounds is not None:
            left, top, right, bottom = bounds
            return (
                {
                    "x": (left + right) / 2.0 / display[0] * 1000.0,
                    "y": (top + bottom) / 2.0 / display[1] * 1000.0,
                },
                transition_target,
            )
    point = _androidworld_action_point(action, observation)
    if point is None:
        focused = [
            node
            for node in editable_nodes
            if str(node.attrib.get("focused") or "").casefold() == "true"
        ]
        node = min(focused, key=_xml_node_area, default=None)
        # WebView accessibility snapshots can keep focused=false immediately
        # after a successful click.  A sole editable node is still an
        # unambiguous input target; multiple unfocused fields remain rejected.
        if node is None and len(editable_nodes) == 1:
            node = editable_nodes[0]
        bounds = _parse_xml_bounds(node.attrib.get("bounds")) if node is not None else None
        if bounds is None:
            return None, None
        left, top, right, bottom = bounds
        return (
            {
                "x": (left + right) / 2.0 / display[0] * 1000.0,
                "y": (top + bottom) / 2.0 / display[1] * 1000.0,
            },
            node,
        )
    x = point["x"] / 1000.0 * display[0]
    y = point["y"] / 1000.0 * display[1]
    containing: list[ET.Element] = []
    for node in editable_nodes:
        bounds = _parse_xml_bounds(node.attrib.get("bounds"))
        if bounds is None or not (
            bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]
        ):
            continue
        containing.append(node)
    return point, min(containing, key=_xml_node_area, default=None)


def _changed_input_target(
    observation: dict[str, Any],
    next_observation: dict[str, Any] | None,
    *,
    input_text: str,
) -> ET.Element | None:
    """Find the editable node whose text changed after the recorded input."""

    if not isinstance(next_observation, dict):
        return None
    try:
        before_root = ET.fromstring(observation_xml(observation))
        after_root = ET.fromstring(observation_xml(next_observation))
    except ET.ParseError:
        return None

    def input_nodes(root: ET.Element) -> list[ET.Element]:
        return [
            node
            for node in root.iter()
            if _is_androidworld_input_node(node)
        ]

    before_nodes = input_nodes(before_root)
    after_nodes = input_nodes(after_root)
    if not before_nodes or not after_nodes:
        return None
    expected = " ".join(str(input_text or "").casefold().split())

    def normalized_text(node: ET.Element) -> str:
        return " ".join(str(node.attrib.get("text") or "").casefold().split())

    def compact_text(value: str) -> str:
        return "".join(character for character in value if character.isalnum())

    def node_key(node: ET.Element) -> tuple[str, str] | None:
        for key in ("resource-id", "id"):
            value = str(node.attrib.get(key) or "").strip()
            if value:
                return key, value
        return None

    after_by_key: dict[tuple[str, str], list[ET.Element]] = {}
    for node in after_nodes:
        key = node_key(node)
        if key is not None:
            after_by_key.setdefault(key, []).append(node)

    matches: list[ET.Element] = []
    for ordinal, before in enumerate(before_nodes):
        after: ET.Element | None = None
        key = node_key(before)
        keyed = after_by_key.get(key) if key is not None else None
        if keyed and len(keyed) == 1:
            after = keyed[0]
        elif len(before_nodes) == len(after_nodes):
            after = after_nodes[ordinal]
        if after is None or normalized_text(before) == normalized_text(after):
            continue
        after_text = normalized_text(after)
        if expected and compact_text(expected) not in compact_text(after_text):
            continue
        matches.append(before)
    return matches[0] if len(matches) == 1 else None


def _is_androidworld_input_node(node: ET.Element) -> bool:
    if str(node.attrib.get("editable") or "").casefold() == "true":
        return True
    class_name = str(node.attrib.get("class") or "")
    return class_name.endswith(("EditText", "AutoCompleteTextView"))


def _xml_node_area(node: ET.Element) -> float:
    bounds = _parse_xml_bounds(node.attrib.get("bounds"))
    if bounds is None:
        return float("inf")
    return (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])


def _androidworld_click_target(
    action: dict[str, Any],
    observation: dict[str, Any],
) -> ET.Element | None:
    """Find the smallest clickable native node containing a recorded click."""

    display = _observation_display(observation)
    point = _androidworld_action_point(action, observation)
    if display is None or point is None:
        return None
    try:
        root = ET.fromstring(observation_xml(observation))
    except ET.ParseError:
        return None
    x = point["x"] / 1000.0 * display[0]
    y = point["y"] / 1000.0 * display[1]
    containing: list[ET.Element] = []
    for node in root.iter():
        if str(node.attrib.get("clickable") or "").casefold() != "true":
            continue
        bounds = _parse_xml_bounds(node.attrib.get("bounds"))
        if bounds is None or not (
            bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]
        ):
            continue
        containing.append(node)
    return min(containing, key=_xml_node_area, default=None)


def _androidworld_action_to_omniflow(
    value: Any,
    *,
    observation: dict[str, Any],
    next_observation: dict[str, Any] | None = None,
    execution_action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    action = dict(value) if isinstance(value, dict) else {}
    action_type = str(action.get("action_type") or "").strip()
    if action_type in {"click", "double_tap"}:
        projected = {
            "tool": "click",
            "args": _required_androidworld_action_point(action, observation),
        }
    elif action_type == "long_press":
        projected = {
            "tool": "long_press",
            "args": _required_androidworld_action_point(action, observation),
        }
    elif action_type == "input_text":
        point, _target = _androidworld_input_target(
            action,
            observation,
            next_observation=next_observation,
        )
        args: dict[str, Any] = {"text": action.get("text", "")}
        if point is not None:
            args.update(point)
        projected = {"tool": "input_text", "args": args}
    elif action_type in {"scroll", "swipe"}:
        swipe_args: dict[str, Any] = {
            "direction": str(action.get("direction") or ""),
        }
        if action_type == "scroll":
            swipe_args.update(
                _androidworld_standard_swipe(
                    action_type,
                    str(action.get("direction") or ""),
                    observation=observation,
                )
            )
        else:
            exact_swipe = _execution_swipe(execution_action, source_action=action)
            if exact_swipe is None:
                exact_swipe = _androidworld_exact_swipe(action, observation)
            swipe_args.update(
                exact_swipe
                if exact_swipe is not None
                else _androidworld_standard_swipe(
                    action_type,
                    str(action.get("direction") or ""),
                    observation=observation,
                )
            )
        projected = {
            "tool": "swipe",
            "args": swipe_args,
        }
    elif action_type == "open_app":
        execution_args = (
            execution_action.get("args")
            if isinstance(execution_action, dict)
            and execution_action.get("tool") == "open_app"
            and isinstance(execution_action.get("args"), dict)
            else {}
        )
        projected = {
            "tool": "open_app",
            "args": {
                "package_name": str(
                    execution_args.get("package_name")
                    or action.get("package_name")
                    or action.get("app_name")
                    or ""
                )
            },
        }
    elif action_type == "navigate_back":
        projected = {"tool": "press_key", "args": {"key": "back"}}
    elif action_type == "navigate_home":
        projected = {"tool": "press_key", "args": {"key": "home"}}
    elif action_type == "keyboard_enter":
        projected = {"tool": "press_key", "args": {"key": "enter"}}
    elif action_type == "press_keyboard":
        keycode = str(action.get("keycode") or "").strip().upper()
        keycode = keycode.removeprefix("KEYCODE_")
        if keycode == "PASTE":
            return canonicalize_action(
                {"tool": "paste", "args": {}},
                replayable_only=True,
                allow_non_action=True,
            )
        key = {
            "DEL": "delete",
            "DELETE": "delete",
            "BACK": "back",
            "HOME": "home",
            "ENTER": "enter",
        }.get(keycode, keycode if keycode in set("0123456789") else "")
        if not key:
            raise ValueError(f"androidworld_press_keyboard_unsupported:{keycode}")
        projected = {"tool": "press_key", "args": {"key": key}}
    elif action_type == "wait":
        projected = {"tool": "wait", "args": {"duration_ms": 1000}}
    else:
        raise ValueError(f"androidworld_action_not_executable:{action_type}")
    return canonicalize_action(projected, replayable_only=True, allow_non_action=True)


def _required_androidworld_action_point(
    action: dict[str, Any],
    observation: dict[str, Any],
) -> dict[str, float]:
    point = _androidworld_action_point(action, observation)
    if point is None:
        raise ValueError("androidworld_action_point_not_transferable")
    return point


def _androidworld_action_point(
    action: dict[str, Any],
    observation: dict[str, Any],
) -> dict[str, float] | None:
    x = action.get("x")
    y = action.get("y")
    if x is None or y is None:
        index = action.get("index")
        elements = observation.get("ui_elements")
        bounds = None
        if (
            isinstance(index, int)
            and not isinstance(index, bool)
            and isinstance(elements, list)
            and 0 <= index < len(elements)
        ):
            bounds = _ui_element_bounds(elements[index])
        if bounds is None and isinstance(index, int) and not isinstance(index, bool):
            bounds = _xml_index_bounds(observation_xml(observation), index)
        if bounds is None:
            return None
        left, top, right, bottom = bounds
        x = (left + right) / 2.0
        y = (top + bottom) / 2.0
    display = _observation_display(observation)
    if display is None:
        raise ValueError("androidworld_action_display_required")
    width, height = display
    return {"x": float(x) / width * 1000.0, "y": float(y) / height * 1000.0}


def _observation_display(observation: dict[str, Any]) -> tuple[int, int] | None:
    return observation_display(observation) or _fullscreen_xml_display(
        observation_xml(observation)
    )


def _xml_index_bounds(xml: str, index: int) -> tuple[float, float, float, float] | None:
    if not xml.strip():
        return None
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None
    elements: list[ET.Element] = []
    windows = list(root.iter("window"))
    for window in windows:
        ordered_nodes: list[tuple[int, ET.Element]] = []
        for element in window.iter("node"):
            raw_id = str(element.attrib.get("id") or "")
            try:
                order = int(raw_id.rsplit(":", maxsplit=1)[1])
            except (IndexError, ValueError):
                return None
            ordered_nodes.append((order, element))
        for _, element in sorted(ordered_nodes, key=lambda item: item[0]):
            child_nodes = [child for child in element if child.tag == "node"]
            if (
                not child_nodes
                or str(element.attrib.get("content-desc") or "").strip()
                or str(element.attrib.get("scrollable") or "").casefold() == "true"
            ):
                elements.append(element)
    if not windows:
        top_nodes = [child for child in root if child.tag == "node"]
        if len(top_nodes) != 1:
            return None
        elements = list(top_nodes[0].iter("node"))[1:]
    if not 0 <= index < len(elements):
        return None
    return _parse_xml_bounds(elements[index].attrib.get("bounds"))


def _parse_xml_bounds(value: Any) -> tuple[float, float, float, float] | None:
    match = re.fullmatch(
        r"\[(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\]"
        r"\[(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\]",
        str(value or ""),
    )
    if match is None:
        return None
    left, top, right, bottom = map(float, match.groups())
    return (left, top, right, bottom) if right > left and bottom > top else None


def _ui_element_bounds(value: Any) -> tuple[float, float, float, float] | None:
    element = _map(value)
    bounds = _map(element.get("bbox_pixels")) or _map(element.get("bbox"))
    for keys in (
        ("x_min", "y_min", "x_max", "y_max"),
        ("left", "top", "right", "bottom"),
    ):
        try:
            left, top, right, bottom = (float(bounds[key]) for key in keys)
        except (KeyError, TypeError, ValueError):
            continue
        if right > left and bottom > top:
            return left, top, right, bottom
    return None


def _androidworld_standard_swipe(
    action_type: str,
    direction: str,
    *,
    observation: dict[str, Any] | None = None,
) -> dict[str, float]:
    scrollable_bounds = _androidworld_scrollable_bounds(observation)
    if scrollable_bounds is not None:
        left, top, right, bottom = scrollable_bounds
        display = _observation_display(observation or {})
        if display is not None:
            width, height = display
            # Keep the gesture strictly inside the source scrollable region.
            # The inset avoids edge/system-bar interception while retaining
            # the original direction and the canonical 0..1000 coordinates.
            inset_x = min((right - left) * 0.15, (right - left) / 2.0 - 1.0)
            inset_y = min((bottom - top) * 0.15, (bottom - top) / 2.0 - 1.0)
            safe_left = left + max(1.0, inset_x)
            safe_top = top + max(1.0, inset_y)
            safe_right = right - max(1.0, inset_x)
            safe_bottom = bottom - max(1.0, inset_y)
            if safe_right > safe_left and safe_bottom > safe_top:
                center_x = (safe_left + safe_right) / 2.0
                center_y = (safe_top + safe_bottom) / 2.0
                if action_type == "scroll":
                    gestures = {
                        "down": (center_x, center_y, center_x, safe_top),
                        "up": (center_x, center_y, center_x, safe_bottom),
                        "right": (center_x, center_y, safe_left, center_y),
                        "left": (center_x, center_y, safe_right, center_y),
                    }
                else:
                    gestures = {
                        "down": (center_x, safe_top, center_x, safe_bottom),
                        "up": (center_x, safe_bottom, center_x, safe_top),
                        "left": (safe_left, center_y, safe_right, center_y),
                        "right": (safe_right, center_y, safe_left, center_y),
                    }
                try:
                    x1, y1, x2, y2 = gestures[direction]
                except KeyError as error:
                    raise ValueError(
                        f"androidworld_action_direction_required:{action_type}"
                    ) from error
                return {
                    "x1": x1 / width * 1000.0,
                    "y1": y1 / height * 1000.0,
                    "x2": x2 / width * 1000.0,
                    "y2": y2 / height * 1000.0,
                }

    gestures = {
        "scroll": {
            "down": (500.0, 500.0, 500.0, 0.0),
            "up": (500.0, 500.0, 500.0, 1000.0),
            "right": (500.0, 500.0, 0.0, 500.0),
            "left": (500.0, 500.0, 1000.0, 500.0),
        },
        "swipe": {
            "down": (500.0, 0.0, 500.0, 1000.0),
            "up": (500.0, 1000.0, 500.0, 0.0),
            "left": (0.0, 500.0, 1000.0, 500.0),
            "right": (1000.0, 500.0, 0.0, 500.0),
        },
    }
    try:
        x1, y1, x2, y2 = gestures[action_type][direction]
    except KeyError as error:
        raise ValueError(f"androidworld_action_direction_required:{action_type}") from error
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _androidworld_scrollable_bounds(
    observation: dict[str, Any] | None,
) -> tuple[float, float, float, float] | None:
    """Return the largest enabled scrollable source region, if present."""

    if not isinstance(observation, dict):
        return None
    xml = observation_xml(observation)
    if not xml.strip():
        return None
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None
    candidates: list[tuple[float, float, float, float]] = []
    for element in root.iter():
        if str(element.attrib.get("scrollable") or "").casefold() != "true":
            continue
        if str(element.attrib.get("enabled", "true")).casefold() == "false":
            continue
        bounds = _parse_xml_bounds(element.attrib.get("bounds"))
        if bounds is not None:
            candidates.append(bounds)
    return max(
        candidates,
        key=lambda bounds: (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]),
        default=None,
    )


def _execution_swipe(
    value: dict[str, Any] | None,
    *,
    source_action: dict[str, Any],
) -> dict[str, float | int] | None:
    if not isinstance(value, dict) or str(value.get("tool") or "") != "swipe":
        return None
    args = value.get("args")
    if not isinstance(args, dict):
        return None
    coordinate_keys = ("x1", "y1", "x2", "y2")
    if not all(args.get(key) is not None for key in coordinate_keys):
        return None
    source_direction = str(source_action.get("direction") or "").strip()
    execution_direction = str(args.get("direction") or "").strip()
    if source_direction and execution_direction and source_direction != execution_direction:
        raise ValueError("androidworld_swipe_execution_direction_mismatch")
    result: dict[str, float | int] = {
        key: float(args[key]) for key in coordinate_keys
    }
    if args.get("duration_ms") is not None:
        result["duration_ms"] = int(args["duration_ms"])
    return result


def _androidworld_exact_swipe(
    action: dict[str, Any],
    observation: dict[str, Any],
) -> dict[str, float | int] | None:
    coordinate_keys = ("x1", "y1", "x2", "y2")
    if not all(action.get(key) is not None for key in coordinate_keys):
        return None
    display = _observation_display(observation)
    if display is None:
        raise ValueError("androidworld_action_display_required:swipe")
    width, height = display
    result: dict[str, float | int] = {
        "x1": float(action["x1"]) / width * 1000.0,
        "y1": float(action["y1"]) / height * 1000.0,
        "x2": float(action["x2"]) / width * 1000.0,
        "y2": float(action["y2"]) / height * 1000.0,
    }
    if action.get("duration_ms") is not None:
        result["duration_ms"] = int(action["duration_ms"])
    return result


def _transfer_state(observation: dict[str, Any]) -> dict[str, Any]:
    state: dict[str, Any] = {"state_id": state_id(observation)}
    xml = observation_xml(observation)
    if xml:
        state["xml"] = xml
    pixels = observation_screenshot(observation)
    if isinstance(pixels, dict) and str(pixels.get("path") or "").strip():
        state["screenshot_path"] = str(pixels["path"]).strip()
    display = observation_display(observation)
    if display is not None:
        state["display"] = {"width": display[0], "height": display[1]}
    auxiliaries = observation.get("auxiliaries")
    if isinstance(auxiliaries, dict):
        for key in ("package_name", "activity_name"):
            if auxiliaries.get(key) not in (None, ""):
                state[key] = str(auxiliaries[key])
    return state


def _map(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return dict(value) if isinstance(value, dict) else {}


__all__ = ["import_run_log_evidence", "project_androidworld_step_actions"]
