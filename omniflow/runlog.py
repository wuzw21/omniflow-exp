from __future__ import annotations

import json
import re
from typing import Any
import xml.etree.ElementTree as ET

from omniflow.core.schemas import canonicalize_action
from omniflow.core.trajectory import (
    canonicalize_run_log,
    observation_display,
    observation_xml,
    state_id,
)


def import_run_log_evidence(
    value: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_log = _hydrate_run_log_display(canonicalize_run_log(value))
    states: dict[str, dict[str, Any]] = {}
    for step in run_log["steps"]:
        observations = [step["observation"]]
        if isinstance(step.get("next_observation"), dict):
            observations.append(step["next_observation"])
        for observation in observations:
            _store_transfer_state(states, observation)
    final_observation = run_log.get("final_observation")
    if isinstance(final_observation, dict):
        _store_transfer_state(states, final_observation)
    return run_log, {
        "schema_version": "omniflow.transfer-state-catalog.v1",
        "run_id": run_log["run_id"],
        "states": states,
    }


def project_androidworld_step_actions(value: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or not isinstance(value.get("observation"), dict):
        raise ValueError("androidworld_run_log_step_required")
    action = dict(value.get("action") or {})
    observation = value["observation"]
    projected: list[dict[str, Any]] = []
    if (
        action.get("action_type") == "input_text"
        and _androidworld_input_point_is_editable(action, observation)
    ):
        projected.append(
            canonicalize_action(
                {"tool": "click", "args": _androidworld_action_point(action, observation)},
                replayable_only=True,
            )
        )
    projected.append(_androidworld_action_to_omniflow(action, observation=observation))
    return projected


def _store_transfer_state(
    states: dict[str, dict[str, Any]],
    observation: dict[str, Any],
) -> None:
    source_state = _transfer_state(observation)
    existing = states.get(source_state["state_id"])
    if existing is not None and existing != source_state:
        raise ValueError(f"source_state_conflict:{source_state['state_id']}")
    states[source_state["state_id"]] = source_state


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


def _androidworld_input_point_is_editable(
    action: dict[str, Any],
    observation: dict[str, Any],
) -> bool:
    point = _androidworld_action_point(action, observation)
    display = observation_display(observation)
    if point is None or display is None:
        return False
    x = point["x"] / 1000.0 * display[0]
    y = point["y"] / 1000.0 * display[1]
    try:
        root = ET.fromstring(observation_xml(observation))
    except ET.ParseError:
        return False
    for node in root.iter():
        bounds = _parse_xml_bounds(node.attrib.get("bounds"))
        if bounds is None or not (
            bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]
        ):
            continue
        if (
            str(node.attrib.get("editable") or "").casefold() == "true"
            or str(node.attrib.get("class") or "").endswith("EditText")
        ):
            return True
    return False


def _androidworld_action_to_omniflow(
    value: Any,
    *,
    observation: dict[str, Any],
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
        projected = {"tool": "input_text", "args": {"text": action.get("text", "")}}
    elif action_type in {"scroll", "swipe"}:
        projected = {
            "tool": "swipe",
            "args": {
                "direction": str(action.get("direction") or ""),
                **_androidworld_standard_swipe(
                    action_type,
                    str(action.get("direction") or ""),
                ),
            },
        }
    elif action_type == "open_app":
        projected = {
            "tool": "open_app",
            "args": {"package_name": str(action.get("app_name") or "")},
        }
    elif action_type == "navigate_back":
        projected = {"tool": "press_key", "args": {"key": "back"}}
    elif action_type == "navigate_home":
        projected = {"tool": "press_key", "args": {"key": "home"}}
    elif action_type == "keyboard_enter":
        projected = {"tool": "press_key", "args": {"key": "enter"}}
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
    display = observation_display(observation)
    if display is None:
        raise ValueError("androidworld_action_display_required")
    width, height = display
    return {"x": float(x) / width * 1000.0, "y": float(y) / height * 1000.0}


def _xml_index_bounds(xml: str, index: int) -> tuple[float, float, float, float] | None:
    if not xml.strip():
        return None
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None
    node = next(
        (
            item
            for item in root.iter("node")
            if str(item.attrib.get("id") or "") == str(index)
        ),
        None,
    )
    return _parse_xml_bounds(node.attrib.get("bounds")) if node is not None else None


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
) -> dict[str, float]:
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


def _transfer_state(observation: dict[str, Any]) -> dict[str, Any]:
    state: dict[str, Any] = {"state_id": state_id(observation)}
    xml = observation_xml(observation)
    if xml:
        state["xml"] = xml
    pixels = observation.get("pixels")
    if isinstance(pixels, dict) and str(pixels.get("path") or "").strip():
        state["screenshot_path"] = str(pixels["path"]).strip()
    auxiliaries = observation.get("auxiliaries")
    if isinstance(auxiliaries, dict):
        for key in ("package_name", "activity_name"):
            if auxiliaries.get(key) not in (None, ""):
                state[key] = str(auxiliaries[key])
        display = auxiliaries.get("display")
        if isinstance(display, dict) and set(display) == {"width", "height"}:
            state["display"] = dict(display)
    return state


def _map(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return dict(value) if isinstance(value, dict) else {}


__all__ = ["import_run_log_evidence", "project_androidworld_step_actions"]
