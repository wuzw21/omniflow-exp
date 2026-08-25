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
) -> list[dict[str, Any]]:
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
    projected_action = _androidworld_action_to_omniflow(
        action,
        observation=observation,
    )
    if (
        action.get("action_type") == "input_text"
        and not str(projected_action["args"].get("target_description") or "").strip()
    ):
        target_description = _previous_click_target_description(previous_step)
        if target_description:
            projected_action["args"]["target_description"] = target_description
    projected.append(projected_action)
    return projected


def _previous_click_target_description(
    previous_step: dict[str, Any] | None,
) -> str:
    if not isinstance(previous_step, dict):
        return ""
    action = previous_step.get("action")
    observation = previous_step.get("observation")
    if (
        not isinstance(action, dict)
        or action.get("action_type") not in {"click", "double_tap"}
        or not isinstance(observation, dict)
    ):
        return ""
    _point, target = _androidworld_input_target(action, observation)
    return _androidworld_input_target_description(target) if target is not None else ""


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


def _androidworld_input_point_is_editable(
    action: dict[str, Any],
    observation: dict[str, Any],
) -> bool:
    if _androidworld_action_point(action, observation) is None:
        return False
    _, node = _androidworld_input_target(action, observation)
    return node is not None


def _androidworld_input_target(
    action: dict[str, Any],
    observation: dict[str, Any],
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
        if str(node.attrib.get("editable") or "").casefold() == "true"
        or str(node.attrib.get("class") or "").endswith("EditText")
    ]
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


def _androidworld_input_target_description(node: ET.Element) -> str:
    for attribute in ("content-desc", "text", "resource-id"):
        value = str(node.attrib.get(attribute) or "").strip()
        if value:
            return value.rsplit("/", 1)[-1]
    # AndroidWorld often exposes a clickable row/container without its child
    # label on the clickable node itself.  Promote the first meaningful
    # descendant label so the Function records the semantic target (for
    # example, a notebook name) instead of the generic "editable text field".
    for descendant in node.iter():
        if descendant is node:
            continue
        for attribute in ("content-desc", "text"):
            value = str(descendant.attrib.get(attribute) or "").strip()
            if value:
                return value
    return "editable text field"


def _androidworld_overlapping_label(
    observation: dict[str, Any],
    target: ET.Element,
) -> str:
    """Find a semantic label in a flattened sibling-based Android tree."""
    target_bounds = _parse_xml_bounds(target.attrib.get("bounds"))
    if target_bounds is None:
        return ""
    try:
        root = ET.fromstring(observation_xml(observation))
    except ET.ParseError:
        return ""
    candidates: list[tuple[float, str]] = []
    for node in root.iter("node"):
        if node is target:
            continue
        label = ""
        for attribute in ("content-desc", "text"):
            label = str(node.attrib.get(attribute) or "").strip()
            if label:
                break
        bounds = _parse_xml_bounds(node.attrib.get("bounds"))
        if not label or bounds is None:
            continue
        left, top, right, bottom = bounds
        t_left, t_top, t_right, t_bottom = target_bounds
        overlap = max(0.0, min(right, t_right) - max(left, t_left)) * max(
            0.0,
            min(bottom, t_bottom) - max(top, t_top),
        )
        if overlap > 0:
            candidates.append((overlap, label))
    return max(candidates, default=(0.0, ""))[1]


def _androidworld_action_to_omniflow(
    value: Any,
    *,
    observation: dict[str, Any],
) -> dict[str, Any]:
    action = dict(value) if isinstance(value, dict) else {}
    action_type = str(action.get("action_type") or "").strip()
    if action_type in {"click", "double_tap"}:
        args = _required_androidworld_action_point(action, observation)
        target = _androidworld_click_target(action, observation)
        if target is not None:
            target_description = _androidworld_input_target_description(target)
            if target_description == "editable text field":
                target_description = _androidworld_overlapping_label(
                    observation,
                    target,
                )
            if target_description:
                args["target_description"] = target_description
        projected = {
            "tool": "click",
            "args": args,
        }
    elif action_type == "long_press":
        projected = {
            "tool": "long_press",
            "args": _required_androidworld_action_point(action, observation),
        }
    elif action_type == "input_text":
        point, target = _androidworld_input_target(action, observation)
        args: dict[str, Any] = {"text": action.get("text", "")}
        if point is not None and target is not None:
            args.update(point)
            args["target_description"] = _androidworld_input_target_description(target)
        projected = {"tool": "input_text", "args": args}
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
    elif action_type == "press_keyboard":
        keycode = str(action.get("keycode") or "").strip().upper()
        keycode = keycode.removeprefix("KEYCODE_")
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
