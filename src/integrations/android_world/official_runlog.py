from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

from omniflow import Observation
from omniflow.schemas import canonicalize_action
from omniflow.trajectory import canonicalize_run_log
from omniflow.transfer import (
    TRANSFER_STATE_CATALOG_VERSION,
    capture_transfer_state,
)
from src.integrations.android_world.host import androidworld_elements_xml


PackageResolver = Callable[[str], str]


def materialize_m3a_episode_runlog(
    episode: dict[str, Any],
    *,
    task_name: str,
    goal: str,
    package_resolver: PackageResolver | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    episode_data = _mapping(episode.get("episode_data"))
    raw_actions = _sequence(episode_data.get("action_output_json"))
    before_elements = _sequence(episode_data.get("before_ui_elements"))
    raw_screenshots = _sequence(episode_data.get("raw_screenshot"))
    after_screenshots = _sequence(episode_data.get("after_screenshot_with_som"))
    summaries = _sequence(episode_data.get("summary"))
    reasons = _sequence(episode_data.get("action_reason"))
    if not raw_actions or not before_elements or not raw_screenshots:
        raise ValueError("official_m3a_episode_evidence_missing")
    if not (
        len(raw_actions) == len(before_elements) == len(raw_screenshots)
    ):
        raise ValueError("official_m3a_episode_evidence_length_mismatch")

    native_states = [
        _native_state(
            _sequence(elements),
            raw_screenshot=raw_screenshots[index],
            step_index=index,
        )
        for index, elements in enumerate(before_elements)
    ]
    canonical_steps: list[dict[str, Any]] = []
    unexecuted_steps: list[dict[str, Any]] = []
    action_facts: list[dict[str, Any]] = []

    for episode_step_index, raw_action in enumerate(raw_actions):
        action_payload = _object_payload(raw_action)
        action_type = _enum_text(action_payload.get("action_type"))
        if not action_type:
            unexecuted_steps.append(
                {
                    "episode_step_index": episode_step_index,
                    "reason": "missing_action_output",
                }
            )
            continue
        if action_type == "status":
            unexecuted_steps.append(
                {
                    "episode_step_index": episode_step_index,
                    "reason": "terminal_status_not_replayable",
                    "raw_action": action_payload,
                }
            )
            continue

        elements = _sequence(before_elements[episode_step_index])
        invalid_reason = _invalid_index_reason(action_payload, elements)
        if invalid_reason:
            unexecuted_steps.append(
                {
                    "episode_step_index": episode_step_index,
                    "reason": invalid_reason,
                    "raw_action": action_payload,
                }
            )
            continue
        if _item(after_screenshots, episode_step_index) is None:
            unexecuted_steps.append(
                {
                    "episode_step_index": episode_step_index,
                    "reason": "missing_after_observation",
                    "raw_action": action_payload,
                }
            )
            continue
        if episode_step_index + 1 >= len(native_states):
            raise ValueError(
                f"official_m3a_after_state_missing:{episode_step_index}"
            )

        action = _canonical_m3a_action(
            action_payload,
            elements=elements,
            display=native_states[episode_step_index]["display"],
            package_resolver=package_resolver,
        )
        metadata = {
            "step_id": f"m3a_episode_step_{episode_step_index}",
            "status": "succeeded",
            "thinking": str(_item(reasons, episode_step_index) or ""),
            "summary": str(_item(summaries, episode_step_index) or ""),
            "official_episode_step_index": episode_step_index,
        }
        canonical_steps.append(
            {
                "step_index": len(canonical_steps),
                "before_state_id": native_states[episode_step_index]["state_id"],
                "action": action,
                "result": {"success": True},
                "after_state_id": native_states[episode_step_index + 1]["state_id"],
                "metadata": metadata,
            }
        )
        action_facts.append(
            {
                "episode_step_index": episode_step_index,
                "action": action,
                "before_state_id": native_states[episode_step_index]["state_id"],
                "after_state_id": native_states[episode_step_index + 1]["state_id"],
            }
        )

    if not canonical_steps:
        raise ValueError("official_m3a_replayable_actions_missing")
    run_identity = {
        "task_name": str(task_name or "").strip(),
        "goal": str(goal or ""),
        "agent_name": str(episode.get("agent_name") or "m3a_gpt4v"),
        "episode_seed": episode.get("seed"),
        "instance_id": episode.get("instance_id"),
        "actions": action_facts,
    }
    run_id = "m3a_" + _stable_hash(run_identity)[:24]
    referenced_state_ids = {
        str(step[field])
        for step in canonical_steps
        for field in ("before_state_id", "after_state_id")
    }
    states_by_id = {
        state["state_id"]: state
        for state in native_states
        if state["state_id"] in referenced_state_ids
    }
    missing_states = sorted(referenced_state_ids - set(states_by_id))
    if missing_states:
        raise ValueError(
            "official_m3a_transfer_states_incomplete:" + ",".join(missing_states)
        )

    run_log = canonicalize_run_log(
        {
            "schema_version": "omniflow.canonical_run_log.v1",
            "run_id": run_id,
            "goal": str(goal or ""),
            "status": "succeeded",
            "success": True,
            "steps": canonical_steps,
            "final_state_id": canonical_steps[-1]["after_state_id"],
            "diagnostics": {
                "task_name": str(task_name or "").strip(),
                "official_agent_name": str(
                    episode.get("agent_name") or "m3a_gpt4v"
                ),
                "state_backend": "androidworld",
                "action_backend": "androidworld",
                "native_androidworld_agent_io": True,
                "display_source": "androidworld_raw_screenshot_shape",
                "source_episode_step_count": len(raw_actions),
                "materialized_action_count": len(canonical_steps),
                "unexecuted_steps": unexecuted_steps,
            },
        }
    )
    transfer_catalog = {
        "schema_version": TRANSFER_STATE_CATALOG_VERSION,
        "run_id": run_id,
        "states": states_by_id,
    }
    return run_log, transfer_catalog


def androidworld_package_resolver(app_name: str) -> str:
    name = str(app_name or "").strip()
    if not name:
        raise ValueError("official_m3a_open_app_name_missing")
    from android_world.env import adb_utils

    activity = adb_utils.get_adb_activity(name)
    if activity:
        package = str(activity).split("/", 1)[0].strip()
        if package:
            return package
    if "." in name and " " not in name:
        return name
    raise ValueError(f"official_m3a_open_app_package_unresolved:{name}")


def _canonical_m3a_action(
    payload: dict[str, Any],
    *,
    elements: list[Any],
    display: dict[str, int],
    package_resolver: PackageResolver | None,
) -> dict[str, Any]:
    action_type = _enum_text(payload.get("action_type"))
    width = int(display["width"])
    height = int(display["height"])
    if action_type in {"click", "long_press", "input_text"}:
        x, y = _action_point(payload, elements)
        args: dict[str, Any] = {
            "x": _normalize_coordinate(x, width),
            "y": _normalize_coordinate(y, height),
        }
        if action_type == "input_text":
            args["text"] = str(payload.get("text") or "")
        if action_type == "long_press" and payload.get("duration_ms") is not None:
            args["duration_ms"] = int(payload["duration_ms"])
        return canonicalize_action(
            {"tool": action_type, "args": args},
            persisted_only=True,
        )
    if action_type in {"scroll", "swipe"}:
        args = _swipe_args(
            payload,
            elements=elements,
            width=width,
            height=height,
        )
        return canonicalize_action(
            {"tool": "swipe", "args": args},
            persisted_only=True,
        )
    if action_type == "open_app":
        resolver = package_resolver or androidworld_package_resolver
        package = resolver(str(payload.get("app_name") or ""))
        return canonicalize_action(
            {"tool": "open_app", "args": {"package_name": package}}
        )
    if action_type in {"navigate_back", "navigate_home", "keyboard_enter"}:
        key = {
            "navigate_back": "back",
            "navigate_home": "home",
            "keyboard_enter": "enter",
        }[action_type]
        return canonicalize_action({"tool": "press_key", "args": {"key": key}})
    if action_type == "wait":
        return canonicalize_action(
            {"tool": "wait", "args": {"duration_ms": 1000}}
        )
    raise ValueError(f"official_m3a_action_unsupported:{action_type}")


def _swipe_args(
    payload: dict[str, Any],
    *,
    elements: list[Any],
    width: int,
    height: int,
) -> dict[str, Any]:
    direction = _enum_text(payload.get("direction"))
    if direction not in {"up", "down", "left", "right"}:
        raise ValueError(f"official_m3a_scroll_direction_invalid:{direction}")
    index = payload.get("index")
    if index is not None:
        bounds = _bounds(elements[int(index)])
        if bounds is None:
            raise ValueError("official_m3a_action_bounds_missing")
        left, top, right, bottom = bounds
        left, top = max(0, left), max(0, top)
        right, bottom = min(width, right), min(height, bottom)
    else:
        left, top, right, bottom = 0, 0, width, height
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    if _enum_text(payload.get("action_type")) == "scroll":
        end = {
            "down": (center_x, top),
            "up": (center_x, bottom),
            "right": (left, center_y),
            "left": (right, center_y),
        }[direction]
        start = (center_x, center_y)
    else:
        start, end = {
            "down": ((center_x, 0), (center_x, height)),
            "up": ((center_x, height), (center_x, 0)),
            "left": ((0, center_y), (width, center_y)),
            "right": ((width, center_y), (0, center_y)),
        }[direction]
    return {
        "direction": direction,
        "x1": _normalize_coordinate(start[0], width),
        "y1": _normalize_coordinate(start[1], height),
        "x2": _normalize_coordinate(end[0], width),
        "y2": _normalize_coordinate(end[1], height),
    }


def _native_state(
    elements: list[Any],
    *,
    raw_screenshot: Any,
    step_index: int,
) -> dict[str, Any]:
    if not elements:
        raise ValueError(f"official_m3a_native_state_missing:{step_index}")
    display = _display(raw_screenshot, elements=elements, step_index=step_index)
    xml_text = androidworld_elements_xml(elements)
    packages = [
        str(_read(element, "package_name", "") or "").strip()
        for element in elements
    ]
    packages = [value for value in packages if value]
    non_system = [value for value in packages if value != "com.android.systemui"]
    observation = Observation(
        xml=xml_text,
        package_name=(non_system or packages or [None])[-1],
        extra={"display": display, "observe_backend": "androidworld"},
    )
    return capture_transfer_state(observation)


def _display(
    raw_screenshot: Any,
    *,
    elements: list[Any],
    step_index: int,
) -> dict[str, int]:
    shape = _read(raw_screenshot, "shape")
    if not isinstance(shape, (list, tuple)) or len(shape) < 2:
        raise ValueError(f"official_m3a_native_display_missing:{step_index}")
    height = int(shape[0])
    width = int(shape[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"official_m3a_native_display_invalid:{step_index}")
    for element_index, element in enumerate(elements):
        bounds = _bounds(element)
        if bounds is None:
            continue
        left, top, right, bottom = bounds
        if (
            left < 0
            or top < 0
            or right < left
            or bottom < top
            or right > width
            or bottom > height
        ):
            raise ValueError(
                "official_m3a_element_bounds_outside_display:"
                f"{step_index}:{element_index}"
            )
    return {"width": int(width), "height": int(height)}


def _action_point(payload: dict[str, Any], elements: list[Any]) -> tuple[float, float]:
    index = payload.get("index")
    if index is not None:
        bounds = _bounds(elements[int(index)])
        if bounds is None:
            raise ValueError("official_m3a_action_bounds_missing")
        left, top, right, bottom = bounds
        return (left + right) / 2.0, (top + bottom) / 2.0
    if payload.get("x") is None or payload.get("y") is None:
        raise ValueError("official_m3a_action_point_missing")
    return float(payload["x"]), float(payload["y"])


def _invalid_index_reason(payload: dict[str, Any], elements: list[Any]) -> str:
    action_type = _enum_text(payload.get("action_type"))
    index = payload.get("index")
    if action_type not in {"click", "long_press", "input_text", "scroll"}:
        return ""
    if index is None:
        return ""
    if isinstance(index, bool) or not isinstance(index, int):
        return "invalid_element_index_type"
    if index < 0 or index >= len(elements):
        return "element_index_out_of_range"
    return ""


def _normalize_coordinate(value: float, extent: int) -> int | float:
    normalized = max(0.0, min(1000.0, float(value) / float(extent) * 1000.0))
    rounded = round(normalized, 6)
    return int(rounded) if rounded.is_integer() else rounded


def _bounds(value: Any) -> tuple[int, int, int, int] | None:
    raw = _read(value, "bbox_pixels") or _read(value, "bbox")
    if raw is None:
        return None
    try:
        return tuple(
            int(float(_read(raw, key)))
            for key in ("x_min", "y_min", "x_max", "y_max")
        )
    except (TypeError, ValueError):
        return None


def _object_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return {
            str(key): _plain(item)
            for key, item in attributes.items()
            if not str(key).startswith("_") and item is not None
        }
    return {}


def _plain(value: Any) -> Any:
    enum_value = getattr(value, "value", None)
    if enum_value is not None and enum_value is not value:
        return _plain(enum_value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return str(value)


def _enum_text(value: Any) -> str:
    plain = _plain(value)
    return str(plain or "").strip().lower()


def _read(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _item(values: list[Any], index: int) -> Any:
    return values[index] if 0 <= index < len(values) else None


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "androidworld_package_resolver",
    "materialize_m3a_episode_runlog",
]
