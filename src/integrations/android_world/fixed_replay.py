"""Fixed raw-replay executor for archived RunLogs (split out of run_episode).

Moved verbatim from ``src.integrations.android_world.run_episode``; that module re-exports these names.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import re
import time
from time import perf_counter
from typing import Any
from src.integrations.android_world.agent import MODE_OMNIFLOW
from src.integrations.android_world.host import make_agent_result
from src.integrations.runlog import import_run_log, project_run_log_step_actions
from src.integrations.android_world.episode_utils import (
    _raw_replay_number,
    to_serializable,
    utc_now_iso,
)


DEFAULT_RAW_REPLAY_ACTION_WAIT_SECONDS = 1.0


def _read_raw_replay_run_log(path_text: str) -> dict[str, Any]:
    path = Path(str(path_text or "").strip()).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"raw replay run log not found: {path}")
    decoded = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(decoded, dict):
        raise ValueError(f"raw replay run log must be a JSON object: {path}")
    return decoded


def _raw_replay_step_actions(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Project successful main-flow RunLog actions through the Function projector."""

    run_log = import_run_log(data)
    actions: list[dict[str, Any]] = []
    previous_successful_step: dict[str, Any] | None = None
    for source_step_index, step in enumerate(run_log["steps"]):
        if step["result"]["success"] is not True:
            continue
        action_type = str(step["action"].get("action_type") or "")
        if action_type in {"answer", "status", "unknown"}:
            continue
        previous_for_projection = previous_successful_step
        previous_successful_step = step
        metadata = step.get("metadata")
        if isinstance(metadata, dict) and metadata.get("origin") == "checker":
            continue
        for projected in project_run_log_step_actions(
            run_log,
            source_step_index,
            previous_step=previous_for_projection,
        ):
            actions.append(
                {
                    "tool": projected["tool"],
                    "args": dict(projected.get("args") or {}),
                    "_source_step_index": source_step_index,
                }
            )
    return actions


def _raw_replay_source_size(data: dict[str, Any]) -> tuple[int, int] | None:
    """Read the source display from the accepted RunLog observation."""

    run_log = import_run_log(data)
    for step in run_log["steps"]:
        observation = step["observation"]
        pixels = observation.get("screenshot")
        if pixels is None:
            pixels = observation.get("pixels")
        if isinstance(pixels, dict):
            return int(pixels["width"]), int(pixels["height"])
        auxiliaries = observation.get("auxiliaries")
        display = auxiliaries.get("display") if isinstance(auxiliaries, dict) else None
        if isinstance(display, dict):
            width = _coerce_positive_int(display.get("width"))
            height = _coerce_positive_int(display.get("height"))
            if width and height:
                return width, height
    return None


def _fixed_replay_parameter_leaves(
    value: Any,
    *,
    path: str = "$",
) -> list[tuple[str, Any]]:
    leaves: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            leaves.extend(
                _fixed_replay_parameter_leaves(
                    item,
                    path=f"{path}.{key}",
                )
            )
        return leaves
    if isinstance(value, list):
        for index, item in enumerate(value):
            leaves.extend(
                _fixed_replay_parameter_leaves(
                    item,
                    path=f"{path}[{index}]",
                )
            )
        return leaves
    if value is None or isinstance(value, bool):
        return leaves
    leaves.append((path, value))
    return leaves


def _fixed_replay_goal_value_boundary(
    text: str,
    *,
    start: int,
    value: str,
) -> bool:
    end = start + len(value)
    if value[0].isalnum() and start > 0 and text[start - 1].isalnum():
        return False
    if value[-1].isalnum() and end < len(text) and text[end].isalnum():
        return False
    return True


def _fixed_replay_goal_parameter_bindings(
    run_log_data: dict[str, Any],
    *,
    target_goal: str,
) -> dict[str, Any]:
    run_log = import_run_log(run_log_data)
    source_goal = str(run_log.get("goal") or "")
    target_goal_text = str(target_goal or "")
    task_parameters = run_log.get("task_parameters")
    task_parameters = task_parameters if isinstance(task_parameters, dict) else {}
    candidate_paths: dict[str, list[str]] = {}
    for path, value in _fixed_replay_parameter_leaves(task_parameters):
        final_key = path.rsplit(".", maxsplit=1)[-1]
        if final_key in {"seed", "browser_task_seed"}:
            continue
        source_value = str(value)
        if not source_value or source_value not in source_goal:
            continue
        candidate_paths.setdefault(source_value, []).append(path)

    occurrences: list[tuple[int, int, str]] = []
    cursor = 0
    candidates = sorted(candidate_paths, key=lambda item: (-len(item), item))
    while cursor < len(source_goal):
        matches = [
            value
            for value in candidates
            if source_goal.startswith(value, cursor)
            and _fixed_replay_goal_value_boundary(
                source_goal,
                start=cursor,
                value=value,
            )
        ]
        if not matches:
            cursor += 1
            continue
        selected = matches[0]
        occurrences.append((cursor, cursor + len(selected), selected))
        cursor += len(selected)

    report: dict[str, Any] = {
        "source_goal": source_goal,
        "target_goal": target_goal_text,
        "status": "no_goal_parameters",
        "bindings": [],
    }
    if source_goal == target_goal_text:
        report["status"] = "identical_goal"
        return report
    if not occurrences:
        return report

    pattern_parts: list[str] = []
    source_cursor = 0
    for index, (start, end, _source_value) in enumerate(occurrences):
        pattern_parts.append(re.escape(source_goal[source_cursor:start]))
        pattern_parts.append(f"(?P<value_{index}>.*?)")
        source_cursor = end
    pattern_parts.append(re.escape(source_goal[source_cursor:]))
    match = re.fullmatch("".join(pattern_parts), target_goal_text, flags=re.DOTALL)
    if match is None:
        report["status"] = "target_goal_template_mismatch"
        return report

    captured_by_source: dict[str, set[str]] = {}
    for index, (_start, _end, source_value) in enumerate(occurrences):
        captured_by_source.setdefault(source_value, set()).add(
            str(match.group(f"value_{index}"))
        )
    bindings: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for source_value, captured_values in captured_by_source.items():
        if len(captured_values) != 1:
            conflicts.append(
                {
                    "source_parameter_paths": sorted(candidate_paths[source_value]),
                    "source_value": source_value,
                    "target_values": sorted(captured_values),
                }
            )
            continue
        target_value = next(iter(captured_values))
        bindings.append(
            {
                "source_parameter_paths": sorted(candidate_paths[source_value]),
                "source_value": source_value,
                "target_value": target_value,
                "changed": source_value != target_value,
            }
        )
    report["bindings"] = bindings
    if conflicts:
        report["status"] = "goal_parameter_capture_conflict"
        report["conflicts"] = conflicts
    else:
        report["status"] = "matched_goal_template"
    return report


def _fixed_replay_bind_action_parameters(
    source_action: dict[str, Any],
    bindings: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bound_action = copy.deepcopy(source_action)
    changed: list[dict[str, Any]] = []
    replacements = [
        item
        for item in bindings
        if bool(item.get("changed")) and str(item.get("source_value") or "")
    ]
    replacements.sort(
        key=lambda item: (-len(str(item["source_value"])), str(item["source_value"]))
    )

    def bind_value(value: Any, *, path: str) -> Any:
        if isinstance(value, dict):
            return {
                key: bind_value(item, path=f"{path}.{key}")
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                bind_value(item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ]
        if not isinstance(value, str):
            return value
        updated = value
        for binding in replacements:
            source_value = str(binding["source_value"])
            target_value = str(binding.get("target_value") or "")
            match_kind = ""
            if updated == source_value:
                updated = target_value
                match_kind = "exact"
            elif len(source_value) >= 3 and source_value in updated:
                updated = updated.replace(source_value, target_value)
                match_kind = "substring"
            if match_kind:
                changed.append(
                    {
                        "action_path": path,
                        "source_parameter_paths": list(
                            binding.get("source_parameter_paths") or ()
                        ),
                        "source_value": source_value,
                        "target_value": target_value,
                        "match": match_kind,
                    }
                )
        return updated

    bound_action = bind_value(bound_action, path="$")
    assert isinstance(bound_action, dict)
    return bound_action, changed


def _coerce_positive_int(value: Any) -> int:
    try:
        number = int(round(float(value)))
    except Exception:
        return 0
    return number if number > 0 else 0


def _raw_replay_scale_coord(
    value: Any,
    *,
    source_extent: int,
    target_extent: int,
) -> int | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    if source_extent > 0 and target_extent > 0:
        numeric = numeric * (float(target_extent) / float(source_extent))
    if target_extent > 0:
        numeric = min(max(numeric, 0.0), float(target_extent - 1))
    return int(round(numeric))


def _raw_replay_pair_from_value(
    value: Any,
    *,
    x_keys: tuple[str, ...],
    y_keys: tuple[str, ...],
) -> tuple[float | None, float | None]:
    if isinstance(value, dict):
        return _raw_replay_number(value, *x_keys), _raw_replay_number(value, *y_keys)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except Exception:
            return None, None
    return None, None


def _raw_replay_relative_coord_pair(
    params: dict[str, Any],
    *,
    x_keys: tuple[str, ...],
    y_keys: tuple[str, ...],
    target_width: int,
    target_height: int,
) -> tuple[int | None, int | None]:
    target_evidence = (
        dict(params.get("target_evidence") or {})
        if isinstance(params.get("target_evidence"), dict)
        else {}
    )
    candidates = (
        params.get("canonical_0_1000"),
        params.get("relative_0_1000"),
        target_evidence.get("relative_0_1000"),
    )
    for candidate in candidates:
        raw_x, raw_y = _raw_replay_pair_from_value(
            candidate,
            x_keys=x_keys,
            y_keys=y_keys,
        )
        if raw_x is None or raw_y is None:
            continue
        x = _raw_replay_scale_coord(
            raw_x,
            source_extent=1000,
            target_extent=target_width,
        )
        y = _raw_replay_scale_coord(
            raw_y,
            source_extent=1000,
            target_extent=target_height,
        )
        return x, y
    return None, None


def _raw_replay_direction_from_points(
    x1: float | None,
    y1: float | None,
    x2: float | None,
    y2: float | None,
) -> str:
    if x1 is None or y1 is None or x2 is None or y2 is None:
        return ""
    dx = float(x2) - float(x1)
    dy = float(y2) - float(y1)
    if abs(dx) > abs(dy):
        return "left" if dx < 0 else "right"
    return "up" if dy < 0 else "down"


def _raw_replay_action_to_payload(
    source_action: dict[str, Any],
    *,
    source_size: tuple[int, int] | None,
    target_size: tuple[int, int],
    resolution: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    if source_action.get("tool") and isinstance(source_action.get("args"), dict):
        return {
            "action_type": "raw_omniflow_action",
            "action": {
                "tool": str(source_action["tool"]),
                "args": dict(source_action["args"]),
            },
        }, None
    action_type = str(
        source_action.get("type")
        or source_action.get("tool")
        or source_action.get("action_type")
        or ""
    ).strip().lower()
    if not action_type:
        return None, "missing_action_type"
    params = (
        dict(source_action.get("params") or {})
        if isinstance(source_action.get("params"), dict)
        else {}
    )
    if not params and isinstance(source_action.get("args"), dict):
        params = dict(source_action.get("args") or {})
    if source_action.get("coordinate_space") == "canonical_0_1000":
        source_width, source_height = 1000, 1000
    else:
        source_width, source_height = source_size or (target_size[0], target_size[1])
    target_width, target_height = target_size

    def _record_resolution(parameter_source: str) -> None:
        if resolution is None:
            return
        resolution["parameter_source"] = parameter_source

    def _scaled_xy() -> tuple[int | None, int | None]:
        relative_x, relative_y = _raw_replay_relative_coord_pair(
            params,
            x_keys=("x", "center_x", "touch_x"),
            y_keys=("y", "center_y", "touch_y"),
            target_width=target_width,
            target_height=target_height,
        )
        if relative_x is not None and relative_y is not None:
            return relative_x, relative_y
        raw_x = _raw_replay_number(params, "x", "center_x", "touch_x")
        raw_y = _raw_replay_number(params, "y", "center_y", "touch_y")
        x = _raw_replay_scale_coord(
            raw_x,
            source_extent=source_width,
            target_extent=target_width,
        )
        y = _raw_replay_scale_coord(
            raw_y,
            source_extent=source_height,
            target_extent=target_height,
        )
        return x, y

    if action_type in {"click", "tap", "double_tap", "long_press", "longpress"}:
        x, y = _scaled_xy()
        _record_resolution("recorded_coordinate")
        if x is None or y is None:
            return None, "missing_coordinates"
        return {
            "action_type": "long_press"
            if action_type in {"long_press", "longpress"}
            else "double_tap"
            if action_type == "double_tap"
            else "click",
            "x": x,
            "y": y,
        }, None

    if action_type in {"input_text", "type_text", "set_text", "type"}:
        payload: dict[str, Any] = {
            "action_type": "input_text",
            "text": str(params.get("text") if params.get("text") is not None else ""),
        }
        if "clear_text" in params:
            payload["clear_text"] = bool(params.get("clear_text"))
        elif "clear" in params:
            payload["clear_text"] = bool(params.get("clear"))
        else:
            # Canonical input_text has no clear flag because the AndroidWorld
            # host always replaces the focused field's existing contents.
            payload["clear_text"] = True
        x, y = _scaled_xy()
        if x is not None and y is not None:
            payload["x"] = x
            payload["y"] = y
        return payload, None

    if action_type == "set_clipboard":
        return {
            "action_type": "raw_set_clipboard",
            "text": str(params.get("text") if params.get("text") is not None else ""),
        }, None

    if action_type in {"open_app", "launch_app", "openapp"}:
        package_name = str(
            params.get("package_name")
            or params.get("package")
            or params.get("packageName")
            or ""
        ).strip()
        app_name = str(
            params.get("app_name") or params.get("app") or ""
        ).strip()
        app_identifier = package_name or app_name
        if not app_identifier:
            return None, "missing_app_identifier"
        return {
            "action_type": "raw_open_app",
            "app_identifier": app_identifier,
        }, None

    if action_type in {"press_key", "key_event", "presskey"}:
        key = str(
            params.get("key") or params.get("key_name") or params.get("keycode") or ""
        ).strip().lower()
        if key in {"back", "navigate_back", "press_back"}:
            return {"action_type": "navigate_back"}, None
        if key in {"home", "navigate_home", "press_home"}:
            return {"action_type": "navigate_home"}, None
        if key in {"enter", "keyboard_enter"}:
            return {"action_type": "keyboard_enter"}, None
        if key in {"delete", "del", "backspace", "keycode_del"}:
            return {
                "action_type": "press_keyboard",
                "keycode": "KEYCODE_DEL",
            }, None
        return None, "unsupported_androidworld_key"

    if action_type in {"back", "press_back", "navigate_back"}:
        return {"action_type": "navigate_back"}, None
    if action_type in {"home", "press_home", "navigate_home"}:
        return {"action_type": "navigate_home"}, None
    if action_type in {"wait", "sleep"}:
        seconds = _raw_replay_number(params, "time_s", "seconds", "duration_s")
        if seconds is None:
            duration_ms = _raw_replay_number(params, "duration_ms", "time_ms")
            if duration_ms is not None:
                seconds = float(duration_ms) / 1000.0
        return {
            "action_type": "raw_wait",
            "seconds": max(0.0, float(seconds if seconds is not None else 1.0)),
        }, None
    if action_type in {"finished", "finish", "done", "status"}:
        return {"action_type": "status", "goal_status": "complete"}, None

    if action_type == "answer":
        return {
            "action_type": "answer",
            "text": str(params.get("text") if params.get("text") is not None else ""),
        }, None

    if action_type in {"swipe", "scroll"}:
        x1 = _raw_replay_number(params, "x1", "start_x", "from_x", "touch_x")
        y1 = _raw_replay_number(params, "y1", "start_y", "from_y", "touch_y")
        x2 = _raw_replay_number(params, "x2", "end_x", "to_x", "lift_x")
        y2 = _raw_replay_number(params, "y2", "end_y", "to_y", "lift_y")
        relative_start = _raw_replay_relative_coord_pair(
            params,
            x_keys=("x1", "start_x", "from_x", "touch_x"),
            y_keys=("y1", "start_y", "from_y", "touch_y"),
            target_width=target_width,
            target_height=target_height,
        )
        relative_end = _raw_replay_relative_coord_pair(
            params,
            x_keys=("x2", "end_x", "to_x", "lift_x"),
            y_keys=("y2", "end_y", "to_y", "lift_y"),
            target_width=target_width,
            target_height=target_height,
        )
        if (
            relative_start[0] is not None
            and relative_start[1] is not None
            and relative_end[0] is not None
            and relative_end[1] is not None
        ):
            sx, sy = relative_start
            ex, ey = relative_end
            return {
                "action_type": "raw_swipe",
                "x1": sx,
                "y1": sy,
                "x2": ex,
                "y2": ey,
                "duration_ms": int(params.get("duration_ms") or 500),
                "input_source": str(params.get("input_source") or "")
                .strip()
                .lower(),
                "direction": _raw_replay_direction_from_points(sx, sy, ex, ey),
            }, None
        if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
            sx = _raw_replay_scale_coord(
                x1,
                source_extent=source_width,
                target_extent=target_width,
            )
            sy = _raw_replay_scale_coord(
                y1,
                source_extent=source_height,
                target_extent=target_height,
            )
            ex = _raw_replay_scale_coord(
                x2,
                source_extent=source_width,
                target_extent=target_width,
            )
            ey = _raw_replay_scale_coord(
                y2,
                source_extent=source_height,
                target_extent=target_height,
            )
            if sx is not None and sy is not None and ex is not None and ey is not None:
                return {
                    "action_type": "raw_swipe",
                    "x1": sx,
                    "y1": sy,
                    "x2": ex,
                    "y2": ey,
                    "duration_ms": int(params.get("duration_ms") or 500),
                    "input_source": str(params.get("input_source") or "")
                    .strip()
                    .lower(),
                    "direction": _raw_replay_direction_from_points(x1, y1, x2, y2),
                }, None
        direction = str(params.get("direction") or "").strip().lower()
        if direction in {"up", "down", "left", "right"}:
            return {
                "action_type": "scroll" if action_type == "scroll" else "swipe",
                "direction": direction,
            }, None
        return None, "missing_swipe_coordinates_or_direction"

    return None, f"unsupported_action_type:{action_type}"


def _sanitize_raw_replay_source_action(action: dict[str, Any]) -> dict[str, Any]:
    params = (
        dict(action.get("params") or {})
        if isinstance(action.get("params"), dict)
        else {}
    )
    for key in (
        "source_context",
        "target_description",
        "target_evidence",
        "xml_ref",
        "index",
    ):
        params.pop(key, None)
    return {"type": str(action.get("type") or action.get("tool") or ""), "params": params}


def _raw_replay_action_wait_seconds(
    source_action: dict[str, Any],
    payload: dict[str, Any],
) -> float:
    if payload.get("action_type") in {"raw_wait", "status"}:
        return 0.0
    params = (
        dict(source_action.get("params") or {})
        if isinstance(source_action.get("params"), dict)
        else {}
    )
    if not params and isinstance(source_action.get("args"), dict):
        params = dict(source_action.get("args") or {})
    seconds = _raw_replay_number(
        params,
        "wait_after_s",
        "post_action_wait_s",
        "post_wait_s",
    )
    if seconds is None:
        seconds = DEFAULT_RAW_REPLAY_ACTION_WAIT_SECONDS
    return max(0.0, float(seconds))


def _execute_raw_replay_host_swipe(
    agent: Any,
    payload: dict[str, Any],
    *,
    target_size: tuple[int, int],
) -> None:
    """Execute a replay swipe through the checked AndroidWorld host path."""

    target_width, target_height = target_size
    args = {
        "x1": float(payload["x1"]) / float(target_width) * 1000.0,
        "y1": float(payload["y1"]) / float(target_height) * 1000.0,
        "x2": float(payload["x2"]) / float(target_width) * 1000.0,
        "y2": float(payload["y2"]) / float(target_height) * 1000.0,
        "duration_ms": int(payload.get("duration_ms") or 500),
    }
    input_source = str(payload.get("input_source") or "").strip().lower()
    if input_source:
        args["input_source"] = input_source
    host_result = agent.host.act({"tool": "swipe", "args": args})
    if getattr(host_result, "success", False) is not True:
        raise RuntimeError(
            str(getattr(host_result, "error", "") or "raw replay swipe failed")
        )


def _raw_replay_observation_record(
    observation: Any,
    *,
    fallback_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    xml = str(getattr(observation, "xml", "") or "")
    package_name = str(getattr(observation, "package_name", "") or "")
    activity_name = str(getattr(observation, "activity_name", "") or "")
    # Observation capture is a measurement sidecar for fixed replay.  An
    # empty transient page must not change the replay action sequence; keep an
    # explicit empty sample and let the action use the device target size.
    extra = dict(getattr(observation, "extra", {}) or {})
    androidworld_state = extra.get("androidworld_state")
    if androidworld_state is not None and not isinstance(androidworld_state, dict):
        raise RuntimeError("raw replay AndroidWorld state is not serializable")
    display = (
        dict(extra.get("display") or {})
        if isinstance(extra.get("display"), dict)
        else {}
    )
    width = _coerce_positive_int(display.get("width"))
    height = _coerce_positive_int(display.get("height"))
    if fallback_size:
        width = width or _coerce_positive_int(fallback_size[0])
        height = height or _coerce_positive_int(fallback_size[1])
    record = {
        "provider": str(extra.get("observe_backend") or "unknown"),
        "package_name": package_name,
        "activity_name": activity_name,
        "xml": xml,
        "xml_available": bool(xml),
        "checker_passed": True,
    }
    if isinstance(androidworld_state, dict):
        record["androidworld_state"] = to_serializable(androidworld_state)
    if width and height:
        record.update(
            width=width,
            height=height,
            screenshot={
                "width": width,
                "height": height,
                "original_width": width,
                "original_height": height,
            },
        )
    return record


def _launch_raw_replay_app(
    app_identifier: str,
    host: Any,
) -> None:
    identifier = str(app_identifier or "").strip()
    if not identifier:
        raise ValueError("raw_replay_open_app_identifier_required")
    result = host.act(
        {"tool": "open_app", "args": {"package_name": identifier}}
    )
    if getattr(result, "success", False) is not True:
        raise RuntimeError(
            str(getattr(result, "error", "") or "raw replay open_app failed")
        )


def _apply_fixed_replay(
    agent: Any,
    *,
    run_log_json_path: str,
    adb_path: str = "",
) -> Any:
    """Replay fixed source actions through recorded AndroidWorld parameters."""

    original_set_max_steps = getattr(agent, "set_max_steps", None)
    run_log_data = _read_raw_replay_run_log(run_log_json_path)
    source_actions = _raw_replay_step_actions(run_log_data)
    source_size = _raw_replay_source_size(run_log_data)
    state: dict[str, Any] = {"ran": False, "payload": None}
    replay_host = getattr(agent, "host", None)
    replay_observe = getattr(replay_host, "observe", None)
    capture_native_observations = str(
        os.environ.get("OMNIFLOW_RAW_REPLAY_CAPTURE_OBSERVATIONS") or ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    capture_observations = capture_native_observations
    if capture_observations and not callable(replay_observe):
        raise RuntimeError("fixed replay observation capture requires host.observe")

    def _forced_reset(go_home: bool = False) -> None:
        state["ran"] = False
        state["payload"] = None
        if go_home:
            agent.env.reset(go_home=True)

    def _forced_set_max_steps(step_budget: int) -> None:
        del step_budget
        if callable(original_set_max_steps):
            original_set_max_steps(max(1, len(source_actions) + 1))

    def _execute_payload(
        payload: dict[str, Any],
        *,
        target_size: tuple[int, int],
    ) -> None:
        if payload.get("action_type") == "raw_omniflow_action":
            host_result = agent.host.act(dict(payload["action"]))
            if getattr(host_result, "success", False) is not True:
                raise RuntimeError(
                    str(
                        getattr(host_result, "error", "")
                        or "fixed replay OmniFlow action failed"
                    )
                )
            return
        if payload.get("action_type") == "raw_open_app":
            _launch_raw_replay_app(
                str(payload["app_identifier"]),
                agent.host,
            )
            return
        if payload.get("action_type") == "raw_wait":
            result = agent.host.act(
                {
                    "tool": "wait",
                    "args": {
                        "duration_ms": int(
                            round(float(payload.get("seconds") or 0.0) * 1000.0)
                        )
                    },
                }
            )
            if getattr(result, "success", False) is not True:
                raise RuntimeError(
                    str(getattr(result, "error", "") or "raw replay wait failed")
                )
            return
        if payload.get("action_type") == "raw_swipe":
            _execute_raw_replay_host_swipe(
                agent,
                payload,
                target_size=target_size,
            )
            return
        if payload.get("action_type") == "raw_set_clipboard":
            raise RuntimeError(
                "fixed_replay_private_action_not_androidworld_json_action:set_clipboard"
            )
        if payload.get("action_type") == "status":
            return
        if payload.get("action_type") in {"click", "long_press"}:
            target_width, target_height = target_size
            host_result = agent.host.act(
                {
                    "tool": str(payload["action_type"]),
                    "args": {
                        "x": float(payload["x"]) / float(target_width) * 1000.0,
                        "y": float(payload["y"]) / float(target_height) * 1000.0,
                        **(
                            {"duration_ms": int(payload.get("duration_ms") or 1000)}
                            if payload.get("action_type") == "long_press"
                            else {}
                        ),
                    },
                }
            )
            if getattr(host_result, "success", False) is not True:
                raise RuntimeError(
                    str(getattr(host_result, "error", "") or "raw replay action failed")
                )
            return
        if payload.get("action_type") == "input_text":
            target_width, target_height = target_size
            args: dict[str, Any] = {
                "text": str(payload.get("text") or ""),
                "clear_text": bool(payload.get("clear_text", True)),
            }
            if payload.get("x") is not None and payload.get("y") is not None:
                args.update(
                    x=float(payload["x"]) / float(target_width) * 1000.0,
                    y=float(payload["y"]) / float(target_height) * 1000.0,
                )
            host_result = agent.host.act({"tool": "input_text", "args": args})
            if getattr(host_result, "success", False) is not True:
                raise RuntimeError(
                    str(getattr(host_result, "error", "") or "raw replay input failed")
                )
            return
        raise RuntimeError(
            "fixed_replay_action_not_projected_through_omniflow_host:"
            f"{payload.get('action_type')}"
        )

    def _forced_step(goal: str):
        goal_text = str(goal or "").strip()
        if bool(state.get("ran")):
            payload = dict(state.get("payload") or {})
            summary = payload.get("run_log_summary")
            summary = dict(summary or {}) if isinstance(summary, dict) else {}
            return make_agent_result(
                done=True,
                data={
                    "summary": "fixed replay already completed",
                    "run_id": payload.get("run_id"),
                    "step_index": 0,
                    "source": "recorded_coordinate_replay_v1",
                    "actions_executed": int(summary.get("actions_executed") or 0),
                    "fallback": False,
                    "error": None,
                    "done_reason": "fixed_replay_already_completed",
                },
            )

        started = perf_counter()
        # Replay coordinates and AndroidWorld accessibility bounds share the
        # logical application display.  The physical display can be rotated
        # or have a different size on tablets/folds (for example 1280x800
        # versus the portrait logical 720x1280), so using it first changes the
        # coordinate space before the first action is dispatched.
        target_size = (0, 0)
        for attribute in ("logical_screen_size", "device_screen_size"):
            candidate = tuple(getattr(agent.env, attribute, ()) or ())
            if (
                len(candidate) == 2
                and float(candidate[0]) > 0
                and float(candidate[1]) > 0
            ):
                target_size = candidate
                break
        if len(target_size) != 2 or not target_size[0] or not target_size[1]:
            target_size = source_size or (1080, 2400)
        step_results: list[dict[str, Any]] = []
        completed = True
        error_text: str | None = None
        actions_executed = 0
        recorded_coordinate_actions = 0
        direct_actions = 0
        parameter_bound_actions = 0
        parameter_bindings_applied = 0
        # Fixed replay invokes Host.act directly instead of entering the
        # normal Planner loop, so it must establish the same initial OOB
        # observation that a normal agent receives before its first action.
        # Otherwise AndroidWorldEpisodeRecorder falls back to native
        # env.get_state() for the pre-action evidence; on an initialized OOB
        # device that snapshot may not contain XML and the first action fails
        # before it is dispatched.
        if source_actions:
            try:
                replay_observe(xml=True, screenshot=False, app_info=True)
            except Exception as exc:  # noqa: BLE001
                completed = False
                error_text = str(exc) or type(exc).__name__
        goal_parameter_binding = _fixed_replay_goal_parameter_bindings(
            run_log_data,
            target_goal=goal_text,
        )
        goal_bindings = list(goal_parameter_binding.get("bindings") or ())
        for index, original_source_action in enumerate(
            source_actions if completed else ()
        ):
            source_action, action_parameter_bindings = (
                _fixed_replay_bind_action_parameters(
                    original_source_action,
                    goal_bindings,
                )
            )
            if action_parameter_bindings:
                parameter_bound_actions += 1
                parameter_bindings_applied += len(action_parameter_bindings)
            observation_record: dict[str, Any] | None = None
            if capture_observations:
                observation_record = _raw_replay_observation_record(
                    replay_observe(xml=True, screenshot=False, app_info=True),
                    fallback_size=(int(target_size[0]), int(target_size[1])),
                )
            action_target_size = (int(target_size[0]), int(target_size[1]))
            if observation_record is not None:
                observed_width = _coerce_positive_int(
                    observation_record.get("width")
                )
                observed_height = _coerce_positive_int(
                    observation_record.get("height")
                )
                if observed_width and observed_height:
                    action_target_size = (observed_width, observed_height)
            action_resolution: dict[str, Any] = {}
            payload, skip_reason = _raw_replay_action_to_payload(
                source_action,
                source_size=source_size,
                target_size=action_target_size,
                resolution=action_resolution,
            )
            parameter_source = str(
                action_resolution.get("parameter_source")
                or "direct_androidworld_action"
            )
            step_record: dict[str, Any] = {
                "index": index,
                "source_action": _sanitize_raw_replay_source_action(
                    original_source_action
                ),
                "androidworld_action": dict(payload or {}),
                "source_screen_size": list(source_size) if source_size else None,
                "target_screen_size": list(action_target_size),
                "completed": False,
                "skipped": False,
                "parameter_source": parameter_source,
            }
            if action_parameter_bindings:
                step_record["bound_source_action"] = (
                    _sanitize_raw_replay_source_action(source_action)
                )
                step_record["task_parameter_bindings"] = action_parameter_bindings
            if observation_record is not None:
                step_record["observation_before_act"] = observation_record
            if skip_reason:
                completed = False
                error_text = skip_reason
                step_record["error"] = skip_reason
                step_results.append(step_record)
                break
            try:
                assert payload is not None
                _execute_payload(
                    payload,
                    target_size=action_target_size,
                )
                actions_executed += 1
                if parameter_source == "recorded_coordinate":
                    recorded_coordinate_actions += 1
                else:
                    direct_actions += 1
                step_record["completed"] = True
                wait_after_s = _raw_replay_action_wait_seconds(
                    source_action,
                    payload,
                )
                step_record["wait_after_s"] = wait_after_s
                if wait_after_s > 0:
                    time.sleep(wait_after_s)
                if payload.get("action_type") == "status":
                    break
            except Exception as exc:  # noqa: BLE001
                completed = False
                error_text = str(exc)
                step_record["error"] = error_text
                step_results.append(step_record)
                break
            step_results.append(step_record)

        final_observation_record: dict[str, Any] | None = None
        final_observation_error: str | None = None
        if capture_observations:
            try:
                final_observation_record = _raw_replay_observation_record(
                    replay_observe(xml=True, screenshot=False, app_info=True),
                    fallback_size=(int(target_size[0]), int(target_size[1])),
                )
            except Exception as exc:  # noqa: BLE001
                final_observation_error = str(exc) or type(exc).__name__
        elapsed_ms = max(0.0, (perf_counter() - started) * 1000.0)
        run_id = (
            "fixed_replay_"
            + str(run_log_data.get("run_id") or Path(run_log_json_path).stem)
            + "_"
            + utc_now_iso()
            .replace("-", "")
            .replace(":", "")
            .replace(".", "")
            .replace("+", "")
        )
        done_reason = (
            "fixed_replay_completed"
            if completed
            else "fixed_replay_failed"
        )
        execution_summary = {
            "completed": bool(completed),
            "replay_completed": bool(completed),
            "execution_backend": "recorded_coordinate_replay_v1",
            "steps": int(actions_executed),
            "actions_executed": int(actions_executed),
            "recorded_coordinate_actions": int(recorded_coordinate_actions),
            "direct_actions": int(direct_actions),
            "parameter_bound_actions": int(parameter_bound_actions),
            "parameter_bindings_applied": int(parameter_bindings_applied),
            "model_calls": 0,
            "tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "elapsed_ms": elapsed_ms,
        }
        if error_text:
            execution_summary["failure_reason"] = error_text
        execution_trace = {
            "run_id": run_id,
            "goal": goal_text,
            "device_label": str(
                os.environ.get("OMNIFLOW_EVAL_DEVICE_LABEL") or ""
            ).strip(),
            "android_serial": str(os.environ.get("ANDROID_SERIAL") or "").strip(),
            "completed": bool(completed),
            "replay_completed": bool(completed),
            "done_reason": done_reason,
            "step_count": len(step_results),
            "agent_step_count": 1,
            "function_step_count": int(actions_executed),
            "actions_executed": int(actions_executed),
            "summary": {"execution_summary": execution_summary},
            "source_run_log": str(run_log_json_path),
            "source_screen_size": list(source_size) if source_size else None,
            "target_screen_size": [int(target_size[0]), int(target_size[1])],
            "goal_parameter_binding": goal_parameter_binding,
            "steps": [
                {
                    "step_index": 0,
                    "selection_source": "fixed_replay",
                    "execution_source": "recorded_coordinate_replay_v1",
                    "provider_detail": {
                        "raw_replay": {
                            "source_run_log": str(run_log_json_path),
                            "source_action_count": len(source_actions),
                            "actions_executed": int(actions_executed),
                            "recorded_coordinate_actions": int(
                                recorded_coordinate_actions
                            ),
                            "direct_actions": int(direct_actions),
                            "parameter_bound_actions": int(parameter_bound_actions),
                            "parameter_bindings_applied": int(
                                parameter_bindings_applied
                            ),
                            "goal_parameter_binding": goal_parameter_binding,
                            "source_screen_size": list(source_size)
                            if source_size
                            else None,
                            "target_screen_size": [
                                int(target_size[0]),
                                int(target_size[1]),
                            ],
                            "step_results": step_results,
                            "final_observation": final_observation_record,
                            "final_observation_error": final_observation_error,
                        }
                    },
                }
            ],
        }
        state["ran"] = True
        state["payload"] = {
            "run_id": run_id,
            "goal": goal_text,
            "run_log_summary": {
                "run_id": run_id,
                "completed": bool(completed),
                "replay_completed": bool(completed),
                "step_count": len(step_results),
                "agent_step_count": 1,
                "function_step_count": int(actions_executed),
                "actions_executed": int(actions_executed),
                "done_reason": done_reason,
                "duration_ms": elapsed_ms,
            },
            "execution_trace": execution_trace,
        }
        sidecar_path = str(os.environ.get("OMNIFLOW_RAW_REPLAY_RESULT_JSON") or "").strip()
        if sidecar_path:
            try:
                sidecar = Path(sidecar_path).expanduser().resolve()
                sidecar.parent.mkdir(parents=True, exist_ok=True)
                sidecar.write_text(
                    json.dumps(
                        to_serializable(
                            {
                                "completed": bool(completed),
                                "replay_completed": bool(completed),
                                "error": error_text,
                                "run_id": run_id,
                                "step_count": len(step_results),
                                "actions_executed": int(actions_executed),
                                "recorded_coordinate_actions": int(
                                    recorded_coordinate_actions
                                ),
                                "direct_actions": int(direct_actions),
                                "parameter_bound_actions": int(
                                    parameter_bound_actions
                                ),
                                "parameter_bindings_applied": int(
                                    parameter_bindings_applied
                                ),
                                "duration_ms": elapsed_ms,
                                "execution_trace": execution_trace,
                            }
                        ),
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    ),
                    encoding="utf-8",
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] failed to write raw replay sidecar {sidecar_path}: {exc}")
        return make_agent_result(
            done=True,
            data={
                "summary": error_text or "fixed replay executed",
                "run_id": run_id,
                "step_index": 0,
                "source": "recorded_coordinate_replay_v1",
                "actions_executed": int(actions_executed),
                "fallback": False,
                "error": error_text,
                "done_reason": done_reason,
            },
        )

    agent.reset = _forced_reset
    agent.set_max_steps = _forced_set_max_steps
    agent.step = _forced_step
    agent.name = MODE_OMNIFLOW
    return agent
