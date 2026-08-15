from __future__ import annotations

import hashlib
import json
import mimetypes
from pathlib import Path
import re
from typing import Any, Callable, Iterable
import xml.etree.ElementTree as ET

from PIL import Image

from omniflow.core.schemas import canonicalize_action
from omniflow.core.trajectory import (
    OMNIFLOW_RUN_LOG_SCHEMA_VERSION,
    canonicalize_run_log,
    observation_display,
    observation_xml,
    state_id,
)
from omniflow.runlog import (
    import_run_log_evidence as _import_run_log_evidence,
)
from omniflow.runlog import (
    project_androidworld_step_actions as _project_androidworld_step_actions,
)

_EXECUTION_TIMING_ARGS = {
    "post_action_wait_s",
    "post_wait_s",
    "wait_after_s",
}


def infer_input_text_target(
    before_xml: str,
    after_xml: str,
    *,
    input_text: str,
) -> dict[str, Any]:
    """Return the unique source input changed by a successful text action."""

    try:
        before_root = ET.fromstring(str(before_xml or ""))
        after_root = ET.fromstring(str(after_xml or ""))
    except ET.ParseError:
        return {}

    def input_nodes(root: ET.Element) -> list[ET.Element]:
        return [
            node
            for node in root.iter()
            if str(node.attrib.get("editable") or "").casefold() == "true"
            or str(node.attrib.get("class") or "") == "android.widget.EditText"
        ]

    before_inputs = input_nodes(before_root)
    after_inputs = input_nodes(after_root)
    if not before_inputs or len(before_inputs) != len(after_inputs):
        return {}

    expected = " ".join(str(input_text or "").casefold().split())
    changed: list[int] = []
    for index, (before, after) in enumerate(
        zip(before_inputs, after_inputs, strict=True)
    ):
        before_text = " ".join(str(before.attrib.get("text") or "").casefold().split())
        after_text = " ".join(str(after.attrib.get("text") or "").casefold().split())
        if before_text == after_text:
            continue
        if expected and expected not in after_text:
            continue
        changed.append(index)
    if len(changed) != 1:
        return {}

    ordinal = changed[0]
    node = before_inputs[ordinal]
    identity = {
        output_key: value
        for output_key, value in (
            ("text", str(node.attrib.get("text") or "").strip()),
            ("content_desc", str(node.attrib.get("content-desc") or "").strip()),
            ("resource_id", str(node.attrib.get("resource-id") or "").strip()),
        )
        if value
    }
    return {
        "input_ordinal": ordinal,
        "identity": identity or {"role": "editable"},
    }


class ScreenshotResolver:
    """Resolve immutable screenshot aliases only inside explicit roots."""

    def __init__(self, roots: Iterable[str | Path] = ()):
        self.roots = tuple(
            sorted(
                {
                    Path(root).expanduser().resolve()
                    for root in roots
                    if str(root).strip()
                }
            )
        )
        for root in self.roots:
            if not root.is_dir():
                raise ValueError(f"screenshot_root_invalid:{root}")

    def resolve(
        self,
        observation: dict[str, Any],
        *,
        task_name: str,
        step_index: int,
        phase: str,
        required: bool,
    ) -> dict[str, Any] | None:
        raw_path = _screenshot_path(observation)
        candidates: set[Path] = set()
        if raw_path:
            direct = Path(raw_path).expanduser()
            if direct.is_absolute() and direct.is_file():
                candidates.add(direct.resolve())
            basename = direct.name
            if basename:
                candidates.update(self._alias_candidates(task_name, basename))
        if not candidates:
            stem = f"step_{int(step_index) + 1:02d}_{phase}"
            for suffix in (".jpg", ".jpeg", ".png", ".webp"):
                candidates.update(
                    self._alias_candidates(task_name, stem + suffix)
                )
        if not candidates:
            if required:
                raise ValueError(
                    f"screenshot_reference_unresolved:{task_name}:{step_index}:{phase}"
                )
            return None
        if len(candidates) != 1:
            joined = ",".join(str(path) for path in sorted(candidates))
            raise ValueError(
                f"screenshot_reference_ambiguous:{task_name}:{step_index}:"
                f"{phase}:{joined}"
            )
        return _screenshot_reference(next(iter(candidates)))

    def _alias_candidates(self, task_name: str, basename: str) -> set[Path]:
        task_dir_name = f"{task_name}.artifacts"
        candidates: set[Path] = set()
        for root in self.roots:
            for candidate in (
                root / task_dir_name / basename,
                root / "artifacts" / task_dir_name / basename,
                root / basename,
            ):
                if candidate.is_file():
                    candidates.add(candidate.resolve())
        return candidates


def import_run_log(
    value: dict[str, Any],
    *,
    package_resolver: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Load the one production RunLog schema; legacy input is rejected."""
    del package_resolver
    return canonicalize_run_log(value)


def adapt_source_run_log(
    value: dict[str, Any],
    *,
    task_name: str,
    task_parameters: dict[str, Any],
    seed: int | None,
    source_path: str | Path,
    source_states: dict[str, dict[str, Any]] | None = None,
    screenshot_roots: Iterable[str | Path] = (),
    require_screenshots: bool = True,
    package_resolver: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Normalize native evidence or validate an existing production RunLog."""
    if (
        value.get("schema_version") == OMNIFLOW_RUN_LOG_SCHEMA_VERSION
        and not _is_legacy_run_log(value)
    ):
        run_log = import_run_log(value)
        expected_task = str(task_name).strip()
        if run_log["task_name"] != expected_task:
            raise ValueError(
                "source_run_log_task_mismatch:"
                f"expected={expected_task}:actual={run_log['task_name']}"
            )
        return run_log
    return convert_legacy_run_log(
        value,
        task_name=task_name,
        task_parameters=task_parameters,
        seed=seed,
        source_path=source_path,
        source_states=source_states,
        screenshot_roots=screenshot_roots,
        require_screenshots=require_screenshots,
        package_resolver=package_resolver,
    )


def _is_legacy_run_log(value: dict[str, Any]) -> bool:
    if any(
        key in value
        for key in ("androidworld", "completed", "done_reason", "trace_id")
    ):
        return True
    steps = value.get("steps")
    return isinstance(steps, list) and any(
        isinstance(step, dict)
        and any(
            key in step
            for key in (
                "observation_before_act",
                "executed_actions",
                "actions",
            )
        )
        for step in steps
    )


def import_run_log_evidence(
    value: dict[str, Any],
    *,
    evidence_root: str | Path | None = None,
    package_resolver: Callable[[str], str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a production RunLog and derive its source transfer-state catalog."""
    del evidence_root, package_resolver
    return _import_run_log_evidence(value)


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
    width, height = (int(value) for value in match.groups())
    return (width, height) if width > 0 and height > 0 else None


def convert_legacy_run_log(
    value: dict[str, Any],
    *,
    task_name: str,
    task_parameters: dict[str, Any],
    seed: int | None,
    source_path: str | Path,
    source_states: dict[str, dict[str, Any]] | None = None,
    screenshot_roots: Iterable[str | Path] = (),
    require_screenshots: bool = True,
    package_resolver: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Perform the one allowed historical-to-AndroidWorld RunLog conversion."""
    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"legacy_run_log_missing:{source}")
    task = str(task_name or "").strip()
    if not task:
        raise ValueError("legacy_run_log_task_required")
    payload = _map(value.get("payload")) or value
    payload = _map(payload.get("run_log")) or payload
    raw_steps = payload.get("steps") or payload.get("cards") or []
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError(f"legacy_run_log_steps_required:{task}")
    states = None if source_states is None else dict(source_states)
    canonical_action_points = (
        str(payload.get("schema_version") or "")
        == "omniflow.canonical_run_log.v1"
    )
    resolver = ScreenshotResolver(screenshot_roots)
    converted_steps: list[dict[str, Any]] = []
    for raw_step_index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, dict):
            raise ValueError(f"legacy_run_log_step_invalid:{task}:{raw_step_index}")
        before = _hydrate_legacy_observation(
            _legacy_before_observation(raw_step),
            state_identifier=raw_step.get("before_state_id"),
            source_states=states,
        )
        after = _hydrate_legacy_observation(
            _legacy_after_observation(raw_step),
            state_identifier=raw_step.get("after_state_id"),
            source_states=states,
        )
        step_index = _integer(raw_step.get("step_index"), raw_step_index)
        observation = _androidworld_state(
            before,
            pixels=resolver.resolve(
                before,
                task_name=task,
                step_index=step_index,
                phase="before",
                required=require_screenshots,
            ),
        )
        next_observation = (
            _androidworld_state(
                after,
                pixels=resolver.resolve(
                    after,
                    task_name=task,
                    step_index=step_index,
                    phase="after",
                    required=False,
                ),
            )
            if after
            else None
        )
        raw_actions = _legacy_actions(raw_step)
        if not raw_actions:
            raise ValueError(f"legacy_run_log_action_required:{task}:{step_index}")
        inferred_package_name = str(
            after.get("package_name")
            or after.get("packageName")
            or _legacy_following_package(raw_steps, raw_step_index)
            or ""
        ).strip()
        for raw_action in raw_actions:
            action = _legacy_action_to_androidworld(
                raw_action,
                observation=before,
                inferred_package_name=inferred_package_name,
                package_resolver=package_resolver,
                default_coordinate_space=(
                    "canonical_0_1000"
                    if canonical_action_points
                    and not any(
                        isinstance(raw_step.get(key), list)
                        and bool(raw_step.get(key))
                        for key in ("executed_actions", "actions")
                    )
                    else ""
                ),
            )
            result = {"success": _success(raw_step, default=True)}
            raw_result = _map(raw_step.get("result"))
            result["success"] = _success(raw_result, default=result["success"])
            error = str(
                raw_result.get("error") or raw_result.get("error_message") or ""
            ).strip()
            if error:
                result["error"] = error
            step: dict[str, Any] = {
                "step_index": len(converted_steps),
                "observation": observation,
                "action": action,
                "result": result,
            }
            if next_observation is not None:
                step["next_observation"] = next_observation
            metadata = _legacy_step_metadata(
                raw_step,
                raw_action=raw_action,
            )
            if action["action_type"] == "unknown":
                metadata["legacy_action"] = json.loads(
                    json.dumps(raw_action, ensure_ascii=False, default=str)
                )
            if metadata:
                step["metadata"] = metadata
            converted_steps.append(step)
            wait_step_count = _legacy_additional_wait_step_count(
                raw_action,
                action_type=str(action.get("action_type") or ""),
            )
            for _ in range(wait_step_count):
                converted_steps.append(
                    {
                        "step_index": len(converted_steps),
                        "observation": next_observation or observation,
                        "action": {"action_type": "wait"},
                        "result": dict(result),
                    }
                )
    success = _success(payload, default=_success(value, default=False))
    source_schema = str(payload.get("schema_version") or "unknown")
    converted: dict[str, Any] = {
        "schema_version": OMNIFLOW_RUN_LOG_SCHEMA_VERSION,
        "run_id": str(
            payload.get("run_id") or value.get("run_id") or f"legacy_{task}"
        ),
        "task_name": task,
        "goal": str(
            payload.get("goal") or payload.get("operation_description") or ""
        ),
        "task_parameters": json.loads(
            json.dumps(task_parameters, ensure_ascii=False)
        ),
        "seed": seed,
        "status": "succeeded" if success else "failed",
        "success": success,
        "validator": {
            "official": True,
            "success": success,
            "reward": _validator_reward(payload, success),
        },
        "provenance": {
            "kind": "legacy_import",
            "source_path": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_schema_version": source_schema,
        },
        "steps": converted_steps,
    }
    for output, aliases in (
        ("started_at_ms", ("started_at_ms",)),
        ("finished_at_ms", ("finished_at_ms",)),
    ):
        raw_time = _first(payload, aliases)
        if isinstance(raw_time, int) and not isinstance(raw_time, bool) and raw_time >= 0:
            converted[output] = raw_time
    diagnostics = _map(payload.get("diagnostics"))
    if diagnostics:
        converted["diagnostics"] = diagnostics
    return canonicalize_run_log(converted)


def project_androidworld_step_actions(value: dict[str, Any]) -> list[dict[str, Any]]:
    """Project one official RunLog step to OmniFlow Function action fields."""
    return _project_androidworld_step_actions(value)


def _androidworld_input_point_is_editable(
    action: dict[str, Any],
    observation: dict[str, Any],
) -> bool:
    point = _androidworld_action_point(action, observation)
    if point is None:
        return False
    display = observation_display(observation)
    if display is None:
        return False
    x = point["x"] / 1000.0 * display[0]
    y = point["y"] / 1000.0 * display[1]
    xml = observation_xml(observation)
    try:
        root = ET.fromstring(xml)
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
    return {
        "x": float(x) / width * 1000.0,
        "y": float(y) / height * 1000.0,
    }


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
    if node is None:
        return None
    match = re.fullmatch(
        r"\[(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\]"
        r"\[(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\]",
        str(node.attrib.get("bounds") or ""),
    )
    if match is None:
        return None
    left, top, right, bottom = map(float, match.groups())
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _ui_element_bounds(value: Any) -> tuple[float, float, float, float] | None:
    element = _map(value)
    bounds = _map(element.get("bbox_pixels")) or _map(element.get("bbox"))
    aliases = (
        ("x_min", "y_min", "x_max", "y_max"),
        ("left", "top", "right", "bottom"),
    )
    for keys in aliases:
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
        raise ValueError(
            f"androidworld_action_direction_required:{action_type}"
        ) from error
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _legacy_action_to_androidworld(
    value: Any,
    *,
    observation: dict[str, Any],
    inferred_package_name: str,
    package_resolver: Callable[[str], str] | None,
    default_coordinate_space: str = "",
) -> dict[str, Any]:
    tool, args = _legacy_action_tool_and_args(value)
    for key in _EXECUTION_TIMING_ARGS:
        args.pop(key, None)
    if default_coordinate_space and not args.get("coordinate_space"):
        args["coordinate_space"] = default_coordinate_space
    if tool in {"click", "tap", "double_tap", "long_press", "longpress"}:
        x, y = _legacy_point(args, observation)
        return {
            "action_type": (
                "double_tap"
                if tool == "double_tap"
                else "long_press"
                if tool in {"long_press", "longpress"}
                else "click"
            ),
            "x": x,
            "y": y,
        }
    if tool in {"input_text", "type_text", "set_text", "type"}:
        action: dict[str, Any] = {
            "action_type": "input_text",
            "text": str(args.get("text") if args.get("text") is not None else ""),
            "clear_text": bool(args.get("clear_text", True)),
        }
        if args.get("x") is not None and args.get("y") is not None:
            action["x"], action["y"] = _legacy_point(args, observation)
        return action
    if tool in {"swipe", "scroll"}:
        gesture_type = _legacy_androidworld_gesture_type(
            tool,
            args,
            observation,
        )
        direction = str(args.get("direction") or "").strip().lower()
        if not direction:
            direction = _legacy_swipe_direction(
                args,
                action_type=gesture_type,
            )
        if direction not in {"left", "right", "down", "up"}:
            raise ValueError(f"legacy_action_direction_required:{tool}")
        return {"action_type": gesture_type, "direction": direction}
    if tool in {"open_app", "start_activity", "launch_app", "openapp"}:
        app_name = str(args.get("app_name") or args.get("app") or "").strip()
        package = str(
            args.get("package_name")
            or args.get("package")
            or inferred_package_name
            or ""
        ).strip()
        if not package and app_name:
            resolver = package_resolver or _default_package_resolver
            package = str(resolver(app_name) or "").strip()
        identifier = package or app_name
        if not identifier:
            raise ValueError("legacy_action_open_app_identifier_required")
        return {"action_type": "open_app", "app_name": identifier}
    if tool in {"press_back", "back", "navigate_back"}:
        return {"action_type": "navigate_back"}
    if tool in {"press_home", "home", "navigate_home"}:
        return {"action_type": "navigate_home"}
    if tool in {"keyboard_enter", "press_enter"}:
        return {"action_type": "keyboard_enter"}
    if tool in {"press_key", "key_event", "presskey"}:
        key = str(args.get("key") or args.get("keycode") or "").strip().upper()
        key = key.removeprefix("KEYCODE_")
        key = {"DELETE": "DEL"}.get(key, key)
        if key in {"BACK", "NAVIGATE_BACK", "PRESS_BACK"}:
            return {"action_type": "navigate_back"}
        if key in {"HOME", "NAVIGATE_HOME", "PRESS_HOME"}:
            return {"action_type": "navigate_home"}
        if key in {"ENTER", "KEYBOARD_ENTER", "PRESS_ENTER"}:
            return {"action_type": "keyboard_enter"}
        return {
            "action_type": "unknown",
            **({"keycode": f"KEYCODE_{key}"} if key else {}),
        }
    if tool in {"wait", "sleep"}:
        return {"action_type": "wait"}
    if tool in {"finished", "finish", "done", "status"}:
        content = str(args.get("content") or "").strip()
        return (
            {"action_type": "answer", "text": content}
            if content
            else {"action_type": "status", "goal_status": "complete"}
        )
    if tool == "answer":
        return {"action_type": "answer", "text": str(args.get("text") or "")}
    return {"action_type": "unknown"}


def _legacy_actions(step: dict[str, Any]) -> list[Any]:
    for key in ("executed_actions", "actions"):
        value = step.get(key)
        if isinstance(value, list) and value:
            return list(value)
    for key in ("action", "tool_call"):
        value = step.get(key)
        if isinstance(value, dict):
            return [value]
    if any(key in step for key in ("tool", "type", "action_type")):
        return [step]
    tool_name = str(step.get("tool_name") or "").strip()
    if tool_name:
        return [{"tool": tool_name, "args": _map(step.get("params"))}]
    provider_detail = _map(step.get("provider_detail"))
    online_action = provider_detail.get("online_action")
    if isinstance(online_action, dict):
        return [online_action]
    return []


def _legacy_before_observation(step: dict[str, Any]) -> dict[str, Any]:
    source_context = _map(step.get("source_context"))
    source_context = _map(source_context.get("src_ctx")) or source_context
    observation = _legacy_observation(
        step.get("state")
        or step.get("observation")
        or step.get("observation_before_act")
        or step.get("before")
    )
    for key, value in source_context.items():
        observation.setdefault(key, value)
    return observation


def _legacy_after_observation(step: dict[str, Any]) -> dict[str, Any]:
    return _legacy_observation(
        step.get("after_state")
        or step.get("next_state")
        or step.get("observation_after_act")
        or step.get("after")
    )


def _legacy_following_package(steps: list[Any], step_index: int) -> str:
    following_index = step_index + 1
    if following_index >= len(steps) or not isinstance(
        steps[following_index], dict
    ):
        return ""
    observation = _legacy_before_observation(steps[following_index])
    return str(
        observation.get("package_name")
        or observation.get("packageName")
        or ""
    ).strip()


def _legacy_observation(value: Any) -> dict[str, Any]:
    return {"xml": value} if isinstance(value, str) else _map(value)


def _hydrate_legacy_observation(
    observation: dict[str, Any],
    *,
    state_identifier: Any,
    source_states: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    identifier = str(state_identifier or "").strip()
    if not identifier or source_states is None:
        return observation
    source_state = source_states.get(identifier)
    if not isinstance(source_state, dict):
        raise ValueError(f"legacy_source_state_missing:{identifier}")
    hydrated: dict[str, Any] = {"state_id": identifier}
    for key in ("xml", "package_name", "activity_name"):
        if _present(source_state.get(key)):
            hydrated[key] = source_state[key]
    display = source_state.get("display")
    if isinstance(display, dict):
        if _present(display.get("width")):
            hydrated["width"] = display["width"]
        if _present(display.get("height")):
            hydrated["height"] = display["height"]
    for key, value in observation.items():
        if _present(value):
            hydrated[key] = value
    return hydrated


def _androidworld_state(
    value: dict[str, Any],
    *,
    pixels: dict[str, Any] | None,
) -> dict[str, Any]:
    forest = value.get("forest")
    if forest is None:
        forest = _first(
            value,
            (
                "xml",
                "observation_xml",
                "hierarchy_xml",
                "raw_xml",
                "parsed_xml",
                "encoded_xml",
                "html_xml",
                "page",
                "source_xml",
            ),
        )
    ui_elements = value.get("ui_elements")
    if not isinstance(ui_elements, list):
        ui_elements = []
    auxiliaries = _map(value.get("auxiliaries"))
    aliases = {
        "state_id": ("state_id",),
        "package_name": ("package_name", "packageName"),
        "activity_name": ("activity_name", "activityName"),
        "provider": ("provider",),
    }
    for output, names in aliases.items():
        item = _first(value, names)
        if _present(item):
            auxiliaries[output] = item
    width = _first(value, ("display_width", "screen_width", "width"))
    height = _first(value, ("display_height", "screen_height", "height"))
    screenshot = _map(value.get("screenshot"))
    width = width if _present(width) else _first(screenshot, ("width", "display_width"))
    height = height if _present(height) else _first(screenshot, ("height", "display_height"))
    if _present(width) and _present(height):
        auxiliaries["display"] = {"width": int(width), "height": int(height)}
    return {
        "pixels": pixels,
        "forest": forest,
        "ui_elements": json.loads(json.dumps(ui_elements, ensure_ascii=False, default=str)),
        "auxiliaries": auxiliaries or None,
    }


def _transfer_state(observation: dict[str, Any]) -> dict[str, Any]:
    identifier = state_id(observation)
    state: dict[str, Any] = {"state_id": identifier}
    xml = observation_xml(observation)
    if xml:
        state["xml"] = xml
    pixels = observation.get("pixels")
    if isinstance(pixels, dict) and str(pixels.get("path") or "").strip():
        state["screenshot_path"] = str(pixels["path"]).strip()
    auxiliaries = observation.get("auxiliaries")
    if isinstance(auxiliaries, dict):
        for key in ("package_name", "activity_name"):
            if _present(auxiliaries.get(key)):
                state[key] = str(auxiliaries[key])
        display = auxiliaries.get("display")
        if isinstance(display, dict) and set(display) == {"width", "height"}:
            state["display"] = dict(display)
    return state


def _legacy_point(
    args: dict[str, Any], observation: dict[str, Any]
) -> tuple[int, int]:
    x = _number(args, "x", "center_x", "touch_x")
    y = _number(args, "y", "center_y", "touch_y")
    if x is None or y is None:
        raise ValueError("legacy_action_coordinates_required")
    coordinate_space = str(args.get("coordinate_space") or "").strip()
    if coordinate_space == "canonical_0_1000":
        width, height = _legacy_display(observation)
        x = x / 1000.0 * width
        y = y / 1000.0 * height
    return max(0, int(round(x))), max(0, int(round(y)))


def _legacy_androidworld_gesture_type(
    action_type: str,
    args: dict[str, Any],
    observation: dict[str, Any],
) -> str:
    if action_type != "swipe" or str(args.get("direction") or "").strip():
        return action_type
    x1 = _number(args, "x1", "start_x", "from_x", "touch_x")
    y1 = _number(args, "y1", "start_y", "from_y", "touch_y")
    x2 = _number(args, "x2", "end_x", "to_x", "lift_x")
    y2 = _number(args, "y2", "end_y", "to_y", "lift_y")
    if None in {x1, y1, x2, y2}:
        return action_type
    try:
        if str(args.get("coordinate_space") or "").strip() == "canonical_0_1000":
            width, height = 1000.0, 1000.0
        else:
            width, height = _legacy_display(observation)
    except ValueError:
        return action_type
    dx = float(x2) - float(x1)
    dy = float(y2) - float(y1)
    if abs(dx) >= abs(dy):
        tolerance = max(1.0, float(width) * 0.02)
        starts_at_edge = (
            float(x1) <= tolerance
            if dx > 0
            else float(x1) >= float(width) - tolerance
        )
    else:
        tolerance = max(1.0, float(height) * 0.02)
        starts_at_edge = (
            float(y1) <= tolerance
            if dy > 0
            else float(y1) >= float(height) - tolerance
        )
    return "swipe" if starts_at_edge else "scroll"


def _legacy_swipe_direction(
    args: dict[str, Any],
    *,
    action_type: str,
) -> str:
    x1 = _number(args, "x1", "start_x", "from_x", "touch_x")
    y1 = _number(args, "y1", "start_y", "from_y", "touch_y")
    x2 = _number(args, "x2", "end_x", "to_x", "lift_x")
    y2 = _number(args, "y2", "end_y", "to_y", "lift_y")
    if None in {x1, y1, x2, y2}:
        return ""
    dx = float(x2) - float(x1)
    dy = float(y2) - float(y1)
    if abs(dx) >= abs(dy):
        return "left" if dx > 0 else "right"
    if action_type == "scroll":
        return "up" if dy > 0 else "down"
    return "down" if dy > 0 else "up"


def _legacy_additional_wait_step_count(
    value: Any,
    *,
    action_type: str,
) -> int:
    _, args = _legacy_action_tool_and_args(value)
    if action_type == "wait":
        seconds = _number(args, "time_s", "seconds", "duration_s")
        duration_ms = _number(args, "duration_ms", "time_ms")
        if seconds is not None and duration_ms is not None:
            raise ValueError("legacy_action_wait_duration_ambiguous")
        if duration_ms is not None:
            seconds = duration_ms / 1000.0
        if seconds is None:
            seconds = 1.0
    else:
        seconds = _number(
            args,
            "wait_after_s",
            "post_action_wait_s",
            "post_wait_s",
        )
        if seconds is None:
            seconds = 1.0
    rounded_seconds = round(seconds)
    if seconds < 1.0 or abs(seconds - rounded_seconds) > 1e-9:
        raise ValueError(f"legacy_action_wait_not_representable:{seconds}")
    return int(rounded_seconds) - 1


def _legacy_action_tool_and_args(value: Any) -> tuple[str, dict[str, Any]]:
    raw = _map(value)
    function = _map(raw.get("function"))
    tool = str(
        raw.get("tool")
        or raw.get("type")
        or raw.get("action_type")
        or raw.get("tool_name")
        or raw.get("name")
        or function.get("name")
        or ""
    ).strip().lower()
    args = _map(
        raw.get("args")
        or raw.get("arguments")
        or raw.get("params")
        or function.get("arguments")
    )
    if tool == "android_privileged_action":
        tool = str(args.pop("tool", "")).strip().lower()
        args.update(_map(args.pop("arguments", None)))
    return tool, args


def _legacy_display(observation: dict[str, Any]) -> tuple[float, float]:
    screenshot = _map(observation.get("screenshot"))
    width = _first(observation, ("display_width", "screen_width", "width"))
    height = _first(observation, ("display_height", "screen_height", "height"))
    width = width if _present(width) else _first(screenshot, ("width", "display_width"))
    height = height if _present(height) else _first(screenshot, ("height", "display_height"))
    try:
        converted = float(width), float(height)
    except (TypeError, ValueError) as error:
        raise ValueError("legacy_observation_display_required") from error
    if converted[0] <= 0 or converted[1] <= 0:
        raise ValueError("legacy_observation_display_required")
    return converted


def _legacy_step_metadata(
    step: dict[str, Any],
    *,
    raw_action: Any,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    source = _map(step.get("metadata")) or _map(step.get("diagnostics"))
    for key in ("thinking", "summary", "action_description", "origin"):
        if _present(source.get(key)):
            metadata[key] = source[key]
    target_evidence = _legacy_action_target_evidence(raw_action)
    if target_evidence:
        metadata["source_target_evidence"] = target_evidence
    return metadata


def _legacy_action_target_evidence(value: Any) -> dict[str, Any]:
    raw = _map(value)
    function = _map(raw.get("function"))
    args = _map(
        raw.get("args")
        or raw.get("arguments")
        or raw.get("params")
        or function.get("arguments")
    )
    evidence: dict[str, Any] = {}
    target_description = str(args.get("target_description") or "").strip()
    if target_description:
        evidence["target_description"] = target_description
    source_context = _map(args.get("source_context"))
    element = _legacy_semantic_identity(source_context.get("element"))
    if element:
        evidence["element"] = element
    target = _legacy_semantic_identity(args.get("target_evidence"))
    if target:
        evidence["target"] = target
    return evidence


def _legacy_semantic_identity(value: Any) -> dict[str, str]:
    raw = _map(value)
    aliases = {
        "text": ("text", "label"),
        "content_desc": (
            "content_desc",
            "content-desc",
            "description",
        ),
        "resource_id": ("resource_id", "resource-id"),
    }
    identity: dict[str, str] = {}
    for output_key, input_keys in aliases.items():
        for input_key in input_keys:
            text = str(raw.get(input_key) or "").strip()
            if text:
                identity[output_key] = text
                break
    return identity


def _screenshot_path(observation: dict[str, Any]) -> str:
    screenshot = _map(observation.get("screenshot"))
    pixels = _map(observation.get("pixels"))
    screenshot_value = observation.get("screenshot")
    pixels_value = observation.get("pixels")
    return str(
        observation.get("screenshot_path")
        or observation.get("image_path")
        or (screenshot_value if isinstance(screenshot_value, str) else "")
        or screenshot.get("path")
        or screenshot.get("screenshot_path")
        or (pixels_value if isinstance(pixels_value, str) else "")
        or pixels.get("path")
        or pixels.get("screenshot_path")
        or ""
    ).strip()


def _screenshot_reference(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    image_bytes = resolved.read_bytes()
    with Image.open(resolved) as image:
        width, height = image.size
        mime_type = Image.MIME.get(image.format or "")
    mime_type = mime_type or mimetypes.guess_type(resolved.name)[0] or ""
    if mime_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise ValueError(f"screenshot_mime_type_unsupported:{resolved}:{mime_type}")
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(image_bytes).hexdigest(),
        "width": int(width),
        "height": int(height),
        "mime_type": mime_type,
    }


def _default_package_resolver(app_name: str) -> str:
    try:
        from src.integrations.android_world.apps import resolve_androidworld_package

        return resolve_androidworld_package(app_name)
    except (ImportError, KeyError, ValueError):
        return ""


def _validator_reward(payload: dict[str, Any], success: bool) -> float:
    for key in ("androidworld_reward", "validator_reward", "reward"):
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return max(0.0, float(value))
    return 1.0 if success else 0.0


def _success(value: dict[str, Any], *, default: bool) -> bool:
    for key in ("success", "run_success", "androidworld_success"):
        if key in value and value[key] is not None:
            return str(value[key]).strip().lower() not in {
                "",
                "0",
                "false",
                "no",
                "none",
            }
    return default


def _number(value: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        item = value.get(key)
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            return float(item)
    return None


def _integer(value: Any, default: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return int(default)


def _first(value: dict[str, Any], keys: tuple[str, ...]) -> Any:
    return next((value[key] for key in keys if value.get(key) is not None), None)


def _map(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return dict(value) if isinstance(value, dict) else {}


def _present(value: Any) -> bool:
    return value is not None and value != ""


__all__ = [
    "ScreenshotResolver",
    "adapt_source_run_log",
    "convert_legacy_run_log",
    "import_run_log",
    "import_run_log_evidence",
    "project_androidworld_step_actions",
]
