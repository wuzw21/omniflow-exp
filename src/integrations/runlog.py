from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from omniflow.core.trajectory import (
    CANONICAL_RUN_LOG_SCHEMA_VERSION,
    canonicalize_run_log,
)

_EXECUTION_TIMING_ARGS = {
    "post_action_wait_s",
    "post_wait_s",
    "wait_after_s",
}


def import_run_log(value: dict[str, Any]) -> dict[str, Any]:
    """Convert historical OOB/AndroidWorld data at the integration boundary."""
    run_log, _source_states = import_run_log_evidence(value)
    return run_log


def import_run_log_evidence(
    value: dict[str, Any],
    *,
    evidence_root: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Import one source RunLog and its source-only transfer state catalog."""
    payload = _map(value.get("payload")) or value
    payload = _map(payload.get("run_log")) or payload
    run_id = str(payload.get("run_id") or value.get("run_id") or "imported-run")
    raw_steps = payload.get("steps") or payload.get("cards") or []
    raw_step_values = raw_steps if isinstance(raw_steps, list) else []
    steps: list[dict[str, Any]] = []
    states: dict[str, dict[str, Any]] = {}
    for raw_step_index, raw_step in enumerate(raw_step_values):
        if not isinstance(raw_step, dict):
            continue
        before_state = _state(
            raw_step.get("state")
            or raw_step.get("observation")
            or raw_step.get("observation_before_act")
            or raw_step.get("before")
            or _map(raw_step.get("source_context")).get("src_ctx")
        )
        after_state = _state(
            raw_step.get("after_state")
            or raw_step.get("next_state")
            or raw_step.get("observation_after_act")
            or raw_step.get("after")
        )
        next_state: dict[str, Any] = {}
        if raw_step_index + 1 < len(raw_step_values):
            next_raw_step = raw_step_values[raw_step_index + 1]
            if isinstance(next_raw_step, dict):
                next_state = _state(
                    next_raw_step.get("state")
                    or next_raw_step.get("observation")
                    or next_raw_step.get("observation_before_act")
                    or next_raw_step.get("before")
                    or _map(next_raw_step.get("source_context")).get("src_ctx")
                )
        inferred_package_name = str(
            after_state.get("package_name")
            or next_state.get("package_name")
            or ""
        ).strip()
        raw_actions = raw_step.get("executed_actions") or raw_step.get("actions")
        raw_action_values = (
            raw_actions
            if isinstance(raw_actions, list)
            else [raw_step.get("action")]
        )
        for raw_action in raw_action_values:
            actions = _actions(
                raw_action or raw_step.get("tool_call") or raw_step,
                source_state=before_state,
                inferred_package_name=inferred_package_name,
            )
            for action in actions:
                index = len(steps)
                before_state_id = str(
                    before_state.get("state_id")
                    or _state_id(run_id, index, before_state)
                )
                _add_source_state(
                    states,
                    state_id=before_state_id,
                    state=before_state,
                    evidence_root=evidence_root,
                )
                after_state_id = str(
                    after_state.get("state_id")
                    or raw_step.get("after_state_id")
                    or _state_id(run_id, index + 1, after_state or before_state)
                )
                raw_result = _map(raw_step.get("result"))
                step_success = _success(
                    raw_result,
                    default=_success(raw_step, default=True),
                )
                result = {"success": step_success}
                result_error = str(
                    raw_result.get("error")
                    or raw_result.get("error_message")
                    or ""
                ).strip()
                if result_error:
                    result["error"] = result_error
                diagnostics = _map(raw_step.get("diagnostics")) or _map(
                    raw_step.get("metadata")
                )
                step = {
                    "step_index": index,
                    "before_state_id": before_state_id,
                    "action": action,
                    "result": result,
                    "after_state_id": after_state_id,
                }
                if diagnostics:
                    step["metadata"] = diagnostics
                steps.append(step)
    success = _success(payload, default=_success(value, default=False))
    canonical = {
        "schema_version": CANONICAL_RUN_LOG_SCHEMA_VERSION,
        "run_id": run_id,
        "goal": str(payload.get("goal") or payload.get("operation_description") or ""),
        "status": "succeeded" if success else "failed",
        "success": success,
        "steps": steps,
    }
    error = str(payload.get("error") or payload.get("error_message") or "").strip()
    if error:
        canonical["error"] = error
    diagnostics = _map(payload.get("diagnostics")) or _map(payload.get("metadata"))
    if diagnostics:
        canonical["diagnostics"] = diagnostics
    imported = canonicalize_run_log(canonical)
    return (
        imported,
        {
            "schema_version": "omniflow.transfer-state-catalog.v1",
            "run_id": imported["run_id"],
            "states": states,
        },
    )


def extract_canonical_step_actions(value: dict[str, Any]) -> list[dict[str, Any]]:
    imported = import_run_log(
        {
            "run_id": "step-adapter",
            "goal": "",
            "success": True,
            "steps": [value],
        }
    )
    return [dict(step["action"]) for step in imported["steps"]]


def _actions(
    value: Any,
    *,
    source_state: dict[str, Any] | None = None,
    inferred_package_name: str = "",
) -> list[dict[str, Any]]:
    raw = _map(value)
    function = _map(raw.get("function"))
    tool = str(
        raw.get("tool")
        or raw.get("type")
        or raw.get("name")
        or function.get("name")
        or ""
    ).strip()
    args = _map(
        raw.get("args")
        or raw.get("arguments")
        or raw.get("params")
        or function.get("arguments")
    )
    # Historical replay records stored pacing controls beside semantic action
    # arguments.  They remain available in the raw record to the replay
    # executor, but are not part of the canonical Action schema.
    for key in _EXECUTION_TIMING_ARGS:
        args.pop(key, None)
    historical = "type" in raw and "tool" not in raw
    if tool == "android_privileged_action":
        tool = str(args.pop("tool", "")).strip()
        args.update(_map(args.pop("arguments", None)))
    if tool == "wait":
        if "duration_ms" not in args:
            if "time_ms" in args:
                args["duration_ms"] = int(float(args["time_ms"]))
            elif "time_s" in args:
                args["duration_ms"] = int(float(args["time_s"]) * 1000)
        args.pop("time_ms", None)
        args.pop("time_s", None)
    if historical:
        if tool == "press_back":
            tool = "press_key"
            args = {"key": "back"}
        elif tool == "press_key":
            raw_key = str(args.pop("key", args.pop("keycode", ""))).strip()
            key = raw_key.lower().removeprefix("keycode_")
            if key == "del":
                key = "delete"
            args = {"key": key}
        elif tool in {"open_app", "start_activity"}:
            package_name = str(
                args.get("package_name") or inferred_package_name
            ).strip()
            tool = "open_app"
            args = {"package_name": package_name} if package_name else {}
        elif tool == "answer":
            return []
        if tool == "swipe":
            aliases = {
                "start_x": "x1",
                "start_y": "y1",
                "end_x": "x2",
                "end_y": "y2",
            }
            for old, new in aliases.items():
                if new not in args and old in args:
                    args[new] = args.pop(old)
            if "direction" not in args and all(
                key in args for key in ("x1", "y1", "x2", "y2")
            ):
                dx = float(args["x2"]) - float(args["x1"])
                dy = float(args["y2"]) - float(args["y1"])
                args["direction"] = (
                    ("right" if dx > 0 else "left")
                    if abs(dx) >= abs(dy)
                    else ("down" if dy > 0 else "up")
                )
        args.pop("clear_text", None)
        _normalize_historical_coordinates(
            tool=tool,
            args=args,
            source_state=source_state or {},
        )
        if tool == "input_text" and all(key in args for key in ("x", "y")):
            click_args = {"x": args.pop("x"), "y": args.pop("y")}
            return [
                {"tool": "click", "args": click_args},
                {"tool": "input_text", "args": args},
            ]
    return [{"tool": tool, "args": args}] if tool else []


def _normalize_historical_coordinates(
    *,
    tool: str,
    args: dict[str, Any],
    source_state: dict[str, Any],
) -> None:
    try:
        width = float(source_state["display_width"])
        height = float(source_state["display_height"])
    except (KeyError, TypeError, ValueError):
        return
    if width <= 0 or height <= 0:
        return
    axis_by_key: dict[str, float]
    if tool in {"click", "input_text", "long_press"}:
        axis_by_key = {"x": width, "y": height}
    elif tool == "swipe":
        axis_by_key = {
            "x1": width,
            "y1": height,
            "x2": width,
            "y2": height,
        }
    else:
        return
    for key, extent in axis_by_key.items():
        value = args.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        args[key] = float(value) / extent * 1000.0


def _state(value: Any) -> dict[str, Any]:
    raw = {"xml": value} if isinstance(value, str) else _map(value)
    aliases = {
        "state_id": ("state_id",),
        "xml": ("xml", "observation_xml", "page", "source_xml"),
        "xml_path": ("xml_path", "observation_xml_path", "page_path"),
        "xml_sha256": ("xml_sha256", "observation_xml_sha256", "page_sha256"),
        "xml_chars": ("xml_chars", "observation_xml_chars", "page_chars"),
        "xml_bytes": ("xml_bytes", "observation_xml_bytes", "page_bytes"),
        "screenshot_path": ("screenshot_path", "image_path"),
        "package_name": ("package_name", "packageName"),
        "activity_name": ("activity_name", "activityName"),
        "display_width": ("display_width", "screen_width", "width"),
        "display_height": ("display_height", "screen_height", "height"),
    }
    state = {
        output: item
        for output, names in aliases.items()
        if _present(item := _first(raw, names))
    }
    screenshot = _map(raw.get("screenshot"))
    state.setdefault("screenshot_path", _first(screenshot, ("path", "screenshot_path")))
    state.setdefault("display_width", _first(screenshot, ("display_width", "width")))
    state.setdefault("display_height", _first(screenshot, ("display_height", "height")))
    return {key: item for key, item in state.items() if _present(item)}


def _state_id(run_id: str, index: int, state: dict[str, Any]) -> str:
    identity = json.dumps(
        {"run_id": run_id, "step_index": index, "state": state},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "state_" + hashlib.sha256(identity.encode()).hexdigest()[:20]


def _add_source_state(
    states: dict[str, dict[str, Any]],
    *,
    state_id: str,
    state: dict[str, Any],
    evidence_root: str | Path | None,
) -> None:
    source_state: dict[str, Any] = {"state_id": state_id}
    xml = state.get("xml")
    if not _present(xml) and _present(state.get("xml_path")):
        xml_path = Path(str(state["xml_path"])).expanduser()
        if not xml_path.is_absolute() and evidence_root is not None:
            xml_path = Path(evidence_root).expanduser() / xml_path
        if xml_path.is_file():
            xml = xml_path.read_text(encoding="utf-8")
    if _present(xml):
        source_state["xml"] = str(xml)
    for key in ("package_name", "activity_name"):
        if _present(state.get(key)):
            source_state[key] = str(state[key])
    width = state.get("display_width")
    height = state.get("display_height")
    if _present(width) and _present(height):
        source_state["display"] = {
            "width": int(width),
            "height": int(height),
        }
    existing = states.get(state_id)
    if existing is not None and existing != source_state:
        raise ValueError(f"source_state_conflict:{state_id}")
    states[state_id] = source_state


def _success(value: dict[str, Any], *, default: bool) -> bool:
    for key in ("success", "run_success", "androidworld_success"):
        if key in value and value[key] is not None:
            return str(value[key]).strip().lower() not in {"", "0", "false", "no", "none"}
    return default


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
    "extract_canonical_step_actions",
    "import_run_log",
    "import_run_log_evidence",
]
