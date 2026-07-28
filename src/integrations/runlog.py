from __future__ import annotations

import hashlib
import json
from typing import Any

from omniflow.trajectory import (
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
    payload = _map(value.get("payload")) or value
    payload = _map(payload.get("run_log")) or payload
    run_id = str(payload.get("run_id") or value.get("run_id") or "imported-run")
    raw_steps = payload.get("steps") or payload.get("cards") or []
    steps: list[dict[str, Any]] = []
    for raw_step in raw_steps if isinstance(raw_steps, list) else []:
        if not isinstance(raw_step, dict):
            continue
        raw_actions = raw_step.get("executed_actions") or raw_step.get("actions")
        actions = raw_actions if isinstance(raw_actions, list) else [raw_step.get("action")]
        for raw_action in actions:
            action = _action(raw_action or raw_step.get("tool_call") or raw_step)
            if not action["tool"]:
                continue
            index = len(steps)
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
            before_state_id = str(
                before_state.get("state_id") or _state_id(run_id, index, before_state)
            )
            after_state_id = str(
                after_state.get("state_id")
                or raw_step.get("after_state_id")
                or _state_id(run_id, index + 1, after_state or before_state)
            )
            raw_result = _map(raw_step.get("result"))
            step_success = _success(raw_result, default=_success(raw_step, default=True))
            result = {"success": step_success}
            result_error = str(
                raw_result.get("error") or raw_result.get("error_message") or ""
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
    return canonicalize_run_log(canonical)


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


def _action(value: Any) -> dict[str, Any]:
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
    return {"tool": tool, "args": args}


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


__all__ = ["extract_canonical_step_actions", "import_run_log"]
