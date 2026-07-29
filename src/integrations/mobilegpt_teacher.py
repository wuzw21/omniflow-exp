"""Teacher-forced MobileGPT learning from AndroidWorld source run logs.

This module intentionally does not write MobileGPT memory files directly.
The learning run still goes through MobileGPT's own server, XML encoder,
explore/select agents, page manager, and ``Memory.save_task`` path.  The only
runtime override is the action produced by ``DeriveAgent.derive``: instead of
asking a VLM for the next primitive UI action, teacher mode migrates the next
AndroidWorld source action onto MobileGPT's current parsed screen.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable
import xml.etree.ElementTree as ET

MOBILEGPT_SUPPORTED_SOURCE_TYPES = {
    "click",
    "input_text",
    "long_press",
    "press_key",
    "swipe",
}

MOBILEGPT_INTERNAL_LAUNCH_ACTION = "__omniflow_launch_package"


@dataclass(frozen=True)
class TeacherActionResult:
    """One migrated MobileGPT action plus replay provenance."""

    action: dict[str, Any]
    source_action_type: str
    source_step_index: int
    source_action_index: int
    matched_index: str
    match_score: float
    match_reason: str
    consumed_source_action: bool = True


def load_teacher_actions(source_run_log: str | Path) -> list[dict[str, Any]]:
    """Load replayable AndroidWorld actions for MobileGPT teacher forcing."""

    path = Path(source_run_log).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    steps = _source_run_log_steps(payload)
    actions: list[dict[str, Any]] = []
    for step_index, step in enumerate(steps or []):
        if not isinstance(step, dict):
            continue
        for action_index, action in enumerate(_teacher_step_actions(step)):
            action_type = str(action.get("type") or "").strip()
            if action_type not in MOBILEGPT_SUPPORTED_SOURCE_TYPES:
                continue
            if action_type == "input_text":
                params = action.get("params")
                if isinstance(params, dict) and str(params.get("text") or "") == "":
                    continue
            if action_type == "press_key" and not _is_supported_press_key(action):
                    continue
            action = _ground_source_action_identity(step, action)
            actions.append(
                {
                    "source_step_index": step_index,
                    "source_action_index": action_index,
                    "action": action,
                }
            )
    return actions


def _teacher_step_actions(step: dict[str, Any]) -> list[dict[str, Any]]:
    raw_actions = step.get("executed_actions") or step.get("actions")
    if not isinstance(raw_actions, list):
        raw_action = step.get("action") or step.get("tool_call")
        raw_actions = [raw_action] if isinstance(raw_action, dict) else []
    actions: list[dict[str, Any]] = []
    for raw_action in raw_actions:
        if not isinstance(raw_action, dict) or raw_action.get("success") is False:
            continue
        function = (
            dict(raw_action.get("function") or {})
            if isinstance(raw_action.get("function"), dict)
            else {}
        )
        action_type = str(
            raw_action.get("type")
            or raw_action.get("tool")
            or raw_action.get("name")
            or function.get("name")
            or ""
        ).strip()
        raw_params = (
            raw_action.get("params")
            or raw_action.get("args")
            or raw_action.get("arguments")
            or function.get("arguments")
            or {}
        )
        params = dict(raw_params) if isinstance(raw_params, dict) else {}
        if action_type == "android_privileged_action":
            action_type = str(params.pop("tool", "") or "").strip()
            nested = params.pop("arguments", None)
            if isinstance(nested, dict):
                params.update(nested)
        actions.append(
            {
                "type": action_type,
                "params": params,
                "description": str(raw_action.get("description") or "").strip(),
            }
        )
    return actions


_SOURCE_COORDINATE_KEYS = {
    "bounds",
    "end_x",
    "end_y",
    "start_x",
    "start_y",
    "x",
    "x1",
    "x2",
    "y",
    "y1",
    "y2",
}


def _ground_source_action_identity(
    step: dict[str, Any],
    action: dict[str, Any],
) -> dict[str, Any]:
    grounded = dict(action)
    params = dict(grounded.get("params") or {})
    action_type = str(grounded.get("type") or "").strip()
    if action_type == "swipe" and not str(params.get("direction") or "").strip():
        params["direction"] = _source_swipe_direction(params)

    source_context = (
        dict(params.get("source_context") or {})
        if isinstance(params.get("source_context"), dict)
        else {}
    )
    observation = _source_step_observation(step)
    page = str(source_context.get("page") or observation.get("xml") or "").strip()
    element = (
        dict(source_context.get("element") or {})
        if isinstance(source_context.get("element"), dict)
        else {}
    )
    if not element and page:
        element = _source_element_at_action_point(page, params)
    element = {
        key: value
        for key, value in element.items()
        if key
        in {
            "content-desc",
            "content_desc",
            "description",
            "label",
            "resource-id",
            "resource_id",
            "text",
        }
    }
    package_name = str(
        source_context.get("package_name")
        or observation.get("package_name")
        or observation.get("packageName")
        or ""
    ).strip()
    if page:
        source_context["page"] = page
    if element:
        source_context["element"] = element
    if package_name:
        source_context["package_name"] = package_name
    if source_context:
        params["source_context"] = source_context
    label = str(
        element.get("text")
        or element.get("description")
        or element.get("content_desc")
        or element.get("content-desc")
        or element.get("resource_id")
        or element.get("resource-id")
        or ""
    ).strip()
    if label and not str(params.get("target_description") or "").strip():
        params["target_description"] = label
    for key in _SOURCE_COORDINATE_KEYS:
        params.pop(key, None)
    grounded["params"] = params
    return grounded


def _source_step_observation(step: dict[str, Any]) -> dict[str, Any]:
    for key in (
        "observation_before_act",
        "observation",
        "state",
        "before",
    ):
        value = step.get(key)
        if isinstance(value, dict):
            return dict(value)
    return {}


def _source_element_at_action_point(
    page: str,
    params: dict[str, Any],
) -> dict[str, str]:
    try:
        x = float(params.get("x"))
        y = float(params.get("y"))
        root = ET.fromstring(page)
    except (TypeError, ValueError, ET.ParseError):
        return {}
    candidates: list[tuple[float, int, ET.Element]] = []

    def visit(node: ET.Element, depth: int) -> None:
        bounds = str(node.attrib.get("bounds") or "")
        match = re.fullmatch(
            r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]",
            bounds,
        )
        if match:
            left, top, right, bottom = [float(value) for value in match.groups()]
            if left <= x <= right and top <= y <= bottom:
                area = max(1.0, (right - left) * (bottom - top))
                candidates.append((area, -depth, node))
        for child in list(node):
            visit(child, depth + 1)

    visit(root, 0)
    candidates.sort(key=lambda item: (item[0], item[1]))
    for _, _, node in candidates:
        text, description, resource_id = _element_identity(node)
        if text or description or resource_id:
            return {
                "text": text,
                "description": description,
                "resource_id": resource_id,
            }
    return {}


def _source_run_log_steps(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    steps = payload.get("steps")
    if isinstance(steps, list):
        return [dict(step) for step in steps if isinstance(step, dict)]

    wrapped_payload = payload.get("payload")
    if isinstance(wrapped_payload, dict):
        wrapped_steps = wrapped_payload.get("steps")
        if isinstance(wrapped_steps, list):
            return [dict(step) for step in wrapped_steps if isinstance(step, dict)]
        cards = wrapped_payload.get("cards")
        if isinstance(cards, list):
            return [
                _payload_card_to_step(card)
                for card in cards
                if isinstance(card, dict)
            ]
    return []


def _payload_card_to_step(card: dict[str, Any]) -> dict[str, Any]:
    tool_call = (
        dict(card.get("tool_call") or {})
        if isinstance(card.get("tool_call"), dict)
        else {}
    )
    action_type = str(
        card.get("action_type")
        or card.get("tool_name")
        or card.get("toolName")
        or tool_call.get("name")
        or ""
    ).strip()
    raw_params = card.get("params")
    if not isinstance(raw_params, dict):
        raw_params = tool_call.get("params")
    if not isinstance(raw_params, dict):
        raw_params = tool_call.get("arguments")
    params = dict(raw_params or {}) if isinstance(raw_params, dict) else {}

    source_context = _payload_card_source_context(card, params)
    if source_context:
        params.setdefault("source_context", source_context)

    action = {
        "type": action_type,
        "params": params,
        "success": bool(card.get("success", True)),
        "description": str(card.get("summary") or card.get("title") or "").strip(),
    }
    if source_context:
        action["source_context"] = source_context

    observation_before = {
        "xml": source_context.get("page", ""),
        "package_name": str(card.get("package_name") or ""),
    }
    return {
        "step_index": card.get("step_index"),
        "status": card.get("status"),
        "success": bool(card.get("success", True)),
        "tool_call": {
            "name": action_type,
            "params": params,
            "reason": action["description"] or None,
        },
        "actions": [action],
        "executed_actions": [action],
        "observation_before_act": observation_before,
        "target_element": source_context.get("element", {}),
    }


def _payload_card_source_context(
    card: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    raw_context = (
        dict(card.get("source_context") or {})
        if isinstance(card.get("source_context"), dict)
        else {}
    )
    src_ctx = (
        dict(raw_context.get("src_ctx") or {})
        if isinstance(raw_context.get("src_ctx"), dict)
        else raw_context
    )
    source_context: dict[str, Any] = {}
    page = str(src_ctx.get("page") or src_ctx.get("xml") or "").strip()
    if page:
        source_context["page"] = page

    target_evidence = (
        dict(params.get("target_evidence") or {})
        if isinstance(params.get("target_evidence"), dict)
        else {}
    )
    label = str(
        target_evidence.get("label")
        or params.get("target_description")
        or card.get("title")
        or ""
    ).strip()
    resource_id = str(
        target_evidence.get("resource_id")
        or target_evidence.get("resource-id")
        or ""
    ).strip()
    if label or resource_id:
        source_context["element"] = {
            "text": label,
            "description": label,
            "resource_id": resource_id,
        }
    package_name = str(card.get("package_name") or "").strip()
    if package_name:
        source_context["package_name"] = package_name
    return source_context


class MobileGPTTeacher:
    """Cursor over AndroidWorld source actions migrated to MobileGPT actions."""

    def __init__(self, source_run_log: str | Path) -> None:
        self.source_run_log = str(Path(source_run_log).expanduser().resolve())
        self._actions = load_teacher_actions(self.source_run_log)
        self._cursor = 0
        self.instruction = ""
        self.task: dict[str, Any] = {}
        self.last_emitted_result: TeacherActionResult | None = None
        self._emitted_preflight_keys: set[str] = set()

    @property
    def exhausted(self) -> bool:
        return self._cursor >= len(self._actions)

    @property
    def action_count(self) -> int:
        return len(self._actions)

    @property
    def cursor(self) -> int:
        return self._cursor

    def current_record(self) -> dict[str, Any] | None:
        if self.exhausted:
            return None
        return dict(self._actions[self._cursor])

    def skip_noop_actions(self, screen: str) -> list[dict[str, Any]]:
        skipped: list[dict[str, Any]] = []
        while not self.exhausted:
            record = self._actions[self._cursor]
            action = dict(record["action"])
            if not _is_noop_source_action(action, screen):
                break
            skipped.append(
                {
                    "cursor": self._cursor,
                    "source_step_index": record.get("source_step_index"),
                    "source_action_index": record.get("source_action_index"),
                    "source_action_type": action.get("type"),
                    "source_action_label": _source_action_label(action),
                }
            )
            self._cursor += 1
        return skipped

    def skip_current_action(self) -> None:
        if not self.exhausted:
            self._cursor += 1

    def mark_exhausted(self) -> None:
        self._cursor = len(self._actions)

    def reset(self, *, instruction: str = "", task: dict[str, Any] | None = None) -> None:
        self._cursor = 0
        self.instruction = str(instruction or "")
        self.task = dict(task or {})
        self.last_emitted_result = None
        self._emitted_preflight_keys.clear()

    def next_action(self, screen: str) -> TeacherActionResult:
        if self.exhausted:
            raise RuntimeError("MobileGPT teacher source run log is exhausted")
        record = self._actions[self._cursor]
        action = dict(record["action"])
        source_type = str(action.get("type") or "").strip()
        app_switch = _source_app_switch_preflight(
            action,
            screen,
            current_app_package=str(self.task.get("app") or ""),
        )
        if app_switch is not None:
            return TeacherActionResult(
                action={
                    "name": MOBILEGPT_INTERNAL_LAUNCH_ACTION,
                    "parameters": {"package_name": app_switch["package_name"]},
                },
                source_action_type="target_preflight_app_switch",
                source_step_index=int(record["source_step_index"]),
                source_action_index=int(record["source_action_index"]),
                matched_index="",
                match_score=100.0,
                match_reason=str(app_switch["reason"]),
                consumed_source_action=False,
            )
        browser_url_preflight = _browser_task_url_preflight(
            action,
            screen,
            cursor=self._cursor,
            emitted_preflight_keys=self._emitted_preflight_keys,
        )
        if browser_url_preflight is not None:
            return TeacherActionResult(
                action={
                    "name": MOBILEGPT_INTERNAL_LAUNCH_ACTION,
                    "parameters": {"package_name": "com.android.chrome"},
                },
                source_action_type="target_preflight_browser_url",
                source_step_index=int(record["source_step_index"]),
                source_action_index=int(record["source_action_index"]),
                matched_index="",
                match_score=100.0,
                match_reason=str(browser_url_preflight["reason"]),
                consumed_source_action=False,
            )
        try:
            migrated = migrate_source_action_to_mobilegpt(action, screen)
        except Exception:
            preflight = _target_preflight_action(screen)
            if preflight is not None:
                return preflight
            raise

        preflight = _target_preflight_action(screen)
        if preflight is not None and float(migrated.get("match_score") or 0.0) < 8.0:
            return preflight

        self._cursor += 1
        return TeacherActionResult(
            action=migrated["action"],
            source_action_type=source_type,
            source_step_index=int(record["source_step_index"]),
            source_action_index=int(record["source_action_index"]),
            matched_index=str(migrated["matched_index"]),
            match_score=float(migrated["match_score"]),
            match_reason=str(migrated["match_reason"]),
        )


def migrate_source_action_to_mobilegpt(
    source_action: dict[str, Any],
    current_screen: str,
) -> dict[str, Any]:
    """Convert one AndroidWorld source action to a MobileGPT primitive action.

    The output uses MobileGPT's action schema: ``click`` / ``input`` /
    ``long-click`` / ``scroll`` with an integer UI ``index`` from the current
    MobileGPT parsed screen.
    """

    action_type = str(source_action.get("type") or "").strip()
    params = dict(source_action.get("params") or {})
    if action_type not in MOBILEGPT_SUPPORTED_SOURCE_TYPES:
        raise ValueError(f"Unsupported teacher action type: {action_type}")

    if action_type == "press_key":
        match = _back_action_match(current_screen)
        if match is None:
            raise RuntimeError(
                "Unable to migrate source action to current MobileGPT screen: "
                f"{_source_action_label(source_action)!r}"
            )
        if match["index"]:
            mobile_action = {"name": "click", "parameters": {"index": str(match["index"])}}
        else:
            mobile_action = {"name": "back", "parameters": {}}
        return {
            "action": mobile_action,
            "matched_index": str(match["index"]),
            "match_score": float(match["score"]),
            "match_reason": str(match["reason"]),
        }

    match = _best_current_screen_match(source_action, current_screen)
    if match is None:
        raise RuntimeError(
            "Unable to migrate source action to current MobileGPT screen: "
            f"{_source_action_label(source_action)!r}"
        )

    index = str(match["index"])
    if action_type == "click":
        mobile_action = {"name": "click", "parameters": {"index": index}}
    elif action_type == "long_press":
        mobile_action = {"name": "long-click", "parameters": {"index": index}}
    elif action_type == "input_text":
        mobile_action = {
            "name": "input",
            "parameters": {
                "index": index,
                "input_text": str(params.get("text") or ""),
            },
        }
    else:
        mobile_action = {
            "name": "scroll",
            "parameters": {
                "index": index,
                "direction": _source_swipe_direction(params),
            },
        }

    return {
        "action": mobile_action,
        "matched_index": index,
        "match_score": float(match["score"]),
        "match_reason": str(match["reason"]),
    }


def install_mobilegpt_teacher(
    *,
    source_run_log: str | Path,
    fallback_to_vlm_on_miss: bool = False,
    stats_writer: Callable[[dict[str, Any]], None] | None = None,
) -> MobileGPTTeacher:
    """Patch MobileGPT runtime classes for a teacher-forced learning server."""

    teacher = MobileGPTTeacher(source_run_log)
    writer = stats_writer or _write_stats_event

    from agents.derive_agent import DeriveAgent
    from mobilegpt import MobileGPT
    from utils import parsing_utils

    original_init = MobileGPT.init
    original_get_next_action = getattr(MobileGPT, "get_next_action", None)
    original_handle_action_error = getattr(MobileGPT, "handle_action_error", None)
    original_teacher_save_subtask = getattr(
        MobileGPT,
        "_MobileGPT__teacher_save_subtask",
        None,
    )
    original_derive = DeriveAgent.derive
    finish_name = f"_{MobileGPT.__name__}__finish_task"

    def _teacher_example_for_action(
        *,
        screen: str,
        action: dict[str, Any],
        response: dict[str, Any],
        instruction: str,
        subtask: dict[str, Any],
    ) -> dict[str, Any]:
        example_screen = str(screen or "")
        index = action.get("parameters", {}).get("index")
        if index is not None:
            try:
                example_screen = parsing_utils.shrink_screen_xml(screen, int(index))
            except Exception:
                example_screen = str(screen or "")
        return {
            "instruction": instruction,
            "subtask": json.dumps(subtask, ensure_ascii=False),
            "screen": example_screen,
            "response": json.dumps(response, ensure_ascii=False),
        }

    def _teacher_result_payload(
        result: TeacherActionResult | None,
    ) -> dict[str, Any]:
        if result is None:
            return {}
        return {
            "mobilegpt_action": result.action,
            "source_action_type": result.source_action_type,
            "source_step_index": result.source_step_index,
            "source_action_index": result.source_action_index,
            "matched_index": result.matched_index,
            "match_score": result.match_score,
            "match_reason": result.match_reason,
            "consumed_source_action": result.consumed_source_action,
        }

    def patched_init(self, instruction: str, task: dict, is_new_task: bool):
        teacher.reset(instruction=instruction, task=task)
        writer(
            {
                "event": "mobilegpt_teacher_started",
                "instruction": instruction,
                "task": task,
                "source_run_log": teacher.source_run_log,
                "teacher_action_count": teacher.action_count,
                "original_is_new_task": bool(is_new_task),
                "forced_is_new_task": True,
            }
        )
        return original_init(self, instruction, task, True)

    def patched_get_next_action(self, parsed_xml=None, hierarchy_xml=None, encoded_xml=None):
        action = original_get_next_action(self, parsed_xml, hierarchy_xml, encoded_xml)
        if _is_internal_launch_action(action):
            _remove_last_internal_action(self, action)
            writer(
                {
                    "event": "mobilegpt_teacher_internal_launch_action",
                    "instruction": getattr(self, "instruction", ""),
                    "source_run_log": teacher.source_run_log,
                    "cursor": teacher.cursor,
                    "teacher_action_count": teacher.action_count,
                    "action": action,
                }
            )
        return action

    def patched_teacher_save_subtask(self, subtask: dict, screen: str, response: dict | None):
        if not isinstance(subtask, dict):
            return None
        name = str(subtask.get("name") or "").strip()
        if not name or name in {"finish", "speak", "scroll_screen", "read_screen"}:
            return None
        memory = getattr(self, "memory", None)
        if memory is None or not hasattr(memory, "save_subtask"):
            if original_teacher_save_subtask is not None:
                return original_teacher_save_subtask(self, subtask, screen, response)
            return None
        normalized = dict(subtask)
        if not isinstance(normalized.get("parameters"), dict):
            normalized["parameters"] = {}
        normalized.setdefault(
            "description",
            "Teacher-forced AndroidWorld replay subtask.",
        )
        example_response = dict(response or {})
        example_response["action"] = normalized
        example_response.setdefault(
            "reasoning",
            "Teacher-forced AndroidWorld replay subtask.",
        )
        example_response.pop("completion_rate", None)
        example = {
            "instruction": str(getattr(self, "instruction", "") or ""),
            "screen": str(screen or ""),
            "response": example_response,
        }
        memory.save_subtask(normalized, example)
        writer(
            {
                "event": "mobilegpt_teacher_subtask_saved",
                "instruction": str(getattr(self, "instruction", "") or ""),
                "source_run_log": teacher.source_run_log,
                "subtask": normalized,
            }
        )
        return None

    def patched_handle_action_error(self, reason: str):
        if original_handle_action_error is not None and not str(reason or "").strip():
            return original_handle_action_error(self, reason)

        failed_result = getattr(teacher, "last_emitted_result", None)
        writer(
            {
                "event": "mobilegpt_teacher_action_error",
                "instruction": getattr(self, "instruction", ""),
                "source_run_log": teacher.source_run_log,
                "cursor": teacher.cursor,
                "teacher_action_count": teacher.action_count,
                "reason": str(reason or ""),
                "failed_action": _teacher_result_payload(failed_result),
            }
        )
        if str(
            os.environ.get("MOBILEGPT_TEACHER_FAIL_ON_ACTION_ERROR") or ""
        ).strip() == "1":
            teacher.mark_exhausted()
            writer(
                {
                    "event": "mobilegpt_teacher_action_error_fail_closed",
                    "instruction": getattr(self, "instruction", ""),
                    "source_run_log": teacher.source_run_log,
                    "cursor": teacher.cursor,
                    "teacher_action_count": teacher.action_count,
                    "reason": str(reason or ""),
                    "failed_action": _teacher_result_payload(failed_result),
                }
            )
            return getattr(self, finish_name)()

        if (
            isinstance(failed_result, TeacherActionResult)
            and not failed_result.consumed_source_action
        ):
            teacher.skip_current_action()
            writer(
                {
                    "event": "mobilegpt_teacher_action_error_skipped_unconsumed",
                    "instruction": getattr(self, "instruction", ""),
                    "source_run_log": teacher.source_run_log,
                    "cursor": teacher.cursor,
                    "teacher_action_count": teacher.action_count,
                    "reason": str(reason or ""),
                    "failed_action": _teacher_result_payload(failed_result),
                }
            )

        try:
            action = self.get_next_action(
                getattr(self, "parsed_xml", ""),
                getattr(self, "hierarchy_xml", ""),
                getattr(self, "encoded_xml", ""),
            )
        except Exception as exc:
            artifact = _write_teacher_miss_artifacts(
                teacher=teacher,
                screen=getattr(self, "encoded_xml", ""),
                error=f"Action error recovery failed after {reason}: {exc}",
                instruction=getattr(self, "instruction", ""),
                subtask=getattr(self, "current_subtask", {}) or {},
            )
            writer(
                {
                    "event": "mobilegpt_teacher_action_error_recovery_failed",
                    "instruction": getattr(self, "instruction", ""),
                    "source_run_log": teacher.source_run_log,
                    "cursor": teacher.cursor,
                    "teacher_action_count": teacher.action_count,
                    "reason": str(reason or ""),
                    "error": str(exc),
                    "artifact": artifact,
                }
            )
            if original_handle_action_error is not None:
                return original_handle_action_error(self, reason)
            return None

        writer(
            {
                "event": "mobilegpt_teacher_action_error_recovered",
                "instruction": getattr(self, "instruction", ""),
                "source_run_log": teacher.source_run_log,
                "cursor": teacher.cursor,
                "teacher_action_count": teacher.action_count,
                "reason": str(reason or ""),
                "next_action": action,
            }
        )
        return action

    def patched_derive(self, screen: str, examples=None):
        skipped_noops = teacher.skip_noop_actions(screen)
        if skipped_noops:
            writer(
                {
                    "event": "mobilegpt_teacher_skipped_noop",
                    "instruction": getattr(self, "instruction", ""),
                    "source_run_log": teacher.source_run_log,
                    "cursor": teacher.cursor,
                    "teacher_action_count": teacher.action_count,
                    "skipped_count": len(skipped_noops),
                    "skipped_actions": skipped_noops,
                }
            )
        if teacher.exhausted:
            response = {
                "reasoning": "AndroidWorld teacher source actions are exhausted.",
                "action": {"name": "finish", "parameters": {}},
                "completion_rate": 100,
            }
            teacher.last_emitted_result = None
            self.response_history.append(response)
            writer(
                {
                    "event": "mobilegpt_teacher_finish_derived",
                    "instruction": getattr(self, "instruction", ""),
                    "source_run_log": teacher.source_run_log,
                    "cursor": teacher.cursor,
                    "teacher_action_count": teacher.action_count,
                }
            )
            return response["action"], {}
        try:
            result = teacher.next_action(screen)
        except Exception as exc:
            artifact = _write_teacher_miss_artifacts(
                teacher=teacher,
                screen=screen,
                error=str(exc),
                instruction=getattr(self, "instruction", ""),
                subtask=getattr(self, "subtask", {}) or {},
            )
            writer(
                {
                    "event": "mobilegpt_teacher_miss",
                    "instruction": getattr(self, "instruction", ""),
                    "source_run_log": teacher.source_run_log,
                    "cursor": teacher.cursor,
                    "error": str(exc),
                    "fallback_to_vlm": bool(fallback_to_vlm_on_miss),
                    "artifact": artifact,
                }
            )
            if fallback_to_vlm_on_miss:
                teacher.skip_current_action()
                return original_derive(self, screen, examples=examples)
            teacher.mark_exhausted()
            response = {
                "reasoning": (
                    "Teacher-forced AndroidWorld source action could not be "
                    "migrated to the current MobileGPT screen; fail closed."
                ),
                "action": {"name": "finish", "parameters": {}},
                "completion_rate": 0,
            }
            self.response_history.append(response)
            writer(
                {
                    "event": "mobilegpt_teacher_failed_finish",
                    "instruction": getattr(self, "instruction", ""),
                    "source_run_log": teacher.source_run_log,
                    "cursor": teacher.cursor,
                    "teacher_action_count": teacher.action_count,
                    "error": str(exc),
                }
            )
            return response["action"], {}

        teacher.last_emitted_result = result
        if _is_internal_launch_action(result.action):
            writer(
                {
                    "event": "mobilegpt_teacher_preflight_app_switch",
                    "instruction": getattr(self, "instruction", ""),
                    "source_run_log": teacher.source_run_log,
                    "cursor": teacher.cursor,
                    "teacher_action_count": teacher.action_count,
                    "source_step_index": result.source_step_index,
                    "source_action_index": result.source_action_index,
                    "mobilegpt_action": result.action,
                    "match_reason": result.match_reason,
                }
            )
            return result.action, {}
        completion_rate = int(
            min(99, round(100.0 * teacher.cursor / max(teacher.action_count, 1)))
        )
        response = {
            "reasoning": (
                "Teacher-forced from AndroidWorld source run log "
                f"step {result.source_step_index} action {result.source_action_index}."
            ),
            "action": result.action,
            "completion_rate": completion_rate,
        }
        self.response_history.append(response)
        self.action_history.append(
            "your past response: "
            + json.dumps(response, ensure_ascii=False)
            + " has been executed successfully."
        )
        example = _teacher_example_for_action(
            screen=screen,
            action=result.action,
            response=response,
            instruction=getattr(self, "instruction", ""),
            subtask=getattr(self, "subtask", {}) or {},
        )
        writer(
            {
                "event": (
                    "mobilegpt_teacher_action"
                    if result.consumed_source_action
                    else "mobilegpt_teacher_preflight_action"
                ),
                "instruction": getattr(self, "instruction", ""),
                "source_run_log": teacher.source_run_log,
                "cursor": teacher.cursor,
                "teacher_action_count": teacher.action_count,
                "source_action_type": result.source_action_type,
                "source_step_index": result.source_step_index,
                "source_action_index": result.source_action_index,
                "mobilegpt_action": result.action,
                "matched_index": result.matched_index,
                "match_score": result.match_score,
                "match_reason": result.match_reason,
            }
        )
        return response["action"], example

    MobileGPT.init = patched_init
    if original_get_next_action is not None:
        MobileGPT.get_next_action = patched_get_next_action
    setattr(MobileGPT, "handle_action_error", patched_handle_action_error)
    if original_teacher_save_subtask is not None:
        setattr(
            MobileGPT,
            "_MobileGPT__teacher_save_subtask",
            patched_teacher_save_subtask,
        )
    DeriveAgent.derive = patched_derive
    return teacher


def _target_preflight_action(current_screen: str) -> TeacherActionResult | None:
    """Handle target-only setup screens without consuming a source action."""

    try:
        root = ET.fromstring(str(current_screen or ""))
    except Exception:
        return None

    setup_labels = {
        "accept & continue",
        "accept and continue",
        "use without an account",
        "no thanks",
        "not now",
        "skip",
        "got it",
    }
    for element in root.iter():
        index = str(element.attrib.get("index") or "").strip()
        if not index:
            continue
        tag = str(element.tag or "").strip().lower()
        if tag not in {"button", "checker"}:
            continue
        text, description, resource_id = _element_identity(element)
        combined = _norm(" ".join([text, description, resource_id]))
        if not any(label in combined for label in setup_labels):
            continue
        return TeacherActionResult(
            action={"name": "click", "parameters": {"index": index}},
            source_action_type="target_preflight",
            source_step_index=-1,
            source_action_index=-1,
            matched_index=index,
            match_score=100.0,
            match_reason="target_preflight_setup_button",
            consumed_source_action=False,
        )
    return None


def _is_internal_launch_action(action: Any) -> bool:
    return (
        isinstance(action, dict)
        and str(action.get("name") or "").strip() == MOBILEGPT_INTERNAL_LAUNCH_ACTION
    )


def _remove_last_internal_action(mobilegpt: Any, action: dict[str, Any]) -> None:
    current_subtask_data = getattr(mobilegpt, "current_subtask_data", None)
    if not isinstance(current_subtask_data, dict):
        return
    actions = current_subtask_data.get("actions")
    if not isinstance(actions, list) or not actions:
        return
    last = actions[-1]
    last_action = last.get("action") if isinstance(last, dict) else None
    if last_action == action or _is_internal_launch_action(last_action):
        actions.pop()


def _source_app_switch_preflight(
    source_action: dict[str, Any],
    current_screen: str,
    *,
    current_app_package: str = "",
) -> dict[str, str] | None:
    if str(source_action.get("type") or "").strip() == "press_key":
        return None
    source_package = _source_action_package(source_action)
    if not source_package:
        return None
    current_package = _screen_package(current_screen)
    if current_package:
        if current_package == source_package:
            return None
    else:
        task_package = str(current_app_package or "").strip()
        if not task_package or task_package == source_package:
            return None
    if _screen_contains_source_target(source_action, current_screen):
        return None
    return {
        "package_name": source_package,
        "reason": f"source_package:{source_package}:current_package:{current_package or 'unknown'}",
    }


def _browser_task_url_preflight(
    source_action: dict[str, Any],
    current_screen: str,
    *,
    cursor: int,
    emitted_preflight_keys: set[str],
) -> dict[str, str] | None:
    task_url = str(os.getenv("OMNIFLOW_MOBILEGPT_BROWSER_TASK_URL") or "").strip()
    if not task_url:
        return None
    if _screen_contains_source_target(source_action, current_screen):
        return None
    source_page = _source_action_page(source_action)
    if not _looks_like_browser_task_source(source_page, source_action):
        return None
    if not _looks_like_chrome_home_screen(current_screen):
        return None
    key = f"browser_task_url:{cursor}:{task_url}"
    if key in emitted_preflight_keys:
        return None
    _open_browser_task_url(task_url)
    emitted_preflight_keys.add(key)
    return {"reason": f"browser_task_url:{task_url}:cursor:{cursor}"}


def _looks_like_browser_task_source(source_page: str, source_action: dict[str, Any]) -> bool:
    page_text = _norm(source_page)
    action_label = _norm(_source_action_label(source_action))
    if "maze puzzle" in page_text and action_label in {"up", "down", "left", "right"}:
        return True
    return (
        "android webkit webview" in page_text
        and all(label in page_text for label in ("up", "down", "left", "right"))
        and action_label in {"up", "down", "left", "right"}
    )


def _looks_like_chrome_home_screen(current_screen: str) -> bool:
    text = _norm(current_screen)
    if "com android chrome" not in text and "search or type web address" not in text:
        return False
    return (
        "search or type web address" in text
        or "search_box_text" in text
        or "feed_stream_recycler_view" in text
        or "new tab" in text
    )


def _open_browser_task_url(task_url: str) -> None:
    adb_path = str(os.getenv("OMNIFLOW_MOBILEGPT_ADB_PATH") or "adb").strip() or "adb"
    serial = str(os.getenv("ANDROID_SERIAL") or "").strip()
    argv = [adb_path]
    if serial:
        argv.extend(["-s", serial])
    argv.extend(
        [
            "shell",
            "am",
            "start",
            "-a",
            "android.intent.action.VIEW",
            "-d",
            task_url,
            "com.android.chrome",
        ]
    )
    subprocess.run(
        argv,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=15,
    )


def _source_action_package(source_action: dict[str, Any]) -> str:
    params = dict(source_action.get("params") or {}) if isinstance(source_action.get("params"), dict) else {}
    direct = str(
        params.get("package_name")
        or params.get("packageName")
        or source_action.get("package_name")
        or source_action.get("packageName")
        or ""
    ).strip()
    if direct:
        return direct

    source_context = (
        dict(params.get("source_context") or {})
        if isinstance(params.get("source_context"), dict)
        else {}
    )
    context_package = str(source_context.get("package_name") or "").strip()
    if context_package:
        return context_package

    page_package = _screen_package(str(source_context.get("page") or ""))
    if page_package:
        return page_package

    evidence = (
        dict(params.get("target_evidence") or {})
        if isinstance(params.get("target_evidence"), dict)
        else {}
    )
    for value in (
        evidence.get("resource_id"),
        evidence.get("resource-id"),
        evidence.get("label"),
        params.get("target_description"),
    ):
        package = _resource_package_from_text(str(value or ""))
        if package:
            return package
    return ""


def _source_action_page(source_action: dict[str, Any]) -> str:
    params = (
        dict(source_action.get("params") or {})
        if isinstance(source_action.get("params"), dict)
        else {}
    )
    source_context = (
        dict(params.get("source_context") or {})
        if isinstance(params.get("source_context"), dict)
        else {}
    )
    page = str(source_context.get("page") or "").strip()
    if page:
        return page
    direct_context = (
        dict(source_action.get("source_context") or {})
        if isinstance(source_action.get("source_context"), dict)
        else {}
    )
    return str(direct_context.get("page") or "").strip()


def _screen_package(screen: str) -> str:
    text = str(screen or "")
    if not text.strip():
        return ""
    packages: dict[str, int] = {}
    for package in re.findall(r"([A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+):id/", text):
        if package in {"android"}:
            continue
        packages[package] = packages.get(package, 0) + 1
    if packages:
        return sorted(packages.items(), key=lambda item: (-item[1], item[0]))[0][0]

    for package in re.findall(r"\b((?:com|net|org)\.[A-Za-z0-9_.]+)\b", text):
        cleaned = package.rstrip(".,;:'\")(")
        if cleaned and cleaned != "com.example.MobileGPT":
            return cleaned
    return ""


def _resource_package_from_text(value: str) -> str:
    match = re.search(r"([A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+):id/", value)
    if match and match.group(1) != "android":
        return match.group(1)
    match = re.search(r"\b((?:com|net|org)\.[A-Za-z0-9_.]+)\b", value)
    return match.group(1).rstrip(".,;:'\")(") if match else ""


def _screen_contains_source_target(
    source_action: dict[str, Any],
    current_screen: str,
) -> bool:
    try:
        return _best_current_screen_match(source_action, current_screen) is not None
    except Exception:
        return False


def _write_teacher_miss_artifacts(
    *,
    teacher: MobileGPTTeacher,
    screen: str,
    error: str,
    instruction: str,
    subtask: dict[str, Any],
) -> dict[str, Any]:
    root = _teacher_artifact_root()
    if root is None:
        return {}

    record = teacher.current_record() or {}
    action = (
        dict(record.get("action") or {})
        if isinstance(record.get("action"), dict)
        else {}
    )
    params = dict(action.get("params") or {}) if isinstance(action.get("params"), dict) else {}
    source_context = (
        dict(params.get("source_context") or {})
        if isinstance(params.get("source_context"), dict)
        else {}
    )
    source_xml = str(source_context.get("page") or "").strip()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    artifact_dir = root / f"miss_{stamp}_cursor{teacher.cursor}"
    try:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        target_xml_path = artifact_dir / "target_screen.xml"
        target_xml_path.write_text(str(screen or ""), encoding="utf-8")
        source_xml_path = artifact_dir / "source_screen.xml"
        if source_xml:
            source_xml_path.write_text(source_xml, encoding="utf-8")

        mobilegpt_log_dir = str(os.getenv("MOBILEGPT_CURRENT_LOG_DIRECTORY") or "").strip()
        screen_index = str(os.getenv("MOBILEGPT_CURRENT_SCREEN_INDEX") or "").strip()
        target_paths = _mobilegpt_current_artifact_paths(mobilegpt_log_dir, screen_index)
        payload = {
            "error": str(error),
            "instruction": str(instruction or ""),
            "subtask": dict(subtask or {}),
            "source_run_log": teacher.source_run_log,
            "cursor": teacher.cursor,
            "teacher_action_count": teacher.action_count,
            "source_step_index": record.get("source_step_index"),
            "source_action_index": record.get("source_action_index"),
            "source_action": action,
            "source_action_label": _source_action_label(action) if action else "",
            "target_xml": str(target_xml_path),
            "source_xml": str(source_xml_path) if source_xml else "",
            "mobilegpt_log_dir": mobilegpt_log_dir,
            "mobilegpt_screen_index": screen_index,
            "target_paths": target_paths,
        }
        miss_json_path = artifact_dir / "miss.json"
        miss_json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return {"artifact_dir": str(artifact_dir), "miss_json": str(miss_json_path)}
    except Exception as artifact_exc:
        return {"artifact_error": str(artifact_exc), "artifact_dir": str(artifact_dir)}


def _teacher_artifact_root() -> Path | None:
    raw = str(os.getenv("MOBILEGPT_TEACHER_ARTIFACT_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    stats_path = str(os.getenv("MOBILEGPT_STATS_JSONL") or "").strip()
    if stats_path:
        return Path(stats_path).expanduser().resolve().parent / "teacher_artifacts"
    return None


def _mobilegpt_current_artifact_paths(log_dir: str, screen_index: str) -> dict[str, str]:
    if not log_dir or not screen_index:
        return {}
    base = Path(log_dir).expanduser()
    index = str(screen_index).strip()
    return {
        "target_screenshot": str(base / "screenshots" / f"{index}.jpg"),
        "target_raw_xml": str(base / "xmls" / f"{index}.xml"),
        "target_parsed_xml": str(base / "xmls" / f"{index}_parsed.xml"),
        "target_hierarchy_xml": str(base / "xmls" / f"{index}_hierarchy_parsed.xml"),
        "target_encoded_xml": str(base / "xmls" / f"{index}_encoded.xml"),
    }


def preflight_teacher_source_run_log(source_run_log: str | Path) -> dict[str, Any]:
    """Inspect a source run log before starting an expensive MobileGPT server."""

    path = Path(source_run_log).expanduser().resolve()
    actions = load_teacher_actions(path)
    source_xml_action_count = 0
    groundable_action_count = 0
    for record in actions:
        action = record.get("action") if isinstance(record, dict) else {}
        params = (
            dict(action.get("params") or {})
            if isinstance(action, dict) and isinstance(action.get("params"), dict)
            else {}
        )
        source_context = (
            dict(params.get("source_context") or {})
            if isinstance(params.get("source_context"), dict)
            else {}
        )
        if str(source_context.get("page") or "").strip():
            source_xml_action_count += 1
        if _teacher_action_is_groundable(action):
            groundable_action_count += 1
    return {
        "source_run_log": str(path),
        "teacher_action_count": len(actions),
        "groundable_action_count": groundable_action_count,
        "source_xml_action_count": source_xml_action_count,
        "has_source_xml": bool(source_xml_action_count),
    }


def _teacher_action_is_groundable(action: dict[str, Any]) -> bool:
    action_type = str(action.get("type") or "").strip()
    if action_type == "press_key":
        return _is_supported_press_key(action)
    if action_type == "swipe":
        params = dict(action.get("params") or {})
        return str(params.get("direction") or "").strip() in {
            "up",
            "down",
            "left",
            "right",
        }
    identity = _source_identity(action)
    return any(str(value or "").strip() for value in identity.values())


def run_teacher_server(argv: list[str] | None = None) -> int:
    """Start a MobileGPT server with teacher-forced DeriveAgent actions."""

    parser = argparse.ArgumentParser(
        description="Run MobileGPT cold-start learning with AndroidWorld teacher actions."
    )
    parser.add_argument("--mobilegpt-root", required=True)
    parser.add_argument("--source-run-log", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--buffer-size", type=int, default=4096)
    parser.add_argument("--fallback-to-vlm-on-teacher-miss", action="store_true")
    args = parser.parse_args(argv)

    preflight = preflight_teacher_source_run_log(args.source_run_log)
    _write_stats_event(
        {
            "event": "mobilegpt_teacher_source_preflight",
            **preflight,
        }
    )
    if int(preflight["teacher_action_count"]) <= 0:
        _write_stats_event(
            {
                "event": "mobilegpt_teacher_preflight_failed",
                **preflight,
                "error": "source run log has no MobileGPT-supported teacher actions",
            }
        )
        raise ValueError(
            "source run log has no MobileGPT-supported teacher actions: "
            f"{preflight['source_run_log']}"
        )
    if int(preflight["groundable_action_count"]) != int(
        preflight["teacher_action_count"]
    ):
        _write_stats_event(
            {
                "event": "mobilegpt_teacher_preflight_failed",
                **preflight,
                "error": "source run log has ungroundable teacher actions",
            }
        )
        raise ValueError(
            "source run log has ungroundable MobileGPT teacher actions: "
            f"{preflight['groundable_action_count']}/"
            f"{preflight['teacher_action_count']}"
        )

    root = Path(args.mobilegpt_root).expanduser().resolve()
    server_root = root / "Server"
    if not server_root.exists():
        raise FileNotFoundError(f"MobileGPT Server directory not found: {server_root}")
    if str(server_root) not in sys.path:
        sys.path.insert(0, str(server_root))
    os.chdir(server_root)

    import main as mobilegpt_main  # noqa: F401
    from server import Server

    from src.integrations.mobilegpt_runtime import install_mobilegpt_openai_runtime

    install_mobilegpt_openai_runtime()
    install_mobilegpt_teacher(
        source_run_log=args.source_run_log,
        fallback_to_vlm_on_miss=bool(args.fallback_to_vlm_on_teacher_miss),
    )
    _write_stats_event(
        {
            "event": "mobilegpt_teacher_server_started",
            "source_run_log": str(Path(args.source_run_log).expanduser().resolve()),
            "host": args.host,
            "port": int(args.port),
        }
    )
    Server(host=args.host, port=int(args.port), buffer_size=int(args.buffer_size)).open()
    return 0


def _best_current_screen_match(
    source_action: dict[str, Any],
    current_screen: str,
) -> dict[str, Any] | None:
    try:
        root = ET.fromstring(str(current_screen or ""))
    except Exception as exc:
        raise RuntimeError(f"Invalid current MobileGPT screen XML: {exc}") from exc

    wanted = _source_identity(source_action)
    wanted_type = str(source_action.get("type") or "").strip()
    if wanted_type == "swipe":
        return _best_scroll_target_match(root, wanted)
    if wanted_type == "press_key":
        return _best_back_target_match(root)

    candidates = []
    for element in root.iter():
        index = str(element.attrib.get("index") or "").strip()
        if not index:
            continue
        tag = str(element.tag or "").strip().lower()
        if wanted_type == "input_text" and tag != "input":
            continue
        if wanted_type in {"click", "long_press"} and tag not in {"button", "checker", "input"}:
            continue
        score, reason = _score_element(element, wanted)
        if wanted_type in {"click", "long_press"} and tag == "button" and score > 0:
            score += 0.5
            reason = ",".join(item for item in (reason, "tag:button") if item)
        if score <= 0:
            continue
        candidates.append({"index": index, "score": score, "reason": reason})
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-float(item["score"]), int(item["index"])))
    return candidates[0]


def _best_back_target_match(root: ET.Element) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    exact_labels = {"navigate up", "back", "go back", "close", "cancel"}
    exact_resource_ids = {
        "back",
        "button_back",
        "navigate_up",
        "navigation_back",
        "action_back",
        "toolbar_navigation_button",
    }
    for element in root.iter():
        index = str(element.attrib.get("index") or "").strip()
        if not index:
            continue
        tag = str(element.tag or "").strip().lower()
        if tag not in {"button", "checker"}:
            continue
        text, description, resource_id = _element_identity(element)
        score = 0.0
        reasons: list[str] = []
        if text in exact_labels:
            score += 20.0
            reasons.append("text:back")
        if description in exact_labels:
            score += 30.0
            reasons.append("description:back")
        if resource_id in exact_resource_ids:
            score += 15.0
            reasons.append("resource_id:back")
        if score <= 0:
            continue
        candidates.append({"index": index, "score": score, "reason": ",".join(reasons)})
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-float(item["score"]), int(item["index"])))
    return candidates[0]


def _back_action_match(current_screen: str) -> dict[str, Any] | None:
    try:
        root = ET.fromstring(str(current_screen or ""))
    except Exception as exc:
        raise RuntimeError(f"Invalid current MobileGPT screen XML: {exc}") from exc

    clickable_back = _best_back_target_match(root)
    if clickable_back is not None:
        return clickable_back
    return {
        "index": "",
        "score": 10.0,
        "reason": "global_back",
    }


def _is_noop_source_action(source_action: dict[str, Any], current_screen: str) -> bool:
    action_type = str(source_action.get("type") or "").strip()
    if action_type != "press_key" or not _is_supported_press_key(source_action):
        return False
    label = _norm(_source_action_label(source_action))
    if not any(
        marker in label
        for marker in (
            "hide keyboard",
            "dismiss keyboard",
            "close keyboard",
            "隐藏键盘",
        )
    ):
        return False
    try:
        root = ET.fromstring(str(current_screen or ""))
    except Exception:
        return True
    return _best_back_target_match(root) is None


def _best_scroll_target_match(
    root: ET.Element,
    wanted: dict[str, str],
) -> dict[str, Any] | None:
    scrollable_candidates: list[dict[str, Any]] = []
    fallback_candidates: list[dict[str, Any]] = []

    def _walk(element: ET.Element, depth: int) -> None:
        index = str(element.attrib.get("index") or "").strip()
        if index:
            tag = str(element.tag or "").strip().lower()
            attrs = dict(element.attrib)
            class_name = str(attrs.get("class") or "").lower()
            scrollable = (
                tag == "scroll"
                or attrs.get("scrollable") == "true"
                or "scrollview" in class_name
                or "recyclerview" in class_name
            )
            actionable_count = sum(
                1
                for node in element.iter()
                if str(node.tag or "").strip().lower()
                in {"button", "input", "checker", "recyclerview", "scroll"}
            )
            if scrollable:
                scrollable_candidates.append(
                    {
                        "index": index,
                        "score": 100.0 + actionable_count + depth,
                        "reason": "scrollable_container",
                    }
                )
            elif index != "0" and tag in {"div", "recyclerview", "listview", "scrollview"}:
                text, description, resource_id = _element_identity(element)
                combined = _norm(" ".join([text, description, resource_id]))
                wanted_text = _norm(" ".join(wanted.values()))
                if actionable_count > 0:
                    score = float(actionable_count + depth * 10)
                    reasons = ["scroll_fallback_container"]
                    if "form" in wanted_text and any(
                        str(node.tag or "").strip().lower() == "input"
                        for node in element.iter()
                    ):
                        score += 20.0
                        reasons.append("form_inputs")
                    if wanted_text and _token_overlap(wanted_text, combined):
                        score += 5.0
                        reasons.append("target_token_overlap")
                    fallback_candidates.append(
                        {
                            "index": index,
                            "score": score,
                            "reason": ",".join(reasons),
                        }
                    )
        for child in list(element):
            _walk(child, depth + 1)

    _walk(root, 0)
    candidates = scrollable_candidates or fallback_candidates
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-float(item["score"]), int(item["index"])))
    return candidates[0]


def _score_element(element: ET.Element, wanted: dict[str, str]) -> tuple[float, str]:
    text, description, resource_id = _element_identity(element)
    target = _norm(wanted.get("target"))
    source_text = _norm(wanted.get("text"))
    source_description = _norm(wanted.get("description"))
    source_resource_id = _short_resource_id(wanted.get("resource_id") or "")

    score = 0.0
    reasons: list[str] = []
    if source_resource_id and resource_id and source_resource_id == resource_id:
        score += 20.0
        reasons.append("resource_id")
    for label, current in (
        ("text", text),
        ("description", description),
        ("resource_id", resource_id),
    ):
        for source_label, source_value in (
            ("source_text", source_text),
            ("source_description", source_description),
            ("target", target),
        ):
            if not current or not source_value:
                continue
            if current == source_value:
                score += 12.0 if source_label != "target" else 10.0
                reasons.append(f"{label}:{source_label}:exact")
            elif source_value in current or current in source_value:
                score += 4.0
                reasons.append(f"{label}:{source_label}:contains")
            elif _token_overlap(source_value, current):
                score += 2.0
                reasons.append(f"{label}:{source_label}:token_overlap")
    if not reasons and target:
        combined = _norm(" ".join([text, description, resource_id]))
        if target and target in combined:
            score += 3.0
            reasons.append("target:combined")
    return score, ",".join(reasons)


def _element_identity(element: ET.Element) -> tuple[str, str, str]:
    texts: list[str] = []
    descriptions: list[str] = []
    resource_ids: list[str] = []
    for node in element.iter():
        attrs = dict(node.attrib)
        node_text = _norm(node.text or attrs.get("text") or "")
        if node_text:
            texts.append(node_text)
        node_description = _norm(
            attrs.get("description") or attrs.get("content-desc") or ""
        )
        if node_description:
            descriptions.append(node_description)
        node_resource_id = _short_resource_id(
            attrs.get("id") or attrs.get("resource-id") or ""
        )
        if node_resource_id:
            resource_ids.append(node_resource_id)
    return (
        _norm(" ".join(dict.fromkeys(texts))),
        _norm(" ".join(dict.fromkeys(descriptions))),
        _norm(" ".join(dict.fromkeys(resource_ids))),
    )


def _source_identity(source_action: dict[str, Any]) -> dict[str, str]:
    params = dict(source_action.get("params") or {})
    source_context = params.get("source_context")
    element = (
        dict(source_context.get("element") or {})
        if isinstance(source_context, dict)
        and isinstance(source_context.get("element"), dict)
        else {}
    )
    target_evidence = (
        dict(params.get("target_evidence") or {})
        if isinstance(params.get("target_evidence"), dict)
        else {}
    )
    return {
        "target": str(params.get("target_description") or target_evidence.get("label") or ""),
        "text": str(element.get("text") or element.get("label") or ""),
        "description": str(
            element.get("description")
            or element.get("content_desc")
            or element.get("content-desc")
            or ""
        ),
        "resource_id": str(
            element.get("resource_id")
            or element.get("resource-id")
            or target_evidence.get("resource_id")
            or target_evidence.get("resource-id")
            or ""
        ),
    }


def _source_action_label(source_action: dict[str, Any]) -> str:
    wanted = _source_identity(source_action)
    return (
        wanted.get("target")
        or wanted.get("text")
        or wanted.get("description")
        or wanted.get("resource_id")
        or str(source_action.get("type") or "")
    )


def _source_swipe_direction(params: dict[str, Any]) -> str:
    raw = str(params.get("direction") or "").strip().lower()
    if raw in {"up", "down", "left", "right"}:
        return raw
    target = _norm(params.get("target_description"))
    if "scroll" in target:
        for direction in ("down", "up", "left", "right"):
            if direction in target:
                return direction
    try:
        x1 = float(params.get("x") if params.get("x") is not None else params.get("x1"))
        y1 = float(params.get("y") if params.get("y") is not None else params.get("y1"))
        x2 = float(params.get("end_x") if params.get("end_x") is not None else params.get("x2"))
        y2 = float(params.get("end_y") if params.get("end_y") is not None else params.get("y2"))
    except Exception:
        return "down"
    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) > abs(dy):
        return "left" if dx > 0 else "right"
    return "up" if dy > 0 else "down"


def _is_supported_press_key(action: dict[str, Any]) -> bool:
    params = dict(action.get("params") or {}) if isinstance(action.get("params"), dict) else {}
    key = _norm(params.get("key") or params.get("keycode") or params.get("name") or "")
    return key in {"back", "keycode_back", "4"}


def _short_resource_id(value: Any) -> str:
    text = str(value or "").strip()
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    return _norm(text)


def _token_overlap(left: str, right: str) -> bool:
    left_tokens = {
        token
        for token in re.split(r"[^a-z0-9]+", str(left or "").lower())
        if len(token) > 1
    }
    right_tokens = {
        token
        for token in re.split(r"[^a-z0-9]+", str(right or "").lower())
        if len(token) > 1
    }
    return bool(left_tokens and right_tokens and left_tokens.intersection(right_tokens))


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _write_stats_event(event: dict[str, Any]) -> None:
    path = os.getenv("MOBILEGPT_STATS_JSONL")
    if not path:
        return
    try:
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(event)
        payload.setdefault("ts", datetime.now(timezone.utc).isoformat())
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(run_teacher_server())
