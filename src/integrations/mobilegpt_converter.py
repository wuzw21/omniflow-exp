"""Compile one verified RunLog into task-local MobileGPT memory."""

from __future__ import annotations

from contextlib import contextmanager
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any, Callable, Iterator, Sequence
import xml.etree.ElementTree as ET

from omniflow.core.trajectory import observation_xml
from src.experiment.mobilegpt_contract import (
    MOBILEGPT_AUDIT_SCHEMA,
    MOBILEGPT_DIRECT_AUDIT_SCHEMA,
)
from src.integrations.mobilegpt_runtime import (
    install_mobilegpt_openai_runtime,
    install_mobilegpt_select_schema_repair,
    mobilegpt_compatible_xml,
)
from src.integrations.runlog import import_run_log

CONVERSION_SOURCE_SCHEMA = "omniflow.mobilegpt-runlog-conversion-source.v1"
CONVERSION_MODE_DIRECT = "runlog_direct"
CONVERSION_MODE_SEMANTIC = "mobilegpt_semantic"
CONVERSION_AUDIT_SCHEMA = MOBILEGPT_DIRECT_AUDIT_SCHEMA

__all__ = [
    "MobileGPTConversionError",
    "convert_runlog_to_mobilegpt_memory",
    "preflight_runlog_conversion",
    "validate_mobilegpt_memory",
    "write_conversion_failure_audit",
]

_SKIPPED_ACTION_TYPES = frozenset({"open_app", "status", "wait"})
_SUPPORTED_ACTION_TYPES = frozenset(
    {
        "answer",
        "click",
        "double_tap",
        "input_text",
        "long_press",
        "navigate_back",
        "swipe",
    }
)
_ANDROIDWORLD_SWIPE_TO_MOBILEGPT_SCROLL = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}


class MobileGPTConversionError(RuntimeError):
    """A stable, machine-readable RunLog conversion failure."""

    def __init__(self, code: str, **details: Any) -> None:
        self.code = str(code)
        self.details = details
        suffix = ":" + json.dumps(details, ensure_ascii=False, sort_keys=True) if details else ""
        super().__init__(self.code + suffix)


@dataclass(frozen=True)
class _RunLogTransition:
    step_index: int
    action: dict[str, Any]
    observation: dict[str, Any]
    forest: str


class _ExploreMemoryCapture:
    def __init__(self) -> None:
        self.available_subtasks: list[dict[str, Any]] | None = None
        self.trigger_uis: dict[str, Any] | None = None
        self.extra_uis: list[Any] | None = None
        self.screen = ""
        self.hierarchy_xml = ""

    def add_node(
        self,
        available_subtasks: list[dict[str, Any]],
        trigger_uis: dict[str, Any],
        extra_uis: list[Any],
        screen: str,
        screen_num: int | None = None,
    ) -> int:
        del screen_num
        self.available_subtasks = list(available_subtasks)
        self.trigger_uis = dict(trigger_uis)
        self.extra_uis = list(extra_uis)
        self.screen = str(screen)
        return 0

    def add_hierarchy_xml(self, hierarchy_xml: str, page_index: int) -> None:
        if page_index != 0:
            raise ValueError("mobilegpt_explore_capture_page_invalid")
        self.hierarchy_xml = str(hierarchy_xml)


class _SelectMemoryCapture:
    def __init__(self) -> None:
        self.examples: dict[str, dict[str, Any]] = {}

    def save_subtask(
        self,
        subtask: dict[str, Any],
        example: dict[str, Any],
    ) -> None:
        name = str(subtask.get("name") or "").strip()
        if name:
            self.examples[name] = dict(example)


def _read_csv_rows(path: Path, *, label: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"mobilegpt_memory_{label}_missing:{path}")
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (csv.Error, OSError) as error:
        raise ValueError(f"mobilegpt_memory_{label}_invalid:{path}") from error


def _json_object(value: Any, *, error_code: str) -> dict[str, Any]:
    candidate = value
    if isinstance(candidate, str):
        try:
            candidate = json.loads(candidate)
        except json.JSONDecodeError as error:
            raise ValueError(error_code) from error
    if not isinstance(candidate, dict):
        raise ValueError(error_code)
    return dict(candidate)


def _validated_action(value: Any) -> dict[str, Any]:
    action = _json_object(
        value,
        error_code="mobilegpt_memory_action_json_invalid",
    )
    name = str(action.get("name") or "").strip()
    if not name:
        raise ValueError("mobilegpt_memory_action_name_invalid")
    parameters = action.get("parameters")
    if isinstance(parameters, str):
        try:
            parameters = json.loads(parameters)
        except json.JSONDecodeError as error:
            raise ValueError("mobilegpt_memory_action_parameters_invalid") from error
    if not isinstance(parameters, dict):
        raise ValueError("mobilegpt_memory_action_parameters_invalid")
    return {"name": name, "parameters": dict(parameters)}


def validate_mobilegpt_memory(
    memory_root: str | Path,
    *,
    require_screenshot: bool = False,
) -> dict[str, Any]:
    """Validate a task-local native MobileGPT memory graph."""

    root = Path(memory_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"mobilegpt_memory_missing:{root}")
    root_tasks = _read_csv_rows(root / "tasks.csv", label="root_tasks")
    if len(root_tasks) != 1:
        raise ValueError("mobilegpt_memory_root_task_count_invalid")
    root_task = root_tasks[0]
    root_task_name = str(root_task.get("name") or "").strip()
    app_name = str(root_task.get("app") or "").strip()
    if not root_task_name or not app_name:
        raise ValueError("mobilegpt_memory_root_task_invalid")
    _json_object(
        root_task.get("parameters") or "{}",
        error_code="mobilegpt_memory_root_task_parameters_invalid",
    )

    app_root = root / app_name
    app_tasks = _read_csv_rows(app_root / "tasks.csv", label="app_tasks")
    if len(app_tasks) != 1:
        raise ValueError("mobilegpt_memory_app_task_count_invalid")
    app_task = app_tasks[0]
    if str(app_task.get("name") or "").strip() != root_task_name:
        raise ValueError("mobilegpt_memory_task_not_local")
    task_path = _json_object(
        app_task.get("path"),
        error_code="mobilegpt_memory_task_path_invalid",
    )
    if not task_path:
        raise ValueError("mobilegpt_memory_task_path_empty")

    pages = _read_csv_rows(app_root / "pages.csv", label="pages")
    hierarchy = _read_csv_rows(app_root / "hierarchy.csv", label="hierarchy")
    page_indexes = {
        str(row.get("index") or "").strip()
        for row in pages
        if str(row.get("index") or "").strip()
    }
    hierarchy_indexes = {
        str(row.get("index") or "").strip()
        for row in hierarchy
        if str(row.get("index") or "").strip()
    }
    if not page_indexes or page_indexes != hierarchy_indexes:
        raise ValueError("mobilegpt_memory_page_graph_invalid")

    task_subtask_count = 0
    action_count = 0
    non_finish_action_count = 0
    screen_file_count = 0
    for raw_page_index, raw_subtasks in task_path.items():
        page_index = str(raw_page_index).strip()
        if page_index not in page_indexes or not isinstance(raw_subtasks, list):
            raise ValueError("mobilegpt_memory_task_path_invalid")
        page_root = app_root / "pages" / page_index
        subtasks = _read_csv_rows(page_root / "subtasks.csv", label="subtasks")
        available = _read_csv_rows(
            page_root / "available_subtasks.csv",
            label="available_subtasks",
        )
        subtask_names = {
            str(row.get("name") or "").strip()
            for row in subtasks
            if str(row.get("name") or "").strip()
        }
        available_names = {
            str(row.get("name") or "").strip()
            for row in available
            if str(row.get("name") or "").strip()
        }
        if not subtask_names or not subtask_names.issubset(available_names):
            raise ValueError("mobilegpt_memory_subtask_graph_invalid")
        for raw_subtask in raw_subtasks:
            subtask_name = str(raw_subtask or "").strip()
            if subtask_name in {"finish", "scroll_screen"}:
                continue
            task_subtask_count += 1
            if subtask_name not in subtask_names:
                raise ValueError("mobilegpt_memory_task_subtask_missing")

        action_rows = _read_csv_rows(page_root / "actions.csv", label="actions")
        next_step_by_subtask: dict[str, int] = {}
        for row in action_rows:
            subtask_name = str(row.get("subtask_name") or "").strip()
            if not subtask_name or subtask_name not in subtask_names:
                raise ValueError("mobilegpt_memory_action_subtask_invalid")
            try:
                step = int(str(row.get("step") or "").strip())
            except ValueError as error:
                raise ValueError("mobilegpt_memory_action_step_invalid") from error
            if step < 0:
                raise ValueError("mobilegpt_memory_action_step_invalid")
            action = _validated_action(row.get("action"))
            expected_step = next_step_by_subtask.get(subtask_name, 0)
            if step != expected_step:
                raise ValueError("mobilegpt_memory_action_steps_not_contiguous")
            action_count += 1
            if action["name"] != "finish":
                non_finish_action_count += 1
                next_step_by_subtask[subtask_name] = step + 1
            else:
                next_step_by_subtask[subtask_name] = 0
        screen_root = page_root / "screen"
        required = {
            "raw.xml",
            "html.xml",
            "hierarchy.xml",
            "parsed.xml",
            "pretty.xml",
        }
        if require_screenshot:
            required.add("screenshot.jpg")
        present = {
            path.name
            for path in screen_root.iterdir()
            if path.is_file() and path.stat().st_size > 0
        } if screen_root.is_dir() else set()
        if not required.issubset(present):
            raise ValueError("mobilegpt_memory_screen_artifacts_incomplete")
        for name in sorted(required):
            if not name.endswith(".xml"):
                continue
            try:
                ET.fromstring((screen_root / name).read_text(encoding="utf-8"))
            except (OSError, UnicodeError, ET.ParseError) as error:
                raise ValueError(
                    f"mobilegpt_memory_screen_xml_invalid:{page_index}:{name}"
                ) from error
        screen_file_count += len(present)

    if task_subtask_count <= 0:
        raise ValueError("mobilegpt_memory_recallable_subtask_missing")
    if non_finish_action_count <= 0:
        raise ValueError("mobilegpt_memory_useful_action_missing")
    return {
        "memory_root": str(root),
        "task_name": root_task_name,
        "app": app_name,
        "task_local_memory": True,
        "page_count": len(page_indexes),
        "subtask_count": task_subtask_count,
        "action_count": action_count,
        "non_finish_action_count": non_finish_action_count,
        "screen_file_count": screen_file_count,
        "native_memory_complete": True,
    }


def _observation_for_step(step: dict[str, Any]) -> dict[str, Any]:
    for key in ("observation", "observation_before_act", "state", "before"):
        value = step.get(key)
        if isinstance(value, dict):
            return dict(value)
    return {}


def _action_type(action: dict[str, Any]) -> str:
    return str(action.get("action_type") or action.get("type") or "").strip()


def _package_from_observation(observation: dict[str, Any]) -> str:
    auxiliaries = observation.get("auxiliaries")
    candidates = [
        observation.get("package_name"),
        observation.get("packageName"),
        auxiliaries.get("package_name") if isinstance(auxiliaries, dict) else None,
        auxiliaries.get("packageName") if isinstance(auxiliaries, dict) else None,
    ]
    return next((str(value).strip() for value in candidates if str(value or "").strip()), "")


def _load_runlog_trajectory(
    source_run_log: str | Path,
    *,
    target_package: str = "",
    target_app: str = "",
) -> dict[str, Any]:
    """Load one successful canonical RunLog for deterministic offline compilation."""

    path = Path(source_run_log).expanduser().resolve()
    payload = import_run_log(json.loads(path.read_text(encoding="utf-8")))
    if payload.get("status") != "succeeded" or payload.get("success") is not True:
        raise MobileGPTConversionError("source_runlog_not_successful", path=str(path))

    transitions: list[_RunLogTransition] = []
    skipped: list[dict[str, Any]] = []
    packages: list[str] = []
    for ordinal, raw_step in enumerate(payload.get("steps") or []):
        if not isinstance(raw_step, dict):
            continue
        step_index = int(raw_step.get("step_index", ordinal))
        action = raw_step.get("action")
        if not isinstance(action, dict):
            raise MobileGPTConversionError("source_action_missing", step_index=step_index)
        action = dict(action)
        action_type = _action_type(action)
        observation = _observation_for_step(raw_step)
        package = _package_from_observation(observation)
        if package:
            packages.append(package)
        if action_type == "open_app":
            app_name = str(action.get("app_name") or action.get("package_name") or "").strip()
            if app_name:
                packages.append(app_name)
        if action_type in _SKIPPED_ACTION_TYPES:
            skipped.append({"step_index": step_index, "action_type": action_type})
            continue
        if action_type not in _SUPPORTED_ACTION_TYPES:
            raise MobileGPTConversionError(
                "source_action_unsupported",
                step_index=step_index,
                action_type=action_type or "missing",
            )
        forest = observation_xml(observation).strip()
        if not forest:
            raise MobileGPTConversionError(
                "source_observation_missing",
                step_index=step_index,
                action_type=action_type,
            )
        try:
            ET.fromstring(forest)
        except ET.ParseError as error:
            raise MobileGPTConversionError(
                "source_observation_invalid_xml",
                step_index=step_index,
                error=str(error),
            ) from error
        transitions.append(
            _RunLogTransition(
                step_index=step_index,
                action=action,
                observation=observation,
                forest=forest,
            )
        )

    if not transitions:
        raise MobileGPTConversionError("source_trajectory_empty")
    package_names = sorted(
        {
            value
            for value in packages
            if value
            not in {
                "android",
                "com.android.systemui",
                "com.google.android.apps.nexuslauncher",
            }
        }
    )
    resolved_target_package = str(target_package or "").strip()
    resolved_target_app = str(target_app or "").strip()
    if not resolved_target_package:
        resolved_target_package = package_names[0] if len(package_names) == 1 else ""
    open_app_packages = [
        str(step.get("action", {}).get("app_name") or "").strip()
        for step in payload.get("steps") or []
        if isinstance(step, dict)
        and isinstance(step.get("action"), dict)
        and _action_type(step["action"]) == "open_app"
        and str(step["action"].get("app_name") or "").strip()
    ]
    if open_app_packages:
        resolved_target_package = open_app_packages[0]
    if not resolved_target_package:
        raise MobileGPTConversionError(
            "source_target_package_unresolved",
            packages=package_names,
        )
    if not resolved_target_app:
        resolved_target_app = resolved_target_package
    return {
        "schema_version": CONVERSION_SOURCE_SCHEMA,
        "source_run_log": str(path),
        "run_id": str(payload.get("run_id") or ""),
        "task_name": str(payload.get("task_name") or ""),
        "instruction": str(payload.get("goal") or ""),
        "task_parameters": dict(payload.get("task_parameters") or {}),
        "source_seed": payload.get("seed"),
        "target_package": resolved_target_package,
        "target_app": resolved_target_app,
        "transitions": transitions,
        "skipped_actions": skipped,
        "source_success_boundary": {
            "status": payload.get("status"),
            "success": payload.get("success"),
            "validator": dict(payload.get("validator") or {}),
        },
    }


def preflight_runlog_conversion(
    source_run_log: str | Path,
    *,
    target_package: str = "",
    target_app: str = "",
) -> dict[str, Any]:
    """Return a deterministic source-only readiness report."""

    path = Path(source_run_log).expanduser().resolve()
    try:
        trajectory = _load_runlog_trajectory(
            path,
            target_package=target_package,
            target_app=target_app,
        )
    except MobileGPTConversionError as error:
        return {
            "schema_version": CONVERSION_SOURCE_SCHEMA,
            "source_run_log": str(path),
            "ready": False,
            "failure_code": error.code,
            "failure_details": error.details,
        }
    transitions: list[_RunLogTransition] = trajectory["transitions"]
    counts: dict[str, int] = {}
    for transition in transitions:
        name = _action_type(transition.action)
        counts[name] = counts.get(name, 0) + 1
    return {
        "schema_version": CONVERSION_SOURCE_SCHEMA,
        "source_run_log": str(path),
        "task_name": trajectory["task_name"],
        "target_package": trajectory["target_package"],
        "transition_count": len(transitions),
        "action_type_counts": counts,
        "skipped_actions": trajectory["skipped_actions"],
        "ready": True,
    }


def _bounds(value: Any) -> tuple[float, float, float, float] | None:
    match = re.fullmatch(
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
        str(value or "").strip(),
    )
    if not match:
        return None
    left, top, right, bottom = (float(value) for value in match.groups())
    return left, top, right, bottom


def _point(action: dict[str, Any]) -> tuple[float, float] | None:
    try:
        return float(action["x"]), float(action["y"])
    except (KeyError, TypeError, ValueError):
        return None


def _element_contains_action(element: ET.Element, action: dict[str, Any]) -> bool:
    point = _point(action)
    element_bounds = _bounds(element.get("bounds"))
    if point is None or element_bounds is None:
        return False
    x, y = point
    left, top, right, bottom = element_bounds
    return left <= x <= right and top <= y <= bottom


def _swipe_direction(action: dict[str, Any]) -> str:
    explicit = str(action.get("direction") or "").strip().lower()
    if explicit:
        return explicit
    try:
        start_x = float(action.get("start_x", action.get("x1")))
        start_y = float(action.get("start_y", action.get("y1")))
        end_x = float(action.get("end_x", action.get("x2")))
        end_y = float(action.get("end_y", action.get("y2")))
    except (TypeError, ValueError):
        return ""
    dx = end_x - start_x
    dy = end_y - start_y
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _screen_identity(hierarchy_xml: str) -> str:
    normalized = re.sub(r">\s+<", "><", str(hierarchy_xml or "").strip())
    return hashlib.sha256(normalized.encode()).hexdigest()


def _target_element(
    action: dict[str, Any],
    parsed_xml: str,
    *,
    step_index: int,
    source_forest: str = "",
) -> ET.Element:
    try:
        root = ET.fromstring(parsed_xml)
    except ET.ParseError as error:
        raise MobileGPTConversionError(
            "mobilegpt_parsed_screen_invalid",
            step_index=step_index,
            error=str(error),
        ) from error
    indexed = [element for element in root.iter() if element.get("index") is not None]
    point = _point(action)
    action_type = _action_type(action)
    if point is None and action_type == "input_text" and source_forest:
        try:
            source_root = ET.fromstring(source_forest)
        except ET.ParseError as error:
            raise MobileGPTConversionError(
                "source_observation_invalid_xml",
                step_index=step_index,
                error=str(error),
            ) from error
        focused_inputs = [
            element
            for element in source_root.iter()
            if str(element.get("focused") or "").strip().lower() == "true"
            and (
                str(element.get("editable") or "").strip().lower() == "true"
                or "edittext" in str(element.get("class") or "").casefold()
            )
            and _bounds(element.get("bounds")) is not None
        ]
        if len(focused_inputs) == 1:
            left, top, right, bottom = _bounds(focused_inputs[0].get("bounds")) or (
                0,
                0,
                0,
                0,
            )
            point = ((left + right) / 2, (top + bottom) / 2)
    if point is not None:
        point_action = dict(action)
        point_action["x"], point_action["y"] = point
        candidates = [
            element
            for element in indexed
            if _element_contains_action(element, point_action)
        ]
        if action_type == "input_text":
            inputs = [element for element in candidates if element.tag == "input"]
            if inputs:
                candidates = inputs
        elif action_type in {"click", "double_tap", "long_press"}:
            actionable = [
                element
                for element in candidates
                if element.tag in {"button", "checker", "input"}
            ]
            if actionable:
                candidates = actionable
        candidates.sort(
            key=lambda element: (
                (_bounds(element.get("bounds")) or (0, 0, float("inf"), float("inf")))[2]
                - (_bounds(element.get("bounds")) or (0, 0, float("inf"), float("inf")))[0]
            )
            * (
                (_bounds(element.get("bounds")) or (0, 0, float("inf"), float("inf")))[3]
                - (_bounds(element.get("bounds")) or (0, 0, float("inf"), float("inf")))[1]
            )
        )
    else:
        candidates = []
    if not candidates:
        raise MobileGPTConversionError(
            "source_action_target_unresolved",
            step_index=step_index,
            action_type=_action_type(action),
        )
    return candidates[0]


def _parameter_values(task_parameters: dict[str, Any]) -> dict[str, str]:
    return {
        str(name): str(value)
        for name, value in task_parameters.items()
        if str(name).strip() and str(value).strip()
    }


def _action_parameter_bindings(
    action: dict[str, Any],
    element: ET.Element | None,
    task_parameters: dict[str, Any],
) -> dict[str, str]:
    searchable = " ".join(
        str(value or "")
        for value in (
            action.get("text"),
            element.text if element is not None else "",
            element.get("text") if element is not None else "",
            element.get("description") if element is not None else "",
        )
    ).casefold()
    return {
        name: value
        for name, value in _parameter_values(task_parameters).items()
        if value.casefold() in searchable
    }


def _parameter_schema(parameters: dict[str, str]) -> dict[str, str]:
    return {
        name: f"Value of the AndroidWorld task parameter '{name}'"
        for name in parameters
    }


def _element_semantic_label(element: ET.Element | None) -> str:
    if element is None:
        return ""
    for candidate in element.iter():
        for value in (
            candidate.text,
            candidate.get("text"),
            candidate.get("description"),
        ):
            label = str(value or "").strip()
            if label:
                return label
    return ""


def _direct_subtask_from_runlog(
    transition: _RunLogTransition,
    parsed_xml: str,
    encoded_xml: str,
    instruction: str,
    task_parameters: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Describe one verified source transition in MobileGPT's subtask schema."""

    action_type = _action_type(transition.action)
    name = f"source_step_{transition.step_index:03d}_{action_type}"
    target: ET.Element | None = None
    if action_type in {"click", "double_tap", "input_text", "long_press"}:
        target = _target_element(
            transition.action,
            parsed_xml,
            step_index=transition.step_index,
            source_forest=transition.forest,
        )
    parameter_values = _action_parameter_bindings(
        transition.action,
        target,
        task_parameters,
    )
    parameter_descriptions = _parameter_schema(parameter_values)
    target_label = _element_semantic_label(target)
    if target_label and target_label not in parameter_values.values():
        parameter_values["target_text"] = target_label
        parameter_descriptions["target_text"] = (
            "Visible text or content description of the UI target"
        )
    input_text = str(transition.action.get("text") or "").strip()
    if (
        action_type == "input_text"
        and input_text
        and input_text not in parameter_values.values()
    ):
        parameter_values["input_text"] = input_text
        parameter_descriptions["input_text"] = "Text to enter into the UI target"
    metadata = {
        "name": name,
        "description": (
            f"Execute verified source transition {transition.step_index}: "
            f"{action_type}"
        ),
        "parameters": parameter_descriptions,
    }
    selected = {
        "name": name,
        "description": metadata["description"],
        "parameters": parameter_values,
    }
    example = {
        "instruction": instruction,
        "response": {
            "action": selected,
            "reasoning": "Compiled from a verified successful source transition.",
            "speak": "",
        },
        "screen": encoded_xml,
    }
    return metadata, selected, example


def _mobilegpt_action_from_runlog(
    transition: _RunLogTransition,
    parsed_xml: str,
    *,
    task_parameters: dict[str, Any],
    selected_subtask: dict[str, Any],
    generalize_action: Callable[[dict[str, Any], dict[str, Any], str], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, str], str]:
    action = transition.action
    action_type = _action_type(action)
    target: ET.Element | None = None
    if action_type in {"click", "double_tap", "input_text", "long_press"}:
        target = _target_element(
            action,
            parsed_xml,
            step_index=transition.step_index,
            source_forest=transition.forest,
        )
        index = str(target.get("index"))
    bindings = _action_parameter_bindings(action, target, task_parameters)
    if action_type == "click":
        converted = {"name": "click", "parameters": {"index": index}}
    elif action_type == "double_tap":
        converted = {
            "name": "repeat-click",
            "parameters": {"index": index, "number": 2},
        }
    elif action_type == "long_press":
        converted = {"name": "long-click", "parameters": {"index": index}}
    elif action_type == "input_text":
        converted = {
            "name": "input",
            "parameters": {
                "index": index,
                "input_text": str(action.get("text") or ""),
            },
        }
    elif action_type == "swipe":
        direction = _ANDROIDWORLD_SWIPE_TO_MOBILEGPT_SCROLL.get(
            _swipe_direction(action),
            "",
        )
        if not direction:
            raise MobileGPTConversionError(
                "source_swipe_direction_unresolved",
                step_index=transition.step_index,
            )
        converted = {"name": "scroll", "parameters": {"direction": direction}}
    elif action_type == "navigate_back":
        converted = {"name": "back", "parameters": {}}
    elif action_type == "answer":
        converted = {
            "name": "speak",
            "parameters": {"message": str(action.get("text") or "")},
        }
    else:
        raise MobileGPTConversionError(
            "source_action_unsupported",
            step_index=transition.step_index,
            action_type=action_type,
        )
    if "index" in converted["parameters"]:
        generalization_screen = f"<hierarchy>{parsed_xml}</hierarchy>"
        selected_parameters = selected_subtask.get("parameters")
        if not isinstance(selected_parameters, dict):
            selected_parameters = {}
        semantic_parameters = {
            str(name): str(value)
            for name, value in selected_parameters.items()
            if str(name).strip()
            and str(value).strip()
            and str(value).strip().casefold() != "unknown"
        }
        semantic_parameters.update(bindings)
        converted = generalize_action(
            converted,
            {
                "name": str(selected_subtask.get("name") or "sourceStep"),
                "parameters": semantic_parameters,
            },
            generalization_screen,
        )
    label = _element_semantic_label(target)
    return converted, bindings, label


def _mobilegpt_action_from_derive(
    derive_agent_class: type,
    transition: _RunLogTransition,
    parsed_xml: str,
    encoded_xml: str,
    *,
    instruction: str,
    selected_subtask: dict[str, Any],
    subtask_history: list[str],
    task_parameters: dict[str, Any],
    generalize_action: Callable[[dict[str, Any], dict[str, Any], str], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, str], str]:
    action_type = _action_type(transition.action)
    if action_type != "input_text":
        raise MobileGPTConversionError(
            "mobilegpt_derive_fallback_unsupported",
            step_index=transition.step_index,
            action_type=action_type,
        )
    derive_agent = derive_agent_class(None, instruction)
    derive_agent.init_subtask(selected_subtask, subtask_history)
    derived_action, _ = derive_agent.derive(encoded_xml)
    converted = _validated_action(derived_action)
    if converted["name"] != "input":
        raise MobileGPTConversionError(
            "mobilegpt_derive_action_mismatch",
            step_index=transition.step_index,
            source_action_type=action_type,
            derived_action_name=converted["name"],
        )
    parameters = converted["parameters"]
    source_text = str(transition.action.get("text") or "")
    if str(parameters.get("input_text") or "") != source_text:
        raise MobileGPTConversionError(
            "mobilegpt_derive_input_text_mismatch",
            step_index=transition.step_index,
        )
    raw_index_value = parameters.get("index")
    raw_index = "" if raw_index_value is None else str(raw_index_value).strip()
    try:
        root = ET.fromstring(parsed_xml)
    except ET.ParseError as error:
        raise MobileGPTConversionError(
            "mobilegpt_parsed_screen_invalid",
            step_index=transition.step_index,
            error=str(error),
        ) from error
    target = next(
        (
            element
            for element in root.iter()
            if str(element.get("index") or "").strip() == raw_index
        ),
        None,
    )
    if target is None or target.tag != "input":
        raise MobileGPTConversionError(
            "mobilegpt_derive_target_invalid",
            step_index=transition.step_index,
            index=raw_index,
        )
    bindings = _action_parameter_bindings(
        transition.action,
        target,
        task_parameters,
    )
    selected_parameters = selected_subtask.get("parameters")
    if not isinstance(selected_parameters, dict):
        selected_parameters = {}
    semantic_parameters = {
        str(name): str(value)
        for name, value in selected_parameters.items()
        if str(name).strip()
        and str(value).strip()
        and str(value).strip().casefold() != "unknown"
    }
    semantic_parameters.update(bindings)
    generalized = generalize_action(
        converted,
        {
            "name": str(selected_subtask.get("name") or "sourceStep"),
            "parameters": semantic_parameters,
        },
        f"<hierarchy>{parsed_xml}</hierarchy>",
    )
    if not isinstance(generalized, dict):
        raise MobileGPTConversionError(
            "mobilegpt_derive_generalization_failed",
            step_index=transition.step_index,
        )
    label = str(
        target.text
        or target.get("text")
        or target.get("description")
        or target.get("id")
        or ""
    ).strip()
    return generalized, bindings, label


def _validated_subtask(value: Any, *, error_code: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MobileGPTConversionError(error_code)
    name = str(value.get("name") or "").strip()
    description = str(value.get("description") or "").strip()
    parameters = value.get("parameters")
    if not name or not description or not isinstance(parameters, dict):
        raise MobileGPTConversionError(error_code, name=name)
    return {
        "name": name,
        "description": description,
        "parameters": dict(parameters),
    }


def _explore_page_with_mobilegpt(
    explore_agent_class: type,
    *,
    parsed_xml: str,
    hierarchy_xml: str,
    encoded_xml: str,
    screen_index: int,
    source_step_index: int,
) -> dict[str, Any]:
    capture = _ExploreMemoryCapture()
    explore_agent_class(capture).explore(
        parsed_xml,
        hierarchy_xml,
        encoded_xml,
        screen_index,
    )
    if capture.available_subtasks is None:
        raise MobileGPTConversionError(
            "mobilegpt_explore_output_missing",
            step_index=source_step_index,
        )
    available_subtasks = [
        _validated_subtask(
            subtask,
            error_code="mobilegpt_explore_subtask_invalid",
        )
        for subtask in capture.available_subtasks
    ]
    names = [subtask["name"] for subtask in available_subtasks]
    if not names:
        raise MobileGPTConversionError(
            "mobilegpt_explore_subtasks_empty",
            step_index=source_step_index,
        )
    if len(names) != len(set(names)):
        raise MobileGPTConversionError(
            "mobilegpt_explore_subtask_names_duplicated",
            step_index=source_step_index,
        )
    if capture.screen != parsed_xml or capture.hierarchy_xml != hierarchy_xml:
        raise MobileGPTConversionError(
            "mobilegpt_explore_capture_incomplete",
            step_index=source_step_index,
        )
    return {
        "available_subtasks": available_subtasks,
        "trigger_uis": capture.trigger_uis or {},
        "extra_uis": capture.extra_uis or [],
    }


def _select_subtask_with_mobilegpt(
    select_agent_class: type,
    default_subtasks: Sequence[dict[str, Any]],
    *,
    instruction: str,
    available_subtasks: list[dict[str, Any]],
    subtask_history: list[str],
    encoded_xml: str,
    source_action_type: str,
    source_step_index: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    capture = _SelectMemoryCapture()
    response, new_action = select_agent_class(capture, instruction).select(
        available_subtasks,
        subtask_history,
        [],
        encoded_xml,
    )
    if not isinstance(response, dict) or not isinstance(response.get("action"), dict):
        raise MobileGPTConversionError(
            "mobilegpt_select_response_invalid",
            step_index=source_step_index,
        )
    selected = dict(response["action"])
    name = str(selected.get("name") or "").strip()
    parameters = selected.get("parameters")
    if not name or not isinstance(parameters, dict):
        raise MobileGPTConversionError(
            "mobilegpt_select_action_invalid",
            step_index=source_step_index,
        )
    if name == "finish":
        raise MobileGPTConversionError(
            "mobilegpt_select_finished_before_source_action",
            step_index=source_step_index,
            source_action_type=source_action_type,
        )
    primitive_source_types = {"scroll_screen": "swipe", "speak": "answer"}
    expected_source_type = primitive_source_types.get(name)
    if expected_source_type and source_action_type != expected_source_type:
        raise MobileGPTConversionError(
            "mobilegpt_select_primitive_mismatch",
            step_index=source_step_index,
            selected_subtask=name,
            source_action_type=source_action_type,
        )
    candidates = list(available_subtasks) + [dict(item) for item in default_subtasks]
    raw_subtask = next(
        (item for item in candidates if str(item.get("name") or "").strip() == name),
        None,
    )
    if raw_subtask is None and isinstance(new_action, dict):
        raw_subtask = new_action
    subtask = _validated_subtask(
        raw_subtask,
        error_code="mobilegpt_select_subtask_missing",
    )
    if not any(item["name"] == subtask["name"] for item in available_subtasks):
        available_subtasks.append(subtask)
    selected["name"] = name
    selected["parameters"] = dict(parameters)
    example = capture.examples.get(name, {})
    return subtask, selected, example


@contextmanager
def _temporary_environment(values: dict[str, str | None]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _temporary_agent_query_provider(
    agent_modules: Sequence[Any],
    provider: Callable[..., Any] | None,
) -> Iterator[None]:
    previous = [module.query for module in agent_modules]
    try:
        for module in agent_modules:
            active_provider = provider or module.query
            agent_name = str(module.__name__).rsplit(".", 1)[-1].removesuffix("_agent")

            def tagged_query(
                messages: list[dict[str, Any]],
                model: str | None = None,
                is_list: bool = False,
                agent_name: str = agent_name,
                active_provider: Callable[..., Any] = active_provider,
                tag_provider: bool = provider is not None,
            ) -> Any:
                if not tag_provider:
                    return active_provider(
                        messages,
                        model=model,
                        is_list=is_list,
                    )
                return active_provider(
                    messages,
                    model=model,
                    is_list=is_list,
                    agent_name=agent_name,
                )

            module.query = tagged_query
        yield
    finally:
        for module, query in zip(agent_modules, previous, strict=True):
            module.query = query


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _write_event(path: Path, event: dict[str, Any]) -> None:
    payload = dict(event)
    payload.setdefault("ts", time.time())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_conversion_failure_audit(
    *,
    source_run_log: str | Path,
    stats_path: str | Path,
    audit_path: str | Path,
    error: BaseException,
    wall_sec: float,
    target_package: str = "",
    target_app: str = "",
    conversion_mode: str = CONVERSION_MODE_DIRECT,
) -> dict[str, Any]:
    """Persist partial evidence after an interrupted offline conversion."""

    trajectory = _load_runlog_trajectory(
        source_run_log,
        target_package=target_package,
        target_app=target_app,
    )
    validation_rows: list[dict[str, Any]] = []
    stats = Path(stats_path).expanduser().resolve()
    if stats.is_file():
        for raw_line in stats.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(event, dict)
                and event.get("event") == "mobilegpt_conversion_action_mapped"
            ):
                validation_rows.append(
                    {
                        key: value
                        for key, value in event.items()
                        if key not in {"event", "ts"}
                    }
                )
    failure_code = (
        error.code
        if isinstance(error, MobileGPTConversionError)
        else type(error).__name__
    )
    failure_details = (
        dict(error.details)
        if isinstance(error, MobileGPTConversionError)
        else {"error": str(error)}
    )
    audit_schema = (
        MOBILEGPT_AUDIT_SCHEMA
        if conversion_mode == CONVERSION_MODE_SEMANTIC
        else MOBILEGPT_DIRECT_AUDIT_SCHEMA
    )
    payload = {
        "schema_version": audit_schema,
        "conversion_mode": conversion_mode,
        "task_name": trajectory["task_name"],
        "source_run_log": trajectory["source_run_log"],
        "target_package": trajectory["target_package"],
        "transition_count": len(trajectory["transitions"]),
        "validated_transition_count": sum(
            int(row.get("consumed_transitions") or 0)
            for row in validation_rows
            if row.get("matched") is True
        ),
        "validation_rows": validation_rows,
        "actions_supplied_to_mobilegpt": True,
        "source_transitions_supplied": True,
        "source_success_boundary_supplied": True,
        "source_success_boundary": trajectory["source_success_boundary"],
        "complete": False,
        "failure_code": str(failure_code),
        "failure_details": failure_details,
        "wall_sec": round(float(wall_sec), 6),
    }
    destination = Path(audit_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return payload


def convert_runlog_to_mobilegpt_memory(
    *,
    source_run_log: str | Path,
    mobilegpt_root: str | Path,
    memory_root: str | Path,
    stats_path: str | Path,
    audit_path: str | Path,
    model: str,
    target_package: str = "",
    target_app: str = "",
    embedding_provider: Callable[[str], Sequence[float]] | None = None,
    semantic_query_provider: Callable[..., Any] | None = None,
    conversion_mode: str = CONVERSION_MODE_DIRECT,
) -> dict[str, Any]:
    """Write one RunLog as an exact native-format MobileGPT database."""

    if conversion_mode not in {
        CONVERSION_MODE_DIRECT,
        CONVERSION_MODE_SEMANTIC,
    }:
        raise ValueError(f"mobilegpt_conversion_mode_invalid:{conversion_mode}")
    semantic_conversion = conversion_mode == CONVERSION_MODE_SEMANTIC

    trajectory = _load_runlog_trajectory(
        source_run_log,
        target_package=target_package,
        target_app=target_app,
    )
    transitions: list[_RunLogTransition] = trajectory["transitions"]
    server_root = Path(mobilegpt_root).expanduser().resolve() / "Server"
    if not server_root.is_dir():
        raise FileNotFoundError(f"mobilegpt_server_root_missing:{server_root}")
    memory = Path(memory_root).expanduser().resolve()
    stats = Path(stats_path).expanduser().resolve()
    audit = Path(audit_path).expanduser().resolve()
    memory.mkdir(parents=True, exist_ok=False)
    stats.parent.mkdir(parents=True, exist_ok=True)
    audit.parent.mkdir(parents=True, exist_ok=True)
    log_root = memory.parent / "conversion_log"

    if str(server_root) not in sys.path:
        sys.path.insert(0, str(server_root))
    environment = {
        "MOBILEGPT_MEMORY_ROOT": str(memory),
        "MOBILEGPT_STATS_JSONL": str(stats),
        "MOBILEGPT_CHAT_MODEL": str(model),
        "MOBILEGPT_CHAT_MAX_ATTEMPTS": "1",
        "MOBILEGPT_TARGET_APP": str(trajectory["target_app"]),
        "MOBILEGPT_TARGET_PACKAGE": str(trajectory["target_package"]),
        "MOBILEGPT_TEACHER_RUNLOG": None,
        "MOBILEGPT_TEACHER_FALLBACK_TO_VLM_ON_MISS": None,
        "MOBILEGPT_CURRENT_LOG_DIRECTORY": str(log_root),
    }

    started = time.monotonic()
    audit_rows: list[dict[str, Any]] = []
    with _temporary_environment(environment), _working_directory(server_root):
        from agents import derive_agent as derive_agent_module
        from agents import explore_agent as explore_agent_module
        from agents import select_agent as select_agent_module
        from agents.derive_agent import DeriveAgent
        from agents.explore_agent import ExploreAgent
        from agents.prompts.select_agent_prompt import default_subtasks
        from agents.select_agent import SelectAgent
        from memory.memory_manager import Memory
        from screenParser.Encoder import xmlEncoder
        from utils.action_utils import generalize_action
        from utils.utils import get_openai_embedding

        if semantic_conversion and semantic_query_provider is None:
            install_mobilegpt_openai_runtime(preserve_original_prompts=True)
            install_mobilegpt_select_schema_repair(SelectAgent)

        task_name = str(trajectory["task_name"] or "").strip()
        app_name = str(trajectory["target_app"] or trajectory["target_package"]).strip()
        if not task_name:
            raise MobileGPTConversionError("source_task_name_missing")
        if not app_name or app_name in {".", ".."} or Path(app_name).name != app_name:
            raise MobileGPTConversionError("mobilegpt_app_path_invalid", app=app_name)
        task = {
            "name": task_name,
            "description": str(trajectory["instruction"] or task_name),
            "parameters": _parameter_schema(
                _parameter_values(trajectory["task_parameters"])
            ),
            "app": app_name,
        }
        _write_event(
            stats,
            {
                "event": "task_started",
                "instruction": trajectory["instruction"],
                "app": app_name,
                "task_name": task_name,
                "mode": "offline_conversion",
            },
        )
        encoder = xmlEncoder()
        encoder.init(str(log_root))
        pages_by_identity: dict[str, dict[str, Any]] = {}
        task_path: dict[str, list[str]] = {}
        subtask_history: list[str] = []
        embed = embedding_provider or get_openai_embedding

        for screen_index, transition in enumerate(transitions):
            raw_xml = mobilegpt_compatible_xml(transition.forest)
            raw_path = Path(encoder.xml_directory) / f"{screen_index}.xml"
            raw_path.write_text(raw_xml, encoding="utf-8")
            parsed_xml, hierarchy_xml, encoded_xml = encoder.encode(
                raw_xml,
                screen_index,
            )
            identity = _screen_identity(hierarchy_xml)
            page = pages_by_identity.get(identity)
            if page is None:
                page_index = len(pages_by_identity)
                page_root = memory / app_name / "pages" / str(page_index)
                screen_root = page_root / "screen"
                screen_root.mkdir(parents=True)
                artifacts = {
                    "raw.xml": raw_path,
                    "html.xml": Path(encoder.xml_directory)
                    / f"{screen_index}_encoded.xml",
                    "hierarchy.xml": Path(encoder.xml_directory)
                    / f"{screen_index}_hierarchy_parsed.xml",
                    "parsed.xml": Path(encoder.xml_directory)
                    / f"{screen_index}_parsed.xml",
                    "pretty.xml": Path(encoder.xml_directory)
                    / f"{screen_index}_pretty.xml",
                }
                for name, source in artifacts.items():
                    if not source.is_file():
                        raise MobileGPTConversionError(
                            "source_screen_artifact_missing",
                            step_index=transition.step_index,
                            artifact=name,
                        )
                    shutil.copy2(source, screen_root / name)
                embedding = [float(value) for value in embed(hierarchy_xml)]
                if not embedding:
                    raise MobileGPTConversionError(
                        "mobilegpt_page_embedding_empty",
                        step_index=transition.step_index,
                    )
                if semantic_conversion:
                    with _temporary_agent_query_provider(
                        (explore_agent_module,),
                        semantic_query_provider,
                    ):
                        explored = _explore_page_with_mobilegpt(
                            ExploreAgent,
                            parsed_xml=parsed_xml,
                            hierarchy_xml=hierarchy_xml,
                            encoded_xml=encoded_xml,
                            screen_index=screen_index,
                            source_step_index=transition.step_index,
                        )
                else:
                    explored = {
                        "available_subtasks": [],
                        "trigger_uis": {},
                        "extra_uis": [],
                    }
                page = {
                    "index": page_index,
                    "parsed_xml": parsed_xml,
                    "hierarchy_xml": hierarchy_xml,
                    "encoded_xml": encoded_xml,
                    "embedding": embedding,
                    "available_subtasks": explored["available_subtasks"],
                    "trigger_uis": explored["trigger_uis"],
                    "extra_uis": explored["extra_uis"],
                    "subtasks": {},
                    "actions": [],
                }
                pages_by_identity[identity] = page
            page_index = int(page["index"])
            if semantic_conversion:
                with _temporary_agent_query_provider(
                    (select_agent_module,),
                    semantic_query_provider,
                ):
                    subtask, selected_subtask, example = _select_subtask_with_mobilegpt(
                        SelectAgent,
                        default_subtasks,
                        instruction=trajectory["instruction"],
                        available_subtasks=page["available_subtasks"],
                        subtask_history=subtask_history,
                        encoded_xml=encoded_xml,
                        source_action_type=_action_type(transition.action),
                        source_step_index=transition.step_index,
                    )
            else:
                subtask, selected_subtask, example = _direct_subtask_from_runlog(
                    transition,
                    parsed_xml,
                    encoded_xml,
                    trajectory["instruction"],
                    trajectory["task_parameters"],
                )
                page["available_subtasks"].append(subtask)
            derive_fallback_used = False
            try:
                converted, bindings, label = _mobilegpt_action_from_runlog(
                    transition,
                    parsed_xml,
                    task_parameters=trajectory["task_parameters"],
                    selected_subtask=selected_subtask,
                    generalize_action=generalize_action,
                )
            except MobileGPTConversionError as error:
                if (
                    not semantic_conversion
                    or
                    error.code != "source_action_target_unresolved"
                    or _action_type(transition.action) != "input_text"
                ):
                    raise
                with _temporary_agent_query_provider(
                    (derive_agent_module,),
                    semantic_query_provider,
                ):
                    converted, bindings, label = _mobilegpt_action_from_derive(
                        DeriveAgent,
                        transition,
                        parsed_xml,
                        encoded_xml,
                        instruction=trajectory["instruction"],
                        selected_subtask=selected_subtask,
                        subtask_history=subtask_history,
                        task_parameters=trajectory["task_parameters"],
                        generalize_action=generalize_action,
                    )
                derive_fallback_used = True
            del label
            existing_subtask = page["subtasks"].get(subtask["name"])
            if existing_subtask is None:
                page["subtasks"][subtask["name"]] = {
                    "metadata": subtask,
                    "example": example,
                }
            elif not existing_subtask["example"] and example:
                existing_subtask["example"] = example
            page["actions"].extend(
                [
                    {
                        "subtask_name": subtask["name"],
                        "step": 0,
                        "action": _json_text(converted),
                        "example": _json_text({}),
                    },
                    {
                        "subtask_name": subtask["name"],
                        "step": 1,
                        "action": _json_text(
                            {"name": "finish", "parameters": {}}
                        ),
                        "example": _json_text({}),
                    },
                ]
            )
            task_path.setdefault(str(page_index), []).append(subtask["name"])
            subtask_history.append(
                f"Performed an action: {json.dumps(selected_subtask, ensure_ascii=False)}"
            )
            row = {
                "source_step_index": transition.step_index,
                "source_action_type": _action_type(transition.action),
                "memory_page_index": page_index,
                "memory_subtask_name": subtask["name"],
                "selected_subtask": selected_subtask,
                "subtask_parameter_bindings": bindings,
                "memory_action": converted,
                "matched": True,
                "reason": (
                    "mobilegpt_explore_select_derive_compiled"
                    if derive_fallback_used
                    else (
                        "mobilegpt_explore_select_compiled"
                        if semantic_conversion
                        else "runlog_direct_compiled"
                    )
                ),
                "derive_fallback_used": derive_fallback_used,
                "consumed_transitions": 1,
            }
            audit_rows.append(row)
            _write_event(stats, {"event": "mobilegpt_conversion_action_mapped", **row})

        final_page_index = str(audit_rows[-1]["memory_page_index"])
        task_path[final_page_index].append("finish")
        app_root = memory / app_name
        _write_csv(
            memory / "tasks.csv",
            ("name", "description", "parameters", "app"),
            [{**task, "parameters": _json_text(task["parameters"])}],
        )
        _write_csv(
            app_root / "tasks.csv",
            ("name", "path"),
            [{"name": task_name, "path": _json_text(task_path)}],
        )
        page_rows: list[dict[str, Any]] = []
        hierarchy_rows: list[dict[str, Any]] = []
        for page in pages_by_identity.values():
            page_index = int(page["index"])
            page_root = app_root / "pages" / str(page_index)
            available_subtasks = list(page["available_subtasks"])
            selected_subtasks = list(page["subtasks"].values())
            page_rows.append(
                {
                    "index": page_index,
                    "available_subtasks": _json_text(available_subtasks),
                    "trigger_uis": _json_text(page["trigger_uis"]),
                    "extra_uis": _json_text(page["extra_uis"]),
                    "screen": page["parsed_xml"],
                }
            )
            hierarchy_rows.append(
                {
                    "index": page_index,
                    "screen": page["hierarchy_xml"],
                    "embedding": str(page["embedding"]),
                }
            )
            _write_csv(
                page_root / "available_subtasks.csv",
                ("name", "description", "parameters"),
                [
                    {
                        "name": subtask["name"],
                        "description": subtask["description"],
                        "parameters": _json_text(subtask["parameters"]),
                    }
                    for subtask in available_subtasks
                ],
            )
            _write_csv(
                page_root / "subtasks.csv",
                ("name", "description", "parameters", "example"),
                [
                    {
                        "name": selected["metadata"]["name"],
                        "description": selected["metadata"]["description"],
                        "parameters": _json_text(
                            selected["metadata"]["parameters"]
                        ),
                        "example": _json_text(selected["example"]),
                    }
                    for selected in selected_subtasks
                ],
            )
            _write_csv(
                page_root / "actions.csv",
                ("subtask_name", "step", "action", "example"),
                page["actions"],
            )
        _write_csv(
            app_root / "pages.csv",
            ("index", "available_subtasks", "trigger_uis", "extra_uis", "screen"),
            page_rows,
        )
        _write_csv(
            app_root / "hierarchy.csv",
            ("index", "screen", "embedding"),
            hierarchy_rows,
        )

        official_memory = Memory(app_name, trajectory["instruction"], task_name)
        if len(official_memory.task_path) != len(task_path):
            raise MobileGPTConversionError("official_memory_task_path_load_failed")
        official_action_count = 0
        pages_by_index = {
            int(page["index"]): page for page in pages_by_identity.values()
        }
        for page in pages_by_identity.values():
            page_index = int(page["index"])
            official_memory.init_page_manager(page_index)
            official_action_count += len(official_memory.page_manager.action_data)
        source_direct_hit_count = 0
        for row in audit_rows:
            page_index = int(row["memory_page_index"])
            page = pages_by_index[page_index]
            official_memory.init_page_manager(page_index)
            recalled = official_memory.page_manager.get_next_action(
                row["selected_subtask"],
                page["encoded_xml"],
                0,
            )
            finished = official_memory.page_manager.get_next_action(
                row["selected_subtask"],
                page["encoded_xml"],
                1,
            )
            if (
                not isinstance(recalled, dict)
                or "examples" in recalled
                or not isinstance(finished, dict)
                or finished.get("name") != "finish"
            ):
                raise MobileGPTConversionError(
                    "mobilegpt_source_direct_hit_failed",
                    step_index=row["source_step_index"],
                    page_index=page_index,
                    subtask_name=row["memory_subtask_name"],
                )
            source_direct_hit_count += 1
        _write_event(
            stats,
            {
                "event": "task_finished",
                "instruction": trajectory["instruction"],
                "elapsed_sec": round(time.monotonic() - started, 6),
                "task_status": "offline_conversion",
                "subtask_count": len(audit_rows),
            },
        )

    audit_payload = {
        "schema_version": (
            MOBILEGPT_AUDIT_SCHEMA
            if semantic_conversion
            else MOBILEGPT_DIRECT_AUDIT_SCHEMA
        ),
        "conversion_mode": conversion_mode,
        "task_name": trajectory["task_name"],
        "source_run_log": trajectory["source_run_log"],
        "target_package": trajectory["target_package"],
        "original_mobilegpt_prompts": semantic_conversion,
        "explore_agent_used": semantic_conversion,
        "select_agent_used": semantic_conversion,
        "derive_agent_fallback_allowed": semantic_conversion,
        "derive_agent_fallback_count": sum(
            row["derive_fallback_used"] is True for row in audit_rows
        ),
        "generalize_action_used": True,
        "direct_subtasks_from_runlog": not semantic_conversion,
        "source_direct_hit_validation": True,
        "transition_count": len(transitions),
        "validated_transition_count": sum(row["consumed_transitions"] for row in audit_rows),
        "validation_rows": audit_rows,
        "actions_supplied_to_mobilegpt": True,
        "source_transitions_supplied": True,
        "source_success_boundary_supplied": True,
        "source_success_boundary": trajectory["source_success_boundary"],
        "official_reader_validation": {
            "task_path_pages": len(task_path),
            "page_count": len(pages_by_identity),
            "action_row_count": official_action_count,
            "source_direct_hit_count": source_direct_hit_count,
            "loadable": True,
        },
        "complete": True,
        "wall_sec": round(time.monotonic() - started, 6),
    }
    audit.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    memory_validation = validate_mobilegpt_memory(memory)
    return {
        "task": task,
        "memory_root": str(memory),
        "stats_path": str(stats),
        "audit_path": str(audit),
        "transition_count": len(transitions),
        "validated_transition_count": audit_payload["validated_transition_count"],
        "target_package": trajectory["target_package"],
        "target_app": trajectory["target_app"],
        "source_success_boundary": trajectory["source_success_boundary"],
        "official_reader_validation": audit_payload["official_reader_validation"],
        "wall_sec": audit_payload["wall_sec"],
        "memory_validation": memory_validation,
    }
