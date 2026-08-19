"""Compile one verified RunLog into task-local MobileGPT memory."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
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

from src.experiment.mobilegpt_contract import (
    MOBILEGPT_AUDIT_SCHEMA,
    MOBILEGPT_LEARNING_MODE,
    MOBILEGPT_MEMORY_MANIFEST,
    MOBILEGPT_MEMORY_SCHEMA,
)
from src.experiment.protocol import SOURCE_SEED
from src.integrations.android_world.host import (
    androidworld_observation_package,
    androidworld_observation_xml,
)
from src.integrations.mobilegpt_runtime import mobilegpt_compatible_xml
from src.integrations.runlog import import_run_log, infer_input_text_target
from omniflow.core.model import Action
from omniflow.transfer.runtime import load_transfer_state_catalog

CONVERSION_SOURCE_SCHEMA = "omniflow.mobilegpt.source.v2"
CONVERSION_MODE_DIRECT = "runlog_direct"
CONVERSION_AUDIT_SCHEMA = MOBILEGPT_AUDIT_SCHEMA

__all__ = [
    "MobileGPTConversionError",
    "convert_runlog_to_mobilegpt_memory",
    "preflight_runlog_conversion",
    "validate_memory_manifest",
    "validate_prepared_memory",
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
_MOBILEGPT_PLACEHOLDER = re.compile(r"<[^>]*>")
_MOBILEGPT_PLACEHOLDER_GRAMMAR = re.compile(r"<([^<>]+)__(-?\d+)>")


class MobileGPTConversionError(RuntimeError):
    """A stable, machine-readable RunLog conversion failure."""

    def __init__(self, code: str, **details: Any) -> None:
        self.code = str(code)
        self.details = details
        suffix = ":" + json.dumps(details, ensure_ascii=False, sort_keys=True) if details else ""
        super().__init__(self.code + suffix)


def _memory_files(root: Path) -> tuple[list[Path], list[Path]]:
    if not root.is_dir():
        return [], []
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    task_files = [
        path
        for path in files
        if path.name == "tasks.csv" and path.parent.parent == root
    ]
    return files, task_files


def _hash_memory_files(files: list[Path], *, root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).parts):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_memory_manifest(memory_root: str | Path) -> dict[str, Any]:
    """Validate the MobileGPT manifest and its sealed evidence files."""

    root = Path(memory_root).expanduser().resolve()
    manifest_path = root.parent / MOBILEGPT_MEMORY_MANIFEST
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get(
        "schema_version"
    ) != MOBILEGPT_MEMORY_SCHEMA:
        raise ValueError("mobilegpt_memory_manifest_schema_invalid")
    if payload.get("source_seed") != SOURCE_SEED:
        raise ValueError("mobilegpt_memory_source_seed_invalid")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("mobilegpt_memory_provenance_missing")
    required_provenance = {
        "native_mobilegpt_learning": False,
        "task_local_memory": True,
        "learning_mode": MOBILEGPT_LEARNING_MODE,
        "teacher_forcing": False,
        "synthetic_subtasks": True,
        "semantic_subtasks": False,
        "original_mobilegpt_prompts": False,
        "actions_supplied_to_mobilegpt": True,
        "source_transitions_supplied": True,
        "source_success_boundary_supplied": True,
        "runlog_transition_compilation": True,
        "complete_transition_mapping": True,
        "official_reader_validation": True,
        "source_emulator_used": False,
        "function_store_used": False,
    }
    if any(provenance.get(key) != value for key, value in required_provenance.items()):
        raise ValueError("mobilegpt_memory_provenance_incomplete")
    forbidden = [
        key
        for key in (
            "function_conversion_enabled",
            "target_inputs_read",
            "target_observations_read",
            "validator_state_read",
            "coordinate_replay",
        )
        if bool(provenance.get(key))
    ]
    if forbidden:
        raise ValueError("mobilegpt_memory_forbidden:" + ",".join(forbidden))
    memory = payload.get("memory")
    if not isinstance(memory, dict):
        raise ValueError("mobilegpt_memory_record_missing")
    recorded_root = (root.parent / str(memory.get("relative_path") or "")).resolve()
    if recorded_root != root:
        raise ValueError("mobilegpt_memory_path_mismatch")
    files, task_files = _memory_files(root)
    digest = _hash_memory_files(files, root=root)
    if digest != str(memory.get("sha256") or ""):
        raise ValueError("mobilegpt_memory_hash_mismatch")
    if len(files) != int(memory.get("file_count") or -1):
        raise ValueError("mobilegpt_memory_file_count_mismatch")
    for label in ("source_run_log", "source_stats", "trajectory_audit"):
        record = payload.get(label)
        if not isinstance(record, dict):
            raise ValueError(f"mobilegpt_memory_{label}_missing")
        path = (root.parent / str(record.get("relative_path") or "")).resolve()
        try:
            path.relative_to(root.parent)
        except ValueError as error:
            raise ValueError(f"mobilegpt_memory_{label}_outside_bundle") from error
        if not path.is_file():
            raise ValueError(f"mobilegpt_memory_{label}_file_missing")
        if hashlib.sha256(path.read_bytes()).hexdigest() != str(
            record.get("sha256") or ""
        ):
            raise ValueError(f"mobilegpt_memory_{label}_hash_mismatch")
    if "official_source_result" in payload:
        raise ValueError("mobilegpt_memory_official_source_forbidden")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "task_name": str(payload.get("task_name") or ""),
        "source_seed": int(payload["source_seed"]),
        "memory_sha256": digest,
        "memory_file_count": len(files),
        "task_file_count": len(task_files),
    }


def validate_prepared_memory(
    memory_root: str | Path,
    *,
    task_name: str,
    source_seed: int,
    source_run_log: str | Path,
    compatible_source_sha256s: Sequence[str] = (),
    expected_model: str = "",
    expected_source_method: str = "",
) -> dict[str, Any]:
    """Validate one MobileGPT prepared memory at the provider seam.

    The experiment index depends on this provider contract, not on the
    AndroidWorld result runner.
    """

    from src.integrations.mobilegpt_memory import validate_mobilegpt_adapted_memory

    return validate_mobilegpt_adapted_memory(
        memory_root,
        task_name=task_name,
        source_seed=source_seed,
        source_run_log=source_run_log,
        compatible_source_sha256s=compatible_source_sha256s,
        expected_model=expected_model,
        expected_source_method=expected_source_method,
    )


@dataclass(frozen=True)
class _RunLogTransition:
    step_index: int
    action: dict[str, Any]
    observation: dict[str, Any]
    forest: str
    next_forest: str = ""


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
        page_is_launch_only = len(task_path) == 1 and raw_subtasks == ["finish"]
        if (
            (not page_is_launch_only and not subtask_names)
            or not subtask_names.issubset(available_names)
        ):
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

    launch_only = len(task_path) == 1 and all(
        raw_subtasks == ["finish"] for raw_subtasks in task_path.values()
    )
    if not launch_only and task_subtask_count <= 0:
        raise ValueError("mobilegpt_memory_recallable_subtask_missing")
    if not launch_only and non_finish_action_count <= 0:
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
        "launch_only": launch_only,
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
    return androidworld_observation_package(observation)


def _load_runlog_trajectory(
    source_run_log: str | Path,
    *,
    target_package: str = "",
    target_app: str = "",
) -> dict[str, Any]:
    """Load one successful canonical RunLog for deterministic offline compilation."""

    path = Path(source_run_log).expanduser().resolve()
    raw_payload = json.loads(path.read_text(encoding="utf-8"))
    if raw_payload.get("schema_version") == "omniflow.canonical_run_log.v1":
        return _load_compact_runlog_trajectory(
            path,
            raw_payload,
            target_package=target_package,
            target_app=target_app,
        )
    payload = import_run_log(raw_payload)
    if payload.get("status") != "succeeded" or payload.get("success") is not True:
        raise MobileGPTConversionError("source_runlog_not_successful", path=str(path))

    transitions: list[_RunLogTransition] = []
    skipped: list[dict[str, Any]] = []
    packages: list[str] = []
    launch_action: dict[str, Any] | None = None
    launch_step_index = -1
    raw_steps = list(payload.get("steps") or [])
    for ordinal, raw_step in enumerate(raw_steps):
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
                launch_action = action
                launch_step_index = step_index
        if action_type in _SKIPPED_ACTION_TYPES:
            skipped.append({"step_index": step_index, "action_type": action_type})
            continue
        if action_type not in _SUPPORTED_ACTION_TYPES:
            raise MobileGPTConversionError(
                "source_action_unsupported",
                step_index=step_index,
                action_type=action_type or "missing",
            )
        forest = androidworld_observation_xml(observation)
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
                next_forest=(
                    androidworld_observation_xml(
                        _observation_for_step(raw_steps[ordinal + 1])
                    )
                    if ordinal + 1 < len(raw_steps)
                    and isinstance(raw_steps[ordinal + 1], dict)
                    else ""
                ),
            )
        )

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
    launch_only = not transitions and launch_action is not None
    terminal_observation = payload.get("final_observation")
    if not isinstance(terminal_observation, dict) and raw_steps:
        candidate = raw_steps[-1].get("next_observation")
        terminal_observation = candidate if isinstance(candidate, dict) else None
    terminal_forest = (
        androidworld_observation_xml(terminal_observation)
        if isinstance(terminal_observation, dict)
        else ""
    )
    if launch_only:
        if not terminal_forest:
            raise MobileGPTConversionError("source_launch_final_observation_missing")
        try:
            ET.fromstring(terminal_forest)
        except ET.ParseError as error:
            raise MobileGPTConversionError(
                "source_launch_final_observation_invalid_xml",
                error=str(error),
            ) from error
    elif not transitions:
        raise MobileGPTConversionError("source_trajectory_empty")
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
        "launch_only": launch_only,
        "launch_action": launch_action,
        "launch_step_index": launch_step_index,
        "terminal_observation": terminal_observation,
        "terminal_forest": terminal_forest,
        "source_success_boundary": {
            "status": payload.get("status"),
            "success": payload.get("success"),
            "validator": dict(payload.get("validator") or {}),
        },
    }


def _load_compact_runlog_trajectory(
    path: Path,
    payload: dict[str, Any],
    *,
    target_package: str,
    target_app: str,
) -> dict[str, Any]:
    """Hydrate the canonical compact trace without changing its source file."""

    if payload.get("status") != "succeeded" or payload.get("success") is not True:
        raise MobileGPTConversionError("source_runlog_not_successful", path=str(path))
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, dict) or diagnostics.get("official_success") is not True:
        raise MobileGPTConversionError("source_runlog_not_official", path=str(path))
    catalog_path = path.with_name("transfer_states.json")
    states = load_transfer_state_catalog(catalog_path)
    transitions: list[_RunLogTransition] = []
    skipped: list[dict[str, Any]] = []
    packages: list[str] = []
    launch_action: dict[str, Any] | None = None
    launch_step_index = -1
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise MobileGPTConversionError("source_trajectory_empty")
    for ordinal, raw_step in enumerate(steps):
        if not isinstance(raw_step, dict) or raw_step.get("step_index") != ordinal:
            raise MobileGPTConversionError(
                "source_step_invalid",
                step_index=ordinal,
            )
        result = raw_step.get("result")
        if not isinstance(result, dict) or result.get("success") is not True:
            skipped.append(
                {"step_index": ordinal, "action_type": "unsuccessful"}
            )
            continue
        before_state_id = str(raw_step.get("before_state_id") or "").strip()
        after_state_id = str(raw_step.get("after_state_id") or "").strip()
        before = states.get(before_state_id)
        after = states.get(after_state_id)
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise MobileGPTConversionError(
                "source_observation_missing",
                step_index=ordinal,
            )
        if before_state_id and before_state_id == after_state_id:
            skipped.append(
                {"step_index": ordinal, "action_type": "no_state_change"}
            )
            continue
        package_name = str(before.get("package_name") or "").strip()
        if package_name:
            packages.append(package_name)
        after_package = str(after.get("package_name") or "").strip()
        if after_package:
            packages.append(after_package)
        action = _compact_action_to_androidworld(
            Action.from_value(raw_step.get("action")),
            state=before,
            step_index=ordinal,
        )
        action_type = _action_type(action)
        observation = _compact_observation(before)
        if (
            action_type == "click"
            and package_name
            in {
                "com.google.android.apps.nexuslauncher",
                "com.android.launcher3",
            }
            and after_package
            and after_package != package_name
            and after_package != "com.android.systemui"
        ):
            launch_action = {
                "action_type": "open_app",
                "app_name": after_package,
            }
            launch_step_index = ordinal
            skipped.append(
                {"step_index": ordinal, "action_type": "open_app"}
            )
            continue
        if action_type == "open_app":
            app_name = str(action.get("app_name") or "").strip()
            if app_name:
                packages.append(app_name)
                launch_action = action
                launch_step_index = ordinal
        if action_type in _SKIPPED_ACTION_TYPES:
            skipped.append({"step_index": ordinal, "action_type": action_type})
            continue
        if action_type not in _SUPPORTED_ACTION_TYPES:
            raise MobileGPTConversionError(
                "source_action_unsupported",
                step_index=ordinal,
                action_type=action_type or "missing",
            )
        forest = str(before.get("xml") or "").strip()
        next_forest = str(after.get("xml") or "").strip()
        try:
            ET.fromstring(forest)
        except ET.ParseError as error:
            raise MobileGPTConversionError(
                "source_observation_invalid_xml",
                step_index=ordinal,
                error=str(error),
            ) from error
        transitions.append(
            _RunLogTransition(
                step_index=ordinal,
                action=action,
                observation=observation,
                forest=forest,
                next_forest=next_forest,
            )
        )

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
    if not resolved_target_package and isinstance(launch_action, dict):
        resolved_target_package = str(launch_action.get("app_name") or "").strip()
    if not resolved_target_package:
        resolved_target_package = package_names[0] if len(package_names) == 1 else ""
    if not resolved_target_package:
        raise MobileGPTConversionError(
            "source_target_package_unresolved",
            packages=package_names,
        )
    resolved_target_app = str(target_app or resolved_target_package).strip()
    final_state_id = str(payload.get("final_state_id") or "").strip()
    terminal = states.get(final_state_id) if final_state_id else None
    task_name = str(
        diagnostics.get("task_id")
        or diagnostics.get("task_name")
        or path.parent.name
    ).strip()
    return {
        "schema_version": CONVERSION_SOURCE_SCHEMA,
        "source_run_log": str(path),
        "run_id": str(payload.get("run_id") or ""),
        "task_name": task_name,
        "instruction": str(payload.get("goal") or ""),
        "task_parameters": dict(payload.get("task_parameters") or {}),
        "source_seed": payload.get("seed"),
        "target_package": resolved_target_package,
        "target_app": resolved_target_app,
        "transitions": transitions,
        "skipped_actions": skipped,
        "launch_only": not transitions and launch_action is not None,
        "launch_action": launch_action,
        "launch_step_index": launch_step_index,
        "terminal_observation": (
            _compact_observation(terminal) if isinstance(terminal, dict) else None
        ),
        "terminal_forest": (
            str(terminal.get("xml") or "") if isinstance(terminal, dict) else ""
        ),
        "source_success_boundary": {
            "status": payload.get("status"),
            "success": payload.get("success"),
            "validator": {"official": True, "success": True},
        },
    }


def _compact_observation(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "forest": str(state.get("xml") or ""),
        "ui_elements": [],
        "auxiliaries": {
            "package_name": str(state.get("package_name") or ""),
            "activity_name": str(state.get("activity_name") or ""),
            "display": dict(state.get("display") or {}),
        },
    }


def _compact_action_to_androidworld(
    action: Action,
    *,
    state: dict[str, Any],
    step_index: int,
) -> dict[str, Any]:
    args = dict(action.args)
    if action.tool in {"click", "double_tap", "input_text", "long_press"}:
        display = state.get("display")
        if not isinstance(display, dict):
            raise MobileGPTConversionError(
                "source_observation_display_missing",
                step_index=step_index,
            )
        width, height = display.get("width"), display.get("height")
        try:
            x = float(args["x"]) * float(width) / 1000.0
            y = float(args["y"]) * float(height) / 1000.0
        except (KeyError, TypeError, ValueError) as error:
            raise MobileGPTConversionError(
                "source_action_point_invalid",
                step_index=step_index,
            ) from error
        converted: dict[str, Any] = {
            "action_type": action.tool,
            "x": int(round(x)),
            "y": int(round(y)),
        }
        if action.tool == "input_text":
            converted["text"] = str(args.get("text") or "")
            converted["clear_text"] = bool(args.get("clear_text", True))
        return converted
    if action.tool == "swipe":
        return {
            "action_type": "swipe",
            "direction": str(args.get("direction") or "").strip().lower(),
        }
    if action.tool in {"press_back", "navigate_back"}:
        return {"action_type": "navigate_back"}
    if action.tool in {"press_home", "navigate_home"}:
        return {"action_type": "navigate_home"}
    if action.tool == "press_key":
        key = str(args.get("key") or args.get("keycode") or "").strip().lower()
        if key.removeprefix("keycode_") == "back":
            return {"action_type": "navigate_back"}
        if key.removeprefix("keycode_") == "home":
            return {"action_type": "navigate_home"}
    if action.tool == "open_app":
        return {
            "action_type": "open_app",
            "app_name": str(
                args.get("package_name") or args.get("app_name") or ""
            ).strip(),
        }
    return {"action_type": action.tool}


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
    next_forest: str = "",
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
    if point is None and action_type == "input_text":
        changed_input = infer_input_text_target(
            source_forest,
            next_forest,
            input_text=str(action.get("text") or ""),
        )
        ordinal = changed_input.get("input_ordinal")
        inputs = [element for element in indexed if element.tag == "input"]
        if isinstance(ordinal, int) and 0 <= ordinal < len(inputs):
            return inputs[ordinal]
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
    parameters: dict[str, str] = {}
    for raw_name, raw_value in task_parameters.items():
        name = str(raw_name).strip()
        value = str(raw_value)
        if not name or not value.strip():
            continue
        normalized = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
        normalized = re.sub(r"_{2,}", "_", normalized) or "parameter"
        if normalized != name:
            digest = hashlib.sha256(name.encode()).hexdigest()[:8]
            normalized = f"{normalized}_{digest}"
        parameters[normalized] = value
    return parameters


def _validate_action_placeholders(
    value: Any,
    *,
    step_index: int,
) -> None:
    if isinstance(value, dict):
        for nested in value.values():
            _validate_action_placeholders(nested, step_index=step_index)
        return
    if isinstance(value, (list, tuple)):
        for nested in value:
            _validate_action_placeholders(nested, step_index=step_index)
        return
    if not isinstance(value, str):
        return
    for match in _MOBILEGPT_PLACEHOLDER.finditer(value):
        placeholder = match.group(0)
        parsed = _MOBILEGPT_PLACEHOLDER_GRAMMAR.fullmatch(placeholder)
        if parsed is None or "__" in parsed.group(1):
            raise MobileGPTConversionError(
                "mobilegpt_action_placeholder_invalid",
                step_index=step_index,
                placeholder=placeholder,
            )


def _contains_parameter_placeholder(value: Any, parameter_name: str) -> bool:
    if isinstance(value, dict):
        return any(
            _contains_parameter_placeholder(nested, parameter_name)
            for nested in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(
            _contains_parameter_placeholder(nested, parameter_name)
            for nested in value
        )
    if not isinstance(value, str):
        return False
    for match in _MOBILEGPT_PLACEHOLDER_GRAMMAR.finditer(value):
        if match.group(1) == parameter_name:
            return True
    return False


def _generalize_action_safely(
    action: dict[str, Any],
    *,
    subtask_name: str,
    semantic_parameters: dict[str, str],
    screen: str,
    step_index: int,
    generalize_action: Callable[[dict[str, Any], dict[str, Any], str], dict[str, Any]],
) -> dict[str, Any]:
    base_subtask = {"name": subtask_name, "parameters": {}}
    converted = generalize_action(deepcopy(action), base_subtask, screen)
    if not isinstance(converted, dict):
        raise MobileGPTConversionError(
            "mobilegpt_action_generalization_failed",
            step_index=step_index,
        )
    for parameter_name, parameter_value in semantic_parameters.items():
        candidate = generalize_action(
            deepcopy(action),
            {
                "name": subtask_name,
                "parameters": {parameter_name: parameter_value},
            },
            screen,
        )
        if not isinstance(candidate, dict):
            continue
        _validate_action_placeholders(candidate, step_index=step_index)
        candidate_parameters = candidate.get("parameters")
        converted_parameters = converted.get("parameters")
        if not isinstance(candidate_parameters, dict) or not isinstance(
            converted_parameters,
            dict,
        ):
            continue
        for key, value in candidate_parameters.items():
            if _contains_parameter_placeholder(value, parameter_name):
                converted_parameters[key] = value
    _validate_action_placeholders(converted, step_index=step_index)
    return converted


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
            next_forest=transition.next_forest,
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


def _native_action_example(
    *,
    instruction: str,
    selected_subtask: dict[str, Any],
    encoded_xml: str,
    action: dict[str, Any],
) -> dict[str, Any]:
    parameters = action.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {}
    concrete_parameters = {
        key: value
        for key, value in parameters.items()
        if key in {"direction", "index", "input_text", "message", "number"}
    }
    concrete_action = {
        "name": str(action.get("name") or ""),
        "parameters": concrete_parameters,
    }
    response = {
        "reasoning": "Follow the verified successful source experience.",
        "action": concrete_action,
        "completion_rate": 1.0,
        "plan": "Execute this action, then verify the subtask state.",
    }
    return {
        "instruction": instruction,
        "subtask": _json_text(selected_subtask),
        "screen": encoded_xml,
        "response": _json_text(response),
    }


def _valid_native_action_examples(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    for example in value:
        if not isinstance(example, dict):
            return False
        if not all(
            isinstance(example.get(key), str) and example.get(key)
            for key in ("instruction", "subtask", "screen", "response")
        ):
            return False
        try:
            subtask = json.loads(example["subtask"])
            response = json.loads(example["response"])
        except json.JSONDecodeError:
            return False
        if not isinstance(subtask, dict) or not isinstance(response, dict):
            return False
        if not isinstance(response.get("action"), dict):
            return False
    return True


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
    changed_input: dict[str, Any] = {}
    if action_type in {"click", "double_tap", "input_text", "long_press"}:
        target = _target_element(
            action,
            parsed_xml,
            step_index=transition.step_index,
            source_forest=transition.forest,
            next_forest=transition.next_forest,
        )
        index = str(target.get("index"))
        if action_type == "input_text":
            changed_input = infer_input_text_target(
                transition.forest,
                transition.next_forest,
                input_text=str(action.get("text") or ""),
            )
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
    anonymous_verified_input = (
        action_type == "input_text"
        and changed_input.get("identity") == {"role": "editable"}
        and len(ET.fromstring(parsed_xml).findall(".//input")) == 1
    )
    if anonymous_verified_input:
        converted["parameters"]["attrib"] = {
            "self": {"tag": "input"},
            "parent": {},
            "children": [],
        }
        selected_parameters = selected_subtask.get("parameters")
        if not isinstance(selected_parameters, dict):
            selected_parameters = {}
        source_text = str(action.get("text") or "")
        matching_parameters = [
            str(name)
            for name, value in {**selected_parameters, **bindings}.items()
            if str(name).strip() and str(value) == source_text
        ]
        if len(matching_parameters) == 1:
            converted["parameters"]["input_text"] = (
                f"<{matching_parameters[0]}__-1>"
            )
    elif "index" in converted["parameters"]:
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
        converted = _generalize_action_safely(
            converted,
            subtask_name=str(selected_subtask.get("name") or "sourceStep"),
            semantic_parameters=semantic_parameters,
            screen=generalization_screen,
            step_index=transition.step_index,
            generalize_action=generalize_action,
        )
    label = _element_semantic_label(target)
    return converted, bindings, label


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
    payload = {
        "schema_version": MOBILEGPT_AUDIT_SCHEMA,
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
    embedding_model: str = "text-embedding-v4",
    target_package: str = "",
    target_app: str = "",
    embedding_provider: Callable[[str], Sequence[float]] | None = None,
    semantic_query_provider: Callable[..., Any] | None = None,
    conversion_mode: str = CONVERSION_MODE_DIRECT,
) -> dict[str, Any]:
    """Write one RunLog as an exact native-format MobileGPT database."""

    if conversion_mode != CONVERSION_MODE_DIRECT:
        raise ValueError(f"mobilegpt_conversion_mode_invalid:{conversion_mode}")
    normalized_embedding_model = (
        str(embedding_model or "").strip() or "text-embedding-v4"
    )

    trajectory = _load_runlog_trajectory(
        source_run_log,
        target_package=target_package,
        target_app=target_app,
    )
    transitions: list[_RunLogTransition] = trajectory["transitions"]
    launch_only = trajectory["launch_only"] is True
    encoded_transitions = transitions
    if launch_only:
        encoded_transitions = [
            _RunLogTransition(
                step_index=int(trajectory["launch_step_index"]),
                action=dict(trajectory["launch_action"]),
                observation=dict(trajectory["terminal_observation"]),
                forest=str(trajectory["terminal_forest"]),
            )
        ]
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
        "MOBILEGPT_EMBEDDING_MODEL": normalized_embedding_model,
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
        from memory.memory_manager import Memory
        from screenParser.Encoder import xmlEncoder
        from utils.action_utils import generalize_action
        if embedding_provider is None:
            from utils.utils import get_openai_embedding

            embed = get_openai_embedding
        else:
            embed = embedding_provider

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
        for screen_index, transition in enumerate(encoded_transitions):
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
                page = {
                    "index": page_index,
                    "parsed_xml": parsed_xml,
                    "hierarchy_xml": hierarchy_xml,
                    "encoded_xml": encoded_xml,
                    "embedding": embedding,
                    "available_subtasks": [],
                    "trigger_uis": {},
                    "extra_uis": [],
                    "subtasks": {},
                    "actions": [],
                }
                pages_by_identity[identity] = page
            page_index = int(page["index"])
            if launch_only:
                task_path[str(page_index)] = ["finish"]
                continue
            subtask, selected_subtask, example = _direct_subtask_from_runlog(
                transition,
                parsed_xml,
                encoded_xml,
                trajectory["instruction"],
                trajectory["task_parameters"],
            )
            page["available_subtasks"].append(subtask)
            converted, bindings, label = _mobilegpt_action_from_runlog(
                transition,
                parsed_xml,
                task_parameters=trajectory["task_parameters"],
                selected_subtask=selected_subtask,
                generalize_action=generalize_action,
            )
            del label
            action_example = _native_action_example(
                instruction=trajectory["instruction"],
                selected_subtask=selected_subtask,
                encoded_xml=encoded_xml,
                action=converted,
            )
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
                        "example": _json_text(action_example),
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
            row = {
                "source_step_index": transition.step_index,
                "source_action_type": _action_type(transition.action),
                "memory_page_index": page_index,
                "memory_subtask_name": subtask["name"],
                "selected_subtask": selected_subtask,
                "subtask_parameter_bindings": bindings,
                "memory_action": converted,
                "matched": True,
                "reason": "runlog_direct_compiled",
                "derive_fallback_used": False,
                "consumed_transitions": 1,
            }
            audit_rows.append(row)
            _write_event(stats, {"event": "mobilegpt_conversion_action_mapped", **row})

        if not launch_only:
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
        source_example_fallback_count = 0
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
            if not isinstance(recalled, dict):
                raise MobileGPTConversionError(
                    "mobilegpt_source_reader_coverage_failed",
                    step_index=row["source_step_index"],
                    page_index=page_index,
                    subtask_name=row["memory_subtask_name"],
                )
            if "examples" in recalled:
                if not _valid_native_action_examples(recalled.get("examples")):
                    raise MobileGPTConversionError(
                        "mobilegpt_source_example_invalid",
                        step_index=row["source_step_index"],
                        page_index=page_index,
                        subtask_name=row["memory_subtask_name"],
                    )
                source_example_fallback_count += 1
                row["reader_resolution"] = "native_example_fallback"
            else:
                source_direct_hit_count += 1
                row["reader_resolution"] = "direct_hit"
            if not isinstance(finished, dict) or finished.get("name") != "finish":
                raise MobileGPTConversionError(
                    "mobilegpt_source_finish_recall_failed",
                    step_index=row["source_step_index"],
                    page_index=page_index,
                    subtask_name=row["memory_subtask_name"],
                )
        launch_finish_validated = False
        if launch_only:
            page = next(iter(pages_by_identity.values()))
            page_index = int(page["index"])
            official_memory.init_page_manager(page_index)
            finish_subtask = official_memory.get_next_subtask(
                page_index,
                [],
                page["encoded_xml"],
            )
            if (
                not isinstance(finish_subtask, dict)
                or finish_subtask.get("name") != "finish"
            ):
                raise MobileGPTConversionError(
                    "mobilegpt_launch_finish_recall_failed",
                    page_index=page_index,
                )
            launch_finish_validated = True
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
        "schema_version": MOBILEGPT_AUDIT_SCHEMA,
        "conversion_mode": conversion_mode,
        "task_name": trajectory["task_name"],
        "source_run_log": trajectory["source_run_log"],
        "target_package": trajectory["target_package"],
        "embedding_model": normalized_embedding_model,
        "original_mobilegpt_prompts": False,
        "explore_agent_used": False,
        "select_agent_used": False,
        "derive_agent_fallback_allowed": True,
        "derive_agent_fallback_count": 0,
        "source_example_fallback_count": source_example_fallback_count,
        "generalize_action_used": bool(audit_rows),
        "direct_subtasks_from_runlog": True,
        "source_direct_hit_validation": source_example_fallback_count == 0,
        "source_reader_coverage_validation": True,
        "launch_only": launch_only,
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
            "source_example_fallback_count": source_example_fallback_count,
            "source_reader_coverage_count": (
                source_direct_hit_count + source_example_fallback_count
            ),
            "launch_finish_validated": launch_finish_validated,
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
        "embedding_model": normalized_embedding_model,
        "source_success_boundary": trajectory["source_success_boundary"],
        "official_reader_validation": audit_payload["official_reader_validation"],
        "wall_sec": audit_payload["wall_sec"],
        "memory_validation": memory_validation,
    }
