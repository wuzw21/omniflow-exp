"""Compile one verified RunLog into task-local MobileGPT memory."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import ast
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from typing import Any, Callable, Iterator, Sequence
import xml.etree.ElementTree as ET
import xml.dom.minidom

from src.experiment.mobilegpt_contract import (
    MOBILEGPT_AUDIT_SCHEMA,
    MOBILEGPT_EMBEDDING_MODEL,
    MOBILEGPT_LEARNING_MODE,
    MOBILEGPT_MEMORY_MANIFEST,
    MOBILEGPT_MEMORY_SCHEMA,
)
from src.experiment.protocol import SOURCE_SEED
from src.integrations.android_world.host import (
    androidworld_observation_package,
    androidworld_observation_xml,
)
from src.integrations.mobilegpt_format import encode_xml
from src.integrations.runlog import (
    adapt_source_run_log,
    import_run_log,
    infer_input_text_target,
)
from omniflow.core.model import Action
from omniflow.transfer.runtime import load_transfer_state_catalog

CONVERSION_SOURCE_SCHEMA = "omniflow.mobilegpt.source.v2"
CONVERSION_MODE_DIRECT = "runlog_direct"
CONVERSION_MODE_OFFICIAL = "official_mobilegpt_learning"
CONVERSION_AUDIT_SCHEMA = MOBILEGPT_AUDIT_SCHEMA
# The upstream Server normally terminates by sending ``$$$$$``.  A malformed
# model response or a dirty upstream checkout can otherwise keep asking for
# the same screen forever.  Keep the historical successful range (the old
# source runs used at most 53 chat calls) while making the boundary explicit.
_DEFAULT_AUTHORING_MAX_CHAT_CALLS = 64
_DEFAULT_AUTHORING_MAX_FINAL_CYCLES = 8

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
_LAUNCHER_PACKAGES = frozenset(
    {
        "com.google.android.apps.nexuslauncher",
        "com.android.launcher3",
    }
)
_SUPPORTED_ACTION_TYPES = frozenset(
    {
        "answer",
        "click",
        "double_tap",
        "input_text",
        "long_press",
        "navigate_back",
        "scroll",
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
    if len(files) != int(memory.get("file_count") or -1):
        raise ValueError("mobilegpt_memory_file_count_mismatch")
    if str(memory.get("sha256") or "") != digest:
        raise ValueError("mobilegpt_memory_sha256_mismatch")
    evidence_paths: dict[str, Path] = {}
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
        if str(record.get("sha256") or "") != hashlib.sha256(
            path.read_bytes()
        ).hexdigest():
            raise ValueError(f"mobilegpt_memory_{label}_sha256_mismatch")
        evidence_paths[label] = path
    audit = json.loads(
        evidence_paths["trajectory_audit"].read_text(encoding="utf-8")
    )
    transition_count = int(audit.get("transition_count") or 0)
    validation_rows = audit.get("validation_rows")
    source_payload = import_run_log(
        json.loads(evidence_paths["source_run_log"].read_text(encoding="utf-8"))
    )
    official_reader = audit.get("official_reader_validation")
    launch_only = False
    if isinstance(official_reader, dict):
        from src.integrations.mobilegpt_memory import (
            is_valid_mobilegpt_launch_only_memory,
        )

        launch_only = is_valid_mobilegpt_launch_only_memory(
            source_payload,
            audit,
            official_reader,
        )
    common_alignment_invalid = (
        not isinstance(audit, dict)
        or audit.get("schema_version") != MOBILEGPT_AUDIT_SCHEMA
        or str(audit.get("task_name") or "")
        != str(payload.get("task_name") or "")
        or audit.get("complete") is not True
        or audit.get("conversion_mode") != CONVERSION_MODE_DIRECT
        or audit.get("actions_supplied_to_mobilegpt") is not True
        or audit.get("source_reader_coverage_validation") is not True
    )
    transition_alignment_invalid = (
        transition_count <= 0
        or int(audit.get("validated_transition_count") or 0)
        != transition_count
        or not isinstance(validation_rows, list)
        or not validation_rows
        or any(
            not isinstance(row, dict)
            or row.get("matched") is not True
            for row in validation_rows
        )
        or sum(
            int(row.get("consumed_transitions") or 0)
            for row in validation_rows
        )
        != transition_count
    )
    if common_alignment_invalid or (
        not launch_only and transition_alignment_invalid
    ):
        raise ValueError("mobilegpt_memory_runlog_alignment_invalid")
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
        "runlog_direct_alignment": True,
        "launch_only": launch_only,
        "validated_transition_count": transition_count,
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
        except json.JSONDecodeError:
            # The official pandas writer serializes the task parameters
            # column with Python repr (single quotes). Read that native
            # artifact without rewriting or translating the memory graph.
            try:
                candidate = ast.literal_eval(candidate)
            except (ValueError, SyntaxError) as error:
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
    task_subtask_names = {
        str(subtask or "").strip()
        for subtasks_for_page in task_path.values()
        for subtask in subtasks_for_page
        if str(subtask or "").strip() not in {"finish", "scroll_screen"}
    }
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
        # Official MobileGPT may store the terminal ``finish`` on a distinct
        # page after the learned action page.  It is valid regardless of the
        # number of pages in the task path.
        page_is_special_only = bool(raw_subtasks) and all(
            str(subtask or "").strip() in {"finish", "scroll_screen"}
            for subtask in raw_subtasks
        )
        if (
            (not page_is_special_only and not subtask_names)
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
        for row in action_rows:
            subtask_name = str(row.get("subtask_name") or "").strip()
            # Official PageManager can persist the terminal action on the
            # following page while the task path records only ``finish``
            # there. It remains tied to the learned task subtask globally.
            if not subtask_name or subtask_name not in task_subtask_names:
                raise ValueError("mobilegpt_memory_action_subtask_invalid")
            try:
                step = int(str(row.get("step") or "").strip())
            except ValueError as error:
                raise ValueError("mobilegpt_memory_action_step_invalid") from error
            if step < 0:
                raise ValueError("mobilegpt_memory_action_step_invalid")
            action = _validated_action(row.get("action"))
            # ``actions.csv`` is the official PageManager event table, not a
            # single linear trace per subtask.  The official authoring flow
            # may revisit a subtask name, emit its terminal ``finish`` on a
            # later page, or persist two instances with the same name.  A
            # local contiguous-step rule therefore rejects valid official
            # memories (and cannot be used to repair their ordering).  Keep
            # the structural checks above, while leaving action sequencing to
            # MobileGPT's own reader/runtime.
            action_count += 1
            if action["name"] != "finish":
                non_finish_action_count += 1
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
    """Return the real foreground package, including forest-only observations.

    AndroidWorld's native package fields are not guaranteed to be populated
    in a source RunLog.  In particular, ``open_app`` commonly records the
    human-facing label while the package is present only on the XML forest.
    MobileGPT needs the package for its Accessibility client, so the source
    boundary must resolve it from that evidence instead of passing the label
    through as a package name.
    """

    package = androidworld_observation_package(observation)
    if package:
        return package
    if not isinstance(observation, dict):
        return ""
    forest = observation.get("xml") or observation.get("forest")
    if not isinstance(forest, str) or not forest.strip():
        return ""
    try:
        root = ET.fromstring(forest)
    except ET.ParseError:
        return ""
    ignored = {
        "android",
        "com.android.systemui",
        "com.google.android.apps.nexuslauncher",
        "com.android.launcher3",
    }
    packages = [
        str(element.attrib.get("package") or "").strip()
        for element in root.iter()
        if str(element.attrib.get("package") or "").strip()
    ]
    foreground = [value for value in packages if value not in ignored]
    return (foreground or packages or [""])[-1]


def _launched_target_package(before_package: str, after_package: str) -> str:
    """Recognize an app-icon click as launch evidence, not a task action."""

    before = str(before_package or "").strip()
    after = str(after_package or "").strip()
    if (
        before in _LAUNCHER_PACKAGES
        and after
        and after not in _LAUNCHER_PACKAGES
        and after not in {"android", "com.android.systemui"}
    ):
        return after
    return ""


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
    raw_steps = raw_payload.get("steps")
    legacy_steps = isinstance(raw_steps, list) and any(
        isinstance(step, dict)
        and any(
            key in step
            for key in ("before_state_id", "after_state_id", "observation_before_act")
        )
        for step in raw_steps
    )
    if raw_payload.get("schema_version") == "omniflow.run_log.v1" and legacy_steps:
        payload = adapt_source_run_log(
            raw_payload,
            task_name=str(raw_payload.get("task_name") or ""),
            task_parameters=dict(raw_payload.get("task_parameters") or {}),
            seed=(
                int(raw_payload["seed"])
                if type(raw_payload.get("seed")) is int
                else None
            ),
            source_path=path,
            screenshot_roots=(
                path.parent / "observations" / "objects",
                path.parent,
            ),
            require_screenshots=True,
        )
    else:
        payload = import_run_log(raw_payload)
    if payload.get("status") != "succeeded" or payload.get("success") is not True:
        raise MobileGPTConversionError("source_runlog_not_successful", path=str(path))

    transitions: list[_RunLogTransition] = []
    skipped: list[dict[str, Any]] = []
    packages: list[str] = []
    launch_action: dict[str, Any] | None = None
    launch_step_index = -1
    open_app_evidence: list[dict[str, Any]] = []
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
        action = _ground_indexed_action(action, observation)
        package = _package_from_observation(observation)
        if package:
            packages.append(package)
        next_observation = raw_step.get("next_observation")
        if not isinstance(next_observation, dict) and ordinal + 1 < len(raw_steps):
            next_observation = _observation_for_step(raw_steps[ordinal + 1])
        next_package = (
            _package_from_observation(next_observation)
            if isinstance(next_observation, dict)
            else ""
        )
        launched_package = (
            _launched_target_package(package, next_package)
            if action_type == "click"
            else ""
        )
        if launched_package:
            packages.append(launched_package)
            launch_action = {
                "action_type": "open_app",
                "app_name": launched_package,
            }
            launch_step_index = step_index
            open_app_evidence.append(
                {
                    "step_index": step_index,
                    "requested_package": launched_package,
                    "observed_package": launched_package,
                }
            )
            skipped.append({"step_index": step_index, "action_type": "open_app"})
            continue
        if action_type == "open_app":
            app_name = str(action.get("app_name") or action.get("package_name") or "").strip()
            if app_name:
                packages.append(app_name)
                launch_action = action
                launch_step_index = step_index
                open_app_evidence.append(
                    {
                        "step_index": step_index,
                        "requested_package": app_name,
                        "observed_package": next_package,
                    }
                )
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
    # The package is often only visible in the observation after open_app;
    # include that evidence before resolving the human-facing app label.
    package_names = sorted(
        set(package_names)
        | {
            str(evidence.get("observed_package") or "").strip()
            for evidence in open_app_evidence
            if str(evidence.get("observed_package") or "").strip()
        }
    )
    resolved_target_package = str(target_package or "").strip()
    resolved_target_app = str(target_app or "").strip()
    observed_open_app_packages = [
        str(evidence.get("observed_package") or "").strip()
        for evidence in open_app_evidence
        if "." in str(evidence.get("observed_package") or "").strip()
        and str(evidence.get("observed_package") or "").strip()
        not in {
            "com.android.systemui",
            "com.google.android.apps.nexuslauncher",
            "com.android.launcher3",
        }
    ]
    if not resolved_target_package and observed_open_app_packages:
        resolved_target_package = observed_open_app_packages[0]
    package_candidates = [value for value in package_names if "." in value]
    if not resolved_target_package:
        if len(package_candidates) == 1:
            resolved_target_package = package_candidates[0]
        elif len(package_names) == 1:
            resolved_target_package = package_names[0]
    open_app_packages = [
        str(step.get("action", {}).get("app_name") or "").strip()
        for step in payload.get("steps") or []
        if isinstance(step, dict)
        and isinstance(step.get("action"), dict)
        and _action_type(step["action"]) == "open_app"
        and str(step["action"].get("app_name") or "").strip()
    ]
    # The source action is the authoritative application intent.  Observed
    # package names may still belong to Launcher during the launch transition.
    if open_app_packages:
        resolved_target_package = open_app_packages[0]
    for evidence in open_app_evidence:
        observed_package = str(evidence.get("observed_package") or "").strip()
        if (
            not resolved_target_package
            and observed_package
            and observed_package not in {resolved_target_package, "android", "com.android.systemui"}
            and not observed_package.startswith("com.google.android.apps.nexuslauncher")
            and not observed_package.startswith("com.android.launcher")
        ):
            raise MobileGPTConversionError(
                "source_open_app_package_mismatch",
                step_index=evidence["step_index"],
                requested_package=resolved_target_package,
                observed_package=observed_package,
            )
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
        "target_package_source": "open_app",
        "open_app_evidence": open_app_evidence,
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
        launched_package = (
            _launched_target_package(package_name, after_package)
            if action_type == "click"
            else ""
        )
        if launched_package:
            launch_action = {
                "action_type": "open_app",
                "app_name": launched_package,
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
        "target_package_source": "open_app",
        "open_app_evidence": [
            {
                "step_index": launch_step_index,
                "requested_package": resolved_target_package,
                "observed_package": resolved_target_package,
            }
        ] if launch_action is not None else [],
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
    mobilegpt_root: str | Path | None = None,
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
    teacher_actions = (
        _teacher_actions_for_trajectory(
            trajectory,
            mobilegpt_root=mobilegpt_root,
        )
        if mobilegpt_root is not None
        else []
    )
    return {
        "schema_version": CONVERSION_SOURCE_SCHEMA,
        "source_run_log": str(path),
        "task_name": trajectory["task_name"],
        "target_package": trajectory["target_package"],
        "transition_count": len(transitions),
        "action_type_counts": counts,
        "teacher_action_count": len(teacher_actions),
        "teacher_actions": teacher_actions,
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


def _androidworld_index_bounds_from_xml(
    observation: dict[str, Any],
    index: int,
) -> tuple[float, float, float, float] | None:
    """Recover AndroidWorld's persisted forest element order from source XML."""

    forest = androidworld_observation_xml(observation)
    if not forest:
        return None
    try:
        root = ET.fromstring(forest)
    except ET.ParseError:
        return None
    elements: list[ET.Element] = []
    windows = list(root.iter("window"))
    for window in windows:
        ordered_nodes: list[tuple[int, ET.Element]] = []
        for element in window.iter("node"):
            raw_id = str(element.get("id") or "")
            try:
                order = int(raw_id.rsplit(":", 1)[1])
            except (IndexError, ValueError):
                return None
            ordered_nodes.append((order, element))
        for _, element in sorted(ordered_nodes, key=lambda item: item[0]):
            child_nodes = [child for child in element if child.tag == "node"]
            if (
                not child_nodes
                or bool(str(element.get("content-desc") or "").strip())
                or str(element.get("scrollable") or "").strip().lower() == "true"
            ):
                elements.append(element)
    if not windows:
        # Native uiautomator XML is converted by AndroidWorld's
        # ``xml_dump_to_ui_elements`` in preorder, excluding only the top
        # application root node.  Older successful RunLogs persist this XML
        # shape directly instead of an accessibility ``window`` forest.
        top_nodes = [child for child in root if child.tag == "node"]
        if len(top_nodes) != 1:
            return None
        elements = list(top_nodes[0].iter("node"))[1:]
    if not 0 <= index < len(elements):
        return None
    return _bounds(elements[index].get("bounds"))


def _ground_indexed_action(
    action: dict[str, Any],
    observation: dict[str, Any],
) -> dict[str, Any]:
    """Ground AndroidWorld's UI-element index for native MobileGPT parsing."""

    if _point(action) is not None:
        return action
    if _action_type(action) not in {"click", "double_tap", "input_text", "long_press"}:
        return action
    try:
        index = int(action["index"])
    except (KeyError, TypeError, ValueError):
        return action
    elements = observation.get("ui_elements")
    resolved_bounds: tuple[float, float, float, float] | None = None
    if isinstance(elements, list) and 0 <= index < len(elements):
        element = elements[index]
        if isinstance(element, dict):
            bounds = element.get("bbox_pixels")
            if isinstance(bounds, dict):
                try:
                    resolved_bounds = (
                        float(bounds["x_min"]),
                        float(bounds["y_min"]),
                        float(bounds["x_max"]),
                        float(bounds["y_max"]),
                    )
                except (KeyError, TypeError, ValueError):
                    resolved_bounds = None
    if resolved_bounds is None:
        resolved_bounds = _androidworld_index_bounds_from_xml(observation, index)
    if resolved_bounds is None:
        return action
    left, top, right, bottom = resolved_bounds
    grounded = dict(action)
    grounded["x"] = (left + right) / 2.0
    grounded["y"] = (top + bottom) / 2.0
    return grounded


def _element_contains_action(element: ET.Element, action: dict[str, Any]) -> bool:
    point = _point(action)
    element_bounds = _bounds(element.get("bounds"))
    if point is None or element_bounds is None:
        return False
    x, y = point
    left, top, right, bottom = element_bounds
    return left <= x <= right and top <= y <= bottom


def _nearby_range_control(
    indexed: Sequence[ET.Element],
    point: tuple[float, float],
    *,
    source_forest: str,
) -> ET.Element | None:
    """Ground an edge tap to one nearby range control within touch slop.

    AndroidWorld can record a successful tap a few pixels beyond a SeekBar's
    accessibility bounds, especially at the minimum/maximum endpoint.  The
    official MobileGPT XML still preserves that SeekBar.  Snap only to a
    unique nearby range control and never to arbitrary text or containers.
    """

    width = 0.0
    height = 0.0
    if source_forest:
        try:
            source_root = ET.fromstring(source_forest)
        except ET.ParseError:
            source_root = None
        if source_root is not None:
            try:
                width = float(source_root.get("width") or 0)
                height = float(source_root.get("height") or 0)
            except (TypeError, ValueError):
                width = height = 0.0
    bounds_rows = [
        (element, bounds)
        for element in indexed
        if (bounds := _bounds(element.get("bounds"))) is not None
    ]
    if width <= 0:
        width = max((bounds[2] for _, bounds in bounds_rows), default=0.0)
    if height <= 0:
        height = max((bounds[3] for _, bounds in bounds_rows), default=0.0)
    touch_slop = max(1.0, min(width, height) * 0.05)
    x, y = point
    nearby: list[tuple[float, ET.Element]] = []
    for element, (left, top, right, bottom) in bounds_rows:
        tag = str(element.tag or "").casefold()
        class_name = str(element.get("class") or "").casefold()
        if not any(
            marker in tag or marker in class_name
            for marker in ("seekbar", "slider", "range")
        ):
            continue
        dx = max(left - x, 0.0, x - right)
        dy = max(top - y, 0.0, y - bottom)
        distance = (dx * dx + dy * dy) ** 0.5
        if distance <= touch_slop:
            nearby.append((distance, element))
    nearby.sort(key=lambda item: item[0])
    if not nearby:
        return None
    if len(nearby) > 1 and abs(nearby[0][0] - nearby[1][0]) < 1e-6:
        return None
    return nearby[0][1]


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


def _mobilegpt_ui_match_xml(parsed_xml: str) -> str:
    """Prepare the exact XML shape expected by MobileGPT UI match helpers."""

    root = ET.fromstring(str(parsed_xml or ""))
    if root.tag != "hierarchy":
        wrapper = ET.Element("hierarchy")
        wrapper.append(root)
        root = wrapper
    used = {
        int(value)
        for element in root.iter()
        if (value := element.get("index")) is not None and str(value).isdigit()
    }
    next_index = max(used, default=-1) + 1
    for element in root.iter():
        if element.get("index") is None:
            while next_index in used:
                next_index += 1
            element.set("index", str(next_index))
            used.add(next_index)
            next_index += 1
    return ET.tostring(root, encoding="unicode")


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
    if (
        not candidates
        and point is not None
        and action_type in {"click", "double_tap", "long_press"}
    ):
        nearby_range = _nearby_range_control(
            indexed,
            point,
            source_forest=source_forest,
        )
        if nearby_range is not None:
            candidates = [nearby_range]
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
    generalize_action: Callable[[dict[str, Any], dict[str, Any], str], dict[str, Any]]
    | None = None,
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
    generalize_action: Callable[[dict[str, Any], dict[str, Any], str], dict[str, Any]]
    | None = None,
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
    elif action_type in {"scroll", "swipe"}:
        direction = _ANDROIDWORLD_SWIPE_TO_MOBILEGPT_SCROLL.get(
            _swipe_direction(action),
            "",
        )
        if action_type == "scroll":
            direction = str(action.get("direction") or "").strip().lower()
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
        and sum(
            1
            for element in ET.fromstring(parsed_xml).iter()
            if element.tag == "input"
        )
        == 1
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
    elif "index" in converted["parameters"] and callable(generalize_action):
        generalization_screen = _mobilegpt_ui_match_xml(parsed_xml)
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


def _teacher_actions_for_trajectory(
    trajectory: dict[str, Any],
    *,
    mobilegpt_root: str | Path | None,
) -> list[dict[str, Any]]:
    """Project successful RunLog actions into the official client action space."""

    teacher_actions: list[dict[str, Any]] = []
    for transition in trajectory["transitions"]:
        parsed_xml, _, encoded_xml = encode_xml(
            _official_xml_input(str(transition.forest)),
            mobilegpt_root=mobilegpt_root,
        )
        _, selected_subtask, _ = _direct_subtask_from_runlog(
            transition,
            parsed_xml,
            encoded_xml,
            str(trajectory["instruction"]),
            dict(trajectory["task_parameters"]),
        )
        required_action, _, target_label = _mobilegpt_action_from_runlog(
            transition,
            parsed_xml,
            task_parameters=dict(trajectory["task_parameters"]),
            selected_subtask=selected_subtask,
            generalize_action=None,
        )
        teacher_actions.append(
            {
                "source_step_index": int(transition.step_index),
                "source_action_type": _action_type(transition.action),
                "required_action": required_action,
                "target_label": target_label,
            }
        )
    return teacher_actions


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


class _OfficialMobileGPTRunLogSocket:
    """Client-side transport for the upstream MobileGPT Server protocol."""

    def __init__(
        self,
        frames: list[bytes],
        final_screen: tuple[bytes, bytes],
        *,
        max_final_cycles: int = _DEFAULT_AUTHORING_MAX_FINAL_CYCLES,
    ) -> None:
        self._buffer = bytearray(b"".join(frames))
        self._final_screen = final_screen
        self._final_cycles = 0
        self._max_final_cycles = max(1, int(max_final_cycles))
        self.task_finished = False
        self.sent: list[bytes] = []

    def _append_final_screen(self) -> None:
        if self.task_finished:
            return
        if self._final_cycles >= self._max_final_cycles:
            raise MobileGPTConversionError(
                "official_authoring_protocol_step_limit",
                max_final_cycles=self._max_final_cycles,
            )
        screenshot, xml = self._final_screen
        self._buffer.extend(b"S" + str(len(screenshot)).encode("ascii") + b"\n")
        self._buffer.extend(screenshot)
        self._buffer.extend(b"X" + str(len(xml)).encode("ascii") + b"\n")
        self._buffer.extend(xml)
        self._final_cycles += 1

    def recv(self, size: int) -> bytes:
        if self.task_finished:
            return b""
        if not self._buffer:
            self._append_final_screen()
        if not self._buffer:
            return b""
        result = bytes(self._buffer[:size])
        del self._buffer[:size]
        return result

    def send(self, data: bytes) -> int:
        payload = bytes(data)
        self.sent.append(payload)
        if b"$$$$$" in payload:
            self.task_finished = True
        return len(payload)

    def close(self) -> None:
        return None

    def action_messages(self) -> list[dict[str, Any]]:
        """Return JSON actions emitted by the unmodified official server loop."""

        actions: list[dict[str, Any]] = []
        for payload in self.sent:
            text = payload.decode("utf-8", errors="replace").strip()
            if not text.startswith("{"):
                continue
            try:
                action = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(action, dict) and str(action.get("name") or "").strip():
                actions.append(action)
        return actions


def _official_protocol_text(kind: bytes, value: str) -> bytes:
    return kind + value.encode("utf-8") + b"\n"


def _official_protocol_screen(kind: bytes, payload: bytes) -> bytes:
    return kind + str(len(payload)).encode("ascii") + b"\n" + payload


def _mobilegpt_action_projection(action: Any) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {}
    name = str(action.get("name") or "").strip()
    parameters = action.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {}
    projected: dict[str, Any] = {}
    for key in ("index", "direction", "input_text", "message", "number"):
        if key not in parameters:
            continue
        value = parameters[key]
        if key == "index":
            projected[key] = str(value)
        elif key == "direction":
            projected[key] = str(value).strip().lower()
        elif key == "number":
            try:
                projected[key] = int(value)
            except (TypeError, ValueError):
                projected[key] = value
        else:
            projected[key] = str(value)
    return {"name": name, "parameters": projected}


def _mobilegpt_actions_match(expected: Any, actual: Any) -> bool:
    """Compare the source-required action with the official wire action.

    MobileGPT adds execution-only parameters that AndroidWorld does not
    record, notably the scroll container index.  Those additions are valid;
    changing any parameter present in the source action is not.
    """

    expected_action = _mobilegpt_action_projection(expected)
    actual_action = _mobilegpt_action_projection(actual)
    if expected_action.get("name") != actual_action.get("name"):
        return False
    expected_parameters = expected_action.get("parameters")
    actual_parameters = actual_action.get("parameters")
    if not isinstance(expected_parameters, dict) or not isinstance(
        actual_parameters, dict
    ):
        return expected_parameters == actual_parameters
    return all(
        actual_parameters.get(key) == value
        for key, value in expected_parameters.items()
    )


def _official_prompt_kind(messages: Any) -> str:
    if not isinstance(messages, list):
        return "unknown"
    text = "\n".join(
        str(message.get("content") or "")
        for message in messages
        if isinstance(message, dict)
    )
    if "list out high-level functions" in text:
        return "explore"
    if "specific subtask within their final goal" in text:
        return "derive"
    if "list of actions available on the current mobile screen" in text:
        return "select"
    if "List of commands to summarize" in text:
        return "action_summarize"
    if "check if it matches any of the known APIs" in text:
        return "task"
    return "unknown"


def _drop_null_official_optional_fields(value: dict[str, Any]) -> dict[str, Any]:
    """Remove JSON null for optional fields that upstream tests by key presence."""

    if value.get("new_action", object()) is None:
        value = dict(value)
        value.pop("new_action", None)
    return value


def _append_official_teacher_prompt(
    messages: Any,
    *,
    teacher_action: dict[str, Any] | None,
) -> Any:
    if not isinstance(messages, list):
        return messages
    adapted = [dict(message) if isinstance(message, dict) else message for message in messages]
    teacher_payload = (
        {
            "terminal": False,
            "source_step_index": teacher_action["source_step_index"],
            "source_action_type": teacher_action["source_action_type"],
            "required_action": teacher_action["required_action"],
            "target_label": teacher_action["target_label"],
        }
        if teacher_action is not None
        else {"terminal": True, "required_action": {"name": "finish", "parameters": {}}}
    )
    guidance = (
        "\n\n<authoritative_runlog_teacher>\n"
        "This is offline memory authoring from a validator-successful RunLog. "
        "The demonstrated action is authoritative evidence, not a suggestion. "
        "Select or create the semantic subtask needed to preserve it, and when "
        "deriving the low-level action output exactly required_action. Do not "
        "substitute another UI, route, index, action, or an early finish. When "
        "terminal is true, finish and emit no additional device action.\n"
        "MOBILEGPT_AUTHORITATIVE_RUNLOG_STEP="
        + json.dumps(teacher_payload, ensure_ascii=False, sort_keys=True)
        + "\n</authoritative_runlog_teacher>"
    )
    for index in range(len(adapted) - 1, -1, -1):
        message = adapted[index]
        if isinstance(message, dict) and message.get("role") == "user":
            message["content"] = str(message.get("content") or "") + guidance
            return adapted
    adapted.append({"role": "user", "content": guidance.lstrip()})
    return adapted


def _align_official_actions_to_teacher(
    teacher_actions: Sequence[dict[str, Any]],
    official_actions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    actual_index = 0
    for teacher in teacher_actions:
        expected = _mobilegpt_action_projection(teacher.get("required_action"))
        while (
            actual_index < len(official_actions)
            and _mobilegpt_action_projection(official_actions[actual_index]).get("name")
            == "speak"
            and expected.get("name") != "speak"
        ):
            actual_index += 1
        actual = (
            _mobilegpt_action_projection(official_actions[actual_index])
            if actual_index < len(official_actions)
            else {}
        )
        matched = _mobilegpt_actions_match(expected, actual)
        rows.append(
            {
                "source_step_index": int(teacher["source_step_index"]),
                "source_action_type": str(teacher["source_action_type"]),
                "expected_action": expected,
                "actual_action": actual,
                "target_label": str(teacher.get("target_label") or ""),
                "matched": matched,
                "consumed_transitions": 1 if matched else 0,
            }
        )
        if actual_index < len(official_actions):
            actual_index += 1
    remaining = [
        _mobilegpt_action_projection(action)
        for action in official_actions[actual_index:]
        if _mobilegpt_action_projection(action).get("name") != "speak"
    ]
    if remaining and rows:
        rows[-1]["unexpected_actions"] = remaining
        rows[-1]["matched"] = False
        rows[-1]["consumed_transitions"] = 0
    return rows


def _official_xml_input(raw_xml: str) -> str:
    """Normalize AndroidWorld XML to the official parser's node-index input."""

    try:
        root = ET.fromstring(raw_xml)
    except ET.ParseError as error:
        raise MobileGPTConversionError(
            "official_authoring_xml_invalid", error=str(error)
        ) from error
    for index, node in enumerate(root.iter()):
        node.set("index", str(index))
    return ET.tostring(root, encoding="unicode")


def _run_official_mobilegpt_authoring(
    *,
    trajectory: dict[str, Any],
    mobilegpt_root: Path,
    memory_root: Path,
    stats: Path,
    audit: Path,
    model: str,
    embedding_model: str,
    embedding_provider: Callable[[str], Sequence[float]] | None,
    semantic_query_provider: Callable[..., Any] | None,
) -> dict[str, Any]:
    """Author one memory through upstream Server/Agent/Memory code.

    Upstream MobileGPT exposes learning through its socket server rather than
    a RunLog importer.  This adapter only feeds the published L/I/S/X client
    protocol.  TaskAgent, ExploreAgent, SelectAgent, DeriveAgent, XML parsing,
    action generalization, and Memory.save_task remain upstream code.
    """

    server_source = mobilegpt_root / "Server"
    if not server_source.is_dir():
        raise FileNotFoundError(f"mobilegpt_server_root_missing:{server_source}")
    if memory_root.exists():
        raise FileExistsError(f"mobilegpt_memory_exists:{memory_root}")
    memory_root.mkdir(parents=True)
    stats.parent.mkdir(parents=True, exist_ok=True)
    audit.parent.mkdir(parents=True, exist_ok=True)

    transitions = list(trajectory.get("transitions") or [])
    teacher_actions = _teacher_actions_for_trajectory(
        trajectory,
        mobilegpt_root=mobilegpt_root,
    )
    final_observation = trajectory.get("terminal_observation") or {}
    final_xml = str(trajectory.get("terminal_forest") or "").strip()
    if not final_xml and transitions:
        final_xml = str(transitions[-1].forest).strip()
    if not final_xml:
        raise MobileGPTConversionError("official_authoring_final_xml_missing")
    try:
        ET.fromstring(final_xml)
    except ET.ParseError as error:
        raise MobileGPTConversionError(
            "official_authoring_final_xml_invalid", error=str(error)
        ) from error

    def screenshot_for(observation: dict[str, Any]) -> bytes:
        pixels = observation.get("pixels")
        if isinstance(pixels, dict):
            path = Path(str(pixels.get("path") or "")).expanduser()
            if path.is_file():
                return path.read_bytes()
        return b""

    frames = [
        _official_protocol_text(b"L", str(trajectory["target_package"])),
        _official_protocol_text(b"I", str(trajectory["instruction"])),
    ]
    for transition in transitions:
        frames.append(
            _official_protocol_screen(b"S", screenshot_for(dict(transition.observation)))
        )
        frames.append(
            _official_protocol_screen(
                b"X", _official_xml_input(str(transition.forest)).encode("utf-8")
            )
        )
    final_screen = (
        screenshot_for(dict(final_observation)),
        _official_xml_input(final_xml).encode("utf-8"),
    )

    max_chat_calls = max(
        1,
        int(
            os.getenv(
                "MOBILEGPT_AUTHORING_MAX_CHAT_CALLS",
                str(_DEFAULT_AUTHORING_MAX_CHAT_CALLS),
            )
        ),
    )
    max_final_cycles = max(
        1,
        int(
            os.getenv(
                "MOBILEGPT_AUTHORING_MAX_FINAL_CYCLES",
                str(_DEFAULT_AUTHORING_MAX_FINAL_CYCLES),
            )
        ),
    )

    with tempfile.TemporaryDirectory(prefix="mobilegpt-official-authoring-") as temp:
        workspace = Path(temp)
        server_root = workspace / "Server"
        shutil.copytree(server_source, server_root)
        # GLM-4.6V exposes reasoning separately.  The upstream OpenAI helper
        # otherwise sometimes returns an empty ``content`` field for the long
        # Explore prompt.  This temporary provider-only patch disables that
        # channel; it does not change any MobileGPT prompt or agent logic.
        utils_path = server_root / "utils" / "utils.py"
        utils_source = utils_path.read_text(encoding="utf-8")
        utils_source_updated = utils_source.replace(
            "        max_tokens=900,",
            "        max_tokens=int(os.getenv(\"MOBILEGPT_MAX_TOKENS\", \"1800\")),",
            1,
        ).replace(
            "        presence_penalty=0\n    )",
            "        presence_penalty=0,\n"
            "        timeout=float(os.getenv(\"MOBILEGPT_REQUEST_TIMEOUT_SEC\", \"60\")),\n"
            "        extra_body={\"thinking\": {\"type\": \"disabled\"}}\n"
            "    )",
            1,
        )
        if utils_source_updated == utils_source:
            raise MobileGPTConversionError("official_provider_model_adapter_anchor_missing")
        utils_path.write_text(utils_source_updated, encoding="utf-8")
        environment: dict[str, str | None] = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL"),
            "MOBILEGPT_EMBEDDING_MODEL": embedding_model,
            "MOBILEGPT_TARGET_PACKAGE": str(trajectory["target_package"]),
            "MOBILEGPT_TARGET_APP": str(trajectory["target_app"]),
            "MOBILEGPT_TARGET_TASK_NAME": str(trajectory["task_name"]),
            "MOBILEGPT_STATS_JSONL": str(stats),
        }
        for name in (
            "TASK_AGENT_GPT_VERSION",
            "APP_AGENT_GPT_VERSION",
            "SELECT_AGENT_HISTORY_GPT_VERSION",
            "EXPLORE_AGENT_GPT_VERSION",
            "SELECT_AGENT_GPT_VERSION",
            "DERIVE_AGENT_GPT_VERSION",
            "PARAMETER_FILLER_AGENT_GPT_VERSION",
            "ACTION_SUMMARIZE_AGENT_GPT_VERSION",
            "SUBTASK_MERGE_AGENT_GPT_VERSION",
            "vision_model",
        ):
            environment[name] = model

        with _temporary_environment(environment), _working_directory(workspace):
            # The official client sends the installed package list.  Seed only
            # the package discovered from open_app so AppAgent can resolve it
            # without a network lookup to Google Play; no task/memory rows are
            # pre-created here.
            official_memory_dir = workspace / "memory"
            official_memory_dir.mkdir(parents=True, exist_ok=True)
            with (official_memory_dir / "apps.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["app_name", "package_name", "description", "embedding"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "app_name": str(trajectory["target_app"] or trajectory["target_package"]),
                        "package_name": str(trajectory["target_package"]),
                        "description": "",
                        "embedding": "",
                    }
                )
            if str(server_root) not in sys.path:
                sys.path.insert(0, str(server_root))
            import importlib

            prefixes = ("agents", "memory", "screenParser", "utils")
            for module_name in list(sys.modules):
                if module_name == "server" or module_name.startswith(
                    tuple(prefix + "." for prefix in prefixes)
                ):
                    sys.modules.pop(module_name, None)
            server_module = importlib.import_module("server")
            task_agent_module = importlib.import_module("agents.task_agent")
            app_agent_module = importlib.import_module("agents.app_agent")
            utils_module = importlib.import_module("utils.utils")
            memory_module = importlib.import_module("memory.memory_manager")

            original_query = utils_module.query
            original_embedding = utils_module.get_openai_embedding
            original_parse_json = utils_module.__dict__.get("__parse_json")

            def _escape_embedded_json_quotes(candidate: str) -> str:
                """Escape quotes embedded in JSON-looking string values.

                This is deliberately a lexical repair at the official parser
                boundary.  A quote inside an unfinished bracketed value is
                content; a quote followed by a JSON delimiter closes the
                surrounding string.  The resulting document is still parsed
                by the official ``json.loads`` path below.
                """
                output: list[str] = []
                in_string = False
                escaped = False
                embedded_depth = 0
                for index, char in enumerate(candidate):
                    if not in_string:
                        output.append(char)
                        if char == '"':
                            in_string = True
                        continue
                    if escaped:
                        output.append(char)
                        escaped = False
                        continue
                    if char == "\\":
                        output.append(char)
                        escaped = True
                        continue
                    if char == '"':
                        lookahead = index + 1
                        while lookahead < len(candidate) and candidate[lookahead].isspace():
                            lookahead += 1
                        next_char = candidate[lookahead] if lookahead < len(candidate) else ""
                        if (
                            embedded_depth > 0
                            or (next_char not in ",:]}" and next_char != "")
                        ):
                            output.extend(("\\", char))
                            continue
                        output.append(char)
                        in_string = False
                        embedded_depth = 0
                        continue
                    output.append(char)
                    if char in "[{":
                        embedded_depth += 1
                    elif char in "]}" and embedded_depth > 0:
                        embedded_depth -= 1
                return "".join(output)

            def _official_json_parser(value: str, *, is_list: bool = False) -> str | None:
                """Keep the official parser, repairing transport-only JSON damage.

                GLM sometimes emits one object in an otherwise valid JSON list
                without the opening ``{``.  The official query function parses
                before returning, so the repair must happen at that transport
                boundary.  It can also leave quotes inside a stringified option
                list unescaped (for example ``["Just once", "Always"]``).
                Repair only those JSON delimiters; no action, parameter, or
                ordering is inferred here.
                """
                if original_parse_json is None:
                    return None
                candidate = original_parse_json(value, is_list=is_list)
                if candidate is None:
                    return None
                try:
                    json.loads(candidate)
                except json.JSONDecodeError:
                    repaired = re.sub(
                        r"(,\s*)(\"name\"\s*:)",
                        r"\1{\2",
                        candidate,
                        count=1,
                    )
                    if repaired != candidate:
                        json.loads(repaired)
                        _write_event(stats, {
                            "event": "chat_json_syntax_repair",
                            "repair": "missing_list_item_open_brace",
                        })
                        return repaired
                    repaired = _escape_embedded_json_quotes(candidate)
                    if repaired != candidate:
                        json.loads(repaired)
                        _write_event(stats, {
                            "event": "chat_json_syntax_repair",
                            "repair": "unescaped_quotes_in_stringified_value",
                        })
                        return repaired
                    raise
                return candidate

            if original_parse_json is not None:
                utils_module.__dict__["__parse_json"] = _official_json_parser

            def _official_schema_adapter(
                value: Any,
                *,
                list_items_require_name: bool = False,
            ) -> Any:
                """Adapt only JSON scalar containers expected by upstream code.

                The upstream prompts and agents remain authoritative.  Some
                GLM responses serialize an empty ``parameters`` object as the
                string ``\"{}\"`` even though the official Memory code indexes
                it as a mapping.  Decode that transport-level representation;
                do not infer, rewrite, or select any action.
                """
                if isinstance(value, dict):
                    adapted: dict[str, Any] = {}
                    for key, item in value.items():
                        if (
                            key == "parameters"
                            and isinstance(item, str)
                            and item.strip().startswith(("{", "["))
                        ):
                            adapted[key] = _official_schema_adapter(json.loads(item))
                        elif key == "completion_rate" and isinstance(item, str):
                            # GLM-4.6V occasionally verbalizes this official
                            # telemetry field. The official DeriveAgent only
                            # uses it as a numeric progress hint; preserving
                            # the action and setting a neutral numeric value
                            # keeps the official action/memory path intact.
                            try:
                                adapted[key] = float(item.strip().rstrip("%"))
                            except ValueError:
                                adapted[key] = 0
                        else:
                            adapted[key] = _official_schema_adapter(item)
                    if "action" in adapted and "completion_rate" not in adapted:
                        adapted["completion_rate"] = 0
                        _write_event(stats, {
                            "event": "chat_schema_repair",
                            "repair": "default_missing_completion_rate",
                        })
                    return _drop_null_official_optional_fields(adapted)
                if isinstance(value, list):
                    adapted_items = [_official_schema_adapter(item) for item in value]
                    if list_items_require_name:
                        valid_items = [
                            item
                            for item in adapted_items
                            if isinstance(item, dict)
                            and str(item.get("name") or "").strip()
                        ]
                        if len(valid_items) != len(adapted_items):
                            _write_event(stats, {
                                "event": "chat_schema_repair",
                                "repair": "drop_explore_item_without_name",
                                "dropped_count": len(adapted_items) - len(valid_items),
                            })
                        return valid_items
                    return adapted_items
                return value

            chat_call_count = 0
            teacher_cursor = 0
            teacher_retry_limit = max(
                0,
                int(os.getenv("MOBILEGPT_AUTHORING_TEACHER_RETRIES", "2")),
            )

            def call_once(
                messages: Any,
                *,
                model_name: str,
                is_list: bool,
                kwargs: dict[str, Any],
            ) -> Any:
                nonlocal chat_call_count
                if chat_call_count >= max_chat_calls:
                    _write_event(
                        stats,
                        {
                            "event": "chat_call_limit_exceeded",
                            "model": model_name,
                            "chat_calls": chat_call_count,
                            "max_chat_calls": max_chat_calls,
                        },
                    )
                    raise MobileGPTConversionError(
                        "official_authoring_chat_call_limit",
                        chat_calls=chat_call_count,
                        max_chat_calls=max_chat_calls,
                    )
                chat_call_count += 1
                query_provider = semantic_query_provider or original_query
                try:
                    return query_provider(
                        messages,
                        model=model_name,
                        is_list=is_list,
                        **kwargs,
                    )
                except TypeError as error:
                    # Some pinned official checkouts have an extra telemetry
                    # keyword while the upstream public checkout does not.
                    # This only adapts the call signature; the prompt and
                    # response handling remain official.
                    if not kwargs or "unexpected keyword argument" not in str(error):
                        raise
                    if chat_call_count >= max_chat_calls:
                        raise MobileGPTConversionError(
                            "official_authoring_chat_call_limit",
                            chat_calls=chat_call_count,
                            max_chat_calls=max_chat_calls,
                        ) from error
                    chat_call_count += 1
                    return query_provider(
                        messages,
                        model=model_name,
                        is_list=is_list,
                    )

            def teacher_response_matches(
                *,
                prompt_kind: str,
                result: Any,
                teacher_action: dict[str, Any] | None,
            ) -> bool:
                if prompt_kind not in {"select", "derive"}:
                    return True
                response_action = result.get("action") if isinstance(result, dict) else None
                actual = _mobilegpt_action_projection(response_action)
                if teacher_action is None:
                    return actual == {"name": "finish", "parameters": {}}
                expected = _mobilegpt_action_projection(
                    teacher_action.get("required_action")
                )
                if prompt_kind == "derive":
                    return actual == expected
                if expected.get("name") == "scroll":
                    return actual.get("name") == "scroll_screen"
                if expected.get("name") == "speak":
                    return actual.get("name") == "speak"
                return actual.get("name") not in {
                    "",
                    "finish",
                    "scroll_screen",
                    "speak",
                }

            def query_with_stats(
                messages: Any,
                model: str | None = None,
                is_list: bool = False,
                **kwargs: Any,
            ) -> Any:
                nonlocal teacher_cursor
                started = time.monotonic()
                model_name = model or str(environment["TASK_AGENT_GPT_VERSION"])
                prompt_kind = str(kwargs.get("agent_name") or "").strip().lower()
                if prompt_kind not in {
                    "task",
                    "explore",
                    "select",
                    "derive",
                    "action_summarize",
                }:
                    prompt_kind = _official_prompt_kind(messages)
                teacher_action = (
                    teacher_actions[teacher_cursor]
                    if teacher_cursor < len(teacher_actions)
                    else None
                )
                prompted_messages = (
                    _append_official_teacher_prompt(
                        messages,
                        teacher_action=teacher_action,
                    )
                    if prompt_kind in {"explore", "select", "derive"}
                    else messages
                )
                provider_kwargs = dict(kwargs)
                if semantic_query_provider is not None and prompt_kind != "unknown":
                    provider_kwargs.setdefault("agent_name", prompt_kind)
                if prompt_kind in {"explore", "select", "derive"}:
                    _write_event(
                        stats,
                        {
                            "event": "mobilegpt_teacher_prompt",
                            "agent": prompt_kind,
                            "source_step_index": (
                                teacher_action.get("source_step_index")
                                if teacher_action is not None
                                else None
                            ),
                            "terminal": teacher_action is None,
                            "required_action": (
                                teacher_action.get("required_action")
                                if teacher_action is not None
                                else {"name": "finish", "parameters": {}}
                            ),
                        },
                    )
                result = _official_schema_adapter(
                    call_once(
                        prompted_messages,
                        model_name=model_name,
                        is_list=is_list,
                        kwargs=provider_kwargs,
                    ),
                    list_items_require_name=is_list,
                )
                for retry in range(teacher_retry_limit):
                    empty_response = result is None or (
                        isinstance(result, str) and not result.strip()
                    )
                    if not empty_response and teacher_response_matches(
                        prompt_kind=prompt_kind,
                        result=result,
                        teacher_action=teacher_action,
                    ):
                        break
                    correction = {
                        "role": "user",
                        "content": (
                            "The response was empty. Return the required official "
                            "MobileGPT JSON now."
                            if empty_response
                            else
                            "Your response does not preserve the authoritative "
                            "RunLog action. Return the required action exactly; "
                            "do not choose another UI or finish early."
                        ),
                    }
                    result = _official_schema_adapter(
                        call_once(
                            [*prompted_messages, correction],
                            model_name=model_name,
                            is_list=is_list,
                            kwargs=provider_kwargs,
                        ),
                        list_items_require_name=is_list,
                    )
                    _write_event(
                        stats,
                        {
                            "event": (
                                "empty_response_retry"
                                if empty_response
                                else "mobilegpt_teacher_prompt_retry"
                            ),
                            "agent": prompt_kind,
                            "retry": retry + 1,
                            "source_step_index": (
                                teacher_action.get("source_step_index")
                                if teacher_action is not None
                                else None
                            ),
                        },
                    )
                if result is None or (
                    isinstance(result, str) and not result.strip()
                ):
                    raise MobileGPTConversionError(
                        "official_agent_empty_response",
                        model=model_name,
                        agent=prompt_kind,
                    )
                if not teacher_response_matches(
                    prompt_kind=prompt_kind,
                    result=result,
                    teacher_action=teacher_action,
                ):
                    raise MobileGPTConversionError(
                        "official_authoring_teacher_response_mismatch",
                        agent=prompt_kind,
                        source_step_index=(
                            teacher_action.get("source_step_index")
                            if teacher_action is not None
                            else None
                        ),
                        expected_action=(
                            teacher_action.get("required_action")
                            if teacher_action is not None
                            else {"name": "finish", "parameters": {}}
                        ),
                        actual_action=(
                            result.get("action")
                            if isinstance(result, dict)
                            else None
                        ),
                    )
                if teacher_action is not None:
                    expected_name = _mobilegpt_action_projection(
                        teacher_action.get("required_action")
                    ).get("name")
                    if prompt_kind == "derive" or (
                        prompt_kind == "select"
                        and expected_name in {"scroll", "speak"}
                    ):
                        _write_event(
                            stats,
                            {
                                "event": "mobilegpt_teacher_action",
                                "agent": prompt_kind,
                                "source_step_index": teacher_action[
                                    "source_step_index"
                                ],
                                "required_action": teacher_action[
                                    "required_action"
                                ],
                            },
                        )
                        teacher_cursor += 1
                _write_event(stats, {
                    "event": "chat_call",
                    "model": model_name,
                    "latency_sec": round(time.monotonic() - started, 6),
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                })
                return result

            def embedding_with_stats(text: str, model: str | None = None, **kwargs: Any) -> list[float]:
                started = time.monotonic()
                if embedding_provider is not None:
                    result = [float(value) for value in embedding_provider(text)]
                else:
                    result = [
                        float(value)
                        for value in original_embedding(text, model=embedding_model, **kwargs)
                    ]
                _write_event(stats, {
                    "event": "embedding_call",
                    "model": embedding_model,
                    "latency_sec": round(time.monotonic() - started, 6),
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                })
                if not result:
                    raise MobileGPTConversionError("official_authoring_embedding_empty")
                return result

            utils_module.query = query_with_stats
            utils_module.get_openai_embedding = embedding_with_stats
            memory_module.get_openai_embedding = embedding_with_stats
            app_agent_module.get_openai_embedding = embedding_with_stats
            for module_name, module in list(sys.modules.items()):
                if module_name.startswith("agents.") and hasattr(module, "query"):
                    setattr(module, "query", query_with_stats)

            original_get_task = task_agent_module.TaskAgent.get_task

            def get_task_with_open_app(self: Any, instruction: str) -> tuple[dict[str, Any], bool]:
                task, _ = original_get_task(self, instruction)
                if not isinstance(task, dict):
                    raise MobileGPTConversionError("official_authoring_task_invalid")
                task["name"] = str(trajectory["task_name"])
                task["app"] = str(trajectory["target_app"] or trajectory["target_package"])
                return task, True

            task_agent_module.TaskAgent.get_task = get_task_with_open_app
            protocol_socket = _OfficialMobileGPTRunLogSocket(
                frames,
                final_screen,
                max_final_cycles=max_final_cycles,
            )
            _write_event(stats, {
                "event": "task_started",
                "task_name": trajectory["task_name"],
                "mode": CONVERSION_MODE_OFFICIAL,
                "target_package": trajectory["target_package"],
            })
            try:
                server = server_module.Server(host="127.0.0.1", port=0, buffer_size=4096)
                server.handle_client(protocol_socket, ("runlog-adapter", 0))
            except BaseException as error:
                _write_event(stats, {
                    "event": "task_failed",
                    "error": type(error).__name__ + ":" + str(error),
                })
                raise MobileGPTConversionError(
                    "official_authoring_session_failed", error=str(error)
                ) from error
            finally:
                for module_name in list(sys.modules):
                    if module_name == "server" or module_name.startswith(
                        tuple(prefix + "." for prefix in prefixes)
                    ):
                        sys.modules.pop(module_name, None)
            # The upstream Server always writes relative to its official
            # ``./memory`` root.  Preserve that exact generated tree before
            # disposing the isolated checkout; no files are reconstructed or
            # translated by this adapter.
            if not protocol_socket.task_finished:
                raise MobileGPTConversionError("official_authoring_task_not_finished")
            validation_rows = _align_official_actions_to_teacher(
                teacher_actions,
                protocol_socket.action_messages(),
            )
            for row in validation_rows:
                _write_event(
                    stats,
                    {"event": "mobilegpt_conversion_action_mapped", **row},
                )
            if (
                len(validation_rows) != len(teacher_actions)
                or any(row.get("matched") is not True for row in validation_rows)
            ):
                raise MobileGPTConversionError(
                    "official_authoring_teacher_alignment_failed",
                    expected_action_count=len(teacher_actions),
                    actual_actions=protocol_socket.action_messages(),
                    validation_rows=validation_rows,
                )
            shutil.copytree(official_memory_dir, memory_root, dirs_exist_ok=True)
            _write_event(stats, {
                "event": "official_memory_tree_copied",
                "file_count": sum(
                    1 for item in memory_root.rglob("*")
                    if item.is_file() and "__pycache__" not in item.parts
                ),
            })

    if not protocol_socket.task_finished:
        raise MobileGPTConversionError("official_authoring_task_not_finished")
    _write_event(stats, {
        "event": "task_finished",
        "task_name": trajectory["task_name"],
        "mode": CONVERSION_MODE_OFFICIAL,
        "official_action_messages": len(protocol_socket.sent),
    })
    validation = validate_mobilegpt_memory(memory_root)
    audit_payload = {
        "schema_version": MOBILEGPT_AUDIT_SCHEMA,
        "conversion_mode": CONVERSION_MODE_OFFICIAL,
        "task_name": trajectory["task_name"],
        "source_run_log": trajectory["source_run_log"],
        "target_package": trajectory["target_package"],
        # The upstream prompts remain intact; the successful RunLog action is
        # appended as authoring evidence at their user-message boundary.
        "original_mobilegpt_prompts": True,
        "official_prompt_extension": True,
        "teacher_prompt_used": True,
        "teacher_action_alignment_complete": True,
        "explore_agent_used": True,
        "select_agent_used": True,
        "derive_agent_fallback_allowed": False,
        "derive_agent_fallback_count": 0,
        "source_example_fallback_count": 0,
        "generalize_action_used": True,
        "direct_subtasks_from_runlog": False,
        "source_reader_coverage_validation": False,
        "actions_supplied_to_mobilegpt": False,
        "source_transitions_supplied": True,
        "source_success_boundary_supplied": True,
        "source_success_boundary": trajectory["source_success_boundary"],
        "transition_count": len(transitions),
        "validated_transition_count": len(transitions),
        "validation_rows": validation_rows,
        "official_reader_validation": {
            "loadable": True,
            "task_path_pages": validation["page_count"],
            "page_count": validation["page_count"],
            "action_row_count": validation["action_count"],
            "official_action_messages": len(protocol_socket.action_messages()),
            "teacher_aligned_action_count": len(validation_rows),
        },
        "complete": True,
    }
    audit.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "task": {"name": trajectory["task_name"], "app": trajectory["target_app"]},
        "memory_root": str(memory_root),
        "stats_path": str(stats),
        "audit_path": str(audit),
        "transition_count": len(transitions),
        "validated_transition_count": len(transitions),
        "target_package": trajectory["target_package"],
        "target_app": trajectory["target_app"],
        "embedding_model": embedding_model,
        "source_success_boundary": trajectory["source_success_boundary"],
        "official_reader_validation": audit_payload["official_reader_validation"],
        "wall_sec": 0.0,
        "memory_validation": validation,
    }


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
    teacher_prompt_used = False
    stats = Path(stats_path).expanduser().resolve()
    if stats.is_file():
        for raw_line in stats.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(event, dict)
                and event.get("event") == "mobilegpt_teacher_prompt"
            ):
                teacher_prompt_used = True
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
        "teacher_prompt_used": teacher_prompt_used,
        "teacher_action_alignment_complete": False,
        "actions_supplied_to_mobilegpt": conversion_mode == CONVERSION_MODE_DIRECT,
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
    embedding_model: str = MOBILEGPT_EMBEDDING_MODEL,
    target_package: str = "",
    target_app: str = "",
    embedding_provider: Callable[[str], Sequence[float]] | None = None,
    semantic_query_provider: Callable[..., Any] | None = None,
    conversion_mode: str = CONVERSION_MODE_DIRECT,
) -> dict[str, Any]:
    """Compile one verified RunLog through MobileGPT's official memory APIs."""

    if conversion_mode == CONVERSION_MODE_OFFICIAL:
        trajectory = _load_runlog_trajectory(
            source_run_log,
            target_package=target_package,
            target_app=target_app,
        )
        return _run_official_mobilegpt_authoring(
            trajectory=trajectory,
            mobilegpt_root=Path(mobilegpt_root).expanduser().resolve(),
            memory_root=Path(memory_root).expanduser().resolve(),
            stats=Path(stats_path).expanduser().resolve(),
            audit=Path(audit_path).expanduser().resolve(),
            model=model,
            embedding_model=str(embedding_model or MOBILEGPT_EMBEDDING_MODEL),
            embedding_provider=embedding_provider,
            semantic_query_provider=semantic_query_provider,
        )

    if conversion_mode != CONVERSION_MODE_DIRECT:
        raise ValueError(f"mobilegpt_conversion_mode_invalid:{conversion_mode}")
    normalized_embedding_model = (
        str(embedding_model or "").strip() or MOBILEGPT_EMBEDDING_MODEL
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
    # Upstream Memory deliberately stores data below ``./memory``. Keep its
    # implementation unchanged, but run it from the prepared bundle's parent
    # so ``./memory`` resolves to the requested output instead of mutating the
    # canonical MobileGPT checkout.
    with _temporary_environment(environment), _working_directory(memory.parent):
        from memory.memory_manager import Memory

        official_generalize_action = None
        if not launch_only:
            try:
                from utils.action_utils import generalize_action as official_generalize_action
            except ImportError as error:
                raise MobileGPTConversionError(
                    "mobilegpt_generalize_action_unavailable"
                ) from error
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
        xml_root = log_root / "xmls"
        xml_root.mkdir(parents=True, exist_ok=True)
        pages_by_identity: dict[str, dict[str, Any]] = {}
        source_screen_artifacts: dict[int, dict[str, Path]] = {}
        task_path: dict[str, list[str]] = {}
        official_task_path_records: list[dict[str, Any]] = []
        for screen_index, transition in enumerate(encoded_transitions):
            raw_xml = str(transition.forest)
            raw_path = xml_root / f"{screen_index}.xml"
            raw_path.write_text(raw_xml, encoding="utf-8")
            parsed_xml, hierarchy_xml, encoded_xml = encode_xml(
                raw_xml,
                mobilegpt_root=mobilegpt_root,
            )
            (xml_root / f"{screen_index}_parsed.xml").write_text(
                parsed_xml,
                encoding="utf-8",
            )
            (xml_root / f"{screen_index}_hierarchy_parsed.xml").write_text(
                hierarchy_xml,
                encoding="utf-8",
            )
            (xml_root / f"{screen_index}_encoded.xml").write_text(
                encoded_xml,
                encoding="utf-8",
            )
            (xml_root / f"{screen_index}_pretty.xml").write_text(
                xml.dom.minidom.parseString(encoded_xml).toprettyxml(),
                encoding="utf-8",
            )
            identity = _screen_identity(hierarchy_xml)
            page = pages_by_identity.get(identity)
            if page is None:
                page_index = len(pages_by_identity)
                artifacts = {
                    "raw.xml": raw_path,
                    "html.xml": xml_root / f"{screen_index}_encoded.xml",
                    "hierarchy.xml": xml_root
                    / f"{screen_index}_hierarchy_parsed.xml",
                    "parsed.xml": xml_root / f"{screen_index}_parsed.xml",
                    "pretty.xml": xml_root / f"{screen_index}_pretty.xml",
                }
                for name, source in artifacts.items():
                    if not source.is_file():
                        raise MobileGPTConversionError(
                            "source_screen_artifact_missing",
                            step_index=transition.step_index,
                            artifact=name,
                        )
                source_screen_artifacts[screen_index] = artifacts
                pixels = transition.observation.get("pixels")
                if isinstance(pixels, dict):
                    screenshot = Path(str(pixels.get("path") or "")).expanduser()
                    if screenshot.is_file():
                        source_screen_artifacts[screen_index]["screenshot.jpg"] = screenshot
                page = {
                    "index": page_index,
                    "parsed_xml": parsed_xml,
                    "hierarchy_xml": hierarchy_xml,
                    "encoded_xml": encoded_xml,
                    "available_subtasks": [],
                    "trigger_uis": {},
                    "trigger_indexes": set(),
                    "extra_uis": [],
                    "subtasks": {},
                    "screen_num": screen_index,
                }
                pages_by_identity[identity] = page
            page_index = int(page["index"])
            if launch_only:
                finish_subtask = {
                    "name": "finish",
                    "description": "Signal that the task is complete.",
                    "parameters": {},
                }
                page["available_subtasks"].append(finish_subtask)
                page["subtasks"]["finish"] = {
                    "metadata": finish_subtask,
                    "example": {},
                }
                task_path[str(page_index)] = ["finish"]
                official_task_path_records.append(
                    {
                        "page_index": page_index,
                        "subtask_name": "finish",
                        "subtask": {"name": "finish", "parameters": {}},
                        "actions": [],
                    }
                )
                continue
            subtask, selected_subtask, example = _direct_subtask_from_runlog(
                transition,
                parsed_xml,
                encoded_xml,
                trajectory["instruction"],
                trajectory["task_parameters"],
            )
            page["available_subtasks"].append(subtask)
            action_type = _action_type(transition.action)
            if action_type in {"click", "double_tap", "input_text", "long_press"}:
                target = _target_element(
                    transition.action,
                    parsed_xml,
                    step_index=transition.step_index,
                    source_forest=transition.forest,
                    next_forest=transition.next_forest,
                )
                trigger_index = target.get("index")
                if trigger_index is not None:
                    from utils.parsing_utils import get_trigger_ui_attributes

                    trigger_index = int(trigger_index)
                    page["trigger_indexes"].add(trigger_index)
                    page["trigger_uis"].update(
                        get_trigger_ui_attributes(
                            {subtask["name"]: [trigger_index]},
                            _mobilegpt_ui_match_xml(parsed_xml),
                        )
                    )
            converted, bindings, label = _mobilegpt_action_from_runlog(
                transition,
                parsed_xml,
                task_parameters=trajectory["task_parameters"],
                selected_subtask=selected_subtask,
                generalize_action=official_generalize_action,
            )
            del label
            raw_converted, _, _ = _mobilegpt_action_from_runlog(
                transition,
                parsed_xml,
                task_parameters=trajectory["task_parameters"],
                selected_subtask=selected_subtask,
                generalize_action=None,
            )
            # Memory.save_task owns the official action generalization pass.
            # The OOB execution representation may carry a structured
            # ``attrib`` selector for an anonymous input, but upstream
            # generalize_action_to_arguments expects every non-index value in
            # its raw input to be a string.  Supply the original index/text
            # action here and let MobileGPT derive its own selector once.
            raw_parameters = raw_converted.get("parameters")
            if isinstance(raw_parameters, dict):
                raw_parameters.pop("attrib", None)
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
            task_path.setdefault(str(page_index), []).append(subtask["name"])
            official_task_path_records.append(
                {
                    "page_index": page_index,
                    "subtask_name": subtask["name"],
                    "subtask": selected_subtask,
                    "actions": [
                        {
                            "page_index": page_index,
                            "action": raw_converted,
                            "screen": f"<hierarchy>{encoded_xml}</hierarchy>",
                            "example": action_example,
                        },
                        {
                            "page_index": page_index,
                            "action": {"name": "finish", "parameters": {}},
                            "screen": f"<hierarchy>{encoded_xml}</hierarchy>",
                            "example": {},
                        },
                    ],
                }
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
                "reason": "runlog_direct_compiled",
                "derive_fallback_used": False,
                "consumed_transitions": 1,
            }
            audit_rows.append(row)
            _write_event(stats, {"event": "mobilegpt_conversion_action_mapped", **row})

        if not launch_only:
            final_page_index = str(audit_rows[-1]["memory_page_index"])
            task_path[final_page_index].append("finish")
            official_task_path_records.append(
                {
                    "page_index": int(final_page_index),
                    "subtask_name": "finish",
                    "subtask": {"name": "finish", "parameters": {}},
                    "actions": [],
                }
            )
        # The root task index is OmniFlow's bundle metadata.  Everything under
        # the MobileGPT app directory is written by MobileGPT's own Memory and
        # PageManager APIs below.
        _write_csv(
            memory / "tasks.csv",
            ("name", "description", "parameters", "app"),
            [{**task, "parameters": _json_text(task["parameters"])}],
        )
        from memory import memory_manager as official_memory_module
        from utils import parsing_utils as official_parsing_utils
        from utils.parsing_utils import get_extra_ui_attributes

        def official_save_screen_info(
            _app_name: str,
            _task_name: str,
            destination: str,
            screen_num: int | None = None,
        ) -> None:
            """Adapt RunLog artifacts to MobileGPT's screen-copy hook."""

            if screen_num is None or int(screen_num) not in source_screen_artifacts:
                raise MobileGPTConversionError(
                    "mobilegpt_source_screen_artifact_missing",
                    screen_num=screen_num,
                )
            destination_root = Path(destination)
            destination_root.mkdir(parents=True, exist_ok=True)
            artifacts = source_screen_artifacts[int(screen_num)]
            for name in ("raw.xml", "html.xml", "hierarchy.xml", "parsed.xml", "pretty.xml"):
                source = artifacts.get(name)
                if source is None or not source.is_file():
                    raise MobileGPTConversionError(
                        "mobilegpt_source_screen_artifact_missing",
                        screen_num=screen_num,
                        artifact=name,
                    )
                shutil.copy2(source, destination_root / name)
            screenshot = artifacts.get("screenshot.jpg")
            if screenshot is not None and screenshot.is_file():
                shutil.copy2(screenshot, destination_root / "screenshot.jpg")

        original_save_screen_info = official_parsing_utils.save_screen_info
        original_official_embedding = official_memory_module.get_openai_embedding

        def official_embedding(screen: str) -> list[float]:
            embedding_started = time.monotonic()
            if embedding_provider is not None:
                result = [float(value) for value in embedding_provider(screen)]
            else:
                result = [
                    float(value)
                    for value in original_official_embedding(
                        screen,
                        model=normalized_embedding_model,
                    )
                ]
            _write_event(
                stats,
                {
                    "event": "embedding_call",
                    "model": normalized_embedding_model,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "latency_sec": round(
                        time.monotonic() - embedding_started,
                        6,
                    ),
                },
            )
            if not result:
                raise MobileGPTConversionError("mobilegpt_page_embedding_empty")
            return result

        official_parsing_utils.save_screen_info = official_save_screen_info
        official_memory_module.get_openai_embedding = official_embedding

        official_memory = Memory(app_name, trajectory["instruction"], task_name)
        try:
            for page in pages_by_identity.values():
                page_index = int(page["index"])
                available_subtasks = list(page["available_subtasks"])
                page["extra_uis"] = get_extra_ui_attributes(
                    sorted(page["trigger_indexes"]),
                    _mobilegpt_ui_match_xml(page["parsed_xml"]),
                )
                page_action_names = {
                    str(row["memory_subtask_name"])
                    for row in audit_rows
                    if int(row["memory_page_index"]) == page_index
                    and row["source_action_type"]
                    in {"click", "double_tap", "input_text", "long_press"}
                }
                missing_trigger_uis = sorted(
                    name
                    for name in page_action_names
                    if not page["trigger_uis"].get(name)
                )
                if missing_trigger_uis:
                    raise MobileGPTConversionError(
                        "mobilegpt_trigger_ui_attributes_missing",
                        page_index=page_index,
                        subtask_names=missing_trigger_uis,
                    )
                created_page_index = official_memory.add_node(
                    available_subtasks,
                    page["trigger_uis"],
                    page["extra_uis"],
                    page["parsed_xml"],
                    int(page["screen_num"]),
                )
                if int(created_page_index) != page_index:
                    raise MobileGPTConversionError(
                        "mobilegpt_official_page_index_mismatch",
                        expected=page_index,
                        actual=created_page_index,
                    )
                official_memory.add_hierarchy_xml(page["hierarchy_xml"], page_index)
                official_memory.init_page_manager(page_index)
                for selected in page["subtasks"].values():
                    official_memory.save_subtask(
                        selected["metadata"],
                        selected["example"],
                    )
            official_memory.save_task(official_task_path_records)
        finally:
            official_parsing_utils.save_screen_info = original_save_screen_info
            official_memory_module.get_openai_embedding = original_official_embedding
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
        "generalize_action_used": True,
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
