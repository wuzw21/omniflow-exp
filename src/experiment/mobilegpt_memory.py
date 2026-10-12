"""MobileGPT converted-memory sealing and target inference (split out of run_task).

Moved verbatim from ``src.experiment.run_task``; that module re-exports these names.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any
import xml.etree.ElementTree as ET
from omniflow.core.trajectory import canonicalize_run_log
from src.experiment.mobilegpt_contract import MOBILEGPT_AUDIT_SCHEMA, MOBILEGPT_LEARNING_MODE, MOBILEGPT_MEMORY_MANIFEST, MOBILEGPT_MEMORY_SCHEMA, MOBILEGPT_SOURCE_METHOD
from src.experiment.paths import resolve_path, sha256_file
from src.experiment.protocol import FORMAL_THINKING, FORMAL_MODEL_BASE_URL, SOURCE_SEED
from src.experiment.source_records import CanonicalRunLog
from src.integrations import mobilegpt_memory
from src.experiment.run_task_common import (
    REPO_ROOT,
    _coerce_float,
    _coerce_int,
    canonicalize_source_run_log,
)


def seal_mobilegpt_converted_memory(
    *,
    memory_root: str | Path,
    source_run_log: str | Path,
    source_stats: str | Path,
    trajectory_audit: str | Path,
    task_name: str,
    source_seed: int = SOURCE_SEED,
    target_package: str = "",
    target_app: str = "",
    source_wall_sec: float = 0.0,
    source_model: str = "",
    memory_schema: str = MOBILEGPT_MEMORY_SCHEMA,
) -> dict[str, Any]:
    """Seal one offline RunLog-to-MobileGPT memory database."""

    if memory_schema != MOBILEGPT_MEMORY_SCHEMA:
        raise ValueError(f"mobilegpt_source_memory_schema_invalid:{memory_schema}")
    audit_schema = MOBILEGPT_AUDIT_SCHEMA
    semantic_learning = True
    source_method = MOBILEGPT_SOURCE_METHOD
    learning_mode = MOBILEGPT_LEARNING_MODE

    memory = resolve_path(memory_root)
    bundle_root = memory.parent.resolve()
    if memory.name != "memory":
        raise ValueError("mobilegpt_virtual_memory_directory_must_be_named_memory")
    manifest_path = bundle_root / MOBILEGPT_MEMORY_MANIFEST
    if manifest_path.exists():
        raise FileExistsError(
            f"immutable_mobilegpt_memory_manifest_exists:{manifest_path}"
        )
    source_path = resolve_path(source_run_log)
    stats_path = resolve_path(source_stats)
    audit_path = resolve_path(trajectory_audit)
    source_payload = canonicalize_run_log(
        json.loads(source_path.read_text(encoding="utf-8"))
    )
    recorded_source_seed = source_payload.get("seed")
    source_validator = source_payload.get("validator")
    if (
        str(source_payload.get("task_name") or "") != str(task_name)
        or type(recorded_source_seed) is not int
        or source_payload.get("status") != "succeeded"
        or source_payload.get("success") is not True
        or not isinstance(source_validator, dict)
        or source_validator.get("official") is not True
        or source_validator.get("success") is not True
    ):
        raise ValueError("mobilegpt_virtual_memory_source_run_log_invalid")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        not isinstance(audit, dict)
        or audit.get("schema_version") != audit_schema
        or str(audit.get("task_name") or "") != str(task_name)
    ):
        raise ValueError("mobilegpt_virtual_memory_audit_invalid")
    transition_count = int(audit.get("transition_count") or 0)
    validated_count = int(audit.get("validated_transition_count") or 0)
    validation_rows = audit.get("validation_rows")
    official_reader = audit.get("official_reader_validation")
    official_learning = (
        semantic_learning
        and audit.get("official_server_finished") is True
        and audit.get("teacher_prompt_used") is False
    )
    launch_only = (
        isinstance(official_reader, dict)
        and mobilegpt_memory.is_valid_mobilegpt_launch_only_memory(
            source_payload,
            audit,
            official_reader,
        )
    )
    if official_learning:
        # A launch-only task has no replayable transition by design.  The
        # official server still has to finish and produce a loadable one-page
        # memory, but requiring transition_count > 0 would reject the valid
        # open_app + terminal-answer case.
        trajectory_complete = (
            launch_only
            or (
                transition_count > 0
                and audit.get("official_server_finished") is True
                and audit.get("source_transitions_supplied") is True
                and audit.get("source_success_boundary_supplied") is True
                and audit.get("complete") is True
            )
        )
    else:
        trajectory_complete = not (
            transition_count <= 0
            or validated_count != transition_count
            or not isinstance(validation_rows, list)
            or not validation_rows
            or any(not isinstance(row, dict) or row.get("matched") is not True for row in validation_rows)
            or (
                not semantic_learning
                and any(
                    not isinstance(row, dict)
                    or row.get("semantic_alignment") is not True
                    for row in validation_rows
                )
            )
            or sum(int(row.get("consumed_transitions") or 0) for row in validation_rows)
            != transition_count
            or (
                not semantic_learning
                and audit.get("actions_supplied_to_mobilegpt") is not True
            )
            or audit.get("source_transitions_supplied") is not True
            or audit.get("source_success_boundary_supplied") is not True
            or audit.get("complete") is not True
        )
    if not (trajectory_complete or launch_only):
        raise ValueError("mobilegpt_virtual_memory_trajectory_incomplete")
    if (
        not isinstance(official_reader, dict)
        or official_reader.get("loadable") is not True
        or int(official_reader.get("task_path_pages") or 0) <= 0
        or int(official_reader.get("page_count") or 0) <= 0
        or (
            (
                not launch_only
                and int(official_reader.get("action_row_count") or 0) <= 0
            )
            if official_learning
            else (
                not launch_only
                and int(official_reader.get("action_row_count") or 0) < transition_count
            )
        )
    ):
        raise ValueError("mobilegpt_virtual_memory_official_reader_invalid")
    from src.integrations.mobilegpt import validate_mobilegpt_memory

    memory_validation = validate_mobilegpt_memory(memory)
    inventory = mobilegpt_memory.inspect_mobilegpt_memory(memory)
    if inventory.get("task_local_memory") is not True:
        raise ValueError("mobilegpt_virtual_memory_not_task_local")
    if (
        not launch_only
        and inventory.get("virtual_source_memory_complete") is not True
    ):
        raise ValueError("mobilegpt_virtual_memory_graph_incomplete")
    if not launch_only and not inventory.get("has_recallable_subtasks"):
        raise ValueError("mobilegpt_virtual_memory_missing_recallable_subtasks")
    if not launch_only and not inventory.get("has_useful_actions"):
        raise ValueError("mobilegpt_virtual_memory_missing_useful_actions")
    stats_summary = mobilegpt_memory.summarize_mobilegpt_stats(stats_path)
    if (
        int(stats_summary.get("task_started_count") or 0) != 1
        or int(stats_summary.get("task_finished_count") or 0) != 1
    ):
        raise ValueError("mobilegpt_virtual_memory_task_lifecycle_incomplete")
    chat_attempts = [
        _coerce_int(value) for value in stats_summary.get("chat_attempts") or []
    ]
    if int(stats_summary.get("embedding_model_calls") or 0) <= 0:
        raise ValueError("mobilegpt_memory_embedding_calls_required")
    required_audit = {
        "conversion_mode": "official_mobilegpt_learning",
        "original_mobilegpt_prompts": True,
        "explore_agent_used": True,
        "select_agent_used": True,
        "derive_agent_fallback_allowed": False,
        "teacher_prompt_used": not official_learning,
        "teacher_action_alignment_complete": not official_learning,
        "actions_supplied_to_mobilegpt": False,
        "source_reader_coverage_validation": False,
        "direct_subtasks_from_runlog": False,
    }
    if official_learning:
        required_audit["official_server_finished"] = True
        required_audit["official_prompt_extension"] = False
    for audit_field, expected in required_audit.items():
        if audit.get(audit_field) != expected:
            raise ValueError(
                f"mobilegpt_memory_audit_invalid:{audit_field}"
            )
    derive_fallback_count = audit.get("derive_agent_fallback_count")
    if type(derive_fallback_count) is not int or derive_fallback_count < 0:
        raise ValueError(
            "mobilegpt_memory_audit_invalid:derive_agent_fallback_count"
        )
    source_example_fallback_count = audit.get("source_example_fallback_count", 0)
    if not semantic_learning and (
        type(source_example_fallback_count) is not int
        or source_example_fallback_count < 0
        or int(official_reader.get("source_reader_coverage_count") or 0)
        != transition_count
        or int(official_reader.get("source_example_fallback_count") or 0)
        != source_example_fallback_count
    ):
        raise ValueError("mobilegpt_memory_audit_invalid:source_reader_coverage")

    provenance_root = bundle_root / (
        "provenance_mobilegpt_runlog_semantic_v1"
        if semantic_learning
        else "provenance_mobilegpt_runlog_direct_v1"
    )
    provenance_root.mkdir(exist_ok=False)

    def copy_evidence(source: Path, name: str) -> Path:
        destination = provenance_root / name
        shutil.copy2(source, destination)
        return destination

    copied_source = copy_evidence(source_path, "source.run_log.json")
    copied_stats = copy_evidence(stats_path, "mobilegpt_stats.jsonl")
    copied_audit = copy_evidence(audit_path, "trajectory_audit.json")
    memory_sha256, memory_file_count = mobilegpt_memory.mobilegpt_memory_digest(memory)
    provenance = {
        "native_mobilegpt_learning": False,
        "task_local_memory": True,
        "learning_mode": learning_mode,
        "teacher_forcing": semantic_learning and not official_learning,
        "synthetic_subtasks": not semantic_learning,
        "semantic_subtasks": semantic_learning,
        "original_mobilegpt_prompts": semantic_learning,
        "actions_supplied_to_mobilegpt": not semantic_learning,
        "source_transitions_supplied": True,
        "source_success_boundary_supplied": True,
        "runlog_transition_compilation": not semantic_learning,
        "complete_transition_mapping": not official_learning,
        "official_reader_validation": True,
        "function_store_used": False,
        "function_conversion_enabled": False,
        "target_inputs_read": False,
        "target_observations_read": False,
        "validator_state_read": False,
        "coordinate_replay": False,
        "source_emulator_used": False,
    }
    if semantic_learning:
        provenance.update(
            {
                "official_authoring_session": True,
                "teacher_action_alignment_complete": not official_learning,
            }
        )
    manifest = {
        "schema_version": memory_schema,
        "task_name": str(task_name),
        "source_seed": int(source_seed),
        "source_method": source_method,
        "source_model": str(source_model or ""),
        "model_contract": {
            "model": str(source_model or ""),
            "endpoint": FORMAL_MODEL_BASE_URL,
            "thinking": {"type": FORMAL_THINKING},
        },
        "target_package": str(target_package),
        "target_app": str(target_app or target_package),
        "memory": {
            "relative_path": memory.relative_to(bundle_root).as_posix(),
            "sha256": memory_sha256,
            "file_count": memory_file_count,
            "inventory": inventory,
            "validation": memory_validation,
        },
        "source_run_log": {
            "relative_path": copied_source.relative_to(bundle_root).as_posix(),
            "sha256": sha256_file(copied_source),
            "recorded_seed": recorded_source_seed,
        },
        "trajectory_audit": {
            "relative_path": copied_audit.relative_to(bundle_root).as_posix(),
            "sha256": sha256_file(copied_audit),
            "transition_count": transition_count,
            "validated_transition_count": validated_count,
        },
        "source_stats": {
            "relative_path": copied_stats.relative_to(bundle_root).as_posix(),
            "sha256": sha256_file(copied_stats),
            "task_started_count": int(stats_summary.get("task_started_count") or 0),
            "task_finished_count": int(stats_summary.get("task_finished_count") or 0),
            "model_calls": _coerce_int(stats_summary.get("model_calls")),
            "chat_model_calls": _coerce_int(
                stats_summary.get("chat_model_calls")
            ),
            "embedding_model_calls": _coerce_int(
                stats_summary.get("embedding_model_calls")
            ),
            "chat_models": list(stats_summary.get("chat_models") or []),
            "chat_attempts": chat_attempts,
            "embedding_models": list(stats_summary.get("embedding_models") or []),
            "prompt_tokens": _coerce_int(stats_summary.get("prompt_tokens")),
            "completion_tokens": _coerce_int(stats_summary.get("completion_tokens")),
            "total_tokens": _coerce_int(stats_summary.get("total_tokens")),
            "token_usage_status": str(
                stats_summary.get("token_usage_status") or ""
            ),
            "task_elapsed_sec": _coerce_float(stats_summary.get("task_elapsed_sec")),
            "wall_sec": float(source_wall_sec or 0.0),
        },
        "provenance": provenance,
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2, ensure_ascii=False))
        handle.write("\n")
    for path in memory.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    memory.chmod(0o555)
    return mobilegpt_memory.validate_mobilegpt_adapted_memory(
        memory,
        task_name=task_name,
        source_seed=source_seed,
        source_run_log=source_path,
        expected_model=str(source_model or ""),
        expected_source_method=source_method,
    )


_MOBILEGPT_IGNORED_TARGET_PACKAGES = {
    "android",
    "com.android.systemui",
    "com.example.MobileGPT",
    "com.google.android.apps.nexuslauncher",
    # Text input is exposed as the foreground package while a source task is
    # typing.  It is not the AndroidWorld target app and must not become the
    # sealed MobileGPT task package.
    "com.google.android.inputmethod.latin",
    "com.android.inputmethod.latin",
}


def _mobilegpt_action_package(action: Any) -> str:
    if not isinstance(action, dict):
        return ""
    action_type = str(
        action.get("action_type") or action.get("type") or action.get("tool") or ""
    ).strip()
    params = (
        dict(action.get("params") or {})
        if isinstance(action.get("params"), dict)
        else {}
    )
    package_name = str(
        params.get("package_name")
        or params.get("packageName")
        or params.get("app_package")
        or action.get("package_name")
        or action.get("packageName")
        or action.get("app_name")
        or ""
    ).strip()
    return package_name if action_type == "open_app" else ""


def _mobilegpt_observation_package(observation: Any) -> str:
    if not isinstance(observation, dict):
        return ""
    package_name = str(
        observation.get("package_name")
        or observation.get("packageName")
        or observation.get("app_package")
        or ""
    ).strip()
    auxiliaries = observation.get("auxiliaries")
    if not package_name and isinstance(auxiliaries, dict):
        package_name = str(
            auxiliaries.get("package_name")
            or auxiliaries.get("packageName")
            or ""
        ).strip()
    if package_name in _MOBILEGPT_IGNORED_TARGET_PACKAGES:
        return ""
    xml = observation.get("xml")
    if not isinstance(xml, str) or not xml.strip():
        xml = observation.get("forest")
    if isinstance(xml, str) and xml.strip():
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            root = None
        if root is not None:
            packages = [
                str(element.attrib.get("package") or "").strip()
                for element in root.iter()
                if str(element.attrib.get("package") or "").strip()
            ]
            non_system = [
                package
                for package in packages
                if package not in _MOBILEGPT_IGNORED_TARGET_PACKAGES
            ]
            xml_package = (non_system or [""])[-1]
            if xml_package:
                return xml_package
    return package_name


def _infer_mobilegpt_target_from_source_run_log(
    item: CanonicalRunLog,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, str]:
    """Infer the Android app MobileGPT should open from the source trajectory."""

    try:
        canonical, materialization, profile, _ = canonicalize_source_run_log(
            item,
        )
    except Exception as exc:
        return {
            "target_package": "",
            "target_app": "",
            "target_source": "unresolved",
            "target_error": str(exc),
        }

    steps = canonical.get("steps") if isinstance(canonical, dict) else []
    if not isinstance(steps, list):
        steps = []

    for step in steps:
        if not isinstance(step, dict):
            continue
        actions = [step.get("action")]
        actions.extend(list(step.get("actions") or []))
        actions.extend(list(step.get("executed_actions") or []))
        for action in actions:
            package_name = _mobilegpt_action_package(action)
            if package_name and package_name not in _MOBILEGPT_IGNORED_TARGET_PACKAGES:
                return {
                    "target_package": package_name,
                    "target_app": package_name,
                    "target_source": "source_runlog_open_app",
                    "source_materialization": materialization,
                }
        tool_call = step.get("tool_call")
        if isinstance(tool_call, dict):
            action = {
                "type": tool_call.get("name"),
                "params": tool_call.get("params")
                if isinstance(tool_call.get("params"), dict)
                else tool_call.get("arguments"),
            }
            package_name = _mobilegpt_action_package(action)
            if package_name and package_name not in _MOBILEGPT_IGNORED_TARGET_PACKAGES:
                return {
                    "target_package": package_name,
                    "target_app": package_name,
                    "target_source": "source_runlog_tool_call_open_app",
                    "source_materialization": materialization,
                }

    for step in steps:
        if not isinstance(step, dict):
            continue
        for key in ("observation_before_act", "observation_after_act", "observation"):
            package_name = _mobilegpt_observation_package(step.get(key))
            if package_name:
                return {
                    "target_package": package_name,
                    "target_app": package_name,
                    "target_source": f"source_runlog_{key}",
                    "source_materialization": materialization,
                }

    package_name = _mobilegpt_observation_package(canonical.get("final_state"))
    if package_name:
        return {
            "target_package": package_name,
            "target_app": package_name,
            "target_source": "source_runlog_final_state",
            "source_materialization": materialization,
        }

    return {
        "target_package": "",
        "target_app": "",
        "target_source": "unresolved",
        "source_materialization": materialization,
    }
