"""Build MobileGPT native memory with a seed-111 cold episode."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import time
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.experiment import run_task as pipeline
from src.experiment.mobilegpt_contract import (
    MOBILEGPT_EMBEDDING_MODEL,
    MOBILEGPT_LEARNING_MODE,
    MOBILEGPT_MEMORY_MANIFEST,
    MOBILEGPT_MEMORY_SCHEMA,
    MOBILEGPT_RUNLOG_MEMORY_SCHEMA,
    MOBILEGPT_SOURCE_METHOD,
    MOBILEGPT_SOURCE_METHOD_BY_SCHEMA,
)
from src.experiment.paths import sha256_file
from src.experiment.protocol import SOURCE_SEED
from src.experiment.source_records import CanonicalRunLog
from src.integrations import mobilegpt_memory
from src.integrations.mobilegpt import (
    MobileGPTConversionError,
    convert_runlog_to_mobilegpt_memory,
    preflight_runlog_conversion,
    write_conversion_failure_audit,
)
from src.integrations.runlog import import_run_log

_IGNORED_SOURCE_PACKAGES = {
    "android",
    "com.android.systemui",
    "com.example.MobileGPT",
    "com.google.android.apps.nexuslauncher",
}


def load_canonical_source_item(
    index_path: str | Path,
    *,
    task_name: str,
) -> CanonicalRunLog:
    matches = [
        item
        for item in pipeline.load_canonical_source_index(index_path)
        if item.task == str(task_name)
    ]
    if len(matches) != 1:
        raise ValueError(
            "mobilegpt_source_task_resolution_failed:"
            f"task={task_name}:matches={len(matches)}"
        )
    item = matches[0]
    source_kind = str(item.meta.get("source_kind") or "").strip()
    if item.meta.get("latest_official_success_source") is not True:
        raise ValueError(
            f"mobilegpt_source_official_success_required:task={task_name}"
        )
    if source_kind and source_kind not in {
        "androidworld_validator_success_source_runlog",
        "one_time_canonicalized_seed111_screenshot_source",
    }:
        raise ValueError(
            "mobilegpt_source_kind_invalid:"
            f"task={task_name}:actual={source_kind}"
        )
    if not item.source_run_log.is_file():
        raise FileNotFoundError(
            f"mobilegpt_source_runlog_missing:{item.source_run_log}"
        )
    canonical = import_run_log(
        json.loads(item.source_run_log.read_text(encoding="utf-8"))
    )
    if (
        canonical.get("status") != "succeeded"
        or canonical.get("success") is not True
        or not canonical.get("steps")
    ):
        raise ValueError(
            f"mobilegpt_source_runlog_not_successful:task={task_name}"
        )
    return item


def _mobilegpt_source_target(
    *,
    item: CanonicalRunLog,
    source: dict[str, Any],
) -> dict[str, str]:
    inferred = pipeline._infer_mobilegpt_target_from_source_run_log(item)
    package_name = str(inferred.get("target_package") or "").strip()
    if package_name and package_name not in _IGNORED_SOURCE_PACKAGES:
        return {
            key: str(value)
            for key, value in inferred.items()
            if value is not None
        }
    source_packages: set[str] = set()
    raw_source_packages: set[str] = set()
    for step in source.get("steps") or []:
        observation = step.get("observation") if isinstance(step, dict) else None
        if isinstance(observation, dict):
            explicit_package = str(
                observation.get("package_name")
                or observation.get("packageName")
                or observation.get("app_package")
                or ""
            ).strip()
            if explicit_package:
                raw_source_packages.add(explicit_package)
            raw_xml = str(
                observation.get("xml") or observation.get("forest") or ""
            ).strip()
            if raw_xml:
                try:
                    root = ET.fromstring(raw_xml)
                except ET.ParseError:
                    root = None
                if root is not None:
                    raw_source_packages.update(
                        str(element.get("package") or "").strip()
                        for element in root.iter()
                        if str(element.get("package") or "").strip()
                    )
        package = pipeline._mobilegpt_observation_package(observation)
        if package and package not in _IGNORED_SOURCE_PACKAGES:
            source_packages.add(package)
    if (
        not source_packages
        and "com.android.systemui" in raw_source_packages
        and raw_source_packages.issubset(_IGNORED_SOURCE_PACKAGES)
    ):
        return {
            "target_package": "com.android.settings",
            "target_app": "com.android.settings",
            "target_source": "system_ui_source_bootstrap",
        }
    launcher_packages = sorted(
        package
        for package in raw_source_packages
        if package in {
            "com.google.android.apps.nexuslauncher",
        }
    )
    if (
        not source_packages
        and len(launcher_packages) == 1
        and raw_source_packages.issubset(_IGNORED_SOURCE_PACKAGES)
    ):
        package_name = launcher_packages[0]
        return {
            "target_package": package_name,
            "target_app": package_name,
            "target_source": "launcher_source_bootstrap",
        }
    if len(source_packages) != 1:
        label = "unresolved" if not source_packages else "ambiguous"
        raise ValueError(
            f"mobilegpt_source_target_package_{label}:"
            + ",".join(sorted(source_packages))
        )
    package_name = next(iter(source_packages))
    return {
        "target_package": package_name,
        "target_app": package_name,
        "target_source": "canonical_source_runlog_observation",
    }


def _source_preflight(
    item: CanonicalRunLog,
) -> tuple[Path, tuple[str, ...], dict[str, Any], dict[str, str]]:
    source_run_log = item.source_run_log
    source_sha256 = sha256_file(source_run_log)
    source = import_run_log(
        json.loads(source_run_log.read_text(encoding="utf-8"))
    )
    target_info = _mobilegpt_source_target(item=item, source=source)
    audit = {
        "schema_version": "omniflow.mobilegpt-native-source-check.v1",
        "task_name": item.task,
        "source_run_log": str(source_run_log),
        "source_run_log_sha256": source_sha256,
        "learning_mode": MOBILEGPT_LEARNING_MODE,
        "teacher_forcing": False,
        "actions_supplied_to_mobilegpt": False,
        "runlog_conversion_used": False,
        "source_emulator_required": True,
        "target_package": target_info["target_package"],
        "function_store_used": False,
    }
    return source_run_log, (source_sha256,), audit, target_info


def preflight_mobilegpt_source(
    *,
    index_path: str | Path,
    task_name: str,
) -> dict[str, Any]:
    item = load_canonical_source_item(index_path, task_name=task_name)
    source_run_log, _, source_audit, target_info = _source_preflight(item)
    return {
        "schema_version": "omniflow.mobilegpt-native-source-preflight.v1",
        "task_name": item.task,
        "source_seed": SOURCE_SEED,
        "source_method": MOBILEGPT_SOURCE_METHOD,
        "source_run_log": str(source_run_log),
        "source_run_log_sha256": source_audit["source_run_log_sha256"],
        "learning_mode": MOBILEGPT_LEARNING_MODE,
        "teacher_forcing": False,
        "synthetic_subtasks": False,
        "semantic_subtasks": True,
        "original_mobilegpt_prompts": True,
        "actions_supplied_to_mobilegpt": False,
        "runlog_conversion_used": False,
        "source_emulator_required": True,
        "function_store_used": False,
        "target_package": target_info["target_package"],
        "target_source": target_info["target_source"],
        "source_audit": source_audit,
        "ready": True,
    }


def validate_mobilegpt_source_memory(
    *,
    index_path: str | Path,
    task_name: str,
    memory_root: str | Path,
    model: str,
    memory_index: str | Path | None = None,
) -> dict[str, Any]:
    item = load_canonical_source_item(index_path, task_name=task_name)
    source_run_log, compatible_sha256s, _, target_info = _source_preflight(item)
    manifest_path = Path(memory_root).expanduser().resolve().parent / MOBILEGPT_MEMORY_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = str(manifest.get("schema_version") or "")
    expected_source_method = MOBILEGPT_SOURCE_METHOD_BY_SCHEMA.get(schema_version)
    if expected_source_method is None:
        raise ValueError("mobilegpt_source_memory_schema_invalid")
    expected_target_package = str(target_info.get("target_package") or "").strip()
    actual_target_package = str(manifest.get("target_package") or "").strip()
    if actual_target_package != expected_target_package:
        raise ValueError(
            "mobilegpt_source_memory_target_package_mismatch:"
            f"expected={expected_target_package}:actual={actual_target_package}"
        )
    validated = mobilegpt_memory.validate_mobilegpt_adapted_memory(
        memory_root,
        task_name=item.task,
        source_seed=SOURCE_SEED,
        source_run_log=source_run_log,
        compatible_source_sha256s=compatible_sha256s,
        expected_model=str(model),
        expected_source_method=expected_source_method,
    )
    result = {
        "schema_version": "omniflow.mobilegpt.memory-check.v2",
        "task_name": item.task,
        "source_seed": SOURCE_SEED,
        "source_method": expected_source_method,
        "source_run_log": str(source_run_log),
        "model": str(model),
        "validated": validated,
    }
    if memory_index is not None:
        result["memory_registration"] = _register_mobilegpt_memory(
            memory_index=memory_index,
            bundle_root=Path(memory_root).expanduser().resolve().parent,
            task_name=item.task,
        )
    return result


def _register_mobilegpt_memory(
    *,
    memory_index: str | Path,
    bundle_root: str | Path,
    task_name: str,
) -> dict[str, Any]:
    from src.experiment.data_index import refresh_data_index_from_pointer

    report = refresh_data_index_from_pointer(
        memory_index=memory_index,
        additional_prepared_memory_roots=(bundle_root,),
        replace_prepared_memory_roots=True,
    )
    registered = report.get("canonical", {}).get("prepared_memories", {}).get(
        str(task_name)
    )
    if not isinstance(registered, dict):
        raise ValueError(f"mobilegpt_memory_registration_missing:{task_name}")
    return registered


def prepare_mobilegpt_source_memory(
    *,
    index_path: str | Path,
    task_name: str,
    mobilegpt_root: str | Path,
    android_world_root: str | Path,
    output_root: str | Path,
    model: str,
    embedding_model: str = MOBILEGPT_EMBEDDING_MODEL,
    memory_index: str | Path | None = None,
    serial: str = "emulator-5560",
    console_port: int = 5560,
    adb_path: str = "",
    max_steps: int = 20,
    server_host: str = "0.0.0.0",
    port: int = 12345,
    server_warmup_sec: float = 5.0,
    wait_start_timeout_sec: float = 60.0,
    wait_finish_timeout_sec: float = 600.0,
    timeout_sec: float = 600.0,
    perform_emulator_setup: bool = True,
) -> dict[str, Any]:
    """Run MobileGPT's native cold learning through its official client."""

    normalized_model = str(model or "").strip()
    if not normalized_model:
        raise ValueError("mobilegpt_source_model_required")
    normalized_embedding_model = (
        str(embedding_model or "").strip() or MOBILEGPT_EMBEDDING_MODEL
    )
    item = load_canonical_source_item(index_path, task_name=task_name)
    source_run_log, _, source_audit, target_info = _source_preflight(item)
    resolved_target_package = pipeline._resolve_mobilegpt_target_package(
        target_info["target_package"],
        adb_path=str(adb_path),
        serial=str(serial),
        android_world_root=android_world_root,
    )
    if not resolved_target_package or "." not in resolved_target_package:
        raise ValueError(
            "mobilegpt_source_target_package_unresolved:"
            + str(target_info["target_package"])
        )
    target_info = {
        **target_info,
        "target_package": resolved_target_package,
    }
    source_audit = {
        **source_audit,
        "target_package": resolved_target_package,
    }
    bundle_root = Path(output_root).expanduser().resolve()
    if bundle_root.exists():
        raise FileExistsError(
            f"immutable_mobilegpt_source_attempt_exists:{bundle_root}"
        )
    bundle_root.mkdir(parents=True)
    memory_root = bundle_root / "memory"
    memory_root.mkdir()
    stats_path = bundle_root / "source_stats.jsonl"
    stats_summary_path = bundle_root / "source_stats_summary.json"
    target_device = pipeline.DeviceTarget(
        label=f"source{int(console_port)}",
        serial=str(serial),
        console_port=int(console_port),
    )

    server_spec = pipeline.build_mobilegpt_server_command(
        "server",
        mobilegpt_root=mobilegpt_root,
        mobilegpt_memory_root=memory_root,
        embedding_model=normalized_embedding_model,
        write_through_memory=True,
        serial=serial,
        adb_path=adb_path,
        server_host=server_host,
        port=int(port),
        stats_jsonl=stats_path,
        target_package=target_info["target_package"],
        target_app=target_info["target_app"],
        target_task_name=item.task,
    )
    server_spec = pipeline._configure_mobilegpt_formal_server(
        server_spec,
        model=normalized_model,
    )
    server_spec = replace(
        server_spec,
        metadata={
            **server_spec.metadata,
            "source_method": MOBILEGPT_SOURCE_METHOD,
            "learning_mode": MOBILEGPT_LEARNING_MODE,
            "teacher_forcing": False,
            "runlog_conversion_used": False,
            "physical_backend": "mobilegpt_official_accessibility",
        },
    )

    episode_spec = pipeline.build_mobilegpt_command(
        item,
        method_name=MOBILEGPT_SOURCE_METHOD,
        target=target_device,
        android_world_root=android_world_root,
        output_root=bundle_root / "_source_episode",
        stats_jsonl=stats_path,
        mobilegpt_root=mobilegpt_root,
        server_host=server_host,
        server_port=int(port),
        target_package=target_info["target_package"],
        max_steps=int(max_steps),
        task_random_seed=SOURCE_SEED,
        fixed_task_seed=True,
        fixed_task_params=True,
        task_params_override=dict(item.params),
        perform_emulator_setup=bool(perform_emulator_setup),
        adb_path=str(adb_path),
        start_timeout_sec=float(wait_start_timeout_sec),
        finish_timeout_sec=float(wait_finish_timeout_sec),
        timeout_sec=float(timeout_sec),
        server_log_path=str(server_spec.metadata.get("log_path") or ""),
    )
    if episode_spec.metadata.get("action_backend") != "mobilegpt_official_accessibility":
        raise RuntimeError("mobilegpt_source_requires_official_accessibility_actions")
    if episode_spec.metadata.get("observe_backend") != "mobilegpt_official_accessibility":
        raise RuntimeError("mobilegpt_source_requires_official_accessibility_observations")
    (bundle_root / "source_episode_command.json").write_text(
        json.dumps(
            {
                "schema_version": "omniflow.mobilegpt-native-source-command.v1",
                "task_name": item.task,
                "source_seed": SOURCE_SEED,
                "server_command": pipeline._command_line(server_spec),
                "episode_command": pipeline._command_line(episode_spec),
                "source_audit": source_audit,
                "physical_backend": "mobilegpt_official_accessibility",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    server = None
    started = time.monotonic()
    try:
        server, server_returncode = pipeline._start_background_command(
            server_spec,
            warmup_sec=float(server_warmup_sec),
        )
        if server_returncode != 0:
            raise RuntimeError(f"mobilegpt_native_server_failed:{server_returncode}")
        episode_returncode = pipeline.run_command(episode_spec)
        if episode_returncode != 0:
            raise RuntimeError(f"mobilegpt_source_episode_failed:{episode_returncode}")
    finally:
        pipeline._stop_background_command(server)
    wall_sec = round(time.monotonic() - started, 6)
    if episode_spec.output_path is None:
        raise RuntimeError("mobilegpt_source_episode_output_missing")
    result_path = episode_spec.output_path / "task_results.jsonl"
    stats_summary = mobilegpt_memory.summarize_mobilegpt_stats(stats_path)
    stats_summary_path.write_text(
        json.dumps(stats_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    sealed = pipeline.seal_mobilegpt_source_memory(
        memory_root=memory_root,
        source_run_log=source_run_log,
        source_stats=stats_path,
        official_source_result=result_path,
        task_name=item.task,
        source_seed=SOURCE_SEED,
        target_package=target_info["target_package"],
        target_app=target_info["target_app"],
        source_wall_sec=wall_sec,
        source_model=normalized_model,
    )
    result = {
        "schema_version": "omniflow.mobilegpt-native-source-prepare.v1",
        "task_name": item.task,
        "source_seed": SOURCE_SEED,
        "source_method": MOBILEGPT_SOURCE_METHOD,
        "model": normalized_model,
        "embedding_model": normalized_embedding_model,
        "memory_root": str(memory_root),
        "learning_mode": MOBILEGPT_LEARNING_MODE,
        "teacher_forcing": False,
        "actions_supplied_to_mobilegpt": False,
        "runlog_conversion_used": False,
        "source_emulator_used": True,
        "physical_backend": "mobilegpt_official_accessibility",
        "source_stats": str(stats_path),
        "source_stats_summary": str(stats_summary_path),
        "official_source_result": str(result_path),
        "source_wall_sec": wall_sec,
        "sealed": sealed,
    }
    if memory_index is not None:
        result["memory_registration"] = _register_mobilegpt_memory(
            memory_index=memory_index,
            bundle_root=bundle_root,
            task_name=item.task,
        )
    return result


def convert_runlog_to_mobilegpt_bundle(
    *,
    source_run_log: str | Path,
    mobilegpt_root: str | Path,
    output_root: str | Path,
    model: str,
    embedding_model: str = MOBILEGPT_EMBEDDING_MODEL,
    target_package: str = "",
    target_app: str = "",
    preflight_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert one valid RunLog into an official-schema MobileGPT bundle."""

    normalized_model = str(model or "").strip()
    if not normalized_model:
        raise ValueError("mobilegpt_source_model_required")
    normalized_embedding_model = (
        str(embedding_model or "").strip() or MOBILEGPT_EMBEDDING_MODEL
    )
    source_path = Path(source_run_log).expanduser().resolve()
    source = import_run_log(json.loads(source_path.read_text(encoding="utf-8")))
    if (
        source.get("status") != "succeeded"
        or source.get("success") is not True
        or (source.get("validator") or {}).get("official") is not True
        or (source.get("validator") or {}).get("success") is not True
    ):
        raise ValueError("mobilegpt_source_runlog_not_successful")
    report = preflight_runlog_conversion(
        source_path,
        target_package=target_package,
        target_app=target_app,
    )
    if report.get("ready") is not True:
        raise MobileGPTConversionError(
            str(report.get("failure_code") or "mobilegpt_conversion_preflight_failed"),
            **dict(report.get("failure_details") or {}),
        )
    source_audit = preflight_audit or {
        "schema_version": "omniflow.mobilegpt.source-check.v2",
        "grounding_source": "canonical_androidworld_run_log",
        "source_run_log": str(source_path),
        "source_run_log_sha256": sha256_file(source_path),
        "actions_supplied_to_mobilegpt": True,
        "function_store_used": False,
        "report": report,
    }
    bundle_root = Path(output_root).expanduser().resolve()
    if bundle_root.exists():
        raise FileExistsError(
            f"immutable_mobilegpt_source_attempt_exists:{bundle_root}"
        )
    bundle_root.mkdir(parents=True)
    preflight_path = bundle_root / "conversion_preflight.json"
    preflight_path.write_text(
        json.dumps(source_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    memory_root = bundle_root / "memory"
    stats_path = bundle_root / "source_stats.jsonl"
    stats_summary_path = bundle_root / "source_stats_summary.json"
    audit_path = bundle_root / "trajectory_audit.json"
    started = time.monotonic()
    try:
        generated = convert_runlog_to_mobilegpt_memory(
            source_run_log=source_path,
            mobilegpt_root=mobilegpt_root,
            memory_root=memory_root,
            stats_path=stats_path,
            audit_path=audit_path,
            model=normalized_model,
            embedding_model=normalized_embedding_model,
            target_package=str(target_package or ""),
            target_app=str(target_app or ""),
        )
    except BaseException as error:
        write_conversion_failure_audit(
            source_run_log=source_path,
            stats_path=stats_path,
            audit_path=audit_path,
            error=error,
            wall_sec=time.monotonic() - started,
            target_package=str(target_package or ""),
            target_app=str(target_app or ""),
        )
        raise
    wall_sec = round(time.monotonic() - started, 6)
    stats_summary = mobilegpt_memory.summarize_mobilegpt_stats(stats_path)
    stats_summary_path.write_text(
        json.dumps(stats_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    sealed = pipeline.seal_mobilegpt_converted_memory(
        memory_root=memory_root,
        source_run_log=source_path,
        source_stats=stats_path,
        trajectory_audit=audit_path,
        task_name=str(source["task_name"]),
        source_seed=SOURCE_SEED,
        target_package=str(target_package or generated.get("target_package") or ""),
        target_app=str(target_app or generated.get("target_app") or ""),
        source_wall_sec=wall_sec,
        source_model=normalized_model,
        memory_schema=MOBILEGPT_RUNLOG_MEMORY_SCHEMA,
    )
    return {
        "schema_version": "omniflow.mobilegpt.memory-prepare.v2",
        "method": "mobilegpt",
        "task_name": str(source["task_name"]),
        "source_seed": SOURCE_SEED,
        "source_run_log": str(source_path),
        "model": normalized_model,
        "embedding_model": normalized_embedding_model,
        "memory_root": str(memory_root),
        "source_stats": str(stats_path),
        "source_stats_summary": str(stats_summary_path),
        "trajectory_audit": str(audit_path),
        "source_wall_sec": wall_sec,
        "generated": generated,
        "sealed": sealed,
        "manifest": sealed,
    }


def _write_failure_marker(output_root: str | Path, error: BaseException) -> None:
    root = Path(output_root).expanduser().resolve()
    if not root.is_dir() or (root / MOBILEGPT_MEMORY_MANIFEST).exists():
        return
    marker = root / "prep_failure.json"
    if marker.exists():
        return
    stats_path = root / "source_stats.jsonl"
    stats_summary = (
        mobilegpt_memory.summarize_mobilegpt_stats(stats_path)
        if stats_path.is_file()
        else {}
    )
    audit_path = root / "trajectory_audit.json"
    trajectory_audit: dict[str, Any] = {}
    if audit_path.is_file():
        try:
            loaded = json.loads(audit_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            loaded = {}
        if isinstance(loaded, dict):
            trajectory_audit = {
                key: loaded.get(key)
                for key in (
                    "transition_count",
                    "validated_transition_count",
                    "failure_code",
                    "failure_details",
                    "wall_sec",
                )
            }
    marker.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.mobilegpt.memory-failure.v2",
                "failed_at": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
                "stats": stats_summary,
                "trajectory_audit": trajectory_audit,
                "retry_allowed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--index", required=True)
    prepare.add_argument("--task", required=True)
    prepare.add_argument("--mobilegpt-root", required=True)
    prepare.add_argument("--android-world-root", required=True)
    prepare.add_argument("--output-root", required=True)
    prepare.add_argument("--model", required=True)
    prepare.add_argument(
        "--embedding-model",
        default=os.environ.get("MOBILEGPT_EMBEDDING_MODEL")
        or MOBILEGPT_EMBEDDING_MODEL,
    )
    prepare.add_argument("--memory-index", required=True)
    prepare.add_argument("--serial", default="emulator-5560")
    prepare.add_argument("--console-port", type=int, default=5560)
    prepare.add_argument("--adb-path", default="")
    prepare.add_argument("--max-steps", type=int, default=20)
    prepare.add_argument("--server-host", default="0.0.0.0")
    prepare.add_argument("--port", type=int, default=12345)
    prepare.add_argument("--server-warmup-sec", type=float, default=5.0)
    prepare.add_argument("--wait-start-timeout-sec", type=float, default=60.0)
    prepare.add_argument("--wait-finish-timeout-sec", type=float, default=600.0)
    prepare.add_argument("--timeout-sec", type=float, default=600.0)
    prepare.add_argument("--no-emulator-setup", action="store_true")
    validate = subparsers.add_parser("validate")
    validate.add_argument("--index", required=True)
    validate.add_argument("--task", required=True)
    validate.add_argument("--memory-root", required=True)
    validate.add_argument("--model", required=True)
    validate.add_argument("--memory-index", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--index", required=True)
    preflight.add_argument("--task", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "prepare":
            result = prepare_mobilegpt_source_memory(
                index_path=args.index,
                task_name=args.task,
                mobilegpt_root=args.mobilegpt_root,
                android_world_root=args.android_world_root,
                output_root=args.output_root,
                model=args.model,
                embedding_model=args.embedding_model,
                memory_index=args.memory_index,
                serial=args.serial,
                console_port=args.console_port,
                adb_path=args.adb_path,
                max_steps=args.max_steps,
                server_host=args.server_host,
                port=args.port,
                server_warmup_sec=args.server_warmup_sec,
                wait_start_timeout_sec=args.wait_start_timeout_sec,
                wait_finish_timeout_sec=args.wait_finish_timeout_sec,
                timeout_sec=args.timeout_sec,
                perform_emulator_setup=not args.no_emulator_setup,
            )
        elif args.command == "validate":
            result = validate_mobilegpt_source_memory(
                index_path=args.index,
                task_name=args.task,
                memory_root=args.memory_root,
                model=args.model,
                memory_index=args.memory_index,
            )
        elif args.command == "preflight":
            result = preflight_mobilegpt_source(
                index_path=args.index,
                task_name=args.task,
            )
        else:
            raise ValueError(f"unsupported_mobilegpt_source_command:{args.command}")
    except BaseException as error:
        if args.command == "prepare":
            _write_failure_marker(args.output_root, error)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
