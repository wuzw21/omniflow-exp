"""Prepare one immutable AppAgent demo memory from a canonical RunLog."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from contextlib import redirect_stderr, redirect_stdout
import datetime
import importlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
import time
from typing import Any

from omniflow.core.trajectory import observation_xml
from src.experiment import androidworld as pipeline
from src.experiment.mobilegpt_source import (
    load_canonical_source_item,
)
from src.experiment.source_assets import (
    build_grounded_teacher_run_log_from_canonical_item,
)
from src.integrations.appagent_adapter import (
    APPAGENT_DEMO_ACTION_TYPES,
    APPAGENT_DEMO_MANIFEST,
    APPAGENT_OFFICIAL_REVISION,
    OfficialAppAgentRuntime,
    build_appagent_teacher_source,
    ground_appagent_teacher_action,
    seal_appagent_demo_memory,
    validate_appagent_demo_memory,
)

SOURCE_SEED = 111


def _appagent_source_method_label(item: pipeline.ArchivedRunLog) -> str:
    return str(item.meta.get("method") or "").strip() or pipeline.DEFAULT_SOURCE_METHOD


def _chat_completions_url(base_url: str) -> str:
    normalized = str(base_url or "").strip().rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return normalized + "/chat/completions"


def _write_runtime_config(
    path: Path,
    *,
    api_key: str,
    endpoint: str,
    model: str,
    timeout_sec: float,
) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("AppAgent source generation requires PyYAML") from exc

    payload = {
        "MODEL": "OpenAI",
        "OPENAI_API_BASE": endpoint,
        "OPENAI_API_KEY": api_key,
        "OPENAI_API_MODEL": model,
        "MAX_TOKENS": 300,
        "TEMPERATURE": 0.0,
        "REQUEST_INTERVAL": 0.0,
        "DASHSCOPE_API_KEY": "",
        "QWEN_MODEL": model,
        "ANDROID_SCREENSHOT_DIR": "/sdcard",
        "ANDROID_XML_DIR": "/sdcard",
        "DOC_REFINE": False,
        "MAX_ROUNDS": 20,
        "DARK_MODE": False,
        "MIN_DIST": 30,
        "OMNIFLOW_REQUEST_TIMEOUT_SEC": float(timeout_sec),
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def run_official_document_generation(
    *,
    appagent_root: str | Path,
    workspace_root: str | Path,
    app_name: str,
    demo_name: str,
    log_path: str | Path,
    usage_path: str | Path,
    model: str,
    timeout_sec: float = 60.0,
) -> dict[str, Any]:
    """Run the pinned official generator once and record exact API usage."""

    root = Path(appagent_root).expanduser().resolve()
    workspace = Path(workspace_root).expanduser().resolve()
    output_log = Path(log_path).expanduser().resolve()
    output_usage = Path(usage_path).expanduser().resolve()
    normalized_model = str(model or "").strip()
    if not normalized_model:
        raise ValueError("appagent_document_model_required")
    if float(timeout_sec) <= 0:
        raise ValueError("appagent_document_timeout_must_be_positive")

    # This verifies both the exact official revision and its runtime modules.
    OfficialAppAgentRuntime(root)
    api_key = str(
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY")
        or ""
    ).strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for AppAgent docs")
    endpoint = _chat_completions_url(
        str(
            os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OMNIFLOW_OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
    )
    output_log.parent.mkdir(parents=True, exist_ok=True)
    output_usage.parent.mkdir(parents=True, exist_ok=True)
    if output_log.exists() or output_usage.exists():
        raise FileExistsError("immutable_appagent_doc_generation_output_exists")

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="appagent-doc-config-") as temporary:
        temporary_root = Path(temporary)
        _write_runtime_config(
            temporary_root / "config.yaml",
            api_key=api_key,
            endpoint=endpoint,
            model=normalized_model,
            timeout_sec=float(timeout_sec),
        )
        previous_cwd = Path.cwd()
        previous_argv = list(sys.argv)
        scripts_dir = root / "scripts"
        sys.path.insert(0, str(scripts_dir))
        official_model = importlib.import_module("model")
        original_post = official_model.requests.post

        def instrumented_post(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("timeout", float(timeout_sec))
            response = original_post(*args, **kwargs)
            payload = response.json()
            usage_payload = (
                payload.get("usage") if isinstance(payload, dict) else None
            )
            usage = (
                dict(usage_payload) if isinstance(usage_payload, dict) else {}
            )
            record = {
                "model": (
                    payload.get("model") if isinstance(payload, dict) else None
                )
                or normalized_model,
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
            }
            if record["total_tokens"] <= 0:
                record["total_tokens"] = (
                    record["prompt_tokens"] + record["completion_tokens"]
                )
            with output_usage.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            class ResponseProxy:
                def json(self) -> Any:
                    return payload

                def __getattr__(self, name: str) -> Any:
                    return getattr(response, name)

            return ResponseProxy()

        try:
            official_model.requests.post = instrumented_post
            os.chdir(temporary_root)
            sys.argv = [
                str(root / "scripts" / "document_generation.py"),
                "--app",
                str(app_name),
                "--demo",
                str(demo_name),
                "--root_dir",
                str(workspace),
            ]
            with output_log.open("x", encoding="utf-8") as log_handle:
                with redirect_stdout(log_handle), redirect_stderr(log_handle):
                    runpy.run_path(
                        str(root / "scripts" / "document_generation.py"),
                        run_name="__main__",
                    )
        finally:
            official_model.requests.post = original_post
            sys.argv = previous_argv
            os.chdir(previous_cwd)
            try:
                sys.path.remove(str(scripts_dir))
            except ValueError:
                pass

    if not output_usage.exists():
        output_usage.touch()
    usage_rows = [
        json.loads(line)
        for line in output_usage.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    models = sorted(
        {
            str(row.get("model") or "").strip()
            for row in usage_rows
            if str(row.get("model") or "").strip()
        }
    )
    docs_root = workspace / "apps" / str(app_name) / "demo_docs"
    return {
        "official_appagent_revision": APPAGENT_OFFICIAL_REVISION,
        "app_name": str(app_name),
        "demo_name": str(demo_name),
        "docs_root": str(docs_root),
        "docs_count": len(
            [path for path in docs_root.glob("*.txt") if path.is_file()]
        ),
        "model_calls": len(usage_rows),
        "models": models,
        "prompt_tokens": sum(
            int(row.get("prompt_tokens") or 0) for row in usage_rows
        ),
        "completion_tokens": sum(
            int(row.get("completion_tokens") or 0) for row in usage_rows
        ),
        "total_tokens": sum(
            int(row.get("total_tokens") or 0) for row in usage_rows
        ),
        "wall_sec": round(time.monotonic() - started, 6),
        "log_path": str(output_log),
        "usage_path": str(output_usage),
        "retry_count": 0,
    }


def _runlog_lineage(payload: dict[str, Any], content_sha256: str) -> set[str]:
    lineage = {content_sha256}
    provenance = payload.get("provenance")
    provenance_sha256 = str(
        provenance.get("source_sha256") if isinstance(provenance, dict) else ""
    ).strip()
    if len(provenance_sha256) == 64:
        lineage.add(provenance_sha256)
    return lineage


def _source_lineage(item: pipeline.ArchivedRunLog) -> tuple[str, set[str]]:
    payload = json.loads(item.source_run_log.read_text(encoding="utf-8"))
    run_id = str(payload.get("run_id") or "").strip()
    source_sha256s = _runlog_lineage(
        payload,
        pipeline._file_sha256(item.source_run_log),
    )
    if not run_id:
        raise ValueError("appagent_native_memory_lineage_missing")
    return run_id, source_sha256s


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _native_memory_evidence(
    *,
    item: pipeline.ArchivedRunLog,
    teacher_source: dict[str, Any],
    evidence_roots: Sequence[str | Path],
    model: str,
) -> dict[str, Any]:
    run_id, source_sha256s = _source_lineage(item)
    candidates: list[dict[str, Any]] = []
    for raw_root in evidence_roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"appagent_native_memory_root_missing:{root}")
        for manifest_path in root.rglob(APPAGENT_DEMO_MANIFEST):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            manifest_runlog = Path(
                str(manifest.get("source_run_log") or "")
            ).expanduser()
            manifest_lineage = {
                str(manifest.get("source_run_log_sha256") or "").strip()
            }
            evidence_runlogs = [manifest_runlog]
            grounded_runlog = manifest_path.parent / "grounded_teacher_run_log.json"
            if grounded_runlog.is_file():
                evidence_runlogs.append(grounded_runlog)
            for evidence_runlog in evidence_runlogs:
                if not evidence_runlog.is_file():
                    continue
                try:
                    manifest_payload = json.loads(
                        evidence_runlog.read_text(encoding="utf-8")
                    )
                    manifest_lineage.update(
                        _runlog_lineage(
                            manifest_payload,
                            pipeline._file_sha256(evidence_runlog),
                        )
                    )
                except (OSError, json.JSONDecodeError):
                    continue
            if (
                manifest.get("official_appagent_revision") != APPAGENT_OFFICIAL_REVISION
                or manifest.get("task_name") != item.task
                or manifest.get("source_seed") != SOURCE_SEED
                or str(manifest.get("source_run_id") or "") != run_id
                or source_sha256s.isdisjoint(manifest_lineage)
            ):
                continue
            app_name = str(manifest.get("app_name") or "").strip()
            demo_name = str(manifest.get("demo_name") or "").strip()
            base = manifest_path.parent
            demo_root = base / "apps" / app_name / "demos" / demo_name
            docs_root = base / "apps" / app_name / "demo_docs"
            log_path = base / "_document_generation" / "document_generation.log"
            usage_path = (
                base
                / "_document_generation"
                / "document_generation_usage.jsonl"
            )
            trace_path = demo_root / "teacher_trace.jsonl"
            if not all(
                path.exists()
                for path in (demo_root, docs_root, log_path, usage_path, trace_path)
            ):
                continue
            usage_rows = _jsonl(usage_path)
            models = {
                str(row.get("model") or "").strip()
                for row in usage_rows
                if str(row.get("model") or "").strip()
            }
            teacher_actions = teacher_source["actions"]
            trace = _jsonl(trace_path)
            if len(trace) != len(teacher_actions) or any(
                str(row.get("action_type") or "")
                != str((record.get("action") or {}).get("type") or "")
                for row, record in zip(trace, teacher_actions, strict=True)
            ):
                continue
            if not models:
                continue
            candidates.append(
                {
                    "manifest": manifest_path.resolve(),
                    "app_name": app_name,
                    "demo_name": demo_name,
                    "demo_root": demo_root.resolve(),
                    "docs_root": docs_root.resolve(),
                    "document_log": log_path.resolve(),
                    "document_usage": usage_path.resolve(),
                    "document_model": min(models),
                    "identity": (
                        str(manifest.get("demo_sha256") or ""),
                        str(manifest.get("demo_docs_sha256") or ""),
                        str(manifest.get("document_generation_usage_sha256") or ""),
                    ),
                }
            )
    if not candidates:
        raise FileNotFoundError(
            "appagent_native_memory_evidence_missing:"
            f"{item.task}:{','.join(sorted(source_sha256s))}"
        )
    identities = {candidate["identity"] for candidate in candidates}
    if len(identities) != 1:
        raise ValueError(
            f"appagent_native_memory_evidence_ambiguous:{item.task}:{len(identities)}"
        )
    return min(candidates, key=lambda candidate: str(candidate["manifest"]))


def _write_teacher_trace(
    *,
    demo_root: Path,
    teacher_source: dict[str, Any],
) -> None:
    source_rows = _jsonl(demo_root / "teacher_trace.jsonl")
    by_step = {
        (
            int(row.get("source_step_index") or 0),
            str(row.get("action_type") or ""),
        ): row
        for row in source_rows
    }
    rows: list[dict[str, Any]] = []
    for cursor, record in enumerate(teacher_source["actions"], 1):
        action = record["action"]
        action_type = str(action.get("type") or "")
        step_index = int(record.get("source_step_index") or 0)
        source = by_step.get((step_index, action_type), {})
        rows.append(
            {
                **source,
                "teacher_cursor": cursor,
                "source_step_index": step_index,
                "source_action_index": int(
                    record.get("source_action_index") or 0
                ),
                "action_type": action_type,
                "source_coordinates_used": False,
                "conversion_mode": "canonical_runlog_offline",
            }
        )
    (demo_root / "teacher_trace.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def validate_appagent_source_memory(
    *,
    index_path: str | Path,
    task_name: str,
    memory_root: str | Path,
    model: str,
) -> dict[str, Any]:
    item = load_canonical_source_item(index_path, task_name=task_name)
    source_method = _appagent_source_method_label(item)
    manifest = validate_appagent_demo_memory(
        memory_root,
        task_name=item.task,
        source_run_log=item.source_run_log,
    )
    if str(manifest.get("source_method") or "") != source_method:
        raise ValueError("appagent_demo_memory_source_method_invalid")
    if str(manifest.get("document_generation_model") or "") != str(model):
        raise ValueError("appagent_demo_memory_model_invalid")
    models = {
        str(value or "").strip()
        for value in (
            (manifest.get("doc_generation_usage") or {}).get("models") or []
        )
        if str(value or "").strip()
    }
    if models != {str(model)}:
        raise ValueError(
            "appagent_demo_memory_usage_model_invalid:"
            f"expected={model}:actual={sorted(models)}"
        )
    return manifest


def _preflight_appagent_teacher(
    *,
    item: pipeline.ArchivedRunLog,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    grounded, grounding_audit = build_grounded_teacher_run_log_from_canonical_item(
        item
    )
    with tempfile.TemporaryDirectory(
        prefix="omniflow-appagent-preflight-"
    ) as temporary:
        grounded_path = Path(temporary) / "grounded.run_log.json"
        grounded_path.write_text(
            json.dumps(grounded, ensure_ascii=False),
            encoding="utf-8",
        )
        teacher_source = build_appagent_teacher_source(
            grounded_path,
            task_name=item.task,
            source_seed=SOURCE_SEED,
            provenance_source_run_log=grounding_audit["source_run_log"],
        )
    grounded_steps = {
        int(step.get("step_index", index)): step
        for index, step in enumerate(grounded.get("steps") or [])
        if isinstance(step, dict)
    }
    groundable_action_count = 0
    for record in teacher_source.get("actions") or []:
        action = dict(record.get("action") or {})
        if str(action.get("type") or "").strip() not in APPAGENT_DEMO_ACTION_TYPES:
            continue
        step_index = int(record.get("source_step_index") or 0)
        step = grounded_steps.get(step_index) or {}
        observation = step.get("observation")
        metadata = step.get("metadata")
        source_context = (
            metadata.get("source_context")
            if isinstance(metadata, dict)
            else None
        )
        xml_text = str(
            (
                source_context.get("page")
                if isinstance(source_context, dict)
                else ""
            )
            or (observation_xml(observation) if isinstance(observation, dict) else "")
        ).strip()
        if not xml_text:
            raise ValueError(
                f"appagent_teacher_source_xml_missing:{step_index}"
            )
        ground_appagent_teacher_action(
            xml_text,
            action,
            min_dist=30.0,
        )
        groundable_action_count += 1
    if groundable_action_count != int(teacher_source["demo_action_count"]):
        raise ValueError("appagent_teacher_source_has_ungroundable_actions")
    grounding_audit["appagent_groundable_action_count"] = (
        groundable_action_count
    )
    return grounded, grounding_audit, teacher_source


def preflight_appagent_source(
    *,
    index_path: str | Path,
    task_name: str,
    evidence_roots: Sequence[str | Path] = (),
    model: str = "",
) -> dict[str, Any]:
    """Validate one source asset without creating a persistent output."""

    item = load_canonical_source_item(index_path, task_name=task_name)
    _, grounding_audit, teacher_source = _preflight_appagent_teacher(
        item=item,
    )
    evidence = None
    if evidence_roots:
        evidence = _native_memory_evidence(
            item=item,
            teacher_source=teacher_source,
            evidence_roots=evidence_roots,
            model=str(model or "").strip(),
        )
    return {
        "schema_version": "omniflow.appagent-source-preflight.v1",
        "task_name": item.task,
        "source_seed": SOURCE_SEED,
        "source_method": _appagent_source_method_label(item),
        "source_run_log": str(grounding_audit["source_run_log"]),
        "action_count": int(teacher_source["action_count"]),
        "demo_action_count": int(teacher_source["demo_action_count"]),
        "grounding": grounding_audit,
        "conversion_mode": "canonical_runlog_offline",
        "source_emulator_used": False,
        "native_memory_evidence": str(evidence["manifest"]) if evidence else None,
        "ready": True,
    }


def prepare_appagent_demo_memory(
    *,
    index_path: str | Path,
    task_name: str,
    appagent_root: str | Path,
    android_world_root: str | Path | None = None,
    memory_root: str | Path,
    model: str,
    evidence_roots: Sequence[str | Path] = (),
    serial: str = "emulator-5560",
    console_port: int = 5560,
    adb_path: str = "",
    timeout_sec: int = 600,
    request_timeout_sec: float = 60.0,
    perform_emulator_setup: bool = True,
) -> dict[str, Any]:
    """Convert one canonical RunLog to AppAgent's native demo memory."""

    normalized_model = str(model or "").strip()
    if not normalized_model:
        raise ValueError("appagent_source_model_required")
    source_environment_repair_reason = str(
        os.environ.get("OMNIFLOW_APPAGENT_SOURCE_ENVIRONMENT_REPAIR_REASON")
        or ""
    ).strip()
    item = load_canonical_source_item(index_path, task_name=task_name)
    source_method = _appagent_source_method_label(item)
    root = Path(memory_root).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"immutable_appagent_memory_exists:{root}")
    grounded_payload, grounding_audit, teacher_source = (
        _preflight_appagent_teacher(
            item=item,
        )
    )
    source_run_log = Path(grounding_audit["source_run_log"]).resolve()
    evidence = _native_memory_evidence(
        item=item,
        teacher_source=teacher_source,
        evidence_roots=evidence_roots,
        model=normalized_model,
    )

    root.mkdir(parents=True)
    prep_started = time.monotonic()
    grounded_source_path = root / "grounded_teacher_run_log.json"
    grounded_source_path.write_text(
        json.dumps(grounded_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    teacher_source_path = root / "teacher_source.json"
    teacher_source_path.write_text(
        json.dumps(teacher_source, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    app_name = str(evidence["app_name"])
    demo_name = str(evidence["demo_name"])
    target_demo = root / "apps" / app_name / "demos" / demo_name
    target_docs = root / "apps" / app_name / "demo_docs"
    shutil.copytree(evidence["demo_root"], target_demo)
    shutil.copytree(evidence["docs_root"], target_docs)
    _write_teacher_trace(
        demo_root=target_demo,
        teacher_source=teacher_source,
    )
    document_root = root / "_document_generation"
    document_root.mkdir()
    document_log = document_root / "document_generation.log"
    document_usage = document_root / "document_generation_usage.jsonl"
    shutil.copyfile(evidence["document_log"], document_log)
    shutil.copyfile(evidence["document_usage"], document_usage)
    conversion_audit = root / "conversion_audit.json"
    conversion_audit.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.appagent-runlog-conversion.v1",
                "task_name": item.task,
                "task_params": item.params,
                "source_seed": SOURCE_SEED,
                "source_method": source_method,
                "source_run_log": str(source_run_log),
                "source_run_log_sha256": pipeline._file_sha256(
                    source_run_log
                ),
                "grounded_teacher_run_log": str(grounded_source_path),
                "grounded_teacher_run_log_sha256": pipeline._file_sha256(
                    grounded_source_path
                ),
                "grounding_audit": grounding_audit,
                "model": normalized_model,
                "conversion_mode": "canonical_runlog_offline",
                "native_memory_evidence": str(evidence["manifest"]),
                "source_emulator_used": False,
                "source_environment_repair_reason": (
                    source_environment_repair_reason
                ),
                "target_inputs_read": False,
                "target_observations_read": False,
                "validator_state_read_for_memory": False,
                "coordinate_replay": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    prep_wall_sec = round(time.monotonic() - prep_started, 6)
    manifest = seal_appagent_demo_memory(
        memory_root=root,
        app_name=app_name,
        demo_name=demo_name,
        teacher_source=teacher_source_path,
        source_result=None,
        document_generation_log=document_log,
        document_generation_usage=document_usage,
        task_name=item.task,
        source_episode_wall_sec=0.0,
        document_generation_wall_sec=0.0,
        prep_wall_sec=prep_wall_sec,
        source_method=source_method,
        document_generation_model=str(evidence["document_model"]),
        source_environment_repair_reason=source_environment_repair_reason,
        conversion_mode="canonical_runlog_offline",
        native_memory_evidence=evidence["manifest"],
    )
    return {
        "schema_version": "omniflow.appagent-source-prepare.v3",
        "task_name": item.task,
        "source_seed": SOURCE_SEED,
        "source_method": source_method,
        "source_run_log": str(source_run_log),
        "model": normalized_model,
        "memory_root": str(root),
        "conversion_mode": "canonical_runlog_offline",
        "native_memory_evidence": str(evidence["manifest"]),
        "source_emulator_used": False,
        "source_environment_repair_reason": source_environment_repair_reason,
        "prep_wall_sec": prep_wall_sec,
        "manifest": manifest,
    }


def _write_failure_marker(memory_root: str | Path, error: BaseException) -> None:
    root = Path(memory_root).expanduser().resolve()
    if not root.is_dir() or (root / APPAGENT_DEMO_MANIFEST).exists():
        return
    marker = root / "prep_failure.json"
    if marker.exists():
        return
    marker.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.appagent-source-failure.v1",
                "failed_at": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
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
    prepare.add_argument("--appagent-root", required=True)
    prepare.add_argument("--android-world-root")
    prepare.add_argument("--memory-root", required=True)
    prepare.add_argument("--model", required=True)
    prepare.add_argument("--evidence-root", action="append", required=True)
    prepare.add_argument("--serial", default="emulator-5560")
    prepare.add_argument("--console-port", type=int, default=5560)
    prepare.add_argument("--adb-path", default="")
    prepare.add_argument("--timeout-sec", type=int, default=600)
    prepare.add_argument("--request-timeout-sec", type=float, default=60.0)
    prepare.add_argument("--no-emulator-setup", action="store_true")

    validate = subparsers.add_parser("validate")
    validate.add_argument("--index", required=True)
    validate.add_argument("--task", required=True)
    validate.add_argument("--memory-root", required=True)
    validate.add_argument("--model", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--index", required=True)
    preflight.add_argument("--task", required=True)
    preflight.add_argument("--model", default="")
    preflight.add_argument("--evidence-root", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "prepare":
            result = prepare_appagent_demo_memory(
                index_path=args.index,
                task_name=args.task,
                appagent_root=args.appagent_root,
                android_world_root=args.android_world_root,
                memory_root=args.memory_root,
                model=args.model,
                evidence_roots=args.evidence_root,
                serial=args.serial,
                console_port=args.console_port,
                adb_path=args.adb_path,
                timeout_sec=args.timeout_sec,
                request_timeout_sec=args.request_timeout_sec,
                perform_emulator_setup=not args.no_emulator_setup,
            )
        elif args.command == "validate":
            result = validate_appagent_source_memory(
                index_path=args.index,
                task_name=args.task,
                memory_root=args.memory_root,
                model=args.model,
            )
        else:
            result = preflight_appagent_source(
                index_path=args.index,
                task_name=args.task,
                evidence_roots=args.evidence_root,
                model=args.model,
            )
    except BaseException as error:
        if args.command == "prepare":
            _write_failure_marker(args.memory_root, error)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
