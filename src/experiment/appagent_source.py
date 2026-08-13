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
    validate_appagent_source_demo,
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


def _captured_app_name(memory_root: Path) -> str:
    apps_root = memory_root / "apps"
    app_dirs = (
        sorted(path for path in apps_root.iterdir() if path.is_dir())
        if apps_root.is_dir()
        else []
    )
    if len(app_dirs) != 1:
        raise ValueError(
            f"appagent_source_app_resolution_failed:count={len(app_dirs)}"
        )
    return app_dirs[0].name


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
) -> dict[str, Any]:
    """Validate one source asset without creating a persistent output."""

    item = load_canonical_source_item(index_path, task_name=task_name)
    _, grounding_audit, teacher_source = _preflight_appagent_teacher(
        item=item,
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
        "ready": True,
    }


def prepare_appagent_demo_memory(
    *,
    index_path: str | Path,
    task_name: str,
    appagent_root: str | Path,
    android_world_root: str | Path,
    memory_root: str | Path,
    model: str,
    serial: str = "emulator-5560",
    console_port: int = 5560,
    adb_path: str = "",
    timeout_sec: int = 600,
    request_timeout_sec: float = 60.0,
    perform_emulator_setup: bool = True,
) -> dict[str, Any]:
    """Capture, document, audit, and freeze one AppAgent source memory."""

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
    demo_name = f"demo_{item.task}_seed{SOURCE_SEED}"
    target = pipeline.DeviceTarget(
        label=f"source{int(console_port)}",
        serial=str(serial),
        console_port=int(console_port),
    )
    source_spec = pipeline.build_appagent_androidworld_command(
        item,
        method_name="appagent_native_source_demo",
        target=target,
        android_world_root=android_world_root,
        output_root=root / "_source_episode",
        appagent_root=appagent_root,
        teacher_source=teacher_source_path,
        workspace_root=root,
        demo_name=demo_name,
        max_steps=int(teacher_source["action_count"]) + 1,
        timeout_sec=int(timeout_sec),
        task_random_seed=SOURCE_SEED,
        fixed_task_seed=True,
        fixed_task_params=True,
        task_params_override=dict(item.params),
        perform_emulator_setup=bool(perform_emulator_setup),
        adb_path=str(adb_path),
    )
    (root / "source_episode_command.json").write_text(
        json.dumps(
            {
                "schema_version": "omniflow.appagent-source-command.v2",
                "command": pipeline._command_line(source_spec),
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
                "serial": str(serial),
                "model": normalized_model,
                "model_attempts": 1,
                "episode_retries": 0,
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

    source_started = time.monotonic()
    returncode = pipeline.run_command(source_spec)
    source_wall_sec = round(time.monotonic() - source_started, 6)
    if returncode != 0:
        raise RuntimeError(f"appagent_source_episode_failed:{returncode}")
    if source_spec.output_path is None:
        raise RuntimeError("appagent_source_episode_output_missing")
    source_result = source_spec.output_path / "task_results.jsonl"
    app_name = _captured_app_name(root)
    validate_appagent_source_demo(
        memory_root=root,
        app_name=app_name,
        demo_name=demo_name,
        source_result=source_result,
        task_name=item.task,
        expected_teacher_action_count=int(teacher_source["action_count"]),
        expected_demo_action_count=int(teacher_source["demo_action_count"]),
    )

    document_root = root / "_document_generation"
    doc_result = run_official_document_generation(
        appagent_root=appagent_root,
        workspace_root=root,
        app_name=app_name,
        demo_name=demo_name,
        log_path=document_root / "document_generation.log",
        usage_path=document_root / "document_generation_usage.jsonl",
        model=normalized_model,
        timeout_sec=float(request_timeout_sec),
    )
    if set(doc_result["models"]) != {normalized_model}:
        raise ValueError(
            "appagent_document_model_mismatch:"
            f"expected={normalized_model}:actual={doc_result['models']}"
        )
    prep_wall_sec = round(time.monotonic() - prep_started, 6)
    manifest = seal_appagent_demo_memory(
        memory_root=root,
        app_name=app_name,
        demo_name=demo_name,
        teacher_source=teacher_source_path,
        source_result=source_result,
        document_generation_log=doc_result["log_path"],
        document_generation_usage=doc_result["usage_path"],
        task_name=item.task,
        source_episode_wall_sec=source_wall_sec,
        document_generation_wall_sec=float(doc_result["wall_sec"]),
        prep_wall_sec=prep_wall_sec,
        source_method=source_method,
        document_generation_model=normalized_model,
        source_environment_repair_reason=source_environment_repair_reason,
    )
    return {
        "schema_version": "omniflow.appagent-source-prepare.v2",
        "task_name": item.task,
        "source_seed": SOURCE_SEED,
        "source_method": source_method,
        "source_run_log": str(source_run_log),
        "model": normalized_model,
        "memory_root": str(root),
        "source_result": str(source_result),
        "source_episode_wall_sec": source_wall_sec,
        "source_environment_repair_reason": source_environment_repair_reason,
        "document_generation": doc_result,
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
    prepare.add_argument("--android-world-root", required=True)
    prepare.add_argument("--memory-root", required=True)
    prepare.add_argument("--model", required=True)
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
            )
    except BaseException as error:
        if args.command == "prepare":
            _write_failure_marker(args.memory_root, error)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
