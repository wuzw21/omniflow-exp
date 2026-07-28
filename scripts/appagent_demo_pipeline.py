#!/usr/bin/env python3
"""Run and seal pinned AppAgent human-demonstration documentation generation."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.integrations.appagent_adapter import (
    APPAGENT_OFFICIAL_REVISION,
    OfficialAppAgentRuntime,
    build_appagent_teacher_source,
    seal_appagent_demo_memory,
    validate_appagent_source_demo,
)


def run_official_document_generation(
    *,
    appagent_root: str | Path,
    workspace_root: str | Path,
    app_name: str,
    demo_name: str,
    log_path: str | Path,
    usage_path: str | Path,
    model: str = "",
) -> dict[str, Any]:
    started = time.monotonic()
    root = Path(appagent_root).expanduser().resolve()
    workspace = Path(workspace_root).expanduser().resolve()
    output_log = Path(log_path).expanduser().resolve()
    output_usage = Path(usage_path).expanduser().resolve()
    OfficialAppAgentRuntime(root)
    api_key = str(
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY")
        or ""
    ).strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for AppAgent docs")
    model_name = str(
        model
        or os.environ.get("OPENAI_MODEL")
        or os.environ.get("OMNIFLOW_PLANNER_MODEL")
        or "qwen3-vl-plus"
    ).strip()
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

    with tempfile.TemporaryDirectory(prefix="appagent-doc-config-") as temporary:
        temporary_root = Path(temporary)
        config_path = temporary_root / "config.yaml"
        _write_runtime_config(
            config_path,
            api_key=api_key,
            endpoint=endpoint,
            model=model_name,
        )
        previous_cwd = Path.cwd()
        scripts_dir = root / "scripts"
        previous_argv = list(sys.argv)
        sys.path.insert(0, str(scripts_dir))
        try:
            os.chdir(temporary_root)
            official_model = importlib.import_module("model")
            original_post = official_model.requests.post

            def instrumented_post(*args: Any, **kwargs: Any) -> Any:
                response = original_post(*args, **kwargs)
                payload = response.json()
                usage = payload.get("usage") if isinstance(payload, dict) else None
                usage = dict(usage or {}) if isinstance(usage, dict) else {}
                record = {
                    "model": (
                        payload.get("model") if isinstance(payload, dict) else None
                    )
                    or model_name,
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

            official_model.requests.post = instrumented_post
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
            sys.argv = previous_argv
            os.chdir(previous_cwd)
            try:
                sys.path.remove(str(scripts_dir))
            except ValueError:
                pass
    if not output_usage.exists():
        output_usage.touch()
    docs_root = workspace / "apps" / str(app_name) / "demo_docs"
    docs_count = len([path for path in docs_root.glob("*.txt") if path.is_file()])
    usage_rows = [
        json.loads(line)
        for line in output_usage.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        "official_appagent_revision": APPAGENT_OFFICIAL_REVISION,
        "app_name": str(app_name),
        "demo_name": str(demo_name),
        "docs_root": str(docs_root),
        "docs_count": docs_count,
        "model_calls": len(usage_rows),
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in usage_rows),
        "completion_tokens": sum(
            int(row.get("completion_tokens") or 0) for row in usage_rows
        ),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in usage_rows),
        "wall_sec": round(time.monotonic() - started, 6),
        "log_path": str(output_log),
        "usage_path": str(output_usage),
    }


def prepare_appagent_demo_memory(
    *,
    appagent_root: str | Path,
    android_world_root: str | Path,
    memory_root: str | Path,
    source_run_log: str | Path,
    task_name: str,
    app_name: str,
    task_params: dict[str, Any],
    serial: str,
    console_port: int,
    adb_path: str = "",
    model: str = "",
    perform_emulator_setup: bool = True,
) -> dict[str, Any]:
    prepare_started = time.monotonic()
    root = Path(memory_root).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"immutable_appagent_memory_exists:{root}")
    root.mkdir(parents=True)
    teacher_source_path = root / "teacher_source.json"
    teacher_source = build_appagent_teacher_source(
        source_run_log,
        task_name=task_name,
        source_seed=111,
    )
    teacher_source_path.write_text(
        json.dumps(teacher_source, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    demo_name = f"demo_{task_name}_seed111"
    source_output = root / "_source_episode"
    source_log = root / "_source_episode.log"
    command = [
        sys.executable,
        "-m",
        "src.integrations.android_world.launch",
        "--android-world-root",
        str(Path(android_world_root).expanduser().resolve()),
        "--tasks",
        str(task_name),
        "--task-random-seed",
        "111",
        "--n-task-combinations",
        "1",
        "--console-port",
        str(int(console_port)),
        "--agent",
        "external:appagent_teacher",
        "--max-steps",
        str(int(teacher_source["action_count"]) + 1),
        "--output-path",
        str(source_output),
        "--task-params-json",
        json.dumps(task_params, ensure_ascii=False, separators=(",", ":")),
        "--fixed-task-seed",
        "--oob-observe-backend",
        "androidworld",
        "--appagent-root",
        str(Path(appagent_root).expanduser().resolve()),
        "--appagent-workspace-root",
        str(root),
        "--appagent-teacher-source",
        str(teacher_source_path),
        "--appagent-demo-name",
        demo_name,
    ]
    if perform_emulator_setup:
        command.append("--perform-emulator-setup")
    if str(adb_path or "").strip():
        command.extend(["--adb-path", str(adb_path).strip()])
    (root / "source_episode_command.json").write_text(
        json.dumps(
            {
                "command": command,
                "android_serial": str(serial),
                "source_seed": 111,
                "target_inputs_read": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    source_environment = dict(os.environ)
    source_environment["ANDROID_SERIAL"] = str(serial)
    source_started = time.monotonic()
    with source_log.open("x", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=source_environment,
            text=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    source_episode_wall_sec = time.monotonic() - source_started
    if completed.returncode != 0:
        raise RuntimeError(
            f"appagent_source_episode_failed:{completed.returncode}:{source_log}"
        )
    validate_appagent_source_demo(
        memory_root=root,
        app_name=app_name,
        demo_name=demo_name,
        source_result=source_output / "task_results.jsonl",
        task_name=task_name,
        expected_action_count=int(teacher_source["action_count"]),
    )
    document_root = root / "_document_generation"
    document_log = document_root / "document_generation.log"
    document_usage = document_root / "document_generation_usage.jsonl"
    doc_result = run_official_document_generation(
        appagent_root=appagent_root,
        workspace_root=root,
        app_name=app_name,
        demo_name=demo_name,
        log_path=document_log,
        usage_path=document_usage,
        model=model,
    )
    manifest = seal_appagent_demo_memory(
        memory_root=root,
        app_name=app_name,
        demo_name=demo_name,
        teacher_source=teacher_source_path,
        source_result=source_output / "task_results.jsonl",
        document_generation_log=document_log,
        document_generation_usage=document_usage,
        task_name=task_name,
        source_episode_wall_sec=source_episode_wall_sec,
        document_generation_wall_sec=float(doc_result["wall_sec"]),
        prep_wall_sec=time.monotonic() - prepare_started,
    )
    return {
        "memory_root": str(root),
        "source_episode_output": str(source_output),
        "document_generation": doc_result,
        "manifest": manifest,
    }


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
) -> None:
    import yaml

    payload = {
        "MODEL": "OpenAI",
        "OPENAI_API_BASE": endpoint,
        "OPENAI_API_KEY": api_key,
        "OPENAI_API_MODEL": model,
        "MAX_TOKENS": int(os.environ.get("APPAGENT_DOC_MAX_TOKENS") or 300),
        "TEMPERATURE": 0.0,
        "REQUEST_INTERVAL": float(
            os.environ.get("APPAGENT_DOC_REQUEST_INTERVAL") or 0.0
        ),
        "DASHSCOPE_API_KEY": "",
        "QWEN_MODEL": model,
        "ANDROID_SCREENSHOT_DIR": "/sdcard",
        "ANDROID_XML_DIR": "/sdcard",
        "DOC_REFINE": False,
        "MAX_ROUNDS": 20,
        "DARK_MODE": False,
        "MIN_DIST": int(os.environ.get("APPAGENT_MIN_DIST") or 30),
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate-docs")
    generate.add_argument("--appagent-root", required=True)
    generate.add_argument("--workspace-root", required=True)
    generate.add_argument("--app", required=True)
    generate.add_argument("--demo", required=True)
    generate.add_argument("--log", required=True)
    generate.add_argument("--usage", required=True)
    generate.add_argument("--model", default="")

    seal = subparsers.add_parser("seal")
    seal.add_argument("--memory-root", required=True)
    seal.add_argument("--app", required=True)
    seal.add_argument("--demo", required=True)
    seal.add_argument("--teacher-source", required=True)
    seal.add_argument("--source-result", required=True)
    seal.add_argument("--document-generation-log", required=True)
    seal.add_argument("--document-generation-usage", required=True)
    seal.add_argument("--task", required=True)
    seal.add_argument("--source-episode-wall-sec", type=float, required=True)
    seal.add_argument("--document-generation-wall-sec", type=float, required=True)
    seal.add_argument("--prep-wall-sec", type=float, required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--appagent-root", required=True)
    prepare.add_argument("--android-world-root", required=True)
    prepare.add_argument("--memory-root", required=True)
    prepare.add_argument("--source-run-log", required=True)
    prepare.add_argument("--task", required=True)
    prepare.add_argument("--app", required=True)
    prepare.add_argument("--task-params-json", required=True)
    prepare.add_argument("--serial", default="emulator-5554")
    prepare.add_argument("--console-port", type=int, default=5554)
    prepare.add_argument("--adb-path", default="")
    prepare.add_argument("--model", default="")
    prepare.add_argument("--no-emulator-setup", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "generate-docs":
        result = run_official_document_generation(
            appagent_root=args.appagent_root,
            workspace_root=args.workspace_root,
            app_name=args.app,
            demo_name=args.demo,
            log_path=args.log,
            usage_path=args.usage,
            model=args.model,
        )
    elif args.command == "seal":
        result = seal_appagent_demo_memory(
            memory_root=args.memory_root,
            app_name=args.app,
            demo_name=args.demo,
            teacher_source=args.teacher_source,
            source_result=args.source_result,
            document_generation_log=args.document_generation_log,
            document_generation_usage=args.document_generation_usage,
            task_name=args.task,
            source_episode_wall_sec=args.source_episode_wall_sec,
            document_generation_wall_sec=args.document_generation_wall_sec,
            prep_wall_sec=args.prep_wall_sec,
        )
    else:
        task_params = json.loads(args.task_params_json)
        if not isinstance(task_params, dict):
            raise ValueError("--task-params-json must decode to an object")
        result = prepare_appagent_demo_memory(
            appagent_root=args.appagent_root,
            android_world_root=args.android_world_root,
            memory_root=args.memory_root,
            source_run_log=args.source_run_log,
            task_name=args.task,
            app_name=args.app,
            task_params=task_params,
            serial=args.serial,
            console_port=args.console_port,
            adb_path=args.adb_path,
            model=args.model,
            perform_emulator_setup=not args.no_emulator_setup,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
