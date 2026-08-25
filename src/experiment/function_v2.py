"""Experiment-side ownership for OmniFlow v2 Function assets."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from omniflow.functions.compiler import compile_runlog_to_store
from omniflow.runlog import import_run_log_evidence
from omniflow.vlm.model_config import resolve_openai_compatible_config


def compile_function_v2(
    run_log: str | Path | dict[str, Any],
    output_root: str | Path,
    *,
    enhance: bool,
    model: str = "",
    model_endpoint_profile: str = "auto",
    model_base_url: str = "",
    timeout: float = 120.0,
    authoring_trace: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compile one source RunLog with the native v2 compiler."""

    root = Path(output_root).expanduser().resolve()
    source_states: Path | dict[str, Any]
    if isinstance(run_log, (str, Path)):
        source_path = Path(run_log).expanduser().resolve()
        source_catalog = source_path.with_name("transfer_states.json")
        if source_catalog.is_file():
            source_states = source_catalog
        else:
            raw = json.loads(source_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("source_runlog_must_be_object")
            _, source_states = import_run_log_evidence(raw)
    else:
        raw_states = run_log.get("transfer_states")
        if isinstance(raw_states, dict):
            source_states = raw_states
        else:
            _, source_states = import_run_log_evidence(run_log)

    options: dict[str, Any] = {
        "source_states": source_states,
        "timeout": float(timeout),
    }
    if enhance:
        selected_model = str(model or "").strip()
        if not selected_model:
            raise ValueError("function_author_model_required")
        options["model"] = selected_model
        from openai import OpenAI
        import httpx

        api_key, base_url = resolve_openai_compatible_config(
            profile=model_endpoint_profile,
            base_url=model_base_url,
        )
        options["client"] = OpenAI(
            api_key=api_key or "not-required",
            base_url=base_url,
            max_retries=0,
            http_client=httpx.Client(trust_env=False),
        )

    client = options.get("client")
    try:
        report = compile_runlog_to_store(run_log, root, **options)
    finally:
        if client is not None:
            client.close()
    if isinstance(run_log, (str, Path)):
        source_path = Path(run_log).expanduser().resolve()
        shutil.copy2(source_path, root / "run_log.json")
        source_screenshots = source_path.parent / "screenshots"
        if source_screenshots.is_dir():
            shutil.copytree(source_screenshots, root / "screenshots")
        else:
            source_observations = source_path.parent / "observations" / "objects"
            if source_observations.is_dir():
                shutil.copytree(source_observations, root / "screenshots")
    report["enhanced"] = bool(enhance)
    if authoring_trace is not None:
        authoring_trace.append(
            {
                "schema_version": "omniflow.function-authoring-event.v1",
                "compiler": "compile_runlog_to_store",
                "function_schema": "omniflow.function.v2",
                "store_path": report["store_path"],
                "function_ids": list(report.get("function_ids") or ()),
                "model": report.get("model"),
            }
        )
    return report


def load_v2_source_calls(store_path: str | Path) -> list[dict[str, Any]]:
    """Load ordered source invocations from old or new v2 compiler reports."""

    store = Path(store_path).expanduser().resolve()
    report_path = store.with_name("compile_report.json")
    if not report_path.is_file():
        raise FileNotFoundError(f"function_compile_report_missing:{report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    raw_calls = report.get("source_calls")
    if raw_calls is not None:
        if not isinstance(raw_calls, list) or not raw_calls:
            raise ValueError("function_compile_source_calls_invalid")
        calls: list[dict[str, Any]] = []
        for call in raw_calls:
            if not isinstance(call, dict) or set(call) != {
                "function_id",
                "arguments",
            }:
                raise ValueError("function_compile_source_calls_invalid")
            function_id = str(call.get("function_id") or "").strip()
            arguments = call.get("arguments")
            if not function_id or not isinstance(arguments, dict):
                raise ValueError("function_compile_source_calls_invalid")
            calls.append(
                {"function_id": function_id, "arguments": dict(arguments)}
            )
        if len({call["function_id"] for call in calls}) != len(calls):
            raise ValueError("function_compile_source_calls_invalid")
        return calls

    # Older collectors persisted only this insertion-ordered mapping. Keep it
    # readable because experiment archives contain both report variants.
    arguments_by_function = report.get("source_arguments")
    if not isinstance(arguments_by_function, dict):
        raise ValueError("function_compile_source_arguments_invalid")
    calls = [
        {"function_id": str(function_id), "arguments": dict(arguments)}
        for function_id, arguments in arguments_by_function.items()
        if str(function_id).strip() and isinstance(arguments, dict)
    ]
    if len(calls) != len(arguments_by_function):
        raise ValueError("function_compile_source_arguments_invalid")
    return calls


__all__ = ["compile_function_v2", "load_v2_source_calls"]
