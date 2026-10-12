#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field, is_dataclass, replace
import datetime
import json
import os
from pathlib import Path
import random
import shlex
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiment.androidworld_paths import (
    canonical_device_seed_name,
    canonical_method_name,
)
from src.experiment.mobilegpt_contract import (
    MOBILEGPT_EMBEDDING_MODEL,
    MOBILEGPT_MEMORY_MANIFEST,
)
from src.experiment.paths import (
    resolve_path,
    resolve_relative_reference,
    safe_relative_path,
    sha256_file,
)
from src.experiment.protocol import (
    ANDROIDWORLD_REVISION,
    APPAGENT_MODEL,
    AUTODROID_MEMORY_METHOD,
    FORMAL_THINKING,
    DEFAULT_DEVICE,
    DEFAULT_METHOD,
    DEVICES,
    ENABLED_METHODS,
    FORMAL_MODEL,
    FORMAL_MODEL_BASE_URL,
    FORMAL_MODEL_ENDPOINT_PROFILE,
    FORMAL_MAX_TOKENS,
    FORMAL_APPAGENT_EMPTY_RESPONSE_RETRIES,
    FORMAL_APPAGENT_RETRY_MAX_TOKENS,
    FORMAL_APPAGENT_TIMEOUT_SEC,
    FORMAL_REQUEST_TIMEOUT_SEC,
    FORMAL_RETRY_WAIT_SEC,
    MAX_FALLBACK_STEPS,
    MAX_STEPS,
    METHODS,
    PLANNER_TIMEOUT_SEC,
    SOURCE_MAX_STEPS,
    SOURCE_METHOD,
    SOURCE_DEVICE,
    SOURCE_SEED,
    TASK_DEADLINE_SEC,
    TASK_SEED,
    omniflow_base_url,
    omniflow_endpoint_profile,
    omniflow_model,
    require_runtime_model,
)
from src.experiment.run_process import run_process, start_process, stop_process
from src.experiment.source_records import CanonicalRunLog, SourceRunLogProfile
from src.integrations import mobilegpt_memory
from src.integrations.android_world.apps import resolve_androidworld_package
from src.integrations.autodroid_memory import validate_autodroid_memory

# Mechanically extracted modules; names re-exported for existing importers.
from src.experiment.run_task_common import (  # noqa: F401
    DeviceTarget,
    _canonical_steps_from_cards,
    _card_args,
    _card_tool_name,
    _coerce_float,
    _coerce_int,
    _drop_empty_input_text_cards,
    _is_empty_input_text_card,
    _last_jsonl_object,
    _normalize_card_source_context,
    _normalize_observation_payload,
    _read_json,
    _safe_stem,
    canonicalize_source_run_log,
    profile_source_run_log,
)
from src.experiment.source_hints import (  # noqa: F401
    _source_action_hint_path_for_item,
    _t3a_hint_bounds,
    _t3a_hint_forbidden_values,
    _t3a_hint_redacted_text,
    _t3a_hint_source_node,
    _t3a_hint_step_action,
    _t3a_hint_step_summary,
    _t3a_hint_summary_target,
    _t3a_hint_target,
    _t3a_hint_text,
    _t3a_semantic_hint_step,
)
from src.experiment.result_report import (  # noqa: F401
    _compact_report_value,
    _execution_audit,
    _percentage_text,
    _result_device_name,
    _task_failure_attribution,
    format_omniflow_result_block,
)
from src.experiment.mobilegpt_memory import (  # noqa: F401
    _MOBILEGPT_IGNORED_TARGET_PACKAGES,
    _infer_mobilegpt_target_from_source_run_log,
    _mobilegpt_action_package,
    _mobilegpt_observation_package,
    seal_mobilegpt_converted_memory,
)

DEFAULT_ANDROID_WORLD_ROOT = Path(
    os.environ.get("OMNIFLOW_ANDROID_WORLD_ROOT")
    or Path("..") / "releases" / f"android-world-{ANDROIDWORLD_REVISION}"
)
DEFAULT_OUTPUT_ROOT = (
    Path("data") / "androidworld"
)
DEFAULT_MOBILEGPT_ROOT = Path("data") / "runtime" / "external" / "mobilegpt"
DEFAULT_MOBILEGPT_STATS_JSONL = (
    DEFAULT_OUTPUT_ROOT / "_mobilegpt_stats" / "mobilegpt_stats.jsonl"
)
DEFAULT_MOBILEGPT_EPISODE_WAIT_TIMEOUT_SEC = 300.0
DEFAULT_MOBILEGPT_APP_READY_TIMEOUT_SEC = 15.0
DEFAULT_MOBILEGPT_APP_READY_POLL_SEC = 0.25
DEFAULT_SOURCE_METHOD = DEFAULT_METHOD


def _mobilegpt_server_port(console_port: int) -> int:
    """Derive one stable, isolated MobileGPT listener port per configured AVD."""

    if int(console_port) == 5560:
        return 12345
    return 12000 + int(console_port) % 40000


@dataclass(frozen=True)
class CommandSpec:
    label: str
    argv: list[str]
    env: dict[str, str]
    cwd: Path
    output_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timeout_sec: float | None = None
    stdin_text: str | None = None


def _json_default(value: Any) -> Any:
    """Serialize AndroidWorld parameter dataclasses for subprocess JSON.

    Task parameters loaded from AndroidWorld can contain frozen SQLite-row
    dataclasses (for example CalendarEvent, Expense, and Recipe).  They are
    valid task parameters but are not directly serializable by ``json``.
    Preserve their field structure instead of stringifying them so the target
    task receives the exact parameters generated for its evaluation seed.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    # A few AndroidWorld task parameters (notably receipt-image tasks) carry
    # an in-memory PIL Image.  The MobileGPT subprocess only needs a JSON
    # representation of the task parameters for goal/context construction;
    # the official task instance retains the real image on the parent side.
    # Preserve a stable descriptive value instead of aborting before the
    # episode starts with ``Image is not JSON serializable``.
    if value.__class__.__name__ == "Image" and value.__class__.__module__.startswith("PIL"):
        return str(value)
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def _local_dotenv_env(*, repo_root: Path = REPO_ROOT) -> dict[str, str]:
    dotenv_path = repo_root / ".env"
    if not dotenv_path.exists():
        return {}
    try:
        from dotenv import dotenv_values

        values = dotenv_values(dotenv_path)
    except Exception:
        values = {}
        for line in dotenv_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            text = line.strip()
            if not text or text.startswith("#") or "=" not in text:
                continue
            key, value = text.split("=", 1)
            values[key.strip()] = value.strip().strip("'\"")
    env = {
        str(key): str(value)
        for key, value in values.items()
        if key and value is not None and str(value).strip() != ""
    }
    env["MOBILEGPT_CHAT_MODEL"] = FORMAL_MODEL
    return env


def _resolve_planner_provider_and_model(
    planner_provider: str = "",
    model: str = "",
    *,
    repo_root: Path = REPO_ROOT,
) -> tuple[str, str]:
    dotenv_env = _local_dotenv_env(repo_root=repo_root)
    resolved_model = str(model or "").strip()
    if not resolved_model:
        resolved_model = FORMAL_MODEL

    resolved_provider = str(planner_provider or "").strip()
    if not resolved_provider:
        has_openai_config = any(
            str(dotenv_env.get(key) or os.environ.get(key) or "").strip()
            for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL")
        )
        if resolved_model or has_openai_config:
            resolved_provider = "openai"
    return resolved_provider, resolved_model


def _subprocess_env(
    spec_env: dict[str, str] | None = None,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, str]:
    dotenv_env = _local_dotenv_env(repo_root=repo_root)
    env = {**dotenv_env, **dict(os.environ), **dict(spec_env or {})}
    if env.get("MOBILEGPT_CLIENT_MODE") == "upstream_accessibility":
        # Native MobileGPT owns its models and provider configuration.
        # Do not alias a different provider's key to OpenAI implicitly.
        return env
    # The pinned MobileGPT Server uses the OpenAI-compatible client API for
    # both chat and embedding calls. Publish the canonical aliases once at the
    # common subprocess boundary so execution and memory authoring use the
    # same endpoint contract.
    if not str(env.get("OPENAI_API_KEY") or "").strip() and str(
        env.get("LLMTHU_API_KEY") or ""
    ).strip():
        env["OPENAI_API_KEY"] = str(env["LLMTHU_API_KEY"])
    experimental_model = str(
        env.get("OMNIFLOW_EXPERIMENTAL_MODEL") or ""
    ).strip()
    if experimental_model:
        # Experimental OmniFlow calls still use the common OpenAI-compatible
        # seam.  The explicit opt-in is inherited by the episode child, while
        # the formal five-method protocol remains pinned to Qwen by default.
        env["OPENAI_BASE_URL"] = omniflow_base_url()
        env["OPENAI_MODEL"] = experimental_model
        env["OMNIFLOW_PLANNER_MODEL"] = experimental_model
        env["OMNIFLOW_MODEL_ENDPOINT_PROFILE"] = omniflow_endpoint_profile()
    else:
        env["OPENAI_BASE_URL"] = FORMAL_MODEL_BASE_URL
        env["OPENAI_MODEL"] = FORMAL_MODEL
    env["MOBILEGPT_CHAT_BASE_URL"] = FORMAL_MODEL_BASE_URL
    env["MOBILEGPT_EMBEDDING_BASE_URL"] = FORMAL_MODEL_BASE_URL
    env["MOBILEGPT_CHAT_MODEL"] = FORMAL_MODEL
    env["MOBILEGPT_EMBEDDING_MODEL"] = MOBILEGPT_EMBEDDING_MODEL
    env["MOBILEGPT_THINKING"] = FORMAL_THINKING
    env["MOBILEGPT_MAX_TOKENS"] = "2048"
    env["MOBILEGPT_LIST_MAX_TOKENS"] = "2048"
    env["MOBILEGPT_MAX_CHAT_CALLS"] = "64"
    env["MOBILEGPT_RESPONSE_RETRIES"] = "1"
    env["MOBILEGPT_REQUEST_TIMEOUT_SEC"] = "60"
    env["APPAGENT_THINKING"] = FORMAL_THINKING
    env["APPAGENT_MODEL_TIMEOUT_SEC"] = str(FORMAL_APPAGENT_TIMEOUT_SEC)
    env["APPAGENT_EMPTY_RESPONSE_RETRIES"] = str(
        FORMAL_APPAGENT_EMPTY_RESPONSE_RETRIES
    )
    env["APPAGENT_RETRY_MAX_TOKENS"] = str(FORMAL_APPAGENT_RETRY_MAX_TOKENS)
    if not experimental_model:
        env["OMNIFLOW_PLANNER_MODEL"] = FORMAL_MODEL
        env["OMNIFLOW_MODEL_ENDPOINT_PROFILE"] = FORMAL_MODEL_ENDPOINT_PROFILE
    if env.get("OMNIFLOW_ANDROIDWORLD_BASELINE") != "1":
        env["OMNIFLOW_PLANNER_TIMEOUT_SEC"] = str(PLANNER_TIMEOUT_SEC)
    env["OMNIFLOW_ANDROIDWORLD_LLM_MAX_TOKENS"] = str(FORMAL_MAX_TOKENS)
    if env.get("OMNIFLOW_ANDROIDWORLD_BASELINE") != "1":
        env["OMNIFLOW_OPENAI_TIMEOUT_SEC"] = str(FORMAL_REQUEST_TIMEOUT_SEC)
    env["OMNIFLOW_OPENAI_RETRY_WAIT_SECONDS"] = str(FORMAL_RETRY_WAIT_SEC)
    env.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")
    return env


def _safe_relative_path(value: str, *, fallback: str = "run") -> Path:
    return safe_relative_path(value, fallback=fallback)


def _device_label(
    *,
    explicit_label: str = "",
    serial: str = "",
    console_port: int | None = None,
) -> str:
    label = str(explicit_label or "").strip()
    if label:
        return _safe_stem(label, fallback="device")
    port = int(console_port or 0)
    text = str(serial or "").strip()
    if text.endswith("5556") or port == 5556:
        return "source5556"
    if text.endswith("5554") or port == 5554:
        return "target5554"
    if text:
        return _safe_stem(text, fallback="device")
    return f"device{port}" if port > 0 else "device"


def _experiment_run_dir(
    output_root: str | Path,
    *,
    task: str,
    method: str,
    device: str = "",
    serial: str = "",
    console_port: int | None = None,
    evaluation_seed: int = TASK_SEED,
    attempt_id: str = "",
    repo_root: Path = REPO_ROOT,
) -> Path:
    path = (
        resolve_path(output_root, root=repo_root)
        / _safe_stem(task)
        / canonical_method_name(method)
        / canonical_device_seed_name(
            label=device,
            serial=serial,
            console_port=console_port,
            source_seed=SOURCE_SEED,
            evaluation_seed=int(evaluation_seed),
        )
    )
    batch_attempt = str(
        attempt_id or os.environ.get("OMNIFLOW_BATCH_ATTEMPT_ID") or ""
    ).strip()
    if batch_attempt:
        path = path / "runlog" / _safe_relative_path(batch_attempt, fallback="run")
    return path


def _task_managed_output_root(
    output_root: str | Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> tuple[Path, str]:
    """Return the caller's temporary working directory."""
    resolved = resolve_path(output_root, root=repo_root)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved, ""


def _source_seed_output_root(output_root: str | Path, source_seed: int) -> Path:
    return resolve_path(output_root) / f"source_seed_{int(source_seed)}"


def _read_object(path: Path) -> dict[str, Any]:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return decoded


def materialize_replay_run_log(
    item: CanonicalRunLog,
    *,
    output_root: str | Path,
    repo_root: Path = REPO_ROOT,
    normalize_wrapped_payload: bool = True,
) -> tuple[Path, str, SourceRunLogProfile]:
    profile = profile_source_run_log(item)
    if not normalize_wrapped_payload or profile.replay_format != "payload_cards":
        return item.source_run_log, "original", profile

    data = _read_json(item.source_run_log)
    payload = dict(data.get("payload") or {})
    payload.setdefault("schema_version", "oob.run_log.cards.v1")
    payload.setdefault(
        "run_id", str(data.get("run_id") or item.meta.get("run_id") or item.task)
    )
    payload.setdefault("goal", item.goal)
    payload.setdefault("success", bool(item.meta.get("androidworld_success", True)))
    payload.setdefault("androidworld_task", item.task)
    if item.params and not isinstance(payload.get("androidworld_params"), dict):
        payload["androidworld_params"] = dict(item.params)
    empty_input_text_cards_removed = _drop_empty_input_text_cards(payload)

    normalized_dir = (
        resolve_path(output_root, root=repo_root) / "_normalized_runlogs"
    )
    normalized_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = normalized_dir / f"{_safe_stem(item.task)}.run_log.json"
    normalized_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    materialization = "payload_unwrapped"
    if empty_input_text_cards_removed:
        materialization += "+empty_input_text_noop"
    return normalized_path.resolve(), materialization, profile


def _copy_replay_run_log_to_memory(
    source_run_log: str | Path,
    *,
    output_root: str | Path,
    task: str,
    repo_root: Path = REPO_ROOT,
) -> Path:
    replay_dir = resolve_path(output_root, root=repo_root) / "_replay_runlogs"
    replay_dir.mkdir(parents=True, exist_ok=True)
    source = resolve_path(source_run_log, root=repo_root)
    destination = replay_dir / f"{_safe_stem(task)}.run_log.json"
    if source.resolve() != destination.resolve():
        destination.write_bytes(source.read_bytes())
    return destination.resolve()


def _materialize_replay_run_log_for_memory(
    item: CanonicalRunLog,
    *,
    memory_root: str | Path,
    repo_root: Path = REPO_ROOT,
    normalize_wrapped_payload: bool = True,
) -> tuple[Path, str, SourceRunLogProfile]:
    replay_run_log, source_materialization, profile = materialize_replay_run_log(
        item,
        output_root=memory_root,
        repo_root=repo_root,
        normalize_wrapped_payload=normalize_wrapped_payload,
    )
    if source_materialization == "original":
        replay_run_log = _copy_replay_run_log_to_memory(
            replay_run_log,
            output_root=memory_root,
            task=item.task,
            repo_root=repo_root,
        )
    return replay_run_log, source_materialization, profile


def build_replay_command(
    item: CanonicalRunLog,
    *,
    android_world_root: str | Path = DEFAULT_ANDROID_WORLD_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    replay_memory_root: str | Path | None = None,
    method_name: str = DEFAULT_SOURCE_METHOD,
    device_label: str = "",
    serial: str = "",
    console_port: int = 5554,
    adb_path: str = "",
    max_steps: int | None = None,
    timeout_sec: int = 180,
    task_random_seed: int | None = None,
    fixed_task_params: bool = True,
    task_params_override: dict[str, Any] | None = None,
    set_datetime: bool = False,
    perform_emulator_setup: bool = True,
    normalize_wrapped_payload: bool = True,
    planner_provider: str = "",
    model: str = "",
    archive_attempt_id: str = "",
    python_executable: str = sys.executable,
    repo_root: Path = REPO_ROOT,
) -> CommandSpec:
    resolved_device = _device_label(
        explicit_label=device_label,
        serial=serial,
        console_port=console_port,
    )
    resolved_method = _safe_stem(method_name, fallback=DEFAULT_SOURCE_METHOD)
    setting_root = _experiment_run_dir(
        output_root,
        task=item.task,
        method=resolved_method,
        device=resolved_device,
        serial=serial,
        console_port=console_port,
        attempt_id=archive_attempt_id,
        repo_root=repo_root,
    )
    resolved_output = _next_attempt(setting_root, "runlog")
    if replay_memory_root:
        replay_run_log, source_materialization, profile = (
            _materialize_replay_run_log_for_memory(
                item,
                memory_root=replay_memory_root,
                repo_root=repo_root,
                normalize_wrapped_payload=normalize_wrapped_payload,
            )
        )
    else:
        replay_run_log, source_materialization, profile = materialize_replay_run_log(
            item,
            output_root=output_root,
            repo_root=repo_root,
            normalize_wrapped_payload=normalize_wrapped_payload,
        )
    step_budget = int(max_steps or max(5, item.step_count + 3))
    resolved_task_seed = int(
        item.replay_seed if task_random_seed is None else task_random_seed
    )
    effective_params = (
        dict(task_params_override) if task_params_override is not None else item.params
    )
    params_json = json.dumps(
        effective_params,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_default,
    )
    raw_replay_result = resolved_output / "raw_replay_result.json"
    env = {
        "OMNIFLOW_ANDROIDWORLD_SET_DATETIME": "1" if set_datetime else "0",
        "OMNIFLOW_EVAL_DEVICE_LABEL": resolved_device,
        "OMNIFLOW_RAW_REPLAY_RESULT_JSON": str(raw_replay_result),
    }
    if serial.strip():
        env["ANDROID_SERIAL"] = serial.strip()

    argv = [
        python_executable,
        "-m",
        "src.integrations.android_world.run_episode",
        "--android-world-root",
        str(resolve_path(android_world_root, root=repo_root)),
        "--tasks",
        item.task,
        "--task-random-seed",
        str(resolved_task_seed),
        "--n-task-combinations",
        "1",
        "--fixed-task-seed",
        "--console-port",
        str(int(console_port)),
        "--agent",
        "fixed_replay",
        "--raw-replay-run-log",
        str(replay_run_log),
        "--max-steps",
        str(step_budget),
        "--output-path",
        str(resolved_output),
    ]
    if fixed_task_params:
        argv.extend(["--task-params-json", params_json])
    if perform_emulator_setup:
        argv.append("--perform-emulator-setup")
    if adb_path.strip():
        argv.extend(["--adb-path", adb_path.strip()])
    if planner_provider.strip():
        argv.extend(["--planner-provider", planner_provider.strip()])
    if model.strip():
        argv.extend(["--model", model.strip()])
    return CommandSpec(
        label=f"fixed_replay:{item.task}",
        argv=argv,
        env=env,
        cwd=repo_root,
        output_path=resolved_output,
        timeout_sec=float(timeout_sec) if timeout_sec and timeout_sec > 0 else None,
        metadata={
            "source_run_log": str(item.source_run_log),
            "source_run_log_sha256": sha256_file(item.source_run_log),
            "replay_run_log": str(replay_run_log),
            "replay_run_log_sha256": sha256_file(replay_run_log),
            "memory_root": str(resolve_path(replay_memory_root, root=repo_root))
            if replay_memory_root
            else "",
            "source_materialization": source_materialization,
            "direct_replay_ready": profile.direct_replay_ready,
            "method": resolved_method,
            "agent": "fixed_replay",
            "device": resolved_device,
            "serial": serial.strip(),
            "console_port": int(console_port),
            "run_dir": str(resolved_output),
            "raw_replay_result": str(raw_replay_result),
            "perform_emulator_setup": bool(perform_emulator_setup),
            "fixed_task_seed": True,
            "fixed_task_params": bool(fixed_task_params),
            "task_random_seed": resolved_task_seed,
            "max_steps": step_budget,
            "timeout_sec": int(timeout_sec),
            "task_params": dict(effective_params) if fixed_task_params else None,
            "task_params_override": (
                dict(task_params_override) if task_params_override is not None else None
            ),
            "state_backend": "androidworld",
            "action_backend": "androidworld",
            "native_androidworld_agent_io": True,
            "execution_backend": "recorded_coordinate_replay_v1",
            "uses_action_transfer": False,
            "uses_source_xml": False,
            "uses_vlm_fallback": False,
        },
    )


def build_official_command(
    item: CanonicalRunLog,
    *,
    android_world_root: str | Path = DEFAULT_ANDROID_WORLD_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    method_name: str = "t3a_hint",
    official_agent_name: str = "t3a_gpt4",
    source_action_hint_path: str | Path | None = None,
    device_label: str = "",
    serial: str = "",
    console_port: int = 5554,
    adb_path: str = "",
    max_steps: int = MAX_STEPS,
    timeout_sec: int = 180,
    task_random_seed: int | None = None,
    fixed_task_seed: bool = True,
    fixed_task_params: bool = False,
    task_params_override: dict[str, Any] | None = None,
    perform_emulator_setup: bool = True,
    model: str = "",
    python_executable: str = sys.executable,
    repo_root: Path = REPO_ROOT,
) -> CommandSpec:
    resolved_device = _device_label(
        explicit_label=device_label,
        serial=serial,
        console_port=console_port,
    )
    resolved_method = _safe_stem(method_name, fallback="t3a_hint")
    resolved_agent = str(official_agent_name or "t3a_gpt4").strip() or "t3a_gpt4"
    setting_root = _experiment_run_dir(
        output_root,
        task=item.task,
        method=resolved_method,
        device=resolved_device,
        serial=serial,
        console_port=console_port,
        evaluation_seed=(
            int(task_random_seed)
            if task_random_seed is not None
            else TASK_SEED
        ),
        repo_root=repo_root,
    )
    resolved_output = _next_attempt(setting_root, "runlog")
    resolved_task_seed = int(
        item.replay_seed if task_random_seed is None else task_random_seed
    )
    effective_params = dict(task_params_override or {})
    env: dict[str, str] = {}
    if serial.strip():
        env["ANDROID_SERIAL"] = serial.strip()

    argv = [
        python_executable,
        "-m",
        "src.integrations.android_world.run_episode",
        "--android-world-root",
        str(resolve_path(android_world_root, root=repo_root)),
        "--tasks",
        item.task,
        "--task-random-seed",
        str(resolved_task_seed),
        "--n-task-combinations",
        "1",
        "--console-port",
        str(int(console_port)),
        "--agent",
        f"official:{resolved_agent}",
        "--max-steps",
        str(int(max_steps)),
        "--output-path",
        str(resolved_output),
    ]
    if fixed_task_params:
        params_json = json.dumps(
            effective_params,
            ensure_ascii=False,
            separators=(",", ":"),
            default=_json_default,
        )
        argv.extend(["--task-params-json", params_json])
    if fixed_task_seed:
        argv.append("--fixed-task-seed")
    if perform_emulator_setup:
        argv.append("--perform-emulator-setup")
    if source_action_hint_path:
        argv.extend(
            [
                "--source-action-hint-path",
                str(resolve_path(source_action_hint_path, root=repo_root)),
            ]
        )
    if adb_path.strip():
        argv.extend(["--adb-path", adb_path.strip()])
    if model.strip():
        argv.extend(["--model", model.strip()])
    hint_path_text = (
        str(resolve_path(source_action_hint_path, root=repo_root))
        if source_action_hint_path
        else ""
    )
    return CommandSpec(
        label=f"official:{resolved_agent}:{item.task}",
        argv=argv,
        env=env,
        cwd=repo_root,
        output_path=resolved_output,
        timeout_sec=float(timeout_sec) if timeout_sec and timeout_sec > 0 else None,
        metadata={
            "source_run_log": str(item.source_run_log),
            "mode": "androidworld_official_agent",
            "agent": f"official:{resolved_agent}",
            "method": resolved_method,
            "official_agent_name": resolved_agent,
            "model": str(model or "").strip(),
            "device": resolved_device,
            "serial": serial.strip(),
            "console_port": int(console_port),
            "run_dir": str(resolved_output),
            "perform_emulator_setup": bool(perform_emulator_setup),
            "fixed_task_seed": bool(fixed_task_seed),
            "fixed_task_params": bool(fixed_task_params),
            "task_params": dict(effective_params) if fixed_task_params else None,
            "task_params_override": (
                dict(task_params_override) if task_params_override is not None else None
            ),
            "task_random_seed": resolved_task_seed,
            "max_steps": int(max_steps),
            "timeout_sec": int(timeout_sec),
            "state_backend": "androidworld",
            "action_backend": "androidworld",
            "native_androidworld_agent_io": True,
            "execution_backend": "androidworld_official_agent",
            "uses_forced_replay": False,
            "uses_omniflow_agent": False,
            "uses_source_action_hints": bool(hint_path_text),
            "source_action_hint_path": hint_path_text,
            "hint_mode": ("official_goal_reference_trace" if hint_path_text else ""),
            "include_indexed_context": False,
        },
    )


def build_task_command(
    item: CanonicalRunLog,
    *,
    android_world_root: str | Path = DEFAULT_ANDROID_WORLD_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    method_name: str = "e2e",
    agent_name: str = "omniflow",
    device_label: str = "",
    run_dir_suffix: str = "",
    serial: str = "",
    console_port: int = 5554,
    adb_path: str = "",
    max_steps: int = MAX_STEPS,
    timeout_sec: int | None = None,
    max_fallback_steps: int | None = None,
    task_random_seed: int | None = None,
    fixed_task_seed: bool = True,
    fixed_task_params: bool = True,
    task_params_override: dict[str, Any] | None = None,
    perform_emulator_setup: bool = True,
    planner_provider: str = "",
    model: str = "",
    planner_timeout_sec: float | None = None,
    store_path: str | Path | None = None,
    autodroid_memory_root: str | Path | None = None,
    omnitransfer_root: str | Path | None = None,
    python_executable: str = sys.executable,
    repo_root: Path = REPO_ROOT,
) -> CommandSpec:
    """Build one complete AndroidWorld episode command.

    AndroidWorld owns task initialization, agent execution, official validation,
    and teardown inside this single launcher process.
    """

    resolved_device = _device_label(
        explicit_label=device_label,
        serial=serial,
        console_port=console_port,
    )
    resolved_method = _safe_stem(method_name, fallback="e2e")
    resolved_agent = str(agent_name or "omniflow").strip() or "omniflow"
    setting_root = _experiment_run_dir(
        output_root,
        task=item.task,
        method=resolved_method,
        device=resolved_device,
        serial=serial,
        console_port=console_port,
        evaluation_seed=(
            int(task_random_seed)
            if task_random_seed is not None
            else TASK_SEED
        ),
        repo_root=repo_root,
    )
    resolved_output = _next_attempt(setting_root, "runlog")
    if str(run_dir_suffix or "").strip():
        resolved_output = resolved_output / _safe_relative_path(
            run_dir_suffix,
            fallback="run",
        )
    resolved_store_path = (
        resolve_path(store_path, root=repo_root)
        if store_path
        else None
    )
    resolved_autodroid_memory_root = (
        resolve_path(autodroid_memory_root, root=repo_root)
        if autodroid_memory_root
        else None
    )
    resolved_task_seed = int(
        item.replay_seed if task_random_seed is None else task_random_seed
    )
    if resolved_agent == "omniflow":
        planner_provider, model = _resolve_planner_provider_and_model(
            planner_provider,
            model,
            repo_root=repo_root,
        )
    else:
        planner_provider, model = "", ""
    env: dict[str, str] = {}
    if serial.strip():
        env["ANDROID_SERIAL"] = serial.strip()
    baseline_without_functions = resolved_agent == "omniflow" and not str(
        store_path or ""
    ).strip()
    if resolved_agent == "omniflow" and max_fallback_steps is not None:
        env["OMNIFLOW_ANDROIDWORLD_MAX_FALLBACK_STEPS"] = str(
            max(0, int(max_fallback_steps))
        )
    if baseline_without_functions:
        env["OMNIFLOW_ANDROIDWORLD_BASELINE"] = "1"
        env.pop("OMNIFLOW_PLANNER_TIMEOUT_SEC", None)
        env.pop("OMNIFLOW_OPENAI_TIMEOUT_SEC", None)
    # The public launcher exports this setting, but make it explicit on every
    # child CommandSpec as well.  This prevents a scheduler worker from
    # silently falling back to AndroidWorld-native observe/act when the
    # campaign requested OOB transport.
    control_backend = str(
        os.environ.get("OMNIFLOW_ANDROIDWORLD_CONTROL_BACKEND", "oob")
    ).strip().lower() or "androidworld"
    if resolved_agent in {"omniflow", "autodroid"}:
        env["OMNIFLOW_ANDROIDWORLD_CONTROL_BACKEND"] = control_backend
        env["OMNIFLOW_TRANSFER_ERROR_CONTEXT"] = json.dumps(
            {
                "task": item.task,
                "method": resolved_method,
                "device": resolved_device,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    resolved_omnitransfer_root = (
        resolve_path(omnitransfer_root, root=repo_root)
        if omnitransfer_root
        else None
    )
    if resolved_omnitransfer_root is not None:
        env["OMNITRANSFER_ROOT"] = str(resolved_omnitransfer_root)
    effective_params = (
        dict(task_params_override) if task_params_override is not None else item.params
    )

    argv = [
        python_executable,
        "-m",
        "src.integrations.android_world.run_episode",
        "--android-world-root",
        str(resolve_path(android_world_root, root=repo_root)),
        "--tasks",
        item.task,
        "--task-random-seed",
        str(resolved_task_seed),
        "--n-task-combinations",
        "1",
        "--console-port",
        str(int(console_port)),
        "--agent",
        resolved_agent,
        "--max-steps",
        str(int(max_steps)),
        "--output-path",
        str(resolved_output),
    ]
    if fixed_task_params:
        params_json = json.dumps(
            effective_params,
            ensure_ascii=False,
            separators=(",", ":"),
            default=_json_default,
        )
        argv.extend(["--task-params-json", params_json])
    if fixed_task_seed:
        argv.append("--fixed-task-seed")
    if perform_emulator_setup:
        argv.append("--perform-emulator-setup")
    if resolved_agent == "omniflow":
        if resolved_store_path is not None:
            argv.extend(["--store-path", str(resolved_store_path)])
        if planner_provider.strip():
            argv.extend(["--planner-provider", planner_provider.strip()])
        if model.strip():
            argv.extend(["--model", model.strip()])
        if (
            planner_timeout_sec is not None
            and float(planner_timeout_sec) > 0
        ):
            argv.extend(["--planner-timeout-sec", str(float(planner_timeout_sec))])
    elif resolved_agent == "autodroid":
        if resolved_autodroid_memory_root is None:
            raise ValueError("autodroid_memory_root_required")
        argv.extend(["--autodroid-memory-root", str(resolved_autodroid_memory_root)])
    if adb_path.strip():
        argv.extend(["--adb-path", adb_path.strip()])
    execution_mode = (
        "normal_omniflow_e2e"
        if resolved_agent == "omniflow"
        else "normal_androidworld_episode"
    )
    return CommandSpec(
        label=f"e2e:{item.task}",
        argv=argv,
        env=env,
        cwd=repo_root,
        output_path=resolved_output,
        timeout_sec=(
            None
            if baseline_without_functions
            else (
                float(timeout_sec)
                if timeout_sec is not None and timeout_sec > 0
                else None
            )
        ),
        metadata={
            "mode": execution_mode,
            "agent": resolved_agent,
            "method": resolved_method,
            "device": resolved_device,
            "serial": serial.strip(),
            "console_port": int(console_port),
            "run_dir": str(resolved_output),
            "run_dir_suffix": str(_safe_relative_path(run_dir_suffix, fallback="run"))
            if str(run_dir_suffix or "").strip()
            else "",
            "store_path": str(resolved_store_path or ""),
            "autodroid_memory_root": str(resolved_autodroid_memory_root or ""),
            "omnitransfer_root": str(resolved_omnitransfer_root or ""),
            "perform_emulator_setup": bool(perform_emulator_setup),
            "fixed_task_seed": bool(fixed_task_seed),
            "fixed_task_params": bool(fixed_task_params),
            "task_params": dict(effective_params) if fixed_task_params else None,
            "task_params_override": (
                dict(task_params_override) if task_params_override is not None else None
            ),
            "task_random_seed": resolved_task_seed,
            "max_steps": int(max_steps),
            "timeout_sec": int(timeout_sec or 0),
            "planner_timeout_sec": (
                float(planner_timeout_sec) if planner_timeout_sec is not None else None
            ),
            "max_fallback_steps": (
                max(0, int(max_fallback_steps))
                if resolved_agent == "omniflow" and max_fallback_steps is not None
                else None
            ),
            "planner_provider": planner_provider,
            "model": model,
            "state_backend": "androidworld",
            "control_backend": (
                "oob_control"
                if resolved_agent in {"omniflow", "autodroid"}
                and control_backend in {"oob", "omniflow", "oob_control"}
                else "androidworld"
            ),
            "action_backend": (
                "oob_control"
                if resolved_agent in {"omniflow", "autodroid"}
                and control_backend in {"oob", "omniflow", "oob_control"}
                else "androidworld"
            ),
            "native_androidworld_agent_io": not (
                resolved_agent in {"omniflow", "autodroid"}
                and control_backend in {"oob", "omniflow", "oob_control"}
            ),
            "include_indexed_context": False,
            "uses_action_transfer": True,
        },
    )


def task_goal_for_params(
    task: str,
    fallback_goal: str,
    *,
    android_world_root: str | Path,
    task_params: dict[str, Any] | None,
) -> str:
    """Render the AndroidWorld task template for one fixed parameter set."""

    params = task_params if isinstance(task_params, dict) else {}
    if not params:
        return str(fallback_goal)
    root = Path(android_world_root).expanduser().resolve()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    try:
        from android_world import registry

        task_type = registry.TaskRegistry().get_registry(family="android_world").get(
            str(task)
        )
        # AndroidWorld has two task-template forms: most task classes expose
        # a class string, while parameterized tasks (e.g. MarkorEditNote)
        # expose ``template`` as an instance property.  Stringifying the
        # latter leaks ``<property object ...>`` into the MobileGPT prompt and
        # makes an otherwise valid run fail before the first action.
        raw_template = getattr(task_type, "template", "")
        # A number of generated information-retrieval classes expose a
        # protobuf ``task_template`` property at class level, but the
        # rendered natural-language string is available on an instance's
        # ``template`` attribute.  Always instantiate when the class value is
        # absent/non-string/property so we never stringify the protobuf or a
        # Python property object into the prompt.
        if not isinstance(raw_template, str) or not raw_template.strip():
            instance = task_type(params)
            instance_template = getattr(instance, "template", "")
            if isinstance(instance_template, str) and instance_template.strip():
                raw_template = instance_template
            else:
                raw_template = getattr(instance, "task_template", "")
        template = str(raw_template or "")
        if template:
            return template.format(**params)
    except Exception:
        # A baseline that cannot render a template retains its archived goal;
        # the official episode still owns validation and records the exact
        # task parameters used.
        pass
    return str(fallback_goal)


def _evaluation_item_for_seed(
    item: CanonicalRunLog,
    *,
    android_world_root: str | Path,
    task_seed: int,
) -> CanonicalRunLog:
    """Build the target task instance from the evaluation seed.

    Source RunLogs identify the task and preserve provenance, but their
    generated parameters belong to the source episode. Target evaluation must
    derive one fresh parameter set from the requested evaluation seed and
    share that set across methods and devices.
    """

    from src.integrations.android_world.run_episode import (
        instantiate_androidworld_task,
    )

    task = instantiate_androidworld_task(
        android_world_root=android_world_root,
        task_name=item.task,
        task_params=None,
        task_seed=int(task_seed),
    )
    parameters = dict(getattr(task, "params", {}) or {})
    goal = str(getattr(task, "goal", "") or item.goal).strip() or item.goal
    return replace(item, goal=goal, params=parameters)


def _command_line(spec: CommandSpec) -> str:
    env_prefix = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in sorted(spec.env.items())
    )
    argv = shlex.join(spec.argv)
    return f"env {env_prefix} {argv}" if env_prefix else argv


def parse_device_targets(raw_targets: str) -> list[DeviceTarget]:
    """Parse only the fixed device triples from the experiment contract."""

    targets: list[DeviceTarget] = []
    configured = {
        (label, serial, int(port))
        for label, serial, port in (*DEVICES, SOURCE_DEVICE)
    }
    seen_labels: set[str] = set()
    seen_serials: set[str] = set()
    seen_ports: set[int] = set()
    for index, raw_item in enumerate(str(raw_targets or "").split(","), 1):
        item = raw_item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(":")]
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"Device targets must use LABEL:SERIAL:PORT, got {item!r}")
        label, serial, port_text = parts
        try:
            console_port = int(port_text)
        except ValueError as exc:
            raise ValueError(f"Invalid console port in device target {item!r}") from exc
        if console_port <= 0:
            raise ValueError(f"Invalid console port in device target {item!r}")
        if (label, serial, console_port) not in configured:
            raise ValueError(f"Unconfigured device target: {item!r}")
        safe_label = _safe_stem(label, fallback=f"device{index}")
        if safe_label in seen_labels:
            raise ValueError(f"Duplicate device target label: {safe_label}")
        if serial in seen_serials:
            raise ValueError(f"Duplicate device target serial: {serial}")
        if console_port in seen_ports:
            raise ValueError(f"Duplicate device target console port: {console_port}")
        seen_labels.add(safe_label)
        seen_serials.add(serial)
        seen_ports.add(console_port)
        targets.append(
            DeviceTarget(
                label=safe_label,
                serial=serial,
                console_port=console_port,
            )
        )
    return targets


def _make_tree_owner_writable(root: Path) -> None:
    for path in [root, *root.rglob("*")]:
        if path.is_symlink():
            continue
        path.chmod(path.stat().st_mode | 0o200)


def freeze_mobilegpt_memory(
    source_memory_root: str | Path,
    frozen_memory_root: str | Path,
) -> dict[str, Any]:
    source_root = resolve_path(source_memory_root)
    frozen_root = resolve_path(frozen_memory_root)
    if not source_root.is_dir():
        raise FileNotFoundError(f"MobileGPT memory root not found: {source_root}")
    if frozen_root.exists():
        raise FileExistsError(f"immutable_frozen_memory_exists:{frozen_root}")
    shutil.copytree(source_root, frozen_root)
    digest, file_count = mobilegpt_memory.mobilegpt_memory_digest(frozen_root)
    for path in frozen_root.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    frozen_root.chmod(0o555)
    return {
        "schema_version": "omniflow.mobilegpt_frozen_memory.v1",
        "source_memory_root": str(source_root),
        "frozen_memory_root": str(frozen_root),
        "digest": digest,
        "file_count": file_count,
        "read_only": True,
    }


def prepare_mobilegpt_episode_memory(
    frozen_memory_root: str | Path,
    episode_memory_root: str | Path,
    *,
    expected_digest: str,
    expected_file_count: int,
) -> dict[str, Any]:
    frozen_root = resolve_path(frozen_memory_root)
    episode_root = resolve_path(episode_memory_root)
    if not frozen_root.is_dir():
        raise FileNotFoundError(f"frozen_mobilegpt_memory_missing:{frozen_root}")
    if episode_root.exists():
        raise FileExistsError(f"immutable_episode_memory_exists:{episode_root}")
    shutil.copytree(frozen_root, episode_root)
    _make_tree_owner_writable(episode_root)
    digest, file_count = mobilegpt_memory.mobilegpt_memory_digest(episode_root)
    non_writable_paths = [
        str(path)
        for path in [episode_root, *episode_root.rglob("*")]
        if not path.is_symlink() and not path.stat().st_mode & 0o200
    ]
    if digest != expected_digest or file_count != expected_file_count:
        raise RuntimeError("MobileGPT episode memory digest mismatch before evaluation")
    if non_writable_paths:
        raise RuntimeError(
            "MobileGPT episode working memory is not owner-writable:"
            + ",".join(non_writable_paths[:5])
        )
    manifest = {
        "schema_version": "omniflow.mobilegpt_episode_memory.v2",
        "frozen_memory_root": str(frozen_root),
        "episode_memory_root": str(episode_root),
        "digest": digest,
        "derived_from_sha256": digest,
        "file_count": file_count,
        "read_only": False,
        "writable": True,
        "reusable": False,
        "write_policy": "isolated_attempt_copy_on_write",
    }
    manifest_path = episode_root.parent / "working_memory_manifest.json"
    try:
        with manifest_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, indent=2, ensure_ascii=False))
            handle.write("\n")
    except FileExistsError as exc:
        raise FileExistsError(
            f"immutable_working_memory_manifest_exists:{manifest_path}"
        ) from exc
    return {**manifest, "manifest_path": str(manifest_path.resolve())}


def audit_mobilegpt_episode_memory(
    episode_memory_root: str | Path,
    *,
    expected_digest: str,
    expected_file_count: int,
) -> tuple[Path, dict[str, Any]]:
    episode_root = resolve_path(episode_memory_root)
    actual_digest, actual_file_count = mobilegpt_memory.mobilegpt_memory_digest(episode_root)
    runtime_mutated = not (
        actual_digest == expected_digest and actual_file_count == expected_file_count
    )
    audit = {
        "schema_version": "omniflow.mobilegpt_episode_memory_audit.v2",
        "status": (
            "ISOLATED_WORKING_COPY_MODIFIED"
            if runtime_mutated
            else "ISOLATED_WORKING_COPY_UNCHANGED"
        ),
        "episode_memory_root": str(episode_root),
        "write_policy": "isolated_attempt_copy_on_write",
        "runtime_mutated": runtime_mutated,
        "reusable": False,
        "promoted": False,
        "derived_from_sha256": expected_digest,
        "final_sha256": actual_digest,
        "expected_digest": expected_digest,
        "actual_digest": actual_digest,
        "expected_file_count": expected_file_count,
        "actual_file_count": actual_file_count,
        "violations": [],
    }
    audit_path = episode_root.parent / "episode_memory_audit.json"
    try:
        with audit_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(audit, indent=2, ensure_ascii=False))
            handle.write("\n")
    except FileExistsError as exc:
        raise FileExistsError(
            f"immutable_episode_memory_audit_exists:{audit_path}"
        ) from exc
    return audit_path.resolve(), audit


def _find_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _mobilegpt_browser_task_html(
    item: CanonicalRunLog,
    *,
    android_world_root: str | Path = DEFAULT_ANDROID_WORLD_ROOT,
    task_params_override: dict[str, Any] | None = None,
) -> str:
    params = dict(task_params_override or item.params or {})
    if "browser_task_seed" not in params:
        return ""
    root = resolve_path(android_world_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from android_world.task_evals.single import browser
    except Exception:
        return ""
    task_type = getattr(browser, item.task, None)
    html = str(getattr(task_type, "HTML", "") or "")
    if not html:
        return ""
    return html.replace("%%SEED%%", str(params["browser_task_seed"]))


def _start_mobilegpt_browser_task_server(
    *,
    item: CanonicalRunLog,
    memory_root: Path,
    android_world_root: str | Path = DEFAULT_ANDROID_WORLD_ROOT,
    task_params_override: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> tuple[dict[str, Any], subprocess.Popen[Any] | None]:
    html = _mobilegpt_browser_task_html(
        item,
        android_world_root=android_world_root,
        task_params_override=task_params_override,
    )
    if not html:
        return {}, None
    serve_dir = memory_root / "browser_http"
    html_path = serve_dir / "task.html"
    log_path = serve_dir / "http_server.log"
    serve_dir.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html, encoding="utf-8")
    port = _find_free_local_port()
    url = f"http://10.0.2.2:{port}/task.html"
    prepare = {
        "url": url,
        "host": "127.0.0.1",
        "port": port,
        "serve_dir": str(serve_dir),
        "html_path": str(html_path),
        "log_path": str(log_path),
    }
    if dry_run:
        return prepare, None
    process = start_process(
        [
            sys.executable,
            "-m",
            "http.server",
            str(port),
            "--bind",
            "127.0.0.1",
            "--directory",
            str(serve_dir),
        ],
        cwd=REPO_ROOT,
        environment=_subprocess_env({}),
        log_path=log_path,
    )
    time.sleep(0.5)
    if process.poll() is not None:
        raise RuntimeError(
            f"MobileGPT browser task HTTP server exited early; see {log_path}"
        )
    print(
        f"[mobilegpt:browser-task-server] serving {html_path} at {url}",
        flush=True,
    )
    return prepare, process


def build_mobilegpt_server_command(
    action: str,
    *,
    mobilegpt_root: str | Path = DEFAULT_MOBILEGPT_ROOT,
    mobilegpt_memory_root: str | Path | None = None,
    mobilegpt_memory_manifest: str | Path | None = None,
    embedding_model: str = "",
    chat_model: str = "",
    write_through_memory: bool = False,
    serial: str = "",
    adb_path: str = "",
    server_host: str = "0.0.0.0",
    port: int = 12345,
    stats_jsonl: str | Path = DEFAULT_MOBILEGPT_STATS_JSONL,
    target_package: str = "",
    target_app: str = "",
    target_task_name: str = "",
    python_executable: str = sys.executable,
    repo_root: Path = REPO_ROOT,
) -> CommandSpec:
    root = resolve_path(mobilegpt_root, root=repo_root)
    server_root = root / "Server"
    if not (server_root / "main.py").is_file():
        raise FileNotFoundError(f"mobilegpt_server_root_missing:{server_root}")
    env: dict[str, str] = {}
    env["MOBILEGPT_SERVER_HOST"] = str(server_host or "0.0.0.0")
    env["MOBILEGPT_CLIENT_MODE"] = "upstream_accessibility"
    if chat_model or embedding_model:
        env["OPENAI_BASE_URL"] = FORMAL_MODEL_BASE_URL
        env["MOBILEGPT_CHAT_MODEL"] = chat_model
        env["MOBILEGPT_VISION_MODEL"] = chat_model
    env["MOBILEGPT_SERVER_PORT"] = str(int(port))
    env["PYTHONUNBUFFERED"] = "1"
    if serial.strip():
        env["ANDROID_SERIAL"] = serial.strip()
    if adb_path.strip():
        env["ADB_PATH"] = adb_path.strip()
    resolved_memory_root = (
        resolve_path(mobilegpt_memory_root, root=repo_root)
        if mobilegpt_memory_root
        else None
    )
    if resolved_memory_root is not None:
        env["MOBILEGPT_MEMORY_ROOT"] = str(resolved_memory_root)
    if str(target_package or "").strip():
        env["MOBILEGPT_TARGET_PACKAGE"] = str(target_package).strip()
        env["MOBILEGPT_SKIP_APP_DISCOVERY"] = "1"
    if str(target_app or "").strip():
        env["MOBILEGPT_TARGET_APP"] = str(target_app).strip()
    if str(target_task_name or "").strip():
        env["MOBILEGPT_TARGET_TASK_NAME"] = str(target_task_name).strip()

    resolved_action = str(action or "").strip().lower()
    if resolved_action == "server":
        if resolved_memory_root is None:
            raise ValueError("mobilegpt_server_memory_required")
        from src.integrations.official_forward import prepare_mobilegpt_server

        staged = resolved_memory_root.parent / "official_server_workspace"
        forward = prepare_mobilegpt_server(
            official_root=root,
            memory_root=resolved_memory_root,
            workspace=staged,
            port=port,
            chat_model=chat_model,
            embedding_model=embedding_model,
            write_through_memory=bool(write_through_memory),
        )
        staged_server_root = Path(forward["server_root"])
        env["MOBILEGPT_STATS_JSONL"] = str(resolve_path(stats_jsonl, root=repo_root))
        argv = [
            python_executable,
            str(staged_server_root / "main.py"),
        ]
        return CommandSpec(
            label="mobilegpt:official-server",
            argv=argv,
            env=env,
            cwd=staged_server_root,
            output_path=None,
            metadata={
                "mobilegpt_root": str(root),
                "mobilegpt_memory_root": str(resolved_memory_root or ""),
                "port": int(port),
                "target_package": str(target_package or "").strip(),
                "target_app": str(target_app or "").strip(),
                "state_backend": "official_mobilegpt",
                "official_server": str(server_root / "main.py"),
                "official_staged_server": str(staged_server_root / "main.py"),
                "embedding_model": embedding_model or "text-embedding-3-small",
                "chat_model": chat_model or "upstream_main.py",
                "external_forward_only": True,
                "write_through_memory": bool(write_through_memory),
                "log_path": str(staged_server_root.parent / "official_server.log"),
            },
        )

    raise ValueError("Unsupported MobileGPT action. Use: server.")


def run_command(spec: CommandSpec, *, dry_run: bool = False) -> int:
    print(f"[{spec.label}] {_command_line(spec)}", flush=True)
    if dry_run:
        spec.metadata["wall_sec"] = 0.0
        return 0
    result = run_process(
        spec.argv,
        cwd=spec.cwd,
        environment=_subprocess_env(spec.env),
        timeout_sec=spec.timeout_sec,
        log_path=Path(str(spec.metadata["log_path"]))
        if spec.metadata.get("log_path")
        else None,
        stdin_text=spec.stdin_text,
    )
    spec.metadata["wall_sec"] = round(float(result["wall_sec"]), 3)
    if result["timed_out"]:
        spec.metadata["timeout_sec"] = float(spec.timeout_sec or 0)
        spec.metadata["timed_out"] = True
        print(
            f"[{spec.label}] timed out after {spec.timeout_sec}s",
            flush=True,
        )
        return 124
    return int(result["returncode"])


def _materialize_mobilegpt_canonical_run_log(
    *,
    output_path: Path,
    item: CanonicalRunLog,
    target: DeviceTarget,
    task_seed: int | None,
) -> Path:
    """Wrap the official MobileGPT result in the shared RunLog contract.

    MobileGPT's upstream client intentionally emits ``task_results.jsonl`` and
    protocol telemetry, not OmniFlow's canonical RunLog.  The experiment
    runner still needs one stable evidence envelope so successful results can
    be promoted and failed results can be archived without changing the
    upstream planner or action implementation.
    """

    result = _last_jsonl_object(output_path / "task_results.jsonl")
    probe = _read_json(output_path / "protocol_probe.json") if (
        output_path / "protocol_probe.json"
    ).is_file() else {}
    if not isinstance(probe, dict):
        probe = {}
    success = bool(result.get("official_validator_success") is True)
    duration_ms = _coerce_float(result.get("duration_ms"))
    execution_summary = {
        "actions_executed": _coerce_int(result.get("actions_executed")),
        "model_calls": _coerce_int(result.get("model_calls")),
        "prompt_tokens": _coerce_int(result.get("prompt_tokens")),
        "completion_tokens": _coerce_int(result.get("completion_tokens")),
        "total_tokens": _coerce_int(result.get("total_tokens")),
        "execution_duration_ms": duration_ms,
        "duration_ms": duration_ms,
        "fallback_steps": _coerce_int(result.get("fallback_steps")),
        "token_usage_status": str(result.get("token_usage_status") or "unavailable"),
        "failure_reason": str(result.get("failure_reason") or "").strip() or None,
        "success": success,
    }
    payload = {
        "schema_version": "oob.run_log.canonical.v1",
        "run_id": "current",
        "task_name": item.task,
        "goal": str(result.get("official_task_instruction") or item.goal),
        "task_parameters": dict(result.get("task_params") or item.params or {}),
        "seed": int(task_seed if task_seed is not None else item.replay_seed),
        "status": "succeeded" if success else "failed",
        "success": success,
        "validator": {
            "official": bool(result.get("official_validator_used")),
            "success": success,
            "reward": result.get("androidworld_validator_result", {}).get("reward"),
        },
        "steps": [],
        "diagnostics": {
            "done_reason": "validator_success" if success else "mobilegpt_failure",
            "execution_summary": execution_summary,
            "mobilegpt_result": result,
            "mobilegpt_protocol_probe": probe,
        },
        "provenance": {
            "kind": "runtime",
            "method": "mobilegpt",
            "device": target.serial,
            "execution_backend": "oob_control",
            "upstream_result": "task_results.jsonl",
        },
    }
    run_log_path = output_path / "run_log.json"
    run_log_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_log_path


def _select_from_args(args: argparse.Namespace) -> list[CanonicalRunLog]:
    source_text = str(getattr(args, "source_run_log", "") or "").strip()
    source = Path(source_text).expanduser() if source_text else Path()
    payload: dict[str, Any] = {}
    if source_text:
        payload = _read_object(source)
    params = payload.get("task_parameters")
    if not isinstance(params, dict):
        params = _task_params_override_from_args(args) or {}
    steps = payload.get("steps")
    return [
        CanonicalRunLog(
            task=str(args.task),
            goal=str(payload.get("goal") or args.task),
            params=dict(params),
            source_run_log=source,
            replay_seed=int(payload.get("seed") or args.source_seed),
            step_count=len(steps) if isinstance(steps, list) else 0,
            meta={},
        )
    ]


def _is_mobilegpt_method(method: str) -> bool:
    return str(method or "").strip() == "mobilegpt"


def _task_params_override_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    if not str(getattr(args, "task_params_json", "") or "").strip():
        return None
    decoded = json.loads(str(args.task_params_json))
    if not isinstance(decoded, dict):
        raise ValueError("--task-params-json must be a JSON object")
    return dict(decoded)


def _claim_method_memory_root(memory_root: str | Path) -> Path:
    root = resolve_path(memory_root)
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(f"immutable_memory_root_exists:{root}") from exc
    return root


def _memory_leakage_audit(
    *,
    memory_mode: str,
    target_inputs_read: bool = False,
    target_observations_read: bool = False,
    validator_state_read: bool = False,
) -> dict[str, Any]:
    violations = [
        name
        for name, present in (
            ("target_inputs_read", target_inputs_read),
            ("target_observations_read", target_observations_read),
            ("validator_state_read", validator_state_read),
        )
        if present
    ]
    if violations:
        status = "INVALID_MEMORY_LEAKAGE"
    elif memory_mode == "none":
        status = "NOT_APPLICABLE_NO_MEMORY"
    else:
        status = "CLEAN_SOURCE_ONLY"
    return {
        "status": status,
        "target_inputs_read": bool(target_inputs_read),
        "target_observations_read": bool(target_observations_read),
        "validator_state_read": bool(validator_state_read),
        "violations": violations,
    }


def _write_method_memory_manifest(
    *,
    memory_root: str | Path,
    task: str,
    method: str,
    memory_mode: str,
    source_seed: int | None,
    evaluation_seed: int | None,
    attempt_id: str,
    artifacts: dict[str, Any],
    source_run_log: str | Path | None = None,
    target_inputs_read: bool = False,
    target_observations_read: bool = False,
    validator_state_read: bool = False,
) -> Path:
    root = resolve_path(memory_root)
    if not root.is_dir():
        raise FileNotFoundError(f"unclaimed_memory_root:{root}")
    source: dict[str, Any] = {"seed": source_seed}
    if source_run_log is not None:
        resolved_source = resolve_path(source_run_log)
        source.update(
            {
                "run_log": str(resolved_source),
                "run_log_sha256": sha256_file(resolved_source),
            }
        )
    leakage_audit = _memory_leakage_audit(
        memory_mode=memory_mode,
        target_inputs_read=target_inputs_read,
        target_observations_read=target_observations_read,
        validator_state_read=validator_state_read,
    )
    manifest = {
        "schema_version": "omniflow.androidworld_method_memory.v2",
        "task_name": task,
        "method": method,
        "memory_mode": memory_mode,
        "memory_root": str(root),
        "source": source,
        "evaluation": {"seed": evaluation_seed},
        "attempt": {
            "id": str(attempt_id or "").strip(),
            "immutable": True,
        },
        "leakage_audit": leakage_audit,
        "artifacts": artifacts,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    path = root / "memory_manifest.json"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, indent=2, ensure_ascii=False))
            handle.write("\n")
    except FileExistsError as exc:
        raise FileExistsError(f"immutable_memory_manifest_exists:{path}") from exc
    if leakage_audit["status"] == "INVALID_MEMORY_LEAKAGE":
        raise ValueError(
            f"INVALID_MEMORY_LEAKAGE:{','.join(leakage_audit['violations'])}"
        )
    return path.resolve()


def _command_record_from_spec(
    spec: CommandSpec,
    *,
    task: str,
    method: str = "",
    device: str = "",
    returncode: int = 0,
    status: str = "",
    error: str = "",
    summary_exclude: bool = False,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {**dict(spec.metadata), **dict(extra_metadata or {})}
    resolved_method = method or str(metadata.get("method") or "").strip()
    resolved_device = device or str(metadata.get("device") or "").strip()
    return {
        "label": spec.label,
        "task": task,
        "method": resolved_method,
        "device": resolved_device,
        "status": status,
        "returncode": int(returncode),
        "output_path": str(spec.output_path or ""),
        "command": _command_line(spec),
        "error": error,
        "summary_exclude": bool(summary_exclude),
        "metadata": metadata,
    }


def _mobilegpt_target_package_from_open_target_app(open_target_app: str) -> str:
    value = str(open_target_app or "").strip()
    if not value:
        return ""
    if "/" in value:
        return value.split("/", 1)[0].strip()
    return value


def _mobilegpt_target_package_from_task_params(
    task_params: dict[str, Any] | None,
) -> str:
    """Return the app selected by this evaluated task instance, if any."""

    if not isinstance(task_params, dict):
        return ""
    return str(task_params.get("app_name") or "").strip()


def _select_mobilegpt_bootstrap_package(
    *,
    explicit_package: str,
    sealed_package: str,
    source_package: str,
    parameter_package: str,
) -> tuple[str, str]:
    """Select the app where a MobileGPT episode must begin.

    An ``app_name`` task parameter can name a destination app in a multi-app
    task.  The sealed successful source trajectory remains authoritative for
    the initial app; the evaluated parameter is only a fallback when the
    source carries no launch context.
    """

    candidates = (
        (explicit_package, "mobilegpt_open_target_app"),
        (sealed_package, "sealed_native_cold_source_memory"),
        (source_package, "source_runlog_target_inference"),
        (parameter_package, "task_parameters.app_name"),
    )
    for value, source in candidates:
        package = str(value or "").strip()
        if package:
            return package, source
    return "", "unresolved"


def _resolve_mobilegpt_target_package(
    candidate: str,
    *,
    adb_path: str,
    serial: str,
    android_world_root: str | Path = "",
) -> str:
    """Resolve an AndroidWorld app alias to the installed package name."""

    value = str(candidate or "").strip()
    if not value or "." in value:
        return value
    subprocess_env = _subprocess_env({})
    official_root = str(
        android_world_root
        or subprocess_env.get("OMNIFLOW_ANDROID_WORLD_ROOT")
        or ""
    ).strip()
    inserted_root = False
    if official_root and official_root not in sys.path:
        sys.path.insert(0, official_root)
        inserted_root = True
    try:
        official_package = resolve_androidworld_package(value)
    except (ImportError, ModuleNotFoundError, RuntimeError, ValueError):
        official_package = ""
    finally:
        if inserted_root:
            sys.path.remove(official_root)
    if official_package:
        return official_package
    # AndroidWorld names the camera app ``Camera`` while the official
    # emulator package is ``com.android.camera2``. Suffix matching cannot
    # infer this because ``camera2`` does not end in ``camera``; keep this
    # explicit mapping in the single target-package resolver so every
    # official MobileGPT device receives a launchable package.
    known_aliases = {
        "camera": "com.android.camera2",
    }
    value = known_aliases.get(value.casefold(), value)
    if not str(subprocess_env.get("OMNIFLOW_REAL_ADB_PATH") or "").strip():
        sdk_root = next(
            (
                str(subprocess_env.get(key) or "").strip()
                for key in (
                    "OMNIFLOW_ANDROID_SDK_ROOT",
                    "ANDROID_SDK_ROOT",
                    "ANDROID_HOME",
                )
                if str(subprocess_env.get(key) or "").strip()
            ),
            "",
        )
        real_adb = Path(sdk_root) / "platform-tools" / "adb" if sdk_root else None
        if real_adb is not None and real_adb.is_file() and os.access(real_adb, os.X_OK):
            # Package discovery is a child process of the public launcher.
            # Give its AndroidWorld compatibility wrapper the same concrete
            # adb selected by that launcher.
            subprocess_env["OMNIFLOW_REAL_ADB_PATH"] = str(real_adb)
    try:
        completed = subprocess.run(
            [
                str(adb_path or "adb"),
                "-s",
                str(serial),
                "shell",
                "pm",
                "list",
                "packages",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=20.0,
            env=subprocess_env,
        )
    except (OSError, subprocess.SubprocessError):
        return value
    packages = [
        line.split(":", 1)[1].strip()
        for line in completed.stdout.splitlines()
        if line.strip().startswith("package:") and ":" in line
    ]
    matches = [
        package
        for package in packages
        if package.casefold().endswith("." + value.casefold())
        or package.casefold() == value.casefold()
        or package.rsplit(".", 1)[-1].casefold().endswith(value.casefold())
    ]
    return matches[0] if len(matches) == 1 else value


def _mobilegpt_server_task_app(memory_app: str, target_package: str) -> str:
    """Return the app key used by the official Memory task database."""

    return str(memory_app or "").strip() or str(target_package or "").strip()


def _mobilegpt_memory_app_key(
    memory_root: str | Path,
    *,
    memory_app: str,
    target_package: str,
    task_name: str,
) -> str:
    """Resolve the case-sensitive app directory owned by MobileGPT Memory."""

    root = resolve_path(memory_root)
    app_directories = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and (path / "tasks.csv").is_file()
        ),
        key=lambda path: path.name,
    )
    requested_keys = tuple(
        key
        for key in (
            str(memory_app or "").strip(),
            str(target_package or "").strip(),
            str(target_package or "").strip().rsplit(".", 1)[-1],
        )
        if key
    )
    for requested_key in requested_keys:
        for app_directory in app_directories:
            if app_directory.name.casefold() == requested_key.casefold():
                return app_directory.name
    for app_directory in app_directories:
        try:
            with (app_directory / "tasks.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                if any(
                    str(row.get("name") or "").strip() == task_name
                    for row in csv.DictReader(handle)
                ):
                    return app_directory.name
        except (OSError, csv.Error):
            continue
    return _mobilegpt_server_task_app(memory_app, target_package)


def _start_background_command(
    spec: CommandSpec,
    *,
    dry_run: bool = False,
    warmup_sec: float = 0.0,
) -> tuple[subprocess.Popen[Any] | None, int]:
    print(f"[{spec.label}:background] {_command_line(spec)}", flush=True)
    if dry_run:
        return None, 0
    if spec.label == "mobilegpt:official-server":
        port = int(spec.metadata.get("port") or 0)
        if port > 0 and not _local_tcp_port_available(port):
            spec.metadata["server_start_failure"] = (
                f"mobilegpt_server_port_in_use:{port}"
            )
            return None, 127
    process = start_process(
        spec.argv,
        cwd=spec.cwd,
        environment=_subprocess_env(spec.env),
        log_path=Path(str(spec.metadata["log_path"]))
        if spec.metadata.get("log_path")
        else None,
    )
    if warmup_sec > 0:
        time.sleep(float(warmup_sec))
    returncode = process.poll()
    if returncode is not None:
        return process, int(returncode)
    if spec.label == "mobilegpt:official-server":
        port = int(spec.metadata.get("port") or 0)
        if port > 0 and not _wait_for_local_tcp_port(port, warmup_sec):
            spec.metadata["server_start_failure"] = (
                f"mobilegpt_server_not_listening:{port}"
            )
            stop_process(process, timeout_sec=2)
            return process, 127
        spec.metadata["server_ready"] = True
    return process, 0


def _local_tcp_port_available(port: int) -> bool:
    """Check the fixed local MobileGPT port before launching a new server."""

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("0.0.0.0", int(port)))
    except OSError:
        return False
    return True


def _wait_for_local_tcp_port(port: int, timeout_sec: float) -> bool:
    deadline = time.monotonic() + max(0.1, float(timeout_sec or 0.1))
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", int(port)), timeout=0.2):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def _stop_background_command(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    stop_process(process, timeout_sec=5)


def _next_attempt(setting_root: Path, group: str) -> Path:
    """Use one private-run slot; the public archive promotes it to ``current``."""

    group_root = setting_root / group
    group_root.mkdir(parents=True, exist_ok=True)
    return group_root / "current"


def build_mobilegpt_command(
    item: CanonicalRunLog,
    *,
    method_name: str,
    target: DeviceTarget,
    android_world_root: str | Path,
    output_root: str | Path,
    stats_jsonl: str | Path,
    mobilegpt_root: str | Path,
    server_host: str,
    server_port: int,
    target_package: str,
    max_steps: int,
    task_random_seed: int | None,
    fixed_task_seed: bool,
    fixed_task_params: bool,
    task_params_override: dict[str, Any] | None,
    perform_emulator_setup: bool,
    adb_path: str,
    start_timeout_sec: float,
    finish_timeout_sec: float,
    app_ready_timeout_sec: float = DEFAULT_MOBILEGPT_APP_READY_TIMEOUT_SEC,
    app_ready_poll_sec: float = DEFAULT_MOBILEGPT_APP_READY_POLL_SEC,
    timeout_sec: float | None = None,
    server_log_path: str | Path = "",
    run_dir_suffix: str = "",
    repo_root: Path = REPO_ROOT,
) -> CommandSpec:
    del fixed_task_seed, start_timeout_sec
    setting_root = _experiment_run_dir(
        output_root,
        task=item.task,
        method=_safe_stem(method_name, fallback="mobilegpt"),
        device=target.label,
        serial=target.serial,
        console_port=target.console_port,
        repo_root=repo_root,
    )
    resolved_output = _next_attempt(setting_root, "runlog")
    if str(run_dir_suffix or "").strip():
        resolved_output = resolved_output / _safe_relative_path(
            run_dir_suffix,
            fallback="run",
        )
    client_runtime_env = _subprocess_env({})
    client_output = resolved_output
    effective_params = dict(task_params_override or item.params or {})
    instruction = task_goal_for_params(
        item.task,
        item.goal,
        android_world_root=android_world_root,
        task_params=effective_params,
    )
    client_argv = [
        sys.executable,
        "-m",
        "src.integrations.official_forward",
        "--root",
        str(resolve_path(mobilegpt_root, root=repo_root)),
        "--serial",
        target.serial,
        "--adb",
        str(adb_path or "adb"),
        "--host",
        "10.0.2.2",
        "--instruction",
        instruction,
        "--output",
        str(client_output),
        "--timeout",
        str(float(finish_timeout_sec)),
        "--android-world-root",
        str(resolve_path(android_world_root, root=repo_root)),
        "--task",
        item.task,
        "--task-params-json",
        json.dumps(
            effective_params
            if (fixed_task_params or task_params_override is not None)
            else {},
            ensure_ascii=False,
            separators=(",", ":"),
            default=_json_default,
        ),
        "--task-seed",
        str(int(task_random_seed if task_random_seed is not None else item.replay_seed)),
        "--console-port",
        str(int(target.console_port)),
        "--grpc-port",
        str(int(target.console_port) + 3000),
        "--max-steps",
        str(int(max_steps)),
        "--server-port",
        str(int(server_port)),
    ]
    if str(server_log_path or "").strip():
        client_argv.extend(["--server-log", str(server_log_path)])
    if not perform_emulator_setup:
        client_argv.append("--no-perform-emulator-setup")
    client_environment = {
        "ANDROID_SERIAL": target.serial,
        "MOBILEGPT_STATS_JSONL": str(resolve_path(stats_jsonl, root=repo_root)),
        "MOBILEGPT_TARGET_PACKAGE": str(target_package or "").strip(),
        "MOBILEGPT_APP_READY_TIMEOUT_SEC": str(float(app_ready_timeout_sec)),
        "MOBILEGPT_CLIENT_MODE": "upstream_accessibility",
        "OMNIFLOW_ANDROIDWORLD_CONTROL_BACKEND": str(
            client_runtime_env.get("OMNIFLOW_ANDROIDWORLD_CONTROL_BACKEND")
            or os.environ.get("OMNIFLOW_ANDROIDWORLD_CONTROL_BACKEND")
            or "oob"
        ).strip().lower()
        or "oob",
        "PYTHONPATH": os.pathsep.join(
            value
            for value in (
                str(repo_root),
                str(repo_root / "src"),
                str(resolve_path(android_world_root, root=repo_root)),
                str(client_runtime_env.get("PYTHONPATH") or ""),
            )
            if value
        ),
    }
    for key in ("ANDROID_HOME", "ANDROID_SDK_ROOT", "PATH"):
        value = str(client_runtime_env.get(key) or "").strip()
        if value:
            client_environment[key] = value
    return CommandSpec(
        label=f"mobilegpt:upstream:{target.label}",
        argv=client_argv,
        env=client_environment,
        cwd=repo_root,
        output_path=resolved_output,
        timeout_sec=(
            float(timeout_sec) if timeout_sec is not None and timeout_sec > 0 else None
        ),
        metadata={
            "mode": "mobilegpt_upstream_client_server",
            "device_target": target.to_dict(),
            "mobilegpt_stats_jsonl": str(stats_jsonl),
            "mobilegpt_server_host": str(server_host),
            "mobilegpt_server_port": int(server_port),
            "target_package": str(target_package or "").strip(),
            "official_lifecycle": "mobilegpt_upstream_client_server",
            "official_server_entry": "Server/main.py",
            "official_client_entry": "App/com.example.MobileGPT",
            "official_client_output": str(client_output),
            "observe_backend": "mobilegpt_accessibility",
            "action_backend": "mobilegpt_accessibility",
            "external_forward_only": True,
            "app_ready_timeout_sec": float(app_ready_timeout_sec),
            "app_ready_poll_sec": float(app_ready_poll_sec),
        },
    )


def build_appagent_command(
    item: CanonicalRunLog,
    *,
    method_name: str,
    target: DeviceTarget,
    android_world_root: str | Path,
    output_root: str | Path,
    appagent_root: str | Path,
    docs_root: str | Path,
    max_steps: int,
    timeout_sec: int,
    task_random_seed: int,
    task_params: dict[str, Any],
    adb_path: str,
    repo_root: Path = REPO_ROOT,
) -> CommandSpec:
    """Build the official AppAgent executor command on the shared OOB bridge."""

    resolved_appagent_root = resolve_path(appagent_root, root=repo_root)
    resolved_docs_root = resolve_path(docs_root, root=repo_root)
    resolved_device = _device_label(
        explicit_label=target.label,
        serial=target.serial,
        console_port=target.console_port,
    )
    setting_root = _experiment_run_dir(
        output_root,
        task=item.task,
        method=_safe_stem(method_name, fallback="appagent"),
        device=resolved_device,
        serial=target.serial,
        console_port=target.console_port,
        repo_root=repo_root,
    )
    resolved_output = _next_attempt(setting_root, "runlog")
    workspace = resolved_output / "official_workspace"
    runtime_env = _subprocess_env({})
    endpoint = FORMAL_MODEL_BASE_URL.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"
    goal = task_goal_for_params(
        item.task,
        item.goal,
        android_world_root=android_world_root,
        task_params=task_params,
    )
    app_name = resolved_docs_root.parent.name
    from src.integrations.official_forward import prepare_appagent_workspace

    forward = prepare_appagent_workspace(
        official_root=resolved_appagent_root,
        docs_root=resolved_docs_root,
        workspace=workspace,
        app_name=app_name,
        serial=target.serial,
        adb_path=adb_path or "adb",
        config={
            "MODEL": "OpenAI",
            "OPENAI_API_BASE": endpoint,
            "OPENAI_API_KEY": str(
                runtime_env.get("LLMTHU_API_KEY")
                or runtime_env.get("OPENAI_API_KEY")
                or ""
            ),
            "OPENAI_API_MODEL": APPAGENT_MODEL,
            "MAX_TOKENS": 512,
            "THINKING": FORMAL_THINKING,
            "TEMPERATURE": 0.0,
            "REQUEST_INTERVAL": 0.0,
            "DASHSCOPE_API_KEY": "",
            "QWEN_MODEL": APPAGENT_MODEL,
            "ANDROID_SCREENSHOT_DIR": "/sdcard",
            "ANDROID_XML_DIR": "/sdcard",
            "DOC_REFINE": False,
            "MAX_ROUNDS": int(max_steps),
            "DARK_MODE": False,
            "MIN_DIST": 30,
        },
    )
    output = resolved_output
    argv = [
        sys.executable,
        "-m",
        "src.integrations.official_forward",
        "--baseline",
        "appagent",
        "--executor",
        str(resolved_appagent_root / "scripts" / "task_executor.py"),
        "--app-name",
        app_name,
        "--serial",
        target.serial,
        "--workspace",
        str(workspace),
        "--output",
        str(output),
        "--goal",
        goal,
        "--timeout",
        str(float(timeout_sec)),
        "--android-world-root",
        str(resolve_path(android_world_root, root=repo_root)),
        "--task",
        item.task,
        "--task-params-json",
        json.dumps(task_params, ensure_ascii=False, separators=(",", ":"), default=_json_default),
        "--task-seed",
        str(int(task_random_seed)),
        "--console-port",
        str(int(target.console_port)),
        "--grpc-port",
        str(int(target.console_port) + 3000),
        "--adb",
        str(adb_path or "adb"),
        "--max-steps",
        str(int(max_steps)),
        "--no-perform-emulator-setup",
    ]
    return CommandSpec(
        label=f"appagent:official:{target.label}",
        argv=argv,
        env={
            "ANDROID_SERIAL": target.serial,
            "OPENAI_MODEL": APPAGENT_MODEL,
            "OPENAI_BASE_URL": FORMAL_MODEL_BASE_URL,
            "APPAGENT_THINKING": FORMAL_THINKING,
            "PYTHONPATH": os.pathsep.join(
                value for value in (str(repo_root), runtime_env.get("PYTHONPATH", "")) if value
            ),
            "PATH": str(Path(forward["adb_proxy"]).parent) + os.pathsep + runtime_env.get("PATH", ""),
        },
        cwd=workspace,
        output_path=output,
        timeout_sec=float(timeout_sec),
        stdin_text="",
        metadata={
            "mode": "appagent_official_deployment",
            "agent": "official_appagent",
            "device_target": target.to_dict(),
            "appagent_root": str(resolved_appagent_root),
            "appagent_docs_root": str(resolved_docs_root),
            "official_workspace": str(workspace),
            "official_app_name": app_name,
            "external_forward_only": True,
            "shared_control_backend": "oob",
        },
    )


def _run_result_mobilegpt(
    *,
    args: argparse.Namespace,
    item: CanonicalRunLog,
    targets: Sequence[DeviceTarget],
    output_root: Path,
    task_params_override: dict[str, Any] | None,
    task_seed: int | None,
    method: str,
    attempt_id: str,
    source_run_log: Path,
    compatible_source_sha256s: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], int]:
    if method != "mobilegpt":
        raise ValueError(f"unsupported_mobilegpt_method:{method}")
    if not targets:
        raise ValueError("mobilegpt_device_target_required")

    memory_root = resolve_path(args.output_path) / "mobilegpt_memory"
    memory_root.mkdir(parents=True, exist_ok=True)
    source_memory_value = str(
        getattr(args, "mobilegpt_source_memory_root", "")
    ).strip()
    cold_start = not source_memory_value
    if cold_start:
        source_memory_root = memory_root / "cold_initial_memory"
        source_memory_root.mkdir()
        source_manifest_path: Path | None = None
        source_method = "none"
        source_prep_type = "empty_cold_start"
        adapted_memory: dict[str, Any] = {}
    else:
        provided_memory_root = resolve_path(source_memory_value)
        if not provided_memory_root.is_dir():
            raise FileNotFoundError(
                f"mobilegpt_source_memory_missing:{provided_memory_root}"
            )
        # Accept both documented forms of the sealed asset:
        #   bundle/                  (manifest + memory/ child)
        #   bundle/memory/           (the native memory root)
        # The public launcher passes the explicitly supplied path verbatim,
        # so resolving this here keeps ``run --memory`` independent of which
        # form the caller used.  The MobileGPT server and digest must always
        # receive the actual native memory directory, never the bundle root.
        bundle_manifest_path = provided_memory_root / MOBILEGPT_MEMORY_MANIFEST
        nested_memory_root = provided_memory_root / "memory"
        if bundle_manifest_path.is_file() and nested_memory_root.is_dir():
            source_memory_root = nested_memory_root
            source_manifest_path = bundle_manifest_path
        else:
            source_memory_root = provided_memory_root
            source_manifest_path = source_memory_root.parent / MOBILEGPT_MEMORY_MANIFEST
        if not source_manifest_path.is_file():
            raise FileNotFoundError(
                f"mobilegpt_source_memory_manifest_missing:{source_manifest_path}"
            )
        source_method = "mobilegpt_official_exploration"
        source_prep_type = "mobilegpt_official_exploration"
        adapted_memory = {}

    frozen_memory_root = memory_root / "frozen_memory"
    frozen_memory_manifest_path = memory_root / "frozen_memory_manifest.json"
    episodes_root = memory_root / "_episodes"
    source_target = _infer_mobilegpt_target_from_source_run_log(item)
    explicit_target_package = _mobilegpt_target_package_from_open_target_app(
        args.mobilegpt_open_target_app
    )
    effective_task_params = (
        dict(task_params_override)
        if task_params_override is not None
        else dict(item.params or {})
    )
    official_task_app = ""
    if not source_run_log.is_file():
        from src.integrations.android_world.run_episode import (
            instantiate_androidworld_task,
        )

        official_task = instantiate_androidworld_task(
            android_world_root=args.android_world_root,
            task_name=item.task,
            task_params=effective_task_params or None,
            task_seed=int(task_seed if task_seed is not None else args.source_seed),
        )
        if not effective_task_params:
            effective_task_params = dict(getattr(official_task, "params", {}) or {})
        official_task_app = next(
            (
                str(app).strip()
                for app in getattr(official_task, "app_names", ())
                if str(app).strip()
            ),
            "",
        )
    parameter_target_package = _mobilegpt_target_package_from_task_params(
        effective_task_params
    )
    target_package, target_source = _select_mobilegpt_bootstrap_package(
        explicit_package=explicit_target_package,
        sealed_package=str(adapted_memory.get("target_package") or ""),
        source_package=str(source_target.get("target_package") or ""),
        parameter_package=parameter_target_package,
    )
    if not target_package and official_task_app:
        target_package = official_task_app
        target_source = "androidworld_task_app"
    target_app = (
        explicit_target_package
        if explicit_target_package
        else str(
            adapted_memory.get("target_app")
            or source_target.get("target_app")
            or target_package
            or ""
        ).strip()
    )
    memory_app = _mobilegpt_memory_app_key(
        source_memory_root,
        memory_app=target_app,
        target_package=target_package,
        task_name=item.task,
    )
    memory_condition = "empty_cold_start" if cold_start else "provided_memory"
    source_memory_digest, source_memory_file_count = mobilegpt_memory.mobilegpt_memory_digest(
        source_memory_root
    )
    adapted_manifest = dict(adapted_memory.get("manifest") or {})
    adapted_source_stats = dict(adapted_memory.get("source_stats_summary") or {})
    adapted_source_stats_record = dict(adapted_manifest.get("source_stats") or {})
    adapted_official_result = (
        dict(adapted_manifest.get("official_source_result") or {})
        if adapted_manifest
        else {}
    )
    mobilegpt_prep = {
        "type": source_prep_type,
        "stats": adapted_source_stats,
        "wall_sec": _coerce_float(adapted_source_stats_record.get("wall_sec")),
        "official_validator_success": adapted_official_result.get(
            "official_validator_success"
        ),
        "manifest_path": str(adapted_memory.get("manifest_path") or ""),
        "manifest_sha256": str(adapted_memory.get("manifest_sha256") or ""),
        "memory_sha256": str(
            adapted_memory.get("memory_sha256") or source_memory_digest
        ),
        "shared_across_targets": True,
    }

    _write_method_memory_manifest(
        memory_root=memory_root,
        task=item.task,
        method=method,
        memory_mode=f"mobilegpt_single_episode_{memory_condition}",
        source_seed=int(args.source_seed),
        evaluation_seed=task_seed,
        attempt_id=attempt_id,
        target_inputs_read=False,
        artifacts={
            "runner": "stock_mobilegpt_single_episode",
            "initial_memory_condition": memory_condition,
            "episode_memory_policy": "isolated_attempt_copy_on_write",
            "source_memory_root": str(source_memory_root),
            "source_memory_sha256": source_memory_digest,
            "source_memory_file_count": source_memory_file_count,
            "function_conversion_enabled": False,
            "adapted_native_memory_manifest": str(
                adapted_memory.get("manifest_path") or ""
            ),
            "adapted_native_memory_manifest_sha256": str(
                adapted_memory.get("manifest_sha256") or ""
            ),
            "native_source_prep": mobilegpt_prep,
            "frozen_memory_root": str(frozen_memory_root),
            "frozen_memory_manifest": str(frozen_memory_manifest_path),
            "source_runlog_target_inference": source_target,
            "target_package": target_package,
            "target_app": target_app,
            "memory_app": memory_app,
            "target_source": target_source,
            "target_inputs_read": False,
            "target_observations_read": False,
            "validator_state_read": False,
        },
    )

    records: list[dict[str, Any]] = []
    failed = 0
    if args.dry_run:
        frozen_memory = {
            "schema_version": "omniflow.mobilegpt_frozen_memory.v1",
            "source_memory_root": str(source_memory_root),
            "frozen_memory_root": str(frozen_memory_root),
            "digest": "dry-run",
            "file_count": 0,
            "read_only": True,
        }
    else:
        frozen_memory = freeze_mobilegpt_memory(
            source_memory_root,
            frozen_memory_root,
        )
        with frozen_memory_manifest_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(frozen_memory, indent=2, ensure_ascii=False))
            handle.write("\n")
    records.append(
        {
            "label": "mobilegpt:freeze-initial-memory",
            "returncode": 0,
            "output_path": str(frozen_memory_root),
            "command": "",
            "task": item.task,
            "method": method,
            "device": targets[0].label,
            "status": "initial_memory_frozen",
            "summary_exclude": True,
            "metadata": {
                "memory_root": str(memory_root),
                "initial_memory_condition": memory_condition,
                "frozen_memory": frozen_memory,
                "frozen_memory_manifest": str(frozen_memory_manifest_path),
            },
        }
    )

    browser_task_prepare, browser_task_server = _start_mobilegpt_browser_task_server(
        item=item,
        memory_root=memory_root,
        android_world_root=args.android_world_root,
        task_params_override=task_params_override,
        dry_run=bool(args.dry_run),
    )
    if browser_task_prepare:
        records.append(
            {
                "label": "mobilegpt:browser-task-server",
                "returncode": 0,
                "output_path": str(memory_root / "browser_http"),
                "command": "",
                "task": item.task,
                "method": method,
                "device": targets[0].label,
                "status": "browser_task_server_started",
                "summary_exclude": True,
                "metadata": {
                    "memory_root": str(memory_root),
                    "browser_task_prepare": browser_task_prepare,
                },
            }
        )

    try:
        for target in targets:
            episode_root = episodes_root / target.label
            episode_memory_root = episode_root / "mobilegpt_memory"
            stats_jsonl = episode_root / "mobilegpt_stats.jsonl"
            stats_summary_path = episode_root / "mobilegpt_stats_summary.json"
            if not args.dry_run:
                if episode_root.exists():
                    raise FileExistsError(
                        f"immutable_mobilegpt_episode_exists:{episode_root}"
                    )
                episode_root.mkdir(parents=True)
                prepare_mobilegpt_episode_memory(
                    frozen_memory_root,
                    episode_memory_root,
                    expected_digest=str(frozen_memory.get("digest") or ""),
                    expected_file_count=int(frozen_memory.get("file_count") or 0),
                )
            else:
                episode_memory_root.mkdir(parents=True, exist_ok=True)
            device_target_package = _resolve_mobilegpt_target_package(
                target_package,
                adb_path=str(args.adb_path or ""),
                serial=target.serial,
                android_world_root=args.android_world_root,
            )
            server_spec = build_mobilegpt_server_command(
                "server",
                mobilegpt_root=args.mobilegpt_root,
                mobilegpt_memory_root=episode_memory_root,
                mobilegpt_memory_manifest=source_manifest_path,
                embedding_model=MOBILEGPT_EMBEDDING_MODEL,
                chat_model=FORMAL_MODEL,
                stats_jsonl=stats_jsonl,
                server_host=args.mobilegpt_server_host,
                port=int(args.mobilegpt_port),
                serial=target.serial,
                adb_path=args.adb_path,
                target_package=device_target_package,
                # ``target_package`` is the installed Android package.  The
                # official Memory reader, however, indexes its task CSV by
                # the source memory app name (for example ``markor``).  Keep
                # those two identities separate so the reader opens the
                # sealed RunLog task path instead of an empty package-named
                # task database.
                target_app=_mobilegpt_server_task_app(
                    memory_app,
                    device_target_package,
                ),
                target_task_name=args.task,
                write_through_memory=bool(cold_start),
            )
            browser_task_url = str(browser_task_prepare.get("url") or "").strip()
            if browser_task_url:
                server_spec = replace(
                    server_spec,
                    env={
                        **server_spec.env,
                        "OMNIFLOW_MOBILEGPT_BROWSER_TASK_URL": browser_task_url,
                        "OMNIFLOW_MOBILEGPT_ADB_PATH": str(args.adb_path or "").strip(),
                    },
                    metadata={
                        **server_spec.metadata,
                        "browser_task_url": browser_task_url,
                        "browser_task_serial": target.serial,
                    },
                )
            server: subprocess.Popen[Any] | None = None
            episode_record: dict[str, Any] | None = None
            try:
                server, server_returncode = _start_background_command(
                    server_spec,
                    dry_run=bool(args.dry_run),
                    warmup_sec=float(args.mobilegpt_server_warmup_sec),
                )
                records.append(
                    _command_record_from_spec(
                        server_spec,
                        task=item.task,
                        method=method,
                        device=target.label,
                        returncode=server_returncode,
                        status=(
                            "episode_server_started"
                            if server_returncode == 0
                            else "episode_server_failed"
                        ),
                        summary_exclude=True,
                        extra_metadata={
                            "memory_root": str(memory_root),
                            "initial_memory_condition": memory_condition,
                            "episode_memory_root": str(episode_memory_root),
                        },
                    )
                )
                if server_returncode != 0:
                    failed += 1
                    if args.fail_fast:
                        break
                    continue
                episode_spec = build_mobilegpt_command(
                    item,
                    method_name=method,
                    target=target,
                    android_world_root=args.android_world_root,
                    output_root=output_root,
                    stats_jsonl=stats_jsonl,
                    mobilegpt_root=args.mobilegpt_root,
                    server_host=args.mobilegpt_server_host,
                    server_port=int(args.mobilegpt_port),
                    target_package=device_target_package,
                    max_steps=int(args.max_steps or MAX_STEPS),
                    task_random_seed=task_seed,
                    fixed_task_seed=not bool(args.no_fixed_task_seed),
                    fixed_task_params=not bool(args.no_fixed_task_params),
                    task_params_override=effective_task_params,
                    perform_emulator_setup=bool(args.perform_emulator_setup),
                    adb_path=args.adb_path,
                    start_timeout_sec=float(args.mobilegpt_wait_start_timeout_sec),
                    finish_timeout_sec=(
                        float(args.mobilegpt_episode_wait_timeout_sec)
                        if args.mobilegpt_episode_wait_timeout_sec is not None
                        else (
                            # Leave a small parent-process grace window so
                            # official_forward can persist protocol/raw logs
                            # and validator output before run_command applies
                            # the outer task deadline.  Without this, a
                            # long-running MobileGPT episode is killed at the
                            # exact same instant as its own timeout and the
                            # canonical attempt directory can remain empty.
                            max(1.0, float(args.timeout_sec) - 15.0)
                            if args.timeout_sec and args.timeout_sec > 0
                            else DEFAULT_MOBILEGPT_EPISODE_WAIT_TIMEOUT_SEC
                        )
                    ),
                    app_ready_timeout_sec=float(
                        args.mobilegpt_app_ready_timeout_sec
                    ),
                    app_ready_poll_sec=float(args.mobilegpt_app_ready_poll_sec),
                    timeout_sec=float(args.timeout_sec or 0),
                    server_log_path=str(
                        server_spec.metadata.get("log_path") or ""
                    ),
                )
                episode_spec.metadata.update(
                    {
                        "memory_root": str(memory_root),
                        "initial_memory_condition": memory_condition,
                        "episode_memory_root": str(episode_memory_root),
                        "mobilegpt_stats_summary": str(stats_summary_path),
                        "mobilegpt_prep": mobilegpt_prep,
                        "mobilegpt_lifecycle": "single_episode",
                        "mobilegpt_memory_read_only": False,
                        "mobilegpt_memory_reusable": False,
                        "mobilegpt_memory_write_policy": (
                            "isolated_attempt_copy_on_write"
                        ),
                        "model": str(args.model or "").strip(),
                    }
                )
                returncode = run_command(
                    episode_spec,
                    dry_run=bool(args.dry_run),
                )
                episode_record = _command_record_from_spec(
                    episode_spec,
                    task=item.task,
                    method=method,
                    device=target.label,
                    returncode=returncode,
                    status="completed" if returncode == 0 else "command_failed",
                    summary_exclude=False,
                )
                records.append(episode_record)
                canonical_run_log: Path | None = None
                if not args.dry_run and episode_spec.output_path is not None:
                    canonical_run_log = _materialize_mobilegpt_canonical_run_log(
                        output_path=episode_spec.output_path,
                        item=item,
                        target=target,
                        task_seed=task_seed,
                    )
                # The official MobileGPT client can stop with its bounded
                # step-budget code after the device has already reached the
                # AndroidWorld goal.  Keep that process code in the raw
                # result, but let the canonical official validator decide
                # the experiment outcome used by the outer scheduler.
                validator_success = False
                if canonical_run_log is not None and canonical_run_log.is_file():
                    try:
                        canonical_payload = json.loads(
                            canonical_run_log.read_text(encoding="utf-8")
                        )
                    except (OSError, json.JSONDecodeError):
                        canonical_payload = {}
                    validator = (
                        canonical_payload.get("validator")
                        if isinstance(canonical_payload, dict)
                        else None
                    )
                    validator_success = bool(
                        isinstance(validator, dict)
                        and validator.get("official") is True
                        and validator.get("success") is True
                    )
                failed += int(
                    not validator_success
                    if canonical_run_log is not None
                    else returncode != 0
                )
            finally:
                _stop_background_command(server)

            if args.dry_run:
                if failed and args.fail_fast:
                    break
                continue

            mobilegpt_summary = mobilegpt_memory.summarize_mobilegpt_stats(stats_jsonl)
            stats_summary_path.write_text(
                json.dumps(mobilegpt_summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            audit_path, audit = audit_mobilegpt_episode_memory(
                episode_memory_root,
                expected_digest=str(frozen_memory.get("digest") or ""),
                expected_file_count=int(frozen_memory.get("file_count") or 0),
            )
            if episode_record is not None:
                episode_record.setdefault("metadata", {})[
                    "working_memory_audit"
                ] = audit
            records.append(
                {
                    "label": "mobilegpt:audit-episode-memory",
                    "returncode": 0,
                    "output_path": str(audit_path),
                    "command": "",
                    "task": item.task,
                    "method": method,
                    "device": target.label,
                    "status": audit["status"],
                    "summary_exclude": True,
                    "metadata": {
                        "memory_root": str(memory_root),
                        "initial_memory_condition": memory_condition,
                        "memory_audit": audit,
                    },
                }
            )

            if failed and args.fail_fast:
                break
    finally:
        _stop_background_command(browser_task_server)

    if not args.dry_run:
        frozen_digest, frozen_file_count = mobilegpt_memory.mobilegpt_memory_digest(frozen_memory_root)
        frozen_unchanged = frozen_digest == frozen_memory.get(
            "digest"
        ) and frozen_file_count == frozen_memory.get("file_count")
        records.append(
            {
                "label": "mobilegpt:verify-initial-memory",
                "returncode": 0 if frozen_unchanged else 1,
                "output_path": str(frozen_memory_root),
                "command": "",
                "task": item.task,
                "method": method,
                "device": targets[0].label,
                "status": (
                    "frozen_memory_unchanged"
                    if frozen_unchanged
                    else "frozen_memory_modified"
                ),
                "summary_exclude": True,
                "metadata": {
                    "memory_root": str(memory_root),
                    "expected_digest": frozen_memory.get("digest"),
                    "actual_digest": frozen_digest,
                    "expected_file_count": frozen_memory.get("file_count"),
                    "actual_file_count": frozen_file_count,
                },
            }
        )
        failed += int(not frozen_unchanged)
    return records, failed


def run_task(args: argparse.Namespace) -> int:
    if args.method not in (*ENABLED_METHODS, SOURCE_METHOD, AUTODROID_MEMORY_METHOD):
        raise ValueError(f"method_not_enabled:{args.method}")
    # The public runner is intentionally the only experiment control surface.
    # Keep this normalization here as a second boundary so a direct invocation
    # of the atomic runner cannot create a second protocol variant.
    args.source_seed = SOURCE_SEED
    requested_evaluation_seed = getattr(args, "evaluation_seed", None)
    args.task_random_seed = (
        SOURCE_SEED
        if args.method == SOURCE_METHOD
        else int(requested_evaluation_seed)
        if requested_evaluation_seed is not None
        else TASK_SEED
    )
    args.max_steps = SOURCE_MAX_STEPS if args.method == SOURCE_METHOD else MAX_STEPS
    baseline_without_functions = args.method == "omniflow" and not str(
        getattr(args, "memory", "") or ""
    ).strip()
    args.max_fallback_steps = (
        SOURCE_MAX_STEPS if args.method == SOURCE_METHOD else MAX_FALLBACK_STEPS
    )
    args.timeout_sec = TASK_DEADLINE_SEC
    args.planner_timeout_sec = (
        None if baseline_without_functions else PLANNER_TIMEOUT_SEC
    )
    args.model = require_runtime_model(
        args.method,
        omniflow_model() if args.method == "omniflow" else FORMAL_MODEL,
    )
    args.planner_provider = ""
    args.random_task_seed = False
    args.no_fixed_task_seed = False
    args.no_fixed_task_params = False
    args.perform_emulator_setup = False
    args.task_params_json = ""
    args.mobilegpt_server_host = "0.0.0.0"
    args.mobilegpt_server_warmup_sec = 30.0
    args.mobilegpt_wait_start_timeout_sec = float(TASK_DEADLINE_SEC)
    args.mobilegpt_episode_wait_timeout_sec = None
    args.mobilegpt_app_ready_timeout_sec = DEFAULT_MOBILEGPT_APP_READY_TIMEOUT_SEC
    args.mobilegpt_app_ready_poll_sec = DEFAULT_MOBILEGPT_APP_READY_POLL_SEC
    args.mobilegpt_open_target_app = ""
    args.appagent_model = APPAGENT_MODEL
    memory = str(getattr(args, "memory", "") or "").strip()
    if memory:
        if args.method == "omniflow":
            # The public --memory contract accepts the canonical Memory
            # directory.  The episode owner consumes its single store.json;
            # normalize that directory here without changing the execution
            # flow or introducing a second input path.
            supplied_memory = resolve_path(memory)
            args.store_path = str(
                supplied_memory / "store.json"
                if supplied_memory.is_dir()
                else supplied_memory
            )
        elif args.method == "mobilegpt":
            args.mobilegpt_source_memory_root = memory
        elif args.method == "appagent":
            args.appagent_memory_root = memory
        elif args.method in {"fixed_replay", "t3a_hint"}:
            args.source_run_log = memory
        elif args.method == AUTODROID_MEMORY_METHOD:
            args.autodroid_memory_root = memory
    # An explicitly supplied OmniFlow Memory must fail at the launcher
    # boundary when its Store is absent.  FunctionStore intentionally permits
    # a missing path so source/no-memory runs can use an empty Store; that
    # permissive behavior must not turn a typo in a formal Memory path into a
    # long Planner fallback episode.
    if args.method == "omniflow" and str(args.store_path or "").strip():
        resolved_store_path = resolve_path(args.store_path)
        if not resolved_store_path.is_file():
            raise FileNotFoundError(
                f"omniflow_memory_store_missing:{resolved_store_path}"
            )
    if args.method == AUTODROID_MEMORY_METHOD:
        autodroid_memory_root = resolve_path(
            str(getattr(args, "autodroid_memory_root", "") or "")
        )
        if not autodroid_memory_root.is_dir():
            raise FileNotFoundError(
                f"autodroid_memory_root_missing:{autodroid_memory_root}"
            )
    selected = _select_from_args(args)
    if len(selected) != 1:
        raise ValueError("result requires exactly one selected --task entry")
    item = selected[0]
    if args.method != SOURCE_METHOD:
        # Keep one protocol-wide episode budget.  Per-source trajectory
        # clipping made the effective ``--max-steps`` vary by task and could
        # terminate a valid cross-device recovery before the Planner had the
        # configured budget.  The formal protocol is the single owner here.
        args.max_steps = MAX_STEPS
        args.max_fallback_steps = (
            None
            if baseline_without_functions
            else min(MAX_FALLBACK_STEPS, MAX_STEPS)
        )
    methods = (args.method,)
    targets = parse_device_targets(args.device)
    if len(targets) != 1:
        raise ValueError("result requires exactly one device")
    if targets[0].serial == SOURCE_DEVICE[1]:
        args.task_random_seed = SOURCE_SEED
    else:
        args.task_random_seed = (
            int(args.evaluation_seed)
            if args.evaluation_seed is not None
            else TASK_SEED
        )
    args.mobilegpt_port = _mobilegpt_server_port(targets[0].console_port)
    mobilegpt_source_run_log = item.source_run_log
    mobilegpt_source_run_log_sha256s: tuple[str, ...] = ()
    if str(item.source_run_log) not in {"", "."} and item.source_run_log.is_file():
        mobilegpt_source_run_log_sha256s = (
            sha256_file(mobilegpt_source_run_log),
        )
    attempt_root, _ = _task_managed_output_root(args.output_path)
    source_seed = int(args.source_seed)
    source_hint_store_path = (
        resolve_path(args.store_path)
        if str(args.store_path or "").strip()
        else None
    )
    archive_root_value = str(
        os.environ.get("OMNIFLOW_ANDROIDWORLD_ARCHIVE_ROOT") or ""
    ).strip()
    output_root = (
        resolve_path(archive_root_value)
        if archive_root_value
        else _source_seed_output_root(attempt_root, source_seed)
    )
    attempt_id = attempt_root.name
    task_seed = (
        random.randint(1, 2**31 - 1)
        if bool(args.random_task_seed)
        else args.task_random_seed
    )
    if args.method == SOURCE_METHOD:
        # Source collection is the only phase that may use its own source
        # parameters. Target evaluation must never replay those parameters.
        task_params_override = dict(item.params or {})
    else:
        # Build one fresh official task instance from the evaluation seed and
        # share its generated parameters across target devices and adapters.
        item = _evaluation_item_for_seed(
            item,
            android_world_root=args.android_world_root,
            task_seed=int(task_seed),
        )
        task_params_override = dict(item.params or {})
    command_records: list[dict[str, Any]] = []
    failed = 0

    for method in methods:
        memory_root = attempt_root / "memory" / method
        # A dry-run is an inspection of the command contract and must not
        # reserve or create an immutable result directory.
        if not args.dry_run:
            _claim_method_memory_root(memory_root)
        source_action_hint_path: Path | None = None
        appagent_docs_root: Path | None = None
        if _is_mobilegpt_method(method):
            mobilegpt_records, mobilegpt_failed = _run_result_mobilegpt(
                args=args,
                item=item,
                targets=targets,
                output_root=output_root,
                task_params_override=task_params_override,
                task_seed=task_seed,
                method=method,
                attempt_id=attempt_id,
                source_run_log=mobilegpt_source_run_log,
                compatible_source_sha256s=mobilegpt_source_run_log_sha256s,
            )
            command_records.extend(mobilegpt_records)
            failed += mobilegpt_failed
            if failed and args.fail_fast:
                break
            continue

        if method == "omniflow":
            store_text = str(args.store_path or "").strip()
            store_path = resolve_path(store_text) if store_text else None
        elif method == AUTODROID_MEMORY_METHOD:
            store_path = memory_root / "unused-store.json"
        else:
            store_path = memory_root / "unused-store.json"

        if method == "appagent":
            source_memory_root = resolve_path(args.appagent_memory_root)
            appagent_manifest = _read_object(
                source_memory_root / "appagent_manifest.json"
            )
            appagent_docs_root = resolve_relative_reference(
                appagent_manifest.get("demo_docs_root"),
                base=source_memory_root,
            ).expanduser()
            if not appagent_docs_root.is_dir():
                raise FileNotFoundError(
                    f"appagent_demo_docs_missing:{appagent_docs_root}"
                )

        if method == "fixed_replay":
            replay_run_log, replay_materialization, replay_profile = (
                _materialize_replay_run_log_for_memory(
                    item,
                    memory_root=memory_root,
                )
            )
            _write_method_memory_manifest(
                memory_root=memory_root,
                task=item.task,
                method=method,
                memory_mode="source_runlog_replay",
                source_seed=source_seed,
                evaluation_seed=task_seed,
                attempt_id=attempt_id,
                source_run_log=item.source_run_log,
                artifacts={
                    "source_run_log": str(item.source_run_log),
                    "source_run_log_sha256": sha256_file(item.source_run_log),
                    "recorded_source_seed": item.replay_seed,
                    "replay_run_log": str(replay_run_log),
                    "replay_run_log_sha256": sha256_file(replay_run_log),
                    "source_materialization": replay_materialization,
                    "replay_memory_root": str(memory_root),
                },
            )
        elif method == "t3a_hint":
            official_agent_name = "t3a_gpt4"
            hint_item = item
            source_action_hint_path = _source_action_hint_path_for_item(
                hint_item,
                output_root=memory_root,
            )
            _write_method_memory_manifest(
                memory_root=memory_root,
                task=item.task,
                method=method,
                memory_mode="source_action_hint",
                source_seed=source_seed,
                evaluation_seed=task_seed,
                attempt_id=attempt_id,
                source_run_log=hint_item.source_run_log,
                artifacts={
                    "source_run_log": str(item.source_run_log),
                    "source_run_log_sha256": sha256_file(item.source_run_log),
                    "recorded_source_seed": item.replay_seed,
                    "source_action_hint_path": str(source_action_hint_path),
                    "source_action_hint_sha256": sha256_file(source_action_hint_path),
                    "semantic_source": (
                        "omniflow_function_store"
                        if source_hint_store_path is not None
                        else "source_run_log"
                    ),
                    "source_store": str(source_hint_store_path or ""),
                    "source_store_sha256": (
                        sha256_file(source_hint_store_path)
                        if source_hint_store_path is not None
                        else None
                    ),
                    "official_agent_name": official_agent_name,
                    "uses_omniflow_agent": False,
                    "uses_source_action_hints": True,
                    "hint_mode": "official_goal_reference_trace",
                    "state_backend": "androidworld_official",
                },
            )
        for target in targets:
            if method == SOURCE_METHOD:
                spec = build_task_command(
                    item,
                    android_world_root=args.android_world_root,
                    output_root=output_root,
                    method_name=SOURCE_METHOD,
                    device_label=target.label,
                    serial=target.serial,
                    console_port=target.console_port,
                    adb_path=args.adb_path,
                    max_steps=int(args.max_steps or MAX_STEPS),
                    timeout_sec=int(args.timeout_sec or 0),
                    max_fallback_steps=int(args.max_steps or MAX_STEPS),
                    task_random_seed=task_seed,
                    fixed_task_seed=not bool(args.no_fixed_task_seed),
                    fixed_task_params=(
                        task_params_override is not None
                        and not bool(args.no_fixed_task_params)
                    ),
                    task_params_override=task_params_override,
                    perform_emulator_setup=bool(args.perform_emulator_setup),
                    planner_provider=args.planner_provider,
                    model=str(args.model or ""),
                    planner_timeout_sec=args.planner_timeout_sec,
                    store_path=None,
                    omnitransfer_root=args.omnitransfer_root,
                )
            elif method == "fixed_replay":
                spec = build_replay_command(
                    item,
                    android_world_root=args.android_world_root,
                    output_root=output_root,
                    replay_memory_root=memory_root,
                    method_name=method,
                    device_label=target.label,
                    serial=target.serial,
                    console_port=target.console_port,
                    adb_path=args.adb_path,
                    max_steps=args.max_steps,
                    timeout_sec=int(args.timeout_sec or 0),
                    task_random_seed=task_seed,
                    fixed_task_params=not bool(args.no_fixed_task_params),
                    task_params_override=task_params_override,
                    perform_emulator_setup=bool(args.perform_emulator_setup),
                    planner_provider=args.planner_provider,
                    model=args.model,
                )
            elif method == "appagent":
                if appagent_docs_root is None:
                    raise ValueError("appagent_memory_docs_root_missing")
                spec = build_appagent_command(
                    item,
                    method_name=method,
                    target=target,
                    android_world_root=args.android_world_root,
                    output_root=output_root,
                    appagent_root=args.appagent_root,
                    docs_root=appagent_docs_root,
                    max_steps=int(args.max_steps or MAX_STEPS),
                    timeout_sec=int(args.timeout_sec or TASK_DEADLINE_SEC),
                    task_random_seed=int(task_seed),
                    task_params=task_params_override,
                    adb_path=args.adb_path,
                )
            elif method == "t3a_hint":
                spec = build_official_command(
                    item,
                    android_world_root=args.android_world_root,
                    output_root=output_root,
                    method_name=method,
                    official_agent_name="t3a_gpt4",
                    source_action_hint_path=source_action_hint_path,
                    device_label=target.label,
                    serial=target.serial,
                    console_port=target.console_port,
                    adb_path=args.adb_path,
                    max_steps=int(args.max_steps or MAX_STEPS),
                    timeout_sec=int(args.timeout_sec or 0),
                    task_random_seed=task_seed,
                    fixed_task_seed=not bool(args.no_fixed_task_seed),
                    fixed_task_params=(
                        task_params_override is not None
                        and not bool(args.no_fixed_task_params)
                    ),
                    task_params_override=task_params_override,
                    perform_emulator_setup=bool(args.perform_emulator_setup),
                    model=str(args.model or ""),
                )
            elif method == AUTODROID_MEMORY_METHOD:
                autodroid_memory_path = resolve_path(
                    str(getattr(args, "autodroid_memory_root", "") or "")
                )
                if not autodroid_memory_path.is_dir():
                    raise FileNotFoundError(
                        f"autodroid_memory_root_missing:{autodroid_memory_path}"
                    )
                validate_autodroid_memory(
                    autodroid_memory_path,
                    source_run_log=item.source_run_log,
                    task_name=item.task,
                )
                spec = build_task_command(
                    item,
                    android_world_root=args.android_world_root,
                    output_root=output_root,
                    method_name=method,
                    agent_name=method,
                    device_label=target.label,
                    serial=target.serial,
                    console_port=target.console_port,
                    adb_path=args.adb_path,
                    max_steps=int(args.max_steps or MAX_STEPS),
                    timeout_sec=int(args.timeout_sec or 0),
                    task_random_seed=task_seed,
                    fixed_task_seed=not bool(args.no_fixed_task_seed),
                    fixed_task_params=not bool(args.no_fixed_task_params),
                    task_params_override=task_params_override,
                    perform_emulator_setup=bool(args.perform_emulator_setup),
                    model=str(args.model or ""),
                    autodroid_memory_root=autodroid_memory_path,
                )
            else:
                spec = build_task_command(
                    item,
                    android_world_root=args.android_world_root,
                    output_root=output_root,
                    method_name=method,
                    device_label=target.label,
                    serial=target.serial,
                    console_port=target.console_port,
                    adb_path=args.adb_path,
                    max_steps=int(args.max_steps or MAX_STEPS),
                    timeout_sec=(
                        None
                        if baseline_without_functions
                        else int(args.timeout_sec or 0)
                    ),
                    max_fallback_steps=args.max_fallback_steps,
                    task_random_seed=task_seed,
                    fixed_task_seed=not bool(args.no_fixed_task_seed),
                    fixed_task_params=not bool(args.no_fixed_task_params),
                    task_params_override=task_params_override,
                    perform_emulator_setup=bool(args.perform_emulator_setup),
                    planner_provider=args.planner_provider,
                    model=args.model,
                    planner_timeout_sec=args.planner_timeout_sec,
                    store_path=store_path,
                    omnitransfer_root=args.omnitransfer_root,
                )
            spec.metadata["memory_root"] = str(memory_root)
            returncode = run_command(spec, dry_run=args.dry_run)
            if method == "omniflow" and not args.dry_run and spec.output_path is not None:
                try:
                    print(
                        format_omniflow_result_block(
                            output_path=spec.output_path,
                            target=target,
                            source_seed=source_seed,
                            evaluation_seed=task_seed,
                            store_path=store_path,
                        ),
                        flush=True,
                    )
                except Exception as error:  # noqa: BLE001
                    print(
                        f"[warn] failed to format OmniFlow result block: {error}",
                        flush=True,
                    )
            if returncode != 0:
                failed += 1
                if args.fail_fast:
                    break
        if failed and args.fail_fast:
            break

    print(
        json.dumps(
            {
                "task": item.task,
                "method": args.method,
                "device": targets[0].label,
                "status": "completed" if failed == 0 else "failed",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and summarize normal AndroidWorld experiment episodes."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    result_parser = subparsers.add_parser(
        "result",
        help=(
            "Run exactly one AndroidWorld method on one device for one archived "
            "task"
        ),
    )
    result_parser.add_argument(
        "--android-world-root", default=str(DEFAULT_ANDROID_WORLD_ROOT)
    )
    result_parser.add_argument(
        "--output-path",
        dest="output_path",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Temporary working directory for this episode.",
    )
    result_parser.add_argument("--task", required=True)
    result_parser.add_argument("--source-run-log", default="")
    result_parser.add_argument(
        "--evaluation-seed",
        type=int,
        default=None,
        help="Optional target task seed for an independent repeat; defaults to protocol seed.",
    )
    result_parser.add_argument(
        "--memory",
        default="",
        help="Optional method-specific Memory path, passed through unchanged.",
    )
    result_parser.add_argument(
        "--method",
        choices=(*METHODS, SOURCE_METHOD, AUTODROID_MEMORY_METHOD),
        default=DEFAULT_SOURCE_METHOD,
        help="One paper method for this AndroidWorld result.",
    )
    result_parser.add_argument(
        "--store-path",
        default="",
        help=("Validated omniflow.store.v2 required by the OmniFlow methods."),
    )
    result_parser.add_argument(
        "--autodroid-memory-root",
        default="",
        help="Explicit converted AutoDroid native Memory directory.",
    )
    result_parser.add_argument(
        "--omnitransfer-root",
        default="",
        help=(
            "Canonical or versioned OmniTransfer repository root. The configured "
            "root is authoritative over installed Python packages."
        ),
    )
    result_parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="One LABEL:SERIAL:PORT AndroidWorld target device.",
    )
    result_parser.add_argument("--adb-path", default="")
    result_parser.add_argument(
        "--mobilegpt-root", default=str(DEFAULT_MOBILEGPT_ROOT)
    )
    result_parser.add_argument(
        "--appagent-root",
        default=os.environ.get(
            "OMNIFLOW_APPAGENT_ROOT",
            "../OmniFlow/runtime/external/appagent",
        ),
    )
    result_parser.add_argument("--appagent-memory-root", default="")
    result_parser.add_argument(
        "--mobilegpt-source-memory-root",
        default="",
        help=(
            "Optional source-only native MobileGPT memory. If omitted, the "
            "same episode runner starts cold from empty memory; if supplied, "
            "it starts warm from an immutable snapshot."
        ),
    )
    result_parser.add_argument("--mobilegpt-port", type=int, default=12345)
    result_parser.add_argument(
        "--mobilegpt-server-warmup-sec",
        type=float,
        default=30.0,
    )
    result_parser.add_argument("--dry-run", action="store_true")
    result_parser.add_argument("--fail-fast", action="store_true")
    result_parser.set_defaults(func=run_task)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
