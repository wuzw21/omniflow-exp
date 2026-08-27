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
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Sequence
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omniflow.core.trajectory import (
    canonicalize_run_log,
)
from src.experiment.androidworld_paths import (
    canonical_device_seed_name,
    canonical_method_name,
)
from src.experiment.mobilegpt_contract import (
    MOBILEGPT_AUDIT_SCHEMA,
    MOBILEGPT_EMBEDDING_MODEL,
    MOBILEGPT_LEARNING_MODE,
    MOBILEGPT_MEMORY_MANIFEST,
    MOBILEGPT_MEMORY_SCHEMA,
    MOBILEGPT_SOURCE_METHOD,
)
from src.experiment.paths import (
    resolve_path,
    safe_component,
    safe_relative_path,
    sha256_file,
)
from src.experiment.protocol import (
    ANDROIDWORLD_REVISION,
    APPAGENT_MODEL,
    FORMAL_THINKING,
    DEFAULT_DEVICE,
    DEFAULT_METHOD,
    DEVICES,
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
    SOURCE_METHOD,
    SOURCE_DEVICE,
    SOURCE_SEED,
    TASK_DEADLINE_SEC,
    TASK_SEED,
    require_formal_model,
)
from src.experiment.run_process import run_process, start_process, stop_process
from src.experiment.source_records import CanonicalRunLog, SourceRunLogProfile
from src.integrations import mobilegpt_memory
from src.integrations.android_world.apps import resolve_androidworld_package
from src.integrations.official_forward import (
    resolve_mobilegpt_client_host,
)

DEFAULT_ANDROID_WORLD_ROOT = (
    Path("..")
    / "releases"
    / f"android-world-{ANDROIDWORLD_REVISION}"
)
DEFAULT_OUTPUT_ROOT = (
    Path("data") / "androidworld"
)
DEFAULT_MOBILEGPT_ROOT = Path("data") / "runtime" / "external" / "mobilegpt"
DEFAULT_MOBILEGPT_STATS_JSONL = (
    DEFAULT_OUTPUT_ROOT / "_mobilegpt_stats" / "mobilegpt_stats.jsonl"
)
DEFAULT_MOBILEGPT_STATS_SUMMARY = (
    DEFAULT_OUTPUT_ROOT / "_mobilegpt_stats" / "mobilegpt_stats_summary.json"
)
DEFAULT_MOBILEGPT_WAIT_START_TIMEOUT_SEC = 60.0
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


@dataclass(frozen=True)
class DeviceTarget:
    label: str
    serial: str
    console_port: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "serial": self.serial,
            "console_port": self.console_port,
        }


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _json_default(value: Any) -> Any:
    """Serialize AndroidWorld parameter dataclasses for subprocess JSON.

    Task parameters loaded from AndroidWorld can contain frozen SQLite-row
    dataclasses (for example CalendarEvent, Expense, and Recipe).  They are
    valid task parameters but are not directly serializable by ``json``.
    Preserve their field structure instead of stringifying them so the target
    task receives the same parameter values as the source run.
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
    # The pinned MobileGPT Server uses the OpenAI-compatible client API for
    # both chat and embedding calls. Publish the canonical aliases once at the
    # common subprocess boundary so execution and memory authoring use the
    # same endpoint contract.
    if not str(env.get("OPENAI_API_KEY") or "").strip() and str(
        env.get("LLMTHU_API_KEY") or ""
    ).strip():
        env["OPENAI_API_KEY"] = str(env["LLMTHU_API_KEY"])
    # A stale shell endpoint is an experimental variable.  The formal
    # protocol owns the provider seam for every AndroidWorld method.
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
    env["OMNIFLOW_PLANNER_MODEL"] = FORMAL_MODEL
    env["OMNIFLOW_MODEL_ENDPOINT_PROFILE"] = FORMAL_MODEL_ENDPOINT_PROFILE
    env["OMNIFLOW_PLANNER_TIMEOUT_SEC"] = str(PLANNER_TIMEOUT_SEC)
    env["OMNIFLOW_ANDROIDWORLD_LLM_MAX_TOKENS"] = str(FORMAL_MAX_TOKENS)
    env["OMNIFLOW_OPENAI_TIMEOUT_SEC"] = str(FORMAL_REQUEST_TIMEOUT_SEC)
    env["OMNIFLOW_OPENAI_RETRY_WAIT_SECONDS"] = str(FORMAL_RETRY_WAIT_SEC)
    env.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")
    return env






def _safe_stem(value: str, *, fallback: str = "task") -> str:
    return safe_component(
        value,
        fallback=fallback,
        max_length=120,
        strip_chars="._",
    )






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
            evaluation_seed=TASK_SEED,
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




_SOURCE_XML_EVIDENCE_KEYS = {
    "page",
    "xml",
    "observation_xml",
    "hierarchy_xml",
    "raw_xml",
    "parsed_xml",
    "encoded_xml",
    "html_xml",
    "screenshot",
    "screenshot_path",
    "screenshot_base64",
    "image_path",
    "image_base64",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"_read_error": f"missing: {path}"}
    except json.JSONDecodeError as exc:
        return {"_read_error": f"invalid JSON: {exc}"}
    return (
        decoded
        if isinstance(decoded, dict)
        else {"_read_error": "JSON root is not an object"}
    )


def _read_object(path: Path) -> dict[str, Any]:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return decoded


def _card_tool_name(card: Any) -> str:
    if not isinstance(card, dict):
        return ""
    tool_call = card.get("tool_call")
    if isinstance(tool_call, dict):
        name = tool_call.get("name")
        if name:
            return str(name).strip()
    action = card.get("action")
    if isinstance(action, dict):
        action_type = action.get("type")
        if action_type:
            return str(action_type).strip()
    return str(card.get("tool") or card.get("type") or "").strip()


def _card_args(card: Any) -> dict[str, Any]:
    if not isinstance(card, dict):
        return {}
    tool_call = card.get("tool_call")
    if isinstance(tool_call, dict) and isinstance(tool_call.get("arguments"), dict):
        return dict(tool_call["arguments"])
    if isinstance(card.get("args"), dict):
        return dict(card["args"])
    action = card.get("action")
    if isinstance(action, dict) and isinstance(action.get("params"), dict):
        return dict(action["params"])
    if isinstance(card.get("params"), dict):
        return dict(card["params"])
    return {}


def _is_empty_input_text_card(card: Any) -> bool:
    if _card_tool_name(card) != "input_text":
        return False
    args = _card_args(card)
    return "text" in args and str(args.get("text") or "") == ""


def _drop_empty_input_text_cards(payload: dict[str, Any]) -> int:
    cards = payload.get("cards")
    if not isinstance(cards, list):
        return 0
    filtered = [card for card in cards if not _is_empty_input_text_card(card)]
    removed_count = len(cards) - len(filtered)
    if removed_count <= 0:
        return 0

    payload["cards"] = filtered
    payload["step_count"] = len(filtered)
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics = {}
        payload["diagnostics"] = diagnostics
    normalizations = diagnostics.get("replay_normalizations")
    if not isinstance(normalizations, list):
        normalizations = []
        diagnostics["replay_normalizations"] = normalizations
    normalizations.append(
        {
            "kind": "drop_empty_input_text_cards",
            "removed_count": removed_count,
            "reason": "optional empty text inputs are no-op during replay",
        }
    )
    return removed_count












def profile_source_run_log(item: CanonicalRunLog) -> SourceRunLogProfile:
    data = _read_json(item.source_run_log)
    notes: list[str] = []
    read_error = str(data.get("_read_error") or "").strip()
    if read_error:
        return SourceRunLogProfile(
            task=item.task,
            source_run_log=item.source_run_log,
            replay_format="missing_or_invalid",
            step_count=0,
            card_count=0,
            latest_official_success_source=bool(
                item.meta.get("latest_official_success_source")
            ),
            direct_replay_ready=False,
            notes=(read_error,),
        )

    steps = data.get("steps")
    if isinstance(steps, list) and steps:
        return SourceRunLogProfile(
            task=item.task,
            source_run_log=item.source_run_log,
            replay_format="canonical_steps",
            step_count=len(steps),
            card_count=0,
            latest_official_success_source=bool(
                item.meta.get("latest_official_success_source")
            ),
            direct_replay_ready=True,
            notes=tuple(notes),
        )

    payload = data.get("payload")
    cards = payload.get("cards") if isinstance(payload, dict) else None
    if isinstance(payload, dict) and isinstance(cards, list) and cards:
        notes.append("wrapped payload must be materialized before OOB replay")
        return SourceRunLogProfile(
            task=item.task,
            source_run_log=item.source_run_log,
            replay_format="payload_cards",
            step_count=_coerce_int(payload.get("step_count"), len(cards)),
            card_count=len(cards),
            latest_official_success_source=bool(
                item.meta.get("latest_official_success_source")
            ),
            direct_replay_ready=True,
            notes=tuple(notes),
        )

    return SourceRunLogProfile(
        task=item.task,
        source_run_log=item.source_run_log,
        replay_format="unsupported",
        step_count=0,
        card_count=0,
        latest_official_success_source=bool(
            item.meta.get("latest_official_success_source")
        ),
        direct_replay_ready=False,
        notes=("no canonical steps or payload cards found",),
    )


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


def _normalize_observation_payload(value: Any) -> dict[str, Any]:
    observation = dict(value or {}) if isinstance(value, dict) else {}
    xml = str(
        observation.get("xml")
        or observation.get("observation_xml")
        or observation.get("page")
        or ""
    )
    if xml and "xml" not in observation:
        observation["xml"] = xml
    return observation


def _normalize_card_source_context(
    card: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    raw_context = card.get("source_context")
    if isinstance(raw_context, dict):
        for key in ("src_ctx", "source_context", "context"):
            nested = raw_context.get(key)
            if isinstance(nested, dict):
                raw_context = nested
                break
        context = dict(raw_context)
    else:
        context = {}

    before = _normalize_observation_payload(card.get("before"))
    if "page" not in context:
        page = str(before.get("xml") or before.get("observation_xml") or "").strip()
        if page:
            context["page"] = page

    for source_key, context_key in (
        ("screenshot_path", "screenshot_path"),
        ("image_path", "screenshot_path"),
        ("image_base64", "image_base64"),
        ("screenshot_base64", "screenshot_base64"),
        ("screenshot", "screenshot"),
    ):
        value = before.get(source_key)
        if value and context_key not in context:
            context[context_key] = value

    target_evidence = params.get("target_evidence")
    if isinstance(target_evidence, dict) and "element" not in context:
        context["element"] = dict(target_evidence)

    return context


def _canonical_steps_from_cards(cards: Sequence[Any]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for fallback_index, raw_card in enumerate(cards):
        if not isinstance(raw_card, dict):
            continue
        tool_name = _card_tool_name(raw_card)
        if not tool_name:
            continue
        params = _card_args(raw_card)
        source_context = _normalize_card_source_context(raw_card, params)
        if source_context and "source_context" not in params:
            params["source_context"] = source_context
        if params.get("target_description") and "clickPrompt" not in params:
            params["clickPrompt"] = str(params.get("target_description") or "")

        title = str(
            raw_card.get("title")
            or raw_card.get("summary")
            or f"{tool_name} from source card"
        ).strip()
        action = {
            "type": tool_name,
            "params": params,
            "success": bool(raw_card.get("success", True)),
            "description": title,
        }
        if source_context:
            action["source_context"] = source_context
        step_index = _coerce_int(raw_card.get("step_index"), fallback_index)
        steps.append(
            {
                "step_index": step_index,
                "status": str(raw_card.get("status") or "success"),
                "success": bool(raw_card.get("success", True)),
                "tool_call": {
                    "name": tool_name,
                    "params": params,
                    "reason": title,
                },
                "actions": [action],
                "executed_actions": [dict(action)],
                "observation_before_act": _normalize_observation_payload(
                    raw_card.get("before")
                ),
                "observation_after_act": _normalize_observation_payload(
                    raw_card.get("after")
                ),
                "source": raw_card.get("source") or "androidworld_source_card",
                "card_id": raw_card.get("card_id") or raw_card.get("tool_call_id"),
            }
        )
    return steps


def canonicalize_source_run_log(
    item: CanonicalRunLog,
) -> tuple[dict[str, Any], str, SourceRunLogProfile, Path | None]:
    """Return one canonical step-based runlog from the archived source asset."""

    profile = profile_source_run_log(item)
    data = _read_json(item.source_run_log)
    if data.get("_read_error"):
        raise ValueError(str(data.get("_read_error")))

    payload = dict(data.get("payload") or data)
    materialization = "original"
    cards = payload.get("cards")
    if isinstance(cards, list) and cards:
        payload = dict(payload)
        _drop_empty_input_text_cards(payload)
        cards = payload.get("cards") if isinstance(payload.get("cards"), list) else []
        steps = _canonical_steps_from_cards(cards)
        materialization = "payload_cards_to_canonical_steps"
    else:
        steps = list(payload.get("steps") or [])

    if not steps:
        raise ValueError(f"source runlog has no canonicalizable steps: {item.task}")

    run_id = str(
        payload.get("run_id")
        or data.get("run_id")
        or item.meta.get("run_id")
        or f"androidworld_canonical_source_{_safe_stem(item.task)}"
    ).strip()
    canonical = dict(payload)
    canonical.update(
        {
            "schema_version": str(
                canonical.get("schema_version") or "oob.run_log.canonical.v1"
            ),
            "run_id": run_id,
            "goal": str(canonical.get("goal") or item.goal),
            "success": bool(canonical.get("success", True)),
            "done_reason": str(canonical.get("done_reason") or "source_success"),
            "steps": steps,
            "step_count": len(steps),
            "source": str(canonical.get("source") or "androidworld_canonical_source"),
            "androidworld_task": str(canonical.get("androidworld_task") or item.task),
            "androidworld_params": dict(
                canonical.get("androidworld_params")
                if isinstance(canonical.get("androidworld_params"), dict)
                else item.params
            ),
        }
    )

    return canonical, materialization, profile, None


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
        effective_params, ensure_ascii=False, separators=(",", ":")
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
    if resolved_agent == "omniflow" and max_fallback_steps is not None:
        env["OMNIFLOW_ANDROIDWORLD_MAX_FALLBACK_STEPS"] = str(
            max(0, int(max_fallback_steps))
        )
    # The public launcher exports this setting, but make it explicit on every
    # child CommandSpec as well.  This prevents a scheduler worker from
    # silently falling back to AndroidWorld-native observe/act when the
    # campaign requested OOB transport.
    control_backend = str(
        os.environ.get("OMNIFLOW_ANDROIDWORLD_CONTROL_BACKEND", "oob")
    ).strip().lower() or "androidworld"
    if resolved_agent == "omniflow":
        env["OMNIFLOW_ANDROIDWORLD_CONTROL_BACKEND"] = control_backend
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
            effective_params, ensure_ascii=False, separators=(",", ":")
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
            float(timeout_sec) if timeout_sec is not None and timeout_sec > 0 else None
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
                if resolved_agent == "omniflow"
                and control_backend in {"oob", "omniflow", "oob_control"}
                else "androidworld"
            ),
            "action_backend": (
                "oob_control"
                if resolved_agent == "omniflow"
                and control_backend in {"oob", "omniflow", "oob_control"}
                else "androidworld"
            ),
            "native_androidworld_agent_io": not (
                resolved_agent == "omniflow"
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
    launch_only = (
        isinstance(official_reader, dict)
        and mobilegpt_memory.is_valid_mobilegpt_launch_only_memory(
            source_payload,
            audit,
            official_reader,
        )
    )
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
        or int(official_reader.get("action_row_count") or 0) < transition_count
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
        "teacher_prompt_used": True,
        "teacher_action_alignment_complete": True,
        "actions_supplied_to_mobilegpt": False,
        "source_reader_coverage_validation": False,
        "direct_subtasks_from_runlog": False,
    }
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
        "teacher_forcing": semantic_learning,
        "synthetic_subtasks": not semantic_learning,
        "semantic_subtasks": semantic_learning,
        "original_mobilegpt_prompts": semantic_learning,
        "actions_supplied_to_mobilegpt": not semantic_learning,
        "source_transitions_supplied": True,
        "source_success_boundary_supplied": True,
        "runlog_transition_compilation": not semantic_learning,
        "complete_transition_mapping": True,
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
                "teacher_action_alignment_complete": True,
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


def _mobilegpt_memory_embedding_model(
    memory_root: Path,
    *,
    manifest_path: Path | None = None,
) -> str:
    """Return the embedding model sealed with one MobileGPT memory bundle."""

    manifest_path = manifest_path or memory_root.parent / MOBILEGPT_MEMORY_MANIFEST
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"mobilegpt_memory_manifest_unreadable:{manifest_path}"
        ) from error
    stats = payload.get("source_stats")
    models = stats.get("embedding_models") if isinstance(stats, dict) else None
    unique_models = sorted(
        {str(model).strip() for model in models or [] if str(model).strip()}
    )
    if unique_models and unique_models != [MOBILEGPT_EMBEDDING_MODEL]:
        raise ValueError(
            "mobilegpt_embedding_model_mismatch:"
            f"memory={','.join(unique_models)} expected={MOBILEGPT_EMBEDDING_MODEL}"
        )
    return MOBILEGPT_EMBEDDING_MODEL


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
        requested_embedding_model = str(embedding_model or "").strip()
        if requested_embedding_model and requested_embedding_model != MOBILEGPT_EMBEDDING_MODEL:
            raise ValueError("mobilegpt_embedding_model_is_fixed")
        resolved_embedding_model = _mobilegpt_memory_embedding_model(
            resolved_memory_root,
            manifest_path=(
                resolve_path(mobilegpt_memory_manifest)
                if mobilegpt_memory_manifest
                else None
            ),
        )
        chat_model = require_formal_model(str(chat_model or FORMAL_MODEL).strip())
        from src.integrations.official_forward import prepare_mobilegpt_server

        staged = resolved_memory_root.parent / "official_server_workspace"
        forward = prepare_mobilegpt_server(
            official_root=root,
            memory_root=resolved_memory_root,
            workspace=staged,
            embedding_model=resolved_embedding_model,
            chat_model=chat_model,
            write_through_memory=bool(write_through_memory),
        )
        staged_server_root = Path(forward["server_root"])
        env["MOBILEGPT_STATS_JSONL"] = str(resolve_path(stats_jsonl, root=repo_root))
        # Keep the official action/list output as direct JSON while preserving
        # the formal Qwen reasoning mode.  Leave enough completion room for
        # selector/derive JSON on screens with many available actions.
        env["MOBILEGPT_THINKING"] = FORMAL_THINKING
        env["MOBILEGPT_MAX_TOKENS"] = "2048"
        env["MOBILEGPT_LIST_MAX_TOKENS"] = "2048"
        env["MOBILEGPT_REQUEST_TIMEOUT_SEC"] = "60"
        # Bound the total provider chat calls for one formal episode.  The
        # staged official server may issue several planner calls per device
        # action; without this cap a stalled episode can consume hundreds of
        # calls before the 600-second wall deadline.
        env["MOBILEGPT_MAX_CHAT_CALLS"] = "64"
        env["MOBILEGPT_RESPONSE_RETRIES"] = "1"
        env["MOBILEGPT_EMBEDDING_MODEL"] = resolved_embedding_model
        if chat_model:
            env["MOBILEGPT_CHAT_MODEL"] = chat_model
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
                "embedding_model": resolved_embedding_model,
                "chat_model": chat_model,
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


def _last_jsonl_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        text = line.strip()
        if not text:
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _result_device_name(target: DeviceTarget) -> str:
    label = str(target.label or "").casefold()
    if "standard" in label or "small" in label or "pixel6" in label:
        return "Standard / Pixel 6 Pro"
    if "fold" in label:
        return "Fold"
    if "tablet" in label or "wxga" in label:
        return "Tablet"
    return str(target.label or target.serial or "Unknown")


def _percentage_text(value: Any) -> str:
    try:
        percentage = float(value) * 100.0
    except (TypeError, ValueError):
        return "n/a"
    if percentage.is_integer():
        return f"{int(percentage)}%"
    return f"{percentage:.1f}%"


def format_omniflow_result_block(
    *,
    output_path: Path,
    target: DeviceTarget,
    source_seed: int,
    evaluation_seed: int,
    store_path: Path | None,
) -> str:
    """Render one copy-ready result block from the sealed official evidence."""

    run_log_path = output_path / "run_log.json"
    task_result = _last_jsonl_object(output_path / "task_results.jsonl")
    run_log = _read_json(run_log_path) if run_log_path.is_file() else {}
    validator = (
        dict(run_log.get("validator") or {})
        if isinstance(run_log.get("validator"), dict)
        else {}
    )
    official = (
        bool(task_result.get("official_validator_used"))
        if "official_validator_used" in task_result
        else bool(validator.get("official"))
    )
    success = (
        bool(task_result.get("success"))
        if "success" in task_result
        else bool(validator.get("success"))
    )
    completed = bool(task_result or run_log)
    actions = _coerce_int(
        task_result.get("actions_executed")
        or len(list(run_log.get("steps") or []))
    )
    reuse = task_result.get("reuse_metrics")
    if not isinstance(reuse, dict):
        from src.integrations.android_world.methods import reuse_metrics

        reuse = reuse_metrics(
            "omniflow",
            actions_executed=actions,
            canonical_run=run_log,
        )
    physical_actions = _coerce_int(reuse.get("physical_action_count"))
    state_changing_actions = _coerce_int(
        reuse.get("state_changing_physical_action_count")
    )
    function_actions = _coerce_int(reuse.get("reuse_numerator"))
    function_denominator = _coerce_int(reuse.get("reuse_denominator"))
    diagnostics = (
        dict(run_log.get("diagnostics") or {})
        if isinstance(run_log.get("diagnostics"), dict)
        else {}
    )
    execution_summary = (
        dict(diagnostics.get("execution_summary") or {})
        if isinstance(diagnostics.get("execution_summary"), dict)
        else {}
    )
    fallback_steps = _coerce_int(
        task_result.get("fallback_steps")
        if "fallback_steps" in task_result
        else execution_summary.get("fallback_steps")
    )
    model_calls = _coerce_int(
        task_result.get("model_calls")
        if "model_calls" in task_result
        else execution_summary.get("model_calls")
    )
    prompt_tokens = _coerce_int(
        task_result.get("prompt_tokens")
        if "prompt_tokens" in task_result
        else execution_summary.get("prompt_tokens")
    )
    completion_tokens = _coerce_int(
        task_result.get("completion_tokens")
        if "completion_tokens" in task_result
        else execution_summary.get("completion_tokens")
    )
    total_tokens = _coerce_int(
        task_result.get("total_tokens")
        if "total_tokens" in task_result
        else execution_summary.get("total_tokens")
        or execution_summary.get("tokens")
    )
    execution_ms = _coerce_float(
        task_result.get("execution_duration_ms")
        if "execution_duration_ms" in task_result
        else execution_summary.get("execution_duration_ms")
    )
    lifecycle_ms = _coerce_float(task_result.get("duration_ms"))
    action_types = {
        str((step.get("action") or {}).get("action_type") or "").strip()
        for step in run_log.get("steps") or []
        if isinstance(step, dict) and isinstance(step.get("action"), dict)
    }
    action_note = "（含非物理 answer）" if "answer" in action_types else ""
    validator_count = f"{1 if official and success else 0}/1" if official else "未覆盖"
    resolved_store = (
        str(store_path.expanduser().resolve()) if store_path is not None else "未提供"
    )
    digest = sha256_file(run_log_path) if run_log_path.is_file() else "未生成"
    return "\n".join(
        (
            "|   |",
            "| - |",
            "",
            f"完成：{'是' if completed else '否'}",
            "",
            f"通过：{'是' if official and success else '否'}（official validator {validator_count}）",
            "",
            f"设备：{_result_device_name(target)}",
            "",
            f"source_seed={int(source_seed)}；evaluation_seed={int(evaluation_seed)}",
            "",
            (
                f"actions={actions}{action_note}；physical={physical_actions}；"
                f"state-changing={state_changing_actions}"
            ),
            "",
            (
                f"Function={function_actions}/{function_denominator}；"
                f"reuse={_percentage_text(reuse.get('reuse_rate'))}；"
                f"fallback={fallback_steps}；model_calls={model_calls}"
            ),
            "",
            f"tokens={prompt_tokens}/{completion_tokens}/{total_tokens}",
            "",
            (
                f"execution={execution_ms / 1000.0:.3f}s（排除 setup/validator）；"
                f"lifecycle={lifecycle_ms / 1000.0:.3f}s"
            ),
            "",
            f"Store：{resolved_store}",
            "",
            f"RunLog：{run_log_path.resolve()}",
            "",
            f"SHA-256：{digest}",
        )
    )






























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




_RESULT_NON_EXECUTED_STATUSES = {
    "INVALID_MEMORY_LEAKAGE",
    "env_failed",
    "init_failed",
    "setup_failed",
}


def _is_mobilegpt_method(method: str) -> bool:
    return str(method or "").strip() == "mobilegpt"






def _task_params_override_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    if not str(getattr(args, "task_params_json", "") or "").strip():
        return None
    decoded = json.loads(str(args.task_params_json))
    if not isinstance(decoded, dict):
        raise ValueError("--task-params-json must be a JSON object")
    return dict(decoded)


def _t3a_hint_forbidden_values(params: dict[str, Any]) -> tuple[str, ...]:
    values: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                collect(item)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                collect(item)
            return
        if value is None or isinstance(value, bool):
            return
        text = re.sub(r"\s+", " ", str(value)).strip().casefold()
        if text:
            values.add(text)
            suffix = Path(text).suffix
            if suffix:
                stem = Path(text).stem.strip().casefold()
                if len(stem) >= 3:
                    values.add(stem)

    collect(params)
    return tuple(sorted(values, key=len, reverse=True))


def _t3a_hint_redacted_text(
    value: Any,
    *,
    forbidden_values: Sequence[str],
    max_len: int = 320,
) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    lowered = text.casefold()
    if not text or any(
        marker in lowered
        for marker in (
            "<hierarchy",
            "<?xml",
            "screenshot",
            ".png",
            ".jpg",
            ".jpeg",
        )
    ):
        return ""
    for forbidden in forbidden_values:
        if len(forbidden) < 3:
            continue
        text = re.sub(
            re.escape(forbidden),
            "<current task value>",
            text,
            flags=re.IGNORECASE,
        )
    if len(text) > max_len:
        text = text[: max(0, max_len - 3)].rstrip() + "..."
    return text


def _t3a_hint_text(
    value: Any,
    *,
    forbidden_values: Sequence[str],
    max_len: int = 100,
) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    lowered = text.casefold()
    if not text or any(
        marker in lowered
        for marker in (
            "<hierarchy",
            "<?xml",
            "screenshot",
            ".png",
            ".jpg",
            ".jpeg",
        )
    ):
        return ""
    for forbidden in forbidden_values:
        if lowered == forbidden:
            return ""
        if len(forbidden) >= 3 and re.search(
            rf"(?<!\w){re.escape(forbidden)}(?!\w)",
            lowered,
        ):
            return ""
    if len(text) > max_len:
        text = text[: max(0, max_len - 3)].rstrip() + "..."
    return text


def _t3a_hint_step_action(step: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(step, dict):
        return "", {}
    for key in ("tool_call", "action", "selected_action"):
        action = step.get(key)
        if not isinstance(action, dict):
            continue
        name = str(
            action.get("name")
            or action.get("type")
            or action.get("tool")
            or action.get("action_type")
            or ""
        ).strip()
        raw_params = action.get("params")
        if raw_params is None:
            raw_params = action.get("arguments")
        if raw_params is None:
            raw_params = action.get("args")
        if name:
            if isinstance(raw_params, dict):
                return name, dict(raw_params)
            return name, {
                str(param_key): param_value
                for param_key, param_value in action.items()
                if param_key
                not in {"name", "type", "tool", "action_type", "params", "arguments", "args"}
            }
    for key in ("executed_actions", "actions"):
        actions = step.get(key)
        if not isinstance(actions, list):
            continue
        for action in actions:
            if not isinstance(action, dict):
                continue
            name = str(
                action.get("name") or action.get("type") or action.get("tool") or ""
            ).strip()
            raw_params = action.get("params")
            if raw_params is None:
                raw_params = action.get("arguments")
            if raw_params is None:
                raw_params = action.get("args")
            if name:
                return name, dict(raw_params or {}) if isinstance(
                    raw_params, dict
                ) else {}
    return "", {}




def _t3a_hint_target(
    params: dict[str, Any],
    *,
    forbidden_values: Sequence[str],
) -> str:
    for key in (
        "target_description",
        "target",
        "label",
        "content_desc",
        "contentDescription",
    ):
        text = _t3a_hint_text(
            params.get(key),
            forbidden_values=forbidden_values,
        )
        if text:
            return text
    for container_key in ("target_evidence", "source_context"):
        container = params.get(container_key)
        if not isinstance(container, dict):
            continue
        for key in ("target_description", "label", "text", "content_desc"):
            text = _t3a_hint_text(
                container.get(key),
                forbidden_values=forbidden_values,
            )
            if text:
                return text
    return ""


def _t3a_hint_step_summary(
    step: Any,
    *,
    forbidden_values: Sequence[str],
) -> str:
    if not isinstance(step, dict):
        return ""
    candidates: list[Any] = []
    metadata = step.get("metadata")
    if isinstance(metadata, dict):
        candidates.append(metadata.get("summary"))
    candidates.extend((step.get("summary"), step.get("title"), step.get("description")))
    for action_key in ("tool_call", "action", "selected_action"):
        action = step.get(action_key)
        if isinstance(action, dict):
            candidates.extend((action.get("reason"), action.get("description")))
    actions = step.get("actions")
    if isinstance(actions, list):
        candidates.extend(
            action.get("description")
            for action in actions
            if isinstance(action, dict)
        )
    for candidate in candidates:
        summary = re.sub(
            r"^Action selected:\s*\{.*?\}\.\s*",
            "",
            str(candidate or ""),
        )
        summary = _t3a_hint_redacted_text(
            summary,
            forbidden_values=forbidden_values,
        )
        if summary:
            return summary
    return ""


def _t3a_hint_summary_target(summary: str) -> str:
    patterns = (
        r"\bClicked\s+(?:the\s+)?(?P<target>.+?)(?:\s*\(index|\s+to\b|\s*;|\s*[—–-])",
        r"\bLong-pressed\s+(?:the\s+)?(?P<target>.+?)(?:\s*\(index|\s+to\b|\s*;|\s*[—–-])",
        r"\b(?:Entered|Typed|Inserted)\s+.+?\s+(?:into|in)\s+(?:the\s+)?(?P<target>.+?)(?:\s*\(index|\s*;|\s*[—–-])",
        r"\bRenamed\s+.+?\s+in\s+(?:the\s+)?(?P<target>.+?)(?:\s*\(index|\s*;|\s*[—–-])",
    )
    for pattern in patterns:
        match = re.search(pattern, summary, flags=re.IGNORECASE)
        if not match:
            continue
        target = re.sub(r'["“”]', "", match.group("target")).strip(" \t'.,")
        if target:
            return target
    return ""


def _t3a_hint_bounds(value: str) -> tuple[int, int, int, int] | None:
    match = re.fullmatch(
        r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]",
        str(value or "").strip(),
    )
    if not match:
        return None
    return tuple(int(item) for item in match.groups())


def _t3a_hint_source_node(
    step: Any,
    *,
    forbidden_values: Sequence[str],
    editable_only: bool = False,
    preceding_step: Any = None,
) -> dict[str, str]:
    if not isinstance(step, dict):
        return {}
    observation = next(
        (
            value
            for value in (
                step.get("observation"),
                step.get("before"),
                step.get("observation_before_act"),
            )
            if isinstance(value, dict)
        ),
        {},
    )
    _, params = _t3a_hint_step_action(step)
    index = params.get("index")
    elements = observation.get("ui_elements")
    if (
        isinstance(index, int)
        and not isinstance(index, bool)
        and isinstance(elements, list)
        and 0 <= index < len(elements)
        and isinstance(elements[index], dict)
    ):
        element = elements[index]
        if editable_only and not bool(element.get("is_editable")):
            return {}
        evidence: dict[str, str] = {}
        for source_key, target_key in (
            ("class_name", "class_name"),
            ("text", "text"),
            ("content_description", "content_description"),
            ("resource_id", "resource_id"),
            ("resource_name", "resource_id"),
            ("package_name", "package_name"),
        ):
            if target_key == "text" and bool(element.get("is_editable")):
                continue
            value = _t3a_hint_redacted_text(
                element.get(source_key),
                forbidden_values=forbidden_values,
                max_len=120,
            )
            if value and target_key not in evidence:
                evidence[target_key] = value
        if evidence:
            return evidence
    if isinstance(elements, list):
        x = params.get("x")
        y = params.get("y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            ui_candidates: list[tuple[tuple[bool, bool, float], dict[str, str]]] = []
            for element in elements:
                if not isinstance(element, dict):
                    continue
                bbox = element.get("bbox_pixels")
                if not isinstance(bbox, dict):
                    continue
                try:
                    left = float(bbox["x_min"])
                    top = float(bbox["y_min"])
                    right = float(bbox["x_max"])
                    bottom = float(bbox["y_max"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not left <= x <= right or not top <= y <= bottom:
                    continue
                if editable_only and not bool(element.get("is_editable")):
                    continue
                evidence: dict[str, str] = {}
                for source_key, target_key in (
                    ("class_name", "class_name"),
                    ("text", "text"),
                    ("content_description", "content_description"),
                    ("resource_id", "resource_id"),
                    ("resource_name", "resource_id"),
                    ("package_name", "package_name"),
                ):
                    value = _t3a_hint_redacted_text(
                        element.get(source_key),
                        forbidden_values=forbidden_values,
                        max_len=120,
                    )
                    if value and target_key not in evidence:
                        evidence[target_key] = value
                actionable = any(
                    bool(element.get(key))
                    for key in ("is_clickable", "is_editable", "is_scrollable")
                )
                area = max(0.0, right - left) * max(0.0, bottom - top)
                ui_candidates.append(((not actionable, not bool(evidence), area), evidence))
            if ui_candidates:
                return min(ui_candidates, key=lambda candidate: candidate[0])[1]
    forest = next(
        (
            value
            for value in (
                observation.get("forest"),
                observation.get("xml"),
                observation.get("observation_xml"),
                observation.get("page"),
            )
            if isinstance(value, str) and "<node" in value
        ),
        "",
    )
    if not forest:
        source_context = params.get("source_context")
        if isinstance(source_context, dict):
            forest = next(
                (
                    value
                    for value in (
                        source_context.get("page"),
                        source_context.get("xml"),
                        source_context.get("observation_xml"),
                    )
                    if isinstance(value, str) and "<node" in value
                ),
                "",
            )
    if not isinstance(forest, str) or "<node" not in forest:
        return {}
    try:
        root = ET.fromstring(forest)
    except ET.ParseError:
        return {}
    action_name, _ = _t3a_hint_step_action(step)
    action_name = action_name.strip().lower()
    x = params.get("x")
    y = params.get("y")
    action_index = (
        index
        if isinstance(index, int) and not isinstance(index, bool)
        else None
    )
    metadata = step.get("metadata")
    raw_summary = (
        str(metadata.get("summary") or "") if isinstance(metadata, dict) else ""
    )
    index_match = re.search(r'"index"\s*:\s*(\d+)', raw_summary)
    indexed_id = index_match.group(1) if index_match else ""
    candidates: list[tuple[tuple[int, int, int], dict[str, str]]] = []
    for node in root.iter():
        if str(node.tag).rsplit("}", 1)[-1] != "node":
            continue
        attributes = {str(key): str(value) for key, value in node.attrib.items()}
        if editable_only and not (
            attributes.get("editable") == "true"
            or "edittext" in attributes.get("class", "").casefold()
        ):
            continue
        bounds = _t3a_hint_bounds(attributes.get("bounds", ""))
        coordinate_match = False
        if bounds is not None and isinstance(x, (int, float)) and isinstance(y, (int, float)):
            left, top, right, bottom = bounds
            coordinate_match = left <= x <= right and top <= y <= bottom
        index_match_node = bool(indexed_id and attributes.get("id") == indexed_id)
        if not coordinate_match and not index_match_node:
            continue
        actionable = any(
            attributes.get(key) == "true"
            for key in ("clickable", "editable", "scrollable")
        )
        semantic = any(
            attributes.get(key)
            for key in ("text", "content-desc", "resource-id")
        )
        area = (
            max(0, bounds[2] - bounds[0]) * max(0, bounds[3] - bounds[1])
            if bounds is not None
            else 10**12
        )
        candidates.append(((not actionable, not semantic, area), attributes))
    if (
        not candidates
        and action_name in {"click", "tap", "long_press"}
        and action_index is not None
    ):
        nodes = [
            node
            for node in root.iter()
            if str(node.tag).rsplit("}", 1)[-1] == "node"
        ]

        def index_matches(attributes: dict[str, str]) -> bool:
            node_id = attributes.get("id", "")
            return node_id == str(action_index) or node_id.endswith(
                f":{action_index}"
            )

        exact_matches = [
            {str(key): str(value) for key, value in node.attrib.items()}
            for node in nodes
            if index_matches({str(key): str(value) for key, value in node.attrib.items()})
        ]
        if len(exact_matches) == 1:
            attributes = exact_matches[0]
            candidates.append(((False, False, 0), attributes))
        else:
            # Manual source logs sometimes retain only an AndroidWorld action
            # index while the serialized observation retains XML. If the
            # clicked control disappears in the immediately following frame,
            # recover it from the before/after tree transition. Only accept a
            # unique clickable transition; ambiguous transitions remain a
            # conversion failure and must be recollected with source element
            # evidence.
            next_observation = (
                step.get("next_observation")
                if isinstance(step, dict)
                else None
            )
            next_forest = next_observation.get("xml") if isinstance(next_observation, dict) else ""
            if isinstance(next_forest, str) and "<node" in next_forest:
                try:
                    next_root = ET.fromstring(next_forest)
                except ET.ParseError:
                    next_root = None
                if next_root is not None:
                    next_ids = {
                        str(node.attrib.get("id"))
                        for node in next_root.iter()
                        if node.attrib.get("id")
                    }
                    transition_matches = []
                    for node in nodes:
                        attributes = {str(key): str(value) for key, value in node.attrib.items()}
                        node_id = attributes.get("id")
                        if not node_id or node_id in next_ids:
                            continue
                        if attributes.get("clickable") != "true":
                            continue
                        bounds = _t3a_hint_bounds(attributes.get("bounds", ""))
                        area = (
                            max(0, bounds[2] - bounds[0]) * max(0, bounds[3] - bounds[1])
                            if bounds is not None
                            else 10**12
                        )
                        transition_matches.append((area, attributes))
                    if len(transition_matches) == 1:
                        _, attributes = transition_matches[0]
                        candidates.append(((False, False, 0), attributes))
    if not candidates and action_name in {
        "input_text",
        "type_text",
        "set_text",
        "enter_text",
    } and not (
        (isinstance(x, (int, float)) and isinstance(y, (int, float)))
        or indexed_id
    ):
        # Some Android accessibility snapshots do not expose focus state. In
        # that case, use the preceding click and its post-action XML to recover
        # the selected editable element without carrying source coordinates
        # into the T3A hint.
        preceding_name, preceding_params = _t3a_hint_step_action(preceding_step)
        if preceding_name.strip().lower() in {"click", "tap"}:
            preceding_observation = (
                preceding_step.get("next_observation")
                if isinstance(preceding_step, dict)
                else None
            )
            preceding_forest = next(
                (
                    value
                    for value in (
                        preceding_observation.get("forest"),
                        preceding_observation.get("xml"),
                        preceding_observation.get("observation_xml"),
                        preceding_observation.get("page"),
                    )
                    if isinstance(value, str) and "<node" in value
                ),
                "",
            ) if isinstance(preceding_observation, dict) else ""
            preceding_x = preceding_params.get("x")
            preceding_y = preceding_params.get("y")
            if preceding_forest and isinstance(preceding_x, (int, float)) and isinstance(preceding_y, (int, float)):
                try:
                    preceding_root = ET.fromstring(preceding_forest)
                except ET.ParseError:
                    preceding_root = None
                if preceding_root is not None:
                    for node in preceding_root.iter():
                        if str(node.tag).rsplit("}", 1)[-1] != "node":
                            continue
                        attributes = {str(key): str(value) for key, value in node.attrib.items()}
                        if attributes.get("editable") != "true":
                            continue
                        bounds = _t3a_hint_bounds(attributes.get("bounds", ""))
                        if bounds is None:
                            continue
                        left, top, right, bottom = bounds
                        if left <= preceding_x <= right and top <= preceding_y <= bottom:
                            candidates.append(((False, False, 0), attributes))
                            break
        editable_nodes: list[tuple[bool, int, dict[str, str]]] = []
        for node in root.iter():
            if str(node.tag).rsplit("}", 1)[-1] != "node":
                continue
            attributes = {str(key): str(value) for key, value in node.attrib.items()}
            if attributes.get("editable") != "true":
                continue
            bounds = _t3a_hint_bounds(attributes.get("bounds", ""))
            area = (
                max(0, bounds[2] - bounds[0]) * max(0, bounds[3] - bounds[1])
                if bounds is not None
                else 10**12
            )
            editable_nodes.append((attributes.get("focused") == "true", area, attributes))
        focused_nodes = [item for item in editable_nodes if item[0]]
        if focused_nodes:
            candidates.extend(
                ((False, False, area), attributes)
                for _, area, attributes in focused_nodes
            )
        elif len(editable_nodes) == 1:
            _, area, attributes = editable_nodes[0]
            candidates.append(((False, False, area), attributes))
        elif not candidates and editable_nodes:
            # AndroidWorld's file dialog opens with the first field selected,
            # even when the exported XML omits focused=true.
            _, area, attributes = editable_nodes[0]
            candidates.append(((False, False, area), attributes))
    if not candidates:
        return {}
    attributes = min(candidates, key=lambda item: item[0])[1]
    field_map = {
        "id": "node_id",
        "class": "class_name",
        "text": "text",
        "content-desc": "content_description",
        "resource-id": "resource_id",
        "package": "package_name",
    }
    evidence: dict[str, str] = {}
    for source_key, target_key in field_map.items():
        if target_key == "text" and attributes.get("editable") == "true":
            continue
        value = _t3a_hint_redacted_text(
            attributes.get(source_key),
            forbidden_values=forbidden_values,
            max_len=120,
        )
        if value:
            evidence[target_key] = value
    return evidence


def _t3a_semantic_hint_step(
    step: Any,
    *,
    forbidden_values: Sequence[str],
    preceding_step: Any = None,
) -> dict[str, Any] | None:
    name, params = _t3a_hint_step_action(step)
    action = name.strip().lower()
    if not action or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", action):
        return None
    if action in {"status", "finish", "done"}:
        return None
    semantic: dict[str, Any] = {"action": action}
    purpose = _t3a_hint_step_summary(
        step,
        forbidden_values=forbidden_values,
    )
    source_node = _t3a_hint_source_node(
        step,
        forbidden_values=forbidden_values,
        preceding_step=preceding_step,
    )
    target = _t3a_hint_target(params, forbidden_values=forbidden_values)
    if not target:
        target = _t3a_hint_summary_target(purpose)
    if not target:
        target = (
            source_node.get("content_description")
            or source_node.get("text")
            or source_node.get("resource_id")
        )
    if (
        action in {"input_text", "type_text", "set_text", "enter_text"}
        and not target
        and not source_node
    ):
        preceding_action, _ = _t3a_hint_step_action(preceding_step)
        if preceding_action.strip().lower() in {"click", "tap"}:
            source_node = _t3a_hint_source_node(
                preceding_step,
                forbidden_values=forbidden_values,
                editable_only=True,
                preceding_step=None,
            )
            if source_node:
                target = "editable text field selected by the preceding action"
    if action in {"click", "tap", "long_press", "input_text", "type_text", "set_text", "enter_text"} and not target and not source_node:
        raise ValueError(f"t3a_hint_unidentified_target:{action}")
    if target:
        semantic["target"] = target
    if purpose:
        semantic["purpose"] = purpose
    if source_node:
        semantic["source_node"] = source_node
    if action in {"open_app", "launch_app"}:
        app = _t3a_hint_text(
            params.get("app_name")
            or params.get("app")
            or params.get("package_name")
            or params.get("packageName"),
            forbidden_values=forbidden_values,
            max_len=80,
        )
        if app:
            semantic["app"] = app
    if action in {"swipe", "scroll"}:
        direction = _t3a_hint_text(
            params.get("direction"),
            forbidden_values=forbidden_values,
            max_len=24,
        )
        if direction:
            semantic["direction"] = direction
    if action in {"press_key", "key_event"}:
        key = _t3a_hint_text(
            params.get("key") or params.get("key_name"),
            forbidden_values=forbidden_values,
            max_len=24,
        )
        if key:
            semantic["key"] = key
    return semantic


def _source_action_hint_path_for_item(
    item: CanonicalRunLog,
    *,
    output_root: str | Path,
    repo_root: Path = REPO_ROOT,
) -> Path:
    payload, _, profile, _ = canonicalize_source_run_log(
        item,
    )
    forbidden_values = _t3a_hint_forbidden_values(item.params)
    source_steps = list(payload.get("steps") or [])
    semantic_source = "source_run_log"
    semantic_input_steps = source_steps
    semantic_preceding_steps = [None, *source_steps[:-1]]
    store_alignment_mode = "not_applicable"
    semantic_steps = []
    for step, preceding_step in zip(
        semantic_input_steps,
        semantic_preceding_steps,
        strict=True,
    ):
        semantic = _t3a_semantic_hint_step(
            step,
            forbidden_values=forbidden_values,
            preceding_step=preceding_step,
        )
        if semantic is not None:
            semantic_steps.append(semantic)
    if not semantic_steps:
        raise ValueError(f"source runlog produced no safe T3A hints: {item.task}")
    hint_payload = {
        "schema_version": "omniflow.t3a_semantic_hint.v2",
        "task": item.task,
        "semantic_source": semantic_source,
        "store_alignment_mode": store_alignment_mode,
        "source_step_count": len(source_steps),
        "semantic_step_count": len(semantic_steps),
        "steps": semantic_steps,
    }
    hint_dir = resolve_path(output_root, root=repo_root) / "_source_action_hints"
    hint_dir.mkdir(parents=True, exist_ok=True)
    hint_path = hint_dir / f"{_safe_stem(item.task)}.source_action_hints.json"
    hint_path.write_text(
        json.dumps(hint_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return hint_path.resolve()


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






_RESULT_METADATA_ROW_KEYS = (
    "source_run_log",
    "memory_root",
    "replay_run_log",
    "store_path",
    "task_random_seed",
    "max_steps",
    "max_fallback_steps",
    "timeout_sec",
    "planner_timeout_sec",
    "fixed_task_seed",
    "fixed_task_params",
    "task_params",
    "task_params_override",
    "serial",
    "console_port",
    "device_target",
    "planner_provider",
    "model",
    "backend",
    "state_backend",
    "action_backend",
    "native_androidworld_agent_io",
    "execution_backend",
    "uses_source_xml",
    "official_agent_name",
    "uses_omniflow_agent",
    "uses_source_action_hints",
    "uses_function_retrieval",
    "perform_emulator_setup",
)














def _next_attempt(setting_root: Path, group: str) -> Path:
    # One setting has one visible attempt.  A second run must be an explicit
    # operator decision, rather than silently creating another result variant.
    attempt_root = setting_root / group / "attempt_001"
    if attempt_root.exists():
        raise FileExistsError(f"immutable_attempt_exists:{attempt_root}")
    return attempt_root


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
    del fixed_task_seed
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
    client_host = resolve_mobilegpt_client_host(
        server_host,
        serial=target.serial,
        adb_path=adb_path,
    )
    client_argv = [
        sys.executable,
        "-m",
        "src.integrations.official_forward",
        "--baseline",
        "mobilegpt",
        "--root",
        str(resolve_path(mobilegpt_root, root=repo_root)),
        "--serial",
        target.serial,
        "--adb",
        str(adb_path or "adb"),
        "--host",
        str(client_host),
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
        "--handshake-timeout-sec",
        str(float(start_timeout_sec)),
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
        "MOBILEGPT_CLIENT_MODE": "official_accessibility",
        "OMNIFLOW_ANDROIDWORLD_CONTROL_BACKEND": str(
            client_runtime_env.get("OMNIFLOW_ANDROIDWORLD_CONTROL_BACKEND")
            or os.environ.get("OMNIFLOW_ANDROIDWORLD_CONTROL_BACKEND")
            or "oob"
        ),
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
        label=f"mobilegpt:official-accessibility:{target.label}",
        argv=client_argv,
        env=client_environment,
        cwd=repo_root,
        output_path=resolved_output,
        timeout_sec=(
            float(timeout_sec) if timeout_sec is not None and timeout_sec > 0 else None
        ),
        metadata={
            "mode": "mobilegpt_official_planner_native_accessibility",
            "device_target": target.to_dict(),
            "mobilegpt_stats_jsonl": str(stats_jsonl),
            "mobilegpt_server_host": str(server_host),
            "mobilegpt_server_port": int(server_port),
            "target_package": str(target_package or "").strip(),
            "official_lifecycle": "mobilegpt_server_and_official_accessibility_client",
            "official_server_entry": "Server/main.py",
            "official_client_entry": "src.integrations.official_forward",
            "official_client_class": "MobileGPTAccessibilityService",
            "official_client_output": str(client_output),
            "observe_backend": "mobilegpt_official_accessibility",
            "action_backend": "mobilegpt_official_accessibility",
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


def _configure_mobilegpt_formal_server(
    spec: CommandSpec,
    *,
    model: str = "",
) -> CommandSpec:
    normalized_model = str(model or "").strip()
    return replace(
        spec,
        env={
            **spec.env,
            "MOBILEGPT_CHAT_BASE_URL": FORMAL_MODEL_BASE_URL,
            "MOBILEGPT_EMBEDDING_BASE_URL": FORMAL_MODEL_BASE_URL,
            "MOBILEGPT_THINKING": FORMAL_THINKING,
            **(
                {"MOBILEGPT_CHAT_MODEL": normalized_model}
                if normalized_model
                else {}
            ),
        },
        metadata={
            **spec.metadata,
            **({"model": normalized_model} if normalized_model else {}),
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
        source_memory_root = resolve_path(source_memory_value)
        if not source_memory_root.is_dir():
            raise FileNotFoundError(
                f"mobilegpt_source_memory_missing:{source_memory_root}"
            )
        source_manifest_path = None
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
                chat_model=str(args.model or ""),
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
            server_spec = _configure_mobilegpt_formal_server(
                server_spec,
                model=str(args.model or ""),
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
                failed += int(returncode != 0)
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
    # The public runner is intentionally the only experiment control surface.
    # Keep this normalization here as a second boundary so a direct invocation
    # of the atomic runner cannot create a second protocol variant.
    args.source_seed = SOURCE_SEED
    args.task_random_seed = (
        SOURCE_SEED if args.method == SOURCE_METHOD else TASK_SEED
    )
    args.max_steps = MAX_STEPS
    args.max_fallback_steps = MAX_FALLBACK_STEPS
    args.timeout_sec = TASK_DEADLINE_SEC
    args.planner_timeout_sec = PLANNER_TIMEOUT_SEC
    args.model = require_formal_model()
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
            args.store_path = memory
        elif args.method == "mobilegpt":
            args.mobilegpt_source_memory_root = memory
        elif args.method == "appagent":
            args.appagent_memory_root = memory
        elif args.method == "fixed_replay":
            args.source_run_log = memory
    selected = _select_from_args(args)
    if len(selected) != 1:
        raise ValueError("result requires exactly one selected --task entry")
    item = selected[0]
    methods = (args.method,)
    targets = parse_device_targets(args.device)
    if len(targets) != 1:
        raise ValueError("result requires exactly one device")
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
    # The source RunLog is the one task-parameter authority shared by all
    # methods. There is no per-method parameter override in the formal run.
    task_params_override = dict(item.params or {})
    task_seed = (
        random.randint(1, 2**31 - 1)
        if bool(args.random_task_seed)
        else args.task_random_seed
    )
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
        else:
            store_path = memory_root / "unused-store.json"

        if method == "appagent":
            source_memory_root = resolve_path(args.appagent_memory_root)
            appagent_manifest = _read_object(
                source_memory_root / "appagent_manifest.json"
            )
            appagent_docs_root = Path(
                str(appagent_manifest.get("demo_docs_root") or "")
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
                    timeout_sec=int(args.timeout_sec or 0),
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
        "--memory",
        default="",
        help="Optional method-specific Memory path, passed through unchanged.",
    )
    result_parser.add_argument(
        "--method",
        choices=(*METHODS, SOURCE_METHOD),
        default=DEFAULT_SOURCE_METHOD,
        help="One paper method for this AndroidWorld result.",
    )
    result_parser.add_argument(
        "--store-path",
        default="",
        help=("Validated omniflow.store.v2 required by the OmniFlow methods."),
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
