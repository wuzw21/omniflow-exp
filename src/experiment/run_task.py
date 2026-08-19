#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
from dataclasses import dataclass, field, replace
import datetime
import hashlib
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
from typing import Any, Iterable, Sequence
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omniflow.core.trajectory import canonicalize_run_log
from omniflow.functions.assets import FunctionStore
from src.experiment.mobilegpt_contract import (
    MOBILEGPT_AUDIT_SCHEMA,
    MOBILEGPT_LEARNING_MODE,
    MOBILEGPT_MEMORY_MANIFEST,
    MOBILEGPT_MEMORY_SCHEMA,
    MOBILEGPT_PREP_TYPE,
    MOBILEGPT_PREP_TYPE_BY_SCHEMA,
    MOBILEGPT_SOURCE_METHOD,
    MOBILEGPT_SOURCE_METHOD_BY_SCHEMA,
)
from src.experiment.paths import (
    resolve_path,
    safe_component,
    safe_relative_path,
    sha256_file,
)
from src.experiment.run_process import run_process
from src.experiment.protocol import (
    ANDROIDWORLD_REVISION,
    DEFAULT_DEVICE,
    DEFAULT_METHOD,
    EPISODE_TIMEOUT_SEC,
    MAX_STEPS,
    METHODS,
    RESULT_COMMANDS_FILE,
    RESULT_MARKDOWN_FILE,
    RESULT_SCHEMA,
    RESULT_SUMMARY_FILE,
    SOURCE_SEED,
    STEP_TIMEOUT_SEC,
    TASK_SEED,
)
from src.experiment.result_registry import register_attempt_summary
from src.experiment.result_schema import RESULT_FIELDS, compact_result_row
from src.experiment.source_records import CanonicalRunLog, SourceRunLogProfile
from src.integrations.android_world.methods import reuse_metrics_from_result_row
from src.integrations.appagent import validate_appagent_memory
from src.integrations import mobilegpt_memory


def _load_mobilegpt_stats_summary(
    *,
    summary_path: str | Path | None,
    stats_jsonl_path: str | Path | None,
) -> dict[str, Any]:
    """Load the shared MobileGPT stats summary for result-row accounting."""

    return mobilegpt_memory._load_mobilegpt_stats_summary(
        summary_path=summary_path,
        stats_jsonl_path=stats_jsonl_path,
    )

DEFAULT_DATA_INDEX = REPO_ROOT / "data" / "current.json"
DEFAULT_ANDROID_WORLD_ROOT = (
    Path.home()
    / "Projects"
    / "Omni"
    / "releases"
    / f"android-world-{ANDROIDWORLD_REVISION}"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "data" / "androidworld"
)
DEFAULT_MOBILEGPT_ROOT = REPO_ROOT / "data" / "runtime" / "external" / "mobilegpt"
DEFAULT_MOBILEGPT_STATS_JSONL = (
    DEFAULT_OUTPUT_ROOT / "_mobilegpt_stats" / "mobilegpt_stats.jsonl"
)
DEFAULT_MOBILEGPT_STATS_SUMMARY = (
    DEFAULT_OUTPUT_ROOT / "_mobilegpt_stats" / "mobilegpt_stats_summary.json"
)
DEFAULT_MOBILEGPT_WAIT_START_TIMEOUT_SEC = 60.0
DEFAULT_MOBILEGPT_EPISODE_WAIT_TIMEOUT_SEC = 120.0
DEFAULT_MOBILEGPT_APP_READY_TIMEOUT_SEC = 15.0
DEFAULT_MOBILEGPT_APP_READY_POLL_SEC = 0.25
DEFAULT_TASK_RANDOM_SEED = TASK_SEED
DEFAULT_SOURCE_METHOD = DEFAULT_METHOD


@dataclass(frozen=True)
class CommandSpec:
    label: str
    argv: list[str]
    env: dict[str, str]
    cwd: Path
    output_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timeout_sec: float | None = None


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


def _rate(numerator: int | float, denominator: int | float) -> float:
    denominator = float(denominator or 0)
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / denominator, 6)


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
    if "MOBILEGPT_CHAT_MODEL" not in env and str(env.get("OPENAI_MODEL") or "").strip():
        env["MOBILEGPT_CHAT_MODEL"] = str(env["OPENAI_MODEL"]).strip()
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
        resolved_model = str(
            dotenv_env.get("OMNIFLOW_PLANNER_MODEL")
            or dotenv_env.get("OPENAI_MODEL")
            or os.environ.get("OMNIFLOW_PLANNER_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or ""
        ).strip()

    resolved_provider = str(planner_provider or "").strip()
    if not resolved_provider:
        has_openai_config = any(
            str(dotenv_env.get(key) or os.environ.get(key) or "").strip()
            for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")
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
    env.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")
    return env


@contextmanager
def _temporary_env(updates: dict[str, str | None]):
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _canonical_source_ref_path(
    value: str | Path,
    *,
    index_path: Path,
    repo_root: Path = REPO_ROOT,
) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()

    index_relative = (index_path.parent / path).resolve()
    if index_relative.exists():
        return index_relative

    for index_ancestor in index_path.parents:
        ancestor_relative = (index_ancestor / path).resolve()
        if ancestor_relative.exists():
            return ancestor_relative

    repo_relative = resolve_path(path, root=repo_root)
    if repo_relative.exists():
        return repo_relative

    return repo_relative


def _safe_stem(value: str, *, fallback: str = "task") -> str:
    return safe_component(
        value,
        fallback=fallback,
        max_length=120,
        strip_chars="._",
    )


def _stable_json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
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
    repo_root: Path = REPO_ROOT,
) -> Path:
    return (
        resolve_path(output_root, root=repo_root)
        / _safe_stem(task)
        / _safe_stem(method, fallback="method")
        / _device_label(
            explicit_label=device,
            serial=serial,
            console_port=console_port,
        )
    )


def _androidworld_validator_root(*, repo_root: Path = REPO_ROOT) -> Path:
    return repo_root / "data" / "runtime" / "evals" / "androidworld_validator"


def _result_registry_root(
    args: argparse.Namespace,
    *,
    attempt_root: Path,
) -> Path:
    explicit_runs = str(getattr(args, "result_registry_root", "") or "").strip()
    if explicit_runs:
        return resolve_path(explicit_runs)

    index_path = resolve_path(args.index)
    for candidate in (index_path.parent, *index_path.parents):
        if candidate.name == "androidworld_validator":
            return candidate / "runs"
    return attempt_root.parent / "_androidworld_result_registry"


def _task_managed_output_root(
    output_root: str | Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> tuple[Path, str]:
    """Keep the caller's exact immutable attempt root authoritative."""
    resolved = resolve_path(output_root, root=repo_root)
    canonical_shared_root = _androidworld_validator_root(repo_root=repo_root) / "runs"
    if resolved == canonical_shared_root:
        raise ValueError(
            f"output_root_must_be_fresh_attempt_child:{canonical_shared_root}"
        )
    return resolved, ""


def _source_seed_output_root(output_root: str | Path, source_seed: int) -> Path:
    return resolve_path(output_root) / f"source_seed_{int(source_seed)}"


def _claim_result_attempt(
    output_root: str | Path,
    *,
    task: str,
    methods: Sequence[str],
    source_seed: int,
    evaluation_seed: int | None,
    dry_run: bool = False,
) -> Path:
    root = resolve_path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "attempt_manifest.json"
    runner_path = Path(__file__).resolve()
    provenance: dict[str, Any] = {
        "runner": str(runner_path),
        "runner_sha256": sha256_file(runner_path),
    }
    manifest = {
        "schema_version": "omniflow.androidworld_attempt.v1",
        "attempt_id": root.name,
        "task_name": task,
        "methods": list(methods),
        "source_seed": int(source_seed),
        "evaluation_seed": evaluation_seed,
        "dry_run": bool(dry_run),
        "immutable": True,
        "provenance": provenance,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    try:
        with manifest_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, indent=2, ensure_ascii=False))
            handle.write("\n")
    except FileExistsError as exc:
        raise FileExistsError(f"immutable_attempt_exists:{root}") from exc
    return manifest_path.resolve()


def _method_root(output_root: str | Path, task: str, method: str) -> Path:
    return resolve_path(output_root) / _safe_stem(task) / _safe_stem(method)


def _method_memory_root(output_root: str | Path, task: str, method: str) -> Path:
    return _method_root(output_root, task, method) / "_memory"


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


def _parse_single_quoted_goal_field(goal: str, marker: str) -> str:
    match = re.search(rf"{re.escape(marker)}\s+'([^']*)'", goal)
    return str(match.group(1) if match else "").strip()


def _calendar_epoch_utc(
    *,
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int = 0,
) -> int:
    dt = datetime.datetime(
        int(year),
        int(month),
        int(day),
        int(hour),
        int(minute),
        tzinfo=datetime.timezone.utc,
    )
    return int(dt.timestamp())


def _calendar_row_from_flat_params(params: dict[str, Any]) -> dict[str, Any] | None:
    try:
        year = _coerce_int(params.get("year"), 2023)
        month = _coerce_int(params.get("month"), 10)
        day = _coerce_int(params.get("day"), 0)
        hour = _coerce_int(params.get("hour"), 0)
        duration_mins = _coerce_int(params.get("duration_mins"), 0)
    except Exception:
        return None
    title = str(params.get("event_title") or "").strip()
    if not day or not duration_mins or not title:
        return None
    start_ts = _calendar_epoch_utc(year=year, month=month, day=day, hour=hour)
    end_ts = start_ts + (duration_mins * 60)
    return {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "title": title,
        "location": "",
        "description": str(params.get("event_description") or ""),
        "repeat_interval": 0,
        "repeat_rule": 0,
        "reminder_1_minutes": -1,
        "reminder_2_minutes": -1,
        "reminder_3_minutes": -1,
        "reminder_1_type": 0,
        "reminder_2_type": 0,
        "reminder_3_type": 0,
        "repeat_limit": 0,
        "repetition_exceptions": "[]",
        "attendees": "",
        "import_id": "",
        "time_zone": "UTC",
        "flags": 0,
        "event_type": 1,
        "parent_id": 0,
        "last_updated": 0,
        "source": "imported-ics",
        "availability": 0,
        "color": 0,
        "type": 0,
        "id": -1,
    }


def complete_androidworld_task_params(
    task: str,
    goal: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Normalize archived source params for AndroidWorld validators."""

    completed = dict(params)
    if task.startswith("RecipeAdd") and "row_objects" not in completed:
        title = str(completed.get("title") or "").strip()
        if title:
            completed["row_objects"] = [
                {
                    "title": title,
                    "description": str(completed.get("description") or ""),
                    "servings": str(completed.get("servings") or ""),
                    "preparationTime": str(completed.get("preparationTime") or ""),
                    "source": str(completed.get("source") or ""),
                    "ingredients": str(completed.get("ingredients") or ""),
                    "directions": str(completed.get("directions") or ""),
                    "favorite": _coerce_int(completed.get("favorite"), 0),
                    "imageName": str(completed.get("imageName") or ""),
                    "recipeId": _coerce_int(completed.get("recipeId"), -1),
                }
            ]
            completed.setdefault("noise_row_objects", [])
            completed.setdefault("text_representation_type", "text_block")

    if task.startswith("SimpleCalendarAddOneEvent") and "row_objects" not in completed:
        completed.setdefault("year", 2023)
        completed.setdefault("month", 10)
        if "event_title" not in completed:
            title = _parse_single_quoted_goal_field(goal, "title")
            if title:
                completed["event_title"] = title
        if "event_description" not in completed:
            description = _parse_single_quoted_goal_field(goal, "description")
            if description:
                completed["event_description"] = description
        row = _calendar_row_from_flat_params(completed)
        if row is not None:
            completed["row_objects"] = [row]
            completed.setdefault("noise_row_objects", [])

    return completed


def load_canonical_source_index(
    index_path: str | Path = DEFAULT_DATA_INDEX,
    *,
    repo_root: Path = REPO_ROOT,
) -> list[CanonicalRunLog]:
    path = resolve_path(index_path, root=repo_root)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Canonical data index must be a JSON object: {path}")
    if data.get("schema_version") == "omniflow.data-index.v2":
        data = data.get("source_index")
        if not isinstance(data, dict):
            raise ValueError(f"Current data index has no source index: {path}")

    source_items: list[CanonicalRunLog] = []
    for task, raw_meta in data.items():
        if not isinstance(raw_meta, dict):
            continue
        retained = str(raw_meta.get("retained_source_run_log") or "").strip()
        if not retained:
            raise ValueError(f"source_run_log_missing_in_current_index:{task}")
        params = (
            raw_meta.get("params") if isinstance(raw_meta.get("params"), dict) else {}
        )
        params = complete_androidworld_task_params(
            str(task), str(raw_meta.get("goal") or ""), params
        )
        seed = _coerce_int(
            raw_meta.get("source_seed")
            or raw_meta.get("replay_seed")
            or raw_meta.get("collect_seed")
            or raw_meta.get("task_random_seed"),
            30,
        )
        source_items.append(
            CanonicalRunLog(
                task=str(task),
                goal=str(raw_meta.get("goal") or ""),
                params=dict(params),
                source_run_log=_canonical_source_ref_path(
                    retained,
                    index_path=path,
                    repo_root=repo_root,
                ),
                replay_seed=seed,
                step_count=_coerce_int(raw_meta.get("step_count"), 0),
                meta=dict(raw_meta),
            )
        )
    return source_items


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
    python_executable: str = sys.executable,
    repo_root: Path = REPO_ROOT,
) -> CommandSpec:
    resolved_device = _device_label(
        explicit_label=device_label,
        serial=serial,
        console_port=console_port,
    )
    resolved_method = _safe_stem(method_name, fallback=DEFAULT_SOURCE_METHOD)
    resolved_output = _experiment_run_dir(
        output_root,
        task=item.task,
        method=resolved_method,
        device=resolved_device,
        serial=serial,
        console_port=console_port,
        repo_root=repo_root,
    )
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
    fixed_task_params: bool = True,
    task_params_override: dict[str, Any] | None = None,
    perform_emulator_setup: bool = True,
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
    resolved_output = _experiment_run_dir(
        output_root,
        task=item.task,
        method=resolved_method,
        device=resolved_device,
        serial=serial,
        console_port=console_port,
        repo_root=repo_root,
    )
    resolved_task_seed = int(
        item.replay_seed if task_random_seed is None else task_random_seed
    )
    effective_params = (
        dict(task_params_override) if task_params_override is not None else item.params
    )
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
    function_id: str = "",
    function_arguments: dict[str, Any] | None = None,
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
    resolved_function_id = str(function_id or "").strip()
    if resolved_function_id and resolved_agent != "omniflow":
        raise ValueError("direct_function_requires_omniflow_agent")
    if function_arguments is not None and not isinstance(function_arguments, dict):
        raise ValueError("direct_function_arguments_must_be_object")
    resolved_output = _experiment_run_dir(
        output_root,
        task=item.task,
        method=resolved_method,
        device=resolved_device,
        serial=serial,
        console_port=console_port,
        repo_root=repo_root,
    )
    if str(run_dir_suffix or "").strip():
        resolved_output = resolved_output / _safe_relative_path(
            run_dir_suffix,
            fallback="run",
        )
    if resolved_agent == "omniflow" and not str(store_path or "").strip():
        raise ValueError("omniflow_function_store_required")
    resolved_store_path = (
        resolve_path(store_path, root=repo_root)
        if store_path
        else None
    )
    resolved_task_seed = int(
        item.replay_seed if task_random_seed is None else task_random_seed
    )
    if resolved_agent == "omniflow" and not resolved_function_id:
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
        argv.extend(["--store-path", str(resolved_store_path)])
        if resolved_function_id:
            argv.extend(
                [
                    "--function-id",
                    resolved_function_id,
                    "--function-arguments-json",
                    json.dumps(
                        dict(function_arguments or {}),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                ]
            )
        elif planner_provider.strip():
            argv.extend(["--planner-provider", planner_provider.strip()])
        if not resolved_function_id and model.strip():
            argv.extend(["--model", model.strip()])
        if (
            not resolved_function_id
            and planner_timeout_sec is not None
            and float(planner_timeout_sec) > 0
        ):
            argv.extend(["--planner-timeout-sec", str(float(planner_timeout_sec))])
    if adb_path.strip():
        argv.extend(["--adb-path", adb_path.strip()])
    execution_mode = (
        "direct_function_e2e"
        if resolved_function_id
        else (
            "normal_omniflow_e2e"
            if resolved_agent == "omniflow"
            else "normal_androidworld_episode"
        )
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
            "store_path": str(resolved_store_path),
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
            "function_id": resolved_function_id,
            "function_arguments": (
                dict(function_arguments or {}) if resolved_function_id else None
            ),
            "state_backend": "androidworld",
            "action_backend": "androidworld",
            "native_androidworld_agent_io": True,
            "include_indexed_context": False,
            "uses_action_transfer": True,
        },
    )


def validate_omniflow_transfer_assets(
    store_path: str | Path,
    *,
    require_action_transfer: bool = True,
) -> dict[str, Any]:
    from omniflow.functions.assets import FunctionStore
    from omniflow.transfer.runtime import (
        TRANSFER_STATE_CATALOG_FILENAME,
        audit_transfer_action_sources,
        load_transfer_state_catalog,
        transfer_state_coverage,
    )

    resolved_store_path = resolve_path(store_path)
    if not resolved_store_path.is_file():
        raise FileNotFoundError(
            f"validated v2 Function Store not found: {resolved_store_path}"
        )
    store = FunctionStore(resolved_store_path)
    if store.load_errors:
        raise ValueError(
            "validated v2 Function Store has invalid Functions:"
            + ",".join(sorted(store.load_errors))
        )
    if not store.functions:
        raise ValueError("validated v2 Function Store contains no Functions")
    catalog_path = resolved_store_path.parent / TRANSFER_STATE_CATALOG_FILENAME
    states = load_transfer_state_catalog(catalog_path)
    coverage = transfer_state_coverage(store.functions, states)
    if require_action_transfer and coverage["required_state_count"]:
        if not catalog_path.is_file():
            raise FileNotFoundError(f"transfer_state_catalog_missing:{catalog_path}")
        if not coverage["complete"]:
            missing = ",".join(coverage["missing_state_ids"])
            raise ValueError(f"transfer_state_catalog_incomplete:{missing}")
        try:
            source_target_audit = audit_transfer_action_sources(
                store.functions,
                states,
            )
        except ValueError as error:
            reason = str(error)
            if not reason.startswith(
                (
                    "transfer_action_source_target_unresolved:",
                    "transfer_action_source_state_not_raw:",
                )
            ):
                raise
            source_target_audit = {
                "source_target_audit_complete": False,
                "source_target_count": 0,
                "source_targets": [],
                "fallback_required": True,
                "failure": reason,
            }
    else:
        source_target_audit = {
            "source_target_audit_complete": not require_action_transfer,
            "source_target_count": 0,
            "source_targets": [],
        }
    return {
        "store_path": str(resolved_store_path),
        "transfer_state_catalog": str(catalog_path) if catalog_path.is_file() else "",
        **coverage,
        **source_target_audit,
    }


def _command_line(spec: CommandSpec) -> str:
    env_prefix = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in sorted(spec.env.items())
    )
    argv = shlex.join(spec.argv)
    return f"env {env_prefix} {argv}" if env_prefix else argv


def parse_device_targets(raw_targets: str) -> list[DeviceTarget]:
    """Parse LABEL:SERIAL:PORT entries for dual-device E2E runs."""

    targets: list[DeviceTarget] = []
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



def seal_mobilegpt_source_memory(
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
    source_method = MOBILEGPT_SOURCE_METHOD
    learning_mode = MOBILEGPT_LEARNING_MODE

    if int(source_seed) != SOURCE_SEED:
        raise ValueError("mobilegpt_virtual_memory_requires_source_seed_111")
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
    if (
        transition_count <= 0
        or validated_count != transition_count
        or not isinstance(validation_rows, list)
        or not validation_rows
        or any(not isinstance(row, dict) or row.get("matched") is not True for row in validation_rows)
        or sum(int(row.get("consumed_transitions") or 0) for row in validation_rows)
        != transition_count
        or audit.get("actions_supplied_to_mobilegpt") is not True
        or audit.get("source_transitions_supplied") is not True
        or audit.get("source_success_boundary_supplied") is not True
        or audit.get("complete") is not True
    ):
        raise ValueError("mobilegpt_virtual_memory_trajectory_incomplete")
    official_reader = audit.get("official_reader_validation")
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
    if inventory.get("virtual_source_memory_complete") is not True:
        raise ValueError("mobilegpt_virtual_memory_graph_incomplete")
    if not inventory.get("has_recallable_subtasks"):
        raise ValueError("mobilegpt_virtual_memory_missing_recallable_subtasks")
    if not inventory.get("has_useful_actions"):
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
    if int(stats_summary.get("chat_model_calls") or 0) != 0:
        raise ValueError("mobilegpt_direct_memory_chat_calls_forbidden")
    required_audit = {
        "conversion_mode": "runlog_direct",
        "original_mobilegpt_prompts": False,
        "explore_agent_used": False,
        "select_agent_used": False,
        "derive_agent_fallback_allowed": True,
        "generalize_action_used": True,
        "source_reader_coverage_validation": True,
        "direct_subtasks_from_runlog": True,
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
    source_example_fallback_count = audit.get("source_example_fallback_count")
    if (
        type(source_example_fallback_count) is not int
        or source_example_fallback_count < 0
        or int(official_reader.get("source_reader_coverage_count") or 0)
        != transition_count
        or int(official_reader.get("source_example_fallback_count") or 0)
        != source_example_fallback_count
    ):
        raise ValueError(
            "mobilegpt_memory_audit_invalid:source_reader_coverage"
        )

    provenance_root = bundle_root / "provenance_mobilegpt_runlog_direct_v1"
    provenance_root.mkdir(exist_ok=False)

    def copy_evidence(source: Path, name: str) -> Path:
        destination = provenance_root / name
        shutil.copy2(source, destination)
        return destination

    copied_source = copy_evidence(source_path, "source.run_log.json")
    copied_stats = copy_evidence(stats_path, "mobilegpt_stats.jsonl")
    copied_audit = copy_evidence(audit_path, "trajectory_audit.json")
    memory_sha256, memory_file_count = mobilegpt_memory.mobilegpt_memory_digest(memory)
    manifest = {
        "schema_version": memory_schema,
        "task_name": str(task_name),
        "source_seed": int(source_seed),
        "source_method": source_method,
        "source_model": "",
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
        "provenance": {
            "native_mobilegpt_learning": False,
            "task_local_memory": True,
            "learning_mode": learning_mode,
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
            "function_store_used": False,
            "function_conversion_enabled": False,
            "target_inputs_read": False,
            "target_observations_read": False,
            "validator_state_read": False,
            "coordinate_replay": False,
            "source_emulator_used": False,
        },
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
        expected_model="",
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
    log_handle = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
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
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # The child keeps its own fd; closing the parent handle avoids leaks.
    log_handle.close()
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
    serial: str = "",
    adb_path: str = "",
    server_host: str = "0.0.0.0",
    port: int = 12345,
    stats_jsonl: str | Path = DEFAULT_MOBILEGPT_STATS_JSONL,
    target_package: str = "",
    target_app: str = "",
    runtime_observe_backend: str = "androidworld",
    python_executable: str = sys.executable,
    repo_root: Path = REPO_ROOT,
) -> CommandSpec:
    root = resolve_path(mobilegpt_root, root=repo_root)
    env: dict[str, str] = {}
    if serial.strip():
        env["ANDROID_SERIAL"] = serial.strip()
    env["MOBILEGPT_RUNTIME_OBSERVE_BACKEND"] = str(
        runtime_observe_backend or "androidworld"
    ).strip()
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

    resolved_action = str(action or "").strip().lower()
    if resolved_action == "server":
        env["MOBILEGPT_STATS_JSONL"] = str(resolve_path(stats_jsonl, root=repo_root))
        env["MOBILEGPT_UPSTREAM_MODE"] = "1"
        argv = [
            python_executable,
            "-m",
            "src.integrations.mobilegpt_runtime",
            "--mobilegpt-root",
            str(root),
            "--host",
            str(server_host or "0.0.0.0"),
            "--port",
            str(int(port)),
            "--upstream",
        ]
        return CommandSpec(
            label="mobilegpt:server",
            argv=argv,
            env=env,
            cwd=repo_root,
            output_path=None,
            metadata={
                "mobilegpt_root": str(root),
                "mobilegpt_memory_root": str(resolved_memory_root or ""),
                "port": int(port),
                "target_package": str(target_package or "").strip(),
                "target_app": str(target_app or "").strip(),
                "state_backend": "androidworld",
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


def _iter_jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            yield decoded


def _is_metrics_result_file(path: Path) -> bool:
    return path.name in {"task_results.jsonl", "all_latest.jsonl"}


def discover_task_result_files(paths: Sequence[str | Path]) -> list[Path]:
    seen: set[Path] = set()
    files: list[Path] = []
    for raw_path in paths:
        path = resolve_path(raw_path)
        candidates = (
            [path]
            if path.is_file()
            else sorted(
                candidate
                for candidate in path.rglob("*.jsonl")
                if _is_metrics_result_file(candidate)
            )
        )
        for candidate in candidates:
            if "_memory" in candidate.parts:
                continue
            if "_init_audit" in candidate.parts or "init_audit" in candidate.parts:
                continue
            if not _is_metrics_result_file(candidate):
                continue
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(resolved)
    return files


def _validator_success(row: dict[str, Any]) -> bool:
    validator = row.get("androidworld_validator_result")
    if isinstance(validator, dict) and "success" in validator:
        return bool(validator.get("success"))
    return bool(row.get("success"))


def _official_validator_used(row: dict[str, Any]) -> bool:
    if "official_validator_used" in row:
        return bool(row.get("official_validator_used"))
    if "uses_androidworld_official_validator" in row:
        return bool(row.get("uses_androidworld_official_validator"))
    if "validator" in row:
        return row.get("validator") == "androidworld_official"
    validator = row.get("androidworld_validator_result")
    if not isinstance(validator, dict):
        return False
    if "uses_androidworld_official_validator" in validator:
        return bool(validator.get("uses_androidworld_official_validator"))
    if "validator" in validator:
        return validator.get("validator") == "androidworld_official"
    return False


def _official_validator_success(row: dict[str, Any]) -> bool:
    return _official_validator_used(row) and _validator_success(row)


def _task_result_path_context(path: Path) -> dict[str, str]:
    if path.name != "task_results.jsonl":
        return {}
    known_methods = {
        "fixed_replay",
        "omniflow",
        "mobilegpt",
        "t3a_hint",
        "appagent",
    }
    stage = ""
    run_dir = path.parent
    device_dir = run_dir
    if run_dir.name in {"androidworld_init", "androidworld_validate"}:
        stage = run_dir.name
        device_dir = run_dir.parent
    method_dir = device_dir.parent
    task_dir = method_dir.parent
    if method_dir.name not in known_methods:
        return {}
    context = {
        "task_name": task_dir.name,
        "method": method_dir.name,
        "device": device_dir.name,
        "run_dir": str(run_dir),
    }
    if stage:
        context["stage"] = stage
    return context


def _compact_relocation_diagnostic(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        key: value.get(key)
        for key in (
            "schema_version",
            "dir",
            "manifest_path",
            "source_xml_path",
            "target_xml_path",
            "target_mapping_xml_path",
            "source_screenshot_path",
            "target_screenshot_path",
            "source_screenshot_available",
            "target_screenshot_available",
            "failure_reason",
            "error",
        )
        if value.get(key) not in (None, "")
    }


def _extract_relocation_diagnostics(
    value: Any, *, limit: int = 20
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(candidate: Any) -> None:
        compact = _compact_relocation_diagnostic(candidate)
        if not compact:
            return
        identity = str(
            compact.get("manifest_path")
            or compact.get("dir")
            or compact.get("target_xml_path")
            or compact
        )
        if identity in seen:
            return
        seen.add(identity)
        diagnostics.append(compact)

    def _walk(item: Any, depth: int = 0) -> None:
        if len(diagnostics) >= limit or depth > 8:
            return
        if isinstance(item, dict):
            if "relocation_diagnostics" in item and isinstance(
                item.get("relocation_diagnostics"),
                list,
            ):
                for diagnostic in item.get("relocation_diagnostics") or []:
                    _add(diagnostic)
            if "relocation_diagnostic" in item:
                _add(item.get("relocation_diagnostic"))
            if str(item.get("schema_version") or "").startswith(
                "omniflow.relocation_failure."
            ):
                _add(item)
            for nested in item.values():
                _walk(nested, depth + 1)
        elif isinstance(item, list):
            for nested in item:
                _walk(nested, depth + 1)

    _walk(value)
    return diagnostics


def _canonical_run_has_replay_material(canonical: dict[str, Any]) -> bool:
    if "replay_completed" in canonical:
        return True
    for step in canonical.get("steps") or []:
        if not isinstance(step, dict):
            continue
        provider_detail = step.get("provider_detail")
        if not isinstance(provider_detail, dict):
            continue
        if isinstance(provider_detail.get("run_function_result"), dict):
            return True
    return False


def _canonical_replay_completed(row: dict[str, Any]) -> bool | None:
    canonical = row.get("canonical_run")
    if not isinstance(canonical, dict) or not _canonical_run_has_replay_material(
        canonical
    ):
        return None
    if "replay_completed" in canonical:
        return bool(canonical.get("replay_completed"))
    if "completed" in canonical:
        return bool(canonical.get("completed"))
    return None


def _extract_replay_step_stats(row: dict[str, Any]) -> tuple[int, int]:
    canonical = row.get("canonical_run")
    if not isinstance(canonical, dict):
        return (0, 0)

    completed = 0
    total = 0
    has_run_function_result = False
    for step in canonical.get("steps") or []:
        if not isinstance(step, dict):
            continue
        provider_detail = step.get("provider_detail")
        if not isinstance(provider_detail, dict):
            continue
        run_result = provider_detail.get("run_function_result")
        if not isinstance(run_result, dict):
            continue
        has_run_function_result = True
        replay = run_result.get("replay")
        if not isinstance(replay, dict):
            continue
        step_total = _coerce_int(
            replay.get("active_step_count")
            or replay.get("step_count")
            or replay.get("actions_executed"),
            0,
        )
        step_completed = _coerce_int(
            replay.get("completed_step_count"),
            0,
        )
        if step_total <= 0:
            continue
        total += step_total
        completed += min(step_completed, step_total)

    if total > 0:
        return completed, total
    fallback_total = _coerce_int(
        canonical.get("actions_executed") or row.get("actions_executed"),
        0,
    )
    if not has_run_function_result:
        return (0, 0)
    replay_completed = _canonical_replay_completed(row)
    return (fallback_total if replay_completed else 0, fallback_total)


def aggregate_task_results(paths: Sequence[str | Path]) -> dict[str, Any]:
    task_result_files = discover_task_result_files(paths)
    rows: list[tuple[Path, dict[str, Any]]] = []
    for file_path in task_result_files:
        rows.extend((file_path, row) for row in _iter_jsonl_rows(file_path))

    official_validator_task_count = sum(
        1 for _, row in rows if _official_validator_used(row)
    )
    official_validator_success_count = sum(
        1 for _, row in rows if _official_validator_success(row)
    )
    replay_states = [_canonical_replay_completed(row) for _, row in rows]
    replay_task_count = sum(1 for value in replay_states if value is not None)
    replay_completed_count = sum(1 for value in replay_states if value is True)
    duration_ms = sum(_coerce_float(row.get("duration_ms")) for _, row in rows)
    actions_executed = 0
    model_calls = sum(_coerce_int(row.get("model_calls")) for _, row in rows)
    fallback_steps = sum(_coerce_int(row.get("fallback_steps")) for _, row in rows)
    total_tokens = sum(_coerce_int(row.get("total_tokens")) for _, row in rows)
    replay_step_completed = 0
    replay_step_total = 0
    relocation_diagnostic_count = 0
    per_task: list[dict[str, Any]] = []

    for file_path, row in rows:
        path_context = _task_result_path_context(file_path)
        task_name = (
            row.get("task_name") or row.get("task") or path_context.get("task_name")
        )
        method = row.get("method") or path_context.get("method")
        device = row.get("device") or path_context.get("device")
        row_actions_executed = _coerce_int(row.get("actions_executed"))
        if row_actions_executed <= 0 and (
            str(row.get("agent") or "").startswith("official:")
            or str(method or "") == "t3a_hint"
        ):
            row_actions_executed = max(
                row_actions_executed,
                _coerce_int(row.get("step_count")),
            )
        actions_executed += row_actions_executed
        step_completed, step_total = _extract_replay_step_stats(row)
        replay_step_completed += step_completed
        replay_step_total += step_total
        relocation_diagnostics = _extract_relocation_diagnostics(row)
        relocation_diagnostic_count += len(relocation_diagnostics)
        official_validator_used = _official_validator_used(row)
        official_validator_success = (
            _official_validator_success(row) if official_validator_used else None
        )
        per_task.append(
            {
                "task_name": task_name,
                "task": row.get("task"),
                "goal": row.get("goal"),
                "task_params": row.get("task_params"),
                "task_params_sha256": row.get("task_params_sha256"),
                "agent": row.get("agent"),
                "backend": row.get("backend"),
                "method": method,
                "device": device,
                "run_dir": row.get("run_dir") or path_context.get("run_dir"),
                "stage": row.get("stage") or path_context.get("stage"),
                "result_file": str(file_path),
                "replay_track": row.get("replay_track"),
                "official_validator_used": official_validator_used,
                "official_validator_success": official_validator_success,
                "replay_completed": _canonical_replay_completed(row),
                "duration_ms": round(_coerce_float(row.get("duration_ms")), 3),
                "duration_sec": round(
                    _coerce_float(row.get("duration_ms")) / 1000.0,
                    3,
                ),
                "actions_executed": row_actions_executed,
                "replay_step_completed_count": step_completed,
                "replay_step_total": step_total,
                "replay_step_completed_rate": _rate(step_completed, step_total),
                "model_calls": _coerce_int(row.get("model_calls")),
                "fallback_steps": _coerce_int(row.get("fallback_steps")),
                "prompt_tokens": _coerce_int(row.get("prompt_tokens")),
                "completion_tokens": _coerce_int(row.get("completion_tokens")),
                "total_tokens": _coerce_int(row.get("total_tokens")),
                "token_usage_status": row.get("token_usage_status"),
                "model": row.get("model"),
                "model_base_url": row.get("model_base_url"),
                "artifact_kind": row.get("artifact_kind"),
                "artifact_ref": row.get("artifact_ref"),
                "target_run_log_path": row.get("target_run_log_path"),
                "target_run_log_sha256": row.get("target_run_log_sha256"),
                "target_transfer_states_path": row.get("target_transfer_states_path"),
                "target_transfer_states_sha256": row.get(
                    "target_transfer_states_sha256"
                ),
                "target_transfer_state_audit": row.get("target_transfer_state_audit"),
                "relocation_diagnostic_count": len(relocation_diagnostics),
                "relocation_diagnostics": relocation_diagnostics,
                "error": row.get("error"),
                "runtime_integrity_error": row.get("runtime_integrity_error"),
                "environment_failure": row.get("environment_failure"),
            }
        )

    task_count = len(rows)
    return {
        "schema_version": "omniflow.androidworld_replay_pipeline_summary.v4",
        "task_count": task_count,
        "task_results_files": [str(path) for path in task_result_files],
        "official_validator_task_count": official_validator_task_count,
        "official_validator_success_count": official_validator_success_count,
        "official_validator_success_rate": _rate(
            official_validator_success_count,
            official_validator_task_count,
        ),
        "official_validator_coverage_rate": _rate(
            official_validator_task_count,
            task_count,
        ),
        "replay_task_count": replay_task_count,
        "replay_completed_count": replay_completed_count,
        "replay_completed_rate": _rate(replay_completed_count, replay_task_count),
        "replay_coverage_rate": _rate(replay_task_count, task_count),
        "duration_ms": round(duration_ms, 3),
        "duration_sec": round(duration_ms / 1000.0, 3),
        "avg_duration_ms": round(duration_ms / max(1, task_count), 3),
        "avg_duration_sec": round((duration_ms / max(1, task_count)) / 1000.0, 3),
        "actions_executed": actions_executed,
        "avg_actions_per_task": round(actions_executed / max(1, task_count), 3),
        "avg_ms_per_action": round(duration_ms / max(1, actions_executed), 3),
        "replay_step_completed_count": replay_step_completed,
        "replay_step_total": replay_step_total,
        "replay_step_completed_rate": _rate(
            replay_step_completed,
            replay_step_total,
        ),
        "model_calls": model_calls,
        "total_tokens": total_tokens,
        "fallback_steps": fallback_steps,
        "relocation_diagnostic_count": relocation_diagnostic_count,
        "per_task": per_task,
    }


def write_metrics_summary(summary: dict[str, Any], output_path: str | Path) -> None:
    path = resolve_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md_lines = [
        "# AndroidWorld Replay Pipeline Summary",
        "",
        f"- task_count: `{summary['task_count']}`",
        (
            "- validator: `"
            f"{summary['official_validator_success_count']}/"
            f"{summary['official_validator_task_count']}`"
        ),
        f"- replay_completed: `{summary['replay_completed_count']}/{summary['replay_task_count']}`",
        f"- replay_coverage: `{summary['replay_task_count']}/{summary['task_count']}`",
        f"- replay_step_completed: `{summary['replay_step_completed_count']}/{summary['replay_step_total']}`",
        f"- actions_executed: `{summary['actions_executed']}`",
        f"- relocation_diagnostics: `{summary.get('relocation_diagnostic_count', 0)}`",
        f"- duration_s: `{round(_coerce_float(summary['duration_ms']) / 1000.0, 3)}`",
        f"- model_calls: `{summary.get('model_calls', 0)}`",
        f"- total_tokens: `{summary.get('total_tokens', 0)}`",
    ]
    md_lines.extend(
        [
            "",
            "| task | validator | replay_completed | actions | model_calls | total_tokens | step_completed | relocation | sec | error |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in summary.get("per_task") or []:
        md_lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("task_name") or item.get("task") or ""),
                    "1" if item.get("official_validator_success") else "0",
                    ""
                    if item.get("replay_completed") is None
                    else "1"
                    if item.get("replay_completed")
                    else "0",
                    str(item.get("actions_executed") or 0),
                    str(item.get("model_calls") or 0),
                    str(item.get("total_tokens") or 0),
                    f"{item.get('replay_step_completed_count') or 0}/{item.get('replay_step_total') or 0}",
                    str(item.get("relocation_diagnostic_count") or 0),
                    str(round(_coerce_float(item.get("duration_ms")) / 1000.0, 3)),
                    str(item.get("error") or ""),
                ]
            )
            + " |"
        )
    path.with_suffix(".md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def _add_androidworld_setup_args(parser: argparse.ArgumentParser) -> None:
    perform_setup_default = (
        str(os.environ.get("OMNIFLOW_ANDROIDWORLD_PERFORM_EMULATOR_SETUP", "1"))
        .strip()
        .lower()
        not in {"0", "false", "no", "off"}
    )
    parser.add_argument(
        "--perform-emulator-setup",
        dest="perform_emulator_setup",
        action="store_true",
        default=perform_setup_default,
        help=(
            "Run AndroidWorld app setup before the task suite so each task "
            "initialize_task(env) can restore a fresh snapshot. Default: on."
        ),
    )
    parser.add_argument(
        "--no-perform-emulator-setup",
        dest="perform_emulator_setup",
        action="store_false",
        help="Skip AndroidWorld app setup; use only for targeted debugging.",
    )


def _select_from_args(args: argparse.Namespace) -> list[CanonicalRunLog]:
    source_items = load_canonical_source_index(args.index)
    by_name = {item.task: item for item in source_items}
    selected: list[CanonicalRunLog] = []
    missing: list[str] = []
    for raw_name in str(args.task or "").split(","):
        name = raw_name.strip()
        if not name:
            continue
        item = by_name.get(name)
        if item is None:
            missing.append(name)
        else:
            selected.append(item)
    if missing:
        raise KeyError(f"Tasks not found in canonical data index: {', '.join(missing)}")
    return selected




_RESULT_NON_EXECUTED_STATUSES = {
    "INVALID_MEMORY_LEAKAGE",
    "env_failed",
    "init_failed",
    "setup_failed",
}


def _is_mobilegpt_method(method: str) -> bool:
    return str(method or "").strip() == "mobilegpt"


def _result_record_has_formal_result(record: dict[str, Any]) -> bool:
    if bool(record.get("summary_exclude")):
        return False
    if str(record.get("status") or "").strip() in _RESULT_NON_EXECUTED_STATUSES:
        return False
    return bool(str(record.get("output_path") or "").strip())


def _formal_result_paths(record: dict[str, Any]) -> list[Path]:
    if not _result_record_has_formal_result(record):
        return []
    output_path = resolve_path(str(record.get("output_path") or ""))
    if output_path.is_dir():
        result_files = sorted(output_path.rglob("task_results.jsonl"))
        if result_files:
            return result_files
    return [output_path]


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


def _t3a_hint_action_identity(step: Any) -> str:
    name, params = _t3a_hint_step_action(step)
    action = name.strip().lower()
    key = str(params.get("key") or params.get("key_name") or "").strip().lower()
    if action in {"press_key", "key_event", "presskey"}:
        action = {
            "back": "navigate_back",
            "home": "navigate_home",
            "enter": "keyboard_enter",
        }.get(key, action)
    action = {
        "back": "navigate_back",
        "press_back": "navigate_back",
        "home": "navigate_home",
        "press_home": "navigate_home",
        "enter": "keyboard_enter",
    }.get(action, action)
    return action


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
    _, params = _t3a_hint_step_action(step)
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
    if not candidates and action_name in {
        "input_text",
        "type_text",
        "set_text",
        "enter_text",
    } and not (
        (isinstance(x, (int, float)) and isinstance(y, (int, float)))
        or indexed_id
    ):
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


def _select_complete_function(store_path: str | Path):
    store = FunctionStore(resolve_path(store_path))
    if store.load_errors:
        raise ValueError(
            "t3a_hint_function_store_invalid:" + ",".join(sorted(store.load_errors))
        )
    functions = [
        function
        for function in store.list_functions(limit=500)
        if re.match(r"^action_\d{3}_", function.id) is None
    ]
    if not functions:
        raise ValueError("t3a_hint_semantic_functions_required")
    if len(functions) == 1:
        return functions[0]

    def action_tools(function: Any) -> tuple[str, ...]:
        return tuple(action.tool for action in function.actions)

    def contains(complete: tuple[str, ...], subsequence: tuple[str, ...]) -> bool:
        if not subsequence or len(subsequence) > len(complete):
            return False
        return any(
            complete[start : start + len(subsequence)] == subsequence
            for start in range(len(complete) - len(subsequence) + 1)
        )

    candidates = [
        candidate
        for candidate in functions
        if all(
            other is candidate or contains(action_tools(candidate), action_tools(other))
            for other in functions
        )
    ]
    maximum = max((len(action_tools(item)) for item in candidates), default=0)
    candidates = [
        candidate for candidate in candidates if len(action_tools(candidate)) == maximum
    ]
    if len(candidates) != 1:
        raise ValueError("t3a_hint_complete_function_ambiguous")
    return candidates[0]


def _source_action_hint_path_for_item(
    item: CanonicalRunLog,
    *,
    output_root: str | Path,
    store_path: str | Path | None = None,
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
    if store_path is not None:
        resolved_store_path = resolve_path(store_path, root=repo_root)
        task_function = _select_complete_function(resolved_store_path)
        raw_store = _read_json(resolved_store_path)
        raw_functions = raw_store.get("functions")
        raw_function = (
            raw_functions.get(task_function.id)
            if isinstance(raw_functions, dict)
            else None
        )
        raw_function_steps = (
            raw_function.get("steps") if isinstance(raw_function, dict) else None
        )
        if not isinstance(raw_function_steps, list):
            raise ValueError(
                f"t3a_hint_complete_function_raw_steps_missing:{task_function.id}"
            )
        semantic_input_steps = []
        semantic_preceding_steps = []
        alignment_modes: list[str] = []
        source_cursor = 0
        for function_step in raw_function_steps:
            function_action, _ = _t3a_hint_step_action(function_step)
            function_action_identity = _t3a_hint_action_identity(function_step)
            function_state_id = str(
                function_step.get("source_state_id")
                if isinstance(function_step, dict)
                else ""
            ).strip()
            aligned_index = None
            alignment_mode = "state_identity"
            for source_index in range(source_cursor, len(source_steps)):
                source_step = source_steps[source_index]
                source_state_id = str(
                    source_step.get("before_state_id")
                    if isinstance(source_step, dict)
                    else ""
                ).strip()
                if (
                    _t3a_hint_action_identity(source_step) == function_action_identity
                    and source_state_id == function_state_id
                ):
                    aligned_index = source_index
                    break
            if aligned_index is None:
                alignment_mode = "ordered_action"
                for source_index in range(source_cursor, len(source_steps)):
                    if (
                        _t3a_hint_action_identity(source_steps[source_index])
                        == function_action_identity
                    ):
                        aligned_index = source_index
                        break
            if aligned_index is None:
                raise ValueError(
                    "t3a_hint_function_runlog_action_mismatch:"
                    f"{function_state_id}:{function_action}"
                )
            semantic_input_steps.append(source_steps[aligned_index])
            semantic_preceding_steps.append(
                source_steps[aligned_index - 1] if aligned_index > 0 else None
            )
            alignment_modes.append(alignment_mode)
            source_cursor = aligned_index + 1
        semantic_source = "omniflow_function_store"
        store_alignment_mode = (
            "state_identity"
            if all(mode == "state_identity" for mode in alignment_modes)
            else "ordered_action"
        )
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
    "com.android.systemui",
    "com.example.MobileGPT",
    "com.google.android.apps.nexuslauncher",
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
    if package_name in _MOBILEGPT_IGNORED_TARGET_PACKAGES:
        return ""
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


def _start_background_command(
    spec: CommandSpec,
    *,
    dry_run: bool = False,
    warmup_sec: float = 0.0,
) -> tuple[subprocess.Popen[Any] | None, int]:
    print(f"[{spec.label}:background] {_command_line(spec)}", flush=True)
    if dry_run:
        return None, 0
    process = subprocess.Popen(
        spec.argv,
        cwd=spec.cwd,
        env=_subprocess_env(spec.env),
    )
    if warmup_sec > 0:
        time.sleep(float(warmup_sec))
    returncode = process.poll()
    if returncode is not None:
        return process, int(returncode)
    return process, 0


def _stop_background_command(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _normalize_result_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)

    episode_model_calls = _coerce_int(
        normalized.get("episode_model_calls") or normalized.get("warm_model_calls")
    )
    episode_prompt_tokens = _coerce_int(
        normalized.get("episode_prompt_tokens") or normalized.get("warm_prompt_tokens")
    )
    episode_completion_tokens = _coerce_int(
        normalized.get("episode_completion_tokens")
        or normalized.get("warm_completion_tokens")
    )
    episode_total_tokens = _coerce_int(
        normalized.get("episode_total_tokens") or normalized.get("warm_total_tokens")
    )
    shared_model_calls = _coerce_int(normalized.get("shared_model_calls"))
    shared_prompt_tokens = _coerce_int(normalized.get("shared_prompt_tokens"))
    shared_completion_tokens = _coerce_int(normalized.get("shared_completion_tokens"))
    shared_total_tokens = _coerce_int(
        normalized.get("shared_total_tokens") or normalized.get("shared_tokens")
    )

    if episode_model_calls > 0:
        normalized["model_calls"] = episode_model_calls
        normalized["model_calls_source"] = "mobilegpt_episode_stats"
    if episode_prompt_tokens > 0:
        normalized["prompt_tokens"] = episode_prompt_tokens
    if episode_completion_tokens > 0:
        normalized["completion_tokens"] = episode_completion_tokens
    if episode_total_tokens > 0:
        normalized["total_tokens"] = episode_total_tokens
        normalized["total_tokens_source"] = "mobilegpt_episode_stats"

    if _coerce_int(normalized.get("model_calls")) <= 0 and shared_model_calls > 0:
        normalized["model_calls"] = shared_model_calls
        normalized["model_calls_source"] = "mobilegpt_stats"
    if _coerce_int(normalized.get("prompt_tokens")) <= 0 and shared_prompt_tokens > 0:
        normalized["prompt_tokens"] = shared_prompt_tokens
    if (
        _coerce_int(normalized.get("completion_tokens")) <= 0
        and shared_completion_tokens > 0
    ):
        normalized["completion_tokens"] = shared_completion_tokens

    total_tokens = _coerce_int(
        normalized.get("total_tokens") or normalized.get("tokens")
    )
    if total_tokens <= 0 and shared_total_tokens > 0:
        total_tokens = shared_total_tokens
        normalized["total_tokens_source"] = "mobilegpt_stats"
    if total_tokens <= 0:
        total_tokens = _coerce_int(normalized.get("prompt_tokens")) + _coerce_int(
            normalized.get("completion_tokens")
        )
    normalized["total_tokens"] = total_tokens
    normalized["model_calls"] = _coerce_int(normalized.get("model_calls"))
    if "episode_actions_executed" in normalized:
        normalized["actions_executed"] = _coerce_int(
            normalized.get("episode_actions_executed")
        )
        normalized["actions_executed_source"] = "mobilegpt_episode_stats"
    if "episode_fallback_count" in normalized:
        normalized["fallback_steps"] = _coerce_int(
            normalized.get("episode_fallback_count")
        )
        normalized["fallback_steps_source"] = "mobilegpt_episode_stats"
    if normalized.get("total_tokens_source") in {
        "mobilegpt_episode_stats",
        "mobilegpt_stats",
    }:
        normalized["token_usage_status"] = str(
            normalized.get("episode_token_usage_status")
            or (
                "tracked"
                if _coerce_int(normalized.get("model_calls")) > 0
                and total_tokens > 0
                and total_tokens
                == _coerce_int(normalized.get("prompt_tokens"))
                + _coerce_int(normalized.get("completion_tokens"))
                else "inconsistent"
            )
        )

    wall_sec = _coerce_float(normalized.get("wall_sec"))
    if wall_sec <= 0:
        wall_sec = _coerce_float(normalized.get("duration_sec"))
    if wall_sec > 0:
        normalized["wall_sec"] = round(wall_sec, 3)

    normalized["reuse_metrics"] = reuse_metrics_from_result_row(normalized)
    return normalized


def _result_row_value(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple)):
        return "; ".join(str(item) for item in value)
    return str(value)


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


def _promote_result_metadata_to_row(
    row: dict[str, Any],
    records: Sequence[dict[str, Any]],
) -> None:
    for record in records:
        metadata = (
            record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        )
        for key in _RESULT_METADATA_ROW_KEYS:
            if key in row and row.get(key) not in (None, "", {}, []):
                continue
            value = metadata.get(key)
            if value not in (None, "", {}, []):
                row[key] = value
    init_audit = row.get("init_audit")
    if isinstance(init_audit, dict):
        for source_key, row_key in (
            ("init_audit_enabled", "init_audit_enabled"),
            ("init_audit_status", "init_audit_status"),
            ("initialized", "initialized"),
            ("init_called", "init_called"),
            ("init_audit_returncode", "init_audit_returncode"),
            ("init_audit_init_summary", "init_audit_init_summary"),
        ):
            if row.get(row_key) in (None, "", {}, []):
                row[row_key] = init_audit.get(source_key)


def _record_wall_sec(record: dict[str, Any]) -> float:
    metadata = (
        record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    )
    value = metadata.get("wall_sec") if metadata else record.get("wall_sec")
    return _coerce_float(value)


def _result_summary_rows(
    *,
    task: str,
    command_records: Sequence[dict[str, Any]],
    aggregate_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_method_device: dict[tuple[str, str], dict[str, Any]] = {}
    mobilegpt_episode_stats: dict[tuple[str, str], dict[str, Any]] = {}
    mobilegpt_teacher_stats: dict[str, dict[str, Any]] = {}
    teacher_wait_by_method: dict[str, dict[str, Any]] = {}
    teacher_wall_sec_by_method: dict[str, float] = {}
    dry_run_summary = any(
        bool((record.get("metadata") or {}).get("dry_run"))
        for record in command_records
        if isinstance(record.get("metadata"), dict)
    )

    for record in command_records:
        metadata = (
            record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        )
        method = str(record.get("method") or metadata.get("method") or "").strip()
        device = str(record.get("device") or metadata.get("device") or "").strip()
        status = str(record.get("status") or "").strip()
        memory_root_value = str(metadata.get("memory_root") or "").strip()
        if (
            _is_mobilegpt_method(method)
            and status in {"teacher_memory_write", "cold_memory_write"}
            and isinstance(metadata.get("mobilegpt_wait_result"), dict)
        ):
            teacher_wait_by_method[method] = {
                "start": dict(metadata.get("mobilegpt_start_wait_result") or {}),
                "finish": dict(metadata.get("mobilegpt_wait_result") or {}),
            }
        if _is_mobilegpt_method(method) and status in {
            "teacher_memory_write",
            "cold_memory_write",
        }:
            teacher_wall_sec_by_method[method] = round(
                teacher_wall_sec_by_method.get(method, 0.0) + _record_wall_sec(record),
                3,
            )
        if not _is_mobilegpt_method(method) or not memory_root_value:
            continue
        if dry_run_summary:
            continue
        memory_root = resolve_path(memory_root_value)
        if status in {"teacher_memory_write", "cold_memory_write"}:
            teacher_stats = _load_mobilegpt_stats_summary(
                summary_path=memory_root / "mobilegpt_stats_summary.json",
                stats_jsonl_path=memory_root / "mobilegpt_stats.jsonl",
            )
            if teacher_stats:
                mobilegpt_teacher_stats[method] = teacher_stats
            continue
        explicit_episode_summary = str(
            metadata.get("mobilegpt_stats_summary") or ""
        ).strip()
        explicit_episode_jsonl = str(
            metadata.get("mobilegpt_stats_jsonl") or ""
        ).strip()
        if not device or not (explicit_episode_summary or explicit_episode_jsonl):
            continue
        episode_stats = _load_mobilegpt_stats_summary(
            summary_path=(
                resolve_path(explicit_episode_summary)
                if explicit_episode_summary
                else None
            ),
            stats_jsonl_path=(
                resolve_path(explicit_episode_jsonl) if explicit_episode_jsonl else None
            ),
        )
        if episode_stats:
            mobilegpt_episode_stats[(method, device)] = episode_stats

    def _row_rank(row: dict[str, Any]) -> int:
        stage = str(row.get("stage") or "").strip()
        if stage == "androidworld_validate":
            return 4
        if not stage and row.get("official_validator_used"):
            return 3
        if not stage:
            return 2
        if stage == "androidworld_init":
            return 1
        return 0

    for row in aggregate_summary.get("per_task") or []:
        if str(row.get("task_name") or row.get("task") or "") != task:
            continue
        method = str(row.get("method") or "").strip()
        device = str(row.get("device") or "").strip()
        if method and device:
            key = (method, device)
            existing = by_method_device.get(key)
            if existing is None or _row_rank(row) >= _row_rank(existing):
                by_method_device[key] = dict(row)

    grouped_records: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in command_records:
        if bool(record.get("summary_exclude")):
            continue
        method = str(record.get("method") or "").strip()
        device = str(record.get("device") or "").strip()
        if not method or not device:
            continue
        key = (method, device)
        grouped_records.setdefault(key, []).append(dict(record))

    seen: set[tuple[str, str]] = set()
    for key, records in grouped_records.items():
        method, device = key
        seen.add(key)
        explicit_statuses = [
            str(record.get("status") or "").strip()
            for record in records
            if str(record.get("status") or "").strip()
        ]
        has_non_executed_status = any(
            status in _RESULT_NON_EXECUTED_STATUSES for status in explicit_statuses
        )
        row = {} if has_non_executed_status else dict(by_method_device.get(key) or {})
        returncodes = [int(record.get("returncode") or 0) for record in records]
        wall_sec = sum(_record_wall_sec(record) for record in records)
        status = explicit_statuses[-1] if explicit_statuses else ""
        if not status:
            status = (
                "completed"
                if all(code == 0 for code in returncodes)
                else "command_failed"
            )
        errors = [
            str(record.get("error") or "").strip()
            for record in records
            if str(record.get("error") or "").strip()
        ]
        row.update(
            {
                "task_name": row.get("task_name") or task,
                "method": method,
                "device": device,
                "status": status,
                "returncode": max(returncodes or [0]),
                "command_count": len(records),
                "command": records[-1].get("command") or "",
                "commands": [record.get("command") or "" for record in records],
                "output_path": records[-1].get("output_path")
                or row.get("run_dir")
                or "",
                "error": row.get("error") or "; ".join(errors),
            }
        )
        _promote_result_metadata_to_row(row, records)
        if wall_sec > 0:
            row["wall_sec"] = round(wall_sec, 3)
        if _is_mobilegpt_method(method):
            row["episode_wall_sec"] = round(wall_sec, 3) if wall_sec > 0 else 0
            teacher_wall_sec = teacher_wall_sec_by_method.get(method, 0.0)
            if teacher_wall_sec > 0:
                row["teacher_wall_sec"] = round(teacher_wall_sec, 3)
        teacher_stats = mobilegpt_teacher_stats.get(method) or {}
        episode_stats = mobilegpt_episode_stats.get(key) or {}
        mobilegpt_prep: dict[str, Any] = {}
        for record in reversed(records):
            metadata = record.get("metadata")
            if not isinstance(metadata, dict):
                continue
            prep_candidate = metadata.get("mobilegpt_prep")
            if isinstance(prep_candidate, dict) and prep_candidate:
                mobilegpt_prep = dict(prep_candidate)
                break
        if mobilegpt_prep:
            prep_stats = dict(mobilegpt_prep.get("stats") or {})
            prep_fields = mobilegpt_memory.mobilegpt_stats_row_fields("prep", prep_stats)
            row.update(prep_fields)
            row.update(
                {
                    "prep_type": str(mobilegpt_prep.get("type") or ""),
                    "prep_duration_sec": prep_fields["prep_task_elapsed_sec"],
                    "prep_wall_sec": _coerce_float(mobilegpt_prep.get("wall_sec")),
                    "prep_official_validator_success": mobilegpt_prep.get(
                        "official_validator_success"
                    ),
                    "prep_manifest": str(mobilegpt_prep.get("manifest_path") or ""),
                    "prep_manifest_sha256": str(
                        mobilegpt_prep.get("manifest_sha256") or ""
                    ),
                    "prep_memory_sha256": str(
                        mobilegpt_prep.get("memory_sha256") or ""
                    ),
                    "prep_shared_across_targets": bool(
                        mobilegpt_prep.get("shared_across_targets")
                    ),
                }
            )
        appagent_prep: dict[str, Any] = {}
        for record in reversed(records):
            metadata = record.get("metadata")
            if not isinstance(metadata, dict):
                continue
            prep_candidate = metadata.get("appagent_prep")
            if isinstance(prep_candidate, dict) and prep_candidate:
                appagent_prep = dict(prep_candidate)
                break
        if appagent_prep:
            row.update(
                {
                    "prep_type": str(appagent_prep.get("type") or ""),
                    "prep_model_calls": _coerce_int(appagent_prep.get("model_calls")),
                    "prep_prompt_tokens": _coerce_int(
                        appagent_prep.get("prompt_tokens")
                    ),
                    "prep_completion_tokens": _coerce_int(
                        appagent_prep.get("completion_tokens")
                    ),
                    "prep_total_tokens": _coerce_int(appagent_prep.get("total_tokens")),
                    "prep_token_usage_status": str(
                        appagent_prep.get("token_usage_status") or ""
                    ),
                    "prep_duration_sec": _coerce_float(appagent_prep.get("wall_sec")),
                    "prep_wall_sec": _coerce_float(appagent_prep.get("wall_sec")),
                    "prep_source_episode_duration_sec": _coerce_float(
                        appagent_prep.get("source_episode_duration_sec")
                    ),
                    "prep_source_episode_wall_sec": _coerce_float(
                        appagent_prep.get("source_episode_wall_sec")
                    ),
                    "prep_document_generation_wall_sec": _coerce_float(
                        appagent_prep.get("document_generation_wall_sec")
                    ),
                    "prep_official_validator_success": appagent_prep.get(
                        "official_validator_success"
                    ),
                    "prep_manifest": str(appagent_prep.get("manifest_path") or ""),
                    "prep_manifest_sha256": str(
                        appagent_prep.get("manifest_sha256") or ""
                    ),
                    "prep_demo_sha256": str(appagent_prep.get("demo_sha256") or ""),
                    "prep_demo_docs_sha256": str(
                        appagent_prep.get("demo_docs_sha256") or ""
                    ),
                    "prep_shared_across_targets": bool(
                        appagent_prep.get("shared_across_targets")
                    ),
                }
            )
        if teacher_stats:
            teacher_fields = mobilegpt_memory.mobilegpt_stats_row_fields("teacher", teacher_stats)
            row.update(teacher_fields)
            row.update(
                {
                    "prep_type": row.get("prep_type")
                    or MOBILEGPT_PREP_TYPE,
                    "prep_model_calls": teacher_fields["teacher_model_calls"],
                    "prep_total_tokens": teacher_fields["teacher_total_tokens"],
                    "prep_prompt_tokens": teacher_fields["teacher_prompt_tokens"],
                    "prep_completion_tokens": teacher_fields[
                        "teacher_completion_tokens"
                    ],
                    "prep_duration_sec": teacher_fields["teacher_task_elapsed_sec"]
                    or row.get("teacher_wall_sec")
                    or 0,
                    "prep_stats_summary": teacher_fields["teacher_stats_summary"],
                    "prep_stats_jsonl": teacher_fields["teacher_stats_jsonl"],
                    "mobilegpt_teacher_action_count": _coerce_int(
                        teacher_stats.get("teacher_action_count")
                    ),
                    "mobilegpt_teacher_groundable_action_count": _coerce_int(
                        teacher_stats.get("teacher_groundable_action_count")
                    ),
                    "mobilegpt_teacher_miss_count": _coerce_int(
                        teacher_stats.get("teacher_miss_count")
                    ),
                    "mobilegpt_teacher_vlm_fallback_count": _coerce_int(
                        teacher_stats.get("teacher_vlm_fallback_count")
                    ),
                    "mobilegpt_teacher_unrecovered_miss_count": _coerce_int(
                        teacher_stats.get("teacher_unrecovered_miss_count")
                    ),
                    "mobilegpt_native_vlm_fallback_only": bool(
                        teacher_stats.get("native_vlm_fallback_only")
                    ),
                }
            )
        if episode_stats:
            episode_fields = mobilegpt_memory.mobilegpt_stats_row_fields("episode", episode_stats)
            row.update(
                {
                    **episode_fields,
                    "episode_stats_scope": "task_device",
                    "shared_model_calls": episode_fields["episode_model_calls"],
                    "chat_model_calls": episode_fields["episode_chat_model_calls"],
                    "embedding_model_calls": episode_fields[
                        "episode_embedding_model_calls"
                    ],
                    "chat_models": episode_fields["episode_chat_models"],
                    "embedding_models": episode_fields["episode_embedding_models"],
                    "shared_tokens": episode_fields["episode_total_tokens"],
                    "shared_prompt_tokens": episode_fields["episode_prompt_tokens"],
                    "shared_completion_tokens": episode_fields[
                        "episode_completion_tokens"
                    ],
                    "shared_chat_latency_sec": episode_fields[
                        "episode_chat_latency_sec"
                    ],
                    "shared_embedding_latency_sec": episode_fields[
                        "episode_embedding_latency_sec"
                    ],
                    "mobilegpt_task_started_count": episode_fields[
                        "episode_task_started_count"
                    ],
                    "mobilegpt_task_finished_count": episode_fields[
                        "episode_task_finished_count"
                    ],
                    "episode_task_elapsed_status": (
                        "complete"
                        if episode_fields["episode_task_finished_count"] > 0
                        else "not_emitted_before_androidworld_termination"
                    ),
                    "episode_duration_source": "androidworld_task_results",
                    "mobilegpt_stats_summary": episode_fields["episode_stats_summary"],
                }
            )
        teacher_wait = teacher_wait_by_method.get(method)
        if teacher_wait:
            teacher_start = teacher_wait.get("start") or {}
            teacher_finish = teacher_wait.get("finish") or {}
            row.update(
                {
                    "mobilegpt_teacher_started": teacher_start.get("seen"),
                    "mobilegpt_teacher_finished": teacher_finish.get("seen"),
                    "mobilegpt_teacher_wait_timeout_sec": teacher_finish.get(
                        "timeout_sec"
                    ),
                    "mobilegpt_teacher_wait_elapsed_sec": teacher_finish.get(
                        "elapsed_sec"
                    ),
                }
            )
        run_wait_records = [
            dict(record.get("metadata") or {})
            for record in records
            if isinstance(record.get("metadata"), dict)
            and isinstance(
                record.get("metadata", {}).get("mobilegpt_wait_result"),
                dict,
            )
        ]
        if run_wait_records:
            run_wait_metadata = run_wait_records[-1]
            run_start = dict(run_wait_metadata.get("mobilegpt_start_wait_result") or {})
            run_finish = dict(run_wait_metadata.get("mobilegpt_wait_result") or {})
            row.update(
                {
                    "mobilegpt_episode_started": run_start.get("seen"),
                    "mobilegpt_episode_finished": run_finish.get("seen"),
                    "mobilegpt_episode_wait_timeout_sec": run_finish.get("timeout_sec"),
                    "mobilegpt_episode_wait_elapsed_sec": run_finish.get("elapsed_sec"),
                }
            )
        rows.append(row)

    for key, row in sorted(by_method_device.items()):
        if key in seen:
            continue
        rows.append({**row, "status": "completed", "returncode": 0})
    return [_normalize_result_row(row) for row in rows]


def _aggregate_normalized_result_rows(
    aggregate_summary: dict[str, Any],
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Recompute the result aggregate from its final canonical rows."""
    aggregate = dict(aggregate_summary)
    for detailed_usage_field in (
        "model_calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "chat_model_calls",
        "embedding_model_calls",
    ):
        aggregate.pop(detailed_usage_field, None)
    canonical_rows = [dict(row) for row in rows]
    task_count = len(canonical_rows)
    official_rows = [
        row for row in canonical_rows if bool(row.get("official_validator_used"))
    ]
    official_success_count = sum(
        1 for row in official_rows if bool(row.get("official_validator_success"))
    )
    replay_rows = [
        row for row in canonical_rows if row.get("replay_completed") is not None
    ]
    replay_completed_count = sum(
        1 for row in replay_rows if row.get("replay_completed") is True
    )
    duration_ms = sum(
        _coerce_float(row.get("duration_ms"))
        or _coerce_float(row.get("duration_sec")) * 1000.0
        for row in canonical_rows
    )
    actions_executed = sum(
        _coerce_int(row.get("actions_executed")) for row in canonical_rows
    )
    replay_step_completed = sum(
        _coerce_int(row.get("replay_step_completed_count")) for row in canonical_rows
    )
    replay_step_total = sum(
        _coerce_int(row.get("replay_step_total")) for row in canonical_rows
    )

    aggregate.update(
        {
            "task_count": task_count,
            "official_validator_task_count": len(official_rows),
            "official_validator_success_count": official_success_count,
            "official_validator_success_rate": _rate(
                official_success_count,
                len(official_rows),
            ),
            "official_validator_coverage_rate": _rate(
                len(official_rows),
                task_count,
            ),
            "replay_task_count": len(replay_rows),
            "replay_completed_count": replay_completed_count,
            "replay_completed_rate": _rate(
                replay_completed_count,
                len(replay_rows),
            ),
            "replay_coverage_rate": _rate(len(replay_rows), task_count),
            "duration_ms": round(duration_ms, 3),
            "duration_sec": round(duration_ms / 1000.0, 3),
            "avg_duration_ms": round(duration_ms / max(1, task_count), 3),
            "avg_duration_sec": round(
                (duration_ms / max(1, task_count)) / 1000.0,
                3,
            ),
            "actions_executed": actions_executed,
            "avg_actions_per_task": round(
                actions_executed / max(1, task_count),
                3,
            ),
            "avg_ms_per_action": round(
                duration_ms / max(1, actions_executed),
                3,
            ),
            "model_calls": sum(
                _coerce_int(row.get("model_calls")) for row in canonical_rows
            ),
            "total_tokens": sum(
                _coerce_int(row.get("total_tokens")) for row in canonical_rows
            ),
            "replay_step_completed_count": replay_step_completed,
            "replay_step_total": replay_step_total,
            "replay_step_completed_rate": _rate(
                replay_step_completed,
                replay_step_total,
            ),
            "fallback_steps": sum(
                _coerce_int(row.get("fallback_steps")) for row in canonical_rows
            ),
            "relocation_diagnostic_count": sum(
                _coerce_int(row.get("relocation_diagnostic_count"))
                for row in canonical_rows
            ),
            "per_task": canonical_rows,
        }
    )
    return aggregate


def _write_result_summary(
    *,
    output_root: str | Path,
    task: str,
    command_records: Sequence[dict[str, Any]],
    aggregate_summary: dict[str, Any],
) -> dict[str, Any]:
    task_root = resolve_path(output_root) / _safe_stem(task)
    task_root.mkdir(parents=True, exist_ok=True)
    rows = _result_summary_rows(
        task=task,
        command_records=command_records,
        aggregate_summary=aggregate_summary,
    )
    compact_rows = [
        compact_result_row(
            row,
            source_seed=SOURCE_SEED,
            evaluation_seed=TASK_SEED,
        )
        for row in rows
    ]
    canonical_aggregate = _aggregate_normalized_result_rows(
        aggregate_summary,
        rows,
    )
    summary = {
        "schema_version": RESULT_SCHEMA,
        "task_name": task,
        "task_root": str(task_root),
        "rows": compact_rows,
        "details": rows,
        "aggregate": canonical_aggregate,
    }
    summary_path = task_root / RESULT_SUMMARY_FILE
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    commands_path = task_root / RESULT_COMMANDS_FILE
    commands_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in command_records)
        + "\n",
        encoding="utf-8",
    )

    visible_columns = [(field, field) for field in RESULT_FIELDS]
    md_lines = [
        f"# AndroidWorld Result Summary: {task}",
        "",
        "| " + " | ".join(label for _, label in visible_columns) + " |",
        "|" + "|".join("---" for _ in visible_columns) + "|",
    ]
    for row in compact_rows:
        md_lines.append(
            "| "
            + " | ".join(_result_row_value(row, key) for key, _ in visible_columns)
            + " |"
        )
    (task_root / RESULT_MARKDOWN_FILE).write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )
    return summary


def _print_result_summary(summary: dict[str, Any]) -> None:
    visible_columns = [(field, field) for field in RESULT_FIELDS]
    print(
        "| " + " | ".join(label for _, label in visible_columns) + " |",
        flush=True,
    )
    print("|" + "|".join("---" for _ in visible_columns) + "|", flush=True)
    for row in summary.get("rows") or []:
        print(
            "| "
            + " | ".join(_result_row_value(row, key) for key, _ in visible_columns)
            + " |",
            flush=True,
        )


def build_mobilegpt_command(
    item: CanonicalRunLog,
    *,
    method_name: str,
    target: DeviceTarget,
    android_world_root: str | Path,
    output_root: str | Path,
    stats_jsonl: str | Path,
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
    run_dir_suffix: str = "",
    repo_root: Path = REPO_ROOT,
) -> CommandSpec:
    spec = build_task_command(
        item,
        android_world_root=android_world_root,
        output_root=output_root,
        method_name=method_name,
        agent_name="mobilegpt",
        device_label=target.label,
        run_dir_suffix=run_dir_suffix,
        serial=target.serial,
        console_port=target.console_port,
        adb_path=adb_path,
        max_steps=max_steps,
        task_random_seed=task_random_seed,
        fixed_task_seed=fixed_task_seed,
        fixed_task_params=fixed_task_params,
        task_params_override=task_params_override,
        perform_emulator_setup=perform_emulator_setup,
        repo_root=repo_root,
    )
    client_host = str(server_host or "127.0.0.1").strip()
    if client_host in {"0.0.0.0", "::", "[::]"}:
        client_host = "127.0.0.1"
    return CommandSpec(
        label=f"mobilegpt:cross-device:{target.label}:androidworld-episode",
        argv=spec.argv,
        env={
            **spec.env,
            "ANDROID_SERIAL": target.serial,
            "MOBILEGPT_STATS_JSONL": str(resolve_path(stats_jsonl, root=repo_root)),
            "MOBILEGPT_RUNTIME_OBSERVE_BACKEND": "androidworld",
            "MOBILEGPT_SERVER_HOST": client_host,
            "MOBILEGPT_SERVER_PORT": str(int(server_port)),
            "MOBILEGPT_TARGET_PACKAGE": str(target_package or "").strip(),
            "MOBILEGPT_WAIT_START_TIMEOUT_SEC": str(float(start_timeout_sec)),
            "MOBILEGPT_WAIT_FINISH_TIMEOUT_SEC": str(float(finish_timeout_sec)),
            "MOBILEGPT_APP_READY_TIMEOUT_SEC": str(float(app_ready_timeout_sec)),
            "MOBILEGPT_APP_READY_POLL_SEC": str(float(app_ready_poll_sec)),
        },
        cwd=spec.cwd,
        output_path=spec.output_path,
        timeout_sec=(
            float(timeout_sec) if timeout_sec is not None and timeout_sec > 0 else None
        ),
        metadata={
            **dict(spec.metadata),
            "mode": "mobilegpt_androidworld_episode",
            "device_target": target.to_dict(),
            "mobilegpt_stats_jsonl": str(stats_jsonl),
            "mobilegpt_server_host": client_host,
            "mobilegpt_server_port": int(server_port),
            "mobilegpt_app_ready_timeout_sec": float(app_ready_timeout_sec),
            "mobilegpt_app_ready_poll_sec": float(app_ready_poll_sec),
            "target_package": str(target_package or "").strip(),
            "official_lifecycle": True,
            "state_backend": "androidworld",
            "action_backend": "androidworld",
            "androidworld_lifecycle_backend": "androidworld",
            "native_androidworld_agent_io": True,
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
    docs_root: str | Path | None = None,
    teacher_source: str | Path | None = None,
    workspace_root: str | Path | None = None,
    demo_name: str = "",
    max_steps: int,
    timeout_sec: int,
    task_random_seed: int | None,
    fixed_task_seed: bool,
    fixed_task_params: bool,
    task_params_override: dict[str, Any] | None,
    perform_emulator_setup: bool,
    adb_path: str,
    python_executable: str = sys.executable,
    repo_root: Path = REPO_ROOT,
) -> CommandSpec:
    teacher_mode = teacher_source is not None
    if teacher_mode and workspace_root is None:
        raise ValueError("appagent_teacher_workspace_required")
    if teacher_mode and docs_root is not None:
        raise ValueError("appagent_teacher_docs_forbidden")
    if not teacher_mode and docs_root is None:
        raise ValueError("appagent_native_memory_required")
    selector = "appagent"
    spec = build_task_command(
        item,
        android_world_root=android_world_root,
        output_root=output_root,
        method_name=method_name,
        agent_name=selector,
        device_label=target.label,
        serial=target.serial,
        console_port=target.console_port,
        adb_path=adb_path,
        max_steps=max_steps,
        timeout_sec=timeout_sec,
        task_random_seed=task_random_seed,
        fixed_task_seed=fixed_task_seed,
        fixed_task_params=fixed_task_params,
        task_params_override=task_params_override,
        perform_emulator_setup=perform_emulator_setup,
        python_executable=python_executable,
        repo_root=repo_root,
    )
    resolved_appagent_root = resolve_path(appagent_root, root=repo_root)
    argv = [*spec.argv, "--appagent-root", str(resolved_appagent_root)]
    resolved_docs_root: Path | None = None
    resolved_teacher_source: Path | None = None
    resolved_workspace_root: Path | None = None
    if docs_root is not None:
        resolved_docs_root = resolve_path(docs_root, root=repo_root)
        argv.extend(["--appagent-docs-root", str(resolved_docs_root)])
    if teacher_mode:
        resolved_teacher_source = resolve_path(teacher_source, root=repo_root)
        resolved_workspace_root = resolve_path(workspace_root, root=repo_root)
        argv.extend(
            [
                "--appagent-teacher-source",
                str(resolved_teacher_source),
                "--appagent-workspace-root",
                str(resolved_workspace_root),
                "--appagent-name",
                str(demo_name or f"demo_{item.task}_seed{task_random_seed}"),
            ]
        )
    return CommandSpec(
        label=f"appagent:{'teacher' if teacher_mode else 'warm'}:{target.label}",
        argv=argv,
        env={**spec.env, "ANDROID_SERIAL": target.serial},
        cwd=spec.cwd,
        output_path=spec.output_path,
        timeout_sec=float(timeout_sec) if timeout_sec and timeout_sec > 0 else None,
        metadata={
            **dict(spec.metadata),
            "mode": (
                "appagent_source_human_demo"
                if teacher_mode
                else "appagent_native_deployment"
            ),
            "agent": selector,
            "device_target": target.to_dict(),
            "appagent_root": str(resolved_appagent_root),
            "appagent_docs_root": str(resolved_docs_root or ""),
            "appagent_teacher_source": str(resolved_teacher_source or ""),
            "appagent_workspace_root": str(resolved_workspace_root or ""),
            "uses_omniflow_function": False,
            "uses_appagent_docs": resolved_docs_root is not None,
            "teacher_mode": teacher_mode,
            "official_lifecycle": True,
            "state_backend": "androidworld",
            "action_backend": "androidworld",
            "androidworld_lifecycle_backend": "androidworld",
            "native_androidworld_agent_io": True,
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
            "MOBILEGPT_CHAT_MAX_ATTEMPTS": "1",
            **(
                {"MOBILEGPT_CHAT_MODEL": normalized_model}
                if normalized_model
                else {}
            ),
        },
        metadata={
            **spec.metadata,
            "model_max_attempts": 1,
            "episode_retries": 0,
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

    if item.meta.get("latest_official_success_source") is not True:
        raise ValueError(
            "mobilegpt_requires_official_success_source:"
            f"task={item.task}"
        )
    source_memory_value = str(
        getattr(args, "mobilegpt_source_memory_root", "")
    ).strip()
    if not source_memory_value:
        raise ValueError(
            "mobilegpt requires --mobilegpt-source-memory-root"
        )
    source_memory_root = resolve_path(source_memory_value)
    if not source_memory_root.is_dir():
        raise FileNotFoundError(f"mobilegpt_source_memory_missing:{source_memory_root}")
    source_manifest_path = source_memory_root.parent / MOBILEGPT_MEMORY_MANIFEST
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_schema = str(source_manifest.get("schema_version") or "")
    try:
        source_method = MOBILEGPT_SOURCE_METHOD_BY_SCHEMA[source_schema]
        source_prep_type = MOBILEGPT_PREP_TYPE_BY_SCHEMA[source_schema]
    except KeyError as error:
        raise ValueError("mobilegpt_source_memory_schema_invalid") from error
    adapted_memory = mobilegpt_memory.validate_mobilegpt_adapted_memory(
        source_memory_root,
        task_name=item.task,
        source_seed=SOURCE_SEED,
        source_run_log=source_run_log,
        compatible_source_sha256s=compatible_source_sha256s,
        expected_model=str(args.model or ""),
        expected_source_method=source_method,
    )

    memory_root = _method_memory_root(output_root, item.task, method)
    memory_root.mkdir(parents=True, exist_ok=True)
    frozen_memory_root = memory_root / "frozen_memory"
    frozen_memory_manifest_path = memory_root / "frozen_memory_manifest.json"
    episodes_root = memory_root / "_episodes"
    source_target = _infer_mobilegpt_target_from_source_run_log(item)
    explicit_target_package = _mobilegpt_target_package_from_open_target_app(
        args.mobilegpt_open_target_app
    )
    target_package = (
        explicit_target_package
        or str(adapted_memory.get("target_package") or "").strip()
        or str(source_target.get("target_package") or "").strip()
    )
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
    target_source = (
        "mobilegpt_open_target_app"
        if explicit_target_package
        else "sealed_converted_source_memory"
    )
    memory_condition = "converted_runlog_memory"
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
        source_seed=SOURCE_SEED,
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
            server_spec = build_mobilegpt_server_command(
                "server",
                mobilegpt_root=args.mobilegpt_root,
                mobilegpt_memory_root=episode_memory_root,
                stats_jsonl=stats_jsonl,
                server_host=args.mobilegpt_server_host,
                port=int(args.mobilegpt_port),
                serial=target.serial,
                adb_path=args.adb_path,
                target_package=target_package,
                target_app=target_app,
                runtime_observe_backend="androidworld",
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
                    server_host=args.mobilegpt_server_host,
                    server_port=int(args.mobilegpt_port),
                    target_package=target_package,
                    max_steps=int(args.max_steps or MAX_STEPS),
                    task_random_seed=task_seed,
                    fixed_task_seed=not bool(args.no_fixed_task_seed),
                    fixed_task_params=not bool(args.no_fixed_task_params),
                    task_params_override=task_params_override,
                    perform_emulator_setup=bool(args.perform_emulator_setup),
                    adb_path=args.adb_path,
                    start_timeout_sec=float(args.mobilegpt_wait_start_timeout_sec),
                    finish_timeout_sec=float(args.mobilegpt_episode_wait_timeout_sec),
                    app_ready_timeout_sec=float(
                        args.mobilegpt_app_ready_timeout_sec
                    ),
                    app_ready_poll_sec=float(args.mobilegpt_app_ready_poll_sec),
                    timeout_sec=float(args.timeout_sec or 0),
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
    selected = _select_from_args(args)
    if len(selected) != 1:
        raise ValueError("result requires exactly one selected --task entry")
    item = selected[0]
    methods = (args.method,)
    targets = parse_device_targets(args.device)
    if len(targets) != 1:
        raise ValueError("result requires exactly one device")
    mobilegpt_source_run_log = item.source_run_log
    mobilegpt_source_run_log_sha256s = (
        sha256_file(mobilegpt_source_run_log),
    )
    attempt_root, _ = _task_managed_output_root(args.output_path)
    source_seed = int(args.source_seed)
    output_root = _source_seed_output_root(attempt_root, source_seed)
    attempt_id = attempt_root.name
    task_params_override = _task_params_override_from_args(args)
    task_seed = (
        random.randint(1, 2**31 - 1)
        if bool(args.random_task_seed)
        else args.task_random_seed
    )
    attempt_manifest_path = _claim_result_attempt(
        attempt_root,
        task=item.task,
        methods=methods,
        source_seed=source_seed,
        evaluation_seed=task_seed,
        dry_run=bool(args.dry_run),
    )
    command_records: list[dict[str, Any]] = []
    failed = 0

    for method in methods:
        memory_root = _method_memory_root(output_root, item.task, method)
        _claim_method_memory_root(memory_root)
        source_action_hint_path: Path | None = None
        appagent_docs_root: Path | None = None
        appagent_prep: dict[str, Any] = {}
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
            if not store_text:
                raise ValueError(
                    f"--store-path is required when result includes {method}"
                )
            store_path = resolve_path(store_text)
        else:
            store_path = memory_root / "unused-store.json"

        if method == "omniflow":
            transfer_asset_audit: dict[str, Any] = {}
            if not args.dry_run:
                transfer_asset_audit = validate_omniflow_transfer_assets(
                    store_path,
                    require_action_transfer=True,
                )
            _write_method_memory_manifest(
                memory_root=memory_root,
                task=item.task,
                method=method,
                memory_mode="omniflow_function_store",
                source_seed=source_seed,
                evaluation_seed=task_seed,
                attempt_id=attempt_id,
                source_run_log=item.source_run_log,
                artifacts={
                    "store_path": str(store_path),
                    "store_sha256": sha256_file(store_path)
                    if store_path.is_file()
                    else None,
                    "recorded_source_seed": item.replay_seed,
                    "function_authoring": "external_agent_skill",
                    "transfer_asset_audit": transfer_asset_audit,
                    "transfer_state_catalog_sha256": (
                        sha256_file(transfer_asset_audit["transfer_state_catalog"])
                        if transfer_asset_audit.get("transfer_state_catalog")
                        else None
                    ),
                },
            )
        if method == "appagent":
            source_memory_text = str(
                getattr(args, "appagent_memory_root", "") or ""
            ).strip()
            if not source_memory_text:
                raise ValueError("appagent requires --appagent-memory-root")
            source_memory_root = resolve_path(source_memory_text)
            provenance = validate_appagent_memory(
                source_memory_root,
                task_name=item.task,
                source_run_log=item.source_run_log,
            )
            appagent_docs_root = Path(provenance["demo_docs_root"]).resolve()
            source_metrics = dict(provenance["source_episode_metrics"])
            document_usage = dict(provenance["doc_generation_usage"])
            prep_model_calls = _coerce_int(
                source_metrics.get("model_calls")
            ) + _coerce_int(document_usage.get("model_calls"))
            prep_prompt_tokens = _coerce_int(
                source_metrics.get("prompt_tokens")
            ) + _coerce_int(document_usage.get("prompt_tokens"))
            prep_completion_tokens = _coerce_int(
                source_metrics.get("completion_tokens")
            ) + _coerce_int(document_usage.get("completion_tokens"))
            prep_total_tokens = prep_prompt_tokens + prep_completion_tokens
            source_memory_manifest = source_memory_root / "appagent_manifest.json"
            appagent_prep = {
                "type": "appagent_native_source_demo_docs",
                "model_calls": prep_model_calls,
                "prompt_tokens": prep_prompt_tokens,
                "completion_tokens": prep_completion_tokens,
                "total_tokens": prep_total_tokens,
                "token_usage_status": (
                    "tracked" if prep_model_calls > 0 else "not_applicable"
                ),
                "wall_sec": _coerce_float(provenance.get("prep_wall_sec")),
                "source_episode_duration_sec": _coerce_float(
                    source_metrics.get("duration_sec")
                ),
                "source_episode_wall_sec": _coerce_float(
                    source_metrics.get("wall_sec")
                ),
                "document_generation_wall_sec": _coerce_float(
                    document_usage.get("wall_sec")
                ),
                "official_validator_success": bool(
                    provenance.get("official_source_success")
                ),
                "manifest_path": str(source_memory_manifest),
                "manifest_sha256": sha256_file(source_memory_manifest),
                "demo_sha256": str(provenance.get("demo_sha256") or ""),
                "demo_docs_sha256": str(provenance.get("demo_docs_sha256") or ""),
                "shared_across_targets": True,
            }
            memory_mode = "appagent_native_demo_docs"
            artifacts = {
                "source_memory_root": str(source_memory_root),
                "source_memory_manifest": str(
                    source_memory_root / "appagent_manifest.json"
                ),
                "source_memory_manifest_sha256": sha256_file(
                    source_memory_root / "appagent_manifest.json"
                ),
                "demo_docs_root": str(appagent_docs_root),
                "demo_docs_sha256": provenance["demo_docs_sha256"],
                "official_appagent_revision": provenance[
                    "official_appagent_revision"
                ],
                "uses_appagent_docs": True,
                "uses_omniflow_function": False,
                "memory_read_only": True,
            }
            _write_method_memory_manifest(
                memory_root=memory_root,
                task=item.task,
                method=method,
                memory_mode=memory_mode,
                source_seed=source_seed,
                evaluation_seed=task_seed,
                attempt_id=attempt_id,
                source_run_log=item.source_run_log,
                artifacts=artifacts,
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
            source_hint_store_path = (
                resolve_path(str(args.store_path))
                if str(args.store_path or "").strip()
                else None
            )
            source_action_hint_path = _source_action_hint_path_for_item(
                item,
                output_root=memory_root,
                store_path=source_hint_store_path,
            )
            _write_method_memory_manifest(
                memory_root=memory_root,
                task=item.task,
                method=method,
                memory_mode="source_action_hint",
                source_seed=source_seed,
                evaluation_seed=task_seed,
                attempt_id=attempt_id,
                source_run_log=item.source_run_log,
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
            if method == "fixed_replay":
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
                spec = build_appagent_command(
                    item,
                    method_name=method,
                    target=target,
                    android_world_root=args.android_world_root,
                    output_root=output_root,
                    appagent_root=args.appagent_root,
                    docs_root=appagent_docs_root,
                    max_steps=int(args.max_steps or MAX_STEPS),
                    timeout_sec=int(args.timeout_sec or 0),
                    task_random_seed=task_seed,
                    fixed_task_seed=not bool(args.no_fixed_task_seed),
                    fixed_task_params=not bool(args.no_fixed_task_params),
                    task_params_override=task_params_override,
                    perform_emulator_setup=bool(args.perform_emulator_setup),
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
                    fixed_task_params=not bool(args.no_fixed_task_params),
                    task_params_override=task_params_override,
                    perform_emulator_setup=bool(args.perform_emulator_setup),
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
            if appagent_prep:
                spec.metadata["appagent_prep"] = dict(appagent_prep)
            returncode = run_command(spec, dry_run=args.dry_run)
            status = "completed" if returncode == 0 else "command_failed"
            command_records.append(
                _command_record_from_spec(
                    spec,
                    task=item.task,
                    method=method,
                    device=target.label,
                    returncode=returncode,
                    status=status,
                    extra_metadata={"device_target": target.to_dict()},
                )
            )
            if returncode != 0:
                failed += 1
                if args.fail_fast:
                    break
        if failed and args.fail_fast:
            break

    if bool(args.dry_run):
        for record in command_records:
            metadata = dict(record.get("metadata") or {})
            metadata["dry_run"] = True
            record["metadata"] = metadata

    aggregate_paths = [
        path
        for record in command_records
        for path in _formal_result_paths(record)
    ]
    aggregate_summary = aggregate_task_results([] if args.dry_run else aggregate_paths)
    summary = _write_result_summary(
        output_root=output_root,
        task=item.task,
        command_records=command_records,
        aggregate_summary=aggregate_summary,
    )
    result_registration: dict[str, Any] = {}
    summary_path = output_root / _safe_stem(item.task) / RESULT_SUMMARY_FILE
    if not bool(args.dry_run):
        result_registry_root = _result_registry_root(
            args,
            attempt_root=attempt_root,
        )
        result_registration = register_attempt_summary(
            summary_path=summary_path,
            attempt_manifest_path=attempt_manifest_path,
            runs_root=result_registry_root,
            local_data_index=Path(
                os.environ["OMNIFLOW_EXP_MEMORY_INDEX"]
            ).expanduser()
            if os.environ.get("OMNIFLOW_EXP_MEMORY_INDEX")
            else None,
        )
    _print_result_summary(summary)
    print(
        f"[result] summary={summary_path}",
        flush=True,
    )
    if result_registration:
        print(
            "[result] registered="
            f"{result_registration.get('registered_results_count', 0)} "
            f"ledger_appended={result_registration.get('ledger_records_appended', 0)} "
            f"registry={result_registry_root}",
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
    result_parser.add_argument("--index", default=str(DEFAULT_DATA_INDEX))
    result_parser.add_argument(
        "--android-world-root", default=str(DEFAULT_ANDROID_WORLD_ROOT)
    )
    result_parser.add_argument(
        "--output-path",
        dest="output_path",
        default=str(DEFAULT_OUTPUT_ROOT),
        help=(
            "Exact fresh immutable attempt directory. The shared canonical "
            "runtime/evals/androidworld_validator/runs root is rejected."
        ),
    )
    result_parser.add_argument(
        "--result-registry-root",
        default=os.environ.get("ANDROIDWORLD_RESULT_REGISTRY_ROOT", ""),
        help=(
            "Canonical immutable task/method/device/attempt registry. Defaults "
            "to the androidworld_validator root containing --index."
        ),
    )
    result_parser.add_argument("--task", required=True)
    result_parser.add_argument("--source-seed", type=int, default=SOURCE_SEED)
    result_parser.add_argument(
        "--method",
        choices=METHODS,
        default=DEFAULT_SOURCE_METHOD,
        help="One paper method for this AndroidWorld result.",
    )
    result_parser.add_argument(
        "--store-path",
        default="",
        help=("Validated omniflow.store.v2 required by the OmniFlow methods."),
    )
    result_parser.add_argument(
        "--store-index",
        default="",
        help="Canonical task-to-Store index used by frozen source assets.",
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
    result_parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    result_parser.add_argument(
        "--max-fallback-steps",
        type=int,
        default=None,
        help=(
            "Maximum VLM fallback planner calls for omniflow. Function actions do not "
            "consume this budget; omitted means the normal max-step behavior."
        ),
    )
    result_parser.add_argument("--timeout-sec", type=int, default=EPISODE_TIMEOUT_SEC)
    result_parser.add_argument(
        "--task-random-seed",
        type=int,
        default=DEFAULT_TASK_RANDOM_SEED,
        help="AndroidWorld warm/target seed. Formal experiments use 113.",
    )
    result_parser.add_argument("--random-task-seed", action="store_true")
    result_parser.add_argument("--no-fixed-task-seed", action="store_true")
    result_parser.add_argument("--no-fixed-task-params", action="store_true")
    result_parser.add_argument(
        "--task-params-json",
        default="",
        help="Override archived task params with this JSON object.",
    )
    result_parser.add_argument("--planner-provider", default="")
    result_parser.add_argument("--model", default="")
    result_parser.add_argument(
        "--planner-timeout-sec", type=float, default=STEP_TIMEOUT_SEC
    )
    _add_androidworld_setup_args(result_parser)
    result_parser.add_argument(
        "--mobilegpt-root", default=str(DEFAULT_MOBILEGPT_ROOT)
    )
    result_parser.add_argument(
        "--appagent-root",
        default=str(REPO_ROOT / "data" / "runtime" / "external" / "appagent"),
    )
    result_parser.add_argument(
        "--appagent-memory-root",
        default="",
        help=(
            "Sealed source-111 AppAgent human-demo workspace required by appagent."
        ),
    )
    result_parser.add_argument("--mobilegpt-server-host", default="0.0.0.0")
    result_parser.add_argument("--mobilegpt-port", type=int, default=12345)
    result_parser.add_argument(
        "--mobilegpt-server-warmup-sec", type=float, default=5.0
    )
    result_parser.add_argument(
        "--mobilegpt-wait-start-timeout-sec",
        type=float,
        default=DEFAULT_MOBILEGPT_WAIT_START_TIMEOUT_SEC,
        help=(
            "Seconds to wait for the native MobileGPT server connection. "
            "Use -1 to wait indefinitely."
        ),
    )
    result_parser.add_argument(
        "--mobilegpt-episode-wait-timeout-sec",
        type=float,
        default=DEFAULT_MOBILEGPT_EPISODE_WAIT_TIMEOUT_SEC,
        help=(
            "Seconds to wait for MobileGPT task_finished before official "
            "AndroidWorld validation. Use -1 to wait indefinitely."
        ),
    )
    result_parser.add_argument(
        "--mobilegpt-app-ready-timeout-sec",
        type=float,
        default=DEFAULT_MOBILEGPT_APP_READY_TIMEOUT_SEC,
        help="Seconds to wait for indexed target-app UI after each app launch.",
    )
    result_parser.add_argument(
        "--mobilegpt-app-ready-poll-sec",
        type=float,
        default=DEFAULT_MOBILEGPT_APP_READY_POLL_SEC,
        help="Polling interval for target-app UI readiness.",
    )
    result_parser.add_argument("--mobilegpt-open-target-app", default="")
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
