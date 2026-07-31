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
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omniflow.core.trajectory import canonicalize_run_log
from omniflow.functions.store import FunctionStore
from src.experiment.result_registry import register_attempt_summary
from src.integrations.appagent_adapter import validate_appagent_demo_memory

DEFAULT_ARCHIVE_INDEX = (
    REPO_ROOT
    / "runtime"
    / "evals"
    / "androidworld_validator"
    / "core_archive"
    / "success_source_runlogs"
    / "index_by_task.json"
)
DEFAULT_ANDROID_WORLD_ROOT = (
    REPO_ROOT / "runtime" / "external" / "droidrun-android-world" / "android_world"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "runtime" / "evals" / "androidworld_validator" / "runs"
)
DEFAULT_MOBILEGPT_ROOT = REPO_ROOT / "runtime" / "external" / "mobilegpt"
DEFAULT_MOBILEGPT_STATS_JSONL = (
    DEFAULT_OUTPUT_ROOT / "_mobilegpt_stats" / "mobilegpt_stats.jsonl"
)
DEFAULT_MOBILEGPT_STATS_SUMMARY = (
    DEFAULT_OUTPUT_ROOT / "_mobilegpt_stats" / "mobilegpt_stats_summary.json"
)
DEFAULT_DEVICE_TARGETS = "small5554:emulator-5554:5554,fold5564:emulator-5564:5564"
DEFAULT_MOBILEGPT_WAIT_START_TIMEOUT_SEC = 60.0
DEFAULT_MOBILEGPT_EPISODE_WAIT_TIMEOUT_SEC = 120.0
DEFAULT_EVAL_TASK_RANDOM_SEED = 113
DEFAULT_SOURCE_METHOD = "fixed_replay"


@dataclass(frozen=True)
class ArchivedRunLog:
    task: str
    goal: str
    params: dict[str, Any]
    source_run_log: Path
    replay_seed: int
    step_count: int
    meta: dict[str, Any]


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


@dataclass(frozen=True)
class SourceRunLogProfile:
    task: str
    source_run_log: Path
    replay_format: str
    step_count: int
    card_count: int
    accepted_first30: bool
    latest_official_success_source: bool
    direct_replay_ready: bool
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "source_run_log": str(self.source_run_log),
            "replay_format": self.replay_format,
            "step_count": self.step_count,
            "card_count": self.card_count,
            "accepted_first30": self.accepted_first30,
            "latest_official_success_source": self.latest_official_success_source,
            "direct_replay_ready": self.direct_replay_ready,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class AndroidWorldAppSpec:
    app_name: str
    package_name: str
    apk_names: tuple[str, ...]
    tasks: tuple[str, ...]
    source_formats: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "app_name": self.app_name,
            "package_name": self.package_name,
            "apk_names": list(self.apk_names),
            "tasks": list(self.tasks),
            "source_formats": list(self.source_formats),
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


def _repo_path(value: str | Path, *, repo_root: Path = REPO_ROOT) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _normalize_config_arg_key(value: str) -> str:
    return str(value or "").strip().lstrip("-").replace("-", "_")


def _load_experiment_config(path: str | Path) -> dict[str, Any]:
    config_path = _repo_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"experiment config not found: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "YAML experiment configs require PyYAML; use JSON instead"
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"experiment config must be a JSON/YAML object: {config_path}")
    return data


def _parser_option_dest_map(parser: argparse.ArgumentParser) -> dict[str, str]:
    option_dests: dict[str, str] = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_dests[option] = action.dest
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                option_dests.update(_parser_option_dest_map(subparser))
    return option_dests


def _explicit_option_dests(
    parser: argparse.ArgumentParser,
    argv: Sequence[str],
) -> set[str]:
    option_dests = _parser_option_dest_map(parser)
    explicit: set[str] = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        option = token.split("=", 1)[0]
        dest = option_dests.get(option)
        if dest:
            explicit.add(dest)
    return explicit


def _section_from_experiment_config(
    config: dict[str, Any],
    command: str,
) -> dict[str, Any]:
    command_key = _normalize_config_arg_key(command)
    sections: list[dict[str, Any]] = []
    defaults = config.get("defaults")
    if isinstance(defaults, dict):
        sections.append(defaults)
    for key in (command, command_key):
        value = config.get(key)
        if isinstance(value, dict):
            sections.append(value)
            break
    if not sections:
        sections.append(config)
    merged: dict[str, Any] = {}
    for section in sections:
        merged.update(section)
    return merged


def apply_experiment_config(
    args: argparse.Namespace,
    argv: Sequence[str],
    *,
    parser: argparse.ArgumentParser,
) -> argparse.Namespace:
    config_path = str(getattr(args, "experiment_config", "") or "").strip()
    if not config_path:
        return args
    config = _load_experiment_config(config_path)
    values = _section_from_experiment_config(
        config,
        str(getattr(args, "command", "") or ""),
    )
    explicit_dests = _explicit_option_dests(parser, argv)
    explicit_dests.discard("experiment_config")
    metadata_keys = {"schema_version", "description", "notes"}
    for raw_key, value in values.items():
        if raw_key in metadata_keys:
            continue
        dest = _normalize_config_arg_key(raw_key)
        if dest in metadata_keys:
            continue
        if not hasattr(args, dest):
            raise ValueError(
                f"unknown experiment config key for {args.command}: {raw_key}"
            )
        if dest in explicit_dests:
            continue
        setattr(args, dest, value)
    setattr(args, "experiment_config_path", str(_repo_path(config_path)))
    return args


def _default_e2e_store_path(
    output_root: str | Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> Path:
    """Use a per-run Function Store so experiments never read shared state."""

    return (
        _repo_path(output_root, repo_root=repo_root)
        / "_runtime"
        / "omniflow"
        / "store.json"
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


def _archive_ref_path(
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

    repo_relative = _repo_path(path, repo_root=repo_root)
    if repo_relative.exists():
        return repo_relative

    return repo_relative


def _safe_stem(value: str, *, fallback: str = "task") -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("._")
    return stem[:120] or fallback


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


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _path_ref_from(base_dir: Path, path: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), base_dir.resolve())
    except ValueError:
        return str(path.resolve())


def _safe_relative_path(value: str, *, fallback: str = "run") -> Path:
    parts = [
        _safe_stem(part, fallback="")
        for part in re.split(r"[\\/]+", str(value or "").strip())
        if str(part or "").strip()
    ]
    parts = [part for part in parts if part]
    if not parts:
        return Path(_safe_stem("", fallback=fallback))
    return Path(*parts)


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
        _repo_path(output_root, repo_root=repo_root)
        / _safe_stem(task)
        / _safe_stem(method, fallback="method")
        / _device_label(
            explicit_label=device,
            serial=serial,
            console_port=console_port,
        )
    )


def _androidworld_validator_root(*, repo_root: Path = REPO_ROOT) -> Path:
    return repo_root / "runtime" / "evals" / "androidworld_validator"


def _result_registration_roots(
    args: argparse.Namespace,
    *,
    attempt_root: Path,
) -> tuple[Path, Path]:
    explicit_runs = str(getattr(args, "result_registry_root", "") or "").strip()
    explicit_master = str(getattr(args, "master_progress_root", "") or "").strip()
    if explicit_runs and explicit_master:
        return _repo_path(explicit_runs), _repo_path(explicit_master)
    if explicit_runs:
        runs_root = _repo_path(explicit_runs)
        return runs_root, runs_root.parent / "master_progress"
    if explicit_master:
        master_root = _repo_path(explicit_master)
        return master_root.parent / "runs", master_root

    index_path = _repo_path(args.index)
    for candidate in (index_path.parent, *index_path.parents):
        if candidate.name == "androidworld_validator":
            return candidate / "runs", candidate / "master_progress"
    fallback = attempt_root.parent / "_androidworld_result_registry"
    return fallback / "runs", fallback / "master_progress"


def _task_managed_output_root(
    output_root: str | Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> tuple[Path, str]:
    """Keep the caller's exact immutable attempt root authoritative."""
    resolved = _repo_path(output_root, repo_root=repo_root)
    canonical_shared_root = _androidworld_validator_root(repo_root=repo_root) / "runs"
    if resolved == canonical_shared_root:
        raise ValueError(
            f"output_root_must_be_fresh_attempt_child:{canonical_shared_root}"
        )
    return resolved, ""


def _source_seed_output_root(output_root: str | Path, source_seed: int) -> Path:
    return _repo_path(output_root) / f"source_seed_{int(source_seed)}"


def _claim_one_task_attempt(
    output_root: str | Path,
    *,
    task: str,
    methods: Sequence[str],
    source_seed: int,
    evaluation_seed: int | None,
    task_iteration: int = 1,
    baseline_environment_repair: str = "",
    dry_run: bool = False,
    experiment_config: str | Path | None = None,
) -> Path:
    if not 1 <= int(task_iteration) <= 3:
        raise ValueError(
            f"task_iteration_out_of_range:expected=1..3:actual={task_iteration}"
        )
    root = _repo_path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "attempt_manifest.json"
    runner_path = Path(__file__).resolve()
    provenance: dict[str, Any] = {
        "runner": str(runner_path),
        "runner_sha256": _file_sha256(runner_path),
    }
    if str(experiment_config or "").strip():
        config_path = _repo_path(str(experiment_config))
        provenance.update(
            {
                "experiment_config": str(config_path),
                "experiment_config_sha256": _file_sha256(config_path),
            }
        )
    manifest = {
        "schema_version": "omniflow.androidworld_attempt.v1",
        "attempt_id": root.name,
        "task_name": task,
        "methods": list(methods),
        "source_seed": int(source_seed),
        "evaluation_seed": evaluation_seed,
        "task_iteration": int(task_iteration),
        "max_task_iterations": 3,
        "baseline_environment_repair": str(baseline_environment_repair or "").strip(),
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
    return _repo_path(output_root) / _safe_stem(task) / _safe_stem(method)


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


def _write_canonical_source_run_log(
    canonical: dict[str, Any],
    *,
    output_root: str | Path,
    task: str,
    suffix: str = "",
    repo_root: Path = REPO_ROOT,
) -> Path:
    materialized_dir = (
        _repo_path(output_root, repo_root=repo_root) / "_canonical_source_runlogs"
    )
    materialized_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(task)
    if str(suffix or "").strip():
        stem = f"{stem}.{_safe_stem(suffix)}"
    materialized_path = materialized_dir / f"{stem}.run_log.json"
    materialized_path.write_text(
        json.dumps(canonical, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return materialized_path.resolve()


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


def load_archive_index(
    index_path: str | Path = DEFAULT_ARCHIVE_INDEX,
    *,
    repo_root: Path = REPO_ROOT,
) -> list[ArchivedRunLog]:
    path = _repo_path(index_path, repo_root=repo_root)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Archive index must be a JSON object: {path}")

    by_task_root = path.parent / "by_task"
    if by_task_root.is_dir():
        for metadata_path in sorted(by_task_root.glob("*/metadata.json")):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                not isinstance(metadata, dict)
                or metadata.get("androidworld_success") is not True
            ):
                continue
            task = str(metadata.get("task") or metadata_path.parent.name).strip()
            source_run_log = metadata_path.parent / "source.run_log.json"
            if not task or task in data or not source_run_log.is_file():
                continue
            data[task] = {
                **metadata,
                "retained_source_run_log": _path_ref_from(
                    path.parent,
                    source_run_log,
                ),
            }

    archive: list[ArchivedRunLog] = []
    for task, raw_meta in data.items():
        if not isinstance(raw_meta, dict):
            continue
        retained = str(raw_meta.get("retained_source_run_log") or "").strip()
        if not retained:
            retained = (
                "runtime/evals/androidworld_validator/core_archive/"
                f"success_source_runlogs/by_task/{task}/source.run_log.json"
            )
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
        archive.append(
            ArchivedRunLog(
                task=str(task),
                goal=str(raw_meta.get("goal") or ""),
                params=dict(params),
                source_run_log=_archive_ref_path(
                    retained,
                    index_path=path,
                    repo_root=repo_root,
                ),
                replay_seed=seed,
                step_count=_coerce_int(raw_meta.get("step_count"), 0),
                meta=dict(raw_meta),
            )
        )
    return archive


def select_archive_items(
    archive: Sequence[ArchivedRunLog],
    *,
    tasks: str = "",
    first60: bool = False,
    limit: int | None = None,
) -> list[ArchivedRunLog]:
    if tasks.strip():
        by_name = {item.task: item for item in archive}
        selected: list[ArchivedRunLog] = []
        missing: list[str] = []
        for raw_name in tasks.split(","):
            name = raw_name.strip()
            if not name:
                continue
            item = by_name.get(name)
            if item is None:
                missing.append(name)
            else:
                selected.append(item)
        if missing:
            raise KeyError(f"Tasks not found in archive index: {', '.join(missing)}")
    else:
        selected = list(archive[:60] if first60 else archive)

    if limit is not None:
        selected = selected[: max(0, int(limit))]
    return selected


def profile_source_run_log(item: ArchivedRunLog) -> SourceRunLogProfile:
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
            accepted_first30=bool(item.meta.get("accepted_first30")),
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
            accepted_first30=bool(item.meta.get("accepted_first30")),
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
            accepted_first30=bool(item.meta.get("accepted_first30")),
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
        accepted_first30=bool(item.meta.get("accepted_first30")),
        latest_official_success_source=bool(
            item.meta.get("latest_official_success_source")
        ),
        direct_replay_ready=False,
        notes=("no canonical steps or payload cards found",),
    )


def filter_archive_items(
    items: Sequence[ArchivedRunLog],
    *,
    source_format: str = "all",
    accepted_first30: bool = False,
) -> list[ArchivedRunLog]:
    selected: list[ArchivedRunLog] = []
    for item in items:
        if accepted_first30 and not bool(item.meta.get("accepted_first30")):
            continue
        if source_format == "all":
            selected.append(item)
            continue
        profile = profile_source_run_log(item)
        if source_format == "canonical" and profile.replay_format == "canonical_steps":
            selected.append(item)
        elif source_format == "payload" and profile.replay_format == "payload_cards":
            selected.append(item)
        elif source_format == "ready" and profile.direct_replay_ready:
            selected.append(item)
    return selected


def _package_from_activity(activity: object) -> str:
    text = str(activity or "").strip()
    if "/" not in text:
        return ""
    return text.split("/", 1)[0].strip()


def _androidworld_cache_dir() -> Path:
    return Path(tempfile.gettempdir()) / "android_world" / "app_data"


def _load_androidworld_modules(
    android_world_root: str | Path = DEFAULT_ANDROID_WORLD_ROOT,
    *,
    repo_root: Path = REPO_ROOT,
) -> tuple[object, object, object]:
    root = _repo_path(android_world_root, repo_root=repo_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from android_world import registry
        from android_world.env import adb_utils
        from android_world.env.setup_device import setup as aw_setup
    except ImportError as exc:
        raise RuntimeError(
            f"Unable to import AndroidWorld modules from {root}: {exc}"
        ) from exc
    return registry, aw_setup, adb_utils


def _apk_names(app_setup: object) -> tuple[str, ...]:
    raw = getattr(app_setup, "apk_names", ())
    if isinstance(raw, str):
        return (raw,) if raw else ()
    try:
        return tuple(str(item) for item in raw if str(item).strip())
    except TypeError:
        return ()


def collect_androidworld_app_specs(
    items: Sequence[ArchivedRunLog],
    *,
    android_world_root: str | Path = DEFAULT_ANDROID_WORLD_ROOT,
    suite_family: str = "android_world",
    repo_root: Path = REPO_ROOT,
) -> list[AndroidWorldAppSpec]:
    registry, aw_setup, adb_utils = _load_androidworld_modules(
        android_world_root,
        repo_root=repo_root,
    )
    task_types = registry.TaskRegistry().get_registry(family=suite_family)
    app_to_tasks: dict[str, list[str]] = {}
    app_to_formats: dict[str, set[str]] = {}
    app_setups: dict[str, object] = {}

    for item in items:
        task_type = task_types.get(item.task)
        app_names: list[str] = []
        for app_name in tuple(getattr(task_type, "app_names", ()) or ()):
            name = str(app_name or "").strip()
            if name and name not in app_names:
                app_names.append(name)
        for app_setup in aw_setup.get_app_list_to_setup([item.task]) or ():
            name = str(getattr(app_setup, "app_name", "") or "").strip()
            if name and name not in app_names:
                app_names.append(name)
                app_setups[name] = app_setup

        profile = profile_source_run_log(item)
        for app_name in app_names:
            app_to_tasks.setdefault(app_name, []).append(item.task)
            app_to_formats.setdefault(app_name, set()).add(profile.replay_format)
            app_setup = aw_setup.get_app_mapping(app_name)
            if app_setup is not None:
                app_setups[app_name] = app_setup

    specs: list[AndroidWorldAppSpec] = []
    for app_name in sorted(app_to_tasks):
        app_setup = app_setups.get(app_name)
        activity = adb_utils.get_adb_activity(app_name)
        package_name = _package_from_activity(activity)
        specs.append(
            AndroidWorldAppSpec(
                app_name=app_name,
                package_name=package_name,
                apk_names=_apk_names(app_setup) if app_setup is not None else (),
                tasks=tuple(app_to_tasks[app_name]),
                source_formats=tuple(sorted(app_to_formats.get(app_name, set()))),
            )
        )
    return specs


def list_cached_androidworld_apks(cache_dir: str | Path | None = None) -> set[str]:
    path = (
        Path(cache_dir).expanduser()
        if cache_dir is not None
        else _androidworld_cache_dir()
    )
    if not path.exists():
        return set()
    return {item.name for item in path.iterdir() if item.is_file()}


def read_adb_installed_packages(*, serial: str = "", adb_path: str = "") -> set[str]:
    argv = [adb_path.strip() or "adb"]
    if serial.strip():
        argv.extend(["-s", serial.strip()])
    argv.extend(["shell", "pm", "list", "packages"])
    completed = subprocess.run(
        argv,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "ADB package check failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    packages: set[str] = set()
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line.startswith("package:"):
            continue
        packages.add(line.removeprefix("package:").strip())
    return packages


def build_device_readiness_summary(
    specs: Sequence[AndroidWorldAppSpec],
    *,
    installed_packages: set[str] | None = None,
    cached_apks: set[str] | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    cache_path = (
        Path(cache_dir).expanduser()
        if cache_dir is not None
        else _androidworld_cache_dir()
    )
    installed_checked = installed_packages is not None
    cached_checked = cached_apks is not None
    app_rows: list[dict[str, Any]] = []
    task_to_apps: dict[str, list[str]] = {}

    for spec in sorted(specs, key=lambda item: item.app_name):
        installed = (
            bool(spec.package_name and spec.package_name in installed_packages)
            if installed_checked
            else None
        )
        missing_apks = (
            [apk for apk in spec.apk_names if apk not in cached_apks]
            if cached_checked
            else []
        )
        setup_cache_ready = (
            not missing_apks if spec.apk_names and cached_checked else None
        )
        row = {
            **spec.to_dict(),
            "installed": installed,
            "missing_apks": missing_apks,
            "setup_cache_ready": setup_cache_ready,
            "replay_ready": installed if installed_checked else None,
        }
        app_rows.append(row)
        for task in spec.tasks:
            task_to_apps.setdefault(task, []).append(spec.app_name)

    app_by_name = {row["app_name"]: row for row in app_rows}
    task_rows: list[dict[str, Any]] = []
    for task in sorted(task_to_apps):
        app_names = sorted(task_to_apps[task])
        missing_packages = [
            name for name in app_names if app_by_name[name].get("installed") is False
        ]
        missing_apks = sorted(
            {
                apk
                for name in app_names
                for apk in list(app_by_name[name].get("missing_apks") or [])
            }
        )
        task_rows.append(
            {
                "task": task,
                "apps": app_names,
                "missing_packages": missing_packages,
                "missing_apks": missing_apks,
                "replay_ready": (not missing_packages if installed_checked else None),
            }
        )

    missing_package_rows = [
        row
        for row in app_rows
        if row.get("installed") is False and row.get("package_name")
    ]
    missing_cached_apks = sorted(
        {apk for row in app_rows for apk in list(row.get("missing_apks") or [])}
    )
    ready_task_count = sum(1 for row in task_rows if row.get("replay_ready") is True)
    return {
        "schema_version": "omniflow.androidworld_replay_device_readiness.v1",
        "installed_checked": installed_checked,
        "apk_cache_checked": cached_checked,
        "apk_cache_dir": str(cache_path),
        "app_count": len(app_rows),
        "task_count": len(task_rows),
        "ready_task_count": ready_task_count if installed_checked else None,
        "missing_package_count": len(missing_package_rows)
        if installed_checked
        else None,
        "missing_cached_apk_count": len(missing_cached_apks)
        if cached_checked
        else None,
        "missing_packages": [
            {
                "app_name": row["app_name"],
                "package_name": row["package_name"],
                "tasks": row["tasks"],
            }
            for row in missing_package_rows
        ],
        "missing_cached_apks": missing_cached_apks,
        "apps": app_rows,
        "tasks": task_rows,
    }


def materialize_replay_run_log(
    item: ArchivedRunLog,
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
        _repo_path(output_root, repo_root=repo_root) / "_normalized_runlogs"
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
    replay_dir = _repo_path(output_root, repo_root=repo_root) / "_replay_runlogs"
    replay_dir.mkdir(parents=True, exist_ok=True)
    source = _repo_path(source_run_log, repo_root=repo_root)
    destination = replay_dir / f"{_safe_stem(task)}.run_log.json"
    if source.resolve() != destination.resolve():
        destination.write_bytes(source.read_bytes())
    return destination.resolve()


def _materialize_replay_run_log_for_memory(
    item: ArchivedRunLog,
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
    item: ArchivedRunLog,
    *,
    output_root: str | Path | None = None,
    repo_root: Path = REPO_ROOT,
    write_materialized: bool = False,
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
        or f"androidworld_archive_{_safe_stem(item.task)}"
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
            "source": str(canonical.get("source") or "androidworld_archive_source"),
            "androidworld_task": str(canonical.get("androidworld_task") or item.task),
            "androidworld_params": dict(
                canonical.get("androidworld_params")
                if isinstance(canonical.get("androidworld_params"), dict)
                else item.params
            ),
        }
    )

    materialized_path: Path | None = None
    if write_materialized:
        if output_root is None:
            raise ValueError("output_root is required when write_materialized=True")
        materialized_dir = (
            _repo_path(output_root, repo_root=repo_root) / "_canonical_source_runlogs"
        )
        materialized_dir.mkdir(parents=True, exist_ok=True)
        materialized_path = materialized_dir / f"{_safe_stem(item.task)}.run_log.json"
        materialized_path.write_text(
            json.dumps(canonical, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        materialized_path = materialized_path.resolve()
    return canonical, materialization, profile, materialized_path


def build_fixed_replay_command(
    item: ArchivedRunLog,
    *,
    android_world_root: str | Path = DEFAULT_ANDROID_WORLD_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    replay_memory_root: str | Path | None = None,
    method_name: str = "fixed_replay",
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
    resolved_method = _safe_stem(method_name, fallback="fixed_replay")
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
        "src.integrations.android_world.launch",
        "--android-world-root",
        str(_repo_path(android_world_root, repo_root=repo_root)),
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
            "replay_run_log": str(replay_run_log),
            "memory_root": str(_repo_path(replay_memory_root, repo_root=repo_root))
            if replay_memory_root
            else "",
            "source_materialization": source_materialization,
            "source_format": profile.replay_format,
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
            "execution_backend": "raw_coordinate_replay",
            "uses_action_transfer": False,
            "uses_source_xml": False,
            "uses_vlm_fallback": False,
        },
    )


def build_official_androidworld_command(
    item: ArchivedRunLog,
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
    max_steps: int = 20,
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
        "src.integrations.android_world.launch",
        "--android-world-root",
        str(_repo_path(android_world_root, repo_root=repo_root)),
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
                str(_repo_path(source_action_hint_path, repo_root=repo_root)),
            ]
        )
    if adb_path.strip():
        argv.extend(["--adb-path", adb_path.strip()])
    hint_path_text = (
        str(_repo_path(source_action_hint_path, repo_root=repo_root))
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


def build_e2e_command(
    item: ArchivedRunLog,
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
    max_steps: int = 20,
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
    resolved_store_path = (
        _repo_path(store_path, repo_root=repo_root)
        if store_path
        else _default_e2e_store_path(output_root, repo_root=repo_root)
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
        env["OMNIFLOW_MAX_FALLBACK_STEPS"] = str(max(0, int(max_fallback_steps)))
    resolved_omnitransfer_root = (
        _repo_path(omnitransfer_root, repo_root=repo_root)
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
        "src.integrations.android_world.launch",
        "--android-world-root",
        str(_repo_path(android_world_root, repo_root=repo_root)),
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
        if planner_provider.strip():
            argv.extend(["--planner-provider", planner_provider.strip()])
        if model.strip():
            argv.extend(["--model", model.strip()])
        if planner_timeout_sec is not None and float(planner_timeout_sec) > 0:
            argv.extend(["--planner-timeout-sec", str(float(planner_timeout_sec))])
    if adb_path.strip():
        argv.extend(["--adb-path", adb_path.strip()])
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
            "mode": "normal_omniflow_e2e"
            if resolved_agent == "omniflow"
            else "normal_androidworld_episode",
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
            "state_backend": "androidworld",
            "action_backend": "androidworld",
            "native_androidworld_agent_io": True,
            "include_indexed_context": False,
            "uses_action_transfer": True,
        },
    )


def validate_ours_transfer_assets(
    store_path: str | Path,
    *,
    require_action_transfer: bool = True,
) -> dict[str, Any]:
    from omniflow.functions.store import FunctionStore
    from omniflow.transfer.runtime import (
        TRANSFER_STATE_CATALOG_FILENAME,
        audit_transfer_action_sources,
        load_transfer_state_catalog,
        transfer_state_coverage,
    )

    resolved_store_path = _repo_path(store_path)
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


def run_commands_parallel(
    specs: Sequence[CommandSpec],
    *,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Run command specs concurrently and return command records."""

    records: list[dict[str, Any]] = []
    processes: list[tuple[CommandSpec, subprocess.Popen[Any]]] = []
    for spec in specs:
        command = _command_line(spec)
        print(f"[{spec.label}] {command}", flush=True)
        if dry_run:
            records.append(
                {
                    "label": spec.label,
                    "returncode": 0,
                    "output_path": str(spec.output_path or ""),
                    "command": command,
                    "metadata": dict(spec.metadata),
                }
            )
            continue
        env = _subprocess_env(spec.env)
        processes.append(
            (
                spec,
                subprocess.Popen(
                    spec.argv,
                    cwd=spec.cwd,
                    env=env,
                ),
            )
        )

    for spec, process in processes:
        returncode = int(process.wait())
        records.append(
            {
                "label": spec.label,
                "returncode": returncode,
                "output_path": str(spec.output_path or ""),
                "command": _command_line(spec),
                "metadata": dict(spec.metadata),
            }
        )
    return records


def _patch_mobilegpt_server_runtime_context(
    *,
    mobilegpt_root: str | Path = DEFAULT_MOBILEGPT_ROOT,
    repo_root: Path = REPO_ROOT,
) -> list[Path]:
    root = _repo_path(mobilegpt_root, repo_root=repo_root)
    server_py = root / "Server" / "server.py"
    if not server_py.exists():
        raise FileNotFoundError(f"MobileGPT server.py not found: {server_py}")

    patched_paths: list[Path] = []
    text = server_py.read_text(encoding="utf-8")
    original = text
    if "MOBILEGPT_TARGET_PACKAGE" not in text:
        text = text.replace(
            """                target_app = task['app']
                if target_app == 'unknown' or target_app == "":
                    target_app = app_agent.predict_app(instruction)
                    task['app'] = target_app

                target_package = app_agent.get_package_name(target_app)
""",
            """                target_app = task['app']
                if target_app == 'unknown' or target_app == "":
                    target_app = app_agent.predict_app(instruction)
                    task['app'] = target_app

                forced_target_app = os.getenv("MOBILEGPT_TARGET_APP", "").strip()
                forced_target_package = os.getenv("MOBILEGPT_TARGET_PACKAGE", "").strip()
                if forced_target_app:
                    target_app = forced_target_app
                    task['app'] = forced_target_app

                target_package = forced_target_package or app_agent.get_package_name(target_app)
""",
        )
    if "MOBILEGPT_CURRENT_LOG_DIRECTORY" not in text:
        text = text.replace(
            """                log_directory += f'/log/{target_app}/{task["name"]}/{dt_string}/'
                screen_parser.init(log_directory)
""",
            """                log_directory += f'/log/{target_app}/{task["name"]}/{dt_string}/'
                os.environ["MOBILEGPT_CURRENT_LOG_DIRECTORY"] = log_directory
                screen_parser.init(log_directory)
""",
        )
    if "MOBILEGPT_CURRENT_SCREEN_INDEX" not in text:
        text = text.replace(
            """            elif message_type == 'X':
                raw_xml = self.__recv_xml(client_socket, screen_count, log_directory)

                parsed_xml, hierarchy_xml, encoded_xml = screen_parser.encode(raw_xml, screen_count)
                screen_count += 1
""",
            """            elif message_type == 'X':
                current_screen_index = screen_count
                raw_xml = self.__recv_xml(client_socket, current_screen_index, log_directory)

                parsed_xml, hierarchy_xml, encoded_xml = screen_parser.encode(raw_xml, current_screen_index)
                os.environ["MOBILEGPT_CURRENT_LOG_DIRECTORY"] = log_directory
                os.environ["MOBILEGPT_CURRENT_SCREEN_INDEX"] = str(current_screen_index)
                screen_count += 1
""",
        )

    if text != original:
        server_py.write_text(text, encoding="utf-8")
        patched_paths.append(server_py.resolve())

    task_agent_py = root / "Server" / "agents" / "task_agent.py"
    if task_agent_py.exists():
        task_agent_text = task_agent_py.read_text(encoding="utf-8")
        patched_task_agent_text = task_agent_text.replace(
            'self.database_path = f"./memory/tasks.csv"',
            "self.database_path = os.path.join("
            'os.getenv("MOBILEGPT_MEMORY_ROOT", "./memory"), "tasks.csv")',
        )
        if patched_task_agent_text != task_agent_text:
            task_agent_py.write_text(patched_task_agent_text, encoding="utf-8")
            patched_paths.append(task_agent_py.resolve())

    mobilegpt_py = root / "Server" / "mobilegpt.py"
    if mobilegpt_py.exists():
        mobilegpt_text = mobilegpt_py.read_text(encoding="utf-8")
        patched_mobilegpt_text = mobilegpt_text.replace(
            'global_task_database_path = f"./memory/tasks.csv"',
            "global_task_database_path = os.path.join("
            'os.getenv("MOBILEGPT_MEMORY_ROOT", "./memory"), "tasks.csv")',
        )
        if patched_mobilegpt_text != mobilegpt_text:
            mobilegpt_py.write_text(patched_mobilegpt_text, encoding="utf-8")
            patched_paths.append(mobilegpt_py.resolve())

    return patched_paths


def _patch_mobilegpt_stats(
    *,
    mobilegpt_root: str | Path = DEFAULT_MOBILEGPT_ROOT,
    repo_root: Path = REPO_ROOT,
) -> list[Path]:
    root = _repo_path(mobilegpt_root, repo_root=repo_root)
    utils_py = root / "Server" / "utils" / "utils.py"
    mobilegpt_py = root / "Server" / "mobilegpt.py"
    server_py = root / "Server" / "server.py"
    if not utils_py.exists():
        raise FileNotFoundError(f"MobileGPT utils.py not found: {utils_py}")
    if not mobilegpt_py.exists():
        raise FileNotFoundError(f"MobileGPT mobilegpt.py not found: {mobilegpt_py}")

    utils_text = utils_py.read_text(encoding="utf-8")
    if "def _omniflow_write_stats_event(" not in utils_text:
        marker = "from ast import literal_eval\n"
        helper = """
from datetime import datetime, timezone
import time


def _omniflow_write_stats_event(event: dict):
    path = os.getenv("MOBILEGPT_STATS_JSONL")
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = dict(event)
        payload.setdefault("ts", datetime.now(timezone.utc).isoformat())
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\\n")
    except Exception:
        pass


def write_omniflow_mobilegpt_event(event: dict):
    _omniflow_write_stats_event(event)

"""
        utils_text = utils_text.replace(marker, marker + helper, 1)
    if (
        "def get_openai_embedding(" in utils_text
        and 'event": "embedding_call"' not in utils_text
    ):
        utils_text = re.sub(
            r"(def get_openai_embedding\([^\n]*\)[^\n]*:\n)",
            r"\1    _omniflow_started = time.time()\n",
            utils_text,
            count=1,
        )
        utils_text = re.sub(
            r"(    response = client\.embeddings\.create\(input=\[text\], model=model, \*\*kwargs\)\n)(\n?    return response\.data\[0\]\.embedding\n)",
            "\\1"
            '    usage = getattr(response, "usage", None)\n'
            "    _omniflow_write_stats_event({\n"
            '        "event": "embedding_call",\n'
            '        "model": model,\n'
            '        "latency_sec": round(time.time() - _omniflow_started, 6),\n'
            '        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),\n'
            '        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),\n'
            "    })\n"
            "\\2",
            utils_text,
            count=1,
        )
    embedding_function_match = re.search(
        r"def get_openai_embedding\([^\n]*\)[^\n]*:\n(?P<body>.*?)(?=\ndef |\Z)",
        utils_text,
        flags=re.DOTALL,
    )
    if (
        embedding_function_match
        and 'event": "embedding_call"' in embedding_function_match.group("body")
        and "_omniflow_started = time.time()"
        not in embedding_function_match.group("body")
    ):
        utils_text = re.sub(
            r"(def get_openai_embedding\([^\n]*\)[^\n]*:\n)",
            r"\1    _omniflow_started = time.time()\n",
            utils_text,
            count=1,
        )
    if "def query(" in utils_text and 'event": "chat_call"' not in utils_text:
        utils_text = re.sub(
            r"(def query\([^\n]*\):\n)",
            r"\1    _omniflow_started = time.time()\n",
            utils_text,
            count=1,
        )
        utils_text = utils_text.replace(
            "    result = response.choices[0].message.content\n"
            "    log(result, 'green')\n",
            '    usage = getattr(response, "usage", None)\n'
            "    _omniflow_write_stats_event({\n"
            '        "event": "chat_call",\n'
            '        "model": model,\n'
            '        "latency_sec": round(time.time() - _omniflow_started, 6),\n'
            '        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),\n'
            '        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),\n'
            '        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),\n'
            "    })\n"
            "    result = response.choices[0].message.content\n"
            "    log(result, 'green')\n",
            1,
        )
    utils_py.write_text(utils_text, encoding="utf-8")

    mobilegpt_text = mobilegpt_py.read_text(encoding="utf-8")
    if "write_omniflow_mobilegpt_event" not in mobilegpt_text:
        mobilegpt_text = mobilegpt_text.replace(
            "from utils.utils import log, parse_completion_rate\n",
            "from utils.utils import log, parse_completion_rate, write_omniflow_mobilegpt_event\n",
            1,
        )
    if '"event": "task_started"' not in mobilegpt_text:
        mobilegpt_text = mobilegpt_text.replace(
            "        log('Mobile Agent Initialized for app: ' + task['app'] + ' / Task: ' + task['name'])\n",
            "        log('Mobile Agent Initialized for app: ' + task['app'] + ' / Task: ' + task['name'])\n"
            "        write_omniflow_mobilegpt_event({\n"
            '            "event": "task_started",\n'
            '            "instruction": instruction,\n'
            "            \"app\": task.get('app'),\n"
            "            \"task_name\": task.get('name'),\n"
            "        })\n",
            1,
        )
    if '"event": "task_finished"' not in mobilegpt_text:
        mobilegpt_text = mobilegpt_text.replace(
            '        log(f"""Completed the execution of “{self.instruction}” you commanded, and the Task took a total of [{minutes} minutes({seconds} seconds)] to run.""", "green")\n',
            '        log(f"""Completed the execution of “{self.instruction}” you commanded, and the Task took a total of [{minutes} minutes({seconds} seconds)] to run.""", "green")\n'
            "        write_omniflow_mobilegpt_event({\n"
            '            "event": "task_finished",\n'
            '            "instruction": self.instruction,\n'
            '            "elapsed_sec": round(elapsed_time, 6),\n'
            '            "task_status": str(self.task_status),\n'
            '            "subtask_count": len(self.task_path),\n'
            "        })\n",
            1,
        )
    if '"event": "memory_lookup"' not in mobilegpt_text:
        mobilegpt_text = mobilegpt_text.replace(
            "        next_action = self.memory.get_next_action(self.current_subtask, self.encoded_xml)\n",
            "        next_action = self.memory.get_next_action(self.current_subtask, self.encoded_xml)\n"
            "        write_omniflow_mobilegpt_event({\n"
            '            "event": "memory_lookup",\n'
            '            "result": ("in_context_fallback"\n'
            '                       if next_action and "examples" in next_action\n'
            '                       else "direct_hit" if next_action\n'
            '                       else "derive_fallback"),\n'
            "        })\n",
            1,
        )
    mobilegpt_py.write_text(mobilegpt_text, encoding="utf-8")

    patched_paths = [utils_py, mobilegpt_py]
    if server_py.is_file():
        server_text = server_py.read_text(encoding="utf-8")
        if "write_omniflow_mobilegpt_event" not in server_text:
            server_text = server_text.replace(
                "from utils.utils import log\n",
                "from utils.utils import log, write_omniflow_mobilegpt_event\n",
                1,
            )
        if '"event": "mobilegpt_action_sent"' not in server_text:
            server_text = server_text.replace(
                "def _omniflow_send_action(client_socket, action):\n",
                "def _omniflow_send_action(client_socket, action):\n"
                '    action_name = (str(action.get("name") or "").strip()\n'
                '                   if isinstance(action, dict) else "")\n'
                "    write_omniflow_mobilegpt_event({\n"
                '        "event": "mobilegpt_action_sent",\n'
                '        "action_name": action_name,\n'
                '        "is_device_action": action_name not in {\n'
                '            "", "finish", "speak", "ask", "read_screen",\n'
                "            OMNIFLOW_INTERNAL_LAUNCH_ACTION,\n"
                "        },\n"
                "    })\n",
                1,
            )
        server_py.write_text(server_text, encoding="utf-8")
        patched_paths.append(server_py)
    return patched_paths


def summarize_mobilegpt_stats(path: str | Path) -> dict[str, Any]:
    stats_path = _repo_path(path)
    rows = list(_iter_jsonl_rows(stats_path))
    chat_rows = [row for row in rows if row.get("event") == "chat_call"]
    embedding_rows = [row for row in rows if row.get("event") == "embedding_call"]
    finished_rows = [row for row in rows if row.get("event") == "task_finished"]
    started_rows = [row for row in rows if row.get("event") == "task_started"]
    teacher_rows = [
        row
        for row in rows
        if str(row.get("event") or "").startswith("mobilegpt_teacher_")
    ]
    teacher_preflight_rows = [
        row
        for row in teacher_rows
        if row.get("event") == "mobilegpt_teacher_source_preflight"
    ]
    teacher_miss_rows = [
        row for row in teacher_rows if row.get("event") == "mobilegpt_teacher_miss"
    ]
    memory_rows = [row for row in rows if row.get("event") == "memory_lookup"]
    action_rows = [row for row in rows if row.get("event") == "mobilegpt_action_sent"]
    device_action_rows = [
        row for row in action_rows if row.get("is_device_action") is True
    ]
    teacher_expected_action_count = max(
        [
            _coerce_int(row.get("teacher_action_count"))
            for row in teacher_rows
            if row.get("event") == "mobilegpt_teacher_source_preflight"
        ]
        or [0]
    )
    teacher_groundable_action_count = max(
        [
            _coerce_int(
                row.get("groundable_action_count"),
                _coerce_int(row.get("teacher_action_count")),
            )
            for row in teacher_preflight_rows
        ]
        or [teacher_expected_action_count]
    )
    teacher_action_count = sum(
        1 for row in teacher_rows if row.get("event") == "mobilegpt_teacher_action"
    )
    teacher_skipped_noop_count = sum(
        _coerce_int(row.get("skipped_count"))
        for row in teacher_rows
        if row.get("event") == "mobilegpt_teacher_skipped_noop"
    )
    teacher_vlm_fallback_count = sum(
        1 for row in teacher_miss_rows if row.get("fallback_to_vlm") is True
    )
    teacher_unrecovered_miss_count = sum(
        1 for row in teacher_miss_rows if row.get("fallback_to_vlm") is not True
    )
    teacher_vlm_fallback_enabled = any(
        row.get("fallback_to_vlm_on_teacher_miss") is True
        for row in teacher_preflight_rows
    )
    native_vlm_fallback_only = any(
        row.get("native_vlm_fallback_only") is True
        for row in teacher_preflight_rows
    )
    memory_hit_count = sum(
        1 for row in memory_rows if row.get("result") == "direct_hit"
    )
    in_context_fallback_count = sum(
        1 for row in memory_rows if row.get("result") == "in_context_fallback"
    )
    derive_fallback_count = sum(
        1 for row in memory_rows if row.get("result") == "derive_fallback"
    )
    prompt_tokens = sum(_coerce_int(row.get("prompt_tokens")) for row in rows)
    completion_tokens = sum(_coerce_int(row.get("completion_tokens")) for row in rows)
    total_tokens = sum(_coerce_int(row.get("total_tokens")) for row in rows)
    model_calls = len(chat_rows) + len(embedding_rows)
    return {
        "schema_version": "omniflow.mobilegpt_stats_summary.v1",
        "stats_path": str(stats_path),
        "event_count": len(rows),
        "task_started_count": len(started_rows),
        "task_finished_count": len(finished_rows),
        "teacher_event_count": len(teacher_rows),
        "teacher_action_count": teacher_action_count,
        "teacher_expected_action_count": teacher_expected_action_count,
        "teacher_groundable_action_count": teacher_groundable_action_count,
        "teacher_consumed_action_count": (
            teacher_action_count
            + teacher_skipped_noop_count
            + teacher_vlm_fallback_count
        ),
        "teacher_skipped_noop_count": teacher_skipped_noop_count,
        "teacher_miss_count": len(teacher_miss_rows),
        "teacher_vlm_fallback_count": teacher_vlm_fallback_count,
        "teacher_unrecovered_miss_count": teacher_unrecovered_miss_count,
        "teacher_vlm_fallback_enabled": teacher_vlm_fallback_enabled,
        "native_vlm_fallback_only": native_vlm_fallback_only,
        "teacher_failed_finish_count": sum(
            1
            for row in teacher_rows
            if row.get("event") == "mobilegpt_teacher_failed_finish"
        ),
        "teacher_forced_select_count": sum(
            1
            for row in teacher_rows
            if row.get("event") == "mobilegpt_teacher_forced_select"
        ),
        "teacher_action_error_count": sum(
            1
            for row in teacher_rows
            if row.get("event") == "mobilegpt_teacher_action_error"
        ),
        "chat_model_calls": len(chat_rows),
        "embedding_model_calls": len(embedding_rows),
        "model_calls": model_calls,
        "chat_models": sorted(
            {
                str(row.get("model") or "").strip()
                for row in chat_rows
                if str(row.get("model") or "").strip()
            }
        ),
        "embedding_models": sorted(
            {
                str(row.get("model") or "").strip()
                for row in embedding_rows
                if str(row.get("model") or "").strip()
            }
        ),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "token_usage_status": (
            "tracked"
            if model_calls > 0
            and total_tokens > 0
            and total_tokens == prompt_tokens + completion_tokens
            else "inconsistent"
            if model_calls > 0
            else "not_applicable"
        ),
        "chat_latency_sec": round(
            sum(_coerce_float(row.get("latency_sec")) for row in chat_rows),
            6,
        ),
        "embedding_latency_sec": round(
            sum(_coerce_float(row.get("latency_sec")) for row in embedding_rows),
            6,
        ),
        "task_elapsed_sec": round(
            sum(_coerce_float(row.get("elapsed_sec")) for row in finished_rows),
            6,
        ),
        "finished_tasks": [
            {
                "instruction": row.get("instruction"),
                "elapsed_sec": _coerce_float(row.get("elapsed_sec")),
                "subtask_count": _coerce_int(row.get("subtask_count")),
            }
            for row in finished_rows
        ],
        "memory_lookup_count": len(memory_rows),
        "memory_hit_count": memory_hit_count,
        "memory_hit_rate": _rate(memory_hit_count, len(memory_rows)),
        "fallback_count": in_context_fallback_count + derive_fallback_count,
        "in_context_fallback_count": in_context_fallback_count,
        "derive_fallback_count": derive_fallback_count,
        "action_sent_count": len(action_rows),
        "actions_executed": len(device_action_rows),
        "action_name_counts": dict(
            Counter(str(row.get("action_name") or "") for row in action_rows)
        ),
    }


def _load_mobilegpt_stats_summary(
    *,
    summary_path: str | Path | None,
    stats_jsonl_path: str | Path | None,
) -> dict[str, Any]:
    summary_value = str(summary_path or "").strip()
    stats_value = str(stats_jsonl_path or "").strip()
    resolved_summary_path = _repo_path(summary_value) if summary_value else None
    resolved_stats_path = _repo_path(stats_value) if stats_value else None
    stats: dict[str, Any] = {}
    if resolved_summary_path is not None and resolved_summary_path.is_file():
        try:
            loaded = json.loads(resolved_summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            loaded = {}
        if isinstance(loaded, dict):
            stats = loaded
    elif resolved_stats_path is not None and resolved_stats_path.is_file():
        stats = summarize_mobilegpt_stats(resolved_stats_path)
    if not stats:
        return {}
    stats.setdefault("stats_path", str(resolved_stats_path or ""))
    stats["summary_path"] = str(resolved_summary_path or "")
    return stats


def _mobilegpt_stats_row_fields(
    prefix: str,
    stats: dict[str, Any],
) -> dict[str, Any]:
    return {
        f"{prefix}_model_calls": _coerce_int(stats.get("model_calls")),
        f"{prefix}_chat_model_calls": _coerce_int(stats.get("chat_model_calls")),
        f"{prefix}_embedding_model_calls": _coerce_int(
            stats.get("embedding_model_calls")
        ),
        f"{prefix}_chat_models": list(stats.get("chat_models") or []),
        f"{prefix}_embedding_models": list(stats.get("embedding_models") or []),
        f"{prefix}_prompt_tokens": _coerce_int(stats.get("prompt_tokens")),
        f"{prefix}_completion_tokens": _coerce_int(stats.get("completion_tokens")),
        f"{prefix}_total_tokens": _coerce_int(stats.get("total_tokens")),
        f"{prefix}_token_usage_status": str(stats.get("token_usage_status") or ""),
        f"{prefix}_chat_latency_sec": _coerce_float(stats.get("chat_latency_sec")),
        f"{prefix}_embedding_latency_sec": _coerce_float(
            stats.get("embedding_latency_sec")
        ),
        f"{prefix}_task_elapsed_sec": _coerce_float(stats.get("task_elapsed_sec")),
        f"{prefix}_task_started_count": _coerce_int(stats.get("task_started_count")),
        f"{prefix}_task_finished_count": _coerce_int(stats.get("task_finished_count")),
        f"{prefix}_memory_lookup_count": _coerce_int(stats.get("memory_lookup_count")),
        f"{prefix}_memory_hit_count": _coerce_int(stats.get("memory_hit_count")),
        f"{prefix}_memory_hit_rate": _coerce_float(stats.get("memory_hit_rate")),
        f"{prefix}_fallback_count": _coerce_int(stats.get("fallback_count")),
        f"{prefix}_in_context_fallback_count": _coerce_int(
            stats.get("in_context_fallback_count")
        ),
        f"{prefix}_derive_fallback_count": _coerce_int(
            stats.get("derive_fallback_count")
        ),
        f"{prefix}_action_sent_count": _coerce_int(stats.get("action_sent_count")),
        f"{prefix}_actions_executed": _coerce_int(stats.get("actions_executed")),
        f"{prefix}_action_name_counts": dict(stats.get("action_name_counts") or {}),
        f"{prefix}_event_count": _coerce_int(stats.get("event_count")),
        f"{prefix}_stats_jsonl": str(stats.get("stats_path") or ""),
        f"{prefix}_stats_summary": str(stats.get("summary_path") or ""),
    }


def inspect_mobilegpt_memory(memory_root: str | Path) -> dict[str, Any]:
    root = _repo_path(memory_root)
    task_files = sorted(root.glob("*/tasks.csv")) if root.exists() else []
    subtask_files = sorted(root.glob("*/pages/*/subtasks.csv")) if root.exists() else []
    action_files = sorted(root.glob("*/pages/*/actions.csv")) if root.exists() else []
    task_rows = 0
    subtask_rows = 0
    action_rows = 0
    non_finish_action_rows = 0
    action_file_rows: list[dict[str, Any]] = []

    for task_file in task_files:
        try:
            with task_file.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except Exception:
            rows = []
        task_rows += len(rows)

    for subtask_file in subtask_files:
        try:
            with subtask_file.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except Exception:
            rows = []
        subtask_rows += len(rows)

    for action_file in action_files:
        file_action_rows = 0
        file_non_finish_rows = 0
        try:
            with action_file.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except Exception:
            rows = []
        for row in rows:
            if not any(str(value or "").strip() for value in row.values()):
                continue
            file_action_rows += 1
            action_text = str(row.get("action") or "")
            if (
                '"name": "finish"' not in action_text
                and "'name': 'finish'" not in action_text
            ):
                file_non_finish_rows += 1
        action_rows += file_action_rows
        non_finish_action_rows += file_non_finish_rows
        if file_action_rows:
            action_file_rows.append(
                {
                    "path": str(action_file),
                    "action_rows": file_action_rows,
                    "non_finish_action_rows": file_non_finish_rows,
                }
            )

    return {
        "memory_root": str(root),
        "task_file_count": len(task_files),
        "task_rows": task_rows,
        "subtask_file_count": len(subtask_files),
        "subtask_rows": subtask_rows,
        "action_file_count": len(action_files),
        "action_rows": action_rows,
        "non_finish_action_rows": non_finish_action_rows,
        "has_recallable_subtasks": subtask_rows > 0,
        "has_useful_actions": non_finish_action_rows > 0 and subtask_rows > 0,
        "action_files": action_file_rows,
    }


def _mobilegpt_memory_digest(memory_root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    file_count = 0
    for path in sorted(memory_root.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(memory_root).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
        file_count += 1
    return digest.hexdigest(), file_count


MOBILEGPT_COLD_MEMORY_SCHEMA = "omniflow.mobilegpt-cold-memory.v1"


def _mobilegpt_manifest_evidence_path(
    bundle_root: Path,
    record: Any,
    *,
    label: str,
) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"mobilegpt_cold_memory_{label}_record_invalid")
    relative_value = str(record.get("relative_path") or "").strip()
    if not relative_value:
        raise ValueError(f"mobilegpt_cold_memory_{label}_path_missing")
    path = (bundle_root / relative_value).resolve()
    try:
        path.relative_to(bundle_root)
    except ValueError as error:
        raise ValueError(f"mobilegpt_cold_memory_{label}_outside_bundle") from error
    if not path.is_file():
        raise ValueError(f"mobilegpt_cold_memory_{label}_missing:{path}")
    expected_sha256 = str(record.get("sha256") or "").strip()
    if not expected_sha256 or _file_sha256(path) != expected_sha256:
        raise ValueError(f"mobilegpt_cold_memory_{label}_hash_mismatch")
    return path


def validate_mobilegpt_adapted_memory(
    memory_root: str | Path,
    *,
    task_name: str,
    source_seed: int,
    source_run_log: str | Path,
    expected_model: str = "",
    expected_source_method: str = "",
) -> dict[str, Any]:
    """Validate one sealed native MobileGPT source-cold memory tree."""

    root = _repo_path(memory_root)
    if not root.is_dir():
        raise FileNotFoundError(f"mobilegpt_source_memory_missing:{root}")
    manifest_path = root.parent / "cold_memory_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"cold_memory_manifest_missing:{manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("mobilegpt_cold_memory_manifest_invalid_json") from error
    if not isinstance(manifest, dict):
        raise ValueError("mobilegpt_cold_memory_manifest_invalid")
    if manifest.get("schema_version") != MOBILEGPT_COLD_MEMORY_SCHEMA:
        raise ValueError("mobilegpt_cold_memory_manifest_schema_invalid")
    if str(manifest.get("task_name") or "") != str(task_name):
        raise ValueError("mobilegpt_cold_memory_task_name_mismatch")
    if int(manifest.get("source_seed") or -1) != int(source_seed):
        raise ValueError("mobilegpt_cold_memory_source_seed_mismatch")
    if int(source_seed) != 111:
        raise ValueError("mobilegpt_cold_memory_requires_source_seed_111")
    normalized_expected_source_method = str(expected_source_method or "").strip()
    if (
        normalized_expected_source_method
        and str(manifest.get("source_method") or "").strip()
        != normalized_expected_source_method
    ):
        raise ValueError(
            "mobilegpt_cold_memory_source_method_mismatch:"
            f"expected={normalized_expected_source_method}:"
            f"actual={str(manifest.get('source_method') or '').strip() or 'missing'}"
        )

    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("mobilegpt_cold_memory_provenance_missing")
    forbidden_true = [
        name
        for name in (
            "function_conversion_enabled",
            "target_inputs_read",
            "target_observations_read",
            "validator_state_read",
            "coordinate_replay",
        )
        if bool(provenance.get(name))
    ]
    if forbidden_true:
        raise ValueError(
            "mobilegpt_cold_memory_provenance_invalid:" + ",".join(forbidden_true)
        )
    if provenance.get("native_mobilegpt_learning") is not True:
        raise ValueError("mobilegpt_cold_memory_native_learning_required")
    if provenance.get("complete_teacher_action_consumption") is not True:
        raise ValueError("mobilegpt_cold_memory_teacher_incomplete")

    memory_record = manifest.get("memory")
    if not isinstance(memory_record, dict):
        raise ValueError("mobilegpt_cold_memory_record_missing")
    expected_memory_path = (
        root.parent / str(memory_record.get("relative_path") or "")
    ).resolve()
    if expected_memory_path != root:
        raise ValueError("mobilegpt_cold_memory_path_mismatch")
    actual_digest, actual_file_count = _mobilegpt_memory_digest(root)
    if actual_digest != str(memory_record.get("sha256") or ""):
        raise ValueError("mobilegpt_cold_memory_hash_mismatch")
    if actual_file_count != int(memory_record.get("file_count") or -1):
        raise ValueError("mobilegpt_cold_memory_file_count_mismatch")
    inventory = inspect_mobilegpt_memory(root)
    if not inventory.get("has_recallable_subtasks"):
        raise ValueError("mobilegpt_cold_memory_missing_recallable_subtasks")
    if not inventory.get("has_useful_actions"):
        raise ValueError("mobilegpt_cold_memory_missing_useful_actions")

    bundle_root = root.parent.resolve()
    teacher_source = manifest.get("teacher_source")
    teacher_source_path = _mobilegpt_manifest_evidence_path(
        bundle_root,
        teacher_source,
        label="teacher_source",
    )
    teacher_payload = json.loads(teacher_source_path.read_text(encoding="utf-8"))
    if (
        not isinstance(teacher_payload, dict)
        or teacher_payload.get("schema_version")
        != "omniflow.mobilegpt-teacher-source.v1"
    ):
        raise ValueError("mobilegpt_cold_memory_teacher_source_invalid")
    if str(teacher_payload.get("task_name") or "") != str(task_name):
        raise ValueError("mobilegpt_cold_memory_teacher_source_task_mismatch")
    if int(teacher_payload.get("source_seed") or -1) != int(source_seed):
        raise ValueError("mobilegpt_cold_memory_teacher_source_seed_mismatch")
    if teacher_payload.get("contains_source_coordinates") is not False:
        raise ValueError("mobilegpt_cold_memory_teacher_coordinates_forbidden")
    source_log_record = manifest.get("source_run_log")
    _mobilegpt_manifest_evidence_path(
        bundle_root,
        source_log_record,
        label="source_run_log",
    )
    expected_source_sha256 = _file_sha256(_repo_path(source_run_log))
    if str(source_log_record.get("sha256") or "") != expected_source_sha256:
        raise ValueError("mobilegpt_cold_memory_source_run_log_mismatch")
    if str(teacher_payload.get("source_run_log_sha256") or "") != (
        expected_source_sha256
    ):
        raise ValueError("mobilegpt_cold_memory_teacher_source_run_log_mismatch")
    source_stats = manifest.get("source_stats")
    source_stats_path = _mobilegpt_manifest_evidence_path(
        bundle_root,
        source_stats,
        label="source_stats",
    )
    stats_summary = summarize_mobilegpt_stats(source_stats_path)
    normalized_expected_model = str(expected_model or "").strip()
    if normalized_expected_model:
        manifest_model = str(manifest.get("source_model") or "").strip()
        if manifest_model != normalized_expected_model:
            raise ValueError(
                "mobilegpt_cold_memory_manifest_model_mismatch:"
                f"expected={normalized_expected_model}:"
                f"actual={manifest_model or 'missing'}"
            )
        source_models = {
            str(model or "").strip()
            for model in stats_summary.get("chat_models") or []
            if str(model or "").strip()
        }
        if source_models != {normalized_expected_model}:
            raise ValueError(
                "mobilegpt_cold_memory_model_mismatch:"
                f"expected={normalized_expected_model}:actual={sorted(source_models)}"
            )
    official_result = manifest.get("official_source_result")
    official_result_path = _mobilegpt_manifest_evidence_path(
        bundle_root,
        official_result,
        label="official_source_result",
    )
    result_summary = _mobilegpt_official_source_result(
        official_result_path,
        task_name=task_name,
    )
    if not result_summary["official_validator_success"]:
        raise ValueError("mobilegpt_cold_memory_official_source_failed")
    write_status = _mobilegpt_memory_write_status(
        stats_summary=stats_summary,
        memory_inventory=inventory,
        cold_validator_success=True,
    )
    if not write_status["memory_written"]:
        raise ValueError(
            "mobilegpt_cold_memory_incomplete:" + ",".join(write_status["reasons"])
        )
    _validate_mobilegpt_teacher_stats(
        teacher_payload,
        stats_summary,
        error_prefix="mobilegpt_cold_memory_teacher_source",
    )
    manifest_count_pairs = {
        "teacher_action_count": write_status["teacher_action_count"],
        "teacher_expected_action_count": write_status["teacher_expected_action_count"],
        "teacher_groundable_action_count": stats_summary[
            "teacher_groundable_action_count"
        ],
        "teacher_consumed_count": write_status["teacher_consumed_action_count"],
        "teacher_vlm_fallback_count": write_status[
            "teacher_vlm_fallback_count"
        ],
        "teacher_unrecovered_miss_count": write_status[
            "teacher_unrecovered_miss_count"
        ],
        "task_finished_count": write_status["task_finished_count"],
    }
    if any(
        int(source_stats.get(key) or 0) != int(value)
        for key, value in manifest_count_pairs.items()
    ):
        raise ValueError("mobilegpt_cold_memory_stats_manifest_mismatch")
    if official_result.get("official_validator_used") is not True or (
        official_result.get("official_validator_success")
        is not result_summary["official_validator_success"]
    ):
        raise ValueError("mobilegpt_cold_memory_result_manifest_mismatch")
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _file_sha256(manifest_path),
        "memory_root": str(root),
        "memory_sha256": actual_digest,
        "memory_file_count": actual_file_count,
        "memory_inventory": inventory,
        "source_stats_summary": stats_summary,
        "source_memory_write_status": write_status,
        "target_package": str(manifest.get("target_package") or ""),
        "target_app": str(manifest.get("target_app") or ""),
    }


def build_mobilegpt_teacher_source(
    source_run_log: str | Path,
    *,
    task_name: str,
    source_seed: int = 111,
    provenance_source_run_log: str | Path | None = None,
    fallback_to_vlm_on_teacher_miss: bool = False,
) -> dict[str, Any]:
    """Build a coordinate-free audit artifact for the native teacher stream."""

    if int(source_seed) != 111:
        raise ValueError("mobilegpt_teacher_source_requires_seed_111")
    source_path = _repo_path(source_run_log)
    provenance_path = (
        _repo_path(provenance_source_run_log)
        if provenance_source_run_log is not None
        else source_path
    )
    if not provenance_path.is_file():
        raise FileNotFoundError(f"mobilegpt_source_run_log_missing:{provenance_path}")
    from src.integrations.mobilegpt_teacher import (
        load_teacher_actions,
        preflight_teacher_source_run_log,
    )

    teacher_preflight = preflight_teacher_source_run_log(source_path)
    teacher_action_count = int(teacher_preflight["teacher_action_count"])
    groundable_action_count = int(teacher_preflight["groundable_action_count"])
    expected_vlm_fallback_action_count = teacher_action_count - groundable_action_count
    native_vlm_fallback_only = (
        teacher_action_count == 0 and fallback_to_vlm_on_teacher_miss
    )
    if expected_vlm_fallback_action_count and not fallback_to_vlm_on_teacher_miss:
        raise ValueError("mobilegpt_teacher_source_has_ungroundable_actions")

    actions: list[dict[str, Any]] = []
    for record in load_teacher_actions(source_path):
        action = dict(record.get("action") or {})
        params = dict(action.get("params") or {})
        source_context = (
            dict(params.get("source_context") or {})
            if isinstance(params.get("source_context"), dict)
            else {}
        )
        element = (
            dict(source_context.get("element") or {})
            if isinstance(source_context.get("element"), dict)
            else {}
        )
        sanitized_params = {
            key: value
            for key, value in params.items()
            if key
            in {
                "text",
                "key",
                "direction",
                "target_description",
                "package_name",
            }
        }
        if element:
            sanitized_params["source_element"] = {
                key: value
                for key, value in element.items()
                if key
                in {
                    "container_anchor",
                    "content_desc",
                    "description",
                    "relation",
                    "resource_id",
                    "role",
                    "text",
                }
            }
        package_name = str(source_context.get("package_name") or "").strip()
        if package_name:
            sanitized_params["source_package"] = package_name
        actions.append(
            {
                "source_step_index": int(record.get("source_step_index") or 0),
                "source_action_index": int(record.get("source_action_index") or 0),
                "primitive": str(action.get("type") or ""),
                "params": sanitized_params,
            }
        )
    if not actions and not native_vlm_fallback_only:
        raise ValueError("mobilegpt_teacher_source_has_no_supported_actions")
    return {
        "schema_version": "omniflow.mobilegpt-teacher-source.v1",
        "task_name": str(task_name),
        "source_seed": int(source_seed),
        "source_run_log": str(provenance_path),
        "source_run_log_sha256": _file_sha256(provenance_path),
        "grounded_teacher_run_log_sha256": _file_sha256(source_path),
        "action_count": len(actions),
        "groundable_action_count": groundable_action_count,
        "expected_vlm_fallback_action_count": expected_vlm_fallback_action_count,
        "fallback_to_vlm_on_teacher_miss": bool(
            fallback_to_vlm_on_teacher_miss
        ),
        "native_vlm_fallback_only": native_vlm_fallback_only,
        "actions": actions,
        "contains_source_coordinates": False,
        "contains_task_or_subtask_semantics": False,
        "target_inputs_read": False,
        "target_observations_read": False,
    }


def _mobilegpt_official_source_result(
    result_path: str | Path,
    *,
    task_name: str,
) -> dict[str, Any]:
    path = _repo_path(result_path)
    rows = list(_iter_jsonl_rows(path))
    task_rows = [
        row
        for row in rows
        if str(row.get("task_name") or row.get("task") or "") == str(task_name)
    ]
    if not task_rows:
        raise ValueError("mobilegpt_source_result_task_missing")
    official_rows = [row for row in task_rows if _official_validator_used(row)]
    if not official_rows:
        raise ValueError("mobilegpt_source_official_validator_required")
    success = any(_official_validator_success(row) for row in official_rows)
    return {
        "row_count": len(task_rows),
        "official_validator_used": True,
        "official_validator_success": success,
    }


def _validate_mobilegpt_teacher_stats(
    teacher_source: dict[str, Any],
    stats_summary: dict[str, Any],
    *,
    error_prefix: str,
) -> tuple[int, bool, bool]:
    action_count = int(teacher_source.get("action_count") or 0)
    groundable_action_count = int(
        teacher_source.get("groundable_action_count") or 0
    )
    fallback_enabled = bool(
        teacher_source.get("fallback_to_vlm_on_teacher_miss")
    )
    native_fallback_only = bool(teacher_source.get("native_vlm_fallback_only"))
    comparisons = (
        (
            action_count,
            int(stats_summary.get("teacher_expected_action_count") or 0),
            "action_count",
        ),
        (
            groundable_action_count,
            int(stats_summary.get("teacher_groundable_action_count") or 0),
            "groundable_count",
        ),
        (
            int(teacher_source.get("expected_vlm_fallback_action_count") or 0),
            action_count - groundable_action_count,
            "fallback_count",
        ),
        (
            fallback_enabled,
            bool(stats_summary.get("teacher_vlm_fallback_enabled")),
            "fallback_policy",
        ),
        (
            native_fallback_only,
            bool(stats_summary.get("native_vlm_fallback_only")),
            "fallback_mode",
        ),
    )
    for expected, actual, label in comparisons:
        if expected != actual:
            raise ValueError(f"{error_prefix}_{label}_mismatch")
    return groundable_action_count, fallback_enabled, native_fallback_only


def seal_mobilegpt_adapted_memory(
    *,
    memory_root: str | Path,
    teacher_source: str | Path,
    source_run_log: str | Path,
    source_stats: str | Path,
    official_source_result: str | Path,
    task_name: str,
    source_seed: int = 111,
    target_package: str = "",
    target_app: str = "",
    source_wall_sec: float = 0.0,
    source_method: str = "",
    source_model: str = "",
) -> dict[str, Any]:
    """Seal a successful native MobileGPT source-cold episode for warm recall."""

    if int(source_seed) != 111:
        raise ValueError("mobilegpt_cold_memory_requires_source_seed_111")
    memory = _repo_path(memory_root)
    bundle_root = memory.parent.resolve()
    if memory.name != "memory":
        raise ValueError("mobilegpt_cold_memory_directory_must_be_named_memory")
    manifest_path = bundle_root / "cold_memory_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"immutable_cold_memory_manifest_exists:{manifest_path}")

    teacher_path = _repo_path(teacher_source)
    teacher = json.loads(teacher_path.read_text(encoding="utf-8"))
    if not isinstance(teacher, dict) or teacher.get("schema_version") != (
        "omniflow.mobilegpt-teacher-source.v1"
    ):
        raise ValueError("mobilegpt_teacher_source_schema_invalid")
    if str(teacher.get("task_name") or "") != str(task_name):
        raise ValueError("mobilegpt_teacher_source_task_mismatch")
    if int(teacher.get("source_seed") or -1) != int(source_seed):
        raise ValueError("mobilegpt_teacher_source_seed_mismatch")
    if teacher.get("contains_source_coordinates") is not False:
        raise ValueError("mobilegpt_teacher_source_coordinates_forbidden")

    source_path = _repo_path(source_run_log)
    source_sha256 = _file_sha256(source_path)
    if str(teacher.get("source_run_log_sha256") or "") != source_sha256:
        raise ValueError("mobilegpt_teacher_source_run_log_mismatch")
    stats_path = _repo_path(source_stats)
    stats_summary = summarize_mobilegpt_stats(stats_path)
    result_path = _repo_path(official_source_result)
    result_summary = _mobilegpt_official_source_result(
        result_path,
        task_name=task_name,
    )
    if not result_summary["official_validator_success"]:
        raise ValueError("mobilegpt_cold_memory_official_source_failed")
    inventory = inspect_mobilegpt_memory(memory)
    write_status = _mobilegpt_memory_write_status(
        stats_summary=stats_summary,
        memory_inventory=inventory,
        cold_validator_success=True,
    )
    if not write_status["memory_written"]:
        raise ValueError(
            "mobilegpt_cold_memory_incomplete:" + ",".join(write_status["reasons"])
        )
    (
        teacher_groundable_action_count,
        teacher_fallback_enabled,
        teacher_native_vlm_fallback_only,
    ) = _validate_mobilegpt_teacher_stats(
        teacher,
        stats_summary,
        error_prefix="mobilegpt_teacher_source",
    )

    provenance_root = bundle_root / "provenance"
    provenance_root.mkdir(exist_ok=False)

    def copy_evidence(source: Path, name: str) -> Path:
        destination = provenance_root / name
        shutil.copy2(source, destination)
        return destination

    copied_source = copy_evidence(source_path, "source.run_log.json")
    copied_stats = copy_evidence(stats_path, "mobilegpt_stats.jsonl")
    copied_result = copy_evidence(result_path, "task_results.jsonl")
    if teacher_path.parent.resolve() != bundle_root:
        copied_teacher = bundle_root / "teacher_source.json"
        shutil.copy2(teacher_path, copied_teacher)
    else:
        copied_teacher = teacher_path

    memory_sha256, memory_file_count = _mobilegpt_memory_digest(memory)
    manifest = {
        "schema_version": MOBILEGPT_COLD_MEMORY_SCHEMA,
        "task_name": str(task_name),
        "source_seed": int(source_seed),
        "source_method": str(source_method or "").strip(),
        "source_model": str(source_model or "").strip(),
        "target_package": str(target_package),
        "target_app": str(target_app or target_package),
        "memory": {
            "relative_path": memory.relative_to(bundle_root).as_posix(),
            "sha256": memory_sha256,
            "file_count": memory_file_count,
            "inventory": inventory,
        },
        "teacher_source": {
            "relative_path": copied_teacher.relative_to(bundle_root).as_posix(),
            "sha256": _file_sha256(copied_teacher),
            "action_count": int(teacher["action_count"]),
            "groundable_action_count": teacher_groundable_action_count,
            "expected_vlm_fallback_action_count": int(
                teacher.get("expected_vlm_fallback_action_count") or 0
            ),
            "fallback_to_vlm_on_teacher_miss": teacher_fallback_enabled,
            "native_vlm_fallback_only": teacher_native_vlm_fallback_only,
        },
        "source_run_log": {
            "relative_path": copied_source.relative_to(bundle_root).as_posix(),
            "sha256": _file_sha256(copied_source),
        },
        "source_stats": {
            "relative_path": copied_stats.relative_to(bundle_root).as_posix(),
            "sha256": _file_sha256(copied_stats),
            "teacher_action_count": int(write_status["teacher_action_count"]),
            "teacher_expected_action_count": int(
                write_status["teacher_expected_action_count"]
            ),
            "teacher_groundable_action_count": teacher_groundable_action_count,
            "teacher_consumed_count": int(
                write_status["teacher_consumed_action_count"]
            ),
            "teacher_vlm_fallback_count": int(
                write_status["teacher_vlm_fallback_count"]
            ),
            "teacher_unrecovered_miss_count": int(
                write_status["teacher_unrecovered_miss_count"]
            ),
            "native_vlm_fallback_only": teacher_native_vlm_fallback_only,
            "task_finished_count": int(write_status["task_finished_count"]),
            "model_calls": _coerce_int(stats_summary.get("model_calls")),
            "chat_models": list(stats_summary.get("chat_models") or []),
            "embedding_models": list(stats_summary.get("embedding_models") or []),
            "prompt_tokens": _coerce_int(stats_summary.get("prompt_tokens")),
            "completion_tokens": _coerce_int(stats_summary.get("completion_tokens")),
            "total_tokens": _coerce_int(stats_summary.get("total_tokens")),
            "task_elapsed_sec": _coerce_float(stats_summary.get("task_elapsed_sec")),
            "wall_sec": float(source_wall_sec or 0.0),
        },
        "official_source_result": {
            "relative_path": copied_result.relative_to(bundle_root).as_posix(),
            "sha256": _file_sha256(copied_result),
            **result_summary,
        },
        "provenance": {
            "native_mobilegpt_learning": True,
            "complete_teacher_action_consumption": True,
            "teacher_groundable_action_count": teacher_groundable_action_count,
            "teacher_vlm_fallback_action_count": int(
                write_status["teacher_vlm_fallback_count"]
            ),
            "native_vlm_fallback_only": teacher_native_vlm_fallback_only,
            "function_conversion_enabled": False,
            "target_inputs_read": False,
            "target_observations_read": False,
            "validator_state_read": False,
            "coordinate_replay": False,
        },
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2, ensure_ascii=False))
        handle.write("\n")
    for path in memory.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    memory.chmod(0o555)
    return validate_mobilegpt_adapted_memory(
        memory,
        task_name=task_name,
        source_seed=source_seed,
        source_run_log=source_path,
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
    source_root = _repo_path(source_memory_root)
    frozen_root = _repo_path(frozen_memory_root)
    if not source_root.is_dir():
        raise FileNotFoundError(f"MobileGPT memory root not found: {source_root}")
    if frozen_root.exists():
        raise FileExistsError(f"immutable_frozen_memory_exists:{frozen_root}")
    shutil.copytree(source_root, frozen_root)
    digest, file_count = _mobilegpt_memory_digest(frozen_root)
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
    frozen_root = _repo_path(frozen_memory_root)
    episode_root = _repo_path(episode_memory_root)
    if not frozen_root.is_dir():
        raise FileNotFoundError(f"frozen_mobilegpt_memory_missing:{frozen_root}")
    if episode_root.exists():
        raise FileExistsError(f"immutable_episode_memory_exists:{episode_root}")
    shutil.copytree(frozen_root, episode_root)
    _make_tree_owner_writable(episode_root)
    digest, file_count = _mobilegpt_memory_digest(episode_root)
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
    episode_root = _repo_path(episode_memory_root)
    actual_digest, actual_file_count = _mobilegpt_memory_digest(episode_root)
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


def _count_mobilegpt_stats_event(path: str | Path, event: str) -> int:
    stats_path = _repo_path(path)
    return sum(1 for row in _iter_jsonl_rows(stats_path) if row.get("event") == event)


def wait_for_mobilegpt_stats_event(
    path: str | Path,
    *,
    event: str = "task_finished",
    previous_count: int = 0,
    timeout_sec: float | None = DEFAULT_MOBILEGPT_EPISODE_WAIT_TIMEOUT_SEC,
    poll_sec: float = 0.5,
) -> dict[str, Any]:
    started = time.time()
    timeout_value = float(timeout_sec) if timeout_sec is not None else -1.0
    unbounded = timeout_value < 0
    deadline = None if unbounded else started + max(0.0, timeout_value)
    last_count = _count_mobilegpt_stats_event(path, event)
    while last_count <= int(previous_count) and (
        unbounded or time.time() < float(deadline)
    ):
        time.sleep(max(0.05, float(poll_sec)))
        last_count = _count_mobilegpt_stats_event(path, event)
    return {
        "event": event,
        "seen": last_count > int(previous_count),
        "previous_count": int(previous_count),
        "count": int(last_count),
        "timeout_sec": None if unbounded else timeout_value,
        "unbounded": unbounded,
        "elapsed_sec": round(time.time() - started, 6),
        "stats_jsonl": str(_repo_path(path)),
    }


def _metadata_float(
    metadata: dict[str, Any],
    key: str,
    default: float,
) -> float:
    value = metadata.get(key)
    if value is None or value == "":
        return float(default)
    return float(value)


def _find_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _mobilegpt_browser_task_html(
    item: ArchivedRunLog,
    *,
    android_world_root: str | Path = DEFAULT_ANDROID_WORLD_ROOT,
    task_params_override: dict[str, Any] | None = None,
) -> str:
    params = dict(task_params_override or item.params or {})
    if "browser_task_seed" not in params:
        return ""
    root = _repo_path(android_world_root)
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
    item: ArchivedRunLog,
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


def build_mobilegpt_command(
    action: str,
    *,
    mobilegpt_root: str | Path = DEFAULT_MOBILEGPT_ROOT,
    mobilegpt_memory_root: str | Path | None = None,
    serial: str = "",
    adb_path: str = "",
    server_host: str = "0.0.0.0",
    port: int = 12345,
    stats_jsonl: str | Path = DEFAULT_MOBILEGPT_STATS_JSONL,
    source_run_log: str | Path = "",
    fallback_to_vlm_on_teacher_miss: bool = False,
    target_package: str = "",
    target_app: str = "",
    runtime_observe_backend: str = "androidworld",
    python_executable: str = sys.executable,
    repo_root: Path = REPO_ROOT,
) -> CommandSpec:
    root = _repo_path(mobilegpt_root, repo_root=repo_root)
    env: dict[str, str] = {}
    if serial.strip():
        env["ANDROID_SERIAL"] = serial.strip()
    env["MOBILEGPT_RUNTIME_OBSERVE_BACKEND"] = str(
        runtime_observe_backend or "androidworld"
    ).strip()
    if adb_path.strip():
        env["ADB_PATH"] = adb_path.strip()
    resolved_memory_root = (
        _repo_path(mobilegpt_memory_root, repo_root=repo_root)
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
        env["MOBILEGPT_STATS_JSONL"] = str(_repo_path(stats_jsonl, repo_root=repo_root))
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

    if resolved_action == "teach-server":
        if not str(source_run_log or "").strip():
            raise ValueError("mobilegpt teach-server requires --source-run-log")
        resolved_stats_jsonl = _repo_path(stats_jsonl, repo_root=repo_root)
        env["MOBILEGPT_STATS_JSONL"] = str(resolved_stats_jsonl)
        env["OMNIFLOW_REPO_ROOT"] = str(repo_root)
        env["MOBILEGPT_TEACHER_ARTIFACT_DIR"] = str(
            resolved_stats_jsonl.parent / "teacher_artifacts"
        )
        env["MOBILEGPT_SERVER_HOST"] = str(server_host or "0.0.0.0")
        env["MOBILEGPT_SERVER_PORT"] = str(int(port))
        argv = [
            python_executable,
            "-m",
            "src.integrations.mobilegpt_teacher",
            "--mobilegpt-root",
            str(root),
            "--source-run-log",
            str(_repo_path(source_run_log, repo_root=repo_root)),
            "--host",
            str(server_host or "0.0.0.0"),
            "--port",
            str(int(port)),
        ]
        if fallback_to_vlm_on_teacher_miss:
            env["MOBILEGPT_TEACHER_FALLBACK_TO_VLM_ON_MISS"] = "1"
            argv.append("--fallback-to-vlm-on-teacher-miss")
        return CommandSpec(
            label="mobilegpt:teach-server",
            argv=argv,
            env=env,
            cwd=repo_root,
            output_path=None,
            metadata={
                "mobilegpt_root": str(root),
                "mobilegpt_memory_root": str(resolved_memory_root or ""),
                "source_run_log": str(_repo_path(source_run_log, repo_root=repo_root)),
                "port": int(port),
                "mode": "mobilegpt_native_teacher_forced_learning",
                "target_package": str(target_package or "").strip(),
                "target_app": str(target_app or "").strip(),
                "state_backend": "androidworld",
            },
        )

    raise ValueError(
        "Unsupported MobileGPT action. Use one of: server, teach-server."
    )


def run_command(spec: CommandSpec, *, dry_run: bool = False) -> int:
    print(f"[{spec.label}] {_command_line(spec)}", flush=True)
    started = time.monotonic()
    if dry_run:
        spec.metadata["wall_sec"] = 0.0
        return 0
    process = subprocess.Popen(
        spec.argv,
        cwd=spec.cwd,
        env=_subprocess_env(spec.env),
        start_new_session=True,
    )
    try:
        process.communicate(timeout=spec.timeout_sec)
        spec.metadata["wall_sec"] = round(time.monotonic() - started, 3)
        return int(process.returncode or 0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        spec.metadata["wall_sec"] = round(time.monotonic() - started, 3)
        spec.metadata["timeout_sec"] = float(spec.timeout_sec or 0)
        spec.metadata["timed_out"] = True
        print(
            f"[{spec.label}] timed out after {spec.timeout_sec}s",
            flush=True,
        )
        return 124
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        raise


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
        path = _repo_path(raw_path)
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
        "ours",
        "mobilegpt_offline_retrieval",
        "t3a_hint",
        "appagent_demo",
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


def _success_source_task_params(row: dict[str, Any]) -> dict[str, Any]:
    context = row.get("androidworld_task_context")
    if isinstance(context, dict):
        for key in ("params", "task_params"):
            value = context.get(key)
            if isinstance(value, dict) and value:
                return dict(value)

    for key in (
        "task_params",
        "params",
        "validator_task_params",
    ):
        value = row.get(key)
        if isinstance(value, dict) and value:
            return dict(value)
    return {}


def _success_source_task_seed(row: dict[str, Any], params: dict[str, Any]) -> int:
    context = row.get("androidworld_task_context")
    if isinstance(context, dict):
        for key in ("task_random_seed", "seed", "collect_seed", "replay_seed"):
            value = context.get(key)
            if value not in (None, ""):
                return _coerce_int(value, DEFAULT_EVAL_TASK_RANDOM_SEED)
    for key in ("task_random_seed", "collect_seed", "replay_seed", "seed"):
        value = row.get(key)
        if value not in (None, ""):
            return _coerce_int(value, DEFAULT_EVAL_TASK_RANDOM_SEED)
    if params.get("seed") not in (None, ""):
        return _coerce_int(params.get("seed"), DEFAULT_EVAL_TASK_RANDOM_SEED)
    return DEFAULT_EVAL_TASK_RANDOM_SEED


def _success_source_step_count(row: dict[str, Any], canonical: dict[str, Any]) -> int:
    for value in (
        canonical.get("step_count"),
        canonical.get("function_step_count"),
        row.get("step_count"),
        row.get("actions_executed"),
    ):
        count = _coerce_int(value, 0)
        if count > 0:
            return count
    steps = canonical.get("steps")
    return len(steps) if isinstance(steps, list) else 0


def _success_source_action_signature_hash(canonical: dict[str, Any]) -> str:
    signature: list[dict[str, Any]] = []
    for step in canonical.get("steps") or []:
        if not isinstance(step, dict):
            continue
        actions: list[dict[str, Any]] = []
        for action in step.get("actions") or []:
            if not isinstance(action, dict):
                continue
            params = (
                dict(action.get("params") or {})
                if isinstance(action.get("params"), dict)
                else {}
            )
            params.pop("source_context", None)
            actions.append({"type": action.get("type"), "params": params})
        if actions or step.get("source") is not None:
            signature.append({"source": step.get("source"), "actions": actions})
    return _stable_json_hash(signature or canonical)


def _success_source_record_key(
    record: dict[str, Any],
) -> tuple[str, str, str, str, str]:
    return (
        str(record.get("task") or ""),
        str(record.get("method") or ""),
        str(record.get("device") or ""),
        str(record.get("run_id") or ""),
        str(record.get("action_signature_hash") or ""),
    )


def _success_source_archive_entry(
    record: dict[str, Any],
    *,
    index_root: Path,
    source_run_log: Path,
) -> dict[str, Any]:
    source_seed = _coerce_int(
        record.get("source_seed")
        or record.get("replay_seed")
        or record.get("task_random_seed"),
        DEFAULT_EVAL_TASK_RANDOM_SEED,
    )
    return {
        "schema_version": "omniflow.androidworld_success_source_runlog_index.v1",
        "source_kind": "androidworld_validator_success_source_runlog",
        "goal": record.get("goal") or "",
        "params": record.get("params")
        if isinstance(record.get("params"), dict)
        else {},
        "retained_source_run_log": _path_ref_from(index_root, source_run_log),
        "source_run_log": _path_ref_from(index_root, source_run_log),
        "source_run_log_sha256": _file_sha256(source_run_log),
        "source_seed": source_seed,
        "replay_seed": _coerce_int(
            record.get("replay_seed") or record.get("task_random_seed"),
            DEFAULT_EVAL_TASK_RANDOM_SEED,
        ),
        "collect_seed": _coerce_int(
            record.get("collect_seed") or record.get("task_random_seed"),
            DEFAULT_EVAL_TASK_RANDOM_SEED,
        ),
        "task_random_seed": _coerce_int(
            record.get("task_random_seed"),
            DEFAULT_EVAL_TASK_RANDOM_SEED,
        ),
        "step_count": _coerce_int(record.get("step_count"), 0),
        "method": record.get("method") or "",
        "device": record.get("device") or "",
        "run_id": record.get("run_id") or None,
        "action_signature_hash": record.get("action_signature_hash") or "",
        "params_hash": record.get("params_hash") or "",
        "latest_official_success_source": True,
        "accepted_first30": False,
    }


def _load_success_source_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _write_success_source_task_indexes(
    *,
    archive_root: Path,
    task_name: str,
    records: Sequence[dict[str, Any]],
    latest_source_run_log: Path,
    latest_record: dict[str, Any],
) -> None:
    task_slug = _safe_stem(task_name)
    by_task_dir = archive_root / "by_task" / task_slug
    by_task_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = by_task_dir / "metadata.json"
    index_path = archive_root / "index_by_task.json"

    _write_json(
        metadata_path,
        {
            **latest_record,
            "retained_source_run_log": _path_ref_from(
                by_task_dir,
                latest_source_run_log,
            ),
            "all_source_runlogs": [
                {
                    "run_id": record.get("run_id"),
                    "method": record.get("method"),
                    "device": record.get("device"),
                    "source_run_log": record.get("local_source_run_log"),
                    "result_file": record.get("result_file"),
                }
                for record in records
            ],
        },
    )
    _write_json(
        archive_root / "source_runlogs.json",
        list(records),
    )
    _write_jsonl(archive_root / "source_runlogs.jsonl", list(records))
    _write_json(
        index_path,
        {
            task_name: _success_source_archive_entry(
                latest_record,
                index_root=archive_root,
                source_run_log=latest_source_run_log,
            )
        },
    )


def _write_success_source_global_index(
    *,
    output_root: Path,
    latest_by_task: dict[str, tuple[dict[str, Any], Path]],
) -> Path | None:
    if not latest_by_task:
        return None
    archive_root = output_root / "_aggregate" / "success_source_runlogs"
    archive_root.mkdir(parents=True, exist_ok=True)
    existing_json = archive_root / "source_runlogs.json"
    records = _load_success_source_records(existing_json)
    by_key = {_success_source_record_key(record): record for record in records}
    for record, _source_path in latest_by_task.values():
        by_key[_success_source_record_key(record)] = record
    merged = list(by_key.values())

    latest_records_by_task: dict[str, tuple[dict[str, Any], Path]] = {}
    for record in merged:
        task_name = str(record.get("task") or record.get("task_name") or "").strip()
        if not task_name:
            continue
        source_text = str(
            record.get("retained_source_run_log")
            or record.get("local_canonical_run_log")
            or record.get("local_source_run_log")
            or ""
        ).strip()
        if not source_text:
            continue
        source_path = Path(source_text).expanduser()
        if source_path.is_absolute():
            resolved_source_path = source_path.resolve()
        else:
            index_relative = (archive_root / source_path).resolve()
            resolved_source_path = (
                index_relative if index_relative.exists() else _repo_path(source_path)
            )
        latest_records_by_task[task_name] = (record, resolved_source_path)
    latest_records_by_task.update(latest_by_task)

    _write_json(existing_json, merged)
    _write_jsonl(archive_root / "source_runlogs.jsonl", merged)
    _write_json(
        archive_root / "index_by_task.json",
        {
            task_name: _success_source_archive_entry(
                record,
                index_root=archive_root,
                source_run_log=source_path,
            )
            for task_name, (record, source_path) in sorted(
                latest_records_by_task.items()
            )
        },
    )
    return (archive_root / "index_by_task.json").resolve()


def save_success_source_runlogs_from_results(
    paths: Sequence[str | Path],
    *,
    output_root: str | Path,
) -> dict[str, Any]:
    output_root_path = _repo_path(output_root)
    result_files = discover_task_result_files(paths)
    saved_records: list[dict[str, Any]] = []
    skipped = {
        "not_official_success": 0,
        "missing_canonical_run": 0,
        "missing_task": 0,
    }
    latest_by_task: dict[str, tuple[dict[str, Any], Path]] = {}

    for file_path in result_files:
        path_context = _task_result_path_context(file_path)
        for row in _iter_jsonl_rows(file_path):
            if not _official_validator_success(row):
                skipped["not_official_success"] += 1
                continue
            canonical = row.get("canonical_run")
            if not isinstance(canonical, dict):
                skipped["missing_canonical_run"] += 1
                continue
            task_name = str(
                row.get("task_name")
                or row.get("task")
                or path_context.get("task_name")
                or ""
            ).strip()
            if not task_name:
                skipped["missing_task"] += 1
                continue

            method = str(row.get("method") or path_context.get("method") or "").strip()
            device = str(row.get("device") or path_context.get("device") or "").strip()
            run_dir = str(
                row.get("run_dir") or path_context.get("run_dir") or ""
            ).strip()
            params = _success_source_task_params(row)
            task_seed = _success_source_task_seed(row, params)
            canonical_to_write = canonicalize_run_log(canonical)
            if canonical_to_write["task_name"] != task_name:
                raise ValueError(
                    "success_source_run_log_task_mismatch:"
                    f"{task_name}:{canonical_to_write['task_name']}"
                )

            run_id = str(
                canonical_to_write.get("run_id") or row.get("artifact_ref") or ""
            ).strip()
            canonical_hash = _stable_json_hash(canonical_to_write)
            run_id_for_path = run_id or canonical_hash[:12]
            action_signature_hash = _success_source_action_signature_hash(
                canonical_to_write
            )
            task_root = output_root_path / _safe_stem(task_name)
            archive_root = task_root / "success_source_runlogs"
            raw_dir = (
                archive_root
                / "raw"
                / _safe_stem(method or "unknown_method")
                / _safe_stem(device or "unknown_device")
            )
            source_name = (
                f"{_safe_stem(task_name)}."
                f"{_safe_stem(method or 'unknown_method')}."
                f"{_safe_stem(device or 'unknown_device')}."
                f"{_safe_stem(run_id_for_path)}.run_log.json"
            )
            raw_source_path = raw_dir / source_name
            latest_source_path = (
                archive_root / "by_task" / _safe_stem(task_name) / "source.run_log.json"
            )
            _write_json(raw_source_path, canonical_to_write)
            _write_json(latest_source_path, canonical_to_write)

            source_pool_record = (
                row.get("source_pool_record")
                if isinstance(row.get("source_pool_record"), dict)
                else {}
            )
            record = {
                "schema_version": "omniflow.androidworld_task_success_source_runlog.v1",
                "source_kind": "androidworld_validator_success_source_runlog",
                "task": task_name,
                "task_name": task_name,
                "goal": row.get("goal") or canonical_to_write.get("goal") or "",
                "params": params,
                "collect_seed": task_seed,
                "replay_seed": task_seed,
                "task_random_seed": task_seed,
                "method": method,
                "device": device,
                "run_dir": run_dir,
                "result_file": str(file_path),
                "run_id": run_id or None,
                "artifact_kind": row.get("artifact_kind"),
                "artifact_ref": row.get("artifact_ref"),
                "androidworld_success": True,
                "official_validator_success": True,
                "androidworld_reward": (
                    row.get("androidworld_validator_result", {}).get("reward")
                    if isinstance(row.get("androidworld_validator_result"), dict)
                    else None
                ),
                "step_count": _success_source_step_count(row, canonical_to_write),
                "actions_executed": _coerce_int(row.get("actions_executed"), 0),
                "duration_ms": _coerce_float(row.get("duration_ms"), 0.0),
                "action_signature_hash": action_signature_hash,
                "params_hash": _stable_json_hash(params),
                "canonical_hash": canonical_hash,
                "local_source_run_log": str(raw_source_path.resolve()),
                "local_canonical_run_log": str(raw_source_path.resolve()),
                "retained_source_run_log": str(latest_source_path.resolve()),
                "launcher_source_pool_run_log": source_pool_record.get(
                    "local_canonical_run_log"
                )
                or source_pool_record.get("local_source_run_log")
                or "",
                "latest_official_success_source": True,
                "accepted_first30": False,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }

            existing_path = archive_root / "source_runlogs.json"
            existing = _load_success_source_records(existing_path)
            by_key = {_success_source_record_key(item): item for item in existing}
            key = _success_source_record_key(record)
            is_new = key not in by_key
            by_key[key] = record
            task_records = list(by_key.values())
            _write_success_source_task_indexes(
                archive_root=archive_root,
                task_name=task_name,
                records=task_records,
                latest_source_run_log=latest_source_path,
                latest_record=record,
            )
            latest_by_task[task_name] = (record, latest_source_path.resolve())
            if is_new:
                saved_records.append(record)

    global_index_path = _write_success_source_global_index(
        output_root=output_root_path,
        latest_by_task=latest_by_task,
    )
    return {
        "schema_version": "omniflow.androidworld_success_source_runlogs_summary.v1",
        "output_root": str(output_root_path),
        "task_results_files": [str(path) for path in result_files],
        "saved_count": len(saved_records),
        "materialized_task_count": len(latest_by_task),
        "saved_records": saved_records,
        "skipped": skipped,
        "global_index_by_task": str(global_index_path or ""),
    }


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
    prompt_tokens = sum(_coerce_int(row.get("prompt_tokens")) for _, row in rows)
    completion_tokens = sum(
        _coerce_int(row.get("completion_tokens")) for _, row in rows
    )
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
                "official_validator_used": _official_validator_used(row),
                "official_validator_success": _official_validator_success(row),
                "success": _official_validator_success(row),
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
                "tokens": _coerce_int(row.get("total_tokens")),
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
            }
        )

    task_count = len(rows)
    return {
        "schema_version": "omniflow.androidworld_replay_pipeline_summary.v2",
        "task_count": task_count,
        "task_results_files": [str(path) for path in task_result_files],
        "official_validator_task_count": official_validator_task_count,
        "official_validator_success_count": official_validator_success_count,
        "official_validator_success_rate": _rate(
            official_validator_success_count,
            official_validator_task_count,
        ),
        "success_count": official_validator_success_count,
        "success_total": official_validator_task_count,
        "success_rate": _rate(
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
        "fallback_steps": fallback_steps,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "tokens": total_tokens,
        "relocation_diagnostic_count": relocation_diagnostic_count,
        "per_task": per_task,
    }


def write_metrics_summary(summary: dict[str, Any], output_path: str | Path) -> None:
    path = _repo_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md_lines = [
        "# AndroidWorld Replay Pipeline Summary",
        "",
        f"- task_count: `{summary['task_count']}`",
        (f"- success: `{summary['success_count']}/{summary['success_total']}`"),
        f"- replay_completed: `{summary['replay_completed_count']}/{summary['replay_task_count']}`",
        f"- replay_coverage: `{summary['replay_task_count']}/{summary['task_count']}`",
        f"- replay_step_completed: `{summary['replay_step_completed_count']}/{summary['replay_step_total']}`",
        f"- actions_executed: `{summary['actions_executed']}`",
        f"- relocation_diagnostics: `{summary.get('relocation_diagnostic_count', 0)}`",
        f"- duration_s: `{round(_coerce_float(summary['duration_ms']) / 1000.0, 3)}`",
        f"- model_calls: `{summary.get('model_calls', 0)}`",
        f"- prompt_tokens: `{summary.get('prompt_tokens', 0)}`",
        f"- completion_tokens: `{summary.get('completion_tokens', 0)}`",
        f"- total_tokens: `{summary.get('total_tokens', 0)}`",
        "",
        "| task | success | replay_completed | actions | calls | tokens | step_completed | relocation | sec | error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in summary.get("per_task") or []:
        md_lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("task_name") or item.get("task") or ""),
                    "1" if item.get("success") else "0",
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
    parser.add_argument(
        "--perform-emulator-setup",
        dest="perform_emulator_setup",
        action="store_true",
        default=True,
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


def _select_from_args(args: argparse.Namespace) -> list[ArchivedRunLog]:
    archive = load_archive_index(args.index)
    selected = select_archive_items(
        archive,
        tasks=args.tasks,
        first60=bool(getattr(args, "first60", False)),
        limit=None,
    )
    selected = filter_archive_items(
        selected,
        source_format=str(getattr(args, "source_format", "all") or "all"),
        accepted_first30=bool(getattr(args, "accepted_first30", False)),
    )
    limit = getattr(args, "limit", None)
    if limit is not None:
        selected = selected[: max(0, int(limit))]
    return selected


def _profile_summary(profiles: Sequence[SourceRunLogProfile]) -> dict[str, Any]:
    by_format: dict[str, int] = {}
    ready_count = 0
    accepted_first30_count = 0
    for profile in profiles:
        by_format[profile.replay_format] = by_format.get(profile.replay_format, 0) + 1
        ready_count += int(profile.direct_replay_ready)
        accepted_first30_count += int(profile.accepted_first30)
    return {
        "schema_version": "omniflow.androidworld_replay_archive_inspect.v1",
        "task_count": len(profiles),
        "direct_replay_ready_count": ready_count,
        "accepted_first30_count": accepted_first30_count,
        "by_format": by_format,
        "tasks": [profile.to_dict() for profile in profiles],
    }


def cmd_list(args: argparse.Namespace) -> int:
    selected = _select_from_args(args)
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "task": item.task,
                        "goal": item.goal,
                        "source_run_log": str(item.source_run_log),
                        "replay_seed": item.replay_seed,
                        "params": item.params,
                        "source_profile": profile_source_run_log(item).to_dict(),
                    }
                    for item in selected
                ],
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    for index, item in enumerate(selected, 1):
        profile = profile_source_run_log(item)
        print(
            f"{index:02d} {item.task} format={profile.replay_format} "
            f"steps={profile.step_count} cards={profile.card_count} "
            f"seed={item.replay_seed} runlog={item.source_run_log}"
        )
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    selected = _select_from_args(args)
    profiles = [profile_source_run_log(item) for item in selected]
    summary = _profile_summary(profiles)
    if args.output:
        output_path = _repo_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    print(
        f"tasks={summary['task_count']} ready={summary['direct_replay_ready_count']} "
        f"accepted_first30={summary['accepted_first30_count']} "
        f"by_format={summary['by_format']}"
    )
    for index, profile in enumerate(profiles, 1):
        notes = "; ".join(profile.notes)
        print(
            f"{index:02d} {profile.task} {profile.replay_format} "
            f"steps={profile.step_count} cards={profile.card_count} "
            f"ready={int(profile.direct_replay_ready)}"
            + (f" notes={notes}" if notes else "")
        )
    return 0


ONE_TASK_ALL_METHODS = (
    "fixed_replay",
    "ours",
    "mobilegpt_offline_retrieval",
    "appagent_demo",
    "t3a_hint",
)
ONE_TASK_SUPPORTED_METHODS = ONE_TASK_ALL_METHODS
MOBILEGPT_METHODS = frozenset({"mobilegpt_offline_retrieval"})
APPAGENT_METHODS = frozenset({"appagent_demo"})
_ONE_TASK_NON_EXECUTED_STATUSES = {
    "INVALID_MEMORY_LEAKAGE",
    "env_failed",
    "init_failed",
    "setup_failed",
}


TOP_LEVEL_COMMANDS = {
    "list",
    "inspect",
    "one-task",
    "mobilegpt",
    "metrics",
    "doctor",
}


def _parse_one_task_methods(raw_methods: str) -> list[str]:
    raw = str(raw_methods or "").strip()
    if not raw or raw == "all":
        return list(ONE_TASK_ALL_METHODS)
    methods = [_safe_stem(item, fallback="") for item in raw.split(",")]
    methods = [method for method in methods if method]
    invalid = [method for method in methods if method not in ONE_TASK_SUPPORTED_METHODS]
    if invalid:
        raise ValueError(f"Unsupported one-task method(s): {', '.join(invalid)}")
    return methods


def _is_mobilegpt_method(method: str) -> bool:
    return str(method or "").strip() in MOBILEGPT_METHODS


def _one_task_record_has_formal_result(record: dict[str, Any]) -> bool:
    if bool(record.get("summary_exclude")):
        return False
    if str(record.get("status") or "").strip() in _ONE_TASK_NON_EXECUTED_STATUSES:
        return False
    return bool(str(record.get("output_path") or "").strip())


def _one_task_formal_result_paths(record: dict[str, Any]) -> list[Path]:
    if not _one_task_record_has_formal_result(record):
        return []
    output_path = _repo_path(str(record.get("output_path") or ""))
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

    collect(params)
    return tuple(sorted(values, key=len, reverse=True))


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
            action.get("name") or action.get("type") or action.get("tool") or ""
        ).strip()
        raw_params = action.get("params")
        if raw_params is None:
            raw_params = action.get("arguments")
        if raw_params is None:
            raw_params = action.get("args")
        if name:
            return name, dict(raw_params or {}) if isinstance(raw_params, dict) else {}
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


def _t3a_semantic_hint_step(
    step: Any,
    *,
    forbidden_values: Sequence[str],
) -> dict[str, str] | None:
    name, params = _t3a_hint_step_action(step)
    action = name.strip().lower()
    if not action or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", action):
        return None
    if action in {"status", "finish", "done"}:
        return None
    semantic: dict[str, str] = {"action": action}
    target = _t3a_hint_target(params, forbidden_values=forbidden_values)
    if target:
        semantic["target"] = target
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
    store = FunctionStore(_repo_path(store_path))
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
    item: ArchivedRunLog,
    *,
    output_root: str | Path,
    store_path: str | Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> Path:
    payload, _, profile, _ = canonicalize_source_run_log(
        item,
        repo_root=repo_root,
        write_materialized=False,
    )
    forbidden_values = _t3a_hint_forbidden_values(item.params)
    source_steps = list(payload.get("steps") or [])
    semantic_source = "source_run_log"
    semantic_input_steps = source_steps
    store_alignment_mode = "not_applicable"
    if store_path is not None:
        resolved_store_path = _repo_path(store_path, repo_root=repo_root)
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
        alignment_modes: list[str] = []
        source_cursor = 0
        for function_step in raw_function_steps:
            function_action, _ = _t3a_hint_step_action(function_step)
            function_state_id = str(
                function_step.get("source_state_id")
                if isinstance(function_step, dict)
                else ""
            ).strip()
            aligned_index = None
            alignment_mode = "state_identity"
            for source_index in range(source_cursor, len(source_steps)):
                source_step = source_steps[source_index]
                source_action, _ = _t3a_hint_step_action(source_step)
                source_state_id = str(
                    source_step.get("before_state_id")
                    if isinstance(source_step, dict)
                    else ""
                ).strip()
                if (
                    source_action.strip().lower() == function_action.strip().lower()
                    and source_state_id == function_state_id
                ):
                    aligned_index = source_index
                    break
            if aligned_index is None:
                alignment_mode = "ordered_action"
                for source_index in range(source_cursor, len(source_steps)):
                    source_action, _ = _t3a_hint_step_action(source_steps[source_index])
                    if source_action.strip().lower() == function_action.strip().lower():
                        aligned_index = source_index
                        break
            if aligned_index is None:
                raise ValueError(
                    "t3a_hint_function_runlog_action_mismatch:"
                    f"{function_state_id}:{function_action}"
                )
            semantic_input_steps.append(function_step)
            alignment_modes.append(alignment_mode)
            source_cursor = aligned_index + 1
        semantic_source = "omniflow_function_store"
        store_alignment_mode = (
            "state_identity"
            if all(mode == "state_identity" for mode in alignment_modes)
            else "ordered_action"
        )
    semantic_steps = [
        semantic
        for semantic in (
            _t3a_semantic_hint_step(
                step,
                forbidden_values=forbidden_values,
            )
            for step in semantic_input_steps
        )
        if semantic is not None
    ]
    if not semantic_steps:
        raise ValueError(f"source runlog produced no safe T3A hints: {item.task}")
    hint_payload = {
        "schema_version": "omniflow.t3a_semantic_hint.v1",
        "task": item.task,
        "source_format": profile.replay_format,
        "semantic_source": semantic_source,
        "store_alignment_mode": store_alignment_mode,
        "source_step_count": len(source_steps),
        "semantic_step_count": len(semantic_steps),
        "steps": semantic_steps,
    }
    hint_dir = _repo_path(output_root, repo_root=repo_root) / "_source_action_hints"
    hint_dir.mkdir(parents=True, exist_ok=True)
    hint_path = hint_dir / f"{_safe_stem(item.task)}.source_action_hints.json"
    hint_path.write_text(
        json.dumps(hint_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return hint_path.resolve()


def _file_sha256(path: str | Path) -> str:
    resolved = _repo_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"provenance_artifact_missing:{resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _claim_method_memory_root(memory_root: str | Path) -> Path:
    root = _repo_path(memory_root)
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
    root = _repo_path(memory_root)
    if not root.is_dir():
        raise FileNotFoundError(f"unclaimed_memory_root:{root}")
    source: dict[str, Any] = {"seed": source_seed}
    if source_run_log is not None:
        resolved_source = _repo_path(source_run_log)
        source.update(
            {
                "run_log": str(resolved_source),
                "run_log_sha256": _file_sha256(resolved_source),
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


def _select_mobilegpt_teacher_target(
    targets: Sequence[DeviceTarget],
    preferred_label: str = "source5556",
) -> DeviceTarget:
    if not targets:
        raise ValueError("mobilegpt one-task requires at least one device target")
    preferred = _safe_stem(preferred_label, fallback="")
    for target in targets:
        if target.label == preferred:
            return target
    for target in targets:
        if target.label == "source5556":
            return target
    return targets[0]


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
    item: ArchivedRunLog,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, str]:
    """Infer the Android app MobileGPT should open from the source trajectory."""

    try:
        canonical, materialization, profile, _ = canonicalize_source_run_log(
            item,
            repo_root=repo_root,
            write_materialized=False,
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
                    "source_format": profile.replay_format,
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
                    "source_format": profile.replay_format,
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
                    "source_format": profile.replay_format,
                }

    package_name = _mobilegpt_observation_package(canonical.get("final_state"))
    if package_name:
        return {
            "target_package": package_name,
            "target_app": package_name,
            "target_source": "source_runlog_final_state",
            "source_materialization": materialization,
            "source_format": profile.replay_format,
        }

    return {
        "target_package": "",
        "target_app": "",
        "target_source": "unresolved",
        "source_materialization": materialization,
        "source_format": profile.replay_format,
    }


def _mobilegpt_target_package_from_open_target_app(open_target_app: str) -> str:
    value = str(open_target_app or "").strip()
    if not value:
        return ""
    if "/" in value:
        return value.split("/", 1)[0].strip()
    return value


def _read_mobilegpt_conversion_manifest(
    memory_root: str | Path,
) -> tuple[Path, dict[str, Any]]:
    root = _repo_path(memory_root)
    manifest_path = root / "conversion_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            f"mobilegpt_prebuilt_conversion_manifest_missing:{manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"mobilegpt_prebuilt_conversion_manifest_invalid:{manifest_path}"
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError(
            f"mobilegpt_prebuilt_conversion_manifest_invalid:{manifest_path}"
        )
    if manifest.get("schema_version") != "omniflow.mobilegpt-function-conversion.v2":
        raise ValueError(
            f"mobilegpt_prebuilt_conversion_manifest_schema_invalid:{manifest_path}"
        )
    task_name = str(manifest.get("task_name") or "").strip()
    source_seed = manifest.get("source_seed")
    if not task_name:
        raise ValueError(f"mobilegpt_prebuilt_task_name_missing:{manifest_path}")
    if not isinstance(source_seed, int) or isinstance(source_seed, bool):
        raise ValueError(f"mobilegpt_prebuilt_source_seed_missing:{manifest_path}")
    return manifest_path, manifest


def _mobilegpt_target_from_conversion_manifest(
    manifest_path: Path,
    manifest: dict[str, Any],
) -> dict[str, str]:
    if manifest.get("target_inputs_read") is not False:
        raise ValueError("mobilegpt_prebuilt_target_inputs_forbidden")
    if manifest.get("coordinate_replay") is not False:
        raise ValueError("mobilegpt_prebuilt_coordinate_replay_forbidden")
    package_name = str(manifest.get("package_name") or "").strip()
    app_name = str(manifest.get("app") or package_name).strip()
    if not package_name or not app_name:
        raise ValueError("mobilegpt_prebuilt_target_missing")
    return {
        "target_package": package_name,
        "target_app": app_name,
        "target_source": "prebuilt_conversion_manifest",
        "conversion_manifest": str(manifest_path),
        "output_memory_sha256": str(manifest.get("output_memory_sha256") or ""),
        "source_store_sha256": str(manifest.get("source_store_sha256") or ""),
        "source_cold_memory_sha256": str(
            manifest.get("source_cold_memory_sha256") or ""
        ),
    }


def _validate_mobilegpt_conversion_provenance(
    manifest: dict[str, Any],
    *,
    task_name: str,
    source_seed: int,
) -> None:
    if manifest.get("task_name") != task_name:
        raise ValueError("mobilegpt_prebuilt_task_name_mismatch")
    if manifest.get("source_seed") != source_seed:
        raise ValueError("mobilegpt_prebuilt_source_seed_mismatch")


def _infer_mobilegpt_target_from_prebuilt_memory(
    memory_root: str | Path,
) -> dict[str, str]:
    manifest_path, manifest = _read_mobilegpt_conversion_manifest(memory_root)
    return _mobilegpt_target_from_conversion_manifest(manifest_path, manifest)


def _mobilegpt_memory_write_status(
    *,
    stats_summary: dict[str, Any],
    memory_inventory: dict[str, Any],
    cold_validator_success: bool = True,
) -> dict[str, Any]:
    task_finished_count = _coerce_int(stats_summary.get("task_finished_count"))
    teacher_action_count = _coerce_int(stats_summary.get("teacher_action_count"))
    teacher_expected_action_count = _coerce_int(
        stats_summary.get("teacher_expected_action_count"),
        teacher_action_count,
    )
    teacher_consumed_action_count = _coerce_int(
        stats_summary.get("teacher_consumed_action_count"),
        teacher_action_count,
    )
    teacher_miss_count = _coerce_int(stats_summary.get("teacher_miss_count"))
    teacher_vlm_fallback_count = _coerce_int(
        stats_summary.get("teacher_vlm_fallback_count")
    )
    teacher_unrecovered_miss_count = _coerce_int(
        stats_summary.get("teacher_unrecovered_miss_count"),
        teacher_miss_count,
    )
    teacher_vlm_fallback_enabled = bool(
        stats_summary.get("teacher_vlm_fallback_enabled")
    )
    native_vlm_fallback_only = bool(
        stats_summary.get("native_vlm_fallback_only")
    )
    teacher_failed_finish_count = _coerce_int(
        stats_summary.get("teacher_failed_finish_count")
    )
    teacher_forced_select_count = _coerce_int(
        stats_summary.get("teacher_forced_select_count")
    )
    teacher_action_error_count = _coerce_int(
        stats_summary.get("teacher_action_error_count")
    )
    has_useful_actions = bool(memory_inventory.get("has_useful_actions"))
    has_recallable_subtasks = bool(memory_inventory.get("has_recallable_subtasks"))
    reasons: list[str] = []
    if task_finished_count <= 0:
        reasons.append("missing_task_finished")
    if not cold_validator_success:
        reasons.append("cold_validator_failed")
    if not has_recallable_subtasks:
        reasons.append("missing_recallable_subtasks")
    if not has_useful_actions:
        reasons.append("missing_non_finish_actions")
    if teacher_expected_action_count <= 0 and not native_vlm_fallback_only:
        reasons.append("missing_teacher_actions")
    if native_vlm_fallback_only and teacher_expected_action_count != 0:
        reasons.append("native_vlm_fallback_has_teacher_actions")
    if native_vlm_fallback_only and _coerce_int(
        stats_summary.get("model_calls")
    ) <= 0:
        reasons.append("missing_native_vlm_fallback_calls")
    if teacher_consumed_action_count != teacher_expected_action_count:
        reasons.append("incomplete_source_actions")
    if teacher_unrecovered_miss_count > 0 or teacher_failed_finish_count > 0:
        reasons.append("teacher_grounding_failed")
    if teacher_vlm_fallback_count > 0 and not teacher_vlm_fallback_enabled:
        reasons.append("unexpected_teacher_vlm_fallback")
    if teacher_action_error_count > 0:
        reasons.append("teacher_action_failed")
    if teacher_forced_select_count > 0:
        reasons.append("non_native_select_override")
    return {
        "memory_written": not reasons,
        "reasons": reasons,
        "task_finished_count": task_finished_count,
        "cold_validator_success": cold_validator_success,
        "has_recallable_subtasks": has_recallable_subtasks,
        "has_useful_actions": has_useful_actions,
        "teacher_action_count": teacher_action_count,
        "teacher_expected_action_count": teacher_expected_action_count,
        "teacher_consumed_action_count": teacher_consumed_action_count,
        "teacher_miss_count": teacher_miss_count,
        "teacher_vlm_fallback_count": teacher_vlm_fallback_count,
        "teacher_unrecovered_miss_count": teacher_unrecovered_miss_count,
        "teacher_vlm_fallback_enabled": teacher_vlm_fallback_enabled,
        "native_vlm_fallback_only": native_vlm_fallback_only,
        "teacher_failed_finish_count": teacher_failed_finish_count,
        "teacher_forced_select_count": teacher_forced_select_count,
        "teacher_action_error_count": teacher_action_error_count,
    }


def _mobilegpt_memory_check_record(
    *,
    task: str,
    method: str,
    device: str,
    memory_root: Path,
    mobilegpt_memory_root: Path,
    stats_summary_path: Path,
    stats_summary: dict[str, Any],
    memory_inventory: dict[str, Any],
    summary_exclude: bool,
    cold_validator_success: bool = True,
) -> dict[str, Any]:
    status = _mobilegpt_memory_write_status(
        stats_summary=stats_summary,
        memory_inventory=memory_inventory,
        cold_validator_success=cold_validator_success,
    )
    memory_written = bool(status.get("memory_written"))
    return {
        "label": "mobilegpt:cold-memory-check",
        "returncode": 0 if memory_written else 1,
        "output_path": str(memory_root),
        "command": "",
        "task": task,
        "method": method,
        "device": device,
        "status": "cold_memory_written" if memory_written else "cold_memory_missing",
        "summary_exclude": bool(summary_exclude),
        "metadata": {
            "memory_root": str(memory_root),
            "mobilegpt_memory_root": str(mobilegpt_memory_root),
            "mobilegpt_stats_summary": str(stats_summary_path),
            "mobilegpt_memory_inventory": memory_inventory,
            "mobilegpt_memory_write_status": status,
            "task_finished_count": status["task_finished_count"],
            "cold_validator_success": status["cold_validator_success"],
        },
    }


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


def _normalize_one_task_result_row(row: dict[str, Any]) -> dict[str, Any]:
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
        normalized["tokens"] = episode_total_tokens
        normalized["tokens_source"] = "mobilegpt_episode_stats"

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
        normalized["tokens_source"] = "mobilegpt_stats"
    if total_tokens <= 0:
        total_tokens = _coerce_int(normalized.get("prompt_tokens")) + _coerce_int(
            normalized.get("completion_tokens")
        )
    normalized["total_tokens"] = total_tokens
    normalized["tokens"] = total_tokens
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
    if normalized.get("tokens_source") in {
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

    if normalized.get("official_validator_success") is not None:
        normalized["success"] = bool(normalized.get("official_validator_success"))
    return normalized


def _one_task_row_value(row: dict[str, Any], key: str) -> str:
    if key == "success":
        if row.get("success") is True:
            return "1"
        if row.get("success") is False:
            return "0"
        return ""
    if key == "replay":
        if row.get("replay_step_total"):
            return (
                f"{row.get('replay_step_completed_count') or 0}/"
                f"{row.get('replay_step_total') or 0}"
            )
        return ""
    value = row.get(key)
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


_ONE_TASK_METADATA_ROW_KEYS = (
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
    "official_agent_name",
    "uses_omniflow_agent",
    "uses_source_action_hints",
    "uses_function_retrieval",
    "function_reference_catalog_path",
    "perform_emulator_setup",
)


def _promote_one_task_metadata_to_row(
    row: dict[str, Any],
    records: Sequence[dict[str, Any]],
) -> None:
    for record in records:
        metadata = (
            record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        )
        for key in _ONE_TASK_METADATA_ROW_KEYS:
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


def _one_task_summary_rows(
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
        memory_root = _repo_path(memory_root_value)
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
                _repo_path(explicit_episode_summary)
                if explicit_episode_summary
                else None
            ),
            stats_jsonl_path=(
                _repo_path(explicit_episode_jsonl) if explicit_episode_jsonl else None
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
            status in _ONE_TASK_NON_EXECUTED_STATUSES for status in explicit_statuses
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
        _promote_one_task_metadata_to_row(row, records)
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
            prep_fields = _mobilegpt_stats_row_fields("prep", prep_stats)
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
            teacher_fields = _mobilegpt_stats_row_fields("teacher", teacher_stats)
            row.update(teacher_fields)
            row.update(
                {
                    "prep_type": row.get("prep_type")
                    or "mobilegpt_native_cold_memory_write",
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
            episode_fields = _mobilegpt_stats_row_fields("episode", episode_stats)
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
    return [_normalize_one_task_result_row(row) for row in rows]


def _aggregate_normalized_one_task_rows(
    aggregate_summary: dict[str, Any],
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Recompute the one-task aggregate from its final canonical rows."""
    aggregate = dict(aggregate_summary)
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
            "success_count": official_success_count,
            "success_total": len(official_rows),
            "success_rate": _rate(official_success_count, len(official_rows)),
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
            "replay_step_completed_count": replay_step_completed,
            "replay_step_total": replay_step_total,
            "replay_step_completed_rate": _rate(
                replay_step_completed,
                replay_step_total,
            ),
            "model_calls": sum(
                _coerce_int(row.get("model_calls")) for row in canonical_rows
            ),
            "fallback_steps": sum(
                _coerce_int(row.get("fallback_steps")) for row in canonical_rows
            ),
            "prompt_tokens": sum(
                _coerce_int(row.get("prompt_tokens")) for row in canonical_rows
            ),
            "completion_tokens": sum(
                _coerce_int(row.get("completion_tokens")) for row in canonical_rows
            ),
            "total_tokens": sum(
                _coerce_int(row.get("total_tokens")) for row in canonical_rows
            ),
            "relocation_diagnostic_count": sum(
                _coerce_int(row.get("relocation_diagnostic_count"))
                for row in canonical_rows
            ),
            "per_task": canonical_rows,
        }
    )
    aggregate["tokens"] = aggregate["total_tokens"]
    return aggregate


def _write_one_task_summary(
    *,
    output_root: str | Path,
    task: str,
    command_records: Sequence[dict[str, Any]],
    aggregate_summary: dict[str, Any],
) -> dict[str, Any]:
    task_root = _repo_path(output_root) / _safe_stem(task)
    task_root.mkdir(parents=True, exist_ok=True)
    rows = _one_task_summary_rows(
        task=task,
        command_records=command_records,
        aggregate_summary=aggregate_summary,
    )
    canonical_aggregate = _aggregate_normalized_one_task_rows(
        aggregate_summary,
        rows,
    )
    summary = {
        "schema_version": "omniflow.androidworld_one_task_methods.v1",
        "task_name": task,
        "task_root": str(task_root),
        "rows": rows,
        "aggregate": canonical_aggregate,
    }
    summary_path = task_root / "one_task_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    commands_path = task_root / "one_task_commands.jsonl"
    commands_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in command_records)
        + "\n",
        encoding="utf-8",
    )

    visible_columns = [
        ("method", "method"),
        ("device", "device"),
        ("status", "status"),
        ("initialized", "initialized"),
        ("init_audit_status", "init"),
        ("success", "success"),
        ("model_calls", "calls"),
        ("fallback_steps", "fallback"),
        ("prompt_tokens", "prompt_tokens"),
        ("completion_tokens", "completion_tokens"),
        ("total_tokens", "tokens"),
        ("token_usage_status", "token_status"),
        ("task_params_sha256", "params_sha256"),
        ("duration_sec", "episode_sec"),
        ("wall_sec", "wall_sec"),
        ("prep_model_calls", "prep_calls"),
        ("prep_total_tokens", "prep_tokens"),
        ("prep_duration_sec", "prep_sec"),
        ("teacher_model_calls", "teacher_calls"),
        ("teacher_total_tokens", "teacher_tokens"),
        ("teacher_wall_sec", "teacher_sec"),
        ("episode_model_calls", "episode_calls"),
        ("episode_total_tokens", "episode_tokens"),
        ("episode_wall_sec", "episode_sec"),
        ("actions_executed", "actions"),
        ("replay", "replay"),
        ("run_dir", "run_dir"),
        ("error", "error"),
    ]
    md_lines = [
        f"# AndroidWorld One Task Summary: {task}",
        "",
        "| " + " | ".join(label for _, label in visible_columns) + " |",
        "|" + "|".join("---" for _ in visible_columns) + "|",
    ]
    for row in rows:
        md_lines.append(
            "| "
            + " | ".join(_one_task_row_value(row, key) for key, _ in visible_columns)
            + " |"
        )
    (task_root / "one_task_summary.md").write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )
    return summary


def _print_one_task_summary(summary: dict[str, Any]) -> None:
    visible_columns = [
        ("method", "method"),
        ("device", "device"),
        ("status", "status"),
        ("initialized", "initialized"),
        ("init_audit_status", "init"),
        ("success", "success"),
        ("model_calls", "calls"),
        ("fallback_steps", "fallback"),
        ("prompt_tokens", "prompt_tokens"),
        ("completion_tokens", "completion_tokens"),
        ("total_tokens", "tokens"),
        ("token_usage_status", "token_status"),
        ("task_params_sha256", "params_sha256"),
        ("duration_sec", "episode_sec"),
        ("wall_sec", "wall_sec"),
        ("prep_model_calls", "prep_calls"),
        ("prep_total_tokens", "prep_tokens"),
        ("prep_duration_sec", "prep_sec"),
        ("teacher_model_calls", "teacher_calls"),
        ("teacher_total_tokens", "teacher_tokens"),
        ("teacher_wall_sec", "teacher_sec"),
        ("episode_model_calls", "episode_calls"),
        ("episode_total_tokens", "episode_tokens"),
        ("episode_wall_sec", "episode_sec"),
        ("actions_executed", "actions"),
        ("replay", "replay"),
    ]
    print(
        "| " + " | ".join(label for _, label in visible_columns) + " |",
        flush=True,
    )
    print("|" + "|".join("---" for _ in visible_columns) + "|", flush=True)
    for row in summary.get("rows") or []:
        print(
            "| "
            + " | ".join(_one_task_row_value(row, key) for key, _ in visible_columns)
            + " |",
            flush=True,
        )


def build_mobilegpt_androidworld_command(
    item: ArchivedRunLog,
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
    run_dir_suffix: str = "",
    repo_root: Path = REPO_ROOT,
) -> CommandSpec:
    spec = build_e2e_command(
        item,
        android_world_root=android_world_root,
        output_root=output_root,
        method_name=method_name,
        agent_name="external:mobilegpt",
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
            "MOBILEGPT_STATS_JSONL": str(_repo_path(stats_jsonl, repo_root=repo_root)),
            "MOBILEGPT_RUNTIME_OBSERVE_BACKEND": "androidworld",
            "MOBILEGPT_SERVER_HOST": client_host,
            "MOBILEGPT_SERVER_PORT": str(int(server_port)),
            "MOBILEGPT_TARGET_PACKAGE": str(target_package or "").strip(),
            "MOBILEGPT_WAIT_START_TIMEOUT_SEC": str(float(start_timeout_sec)),
            "MOBILEGPT_WAIT_FINISH_TIMEOUT_SEC": str(float(finish_timeout_sec)),
        },
        cwd=spec.cwd,
        output_path=spec.output_path,
        metadata={
            **dict(spec.metadata),
            "mode": "mobilegpt_androidworld_episode",
            "device_target": target.to_dict(),
            "mobilegpt_stats_jsonl": str(stats_jsonl),
            "mobilegpt_server_host": client_host,
            "mobilegpt_server_port": int(server_port),
            "target_package": str(target_package or "").strip(),
            "official_lifecycle": True,
            "state_backend": "androidworld",
            "action_backend": "androidworld",
            "androidworld_lifecycle_backend": "androidworld",
            "native_androidworld_agent_io": True,
        },
    )


def build_appagent_androidworld_command(
    item: ArchivedRunLog,
    *,
    method_name: str,
    target: DeviceTarget,
    android_world_root: str | Path,
    output_root: str | Path,
    appagent_root: str | Path,
    docs_root: str | Path | None = None,
    action_source: str | Path | None = None,
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
    if teacher_mode and action_source is not None:
        raise ValueError("appagent_teacher_action_source_forbidden")
    if not teacher_mode and action_source is None:
        raise ValueError("appagent_action_source_required")
    selector = "external:appagent_teacher" if teacher_mode else "external:appagent"
    spec = build_e2e_command(
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
    resolved_appagent_root = _repo_path(appagent_root, repo_root=repo_root)
    argv = [*spec.argv, "--appagent-root", str(resolved_appagent_root)]
    resolved_docs_root: Path | None = None
    resolved_action_source: Path | None = None
    resolved_teacher_source: Path | None = None
    resolved_workspace_root: Path | None = None
    if docs_root is not None:
        resolved_docs_root = _repo_path(docs_root, repo_root=repo_root)
        argv.extend(["--appagent-docs-root", str(resolved_docs_root)])
    if action_source is not None:
        resolved_action_source = _repo_path(action_source, repo_root=repo_root)
        argv.extend(["--appagent-action-source", str(resolved_action_source)])
    if teacher_mode:
        resolved_teacher_source = _repo_path(teacher_source, repo_root=repo_root)
        resolved_workspace_root = _repo_path(workspace_root, repo_root=repo_root)
        argv.extend(
            [
                "--appagent-teacher-source",
                str(resolved_teacher_source),
                "--appagent-workspace-root",
                str(resolved_workspace_root),
                "--appagent-demo-name",
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
            "appagent_action_source": str(resolved_action_source or ""),
            "appagent_teacher_source": str(resolved_teacher_source or ""),
            "appagent_workspace_root": str(resolved_workspace_root or ""),
            "uses_omniflow_function": False,
            "uses_appagent_demo_docs": resolved_docs_root is not None,
            "teacher_mode": teacher_mode,
            "official_lifecycle": True,
            "state_backend": "androidworld",
            "action_backend": "androidworld",
            "androidworld_lifecycle_backend": "androidworld",
            "native_androidworld_agent_io": True,
        },
    )


def _run_one_task_mobilegpt(
    *,
    args: argparse.Namespace,
    item: ArchivedRunLog,
    targets: Sequence[DeviceTarget],
    output_root: Path,
    task_params_override: dict[str, Any] | None,
    task_seed: int | None,
    method: str,
    attempt_id: str,
    source_run_log: Path,
) -> tuple[list[dict[str, Any]], int]:
    if method not in MOBILEGPT_METHODS:
        raise ValueError(f"unsupported_mobilegpt_method:{method}")
    if not targets:
        raise ValueError("mobilegpt_device_target_required")

    offline_retrieval = method == "mobilegpt_offline_retrieval"
    if offline_retrieval:
        source_method = (
            str(item.meta.get("method") or "").strip() or DEFAULT_SOURCE_METHOD
        )
        if item.meta.get("latest_official_success_source") is not True:
            raise ValueError(
                "mobilegpt_offline_retrieval_requires_official_success_source:"
                f"task={item.task}"
            )
    source_memory_value = str(
        getattr(args, "mobilegpt_source_memory_root", "") if offline_retrieval else ""
    ).strip()
    source_memory_root = (
        _repo_path(source_memory_value) if source_memory_value else None
    )
    if offline_retrieval and source_memory_root is None:
        raise ValueError(
            "mobilegpt_offline_retrieval requires --mobilegpt-source-memory-root"
        )
    if source_memory_root is not None and not source_memory_root.is_dir():
        raise FileNotFoundError(f"mobilegpt_source_memory_missing:{source_memory_root}")

    adapted_memory: dict[str, Any] = {}
    if source_memory_root is not None:
        adapted_memory = validate_mobilegpt_adapted_memory(
            source_memory_root,
            task_name=item.task,
            source_seed=item.replay_seed,
            source_run_log=source_run_log,
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
        else "sealed_native_source_memory"
        if adapted_memory
        else str(source_target.get("target_source") or "unresolved")
    )
    memory_condition = (
        "adapted_native_memory" if source_memory_root is not None else "empty_memory"
    )
    source_memory_digest = ""
    source_memory_file_count = 0
    if source_memory_root is not None:
        source_memory_digest, source_memory_file_count = _mobilegpt_memory_digest(
            source_memory_root
        )
    adapted_manifest = (
        dict(adapted_memory.get("manifest") or {}) if adapted_memory else {}
    )
    adapted_source_stats = (
        dict(adapted_memory.get("source_stats_summary") or {}) if adapted_memory else {}
    )
    adapted_source_stats_record = (
        dict(adapted_manifest.get("source_stats") or {}) if adapted_manifest else {}
    )
    adapted_official_result = (
        dict(adapted_manifest.get("official_source_result") or {})
        if adapted_manifest
        else {}
    )
    mobilegpt_prep = (
        {
            "type": "mobilegpt_native_source_cold_memory",
            "stats": adapted_source_stats,
            "wall_sec": _coerce_float(adapted_source_stats_record.get("wall_sec")),
            "official_validator_success": adapted_official_result.get(
                "official_validator_success"
            ),
            "manifest_path": str(adapted_memory.get("manifest_path") or ""),
            "manifest_sha256": str(adapted_memory.get("manifest_sha256") or ""),
            "memory_sha256": str(adapted_memory.get("memory_sha256") or ""),
            "shared_across_targets": True,
        }
        if adapted_memory
        else {}
    )

    _write_method_memory_manifest(
        memory_root=memory_root,
        task=item.task,
        method=method,
        memory_mode=f"mobilegpt_single_episode_{memory_condition}",
        source_seed=item.replay_seed,
        evaluation_seed=task_seed,
        attempt_id=attempt_id,
        target_inputs_read=False,
        artifacts={
            "runner": "stock_mobilegpt_single_episode",
            "initial_memory_condition": memory_condition,
            "episode_memory_policy": (
                "isolated_attempt_copy_on_write"
                if source_memory_root is not None
                else "mutable_empty_memory"
            ),
            "source_memory_root": str(source_memory_root or ""),
            "source_memory_sha256": source_memory_digest or None,
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

    if not args.dry_run:
        for path in _patch_mobilegpt_stats(mobilegpt_root=args.mobilegpt_root):
            print(f"[mobilegpt:patch-stats] {path}", flush=True)
    if not args.dry_run:
        for path in _patch_mobilegpt_server_runtime_context(
            mobilegpt_root=args.mobilegpt_root,
        ):
            print(f"[mobilegpt:patch-server-runtime] {path}", flush=True)

    records: list[dict[str, Any]] = []
    failed = 0
    condition_memory_root = source_memory_root

    frozen_memory: dict[str, Any] | None = None
    if condition_memory_root is not None:
        if args.dry_run:
            frozen_memory = {
                "schema_version": "omniflow.mobilegpt_frozen_memory.v1",
                "source_memory_root": str(condition_memory_root),
                "frozen_memory_root": str(frozen_memory_root),
                "digest": "dry-run",
                "file_count": 0,
                "read_only": True,
            }
        else:
            frozen_memory = freeze_mobilegpt_memory(
                condition_memory_root,
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
                if frozen_memory is None:
                    episode_memory_root.mkdir()
                else:
                    prepare_mobilegpt_episode_memory(
                        frozen_memory_root,
                        episode_memory_root,
                        expected_digest=str(frozen_memory.get("digest") or ""),
                        expected_file_count=int(frozen_memory.get("file_count") or 0),
                    )
            server_spec = build_mobilegpt_command(
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
            if str(args.model or "").strip():
                server_spec = replace(
                    server_spec,
                    env={
                        **server_spec.env,
                        "MOBILEGPT_CHAT_MODEL": str(args.model).strip(),
                    },
                    metadata={
                        **server_spec.metadata,
                        "model": str(args.model).strip(),
                    },
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
                episode_spec = build_mobilegpt_androidworld_command(
                    item,
                    method_name=method,
                    target=target,
                    android_world_root=args.android_world_root,
                    output_root=output_root,
                    stats_jsonl=stats_jsonl,
                    server_host=args.mobilegpt_server_host,
                    server_port=int(args.mobilegpt_port),
                    target_package=target_package,
                    max_steps=int(args.max_steps or 20),
                    task_random_seed=task_seed,
                    fixed_task_seed=not bool(args.no_fixed_task_seed),
                    fixed_task_params=not bool(args.no_fixed_task_params),
                    task_params_override=task_params_override,
                    perform_emulator_setup=bool(args.perform_emulator_setup),
                    adb_path=args.adb_path,
                    start_timeout_sec=float(args.mobilegpt_wait_start_timeout_sec),
                    finish_timeout_sec=float(args.mobilegpt_episode_wait_timeout_sec),
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
                            if frozen_memory is not None
                            else "isolated_attempt_empty_memory"
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

            mobilegpt_summary = summarize_mobilegpt_stats(stats_jsonl)
            stats_summary_path.write_text(
                json.dumps(mobilegpt_summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if frozen_memory is not None:
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
            else:
                memory_inventory = inspect_mobilegpt_memory(episode_memory_root)
                cold_frozen_root = episode_root / "frozen_memory"
                if memory_inventory.get("has_useful_actions"):
                    cold_snapshot = freeze_mobilegpt_memory(
                        episode_memory_root,
                        cold_frozen_root,
                    )
                    records.append(
                        {
                            "label": "mobilegpt:capture-cold-memory",
                            "returncode": 0,
                            "output_path": str(cold_frozen_root),
                            "command": "",
                            "task": item.task,
                            "method": method,
                            "device": target.label,
                            "status": "cold_memory_captured",
                            "summary_exclude": True,
                            "metadata": {
                                "memory_root": str(memory_root),
                                "mobilegpt_memory_inventory": memory_inventory,
                                "frozen_memory": cold_snapshot,
                            },
                        }
                    )

            if failed and args.fail_fast:
                break
    finally:
        _stop_background_command(browser_task_server)

    if frozen_memory is not None and not args.dry_run:
        frozen_digest, frozen_file_count = _mobilegpt_memory_digest(frozen_memory_root)
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


def cmd_one_task(args: argparse.Namespace) -> int:
    selected = _select_from_args(args)
    if len(selected) != 1:
        raise ValueError("one-task requires exactly one selected --tasks entry")
    item = selected[0]
    methods = _parse_one_task_methods(args.methods)
    targets = parse_device_targets(args.device_targets)
    source_memory_run_log = item.source_run_log
    attempt_root, _ = _task_managed_output_root(args.output_root)
    output_root = _source_seed_output_root(attempt_root, item.replay_seed)
    attempt_id = attempt_root.name
    task_params_override = _task_params_override_from_args(args)
    task_seed = (
        random.randint(1, 2**31 - 1)
        if bool(args.random_task_seed)
        else args.task_random_seed
    )
    attempt_manifest_path = _claim_one_task_attempt(
        attempt_root,
        task=item.task,
        methods=methods,
        source_seed=item.replay_seed,
        evaluation_seed=task_seed,
        task_iteration=int(args.task_iteration),
        baseline_environment_repair=str(args.baseline_environment_repair or ""),
        dry_run=bool(args.dry_run),
        experiment_config=getattr(args, "experiment_config_path", None),
    )
    command_records: list[dict[str, Any]] = []
    failed = 0

    for method in methods:
        memory_root = _method_memory_root(output_root, item.task, method)
        _claim_method_memory_root(memory_root)
        source_action_hint_path: Path | None = None
        appagent_docs_root: Path | None = None
        appagent_action_source: Path | None = None
        appagent_prep: dict[str, Any] = {}
        if _is_mobilegpt_method(method):
            mobilegpt_records, mobilegpt_failed = _run_one_task_mobilegpt(
                args=args,
                item=item,
                targets=targets,
                output_root=output_root,
                task_params_override=task_params_override,
                task_seed=task_seed,
                method=method,
                attempt_id=attempt_id,
                source_run_log=source_memory_run_log,
            )
            command_records.extend(mobilegpt_records)
            failed += mobilegpt_failed
            if failed and args.fail_fast:
                break
            continue

        if method == "ours":
            store_text = str(args.store_path or "").strip()
            if not store_text:
                raise ValueError(
                    f"--store-path is required when one-task includes {method}"
                )
            store_path = _repo_path(store_text)
        else:
            store_path = memory_root / "unused-store.json"

        if method == "ours":
            transfer_asset_audit: dict[str, Any] = {}
            if not args.dry_run:
                transfer_asset_audit = validate_ours_transfer_assets(
                    store_path,
                    require_action_transfer=True,
                )
            _write_method_memory_manifest(
                memory_root=memory_root,
                task=item.task,
                method=method,
                memory_mode="omniflow_function_store",
                source_seed=item.replay_seed,
                evaluation_seed=task_seed,
                attempt_id=attempt_id,
                source_run_log=item.source_run_log,
                artifacts={
                    "store_path": str(store_path),
                    "store_sha256": _file_sha256(store_path)
                    if store_path.is_file()
                    else None,
                    "function_authoring": "external_agent_skill",
                    "transfer_asset_audit": transfer_asset_audit,
                    "transfer_state_catalog_sha256": (
                        _file_sha256(transfer_asset_audit["transfer_state_catalog"])
                        if transfer_asset_audit.get("transfer_state_catalog")
                        else None
                    ),
                },
            )
        if method in APPAGENT_METHODS:
            if method == "appagent_demo":
                source_memory_text = str(
                    getattr(args, "appagent_demo_memory_root", "") or ""
                ).strip()
                if not source_memory_text:
                    raise ValueError(
                        "appagent_demo requires --appagent-demo-memory-root"
                    )
                source_memory_root = _repo_path(source_memory_text)
                provenance = validate_appagent_demo_memory(
                    source_memory_root,
                    task_name=item.task,
                    source_run_log=source_memory_run_log,
                )
                appagent_docs_root = Path(provenance["demo_docs_root"]).resolve()
                appagent_action_source = Path(provenance["teacher_source"]).resolve()
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
                source_memory_manifest = (
                    source_memory_root / "appagent_demo_manifest.json"
                )
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
                    "manifest_sha256": _file_sha256(source_memory_manifest),
                    "demo_sha256": str(provenance.get("demo_sha256") or ""),
                    "demo_docs_sha256": str(provenance.get("demo_docs_sha256") or ""),
                    "shared_across_targets": True,
                }
                memory_mode = "appagent_native_demo_docs"
                artifacts = {
                    "source_memory_root": str(source_memory_root),
                    "source_memory_manifest": str(
                        source_memory_root / "appagent_demo_manifest.json"
                    ),
                    "source_memory_manifest_sha256": _file_sha256(
                        source_memory_root / "appagent_demo_manifest.json"
                    ),
                    "demo_docs_root": str(appagent_docs_root),
                    "demo_docs_sha256": provenance["demo_docs_sha256"],
                    "action_source": str(appagent_action_source),
                    "action_source_sha256": provenance["teacher_source_sha256"],
                    "official_appagent_revision": provenance[
                        "official_appagent_revision"
                    ],
                    "uses_appagent_demo_docs": True,
                    "uses_omniflow_function": False,
                    "memory_read_only": True,
                }
            else:
                memory_mode = "none"
                artifacts = {
                    "official_appagent_revision": (
                        "2c1900422caf6f9e94e96d5dd984b530e5a5fbf8"
                    ),
                    "uses_appagent_demo_docs": False,
                    "uses_omniflow_function": False,
                }
            _write_method_memory_manifest(
                memory_root=memory_root,
                task=item.task,
                method=method,
                memory_mode=memory_mode,
                source_seed=(item.replay_seed if method == "appagent_demo" else None),
                evaluation_seed=task_seed,
                attempt_id=attempt_id,
                source_run_log=(
                    source_memory_run_log if method == "appagent_demo" else None
                ),
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
                source_seed=item.replay_seed,
                evaluation_seed=task_seed,
                attempt_id=attempt_id,
                source_run_log=item.source_run_log,
                artifacts={
                    "source_run_log": str(item.source_run_log),
                    "source_run_log_sha256": _file_sha256(item.source_run_log),
                    "replay_run_log": str(replay_run_log),
                    "replay_run_log_sha256": _file_sha256(replay_run_log),
                    "source_materialization": replay_materialization,
                    "source_format": replay_profile.replay_format,
                    "replay_memory_root": str(memory_root),
                },
            )
        elif method == "t3a_hint":
            official_agent_name = "t3a_gpt4"
            source_hint_store_path = (
                _repo_path(str(args.store_path))
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
                source_seed=item.replay_seed,
                evaluation_seed=task_seed,
                attempt_id=attempt_id,
                source_run_log=item.source_run_log,
                artifacts={
                    "source_run_log": str(item.source_run_log),
                    "source_run_log_sha256": _file_sha256(item.source_run_log),
                    "source_action_hint_path": str(source_action_hint_path),
                    "source_action_hint_sha256": _file_sha256(source_action_hint_path),
                    "semantic_source": (
                        "omniflow_function_store"
                        if source_hint_store_path is not None
                        else "source_run_log"
                    ),
                    "source_store": str(source_hint_store_path or ""),
                    "source_store_sha256": (
                        _file_sha256(source_hint_store_path)
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
                spec = build_fixed_replay_command(
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
            elif method in APPAGENT_METHODS:
                spec = build_appagent_androidworld_command(
                    item,
                    method_name=method,
                    target=target,
                    android_world_root=args.android_world_root,
                    output_root=output_root,
                    appagent_root=args.appagent_root,
                    docs_root=appagent_docs_root,
                    action_source=appagent_action_source,
                    max_steps=int(args.max_steps or 20),
                    timeout_sec=int(args.timeout_sec or 0),
                    task_random_seed=task_seed,
                    fixed_task_seed=not bool(args.no_fixed_task_seed),
                    fixed_task_params=not bool(args.no_fixed_task_params),
                    task_params_override=task_params_override,
                    perform_emulator_setup=bool(args.perform_emulator_setup),
                    adb_path=args.adb_path,
                )
            elif method == "t3a_hint":
                spec = build_official_androidworld_command(
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
                    max_steps=int(args.max_steps or 20),
                    timeout_sec=int(args.timeout_sec or 0),
                    task_random_seed=task_seed,
                    fixed_task_seed=not bool(args.no_fixed_task_seed),
                    fixed_task_params=not bool(args.no_fixed_task_params),
                    task_params_override=task_params_override,
                    perform_emulator_setup=bool(args.perform_emulator_setup),
                )
            else:
                spec = build_e2e_command(
                    item,
                    android_world_root=args.android_world_root,
                    output_root=output_root,
                    method_name=method,
                    device_label=target.label,
                    serial=target.serial,
                    console_port=target.console_port,
                    adb_path=args.adb_path,
                    max_steps=int(args.max_steps or 20),
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
        for path in _one_task_formal_result_paths(record)
    ]
    aggregate_summary = aggregate_task_results([] if args.dry_run else aggregate_paths)
    source_runlog_summary: dict[str, Any] = {}
    if not bool(args.dry_run) and bool(
        getattr(args, "save_success_source_runlog", False)
    ):
        source_runlog_summary = save_success_source_runlogs_from_results(
            aggregate_paths,
            output_root=output_root,
        )
        aggregate_summary["success_source_runlogs"] = source_runlog_summary
    summary = _write_one_task_summary(
        output_root=output_root,
        task=item.task,
        command_records=command_records,
        aggregate_summary=aggregate_summary,
    )
    result_registration: dict[str, Any] = {}
    summary_path = output_root / _safe_stem(item.task) / "one_task_summary.json"
    if not bool(args.dry_run):
        result_registry_root, master_progress_root = _result_registration_roots(
            args,
            attempt_root=attempt_root,
        )
        result_registration = register_attempt_summary(
            summary_path=summary_path,
            attempt_manifest_path=attempt_manifest_path,
            runs_root=result_registry_root,
            master_root=master_progress_root,
            source_index_path=_repo_path(
                str(getattr(args, "master_source_index", "") or args.index)
            ),
            artifact_memory_index=Path(
                os.environ["OMNIFLOW_EXP_MEMORY_INDEX"]
            ).expanduser()
            if os.environ.get("OMNIFLOW_EXP_MEMORY_INDEX")
            else None,
        )
    _print_one_task_summary(summary)
    print(
        f"[one-task] summary={summary_path}",
        flush=True,
    )
    if result_registration:
        print(
            "[one-task] registered="
            f"{result_registration.get('registered_cells', 0)} "
            f"ledger_appended={result_registration.get('ledger_records_appended', 0)} "
            f"master={master_progress_root}",
            flush=True,
        )
    if source_runlog_summary:
        print(
            "[one-task] success_source_runlogs="
            f"{source_runlog_summary.get('saved_count', 0)} "
            f"index={source_runlog_summary.get('global_index_by_task') or ''}",
            flush=True,
        )
    return 1 if failed else 0


def cmd_mobilegpt(args: argparse.Namespace) -> int:
    if args.mobilegpt_action in {"server", "teach-server"}:
        patched_server = _patch_mobilegpt_server_runtime_context(
            mobilegpt_root=args.mobilegpt_root,
        )
        for path in patched_server:
            print(f"[mobilegpt:patch-server-runtime] {path}", flush=True)

    if bool(args.patch_stats):
        patched_stats = _patch_mobilegpt_stats(mobilegpt_root=args.mobilegpt_root)
        for path in patched_stats:
            print(f"[mobilegpt:patch-stats] {path}", flush=True)

    if args.mobilegpt_action == "stats":
        summary = summarize_mobilegpt_stats(args.stats_jsonl)
        output_path = _repo_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if args.json:
            print(json.dumps(summary, indent=2, ensure_ascii=False))
        else:
            print(
                "[mobilegpt:stats] "
                f"tasks={summary['task_finished_count']}/{summary['task_started_count']} "
                f"calls={summary['model_calls']} tokens={summary['total_tokens']} "
                f"summary={output_path}",
                flush=True,
            )
        return 0

    spec = build_mobilegpt_command(
        args.mobilegpt_action,
        mobilegpt_root=args.mobilegpt_root,
        serial=args.serial,
        adb_path=args.adb_path,
        server_host=args.server_host,
        port=args.port,
        stats_jsonl=args.stats_jsonl,
        source_run_log=args.source_run_log,
        fallback_to_vlm_on_teacher_miss=bool(args.fallback_to_vlm_on_teacher_miss),
    )
    return run_command(spec, dry_run=args.dry_run)


def cmd_metrics(args: argparse.Namespace) -> int:
    summary = aggregate_task_results(args.paths)
    write_metrics_summary(summary, args.output)
    print(
        "[metrics] "
        f"success={summary['success_count']}/{summary['success_total']} "
        f"replay_completed={summary['replay_completed_count']}/"
        f"{summary['replay_task_count']} "
        f"summary={_repo_path(args.output)}"
    )
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    selected = _select_from_args(args)
    specs = collect_androidworld_app_specs(
        selected,
        android_world_root=args.android_world_root,
        suite_family=args.suite_family,
    )
    installed_packages: set[str] | None = None
    if bool(args.check_device) or bool(str(args.serial or "").strip()):
        installed_packages = read_adb_installed_packages(
            serial=args.serial,
            adb_path=args.adb_path,
        )
    cached_apks = list_cached_androidworld_apks(args.apk_cache_dir or None)
    summary = build_device_readiness_summary(
        specs,
        installed_packages=installed_packages,
        cached_apks=cached_apks,
        cache_dir=args.apk_cache_dir or None,
    )
    if args.output:
        output_path = _repo_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    ready = summary.get("ready_task_count")
    ready_text = (
        f" ready_tasks={ready}/{summary['task_count']}" if ready is not None else ""
    )
    print(
        f"apps={summary['app_count']} tasks={summary['task_count']}{ready_text} "
        f"missing_packages={summary['missing_package_count']} "
        f"missing_cached_apks={summary['missing_cached_apk_count']}"
    )
    for row in summary.get("missing_packages") or []:
        print(
            f"missing-package app={row['app_name']} package={row['package_name']} "
            f"tasks={','.join(row['tasks'])}"
        )
    for apk in summary.get("missing_cached_apks") or []:
        print(f"missing-apk {apk}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and summarize normal AndroidWorld experiment episodes."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser(
        "list", help="List archived success source runlogs"
    )
    list_parser.add_argument("--index", default=str(DEFAULT_ARCHIVE_INDEX))
    list_parser.add_argument("--tasks", default="")
    list_parser.add_argument("--first60", action="store_true")
    list_parser.add_argument("--limit", type=int, default=None)
    list_parser.add_argument(
        "--source-format",
        choices=["all", "ready", "canonical", "payload"],
        default="all",
    )
    list_parser.add_argument("--accepted-first30", action="store_true")
    list_parser.add_argument("--json", action="store_true")
    list_parser.set_defaults(func=cmd_list)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect archived source runlog formats before replay",
    )
    inspect_parser.add_argument("--index", default=str(DEFAULT_ARCHIVE_INDEX))
    inspect_parser.add_argument("--tasks", default="")
    inspect_parser.add_argument("--first60", action="store_true")
    inspect_parser.add_argument("--limit", type=int, default=None)
    inspect_parser.add_argument(
        "--source-format",
        choices=["all", "ready", "canonical", "payload"],
        default="all",
    )
    inspect_parser.add_argument("--accepted-first30", action="store_true")
    inspect_parser.add_argument("--json", action="store_true")
    inspect_parser.add_argument("--output", default="")
    inspect_parser.set_defaults(func=cmd_inspect)

    one_task_parser = subparsers.add_parser(
        "one-task",
        help=(
            "Run all configured AndroidWorld methods for exactly one archived "
            "task and write a compact per-method summary"
        ),
    )
    one_task_parser.add_argument("--index", default=str(DEFAULT_ARCHIVE_INDEX))
    one_task_parser.add_argument(
        "--master-source-index",
        default="",
        help=(
            "Complete canonical task index used to materialize master progress. "
            "Defaults to --index; task-limited execution indexes must override it."
        ),
    )
    one_task_parser.add_argument(
        "--android-world-root", default=str(DEFAULT_ANDROID_WORLD_ROOT)
    )
    one_task_parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help=(
            "Exact fresh immutable attempt directory. The shared canonical "
            "runtime/evals/androidworld_validator/runs root is rejected."
        ),
    )
    one_task_parser.add_argument(
        "--result-registry-root",
        default=os.environ.get("ANDROIDWORLD_RESULT_REGISTRY_ROOT", ""),
        help=(
            "Canonical immutable task/method/device/attempt registry. Defaults "
            "to the androidworld_validator root containing --index."
        ),
    )
    one_task_parser.add_argument(
        "--master-progress-root",
        default=os.environ.get("ANDROIDWORLD_MASTER_PROGRESS_ROOT", ""),
        help=(
            "Canonical master table root updated after result registration. "
            "Defaults beside --result-registry-root."
        ),
    )
    one_task_parser.add_argument(
        "--experiment-config",
        default="",
        help=(
            "JSON/YAML defaults for this one-task experiment. Command-line "
            "arguments override config values."
        ),
    )
    one_task_parser.add_argument("--tasks", required=True)
    one_task_parser.add_argument(
        "--task-iteration",
        type=int,
        choices=range(1, 4),
        default=1,
        help=(
            "Immutable task-level revision number. Formal and diagnostic task "
            "revisions are capped at three; setup/preflight is recorded separately."
        ),
    )
    one_task_parser.add_argument(
        "--baseline-environment-repair",
        default="",
        help=(
            "Audit reason for rerunning a frozen baseline after an environment "
            "failure. This never authorizes changing baseline method details."
        ),
    )
    one_task_parser.add_argument(
        "--methods",
        default="all",
        help=(
            "Comma-separated methods or `all`: fixed_replay, ours, "
            "mobilegpt_offline_retrieval, appagent_demo, and t3a_hint."
        ),
    )
    one_task_parser.add_argument(
        "--store-path",
        default="",
        help=("Validated omniflow.store.v2 required by the OmniFlow methods."),
    )
    one_task_parser.add_argument(
        "--store-index",
        default="",
        help="Canonical task-to-Store index used by frozen source assets.",
    )
    one_task_parser.add_argument(
        "--omnitransfer-root",
        default="",
        help=(
            "Canonical or versioned OmniTransfer repository root. The configured "
            "root is authoritative over installed Python packages."
        ),
    )
    one_task_parser.add_argument(
        "--device-targets",
        default=DEFAULT_DEVICE_TARGETS,
        help="Comma-separated LABEL:SERIAL:PORT entries.",
    )
    one_task_parser.add_argument(
        "--source-format",
        choices=["all", "ready", "canonical", "payload"],
        default="all",
    )
    one_task_parser.add_argument("--accepted-first30", action="store_true")
    one_task_parser.add_argument("--limit", type=int, default=None)
    one_task_parser.add_argument("--adb-path", default="")
    one_task_parser.add_argument("--max-steps", type=int, default=20)
    one_task_parser.add_argument(
        "--max-fallback-steps",
        type=int,
        default=None,
        help=(
            "Maximum VLM fallback planner calls for ours. Function actions do not "
            "consume this budget; omitted means the normal max-step behavior."
        ),
    )
    one_task_parser.add_argument("--timeout-sec", type=int, default=180)
    one_task_parser.add_argument(
        "--task-random-seed",
        type=int,
        default=DEFAULT_EVAL_TASK_RANDOM_SEED,
        help="AndroidWorld warm/target seed. Formal experiments use 113.",
    )
    one_task_parser.add_argument("--random-task-seed", action="store_true")
    one_task_parser.add_argument("--no-fixed-task-seed", action="store_true")
    one_task_parser.add_argument("--no-fixed-task-params", action="store_true")
    one_task_parser.add_argument(
        "--task-params-json",
        default="",
        help="Override archived task params with this JSON object.",
    )
    one_task_parser.add_argument("--planner-provider", default="")
    one_task_parser.add_argument("--model", default="")
    one_task_parser.add_argument("--planner-timeout-sec", type=float, default=60.0)
    _add_androidworld_setup_args(one_task_parser)
    one_task_parser.add_argument(
        "--mobilegpt-root", default=str(DEFAULT_MOBILEGPT_ROOT)
    )
    one_task_parser.add_argument(
        "--appagent-root",
        default=str(REPO_ROOT / "runtime" / "external" / "appagent"),
    )
    one_task_parser.add_argument(
        "--appagent-demo-memory-root",
        default="",
        help=(
            "Sealed source-111 AppAgent human-demo workspace required by appagent_demo."
        ),
    )
    one_task_parser.add_argument("--mobilegpt-server-host", default="0.0.0.0")
    one_task_parser.add_argument("--mobilegpt-port", type=int, default=12345)
    one_task_parser.add_argument(
        "--mobilegpt-server-warmup-sec", type=float, default=5.0
    )
    one_task_parser.add_argument(
        "--mobilegpt-wait-start-timeout-sec",
        type=float,
        default=DEFAULT_MOBILEGPT_WAIT_START_TIMEOUT_SEC,
        help=(
            "Seconds to wait for the native MobileGPT server connection. "
            "Use -1 to wait indefinitely."
        ),
    )
    one_task_parser.add_argument(
        "--mobilegpt-episode-wait-timeout-sec",
        type=float,
        default=DEFAULT_MOBILEGPT_EPISODE_WAIT_TIMEOUT_SEC,
        help=(
            "Seconds to wait for MobileGPT task_finished before official "
            "AndroidWorld validation. Use -1 to wait indefinitely."
        ),
    )
    one_task_parser.add_argument("--mobilegpt-open-target-app", default="")
    one_task_parser.add_argument(
        "--mobilegpt-source-memory-root",
        default="",
        help=(
            "Optional source-only native MobileGPT memory. If omitted, the "
            "same episode runner starts cold from empty memory; if supplied, "
            "it starts warm from an immutable snapshot."
        ),
    )
    one_task_parser.add_argument(
        "--save-success-source-runlog",
        action="store_true",
        help=(
            "Explicitly materialize official-validator successful canonical runs "
            "inside this output root for later source-authoring workflows. Disabled "
            "by default so target evaluation remains read-only with respect to memory."
        ),
    )
    one_task_parser.add_argument("--dry-run", action="store_true")
    one_task_parser.add_argument("--fail-fast", action="store_true")
    one_task_parser.set_defaults(func=cmd_one_task)

    mobilegpt_parser = subparsers.add_parser(
        "mobilegpt",
        help="Run MobileGPT external baseline setup/actions through this unified script",
    )
    mobilegpt_parser.add_argument(
        "mobilegpt_action",
        choices=["server", "teach-server", "stats"],
        help=(
            "`server` starts MobileGPT Server/main.py; `teach-server` starts "
            "MobileGPT cold-start learning with AndroidWorld source actions as "
            "native teacher-forced DeriveAgent outputs; `stats` aggregates the "
            "server JSONL stats."
        ),
    )
    mobilegpt_parser.add_argument(
        "--mobilegpt-root",
        default=str(DEFAULT_MOBILEGPT_ROOT),
    )
    mobilegpt_parser.add_argument("--index", default=str(DEFAULT_ARCHIVE_INDEX))
    mobilegpt_parser.add_argument(
        "--android-world-root", default=str(DEFAULT_ANDROID_WORLD_ROOT)
    )
    mobilegpt_parser.add_argument("--tasks", default="")
    mobilegpt_parser.add_argument(
        "--source-format",
        choices=["all", "ready", "canonical", "payload"],
        default="all",
    )
    mobilegpt_parser.add_argument("--accepted-first30", action="store_true")
    mobilegpt_parser.add_argument("--limit", type=int, default=None)
    mobilegpt_parser.add_argument("--serial", default="")
    mobilegpt_parser.add_argument("--adb-path", default="")
    mobilegpt_parser.add_argument("--server-host", default="0.0.0.0")
    mobilegpt_parser.add_argument("--port", type=int, default=12345)
    mobilegpt_parser.add_argument("--max-steps", type=int, default=20)
    mobilegpt_parser.add_argument("--source-run-log", default="")
    mobilegpt_parser.add_argument(
        "--fallback-to-vlm-on-teacher-miss",
        action="store_true",
        help=(
            "Only for `teach-server`: if a source action cannot be migrated to "
            "the current MobileGPT screen, fall back to MobileGPT's original VLM "
            "DeriveAgent for that step. Default is fail-closed."
        ),
    )
    mobilegpt_parser.add_argument(
        "--stats-jsonl",
        default=str(DEFAULT_MOBILEGPT_STATS_JSONL),
        help="JSONL path used by patched MobileGPT server stats.",
    )
    mobilegpt_parser.add_argument(
        "--output",
        default=str(DEFAULT_MOBILEGPT_STATS_SUMMARY),
        help="Output path for `mobilegpt stats` summary JSON.",
    )
    mobilegpt_parser.add_argument("--json", action="store_true")
    mobilegpt_parser.add_argument(
        "--patch-stats",
        action="store_true",
        help=(
            "Patch MobileGPT Server utils/mobilegpt modules to write OpenAI usage "
            "and task finish events to --stats-jsonl."
        ),
    )
    mobilegpt_parser.add_argument("--dry-run", action="store_true")
    mobilegpt_parser.set_defaults(func=cmd_mobilegpt)

    metrics_parser = subparsers.add_parser(
        "metrics", help="Aggregate task_results.jsonl files"
    )
    metrics_parser.add_argument("paths", nargs="+")
    metrics_parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_ROOT / "aggregate_summary.json"),
    )
    metrics_parser.set_defaults(func=cmd_metrics)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check archive task app requirements against the target device/cache",
    )
    doctor_parser.add_argument("--index", default=str(DEFAULT_ARCHIVE_INDEX))
    doctor_parser.add_argument(
        "--android-world-root", default=str(DEFAULT_ANDROID_WORLD_ROOT)
    )
    doctor_parser.add_argument("--suite-family", default="android_world")
    doctor_parser.add_argument("--tasks", default="")
    doctor_parser.add_argument("--first60", action="store_true")
    doctor_parser.add_argument("--limit", type=int, default=None)
    doctor_parser.add_argument(
        "--source-format",
        choices=["all", "ready", "canonical", "payload"],
        default="all",
    )
    doctor_parser.add_argument("--accepted-first30", action="store_true")
    doctor_parser.add_argument("--serial", default="")
    doctor_parser.add_argument("--adb-path", default="")
    doctor_parser.add_argument("--check-device", action="store_true")
    doctor_parser.add_argument("--apk-cache-dir", default="")
    doctor_parser.add_argument("--json", action="store_true")
    doctor_parser.add_argument("--output", default="")
    doctor_parser.set_defaults(func=cmd_doctor)

    return parser


def _has_cli_option(argv: Sequence[str], option: str) -> bool:
    prefix = f"{option}="
    return any(token == option or token.startswith(prefix) for token in argv)


def _normalize_default_one_task_argv(raw_argv: Sequence[str]) -> list[str]:
    argv = list(raw_argv)
    if not argv:
        return argv
    first = str(argv[0])
    if first in TOP_LEVEL_COMMANDS or first in {"-h", "--help"}:
        return argv
    if not _has_cli_option(argv, "--tasks"):
        return argv

    normalized = ["one-task"]
    if not _has_cli_option(argv, "--experiment-config"):
        normalized.extend(["--experiment-config", "androidworld_eval.json"])
    if not _has_cli_option(argv, "--methods"):
        normalized.extend(["--methods", "all"])
    normalized.extend(argv)
    return normalized


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    raw_argv = _normalize_default_one_task_argv(raw_argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    args = apply_experiment_config(args, raw_argv, parser=parser)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
