"""Bounded source-to-result AndroidWorld task pipeline."""

from __future__ import annotations

import argparse
import concurrent.futures
from contextlib import contextmanager
import csv
import dataclasses
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Any, Callable, Sequence

from omniflow.core.trajectory import require_complete_source_run_log
from src.experiment.androidworld_paths import (
    canonical_device_model,
    canonical_device_seed_name,
    next_attempt_name,
)
from src.experiment.data_index import (
    canonical_prepared_memory_from_index,
    load_data_index,
    refresh_data_index_from_pointer,
    register_source_run_log_success,
)
from src.experiment.function_v2 import compile_function_v2
from src.experiment.mobilegpt_contract import (
    MOBILEGPT_EMBEDDING_MODEL,
    MOBILEGPT_SOURCE_METHOD_BY_SCHEMA,
)
from src.experiment.observation_evidence import (
    build_androidworld_run_log,
    persist_androidworld_run_log,
)
from src.experiment.paths import resolve_path, safe_component
from src.experiment.protocol import (
    ANDROIDWORLD_REVISION,
    APPAGENT_MODEL,
    BMOCA_RESULT_TIMEOUT_SEC,
    DEFAULT_DEVICE,
    DEFAULT_METHOD,
    DEFAULT_TASK,
    DEVICES,
    FIXED_TASK_PARAMS,
    FORMAL_MODEL,
    FUNCTION_ENHANCEMENT_TIMEOUT_SEC,
    MAX_FALLBACK_STEPS,
    MAX_STEPS,
    METHODS,
    OMNIFLOW_PLANNER_MODEL,
    SOURCE_AVD,
    SOURCE_DEVICE,
    SOURCE_SEED,
    STEP_TIMEOUT_SEC,
    SUPPLEMENTAL_DEVICES,
    SUPPLEMENTAL_METHODS,
    SUPPLEMENTAL_RESULTS_NAMESPACE,
    TASK_DEADLINE_SEC,
    TASK_SEED,
    VALIDATOR_FLUSH_GRACE_SEC,
)
from src.experiment.run_process import (
    run_process,
)
from src.experiment.run_task import (
    build_replay_command,
    validate_omniflow_transfer_assets,
)
from src.experiment.source_records import CanonicalRunLog
from src.integrations.mobilegpt import (
    convert_runlog_to_mobilegpt_bundle,
    convert_runlog_to_mobilegpt_memory,
)
from src.integrations.official_forward import validate_autodroid_memory_root
from src.integrations.runlog import project_androidworld_step_actions
from src.integrations.skilldroid_replay import compile_droidrun_macro


@contextmanager
def _manual_source_device_lock(results_root: Path):
    """Serialize the complete manual source-device lifecycle across processes."""

    lock_path = (
        Path(results_root).expanduser().resolve()
        / "androidworld"
        / ".manual_source_device.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _read_object(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{resolved}")
    return value


def _generate_missing_androidworld_task_params(
    *, task: str, source_seed: int | None
) -> dict[str, Any]:
    """Recreate generated task parameters when the source index lost them."""

    from android_world import registry

    task_type = registry.TaskRegistry().get_registry(family="android_world").get(task)
    if task_type is None:
        raise ValueError(f"unknown_androidworld_task:{task}")
    generator = getattr(task_type, "generate_random_params", None)
    if not callable(generator):
        raise ValueError(f"androidworld_task_params_missing:{task}")
    random_state = random.getstate()
    try:
        random.seed(int(source_seed if source_seed is not None else SOURCE_SEED))
        generated = generator()
    finally:
        random.setstate(random_state)

    def jsonable(value: Any) -> Any:
        if dataclasses.is_dataclass(value):
            return {
                key: jsonable(item)
                for key, item in dataclasses.asdict(value).items()
            }
        if isinstance(value, dict):
            return {str(key): jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [jsonable(item) for item in value]
        return value

    result = jsonable(generated)
    if not isinstance(result, dict):
        raise ValueError(f"androidworld_task_params_invalid:{task}")
    return result


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


class Deadline:
    def __init__(self, seconds: int) -> None:
        if seconds <= 0:
            raise ValueError("task_deadline_must_be_positive")
        self.seconds = int(seconds)
        self.started = time.monotonic()

    def remaining(self, cap: float | None = None) -> float:
        value = max(0.0, self.seconds - (time.monotonic() - self.started))
        return min(value, float(cap)) if cap is not None else value

    @property
    def expired(self) -> bool:
        return self.remaining() <= 0

    @property
    def elapsed(self) -> float:
        return round(time.monotonic() - self.started, 6)


class PipelinePhaseError(RuntimeError):
    def __init__(self, message: str, phase: dict[str, Any]) -> None:
        super().__init__(message)
        self.phase = dict(phase)


def run_logged_command(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
    timeout_sec: float,
) -> dict[str, Any]:
    """Run one child through the shared experiment process lifecycle."""

    return run_process(
        command,
        cwd=cwd,
        environment=environment,
        log_path=log_path,
        stdin_devnull=True,
        timeout_sec=timeout_sec,
    )


def _usage_from_result(row: dict[str, Any]) -> dict[str, int]:
    return {
        "model_calls": int(row.get("model_calls") or 0),
        "prompt_tokens": int(row.get("prompt_tokens") or 0),
        "completion_tokens": int(row.get("completion_tokens") or 0),
        "total_tokens": int(row.get("total_tokens") or 0),
    }


def _perform_androidworld_emulator_setup() -> bool:
    return (
        str(os.environ.get("OMNIFLOW_ANDROIDWORLD_PERFORM_EMULATOR_SETUP", "1"))
        .strip()
        .lower()
        not in {"0", "false", "no", "off"}
    )


def _usage_accounting_status(row: dict[str, Any]) -> str:
    return "tracked" if row else "unavailable"


def _last_jsonl_row(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows[-1] if rows else {}


def _canonical_source(
    memory_index: Path,
    task: str,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    registry = load_data_index(memory_index)
    record = registry.get("canonical", {}).get("source_run_logs", {}).get(task)
    if not isinstance(record, dict):
        raise ValueError(f"canonical_source_missing:{task}")
    path = Path(str(record.get("object_path") or "")).expanduser().resolve()
    run_log = require_complete_source_run_log(_read_object(path))
    return registry, path, run_log


def _replayable_historical_source(
    memory_index: Path,
    task: str,
) -> tuple[Path, dict[str, Any]]:
    source_root = memory_index.parent / "androidworld" / safe_component(task) / "source"
    candidates: list[tuple[tuple[int, int, int], Path, dict[str, Any]]] = []
    for path in source_root.glob("*/runlog/*/run_log.json"):
        if ".archive" in path.parts:
            continue
        try:
            run_log = require_complete_source_run_log(_read_object(path))
        except (OSError, TypeError, ValueError):
            continue
        validator = run_log.get("validator")
        official_success = int(
            isinstance(validator, dict)
            and validator.get("official") is True
            and validator.get("success") is True
        )
        if (
            run_log.get("task_name") != task
            or run_log.get("success") is not True
            or not run_log.get("steps")
            or not official_success
        ):
            continue
        candidates.append(
            (
                (
                    official_success,
                    int(run_log.get("finished_at_ms") or 0),
                    path.stat().st_mtime_ns,
                ),
                path.resolve(),
                run_log,
            )
        )
    if not candidates:
        raise ValueError(f"replayable_historical_source_missing:{task}")
    _, path, run_log = max(candidates, key=lambda value: value[0])
    return path, run_log


def _appagent_source_path(
    memory_index: Path,
    task: str,
    source_path: Path,
) -> Path:
    try:
        source = _read_object(source_path)
        steps = source.get("steps") if isinstance(source, dict) else None
        last_step = steps[-1] if isinstance(steps, list) and steps else None
        has_after_observation = isinstance(source.get("final_observation"), dict) or (
            isinstance(last_step, dict)
            and isinstance(last_step.get("next_observation"), dict)
        )
    except (OSError, TypeError, ValueError):
        has_after_observation = False
    if has_after_observation:
        return source_path
    source_root = memory_index.parent / "androidworld" / safe_component(task) / "source"
    candidates: list[tuple[tuple[int, int], Path]] = []
    for candidate_path in source_root.glob("*/runlog/*/run_log.json"):
        if ".archive" in candidate_path.parts:
            continue
        try:
            candidate = _read_object(candidate_path)
        except (OSError, TypeError, ValueError):
            continue
        steps = candidate.get("steps") if isinstance(candidate, dict) else None
        last_candidate_step = (
            steps[-1] if isinstance(steps, list) and steps else None
        )
        candidate_has_after = isinstance(
            candidate.get("final_observation") if isinstance(candidate, dict) else None,
            dict,
        ) or (
            isinstance(last_candidate_step, dict)
            and isinstance(last_candidate_step.get("next_observation"), dict)
        )
        validator = candidate.get("validator") if isinstance(candidate, dict) else None
        if (
            not isinstance(candidate, dict)
            or candidate.get("task_name") != task
            or candidate.get("success") is not True
            or not isinstance(steps, list)
            or not steps
            or not candidate_has_after
            or not isinstance(validator, dict)
            or validator.get("official") is not True
            or validator.get("success") is not True
        ):
            continue
        candidates.append(
            (
                (
                    int(candidate.get("finished_at_ms") or 0),
                    candidate_path.stat().st_mtime_ns,
                ),
                candidate_path.resolve(),
            )
        )
    if not candidates:
        return source_path
    return max(candidates, key=lambda value: value[0])[1]


def _audited_historical_source(
    memory_index: Path,
    task: str,
) -> tuple[Path, dict[str, Any]] | None:
    task_root = memory_index.parent / task
    qualification_path = task_root / "source_qualification.json"
    if not qualification_path.is_file():
        return None
    qualification = _read_object(qualification_path)
    latest = qualification.get("latest_attempt")
    if not isinstance(latest, dict):
        return None
    if (
        latest.get("mature") is not True
        or latest.get("official_validator_success") is not True
        or latest.get("replay_status") != "succeeded"
        or int(latest.get("model_calls") or 0) != 0
        or int(latest.get("fallback_steps") or 0) != 0
    ):
        return None
    attempt_id = str(latest.get("attempt_id") or "").strip()
    if not attempt_id:
        return None
    attempt_roots = (task_root / "source_attempts").glob("*/" + attempt_id)
    candidates = sorted(
        path
        for attempt_root in attempt_roots
        for pattern in ("**/run_log.json", "**/target.run_log.json")
        for path in attempt_root.glob(pattern)
        if path.is_file()
    )
    for candidate in candidates:
        try:
            run_log = require_complete_source_run_log(_read_object(candidate))
        except (TypeError, ValueError):
            continue
        if (
            run_log["task_name"] == task
            and run_log["success"] is True
        ):
            return candidate.resolve(), run_log
    return None


def _official_success(row: dict[str, Any]) -> bool:
    validator = row.get("androidworld_validator_result")
    return bool(
        row.get("official_validator_success") is True
        or isinstance(validator, dict)
        and validator.get("success") is True
    )


def _captured_androidworld_state(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("fixed_replay_capture_observation_required")
    state = record.get("androidworld_state")
    if not isinstance(state, dict):
        raise ValueError("fixed_replay_capture_androidworld_state_required")
    pixels = state.get("screenshot")
    if pixels is None:
        pixels = state.get("pixels")
    if not isinstance(pixels, dict):
        raise ValueError("fixed_replay_capture_screenshot_required")
    screenshot = Path(str(pixels.get("path") or "")).expanduser().resolve()
    if not screenshot.is_file():
        raise FileNotFoundError(f"fixed_replay_capture_screenshot_missing:{screenshot}")
    canonical = json.loads(json.dumps(state, ensure_ascii=False))
    if "screenshot" in state:
        canonical["screenshot"] = dict(pixels)
        canonical.pop("pixels", None)
    else:
        canonical["pixels"] = dict(pixels)
    return canonical


def _fixed_replay_source_step_width(source_step: dict[str, Any]) -> int:
    """Return the number of raw AndroidWorld actions for one semantic step."""

    action = source_step.get("action")
    action_type = str(action.get("action_type") or "") if isinstance(action, dict) else ""
    if action_type in {"status", "unknown"}:
        return 0
    if action_type == "answer":
        return 1
    return len(project_androidworld_step_actions(source_step))


def _next_source_attempt_id(args: argparse.Namespace) -> str:
    runlog_root = (
        Path(args.results_root).expanduser().resolve()
        / "androidworld"
        / safe_component(args.task)
        / "source"
        / f"{safe_component(args.source_avd)}_seed{SOURCE_SEED}"
        / "runlog"
    )
    return next_attempt_name(runlog_root)


def _next_result_attempt_id(
    args: argparse.Namespace,
    *,
    method: str,
    device: tuple[str, str, int],
) -> str:
    label, serial, port = device
    runlog_root = (
        Path(args.results_root).expanduser().resolve()
        / "androidworld"
        / safe_component(args.task)
        / safe_component(method)
        / canonical_device_seed_name(
            label=label,
            serial=serial,
            console_port=port,
            source_seed=_e2e_source_seed(args),
            evaluation_seed=_e2e_evaluation_seed(args),
        )
        / "runlog"
    )
    return next_attempt_name(runlog_root)


def _next_pipeline_attempt_id(
    args: argparse.Namespace,
    outcomes_root: Path,
) -> str:
    """Allocate an attempt id that is new in both output and outcome roots."""

    task_root = Path(args.output_root) / safe_component(args.task)
    candidate = str(
        getattr(args, "attempt_id", "") or next_attempt_name(task_root)
    )
    methods = _e2e_methods(args)
    devices = _e2e_devices(args)
    while True:
        output_collision = (task_root / candidate).exists()
        outcome_path_collision = any(
            (
                outcomes_root
                / safe_component(args.task)
                / safe_component(method)
                / safe_component(device[0])
                / candidate
            ).exists()
            for method in methods
            for device in devices
        )
        if not output_collision and not outcome_path_collision:
            return candidate
        match = re.fullmatch(r"(.*?)(\d+)", candidate)
        if match:
            prefix, number = match.groups()
            candidate = f"{prefix}{int(number) + 1:0{len(number)}d}"
        else:
            candidate = f"{candidate}_r2"


def _captured_source_run_log(
    *,
    source_path: Path,
    source_run_log: dict[str, Any],
    raw_replay_result: Path,
    task_result: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    replay = _read_object(raw_replay_result)
    trace = replay.get("execution_trace")
    trace_steps = trace.get("steps") if isinstance(trace, dict) else None
    provider = (
        trace_steps[0].get("provider_detail")
        if isinstance(trace_steps, list)
        and trace_steps
        and isinstance(trace_steps[0], dict)
        else None
    )
    raw = provider.get("raw_replay") if isinstance(provider, dict) else None
    captured_steps = raw.get("step_results") if isinstance(raw, dict) else None
    source_steps = list(source_run_log.get("steps") or ())
    if replay.get("completed") is not True or replay.get("replay_completed") is not True:
        raise ValueError("fixed_replay_capture_not_completed")
    widths = [_fixed_replay_source_step_width(step) for step in source_steps]
    expected_raw_steps = sum(widths)
    if not isinstance(captured_steps, list) or len(captured_steps) != expected_raw_steps:
        raise ValueError(
            "fixed_replay_capture_step_count_mismatch:"
            f"expected={expected_raw_steps}:actual="
            f"{len(captured_steps) if isinstance(captured_steps, list) else 0}"
        )
    final_observation = _captured_androidworld_state(
        raw.get("final_observation") if isinstance(raw, dict) else None
    )
    semantic_observations: list[dict[str, Any] | None] = []
    raw_index = 0
    for source_index, width in enumerate(widths):
        if width == 0:
            semantic_observations.append(None)
            continue
        first_raw_step = captured_steps[raw_index]
        for offset in range(width):
            raw_step = captured_steps[raw_index + offset]
            if not isinstance(raw_step, dict) or raw_step.get("completed") is not True:
                raise ValueError(
                    f"fixed_replay_capture_step_failed:{source_index}:{offset}"
                )
        semantic_observations.append(
            _captured_androidworld_state(first_raw_step.get("observation_before_act"))
        )
        raw_index += width
    if raw_index != len(captured_steps):
        raise ValueError(
            "fixed_replay_capture_raw_step_accounting_mismatch:"
            f"consumed={raw_index}:actual={len(captured_steps)}"
        )
    for index, observation in enumerate(semantic_observations):
        if observation is not None:
            continue
        next_observation = next(
            (
                candidate
                for candidate in semantic_observations[index + 1 :]
                if candidate is not None
            ),
            final_observation,
        )
        semantic_observations[index] = next_observation
    observations = [
        observation if observation is not None else final_observation
        for observation in semantic_observations
    ]
    validator = task_result.get("androidworld_validator_result")
    reward = validator.get("reward") if isinstance(validator, dict) else None
    if not isinstance(reward, (int, float)) or isinstance(reward, bool):
        raise ValueError("fixed_replay_capture_validator_reward_required")
    steps: list[dict[str, Any]] = []
    for index, source_step in enumerate(source_steps):
        next_observation = (
            observations[index + 1]
            if index + 1 < len(observations)
            else final_observation
        )
        steps.append(
            {
                "step_index": index,
                "observation": observations[index],
                "action": dict(source_step["action"]),
                "result": {"success": True},
                "next_observation": next_observation,
                "metadata": {
                    "capture": "fixed_replay",
                    "source_step_index": int(source_step["step_index"]),
                    "reasoning": str(
                        (source_step.get("metadata") or {}).get("reasoning")
                        or (
                            "Replay the corresponding action from the official "
                            "successful historical source and verify it against "
                            "the freshly captured native observation."
                        )
                    ),
                    "screenshot_path": str(
                        (
                            next_observation.get("screenshot")
                            or next_observation.get("pixels")
                            or {}
                        ).get("path")
                        or ""
                    ),
                },
            }
        )
    captured = build_androidworld_run_log(
        run_id=str(replay.get("run_id") or "fixed_replay_capture"),
        task_name=source_run_log["task_name"],
        goal=source_run_log["goal"],
        task_parameters=dict(source_run_log["task_parameters"]),
        seed=SOURCE_SEED,
        validator_success=True,
        validator_reward=float(reward),
        validator_official=True,
        provenance={"kind": "runtime"},
        steps=steps,
        final_observation=final_observation,
        diagnostics={
            "capture": "fixed_replay",
            "model_calls": 0,
        },
    )
    persist_androidworld_run_log(output_path.parent, run_log=captured)
    return captured


def collect_replayed_source(
    *,
    args: argparse.Namespace,
    deadline: Deadline,
    attempt_root: Path,
    source_path: Path,
    source_run_log: dict[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    phase_root = attempt_root / "source" / "fixed_replay_capture"
    source_label, source_serial, source_console_port = args.source_device
    item = CanonicalRunLog(
        task=args.task,
        goal=str(source_run_log["goal"]),
        params=dict(source_run_log["task_parameters"]),
        source_run_log=source_path,
        replay_seed=SOURCE_SEED,
        step_count=len(source_run_log["steps"]),
        meta={"androidworld_success": True},
    )
    source_attempt_id = _next_source_attempt_id(args)
    command_spec = build_replay_command(
        item,
        android_world_root=args.android_world_root,
        output_root=args.results_root / "androidworld",
        method_name="source",
        device_label=source_label,
        serial=source_serial,
        console_port=source_console_port,
        adb_path=str(args.adb_path),
        max_steps=len(source_run_log["steps"]) + 1,
        timeout_sec=int(TASK_DEADLINE_SEC),
        task_random_seed=SOURCE_SEED,
        task_params_override=dict(source_run_log["task_parameters"]),
        perform_emulator_setup=_perform_androidworld_emulator_setup(),
        archive_attempt_id=source_attempt_id,
        python_executable=str(args.python_bin),
        repo_root=args.repo,
    )
    if command_spec.output_path is None:
        raise RuntimeError("fixed_replay_capture_output_path_required")
    environment = dict(os.environ)
    environment.update(
        {
            "ANDROID_SERIAL": source_serial,
            **command_spec.env,
            "OMNIFLOW_RAW_REPLAY_CAPTURE_OBSERVATIONS": "1",
            "PYTHONPATH": f"{args.repo}:{args.repo / 'src'}:{args.android_world_root}",
        }
    )
    result = run_logged_command(
        command_spec.argv,
        cwd=args.repo,
        environment=environment,
        log_path=phase_root / "fixed_replay.log",
        timeout_sec=deadline.remaining(TASK_DEADLINE_SEC),
    )
    task_results = command_spec.output_path / "task_results.jsonl"
    row = _last_jsonl_row(task_results)
    usage = _usage_from_result(row)
    result.update(
        {
            "usage": usage,
            "model_calls": usage["model_calls"],
            "total_tokens": usage["total_tokens"],
            "usage_accounting_status": _usage_accounting_status(row),
        }
    )
    result["official_validator_success"] = _official_success(row)
    if (
        result["returncode"] != 0
        or not result["official_validator_success"]
        or usage["model_calls"] != 0
    ):
        raise PipelinePhaseError(
            (
                f"fixed_replay_capture_failed:returncode={result['returncode']}:"
                f"validator={result['official_validator_success']}:"
                f"model_calls={usage['model_calls']}"
            ),
            result,
        )
    captured_path = command_spec.output_path / "run_log.json"
    captured = _captured_source_run_log(
        source_path=source_path,
        source_run_log=source_run_log,
        raw_replay_result=Path(str(command_spec.metadata["raw_replay_result"])),
        task_result=row,
        output_path=captured_path,
    )
    # Source-only collection produces one immutable RunLog candidate. The
    # caller audits and selects it after the collection batch.
    selected_path, selected = captured_path, captured
    result["input_source"] = str(source_path)
    result["captured_source"] = str(captured_path)
    result["captured_steps"] = len(captured["steps"])
    result["selected_source"] = str(selected_path)
    result["status"] = "collected"
    return selected_path, selected, result


def collect_manual_source(
    *,
    args: argparse.Namespace,
    deadline: Deadline,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    registry = load_data_index(args.memory_index)
    record = registry.get("source_index", {}).get(args.task)
    if not isinstance(record, dict):
        record = registry.get("canonical", {}).get("source_run_logs", {}).get(
            args.task
        )
    if not isinstance(record, dict):
        raise ValueError(f"manual_source_task_reference_missing:{args.task}")
    goal = str(record.get("goal") or "").strip()
    params = record.get("params")
    if not goal or not isinstance(params, dict):
        raise ValueError(f"manual_source_task_reference_invalid:{args.task}")
    params = dict(params)
    if not params:
        params = _generate_missing_androidworld_task_params(
            task=args.task,
            source_seed=(
                int(record["source_seed"])
                if record.get("source_seed") is not None
                else SOURCE_SEED
            ),
        )

    source_label, source_serial, source_console_port = args.source_device
    source_attempt_id = _next_source_attempt_id(args)
    output_path = (
        Path(args.results_root).expanduser().resolve()
        / "androidworld"
        / safe_component(args.task)
        / "source"
        / f"{safe_component(args.source_avd)}_seed{SOURCE_SEED}"
        / "runlog"
        / source_attempt_id
    )
    command = [
        str(args.python_bin),
        str(args.repo / "tools" / "manual_androidworld_harness.py"),
        "--android-world-root",
        str(args.android_world_root),
        "--task",
        args.task,
        "--task-params-json",
        json.dumps(params, ensure_ascii=False, separators=(",", ":")),
        "--seed",
        str(SOURCE_SEED),
        "--console-port",
        str(source_console_port),
        "--grpc-port",
        str(source_console_port + 3000),
        "--adb-path",
        str(args.adb_path),
        "--output",
        str(output_path),
    ]
    backend = str(
        os.environ.get("OMNIFLOW_ANDROIDWORLD_CONTROL_BACKEND", "androidworld")
    ).strip().lower()
    if (
        backend not in {"oob", "omniflow", "oob_control"}
        and str(
            os.environ.get("OMNIFLOW_ANDROIDWORLD_MANUAL_INSTALL_A11Y_FORWARDER", "1")
        ).strip().lower()
        in {"1", "true", "yes", "on"}
    ):
        command.append("--install-a11y-forwarder")
    if getattr(args, "manual_reuse_emulator", False):
        command.append("--skip-emulator-setup")
    environment = dict(os.environ)
    environment.update(
        {
            "ANDROID_SERIAL": source_serial,
            "PYTHONPATH": os.pathsep.join(
                str(path)
                for path in (
                    args.repo,
                    args.repo / "src",
                    args.android_world_root,
                    environment.get("PYTHONPATH", ""),
                )
                if str(path)
            ),
        }
    )
    try:
        completed = subprocess.run(
            command,
            cwd=args.repo,
            env=environment,
            timeout=deadline.remaining(TASK_DEADLINE_SEC),
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise PipelinePhaseError(
            "manual_source_collection_timed_out",
            {
                "status": "failed",
                "timed_out": True,
                "model_calls": 0,
                "total_tokens": 0,
                "output_path": str(output_path),
            },
        ) from error
    run_log_path = output_path / "run_log.json"
    if completed.returncode != 0 or not run_log_path.is_file():
        raise PipelinePhaseError(
            f"manual_source_collection_failed:returncode={completed.returncode}",
            {
                "status": "failed",
                "returncode": completed.returncode,
                "model_calls": 0,
                "total_tokens": 0,
                "output_path": str(output_path),
            },
        )
    run_log = require_complete_source_run_log(_read_object(run_log_path))
    validator = run_log.get("validator")
    qualified = bool(
        run_log.get("task_name") == args.task
        and run_log.get("success") is True
        and run_log.get("steps")
        and isinstance(validator, dict)
        and validator.get("official") is True
        and validator.get("success") is True
    )
    if not qualified:
        raise PipelinePhaseError(
            "manual_source_official_validation_failed",
            {
                "status": "failed",
                "returncode": completed.returncode,
                "official_validator_success": False,
                "model_calls": 0,
                "total_tokens": 0,
                "output_path": str(output_path),
                "run_log": str(run_log_path),
            },
        )
    register_source_run_log_success(
        memory_index=args.memory_index,
        task=args.task,
        run_log_path=run_log_path,
        task_parameters=params,
    )
    return run_log_path, run_log, {
        "status": "collected",
        "collection": "manual_native_androidworld",
        "device_label": source_label,
        "device_serial": source_serial,
        "official_validator_success": True,
        "captured_steps": len(run_log["steps"]),
        "captured_source": str(run_log_path),
        "selected_source": str(run_log_path),
        "model_calls": 0,
        "total_tokens": 0,
    }


def _canonical_function_store(
    memory_index: Path,
    task: str,
) -> dict[str, Any] | None:
    registry = load_data_index(memory_index)
    record = registry.get("canonical", {}).get("function_stores", {}).get(task)
    return dict(record) if isinstance(record, dict) else None


def _validate_function_store_record(
    record: dict[str, Any],
    *,
    task: str,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    store_path = Path(str(record.get("store_path") or "")).resolve()
    source_calls = record.get("source_calls")
    if (
        not isinstance(source_calls, list)
        or any(
            not isinstance(source_call, dict)
            or not str(source_call.get("function_id") or "").strip()
            or not isinstance(source_call.get("arguments"), dict)
            for source_call in source_calls
        )
    ):
        raise ValueError(f"canonical_function_source_calls_missing:{task}")
    transfer_audit = validate_omniflow_transfer_assets(
        store_path,
        require_action_transfer=True,
    )
    return store_path, source_calls, transfer_audit


def prepare_function_asset(
    *,
    args: argparse.Namespace,
    source_path: Path,
    run_log: dict[str, Any],
    attempt_root: Path,
    deadline: Deadline,
) -> tuple[dict[str, Any], dict[str, Any]]:
    existing = _canonical_function_store(args.memory_index, args.task)
    replaced_invalid_store = ""
    if existing is not None:
        try:
            _validate_function_store_record(existing, task=args.task)
        except Exception as error:
            if not getattr(args, "ensure_function", False):
                raise
            replaced_invalid_store = f"{type(error).__name__}: {error}"
            existing = None
    created = False
    creation_report: dict[str, Any] | None = None
    try:
        enhancement_round = max(
            0,
            min(
                3,
                int(os.environ.get("OMNIFLOW_FUNCTION_ENHANCEMENT_ROUND", "0")),
            ),
        )
    except ValueError:
        enhancement_round = 0
    force_enhancement = enhancement_round > 0
    repair_deterministic = (
        str(os.environ.get("OMNIFLOW_FUNCTION_DETERMINISTIC_REPAIR", "0"))
        .strip()
        .lower()
        in {"1", "true", "yes", "on"}
    )
    usage = {
        "model_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    authoring_trace: list[dict[str, Any]] = []
    if existing is None or force_enhancement or repair_deterministic:
        if not getattr(args, "ensure_function", False):
            raise FileNotFoundError(f"canonical_function_store_missing:{args.task}")
        source_device = getattr(args, "source_device", SOURCE_DEVICE)
        source_label = safe_component(str(source_device[0]))
        function_root = (
            args.asset_root
            / "androidworld"
            / safe_component(args.task)
            / source_label
            / "function"
            / "function_authoring"
            / safe_component(attempt_root.name)
        )
        if function_root.exists() and any(function_root.iterdir()) and not repair_deterministic:
            raise FileExistsError(
                f"function_authoring_artifact_already_exists:{function_root}"
            )
        try:
            creation_report = compile_function_v2(
                source_path,
                function_root,
                enhance=not repair_deterministic,
                model=args.formal_model,
                timeout=float(FUNCTION_ENHANCEMENT_TIMEOUT_SEC),
                authoring_trace=authoring_trace,
            )
            for key in usage:
                usage[key] = int(creation_report.get(key) or 0)
            store_path = Path(creation_report["store_path"]).resolve()
            refresh_data_index_from_pointer(
                memory_index=args.memory_index,
                additional_runlog_roots=(args.asset_root,),
                additional_result_roots=(args.results_root,),
                replace_recorded_roots=False,
            )
            existing = _canonical_function_store(args.memory_index, args.task)
            if existing is None:
                raise FileNotFoundError(
                    f"created_function_store_not_indexed:{args.task}"
                )
            indexed_store_path = Path(str(existing.get("store_path") or ""))
            if indexed_store_path.resolve() != store_path.resolve():
                raise ValueError(
                    "created_function_store_not_canonical:"
                    f"expected={store_path}:actual={indexed_store_path}"
                )
            created = True
        except Exception as error:
            _write_json(
                attempt_root / "assets" / "function_authoring_trace.json",
                {
                    "schema_version": "omniflow.function-authoring-trace.v1",
                    "task": args.task,
                    "source_run_log": str(source_path),
                    "status": "failed",
                    "events": authoring_trace,
                },
            )
            raise PipelinePhaseError(
                "function_asset_creation_failed",
                {
                    "status": "failed",
                    "model_calls": usage["model_calls"],
                    "total_tokens": usage["total_tokens"],
                    "error": f"{type(error).__name__}: {error}",
                },
            ) from error
        _write_json(
            attempt_root / "assets" / "function_authoring_trace.json",
            {
                "schema_version": "omniflow.function-authoring-trace.v1",
                "task": args.task,
                "source_run_log": str(source_path),
                "status": "succeeded",
                "events": authoring_trace,
            },
        )
    store_path, source_calls, transfer_audit = _validate_function_store_record(
        existing,
        task=args.task,
    )
    phase = {
        "status": (
            "enhanced_retry"
            if created and force_enhancement
            else "created"
            if created
            else "reused"
        ),
        "enhancement_round": enhancement_round,
        "model_calls": usage["model_calls"],
        "total_tokens": usage["total_tokens"],
        "store": str(store_path),
        "source_calls": source_calls,
        "transfer_audit": transfer_audit,
    }
    if creation_report is not None:
        phase["enhanced"] = creation_report.get("enhanced") is True
        phase["function_ids"] = list(creation_report.get("function_ids") or ())
    if replaced_invalid_store:
        phase["replaced_invalid_store"] = replaced_invalid_store
    return existing, {
        **phase,
    }


def _validate_prepared_mobilegpt_memory(
    memory_root: str | Path,
    *,
    task_name: str,
    source_seed: int,
) -> dict[str, Any]:
    root = Path(memory_root).expanduser().resolve()
    manifest_path = root.parent / "mobilegpt_memory_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"mobilegpt_source_memory_manifest_invalid:{manifest_path}"
        ) from error
    schema_version = str(manifest.get("schema_version") or "")
    expected_source_method = MOBILEGPT_SOURCE_METHOD_BY_SCHEMA.get(schema_version)
    if (
        expected_source_method is None
        or str(manifest.get("source_method") or "") != expected_source_method
        or str(manifest.get("task_name") or "") != str(task_name)
    ):
        raise ValueError(
            f"mobilegpt_source_memory_not_authoritative:{manifest_path}"
        )
    source_record = manifest.get("source_run_log")
    if not isinstance(source_record, dict):
        raise ValueError(
            f"mobilegpt_source_memory_source_evidence_missing:{manifest_path}"
        )
    source_path = (
        root.parent / str(source_record.get("relative_path") or "")
    ).resolve()
    from src.integrations.mobilegpt_memory import validate_mobilegpt_adapted_memory

    validation = validate_mobilegpt_adapted_memory(
        root,
        task_name=str(task_name),
        source_seed=source_seed,
        source_run_log=source_path,
        expected_source_method=expected_source_method,
    )
    return {
        "task_name": str(task_name),
        "memory_sha256": str(validation["memory_sha256"]),
        "memory_file_count": int(validation["memory_file_count"]),
        "memory_inventory": dict(validation["memory_inventory"]),
    }


def _validate_mobilegpt_memory_for_current_source(
    *,
    args: argparse.Namespace,
    memory_root: str | Path,
) -> dict[str, Any]:
    """Validate both the sealed memory and its current source binding."""

    validation = _validate_prepared_mobilegpt_memory(
        memory_root,
        task_name=args.task,
        source_seed=_e2e_source_seed(args),
    )
    # The single adapter owns both schema and evidence validation.  There is
    # no second native-cold source validator in the active path.
    from src.integrations.mobilegpt import validate_memory_manifest

    source_validation = validate_memory_manifest(memory_root)
    return {
        **validation,
        "source_validation": source_validation,
    }


def prepare_mobilegpt_memory(
    *,
    args: argparse.Namespace,
    attempt_root: Path,
    deadline: Deadline,
) -> tuple[Path, dict[str, Any]]:
    configured_root = str(
        os.environ.get("OMNIFLOW_MOBILEGPT_SOURCE_MEMORY_ROOT") or ""
    ).strip()
    if configured_root:
        root = Path(configured_root).expanduser().resolve()
        manifest_path = root.parent / "mobilegpt_memory_manifest.json"
        if not root.is_dir() or not manifest_path.is_file():
            raise FileNotFoundError(
                f"mobilegpt_source_memory_missing:{root}"
            )
        validation = _validate_mobilegpt_memory_for_current_source(
            args=args,
            memory_root=root,
        )
        return root, {
            "status": "reused",
            "model_calls": 0,
            "total_tokens": 0,
            "memory_root": str(root),
            "selection_reason": "explicit_official_memory_root",
            "memory_validation": validation,
        }
    existing = canonical_prepared_memory_from_index(
        memory_index=args.memory_index,
        task_name=args.task,
    )
    existing_is_official = False
    if existing is not None:
        manifest_path = (
            Path(str(existing["memory_root"])).expanduser().resolve().parent
            / "mobilegpt_memory_manifest.json"
        )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        target_package = str(manifest.get("target_package") or "").strip()
        if target_package not in {
            "com.google.android.inputmethod.latin",
            "com.android.inputmethod.latin",
        }:
            try:
                _validate_mobilegpt_memory_for_current_source(
                    args=args,
                    memory_root=existing["memory_root"],
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                existing_is_official = False
            else:
                existing_is_official = True
    if existing is not None and existing_is_official:
        validation = _validate_mobilegpt_memory_for_current_source(
            args=args,
            memory_root=existing["memory_root"],
        )
        return Path(str(existing["memory_root"])).resolve(), {
            "status": "reused",
            "model_calls": 0,
            "total_tokens": 0,
            "memory_root": str(existing["memory_root"]),
            "memory_validation": validation,
        }
    output_root = attempt_root / "assets" / "mobilegpt"
    _, source_path, _ = _canonical_source(
        args.memory_index,
        args.task,
        require_protocol_seed=True,
    )
    conversion = convert_runlog_to_mobilegpt_bundle(
        source_run_log=source_path,
        mobilegpt_root=args.mobilegpt_root,
        output_root=output_root,
        model=args.formal_model,
        embedding_model=(
            str(os.environ.get("MOBILEGPT_EMBEDDING_MODEL") or "")
            or MOBILEGPT_EMBEDDING_MODEL
        ),
        source_seed=_e2e_source_seed(args),
    )
    stats_summary_path = Path(str(conversion["source_stats_summary"])).resolve()
    stats_summary = json.loads(stats_summary_path.read_text(encoding="utf-8"))
    refresh_data_index_from_pointer(
        memory_index=args.memory_index,
        additional_prepared_memory_roots=(output_root,),
        replace_prepared_memory_roots=True,
    )
    result = {
        "returncode": 0,
        "timed_out": False,
        "wall_sec": float(conversion.get("source_wall_sec") or 0.0),
    }
    phase = {
        **result,
        "status": "created",
        "model_calls": int(stats_summary.get("model_calls") or 0),
        "total_tokens": int(stats_summary.get("total_tokens") or 0),
        "conversion": conversion,
    }
    existing = canonical_prepared_memory_from_index(
        memory_index=args.memory_index,
        task_name=args.task,
    )
    if existing is None:
        raise PipelinePhaseError("mobilegpt_memory_not_registered", phase)
    prepared_root = Path(str(existing["memory_root"])).resolve()
    phase["memory_root"] = str(prepared_root)
    phase["memory_validation"] = _validate_mobilegpt_memory_for_current_source(
        args=args,
        memory_root=prepared_root,
    )
    return prepared_root, phase


def prepare_appagent_memory(
    *,
    args: argparse.Namespace,
    attempt_root: Path,
    deadline: Deadline,
    source_run_log: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    explicit = str(args.appagent_memory_root or "").strip()
    if explicit:
        root = Path(explicit).expanduser().resolve()
        return root, {
            "status": "reused",
            "model_calls": 0,
            "total_tokens": 0,
            "memory_root": str(root),
        }
    root = attempt_root / "assets" / "appagent"
    command = [
        str(args.python_bin),
        "-m",
        "src.experiment.appagent_source",
        "prepare",
        "--index",
        str(args.memory_index),
        "--task",
        args.task,
        "--appagent-root",
        str(args.appagent_root),
        "--memory-root",
        str(root),
        "--model",
        getattr(args, "appagent_model", APPAGENT_MODEL),
    ]
    if source_run_log is not None:
        command.extend(("--source-run-log", str(source_run_log)))
    environment = dict(os.environ)
    result = run_logged_command(
        command,
        cwd=args.repo,
        environment=environment,
        log_path=attempt_root / "prep" / "appagent.log",
        timeout_sec=deadline.remaining(TASK_DEADLINE_SEC),
    )
    if result["returncode"] != 0:
        raise PipelinePhaseError(
            f"appagent_memory_prep_failed:{result['returncode']}",
            {**result, "model_calls": 0, "total_tokens": 0},
        )
    manifest = _read_object(root / "appagent_manifest.json")
    usage = manifest.get("doc_generation_usage")
    usage = usage if isinstance(usage, dict) else {}
    return root, {
        **result,
        "status": "created",
        "model_calls": int(usage.get("model_calls") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "memory_root": str(root),
    }


def prepare_autodroid_memory(
    *,
    args: argparse.Namespace,
    attempt_root: Path,
    deadline: Deadline,
) -> tuple[Path, dict[str, Any]]:
    """Register the official AutoDroid dataset without converting its memory."""

    del attempt_root, deadline
    root_text = str(getattr(args, "autodroid_memory_root", "") or "").strip()
    if not root_text:
        raise ValueError("autodroid_memory_root_required")
    root = Path(root_text).expanduser().resolve()
    validation = validate_autodroid_memory_root(root)
    return root, {
        "status": "reused",
        "model_calls": 0,
        "total_tokens": 0,
        "memory_root": str(root),
        "memory_format": "droidbot_utg_events",
        "official_manifest": validation,
    }


def _e2e_methods(args: argparse.Namespace) -> tuple[str, ...]:
    selected = str(getattr(args, "e2e_method", "") or "").strip()
    if not selected or selected == "all":
        return METHODS
    methods = tuple(value.strip() for value in selected.split(",") if value.strip())
    invalid = tuple(
        value
        for value in methods
        if value not in METHODS and value not in SUPPLEMENTAL_METHODS
    )
    if invalid:
        raise ValueError(
            "androidworld_e2e_method_invalid:" + ",".join(sorted(set(invalid)))
        )
    selected_supplemental = tuple(
        value for value in methods if value in SUPPLEMENTAL_METHODS
    )
    if selected_supplemental and methods != selected_supplemental:
        raise ValueError(
            "androidworld_supplemental_method_must_run_alone:"
            + ",".join(selected_supplemental)
        )
    return methods


def _e2e_devices(args: argparse.Namespace) -> tuple[tuple[str, str, int], ...]:
    methods = _e2e_methods(args)
    supplemental = set(methods).issubset(SUPPLEMENTAL_METHODS)
    allowed_devices = SUPPLEMENTAL_DEVICES if supplemental else DEVICES
    selected = getattr(args, "e2e_device", None)
    if selected is None or selected == "" or selected == "all":
        return allowed_devices
    if isinstance(selected, tuple):
        return (selected,)
    raw_devices = tuple(value.strip() for value in str(selected).split(",") if value.strip())
    devices_by_label = {device[0]: device for device in allowed_devices}
    devices = tuple(
        devices_by_label[value]
        if value in devices_by_label
        else _parse_source_device(value)
        for value in raw_devices
    )
    known = {device[0] for device in allowed_devices}
    unknown = tuple(device[0] for device in devices if device[0] not in known)
    if unknown:
        raise ValueError(
            "androidworld_e2e_device_invalid:" + ",".join(sorted(set(unknown)))
        )
    if len({device[0] for device in devices}) != len(devices):
        raise ValueError("androidworld_e2e_device_duplicate")
    return devices


def _e2e_device_models(args: argparse.Namespace) -> dict[str, str]:
    return {
        label: canonical_device_model(
            label=label,
            serial=serial,
            console_port=port,
        )
        for label, serial, port in _e2e_devices(args)
    }


def _e2e_source_seed(args: argparse.Namespace) -> int:
    return int(getattr(args, "e2e_source_seed", SOURCE_SEED))


def _e2e_evaluation_seed(args: argparse.Namespace) -> int:
    return int(getattr(args, "e2e_evaluation_seed", TASK_SEED))


def _result_environment(
    *,
    args: argparse.Namespace,
    attempt_id: str,
    attempt_root: Path,
    method: str,
    device: tuple[str, str, int],
    store_path: Path | None,
    mobilegpt_memory: Path | None,
    appagent_memory: Path | None,
    autodroid_memory: Path | None = None,
    runlog_attempt_id: str = "",
) -> dict[str, str]:
    label, serial, port = device
    result_attempt_id = f"{attempt_id}.{method}.{label}"
    result_attempt_root = (
        attempt_root / "target_attempts" / label / method / result_attempt_id
    )
    emulator_bin = str(getattr(args, "emulator_bin", "") or "").strip()
    sdk_root = ""
    if emulator_bin:
        sdk_root = str(Path(emulator_bin).resolve().parent.parent)
    else:
        sdk_root = str(
            os.environ.get("ANDROID_SDK_ROOT")
            or os.environ.get("ANDROID_HOME")
            or ""
        ).strip()
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHON_BIN": str(args.python_bin),
            "OMNIFLOW_EXP_ASSET_ROOT": str(args.asset_root),
            "OMNIFLOW_EXP_RESULTS_ROOT": str(args.results_root),
            "OMNIFLOW_EXP_MEMORY_ROOT": str(args.memory_index.parent),
            "OMNIFLOW_EXP_MEMORY_INDEX": str(args.memory_index),
            "OMNIFLOW_ANDROID_WORLD_ROOT": str(args.android_world_root),
            "OMNITRANSFER_ROOT": str(args.omnitransfer_root),
            "OMNIFLOW_MOBILEGPT_ROOT": str(args.mobilegpt_root),
            "OMNIFLOW_APPAGENT_ROOT": str(args.appagent_root),
            "OMNIFLOW_APPAGENT_MODEL": str(
                getattr(args, "appagent_model", APPAGENT_MODEL)
            ),
            "OMNIFLOW_BATCH_ATTEMPT_ID": (
                runlog_attempt_id
                or _next_result_attempt_id(args, method=method, device=device)
            ),
            "OMNIFLOW_BATCH_CHILD": "1",
            "OMNIFLOW_ANDROIDWORLD_TASK": args.task,
            "OMNIFLOW_ANDROIDWORLD_METHOD": method,
            "OMNIFLOW_ANDROIDWORLD_DEVICE": f"{label}:{serial}:{port}",
            "OMNIFLOW_ANDROIDWORLD_MAX_STEPS": str(args.max_steps),
            "OMNIFLOW_ANDROIDWORLD_MAX_FALLBACK_STEPS": str(
                args.max_fallback_steps
            ),
            "OMNIFLOW_ANDROIDWORLD_OUTPUT_PATH": str(result_attempt_root),
            # The attempt directory remains the immutable scheduler record;
            # episode RunLogs and reusable memory are published into the
            # shared task/method/device-model archive.
            "OMNIFLOW_ANDROIDWORLD_ARCHIVE_ROOT": str(
                args.results_root / "androidworld"
            ),
            "OMNIFLOW_COLLECT_PERFORMANCE": "1",
        }
    )
    if sdk_root:
        environment.update(
            {
                "ANDROID_SDK_ROOT": sdk_root,
                "ANDROID_HOME": sdk_root,
                "OMNIFLOW_REAL_ADB_PATH": str(Path(sdk_root) / "platform-tools" / "adb"),
            }
        )
    if store_path is not None:
        environment["OMNIFLOW_ANDROIDWORLD_STORE_PATH"] = str(store_path)
    if mobilegpt_memory is not None:
        environment["OMNIFLOW_MOBILEGPT_SOURCE_MEMORY_ROOT"] = str(
            mobilegpt_memory
        )
    if appagent_memory is not None:
        environment["OMNIFLOW_APPAGENT_MEMORY_ROOT"] = str(appagent_memory)
    if autodroid_memory is not None:
        environment["OMNIFLOW_AUTODROID_MEMORY_ROOT"] = str(autodroid_memory)
    return environment


def _supplemental_outcomes_root(args: argparse.Namespace) -> Path:
    methods = _e2e_methods(args)
    if set(methods).issubset(SUPPLEMENTAL_METHODS):
        return args.results_root / SUPPLEMENTAL_RESULTS_NAMESPACE
    return args.results_root / "androidworld" / ".archive" / "outcomes" / "formal"


def _is_autodroid_supplemental(args: argparse.Namespace) -> bool:
    return _e2e_methods(args) == ("autodroid",)


def _autodroid_task_params_from_index(
    memory_index: Path,
    task: str,
) -> dict[str, Any]:
    registry = load_data_index(memory_index)
    record = registry.get("source_index", {}).get(task)
    if not isinstance(record, dict):
        record = (
            registry.get("canonical", {})
            .get("source_run_logs", {})
            .get(task)
        )
    if not isinstance(record, dict):
        raise ValueError(f"autodroid_task_reference_missing:{task}")
    params = record.get("params")
    if isinstance(params, dict):
        resolved = dict(params)
    else:
        retained_source_run_log = str(record.get("retained_source_run_log") or "")
        if not retained_source_run_log:
            return {}
        source_path = resolve_path(retained_source_run_log)
        try:
            source = _read_object(source_path)
        except FileNotFoundError:
            return {}
        source_params = source.get("task_parameters")
        resolved = dict(source_params) if isinstance(source_params, dict) else {}
    resolved.pop("seed", None)
    return resolved


def _androidworld_result_command(
    *,
    args: argparse.Namespace,
    attempt_id: str,
    attempt_root: Path,
    method: str,
    device: tuple[str, str, int],
    store_path: Path | None,
    mobilegpt_memory: Path | None,
    appagent_memory: Path | None,
    autodroid_memory: Path | None = None,
    autodroid_task_params: dict[str, Any] | None = None,
    evaluation_task_params: dict[str, Any] | None = None,
) -> list[str]:
    """Build the one child command for an AndroidWorld result.

    The public shell remains the human-facing entry point. Once the scheduler
    has prepared a task, however, it calls the single-result runner directly;
    the shell is not an internal RPC boundary.
    """

    label, serial, console_port = device
    result_attempt_id = f"{attempt_id}.{method}.{label}"
    result_attempt_root = (
        attempt_root / "target_attempts" / label / method / result_attempt_id
    )
    timeout_sec = (
        int(args.max_steps) * STEP_TIMEOUT_SEC + VALIDATOR_FLUSH_GRACE_SEC
    )
    command = [
        str(args.python_bin),
        "-m",
        "src.experiment.run_task",
        "result",
        "--index",
        str(args.memory_index),
        "--android-world-root",
        str(args.android_world_root),
        "--adb-path",
        str(args.adb_path),
        "--task",
        str(args.task),
        "--source-seed",
        str(_e2e_source_seed(args)),
        "--output-path",
        str(result_attempt_root),
        "--result-registry-root",
        str(
            Path(args.results_root)
            / "androidworld"
            / ".archive"
            / "result_registry"
        ),
        "--omnitransfer-root",
        str(args.omnitransfer_root),
        "--store-path",
        str(store_path or ""),
        "--store-index",
        str(args.memory_index),
        "--mobilegpt-root",
        str(args.mobilegpt_root),
        "--mobilegpt-source-memory-root",
        str(mobilegpt_memory or ""),
        "--appagent-root",
        str(args.appagent_root),
        "--timeout-sec",
        str(timeout_sec),
        "--max-steps",
        str(args.max_steps),
        "--max-fallback-steps",
        str(args.max_fallback_steps),
        "--task-random-seed",
        str(_e2e_evaluation_seed(args)),
        "--model",
        str(
            OMNIFLOW_PLANNER_MODEL
            if method == "omniflow"
            else getattr(args, "formal_model", FORMAL_MODEL)
        ),
        "--planner-provider",
        "openai",
        "--method",
        str(method),
        "--device",
        f"{label}:{serial}:{int(console_port)}",
    ]
    if (
        evaluation_task_params is None
        and not FIXED_TASK_PARAMS
        and method != "autodroid"
    ):
        command.extend(("--no-fixed-task-params", "--task-params-json", ""))
    elif evaluation_task_params is not None and method != "autodroid":
        command.extend(
            (
                "--task-params-json",
                json.dumps(
                    evaluation_task_params,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
        )
    if method == "appagent":
        command.extend(
            (
                "--appagent-model",
                str(getattr(args, "appagent_model", APPAGENT_MODEL)),
            )
        )
    if method == "mobilegpt":
        command.extend(
            (
                "--mobilegpt-port",
                str(_mobilegpt_port_for_device(args, device)),
            )
        )
    if appagent_memory is not None:
        command.extend(("--appagent-memory-root", str(appagent_memory)))
    if method == "autodroid":
        autodroid_root = getattr(args, "autodroid_root", "")
        autodroid_memory_root = autodroid_memory or getattr(
            args, "autodroid_memory_root", ""
        )
        command.extend(
            (
                "--autodroid-root",
                str(autodroid_root),
                "--autodroid-memory-root",
                str(autodroid_memory_root),
                "--autodroid-app",
                str(getattr(args, "autodroid_app", "")),
                "--autodroid-policy",
                str(getattr(args, "autodroid_policy", "replay")),
            )
        )
        if autodroid_task_params is not None:
            command.extend(
                (
                    "--task-params-json",
                    json.dumps(
                        autodroid_task_params,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                )
            )
    if args.dry_run:
        command.append("--dry-run")
    return command


def _mobilegpt_port_for_device(
    args: argparse.Namespace,
    device: tuple[str, str, int],
) -> int:
    """Allocate one deterministic Server port per selected target device.

    The official MobileGPT Server still listens on one TCP socket per
    process.  The scheduler launches one result child per device, so sharing
    the historical 12345 port would serialize the workers or make later
    servers fail to bind.  Keep the first selected device on 12345 for
    backwards compatibility and use isolated slots for the remaining
    devices.  This changes only the transport adapter; MobileGPT's client,
    planner, executor, and memory code remain untouched.
    """

    base_port = int(os.environ.get("OMNIFLOW_MOBILEGPT_BASE_PORT", "12345"))
    selected = _e2e_devices(args)
    try:
        slot = selected.index(device)
    except ValueError:
        slot = 0
    return base_port + (slot * 10)


def run_target_workers(
    *,
    args: argparse.Namespace,
    deadline: Deadline,
    attempt_id: str,
    attempt_root: Path,
    outcomes_root: Path,
    store_path: Path | None,
    mobilegpt_memory: Path | None,
    appagent_memory: Path | None,
    blocked_methods: dict[str, tuple[str, str, str]],
    autodroid_memory: Path | None = None,
    command_runner: Callable[..., dict[str, Any]] = run_logged_command,
) -> list[dict[str, Any]]:
    """Invoke one normal run_task command for every selected cell."""

    del outcomes_root
    methods = _e2e_methods(args)
    devices = _e2e_devices(args)
    autodroid_task_params = (
        _autodroid_task_params_from_index(args.memory_index, args.task)
        if "autodroid" in methods
        else None
    )

    def worker(device: tuple[str, str, int]) -> list[dict[str, Any]]:
        label, _, _ = device
        rows: list[dict[str, Any]] = []
        for method in methods:
            if method in blocked_methods:
                rows.append(
                    {
                        "method": method,
                        "device": label,
                        "status": "memory_failed",
                        "error": blocked_methods[method][1],
                    }
                )
                continue
            log_path = attempt_root / "logs" / label / f"{method}.log"
            result = command_runner(
                _androidworld_result_command(
                    args=args,
                    attempt_id=attempt_id,
                    attempt_root=attempt_root,
                    method=method,
                    device=device,
                    store_path=store_path,
                    mobilegpt_memory=mobilegpt_memory,
                    appagent_memory=appagent_memory,
                    autodroid_memory=autodroid_memory,
                    autodroid_task_params=autodroid_task_params,
                    evaluation_task_params=None,
                ),
                cwd=args.repo,
                environment=_result_environment(
                    args=args,
                    attempt_id=attempt_id,
                    attempt_root=attempt_root,
                    method=method,
                    device=device,
                    store_path=store_path,
                    mobilegpt_memory=mobilegpt_memory,
                    appagent_memory=appagent_memory,
                    autodroid_memory=autodroid_memory,
                    runlog_attempt_id=f"run_{uuid.uuid4().hex}",
                ),
                log_path=log_path,
                timeout_sec=deadline.remaining(args.task_deadline_sec),
            )
            rows.append(
                {
                    "method": method,
                    "device": label,
                    "status": (
                        "completed" if result.get("returncode") == 0 else "failed"
                    ),
                    **result,
                }
            )
        return rows

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, len(devices))
    ) as executor:
        futures = [executor.submit(worker, device) for device in devices]
        return [row for future in futures for row in future.result()]


def _blocked_all(**_: Any) -> None:
    return None


def _report(
    *,
    args: argparse.Namespace,
    attempt_id: str,
    attempt_root: Path,
    outcomes_root: Path,
    deadline: Deadline,
    phases: dict[str, Any],
) -> dict[str, Any]:
    del outcomes_root
    target_phase = phases.get("targets")
    workers = (
        list(target_phase.get("workers") or ())
        if isinstance(target_phase, dict)
        else []
    )
    failed = sum(row.get("status") != "completed" for row in workers)
    total = len(_e2e_methods(args)) * len(_e2e_devices(args))
    counts = {
        "total": total,
        "completed": total - failed if workers else 0,
        "failed": failed,
        "pending": max(0, total - len(workers)),
    }
    summary = {
        "schema_version": "omniflow.androidworld.task-run.v1",
        "task": args.task,
        "attempt_id": attempt_id,
        "status": (
            "complete"
            if workers and failed == 0 and len(workers) == total
            else "deadline_exceeded"
            if deadline.expired
            else "failed"
        ),
        "source_seed": _e2e_source_seed(args),
        "evaluation_seed": _e2e_evaluation_seed(args),
        "deadline_sec": deadline.seconds,
        "outer_wall_sec": deadline.elapsed,
        "counts": counts,
        "phases": phases,
    }
    return summary


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    deadline = Deadline(args.task_deadline_sec)
    outcomes_root = _supplemental_outcomes_root(args)
    attempt_id = "run"
    methods = _e2e_methods(args)
    devices = _e2e_devices(args)
    if args.dry_run:
        return {
            "schema_version": "omniflow.androidworld.task-command.v1",
            "task": args.task,
            "deadline_sec": args.task_deadline_sec,
            "max_steps": int(args.max_steps),
            "max_fallback_steps": int(args.max_fallback_steps),
            "source_seed": _e2e_source_seed(args),
            "evaluation_seed": _e2e_evaluation_seed(args),
            "methods": list(methods),
            "devices": [list(device) for device in devices],
            "schedule": {device[0]: list(methods) for device in devices},
            "writes": False,
        }

    temporary = tempfile.TemporaryDirectory(prefix="omniflow-androidworld-")
    attempt_root = Path(temporary.name)
    phases: dict[str, Any] = {}

    source_path: Path
    run_log: dict[str, Any]
    if _is_autodroid_supplemental(args):
        source_path = args.memory_index
        run_log = {"task_name": args.task, "task_parameters": {}}
    else:
        _, source_path, run_log = _canonical_source(args.memory_index, args.task)
    phases["source"] = {
        "status": "reused",
        "source_run_log": str(source_path),
    }

    blocked_methods: dict[str, tuple[str, str, str]] = {}
    store_path: Path | None = None
    mobilegpt_memory: Path | None = None
    appagent_memory: Path | None = None
    autodroid_memory: Path | None = None

    if "omniflow" in methods:
        try:
            function_store, phases["omniflow_memory"] = prepare_function_asset(
                args=args,
                source_path=source_path,
                run_log=run_log,
                attempt_root=attempt_root,
                deadline=deadline,
            )
            store_path = Path(str(function_store["store_path"]))
        except Exception as error:
            phases["omniflow_memory"] = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
            blocked_methods["omniflow"] = ("failed", "memory", str(error))

    if "mobilegpt" in methods:
        try:
            mobilegpt_memory, phases["mobilegpt_memory"] = prepare_mobilegpt_memory(
                args=args, attempt_root=attempt_root, deadline=deadline
            )
        except Exception as error:
            phases["mobilegpt_memory"] = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
            blocked_methods["mobilegpt"] = ("failed", "memory", str(error))

    if "appagent" in methods:
        try:
            appagent_memory, phases["appagent_memory"] = prepare_appagent_memory(
                args=args,
                attempt_root=attempt_root,
                deadline=deadline,
                source_run_log=source_path,
            )
        except Exception as error:
            phases["appagent_memory"] = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
            blocked_methods["appagent"] = ("failed", "memory", str(error))

    if "autodroid" in methods:
        try:
            autodroid_memory, phases["autodroid_memory"] = prepare_autodroid_memory(
                args=args, attempt_root=attempt_root, deadline=deadline
            )
        except Exception as error:
            phases["autodroid_memory"] = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
            blocked_methods["autodroid"] = ("failed", "memory", str(error))

    phases["targets"] = {
        "status": "finished",
        "workers": run_target_workers(
            args=args,
            deadline=deadline,
            attempt_id=attempt_id,
            attempt_root=attempt_root,
            outcomes_root=outcomes_root,
            store_path=store_path,
            mobilegpt_memory=mobilegpt_memory,
            appagent_memory=appagent_memory,
            autodroid_memory=autodroid_memory,
            blocked_methods=blocked_methods,
        ),
    }
    try:
        return _report(
            args=args,
            attempt_id=attempt_id,
            attempt_root=attempt_root,
            outcomes_root=outcomes_root,
            deadline=deadline,
            phases=phases,
        )
    finally:
        temporary.cleanup()


_BMOCA_ENVIRONMENT_IDS = tuple(str(value) for value in range(100, 110))
_BMOCA_METHODS = (
    "ours_replay",
    "mobilegpt_replay",
    "skilldroid_replay",
)
_BMOCA_PROGRESS_FIELDS = (
    "task",
    "method",
    "environment_id",
    "status",
    "official_success",
    "method_success",
    "actions_executed",
    "model_calls",
    "embedding_calls",
    "fallback_steps",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "error",
    "started_at",
    "finished_at",
    "wall_sec",
    "process_pid",
    "emulator_serial",
    "appium_port",
    "appium_system_port",
    "emulator_console_port",
    "emulator_adb_port",
    "emulator_grpc_port",
    "avd_home",
    "store_path",
    "memory_path",
    "summary_path",
    "run_log_path",
    "log_path",
)
_MODEL_ENVIRONMENT_KEYS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "LLMTHU_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_MODEL",
    "OMNIFLOW_MODEL_ENDPOINT_PROFILE",
    "OMNIFLOW_PLANNER_MODEL",
    "OMNIFLOW_PLANNER_PROVIDER",
)


def _bmoca_manifest_tasks(
    manifest_path: Path,
    selection: str,
) -> tuple[list[str], dict[str, Path]]:
    manifest = _read_object(manifest_path)
    raw_tasks = manifest.get("tasks")
    raw_traces = manifest.get("traces")
    if not isinstance(raw_tasks, list) or not isinstance(raw_traces, list):
        raise ValueError("bmoca_corpus_manifest_invalid")
    available = [str(task).strip() for task in raw_tasks if str(task).strip()]
    requested = (
        available
        if selection.strip() == "*"
        else [item.strip() for item in selection.split(",") if item.strip()]
    )
    if not requested:
        raise ValueError("bmoca_campaign_tasks_required")
    unknown = [task for task in requested if task not in available]
    if unknown:
        raise ValueError("bmoca_campaign_unknown_tasks:" + ",".join(unknown))
    if len(requested) != len(set(requested)):
        raise ValueError("bmoca_campaign_duplicate_tasks")
    source_run_logs: dict[str, Path] = {}
    for trace in raw_traces:
        if not isinstance(trace, dict):
            continue
        task = str(trace.get("task_id") or "").strip()
        if (
            task not in requested
            or str(trace.get("environment_id") or "") != "100"
            or str(trace.get("role") or "") != "source"
        ):
            continue
        runlog = trace.get("runlog")
        relative = str(runlog.get("path") or "") if isinstance(runlog, dict) else ""
        path = (manifest_path.parent / relative).resolve()
        if not relative or not path.is_file():
            raise FileNotFoundError(f"bmoca_source_runlog_missing:{task}:{path}")
        if task in source_run_logs:
            raise ValueError(f"bmoca_source_runlog_ambiguous:{task}")
        source_run_logs[task] = path
    return requested, source_run_logs


def _bmoca_avd_names(bmoca_root: Path) -> dict[str, str]:
    catalog = bmoca_root / "asset/environments/config/environments_test.csv"
    with catalog.open(newline="", encoding="utf-8") as stream:
        rows = {
            str(row["idx"]): f"{row['device_id']}_test_00"
            for row in csv.DictReader(stream)
        }
    missing = [item for item in _BMOCA_ENVIRONMENT_IDS if item not in rows]
    if missing:
        raise ValueError("bmoca_environment_catalog_incomplete:" + ",".join(missing))
    return rows


def _clone_bmoca_avd_home(
    *,
    source_home: Path,
    target_home: Path,
    avd_name: str,
) -> Path:
    """Clone one AVD into an isolated home; Linux reflinks keep this cheap."""

    source_avd = source_home / f"{avd_name}.avd"
    source_ini = source_home / f"{avd_name}.ini"
    if not source_avd.is_dir() or not source_ini.is_file():
        raise FileNotFoundError(f"bmoca_source_avd_missing:{source_avd}")
    target_home.mkdir(parents=True, exist_ok=False)
    target_avd = target_home / source_avd.name
    copy_command = ["cp", "-a"]
    if os.uname().sysname == "Linux":
        copy_command.append("--reflink=auto")
    subprocess.run(
        [*copy_command, str(source_avd), str(target_avd)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    target_ini = target_home / source_ini.name
    subprocess.run(
        ["cp", "-a", str(source_ini), str(target_ini)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    lines = []
    for line in target_ini.read_text(encoding="utf-8").splitlines():
        if line.startswith("path="):
            lines.append(f"path={target_avd}")
        elif line.startswith("path.rel="):
            lines.append(f"path.rel=avd/{avd_name}.avd")
        else:
            lines.append(line)
    if not any(line.startswith("path=") for line in lines):
        lines.append(f"path={target_avd}")
    target_ini.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target_home


def _save_bmoca_function_once(
    *,
    args: argparse.Namespace,
    task: str,
    source_run_log: Path,
    task_root: Path,
) -> tuple[Path, dict[str, Any]]:
    attempt_id = safe_component(
        str(
            getattr(args, "attempt_id", "")
            or getattr(getattr(args, "output_root", None), "name", "")
            or task_root.name
        )
    )
    repository_root = Path(
        getattr(args, "repo", task_root.parent)
    ).expanduser().resolve()
    function_root = (
        repository_root
        / "data"
        / "bmoca"
        / safe_component(task)
        / "env100"
        / "function"
        / "function_authoring"
        / attempt_id
    )
    if function_root.exists() and any(function_root.iterdir()):
        raise FileExistsError(f"bmoca_function_store_already_exists:{function_root}")
    usage = {
        "model_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    started = time.monotonic()
    try:
        report = compile_function_v2(
            source_run_log,
            function_root,
            enhance=True,
            model=args.formal_model,
            timeout=float(FUNCTION_ENHANCEMENT_TIMEOUT_SEC),
        )
    except Exception as error:
        _write_json(
            task_root / "enhancement_failure.json",
            {
                "schema_version": "omniflow.bmoca-function-enhancement.v1",
                "status": "failed",
                "task": task,
                "source_run_log": str(source_run_log),
                "compile_function_calls": 1,
                "wall_sec": round(time.monotonic() - started, 6),
                "error": f"{type(error).__name__}: {error}",
                **usage,
            },
        )
        raise
    for key in usage:
        usage[key] = int(report.get(key) or 0)
    store_path = Path(report["store_path"]).resolve()
    enhancement = {
        "schema_version": "omniflow.bmoca-function-enhancement.v1",
        "task": task,
        "source_run_log": str(source_run_log),
        "compile_function_calls": 1,
        "enhanced": report.get("enhanced") is True,
        "function_ids": list(report.get("function_ids") or ()),
        "store_path": str(store_path),
        "transfer_state_count": int(report.get("transfer_state_count") or 0),
        "wall_sec": round(time.monotonic() - started, 6),
        **usage,
    }
    _write_json(task_root / "enhancement.json", enhancement)
    return store_path, enhancement


def _prepare_bmoca_skilldroid_memory(
    *,
    task: str,
    source_run_log: Path,
    task_root: Path,
) -> tuple[Path, dict[str, Any]]:
    memory_path = task_root / "memory" / "skilldroid" / "macro.json"
    started = time.monotonic()
    try:
        report = compile_droidrun_macro(
            source_run_log=source_run_log,
            source_state_catalog=source_run_log.with_name("transfer_states.json"),
            output_path=memory_path,
        )
    except Exception as error:
        failure = {
            "schema_version": "omniflow.bmoca-reuse-memory.v1",
            "status": "failed",
            "task": task,
            "method": "skilldroid_replay",
            "source_run_log": str(source_run_log),
            "wall_sec": round(time.monotonic() - started, 6),
            "error": f"{type(error).__name__}: {error}",
        }
        _write_json(task_root / "skilldroid_memory_failure.json", failure)
        raise
    prepared = {
        **report,
        "status": "prepared",
        "task": task,
        "method": "skilldroid_replay",
        "source_run_log": str(source_run_log),
        "wall_sec": round(time.monotonic() - started, 6),
    }
    _write_json(task_root / "skilldroid_memory.json", prepared)
    return memory_path, prepared


def _prepare_bmoca_mobilegpt_memory(
    *,
    args: argparse.Namespace,
    task: str,
    source_run_log: Path,
    task_root: Path,
) -> tuple[Path, dict[str, Any]]:
    root = task_root / "memory" / "mobilegpt"
    memory_path = root / "memory"
    stats_path = root / "conversion.stats.jsonl"
    audit_path = root / "conversion.audit.json"
    started = time.monotonic()
    embedding_model = str(
        os.environ.get("MOBILEGPT_EMBEDDING_MODEL") or MOBILEGPT_EMBEDDING_MODEL
    ).strip()
    embedding_api_key = str(
        os.environ.get("MOBILEGPT_EMBEDDING_API_KEY") or ""
    ).strip()
    embedding_base_url = str(
        os.environ.get("MOBILEGPT_EMBEDDING_BASE_URL") or ""
    ).strip()
    if not embedding_api_key or not embedding_base_url:
        error = ValueError("mobilegpt_embedding_endpoint_required")
        _write_json(
            task_root / "mobilegpt_memory_failure.json",
            {
                "schema_version": "omniflow.bmoca-reuse-memory.v1",
                "status": "failed",
                "task": task,
                "method": "mobilegpt_replay",
                "source_run_log": str(source_run_log),
                "wall_sec": round(time.monotonic() - started, 6),
                "embedding_calls": 0,
                "error": f"ValueError: {error}",
            },
        )
        raise error
    from openai import OpenAI

    embedding_client = OpenAI(
        api_key=embedding_api_key,
        base_url=embedding_base_url,
        max_retries=0,
        timeout=60.0,
    )
    embedding_calls = 0

    def embed(screen: str) -> list[float]:
        nonlocal embedding_calls
        embedding_calls += 1
        response = embedding_client.embeddings.create(
            model=embedding_model,
            input=[screen],
        )
        return [float(value) for value in response.data[0].embedding]

    try:
        report = convert_runlog_to_mobilegpt_memory(
            source_run_log=source_run_log,
            mobilegpt_root=args.mobilegpt_root,
            memory_root=memory_path,
            stats_path=stats_path,
            audit_path=audit_path,
            model=args.formal_model,
            embedding_model=embedding_model,
            embedding_provider=embed,
        )
    except Exception as error:
        failure = {
            "schema_version": "omniflow.bmoca-reuse-memory.v1",
            "status": "failed",
            "task": task,
            "method": "mobilegpt_replay",
            "source_run_log": str(source_run_log),
            "wall_sec": round(time.monotonic() - started, 6),
            "embedding_calls": embedding_calls,
            "error": f"{type(error).__name__}: {error}",
        }
        _write_json(task_root / "mobilegpt_memory_failure.json", failure)
        raise
    prepared = {
        "schema_version": "omniflow.bmoca-reuse-memory.v1",
        "status": "prepared",
        "task": task,
        "method": "mobilegpt_replay",
        "source_run_log": str(source_run_log),
        "memory_path": str(memory_path),
        "embedding_model": embedding_model,
        "transition_count": int(report.get("transition_count") or 0),
        "embedding_calls": embedding_calls,
        "wall_sec": round(time.monotonic() - started, 6),
    }
    _write_json(task_root / "mobilegpt_memory.json", prepared)
    return memory_path, prepared


def _write_bmoca_progress(
    path: Path,
    rows: dict[tuple[str, str, str], dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_BMOCA_PROGRESS_FIELDS)
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow(
                {field: rows[key].get(field, "") for field in _BMOCA_PROGRESS_FIELDS}
            )
    temporary.replace(path)


def _append_bmoca_progress_event(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _bmoca_result_environment(
    *,
    args: argparse.Namespace,
    task: str,
    method: str,
    environment_id: str,
    store_path: Path,
    memory_path: Path | None,
    output_path: Path,
    avd_home: Path,
    appium_port: int,
    appium_system_port: int,
    emulator_console_port: int,
    emulator_adb_port: int,
    emulator_grpc_port: int,
) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHON_BIN": str(args.python_bin),
            "PYTHONPATH": ":".join(
                str(path)
                for path in (
                    args.repo,
                    args.repo / "src",
                    args.bmoca_root,
                    args.bmoca_android_env_root,
                )
            ),
            "OMNITRANSFER_ROOT": str(args.omnitransfer_root),
            "OMNIFLOW_BMOCA_ROOT": str(args.bmoca_root),
            "OMNIFLOW_BMOCA_ANDROID_ENV_ROOT": str(args.bmoca_android_env_root),
            "OMNIFLOW_BMOCA_AVD_HOME": str(avd_home),
            "OMNIFLOW_BMOCA_OUTPUT_PATH": str(output_path),
            "OMNIFLOW_BMOCA_SINGLE_ENVIRONMENT_ID": environment_id,
            "OMNIFLOW_BMOCA_APPIUM_PORT": str(appium_port),
            "OMNIFLOW_BMOCA_APPIUM_SYSTEM_PORT": str(appium_system_port),
            "OMNIFLOW_BMOCA_EMULATOR_CONSOLE_PORT": str(emulator_console_port),
            "OMNIFLOW_BMOCA_EMULATOR_ADB_PORT": str(emulator_adb_port),
            "OMNIFLOW_BMOCA_EMULATOR_GRPC_PORT": str(emulator_grpc_port),
            "OMNIFLOW_ANDROIDWORLD_STORE_PATH": str(store_path),
            "OMNIFLOW_ANDROID_SDK_ROOT": str(args.android_sdk_root),
            "OMNIFLOW_ANDROIDWORLD_MAX_FALLBACK_STEPS": "0",
        }
    )
    if memory_path is not None:
        environment["OMNIFLOW_BMOCA_REUSE_MEMORY_PATH"] = str(memory_path)
    if method == "mobilegpt_replay":
        environment["OMNIFLOW_MOBILEGPT_ROOT"] = str(args.mobilegpt_root)
        environment["MOBILEGPT_CHAT_MODEL"] = str(args.formal_model)
        environment["MOBILEGPT_EMBEDDING_MODEL"] = MOBILEGPT_EMBEDDING_MODEL
    else:
        for key in (*_MODEL_ENVIRONMENT_KEYS, "OMNIFLOW_ENV_FILE"):
            environment.pop(key, None)
    return environment


def _bmoca_environment_failure(error: str) -> bool:
    normalized = str(error or "").lower()
    return any(
        marker in normalized
        for marker in (
            "appium",
            "adb",
            "avd",
            "emulator",
            "simulator",
            "snapshot",
            "connection",
            "timed out",
            "timeout",
            "environment",
        )
    )


def _bmoca_result_command(
    *,
    args: argparse.Namespace,
    task: str,
    method: str,
    environment_id: str,
    store_path: Path,
    memory_path: Path | None,
    output_path: Path,
    avd_home: Path,
    appium_port: int,
    appium_system_port: int,
    emulator_console_port: int,
    emulator_adb_port: int,
    emulator_grpc_port: int,
) -> list[str]:
    """Build the native B-MoCA episode command without re-entering the shell."""

    command = [
        str(args.python_bin),
        "-m",
        "src.integrations.android_world.run_episode",
        "--environment",
        "bmoca",
        "--bmoca-root",
        str(args.bmoca_root),
        "--environment-ids",
        str(environment_id),
        "--android-sdk-root",
        str(args.android_sdk_root),
        "--android-avd-home",
        str(avd_home),
        "--appium-port",
        str(appium_port),
        "--appium-system-port",
        str(appium_system_port),
        "--emulator-console-port",
        str(emulator_console_port),
        "--emulator-adb-port",
        str(emulator_adb_port),
        "--emulator-grpc-port",
        str(emulator_grpc_port),
        "--tasks",
        str(task),
        "--agent",
        str(method),
        "--store-path",
        str(store_path),
        "--output-path",
        str(output_path),
    ]
    if method != "ours_replay":
        command.extend(("--reuse-memory-path", str(memory_path or "")))
    if method == "mobilegpt_replay":
        command.extend(("--mobilegpt-root", str(args.mobilegpt_root)))
    return command


def _run_bmoca_result(
    *,
    args: argparse.Namespace,
    task: str,
    method: str,
    environment_id: str,
    store_path: Path,
    memory_path: Path | None,
    task_root: Path,
    avd_home: Path,
    appium_port: int,
    appium_system_port: int,
    emulator_console_port: int,
    emulator_adb_port: int,
    emulator_grpc_port: int,
    timeout_sec: float,
    command_runner: Callable[..., dict[str, Any]] = run_logged_command,
) -> dict[str, Any]:
    result_root = task_root / "attempts" / method / f"env_{environment_id}"
    summary_path = result_root / "summary.json"
    log_path = task_root / "logs" / method / f"env_{environment_id}.log"
    live_started = time.monotonic()
    result = command_runner(
        _bmoca_result_command(
            args=args,
            task=task,
            method=method,
            environment_id=environment_id,
            store_path=store_path,
            memory_path=memory_path,
            output_path=result_root,
            avd_home=avd_home,
            appium_port=appium_port,
            appium_system_port=appium_system_port,
            emulator_console_port=emulator_console_port,
            emulator_adb_port=emulator_adb_port,
            emulator_grpc_port=emulator_grpc_port,
        ),
        cwd=args.repo,
        environment=_bmoca_result_environment(
            args=args,
            task=task,
            method=method,
            environment_id=environment_id,
            store_path=store_path,
            memory_path=memory_path,
            output_path=result_root,
            avd_home=avd_home,
            appium_port=appium_port,
            appium_system_port=appium_system_port,
            emulator_console_port=emulator_console_port,
            emulator_adb_port=emulator_adb_port,
            emulator_grpc_port=emulator_grpc_port,
        ),
        log_path=log_path,
        timeout_sec=timeout_sec,
    )
    live_finished = time.monotonic()
    summary = _read_object(summary_path) if summary_path.is_file() else {}
    results = summary.get("results") if isinstance(summary.get("results"), list) else []
    episode = results[0] if len(results) == 1 and isinstance(results[0], dict) else {}
    official_success = episode.get("official_success") is True
    error = str(episode.get("error") or "").strip()
    if not summary:
        error = error or f"bmoca_child_exit_{result.get('returncode')}"
        status = "environment_failure"
    elif _bmoca_environment_failure(error):
        status = "environment_failure"
    else:
        status = "success" if official_success else "method_failure"
    evidence = episode.get("run_log_evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    return {
        "task": task,
        "method": method,
        "environment_id": environment_id,
        "status": status,
        "official_success": official_success,
        "method_success": episode.get("method_success") is True,
        "actions_executed": int(episode.get("actions_executed") or 0),
        "model_calls": int(episode.get("model_calls") or 0),
        "embedding_calls": int(episode.get("embedding_calls") or 0),
        "fallback_steps": int(episode.get("fallback_steps") or 0),
        "prompt_tokens": int(episode.get("prompt_tokens") or 0),
        "completion_tokens": int(episode.get("completion_tokens") or 0),
        "total_tokens": int(episode.get("total_tokens") or 0),
        "error": error,
        "started_at": str(result.get("started_at") or ""),
        "finished_at": str(result.get("finished_at") or ""),
        "wall_sec": float(result.get("wall_sec") or 0),
        "process_pid": int(result.get("process_pid") or 0),
        "emulator_serial": str(episode.get("emulator_serial") or ""),
        "appium_port": appium_port,
        "appium_system_port": appium_system_port,
        "emulator_console_port": emulator_console_port,
        "emulator_adb_port": emulator_adb_port,
        "emulator_grpc_port": emulator_grpc_port,
        "avd_home": str(avd_home),
        "store_path": str(store_path),
        "memory_path": str(memory_path) if memory_path is not None else "",
        "summary_path": str(summary_path) if summary_path.is_file() else "",
        "run_log_path": str(
            evidence.get("run_log_path")
            or evidence.get("target_run_log_path")
            or ""
        ),
        "log_path": str(log_path),
        "_live_started": live_started,
        "_live_finished": live_finished,
    }


def _max_live_bmoca_results(rows: Sequence[dict[str, Any]]) -> int:
    events: list[tuple[float, int]] = []
    for row in rows:
        started = row.get("_live_started")
        finished = row.get("_live_finished")
        if isinstance(started, (int, float)) and isinstance(finished, (int, float)):
            events.extend(((float(started), 1), (float(finished), -1)))
    active = maximum = 0
    for _, delta in sorted(events, key=lambda item: (item[0], -item[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum


def _run_bmoca_method_results(
    *,
    args: argparse.Namespace,
    task: str,
    method: str,
    store_path: Path,
    memory_path: Path | None = None,
    task_root: Path,
    avd_homes: dict[str, Path],
    environment_ids: Sequence[str] = _BMOCA_ENVIRONMENT_IDS,
    command_runner: Callable[..., dict[str, Any]] = run_logged_command,
) -> list[dict[str, Any]]:
    def worker(environment_id: str) -> dict[str, Any]:
        offset = int(environment_id) - 100
        ports = {
            "appium_port": 4723 + offset,
            "appium_system_port": 8200 + offset,
            "emulator_console_port": 5600 + 2 * offset,
            "emulator_adb_port": 5601 + 2 * offset,
            "emulator_grpc_port": 8554 + offset,
        }
        try:
            return _run_bmoca_result(
                args=args,
                task=task,
                method=method,
                environment_id=environment_id,
                store_path=store_path,
                memory_path=memory_path,
                task_root=task_root,
                avd_home=avd_homes[environment_id],
                timeout_sec=float(BMOCA_RESULT_TIMEOUT_SEC),
                command_runner=command_runner,
                **ports,
            )
        except Exception as error:  # noqa: BLE001 - conclude this immutable result
            return {
                "task": task,
                "method": method,
                "environment_id": environment_id,
                "status": "environment_failure",
                "official_success": False,
                "method_success": False,
                "actions_executed": 0,
                "model_calls": 0,
                "embedding_calls": 0,
                "fallback_steps": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "error": f"{type(error).__name__}: {error}",
                "avd_home": str(avd_homes[environment_id]),
                "store_path": str(store_path),
                "memory_path": str(memory_path) if memory_path is not None else "",
                "log_path": str(
                    task_root / "logs" / method / f"env_{environment_id}.log"
                ),
                **ports,
            }

    rows: list[dict[str, Any]] = []
    selected = tuple(
        environment_id
        for environment_id in environment_ids
        if environment_id in avd_homes
    )
    if not selected:
        return []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(10, len(selected))
    ) as executor:
        futures = {
            executor.submit(worker, environment_id): environment_id
            for environment_id in selected
        }
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
    return sorted(rows, key=lambda row: str(row["environment_id"]))


def _bmoca_source_replay_qualified(row: dict[str, Any]) -> bool:
    return (
        row.get("status") == "success"
        and row.get("official_success") is True
        and row.get("method_success") is True
        and int(row.get("model_calls") or 0) == 0
        and int(row.get("fallback_steps") or 0) == 0
    )


def run_bmoca_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """Task-major B-MoCA campaign owned by the existing E2E scheduler."""

    tasks, source_run_logs = _bmoca_manifest_tasks(
        args.bmoca_corpus_manifest,
        args.task,
    )
    campaign_root = args.output_root
    campaign_root.mkdir(parents=True, exist_ok=False)
    progress_csv = campaign_root / "progress.csv"
    progress_jsonl = campaign_root / "progress.jsonl"
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    _write_json(
        campaign_root / "campaign_manifest.json",
        {
            "schema_version": "omniflow.bmoca-e2e-campaign.v1",
            "tasks": tasks,
            "methods": list(_BMOCA_METHODS),
            "environment_ids": list(_BMOCA_ENVIRONMENT_IDS),
            "concurrency_per_method": 10,
            "corpus_manifest": str(args.bmoca_corpus_manifest),
            "bmoca_root": str(args.bmoca_root),
            "source_avd_home": str(args.bmoca_avd_home),
            "started_at": started_at,
        },
    )
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for task in tasks:
        for method in _BMOCA_METHODS:
            for environment_id in _BMOCA_ENVIRONMENT_IDS:
                key = (task, method, environment_id)
                rows[key] = {
                    "task": task,
                    "method": method,
                    "environment_id": environment_id,
                    "status": "pending",
                    "official_success": "",
                    "method_success": "",
                    "actions_executed": 0,
                    "model_calls": 0,
                    "embedding_calls": 0,
                    "fallback_steps": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "error": "",
                }
    _write_bmoca_progress(progress_csv, rows)

    avd_names = _bmoca_avd_names(args.bmoca_root)
    avd_homes: dict[str, Path] = {}
    avd_failures: dict[str, str] = {}
    observed_max_concurrency = 0
    enhancement_reports: list[dict[str, Any]] = []
    memory_reports: list[dict[str, Any]] = []
    for task in tasks:
        task_root = campaign_root / "tasks" / safe_component(task)
        source_run_log = source_run_logs.get(task)
        if source_run_log is None:
            for method in _BMOCA_METHODS:
                for environment_id in _BMOCA_ENVIRONMENT_IDS:
                    key = (task, method, environment_id)
                    rows[key].update(
                        status="prep_failed",
                        error="bmoca_env100_success_source_missing",
                    )
                    _append_bmoca_progress_event(progress_jsonl, rows[key])
            _write_bmoca_progress(progress_csv, rows)
            continue
        method_assets: dict[str, Path] = {}
        method_prep_errors: dict[str, str] = {}
        try:
            store_path, enhancement = _save_bmoca_function_once(
                args=args,
                task=task,
                source_run_log=source_run_log,
                task_root=task_root,
            )
            enhancement_reports.append(enhancement)
            method_assets["ours_replay"] = store_path
        except Exception as error:  # noqa: BLE001 - preserve other methods
            failure_path = task_root / "enhancement_failure.json"
            failure = (
                _read_object(failure_path)
                if failure_path.is_file()
                else {
                    "schema_version": "omniflow.bmoca-function-enhancement.v1",
                    "status": "failed",
                    "task": task,
                    "source_run_log": str(source_run_log),
                    "compile_function_calls": 1,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            if not failure_path.is_file():
                _write_json(failure_path, failure)
            enhancement_reports.append(failure)
            method_prep_errors["ours_replay"] = str(failure["error"])

        if "ours_replay" not in method_assets:
            error = method_prep_errors["ours_replay"]
            for method in _BMOCA_METHODS:
                for environment_id in _BMOCA_ENVIRONMENT_IDS:
                    key = (task, method, environment_id)
                    rows[key].update(status="prep_failed", error=error)
                    _append_bmoca_progress_event(progress_jsonl, rows[key])
            _write_bmoca_progress(progress_csv, rows)
            continue

        omniflow_gate_passed = False
        if "ours_replay" in method_assets and "100" not in avd_homes and "100" not in avd_failures:
            try:
                avd_homes["100"] = _clone_bmoca_avd_home(
                    source_home=args.bmoca_avd_home,
                    target_home=campaign_root / "runtime" / "avd" / "env_100",
                    avd_name=avd_names["100"],
                )
            except Exception as error:  # noqa: BLE001 - preserve the campaign table
                avd_failures["100"] = f"{type(error).__name__}: {error}"
        gate_key = (task, "ours_replay", "100")
        if "ours_replay" not in method_assets:
            gate_rows = []
        elif "100" not in avd_homes:
            rows[gate_key].update(
                status="environment_failure",
                error=avd_failures.get("100", "bmoca_env100_avd_unavailable"),
            )
            _append_bmoca_progress_event(progress_jsonl, rows[gate_key])
            gate_rows: list[dict[str, Any]] = []
        else:
            rows[gate_key].update(status="running", store_path=str(store_path))
            _append_bmoca_progress_event(progress_jsonl, rows[gate_key])
            _write_bmoca_progress(progress_csv, rows)
            gate_rows = _run_bmoca_method_results(
                args=args,
                task=task,
                method="ours_replay",
                store_path=store_path,
                task_root=task_root,
                avd_homes={"100": avd_homes["100"]},
                environment_ids=("100",),
            )
            if gate_rows:
                gate_row = gate_rows[0]
                rows[gate_key] = {
                    key: value
                    for key, value in gate_row.items()
                    if not key.startswith("_")
                }
                _append_bmoca_progress_event(progress_jsonl, rows[gate_key])
                observed_max_concurrency = max(
                    observed_max_concurrency,
                    _max_live_bmoca_results(gate_rows),
                )
        gate_row = rows[gate_key]
        omniflow_gate_passed = _bmoca_source_replay_qualified(gate_row)
        if "ours_replay" in method_assets and not omniflow_gate_passed:
            gate_error = (
                "bmoca_source_replay_gate_failed:"
                f"status={gate_row.get('status')},"
                f"official_success={gate_row.get('official_success')},"
                f"method_success={gate_row.get('method_success')},"
                f"model_calls={gate_row.get('model_calls')},"
                f"fallback_steps={gate_row.get('fallback_steps')}"
            )
            for key, row in rows.items():
                if (
                    key[0] == task
                    and key != gate_key
                    and row["status"] == "pending"
                ):
                    row.update(status="prep_failed", error=gate_error)
                    _append_bmoca_progress_event(progress_jsonl, row)
            _write_bmoca_progress(progress_csv, rows)
            continue

        for method, prepare in (
            ("mobilegpt_replay", _prepare_bmoca_mobilegpt_memory),
            ("skilldroid_replay", _prepare_bmoca_skilldroid_memory),
        ):
            try:
                prepared_path, prepared = prepare(
                    **(
                        {"args": args}
                        if method == "mobilegpt_replay"
                        else {}
                    ),
                    task=task,
                    source_run_log=source_run_log,
                    task_root=task_root,
                )
                method_assets[method] = prepared_path
                memory_reports.append(prepared)
            except Exception as error:  # noqa: BLE001 - one immutable prep
                failure_path = task_root / f"{method.removesuffix('_replay')}_memory_failure.json"
                failure = (
                    _read_object(failure_path)
                    if failure_path.is_file()
                    else {
                        "status": "failed",
                        "task": task,
                        "method": method,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                memory_reports.append(failure)
                method_prep_errors[method] = str(failure["error"])

        for method, error in method_prep_errors.items():
            for environment_id in _BMOCA_ENVIRONMENT_IDS:
                key = (task, method, environment_id)
                rows[key].update(status="prep_failed", error=error)
                _append_bmoca_progress_event(progress_jsonl, rows[key])
        _write_bmoca_progress(progress_csv, rows)

        for environment_id in _BMOCA_ENVIRONMENT_IDS:
            if environment_id in avd_homes or environment_id in avd_failures:
                continue
            try:
                avd_homes[environment_id] = _clone_bmoca_avd_home(
                    source_home=args.bmoca_avd_home,
                    target_home=(
                        campaign_root / "runtime" / "avd" / f"env_{environment_id}"
                    ),
                    avd_name=avd_names[environment_id],
                )
            except Exception as error:  # noqa: BLE001 - preserve the campaign table
                avd_failures[environment_id] = f"{type(error).__name__}: {error}"

        for method in _BMOCA_METHODS:
            if method not in method_assets:
                continue
            if method == "ours_replay" and not omniflow_gate_passed:
                continue
            method_environment_ids = (
                tuple(
                    environment_id
                    for environment_id in _BMOCA_ENVIRONMENT_IDS
                    if environment_id != "100"
                )
                if method == "ours_replay"
                else _BMOCA_ENVIRONMENT_IDS
            )
            for environment_id, error in avd_failures.items():
                if environment_id not in method_environment_ids:
                    continue
                key = (task, method, environment_id)
                if rows[key]["status"] == "pending":
                    rows[key].update(status="environment_failure", error=error)
                    _append_bmoca_progress_event(progress_jsonl, rows[key])
            runnable_homes = {
                environment_id: home
                for environment_id, home in avd_homes.items()
                if environment_id in method_environment_ids
                and rows[(task, method, environment_id)]["status"] == "pending"
            }
            memory_path = (
                None if method == "ours_replay" else method_assets[method]
            )
            for environment_id in runnable_homes:
                key = (task, method, environment_id)
                rows[key].update(
                    status="running",
                    store_path=str(store_path),
                    memory_path=(str(memory_path) if memory_path is not None else ""),
                )
                _append_bmoca_progress_event(progress_jsonl, rows[key])
            _write_bmoca_progress(progress_csv, rows)
            method_rows = _run_bmoca_method_results(
                args=args,
                task=task,
                method=method,
                store_path=store_path,
                memory_path=memory_path,
                task_root=task_root,
                avd_homes=runnable_homes,
                environment_ids=method_environment_ids,
            )
            observed_max_concurrency = max(
                observed_max_concurrency,
                _max_live_bmoca_results(method_rows),
            )
            for row in method_rows:
                key = (task, method, str(row["environment_id"]))
                rows[key] = {key: value for key, value in row.items() if not key.startswith("_")}
                _append_bmoca_progress_event(progress_jsonl, rows[key])
            _write_bmoca_progress(progress_csv, rows)

    status_counts: dict[str, int] = {}
    for row in rows.values():
        status = str(row.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    method_summaries: dict[str, dict[str, Any]] = {}
    for method in _BMOCA_METHODS:
        selected = [row for row in rows.values() if row["method"] == method]
        cross_device = [
            row for row in selected if str(row["environment_id"]) != "100"
        ]
        method_summaries[method] = {
            "result_count": len(selected),
            "official_success_count": sum(
                row.get("official_success") is True for row in selected
            ),
            "official_success_rate": (
                sum(row.get("official_success") is True for row in selected)
                / len(selected)
                if selected
                else 0.0
            ),
            "cross_device_result_count": len(cross_device),
            "cross_device_official_success_count": sum(
                row.get("official_success") is True for row in cross_device
            ),
            "cross_device_official_success_rate": (
                sum(
                    row.get("official_success") is True
                    for row in cross_device
                )
                / len(cross_device)
                if cross_device
                else 0.0
            ),
            "actions_executed": sum(
                int(row.get("actions_executed") or 0) for row in selected
            ),
            "model_calls": sum(
                int(row.get("model_calls") or 0) for row in selected
            ),
            "embedding_calls": sum(
                int(row.get("embedding_calls") or 0) for row in selected
            ),
            "fallback_steps": sum(
                int(row.get("fallback_steps") or 0) for row in selected
            ),
            "prompt_tokens": sum(
                int(row.get("prompt_tokens") or 0) for row in selected
            ),
            "completion_tokens": sum(
                int(row.get("completion_tokens") or 0) for row in selected
            ),
            "total_tokens": sum(
                int(row.get("total_tokens") or 0) for row in selected
            ),
        }
    environment_summaries: dict[str, dict[str, Any]] = {}
    for environment_id in _BMOCA_ENVIRONMENT_IDS:
        environment_summaries[environment_id] = {}
        for method in _BMOCA_METHODS:
            selected = [
                row
                for row in rows.values()
                if row["method"] == method
                and row["environment_id"] == environment_id
            ]
            environment_summaries[environment_id][method] = {
                "result_count": len(selected),
                "official_success_count": sum(
                    row.get("official_success") is True for row in selected
                ),
            }
    summary = {
        "schema_version": "omniflow.bmoca-e2e-campaign-summary.v1",
        "status": "complete",
        "started_at": started_at,
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "task_count": len(tasks),
        "method_count": len(_BMOCA_METHODS),
        "environment_count": len(_BMOCA_ENVIRONMENT_IDS),
        "result_count": len(rows),
        "status_counts": status_counts,
        "official_success_count": sum(
            row.get("official_success") is True for row in rows.values()
        ),
        "methods": method_summaries,
        "environments": environment_summaries,
        "observed_max_live_results": observed_max_concurrency,
        "process_overlap_proven": observed_max_concurrency > 1,
        "enhancement_count": len(enhancement_reports),
        "enhancement_success_count": sum(
            report.get("enhanced") is True for report in enhancement_reports
        ),
        "memory_preparation_count": len(memory_reports),
        "memory_preparation_success_count": sum(
            report.get("status") == "prepared" for report in memory_reports
        ),
        "progress_csv": str(progress_csv),
        "progress_jsonl": str(progress_jsonl),
    }
    _write_json(campaign_root / "campaign_summary.json", summary)
    return summary


def _parse_source_device(value: str) -> tuple[str, str, int]:
    parts = value.split(":")
    if len(parts) != 3 or not parts[0] or not parts[2].isdigit():
        raise argparse.ArgumentTypeError(f"invalid_source_device:{value}")
    console_port = int(parts[2])
    if parts[1] != f"emulator-{console_port}":
        raise argparse.ArgumentTypeError(f"source_serial_console_mismatch:{value}")
    return parts[0], parts[1], console_port


def build_parser() -> argparse.ArgumentParser:
    repo = Path(__file__).resolve().parents[2]
    data = repo / "data"
    android_world = repo.parent / "releases" / f"android-world-{ANDROIDWORLD_REVISION}"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Experiment JSON; use the shell entry.")
    parser.add_argument(
        "--environment",
        choices=("androidworld", "bmoca"),
        default="androidworld",
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument(
        "--method",
        dest="e2e_method",
        default=DEFAULT_METHOD,
        help="Method, comma-separated methods, or all.",
    )
    parser.add_argument(
        "--device",
        dest="e2e_device",
        default=DEFAULT_DEVICE,
        help="Device, comma-separated devices, or all.",
    )
    parser.add_argument(
        "--source-seed",
        dest="e2e_source_seed",
        type=int,
        default=SOURCE_SEED,
    )
    parser.add_argument(
        "--evaluation-seed",
        dest="e2e_evaluation_seed",
        type=int,
        default=TASK_SEED,
    )
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--max-fallback-steps", type=int, default=MAX_FALLBACK_STEPS)
    parser.add_argument(
        "--deadline",
        "--task-deadline-sec",
        dest="task_deadline_sec",
        type=int,
        default=TASK_DEADLINE_SEC,
    )
    parser.add_argument("--model", dest="formal_model", default=FORMAL_MODEL)
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(
        repo=repo,
        script=repo / "scripts" / "exp" / "run_androidworld.sh",
        memory_index=data / "current.json",
        asset_root=data,
        results_root=data,
        output_root=data,
        android_world_root=android_world,
        omnitransfer_root=Path.home() / "Projects" / "Omni" / "OmniTransfer",
        mobilegpt_root=data / "runtime" / "external" / "mobilegpt",
        appagent_root=data / "runtime" / "external" / "appagent",
        appagent_memory_root=None,
        autodroid_root=data / "runtime" / "external" / "autodroid",
        autodroid_memory_root=data / "runtime" / "autodroid" / "androidworld_apps",
        autodroid_app="",
        autodroid_policy="replay",
        python_bin=Path(sys.executable),
        adb_path=Path(os.environ.get("ADB_PATH") or "adb"),
        emulator_bin=None,
        source_device=SOURCE_DEVICE,
        source_avd=SOURCE_AVD,
        emulator_gpu="swiftshader_indirect",
        appagent_model=APPAGENT_MODEL,
        bmoca_root=(Path(value) if (value := os.environ.get("OMNIFLOW_BMOCA_ROOT")) else None),
        bmoca_corpus_manifest=(
            Path(value)
            if (value := os.environ.get("OMNIFLOW_BMOCA_CORPUS_MANIFEST"))
            else None
        ),
        bmoca_avd_home=(
            Path(value) if (value := os.environ.get("OMNIFLOW_BMOCA_AVD_HOME")) else None
        ),
        bmoca_android_env_root=(
            Path(value)
            if (value := os.environ.get("OMNIFLOW_BMOCA_ANDROID_ENV_ROOT"))
            else None
        ),
        android_sdk_root=(
            Path(value) if (value := os.environ.get("ANDROID_HOME")) else None
        ),
        attempt_id="",
        ensure_function=False,
        mobilegpt_memory_only=False,
        source_only=False,
        manual_source=False,
        manual_reuse_emulator=False,
    )
    return parser


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    path_fields = (
        "repo",
        "script",
        "output_root",
        "omnitransfer_root",
        "python_bin",
        "memory_index",
        "asset_root",
        "results_root",
        "android_world_root",
        "mobilegpt_root",
        "appagent_root",
        "appagent_memory_root",
        "adb_path",
        "emulator_bin",
    )
    for field in path_fields:
        value = getattr(args, field, None)
        if value is not None:
            setattr(args, field, value.expanduser())
    if getattr(args, "environment", "androidworld") == "bmoca":
        for field in (
            "bmoca_root",
            "bmoca_corpus_manifest",
            "bmoca_avd_home",
            "bmoca_android_env_root",
            "android_sdk_root",
            "mobilegpt_root",
        ):
            value = getattr(args, field)
            if value is None:
                raise ValueError(f"bmoca_required_path_missing:{field}")
            setattr(args, field, value.expanduser().resolve())
        if not args.bmoca_corpus_manifest.is_file():
            raise FileNotFoundError(
                f"bmoca_corpus_manifest_missing:{args.bmoca_corpus_manifest}"
            )
        for field in (
            "bmoca_root",
            "bmoca_avd_home",
            "bmoca_android_env_root",
            "android_sdk_root",
            "mobilegpt_root",
        ):
            if not getattr(args, field).is_dir():
                raise FileNotFoundError(
                    f"bmoca_required_directory_missing:{field}:{getattr(args, field)}"
                )
        if str(args.formal_model).strip().casefold() not in {"glm-4.6v", "glm-5.1"}:
            raise ValueError("bmoca_campaign_requires_GLM-4.6V")
        if args.max_fallback_steps != 0:
            raise ValueError("bmoca_campaign_fallback_must_be_zero")
        return args
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _resolve_args(build_parser().parse_args(argv))
    if args.environment == "bmoca":
        result = run_bmoca_pipeline(args)
    else:
        result = run_pipeline(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.environment == "bmoca":
        return 0 if result.get("status") == "complete" else 1
    if args.dry_run:
        return 0
    if args.source_only:
        return 0 if result.get("status") == "collected" else 1
    if args.mobilegpt_memory_only:
        return 0 if result.get("status") == "validated" else 1
    counts = result.get("counts") if isinstance(result, dict) else None
    if result.get("function_retry_needed") is True:
        return 75
    return (
        0
        if result.get("status") == "complete"
        and isinstance(counts, dict)
        and counts.get("pending") == 0
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
