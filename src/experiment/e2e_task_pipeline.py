"""Bounded source-to-result AndroidWorld task pipeline."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
from typing import Any, Callable, Sequence

from omniflow.core.trajectory import require_complete_source_run_log
from omniflow.functions.assets import save_function
from omniflow.vlm.model_config import resolve_openai_compatible_config
from src.experiment.androidworld import CanonicalRunLog, build_fixed_replay_command
from src.experiment.local_data import (
    canonical_mobilegpt_memory_from_memory,
    load_local_data,
    registered_result_plan_from_memory,
)
from src.experiment.batch_outcomes import (
    concluded_result_keys,
    record_result_outcome,
    summarize_results,
)
from src.experiment.protocol import (
    BMOCA_RESULT_TIMEOUT_SEC,
    DEVICES,
    FORMAL_MODEL,
    FORMAL_MODEL_BASE_URL,
    FUNCTION_ENHANCEMENT_TIMEOUT_SEC,
    MAX_FALLBACK_STEPS,
    MAX_STEPS,
    METHODS,
    SOURCE_AVD,
    SOURCE_DEVICE,
    SOURCE_MAX_STEPS,
    SOURCE_SEED,
    STEP_TIMEOUT_SEC,
    TASK_DEADLINE_SEC,
    TASK_SEED,
)
from src.integrations.mobilegpt_converter import (
    convert_runlog_to_mobilegpt_memory,
)
from src.integrations.runlog import project_androidworld_step_actions
from src.integrations.skilldroid_replay import compile_droidrun_macro


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_object(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{resolved}")
    return value


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _safe_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return normalized.strip("._") or "item"


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
    """Run one child in its own process group and preserve its complete log."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    if timeout_sec <= 0:
        log_path.write_text(
            "global task deadline exceeded before launch\n",
            encoding="utf-8",
        )
        return {
            "command": list(command),
            "returncode": 124,
            "timed_out": True,
            "wall_sec": 0.0,
            "log_path": str(log_path),
        }
    started = time.monotonic()
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    timed_out = False
    with log_path.open("x", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=max(0.1, float(timeout_sec)))
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            returncode = 124
    return {
        "command": list(command),
        "returncode": int(returncode),
        "timed_out": timed_out,
        "wall_sec": round(time.monotonic() - started, 6),
        "log_path": str(log_path),
        "process_pid": int(process.pid),
        "started_at": started_at,
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def _usage_from_result(row: dict[str, Any]) -> dict[str, int]:
    return {
        "model_calls": int(row.get("model_calls") or 0),
        "prompt_tokens": int(row.get("prompt_tokens") or 0),
        "completion_tokens": int(row.get("completion_tokens") or 0),
        "total_tokens": int(row.get("total_tokens") or 0),
    }


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


def _adb_output(args: argparse.Namespace, *command: str) -> str:
    try:
        completed = subprocess.run(
            [str(args.adb_path), *command],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout.strip()


def _source_device_ready(args: argparse.Namespace) -> bool:
    source_serial = args.source_device[1]
    avd_name = _adb_output(
        args,
        "-s",
        source_serial,
        "emu",
        "avd",
        "name",
    ).replace("\r", "").splitlines()
    active_avd = avd_name[0].strip() if avd_name else ""
    return (
        _adb_output(args, "-s", source_serial, "get-state") == "device"
        and active_avd == args.source_avd
        and _adb_output(
            args,
            "-s",
            source_serial,
            "shell",
            "getprop",
            "sys.boot_completed",
        ).replace("\r", "")
        == "1"
    )


def ensure_source_device(
    *,
    args: argparse.Namespace,
    attempt_root: Path,
    deadline: Deadline,
) -> dict[str, Any]:
    started = time.monotonic()
    source_serial = args.source_device[1]
    source_console_port = args.source_device[2]
    if source_serial in _adb_output(args, "devices"):
        _adb_output(args, "-s", source_serial, "emu", "kill")
        time.sleep(2)
    log_path = attempt_root / "preflight" / "source_emulator.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("x", encoding="utf-8")
    emulator_process = subprocess.Popen(
        [
            str(args.emulator_bin),
            "-avd",
            args.source_avd,
            "-port",
            str(source_console_port),
            "-grpc",
            str(source_console_port + 3000),
            "-no-window",
            "-no-audio",
            "-no-boot-anim",
            "-read-only",
            "-no-snapshot-load",
            "-no-snapshot-save",
            "-gpu",
            args.emulator_gpu,
        ],
        cwd=args.repo,
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_file.close()
    boot_timeout = deadline.remaining(TASK_DEADLINE_SEC)
    boot_deadline = time.monotonic() + boot_timeout
    while time.monotonic() < boot_deadline:
        if _source_device_ready(args):
            break
        poll = getattr(emulator_process, "poll", None)
        returncode = poll() if callable(poll) else None
        if returncode is not None:
            raise RuntimeError(f"source_emulator_exited:{returncode}")
        time.sleep(1)
    else:
        raise RuntimeError(f"source_emulator_not_ready:{source_serial}")
    preflight_path = attempt_root / "preflight" / "source_native.json"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        str(path)
        for path in (
            args.repo,
            args.repo / "src",
            args.android_world_root,
            environment.get("PYTHONPATH", ""),
        )
        if str(path)
    )
    command = [
        str(args.python_bin),
        str(args.runtime_preflight),
        "--repo",
        str(args.asset_root),
        "--android-world-root",
        str(args.android_world_root),
        "--code-root",
        str(args.repo),
        "--profile",
        "androidworld_native",
        "--serial",
        source_serial,
        "--require-kvm",
        "--require-device",
        "--source-index",
        str(args.memory_index),
        "--source-task",
        args.task,
        *(["--source-only"] if getattr(args, "source_only", False) else []),
        "--json-out",
        str(preflight_path),
    ]
    result = run_logged_command(
        command,
        cwd=args.repo,
        environment=environment,
        log_path=attempt_root / "preflight" / "source_native.log",
        timeout_sec=deadline.remaining(STEP_TIMEOUT_SEC),
    )
    if result["returncode"] != 0:
        raise RuntimeError(f"source_runtime_preflight_failed:{result['returncode']}")
    return {
        **result,
        "status": "ready",
        "launched": True,
        "serial": source_serial,
        "avd": args.source_avd,
        "wall_sec": round(time.monotonic() - started, 6),
        "model_calls": 0,
        "total_tokens": 0,
        "preflight": str(preflight_path),
    }


def _canonical_source(
    memory_index: Path,
    task: str,
    *,
    require_protocol_seed: bool = True,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    registry = load_local_data(memory_index)
    record = registry.get("canonical", {}).get("source_run_logs", {}).get(task)
    if not isinstance(record, dict):
        raise ValueError(f"canonical_source_missing:{task}")
    if not require_protocol_seed and record.get("latest_official_success_source") is not True:
        fallback = _audited_historical_source(memory_index, task)
        if fallback is not None:
            return registry, fallback[0], fallback[1]
    path = Path(str(record.get("object_path") or "")).expanduser().resolve()
    try:
        if not path.is_file() or _sha256(path) != str(record.get("sha256") or ""):
            raise ValueError(f"canonical_source_object_invalid:{task}:{path}")
        run_log = require_complete_source_run_log(_read_object(path))
        if run_log["task_name"] != task:
            raise ValueError(f"canonical_source_task_mismatch:{task}")
        if require_protocol_seed and run_log["seed"] != SOURCE_SEED:
            raise ValueError(f"canonical_source_seed_mismatch:{task}:{run_log['seed']}")
        if run_log["success"] is not True:
            raise ValueError(f"canonical_source_not_successful:{task}")
        return registry, path, run_log
    except (FileNotFoundError, TypeError, ValueError):
        if require_protocol_seed:
            raise
        fallback = _audited_historical_source(memory_index, task)
        if fallback is None:
            raise
        return registry, fallback[0], fallback[1]


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
    candidates = sorted(
        path
        for path in (task_root / "source_attempts").glob("*/" + attempt_id + "/**/target.run_log.json")
        if path.is_file()
    )
    for candidate in candidates:
        try:
            run_log = require_complete_source_run_log(_read_object(candidate))
        except (TypeError, ValueError):
            continue
        if (
            run_log["task_name"] == task
            and run_log["seed"] == SOURCE_SEED
            and run_log["success"] is True
        ):
            return candidate.resolve(), run_log
    return None


def _resolve_reference(index_path: Path, value: Any) -> Path:
    path = Path(str(value or "")).expanduser()
    if not path.is_absolute():
        path = index_path.parent / path
    return path.resolve()


def _official_success(row: dict[str, Any]) -> bool:
    validator = row.get("androidworld_validator_result")
    return bool(
        row.get("official_validator_success") is True
        or isinstance(validator, dict)
        and validator.get("success") is True
    )


def _function_replay_success(row: dict[str, Any]) -> bool:
    """Return the atomic Function result, independent of task validation."""

    canonical = row.get("canonical_run")
    if not isinstance(canonical, dict):
        return False
    diagnostics = canonical.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return False
    execution_summary = diagnostics.get("execution_summary")
    if not isinstance(execution_summary, dict):
        return False
    if execution_summary.get("success") is not True:
        return False
    trace = diagnostics.get("execution_trace")
    if isinstance(trace, list):
        return bool(trace) and all(
            isinstance(step, dict)
            and isinstance(step.get("result"), dict)
            and step["result"].get("success") is True
            for step in trace
        )
    return int(execution_summary.get("steps") or 0) > 0


def _captured_androidworld_state(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("fixed_replay_capture_observation_required")
    state = record.get("androidworld_state")
    if not isinstance(state, dict):
        raise ValueError("fixed_replay_capture_androidworld_state_required")
    pixels = state.get("pixels")
    if not isinstance(pixels, dict):
        raise ValueError("fixed_replay_capture_screenshot_required")
    screenshot = Path(str(pixels.get("path") or "")).expanduser().resolve()
    if not screenshot.is_file():
        raise FileNotFoundError(f"fixed_replay_capture_screenshot_missing:{screenshot}")
    expected = str(pixels.get("sha256") or "").strip().lower()
    actual = _sha256(screenshot)
    if expected != actual:
        raise ValueError(
            "fixed_replay_capture_screenshot_hash_mismatch:"
            f"expected={expected}:actual={actual}"
        )
    return json.loads(json.dumps(state, ensure_ascii=False))


def _fixed_replay_source_step_width(source_step: dict[str, Any]) -> int:
    """Return the number of raw AndroidWorld actions for one semantic step."""

    action = source_step.get("action")
    action_type = str(action.get("action_type") or "") if isinstance(action, dict) else ""
    if action_type in {"status", "unknown"}:
        return 0
    if action_type == "answer":
        return 1
    return len(project_androidworld_step_actions(source_step))


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
                },
            }
        )
    captured = require_complete_source_run_log(
        {
            "schema_version": "omniflow.run_log.v1",
            "run_id": str(replay.get("run_id") or "fixed_replay_capture"),
            "task_name": source_run_log["task_name"],
            "goal": source_run_log["goal"],
            "task_parameters": dict(source_run_log["task_parameters"]),
            "seed": SOURCE_SEED,
            "status": "succeeded",
            "success": True,
            "validator": {
                "official": True,
                "success": True,
                "reward": float(reward),
            },
            "provenance": {"kind": "runtime"},
            "steps": steps,
            "final_observation": final_observation,
            "diagnostics": {
                "capture": "fixed_replay",
                "source_run_log": str(source_path),
                "source_run_log_sha256": _sha256(source_path),
                "model_calls": 0,
            },
        }
    )
    _write_json(output_path, captured)
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
    command_spec = build_fixed_replay_command(
        item,
        android_world_root=args.android_world_root,
        output_root=phase_root,
        method_name="source_capture",
        device_label=source_label,
        serial=source_serial,
        console_port=source_console_port,
        adb_path=str(args.adb_path),
        max_steps=len(source_run_log["steps"]) + 1,
        timeout_sec=int(TASK_DEADLINE_SEC),
        task_random_seed=SOURCE_SEED,
        task_params_override=dict(source_run_log["task_parameters"]),
        perform_emulator_setup=True,
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
    captured_path = phase_root / "source.run_log.json"
    captured = _captured_source_run_log(
        source_path=source_path,
        source_run_log=source_run_log,
        raw_replay_result=Path(str(command_spec.metadata["raw_replay_result"])),
        task_result=row,
        output_path=captured_path,
    )
    # Source-only collection produces an immutable candidate attempt.  Do not
    # refresh the long-term memory here: a changed source hash may invalidate
    # derived baseline assets, and memory selection must be audited after the
    # complete collection batch.  The caller still gets the fully validated
    # captured RunLog for the source-only report.
    selected_path, selected = captured_path, captured
    result["input_source"] = str(source_path)
    result["input_source_sha256"] = _sha256(source_path)
    result["captured_source"] = str(captured_path)
    result["captured_source_sha256"] = _sha256(captured_path)
    result["captured_steps"] = len(captured["steps"])
    result["selected_source"] = str(selected_path)
    result["status"] = "collected"
    return selected_path, selected, result


def _canonical_function_store(
    memory_index: Path,
    task: str,
) -> dict[str, Any] | None:
    registry = load_local_data(memory_index)
    record = registry.get("canonical", {}).get("function_stores", {}).get(task)
    return dict(record) if isinstance(record, dict) else None


def prepare_function_asset(
    *,
    args: argparse.Namespace,
    source_path: Path,
    run_log: dict[str, Any],
    attempt_root: Path,
    deadline: Deadline,
) -> tuple[dict[str, Any], dict[str, Any]]:
    existing = _canonical_function_store(args.memory_index, args.task)
    if existing is None:
        raise FileNotFoundError(f"canonical_function_store_missing:{args.task}")
    if str(existing.get("source_run_log_sha256") or "") != _sha256(source_path):
        raise ValueError(f"canonical_function_source_mismatch:{args.task}")
    store_path = Path(str(existing["store_path"])).resolve()
    source_calls = existing.get("source_calls")
    if (
        not isinstance(source_calls, list)
        or len(source_calls) != 1
        or any(
            not isinstance(source_call, dict)
            or not str(source_call.get("function_id") or "").strip()
            or not isinstance(source_call.get("arguments"), dict)
            for source_call in source_calls
        )
    ):
        raise ValueError(f"canonical_function_source_calls_missing:{args.task}")
    return existing, {
        "status": "reused",
        "model_calls": 0,
        "total_tokens": 0,
        "store": str(store_path),
        "source_calls": source_calls,
    }


def qualify_source_function(
    *,
    args: argparse.Namespace,
    source_path: Path,
    run_log: dict[str, Any],
    function_store: dict[str, Any],
    source_call: dict[str, Any],
    attempt_root: Path,
    deadline: Deadline,
    round_index: int,
) -> dict[str, Any]:
    output_root = attempt_root / "source_qualification" / f"round_{round_index:02d}"
    store_path = Path(str(function_store["store_path"])).resolve()
    command = [
        str(args.python_bin),
        "-m",
        "src.integrations.android_world.launch",
        "--function-id",
        str(source_call["function_id"]),
        "--function-arguments-json",
        json.dumps(source_call["arguments"], ensure_ascii=False),
        "--android-world-root",
        str(args.android_world_root),
        "--tasks",
        args.task,
        "--task-random-seed",
        str(SOURCE_SEED),
        "--n-task-combinations",
        "1",
        "--console-port",
        str(args.source_device[2]),
        "--agent",
        "omniflow",
        "--max-steps",
        str(SOURCE_MAX_STEPS),
        "--output-path",
        str(output_root),
        "--store-path",
        str(store_path),
        "--task-params-json",
        json.dumps(run_log["task_parameters"], ensure_ascii=False),
        "--fixed-task-seed",
        "--perform-emulator-setup",
        "--adb-path",
        str(args.adb_path),
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "ANDROID_SERIAL": args.source_device[1],
            "OMNIFLOW_ANDROIDWORLD_MAX_FALLBACK_STEPS": "0",
            "OMNITRANSFER_ROOT": str(args.omnitransfer_root),
            "PYTHONPATH": f"{args.repo}:{args.repo / 'src'}:{args.android_world_root}",
        }
    )
    for key in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "LLMTHU_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
    ):
        environment.pop(key, None)
    result = run_logged_command(
        command,
        cwd=args.repo,
        environment=environment,
        log_path=output_root.parent / "qualification.log",
        timeout_sec=deadline.remaining(TASK_DEADLINE_SEC),
    )
    row = _last_jsonl_row(output_root / "task_results.jsonl")
    canonical = row.get("canonical_run")
    canonical = canonical if isinstance(canonical, dict) else {}
    result.update(
        {
            "qualification_scope": "atomic_function_replay",
            "official_validator_success": _official_success(row),
            "function_replay_success": _function_replay_success(row),
            "model_calls": int(row.get("model_calls") or 0),
            "fallback_steps": int(row.get("fallback_steps") or 0),
            "task_run_status": str(canonical.get("status") or ""),
            "function_id": str(row.get("function_id") or source_call["function_id"]),
            "source_run_log": str(source_path),
            "source_run_log_sha256": _sha256(source_path),
            "store_path": str(store_path),
            "store_sha256": _sha256(store_path),
            "transfer_states_sha256": str(
                function_store.get("transfer_states_sha256") or ""
            ),
            "source_call": source_call,
        }
    )
    result["qualified"] = bool(
        result["returncode"] == 0
        and result["function_replay_success"]
        and result["official_validator_success"]
        and result["model_calls"] == 0
        and result["fallback_steps"] == 0
    )
    _write_json(output_root.parent / "qualification.json", result)
    return result


def prepare_mobilegpt_memory(
    *,
    args: argparse.Namespace,
    attempt_root: Path,
    deadline: Deadline,
) -> tuple[Path, dict[str, Any]]:
    existing = canonical_mobilegpt_memory_from_memory(
        memory_index=args.memory_index,
        task_name=args.task,
    )
    if existing is not None:
        return Path(str(existing["memory_root"])).resolve(), {
            "status": "reused",
            "model_calls": 0,
            "total_tokens": 0,
            "memory_root": str(existing["memory_root"]),
        }
    output_root = attempt_root / "assets" / "mobilegpt"
    source_index = args.memory_index
    result = run_logged_command(
        [
            str(args.python_bin),
            "-m",
            "src.experiment.mobilegpt_source",
            "batch",
            "--index",
            str(source_index),
            "--task",
            args.task,
            "--mobilegpt-root",
            str(args.mobilegpt_root),
            "--output-root",
            str(output_root),
            "--model",
            args.formal_model,
            "--memory-index",
            str(args.memory_index),
        ],
        cwd=args.repo,
        environment=dict(os.environ),
        log_path=attempt_root / "prep" / "mobilegpt.log",
        timeout_sec=deadline.remaining(TASK_DEADLINE_SEC),
    )
    stats = []
    for path in output_root.rglob("*stats.jsonl"):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                stats.append(value)
    phase = {
        **result,
        "status": "created",
        "model_calls": sum(
            str(row.get("event") or "") in {"chat_call", "embedding_call"}
            for row in stats
        ),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in stats),
    }
    if result["returncode"] != 0:
        raise PipelinePhaseError(
            f"mobilegpt_memory_prep_failed:{result['returncode']}",
            phase,
        )
    existing = canonical_mobilegpt_memory_from_memory(
        memory_index=args.memory_index,
        task_name=args.task,
    )
    if existing is None:
        raise PipelinePhaseError("mobilegpt_memory_not_registered", phase)
    phase["memory_root"] = str(existing["memory_root"])
    return Path(str(existing["memory_root"])).resolve(), phase


def prepare_appagent_memory(
    *,
    args: argparse.Namespace,
    attempt_root: Path,
    deadline: Deadline,
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
        args.formal_model,
    ]
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


def _concluded_results(
    args: argparse.Namespace,
    outcomes_root: Path,
    attempt_id: str,
) -> set[tuple[str, str]]:
    plan = registered_result_plan_from_memory(
        memory_index=args.memory_index,
        task_name=args.task,
        methods=METHODS,
        devices=tuple(device[0] for device in DEVICES),
        source_seed=SOURCE_SEED,
        evaluation_seed=TASK_SEED,
        formal_max_steps=int(args.max_steps),
    )
    return set(plan["completed"]) | concluded_result_keys(
        outcomes_root=outcomes_root,
        task_name=args.task,
        methods=METHODS,
        devices=tuple(device[0] for device in DEVICES),
        source_seed=SOURCE_SEED,
        evaluation_seed=TASK_SEED,
        attempt_id=attempt_id,
    )


def _result_environment(
    *,
    args: argparse.Namespace,
    attempt_id: str,
    attempt_root: Path,
    method: str,
    device: tuple[str, str, int],
    store_path: Path,
    mobilegpt_memory: Path | None,
    appagent_memory: Path | None,
) -> dict[str, str]:
    label, serial, port = device
    result_attempt_id = f"{attempt_id}.{method}.{label}"
    result_attempt_root = (
        attempt_root / "target_attempts" / label / method / result_attempt_id
    )
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
            "OMNIFLOW_BATCH_ATTEMPT_ID": result_attempt_id,
            "OMNIFLOW_BATCH_CHILD": "1",
            "OMNIFLOW_ANDROIDWORLD_TASK": args.task,
            "OMNIFLOW_ANDROIDWORLD_METHOD": method,
            "OMNIFLOW_ANDROIDWORLD_DEVICE": f"{label}:{serial}:{port}",
            "OMNIFLOW_ANDROIDWORLD_STORE_PATH": str(store_path),
            "OMNIFLOW_ANDROIDWORLD_MAX_STEPS": str(args.max_steps),
            "OMNIFLOW_ANDROIDWORLD_MAX_FALLBACK_STEPS": str(
                args.max_fallback_steps
            ),
            "OMNIFLOW_ANDROIDWORLD_OUTPUT_PATH": str(result_attempt_root),
        }
    )
    if mobilegpt_memory is not None:
        environment["OMNIFLOW_MOBILEGPT_SOURCE_MEMORY_ROOT"] = str(
            mobilegpt_memory
        )
    if appagent_memory is not None:
        environment["OMNIFLOW_APPAGENT_MEMORY_ROOT"] = str(appagent_memory)
    return environment


def run_target_workers(
    *,
    args: argparse.Namespace,
    deadline: Deadline,
    attempt_id: str,
    attempt_root: Path,
    outcomes_root: Path,
    store_path: Path,
    mobilegpt_memory: Path | None,
    appagent_memory: Path | None,
    blocked_methods: dict[str, tuple[str, str, str]],
    command_runner: Callable[..., dict[str, Any]] = run_logged_command,
) -> list[dict[str, Any]]:
    """Run methods sequentially per device and devices concurrently."""

    completed = _concluded_results(args, outcomes_root, attempt_id)
    stop_event = threading.Event()
    for method, (status, stage, evidence) in blocked_methods.items():
        for label, serial, _ in DEVICES:
            if (method, label) in completed:
                continue
            record_result_outcome(
                outcomes_root=outcomes_root,
                task_name=args.task,
                method=method,
                device=label,
                device_serial=serial,
                attempt_id=attempt_id,
                source_seed=SOURCE_SEED,
                evaluation_seed=TASK_SEED,
                status=status,
                stage=stage,
                task_log=evidence if Path(evidence).is_file() else None,
                artifact_root=None,
            )

    def worker(device: tuple[str, str, int]) -> list[dict[str, Any]]:
        label, serial, _ = device
        worker_results: list[dict[str, Any]] = []
        for method in METHODS:
            if stop_event.is_set():
                break
            if method in blocked_methods or (method, label) in completed:
                worker_results.append(
                    {"method": method, "device": label, "status": "reused_or_blocked"}
                )
                continue
            if deadline.expired:
                outcome_path = record_result_outcome(
                    outcomes_root=outcomes_root,
                    task_name=args.task,
                    method=method,
                    device=label,
                    device_serial=serial,
                    attempt_id=attempt_id,
                    source_seed=SOURCE_SEED,
                    evaluation_seed=TASK_SEED,
                    status="deadline_exceeded",
                    stage="target_episode",
                    artifact_root=attempt_root,
                    outer_wall_sec=0,
                )
                worker_results.append(
                    {"method": method, "device": label, "status": "deadline_exceeded"}
                )
                stop_event.set()
                raise PipelinePhaseError(
                    "target_episode_environment_failure",
                    {
                        "status": "deadline_exceeded",
                        "stage": "target_episode",
                        "method": method,
                        "device": label,
                        "outcome": str(outcome_path),
                    },
                )
            log_path = attempt_root / "logs" / label / f"{method}.log"
            result = command_runner(
                ["bash", str(args.script)],
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
                ),
                log_path=log_path,
                timeout_sec=deadline.remaining(TASK_DEADLINE_SEC),
            )
            if result.get("returncode") == 0:
                completed.add((method, label))
            if (method, label) not in completed:
                status = (
                    "deadline_exceeded"
                    if result.get("timed_out") or deadline.expired
                    else "environment_failure"
                )
                outcome_path = record_result_outcome(
                    outcomes_root=outcomes_root,
                    task_name=args.task,
                    method=method,
                    device=label,
                    device_serial=serial,
                    attempt_id=attempt_id,
                    source_seed=SOURCE_SEED,
                    evaluation_seed=TASK_SEED,
                    status=status,
                    stage="target_episode",
                    task_log=log_path,
                    artifact_root=(
                        attempt_root
                        / "target_attempts"
                        / label
                        / method
                        / f"{attempt_id}.{method}.{label}"
                    ),
                    outer_wall_sec=float(result.get("wall_sec") or 0),
                )
                stop_event.set()
                raise PipelinePhaseError(
                    "target_episode_environment_failure",
                    {
                        "status": status,
                        "stage": "target_episode",
                        "method": method,
                        "device": label,
                        "outcome": str(outcome_path),
                    },
                )
            worker_results.append({"method": method, "device": label, **result})
        return worker_results

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker, device) for device in DEVICES]
        return [result for future in futures for result in future.result()]


def _blocked_all(
    *,
    args: argparse.Namespace,
    attempt_id: str,
    attempt_root: Path,
    outcomes_root: Path,
    status: str,
    stage: str,
    evidence: Path,
) -> None:
    if getattr(args, "source_qualification_only", False) or getattr(
        args, "source_only", False
    ):
        return
    completed = _concluded_results(args, outcomes_root, attempt_id)
    for method in METHODS:
        for label, serial, _ in DEVICES:
            if (method, label) in completed:
                continue
            record_result_outcome(
                outcomes_root=outcomes_root,
                task_name=args.task,
                method=method,
                device=label,
                device_serial=serial,
                attempt_id=attempt_id,
                source_seed=SOURCE_SEED,
                evaluation_seed=TASK_SEED,
                status=status,
                stage=stage,
                task_log=evidence if evidence.is_file() else None,
                artifact_root=attempt_root,
            )


def _report(
    *,
    args: argparse.Namespace,
    attempt_id: str,
    attempt_root: Path,
    outcomes_root: Path,
    deadline: Deadline,
    phases: dict[str, Any],
) -> dict[str, Any]:
    if getattr(args, "source_only", False):
        source = phases.get("source")
        collected = isinstance(source, dict) and source.get("status") == "collected"
        summary = {
            "schema_version": "omniflow.androidworld.source-collection-report.v1",
            "immutable": True,
            "task": args.task,
            "attempt_id": attempt_id,
            "status": "collected" if collected else "failed",
            "source_seed": SOURCE_SEED,
            "outer_wall_sec": deadline.elapsed,
            "model_calls": sum(
                int(phase.get("model_calls") or 0)
                for phase in phases.values()
                if isinstance(phase, dict)
            ),
            "total_tokens": sum(
                int(phase.get("total_tokens") or 0)
                for phase in phases.values()
                if isinstance(phase, dict)
            ),
            "phases": phases,
        }
        _write_json(attempt_root / "pipeline_summary.json", summary)
        return summary
    if getattr(args, "source_qualification_only", False):
        qualification = phases.get("source_qualification")
        qualified = (
            isinstance(qualification, dict)
            and qualification.get("qualified") is True
        )
        summary = {
            "schema_version": "omniflow.androidworld.source-qualification-report.v1",
            "immutable": True,
            "task": args.task,
            "attempt_id": attempt_id,
            "status": "qualified" if qualified else "failed",
            "source_seed": SOURCE_SEED,
            "outer_wall_sec": deadline.elapsed,
            "model_calls": sum(
                int(phase.get("model_calls") or 0)
                for phase in phases.values()
                if isinstance(phase, dict)
            ),
            "total_tokens": sum(
                int(phase.get("total_tokens") or 0)
                for phase in phases.values()
                if isinstance(phase, dict)
            ),
            "phases": phases,
        }
        _write_json(attempt_root / "pipeline_summary.json", summary)
        return summary
    result_summary = summarize_results(
        memory_index=args.memory_index,
        outcomes_root=outcomes_root,
        tasks=(args.task,),
        methods=METHODS,
        devices=tuple(device[0] for device in DEVICES),
        source_seed=SOURCE_SEED,
        evaluation_seed=TASK_SEED,
        attempt_id=attempt_id,
    )
    prep_model_calls = sum(
        int(phase.get("model_calls") or 0)
        for phase in phases.values()
        if isinstance(phase, dict)
    )
    prep_total_tokens = sum(
        int(phase.get("total_tokens") or 0)
        for phase in phases.values()
        if isinstance(phase, dict)
    )
    counts = result_summary["counts"]
    status = (
        "complete"
        if counts["pending"] == 0
        else "deadline_exceeded"
        if deadline.expired
        else "partial"
    )
    total_model_calls = int(result_summary["model_calls"]) + prep_model_calls
    total_tokens = int(result_summary["total_tokens"]) + prep_total_tokens
    summary = {
        "schema_version": "omniflow.androidworld.e2e-task-report.v2",
        "immutable": True,
        "task": args.task,
        "attempt_id": attempt_id,
        "status": status,
        "source_seed": SOURCE_SEED,
        "evaluation_seed": TASK_SEED,
        "deadline_sec": deadline.seconds,
        "outer_wall_sec": deadline.elapsed,
        "counts": counts,
        "model_calls": total_model_calls,
        "total_tokens": total_tokens,
        "result_summary": result_summary,
        "phases": phases,
    }
    _write_json(attempt_root / "pipeline_summary.json", summary)
    return summary


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    deadline = Deadline(args.task_deadline_sec)
    attempt_id = args.attempt_id or (
        "e2e_"
        + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"_{os.getpid()}"
    )
    attempt_root = args.output_root / _safe_component(args.task) / attempt_id
    outcomes_root = args.results_root / "androidworld_validator" / "result_outcomes"
    if args.dry_run:
        registry = load_local_data(args.memory_index)
        plan = registered_result_plan_from_memory(
            memory_index=args.memory_index,
            task_name=args.task,
            methods=METHODS,
            devices=tuple(device[0] for device in DEVICES),
            source_seed=SOURCE_SEED,
            evaluation_seed=TASK_SEED,
            formal_max_steps=int(args.max_steps),
        )
        return {
            "schema_version": "omniflow.androidworld.e2e-task-plan.v1",
            "task": args.task,
            "deadline_sec": args.task_deadline_sec,
            "max_steps": int(args.max_steps),
            "max_fallback_steps": int(args.max_fallback_steps),
            "source_seed": SOURCE_SEED,
            "evaluation_seed": TASK_SEED,
            "methods": list(METHODS),
            "devices": [list(device) for device in DEVICES],
            "schedule": {
                device[0]: list(METHODS) for device in DEVICES
            },
            "completed": [list(result) for result in plan["completed"]],
            "pending": [list(result) for result in plan["pending"]],
            "canonical_source_available": args.task
            in registry.get("canonical", {}).get("source_run_logs", {}),
            "canonical_function_available": args.task
            in registry.get("canonical", {}).get("function_stores", {}),
            "writes": False,
        }
    attempt_root.mkdir(parents=True, exist_ok=False)
    phases: dict[str, Any] = {}
    _write_json(
        attempt_root / "pipeline_manifest.json",
        {
            "schema_version": "omniflow.androidworld.e2e-task-attempt.v1",
            "immutable": True,
            "task": args.task,
            "attempt_id": attempt_id,
            "deadline_sec": args.task_deadline_sec,
            "max_steps": int(args.max_steps),
            "max_fallback_steps": int(args.max_fallback_steps),
            "methods": list(METHODS),
            "devices": [list(device) for device in DEVICES],
        },
    )
    try:
        phases["source_device"] = ensure_source_device(
            args=args,
            attempt_root=attempt_root,
            deadline=deadline,
        )
    except Exception as error:
        error_path = _write_json(
            attempt_root / "preflight" / "source_failure.json",
            {"error": f"{type(error).__name__}: {error}"},
        )
        phases["source_device"] = {
            "status": "failed",
            "model_calls": 0,
            "total_tokens": 0,
            "error": str(error),
        }
        _blocked_all(
            args=args,
            attempt_id=attempt_id,
            attempt_root=attempt_root,
            outcomes_root=outcomes_root,
            status="prep_failed",
            stage="source_runtime_preflight",
            evidence=error_path,
        )
        return _report(
            args=args,
            attempt_id=attempt_id,
            attempt_root=attempt_root,
            outcomes_root=outcomes_root,
            deadline=deadline,
            phases=phases,
        )
    try:
        source_path: Path
        run_log: dict[str, Any]
        if getattr(args, "source_only", False):
            _, source_path, run_log = _canonical_source(
                args.memory_index,
                args.task,
                require_protocol_seed=False,
            )
            source_path, run_log, source_phase = collect_replayed_source(
                args=args,
                deadline=deadline,
                attempt_root=attempt_root,
                source_path=source_path,
                source_run_log=run_log,
            )
            phases["source"] = source_phase
        else:
            _, source_path, run_log = _canonical_source(
                args.memory_index,
                args.task,
            )
            phases["source"] = {
                "status": "reused",
                "model_calls": 0,
                "total_tokens": 0,
                "source_run_log": str(source_path),
            }
    except Exception as error:
        failure_phase = getattr(error, "phase", {})
        error_path = _write_json(
            attempt_root / "source" / "failure.json",
            {"error": f"{type(error).__name__}: {error}"},
        )
        phases["source"] = {
            **failure_phase,
            "status": "failed",
            "model_calls": int(failure_phase.get("model_calls") or 0),
            "total_tokens": int(failure_phase.get("total_tokens") or 0),
            "error": str(error),
        }
        _blocked_all(
            args=args,
            attempt_id=attempt_id,
            attempt_root=attempt_root,
            outcomes_root=outcomes_root,
            status="prep_failed",
            stage="source_run_log",
            evidence=error_path,
        )
        return _report(
            args=args,
            attempt_id=attempt_id,
            attempt_root=attempt_root,
            outcomes_root=outcomes_root,
            deadline=deadline,
            phases=phases,
        )

    if getattr(args, "source_only", False):
        return _report(
            args=args,
            attempt_id=attempt_id,
            attempt_root=attempt_root,
            outcomes_root=outcomes_root,
            deadline=deadline,
            phases=phases,
        )

    blocked_methods: dict[str, tuple[str, str, str]] = {}
    function_store: dict[str, Any] | None = None
    try:
        function_store, function_phase = prepare_function_asset(
            args=args,
            source_path=source_path,
            run_log=run_log,
            attempt_root=attempt_root,
            deadline=deadline,
        )
        phases["function"] = function_phase
    except Exception as error:
        failure_phase = getattr(error, "phase", {})
        failure = _write_json(
            attempt_root / "assets" / "omniflow_failure.json",
            {"error": f"{type(error).__name__}: {error}"},
        )
        phases["function"] = {
            **failure_phase,
            "status": "failed",
            "model_calls": int(failure_phase.get("model_calls") or 0),
            "total_tokens": int(failure_phase.get("total_tokens") or 0),
            "error": str(error),
        }
        blocked_methods["omniflow"] = (
            "prep_failed",
            "function_asset",
            str(failure),
        )
        _blocked_all(
            args=args,
            attempt_id=attempt_id,
            attempt_root=attempt_root,
            outcomes_root=outcomes_root,
            status="prep_failed",
            stage="function_asset",
            evidence=failure,
        )
        return _report(
            args=args,
            attempt_id=attempt_id,
            attempt_root=attempt_root,
            outcomes_root=outcomes_root,
            deadline=deadline,
            phases=phases,
        )

    source_calls = phases["function"].get("source_calls")
    if not isinstance(source_calls, list) or len(source_calls) != 1:
        failure = _write_json(
            attempt_root / "source_qualification" / "failure.json",
            {"error": "canonical_function_single_source_call_required"},
        )
        phases["source_qualification"] = {
            "status": "failed",
            "model_calls": 0,
            "total_tokens": 0,
            "error": "canonical_function_single_source_call_required",
        }
        blocked_methods["omniflow"] = (
            "prep_failed",
            "source_qualification",
            str(failure),
        )
        blocked_methods["t3a_hint"] = (
            "prep_failed",
            "source_qualification",
            str(failure),
        )
    else:
        try:
            qualification = qualify_source_function(
                args=args,
                source_path=source_path,
                run_log=run_log,
                function_store=function_store,
                source_call=source_calls[0],
                attempt_root=attempt_root,
                deadline=deadline,
                round_index=1,
            )
            phases["source_qualification"] = qualification
            if not qualification["qualified"]:
                failure = Path(str(qualification["log_path"])).resolve()
                blocked_methods["omniflow"] = (
                    "prep_failed",
                    "source_qualification",
                    str(failure),
                )
                blocked_methods["t3a_hint"] = (
                    "prep_failed",
                    "source_qualification",
                    str(failure),
                )
        except Exception as error:
            failure = _write_json(
                attempt_root / "source_qualification" / "failure.json",
                {"error": f"{type(error).__name__}: {error}"},
            )
            phases["source_qualification"] = {
                "status": "failed",
                "model_calls": 0,
                "total_tokens": 0,
                "error": str(error),
            }
            blocked_methods["omniflow"] = (
                "prep_failed",
                "source_qualification",
                str(failure),
            )
            blocked_methods["t3a_hint"] = (
                "prep_failed",
                "source_qualification",
                str(failure),
            )

    if getattr(args, "source_qualification_only", False):
        return _report(
            args=args,
            attempt_id=attempt_id,
            attempt_root=attempt_root,
            outcomes_root=outcomes_root,
            deadline=deadline,
            phases=phases,
        )

    mobilegpt_memory: Path | None = None
    try:
        mobilegpt_memory, phases["mobilegpt_memory"] = prepare_mobilegpt_memory(
            args=args,
            attempt_root=attempt_root,
            deadline=deadline,
        )
    except Exception as error:
        failure_phase = getattr(error, "phase", {})
        failure = _write_json(
            attempt_root / "assets" / "mobilegpt_failure.json",
            {"error": f"{type(error).__name__}: {error}"},
        )
        phases["mobilegpt_memory"] = {
            **failure_phase,
            "status": "failed",
            "model_calls": int(failure_phase.get("model_calls") or 0),
            "total_tokens": int(failure_phase.get("total_tokens") or 0),
            "error": str(error),
        }
        blocked_methods["mobilegpt"] = (
            "prep_failed",
            "source_memory",
            str(failure),
        )

    appagent_memory: Path | None = None
    try:
        appagent_memory, phases["appagent_memory"] = prepare_appagent_memory(
            args=args,
            attempt_root=attempt_root,
            deadline=deadline,
        )
    except Exception as error:
        failure_phase = getattr(error, "phase", {})
        failure = _write_json(
            attempt_root / "assets" / "appagent_failure.json",
            {"error": f"{type(error).__name__}: {error}"},
        )
        phases["appagent_memory"] = {
            **failure_phase,
            "status": "failed",
            "model_calls": int(failure_phase.get("model_calls") or 0),
            "total_tokens": int(failure_phase.get("total_tokens") or 0),
            "error": str(error),
        }
        blocked_methods["appagent"] = (
            "prep_failed",
            "source_memory",
            str(failure),
        )

    if function_store is None:
        failure = _write_json(
            attempt_root / "assets" / "function_store_missing.json",
            {"error": "canonical_function_store_unavailable"},
        )
        _blocked_all(
            args=args,
            attempt_id=attempt_id,
            attempt_root=attempt_root,
            outcomes_root=outcomes_root,
            status="prep_failed",
            stage="function_asset",
            evidence=failure,
        )
        return _report(
            args=args,
            attempt_id=attempt_id,
            attempt_root=attempt_root,
            outcomes_root=outcomes_root,
            deadline=deadline,
            phases=phases,
        )
    store_path = Path(str(function_store["store_path"])).resolve()
    try:
        workers = run_target_workers(
            args=args,
            deadline=deadline,
            attempt_id=attempt_id,
            attempt_root=attempt_root,
            outcomes_root=outcomes_root,
            store_path=store_path,
            mobilegpt_memory=mobilegpt_memory,
            appagent_memory=appagent_memory,
            blocked_methods=blocked_methods,
        )
        phases["targets"] = {
            "status": "finished",
            "model_calls": 0,
            "total_tokens": 0,
            "workers": workers,
        }
    except Exception as error:
        failure = _write_json(
            attempt_root / "targets" / "failure.json",
            {"error": f"{type(error).__name__}: {error}"},
        )
        phases["targets"] = {
            "status": "failed",
            "model_calls": 0,
            "total_tokens": 0,
            "error": str(error),
        }
        _blocked_all(
            args=args,
            attempt_id=attempt_id,
            attempt_root=attempt_root,
            outcomes_root=outcomes_root,
            status="deadline_exceeded" if deadline.expired else "execution_failed",
            stage="target_scheduler",
            evidence=failure,
        )
    return _report(
        args=args,
        attempt_id=attempt_id,
        attempt_root=attempt_root,
        outcomes_root=outcomes_root,
        deadline=deadline,
        phases=phases,
    )


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
        expected_sha = (
            str(runlog.get("sha256") or "") if isinstance(runlog, dict) else ""
        )
        path = (manifest_path.parent / relative).resolve()
        if not relative or not path.is_file():
            raise FileNotFoundError(f"bmoca_source_runlog_missing:{task}:{path}")
        if expected_sha and _sha256(path) != expected_sha:
            raise ValueError(f"bmoca_source_runlog_sha256_mismatch:{task}")
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


def _function_enhancement_transport(
    *,
    model: str,
    timeout_sec: float,
    usage: dict[str, int],
) -> Callable[[str, dict[str, Any]], str]:
    """Return only the model transport required by canonical save_function."""

    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("Install omniflow[llm] for Function enhancement") from error
    api_key, base_url = resolve_openai_compatible_config(
        profile="llmthu",
        base_url=FORMAL_MODEL_BASE_URL,
    )
    client = OpenAI(
        api_key=api_key or "not-required",
        base_url=base_url,
        max_retries=0,
        timeout=float(timeout_sec),
    )
    def complete_json(prompt: str, tool: dict[str, Any]) -> str:
        usage["model_calls"] += 1
        tool_name = str(tool["function"]["name"])
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=2048,
            temperature=0,
            tools=[tool],
            tool_choice={
                "type": "function",
                "function": {"name": tool_name},
            },
            parallel_tool_calls=False,
            reasoning_effort="none",
        )
        response_usage = getattr(response, "usage", None)
        usage["prompt_tokens"] += int(
            getattr(response_usage, "prompt_tokens", 0) or 0
        )
        usage["completion_tokens"] += int(
            getattr(response_usage, "completion_tokens", 0) or 0
        )
        usage["total_tokens"] += int(
            getattr(response_usage, "total_tokens", 0) or 0
        )
        choices = getattr(response, "choices", None) or ()
        message = getattr(choices[0], "message", None) if choices else None
        calls = getattr(message, "tool_calls", None) or ()
        if len(calls) != 1:
            raise ValueError("function_enhancer_tool_call_invalid")
        function = getattr(calls[0], "function", None)
        if str(getattr(function, "name", "") or "") != tool_name:
            raise ValueError("function_enhancer_tool_name_invalid")
        arguments = getattr(function, "arguments", None)
        if not isinstance(arguments, str):
            raise ValueError("function_enhancer_arguments_invalid")
        return arguments

    return complete_json


def _save_bmoca_function_once(
    *,
    args: argparse.Namespace,
    task: str,
    source_run_log: Path,
    task_root: Path,
) -> tuple[Path, dict[str, Any]]:
    attempt_id = _safe_component(
        str(
            getattr(args, "attempt_id", "")
            or getattr(getattr(args, "output_root", None), "name", "")
            or task_root.name
        )
    )
    repository_root = Path(
        getattr(args, "repo", task_root.parent)
    ).expanduser().resolve()
    store_path = (
        repository_root
        / "data"
        / "bmoca"
        / _safe_component(task)
        / "env100"
        / "function"
        / "function_authoring"
        / attempt_id
        / "function_store.json"
    )
    if store_path.exists():
        raise FileExistsError(f"bmoca_function_store_already_exists:{store_path}")
    usage = {
        "model_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    started = time.monotonic()
    try:
        report = save_function(
            source_run_log,
            store_path,
            enhance=True,
            complete_json=_function_enhancement_transport(
                model=args.formal_model,
                timeout_sec=float(FUNCTION_ENHANCEMENT_TIMEOUT_SEC),
                usage=usage,
            ),
        )
    except Exception as error:
        _write_json(
            task_root / "enhancement_failure.json",
            {
                "schema_version": "omniflow.bmoca-function-enhancement.v1",
                "status": "failed",
                "task": task,
                "source_run_log": str(source_run_log),
                "source_run_log_sha256": _sha256(source_run_log),
                "save_function_calls": 1,
                "wall_sec": round(time.monotonic() - started, 6),
                "error": f"{type(error).__name__}: {error}",
                **usage,
            },
        )
        raise
    enhancement = {
        "schema_version": "omniflow.bmoca-function-enhancement.v1",
        "task": task,
        "source_run_log": str(source_run_log),
        "source_run_log_sha256": _sha256(source_run_log),
        "save_function_calls": 1,
        "enhanced": report.get("enhanced") is True,
        "function_ids": list(report.get("function_ids") or ()),
        "store_path": str(store_path),
        "store_sha256": _sha256(store_path),
        "transfer_state_catalog": str(report.get("transfer_state_catalog") or ""),
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
        "memory_sha256": _sha256(memory_path),
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
        os.environ.get("MOBILEGPT_EMBEDDING_MODEL") or "text-embedding-v4"
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
        [
            "bash",
            str(args.script),
            "--environment",
            "bmoca",
            "--method",
            method,
            "--tasks",
            task,
        ],
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
        "run_log_path": str(evidence.get("target_run_log_path") or ""),
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
        task_root = campaign_root / "tasks" / _safe_component(task)
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
                    "save_function_calls": 1,
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

        ours_gate_passed = False
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
        ours_gate_passed = _bmoca_source_replay_qualified(gate_row)
        if "ours_replay" in method_assets and not ours_gate_passed:
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
            if method == "ours_replay" and not ours_gate_passed:
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment",
        choices=("androidworld", "bmoca"),
        default="androidworld",
    )
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-deadline-sec", type=int, default=TASK_DEADLINE_SEC)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument(
        "--max-fallback-steps", type=int, default=MAX_FALLBACK_STEPS
    )
    parser.add_argument("--memory-index", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--android-world-root", type=Path)
    parser.add_argument("--omnitransfer-root", type=Path, required=True)
    parser.add_argument("--mobilegpt-root", type=Path)
    parser.add_argument("--appagent-root", type=Path)
    parser.add_argument("--appagent-memory-root", type=Path)
    parser.add_argument("--python-bin", type=Path, required=True)
    parser.add_argument("--adb-path", type=Path)
    parser.add_argument("--emulator-bin", type=Path)
    parser.add_argument(
        "--source-device",
        type=_parse_source_device,
        default=SOURCE_DEVICE,
    )
    parser.add_argument("--source-avd", default=SOURCE_AVD)
    parser.add_argument("--emulator-gpu", default="swiftshader_indirect")
    parser.add_argument("--runtime-preflight", type=Path)
    parser.add_argument("--formal-model", default=FORMAL_MODEL)
    parser.add_argument("--bmoca-root", type=Path)
    parser.add_argument("--bmoca-corpus-manifest", type=Path)
    parser.add_argument("--bmoca-avd-home", type=Path)
    parser.add_argument("--bmoca-android-env-root", type=Path)
    parser.add_argument("--android-sdk-root", type=Path)
    parser.add_argument("--attempt-id", default="")
    parser.add_argument("--source-qualification-only", action="store_true")
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    for field in ("repo", "script", "output_root", "omnitransfer_root"):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    args.python_bin = args.python_bin.expanduser().absolute()
    if args.appagent_memory_root is not None:
        args.appagent_memory_root = args.appagent_memory_root.expanduser().resolve()
    if args.task_deadline_sec > TASK_DEADLINE_SEC:
        raise ValueError("task_deadline_exceeds_1800_seconds")
    if args.task_deadline_sec <= 0:
        raise ValueError("task_deadline_must_be_positive")
    if args.max_steps <= 0:
        raise ValueError("max_steps_must_be_positive")
    if not 0 <= args.max_fallback_steps <= MAX_FALLBACK_STEPS:
        raise ValueError("max_fallback_steps_out_of_range")
    if not args.script.is_file() or not args.python_bin.is_file():
        raise FileNotFoundError("required_script_or_python_missing")
    if args.omnitransfer_root != (Path.home() / "Projects/Omni/OmniTransfer").resolve():
        raise ValueError("canonical_omnitransfer_root_required")
    local_data_root = (args.repo / "data").resolve()
    if args.output_root != local_data_root and local_data_root not in args.output_root.parents:
        raise ValueError("local_data_output_root_required")
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
        if args.formal_model != "GLM-5.1":
            raise ValueError("bmoca_campaign_requires_GLM-5.1")
        if args.max_fallback_steps != 0:
            raise ValueError("bmoca_campaign_fallback_must_be_zero")
        return args
    for field in (
        "memory_index",
        "asset_root",
        "results_root",
        "android_world_root",
        "mobilegpt_root",
        "appagent_root",
        "adb_path",
        "emulator_bin",
        "runtime_preflight",
    ):
        value = getattr(args, field)
        if value is None:
            raise ValueError(f"androidworld_required_path_missing:{field}")
        setattr(args, field, value.expanduser().resolve())
    if not args.task.isalnum():
        raise ValueError("androidworld_task_name_invalid")
    if not args.source_avd.strip():
        raise ValueError("source_avd_required")
    for field in (
        "memory_index",
        "python_bin",
        "adb_path",
        "emulator_bin",
        "runtime_preflight",
    ):
        if not getattr(args, field).is_file():
            raise FileNotFoundError(f"required_file_missing:{field}:{getattr(args, field)}")
    if args.asset_root != local_data_root:
        raise ValueError("local_asset_root_required")
    for field in ("results_root", "output_root"):
        path = getattr(args, field)
        if path != local_data_root and local_data_root not in path.parents:
            raise ValueError(f"local_{field}_required")
    local_data_index = local_data_root / "current.json"
    if args.memory_index == args.repo or (
        args.repo in args.memory_index.parents
        and args.memory_index != local_data_index
    ):
        raise ValueError("external_memory_index_required")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _resolve_args(build_parser().parse_args(argv))
    result = run_bmoca_pipeline(args) if args.environment == "bmoca" else run_pipeline(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.environment == "bmoca":
        return 0 if result.get("status") == "complete" else 1
    if args.dry_run:
        return 0
    if args.source_only:
        return 0 if result.get("status") == "collected" else 1
    if args.source_qualification_only:
        return 0 if result.get("status") == "qualified" else 1
    counts = result.get("counts") if isinstance(result, dict) else None
    return 0 if isinstance(counts, dict) and counts.get("pending") == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
