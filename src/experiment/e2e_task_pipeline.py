"""Bounded source-to-result AndroidWorld task pipeline."""

from __future__ import annotations

import argparse
import concurrent.futures
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
from src.experiment.androidworld import ArchivedRunLog, build_fixed_replay_command
from src.experiment.artifact_memory import (
    canonical_mobilegpt_memory_from_memory,
    load_artifact_memory,
    registered_result_plan_from_memory,
)
from src.experiment.batch_outcomes import (
    concluded_result_keys,
    record_result_outcome,
    write_batch_report,
)
from src.integrations.runlog import project_androidworld_step_actions
from src.experiment.protocol import (
    DEVICES,
    FORMAL_MODEL,
    MAX_FALLBACK_STEPS,
    MAX_STEPS,
    METHODS,
    SOURCE_DEVICE,
    SOURCE_MAX_STEPS,
    SOURCE_SEED,
    STEP_TIMEOUT_SEC,
    TASK_DEADLINE_SEC,
    TASK_SEED,
)

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
    subprocess.Popen(
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
        time.sleep(1)
    else:
        raise RuntimeError(f"source_emulator_not_ready:{source_serial}")
    pointer = _read_object(args.memory_index)
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
        "--expected-tasks",
        "116",
        "--source-index",
        str(_resolve_reference(args.memory_index, pointer["source_index"])),
        "--source-task",
        args.task,
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
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    registry = load_artifact_memory(memory_index)
    record = registry.get("canonical", {}).get("source_run_logs", {}).get(task)
    if not isinstance(record, dict):
        raise ValueError(f"canonical_source_missing:{task}")
    path = Path(str(record.get("object_path") or "")).expanduser().resolve()
    if not path.is_file() or _sha256(path) != str(record.get("sha256") or ""):
        raise ValueError(f"canonical_source_object_invalid:{task}:{path}")
    run_log = require_complete_source_run_log(_read_object(path))
    if run_log["task_name"] != task:
        raise ValueError(f"canonical_source_task_mismatch:{task}")
    if run_log["seed"] != SOURCE_SEED:
        raise ValueError(f"canonical_source_seed_mismatch:{task}:{run_log['seed']}")
    if run_log["success"] is not True:
        raise ValueError(f"canonical_source_not_successful:{task}")
    return registry, path, run_log


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
            "seed": source_run_log["seed"],
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
    item = ArchivedRunLog(
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
    registry = load_artifact_memory(memory_index)
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
    provenance_path = Path(str(existing["provenance_path"])).resolve()
    provenance = _read_object(provenance_path)
    source_calls = provenance.get("source_calls")
    if (
        not isinstance(source_calls, list)
        or not source_calls
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
        "src.experiment.direct_function_launch",
        "--repo",
        str(args.repo),
        "--function-id",
        str(source_call["function_id"]),
        "--function-arguments-json",
        json.dumps(source_call["arguments"], ensure_ascii=False),
        "--",
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
        "LLMTHU_KEY",
        "LLMTHU_BASE_URL",
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
        and result["model_calls"] == 0
        and result["fallback_steps"] == 0
    )
    _write_json(output_root.parent / "qualification.json", result)
    return result


def qualify_source_functions(
    *,
    args: argparse.Namespace,
    source_path: Path,
    run_log: dict[str, Any],
    function_store: dict[str, Any],
    source_calls: list[dict[str, Any]],
    attempt_root: Path,
    deadline: Deadline,
) -> dict[str, Any]:
    output_root = attempt_root / "source_qualification" / "ordered_sequence"
    store_path = Path(str(function_store["store_path"])).resolve()
    command = [
        str(args.python_bin),
        "-m",
        "src.experiment.direct_function_launch",
        "--repo",
        str(args.repo),
        "--function-calls-json",
        json.dumps(source_calls, ensure_ascii=False),
        "--",
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
        str(max(SOURCE_MAX_STEPS, len(source_calls))),
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
        "LLMTHU_KEY",
        "LLMTHU_BASE_URL",
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
            "qualification_scope": "ordered_function_sequence_replay",
            "official_validator_success": _official_success(row),
            "function_replay_success": _function_replay_success(row),
            "model_calls": int(row.get("model_calls") or 0),
            "fallback_steps": int(row.get("fallback_steps") or 0),
            "task_run_status": str(canonical.get("status") or ""),
            "source_run_log": str(source_path),
            "source_run_log_sha256": _sha256(source_path),
            "store_path": str(store_path),
            "store_sha256": _sha256(store_path),
            "transfer_states_sha256": str(
                function_store.get("transfer_states_sha256") or ""
            ),
            "source_calls": source_calls,
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
            "OMNIFLOW_MOBILEGPT_MEMORY_OUTPUT_ROOT": str(output_root),
        }
    )
    result = run_logged_command(
        [
            "bash",
            str(args.script),
            "--prepare-mobilegpt-memory",
            "--tasks",
            args.task,
        ],
        cwd=args.repo,
        environment=environment,
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
    memory_pointer = _read_object(args.memory_index)
    command = [
        str(args.python_bin),
        "-m",
        "src.experiment.appagent_source",
        "prepare",
        "--index",
        str(_resolve_reference(args.memory_index, memory_pointer["source_index"])),
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
    manifest = _read_object(root / "appagent_demo_manifest.json")
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
            "OMNIFLOW_ANDROIDWORLD_SOURCE_SEED": str(SOURCE_SEED),
            "OMNIFLOW_ANDROIDWORLD_TASK_SEED": str(TASK_SEED),
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
        environment["OMNIFLOW_APPAGENT_DEMO_MEMORY_ROOT"] = str(appagent_memory)
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


def _markdown_cell(value: Any) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace(
        "\n", " "
    )


def _write_pipeline_markdown(
    path: Path,
    *,
    task: str,
    attempt_id: str,
    status: str,
    wall_sec: float,
    model_calls: int,
    total_tokens: int,
    phases: dict[str, Any],
    results_markdown: str,
) -> Path:
    lines = [
        f"# AndroidWorld E2E Task: {task}",
        "",
        f"- Attempt: `{attempt_id}`",
        f"- Status: `{status}`",
        f"- Outer wall seconds: {wall_sec}",
        f"- Model calls: {model_calls}",
        f"- Total tokens: {total_tokens}",
        f"- Result table: `{results_markdown}`",
        "",
        "| phase | status | model_calls | total_tokens | wall_sec | error | evidence |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for name, raw_phase in phases.items():
        phase = raw_phase if isinstance(raw_phase, dict) else {}
        evidence = (
            phase.get("preflight")
            or phase.get("source_run_log")
            or phase.get("store")
            or phase.get("memory_root")
            or phase.get("log_path")
            or ""
        )
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(value)
                for value in (
                    name,
                    phase.get("status", ""),
                    phase.get("model_calls", 0),
                    phase.get("total_tokens", 0),
                    phase.get("wall_sec", 0),
                    phase.get("error", ""),
                    evidence,
                )
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


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
    pointer = _read_object(args.memory_index)
    target_report = write_batch_report(
        report_root=attempt_root / "report",
        memory_index=args.memory_index,
        outcomes_root=outcomes_root,
        source_index=_resolve_reference(args.memory_index, pointer["source_index"]),
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
    counts = target_report["counts"]
    status = (
        "complete"
        if counts["pending"] == 0
        else "deadline_exceeded"
        if deadline.expired
        else "partial"
    )
    total_model_calls = int(target_report["model_calls"]) + prep_model_calls
    total_tokens = int(target_report["total_tokens"]) + prep_total_tokens
    pipeline_markdown = attempt_root / "pipeline.md"
    _write_pipeline_markdown(
        pipeline_markdown,
        task=args.task,
        attempt_id=attempt_id,
        status=status,
        wall_sec=deadline.elapsed,
        model_calls=total_model_calls,
        total_tokens=total_tokens,
        phases=phases,
        results_markdown=target_report["results_markdown"],
    )
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
        "target_report": target_report["summary"],
        "results_jsonl": target_report["results_jsonl"],
        "results_csv": target_report["results_csv"],
        "results_markdown": target_report["results_markdown"],
        "pipeline_markdown": str(pipeline_markdown),
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
        registry = load_artifact_memory(args.memory_index)
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
            attempt_root / "assets" / "ours_failure.json",
            {"error": f"{type(error).__name__}: {error}"},
        )
        phases["function"] = {
            **failure_phase,
            "status": "failed",
            "model_calls": int(failure_phase.get("model_calls") or 0),
            "total_tokens": int(failure_phase.get("total_tokens") or 0),
            "error": str(error),
        }
        blocked_methods["ours"] = (
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
    if not isinstance(source_calls, list) or not source_calls:
        failure = _write_json(
            attempt_root / "source_qualification" / "failure.json",
            {"error": "canonical_function_source_calls_missing"},
        )
        phases["source_qualification"] = {
            "status": "failed",
            "model_calls": 0,
            "total_tokens": 0,
            "error": "canonical_function_source_calls_missing",
        }
        blocked_methods["ours"] = (
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
            qualification = qualify_source_functions(
                args=args,
                source_path=source_path,
                run_log=run_log,
                function_store=function_store,
                source_calls=source_calls,
                attempt_root=attempt_root,
                deadline=deadline,
            )
            phases["source_qualification"] = qualification
            if not qualification["qualified"]:
                failure = Path(str(qualification["log_path"])).resolve()
                blocked_methods["ours"] = (
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
            blocked_methods["ours"] = (
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
        blocked_methods["mobilegpt_offline_retrieval"] = (
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
        blocked_methods["appagent_demo"] = (
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
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-deadline-sec", type=int, default=TASK_DEADLINE_SEC)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument(
        "--max-fallback-steps", type=int, default=MAX_FALLBACK_STEPS
    )
    parser.add_argument("--memory-index", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--android-world-root", type=Path, required=True)
    parser.add_argument("--omnitransfer-root", type=Path, required=True)
    parser.add_argument("--mobilegpt-root", type=Path, required=True)
    parser.add_argument("--appagent-root", type=Path, required=True)
    parser.add_argument("--appagent-memory-root", type=Path)
    parser.add_argument("--python-bin", type=Path, required=True)
    parser.add_argument("--adb-path", type=Path, required=True)
    parser.add_argument("--emulator-bin", type=Path, required=True)
    parser.add_argument(
        "--source-device",
        type=_parse_source_device,
        default=SOURCE_DEVICE,
    )
    parser.add_argument("--source-avd", default="SmallPhone")
    parser.add_argument("--emulator-gpu", default="swiftshader_indirect")
    parser.add_argument("--runtime-preflight", type=Path, required=True)
    parser.add_argument("--formal-model", default=FORMAL_MODEL)
    parser.add_argument("--attempt-id", default="")
    parser.add_argument("--source-qualification-only", action="store_true")
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    for field in (
        "repo",
        "script",
        "memory_index",
        "asset_root",
        "results_root",
        "output_root",
        "android_world_root",
        "omnitransfer_root",
        "mobilegpt_root",
        "appagent_root",
        "adb_path",
        "emulator_bin",
        "runtime_preflight",
    ):
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
    if not args.task.isalnum():
        raise ValueError("androidworld_task_name_invalid")
    if not args.source_avd.strip():
        raise ValueError("source_avd_required")
    for field in (
        "script",
        "memory_index",
        "python_bin",
        "adb_path",
        "emulator_bin",
        "runtime_preflight",
    ):
        if not getattr(args, field).is_file():
            raise FileNotFoundError(f"required_file_missing:{field}:{getattr(args, field)}")
    if args.omnitransfer_root != (Path.home() / "Projects/Omni/OmniTransfer").resolve():
        raise ValueError("canonical_omnitransfer_root_required")
    for field in ("asset_root", "results_root", "output_root"):
        path = getattr(args, field)
        if path == args.repo or args.repo in path.parents:
            raise ValueError(f"external_{field}_required")
    if args.memory_index == args.repo or args.repo in args.memory_index.parents:
        raise ValueError("external_memory_index_required")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _resolve_args(build_parser().parse_args(argv))
    result = run_pipeline(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    counts = result.get("counts") if isinstance(result, dict) else None
    return 0 if isinstance(counts, dict) and counts.get("pending") == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
