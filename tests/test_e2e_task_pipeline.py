from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from runlog_fixtures import androidworld_run_log

from omniflow.functions.assets import function_authoring_tool
from src.experiment.batch_outcomes import record_result_outcome
from src.experiment.e2e_task_pipeline import (
    Deadline,
    PipelinePhaseError,
    _canonical_bmoca_enhancement_transport,
    _fixed_replay_source_step_width,
    _function_replay_success,
    _max_live_bmoca_cells,
    _parse_source_device,
    _report,
    _resolve_args,
    _run_bmoca_method_cells,
    _save_bmoca_function_once,
    _source_device_ready,
    build_parser,
    collect_replayed_source,
    ensure_source_device,
    qualify_source_function,
    qualify_source_functions,
    run_logged_command,
    run_pipeline,
    run_target_workers,
)
from src.experiment.protocol import (
    DEVICES,
    EPISODE_TIMEOUT_SEC,
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


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        task="BrowserDraw",
        task_deadline_sec=TASK_DEADLINE_SEC,
        max_steps=MAX_STEPS,
        max_fallback_steps=MAX_FALLBACK_STEPS,
        attempt_id="attempt-test",
        output_root=tmp_path / "output",
        results_root=tmp_path / "results",
        memory_index=tmp_path / "memory" / "current.json",
        repo=tmp_path / "repo",
        script=tmp_path / "repo" / "scripts" / "exp" / "run_androidworld.sh",
        asset_root=tmp_path / "assets",
        android_world_root=tmp_path / "android_world",
        omnitransfer_root=tmp_path / "OmniTransfer",
        mobilegpt_root=tmp_path / "MobileGPT",
        appagent_root=tmp_path / "AppAgent",
        python_bin=tmp_path / "python",
        adb_path=tmp_path / "adb",
        source_model="glm-5.1",
        source_device=SOURCE_DEVICE,
        source_qualification_only=False,
        source_only=False,
        dry_run=False,
    )


def test_dry_run_has_fixed_task_method_device_schedule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    args.dry_run = True
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.load_artifact_memory",
        lambda _: {"canonical": {"source_run_logs": {}, "function_stores": {}}},
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.registered_result_plan_from_memory",
        lambda **_: {
            "completed": [],
            "pending": [
                (method, device[0]) for method in METHODS for device in DEVICES
            ],
        },
    )

    plan = run_pipeline(args)

    assert len(plan["pending"]) == 10
    assert plan["source_seed"] == SOURCE_SEED == 111
    assert plan["evaluation_seed"] == TASK_SEED == 113
    assert plan["methods"] == list(METHODS)
    assert plan["devices"] == [list(device) for device in DEVICES]
    assert plan["schedule"] == {
        device[0]: list(METHODS) for device in DEVICES
    }
    assert MAX_FALLBACK_STEPS == 5
    assert SOURCE_MAX_STEPS == 30
    assert MAX_STEPS == 20
    assert SOURCE_DEVICE == ("source5560", "emulator-5560", 5560)
    assert plan["writes"] is False


def test_source_device_uses_independent_small_phone_avd() -> None:
    parser = build_parser()

    assert parser.get_default("source_avd") == "SmallPhone"
    assert parser.get_default("source_device") == SOURCE_DEVICE


def test_source_device_accepts_an_isolated_console_port() -> None:
    assert _parse_source_device("source5570:emulator-5570:5570") == (
        "source5570",
        "emulator-5570",
        5570,
    )


def test_bmoca_method_launches_ten_isolated_overlapping_subprocess_cells(
    tmp_path: Path,
) -> None:
    lock = threading.Lock()
    active = 0
    maximum = 0
    environments: list[dict[str, str]] = []
    commands: list[list[str]] = []

    def runner(command: list[str], **kwargs: object) -> dict[str, object]:
        nonlocal active, maximum
        environment = dict(kwargs["environment"])
        output = Path(environment["OMNIFLOW_BMOCA_OUTPUT_PATH"])
        with lock:
            active += 1
            maximum = max(maximum, active)
            environments.append(environment)
            commands.append(command)
        time.sleep(0.05)
        output.mkdir(parents=True)
        environment_id = environment["OMNIFLOW_BMOCA_SINGLE_ENVIRONMENT_ID"]
        (output / "summary.json").write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "environment_id": environment_id,
                            "emulator_serial": (
                                "emulator-"
                                + environment["OMNIFLOW_BMOCA_EMULATOR_CONSOLE_PORT"]
                            ),
                            "official_success": True,
                            "method_success": True,
                            "actions_executed": 1,
                            "model_calls": 0,
                            "fallback_steps": 0,
                            "run_log_evidence": {
                                "target_run_log_path": str(output / "target.run_log.json")
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        with lock:
            active -= 1
        return {
            "returncode": 0,
            "wall_sec": 0.05,
            "process_pid": 10000 + int(environment_id),
            "started_at": "2026-08-18T00:00:00+00:00",
            "finished_at": "2026-08-18T00:00:01+00:00",
        }

    args = SimpleNamespace(
        repo=tmp_path / "repo",
        script=tmp_path / "repo/scripts/exp/run_androidworld.sh",
        python_bin=tmp_path / "python",
        omnitransfer_root=tmp_path / "OmniTransfer",
        bmoca_root=tmp_path / "BMoCA",
        bmoca_android_env_root=tmp_path / "AndroidEnv",
        android_sdk_root=tmp_path / "sdk",
        bmoca_cell_timeout_sec=600,
    )
    rows = _run_bmoca_method_cells(
        args=args,
        task="clock/create_alarm_at_06:30_am",
        method="script_replay",
        store_path=tmp_path / "store.json",
        task_root=tmp_path / "task",
        avd_homes={
            str(value): tmp_path / f"avd/env_{value}" for value in range(100, 110)
        },
        command_runner=runner,
    )

    assert len(rows) == 10
    assert maximum == 10
    assert _max_live_bmoca_cells(rows) == 10
    assert len({row["process_pid"] for row in rows}) == 10
    assert len({env["OMNIFLOW_BMOCA_AVD_HOME"] for env in environments}) == 10
    assert len({env["OMNIFLOW_BMOCA_APPIUM_PORT"] for env in environments}) == 10
    assert len({env["OMNIFLOW_BMOCA_EMULATOR_CONSOLE_PORT"] for env in environments}) == 10
    assert all(command[:2] == ["bash", str(args.script)] for command in commands)
    assert all("OPENAI_API_KEY" not in env for env in environments)
    assert all("OMNIFLOW_ENV_FILE" not in env for env in environments)


def test_bmoca_offline_enhancement_calls_only_canonical_save_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.run_log.json"
    source.write_text("{}", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def writer(run_log: Path, store_path: Path, **kwargs: object) -> dict[str, object]:
        calls.append({"run_log": run_log, "store_path": store_path, **kwargs})
        store_path.parent.mkdir(parents=True)
        store_path.write_text("{}", encoding="utf-8")
        transfer = store_path.with_name("transfer_states.json")
        transfer.write_text("{}", encoding="utf-8")
        return {
            "enhanced": True,
            "function_ids": ["complete"],
            "transfer_state_catalog": str(transfer),
        }

    monkeypatch.setattr("src.experiment.e2e_task_pipeline.save_function", writer)
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline._canonical_bmoca_enhancement_transport",
        lambda **_: (lambda _prompt, _tool: "{}"),
    )
    args = SimpleNamespace(formal_model="GLM-5.1", enhancement_timeout_sec=180)

    _, report = _save_bmoca_function_once(
        args=args,
        task="clock/create_alarm_at_06:30_am",
        source_run_log=source,
        task_root=tmp_path / "task",
    )

    assert len(calls) == 1
    assert calls[0]["enhance"] is True
    assert callable(calls[0]["complete_json"])
    assert report["save_function_calls"] == 1


def test_bmoca_enhancement_failure_preserves_stage_and_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.run_log.json"
    source.write_text("{}", encoding="utf-8")

    def transport(**kwargs: object):
        usage = kwargs["usage"]

        def fail(_prompt: str, _tool: dict[str, object]) -> str:
            usage["model_calls"] += 1
            raise TimeoutError("endpoint did not answer")

        return fail

    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline._canonical_bmoca_enhancement_transport",
        transport,
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.save_function",
        lambda *_args, **kwargs: kwargs["complete_json"](
            "split prompt",
            function_authoring_tool(stage="split", current_bundle=None),
        ),
    )
    args = SimpleNamespace(formal_model="GLM-5.1", enhancement_timeout_sec=180)
    task_root = tmp_path / "task"

    with pytest.raises(TimeoutError, match="endpoint did not answer"):
        _save_bmoca_function_once(
            args=args,
            task="clock/create_alarm_at_06:30_am",
            source_run_log=source,
            task_root=task_root,
        )

    failure = json.loads(
        (task_root / "enhancement_failure.json").read_text(encoding="utf-8")
    )
    assert failure["status"] == "failed"
    assert failure["save_function_calls"] == 1
    assert failure["model_calls"] == 1
    assert failure["error"] == "TimeoutError: endpoint did not answer"


def test_bmoca_enhancement_uses_the_shared_complete_bundle_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Completions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                ),
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            tool_calls=[
                                SimpleNamespace(
                                    function=SimpleNamespace(
                                        name="submit_function_bundle",
                                        arguments='{"functions":[],"arguments":{}}',
                                    )
                                )
                            ]
                        )
                    )
                ],
            )

    class OpenAI:
        def __init__(self, **_: object) -> None:
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=OpenAI))
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.resolve_openai_compatible_config",
        lambda **_: ("key", "https://example.invalid/v1"),
    )
    usage = {
        "model_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    complete = _canonical_bmoca_enhancement_transport(
        model="GLM-5.1",
        timeout_sec=180,
        usage=usage,
    )
    tool = function_authoring_tool(stage="split", current_bundle=None)

    assert complete("Return the bundle", tool) == '{"functions":[],"arguments":{}}'
    assert captured["tools"] == [tool]
    assert captured["tool_choice"] == {
        "type": "function",
        "function": {"name": "submit_function_bundle"},
    }
    assert usage == {
        "model_calls": 1,
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
    }


def test_resolve_args_preserves_symlinked_virtualenv_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    args.emulator_bin = tmp_path / "emulator"
    args.runtime_preflight = tmp_path / "repo" / "src" / "experiment" / "preflight.py"
    for path in (
        args.script,
        args.memory_index,
        args.adb_path,
        args.emulator_bin,
        args.runtime_preflight,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    real_python = tmp_path / "runtime" / "python"
    real_python.parent.mkdir(parents=True)
    real_python.touch()
    args.python_bin.parent.mkdir(parents=True, exist_ok=True)
    args.python_bin.symlink_to(real_python)
    for path in (
        args.asset_root,
        args.results_root,
        args.output_root,
        args.android_world_root,
        args.mobilegpt_root,
        args.appagent_root,
    ):
        path.mkdir(parents=True, exist_ok=True)
    canonical_transfer = tmp_path / "Projects" / "Omni" / "OmniTransfer"
    canonical_transfer.mkdir(parents=True)
    args.omnitransfer_root = canonical_transfer
    args.appagent_memory_root = None
    args.source_avd = "SmallPhone"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    resolved = _resolve_args(args)

    assert resolved.python_bin == args.python_bin.absolute()
    assert resolved.python_bin.is_symlink()


def test_result_environment_uses_orchestrator_budget_and_child_guard(
    tmp_path: Path,
) -> None:
    from src.experiment.e2e_task_pipeline import _result_environment

    args = _args(tmp_path)
    args.max_steps = 7
    args.max_fallback_steps = 2
    environment = _result_environment(
        args=args,
        attempt_id="attempt-test",
        attempt_root=tmp_path / "attempt",
        method="t3a_hint",
        device=DEVICES[0],
        store_path=tmp_path / "store.json",
        mobilegpt_memory=None,
        appagent_memory=None,
    )

    result_attempt_id = "attempt-test.t3a_hint.small5554"
    assert environment["OMNIFLOW_BATCH_CHILD"] == "1"
    assert environment["OMNIFLOW_BATCH_ATTEMPT_ID"] == result_attempt_id
    assert environment["OMNIFLOW_ANDROIDWORLD_MAX_STEPS"] == "7"
    assert environment["OMNIFLOW_ANDROIDWORLD_MAX_FALLBACK_STEPS"] == "2"


def test_formal_timeout_covers_frozen_steps_and_validator_flush() -> None:
    assert EPISODE_TIMEOUT_SEC == MAX_STEPS * STEP_TIMEOUT_SEC + 300
    assert EPISODE_TIMEOUT_SEC > 600


def test_source_device_ready_requires_exact_avd_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = SimpleNamespace(source_avd="SmallPhone", source_device=SOURCE_DEVICE)

    def adb_output(_args: object, *command: str) -> str:
        if command[-3:] == ("emu", "avd", "name"):
            return "AndroidWorldAvd\nOK"
        if command[-1] == "get-state":
            return "device"
        if command[-2:] == ("getprop", "sys.boot_completed"):
            return "1"
        return ""

    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline._adb_output",
        adb_output,
    )

    assert _source_device_ready(args) is False


def test_source_device_is_cold_restarted_when_already_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    args.source_avd = "SmallPhone"
    args.emulator_bin = tmp_path / "emulator"
    args.emulator_gpu = "swiftshader_indirect"
    args.runtime_preflight = tmp_path / "preflight.py"
    args.task = "ContactsAddContact"
    adb_calls: list[tuple[str, ...]] = []
    preflight_commands: list[list[str]] = []
    preflight_environments: list[dict[str, str]] = []

    def adb_output(_args: object, *command: str) -> str:
        adb_calls.append(command)
        if command == ("devices",):
            return "List of devices attached\nemulator-5560\tdevice"
        return ""

    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline._adb_output",
        adb_output,
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline._source_device_ready",
        lambda _args: True,
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline._read_object",
        lambda _path: {"source_index": "source-index.json"},
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.run_logged_command",
        lambda command, **kwargs: (
            preflight_commands.append(command)
            or preflight_environments.append(kwargs["environment"])
            or {"returncode": 0}
        ),
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.subprocess.Popen",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.time.sleep",
        lambda _seconds: None,
    )

    result = ensure_source_device(
        args=args,
        attempt_root=tmp_path / "attempt",
        deadline=Deadline(120),
    )

    assert ("-s", "emulator-5560", "emu", "kill") in adb_calls
    assert result["launched"] is True
    assert "--require-contacts-ready" not in preflight_commands[0]
    assert preflight_commands[0][preflight_commands[0].index("--android-world-root") + 1] == str(
        args.android_world_root
    )
    assert str(args.android_world_root) in preflight_environments[0]["PYTHONPATH"]


def test_target_workers_parallelize_devices_and_serialize_methods(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    calls: list[tuple[str, str, float, float]] = []
    completed: set[tuple[str, str]] = set()
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline._concluded_results",
        lambda *_: set(completed),
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.concluded_result_keys",
        lambda **_: set(),
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.record_result_outcome",
        lambda **_: tmp_path / "outcome.json",
    )

    def runner(command: list[str], **kwargs: object) -> dict[str, object]:
        environment = kwargs["environment"]
        assert isinstance(environment, dict)
        method = str(environment["OMNIFLOW_ANDROIDWORLD_METHOD"])
        device = str(environment["OMNIFLOW_ANDROIDWORLD_DEVICE"]).split(":")[0]
        started = time.monotonic()
        time.sleep(0.03)
        finished = time.monotonic()
        calls.append((device, method, started, finished))
        completed.add((method, device))
        return {"returncode": 0, "timed_out": False, "wall_sec": 0.03}

    run_target_workers(
        args=args,
        deadline=Deadline(10),
        attempt_id="attempt-test",
        attempt_root=tmp_path / "attempt",
        outcomes_root=tmp_path / "outcomes",
        store_path=tmp_path / "store.json",
        mobilegpt_memory=tmp_path / "mobilegpt",
        appagent_memory=tmp_path / "appagent",
        blocked_methods={},
        command_runner=runner,
    )

    assert len(calls) == 10
    for device in ("small5554", "fold5564"):
        rows = sorted((row for row in calls if row[0] == device), key=lambda row: row[2])
        assert [row[1] for row in rows] == list(METHODS)
        assert all(current[3] <= following[2] for current, following in zip(rows, rows[1:]))
    small_first = next(row for row in calls if row[:2] == ("small5554", METHODS[0]))
    fold_first = next(row for row in calls if row[:2] == ("fold5564", METHODS[0]))
    assert max(small_first[2], fold_first[2]) < min(small_first[3], fold_first[3])


def test_target_workers_fail_stop_after_pending_environment_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    calls: list[tuple[str, str]] = []
    recorded: list[dict[str, object]] = []
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline._concluded_results",
        lambda *_: set(),
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.concluded_result_keys",
        lambda **_: set(),
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.record_result_outcome",
        lambda **kwargs: recorded.append(kwargs) or tmp_path / "outcome.json",
    )

    def runner(command: list[str], **kwargs: object) -> dict[str, object]:
        environment = kwargs["environment"]
        assert isinstance(environment, dict)
        calls.append(
            (
                str(environment["OMNIFLOW_ANDROIDWORLD_DEVICE"]).split(":")[0],
                str(environment["OMNIFLOW_ANDROIDWORLD_METHOD"]),
            )
        )
        return {"returncode": 1, "timed_out": False, "wall_sec": 0.01}

    with pytest.raises(PipelinePhaseError, match="target_episode_environment_failure"):
        run_target_workers(
            args=args,
            deadline=Deadline(10),
            attempt_id="attempt-test",
            attempt_root=tmp_path / "attempt",
            outcomes_root=tmp_path / "outcomes",
            store_path=tmp_path / "store.json",
            mobilegpt_memory=tmp_path / "mobilegpt",
            appagent_memory=tmp_path / "appagent",
            blocked_methods={},
            command_runner=runner,
        )

    assert recorded
    assert all(row["status"] == "environment_failure" for row in recorded)
    for device in ("small5554", "fold5564"):
        assert len([call for call in calls if call[0] == device]) <= 1


def test_blocked_cells_do_not_duplicate_shared_prep_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    recorded: list[dict[str, object]] = []
    completed: set[tuple[str, str]] = set()
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline._concluded_results",
        lambda *_: set(completed),
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.concluded_result_keys",
        lambda **_: set(),
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.record_result_outcome",
        lambda **kwargs: recorded.append(kwargs) or tmp_path / "outcome.json",
    )

    run_target_workers(
        args=args,
        deadline=Deadline(10),
        attempt_id="attempt-test",
        attempt_root=tmp_path / "attempt",
        outcomes_root=tmp_path / "outcomes",
        store_path=tmp_path / "store.json",
        mobilegpt_memory=None,
        appagent_memory=None,
        blocked_methods={
            "ours": ("prep_failed", "function_asset", str(tmp_path / "failure.json"))
        },
        command_runner=lambda *args, **kwargs: (
            completed.add(
                (
                    str(kwargs["environment"]["OMNIFLOW_ANDROIDWORLD_METHOD"]),
                    str(
                        kwargs["environment"]["OMNIFLOW_ANDROIDWORLD_DEVICE"]
                    ).split(":")[0],
                )
            )
            or {"returncode": 0, "timed_out": False, "wall_sec": 0}
        ),
    )

    ours = [row for row in recorded if row["method"] == "ours"]
    assert len(ours) == 2
    assert all(row["artifact_root"] is None for row in ours)


def test_zero_remaining_deadline_does_not_launch_child(tmp_path: Path) -> None:
    log_path = tmp_path / "deadline.log"

    result = run_logged_command(
        ["this-command-must-not-run"],
        cwd=tmp_path,
        environment={},
        log_path=log_path,
        timeout_sec=0,
    )

    assert result["returncode"] == 124
    assert result["timed_out"] is True
    assert "deadline exceeded" in log_path.read_text(encoding="utf-8")


def test_collect_replayed_source_uses_fixed_replay_and_captures_screenshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    source_path = tmp_path / "source.run_log.json"
    source = androidworld_run_log(
        [{"action_type": "click", "x": 50, "y": 50}],
        task_name=args.task,
    )
    source_path.write_text(json.dumps(source), encoding="utf-8")
    screenshot = tmp_path / "screen.png"
    screenshot.write_bytes(b"captured-screen")
    screenshot_hash = hashlib.sha256(screenshot.read_bytes()).hexdigest()
    state = {
        "pixels": {
            "path": str(screenshot.resolve()),
            "sha256": screenshot_hash,
            "width": 100,
            "height": 200,
            "mime_type": "image/png",
        },
        "forest": "<hierarchy />",
        "ui_elements": [],
        "auxiliaries": {
            "state_id": "captured-state",
            "display": {"width": 100, "height": 200},
        },
    }

    def runner(command: list[str], **kwargs: object) -> dict[str, object]:
        assert command[command.index("--agent") + 1] == "fixed_replay"
        assert command[command.index("--raw-replay-run-log") + 1] == str(source_path)
        assert "--model" not in command
        assert "--planner-provider" not in command
        assert "--store-path" not in command
        environment = kwargs["environment"]
        assert isinstance(environment, dict)
        assert environment["OMNIFLOW_RAW_REPLAY_CAPTURE_OBSERVATIONS"] == "1"
        output = Path(command[command.index("--output-path") + 1])
        output.mkdir(parents=True, exist_ok=True)
        (output / "task_results.jsonl").write_text(
            json.dumps(
                {
                    "official_validator_used": True,
                    "androidworld_validator_result": {
                        "success": True,
                        "reward": 1.0,
                        "uses_androidworld_official_validator": True,
                    },
                    "model_calls": 0,
                    "total_tokens": 0,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        raw_replay_result = Path(environment["OMNIFLOW_RAW_REPLAY_RESULT_JSON"])
        raw_replay_result.parent.mkdir(parents=True, exist_ok=True)
        raw_replay_result.write_text(
            json.dumps(
                {
                    "completed": True,
                    "replay_completed": True,
                    "run_id": "fixed-replay-capture",
                    "execution_trace": {
                        "steps": [
                            {
                                "provider_detail": {
                                    "raw_replay": {
                                        "step_results": [
                                            {
                                                "completed": True,
                                                "observation_before_act": {
                                                    "androidworld_state": state
                                                },
                                            }
                                        ],
                                        "final_observation": {
                                            "androidworld_state": state
                                        },
                                    }
                                }
                            }
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )
        return {
            "returncode": 0,
            "timed_out": False,
            "wall_sec": 0.1,
            "log_path": str(kwargs["log_path"]),
        }

    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.run_logged_command",
        runner,
    )
    captured_path, captured, phase = collect_replayed_source(
        args=args,
        deadline=Deadline(10),
        attempt_root=tmp_path / "attempt",
        source_path=source_path,
        source_run_log=source,
    )

    assert captured_path.is_file()
    assert captured["steps"][0]["action"] == source["steps"][0]["action"]
    assert captured["steps"][0]["observation"]["pixels"]["sha256"] == screenshot_hash
    assert phase["model_calls"] == 0
    assert phase["total_tokens"] == 0
    assert phase["status"] == "collected"


def test_fixed_replay_groups_editable_input_text_raw_actions() -> None:
    source = androidworld_run_log(
        [{"action_type": "input_text", "x": 50, "y": 50, "text": "hello"}],
        observations=[
            {
                "pixels": None,
                "forest": (
                    '<hierarchy><node class="android.widget.EditText" '
                    'editable="true" bounds="[0,0][100,100]" /></hierarchy>'
                ),
                "ui_elements": [],
                "auxiliaries": {"display": {"width": 1000, "height": 1000}},
            }
        ],
    )

    assert _fixed_replay_source_step_width(source["steps"][0]) == 2


def test_collect_replayed_source_rejects_model_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    source_path = tmp_path / "source.run_log.json"
    source = androidworld_run_log(
        [{"action_type": "click", "x": 50, "y": 50}],
        task_name=args.task,
    )
    source_path.write_text(json.dumps(source), encoding="utf-8")

    def runner(command: list[str], **kwargs: object) -> dict[str, object]:
        output = Path(command[command.index("--output-path") + 1])
        output.mkdir(parents=True, exist_ok=True)
        (output / "task_results.jsonl").write_text(
            json.dumps(
                {
                    "official_validator_used": True,
                    "androidworld_validator_result": {
                        "success": True,
                        "reward": 1.0,
                        "uses_androidworld_official_validator": True,
                    },
                    "model_calls": 1,
                    "total_tokens": 10,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "returncode": 0,
            "timed_out": False,
            "wall_sec": 0.1,
            "log_path": str(kwargs["log_path"]),
        }

    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.run_logged_command",
        runner,
    )

    with pytest.raises(PipelinePhaseError) as raised:
        collect_replayed_source(
            args=args,
            deadline=Deadline(10),
            attempt_root=tmp_path / "attempt",
            source_path=source_path,
            source_run_log=source,
        )

    assert raised.value.phase["model_calls"] == 1
    assert raised.value.phase["total_tokens"] == 10


def test_pipeline_does_not_collect_missing_canonical_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.ensure_source_device",
        lambda **_: {"status": "ready", "model_calls": 0, "total_tokens": 0},
    )
    collected = False

    def collect(**_kwargs: object) -> object:
        nonlocal collected
        collected = True
        raise AssertionError("formal orchestration must not collect source data")

    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.collect_replayed_source",
        collect,
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline._blocked_all",
        lambda **_: None,
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline._report",
        lambda **kwargs: kwargs["phases"],
    )

    phases = run_pipeline(args)

    assert phases["source"]["status"] == "failed"
    assert phases["source"]["model_calls"] == 0
    assert phases["source"]["total_tokens"] == 0
    assert collected is False


def test_source_only_pipeline_collects_replayed_source_and_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    args.source_only = True
    source_path = tmp_path / "source.run_log.json"
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.ensure_source_device",
        lambda **_: {"status": "ready", "model_calls": 0, "total_tokens": 0},
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline._canonical_source",
        lambda *_: ({}, source_path, {"task_name": args.task}),
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.collect_replayed_source",
        lambda **_: (
            source_path,
            {"task_name": args.task},
            {
                "status": "collected",
                "source_run_log": str(source_path),
                "model_calls": 0,
                "total_tokens": 0,
            },
        ),
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.prepare_function_asset",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("source-only collection must not prepare Functions")
        ),
    )

    report = run_pipeline(args)

    assert report["schema_version"] == (
        "omniflow.androidworld.source-collection-report.v1"
    )
    assert report["status"] == "collected"
    assert report["phases"]["source"]["source_run_log"] == str(source_path)


def test_pipeline_stops_when_canonical_function_store_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    source_path = tmp_path / "source.json"
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.ensure_source_device",
        lambda **_: {"status": "ready", "model_calls": 0, "total_tokens": 0},
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline._canonical_source",
        lambda *_: ({}, source_path, {}),
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.prepare_function_asset",
        lambda **_: (_ for _ in ()).throw(RuntimeError("function failed")),
    )
    mobilegpt_called = False

    def prepare_mobilegpt(**_kwargs: object) -> object:
        nonlocal mobilegpt_called
        mobilegpt_called = True
        raise AssertionError("asset preflight must stop before baseline preparation")

    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.prepare_mobilegpt_memory",
        prepare_mobilegpt,
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.prepare_appagent_memory",
        lambda **_: (_ for _ in ()).throw(RuntimeError("appagent failed")),
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.run_target_workers",
        lambda **_: [],
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline._report",
        lambda **kwargs: kwargs["phases"],
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline._blocked_all",
        lambda **_: None,
    )

    phases = run_pipeline(args)

    assert phases["function"]["status"] == "failed"
    assert mobilegpt_called is False


def test_pipeline_qualifies_ordered_source_calls_before_target_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    source_path = tmp_path / "source.json"
    source_path.write_text("{}", encoding="utf-8")
    store_path = tmp_path / "store.json"
    store_path.write_text("{}", encoding="utf-8")
    source_calls = [
        {"function_id": "create_note", "arguments": {"name": "note"}},
        {"function_id": "save_note", "arguments": {"text": "body"}},
    ]
    events: list[str] = []
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.ensure_source_device",
        lambda **_: {"status": "ready", "model_calls": 0, "total_tokens": 0},
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline._canonical_source",
        lambda *_: ({}, source_path, {"task_parameters": {}}),
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.prepare_function_asset",
        lambda **_: (
            {"store_path": str(store_path)},
            {
                "status": "reused",
                "model_calls": 0,
                "total_tokens": 0,
                "source_calls": source_calls,
            },
        ),
    )

    def qualify(**kwargs: object) -> dict[str, object]:
        events.append("qualify")
        assert kwargs["source_calls"] == source_calls
        return {
            "status": "qualified",
            "qualified": True,
            "model_calls": 0,
            "total_tokens": 0,
        }

    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.qualify_source_functions",
        qualify,
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.prepare_mobilegpt_memory",
        lambda **_: (tmp_path / "mobilegpt", {"status": "reused"}),
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.prepare_appagent_memory",
        lambda **_: (tmp_path / "appagent", {"status": "reused"}),
    )

    def workers(**kwargs: object) -> list[dict[str, object]]:
        events.append("targets")
        assert kwargs["blocked_methods"] == {}
        return []

    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.run_target_workers",
        workers,
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline._report",
        lambda **kwargs: kwargs["phases"],
    )

    phases = run_pipeline(args)

    assert events == ["qualify", "targets"]
    assert phases["source_qualification"]["qualified"] is True


def test_source_qualification_only_stops_before_baselines_and_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    args.source_qualification_only = True
    source_path = tmp_path / "source.json"
    source_path.write_text("{}", encoding="utf-8")
    store_path = tmp_path / "store.json"
    store_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.ensure_source_device",
        lambda **_: {"status": "ready", "model_calls": 0, "total_tokens": 0},
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline._canonical_source",
        lambda *_: ({}, source_path, {"task_parameters": {}}),
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.prepare_function_asset",
        lambda **_: (
            {"store_path": str(store_path)},
            {
                "status": "reused",
                "model_calls": 0,
                "total_tokens": 0,
                "source_calls": [
                    {"function_id": "create_note", "arguments": {}}
                ],
            },
        ),
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.qualify_source_functions",
        lambda **_: {
            "status": "qualified",
            "qualified": True,
            "model_calls": 0,
            "total_tokens": 0,
        },
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.prepare_mobilegpt_memory",
        lambda **_: (_ for _ in ()).throw(AssertionError("baseline prepared")),
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.prepare_appagent_memory",
        lambda **_: (_ for _ in ()).throw(AssertionError("baseline prepared")),
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.run_target_workers",
        lambda **_: (_ for _ in ()).throw(AssertionError("targets started")),
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline._report",
        lambda **kwargs: kwargs["phases"],
    )

    phases = run_pipeline(args)

    assert phases["source_qualification"]["qualified"] is True


@pytest.mark.parametrize(
    ("model_calls", "fallback_steps", "expected"),
    [(0, 0, True), (1, 0, False), (0, 1, False)],
)
def test_source_function_qualification_requires_zero_model_and_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_calls: int,
    fallback_steps: int,
    expected: bool,
) -> None:
    args = _args(tmp_path)
    store = tmp_path / "store.json"
    store.write_text("{}", encoding="utf-8")
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")

    def runner(command: list[str], **kwargs: object) -> dict[str, object]:
        output = Path(command[command.index("--output-path") + 1])
        output.mkdir(parents=True)
        (output / "task_results.jsonl").write_text(
            json.dumps(
                {
                    "official_validator_success": True,
                    "model_calls": model_calls,
                    "fallback_steps": fallback_steps,
                    "canonical_run": {
                        "status": "succeeded",
                        "diagnostics": {
                            "execution_summary": {"success": True, "steps": 1},
                            "execution_trace": [{"result": {"success": True}}],
                        },
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "returncode": 0,
            "timed_out": False,
            "wall_sec": 0.1,
            "log_path": str(kwargs["log_path"]),
        }

    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.run_logged_command",
        runner,
    )
    result = qualify_source_function(
        args=args,
        source_path=source,
        run_log={"task_parameters": {}},
        function_store={
            "store_path": str(store),
            "transfer_states_sha256": "a" * 64,
        },
        source_call={"function_id": "draw", "arguments": {}},
        attempt_root=tmp_path / "attempt",
        deadline=Deadline(10),
        round_index=1,
    )

    assert result["qualified"] is expected


def test_source_function_qualification_does_not_require_whole_task_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    store = tmp_path / "store.json"
    store.write_text("{}", encoding="utf-8")
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")

    def runner(command: list[str], **kwargs: object) -> dict[str, object]:
        output = Path(command[command.index("--output-path") + 1])
        output.mkdir(parents=True)
        (output / "task_results.jsonl").write_text(
            json.dumps(
                {
                    "official_validator_success": False,
                    "model_calls": 0,
                    "fallback_steps": 0,
                    "canonical_run": {
                        "status": "failed",
                        "diagnostics": {
                            "execution_summary": {"success": True, "steps": 2},
                            "execution_trace": [
                                {"result": {"success": True}},
                                {"result": {"success": True}},
                            ],
                        },
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "returncode": 0,
            "timed_out": False,
            "wall_sec": 0.1,
            "log_path": str(kwargs["log_path"]),
        }

    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.run_logged_command",
        runner,
    )
    result = qualify_source_function(
        args=args,
        source_path=source,
        run_log={"task_parameters": {}},
        function_store={"store_path": str(store), "transfer_states_sha256": "a" * 64},
        source_call={"function_id": "draw", "arguments": {}},
        attempt_root=tmp_path / "attempt",
        deadline=Deadline(10),
        round_index=1,
    )

    assert result["official_validator_success"] is False
    assert result["function_replay_success"] is True
    assert result["qualified"] is True


def test_source_function_sequence_qualification_uses_one_ordered_episode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    store = tmp_path / "store.json"
    store.write_text("{}", encoding="utf-8")
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    captured: list[list[str]] = []

    def runner(command: list[str], **kwargs: object) -> dict[str, object]:
        captured.append(command)
        output = Path(command[command.index("--output-path") + 1])
        output.mkdir(parents=True)
        (output / "task_results.jsonl").write_text(
            json.dumps(
                {
                    "official_validator_success": True,
                    "model_calls": 0,
                    "fallback_steps": 0,
                    "canonical_run": {
                        "status": "succeeded",
                        "diagnostics": {
                            "execution_summary": {"success": True, "steps": 6},
                            "execution_trace": [
                                {"result": {"success": True}}
                                for _ in range(6)
                            ],
                        },
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "returncode": 0,
            "timed_out": False,
            "wall_sec": 0.1,
            "log_path": str(kwargs["log_path"]),
        }

    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.run_logged_command",
        runner,
    )
    source_calls = [
        {"function_id": "create_note", "arguments": {"name": "note"}},
        {"function_id": "save_note", "arguments": {"text": "body"}},
    ]

    result = qualify_source_functions(
        args=args,
        source_path=source,
        run_log={"task_parameters": {}},
        function_store={
            "store_path": str(store),
            "transfer_states_sha256": "a" * 64,
        },
        source_calls=source_calls,
        attempt_root=tmp_path / "attempt",
        deadline=Deadline(10),
    )

    assert result["qualified"] is True
    assert result["qualification_scope"] == "ordered_function_sequence_replay"
    assert result["source_calls"] == source_calls
    assert len(captured) == 1
    command = captured[0]
    calls_index = command.index("--function-calls-json") + 1
    assert json.loads(command[calls_index]) == source_calls


def test_ordered_source_qualification_requires_official_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    store = tmp_path / "store.json"
    store.write_text("{}\n", encoding="utf-8")
    args = _args(tmp_path)

    def runner(command, **kwargs):
        output_root = Path(command[command.index("--output-path") + 1])
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "task_results.jsonl").write_text(
            json.dumps(
                {
                    "official_validator_success": False,
                    "model_calls": 0,
                    "fallback_steps": 0,
                    "canonical_run": {
                        "status": "failed",
                        "diagnostics": {
                            "execution_summary": {"success": True, "steps": 1},
                            "execution_trace": [{"result": {"success": True}}],
                        },
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "returncode": 0,
            "timed_out": False,
            "wall_sec": 0.1,
            "log_path": str(kwargs["log_path"]),
        }

    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.run_logged_command",
        runner,
    )
    result = qualify_source_functions(
        args=args,
        source_path=source,
        run_log={"task_parameters": {}},
        function_store={
            "store_path": str(store),
            "transfer_states_sha256": "a" * 64,
        },
        source_calls=[{"function_id": "create_note", "arguments": {}}],
        attempt_root=tmp_path / "attempt",
        deadline=Deadline(10),
    )

    assert result["function_replay_success"] is True
    assert result["official_validator_success"] is False
    assert result["qualified"] is False


def test_function_replay_success_is_independent_of_validator() -> None:
    row = {
        "official_validator_success": False,
        "canonical_run": {
            "diagnostics": {
                "execution_summary": {"success": True, "steps": 1},
                "execution_trace": [{"result": {"success": True}}],
            }
        },
    }
    assert _function_replay_success(row) is True


def test_function_replay_success_rejects_whole_task_status_without_runtime_evidence() -> None:
    row = {
        "official_validator_success": True,
        "canonical_run": {"status": "succeeded"},
    }
    assert _function_replay_success(row) is False


def test_pipeline_report_always_materializes_four_report_formats(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.memory_index.parent.mkdir(parents=True)
    source_index = tmp_path / "source_index.json"
    source_index.write_text(json.dumps({args.task: {}}), encoding="utf-8")
    result_cells = tmp_path / "result_cells.json"
    result_cells.write_text("{}", encoding="utf-8")
    args.memory_index.write_text(
        json.dumps(
            {
                "source_index": str(source_index),
                "result_cells": str(result_cells),
            }
        ),
        encoding="utf-8",
    )
    attempt_root = tmp_path / "attempt"
    attempt_root.mkdir()
    outcomes_root = tmp_path / "outcomes"
    for method in METHODS:
        for label, serial, _ in DEVICES:
            record_result_outcome(
                outcomes_root=outcomes_root,
                task_name=args.task,
                method=method,
                device=label,
                device_serial=serial,
                attempt_id="attempt-test",
                source_seed=SOURCE_SEED,
                evaluation_seed=TASK_SEED,
                status="prep_failed",
                stage="test",
            )

    summary = _report(
        args=args,
        attempt_id="attempt-test",
        attempt_root=attempt_root,
        outcomes_root=outcomes_root,
        deadline=Deadline(10),
        phases={"source": {"status": "failed", "model_calls": 1, "total_tokens": 7}},
    )

    assert summary["counts"]["planned"] == 10
    assert summary["counts"]["pending"] == 0
    assert summary["model_calls"] == 1
    assert summary["total_tokens"] == 7
    for field in ("results_jsonl", "results_csv", "results_markdown", "pipeline_markdown"):
        assert Path(summary[field]).is_file()
    assert (attempt_root / "pipeline_summary.json").is_file()
    assert len(Path(summary["results_jsonl"]).read_text(encoding="utf-8").splitlines()) == 10
    assert "tool_calls" not in summary
    assert "tokens" not in summary
