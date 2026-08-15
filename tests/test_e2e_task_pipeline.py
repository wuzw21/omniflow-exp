from __future__ import annotations

import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from src.experiment.batch_outcomes import record_cell_outcome
from src.experiment.e2e_task_pipeline import (
    DEFAULT_DEADLINE_SEC,
    DEVICES,
    EVALUATION_SEED,
    FORMAL_MAX_STEPS,
    MAX_FALLBACK_STEPS,
    METHODS,
    SOURCE_DEVICE,
    SOURCE_MAX_STEPS,
    SOURCE_SEED,
    Deadline,
    PipelinePhaseError,
    _function_replay_success,
    _report,
    _resolve_args,
    _source_device_ready,
    _source_selection_manifest,
    build_parser,
    collect_online_source,
    qualify_source_function,
    qualify_source_functions,
    run_logged_command,
    run_pipeline,
    run_target_workers,
)


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        task="BrowserDraw",
        source_backend="auto",
        task_deadline_sec=DEFAULT_DEADLINE_SEC,
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
        dry_run=False,
    )


def test_dry_run_has_fixed_ten_cell_schedule(
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
        "src.experiment.e2e_task_pipeline.registered_cell_plan_from_memory",
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
    assert plan["evaluation_seed"] == EVALUATION_SEED == 113
    assert plan["methods"] == list(METHODS)
    assert plan["devices"] == [list(device) for device in DEVICES]
    assert plan["schedule"] == {
        device[0]: list(METHODS) for device in DEVICES
    }
    assert MAX_FALLBACK_STEPS == 5
    assert SOURCE_MAX_STEPS == 30
    assert FORMAL_MAX_STEPS == 20
    assert SOURCE_DEVICE == ("source5560", "emulator-5560", 5560)
    assert plan["writes"] is False


def test_source_device_uses_independent_small_phone_avd() -> None:
    parser = build_parser()

    assert parser.get_default("source_avd") == "SmallPhone"


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
    args.source_run_log = None
    args.appagent_memory_root = None
    args.source_avd = "SmallPhone"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    resolved = _resolve_args(args)

    assert resolved.python_bin == args.python_bin.absolute()
    assert resolved.python_bin.is_symlink()


def test_cell_environment_uses_orchestrator_budget_and_child_guard(
    tmp_path: Path,
) -> None:
    from src.experiment.e2e_task_pipeline import PHASE_TIMEOUTS_SEC, _cell_environment

    environment = _cell_environment(
        args=_args(tmp_path),
        attempt_id="attempt-test",
        attempt_root=tmp_path / "attempt",
        method="t3a_hint",
        device=DEVICES[0],
        store_path=tmp_path / "store.json",
        mobilegpt_memory=None,
        appagent_memory=None,
    )

    cell_attempt_id = "attempt-test.t3a_hint.small5554"
    assert environment["OMNIFLOW_BATCH_CHILD"] == "1"
    assert environment["OMNIFLOW_BATCH_ATTEMPT_ID"] == cell_attempt_id
    assert environment["OMNIFLOW_SINGLE_TASK_OUTPUT_ROOT"] == str(
        tmp_path
        / "attempt"
        / "target_attempts"
        / "small5554"
        / "t3a_hint"
        / cell_attempt_id
    )
    assert environment["OMNIFLOW_SINGLE_TASK_TIMEOUT_SEC"] == str(
        PHASE_TIMEOUTS_SEC["target_episode"]
    )
    assert PHASE_TIMEOUTS_SEC["target_cell"] > PHASE_TIMEOUTS_SEC["target_episode"]


def test_source_device_ready_requires_exact_avd_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = SimpleNamespace(source_avd="SmallPhone")

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


def test_target_workers_parallelize_devices_and_serialize_methods(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    calls: list[tuple[str, str, float, float]] = []
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline._completed_cells",
        lambda _: set(),
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.concluded_cell_keys",
        lambda **_: set(),
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.record_cell_outcome",
        lambda **_: tmp_path / "outcome.json",
    )

    def runner(command: list[str], **kwargs: object) -> dict[str, object]:
        environment = kwargs["environment"]
        assert isinstance(environment, dict)
        method = str(environment["OMNIFLOW_SINGLE_TASK_METHODS"])
        device = str(environment["OMNIFLOW_SINGLE_TASK_DEVICE_TARGETS"]).split(":")[0]
        started = time.monotonic()
        time.sleep(0.03)
        finished = time.monotonic()
        calls.append((device, method, started, finished))
        return {"returncode": 1, "timed_out": False, "wall_sec": 0.03}

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


def test_blocked_cells_do_not_duplicate_shared_prep_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    recorded: list[dict[str, object]] = []
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline._completed_cells",
        lambda _: set(),
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.concluded_cell_keys",
        lambda **_: set(),
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.record_cell_outcome",
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
        command_runner=lambda *args, **kwargs: {
            "returncode": 1,
            "timed_out": False,
            "wall_sec": 0,
        },
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


def test_failed_online_source_preserves_model_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)

    def runner(command: list[str], **kwargs: object) -> dict[str, object]:
        output = Path(command[command.index("--output-path") + 1])
        output.mkdir(parents=True)
        (output / "task_results.jsonl").write_text(
            json.dumps(
                {
                    "official_validator_success": False,
                    "model_calls": 2,
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "returncode": 1,
            "timed_out": False,
            "wall_sec": 0.1,
            "log_path": str(kwargs["log_path"]),
        }

    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.run_logged_command",
        runner,
    )

    with pytest.raises(PipelinePhaseError) as raised:
        collect_online_source(
            args=args,
            deadline=Deadline(10),
            attempt_root=tmp_path / "attempt",
            round_index=1,
        )

    assert raised.value.phase["tool_calls"] == 2
    assert raised.value.phase["tokens"] == 120


def test_failed_online_source_marks_missing_usage_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)

    def runner(command: list[str], **kwargs: object) -> dict[str, object]:
        output = Path(command[command.index("--output-path") + 1])
        output.mkdir(parents=True)
        return {
            "returncode": 124,
            "timed_out": True,
            "wall_sec": 0.1,
            "log_path": str(kwargs["log_path"]),
        }

    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.run_logged_command",
        runner,
    )

    with pytest.raises(PipelinePhaseError) as raised:
        collect_online_source(
            args=args,
            deadline=Deadline(10),
            attempt_root=tmp_path / "attempt",
            round_index=1,
        )

    assert raised.value.phase["usage_accounting_status"] == "unavailable"


def test_pipeline_does_not_collect_missing_canonical_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    args.source_backend = "online"
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.ensure_source_device",
        lambda **_: {"status": "ready", "tool_calls": 0, "tokens": 0},
    )
    collected = False

    def collect(**_kwargs: object) -> object:
        nonlocal collected
        collected = True
        raise AssertionError("formal orchestration must not collect source data")

    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.collect_online_source",
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
    assert phases["source"]["tool_calls"] == 0
    assert phases["source"]["tokens"] == 0
    assert collected is False


def test_pipeline_stops_when_canonical_function_store_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    source_path = tmp_path / "source.json"
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.ensure_source_device",
        lambda **_: {"status": "ready", "tool_calls": 0, "tokens": 0},
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
        lambda **_: {"status": "ready", "tool_calls": 0, "tokens": 0},
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
                "tool_calls": 0,
                "tokens": 0,
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
            "tool_calls": 0,
            "tokens": 0,
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


def test_source_selection_uses_original_index_without_canonical_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text('{"baseline": true}\n', encoding="utf-8")
    selected = tmp_path / "selected.json"
    selected.write_text('{"selected": true}\n', encoding="utf-8")
    source_index = tmp_path / "source_index.json"
    source_index.write_text(
        json.dumps(
            {"BrowserDraw": {"retained_source_run_log": str(baseline)}}
        ),
        encoding="utf-8",
    )
    memory_index = tmp_path / "memory" / "current.json"
    memory_index.parent.mkdir()
    memory_index.write_text(
        json.dumps({"source_index": str(tmp_path / "canonical_source_index.json")}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.experiment.e2e_task_pipeline.load_artifact_memory",
        lambda _: {
            "inputs": {"source_index": str(source_index)},
            "canonical": {"source_run_logs": {}},
        },
    )

    manifest_path = _source_selection_manifest(
        memory_index=memory_index,
        task="BrowserDraw",
        selected_run_log=selected,
        output_path=tmp_path / "selection.json",
        reason="Official successful replacement.",
    )

    assert manifest_path is not None
    selection = json.loads(manifest_path.read_text(encoding="utf-8"))["selections"]
    assert selection["BrowserDraw"]["selected_source_run_log_sha256"] != selection[
        "BrowserDraw"
    ]["expected_source_run_log_sha256"]


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
            record_cell_outcome(
                outcomes_root=outcomes_root,
                task_name=args.task,
                method=method,
                device=label,
                device_serial=serial,
                attempt_id="attempt-test",
                source_seed=SOURCE_SEED,
                evaluation_seed=EVALUATION_SEED,
                status="prep_failed",
                stage="test",
            )

    summary = _report(
        args=args,
        attempt_id="attempt-test",
        attempt_root=attempt_root,
        outcomes_root=outcomes_root,
        deadline=Deadline(10),
        phases={"source": {"status": "failed", "tool_calls": 1, "tokens": 7}},
    )

    assert summary["counts"]["planned"] == 10
    assert summary["counts"]["pending"] == 0
    assert summary["tool_calls"] == 1
    assert summary["tokens"] == 7
    for field in ("cells_jsonl", "cells_csv", "cells_markdown", "pipeline_markdown"):
        assert Path(summary[field]).is_file()
    assert (attempt_root / "pipeline_summary.json").is_file()
    assert len(Path(summary["cells_jsonl"]).read_text(encoding="utf-8").splitlines()) == 10
    for detailed in ("model_calls", "prompt_tokens", "completion_tokens", "total_tokens"):
        assert detailed not in summary
