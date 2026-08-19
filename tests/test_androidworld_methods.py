from __future__ import annotations

from types import SimpleNamespace

import pytest

from omniflow import OmniFlowConfig, RunResult, RuntimeSettings
from src.integrations.android_world.methods import (
    MethodAdapter,
    MethodAdapterContext,
    MethodAdapterRegistry,
    default_method_adapter_registry,
    reuse_metrics,
)


def _context(selector: str) -> MethodAdapterContext:
    return MethodAdapterContext(
        selector=selector,
        env=SimpleNamespace(),
        store_path="store.json",
        adb_serial="emulator-5554",
    )


def test_method_registry_resolves_exactly_one_adapter() -> None:
    registry = MethodAdapterRegistry(
        (
            MethodAdapter("first", lambda selector: selector == "first", lambda _: 1),
            MethodAdapter("second", lambda selector: selector == "second", lambda _: 2),
        )
    )

    assert registry.build(_context("second")) == 2


def test_method_registry_rejects_overlapping_adapters() -> None:
    registry = MethodAdapterRegistry(
        (
            MethodAdapter("first", lambda _: True, lambda _: 1),
            MethodAdapter("second", lambda _: True, lambda _: 2),
        )
    )

    with pytest.raises(
        RuntimeError,
        match="androidworld_method_adapter_ambiguous:omniflow:first,second",
    ):
        registry.build(_context("omniflow"))


def test_default_registry_preserves_unknown_selector_error() -> None:
    with pytest.raises(ValueError, match="Unsupported AndroidWorld agent selector"):
        default_method_adapter_registry().build(_context("unknown"))


def test_default_registry_uses_canonical_omniflow_owner_name() -> None:
    registry = default_method_adapter_registry()

    assert registry._adapters[0].name == "omniflow"


def test_reuse_metrics_counts_function_actions_from_trace() -> None:
    metrics = reuse_metrics(
        "omniflow",
        actions_executed=3,
        canonical_run={
            "diagnostics": {
                "execution_trace": [
                    {"metadata": {"origin": "action"}},
                    {"metadata": {"function_id": "create_note"}},
                    {"metadata": {"function_id": "create_note"}},
                ]
            }
        },
    )

    assert metrics["reuse_numerator"] == 2
    assert metrics["reuse_denominator"] == 3
    assert metrics["reuse_rate"] == 0.666667
    assert metrics["reuse_unit"] == "gui_action"


def test_reuse_metrics_preserves_zero_mobilegpt_hits() -> None:
    metrics = reuse_metrics(
        "mobilegpt",
        mobilegpt_stats={"memory_lookup_count": 4, "memory_hit_count": 0},
    )

    assert metrics["reuse_numerator"] == 0
    assert metrics["reuse_denominator"] == 4
    assert metrics["reuse_rate"] == 0.0
    assert metrics["artifact_used"] is True
    assert metrics["evidence_status"] == "exact_native_memory_events"


def test_appagent_asset_use_is_distinct_from_document_utilization() -> None:
    metrics = reuse_metrics(
        "appagent",
        appagent_result={
            "decision_round_count": 16,
            "documentation_round_count": 0,
            "startup_action_count": 1,
        },
    )

    assert metrics["artifact_used"] is False
    assert metrics["reuse_rate"] == 0.0
    assert metrics["reuse_unit"] == "decision_round"


def test_appagent_is_not_an_androidworld_agent_adapter() -> None:
    with pytest.raises(ValueError, match="Unsupported AndroidWorld agent selector"):
        default_method_adapter_registry().build(
            MethodAdapterContext(
                selector="appagent",
                env=SimpleNamespace(),
                store_path="store.json",
                adb_serial="emulator-5554",
            )
        )


def test_omniflow_adapter_preserves_launcher_step_cap() -> None:
    captured: dict[str, object] = {}

    def build_agent(**options: object) -> SimpleNamespace:
        captured.update(options)
        return SimpleNamespace()

    context = MethodAdapterContext(
        selector="omniflow",
        env=SimpleNamespace(),
        store_path="store.json",
        adb_serial="emulator-5554",
        max_steps=20,
        build_omniflow_agent=build_agent,
    )

    default_method_adapter_registry().build(context)

    assert captured["max_steps"] == 20


def test_direct_function_is_a_method_adapter_not_a_runner_override() -> None:
    captured: dict[str, object] = {}

    def build_agent(**options: object) -> object:
        captured.update(options)
        return object()

    context = MethodAdapterContext(
        selector="omniflow",
        env=SimpleNamespace(),
        store_path="store.json",
        adb_serial="emulator-5554",
        direct_function_id="complete_task",
        direct_function_arguments={"target": "Alarm"},
        planner_model="must-not-build",
        build_omniflow_agent=build_agent,
    )

    default_method_adapter_registry().build(context)

    assert captured["direct_function_id"] == "complete_task"
    assert captured["direct_function_arguments"] == {"target": "Alarm"}


def test_omniflow_adapter_uses_canonical_planner_configuration(
    monkeypatch,
) -> None:
    planner_options: dict[str, object] = {}

    class CapturingPlanner:
        def __init__(self, **options: object) -> None:
            planner_options.update(options)

    monkeypatch.setattr("omniflow.vlm.planner.VLMPlanner", CapturingPlanner)
    monkeypatch.setenv("OPENAI_API_KEY", "not-required")

    context = MethodAdapterContext(
        selector="omniflow",
        env=SimpleNamespace(),
        store_path="store.json",
        adb_serial="emulator-5554",
        planner_model="test-model",
        build_omniflow_agent=lambda **options: SimpleNamespace(**options),
    )

    default_method_adapter_registry().build(context)

    assert planner_options


def test_fixed_replay_never_builds_a_planner(monkeypatch) -> None:
    def fail_if_built(**_options: object) -> None:
        raise AssertionError("fixed_replay_planner_built")

    monkeypatch.setattr("omniflow.vlm.planner.VLMPlanner", fail_if_built)
    context = MethodAdapterContext(
        selector="fixed_replay",
        env=SimpleNamespace(),
        store_path="store.json",
        adb_serial="emulator-5554",
        planner_model="must-not-build",
        raw_replay_run_log="run_log.json",
        build_omniflow_agent=lambda **options: SimpleNamespace(**options),
        apply_fixed_replay=lambda agent, **_options: agent,
    )

    assert default_method_adapter_registry().build(context) is not None


def test_androidworld_complexity_budget_cannot_raise_omniflow_step_cap(
    monkeypatch,
) -> None:
    flow = SimpleNamespace(
        config=OmniFlowConfig(runtime=RuntimeSettings(max_steps=20)),
    )

    monkeypatch.setattr(
        "src.integrations.android_world.agent.OmniFlow",
        lambda *_args, **kwargs: (
            setattr(flow, "config", kwargs["config"]) or flow
        ),
    )
    monkeypatch.setattr(
        "src.integrations.android_world.agent.AndroidWorldHost",
        lambda *_args, **_kwargs: SimpleNamespace(installed_packages=lambda: set()),
    )
    monkeypatch.setattr(
        "src.integrations.android_world.agent.load_transfer_state_catalog",
        lambda _path: {},
    )
    monkeypatch.setattr(
        "src.integrations.android_world.agent.transfer_state_coverage",
        lambda *_args: {
            "required_state_count": 0,
            "complete": True,
            "missing_state_ids": [],
        },
    )
    flow.store = SimpleNamespace(functions={})

    from src.integrations.android_world.agent import build_agent

    built = build_agent(env=SimpleNamespace(), store_path="store.json", max_steps=20)
    built.set_max_steps(60)

    assert built.config.runtime.max_steps == 20


def test_androidworld_complexity_budget_can_lower_omniflow_step_cap(
    monkeypatch,
) -> None:
    flow = SimpleNamespace(
        config=OmniFlowConfig(runtime=RuntimeSettings(max_steps=20)),
    )
    monkeypatch.setattr(
        "src.integrations.android_world.agent.OmniFlow",
        lambda *_args, **kwargs: (
            setattr(flow, "config", kwargs["config"]) or flow
        ),
    )
    monkeypatch.setattr(
        "src.integrations.android_world.agent.AndroidWorldHost",
        lambda *_args, **_kwargs: SimpleNamespace(installed_packages=lambda: set()),
    )
    monkeypatch.setattr(
        "src.integrations.android_world.agent.load_transfer_state_catalog",
        lambda _path: {},
    )
    monkeypatch.setattr(
        "src.integrations.android_world.agent.transfer_state_coverage",
        lambda *_args: {
            "required_state_count": 0,
            "complete": True,
            "missing_state_ids": [],
        },
    )
    flow.store = SimpleNamespace(functions={})

    from src.integrations.android_world.agent import build_agent

    built = build_agent(env=SimpleNamespace(), store_path="store.json", max_steps=20)
    built.set_max_steps(7)

    assert built.config.runtime.max_steps == 7


def test_androidworld_step_runs_one_complete_omniflow_cycle(
    monkeypatch,
) -> None:
    result = RunResult(
        success=True,
        error=None,
        function_id="turn_on_bluetooth",
        actions_executed=4,
        model_calls=3,
        fallback_steps=1,
        detail={
            "done_reason": "finished",
            "planner_steps": 3,
            "llm_usage": {"model_calls": 3},
            "function_resume": {"attempt_count": 1, "success_count": 1},
        },
    )
    run_calls: list[str] = []

    def run(goal: str, **_kwargs: object) -> RunResult:
        run_calls.append(goal)
        return result

    flow = SimpleNamespace(
        config=OmniFlowConfig(runtime=RuntimeSettings(max_steps=20)),
        run=run,
        store=SimpleNamespace(functions={}),
    )

    monkeypatch.setattr(
        "src.integrations.android_world.agent.OmniFlow",
        lambda *_args, **kwargs: (
            setattr(flow, "config", kwargs["config"])
            or setattr(flow, "host", kwargs["host"])
            or flow
        ),
    )
    monkeypatch.setattr(
        "src.integrations.android_world.agent.AndroidWorldHost",
        lambda *_args, **_kwargs: SimpleNamespace(installed_packages=lambda: set()),
    )
    monkeypatch.setattr(
        "src.integrations.android_world.agent.load_transfer_state_catalog",
        lambda _path: {},
    )
    monkeypatch.setattr(
        "src.integrations.android_world.agent.transfer_state_coverage",
        lambda *_args: {
            "required_state_count": 0,
            "complete": True,
            "missing_state_ids": [],
        },
    )

    from src.integrations.android_world.agent import build_agent

    agent = build_agent(env=SimpleNamespace(), store_path="store.json", max_steps=20)
    first = agent.step("Turn bluetooth on")
    second = agent.step("Turn bluetooth on")

    assert first.done is True
    assert first.data["done_reason"] == "finished"
    assert first.data["planner_steps"] == 3
    assert first.data["actions_executed"] == 4
    assert first.data["fallback"] is True
    assert second.done is True
    assert second.data["done_reason"] == "omniflow_cycle_already_completed"
    assert run_calls == ["Turn bluetooth on"]
    assert agent.host.state["last_result"] is result


def test_androidworld_direct_function_runs_through_episode_step_without_planner(
    monkeypatch,
) -> None:
    result = RunResult(
        success=True,
        function_id="complete_task",
        actions_executed=3,
        model_calls=0,
        fallback_steps=0,
        detail={"done_reason": "finished", "planner_steps": 0},
    )
    tool_calls: list[object] = []
    store = SimpleNamespace(
        functions={},
        get_function=lambda function_id: object()
        if function_id == "complete_task"
        else None,
    )
    flow = SimpleNamespace(
        config=OmniFlowConfig(runtime=RuntimeSettings(max_steps=20)),
        store=store,
        call_tool=lambda tool_call, **_kwargs: (tool_calls.append(tool_call) or result),
    )
    monkeypatch.setattr(
        "src.integrations.android_world.agent.OmniFlow",
        lambda *_args, **kwargs: (
            setattr(flow, "config", kwargs["config"])
            or setattr(flow, "host", kwargs["host"])
            or flow
        ),
    )
    monkeypatch.setattr(
        "src.integrations.android_world.agent.AndroidWorldHost",
        lambda *_args, **_kwargs: SimpleNamespace(installed_packages=lambda: set()),
    )
    monkeypatch.setattr(
        "src.integrations.android_world.agent.load_transfer_state_catalog",
        lambda _path: {},
    )
    monkeypatch.setattr(
        "src.integrations.android_world.agent.transfer_state_coverage",
        lambda *_args: {
            "required_state_count": 0,
            "complete": True,
            "missing_state_ids": [],
        },
    )

    from src.integrations.android_world.agent import build_agent

    agent = build_agent(
        env=SimpleNamespace(),
        store_path="store.json",
        max_steps=20,
        direct_function_id="complete_task",
        direct_function_arguments={"target": "Alarm"},
    )

    episode_result = agent.step("planner must not run")

    assert episode_result.data["function_id"] == "complete_task"
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "complete_task"
    assert tool_calls[0].arguments == {"target": "Alarm"}
    assert tool_calls[0].name != "planner must not run"


def test_androidworld_stops_at_the_planner_step_budget(monkeypatch) -> None:
    result = RunResult(
        False,
        actions_executed=1,
        model_calls=1,
        detail={
            "done_reason": "step_completed",
            "planner_steps": 2,
            "llm_usage": {"model_calls": 1},
        },
    )
    flow = SimpleNamespace(
        config=OmniFlowConfig(runtime=RuntimeSettings(max_steps=2)),
        run=lambda *_args, **_kwargs: result,
        store=SimpleNamespace(functions={}),
    )
    monkeypatch.setattr(
        "src.integrations.android_world.agent.OmniFlow",
        lambda *_args, **kwargs: (
            setattr(flow, "config", kwargs["config"]) or flow
        ),
    )
    monkeypatch.setattr(
        "src.integrations.android_world.agent.AndroidWorldHost",
        lambda *_args, **_kwargs: SimpleNamespace(installed_packages=lambda: set()),
    )
    monkeypatch.setattr(
        "src.integrations.android_world.agent.load_transfer_state_catalog",
        lambda _path: {},
    )
    monkeypatch.setattr(
        "src.integrations.android_world.agent.transfer_state_coverage",
        lambda *_args: {
            "required_state_count": 0,
            "complete": True,
            "missing_state_ids": [],
        },
    )

    from src.integrations.android_world.agent import build_agent

    agent = build_agent(env=SimpleNamespace(), store_path="store.json", max_steps=2)

    exhausted = agent.step("Continue")
    assert exhausted.done is True
    assert exhausted.data["planner_steps"] == 2
    assert exhausted.data["done_reason"] == "max_steps_exceeded"


def test_androidworld_stops_after_a_fatal_planner_failure(monkeypatch) -> None:
    error = "vlm_planner_failed:Error code: 429"
    result = RunResult(
        False,
        model_calls=1,
        error=error,
        detail={
            "planner_steps": 1,
            "llm_usage": {"model_calls": 1},
        },
    )
    flow = SimpleNamespace(
        config=OmniFlowConfig(runtime=RuntimeSettings(max_steps=20)),
        run=lambda *_args, **_kwargs: result,
        store=SimpleNamespace(functions={}),
    )
    monkeypatch.setattr(
        "src.integrations.android_world.agent.OmniFlow",
        lambda *_args, **kwargs: (
            setattr(flow, "config", kwargs["config"]) or flow
        ),
    )
    monkeypatch.setattr(
        "src.integrations.android_world.agent.AndroidWorldHost",
        lambda *_args, **_kwargs: SimpleNamespace(installed_packages=lambda: set()),
    )
    monkeypatch.setattr(
        "src.integrations.android_world.agent.load_transfer_state_catalog",
        lambda _path: {},
    )
    monkeypatch.setattr(
        "src.integrations.android_world.agent.transfer_state_coverage",
        lambda *_args: {
            "required_state_count": 0,
            "complete": True,
            "missing_state_ids": [],
        },
    )

    from src.integrations.android_world.agent import build_agent

    agent = build_agent(env=SimpleNamespace(), store_path="store.json", max_steps=20)
    failed = agent.step("Continue")

    assert failed.done is True
    assert failed.data["planner_steps"] == 1
    assert failed.data["done_reason"] == "planner_failed"
    assert failed.data["error"] == error
