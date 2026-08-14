from __future__ import annotations

from types import SimpleNamespace

import pytest

from omniflow import RunResult

from src.integrations.android_world.methods import (
    MethodAdapter,
    MethodAdapterContext,
    MethodAdapterRegistry,
    default_method_adapter_registry,
)
from src.integrations.android_world.agent import _accumulate_results


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


def test_androidworld_complexity_budget_cannot_raise_omniflow_step_cap(
    monkeypatch,
) -> None:
    flow = SimpleNamespace(
        config=SimpleNamespace(
            runtime=SimpleNamespace(max_steps=1, max_fallback_steps=None)
        )
    )

    monkeypatch.setattr(
        "src.integrations.android_world.agent.OmniFlow",
        lambda *_args, **_kwargs: flow,
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

    assert built.config.runtime.max_steps == 1


def test_androidworld_step_continues_after_one_nonterminal_action(
    monkeypatch,
) -> None:
    result = RunResult(
        success=False,
        error=None,
        function_id=None,
        actions_executed=1,
        model_calls=1,
        fallback_steps=0,
        detail={
            "done_reason": "step_completed",
            "planner_steps": 1,
            "llm_usage": {"model_calls": 1},
        },
    )
    flow = SimpleNamespace(
        config=SimpleNamespace(
            runtime=SimpleNamespace(max_steps=20, max_fallback_steps=None)
        ),
        run=lambda *_args, **_kwargs: result,
        store=SimpleNamespace(functions={}),
    )

    monkeypatch.setattr(
        "src.integrations.android_world.agent.OmniFlow",
        lambda *_args, **_kwargs: flow,
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
    interaction = agent.step("Turn bluetooth on")

    assert interaction.done is False
    assert interaction.data["done_reason"] == "step_completed"
    assert interaction.data["planner_steps"] == 1


def test_androidworld_accumulates_planner_steps_separately_from_actions() -> None:
    accumulated = _accumulate_results(
        [
            RunResult(
                False,
                actions_executed=3,
                model_calls=1,
                detail={"planner_steps": 1, "llm_usage": {"model_calls": 1}},
            ),
            RunResult(
                False,
                actions_executed=1,
                model_calls=1,
                detail={"planner_steps": 1, "llm_usage": {"model_calls": 1}},
            ),
        ],
        max_steps=20,
    )

    assert accumulated.detail["planner_steps"] == 2
    assert accumulated.actions_executed == 4
    assert accumulated.execution_summary["steps"] == 4
    assert accumulated.execution_summary["planner_steps"] == 2
    assert accumulated.execution_summary["actions_executed"] == 4
    assert accumulated.detail["runtime_limits"]["max_steps"] == 20


def test_androidworld_stops_at_the_planner_step_budget(monkeypatch) -> None:
    result = RunResult(
        False,
        actions_executed=1,
        model_calls=1,
        detail={
            "done_reason": "step_completed",
            "planner_steps": 1,
            "llm_usage": {"model_calls": 1},
        },
    )
    flow = SimpleNamespace(
        config=SimpleNamespace(runtime=SimpleNamespace(max_steps=1)),
        run=lambda *_args, **_kwargs: result,
        store=SimpleNamespace(functions={}),
    )
    monkeypatch.setattr(
        "src.integrations.android_world.agent.OmniFlow",
        lambda *_args, **_kwargs: flow,
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

    assert agent.step("Continue").done is False
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
        config=SimpleNamespace(runtime=SimpleNamespace(max_steps=1)),
        run=lambda *_args, **_kwargs: result,
        store=SimpleNamespace(functions={}),
    )
    monkeypatch.setattr(
        "src.integrations.android_world.agent.OmniFlow",
        lambda *_args, **_kwargs: flow,
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
