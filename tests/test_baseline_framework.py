"""Contract tests for the same-harness baseline registry and suite runner."""

from types import SimpleNamespace

import pytest

from src.experiment import run_tasks
from src.experiment.baselines import (
    BASELINE_SUITE,
    BASELINE_SUITE_ALIAS,
    baseline_names,
    baseline_result_slot,
    baseline_spec,
    validate_baseline_request,
)


def test_registry_is_the_single_public_baseline_order():
    assert baseline_names() == BASELINE_SUITE
    assert baseline_names(include_suite_alias=True) == (
        *BASELINE_SUITE,
        BASELINE_SUITE_ALIAS,
    )
    assert len(BASELINE_SUITE) == len(set(BASELINE_SUITE))
    assert {baseline_spec(name).name for name in BASELINE_SUITE} == set(BASELINE_SUITE)


def test_result_slots_keep_omniflow_and_baselines_separate():
    assert baseline_result_slot("omniflow", "none") == "omniflow"
    for baseline in BASELINE_SUITE:
        if baseline != "none":
            assert baseline_result_slot("omniflow", baseline) == baseline
    assert baseline_result_slot("fixed_replay", "none") == "fixed_replay"


def test_suite_validation_requires_all_reusable_inputs():
    with pytest.raises(ValueError, match="baseline_suite_requires"):
        validate_baseline_request(
            BASELINE_SUITE_ALIAS,
            action="run",
            method="omniflow",
            harness="builtin",
            completion_oracle="off",
            memory="store.json",
            source_run_log="",
            element_docs="docs",
        )

    validate_baseline_request(
        BASELINE_SUITE_ALIAS,
        action="run",
        method="omniflow",
        harness="builtin",
        completion_oracle="off",
        memory="store.json",
        source_run_log="run_log.json",
        element_docs="docs",
    )


def test_transfer_factories_are_lazy_and_only_mapper_baselines_override_transfer():
    assert baseline_spec("none").transfer_factory()() is None
    assert callable(baseline_spec("mobilegpt_style").transfer_factory()())
    assert callable(baseline_spec("mobimem_style").transfer_factory()())
    assert baseline_spec("appagent_style").function_memory_enabled is False
    assert baseline_spec("demo_hint").function_memory_enabled is False


def test_suite_runner_uses_fixed_order_and_restores_environment(monkeypatch):
    seen = []
    previous = "caller_value"
    monkeypatch.setenv("OMNIFLOW_BASELINE", previous)

    def fake_run(_args):
        baseline = __import__("os").environ["OMNIFLOW_BASELINE"]
        seen.append(baseline)
        return {"action": "run", "status": "completed", "runs": []}

    monkeypatch.setattr(run_tasks, "_run", fake_run)
    result = run_tasks._run_baseline_suite(SimpleNamespace(task="T"))

    assert seen == list(BASELINE_SUITE)
    assert [row["baseline"] for row in result["baselines"]] == list(BASELINE_SUITE)
    assert result["status"] == "completed"
    assert __import__("os").environ["OMNIFLOW_BASELINE"] == previous


def test_explicit_memory_reuse_skips_authoring_compiler(tmp_path, monkeypatch):
    source = tmp_path / "source_run_log.json"
    source.write_text('{"task_name": "T", "steps": []}\n', encoding="utf-8")
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "store.json").write_text(
        '{"schema_version": "omniflow.store.v2", "functions": {"f": {}}}\n',
        encoding="utf-8",
    )
    (memory / "run_log.json").write_bytes(source.read_bytes())
    (memory / "compile_report.json").write_text(
        '{"model": "Qwen3.6-Plus"}\n',
        encoding="utf-8",
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("authoring compiler must not run for an explicit hit")

    monkeypatch.setattr(run_tasks, "compile_function_v2", fail_if_called)
    monkeypatch.setattr(
        run_tasks,
        "write_function_review",
        lambda **_kwargs: memory / "REVIEW.md",
    )
    args = SimpleNamespace(
        method="omniflow",
        source_run_log=str(source),
        memory=str(memory),
        task="T",
        mobilegpt_root="",
        appagent_root="",
        autodroid_utg="",
        autodroid_official_memory="",
        autodroid_official_app="",
        autodroid_source_memory="",
    )

    result = run_tasks._convert_memory(args)

    assert result["method"] == "omniflow"
    assert result["memory"].endswith("/store.json")
