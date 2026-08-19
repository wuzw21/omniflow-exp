from __future__ import annotations

from pathlib import Path

from src.experiment.paths import resolve_path, resolve_reference, safe_component


def test_resolve_path_has_one_repository_relative_rule(tmp_path: Path) -> None:
    assert resolve_path("data/current.json", root=tmp_path) == (
        tmp_path / "data/current.json"
    ).resolve()
    absolute = tmp_path / "absolute.json"
    assert resolve_path(absolute, root=Path("/ignored")) == absolute.resolve()


def test_resolve_reference_is_relative_to_its_index(tmp_path: Path) -> None:
    index = tmp_path / "data" / "current.json"
    assert resolve_reference(index, "runs/task/run_log.json") == (
        tmp_path / "data/runs/task/run_log.json"
    ).resolve()


def test_safe_component_cannot_create_nested_artifact_paths() -> None:
    assert safe_component("../Task Name", fallback="task") == "Task_Name"
    assert safe_component("", fallback="task") == "task"

