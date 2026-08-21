from __future__ import annotations

import argparse
from pathlib import Path

from src.experiment.checks import run_integration_checks


REPO = Path(__file__).resolve().parents[1]


def _args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    values = {
        "repo": str(REPO),
        "integration_method": "all",
        "integration_model": "GLM-4.6V",
        "embedding_model": "GLM-Embedding-2",
        "model_endpoint_profile": "llmthu",
        "model_base_url": "https://example.invalid/v1",
        "mobilegpt_root": str(tmp_path / "missing-mobilegpt"),
        "mobilegpt_memory_root": "",
        "appagent_root": str(tmp_path / "missing-appagent"),
        "appagent_memory_root": "",
        "serial": "emulator-5554",
        "server_port": 12345,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_integration_checker_reports_missing_official_roots_without_running_them(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("LLMTHU_API_KEY", "test-key")
    report = run_integration_checks(_args(tmp_path))

    assert report["ready"] is False
    assert report["contract"] == {
        "model_calls": 0,
        "official_source_modified": False,
        "staging_only": True,
    }
    assert any(
        item["method"] == "mobilegpt"
        and item["name"] == "official_root"
        and item["status"] == "fail"
        for item in report["checks"]
    )
    assert any(
        item["method"] == "appagent"
        and item["name"] == "official_root"
        and item["status"] == "fail"
        for item in report["checks"]
    )


def test_integration_checker_rejects_non_glm_embedding_configuration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("LLMTHU_API_KEY", "test-key")
    report = run_integration_checks(
        _args(tmp_path, embedding_model="text-embedding-3-small")
    )

    embedding_checks = [
        item
        for item in report["checks"]
        if item["method"] == "mobilegpt" and item["name"] == "embedding_model"
    ]
    assert embedding_checks == [
        {
            "method": "mobilegpt",
            "name": "embedding_model",
            "status": "fail",
            "detail": "text-embedding-3-small",
            "remediation": "Set MOBILEGPT_EMBEDDING_MODEL=GLM-Embedding-2.",
        }
    ]
