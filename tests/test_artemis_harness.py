"""ARTEMIS is an external harness: it must not gain OmniFlow privileges."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from src.integrations import artemis_harness
from src.integrations.gui_agent_harness import HarnessContext


class _Host:
    adb_serial = "emulator-45562"


class _Flow:
    host = _Host()


def _context(tmp_path: Path, timeout: float = 5.0) -> HarnessContext:
    return HarnessContext(
        flow=_Flow(),
        goal="Turn bluetooth on.",
        max_steps=30,
        evidence_root=tmp_path,
        timeout_seconds=timeout,
    )


def test_factory_reports_that_it_needs_no_omniflow_planner(monkeypatch):
    harness = artemis_harness.factory()
    # ARTEMIS brings its own planner; the launcher must not build an OmniFlow one.
    assert harness.requires_builtin_planner is False


def test_missing_root_is_an_episode_failure_not_a_crash(monkeypatch, tmp_path):
    monkeypatch.delenv(artemis_harness.ARTEMIS_ROOT_ENV, raising=False)
    context = _context(tmp_path)
    context.flow.host.adb_serial = "emulator-45562"
    result = asyncio.run(artemis_harness.ArtemisHarness().arun(context))
    assert result.success is False
    assert result.error == "artemis_root_required:set OMNIFLOW_ARTEMIS_ROOT"


def test_invalid_root_is_rejected_by_content_not_by_name(monkeypatch, tmp_path):
    bogus = tmp_path / "not-artemis"
    bogus.mkdir()
    monkeypatch.setenv(artemis_harness.ARTEMIS_ROOT_ENV, str(bogus))
    result = asyncio.run(artemis_harness.ArtemisHarness().arun(_context(tmp_path)))
    assert result.success is False
    assert result.error.startswith("artemis_root_invalid:")


def test_external_checkout_uses_its_own_python_environment(tmp_path):
    root = tmp_path / "artemis-root"
    python = root / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)
    assert artemis_harness._artemis_python(root) == str(python)


def test_missing_device_serial_is_reported_without_launching(monkeypatch, tmp_path):
    root = tmp_path / "artemis-root"
    (root / "artemis").mkdir(parents=True)
    monkeypatch.setenv(artemis_harness.ARTEMIS_ROOT_ENV, str(root))
    monkeypatch.delenv("ANDROID_SERIAL", raising=False)
    context = _context(tmp_path)
    context.flow.host.adb_serial = ""
    result = asyncio.run(artemis_harness.ArtemisHarness().arun(context))
    assert result.success is False
    assert result.error == "artemis_device_serial_required"


def test_launch_enables_trace_recording_in_episode_evidence(monkeypatch, tmp_path):
    root = tmp_path / "artemis-root"
    (root / "artemis").mkdir(parents=True)
    monkeypatch.setenv(artemis_harness.ARTEMIS_ROOT_ENV, str(root))
    captured = {}

    class _Process:
        returncode = 0

        async def wait(self):
            return 0

    async def fake_create_subprocess_exec(*command, **kwargs):
        captured["command"] = command
        return _Process()

    monkeypatch.setattr(
        artemis_harness.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    context = _context(tmp_path)
    context.flow.host.adb_serial = "emulator-45562"
    result = asyncio.run(artemis_harness.ArtemisHarness().arun(context))
    assert result.success is True
    command = list(captured["command"])
    assert command[command.index("--test-name") + 1] == "omniflow_episode"
    assert command[command.index("--traces-path") + 1] == str(tmp_path / "artemis")


def test_config_pins_every_node_to_the_formal_model(monkeypatch, tmp_path):
    from src.experiment.protocol import FORMAL_MODEL

    config_path = artemis_harness._write_config(tmp_path, tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["default"]["model"] == FORMAL_MODEL
    assert payload["default"]["provider"] == "openai"
    assert payload["default"]["fallback"]["model"] == FORMAL_MODEL
    # A node that needs a Gemini-only model must be pinned, never left to inherit
    # a different provider than the row it is compared against.
    assert payload["nodes"]["object_detector"]["model"] == FORMAL_MODEL


def test_child_environment_does_not_borrow_a_gemini_key(monkeypatch, tmp_path):
    from src.experiment.protocol import FORMAL_MODEL, FORMAL_MODEL_BASE_URL

    monkeypatch.setenv("GOOGLE_API_KEY", "host-gemini-key")
    monkeypatch.setenv("LLMTHU_API_KEY", "formal-key")
    config_path = tmp_path / "artemis.jsonc"
    config_path.write_text("{}")
    environment = artemis_harness._child_environment(tmp_path, config_path)
    assert environment["OPENAI_MODEL"] == FORMAL_MODEL
    assert environment["OPENAI_BASE_URL"] == FORMAL_MODEL_BASE_URL.rstrip("/")
    assert environment["OPENAI_API_KEY"] == "formal-key"
    # An ARTEMIS row is defined as model-matched, so a host credential must not
    # quietly make some sub-agent fall back to Gemini.
    assert environment["GOOGLE_API_KEY"] == ""
    assert environment["GEMINI_API_KEY"] == ""
    assert environment["ARTEMIS_ARTEMIS_JSONC"] == str(config_path)
    assert environment["ARTEMIS_TRACES_DIR"] == str(config_path.parent)


def test_child_environment_uses_explicit_dashscope_experiment(monkeypatch, tmp_path):
    config_path = tmp_path / "artemis.jsonc"
    config_path.write_text("{}")
    monkeypatch.setenv("OMNIFLOW_EXPERIMENTAL_MODEL", "qwen3.6-plus")
    monkeypatch.setenv(
        "OMNIFLOW_EXPERIMENTAL_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    environment = artemis_harness._child_environment(tmp_path, config_path)
    assert environment["OPENAI_MODEL"] == "qwen3.6-plus"
    assert environment["ARTEMIS_DEFAULT_MODEL"] == "qwen3.6-plus"
    assert environment["OPENAI_BASE_URL"] == (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


def test_child_environment_prefers_dashscope_key_for_dashscope_endpoint(
    monkeypatch, tmp_path
):
    config_path = tmp_path / "artemis.jsonc"
    config_path.write_text("{}")
    monkeypatch.setenv("OMNIFLOW_EXPERIMENTAL_MODEL", "qwen3.6-plus")
    monkeypatch.setenv(
        "OMNIFLOW_EXPERIMENTAL_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-key")
    monkeypatch.setenv("LLMTHU_API_KEY", "stale-llmthu-key")
    environment = artemis_harness._child_environment(tmp_path, config_path)
    assert environment["OPENAI_API_KEY"] == "dashscope-key"


def test_outcome_is_read_from_disk_and_absent_outcome_is_not_invented(tmp_path):
    assert artemis_harness._read_outcome(tmp_path) == {}
    (tmp_path / "run_outcome.json").write_text(json.dumps({"task_status": "completed"}))
    assert artemis_harness._read_outcome(tmp_path) == {"task_status": "completed"}
    (tmp_path / "run_outcome.json").write_text("{ not json")
    assert artemis_harness._read_outcome(tmp_path) == {}


def test_child_diagnostics_keep_a_bounded_stderr_tail(tmp_path):
    path = tmp_path / "stderr.log"
    path.write_text("0123456789")
    assert artemis_harness._read_tail(path, limit=4) == "6789"
