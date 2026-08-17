from __future__ import annotations

import json
from pathlib import Path
import sys
import types
import subprocess

from src.integrations.android_world.launch import (
    _ActionConsistencyLlmWrapper,
    _apply_candidate_harness_proposal,
    _build_official_androidworld_agent,
    _prepare_androidworld_snapshot_restore,
    _prepare_official_harness_episode,
    _persist_official_step_captures,
)


class _FakeTracker:
    request_records = [
        {
            "kind": "action",
            "prompt": "stock prompt",
            "image_payloads": [b"exact-jpeg-payload"],
            "duration_ms": 12.5,
        }
    ]


class _FakeAgent:
    _omniflow_llm_usage_tracker = _FakeTracker()
    history = [
        {
            "action_prompt": "stock prompt",
            "action_output": 'Reason: target\nAction: {"action_type":"click","index":1}',
            "action_raw_response": {"id": "response"},
            "action_output_json": {"action_type": "click", "index": 1},
        }
    ]


def test_stock_capture_persists_exact_model_images_without_reference_action(
    tmp_path: Path,
) -> None:
    result = _persist_official_step_captures(
        output_dir=tmp_path,
        agent=_FakeAgent(),
        selected_agent="official:m3a",
        task_name="ContactsAddContact",
        goal="Create a contact",
        task_params_sha256="params-sha",
    )

    assert result is not None
    payload = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    row = payload["steps"][0]
    assert row["harness_id"] == "m3a"
    assert row["modality"] == "vision_text"
    assert row["reference_action_available_to_runtime"] is False
    assert "reference_action" not in row
    image = row["model_input_images"][0]
    assert Path(image["path"]).read_bytes() == b"exact-jpeg-payload"
    assert image["exact_model_payload"] is True


def test_stock_capture_persists_history_beyond_legacy_seven_step_limit(
    tmp_path: Path,
) -> None:
    history = [
        {
            "action_prompt": f"prompt-{index}",
            "action_output": '{"action_type":"wait"}',
            "action_raw_response": {"id": str(index)},
            "action_output_json": {"action_type": "wait"},
        }
        for index in range(8)
    ]
    tracker = type(
        "Tracker",
        (),
        {
            "request_records": [
                {"kind": "action", "image_payloads": [f"image-{index}".encode()]}
                for index in range(8)
            ]
        },
    )
    agent = type("Agent", (), {"history": history, "_omniflow_llm_usage_tracker": tracker})()

    result = _persist_official_step_captures(
        output_dir=tmp_path,
        agent=agent,
        selected_agent="official:t3a",
        task_name="BrowserDraw",
        goal="Draw",
        task_params_sha256="params-sha",
    )

    assert result is not None
    payload = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert payload["steps"][-1]["step_index"] == 8
    assert len(payload["steps"]) == 8


def test_official_builder_dispatches_stock_t3a_and_m3a(monkeypatch) -> None:
    class FakeT3A:
        def __init__(self, env, llm):
            self.env = env
            self.llm = llm

    class FakeM3A(FakeT3A):
        pass

    agents_module = types.ModuleType("android_world.agents")
    agents_module.t3a = types.SimpleNamespace(T3A=FakeT3A)
    agents_module.m3a = types.SimpleNamespace(M3A=FakeM3A)
    android_world_module = types.ModuleType("android_world")
    android_world_module.agents = agents_module
    monkeypatch.setitem(sys.modules, "android_world", android_world_module)
    monkeypatch.setitem(sys.modules, "android_world.agents", agents_module)
    monkeypatch.setitem(sys.modules, "android_world.agents.t3a", agents_module.t3a)
    monkeypatch.setitem(sys.modules, "android_world.agents.m3a", agents_module.m3a)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    t3a_agent = _build_official_androidworld_agent(
        env="env", official_agent_name="t3a", model_name="GLM-5.1"
    )
    m3a_agent = _build_official_androidworld_agent(
        env="env", official_agent_name="m3a", model_name="GLM-5.1"
    )

    assert isinstance(t3a_agent, FakeT3A)
    assert not isinstance(t3a_agent, FakeM3A)
    assert isinstance(m3a_agent, FakeM3A)
    assert t3a_agent.name == "t3a"
    assert m3a_agent.name == "m3a"
    assert t3a_agent.llm.model == "GLM-5.1"
    assert m3a_agent.llm.model == "GLM-5.1"


def test_candidate_harness_wrapper_applies_only_generic_append_guidance(
    tmp_path: Path,
) -> None:
    proposal = tmp_path / "proposal.json"
    proposal.write_text(
        json.dumps(
            {
                "schema_version": "skymark.harness_revision.v1",
                "proposal_id": "m3a:proposal:r1",
                "harness_version_id": "m3a:candidate:r1",
                "harness_id": "m3a",
                "patches": [
                    {
                        "seam": "history_policy",
                        "operation": "append",
                        "value": "Track unsatisfied Goal constraints.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    guidelines: list[str] = []
    agent_type = type(
        "M3A",
        (),
        {
            "history": [],
            "set_task_guidelines": lambda self, values: guidelines.extend(values),
        },
    )

    applied = _apply_candidate_harness_proposal(agent_type(), str(proposal))

    assert applied is not None
    assert applied["harness_version_id"] == "m3a:candidate:r1"
    assert guidelines == ["Track unsatisfied Goal constraints."]


def test_stock_capture_script_forwards_fixed_task_params(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    output = tmp_path / "capture"
    env_file = tmp_path / "model.env"
    env_file.write_text(
        "LLMTHU_KEY=test\nLLMTHU_BASE_URL=https://llmapi.paratera.com/v1\n",
        encoding="utf-8",
    )
    android_world = tmp_path / "android-world"
    (android_world / "android_world").mkdir(parents=True)
    sdk = tmp_path / "sdk"
    (sdk / "platform-tools").mkdir(parents=True)
    (sdk / "emulator").mkdir(parents=True)
    for executable in (sdk / "platform-tools" / "adb", sdk / "emulator" / "emulator"):
        executable.write_text("#!/bin/sh\n", encoding="utf-8")
        executable.chmod(0o755)
    task_params = '{"name":"OOB Contact Omicron","number":"5558642097"}'
    result = subprocess.run(
        [
            "bash",
            str(repo / "scripts/exp/run_androidworld.sh"),
            "--stock-capture",
            "m3a",
            "--tasks",
            "ContactsAddContact",
            "--dry-run",
        ],
        cwd=repo,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
            "PYTHON_BIN": sys.executable,
            "OMNIFLOW_ANDROID_WORLD_ROOT": str(android_world),
            "OMNIFLOW_ANDROID_SDK_ROOT": str(sdk),
            "OMNIFLOW_ENV_FILE": str(env_file),
            "OMNIFLOW_STOCK_CAPTURE_OUTPUT_PATH": str(output),
            "OMNIFLOW_STOCK_CAPTURE_TASK_PARAMS_JSON": task_params,
        },
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--task-params-json" in result.stdout
    assert "OOB\\ Contact\\ Omicron" in result.stdout


def test_stock_capture_script_allows_bounded_full_episode_for_candidate_e2e(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[1]
    output = tmp_path / "capture"
    env_file = tmp_path / "model.env"
    env_file.write_text(
        "LLMTHU_KEY=test\nLLMTHU_BASE_URL=https://llmapi.paratera.com/v1\n",
        encoding="utf-8",
    )
    android_world = tmp_path / "android-world"
    (android_world / "android_world").mkdir(parents=True)
    sdk = tmp_path / "sdk"
    (sdk / "platform-tools").mkdir(parents=True)
    (sdk / "emulator").mkdir(parents=True)
    for executable in (sdk / "platform-tools" / "adb", sdk / "emulator" / "emulator"):
        executable.write_text("#!/bin/sh\n", encoding="utf-8")
        executable.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            str(repo / "scripts/exp/run_androidworld.sh"),
            "--stock-capture",
            "m3a",
            "--tasks",
            "ContactsAddContact",
            "--dry-run",
        ],
        cwd=repo,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
            "PYTHON_BIN": sys.executable,
            "OMNIFLOW_ANDROID_WORLD_ROOT": str(android_world),
            "OMNIFLOW_ANDROID_SDK_ROOT": str(sdk),
            "OMNIFLOW_ENV_FILE": str(env_file),
            "OMNIFLOW_STOCK_CAPTURE_OUTPUT_PATH": str(output),
            "OMNIFLOW_STOCK_CAPTURE_MAX_STEPS": "20",
        },
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--max-steps 20" in result.stdout


def test_stock_capture_script_passes_semantic_source_hint(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    output = tmp_path / "capture"
    env_file = tmp_path / "model.env"
    env_file.write_text(
        "LLMTHU_KEY=test\nLLMTHU_BASE_URL=https://llmapi.paratera.com/v1\n",
        encoding="utf-8",
    )
    hint = tmp_path / "hint.json"
    hint.write_text('{"schema_version":"omniflow.t3a_semantic_hint.v2"}', encoding="utf-8")
    android_world = tmp_path / "android-world"
    (android_world / "android_world").mkdir(parents=True)
    sdk = tmp_path / "sdk"
    (sdk / "platform-tools").mkdir(parents=True)
    (sdk / "emulator").mkdir(parents=True)
    for executable in (sdk / "platform-tools" / "adb", sdk / "emulator" / "emulator"):
        executable.write_text("#!/bin/sh\n", encoding="utf-8")
        executable.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            str(repo / "scripts/exp/run_androidworld.sh"),
            "--stock-capture",
            "m3a",
            "--tasks",
            "ContactsAddContact",
            "--dry-run",
        ],
        cwd=repo,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
            "PYTHON_BIN": sys.executable,
            "OMNIFLOW_ANDROID_WORLD_ROOT": str(android_world),
            "OMNIFLOW_ANDROID_SDK_ROOT": str(sdk),
            "OMNIFLOW_ENV_FILE": str(env_file),
            "OMNIFLOW_STOCK_CAPTURE_OUTPUT_PATH": str(output),
            "OMNIFLOW_STOCK_CAPTURE_SOURCE_ACTION_HINT_PATH": str(hint),
        },
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--source-action-hint-path" in result.stdout
    assert str(hint) in result.stdout


def test_action_consistency_wrapper_returns_reviewed_action(monkeypatch) -> None:
    class Delegate:
        def __init__(self) -> None:
            self.calls = []

        def predict_mm(self, prompt, images):
            self.calls.append((prompt, images))
            if len(self.calls) == 1:
                return (
                    'Reason: explore optional fields\nAction: {"action_type":"click","index":9}',
                    None,
                    {"id": "first"},
                )
            return (
                'Reason: recover the required phone field\nAction: {"action_type":"scroll","direction":"up"}',
                None,
                {"id": "review"},
            )

    agents_module = types.ModuleType("android_world.agents")
    agents_module.m3a_utils = types.SimpleNamespace(
        parse_reason_action_output=lambda text: text.split("\nAction: ", 1)
    )
    agents_module.agent_utils = types.SimpleNamespace(
        extract_json=lambda text: json.loads(text)
    )
    android_world_module = types.ModuleType("android_world")
    android_world_module.agents = agents_module
    monkeypatch.setitem(sys.modules, "android_world", android_world_module)
    monkeypatch.setitem(sys.modules, "android_world.agents", agents_module)

    delegate = Delegate()
    wrapper = _ActionConsistencyLlmWrapper(
        delegate,
        {"mode": "always", "instruction": "Check direct progress."},
    )
    output, _, metadata = wrapper.predict_mm("Your Answer:", ["image"])

    assert '"direction":"up"' in output
    assert metadata["action_consistency_applied"] is True
    assert len(delegate.calls) == 2
    assert delegate.calls[0][1] == delegate.calls[1][1]


def test_keyboard_obstruction_guard_dismisses_keyboard_without_second_call() -> None:
    class Delegate:
        def __init__(self) -> None:
            self.calls = []

        def predict_mm(self, prompt, images):
            self.calls.append((prompt, images))
            return (
                'Reason: find the hidden phone field\nAction: {"action_type":"scroll","direction":"down"}',
                None,
                {"id": "first"},
            )

    prompt = (
        'Step 4- Action selected: {"action_type": "input_text", "text": "Mohammed", "index": 8}.\n'
        'UI element 9: {"content_description": "Switch input method"}\n'
        'UI element 45: {"content_description": "Symbol keyboard"}\n'
        'UI element 47: {"content_description": "Emoji button"}\n'
        "Your Answer:"
    )
    delegate = Delegate()
    wrapper = _ActionConsistencyLlmWrapper(
        delegate,
        {"mode": "keyboard_obstruction_guard", "instruction": "Dismiss first."},
    )

    output, _, metadata = wrapper.predict_mm(prompt, ["image"])

    assert output.endswith('Action: {"action_type":"navigate_back"}')
    assert metadata["action_consistency_applied"] is True
    assert metadata["action_consistency_mode"] == "keyboard_obstruction_guard"
    assert len(delegate.calls) == 1


def test_keyboard_obstruction_guard_passes_nonmatching_action_through() -> None:
    class Delegate:
        def __init__(self) -> None:
            self.calls = []

        def predict_mm(self, prompt, images):
            self.calls.append((prompt, images))
            return (
                'Reason: select the visible phone field\nAction: {"action_type":"click","index":5}',
                None,
                {"id": "first"},
            )

    delegate = Delegate()
    wrapper = _ActionConsistencyLlmWrapper(
        delegate,
        {"mode": "keyboard_obstruction_guard", "instruction": "Dismiss first."},
    )
    output, _, metadata = wrapper.predict_mm("Your Answer:", ["image"])

    assert '"action_type":"click"' in output
    assert metadata == {"id": "first"}
    assert len(delegate.calls) == 1


def test_official_harness_episode_starts_from_home(monkeypatch) -> None:
    calls = []
    adb_utils_module = types.ModuleType("android_world.env.adb_utils")
    adb_utils_module.press_home_button = lambda controller: calls.append(controller)
    env_module = types.ModuleType("android_world.env")
    env_module.adb_utils = adb_utils_module
    android_world_module = types.ModuleType("android_world")
    android_world_module.env = env_module
    monkeypatch.setitem(sys.modules, "android_world", android_world_module)
    monkeypatch.setitem(sys.modules, "android_world.env", env_module)
    monkeypatch.setitem(sys.modules, "android_world.env.adb_utils", adb_utils_module)
    controller = object()
    env = types.SimpleNamespace(
        controller=controller
    )

    _prepare_official_harness_episode(env, selected_agent="official:m3a")

    assert calls == [controller]


def test_non_official_episode_does_not_force_home() -> None:
    calls = []
    env = types.SimpleNamespace(
        controller=types.SimpleNamespace(
            execute_action=lambda action: calls.append(action)
        )
    )

    _prepare_official_harness_episode(env, selected_agent="omniflow")

    assert calls == []


def test_snapshot_restore_prepares_empty_app_data_directory(monkeypatch) -> None:
    calls = []
    adb_utils_module = types.ModuleType("android_world.env.adb_utils")
    adb_utils_module.issue_generic_request = (
        lambda arguments, controller: calls.append((arguments, controller)) or "ok"
    )
    adb_utils_module.check_ok = lambda response, message: None
    env_module = types.ModuleType("android_world.env")
    env_module.adb_utils = adb_utils_module
    android_world_module = types.ModuleType("android_world")
    android_world_module.env = env_module
    monkeypatch.setitem(sys.modules, "android_world", android_world_module)
    monkeypatch.setitem(sys.modules, "android_world.env", env_module)
    monkeypatch.setitem(sys.modules, "android_world.env.adb_utils", adb_utils_module)
    controller = object()
    env = types.SimpleNamespace(controller=controller)
    app = types.SimpleNamespace(package_name=lambda: "com.example.app")

    _prepare_androidworld_snapshot_restore(env, [app])

    assert calls == [
        (["shell", "mkdir", "-p", "/data/data/com.example.app"], controller),
        (
            [
                "shell",
                "touch",
                "/data/data/com.example.app/omniflow_snapshot_restore_placeholder",
            ],
            controller,
        ),
    ]
