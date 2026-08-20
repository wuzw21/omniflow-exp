from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from src.integrations.official_forward import (
    _autodroid_task_app_name,
    prepare_appagent_workspace,
    prepare_mobilegpt_server,
    resolve_mobilegpt_client_host,
    run_appagent_executor,
    validate_autodroid_memory_root,
)


@contextmanager
def _fake_androidworld_session(task: object):
    yield object(), task


def test_appagent_forwarder_only_mounts_official_inputs(tmp_path: Path) -> None:
    official = tmp_path / "AppAgent"
    (official / "scripts").mkdir(parents=True)
    (official / "run.py").write_text("print('official')\n", encoding="utf-8")
    docs = tmp_path / "memory" / "calculator" / "demo_docs"
    docs.mkdir(parents=True)
    (docs / "button.txt").write_text("{}\n", encoding="utf-8")

    result = prepare_appagent_workspace(
        official_root=official,
        docs_root=docs,
        workspace=tmp_path / "workspace",
        app_name="calculator",
        serial="emulator-5554",
        adb_path="adb",
        config={"MODEL": "OpenAI"},
    )

    assert Path(result["workspace"]).is_dir()
    assert (Path(result["workspace"]) / "scripts").resolve() == (
        official / "scripts"
    ).resolve()
    assert Path(result["app_dir"]).is_symlink()
    assert Path(result["app_dir"]).resolve() == docs.parent.resolve()
    proxy_text = Path(result["adb_proxy"]).read_text(encoding="utf-8")
    assert "emulator-5554" in proxy_text
    assert "Physical size:" in proxy_text


def test_appagent_forwarder_writes_validator_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Task:
        def is_successful(self, _env: object) -> float:
            return 1.0

    monkeypatch.setattr(
        "src.integrations.official_forward._androidworld_task_startup",
        lambda **_kwargs: _fake_androidworld_session(Task()),
    )
    monkeypatch.setattr(
        "src.integrations.official_forward.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )

    output = tmp_path / "result"
    assert run_appagent_executor(
        python_executable="python",
        executor=tmp_path / "task_executor.py",
        app_name="settings",
        serial="emulator-5554",
        workspace=tmp_path / "workspace",
        goal="Turn on Bluetooth",
        timeout_sec=10,
        android_world_root=tmp_path / "android-world",
        task_name="TurnOffWifiAndTurnOnBluetooth",
        task_params_json="{}",
        task_seed=113,
        console_port=5554,
        grpc_port=8554,
        adb_path="adb",
        output_root=output,
        perform_emulator_setup=False,
    ) == 0

    row = json.loads((output / "task_results.jsonl").read_text())
    assert row["official_validator_used"] is True
    assert row["official_validator_success"] is True
    assert row["androidworld_validator_result"]["validator"] == (
        "androidworld_official"
    )


def test_mobilegpt_forwarder_configures_staged_glm_models_and_overlays_memory(
    tmp_path: Path,
) -> None:
    official = tmp_path / "MobileGPT"
    server = official / "Server"
    (server / "memory").mkdir(parents=True)
    (server / "main.py").write_text("print('official server')\n", encoding="utf-8")
    (server / "memory" / "memory_manager.py").write_text(
        "# official memory package\n", encoding="utf-8"
    )
    memory = tmp_path / "prepared"
    (memory / "frozen_memory" / "com.example.app").mkdir(parents=True)
    (memory / "frozen_memory" / "com.example.app" / "pages.csv").write_text(
        "index\n", encoding="utf-8"
    )

    result = prepare_mobilegpt_server(
        official_root=official,
        memory_root=memory,
        workspace=tmp_path / "workspace",
    )

    staged = Path(result["server_root"])
    assert (staged / "main.py").read_text(encoding="utf-8") == (
        "print('official server')\n"
    )
    assert (staged / "memory" / "memory_manager.py").is_file()
    assert (staged / "memory" / "com.example.app" / "pages.csv").is_file()


def test_mobilegpt_forwarder_keeps_official_server_source_unchanged(
    tmp_path: Path,
) -> None:
    official = tmp_path / "MobileGPT"
    server = official / "Server"
    (server / "memory").mkdir(parents=True)
    (server / "utils").mkdir()
    (server / "agents").mkdir()
    (server / "main.py").write_text(
        'os.environ["TASK_AGENT_GPT_VERSION"] = "gpt-4o"\n'
        'os.environ["vision_model"] = "gpt-4o"\n',
        encoding="utf-8",
    )
    (server / "utils" / "utils.py").write_text(
        'import os\n'
        'import re\n'
        'def get_openai_embedding(text: str, model="text-embedding-3-small", **kwargs):\n'
        '    return []\n'
        'def parse_completion_rate(completion_rate):\n'
        '    input_str = str(completion_rate).strip()\n'
        '    if input_str.endswith("%"):\n'
        '        return int(float(input_str[:-1]))\n'
        '    else:\n'
        '        value = float(input_str)\n'
        '    return int(value)\n',
        encoding="utf-8",
    )
    (server / "agents" / "param_fill_agent.py").write_text(
        'def call(query):\n'
        '    return query(model="gpt-4o")\n',
        encoding="utf-8",
    )
    memory = tmp_path / "prepared"
    (memory / "frozen_memory").mkdir(parents=True)

    result = prepare_mobilegpt_server(
        official_root=official,
        memory_root=memory,
        workspace=tmp_path / "workspace",
        embedding_model="GLM-Embedding-2",
        chat_model="GLM-5.1",
    )

    staged = Path(result["server_root"])
    assert (staged / "main.py").read_bytes() != (server / "main.py").read_bytes()
    assert "GLM-5.1" in (staged / "main.py").read_text(encoding="utf-8")
    assert (staged / "utils" / "utils.py").read_bytes() != (
        server / "utils" / "utils.py"
    ).read_bytes()
    assert "GLM-Embedding-2" in (
        staged / "utils" / "utils.py"
    ).read_text(encoding="utf-8")
    assert (staged / "agents" / "param_fill_agent.py").read_bytes() == (
        server / "agents" / "param_fill_agent.py"
    ).read_bytes()
    assert "gpt-4o" in (server / "main.py").read_text(encoding="utf-8")


def test_mobilegpt_forwarder_does_not_patch_task_agent(
    tmp_path: Path,
) -> None:
    official = tmp_path / "MobileGPT"
    server = official / "Server"
    (server / "memory").mkdir(parents=True)
    (server / "agents").mkdir()
    (server / "main.py").write_text("# official\n", encoding="utf-8")
    (server / "agents" / "task_agent.py").write_text(
        "import os\n"
        "from agents.prompts import task_agent_prompt\n"
        "from utils.utils import query, log\n\n"
        "class TaskAgent:\n"
        "    def get_task(self, instruction):\n"
        "        known_tasks = []\n"
        "        response = query(messages=task_agent_prompt.get_prompts(instruction, known_tasks),\n"
        "                         model=os.getenv(\"TASK_AGENT_GPT_VERSION\"))\n"
        "        task = response[\"api\"]\n"
        "        return task, True\n",
        encoding="utf-8",
    )
    memory = tmp_path / "prepared"
    (memory / "frozen_memory").mkdir(parents=True)

    result = prepare_mobilegpt_server(
        official_root=official,
        memory_root=memory,
        workspace=tmp_path / "workspace",
        chat_model="GLM-4.6V",
    )

    staged_source = (
        Path(result["server_root"]) / "agents" / "task_agent.py"
    ).read_bytes()
    assert staged_source == (server / "agents" / "task_agent.py").read_bytes()


def test_mobilegpt_host_defaults_to_emulator_alias() -> None:
    assert resolve_mobilegpt_client_host(
        "",
        serial="emulator-5560",
        adb_path="adb",
    ) == "10.0.2.2"


def test_mobilegpt_explicit_host_wins_over_device_detection() -> None:
    assert resolve_mobilegpt_client_host(
        "192.168.1.155",
        serial="physical-device",
        adb_path="missing-adb",
    ) == "192.168.1.155"


def test_autodroid_memory_manifest_is_validated_without_conversion(
    tmp_path: Path,
) -> None:
    root = tmp_path / "androidworld_apps"
    root.mkdir()
    (root / "memory_manifest.json").write_text(
        '{"format":"autodroid-droidbot-memory-manifest-v1",'
        '"apps":[{"name":"audio"}],"device":{}}\n',
        encoding="utf-8",
    )

    result = validate_autodroid_memory_root(root)

    assert result["app_count"] == 1
    assert result["memory_root"] == str(root.resolve())
    assert len(result["manifest_sha256"]) == 64


def test_autodroid_task_app_name_maps_androidworld_alias() -> None:
    assert _autodroid_task_app_name(
        type("Task", (), {"app_names": ("simple sms messenger",)})()
    ) == "sms"


def test_autodroid_task_app_name_uses_first_declared_app_for_composite_task() -> None:
    assert _autodroid_task_app_name(
        type(
            "Task",
            (),
            {"app_names": ("pro expense", "simple gallery pro")},
        )()
    ) == "expense"
