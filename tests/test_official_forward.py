from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
import pytest

from src.integrations.official_forward import (
    MOBILEGPT_HANDSHAKE_RETURN_CODE,
    _mobilegpt_protocol_probe,
    _count_appagent_actions,
    _autodroid_task_app_name,
    _autodroid_official_memory_key,
    prepare_appagent_workspace,
    prepare_mobilegpt_server,
    resolve_mobilegpt_client_host,
    run_appagent_executor,
    validate_autodroid_memory_root,
    write_adb_proxy,
)


def test_mobilegpt_protocol_probe_distinguishes_handshake_failure(
    tmp_path: Path,
) -> None:
    stats = tmp_path / "mobilegpt_stats.jsonl"
    stats.write_text("", encoding="utf-8")

    probe = _mobilegpt_protocol_probe(
        stats,
        "MobileGPT_Service: # of Apps : 21\n"
        "MobileGPT_CLIENT: server offline\n",
    )

    assert probe["client_service_ready"] is True
    assert probe["client_error"] is True
    assert probe["task_started"] is False
    assert probe["phase"] == "client_server_handshake"
    assert MOBILEGPT_HANDSHAKE_RETURN_CODE == 127


def test_mobilegpt_protocol_probe_marks_episode_after_server_task_start(
    tmp_path: Path,
) -> None:
    stats = tmp_path / "mobilegpt_stats.jsonl"
    stats.write_text(
        '{"event":"task_started"}\n{"event":"mobilegpt_action_sent"}\n',
        encoding="utf-8",
    )

    probe = _mobilegpt_protocol_probe(stats, "MobileGPT_Service: receive broadcast")

    assert probe["task_started"] is True
    assert probe["action_sent_count"] == 1
    assert probe["phase"] == "episode"


def test_autodroid_official_memory_key_uses_paper_app_names() -> None:
    assert _autodroid_official_memory_key("audio") == "voicerecorder"
    assert _autodroid_official_memory_key("files") == "filemanager"
    assert _autodroid_official_memory_key("camera") == "camera"


@contextmanager
def _fake_androidworld_session(task: object):
    yield object(), task


def test_appagent_forwarder_only_mounts_official_inputs(tmp_path: Path) -> None:
    official = tmp_path / "AppAgent"
    (official / "scripts").mkdir(parents=True)
    (official / "run.py").write_text("print('official')\n", encoding="utf-8")
    official_executor = (
        'import os\nimport re\n'
        'doc_path = os.path.join(docs_dir, f"{elem.uid}.txt")\n'
    )
    (official / "scripts" / "task_executor.py").write_text(
        official_executor, encoding="utf-8"
    )
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
    staged_scripts = Path(result["workspace"]) / "scripts"
    assert staged_scripts.is_dir()
    assert not staged_scripts.is_symlink()
    assert "_omniflow_resolution_agnostic_doc_path" in (
        staged_scripts / "task_executor.py"
    ).read_text(encoding="utf-8")
    compile(
        (staged_scripts / "task_executor.py").read_text(encoding="utf-8"),
        str(staged_scripts / "task_executor.py"),
        "exec",
    )
    assert (official / "scripts" / "task_executor.py").read_text(
        encoding="utf-8"
    ) == official_executor
    assert Path(result["app_dir"]).is_symlink()
    assert Path(result["app_dir"]).resolve() == docs.parent.resolve()
    proxy_text = Path(result["adb_proxy"]).read_text(encoding="utf-8")
    assert "emulator-5554" in proxy_text
    assert "Physical size:" in proxy_text


def test_appagent_adb_proxy_exposes_override_size_for_xml_coordinates(
    tmp_path: Path,
) -> None:
    real_adb = tmp_path / "adb"
    real_adb.write_text(
        "#!/bin/sh\n"
        "printf 'Physical size: 1768x2208\\nOverride size: 2208x1840\\n'\n",
        encoding="utf-8",
    )
    real_adb.chmod(0o755)
    proxy = write_adb_proxy(
        tmp_path / "workspace",
        serial="emulator-5554",
        adb_path=str(real_adb),
    )
    result = subprocess.run(
        [str(proxy), "-s", "emulator-5554", "shell", "wm", "size"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "Physical size: 2208x1840"


def test_adb_proxy_forwards_implicit_official_calls_to_selected_device(
    tmp_path: Path,
) -> None:
    real_adb = tmp_path / "adb"
    args_file = tmp_path / "args"
    real_adb.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$@\" > {args_file}\n"
        "printf 'device\\n'\n",
        encoding="utf-8",
    )
    real_adb.chmod(0o755)

    proxy = write_adb_proxy(
        tmp_path / "workspace",
        serial="emulator-5590",
        adb_path=str(real_adb),
    )

    result = subprocess.run(
        [str(proxy), "wait-for-device"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert args_file.read_text(encoding="utf-8").splitlines() == [
        "-s",
        "emulator-5590",
        "wait-for-device",
    ]


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

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    executor = tmp_path / "task_executor.py"
    executor.write_text("print('official')\n", encoding="utf-8")
    output = tmp_path / "result"
    assert run_appagent_executor(
        python_executable=sys.executable,
        executor=executor,
        app_name="settings",
        serial="emulator-5554",
        workspace=workspace,
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


def test_appagent_action_count_strips_ansi_and_excludes_finish() -> None:
    log = (
        "\x1b[33mRound 1\n"
        "\x1b[33mAction:\n"
        "\x1b[35mtap(6)\n"
        "\x1b[33mRound 2\n"
        "\x1b[33mAction:\n"
        "\x1b[35mtap(3)\n"
        "\x1b[33mRound 3\n"
        "\x1b[33mAction:\n"
        "\x1b[35mFINISH\n"
    )

    assert _count_appagent_actions(log) == 2


def test_appagent_validator_failure_keeps_clean_process_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Task:
        def is_successful(self, _env: object) -> float:
            return 0.0

    monkeypatch.setattr(
        "src.integrations.official_forward._androidworld_task_startup",
        lambda **_kwargs: _fake_androidworld_session(Task()),
    )
    output = tmp_path / "result"
    log_path = output / "official_appagent.log"
    official_output = (
        "\x1b[33mRound 1\n\x1b[33mAction:\n\x1b[35mtap(6)\n"
        "\x1b[33mRound 2\n\x1b[33mAction:\n\x1b[35mFINISH\n"
    )

    def fake_run(*_args, **_kwargs):
        output.mkdir(parents=True, exist_ok=True)
        log_path.write_text(official_output, encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "src.integrations.official_forward.subprocess.run", fake_run
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    executor = tmp_path / "task_executor.py"
    executor.write_text(
        "print(" + repr(official_output) + ")\n", encoding="utf-8"
    )
    assert run_appagent_executor(
        python_executable=sys.executable,
        executor=executor,
        app_name="settings",
        serial="emulator-5554",
        workspace=workspace,
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
    assert row["process_returncode"] == 0
    assert row["classification"] == "method_failure"
    assert row["official_validator_success"] is False
    assert row["actions_executed"] == 1
    assert row["model_calls"] == 0
    assert row["target_app_prelaunch_package"] == ""
    assert row["target_app_prelaunch_returncode"] is None


def test_appagent_step_budget_stops_official_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Task:
        def is_successful(self, _env: object) -> float:
            return 0.0

    monkeypatch.setattr(
        "src.integrations.official_forward._androidworld_task_startup",
        lambda **_kwargs: _fake_androidworld_session(Task()),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    executor = tmp_path / "task_executor.py"
    executor.write_text(
        "import time\n"
        "for index in range(10):\n"
        "    print(f'Round {index + 1}', flush=True)\n"
        "    print('Action:', flush=True)\n"
        "    print('tap(1)', flush=True)\n"
        "    time.sleep(0.1)\n",
        encoding="utf-8",
    )

    output = tmp_path / "result"
    assert run_appagent_executor(
        python_executable=sys.executable,
        executor=executor,
        app_name="settings",
        serial="emulator-5554",
        workspace=workspace,
        goal="Turn on Bluetooth",
        timeout_sec=5,
        android_world_root=tmp_path / "android-world",
        task_name="TurnOffWifiAndTurnOnBluetooth",
        task_params_json="{}",
        task_seed=113,
        console_port=5554,
        grpc_port=8554,
        adb_path="adb",
        output_root=output,
        perform_emulator_setup=False,
        max_steps=2,
    ) == 125

    row = json.loads((output / "task_results.jsonl").read_text())
    assert row["runtime_integrity_error"] == "appagent_step_budget_exhausted"
    assert row["actions_executed"] >= 2


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
    (server / "memory" / "memory_manager.py").write_text(
        'import os\n'
        'def select(candidates):\n'
        '    highest_similarity = candidates[0]\n'
        '    if highest_similarity > 0.95:\n'
        '        return 0\n',
        encoding="utf-8",
    )
    (server / "main.py").write_text(
        'os.environ["TASK_AGENT_GPT_VERSION"] = "gpt-4o"\n'
        'os.environ["vision_model"] = "gpt-4o"\n',
        encoding="utf-8",
    )
    (server / "utils" / "utils.py").write_text(
        'import os\n'
        'import re\n'
        'from openai import OpenAI\n'
        'def get_openai_embedding(text: str, model="text-embedding-3-small", **kwargs):\n'
        '    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))\n'
        '    text = text.replace("\\n", " ")\n'
        '    response = client.embeddings.create(input=[text], model=model, **kwargs)\n'
        '\n'
        '    return response.data[0].embedding\n'
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
    assert "MOBILEGPT_MEMORY_SIMILARITY_THRESHOLD" in (
        staged / "memory" / "memory_manager.py"
    ).read_text(encoding="utf-8")
    utils_source = (staged / "utils" / "utils.py").read_text(encoding="utf-8")
    assert "embedding_retry" in utils_source
    assert "range(1, 4)" in utils_source
    match = re.search(
        r"def parse_completion_rate\(completion_rate\).*?(?=\n\ndef |\Z)",
        utils_source,
        flags=re.DOTALL,
    )
    assert match is not None
    namespace: dict[str, object] = {"re": re}
    exec(match.group(0), namespace)
    assert namespace["parse_completion_rate"](
        "Starting the capture process, so 0% complete"
    ) == 0
    assert (staged / "agents" / "param_fill_agent.py").read_bytes() == (
        server / "agents" / "param_fill_agent.py"
    ).read_bytes()
    assert "gpt-4o" in (server / "main.py").read_text(encoding="utf-8")


def test_mobilegpt_forwarder_bridges_finish_to_official_client_frame(
    tmp_path: Path,
) -> None:
    official = tmp_path / "MobileGPT"
    server = official / "Server"
    (server / "memory").mkdir(parents=True)
    (server / "utils").mkdir()
    (server / "main.py").write_text("# official\n", encoding="utf-8")
    server_source = (
        "from utils.utils import log\n"
        "OMNIFLOW_INTERNAL_LAUNCH_ACTION = \"__omniflow_launch_package\"\n\n"
        "def _omniflow_send_action(client_socket, action):\n"
        "    action_name = (str(action.get(\"name\") or \"\").strip()\n"
        "                   if isinstance(action, dict) else \"\")\n"
        "    if (\n"
        "        isinstance(action, dict)\n"
        "        and str(action.get(\"name\") or \"\").strip() == OMNIFLOW_INTERNAL_LAUNCH_ACTION\n"
        "    ):\n"
        "        client_socket.send(\"launch\".encode())\n"
        "        return\n"
        "                task, is_new_task = task_agent.get_task(instruction)\n"
    )
    (server / "server.py").write_text(server_source, encoding="utf-8")
    (server / "utils" / "utils.py").write_text(
        "import os\nimport json\n\ndef log(*args):\n    pass\n",
        encoding="utf-8",
    )
    memory = tmp_path / "prepared"
    (memory / "frozen_memory").mkdir(parents=True)

    result = prepare_mobilegpt_server(
        official_root=official,
        memory_root=memory,
        workspace=tmp_path / "workspace",
    )

    staged_source = (Path(result["server_root"]) / "server.py").read_text(
        encoding="utf-8"
    )
    assert 'action_name == "finish"' in staged_source
    assert 'client_socket.send("$$$$$".encode())' in staged_source
    assert "MOBILEGPT_TARGET_TASK_NAME" in staged_source
    assert (server / "server.py").read_text(encoding="utf-8") == server_source


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
