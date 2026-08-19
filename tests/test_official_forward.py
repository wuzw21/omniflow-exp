from __future__ import annotations

from pathlib import Path

from src.integrations.official_forward import (
    prepare_appagent_workspace,
    prepare_mobilegpt_server,
    resolve_mobilegpt_client_host,
)


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


def test_mobilegpt_forwarder_keeps_server_code_and_overlays_memory(
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


def test_mobilegpt_forwarder_configures_models_only_in_staging(
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
        'def get_openai_embedding(text: str, model="text-embedding-3-small", **kwargs):\n'
        '    return []\n',
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
    staged_utils = (staged / "utils" / "utils.py").read_text(encoding="utf-8")
    staged_main = (staged / "main.py").read_text(encoding="utf-8")
    staged_param = (staged / "agents" / "param_fill_agent.py").read_text(
        encoding="utf-8"
    )
    assert "MOBILEGPT_EMBEDDING_MODEL" in staged_utils
    assert "MOBILEGPT_CHAT_MODEL" in staged_main
    assert "MOBILEGPT_CHAT_MODEL" in staged_param
    assert "text-embedding-3-small" in (
        server / "utils" / "utils.py"
    ).read_text(encoding="utf-8")


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
