from __future__ import annotations

from pathlib import Path

from src.integrations.official_forward import (
    prepare_appagent_workspace,
    prepare_mobilegpt_server,
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
    assert "emulator-5554" in Path(result["adb_proxy"]).read_text(encoding="utf-8")


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
