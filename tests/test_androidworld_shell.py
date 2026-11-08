from __future__ import annotations

import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_public_shell_entry_prefers_dashscope_after_loading_runtime_files(tmp_path):
    stub_python = tmp_path / "python"
    stub_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'OPENAI_API_KEY=%s\\n' \"${OPENAI_API_KEY:-}\"\n"
        "printf 'OMNIFLOW_EXPERIMENTAL_BASE_URL=%s\\n' "
        "\"${OMNIFLOW_EXPERIMENTAL_BASE_URL:-}\"\n",
        encoding="utf-8",
    )
    stub_python.chmod(0o755)

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON_BIN": str(stub_python),
            "OPENAI_API_KEY": "",
            "DASHSCOPE_API_KEY": "dashscope-key",
            "OMNIFLOW_EXPERIMENTAL_BASE_URL": (
                "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
        }
    )
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts/exp/run_androidworld.sh"), "run"],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "OPENAI_API_KEY=dashscope-key" in result.stdout
    assert (
        "OMNIFLOW_EXPERIMENTAL_BASE_URL="
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
        in result.stdout
    )
