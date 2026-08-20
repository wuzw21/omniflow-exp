from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_4090_resources.sh"


def test_4090_builder_help_is_available() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--ssh HOST" in result.stdout
    assert "--upgrade-python-deps" in result.stdout


def test_4090_builder_requires_a_target() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--ssh HOST" in result.stderr


def test_latest_deployment_can_override_protocol_revisions() -> None:
    protocol = (ROOT / "src/experiment/protocol.py").read_text()
    launcher = (ROOT / "scripts/exp/run_androidworld.sh").read_text()
    assert "OMNIFLOW_ANDROIDWORLD_REVISION" in protocol
    assert "OMNIFLOW_BMOCA_REVISION" in launcher
