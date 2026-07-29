from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "exp" / "run_androidworld.sh"


def test_experiment_script_is_the_only_shell_entry_and_has_safe_help() -> None:
    scripts = sorted(
        path.relative_to(REPO).as_posix()
        for path in (REPO / "scripts").rglob("*.sh")
    )
    assert scripts == ["scripts/exp/run_androidworld.sh"]

    completed = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--check-only" in completed.stdout
    assert "--all-tasks" in completed.stdout
    assert "--eight-cells" in completed.stdout
    assert "--tasks" in completed.stdout
    assert "--convert-ours-assets" in completed.stdout
    assert "OMNIFLOW_EXP_ASSET_ROOT" in completed.stdout
    assert completed.stderr == ""


def test_check_only_is_read_only_before_any_runtime_output(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    env_file = assets / ".env"
    env_file.write_text("", encoding="utf-8")
    source_run_log = assets / "source.run_log.json"
    source_run_log.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.canonical_run_log.v1",
                "run_id": "source",
                "goal": "Turn Bluetooth on.",
                "status": "succeeded",
                "success": True,
                "steps": [
                    {
                        "step_index": 0,
                        "before_state_id": "before",
                        "action": {
                            "tool": "click",
                            "args": {"x": 500, "y": 500},
                        },
                        "result": {"success": True},
                        "after_state_id": "after",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    source_index = assets / "index.json"
    source_index.write_text(
        json.dumps(
            {
                "SystemBluetoothTurnOn": {
                    "goal": "Turn Bluetooth on.",
                    "params": {"on_or_off": "on"},
                    "replay_seed": 111,
                    "step_count": 1,
                    "retained_source_run_log": str(source_run_log),
                    "method": "fixed_replay",
                    "latest_official_success_source": True,
                    "source_kind": (
                        "androidworld_validator_success_source_runlog"
                    ),
                    "source_run_log_sha256": hashlib.sha256(
                        source_run_log.read_bytes()
                    ).hexdigest(),
                }
            }
        ),
        encoding="utf-8",
    )
    android_world = assets / "android_world"
    setup_file = (
        android_world
        / "android_world"
        / "env"
        / "setup_device"
        / "apps.py"
    )
    setup_file.parent.mkdir(parents=True)
    setup_file.write_text("", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for name in ("adb", "java"):
        executable = fake_bin / name
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
    results = tmp_path / "results-never-created"

    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
        "OMNIFLOW_EXP_ASSET_ROOT": str(assets),
        "OMNIFLOW_EXP_RESULTS_ROOT": str(results),
        "OMNIFLOW_ENV_FILE": str(env_file),
        "OMNIFLOW_SINGLE_TASK_SOURCE_INDEX": str(source_index),
        "OMNIFLOW_MASTER_SOURCE_INDEX": str(source_index),
        "OMNIFLOW_SOURCE_INDEX_EXPECTED_TASKS": "1",
        "OMNIFLOW_ANDROID_WORLD_ROOT": str(android_world),
        "OMNIFLOW_ADB_PATH": str(fake_bin / "adb"),
        "OMNIFLOW_SINGLE_TASK_MANAGE_EMULATORS": "0",
        "OMNIFLOW_SINGLE_TASK_METHODS": "fixed_replay",
        "PYTHON_BIN": sys.executable,
    }
    completed = subprocess.run(
        ["bash", str(SCRIPT), "--check-only"],
        cwd=REPO,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "[static] ready" in completed.stdout
    assert not results.exists()


def test_asset_conversion_routes_through_the_only_script(
    tmp_path: Path,
) -> None:
    legacy_v1 = tmp_path / "legacy-v1"
    legacy_v2 = tmp_path / "legacy-v2"
    legacy_v1.mkdir()
    legacy_v2.mkdir()
    source_index = tmp_path / "source-index.json"
    source_index.write_text("{}", encoding="utf-8")
    output_root = tmp_path / "converted"
    captured = tmp_path / "python-args.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$CAPTURE_ARGS\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = {
        **os.environ,
        "PYTHON_BIN": str(fake_python),
        "CAPTURE_ARGS": str(captured),
        "OMNIFLOW_LEGACY_FUNCTION_ROOTS": (
            f"{legacy_v1}:{legacy_v2}"
        ),
        "OMNIFLOW_OURS_SOURCE_ASSET_INDEX": str(source_index),
        "OMNIFLOW_OURS_CONVERTED_ASSET_ROOT": str(output_root),
    }

    completed = subprocess.run(
        ["bash", str(SCRIPT), "--convert-ours-assets"],
        cwd=REPO,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert captured.read_text(encoding="utf-8").splitlines() == [
        "-m",
        "src.experiment.function_assets",
        "--source-asset-index",
        str(source_index),
        "--output-root",
        str(output_root),
        "--legacy-root",
        str(legacy_v1),
        "--legacy-root",
        str(legacy_v2),
    ]
