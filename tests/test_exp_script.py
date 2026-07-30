from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from src.experiment.artifact_memory import refresh_artifact_memory
from src.experiment.preflight import REQUIRED_DISTRIBUTION_VERSIONS

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "exp" / "run_androidworld.sh"


def test_android_env_version_is_locked_and_preflight_enforced() -> None:
    assert REQUIRED_DISTRIBUTION_VERSIONS == {"android-env": "1.2.3"}
    assert '"android-env==1.2.3"' in (REPO / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    lock_text = (REPO / "uv.lock").read_text(encoding="utf-8")
    assert 'name = "android-env"\nversion = "1.2.3"' in lock_text


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
    assert "--refresh-memory" in completed.stdout
    assert "OMNIFLOW_EXP_ASSET_ROOT" in completed.stdout
    assert "OMNIFLOW_EXP_MEMORY_ROOT" in completed.stdout
    assert "OMNIFLOW_OURS_AUTHORING_MANIFEST" in completed.stdout
    assert completed.stderr == ""
    assert SCRIPT.read_text(encoding="utf-8").count('bash "$0"') == 2


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
                "schema_version": "omniflow.run_log.v1",
                "run_id": "source",
                "goal": "Turn Bluetooth on.",
                "completed": True,
                "success": True,
                "steps": [
                    {
                        "step_index": 0,
                        "observation_before_act": {
                            "state_id": "before",
                            "width": 1000,
                            "height": 1000,
                        },
                        "executed_actions": [
                            {
                                "type": "click",
                                "params": {"x": 500, "y": 500},
                            }
                        ],
                        "success": True,
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
                    "retained_source_run_log_sha256": hashlib.sha256(
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
    fake_adb = fake_bin / "adb"
    fake_adb.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_adb.chmod(0o755)
    fake_java = fake_bin / "java"
    fake_java.write_text(
        "#!/bin/sh\nprintf '%s\\n' 'openjdk version \"17.0.19\"' >&2\n",
        encoding="utf-8",
    )
    fake_java.chmod(0o755)
    results = tmp_path / "results-never-created"
    memory_root = tmp_path / "memory"
    refresh_artifact_memory(
        memory_root=memory_root,
        source_index=source_index,
        function_catalogs=(),
        runlog_roots=(assets,),
        result_roots=(),
    )

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
        "OMNIFLOW_ADB_PATH": str(fake_adb),
        "OMNIFLOW_SINGLE_TASK_MANAGE_EMULATORS": "0",
        "OMNIFLOW_SINGLE_TASK_METHODS": "fixed_replay",
        "OMNIFLOW_EXP_MEMORY_INDEX": str(memory_root / "current.json"),
        "OMNIFLOW_JAVA_HOME": str(tmp_path),
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

    fake_java.write_text(
        "#!/bin/sh\nprintf '%s\\n' 'openjdk version \"11.0.31\"' >&2\n",
        encoding="utf-8",
    )
    unsupported_java = subprocess.run(
        ["bash", str(SCRIPT), "--check-only"],
        cwd=REPO,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert unsupported_java.returncode != 0
    assert "Java 17 or newer is required" in unsupported_java.stderr
    assert not results.exists()


def test_asset_conversion_routes_through_the_only_script(
    tmp_path: Path,
) -> None:
    source_index = tmp_path / "source-index.json"
    source_index.write_text("{}", encoding="utf-8")
    output_root = tmp_path / "converted"
    memory_index = tmp_path / "current.json"
    memory_index.write_text("{}", encoding="utf-8")
    authoring_manifest = tmp_path / "authoring-manifest.json"
    authoring_manifest.write_text("{}", encoding="utf-8")
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
        "OMNIFLOW_OURS_SOURCE_ASSET_INDEX": str(source_index),
        "OMNIFLOW_OURS_CONVERTED_ASSET_ROOT": str(output_root),
        "OMNIFLOW_OURS_AUTHORING_MANIFEST": str(authoring_manifest),
        "OMNIFLOW_EXP_MEMORY_INDEX": str(memory_index),
    }

    completed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--convert-ours-assets",
            "--tasks",
            "RecordWithName",
        ],
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
        "--authoring-manifest",
        str(authoring_manifest),
        "--output-root",
        str(output_root),
        "--memory-index",
        str(memory_index),
        "--task",
        "RecordWithName",
    ]


def test_one_task_run_adapts_all_methods_then_replays(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "assets"
    results = tmp_path / "results"
    memory_index = tmp_path / "memory" / "current.json"
    source_index = tmp_path / "memory" / "source_index.json"
    store_index = tmp_path / "memory" / "store_index.json"
    converted_root = assets / "converted"
    store_path = assets / "store.json"
    android_world = assets / "android_world"
    omnitransfer = assets / "OmniTransfer"
    env_file = assets / ".env"
    authoring_manifest = assets / "authoring-manifest.json"
    for path, content in (
        (memory_index, "{}"),
        (source_index, "{}"),
        (store_index, "{}"),
        (store_path, "{}"),
        (env_file, ""),
        (authoring_manifest, "{}"),
        (
            android_world
            / "android_world"
            / "env"
            / "setup_device"
            / "apps.py",
            "",
        ),
        (omnitransfer / "src" / "omnitransfer" / "runtime.py", ""),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    call_log = tmp_path / "calls.txt"
    converted_marker = tmp_path / "converted.marker"
    replayed_marker = tmp_path / "replayed.marker"
    mobilegpt_marker = tmp_path / "mobilegpt.marker"
    mobilegpt_install_marker = tmp_path / "mobilegpt-install.marker"
    appagent_marker = tmp_path / "appagent.marker"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        """#!/bin/sh
printf '%s\n' "$*" >> "$CALL_LOG"
if [ "$1" = "-m" ] && [ "$2" = "src.experiment.function_assets" ]; then
  : > "$CONVERTED_MARKER"
  exit 0
fi
if [ "$1" = "-m" ] && [ "$2" = "src.experiment.androidworld" ]; then
  if [ "$3" = "mobilegpt" ] && [ "$4" = "prepare-client" ]; then
    if [ "$ANDROID_SDK_ROOT" != "$EXPECTED_ANDROID_SDK_ROOT" ] || [ "$ANDROID_HOME" != "$EXPECTED_ANDROID_SDK_ROOT" ]; then
      exit 42
    fi
  fi
  : > "$REPLAYED_MARKER"
  exit 0
fi
if [ "$1" = "-m" ] && [ "$2" = "src.experiment.mobilegpt_source" ]; then
  if [ "$3" = "prepare" ]; then
    : > "$MOBILEGPT_MARKER"
    mkdir -p "$(dirname "$MOBILEGPT_MANIFEST")"
    : > "$MOBILEGPT_MANIFEST"
  fi
  exit 0
fi
if [ "$1" = "-m" ] && [ "$2" = "src.experiment.appagent_source" ]; then
  if [ "$3" = "prepare" ]; then
    : > "$APPAGENT_MARKER"
    mkdir -p "$(dirname "$APPAGENT_MANIFEST")"
    : > "$APPAGENT_MANIFEST"
  fi
  exit 0
fi
if [ "$1" = "-" ] && [ "$2" = "$REPO_PATH" ] && [ "$3" = "$MEMORY_INDEX" ]; then
  printf '%s\t%s\n' "$SOURCE_INDEX" "$STORE_INDEX"
  exit 0
fi
if [ "$1" = "-" ] && [ "$2" = "$STORE_INDEX" ]; then
  if [ -f "$CONVERTED_MARKER" ]; then
    printf '%s\n' "$STORE_PATH"
    exit 0
  fi
  exit 3
fi
if [ "$1" = "-" ] && [ "$2" = "$CONFIG_PATH" ]; then
  printf '%s\n' 'qwen3-vl-plus'
  exit 0
fi
exit 0
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    android_sdk_root = assets / "android-sdk"
    android_sdk_root.mkdir()
    fake_adb = fake_bin / "adb"
    fake_adb.write_text(
        """#!/bin/sh
if [ "$1" = "devices" ]; then
  printf 'List of devices attached\nemulator-5554\tdevice\nemulator-5560\tdevice\n'
elif [ "$1" = "-s" ] && [ "$3" = "shell" ] && [ "$4" = "pm" ]; then
  printf 'package:/data/app/mobilegpt.apk\n'
elif [ "$1" = "-s" ] && [ "$3" = "shell" ] && [ "$4" = "sha256sum" ]; then
  if [ -f "$MOBILEGPT_INSTALL_MARKER" ]; then
    printf '%s  %s\n' "$MOBILEGPT_APK_SHA" "$5"
  else
    printf '%064d  %s\n' 0 "$5"
  fi
elif [ "$1" = "-s" ] && [ "$3" = "install" ] && [ "$4" = "-r" ]; then
  printf '%s\n' 'Failure [INSTALL_FAILED_UPDATE_INCOMPATIBLE: signatures do not match]' >&2
  exit 1
elif [ "$1" = "-s" ] && [ "$3" = "uninstall" ]; then
  exit 0
elif [ "$1" = "-s" ] && [ "$3" = "install" ]; then
  : > "$MOBILEGPT_INSTALL_MARKER"
fi
exit 0
""",
        encoding="utf-8",
    )
    fake_adb.chmod(0o755)
    fake_java = fake_bin / "java"
    fake_java.write_text(
        "#!/bin/sh\nprintf '%s\\n' 'openjdk version \"17.0.19\"' >&2\n",
        encoding="utf-8",
    )
    fake_java.chmod(0o755)
    fake_jq = fake_bin / "jq"
    fake_jq.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_jq.chmod(0o755)
    mobilegpt_root = assets / "mobilegpt"
    appagent_root = assets / "appagent"
    for path in (
        mobilegpt_root / "Server" / "main.py",
        mobilegpt_root / "App" / "app" / "build" / "outputs" / "apk" / "debug"
        / "app-debug.apk",
        appagent_root / "README.md",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
        "PYTHON_BIN": str(fake_python),
        "CALL_LOG": str(call_log),
        "CONVERTED_MARKER": str(converted_marker),
        "REPLAYED_MARKER": str(replayed_marker),
        "MOBILEGPT_MARKER": str(mobilegpt_marker),
        "MOBILEGPT_APK_SHA": hashlib.sha256(b"").hexdigest(),
        "MOBILEGPT_INSTALL_MARKER": str(mobilegpt_install_marker),
        "APPAGENT_MARKER": str(appagent_marker),
        "MOBILEGPT_MANIFEST": str(
            assets / "mobilegpt-source" / "cold_memory_manifest.json"
        ),
        "APPAGENT_MANIFEST": str(
            assets / "appagent-source" / "appagent_demo_manifest.json"
        ),
        "REPO_PATH": str(REPO),
        "MEMORY_INDEX": str(memory_index),
        "SOURCE_INDEX": str(source_index),
        "STORE_INDEX": str(store_index),
        "STORE_PATH": str(store_path),
        "CONFIG_PATH": str(REPO / "config" / "paper_androidworld.json"),
        "EXPECTED_ANDROID_SDK_ROOT": str(android_sdk_root),
        "OMNIFLOW_EXP_ASSET_ROOT": str(assets),
        "OMNIFLOW_EXP_RESULTS_ROOT": str(results),
        "OMNIFLOW_EXP_MEMORY_INDEX": str(memory_index),
        "OMNIFLOW_ENV_FILE": str(env_file),
        "OMNIFLOW_ANDROID_WORLD_ROOT": str(android_world),
        "OMNIFLOW_ANDROID_SDK_ROOT": str(android_sdk_root),
        "OMNIFLOW_ADB_PATH": str(fake_adb),
        "OMNITRANSFER_ROOT": str(omnitransfer),
        "OMNIFLOW_OURS_CONVERTED_ASSET_ROOT": str(converted_root),
        "OMNIFLOW_OURS_AUTHORING_MANIFEST": str(authoring_manifest),
        "OMNIFLOW_MOBILEGPT_ROOT": str(mobilegpt_root),
        "OMNIFLOW_MOBILEGPT_SOURCE_MEMORY_ROOT": str(
            assets / "mobilegpt-source" / "memory"
        ),
        "OMNIFLOW_APPAGENT_ROOT": str(appagent_root),
        "OMNIFLOW_APPAGENT_DEMO_MEMORY_ROOT": str(
            assets / "appagent-source"
        ),
        "OMNIFLOW_SINGLE_TASK_TASK": "RecordWithName",
        "OMNIFLOW_SINGLE_TASK_METHODS": (
            "fixed_replay,ours,mobilegpt_offline_retrieval,"
            "appagent_demo,t3a_hint"
        ),
        "OMNIFLOW_SINGLE_TASK_DEVICE_TARGETS": (
            "small5554:emulator-5554:5554"
        ),
        "OMNIFLOW_SINGLE_TASK_MANAGE_EMULATORS": "0",
        "OMNIFLOW_SINGLE_TASK_FOLD_SERIAL": "",
        "OMNIFLOW_SOURCE_INDEX_EXPECTED_TASKS": "1",
    }

    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert converted_marker.is_file()
    assert mobilegpt_marker.is_file()
    assert mobilegpt_install_marker.is_file()
    assert appagent_marker.is_file()
    assert replayed_marker.is_file()
    calls = call_log.read_text(encoding="utf-8")
    assert calls.index("src.experiment.function_assets") < calls.index(
        "src.experiment.androidworld"
    )
    assert f"--store-index {store_index}" in calls

    repeated = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert repeated.returncode == 0, repeated.stderr
    repeated_calls = call_log.read_text(encoding="utf-8")
    assert repeated_calls.count("src.experiment.function_assets") == 1
    assert repeated_calls.count(
        "src.experiment.mobilegpt_source prepare"
    ) == 1
    assert repeated_calls.count("src.experiment.appagent_source prepare") == 1
    assert repeated_calls.count("src.experiment.androidworld one-task") == 2

    checked = subprocess.run(
        ["bash", str(SCRIPT), "--check-only"],
        cwd=REPO,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert checked.returncode == 0, checked.stderr
    checked_calls = call_log.read_text(encoding="utf-8")
    assert checked_calls.count(
        "src.experiment.androidworld mobilegpt audit-client"
    ) == 3
    assert checked_calls.count(
        "src.experiment.androidworld mobilegpt prepare-client"
    ) == 2


def test_task_major_completed_cells_skip_before_asset_generation(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "assets"
    results = tmp_path / "results"
    memory_index = tmp_path / "memory" / "current.json"
    source_index = tmp_path / "memory" / "source_index.json"
    store_index = tmp_path / "memory" / "store_index.json"
    for path in (memory_index, source_index, store_index):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    call_log = tmp_path / "calls.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        """#!/bin/sh
printf '%s\n' "$*" >> "$CALL_LOG"
if [ "$1" = "-" ] && [ "$2" = "$REPO_PATH" ] && [ "$3" = "$MEMORY_INDEX" ]; then
  printf '%s\t%s\n' "$SOURCE_INDEX" "$STORE_INDEX"
  exit 0
fi
if [ "$1" = "-" ] && [ "$2" = "$SOURCE_INDEX" ]; then
  printf '%s\n' 'AudioRecorderRecordAudio'
  exit 0
fi
if [ "$1" = "-" ] && [ "$2" = "$REPO_PATH" ]; then
  if [ "${PLAN_MODE:-complete}" = "missing_store" ]; then
    printf 'summary\t0\t1\npending\tours\tsmall5554\temulator-5554\t5554\n'
  else
    printf 'summary\t10\t0\n'
  fi
  exit 0
fi
if [ "$1" = "-" ] && [ "$2" = "$STORE_INDEX" ]; then
  exit 3
fi
exit 1
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = {
        **os.environ,
        "PYTHON_BIN": str(fake_python),
        "CALL_LOG": str(call_log),
        "REPO_PATH": str(REPO),
        "MEMORY_INDEX": str(memory_index),
        "SOURCE_INDEX": str(source_index),
        "STORE_INDEX": str(store_index),
        "OMNIFLOW_EXP_ASSET_ROOT": str(assets),
        "OMNIFLOW_EXP_RESULTS_ROOT": str(results),
        "OMNIFLOW_EXP_MEMORY_INDEX": str(memory_index),
        "OMNIFLOW_ENV_FILE": str(assets / ".env"),
        "OMNIFLOW_ANDROID_WORLD_ROOT": str(assets / "android_world"),
        "OMNITRANSFER_ROOT": str(assets / "OmniTransfer"),
        "OMNIFLOW_MOBILEGPT_SOURCE_MEMORY_ROOT": str(
            assets / "mobilegpt-source" / "memory"
        ),
        "OMNIFLOW_APPAGENT_DEMO_MEMORY_ROOT": str(
            assets / "appagent-source"
        ),
    }

    completed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--check-only",
            "--tasks",
            "AudioRecorderRecordAudio",
        ],
        cwd=REPO,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "already-complete" in completed.stdout
    calls = call_log.read_text(encoding="utf-8")
    assert "src.experiment.function_assets" not in calls
    assert any(
        line.startswith(f"- {REPO} ")
        and line.endswith(" 111 113")
        for line in calls.splitlines()
    )

    missing_store = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--check-only",
            "--tasks",
            "AudioRecorderRecordAudio",
        ],
        cwd=REPO,
        env={**environment, "PLAN_MODE": "missing_store"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert missing_store.returncode == 3
    assert (
        "Canonical Function asset missing for "
        "task=AudioRecorderRecordAudio"
    ) in missing_store.stderr


def test_memory_refresh_routes_all_evidence_through_the_only_script(
    tmp_path: Path,
) -> None:
    runlogs = tmp_path / "runlogs"
    results = tmp_path / "results"
    runlogs.mkdir()
    results.mkdir()
    source_index = tmp_path / "source-index.json"
    source_index.write_text("{}", encoding="utf-8")
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}", encoding="utf-8")
    memory_root = tmp_path / "memory"
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
        "OMNIFLOW_EXP_MEMORY_ROOT": str(memory_root),
        "OMNIFLOW_MASTER_SOURCE_INDEX": str(source_index),
        "OMNIFLOW_MEMORY_RUNLOG_ROOTS": str(runlogs),
        "OMNIFLOW_MEMORY_RESULT_ROOTS": str(results),
        "OMNIFLOW_MEMORY_FUNCTION_CATALOGS": str(catalog),
    }

    completed = subprocess.run(
        ["bash", str(SCRIPT), "--refresh-memory"],
        cwd=REPO,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert captured.read_text(encoding="utf-8").splitlines() == [
        "-m",
        "src.experiment.artifact_memory",
        "refresh",
        "--memory-root",
        str(memory_root),
        "--source-index",
        str(source_index),
        "--runlog-root",
        str(runlogs),
        "--result-root",
        str(results),
        "--function-catalog",
        str(catalog),
    ]
