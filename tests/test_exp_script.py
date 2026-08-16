from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from PIL import Image
from runlog_fixtures import androidworld_run_log

from src.experiment.artifact_memory import refresh_artifact_memory
from src.experiment.mobilegpt_contract import (
    MOBILEGPT_MEMORY_SCHEMA,
    MOBILEGPT_SOURCE_METHOD,
)
from src.experiment.preflight import (
    APPAGENT_REQUIRED_MODULES,
    REQUIRED_DISTRIBUTION_VERSIONS,
    _valid_appagent_demo_manifest,
)

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "exp" / "run_androidworld.sh"


def test_android_env_version_is_locked_and_preflight_enforced() -> None:
    assert REQUIRED_DISTRIBUTION_VERSIONS == {"android-env": "1.2.3"}
    assert APPAGENT_REQUIRED_MODULES == (
        "colorama",
        "cv2",
        "dashscope",
        "pyshine",
        "requests",
        "yaml",
    )
    assert '"android-env==1.2.3"' in (REPO / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    lock_text = (REPO / "uv.lock").read_text(encoding="utf-8")
    assert 'name = "android-env"\nversion = "1.2.3"' in lock_text
    assert '"android_world.registry"' in (
        REPO / "src" / "experiment" / "preflight.py"
    ).read_text(encoding="utf-8")


def test_preflight_accepts_offline_appagent_memory() -> None:
    assert _valid_appagent_demo_manifest(
        {
            "schema_version": "omniflow.appagent-demo-memory.v2",
            "official_appagent_revision": (
                "2c1900422caf6f9e94e96d5dd984b530e5a5fbf8"
            ),
            "source_seed": 111,
            "conversion_mode": "canonical_runlog_offline",
            "source_emulator_used": False,
            "native_memory_evidence": "/immutable/manifest.json",
            "teacher_complete": True,
            "teacher_action_count": 6,
            "teacher_actions_consumed": 6,
            "demo_action_count": 5,
            "source_episode_metrics": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "doc_generation_usage": {
                "model_calls": 5,
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
            },
            "prep_wall_sec": 0.1,
            "uses_omniflow_function": False,
            "target_inputs_read": False,
            "target_observations_read": False,
            "validator_state_read_for_memory": False,
        }
    )


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
    assert "--development-run" in completed.stdout
    assert "--stock-capture" in completed.stdout
    assert "--all-tasks" in completed.stdout
    assert "--eight-cells" in completed.stdout
    assert "--methods" in completed.stdout
    assert "--devices" in completed.stdout
    assert "--tasks" in completed.stdout
    assert "--convert-ours-assets" in completed.stdout
    assert "--convert-runlog-memory" in completed.stdout
    assert "--refresh-memory" in completed.stdout
    assert "--e2e-task" in completed.stdout
    assert "--source-backend" in completed.stdout
    assert "--source-runlog" in completed.stdout
    assert "--task-deadline-sec" in completed.stdout
    assert "OMNIFLOW_EXP_ASSET_ROOT" in completed.stdout
    assert "OMNIFLOW_EXP_MEMORY_ROOT" in completed.stdout
    assert "OMNIFLOW_OURS_AUTHORING_MANIFEST" in completed.stdout
    assert "OMNIFLOW_RUNLOG_MEMORY_OUTPUT_ROOT" in completed.stdout
    assert "OMNIFLOW_DEVELOPMENT_OUTPUT_PATH" in completed.stdout
    assert "OMNIFLOW_STOCK_CAPTURE_OUTPUT_PATH" in completed.stdout
    assert "OMNIFLOW_SINGLE_TASK_PERFORM_EMULATOR_SETUP" in completed.stdout
    assert "cold-restarted before every pending cell" in completed.stdout
    assert completed.stderr == ""
    script_text = SCRIPT.read_text(encoding="utf-8")
    assert 'workspace_root="$(cd "$repo/.." && pwd)"' in script_text
    assert 'default_asset_root="$workspace_root/OmniFlow"' in script_text
    assert "OMNIFLOW_APPAGENT_NATIVE_MEMORY_ROOTS" not in script_text
    assert (
        'default_memory_root="$workspace_root/assets/'
        'androidworld-experiment-memory-v1"' in script_text
    )
    assert 'omnitransfer_root="${OMNITRANSFER_ROOT:-$workspace_root/OmniTransfer}"' in (
        script_text
    )
    assert "ours_store_index_mechanical_asset" in script_text
    assert "androidworld_runlog_harvester_skill" in script_text
    assert f'mobilegpt_source_schema="{MOBILEGPT_MEMORY_SCHEMA}"' in script_text
    assert f'mobilegpt_source_method="{MOBILEGPT_SOURCE_METHOD}"' in script_text
    assert script_text.count(MOBILEGPT_MEMORY_SCHEMA) >= 2
    assert "MOBILEGPT_SOURCE_METHOD_BY_SCHEMA" in script_text
    assert "indexed_source_method = MOBILEGPT_SOURCE_METHOD_BY_SCHEMA.get(" in script_text
    assert "validate_mobilegpt_adapted_memory(" in script_text
    assert "omniflow.mobilegpt-runlog-offline-memory.v3" not in script_text
    assert "unset MOBILEGPT_MEMORY_ONLY" in script_text
    assert script_text.count('bash "$0"') == 2
    assert "-read-only" in script_text
    assert "-no-snapshot-load" in script_text
    assert "-no-snapshot-save" in script_text
    assert "from src.experiment.result_registry import registered_cell_plan" not in script_text
    assert script_text.count("registered_cell_plan_from_memory(") == 1
    assert "-m src.experiment.e2e_task_pipeline" in script_text
    assert '(( e2e_task_deadline_sec > 1800 ))' in script_text
    native_preflight = script_text.split(
        'if [[ "$profile" == "androidworld_native" ]]; then',
        maxsplit=1,
    )[1].split("\n  fi", maxsplit=1)[0]
    assert "--require-contacts-ready" in native_preflight
    assert 'if [[ "$task" == Contacts* ]]' in native_preflight
    assert (
        'if [[ "$profile" == "mobilegpt" && "$task" == Contacts* ]]'
        in script_text
    )
    assert 'while [[ -n "$(device_state "$serial")" ]] || grpc_ready "$grpc_port"; do' in script_text
    assert 'echo "[emulator] already stopped serial=$serial"' in script_text
    assert "No emulator process found while device remained visible" in script_text
    assert 'formal_step_timeout_sec=60' in script_text
    assert 'official_validator_flush_grace_sec=300' in script_text
    assert 'formal_episode_timeout_sec="$((formal_max_steps * formal_step_timeout_sec + official_validator_flush_grace_sec))"' in script_text
    assert 'timeout_sec="${OMNIFLOW_SINGLE_TASK_TIMEOUT_SEC:-$formal_episode_timeout_sec}"' in script_text


def test_runlog_memory_mode_does_not_treat_disabled_stock_capture_as_active(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    completed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--convert-runlog-memory",
            "mobilegpt_offline_retrieval",
            "--source-runlog",
            str(source),
        ],
        cwd=REPO,
        env={
            **os.environ,
            "OMNIFLOW_RUNLOG_MEMORY_OUTPUT_ROOT": str(tmp_path / "output"),
            "OMNIFLOW_ENV_FILE": str(tmp_path / "missing.env"),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "cannot be combined" not in completed.stderr
    assert "OMNIFLOW_ENV_FILE" in completed.stderr


def test_selected_model_profile_is_exported_for_native_openai_clients() -> None:
    script_text = SCRIPT.read_text(encoding="utf-8")

    assert 'export OPENAI_API_KEY="$selected_model_api_key"' in script_text
    assert 'export OPENAI_BASE_URL="$selected_model_base_url"' in script_text


def test_appagent_runlog_conversion_uses_offline_visual_document_model() -> None:
    script_text = SCRIPT.read_text(encoding="utf-8")

    assert (
        'appagent_document_model="${OMNIFLOW_APPAGENT_DOCUMENT_MODEL:-$formal_model}"'
        in script_text
    )
    assert 'runlog_memory_model="$appagent_document_model"' in script_text
    assert (
        'if [[ "$convert_runlog_memory_method" == "mobilegpt_offline_retrieval" ]]'
        in script_text
    )


def test_mobilegpt_runlog_conversion_uses_independent_embedding_model() -> None:
    script_text = SCRIPT.read_text(encoding="utf-8")

    assert (
        'mobilegpt_embedding_model="${OMNIFLOW_MOBILEGPT_EMBEDDING_MODEL:-text-embedding-v4}"'
        in script_text
    )
    assert 'runlog_memory_embedding_model="$mobilegpt_embedding_model"' in script_text
    assert 'runlog_memory_model="$formal_model"' in script_text


def test_mobilegpt_runtime_uses_sealed_embedding_contract_and_split_endpoints() -> None:
    script_text = SCRIPT.read_text(encoding="utf-8")
    runtime_text = (
        REPO / "src" / "integrations" / "mobilegpt_runtime.py"
    ).read_text(encoding="utf-8")

    assert 'mobilegpt_embedding_api_key="${OPENAI_API_KEY:-}"' in script_text
    assert 'export MOBILEGPT_CHAT_API_KEY="$selected_model_api_key"' in script_text
    assert (
        'export MOBILEGPT_EMBEDDING_API_KEY="$mobilegpt_embedding_api_key"'
        in script_text
    )
    assert "preflight-endpoints" in script_text
    assert "mobilegpt_embedding_dimension_mismatch" in runtime_text


def test_formal_dry_run_exits_before_output_and_emulator_management() -> None:
    script_text = SCRIPT.read_text(encoding="utf-8")
    dry_run_gate = script_text.index(
        'echo "[dry-run] ready task=$task methods=$methods devices=$device_targets; '
        'no device or persistent output created"'
    )

    assert dry_run_gate < script_text.index('mkdir -p "$preflight_output_root"')
    assert dry_run_gate < script_text.index(
        'for serial in "${target_serials[@]}"; do\n  ensure_emulator "$serial"'
    )
    assert 'command+=(--dry-run)' not in script_text


def test_androidworld_prefers_pinned_immutable_release_when_present() -> None:
    script_text = SCRIPT.read_text(encoding="utf-8")

    assert (
        'android_world_revision="38d1214d7df6cc7ec8503d912dad7aec814dd640"'
        in script_text
    )
    assert "OMNIFLOW_ANDROIDWORLD_RELEASE_ROOT" in script_text
    assert (
        'default_android_world_root="$android_world_release_root"'
        in script_text
    )


def test_unified_script_discovers_android_studio_jbr_on_macos() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert '/Applications/Android Studio.app/Contents/jbr/Contents/Home' in source
    assert '$account_root/Applications/Android Studio.app/Contents/jbr/Contents/Home' in source


def test_development_run_routes_through_the_only_script_without_repeated_setup(
    tmp_path: Path,
) -> None:
    android_world = tmp_path / "android-world"
    (android_world / "android_world").mkdir(parents=True)
    store = tmp_path / "store.json"
    store.write_text("{}", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=dashscope-key\n"
        "OPENAI_BASE_URL=https://dashscope.example/v1\n"
        "LLMTHU_KEY=llmthu-key\n"
        "LLMTHU_BASE_URL=https://llmapi.paratera.com/v1\n",
        encoding="utf-8",
    )
    adb = tmp_path / "adb"
    adb.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    adb.chmod(0o755)
    output = tmp_path / "development-run"

    completed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--development-run",
            "--dry-run",
            "--tasks",
            "ExpenseAddMultipleFromGallery",
        ],
        cwd=REPO,
        env={
            **os.environ,
            "PYTHON_BIN": sys.executable,
            "OMNIFLOW_ANDROID_WORLD_ROOT": str(android_world),
            "OMNIFLOW_ADB_PATH": str(adb),
            "OMNIFLOW_ENV_FILE": str(env_file),
            "OMNIFLOW_SINGLE_TASK_STORE_PATH": str(store),
            "OMNIFLOW_DEVELOPMENT_OUTPUT_PATH": str(output),
            "OMNIFLOW_DEVELOPMENT_MODEL": "GLM-5.1",
            "OMNIFLOW_SINGLE_TASK_PERFORM_EMULATOR_SETUP": "0",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "src.integrations.android_world.launch" in completed.stdout
    assert "ExpenseAddMultipleFromGallery" in completed.stdout
    assert "GLM-5.1" in completed.stdout
    assert "model_endpoint_profile=llmthu" in completed.stdout
    assert "model_endpoint=https://llmapi.paratera.com/v1" in completed.stdout
    assert "dashscope.example" not in completed.stdout
    assert "--perform-emulator-setup" not in completed.stdout
    assert not output.exists()


def test_development_run_rejects_qwen3_vl_plus_before_device_start(
    tmp_path: Path,
) -> None:
    android_world = tmp_path / "android-world"
    (android_world / "android_world").mkdir(parents=True)
    store = tmp_path / "store.json"
    store.write_text("{}", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLMTHU_KEY=llmthu-key\n"
        "LLMTHU_BASE_URL=https://llmapi.paratera.com/v1\n",
        encoding="utf-8",
    )
    adb = tmp_path / "adb"
    adb.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    adb.chmod(0o755)

    completed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--development-run",
            "--dry-run",
            "--tasks",
            "ExpenseAddMultipleFromGallery",
        ],
        cwd=REPO,
        env={
            **os.environ,
            "PYTHON_BIN": sys.executable,
            "OMNIFLOW_ANDROID_WORLD_ROOT": str(android_world),
            "OMNIFLOW_ADB_PATH": str(adb),
            "OMNIFLOW_ENV_FILE": str(env_file),
            "OMNIFLOW_SINGLE_TASK_STORE_PATH": str(store),
            "OMNIFLOW_DEVELOPMENT_OUTPUT_PATH": str(tmp_path / "attempt"),
            "OMNIFLOW_DEVELOPMENT_MODEL": "qwen3-vl-plus",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "qwen3-vl-plus is prohibited" in completed.stderr


def test_development_run_rejects_incomplete_code_release_before_device_start(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    release_script = release / "scripts" / "exp" / "run_androidworld.sh"
    release_script.parent.mkdir(parents=True)
    release_script.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    android_world = tmp_path / "android-world"
    (android_world / "android_world").mkdir(parents=True)
    store = tmp_path / "store.json"
    store.write_text("{}", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLMTHU_KEY=test-key\nLLMTHU_BASE_URL=https://llmapi.paratera.com/v1\n",
        encoding="utf-8",
    )
    adb = tmp_path / "adb"
    emulator = tmp_path / "emulator"
    for executable in (adb, emulator):
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)

    completed = subprocess.run(
        [
            "bash",
            str(release_script),
            "--development-run",
            "--dry-run",
            "--tasks",
            "MarkorCreateNote",
        ],
        cwd=release,
        env={
            **os.environ,
            "PYTHONPATH": str(REPO),
            "PYTHON_BIN": sys.executable,
            "OMNIFLOW_ANDROID_WORLD_ROOT": str(android_world),
            "OMNIFLOW_ADB_PATH": str(adb),
            "OMNIFLOW_EMULATOR_BIN": str(emulator),
            "OMNIFLOW_ENV_FILE": str(env_file),
            "OMNIFLOW_SINGLE_TASK_STORE_PATH": str(store),
            "OMNIFLOW_DEVELOPMENT_OUTPUT_PATH": str(tmp_path / "attempt"),
            "OMNIFLOW_DEVELOPMENT_MODEL": "GLM-5.1",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "Development runtime deployment incomplete before device startup" in (
        completed.stderr
    )
    assert "src/experiment/development_emulator.py" in completed.stderr


def test_experiment_script_prefers_existing_miniconda_base_python(
    tmp_path: Path,
) -> None:
    account_root = tmp_path / "account"
    base_python = account_root / "miniconda3" / "bin" / "python"
    base_python.parent.mkdir(parents=True)
    base_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    base_python.chmod(0o755)
    script_prefix = tmp_path / "script-prefix.sh"
    script_prefix.write_text(
        SCRIPT.read_text(encoding="utf-8").split("\nenv_file=", maxsplit=1)[0]
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            "bash",
            "-c",
            "source \"$SCRIPT_PREFIX\"; printf '%s\\n' \"$python_bin\"",
        ],
        cwd=REPO,
        env={
            **os.environ,
            "HOME": str(account_root),
            "SCRIPT_PREFIX": str(script_prefix),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == str(base_python)


def test_e2e_task_dispatches_through_the_only_shell_entry(tmp_path: Path) -> None:
    account_root = tmp_path / "account"
    assets = tmp_path / "assets"
    results = tmp_path / "results"
    memory = tmp_path / "memory"
    android_world = tmp_path / "AndroidWorld"
    mobilegpt = tmp_path / "MobileGPT"
    appagent = tmp_path / "AppAgent"
    sdk = tmp_path / "sdk"
    omnitransfer = account_root / "Projects" / "Omni" / "OmniTransfer"
    for directory in (
        assets,
        results,
        memory,
        android_world / "android_world",
        mobilegpt,
        appagent,
        sdk / "platform-tools",
        sdk / "emulator",
        omnitransfer,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    memory_index = memory / "current.json"
    memory_index.write_text("{}", encoding="utf-8")
    env_file = assets / ".env"
    env_file.write_text("LLMTHU_KEY=test-only\n", encoding="utf-8")
    capture = tmp_path / "invocation.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$CAPTURE\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    for executable in (sdk / "platform-tools" / "adb", sdk / "emulator" / "emulator"):
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)

    completed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--e2e-task",
            "BrowserDraw",
            "--source-backend",
            "reuse-only",
            "--task-deadline-sec",
            "1800",
            "--dry-run",
        ],
        cwd=REPO,
        env={
            **os.environ,
            "HOME": str(account_root),
            "CAPTURE": str(capture),
            "PYTHON_BIN": str(fake_python),
            "OMNIFLOW_EXP_ASSET_ROOT": str(assets),
            "OMNIFLOW_EXP_RESULTS_ROOT": str(results),
            "OMNIFLOW_EXP_MEMORY_ROOT": str(memory),
            "OMNIFLOW_EXP_MEMORY_INDEX": str(memory_index),
            "OMNIFLOW_ENV_FILE": str(env_file),
            "OMNIFLOW_ANDROID_WORLD_ROOT": str(android_world),
            "OMNIFLOW_ANDROID_SDK_ROOT": str(sdk),
            "OMNIFLOW_MOBILEGPT_ROOT": str(mobilegpt),
            "OMNIFLOW_APPAGENT_ROOT": str(appagent),
            "OMNITRANSFER_ROOT": str(omnitransfer),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    invocation = capture.read_text(encoding="utf-8").splitlines()
    assert invocation[:2] == ["-m", "src.experiment.e2e_task_pipeline"]
    assert invocation[invocation.index("--task") + 1] == "BrowserDraw"
    assert invocation[invocation.index("--source-backend") + 1] == "reuse-only"
    assert invocation[invocation.index("--task-deadline-sec") + 1] == "1800"
    assert invocation[-1] == "--dry-run"


@pytest.mark.parametrize(
    ("host_machine", "expected_abi"),
    [
        ("x86_64", "x86_64"),
        ("amd64", "x86_64"),
        ("arm64", "arm64-v8a"),
        ("aarch64", "arm64-v8a"),
    ],
)
def test_default_avd_system_image_matches_host_architecture(
    tmp_path: Path,
    host_machine: str,
    expected_abi: str,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uname = fake_bin / "uname"
    fake_uname.write_text(
        f"#!/bin/sh\nprintf '%s\\n' '{host_machine}'\n",
        encoding="utf-8",
    )
    fake_uname.chmod(0o755)
    script_prefix = tmp_path / "script-prefix.sh"
    script_prefix.write_text(
        SCRIPT.read_text(encoding="utf-8").split("\ndry_run=0\n", maxsplit=1)[0]
        + "\ndry_run=0\n",
        encoding="utf-8",
    )
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
        "SCRIPT_PREFIX": str(script_prefix),
    }

    completed = subprocess.run(
        [
            "bash",
            "-c",
            "source \"$SCRIPT_PREFIX\"; "
            "printf '%s\\n%s\\n' \"$emulator_avds\" \"$emulator_avd_specs\"",
        ],
        cwd=REPO,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.count(
        f"system-images;android-33;google_apis;{expected_abi}"
    ) == 2
    assert completed.stdout.count(
        f"system-images;android-34;google_apis;{expected_abi}"
    ) == 1
    assert "emulator-5554=OmniFlowTargetSmall" in completed.stdout
    assert "emulator-5560=SmallPhone" in completed.stdout
    assert "emulator-5564=OmniFlowTargetFold" in completed.stdout


def test_default_android_sdk_root_prefers_macos_standard_path(tmp_path: Path) -> None:
    macos_sdk = tmp_path / "Library" / "Android" / "sdk"
    macos_sdk.mkdir(parents=True)
    script_prefix = tmp_path / "script-prefix.sh"
    script_prefix.write_text(
        SCRIPT.read_text(encoding="utf-8").split("\ndry_run=0\n", maxsplit=1)[0]
        + "\ndry_run=0\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            "bash",
            "-c",
            "source \"$SCRIPT_PREFIX\"; resolve_default_android_sdk_root \"$TEST_ROOT\"",
        ],
        cwd=REPO,
        env={
            **os.environ,
            "SCRIPT_PREFIX": str(script_prefix),
            "TEST_ROOT": str(tmp_path),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == str(macos_sdk)


def test_default_topology_uses_three_distinct_device_instances(
    tmp_path: Path,
) -> None:
    script_prefix = tmp_path / "script-prefix.sh"
    script_prefix.write_text(
        SCRIPT.read_text(encoding="utf-8").split("\ndry_run=0\n", maxsplit=1)[0]
        + "\ndry_run=0\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            "bash",
            "-c",
            "source \"$SCRIPT_PREFIX\"; "
            "printf '%s\\n%s\\n%s\\n' "
            "\"$source_device\" \"$device_targets\" \"$emulator_avds\"",
        ],
        cwd=REPO,
        env={**os.environ, "SCRIPT_PREFIX": str(script_prefix)},
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        "source5560:emulator-5560:5560",
        "small5554:emulator-5554:5554,fold5564:emulator-5564:5564",
        (
                "emulator-5554=OmniFlowTargetSmall,emulator-5560=SmallPhone,"
            "emulator-5564=OmniFlowTargetFold"
        ),
    ]
    avd_names = [
        mapping.split("=", maxsplit=1)[1]
        for mapping in completed.stdout.splitlines()[2].split(",")
    ]
    assert len(avd_names) == len(set(avd_names)) == 3
    config = json.loads(
        (REPO / "config" / "paper_androidworld.json").read_text(encoding="utf-8")
    )
    assert config["one_task"]["source_device"] == completed.stdout.splitlines()[0]


@pytest.mark.parametrize(
    ("environment_override", "message"),
    [
        (
            {"OMNIFLOW_SOURCE_DEVICE": "source5560:emulator-5554:5554"},
            "Source serial must be separate from target serials",
        ),
        (
            {
                "OMNIFLOW_SINGLE_TASK_DEVICE_TARGETS": (
                    "small5554:emulator-5554:5554,"
                    "fold5564:emulator-5554:5554"
                )
            },
            "Duplicate target serial",
        ),
        (
            {
                "OMNIFLOW_SINGLE_TASK_DEVICE_TARGETS": (
                    "small5554:emulator-5554:5554,"
                    "fold5564:emulator-5564:5554"
                )
            },
            "Device target serial/console mismatch",
        ),
    ],
)
def test_experiment_topology_rejects_shared_device_identity(
    environment_override: dict[str, str],
    message: str,
) -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO,
        env={**os.environ, **environment_override},
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert message in completed.stderr


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            [
                "--methods",
                "mobilegpt_offline_retrieval,mobilegpt_offline_retrieval",
            ],
            "Duplicate method",
        ),
        (["--methods", "unknown_method"], "Unsupported paper method"),
        (["--devices", "small5554,small5554"], "Duplicate device"),
        (["--devices", "unknown_device"], "Unsupported formal device"),
    ],
)
def test_experiment_axes_reject_invalid_selections(
    arguments: list[str],
    message: str,
) -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT), *arguments],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert message in completed.stderr


def test_eight_cells_alias_requires_both_distinct_formal_devices() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT), "--eight-cells", "--devices", "small5554"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "--eight-cells requires exactly both formal target devices" in completed.stderr


def test_task_method_and_device_axes_are_independent(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    results = tmp_path / "results"
    memory_index = tmp_path / "memory" / "current.json"
    source_index = tmp_path / "memory" / "source_index.json"
    store_index = tmp_path / "memory" / "store_index.json"
    capture = tmp_path / "selection.txt"
    for path in (memory_index, source_index, store_index):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    fake_python = tmp_path / "python"
    fake_python.write_text(
        """#!/bin/sh
if [ "$1" = "-" ] && [ "$2" = "$REPO_PATH" ] && [ "$3" = "$MEMORY_INDEX" ]; then
  printf '%s\t%s\n' "$SOURCE_INDEX" "$STORE_INDEX"
  exit 0
fi
if [ "$1" = "-" ] && [ "$2" = "$SOURCE_INDEX" ]; then
  printf '%s\n' 'TaskA'
  exit 0
fi
if [ "$1" = "-" ] && [ "$2" = "$REPO_PATH" ]; then
  printf '%s|%s\n' "$6" "$7" > "$CAPTURE"
  printf 'summary\t1\t0\n'
  exit 0
fi
exit 1
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = {
        **os.environ,
        "PYTHON_BIN": str(fake_python),
        "REPO_PATH": str(REPO),
        "MEMORY_INDEX": str(memory_index),
        "SOURCE_INDEX": str(source_index),
        "STORE_INDEX": str(store_index),
        "CAPTURE": str(capture),
        "OMNIFLOW_EXP_ASSET_ROOT": str(assets),
        "OMNIFLOW_EXP_RESULTS_ROOT": str(results),
        "OMNIFLOW_EXP_MEMORY_INDEX": str(memory_index),
    }

    completed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--check-only",
            "--tasks",
            "TaskA",
            "--methods",
            "mobilegpt_offline_retrieval",
            "--devices",
            "fold5564",
        ],
        cwd=REPO,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert capture.read_text(encoding="utf-8").strip() == (
        "mobilegpt_offline_retrieval|fold5564:emulator-5564:5564"
    )
    assert "already-complete task=TaskA cells=1/1" in completed.stdout


def test_check_only_is_read_only_before_any_runtime_output(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    env_file = assets / ".env"
    env_file.write_text("", encoding="utf-8")
    screenshot = assets / "state-0.png"
    Image.new("RGB", (8, 6), color="blue").save(screenshot)
    run_log = androidworld_run_log(
        [{"action_type": "click", "x": 500, "y": 500}],
        task_name="SystemBluetoothTurnOn",
        run_id="source",
        goal="Turn Bluetooth on.",
    )
    run_log["steps"][0]["observation"]["pixels"] = {
        "path": str(screenshot.resolve()),
        "sha256": hashlib.sha256(screenshot.read_bytes()).hexdigest(),
        "width": 8,
        "height": 6,
        "mime_type": "image/png",
    }
    source_run_log = assets / "source.run_log.json"
    source_run_log.write_text(
        json.dumps(run_log),
        encoding="utf-8",
    )
    source_index = assets / "index.json"
    source_index.write_text(
        json.dumps(
            {
                "SystemBluetoothTurnOn": {
                    "goal": "Turn Bluetooth on.",
                    "params": {"on_or_off": "on"},
                    "replay_seed": 3936510006,
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
    assert "protocol_seed=111 recorded_seed=3936510006" in completed.stdout
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


def test_asset_revision_routes_reason_through_the_only_script(
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

    completed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--convert-ours-assets",
            "--tasks",
            "RecordWithName",
        ],
        cwd=REPO,
        env={
            **os.environ,
            "PYTHON_BIN": str(fake_python),
            "CAPTURE_ARGS": str(captured),
            "OMNIFLOW_OURS_SOURCE_ASSET_INDEX": str(source_index),
            "OMNIFLOW_OURS_CONVERTED_ASSET_ROOT": str(output_root),
            "OMNIFLOW_OURS_AUTHORING_MANIFEST": str(authoring_manifest),
            "OMNIFLOW_OURS_REVISION_REASON": "source qualification failed",
            "OMNIFLOW_EXP_MEMORY_INDEX": str(memory_index),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    arguments = captured.read_text(encoding="utf-8").splitlines()
    reason_index = arguments.index("--revision-reason") + 1
    assert arguments[reason_index] == "source qualification failed"


def test_asset_conversion_defaults_to_canonical_memory_source_index(
    tmp_path: Path,
) -> None:
    archived_source_index = tmp_path / "archive" / "source-index.json"
    canonical_source_index = tmp_path / "memory" / "source-index.json"
    store_index = tmp_path / "memory" / "store-index.json"
    memory_index = tmp_path / "memory" / "current.json"
    output_root = tmp_path / "converted"
    authoring_manifest = tmp_path / "authoring-manifest.json"
    captured = tmp_path / "python-args.txt"
    for path in (
        archived_source_index,
        canonical_source_index,
        store_index,
        memory_index,
        authoring_manifest,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    fake_python = tmp_path / "python"
    fake_python.write_text(
        """#!/bin/sh
if [ "$1" = "-" ] && [ "$2" = "$REPO_PATH" ] && [ "$3" = "$MEMORY_INDEX" ]; then
  printf '%s\t%s\n' "$CANONICAL_SOURCE_INDEX" "$STORE_INDEX"
  exit 0
fi
printf '%s\n' "$@" > "$CAPTURE_ARGS"
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    completed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--convert-ours-assets",
            "--tasks",
            "RecordWithName",
        ],
        cwd=REPO,
        env={
            **os.environ,
            "PYTHON_BIN": str(fake_python),
            "REPO_PATH": str(REPO),
            "MEMORY_INDEX": str(memory_index),
            "CANONICAL_SOURCE_INDEX": str(canonical_source_index),
            "STORE_INDEX": str(store_index),
            "CAPTURE_ARGS": str(captured),
            "OMNIFLOW_MASTER_SOURCE_INDEX": str(archived_source_index),
            "OMNIFLOW_OURS_CONVERTED_ASSET_ROOT": str(output_root),
            "OMNIFLOW_OURS_AUTHORING_MANIFEST": str(authoring_manifest),
            "OMNIFLOW_EXP_MEMORY_INDEX": str(memory_index),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    arguments = captured.read_text(encoding="utf-8").splitlines()
    source_index_argument = arguments.index("--source-asset-index") + 1
    assert arguments[source_index_argument] == str(canonical_source_index)


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
            (
                env_file,
                "LLMTHU_KEY=test-only\n"
                "LLMTHU_BASE_URL=https://llmapi.paratera.com/v1\n"
                "OPENAI_API_KEY=embedding-test-only\n"
                "OPENAI_BASE_URL=https://embedding.example/v1\n",
        ),
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
if [ "$1" = "-" ] && [ "$2" = "llmthu" ]; then
  printf '%s\n' 'https://llmapi.paratera.com/v1'
  exit 0
fi
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
  printf '%s\n' 'GLM-5.1'
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
            assets / "mobilegpt-source" / "mobilegpt_memory_manifest.json"
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
    assert not mobilegpt_install_marker.exists()
    assert appagent_marker.is_file()
    assert replayed_marker.is_file()
    calls = call_log.read_text(encoding="utf-8")
    assert calls.index("src.experiment.function_assets") < calls.index(
        "src.experiment.androidworld"
    )
    assert f"--store-index {store_index}" in calls
    one_task_calls = [
        line
        for line in calls.splitlines()
        if "src.experiment.androidworld one-task" in line
    ]
    assert one_task_calls
    assert all(f"--adb-path {fake_adb}" in line for line in one_task_calls)
    mobilegpt_source_calls = [
        line
        for line in calls.splitlines()
        if "src.experiment.mobilegpt_source" in line
    ]
    assert mobilegpt_source_calls
    assert all(
        "--store-index" not in line
        for line in mobilegpt_source_calls
    )
    assert calls.index(
        "src.experiment.mobilegpt_source prepare"
    ) < calls.index(
        "src.integrations.mobilegpt_runtime preflight-endpoints"
    )

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
    assert "src.experiment.androidworld mobilegpt audit-client" not in checked_calls
    assert "src.experiment.androidworld mobilegpt prepare-client" not in checked_calls


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
        "OMNIFLOW_BATCH_ATTEMPT_ID": "iteration_01-resume-test",
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
        and " 111 113 20 iteration_01-resume-test" in line
        for line in calls.splitlines()
    )

    for variable, value in (
        ("OMNIFLOW_SINGLE_TASK_SOURCE_SEED", "112"),
        ("OMNIFLOW_SINGLE_TASK_EVALUATION_SEED", "114"),
        ("OMNIFLOW_SINGLE_TASK_MAX_STEPS", "30"),
        ("OMNIFLOW_SINGLE_TASK_MAX_FALLBACK_STEPS", "4"),
        ("OMNIFLOW_SINGLE_TASK_FIXED_TASK_PARAMS", "1"),
        ("OMNIFLOW_SINGLE_TASK_FOLD_STATE", "1"),
        ("OMNIFLOW_SINGLE_TASK_FOLD_SIZE", "1768x2208"),
    ):
        rejected = subprocess.run(
            [
                "bash",
                str(SCRIPT),
                "--check-only",
                "--tasks",
                "AudioRecorderRecordAudio",
            ],
            cwd=REPO,
            env={**environment, variable: value},
            check=False,
            capture_output=True,
            text=True,
        )
        assert rejected.returncode == 2
        assert "--all-tasks requires the frozen formal protocol" in rejected.stderr

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

    assert missing_store.returncode == 1
    assert (
        "Canonical Function asset missing for "
        "task=AudioRecorderRecordAudio"
    ) in missing_store.stderr
    assert "[batch:static] incomplete terminal=1 pending=1 total=10" in (
        missing_store.stderr
    )


def test_task_major_missing_function_asset_does_not_block_later_tasks(
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
    planned_tasks = tmp_path / "planned-tasks.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        """#!/bin/sh
if [ "$1" = "-" ] && [ "$2" = "llmthu" ]; then
  printf '%s\n' 'https://llmapi.paratera.com/v1'
  exit 0
fi
if [ "$1" = "-" ] && [ "$2" = "$REPO_PATH" ] && [ "$3" = "$MEMORY_INDEX" ]; then
  printf '%s\t%s\n' "$SOURCE_INDEX" "$STORE_INDEX"
  exit 0
fi
if [ "$1" = "-" ] && [ "$2" = "$SOURCE_INDEX" ]; then
  printf '%s\n' 'TaskMissing' 'TaskComplete'
  exit 0
fi
if [ "$1" = "-" ] && [ "$2" = "$REPO_PATH" ]; then
  task="$5"
  printf '%s\n' "$task" >> "$PLANNED_TASKS"
  if [ "$task" = "TaskMissing" ]; then
    printf 'summary\t0\t2\npending\tours\tsmall5554\temulator-5554\t5554\npending\tours\tfold5564\temulator-5564\t5564\n'
  else
    printf 'summary\t2\t0\n'
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
        "REPO_PATH": str(REPO),
        "MEMORY_INDEX": str(memory_index),
        "SOURCE_INDEX": str(source_index),
        "STORE_INDEX": str(store_index),
        "PLANNED_TASKS": str(planned_tasks),
        "OMNIFLOW_EXP_ASSET_ROOT": str(assets),
        "OMNIFLOW_EXP_RESULTS_ROOT": str(results),
        "OMNIFLOW_EXP_MEMORY_INDEX": str(memory_index),
        "OMNIFLOW_BATCH_ATTEMPT_ID": "batch-expansion-test",
        "OMNIFLOW_ENV_FILE": str(assets / ".env"),
        "OMNIFLOW_ANDROID_WORLD_ROOT": str(assets / "android_world"),
        "OMNITRANSFER_ROOT": str(assets / "OmniTransfer"),
    }

    completed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--check-only",
            "--methods",
            "ours",
            "--tasks",
            "TaskMissing,TaskComplete",
        ],
        cwd=REPO,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert planned_tasks.read_text(encoding="utf-8").splitlines() == [
        "TaskMissing",
        "TaskComplete",
    ]
    assert "terminal task=TaskMissing method=ours pending=2" in completed.stdout
    assert "already-complete task=TaskComplete cells=2/2" in completed.stdout
    assert "incomplete terminal=2 pending=2 total=4" in completed.stderr


@pytest.mark.parametrize("terminal_phase", ["static", "runtime"])
def test_task_major_terminal_source_failure_continues_later_cells(
    tmp_path: Path,
    terminal_phase: str,
) -> None:
    assets = tmp_path / "assets"
    results = tmp_path / "results"
    memory_index = tmp_path / "memory" / "current.json"
    source_index = tmp_path / "memory" / "source_index.json"
    store_index = tmp_path / "memory" / "store_index.json"
    env_file = assets / ".env"
    android_world = assets / "android_world"
    mobilegpt_root = assets / "mobilegpt"
    mobilegpt_bundle = assets / "mobilegpt-source" / "BrowserDraw"
    for path, content in (
        (memory_index, "{}"),
        (source_index, "{}"),
        (store_index, "{}"),
            (
                env_file,
                "LLMTHU_KEY=test-only\n"
                "LLMTHU_BASE_URL=https://llmapi.paratera.com/v1\n"
                "OPENAI_API_KEY=embedding-test-only\n"
                "OPENAI_BASE_URL=https://embedding.example/v1\n",
        ),
        (
            android_world
            / "android_world"
            / "env"
            / "setup_device"
            / "apps.py",
            "",
        ),
        (mobilegpt_root / "Server" / "main.py", ""),
        (
            mobilegpt_root
            / "App"
            / "app"
            / "build"
            / "outputs"
            / "apk"
            / "debug"
            / "app-debug.apk",
            "",
        ),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    fake_python = tmp_path / "python"
    fake_python.write_text(
        """#!/bin/sh
if [ "$1" = "-" ] && [ "$2" = "llmthu" ]; then
  printf '%s\n' 'https://llmapi.paratera.com/v1'
  exit 0
fi
if [ "$1" = "-" ] && [ "$2" = "$REPO_PATH" ] && [ "$3" = "$MEMORY_INDEX" ]; then
  printf '%s\t%s\n' "$SOURCE_INDEX" "$STORE_INDEX"
  exit 0
fi
if [ "$1" = "-" ] && [ "$2" = "$SOURCE_INDEX" ] && [ "$#" -eq 3 ]; then
  printf '%s\n' 'BrowserDraw'
  exit 0
fi
if [ "$1" = "-" ] && [ "$2" = "$REPO_PATH" ] && { [ "$5" = "$SOURCE_INDEX" ] || [ "$5" = "$STORE_INDEX" ]; }; then
  if [ "$4" = "mobilegpt_memory_manifest.json" ]; then
    if [ "$5" != "$SOURCE_INDEX" ]; then
      exit 44
    fi
    if [ "$TERMINAL_PHASE" = "static" ] || [ -f "$STATE_DIR/mobilegpt-terminal" ]; then
      printf '%s\n' 'source_asset_retry_forbidden:/immutable/mobilegpt:official_source_failed' >&2
      exit 75
    fi
    printf '%s\n' "$MOBILEGPT_BUNDLE"
    exit 0
  fi
fi
if [ "$1" = "-" ] && [ "$2" = "$REPO_PATH" ]; then
  completed=6
  pending=4
  if [ -f "$STATE_DIR/outcome-mobilegpt_offline_retrieval-small5554" ]; then
    completed=$((completed + 1))
    pending=$((pending - 1))
  fi
  if [ -f "$STATE_DIR/outcome-mobilegpt_offline_retrieval-fold5564" ]; then
    completed=$((completed + 1))
    pending=$((pending - 1))
  fi
  if [ -f "$STATE_DIR/t3a_hint-small5554" ]; then
    completed=$((completed + 1))
    pending=$((pending - 1))
  fi
  if [ -f "$STATE_DIR/t3a_hint-fold5564" ]; then
    completed=$((completed + 1))
    pending=$((pending - 1))
  fi
  printf 'summary\t%s\t%s\n' "$completed" "$pending"
  if [ ! -f "$STATE_DIR/outcome-mobilegpt_offline_retrieval-small5554" ]; then
    printf 'pending\tmobilegpt_offline_retrieval\tsmall5554\temulator-5554\t5554\n'
  fi
  if [ ! -f "$STATE_DIR/outcome-mobilegpt_offline_retrieval-fold5564" ]; then
    printf 'pending\tmobilegpt_offline_retrieval\tfold5564\temulator-5564\t5564\n'
  fi
  if [ ! -f "$STATE_DIR/t3a_hint-small5554" ]; then
    printf 'pending\tt3a_hint\tsmall5554\temulator-5554\t5554\n'
  fi
  if [ ! -f "$STATE_DIR/t3a_hint-fold5564" ]; then
    printf 'pending\tt3a_hint\tfold5564\temulator-5564\t5564\n'
  fi
  exit 0
fi
if [ "$1" = "-" ] && [ "$2" = "$CONFIG_PATH" ]; then
  printf '%s\n' 'GLM-5.1'
  exit 0
fi
if [ "$1" = "-m" ] && [ "$2" = "src.experiment.androidworld" ] && [ "$3" = "one-task" ]; then
  device="${OMNIFLOW_SINGLE_TASK_DEVICE_TARGETS%%:*}"
  : > "$STATE_DIR/${OMNIFLOW_SINGLE_TASK_METHODS}-${device}"
  exit 0
fi
if [ "$1" = "-m" ] && [ "$2" = "src.experiment.mobilegpt_source" ] && [ "$3" = "prepare" ]; then
  mkdir -p "$MOBILEGPT_BUNDLE"
  printf '%s\n' '{"retry_allowed":false}' > "$MOBILEGPT_BUNDLE/prep_failure.json"
  : > "$STATE_DIR/mobilegpt-terminal"
  exit 9
fi
if [ "$1" = "-m" ] && [ "$2" = "src.experiment.batch_outcomes" ]; then
  command="$3"
  shift 3
  if [ "$command" = "record" ]; then
    method=""
    device=""
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --method) method="$2"; shift 2 ;;
        --device) device="$2"; shift 2 ;;
        *) shift ;;
      esac
    done
    : > "$STATE_DIR/outcome-${method}-${device}"
    printf '%s\n' "$STATE_DIR/outcome-${method}-${device}"
    exit 0
  fi
  if [ "$command" = "report" ]; then
    : > "$STATE_DIR/batch-report"
    printf '%s\n' '{"counts":{"pending":0}}'
    exit 0
  fi
fi
exit 0
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_adb = fake_bin / "adb"
    fake_adb.write_text(
        """#!/bin/sh
if [ "$1" = "devices" ]; then
  printf 'List of devices attached\nemulator-5554\tdevice\nemulator-5560\tdevice\nemulator-5564\tdevice\n'
elif [ "$1" = "-s" ] && [ "$3" = "shell" ] && [ "$4" = "pm" ]; then
  printf 'package:/data/app/mobilegpt.apk\n'
elif [ "$1" = "-s" ] && [ "$3" = "shell" ] && [ "$4" = "sha256sum" ]; then
  printf '%s  %s\n' "$MOBILEGPT_APK_SHA" "$5"
elif [ "$1" = "-s" ] && [ "$2" = "emulator-5564" ] && [ "$6" = "print-state" ]; then
  printf '2\n'
elif [ "$1" = "-s" ] && [ "$2" = "emulator-5564" ] && [ "$4" = "wm" ]; then
  printf 'Physical size: 2208x1840\n'
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
    android_sdk_root = assets / "android-sdk"
    android_sdk_root.mkdir()
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
        "PYTHON_BIN": str(fake_python),
        "STATE_DIR": str(state_dir),
        "REPO_PATH": str(REPO),
        "MEMORY_INDEX": str(memory_index),
        "SOURCE_INDEX": str(source_index),
        "STORE_INDEX": str(store_index),
        "MOBILEGPT_BUNDLE": str(mobilegpt_bundle),
        "MOBILEGPT_APK_SHA": hashlib.sha256(b"").hexdigest(),
        "TERMINAL_PHASE": terminal_phase,
        "CONFIG_PATH": str(REPO / "config" / "paper_androidworld.json"),
        "OMNIFLOW_EXP_ASSET_ROOT": str(assets),
        "OMNIFLOW_EXP_RESULTS_ROOT": str(results),
        "OMNIFLOW_EXP_MEMORY_INDEX": str(memory_index),
        "OMNIFLOW_ENV_FILE": str(env_file),
        "OMNIFLOW_ANDROID_WORLD_ROOT": str(android_world),
        "OMNIFLOW_ANDROID_SDK_ROOT": str(android_sdk_root),
        "OMNIFLOW_ADB_PATH": str(fake_adb),
        "OMNIFLOW_JAVA_HOME": str(tmp_path),
        "OMNITRANSFER_ROOT": str(assets / "OmniTransfer"),
        "OMNIFLOW_MOBILEGPT_ROOT": str(mobilegpt_root),
        "OMNIFLOW_SINGLE_TASK_MANAGE_EMULATORS": "0",
    }

    completed = subprocess.run(
        ["bash", str(SCRIPT), "--tasks", "BrowserDraw"],
        cwd=REPO,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (state_dir / "t3a_hint-small5554").is_file()
    assert (state_dir / "t3a_hint-fold5564").is_file()
    assert not list(state_dir.glob("mobilegpt_offline_retrieval-*"))
    assert (state_dir / "outcome-mobilegpt_offline_retrieval-small5554").is_file()
    assert (state_dir / "outcome-mobilegpt_offline_retrieval-fold5564").is_file()
    assert (state_dir / "batch-report").is_file()
    terminal_prefix = (
        "[batch:static]" if terminal_phase == "static" else "[batch]"
    )
    assert (
        f"{terminal_prefix} terminal task=BrowserDraw "
        "method=mobilegpt_offline_retrieval pending=2"
    ) in completed.stdout
    assert (
        "[batch] complete completed=2 skipped=6 failed=2 total=10"
    ) in completed.stdout


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
        "OMNIFLOW_EXP_RESULTS_ROOT": str(results),
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

    environment.pop("OMNIFLOW_MEMORY_FUNCTION_CATALOGS")
    without_catalog = subprocess.run(
        ["bash", str(SCRIPT), "--refresh-memory"],
        cwd=REPO,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert without_catalog.returncode == 0, without_catalog.stderr
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
    ]

    memory_root.mkdir()
    (memory_root / "current.json").write_text("{}", encoding="utf-8")
    source_index.unlink()
    from_existing_memory = subprocess.run(
        ["bash", str(SCRIPT), "--refresh-memory"],
        cwd=REPO,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert from_existing_memory.returncode == 0, from_existing_memory.stderr
    assert captured.read_text(encoding="utf-8").splitlines() == [
        "-m",
        "src.experiment.artifact_memory",
        "refresh",
        "--memory-root",
        str(memory_root),
        "--runlog-root",
        str(runlogs),
        "--result-root",
        str(results),
    ]
