from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest
from runlog_fixtures import androidworld_run_log

from src.experiment.data_index import refresh_artifact_index
from src.experiment.mobilegpt_contract import (
    MOBILEGPT_MEMORY_SCHEMA,
    MOBILEGPT_SOURCE_METHOD,
)
from src.experiment.checks import (
    APPAGENT_REQUIRED_MODULES,
    REQUIRED_DISTRIBUTION_VERSIONS,
    _valid_appagent_manifest,
)
from src.experiment.protocol import DROIDRUN_VERSION

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
    assert f'name = "droidrun"\nversion = "{DROIDRUN_VERSION}"' in lock_text
    assert '"android_world.registry"' in (
        REPO / "src" / "experiment" / "checks.py"
    ).read_text(encoding="utf-8")


def test_bmoca_skilldroid_uses_pinned_droidrun_runtime() -> None:
    script_text = SCRIPT.read_text(encoding="utf-8")
    assert "from src.experiment.protocol import DROIDRUN_VERSION" in script_text
    assert 'version("droidrun")' in script_text
    assert "load_official_droidrun_macro_player" in script_text
    assert '[[ -z "$selected_method_arg" || "$selected_method_arg" == "skilldroid_replay" ]]' in script_text
    assert 'uv sync --extra bmoca' in script_text
    assert "B-MoCA campaign requires the official env100 source AVD" in script_text


def test_preflight_accepts_offline_appagent_memory() -> None:
    assert _valid_appagent_manifest(
        {
            "schema_version": "omniflow.appagent.memory.v3",
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
    assert "--environment" in completed.stdout
    assert "OMNIFLOW_BMOCA_CORPUS_MANIFEST" in completed.stdout
    assert "OMNIFLOW_BMOCA_WORKERS" not in completed.stdout
    assert "OMNIFLOW_BMOCA_ENVIRONMENT_RETRIES" not in completed.stdout
    assert "--development-run" in completed.stdout
    assert "--control-backend" in completed.stdout
    assert "--all-tasks" in completed.stdout
    assert "--method" in completed.stdout
    assert "--device" in completed.stdout
    assert "--methods" not in completed.stdout
    assert "--devices" not in completed.stdout
    assert "--tasks" in completed.stdout
    assert "--convert-omniflow-assets" not in completed.stdout
    assert "--convert-runlog-memory" not in completed.stdout
    assert "--convert-source-runlogs" not in completed.stdout
    assert "--prepare-mobilegpt-memory" not in completed.stdout
    assert "--refresh-memory" in completed.stdout
    assert "--e2e-task" in completed.stdout
    assert "--source-backend" not in completed.stdout
    assert "--page-store" not in completed.stdout
    assert "--mobilegpt-native-cold-warm" not in completed.stdout
    assert "--source-runlog" not in completed.stdout
    assert "--task-deadline-sec" in completed.stdout
    assert "OMNIFLOW_EXP_ASSET_ROOT" in completed.stdout
    assert "OMNIFLOW_EXP_MEMORY_ROOT" in completed.stdout
    assert "AUTHORING_MANIFEST" not in completed.stdout
    assert "OMNIFLOW_RUNLOG_MEMORY_OUTPUT_ROOT" not in completed.stdout
    assert "OMNIFLOW_SOURCE_SELECTION_MANIFEST" not in completed.stdout
    assert "OMNIFLOW_FUNCTION_STORE_SELECTION_MANIFEST" not in completed.stdout
    assert "OMNIFLOW_DEVELOPMENT_OUTPUT_PATH" in completed.stdout
    assert "OMNIFLOW_ANDROIDWORLD_PERFORM_EMULATOR_SETUP" in completed.stdout
    assert "cold-restarted before every pending result" in completed.stdout
    assert completed.stderr == ""
    script_text = SCRIPT.read_text(encoding="utf-8")
    assert 'workspace_root="$(cd "$repo/.." && pwd)"' in script_text
    assert 'default_asset_root="$repo/data"' in script_text
    assert 'default_python="$repo/.venv/bin/python"' in script_text
    assert "miniconda3/envs/omniflow-py31113" not in script_text
    assert 'python_bin="${PYTHON_BIN:-$default_python}"' in script_text
    assert "validate_page_encoder_runtime" in script_text
    assert "OMNIFLOW_APPAGENT_NATIVE_MEMORY_ROOTS" not in script_text
    assert 'default_memory_root="$repo/data"' in script_text
    assert "validate_model_endpoint_auth" in script_text
    assert 'f\"{base_url}/models\"' in script_text
    assert 'omnitransfer_root="${OMNITRANSFER_ROOT:-$workspace_root/OmniTransfer}"' in (
        script_text
    )
    assert "provenance_path" not in script_text
    assert "androidworld_runlog_harvester_skill" not in script_text
    assert f'mobilegpt_source_schema="{MOBILEGPT_MEMORY_SCHEMA}"' in script_text
    assert f'mobilegpt_source_method="{MOBILEGPT_SOURCE_METHOD}"' in script_text
    assert script_text.count(MOBILEGPT_MEMORY_SCHEMA) >= 2
    assert "MOBILEGPT_SOURCE_METHOD_BY_SCHEMA" in script_text
    assert "indexed_source_method = MOBILEGPT_SOURCE_METHOD_BY_SCHEMA.get(" in script_text
    assert "validate_mobilegpt_adapted_memory(" in script_text
    assert "omniflow.mobilegpt-runlog-offline-memory.v3" not in script_text
    assert "unset MOBILEGPT_MEMORY_ONLY" in script_text
    assert script_text.count('bash "$0"') == 1
    assert "-read-only" in script_text
    assert "-no-snapshot-load" in script_text
    assert "-no-snapshot-save" in script_text
    assert "from src.experiment.result_registry import registered_result_plan" not in script_text
    assert "--master-progress-root" not in script_text
    assert script_text.count("registered_result_plan_from_memory(") == 0
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
    assert "src.experiment.protocol" in script_text
    assert "protocol_values=" in script_text
    assert 'PYTHONPATH="$repo:$repo/src${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" -' in script_text
    assert 'protocol_values="$("$python_bin" -' in script_text
    assert "python3 - <<'PY'" not in script_text
    assert 'timeout_sec="$((max_steps * formal_step_timeout_sec + official_validator_flush_grace_sec))"' in script_text
    assert "formal_cell_timeout_sec" not in script_text


def test_selected_model_profile_is_exported_for_native_openai_clients() -> None:
    script_text = SCRIPT.read_text(encoding="utf-8")

    assert 'export OPENAI_API_KEY="$selected_model_api_key"' in script_text
    assert 'export OPENAI_BASE_URL="$selected_model_base_url"' in script_text


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
        'echo "[dry-run] ready task=$task method=$method device=$device_target; '
        'no device or persistent output created"'
    )

    assert dry_run_gate < script_text.index('mkdir -p "$preflight_output_root"')
    assert dry_run_gate < script_text.index(
        'for serial in "${target_serials[@]}"; do\n  ensure_emulator "$serial"'
    )
    assert 'command+=(--dry-run)' not in script_text


def test_task_filter_handles_empty_batch_before_first_task() -> None:
    script_text = SCRIPT.read_text(encoding="utf-8")

    assert (
        'if ((${#batch_tasks[@]} > 0)); then\n'
        '        for selected_task in "${batch_tasks[@]}"; do'
    ) in script_text


def test_androidworld_defaults_to_pinned_immutable_release_without_fallback(
    tmp_path: Path,
) -> None:
    script_text = SCRIPT.read_text(encoding="utf-8")

    protocol = json.loads(
        (REPO / "config" / "paper_androidworld.json").read_text(encoding="utf-8")
    )["protocol"]
    assert "android_world_revision=\"$(" in script_text
    assert protocol["androidworld_revision"] == "632ac95959ace58c8e2ed2db8e4209cc3d9c26ef"
    asset_root = tmp_path / "OmniFlow"
    dirty_fallback = (
        asset_root / "runtime" / "external" / "droidrun-android-world" / "android_world"
    )
    dirty_fallback.mkdir(parents=True)
    environment = os.environ.copy()
    environment["OMNIFLOW_EXP_ASSET_ROOT"] = str(asset_root)
    environment.pop("OMNIFLOW_ANDROIDWORLD_RELEASE_ROOT", None)
    environment.pop("OMNIFLOW_ANDROID_WORLD_ROOT", None)

    completed = subprocess.run(
        ["bash", "-x", str(SCRIPT), "--help"],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    expected_release = (
        REPO.parent
        / "releases"
        / "android-world-632ac95959ace58c8e2ed2db8e4209cc3d9c26ef"
    )
    assert completed.returncode == 0, completed.stderr
    assert f"+ android_world_root={expected_release}" in completed.stderr
    assert f"+ android_world_root={dirty_fallback}" not in completed.stderr


def test_unified_script_repairs_missing_androidworld_sqlite_fts4_support() -> None:
    script_text = SCRIPT.read_text(encoding="utf-8")

    assert "ensure_androidworld_sqlite_fts4()" in script_text
    assert script_text.count(
        'CREATE VIRTUAL TABLE androidworld_fts4_probe USING fts4(value)'
    ) == 2
    assert 'OMNIFLOW_SQLITE_FTS4_LIBRARY' in script_text
    assert 'export LD_PRELOAD="$candidate_preload"' in script_text
    sqlite_gate = script_text.rindex("ensure_androidworld_sqlite_fts4\n")
    dry_run_gate = script_text.index(
        'echo "[dry-run] ready task=$task method=$method device=$device_target; '
    )
    assert dry_run_gate < sqlite_gate
    assert sqlite_gate < script_text.index('mkdir -p "$preflight_output_root"')


def test_e2e_source_collection_runs_the_sqlite_fts4_gate_before_dispatch() -> None:
    script_text = SCRIPT.read_text(encoding="utf-8")
    e2e_dispatch = script_text.index('exec "$python_bin" "${e2e_args[@]}"')
    e2e_gate = script_text.index("ensure_androidworld_sqlite_fts4\n", e2e_dispatch - 800)
    assert e2e_gate < e2e_dispatch


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
        "LLMTHU_API_KEY=llmthu-key\n",
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
            "OMNIFLOW_ANDROIDWORLD_STORE_PATH": str(store),
            "OMNIFLOW_DEVELOPMENT_OUTPUT_PATH": str(output),
            "OMNIFLOW_DEVELOPMENT_MODEL": "GLM-5.1",
            "OMNIFLOW_ANDROIDWORLD_PERFORM_EMULATOR_SETUP": "0",
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
        "LLMTHU_API_KEY=llmthu-key\n",
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
            "OMNIFLOW_ANDROIDWORLD_STORE_PATH": str(store),
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
        "LLMTHU_API_KEY=test-key\n",
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
            "OMNIFLOW_ANDROIDWORLD_STORE_PATH": str(store),
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


def test_experiment_script_uses_explicit_python_override(
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
            "PYTHON_BIN": str(base_python),
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
    env_file.write_text("LLMTHU_API_KEY=test-only\n", encoding="utf-8")
    capture = tmp_path / "invocation.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/bin/sh\n"
        "if [ \"${1:-}\" = \"-\" ]; then\n"
        "  exec \"$REAL_PYTHON\" \"$@\"\n"
        "fi\n"
        "printf '%s\\n' \"$@\" > \"$CAPTURE\"\n",
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
            "--task-deadline-sec",
            "1800",
            "--dry-run",
        ],
        cwd=REPO,
        env={
            **os.environ,
            "HOME": str(account_root),
            "CAPTURE": str(capture),
            "REAL_PYTHON": sys.executable,
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
    assert "--source-backend" not in invocation
    assert invocation[invocation.index("--task-deadline-sec") + 1] == "1800"
    assert invocation[invocation.index("--source-device") + 1] == (
        "source5560:emulator-5560:5560"
    )
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
        "PYTHON_BIN": sys.executable,
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
    assert "emulator-5560=OmniFlowSourceSmall" in completed.stdout
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
            "PYTHON_BIN": sys.executable,
            "SCRIPT_PREFIX": str(script_prefix),
            "TEST_ROOT": str(tmp_path),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == str(macos_sdk)


def test_missing_protocol_avd_is_provisioned_by_shared_helper(
    tmp_path: Path,
) -> None:
    script_prefix = tmp_path / "script-prefix.sh"
    script_prefix.write_text(
        SCRIPT.read_text(encoding="utf-8").split("\ndry_run=0\n", maxsplit=1)[0]
        + "\ndry_run=0\n",
        encoding="utf-8",
    )
    marker = tmp_path / "created"
    emulator = tmp_path / "emulator"
    emulator.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = '-list-avds' ] && [ -f \"$MARKER\" ]; then\n"
        "  printf '%s\\n' OmniFlowSourceSmall\n"
        "fi\n",
        encoding="utf-8",
    )
    emulator.chmod(0o755)
    avdmanager = tmp_path / "avdmanager"
    avdmanager.write_text("#!/bin/sh\ntouch \"$MARKER\"\n", encoding="utf-8")
    avdmanager.chmod(0o755)
    sdk = tmp_path / "sdk"
    for abi in ("arm64-v8a", "x86_64"):
        (sdk / "system-images" / "android-33" / "google_apis" / abi).mkdir(
            parents=True
        )

    completed = subprocess.run(
        [
            "bash",
            "-c",
            "source \"$SCRIPT_PREFIX\"; "
            "ensure_avd_installed OmniFlowSourceSmall "
            "\"$TEST_EMULATOR\" \"$TEST_AVDMANAGER\" \"$TEST_SDK\"",
        ],
        cwd=REPO,
        env={
            **os.environ,
            "PYTHON_BIN": sys.executable,
            "SCRIPT_PREFIX": str(script_prefix),
            "TEST_EMULATOR": str(emulator),
            "TEST_AVDMANAGER": str(avdmanager),
            "TEST_SDK": str(sdk),
            "MARKER": str(marker),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert marker.exists()


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
            "\"$source_device\" \"$device_target\" \"$emulator_avds\"",
        ],
        cwd=REPO,
        env={
            **os.environ,
            "PYTHON_BIN": sys.executable,
            "SCRIPT_PREFIX": str(script_prefix),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        "source5560:emulator-5560:5560",
        "small5554:emulator-5554:5554",
        (
            "emulator-5554=OmniFlowTargetSmall,emulator-5564=OmniFlowTargetFold,"
            "emulator-5560=OmniFlowSourceSmall"
        ),
    ]
    avd_names = [
        mapping.split("=", maxsplit=1)[1]
        for mapping in completed.stdout.splitlines()[2].split(",")
    ]
    assert len(avd_names) == len(set(avd_names)) == 3


@pytest.mark.parametrize(
    ("environment_override", "message"),
    [
        (
            {
                "OMNIFLOW_ANDROIDWORLD_DEVICE": (
                    "small5554:emulator-5554:5554,"
                    "fold5564:emulator-5554:5554"
                )
            },
            "Duplicate target serial",
        ),
        (
            {
                "OMNIFLOW_ANDROIDWORLD_DEVICE": (
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
        (["--method", "unknown_method"], "Unsupported paper method"),
        (
            ["--device", "small5554,small5554"],
            "Invalid device target",
        ),
    ],
)
def test_single_result_options_reject_invalid_selections(
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
    refresh_artifact_index(
        memory_root=memory_root,
        source_index=source_index,
        runlog_roots=(assets,),
        result_roots=(),
    )

    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
        "OMNIFLOW_EXP_ASSET_ROOT": str(assets),
        "OMNIFLOW_EXP_RESULTS_ROOT": str(results),
        "OMNIFLOW_ENV_FILE": str(env_file),
        "OMNIFLOW_ANDROID_WORLD_ROOT": str(android_world),
        "OMNIFLOW_ADB_PATH": str(fake_adb),
        "OMNIFLOW_ANDROIDWORLD_MANAGE_EMULATORS": "0",
        "OMNIFLOW_ANDROIDWORLD_METHOD": "fixed_replay",
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


def test_memory_refresh_routes_all_evidence_through_the_only_script(
    tmp_path: Path,
) -> None:
    runlogs = tmp_path / "runlogs"
    results = tmp_path / "results"
    runlogs.mkdir()
    results.mkdir()
    assets = tmp_path / "assets"
    source_index = assets / "inputs" / "final_source_index.json"
    source_index.parent.mkdir(parents=True)
    source_index.write_text("{}", encoding="utf-8")
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
        "OMNIFLOW_EXP_ASSET_ROOT": str(assets),
        "OMNIFLOW_EXP_MEMORY_ROOT": str(memory_root),
        "OMNIFLOW_MEMORY_RUNLOG_ROOTS": str(runlogs),
        "OMNIFLOW_MEMORY_RESULT_ROOTS": str(results),
        "OMNIFLOW_EXP_RESULTS_ROOT": str(results),
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
        "src.experiment.data_index",
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

    without_external_function_index = subprocess.run(
        ["bash", str(SCRIPT), "--refresh-memory"],
        cwd=REPO,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert without_external_function_index.returncode == 0, (
        without_external_function_index.stderr
    )
    assert captured.read_text(encoding="utf-8").splitlines() == [
        "-m",
        "src.experiment.data_index",
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
        "src.experiment.data_index",
        "refresh",
        "--memory-root",
        str(memory_root),
        "--runlog-root",
        str(runlogs),
        "--result-root",
        str(results),
    ]
