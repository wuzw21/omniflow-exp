#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from typing import Any
from urllib import parse, request

APPAGENT_OFFICIAL_REVISION = "2c1900422caf6f9e94e96d5dd984b530e5a5fbf8"
MOBILE_AGENT_V3_OFFICIAL_REVISION = "11cea575561fb7800b5fb6b6cafa56f7a91de11f"
GUI_OWL_7B_MODEL_REVISION = "7c1644c0288da07435a485701d0fea0ac353f38a"
QWEN_VL_UTILS_VERSION = "0.0.14"
TORCH_VERSION = "2.9.0+cpu"
TORCHVISION_VERSION = "0.24.0+cpu"


@dataclass
class Check:
    name: str
    status: str
    detail: str


def _run(command: list[str], timeout: float = 10.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )


def _command_targets_serial(command: str, serial: str) -> bool:
    requested = str(serial or "").strip()
    if not requested:
        return True
    if requested in command:
        return True
    port_match = re.search(r"(?:^|-)({})$".format(r"\d+"), requested)
    if port_match:
        port = port_match.group(1)
        if re.search(rf"(?:--console-port|--port)(?:=|\s+){re.escape(port)}(?:\s|$)", command):
            return True
    explicit_serials = re.findall(
        r"(?:--serial|--device-serial|-s)(?:=|\s+)([^\s]+)", command
    )
    if requested in explicit_serials:
        return True
    target_values = re.findall(
        r"(?:--device-targets?|--target-device)(?:=|\s+)([^\s]+)", command
    )
    if any(requested in value.split(":") for value in target_values):
        return True
    android_serials = re.findall(r"(?:^|\s)ANDROID_SERIAL=([^\s]+)", command)
    return requested in android_serials


def _stale_processes(serial: str) -> list[str]:
    result = _run(
        [
            "pgrep",
            "-af",
            (
                "src.experiment.androidworld|androidworld.py|"
                "mobile_agent_v3_runner.py|python main.py"
            ),
        ],
        timeout=3,
    )
    stale: list[str] = []
    for line in result.stdout.splitlines():
        match = re.match(r"\s*(\d+)\s+(.*)", line)
        if not match or int(match.group(1)) == os.getpid():
            continue
        if _command_targets_serial(match.group(2), serial):
            stale.append(match.group(1))
    return stale


def _wake_and_unlock(adb: str, serial: str) -> None:
    _run([adb, "-s", serial, "shell", "input", "keyevent", "WAKEUP"], timeout=10)
    _run([adb, "-s", serial, "shell", "wm", "dismiss-keyguard"], timeout=10)


def _resolve_tool(name: str, subdir: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    candidates = [
        Path(os.environ.get("ANDROID_HOME", "")) / subdir / name,
        Path(os.environ.get("ANDROID_SDK_ROOT", "")) / subdir / name,
        Path.home() / "Library" / "Android" / "sdk" / subdir / name,
        Path.home() / "Android" / "Sdk" / subdir / name,
        Path.home() / ".cache" / "omniflow" / "android-platform-tools" / subdir / name,
    ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return ""


def _hardware_acceleration_status(emulator: str) -> tuple[bool, str, str]:
    if platform.system() == "Linux":
        kvm = Path("/dev/kvm")
        return (
            kvm.exists() and os.access(kvm, os.R_OK | os.W_OK),
            "kvm",
            str(kvm),
        )
    if not emulator:
        return False, "hardware_acceleration", "emulator missing"
    result = _run([emulator, "-accel-check"], timeout=10)
    detail = result.stdout.strip() or f"exit={result.returncode}"
    return result.returncode == 0, "hardware_acceleration", detail


def _hash_files(paths: list[Path], *, relative_to: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        paths,
        key=lambda item: item.relative_to(relative_to).parts,
    ):
        digest.update(path.relative_to(relative_to).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _source_memory_files(memory_root: Path) -> tuple[list[Path], list[Path]]:
    if not memory_root.is_dir():
        return [], []
    files = sorted(
        path
        for path in memory_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    app_task_files = [
        path
        for path in files
        if path.name == "tasks.csv" and path.parent.parent == memory_root
    ]
    task_files = app_task_files or [
        path
        for path in files
        if path.name == "tasks.csv" and path.parent == memory_root
    ]
    return files, task_files


def _validate_mobilegpt_cold_manifest(memory_root: Path) -> dict[str, Any]:
    root = memory_root.expanduser().resolve()
    manifest_path = root.parent / "cold_memory_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        "omniflow.mobilegpt-cold-memory.v1"
    ):
        raise ValueError("mobilegpt_cold_memory_manifest_schema_invalid")
    if payload.get("source_seed") != 111:
        raise ValueError("mobilegpt_cold_memory_source_seed_invalid")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("mobilegpt_cold_memory_provenance_missing")
    if provenance.get("native_mobilegpt_learning") is not True or provenance.get(
        "complete_teacher_action_consumption"
    ) is not True:
        raise ValueError("mobilegpt_cold_memory_native_learning_incomplete")
    forbidden = [
        key
        for key in (
            "function_conversion_enabled",
            "target_inputs_read",
            "target_observations_read",
            "validator_state_read",
            "coordinate_replay",
        )
        if bool(provenance.get(key))
    ]
    if forbidden:
        raise ValueError("mobilegpt_cold_memory_forbidden:" + ",".join(forbidden))
    memory = payload.get("memory")
    if not isinstance(memory, dict):
        raise ValueError("mobilegpt_cold_memory_record_missing")
    recorded_root = (root.parent / str(memory.get("relative_path") or "")).resolve()
    if recorded_root != root:
        raise ValueError("mobilegpt_cold_memory_path_mismatch")
    files, task_files = _source_memory_files(root)
    digest = _hash_files(files, relative_to=root)
    if digest != str(memory.get("sha256") or ""):
        raise ValueError("mobilegpt_cold_memory_hash_mismatch")
    if len(files) != int(memory.get("file_count") or -1):
        raise ValueError("mobilegpt_cold_memory_file_count_mismatch")
    for label in (
        "teacher_source",
        "source_run_log",
        "source_stats",
        "official_source_result",
    ):
        record = payload.get(label)
        if not isinstance(record, dict):
            raise ValueError(f"mobilegpt_cold_memory_{label}_missing")
        path = (root.parent / str(record.get("relative_path") or "")).resolve()
        try:
            path.relative_to(root.parent)
        except ValueError as error:
            raise ValueError(
                f"mobilegpt_cold_memory_{label}_outside_bundle"
            ) from error
        if not path.is_file():
            raise ValueError(f"mobilegpt_cold_memory_{label}_file_missing")
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha256 != str(record.get("sha256") or ""):
            raise ValueError(f"mobilegpt_cold_memory_{label}_hash_mismatch")
    official = payload["official_source_result"]
    if official.get("official_validator_used") is not True or official.get(
        "official_validator_success"
    ) is not True:
        raise ValueError("mobilegpt_cold_memory_official_source_invalid")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "task_name": str(payload.get("task_name") or ""),
        "source_seed": int(payload["source_seed"]),
        "memory_sha256": digest,
        "memory_file_count": len(files),
        "task_file_count": len(task_files),
    }


def _valid_appagent_demo_manifest(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    source_metrics = payload.get("source_episode_metrics")
    doc_usage = payload.get("doc_generation_usage")
    if not isinstance(source_metrics, dict) or not isinstance(doc_usage, dict):
        return False
    try:
        teacher_action_count = int(payload.get("teacher_action_count") or 0)
        teacher_actions_consumed = int(payload.get("teacher_actions_consumed") or 0)
        source_prompt = int(source_metrics.get("prompt_tokens") or 0)
        source_completion = int(source_metrics.get("completion_tokens") or 0)
        source_total = int(source_metrics.get("total_tokens") or 0)
        doc_prompt = int(doc_usage.get("prompt_tokens") or 0)
        doc_completion = int(doc_usage.get("completion_tokens") or 0)
        doc_total = int(doc_usage.get("total_tokens") or 0)
        return (
            payload.get("schema_version") == "omniflow.appagent-demo-memory.v1"
            and payload.get("official_appagent_revision")
            == APPAGENT_OFFICIAL_REVISION
            and payload.get("source_seed") == 111
            and payload.get("official_source_success") is True
            and payload.get("teacher_complete") is True
            and teacher_action_count > 0
            and teacher_actions_consumed == teacher_action_count
            and float(source_metrics.get("duration_sec") or 0.0) > 0
            and float(source_metrics.get("wall_sec") or 0.0) > 0
            and source_total == source_prompt + source_completion
            and int(doc_usage.get("model_calls") or 0) > 0
            and doc_total == doc_prompt + doc_completion
            and doc_total > 0
            and float(doc_usage.get("wall_sec") or 0.0) > 0
            and float(payload.get("prep_wall_sec") or 0.0) > 0
            and payload.get("uses_omniflow_function") is False
            and payload.get("target_inputs_read") is False
            and payload.get("target_observations_read") is False
            and payload.get("validator_state_read_for_memory") is False
        )
    except (TypeError, ValueError):
        return False


def _port_is_free(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError:
        return False
    finally:
        sock.close()
    return True


def _oob_healthy(url: str) -> bool:
    try:
        with request.urlopen(url.rstrip("/") + "/health", timeout=3) as response:
            return 200 <= int(response.status) < 300
    except Exception:
        return False


def _openai_model_endpoint(
    base_url: str,
    api_key: str,
    model: str,
) -> tuple[bool, str]:
    url = f"{base_url.rstrip('/')}/models"
    try:
        endpoint_request = request.Request(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with request.urlopen(endpoint_request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = [
            str(item.get("id") or "")
            for item in list(payload.get("data") or [])
            if isinstance(item, dict)
        ]
        return model in models, json.dumps(
            {"url": url, "requested": model, "available": models},
            sort_keys=True,
        )
    except Exception as error:  # noqa: BLE001 - rendered into deterministic gate
        return False, f"{error.__class__.__name__}: {error}"


def _restore_oob_http(
    adb: str,
    serial: str,
    package: str,
    activity: str,
    url: str,
    receiver: str = ".DebugGetStateReceiver",
    accessibility_service: str = (
        "com.google.android.accessibility.selecttospeak.SelectToSpeakService"
    ),
) -> tuple[bool, str]:
    parsed = parse.urlparse(url)
    port = int(parsed.port or 8910)
    component = activity if "/" in activity else f"{package}/{activity}"
    service_component = (
        accessibility_service
        if "/" in accessibility_service
        else f"{package}/{accessibility_service}"
    )
    current = _run(
        [
            adb,
            "-s",
            serial,
            "shell",
            "settings",
            "get",
            "secure",
            "enabled_accessibility_services",
        ],
        timeout=10,
    )
    services = [
        item
        for item in current.stdout.strip().split(":")
        if item and item != "null"
    ]
    reset = None
    if service_component in services:
        reset = _run(
            [
                adb,
                "-s",
                serial,
                "shell",
                "settings",
                "put",
                "secure",
                "enabled_accessibility_services",
                ":".join(item for item in services if item != service_component),
            ],
            timeout=10,
        )
    if service_component not in services:
        services.append(service_component)
    bind_commands = [
        [
            adb,
            "-s",
            serial,
            "shell",
            "settings",
            "put",
            "secure",
            "enabled_accessibility_services",
            ":".join(services),
        ],
        [
            adb,
            "-s",
            serial,
            "shell",
            "settings",
            "put",
            "secure",
            "accessibility_enabled",
            "1",
        ],
    ]
    bind_results = [_run(command, timeout=10) for command in bind_commands]
    commands = [
        [adb, "-s", serial, "forward", "--remove", f"tcp:{port}"],
        [adb, "-s", serial, "forward", f"tcp:{port}", f"tcp:{port}"],
        [adb, "-s", serial, "shell", "am", "start", "-n", component],
    ]
    results = [_run(command, timeout=15) for command in commands]
    required = [current, *([] if reset is None else [reset]), *bind_results, *results[1:]]
    if any(result.returncode != 0 for result in required):
        detail = next(
            (
                result.stdout.strip()
                for result in required
                if result.returncode != 0 and result.stdout.strip()
            ),
            "adb forward or activity start failed",
        )
        return False, detail
    for _ in range(20):
        if _oob_healthy(url):
            return True, f"http:{url}"
        time.sleep(0.5)
    receiver_component = (
        f"{package}/{receiver}"
        if receiver.startswith(".")
        else receiver
        if "/" in receiver
        else f"{package}/{receiver}"
    )
    result_path = "files/debug-get-state-result.json"
    _run(
        [adb, "-s", serial, "shell", "run-as", package, "rm", "-f", result_path],
        timeout=10,
    )
    broadcast = _run(
        [
            adb,
            "-s",
            serial,
            "shell",
            "am",
            "broadcast",
            "-a",
            f"{package}.RUN_GET_STATE",
            "-n",
            receiver_component,
            "--ez",
            "includeXml",
            "false",
            "--ez",
            "includeScreenshot",
            "false",
        ],
        timeout=30,
    )
    if broadcast.returncode != 0:
        return False, broadcast.stdout.strip() or "OOB receiver broadcast failed"
    for _ in range(40):
        result = _run(
            [adb, "-s", serial, "shell", "run-as", package, "cat", result_path],
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            try:
                payload = json.loads(result.stdout)
            except json.JSONDecodeError as error:
                return False, f"invalid OOB receiver JSON: {error}"
            if payload.get("success") is True:
                return True, f"receiver:{receiver_component}"
            return False, str(payload.get("error_message") or payload.get("error") or "OOB receiver failed")
        time.sleep(0.5)
    return False, f"HTTP and receiver unavailable after starting {component}"


def _xml_resource_center(screen: str, resource_id: str) -> tuple[int, int] | None:
    match = re.search(
        rf'resource-id="{re.escape(resource_id)}"[^>]*bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"',
        screen,
    )
    if match is None:
        return None
    left, top, right, bottom = (int(value) for value in match.groups())
    return ((left + right) // 2, (top + bottom) // 2)


def _dump_ui_xml(adb: str, serial: str) -> str:
    path = "/sdcard/omniflow_preflight.xml"
    _run([adb, "-s", serial, "shell", "uiautomator", "dump", path], timeout=10)
    return _run([adb, "-s", serial, "shell", "cat", path], timeout=10).stdout


def _system_crash_dialog_present(*texts: str) -> bool:
    combined = "\n".join(str(text or "") for text in texts).casefold()
    return "keeps stopping" in combined or "application error:" in combined


def _contacts_home_ready(screen: str) -> bool:
    contacts_package = 'package="com.google.android.contacts"' in screen
    onboarding = 'text="Skip"' in screen or 'text="Sign in"' in screen
    home_capability = (
        'content-desc="Create contact"' in screen
        or 'text="Search contacts"' in screen
    )
    return contacts_package and home_capability and not onboarding


def _contacts_setup_ready(screen: str) -> bool:
    onboarding_ready = (
        'package="com.google.android.contacts"' in screen
        and 'text="Skip"' in screen
        and 'text="Sign in"' in screen
    )
    return _contacts_home_ready(screen) or onboarding_ready


def _navigate_contacts_back_to_home(
    adb: str,
    serial: str,
    screen: str,
    *,
    max_back_presses: int = 2,
) -> str:
    current = screen
    for _ in range(max_back_presses):
        if _contacts_home_ready(current):
            break
        contacts_package = 'package="com.google.android.contacts"' in current
        onboarding = 'text="Skip"' in current or 'text="Sign in"' in current
        if not contacts_package or onboarding:
            break
        _run(
            [adb, "-s", serial, "shell", "input", "keyevent", "BACK"],
            timeout=10,
        )
        time.sleep(1)
        current = _dump_ui_xml(adb, serial)
    return current


def _permission_deny_center(screen: str) -> tuple[int, int] | None:
    return next(
        (
            center
            for resource_id in (
                "com.android.permissioncontroller:id/permission_deny_button",
                "com.android.permissioncontroller:id/permission_deny_and_dont_ask_again_button",
            )
            if (center := _xml_resource_center(screen, resource_id)) is not None
        ),
        None,
    )


def _permission_is_fixed_denied(
    package_state: str,
    permission: str,
) -> bool:
    match = re.search(
        rf"{re.escape(permission)}:\s+granted=false,\s+flags=\[([^]]*)]",
        package_state,
    )
    if match is None:
        return False
    flags = set(match.group(1).replace("|", " ").split())
    return {"USER_FIXED", "USER_SET"}.issubset(flags)


def _validate_function_manifest(manifest_path: Path, repo: Path) -> dict[str, Any]:
    resolved_manifest = manifest_path.expanduser().resolve()
    if not resolved_manifest.is_file():
        raise ValueError(f"function_manifest_missing:{resolved_manifest}")
    try:
        resolved_manifest.relative_to(repo)
    except ValueError as error:
        raise ValueError("function_manifest_outside_repo") from error
    payload = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "omniflow.androidworld.agent-function-suite.v1":
        raise ValueError("function_manifest_schema_invalid")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("function_manifest_tasks_required")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from omniflow.functions.store import FunctionStore

    task_names = set()
    store_paths = []
    function_count = 0
    for row in tasks:
        if not isinstance(row, dict):
            raise ValueError("function_manifest_task_invalid")
        task_name = str(row.get("task_name") or "").strip()
        if not task_name or task_name in task_names:
            raise ValueError(f"function_manifest_task_duplicate_or_empty:{task_name}")
        task_names.add(task_name)
        enhancement_value = str(row.get("enhancement_root") or "").strip()
        if not enhancement_value:
            raise ValueError(f"function_manifest_enhancement_missing:{task_name}")
        enhancement_root = Path(enhancement_value)
        if not enhancement_root.is_absolute():
            enhancement_root = repo / enhancement_root
        enhancement_root = enhancement_root.resolve()
        try:
            enhancement_root.relative_to(repo)
        except ValueError as error:
            raise ValueError(
                f"function_manifest_enhancement_outside_repo:{task_name}"
            ) from error
        store_path = enhancement_root / "store.json"
        store = FunctionStore(store_path)
        functions = store.list_functions(limit=500)
        if not functions:
            raise ValueError(f"function_store_empty:{task_name}")
        store_paths.append(store_path)
        function_count += len(functions)
    return {
        "task_count": len(tasks),
        "function_count": function_count,
        "files": [resolved_manifest, *store_paths],
    }


def _validate_source_index(
    index_path: Path,
    *,
    source_root: Path,
    expected_tasks: int,
) -> dict[str, Any]:
    resolved_index = index_path.expanduser().resolve()
    payload = json.loads(resolved_index.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or len(payload) != expected_tasks:
        raise ValueError(
            f"source_index_task_count_invalid:{len(payload) if isinstance(payload, dict) else 0}/{expected_tasks}"
        )
    run_logs: list[Path] = []
    invalid: list[str] = []
    for task, metadata in payload.items():
        if not isinstance(metadata, dict) or metadata.get("replay_seed") != 111:
            invalid.append(str(task))
            continue
        run_log = Path(str(metadata.get("retained_source_run_log") or "")).expanduser()
        if not run_log.is_absolute():
            run_log = (source_root / run_log).resolve()
        if not run_log.is_file():
            invalid.append(str(task))
            continue
        run_logs.append(run_log)
    if invalid:
        raise ValueError(
            f"source_index_invalid_tasks:{len(invalid)}:" + ",".join(invalid[:10])
        )
    return {
        "task_count": len(payload),
        "index": resolved_index,
        "index_sha256": hashlib.sha256(resolved_index.read_bytes()).hexdigest(),
        "run_log_count": len(run_logs),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate AndroidWorld runtime health.")
    parser.add_argument("--repo", required=True)
    parser.add_argument(
        "--code-root",
        help="Exact release root for code checks; --repo remains the runtime asset root.",
    )
    parser.add_argument(
        "--profile",
        choices=[
            "mobilegpt",
            "appagent",
            "function",
            "androidworld_native",
            "mobile_agent_v3",
        ],
        default="",
        help="Runtime dependency profile. Function mode is inferred from --function-manifest.",
    )
    parser.add_argument("--serial", default="emulator-5554")
    parser.add_argument("--expected-tasks", type=int)
    parser.add_argument("--source-index")
    parser.add_argument("--source-root")
    parser.add_argument("--source-memory-root")
    parser.add_argument("--expected-memory-tasks", type=int)
    parser.add_argument("--appagent-root")
    parser.add_argument("--appagent-demo-memory-root")
    parser.add_argument("--mobile-agent-v3-root")
    parser.add_argument("--mobile-agent-v3-model-root")
    parser.add_argument(
        "--mobile-agent-v3-official-revision",
        default=MOBILE_AGENT_V3_OFFICIAL_REVISION,
    )
    parser.add_argument(
        "--mobile-agent-v3-model-revision",
        default=GUI_OWL_7B_MODEL_REVISION,
    )
    parser.add_argument("--mobile-agent-v3-model", default="GUI-Owl-7B")
    parser.add_argument(
        "--mobile-agent-v3-base-url",
        default="http://127.0.0.1:4243/v1",
    )
    parser.add_argument("--mobile-agent-v3-api-key", default="local-vllm")
    parser.add_argument("--function-manifest")
    parser.add_argument("--minimum-free-gb", type=float, default=40.0)
    parser.add_argument("--server-port", type=int, default=12345)
    parser.add_argument("--oob-url", default=os.getenv("OMNIFLOW_OOB_DEVICE_URL", "http://127.0.0.1:8910"))
    parser.add_argument(
        "--oob-package",
        default=os.getenv("OMNIFLOW_OOB_PACKAGE", "cn.com.omnimind.bot.debug"),
    )
    parser.add_argument(
        "--oob-activity",
        default=os.getenv(
            "OMNIFLOW_OOB_ACTIVITY",
            "cn.com.omnimind.bot.activity.LauncherActivity",
        ),
    )
    parser.add_argument(
        "--oob-receiver",
        default=os.getenv("OMNIFLOW_OOB_GET_STATE_RECEIVER", ".DebugGetStateReceiver"),
    )
    parser.add_argument(
        "--oob-accessibility-service",
        default=os.getenv(
            "OMNIFLOW_OOB_ACCESSIBILITY_SERVICE",
            "com.google.android.accessibility.selecttospeak.SelectToSpeakService",
        ),
    )
    parser.add_argument("--require-kvm", action="store_true")
    parser.add_argument("--require-device", action="store_true")
    parser.add_argument("--require-contacts-ready", action="store_true")
    parser.add_argument("--json-out")
    return parser


def _required_files(profile: str) -> list[str]:
    if profile == "function":
        return [
            "omniflow/functions/artifact.py",
            "omniflow/functions/store.py",
            "skills/androidworld-runlog-harvester/scripts/run_4090_function_campaign.py",
            "runtime/external/droidrun-android-world/android_world/android_world/env/setup_device/apps.py",
        ]
    if profile == "appagent":
        return [
            "src/experiment/androidworld.py",
            "src/integrations/appagent_adapter.py",
            "src/integrations/android_world/launch.py",
            "runtime/external/appagent/scripts/document_generation.py",
            "runtime/external/droidrun-android-world/android_world/android_world/env/setup_device/apps.py",
        ]
    if profile == "androidworld_native":
        return [
            "src/experiment/androidworld.py",
            "src/integrations/android_world/launch.py",
            "runtime/external/droidrun-android-world/android_world/android_world/env/setup_device/apps.py",
        ]
    if profile == "mobile_agent_v3":
        return [
            "src/experiment/androidworld.py",
            "src/integrations/mobile_agent_v3_runner.py",
            "src/integrations/mobile_agent_v3_adapter.py",
        ]
    if profile == "mobilegpt":
        return [
            "src/experiment/androidworld.py",
            "src/integrations/mobilegpt_runtime.py",
            "src/integrations/mobilegpt_teacher.py",
            "runtime/external/mobilegpt/Server/main.py",
            "runtime/external/droidrun-android-world/android_world/android_world/env/setup_device/apps.py",
        ]
    raise ValueError(f"unsupported_preflight_profile:{profile}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = Path(args.repo).expanduser().resolve()
    code_root = Path(args.code_root or repo).expanduser().resolve()
    checks: list[Check] = []
    requested_profile = str(args.profile or "").strip()
    function_mode = requested_profile == "function" or bool(
        str(args.function_manifest or "").strip()
    )
    appagent_mode = requested_profile == "appagent"
    native_mode = requested_profile == "androidworld_native"
    mobile_agent_v3_mode = requested_profile == "mobile_agent_v3"
    if function_mode and appagent_mode:
        raise ValueError("preflight_profile_conflicts_with_function_manifest")
    profile = (
        "function"
        if function_mode
        else "appagent"
        if appagent_mode
        else "androidworld_native"
        if native_mode
        else "mobile_agent_v3"
        if mobile_agent_v3_mode
        else "mobilegpt"
    )

    def add(name: str, passed: bool, detail: str, warning: bool = False) -> None:
        checks.append(Check(name, "ok" if passed else "warning" if warning else "fail", detail))

    for relative in _required_files(profile):
        root = repo if relative.startswith("runtime/") else code_root
        add(f"file:{relative}", (root / relative).is_file(), relative)
    if appagent_mode:
        appagent_root = Path(
            args.appagent_root
            or os.getenv("APPAGENT_ROOT", str(repo / "runtime/external/appagent"))
        ).expanduser().resolve()
        add(
            "appagent_root",
            (appagent_root / "scripts" / "model.py").is_file()
            and (appagent_root / ".git").exists(),
            str(appagent_root),
        )
        revision = _run(
            ["git", "-C", str(appagent_root), "rev-parse", "HEAD"],
            timeout=10,
        )
        actual_revision = revision.stdout.strip()
        add(
            "appagent_revision",
            revision.returncode == 0
            and actual_revision == APPAGENT_OFFICIAL_REVISION,
            actual_revision or "unavailable",
        )
    if mobile_agent_v3_mode:
        official_root_value = str(
            args.mobile_agent_v3_root
            or os.getenv("MOBILE_AGENT_V3_ROOT", "")
        ).strip()
        official_root = (
            Path(official_root_value).expanduser().resolve()
            if official_root_value
            else Path("/__missing_mobile_agent_v3_root__")
        )
        official_code_root = official_root / "Mobile-Agent-v3" / "android_world_v3"
        official_files = (
            "run_ma3.py",
            "android_world/agents/infer_ma3.py",
            "android_world/agents/mobile_agent_v3.py",
            "android_world/agents/mobile_agent_v3_agent.py",
            "android_world/agents/new_json_action.py",
            "android_world/suite_utils.py",
        )
        add(
            "mobile_agent_v3_root",
            official_root.is_dir()
            and all((official_code_root / relative).is_file() for relative in official_files),
            str(official_root),
        )
        revision = _run(
            ["git", "-C", str(official_root), "rev-parse", "HEAD"],
            timeout=10,
        )
        actual_revision = revision.stdout.strip()
        add(
            "mobile_agent_v3_revision",
            revision.returncode == 0
            and actual_revision == str(args.mobile_agent_v3_official_revision),
            actual_revision or "unavailable",
        )
        tracked_status = _run(
            [
                "git", "-C", str(official_root), "status", "--short",
                "--untracked-files=no",
            ],
            timeout=10,
        )
        add(
            "mobile_agent_v3_tracked_checkout_clean",
            tracked_status.returncode == 0 and not tracked_status.stdout.strip(),
            tracked_status.stdout.strip() or "clean",
        )
        model_root_value = str(args.mobile_agent_v3_model_root or "").strip()
        try:
            if str(code_root) not in sys.path:
                sys.path.insert(0, str(code_root))
            from src.integrations.mobile_agent_v3_adapter import inspect_gui_owl_model

            model_audit = inspect_gui_owl_model(
                model_root_value,
                revision=str(args.mobile_agent_v3_model_revision),
            )
        except Exception as error:  # noqa: BLE001 - deterministic preflight row
            add("gui_owl_model_snapshot", False, str(error))
        else:
            add(
                "gui_owl_model_snapshot",
                bool(model_audit.get("revision_metadata_complete")),
                json.dumps(
                    {
                        "root": model_audit.get("model_root"),
                        "revision": model_audit.get("revision"),
                        "required_file_count": model_audit.get("required_file_count"),
                        "total_bytes": model_audit.get("total_bytes"),
                    },
                    sort_keys=True,
                ),
            )
        endpoint_ready, endpoint_detail = _openai_model_endpoint(
            str(args.mobile_agent_v3_base_url),
            str(args.mobile_agent_v3_api_key),
            str(args.mobile_agent_v3_model),
        )
        add("gui_owl_model_endpoint", endpoint_ready, endpoint_detail)

    source_memory_value = str(
        args.source_memory_root or os.getenv("MOBILEGPT_SOURCE_MEMORY_ROOT", "")
    ).strip()
    memory_root = (
        Path(source_memory_value).expanduser().resolve()
        if source_memory_value
        else None
    )
    memory_files: list[Path] = []
    memory_tasks: list[Path] = []
    function_files: list[Path] = []
    if function_mode:
        try:
            function_validation = _validate_function_manifest(
                Path(args.function_manifest),
                repo,
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            add("function_manifest", False, str(error))
            add("function_validation", False, str(error))
        else:
            task_count = int(function_validation["task_count"])
            expected_tasks = args.expected_tasks or task_count
            add(
                "function_manifest",
                task_count == expected_tasks,
                f"{task_count}/{expected_tasks}",
            )
            add(
                "function_validation",
                int(function_validation["function_count"]) > 0,
                json.dumps(
                    {
                        "task_count": task_count,
                        "function_count": function_validation["function_count"],
                    }
                ),
            )
            function_files = list(function_validation["files"])
    elif appagent_mode:
        memory_root = None
        demo_memory_value = str(
            args.appagent_demo_memory_root
            or os.getenv("APPAGENT_DEMO_MEMORY_ROOT", "")
        ).strip()
        if not demo_memory_value:
            add("appagent_demo_memory", True, "not required for appagent_baseline")
        else:
            demo_memory = Path(demo_memory_value).expanduser().resolve()
            manifest = demo_memory / "appagent_demo_manifest.json"
            valid = False
            detail = str(manifest)
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                valid = _valid_appagent_demo_manifest(payload)
                detail = f"sealed manifest at {manifest}"
            except (OSError, ValueError, json.JSONDecodeError) as error:
                detail = str(error)
            add("appagent_demo_memory", valid, detail)
            if valid:
                memory_files = [
                    path
                    for path in demo_memory.rglob("*")
                    if path.is_file() and "__pycache__" not in path.parts
                ]
                memory_root = demo_memory
    elif native_mode or mobile_agent_v3_mode:
        memory_root = None
        expected_tasks = args.expected_tasks or 116
        source_index_value = str(args.source_index or "").strip()
        if not source_index_value:
            add("source_index", False, "--source-index is required")
        else:
            source_root = Path(args.source_root or repo).expanduser().resolve()
            try:
                source_validation = _validate_source_index(
                    Path(source_index_value),
                    source_root=source_root,
                    expected_tasks=expected_tasks,
                )
            except (OSError, ValueError, json.JSONDecodeError) as error:
                add("source_index", False, str(error))
            else:
                add(
                    "source_index",
                    source_validation["task_count"] == expected_tasks,
                    json.dumps(
                        {
                            "tasks": source_validation["task_count"],
                            "run_logs": source_validation["run_log_count"],
                            "sha256": source_validation["index_sha256"],
                        },
                        sort_keys=True,
                    ),
                )
    else:
        expected_tasks = args.expected_tasks or 116
        source_root = repo / "runtime/evals/androidworld_validator/core_archive/success_source_runlogs/by_task"
        source_metadata = list(source_root.glob("*/metadata.json"))
        add("source_runlogs", len(source_metadata) == expected_tasks, f"{len(source_metadata)}/{expected_tasks}")

        if memory_root is None:
            add("initial_memory", True, "empty_memory")
        else:
            memory_files, memory_tasks = _source_memory_files(memory_root)
            expected_memory_tasks = args.expected_memory_tasks
            task_count_valid = (
                len(memory_tasks) == expected_memory_tasks
                if expected_memory_tasks is not None
                else bool(memory_tasks)
            )
            add(
                "source_memory",
                memory_root.is_dir() and task_count_valid,
                (
                    f"{len(memory_tasks)}/{expected_memory_tasks} tasks.csv at "
                    f"{memory_root}"
                    if expected_memory_tasks is not None
                    else f"{len(memory_tasks)} tasks.csv at {memory_root}"
                ),
            )
            invalid_task_files = [
                path for path in memory_tasks if path.stat().st_size <= 0
            ]
            add(
                "memory_validation",
                bool(memory_tasks) and not invalid_task_files,
                (
                    "native source memory readable"
                    if memory_tasks and not invalid_task_files
                    else "missing or empty tasks.csv"
                ),
            )
            try:
                cold_manifest = _validate_mobilegpt_cold_manifest(memory_root)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                add("cold_memory_manifest", False, str(error))
            else:
                add(
                    "cold_memory_manifest",
                    True,
                    json.dumps(cold_manifest, sort_keys=True),
                )

    disk = shutil.disk_usage(repo if repo.exists() else Path.home())
    free_gb = disk.free / (1024 ** 3)
    add("disk_free", free_gb >= args.minimum_free_gb, f"{free_gb:.2f} GiB free; require {args.minimum_free_gb:.2f}")
    add("python", sys.version_info >= (3, 11), platform.python_version())
    module_names = [
        "absl",
        "android_env.proto.a11y",
        "android_world.env.android_world_controller",
        "dotenv",
        "grpc",
        "json_repair",
        "numpy",
        "openai",
        "pandas",
    ]
    if mobile_agent_v3_mode:
        module_names.append("qwen_vl_utils")
    if not function_mode:
        module_names.append("uiautomator2")
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
        except ImportError as error:
            add(f"python_module:{module_name}", False, str(error))
        else:
            add(f"python_module:{module_name}", True, "importable")
    if mobile_agent_v3_mode:
        for distribution, expected_version in (
            ("qwen-vl-utils", QWEN_VL_UTILS_VERSION),
            ("torch", TORCH_VERSION),
            ("torchvision", TORCHVISION_VERSION),
        ):
            try:
                actual_version = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                actual_version = "missing"
            add(
                f"package_version:{distribution}",
                actual_version == expected_version,
                f"actual={actual_version}; expected={expected_version}",
            )
    if appagent_mode or native_mode or mobile_agent_v3_mode:
        add("jq", True, f"not required by {profile} profile")
    else:
        add("jq", bool(shutil.which("jq")), shutil.which("jq") or "missing")
    add("java", bool(shutil.which("java")), shutil.which("java") or "missing")
    if mobile_agent_v3_mode:
        add("model_key", True, "not required by local GUI-Owl endpoint")
    else:
        add("model_key", bool(os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")), "configured" if os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY") else "missing")
    if appagent_mode or native_mode or mobile_agent_v3_mode:
        add("server_port", True, f"not required by {profile} profile")
    else:
        add("server_port", _port_is_free(args.server_port), f"127.0.0.1:{args.server_port}")

    stale_pids = _stale_processes(args.serial)
    add("stale_processes", not stale_pids, ",".join(stale_pids) if stale_pids else "none")

    if args.require_kvm:
        emulator = _resolve_tool("emulator", "emulator")
        add("emulator", bool(emulator), emulator or "missing")
        acceleration_ok, acceleration_name, acceleration_detail = (
            _hardware_acceleration_status(emulator)
        )
        add(acceleration_name, acceleration_ok, acceleration_detail)

    adb = _resolve_tool("adb", "platform-tools")
    add("adb", bool(adb), adb or "missing")
    if adb and args.require_device:
        devices = _run([adb, "devices"], timeout=10).stdout
        ready = any(line.split()[:2] == [args.serial, "device"] for line in devices.splitlines())
        add("device", ready, args.serial)
        if ready:
            boot = _run([adb, "-s", args.serial, "shell", "getprop", "sys.boot_completed"], timeout=5).stdout.strip()
            add("boot_completed", boot == "1", boot or "empty")
            _wake_and_unlock(adb, args.serial)
            focused_windows = _run(
                [adb, "-s", args.serial, "shell", "dumpsys", "window", "windows"],
                timeout=10,
            ).stdout
            crash_dialog_present = _system_crash_dialog_present(focused_windows)
            add(
                "system_crash_dialog",
                not crash_dialog_present,
                "present" if crash_dialog_present else "none",
            )
            if function_mode:
                required_packages = (args.oob_package, "com.google.android.contacts")
            elif appagent_mode:
                required_packages = (
                    ("com.google.android.contacts",)
                    if args.require_contacts_ready
                    else ()
                )
            elif native_mode or mobile_agent_v3_mode:
                required_packages = (
                    ("com.google.android.contacts",)
                    if args.require_contacts_ready
                    else ()
                )
            else:
                required_packages = ("com.example.MobileGPT", "com.google.android.contacts")
            for package_name in required_packages:
                package = _run([adb, "-s", args.serial, "shell", "pm", "path", package_name], timeout=10)
                add(f"package:{package_name}", package.stdout.strip().startswith("package:"), package.stdout.strip() or "missing")

            if args.require_contacts_ready:
                _wake_and_unlock(adb, args.serial)
                notification_permission = "android.permission.POST_NOTIFICATIONS"
                _run(
                    [
                        adb,
                        "-s",
                        args.serial,
                        "shell",
                        "pm",
                        "revoke",
                        "com.google.android.contacts",
                        notification_permission,
                    ],
                    timeout=10,
                )
                _run(
                    [
                        adb,
                        "-s",
                        args.serial,
                        "shell",
                        "pm",
                        "set-permission-flags",
                        "com.google.android.contacts",
                        notification_permission,
                        "user-set",
                        "user-fixed",
                    ],
                    timeout=10,
                )
                contacts_package_state = _run(
                    [
                        adb,
                        "-s",
                        args.serial,
                        "shell",
                        "dumpsys",
                        "package",
                        "com.google.android.contacts",
                    ],
                    timeout=10,
                ).stdout
                contacts_notification_fixed = _permission_is_fixed_denied(
                    contacts_package_state,
                    notification_permission,
                )
                add(
                    "contacts_notification_policy",
                    contacts_notification_fixed,
                    "denied and user-fixed"
                    if contacts_notification_fixed
                    else "notification permission not fixed denied",
                )
                _run(
                    [adb, "-s", args.serial, "shell", "am", "force-stop", args.oob_package],
                    timeout=10,
                )
                screen = ""
                for _ in range(3):
                    _run([adb, "-s", args.serial, "shell", "am", "start", "-n", "com.google.android.contacts/com.android.contacts.activities.PeopleActivity"], timeout=10)
                    time.sleep(2)
                    screen = _dump_ui_xml(adb, args.serial)
                    if (
                        'package="com.google.android.contacts"' in screen
                        or 'package="com.google.android.permissioncontroller"' in screen
                    ):
                        break
                for _ in range(3):
                    deny_center = _permission_deny_center(screen)
                    if deny_center is not None:
                        _run(
                            [
                                adb,
                                "-s",
                                args.serial,
                                "shell",
                                "input",
                                "tap",
                                str(deny_center[0]),
                                str(deny_center[1]),
                            ],
                            timeout=10,
                        )
                        time.sleep(2)
                    screen = _dump_ui_xml(adb, args.serial)
                    if _contacts_home_ready(screen):
                        break
                    time.sleep(1)
                screen = _navigate_contacts_back_to_home(adb, args.serial, screen)
                contacts_ready = _contacts_setup_ready(screen)
                contacts_detail = (
                    "contacts home"
                    if _contacts_home_ready(screen)
                    else "contacts onboarding ready"
                    if contacts_ready
                    else "unknown contacts screen"
                )
                add("contacts_ready", contacts_ready, contacts_detail)
                _run([adb, "-s", args.serial, "shell", "am", "force-stop", "com.google.android.contacts"], timeout=5)

    if native_mode or mobile_agent_v3_mode:
        add("oob_health", True, f"not required by {profile} profile")
    elif args.oob_url and adb and args.require_device:
        restored, detail = _restore_oob_http(
            adb,
            args.serial,
            args.oob_package,
            args.oob_activity,
            args.oob_url,
            args.oob_receiver,
            args.oob_accessibility_service,
        )
        add("oob_health", restored, detail)
    elif args.oob_url:
        add("oob_health", _oob_healthy(args.oob_url), args.oob_url)

    failures = [check for check in checks if check.status == "fail"]
    warnings = [check for check in checks if check.status == "warning"]
    fingerprint_files = (
        [path for path in function_files if path.is_file()]
        if function_mode
        else memory_files
    )
    fingerprint_root = repo if function_mode or memory_root is None else memory_root
    report = {
        "schema_version": "omniflow.androidworld_runtime_preflight.v1",
        "ready": not failures,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "repo": str(repo),
        "code_root": str(code_root),
        "serial": args.serial,
        "profile": profile,
        "initial_memory_condition": (
            "function_suite"
            if function_mode
            else "appagent_demo_memory"
            if appagent_mode and memory_root is not None
            else "native_memory"
            if memory_root is not None
            else "empty_memory"
        ),
        "source_memory_root": str(memory_root or ""),
        "memory_fingerprint": _hash_files(fingerprint_files, relative_to=fingerprint_root) if fingerprint_files else "",
        "checks": [asdict(check) for check in checks],
        "failure_count": len(failures),
        "warning_count": len(warnings),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json_out:
        output = Path(args.json_out).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
