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

from omniflow.core.trajectory import canonicalize_run_log as import_run_log
from src.experiment.protocol import SOURCE_SEED
from src.integrations.appagent import is_memory_manifest_valid
from src.integrations.mobilegpt import validate_memory_manifest

APPAGENT_OFFICIAL_REVISION = "2c1900422caf6f9e94e96d5dd984b530e5a5fbf8"
APPAGENT_REQUIRED_MODULES = (
    "colorama",
    "cv2",
    "dashscope",
    "pyshine",
    "requests",
    "yaml",
)
REQUIRED_DISTRIBUTION_VERSIONS = {"android-env": "1.2.3"}


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


def _collect_memory_files(root: Path) -> tuple[list[Path], list[Path]]:
    """Collect files for generic check reporting without knowing a provider schema."""

    if not root.is_dir():
        return [], []
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    task_files = [
        path
        for path in files
        if path.name == "tasks.csv" and path.parent.parent == root
    ]
    return files, task_files


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
    target_values = re.findall(r"(?:--device)(?:=|\s+)([^\s]+)", command)
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
                "src.experiment.run_task|run_task.py|"
                "python main.py"
            ),
        ],
        timeout=3,
    )
    stale: list[str] = []
    ignored_pids = {os.getpid(), os.getppid()}
    for line in result.stdout.splitlines():
        match = re.match(r"\s*(\d+)\s+(.*)", line)
        if not match or int(match.group(1)) in ignored_pids:
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


def _port_is_free(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError:
        return False
    finally:
        sock.close()
    return True


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


def _dismiss_known_accessibility_crash_dialog(
    adb: str,
    serial: str,
    focused_windows: str,
) -> str:
    marker = "application error: com.google.androidenv.accessibilityforwarder"
    if marker not in str(focused_windows or "").casefold():
        return focused_windows
    dismissed = _run(
        [
            adb,
            "-s",
            serial,
            "shell",
            "am",
            "broadcast",
            "--async",
            "-a",
            "android.intent.action.CLOSE_SYSTEM_DIALOGS",
        ],
        timeout=10,
    )
    if dismissed.returncode != 0:
        return focused_windows
    time.sleep(0.2)
    return _run(
        [adb, "-s", serial, "shell", "dumpsys", "window", "windows"],
        timeout=10,
    ).stdout


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


def _reset_contacts_setup_screen(adb: str, serial: str) -> str:
    package_name = "com.google.android.contacts"
    component = f"{package_name}/com.android.contacts.activities.PeopleActivity"
    _run(
        [adb, "-s", serial, "shell", "am", "force-stop", package_name],
        timeout=10,
    )
    _run(
        [
            adb,
            "-s",
            serial,
            "shell",
            "am",
            "start",
            "-S",
            "-W",
            "--activity-clear-task",
            "--activity-new-task",
            "-n",
            component,
        ],
        timeout=15,
    )
    screen = ""
    for _ in range(5):
        time.sleep(1)
        screen = _dump_ui_xml(adb, serial)
        deny_center = _permission_deny_center(screen)
        if deny_center is not None:
            _run(
                [
                    adb,
                    "-s",
                    serial,
                    "shell",
                    "input",
                    "tap",
                    str(deny_center[0]),
                    str(deny_center[1]),
                ],
                timeout=10,
            )
            continue
        if _contacts_setup_ready(screen):
            return screen
    return _navigate_contacts_back_to_home(
        adb,
        serial,
        screen,
        max_back_presses=4,
    )


def _validate_source_index(
    index_path: Path,
    *,
    source_root: Path,
    expected_tasks: int | None,
    task_names: tuple[str, ...] = (),
    allow_historical_source: bool = False,
) -> dict[str, Any]:
    resolved_index = index_path.expanduser().resolve()
    payload = json.loads(resolved_index.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("source_index_must_be_object")
    if payload.get("schema_version") == "omniflow.data-index.v2":
        payload = payload.get("source_index")
        if not isinstance(payload, dict):
            raise ValueError("current_source_index_must_be_object")
    if expected_tasks is not None and len(payload) != expected_tasks:
        raise ValueError(
            f"source_index_task_count_invalid:{len(payload)}/{expected_tasks}"
        )
    selected_tasks = tuple(dict.fromkeys(task_names))
    missing_tasks = [task for task in selected_tasks if task not in payload]
    if missing_tasks:
        raise ValueError(
            "source_index_selected_tasks_missing:" + ",".join(missing_tasks)
        )
    items = (
        ((task, payload[task]) for task in selected_tasks)
        if selected_tasks
        else payload.items()
    )
    run_logs: list[Path] = []
    invalid: list[str] = []
    for task, metadata in items:
        if not isinstance(metadata, dict):
            invalid.append(str(task))
            continue
        source_kind = str(metadata.get("source_kind") or "").strip()
        if (
            metadata.get("latest_official_success_source") is not True
            and not allow_historical_source
        ):
            invalid.append(str(task))
            continue
        if source_kind and source_kind not in {
            "androidworld_validator_success_source_runlog",
            "one_time_canonicalized_seed111_screenshot_source",
        }:
            invalid.append(str(task))
            continue
        run_log_value = str(
            metadata.get("retained_source_run_log")
            or metadata.get("source_run_log")
            or ""
        ).strip()
        if not run_log_value:
            invalid.append(str(task))
            continue
        run_log = Path(run_log_value).expanduser()
        if not run_log.is_absolute():
            index_relative = (resolved_index.parent / run_log).resolve()
            source_relative = (source_root / run_log).resolve()
            run_log = (
                index_relative if index_relative.is_file() else source_relative
            )
        if not run_log.is_file():
            invalid.append(str(task))
            continue
        expected_sha256 = str(
            metadata.get("retained_source_run_log_sha256")
            or metadata.get("source_run_log_sha256")
            or ""
        ).strip()
        actual_sha256 = hashlib.sha256(run_log.read_bytes()).hexdigest()
        if not expected_sha256 or expected_sha256 != actual_sha256:
            invalid.append(str(task))
            continue
        try:
            canonical = import_run_log(
                json.loads(run_log.read_text(encoding="utf-8"))
            )
        except (OSError, ValueError, json.JSONDecodeError):
            invalid.append(str(task))
            continue
        if (
            canonical.get("status") != "succeeded"
            or canonical.get("success") is not True
            or not canonical.get("steps")
        ):
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
        "--android-world-root",
        help="Explicit AndroidWorld checkout root used for runtime imports.",
    )
    parser.add_argument(
        "--code-root",
        help="Exact release root for code checks; --repo remains the runtime asset root.",
    )
    parser.add_argument(
        "--profile",
        choices=[
            "mobilegpt",
            "appagent",
            "androidworld_native",
        ],
        default="",
        help="Runtime dependency profile for one of the formal paper methods.",
    )
    parser.add_argument("--serial", default="emulator-5554")
    parser.add_argument("--expected-tasks", type=int)
    parser.add_argument("--source-index")
    parser.add_argument("--source-task", action="append", default=[])
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--source-root")
    parser.add_argument("--source-memory-root")
    parser.add_argument("--expected-memory-tasks", type=int)
    parser.add_argument("--appagent-root")
    parser.add_argument("--appagent-memory-root")
    parser.add_argument("--minimum-free-gb", type=float, default=20.0)
    parser.add_argument("--server-port", type=int, default=12345)
    parser.add_argument("--require-kvm", action="store_true")
    parser.add_argument("--require-device", action="store_true")
    parser.add_argument("--require-contacts-ready", action="store_true")
    parser.add_argument("--json-out")
    return parser


def _required_files(profile: str) -> list[str]:
    if profile == "appagent":
        return [
            "src/experiment/run_task.py",
            "src/integrations/appagent.py",
            "src/integrations/android_world/run_episode.py",
            "runtime/external/appagent/scripts/document_generation.py",
            "runtime/external/droidrun-android-world/android_world/android_world/env/setup_device/apps.py",
        ]
    if profile == "androidworld_native":
        return [
            "src/experiment/run_task.py",
            "src/integrations/android_world/run_episode.py",
        ]
    if profile == "mobilegpt":
        return [
            "src/experiment/run_task.py",
            "src/integrations/mobilegpt.py",
            "src/integrations/mobilegpt_runtime.py",
            "runtime/external/mobilegpt/Server/main.py",
            "runtime/external/droidrun-android-world/android_world/android_world/env/setup_device/apps.py",
        ]
    raise ValueError(f"unsupported_preflight_profile:{profile}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    android_world_root = str(
        args.android_world_root or os.getenv("OMNIFLOW_ANDROID_WORLD_ROOT", "")
    ).strip()
    if android_world_root:
        sys.path.insert(0, str(Path(android_world_root).expanduser().resolve()))
    repo = Path(args.repo).expanduser().resolve()
    code_root = Path(args.code_root or repo).expanduser().resolve()
    checks: list[Check] = []
    requested_profile = str(args.profile or "").strip()
    appagent_mode = requested_profile == "appagent"
    native_mode = requested_profile == "androidworld_native"
    profile = (
        "appagent"
        if appagent_mode
        else "androidworld_native"
        if native_mode
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
            or os.getenv(
                "OMNIFLOW_APPAGENT_ROOT", str(repo / "runtime/external/appagent")
            )
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
    source_memory_value = str(
        args.source_memory_root
        or os.getenv("OMNIFLOW_MOBILEGPT_SOURCE_MEMORY_ROOT", "")
    ).strip()
    memory_root = (
        Path(source_memory_value).expanduser().resolve()
        if source_memory_value
        else None
    )
    memory_files: list[Path] = []
    memory_tasks: list[Path] = []
    if appagent_mode:
        memory_root = None
        demo_memory_value = str(
            args.appagent_memory_root
            or os.getenv("OMNIFLOW_APPAGENT_MEMORY_ROOT", "")
        ).strip()
        if not demo_memory_value:
            add("appagent_memory", True, "not supplied for source preparation")
        else:
            demo_memory = Path(demo_memory_value).expanduser().resolve()
            manifest = demo_memory / "appagent_manifest.json"
            valid = False
            detail = str(manifest)
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                valid = is_memory_manifest_valid(payload)
                detail = f"sealed manifest at {manifest}"
            except (OSError, ValueError, json.JSONDecodeError) as error:
                detail = str(error)
            add("appagent_memory", valid, detail)
            if valid:
                memory_files = [
                    path
                    for path in demo_memory.rglob("*")
                    if path.is_file() and "__pycache__" not in path.parts
                ]
                memory_root = demo_memory
    elif native_mode:
        memory_root = None
        expected_tasks = args.expected_tasks
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
                    task_names=tuple(args.source_task),
                    allow_historical_source=args.source_only,
                )
            except (OSError, ValueError, json.JSONDecodeError) as error:
                add("source_index", False, str(error))
            else:
                add(
                    "source_index",
                    expected_tasks is None
                    or source_validation["task_count"] == expected_tasks,
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
        source_index = repo / "data" / "current.json"
        try:
            source_validation = _validate_source_index(
                source_index,
                source_root=repo / "data",
                expected_tasks=expected_tasks,
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            add("source_index", False, str(error))
        else:
            add(
                "source_index",
                True,
                json.dumps(
                    {
                        "tasks": source_validation["task_count"],
                        "run_logs": source_validation["run_log_count"],
                        "sha256": source_validation["index_sha256"],
                    },
                    sort_keys=True,
                ),
            )

        if memory_root is None:
            add("initial_memory", True, "empty_memory")
        else:
            memory_files, memory_tasks = _collect_memory_files(memory_root)
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
                mobilegpt_manifest = validate_memory_manifest(memory_root)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                add("mobilegpt_memory_manifest", False, str(error))
            else:
                add(
                    "mobilegpt_memory_manifest",
                    True,
                    json.dumps(mobilegpt_manifest, sort_keys=True),
                )

    disk = shutil.disk_usage(repo if repo.exists() else Path.home())
    free_gb = disk.free / (1024 ** 3)
    add("disk_free", free_gb >= args.minimum_free_gb, f"{free_gb:.2f} GiB free; require {args.minimum_free_gb:.2f}")
    add("python", sys.version_info >= (3, 11), platform.python_version())
    module_names = [
        "absl",
        "android_env.proto.a11y",
        "android_world.env.android_world_controller",
        "android_world.registry",
        "dotenv",
        "grpc",
        "json_repair",
        "numpy",
        "openai",
        "pandas",
    ]
    if appagent_mode:
        module_names.extend(APPAGENT_REQUIRED_MODULES)
    module_names.append("uiautomator2")
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
        except ImportError as error:
            add(f"python_module:{module_name}", False, str(error))
        else:
            add(f"python_module:{module_name}", True, "importable")
    for distribution_name, expected_version in REQUIRED_DISTRIBUTION_VERSIONS.items():
        try:
            installed_version = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            add(
                f"python_distribution:{distribution_name}",
                False,
                f"missing; require {expected_version}",
            )
        else:
            add(
                f"python_distribution:{distribution_name}",
                installed_version == expected_version,
                f"{installed_version}; require {expected_version}",
            )
    add("jq", True, f"not required by {profile} profile")
    add("java", bool(shutil.which("java")), shutil.which("java") or "missing")
    model_key = (
        os.getenv("LLMTHU_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    add("model_key", bool(model_key), "configured" if model_key else "missing")
    if appagent_mode or native_mode:
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
            focused_windows = _dismiss_known_accessibility_crash_dialog(
                adb,
                args.serial,
                focused_windows,
            )
            crash_dialog_present = _system_crash_dialog_present(focused_windows)
            add(
                "system_crash_dialog",
                not crash_dialog_present,
                "present" if crash_dialog_present else "none",
            )
            if appagent_mode:
                required_packages = (
                    ("com.google.android.contacts",)
                    if args.require_contacts_ready
                    else ()
                )
            elif native_mode:
                required_packages = (
                    ("com.google.android.contacts",)
                    if args.require_contacts_ready
                    else ()
                )
            else:
                required_packages = (
                    ("com.google.android.contacts",)
                    if args.require_contacts_ready
                    else ()
                )
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
                contacts_task_reset = False
                if not _contacts_setup_ready(screen):
                    contacts_task_reset = True
                    screen = _reset_contacts_setup_screen(adb, args.serial)
                contacts_ready = _contacts_setup_ready(screen)
                contacts_detail = (
                    "contacts home after task reset"
                    if contacts_task_reset and _contacts_home_ready(screen)
                    else "contacts onboarding ready after task reset"
                    if contacts_task_reset and contacts_ready
                    else "contacts home"
                    if _contacts_home_ready(screen)
                    else "contacts onboarding ready"
                    if contacts_ready
                    else "unknown contacts screen"
                )
                add("contacts_ready", contacts_ready, contacts_detail)
                _run([adb, "-s", args.serial, "shell", "am", "force-stop", "com.google.android.contacts"], timeout=5)

    failures = [check for check in checks if check.status == "fail"]
    warnings = [check for check in checks if check.status == "warning"]
    fingerprint_files = memory_files
    fingerprint_root = repo if memory_root is None else memory_root
    memory_provider = (
        "appagent"
        if appagent_mode and memory_root is not None
        else "mobilegpt"
        if memory_root is not None
        else "none"
    )
    memory_condition = (
        "prepared_memory"
        if memory_root is not None
        else "empty_memory"
    )
    report = {
        "schema_version": "omniflow.run-check.v2",
        "ready": not failures,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "paths": {
            "repo": str(repo),
            "code_root": str(code_root),
        },
        "target": {
            "serial": args.serial,
            "profile": profile,
        },
        "memory": {
            "provider": memory_provider,
            "condition": memory_condition,
            "root": str(memory_root or ""),
            "fingerprint": (
                _hash_files(fingerprint_files, relative_to=fingerprint_root)
                if fingerprint_files
                else ""
            ),
        },
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
