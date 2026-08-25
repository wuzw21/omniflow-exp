#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
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
import tempfile
import time
from typing import Any

from omniflow.core.trajectory import canonicalize_run_log as import_run_log
from src.integrations.appagent import is_memory_manifest_valid
from src.integrations.android_world.oob_control import (
    CONTROL_ACCESSIBILITY_SERVICE as OOB_ACCESSIBILITY_SERVICE,
    CONTROL_PACKAGE as OOB_PACKAGE,
    OobControlClient,
)
from src.integrations.mobilegpt import validate_memory_manifest

APPAGENT_OFFICIAL_REVISION = os.environ.get(
    "OMNIFLOW_APPAGENT_REVISION"
) or "2c1900422caf6f9e94e96d5dd984b530e5a5fbf8"
APPAGENT_REQUIRED_MODULES = (
    "colorama",
    "cv2",
    "dashscope",
    "pyshine",
    "requests",
    "yaml",
)
REQUIRED_DISTRIBUTION_VERSIONS = {"android-env": "1.2.3"}

# OOB is the only physical observation/action service in formal AndroidWorld
# experiments.  The legacy AndroidWorld forwarder and MobileGPT client may be
# installed as dependency artifacts, but preflight must actively remove them
# from the enabled service list so they cannot contend with OOB.
LEGACY_ACCESSIBILITY_SERVICES = (
    "com.google.androidenv.accessibilityforwarder/com.google.androidenv.accessibilityforwarder.AccessibilityForwarder",
    "com.example.MobileGPT/.MobileGPTAccessibilityService",
)
MANAGED_ACCESSIBILITY_SERVICES = (
    OOB_ACCESSIBILITY_SERVICE,
    *LEGACY_ACCESSIBILITY_SERVICES,
)
DEFAULT_ACCESSIBILITY_SERVICES = (OOB_ACCESSIBILITY_SERVICE,)
OOB_ACTIVITY = "cn.com.omnimind.bot.activity.LauncherActivity"


@dataclass
class Check:
    name: str
    status: str
    detail: str


@dataclass
class IntegrationCheck:
    method: str
    name: str
    status: str
    detail: str
    remediation: str = ""


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


def _root_access(adb: str, serial: str) -> tuple[bool, str]:
    """Return whether the selected device can execute a root shell command."""

    direct = _run([adb, "-s", serial, "shell", "id"], timeout=10).stdout.strip()
    if re.search(r"\buid=0(?:\(|\s|$)", direct):
        return True, f"direct root: {direct}"
    via_su = _run(
        [adb, "-s", serial, "shell", "su", "-c", "id"],
        timeout=10,
    ).stdout.strip()
    if re.search(r"\buid=0(?:\(|\s|$)", via_su):
        return True, f"su root: {via_su}"
    return False, via_su or direct or "root shell unavailable"


def _installed_accessibility_services(adb: str, serial: str) -> tuple[str, ...]:
    """Keep only known services whose APK/service is present on this device."""

    installed: list[str] = []
    package_dumps: dict[str, str] = {}
    for component in DEFAULT_ACCESSIBILITY_SERVICES:
        package, service = component.split("/", 1)
        service_name = service.rsplit(".", 1)[-1]
        package_dump = package_dumps.setdefault(
            package,
            _run(
                [adb, "-s", serial, "shell", "dumpsys", "package", package],
                timeout=10,
            ).stdout,
        )
        if service in package_dump or service_name in package_dump:
            installed.append(component)
    return tuple(installed)


def configure_default_device_services(
    adb: str,
    serial: str,
    *,
    profile: str = "oob",
) -> dict[str, object]:
    """Configure the one physical Accessibility owner for a formal profile.

    The operation is idempotent and preserves unrelated user-enabled
    services. It intentionally does not install APKs or modify provider source
    code. MobileGPT uses its official Accessibility client; other profiles
    retain the canonical OOB service.
    """

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
    ).stdout.strip()
    current_enabled = [
        value
        for value in current.split(":")
        if value and value != "null"
    ]
    managed_identities = {
        (
            component.split("/", 1)[0],
            component.rsplit(".", 1)[-1],
        )
        for component in MANAGED_ACCESSIBILITY_SERVICES
    }
    enabled = [
        value
        for value in current_enabled
        if (
            value.split("/", 1)[0],
            value.rsplit(".", 1)[-1],
        )
        not in managed_identities
    ]
    installed = _installed_accessibility_services(adb, serial)
    normalized_profile = str(profile or "oob").strip().lower()
    desired_services = (
        ("com.example.MobileGPT/.MobileGPTAccessibilityService",)
        if normalized_profile == "mobilegpt"
        else DEFAULT_ACCESSIBILITY_SERVICES
    )
    for component in installed:
        if component in desired_services and component not in enabled:
            enabled.append(component)
    for component in desired_services:
        if component in installed and component not in enabled:
            enabled.append(component)
    enabled = list(dict.fromkeys(enabled))
    value = ":".join(enabled)
    result = _run(
        [
            adb,
            "-s",
            serial,
            "shell",
            "settings",
            "put",
            "secure",
            "enabled_accessibility_services",
            value,
        ],
        timeout=10,
    )
    enabled_flag = _run(
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
        timeout=10,
    )
    confirmed = _run(
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
    confirmed_enabled = {
        item
        for item in confirmed.stdout.strip().split(":")
        if item and item != "null"
    }
    accessibility_dump = _run(
        [adb, "-s", serial, "shell", "dumpsys", "accessibility"],
        timeout=10,
    ).stdout
    crashed_section = (
        accessibility_dump.split("Crashed services:", 1)[1].split(
            "Client list info:",
            1,
        )[0]
        if "Crashed services:" in accessibility_dump
        else ""
    )
    service_health = {
        component: (
            component in confirmed_enabled
            and component.rsplit(".", 1)[-1] not in crashed_section
        )
        for component in installed
    }
    return {
        "installed": list(installed),
        "enabled": enabled,
        "service_health": service_health,
        "settings_write_ok": (
            result.returncode == 0
            and enabled_flag.returncode == 0
            and confirmed.returncode == 0
            and all(service_health.values())
        ),
    }


def ensure_oob_device_ready(
    adb: str,
    serial: str,
    *,
    timeout_seconds: float = 30.0,
    repair: bool = True,
) -> dict[str, object]:
    """Require the complete OOB reset/observe contract before an episode."""

    def probe() -> dict[str, object]:
        client = OobControlClient(
            object(),
            adb_serial=serial,
            adb_path=adb,
            timeout_seconds=timeout_seconds,
        )
        client.reset()
        state = client.observe(wait_to_stabilize=True)
        xml = str(state.get("xml") or "") if isinstance(state, dict) else ""
        if not xml.strip():
            raise RuntimeError("oob_ready_observe_xml_missing")
        return {
            "xml_chars": len(xml),
            "package_name": str(state.get("package_name") or ""),
        }

    try:
        observed = probe()
        return {
            "ready": True,
            "repaired": False,
            **observed,
        }
    except Exception as initial_error:  # noqa: BLE001
        initial_detail = str(initial_error)
        if not repair:
            return {
                "ready": False,
                "repaired": False,
                "error": initial_detail,
            }

    commands = (
        [adb, "-s", serial, "shell", "am", "force-stop", OOB_PACKAGE],
        [
            adb,
            "-s",
            serial,
            "shell",
            "am",
            "start",
            "-n",
            f"{OOB_PACKAGE}/{OOB_ACTIVITY}",
        ],
        [adb, "-s", serial, "shell", "input", "keyevent", "HOME"],
    )
    repair_commands: list[dict[str, object]] = []
    for command in commands:
        result = _run(command, timeout=15)
        repair_commands.append(
            {
                "operation": " ".join(command[4:]),
                "returncode": int(result.returncode),
            }
        )
        if result.returncode != 0:
            return {
                "ready": False,
                "repaired": True,
                "initial_error": initial_detail,
                "error": (result.stdout or "oob_repair_command_failed").strip(),
                "repair_commands": repair_commands,
            }
    configured = configure_default_device_services(adb, serial)
    time.sleep(2)
    repair_deadline = time.monotonic() + max(1.0, float(timeout_seconds))
    while True:
        try:
            observed = probe()
            break
        except Exception as final_error:  # noqa: BLE001
            if time.monotonic() >= repair_deadline:
                return {
                    "ready": False,
                    "repaired": True,
                    "initial_error": initial_detail,
                    "error": str(final_error),
                    "repair_commands": repair_commands,
                    "device_services": configured,
                }
            time.sleep(min(1.0, max(0.0, repair_deadline - time.monotonic())))
    return {
        "ready": True,
        "repaired": True,
        "initial_error": initial_detail,
        "repair_commands": repair_commands,
        "device_services": configured,
        **observed,
    }


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
        run_log_value = str(
            metadata.get("retained_source_run_log")
            or metadata.get("source_run_log")
            or ""
        ).strip()
        if allow_historical_source and not run_log_value:
            if (
                source_kind
                not in {
                    "",
                    "pending_source_recollection",
                    "one_time_canonicalized_seed111_screenshot_source",
                }
                or not str(metadata.get("goal") or "").strip()
                or not isinstance(metadata.get("params"), dict)
            ):
                invalid.append(str(task))
            continue
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


def _integration_add(
    checks: list[IntegrationCheck],
    method: str,
    name: str,
    status: str,
    detail: str,
    remediation: str = "",
) -> None:
    checks.append(
        IntegrationCheck(
            method=method,
            name=name,
            status=status,
            detail=str(detail),
            remediation=str(remediation),
        )
    )


def _integration_file_check(
    checks: list[IntegrationCheck],
    method: str,
    path: Path,
    *,
    label: str,
    remediation: str,
) -> bool:
    present = path.is_dir() if label == "official_root" else path.is_file()
    _integration_add(
        checks,
        method,
        label,
        "pass" if present else "fail",
        str(path),
        "" if present else remediation,
    )
    return present


def _integration_python_check(
    checks: list[IntegrationCheck],
    method: str,
    path: Path,
) -> None:
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as error:
        _integration_add(
            checks,
            method,
            f"python_syntax:{path.name}",
            "fail",
            str(error),
            "Fix the syntax in the disposable integration input before running E2E.",
        )
    else:
        _integration_add(
            checks,
            method,
            f"python_syntax:{path.name}",
            "pass",
            "AST parse succeeded",
        )


def _integration_model_config(
    args: argparse.Namespace,
    checks: list[IntegrationCheck],
) -> dict[str, str]:
    from omniflow.vlm.model_config import resolve_openai_compatible_config
    from src.experiment.protocol import (
        FORMAL_MODEL,
        FORMAL_MODEL_BASE_URL,
        FORMAL_MODEL_ENDPOINT_PROFILE,
    )

    model = str(
        args.integration_model
        or os.environ.get("OPENAI_MODEL")
        or FORMAL_MODEL
    ).strip()
    embedding_model = str(
        args.embedding_model
        or os.environ.get("MOBILEGPT_EMBEDDING_MODEL")
        or "GLM-Embedding-2"
    ).strip()
    profile = str(
        args.model_endpoint_profile
        or os.environ.get("OMNIFLOW_MODEL_ENDPOINT_PROFILE")
        or FORMAL_MODEL_ENDPOINT_PROFILE
    ).strip()
    base_url = str(
        args.model_base_url
        or os.environ.get("OPENAI_BASE_URL")
        or FORMAL_MODEL_BASE_URL
    ).strip()
    environment = dict(os.environ)
    if base_url:
        environment.setdefault("OPENAI_BASE_URL", base_url)
    try:
        api_key, resolved_base_url = resolve_openai_compatible_config(
            profile=profile,
            base_url=base_url,
            environment=environment,
        )
    except ValueError as error:
        api_key, resolved_base_url = None, None
        _integration_add(
            checks,
            "shared",
            "model_endpoint",
            "fail",
            str(error),
            "Source the model.env used by the launcher and select a complete endpoint profile.",
        )
    else:
        _integration_add(
            checks,
            "shared",
            "model_endpoint",
            "pass" if api_key and resolved_base_url else "fail",
            (
                f"profile={profile} base_url={resolved_base_url or 'missing'} "
                f"api_key={'configured' if api_key else 'missing'}"
            ),
            "Set the endpoint API key and base URL; the key value is never printed.",
        )
    allowed_models = {"glm-4.6v", "glm-5.1"}
    model_ok = model.lower() in allowed_models
    _integration_add(
        checks,
        "shared",
        "chat_model",
        "pass" if model_ok else "fail",
        model or "missing",
        "Use GLM-4.6V or GLM-5.1 for both formal integrations.",
    )
    supported_embedding_models = {"GLM-Embedding-2", "text-embedding-v4"}
    embedding_ok = embedding_model in supported_embedding_models
    _integration_add(
        checks,
        "mobilegpt",
        "embedding_model",
        "pass" if embedding_ok else "fail",
        embedding_model or "missing",
        "Set MOBILEGPT_EMBEDDING_MODEL to the embedding model available at the selected endpoint.",
    )
    return {
        "model": model,
        "embedding_model": embedding_model,
        "profile": profile,
        "api_key": str(api_key or ""),
        "base_url": str(resolved_base_url or base_url),
    }


def _find_mobilegpt_global(root: Path) -> Path | None:
    candidates = sorted(root.rglob("MobileGPTGlobal.java"))
    return candidates[0] if candidates else None


def _run_mobilegpt_integration_checks(
    checks: list[IntegrationCheck],
    *,
    repo: Path,
    root: Path,
    memory_root: Path | None,
    config: dict[str, str],
    server_port: int,
) -> None:
    method = "mobilegpt"
    required = {
        "official_root": root,
        "server_entry": root / "Server" / "main.py",
        "server_socket": root / "Server" / "server.py",
        "server_utils": root / "Server" / "utils" / "utils.py",
    }
    for label, path in required.items():
        _integration_file_check(
            checks,
            method,
            path,
            label=label,
            remediation="Point OMNIFLOW_MOBILEGPT_ROOT to the unmodified official MobileGPT checkout.",
        )
    global_java = _find_mobilegpt_global(root)
    _integration_add(
        checks,
        method,
        "android_client_config",
        "pass" if global_java else "fail",
        str(global_java or root / "App/**/MobileGPTGlobal.java"),
        "The official Android client must contain MobileGPTGlobal.java with HOST_IP/HOST_PORT.",
    )
    source_files = [path for path in required.values() if path.suffix == ".py" and path.is_file()]
    for path in source_files:
        _integration_python_check(checks, method, path)
    if global_java is not None:
        text = global_java.read_text(encoding="utf-8", errors="replace")
        host_ok = bool(re.search(r"HOST_IP", text))
        port_ok = bool(re.search(r"HOST_PORT", text))
        _integration_add(
            checks,
            method,
            "android_client_host_port_fields",
            "pass" if host_ok and port_ok else "fail",
            f"HOST_IP={'present' if host_ok else 'missing'} HOST_PORT={'present' if port_ok else 'missing'}",
            "Keep the official client fields; the runner patches them only in its disposable APK staging flow.",
        )
    wiring = (repo / "src" / "integrations" / "official_forward.py").read_text(
        encoding="utf-8", errors="replace"
    ) if (repo / "src" / "integrations" / "official_forward.py").is_file() else ""
    run_task = (repo / "src" / "experiment" / "run_task.py").read_text(
        encoding="utf-8", errors="replace"
    ) if (repo / "src" / "experiment" / "run_task.py").is_file() else ""
    wiring_ok = all(
        marker in wiring for marker in ("prepare_mobilegpt_server", "run_mobilegpt_client")
    ) and "build_mobilegpt_server_command" in run_task
    _integration_add(
        checks,
        method,
        "pipeline_wiring",
        "pass" if wiring_ok else "fail",
        "official_forward + run_task MobileGPT server/client seam",
        "Restore the single official_forward/run_task integration seam.",
    )
    if memory_root is not None:
        try:
            manifest = validate_memory_manifest(memory_root)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            _integration_add(
                checks,
                method,
                "memory_bundle",
                "fail",
                str(error),
                "Regenerate or select a complete MobileGPT memory bundle; do not hand-edit its manifest.",
            )
        else:
            _integration_add(
                checks,
                method,
                "memory_bundle",
                "pass",
                json.dumps(manifest, sort_keys=True),
            )
    else:
        _integration_add(
            checks,
            method,
            "memory_bundle",
            "warning",
            "not supplied; static integration check only",
            "Pass --mobilegpt-memory-root for a warm-run readiness check.",
        )
    if not root.is_dir():
        return
    _integration_add(
        checks,
        method,
        "server_port_free",
        "pass" if _port_is_free(server_port) else "fail",
        f"127.0.0.1:{server_port}",
        "Stop the stale MobileGPT server or choose the protocol port before starting an E2E run.",
    )
    try:
        with tempfile.TemporaryDirectory(prefix="omniflow-mobilegpt-check-") as temporary:
            temporary_root = Path(temporary)
            staged_memory = temporary_root / "memory"
            staged_memory.mkdir()
            staged = temporary_root / "workspace"
            from src.integrations.official_forward import prepare_mobilegpt_server

            result = prepare_mobilegpt_server(
                official_root=root,
                memory_root=memory_root or staged_memory,
                workspace=staged,
                embedding_model=config["embedding_model"],
                chat_model=config["model"],
            )
            staged_server = Path(result["server_root"])
            staged_text = "\n".join(
                path.read_text(encoding="utf-8", errors="replace")
                for path in staged_server.rglob("*.py")
            )
            for path in staged_server.rglob("*.py"):
                _integration_python_check(checks, method, path)
            stage_ok = (
                config["embedding_model"] in staged_text
                and "MOBILEGPT_EMBEDDING_MODEL" in staged_text
                and config["model"] in staged_text
                and "MOBILEGPT_CHAT_MODEL" in staged_text
                and "text-embedding-3-small" not in staged_text
            )
            _integration_add(
                checks,
                method,
                "disposable_server_config",
                "pass" if stage_ok else "fail",
                f"staged_server={staged_server} port={server_port}",
                "The disposable Server must route chat to the selected GLM and embeddings to the selected endpoint model.",
            )
    except Exception as error:
        _integration_add(
            checks,
            method,
            "disposable_server_config",
            "fail",
            f"{type(error).__name__}: {error}",
            "Fix the official MobileGPT root or the disposable staging seam before E2E.",
        )


def _run_appagent_integration_checks(
    checks: list[IntegrationCheck],
    *,
    repo: Path,
    root: Path,
    memory_root: Path | None,
    config: dict[str, str],
    serial: str,
    adb_path: str,
) -> None:
    method = "appagent"
    required = {
        "official_root": root,
        "official_entry": root / "run.py",
        "official_executor": root / "scripts" / "task_executor.py",
        "official_model": root / "scripts" / "model.py",
        "official_controller": root / "scripts" / "and_controller.py",
        "official_config": root / "config.yaml",
    }
    for label, path in required.items():
        _integration_file_check(
            checks,
            method,
            path,
            label=label,
            remediation="Point OMNIFLOW_APPAGENT_ROOT to the pinned official AppAgent checkout.",
        )
    revision = _run(["git", "-C", str(root), "rev-parse", "HEAD"], timeout=10)
    actual_revision = revision.stdout.strip()
    _integration_add(
        checks,
        method,
        "official_revision",
        "pass" if actual_revision == APPAGENT_OFFICIAL_REVISION else "fail",
        actual_revision or "unavailable",
        f"Checkout the pinned AppAgent revision {APPAGENT_OFFICIAL_REVISION}.",
    )
    for path in required.values():
        if path.suffix == ".py" and path.is_file():
            _integration_python_check(checks, method, path)
    wiring = (repo / "src" / "integrations" / "official_forward.py").read_text(
        encoding="utf-8", errors="replace"
    ) if (repo / "src" / "integrations" / "official_forward.py").is_file() else ""
    run_task = (repo / "src" / "experiment" / "run_task.py").read_text(
        encoding="utf-8", errors="replace"
    ) if (repo / "src" / "experiment" / "run_task.py").is_file() else ""
    wiring_ok = all(
        marker in wiring
        for marker in ("prepare_appagent_workspace", "--baseline", "appagent")
    ) and "build_appagent_command" in run_task
    _integration_add(
        checks,
        method,
        "pipeline_wiring",
        "pass" if wiring_ok else "fail",
        "official_forward + run_task AppAgent executor seam",
        "Restore the single official_forward/run_task AppAgent integration seam.",
    )
    docs_root: Path | None = None
    if memory_root is not None:
        manifest_path = memory_root / "appagent_manifest.json"
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            valid = is_memory_manifest_valid(payload)
            docs_value = str(payload.get("demo_docs_root") or "").strip()
            docs_root = Path(docs_value).expanduser().resolve() if docs_value else None
            valid = valid and docs_root is not None and docs_root.is_dir()
            detail = f"manifest={manifest_path} docs={docs_root or 'missing'}"
        except (OSError, ValueError, json.JSONDecodeError) as error:
            valid = False
            detail = str(error)
        _integration_add(
            checks,
            method,
            "memory_bundle",
            "pass" if valid else "fail",
            detail,
            "Regenerate or select a complete AppAgent memory bundle; do not hand-edit its manifest.",
        )
    else:
        _integration_add(
            checks,
            method,
            "memory_bundle",
            "warning",
            "not supplied; static integration check only",
            "Pass --appagent-memory-root for a warm-run readiness check.",
        )
    if not root.is_dir():
        return
    try:
        with tempfile.TemporaryDirectory(prefix="omniflow-appagent-check-") as temporary:
            temporary_root = Path(temporary)
            if docs_root is None:
                docs_root = temporary_root / "apps" / "Contacts" / "demo_docs"
                docs_root.mkdir(parents=True)
                (docs_root / "probe.txt").write_text("probe\n", encoding="utf-8")
            staged = temporary_root / "workspace"
            from src.integrations.official_forward import prepare_appagent_workspace

            result = prepare_appagent_workspace(
                official_root=root,
                docs_root=docs_root,
                workspace=staged,
                app_name=docs_root.parent.name,
                serial=serial,
                adb_path=adb_path or "adb",
                config={
                    "MODEL": "OpenAI",
                    "OPENAI_API_BASE": config["base_url"].rstrip("/") + "/chat/completions",
                    "OPENAI_API_KEY": config["api_key"] or "not-required",
                    "OPENAI_API_MODEL": config["model"],
                    "MAX_TOKENS": 1024,
                    "TEMPERATURE": 0.0,
                    "REQUEST_INTERVAL": 0.0,
                    "DARK_MODE": False,
                    "MIN_DIST": 30,
                },
            )
            staged_executor = Path(result["workspace"]) / "scripts" / "task_executor.py"
            executor_text = staged_executor.read_text(encoding="utf-8", errors="replace")
            executor_ok = (
                "_omniflow_resolution_agnostic_doc_path" in executor_text
                and "os.path.join(docs_dir, f\"{elem.uid}.txt\")" not in executor_text
            )
            _integration_add(
                checks,
                method,
                "disposable_executor_config",
                "pass" if executor_ok else "fail",
                f"staged_executor={staged_executor} model={config['model']}",
                "The disposable executor must use the resolution-agnostic document lookup and the selected GLM model.",
            )
            _integration_python_check(checks, method, staged_executor)
            proxy = Path(result["adb_proxy"])
            _integration_add(
                checks,
                method,
                "adb_proxy",
                "pass" if proxy.is_file() and os.access(proxy, os.X_OK) else "fail",
                str(proxy),
                "The official executor must receive the selected device through the disposable ADB proxy.",
            )
    except Exception as error:
        _integration_add(
            checks,
            method,
            "disposable_executor_config",
            "fail",
            f"{type(error).__name__}: {error}",
            "Fix the official AppAgent root, docs root, or disposable workspace seam before E2E.",
        )


def run_integration_checks(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(args.repo).expanduser().resolve()
    checks: list[IntegrationCheck] = []
    config = _integration_model_config(args, checks)
    selected = str(args.integration_method or "all").strip().lower()
    if selected in {"all", "mobilegpt"}:
        root_value = args.mobilegpt_root or os.environ.get("OMNIFLOW_MOBILEGPT_ROOT", "")
        root = Path(root_value).expanduser().resolve() if root_value else repo / "runtime" / "external" / "mobilegpt"
        memory_value = args.mobilegpt_memory_root or os.environ.get("OMNIFLOW_MOBILEGPT_SOURCE_MEMORY_ROOT", "")
        memory = Path(memory_value).expanduser().resolve() if memory_value else None
        _run_mobilegpt_integration_checks(
            checks,
            repo=repo,
            root=root,
            memory_root=memory,
            config=config,
            server_port=int(args.server_port),
        )
    if selected in {"all", "appagent"}:
        root_value = args.appagent_root or os.environ.get("OMNIFLOW_APPAGENT_ROOT", "")
        root = Path(root_value).expanduser().resolve() if root_value else repo / "runtime" / "external" / "appagent"
        memory_value = args.appagent_memory_root or os.environ.get("OMNIFLOW_APPAGENT_MEMORY_ROOT", "")
        memory = Path(memory_value).expanduser().resolve() if memory_value else None
        adb_path = str(os.environ.get("OMNIFLOW_REAL_ADB_PATH") or _resolve_tool("adb", "platform-tools"))
        _run_appagent_integration_checks(
            checks,
            repo=repo,
            root=root,
            memory_root=memory,
            config=config,
            serial=str(args.serial),
            adb_path=adb_path,
        )
    if getattr(args, "require_device", False):
        adb = str(os.environ.get("OMNIFLOW_REAL_ADB_PATH") or _resolve_tool("adb", "platform-tools"))
        adb_ready = bool(adb and Path(adb).is_file() and os.access(adb, os.X_OK))
        _integration_add(
            checks,
            "shared",
            "adb",
            "pass" if adb_ready else "fail",
            adb or "missing",
            "Set OMNIFLOW_REAL_ADB_PATH to the real platform-tools/adb binary.",
        )
        if adb_ready:
            devices = _run([adb, "devices"], timeout=10).stdout
            device_ready = any(
                line.split()[:2] == [str(args.serial), "device"]
                for line in devices.splitlines()
            )
            _integration_add(
                checks,
                "shared",
                "device",
                "pass" if device_ready else "fail",
                str(args.serial),
                "Start the selected emulator and wait for adb state=device.",
            )
            if device_ready:
                boot = _run(
                    [adb, "-s", str(args.serial), "shell", "getprop", "sys.boot_completed"],
                    timeout=10,
                ).stdout.strip()
                _integration_add(
                    checks,
                    "shared",
                    "boot_completed",
                    "pass" if boot == "1" else "fail",
                    boot or "empty",
                    "Wait for Android sys.boot_completed=1 before launching either official method.",
                )
    if selected not in {"all", "mobilegpt", "appagent"}:
        _integration_add(
            checks,
            "shared",
            "integration_method",
            "fail",
            selected,
            "Use --integration-method mobilegpt, appagent, or all.",
        )
    failures = [item for item in checks if item.status == "fail"]
    warnings = [item for item in checks if item.status == "warning"]
    return {
        "schema_version": "omniflow.integration-check.v1",
        "ready": not failures,
        "checks": [asdict(item) for item in checks],
        "summary": {
            "pass": sum(item.status == "pass" for item in checks),
            "warning": len(warnings),
            "fail": len(failures),
        },
        "contract": {
            "model_calls": 0,
            "official_source_modified": False,
            "staging_only": True,
        },
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
    parser.add_argument(
        "--require-root",
        action="store_true",
        help="Require a root-capable device (direct root or su root).",
    )
    parser.add_argument(
        "--configure-device",
        action="store_true",
        help="Enable every installed repository accessibility service.",
    )
    parser.add_argument("--require-contacts-ready", action="store_true")
    parser.add_argument("--json-out")
    parser.add_argument(
        "--integration-check",
        action="store_true",
        help="Check the real MobileGPT/AppAgent integration seams using disposable staging only.",
    )
    parser.add_argument(
        "--integration-method",
        choices=["all", "mobilegpt", "appagent"],
        default="all",
    )
    parser.add_argument("--mobilegpt-root")
    parser.add_argument("--mobilegpt-memory-root")
    parser.add_argument("--integration-model")
    parser.add_argument("--embedding-model")
    parser.add_argument("--model-endpoint-profile")
    parser.add_argument("--model-base-url")
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
            "src/integrations/mobilegpt_format.py",
            "runtime/external/mobilegpt/Server/main.py",
            "runtime/external/droidrun-android-world/android_world/android_world/env/setup_device/apps.py",
        ]
    raise ValueError(f"unsupported_preflight_profile:{profile}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.integration_check:
        report = run_integration_checks(args)
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        print(rendered)
        if args.json_out:
            output = Path(args.json_out).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered + "\n", encoding="utf-8")
        return 0 if report["ready"] else 1
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

    appagent_root = Path(
        args.appagent_root
        or os.getenv(
            "OMNIFLOW_APPAGENT_ROOT", str(repo / "runtime/external/appagent")
        )
    ).expanduser().resolve()
    mobilegpt_root = Path(
        os.getenv(
            "OMNIFLOW_MOBILEGPT_ROOT", str(repo / "runtime/external/mobilegpt")
        )
    ).expanduser().resolve()
    android_world_path = (
        Path(android_world_root).expanduser().resolve()
        if android_world_root
        else repo / "runtime/external/droidrun-android-world"
    )

    def required_path(relative: str) -> Path:
        if relative == "runtime/external/appagent/scripts/document_generation.py":
            return appagent_root / "scripts/document_generation.py"
        if relative == "runtime/external/mobilegpt/Server/main.py":
            return mobilegpt_root / "Server/main.py"
        if (
            relative
            == "runtime/external/droidrun-android-world/android_world/"
            "android_world/env/setup_device/apps.py"
        ):
            return android_world_path / "android_world/env/setup_device/apps.py"
        root = repo if relative.startswith("runtime/") else code_root
        return root / relative

    def add(name: str, passed: bool, detail: str, warning: bool = False) -> None:
        checks.append(Check(name, "ok" if passed else "warning" if warning else "fail", detail))

    for relative in _required_files(profile):
        add(f"file:{relative}", required_path(relative).is_file(), relative)
    if appagent_mode:
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
        source_index_value = str(args.source_index or "").strip()
        source_index = (
            Path(source_index_value).expanduser().resolve()
            if source_index_value
            else repo / "data" / "current.json"
        )
        try:
            source_validation = _validate_source_index(
                source_index,
                source_root=Path(args.source_root or source_index.parent)
                .expanduser()
                .resolve(),
                expected_tasks=expected_tasks,
                task_names=tuple(args.source_task),
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
            if args.require_root or args.configure_device:
                root_ready, root_detail = _root_access(adb, args.serial)
                if args.require_root:
                    add("root_access", root_ready, root_detail)
                elif not root_ready:
                    add(
                        "root_access",
                        True,
                        f"not required; configure skipped: {root_detail}",
                        warning=True,
                    )
                if args.configure_device:
                    if not root_ready and args.require_root:
                        add(
                            "device_services",
                            False,
                            "root-capable device required before service configuration",
                        )
                    else:
                        configured = configure_default_device_services(
                            adb,
                            args.serial,
                            profile="mobilegpt" if profile == "mobilegpt" else "oob",
                        )
                        add(
                            "device_services",
                            bool(configured["settings_write_ok"]),
                            json.dumps(configured, sort_keys=True),
                        )
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
