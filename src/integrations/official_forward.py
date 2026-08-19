"""Small boundary for launching the pinned external baselines.

This module deliberately does not know how an external baseline plans or
executes an action.  It only makes the official checkout look like the
official README expects.  AutoDroid receives its original DroidBot memory and
is launched through the original ``droidbot.start`` replay entrypoint.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import time
from typing import Any, Iterator, Sequence

from src.experiment.autodroid_contract import (
    AUTODROID_MEMORY_MANIFEST_FORMAT,
    AUTODROID_RESULT_SCHEMA,
)


@contextmanager
def _androidworld_task_startup(
    *,
    android_world_root: str | Path,
    task_name: str,
    task_params_json: str,
    task_seed: int,
    console_port: int,
    grpc_port: int,
    adb_path: str,
    perform_emulator_setup: bool,
) -> Iterator[tuple[Any, Any]]:
    """Prepare one official task through the canonical AndroidWorld seam."""

    from src.integrations.android_world.run_episode import (
        start_androidworld_task_session,
    )

    decoded = json.loads(str(task_params_json or "{}"))
    if not isinstance(decoded, dict):
        raise ValueError("androidworld_task_params_must_be_object")
    startup, task = start_androidworld_task_session(
        android_world_root=android_world_root,
        task_name=task_name,
        task_params=decoded,
        task_seed=int(task_seed),
        console_port=int(console_port),
        adb_path=adb_path,
        grpc_port=int(grpc_port),
        perform_emulator_setup=bool(perform_emulator_setup),
        use_uiautomator=True,
    )
    try:
        yield startup.env, task
    finally:
        try:
            task.tear_down(startup.env)
        finally:
            close = getattr(startup.env, "close", None)
            if callable(close):
                close()


def validate_autodroid_memory_root(memory_root: str | Path) -> dict[str, Any]:
    """Validate one local copy of official AutoDroid/DroidBot memory.

    AutoDroid memory is intentionally not converted into an OmniFlow schema.
    The runner only checks the official replay inputs that it will read.
    """

    root = Path(memory_root).expanduser().resolve()
    manifest_path = root / "memory_manifest.json"
    if not root.is_dir():
        raise FileNotFoundError(f"autodroid_memory_root_missing:{root}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"autodroid_memory_manifest_missing:{manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"autodroid_memory_manifest_invalid:{manifest_path}") from error
    if manifest.get("format") != AUTODROID_MEMORY_MANIFEST_FORMAT:
        raise ValueError("autodroid_memory_manifest_format_invalid")
    apps = manifest.get("apps")
    if not isinstance(apps, list) or not apps:
        raise ValueError("autodroid_memory_apps_missing")
    return {
        "memory_root": str(root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "app_count": len(apps),
        "device": dict(manifest.get("device") or {}),
    }


def _autodroid_active_package(
    *,
    adb_path: str,
    serial: str,
) -> str:
    output = _run_adb(
        adb_path,
        serial,
        ["shell", "dumpsys", "activity", "activities"],
        check=False,
    ).stdout
    patterns = (
        r"mResumedActivity:.*?\s([A-Za-z0-9_.]+)/(?:[A-Za-z0-9_.$]+)",
        r"mCurrentFocus=Window\{[^}]*\s([A-Za-z0-9_.]+)/(?:[A-Za-z0-9_.$]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, output)
        if match:
            return match.group(1)
    return ""


def _autodroid_memory_for_app(
    *,
    memory_root: str | Path,
    adb_path: str,
    serial: str,
    app_name: str = "",
) -> dict[str, str]:
    root = Path(memory_root).expanduser().resolve()
    validate_autodroid_memory_root(root)
    runs_root = root / "runs"
    apks_root = root / "apks"
    requested = str(app_name or "").strip()
    candidates = sorted(
        path for path in runs_root.iterdir() if path.is_dir()
    ) if runs_root.is_dir() else []
    selected = next((path for path in candidates if path.name == requested), None)
    active_package = ""
    if selected is None and not requested:
        active_package = _autodroid_active_package(
            adb_path=adb_path,
            serial=serial,
        )
        for path in candidates:
            package_files = path.glob("dumpsys_package_*.txt")
            if any(
                file.name.removeprefix("dumpsys_package_").removesuffix(".txt")
                == active_package
                for file in package_files
            ):
                selected = path
                break
    if selected is None:
        detail = requested or active_package or "active_package_unknown"
        raise ValueError(f"autodroid_memory_app_not_found:{detail}")
    package_files = sorted(selected.glob("dumpsys_package_*.txt"))
    package = (
        package_files[0].name.removeprefix("dumpsys_package_").removesuffix(".txt")
        if package_files
        else ""
    )
    apk = apks_root / f"{selected.name}.apk"
    if not apk.is_file():
        raise FileNotFoundError(f"autodroid_memory_apk_missing:{apk}")
    events = sorted((selected / "events").glob("event_*.json"))
    if not events:
        raise ValueError(f"autodroid_memory_events_missing:{selected}")
    invalid = []
    for event in events:
        try:
            json.loads(event.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            invalid.append(event.name)
    if invalid:
        raise ValueError(
            "autodroid_memory_events_invalid:" + ",".join(invalid)
        )
    return {
        "app_name": selected.name,
        "package": package,
        "memory": str(selected),
        "apk": str(apk),
        "event_count": str(len(events)),
        "active_package": active_package,
    }


def resolve_mobilegpt_client_host(
    host: str = "",
    *,
    serial: str = "",
    adb_path: str = "adb",
) -> str:
    """Choose a host address reachable from the selected Android device.

    Emulators use Android's documented host alias.  Physical/root devices use
    the host-side address selected by the route to the device, so the official
    MobileGPT client does not need a hand-edited ``HOST_IP`` for every run.
    An explicit non-wildcard host always wins.
    """

    explicit = str(host or "").strip()
    if explicit and explicit not in {"0.0.0.0", "::", "[::]", "127.0.0.1"}:
        return explicit
    if str(serial or "").startswith("emulator-"):
        return "10.0.2.2"
    try:
        route = subprocess.run(
            [str(adb_path or "adb"), "-s", str(serial), "shell", "ip", "route"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).stdout
        device_ips = re.findall(r"\bsrc\s+(\d{1,3}(?:\.\d{1,3}){3})\b", route)
        for device_ip in device_ips:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.connect((device_ip, 1))
                local_ip = str(probe.getsockname()[0] or "").strip()
            if local_ip and local_ip != "127.0.0.1":
                return local_ip
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return "10.0.2.2"


def _link_or_fail(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"official_forward_source_missing:{source}")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"official_forward_target_exists:{target}")
    target.symlink_to(source, target_is_directory=source.is_dir())


def write_adb_proxy(
    workspace: str | Path,
    *,
    serial: str,
    adb_path: str = "adb",
) -> Path:
    """Expose exactly one device to an unmodified official subprocess."""

    resolved_workspace = Path(workspace).expanduser().resolve()
    proxy_dir = resolved_workspace / "bin"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    real_adb = shutil.which(adb_path) or adb_path
    proxy = proxy_dir / "adb"
    script = f'''#!/bin/sh
set -eu
real_adb={shlex_quote(real_adb)}
serial={shlex_quote(str(serial))}
if [ "$#" -eq 1 ] && [ "$1" = "devices" ]; then
  printf 'List of devices attached\\n%s\\tdevice\\n' "$serial"
  exit 0
fi
# AppAgent's upstream controller expects one `Physical size:` line. Newer
# Android releases also print `Override size:`; keep that device detail out of
# the official parser without changing AppAgent itself.
if [ "$#" -ge 4 ] && [ "$1" = "-s" ] && [ "$3" = "shell" ] && [ "$4" = "wm" ] && [ "${5:-}" = "size" ]; then
  wm_output=$("$real_adb" "$@" </dev/null)
  printf '%s\\n' "$wm_output" | awk '/^[[:space:]]*Physical size:/{{print; found=1; exit}} END{{if (!found) exit 1}}' || printf '%s\\n' "$wm_output" | sed -n '1p'
  exit 0
fi
if [ "$#" -ge 3 ] && [ "$1" = "shell" ] && [ "$2" = "wm" ] && [ "${3:-}" = "size" ]; then
  wm_output=$("$real_adb" -s "$serial" "$@" </dev/null)
  printf '%s\\n' "$wm_output" | awk '/^[[:space:]]*Physical size:/{{print; found=1; exit}} END{{if (!found) exit 1}}' || printf '%s\\n' "$wm_output" | sed -n '1p'
  exit 0
fi
has_serial=0
for arg in "$@"; do
  if [ "$arg" = "-s" ]; then has_serial=1; fi
done
if [ "$has_serial" -eq 1 ]; then
  exec "$real_adb" "$@" </dev/null
fi
exec "$real_adb" -s "$serial" "$@" </dev/null
'''
    proxy.write_text(script, encoding="utf-8")
    proxy.chmod(proxy.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return proxy


def shlex_quote(value: str) -> str:
    return shlex.quote(str(value))


def prepare_appagent_workspace(
    *,
    official_root: str | Path,
    docs_root: str | Path,
    workspace: str | Path,
    app_name: str,
    config: dict[str, Any],
    serial: str,
    adb_path: str = "adb",
) -> dict[str, str]:
    """Prepare an AppAgent workspace without importing AppAgent internals."""

    root = Path(official_root).expanduser().resolve()
    docs = Path(docs_root).expanduser().resolve()
    work = Path(workspace).expanduser().resolve()
    if not (root / "run.py").is_file():
        raise FileNotFoundError(f"official_appagent_entry_missing:{root / 'run.py'}")
    if not docs.is_dir():
        raise FileNotFoundError(f"official_appagent_docs_missing:{docs}")
    if not str(app_name).strip():
        raise ValueError("official_appagent_app_name_required")
    work.mkdir(parents=True, exist_ok=False)
    (work / "apps").mkdir()
    (work / "tasks").mkdir()
    _link_or_fail(root / "scripts", work / "scripts")
    _link_or_fail(docs.parent, work / "apps" / str(app_name).strip())
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("official_appagent_forward_requires_pyyaml") from exc
    (work / "config.yaml").write_text(
        yaml.safe_dump(dict(config), sort_keys=False),
        encoding="utf-8",
    )
    proxy = write_adb_proxy(work, serial=serial, adb_path=adb_path)
    return {
        "workspace": str(work),
        "app_dir": str(work / "apps" / str(app_name).strip()),
        "config": str(work / "config.yaml"),
        "adb_proxy": str(proxy),
    }


def prepare_mobilegpt_server(
    *,
    official_root: str | Path,
    memory_root: str | Path,
    workspace: str | Path,
    embedding_model: str = "",
    chat_model: str = "",
) -> dict[str, str]:
    """Stage the official Server so its documented relative ``./memory`` works."""

    root = Path(official_root).expanduser().resolve()
    source = root / "Server"
    work = Path(workspace).expanduser().resolve()
    target = work / "Server"
    if not (source / "main.py").is_file():
        raise FileNotFoundError(f"official_mobilegpt_server_missing:{source / 'main.py'}")
    memory = Path(memory_root).expanduser().resolve()
    if not memory.is_dir():
        raise FileNotFoundError(f"official_mobilegpt_memory_missing:{memory}")
    overlay = memory / "frozen_memory"
    if not overlay.is_dir():
        overlay = memory
    work.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source, target, symlinks=True)
    configured_embedding_model = str(embedding_model or "").strip()
    configured_chat_model = str(chat_model or "").strip()
    if configured_embedding_model or configured_chat_model:
        _configure_mobilegpt_server(
            target,
            embedding_model=configured_embedding_model,
            chat_model=configured_chat_model,
        )
    staged_memory = target / "memory"
    for entry in overlay.iterdir():
        destination = staged_memory / entry.name
        if entry.is_dir():
            shutil.copytree(entry, destination, symlinks=True, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, destination)
    return {
        "workspace": str(work),
        "server_root": str(target),
        "memory_root": str(staged_memory),
    }


def _configure_mobilegpt_server(
    server_root: Path,
    *,
    embedding_model: str = "",
    chat_model: str = "",
) -> None:
    """Inject provider names into a temporary copy of the official Server.

    MobileGPT's upstream code keeps provider model names as constants. This
    edits only the disposable staging copy; the planner, memory reader,
    protocol, and action implementation remain upstream code.
    """

    normalized_embedding = str(embedding_model or "").strip()
    normalized_chat = str(chat_model or "").strip()
    utils_path = server_root / "utils" / "utils.py"
    if normalized_embedding and utils_path.is_file():
        source = utils_path.read_text(encoding="utf-8")
        source = re.sub(
            r'def get_openai_embedding\(text: str, model="text-embedding-3-small", \*\*kwargs\)(?: -> [^:]+)?:',
            'def get_openai_embedding(text: str, model=None, **kwargs):\n'
            '    model = model or os.getenv("MOBILEGPT_EMBEDDING_MODEL", "GLM-Embedding-2")',
            source,
            count=1,
        )
        utils_path.write_text(source, encoding="utf-8")
    if normalized_chat:
        main_path = server_root / "main.py"
        source = main_path.read_text(encoding="utf-8")
        for name in (
            "TASK_AGENT_GPT_VERSION",
            "APP_AGENT_GPT_VERSION",
            "SELECT_AGENT_HISTORY_GPT_VERSION",
            "EXPLORE_AGENT_GPT_VERSION",
            "SELECT_AGENT_GPT_VERSION",
            "DERIVE_AGENT_GPT_VERSION",
            "PARAMETER_FILLER_AGENT_GPT_VERSION",
            "ACTION_SUMMARIZE_AGENT_GPT_VERSION",
            "SUBTASK_MERGE_AGENT_GPT_VERSION",
            "gpt_4",
            "gpt_4_turbo",
            "gpt_3_5_turbo",
        ):
            source = re.sub(
                rf'os\.environ\["{re.escape(name)}"\] = "[^"]+"',
                f'os.environ["{name}"] = os.environ.get("MOBILEGPT_CHAT_MODEL", "{normalized_chat}")',
                source,
            )
        source = source.replace(
            'os.environ["vision_model"] = "gpt-4o"',
            'os.environ["vision_model"] = os.environ.get("MOBILEGPT_VISION_MODEL", os.environ.get("MOBILEGPT_CHAT_MODEL", "GLM-5.1"))',
        )
        main_path.write_text(source, encoding="utf-8")
        param_path = server_root / "agents" / "param_fill_agent.py"
        if param_path.is_file():
            param_source = param_path.read_text(encoding="utf-8")
            param_source = param_source.replace(
                'model="gpt-4o"',
                'model=os.getenv("MOBILEGPT_CHAT_MODEL", "GLM-5.1")',
            )
            param_path.write_text(param_source, encoding="utf-8")


def _run_adb(
    adb_path: str,
    serial: str,
    args: Sequence[str],
    *,
    check: bool = True,
    timeout_sec: float = 30.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [adb_path, "-s", serial, *args],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=max(1.0, float(timeout_sec)),
    )


def _run_mobilegpt_client(
    *,
    official_root: str | Path,
    serial: str,
    adb_path: str,
    host: str,
    instruction: str,
    output_root: str | Path,
    timeout_sec: float,
) -> int:
    """Build, install, and signal the untouched official MobileGPT client."""

    root = Path(official_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    client_root = output / "official_client"
    shutil.copytree(root / "App", client_root)
    global_java = (
        client_root
        / "app/src/main/java/com/example/MobileGPT/MobileGPTGlobal.java"
    )
    source = global_java.read_text(encoding="utf-8")
    source = source.replace(
        'HOST_IP = "INPUT_YOUR_SERVER_IP_ADDRESS"',
        f'HOST_IP = "{str(host).replace(chr(34), "")}"',
    )
    global_java.write_text(source, encoding="utf-8")
    sdk = str(
        os.environ.get("ANDROID_HOME")
        or os.environ.get("ANDROID_SDK_ROOT")
        or ""
    ).strip()
    if not sdk and Path(adb_path).is_file():
        adb_parent = Path(adb_path).expanduser().resolve().parent
        if adb_parent.name == "platform-tools":
            sdk = str(adb_parent.parent)
    if sdk:
        (client_root / "local.properties").write_text(
            f"sdk.dir={sdk}\n",
            encoding="utf-8",
        )
    plugin_version = str(
        os.environ.get("OMNIFLOW_ANDROID_GRADLE_PLUGIN") or "8.13.2"
    ).strip()
    build_file = client_root / "build.gradle"
    build_file.write_text(
        re.sub(
            r"version ['\"]8\.0\.1['\"]",
            f"version '{plugin_version}'",
            build_file.read_text(encoding="utf-8"),
        ),
        encoding="utf-8",
    )
    gradle = shutil.which(os.environ.get("OMNIFLOW_GRADLE_BIN", "gradle"))
    if not gradle:
        candidates = sorted(
            Path.home().glob(".gradle/wrapper/dists/*/*/gradle-*/bin/gradle"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        gradle = str(candidates[0]) if candidates else ""
    if not gradle:
        raise RuntimeError(
            "official_mobilegpt_client_requires_gradle:"
            " install Gradle or provide the official App debug APK"
        )
    subprocess.run(
        [gradle, ":app:assembleDebug"],
        cwd=client_root,
        check=True,
        text=True,
    )
    apk = client_root / "app/build/outputs/apk/debug/app-debug.apk"
    if not apk.is_file():
        raise FileNotFoundError(f"official_mobilegpt_apk_missing:{apk}")
    _run_adb(adb_path, serial, ["install", "-r", str(apk)])
    service = "com.example.MobileGPT/com.example.MobileGPT.MobileGPTAccessibilityService"
    current = _run_adb(
        adb_path,
        serial,
        ["shell", "settings", "get", "secure", "enabled_accessibility_services"],
        check=False,
    ).stdout.strip()
    services = [value for value in current.split(":") if value and value != "null"]
    if service not in services:
        services.append(service)
    _run_adb(
        adb_path,
        serial,
        ["shell", "settings", "put", "secure", "enabled_accessibility_services", ":".join(services)],
    )
    _run_adb(adb_path, serial, ["shell", "settings", "put", "secure", "accessibility_enabled", "1"])
    _run_adb(adb_path, serial, ["shell", "monkey", "-p", "com.example.MobileGPT", "1"])
    time.sleep(2.0)
    _run_adb(adb_path, serial, ["logcat", "-c"])
    _run_adb(
        adb_path,
        serial,
        [
            "shell",
            "am",
            "broadcast",
            "-a",
            "com.example.MobileGPT.STRING_ACTION",
            "-p",
            "com.example.MobileGPT",
            "--es",
            "com.example.MobileGPT.INSTRUCTION_EXTRA",
            instruction,
        ],
    )
    deadline = time.monotonic() + max(1.0, float(timeout_sec))
    while time.monotonic() < deadline:
        log = _run_adb(
            adb_path,
            serial,
            ["logcat", "-d", "-s", "MobileGPT_Service:D", "*:S"],
            check=False,
        ).stdout
        if "Task finished" in log or "-----------Task finished--------" in log:
            (output / "client_log.txt").write_text(log, encoding="utf-8")
            return 0
        time.sleep(1.0)
    (output / "client_log.txt").write_text(log, encoding="utf-8")
    return 124


def run_mobilegpt_client(
    *,
    official_root: str | Path,
    serial: str,
    adb_path: str,
    host: str,
    instruction: str,
    output_root: str | Path,
    timeout_sec: float,
    android_world_root: str | Path | None = None,
    task_name: str = "",
    task_params_json: str = "{}",
    task_seed: int = 113,
    console_port: int = 5560,
    grpc_port: int = 8560,
    perform_emulator_setup: bool = True,
) -> int:
    """Run MobileGPT from the same initialized AndroidWorld task state."""

    if not android_world_root or not str(task_name).strip():
        return _run_mobilegpt_client(
            official_root=official_root,
            serial=serial,
            adb_path=adb_path,
            host=host,
            instruction=instruction,
            output_root=output_root,
            timeout_sec=timeout_sec,
        )
    with _androidworld_task_startup(
        android_world_root=android_world_root,
        task_name=task_name,
        task_params_json=task_params_json,
        task_seed=task_seed,
        console_port=console_port,
        grpc_port=grpc_port,
        adb_path=adb_path,
        perform_emulator_setup=perform_emulator_setup,
    ) as (env, task):
        returncode = _run_mobilegpt_client(
            official_root=official_root,
            serial=serial,
            adb_path=adb_path,
            host=host,
            instruction=instruction,
            output_root=output_root,
            timeout_sec=timeout_sec,
        )
        reward = float(task.is_successful(env))
        return returncode if returncode != 0 else (0 if reward > 0.5 else 1)


def run_appagent_executor(
    *,
    python_executable: str,
    executor: str | Path,
    app_name: str,
    workspace: str | Path,
    goal: str,
    timeout_sec: float,
    android_world_root: str | Path,
    task_name: str,
    task_params_json: str,
    task_seed: int,
    console_port: int,
    grpc_port: int,
    adb_path: str,
    perform_emulator_setup: bool = True,
) -> int:
    """Run official AppAgent after the canonical task initialization."""

    with _androidworld_task_startup(
        android_world_root=android_world_root,
        task_name=task_name,
        task_params_json=task_params_json,
        task_seed=task_seed,
        console_port=console_port,
        grpc_port=grpc_port,
        adb_path=adb_path,
        perform_emulator_setup=perform_emulator_setup,
    ) as (env, task):
        try:
            result = subprocess.run(
                [
                    str(python_executable),
                    str(executor),
                    "--app",
                    str(app_name),
                    "--root_dir",
                    str(workspace),
                ],
                cwd=str(workspace),
                input=str(goal) + "\n",
                text=True,
                check=False,
                timeout=max(1.0, float(timeout_sec)),
            )
        except subprocess.TimeoutExpired:
            return 124
        if result.returncode != 0:
            return result.returncode
        reward = float(task.is_successful(env))
        return 0 if reward > 0.5 else 1


def run_autodroid_replay(
    *,
    official_root: str | Path,
    memory_root: str | Path,
    serial: str,
    adb_path: str,
    output_root: str | Path,
    timeout_sec: float,
    max_events: int,
    android_world_root: str | Path,
    task_name: str,
    task_params_json: str,
    task_seed: int,
    console_port: int,
    grpc_port: int,
    app_name: str = "",
    perform_emulator_setup: bool = True,
) -> int:
    """Replay official AutoDroid memory inside the shared task lifecycle."""

    root = Path(official_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not (root / "droidbot" / "start.py").is_file():
        raise FileNotFoundError(f"official_autodroid_entry_missing:{root}")
    memory_info = None
    started = time.monotonic()
    with _androidworld_task_startup(
        android_world_root=android_world_root,
        task_name=task_name,
        task_params_json=task_params_json,
        task_seed=task_seed,
        console_port=console_port,
        grpc_port=grpc_port,
        adb_path=adb_path,
        perform_emulator_setup=perform_emulator_setup,
    ) as (env, task):
        memory_info = _autodroid_memory_for_app(
            memory_root=memory_root,
            adb_path=adb_path,
            serial=serial,
            app_name=app_name,
        )
        droidbot_output = output / "droidbot"
        # This AutoDroid checkout contains an old replay-policy method
        # signature while its shared InputPolicy.start() passes the manager.
        # Keep the official entrypoint and policy untouched; adapt only that
        # Python call boundary in the disposable child process.
        official_launcher = (
            "from droidbot.input_policy import UtgReplayPolicy; "
            "_official_generate_event = UtgReplayPolicy.generate_event; "
            "UtgReplayPolicy.generate_event = "
            "lambda self, input_manager=None: _official_generate_event(self); "
            "from droidbot.start import main; main()"
        )
        command = [
            sys.executable,
            "-c",
            official_launcher,
            "-d",
            serial,
            "-a",
            memory_info["apk"],
            "-o",
            str(droidbot_output),
            "-policy",
            "replay",
            "-replay_output",
            memory_info["memory"],
            "-count",
            str(max(1, int(max_events))),
            "-interval",
            "0",
            "-timeout",
            str(max(1, int(timeout_sec))),
            "-keep_app",
            "-keep_env",
            "-grant_perm",
            "-is_emulator",
            "-accessibility_auto",
        ]
        env_vars = dict(os.environ)
        env_vars["PYTHONPATH"] = os.pathsep.join(
            value
            for value in (str(root), env_vars.get("PYTHONPATH", ""))
            if value
        )
        adb_parent = Path(adb_path).expanduser().resolve().parent
        env_vars["PATH"] = os.pathsep.join(
            value
            for value in (str(adb_parent), env_vars.get("PATH", ""))
            if value
        )
        try:
            process = subprocess.run(
                command,
                cwd=str(root),
                env=env_vars,
                check=False,
                timeout=max(1.0, float(timeout_sec)),
            )
            returncode = int(process.returncode)
        except subprocess.TimeoutExpired:
            returncode = 124
        reward = float(task.is_successful(env)) if returncode == 0 else 0.0
        success = returncode == 0 and reward > 0.5
        replayed_event_count = min(
            int(memory_info["event_count"]), max(1, int(max_events))
        )
        result = {
            "schema_version": AUTODROID_RESULT_SCHEMA,
            "task": task_name,
            "method": "autodroid",
            "device": serial,
            "memory": memory_info,
            "official_validator_used": True,
            "official_validator_success": success,
            "androidworld_validator_result": {
                "validator": "androidworld_official",
                "success": success,
                "reward": reward,
            },
            "process_returncode": returncode,
            "actions_executed": replayed_event_count,
            "replay_event_limit": max(1, int(max_events)),
            "model_calls": 0,
            "fallback_steps": 0,
            "duration_ms": round((time.monotonic() - started) * 1000.0, 3),
        }
        (output / "autodroid_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output / "task_results.jsonl").write_text(
            json.dumps(
                {
                    **result,
                    "task_name": task_name,
                    "goal": str(getattr(task, "goal", "") or task_name),
                    "agent": "autodroid_official_replay",
                    "backend": "official_droidbot",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return 0 if success else (returncode or 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Forward one task to an official baseline")
    parser.add_argument(
        "--baseline", choices=("mobilegpt", "appagent", "autodroid"), default="mobilegpt"
    )
    parser.add_argument("--root")
    parser.add_argument("--serial", default="")
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--host", default="")
    parser.add_argument("--instruction", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--android-world-root")
    parser.add_argument("--task")
    parser.add_argument("--task-params-json", default="{}")
    parser.add_argument("--task-seed", type=int, default=113)
    parser.add_argument("--console-port", type=int, default=5560)
    parser.add_argument("--grpc-port", type=int, default=8560)
    parser.add_argument("--no-perform-emulator-setup", action="store_true")
    parser.add_argument("--executor")
    parser.add_argument("--app-name")
    parser.add_argument("--workspace")
    parser.add_argument("--goal", default="")
    parser.add_argument("--memory-root", default="")
    parser.add_argument("--max-events", type=int, default=20)
    args = parser.parse_args()
    if args.baseline == "mobilegpt":
        required = {
            "root": args.root,
            "serial": args.serial,
            "host": args.host,
            "instruction": args.instruction,
            "output": args.output,
        }
        missing = [name for name, value in required.items() if not str(value or "").strip()]
        if missing:
            parser.error("mobilegpt arguments required: " + ",".join(missing))
        return run_mobilegpt_client(
            official_root=args.root,
            serial=args.serial,
            adb_path=args.adb,
            host=args.host,
            instruction=args.instruction,
            output_root=args.output,
            timeout_sec=args.timeout,
            android_world_root=args.android_world_root,
            task_name=args.task or "",
            task_params_json=args.task_params_json,
            task_seed=args.task_seed,
            console_port=args.console_port,
            grpc_port=args.grpc_port,
            perform_emulator_setup=not args.no_perform_emulator_setup,
        )
    if args.baseline == "autodroid":
        required = {
            "root": args.root,
            "memory-root": args.memory_root,
            "serial": args.serial,
            "output": args.output,
            "task": args.task,
            "android-world-root": args.android_world_root,
        }
        missing = [name for name, value in required.items() if not str(value or "").strip()]
        if missing:
            parser.error("autodroid arguments required: " + ",".join(missing))
        return run_autodroid_replay(
            official_root=args.root,
            memory_root=args.memory_root,
            serial=args.serial,
            adb_path=args.adb,
            output_root=args.output,
            timeout_sec=args.timeout,
            max_events=args.max_events,
            android_world_root=args.android_world_root,
            task_name=args.task,
            task_params_json=args.task_params_json,
            task_seed=args.task_seed,
            console_port=args.console_port,
            grpc_port=args.grpc_port,
            app_name=args.app_name or "",
            perform_emulator_setup=not args.no_perform_emulator_setup,
        )
    required = {
        "executor": args.executor,
        "app-name": args.app_name,
        "workspace": args.workspace,
        "task": args.task,
        "android-world-root": args.android_world_root,
    }
    missing = [name for name, value in required.items() if not str(value or "").strip()]
    if missing:
        parser.error("appagent arguments required: " + ",".join(missing))
    return run_appagent_executor(
        python_executable=sys.executable,
        executor=args.executor,
        app_name=args.app_name,
        workspace=args.workspace,
        goal=args.goal,
        timeout_sec=args.timeout,
        android_world_root=args.android_world_root,
        task_name=args.task,
        task_params_json=args.task_params_json,
        task_seed=args.task_seed,
        console_port=args.console_port,
        grpc_port=args.grpc_port,
        adb_path=args.adb,
        perform_emulator_setup=not args.no_perform_emulator_setup,
    )


if __name__ == "__main__":
    raise SystemExit(main())
