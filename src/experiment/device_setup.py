"""Idempotent host and device setup for the AndroidWorld experiment.

This is deliberately a setup/check path, not another task runner.  The shell
launcher remains the only public entry point; it invokes this module only for
``--setup-device``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable

from src.experiment.checks import (
    OOB_ACCESSIBILITY_SERVICE,
    configure_default_device_services,
)
from src.experiment.protocol import (
    ANDROIDWORLD_REVISION,
    DEVICES,
    FOLD_SIZE,
    FOLD_STATE,
    SOURCE_AVD,
    SOURCE_DEVICE,
)


OOB_PACKAGE = "cn.com.omnimind.bot.debug"
OOB_ACTIVITY = "cn.com.omnimind.bot.activity.LauncherActivity"
OOB_REQUIRED_RECEIVERS = {
    ".DebugOmniFlowControlReceiver": "cn.com.omnimind.bot.debug.CONTROL_OMNIFLOW",
    ".DebugOmniFlowObserveReceiver": "cn.com.omnimind.bot.debug.OBSERVE_OMNIFLOW",
}
MOBILEGPT_PACKAGE = "com.example.MobileGPT"
_REPORT_PATH: Path | None = None


def _python_install_timeout_sec() -> float:
    raw = os.environ.get("OMNIFLOW_SETUP_PYTHON_TIMEOUT_SEC", "7200").strip()
    try:
        timeout = float(raw)
    except ValueError as error:
        raise ValueError(f"invalid setup Python timeout: {raw!r}") from error
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(f"invalid setup Python timeout: {raw!r}")
    return timeout


def _python_import_probe_code() -> str:
    return (
        "import android_world, omniflow, openai, uiautomator2, "
        "cv2, yaml, colorama, requests, dashscope; "
        "from serpapi import GoogleSearch"
    )


def _python_requirement_installs(
    *,
    repo: Path,
    android_world_root: Path,
    mobilegpt_root: Path,
    appagent_root: Path,
) -> list[tuple[str, list[str]]]:
    androidworld_requirements = android_world_root / "requirements.txt"
    if not androidworld_requirements.is_file():
        androidworld_requirements = android_world_root / "android_world" / "requirements.txt"
    installs = [
        # The runtime may already carry large pinned wheels (Torch, CUDA,
        # etc.).  Register this checkout without asking pip to resolve or
        # replace those wheels; the component requirement files below are
        # the explicit dependency contract for setup.
        ("editable", ["--no-deps", "-e", str(repo)]),
        ("requirements", ["-r", str(androidworld_requirements)]),
        ("editable", ["--no-deps", "-e", str(android_world_root)]),
        (
            "requirements",
            ["-r", str(mobilegpt_root / "Server" / "requirements.txt")],
        ),
    ]
    appagent_requirements = appagent_root / "requirements.txt"
    if appagent_requirements.is_file():
        installs.append(("requirements", ["-r", str(appagent_requirements)]))
    # MobileGPT's pinned requirements contain two distributions which own the
    # same ``serpapi`` import namespace.  The Server imports GoogleSearch from
    # google-search-results, so make that official symbol the final owner.
    installs.append(("uninstall", ["-y", "serpapi"]))
    installs.append(
        (
            "package",
            ["--force-reinstall", "--no-deps", "google-search-results==2.4.2"],
        )
    )
    installs.append(("package", ["colorama"]))
    installs.append(("package", ["dashscope"]))
    installs.append(("package", ["imageio-ffmpeg>=0.6"]))
    return installs


def _packaged_ffmpeg(python_bin: Path) -> Path | None:
    probe = _run(
        [
            str(python_bin),
            "-c",
            "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())",
        ],
        timeout=600,
    )
    if probe.returncode != 0:
        return None
    lines = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    candidate = Path(lines[-1]).expanduser()
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return candidate.resolve()
    return None


def _ensure_user_ffmpeg(
    *,
    python_bin: Path,
    install_python: bool,
    user_bin_dir: Path | None = None,
) -> Path | None:
    """Expose a user-owned ffmpeg required by AndroidWorld audio tasks."""

    existing = shutil.which("ffmpeg")
    if existing:
        candidate = Path(existing).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()

    packaged = _packaged_ffmpeg(python_bin)
    if packaged is None and install_python:
        install = _run(
            [
                str(python_bin),
                "-m",
                "pip",
                "install",
                "imageio-ffmpeg>=0.6",
            ],
            timeout=_python_install_timeout_sec(),
        )
        if install.returncode == 0:
            packaged = _packaged_ffmpeg(python_bin)
    if packaged is None:
        return None

    target_dir = user_bin_dir or (Path.home() / ".local" / "bin")
    target_dir.mkdir(parents=True, exist_ok=True)
    exposed = target_dir / "ffmpeg"
    if exposed.is_symlink():
        if exposed.resolve(strict=False) == packaged:
            return exposed
        exposed.unlink()
    elif exposed.exists():
        return exposed if os.access(exposed, os.X_OK) else None
    exposed.symlink_to(packaged)
    return exposed


def _is_required_setup_checkout(name: str) -> bool:
    return str(name) != "appagent_checkout"


@dataclass(frozen=True)
class Device:
    label: str
    serial: str
    console_port: int
    avd: str
    profile: str


def _devices() -> dict[str, Device]:
    raw = json.loads(
        (Path(__file__).resolve().parents[2] / "config" / "paper_androidworld.json")
        .read_text(encoding="utf-8")
    )["protocol"]
    result = {
        str(item["label"]): Device(
            label=str(item["label"]),
            serial=str(item["serial"]),
            console_port=int(item["console_port"]),
            avd=str(item["avd"]),
            profile=str(item["profile"]),
        )
        for item in raw["devices"]
    }
    source = raw["source_device"]
    result[str(source["label"])] = Device(
        label=str(source["label"]),
        serial=str(source["serial"]),
        console_port=int(source["console_port"]),
        avd=str(source["avd"]),
        profile=str(source["profile"]),
    )
    return result


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    timeout: float = 30,
    check: bool = False,
    input_data: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=check,
        input=input_data,
    )


def _safe_component(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"unsafe path component: {value!r}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_existing(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _resolve_tool(explicit: str, name: str, sdk_root: Path, subdir: str) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    found = shutil.which(name)
    if found:
        candidates.append(Path(found))
    candidates.append(sdk_root / subdir / name)
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise RuntimeError(f"missing executable {name}; checked: {candidates}")


def _aapt(sdk_root: Path) -> Path:
    candidates = sorted(
        sdk_root.glob("build-tools/*/aapt"),
        key=lambda path: path.parent.parent.name,
        reverse=True,
    )
    if candidates:
        return candidates[0].resolve()
    return _resolve_tool("", "aapt", sdk_root, "build-tools/latest")


def _aapt_text(aapt: Path, apk: Path, *args: str) -> str:
    command = [str(aapt), "dump", *args]
    if args and args[0] == "xmltree":
        command.extend([str(apk), "AndroidManifest.xml"])
    else:
        command.append(str(apk))
    result = _run(command, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"aapt failed for {apk}: {result.stdout[-1000:]}")
    return result.stdout


def _check_host(
    *,
    repo: Path,
    python_bin: Path,
    android_world_root: Path,
    omnitransfer_root: Path,
    mobilegpt_root: Path,
    appagent_root: Path,
    sdk_root: Path,
    env_file: Path | None,
    install_python: bool,
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    checks: list[dict[str, Any]] = []

    def record(name: str, ok: bool, detail: str, *, required: bool = True) -> None:
        checks.append(
            {"name": name, "status": "ok" if ok else "failed", "required": required, "detail": detail}
        )
        if not ok and required:
            raise RuntimeError(f"{name}: {detail}")

    record("python", python_bin.is_file() and os.access(python_bin, os.X_OK), str(python_bin))
    version = _run([str(python_bin), "--version"], timeout=10)
    record("python_version", version.returncode == 0, version.stdout.strip())

    def import_probe() -> subprocess.CompletedProcess[str]:
        return _run(
            [
                str(python_bin),
                "-c",
                _python_import_probe_code(),
            ],
            timeout=60,
        )

    initial_import_probe = import_probe()
    if install_python and initial_import_probe.returncode != 0:
        installs = _python_requirement_installs(
            repo=repo,
            android_world_root=android_world_root,
            mobilegpt_root=mobilegpt_root,
            appagent_root=appagent_root,
        )
        appagent_requirements = appagent_root / "requirements.txt"
        if not appagent_requirements.is_file():
            record(
                "python_requirement",
                False,
                f"optional AppAgent requirements missing: {appagent_requirements}",
                required=False,
            )
        for kind, arguments in installs:
            requirement = Path(arguments[-1]) if kind == "requirements" else None
            if requirement is not None and not requirement.exists():
                record("python_requirement", False, f"missing {requirement}")
            pip_action = "uninstall" if kind == "uninstall" else "install"
            command = [str(python_bin), "-m", "pip", pip_action, *arguments]
            result = _run(command, timeout=_python_install_timeout_sec())
            record("python_install", result.returncode == 0, result.stdout[-2000:])

    final_import_probe = import_probe()
    record("python_imports", final_import_probe.returncode == 0, final_import_probe.stdout[-1200:])
    ffmpeg = _ensure_user_ffmpeg(
        python_bin=python_bin,
        install_python=install_python,
    )
    record("ffmpeg", ffmpeg is not None, str(ffmpeg or "missing"))

    record(
        "androidworld_checkout",
        (android_world_root / "android_world").is_dir(),
        str(android_world_root),
    )
    if (android_world_root / ".git").exists():
        revision = _run(["git", "-C", str(android_world_root), "rev-parse", "HEAD"], timeout=10)
        actual = revision.stdout.strip()
        record(
            "androidworld_revision",
            actual == ANDROIDWORLD_REVISION,
            f"expected={ANDROIDWORLD_REVISION} actual={actual or 'missing'}",
        )
    else:
        record("androidworld_revision", False, "checkout has no .git directory")

    canonical_transfer = (Path.home() / "Projects" / "Omni" / "OmniTransfer").resolve()
    transfer_actual = omnitransfer_root.resolve()
    record(
        "omnitransfer_checkout",
        transfer_actual == canonical_transfer and transfer_actual.is_dir(),
        f"expected={canonical_transfer} actual={transfer_actual}",
    )
    record("mobilegpt_checkout", mobilegpt_root.is_dir(), str(mobilegpt_root))
    record(
        "appagent_checkout",
        appagent_root.is_dir(),
        str(appagent_root),
        required=_is_required_setup_checkout("appagent_checkout"),
    )
    if env_file is not None:
        record("model_env", env_file.is_file(), str(env_file))
        text = env_file.read_text(encoding="utf-8", errors="replace")
        record(
            "model_env_key",
            bool(re.search(r"^(?:LLMTHU_API_KEY|OPENAI_API_KEY)\s*=\s*.+$", text, re.MULTILINE)),
            "credential key is present (value omitted)",
        )

    try:
        aapt = _aapt(sdk_root)
        record("aapt", True, str(aapt))
    except RuntimeError as error:
        record("aapt", False, str(error))

    component_apks: dict[str, Path] = {}
    mobilegpt_apk = _first_existing(
        [
            mobilegpt_root / "App" / "app" / "build" / "outputs" / "apk" / "debug" / "app-debug.apk",
            mobilegpt_root / "App_Explorer" / "app" / "build" / "outputs" / "apk" / "debug" / "app-debug.apk",
        ]
    )
    if mobilegpt_apk is None and install_python:
        gradle = mobilegpt_root / "App" / "gradlew"
        if gradle.is_file():
            result = _run([str(gradle), "assembleDebug"], cwd=gradle.parent, timeout=1800)
            record("mobilegpt_build", result.returncode == 0, result.stdout[-2000:])
            mobilegpt_apk = _first_existing(
                [mobilegpt_root / "App" / "app" / "build" / "outputs" / "apk" / "debug" / "app-debug.apk"]
            )
    record("mobilegpt_apk", mobilegpt_apk is not None, str(mobilegpt_apk or "missing"))
    if mobilegpt_apk is not None:
        component_apks["mobilegpt"] = mobilegpt_apk

    return checks, component_apks


def _find_oob_apk(repo: Path, workspace_root: Path, asset_root: Path) -> Path | None:
    configured = os.environ.get("OMNIFLOW_OOB_APK", "").strip()
    candidates = [Path(configured)] if configured else []
    candidates.extend(
        [
            repo / "runtime" / "assets" / "oob-x86_64-debug.apk",
            asset_root / "runtime" / "assets" / "oob-x86_64-debug.apk",
            workspace_root / "OpenOmniBot" / "app" / "build" / "outputs" / "apk" / "developStandard" / "debug" / "app-develop-standard-debug.apk",
            workspace_root / "oob-downloads" / "v0.5.8.4" / "OpenOmniBot-v0.5.8.4-develop-standard-debug.apk",
            workspace_root / "releases" / "OmniFlow-mobilegpt-20260719" / ".artifacts" / "oob-develop-standard-debug.apk",
            workspace_root / "OmniFlow" / "runtime" / "assets" / "oob-x86_64-debug.apk",
            workspace_root / "evals" / "_workspace_runtime_archive_20260728" / "OmniFlow-4090-20260719-v1" / "runtime" / "assets" / "oob-x86_64-debug.apk",
        ]
    )
    return _first_existing(candidates)


def _validate_apks(
    *,
    aapt: Path,
    oob_apk: Path | None,
    mobilegpt_apk: Path,
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    checks: list[dict[str, Any]] = []
    apks = {"mobilegpt": mobilegpt_apk}

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "status": "ok" if ok else "failed", "required": True, "detail": detail})
        if not ok:
            raise RuntimeError(f"{name}: {detail}")

    if oob_apk is None:
        record("oob_apk", False, "no OOB APK; set OMNIFLOW_OOB_APK or provide a build artifact")
    assert oob_apk is not None
    oob_badging = _aapt_text(aapt, oob_apk, "badging")
    record("oob_package", f"package: name='{OOB_PACKAGE}'" in oob_badging, oob_badging.splitlines()[0])
    oob_manifest = _aapt_text(aapt, oob_apk, "xmltree")
    for receiver, action in OOB_REQUIRED_RECEIVERS.items():
        record(
            f"oob_receiver_{receiver.lstrip('.')}",
            receiver in oob_manifest and action in oob_manifest,
            f"required receiver={receiver} action={action}",
        )
    mobile_badging = _aapt_text(aapt, mobilegpt_apk, "badging")
    record(
        "mobilegpt_package",
        f"package: name='{MOBILEGPT_PACKAGE}'" in mobile_badging,
        mobile_badging.splitlines()[0],
    )
    checks.append(
        {
            "name": "apk_sha256",
            "status": "ok",
            "required": True,
            "detail": json.dumps({key: _sha256(path) for key, path in {"oob": oob_apk, **apks}.items()}),
        }
    )
    apks["oob"] = oob_apk
    return checks, apks


def _adb(adb: Path, serial: str, *args: str, timeout: float = 30) -> subprocess.CompletedProcess[str]:
    return _run([str(adb), "-s", serial, *args], timeout=timeout)


def _wait_for_device(
    adb: Path,
    serial: str,
    timeout: int = 240,
    process: subprocess.Popen[bytes] | None = None,
    log_path: Path | None = None,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            detail = ""
            if log_path and log_path.is_file():
                detail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            raise RuntimeError(
                f"emulator exited before {serial} booted: returncode={process.returncode}; {detail}"
            )
        state = _adb(adb, serial, "get-state", timeout=10)
        boot = _adb(adb, serial, "shell", "getprop", "sys.boot_completed", timeout=10)
        if state.returncode == 0 and state.stdout.strip() == "device" and boot.stdout.strip() == "1":
            return
        time.sleep(2)
    raise RuntimeError(f"device did not boot: {serial}")


def _ensure_emulator(
    *,
    adb: Path,
    emulator: Path,
    avdmanager: Path,
    sdk_root: Path,
    device: Device,
    log_root: Path,
    start: bool,
) -> subprocess.Popen[bytes] | None:
    avds = _run([str(emulator), "-list-avds"], timeout=30)
    if device.avd not in {line.strip() for line in avds.stdout.splitlines()}:
        api_level = "34" if device.profile == "pixel_fold" else "33"
        system_image = (
            f"system-images;android-{api_level};google_apis;"
            f"{os.environ.get('OMNIFLOW_ANDROID_SYSTEM_IMAGE_ABI', 'x86_64')}"
        )
        # SDK device-profile ids vary by command-line-tools release.  The
        # The 7.6-inch foldable exposes the CLOSED/HALF_OPENED/OPENED state
        # contract used by the protocol; apply the protocol display override
        # below because its physical panel differs across SDK releases.
        device_profile = (
            "7.6in Foldable"
            if device.profile == "pixel_fold"
            else "10.1in WXGA (Tablet)"
            if device.profile == "tablet"
            else "pixel_2"
        )
        created = _run(
            [
                str(avdmanager),
                "create",
                "avd",
                "--force",
                "--name",
                device.avd,
                "--package",
                system_image,
                "--device",
                device_profile,
            ],
            cwd=sdk_root,
            timeout=300,
            input_data="no\n",
        )
        if created.returncode != 0:
            raise RuntimeError(
                f"could not create AVD {device.avd} from {system_image}: "
                f"{created.stdout[-1200:]}"
            )
    state = _adb(adb, device.serial, "get-state", timeout=10)
    if state.returncode == 0 and state.stdout.strip() == "device":
        _wait_for_device(adb, device.serial)
        return None
    if not start:
        raise RuntimeError(f"device is not ready: {device.serial}")
    log_root.mkdir(parents=True, exist_ok=True)
    log = (log_root / f"{device.label}.emulator.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            str(emulator),
            "-avd",
            device.avd,
            "-port",
            str(device.console_port),
            "-no-window",
            "-no-audio",
            "-no-boot-anim",
            "-gpu",
            "swiftshader_indirect",
            "-no-snapshot-load",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        _wait_for_device(adb, device.serial, process=process, log_path=log_root / f"{device.label}.emulator.log")
    except Exception:
        process.terminate()
        raise
    return process


def _install(adb: Path, serial: str, apk: Path) -> None:
    result = _adb(adb, serial, "install", "-r", "-t", str(apk), timeout=300)
    if result.returncode != 0 or "Success" not in result.stdout:
        raise RuntimeError(f"install failed {apk.name} on {serial}: {result.stdout[-1200:]}")


def _configure_device(
    adb: Path,
    device: Device,
    apks: dict[str, Path],
    a11y_apk: Path,
) -> list[dict[str, Any]]:
    serial = device.serial
    _install(adb, serial, apks["oob"])
    _install(adb, serial, apks["mobilegpt"])
    if not a11y_apk.is_file():
        raise RuntimeError(f"accessibility forwarder APK missing: {a11y_apk}")
    _install(adb, serial, a11y_apk)
    configured_services = configure_default_device_services(str(adb), serial)
    if not bool(configured_services.get("settings_write_ok")):
        raise RuntimeError(f"could not enable accessibility services on {serial}")
    start = _adb(adb, serial, "shell", "am", "start", "-n", f"{OOB_PACKAGE}/{OOB_ACTIVITY}")
    if start.returncode != 0:
        raise RuntimeError(f"could not start OOB on {serial}: {start.stdout[-1000:]}")
    _adb(adb, serial, "shell", "input", "keyevent", "HOME")
    time.sleep(3)
    package_dump = _adb(adb, serial, "shell", "dumpsys", "package", OOB_PACKAGE).stdout
    mobile_dump = _adb(adb, serial, "shell", "dumpsys", "package", MOBILEGPT_PACKAGE).stdout
    accessibility_dump = _adb(adb, serial, "shell", "dumpsys", "accessibility").stdout
    checks = [
        {"name": "oob_installed", "status": "ok" if OOB_PACKAGE in package_dump else "failed", "required": True, "detail": serial},
        {"name": "mobilegpt_installed", "status": "ok" if MOBILEGPT_PACKAGE in mobile_dump else "failed", "required": True, "detail": serial},
        {"name": "accessibility_bound", "status": "ok" if OOB_ACCESSIBILITY_SERVICE.rsplit("/", 1)[-1] in accessibility_dump else "failed", "required": True, "detail": serial},
    ]
    failed = [check for check in checks if check["status"] != "ok"]
    if failed:
        raise RuntimeError("device service check failed: " + json.dumps(failed))
    if device.profile == "pixel_fold":
        state = _adb(adb, serial, "shell", "cmd", "device_state", "state", str(FOLD_STATE))
        if state.returncode != 0:
            raise RuntimeError(f"could not set fold state {FOLD_STATE} on {serial}: {state.stdout[-1000:]}")
        resized = _adb(adb, serial, "shell", "wm", "size", FOLD_SIZE)
        if resized.returncode != 0:
            raise RuntimeError(f"could not set fold display size on {serial}: {resized.stdout[-1000:]}")
        _wait_for_device(adb, serial, timeout=60)
        size = _adb(adb, serial, "shell", "wm", "size").stdout
        if FOLD_SIZE not in size:
            raise RuntimeError(f"fold display size mismatch: expected={FOLD_SIZE} actual={size.strip()}")
    return checks


def _health_probe(adb: Path, device: Device) -> dict[str, Any]:
    from src.integrations.android_world.oob_control import OobControlClient

    try:
        state = OobControlClient(
            object(),
            adb_serial=device.serial,
            adb_path=str(adb),
            timeout_seconds=30,
        ).observe(wait_to_stabilize=True)
        xml = str(state.get("xml") or "") if isinstance(state, dict) else ""
        if not xml.strip():
            raise RuntimeError("OOB observe returned no XML")
        return {
            "name": "oob_observe_smoke",
            "status": "ok",
            "required": True,
            "detail": f"xml_chars={len(xml)}",
        }
    except Exception as error:
        return {
            "name": "oob_observe_smoke",
            "status": "failed",
            "required": True,
            "detail": str(error),
        }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install and health-check one AndroidWorld device.")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--device", required=True, help="device label or all")
    parser.add_argument("--python", dest="python_bin", type=Path, required=True)
    parser.add_argument("--sdk-root", type=Path, required=True)
    parser.add_argument("--android-world-root", type=Path, required=True)
    parser.add_argument("--omnitransfer-root", type=Path, required=True)
    parser.add_argument("--mobilegpt-root", type=Path, required=True)
    parser.add_argument("--appagent-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--a11y-apk", type=Path, required=True)
    parser.add_argument("--adb")
    parser.add_argument("--emulator")
    parser.add_argument("--avdmanager")
    parser.add_argument("--install-python", action="store_true")
    parser.add_argument("--no-start", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    repo = args.repo.resolve()
    sdk_root = args.sdk_root.expanduser().resolve()
    devices = _devices()
    if args.device == "all":
        selected = list(devices.values())
    else:
        labels = [value.strip() for value in args.device.split(",") if value.strip()]
        missing = [value for value in labels if value not in devices]
        if missing:
            raise SystemExit(f"unknown setup device(s): {', '.join(missing)}")
        selected = [devices[value] for value in labels]
    adb = _resolve_tool(args.adb or "", "adb", sdk_root, "platform-tools")
    emulator = _resolve_tool(args.emulator or "", "emulator", sdk_root, "emulator")
    avdmanager = _resolve_tool(args.avdmanager or "", "avdmanager", sdk_root, "cmdline-tools/latest/bin")
    report_root = args.report_root.resolve()
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    report_dir = report_root / "setup" / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    global _REPORT_PATH
    _REPORT_PATH = report_dir / "setup_report.json"
    workspace_root = repo.parent
    host_checks, host_apks = _check_host(
        repo=repo,
        # Keep the venv launcher intact.  Resolving its symlink can drop the
        # venv prefix and make imports run against the base interpreter.
        python_bin=args.python_bin.expanduser(),
        android_world_root=args.android_world_root.expanduser().resolve(),
        omnitransfer_root=args.omnitransfer_root.expanduser().resolve(),
        mobilegpt_root=args.mobilegpt_root.expanduser().resolve(),
        appagent_root=args.appagent_root.expanduser().resolve(),
        sdk_root=sdk_root,
        env_file=args.env_file.expanduser().resolve() if args.env_file else None,
        install_python=args.install_python,
    )
    try:
        aapt = _aapt(sdk_root)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    oob_apk = _find_oob_apk(repo, workspace_root, args.asset_root.resolve())
    apk_checks, apks = _validate_apks(aapt=aapt, oob_apk=oob_apk, mobilegpt_apk=host_apks["mobilegpt"])
    all_device_reports = []
    started: list[subprocess.Popen[bytes]] = []
    try:
        for device in selected:
            device_report: dict[str, Any] = {
                "label": device.label,
                "serial": device.serial,
                "avd": device.avd,
                "profile": device.profile,
                "checks": [],
            }
            process = _ensure_emulator(
                adb=adb,
                emulator=emulator,
                avdmanager=avdmanager,
                sdk_root=sdk_root,
                device=device,
                log_root=report_dir,
                start=not args.no_start,
            )
            if process is not None:
                started.append(process)
            device_report["checks"].extend(_configure_device(adb, device, apks, args.a11y_apk.expanduser().resolve()))
            probe = _health_probe(adb, device)
            device_report["checks"].append(probe)
            if probe["status"] != "ok":
                raise RuntimeError(f"{device.label}: {probe['detail']}")
            device_report["status"] = "ok"
            all_device_reports.append(device_report)
            print(f"[setup] {device.label}: ready ({device.serial})")
    finally:
        if args.no_start:
            pass
    report = {
        "schema_version": "omniflow.androidworld.device_setup.v1",
        "status": "ok",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "androidworld_revision": ANDROIDWORLD_REVISION,
        "host": {"checks": host_checks + apk_checks, "adb": str(adb), "emulator": str(emulator)},
        "devices": all_device_reports,
        "artifacts": {key: str(value) for key, value in apks.items()},
    }
    _REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(_REPORT_PATH)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.SubprocessError, OSError) as error:
        if _REPORT_PATH is not None:
            _REPORT_PATH.write_text(
                json.dumps(
                    {
                        "schema_version": "omniflow.androidworld.device_setup.v1",
                        "status": "failed",
                        "error": str(error),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"[setup:error] {error}; report={_REPORT_PATH}", file=sys.stderr)
        else:
            print(f"[setup:error] {error}", file=sys.stderr)
        raise SystemExit(1) from error
