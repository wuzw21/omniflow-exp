"""Minimal AndroidWorld memory conversion and task runner."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence

from src.experiment.function_v2 import compile_function_v2, write_function_review
from src.experiment.appagent_source import convert_runlog_to_appagent_memory
from src.experiment.paths import relative_reference, resolve_path, sha256_file
from src.experiment.androidworld_paths import (
    canonical_device_seed_name,
    canonical_method_name,
)
from src.experiment.protocol import (
    DEFAULT_DEVICE,
    DEFAULT_METHOD,
    DEFAULT_TASK,
    DEVICE_AVDS,
    DEVICES,
    ENABLED_METHODS,
    FORMAL_MODEL,
    FORMAL_MODEL_BASE_URL,
    FORMAL_MODEL_ENDPOINT_PROFILE,
    MAX_FALLBACK_STEPS,
    MAX_STEPS,
    METHODS,
    SOURCE_DEVICE,
    SOURCE_METHOD,
    SOURCE_SEED,
    TASK_DEADLINE_SEC,
    TASK_SEED,
    require_formal_model,
)
from src.integrations.mobilegpt import convert_runlog_to_mobilegpt_bundle
from src.integrations.mobilegpt import validate_prepared_memory
from src.experiment.mobilegpt_contract import MOBILEGPT_SOURCE_METHOD
from src.integrations.appagent import validate_appagent_memory

REPO_ROOT = Path(__file__).resolve().parents[2]


def _methods(value: str) -> tuple[str, ...]:
    if not value or value == "all":
        return ENABLED_METHODS
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = tuple(
        item for item in selected if item not in (*METHODS, SOURCE_METHOD)
    )
    if unknown:
        raise ValueError("unknown_method:" + ",".join(unknown))
    disabled = tuple(item for item in selected if item not in (*ENABLED_METHODS, SOURCE_METHOD))
    if disabled:
        raise ValueError("method_not_enabled:" + ",".join(disabled))
    return selected


def _devices(value: str) -> tuple[tuple[str, str, int], ...]:
    if not value or value == "all":
        return DEVICES
    configured_devices = {item[0]: item for item in (*DEVICES, SOURCE_DEVICE)}
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    devices: list[tuple[str, str, int]] = []
    for item in selected:
        configured = configured_devices.get(item)
        if configured is None:
            raise ValueError(f"unknown_configured_device:{item}")
        devices.append(configured)
    return tuple(devices)


def _sdk_tool(name: str) -> str:
    discovered = shutil.which(name)
    if discovered:
        return discovered
    adb = shutil.which("adb")
    if adb:
        candidate = Path(adb).resolve().parent.parent / "emulator" / name
        if candidate.is_file():
            return str(candidate)
    for variable in ("ANDROID_SDK_ROOT", "ANDROID_HOME"):
        root = str(os.environ.get(variable) or "").strip()
        if root:
            tool_directory = "platform-tools" if name == "adb" else "emulator"
            candidate = Path(root).expanduser() / tool_directory / name
            if candidate.is_file():
                return str(candidate)
    tool_directory = "platform-tools" if name == "adb" else "emulator"
    for root in (Path.home() / "Library/Android/sdk", Path.home() / "Android/Sdk"):
        candidate = root / tool_directory / name
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError(f"android_sdk_tool_missing:{name}")


def _mobilegpt_server_port(console_port: int) -> int:
    """Derive a deterministic local TCP port for one AVD.

    MobileGPT's official Server is one process per device.  The upstream
    checkout defaults to 12345, so parallel AVD runs would contend for the
    same listener even though their Android serials are distinct.  Reuse the
    configured emulator console port as the stable source for an isolated,
    unprivileged server port.
    """

    # The original source5560 APK is the canonical MobileGPT collector and
    # embeds the upstream server port 12345.  Keep that source-only contract;
    # formal target AVDs use deterministic isolated ports derived from their
    # own console ports below.
    if int(console_port) == 5560:
        return 12345
    return 12000 + int(console_port) % 40000


def _method_memory_path(memory_root: str | Path, method: str) -> Path:
    """Resolve the fixed output of the one-shot conversion layout.

    This is a direct path convention, not a catalog or historical-result
    lookup.  The source RunLog remains outside this directory and is passed
    separately to every atomic run.
    """

    root = resolve_path(memory_root)
    if method == "omniflow":
        return root / "omniflow" / "store.json"
    if method == "mobilegpt":
        return root / "mobilegpt" / "memory"
    if method == "appagent":
        return root / "appagent"
    raise ValueError(f"method_has_no_memory_input:{method}")


def _golden_run_root(
    output_root: str | Path,
    *,
    task: str,
    method: str,
    device: tuple[str, str, int],
) -> Path:
    """Return the one stable result directory for a task setting.

    Episode work is created in the runner's private temporary workspace.  Only
    the selected successful result is promoted here, so the visible archive
    never grows a sequence of attempt directories.
    """

    label, serial, console_port = device
    return (
        resolve_path(output_root)
        / str(task)
        / canonical_method_name(method)
        / canonical_device_seed_name(
            label=label,
            serial=serial,
            console_port=console_port,
            source_seed=SOURCE_SEED,
            evaluation_seed=TASK_SEED,
        )
        / "runlog"
        / "current"
    )


def _run_quality(path: Path) -> tuple[int, int, int, int, int, int] | None:
    """Score one sealed run without using history selection or a registry."""

    run_log_path = path / "run_log.json"
    if not run_log_path.is_file():
        return None
    try:
        payload = json.loads(run_log_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    validator = payload.get("validator")
    diagnostics = payload.get("diagnostics")
    execution = diagnostics.get("execution_summary") if isinstance(diagnostics, dict) else {}
    if not isinstance(execution, dict):
        execution = {}
    official = int(isinstance(validator, dict) and validator.get("official") is True)
    succeeded = int(payload.get("status") == "succeeded" and payload.get("success") is True)
    fallback_steps = int(execution.get("fallback_steps") or 0)
    model_calls = int(execution.get("model_calls") or 0)
    clean_execution = int(not str(execution.get("failure_reason") or "").strip())
    mobilegpt_result = (
        diagnostics.get("mobilegpt_result")
        if isinstance(diagnostics, dict)
        else {}
    )
    mobilegpt_protocol = (
        mobilegpt_result.get("mobilegpt_protocol")
        if isinstance(mobilegpt_result, dict)
        else {}
    )
    protocol_finished = int(
        isinstance(mobilegpt_protocol, dict)
        and mobilegpt_protocol.get("task_finished") is True
    )
    # Prefer official success, then fewer fallback/model calls.  This keeps a
    # successful Function replay as the visible golden record without making
    # any claim from an unsuccessful or incomplete candidate.
    return (
        official,
        succeeded,
        clean_execution,
        protocol_finished,
        -fallback_steps,
        -model_calls,
    )


def _promote_golden_run(
    *,
    candidate: Path,
    destination: Path,
) -> bool:
    """Promote one successful candidate into the fixed ``runlog/current`` slot."""

    candidate_quality = _run_quality(candidate)
    if candidate_quality is None or candidate_quality[:2] != (1, 1):
        return False
    current_quality = _run_quality(destination) if destination.is_dir() else None
    if current_quality is not None and candidate_quality <= current_quality:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(".current.staging")
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(candidate, staging)
    if destination.exists():
        shutil.rmtree(destination)
    staging.replace(destination)
    return True


def _archive_failed_run(
    *,
    candidate: Path,
    output_root: str | Path,
    task: str,
    method: str,
    device: tuple[str, str, int],
    archive_kind: str = "failure",
) -> Path | None:
    """Keep one sealed result as non-runtime evidence.

    A successful candidate can lose promotion because the existing golden
    result is equally good.  It must not be labelled as a failure: that would
    corrupt the evidence trail and make later audits misclassify a real
    successful execution.
    """

    if not candidate.is_dir():
        return None
    evidence_files = [
        path
        for name in (
            "run_log.json",
            "task_results.jsonl",
            "protocol_probe.json",
            "client_log.txt",
        )
        for path in candidate.rglob(name)
    ]
    if not evidence_files:
        return None
    label = device[0]
    archive_root = (
        resolve_path(output_root)
        / ".archive"
        / str(task)
        / canonical_method_name(method)
        / label
    )
    archive_root.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        destination = archive_root / f"{archive_kind}_{index:03d}"
        if not destination.exists():
            shutil.copytree(candidate, destination)
            return destination
        index += 1


def _promote_mobilegpt_source_memory(
    *,
    candidate: Path,
    destination: Path,
) -> bool:
    """Preserve the Memory learned by one official cold source episode.

    MobileGPT writes its learned graph into the isolated episode directory.
    The public runner owns that temporary directory, so a successful source
    run must promote the graph before the temporary workspace is removed.
    This is an explicit path inside the current candidate, not a history scan.
    """

    learned = (
        candidate
        / "mobilegpt_memory"
        / "_episodes"
        / "source5560"
        / "mobilegpt_memory"
    )
    if not learned.is_dir() or not (learned / "tasks.csv").is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(".current.staging")
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(learned, staging, symlinks=True)
    if destination.exists():
        shutil.rmtree(destination)
    staging.replace(destination)
    return True


def _candidate_run_directories(candidate: Path) -> list[Path]:
    """Find this command's sealed run directories.

    AndroidWorld may write either directly below ``runlog/current`` or below
    its timestamped child (``runlog/current/run_<timestamp>``).  Both layouts
    are command-local evidence; the runner must promote/archive either one
    before its temporary workspace is removed.
    """

    directories: set[Path] = set()
    for run_log in candidate.rglob("run_log.json"):
        parts = run_log.relative_to(candidate).parts
        if "runlog" not in parts or "current" not in parts:
            continue
        directories.add(run_log.parent)
    return sorted(directories)


def _candidate_evidence_directories(candidate: Path) -> list[Path]:
    """Find command-local directories containing persisted result evidence."""

    directories: set[Path] = set()
    for filename in ("task_results.jsonl", "protocol_probe.json"):
        for evidence in candidate.rglob(filename):
            parts = evidence.relative_to(candidate).parts
            if "runlog" not in parts or "current" not in parts:
                continue
            directories.add(evidence.parent)
    return sorted(directories)


def _relative_output(value: str | Path) -> str:
    """Return a repository-relative address for CLI/report output."""

    return relative_reference(Path(value).expanduser().resolve(), base=REPO_ROOT)


def _remove_tree(path: Path) -> None:
    """Remove one runner-owned staging tree, including read-only files."""

    if not path.exists() or path.is_symlink():
        return

    def _make_writable(func: Any, filename: str, _error: Any) -> None:
        os.chmod(filename, 0o700)
        func(filename)

    shutil.rmtree(path, onerror=_make_writable)


def _reuse_existing_memory(
    *,
    method: str,
    source: Path,
    memory_root: Path,
    task_name: str,
) -> tuple[dict[str, Any], Path] | None:
    """Reuse one explicit, complete Memory without any authoring call.

    Existing output is intentionally not searched for or selected.  The
    caller supplies the exact address; a partial or incompatible address is a
    hard error so a stale experiment can never silently become a new result.
    """

    if not memory_root.exists():
        return None
    source_digest = sha256_file(source)
    if method == "appagent":
        validated = validate_appagent_memory(
            memory_root,
            task_name=task_name,
            source_run_log=source,
        )
        if str(validated.get("document_generation_model") or "") != FORMAL_MODEL:
            raise ValueError("appagent_memory_model_mismatch")
        if str(validated.get("source_run_log_sha256") or "") != source_digest:
            raise ValueError("appagent_memory_source_run_log_mismatch")
        return (
            {
                "method": method,
                "task_name": task_name,
                "memory_root": _relative_output(memory_root),
                "manifest": validated,
                "reused": True,
            },
            memory_root,
        )
    if method == "mobilegpt":
        memory = memory_root / "memory"
        validated = validate_prepared_memory(
            memory,
            task_name=task_name,
            source_seed=SOURCE_SEED,
            source_run_log=source,
            expected_model=FORMAL_MODEL,
            expected_source_method=MOBILEGPT_SOURCE_METHOD,
        )
        if validated.get("model_verified") is not True:
            raise ValueError("mobilegpt_memory_model_unverified")
        manifest = validated.get("manifest") or {}
        recorded = ((manifest.get("source_run_log") or {}).get("sha256"))
        if recorded and str(recorded) != source_digest:
            raise ValueError("mobilegpt_memory_source_run_log_mismatch")
        return (
            {
                "method": method,
                "task_name": task_name,
                "memory_root": _relative_output(memory),
                "manifest": manifest,
                "reused": True,
            },
            memory,
        )
    if method == "omniflow":
        store = memory_root / "store.json"
        copied_source = memory_root / "run_log.json"
        report_path = memory_root / "compile_report.json"
        if not store.is_file() or not copied_source.is_file() or not report_path.is_file():
            raise FileNotFoundError("omniflow_memory_bundle_incomplete")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict) or str(report.get("model") or "") != FORMAL_MODEL:
            raise ValueError("omniflow_memory_model_mismatch")
        if sha256_file(copied_source) != source_digest:
            raise ValueError("omniflow_memory_source_run_log_mismatch")
        store_payload = json.loads(store.read_text(encoding="utf-8"))
        if not isinstance(store_payload, dict) or not store_payload:
            raise ValueError("omniflow_memory_store_invalid")
        return (
            {
                "method": method,
                "task_name": task_name,
                "memory_root": _relative_output(store),
                "manifest": report,
                "reused": True,
            },
            store,
        )
    raise ValueError(f"method_has_no_memory_reuse:{method}")


def _device_booted(adb: str, serial: str) -> bool:
    try:
        state = subprocess.run(
            (adb, "-s", serial, "get-state"),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if state.returncode != 0 or state.stdout.strip() != "device":
            return False
        booted = subprocess.run(
            (adb, "-s", serial, "shell", "getprop", "sys.boot_completed"),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return booted.returncode == 0 and booted.stdout.strip() == "1"
    except (OSError, subprocess.SubprocessError):
        return False


def _wait_for_device(adb: str, serial: str, timeout: int) -> None:
    deadline = time.monotonic() + max(1, timeout)
    while time.monotonic() < deadline:
        if _device_booted(adb, serial):
            return
        time.sleep(1)
    raise RuntimeError(f"android_emulator_boot_timeout:{serial}")


def _running_avd_name(adb: str, serial: str) -> str:
    try:
        completed = subprocess.run(
            (adb, "-s", serial, "emu", "avd", "name"),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode == 0:
        name = next(
            (
                line.strip()
                for line in completed.stdout.splitlines()
                if line.strip() and line.strip().casefold() != "ok"
            ),
            "",
        )
        if name:
            return name
    # ADB-over-SSH/proxy clients can successfully execute ``emu avd name``
    # but lose the console response body.  The emulator publishes the same
    # identity through the read-only boot property; use it only as a fallback
    # so configured AVD validation remains enabled across transports.
    try:
        fallback = subprocess.run(
            (adb, "-s", serial, "shell", "getprop", "ro.boot.qemu.avd_name"),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return fallback.stdout.strip() if fallback.returncode == 0 else ""


def _validate_configured_avd_identity(
    adb: str,
    devices: tuple[tuple[str, str, int], ...],
    avds: dict[str, str],
) -> None:
    for _label, serial, _port in devices:
        expected = str(avds.get(serial) or "").strip()
        if not expected:
            continue
        actual = _running_avd_name(adb, serial)
        if actual != expected:
            raise RuntimeError(
                "android_emulator_avd_mismatch:"
                f"{serial}:expected={expected}:actual={actual or 'unknown'}"
            )


def _ensure_devices_started(
    devices: tuple[tuple[str, str, int], ...],
    *,
    timeout: int,
) -> None:
    """Reuse online configured AVDs and start only the missing ones."""

    adb = _sdk_tool("adb")
    # Start the single host ADB server before probing or booting devices.
    # Without this serialized bootstrap, the per-device startup workers can
    # all observe a missing server and race to create it; adb then drops
    # transports (or reports ``device offline``) even though the AVDs are
    # otherwise healthy.  The device work itself remains parallel below.
    try:
        subprocess.run(
            (adb, "start-server"),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"android_adb_server_start_failed:{exc}") from exc
    avds = dict(DEVICE_AVDS)
    missing = tuple(device for device in devices if not _device_booted(adb, device[1]))
    if missing:
        emulator = _sdk_tool("emulator")
        for _label, serial, port in missing:
            avd = avds.get(serial)
            if not avd:
                continue
            subprocess.Popen(
                (
                    emulator,
                    "-avd",
                    avd,
                    "-port",
                    str(port),
                    "-grpc",
                    str(port + 3000),
                    "-no-window",
                    "-no-audio",
                    "-no-boot-anim",
                    "-no-snapshot-save",
                    "-gpu",
                    "swiftshader_indirect",
                ),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, len(devices))
    ) as executor:
        tuple(
            executor.map(
                lambda device: _wait_for_device(adb, device[1], timeout),
                devices,
            )
        )
    _validate_configured_avd_identity(adb, devices, avds)


def _convert_memory(args: argparse.Namespace) -> dict[str, Any]:
    source = resolve_path(args.source_run_log)
    output = resolve_path(args.memory)
    output_preexisting = output.exists()

    methods = _methods(args.method)
    if not source.is_file():
        raise FileNotFoundError(f"source_run_log_missing:{source}")
    source_payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(source_payload, dict):
        raise ValueError("source_run_log_object_required")
    task_name = str(source_payload.get("task_name") or args.task or "").strip()
    if not task_name:
        raise ValueError("source_run_log_task_name_required")
    reports: list[dict[str, Any]] = []
    memories: dict[str, str] = {}
    newly_created_outputs: list[Path] = []
    try:
        for method in methods:
            method_output = output / method if args.method == "all" else output
            output_was_present = method_output.exists()
            reused = _reuse_existing_memory(
                method=method,
                source=source,
                memory_root=method_output,
                task_name=task_name,
            )
            if reused is not None:
                report, memory_path = reused
                if method == "omniflow":
                    compiler_report = json.loads(
                        (method_output / "compile_report.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    review_path = write_function_review(
                        source_run_log=source,
                        memory_root=method_output,
                        report=compiler_report,
                    )
                    report["review_path"] = _relative_output(review_path)
                reports.append(report)
                memories[method] = _relative_output(memory_path)
                continue
            if not output_was_present:
                newly_created_outputs.append(method_output)
            if method == "omniflow":
                report = compile_function_v2(
                    source,
                    method_output,
                    enhance=True,
                    model=FORMAL_MODEL,
                    model_endpoint_profile=FORMAL_MODEL_ENDPOINT_PROFILE,
                    model_base_url=FORMAL_MODEL_BASE_URL,
                )
                memory = Path(str(report["store_path"]))
            elif method == "mobilegpt":
                if not str(args.mobilegpt_root or "").strip():
                    raise ValueError("conversion_requires_mobilegpt_root:mobilegpt")
                mobilegpt_root = resolve_path(args.mobilegpt_root)
                if not mobilegpt_root.is_dir():
                    raise FileNotFoundError(
                        f"conversion_dependency_missing:mobilegpt:{args.mobilegpt_root}"
                    )
                report = convert_runlog_to_mobilegpt_bundle(
                    source_run_log=source,
                    mobilegpt_root=mobilegpt_root,
                    output_root=method_output,
                    model=FORMAL_MODEL,
                    source_seed=SOURCE_SEED,
                )
                memory = Path(str(report["memory_root"]))
            elif method == "appagent":
                if not str(args.appagent_root or "").strip():
                    raise ValueError("conversion_requires_appagent_root:appagent")
                appagent_root = resolve_path(args.appagent_root)
                if not appagent_root.is_dir():
                    raise FileNotFoundError(
                        f"conversion_dependency_missing:appagent:{args.appagent_root}"
                    )
                report = convert_runlog_to_appagent_memory(
                    source_run_log=source,
                    appagent_root=appagent_root,
                    memory_root=method_output,
                    model=FORMAL_MODEL,
                )
                memory = Path(str(report["memory_root"]))
            else:
                raise ValueError(f"unknown_conversion_method:{method}")
            reports.append(report)
            memories[method] = _relative_output(memory)
    except BaseException:
        for candidate in reversed(newly_created_outputs):
            if candidate.is_dir() and not candidate.is_symlink():
                _remove_tree(candidate)
        if not output_preexisting and output.is_dir() and not any(output.iterdir()):
            output.rmdir()
        raise

    if args.method == "all":
        return {
            "action": "convert-memory",
            "task": task_name,
            "methods": list(methods),
            "source_run_log": _relative_output(source),
            "memory_root": _relative_output(output),
            "memories": memories,
        }
    return {
        "action": "convert-memory",
        "task": task_name,
        "method": args.method,
        "memory": next(iter(memories.values())),
    }


def _run_command(
    args: argparse.Namespace,
    method: str,
    device: tuple[str, str, int],
    temporary_root: Path,
) -> tuple[str, str, int]:
    label, serial, port = device
    command = [
        sys.executable,
        "-m",
        "src.experiment.run_task",
        "result",
        "--task",
        args.task,
        "--method",
        method,
        "--device",
        f"{label}:{serial}:{port}",
        "--output-path",
        str(temporary_root / method / label),
    ]
    if args.mobilegpt_root:
        command.extend(("--mobilegpt-root", args.mobilegpt_root))
    if args.appagent_root:
        command.extend(("--appagent-root", args.appagent_root))
    if method == "mobilegpt":
        command.extend(
            (
                "--mobilegpt-port",
                str(_mobilegpt_server_port(port)),
                # The official Server imports its vision/memory stack before
                # opening the per-device socket.  Five seconds is too short
                # on a cold Python process and is reported as an environment
                # failure before an episode starts; keep this startup wait
                # outside the episode wall-clock measurement.
                "--mobilegpt-server-warmup-sec",
                "30",
            )
        )
    if args.source_run_log:
        command.extend(("--source-run-log", args.source_run_log))
    if args.memory:
        if args.method == "all":
            if method in {"omniflow", "mobilegpt", "appagent"}:
                command.extend(("--memory", str(_method_memory_path(args.memory, method))))
        else:
            command.extend(("--memory", args.memory))
    if args.dry_run:
        command.append("--dry-run")
    # Keep the episode itself in the temporary runner workspace.  A successful
    # sealed result is promoted below to the one stable current slot; failures
    # are archived as explicit evidence before the private workspace ends.
    environment = dict(os.environ)
    if method == "mobilegpt":
        # The official client embeds its server port. Keep one immutable APK
        # per AVD so parallel devices never share a client/server endpoint.
        runtime_root = resolve_path(
            os.environ.get("OMNIFLOW_MOBILEGPT_RUNTIME_ROOT")
            or Path("data") / "runtime" / "mobilegpt"
        )
        per_device_apk = runtime_root / f"client_{label}.apk"
        if per_device_apk.is_file():
            environment["OMNIFLOW_MOBILEGPT_APK"] = str(per_device_apk)
    completed = subprocess.run(
        command,
        env=environment,
        cwd=REPO_ROOT,
        check=False,
    )
    candidate = temporary_root / method / label
    # run_task nests the canonical task/method/device path below its
    # caller-owned output root.  Locate only this command's own candidate,
    # never scan previous results.  A failed command can still have produced
    # truthful protocol_probe/task_results evidence; preserve it instead of
    # deleting the only explanation with the temporary workspace.
    matches = _candidate_run_directories(candidate)
    if completed.returncode == 0 and len(matches) == 1:
        promoted = _promote_golden_run(
            candidate=matches[0],
            destination=_golden_run_root(
                args.output,
                task=args.task,
                method=method,
                device=device,
            ),
        )
        if not promoted:
            _archive_failed_run(
                candidate=matches[0].parent,
                output_root=args.output,
                task=args.task,
                method=method,
                device=device,
                archive_kind="superseded",
            )
        if method == "mobilegpt" and serial == SOURCE_DEVICE[1]:
            source_memory = (
                resolve_path(args.output)
                / args.task
                / "mobilegpt"
                / canonical_device_seed_name(
                    label=label,
                    serial=serial,
                    console_port=port,
                    source_seed=SOURCE_SEED,
                    evaluation_seed=TASK_SEED,
                )
                / "memory"
                / "current"
            )
            if not _promote_mobilegpt_source_memory(
                candidate=candidate,
                destination=source_memory,
            ):
                completed = subprocess.CompletedProcess(
                    completed.args,
                    1,
                    completed.stdout,
                    completed.stderr,
                )
    else:
        # A task can finish at the process level with return code 0 while the
        # official validator fails or the episode writer flushes only the
        # parent ``runlog`` layout.  Preserve that command-scoped evidence as
        # well; otherwise the failed run disappears from the experiment.
        failed_evidence = _candidate_evidence_directories(candidate)
        # Some AndroidWorld failure paths flush the sealed result one level
        # above ``runlog/current``.  It is still truthful command-scoped
        # evidence and must not disappear with the temporary workspace.
        if not failed_evidence:
            fallback_dirs: set[Path] = set()
            for filename in ("task_results.jsonl", "protocol_probe.json"):
                fallback_dirs.update(
                    path.parent for path in candidate.rglob(filename)
                    if "runlog" in path.relative_to(candidate).parts
                )
            failed_evidence = sorted(fallback_dirs)
        if failed_evidence:
            _archive_failed_run(
                candidate=failed_evidence[0],
                output_root=args.output,
                task=args.task,
                method=method,
                device=device,
            )
    return method, label, completed.returncode


def _run(args: argparse.Namespace) -> dict[str, Any]:
    methods = _methods(args.method)
    devices = _devices(args.device)
    if SOURCE_METHOD in methods and devices != (SOURCE_DEVICE,):
        raise ValueError("source_method_requires_device:source5560")
    if args.dry_run:
        return {
            "action": "run",
            "task": args.task,
            "methods": list(methods),
            "devices": [item[0] for item in devices],
            "memory": args.memory or None,
            "source_run_log": args.source_run_log or None,
        }
    if len(methods) > 1:
        if not args.source_run_log:
            raise ValueError("multi_method_run_requires_one_source_run_log")
        required_memory_methods = {
            method for method in methods if method in {"omniflow", "mobilegpt", "appagent"}
        }
        if required_memory_methods and not args.memory:
            raise ValueError("multi_method_run_requires_one_memory_root")
        for method in required_memory_methods:
            memory_path = _method_memory_path(args.memory, method)
            if not memory_path.exists():
                raise FileNotFoundError(
                    f"multi_method_memory_missing:{method}:{memory_path}"
                )
    _ensure_devices_started(devices, timeout=min(TASK_DEADLINE_SEC, 180))
    started = time.monotonic()
    with tempfile.TemporaryDirectory(
        prefix="omniflow-androidworld-"
    ) as temporary:
        root = Path(temporary)
        # Methods are the experimental groups: keep groups sequential so
        # their wall time is comparable, while the three target devices in
        # each group run concurrently.  Do not make the device the outer
        # loop; that would serialize five methods on every emulator and make
        # the reported "parallel" time misleading.
        rows: list[tuple[str, str, int]] = []
        for method in methods:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, len(devices))
            ) as executor:
                method_rows = list(
                    executor.map(
                        lambda device: _run_command(args, method, device, root),
                        devices,
                    )
                )
            rows.extend(method_rows)
    return {
        "action": "run",
        "task": args.task,
        "status": "completed" if all(row[2] == 0 for row in rows) else "failed",
        "wall_sec": round(time.monotonic() - started, 3),
        "runs": [
            {"method": method, "device": device, "returncode": returncode}
            for method, device, returncode in rows
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        nargs="?",
        choices=("run", "convert-memory"),
        default="run",
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--memory", default="")
    parser.add_argument("--source-run-log", default="")
    parser.add_argument("--output", default="data/androidworld")
    parser.add_argument(
        "--mobilegpt-root",
        default=os.environ.get("OMNIFLOW_MOBILEGPT_ROOT", ""),
    )
    parser.add_argument(
        "--appagent-root",
        default=os.environ.get(
            "OMNIFLOW_APPAGENT_ROOT",
            "../OmniFlow/runtime/external/appagent",
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(repo=repo)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_formal_model()
    if args.action == "convert-memory":
        if not args.source_run_log or not args.memory:
            raise ValueError("convert-memory requires --source-run-log and --memory")
        if args.method != "all" and args.method not in METHODS:
            raise ValueError(f"unknown_method:{args.method}")
        selected_methods = _methods(args.method)
        result = (
            {
                "action": "convert-memory",
                "task": args.task,
                "methods": list(selected_methods),
                "source_run_log": args.source_run_log,
                "memory": args.memory,
            }
            if args.dry_run
            else _convert_memory(args)
        )
    else:
        result = _run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status", "completed") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
