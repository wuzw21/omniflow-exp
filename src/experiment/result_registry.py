"""Validate and register immutable AndroidWorld results."""

from __future__ import annotations

from contextlib import contextmanager
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import tempfile
from typing import Any, Mapping

from src.experiment.mobilegpt_contract import MOBILEGPT_OOB_ACTION_INDEX_PROTOCOL
from src.experiment.paths import safe_component, sha256_file
from src.experiment.protocol import DEVICES, RESULT_COMMANDS_FILE
from src.experiment.result_schema import compact_result_row

DEVICE_TARGETS = {label: (serial, port) for label, serial, port in DEVICES}
_REGISTERED_RESULT_SCHEMA = "omniflow.androidworld_registered_result.v1"
_REGISTRATION_SCHEMA = "omniflow.androidworld_result_registration.v1"
LEGACY_EXTERNAL_METHODS = frozenset({"mobilegpt", "appagent"})


def registered_result_matches_device_model(
    row: Mapping[str, Any], expected_model: str
) -> bool:
    """Require immutable evidence to identify the current physical AVD."""

    expected = str(expected_model or "").strip()
    if not expected:
        return True
    target = row.get("device_target")
    if isinstance(target, Mapping):
        for key in ("device_model", "avd"):
            if str(target.get(key) or "").strip() == expected:
                return True
    for key in ("device_model", "avd"):
        if str(row.get(key) or "").strip() == expected:
            return True
    for key in ("run_dir", "output_path", "artifact_root", "command"):
        value = str(row.get(key) or "")
        if f"/{expected}_seed" in value or f"/{expected}/" in value:
            return True
    return False


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _commands_path_for_summary(summary_path: Path) -> Path:
    current = summary_path.with_name(RESULT_COMMANDS_FILE)
    if current.exists():
        return current
    return summary_path.with_name("one_task_commands.jsonl")


def _load_verified_registered_result(summary_path: Path) -> dict[str, Any]:
    summary = _load_json(summary_path)
    if (
        not isinstance(summary, dict)
        or summary.get("schema_version") != _REGISTERED_RESULT_SCHEMA
    ):
        raise ValueError(f"registered result schema invalid: {summary_path}")
    manifest_path = summary_path.with_name("registration_manifest.json")
    recorded_manifest = Path(
        str(summary.get("registration_manifest") or "")
    ).expanduser()
    if not recorded_manifest.is_absolute():
        recorded_manifest = (summary_path.parent / recorded_manifest).resolve()
    else:
        recorded_manifest = recorded_manifest.resolve()
    if recorded_manifest != manifest_path.resolve():
        raise ValueError(f"registered result manifest path mismatch: {summary_path}")
    manifest = _load_json(manifest_path)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != _REGISTRATION_SCHEMA
        or manifest.get("immutable") is not True
    ):
        raise ValueError(f"registration manifest invalid: {manifest_path}")
    for field in (
        "registration_id",
        "attempt_id",
        "task_name",
        "source_seed",
        "evaluation_seed",
    ):
        if str(summary.get(field) or "") != str(manifest.get(field) or ""):
            raise ValueError(f"registered result {field} mismatch: {summary_path}")
    rows = [row for row in summary.get("rows") or [] if isinstance(row, dict)]
    if len(rows) != 1:
        raise ValueError(
            f"registered result must contain exactly one row: {summary_path}"
        )
    row = rows[0]
    if (
        str(row.get("method") or "") != str(manifest.get("method") or "")
        or str(row.get("device") or "") != str(manifest.get("device") or "")
    ):
        raise ValueError(f"registered result result mismatch: {summary_path}")
    return summary


def mobilegpt_runtime_integrity_error(value: Any) -> str | None:
    """Classify MobileGPT transport/runtime evidence for run accounting."""

    error = str(value or "").strip()
    if not error or "mobilegpt_step_budget_exhausted" in error:
        return None
    markers = (
        "ConnectionError:",
        "ConnectionRefusedError:",
        "TimeoutError:",
        "mobilegpt_androidworld_state_",
        "mobilegpt_app_ui_not_ready:",
        "mobilegpt_launch_response_invalid:",
        "mobilegpt_native_xml_invalid:",
        "mobilegpt_server_closed_connection",
        "mobilegpt_handshake_failed",
        "mobilegpt_handshake_timeout",
        "mobilegpt_step_timeout",
        "mobilegpt_server_handler_failed",
    )
    return error if any(marker in error for marker in markers) else None


def formal_result_environment_failure_reasons(
    row: dict[str, Any],
) -> tuple[str, ...]:
    """Return explicit runtime/environment failures that forbid registration."""

    if has_official_validator_conclusion(row):
        return ()
    reasons: list[str] = []
    runtime_error = row.get("runtime_integrity_error")
    if (
        isinstance(runtime_error, str)
        and runtime_error.strip()
        or runtime_error is not None
        and not isinstance(runtime_error, str)
        and bool(runtime_error)
    ):
        reasons.append("runtime_integrity_error")
    environment_failure = row.get("environment_failure")
    if environment_failure is True or (
        isinstance(environment_failure, str)
        and environment_failure.strip().lower()
        not in {"", "0", "false", "no", "none", "null"}
    ):
        reasons.append("environment_failure")
    for field in (
        "classification",
        "failure_classification",
        "result_classification",
        "status",
    ):
        if str(row.get(field) or "").strip().lower() == "environment_failure":
            reasons.append(field)
    return tuple(dict.fromkeys(reasons))


def has_official_validator_conclusion(row: dict[str, Any]) -> bool:
    return row.get("official_validator_used") is True and isinstance(
        row.get("official_validator_success"), bool
    )


def legacy_external_result_protocol_compatible(
    row: Mapping[str, Any],
    *,
    task_name: str,
    method: str,
    device: str,
    evaluation_seed: int,
) -> bool:
    """Validate immutable results written by the pre-v2 external adapter.

    Older MobileGPT/AppAgent runs used ``official_forward`` and therefore did
    not emit the native runner metadata (``fixed_task_seed``,
    ``state_backend``, and ``max_steps``).  They still contain enough
    immutable evidence to identify a formal cell: the official validator
    conclusion, the exact target, and the task seed in the command.  This is a
    read-only import path; it never rewrites or weakens the native method
    protocol and rejects environment/runtime failures.
    """

    if method not in LEGACY_EXTERNAL_METHODS:
        return False
    if str(row.get("task_name") or task_name) != task_name:
        return False
    if str(row.get("method") or "") != method:
        return False
    if str(row.get("device") or "") != device:
        return False
    if not has_official_validator_conclusion(dict(row)):
        return False
    runtime_error = str(row.get("runtime_integrity_error") or "").strip()
    if runtime_error:
        return False
    environment_failure = row.get("environment_failure")
    if environment_failure is True or (
        isinstance(environment_failure, str)
        and environment_failure.strip().lower()
        not in {"", "0", "false", "no", "none", "null"}
    ):
        return False

    target = row.get("device_target")
    expected_target = DEVICE_TARGETS.get(device)
    if not isinstance(target, Mapping) or expected_target is None:
        return False
    if (
        str(target.get("serial") or "") != expected_target[0]
        or int(target.get("console_port") or -1) != expected_target[1]
    ):
        return False

    try:
        command = shlex.split(str(row.get("command") or ""))
    except ValueError:
        return False
    if "-m" not in command or "src.integrations.official_forward" not in command:
        return False
    try:
        baseline_index = command.index("--baseline")
        if command[baseline_index + 1] != method:
            return False
    except (ValueError, IndexError):
        return False

    def option_value(name: str) -> str | None:
        for index, token in enumerate(command):
            if token == name and index + 1 < len(command):
                return command[index + 1]
            if token.startswith(name + "="):
                return token.partition("=")[2]
        return None

    task_seed = option_value("--task-seed")
    if task_seed is None:
        task_seed = option_value("--task-random-seed")
    if task_seed != str(evaluation_seed):
        return False
    if option_value("--serial") != expected_target[0]:
        return False
    try:
        if int(option_value("--console-port") or "") != expected_target[1]:
            return False
    except ValueError:
        return False
    return isinstance(row.get("task_params"), dict)


def validate_formal_result_protocol(
    row: dict[str, Any],
    *,
    task_name: str,
    method: str,
    device: str,
    evaluation_seed: int,
    max_steps: int,
) -> None:
    """Reject a registered conclusion produced outside the frozen protocol."""

    violations: list[str] = []
    normalized_device = {
        "target5554": "small5554",
        "target5564": "fold5564",
    }.get(str(row.get("device") or ""), str(row.get("device") or ""))
    if legacy_external_result_protocol_compatible(
        row,
        task_name=task_name,
        method=method,
        device=normalized_device,
        evaluation_seed=evaluation_seed,
    ):
        return
    if str(row.get("task_name") or task_name) != task_name:
        violations.append("task")
    if str(row.get("method") or "") != method:
        violations.append("method")
    if normalized_device != device:
        violations.append("device")
    expected_device = DEVICE_TARGETS.get(device)
    if expected_device is None:
        violations.append("unsupported_device")
    elif (
        str(row.get("serial") or "") != expected_device[0]
        or row.get("console_port") != expected_device[1]
    ):
        violations.append("device_target")
    if row.get("task_random_seed") != evaluation_seed:
        violations.append("task_random_seed")
    if row.get("fixed_task_seed") is not True:
        violations.append("fixed_task_seed")
    if row.get("fixed_task_params") is not False:
        violations.append("fixed_task_params")
    if row.get("perform_emulator_setup") is not True:
        violations.append("perform_emulator_setup")
    if row.get("state_backend") != "androidworld":
        violations.append("state_backend")
    if (
        method == "fixed_replay"
        and row.get("execution_backend") != "recorded_coordinate_replay_v1"
    ):
        violations.append("execution_backend")

    task_params = row.get("task_params")
    params_sha256 = str(row.get("task_params_sha256") or "")
    if not isinstance(task_params, dict):
        violations.append("task_params")

    try:
        command = shlex.split(str(row.get("command") or ""))
    except ValueError:
        command = []
        violations.append("command")
    if any(
        token == "--oob-observe-backend"
        or token.startswith("--oob-observe-backend=")
        for token in command
    ):
        violations.append("oob_observe_backend")
    explicit_max_steps = row.get("max_steps")
    if explicit_max_steps is not None and explicit_max_steps != max_steps:
        violations.append("max_steps_metadata")
    recorded_max_steps_value: str | None = None
    for index, token in enumerate(command):
        if token == "--max-steps":
            if index + 1 < len(command):
                recorded_max_steps_value = command[index + 1]
            break
        if token.startswith("--max-steps="):
            recorded_max_steps_value = token.partition("=")[2]
            break
    try:
        recorded_max_steps = int(recorded_max_steps_value or "")
    except ValueError:
        violations.append("max_steps")
    else:
        if recorded_max_steps != max_steps:
            violations.append("max_steps")
    if violations:
        raise ValueError(
            "formal_result_protocol_mismatch:"
            f"{task_name}:{method}:{device}:{','.join(violations)}"
        )


def registered_result_plan(
    *,
    runs_root: Path,
    task_name: str,
    methods: tuple[str, ...],
    devices: tuple[str, ...],
    source_seed: int,
    evaluation_seed: int,
    formal_max_steps: int | None = None,
    device_models: Mapping[str, str] | None = None,
) -> dict[str, list[tuple[str, str]]]:
    """Read immutable results for older indexes without creating another plan."""

    root = runs_root.expanduser().resolve()
    expected = [(method, device) for method in methods for device in devices]
    completed: list[tuple[str, str]] = []
    for method, device in expected:
        result_root = root / task_name / method / device
        for path in sorted(result_root.glob("*/registered_result.json")):
            summary = _load_verified_registered_result(path)
            public_row = summary["rows"][0]
            detail_row = next(
                (
                    detail
                    for detail in summary.get("details") or []
                    if isinstance(detail, dict)
                    and str(detail.get("method") or "") == method
                    and str(detail.get("device") or "") == device
                ),
                public_row,
            )
            if (
                str(summary.get("task_name") or "") != task_name
                or str(public_row.get("method") or "") != method
                or str(public_row.get("device") or "") != device
            ):
                raise ValueError(
                    f"registered result does not match expected result: {path}"
                )
            if (
                summary.get("source_seed") != source_seed
                or summary.get("evaluation_seed") != evaluation_seed
            ):
                continue
            if formal_result_environment_failure_reasons(detail_row):
                continue
            if not has_official_validator_conclusion(detail_row):
                continue
            expected_model = (device_models or {}).get(device)
            if expected_model and not registered_result_matches_device_model(
                detail_row, expected_model
            ):
                continue
            # Native OmniFlow failures remain retryable.  For the two external
            # methods, a validator=false conclusion is still a usable final
            # method outcome when it is a verified legacy official_forward
            # result and not an environment failure.
            if detail_row.get("official_validator_success") is not True and not (
                legacy_external_result_protocol_compatible(
                    detail_row,
                    task_name=task_name,
                    method=method,
                    device=device,
                    evaluation_seed=evaluation_seed,
                )
            ):
                continue
            if formal_max_steps is not None:
                validate_formal_result_protocol(
                    detail_row,
                    task_name=task_name,
                    method=method,
                    device=device,
                    evaluation_seed=evaluation_seed,
                    max_steps=formal_max_steps,
                )
            completed.append((method, device))
            break
    return {
        "completed": completed,
        "pending": [result for result in expected if result not in completed],
    }


def registered_result_keys_matching_task_params(
    *,
    runs_root: Path,
    task_name: str,
    methods: tuple[str, ...],
    devices: tuple[str, ...],
    source_seed: int,
    evaluation_seed: int,
    task_params: Mapping[str, Any],
    device_models: Mapping[str, str] | None = None,
) -> set[tuple[str, str]]:
    """Return immutable conclusions for the exact generated task instance.

    The evaluation seed alone is insufficient evidence when multiple device
    workers race through AndroidWorld's global-RNG parameter generator.  Keep
    every historical registration immutable, but only reuse a cell whose
    recorded parameters equal the canonical instance generated for that seed.
    """

    root = runs_root.expanduser().resolve()
    expected_params = dict(task_params)
    # Older source-derived registrations retained the source RunLog seed as a
    # top-level provenance field. It does not alter AndroidWorld's generated
    # task instance and must not invalidate otherwise identical parameters.
    expected_params.pop("seed", None)
    matched: set[tuple[str, str]] = set()
    for method in methods:
        for device in devices:
            result_root = root / task_name / method / device
            for path in sorted(result_root.glob("*/registered_result.json")):
                summary = _load_verified_registered_result(path)
                if (
                    str(summary.get("task_name") or "") != task_name
                    or summary.get("source_seed") != source_seed
                    or summary.get("evaluation_seed") != evaluation_seed
                ):
                    continue
                public_row = summary["rows"][0]
                detail_row = next(
                    (
                        detail
                        for detail in summary.get("details") or []
                        if isinstance(detail, dict)
                        and str(detail.get("method") or "") == method
                        and str(detail.get("device") or "") == device
                    ),
                    public_row,
                )
                recorded_params = detail_row.get("task_params")
                if isinstance(recorded_params, dict):
                    recorded_params = dict(recorded_params)
                    recorded_params.pop("seed", None)
                if (
                    str(public_row.get("method") or "") != method
                    or str(public_row.get("device") or "") != device
                    or not has_official_validator_conclusion(detail_row)
                    or formal_result_environment_failure_reasons(detail_row)
                    or detail_row.get("runtime_integrity_error")
                    == "mobilegpt_oob_action_target_missing"
                    or (
                        method == "mobilegpt"
                        and detail_row.get("official_validator_success") is False
                        and detail_row.get("oob_action_index_protocol")
                        != MOBILEGPT_OOB_ACTION_INDEX_PROTOCOL
                    )
                    or recorded_params != expected_params
                ):
                    continue
                expected_model = (device_models or {}).get(device)
                if expected_model and not registered_result_matches_device_model(
                    detail_row, expected_model
                ):
                    continue
                matched.add((method, device))
                break
    return matched


@contextmanager
def _registry_lock(registry_root: Path):
    registry_root.mkdir(parents=True, exist_ok=True)
    lock_path = registry_root / ".result_registry.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _registration_fingerprint(
    *,
    attempt_manifest_sha256: str,
    source_summary_sha256: str,
    task_name: str,
    method: str,
    device: str,
) -> str:
    payload = "\0".join(
        (
            attempt_manifest_sha256,
            source_summary_sha256,
            task_name,
            method,
            device,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _append_ledger_records(path: Path, records: list[dict[str, Any]]) -> int:
    existing_ids: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            registration_id = str(row.get("registration_id") or "").strip()
            if registration_id:
                existing_ids.add(registration_id)
    new_records = [
        row
        for row in records
        if str(row.get("registration_id") or "").strip() not in existing_ids
    ]
    if not new_records:
        return 0
    with path.open("a", encoding="utf-8") as file:
        for row in new_records:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        file.flush()
        os.fsync(file.fileno())
    return len(new_records)


def register_attempt_summary(
    *,
    summary_path: Path,
    attempt_manifest_path: Path,
    runs_root: Path,
    local_data_index: Path | None = None,
) -> dict[str, Any]:
    """Register compact public rows and one detailed evidence block."""

    summary_path = summary_path.expanduser().resolve()
    attempt_manifest_path = attempt_manifest_path.expanduser().resolve()
    runs_root = runs_root.expanduser().resolve()
    summary = _load_json(summary_path)
    attempt_manifest = _load_json(attempt_manifest_path)
    if not isinstance(summary, dict):
        raise ValueError(f"summary must be a JSON object: {summary_path}")
    if (
        not isinstance(attempt_manifest, dict)
        or attempt_manifest.get("immutable") is not True
    ):
        raise ValueError(
            f"attempt manifest must declare immutable=true: {attempt_manifest_path}"
        )

    task_name = str(summary.get("task_name") or "").strip()
    attempt_id = str(attempt_manifest.get("attempt_id") or "").strip()
    if not task_name or not attempt_id:
        raise ValueError("task_name and attempt_id are required for result registration")
    source_seed = int(attempt_manifest.get("source_seed") or 0)
    evaluation_seed = int(attempt_manifest.get("evaluation_seed") or 0)
    source_summary_sha256 = sha256_file(summary_path)
    attempt_manifest_sha256 = sha256_file(attempt_manifest_path)
    commands_path = _commands_path_for_summary(summary_path)
    registered_at = _utc_now()
    rows = [row for row in summary.get("rows") or [] if isinstance(row, dict)]
    if not rows:
        raise ValueError(f"summary contains no result rows: {summary_path}")
    details = [
        row for row in summary.get("details") or [] if isinstance(row, dict)
    ]
    details_by_key = {
        (str(row.get("method") or ""), str(row.get("device") or "")): row
        for row in details
    }
    validation_rows = [
        details_by_key.get(
            (str(row.get("method") or ""), str(row.get("device") or "")), row
        )
        for row in rows
    ]
    public_rows = [
        compact_result_row(
            {
                **row,
                "task": str(row.get("task") or row.get("task_name") or task_name),
            },
            source_seed=source_seed,
            evaluation_seed=evaluation_seed,
        )
        for row in rows
    ]
    for row in validation_rows:
        reasons = formal_result_environment_failure_reasons(row)
        if reasons:
            raise ValueError(
                "formal_result_environment_failure:"
                f"{task_name}:{','.join(reasons)}"
            )
        if not has_official_validator_conclusion(row):
            # The official client can terminate with a protocol/method
            # failure before AndroidWorld emits a validator conclusion. This
            # is still a terminal experiment observation when no environment
            # failure was reported; keep it in the immutable registry so a
            # campaign can continue and resume can skip the same cell.
            terminal_method_failure = str(row.get("status") or "").strip().lower() in {
                "command_failed",
                "method_failed",
                "execution_failed",
            }
            if not terminal_method_failure:
                raise ValueError(
                    "official_validator_conclusion_missing:"
                    f"{task_name}:{row.get('method')}:{row.get('device')}"
                )

    ledger_records: list[dict[str, Any]] = []
    registered_paths: list[str] = []
    with _registry_lock(runs_root):
        for row, detail in zip(public_rows, validation_rows, strict=True):
            method = str(row.get("method") or "").strip()
            device = str(row.get("device") or "").strip()
            if not method or not device:
                raise ValueError(
                    f"result row must contain method and device: {summary_path}"
                )
            fingerprint = _registration_fingerprint(
                attempt_manifest_sha256=attempt_manifest_sha256,
                source_summary_sha256=source_summary_sha256,
                task_name=task_name,
                method=method,
                device=device,
            )
            safe_attempt = safe_component(
                attempt_id,
                fallback=fingerprint[:12],
                strip_chars="._-",
            )
            registration_id = ".".join(
                (
                    safe_component(task_name, fallback="task", strip_chars="._-"),
                    safe_component(method, fallback="method", strip_chars="._-"),
                    safe_component(device, fallback="device", strip_chars="._-"),
                    safe_attempt,
                    fingerprint[:12],
                )
            )
            destination = (
                runs_root
                / safe_component(task_name, fallback="task", strip_chars="._-")
                / safe_component(method, fallback="method", strip_chars="._-")
                / safe_component(device, fallback="device", strip_chars="._-")
                / safe_attempt
            )
            manifest_path = destination / "registration_manifest.json"
            result_path = destination / "registered_result.json"
            manifest = {
                "schema_version": _REGISTRATION_SCHEMA,
                "registration_id": registration_id,
                "fingerprint_sha256": fingerprint,
                "immutable": True,
                "task_name": task_name,
                "method": method,
                "device": device,
                "attempt_id": attempt_id,
                "source_seed": source_seed,
                "evaluation_seed": evaluation_seed,
                "source_summary": str(summary_path),
                "source_summary_sha256": source_summary_sha256,
                "attempt_manifest": str(attempt_manifest_path),
                "attempt_manifest_sha256": attempt_manifest_sha256,
                "source_commands": str(commands_path) if commands_path.exists() else "",
                "source_commands_sha256": (
                    sha256_file(commands_path) if commands_path.exists() else ""
                ),
                "registered_at": registered_at,
            }
            registered = {
                "schema_version": _REGISTERED_RESULT_SCHEMA,
                "registration_id": registration_id,
                "attempt_id": attempt_id,
                "task_name": task_name,
                "task_root": str(destination),
                "source_seed": source_seed,
                "evaluation_seed": evaluation_seed,
                "source_summary": str(summary_path),
                "source_summary_sha256": source_summary_sha256,
                "registration_manifest": str(manifest_path),
                "rows": [row],
                "details": [detail],
            }
            registered_text = (
                json.dumps(registered, indent=2, ensure_ascii=False) + "\n"
            )
            registered_sha256 = hashlib.sha256(
                registered_text.encode("utf-8")
            ).hexdigest()
            manifest["registered_result_sha256"] = registered_sha256

            if destination.exists():
                # Attempt directories are allocated monotonically. A stale
                # digest must not turn a runnable cell into a conflict.
                continue
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = Path(
                    tempfile.mkdtemp(
                        dir=destination.parent,
                        prefix=f".{destination.name}.",
                    )
                )
                try:
                    (temporary / "registration_manifest.json").write_text(
                        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                    (temporary / "registered_result.json").write_text(
                        registered_text,
                        encoding="utf-8",
                    )
                    os.replace(temporary, destination)
                finally:
                    shutil.rmtree(temporary, ignore_errors=True)

            ledger_records.append({**manifest, "registered_result": str(result_path)})
            registered_paths.append(str(result_path))

        appended = _append_ledger_records(runs_root / "registry.jsonl", ledger_records)

    local_data_updated = False
    local_data_update_error = ""
    if local_data_index is not None:
        from src.experiment.data_index import refresh_data_index_from_pointer

        try:
            refresh_data_index_from_pointer(
                memory_index=local_data_index,
                additional_result_roots=(runs_root,),
            )
        except ValueError as error:
            # Registration is already immutable and complete at this point.
            # A stale unrelated source in current.json must not turn a valid
            # task result into a process failure; the registry remains the
            # authoritative fallback for skip planning.
            if not str(error).startswith("indexed_source_run_log_invalid:"):
                raise
            local_data_update_error = str(error)
        else:
            local_data_updated = True

    result = {
        "task_name": task_name,
        "attempt_id": attempt_id,
        "registered_results_count": len(registered_paths),
        "ledger_records_appended": appended,
        "registered_results": registered_paths,
        "local_data_updated": local_data_updated,
    }
    if local_data_update_error:
        result["local_data_update_error"] = local_data_update_error
    return result


__all__ = [
    "formal_result_environment_failure_reasons",
    "has_official_validator_conclusion",
    "legacy_external_result_protocol_compatible",
    "LEGACY_EXTERNAL_METHODS",
    "mobilegpt_runtime_integrity_error",
    "register_attempt_summary",
    "registered_result_plan",
    "validate_formal_result_protocol",
]
