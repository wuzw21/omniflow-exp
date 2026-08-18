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
from typing import Any

from src.experiment.protocol import DEVICES, RESULT_COMMANDS_FILE
from src.experiment.result_schema import compact_result_row

DEVICE_TARGETS = {label: (serial, port) for label, serial, port in DEVICES}
_REGISTERED_RESULT_SCHEMA = "omniflow.androidworld_registered_result.v1"
_REGISTRATION_SCHEMA = "omniflow.androidworld_result_registration.v1"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _commands_path_for_summary(summary_path: Path) -> Path:
    current = summary_path.with_name(RESULT_COMMANDS_FILE)
    if current.exists():
        return current
    return summary_path.with_name("one_task_commands.jsonl")


def _safe_component(value: str, *, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip(
        "._-"
    )
    return normalized or fallback


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    expected_sha256 = str(manifest.get("registered_result_sha256") or "")
    if not expected_sha256 or _sha256(summary_path) != expected_sha256:
        raise ValueError(f"registered result checksum mismatch: {summary_path}")
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
    if not isinstance(task_params, dict) or not re.fullmatch(
        r"[0-9a-f]{64}", params_sha256
    ):
        violations.append("task_params_sha256")
    elif hashlib.sha256(
        json.dumps(
            task_params,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest() != params_sha256:
        violations.append("task_params_hash_mismatch")

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
    source_summary_sha256 = _sha256(summary_path)
    attempt_manifest_sha256 = _sha256(attempt_manifest_path)
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
            safe_attempt = _safe_component(attempt_id, fallback=fingerprint[:12])
            registration_id = ".".join(
                (
                    _safe_component(task_name, fallback="task"),
                    _safe_component(method, fallback="method"),
                    _safe_component(device, fallback="device"),
                    safe_attempt,
                    fingerprint[:12],
                )
            )
            destination = (
                runs_root
                / _safe_component(task_name, fallback="task")
                / _safe_component(method, fallback="method")
                / _safe_component(device, fallback="device")
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
                    _sha256(commands_path) if commands_path.exists() else ""
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
                existing_manifest = _load_json(manifest_path)
                if existing_manifest.get("fingerprint_sha256") != fingerprint:
                    raise FileExistsError(
                        f"immutable result registration conflict: {destination}"
                    )
                if _sha256(result_path) != registered_sha256:
                    raise ValueError(
                        f"registered result checksum mismatch: {result_path}"
                    )
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
    if local_data_index is not None:
        from src.experiment.local_data import refresh_local_data_from_pointer

        refresh_local_data_from_pointer(
            memory_index=local_data_index,
            additional_result_roots=(runs_root,),
        )
        local_data_updated = True

    return {
        "task_name": task_name,
        "attempt_id": attempt_id,
        "registered_results_count": len(registered_paths),
        "ledger_records_appended": appended,
        "registered_results": registered_paths,
        "local_data_updated": local_data_updated,
    }


__all__ = [
    "formal_result_environment_failure_reasons",
    "has_official_validator_conclusion",
    "mobilegpt_runtime_integrity_error",
    "register_attempt_summary",
    "registered_result_plan",
    "validate_formal_result_protocol",
]
