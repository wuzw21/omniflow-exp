"""Durable, process-safe capture for failed transfer attempts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from omniflow.core.model import Action, Observation, TransferResult


ERROR_POOL_ENV = "OMNIFLOW_TRANSFER_ERROR_POOL"
ERROR_POOL_FILENAME = "transfer_errors.jsonl"


def record_transfer_error(
    *,
    action: Action,
    result: TransferResult,
    source_page: Observation | None,
    target_page: Observation | None,
) -> None:
    """Append one failed mapping and its page pair to the shared error pool."""

    if result.action is not None:
        return
    try:
        source = _page_descriptor(source_page)
        target = _page_descriptor(target_page)
        record = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "context": _error_context(),
            "tool": action.tool,
            "action": action.to_dict(),
            "reason": result.reason or "transfer_failed",
            "page_pair": {
                "source_page": source,
                "target_page": target,
                "complete": bool(
                    source
                    and target
                    and source.get("xml_present")
                    and target.get("xml_present")
                ),
            },
            "transfer_detail": result.detail or {},
        }
        payload = (json.dumps(record, ensure_ascii=False, default=str) + "\n").encode(
            "utf-8"
        )
        path = _error_pool_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(descriptor, payload)
        finally:
            os.close(descriptor)
    except Exception:  # noqa: BLE001 - diagnostics must never alter execution
        return


def _error_pool_path() -> Path:
    configured = str(os.environ.get(ERROR_POOL_ENV) or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    current = Path.cwd().resolve()
    for root in (current, *current.parents):
        if (root / "data" / "androidworld").is_dir():
            return root / "data" / "androidworld" / ERROR_POOL_FILENAME
    return Path(tempfile.gettempdir()) / "omniflow" / ERROR_POOL_FILENAME


def _error_context() -> dict[str, Any]:
    raw = str(os.environ.get("OMNIFLOW_TRANSFER_ERROR_CONTEXT") or "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}
    return value if isinstance(value, dict) else {"raw": raw}


def _page_descriptor(observation: Observation | None) -> dict[str, Any] | None:
    if observation is None:
        return None
    xml = str(observation.xml or "")
    extra = observation.extra if isinstance(observation.extra, dict) else {}
    explicit_state_id = str(extra.get("state_id") or "").strip()
    if explicit_state_id:
        state_id = explicit_state_id
    else:
        identity = json.dumps(
            {
                "xml": xml,
                "package_name": observation.package_name,
                "activity_name": observation.activity_name,
                "display": extra.get("display") or {},
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        state_id = "state_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    display = extra.get("display")
    return {
        "state_id": state_id,
        "package_name": str(observation.package_name or ""),
        "activity_name": str(observation.activity_name or ""),
        "display": dict(display) if isinstance(display, dict) else {},
        "xml_present": bool(xml),
        "xml_sha256": hashlib.sha256(xml.encode("utf-8")).hexdigest() if xml else "",
        "screenshot_path": str(extra.get("screenshot_path") or ""),
    }


__all__ = ["ERROR_POOL_ENV", "ERROR_POOL_FILENAME", "record_transfer_error"]
