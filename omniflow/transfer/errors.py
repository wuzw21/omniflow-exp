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
from omniflow.runtime.control import invoke
from omniflow.runtime.timing import measure
from omniflow.transfer.failure_inputs import save_failure_inputs


ERROR_POOL_ENV = "OMNIFLOW_TRANSFER_ERROR_POOL"
ERROR_POOL_FILENAME = "transfer_errors.jsonl"


async def attempt_transfer(transfer, action, target, source, *, phase):
    """One mapping boundary for recall, fast execution and stable retry."""
    try:
        with measure("transfer"):
            result = await invoke(transfer, action, target, source)
        if not isinstance(result, TransferResult):
            result = TransferResult(None, reason="transfer_result_invalid")
    except Exception as error:
        result = TransferResult(None, reason=f"transfer_exception:{type(error).__name__}",
                                detail={"exception_type": type(error).__name__})
    if result.action is None:
        with measure("evidence.failure_pair"):
            await invoke(record_transfer_error, action=action, result=result, source_page=source,
                         target_page=target, phase=phase)
    return result


def record_transfer_error(
    *,
    action: Action,
    result: TransferResult,
    source_page: Observation | None,
    target_page: Observation | None,
    phase: str = "mapping",
) -> None:
    """Append one failed mapping and its page pair to the shared error pool."""

    if result.action is not None:
        return
    try:
        path = _error_pool_path()
        assets = path.parent / (path.stem + "_assets")
        try:
            if result.detail.get("replay_unavailable"):
                raise ValueError("preprocessing_context_required")
            reference = save_failure_inputs(assets, action, source_page, target_page)
            replay = {"status": "ready", "path": str(Path(assets.name) / reference)}
        except Exception as error:
            replay = {"status": "unavailable", "reason":
                      "preprocessing_context_required" if result.detail.get("replay_unavailable")
                      else type(error).__name__}
        source = _page_descriptor(source_page)
        target = _page_descriptor(target_page)
        action_identity = hashlib.sha256(json.dumps(action.to_dict(), sort_keys=True).encode()).hexdigest()
        pair_identity = {"source": (source or {}).get("identity_sha256"),
                         "target": (target or {}).get("identity_sha256"), "action": action_identity}
        record = {
            "schema_version": "omniflow.failure-pair.v1",
            "pair_id": hashlib.sha256(json.dumps(pair_identity, sort_keys=True).encode()).hexdigest(),
            "phase": phase,
            "replay": replay,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "context": _error_context(),
            "tool": action.tool,
            "action": {"tool": action.tool, "sha256": action_identity,
                       "args": {k: v if isinstance(v, (int, float, bool)) or v is None
                                else {"sha256": hashlib.sha256(str(v).encode()).hexdigest(), "length": len(str(v))}
                                for k, v in action.args.items()}},
            "reason": str(result.reason or "transfer_failed")[:256],
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
            "transfer_detail": _compact_detail(result.detail or {}),
        }
        payload = (json.dumps(record, ensure_ascii=False, default=str) + "\n").encode(
            "utf-8"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(descriptor, payload)
        finally:
            os.close(descriptor)
    except Exception:  # noqa: BLE001 - diagnostics must never alter execution
        return


def _compact_detail(detail, depth=0):
    if not isinstance(detail, dict):
        return {}
    allowed = {"reason", "mapped", "score", "confidence", "page_similarity", "pair_confidence",
               "absolute_contextual_confidence", "target_candidate_id", "target_execution_candidate_id",
               "target_bbox", "bbox", "execution_bbox", "executable", "exception_type",
               "matcher_release", "matcher_checkpoint_sha256", "mapping_mode"}
    result = {k: v for k, v in detail.items() if k in allowed and
              (v is None or isinstance(v, (bool, int, float)) or isinstance(v, str) and len(v) <= 256
               or isinstance(v, (list, tuple)) and len(v) <= 4 and all(isinstance(x, (int, float)) for x in v))}
    candidates = detail.get("candidates")
    if isinstance(candidates, list) and depth < 1:
        result["candidates"] = [_compact_detail(c, depth+1) for c in candidates[:8] if isinstance(c, dict)]
        result["candidate_count"] = len(candidates)
    return result


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
        value = None
    # Context is an audit label, never a second payload store.
    labels = {"task", "method", "device", "run_id", "function_id", "step_index"}
    result = {key: item for key, item in (value.items() if isinstance(value, dict) else ())
              if key in labels and (item is None or isinstance(item, (bool, int, float))
                                    or isinstance(item, str) and len(item) <= 256)}
    if value != result:
        result["context_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    return result


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
        "identity_sha256": hashlib.sha256(json.dumps({"xml": xml,
            "package_name": observation.package_name, "activity_name": observation.activity_name,
            "display": display or {}}, sort_keys=True, default=str).encode()).hexdigest(),
        "state_id": state_id,
        "package_name": str(observation.package_name or ""),
        "activity_name": str(observation.activity_name or ""),
        "display": {key: value for key, value in (display.items() if isinstance(display, dict) else ())
                    if key in {"width", "height", "rotation", "density", "density_dpi"}
                    and isinstance(value, (int, float)) and not isinstance(value, bool)},
        "xml_present": bool(xml),
        "xml_sha256": hashlib.sha256(xml.encode("utf-8")).hexdigest() if xml else "",
        "screenshot_path": str(extra.get("screenshot_path") or ""),
    }


__all__ = ["ERROR_POOL_ENV", "ERROR_POOL_FILENAME", "record_transfer_error", "attempt_transfer"]
