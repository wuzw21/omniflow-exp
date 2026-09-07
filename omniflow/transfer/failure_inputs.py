"""Content-addressed inputs for replaying a failed mapping at its original seam.

Only failing attempts use this store. No task history, candidate tensors, model
weights or inferred target labels are retained. Loading verifies every blob.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile

from omniflow.core.model import Action, Observation

MAX_BLOB_BYTES = 32 * 1024 * 1024


def _put(root: Path, data: bytes, suffix: str) -> str:
    if len(data) > MAX_BLOB_BYTES:
        raise ValueError("failure_input_too_large")
    root.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(data).hexdigest() + suffix
    path = root / name
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("failure_input_hash_conflict")
        return name
    descriptor, temporary = tempfile.mkstemp(dir=root, prefix=".input-")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return name


def _get(root: Path, name: str) -> bytes:
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("failure_input_reference_invalid")
    path = root / name
    if path.stat().st_size > MAX_BLOB_BYTES:
        raise ValueError("failure_input_too_large")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != name.split(".", 1)[0]:
        raise ValueError("failure_input_hash_mismatch")
    return data


def _put_json(root: Path, value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode()
    if len(raw) > MAX_BLOB_BYTES:
        raise ValueError("failure_input_too_large")
    return _put(root, gzip.compress(raw, mtime=0), ".json.gz")


def _get_json(root: Path, name: str):
    from io import BytesIO
    with gzip.GzipFile(fileobj=BytesIO(_get(root, name))) as stream:
        raw = stream.read(MAX_BLOB_BYTES + 1)
    if len(raw) > MAX_BLOB_BYTES:
        raise ValueError("failure_input_too_large")
    return json.loads(raw)


def _save_page(root: Path, observation: Observation | None):
    if observation is None:
        return None
    # Deep copy JSON data: adapters may retain the original observation.
    original_value = observation.to_dict()
    encoded = original_value.pop("image_base64", None)
    value = json.loads(json.dumps(original_value, allow_nan=False))
    extra = value["extra"]
    media = {}
    for label, parent, key in (
        ("screenshot", extra, "screenshot_path"),
        ("pixels", (extra.get("androidworld_state") or {}).get("pixels") or {}, "path"),
        ("canonical_screenshot", (extra.get("androidworld_state") or {}).get("screenshot") or {}, "path"),
    ):
        original = parent.get(key)
        if original:
            path = Path(original).expanduser()
            if path.stat().st_size > MAX_BLOB_BYTES:
                raise ValueError("failure_input_too_large")
            media[label] = _put(root, path.read_bytes(), path.suffix or ".image")
            parent[key] = None
    if encoded is not None:
        # Preserve the exact representation for custom adapters; PNG/path blobs
        # remain independently content-addressed for the canonical mapper.
        if len(encoded.encode()) > MAX_BLOB_BYTES:
            raise ValueError("failure_input_too_large")
        media["image_base64"] = _put(root, gzip.compress(encoded.encode(), mtime=0), ".b64.gz")
    return _put_json(root, {"observation": value, "media": media})


def _load_page(root: Path, reference):
    if reference is None:
        return None
    payload = _get_json(root, reference)
    value, media = payload["observation"], payload["media"]
    extra = value["extra"]
    for label, parent, key in (
        ("screenshot", extra, "screenshot_path"),
        ("pixels", (extra.get("androidworld_state") or {}).get("pixels") or {}, "path"),
        ("canonical_screenshot", (extra.get("androidworld_state") or {}).get("screenshot") or {}, "path"),
    ):
        if label in media:
            _get(root, media[label])
            parent[key] = str((root / media[label]).resolve())
    if "image_base64" in media:
        from io import BytesIO
        with gzip.GzipFile(fileobj=BytesIO(_get(root, media["image_base64"]))) as stream:
            raw = stream.read(MAX_BLOB_BYTES + 1)
        if len(raw) > MAX_BLOB_BYTES:
            raise ValueError("failure_input_too_large")
        value["image_base64"] = raw.decode()
    return Observation.from_value(value)


def save_failure_inputs(root: Path, action: Action, source: Observation | None,
                        target: Observation | None) -> str:
    return _put_json(root, {
        "schema_version": "omniflow.failure-inputs.v1",
        "action": action.to_dict(),
        "source": _save_page(root, source),
        "target": _save_page(root, target),
    })


def load_failure_inputs(path: str | Path):
    """Load one explicit input reference, never discover inputs from history."""
    path = Path(path)
    payload = _get_json(path.parent, path.name)
    if payload.get("schema_version") != "omniflow.failure-inputs.v1":
        raise ValueError("failure_inputs_version_invalid")
    return (Action.from_value(payload["action"]), _load_page(path.parent, payload["target"]),
            _load_page(path.parent, payload["source"]))


async def replay_failure_inputs(path: str | Path, *, transfer=None):
    """Read-only mapping replay; never dispatch a device action or label success."""
    from omniflow.runtime.execution import default_transfer
    from omniflow.transfer.errors import attempt_transfer
    action, target, source = load_failure_inputs(path)
    return await attempt_transfer(transfer or default_transfer, action, target, source, phase="replay")
