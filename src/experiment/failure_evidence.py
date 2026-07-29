"""Persist failed AndroidWorld observations without changing method behavior."""

from __future__ import annotations

import base64
import binascii
import hashlib
from pathlib import Path
from typing import Any, Sequence


def write_failure_observations(
    output_dir: str | Path,
    *,
    task_name: str,
    run_id: str,
    observations: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Write immutable screenshot objects and return lightweight event records."""
    if not str(task_name or "").strip():
        raise ValueError("failure_evidence_task_name_required")
    if not str(run_id or "").strip():
        raise ValueError("failure_evidence_run_id_required")
    root = Path(output_dir).expanduser().resolve()
    object_dir = root / "failure_evidence" / "objects"
    records: list[dict[str, Any]] = []
    for observation in observations:
        if not isinstance(observation, dict):
            raise ValueError("failure_observation_must_be_object")
        image = _decode_image(observation.get("image_base64"))
        digest = hashlib.sha256(image).hexdigest()
        relative_path = Path("failure_evidence") / "objects" / f"{digest}.png"
        destination = root / relative_path
        object_dir.mkdir(parents=True, exist_ok=True)
        _write_immutable(destination, image)
        record = {
            key: observation[key]
            for key in (
                "event",
                "step_index",
                "error",
                "state_id",
                "package_name",
                "activity_name",
                "display",
            )
            if observation.get(key) is not None
        }
        record.update(
            {
                "path": relative_path.as_posix(),
                "sha256": digest,
            }
        )
        records.append(record)
    return records


def _decode_image(value: Any) -> bytes:
    text = str(value or "").strip()
    if not text:
        raise ValueError("failure_observation_image_required")
    if text.startswith("data:") and "," in text:
        text = text.split(",", 1)[1]
    try:
        image = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("failure_observation_image_invalid") from error
    if not image.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("failure_observation_image_not_png")
    return image


def _write_immutable(path: Path, content: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise ValueError(f"failure_evidence_hash_collision:{path}")


__all__ = ["write_failure_observations"]
