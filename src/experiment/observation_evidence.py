"""Capture every AndroidWorld observation and persist immutable screenshots."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Callable

from PIL import Image


class ObservationArchive:
    """Transparent ``get_state`` adapter with ordered screenshot persistence."""

    def __init__(self, get_state: Callable[..., Any]):
        if not callable(get_state):
            raise TypeError("observation_get_state_callable_required")
        self._get_state = get_state
        self._observations: list[dict[str, Any]] = []

    def get_state(self, *args: Any, **kwargs: Any) -> Any:
        state = self._get_state(*args, **kwargs)
        self._observations.append(
            _snapshot_observation(
                state,
                observation_index=len(self._observations),
            )
        )
        return state

    def persist(self, output_dir: str | Path) -> list[dict[str, Any]]:
        root = Path(output_dir).expanduser().resolve()
        observation_dir = root / "observations"
        object_dir = observation_dir / "objects"
        records: list[dict[str, Any]] = []
        for observation in self._observations:
            record = {
                key: observation[key]
                for key in (
                    "observation_index",
                    "state_id",
                    "package_name",
                    "activity_name",
                )
                if observation.get(key) is not None
            }
            capture_error = str(observation.get("error") or "").strip()
            if capture_error:
                record["error"] = capture_error
                records.append(record)
                continue
            try:
                image, display = _png_bytes(observation["pixels"])
                digest = hashlib.sha256(image).hexdigest()
                relative_path = Path("observations") / "objects" / f"{digest}.png"
                destination = root / relative_path
                object_dir.mkdir(parents=True, exist_ok=True)
                _write_immutable(destination, image)
                record.update(
                    {
                        "display": display,
                        "path": relative_path.as_posix(),
                        "sha256": digest,
                    }
                )
            except (OSError, TypeError, ValueError) as error:
                record["error"] = f"observation_image_encode_failed:{error}"
            records.append(record)
        observation_dir.mkdir(parents=True, exist_ok=True)
        index = {
            "schema_version": "omniflow.androidworld-observations.v1",
            "observation_count": len(records),
            "observations": records,
        }
        _write_immutable(
            observation_dir / "index.json",
            (
                json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8"),
        )
        return records


def _snapshot_observation(
    state: Any,
    *,
    observation_index: int,
) -> dict[str, Any]:
    observation: dict[str, Any] = {"observation_index": int(observation_index)}
    for key in ("state_id", "package_name", "activity_name"):
        value = _read(state, key)
        text = str(value or "").strip()
        if text:
            observation[key] = text
    pixels = _read(state, "pixels")
    if pixels is None:
        observation["error"] = "observation_image_missing"
        return observation
    try:
        if isinstance(pixels, str):
            observation["pixels"] = str(pixels)
        elif isinstance(pixels, (bytes, bytearray)):
            observation["pixels"] = bytes(pixels)
        else:
            copy = getattr(pixels, "copy", None)
            observation["pixels"] = copy() if callable(copy) else pixels
    except Exception as error:  # noqa: BLE001
        observation["error"] = f"observation_image_copy_failed:{error}"
    return observation


def _png_bytes(pixels: Any) -> tuple[bytes, dict[str, int]]:
    if isinstance(pixels, str):
        encoded = pixels.split(",", 1)[-1]
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("base64_invalid") from error
        image = Image.open(io.BytesIO(raw))
    elif isinstance(pixels, (bytes, bytearray)):
        image = Image.open(io.BytesIO(bytes(pixels)))
    elif isinstance(pixels, Image.Image):
        image = pixels
    else:
        image = Image.fromarray(pixels)
    image.load()
    output = io.BytesIO()
    image.save(output, format="PNG")
    width, height = image.size
    return output.getvalue(), {"width": int(width), "height": int(height)}


def _read(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _write_immutable(path: Path, content: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise ValueError(f"observation_evidence_hash_collision:{path}")


__all__ = ["ObservationArchive"]
