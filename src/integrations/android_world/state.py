from __future__ import annotations

import base64
import binascii
import dataclasses
import hashlib
import io
import math
from pathlib import Path
from typing import Any

from PIL import Image

ANDROIDWORLD_STATE_FIELDS = ("pixels", "forest", "ui_elements", "auxiliaries")


def snapshot_androidworld_state(
    state: Any,
    *,
    evidence_root: str | Path | None,
) -> dict[str, Any]:
    missing = [field for field in ANDROIDWORLD_STATE_FIELDS if not hasattr(state, field)]
    if missing:
        raise ValueError("androidworld_state_fields_missing:" + ",".join(missing))
    pixels = getattr(state, "pixels")
    return {
        "pixels": _screenshot_reference(pixels, evidence_root=evidence_root),
        "forest": _json_value(getattr(state, "forest")),
        "ui_elements": _json_value(list(getattr(state, "ui_elements") or ())),
        "auxiliaries": _json_value(getattr(state, "auxiliaries")),
    }


def _screenshot_reference(
    pixels: Any,
    *,
    evidence_root: str | Path | None,
) -> dict[str, Any] | None:
    if pixels is None:
        return None
    if evidence_root is None:
        raise ValueError("androidworld_state_evidence_root_required")
    image_bytes, width, height = _png_bytes(pixels)
    digest = hashlib.sha256(image_bytes).hexdigest()
    root = Path(evidence_root).expanduser().resolve()
    destination = root / "observations" / "objects" / f"{digest}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable(destination, image_bytes)
    return {
        "path": str(destination),
        "sha256": digest,
        "width": width,
        "height": height,
        "mime_type": "image/png",
    }


def _png_bytes(pixels: Any) -> tuple[bytes, int, int]:
    if isinstance(pixels, str):
        encoded = pixels.split(",", 1)[-1]
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("androidworld_state_pixels_base64_invalid") from error
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
    return output.getvalue(), int(width), int(height)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("androidworld_state_number_not_finite")
        return value
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return _json_value(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    descriptor = getattr(value, "DESCRIPTOR", None)
    if descriptor is not None:
        from google.protobuf.json_format import MessageToDict

        return MessageToDict(value, preserving_proto_field_name=True)
    scalar = getattr(value, "item", None)
    if callable(scalar):
        try:
            return _json_value(scalar())
        except ValueError:
            raise
        except Exception:
            pass
    array = getattr(value, "tolist", None)
    if callable(array):
        return _json_value(array())
    enum_value = getattr(value, "value", None)
    if enum_value is not None and enum_value is not value:
        return _json_value(enum_value)
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return {
            str(key): _json_value(item)
            for key, item in attributes.items()
            if not str(key).startswith("_")
        }
    raise TypeError(f"androidworld_state_value_not_serializable:{type(value).__name__}")


def _write_immutable(path: Path, content: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise ValueError(f"androidworld_state_screenshot_hash_collision:{path}")


__all__ = [
    "ANDROIDWORLD_STATE_FIELDS",
    "snapshot_androidworld_state",
]
