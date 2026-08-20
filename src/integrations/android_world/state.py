from __future__ import annotations

import base64
import binascii
import io
from pathlib import Path
from typing import Any

from PIL import Image

from omniflow.core.androidworld_accessibility import androidworld_forest_xml

ANDROIDWORLD_STATE_FIELDS = ("pixels", "forest", "ui_elements")


def snapshot_androidworld_state(
    state: Any,
    *,
    evidence_root: str | Path | None,
) -> dict[str, Any]:
    missing = [field for field in ANDROIDWORLD_STATE_FIELDS if not hasattr(state, field)]
    if missing:
        raise ValueError("androidworld_state_fields_missing:" + ",".join(missing))
    pixels = getattr(state, "pixels")
    pixels_reference = _screenshot_reference(pixels, evidence_root=evidence_root)
    xml = _state_xml(state, pixels_reference)
    if not xml:
        return {
            "pixels": pixels_reference,
            "forest": None,
            "ui_elements": [],
            "auxiliaries": None,
        }
    return {
        "screenshot": pixels_reference,
        "xml": xml,
    }


def _state_xml(
    state: Any,
    pixels_reference: dict[str, Any] | None,
) -> str:
    forest = getattr(state, "forest", None)
    if isinstance(forest, str) and forest.strip():
        forest_xml = forest.strip()
    elif forest is not None:
        width, height = _display_size(state, pixels_reference)
        forest_xml = androidworld_forest_xml(
            forest,
            screen_size=(width, height),
        ).strip()
    else:
        forest_xml = ""
    elements = list(getattr(state, "ui_elements", ()) or ())
    if elements:
        # The host owns the shared element-to-XML projection.  This import is
        # deliberately lazy because the host imports this state module.
        from src.integrations.android_world.host import (
            _xml_semantic_score,
            androidworld_elements_xml,
        )

        elements_xml = androidworld_elements_xml(elements).strip()
        if forest_xml and _xml_semantic_score(forest_xml) >= _xml_semantic_score(
            elements_xml
        ):
            return forest_xml
        return elements_xml
    return forest_xml


def _display_size(
    state: Any,
    pixels_reference: dict[str, Any] | None,
) -> tuple[int, int]:
    if isinstance(pixels_reference, dict):
        return int(pixels_reference["width"]), int(pixels_reference["height"])
    auxiliaries = getattr(state, "auxiliaries", None)
    display = auxiliaries.get("display") if isinstance(auxiliaries, dict) else None
    if isinstance(display, dict):
        width = display.get("width")
        height = display.get("height")
        if (
            isinstance(width, (int, float))
            and width > 0
            and isinstance(height, (int, float))
            and height > 0
        ):
            return int(width), int(height)
    return 1000, 1000


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
    root = Path(evidence_root).expanduser().resolve()
    screenshot_root = root / "screenshots"
    screenshot_root.mkdir(parents=True, exist_ok=True)
    destination = _write_next_screenshot(screenshot_root, image_bytes)
    return {
        "path": str(destination),
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


def _write_next_screenshot(root: Path, content: bytes) -> Path:
    """Write one screenshot with a stable sequential name, never a hash name."""

    index = 1
    while True:
        path = root / f"screenshot_{index:06d}.png"
        try:
            with path.open("xb") as handle:
                handle.write(content)
            return path
        except FileExistsError:
            index += 1


__all__ = [
    "ANDROIDWORLD_STATE_FIELDS",
    "snapshot_androidworld_state",
]
