"""Small bridge to MobileGPT's official memory file format.

This module does not run MobileGPT and does not change its prompts or actions.
It only calls the pinned upstream XML encoder while compiling a source RunLog
into the files that the upstream ``Memory`` class already understands.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import sys
import tempfile
from typing import Iterator
import xml.etree.ElementTree as ET


def _server_root(root: str | Path | None = None) -> Path:
    configured = str(root or os.environ.get("MOBILEGPT_ROOT") or "").strip()
    candidates = [
        Path(configured).expanduser() if configured else None,
        Path("/Users/wuzewen/Projects/MobileGPT"),
        Path("/Users/wuzewen/Projects/Omni/OmniFlow/runtime/external/mobilegpt-official"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        server = candidate / "Server"
        if server.is_dir():
            return server.resolve()
    raise FileNotFoundError(
        "MobileGPT official checkout is required; set MOBILEGPT_ROOT to its root"
    )


@contextmanager
def _official_import_path(server_root: Path) -> Iterator[None]:
    text = str(server_root)
    inserted = text not in sys.path
    if inserted:
        sys.path.insert(0, text)
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(text)
            except ValueError:
                pass


def encode_xml(
    raw_xml: str,
    *,
    mobilegpt_root: str | Path | None = None,
) -> tuple[str, str, str]:
    """Return ``(parsed_xml, hierarchy_xml, encoded_xml)`` from upstream.

    MobileGPT's encoder writes diagnostic XML files beside its input.  A
    temporary directory keeps that implementation detail out of the source
    checkout while preserving exactly the upstream transformation.
    """

    server_root = _server_root(mobilegpt_root)
    source_xml = _as_mobilegpt_input_xml(raw_xml)
    with _official_import_path(server_root):
        from screenParser.Encoder import xmlEncoder

        with tempfile.TemporaryDirectory(prefix="omniflow-mobilegpt-") as work:
            encoder = xmlEncoder()
            encoder.init(work)
            return tuple(encoder.encode(source_xml, 0))  # type: ignore[return-value]


def _as_mobilegpt_input_xml(raw_xml: str) -> str:
    """Fill only the fields emitted by MobileGPT's Android XML dumper.

    AndroidWorld names a few accessibility attributes differently and some
    recorded fixtures omit indexes.  The official MobileGPT parser expects
    the dumper's names and preorder indexes; no action or prompt is invented
    here.
    """

    root = ET.fromstring(str(raw_xml or "").strip())
    next_index = 0
    for element in root.iter():
        if element is root:
            continue
        attributes = element.attrib
        attributes.setdefault("index", str(next_index))
        next_index += 1
        if not attributes.get("id"):
            value = attributes.get("resource-id") or attributes.get("resource_id")
            if value:
                attributes["id"] = value
        if not attributes.get("description"):
            value = attributes.get("content-desc") or attributes.get(
                "content_description"
            )
            if value:
                attributes["description"] = value
        if not attributes.get("class"):
            if str(attributes.get("editable") or "").lower() == "true":
                attributes["class"] = "android.widget.EditText"
            elif str(attributes.get("checkable") or "").lower() == "true":
                attributes["class"] = "android.widget.CheckBox"
            elif str(attributes.get("clickable") or "").lower() == "true":
                attributes["class"] = "android.widget.Button"
            elif str(attributes.get("scrollable") or "").lower() == "true":
                attributes["class"] = "android.widget.ScrollView"
            elif len(element):
                attributes["class"] = "android.view.ViewGroup"
            else:
                attributes["class"] = "android.widget.TextView"
        if (
            "important" not in attributes
            and (
                str(attributes.get("editable") or "").lower() == "true"
                or str(attributes.get("class") or "").endswith("EditText")
            )
        ):
            attributes["important"] = "true"
    return ET.tostring(root, encoding="unicode")


__all__ = ["encode_xml"]
