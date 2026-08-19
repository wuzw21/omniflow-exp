"""The single page representation used by OmniFlow.

Page identity is owned by the canonical OmniTransfer checkout. OmniFlow only
adapts its page-level vector to the local observation contract; it does not
maintain a second encoder or a local pooling method.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from omniflow.core.model import Observation

_LATEST_PAGE_CHECKPOINT_SHA256 = (
    "0494224f76c410f17d47b4aaaeacf99e2060c1174da628884c287a6922882ada"
)


@dataclass(frozen=True)
class PageEmbedding:
    """One OmniTransfer page vector and its immutable provenance."""

    vector: np.ndarray
    element_count: int
    encoder_version: str
    checkpoint_path: str
    checkpoint_sha256: str

    def similarity(self, other: "PageEmbedding") -> float:
        denominator = float(np.linalg.norm(self.vector) * np.linalg.norm(other.vector))
        if denominator <= 0.0:
            return 0.0
        return max(
            0.0,
            min(1.0, float(np.dot(self.vector, other.vector) / denominator)),
        )


def _canonical_omnitransfer_root(configured_root: str | Path | None = None) -> Path:
    configured = str(
        configured_root
        if configured_root is not None
        else os.environ.get("OMNITRANSFER_ROOT") or ""
    ).strip()
    root = (
        Path(configured).expanduser()
        if configured
        else Path.home() / "Projects" / "Omni" / "OmniTransfer"
    )
    root = root.resolve()
    canonical = (Path.home() / "Projects" / "Omni" / "OmniTransfer").resolve()
    if root != canonical:
        raise ValueError(f"canonical_omnitransfer_root_required:{canonical}")
    if not (root / "src" / "omnitransfer").is_dir():
        raise RuntimeError(f"omnitransfer_root_missing:{root}")
    return root


def _latest_page_checkpoint(root: Path) -> Path:
    return (
        root
        / "src"
        / "omnitransfer"
        / "checkpoints"
        / "omnitransfer_unified_association_v1_20260819"
        / "relation_slots_l3_h64_seed17.npz"
    )


class OmniTransferPageEncoder:
    """Use OmniTransfer's canonical frozen page embedding implementation."""

    name = "omnitransfer_page_embedding"

    def __init__(
        self,
        *,
        omnitransfer_root: str | Path | None = None,
        checkpoint: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        root = _canonical_omnitransfer_root(omnitransfer_root)
        selected_checkpoint = (
            Path(checkpoint).expanduser().resolve()
            if checkpoint is not None
            else _latest_page_checkpoint(root)
        )
        if selected_checkpoint != _latest_page_checkpoint(root):
            raise ValueError("canonical_omnitransfer_page_checkpoint_required")
        if not selected_checkpoint.is_file():
            raise FileNotFoundError(
                f"omnitransfer_page_checkpoint_missing:{selected_checkpoint}"
            )
        checkpoint_sha256 = hashlib.sha256(selected_checkpoint.read_bytes()).hexdigest()
        if checkpoint_sha256 != _LATEST_PAGE_CHECKPOINT_SHA256:
            raise ValueError("omnitransfer_page_checkpoint_checksum_mismatch")
        source_root = str(root / "src")
        package_root = root / "src" / "omnitransfer"
        sys.path[:] = [
            item
            for item in sys.path
            if str(Path(item or ".").resolve()) != source_root
        ]
        sys.path.insert(0, source_root)
        importlib.invalidate_caches()
        for module_name in tuple(sys.modules):
            if module_name == "omnitransfer" or module_name.startswith(
                "omnitransfer."
            ):
                module = sys.modules[module_name]
                module_path = Path(str(getattr(module, "__file__", ""))).resolve()
                if package_root not in module_path.parents:
                    del sys.modules[module_name]
        embedder_module = importlib.import_module("omnitransfer.page_embedding")
        self._embedder = embedder_module.OmniTransferPageEmbedder(
            selected_checkpoint,
            device=device,
        )
        self.checkpoint_path = selected_checkpoint
        self.checkpoint_sha256 = checkpoint_sha256
        self.encoder_version = (
            f"{self._embedder.architecture}:{self._embedder.text_encoder}:"
            f"{self._embedder.backend}"
        )

    @property
    def dimension(self) -> int:
        return int(self._embedder.embedding_dim)

    def embed(self, value: Observation | dict[str, Any] | str) -> PageEmbedding:
        observation = (
            Observation(xml=value)
            if isinstance(value, str)
            else Observation.from_value(value)
        )
        xml = str(observation.xml or "")
        if not xml.strip():
            raise ValueError("omnitransfer_page_xml_required")
        pixels: dict[str, Any] = {}
        androidworld_state = observation.extra.get("androidworld_state")
        if isinstance(androidworld_state, dict):
            raw_pixels = androidworld_state.get("pixels")
            if isinstance(raw_pixels, dict):
                pixels.update(raw_pixels)
        screenshot_path = observation.extra.get("screenshot_path")
        if isinstance(screenshot_path, str) and screenshot_path.strip():
            pixels["path"] = screenshot_path
        graph_id = hashlib.sha256(xml.encode("utf-8")).hexdigest()[:20]
        embedded = self._embedder.embed(xml, graph_id=graph_id, pixels=pixels)
        vector = np.asarray(embedded.vector, dtype=np.float32)
        if vector.ndim != 1 or vector.shape[0] != self._embedder.embedding_dim:
            raise ValueError("omnitransfer_page_embedding_shape_invalid")
        if not np.all(np.isfinite(vector)) or np.linalg.norm(vector) <= 0.0:
            raise ValueError("omnitransfer_page_embedding_unusable")
        return PageEmbedding(
            vector=vector,
            element_count=int(embedded.node_count),
            encoder_version=self.encoder_version,
            checkpoint_path=str(self.checkpoint_path),
            checkpoint_sha256=self.checkpoint_sha256,
        )

    def similarity(
        self,
        source: Observation | dict[str, Any] | str,
        current: Observation | dict[str, Any] | str,
    ) -> float:
        source_page = self.embed(source)
        current_page = self.embed(current)
        return source_page.similarity(current_page)


__all__ = ["OmniTransferPageEncoder", "PageEmbedding"]
