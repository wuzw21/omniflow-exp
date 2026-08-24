"""Canonical learned Page Embedding adapter for OmniFlow.

OmniFlow does not own a second page encoder.  It loads the pinned
OmniTransfer v10 state-attention checkpoint and exposes only its normalized
page readout to Function retrieval and offline regression.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from omniflow.core.model import Observation

V10_PAGE_EMBEDDING_ARCHITECTURE = (
    "omnitransfer_point_conditioned_sparse_graph_v10"
)
V10_PAGE_EMBEDDING_DIMENSION = 1024
V10_PAGE_EMBEDDING_RELATIVE_PATH = Path(
    "output/point_sparse_graph_original_multimodal_v1/full_seed17/model.pt"
)
V10_PAGE_EMBEDDING_SHA256 = (
    "3b783ed113fc37397e2f092d133e970ec36bdbf0d26e262d9273389e4729d16f"
)


@dataclass(frozen=True)
class ElementEmbedding:
    """Legacy RunLog-alignment node shape; not a Page Encoder input."""

    id: str
    parent_id: str | None
    children_ids: tuple[str, ...]
    depth: int
    bounds: tuple[int, int, int, int]
    attributes: dict[str, Any]
    vector: np.ndarray


@dataclass(frozen=True)
class TreeEmbedding:
    """Learned page vector and immutable OmniTransfer provenance."""

    elements: tuple[ElementEmbedding, ...]
    vector: np.ndarray
    root_bounds: tuple[int, int, int, int]
    encoder_version: str
    weights_hash: str
    node_count: int = 0
    architecture: str = ""
    backend: str = ""
    checkpoint_path: str = ""
    checkpoint_sha256: str = ""


class PageEncoder:
    """Thin adapter over the canonical learned OmniTransfer v10 readout."""

    name = "omnitransfer_state_attention"
    element_dimension = 0
    dimension = V10_PAGE_EMBEDDING_DIMENSION
    version = "omnitransfer-v10:learned-state-attention"

    def __init__(self, checkpoint: str | Path | None = None):
        selected = _v10_page_embedding_checkpoint(checkpoint)
        self._embedder = _load_v10_page_embedder(str(selected))
        self.architecture = str(self._embedder.architecture)
        self.backend = str(self._embedder.backend)
        self.checkpoint_path = str(self._embedder.checkpoint_path)
        self.checkpoint_sha256 = str(self._embedder.checkpoint_sha256)
        self.weights_hash = None
        if self.architecture != V10_PAGE_EMBEDDING_ARCHITECTURE:
            raise ValueError(
                "omnitransfer_page_embedding_architecture_mismatch:"
                f"{self.architecture}"
            )
        if self.checkpoint_sha256 != V10_PAGE_EMBEDDING_SHA256:
            raise ValueError(
                "omnitransfer_page_embedding_checkpoint_sha256_mismatch:"
                f"{self.checkpoint_sha256}"
            )
        if int(self._embedder.embedding_dim) != self.dimension:
            raise ValueError(
                "omnitransfer_page_embedding_dimension_mismatch:"
                f"{self._embedder.embedding_dim}"
            )

    def embed(self, value: Observation | dict[str, Any] | str) -> TreeEmbedding:
        observation = (
            Observation(xml=value)
            if isinstance(value, str)
            else Observation.from_value(value)
        )
        xml = str(observation.xml or "")
        if not xml.strip():
            raise ValueError("omnitransfer_page_xml_required")
        graph_id = str(observation.extra.get("state_id") or "").strip()
        if not graph_id:
            graph_id = hashlib.sha256(xml.encode("utf-8")).hexdigest()[:20]
        learned = self._embedder.embed(
            xml,
            graph_id=graph_id,
            pixels=_omnitransfer_pixels(observation),
        )
        return TreeEmbedding(
            elements=(),
            vector=np.asarray(learned.vector, dtype=np.float32),
            root_bounds=_observation_root_bounds(observation),
            encoder_version=self.version,
            weights_hash="",
            node_count=int(learned.node_count),
            architecture=str(learned.architecture),
            backend=str(learned.backend),
            checkpoint_path=str(learned.checkpoint_path),
            checkpoint_sha256=str(learned.checkpoint_sha256),
        )


def _canonical_omnitransfer_root() -> Path:
    configured = str(os.environ.get("OMNITRANSFER_ROOT") or "").strip()
    root = (
        Path(configured).expanduser()
        if configured
        else Path.home() / "Projects" / "Omni" / "OmniTransfer"
    ).resolve()
    canonical = (Path.home() / "Projects" / "Omni" / "OmniTransfer").resolve()
    if root != canonical:
        raise ValueError(f"canonical_omnitransfer_root_required:{canonical}")
    if not (root / "src" / "omnitransfer").is_dir():
        raise RuntimeError(f"omnitransfer_root_missing:{root}")
    return root


def _v10_page_embedding_checkpoint(value: str | Path | None) -> Path:
    root = _canonical_omnitransfer_root()
    selected = (
        Path(value).expanduser().resolve()
        if value is not None
        else (root / V10_PAGE_EMBEDDING_RELATIVE_PATH).resolve()
    )
    if selected != root and root not in selected.parents:
        raise ValueError(f"omnitransfer_checkpoint_must_be_under:{root}")
    if not selected.is_file():
        raise FileNotFoundError(
            f"omnitransfer_v10_page_embedding_checkpoint_missing:{selected}"
        )
    return selected


@lru_cache(maxsize=2)
def _load_v10_page_embedder(checkpoint: str) -> Any:
    module = _canonical_omnitransfer_module("omnitransfer.page_embedding")
    return module.OmniTransferPageEmbedder(checkpoint=checkpoint, device="cpu")


def _canonical_omnitransfer_module(name: str) -> Any:
    root = _canonical_omnitransfer_root()
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
        if module_name == "omnitransfer" or module_name.startswith("omnitransfer."):
            module_path = Path(
                str(getattr(sys.modules[module_name], "__file__", ""))
            ).resolve()
            if package_root not in module_path.parents:
                del sys.modules[module_name]
    return importlib.import_module(name)


def _omnitransfer_pixels(observation: Observation) -> dict[str, Any]:
    pixels: dict[str, Any] = {}
    display = observation.extra.get("display")
    if isinstance(display, dict):
        pixels["width"] = display.get("width")
        pixels["height"] = display.get("height")
    androidworld_state = observation.extra.get("androidworld_state")
    if isinstance(androidworld_state, dict):
        native_pixels = androidworld_state.get("pixels")
        if isinstance(native_pixels, dict):
            pixels["width"] = native_pixels.get("width") or pixels.get("width")
            pixels["height"] = native_pixels.get("height") or pixels.get("height")
            if isinstance(native_pixels.get("path"), str):
                pixels["path"] = native_pixels["path"]
    screenshot_path = observation.extra.get("screenshot_path")
    if isinstance(screenshot_path, str) and screenshot_path.strip():
        pixels["path"] = screenshot_path
    return pixels


def _observation_root_bounds(
    observation: Observation,
) -> tuple[int, int, int, int]:
    pixels = _omnitransfer_pixels(observation)
    try:
        width = round(float(pixels.get("width") or 0))
        height = round(float(pixels.get("height") or 0))
    except (TypeError, ValueError):
        width, height = 0, 0
    if width > 0 and height > 0:
        return 0, 0, width, height
    return 0, 0, 1000, 1000


__all__ = [
    "ElementEmbedding",
    "PageEncoder",
    "TreeEmbedding",
    "V10_PAGE_EMBEDDING_ARCHITECTURE",
    "V10_PAGE_EMBEDDING_DIMENSION",
    "V10_PAGE_EMBEDDING_RELATIVE_PATH",
    "V10_PAGE_EMBEDDING_SHA256",
]
