"""Legacy PageEncoder API backed by canonical OmniTransfer embedding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from omniflow.core.model import Observation
from omniflow.transfer.page_embedding import OmniTransferPageEncoder


@dataclass(frozen=True)
class EncoderWeights:
    hash: str

    @classmethod
    def manual_default(cls) -> "EncoderWeights":
        return cls(OmniTransferPageEncoder().checkpoint_sha256)


@dataclass(frozen=True)
class ElementEmbedding:
    id: str
    parent_id: str | None
    children_ids: tuple[str, ...]
    depth: int
    bounds: tuple[int, int, int, int]
    attributes: dict[str, Any]
    vector: np.ndarray


@dataclass(frozen=True)
class TreeEmbedding:
    elements: tuple[ElementEmbedding, ...]
    vector: np.ndarray
    root_bounds: tuple[int, int, int, int]
    encoder_version: str
    weights_hash: str


class PageEncoder:
    """Preserve the 8/13 interface; do not implement a second encoder."""

    name = "omnitransfer_page_embedding"

    def __init__(self, weights: EncoderWeights | None = None):
        del weights
        self._encoder = OmniTransferPageEncoder()
        self.version = self._encoder.encoder_version
        self.dimension = self._encoder.dimension
        self.weights = EncoderWeights(self._encoder.checkpoint_sha256)

    def embed(self, value: Observation | dict[str, Any] | str) -> TreeEmbedding:
        page = self._encoder.embed(value)
        return TreeEmbedding(
            elements=(),
            vector=np.asarray(page.vector, dtype=np.float32),
            root_bounds=(0, 0, 0, 0),
            encoder_version=page.encoder_version,
            weights_hash=page.checkpoint_sha256,
        )


__all__ = ["ElementEmbedding", "EncoderWeights", "PageEncoder", "TreeEmbedding"]
