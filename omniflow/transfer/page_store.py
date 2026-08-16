"""Incremental page clusters using explicit, auditable decisions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

PAGE_STORE_VERSION = "omniflow.page-store.v1"


@dataclass(frozen=True)
class EmbeddingConfig:
    name: str = "omniflow_native_512d_page_embedding"
    dimension: int = 512
    source_dimension: int = 512
    pooling: str = "native_page_encoder"
    provenance: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", dict(self.provenance or {}))
        if not self.name.strip():
            raise ValueError("page_store_embedding_name_required")
        if self.dimension <= 0 or self.source_dimension <= 0:
            raise ValueError("page_store_embedding_dimension_invalid")
        if self.source_dimension > self.dimension:
            raise ValueError("page_store_source_dimension_exceeds_storage")
        if not self.pooling.strip():
            raise ValueError("page_store_embedding_pooling_required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dimension": self.dimension,
            "source_dimension": self.source_dimension,
            "pooling": self.pooling,
            "provenance": dict(self.provenance or {}),
        }

    @classmethod
    def from_dict(cls, value: Any) -> EmbeddingConfig:
        if not isinstance(value, dict):
            raise ValueError("page_store_embedding_config_invalid")
        return cls(
            name=str(value.get("name") or ""),
            dimension=int(value.get("dimension") or 0),
            source_dimension=int(value.get("source_dimension") or 0),
            pooling=str(value.get("pooling") or ""),
            provenance=dict(value.get("provenance") or {}),
        )


@dataclass(frozen=True)
class PageCandidate:
    cluster_id: str
    cluster_name: str
    score: float
    page_count: int
    representative_page_id: str
    package_names: tuple[str, ...]


@dataclass(frozen=True)
class PageProposal:
    candidates: tuple[PageCandidate, ...]


@dataclass(frozen=True)
class PageContribution:
    page_id: str
    cluster_id: str
    cluster_name: str
    decision: str
    contribution_weight: float
    matched_score: float | None


@dataclass(frozen=True)
class PageCluster:
    cluster_id: str
    name: str
    representative_page_id: str
    page_ids: tuple[str, ...]
    centroid: np.ndarray
    package_names: tuple[str, ...]
    word_presence: np.ndarray | None = None

    @property
    def page_count(self) -> int:
        return len(self.page_ids)


class PageStore:
    """Persist page evidence and apply explicit cluster decisions."""

    def __init__(
        self,
        root: str | Path,
        *,
        embedding_config: EmbeddingConfig | None = None,
    ):
        self.root = Path(root).expanduser().resolve()
        self.index_path = self.root / "current.json"
        self.events_path = self.root / "events.jsonl"
        self.objects = self.root / "objects"
        self.clusters: dict[str, PageCluster] = {}
        self.events: list[dict[str, Any]] = []
        self.embedding_config = embedding_config or EmbeddingConfig()
        self._load()

    def propose(
        self,
        vector: np.ndarray,
        *,
        word_presence: np.ndarray | None = None,
        package_name: str = "",
        limit: int = 5,
    ) -> PageProposal:
        del package_name
        page_vector = _vector(vector, dimension=self.embedding_config.dimension)
        page_presence = _word_presence(word_presence)
        ranked = sorted(
            (
                PageCandidate(
                    cluster_id=cluster.cluster_id,
                    cluster_name=cluster.name,
                    score=_page_similarity(
                        page_vector,
                        cluster.centroid,
                        page_presence,
                        cluster.word_presence,
                    ),
                    page_count=cluster.page_count,
                    representative_page_id=cluster.representative_page_id,
                    package_names=cluster.package_names,
                )
                for cluster in self.clusters.values()
            ),
            key=lambda candidate: (-candidate.score, candidate.cluster_id),
        )
        return PageProposal(tuple(ranked[: max(0, int(limit))]))

    def add_page(
        self,
        *,
        xml: str,
        vector: np.ndarray,
        word_presence: np.ndarray | None = None,
        package_name: str = "",
        activity_name: str = "",
        screenshot_path: str | Path | None = None,
        device_serial: str = "",
        capture_metadata: dict[str, Any] | None = None,
        decision: str,
        cluster_id: str | None = None,
        cluster_name: str = "",
        proposal: PageProposal | None = None,
        decision_source: str = "human",
    ) -> PageContribution:
        if decision not in {"merge", "new"}:
            raise ValueError("page_store_decision_must_be_merge_or_new")
        page_vector = _vector(vector, dimension=self.embedding_config.dimension)
        page_presence = _word_presence(word_presence)
        if decision == "merge":
            if not cluster_id or cluster_id not in self.clusters:
                raise ValueError("page_store_merge_cluster_missing")
        elif cluster_id is not None:
            raise ValueError("page_store_new_cluster_id_must_be_omitted")
        if decision == "new":
            cluster_name = _cluster_name(cluster_name)
        decision_source = " ".join(str(decision_source or "").split())
        if not decision_source:
            raise ValueError("page_store_decision_source_required")

        xml_digest = hashlib.sha256(xml.encode("utf-8")).hexdigest()
        page_id = f"page-{xml_digest[:20]}"
        xml_path = self.objects / xml_digest[:2] / f"{xml_digest}.xml"
        _write_once(xml_path, xml.encode("utf-8"))
        screenshot = _store_screenshot(self.objects, screenshot_path)

        candidate_rows = [
            {
                "cluster_id": candidate.cluster_id,
                "cluster_name": candidate.cluster_name,
                "score": candidate.score,
                "page_count": candidate.page_count,
            }
            for candidate in (proposal.candidates if proposal else ())
        ]
        if decision == "new":
            cluster_id = f"cluster-{xml_digest[:16]}"
            suffix = 1
            base = cluster_id
            while cluster_id in self.clusters:
                suffix += 1
                cluster_id = f"{base}-{suffix}"
            cluster = PageCluster(
                cluster_id=cluster_id,
                name=cluster_name,
                representative_page_id=page_id,
                page_ids=(page_id,),
                centroid=page_vector,
                package_names=_names(package_name),
                word_presence=page_presence,
            )
            contribution_weight = 1.0
            matched_score = None
        else:
            cluster = self.clusters[cluster_id]
            matched_score = _page_similarity(
                page_vector,
                cluster.centroid,
                page_presence,
                cluster.word_presence,
            )
            page_ids = (*cluster.page_ids, page_id)
            centroid = _normalize(
                cluster.centroid * cluster.page_count + page_vector
            )
            cluster = PageCluster(
                cluster_id=cluster.cluster_id,
                name=cluster.name,
                representative_page_id=cluster.representative_page_id,
                page_ids=page_ids,
                centroid=centroid,
                package_names=tuple(
                    sorted(set(cluster.package_names).union(_names(package_name)))
                ),
                word_presence=_merge_presence(
                    cluster.word_presence,
                    page_presence,
                    previous_count=cluster.page_count,
                ),
            )
            contribution_weight = 1.0 / cluster.page_count
        self.clusters[cluster.cluster_id] = cluster

        event = {
            "schema_version": PAGE_STORE_VERSION,
            "event_index": len(self.events),
            "captured_at_ms": int(time.time() * 1000),
            "page_id": page_id,
            "cluster_id": cluster.cluster_id,
            "cluster_name": cluster.name,
            "decision": decision,
            "decision_source": decision_source,
            "contribution_weight": contribution_weight,
            "matched_score": matched_score,
            "candidates": candidate_rows,
            "evidence": {
                "xml_path": str(xml_path),
                "xml_sha256": xml_digest,
                "screenshot_path": screenshot,
                "package_name": str(package_name or ""),
                "activity_name": str(activity_name or ""),
                "device_serial": str(device_serial or ""),
                "capture_metadata": dict(capture_metadata or {}),
            },
        }
        self.events.append(event)
        self._save()
        return PageContribution(
            page_id=page_id,
            cluster_id=cluster.cluster_id,
            cluster_name=cluster.name,
            decision=decision,
            contribution_weight=contribution_weight,
            matched_score=matched_score,
        )

    def rename_cluster(
        self,
        cluster_id: str,
        name: str,
        *,
        decision_source: str,
    ) -> None:
        if cluster_id not in self.clusters:
            raise ValueError("page_store_rename_cluster_missing")
        name = _cluster_name(name)
        decision_source = " ".join(str(decision_source or "").split())
        if not decision_source:
            raise ValueError("page_store_decision_source_required")
        cluster = self.clusters[cluster_id]
        if cluster.name == name:
            return
        self.clusters[cluster_id] = PageCluster(
            cluster_id=cluster.cluster_id,
            name=name,
            representative_page_id=cluster.representative_page_id,
            page_ids=cluster.page_ids,
            centroid=cluster.centroid,
            package_names=cluster.package_names,
            word_presence=cluster.word_presence,
        )
        self.events.append(
            {
                "schema_version": PAGE_STORE_VERSION,
                "event_index": len(self.events),
                "captured_at_ms": int(time.time() * 1000),
                "event_type": "cluster_rename",
                "cluster_id": cluster_id,
                "previous_cluster_name": cluster.name,
                "cluster_name": name,
                "decision_source": decision_source,
            }
        )
        self._save()

    def _load(self) -> None:
        if not self.index_path.is_file():
            return
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != PAGE_STORE_VERSION:
            raise ValueError("unsupported_page_store_version")
        stored_config = EmbeddingConfig.from_dict(payload.get("embedding_config"))
        if stored_config != self.embedding_config:
            raise ValueError("page_store_embedding_config_mismatch")
        loaded: dict[str, PageCluster] = {}
        for cluster_id, value in (payload.get("clusters") or {}).items():
            centroid = _vector(
                np.asarray(value.get("centroid"), dtype=np.float32),
                dimension=self.embedding_config.dimension,
            )
            loaded[cluster_id] = PageCluster(
                cluster_id=cluster_id,
                name=str(value.get("name") or cluster_id),
                representative_page_id=str(value["representative_page_id"]),
                page_ids=tuple(str(item) for item in value.get("page_ids") or ()),
                centroid=centroid,
                package_names=tuple(str(item) for item in value.get("package_names") or ()),
                word_presence=_word_presence(value.get("word_presence")),
            )
        self.clusters = loaded
        if self.events_path.is_file():
            self.events = [
                json.loads(line)
                for line in self.events_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

    def _save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": PAGE_STORE_VERSION,
            "embedding_config": self.embedding_config.to_dict(),
            "cluster_update": "explicit_decision_normalized_centroid",
            "clusters": {
                cluster_id: {
                    "name": cluster.name,
                    "representative_page_id": cluster.representative_page_id,
                    "page_ids": list(cluster.page_ids),
                    "page_count": cluster.page_count,
                    "centroid": cluster.centroid.tolist(),
                    "package_names": list(cluster.package_names),
                    "word_presence": (
                        cluster.word_presence.tolist()
                        if cluster.word_presence is not None
                        else None
                    ),
                }
                for cluster_id, cluster in sorted(self.clusters.items())
            },
        }
        _atomic_json(self.index_path, payload)
        temporary = self.events_path.with_suffix(".jsonl.tmp")
        temporary.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in self.events),
            encoding="utf-8",
        )
        temporary.replace(self.events_path)


def _vector(value: np.ndarray, *, dimension: int) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.shape != (dimension,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"page_store_requires_finite_{dimension}d_vector")
    normalized = _normalize(vector)
    if not np.any(normalized):
        raise ValueError("page_store_requires_nonzero_vector")
    return normalized


def _cluster_name(value: str) -> str:
    name = " ".join(str(value or "").split())
    if not name:
        raise ValueError("page_store_new_cluster_name_required")
    if len(name) > 120:
        raise ValueError("page_store_cluster_name_too_long")
    return name


def _normalize(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    return (value / norm).astype(np.float32) if norm > 1.0e-9 else value


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.clip(np.dot(left, right), -1.0, 1.0))


def _page_similarity(
    left: np.ndarray,
    right: np.ndarray,
    left_presence: np.ndarray | None,
    right_presence: np.ndarray | None,
) -> float:
    if left_presence is None or right_presence is None:
        return _cosine(left, right)
    if left.size % 8:
        raise ValueError("presence_aware_page_embedding_must_have_eight_words")
    left_words = left.reshape(8, -1)
    right_words = right.reshape(8, -1)
    weights = left_presence * right_presence
    denominator = float(np.sum(weights))
    if denominator <= 1.0e-9:
        return _cosine(left, right)
    similarities = np.asarray(
        [_cosine(_normalize(a), _normalize(b)) for a, b in zip(left_words, right_words)],
        dtype=np.float32,
    )
    return float(np.sum(similarities * weights) / denominator)


def _word_presence(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    presence = np.asarray(value, dtype=np.float32).reshape(-1)
    if presence.shape != (8,) or not np.all(np.isfinite(presence)):
        raise ValueError("page_store_word_presence_must_be_finite_8d")
    return np.clip(presence, 0.0, 1.0)


def _merge_presence(
    previous: np.ndarray | None,
    current: np.ndarray | None,
    *,
    previous_count: int,
) -> np.ndarray | None:
    if previous is None or current is None:
        return None
    return (
        (previous * previous_count + current) / (previous_count + 1)
    ).astype(np.float32)


def _names(value: str) -> tuple[str, ...]:
    normalized = str(value or "").strip()
    return (normalized,) if normalized else ()


def _write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"page_store_object_conflict:{path}")
        return
    path.write_bytes(content)


def _store_screenshot(objects: Path, screenshot_path: str | Path | None) -> str | None:
    if screenshot_path is None:
        return None
    source = Path(screenshot_path).expanduser().resolve()
    if not source.is_file():
        return None
    content = source.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    suffix = source.suffix.lower() or ".bin"
    target = objects / digest[:2] / f"{digest}{suffix}"
    _write_once(target, content)
    return str(target)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "PAGE_STORE_VERSION",
    "EmbeddingConfig",
    "PageCandidate",
    "PageCluster",
    "PageContribution",
    "PageProposal",
    "PageStore",
]
