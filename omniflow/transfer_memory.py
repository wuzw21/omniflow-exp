"""Persistent bidirectional point-pair evidence for OmniTransfer learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from omniflow.runlog_alignment import RunlogEndpoint, RunlogStepPair, align_runlogs


TRANSFER_PAIR_MEMORY_VERSION = "omniflow.transfer-pair-memory.v1"


@dataclass(frozen=True)
class TransferDirection:
    pair_id: str
    source: dict[str, Any]
    target: dict[str, Any]
    evidence: dict[str, Any]


@dataclass(frozen=True)
class TransferPair:
    pair_id: str
    source: dict[str, Any]
    target: dict[str, Any]
    evidence: dict[str, Any]
    bidirectional: bool = True

    @classmethod
    def from_alignment(cls, pair: RunlogStepPair) -> "TransferPair":
        source = _endpoint_dict(pair.left_endpoint)
        target = _endpoint_dict(pair.right_endpoint)
        return cls(
            pair_id=_pair_id(pair.left_endpoint, pair.right_endpoint),
            source=source,
            target=target,
            evidence={
                "alignment_score": pair.score,
                "page_similarity": pair.page_similarity,
                "node_similarity": pair.node_similarity,
            },
        )

    @classmethod
    def from_dict(cls, value: Any) -> "TransferPair":
        if not isinstance(value, dict) or set(value) != {
            "pair_id",
            "bidirectional",
            "source",
            "target",
            "evidence",
        }:
            raise ValueError("transfer_pair_contract_invalid")
        if value.get("bidirectional") is not True:
            raise ValueError("transfer_pair_must_be_bidirectional")
        pair_id = str(value.get("pair_id") or "").strip()
        if not pair_id:
            raise ValueError("transfer_pair_id_required")
        evidence = value.get("evidence")
        if not isinstance(evidence, dict):
            raise ValueError("transfer_pair_evidence_invalid")
        return cls(
            pair_id=pair_id,
            source=_validate_endpoint(value.get("source"), side="source"),
            target=_validate_endpoint(value.get("target"), side="target"),
            evidence=dict(evidence),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "bidirectional": True,
            "source": dict(self.source),
            "target": dict(self.target),
            "evidence": dict(self.evidence),
        }

    def directions(self) -> tuple[TransferDirection, TransferDirection]:
        evidence = dict(self.evidence)
        return (
            TransferDirection(
                self.pair_id,
                dict(self.source),
                dict(self.target),
                evidence,
            ),
            TransferDirection(
                self.pair_id,
                dict(self.target),
                dict(self.source),
                evidence,
            ),
        )


class TransferPairStore:
    """Store pair evidence; runtime execution remains owned by OmniTransfer."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.pairs: dict[str, TransferPair] = {}
        self._load()

    def ingest_runlogs(
        self,
        source_runlog: dict[str, Any],
        target_runlog: dict[str, Any],
        *,
        min_score: float = 0.72,
        gap_penalty: float = -0.18,
    ) -> tuple[TransferPair, ...]:
        alignment = align_runlogs(
            source_runlog,
            target_runlog,
            min_score=min_score,
            gap_penalty=gap_penalty,
        )
        added: list[TransferPair] = []
        for aligned in alignment.pairs:
            if not _transfer_endpoint_complete(aligned.left_endpoint):
                continue
            if not _transfer_endpoint_complete(aligned.right_endpoint):
                continue
            pair = TransferPair.from_alignment(aligned)
            existing = self.pairs.get(pair.pair_id)
            if existing is not None:
                if existing == pair or _is_reverse(existing, pair):
                    continue
                raise ValueError(f"transfer_pair_conflict:{pair.pair_id}")
            self.pairs[pair.pair_id] = pair
            added.append(pair)
        self.save()
        return tuple(added)

    def directions(
        self,
        source_state_id: str | None = None,
    ) -> tuple[TransferDirection, ...]:
        state_id = str(source_state_id or "").strip()
        values = (
            direction
            for pair in self.pairs.values()
            for direction in pair.directions()
        )
        return tuple(
            direction
            for direction in values
            if not state_id or direction.source["state_id"] == state_id
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": TRANSFER_PAIR_MEMORY_VERSION,
            "pairs": {
                pair_id: pair.to_dict()
                for pair_id, pair in sorted(self.pairs.items())
            },
        }
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def _load(self) -> None:
        if not self.path.exists():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != {"schema_version", "pairs"}:
            raise ValueError("transfer_pair_memory_contract_invalid")
        if payload.get("schema_version") != TRANSFER_PAIR_MEMORY_VERSION:
            raise ValueError("unsupported_transfer_pair_memory_version")
        raw_pairs = payload.get("pairs")
        if not isinstance(raw_pairs, dict):
            raise ValueError("transfer_pair_memory_pairs_invalid")
        loaded: dict[str, TransferPair] = {}
        for key, value in raw_pairs.items():
            pair = TransferPair.from_dict(value)
            if str(key) != pair.pair_id:
                raise ValueError("transfer_pair_memory_key_mismatch")
            loaded[pair.pair_id] = pair
        self.pairs = loaded


def _endpoint_dict(endpoint: RunlogEndpoint) -> dict[str, Any]:
    return _validate_endpoint(asdict(endpoint), side="endpoint")


def _validate_endpoint(value: Any, *, side: str) -> dict[str, Any]:
    fields = {
        "run_id",
        "state_id",
        "step_index",
        "action_tool",
        "page_id",
        "package_name",
        "activity_name",
        "screenshot_path",
        "width",
        "height",
        "point",
        "node",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"transfer_pair_{side}_contract_invalid")
    endpoint = dict(value)
    for field in ("run_id", "state_id", "action_tool", "page_id"):
        if not isinstance(endpoint[field], str) or not endpoint[field].strip():
            raise ValueError(f"transfer_pair_{side}_{field}_required")
    for field in ("package_name", "activity_name", "screenshot_path"):
        if not isinstance(endpoint[field], str):
            raise ValueError(f"transfer_pair_{side}_{field}_invalid")
    if not isinstance(endpoint["step_index"], int) or isinstance(
        endpoint["step_index"], bool
    ):
        raise ValueError(f"transfer_pair_{side}_step_index_invalid")
    for field in ("width", "height"):
        number = endpoint[field]
        if (
            not isinstance(number, (int, float))
            or isinstance(number, bool)
            or not math.isfinite(float(number))
            or float(number) <= 0
        ):
            raise ValueError(f"transfer_pair_{side}_{field}_invalid")
        endpoint[field] = float(number)
    point = endpoint.get("point")
    if not isinstance(point, dict) or set(point) != {"x", "y", "coordinate_space"}:
        raise ValueError(f"transfer_pair_{side}_point_invalid")
    if point.get("coordinate_space") != "page_pixels":
        raise ValueError(f"transfer_pair_{side}_coordinate_space_invalid")
    for field, limit in (("x", endpoint["width"]), ("y", endpoint["height"])):
        number = point.get(field)
        if (
            not isinstance(number, (int, float))
            or isinstance(number, bool)
            or not math.isfinite(float(number))
            or float(number) < 0
            or float(number) > limit
        ):
            raise ValueError(f"transfer_pair_{side}_point_invalid")
    endpoint["point"] = {
        "x": float(point["x"]),
        "y": float(point["y"]),
        "coordinate_space": "page_pixels",
    }
    if not isinstance(endpoint.get("node"), dict):
        raise ValueError(f"transfer_pair_{side}_node_invalid")
    return endpoint


def _transfer_endpoint_complete(endpoint: RunlogEndpoint) -> bool:
    return (
        endpoint.point is not None
        and endpoint.node is not None
        and endpoint.width > 0
        and endpoint.height > 0
    )


def _pair_id(source: RunlogEndpoint, target: RunlogEndpoint) -> str:
    endpoints = sorted(
        (
            "\0".join((source.run_id, str(source.step_index), source.state_id)),
            "\0".join((target.run_id, str(target.step_index), target.state_id)),
        )
    )
    identity = "\0\0".join(endpoints)
    digest = hashlib.blake2b(identity.encode(), digest_size=12).hexdigest()
    return f"transfer-pair-{digest}"


def _is_reverse(existing: TransferPair, candidate: TransferPair) -> bool:
    return (
        existing.source == candidate.target
        and existing.target == candidate.source
        and existing.evidence == candidate.evidence
    )


__all__ = [
    "TRANSFER_PAIR_MEMORY_VERSION",
    "TransferDirection",
    "TransferPair",
    "TransferPairStore",
]
