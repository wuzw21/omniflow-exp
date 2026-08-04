"""Sidecar transfer scoring and action-aware monotonic replay alignment.

This module is experiment-only.  It does not participate in OmniFlow's runtime
replay path.  Identity attributes are intentionally absent from the alignment
contract: OmniTransfer supplies continuous pair evidence, while action kind is
used only to price gaps in the sequence path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

from omniflow.transfer.runtime import transfer_action

TRANSFER_SEQUENCE_PROTOCOL = "transfer_sequence_dp_v1"
_LOW_COST_ACTIONS = frozenset({"open_app", "press_key", "wait"})
_MEDIUM_COST_ACTIONS = frozenset({"scroll", "swipe"})


@dataclass(frozen=True)
class TransferMatchScore:
    probability: float | None = None
    top_probability: float | None = None
    mapped_point: tuple[float, float] | None = None
    source_point: tuple[float, float] | None = None
    source_bounds: tuple[float, float, float, float] | None = None
    mapped_bounds: tuple[float, float, float, float] | None = None
    target_bounds: tuple[float, float, float, float] | None = None
    candidates: tuple[
        tuple[tuple[float, float, float, float], float], ...
    ] = ()
    candidate_rank: int | None = None
    exact_hit: bool = False
    mapped: bool = False
    reason: str = ""
    mapping_mode: str = ""


@dataclass(frozen=True)
class ReplayToken:
    index: int
    action_kind: str = ""


@dataclass(frozen=True)
class ReplayPair:
    source_index: int
    target_index: int
    probability: float
    contribution: float


@dataclass(frozen=True)
class ReplayGap:
    side: Literal["source", "target"]
    index: int
    action_kind: str
    cost: float


@dataclass(frozen=True)
class ReplayAlignmentConfig:
    minimum_probability: float = 0.5
    passive_gap_cost: float = 0.05
    unknown_target_gap_cost: float = 0.08
    gesture_gap_cost: float = 0.35
    interaction_gap_cost: float = 1.1


@dataclass(frozen=True)
class ReplayAlignment:
    protocol: str
    mode: Literal["global", "target_prefix"]
    pairs: tuple[ReplayPair, ...]
    source_gaps: tuple[ReplayGap, ...]
    target_gaps: tuple[ReplayGap, ...]
    score: float
    source_endpoint: int
    target_endpoint: int
    minimum_probability: float


def score_transfer_match(
    *,
    source_xml: str,
    target_xml: str,
    source_point: tuple[float, float],
    target_bounds: tuple[float, float, float, float] | None = None,
    top_k: int = 64,
) -> TransferMatchScore:
    """Return OmniTransfer evidence without package/action/selector gates."""

    try:
        result = transfer_action(
            source_xml=source_xml,
            target_xml=target_xml,
            source_point=source_point,
            top_k=max(1, int(top_k)),
        )
    except Exception as error:  # noqa: BLE001 - sidecar records an unavailable score
        return TransferMatchScore(reason=f"omnitransfer_error:{error}")
    if not isinstance(result, dict):
        return TransferMatchScore(reason="omnitransfer_result_invalid")

    mapped_point = _point(result.get("new_x"), result.get("new_y"))
    top_probability = _probability(result.get("score"))
    if top_probability is None:
        top_probability = _probability(result.get("pair_confidence"))
    raw_candidates = tuple(
        candidate
        for candidate in (result.get("top_candidates") or ())
        if isinstance(candidate, dict)
    )
    candidates = tuple(
        (bounds, probability)
        for candidate in raw_candidates
        if (bounds := _bounds(candidate.get("bbox"))) is not None
        and (probability := _probability(candidate.get("score"))) is not None
    )
    if top_probability is None and candidates:
        top_probability = candidates[0][1]

    source = result.get("src_element")
    source_bounds = _bounds(source.get("bounds")) if isinstance(source, dict) else None
    base = TransferMatchScore(
        probability=top_probability,
        top_probability=top_probability,
        mapped_point=mapped_point,
        source_point=source_point,
        source_bounds=source_bounds,
        mapped_bounds=_bounds(result.get("target_bbox")),
        target_bounds=_bounds(result.get("target_bbox")),
        candidates=candidates,
        candidate_rank=1 if top_probability is not None else None,
        mapped=result.get("mapped") is True,
        reason=str(result.get("reason") or ""),
        mapping_mode=str(result.get("mapping_mode") or ""),
    )
    if target_bounds is None:
        return base
    return retarget_transfer_score(
        base,
        source_point=source_point,
        target_bounds=target_bounds,
    )


def retarget_transfer_score(
    evidence: TransferMatchScore,
    *,
    source_point: tuple[float, float],
    target_bounds: tuple[float, float, float, float],
) -> TransferMatchScore:
    """Reuse one node/page matcher result for a target action endpoint."""

    selected = _target_candidate(evidence.candidates, target_bounds)
    if selected is None:
        candidate_rank = None
        probability = None
        selected_bounds = None
    else:
        candidate_rank, probability, selected_bounds = selected
    mapped_point = evidence.mapped_point
    if (
        source_point != evidence.source_point
        and evidence.source_bounds is not None
        and evidence.mapped_bounds is not None
    ):
        mapped_point = _project_offset(
            source_point,
            evidence.source_bounds,
            evidence.mapped_bounds,
        )
    return TransferMatchScore(
        probability=probability,
        top_probability=evidence.top_probability,
        mapped_point=mapped_point,
        source_point=source_point,
        source_bounds=evidence.source_bounds,
        mapped_bounds=evidence.mapped_bounds,
        target_bounds=selected_bounds,
        candidates=evidence.candidates,
        candidate_rank=candidate_rank,
        exact_hit=(
            evidence.mapped
            and mapped_point is not None
            and _contains(target_bounds, mapped_point)
        ),
        mapped=evidence.mapped,
        reason=evidence.reason,
        mapping_mode=evidence.mapping_mode,
    )


def align_transfer_replay(
    source: Sequence[ReplayToken],
    target: Sequence[ReplayToken],
    probabilities: Sequence[Sequence[float | None]],
    *,
    mode: Literal["global", "target_prefix"] = "global",
    config: ReplayAlignmentConfig | None = None,
) -> ReplayAlignment:
    """Align two monotonic sequences using transfer evidence and typed gaps.

    ``global`` consumes both complete sequences. ``target_prefix`` consumes the
    full target history and chooses the best matched source prefix, leaving the
    unobserved source suffix outside the optimization.
    """

    if mode not in {"global", "target_prefix"}:
        raise ValueError("unsupported_replay_alignment_mode")
    resolved = config or ReplayAlignmentConfig()
    if not 0.0 < resolved.minimum_probability < 1.0:
        raise ValueError("minimum_probability_must_be_between_zero_and_one")
    if len(probabilities) != len(source) or any(
        len(row) != len(target) for row in probabilities
    ):
        raise ValueError("transfer_probability_matrix_shape_mismatch")

    source_size = len(source)
    target_size = len(target)
    negative_infinity = float("-inf")
    scores = [
        [negative_infinity for _ in range(target_size + 1)]
        for _ in range(source_size + 1)
    ]
    back: list[list[str | None]] = [
        [None for _ in range(target_size + 1)]
        for _ in range(source_size + 1)
    ]
    scores[0][0] = 0.0
    for source_position, token in enumerate(source, start=1):
        scores[source_position][0] = (
            scores[source_position - 1][0] - _gap_cost(token, resolved, "source")
        )
        back[source_position][0] = "source_gap"
    for target_position, token in enumerate(target, start=1):
        scores[0][target_position] = (
            scores[0][target_position - 1] - _gap_cost(token, resolved, "target")
        )
        back[0][target_position] = "target_gap"

    priority = {"target_gap": 0, "source_gap": 1, "match": 2}
    for source_position, source_token in enumerate(source, start=1):
        for target_position, target_token in enumerate(target, start=1):
            choices = [
                (
                    scores[source_position - 1][target_position]
                    - _gap_cost(source_token, resolved, "source"),
                    "source_gap",
                ),
                (
                    scores[source_position][target_position - 1]
                    - _gap_cost(target_token, resolved, "target"),
                    "target_gap",
                ),
            ]
            probability = _probability(
                probabilities[source_position - 1][target_position - 1]
            )
            if (
                probability is not None
                and probability >= resolved.minimum_probability
            ):
                choices.append(
                    (
                        scores[source_position - 1][target_position - 1]
                        + _log_odds(probability),
                        "match",
                    )
                )
            score, operation = max(
                choices,
                key=lambda choice: (choice[0], priority[choice[1]]),
            )
            scores[source_position][target_position] = score
            back[source_position][target_position] = operation

    source_endpoint = source_size
    if mode == "target_prefix":
        candidates = [
            position
            for position in range(1, source_size + 1)
            if back[position][target_size] == "match"
            and scores[position][target_size] > 0.0
        ]
        if not candidates:
            return ReplayAlignment(
                protocol=TRANSFER_SEQUENCE_PROTOCOL,
                mode=mode,
                pairs=(),
                source_gaps=(),
                target_gaps=(),
                score=negative_infinity,
                source_endpoint=0,
                target_endpoint=target_size,
                minimum_probability=resolved.minimum_probability,
            )
        source_endpoint = max(
            candidates,
            key=lambda position: (scores[position][target_size], position),
        )

    pairs: list[ReplayPair] = []
    source_gaps: list[ReplayGap] = []
    target_gaps: list[ReplayGap] = []
    source_position = source_endpoint
    target_position = target_size
    while source_position > 0 or target_position > 0:
        operation = back[source_position][target_position]
        if operation == "match":
            probability = _probability(
                probabilities[source_position - 1][target_position - 1]
            )
            if probability is None:
                raise RuntimeError("replay_alignment_match_without_probability")
            pairs.append(
                ReplayPair(
                    source_index=source[source_position - 1].index,
                    target_index=target[target_position - 1].index,
                    probability=probability,
                    contribution=_log_odds(probability),
                )
            )
            source_position -= 1
            target_position -= 1
        elif operation == "source_gap":
            token = source[source_position - 1]
            source_gaps.append(
                ReplayGap(
                    side="source",
                    index=token.index,
                    action_kind=token.action_kind,
                    cost=_gap_cost(token, resolved, "source"),
                )
            )
            source_position -= 1
        elif operation == "target_gap":
            token = target[target_position - 1]
            target_gaps.append(
                ReplayGap(
                    side="target",
                    index=token.index,
                    action_kind=token.action_kind,
                    cost=_gap_cost(token, resolved, "target"),
                )
            )
            target_position -= 1
        else:
            break
    pairs.reverse()
    source_gaps.reverse()
    target_gaps.reverse()
    return ReplayAlignment(
        protocol=TRANSFER_SEQUENCE_PROTOCOL,
        mode=mode,
        pairs=tuple(pairs),
        source_gaps=tuple(source_gaps),
        target_gaps=tuple(target_gaps),
        score=scores[source_endpoint][target_size],
        source_endpoint=source_endpoint,
        target_endpoint=target_size,
        minimum_probability=resolved.minimum_probability,
    )


def _gap_cost(
    token: ReplayToken,
    config: ReplayAlignmentConfig,
    side: Literal["source", "target"],
) -> float:
    action_kind = str(token.action_kind or "").strip().lower()
    if not action_kind and side == "target":
        return config.unknown_target_gap_cost
    if action_kind in _LOW_COST_ACTIONS:
        return config.passive_gap_cost
    if action_kind in _MEDIUM_COST_ACTIONS:
        return config.gesture_gap_cost
    return config.interaction_gap_cost


def _target_candidate(
    candidates: Sequence[
        tuple[tuple[float, float, float, float], float]
    ],
    target_bounds: tuple[float, float, float, float],
) -> tuple[int, float, tuple[float, float, float, float]] | None:
    ranked = []
    for rank, (bounds, probability) in enumerate(candidates, start=1):
        overlap = _intersection_over_union(bounds, target_bounds)
        if overlap <= 0.0:
            continue
        ranked.append((overlap, probability, -rank, rank, bounds))
    if not ranked:
        return None
    _, probability, _, rank, bounds = max(ranked)
    return rank, probability, bounds


def _probability(value: object) -> float | None:
    try:
        probability = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(probability):
        return None
    return min(1.0, max(0.0, probability))


def _log_odds(probability: float) -> float:
    bounded = min(1.0 - 1e-9, max(1e-9, probability))
    return math.log(bounded / (1.0 - bounded))


def _point(x: object, y: object) -> tuple[float, float] | None:
    try:
        point = float(x), float(y)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return point if all(math.isfinite(value) for value in point) else None


def _bounds(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        bounds = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    left, top, right, bottom = bounds
    if not all(math.isfinite(item) for item in bounds) or right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _contains(
    bounds: tuple[float, float, float, float],
    point: tuple[float, float],
) -> bool:
    return bounds[0] <= point[0] <= bounds[2] and bounds[1] <= point[1] <= bounds[3]


def _intersection_over_union(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    if intersection <= 0.0:
        return 0.0
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


def _project_offset(
    point: tuple[float, float],
    source_bounds: tuple[float, float, float, float],
    target_bounds: tuple[float, float, float, float],
) -> tuple[float, float]:
    source_width = source_bounds[2] - source_bounds[0]
    source_height = source_bounds[3] - source_bounds[1]
    x_offset = (point[0] - source_bounds[0]) / source_width
    y_offset = (point[1] - source_bounds[1]) / source_height
    return (
        target_bounds[0] + x_offset * (target_bounds[2] - target_bounds[0]),
        target_bounds[1] + y_offset * (target_bounds[3] - target_bounds[1]),
    )


__all__ = [
    "ReplayAlignment",
    "ReplayAlignmentConfig",
    "ReplayGap",
    "ReplayPair",
    "ReplayToken",
    "TRANSFER_SEQUENCE_PROTOCOL",
    "TransferMatchScore",
    "align_transfer_replay",
    "retarget_transfer_score",
    "score_transfer_match",
]
