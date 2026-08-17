from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

from omniflow.core.model import Function, Observation
from omniflow.transfer.page_embedding import OmniTransferPageEncoder, PageEmbedding

RECALL_AUDIT_VERSION = "omniflow.function-recall.v1"
PAGE_SIMILARITY_WEIGHT = 0.30
GOAL_LEXICAL_WEIGHT = 0.70


@dataclass(frozen=True)
class RecallResult:
    functions: tuple[Function, ...]
    audit: dict[str, Any]


def recall_functions(
    goal: str,
    *,
    observation: Observation,
    functions: dict[str, Function] | list[Function] | tuple[Function, ...],
    source_states: Mapping[str, Observation | None],
    limit: int = 8,
    page_encoder: OmniTransferPageEncoder | None = None,
) -> RecallResult:
    """Recall Planner tools using page and lexical evidence without page gating."""

    encoder = page_encoder or OmniTransferPageEncoder()
    current_page = _embed_if_available(encoder, observation)
    values = functions.values() if isinstance(functions, dict) else functions
    candidates: list[tuple[float, Function, dict[str, Any]]] = []
    decisions: list[dict[str, Any]] = []

    for function in values:
        decision = _score_function(
            str(goal),
            function,
            current_page=current_page,
            source_states=source_states,
            encoder=encoder,
        )
        decisions.append(decision)
        candidates.append((float(decision["score"]), function, decision))

    ranked = sorted(candidates, key=lambda item: (-item[0], item[1].id))
    selected = ranked[: max(0, int(limit))]
    selected_ids = {function.id for _score, function, _audit in selected}
    for decision in decisions:
        decision["selected"] = decision["function_id"] in selected_ids
        if not decision["selected"]:
            decision["rejection_reason"] = "candidate_limit"

    return RecallResult(
        tuple(function for _score, function, _audit in selected),
        {
            "schema_version": RECALL_AUDIT_VERSION,
            "encoder": {
                "name": encoder.name,
                "version": encoder.encoder_version,
                "dimension": encoder.dimension,
                "checkpoint_path": str(encoder.checkpoint_path),
                "checkpoint_sha256": encoder.checkpoint_sha256,
            },
            "current_page": {
                "available": current_page is not None,
                "element_count": (
                    current_page.element_count if current_page is not None else 0
                ),
            },
            "ranking_weights": {
                "page_similarity": PAGE_SIMILARITY_WEIGHT,
                "goal_lexical": GOAL_LEXICAL_WEIGHT,
            },
            "candidate_function_ids": [
                function.id for _score, function, _audit in selected
            ],
            "decisions": decisions,
        },
    )


def _score_function(
    goal: str,
    function: Function,
    *,
    current_page: PageEmbedding | None,
    source_states: Mapping[str, Observation | None],
    encoder: OmniTransferPageEncoder,
) -> dict[str, Any]:
    source_state_id = function.steps[0].source_state_id if function.steps else ""
    source_observation = source_states.get(source_state_id)
    source_page = _embed_if_available(encoder, source_observation)
    page_similarity = (
        current_page.similarity(source_page)
        if current_page is not None and source_page is not None
        else 0.0
    )
    goal_score = _jaccard(
        _tokens(goal),
        _tokens(f"{function.name} {function.description}"),
    )
    score = (
        PAGE_SIMILARITY_WEIGHT * page_similarity
        + GOAL_LEXICAL_WEIGHT * goal_score
    )

    return {
        "function_id": function.id,
        "source_state_id": source_state_id,
        "page_similarity": page_similarity,
        "goal_lexical_score": goal_score,
        "score": score,
        "selected": False,
        "rejection_reason": None,
    }


def _embed_if_available(
    encoder: OmniTransferPageEncoder,
    observation: Observation | None,
) -> PageEmbedding | None:
    if observation is None or not str(observation.xml or "").strip():
        return None
    return encoder.embed(observation)


def _tokens(value: str) -> set[str]:
    normalized = str(value or "").lower()
    tokens = set(re.findall(r"[a-z0-9_]+", normalized))
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]+", normalized))
    if chinese:
        tokens.add(chinese)
        tokens.update(
            chinese[index : index + 2] for index in range(max(0, len(chinese) - 1))
        )
    return tokens


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


__all__ = [
    "GOAL_LEXICAL_WEIGHT",
    "PAGE_SIMILARITY_WEIGHT",
    "RECALL_AUDIT_VERSION",
    "RecallResult",
    "recall_functions",
]
