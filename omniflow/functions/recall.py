from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

import numpy as np

from omniflow.core.model import Function, Observation
from omniflow.transfer.embedding import PageEncoder, TreeEmbedding

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
    page_encoder: PageEncoder | None = None,
) -> RecallResult:
    """Recall Planner tools using page and lexical evidence without page gating."""

    encoder = page_encoder or PageEncoder()
    current_page = _embed_page(encoder, observation)
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
                "version": encoder.version,
                "dimension": encoder.dimension,
                "weights_hash": encoder.weights.hash,
            },
            "current_page": {
                "element_count": (
                    len(current_page.elements) if current_page is not None else 0
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
    current_page: TreeEmbedding | None,
    source_states: Mapping[str, Observation | None],
    encoder: PageEncoder,
) -> dict[str, Any]:
    source_state_id = function.steps[0].source_state_id if function.steps else ""
    source_observation = source_states.get(source_state_id)
    source_page = _embed_page(encoder, source_observation)
    page_similarity = (
        _cosine(current_page.vector, source_page.vector)
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


def _embed_page(
    encoder: PageEncoder,
    observation: Observation | None,
) -> TreeEmbedding | None:
    if observation is None:
        return None
    try:
        return encoder.embed(observation)
    except ValueError as error:
        if str(error) == "omnitransfer_page_xml_required":
            return None
        raise


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(left, right) / denominator)))


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
