from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

from omniflow.core.model import Function, Observation
RECALL_AUDIT_VERSION = "omniflow.function-recall.v1"
GOAL_LEXICAL_WEIGHT = 1.0


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
    page_encoder: object | None = None,
) -> RecallResult:
    """Recall Planner tools from task semantics only.

    Page embeddings are intentionally not part of Function selection.  They are
    too easy to trigger on an unrelated but visually similar page; OmniTransfer
    remains the sole mechanism for mapping an action after a Function has been
    selected.
    """

    del observation, source_states, page_encoder
    values = functions.values() if isinstance(functions, dict) else functions
    candidates: list[tuple[float, Function, dict[str, Any]]] = []
    decisions: list[dict[str, Any]] = []

    for function in values:
        decision = _score_function(
            str(goal),
            function,
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
            "selection_policy": "goal_lexical_only",
            "ranking_weights": {
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
) -> dict[str, Any]:
    goal_score = _jaccard(
        _tokens(goal),
        _tokens(f"{function.name} {function.description}"),
    )

    return {
        "function_id": function.id,
        "goal_lexical_score": goal_score,
        "score": goal_score,
        "selected": False,
        "rejection_reason": None,
    }


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
    "RECALL_AUDIT_VERSION",
    "RecallResult",
    "recall_functions",
]
