from __future__ import annotations

import re

from omniflow.core.model import Function


def recall_functions(
    goal: str,
    *,
    functions: dict[str, Function] | list[Function] | tuple[Function, ...],
    limit: int = 8,
) -> list[Function]:
    """Return the visible Functions whose semantics best match the current goal."""

    return [
        function
        for _score, function in _rank_functions(
            goal,
            functions=functions,
            limit=limit,
        )
    ]


def _rank_functions(
    goal: str,
    *,
    functions: dict[str, Function] | list[Function] | tuple[Function, ...],
    limit: int = 8,
) -> list[tuple[float, Function]]:
    """Rank Function tools for injection; this function never selects execution."""

    values = functions.values() if isinstance(functions, dict) else functions
    goal_tokens = _tokens(goal)
    scored: list[tuple[float, Function]] = []
    for function in values:
        if not function.agent_visible:
            continue
        score = max(
            _jaccard(goal_tokens, _tokens(function.name)),
            _jaccard(goal_tokens, _tokens(function.description)),
        )
        if score > 0:
            scored.append((score, function))
    ranked = sorted(scored, key=lambda item: (-item[0], item[1].id))
    if not ranked:
        return []
    relevance_floor = ranked[0][0] * 0.5
    return [item for item in ranked if item[0] >= relevance_floor][: max(0, int(limit))]


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


__all__ = ["recall_functions"]
