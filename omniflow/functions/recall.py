from __future__ import annotations

from dataclasses import dataclass
import inspect
import re
from typing import Any, Mapping

import numpy as np

from omniflow.core.model import Function, Observation, Transfer, TransferResult
from omniflow.transfer.admission import assess_transfer, requires_contextual_mapping
from omniflow.transfer.embedding import PageEncoder, TreeEmbedding

RECALL_AUDIT_VERSION = "omniflow.function-recall.v1"
PAGE_SIMILARITY_WEIGHT = 0.30
GOAL_LEXICAL_WEIGHT = 0.70
FUNCTION_PAGE_SIMILARITY_THRESHOLD = 0.80


@dataclass(frozen=True)
class RecallResult:
    functions: tuple[Function, ...]
    audit: dict[str, Any]


async def recall_functions(
    goal: str,
    *,
    observation: Observation,
    functions: dict[str, Function] | list[Function] | tuple[Function, ...],
    source_states: Mapping[str, Observation | None],
    limit: int = 8,
    page_encoder: PageEncoder | None = None,
    transfer: Transfer | None = None,
    exclude_function_ids: frozenset[str] = frozenset(),
) -> RecallResult:
    """Coarsely rank Functions, then expose only first-step transfer matches."""

    encoder = page_encoder or PageEncoder()
    current_page = _embed_page(encoder, observation)
    values = functions.values() if isinstance(functions, dict) else functions
    candidates: list[tuple[float, Function, dict[str, Any]]] = []
    decisions: list[dict[str, Any]] = []

    for function in values:
        decision = _score_function(
            str(goal),
            function,
            current_observation=observation,
            current_page=current_page,
            source_states=source_states,
            encoder=encoder,
        )
        decisions.append(decision)
        if (
            function.agent_visible
            and function.steps
            and function.id not in exclude_function_ids
        ):
            candidates.append((float(decision["score"]), function, decision))

    ranked = sorted(candidates, key=lambda item: (-item[0], item[1].id))
    coarse_limit = max(0, int(limit)) * 3
    coarse = ranked[:coarse_limit]
    admitted: list[tuple[float, Function, dict[str, Any]]] = []
    for _score, function, decision in coarse:
        decision["coarse_selected"] = True
        source_state_id = function.steps[0].source_state_id
        source_observation = source_states.get(source_state_id)
        if not requires_contextual_mapping(function.steps[0].action.tool):
            transfer_result = TransferResult(function.steps[0].action)
        elif transfer is None:
            decision["rejection_reason"] = "function_transfer_unavailable"
            continue
        else:
            try:
                mapped = await _await(
                    transfer(
                        function.steps[0].action,
                        observation,
                        source_observation,
                    )
                )
                transfer_result = (
                    mapped
                    if isinstance(mapped, TransferResult)
                    else TransferResult(None, reason="omnitransfer_result_invalid")
                )
            except Exception as error:  # noqa: BLE001
                transfer_result = TransferResult(
                    None,
                    reason=f"omnitransfer_error:{error}",
                )
        admission = assess_transfer(
            transfer_result,
            observation=observation,
        )
        decision["mapping_confidence"] = admission.confidence
        decision["entry_mapping_reason"] = (
            admission.reason or transfer_result.reason
        )
        decision["entry_mapping_target"] = (
            transfer_result.action.to_dict()
            if transfer_result.action is not None
            else None
        )
        if not admission.accepted:
            decision["rejection_reason"] = (
                admission.reason or "function_entry_mapping_rejected"
            )
            continue
        decision["rejection_reason"] = None
        admitted.append((_score, function, decision))

    selected = admitted[: max(0, int(limit))]
    selected_ids = {function.id for _score, function, _audit in selected}
    for decision in decisions:
        decision["selected"] = decision["function_id"] in selected_ids
        if not decision["selected"] and not decision.get("coarse_selected"):
            decision["rejection_reason"] = (
                "function_excluded_for_run"
                if decision["function_id"] in exclude_function_ids
                else (
                    "function_not_agent_visible"
                    if decision.get("agent_visible") is False
                    else "coarse_candidate_limit"
                )
            )
        elif not decision["selected"] and decision["rejection_reason"] is None:
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
                "checkpoint_sha256": getattr(
                    encoder, "checkpoint_sha256", None
                ),
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
            "page_similarity_threshold": FUNCTION_PAGE_SIMILARITY_THRESHOLD,
            "coarse_candidate_function_ids": [
                function.id for _score, function, _audit in coarse
            ],
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
    current_observation: Observation,
    current_page: TreeEmbedding | None,
    source_states: Mapping[str, Observation | None],
    encoder: PageEncoder,
) -> dict[str, Any]:
    source_state_id = function.steps[0].source_state_id if function.steps else ""
    source_observation = source_states.get(source_state_id)
    page_match = match_function_page(
        current=current_page,
        current_observation=current_observation,
        source_observation=source_observation,
        encoder=encoder,
    )
    observed_page_similarity = float(page_match["page_similarity"] or 0.0)
    entry_page_override = (
        "open_app"
        if function.steps and function.steps[0].action.tool == "open_app"
        else None
    )
    page_similarity = (
        1.0 if entry_page_override is not None else observed_page_similarity
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
        "agent_visible": function.agent_visible,
        "source_state_id": source_state_id,
        "page_similarity": page_similarity,
        "observed_page_similarity": observed_page_similarity,
        "entry_page_override": entry_page_override,
        "page_similarity_threshold": FUNCTION_PAGE_SIMILARITY_THRESHOLD,
        "page_match": page_match["matched"],
        "goal_lexical_score": goal_score,
        "score": score,
        "selected": False,
        "coarse_selected": False,
        "mapping_confidence": None,
        "entry_mapping_reason": None,
        "entry_mapping_target": None,
        "rejection_reason": None,
    }


async def _await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def match_function_page(
    *,
    source_observation: Observation | None,
    encoder: PageEncoder,
    current_observation: Observation | None = None,
    current: TreeEmbedding | None = None,
) -> dict[str, Any]:
    """Apply the hard Function page-identity gate using the native 512D encoder."""

    if source_observation is None:
        return _page_match_failure("function_source_page_missing")
    if current_observation is None and current is None:
        return _page_match_failure("function_current_page_missing")
    if current_observation is not None:
        source_package = _observation_package(source_observation)
        current_package = _observation_package(current_observation)
        if source_package and current_package and source_package != current_package:
            return _page_match_failure(
                "function_page_package_mismatch",
                source_package=source_package,
                current_package=current_package,
            )
    try:
        source_page = _embed_page(encoder, source_observation)
        current_page = current or _embed_page(encoder, current_observation)
    except (RuntimeError, ValueError) as error:
        return _page_match_failure(f"function_page_embedding_failed:{error}")
    if source_page is None or current_page is None:
        return _page_match_failure("function_page_embedding_missing")
    similarity = _cosine(current_page.vector, source_page.vector)
    matched = similarity >= FUNCTION_PAGE_SIMILARITY_THRESHOLD
    return {
        "matched": matched,
        "reason": None if matched else "function_page_similarity_below_threshold",
        "page_similarity": similarity,
        "minimum_page_similarity": FUNCTION_PAGE_SIMILARITY_THRESHOLD,
    }


def _page_match_failure(reason: str, **detail: Any) -> dict[str, Any]:
    return {
        "matched": False,
        "reason": reason,
        "page_similarity": None,
        "minimum_page_similarity": FUNCTION_PAGE_SIMILARITY_THRESHOLD,
        **detail,
    }


def _embed_page(
    encoder: PageEncoder,
    observation: Observation | None,
) -> TreeEmbedding | None:
    if observation is None:
        return None
    try:
        page = encoder.embed(observation)
    except ValueError as error:
        if str(error) == "omnitransfer_page_xml_required":
            return None
        raise
    if not page.elements or float(np.linalg.norm(page.vector)) <= 0.0:
        return None
    return page


def _observation_package(observation: Observation) -> str:
    explicit = str(observation.package_name or "").strip()
    if explicit:
        return explicit
    packages = re.findall(r'\bpackage="([^"]+)"', str(observation.xml or ""))
    app_packages = [
        package
        for package in packages
        if package and package != "com.android.systemui"
    ]
    if not app_packages:
        return ""
    counts = {package: app_packages.count(package) for package in set(app_packages)}
    return min(counts, key=lambda package: (-counts[package], package))


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
    "FUNCTION_PAGE_SIMILARITY_THRESHOLD",
    "PAGE_SIMILARITY_WEIGHT",
    "RECALL_AUDIT_VERSION",
    "RecallResult",
    "match_function_page",
    "recall_functions",
]
