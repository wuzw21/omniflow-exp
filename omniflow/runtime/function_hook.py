from __future__ import annotations

from typing import Any, Protocol

from omniflow.core.model import Function, Observation


class FunctionPlannerHook(Protocol):
    """Optional seam shared by every Planner implementation.

    The hook does not choose actions or change the Planner prompt.  It only
    controls which registered Functions are exposed and carries a failed
    Transfer's ranked candidates into the next Planner observation.
    """

    def register_functions(
        self,
        *,
        goal: str,
        observation: Observation,
        functions: tuple[Function, ...],
    ) -> tuple[Function, ...]: ...

    def on_replay_failure(
        self,
        *,
        observation: Observation,
        trace: Any,
    ) -> Observation: ...


class DefaultFunctionPlannerHook:
    """Small, provider-neutral Function registration/fallback adapter."""

    def register_functions(
        self,
        *,
        goal: str,
        observation: Observation,
        functions: tuple[Function, ...],
    ) -> tuple[Function, ...]:
        del goal, observation
        visible: list[Function] = []
        seen: set[str] = set()
        for function in functions:
            if not function.agent_visible or function.id in seen:
                continue
            seen.add(function.id)
            visible.append(function)
        return tuple(visible)

    def on_replay_failure(
        self,
        *,
        observation: Observation,
        trace: Any,
    ) -> Observation:
        hint = extract_transfer_candidates_hint(trace)
        if not hint:
            return observation
        return Observation(
            xml=observation.xml,
            package_name=observation.package_name,
            activity_name=observation.activity_name,
            image_base64=observation.image_base64,
            extra={
                **dict(observation.extra),
                "transfer_candidates_hint": hint,
            },
        )


def extract_transfer_candidates_hint(
    trace: Any,
    *,
    limit: int = 5,
) -> dict[str, Any] | None:
    """Return a compact ranked hint for normal Planner fallback."""

    if not isinstance(trace, list):
        return None
    for item in reversed(trace):
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata")
        transfer = metadata.get("transfer") if isinstance(metadata, dict) else None
        if not isinstance(transfer, dict):
            continue
        candidates = transfer.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            continue
        selected: list[dict[str, Any]] = []
        for candidate in candidates[: max(1, int(limit))]:
            if not isinstance(candidate, dict):
                continue
            selected.append(
                {
                    key: candidate[key]
                    for key in (
                        "rank",
                        "text",
                        "content_desc",
                        "class",
                        "bounds",
                        "execution_bounds",
                        "resource_id",
                        "execution_candidate_id",
                        "executable",
                        "score",
                    )
                    if candidate.get(key) is not None
                }
            )
        if not selected:
            return None
        return {
            "reason": "OmniTransfer candidate hint after a recoverable mapping rejection.",
            "mapping_confidence": transfer.get("mapping_confidence"),
            "candidates": selected,
        }
    return None


__all__ = [
    "DefaultFunctionPlannerHook",
    "FunctionPlannerHook",
    "extract_transfer_candidates_hint",
]
