"""AgentRR transfer: the MobiMem-style same-harness baseline.

MobiMem (arXiv:2512.15784, "Beyond Training: Enabling Self-Evolution of Agents
with MOBIMEM") organises agent memory into three primitives.  Only one of them is
comparable to OmniFlow's Transfer, so only that one is reproduced here:

* Profile Memory (DisGraph over user preferences) has no counterpart in an
  AndroidWorld episode: the tasks carry no user-preference signal to retrieve.
* Experience Memory (multi-level templates) is MobiMem's version of the Function
  asset, which OmniFlow already owns.  Reproducing it would change what the
  score measures, not the mapper.
* Action Memory + AgentRR (agent record-and-replay) replays a recorded *action
  sequence*, reusing the recorded interaction instead of re-deriving a target
  for one action from a source screen.  This is the direct alternative to
  OmniTransfer, and it is what this module implements.

The difference to :mod:`omniflow.transfer.attribute` (MobileGPT-style) is real
and worth keeping distinct:

* attribute transfer re-locates **one** recorded element on the new screen by
  resource-id / text / content-desc and re-binds this action's own arguments.
* AgentRR replays the recorded **sequence** positionally and checks a safety
  condition before each dispatch.  When the guard fails it abstains
  (``agentrr_guard_*``) instead of searching for a better match, because a
  replay mechanism that starts guessing is no longer a replay mechanism.

Both share the same Function recall, argument binding, Checker and Planner, so
the only difference from the OmniFlow row is the mapper.  That is what makes the
row a same-harness baseline rather than a different system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omniflow.core.model import Action, Observation, TransferResult
from omniflow.runtime.execution import _display_size, _elements, _relative_source_point

# Actions whose target is a point on the screen and therefore need a safety
# check before a recorded coordinate may be replayed.
_POINT_TOOLS = frozenset({"click", "long_press", "input_text"})
# Actions whose payload is not positional: replaying them unchanged is safe as
# long as the package still matches (``input_text`` is handled as a point tool
# because its field must be found first).
_PASS_THROUGH_TOOLS = frozenset({"press_key", "back", "home", "open_app"})


@dataclass(frozen=True)
class AgentRRPolicy:
    """Safety conditions a replayed action must satisfy.

    ``require_package_match`` is MobiMem's own precondition: a recorded action is
    only replayable inside the application it was recorded in.  ``require_same_
    signature`` additionally requires the recorded element identity to still be
    present on the screen; turning it off is how a pure positional replay is
    expressed, which is the weakest form of the baseline.
    """

    require_package_match: bool = True
    require_same_signature: bool = True


DEFAULT_POLICY = AgentRRPolicy()


def agentrr_transfer(
    action: Action,
    observation: Observation,
    source_state: Observation | None = None,
) -> TransferResult:
    """Replay a recorded action onto the current observation, guarded."""

    return agentrr_transfer_with(DEFAULT_POLICY)(action, observation, source_state)


def agentrr_transfer_with(policy: AgentRRPolicy):
    """Build the Transfer callable for one policy (used by the CLI seam)."""

    def transfer(
        action: Action,
        observation: Observation,
        source_state: Observation | None = None,
    ) -> TransferResult:
        if action.tool in _PASS_THROUGH_TOOLS:
            return TransferResult(action)
        if action.tool == "swipe":
            # MobiMem's AgentRR replays a recorded drag verbatim when both
            # endpoints are still inside a scrollable container; a drag has no
            # element identity to compare, so the guard is the container only.
            return _replay_swipe(action, observation, source_state)
        if action.tool not in _POINT_TOOLS:
            return TransferResult(action)
        return _replay_point(action, observation, source_state, policy)

    return transfer


def _replay_point(
    action: Action,
    observation: Observation,
    source_state: Observation | None,
    policy: AgentRRPolicy,
) -> TransferResult:
    if source_state is None or not source_state.xml:
        return TransferResult(None, reason="agentrr_source_state_missing")
    try:
        point = _relative_source_point(
            source_state, float(action.args["x"]), float(action.args["y"])
        )
    except (KeyError, TypeError, ValueError):
        return TransferResult(None, reason="agentrr_invalid_source_point")
    source_elements = _elements(str(source_state.xml))
    recorded = _element_at(source_elements, point)
    if recorded is None:
        return TransferResult(None, reason="agentrr_recorded_element_missing")

    targets = _elements(str(observation.xml or ""))
    display = _display_size(observation, targets)
    if display is None:
        return TransferResult(None, reason="agentrr_display_size_missing")

    if policy.require_package_match:
        # Compare the screen's package, not the recorded leaf element's: most
        # Android nodes carry an empty ``package`` attribute and only the window
        # root declares it, so reading the leaf would make this guard a no-op.
        recorded_package = str(
            source_state.package_name or _package_from(source_elements) or ""
        ).strip()
        current_package = str(
            observation.package_name or _package_from(targets) or ""
        ).strip()
        if recorded_package and current_package and recorded_package != current_package:
            return TransferResult(
                None,
                reason="agentrr_guard_package_changed",
                detail={"recorded_package": recorded_package,
                        "current_package": current_package},
            )

    replayed = _replay_target(recorded, targets, point, policy)
    if replayed is None:
        return TransferResult(
            None,
            reason="agentrr_guard_signature_absent",
            detail={"recorded": _identity(recorded)},
        )
    if replayed["guard"] == "position":
        # Pure positional replay re-dispatches the recorded point itself.  There
        # is no target element to re-centre on, so the coordinates are already
        # the ones that were recorded; taking an element's centre here would
        # silently turn positional replay back into a re-location policy.
        params = {
            key: value
            for key, value in action.args.items()
            if key not in {"node_id", "node_resource_id"}
        }
        return TransferResult(
            Action(action.tool, params),
            reason="agentrr_replay_position",
            detail={"recorded": _identity(recorded)},
        )

    left, top, right, bottom = replayed["bounds"]
    params = {
        key: value
        for key, value in action.args.items()
        if key not in {"node_id", "node_resource_id"}
    }
    params["x"] = (left + right) / 2.0 / display[0] * 1000.0
    params["y"] = (top + bottom) / 2.0 / display[1] * 1000.0
    return TransferResult(
        Action(action.tool, params),
        # The reason states which safety condition admitted the replay, so a
        # result can be audited without re-running the guard.
        reason=f"agentrr_replay_{replayed['guard']}",
        detail={"recorded": _identity(recorded), "target": _identity(replayed)},
    )


def _replay_swipe(
    action: Action,
    observation: Observation,
    source_state: Observation | None,
) -> TransferResult:
    """Replay a recorded drag, scaled into the current display."""

    if source_state is None or not source_state.xml:
        return TransferResult(None, reason="agentrr_source_state_missing")
    try:
        start = _relative_source_point(
            source_state, float(action.args["x1"]), float(action.args["y1"])
        )
        end = _relative_source_point(
            source_state, float(action.args["x2"]), float(action.args["y2"])
        )
    except (KeyError, TypeError, ValueError):
        return TransferResult(None, reason="agentrr_invalid_source_point")
    source_elements = _elements(str(source_state.xml))
    container = _element_at(source_elements, start)
    targets = _elements(str(observation.xml or ""))
    display = _display_size(observation, targets)
    if display is None:
        return TransferResult(None, reason="agentrr_display_size_missing")
    if container is None:
        return TransferResult(None, reason="agentrr_recorded_element_missing")
    replayed = _match_container(container, targets)
    if replayed is None:
        return TransferResult(
            None,
            reason="agentrr_guard_signature_absent",
            detail={"recorded": _identity(container)},
        )
    # Replay the recorded gesture's *shape* inside the matched container rather
    # than its absolute coordinates: the container is what transferred.
    source_bounds = container["bounds"]
    target_bounds = replayed["bounds"]
    scale_x = (target_bounds[2] - target_bounds[0]) / max(1, source_bounds[2] - source_bounds[0])
    scale_y = (target_bounds[3] - target_bounds[1]) / max(1, source_bounds[3] - source_bounds[1])
    params = dict(action.args)
    params["x1"] = (
        target_bounds[0] + (start[0] - source_bounds[0]) * scale_x
    ) / display[0] * 1000.0
    params["y1"] = (
        target_bounds[1] + (start[1] - source_bounds[1]) * scale_y
    ) / display[1] * 1000.0
    params["x2"] = (
        target_bounds[0] + (end[0] - source_bounds[0]) * scale_x
    ) / display[0] * 1000.0
    params["y2"] = (
        target_bounds[1] + (end[1] - source_bounds[1]) * scale_y
    ) / display[1] * 1000.0
    return TransferResult(
        Action(action.tool, params),
        reason=f"agentrr_replay_{replayed['guard']}",
        detail={"recorded": _identity(container), "target": _identity(replayed)},
    )


def _replay_target(
    recorded: dict[str, Any],
    targets: list[dict[str, Any]],
    point: tuple[float, float],
    policy: AgentRRPolicy,
) -> dict[str, Any] | None:
    """Return the on-screen element the recorded action replays onto.

    A replay keeps the recorded interaction's identity.  When the exact identity
    is gone, AgentRR abstains rather than choosing a loose alternative, unless the
    policy explicitly allows positional replay.
    """

    if policy.require_same_signature:
        exact = _match_exact(recorded, targets)
        if exact is not None:
            return {**exact, "guard": "signature"}
        return None
    # The unguarded mode does not look for a target at all: the recorded point is
    # the action.  ``point`` is returned as a sentinel marker only.
    return {"guard": "position", "bounds": (point[0], point[1], point[0], point[1])}


def _match_exact(
    recorded: dict[str, Any], targets: list[dict[str, Any]]
) -> dict[str, Any] | None:
    rid = recorded.get("resource_id") or ""
    text = recorded.get("text") or ""
    description = recorded.get("description") or ""
    klass = recorded.get("class") or ""
    for rule in (
        lambda e: rid and e["resource_id"] == rid and e["text"] == text,
        lambda e: rid and e["resource_id"] == rid and e["class"] == klass,
        lambda e: text and e["text"] == text and e["class"] == klass,
        lambda e: description and e["description"] == description and e["class"] == klass,
    ):
        hits = [element for element in targets if rule(element)]
        if len(hits) == 1:
            # Ambiguity is a guard failure: two candidates means the recorded
            # interaction no longer identifies one target.
            return hits[0]
        if len(hits) > 1:
            return None
    return None


def _match_container(
    recorded: dict[str, Any], targets: list[dict[str, Any]]
) -> dict[str, Any] | None:
    matched = _match_exact(recorded, targets)
    return None if matched is None else {**matched, "guard": "signature"}


def _element_at(
    elements: list[dict[str, Any]], point: tuple[float, float]
) -> dict[str, Any] | None:
    x, y = point
    hits = [
        element
        for element in elements
        if element["bounds"][0] <= x <= element["bounds"][2]
        and element["bounds"][1] <= y <= element["bounds"][3]
    ]
    labelled = [
        element
        for element in hits
        if element["resource_id"] or element["text"] or element["description"]
    ]
    pool = labelled or hits
    return min(pool, key=_area) if pool else None


def _package_from(elements: list[dict[str, Any]]) -> str:
    for element in elements:
        package = str(element.get("package") or "").strip()
        if package:
            return package
    return ""


def _area(element: dict[str, Any]) -> int:
    left, top, right, bottom = element["bounds"]
    return (right - left) * (bottom - top)


def _identity(element: dict[str, Any]) -> dict[str, str]:
    return {key: element[key] for key in ("resource_id", "text", "description", "class")}


__all__ = [
    "AgentRRPolicy",
    "DEFAULT_POLICY",
    "agentrr_transfer",
    "agentrr_transfer_with",
]
