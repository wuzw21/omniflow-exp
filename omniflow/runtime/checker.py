from __future__ import annotations

from typing import Any

from omniflow.core.schemas import canonicalize_action

_CHECKER_ACTIONS = frozenset({"click", "input_text", "long_press"})
_RULE_FIELDS = {"source_state_id", "action"}


def validate_checker_rule(value: Any) -> dict[str, Any]:
    """Validate one RunLog-grounded condition executed through OmniTransfer."""

    if not isinstance(value, dict) or set(value) != _RULE_FIELDS:
        raise ValueError("checker_rule_contract_invalid")
    source_state_id = str(value.get("source_state_id") or "").strip()
    if not source_state_id:
        raise ValueError("checker_source_state_id_required")
    action = canonicalize_action(value.get("action"), replayable_only=True)
    if action["tool"] not in _CHECKER_ACTIONS:
        raise ValueError(
            f"checker_action_requires_transfer_target:{action['tool']}"
        )
    if not all(action["args"].get(name) is not None for name in ("x", "y")):
        raise ValueError(
            f"checker_action_requires_source_target:{action['tool']}"
        )
    return {
        "source_state_id": source_state_id,
        "action": action,
    }


__all__ = ["validate_checker_rule"]
