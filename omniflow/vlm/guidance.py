from __future__ import annotations

DEFAULT_STEP_GUIDANCE = ""


def resolve_step_guidance(goal: str, explicit: str = "") -> str:
    del goal
    return str(explicit or "").strip()


__all__ = [
    "DEFAULT_STEP_GUIDANCE",
    "resolve_step_guidance",
]
