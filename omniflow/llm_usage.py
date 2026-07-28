from __future__ import annotations

from typing import Any


_COUNTER_KEYS = (
    "model_calls",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "responses_with_usage",
    "responses_without_usage",
    "failed_calls",
)


class LLMUsageTracker:
    def __init__(self, *, component: str, model: str):
        self.component = str(component).strip()
        self.model = str(model).strip()
        self._pending = _empty_counters()

    def start_call(self) -> None:
        self._pending["model_calls"] += 1

    def record_response(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is None:
            self._pending["responses_without_usage"] += 1
            return
        prompt_tokens = _usage_int(usage, "prompt_tokens")
        completion_tokens = _usage_int(usage, "completion_tokens")
        total_tokens = _usage_int(usage, "total_tokens")
        if total_tokens <= 0:
            total_tokens = prompt_tokens + completion_tokens
        self._pending["prompt_tokens"] += prompt_tokens
        self._pending["completion_tokens"] += completion_tokens
        self._pending["total_tokens"] += total_tokens
        self._pending["responses_with_usage"] += 1

    def record_failure(self) -> None:
        self._pending["failed_calls"] += 1

    def take_usage(self) -> dict[str, Any]:
        counters = dict(self._pending)
        self._pending = _empty_counters()
        return {
            "component": self.component,
            "model": self.model,
            **counters,
            "token_usage_status": token_usage_status(counters),
        }


def merge_usage(
    aggregate: dict[str, Any],
    usage: dict[str, Any] | None,
    *,
    component: str,
) -> None:
    if not usage:
        return
    component_name = str(usage.get("component") or component).strip() or component
    component_usage = {
        key: max(0, _coerce_int(usage.get(key))) for key in _COUNTER_KEYS
    }
    component_usage["model"] = str(usage.get("model") or "").strip() or None
    component_usage["token_usage_status"] = token_usage_status(component_usage)
    by_component = aggregate.setdefault("by_component", {})
    previous = by_component.get(component_name)
    if isinstance(previous, dict):
        for key in _COUNTER_KEYS:
            component_usage[key] += max(0, _coerce_int(previous.get(key)))
        component_usage["model"] = component_usage["model"] or previous.get("model")
        component_usage["token_usage_status"] = token_usage_status(component_usage)
    by_component[component_name] = component_usage
    for key in _COUNTER_KEYS:
        aggregate[key] = sum(
            max(0, _coerce_int(value.get(key)))
            for value in by_component.values()
            if isinstance(value, dict)
        )
    aggregate["token_usage_status"] = token_usage_status(aggregate)


def token_usage_status(usage: dict[str, Any]) -> str:
    model_calls = max(0, _coerce_int(usage.get("model_calls")))
    responses_with_usage = max(
        0,
        _coerce_int(usage.get("responses_with_usage")),
    )
    if model_calls == 0:
        return "not_applicable"
    if responses_with_usage == 0:
        return "unavailable"
    if responses_with_usage < model_calls:
        return "partial"
    return "tracked"


def _empty_counters() -> dict[str, int]:
    return {key: 0 for key in _COUNTER_KEYS}


def _usage_int(usage: Any, key: str) -> int:
    value = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, 0)
    return max(0, _coerce_int(value))


def _coerce_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
