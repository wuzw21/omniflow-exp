from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from omniflow.core.config import DEFAULT_MAX_STEPS
from omniflow.core.model import Function, Observation, ToolCall
from omniflow.vlm.gui import (
    ModelToolCallError,
    build_model_turn_request,
    parse_model_turn_response,
)
from omniflow.vlm.guidance import resolve_step_guidance
from omniflow.vlm.model_config import resolve_openai_compatible_config
from omniflow.vlm.usage import LLMUsageTracker

_MODEL_TOOL_CALL_ATTEMPTS = 2
ModelTurnTransport = Callable[[dict[str, Any]], dict[str, Any]]
MetadataSink = Callable[[dict[str, Any]], None]


class VLMPlanner:
    """The single GUI Planner used by both the bridge and experiments."""

    def __init__(
        self,
        *,
        model: str,
        provider: str = "openai",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        client: Any | None = None,
        transport: ModelTurnTransport | None = None,
        target_package_name: str = "",
        step_skill_guidance: str = "",
        max_steps: int = DEFAULT_MAX_STEPS,
        metadata_sink: MetadataSink | None = None,
    ):
        if provider not in {"openai", "openai_compatible"}:
            raise ValueError("VLMPlanner supports OpenAI-compatible providers only")
        self.model = str(model).strip()
        if not self.model:
            raise ValueError("planner_model_required")
        self.timeout = float(timeout)
        self.target_package_name = str(target_package_name).strip()
        self.step_skill_guidance = str(step_skill_guidance).strip()
        self.max_steps = max(1, int(max_steps))
        self._client = client
        self._transport = transport
        self._metadata_sink = metadata_sink
        self._api_key, self._base_url = resolve_openai_compatible_config(
            api_key=api_key,
            base_url=base_url,
        )
        self._usage = LLMUsageTracker(component="planner", model=self.model)
        self._metadata: dict[str, Any] = {}
        self._rejected_tool_calls: list[dict[str, Any]] = []
        self._turn_index = 0

    async def one_step_tool_call(
        self,
        goal: str,
        observation: Observation,
        functions: tuple[Function, ...] = (),
        installed_apps: dict[str, str] | None = None,
    ) -> ToolCall:
        state = planner_state(observation)
        installed_app_catalog = dict(installed_apps or {})
        validation_error = ""
        retry_tool_name = ""
        rejected_tool_call: dict[str, Any] | None = None
        lightweight_retry = False
        self._metadata.clear()
        self._rejected_tool_calls.clear()
        metadata: dict[str, Any] = {}

        for attempt in range(_MODEL_TOOL_CALL_ATTEMPTS):
            self._turn_index += 1
            guidance = resolve_step_guidance(goal, self.step_skill_guidance)
            request = build_model_turn_request(
                goal=str(goal),
                model=self.model,
                state=state,
                target_package_name=self.target_package_name,
                step_skill_guidance=guidance,
                installed_apps=installed_app_catalog,
                functions=functions,
                max_steps=self.max_steps,
                turn_index=self._turn_index,
                validation_error=validation_error,
                retry_tool_name=retry_tool_name,
                rejected_tool_call=rejected_tool_call,
                lightweight_retry=lightweight_retry,
            )
            envelope = {
                "goal": str(goal),
                "model": self.model,
                "state": state,
                "target_package_name": self.target_package_name,
                "step_skill_guidance": guidance,
                "max_steps": self.max_steps,
                "request": request,
            }
            self._usage.start_call()
            try:
                response = self._call_transport(envelope)
            except Exception:
                self._usage.record_failure()
                raise
            self._usage.record_response(response)
            try:
                tool_call, metadata = parse_model_turn_response(
                    response,
                    requested_model=self.model,
                    turn_index=self._turn_index,
                    functions=functions,
                    installed_apps=installed_app_catalog,
                    display=(
                        state.get("display")
                        if isinstance(state.get("display"), dict)
                        else None
                    ),
                )
                break
            except ModelToolCallError as error:
                rejected_entry: dict[str, Any] = {
                    "turn_index": self._turn_index,
                    "tool": error.tool_name or None,
                    "error": str(error),
                }
                if error.arguments is not None:
                    rejected_entry["arguments"] = error.arguments
                self._rejected_tool_calls.append(rejected_entry)
                if attempt == _MODEL_TOOL_CALL_ATTEMPTS - 1:
                    self._publish_metadata(
                        {"rejected_tool_calls": list(self._rejected_tool_calls)}
                    )
                    raise
                validation_error = str(error)
                retry_tool_name = (
                    ""
                    if error.code.startswith("model_turn_tool_not_visible:")
                    else error.tool_name
                )
                rejected_tool_call = {
                    "tool": error.tool_name or None,
                    "arguments": error.arguments,
                }
                # The state has not changed, and retries expose only the rejected
                # tool. Re-sending screenshots, projected UI, and skill context
                # adds tokens without adding evidence needed to repair arguments.
                lightweight_retry = True
        else:
            raise AssertionError("unreachable")

        if self._rejected_tool_calls:
            metadata["rejected_tool_calls"] = list(self._rejected_tool_calls)
        self._publish_metadata(metadata)
        return tool_call

    def take_metadata(self) -> dict[str, Any]:
        metadata = dict(self._metadata)
        self._metadata.clear()
        return metadata

    def take_usage(self) -> dict[str, Any]:
        return self._usage.take_usage()

    def _publish_metadata(self, metadata: dict[str, Any]) -> None:
        self._metadata = dict(metadata)
        if self._metadata_sink is not None:
            self._metadata_sink(dict(metadata))

    def _call_transport(self, envelope: dict[str, Any]) -> dict[str, Any]:
        if self._transport is not None:
            response = self._transport(envelope)
            if not isinstance(response, dict):
                raise ValueError("model_turn_transport_response_invalid")
            return response
        request = dict(envelope["request"])
        extra_body = dict(request.get("extra_body") or {})
        for field in ("enable_thinking", "thinking"):
            extra_body[field] = request.pop(field)
        request["extra_body"] = extra_body
        client = self._client or self._build_client()
        response = client.chat.completions.create(
            **request,
            timeout=self.timeout,
        )
        return normalize_openai_model_turn_response(
            response,
            requested_model=self.model,
        )

    def _build_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install omniflow[llm] to use VLMPlanner") from exc
        options: dict[str, Any] = {"api_key": self._api_key or "not-required"}
        if self._base_url:
            options["base_url"] = self._base_url
        return OpenAI(**options)


def planner_state(observation: Observation) -> dict[str, Any]:
    state = observation.to_dict()
    state["state_id"] = str(observation.extra.get("state_id") or "").strip()
    for key in ("display", "screenshot_path"):
        if observation.extra.get(key) is not None:
            state[key] = observation.extra[key]
    state["extra"] = {
        key: value
        for key, value in observation.extra.items()
        if key not in {"state_id", "display", "screenshot_path"}
    }
    return {key: value for key, value in state.items() if value is not None}


def normalize_openai_model_turn_response(
    response: Any,
    *,
    requested_model: str,
) -> dict[str, Any]:
    if _field(response, "choices") is not None:
        return _normalize_completion(response, requested_model=requested_model)
    if isinstance(response, dict) or not isinstance(response, Iterable):
        raise TypeError("model_turn_openai_response_invalid")
    return _normalize_stream(response, requested_model=requested_model)


def _normalize_completion(response: Any, *, requested_model: str) -> dict[str, Any]:
    choices = list(_field(response, "choices") or ())
    message = _field(choices[0], "message") if choices else None
    tool_calls = [
        _normalized_tool_call(value)
        for value in (_field(message, "tool_calls") or ())
    ]
    return {
        "requested_model": requested_model,
        "resolved_model": str(_field(response, "model") or requested_model),
        "tool_calls": tool_calls,
        "reasoning": str(
            _field(message, "reasoning_content")
            or _field(message, "reasoning")
            or ""
        ),
        "usage": _usage_dict(_field(response, "usage")) or None,
    }


def _normalize_stream(
    chunks: Iterable[Any],
    *,
    requested_model: str,
) -> dict[str, Any]:
    tool_calls: dict[int, dict[str, str]] = {}
    reasoning_parts: list[str] = []
    usage: dict[str, int] = {}
    resolved_model = requested_model
    for chunk in chunks:
        resolved_model = str(_field(chunk, "model") or resolved_model)
        chunk_usage = _usage_dict(_field(chunk, "usage"))
        if chunk_usage:
            usage = chunk_usage
        for choice in _field(chunk, "choices") or ():
            delta = _field(choice, "delta")
            reasoning = _field(delta, "reasoning_content") or _field(
                delta, "reasoning"
            )
            if reasoning:
                reasoning_parts.append(str(reasoning))
            for fallback_index, value in enumerate(
                _field(delta, "tool_calls") or ()
            ):
                index = int(_field(value, "index") or fallback_index)
                function = _field(value, "function")
                entry = tool_calls.setdefault(index, {"name": "", "arguments": ""})
                name = _field(function, "name")
                arguments = _field(function, "arguments")
                if name:
                    entry["name"] += str(name)
                if arguments:
                    entry["arguments"] += str(arguments)
    return {
        "requested_model": requested_model,
        "resolved_model": resolved_model,
        "tool_calls": [
            {"function": tool_calls[index]}
            for index in sorted(tool_calls)
        ],
        "reasoning": "".join(reasoning_parts),
        "usage": usage or None,
    }


def _normalized_tool_call(value: Any) -> dict[str, Any]:
    function = _field(value, "function")
    return {
        "function": {
            "name": str(_field(function, "name") or ""),
            "arguments": str(_field(function, "arguments") or ""),
        }
    }


def _usage_dict(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    result: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        raw = _field(value, key)
        try:
            result[key] = max(0, int(raw or 0))
        except (TypeError, ValueError):
            result[key] = 0
    return result


def _field(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else getattr(value, key, None)


__all__ = [
    "ModelTurnTransport",
    "VLMPlanner",
    "normalize_openai_model_turn_response",
    "planner_state",
]
