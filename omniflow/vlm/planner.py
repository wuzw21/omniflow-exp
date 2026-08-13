from __future__ import annotations

from collections.abc import Callable
from typing import Any

from omniflow.core.config import PromptSet
from omniflow.core.model import Function, Observation, ToolCall
from omniflow.vlm.gui import (
    ModelToolCallError,
    SYSTEM_PROMPT,
    build_model_turn_request,
    parse_model_turn_response,
)
from omniflow.vlm.model_config import resolve_openai_compatible_config
from omniflow.vlm.usage import LLMUsageTracker

ModelTransport = Callable[[dict[str, Any]], Any]


class VLMPlanner:
    """Choose one native GUI or recalled Function tool per model turn."""

    def __init__(
        self,
        *,
        model: str,
        provider: str = "openai",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        client: Any | None = None,
        transport: ModelTransport | None = None,
        metadata_sink: Callable[[dict[str, Any]], None] | None = None,
        prompts: PromptSet | None = None,
        target_package_name: str = "",
        step_skill_guidance: str = "",
        max_steps: int = 20,
    ):
        if provider not in {"openai", "openai_compatible"}:
            raise ValueError("VLMPlanner supports OpenAI-compatible providers only")
        if client is not None and transport is not None:
            raise ValueError("planner_model_transport_ambiguous")
        self.model = str(model).strip()
        self.timeout = float(timeout)
        self._client = client
        self._transport = transport
        self._metadata_sink = metadata_sink
        self._api_key, self._base_url = resolve_openai_compatible_config(
            api_key=api_key,
            base_url=base_url,
        )
        self.system_prompt = prompts.planner_system if prompts is not None else SYSTEM_PROMPT
        self.target_package_name = str(target_package_name).strip()
        self.step_skill_guidance = str(step_skill_guidance).strip()
        self.max_steps = max(1, int(max_steps))
        self._turn_index = 0
        self._metadata: dict[str, Any] = {}
        self._usage = LLMUsageTracker(component="planner", model=self.model)

    async def one_step_tool_call(
        self,
        goal: str,
        observation: Observation,
        functions: tuple[Function, ...] = (),
        installed_apps: dict[str, str] | None = None,
    ) -> ToolCall:
        state = _planner_state(observation)
        validation_error = ""
        retry_tool_name = ""
        rejected_tool_call: dict[str, Any] | None = None
        rejected_calls: list[dict[str, Any]] = []
        for attempt in range(2):
            self._turn_index += 1
            request = build_model_turn_request(
                goal=str(goal),
                model=self.model,
                state=state,
                target_package_name=self.target_package_name,
                step_skill_guidance=self.step_skill_guidance,
                installed_apps=installed_apps or {},
                functions=functions,
                max_steps=self.max_steps,
                turn_index=self._turn_index,
                validation_error=validation_error,
                retry_tool_name=retry_tool_name,
                rejected_tool_call=rejected_tool_call,
                lightweight_retry=attempt > 0 and not retry_tool_name,
                system_prompt=self.system_prompt,
            )
            self._usage.start_call()
            try:
                response = self._call_model(
                    {
                        "goal": str(goal),
                        "model": self.model,
                        "target_package_name": self.target_package_name,
                        "step_skill_guidance": self.step_skill_guidance,
                        "max_steps": self.max_steps,
                        "request": request,
                    }
                )
                self._usage.record_response(response)
                tool_call, metadata = parse_model_turn_response(
                    _normalize_response(response, requested_model=self.model),
                    requested_model=self.model,
                    turn_index=self._turn_index,
                    functions=functions,
                    display=state.get("display"),
                )
            except ModelToolCallError as error:
                rejected = {
                    "turn_index": self._turn_index,
                    "tool": error.tool_name or None,
                    "error": str(error),
                }
                if error.arguments is not None:
                    rejected["arguments"] = error.arguments
                rejected_calls.append(rejected)
                if attempt == 1:
                    self._metadata = {"rejected_tool_calls": rejected_calls}
                    raise
                validation_error = str(error)
                retry_tool_name = error.tool_name
                rejected_tool_call = {
                    "tool": error.tool_name or None,
                    "arguments": error.arguments,
                }
                continue
            except Exception:
                self._usage.record_failure()
                raise
            if rejected_calls:
                metadata["rejected_tool_calls"] = rejected_calls
            self._metadata = metadata
            if self._metadata_sink is not None:
                self._metadata_sink(dict(metadata))
            return tool_call
        raise AssertionError("unreachable")

    def take_metadata(self) -> dict[str, Any]:
        metadata = dict(self._metadata)
        self._metadata.clear()
        return metadata

    def take_usage(self) -> dict[str, Any]:
        return self._usage.take_usage()

    def _call_model(self, envelope: dict[str, Any]) -> Any:
        if self._transport is not None:
            return self._transport(envelope)
        client = self._client or self._build_client()
        options = dict(envelope["request"])
        options.pop("enable_thinking", None)
        options["stream"] = False
        options.pop("stream_options", None)
        options["timeout"] = self.timeout
        return client.chat.completions.create(**options)

    def _build_client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install omniflow[llm] to use VLMPlanner") from exc
        options: dict[str, Any] = {
            "api_key": self._api_key or "not-required",
            "max_retries": 0,
        }
        if self._base_url:
            options["base_url"] = self._base_url
        return OpenAI(**options)


def _planner_state(observation: Observation) -> dict[str, Any]:
    state = observation.to_dict()
    state["state_id"] = str(observation.extra.get("state_id") or "").strip()
    for key in ("display", "screenshot_path"):
        if observation.extra.get(key) is not None:
            state[key] = observation.extra[key]
    return state


def _normalize_response(value: Any, *, requested_model: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    choices = getattr(value, "choices", None) or ()
    message = getattr(choices[0], "message", None) if choices else None
    calls = getattr(message, "tool_calls", None) or ()
    usage = getattr(value, "usage", None)
    return {
        "requested_model": requested_model,
        "resolved_model": str(getattr(value, "model", None) or requested_model),
        "reasoning": str(getattr(message, "reasoning", None) or ""),
        "tool_calls": [
            {
                "function": {
                    "name": str(getattr(call.function, "name", "") or ""),
                    "arguments": str(getattr(call.function, "arguments", "") or ""),
                }
            }
            for call in calls
        ],
        "usage": (
            {
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            }
            if usage is not None
            else None
        ),
    }


__all__ = ["ModelTransport", "VLMPlanner"]
