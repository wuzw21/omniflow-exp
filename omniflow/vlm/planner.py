from __future__ import annotations

import base64
from collections.abc import Callable
from copy import deepcopy
import json
import math
from pathlib import Path
import re
from typing import Any

from omniflow.core.config import DEFAULT_PLANNER_SYSTEM_PROMPT, PromptSet
from omniflow.core.model import Function, Observation, ToolCall
from omniflow.core.schemas import canonicalize_action, vlm_action_tools
from omniflow.functions.assets import validate_arguments
from omniflow.vlm.context import analyze_page_context
from omniflow.vlm.model_config import resolve_openai_compatible_config
from omniflow.vlm.usage import LLMUsageTracker
from omniflow.vlm_coordinates import screen_pixel_args_to_canonical

SYSTEM_PROMPT = DEFAULT_PLANNER_SYSTEM_PROMPT


class ModelToolCallError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        tool_name: str = "",
        arguments: Any = None,
    ):
        self.code = str(message)
        self.tool_name = str(tool_name).strip()
        self.arguments = arguments
        super().__init__(message)


def build_model_turn_request(
    *,
    goal: str,
    model: str,
    state: dict[str, Any],
    max_steps: int,
    turn_index: int,
    step_skill_guidance: str = "",
    installed_apps: dict[str, str] | None = None,
    functions: list[Function] | tuple[Function, ...] = (),
    system_prompt: str = SYSTEM_PROMPT,
    history: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    text = _turn_text(
        goal=goal,
        state=state,
        step_skill_guidance=step_skill_guidance,
    )
    content: list[dict[str, Any]] = []
    current_image = _state_image_data_uri(state)
    if current_image:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": current_image, "detail": "high"},
            }
        )
    content.append({"type": "text", "text": text})
    tools = function_tools(functions)
    tools.extend(vlm_action_tools())
    tools = constrain_open_app_tool(
        tools,
        installed_apps or {},
    )
    return {
        "model": str(model),
        "messages": [
            {"role": "system", "content": str(system_prompt)},
            *deepcopy(list(history)),
            {"role": "user", "content": content},
        ],
        "max_completion_tokens": 512,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": tools,
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "reasoning_effort": "none",
        "enable_thinking": False,
    }

def parse_model_turn_response(
    value: Any,
    *,
    requested_model: str,
    turn_index: int,
    display: dict[str, Any] | None = None,
    functions: list[Function] | tuple[Function, ...] = (),
) -> tuple[ToolCall, dict[str, Any]]:
    if not isinstance(value, dict):
        raise ValueError("model_turn_response_invalid")
    if str(value.get("requested_model") or "").strip() != requested_model:
        raise ValueError("model_turn_requested_model_mismatch")
    tool_calls = value.get("tool_calls")
    if not isinstance(tool_calls, list):
        raise ModelToolCallError("model_turn_tool_calls_invalid")
    if len(tool_calls) != 1:
        raise ModelToolCallError(
            f"provider_tool_call_contract_violation:expected_one_native_tool_call:got_{len(tool_calls)}"
        )
    tool_call = tool_calls[0]
    if not isinstance(tool_call, dict):
        raise ModelToolCallError("model_turn_tool_call_invalid")
    function = tool_call.get("function")
    if not isinstance(function, dict):
        raise ModelToolCallError("model_turn_tool_call_function_invalid")
    tool = str(function.get("name") or "").strip()
    model_visible_tools = {
        str(item.get("function", {}).get("name") or "")
        for item in vlm_action_tools()
        if isinstance(item, dict) and isinstance(item.get("function"), dict)
    }
    function_catalog = {function.id: function for function in functions}
    model_visible_tools.update(function_catalog)
    if tool not in model_visible_tools:
        raise ModelToolCallError(f"model_turn_tool_not_visible:{tool}")
    raw_arguments = function.get("arguments")
    if not isinstance(raw_arguments, str):
        raise ModelToolCallError(
            "model_turn_tool_arguments_invalid",
            tool_name=tool,
            arguments=raw_arguments,
        )
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as error:
        raise ModelToolCallError(
            "model_turn_tool_arguments_must_be_json",
            tool_name=tool,
            arguments=raw_arguments,
        ) from error
    if not isinstance(arguments, dict):
        raise ModelToolCallError(
            "model_turn_tool_arguments_must_be_object",
            tool_name=tool,
            arguments=arguments,
        )
    rejected_arguments = dict(arguments)
    resolved_model = str(value.get("resolved_model") or requested_model).strip()
    adapter_metadata = None
    try:
        if tool in function_catalog:
            validate_arguments(function_catalog[tool].input_schema, arguments)
        else:
            arguments, adapter_metadata = adapt_tool_arguments(
                tool=tool,
                arguments=arguments,
                requested_model=requested_model,
                resolved_model=resolved_model,
                display=display,
            )
            canonical = canonicalize_action(
                {"tool": tool, "args": arguments},
                persisted_only=False,
                allow_non_action=True,
            )
            arguments = dict(canonical["args"])
    except ValueError as error:
        raise ModelToolCallError(
            str(error),
            tool_name=tool,
            arguments=rejected_arguments,
        ) from error
    metadata: dict[str, Any] = {}
    if adapter_metadata is not None:
        metadata["model_adapter"] = adapter_metadata
    thinking = str(value.get("reasoning") or "").strip()
    if thinking:
        metadata["thinking"] = thinking
    usage = value.get("usage")
    if isinstance(usage, dict):
        metadata["token_usage"] = {
            **usage,
            "model": requested_model,
            "resolved_model": resolved_model,
            "turn_index": int(turn_index),
        }
    return ToolCall(tool, arguments), metadata


def function_tools(
    functions: list[Function] | tuple[Function, ...],
) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for function in functions:
        parameters = deepcopy(function.input_schema)
        properties = parameters.setdefault("properties", {})
        parameters["properties"] = properties
        parameters["required"] = list(parameters.get("required") or ())
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": function.id,
                    "description": function.description,
                    "strict": True,
                    "parameters": parameters,
                },
            }
        )
    return tools


def _turn_text(
    *,
    goal: str,
    state: dict[str, Any],
    step_skill_guidance: str,
) -> str:
    display = state.get("display") if isinstance(state.get("display"), dict) else {}
    lines = [
        f"Task: {goal}",
        (
            "Current screen: "
            f"package={state.get('package_name') or '-'}, "
            f"activity={state.get('activity_name') or '-'}, "
            f"display={display.get('width') or '?'}x{display.get('height') or '?'}"
        ),
    ]
    if step_skill_guidance.strip():
        lines.append(f"Task guidance: {step_skill_guidance.strip()}")
    page_context = analyze_page_context(state)
    if page_context.useful:
        lines.append(page_context.evidence)
    raw_extra = state.get("extra")
    if isinstance(raw_extra, dict):
        previous_error = str(raw_extra.get("previous_action_error") or "").strip()
        if previous_error:
            lines.append(f"Previous action error: {previous_error}")
        function_execution = raw_extra.get("function_execution")
        if isinstance(function_execution, dict):
            function_id = str(function_execution.get("function_id") or "").strip()
            status = str(function_execution.get("replay_status") or "").strip()
            if function_id and status:
                result = "succeeded" if status == "actions_succeeded" else status
                lines.append(f"Previous Function result: {function_id} {result}")
        user_input = str(raw_extra.get("user_input") or "").strip()
        if user_input:
            lines.append(f"User response: {user_input}")
    return "\n".join(lines)


def constrain_open_app_tool(
    tools: list[dict[str, Any]],
    installed_apps: dict[str, str],
) -> list[dict[str, Any]]:
    candidates = _installed_app_candidates(installed_apps)
    packages = list(dict.fromkeys(package for _label, package in candidates))
    constrained: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            constrained.append(tool)
            continue
        parameters = function.get("parameters")
        properties = (
            parameters.get("properties") if isinstance(parameters, dict) else None
        )
        package_schema = (
            properties.get("package_name") if isinstance(properties, dict) else None
        )
        if not isinstance(package_schema, dict):
            if function.get("name") == "open_app":
                raise ValueError("open_app_package_schema_missing")
            constrained.append(tool)
            continue
        if packages:
            package_schema["enum"] = packages
            choices = (
                "Choose one installed app (app label -> package): "
                + "; ".join(f"{label} -> {package}" for label, package in candidates)
            )
            description = str(package_schema.get("description") or "").strip()
            package_schema["description"] = " ".join(
                part for part in (description, choices) if part
            )
        else:
            package_schema.pop("enum", None)
        constrained.append(tool)
    return constrained


def _installed_app_candidates(
    installed_apps: dict[str, str],
) -> list[tuple[str, str]]:
    candidates = {
        (str(label).strip(), str(package).strip())
        for label, package in installed_apps.items()
        if str(label).strip() and str(package).strip()
    }
    return sorted(candidates, key=lambda item: (item[0].casefold(), item[1]))


def _state_image_data_uri(state: dict[str, Any]) -> str:
    image = str(state.get("image_base64") or "").strip()
    if image:
        return (
            image
            if image.startswith("data:image/")
            else f"data:image/jpeg;base64,{image}"
        )
    return _image_data_uri(str(state.get("screenshot_path") or ""))


def _image_data_uri(path: str) -> str:
    candidate = Path(str(path or "").strip())
    if not candidate.is_file():
        return ""
    try:
        payload = candidate.read_bytes()
    except OSError:
        return ""
    mime_type = _image_mime_type(payload)
    if not mime_type:
        return ""
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime_type};base64,{encoded}" if encoded else ""


def _image_mime_type(payload: bytes) -> str:
    if payload.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(payload) >= 12 and payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        return "image/webp"
    return ""



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
        self.step_skill_guidance = str(step_skill_guidance).strip()
        self.max_steps = max(1, int(max_steps))
        self._turn_index = 0
        self._metadata: dict[str, Any] = {}
        self._usage = LLMUsageTracker(component="planner", model=self.model)
        self._history: list[dict[str, Any]] = []

    async def one_step_tool_call(
        self,
        goal: str,
        observation: Observation,
        functions: tuple[Function, ...] = (),
        installed_apps: dict[str, str] | None = None,
    ) -> ToolCall:
        state = _planner_state(observation)
        self._turn_index += 1
        request = build_model_turn_request(
            goal=str(goal),
            model=self.model,
            state=state,
            step_skill_guidance=self.step_skill_guidance,
            installed_apps=installed_apps or {},
            functions=functions,
            max_steps=self.max_steps,
            turn_index=self._turn_index,
            system_prompt=self.system_prompt,
            history=self._history,
        )
        self._usage.start_call()
        try:
            response = self._call_model(
                {
                    "goal": str(goal),
                    "model": self.model,
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
        except Exception:
            self._usage.record_failure()
            raise
        self._metadata = metadata
        self._remember_turn(tool_call)
        if self._metadata_sink is not None:
            self._metadata_sink(dict(metadata))
        return tool_call

    def _remember_turn(
        self,
        tool_call: ToolCall,
    ) -> None:
        call_id = f"omniflow_planner_{self._turn_index}"
        self._history.extend(
            (
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": tool_call.name,
                                "arguments": json.dumps(
                                    dict(tool_call.arguments),
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": "Execution result pending.",
                },
            )
        )

    def record_action_result(self, payload: dict[str, Any]) -> None:
        content = "Action result: " + json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for message in reversed(self._history):
            if message.get("role") != "tool":
                continue
            if message.get("content") != "Execution result pending.":
                continue
            message["content"] = content
            return

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


DEFAULT_STEP_GUIDANCE = ""
ORDERING_STEP_GUIDANCE = ""


def resolve_step_guidance(goal: str, explicit: str = "") -> str:
    return str(explicit or "").strip()


_ADAPTER_NAME = "qwen_vl_coordinate_arrays.v1"
_QWEN_VL_MODEL = re.compile(
    r"(?:^|[^a-z0-9])qwen(?:\d+(?:\.\d+)?)?(?:[-_.]?vl|[-_.]?plus)(?:[^a-z0-9]|$)",
    re.IGNORECASE,
)
_COORDINATE_PAIRS = {
    "click": (("x", "y"),),
    "long_press": (("x", "y"),),
    "input_text": (("x", "y"),),
    "swipe": (("x1", "y1"), ("x2", "y2")),
}
_MISSING = object()


def adapt_tool_arguments(
    *,
    tool: str,
    arguments: dict[str, Any],
    requested_model: str,
    resolved_model: str,
    display: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    model = _adapter_model(requested_model, resolved_model)
    coordinate_pairs = _COORDINATE_PAIRS.get(tool)
    if coordinate_pairs is None:
        return dict(arguments), None

    adapted = dict(arguments)
    changes: list[dict[str, Any]] = []
    if model:
        for x_field, y_field in coordinate_pairs:
            _adapt_coordinate_pair(
                adapted,
                x_field,
                y_field,
                changes,
            )
    raw_pixel_changes = _adapt_raw_pixel_coordinates(
        tool=tool,
        arguments=adapted,
        display=display,
    )
    if raw_pixel_changes is not None:
        adapted, conversion = raw_pixel_changes
        changes.append(conversion)
    if not changes:
        return adapted, None
    return adapted, {
        "name": (
            _ADAPTER_NAME
            if all(change.get("source_shape") for change in changes)
            else "planner_coordinate_adapter.v1"
        ),
        "model": model or str(resolved_model or requested_model).strip(),
        "tool": tool,
        "changes": changes,
    }


def _adapt_raw_pixel_coordinates(
    *,
    tool: str,
    arguments: dict[str, Any],
    display: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    pairs = _COORDINATE_PAIRS.get(tool) or ()
    present_pairs = [
        (x_field, y_field)
        for x_field, y_field in pairs
        if x_field in arguments and y_field in arguments
    ]
    if not present_pairs:
        return None
    fields = [field for pair in present_pairs for field in pair]
    values = [arguments[field] for field in fields]
    if not all(_is_number(value) for value in values):
        return None
    if not any(float(value) > 1000 for value in values):
        return None
    width = float((display or {}).get("width") or 0)
    height = float((display or {}).get("height") or 0)
    dimensions = {
        field: width if field.startswith("x") else height for field in fields
    }
    raw_fields = {
        field: arguments[field]
        for field in fields
        if float(arguments[field]) > 1000
    }
    if all(float(arguments[field]) <= dimensions[field] for field in fields):
        raw_fields = {field: arguments[field] for field in fields}
    converted_fields, metadata = screen_pixel_args_to_canonical(
        tool=tool,
        args=raw_fields,
        display=display,
    )
    converted = dict(arguments)
    converted.update(converted_fields)
    return converted, metadata or {}


def _adapter_model(requested_model: str, resolved_model: str) -> str:
    for candidate in (resolved_model, requested_model):
        normalized = str(candidate or "").strip()
        if normalized and _QWEN_VL_MODEL.search(normalized):
            return normalized
    return ""


def _adapt_coordinate_pair(
    arguments: dict[str, Any],
    x_field: str,
    y_field: str,
    changes: list[dict[str, Any]],
) -> None:
    x_value = arguments.get(x_field, _MISSING)
    y_value = arguments.get(y_field, _MISSING)
    if (
        isinstance(x_value, list)
        and len(x_value) == 2
        and all(_is_number(value) for value in x_value)
        and _matches_inferred_y(y_value, x_value[1])
    ):
        arguments[x_field] = x_value[0]
        arguments[y_field] = x_value[1]
        changes.append(
            {
                "source_field": x_field,
                "source_shape": "number_pair",
                "target_fields": [x_field, y_field],
            }
        )
        return

    for field in (x_field, y_field):
        value = arguments.get(field, _MISSING)
        if isinstance(value, list) and len(value) == 1 and _is_number(value[0]):
            arguments[field] = value[0]
            changes.append(
                {
                    "source_field": field,
                    "source_shape": "singleton_number_array",
                    "target_fields": [field],
                }
            )


def _matches_inferred_y(value: Any, inferred_y: int | float) -> bool:
    if value is _MISSING:
        return True
    if _is_number(value):
        return value == inferred_y
    return (
        isinstance(value, list)
        and (
            (
                len(value) == 1
                and _is_number(value[0])
                and value[0] == inferred_y
            )
            or (
                len(value) == 2
                and all(_is_number(item) for item in value)
                and value[1] == inferred_y
            )
        )
    )


def _is_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


__all__ = [
    "DEFAULT_STEP_GUIDANCE",
    "ModelTransport",
    "ORDERING_STEP_GUIDANCE",
    "SYSTEM_PROMPT",
    "VLMPlanner",
    "build_model_turn_request",
    "function_tools",
    "resolve_step_guidance",
]
