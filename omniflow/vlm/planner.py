from __future__ import annotations

import base64
from collections.abc import Callable
from copy import deepcopy
import json
import math
from pathlib import Path
import re
from typing import Any

from json_repair import loads as repair_json_loads

from omniflow.core.config import DEFAULT_PLANNER_SYSTEM_PROMPT, PromptSet
from omniflow.core.model import Function, Observation, ToolCall
from omniflow.core.schemas import canonicalize_action, vlm_action_tools
from omniflow.functions.assets import validate_arguments
from omniflow.vlm.model_config import resolve_openai_compatible_config
from omniflow.vlm.usage import LLMUsageTracker
from omniflow.vlm_coordinates import (
    screen_pixel_args_to_canonical,
    screen_pixel_tools,
)

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
    target_package_name: str = "",
    step_skill_guidance: str = "",
    installed_apps: dict[str, str] | None = None,
    functions: list[Function] | tuple[Function, ...] = (),
    previous_screenshot_path: str = "",
    validation_error: str = "",
    retry_tool_name: str = "",
    rejected_tool_call: dict[str, Any] | None = None,
    lightweight_retry: bool = False,
    system_prompt: str = SYSTEM_PROMPT,
    history: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    text = _turn_text(
        goal=goal,
        state=state,
        target_package_name=target_package_name,
        step_skill_guidance=step_skill_guidance,
        validation_error=validation_error,
        rejected_tool_call=rejected_tool_call,
        lightweight_retry=lightweight_retry,
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    include_images = not validation_error.strip()
    current_image = _state_image_data_uri(state) if include_images else ""
    if current_image:
        content.append({"type": "image_url", "image_url": {"url": current_image}})
    display = state.get("display") if isinstance(state.get("display"), dict) else None
    tools = function_tools(functions, include_summary=True)
    tools.extend(screen_pixel_tools(
        vlm_action_tools(include_summary=True),
        display,
    ))
    tools = constrain_open_app_tool(tools, installed_apps or {})
    if retry_tool_name:
        tools = [
            tool
            for tool in tools
            if tool.get("function", {}).get("name") == retry_tool_name
        ]
        if len(tools) != 1:
            raise ValueError(f"model_turn_retry_tool_not_visible:{retry_tool_name}")
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
        arguments, arguments_repaired = load_tool_arguments(raw_arguments)
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
    summary = str(arguments.pop("summary", "") or "").strip()
    if not summary:
        raise ModelToolCallError(
            "model_turn_summary_required",
            tool_name=tool,
            arguments=rejected_arguments,
        )
    resolved_model = str(value.get("resolved_model") or requested_model).strip()
    adapter_metadata = None
    coordinate_metadata = None
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
            arguments, coordinate_metadata = screen_pixel_args_to_canonical(
                tool=tool,
                args=arguments,
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
    metadata: dict[str, Any] = {"summary": summary}
    if arguments_repaired:
        metadata["json_repair"] = {
            "name": "json_repair",
            "applied": True,
        }
    if adapter_metadata is not None:
        metadata["model_adapter"] = adapter_metadata
    if coordinate_metadata is not None:
        metadata["coordinate_conversion"] = coordinate_metadata
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
    *,
    include_summary: bool,
) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for function in functions:
        parameters = deepcopy(function.input_schema)
        properties = parameters.setdefault("properties", {})
        required = list(parameters.get("required") or ())
        if include_summary:
            properties = {
                "summary": {
                    "type": "string",
                    "description": "Brief plan and reason for the next action.",
                },
                **properties,
            }
            parameters["properties"] = properties
            required = ["summary", *required]
        parameters["required"] = required
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
    target_package_name: str,
    step_skill_guidance: str,
    validation_error: str,
    rejected_tool_call: dict[str, Any] | None,
    lightweight_retry: bool,
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
    if target_package_name:
        lines.append(f"Target package: {target_package_name}")
    if step_skill_guidance.strip() and not lightweight_retry:
        lines.append(f"Task guidance: {step_skill_guidance.strip()}")
    if validation_error.strip():
        lines.append(f"Previous tool call was invalid: {validation_error.strip()}")
        if rejected_tool_call:
            lines.append(
                "Rejected call: "
                + json.dumps(
                    rejected_tool_call,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
    raw_extra = state.get("extra")
    if not lightweight_retry and isinstance(raw_extra, dict):
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
    packages = list(
        dict.fromkeys(
            package for _label, package in _installed_app_candidates(installed_apps)
        )
    )
    constrained: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict) or function.get("name") != "open_app":
            constrained.append(tool)
            continue
        if not packages:
            continue
        parameters = function.get("parameters")
        properties = (
            parameters.get("properties") if isinstance(parameters, dict) else None
        )
        package_schema = (
            properties.get("package_name") if isinstance(properties, dict) else None
        )
        if not isinstance(package_schema, dict):
            raise ValueError("open_app_package_schema_missing")
        package_schema["enum"] = packages
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
        self._history: list[dict[str, Any]] = []

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
                history=self._history,
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
            self._remember_turn(tool_call, metadata)
            if self._metadata_sink is not None:
                self._metadata_sink(dict(metadata))
            return tool_call
        raise AssertionError("unreachable")

    def _remember_turn(
        self,
        tool_call: ToolCall,
        metadata: dict[str, Any],
    ) -> None:
        call_id = f"omniflow_planner_{self._turn_index}"
        arguments = {
            "summary": str(metadata.get("summary") or "").strip(),
            **dict(tool_call.arguments),
        }
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
                                    arguments,
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
                    "content": "Action dispatched; inspect the current screenshot.",
                },
            )
        )

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
    r"(?:^|[^a-z0-9])qwen(?:\d+(?:\.\d+)?)?[-_.]?vl(?:[^a-z0-9]|$)",
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
    if not model or coordinate_pairs is None:
        return dict(arguments), None

    adapted = dict(arguments)
    changes: list[dict[str, Any]] = []
    display_width = _positive_number((display or {}).get("width"))
    display_height = _positive_number((display or {}).get("height"))
    for x_field, y_field in coordinate_pairs:
        _adapt_coordinate_pair(
            adapted,
            x_field,
            y_field,
            changes,
            display_width=display_width,
            display_height=display_height,
        )
    if not changes:
        return adapted, None
    return adapted, {
        "name": _ADAPTER_NAME,
        "model": model,
        "tool": tool,
        "changes": changes,
    }


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
    *,
    display_width: float | None,
    display_height: float | None,
) -> None:
    x_value = arguments.get(x_field, _MISSING)
    y_value = arguments.get(y_field, _MISSING)
    if (
        display_width is not None
        and display_height is not None
        and _is_number(x_value)
        and _is_number(y_value)
        and 0 <= float(x_value) <= 1000
        and 0 <= float(y_value) <= 1000
        and (
            float(x_value) > display_width
            or float(y_value) > display_height
        )
    ):
        arguments[x_field] = float(x_value) / 1000.0 * display_width
        arguments[y_field] = float(y_value) / 1000.0 * display_height
        changes.append(
            {
                "source_fields": [x_field, y_field],
                "source_shape": "normalized_0_1000_scalar_pair",
                "target_fields": [x_field, y_field],
            }
        )
        return
    if (
        display_width is not None
        and display_height is not None
        and isinstance(x_value, list)
        and len(x_value) == 2
        and all(_is_number(value) for value in x_value)
        and isinstance(y_value, list)
        and len(y_value) == 1
        and _is_number(y_value[0])
        and 0 <= float(x_value[0]) <= display_width
        and 0 <= float(x_value[1]) <= display_height
        and 0 <= float(y_value[0]) <= display_height
        and (float(x_value[1]) > 1000 or float(y_value[0]) > 1000)
    ):
        arguments[x_field] = x_value[0]
        arguments[y_field] = x_value[1]
        changes.append(
            {
                "source_field": x_field,
                "source_shape": "pixel_point_with_trailing_y",
                "target_fields": [x_field, y_field],
            }
        )
        return
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
        and len(value) == 1
        and _is_number(value[0])
        and value[0] == inferred_y
    )


def _is_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _positive_number(value: Any) -> float | None:
    return float(value) if _is_number(value) and float(value) > 0 else None


def load_tool_arguments(raw_arguments: str) -> tuple[Any, bool]:
    """Parse tool arguments, repairing only structurally truncated JSON."""
    try:
        return json.loads(raw_arguments), False
    except json.JSONDecodeError as original_error:
        if _ends_inside_json_string(raw_arguments):
            raise original_error
        try:
            return (
                repair_json_loads(raw_arguments, skip_json_loads=True),
                True,
            )
        except (TypeError, ValueError, RecursionError):
            raise original_error


def _ends_inside_json_string(value: str) -> bool:
    in_string = False
    escaped = False
    for character in value:
        if not in_string:
            if character == '"':
                in_string = True
            continue
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == '"':
            in_string = False
    return in_string


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
