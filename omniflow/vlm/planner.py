from __future__ import annotations

import base64
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET

from json_repair import loads as repair_json_loads

from omniflow.core.config import PromptSet
from omniflow.core.model import Function, Observation, ToolCall
from omniflow.core.schemas import canonicalize_action, vlm_action_tools
from omniflow.functions.assets import validate_arguments
from omniflow.vlm.model_config import resolve_openai_compatible_config
from omniflow.vlm.usage import LLMUsageTracker
from omniflow.vlm_coordinates import (
    display_size,
    screen_context_to_pixels,
    screen_pixel_args_to_canonical,
    screen_pixel_tools,
)

SYSTEM_PROMPT = """
You are an Android GUI agent. Complete the user goal from the compact relevant UI
elements and current screenshot. Treat the screenshot as primary evidence for icon
identity and spatial relationships, and XML as evidence for text and control state.
UI elements are grouped by priority; global controls come first, and goal_controls
are actionable visual elements adaptively associated with nearby goal text. The `v`
field is a stable visual reference for an actionable element at its XML bounds.
Return exactly one tool_call each turn. Recalled Function APIs and native GUI
actions are peer tools. Prefer a matching recalled Function API over manually
repeating its GUI actions. When its required arguments are known and it covers the
intended next GUI segment, call that Function API now instead of issuing its first
native action manually. You may call the same Function API multiple times with
different arguments, or call several Function APIs in sequence. A successful
Function result returns control to you and does not mean the overall task is complete.
Never put
action JSON or tool syntax in assistant text. Choose one action, wait for its
result, then inspect the fresh state before choosing another action. Coordinates
are raw pixels in the current original Display coordinate frame, never normalized
0..1000 values. XML bounds use that same raw-pixel frame. A screenshot may be
resized for transport, but its coordinates must still refer to the original
Display. Every tool call
must include a concise summary explaining why that action is the best next step.
When a later step needs facts visible only on the current screen, copy every needed
fact exactly into the summary before navigating away; do not infer, abbreviate, or
drop field values.
Every coordinate is one scalar raw-pixel number, never an array, object, string,
boolean, normalized value, or combined coordinate pair.
Use finished only when current evidence directly proves the goal is complete.
When calling finished, keep content to one short factual sentence describing only
the outcome directly supported by the current screen or previous tool result. Do
not claim that a RunLog or reusable Function was registered; the host reports the
real registration state after execution.
For switches and checkboxes, checked=false means off and checked=true means on.
Never toggle a switch when its checked state already matches the requested goal.
Prefer stable, reusable navigation. When the current app or page provides search,
use search and type the requested text directly before browsing long menus or
swiping. Do not select history, recent, suggestion, or cached-value items when the
requested value can be entered directly. Swipe only when no usable search or input
path exists, or when search results still require browsing.
""".strip()


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
    projection = (
        UIProjection("<omitted>", 0, 0, 0)
        if lightweight_retry
        else project_ui(str(state.get("xml") or ""), goal)
    )
    text = _turn_text(
        goal=goal,
        state=state,
        max_steps=max_steps,
        turn_index=turn_index,
        target_package_name=target_package_name,
        step_skill_guidance=step_skill_guidance,
        validation_error=validation_error,
        rejected_tool_call=rejected_tool_call,
        lightweight_retry=lightweight_retry,
        projection=projection,
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
                    "description": (
                        "Running task memory and next-tool reason. Before leaving "
                        "a screen that contains facts needed later, copy every "
                        "such field and value exactly here; never shorten away "
                        "required facts. Also preserve completed work and why "
                        "this tool is next."
                    ),
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
    max_steps: int,
    turn_index: int,
    target_package_name: str,
    step_skill_guidance: str,
    validation_error: str,
    rejected_tool_call: dict[str, Any] | None,
    lightweight_retry: bool,
    projection: UIProjection,
) -> str:
    display = state.get("display") if isinstance(state.get("display"), dict) else {}
    width, height = display_size(display)
    center_x = int(width / 2)
    center_y = int(height / 2)
    upper_y = int(height * 0.7)
    lower_y = int(height * 0.3)
    lines = [
        f"Goal: {goal}",
        f"Progress: {turn_index}/{max_steps} model turns used",
        f"Current package: {state.get('package_name') or ''}",
        f"Current activity: {state.get('activity_name') or ''}",
        f"Display: {display.get('width') or ''}x{display.get('height') or ''}",
        (
            "Coordinate contract: every tool coordinate is one raw pixel in the "
            f"current original Display frame (X 0..{int(width)}, Y 0..{int(height)}). "
            "Do not output normalized 0..1000 coordinates."
        ),
        (
            'Raw-pixel examples: click {"summary":"Tap center","x":'
            f'{center_x},"y":{center_y}'
            '}; swipe {"summary":"Scroll up","direction":"up","x1":'
            f'{center_x},"y1":{upper_y},"x2":{center_x},"y2":{lower_y}'
            "}."
        ),
    ]
    if target_package_name:
        lines.append(f"Target package: {target_package_name}")
    if step_skill_guidance.strip() and not lightweight_retry:
        lines.extend(("Task guidance:", step_skill_guidance.strip()))
    if validation_error.strip():
        lines.extend(
            (
                "Your previous native tool_call was rejected by the registered schema:",
                validation_error.strip(),
                "Return one corrected native tool_call using the schema exactly. Do not rename, wrap, combine, or infer fields.",
                (
                    "Coordinate fields such as x and y must each be one raw-pixel "
                    f"JSON number in the current Display (X 0..{int(width)}, "
                    f"Y 0..{int(height)}). Never use normalized 0..1000 values, "
                    "[x, y], an object, string, or boolean."
                ),
            )
        )
        if rejected_tool_call:
            lines.extend(
                (
                    "Rejected native tool_call from your immediately previous attempt (verbatim):",
                    json.dumps(
                        rejected_tool_call,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "Do not repeat that argument shape. Return a new tool_call; do not explain or repair it in text.",
                    (
                        'Valid raw-pixel scalar shape: {"x":'
                        f'{center_x},"y":{center_y},"x1":{center_x},'
                        f'"y1":{upper_y},"x2":{center_x},"y2":{lower_y}'
                        '}. Invalid array shape: {"x":[500],"y":[464],"x1":[500,800]}.'
                    ),
                    "If your rejected call placed one intended point in x as [X,Y], choose the scalars yourself and emit x:X and y:Y in the new call. The runtime will not transform the array for you.",
                )
            )
    raw_extra = state.get("extra")
    extra = (
        screen_context_to_pixels(raw_extra, display)
        if isinstance(raw_extra, dict)
        else raw_extra
    )
    if not lightweight_retry and isinstance(extra, dict) and extra:
        context = dict(extra)
        context.pop("installed_apps", None)
        execution_history = str(context.pop("execution_history", "") or "").strip()
        if has_successful_function_action(context):
            lines.append(
                "The previous recalled Function tool call finished all of its "
                "actions successfully. Those actions are already applied. Judge "
                "the complete user goal from the current screenshot and UI state. "
                "Choose finished only if the whole goal is satisfied; otherwise "
                "choose exactly one next tool. Never repeat or toggle the last "
                "successful action merely to verify it, because that can undo the "
                "completed operation."
            )
        if context.get("previous_action_error") or context.get("recent_actions"):
            lines.append(
                "Use the recent action history and error before selecting again. "
                "Do not repeat the same action when it already succeeded or made no "
                "progress; choose a different schema-valid action, finish, or abort."
            )
        if execution_history:
            lines.extend(("Completed tool-call history:", execution_history))
        if context:
            lines.extend(
                (
                    "Recent execution context:",
                    json.dumps(context, ensure_ascii=False, separators=(",", ":")),
                )
            )
    if not lightweight_retry:
        lines.extend(
            (
                f"Relevant UI elements (1-{projection.selected_count}); {projection.candidate_count} candidates:",
                projection.text,
            )
        )
    return "\n".join(lines)


def has_successful_function_action(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    recent_actions = value.get("recent_actions")
    return isinstance(recent_actions, list) and any(
        isinstance(item, dict)
        and item.get("success") is True
        and bool(str(item.get("function_id") or "").strip())
        for item in recent_actions
    )


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
            self._remember_turn(request, tool_call, metadata)
            if self._metadata_sink is not None:
                self._metadata_sink(dict(metadata))
            return tool_call
        raise AssertionError("unreachable")

    def _remember_turn(
        self,
        request: dict[str, Any],
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
                    "role": "user",
                    "content": (
                        "Prior device evidence is summarized by the following "
                        "tool call. Preserve its summary while using the fresh "
                        "state in the next user message."
                    ),
                },
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
                    "content": (
                        "The tool was dispatched. Its execution result and fresh "
                        "device state are provided in the next user message."
                    ),
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


DEFAULT_STEP_GUIDANCE = (
    "Prefer an installed native app over a browser or web search for phone-app "
    "tasks. Launch it with open_app using a package supplied by the runtime. If "
    "the current app is off target, use home, back, or open_app instead of "
    "continuing there. Prefer direct search and exact user-provided text over "
    "browsing long menus, history suggestions, or repeated swipes. Treat "
    "previous_action_error, recent_actions, and execution_history as authoritative: "
    "never repeat an action that already succeeded or made no observable progress. "
    "If a primary button is disabled, satisfy a visible required choice first. "
    "Never authorize payment, enter payment credentials, accept payment-app terms, "
    "enter a password or verification code, or trigger biometric authentication. "
    "When an order reaches payment confirmation, finish with a pending unpaid order "
    "without clicking a payment control."
)

ORDERING_STEP_GUIDANCE = (
    "For ordering tasks, advance through the visible forward path without reopening "
    "or resubmitting a correct search. Choose a semantically compatible product "
    "variant unless the user specified an exact flavor, ingredient, dietary, size, "
    "temperature, sugar, or other required constraint. Select required options before "
    "a disabled primary button, add the requested item once, and default quantity to "
    "one unless specified. Do not add paid extras, memberships, coupons, or unrelated "
    "recommendations. Stop before payment."
)

_ORDERING_TERMS = re.compile(
    r"点外卖|叫外卖|订外卖|点餐|订餐|下单|点一|点杯|点份|咖啡|拿铁|奶茶|"
    r"order(?: me)?|food delivery|takeaway|takeout|coffee|latte|milk tea|burger|pizza",
    re.IGNORECASE,
)


def resolve_step_guidance(goal: str, explicit: str = "") -> str:
    custom = str(explicit or "").strip()
    if custom:
        return custom
    guidance = DEFAULT_STEP_GUIDANCE
    if _ORDERING_TERMS.search(str(goal or "")):
        guidance = f"{guidance}\n\n{ORDERING_STEP_GUIDANCE}"
    return guidance


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


_SEMANTIC_ATTRIBUTES = (
    ("text", "t"),
    ("content-desc", "d"),
    ("hint-text", "h"),
    ("resource-id", "r"),
)
_ACTION_ATTRIBUTES = (
    ("clickable", "click"),
    ("long-clickable", "long"),
    ("editable", "edit"),
    ("scrollable", "scroll"),
    ("checkable", "check"),
    ("focused", "focus"),
)
_ENGLISH_TOKEN = re.compile(r"[a-z0-9]+")
_CHINESE_TOKEN = re.compile(r"[\u4e00-\u9fff]+")
_VISUAL_GOAL_MARKERS = ("广告", "弹窗", "遮挡", "popup", "overlay", "close ad")
_GLOBAL_CONTROL_MARKERS = (
    "back",
    "basket",
    "cart",
    "close",
    "input",
    "menu",
    "more options",
    "navigate up",
    "navigate_up",
    "search",
    "关闭",
    "输入",
    "返回",
    "搜索",
    "更多",
    "查找",
    "菜单",
    "购物车",
)
_GROUP_ORDER = ("global", "goal", "goal_control", "visual", "other")
_GROUP_HEADERS = {
    "global": "[global_controls]",
    "goal": "[goal_matches]",
    "goal_control": "[goal_controls]",
    "visual": "[visual_controls]",
    "other": "[other_context]",
}
_GROUP_LIMITS = {
    "global": 6,
    "goal": 10,
    "goal_control": 8,
    "visual": 4,
    "other": 2,
}


@dataclass(frozen=True)
class _Candidate:
    order: int
    score: int
    goal_match: bool
    group: str
    compact: dict[str, object]
    bounds: tuple[int, int, int, int] | None


@dataclass(frozen=True)
class UIProjection:
    text: str
    candidate_count: int
    selected_count: int
    goal_match_count: int
    visual_context_required: bool = False
    visual_candidate_count: int = 0

    @property
    def requires_screenshot(self) -> bool:
        return True


def project_ui(xml_text: str, goal: str, *, max_nodes: int = 30) -> UIProjection:
    try:
        root = ET.fromstring(str(xml_text or ""))
    except ET.ParseError:
        return UIProjection("<none>", 0, 0, 0)
    goal_terms = _terms(goal)
    candidates: list[_Candidate] = []
    for order, element in enumerate(root.iter()):
        compact: dict[str, object] = {}
        semantic_values: list[str] = []
        visible_semantic_values: list[str] = []
        node_id = str(element.attrib.get("id") or "").strip()
        if node_id:
            compact["i"] = node_id
        for attribute, output_key in _SEMANTIC_ATTRIBUTES:
            value = str(element.attrib.get(attribute) or "").strip()
            if value:
                compact[output_key] = value
                semantic_values.append(value)
                if attribute != "resource-id":
                    visible_semantic_values.append(value)
        actions = [
            output_value
            for attribute, output_value in _ACTION_ATTRIBUTES
            if str(element.attrib.get(attribute) or "").strip().lower() == "true"
        ]
        if not semantic_values and not actions:
            continue
        descendant_context = (
            _descendant_context(element) if actions and not visible_semantic_values else ""
        )
        if descendant_context:
            compact["c"] = descendant_context
        bounds = str(element.attrib.get("bounds") or "").strip()
        parsed_bounds = _parse_bounds(bounds)
        if bounds:
            compact["b"] = bounds
        if actions:
            compact["a"] = actions
        checked = str(element.attrib.get("checked") or "").strip().lower()
        if checked in {"true", "false"}:
            compact["checked"] = checked == "true"
        candidate_terms = _terms(" ".join((*semantic_values, descendant_context)))
        overlap = goal_terms.intersection(candidate_terms)
        goal_match = bool(overlap)
        score = len(overlap) * 1000
        global_control = _is_global_control(semantic_values, actions)
        visual_control = bool(actions and not visible_semantic_values)
        if global_control:
            group = "global"
            score += 5000
        elif goal_match:
            group = "goal"
        elif visual_control:
            group = "visual"
            score += 200 + _visual_specificity(parsed_bounds)
        else:
            group = "other"
        score += 400 if "edit" in actions or "focus" in actions else 0
        score += 50 if actions else 0
        score += min(10, len(semantic_values))
        candidates.append(
            _Candidate(
                order=order,
                score=score,
                goal_match=goal_match,
                group=group,
                compact=compact,
                bounds=parsed_bounds,
            )
        )
    candidates = _promote_goal_controls(candidates)
    selected = _select_candidates(candidates, max_nodes=max_nodes)
    text = _render_candidates(selected)
    return UIProjection(
        text=text or "<none>",
        candidate_count=len(candidates),
        selected_count=len(selected),
        goal_match_count=sum(1 for item in candidates if item.goal_match),
        visual_context_required=any(
            marker in str(goal or "").casefold() for marker in _VISUAL_GOAL_MARKERS
        ),
        visual_candidate_count=sum(
            1 for item in selected if item.group in {"goal_control", "visual"}
        ),
    )


def _promote_goal_controls(candidates: list[_Candidate]) -> list[_Candidate]:
    goal_bounds = [
        item.bounds
        for item in candidates
        if item.group == "goal" and item.bounds is not None
    ]
    if not goal_bounds:
        return candidates
    promoted: list[_Candidate] = []
    for item in candidates:
        if item.group != "visual" or item.bounds is None:
            promoted.append(item)
            continue
        proximity = min(_rectangle_gap(item.bounds, target) for target in goal_bounds)
        if proximity > 180:
            promoted.append(item)
            continue
        promoted.append(
            replace(
                item,
                group="goal_control",
                score=item.score + 2000 - proximity * 5,
            )
        )
    return promoted


def _select_candidates(
    candidates: list[_Candidate],
    *,
    max_nodes: int,
) -> list[_Candidate]:
    if max_nodes <= 0:
        return []
    selected: list[_Candidate] = []
    selected_orders: set[int] = set()
    for group in _GROUP_ORDER:
        remaining = max_nodes - len(selected)
        if remaining <= 0:
            break
        limit = min(_GROUP_LIMITS[group], remaining)
        group_candidates = sorted(
            (item for item in candidates if item.group == group),
            key=_candidate_rank,
        )
        for item in group_candidates[:limit]:
            selected.append(item)
            selected_orders.add(item.order)
    remaining = max_nodes - len(selected)
    if remaining > 0:
        overflow = sorted(
            (item for item in candidates if item.order not in selected_orders),
            key=lambda item: (_GROUP_ORDER.index(item.group), *_candidate_rank(item)),
        )
        selected.extend(overflow[:remaining])
    return sorted(
        selected,
        key=lambda item: (
            _GROUP_ORDER.index(item.group),
            _screen_order(item),
            item.order,
        ),
    )


def _render_candidates(candidates: list[_Candidate]) -> str:
    lines: list[str] = []
    visual_reference = 0
    for group in _GROUP_ORDER:
        group_candidates = [item for item in candidates if item.group == group]
        if not group_candidates:
            continue
        lines.append(_GROUP_HEADERS[group])
        for item in group_candidates:
            compact = dict(item.compact)
            if compact.get("a"):
                visual_reference += 1
                compact = {"v": f"A{visual_reference:02d}", **compact}
            lines.append(
                json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
            )
    return "\n".join(lines)


def _candidate_rank(item: _Candidate) -> tuple[int, tuple[int, int], int]:
    return (-item.score, _screen_order(item), item.order)


def _screen_order(item: _Candidate) -> tuple[int, int]:
    if item.bounds is None:
        return (10**9, 10**9)
    left, top, _right, _bottom = item.bounds
    return (top, left)


def _is_global_control(semantic_values: list[str], actions: list[str]) -> bool:
    if "edit" in actions or "focus" in actions:
        return True
    if not actions:
        return False
    semantic_text = " ".join(semantic_values).casefold()
    return any(marker in semantic_text for marker in _GLOBAL_CONTROL_MARKERS)


def _descendant_context(element: ET.Element) -> str:
    values: list[str] = []
    for descendant in element.iter():
        if descendant is element:
            continue
        for attribute in ("text", "content-desc", "hint-text"):
            value = str(descendant.attrib.get(attribute) or "").strip()
            if value and value not in values:
                values.append(value)
                if len(values) == 2:
                    break
        if len(values) == 2:
            break
    return " | ".join(values)[:80]


def _visual_specificity(bounds: tuple[int, int, int, int] | None) -> int:
    if bounds is None:
        return 0
    left, top, right, bottom = bounds
    width = right - left
    height = bottom - top
    longest = max(width, height)
    score = 500 if longest <= 160 else 300 if longest <= 320 else 100 if longest <= 640 else 0
    if min(width, height) / longest >= 0.7:
        score += 100
    return score


def _rectangle_gap(
    left_bounds: tuple[int, int, int, int],
    right_bounds: tuple[int, int, int, int],
) -> int:
    left_left, left_top, left_right, left_bottom = left_bounds
    right_left, right_top, right_right, right_bottom = right_bounds
    horizontal = max(right_left - left_right, left_left - right_right, 0)
    vertical = max(right_top - left_bottom, left_top - right_bottom, 0)
    return horizontal + vertical


def _parse_bounds(value: str) -> tuple[int, int, int, int] | None:
    numbers = re.fullmatch(
        r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]",
        str(value or "").strip(),
    )
    if numbers is None:
        return None
    left, top, right, bottom = (int(item) for item in numbers.groups())
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _terms(value: str) -> set[str]:
    normalized = str(value or "").casefold()
    terms = {
        token
        for token in _ENGLISH_TOKEN.findall(normalized)
        if len(token) >= 2
    }
    for segment in _CHINESE_TOKEN.findall(normalized):
        if len(segment) == 1:
            terms.add(segment)
        else:
            terms.update(segment[index : index + 2] for index in range(len(segment) - 1))
    return terms


__all__ = [
    "DEFAULT_STEP_GUIDANCE",
    "ModelTransport",
    "ORDERING_STEP_GUIDANCE",
    "SYSTEM_PROMPT",
    "VLMPlanner",
    "build_model_turn_request",
    "function_tools",
    "project_ui",
    "resolve_step_guidance",
]
