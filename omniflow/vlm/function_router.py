from __future__ import annotations

from copy import deepcopy
import json
import os
from typing import Any

from omniflow.core.model import Function, ToolCall
from omniflow.functions.artifact import validate_arguments
from omniflow.vlm.usage import LLMUsageTracker

REJECT_FUNCTION_TOOL = "reject_recalled_function"

FUNCTION_ROUTER_SYSTEM_PROMPT = (
    "Choose one recalled GUI Function only when its name, description, fixed choices, "
    "and parameter schema fully cover the user's complete goal. Fill arguments only "
    "from explicit, unambiguous goal values and follow each parameter description "
    "exactly. Otherwise call reject_recalled_function. Return exactly one native "
    "tool call."
)

_PACKAGE_ARGUMENT_PROMPT = (
    " When package_name is required, copy the exact package from installed_apps whose "
    "label matches the requested app; reject if the mapping is ambiguous."
)


class VLMFunctionRouter:
    def __init__(
        self,
        *,
        model: str,
        provider: str = "openai",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        client: Any | None = None,
    ):
        if provider not in {"openai", "openai_compatible"}:
            raise ValueError(
                "VLMFunctionRouter supports OpenAI-compatible providers only"
            )
        self.model = str(model).strip()
        self.timeout = float(timeout)
        self._client = client
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self._installed_apps: dict[str, str] = {}
        self._usage = LLMUsageTracker(
            component="function_router",
            model=self.model,
        )

    def set_installed_apps(self, installed_apps: dict[str, str]) -> None:
        self._installed_apps = {
            str(label).strip(): str(package).strip()
            for label, package in installed_apps.items()
            if str(label).strip() and str(package).strip()
        }

    async def route_function(
        self,
        goal: str,
        functions: tuple[Function, ...],
    ) -> ToolCall | None:
        function_catalog = {
            function.id: function for function in functions if function.agent_visible
        }
        if not function_catalog:
            return None
        if REJECT_FUNCTION_TOOL in function_catalog:
            raise ValueError("function_router_reject_tool_name_reserved")

        tools = [
            {
                "type": "function",
                "function": {
                    "name": function.id,
                    "description": _router_function_description(function),
                    "strict": True,
                    "parameters": deepcopy(function.input_schema),
                },
            }
            for function in sorted(function_catalog.values(), key=lambda item: item.id)
        ]
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": REJECT_FUNCTION_TOOL,
                    "description": (
                        "Reject all recalled Functions because none fully covers "
                        "the complete user goal or required arguments are missing."
                    ),
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            }
        )
        package_argument_required = any(
            "package_name"
            in (
                function.input_schema.get("properties")
                if isinstance(function.input_schema.get("properties"), dict)
                else {}
            )
            for function in function_catalog.values()
        )
        user_context: dict[str, Any] = {"goal": str(goal).strip()}
        system_prompt = FUNCTION_ROUTER_SYSTEM_PROMPT
        if package_argument_required:
            system_prompt += _PACKAGE_ARGUMENT_PROMPT
            user_context["installed_apps"] = dict(self._installed_apps)
        client = self._client or self._build_client()
        self._usage.start_call()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(
                            user_context,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                ],
                tools=tools,
                tool_choice="required",
                parallel_tool_calls=False,
                temperature=0,
                timeout=self.timeout,
            )
        except Exception:
            self._usage.record_failure()
            raise
        self._usage.record_response(response)

        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or ()
        if len(tool_calls) != 1:
            raise ValueError(
                "function_router_tool_call_contract_violation:"
                f"expected_one:got_{len(tool_calls)}"
            )
        call = tool_calls[0].function
        tool_name = str(call.name or "").strip()
        try:
            arguments = json.loads(str(call.arguments or "{}"))
        except json.JSONDecodeError as error:
            raise ValueError("function_router_arguments_must_be_json") from error
        if not isinstance(arguments, dict):
            raise ValueError("function_router_arguments_must_be_object")
        if tool_name == REJECT_FUNCTION_TOOL:
            validate_arguments(tools[-1]["function"]["parameters"], arguments)
            return None
        function = function_catalog.get(tool_name)
        if function is None:
            raise ValueError(f"function_router_tool_not_visible:{tool_name}")
        validate_arguments(function.input_schema, arguments)
        return ToolCall(tool_name, arguments)

    def take_usage(self) -> dict[str, Any]:
        return self._usage.take_usage()

    def _build_client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install omniflow[llm] to use VLMFunctionRouter"
            ) from exc
        options: dict[str, Any] = {
            "api_key": self._api_key or "not-required",
            "max_retries": 0,
        }
        if self._base_url:
            options["base_url"] = self._base_url
        return OpenAI(**options)


def _router_function_description(function: Function) -> str:
    description = " ".join(str(function.description or "").split())
    return f"{function.name}: {description}"


__all__ = [
    "FUNCTION_ROUTER_SYSTEM_PROMPT",
    "REJECT_FUNCTION_TOOL",
    "VLMFunctionRouter",
]
