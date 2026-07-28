from __future__ import annotations

import json
import os
from typing import Any

from omniflow.core.model import Observation
from omniflow.vlm.usage import LLMUsageTracker

CONFIRM_COMPLETION_TOOL = "confirm_goal_finished"
REJECT_COMPLETION_TOOL = "reject_goal_finished"

COMPLETION_CHECKER_SYSTEM_PROMPT = (
    "Check whether the final screenshot directly proves that the user's complete "
    "goal is satisfied. Use the action summary only as context; successful action "
    "execution alone is not proof. Return exactly one provided native tool call."
)

_EMPTY_PARAMETERS = {
    "type": "object",
    "properties": {},
    "required": [],
    "additionalProperties": False,
}


class VLMCompletionChecker:
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
                "VLMCompletionChecker supports OpenAI-compatible providers only"
            )
        self.model = str(model).strip()
        self.timeout = float(timeout)
        self._client = client
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self._usage = LLMUsageTracker(
            component="completion_checker",
            model=self.model,
        )

    async def check_completion(
        self,
        goal: str,
        observation: Observation,
        action_summary: str,
    ) -> bool:
        image = str(observation.image_base64 or "").strip()
        if not image:
            return False
        image_url = image if image.startswith("data:image/") else (
            f"data:image/png;base64,{image}"
        )
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "goal": str(goal).strip(),
                        "action_summary": str(action_summary).strip(),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": CONFIRM_COMPLETION_TOOL,
                    "description": (
                        "The final screenshot directly proves the complete goal "
                        "is satisfied."
                    ),
                    "strict": True,
                    "parameters": dict(_EMPTY_PARAMETERS),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": REJECT_COMPLETION_TOOL,
                    "description": (
                        "The final screenshot does not directly prove the complete "
                        "goal is satisfied."
                    ),
                    "strict": True,
                    "parameters": dict(_EMPTY_PARAMETERS),
                },
            },
        ]
        client = self._client or self._build_client()
        self._usage.start_call()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": COMPLETION_CHECKER_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
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
        tool_calls = list(getattr(message, "tool_calls", None) or ())
        if len(tool_calls) != 1:
            raise ValueError(
                "completion_checker_tool_call_contract_violation:"
                f"expected_one:got_{len(tool_calls)}"
            )
        call = tool_calls[0].function
        tool_name = str(call.name or "").strip()
        try:
            arguments = json.loads(str(call.arguments or "{}"))
        except json.JSONDecodeError as error:
            raise ValueError("completion_checker_arguments_must_be_json") from error
        if arguments != {}:
            raise ValueError("completion_checker_arguments_must_be_empty")
        if tool_name == CONFIRM_COMPLETION_TOOL:
            return True
        if tool_name == REJECT_COMPLETION_TOOL:
            return False
        raise ValueError(f"completion_checker_tool_not_visible:{tool_name}")

    def take_usage(self) -> dict[str, Any]:
        return self._usage.take_usage()

    def _build_client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install omniflow[llm] to use VLMCompletionChecker"
            ) from exc
        options: dict[str, Any] = {"api_key": self._api_key or "not-required"}
        if self._base_url:
            options["base_url"] = self._base_url
        return OpenAI(**options)


__all__ = [
    "COMPLETION_CHECKER_SYSTEM_PROMPT",
    "CONFIRM_COMPLETION_TOOL",
    "REJECT_COMPLETION_TOOL",
    "VLMCompletionChecker",
]
