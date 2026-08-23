from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol


@dataclass(frozen=True)
class Observation:
    xml: str | None = None
    package_name: str | None = None
    activity_name: str | None = None
    image_base64: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any) -> "Observation":
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            extra = dict(value.get("extra") or {})
            for key in ("state_id", "screenshot_path"):
                if value.get(key) is not None:
                    extra[key] = value[key]
            display = value.get("display")
            if isinstance(display, dict):
                extra["display"] = dict(display)
            return cls(
                xml=value.get("xml"),
                package_name=value.get("package_name"),
                activity_name=value.get("activity_name"),
                image_base64=value.get("image_base64"),
                extra=extra,
            )
        return cls(
            xml=getattr(value, "xml", None),
            package_name=getattr(value, "package_name", None),
            activity_name=getattr(value, "activity_name", None),
            image_base64=getattr(value, "image_base64", None),
            extra=dict(getattr(value, "extra", {}) or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "xml": self.xml,
            "package_name": self.package_name,
            "activity_name": self.activity_name,
            "image_base64": self.image_base64,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class Action:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any) -> "Action":
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError("action_must_be_object")
        if set(value) != {"tool", "args"}:
            raise ValueError("action_contract_invalid")
        tool = value.get("tool")
        args = value.get("args")
        if not isinstance(tool, str) or not tool.strip():
            raise ValueError("action_tool_required")
        if not isinstance(args, dict):
            raise ValueError("action_args_must_be_object")
        return cls(tool.strip(), dict(args))

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "args": dict(self.args)}


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any) -> "ToolCall":
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict) or set(value) != {"name", "arguments"}:
            raise ValueError("tool_call_contract_invalid")
        name = value.get("name")
        arguments = value.get("arguments")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("tool_call_name_required")
        if not isinstance(arguments, dict):
            raise ValueError("tool_call_arguments_must_be_object")
        return cls(name.strip(), dict(arguments))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": dict(self.arguments)}


@dataclass(frozen=True)
class ActionResult:
    success: bool
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any) -> "ActionResult":
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls(value)
        if isinstance(value, dict):
            return cls(
                bool(value.get("success")),
                value.get("error") or value.get("error_message"),
                dict(value.get("extra") or {}),
            )
        return cls(
            bool(getattr(value, "success", False)),
            getattr(value, "error", None),
            dict(getattr(value, "extra", {}) or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "error": self.error,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class FunctionStep:
    step_index: int
    action: Action
    source_state_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "source_state_id": self.source_state_id,
            "action": self.action.to_dict(),
        }


@dataclass(frozen=True)
class Function:
    function_id: str
    name: str
    description: str
    steps: tuple[FunctionStep, ...]
    schema_version: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    bindings: tuple[dict[str, str], ...] = ()
    checker_rules: tuple[dict[str, Any], ...] = ()
    agent_visible: bool = True

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Function":
        return cls(
            function_id=str(value.get("function_id") or ""),
            name=str(value.get("name") or ""),
            description=str(value.get("description") or ""),
            steps=tuple(
                FunctionStep(
                    step_index=int(step.get("step_index") or 0),
                    source_state_id=str(step.get("source_state_id") or ""),
                    action=Action.from_value(step.get("action") or {}),
                )
                for step in value.get("steps") or ()
                if isinstance(step, dict)
            ),
            schema_version=str(value.get("schema_version") or ""),
            input_schema=dict(value.get("input_schema") or {}),
            bindings=tuple(
                {
                    "source": str(binding.get("source") or ""),
                    "target": str(binding.get("target") or ""),
                }
                for binding in value.get("bindings") or ()
                if isinstance(binding, dict)
            ),
            checker_rules=tuple(value.get("checker_rules") or ()),
            agent_visible=value.get("agent_visible") is True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "function_id": self.function_id,
            "name": self.name,
            "description": self.description,
            "input_schema": dict(self.input_schema),
            "bindings": [dict(binding) for binding in self.bindings],
            "steps": [step.to_dict() for step in self.steps],
            "checker_rules": list(self.checker_rules),
            "agent_visible": self.agent_visible,
        }

    @property
    def id(self) -> str:
        return self.function_id

    @property
    def actions(self) -> tuple[Action, ...]:
        return tuple(step.action for step in self.steps)


@dataclass(frozen=True)
class RunResult:
    success: bool
    function_id: str | None = None
    actions_executed: int = 0
    model_calls: int = 0
    fallback_steps: int = 0
    error: str | None = None
    final_state: Observation | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def execution_summary(self) -> dict[str, Any]:
        usage = self.detail.get("llm_usage")
        usage = usage if isinstance(usage, dict) else {}
        prompt_tokens = max(0, _coerce_int(usage.get("prompt_tokens")))
        completion_tokens = max(0, _coerce_int(usage.get("completion_tokens")))
        total_tokens = max(0, _coerce_int(usage.get("total_tokens")))
        if total_tokens <= 0:
            total_tokens = prompt_tokens + completion_tokens
        return {
            "success": self.success,
            "steps": self.actions_executed,
            "model_calls": self.model_calls,
            "fallback_steps": self.fallback_steps,
            "completion_review_calls": max(
                0,
                _coerce_int(self.detail.get("completion_review_calls")),
            ),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "tokens": total_tokens,
            "token_usage_status": str(
                usage.get("token_usage_status") or "not_applicable"
            ),
            "failure_reason": self.error,
        }


@dataclass(frozen=True)
class ActionDecision:
    kind: str
    action: Action | None = None
    reason: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepResult:
    success: bool
    action: Action | None = None
    before: Observation | None = None
    after: Observation | None = None
    result: ActionResult | None = None
    actions_executed: int = 0
    error: str | None = None
    origin: str = "action"
    executed_steps: tuple["StepResult", ...] = ()
    function_id: str | None = None
    checker_trigger: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckerContext:
    source: Observation | None
    current: Observation
    action: Action


@dataclass(frozen=True)
class TransferResult:
    action: Action | None
    reason: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


Checker = Callable[
    [CheckerContext],
    Action | None | Awaitable[Action | None],
]
Transfer = Callable[
    [Action, Observation, Observation | None],
    TransferResult | Awaitable[TransferResult],
]


class Host(Protocol):
    def observe(self, **kwargs: Any) -> Observation | Awaitable[Observation]: ...

    def act(self, action: Action) -> ActionResult | Awaitable[ActionResult]: ...

    def get_state(
        self, source_state_id: str
    ) -> Observation | Awaitable[Observation]: ...


class Planner(Protocol):
    def one_step_tool_call(
        self,
        goal: str,
        observation: Observation,
        functions: tuple[Function, ...],
        installed_apps: dict[str, str],
    ) -> ToolCall | Awaitable[ToolCall]: ...


class FunctionRouter(Protocol):
    def route_function(
        self,
        goal: str,
        functions: tuple[Function, ...],
    ) -> ToolCall | None | Awaitable[ToolCall | None]: ...


def _coerce_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
