"""Shared operation budget and cooperative cancellation, with no planning loop."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
import inspect
import math
from threading import Event
import time
from typing import Any

from omniflow.core.model import Observation, RunResult


class ExecutionStopped(BaseException):
    """Control flow must not be swallowed by tool/model recovery handlers."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass
class ExecutionControl:
    timeout_seconds: float = 600.0
    clock: Callable[[], float] = time.monotonic
    cancelled: Event = field(default_factory=Event)
    actions_executed: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)
    observation: Observation | None = None
    effect_unknown: bool = False

    def __post_init__(self):
        if not math.isfinite(self.timeout_seconds) or not 0 < self.timeout_seconds <= 600:
            raise ValueError("execution_deadline_must_be_within_600_seconds")
        self.deadline = self.clock() + self.timeout_seconds

    @property
    def remaining(self) -> float:
        return max(0.0, self.deadline - self.clock())

    def check(self) -> None:
        if self.cancelled.is_set():
            raise ExecutionStopped("cancelled")
        if self.remaining <= 0:
            raise ExecutionStopped("deadline_exceeded")

    def stopped_result(self, reason: str, *, trace_offset=0, action_offset=0) -> RunResult:
        return RunResult(
            False, actions_executed=self.actions_executed - action_offset,
            error=reason, final_state=self.observation,
            detail={"done_reason": reason, "trace": self.trace[trace_offset:],
                    "effect_unknown": self.effect_unknown,
                    "failed_action_dispatched": None if self.effect_unknown else False},
        )


CURRENT_CONTROL: ContextVar[ExecutionControl | None] = ContextVar("omniflow_execution_control", default=None)


def checkpoint() -> None:
    control = CURRENT_CONTROL.get()
    if control is not None:
        control.check()


def bounded_timeout(timeout: float) -> float:
    control = CURRENT_CONTROL.get()
    if control is None:
        return timeout
    if control.remaining <= 0:
        raise ExecutionStopped("deadline_exceeded")
    return min(float(timeout), control.remaining)


async def invoke(callback: Callable, *args, **kwargs) -> Any:
    """Keep synchronous device/model I/O off the event loop.

    An already-dispatched call is drained before releasing its device owner.
    Cancellation forbids the next action; it cannot undo an in-flight action.
    Actual transport timeouts remain bounded by the shared remaining deadline.
    """
    if inspect.iscoroutinefunction(callback):
        work = callback(*args, **kwargs)
    else:
        work = asyncio.to_thread(callback, *args, **kwargs)
    task = asyncio.ensure_future(work)
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            control = CURRENT_CONTROL.get()
            if control is None or task.cancelled():
                raise
            control.cancelled.set()
    return await result if inspect.isawaitable(result) else result
