from __future__ import annotations

from omniflow.runtime.timing import timed, account_invocation

import asyncio
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from threading import Lock
from typing import Any

from omniflow.catalog import CatalogSnapshot
from omniflow.core.config import Experiment, OmniFlowConfig
from omniflow.core.model import (
    Function, FunctionRouter, Host, InputRequired, Observation, Planner, RunResult, ToolCall,
)
from omniflow.core.schemas import action_from_tool_call
from omniflow.runtime.builtin_harness import run_builtin
from omniflow.functions.artifact import bind_function
from omniflow.functions.recall import RecallResult, recall_functions
from omniflow.functions.store import FunctionStore
from omniflow.runtime.checker import CheckerLibrary
from omniflow.runtime.control import (
    CURRENT_CONTROL,
    ExecutionControl,
    ExecutionStopped,
    checkpoint,
    invoke,
)
from omniflow.runtime.execution import (
    execute_function,
    execute_robust_action,
    record_execution,
)
from omniflow.runtime.protocol import invocation_feedback
from omniflow.transfer.embedding import PageEncoder


class OmniFlow:
    def __init__(
        self,
        store_path: str | Path,
        *,
        host: Host | None = None,
        planner: Planner | None = None,
        function_router: FunctionRouter | None = None,
        completion_checker: Callable[[], Any] | None = None,
        installed_apps: dict[str, str] | None = None,
        config: OmniFlowConfig | None = None,
        catalog: CatalogSnapshot | None = None,
    ):
        self.config = config or OmniFlowConfig()
        self.catalog = catalog
        self.store = FunctionStore(
            store_path,
            seed_functions=(catalog.functions.values() if catalog is not None else ()),
            replace_seeded=catalog is not None,
        )
        # Checkers are runtime-wide recovery policy, not Function-local data.
        # Keep one shared library so every Function sees the same rules and
        # trigger budgets; a Memory package must not carry a private copy.
        self.checker_library = CheckerLibrary.load() if self.config.runtime.checker_enabled else CheckerLibrary()
        self.host = host
        self.planner = planner
        self.function_router = function_router
        self.completion_checker = completion_checker
        self.installed_apps = (
            {
                str(label).strip(): str(package).strip()
                for label, package in installed_apps.items()
                if str(label).strip() and str(package).strip()
            }
            if installed_apps is not None
            else {}
        )
        self.installed_packages = (
            frozenset(self.installed_apps.values())
            if installed_apps is not None
            else None
        )
        set_router_installed_apps = getattr(
            self.function_router,
            "set_installed_apps",
            None,
        )
        if callable(set_router_installed_apps):
            set_router_installed_apps(dict(self.installed_apps))
        self.plugins = self.config.resolved_plugins()
        self._page_encoder: PageEncoder | None = None
        self._operation_lock = Lock()
        self._active_control: ExecutionControl | None = None

    def set_completion_checker(
        self,
        checker: Callable[[], Any] | None,
    ) -> None:
        """Attach the current benchmark task's authoritative completion check."""

        self.completion_checker = checker


    @timed("observe")
    async def _observe(self, *, screenshot: bool) -> Observation:
        checkpoint()
        observation = Observation.from_value(
            await invoke(self.host.observe, xml=True, screenshot=screenshot, app_info=True)
        )
        control = CURRENT_CONTROL.get()
        if control is not None:
            control.observation = observation
        return observation


    async def _ensure_installed_apps(self) -> None:
        """Resolve optional Host inventory under the invocation's cancellation/deadline."""
        inventory = getattr(self.host, "installed_apps", None)
        if self.installed_packages is not None or not callable(inventory):
            return
        apps = await invoke(inventory)
        if apps is None:
            return
        if not isinstance(apps, dict) or any(
            not isinstance(label, str) or not label.strip()
            or not isinstance(package, str) or not package.strip()
            for label, package in apps.items()
        ):
            raise ValueError("host_installed_apps_invalid")
        self.installed_apps = {label.strip(): package.strip() for label, package in apps.items()}
        self.installed_packages = frozenset(self.installed_apps.values())
        setter = getattr(self.function_router, "set_installed_apps", None)
        if callable(setter):
            await invoke(setter, dict(self.installed_apps))

    async def _invoke_tool(
        self,
        tool_call: ToolCall | dict[str, Any],
        *,
        experiment: Experiment | str | None = None,
        checker_trigger_counts: dict[str, int] | None = None,
        function_only: bool = False,
    ) -> RunResult:
        """Execute one invocation and return to its caller, including on failure.

        Only arun owns the task-level Planner loop. Function execution still
        uses the same per-step Checker, Transfer, act, and observe owner.
        """
        call = ToolCall.from_value(tool_call)
        if self.host is None:
            return RunResult(False, function_id=call.name, error="host_not_set")
        function = self.store.get_function(call.name)
        if function_only and function is None:
            return RunResult(False, function_id=call.name, error="function_not_registered")
        await self._ensure_installed_apps()
        observation = await self._observe(screenshot=False)
        counts = checker_trigger_counts if checker_trigger_counts is not None else {}
        if function is not None:
            if not function.agent_visible:
                return RunResult(False, function_id=call.name, error="function_not_agent_visible", final_state=observation)
            try:
                bound = bind_function(function, call.arguments)
            except ValueError as error:
                return RunResult(False, function_id=call.name, error=str(error), final_state=observation,
                                 detail={"failed_action_dispatched": False})
            result = await execute_function(
                bound, host=self.host, plugins=self.plugins, observation=observation,
                installed_packages=self.installed_packages,
                state_loader=self.catalog.get_state if self.catalog is not None else None,
                checker_rules=self.checker_library.rules, checker_trigger_counts=counts,
            )
            result = replace(result, detail={
                **result.detail,
                "done_reason": "function_completed" if result.success else "function_yielded",
                "checker_trigger_counts": dict(counts),
                "function_resolution": {
                    "status": "direct", "selected_function_id": call.name,
                    "store_path": str(self.store.path.resolve()), "arguments": dict(call.arguments),
                    "replay_status": "succeeded" if result.success else "failed",
                },
            })
        else:
            try:
                action = action_from_tool_call(call)
            except ValueError as error:
                return RunResult(False, error=f"tool_not_found:{error}", final_state=observation)
            if action.tool in {"finished", "abort", "info", "get_state"}:
                reason = {"info": "waiting_input", "get_state": "observed"}.get(action.tool, action.tool)
                result = RunResult(action.tool in {"finished", "get_state"}, final_state=observation,
                                   detail={"done_reason": reason, "finished_content": str(action.args.get("content") or "")})
            else:
                step = await execute_robust_action(
                    action, observation=observation, host=self.host, plugins=self.plugins,
                    installed_packages=self.installed_packages,
                    checker_rules=self.checker_library.rules, checker_trigger_counts=counts,
                )
                trace = await record_execution(self.host, step, trace_start_index=0)
                result = RunResult(step.success, actions_executed=step.actions_executed,
                                   error=step.error, final_state=step.after or observation,
                                   detail={"trace": trace, "done_reason": "tool_completed" if step.success else "error"})
        return replace(result, detail={**result.detail, "feedback": invocation_feedback(result)})

    async def arun(
        self,
        goal: str,
        *,
        experiment: Experiment | str | None = None,
    ) -> RunResult:
        return await self._controlled(run_builtin, self, str(goal), experiment=experiment)

    async def acall_tool(self, tool_call, *, experiment=None, checker_trigger_counts=None) -> RunResult:
        return await self._controlled(
            self._invoke_tool, tool_call, experiment=experiment,
            checker_trigger_counts=checker_trigger_counts,
        )

    async def aexecute_function(self, function_id: str, arguments: dict[str, Any], *,
                                checker_trigger_counts=None) -> RunResult:
        """Execute a registered Function; primitives and task control stay with the host."""
        return await self._controlled(self._invoke_tool, ToolCall(function_id, arguments),
                                      function_only=True, checker_trigger_counts=checker_trigger_counts)

    async def arecall(self, goal: str, *, limit: int = 8) -> RunResult:
        """Read current state and recall through the same owner as the built-in Planner."""
        if not str(goal).strip() or not 1 <= limit <= 32:
            raise ValueError("recall_goal_and_limit_required_1_to_32")
        return await self._controlled(self._recall_invocation, str(goal), limit=limit)

    async def _recall_invocation(self, goal: str, *, limit: int) -> RunResult:
        observation = await self._observe(screenshot=False)
        # Empty Memory has a useful empty result, without requiring model weights.
        if not any(f.agent_visible and f.steps for f in self.store.functions.values()):
            recalled = RecallResult((), {"candidate_function_ids": [], "reason": "empty_memory"})
        else:
            recalled = await self._recall(goal, observation=observation, source_states={}, limit=limit)
        return RunResult(True, final_state=observation, detail={
            "done_reason": "functions_recalled", "recall": {
                "functions": [{"function_id": f.id, "name": f.name, "description": f.description,
                               "input_schema": f.input_schema} for f in recalled.functions],
                "audit": recalled.audit,
            },
        })

    def cancel(self) -> bool:
        control = self._active_control
        if control is None:
            return False
        control.cancelled.set()
        return True

    @account_invocation
    async def _controlled(self, callback, *args, **kwargs) -> RunResult:
        if not self._operation_lock.acquire(blocking=False):
            result = RunResult(False, error="execution_busy", detail={"done_reason": "execution_busy"})
            return replace(result, detail={**result.detail, "feedback": invocation_feedback(result)})
        control = CURRENT_CONTROL.get() or ExecutionControl()
        token = CURRENT_CONTROL.set(control)
        self._active_control = control
        trace_offset, action_offset = len(control.trace), control.actions_executed
        try:
            control.check()
            result = await callback(*args, **kwargs)
            control.check()
            if control.effect_unknown:
                raise ExecutionStopped("effect_unknown")
        except ExecutionStopped as error:
            result = control.stopped_result(error.reason, trace_offset=trace_offset, action_offset=action_offset)
        except asyncio.CancelledError:
            control.cancelled.set()
            result = control.stopped_result("cancelled", trace_offset=trace_offset, action_offset=action_offset)
        except Exception as error:  # noqa: BLE001 -- seal failures at the invocation boundary
            result = control.stopped_result(
                "effect_unknown" if control.effect_unknown else "execution_error",
                trace_offset=trace_offset, action_offset=action_offset,
            )
            result = replace(result, error=f"{type(error).__name__}:{error}")
        finally:
            self._active_control = None
            CURRENT_CONTROL.reset(token)
            self._operation_lock.release()
        return replace(result, detail={**result.detail, "feedback": invocation_feedback(result),
            "runtime_policy": {
                "checker_enabled": self.config.runtime.checker_enabled,
                "function_memory_enabled": self.config.runtime.function_memory_enabled,
                "function_reentry_enabled": self.config.runtime.function_reentry_enabled,
            }})

    @timed("recall")
    async def _recall(
        self,
        goal: str,
        *,
        observation: Observation,
        source_states: dict[str, Observation | None],
        limit: int | None = None,
        exclude_function_ids: frozenset[str] = frozenset(),
    ) -> RecallResult:
        for function in self.store.functions.values():
            if not function.steps:
                continue
            source_state_id = function.steps[0].source_state_id
            if source_state_id in source_states:
                continue
            source_state = (
                self.catalog.get_state(source_state_id)
                if self.catalog is not None
                else None
            )
            if source_state is None and self.host is not None:
                get_state = getattr(self.host, "get_state", None)
                if callable(get_state):
                    try:
                        value = await invoke(get_state, source_state_id)
                        source_state = (
                            Observation.from_value(value) if value is not None else None
                        )
                    except Exception:  # noqa: BLE001
                        source_state = None
            source_states[source_state_id] = source_state

        resolved_limit = (
            self.config.runtime.max_function_tools if limit is None else int(limit)
        )
        return await recall_functions(
            str(goal),
            observation=observation,
            functions=self.store.functions,
            source_states=source_states,
            limit=max(0, int(resolved_limit)),
            page_encoder=await invoke(self._get_page_encoder),
            transfer=self.plugins.transfer,
            exclude_function_ids=exclude_function_ids,
        )

    def recall(
        self,
        goal: str,
        *,
        observation: Observation,
        source_states: dict[str, Observation | None],
        limit: int | None = None,
    ) -> list[Function]:
        """Synchronously inspect page-aware recall with explicit source states."""

        resolved_limit = (
            self.config.runtime.max_function_tools if limit is None else int(limit)
        )
        return list(
            asyncio.run(
                recall_functions(
                    str(goal),
                    observation=Observation.from_value(observation),
                    functions=self.store.functions,
                    source_states=source_states,
                    limit=max(0, int(resolved_limit)),
                    page_encoder=self._get_page_encoder(),
                    transfer=self.plugins.transfer,
                )
            ).functions
        )

    def _get_page_encoder(self) -> PageEncoder:
        if self._page_encoder is None:
            self._page_encoder = PageEncoder()
        return self._page_encoder

    def call_tool(
        self,
        tool_call: ToolCall | dict[str, Any],
        *,
        experiment: Experiment | str | None = None,
        checker_trigger_counts: dict[str, int] | None = None,
    ) -> RunResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.acall_tool(
                    tool_call,
                    experiment=experiment,
                    checker_trigger_counts=checker_trigger_counts,
                )
            )
        raise RuntimeError(
            "OmniFlow.call_tool cannot run inside an event loop; await acall_tool"
        )

    def run(
        self,
        goal: str,
        *,
        experiment: Experiment | str | None = None,
    ) -> RunResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(str(goal), experiment=experiment))
        raise RuntimeError("OmniFlow.run cannot run inside an event loop; await arun")
