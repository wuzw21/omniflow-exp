from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any

from omniflow.core.config import Experiment, OmniFlowConfig
from omniflow.core.model import (
    Action,
    CompletionChecker,
    Function,
    FunctionRouter,
    Host,
    Observation,
    Planner,
    RunResult,
    ToolCall,
)
from omniflow.core.schemas import canonicalize_action
from omniflow.functions.artifact import bind_function
from omniflow.functions.recall import recall_functions
from omniflow.functions.store import FunctionStore
from omniflow.runtime.execution import (
    align_function_resume,
    execute_action,
    execute_function,
    record_execution,
)
from omniflow.vlm.usage import merge_usage, token_usage_status


class InputRequired(RuntimeError):
    def __init__(self, question: str):
        self.question = str(question).strip()
        super().__init__(self.question or "input_required")


class OmniFlow:
    def __init__(
        self,
        store_path: str | Path,
        *,
        host: Host | None = None,
        planner: Planner | None = None,
        function_router: FunctionRouter | None = None,
        completion_checker: CompletionChecker | None = None,
        installed_apps: dict[str, str] | None = None,
        config: OmniFlowConfig | None = None,
    ):
        self.config = config or OmniFlowConfig()
        self.store = FunctionStore(store_path)
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
        self.plugins = self.config.resolved_plugins()

    async def _execute(
        self,
        goal: str,
        direct_tool_call: ToolCall | None,
        *,
        experiment: Experiment | str | None = None,
    ) -> RunResult:
        goal = str(goal).strip()
        if self.host is None:
            return RunResult(
                False,
                function_id=direct_tool_call.name if direct_tool_call else None,
                error="host_not_set",
            )
        profile = _experiment(experiment)
        actions_executed = 0
        model_calls = 0
        fallback_steps = 0
        trace: list[dict[str, Any]] = []
        last_error = "tool_not_selected"
        llm_usage: dict[str, Any] = {}
        failed_function_id: str | None = None
        replayed_function_id: str | None = None
        bound_function: Function | None = None
        failed_step_index: int | None = None
        fallback_observations: list[Observation] = []
        completed_function: Function | None = None
        observation = await self._observe(screenshot=False)
        planner_functions = tuple(self.recall(goal))
        planner_function_catalog = {
            function.id: function for function in planner_functions
        }

        routed_tool_call: ToolCall | None = None
        if (
            direct_tool_call is None
            and self.function_router is not None
            and planner_functions
        ):
            try:
                routed_value = await _await(
                    self.function_router.route_function(goal, planner_functions)
                )
                routed_tool_call = (
                    ToolCall.from_value(routed_value)
                    if routed_value is not None
                    else None
                )
            except Exception:  # noqa: BLE001
                routed_tool_call = None
            router_usage = _take_llm_usage(self.function_router)
            merge_usage(llm_usage, router_usage, component="function_router")
            model_calls += _usage_model_calls(router_usage, fallback=1)
            planner_functions = ()

        selected_function: Function | None = None
        resolved_arguments: dict[str, Any] = {}
        selected_tool_call = direct_tool_call or routed_tool_call
        if selected_tool_call is not None:
            selected_function = planner_function_catalog.get(selected_tool_call.name)
            if direct_tool_call is not None:
                selected_function = self.store.get_function(selected_tool_call.name)
            resolved_arguments = dict(selected_tool_call.arguments)

        if selected_function is not None:
            replayed_function_id = selected_function.id
            try:
                bound_function = bind_function(selected_function, resolved_arguments)
            except ValueError as error:
                replay = RunResult(
                    False,
                    function_id=selected_function.id,
                    error=str(error),
                    final_state=observation,
                )
            else:
                replay = await execute_function(
                    bound_function,
                    host=self.host,
                    plugins=self.plugins,
                    observation=observation,
                    max_actions=self.config.runtime.max_steps,
                    installed_packages=self.installed_packages,
                )
            actions_executed += replay.actions_executed
            trace.extend(replay.detail.get("trace") or ())
            if replay.success:
                observation = replay.final_state or observation
                last_error = "function_replay_completed_e2e_unverified"
                if bound_function is not None:
                    completed_function = bound_function
            else:
                failed_function_id = selected_function.id
            observation = replay.final_state or observation
            if not replay.success:
                last_error = replay.error or "function_replay_failed"
                failed_step_index = _optional_step_index(
                    replay.detail.get("failed_step_index")
                )
                if bound_function is not None and failed_step_index is not None:
                    fallback_observations = [observation]

            if direct_tool_call is not None:
                return self._result(
                    replay.success,
                    profile=profile,
                    trace=trace,
                    function_id=direct_tool_call.name,
                    actions_executed=actions_executed,
                    model_calls=model_calls,
                    llm_usage=llm_usage,
                    error=None if replay.success else last_error,
                    final_state=observation,
                    terminal_detail={
                        "done_reason": (
                            "function_completed" if replay.success else "error"
                        )
                    },
                )

            if replay.success and self.completion_checker is not None:
                observation = await self._observe(
                    screenshot=True,
                    xml=False,
                    app_info=False,
                )
                try:
                    completion_confirmed = bool(
                        await _await(
                            self.completion_checker.check_completion(
                                goal,
                                observation,
                                _function_completion_summary(
                                    selected_function,
                                    trace,
                                ),
                            )
                        )
                    )
                except Exception:  # noqa: BLE001
                    completion_confirmed = False
                checker_usage = _take_llm_usage(self.completion_checker)
                merge_usage(
                    llm_usage,
                    checker_usage,
                    component="completion_checker",
                )
                model_calls += _usage_model_calls(checker_usage, fallback=1)
                if completion_confirmed:
                    return self._result(
                        True,
                        profile=profile,
                        trace=trace,
                        function_id=selected_function.id,
                        actions_executed=actions_executed,
                        model_calls=model_calls,
                        llm_usage=llm_usage,
                        final_state=observation,
                        terminal_detail={
                            "done_reason": "function_completion_confirmed"
                        },
                    )

        if direct_tool_call is not None:
            try:
                direct_action = _action_from_tool_call(direct_tool_call)
            except ValueError as error:
                return self._result(
                    False,
                    profile=profile,
                    trace=trace,
                    actions_executed=actions_executed,
                    model_calls=model_calls,
                    llm_usage=llm_usage,
                    error=f"tool_not_found:{error}",
                    final_state=observation,
                )
            if direct_action.tool == "finished":
                return self._result(
                    True,
                    profile=profile,
                    trace=trace,
                    actions_executed=actions_executed,
                    model_calls=model_calls,
                    llm_usage=llm_usage,
                    final_state=observation,
                    terminal_detail={
                        "done_reason": "finished",
                        "finished_content": str(
                            direct_action.args.get("content") or ""
                        ),
                    },
                )
            if direct_action.tool in {"abort", "info", "get_state"}:
                return self._result(
                    False,
                    profile=profile,
                    trace=trace,
                    actions_executed=actions_executed,
                    model_calls=model_calls,
                    llm_usage=llm_usage,
                    error=f"tool_not_directly_invokable:{direct_action.tool}",
                    final_state=observation,
                )
            step = await execute_action(
                direct_action,
                observation=observation,
                host=self.host,
                plugins=self.plugins,
                installed_packages=self.installed_packages,
            )
            trace.extend(
                await record_execution(
                    self.host,
                    step,
                    trace_start_index=0,
                )
            )
            return self._result(
                step.success,
                profile=profile,
                trace=trace,
                actions_executed=step.actions_executed,
                model_calls=model_calls,
                llm_usage=llm_usage,
                error=None if step.success else step.error,
                final_state=step.after or observation,
                terminal_detail={
                    "done_reason": "tool_completed" if step.success else "error"
                },
            )

        if self.planner is None:
            return self._result(
                False,
                profile=profile,
                trace=trace,
                function_id=replayed_function_id or failed_function_id,
                actions_executed=actions_executed,
                model_calls=model_calls,
                llm_usage=llm_usage,
                error=last_error,
                final_state=observation,
            )

        runtime_steps_used = max(actions_executed, len(trace))
        previous_action_error: str | None = (
            last_error if failed_function_id is not None else None
        )
        previous_action: Action | None = None
        stalled_action: Action | None = None
        pending_user_input: str | None = None
        planner_diagnostics: dict[str, Any] = {}
        while runtime_steps_used < self.config.runtime.max_steps:
            max_fallback_steps = self.config.runtime.max_fallback_steps
            if max_fallback_steps is not None and fallback_steps >= max(
                0, int(max_fallback_steps)
            ):
                return self._result(
                    False,
                    profile=profile,
                    trace=trace,
                    function_id=replayed_function_id or failed_function_id,
                    actions_executed=actions_executed,
                    model_calls=model_calls,
                    llm_usage=llm_usage,
                    fallback_steps=fallback_steps,
                    error="fallback_budget_exhausted",
                    final_state=observation,
                    planner_diagnostics=planner_diagnostics,
                )
            observation = await self._observe(screenshot=True)
            recent_actions = _recent_actions(trace)
            execution_history = (
                _execution_history(trace, completed_function=completed_function)
                if trace
                else None
            )
            if (
                previous_action_error
                or recent_actions
                or pending_user_input
                or execution_history
            ):
                observation = Observation(
                    xml=observation.xml,
                    package_name=observation.package_name,
                    activity_name=observation.activity_name,
                    image_base64=observation.image_base64,
                    extra={
                        **dict(observation.extra),
                        "previous_action_error": previous_action_error,
                        "previous_action": previous_action.to_dict()
                        if previous_action is not None
                        else None,
                        **(
                            {"recent_actions": recent_actions} if recent_actions else {}
                        ),
                        **(
                            {"execution_history": execution_history}
                            if execution_history
                            else {}
                        ),
                        **(
                            {"user_input": pending_user_input}
                            if pending_user_input
                            else {}
                        ),
                    },
                )
            pending_user_input = None
            try:
                planned_call = ToolCall.from_value(
                    await _await(
                        self.planner.one_step_tool_call(
                            goal,
                            observation,
                            planner_functions,
                            dict(self.installed_apps),
                        )
                    )
                )
            except Exception as error:  # noqa: BLE001
                planner_metadata = _take_planner_metadata(self.planner)
                _merge_planner_diagnostics(planner_diagnostics, planner_metadata)
                planner_usage = _take_llm_usage(self.planner)
                merge_usage(llm_usage, planner_usage, component="planner")
                model_calls += _usage_model_calls(planner_usage, fallback=1)
                return self._result(
                    False,
                    profile=profile,
                    trace=trace,
                    function_id=replayed_function_id or failed_function_id,
                    actions_executed=actions_executed,
                    model_calls=model_calls,
                    llm_usage=llm_usage,
                    fallback_steps=fallback_steps,
                    error=f"vlm_planner_failed:{error}",
                    final_state=observation,
                    planner_diagnostics=planner_diagnostics,
                )
            planner_usage = _take_llm_usage(self.planner)
            merge_usage(llm_usage, planner_usage, component="planner")
            model_calls += _usage_model_calls(planner_usage, fallback=1)
            fallback_steps += 1
            runtime_steps_used += 1
            planner_metadata = _take_planner_metadata(self.planner)
            _merge_planner_diagnostics(planner_diagnostics, planner_metadata)
            selected_function = planner_function_catalog.get(planned_call.name)
            if selected_function is not None:
                replayed_function_id = selected_function.id
                try:
                    bound_function = bind_function(
                        selected_function,
                        planned_call.arguments,
                    )
                except ValueError as error:
                    previous_action_error = str(error)
                    continue
                replay = await execute_function(
                    bound_function,
                    host=self.host,
                    plugins=self.plugins,
                    observation=observation,
                    max_actions=max(
                        0,
                        self.config.runtime.max_steps - (runtime_steps_used - 1),
                    ),
                    trace_start_index=len(trace),
                    installed_packages=self.installed_packages,
                )
                actions_executed += replay.actions_executed
                replay_trace = list(replay.detail.get("trace") or ())
                trace.extend(replay_trace)
                runtime_steps_used += max(
                    0, max(replay.actions_executed, len(replay_trace)) - 1
                )
                observation = replay.final_state or observation
                if replay.success:
                    completed_function = bound_function
                    failed_function_id = None
                    failed_step_index = None
                    fallback_observations = []
                    previous_action_error = None
                else:
                    failed_function_id = bound_function.id
                    failed_step_index = _optional_step_index(
                        replay.detail.get("failed_step_index")
                    )
                    fallback_observations = (
                        [observation] if failed_step_index is not None else []
                    )
                    previous_action_error = replay.error or "function_replay_failed"
                continue
            try:
                planned = _action_from_tool_call(planned_call)
            except ValueError as error:
                previous_action_error = str(error)
                continue
            if _action_already_succeeded_on_current_state(
                trace,
                planned,
                observation,
            ):
                previous_action_error = "action_already_succeeded_on_current_state"
                previous_action = planned
                continue
            if stalled_action is not None and planned == stalled_action:
                previous_action_error = "repeated_action_without_progress"
                previous_action = planned
                continue
            stalled_action = None
            if planned.tool == "finished":
                return self._result(
                    True,
                    profile=profile,
                    trace=trace,
                    function_id=replayed_function_id or failed_function_id,
                    actions_executed=actions_executed,
                    model_calls=model_calls,
                    llm_usage=llm_usage,
                    fallback_steps=fallback_steps,
                    final_state=observation,
                    planner_diagnostics=planner_diagnostics,
                    terminal_detail={
                        "done_reason": "finished",
                        "finished_content": str(planned.args.get("content") or ""),
                    },
                )
            if planned.tool == "abort":
                message = str(planned.args.get("value") or "").strip() or "vlm_aborted"
                return self._result(
                    False,
                    profile=profile,
                    trace=trace,
                    function_id=replayed_function_id or failed_function_id,
                    actions_executed=actions_executed,
                    model_calls=model_calls,
                    llm_usage=llm_usage,
                    fallback_steps=fallback_steps,
                    error=message,
                    final_state=observation,
                    planner_diagnostics=planner_diagnostics,
                    terminal_detail={"done_reason": "abort"},
                )
            if planned.tool == "info":
                question = str(planned.args.get("value") or "").strip()
                if not question:
                    previous_action_error = "info_question_required"
                    previous_action = planned
                    continue
                try:
                    pending_user_input = str(
                        await _await(_request_input(self.host, question))
                    )
                except InputRequired as error:
                    return self._result(
                        False,
                        profile=profile,
                        trace=trace,
                        function_id=replayed_function_id or failed_function_id,
                        actions_executed=actions_executed,
                        model_calls=model_calls,
                        llm_usage=llm_usage,
                        fallback_steps=fallback_steps,
                        error="input_required",
                        final_state=observation,
                        planner_diagnostics=planner_diagnostics,
                        terminal_detail={
                            "done_reason": "waiting_input",
                            "finished_content": error.question,
                        },
                    )
                except Exception as error:  # noqa: BLE001
                    return self._result(
                        False,
                        profile=profile,
                        trace=trace,
                        function_id=replayed_function_id or failed_function_id,
                        actions_executed=actions_executed,
                        model_calls=model_calls,
                        llm_usage=llm_usage,
                        fallback_steps=fallback_steps,
                        error=f"request_input_failed:{error}",
                        final_state=observation,
                        planner_diagnostics=planner_diagnostics,
                    )
                previous_action_error = None
                previous_action = None
                continue
            if planned.tool == "get_state":
                previous_action_error = None
                previous_action = None
                continue
            step = await execute_action(
                planned,
                observation=observation,
                host=self.host,
                plugins=self.plugins,
                installed_packages=self.installed_packages,
            )
            trace.extend(
                await record_execution(
                    self.host,
                    step,
                    trace_start_index=len(trace),
                    metadata=planner_metadata,
                )
            )
            actions_executed += step.actions_executed
            if not step.success:
                previous_action_error = step.error or "fallback_action_failed"
                previous_action = planned
                if (
                    step.actions_executed > 0
                    and step.after is not None
                    and _same_observation(step.before, step.after)
                ):
                    stalled_action = planned
                continue
            observation = step.after or observation
            if bound_function is not None and failed_step_index is not None:
                fallback_observations.append(observation)
                alignment = await align_function_resume(
                    bound_function,
                    host=self.host,
                    plugins=self.plugins,
                    observations=fallback_observations,
                    start_step_index=failed_step_index,
                )
                if alignment is not None:
                    replay = await execute_function(
                        bound_function,
                        host=self.host,
                        plugins=self.plugins,
                        observation=observation,
                        max_actions=max(
                            0,
                            self.config.runtime.max_steps - runtime_steps_used,
                        ),
                        start_step_index=int(alignment["resume_step_index"]),
                        trace_start_index=len(trace),
                        resume_metadata=alignment,
                        installed_packages=self.installed_packages,
                    )
                    actions_executed += replay.actions_executed
                    replay_trace = list(replay.detail.get("trace") or ())
                    trace.extend(replay_trace)
                    runtime_steps_used += max(
                        replay.actions_executed,
                        len(replay_trace),
                    )
                    observation = replay.final_state or observation
                    if replay.success:
                        failed_function_id = None
                        failed_step_index = None
                        fallback_observations = []
                        last_error = "function_replay_completed_e2e_unverified"
                        completed_function = bound_function
                        previous_action_error = None
                        previous_action = None
                    else:
                        failed_function_id = bound_function.id
                        last_error = replay.error or "function_replay_failed"
                        failed_step_index = _optional_step_index(
                            replay.detail.get("failed_step_index")
                        )
                        fallback_observations = (
                            [observation] if failed_step_index is not None else []
                        )
                        previous_action_error = last_error
                        previous_action = None
                    continue
            if _same_observation(step.before, step.after):
                previous_action_error = "action_completed_without_state_change"
                previous_action = planned
                stalled_action = planned
            else:
                previous_action_error = None
                previous_action = None

        return self._result(
            False,
            profile=profile,
            trace=trace,
            function_id=replayed_function_id or failed_function_id,
            actions_executed=actions_executed,
            model_calls=model_calls,
            llm_usage=llm_usage,
            fallback_steps=fallback_steps,
            error=previous_action_error or "max_steps_exceeded",
            final_state=observation,
            planner_diagnostics=planner_diagnostics,
        )

    async def _observe(
        self,
        *,
        screenshot: bool,
        xml: bool = True,
        app_info: bool = True,
    ) -> Observation:
        return Observation.from_value(
            await _await(
                self.host.observe(
                    xml=xml,
                    screenshot=screenshot,
                    app_info=app_info,
                )
            )
        )

    async def acall_tool(
        self,
        tool_call: ToolCall | dict[str, Any],
        *,
        experiment: Experiment | str | None = None,
    ) -> RunResult:
        return await self._execute(
            "",
            ToolCall.from_value(tool_call),
            experiment=experiment,
        )

    async def arun(
        self,
        goal: str,
        *,
        experiment: Experiment | str | None = None,
    ) -> RunResult:
        return await self._execute(
            str(goal),
            None,
            experiment=experiment,
        )

    def recall(self, goal: str, *, limit: int | None = None) -> list[Function]:
        """Shortlist Function tools for Planner injection without executing one."""

        resolved_limit = (
            self.config.runtime.max_function_tools if limit is None else int(limit)
        )
        return recall_functions(
            str(goal),
            functions=self.store.functions,
            limit=max(0, int(resolved_limit)),
        )

    def call_tool(
        self,
        tool_call: ToolCall | dict[str, Any],
        *,
        experiment: Experiment | str | None = None,
    ) -> RunResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.acall_tool(tool_call, experiment=experiment)
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

    def _result(
        self,
        success: bool,
        *,
        profile: Experiment,
        trace: list[dict[str, Any]],
        function_id: str | None = None,
        actions_executed: int = 0,
        model_calls: int = 0,
        llm_usage: dict[str, Any] | None = None,
        fallback_steps: int = 0,
        error: str | None = None,
        final_state: Observation | None = None,
        planner_diagnostics: dict[str, Any] | None = None,
        terminal_detail: dict[str, Any] | None = None,
    ) -> RunResult:
        detail: dict[str, Any] = {
            "experiment": profile.name,
            "trace": list(trace),
        }
        usage = dict(llm_usage or {})
        tracked_model_calls = _usage_model_calls(usage, fallback=0)
        if tracked_model_calls < model_calls:
            usage["untracked_model_calls"] = model_calls - tracked_model_calls
            usage["model_calls"] = model_calls
        usage["token_usage_status"] = token_usage_status(usage)
        detail["llm_usage"] = usage
        if planner_diagnostics:
            detail["planner_diagnostics"] = dict(planner_diagnostics)
        if terminal_detail:
            detail.update(terminal_detail)
        return RunResult(
            success,
            function_id,
            actions_executed,
            model_calls,
            fallback_steps,
            error,
            final_state,
            detail,
        )


def _experiment(value: Experiment | str | None) -> Experiment:
    if isinstance(value, Experiment):
        return value
    return Experiment.for_method(str(value or "ours"))


def _action_from_tool_call(tool_call: ToolCall) -> Action:
    return Action.from_value(
        canonicalize_action(
            {
                "tool": tool_call.name,
                "args": tool_call.arguments,
            },
            persisted_only=False,
            allow_non_action=True,
        )
    )


def _recent_actions(
    trace: list[dict[str, Any]],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for step in trace[-max(1, int(limit)) :]:
        if not isinstance(step, dict):
            continue
        action = step["action"]
        result = step["result"]
        metadata = step.get("metadata") or {}
        history.append(
            {
                "tool": str(action.get("tool") or ""),
                "args": dict(action.get("args") or {}),
                "success": result.get("success") is True,
                "error": result.get("error"),
                "function_id": metadata.get("function_id") or None,
            }
        )
    return history


def _execution_history(
    trace: list[dict[str, Any]],
    *,
    completed_function: Function | None = None,
) -> str:
    lines = ["Action execution history on the target device:"]
    if completed_function is not None:
        lines.extend(
            [
                (
                    f"Function `{completed_function.id}` "
                    f"({completed_function.name}) completed successfully."
                ),
                f"Function purpose: {completed_function.description}",
            ]
        )
    for index, step in enumerate(trace, start=1):
        if not isinstance(step, dict):
            continue
        try:
            action = Action.from_value(step.get("action"))
        except (TypeError, ValueError):
            continue
        metadata = step.get("metadata")
        function_id = (
            str(metadata.get("function_id") or "").strip()
            if isinstance(metadata, dict)
            else ""
        )
        source = f"Function `{function_id}`" if function_id else "Planner"
        result = step.get("result")
        success = isinstance(result, dict) and result.get("success") is True
        if success:
            description = _describe_completed_action(action)
        else:
            error = (
                str(result.get("error") or "unknown execution error")
                if isinstance(result, dict)
                else "unknown execution error"
            )
            description = f"Action `{action.tool}` failed: {error}."
        lines.append(f"{index}. [{source}] {description}")
    lines.extend(
        [
            (
                "This history records tool execution only, not independent task "
                "validation."
            ),
            (
                "Before making any further Action, verify whether the complete "
                "user goal is already satisfied."
            ),
            (
                "Do not repeat successful Actions. If the goal is complete, "
                "call `finished`."
            ),
        ]
    )
    return "\n".join(lines)


def _function_completion_summary(
    function: Function,
    trace: list[dict[str, Any]],
) -> str:
    successful_actions = sum(
        1
        for step in trace
        if isinstance(step, dict)
        and isinstance(step.get("result"), dict)
        and step["result"].get("success") is True
    )
    action_counts: dict[str, int] = {}
    for step in trace:
        action = step.get("action") if isinstance(step, dict) else None
        tool = str(action.get("tool") or "").strip() if isinstance(action, dict) else ""
        if tool:
            action_counts[tool] = action_counts.get(tool, 0) + 1
    actions = ", ".join(
        f"{tool} x{count}" for tool, count in sorted(action_counts.items())
    )
    return (
        f'Function "{function.name}" completed {successful_actions} successful '
        f"actions ({actions or 'none'}). Intended outcome: {function.description}"
    )


def _describe_completed_action(action: Action) -> str:
    args = action.args
    target = str(args.get("target_description") or "").strip()
    if action.tool == "open_app":
        package_name = str(args.get("package_name") or "").strip()
        return f'Opened app package "{package_name}" successfully.'
    if action.tool == "click":
        if target:
            return f'Clicked target "{target}" successfully.'
        return "Clicked the recorded target position successfully."
    if action.tool == "long_press":
        if target:
            return f'Long-pressed target "{target}" successfully.'
        return "Long-pressed the recorded screen position successfully."
    if action.tool == "input_text":
        if target:
            return f'Entered text into target "{target}" successfully.'
        return "Entered the required text successfully."
    if action.tool == "swipe":
        direction = str(args.get("direction") or "").strip()
        return (
            f"Swiped {direction} successfully."
            if direction
            else "Completed the recorded swipe successfully."
        )
    if action.tool == "press_key":
        key = str(args.get("key") or "").strip()
        return f'Pressed key "{key}" successfully.'
    if action.tool == "wait":
        return f"Waited for {args.get('duration_ms')} ms successfully."
    return f"Completed action `{action.tool}` successfully."


def _same_observation(
    before: Observation | None,
    after: Observation | None,
) -> bool:
    if before is None or after is None:
        return False
    return (
        before.package_name,
        before.activity_name,
        before.xml,
        before.image_base64,
    ) == (
        after.package_name,
        after.activity_name,
        after.xml,
        after.image_base64,
    )


def _action_already_succeeded_on_current_state(
    trace: list[dict[str, Any]],
    action: Action,
    observation: Observation,
) -> bool:
    state_id = str(observation.extra.get("state_id") or "").strip()
    if not state_id:
        return False
    for step in reversed(trace):
        if not isinstance(step, dict):
            continue
        if (
            str(step.get("before_state_id") or "") != state_id
            or str(step.get("after_state_id") or "") != state_id
        ):
            continue
        result = step.get("result")
        if not isinstance(result, dict) or result.get("success") is not True:
            continue
        try:
            completed = Action.from_value(step.get("action"))
        except (TypeError, ValueError):
            continue
        if completed == action:
            return True
    return False


def _optional_step_index(value: Any) -> int | None:
    try:
        step_index = int(value)
    except (TypeError, ValueError):
        return None
    return step_index if step_index >= 0 else None


async def _await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _take_planner_metadata(planner: Planner) -> dict[str, Any]:
    take_metadata = getattr(planner, "take_metadata", None)
    if not callable(take_metadata):
        return {}
    value = take_metadata()
    return dict(value) if isinstance(value, dict) else {}


def _merge_planner_diagnostics(
    diagnostics: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    rejected_calls = metadata.get("rejected_tool_calls")
    if not isinstance(rejected_calls, list):
        return
    accumulated = diagnostics.setdefault("rejected_tool_calls", [])
    for value in rejected_calls:
        if not isinstance(value, dict):
            continue
        error = str(value.get("error") or "").strip()
        if not error:
            continue
        item: dict[str, Any] = {"error": error}
        try:
            turn_index = int(value.get("turn_index"))
        except (TypeError, ValueError):
            turn_index = -1
        if turn_index >= 0:
            item["turn_index"] = turn_index
        tool = str(value.get("tool") or "").strip()
        if tool:
            item["tool"] = tool
        if "arguments" in value:
            item["arguments"] = value.get("arguments")
        if item not in accumulated:
            accumulated.append(item)


def _take_llm_usage(component: Any) -> dict[str, Any] | None:
    take_usage = getattr(component, "take_usage", None)
    if not callable(take_usage):
        return None
    value = take_usage()
    return dict(value) if isinstance(value, dict) else {}


def _usage_model_calls(
    usage: dict[str, Any] | None,
    *,
    fallback: int,
) -> int:
    if usage is None:
        return max(0, int(fallback))
    try:
        return max(0, int(usage.get("model_calls") or 0))
    except (TypeError, ValueError):
        return max(0, int(fallback))


async def _request_input(host: Host, question: str) -> str:
    request_input = getattr(host, "request_input", None)
    if not callable(request_input):
        raise RuntimeError("request_input_not_supported")
    return str(await _await(request_input(question)))
