from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import inspect
import json
from pathlib import Path
from typing import Any

from omniflow.catalog import CatalogSnapshot
from omniflow.core.config import Experiment, OmniFlowConfig
from omniflow.core.model import (
    Action,
    Function,
    Host,
    Observation,
    Planner,
    RunResult,
    ToolCall,
)
from omniflow.core.schemas import canonicalize_action
from omniflow.functions.assets import FunctionStore, bind_function
from omniflow.functions.recall import RecallResult, recall_functions
from omniflow.runtime.execution import (
    execute_function,
    execute_robust_action,
    record_execution,
)
from omniflow.vlm.usage import merge_usage, token_usage_status


class InputRequired(RuntimeError):
    def __init__(self, question: str):
        self.question = str(question).strip()
        super().__init__(self.question or "input_required")


@dataclass
class _FunctionSession:
    selected_id: str | None = None
    bound: Function | None = None
    failed: bool = False
    failed_step_index: int | None = None
    completed: Function | None = None
    resume_events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def failed_id(self) -> str | None:
        return self.selected_id if self.failed else None

    def mark_completed(self) -> None:
        self.completed = self.bound
        self.failed = False
        self.failed_step_index = None

    def mark_failed(self, replay: RunResult) -> None:
        self.failed = True
        self.completed = None
        self.failed_step_index = _optional_step_index(
            replay.detail.get("failed_step_index")
        )


class OmniFlow:
    def __init__(
        self,
        store_path: str | Path,
        *,
        host: Host | None = None,
        planner: Planner | None = None,
        installed_apps: dict[str, str] | None = None,
        config: OmniFlowConfig | None = None,
        catalog: CatalogSnapshot | None = None,
    ):
        self.config = config or OmniFlowConfig()
        self.catalog = catalog
        self.store = FunctionStore(store_path)
        self.host = host
        self.planner = planner
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
        planner_steps = 0
        model_calls = 0
        fallback_steps = 0
        trace: list[dict[str, Any]] = []
        checker_decisions: list[dict[str, Any]] = []
        last_error = "tool_not_selected"
        llm_usage: dict[str, Any] = {}
        function_session = _FunctionSession()
        observation = await self._observe(screenshot=True)
        planner_functions: tuple[Function, ...] = ()
        planner_function_catalog: dict[str, Function] = {}
        recall_events: list[dict[str, Any]] = []
        recall_source_states: dict[str, Observation | None] = {}
        function_resolution: dict[str, Any] = {
            "candidate_count": 0,
            "candidate_function_ids": [],
            "status": "direct" if direct_tool_call is not None else "planner_tool_space",
        }

        def finish(success: bool, **kwargs: Any) -> RunResult:
            kwargs.setdefault("planner_steps", planner_steps)
            if recall_events:
                function_resolution["recall"] = {
                    "schema_version": "omniflow.function-recall-events.v1",
                    "events": [dict(event) for event in recall_events],
                }
            if function_session.resume_events:
                terminal_detail = dict(kwargs.get("terminal_detail") or {})
                terminal_detail["function_resume"] = {
                    "schema_version": "omniflow.function-resume-audit.v1",
                    "events": [dict(event) for event in function_session.resume_events],
                    "attempt_count": len(function_session.resume_events),
                    "success_count": sum(
                        event.get("status") == "succeeded"
                        for event in function_session.resume_events
                    ),
                }
                kwargs["terminal_detail"] = terminal_detail
            if checker_decisions:
                terminal_detail = dict(kwargs.get("terminal_detail") or {})
                terminal_detail["checker_decisions"] = [
                    dict(decision) for decision in checker_decisions
                ]
                kwargs["terminal_detail"] = terminal_detail
            return self._result(
                success,
                function_resolution=function_resolution,
                **kwargs,
            )

        selected_function: Function | None = None
        resolved_arguments: dict[str, Any] = {}
        selected_tool_call = direct_tool_call
        if selected_tool_call is not None:
            selected_function = self.store.get_function(selected_tool_call.name)
            resolved_arguments = dict(selected_tool_call.arguments)
            function_resolution["selected_function_id"] = selected_tool_call.name
            function_resolution["arguments"] = dict(resolved_arguments)
            if selected_function is None:
                function_resolution["status"] = "unknown_selection"
            elif direct_tool_call is not None and not goal:
                goal = _direct_function_fallback_goal(
                    selected_function,
                    resolved_arguments,
                )

        if selected_function is not None:
            function_session.selected_id = selected_function.id
            try:
                function_session.bound = bind_function(
                    selected_function, resolved_arguments
                )
            except ValueError as error:
                function_resolution["binding_status"] = "failed"
                function_resolution["binding_error"] = str(error)
                function_resolution["replay_status"] = "not_started"
                replay = RunResult(
                    False,
                    function_id=selected_function.id,
                    error=str(error),
                    final_state=observation,
                )
            else:
                function_resolution["binding_status"] = "succeeded"
                replay = await execute_function(
                    function_session.bound,
                    host=self.host,
                    plugins=self.plugins,
                    observation=observation,
                    installed_packages=self.installed_packages,
                    state_loader=(
                        self.catalog.get_state if self.catalog is not None else None
                    ),
                    checker_action_confidence=(
                        self.config.runtime.checker_action_confidence
                    ),
                )
            actions_executed += replay.actions_executed
            trace.extend(replay.detail.get("trace") or ())
            checker_decisions.extend(replay.detail.get("checker_decisions") or ())
            if replay.success:
                function_resolution["replay_status"] = "succeeded"
                observation = replay.final_state or observation
                last_error = "function_replay_completed_e2e_unverified"
                function_session.mark_completed()
            else:
                function_resolution["replay_status"] = "failed"
                function_resolution["replay_error"] = (
                    replay.error or "function_replay_failed"
                )
            observation = replay.final_state or observation
            if not replay.success:
                last_error = replay.error or "function_replay_failed"
                function_session.mark_failed(replay)
                function_resolution["failed_step_index"] = (
                    function_session.failed_step_index
                )

            if direct_tool_call is not None and replay.success:
                return finish(
                    True,
                    profile=profile,
                    trace=trace,
                    function_id=direct_tool_call.name,
                    actions_executed=actions_executed,
                    model_calls=model_calls,
                    llm_usage=llm_usage,
                    error=None,
                    final_state=observation,
                    terminal_detail={
                        "done_reason": "function_completed"
                    },
                )

        if direct_tool_call is not None and selected_function is None:
            try:
                direct_action = _action_from_tool_call(direct_tool_call)
            except ValueError as error:
                return finish(
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
                return finish(
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
                return finish(
                    False,
                    profile=profile,
                    trace=trace,
                    actions_executed=actions_executed,
                    model_calls=model_calls,
                    llm_usage=llm_usage,
                    error=f"tool_not_directly_invokable:{direct_action.tool}",
                    final_state=observation,
                )
            step = await execute_robust_action(
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
            return finish(
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
            return finish(
                False,
                profile=profile,
                trace=trace,
                function_id=function_session.selected_id or function_session.failed_id,
                actions_executed=actions_executed,
                model_calls=model_calls,
                llm_usage=llm_usage,
                error=last_error,
                final_state=observation,
            )

        previous_action_error: str | None = (
            last_error if function_session.failed else None
        )
        previous_action: Action | None = None
        pending_user_input: str | None = None
        planner_diagnostics: dict[str, Any] = {}
        while planner_steps < self.config.runtime.max_steps:
            max_fallback_steps = self.config.runtime.max_fallback_steps
            fallback_this_turn = function_session.failed
            if (
                fallback_this_turn
                and max_fallback_steps is not None
                and fallback_steps >= max(
                    0, int(max_fallback_steps)
                )
            ):
                return finish(
                    False,
                    profile=profile,
                    trace=trace,
                    function_id=function_session.selected_id or function_session.failed_id,
                    actions_executed=actions_executed,
                    model_calls=model_calls,
                    llm_usage=llm_usage,
                    fallback_steps=fallback_steps,
                    error="fallback_budget_exhausted",
                    final_state=observation,
                    planner_diagnostics=planner_diagnostics,
                )
            evidence_function = function_session.completed or (
                function_session.bound if function_session.failed else None
            )
            function_execution = (
                _function_execution_evidence(
                    trace,
                    function=evidence_function,
                    final_observation=observation,
                    succeeded=function_session.completed is not None,
                )
                if evidence_function is not None
                else None
            )
            if (
                previous_action_error
                or pending_user_input
                or function_execution
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
                            {"function_execution": function_execution}
                            if function_execution
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
            recall_result = await self._recall(
                goal,
                observation=observation,
                source_states=recall_source_states,
            )
            planner_functions = recall_result.functions
            planner_function_catalog = {
                function.id: function for function in planner_functions
            }
            recall_event = {
                "planner_turn": planner_steps,
                **recall_result.audit,
            }
            recall_events.append(recall_event)
            function_resolution["candidate_count"] = len(planner_functions)
            function_resolution["candidate_function_ids"] = [
                function.id for function in planner_functions
            ]
            planner_steps += 1
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
                return finish(
                    False,
                    profile=profile,
                    trace=trace,
                    function_id=function_session.selected_id or function_session.failed_id,
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
            if fallback_this_turn:
                fallback_steps += 1
            planner_metadata = _take_planner_metadata(self.planner)
            _merge_planner_diagnostics(planner_diagnostics, planner_metadata)
            selected_function = planner_function_catalog.get(planned_call.name)
            if selected_function is not None:
                retry_step_index = (
                    function_session.failed_step_index
                    if function_session.failed
                    and function_session.selected_id == selected_function.id
                    else None
                )
                previous_bound = function_session.bound
                try:
                    bound_function = bind_function(
                        selected_function,
                        planned_call.arguments,
                    )
                except ValueError as error:
                    previous_action_error = str(error)
                    if planner_steps >= self.config.runtime.max_steps:
                        return finish(
                            False,
                            profile=profile,
                            trace=trace,
                            function_id=selected_function.id,
                            actions_executed=actions_executed,
                            model_calls=model_calls,
                            llm_usage=llm_usage,
                            fallback_steps=fallback_steps,
                            error=previous_action_error,
                            final_state=observation,
                            planner_diagnostics=planner_diagnostics,
                            terminal_detail={"done_reason": "step_completed"},
                        )
                    continue
                if previous_bound != bound_function:
                    retry_step_index = None
                function_session.selected_id = selected_function.id
                function_session.bound = bound_function
                retry_metadata = None
                resume_event = None
                if retry_step_index is not None:
                    retry_step = next(
                        (
                            step
                            for step in bound_function.steps
                            if step.step_index == retry_step_index
                        ),
                        None,
                    )
                    if retry_step is not None:
                        retry_metadata = {
                            "protocol": "explicit_function_retry_v1",
                            "start_step_index": int(retry_step_index),
                            "resume_step_index": int(retry_step_index),
                            "source_state_id": retry_step.source_state_id,
                        }
                        resume_event = {
                            "start_step_index": int(retry_step_index),
                            "status": "retrying",
                            "trigger": "explicit_function_call",
                            "resume_step_index": int(retry_step_index),
                            "source_state_id": retry_step.source_state_id,
                        }
                        function_session.resume_events.append(resume_event)
                replay = await execute_function(
                    bound_function,
                    host=self.host,
                    plugins=self.plugins,
                    observation=observation,
                    start_step_index=int(retry_step_index or 0),
                    trace_start_index=len(trace),
                    resume_metadata=retry_metadata,
                    installed_packages=self.installed_packages,
                    state_loader=(
                        self.catalog.get_state if self.catalog is not None else None
                    ),
                    checker_action_confidence=(
                        self.config.runtime.checker_action_confidence
                    ),
                )
                actions_executed += replay.actions_executed
                replay_trace = list(replay.detail.get("trace") or ())
                trace.extend(replay_trace)
                checker_decisions.extend(
                    replay.detail.get("checker_decisions") or ()
                )
                observation = replay.final_state or observation
                if replay.success:
                    if resume_event is not None:
                        resume_event["status"] = "succeeded"
                    function_session.mark_completed()
                    previous_action_error = None
                else:
                    if resume_event is not None:
                        resume_event["status"] = "failed"
                        resume_event["error"] = replay.error or "function_replay_failed"
                    function_session.mark_failed(replay)
                    previous_action_error = replay.error or "function_replay_failed"
                if planner_steps >= self.config.runtime.max_steps:
                    return finish(
                        False,
                        profile=profile,
                        trace=trace,
                        function_id=selected_function.id,
                        actions_executed=actions_executed,
                        model_calls=model_calls,
                        llm_usage=llm_usage,
                        fallback_steps=fallback_steps,
                        error=None if replay.success else previous_action_error,
                        final_state=observation,
                        planner_diagnostics=planner_diagnostics,
                        terminal_detail={"done_reason": "step_completed"},
                    )
                continue
            try:
                planned = _action_from_tool_call(planned_call)
            except ValueError as error:
                previous_action_error = str(error)
                if planner_steps >= self.config.runtime.max_steps:
                    return finish(
                        False,
                        profile=profile,
                        trace=trace,
                        function_id=function_session.selected_id or function_session.failed_id,
                        actions_executed=actions_executed,
                        model_calls=model_calls,
                        llm_usage=llm_usage,
                        fallback_steps=fallback_steps,
                        error=previous_action_error,
                        final_state=observation,
                        planner_diagnostics=planner_diagnostics,
                        terminal_detail={"done_reason": "step_completed"},
                    )
                continue
            if planned.tool == "finished":
                return finish(
                    True,
                    profile=profile,
                    trace=trace,
                    function_id=function_session.selected_id or function_session.failed_id,
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
                return finish(
                    False,
                    profile=profile,
                    trace=trace,
                    function_id=function_session.selected_id or function_session.failed_id,
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
                    return finish(
                        False,
                        profile=profile,
                        trace=trace,
                        function_id=function_session.selected_id or function_session.failed_id,
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
                    return finish(
                        False,
                        profile=profile,
                        trace=trace,
                        function_id=function_session.selected_id or function_session.failed_id,
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
                observation = await self._observe(screenshot=True)
                previous_action_error = None
                previous_action = None
                if planner_steps >= self.config.runtime.max_steps:
                    return finish(
                        False,
                        profile=profile,
                        trace=trace,
                        function_id=function_session.selected_id or function_session.failed_id,
                        actions_executed=actions_executed,
                        model_calls=model_calls,
                        llm_usage=llm_usage,
                        fallback_steps=fallback_steps,
                        error=None,
                        final_state=observation,
                        planner_diagnostics=planner_diagnostics,
                        terminal_detail={"done_reason": "step_completed"},
                    )
                continue
            step = await execute_robust_action(
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
            observation = step.after or observation
            previous_action_error = (
                None if step.success else step.error or "fallback_action_failed"
            )
            if planner_steps >= self.config.runtime.max_steps:
                return finish(
                    False,
                    profile=profile,
                    trace=trace,
                    function_id=function_session.selected_id or function_session.failed_id,
                    actions_executed=actions_executed,
                    model_calls=model_calls,
                    llm_usage=llm_usage,
                    fallback_steps=fallback_steps,
                    error=previous_action_error,
                    final_state=observation,
                    planner_diagnostics=planner_diagnostics,
                    terminal_detail={"done_reason": "step_completed"},
                )
            previous_action = None if step.success else planned

        return finish(
            False,
            profile=profile,
            trace=trace,
            function_id=function_session.selected_id or function_session.failed_id,
            actions_executed=actions_executed,
            model_calls=model_calls,
            llm_usage=llm_usage,
            fallback_steps=fallback_steps,
            error=previous_action_error or "max_steps_exceeded",
            final_state=observation,
            planner_diagnostics=planner_diagnostics,
        )

    async def _observe(self, *, screenshot: bool) -> Observation:
        return Observation.from_value(
            await _await(
                self.host.observe(
                    xml=True,
                    screenshot=screenshot,
                    app_info=True,
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

    async def _recall(
        self,
        goal: str,
        *,
        observation: Observation,
        source_states: dict[str, Observation | None],
        limit: int | None = None,
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
                        value = await _await(get_state(source_state_id))
                        source_state = (
                            Observation.from_value(value) if value is not None else None
                        )
                    except Exception:  # noqa: BLE001
                        source_state = None
            source_states[source_state_id] = source_state

        resolved_limit = (
            self.config.runtime.max_function_tools if limit is None else int(limit)
        )
        return recall_functions(
            str(goal),
            observation=observation,
            functions=self.store.functions,
            source_states=source_states,
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
        function_resolution: dict[str, Any] | None = None,
        terminal_detail: dict[str, Any] | None = None,
        planner_steps: int = 0,
    ) -> RunResult:
        detail: dict[str, Any] = {
            "experiment": profile.name,
            "trace": list(trace),
            "runtime_limits": {
                "max_steps": int(self.config.runtime.max_steps),
                "max_fallback_steps": (
                    int(self.config.runtime.max_fallback_steps)
                    if self.config.runtime.max_fallback_steps is not None
                    else None
                ),
            },
            "planner_steps": max(0, int(planner_steps)),
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
        if function_resolution is not None:
            detail["function_resolution"] = dict(function_resolution)
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


def _direct_function_fallback_goal(
    function: Function,
    arguments: dict[str, Any],
) -> str:
    return (
        f'Continue Function "{function.name}" from the current screen after '
        "offline replay could not map its next step. Do not repeat actions that "
        "already succeeded. "
        f"Requested arguments: {json.dumps(arguments, ensure_ascii=False, sort_keys=True)}. "
        f"{function.description}"
    ).strip()


def _function_execution_evidence(
    trace: list[dict[str, Any]],
    *,
    function: Function,
    final_observation: Observation,
    succeeded: bool,
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    for raw_step in trace:
        if not isinstance(raw_step, dict):
            continue
        metadata = raw_step.get("metadata")
        if not isinstance(metadata, dict) or str(
            metadata.get("function_id") or ""
        ).strip() != function.id:
            continue
        try:
            action = Action.from_value(raw_step.get("action"))
        except (TypeError, ValueError):
            continue
        result = raw_step.get("result")
        success = isinstance(result, dict) and result.get("success") is True
        step = {
            "step_index": int(raw_step.get("step_index") or len(steps)),
            "before_state_id": str(raw_step.get("before_state_id") or ""),
            "after_state_id": str(raw_step.get("after_state_id") or ""),
            "tool": action.tool,
            "success": success,
        }
        if not success and isinstance(result, dict) and str(
            result.get("error") or ""
        ).strip():
            step["error"] = str(result["error"])
        steps.append(step)
    final_state_id = str(final_observation.extra.get("state_id") or "").strip()
    if not final_state_id and steps:
        final_state_id = str(steps[-1]["after_state_id"] or "").strip()
    return {
        "schema_version": "omniflow.function-execution-evidence.v1",
        "function_id": function.id,
        "function_name": function.name,
        "function_description": function.description,
        "replay_status": "actions_succeeded" if succeeded else "actions_failed",
        "official_validator_status": "pending",
        "steps": steps,
        "final_observation": {
            "state_id": final_state_id,
            "package_name": str(final_observation.package_name or ""),
            "activity_name": str(final_observation.activity_name or ""),
        },
    }


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
