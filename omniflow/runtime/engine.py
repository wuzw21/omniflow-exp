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
    FunctionRouter,
    Host,
    Observation,
    Planner,
    RunResult,
    ToolCall,
)
from omniflow.core.schemas import canonicalize_action
from omniflow.functions.artifact import bind_function
from omniflow.functions.recall import RecallResult, recall_functions
from omniflow.functions.store import FunctionStore
from omniflow.runtime.checker import CheckerLibrary
from omniflow.runtime.execution import (
    align_function_resume,
    execute_function,
    execute_robust_action,
    record_execution,
)
from omniflow.transfer.embedding import PageEncoder
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
    fallback_observations: list[Observation] = field(default_factory=list)
    completed: Function | None = None
    resume_events: list[dict[str, Any]] = field(default_factory=list)
    resume_trigger: str | None = None
    excluded_ids: set[str] = field(default_factory=set)

    @property
    def failed_id(self) -> str | None:
        return self.selected_id if self.failed else None

    def mark_completed(self) -> None:
        self.completed = self.bound
        if (
            self.bound is not None
            and self.bound.steps
            and self.bound.steps[0].action.tool == "open_app"
        ):
            self.excluded_ids.add(self.bound.id)
        self.failed = False
        self.failed_step_index = None
        self.fallback_observations.clear()
        self.resume_trigger = None

    def mark_failed(self, replay: RunResult, observation: Observation) -> None:
        if self.selected_id is not None:
            self.excluded_ids.add(self.selected_id)
        self.failed = True
        self.failed_step_index = _optional_step_index(
            replay.detail.get("failed_step_index")
        )
        self.fallback_observations = (
            [observation] if self.failed_step_index is not None else []
        )
        self.resume_trigger = "function_replay_failure"


class OmniFlow:
    def __init__(
        self,
        store_path: str | Path,
        *,
        host: Host | None = None,
        planner: Planner | None = None,
        function_router: FunctionRouter | None = None,
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
        self.checker_library = CheckerLibrary.load(
            Path(store_path).expanduser().resolve().with_name("checker_store.json")
        )
        self.host = host
        self.planner = planner
        self.function_router = function_router
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
        self._page_encoder: PageEncoder | None = None

    async def _execute(
        self,
        goal: str,
        direct_tool_call: ToolCall | None,
        *,
        experiment: Experiment | str | None = None,
        checker_trigger_counts: dict[str, int] | None = None,
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
        completion_review_calls = 0
        trace: list[dict[str, Any]] = []
        last_error = "tool_not_selected"
        llm_usage: dict[str, Any] = {}
        function_session = _FunctionSession()
        shared_checker_trigger_counts = (
            checker_trigger_counts if checker_trigger_counts is not None else {}
        )
        observation = await self._observe(screenshot=False)
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
            kwargs.setdefault("completion_review_calls", completion_review_calls)
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
                    checker_rules=self.checker_library.rules,
                    checker_trigger_counts=shared_checker_trigger_counts,
                )
            actions_executed += replay.actions_executed
            trace.extend(replay.detail.get("trace") or ())
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
                function_session.mark_failed(replay, observation)
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

        runtime_steps_used = 0
        previous_action_error: str | None = (
            last_error if function_session.failed else None
        )
        previous_action: Action | None = None
        pending_user_input: str | None = None
        planner_diagnostics: dict[str, Any] = {}
        while runtime_steps_used < self.config.runtime.max_steps:
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
            if not observation.image_base64:
                observation = await self._observe(screenshot=True)
            recent_actions = _recent_actions(trace)
            execution_history = (
                _execution_history(trace, completed_function=function_session.completed)
                if trace
                else None
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
                or recent_actions
                or pending_user_input
                or execution_history
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
                            {"recent_actions": recent_actions} if recent_actions else {}
                        ),
                        **(
                            {"execution_history": execution_history}
                            if execution_history
                            else {}
                        ),
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
                exclude_function_ids=frozenset(function_session.excluded_ids),
            )
            planner_functions = recall_result.functions
            planner_function_catalog = {
                function.id: function for function in planner_functions
            }
            recall_event = {
                "planner_turn": runtime_steps_used,
                **recall_result.audit,
            }
            recall_events.append(recall_event)
            function_resolution["candidate_count"] = len(planner_functions)
            function_resolution["candidate_function_ids"] = [
                function.id for function in planner_functions
            ]
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
            runtime_steps_used += 1
            planner_metadata = _take_planner_metadata(self.planner)
            _merge_planner_diagnostics(planner_diagnostics, planner_metadata)
            selected_function = planner_function_catalog.get(planned_call.name)
            if selected_function is not None:
                function_session.selected_id = selected_function.id
                try:
                    function_session.bound = bind_function(
                        selected_function,
                        planned_call.arguments,
                    )
                except ValueError as error:
                    previous_action_error = str(error)
                    continue
                current_entry_observation = await self._observe(screenshot=True)
                if not _same_entry_observation(
                    observation,
                    current_entry_observation,
                ):
                    observation = current_entry_observation
                    previous_action_error = (
                        "function_entry_state_changed_after_mapping"
                    )
                    continue
                observation = current_entry_observation
                replay = await execute_function(
                    function_session.bound,
                    host=self.host,
                    plugins=self.plugins,
                    observation=observation,
                    trace_start_index=len(trace),
                    installed_packages=self.installed_packages,
                    state_loader=(
                        self.catalog.get_state if self.catalog is not None else None
                    ),
                    checker_rules=self.checker_library.rules,
                    checker_trigger_counts=shared_checker_trigger_counts,
                )
                actions_executed += replay.actions_executed
                replay_trace = list(replay.detail.get("trace") or ())
                trace.extend(replay_trace)
                observation = replay.final_state or observation
                if replay.success:
                    function_session.mark_completed()
                    previous_action_error = None
                else:
                    function_session.mark_failed(replay, observation)
                    previous_action_error = replay.error or "function_replay_failed"
                continue
            try:
                planned = _action_from_tool_call(planned_call)
            except ValueError as error:
                previous_action_error = str(error)
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
            if not step.success:
                previous_action_error = step.error or "fallback_action_failed"
                previous_action = planned
                continue
            observation = step.after or observation
            if (
                function_session.bound is not None
                and function_session.failed_step_index is not None
            ):
                function_session.fallback_observations.append(observation)
                alignment = await align_function_resume(
                    function_session.bound,
                    host=self.host,
                    plugins=self.plugins,
                    observations=function_session.fallback_observations,
                    start_step_index=function_session.failed_step_index,
                    state_loader=(
                        self.catalog.get_state if self.catalog is not None else None
                    ),
                )
                resume_event = {
                    "start_step_index": int(function_session.failed_step_index),
                    "status": "aligned" if alignment is not None else "not_aligned",
                }
                if function_session.resume_trigger:
                    resume_event["trigger"] = function_session.resume_trigger
                if alignment is not None:
                    resume_event.update(
                        {
                            "resume_step_index": int(alignment["resume_step_index"]),
                            "probability": alignment.get("probability"),
                            "score": alignment.get("score"),
                        }
                    )
                function_session.resume_events.append(resume_event)
                if alignment is not None:
                    replay = await execute_function(
                        function_session.bound,
                        host=self.host,
                        plugins=self.plugins,
                        observation=observation,
                        start_step_index=int(alignment["resume_step_index"]),
                        trace_start_index=len(trace),
                        resume_metadata=alignment,
                        installed_packages=self.installed_packages,
                        state_loader=(
                            self.catalog.get_state if self.catalog is not None else None
                        ),
                        checker_rules=self.checker_library.rules,
                        checker_trigger_counts=shared_checker_trigger_counts,
                    )
                    actions_executed += replay.actions_executed
                    replay_trace = list(replay.detail.get("trace") or ())
                    trace.extend(replay_trace)
                    observation = replay.final_state or observation
                    if replay.success:
                        resume_event["status"] = "succeeded"
                        function_session.mark_completed()
                        last_error = "function_replay_completed_e2e_unverified"
                        previous_action_error = None
                        previous_action = None
                    else:
                        resume_event["status"] = "failed"
                        resume_event["error"] = (
                            replay.error or "function_replay_failed"
                        )
                        last_error = replay.error or "function_replay_failed"
                        function_session.mark_failed(replay, observation)
                        previous_action_error = last_error
                        previous_action = None
                    continue
            if _same_observation(step.before, step.after):
                previous_action_error = "action_completed_without_state_change"
                previous_action = planned
            else:
                previous_action_error = None
                previous_action = None

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
        checker_trigger_counts: dict[str, int] | None = None,
    ) -> RunResult:
        return await self._execute(
            "",
            ToolCall.from_value(tool_call),
            experiment=experiment,
            checker_trigger_counts=checker_trigger_counts,
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
        return await recall_functions(
            str(goal),
            observation=observation,
            functions=self.store.functions,
            source_states=source_states,
            limit=max(0, int(resolved_limit)),
            page_encoder=self._get_page_encoder(),
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
        completion_review_calls: int = 0,
        error: str | None = None,
        final_state: Observation | None = None,
        planner_diagnostics: dict[str, Any] | None = None,
        function_resolution: dict[str, Any] | None = None,
        terminal_detail: dict[str, Any] | None = None,
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
        }
        usage = dict(llm_usage or {})
        tracked_model_calls = _usage_model_calls(usage, fallback=0)
        if tracked_model_calls < model_calls:
            usage["untracked_model_calls"] = model_calls - tracked_model_calls
            usage["model_calls"] = model_calls
        usage["token_usage_status"] = token_usage_status(usage)
        detail["llm_usage"] = usage
        detail["completion_review_calls"] = max(0, int(completion_review_calls))
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


def _same_entry_observation(
    before: Observation | None,
    after: Observation | None,
) -> bool:
    """Check the semantic state at the Function mapping/execution boundary.

    The entry gate must reject a real UI-state change, but AndroidWorld's
    screenshot can change while the canonical accessibility state remains the
    same (for example, a clock tick or screenshot encoding difference).  The
    canonical state id is therefore the authority when both observations have
    one.  Hosts without a state id retain the strict observation comparison.
    """
    if before is None or after is None:
        return False
    before_state_id = str(before.extra.get("state_id") or "").strip()
    after_state_id = str(after.extra.get("state_id") or "").strip()
    if before_state_id or after_state_id:
        return bool(before_state_id and after_state_id) and (
            before_state_id == after_state_id
        )
    return _same_observation(before, after)


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
