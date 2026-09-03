from __future__ import annotations

from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any

from omniflow import (
    Observation,
    OmniFlow,
    OmniFlowConfig,
    RunResult,
    RuntimeSettings,
)
from omniflow.core.config import DEFAULT_MAX_STEPS, Experiment
from omniflow.core.trajectory import state_id
from omniflow.functions.store import FunctionStore
from omniflow.transfer.runtime import (
    TRANSFER_STATE_CATALOG_FILENAME,
    capture_transfer_state as _transfer_state,
    load_transfer_state_catalog,
    transfer_state_coverage,
)
from src.experiment.observation_evidence import canonicalize_run_log_observation
from src.experiment.performance_metrics import PerformanceMetrics
from src.integrations.android_world.host import AndroidWorldHost, make_agent_result

MODE_OMNIFLOW = "omniflow"


def _emit_official_completion(env: Any, answer: str = "") -> None:
    json_action = importlib.import_module("android_world.env.json_action")
    answer_text = str(answer or "").strip()
    if answer_text:
        env.execute_action(
            json_action.JSONAction(
                action_type=json_action.ANSWER,
                text=answer_text,
            )
        )
        return
    env.execute_action(
        json_action.JSONAction(
            action_type=json_action.STATUS,
            goal_status="complete",
        )
    )


class _TaskHost:
    def __init__(
        self,
        host: AndroidWorldHost,
        state: dict[str, Any],
        transfer_states: dict[str, dict[str, Any]],
    ):
        self.host = host
        self.state = state
        self.transfer_states = transfer_states

    @property
    def env(self) -> Any:
        return self.host.env

    def observe(self, **kwargs: Any) -> Observation:
        observation = Observation.from_value(self.host.observe(**kwargs))
        official_state = observation.extra.get("androidworld_state")
        valid_fields = (
            {"screenshot", "xml"},
            {"pixels", "forest", "ui_elements", "auxiliaries"},
        )
        if (
            not isinstance(official_state, dict)
            or set(official_state) not in valid_fields
        ):
            raise ValueError("androidworld_state_snapshot_required")
        official_state = _json_copy(official_state)
        official_state_id = state_id(
            canonicalize_run_log_observation(official_state)
        )
        identified = Observation.from_value(
            {
                **observation.to_dict(),
                "state_id": official_state_id,
            }
        )
        transfer_state = _transfer_state(identified)
        self.state["captured_transfer_states"][transfer_state["state_id"]] = (
            transfer_state
        )
        return identified

    def act(self, action: Any):
        return self.host.act(action)

    def get_state(self, source_state_id: str) -> Observation | None:
        value = self.transfer_states.get(str(source_state_id or "").strip())
        return Observation.from_value(value) if value is not None else None

def build_agent(
    *,
    env: Any,
    store_path: str | None = None,
    runtime: Any | None = None,
    planner: Any | None = None,
    function_router: Any | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    adb_serial: str = "",
    adb_path: str = "",
    post_action_wait_seconds: float = 0.0,
    task_seed: int | None = None,
    evidence_root: str | Path | None = None,
    performance_metrics: PerformanceMetrics | None = None,
    allow_empty_store: bool = False,
) -> OmniFlow | SimpleNamespace:
    if env is None:
        raise TypeError("build_agent requires env parameter")
    del runtime
    if not store_path and not allow_empty_store:
        raise ValueError("function_store_required")
    raw_host = AndroidWorldHost(
        env,
        adb_serial=adb_serial,
        adb_path=adb_path,
        post_action_wait_seconds=post_action_wait_seconds,
        open_app_ready_timeout_seconds=(
            float(
                os.environ.get(
                    "OMNIFLOW_OPEN_APP_READY_TIMEOUT_SECONDS", "15"
                )
            )
            if store_path
            else 0.0
        ),
        evidence_root=evidence_root,
        performance_metrics=performance_metrics,
        control_backend=os.environ.get(
            "OMNIFLOW_ANDROIDWORLD_CONTROL_BACKEND", "oob"
        ),
    )
    empty_store_tempdir: tempfile.TemporaryDirectory[str] | None = None
    if not store_path and planner is None:
        # Fixed replay owns its action sequence in _apply_fixed_replay.  It
        # still needs the canonical AndroidWorld Host and lifecycle wrapper,
        # but it has no Function Store to load.  Keep the adapter object small
        # and let the shared replay seam remain the only executor.
        return SimpleNamespace(
            env=env,
            host=raw_host,
            name=MODE_OMNIFLOW,
            set_max_steps=lambda _step_budget: None,
            reset=lambda go_home=False: raw_host.reset(go_home=go_home),
        )
    if not store_path:
        empty_store_tempdir = tempfile.TemporaryDirectory(
            prefix="omniflow-empty-store-"
        )
        empty_store_path = Path(empty_store_tempdir.name) / "store.json"
        FunctionStore(empty_store_path).save()
        store_path = str(empty_store_path)
    resolved_store_path = Path(store_path).expanduser().resolve()
    transfer_state_path = (
        resolved_store_path.parent / TRANSFER_STATE_CATALOG_FILENAME
    )
    state: dict[str, Any] = {
        "task_name": "",
        "goal": "",
        "task_parameters": {},
        "seed": task_seed,
        "last_result": None,
        "step_index": 0,
        "max_steps": max(1, int(max_steps)),
        "captured_transfer_states": {},
    }
    transfer_states = load_transfer_state_catalog(transfer_state_path)
    host = _TaskHost(raw_host, state, transfer_states)
    fallback_limit_text = str(
        os.environ.get("OMNIFLOW_ANDROIDWORLD_MAX_FALLBACK_STEPS") or ""
    ).strip()
    max_fallback_steps = (
        max(0, int(fallback_limit_text)) if fallback_limit_text else None
    )
    configured_max_steps = max(1, int(max_steps))
    flow = OmniFlow(
        resolved_store_path,
        host=host,
        planner=planner,
        function_router=function_router,
        installed_apps=raw_host.installed_apps(),
        config=OmniFlowConfig(
            runtime=RuntimeSettings(
                max_steps=configured_max_steps,
                max_fallback_steps=max_fallback_steps,
            ),
        ),
    )
    coverage = transfer_state_coverage(flow.store.functions, transfer_states)
    if coverage["required_state_count"] and not transfer_state_path.is_file():
        raise RuntimeError(f"transfer_state_catalog_missing:{transfer_state_path}")
    if not coverage["complete"]:
        missing = ",".join(coverage["missing_state_ids"])
        raise RuntimeError(f"transfer_state_catalog_incomplete:{missing}")
    flow.mode = MODE_OMNIFLOW
    flow.name = MODE_OMNIFLOW
    flow.env = env
    flow.transition_pause = None
    if empty_store_tempdir is not None:
        flow._empty_store_tempdir = empty_store_tempdir

    def reset(go_home: bool = False) -> None:
        state.update(
            last_result=None,
            step_index=0,
            captured_transfer_states={},
        )
        raw_host.reset(go_home=go_home)

    def set_max_steps(step_budget: int) -> None:
        resolved_budget = min(configured_max_steps, max(1, int(step_budget)))
        state["max_steps"] = resolved_budget
        flow.config = replace(
            flow.config,
            runtime=replace(flow.config.runtime, max_steps=resolved_budget),
        )
        set_planner_max_steps = getattr(planner, "set_max_steps", None)
        if callable(set_planner_max_steps):
            set_planner_max_steps(resolved_budget)

    def set_current_task(
        task_name: str,
        goal: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        task_context = dict(context or {})
        task_parameters = task_context.get("task_parameters")
        if not isinstance(task_parameters, dict):
            task_parameters = {}
        state.update(
            task_name=str(task_name or "").strip(),
            goal=str(goal or "").strip(),
            task_parameters=_json_copy(task_parameters),
        )

    def update_current_task_context(task: Any) -> dict[str, Any]:
        raw_parameters = getattr(task, "params", {})
        parameters = dict(raw_parameters) if isinstance(raw_parameters, dict) else {}
        return {
            "task_parameters": _json_copy(parameters),
        }

    def step(goal: str):
        goal_text = str(goal or "").strip()
        if state["last_result"] is not None:
            return make_agent_result(
                done=True,
                data={
                    "summary": "OmniFlow cycle already completed.",
                    "step_index": int(state["step_index"]),
                    "planner_steps": 0,
                    "source": "planner",
                    "function_id": state["last_result"].function_id,
                    "actions_executed": 0,
                    "fallback": False,
                    "error": None,
                    "done_reason": "omniflow_cycle_already_completed",
                    "finished_content": None,
                },
            )
        state["goal"] = goal_text
        planner_goal = _goal_with_task_parameters(
            goal_text,
            state.get("task_parameters"),
        )
        result = flow.run(planner_goal, experiment=Experiment(name="androidworld"))
        planner_steps = int(result.detail.get("planner_steps") or 0)
        finished_content = str(result.detail.get("finished_content") or "").strip()
        done_reason = str(result.detail.get("done_reason") or "").strip()
        planner_failed = str(result.error or "").startswith("vlm_planner_failed:")
        budget_exhausted = planner_steps >= int(state["max_steps"])
        if planner_failed:
            done_reason = "planner_failed"
        elif budget_exhausted and done_reason not in {
            "finished",
            "abort",
            "waiting_input",
        }:
            done_reason = "max_steps_exceeded"
            result = RunResult(
                False,
                result.function_id,
                result.actions_executed,
                result.model_calls,
                result.fallback_steps,
                "max_steps_exceeded",
                result.final_state,
                {**result.detail, "done_reason": done_reason},
            )
        if done_reason == "finished":
            _emit_official_completion(env, finished_content)
        state["step_index"] = 1
        state["last_result"] = result
        return make_agent_result(
            done=True,
            data={
                "summary": result.error or done_reason or "step completed",
                "step_index": 1,
                "planner_steps": planner_steps,
                "source": "planner",
                "function_id": result.function_id,
                "actions_executed": result.actions_executed,
                "fallback": result.fallback_steps > 0,
                "error": result.error,
                "done_reason": done_reason or (
                    "omniflow_cycle_completed"
                    if result.actions_executed
                    else "planner_failed"
                ),
                "finished_content": finished_content or None,
            },
        )

    def get_captured_transfer_states() -> dict[str, dict[str, Any]]:
        captured = state["captured_transfer_states"]
        return {identifier: _json_copy(captured[identifier]) for identifier in sorted(captured)}

    flow.reset = reset
    flow.set_max_steps = set_max_steps
    flow.set_current_task = set_current_task
    flow.update_current_task_context = update_current_task_context
    flow.step = step
    flow.get_captured_transfer_states = get_captured_transfer_states
    return flow


def _goal_with_task_parameters(goal: str, task_parameters: Any) -> str:
    """Give the Planner public task API values without changing the task goal."""

    if not isinstance(task_parameters, dict) or not task_parameters:
        return str(goal or "").strip()
    public_parameters = {
        str(name): value
        for name, value in task_parameters.items()
        if str(name).strip() and str(name).strip().casefold() != "seed"
    }
    if not public_parameters:
        return str(goal or "").strip()
    encoded = json.dumps(public_parameters, ensure_ascii=False, sort_keys=True)
    return (
        f"{str(goal or '').strip()}\n"
        "Known task parameters are public Function API values. Copy each value "
        "verbatim into the Function arguments; do not translate, approximate, "
        "calculate, normalize, or substitute another representation:\n"
        f"{encoded}"
    ).strip()


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


__all__ = ["MODE_OMNIFLOW", "AndroidWorldHost", "build_agent"]
