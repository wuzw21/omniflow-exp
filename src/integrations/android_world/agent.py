from __future__ import annotations

from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
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
from omniflow.transfer.runtime import (
    TRANSFER_STATE_CATALOG_FILENAME,
    load_transfer_state_catalog,
    transfer_state_coverage,
)
from omniflow.transfer.runtime import (
    capture_transfer_state as _transfer_state,
)
from src.integrations.android_world.host import AndroidWorldHost, make_agent_result

MODE_OMNIFLOW = "omniflow"


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
        if not isinstance(official_state, dict) or set(official_state) != {
            "pixels",
            "forest",
            "ui_elements",
            "auxiliaries",
        }:
            raise ValueError("androidworld_state_snapshot_required")
        official_state = _json_copy(official_state)
        official_state_id = state_id(official_state)
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

    def get_state(self, state_id: str) -> Observation | None:
        value = self.transfer_states.get(str(state_id or "").strip())
        return Observation.from_value(value) if value is not None else None


def build_agent(
    *,
    env: Any,
    store_path: str | None = None,
    runtime: Any | None = None,
    planner: Any | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    adb_serial: str = "",
    adb_path: str = "",
    post_action_wait_seconds: float = 0.0,
    task_seed: int | None = None,
    evidence_root: str | Path | None = None,
) -> OmniFlow:
    if env is None:
        raise TypeError("build_agent requires env parameter")
    del runtime
    default_store = (
        Path(os.environ.get("OMNIFLOW_RUNTIME_DIR") or "runtime") / "omniflow.json"
    )
    resolved_store_path = Path(store_path or default_store).expanduser().resolve()
    transfer_state_path = resolved_store_path.parent / TRANSFER_STATE_CATALOG_FILENAME
    raw_host = AndroidWorldHost(
        env,
        adb_serial=adb_serial,
        adb_path=adb_path,
        post_action_wait_seconds=post_action_wait_seconds,
        evidence_root=evidence_root,
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
        installed_apps={package: package for package in raw_host.installed_packages()},
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
        return {"task_parameters": _json_copy(parameters)}

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
                    "answer": None,
                },
            )
        state["goal"] = goal_text
        result = flow.run(
            goal_text,
            experiment=Experiment(name="androidworld"),
        )
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
        if done_reason == "finished" and finished_content:
            json_action = importlib.import_module("android_world.env.json_action")
            env.execute_action(
                json_action.JSONAction(
                    action_type=json_action.ANSWER,
                    text=finished_content,
                )
            )
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
                "answer": finished_content or None,
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


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


__all__ = ["MODE_OMNIFLOW", "AndroidWorldHost", "build_agent"]
