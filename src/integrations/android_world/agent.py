from __future__ import annotations

from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
from typing import Any
import uuid

from omniflow import (
    FunctionRouter,
    Observation,
    OmniFlow,
    OmniFlowConfig,
    RuntimeSettings,
)
from omniflow.core.config import Experiment
from omniflow.core.trajectory import canonicalize_run_log
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
DEFAULT_RUN_MAX_STEPS = 8


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

    def observe(self, **kwargs: Any) -> Observation:
        observation = Observation.from_value(self.host.observe(**kwargs))
        transfer_state = _transfer_state(observation)
        self.state["captured_transfer_states"][transfer_state["state_id"]] = (
            transfer_state
        )
        identified = Observation.from_value(
            {
                **observation.to_dict(),
                "state_id": transfer_state["state_id"],
            }
        )
        if identified.image_base64:
            self.state["captured_observations"][transfer_state["state_id"]] = (
                _failure_observation(identified)
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
    function_router: FunctionRouter | None = None,
    max_steps: int = DEFAULT_RUN_MAX_STEPS,
    adb_serial: str = "",
    adb_path: str = "",
    post_action_wait_seconds: float = 0.0,
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
    )
    state: dict[str, Any] = {
        "task_name": "",
        "goal": "",
        "last_result": None,
        "last_run_id": "",
        "last_run_log": None,
        "captured_transfer_states": {},
        "captured_observations": {},
        "transfer_catalog_preexisting": transfer_state_path.is_file(),
    }
    transfer_states = load_transfer_state_catalog(transfer_state_path)
    host = _TaskHost(raw_host, state, transfer_states)
    fallback_limit_text = str(
        os.environ.get("OMNIFLOW_MAX_FALLBACK_STEPS") or ""
    ).strip()
    max_fallback_steps = (
        max(0, int(fallback_limit_text)) if fallback_limit_text else None
    )
    flow = OmniFlow(
        resolved_store_path,
        host=host,
        planner=planner,
        function_router=function_router,
        installed_apps={package: package for package in raw_host.installed_packages()},
        config=OmniFlowConfig(
            runtime=RuntimeSettings(
                max_steps=max_steps,
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
            last_run_id="",
            last_run_log=None,
            captured_transfer_states={},
            captured_observations={},
        )
        raw_host.reset(go_home=go_home)

    def set_max_steps(step_budget: int) -> None:
        flow.config = replace(
            flow.config,
            runtime=replace(flow.config.runtime, max_steps=max(1, int(step_budget))),
        )

    def set_current_task(
        task_name: str,
        goal: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        del context
        state.update(
            task_name=str(task_name or "").strip(),
            goal=str(goal or "").strip(),
        )

    def update_current_task_context(task: Any) -> dict[str, Any]:
        del task
        return {}

    def step(goal: str):
        goal_text = str(goal or "").strip()
        state["goal"] = goal_text
        result = flow.run(
            goal_text,
            experiment=Experiment(name="androidworld"),
        )
        finished_content = str(result.detail.get("finished_content") or "").strip()
        if result.success and finished_content:
            json_action = importlib.import_module("android_world.env.json_action")
            env.execute_action(
                json_action.JSONAction(
                    action_type=json_action.ANSWER,
                    text=finished_content,
                )
            )
        run_id = f"run_{uuid.uuid4().hex}"
        trace = list(result.detail.get("trace") or ())
        run_log = canonicalize_run_log(
            {
                "schema_version": "omniflow.canonical_run_log.v1",
                "run_id": run_id,
                "goal": goal_text,
                "status": "succeeded" if result.success else "failed",
                "success": result.success,
                **({"error": result.error} if result.error else {}),
                "steps": trace,
                "diagnostics": {
                    "done_reason": result.error or "goal_completed",
                    "step_count": len(trace),
                    "function_id": result.function_id,
                    "execution_summary": result.execution_summary,
                },
            }
        )
        state.update(last_result=result, last_run_id=run_id, last_run_log=run_log)
        return make_agent_result(
            done=True,
            data={
                "summary": result.error or "goal completed",
                "run_id": run_id,
                "step_index": 0,
                "source": "planner",
                "function_id": result.function_id,
                "actions_executed": result.actions_executed,
                "fallback": result.fallback_steps > 0,
                "error": result.error,
                "done_reason": result.error or "goal_completed",
                "answer": finished_content or None,
            },
        )

    def save_run_log(
        success: bool = False,
        done_reason: str = "",
        auto_import: bool = True,
    ) -> dict[str, Any] | None:
        del auto_import
        run_log = state.get("last_run_log")
        if not isinstance(run_log, dict):
            return None
        run_log = dict(run_log)
        run_log["success"] = bool(success)
        run_log["status"] = "succeeded" if success else "failed"
        diagnostics = dict(run_log.get("diagnostics") or {})
        diagnostics["done_reason"] = str(
            done_reason or diagnostics.get("done_reason") or ""
        )
        run_log["diagnostics"] = diagnostics
        run_log = canonicalize_run_log(run_log)
        if success and not state["transfer_catalog_preexisting"]:
            referenced_state_ids = {
                str(step[field])
                for step in run_log["steps"]
                for field in ("before_state_id", "after_state_id")
            }
            captured = state["captured_transfer_states"]
            catalog_states = {
                state_id: captured[state_id]
                for state_id in sorted(referenced_state_ids)
                if state_id in captured
            }
            missing = sorted(referenced_state_ids - set(catalog_states))
            if missing:
                raise RuntimeError(
                    "captured_transfer_states_incomplete:" + ",".join(missing)
                )
            transfer_state_path.parent.mkdir(parents=True, exist_ok=True)
            with transfer_state_path.open("x", encoding="utf-8") as handle:
                json.dump(
                    {
                        "schema_version": "omniflow.transfer-state-catalog.v1",
                        "run_id": run_log["run_id"],
                        "states": catalog_states,
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.write("\n")
            state["transfer_catalog_preexisting"] = True
        payload = {
            "run_id": run_log["run_id"],
            "goal": run_log["goal"],
            "run_log_summary": {
                "run_id": run_log["run_id"],
                "completed": bool(success),
                "runtime_completed": bool(
                    getattr(state.get("last_result"), "success", False)
                ),
                "official_validator_success": bool(success),
                "step_count": len(run_log.get("steps") or ()),
                "done_reason": diagnostics["done_reason"] or None,
            },
            "run_log": run_log,
        }
        if not success:
            failure_observations = _run_failure_observations(
                run_log,
                state,
            )
            if not any(
                item.get("event") == "terminal_failure"
                for item in failure_observations
            ):
                try:
                    terminal = host.observe(
                        xml=True,
                        screenshot=True,
                        app_info=True,
                    )
                except Exception:  # noqa: BLE001
                    terminal = None
                if terminal is not None and terminal.image_base64:
                    failure_observations.append(
                        {
                            "event": "terminal_failure",
                            **_failure_observation(terminal),
                        }
                    )
            payload["failure_observations"] = failure_observations
        return payload

    flow.reset = reset
    flow.set_max_steps = set_max_steps
    flow.set_current_task = set_current_task
    flow.update_current_task_context = update_current_task_context
    flow.step = step
    flow.save_run_log = save_run_log
    return flow


def _failure_observation(observation: Observation) -> dict[str, Any]:
    item: dict[str, Any] = {
        "state_id": str(observation.extra.get("state_id") or "").strip(),
        "image_base64": str(observation.image_base64 or ""),
    }
    for key, value in (
        ("package_name", observation.package_name),
        ("activity_name", observation.activity_name),
    ):
        text = str(value or "").strip()
        if text:
            item[key] = text
    display = observation.extra.get("display")
    if isinstance(display, dict) and set(display) == {"width", "height"}:
        item["display"] = dict(display)
    return item


def _run_failure_observations(
    run_log: dict[str, Any],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    captured = dict(state.get("captured_observations") or {})
    evidence: list[dict[str, Any]] = []
    for step in run_log.get("steps") or ():
        result = step.get("result")
        if not isinstance(result, dict) or result.get("success") is not False:
            continue
        state_id = str(step.get("before_state_id") or "").strip()
        observation = captured.get(state_id)
        if not isinstance(observation, dict) or not observation.get("image_base64"):
            continue
        evidence.append(
            {
                "event": "action_failure",
                "step_index": int(step["step_index"]),
                "error": str(result.get("error") or "").strip() or None,
                **dict(observation),
            }
        )

    last_result = state.get("last_result")
    final_state = getattr(last_result, "final_state", None)
    terminal = (
        _failure_observation(Observation.from_value(final_state))
        if final_state is not None
        else {}
    )
    state_id = str(terminal.get("state_id") or "").strip()
    if not terminal.get("image_base64") and state_id:
        terminal = dict(captured.get(state_id) or {})
    if terminal.get("image_base64"):
        evidence.append({"event": "terminal_failure", **terminal})
    return evidence


__all__ = ["MODE_OMNIFLOW", "AndroidWorldHost", "build_agent"]
