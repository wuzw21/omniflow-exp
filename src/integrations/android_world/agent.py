from __future__ import annotations

from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import time
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
from omniflow.core.trajectory import (
    OMNIFLOW_RUN_LOG_SCHEMA_VERSION,
    canonicalize_run_log,
    state_id,
)
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
    task_seed: int | None = None,
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
        "task_parameters": {},
        "seed": task_seed,
        "last_result": None,
        "last_run_id": "",
        "last_run_log": None,
        "captured_transfer_states": {},
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
        state["goal"] = goal_text
        started_at_ms = int(time.time() * 1000)
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
        task_name = str(state.get("task_name") or "").strip()
        if not task_name:
            raise ValueError("androidworld_task_name_required")
        steps = _androidworld_run_log_steps(
            trace,
            state["captured_transfer_states"],
        )
        run_log = canonicalize_run_log(
            {
                "schema_version": OMNIFLOW_RUN_LOG_SCHEMA_VERSION,
                "run_id": run_id,
                "task_name": task_name,
                "goal": goal_text,
                "task_parameters": _json_copy(state["task_parameters"]),
                "seed": state["seed"],
                "status": "succeeded" if result.success else "failed",
                "success": result.success,
                "validator": {
                    "official": True,
                    "success": result.success,
                    "reward": 1.0 if result.success else 0.0,
                },
                "provenance": {"kind": "runtime"},
                "started_at_ms": started_at_ms,
                "finished_at_ms": int(time.time() * 1000),
                "steps": steps,
                "diagnostics": {
                    "done_reason": result.error or "goal_completed",
                    "step_count": len(steps),
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
        run_log["validator"] = {
            "official": True,
            "success": bool(success),
            "reward": 1.0 if success else 0.0,
        }
        diagnostics = dict(run_log.get("diagnostics") or {})
        diagnostics["done_reason"] = str(
            done_reason or diagnostics.get("done_reason") or ""
        )
        run_log["diagnostics"] = diagnostics
        run_log = canonicalize_run_log(run_log)
        referenced_state_ids = sorted(
            {
                state_id(observation)
                for step in run_log["steps"]
                for observation in (
                    step["observation"],
                    step.get("next_observation"),
                )
                if isinstance(observation, dict)
            }
        )
        captured = state["captured_transfer_states"]
        captured_transfer_states = {
            state_id: captured[state_id]
            for state_id in sorted(captured)
        }
        missing_state_ids = sorted(
            set(referenced_state_ids) - set(captured_transfer_states)
        )
        transfer_state_audit = {
            "referenced_state_ids": referenced_state_ids,
            "captured_state_ids": sorted(captured_transfer_states),
            "missing_state_ids": missing_state_ids,
            "referenced_state_count": len(referenced_state_ids),
            "captured_state_count": len(captured_transfer_states),
            "missing_state_count": len(missing_state_ids),
            "complete": not missing_state_ids,
        }
        if success and not state["transfer_catalog_preexisting"]:
            catalog_states = {
                state_id: captured_transfer_states[state_id]
                for state_id in referenced_state_ids
                if state_id in captured_transfer_states
            }
            if missing_state_ids:
                raise RuntimeError(
                    "captured_transfer_states_incomplete:"
                    + ",".join(missing_state_ids)
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
            "captured_transfer_states": captured_transfer_states,
            "transfer_state_audit": transfer_state_audit,
        }
        return payload

    flow.reset = reset
    flow.set_max_steps = set_max_steps
    flow.set_current_task = set_current_task
    flow.update_current_task_context = update_current_task_context
    flow.step = step
    flow.save_run_log = save_run_log
    return flow


def _androidworld_run_log_steps(
    trace: list[Any],
    captured_states: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for raw_step in trace:
        if not isinstance(raw_step, dict):
            raise ValueError("omniflow_trace_step_invalid")
        before_id = str(raw_step.get("before_state_id") or "").strip()
        after_id = str(raw_step.get("after_state_id") or before_id).strip()
        before = captured_states.get(before_id)
        after = captured_states.get(after_id)
        if not isinstance(before, dict):
            raise ValueError(f"captured_transfer_state_missing:{before_id}")
        if not isinstance(after, dict):
            raise ValueError(f"captured_transfer_state_missing:{after_id}")
        observation = _androidworld_state_from_transfer_state(before)
        result = raw_step.get("result")
        if not isinstance(result, dict) or not isinstance(
            result.get("success"), bool
        ):
            raise ValueError("omniflow_trace_result_invalid")
        projected: dict[str, Any] = {
            "step_index": len(steps),
            "observation": observation,
            "action": _omniflow_action_to_androidworld(
                raw_step.get("action"),
                observation=observation,
            ),
            "result": {
                "success": result["success"],
                **(
                    {"error": str(result["error"])}
                    if str(result.get("error") or "").strip()
                    else {}
                ),
            },
        }
        if after_id != before_id:
            projected["next_observation"] = _androidworld_state_from_transfer_state(
                after
            )
        metadata = raw_step.get("metadata")
        if isinstance(metadata, dict) and metadata:
            projected["metadata"] = _json_copy(metadata)
        steps.append(projected)
    return steps


def _androidworld_state_from_transfer_state(
    value: dict[str, Any],
) -> dict[str, Any]:
    auxiliaries = {
        key: _json_copy(value[key])
        for key in ("state_id", "package_name", "activity_name", "display")
        if value.get(key) is not None
    }
    return {
        "pixels": None,
        "forest": value.get("xml"),
        "ui_elements": [],
        "auxiliaries": auxiliaries or None,
    }


def _omniflow_action_to_androidworld(
    value: Any,
    *,
    observation: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("omniflow_trace_action_invalid")
    tool = str(value.get("tool") or "").strip()
    args = value.get("args")
    if not isinstance(args, dict):
        raise ValueError("omniflow_trace_action_args_invalid")
    display = observation.get("auxiliaries")
    display = display.get("display") if isinstance(display, dict) else None

    def pixel_point() -> dict[str, int]:
        if not isinstance(display, dict):
            raise ValueError("omniflow_trace_action_display_required")
        try:
            width = int(display["width"])
            height = int(display["height"])
            x = float(args["x"])
            y = float(args["y"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("omniflow_trace_action_point_required") from error
        return {
            "x": max(0, min(width - 1, int(round(x / 1000.0 * width)))),
            "y": max(0, min(height - 1, int(round(y / 1000.0 * height)))),
        }

    if tool == "click":
        return {"action_type": "click", **pixel_point()}
    if tool == "long_press":
        return {"action_type": "long_press", **pixel_point()}
    if tool == "input_text":
        action = {
            "action_type": "input_text",
            "text": str(args.get("text") or ""),
            "clear_text": True,
        }
        if args.get("x") is not None or args.get("y") is not None:
            action.update(pixel_point())
        return action
    if tool == "swipe":
        return {
            "action_type": "scroll",
            "direction": str(args.get("direction") or ""),
        }
    if tool == "open_app":
        app_name = str(
            args.get("package_name") or args.get("app_name") or ""
        ).strip()
        if not app_name:
            raise ValueError("omniflow_trace_open_app_identifier_required")
        return {"action_type": "open_app", "app_name": app_name}
    if tool in {"press_back", "press_home", "press_enter"}:
        return {
            "action_type": {
                "press_back": "navigate_back",
                "press_home": "navigate_home",
                "press_enter": "keyboard_enter",
            }[tool]
        }
    if tool == "press_key":
        key = str(args.get("keycode") or args.get("key") or "").strip().upper()
        key = key.removeprefix("KEYCODE_")
        if key in {"BACK", "NAVIGATE_BACK", "PRESS_BACK"}:
            return {"action_type": "navigate_back"}
        if key in {"HOME", "NAVIGATE_HOME", "PRESS_HOME"}:
            return {"action_type": "navigate_home"}
        if key in {"ENTER", "KEYBOARD_ENTER", "PRESS_ENTER"}:
            return {"action_type": "keyboard_enter"}
        if key == "DELETE":
            key = "DEL"
        if not key:
            raise ValueError("omniflow_trace_keycode_required")
        return {"action_type": "press_keyboard", "keycode": f"KEYCODE_{key}"}
    if tool == "wait":
        return {"action_type": "wait"}
    raise ValueError(f"omniflow_trace_action_unsupported:{tool or 'missing'}")


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


__all__ = ["MODE_OMNIFLOW", "AndroidWorldHost", "build_agent"]
