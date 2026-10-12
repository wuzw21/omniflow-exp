from __future__ import annotations

from dataclasses import replace
import asyncio
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
from omniflow.core.config import PluginSet
from omniflow.transfer.attribute import attribute_transfer
from omniflow.core.trajectory import state_id
from omniflow.functions.store import FunctionStore
from omniflow.transfer.runtime import (
    TRANSFER_STATE_CATALOG_FILENAME,
    load_transfer_state_catalog,
    transfer_state_coverage,
)
from omniflow.transfer.runtime import (
    capture_transfer_state as _transfer_state,
)
from src.experiment.observation_evidence import canonicalize_run_log_observation
from src.experiment.performance_metrics import PerformanceMetrics
from src.experiment.protocol import MAX_STEPS
from src.integrations.android_world.host import AndroidWorldHost, make_agent_result

MODE_OMNIFLOW = "omniflow"
# Same-harness baselines (docs/BASELINES_SAME_HARNESS.md): identical Planner,
# Host, budget and post-episode validator; only the reuse mechanism differs.
SAME_HARNESS_BASELINES = ("none", "mobilegpt_style", "appagent_style")


def same_harness_baseline() -> str:
    value = os.environ.get("OMNIFLOW_BASELINE", "none")
    if value not in SAME_HARNESS_BASELINES:
        raise ValueError(f"unknown_same_harness_baseline:{value}")
    return value


def load_element_docs(root: str | Path) -> dict[str, dict[str, str]]:
    """Load AppAgent ``demo_docs`` (one ``<element_id>.txt`` dict per element)."""

    import ast

    docs: dict[str, dict[str, str]] = {}
    for path in sorted(Path(root).expanduser().glob("*.txt")):
        payload = ast.literal_eval(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            docs[path.stem] = {str(k): str(v or "").strip() for k, v in payload.items()}
    if not docs:
        raise ValueError(f"element_docs_empty:{root}")
    return docs


def element_docs_for_xml(xml_text: str, docs: dict[str, dict[str, str]]) -> str:
    """Render docs of on-screen elements the way AppAgent's prompt does."""

    import xml.etree.ElementTree as ET

    from src.integrations.appagent import _appagent_element_id

    try:
        root = ET.fromstring(xml_text or "")
    except ET.ParseError:
        return ""
    lines: list[str] = []
    seen: set[str] = set()
    for node in root.iter():
        try:
            element_id = _appagent_element_id(node)
        except ValueError:
            continue
        doc = docs.get(element_id)
        if doc is None or element_id in seen:
            continue
        seen.add(element_id)
        label = node.attrib.get("text") or node.attrib.get("content-desc") or element_id
        usage = "; ".join(f"{k}: {v}" for k, v in doc.items() if v)
        lines.append(f"- {label} [{node.attrib.get('bounds', '')}]: {usage}")
    return "\n".join(lines)


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


def _install_official_completion_checker(
    flow: OmniFlow,
    task: Any,
    env: Any,
) -> None:
    """Offer the AndroidWorld validator as an in-loop oracle.

    ``OmniFlow.set_completion_checker`` discards it unless the run is an
    explicit ``--completion-oracle on`` ablation, so formal runs never see
    ground truth before the single post-episode official evaluation.
    """

    flow.set_completion_checker(lambda: task.is_successful(env))


class _TaskHost:
    def __init__(
        self,
        host: AndroidWorldHost,
        state: dict[str, Any],
        transfer_states: dict[str, dict[str, Any]],
        element_docs: dict[str, dict[str, str]] | None = None,
    ):
        self.host = host
        self.state = state
        self.transfer_states = transfer_states
        self.element_docs = element_docs

    @property
    def env(self) -> Any:
        return self.host.env

    def observe(self, **kwargs: Any) -> Observation:
        return self._capture_observation(self.host.observe(**kwargs))

    def observe_stable(self, **kwargs: Any) -> Observation:
        return self._capture_observation(self.host.observe_stable(**kwargs))

    def take_after_action_observation(self) -> Observation | None:
        observation = self.host.take_after_action_observation()
        return self._capture_observation(observation) if observation is not None else None

    def _capture_observation(self, value: Any) -> Observation:
        observation = Observation.from_value(value)
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
        if self.element_docs:
            docs_text = element_docs_for_xml(identified.xml, self.element_docs)
            if docs_text:
                payload = identified.to_dict()
                payload["extra"] = {**payload.get("extra", {}), "element_docs": docs_text}
                identified = Observation.from_value(payload)
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
    max_steps: int = MAX_STEPS,
    adb_serial: str = "",
    adb_path: str = "",
    post_action_wait_seconds: float = 0.0,
    task_seed: int | None = None,
    evidence_root: str | Path | None = None,
    performance_metrics: PerformanceMetrics | None = None,
    allow_empty_store: bool = False,
    harness: Any | None = None,
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
    if not store_path and planner is None and harness is None:
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
    baseline = same_harness_baseline()
    element_docs = None
    if baseline == "appagent_style":
        docs_root = str(os.environ.get("OMNIFLOW_ELEMENT_DOCS") or "").strip()
        if not docs_root:
            raise ValueError("appagent_style_requires_OMNIFLOW_ELEMENT_DOCS")
        element_docs = load_element_docs(docs_root)
    host = _TaskHost(raw_host, state, transfer_states, element_docs)
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
                checker_enabled=os.environ.get("OMNIFLOW_CHECKER_MODE", "on") == "on",
                # AppAgent-style reuses element docs only, never Functions.
                function_memory_enabled=(
                    os.environ.get("OMNIFLOW_FUNCTION_MEMORY", "on") == "on"
                    and baseline != "appagent_style"
                ),
                function_reentry_enabled=os.environ.get("OMNIFLOW_FUNCTION_REENTRY", "on") == "on",
                completion_oracle_enabled=os.environ.get("OMNIFLOW_COMPLETION_ORACLE", "off") == "on",
            ),
            plugins=PluginSet(
                transfer=attribute_transfer if baseline == "mobilegpt_style" else None,
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
        _install_official_completion_checker(flow, task, env)
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
        from src.integrations.gui_agent_harness import HarnessContext, build_harness
        task_harness = harness or build_harness("builtin")
        result = asyncio.run(task_harness.arun(HarnessContext(
            flow=flow, goal=planner_goal, max_steps=int(state["max_steps"]),
            evidence_root=Path(evidence_root or resolved_store_path.parent),
        )))
        planner_steps = int(result.detail.get("planner_steps") or 0)
        finished_content = str(result.detail.get("finished_content") or "").strip()
        done_reason = str(result.detail.get("done_reason") or "").strip()
        planner_failed = str(result.error or "").startswith("vlm_planner_failed:")
        budget_exhausted = planner_steps >= int(state["max_steps"])
        if planner_failed:
            done_reason = "planner_failed"
        elif budget_exhausted and done_reason not in {
            "finished",
            "function_completed_verified",
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
        if done_reason in {"finished", "function_completed_verified"}:
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
        "Known task parameters are public task values. For parameters declared "
        "by the selected Function input_schema, copy the corresponding value "
        "verbatim into its arguments; omit fields absent from that schema. "
        "Do not translate, approximate, "
        "calculate, normalize, or substitute another representation:\n"
        f"{encoded}"
    ).strip()


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


__all__ = ["MODE_OMNIFLOW", "AndroidWorldHost", "build_agent"]
