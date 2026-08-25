"""Method Adapter registry for the shared AndroidWorld experiment environment."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Callable

from omniflow.vlm.model_config import resolve_openai_compatible_config
from src.experiment.protocol import (
    FORMAL_MODEL_BASE_URL,
    FORMAL_MODEL_ENDPOINT_PROFILE,
    MAX_STEPS,
)

_UNSUPPORTED_SELECTOR_ERROR = (
    "Unsupported AndroidWorld agent selector. Use `omniflow`, `fixed_replay`, "
    "or `official:<name>`. External baselines are launched by the official "
    "baseline forwarder."
)

REUSE_METRICS_SCHEMA = "omniflow.androidworld.reuse-metrics.v2"
_PHYSICAL_ACTION_TOOLS = frozenset(
    {"click", "long_press", "input_text", "swipe", "open_app", "press_key"}
)


def reuse_metrics(
    method: str,
    *,
    actions_executed: int = 0,
    canonical_run: dict[str, Any] | None = None,
    mobilegpt_stats: dict[str, Any] | None = None,
    appagent_result: dict[str, Any] | None = None,
    appagent_log: str | Path | None = None,
    source_action_hint: dict[str, Any] | None = None,
    uses_source_action_hints: bool = False,
) -> dict[str, Any]:
    """Return one evidence-backed reuse/utilization metric for a method result."""

    normalized = str(method or "").strip()
    actions = max(0, int(actions_executed or 0))
    physical_trace = _physical_execution_trace(canonical_run)
    physical_actions = len(physical_trace)
    state_changing_trace = [
        step for step in physical_trace if _execution_step_changed_state(step)
    ]
    state_changing_actions = len(state_changing_trace)
    numerator = 0
    denominator = 0
    unit = ""
    evidence = "unavailable"
    artifact_used = False

    if normalized == "fixed_replay":
        numerator = denominator = physical_actions or actions
        unit = "gui_action"
        evidence = "exact_source_replay_actions" if actions else "unavailable"
        artifact_used = actions > 0
    elif normalized == "omniflow":
        numerator = sum(
            1
            for step in physical_trace
            if str((step.get("metadata") or {}).get("function_id") or "").strip()
        )
        denominator = physical_actions
        unit = "gui_action"
        evidence = (
            "exact_function_trace"
            if physical_trace or actions == 0
            else "unavailable"
        )
        artifact_used = bool(canonical_run) or numerator > 0
    elif normalized == "mobilegpt":
        stats = dict(mobilegpt_stats or {})
        denominator = max(0, int(stats.get("memory_lookup_count") or 0))
        numerator = min(
            denominator,
            max(0, int(stats.get("memory_hit_count") or 0)),
        )
        unit = "memory_lookup"
        evidence = "exact_native_memory_events" if denominator else "unavailable"
        artifact_used = denominator > 0
    elif normalized == "appagent":
        result = dict(appagent_result or {})
        if appagent_log and not result:
            result = _appagent_log_usage(Path(appagent_log).expanduser())
        denominator = max(0, int(result.get("decision_round_count") or 0))
        numerator = min(
            denominator,
            max(0, int(result.get("documentation_round_count") or 0)),
        )
        unit = "decision_round"
        evidence = "exact_native_document_rounds" if denominator else "unavailable"
        artifact_used = numerator > 0
    elif normalized == "t3a_hint":
        hint = dict(source_action_hint or {})
        hint_active = bool(
            uses_source_action_hints
            or int(hint.get("rendered_steps") or 0) > 0
        )
        denominator = physical_actions or actions
        numerator = denominator if hint_active else 0
        unit = "gui_action"
        evidence = "exact_goal_hint_injection" if denominator else "unavailable"
        artifact_used = hint_active and denominator > 0

    rate = (
        round(float(numerator) / float(denominator), 6)
        if denominator > 0 and evidence != "unavailable"
        else None
    )
    state_change_rate = (
        round(float(state_changing_actions) / float(physical_actions), 6)
        if physical_actions > 0
        else None
    )
    return {
        "schema_version": REUSE_METRICS_SCHEMA,
        "artifact_used": artifact_used,
        "reuse_numerator": numerator,
        "reuse_denominator": denominator,
        "reuse_rate": rate,
        "reuse_unit": unit,
        "evidence_status": evidence,
        "physical_action_count": physical_actions,
        "state_changing_physical_action_count": state_changing_actions,
        "state_changing_physical_action_rate": state_change_rate,
    }


def reuse_metrics_from_result_row(
    row: dict[str, Any],
    *,
    method: str | None = None,
) -> dict[str, Any]:
    """Derive reuse metrics from a current or immutable historical result row."""

    existing = row.get("reuse_metrics")
    if (
        isinstance(existing, dict)
        and existing.get("schema_version") == REUSE_METRICS_SCHEMA
        and existing.get("evidence_status") != "unavailable"
    ):
        return dict(existing)
    normalized = str(method or row.get("method") or row.get("agent") or "").strip()
    actions = int(
        row.get("episode_actions_executed")
        or row.get("actions_executed")
        or 0
    )
    canonical_run = row.get("canonical_run")
    if not isinstance(canonical_run, dict):
        canonical_run = _json_object(
            row.get("run_log_path") or row.get("target_run_log_path")
        )
    if not isinstance(canonical_run, dict) or not canonical_run:
        evidence_paths = row.get("evidence_paths")
        if isinstance(evidence_paths, (list, tuple)):
            for evidence_path in evidence_paths:
                candidate = Path(str(evidence_path)).expanduser()
                if candidate.name == "run_log.json":
                    canonical_run = _json_object(candidate)
                    if canonical_run:
                        break
    mobilegpt_stats = {
        "memory_lookup_count": row.get("episode_memory_lookup_count"),
        "memory_hit_count": row.get("episode_memory_hit_count"),
    }
    appagent_result = row.get("appagent_result")
    if not isinstance(appagent_result, dict):
        appagent_result = {}
    appagent_log = None
    output_path = str(row.get("output_path") or row.get("run_dir") or "").strip()
    if output_path:
        appagent_log = Path(output_path).expanduser() / "appagent_runtime/appagent_task_log.jsonl"
    return reuse_metrics(
        normalized,
        actions_executed=actions,
        canonical_run=canonical_run,
        mobilegpt_stats=mobilegpt_stats,
        appagent_result=appagent_result,
        appagent_log=appagent_log,
        source_action_hint=(
            row.get("source_action_hint")
            if isinstance(row.get("source_action_hint"), dict)
            else None
        ),
        uses_source_action_hints=bool(row.get("uses_source_action_hints")),
    )


def _canonical_execution_trace(
    canonical_run: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    run = canonical_run if isinstance(canonical_run, dict) else {}
    diagnostics = run.get("diagnostics")
    trace = diagnostics.get("execution_trace") if isinstance(diagnostics, dict) else None
    traced_steps = [step for step in trace or [] if isinstance(step, dict)]
    if traced_steps:
        return traced_steps
    # Planner-backed methods retain their detailed attempts in
    # diagnostics.execution_trace.  Exact replay instead records the physical
    # actions directly as canonical RunLog steps, so an empty diagnostics trace
    # must fall back to those steps rather than reporting zero physical work.
    steps = run.get("steps")
    return [step for step in steps or [] if isinstance(step, dict)]


def _physical_execution_trace(
    canonical_run: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    physical: list[dict[str, Any]] = []
    for step in _canonical_execution_trace(canonical_run):
        action = step.get("action")
        result = step.get("result")
        tool = (
            str(action.get("tool") or action.get("action_type") or "")
            if isinstance(action, dict)
            else ""
        )
        if (
            tool in _PHYSICAL_ACTION_TOOLS
            and isinstance(result, dict)
            and result.get("success") is True
        ):
            physical.append(step)
    return physical


def _execution_step_changed_state(step: dict[str, Any]) -> bool:
    metadata = step.get("metadata")
    effect = metadata.get("action_effect") if isinstance(metadata, dict) else None
    if isinstance(effect, dict) and "state_changed" in effect:
        return effect.get("state_changed") is True
    observation = step.get("observation")
    next_observation = step.get("next_observation")
    if not isinstance(observation, dict) or not isinstance(next_observation, dict):
        return False
    before_xml = str(observation.get("xml") or "").strip()
    after_xml = str(next_observation.get("xml") or "").strip()
    return bool(before_xml and after_xml and before_xml != after_xml)


def _appagent_log_usage(path: Path) -> dict[str, int]:
    decision_round_count = 0
    documentation_round_count = 0
    if not path.is_file():
        return {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or "round" not in row or "response" not in row:
            continue
        decision_round_count += 1
        if row.get("visible_document_uids"):
            documentation_round_count += 1
    return {
        "decision_round_count": decision_round_count,
        "documentation_round_count": documentation_round_count,
    }


def _json_object(value: Any) -> dict[str, Any]:
    path_text = str(value or "").strip()
    if not path_text:
        return {}
    path = Path(path_text).expanduser()
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


@dataclass(frozen=True)
class MethodAdapterContext:
    """Inputs shared by every launcher-facing Method Adapter."""

    selector: str
    env: Any
    store_path: str
    adb_serial: str
    adb_path: str = ""
    planner_provider: str = ""
    planner_model: str = ""
    model_endpoint_profile: str = ""
    planner_timeout_sec: float | None = None
    max_steps: int = MAX_STEPS
    raw_replay_run_log: str = ""
    appagent_root: str = ""
    appagent_workspace_root: str = ""
    appagent_docs_root: str = ""
    appagent_teacher_source: str = ""
    appagent_name: str = ""
    appagent_output_root: str = ""
    task_seed: int | None = None
    evidence_root: str = ""
    performance_metrics: Any | None = None
    build_omniflow_agent: Callable[..., Any] | None = None
    apply_fixed_replay: Callable[..., Any] | None = None
    build_official_agent: Callable[..., Any] | None = None
    appagent_llm_factory: Callable[[], Any] | None = None


@dataclass(frozen=True)
class MethodAdapter:
    """One concrete adapter at the method-selection seam."""

    name: str
    accepts: Callable[[str], bool]
    build: Callable[[MethodAdapterContext], Any]


class MethodAdapterRegistry:
    """Resolve exactly one Method Adapter and hide selector dispatch."""

    def __init__(self, adapters: tuple[MethodAdapter, ...]) -> None:
        if not adapters:
            raise ValueError("androidworld_method_adapters_required")
        self._adapters = adapters

    def build(self, context: MethodAdapterContext) -> Any:
        matches = [
            adapter for adapter in self._adapters if adapter.accepts(context.selector)
        ]
        if not matches:
            raise ValueError(_UNSUPPORTED_SELECTOR_ERROR)
        if len(matches) != 1:
            names = ",".join(adapter.name for adapter in matches)
            raise RuntimeError(
                f"androidworld_method_adapter_ambiguous:{context.selector}:{names}"
            )
        return matches[0].build(context)


def default_method_adapter_registry() -> MethodAdapterRegistry:
    """Return the one registry used by the AndroidWorld launcher."""
    return MethodAdapterRegistry(
        (
            MethodAdapter(
                name="omniflow",
                accepts=lambda selector: selector in {"omniflow", "fixed_replay"},
                build=_build_omniflow,
            ),
            MethodAdapter(
                name="official_androidworld",
                accepts=lambda selector: selector.startswith("official:"),
                build=_build_official,
            ),
        )
    )


def _build_omniflow(context: MethodAdapterContext) -> Any:
    build_agent = _required_dependency(
        context.build_omniflow_agent,
        "build_omniflow_agent",
    )
    resolved_planner_model = str(
        context.planner_model or os.environ.get("OMNIFLOW_PLANNER_MODEL") or ""
    ).strip()
    resolved_planner_provider = str(
        context.planner_provider or os.environ.get("OMNIFLOW_PLANNER_PROVIDER") or ""
    ).strip()
    resolved_planner_timeout = float(
        context.planner_timeout_sec
        or os.environ.get("OMNIFLOW_PLANNER_TIMEOUT_SEC")
        or 60.0
    )
    resolved_endpoint_profile = (
        str(context.model_endpoint_profile or FORMAL_MODEL_ENDPOINT_PROFILE).strip()
        or FORMAL_MODEL_ENDPOINT_PROFILE
    )
    planner = None
    if (
        resolved_planner_model
        or resolved_planner_provider
        or _read_env_bool("OMNIFLOW_ENABLE_ONLINE_PLANNER", False)
    ) and context.selector != "fixed_replay":
        from omniflow.vlm.planner import VLMPlanner

        planner_api_key, planner_base_url = resolve_openai_compatible_config(
            profile=resolved_endpoint_profile,
            base_url=(
                FORMAL_MODEL_BASE_URL
                if resolved_endpoint_profile == FORMAL_MODEL_ENDPOINT_PROFILE
                else None
            ),
        )
        planner = VLMPlanner(
            provider=resolved_planner_provider or "openai",
            model=resolved_planner_model,
            api_key=planner_api_key,
            base_url=planner_base_url,
            timeout=resolved_planner_timeout,
            max_steps=context.max_steps,
        )
    build_kwargs: dict[str, Any] = {
        "env": context.env,
        "store_path": context.store_path,
        "adb_serial": context.adb_serial,
        "adb_path": context.adb_path,
        "max_steps": context.max_steps,
        "task_seed": context.task_seed,
        "evidence_root": context.evidence_root or None,
        "performance_metrics": context.performance_metrics,
    }
    if context.selector == "fixed_replay" or not str(context.store_path).strip():
        build_kwargs["allow_empty_store"] = True
    if planner is not None:
        build_kwargs["planner"] = planner
    built_agent = build_agent(**build_kwargs)
    if context.selector != "fixed_replay":
        return built_agent
    if not str(context.raw_replay_run_log or "").strip():
        raise ValueError("fixed_replay requires --raw-replay-run-log")
    apply_fixed_replay = _required_dependency(
        context.apply_fixed_replay,
        "apply_fixed_replay",
    )
    return apply_fixed_replay(
        built_agent,
        run_log_json_path=str(context.raw_replay_run_log).strip(),
        adb_path=context.adb_path,
    )


def _build_appagent(context: MethodAdapterContext) -> Any:
    del context
    raise ValueError(
        "appagent_is_external_only: use scripts/exp/run_androidworld.sh "
        "with OMNIFLOW_ANDROIDWORLD_METHOD=appagent"
    )


def _build_official(context: MethodAdapterContext) -> Any:
    official_agent_name = str(
        context.selector.split(":", maxsplit=1)[1] or ""
    ).strip()
    if not official_agent_name:
        raise ValueError(
            "--agent official:<name> requires one upstream AndroidWorld agent name"
        )
    build_official_agent = _required_dependency(
        context.build_official_agent,
        "build_official_agent",
    )
    return build_official_agent(
        env=context.env,
        official_agent_name=official_agent_name,
        model_name=context.planner_model,
    )


def _required_dependency(value: Any, name: str) -> Any:
    if not callable(value):
        raise RuntimeError(f"androidworld_method_dependency_missing:{name}")
    return value


def _read_env_bool(name: str, default: bool) -> bool:
    value = str(os.environ.get(name) or "").strip()
    if not value:
        return default
    return value.lower() in {"1", "true", "yes", "on"}
