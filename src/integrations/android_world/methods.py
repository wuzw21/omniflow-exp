"""Method Adapter registry for the shared AndroidWorld experiment environment."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable

from omniflow.vlm.model_config import resolve_openai_compatible_config

_UNSUPPORTED_SELECTOR_ERROR = (
    "Unsupported AndroidWorld agent selector. Use `omniflow`, `fixed_replay`, "
    "`external:mobilegpt`, `external:appagent`, "
    "`external:appagent_teacher`, or `official:<name>`."
)


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
    planner_timeout_sec: float | None = None
    raw_replay_run_log: str = ""
    appagent_root: str = ""
    appagent_workspace_root: str = ""
    appagent_docs_root: str = ""
    appagent_action_source: str = ""
    appagent_teacher_source: str = ""
    appagent_demo_name: str = ""
    appagent_output_root: str = ""
    task_seed: int | None = None
    evidence_root: str = ""
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
                name="omniflow_replay",
                accepts=lambda selector: selector in {"omniflow", "fixed_replay"},
                build=_build_omniflow_replay,
            ),
            MethodAdapter(
                name="mobilegpt",
                accepts=lambda selector: selector == "external:mobilegpt",
                build=_build_mobilegpt,
            ),
            MethodAdapter(
                name="appagent",
                accepts=lambda selector: selector
                in {"external:appagent", "external:appagent_teacher"},
                build=_build_appagent,
            ),
            MethodAdapter(
                name="official_androidworld",
                accepts=lambda selector: selector.startswith("official:"),
                build=_build_official,
            ),
        )
    )


def _build_omniflow_replay(context: MethodAdapterContext) -> Any:
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
    planner_api_key, planner_base_url = resolve_openai_compatible_config()
    planner = None
    if (
        resolved_planner_model
        or resolved_planner_provider
        or _read_env_bool("OMNIFLOW_ENABLE_ONLINE_PLANNER", False)
    ):
        from omniflow.vlm.planner import VLMPlanner

        planner = VLMPlanner(
            provider=resolved_planner_provider or None,
            model=resolved_planner_model or None,
            api_key=planner_api_key,
            base_url=planner_base_url,
            timeout=resolved_planner_timeout,
        )
    build_kwargs: dict[str, Any] = {
        "env": context.env,
        "store_path": context.store_path,
        "adb_serial": context.adb_serial,
        "adb_path": context.adb_path,
        "task_seed": context.task_seed,
        "evidence_root": context.evidence_root or None,
    }
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


def _build_mobilegpt(context: MethodAdapterContext) -> Any:
    from src.integrations.android_world.mobilegpt_agent import build_mobilegpt_agent

    return build_mobilegpt_agent(
        env=context.env,
        evidence_root=context.evidence_root or None,
    )


def _build_appagent(context: MethodAdapterContext) -> Any:
    from src.integrations.appagent_adapter import (
        AppAgentAndroidWorldAgent,
        AppAgentTeacherAgent,
        OfficialAppAgentRuntime,
    )

    runtime = OfficialAppAgentRuntime(context.appagent_root)
    if context.selector == "external:appagent_teacher":
        if not str(context.appagent_teacher_source or "").strip():
            raise ValueError(
                "external:appagent_teacher requires --appagent-teacher-source"
            )
        if not str(context.appagent_workspace_root or "").strip():
            raise ValueError(
                "external:appagent_teacher requires --appagent-workspace-root"
            )
        return AppAgentTeacherAgent(
            env=context.env,
            official_runtime=runtime,
            teacher_source=context.appagent_teacher_source,
            workspace_root=context.appagent_workspace_root,
            demo_name=context.appagent_demo_name,
        )
    llm_factory = _required_dependency(
        context.appagent_llm_factory,
        "appagent_llm_factory",
    )
    return AppAgentAndroidWorldAgent(
        env=context.env,
        official_runtime=runtime,
        llm=llm_factory(),
        output_root=context.appagent_output_root,
        docs_root=(context.appagent_docs_root or None),
        action_source=(context.appagent_action_source or None),
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
