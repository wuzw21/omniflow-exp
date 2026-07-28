#!/usr/bin/env python3
"""Run the pinned official Mobile-Agent-V3 AndroidWorld implementation.

This is deliberately a thin evidence adapter: the external repository owns the
Manager/Executor/ActionReflector/Notetaker policy, action conversion, episode
loop, and AndroidWorld validator.  OmniFlow only seals provenance and records
per-request usage/timing in its common result schema.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.integrations.android_world.setup_compat import (  # noqa: E402
    patch_androidworld_setup_click_retry,
)
from src.integrations.mobile_agent_v3_adapter import (  # noqa: E402
    GUI_OWL_7B_MODEL_REVISION,
    MOBILE_AGENT_V3_OFFICIAL_REVISION,
    MobileAgentV3UsageLedger,
    audit_mobile_agent_v3_call_evidence,
    count_mobile_agent_v3_actions,
    inspect_gui_owl_model,
)


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _to_serializable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(item) for item in value]
    enum_value = getattr(value, "value", None)
    if enum_value is not None and enum_value is not value:
        return _to_serializable(enum_value)
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return {
            str(key): _to_serializable(item)
            for key, item in attributes.items()
            if not str(key).startswith("_")
        }
    return str(value)


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        _to_serializable(value),
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _audit_official_checkout(root: Path, expected_revision: str) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(f"mobile_agent_v3_root_missing:{root}")
    revision = _git_output(root, "rev-parse", "HEAD")
    if revision != expected_revision:
        raise ValueError(
            "mobile_agent_v3_revision_mismatch:"
            f"expected={expected_revision}:actual={revision}"
        )
    tracked_diff = _git_output(root, "status", "--short", "--untracked-files=no")
    if tracked_diff:
        raise ValueError(f"mobile_agent_v3_tracked_checkout_dirty:{tracked_diff}")
    code_root = root / "Mobile-Agent-v3" / "android_world_v3"
    critical = (
        "run_ma3.py",
        "android_world/agents/infer_ma3.py",
        "android_world/agents/mobile_agent_v3.py",
        "android_world/agents/mobile_agent_v3_agent.py",
        "android_world/agents/new_json_action.py",
        "android_world/suite_utils.py",
    )
    files: list[dict[str, Any]] = []
    for relative in critical:
        path = code_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"mobile_agent_v3_official_file_missing:{path}")
        files.append(
            {
                "relative_path": relative,
                "sha256": _file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "repository_root": str(root),
        "code_root": str(code_root),
        "revision": revision,
        "tracked_checkout_clean": True,
        "critical_files": files,
    }


def _audit_model_endpoint(base_url: str, api_key: str, model: str) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/models"
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=10.0) as response:
        payload = json.loads(response.read().decode("utf-8"))
    model_ids = [
        str(item.get("id") or "")
        for item in list(payload.get("data") or [])
        if isinstance(item, dict)
    ]
    if model not in model_ids:
        raise ValueError(
            f"mobile_agent_v3_model_not_served:requested={model}:available={model_ids}"
        )
    return {
        "url": url,
        "requested_model": model,
        "available_models": model_ids,
        "wall_sec": round(time.perf_counter() - started, 6),
    }


def _response_metadata(response: Any) -> dict[str, Any]:
    choices = list(getattr(response, "choices", None) or [])
    first = choices[0] if choices else None
    return {
        "id": str(getattr(response, "id", "") or ""),
        "model": str(getattr(response, "model", "") or ""),
        "created": getattr(response, "created", None),
        "system_fingerprint": str(
            getattr(response, "system_fingerprint", "") or ""
        ),
        "finish_reason": str(getattr(first, "finish_reason", "") or ""),
    }


def _response_usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    model_dump = getattr(usage, "model_dump", None)
    if callable(model_dump):
        payload = model_dump(mode="json")
        return dict(payload) if isinstance(payload, dict) else {}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
        "completion_tokens": getattr(usage, "completion_tokens", 0),
        "total_tokens": getattr(usage, "total_tokens", 0),
    }


def _instrumented_wrapper_class(base_class: type, error_text: str) -> type:
    class InstrumentedGUIOwlWrapper(base_class):
        def __init__(self, *args: Any, usage_ledger: MobileAgentV3UsageLedger, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self._usage_ledger = usage_ledger

        def predict_mm(
            self,
            text_prompt: str,
            images: list[Any],
            messages: Any = None,
        ) -> tuple[str, Any, Any]:
            if messages is None:
                payload = [
                    {
                        "role": "user",
                        "content": [{"text": text_prompt}],
                    }
                ]
                for image in images:
                    payload[0]["content"].append({"image": image})
            else:
                payload = messages
            payload = self.convert_messages_format_to_openaiurl(payload)

            counter = self.max_retry
            wait_seconds = self.RETRY_WAITING_SECONDS
            while counter > 0:
                started = time.perf_counter()
                try:
                    response = self.bot.chat.completions.create(
                        model=self.model,
                        messages=payload,
                        **{},
                    )
                    output = response.choices[0].message.content
                    self._usage_ledger.record_call(
                        prompt=text_prompt,
                        images=images,
                        response_text=str(output or ""),
                        response_metadata=_response_metadata(response),
                        usage=_response_usage(response),
                        wall_sec=time.perf_counter() - started,
                        ok=True,
                    )
                    return output, payload, response
                except Exception as exc:  # noqa: BLE001 - preserve official retry loop
                    self._usage_ledger.record_call(
                        prompt=text_prompt,
                        images=images,
                        response_text="",
                        response_metadata={"model": str(self.model)},
                        usage={},
                        wall_sec=time.perf_counter() - started,
                        ok=False,
                        error=f"{exc.__class__.__name__}: {exc}",
                    )
                    time.sleep(wait_seconds)
                    counter -= 1
                    print("Error calling LLM, will retry soon...", flush=True)
                    print(exc, flush=True)
            return error_text, None, None

    InstrumentedGUIOwlWrapper.__name__ = "InstrumentedGUIOwlWrapper"
    return InstrumentedGUIOwlWrapper


def _rehydrate_task_params(params: dict[str, object]) -> dict[str, object]:
    hydrated = dict(params)
    serialized_rows = [
        row
        for key in ("row_objects", "noise_row_objects")
        for row in (
            hydrated.get(key) if isinstance(hydrated.get(key), list) else []
        )
        if isinstance(row, dict)
    ]
    if not serialized_rows:
        return hydrated
    from android_world.task_evals.utils import sqlite_schema_utils

    def hydrate_row(row: object) -> object:
        if not isinstance(row, dict):
            return row
        keys = set(row)
        if keys & {"start_ts", "end_ts", "repeat_rule", "event_type"}:
            allowed = {
                "start_ts", "end_ts", "title", "location", "description",
                "repeat_interval", "repeat_rule", "reminder_1_minutes",
                "reminder_2_minutes", "reminder_3_minutes", "reminder_1_type",
                "reminder_2_type", "reminder_3_type", "repeat_limit",
                "repetition_exceptions", "attendees", "import_id", "time_zone",
                "flags", "event_type", "parent_id", "last_updated", "source",
                "availability", "color", "type", "id",
            }
            return sqlite_schema_utils.CalendarEvent(
                **{key: row[key] for key in allowed if key in row}
            )
        if keys & {"ingredients", "directions", "preparationTime", "recipeId"}:
            allowed = {
                "title", "description", "servings", "preparationTime", "source",
                "ingredients", "directions", "favorite", "imageName", "recipeId",
            }
            return sqlite_schema_utils.Recipe(
                **{key: row[key] for key in allowed if key in row}
            )
        if keys & {"amount", "expense_id", "created_date", "modified_date"}:
            allowed = {
                "name", "amount", "category", "note", "created_date",
                "modified_date", "expense_id",
            }
            return sqlite_schema_utils.Expense(
                **{key: row[key] for key in allowed if key in row}
            )
        return row

    for key in ("row_objects", "noise_row_objects"):
        rows = hydrated.get(key)
        if isinstance(rows, list):
            hydrated[key] = [hydrate_row(row) for row in rows]
    return hydrated


def _setup_apps_for_task(
    setup_module: Any,
    *,
    env: Any,
    task_type: type,
    task_name: str,
) -> None:
    setup_apps: list[type] = []
    seen_setup_apps: set[type] = set()
    for app_name in tuple(getattr(task_type, "app_names", ()) or ()):
        app_setup = setup_module.get_app_mapping(str(app_name))
        if app_setup is not None and app_setup not in seen_setup_apps:
            setup_apps.append(app_setup)
            seen_setup_apps.add(app_setup)
    for app_setup in setup_module.get_app_list_to_setup([task_name]) or ():
        if app_setup not in seen_setup_apps:
            setup_apps.append(app_setup)
            seen_setup_apps.add(app_setup)
    setup_module.setup_apps(
        env,
        app_list=tuple(setup_apps) if setup_apps else None,
    )


def _bounded_step_budget(
    official_allocator: Any,
    task_complexity: float,
    max_steps: int,
) -> int:
    """Cap AndroidWorld's official dynamic budget at the campaign limit."""

    cap = int(max_steps)
    if cap <= 0:
        raise ValueError("mobile_agent_v3_max_steps_must_be_positive")
    official_budget = int(official_allocator(task_complexity))
    if official_budget <= 0:
        raise ValueError("mobile_agent_v3_official_step_budget_invalid")
    return min(official_budget, cap)


def _write_summary(
    output_root: Path,
    *,
    row: dict[str, Any],
    checkpoint_dir: Path,
    runner_wall_sec: float,
) -> None:
    success = bool(row["androidworld_validator_result"]["success"])
    summary = {
        "schema_version": "omniflow.androidworld_run_summary.v2",
        "agent": "official:mobile_agent_v3",
        "tasks_requested": [row["task_name"]],
        "task_results_path": str(output_root / "task_results.jsonl"),
        "checkpoint_dir": str(checkpoint_dir),
        "task_count": 1,
        "official_validator_task_count": 1,
        "official_validator_success_count": int(success),
        "official_validator_failure_count": int(not success),
        "official_validator_success_rate": float(success),
        "official_validator_coverage_rate": 1.0,
        "duration_ms": round(float(row["duration_ms"]), 3),
        "avg_duration_ms": round(float(row["duration_ms"]), 3),
        "runner_wall_sec": round(float(runner_wall_sec), 6),
        "actions_executed": int(row["actions_executed"]),
        "avg_actions_per_task": float(row["actions_executed"]),
        "model_calls": int(row["model_calls"]),
        "prompt_tokens": int(row["prompt_tokens"]),
        "completion_tokens": int(row["completion_tokens"]),
        "total_tokens": int(row["total_tokens"]),
        "avg_tokens_per_task": float(row["total_tokens"]),
        "per_task": [
            {
                "task_name": row["task_name"],
                "goal": row["goal"],
                "official_validator_used": True,
                "official_validator_success": success,
                "duration_ms": round(float(row["duration_ms"]), 3),
                "actions_executed": int(row["actions_executed"]),
                "model_calls": int(row["model_calls"]),
                "prompt_tokens": int(row["prompt_tokens"]),
                "completion_tokens": int(row["completion_tokens"]),
                "total_tokens": int(row["total_tokens"]),
                "error": row.get("error"),
            }
        ],
    }
    (output_root / "summary.json").write_text(
        json.dumps(_to_serializable(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mobile-agent-v3-root", required=True)
    parser.add_argument(
        "--official-revision", default=MOBILE_AGENT_V3_OFFICIAL_REVISION
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--model-revision", default=GUI_OWL_7B_MODEL_REVISION)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-random-seed", type=int, required=True)
    parser.add_argument("--fixed-task-seed", action="store_true")
    parser.add_argument("--task-params-json", default="")
    parser.add_argument("--console-port", type=int, required=True)
    parser.add_argument("--adb-path", default="")
    parser.add_argument("--perform-emulator-setup", action="store_true")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    runner_started = time.perf_counter()
    output_root = Path(args.output_path).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    checkout_root = Path(args.mobile_agent_v3_root).expanduser().resolve()
    checkout_audit = _audit_official_checkout(
        checkout_root, str(args.official_revision)
    )
    model_audit = inspect_gui_owl_model(
        args.model_root,
        revision=str(args.model_revision),
    )
    endpoint_audit = _audit_model_endpoint(
        str(args.base_url), str(args.api_key), str(args.model)
    )
    provenance = {
        "schema_version": "omniflow.mobile-agent-v3-provenance.v1",
        "official_checkout": checkout_audit,
        "model_snapshot": model_audit,
        "model_endpoint": endpoint_audit,
        "runtime_packages": {
            "qwen-vl-utils": importlib.metadata.version("qwen-vl-utils"),
            "openai": importlib.metadata.version("openai"),
            "torch": importlib.metadata.version("torch"),
            "torchvision": importlib.metadata.version("torchvision"),
        },
        "official_agent_components": [
            "Manager", "Executor", "ActionReflector", "Notetaker"
        ],
        "uses_source_runlog": False,
        "uses_omniflow_function": False,
        "uses_source_action_hints": False,
        "max_steps_cap": int(args.max_steps),
        "created_at": _utc_now_iso(),
    }
    provenance_path = output_root / "mobile_agent_v3_provenance.json"
    provenance_path.write_text(
        json.dumps(_to_serializable(provenance), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    official_code_root = Path(checkout_audit["code_root"])
    sys.path.insert(0, str(official_code_root))
    from android_world import checkpointer as checkpointer_lib
    from android_world import registry, suite_utils
    from android_world.agents import infer_ma3, mobile_agent_v3
    from android_world.env import env_launcher
    from android_world.env import tools as android_world_tools
    from android_world.env.setup_device import setup as aw_setup

    patch_androidworld_setup_click_retry(android_world_tools)

    usage_ledger = MobileAgentV3UsageLedger(
        output_root / "mobile_agent_v3_calls.jsonl"
    )
    wrapper_class = _instrumented_wrapper_class(
        infer_ma3.GUIOwlWrapper,
        infer_ma3.ERROR_CALLING_LLM,
    )
    env = None
    checkpoint_dir = output_root / "checkpoints"
    trajectory_dir = output_root / "trajectory"
    try:
        task_registry = registry.TaskRegistry()
        family = registry.TaskRegistry.ANDROID_WORLD_FAMILY
        task_types = task_registry.get_registry(family=family)
        if args.task not in task_types:
            raise ValueError(f"mobile_agent_v3_task_not_registered:{args.task}")
        env = env_launcher.load_and_setup_env(
            console_port=int(args.console_port),
            emulator_setup=False,
            adb_path=str(args.adb_path or ""),
            grpc_port=int(args.console_port) + 3000,
        )
        if args.perform_emulator_setup:
            task_type = task_types[args.task]
            _setup_apps_for_task(
                aw_setup,
                env=env,
                task_type=task_type,
                task_name=args.task,
            )
        suite = suite_utils.create_suite(
            task_types,
            n_task_combinations=1,
            seed=int(args.task_random_seed),
            tasks=[args.task],
            use_identical_params=bool(args.fixed_task_seed),
        )
        if str(args.task_params_json or "").strip():
            parsed = json.loads(args.task_params_json)
            if not isinstance(parsed, dict):
                raise ValueError("mobile_agent_v3_task_params_must_be_object")
            task_type = task_types[args.task]
            task_type.set_device_time(env)
            suite[args.task] = [task_type(_rehydrate_task_params(parsed))]
        suite.suite_family = family
        task_instance = suite[args.task][0]
        task_params = _to_serializable(dict(task_instance.params or {}))
        task_params_sha256 = _stable_hash(task_params)
        official_allocate_step_budget = suite_utils._allocate_step_budget
        official_step_budget = int(
            official_allocate_step_budget(task_instance.complexity)
        )
        effective_step_budget = _bounded_step_budget(
            official_allocate_step_budget,
            task_instance.complexity,
            int(args.max_steps),
        )
        suite_utils._allocate_step_budget = lambda _complexity: effective_step_budget

        model_wrapper = wrapper_class(
            str(args.api_key),
            str(args.base_url),
            str(args.model),
            usage_ledger=usage_ledger,
        )
        agent = mobile_agent_v3.MobileAgentV3_M3A(
            env,
            model_wrapper,
            output_path=str(trajectory_dir),
        )
        agent.get_task_name(suite)
        agent.transition_pause = None
        agent.name = "mobile_agent_v3"
        started_at = _utc_now_iso()
        try:
            episodes = suite_utils.run(
                suite,
                agent,
                checkpointer=checkpointer_lib.IncrementalCheckpointer(
                    str(checkpoint_dir)
                ),
                demo_mode=False,
                return_full_episode_data=True,
            )
        finally:
            suite_utils._allocate_step_budget = official_allocate_step_budget
        if len(episodes) != 1:
            raise RuntimeError(
                f"mobile_agent_v3_expected_one_episode:actual={len(episodes)}"
            )
        episode = episodes[0]
    finally:
        if env is not None:
            env.close()

    usage = usage_ledger.get_usage_summary()
    call_evidence_audit = audit_mobile_agent_v3_call_evidence(
        usage_ledger.calls_jsonl,
        expected_usage=usage,
    )
    call_evidence_audit_path = (
        output_root / "mobile_agent_v3_call_evidence_audit.json"
    )
    call_evidence_audit_path.write_text(
        json.dumps(call_evidence_audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    success_reward = float(episode.get("is_successful") or 0.0)
    success = success_reward > 0.5
    error_text = str(episode.get("exception_info") or "").strip() or None
    row = {
        "task_name": str(episode.get("task_template") or args.task),
        "goal": str(episode.get("goal") or getattr(task_instance, "goal", "")),
        "agent": "official:mobile_agent_v3",
        "task_random_seed": int(args.task_random_seed),
        "task_params": task_params,
        "task_params_sha256": task_params_sha256,
        "state_backend": "androidworld_official_mobile_agent_v3_fork",
        "action_backend": "androidworld_official_mobile_agent_v3_fork",
        "native_androidworld_agent_io": True,
        "success": success,
        "official_validator_used": True,
        "androidworld_validator_result": {
            "success": success,
            "reward": success_reward,
            "error": error_text,
            "uses_androidworld_official_validator": True,
            "validator": "androidworld_official",
        },
        "response_acceptance": {"generic": success, "androidworld": success},
        "started_at": started_at,
        "duration_ms": max(0.0, float(episode.get("run_time") or 0.0) * 1000.0),
        "step_count": int(episode.get("episode_length") or 0),
        "official_step_budget": official_step_budget,
        "max_steps_cap": int(args.max_steps),
        "effective_step_budget": effective_step_budget,
        "actions_executed": count_mobile_agent_v3_actions(
            episode.get("episode_data")
        ),
        "model_calls": int(usage["model_calls"]),
        "fallback_steps": 0,
        "prompt_tokens": int(usage["prompt_tokens"]),
        "completion_tokens": int(usage["completion_tokens"]),
        "total_tokens": int(usage["total_tokens"]),
        "token_usage_status": str(usage["token_usage_status"]),
        "model": str(args.model),
        "model_base_url": str(args.base_url),
        "artifact_kind": "checkpoint",
        "artifact_ref": str(checkpoint_dir),
        "error": error_text,
        "llm_usage": usage,
        "llm_evidence_audit": {
            **call_evidence_audit,
            "audit_path": str(call_evidence_audit_path),
            "audit_sha256": _file_sha256(call_evidence_audit_path),
        },
        "mobile_agent_v3_provenance": {
            "official_revision": str(args.official_revision),
            "model_revision": str(args.model_revision),
            "provenance_path": str(provenance_path),
            "provenance_sha256": _file_sha256(provenance_path),
            "official_framework_unmodified": True,
        },
    }
    (output_root / "task_results.jsonl").write_text(
        json.dumps(_to_serializable(row), ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    runner_wall_sec = time.perf_counter() - runner_started
    _write_summary(
        output_root,
        row=row,
        checkpoint_dir=checkpoint_dir,
        runner_wall_sec=runner_wall_sec,
    )
    if usage["token_usage_status"] != "tracked":
        print(
            "[mobile-agent-v3] rejected: incomplete token accounting "
            f"status={usage['token_usage_status']}",
            file=sys.stderr,
        )
        return 3
    print(
        "[mobile-agent-v3] "
        f"task={row['task_name']} success={int(success)} "
        f"calls={usage['model_calls']} tokens={usage['total_tokens']} "
        f"episode_sec={row['duration_ms'] / 1000.0:.3f} "
        f"runner_wall_sec={runner_wall_sec:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
