"""B-MoCA end-to-end episode runner (split out of run_episode).

Moved verbatim from ``src.integrations.android_world.run_episode``; that module re-exports these names.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any
from src.experiment.observation_evidence import persist_target_run_evidence


def _run_bmoca_e2e(args: argparse.Namespace) -> int:
    """Use the normal OmniFlow runtime against the B-MoCA Host adapter."""

    from src.experiment.observation_evidence import persist_target_run_evidence
    from src.integrations.bmoca import (
        BMocaEnvironmentConfig,
        discover_bmoca_episodes,
        open_bmoca_episode,
    )

    method = str(args.agent or "").strip()
    if method not in {"ours_replay", "skilldroid_replay"}:
        raise ValueError(f"bmoca_method_unsupported:{method}")
    selected_tasks = [item.strip() for item in str(args.tasks).split(",") if item.strip()]
    if len(selected_tasks) != 1:
        raise ValueError("bmoca_e2e_requires_exactly_one_task")
    store_path = Path(args.store_path).expanduser().resolve()
    reuse_memory_path = Path(args.reuse_memory_path).expanduser().resolve()
    if method == "ours_replay":
        if not store_path.is_file():
            raise FileNotFoundError(f"bmoca_function_store_missing:{store_path}")
    elif not reuse_memory_path.exists():
        raise FileNotFoundError(
            f"bmoca_reuse_memory_missing:{reuse_memory_path}"
        )
    environment_ids = tuple(
        item.strip() for item in str(args.environment_ids).split(",") if item.strip()
    )
    if len(environment_ids) != 1:
        raise ValueError("bmoca_single_result_requires_exactly_one_environment")
    ports = (
        int(args.appium_port),
        int(args.appium_system_port),
        int(args.emulator_console_port),
        int(args.emulator_adb_port),
        int(args.emulator_grpc_port),
    )
    if any(port <= 0 for port in ports) or len(set(ports)) != len(ports):
        raise ValueError("bmoca_isolated_ports_invalid")
    if int(args.emulator_adb_port) != int(args.emulator_console_port) + 1:
        raise ValueError("bmoca_emulator_port_pair_invalid")
    config = BMocaEnvironmentConfig.resolve(
        bmoca_root=args.bmoca_root,
        android_sdk_root=args.android_sdk_root,
        android_avd_home=args.android_avd_home,
        avd_template_home=args.bmoca_avd_template_home or None,
        run_headless=not bool(args.show_emulator),
        appium_port=int(args.appium_port),
        appium_system_port=int(args.appium_system_port),
        emulator_console_port=int(args.emulator_console_port),
        emulator_adb_port=int(args.emulator_adb_port),
        emulator_grpc_port=int(args.emulator_grpc_port),
    )
    for required_path, error_code in (
        (config.bmoca_root, "bmoca_root_missing"),
        (config.android_sdk_root, "bmoca_android_sdk_missing"),
        (config.android_avd_home, "bmoca_avd_home_missing"),
    ):
        if not required_path.is_dir():
            raise FileNotFoundError(f"{error_code}:{required_path}")
    episodes = discover_bmoca_episodes(
        config.bmoca_root,
        task_id=selected_tasks[0],
        environment_ids=environment_ids,
    )
    source_states: dict[str, dict[str, Any]] = {}
    output_dir = Path(args.output_path).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    episodes_path = output_dir / "episodes.jsonl"
    if summary_path.exists() or episodes_path.exists():
        raise FileExistsError(f"bmoca_attempt_already_exists:{output_dir}")
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        started = perf_counter()
        run_evidence: dict[str, Any] = {}
        observation_evidence: list[dict[str, Any]] = []
        emulator_serial = ""
        try:
            episode_root = output_dir / f"env_{episode.environment_id}"
            with open_bmoca_episode(
                episode,
                config=config,
                source_states=source_states,
                evidence_root=episode_root,
            ) as host:
                emulator_serial = host.emulator_serial
                result = None
                run_error: Exception | None = None
                try:
                    if method == "ours_replay":
                        from src.integrations.script_replay import run_script_replay

                        result = run_script_replay(
                            store_path=store_path,
                            host=host,
                        )
                    elif method == "skilldroid_replay":
                        from src.integrations.skilldroid_replay import (
                            run_droidrun_macro_replay,
                        )

                        result = run_droidrun_macro_replay(
                            memory_path=reuse_memory_path,
                            host=host,
                        )
                    else:
                        raise ValueError(
                            "bmoca_method_unsupported_without_provider_adapter:"
                            f"{method}"
                        )
                except Exception as error:  # noqa: BLE001 - seal failed episodes too
                    run_error = error
                run_log = host.seal_run_log(
                    task_name=episode.task_id,
                    goal=episode.goal,
                    diagnostics={
                        "method": method,
                        "emulator_serial": emulator_serial,
                        **(
                            {"runtime_error": str(run_error)}
                            if run_error is not None else {}
                        ),
                        **(
                            {
                                "reuse_replay_trace": list(
                                    result.detail.get("trace") or []
                                )
                            }
                            if result is not None
                            else {}
                        ),
                    },
                )
                observation_evidence = list(host.persist_observations() or [])
                if run_log is not None:
                    run_evidence = persist_target_run_evidence(
                        episode_root,
                        run_log=run_log,
                        captured_transfer_states=host.get_captured_transfer_states(),
                    )
                if run_error is not None:
                    raise run_error
                if result is None:
                    raise RuntimeError("bmoca_result_missing")
                row = {
                    "task_id": episode.task_id,
                    "environment_id": episode.environment_id,
                    "snapshot_id": episode.snapshot_id,
                    "avd_name": episode.avd_name,
                    "official_success": host.official_success,
                    "method": method,
                    "emulator_serial": emulator_serial,
                    "method_success": result.success,
                    "error": result.error,
                    **result.execution_summary,
                    "function_id": result.function_id,
                    "function_resolution": dict(
                        result.detail.get("function_resolution") or {}
                    ),
                    "checker_decisions": list(
                        result.detail.get("checker_decisions") or []
                    ),
                    "embedding_calls": int(
                        result.detail.get("embedding_calls") or 0
                    ),
                    "trace": list(result.detail.get("trace") or []),
                    "run_log_evidence": run_evidence,
                    "observation_evidence": observation_evidence,
                    "wall_seconds": round(perf_counter() - started, 6),
                }
                if row["fallback_steps"] != 0:
                    raise RuntimeError("bmoca_function_fallback_must_be_zero")
        except Exception as error:  # noqa: BLE001 - preserve one environment result
            row = {
                "task_id": episode.task_id,
                "environment_id": episode.environment_id,
                "snapshot_id": episode.snapshot_id,
                "avd_name": episode.avd_name,
                "official_success": False,
                "method": method,
                "emulator_serial": emulator_serial,
                "method_success": False,
                "error": str(error),
                "actions_executed": 0,
                "model_calls": 0,
                "embedding_calls": 0,
                "fallback_steps": 0,
                "run_log_evidence": run_evidence,
                "observation_evidence": observation_evidence,
                "wall_seconds": round(perf_counter() - started, 6),
            }
        rows.append(row)
        with episodes_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    official_successes = sum(row["official_success"] is True for row in rows)
    revision = subprocess.run(
        ["git", "-C", str(config.bmoca_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    summary = {
        "schema_version": "omniflow.environment-e2e.v1",
        "environment": "bmoca",
        "method": method,
        "task_id": selected_tasks[0],
        "bmoca_root": str(config.bmoca_root),
        "bmoca_revision": revision or None,
        "environment_ids": list(environment_ids),
        "episode_count": len(rows),
        "official_success_count": official_successes,
        "official_success_rate": official_successes / len(rows) if rows else 0.0,
        "actions_executed": sum(int(row.get("actions_executed") or 0) for row in rows),
        "model_calls": sum(int(row.get("model_calls") or 0) for row in rows),
        "embedding_calls": sum(
            int(row.get("embedding_calls") or 0) for row in rows
        ),
        "fallback_steps": sum(int(row.get("fallback_steps") or 0) for row in rows),
        "results": rows,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if official_successes == len(rows) else 1
