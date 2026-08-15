"""Launch one explicit Function for atomic source replay qualification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence


def _scrub_planner_environment(environment: dict[str, str]) -> None:
    for key in tuple(environment):
        if key.startswith("OMNIFLOW_PLANNER_") or key.startswith(
            "OMNIFLOW_RECALL_"
        ):
            environment.pop(key, None)
    environment.pop("OMNIFLOW_ENABLE_ONLINE_PLANNER", None)
    environment.pop("OPENAI_API_KEY", None)


def _install_direct_run(
    flow: Any,
    *,
    function_id: str,
    arguments: dict[str, Any],
) -> Any:
    from omniflow import ToolCall

    if flow.store.get_function(function_id) is None:
        raise ValueError(f"direct_function_not_found:{function_id}")
    original_call_tool = flow.call_tool

    def direct_run(_goal: str, *, experiment: Any = None) -> Any:
        return original_call_tool(
            ToolCall(function_id, dict(arguments)),
            experiment=experiment,
        )

    flow.run = direct_run
    flow.direct_function_id = function_id
    flow.direct_function_arguments = dict(arguments)
    return flow


def _install_direct_runs(
    flow: Any,
    *,
    calls: list[dict[str, Any]],
) -> Any:
    from omniflow import RunResult, ToolCall

    if not calls:
        raise ValueError("direct_function_calls_required")
    normalized: list[dict[str, Any]] = []
    for call in calls:
        function_id = str(call.get("function_id") or "").strip()
        arguments = call.get("arguments")
        if not function_id or not isinstance(arguments, dict):
            raise ValueError("direct_function_call_invalid")
        if flow.store.get_function(function_id) is None:
            raise ValueError(f"direct_function_not_found:{function_id}")
        normalized.append(
            {"function_id": function_id, "arguments": dict(arguments)}
        )
    original_call_tool = flow.call_tool
    next_call = 0

    def direct_run(_goal: str, *, experiment: Any = None) -> Any:
        nonlocal next_call
        call = normalized[next_call]
        result = original_call_tool(
            ToolCall(call["function_id"], dict(call["arguments"])),
            experiment=experiment,
        )
        if not result.success:
            return result
        next_call += 1
        if next_call < len(normalized):
            return result
        return RunResult(
            result.success,
            result.function_id,
            result.actions_executed,
            result.model_calls,
            result.fallback_steps,
            result.error,
            result.final_state,
            {**result.detail, "done_reason": "finished"},
        )

    flow.run = direct_run
    flow.direct_function_calls = normalized
    return flow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--function-id")
    parser.add_argument("--function-arguments-json")
    parser.add_argument("--function-calls-json")
    parser.add_argument("launch_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    repo = args.repo.expanduser().resolve()
    if args.function_calls_json:
        if args.function_id or args.function_arguments_json:
            raise ValueError("direct_function_call_mode_ambiguous")
        calls = json.loads(args.function_calls_json)
        if not isinstance(calls, list):
            raise ValueError("function_calls_must_be_an_array")
    else:
        if not args.function_id or args.function_arguments_json is None:
            raise ValueError("function_id_and_arguments_required")
        arguments = json.loads(args.function_arguments_json)
        if not isinstance(arguments, dict):
            raise ValueError("function_arguments_must_be_an_object")
        calls = [
            {
                "function_id": str(args.function_id).strip(),
                "arguments": arguments,
            }
        ]
    launch_args = list(args.launch_args)
    if launch_args and launch_args[0] == "--":
        launch_args.pop(0)
    if not launch_args:
        raise ValueError("androidworld_launch_arguments_required")

    _scrub_planner_environment(os.environ)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from src.integrations.android_world import launch

    original_build_agent = launch.build_agent

    def build_direct_agent(**kwargs: Any) -> Any:
        kwargs.pop("planner", None)
        flow = original_build_agent(**kwargs)
        return _install_direct_runs(flow, calls=calls)

    launch.build_agent = build_direct_agent
    return int(launch.main(launch_args))


if __name__ == "__main__":
    raise SystemExit(main())
