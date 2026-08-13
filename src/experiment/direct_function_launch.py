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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--function-id", required=True)
    parser.add_argument("--function-arguments-json", required=True)
    parser.add_argument("launch_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    repo = args.repo.expanduser().resolve()
    arguments = json.loads(args.function_arguments_json)
    if not isinstance(arguments, dict):
        raise ValueError("function_arguments_must_be_an_object")
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
        return _install_direct_run(
            flow,
            function_id=str(args.function_id).strip(),
            arguments=arguments,
        )

    launch.build_agent = build_direct_agent
    return int(launch.main(launch_args))


if __name__ == "__main__":
    raise SystemExit(main())
