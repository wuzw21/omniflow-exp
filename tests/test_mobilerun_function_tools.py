from __future__ import annotations

import asyncio
import unittest

from omniflow.core.model import Function, RunResult
from src.integrations.gui_agent_tools import GuiAgentToolRuntime
from src.integrations.mobilerun_function_tools import (
    build_custom_tools,
    build_omniflow_custom_tools,
    build_runtime_custom_tools,
)


def _function(*, agent_visible: bool = True) -> Function:
    return Function.from_dict(
        {
            "schema_version": "omniflow.function.v2",
            "function_id": "search_records",
            "name": "Search records",
            "description": "Search records with the requested query.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "search.query": {
                        "type": "string",
                        "description": "Requested query",
                    },
                    "limit": {"type": "integer"},
                },
                "required": ["search.query"],
                "additionalProperties": False,
            },
            "bindings": [],
            "steps": [
                {
                    "step_index": 0,
                    "source_state_id": "state-1",
                    "action": {"tool": "wait", "args": {"duration_ms": 1}},
                }
            ],
            "agent_visible": agent_visible,
        }
    )


class MobilerunFunctionToolsTest(unittest.TestCase):
    def test_maps_schema_and_restores_canonical_argument_names(self) -> None:
        calls: list[tuple[str, dict, object]] = []

        async def invoker(function: Function, arguments: dict, ctx: object) -> str:
            calls.append((function.id, arguments, ctx))
            return "Function completed"

        tools = build_custom_tools([_function()], invoker=invoker)
        self.assertEqual(set(tools), {"search_records"})
        self.assertEqual(
            tools["search_records"]["parameters"]["search_query"],
            {
                "type": "string",
                "required": True,
                "description": "Requested query",
            },
        )

        result = asyncio.run(
            tools["search_records"]["function"](
                search_query="calendar", ctx="mobilerun-context"
            )
        )

        self.assertEqual(result, "Function completed")
        self.assertEqual(
            calls,
            [("search_records", {"search.query": "calendar"}, "mobilerun-context")],
        )

    def test_hides_non_agent_visible_functions(self) -> None:
        tools = build_custom_tools(
            [_function(agent_visible=False)],
            invoker=lambda *_args: "unused",
        )
        self.assertEqual(tools, {})

    def test_omniflow_adapter_routes_through_runtime(self) -> None:
        class Flow:
            def __init__(self) -> None:
                self.host = object()
                self.calls = []
            async def arecall(self, goal, *, limit):
                self.calls.append((goal, limit))
                return RunResult(True, detail={'recall': {'functions': []}})
            async def aexecute_function(self, *args, **kwargs):
                raise AssertionError('recall executed a Function')

        flow = Flow()
        tools = build_omniflow_custom_tools(flow)
        self.assertEqual(set(tools), {'omniflow_recall', 'omniflow_execute'})
        result = asyncio.run(tools['omniflow_recall']['function'](
            task_id='task-one', goal='Find calendar records', limit=8))
        self.assertTrue(result.startswith('Completed:'))
        self.assertEqual(flow.calls, [('Find calendar records', 8)])

    def test_failed_run_result_is_visible_as_tool_failure(self) -> None:
        async def invoker(
            function: Function,
            arguments: dict,
            ctx: object,
        ) -> RunResult:
            return RunResult(
                False,
                function_id=function.id,
                error="transfer_failed",
            )

        tools = build_custom_tools([_function()], invoker=invoker)
        registry_result = asyncio.run(
            tools["search_records"]["function"](search_query="calendar")
        )

        self.assertTrue(registry_result.startswith("Failed:"))

    def test_generic_runtime_overrides_atomic_actions_with_oob_tools(self) -> None:
        class Host:
            def __init__(self) -> None:
                self.actions: list[dict] = []

            def act(self, action: dict) -> dict:
                self.actions.append(action)
                return {"success": True, "extra": {"transport": "oob"}}

        host = Host()
        runtime = GuiAgentToolRuntime(host=host, experiment="mobilerun")
        tools = build_runtime_custom_tools(runtime)

        result = asyncio.run(tools["click"]["function"](x=500, y=500, ctx=object()))

        self.assertTrue(result.startswith("Completed:"))
        self.assertEqual(
            host.actions, [{"tool": "click", "args": {"x": 500, "y": 500}}]
        )


if __name__ == "__main__":
    unittest.main()
