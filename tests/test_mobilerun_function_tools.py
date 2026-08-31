from __future__ import annotations

import asyncio
import unittest

from omniflow.core.model import Function
from src.integrations.mobilerun_function_tools import (
    build_custom_tools,
    build_omniflow_custom_tools,
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
        class Store:
            def list_functions(self, *, include_hidden: bool) -> list[Function]:
                self.include_hidden = include_hidden
                return [_function()]

        class Flow:
            def __init__(self) -> None:
                self.store = Store()
                self.calls: list[tuple[dict, str]] = []

            async def acall_tool(self, tool_call: dict, *, experiment: str):
                self.calls.append((tool_call, experiment))
                return "runtime completed"

        flow = Flow()
        tools = build_omniflow_custom_tools(flow)
        result = asyncio.run(
            tools["search_records"]["function"](search_query="calendar")
        )

        self.assertEqual(result, "runtime completed")
        self.assertEqual(
            flow.calls,
            [
                (
                    {"name": "search_records", "arguments": {"search.query": "calendar"}},
                    "mobilerun",
                )
            ],
        )
        self.assertFalse(flow.store.include_hidden)


if __name__ == "__main__":
    unittest.main()
