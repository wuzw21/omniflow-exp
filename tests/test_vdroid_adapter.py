from __future__ import annotations

import asyncio
import unittest

from omniflow.core.model import Function
from src.integrations.gui_agent_tools import GuiAgentToolRuntime
from src.integrations.vdroid_adapter import VDroidAdapter


class VDroidAdapterTest(unittest.TestCase):
    def _adapter(self):
        class Host:
            def __init__(self) -> None:
                self.actions: list[dict] = []

            def act(self, action: dict):
                self.actions.append(action)
                return {"success": True}

        host = Host()
        runtime = GuiAgentToolRuntime(host=host, experiment="vdroid")
        return VDroidAdapter(runtime), host

    def test_executes_highest_scored_live_index_candidate(self) -> None:
        adapter, host = self._adapter()
        observation = {
            "xml": (
                '<hierarchy><node id="0" bounds="[0,0][100,100]" />'
                '<node id="1" bounds="[200,300][600,700]" clickable="true" />'
                "</hierarchy>"
            ),
            "extra": {"display": {"width": 1000, "height": 2000}},
        }

        outcome = asyncio.run(
            adapter.execute_selected(
                [
                    '{"action_type":"navigate_back"}',
                    '{"action_type":"click","index":1}',
                ],
                [0.1, 0.9],
                observation,
            )
        )

        self.assertTrue(outcome.success)
        self.assertEqual(
            host.actions, [{"tool": "click", "args": {"x": 400, "y": 250}}]
        )

    def test_scroll_semantics_are_converted_to_physical_gesture(self) -> None:
        adapter, host = self._adapter()
        observation = {
            "xml": '<hierarchy><node id="0" bounds="[0,0][1000,2000]" /></hierarchy>',
            "extra": {"display": {"width": 1000, "height": 2000}},
        }

        outcome = asyncio.run(
            adapter.execute_action(
                {"action_type": "scroll", "direction": "down", "index": 0},
                observation,
            )
        )

        self.assertTrue(outcome.success)
        self.assertEqual(
            host.actions,
            [
                {
                    "tool": "swipe",
                    "args": {
                        "direction": "up",
                        "x1": 500,
                        "y1": 500,
                        "x2": 500,
                        "y2": 0,
                        "duration_ms": 400,
                    },
                }
            ],
        )

    def test_missing_live_index_fails_without_coordinate_fallback(self) -> None:
        adapter, host = self._adapter()
        observation = {
            "xml": '<hierarchy><node id="0" bounds="[0,0][100,100]" /></hierarchy>',
            "extra": {"display": {"width": 1000, "height": 2000}},
        }

        outcome = asyncio.run(
            adapter.execute_action(
                {"action_type": "click", "index": 9},
                observation,
            )
        )

        self.assertFalse(outcome.success)
        self.assertEqual(outcome.message, "vdroid_live_index_unresolved:9")
        self.assertEqual(host.actions, [])

    def test_verifier_can_select_an_omniflow_function_candidate(self) -> None:
        function = Function.from_dict(
            {
                "schema_version": "omniflow.function.v2",
                "function_id": "saved_flow",
                "name": "Saved flow",
                "description": "Run a saved flow.",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
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
                "agent_visible": True,
            }
        )

        class Store:
            def list_functions(self, *, include_hidden: bool):
                return [function]

        class Flow:
            store = Store()

            async def acall_tool(self, tool_call: dict, *, experiment: str):
                self.call = (tool_call, experiment)
                return {"success": True}

        class Host:
            def act(self, action: dict):
                raise AssertionError(action)

        flow = Flow()
        adapter = VDroidAdapter(
            GuiAgentToolRuntime(host=Host(), flow=flow, experiment="vdroid")
        )
        candidate = {
            "action_type": "omniflow_function",
            "name": "saved_flow",
            "arguments": {"query": "calendar"},
        }

        outcome = asyncio.run(adapter.execute_action(candidate, {}))

        self.assertTrue(outcome.success)
        self.assertEqual(
            flow.call,
            (
                {"name": "saved_flow", "arguments": {"query": "calendar"}},
                "vdroid",
            ),
        )


if __name__ == "__main__":
    unittest.main()
