from __future__ import annotations

import asyncio
import json
import unittest

from omniflow.core.model import Function
from src.integrations.gui_agent_tools import GuiAgentToolRuntime
from src.integrations.gui_owl_adapter import GuiOwlAdapter, parse_gui_owl_tool_call


def _response(name: str, arguments: dict) -> str:
    return (
        "Action: execute\n<tool_call>\n"
        + json.dumps({"name": name, "arguments": arguments})
        + "\n</tool_call>"
    )


class GuiOwlAdapterTest(unittest.TestCase):
    def _adapter(self):
        class Host:
            def __init__(self) -> None:
                self.actions: list[dict] = []

            def act(self, action: dict):
                self.actions.append(action)
                return {"success": True}

        host = Host()
        runtime = GuiAgentToolRuntime(host=host, experiment="gui_owl_1_5")
        return GuiOwlAdapter(runtime), host

    def test_parses_upstream_mobile_use_call_without_coordinate_rescaling(self) -> None:
        adapter, host = self._adapter()

        outcome = asyncio.run(
            adapter.execute_output(
                _response(
                    "mobile_use",
                    {"action": "click", "coordinate": [250, 750]},
                )
            )
        )

        self.assertTrue(outcome.success)
        self.assertEqual(
            host.actions, [{"tool": "click", "args": {"x": 250, "y": 750}}]
        )

    def test_type_reuses_the_explicitly_focused_point(self) -> None:
        adapter, host = self._adapter()
        asyncio.run(
            adapter.execute_output(
                _response("mobile_use", {"action": "click", "coordinate": [400, 600]})
            )
        )

        outcome = asyncio.run(
            adapter.execute_output(
                _response("mobile_use", {"action": "type", "text": "hello"})
            )
        )

        self.assertTrue(outcome.success)
        self.assertEqual(
            host.actions[-1],
            {"tool": "input_text", "args": {"text": "hello", "x": 400, "y": 600}},
        )

    def test_terminate_is_terminal_and_does_not_touch_device(self) -> None:
        adapter, host = self._adapter()

        outcome = asyncio.run(
            adapter.execute_output(
                _response("mobile_use", {"action": "terminate", "status": "success"})
            )
        )

        self.assertTrue(outcome.finished)
        self.assertTrue(outcome.success)
        self.assertEqual(host.actions, [])

    def test_parser_rejects_trailing_or_ambiguous_tool_calls(self) -> None:
        text = _response("mobile_use", {"action": "click", "coordinate": [1, 2]})
        with self.assertRaisesRegex(ValueError, "gui_owl_tool_call_count_invalid"):
            parse_gui_owl_tool_call(text + text)

    def test_direct_function_tool_call_uses_the_same_runtime(self) -> None:
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
        adapter = GuiOwlAdapter(
            GuiAgentToolRuntime(host=Host(), flow=flow, experiment="gui_owl_1_5")
        )

        outcome = asyncio.run(
            adapter.execute_output(_response("saved_flow", {"query": "calendar"}))
        )

        self.assertTrue(outcome.success)
        self.assertEqual(
            flow.call,
            (
                {"name": "saved_flow", "arguments": {"query": "calendar"}},
                "gui_owl_1_5",
            ),
        )


if __name__ == "__main__":
    unittest.main()
