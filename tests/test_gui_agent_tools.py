from __future__ import annotations

import unittest

from omniflow.core.model import ActionResult, Function, Observation, RunResult
from src.integrations.gui_agent_tools import GuiAgentToolRuntime


def _function(*, agent_visible: bool = True) -> Function:
    return Function.from_dict(
        {
            "schema_version": "omniflow.function.v2",
            "function_id": "search_records",
            "name": "Search records",
            "description": "Search records with the requested query.",
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
            "agent_visible": agent_visible,
        }
    )


class GuiAgentToolRuntimeTest(unittest.TestCase):
    def test_discovers_canonical_actions_and_visible_functions(self) -> None:
        class Store:
            def list_functions(self, *, include_hidden: bool) -> list[Function]:
                self.include_hidden = include_hidden
                return [_function(), _function(agent_visible=False)]

        class Flow:
            def __init__(self) -> None:
                self.store = Store()

        runtime = GuiAgentToolRuntime(host=object(), flow=Flow())

        tools = {tool.name: tool for tool in runtime.list_tools()}

        self.assertIn("click", tools)
        self.assertEqual(tools["click"].kind, "action")
        self.assertEqual(tools["click"].input_schema["required"], ["x", "y"])
        self.assertIn("search_records", tools)
        self.assertEqual(tools["search_records"].kind, "function")
        self.assertEqual(
            tools["search_records"].input_schema["properties"],
            {"query": {"type": "string"}},
        )
        self.assertNotIn("wait", tools)
        self.assertFalse(runtime.flow.store.include_hidden)

    def test_executes_canonical_action_only_through_host(self) -> None:
        class Host:
            def __init__(self) -> None:
                self.actions: list[dict] = []

            def act(self, action: dict) -> ActionResult:
                self.actions.append(action)
                return ActionResult(True, extra={"transport": "oob"})

        host = Host()
        runtime = GuiAgentToolRuntime(host=host)

        result = runtime.call_tool_sync("click", {"x": 125, "y": 875})

        self.assertTrue(result.success)
        self.assertEqual(result.kind, "action")
        self.assertEqual(
            host.actions, [{"tool": "click", "args": {"x": 125, "y": 875}}]
        )
        self.assertEqual(result.output["extra"], {"transport": "oob"})

    def test_function_failure_stays_visible_to_calling_agent(self) -> None:
        class Store:
            def list_functions(self, *, include_hidden: bool) -> list[Function]:
                return [_function()]

        class Flow:
            def __init__(self) -> None:
                self.store = Store()
                self.calls: list[tuple[dict, str]] = []

            async def acall_tool(
                self, tool_call: dict, *, experiment: str
            ) -> RunResult:
                self.calls.append((tool_call, experiment))
                return RunResult(
                    False,
                    function_id=tool_call["name"],
                    error="transfer_failed",
                )

        flow = Flow()
        runtime = GuiAgentToolRuntime(
            host=object(),
            flow=flow,
            experiment="mobilerun",
        )

        result = runtime.call_tool_sync("search_records", {"query": "calendar"})

        self.assertFalse(result.success)
        self.assertEqual(result.kind, "function")
        self.assertEqual(result.error, "transfer_failed")
        self.assertEqual(
            flow.calls,
            [
                (
                    {"name": "search_records", "arguments": {"query": "calendar"}},
                    "mobilerun",
                )
            ],
        )

    def test_rejects_invalid_action_before_host_execution(self) -> None:
        class Host:
            def __init__(self) -> None:
                self.actions: list[dict] = []

            def act(self, action: dict) -> ActionResult:
                self.actions.append(action)
                return ActionResult(True)

        host = Host()
        runtime = GuiAgentToolRuntime(host=host)

        with self.assertRaisesRegex(
            ValueError,
            "canonical_action_arg_range_invalid:x",
        ):
            runtime.call_tool_sync("click", {"x": 1200, "y": 500})

        self.assertEqual(host.actions, [])

    def test_observes_only_through_host_and_returns_canonical_payload(self) -> None:
        class Host:
            def observe(self, **kwargs: object) -> Observation:
                self.kwargs = kwargs
                return Observation(
                    xml="<hierarchy />",
                    package_name="com.example",
                    image_base64="image-data",
                    extra={
                        "state_id": "state-1",
                        "observe_backend": "oob_control",
                    },
                )

        host = Host()
        runtime = GuiAgentToolRuntime(host=host)

        observation = runtime.observe()

        self.assertEqual(
            host.kwargs, {"xml": True, "screenshot": True, "app_info": True}
        )
        self.assertEqual(observation["extra"]["state_id"], "state-1")
        self.assertEqual(observation["package_name"], "com.example")
        self.assertEqual(observation["extra"]["observe_backend"], "oob_control")

    def test_tool_catalog_exports_openai_and_mcp_without_schema_rewrite(self) -> None:
        runtime = GuiAgentToolRuntime(host=object())
        click = next(tool for tool in runtime.list_tools() if tool.name == "click")

        self.assertEqual(
            click.to_mcp_tool(),
            {
                "name": "click",
                "description": click.description,
                "inputSchema": click.input_schema,
            },
        )
        self.assertEqual(
            click.to_openai_tool(),
            {
                "type": "function",
                "function": {
                    "name": "click",
                    "description": click.description,
                    "strict": True,
                    "parameters": click.input_schema,
                },
            },
        )


if __name__ == "__main__":
    unittest.main()
