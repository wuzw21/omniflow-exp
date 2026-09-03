from __future__ import annotations

import asyncio
from types import SimpleNamespace
import unittest

from src.integrations.gui_agent_tools import GuiAgentToolRuntime
from src.integrations.minitap_oob_controller import MinitapOobController


class MinitapOobControllerTest(unittest.TestCase):
    def _controller(self):
        class Host:
            def __init__(self) -> None:
                self.actions: list[dict] = []

            def observe(self, **_kwargs):
                return {
                    "xml": (
                        '<hierarchy><node resource-id="search" text="Search" '
                        'content-desc="Find" bounds="[100,200][300,400]" '
                        'clickable="true" /></hierarchy>'
                    ),
                    "package_name": "com.example",
                    "image_base64": "cG5n",
                    "extra": {"display": {"width": 1440, "height": 3120}},
                }

            def act(self, action: dict):
                self.actions.append(action)
                return {"success": True}

        host = Host()
        runtime = GuiAgentToolRuntime(host=host, experiment="minitap")
        return MinitapOobController(runtime, width=1440, height=3120), host

    def test_pixel_actions_are_normalized_and_executed_through_runtime(self) -> None:
        controller, host = self._controller()

        tap = asyncio.run(controller.tap(SimpleNamespace(x=720, y=1560)))
        typed = asyncio.run(controller.input_text("hello"))
        swiped = asyncio.run(
            controller.swipe(
                SimpleNamespace(x=720, y=2500),
                SimpleNamespace(x=720, y=500),
                duration=450,
            )
        )

        self.assertIsNone(tap.error)
        self.assertTrue(typed)
        self.assertIsNone(swiped)
        self.assertEqual(
            host.actions,
            [
                {"tool": "click", "args": {"x": 500, "y": 500}},
                {
                    "tool": "input_text",
                    "args": {"text": "hello", "x": 500, "y": 500},
                },
                {
                    "tool": "swipe",
                    "args": {
                        "direction": "up",
                        "x1": 500,
                        "y1": 801,
                        "x2": 500,
                        "y2": 160,
                        "duration_ms": 450,
                    },
                },
            ],
        )

    def test_observation_is_presented_in_minitap_screen_shape(self) -> None:
        controller, _host = self._controller()

        screen = asyncio.run(controller.get_screen_data())
        element = screen.elements[0]

        self.assertEqual((screen.width, screen.height), (1440, 3120))
        self.assertEqual(screen.base64, "cG5n")
        self.assertEqual(element["resource-id"], "search")
        self.assertEqual(element["accessibilityText"], "Find")

    def test_unsupported_operations_fail_closed(self) -> None:
        controller, host = self._controller()

        self.assertFalse(asyncio.run(controller.open_url("https://example.com")))
        self.assertFalse(asyncio.run(controller.terminate_app("com.example")))
        self.assertFalse(asyncio.run(controller.erase_text()))
        self.assertEqual(host.actions, [])


if __name__ == "__main__":
    unittest.main()
