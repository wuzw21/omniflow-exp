from __future__ import annotations

import unittest

from src.integrations.gui_agent_oob_host import OobGuiAgentHost


class GuiAgentOobHostTest(unittest.TestCase):
    def test_observe_and_act_delegate_only_to_oob_client(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.observations: list[bool] = []
                self.actions: list[dict] = []

            def observe(self, *, wait_to_stabilize: bool) -> dict:
                self.observations.append(wait_to_stabilize)
                return {
                    "xml": '<hierarchy width="1440" height="3120" />',
                    "package_name": "com.android.settings",
                    "activity_name": ".Settings",
                    "image_base64": "cG5n",
                    "display": {"width": 1440, "height": 3120},
                    "state_id": "state-1",
                    "success": True,
                }

            def act(self, action: dict) -> dict:
                self.actions.append(action)
                return {"success": True, "extra": {"transport": "oob"}}

        client = Client()
        host = OobGuiAgentHost(client)

        observation = host.observe(xml=True, screenshot=True, app_info=True)
        result = host.act({"tool": "press_key", "args": {"key": "home"}})

        self.assertEqual(client.observations, [True])
        self.assertEqual(
            client.actions,
            [{"tool": "press_key", "args": {"key": "home"}}],
        )
        self.assertEqual(observation.package_name, "com.android.settings")
        self.assertEqual(observation.extra["display"], {"width": 1440, "height": 3120})
        self.assertEqual(observation.extra["state_id"], "state-1")
        self.assertTrue(result.success)

    def test_input_text_focuses_through_oob_and_refreshes_precondition(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.events: list[object] = []

            def observe(self, *, wait_to_stabilize: bool) -> dict:
                self.events.append(("observe", wait_to_stabilize))
                return {
                    "xml": "<hierarchy />",
                    "display": {"width": 1000, "height": 1000},
                }

            def act(self, action: dict) -> dict:
                self.events.append(("act", action))
                return {"success": True}

        client = Client()
        host = OobGuiAgentHost(client)

        result = host.act(
            {"tool": "input_text", "args": {"text": "hello", "x": 300, "y": 400}}
        )

        self.assertTrue(result.success)
        self.assertEqual(
            client.events,
            [
                ("act", {"tool": "click", "args": {"x": 300, "y": 400}}),
                ("observe", True),
                (
                    "act",
                    {
                        "tool": "input_text",
                        "args": {"text": "hello", "x": 300, "y": 400},
                    },
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
