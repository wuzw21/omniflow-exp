from __future__ import annotations

from src.integrations.mobilegpt_oob_client import (
    _action_with_bounds,
    _oob_action,
)


class _FakeOob:
    def __init__(self) -> None:
        self.actions: list[dict] = []

    def act(self, action: dict) -> dict:
        self.actions.append(action)
        return {"success": True}


def test_official_index_action_is_mapped_to_current_oob_bounds() -> None:
    xml = (
        '<hierarchy><node index="0" bounds="[0,0][1280,800]">'
        '<node index="4" bounds="[100,200][300,400]" />'
        "</node></hierarchy>"
    )
    action = {"name": "click", "parameters": {"index": "4"}}

    mapped = _action_with_bounds(action, xml)

    assert mapped["parameters"]["oob_bounds"] == "[100,200][300,400]"


def test_oob_action_executes_official_click_and_input_schema() -> None:
    xml = '<hierarchy><node index="2" bounds="[100,200][300,400]" /></hierarchy>'
    oob = _FakeOob()

    _oob_action(
        oob,
        {"name": "click", "parameters": {"index": "2"}},
        {"width": 1000, "height": 1000},
        xml,
    )
    _oob_action(
        oob,
        {
            "name": "input",
            "parameters": {"index": "2", "input_text": "Sara Ahmed"},
        },
        {"width": 1000, "height": 1000},
        xml,
    )

    assert oob.actions == [
        {"tool": "click", "args": {"x": 200, "y": 300}},
        {"tool": "click", "args": {"x": 200, "y": 300}},
        {
            "tool": "input_text",
            "args": {"text": "Sara Ahmed", "clear_text": True},
        },
    ]
