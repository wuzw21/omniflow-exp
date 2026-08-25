from omniflow.core.config import DEFAULT_PLANNER_SYSTEM_PROMPT, GUI_AGENT_RULES
from omniflow.vlm.gui import SYSTEM_PROMPT


def test_v2_planner_prompt_uses_device_independent_coordinates() -> None:
    assert "normalized 0..1000 coordinates" in DEFAULT_PLANNER_SYSTEM_PROMPT
    assert "click centers of XML bounds" in DEFAULT_PLANNER_SYSTEM_PROMPT


def test_v2_planner_prompt_preserves_function_fallback() -> None:
    assert "after OmniTransfer failure" in DEFAULT_PLANNER_SYSTEM_PROMPT
    assert "choose a fresh action" in DEFAULT_PLANNER_SYSTEM_PROMPT


def test_v2_planner_prompt_preserves_general_safety_details() -> None:
    prompt = DEFAULT_PLANNER_SYSTEM_PROMPT

    assert "never reuse source-device coordinates" in prompt
    assert "checked=false" in prompt
    assert "never claim RunLog or Function registration" in prompt


def test_v2_planner_uses_one_authoritative_prompt() -> None:
    assert len(GUI_AGENT_RULES) == 9
    assert SYSTEM_PROMPT == DEFAULT_PLANNER_SYSTEM_PROMPT
