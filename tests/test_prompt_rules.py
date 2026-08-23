from omniflow.core.config import DEFAULT_PLANNER_SYSTEM_PROMPT


def test_v2_planner_prompt_uses_raw_target_pixels() -> None:
    assert "raw pixels" in DEFAULT_PLANNER_SYSTEM_PROMPT
    assert (
        "never output normalized 0..1000 coordinates"
        in DEFAULT_PLANNER_SYSTEM_PROMPT
    )


def test_v2_planner_prompt_preserves_function_fallback() -> None:
    assert "omnitransfer_" in DEFAULT_PLANNER_SYSTEM_PROMPT
    assert "continue from the current screen" in DEFAULT_PLANNER_SYSTEM_PROMPT


def test_v2_planner_prompt_preserves_general_safety_details() -> None:
    prompt = DEFAULT_PLANNER_SYSTEM_PROMPT

    assert "do not reuse source-device coordinates" in prompt
    assert "checked=false" in prompt
    assert "Do not claim that a RunLog or reusable Function was registered" in prompt
