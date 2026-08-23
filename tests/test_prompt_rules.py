from omniflow.core.config import DEFAULT_PLANNER_SYSTEM_PROMPT, GUI_AGENT_RULES


def test_gui_agent_rules_are_individually_auditable() -> None:
    assert len(GUI_AGENT_RULES) == len(set(GUI_AGENT_RULES))
    assert all(rule.strip() and rule.endswith((".", "?")) for rule in GUI_AGENT_RULES)
    assert DEFAULT_PLANNER_SYSTEM_PROMPT.endswith(" ".join(GUI_AGENT_RULES))


def test_gui_agent_rules_prevent_runlog_loops_without_runtime_policy() -> None:
    loop_rule = next(rule for rule in GUI_AGENT_RULES if "RunLog" in rule)

    assert "repeated action" in loop_rule
    assert "alternating action sequence" in loop_rule
    assert "no progress" in loop_rule
    assert "do not repeat the loop" in loop_rule
    assert "different visible control or path" in loop_rule


def test_gui_agent_rules_preserve_previous_general_safety_details() -> None:
    prompt = DEFAULT_PLANNER_SYSTEM_PROMPT

    assert "OmniTransfer failure" in prompt
    assert "never reuse source-device coordinates" in prompt
    assert "checked=false means off" in prompt
    assert "Prefer direct search or text input" in prompt
    assert "never claim RunLog or Function registration" in prompt


def test_gui_agent_rules_keep_partial_two_pane_order_from_repeating_navigation() -> None:
    prompt = DEFAULT_PLANNER_SYSTEM_PROMPT

    assert "partial" in prompt
    assert "selected left navigation item is context" in prompt
    assert "right pane" in prompt
