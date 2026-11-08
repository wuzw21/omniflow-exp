"""AgentRR is a memory design, so its guards must be testable without adb."""

from __future__ import annotations

import pytest

from omniflow.core.model import Action, Observation
from omniflow.transfer.agentrr import (
    AgentRRPolicy,
    agentrr_transfer,
    agentrr_transfer_with,
)

SOURCE_XML = (
    '<hierarchy><node class="android.widget.FrameLayout" package="com.android.settings" '
    'bounds="[0,0,1080,2400]">'
    '<node class="android.widget.Switch" resource-id="com.android.settings:id/switch_widget" '
    'text="Bluetooth" bounds="[900,400,1020,480]" clickable="true"/>'
    "</node></hierarchy>"
)
# Same control, moved and resized: a replay must land on it, not on its old point.
MOVED_XML = (
    '<hierarchy><node class="android.widget.FrameLayout" package="com.android.settings" '
    'bounds="[0,0,1440,3120]">'
    '<node class="android.widget.Switch" resource-id="com.android.settings:id/switch_widget" '
    'text="Bluetooth" bounds="[1200,1200,1400,1320]" clickable="true"/>'
    "</node></hierarchy>"
)
# The control is absent; AgentRR must abstain rather than guess.
ABSENT_XML = (
    '<hierarchy><node class="android.widget.FrameLayout" package="com.android.settings" '
    'bounds="[0,0,1440,3120]">'
    '<node class="android.widget.TextView" text="About phone" bounds="[100,100,600,200]"/>'
    "</node></hierarchy>"
)


def _source() -> Observation:
    return Observation(xml=SOURCE_XML, package_name="com.android.settings")


def _observation(xml: str, package: str = "com.android.settings") -> Observation:
    return Observation(xml=xml, package_name=package)


def _click_at_source_switch() -> Action:
    # The source switch centre is (960, 440) on a 1080x2400 display.
    return Action("click", {"x": 960 / 1080 * 1000.0, "y": 440 / 2400 * 1000.0})


def test_replay_lands_on_the_recorded_control_after_it_moves():
    result = agentrr_transfer(
        _click_at_source_switch(), _observation(MOVED_XML), _source()
    )
    assert result.action is not None
    assert result.reason == "agentrr_replay_signature"
    # Centre of [1200,1200,1400,1320] on a 1440x3120 display.
    assert result.action.args["x"] == pytest.approx(1300 / 1440 * 1000.0)
    assert result.action.args["y"] == pytest.approx(1260 / 3120 * 1000.0)


def test_absent_control_makes_agentrr_abstain_instead_of_guessing():
    result = agentrr_transfer(
        _click_at_source_switch(), _observation(ABSENT_XML), _source()
    )
    assert result.action is None
    assert result.reason == "agentrr_guard_signature_absent"


def test_package_change_is_refused_before_any_target_search():
    result = agentrr_transfer(
        _click_at_source_switch(),
        _observation(MOVED_XML, package="com.android.chrome"),
        _source(),
    )
    assert result.action is None
    assert result.reason == "agentrr_guard_package_changed"


def test_missing_source_state_is_a_failure_not_a_position_replay():
    result = agentrr_transfer(_click_at_source_switch(), _observation(MOVED_XML), None)
    assert result.action is None
    assert result.reason == "agentrr_source_state_missing"


def test_policy_off_switches_to_pure_positional_replay():
    # This is the weakest form of the baseline: it replays the recorded point on
    # the new display without requiring the recorded identity to still exist.
    positional = agentrr_transfer_with(
        AgentRRPolicy(require_package_match=False, require_same_signature=False)
    )
    result = positional(_click_at_source_switch(), _observation(MOVED_XML), _source())
    assert result.action is not None
    assert result.reason == "agentrr_replay_position"
    # The relative point is preserved across the display change, so the replay
    # lands on the same fraction of the screen, not on the moved control.
    assert result.action.args["x"] == pytest.approx(960 / 1080 * 1000.0)
    assert result.action.args["y"] == pytest.approx(440 / 2400 * 1000.0)


def test_non_positional_actions_pass_through_unchanged():
    action = Action("press_key", {"key": "back"})
    result = agentrr_transfer(action, _observation(MOVED_XML), _source())
    assert result.action is action


def test_ambiguous_signature_is_not_resolved_by_guessing():
    duplicated = (
        '<hierarchy><node class="android.widget.FrameLayout" package="com.android.settings" '
        'bounds="[0,0,1440,3120]">'
        '<node class="android.widget.Switch" resource-id="com.android.settings:id/switch_widget" '
        'text="Bluetooth" bounds="[100,100,200,200]"/>'
        '<node class="android.widget.Switch" resource-id="com.android.settings:id/switch_widget" '
        'text="Bluetooth" bounds="[100,800,200,900]"/>'
        "</node></hierarchy>"
    )
    result = agentrr_transfer(
        _click_at_source_switch(), _observation(duplicated), _source()
    )
    assert result.action is None
    assert result.reason == "agentrr_guard_signature_absent"


def test_baseline_registry_exposes_mobimem_as_a_transfer_only_swap():
    from src.integrations.android_world.agent import SAME_HARNESS_BASELINES, _baseline_transfer

    assert "mobimem_style" in SAME_HARNESS_BASELINES
    transfer = _baseline_transfer("mobimem_style")
    assert callable(transfer)
    # OmniFlow's own row keeps the canonical mapper: no baseline override.
    assert _baseline_transfer("none") is None
