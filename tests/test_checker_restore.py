from __future__ import annotations

from omniflow.core.config import OmniFlowConfig
from omniflow.core.model import Action, CheckerContext, Observation
from omniflow.runlog import import_run_log_evidence
from omniflow.runtime.checker import (
    checker_rule_action,
    checker_rule_matches,
    default_checker,
)


def test_restore_target_app_uses_system_ui_recovery_gesture() -> None:
    source = Observation(
        xml=(
            '<hierarchy><node package="com.android.systemui" '
            'class="android.widget.SeekBar" /></hierarchy>'
        )
    )
    current = Observation(
        package_name="com.google.android.googlequicksearchbox",
        xml=(
            '<hierarchy><node package="com.google.android.googlequicksearchbox" '
            'class="android.view.View" /></hierarchy>'
        ),
    )
    rule = {
        "schema_version": "omniflow.checker_rule.v1",
        "id": "restore_target_app",
        "enabled": True,
        "phase": "pre_transfer",
        "condition": {"package_mismatch": True},
        "action": {"action": "open_app"},
        "budget": {"max_triggers_per_run": 1},
    }
    action = Action("swipe", {"x1": 100, "y1": 500, "x2": 900, "y2": 500})

    assert checker_rule_matches(
        rule,
        current=current,
        source=source,
        function_id="set_brightness_to_max",
        step_index=0,
        action=action,
    )
    assert checker_rule_action(rule, current=current, source=source) == Action(
        "swipe",
        {
            "x1": 500,
            "y1": 0,
            "x2": 500,
            "y2": 1000,
            "duration_ms": 500,
        },
    )


def test_system_ui_source_does_not_trigger_overlay_dismissal() -> None:
    source = Observation(
        xml=(
            '<hierarchy><node package="com.android.systemui" '
            'resource-id="com.android.systemui:id/notification_panel" />'
            '</hierarchy>'
        )
    )
    current = Observation(
        package_name="com.android.systemui",
        xml=(
            '<hierarchy><node package="com.android.systemui" '
            'resource-id="com.android.systemui:id/notification_panel" />'
            '</hierarchy>'
        ),
    )
    rule = {
        "schema_version": "omniflow.checker_rule.v1",
        "id": "dismiss_system_overlay",
        "enabled": True,
        "phase": "pre_transfer",
        "condition": {
            "xpath_exists": (
                "//node[contains(@package,'com.android.systemui') and "
                "contains(@resource-id,'notification_panel')]"
            )
        },
        "action": {"action": "wait", "wait_ms": 1},
    }

    assert not checker_rule_matches(
        rule,
        current=current,
        source=source,
        function_id="set_brightness_to_max",
        step_index=0,
        action=Action("swipe", {"x1": 100, "y1": 500, "x2": 900, "y2": 500}),
    )


def test_runlog_source_state_preserves_main_package_from_xml() -> None:
    run_log = {
        "schema_version": "omniflow.run_log.v1",
        "run_id": "run-1",
        "task_name": "MarkorDeleteNote",
        "goal": "Delete the requested note.",
        "status": "succeeded",
        "success": True,
        "seed": 111,
        "task_parameters": {},
        "provenance": {"kind": "runtime"},
        "validator": {"official": True, "success": True, "reward": 1},
        "steps": [
            {
                "step_index": 0,
                "observation": {
                    "screenshot": None,
                    "xml": (
                        '<hierarchy><node package="net.gsantner.markor" />'
                        '<node package="com.android.systemui" /></hierarchy>'
                    )
                },
                "action": {
                    "action_type": "swipe",
                    "x1": 1,
                    "y1": 1,
                    "x2": 1,
                    "y2": 2,
                },
                "result": {"success": True},
            }
        ],
    }

    _, catalog = import_run_log_evidence(run_log)

    assert next(iter(catalog["states"].values()))["package_name"] == (
        "net.gsantner.markor"
    )


def test_default_checker_recovers_from_xml_only_package_identity() -> None:
    source = Observation(
        xml='<hierarchy><node package="net.gsantner.markor" /></hierarchy>'
    )
    current = Observation(
        xml=(
            '<hierarchy><node package="com.google.android.apps.nexuslauncher" />'
            '</hierarchy>'
        )
    )

    assert default_checker(
        CheckerContext(source, current, Action("swipe", {"direction": "up"}))
    ) == Action("open_app", {"package_name": "net.gsantner.markor"})


def test_default_checker_is_injected_for_every_action_path() -> None:
    assert OmniFlowConfig().resolved_plugins().checker is default_checker
