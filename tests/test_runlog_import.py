from __future__ import annotations

from src.integrations.android_world.launch import _raw_replay_step_actions
from src.integrations.runlog import import_run_log_evidence


def test_import_run_log_evidence_keeps_source_ui_for_each_canonical_step() -> None:
    xml = (
        '<hierarchy><node text="Record" resource-id="record" '
        'bounds="[400,700][680,900]" /></hierarchy>'
    )

    run_log, source_states = import_run_log_evidence(
        {
            "run_id": "source-111",
            "goal": "Record audio.",
            "success": True,
            "steps": [
                {
                    "observation_before_act": {
                        "state_id": "source-state-0",
                        "xml": xml,
                        "package_name": "com.example.recorder",
                        "activity_name": ".MainActivity",
                        "display_width": 1080,
                        "display_height": 2400,
                        "screenshot_path": "/source-only/screen.png",
                    },
                    "action": {
                        "tool": "click",
                        "args": {"x": 500, "y": 800},
                    },
                    "result": {"success": True},
                }
            ],
        }
    )

    assert run_log["steps"][0]["before_state_id"] == "source-state-0"
    assert source_states == {
        "schema_version": "omniflow.transfer-state-catalog.v1",
        "run_id": "source-111",
        "states": {
            "source-state-0": {
                "state_id": "source-state-0",
                "xml": xml,
                "package_name": "com.example.recorder",
                "activity_name": ".MainActivity",
                "display": {"width": 1080, "height": 2400},
            }
        },
    }


def test_import_run_log_evidence_normalizes_historical_pixels_by_full_display() -> None:
    run_log, _source_states = import_run_log_evidence(
        {
            "run_id": "source-pixels",
            "goal": "Tap the record button.",
            "success": True,
            "steps": [
                {
                    "observation_before_act": {
                        "width": 720,
                        "height": 1280,
                    },
                    "executed_actions": [
                        {
                            "type": "click",
                            "params": {"x": 360, "y": 1090},
                        }
                    ],
                    "success": True,
                }
            ],
        }
    )

    assert run_log["steps"][0]["action"] == {
        "tool": "click",
        "args": {"x": 500, "y": 851.5625},
    }


def test_import_run_log_evidence_adapts_legacy_canonical_schema_without_status() -> None:
    run_log, _source_states = import_run_log_evidence(
        {
            "schema_version": "omniflow.canonical_run_log.v1",
            "run_id": "legacy-canonical",
            "goal": "Tap the setting.",
            "completed": True,
            "success": True,
            "steps": [
                {
                    "observation_before_act": {
                        "width": 720,
                        "height": 1280,
                    },
                    "executed_actions": [
                        {
                            "type": "click",
                            "params": {"x": 360, "y": 640},
                        }
                    ],
                    "success": True,
                }
            ],
        }
    )

    assert run_log["status"] == "succeeded"
    assert run_log["steps"][0]["action"] == {
        "tool": "click",
        "args": {"x": 500, "y": 500},
    }


def test_import_run_log_evidence_resolves_historical_open_app_name() -> None:
    run_log, _source_states = import_run_log_evidence(
        {
            "run_id": "historical-open-app",
            "goal": "Record audio.",
            "success": True,
            "steps": [
                {
                    "executed_actions": [
                        {
                            "type": "open_app",
                            "params": {"app_name": "audio recorder"},
                        }
                    ],
                    "success": True,
                }
            ],
        },
        package_resolver=lambda name: (
            "com.dimowner.audiorecorder" if name == "audio recorder" else ""
        ),
    )

    assert run_log["steps"][0]["action"] == {
        "tool": "open_app",
        "args": {"package_name": "com.dimowner.audiorecorder"},
    }


def test_import_run_log_evidence_adapts_historical_actions_before_compiling() -> None:
    run_log, _source_states = import_run_log_evidence(
        {
            "run_id": "historical-actions",
            "goal": "Edit an item.",
            "success": True,
            "steps": [
                {
                    "observation_before_act": {
                        "width": 720,
                        "height": 1280,
                    },
                    "executed_actions": [
                        {
                            "type": "open_app",
                            "params": {"app_name": "notes"},
                        }
                    ],
                    "success": True,
                },
                {
                    "observation_before_act": {
                        "package_name": "com.example.notes",
                        "width": 720,
                        "height": 1280,
                    },
                    "executed_actions": [
                        {
                            "type": "input_text",
                            "params": {
                                "text": "hello",
                                "clear_text": True,
                                "x": 360,
                                "y": 320,
                            },
                        }
                    ],
                    "success": True,
                },
                {
                    "observation_before_act": {
                        "package_name": "com.example.notes",
                        "width": 720,
                        "height": 1280,
                    },
                    "executed_actions": [
                        {
                            "type": "swipe",
                            "params": {
                                "start_x": 360,
                                "start_y": 1000,
                                "end_x": 360,
                                "end_y": 200,
                                "duration_ms": 500,
                            },
                        }
                    ],
                    "success": True,
                },
                {
                    "observation_before_act": {
                        "package_name": "com.example.notes",
                        "width": 720,
                        "height": 1280,
                    },
                    "executed_actions": [
                        {"type": "press_back", "params": {}},
                        {"type": "answer", "params": {"text": "done"}},
                    ],
                    "success": True,
                },
            ],
        }
    )

    assert [step["action"] for step in run_log["steps"]] == [
        {
            "tool": "open_app",
            "args": {"package_name": "com.example.notes"},
        },
        {"tool": "click", "args": {"x": 500, "y": 250}},
        {"tool": "input_text", "args": {"text": "hello"}},
        {
            "tool": "swipe",
            "args": {
                "direction": "up",
                "x1": 500,
                "y1": 781.25,
                "x2": 500,
                "y2": 156.25,
                "duration_ms": 500,
            },
        },
        {"tool": "press_key", "args": {"key": "back"}},
    ]


def test_fixed_replay_imports_the_full_runlog_before_extracting_actions() -> None:
    actions = _raw_replay_step_actions(
        {
            "schema_version": "omniflow.run_log.v1",
            "run_id": "historical-full-context",
            "goal": "Turn Wi-Fi on.",
            "completed": True,
            "success": True,
            "steps": [
                {
                    "executed_actions": [
                        {
                            "type": "open_app",
                            "params": {"app_name": "settings"},
                        }
                    ],
                    "success": True,
                },
                {
                    "observation_before_act": {
                        "package_name": "com.android.settings",
                        "width": 720,
                        "height": 1280,
                    },
                    "executed_actions": [
                        {
                            "type": "click",
                            "params": {"x": 360, "y": 640},
                        }
                    ],
                    "success": True,
                },
            ],
        }
    )

    assert actions == [
        {
            "type": "open_app",
            "params": {"package_name": "com.android.settings"},
        },
        {
            "type": "click",
            "params": {"x": 500.0, "y": 500.0},
            "coordinate_space": "canonical_0_1000",
        },
    ]
