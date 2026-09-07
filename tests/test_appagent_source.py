import hashlib
import json

import pytest

from omniflow.core.trajectory import canonicalize_run_log_step
from omniflow.runlog import _androidworld_input_target
from omniflow.runlog import _xml_index_bounds
from src.experiment.appagent_source import _resolve_appagent_screenshot
from src.integrations.appagent import _looks_like_android_package
from src.integrations.runlog import project_androidworld_step_actions


def test_resolves_screenshot_from_sibling_runlog_attempt(tmp_path):
    runlog_root = tmp_path / "data" / "androidworld" / "Task" / "source" / "runlog"
    source_run_log = runlog_root / "current" / "run_log.json"
    source_run_log.parent.mkdir(parents=True)
    source_run_log.write_text("{}", encoding="utf-8")

    screenshot = runlog_root / "attempt_001" / "screenshots" / "screen.png"
    screenshot.parent.mkdir(parents=True)
    screenshot.write_bytes(b"screenshot-evidence")
    digest = hashlib.sha256(screenshot.read_bytes()).hexdigest()

    resolved = _resolve_appagent_screenshot(
        {
            "path": "/Users/author/Projects/OmniFlow-exp/data/androidworld/Task/source/OmniFlowSourceSmall_seed111/runlog/attempt_001/screenshots/screen.png",
            "sha256": digest,
        },
        source_run_log=source_run_log,
    )

    assert resolved == screenshot.resolve()


def test_resolves_renamed_screenshot_by_same_source_action(tmp_path):
    runlog_root = tmp_path / "data" / "androidworld" / "Task" / "source" / "runlog"
    source_run_log = runlog_root / "current" / "run_log.json"
    source_run_log.parent.mkdir(parents=True)
    screenshot = runlog_root / "attempt_001" / "screenshots" / "screenshot_000002.png"
    screenshot.parent.mkdir(parents=True)
    screenshot.write_bytes(b"screenshot-evidence")
    sibling_log = runlog_root / "attempt_001" / "run_log.json"
    sibling_log.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "action": {"action_type": "click", "x": 10, "y": 20},
                        "observation": {
                            "screenshot": {
                                "path": "/old/data/androidworld/Task/source/runlog/attempt_001/screenshots/screenshot_000002.png",
                            }
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    source_run_log.write_text(
        json.dumps(
            {
                "steps": [
                    {"action": {"action_type": "click", "x": 10, "y": 20}}
                ]
            }
        ),
        encoding="utf-8",
    )

    resolved = _resolve_appagent_screenshot(
        {
            "path": "/old/data/androidworld/Task/source/runlog/current/observations/objects/renamed.png"
        },
        source_run_log=source_run_log,
        source_step_index=0,
    )

    assert resolved == screenshot.resolve()


def test_app_label_is_not_counted_as_second_android_package():
    assert _looks_like_android_package("com.arduia.expense")
    assert not _looks_like_android_package("pro expense")


def test_androidworld_swipe_projection_uses_canonical_endpoint_args():
    projected = project_androidworld_step_actions(
        {
            "observation": {
                "screenshot": {"width": 720, "height": 1280},
                "xml": (
                    '<hierarchy width="720" height="1280">'
                    '<node scrollable="true" bounds="[0,0][720,1200]" />'
                    "</hierarchy>"
                ),
            },
            "action": {"action_type": "scroll", "direction": "up"},
        }
    )

    assert projected == [
        {
            "tool": "swipe",
            "args": {"x1": 500, "y1": 468.75, "x2": 500, "y2": 796.875},
        }
    ]


def test_runlog_schema_accepts_coordinate_swipe_with_direction():
    step = canonicalize_run_log_step(
        {
            "step_index": 12,
            "observation": {"screenshot": None, "xml": "<hierarchy />"},
            "action": {
                "action_type": "swipe",
                "direction": "left",
                "x1": 648.0,
                "y1": 569.6,
                "x2": 108.0,
                "y2": 569.6,
                "duration_ms": 500,
            },
            "result": {"success": True},
        }
    )

    assert step["action"]["x1"] == 648.0
    assert step["action"]["y2"] == 569.6


def test_runlog_schema_rejects_partial_or_non_swipe_endpoints():
    base = {
        "step_index": 0,
        "observation": {"screenshot": None, "xml": "<hierarchy />"},
        "result": {"success": True},
    }

    with pytest.raises(ValueError, match="run_log_schema_invalid:run_log_step"):
        canonicalize_run_log_step(
            {
                **base,
                "action": {
                    "action_type": "swipe",
                    "x1": 1,
                    "y1": 1,
                    "direction": "left",
                },
            }
        )

    with pytest.raises(ValueError, match="run_log_schema_invalid:run_log_step"):
        canonicalize_run_log_step(
            {
                **base,
                "action": {
                    "action_type": "click",
                    "x": 1,
                    "y": 1,
                    "x1": 1,
                    "y1": 1,
                    "x2": 2,
                    "y2": 2,
                },
            }
        )


def test_androidworld_projection_accepts_numeric_xml_node_ids():
    xml = (
        '<hierarchy width="720" height="1280">'
        '<window><node id="0" bounds="[0,0][720,1280]">'
        '<node id="1" clickable="true" bounds="[10,20][110,120]" />'
        "</node></window></hierarchy>"
    )

    assert _xml_index_bounds(xml, 0) == (10.0, 20.0, 110.0, 120.0)
    assert project_androidworld_step_actions(
        {
            "observation": {"screenshot": {"width": 720, "height": 1280}, "xml": xml},
            "action": {"action_type": "click", "index": 0},
        }
    ) == [{"tool": "click", "args": {"x": 83.33333333333333, "y": 54.6875}}]


def test_input_projection_uses_unique_existing_text_without_coordinates():
    xml = (
        '<hierarchy width="720" height="1280">'
        '<node bounds="[0,0][720,1280]">'
        '<node class="android.widget.EditText" editable="true" text="name" bounds="[0,0][360,100]" />'
        '<node class="android.widget.EditText" editable="true" text=".txt" bounds="[360,0][720,100]" />'
        "</node></hierarchy>"
    )
    action = {"action_type": "input_text", "text": ".txt"}
    observation = {"screenshot": {"width": 720, "height": 1280}, "xml": xml}

    point, _ = _androidworld_input_target(action, observation, observation)

    assert point == {"x": 750.0, "y": 39.0625}
