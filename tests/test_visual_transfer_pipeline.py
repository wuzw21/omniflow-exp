from __future__ import annotations

import asyncio
import json
from pathlib import Path

from runlog_fixtures import androidworld_run_log, androidworld_state

from omniflow.core.config import PluginSet
from omniflow.core.model import Action, ActionResult, Observation, TransferResult
from omniflow.functions.assets import save_function
from omniflow.runtime import execution
from omniflow.transfer.runtime import capture_transfer_state

SOURCE_XML = (
    '<hierarchy bounds="[0,0][100,200]">'
    '<node class="android.widget.Button" clickable="true" enabled="true" '
    'bounds="[10,20][50,80]" />'
    "</hierarchy>"
)
TARGET_XML = SOURCE_XML.replace("[10,20][50,80]", "[40,60][90,140]")


def test_capture_transfer_state_preserves_screenshot_path(tmp_path: Path) -> None:
    screenshot = (tmp_path / "source.jpg").resolve()
    screenshot.write_bytes(b"jpeg")
    state = capture_transfer_state(
        Observation(
            xml=SOURCE_XML,
            package_name="com.example",
            extra={
                "state_id": "source-state",
                "display": {"width": 100, "height": 200},
                "screenshot_path": str(screenshot),
            },
        )
    )

    assert state["screenshot_path"] == str(screenshot)


def test_capture_transfer_state_reads_androidworld_pixels_reference(
    tmp_path: Path,
) -> None:
    screenshot = (tmp_path / "source.png").resolve()
    screenshot.write_bytes(b"png")

    state = capture_transfer_state(
        Observation(
            xml=SOURCE_XML,
            extra={
                "state_id": "source-state",
                "androidworld_state": {"pixels": {"path": str(screenshot)}},
            },
        )
    )

    assert state["screenshot_path"] == str(screenshot)


def test_function_compiler_preserves_source_screenshot_reference(
    tmp_path: Path,
) -> None:
    screenshot = (tmp_path / "source.jpg").resolve()
    screenshot.write_bytes(b"jpeg")
    run_log = androidworld_run_log(
        [
            {"action_type": "open_app", "app_name": "com.example"},
            {"action_type": "wait"},
        ],
        observations=[androidworld_state("state_0"), androidworld_state("state_1")],
        goal="Open the example and wait.",
    )
    for step in run_log["steps"]:
        step["observation"]["pixels"] = {
            "path": str(screenshot),
            "sha256": "0" * 64,
            "width": 100,
            "height": 200,
            "mime_type": "image/jpeg",
        }

    bundle = {
            "schema_version": "omniflow.function-bundle.v2",
            "run_id": "source-run",
            "arguments": {"open_example_and_wait": {}},
            "functions": [
                {
                    "schema_version": "omniflow.function.v2",
                    "function_id": "open_example_and_wait",
                    "name": "Open the recorded example app and wait once",
                    "description": (
                        "Open the fixed recorded example package and wait once. "
                        "This Function does not perform or verify another task."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                    "bindings": [],
                    "steps": [
                        {
                            "step_index": 0,
                            "source_state_id": "state_0",
                            "action": {
                                "tool": "open_app",
                                "args": {"package_name": "com.example"},
                            },
                        },
                        {
                            "step_index": 1,
                            "source_state_id": "state_1",
                            "action": {
                                "tool": "wait",
                                "args": {"duration_ms": 1000},
                            },
                        },
                    ],
                    "checker_rules": [],
                    "agent_visible": True,
                }
            ],
        }
    result = save_function(
        run_log,
        tmp_path / "output" / "store.json",
        functions=bundle["functions"],
        arguments=bundle["arguments"],
    )

    states = json.loads(
        Path(result["transfer_state_catalog"]).read_text(encoding="utf-8")
    )["states"]
    assert states["state_0"]["screenshot_path"] == str(screenshot)
    assert states["state_1"]["screenshot_path"] == str(screenshot)


def test_default_transfer_forwards_source_and_target_screenshots(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_screenshot = (tmp_path / "source.jpg").resolve()
    target_screenshot = (tmp_path / "target.jpg").resolve()
    source_screenshot.write_bytes(b"source")
    target_screenshot.write_bytes(b"target")
    calls: list[dict] = []

    def fake_transfer_action(**request):
        calls.append(request)
        return {
            "mapped": True,
            "mapping_mode": "test",
            "new_x": 65.0,
            "new_y": 100.0,
            "target_bbox": [40.0, 60.0, 90.0, 140.0],
            "candidates": [{"score": 1.0}],
        }

    monkeypatch.setattr(execution, "transfer_action", fake_transfer_action)
    source = Observation(
        xml=SOURCE_XML,
        package_name="com.example",
        extra={
            "display": {"width": 100, "height": 200},
            "screenshot_path": str(source_screenshot),
            "visual_rgb": {"width": 1, "height": 1, "data_base64": "AAA="},
        },
    )
    target = Observation(
        xml=TARGET_XML,
        package_name="com.example",
        extra={
            "display": {"width": 100, "height": 200},
            "screenshot_path": str(target_screenshot),
            "visual_rgb": {"width": 1, "height": 1, "data_base64": "AAA="},
        },
    )

    result = execution.default_transfer(
        Action("click", {"x": 300.0, "y": 250.0}),
        target,
        source,
    )

    assert result.action is not None
    assert calls[0]["source_screenshot_path"] == str(source_screenshot)
    assert calls[0]["target_screenshot_path"] == str(target_screenshot)
    assert calls[0]["source_visual_rgb"]["width"] == 1
    assert calls[0]["target_visual_rgb"]["height"] == 1


def test_function_transfer_captures_target_screenshot_before_matching(
    tmp_path: Path,
) -> None:
    source_screenshot = (tmp_path / "source.jpg").resolve()
    target_screenshot = (tmp_path / "target.jpg").resolve()
    source_screenshot.write_bytes(b"source")
    target_screenshot.write_bytes(b"target")
    transfer_observations: list[Observation] = []

    class Host:
        def observe(self, **kwargs):
            assert kwargs["screenshot"] is True
            return Observation(
                xml=TARGET_XML,
                package_name="com.example",
                extra={
                    "display": {"width": 100, "height": 200},
                    "screenshot_path": str(target_screenshot),
                },
            )

        def act(self, _action):
            return ActionResult(False, "stop_after_visual_probe")

    async def transfer(action, observation, _source_state):
        transfer_observations.append(observation)
        return TransferResult(action, reason="visual_probe")

    source = Observation(
        xml=SOURCE_XML,
        package_name="com.example",
        extra={
            "display": {"width": 100, "height": 200},
            "screenshot_path": str(source_screenshot),
        },
    )
    initial_target = Observation(
        xml=TARGET_XML,
        package_name="com.example",
        extra={"display": {"width": 100, "height": 200}},
    )

    asyncio.run(
        execution.execute_robust_action(
            Action("click", {"x": 300.0, "y": 250.0}),
            observation=initial_target,
            host=Host(),
            plugins=PluginSet(transfer=transfer),
            source_state=source,
        )
    )

    assert transfer_observations[0].extra["screenshot_path"] == str(
        target_screenshot
    )
