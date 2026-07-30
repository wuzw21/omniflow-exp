from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from src.integrations.android_world.launch import _raw_replay_step_actions
from src.integrations.runlog import (
    convert_legacy_run_log,
    import_run_log,
    import_run_log_evidence,
)
from runlog_fixtures import androidworld_run_log, androidworld_state


def test_production_import_keeps_androidworld_state_and_action() -> None:
    xml = (
        '<hierarchy><node text="Record" resource-id="record" '
        'bounds="[400,700][680,900]" /></hierarchy>'
    )
    payload = androidworld_run_log(
        [{"action_type": "click", "x": 500, "y": 800}],
        observations=[
            androidworld_state(
                "source-state-0",
                forest=xml,
                package_name="com.example.recorder",
                width=1080,
                height=2400,
            )
        ],
        run_id="source-111",
        goal="Record audio.",
    )

    run_log, source_states = import_run_log_evidence(payload)

    assert run_log == payload
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


def test_production_import_rejects_legacy_schema() -> None:
    with pytest.raises(ValueError, match="run_log_schema_invalid"):
        import_run_log(
            {
                "schema_version": "omniflow.run_log.v1",
                "run_id": "legacy",
                "steps": [],
            }
        )


def test_production_import_rejects_retired_androidworld_schema_name() -> None:
    run_log = androidworld_run_log([{"action_type": "wait"}])
    run_log["schema_version"] = "omniflow.androidworld.run_log.v1"

    with pytest.raises(
        ValueError,
        match="run_log_schema_invalid:run_log.schema_version:const",
    ):
        import_run_log(run_log)


def test_production_import_accepts_androidworld_float_coordinates() -> None:
    run_log = androidworld_run_log(
        [{"action_type": "click", "x": 50.2, "y": 50.3}]
    )

    assert import_run_log(run_log) == run_log


@pytest.mark.parametrize(
    "action",
    [
        {"action_type": "open_app", "app_name": ""},
        {"action_type": "status", "goal_status": ""},
    ],
)
def test_production_import_keeps_androidworld_optional_empty_strings(
    action: dict[str, object],
) -> None:
    run_log = androidworld_run_log([action])

    assert import_run_log(run_log) == run_log


def test_production_import_rejects_nonofficial_action_type() -> None:
    run_log = androidworld_run_log(
        [{"action_type": "press_keyboard", "keycode": "KEYCODE_DEL"}]
    )

    with pytest.raises(ValueError, match="run_log_schema_invalid"):
        import_run_log(run_log)


def test_explicit_converter_emits_only_omniflow_schema(tmp_path: Path) -> None:
    source = tmp_path / "legacy.run_log.json"
    payload = {
        "schema_version": "omniflow.run_log.v1",
        "run_id": "legacy-source",
        "goal": "Open settings and tap Wi-Fi.",
        "success": True,
        "steps": [
            {
                "observation_before_act": {"width": 720, "height": 1280},
                "executed_actions": [
                    {"type": "open_app", "params": {"app_name": "settings"}}
                ],
                "success": True,
            },
            {
                "observation_before_act": {"width": 720, "height": 1280},
                "executed_actions": [
                    {"type": "click", "params": {"x": 360, "y": 640}}
                ],
                "success": True,
            },
        ],
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    converted = convert_legacy_run_log(
        payload,
        task_name="WifiTask",
        task_parameters={"enabled": True},
        seed=111,
        source_path=source,
        require_screenshots=False,
        package_resolver=lambda name: (
            "com.android.settings" if name == "settings" else ""
        ),
    )

    assert converted["schema_version"] == "omniflow.run_log.v1"
    assert converted["task_name"] == "WifiTask"
    assert converted["task_parameters"] == {"enabled": True}
    assert [step["action"] for step in converted["steps"]] == [
        {"action_type": "open_app", "app_name": "com.android.settings"},
        {"action_type": "click", "x": 360, "y": 640},
    ]


def test_explicit_converter_records_screenshot_reference(tmp_path: Path) -> None:
    screenshot = tmp_path / "screen.png"
    Image.new("RGB", (32, 48), color="white").save(screenshot)
    source = tmp_path / "legacy.run_log.json"
    payload = {
        "run_id": "legacy-source",
        "goal": "Wait.",
        "success": True,
        "steps": [
            {
                "observation_before_act": {
                    "width": 32,
                    "height": 48,
                },
                "source_context": {
                    "src_ctx": {"screenshot_path": str(screenshot)}
                },
                "action": {"type": "wait", "params": {}},
                "success": True,
            }
        ],
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    converted = convert_legacy_run_log(
        payload,
        task_name="WaitTask",
        task_parameters={},
        seed=111,
        source_path=source,
    )

    assert converted["steps"][0]["observation"]["pixels"] == {
        "path": str(screenshot.resolve()),
        "sha256": __import__("hashlib").sha256(screenshot.read_bytes()).hexdigest(),
        "width": 32,
        "height": 48,
        "mime_type": "image/png",
    }


def test_explicit_converter_marks_unavailable_screenshot_as_null(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy.run_log.json"
    payload = {
        "run_id": "legacy-source",
        "goal": "Wait.",
        "success": True,
        "steps": [
            {
                "observation_before_act": {"width": 32, "height": 48},
                "source_context": {
                    "src_ctx": {
                        "screenshot_path": str(tmp_path / "missing.png")
                    }
                },
                "action": {"type": "wait", "params": {}},
                "success": True,
            }
        ],
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    converted = convert_legacy_run_log(
        payload,
        task_name="WaitTask",
        task_parameters={},
        seed=111,
        source_path=source,
        require_screenshots=False,
    )

    assert converted["steps"][0]["observation"]["pixels"] is None


def test_explicit_converter_rejects_private_action(tmp_path: Path) -> None:
    source = tmp_path / "legacy.run_log.json"
    payload = {
        "run_id": "legacy-private-action",
        "success": True,
        "steps": [
            {
                "executed_actions": [
                    {"type": "set_clipboard", "params": {"text": "secret"}}
                ],
                "success": True,
            }
        ],
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="legacy_action_unsupported:set_clipboard"):
        convert_legacy_run_log(
            payload,
            task_name="ClipboardTask",
            task_parameters={},
            seed=111,
            source_path=source,
            require_screenshots=False,
        )


def test_explicit_converter_rejects_nonofficial_keyboard_action(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy.run_log.json"
    payload = {
        "run_id": "legacy-keyboard-action",
        "success": True,
        "steps": [
            {
                "action": {"type": "press_key", "params": {"key": "delete"}},
                "success": True,
            }
        ],
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="legacy_action_unsupported:press_key"):
        convert_legacy_run_log(
            payload,
            task_name="KeyboardTask",
            task_parameters={},
            seed=111,
            source_path=source,
            require_screenshots=False,
        )


def test_fixed_replay_accepts_only_omniflow_run_log() -> None:
    run_log = androidworld_run_log(
        [
            {"action_type": "open_app", "app_name": "com.android.settings"},
            {"action_type": "click", "x": 360, "y": 640},
        ],
        observations=[
            androidworld_state("launcher", width=720, height=1280),
            androidworld_state("settings", width=720, height=1280),
        ],
    )

    assert _raw_replay_step_actions(run_log) == [
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
