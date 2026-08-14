from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from PIL import Image
import pytest
from runlog_fixtures import androidworld_run_log, androidworld_state

from src.experiment.source_runlogs import convert_source_index
from src.integrations.android_world.launch import (
    _apply_fixed_replay,
    _fixed_replay_bind_action_parameters,
    _fixed_replay_goal_parameter_bindings,
    _launch_raw_replay_app,
    _raw_replay_action_to_payload,
    _raw_replay_step_actions,
)
from src.integrations.runlog import (
    adapt_source_run_log,
    convert_legacy_run_log,
    import_run_log,
    import_run_log_evidence,
    project_androidworld_step_actions,
)


def test_runlog_import_recovers_missing_display_from_fullscreen_xml() -> None:
    payload = androidworld_run_log(
        [
            {"action_type": "open_app", "app_name": "net.gsantner.markor"},
            {"action_type": "click", "x": 540, "y": 1200},
        ]
    )
    payload["steps"][0]["observation"] = {
        "pixels": None,
        "forest": (
            '<hierarchy rotation="0"><node class="android.widget.FrameLayout" '
            'bounds="[0,0][1080,2400]" /></hierarchy>'
        ),
        "ui_elements": [],
        "auxiliaries": {"state_id": "source-state-0"},
    }
    payload["steps"][1]["observation"] = {
        "pixels": None,
        "forest": (
            '<hierarchy rotation="0"><node class="android.widget.FrameLayout" '
            'bounds="[0,408][1080,1236]" /></hierarchy>'
        ),
        "ui_elements": [],
        "auxiliaries": {"state_id": "source-state-1"},
    }

    run_log, source_states = import_run_log_evidence(payload)

    assert run_log["steps"][1]["observation"]["auxiliaries"]["display"] == {
        "width": 1080,
        "height": 2400,
    }
    assert source_states["states"]["source-state-1"]["display"] == {
        "width": 1080,
        "height": 2400,
    }
    assert project_androidworld_step_actions(run_log["steps"][1]) == [
        {"tool": "click", "args": {"x": 500.0, "y": 500.0}}
    ]


def test_input_text_point_outside_editable_node_does_not_add_click() -> None:
    payload = androidworld_run_log(
        [{"action_type": "input_text", "text": "folder", "x": 374, "y": 778}]
    )
    payload["steps"][0]["observation"] = {
        "pixels": None,
        "forest": (
            '<hierarchy><node class="android.widget.LinearLayout" '
            'bounds="[84,492][996,1030]" /></hierarchy>'
        ),
        "ui_elements": [],
        "auxiliaries": {
            "state_id": "dialog",
            "display": {"width": 1080, "height": 2400},
        },
    }

    assert project_androidworld_step_actions(payload["steps"][0]) == [
        {"tool": "input_text", "args": {"text": "folder"}}
    ]


def test_runlog_import_rejects_conflicting_fullscreen_display_evidence() -> None:
    payload = androidworld_run_log(
        [
            {"action_type": "click", "x": 360, "y": 640},
            {"action_type": "click", "x": 540, "y": 1200},
        ]
    )
    for step, bounds in zip(
        payload["steps"],
        ("[0,0][720,1280]", "[0,0][1080,2400]"),
        strict=True,
    ):
        step["observation"] = {
            "pixels": None,
            "forest": (
                '<hierarchy rotation="0"><node class="android.widget.FrameLayout" '
                f'bounds="{bounds}" /></hierarchy>'
            ),
            "ui_elements": [],
            "auxiliaries": {"state_id": bounds},
        }

    with pytest.raises(ValueError, match="androidworld_run_log_display_conflict"):
        import_run_log_evidence(payload)


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


def test_production_import_accepts_official_press_keyboard_action() -> None:
    run_log = androidworld_run_log(
        [{"action_type": "press_keyboard", "keycode": "KEYCODE_DEL"}]
    )

    assert import_run_log(run_log) == run_log


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


def test_canonical_legacy_inline_observation_keeps_pixel_coordinates(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy.run_log.json"
    payload = {
        "schema_version": "omniflow.canonical_run_log.v1",
        "run_id": "legacy-inline-pixels",
        "goal": "Confirm deletion.",
        "success": True,
        "steps": [
            {
                "observation_before_act": {"width": 720, "height": 1280},
                "executed_actions": [
                    {"type": "click", "params": {"x": 637, "y": 717}}
                ],
                "success": True,
            },
            {
                "observation_before_act": {"width": 720, "height": 1280},
                "executed_actions": [
                    {"type": "click", "params": {"x": 360, "y": 1144}}
                ],
                "success": True,
            },
        ],
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    converted = convert_legacy_run_log(
        payload,
        task_name="DeleteTask",
        task_parameters={},
        seed=111,
        source_path=source,
        require_screenshots=False,
    )

    assert [step["action"] for step in converted["steps"]] == [
        {"action_type": "click", "x": 637, "y": 717},
        {"action_type": "click", "x": 360, "y": 1144},
    ]


def test_explicit_converter_preserves_filtered_source_target_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy-target.run_log.json"
    payload = {
        "run_id": "legacy-target",
        "success": True,
        "steps": [
            {
                "observation_before_act": {
                    "xml": '<hierarchy><node text="Continue" /></hierarchy>',
                    "width": 100,
                    "height": 100,
                },
                "action": {
                    "type": "click",
                    "params": {
                        "x": 50,
                        "y": 50,
                        "target_description": "Continue",
                        "source_context": {
                            "element": {
                                "text": "Continue",
                                "resource_id": "app:id/continue",
                                "bounds": [0, 0, 100, 100],
                            }
                        },
                        "target_evidence": {
                            "label": "Continue",
                            "resource-id": "app:id/continue",
                            "x": 50,
                            "y": 50,
                        },
                    },
                },
                "success": True,
            }
        ],
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    converted = convert_legacy_run_log(
        payload,
        task_name="TargetTask",
        task_parameters={},
        seed=111,
        source_path=source,
        require_screenshots=False,
    )

    assert converted["steps"][0]["metadata"]["source_target_evidence"] == {
        "target_description": "Continue",
        "element": {
            "text": "Continue",
            "resource_id": "app:id/continue",
        },
        "target": {
            "text": "Continue",
            "resource_id": "app:id/continue",
        },
    }


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


def test_source_adapter_keeps_official_xml_and_screenshot_reference(
    tmp_path: Path,
) -> None:
    screenshot = tmp_path / "official.png"
    Image.new("RGB", (32, 48), color="white").save(screenshot)
    run_log = androidworld_run_log(
        [{"action_type": "click", "x": 16, "y": 24}],
        observations=[
            androidworld_state(
                "official-state",
                forest='<hierarchy><node text="Open" /></hierarchy>',
                width=32,
                height=48,
            )
        ],
        task_name="OfficialTask",
    )
    run_log["steps"][0]["observation"]["pixels"] = {
        "path": str(screenshot.resolve()),
        "sha256": __import__("hashlib").sha256(screenshot.read_bytes()).hexdigest(),
        "width": 32,
        "height": 48,
        "mime_type": "image/png",
    }
    source = tmp_path / "official.run_log.json"
    source.write_text(json.dumps(run_log), encoding="utf-8")

    adapted = adapt_source_run_log(
        run_log,
        task_name="OfficialTask",
        task_parameters={},
        seed=111,
        source_path=source,
        screenshot_roots=(),
    )

    assert adapted == run_log


def test_source_adapter_normalizes_native_xml_and_screenshot_aliases(
    tmp_path: Path,
) -> None:
    screenshot = tmp_path / "native.png"
    Image.new("RGB", (32, 48), color="white").save(screenshot)
    source = tmp_path / "native.run_log.json"
    payload = {
        "run_id": "native-source",
        "goal": "Tap Open.",
        "success": True,
        "steps": [
            {
                "observation_before_act": {
                    "hierarchy_xml": '<hierarchy><node text="Open" /></hierarchy>',
                    "screenshot": str(screenshot),
                    "width": 32,
                    "height": 48,
                },
                "action": {"type": "click", "params": {"x": 16, "y": 24}},
                "success": True,
            }
        ],
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    adapted = adapt_source_run_log(
        payload,
        task_name="NativeTask",
        task_parameters={},
        seed=111,
        source_path=source,
        screenshot_roots=(),
    )

    observation = adapted["steps"][0]["observation"]
    assert observation["forest"] == '<hierarchy><node text="Open" /></hierarchy>'
    assert observation["pixels"] == {
        "path": str(screenshot.resolve()),
        "sha256": __import__("hashlib").sha256(screenshot.read_bytes()).hexdigest(),
        "width": 32,
        "height": 48,
        "mime_type": "image/png",
    }


def test_source_adapter_converts_legacy_log_with_current_schema_name(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy-current-name.run_log.json"
    payload = {
        "schema_version": "omniflow.run_log.v1",
        "run_id": "legacy-current-name",
        "trace_id": "legacy-trace",
        "goal": "Wait.",
        "completed": True,
        "success": True,
        "steps": [
            {
                "observation_before_act": {"width": 32, "height": 48},
                "actions": [{"type": "wait", "params": {}}],
                "success": True,
            }
        ],
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    adapted = adapt_source_run_log(
        payload,
        task_name="LegacyTask",
        task_parameters={},
        seed=111,
        source_path=source,
        require_screenshots=False,
    )

    assert adapted["task_name"] == "LegacyTask"
    assert adapted["steps"][0]["action"] == {"action_type": "wait"}


def test_legacy_start_activity_uses_following_observation_package(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy-start-activity.run_log.json"
    payload = {
        "run_id": "legacy-start-activity",
        "success": True,
        "steps": [
            {
                "observation_before_act": {
                    "package_name": "com.google.android.apps.nexuslauncher",
                    "width": 720,
                    "height": 1280,
                },
                "actions": [
                    {
                        "type": "start_activity",
                        "params": {"action": "android.settings.SETTINGS"},
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
                "actions": [{"type": "wait", "params": {}}],
                "success": True,
            },
        ],
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    converted = convert_legacy_run_log(
        payload,
        task_name="WifiTask",
        task_parameters={},
        seed=111,
        source_path=source,
        require_screenshots=False,
    )

    assert converted["steps"][0]["action"] == {
        "action_type": "open_app",
        "app_name": "com.android.settings",
    }


def test_source_index_accepts_official_runlog_without_screenshot_roots(
    tmp_path: Path,
) -> None:
    source = tmp_path / "official.run_log.json"
    run_log = androidworld_run_log(
        [{"action_type": "wait"}],
        task_name="OfficialTask",
    )
    source.write_text(json.dumps(run_log), encoding="utf-8")
    index = tmp_path / "source-index.json"
    index.write_text(
        json.dumps(
            {
                "OfficialTask": {
                    "retained_source_run_log": str(source),
                    "params": {},
                    "task_random_seed": 111,
                }
            }
        ),
        encoding="utf-8",
    )

    converted = convert_source_index(
        source_index=index,
        output_root=tmp_path / "converted",
        screenshot_roots=(),
    )

    assert converted["task_count"] == 1
    output_index = json.loads(
        Path(converted["output_index"]).read_text(encoding="utf-8")
    )
    output_run_log = json.loads(
        Path(output_index["OfficialTask"]["retained_source_run_log"]).read_text(
            encoding="utf-8"
        )
    )
    assert output_run_log == run_log


def test_source_index_hydrates_state_catalog_and_denormalizes_legacy_points(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy.run_log.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.canonical_run_log.v1",
                "run_id": "legacy-source",
                "goal": "Tap Continue.",
                "success": True,
                "steps": [
                    {
                        "step_index": 0,
                        "before_state_id": "state-before",
                        "after_state_id": "state-after",
                        "action": {
                            "tool": "click",
                            "args": {"x": 500, "y": 250},
                        },
                        "result": {"success": True},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    state_catalog = tmp_path / "transfer_states.json"
    state_catalog.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.transfer-state-catalog.v1",
                "run_id": "legacy-source",
                "states": {
                    "state-before": {
                        "state_id": "state-before",
                        "xml": (
                            '<hierarchy><node text="Continue" clickable="true" '
                            'bounds="[0,0][720,1280]" /></hierarchy>'
                        ),
                        "package_name": "com.example.app",
                        "activity_name": ".MainActivity",
                        "display": {"width": 720, "height": 1280},
                    },
                    "state-after": {
                        "state_id": "state-after",
                        "xml": "<hierarchy />",
                        "package_name": "com.example.app",
                        "activity_name": ".MainActivity",
                        "display": {"width": 720, "height": 1280},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    state_catalog_sha256 = hashlib.sha256(state_catalog.read_bytes()).hexdigest()
    index = tmp_path / "source-index.json"
    index.write_text(
        json.dumps(
            {
                "LegacyTask": {
                    "retained_source_run_log": str(source),
                    "transfer_state_catalog": str(state_catalog),
                    "transfer_state_catalog_sha256": state_catalog_sha256,
                    "params": {},
                    "task_random_seed": 111,
                }
            }
        ),
        encoding="utf-8",
    )

    converted = convert_source_index(
        source_index=index,
        output_root=tmp_path / "converted",
        screenshot_roots=(),
    )

    output_index = json.loads(
        Path(converted["output_index"]).read_text(encoding="utf-8")
    )
    output_row = output_index["LegacyTask"]
    output_run_log = json.loads(
        Path(output_row["retained_source_run_log"]).read_text(encoding="utf-8")
    )
    observation = output_run_log["steps"][0]["observation"]
    assert observation["forest"].startswith("<hierarchy>")
    assert observation["auxiliaries"] == {
        "state_id": "state-before",
        "package_name": "com.example.app",
        "activity_name": ".MainActivity",
        "display": {"width": 720, "height": 1280},
    }
    assert output_run_log["steps"][0]["action"] == {
        "action_type": "click",
        "x": 360,
        "y": 320,
    }
    assert output_row["source_state_catalog"] == str(state_catalog)
    assert output_row["source_state_catalog_sha256"] == state_catalog_sha256


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


def test_explicit_converter_preserves_private_action_as_unknown(
    tmp_path: Path,
) -> None:
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

    converted = convert_legacy_run_log(
        payload,
        task_name="ClipboardTask",
        task_parameters={},
        seed=111,
        source_path=source,
        require_screenshots=False,
    )

    assert converted["steps"][0]["action"] == {"action_type": "unknown"}
    assert converted["steps"][0]["metadata"]["legacy_action"] == {
        "type": "set_clipboard",
        "params": {"text": "secret"},
    }


def test_explicit_converter_preserves_nonofficial_keyboard_action(
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

    converted = convert_legacy_run_log(
        payload,
        task_name="KeyboardTask",
        task_parameters={},
        seed=111,
        source_path=source,
        require_screenshots=False,
    )

    assert converted["steps"][0]["action"] == {
        "action_type": "unknown",
        "keycode": "KEYCODE_DEL",
    }
    assert converted["steps"][0]["metadata"]["legacy_action"] == {
        "type": "press_key",
        "params": {"key": "delete"},
    }


def test_explicit_converter_uses_provider_online_action(tmp_path: Path) -> None:
    source = tmp_path / "legacy.run_log.json"
    payload = {
        "run_id": "legacy-online-action",
        "success": True,
        "steps": [
            {
                "goal_completed": True,
                "provider_detail": {
                    "online_action": {"type": "finished", "params": {}}
                },
            }
        ],
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    converted = convert_legacy_run_log(
        payload,
        task_name="FinishedTask",
        task_parameters={},
        seed=111,
        source_path=source,
        require_screenshots=False,
    )

    assert converted["steps"][0]["action"] == {
        "action_type": "status",
        "goal_status": "complete",
    }


def test_explicit_converter_avoids_system_edge_for_interior_legacy_swipe(
    tmp_path: Path,
) -> None:
    source = tmp_path / "interior_swipe.run_log.json"
    payload = {
        "run_id": "interior-swipe",
        "success": True,
        "steps": [
            {
                "observation_before_act": {"width": 720, "height": 1280},
                "action": {
                    "type": "swipe",
                    "params": {
                        "start_x": 100,
                        "start_y": 600,
                        "end_x": 600,
                        "end_y": 600,
                    },
                },
            }
        ],
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    converted = convert_legacy_run_log(
        payload,
        task_name="GestureTask",
        task_parameters={},
        seed=111,
        source_path=source,
        require_screenshots=False,
    )

    assert converted["steps"][0]["action"] == {
        "action_type": "scroll",
        "direction": "left",
    }


def test_explicit_converter_preserves_camera_gesture_and_wait_semantics(
    tmp_path: Path,
) -> None:
    source = tmp_path / "camera_take_video.run_log.json"
    payload = {
        "run_id": "camera-take-video",
        "success": True,
        "steps": [
            {
                "observation_before_act": {"width": 720, "height": 1280},
                "action": {
                    "type": "swipe",
                    "params": {
                        "start_x": 100,
                        "start_y": 600,
                        "end_x": 600,
                        "end_y": 600,
                        "duration_ms": 500,
                        "wait_after_s": 1.0,
                    },
                },
            },
            {
                "observation_before_act": {"width": 720, "height": 1280},
                "action": {
                    "type": "click",
                    "params": {"x": 200, "y": 580, "wait_after_s": 3.0},
                },
            },
            {
                "observation_before_act": {"width": 720, "height": 1280},
                "action": {"type": "wait", "params": {"time_s": 5.0}},
            },
            {
                "observation_before_act": {"width": 720, "height": 1280},
                "action": {
                    "type": "click",
                    "params": {"x": 360, "y": 1136, "wait_after_s": 2.0},
                },
            },
        ],
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    converted = convert_legacy_run_log(
        payload,
        task_name="CameraTakeVideo",
        task_parameters={},
        seed=111,
        source_path=source,
        require_screenshots=False,
    )

    assert [step["action"] for step in converted["steps"]] == [
        {"action_type": "scroll", "direction": "left"},
        {"action_type": "click", "x": 200, "y": 580},
        {"action_type": "wait"},
        {"action_type": "wait"},
        *[{"action_type": "wait"} for _ in range(5)],
        {"action_type": "click", "x": 360, "y": 1136},
        {"action_type": "wait"},
    ]


@pytest.mark.parametrize(
    ("action_type", "end", "expected_action_type", "expected_direction"),
    [
        ("swipe", (900, 500), "scroll", "left"),
        ("swipe", (100, 500), "scroll", "right"),
        ("swipe", (500, 900), "scroll", "up"),
        ("swipe", (500, 100), "scroll", "down"),
        ("scroll", (900, 500), "scroll", "left"),
        ("scroll", (100, 500), "scroll", "right"),
        ("scroll", (500, 900), "scroll", "up"),
        ("scroll", (500, 100), "scroll", "down"),
    ],
)
def test_explicit_converter_maps_endpoint_gestures_to_androidworld_direction(
    tmp_path: Path,
    action_type: str,
    end: tuple[int, int],
    expected_action_type: str,
    expected_direction: str,
) -> None:
    source = tmp_path / f"{action_type}-{expected_direction}.run_log.json"
    payload = {
        "run_id": "endpoint-gesture",
        "success": True,
        "steps": [
            {
                "observation_before_act": {"width": 1000, "height": 1000},
                "action": {
                    "type": action_type,
                    "params": {
                        "start_x": 500,
                        "start_y": 500,
                        "end_x": end[0],
                        "end_y": end[1],
                    },
                },
            }
        ],
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    converted = convert_legacy_run_log(
        payload,
        task_name="GestureTask",
        task_parameters={},
        seed=111,
        source_path=source,
        require_screenshots=False,
    )

    assert converted["steps"][0]["action"] == {
        "action_type": expected_action_type,
        "direction": expected_direction,
    }


def test_explicit_converter_preserves_system_edge_swipe(tmp_path: Path) -> None:
    source = tmp_path / "edge_swipe.run_log.json"
    payload = {
        "run_id": "edge-swipe",
        "success": True,
        "steps": [
            {
                "observation_before_act": {"width": 720, "height": 1280},
                "action": {
                    "type": "swipe",
                    "params": {
                        "start_x": 0,
                        "start_y": 600,
                        "end_x": 600,
                        "end_y": 600,
                    },
                },
            }
        ],
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    converted = convert_legacy_run_log(
        payload,
        task_name="GestureTask",
        task_parameters={},
        seed=111,
        source_path=source,
        require_screenshots=False,
    )

    assert converted["steps"][0]["action"] == {
        "action_type": "swipe",
        "direction": "left",
    }


def test_fixed_replay_accepts_only_omniflow_run_log() -> None:
    settings_xml = (
        '<hierarchy><node text="Network &amp; internet" '
        'resource-id="com.android.settings:id/network_dashboard" '
        'bounds="[100,500][620,780]" clickable="true" /></hierarchy>'
    )
    run_log = androidworld_run_log(
        [
            {"action_type": "open_app", "app_name": "com.android.settings"},
            {"action_type": "click", "x": 360, "y": 640},
        ],
        observations=[
            androidworld_state("launcher", width=720, height=1280),
            androidworld_state(
                "settings",
                forest=settings_xml,
                width=720,
                height=1280,
            ),
        ],
    )

    assert _raw_replay_step_actions(run_log) == [
        {
            "type": "open_app",
            "params": {"package_name": "com.android.settings"},
        },
        {
            "type": "click",
            "params": {
                "selector": {
                    "text": "Network & internet",
                    "resource_id": "com.android.settings:id/network_dashboard",
                },
                "x": 500.0,
                "y": 500.0,
            },
            "coordinate_space": "canonical_0_1000",
        },
    ]


def test_fixed_replay_binds_goal_parameter_to_input_text() -> None:
    run_log = androidworld_run_log(
        [{"action_type": "input_text", "text": "source_name.m4a"}],
        goal='Save the recording as "source_name.m4a".',
    )
    run_log["task_parameters"] = {"file_name": "source_name.m4a"}

    report = _fixed_replay_goal_parameter_bindings(
        run_log,
        target_goal='Save the recording as "target_name.m4a".',
    )
    bound, changed = _fixed_replay_bind_action_parameters(
        _raw_replay_step_actions(run_log)[0],
        report["bindings"],
    )

    assert report["status"] == "matched_goal_template"
    assert report["bindings"] == [
        {
            "source_parameter_paths": ["$.file_name"],
            "source_value": "source_name.m4a",
            "target_value": "target_name.m4a",
            "changed": True,
        }
    ]
    assert bound == {
        "type": "input_text",
        "params": {"text": "target_name.m4a"},
    }
    assert changed == [
        {
            "action_path": "$.params.text",
            "source_parameter_paths": ["$.file_name"],
            "source_value": "source_name.m4a",
            "target_value": "target_name.m4a",
            "match": "exact",
        }
    ]


def test_fixed_replay_goal_binding_does_not_read_hidden_parameters() -> None:
    run_log = androidworld_run_log(
        [{"action_type": "input_text", "text": "visible"}],
        goal='Enter "visible".',
    )
    run_log["task_parameters"] = {
        "visible": "visible",
        "hidden_validator_value": "secret-source",
    }

    report = _fixed_replay_goal_parameter_bindings(
        run_log,
        target_goal='Enter "changed".',
    )

    assert report["bindings"] == [
        {
            "source_parameter_paths": ["$.visible"],
            "source_value": "visible",
            "target_value": "changed",
            "changed": True,
        }
    ]


def test_fixed_replay_preserves_androidworld_directional_gestures() -> None:
    run_log = androidworld_run_log(
        [
            {"action_type": "swipe", "direction": "right"},
            {"action_type": "scroll", "direction": "down"},
        ],
        observations=[
            androidworld_state("camera", width=720, height=1280),
            androidworld_state("list", width=720, height=1280),
        ],
    )

    replay_actions = _raw_replay_step_actions(run_log)

    assert replay_actions == [
        {"type": "swipe", "params": {"direction": "right"}},
        {"type": "scroll", "params": {"direction": "down"}},
    ]
    assert _raw_replay_action_to_payload(
        replay_actions[0],
        source_size=(720, 1280),
        target_size=(2208, 1840),
    ) == ({"action_type": "swipe", "direction": "right"}, None)


def test_fixed_replay_resolves_click_from_target_selector() -> None:
    target_xml = (
        '<hierarchy><node text="Network &amp; internet" '
        'resource-id="com.android.settings:id/network_dashboard" '
        'bounds="[200,800][600,1000]" clickable="true" /></hierarchy>'
    )

    payload, error = _raw_replay_action_to_payload(
        {
            "type": "click",
            "params": {
                "selector": {
                    "text": "Network & internet",
                    "resource_id": "com.android.settings:id/network_dashboard",
                }
            },
        },
        source_size=(720, 1280),
        target_size=(800, 1600),
        target_xml=target_xml,
    )

    assert error is None
    assert payload == {"action_type": "click", "x": 400, "y": 900}


def test_fixed_replay_normalizes_selector_against_observation_display(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_log_path = tmp_path / "fold.run_log.json"
    run_log_path.write_text(
        json.dumps(
            androidworld_run_log(
                [{"action_type": "click", "x": 360, "y": 640}],
                observations=[
                    androidworld_state(
                        "source",
                        forest=(
                            '<hierarchy><node text="Target" '
                            'bounds="[300,500][420,780]" clickable="true" />'
                            "</hierarchy>"
                        ),
                        width=720,
                        height=1280,
                    )
                ],
            )
        ),
        encoding="utf-8",
    )
    acted: list[dict[str, object]] = []
    host = SimpleNamespace(
        observe=lambda **_kwargs: SimpleNamespace(
            xml=(
                '<hierarchy><node text="Target" '
                'bounds="[1000,600][1200,800]" clickable="true" />'
                "</hierarchy>"
            ),
            package_name="com.example",
            activity_name="com.example/.MainActivity",
            extra={
                "observe_backend": "androidworld",
                "display": {"width": 2208, "height": 1840},
            },
        ),
        act=lambda action: acted.append(action) or SimpleNamespace(success=True),
    )
    agent = SimpleNamespace(
        env=SimpleNamespace(
            device_screen_size=(2208, 1840),
            logical_screen_size=(1080, 2092),
            controller=object(),
        ),
        host=host,
        set_max_steps=lambda _steps: None,
    )
    android_world = ModuleType("android_world")
    android_world_env = ModuleType("android_world.env")
    android_world_env.actuation = SimpleNamespace()
    android_world_env.adb_utils = SimpleNamespace()
    android_world_env.json_action = SimpleNamespace()
    android_world.env = android_world_env
    monkeypatch.setitem(sys.modules, "android_world", android_world)
    monkeypatch.setitem(sys.modules, "android_world.env", android_world_env)

    _apply_fixed_replay(agent, run_log_json_path=str(run_log_path))
    agent.step("Tap Target")

    assert acted == [
        {
            "tool": "click",
            "args": {
                "x": pytest.approx(1100 / 2208 * 1000),
                "y": pytest.approx(700 / 1840 * 1000),
            },
        }
    ]


def test_fixed_replay_resolves_selector_within_foreground_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_log_path = tmp_path / "fold.run_log.json"
    run_log_path.write_text(
        json.dumps(
            androidworld_run_log(
                [{"action_type": "click", "x": 356, "y": 781}],
                observations=[
                    androidworld_state(
                        "source",
                        forest=(
                            '<hierarchy><node text="Phone" '
                            'package="com.google.android.contacts" '
                            'bounds="[104,725][608,838]" clickable="true" '
                            'editable="true" /></hierarchy>'
                        ),
                        package_name="com.google.android.contacts",
                        width=720,
                        height=1280,
                    )
                ],
            )
        ),
        encoding="utf-8",
    )
    acted: list[dict[str, object]] = []
    host = SimpleNamespace(
        observe=lambda **_kwargs: SimpleNamespace(
            xml=(
                '<hierarchy><node text="Phone" '
                'package="com.google.android.contacts" '
                'bounds="[1083,896][2061,1045]" clickable="true" '
                'editable="true" />'
                '<node text="Phone" content-desc="Phone" '
                'package="com.google.android.apps.nexuslauncher" '
                'bounds="[679,1697][806,1824]" clickable="true" />'
                "</hierarchy>"
            ),
            package_name="com.google.android.contacts",
            activity_name="com.google.android.contacts/.ContactEditorActivity",
            extra={
                "observe_backend": "androidworld",
                "display": {"width": 2208, "height": 1840},
            },
        ),
        act=lambda action: acted.append(action) or SimpleNamespace(success=True),
    )
    agent = SimpleNamespace(
        env=SimpleNamespace(
            device_screen_size=(2208, 1840),
            logical_screen_size=(1080, 2092),
            controller=object(),
        ),
        host=host,
        set_max_steps=lambda _steps: None,
    )
    android_world = ModuleType("android_world")
    android_world_env = ModuleType("android_world.env")
    android_world_env.actuation = SimpleNamespace()
    android_world_env.adb_utils = SimpleNamespace()
    android_world_env.json_action = SimpleNamespace()
    android_world.env = android_world_env
    monkeypatch.setitem(sys.modules, "android_world", android_world)
    monkeypatch.setitem(sys.modules, "android_world.env", android_world_env)

    _apply_fixed_replay(agent, run_log_json_path=str(run_log_path))
    agent.step("Enter a phone number")

    assert acted == [
        {
            "tool": "click",
            "args": {
                "x": pytest.approx(1572 / 2208 * 1000),
                "y": pytest.approx(970 / 1840 * 1000),
            },
        }
    ]


def test_fixed_replay_opens_packages_through_androidworld_launcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = object()
    calls: list[dict[str, object]] = []
    adb_utils = SimpleNamespace(
        get_all_apps=lambda actual_controller: (
            ["settings"] if actual_controller is controller else []
        ),
        get_adb_activity=lambda app: (
            "com.android.settings/.Settings" if app == "settings" else None
        ),
        extract_package_name=lambda activity: activity.split("/", 1)[0],
    )
    android_world = ModuleType("android_world")
    android_world_env = ModuleType("android_world.env")
    android_world_env.adb_utils = adb_utils
    android_world.env = android_world_env
    monkeypatch.setitem(sys.modules, "android_world", android_world)
    monkeypatch.setitem(sys.modules, "android_world.env", android_world_env)

    host = SimpleNamespace(
        env=SimpleNamespace(controller=controller),
        act=lambda action: calls.append(action) or SimpleNamespace(success=True),
    )
    _launch_raw_replay_app("com.android.settings", host)

    assert calls == [
        {"tool": "open_app", "args": {"package_name": "com.android.settings"}},
    ]


def test_fixed_replay_waits_for_open_app_before_resolving_selector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_log_path = tmp_path / "contacts.run_log.json"
    run_log_path.write_text(
        json.dumps(
            androidworld_run_log(
                [
                    {
                        "action_type": "open_app",
                        "app_name": "com.google.android.contacts",
                    },
                    {"action_type": "click", "x": 650, "y": 950},
                ],
                observations=[
                    androidworld_state(
                        "contacts-launch",
                        package_name="com.google.android.contacts",
                        width=720,
                        height=1280,
                    ),
                    androidworld_state(
                        "contacts-home",
                        forest=(
                            '<hierarchy><node content-desc="Create contact" '
                            'bounds="[600,900][700,1000]" clickable="true" />'
                            "</hierarchy>"
                        ),
                        package_name="com.google.android.contacts",
                        width=720,
                        height=1280,
                    ),
                ],
            )
        ),
        encoding="utf-8",
    )
    launcher_xml = (
        '<hierarchy><node package="com.google.android.apps.nexuslauncher" '
        'bounds="[0,0][2208,1840]" /></hierarchy>'
    )
    contacts_xml = (
        '<hierarchy><node content-desc="Create contact" '
        'package="com.google.android.contacts" '
        'bounds="[1998,1283][2145,1430]" clickable="true" /></hierarchy>'
    )
    current_page = {"xml": launcher_xml, "package": "com.google.android.apps.nexuslauncher"}
    acted: list[dict[str, object]] = []

    def observe(**_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            xml=current_page["xml"],
            package_name=current_page["package"],
            activity_name=f'{current_page["package"]}/.MainActivity',
            extra={
                "observe_backend": "androidworld",
                "display": {"width": 2208, "height": 1840},
            },
        )

    def act(action: dict[str, object]) -> SimpleNamespace:
        acted.append(action)
        if action["tool"] == "open_app":
            current_page.update(
                xml=contacts_xml,
                package="com.google.android.contacts",
            )
        return SimpleNamespace(success=True)

    controller = object()
    env = SimpleNamespace(
        device_screen_size=(2208, 1840),
        logical_screen_size=(1080, 2092),
        controller=controller,
    )
    agent = SimpleNamespace(
        env=env,
        host=SimpleNamespace(env=env, observe=observe, act=act),
        set_max_steps=lambda _steps: None,
    )
    adb_utils = SimpleNamespace(
        get_all_apps=lambda actual_controller: (
            ["contacts"] if actual_controller is controller else []
        ),
        get_adb_activity=lambda app: (
            "com.google.android.contacts/.PeopleActivity"
            if app == "contacts"
            else None
        ),
        extract_package_name=lambda activity: activity.split("/", 1)[0],
    )
    android_world = ModuleType("android_world")
    android_world_env = ModuleType("android_world.env")
    android_world_env.actuation = SimpleNamespace()
    android_world_env.adb_utils = adb_utils
    android_world_env.json_action = SimpleNamespace()
    android_world.env = android_world_env
    monkeypatch.setitem(sys.modules, "android_world", android_world)
    monkeypatch.setitem(sys.modules, "android_world.env", android_world_env)

    _apply_fixed_replay(agent, run_log_json_path=str(run_log_path))
    agent.step("Create a contact")

    assert acted == [
        {
            "tool": "open_app",
            "args": {"package_name": "com.google.android.contacts"},
        },
        {
            "tool": "click",
            "args": {
                "x": pytest.approx(2071 / 2208 * 1000),
                "y": pytest.approx(1356 / 1840 * 1000),
            },
        },
    ]


def test_fixed_replay_scales_coordinates_only_without_selector() -> None:
    payload, error = _raw_replay_action_to_payload(
        {
            "type": "click",
            "params": {"x": 500, "y": 500},
            "coordinate_space": "canonical_0_1000",
        },
        source_size=(720, 1280),
        target_size=(1440, 2560),
    )

    assert error is None
    assert payload == {"action_type": "click", "x": 720, "y": 1280}


def test_fixed_replay_scales_source_coordinates_when_selector_misses() -> None:
    resolution: dict[str, object] = {}
    payload, error = _raw_replay_action_to_payload(
        {
            "type": "click",
            "params": {
                "selector": {"text": "Missing target"},
                "x": 500,
                "y": 500,
            },
            "coordinate_space": "canonical_0_1000",
        },
        source_size=(720, 1280),
        target_size=(1440, 2560),
        target_xml=(
            '<hierarchy><node text="Different target" '
            'bounds="[100,100][300,300]" /></hierarchy>'
        ),
        resolution=resolution,
    )

    assert error is None
    assert payload == {"action_type": "click", "x": 720, "y": 1280}
    assert resolution == {
        "parameter_source": "scaled_coordinate_fallback",
        "selector_error": "selector_target_not_found",
    }


def test_fixed_replay_scales_source_coordinates_when_selector_is_ambiguous() -> None:
    resolution: dict[str, object] = {}
    payload, error = _raw_replay_action_to_payload(
        {
            "type": "click",
            "params": {
                "selector": {
                    "relation": "unique_actionable_descendant",
                    "container_anchor": {"text": "task.html"},
                },
                "x": 500,
                "y": 500,
            },
            "coordinate_space": "canonical_0_1000",
        },
        source_size=(720, 1280),
        target_size=(1440, 2560),
        target_xml=(
            '<hierarchy><node><node text="task.html" /></node>'
            '<node><node text="task.html" /></node></hierarchy>'
        ),
        resolution=resolution,
    )

    assert error is None
    assert payload == {"action_type": "click", "x": 720, "y": 1280}
    assert resolution == {
        "parameter_source": "scaled_coordinate_fallback",
        "selector_error": "selector_container_anchor_ambiguous",
    }


def test_fixed_replay_resolves_structural_selector() -> None:
    target_xml = (
        '<hierarchy><node bounds="[0,0][600,300]">'
        '<node text="Dreamer&apos;s Awake" bounds="[20,40][400,180]" />'
        '<node clickable="true" bounds="[440,40][580,180]" />'
        "</node></hierarchy>"
    )

    payload, error = _raw_replay_action_to_payload(
        {
            "type": "click",
            "params": {
                "selector": {
                    "relation": "unique_actionable_descendant",
                    "container_anchor": {"text": "Dreamer's Awake"},
                }
            },
        },
        source_size=(600, 300),
        target_size=(600, 300),
        target_xml=target_xml,
    )

    assert error is None
    assert payload == {"action_type": "click", "x": 510, "y": 110}
