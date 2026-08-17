from pathlib import Path

import pytest

from omniflow.core.model import Action, Function, FunctionStep
from omniflow.functions.assets import FUNCTION_ARTIFACT_VERSION, FunctionStore
from src.experiment.androidworld import (
    ArchivedRunLog,
    _parse_one_task_methods,
    _promote_one_task_metadata_to_row,
    _select_complete_function,
    _t3a_hint_action_identity,
    _t3a_hint_step_action,
    _t3a_semantic_hint_step,
    build_official_androidworld_command,
)
from src.integrations.android_world.launch import _render_official_reference_prompt


def test_formal_one_task_method_set_is_exact() -> None:
    assert _parse_one_task_methods("all") == [
        "fixed_replay",
        "ours",
        "mobilegpt_offline_retrieval",
        "appagent_demo",
        "t3a_hint",
    ]
    with pytest.raises(ValueError, match="Unsupported one-task method"):
        _parse_one_task_methods("mobile_agent_v3")


def test_t3a_hint_is_a_supported_one_task_method() -> None:
    assert _parse_one_task_methods("t3a_hint") == ["t3a_hint"]


def test_t3a_hint_reads_androidworld_action_type() -> None:
    assert _t3a_hint_step_action(
        {"action": {"action_type": "open_app", "app_name": "settings"}}
    ) == ("open_app", {"app_name": "settings"})


def test_t3a_hint_preserves_semantic_target_and_source_node() -> None:
    semantic = _t3a_semantic_hint_step(
        {
            "action": {"action_type": "click", "x": 632, "y": 1032},
            "metadata": {
                "summary": (
                    'Action selected: {"action_type": "click", "index": 30}. '
                    'Clicked the "+" button (index 30) to create a new file.'
                )
            },
            "observation": {
                "forest": (
                    '<hierarchy><node id="30" class="android.widget.ImageButton" '
                    'text="" content-desc="Create a new file or folder" '
                    'resource-id="net.gsantner.markor:id/fab_add_new_item" '
                    'package="net.gsantner.markor" bounds="[576,976][688,1088]" '
                    'clickable="true" editable="false" scrollable="false" />'
                    "</hierarchy>"
                )
            },
        },
        forbidden_values=(),
    )

    assert semantic == {
        "action": "click",
        "target": "+ button",
        "purpose": 'Clicked the "+" button (index 30) to create a new file.',
        "source_node": {
            "node_id": "30",
            "class_name": "android.widget.ImageButton",
            "content_description": "Create a new file or folder",
            "resource_id": "net.gsantner.markor:id/fab_add_new_item",
            "package_name": "net.gsantner.markor",
        },
    }


def test_t3a_hint_reads_canonical_runlog_xml_with_android_namespace() -> None:
    semantic = _t3a_semantic_hint_step(
        {
            "tool_call": {
                "name": "click",
                "params": {
                    "target_description": "Create a new file or folder",
                    "x": 632.0,
                    "y": 1032.0,
                },
                "reason": "Click Create a new file or folder.",
            },
            "observation_before_act": {
                "xml": (
                    '<hierarchy xmlns="http://schemas.android.com/apk/res/android">'
                    '<node id="30" class="android.widget.ImageButton" '
                    'content-desc="Create a new file or folder" '
                    'resource-id="net.gsantner.markor:id/fab_add_new_item" '
                    'bounds="[576,976][688,1088]" clickable="true" />'
                    "</hierarchy>"
                )
            },
        },
        forbidden_values=(),
    )

    assert semantic is not None
    assert semantic["target"] == "Create a new file or folder"
    assert semantic["purpose"] == "Click Create a new file or folder."
    assert semantic["source_node"] == {
        "node_id": "30",
        "class_name": "android.widget.ImageButton",
        "content_description": "Create a new file or folder",
        "resource_id": "net.gsantner.markor:id/fab_add_new_item",
    }


def test_t3a_hint_uses_focused_editable_node_for_unlocated_text_input() -> None:
    semantic = _t3a_semantic_hint_step(
        {
            "action": {"action_type": "input_text", "text": "Paris"},
            "observation": {
                "forest": (
                    '<hierarchy><node id="9" class="android.widget.EditText" '
                    'text="Type to search all" content-desc="" '
                    'resource-id="net.osmand:id/search_text" package="net.osmand" '
                    'bounds="[144,48][720,160]" focused="true" '
                    'editable="true" clickable="true" />'
                    "</hierarchy>"
                )
            },
        },
        forbidden_values=("Paris",),
    )

    assert semantic == {
        "action": "input_text",
        "target": "Type to search all",
        "source_node": {
            "node_id": "9",
            "class_name": "android.widget.EditText",
            "text": "Type to search all",
            "resource_id": "net.osmand:id/search_text",
            "package_name": "net.osmand",
        },
    }

    semantic_without_focus = _t3a_semantic_hint_step(
        {
            "action": {"action_type": "input_text", "text": "Paris"},
            "observation": {
                "forest": (
                    '<hierarchy><node id="9" class="android.widget.EditText" '
                    'text="Type to search all" resource-id="net.osmand:id/search_text" '
                    'package="net.osmand" bounds="[144,48][720,160]" '
                    'editable="true" />'
                    "</hierarchy>"
                )
            },
        },
        forbidden_values=("Paris",),
    )
    assert semantic_without_focus == semantic


def test_t3a_hint_uses_immediately_selected_editable_for_keyboard_obscured_input() -> None:
    semantic = _t3a_semantic_hint_step(
        {
            "action": {
                "action_type": "input_text",
                "clear_text": True,
                "text": "New track name",
            },
            "observation": {
                "forest": (
                    '<hierarchy><node id="0" bounds="[263,273][343,353]" />'
                    "</hierarchy>"
                )
            },
        },
        preceding_step={
            "action": {"action_type": "click", "x": 320, "y": 675},
            "observation": {
                "forest": (
                    '<hierarchy><node id="14" class="android.widget.EditText" '
                    'text="Default name" bounds="[32,617][688,728]" '
                    'editable="true" clickable="true" /></hierarchy>'
                )
            },
        },
        forbidden_values=("New track name",),
    )

    assert semantic == {
        "action": "input_text",
        "target": "editable text field selected by the preceding action",
        "source_node": {
            "node_id": "14",
            "class_name": "android.widget.EditText",
            "text": "Default name",
        },
    }

    with pytest.raises(ValueError, match="t3a_hint_unidentified_target:input_text"):
        _t3a_semantic_hint_step(
            {
                "action": {"action_type": "input_text", "text": "value"},
                "observation": {"forest": "<hierarchy />"},
            },
            preceding_step={
                "action": {"action_type": "click", "x": 20, "y": 20},
                "observation": {
                    "forest": (
                        '<hierarchy><node id="7" text="Save" '
                        'bounds="[0,0][40,40]" clickable="true" /></hierarchy>'
                    )
                },
            },
            forbidden_values=("value",),
        )


def test_t3a_hint_rejects_unidentified_pointer_action() -> None:
    with pytest.raises(ValueError, match="t3a_hint_unidentified_target:click"):
        _t3a_semantic_hint_step(
            {"action": {"action_type": "click", "x": 10, "y": 20}},
            forbidden_values=(),
        )


def test_t3a_hint_redacts_old_values_and_renders_node_evidence() -> None:
    semantic = _t3a_semantic_hint_step(
        {
            "action": {
                "action_type": "input_text",
                "clear_text": True,
                "text": "old_note",
            },
            "metadata": {
                "summary": (
                    'Action selected: {"action_type": "input_text", "index": 12}. '
                    'Renamed the note to "old_note" in the name field (index 12).'
                )
            },
            "observation": {
                "forest": (
                    '<hierarchy><node id="12" class="android.widget.EditText" '
                    'text="old_note" content-desc="" resource-id="name_input" '
                    'package="example" bounds="[64,399][433,482]" '
                    'clickable="true" editable="true" scrollable="false" />'
                    "</hierarchy>"
                )
            },
        },
        forbidden_values=("old_note",),
    )

    assert semantic is not None
    assert "old_note" not in str(semantic)
    assert semantic["target"] == "name field"
    prompt, rendered_steps = _render_official_reference_prompt([semantic])
    assert rendered_steps == 1
    assert "name field" in prompt
    assert "resource-id='name_input'" in prompt
    assert "class='android.widget.EditText'" in prompt
    assert "source coordinates" in prompt
    assert "old_note" not in prompt


@pytest.mark.parametrize(
    ("function_action", "runlog_action", "identity"),
    [
        (
            {"action": {"tool": "press_key", "args": {"key": "back"}}},
            {"action": {"action_type": "navigate_back"}},
            "navigate_back",
        ),
        (
            {"action": {"tool": "press_key", "args": {"key": "home"}}},
            {"action": {"action_type": "navigate_home"}},
            "navigate_home",
        ),
        (
            {"action": {"tool": "press_key", "args": {"key": "enter"}}},
            {"action": {"action_type": "keyboard_enter"}},
            "keyboard_enter",
        ),
    ],
)
def test_t3a_hint_alignment_normalizes_androidworld_key_aliases(
    function_action: dict[str, object],
    runlog_action: dict[str, object],
    identity: str,
) -> None:
    assert _t3a_hint_action_identity(function_action) == identity
    assert _t3a_hint_action_identity(runlog_action) == identity


def test_fixed_replay_coordinate_metadata_is_promoted() -> None:
    row: dict[str, object] = {}

    _promote_one_task_metadata_to_row(
        row,
        [
            {
                "metadata": {
                    "execution_backend": "recorded_coordinate_replay_v1",
                    "uses_source_xml": False,
                }
            }
        ],
    )

    assert row["execution_backend"] == "recorded_coordinate_replay_v1"
    assert row["uses_source_xml"] is False


def _put_function(
    store: FunctionStore,
    function_id: str,
    tools: tuple[str, ...],
) -> None:
    store.put_function(
        Function(
            function_id=function_id,
            name=function_id.replace("_", " "),
            description=f"Complete {function_id}.",
            steps=tuple(
                FunctionStep(
                    step_index=index,
                    source_state_id=f"{function_id}_state_{index}",
                    action=Action(
                        tool,
                        (
                            {"package_name": "com.android.settings"}
                            if tool == "open_app"
                            else {"duration_ms": 100}
                        ),
                    ),
                )
                for index, tool in enumerate(tools)
            ),
            schema_version=FUNCTION_ARTIFACT_VERSION,
            input_schema={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            checker_rules=(),
            agent_visible=True,
        )
    )


def test_t3a_hint_selects_unique_function_containing_all_subtraces(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "store.json"
    store = FunctionStore(store_path)
    _put_function(store, "partial_wait", ("wait",))
    _put_function(store, "complete_run_settings", ("open_app", "wait"))

    selected = _select_complete_function(store_path)

    assert selected.id == "complete_run_settings"


def test_t3a_hint_uses_official_t3a_with_source_trace(tmp_path: Path) -> None:
    source = tmp_path / "source.run_log.json"
    source.write_text("{}", encoding="utf-8")
    hint = tmp_path / "source_action_hints.json"
    hint.write_text("{}", encoding="utf-8")
    item = ArchivedRunLog(
        task="SystemBluetoothTurnOff",
        goal="Turn Bluetooth off.",
        params={},
        source_run_log=source,
        replay_seed=111,
        step_count=5,
        meta={},
    )

    spec = build_official_androidworld_command(
        item,
        android_world_root=tmp_path / "android_world",
        output_root=tmp_path / "results",
        method_name="t3a_hint",
        official_agent_name="t3a_gpt4",
        source_action_hint_path=hint,
        device_label="fold5564",
        serial="emulator-5564",
        console_port=5564,
        max_steps=30,
        timeout_sec=1200,
        task_random_seed=113,
        fixed_task_seed=True,
        fixed_task_params=False,
        perform_emulator_setup=True,
        repo_root=tmp_path,
    )

    assert spec.metadata["method"] == "t3a_hint"
    assert spec.metadata["official_agent_name"] == "t3a_gpt4"
    assert spec.metadata["uses_source_action_hints"] is True
    assert spec.metadata["fixed_task_params"] is False
    assert spec.metadata["device"] == "fold5564"
    assert "--source-action-hint-path" in spec.argv
    assert "official:t3a_gpt4" in spec.argv
    assert "--task-params-json" not in spec.argv
