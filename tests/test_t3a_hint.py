from pathlib import Path

import pytest

from omniflow.core.model import Action, Function, FunctionStep
from omniflow.functions.artifact import FUNCTION_ARTIFACT_VERSION
from omniflow.functions.store import FunctionStore
from src.experiment.androidworld import (
    ArchivedRunLog,
    _parse_one_task_methods,
    _promote_one_task_metadata_to_row,
    _select_complete_function,
    _t3a_hint_action_identity,
    _t3a_hint_step_action,
    build_official_androidworld_command,
)


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
    ) == ("open_app", {})


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


def test_fixed_replay_source_xml_metadata_is_promoted() -> None:
    row: dict[str, object] = {}

    _promote_one_task_metadata_to_row(
        row,
        [
            {
                "metadata": {
                    "execution_backend": "selector_then_scaled_coordinate_fallback_v2",
                    "uses_source_xml": True,
                }
            }
        ],
    )

    assert row["execution_backend"] == "selector_then_scaled_coordinate_fallback_v2"
    assert row["uses_source_xml"] is True


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
