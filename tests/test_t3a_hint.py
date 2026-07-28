from pathlib import Path

from src.experiment.androidworld import (
    ArchivedRunLog,
    _parse_one_task_methods,
    build_official_androidworld_command,
)


def test_t3a_hint_is_a_supported_one_task_method() -> None:
    assert _parse_one_task_methods("t3a_hint") == ["t3a_hint"]


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
