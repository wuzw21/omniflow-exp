from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from runlog_fixtures import androidworld_run_log, androidworld_state

from src.integrations.mobilegpt_converter import (
    MobileGPTConversionError,
    _load_runlog_trajectory,
    _temporary_agent_query_provider,
    convert_runlog_to_mobilegpt_memory,
    preflight_runlog_conversion,
    validate_mobilegpt_memory,
    write_conversion_failure_audit,
)

MOBILEGPT_ROOT = Path(
    os.environ.get(
        "MOBILEGPT_TEST_ROOT",
        "/Users/wuzewen/Projects/Omni/OmniFlow/runtime/external/mobilegpt-official",
    )
)


def test_runtime_query_provider_keeps_original_signature() -> None:
    calls: list[tuple[object, object, object]] = []

    def runtime_query(
        messages: list[dict],
        model: str | None = None,
        is_list: bool = False,
    ) -> object:
        calls.append((messages, model, is_list))
        return {"ok": True}

    module = SimpleNamespace(
        __name__="agents.explore_agent",
        query=runtime_query,
    )
    messages = [{"role": "user", "content": "prompt"}]

    with _temporary_agent_query_provider((module,), None):
        assert module.query(messages, model="qwen3-vl-plus", is_list=True) == {
            "ok": True
        }

    assert calls == [(messages, "qwen3-vl-plus", True)]
    assert module.query is runtime_query


def _semantic_query(
    messages: list[dict],
    *,
    model: str | None = None,
    is_list: bool = False,
    agent_name: str = "unknown",
) -> object:
    del model
    prompt = "\n".join(str(message.get("content") or "") for message in messages)
    if agent_name == "explore":
        assert is_list is True
        assert "list out high-level functions" in prompt
        return [
            {
                "name": "open_drawing",
                "description": "Open the visible drawing",
                "parameters": {},
                "trigger_UIs": [0],
            }
        ]
    if agent_name == "select":
        assert is_list is False
        assert "List of available actions:" in prompt
        return {
            "reasoning": "The requested drawing is visible.",
            "action": {"name": "open_drawing", "parameters": {}},
            "completion_rate": 50,
            "speak": "Opening the drawing.",
        }
    raise AssertionError(f"unexpected MobileGPT agent: {agent_name}")


def _new_action_semantic_query(
    messages: list[dict],
    *,
    model: str | None = None,
    is_list: bool = False,
    agent_name: str = "unknown",
) -> object:
    del model
    prompt = "\n".join(str(message.get("content") or "") for message in messages)
    if agent_name == "explore":
        assert is_list is True
        assert "list out high-level functions" in prompt
        return [
            {
                "name": "open_settings",
                "description": "Open application settings",
                "parameters": {},
                "trigger_UIs": [1],
            }
        ]
    if agent_name == "select":
        assert is_list is False
        assert "List of available actions:" in prompt
        new_action = {
            "name": "open_drawing",
            "description": "Open the visible drawing",
            "parameters": {},
        }
        return {
            "reasoning": "The required drawing action is not listed.",
            "new_action": new_action,
            "action": {"name": "open_drawing", "parameters": {}},
            "completion_rate": 50,
            "speak": "Opening the drawing.",
        }
    raise AssertionError(f"unexpected MobileGPT agent: {agent_name}")


def _input_derive_semantic_query(
    messages: list[dict],
    *,
    model: str | None = None,
    is_list: bool = False,
    agent_name: str = "unknown",
) -> object:
    del model
    prompt = "\n".join(str(message.get("content") or "") for message in messages)
    if agent_name == "explore":
        assert is_list is True
        return [
            {
                "name": "create_new_item",
                "description": "Create a new file or folder",
                "parameters": {
                    "item_name": "Name of the item to create",
                },
                "trigger_UIs": [0],
            }
        ]
    if agent_name == "select":
        assert is_list is False
        return {
            "reasoning": "The item name must be entered.",
            "action": {
                "name": "create_new_item",
                "parameters": {"item_name": "folder_20260725_190339"},
            },
            "completion_rate": 50,
            "speak": "Entering the new folder name.",
        }
    if agent_name == "derive":
        assert is_list is False
        assert "Input text on the screen" in prompt
        return {
            "reasoning": "Replace the default item name.",
            "action": {
                "name": "input",
                "parameters": {
                    "index": 0,
                    "input_text": "folder_20260725_190339",
                },
            },
            "completion_rate": 75,
            "plan": "Confirm folder creation.",
        }
    raise AssertionError(f"unexpected MobileGPT agent: {agent_name}")


def _mismatched_derive_semantic_query(
    messages: list[dict],
    *,
    model: str | None = None,
    is_list: bool = False,
    agent_name: str = "unknown",
) -> object:
    if agent_name != "derive":
        return _semantic_query(
            messages,
            model=model,
            is_list=is_list,
            agent_name=agent_name,
        )
    return {
        "reasoning": "The intended input field is unclear.",
        "action": {
            "name": "ask",
            "parameters": {
                "info_name": "target_field",
                "question": "Which field should receive the value?",
            },
        },
        "completion_rate": 25,
        "plan": "Ask for the missing target field.",
    }


def _write_runlog(
    path: Path,
    actions: list[dict],
    *,
    forests: list[str],
    packages: list[str] | None = None,
) -> Path:
    package_names = packages or ["com.example.app"] * len(actions)
    payload = androidworld_run_log(
        actions,
        observations=[
            androidworld_state(
                f"state-{index}",
                forest=forest,
                package_name=package_names[index],
            )
            for index, forest in enumerate(forests)
        ],
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_conversion_uses_first_open_app_as_task_app(tmp_path: Path) -> None:
    path = _write_runlog(
        tmp_path / "source.json",
        [
            {"action_type": "open_app", "app_name": "com.android.documentsui"},
            {"action_type": "click", "x": 50, "y": 50},
        ],
        forests=[
            "<hierarchy />",
            '<hierarchy><node clickable="true" bounds="[0,0][100,100]" /></hierarchy>',
        ],
        packages=["com.google.android.apps.nexuslauncher", "com.android.chrome"],
    )

    trajectory = _load_runlog_trajectory(path)

    assert trajectory["target_package"] == "com.android.documentsui"
    assert len(trajectory["transitions"]) == 1


def test_conversion_explicit_target_overrides_runlog_inference(tmp_path: Path) -> None:
    path = _write_runlog(
        tmp_path / "source.json",
        [{"action_type": "click", "x": 50, "y": 50}],
        forests=['<hierarchy><node clickable="true" bounds="[0,0][100,100]" /></hierarchy>'],
    )

    trajectory = _load_runlog_trajectory(
        path,
        target_package="com.example.target",
        target_app="Example",
    )

    assert trajectory["target_package"] == "com.example.target"
    assert trajectory["target_app"] == "Example"


def test_conversion_rejects_missing_forest(tmp_path: Path) -> None:
    path = _write_runlog(
        tmp_path / "source.json",
        [{"action_type": "click", "x": 50, "y": 50}],
        forests=[""],
    )

    report = preflight_runlog_conversion(path)

    assert report["ready"] is False
    assert report["failure_code"] == "source_observation_missing"


def test_conversion_rejects_unrepresentable_actions(tmp_path: Path) -> None:
    path = _write_runlog(
        tmp_path / "source.json",
        [{"action_type": "navigate_home"}],
        forests=["<hierarchy />"],
    )

    report = preflight_runlog_conversion(path)

    assert report["ready"] is False
    assert report["failure_code"] == "source_action_unsupported"
    assert report["failure_details"]["action_type"] == "navigate_home"


def test_conversion_writes_runlog_action_and_official_reader_loads_it(
    tmp_path: Path,
) -> None:
    source = _write_runlog(
        tmp_path / "source.json",
        [{"action_type": "click", "x": 50, "y": 50}],
        forests=[
            '<hierarchy><node text="Draw" clickable="true" '
            'bounds="[0,0][100,100]" /><node text="Settings" clickable="true" '
            'bounds="[100,0][200,100]" /></hierarchy>'
        ],
    )
    memory = tmp_path / "memory"
    stats = tmp_path / "stats.jsonl"
    audit = tmp_path / "audit.json"

    result = convert_runlog_to_mobilegpt_memory(
        source_run_log=source,
        mobilegpt_root=MOBILEGPT_ROOT,
        memory_root=memory,
        stats_path=stats,
        audit_path=audit,
        model="unused-offline",
        embedding_provider=lambda _screen: [0.25, 0.75],
        semantic_query_provider=_semantic_query,
        conversion_mode="mobilegpt_semantic",
    )

    with (
        memory / "com.example.app" / "pages" / "0" / "available_subtasks.csv"
    ).open(encoding="utf-8") as handle:
        available_rows = list(csv.DictReader(handle))
    with (
        memory / "com.example.app" / "pages" / "0" / "actions.csv"
    ).open(encoding="utf-8") as handle:
        action_rows = list(csv.DictReader(handle))
    with (memory / "com.example.app" / "tasks.csv").open(
        encoding="utf-8"
    ) as handle:
        task_rows = list(csv.DictReader(handle))
    first_action = json.loads(action_rows[0]["action"])
    assert available_rows[0]["name"] == "open_drawing"
    assert action_rows[0]["subtask_name"] == "open_drawing"
    assert json.loads(task_rows[0]["path"]) == {"0": ["open_drawing", "finish"]}
    assert first_action["name"] == "click"
    assert first_action["parameters"]["text"] == "Draw"
    assert json.loads(action_rows[1]["action"])["name"] == "finish"
    assert result["validated_transition_count"] == 1
    assert result["official_reader_validation"]["loadable"] is True
    audit_payload = json.loads(audit.read_text(encoding="utf-8"))
    assert audit_payload["actions_supplied_to_mobilegpt"] is True
    assert audit_payload["official_reader_validation"]["loadable"] is True


def test_direct_conversion_uses_runlog_actions_without_semantic_agents(
    tmp_path: Path,
) -> None:
    source = _write_runlog(
        tmp_path / "source.json",
        [
            {"action_type": "click", "x": 50, "y": 50},
            {"action_type": "navigate_back"},
        ],
        forests=[
            '<hierarchy><node text="Draw" clickable="true" '
            'bounds="[0,0][100,100]" /></hierarchy>',
            '<hierarchy><node text="Back target" clickable="true" '
            'bounds="[0,0][100,100]" /></hierarchy>',
        ],
    )
    memory = tmp_path / "memory"
    audit = tmp_path / "audit.json"

    def reject_semantic_query(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("direct conversion must not call semantic agents")

    result = convert_runlog_to_mobilegpt_memory(
        source_run_log=source,
        mobilegpt_root=MOBILEGPT_ROOT,
        memory_root=memory,
        stats_path=tmp_path / "stats.jsonl",
        audit_path=audit,
        model="unused-offline",
        embedding_provider=lambda _screen: [0.25, 0.75],
        semantic_query_provider=reject_semantic_query,
    )

    with (memory / "com.example.app" / "tasks.csv").open(
        encoding="utf-8"
    ) as handle:
        task_rows = list(csv.DictReader(handle))
    task_path = json.loads(task_rows[0]["path"])
    assert task_path == {
        "0": [
            "source_step_000_click",
            "source_step_001_navigate_back",
            "finish",
        ],
    }
    payload = json.loads(audit.read_text(encoding="utf-8"))
    assert payload["conversion_mode"] == "runlog_direct"
    assert payload["explore_agent_used"] is False
    assert payload["select_agent_used"] is False
    assert payload["derive_agent_fallback_allowed"] is False
    assert payload["validated_transition_count"] == 2
    assert result["official_reader_validation"]["source_direct_hit_count"] == 2


def test_direct_conversion_grounds_container_click_to_visible_child(
    tmp_path: Path,
) -> None:
    source = _write_runlog(
        tmp_path / "source.json",
        [{"action_type": "click", "x": 50, "y": 50}],
        forests=[
            '<hierarchy><node clickable="true" bounds="[0,0][100,100]">'
            '<node text="task.html" bounds="[0,0][100,50]" />'
            '<node text="4.77 kB" bounds="[0,50][100,100]" />'
            "</node></hierarchy>"
        ],
    )
    memory = tmp_path / "memory"
    audit = tmp_path / "audit.json"

    result = convert_runlog_to_mobilegpt_memory(
        source_run_log=source,
        mobilegpt_root=MOBILEGPT_ROOT,
        memory_root=memory,
        stats_path=tmp_path / "stats.jsonl",
        audit_path=audit,
        model="unused-offline",
        embedding_provider=lambda _screen: [0.25, 0.75],
    )

    payload = json.loads(audit.read_text(encoding="utf-8"))
    row = payload["validation_rows"][0]
    assert row["selected_subtask"]["parameters"] == {
        "target_text": "task.html"
    }
    assert result["official_reader_validation"]["source_direct_hit_count"] == 1


def test_conversion_grounds_coordinate_free_input_to_focused_field(
    tmp_path: Path,
) -> None:
    source = _write_runlog(
        tmp_path / "source.json",
        [{"action_type": "input_text", "text": "5558642097"}],
        forests=[
            '<hierarchy><node text="First name" editable="true" '
            'bounds="[0,0][100,100]"/><node text="Phone" editable="true" '
            'focused="true" bounds="[0,100][100,200]"/></hierarchy>'
        ],
    )
    memory = tmp_path / "memory"

    convert_runlog_to_mobilegpt_memory(
        source_run_log=source,
        mobilegpt_root=MOBILEGPT_ROOT,
        memory_root=memory,
        stats_path=tmp_path / "stats.jsonl",
        audit_path=tmp_path / "audit.json",
        model="unused-offline",
        embedding_provider=lambda _screen: [0.25, 0.75],
        semantic_query_provider=_semantic_query,
        conversion_mode="mobilegpt_semantic",
    )

    with (
        memory / "com.example.app" / "pages" / "0" / "actions.csv"
    ).open(encoding="utf-8") as handle:
        action_rows = list(csv.DictReader(handle))
    first_action = json.loads(action_rows[0]["action"])
    assert first_action["name"] == "input"
    assert first_action["parameters"]["input_text"] == "5558642097"
    assert first_action["parameters"]["text"] == "Phone"


def test_conversion_uses_native_derive_for_ambiguous_coordinate_free_input(
    tmp_path: Path,
) -> None:
    source = _write_runlog(
        tmp_path / "source.json",
        [
            {
                "action_type": "input_text",
                "text": "folder_20260725_190339",
            }
        ],
        forests=[
            '<hierarchy><node class="android.widget.EditText" text="my_note" '
            'bounds="[0,0][100,100]"/><node class="android.widget.EditText" '
            'text=".md" bounds="[100,0][200,100]"/></hierarchy>'
        ],
    )
    memory = tmp_path / "memory"

    convert_runlog_to_mobilegpt_memory(
        source_run_log=source,
        mobilegpt_root=MOBILEGPT_ROOT,
        memory_root=memory,
        stats_path=tmp_path / "stats.jsonl",
        audit_path=tmp_path / "audit.json",
        model="unused-offline",
        embedding_provider=lambda _screen: [0.25, 0.75],
        semantic_query_provider=_input_derive_semantic_query,
        conversion_mode="mobilegpt_semantic",
    )

    with (
        memory / "com.example.app" / "pages" / "0" / "actions.csv"
    ).open(encoding="utf-8") as handle:
        action_rows = list(csv.DictReader(handle))
    first_action = json.loads(action_rows[0]["action"])
    assert first_action["name"] == "input"
    assert first_action["parameters"]["input_text"] == "<item_name__-1>"
    assert first_action["parameters"]["text"] == "my_note"


def test_conversion_preserves_repeated_selected_subtask_episodes(
    tmp_path: Path,
) -> None:
    forest = (
        '<hierarchy><node text="Draw" clickable="true" '
        'bounds="[0,0][100,100]" /><node text="Settings" clickable="true" '
        'bounds="[100,0][200,100]" /></hierarchy>'
    )
    source = _write_runlog(
        tmp_path / "source.json",
        [
            {"action_type": "click", "x": 50, "y": 50},
            {"action_type": "click", "x": 50, "y": 50},
        ],
        forests=[forest, forest],
    )
    memory = tmp_path / "memory"

    result = convert_runlog_to_mobilegpt_memory(
        source_run_log=source,
        mobilegpt_root=MOBILEGPT_ROOT,
        memory_root=memory,
        stats_path=tmp_path / "stats.jsonl",
        audit_path=tmp_path / "audit.json",
        model="unused-offline",
        embedding_provider=lambda _screen: [0.25, 0.75],
        semantic_query_provider=_semantic_query,
        conversion_mode="mobilegpt_semantic",
    )

    with (memory / "com.example.app" / "tasks.csv").open(
        encoding="utf-8"
    ) as handle:
        task_rows = list(csv.DictReader(handle))
    with (
        memory / "com.example.app" / "pages" / "0" / "actions.csv"
    ).open(encoding="utf-8") as handle:
        action_rows = list(csv.DictReader(handle))
    assert json.loads(task_rows[0]["path"]) == {
        "0": ["open_drawing", "open_drawing", "finish"]
    }
    assert [int(row["step"]) for row in action_rows] == [0, 1, 0, 1]
    assert result["official_reader_validation"]["source_direct_hit_count"] == 2


def test_conversion_persists_select_new_action_in_semantic_closure(
    tmp_path: Path,
) -> None:
    source = _write_runlog(
        tmp_path / "source.json",
        [{"action_type": "click", "x": 50, "y": 50}],
        forests=[
            '<hierarchy><node text="Draw" clickable="true" '
            'bounds="[0,0][100,100]" /><node text="Settings" clickable="true" '
            'bounds="[100,0][200,100]" /></hierarchy>'
        ],
    )
    memory = tmp_path / "memory"

    convert_runlog_to_mobilegpt_memory(
        source_run_log=source,
        mobilegpt_root=MOBILEGPT_ROOT,
        memory_root=memory,
        stats_path=tmp_path / "stats.jsonl",
        audit_path=tmp_path / "audit.json",
        model="unused-offline",
        embedding_provider=lambda _screen: [0.25, 0.75],
        semantic_query_provider=_new_action_semantic_query,
        conversion_mode="mobilegpt_semantic",
    )

    with (
        memory / "com.example.app" / "pages" / "0" / "available_subtasks.csv"
    ).open(encoding="utf-8") as handle:
        available_names = [row["name"] for row in csv.DictReader(handle)]
    with (memory / "com.example.app" / "tasks.csv").open(
        encoding="utf-8"
    ) as handle:
        task_rows = list(csv.DictReader(handle))
    assert available_names == ["open_settings", "open_drawing"]
    assert json.loads(task_rows[0]["path"]) == {"0": ["open_drawing", "finish"]}


def test_conversion_rejects_input_when_native_derive_cannot_match_source(
    tmp_path: Path,
) -> None:
    source = _write_runlog(
        tmp_path / "source.json",
        [{"action_type": "input_text", "text": "5558642097"}],
        forests=[
            '<hierarchy><node text="First name" editable="true" '
            'bounds="[0,0][100,100]"/><node text="Phone" editable="true" '
            'bounds="[0,100][100,200]"/></hierarchy>'
        ],
    )

    with pytest.raises(
        MobileGPTConversionError,
        match="mobilegpt_derive_action_mismatch",
    ):
        convert_runlog_to_mobilegpt_memory(
            source_run_log=source,
            mobilegpt_root=MOBILEGPT_ROOT,
            memory_root=tmp_path / "memory",
            stats_path=tmp_path / "stats.jsonl",
            audit_path=tmp_path / "audit.json",
            model="unused-offline",
            embedding_provider=lambda _screen: [0.25, 0.75],
            semantic_query_provider=_mismatched_derive_semantic_query,
            conversion_mode="mobilegpt_semantic",
        )


def test_memory_validation_rejects_action_without_parameters(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    page = memory / "com.example.app" / "pages" / "0"
    screen = page / "screen"
    screen.mkdir(parents=True)
    (memory / "tasks.csv").write_text(
        "name,description,parameters,app\n"
        "openExample,Open Example,{},com.example.app\n",
        encoding="utf-8",
    )
    (memory / "com.example.app" / "tasks.csv").write_text(
        'name,path\nopenExample,"{""0"": [""tapExample""]}"\n',
        encoding="utf-8",
    )
    (memory / "com.example.app" / "pages.csv").write_text(
        'index,available_subtasks,trigger_uis,extra_uis,screen\n'
        '0,"[]","{}","[]",screen-0\n',
        encoding="utf-8",
    )
    (memory / "com.example.app" / "hierarchy.csv").write_text(
        "index,screen,embedding\n0,screen-0,[0.0]\n",
        encoding="utf-8",
    )
    for name in ("available_subtasks.csv", "subtasks.csv"):
        (page / name).write_text(
            "name,description,parameters\n"
            "tapExample,Tap Example,{}\n",
            encoding="utf-8",
        )
    (page / "actions.csv").write_text(
        "subtask_name,step,action,example\n"
        'tapExample,0,"{""name"": ""click""}",{}\n',
        encoding="utf-8",
    )
    for name in ("raw.xml", "html.xml", "hierarchy.xml", "parsed.xml", "pretty.xml"):
        (screen / name).write_text("<hierarchy />\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mobilegpt_memory_action_parameters_invalid"):
        validate_mobilegpt_memory(memory)


def test_memory_validation_rejects_malformed_screen_xml(tmp_path: Path) -> None:
    source = _write_runlog(
        tmp_path / "source.json",
        [{"action_type": "click", "x": 50, "y": 50}],
        forests=[
            '<hierarchy><node text="Draw" clickable="true" '
            'bounds="[0,0][100,100]" /><node text="Settings" clickable="true" '
            'bounds="[100,0][200,100]" /></hierarchy>'
        ],
    )
    memory = tmp_path / "memory"
    convert_runlog_to_mobilegpt_memory(
        source_run_log=source,
        mobilegpt_root=MOBILEGPT_ROOT,
        memory_root=memory,
        stats_path=tmp_path / "stats.jsonl",
        audit_path=tmp_path / "audit.json",
        model="unused-offline",
        embedding_provider=lambda _screen: [0.25, 0.75],
        semantic_query_provider=_semantic_query,
        conversion_mode="mobilegpt_semantic",
    )
    screen_xml = memory / "com.example.app" / "pages" / "0" / "screen" / "raw.xml"
    screen_xml.write_text("<hierarchy>", encoding="utf-8")

    with pytest.raises(ValueError, match="mobilegpt_memory_screen_xml_invalid"):
        validate_mobilegpt_memory(memory)


def test_failure_audit_preserves_partial_validation_rows(tmp_path: Path) -> None:
    forest = (
        '<hierarchy><node index="13" text="Draw" clickable="true" '
        'bounds="[0,0][100,100]" /></hierarchy>'
    )
    source = _write_runlog(
        tmp_path / "source.json",
        [{"action_type": "click", "x": 50, "y": 50}],
        forests=[forest],
    )
    stats = tmp_path / "stats.jsonl"
    stats.write_text(
        json.dumps(
            {
                "event": "mobilegpt_conversion_action_mapped",
                "source_step_index": 0,
                "matched": False,
                "reason": "source_action_target_unresolved",
                "consumed_transitions": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    audit = tmp_path / "audit.json"

    result = write_conversion_failure_audit(
        source_run_log=source,
        stats_path=stats,
        audit_path=audit,
        error=MobileGPTConversionError(
            "source_action_target_unresolved",
            step_index=0,
        ),
        wall_sec=1.25,
    )

    assert result["complete"] is False
    assert result["failure_code"] == "source_action_target_unresolved"
    assert result["validated_transition_count"] == 0
    assert result["validation_rows"][0]["reason"] == "source_action_target_unresolved"
    assert json.loads(audit.read_text(encoding="utf-8")) == result
