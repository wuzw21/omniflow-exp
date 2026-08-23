from __future__ import annotations

import csv
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import sys

import pytest
from runlog_fixtures import androidworld_run_log, androidworld_state

from src.integrations import mobilegpt as mobilegpt_module
from src.integrations.mobilegpt import (
    CONVERSION_MODE_OFFICIAL,
    MobileGPTConversionError,
    _load_runlog_trajectory,
    _mobilegpt_action_from_runlog,
    _parameter_values,
    _target_element,
    convert_runlog_to_mobilegpt_memory,
    preflight_runlog_conversion,
    validate_mobilegpt_memory,
    write_conversion_failure_audit,
)
from src.integrations.mobilegpt_format import encode_xml

MOBILEGPT_ROOT = Path(
    os.environ.get(
        "MOBILEGPT_TEST_ROOT",
        "/Users/wuzewen/Projects/Omni/OmniFlow/runtime/external/mobilegpt-official",
    )
)


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


def test_preflight_accepts_compact_bmoca_runlog_with_state_catalog(
    tmp_path: Path,
) -> None:
    source = tmp_path / "runlog.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.canonical_run_log.v1",
                "run_id": "bmoca-source",
                "goal": "Save the item",
                "status": "succeeded",
                "success": True,
                "steps": [
                    {
                        "step_index": 0,
                        "before_state_id": "before",
                        "action": {
                            "tool": "click",
                            "args": {"x": 999, "y": 999},
                        },
                        "result": {"success": True},
                        "after_state_id": "before",
                    },
                    {
                        "step_index": 1,
                        "before_state_id": "before",
                        "action": {
                            "tool": "click",
                            "args": {"x": 500, "y": 250},
                        },
                        "result": {"success": True},
                        "after_state_id": "after",
                    }
                ],
                "diagnostics": {
                    "official_success": True,
                    "task_id": "example/save_item",
                },
                "final_state_id": "after",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "transfer_states.json").write_text(
        json.dumps(
            {
                "schema_version": "omniflow.transfer-state-catalog.v1",
                "run_id": "bmoca-source",
                "states": {
                    "before": {
                        "state_id": "before",
                        "xml": (
                            '<hierarchy bounds="[0,0][200,400]">'
                            '<node package="com.example" clickable="true" '
                            'bounds="[80,80][120,120]" /></hierarchy>'
                        ),
                        "package_name": "com.example",
                        "activity_name": ".MainActivity",
                        "display": {"width": 200, "height": 400},
                    },
                    "after": {
                        "state_id": "after",
                        "xml": '<hierarchy bounds="[0,0][200,400]" />',
                        "package_name": "com.example",
                        "activity_name": ".MainActivity",
                        "display": {"width": 200, "height": 400},
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    report = preflight_runlog_conversion(source)
    trajectory = _load_runlog_trajectory(source)

    assert report["ready"] is True
    assert report["transition_count"] == 1
    assert trajectory["task_name"] == "example/save_item"
    assert trajectory["transitions"][0].action == {
        "action_type": "click",
        "x": 100,
        "y": 100,
    }


def test_open_app_only_conversion_uses_final_observation_as_finish_page(
    tmp_path: Path,
) -> None:
    source = _write_runlog(
        tmp_path / "source.json",
        [{"action_type": "open_app", "app_name": "com.android.contacts"}],
        forests=["<hierarchy />"],
        packages=["com.google.android.apps.nexuslauncher"],
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["final_observation"] = androidworld_state(
        "contacts-ready",
        forest=(
            '<hierarchy><node text="Contacts" '
            'bounds="[0,0][100,100]" /></hierarchy>'
        ),
        package_name="com.android.contacts",
    )
    source.write_text(json.dumps(payload), encoding="utf-8")
    memory = tmp_path / "memory"

    preflight = preflight_runlog_conversion(source)
    result = convert_runlog_to_mobilegpt_memory(
        source_run_log=source,
        mobilegpt_root=MOBILEGPT_ROOT,
        memory_root=memory,
        stats_path=tmp_path / "stats.jsonl",
        audit_path=tmp_path / "audit.json",
        model="unused-offline",
        embedding_provider=lambda _screen: [0.25, 0.75],
    )

    with (memory / "com.android.contacts" / "tasks.csv").open(
        encoding="utf-8"
    ) as handle:
        task_rows = list(csv.DictReader(handle))
    with (
        memory / "com.android.contacts" / "pages" / "0" / "actions.csv"
    ).open(encoding="utf-8") as handle:
        action_rows = list(csv.DictReader(handle))

    assert preflight["ready"] is True
    assert preflight["transition_count"] == 0
    assert json.loads(task_rows[0]["path"]) == {"0": ["finish"]}
    assert action_rows == []
    assert result["transition_count"] == 0
    assert result["validated_transition_count"] == 0
    assert result["memory_validation"]["launch_only"] is True
    assert result["official_reader_validation"]["launch_finish_validated"] is True


def test_conversion_scopes_official_embedding_model_to_offline_memory(
    tmp_path: Path,
) -> None:
    source = _write_runlog(
        tmp_path / "source.json",
        [{"action_type": "click", "x": 50, "y": 50}],
        forests=[
            '<hierarchy><node text="Draw" clickable="true" '
            'bounds="[0,0][100,100]" /></hierarchy>'
        ],
    )
    observed: list[str | None] = []

    def embedding_provider(_screen: str) -> list[float]:
        observed.append(os.environ.get("MOBILEGPT_EMBEDDING_MODEL"))
        return [0.25, 0.75]

    convert_runlog_to_mobilegpt_memory(
        source_run_log=source,
        mobilegpt_root=MOBILEGPT_ROOT,
        memory_root=tmp_path / "memory",
        stats_path=tmp_path / "stats.jsonl",
        audit_path=tmp_path / "audit.json",
        model="GLM-5.1",
        embedding_model="GLM-Embedding-3",
        embedding_provider=embedding_provider,
    )

    assert observed == ["GLM-Embedding-3"]
    assert os.environ.get("MOBILEGPT_EMBEDDING_MODEL") != "GLM-Embedding-3"


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


def test_scroll_source_is_passed_to_official_authoring_boundary(tmp_path: Path) -> None:
    source = _write_runlog(
        tmp_path / "source.json",
        [{"action_type": "scroll", "direction": "down"}],
        forests=[
            '<hierarchy><node class="androidx.recyclerview.widget.RecyclerView" '
            'scrollable="true" bounds="[0,0][100,100]" /></hierarchy>'
        ],
    )

    report = preflight_runlog_conversion(source)
    assert report["ready"] is True
    assert report["action_type_counts"] == {"scroll": 1}


def test_direct_scroll_memory_matches_official_executor_schema(tmp_path: Path) -> None:
    source = _write_runlog(
        tmp_path / "source.json",
        [{"action_type": "scroll", "direction": "down"}],
        forests=[
            '<hierarchy><node class="androidx.recyclerview.widget.RecyclerView" '
            'scrollable="true" bounds="[0,0][100,100]">'
            '<node class="android.widget.TextView" text="Item 1" '
            'bounds="[0,0][100,50]" />'
            '<node class="android.widget.TextView" text="Item 2" '
            'bounds="[0,50][100,100]" /></node></hierarchy>'
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
    )

    action_path = memory / "com.example.app" / "pages" / "0" / "actions.csv"
    with action_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    action = json.loads(rows[0]["action"])

    assert action["name"] == "scroll"
    assert action["parameters"]["index"] == "0"
    assert action["parameters"]["direction"] == "down"
    assert action["parameters"]["attrib"]["self"]["tag"] == "scroll"


def test_preflight_exposes_authoritative_mobilegpt_teacher_actions(
    tmp_path: Path,
) -> None:
    source = _write_runlog(
        tmp_path / "source.json",
        [{"action_type": "click", "x": 50, "y": 50}],
        forests=[
            '<hierarchy><node text="Stopwatch" clickable="true" '
            'bounds="[0,0][100,100]" /><node text="More options" '
            'clickable="true" bounds="[100,0][200,100]" /></hierarchy>'
        ],
    )

    report = preflight_runlog_conversion(
        source,
        mobilegpt_root=MOBILEGPT_ROOT,
    )

    assert report["ready"] is True
    assert report["teacher_action_count"] == 1
    assert report["teacher_actions"] == [
        {
            "source_step_index": 0,
            "source_action_type": "click",
            "required_action": {
                "name": "click",
                "parameters": {"index": "1"},
            },
            "target_label": "Stopwatch",
        }
    ]


def test_teacher_alignment_allows_official_scroll_container_index() -> None:
    rows = mobilegpt_module._align_official_actions_to_teacher(
        [
            {
                "source_step_index": 0,
                "source_action_type": "scroll",
                "required_action": {
                    "name": "scroll",
                    "parameters": {"direction": "down"},
                },
                "target_label": "",
            }
        ],
        [
            {
                "name": "scroll",
                "parameters": {"index": 3, "direction": "down"},
            }
        ],
    )

    assert rows[0]["matched"] is True


def test_teacher_alignment_rejects_a_different_click_target() -> None:
    rows = mobilegpt_module._align_official_actions_to_teacher(
        [
            {
                "source_step_index": 0,
                "source_action_type": "click",
                "required_action": {
                    "name": "click",
                    "parameters": {"index": "1"},
                },
                "target_label": "Stopwatch",
            }
        ],
        [{"name": "click", "parameters": {"index": "2"}}],
    )

    assert rows[0]["matched"] is False


def test_official_conversion_forwards_teacher_query_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_runlog(
        tmp_path / "source.json",
        [{"action_type": "click", "x": 50, "y": 50}],
        forests=[
            '<hierarchy><node text="Stopwatch" clickable="true" '
            'bounds="[0,0][100,100]" /></hierarchy>'
        ],
    )
    captured: dict[str, object] = {}

    def teacher_query(*_args: object, **_kwargs: object) -> object:
        return {}

    def official_authoring(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"memory_root": str(kwargs["memory_root"])}

    monkeypatch.setattr(
        mobilegpt_module,
        "_run_official_mobilegpt_authoring",
        official_authoring,
    )

    convert_runlog_to_mobilegpt_memory(
        source_run_log=source,
        mobilegpt_root=MOBILEGPT_ROOT,
        memory_root=tmp_path / "memory",
        stats_path=tmp_path / "stats.jsonl",
        audit_path=tmp_path / "audit.json",
        model="GLM-4.6V",
        semantic_query_provider=teacher_query,
        conversion_mode=CONVERSION_MODE_OFFICIAL,
    )

    assert captured["semantic_query_provider"] is teacher_query


def test_official_authoring_prompt_preserves_runlog_action_end_to_end(
    tmp_path: Path,
) -> None:
    source = _write_runlog(
        tmp_path / "source.json",
        [
            {"action_type": "click", "x": 50, "y": 50},
            {"action_type": "click", "x": 150, "y": 50},
        ],
        forests=[
            '<hierarchy><node text="Stopwatch" clickable="true" '
            'bounds="[0,0][100,100]" /><node text="More options" '
            'clickable="true" bounds="[100,0][200,100]" /></hierarchy>',
            '<hierarchy><node text="Stopwatch" clickable="true" '
            'bounds="[0,0][100,100]" /><node text="Start" '
            'clickable="true" bounds="[100,0][200,100]" /></hierarchy>',
        ],
    )
    observed_teacher_prompts: list[dict[str, object]] = []
    derive_attempts: list[int] = []

    def teacher_payload(messages: object) -> dict[str, object] | None:
        text = "\n".join(
            str(message.get("content") or "")
            for message in messages
            if isinstance(message, dict)
        )
        match = re.search(
            r"MOBILEGPT_AUTHORITATIVE_RUNLOG_STEP=(\{.*\})\n"
            r"</authoritative_runlog_teacher>",
            text,
        )
        return json.loads(match.group(1)) if match else None

    def query(
        messages: object,
        *,
        agent_name: str = "",
        is_list: bool = False,
        **_kwargs: object,
    ) -> object:
        teacher = teacher_payload(messages)
        if agent_name in {"explore", "select", "derive"}:
            assert teacher is not None
            observed_teacher_prompts.append(teacher)
        if agent_name == "task":
            return {
                "found_match": False,
                "api": {
                    "name": "Task",
                    "description": "Complete the task.",
                    "parameters": {},
                    "app": "com.example.app",
                },
            }
        if agent_name == "explore":
            if teacher and teacher.get("terminal") is False:
                required = teacher["required_action"]
                index = int(required["parameters"]["index"])
                return [
                    {
                        "name": "follow_demonstrated_step",
                        "description": "Follow the demonstrated successful step.",
                        "parameters": {},
                        "trigger_UIs": [index],
                    }
                ]
            return [] if is_list else {}
        if agent_name == "select":
            if teacher and teacher.get("terminal") is True:
                return {
                    "action": {"name": "finish", "parameters": {}},
                    "completion_rate": 1,
                    "speak": "",
                }
            return {
                "action": {
                    "name": "follow_demonstrated_step",
                    "parameters": {},
                },
                "completion_rate": 0,
                "speak": "",
            }
        if agent_name == "derive":
            assert teacher is not None
            derive_attempts.append(len(derive_attempts) + 1)
            if len(derive_attempts) == 1:
                return None
            return {
                "reasoning": "Preserve the successful RunLog action.",
                "action": teacher["required_action"],
                "completion_rate": 1,
                "plan": "Verify the next screen.",
            }
        if agent_name == "action_summarize":
            return "Followed the demonstrated successful step."
        raise AssertionError(f"unexpected MobileGPT query: {agent_name}")

    audit = tmp_path / "audit.json"
    result = convert_runlog_to_mobilegpt_memory(
        source_run_log=source,
        mobilegpt_root=MOBILEGPT_ROOT,
        memory_root=tmp_path / "memory",
        stats_path=tmp_path / "stats.jsonl",
        audit_path=audit,
        model="GLM-4.6V",
        embedding_provider=lambda _screen: [0.25, 0.75],
        semantic_query_provider=query,
        conversion_mode=CONVERSION_MODE_OFFICIAL,
    )

    payload = json.loads(audit.read_text(encoding="utf-8"))
    assert observed_teacher_prompts
    # The first action needs one empty-response retry, then the second action
    # and official terminal cycle each use one derive call.
    assert derive_attempts == [1, 2, 3, 4]
    assert payload["teacher_prompt_used"] is True
    assert payload["teacher_action_alignment_complete"] is True
    assert payload["validation_rows"][0]["expected_action"] == {
        "name": "click",
        "parameters": {"index": "1"},
    }
    assert payload["validation_rows"][0]["actual_action"] == payload[
        "validation_rows"
    ][0]["expected_action"]
    assert payload["validation_rows"][1]["expected_action"] == {
        "name": "click",
        "parameters": {"index": "2"},
    }
    assert payload["validation_rows"][1]["actual_action"] == payload[
        "validation_rows"
    ][1]["expected_action"]
    assert result["official_reader_validation"][
        "teacher_aligned_action_count"
    ] == 2


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
    assert available_rows[0]["name"] == "source_step_000_click"
    assert action_rows[0]["subtask_name"] == "source_step_000_click"
    assert json.loads(task_rows[0]["path"]) == {
        "0": ["source_step_000_click", "finish"]
    }
    assert first_action["name"] == "click"
    assert first_action["parameters"]["index"] == "0"
    assert first_action["parameters"]["text"] == "<target_text__-1>"
    assert json.loads(action_rows[1]["action"])["name"] == "finish"
    assert result["validated_transition_count"] == 1
    assert result["official_reader_validation"]["loadable"] is True
    audit_payload = json.loads(audit.read_text(encoding="utf-8"))
    assert audit_payload["actions_supplied_to_mobilegpt"] is True
    assert audit_payload["official_reader_validation"]["loadable"] is True


def test_direct_conversion_runs_official_memory_in_bundle_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_runlog(
        tmp_path / "source.json",
        [{"action_type": "click", "x": 50, "y": 50}],
        forests=[
            '<hierarchy><node text="Draw" clickable="true" '
            'bounds="[0,0][100,100]" /></hierarchy>'
        ],
    )
    memory = tmp_path / "bundle" / "memory"
    observed: list[Path] = []
    original = mobilegpt_module._working_directory

    @contextmanager
    def observe_working_directory(path: Path):
        observed.append(path)
        with original(path):
            yield

    monkeypatch.setattr(
        mobilegpt_module,
        "_working_directory",
        observe_working_directory,
    )

    convert_runlog_to_mobilegpt_memory(
        source_run_log=source,
        mobilegpt_root=MOBILEGPT_ROOT,
        memory_root=memory,
        stats_path=tmp_path / "stats.jsonl",
        audit_path=tmp_path / "audit.json",
        model="unused-offline",
        embedding_provider=lambda _screen: [0.25, 0.75],
    )

    assert observed == [memory.parent]


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
    assert payload["derive_agent_fallback_allowed"] is True
    assert payload["validated_transition_count"] == 2
    assert result["official_reader_validation"]["source_direct_hit_count"] == 2


def test_runlog_index_click_is_grounded_from_ui_element_bounds(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            androidworld_run_log(
                [{"action_type": "click", "index": 1}],
                observations=[
                    androidworld_state(
                        "indexed",
                        forest=(
                            '<hierarchy><node text="Get started" '
                            'clickable="true" bounds="[20,30][80,90]" />'
                            "</hierarchy>"
                        ),
                        ui_elements=[
                            {},
                            {
                                "bbox_pixels": {
                                    "x_min": 20,
                                    "y_min": 30,
                                    "x_max": 80,
                                    "y_max": 90,
                                }
                            },
                        ],
                    )
                ],
            )
        ),
        encoding="utf-8",
    )

    trajectory = _load_runlog_trajectory(source)

    assert trajectory["transitions"][0].action["x"] == 50
    assert trajectory["transitions"][0].action["y"] == 60


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


def test_conversion_preserves_native_example_when_action_cannot_adapt(
    tmp_path: Path,
) -> None:
    source = _write_runlog(
        tmp_path / "source.json",
        [{"action_type": "click", "x": 180, "y": 180}],
        forests=[
            '<hierarchy><node bounds="[0,0][200,200]">'
            '<node clickable="true" bounds="[0,0][50,50]" />'
            '<node text="Home" bounds="[50,0][100,50]" />'
            '<node clickable="true" bounds="[150,150][200,200]" />'
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

    action_path = memory / "com.example.app" / "pages" / "0" / "actions.csv"
    with action_path.open(encoding="utf-8") as handle:
        action_rows = list(csv.DictReader(handle))
    example = json.loads(action_rows[0]["example"])
    response = json.loads(example["response"])
    payload = json.loads(audit.read_text(encoding="utf-8"))

    assert response["action"] == {
        "name": "click",
        "parameters": {"index": "3"},
    }
    assert payload["derive_agent_fallback_allowed"] is True
    assert payload["source_example_fallback_count"] == 1
    assert payload["source_reader_coverage_validation"] is True
    assert payload["validation_rows"][0]["reader_resolution"] == (
        "native_example_fallback"
    )
    assert result["official_reader_validation"]["source_reader_coverage_count"] == 1


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
    )

    with (
        memory / "com.example.app" / "pages" / "0" / "actions.csv"
    ).open(encoding="utf-8") as handle:
        action_rows = list(csv.DictReader(handle))
    first_action = json.loads(action_rows[0]["action"])
    assert first_action["name"] == "input"
    assert first_action["parameters"]["input_text"] == "<input_text__-1>"


def test_direct_conversion_writes_official_trigger_and_extra_ui_sets(
    tmp_path: Path,
) -> None:
    source = _write_runlog(
        tmp_path / "source.json",
        [{"action_type": "click", "x": 50, "y": 50}],
        forests=[
            '<hierarchy><node text="Primary" clickable="true" '
            'bounds="[0,0][100,100]" /><node text="Secondary" clickable="true" '
            'bounds="[100,0][200,100]" /></hierarchy>'
        ],
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
    )

    with (memory / "com.example.app" / "pages.csv").open(
        encoding="utf-8"
    ) as handle:
        page = next(csv.DictReader(handle))
    trigger_uis = json.loads(page["trigger_uis"])
    extra_uis = json.loads(page["extra_uis"])
    assert trigger_uis["source_step_000_click"][0]["self"]["tag"] == "button"
    assert extra_uis[0]["self"]["tag"] == "button"
    assert result["official_reader_validation"]["loadable"] is True


def test_direct_conversion_is_recalled_by_official_page_matcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_runlog(
        tmp_path / "source.json",
        [{"action_type": "click", "x": 50, "y": 50}],
        forests=[
            '<hierarchy><node text="Primary" clickable="true" '
            'bounds="[0,0][100,100]" /><node text="Secondary" clickable="true" '
            'bounds="[100,0][200,100]" /></hierarchy>'
        ],
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
    )

    screen_root = memory / "com.example.app" / "pages" / "0" / "screen"
    monkeypatch.setenv("MOBILEGPT_MEMORY_ROOT", str(memory))
    from memory import memory_manager as official_memory_manager

    monkeypatch.setattr(
        official_memory_manager,
        "get_openai_embedding",
        lambda _screen, **_kwargs: [0.25, 0.75],
    )
    official_memory = official_memory_manager.Memory(
        "com.example.app",
        "Complete the task.",
        result["task"]["name"],
    )
    page_index, _ = official_memory.search_node(
        (screen_root / "parsed.xml").read_text(encoding="utf-8"),
        (screen_root / "hierarchy.xml").read_text(encoding="utf-8"),
        (screen_root / "html.xml").read_text(encoding="utf-8"),
    )

    assert page_index == 0


def test_conversion_grounds_input_from_verified_text_change(
    tmp_path: Path,
) -> None:
    source = _write_runlog(
        tmp_path / "source.json",
        [
            {"action_type": "input_text", "text": "copy_warm_tree"},
            {"action_type": "click", "x": 180, "y": 180},
        ],
        forests=[
            '<hierarchy><node id="12" class="android.widget.EditText" '
            'text="my_note" clickable="true" bounds="[0,0][100,100]"/>'
            '<node id="13" class="android.widget.EditText" text=".md" '
            'clickable="true" bounds="[100,0][200,100]"/></hierarchy>',
            '<hierarchy><node id="12" class="android.widget.EditText" '
            'text="copy_warm_tree" clickable="true" bounds="[0,0][100,100]"/>'
            '<node id="13" class="android.widget.EditText" text=".md" '
            'clickable="true" bounds="[100,0][200,100]"/>'
            '<node text="OK" clickable="true" bounds="[150,150][200,200]"/>'
            '</hierarchy>',
        ],
    )
    trajectory = _load_runlog_trajectory(source)
    transition = trajectory["transitions"][0]
    server_root = MOBILEGPT_ROOT / "Server"
    if str(server_root) not in sys.path:
        sys.path.insert(0, str(server_root))
    parsed_xml, _, _ = encode_xml(transition.forest, mobilegpt_root=MOBILEGPT_ROOT)
    target = _target_element(
        transition.action,
        parsed_xml,
        step_index=transition.step_index,
        source_forest=transition.forest,
        next_forest=transition.next_forest,
    )

    assert target.tag == "input"
    assert target.get("index") == "0"
    assert target.text == "my_note"


def test_conversion_preserves_empty_input_for_verified_text_change(
    tmp_path: Path,
) -> None:
    source = _write_runlog(
        tmp_path / "source.json",
        [
            {"action_type": "input_text", "text": "Note body"},
            {"action_type": "click", "x": 50, "y": 50},
        ],
        forests=[
            '<hierarchy><node id="19" class="android.widget.EditText" text="" '
            'clickable="true" bounds="[0,0][100,100]"/></hierarchy>',
            '<hierarchy><node id="19" class="android.widget.EditText" '
            'text="Note body" clickable="true" bounds="[0,0][100,100]"/>'
            '<node text="Save" clickable="true" bounds="[0,0][100,100]"/>'
            '</hierarchy>',
        ],
    )
    trajectory = _load_runlog_trajectory(source)
    transition = trajectory["transitions"][0]
    server_root = MOBILEGPT_ROOT / "Server"
    if str(server_root) not in sys.path:
        sys.path.insert(0, str(server_root))
    parsed_xml, _, _ = encode_xml(transition.forest, mobilegpt_root=MOBILEGPT_ROOT)
    target = _target_element(
        transition.action,
        parsed_xml,
        step_index=transition.step_index,
        source_forest=transition.forest,
        next_forest=transition.next_forest,
    )

    assert target.tag == "input"
    assert target.get("index") == "0"
    assert target.text is None


def test_anonymous_verified_input_avoids_unrelated_children_generalization(
    tmp_path: Path,
) -> None:
    source = _write_runlog(
        tmp_path / "source.json",
        [
            {"action_type": "input_text", "text": "Note body"},
            {"action_type": "click", "x": 50, "y": 50},
        ],
        forests=[
            '<hierarchy><node id="19" class="android.widget.EditText" text="" '
            'clickable="true" bounds="[0,0][100,100]"/></hierarchy>',
            '<hierarchy><node id="19" class="android.widget.EditText" '
            'text="Note body" clickable="true" bounds="[0,0][100,100]"/>'
            '<node text="Save" clickable="true" bounds="[0,0][100,100]"/>'
            '</hierarchy>',
        ],
    )
    trajectory = _load_runlog_trajectory(source)
    transition = trajectory["transitions"][0]
    server_root = MOBILEGPT_ROOT / "Server"
    if str(server_root) not in sys.path:
        sys.path.insert(0, str(server_root))
    parsed_xml, _, _ = encode_xml(transition.forest, mobilegpt_root=MOBILEGPT_ROOT)
    converted, _, _ = _mobilegpt_action_from_runlog(
        transition,
        parsed_xml,
        task_parameters={"text": "Note body"},
        selected_subtask={
            "name": "enter_note",
            "parameters": {"text": "Note body"},
        },
        generalize_action=lambda *_args: pytest.fail(
            "anonymous verified input must not generalize unrelated children"
        ),
    )

    assert converted == {
        "name": "input",
        "parameters": {
            "index": "0",
            "input_text": "<text__-1>",
            "attrib": {
                "self": {"tag": "input"},
                "parent": {},
                "children": [],
            },
        },
    }


def test_action_generalization_avoids_nested_native_placeholders(
    tmp_path: Path,
) -> None:
    source = _write_runlog(
        tmp_path / "source.json",
        [{"action_type": "input_text", "text": "A useful description"}],
        forests=[
            '<hierarchy><node class="android.widget.EditText" '
            'text="Description" focused="true" editable="true" '
            'bounds="[0,0][100,100]"/></hierarchy>'
        ],
    )
    trajectory = _load_runlog_trajectory(source)
    transition = trajectory["transitions"][0]
    server_root = MOBILEGPT_ROOT / "Server"
    if str(server_root) not in sys.path:
        sys.path.insert(0, str(server_root))
    parsed_xml, _, _ = encode_xml(transition.forest, mobilegpt_root=MOBILEGPT_ROOT)
    calls: list[dict[str, str]] = []

    def generalize(action: dict, subtask: dict, _screen: str) -> dict:
        calls.append(dict(subtask["parameters"]))
        assert len(subtask["parameters"]) <= 1
        parameters = subtask["parameters"]
        if "input_text" in parameters:
            action["parameters"]["input_text"] = "<input_text__-1>"
        if "target_text" in parameters:
            action["parameters"]["text"] = "<target_text__-1>"
        else:
            action["parameters"]["text"] = "Description"
        return action

    converted, _, _ = _mobilegpt_action_from_runlog(
        transition,
        parsed_xml,
        task_parameters={},
        selected_subtask={
            "name": "enter_description",
            "parameters": {
                "target_text": "Description",
                "input_text": "A useful description",
            },
        },
        generalize_action=generalize,
    )

    assert calls == []
    assert converted["parameters"]["input_text"] == "<input_text__-1>"


def test_action_generalization_rejects_invalid_native_placeholder(
    tmp_path: Path,
) -> None:
    source = _write_runlog(
        tmp_path / "source.json",
        [{"action_type": "click", "x": 50, "y": 50}],
        forests=[
            '<hierarchy><node text="Save" clickable="true" '
            'bounds="[0,0][100,100]"/></hierarchy>'
        ],
    )
    transition = _load_runlog_trajectory(source)["transitions"][0]
    server_root = MOBILEGPT_ROOT / "Server"
    if str(server_root) not in sys.path:
        sys.path.insert(0, str(server_root))
    parsed_xml, _, _ = encode_xml(transition.forest, mobilegpt_root=MOBILEGPT_ROOT)

    with pytest.raises(
        MobileGPTConversionError,
        match="mobilegpt_action_placeholder_invalid",
    ):
        _mobilegpt_action_from_runlog(
            transition,
            parsed_xml,
            task_parameters={},
            selected_subtask={
                "name": "save",
                "parameters": {"target_text": "Save"},
            },
            generalize_action=lambda action, *_args: {
                **action,
                "parameters": {
                    **action["parameters"],
                    "text": "<t<input_text__0>rget_text__-1>",
                },
            },
        )


def test_parameter_names_follow_native_placeholder_grammar() -> None:
    parameters = _parameter_values(
        {
            "recipe__name": "Soup",
            "servings count": 4,
        }
    )

    assert set(parameters.values()) == {"Soup", "4"}
    assert any(name.startswith("recipe_name_") for name in parameters)
    assert any(name.startswith("servings_count_") for name in parameters)
    assert all("__" not in name for name in parameters)


def test_conversion_rejects_ambiguous_coordinate_free_input(
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

    with pytest.raises(
        MobileGPTConversionError,
        match="source_action_target_unresolved",
    ):
        convert_runlog_to_mobilegpt_memory(
            source_run_log=source,
            mobilegpt_root=MOBILEGPT_ROOT,
            memory_root=memory,
            stats_path=tmp_path / "stats.jsonl",
            audit_path=tmp_path / "audit.json",
            model="unused-offline",
            embedding_provider=lambda _screen: [0.25, 0.75],
        )


def test_conversion_preserves_repeated_runlog_actions(
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
        "0": ["source_step_000_click", "source_step_001_click", "finish"]
    }
    assert [int(row["step"]) for row in action_rows] == [0, 1, 0, 1]
    assert result["official_reader_validation"]["source_direct_hit_count"] == 2


def test_conversion_rejects_removed_semantic_mode(
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
    with pytest.raises(ValueError, match="mobilegpt_conversion_mode_invalid"):
        convert_runlog_to_mobilegpt_memory(
            source_run_log=source,
            mobilegpt_root=MOBILEGPT_ROOT,
            memory_root=tmp_path / "memory",
            stats_path=tmp_path / "stats.jsonl",
            audit_path=tmp_path / "audit.json",
            model="unused-offline",
            embedding_provider=lambda _screen: [0.25, 0.75],
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


def test_memory_validation_accepts_scroll_only_official_page(
    tmp_path: Path,
) -> None:
    memory = tmp_path / "memory"
    app = memory / "com.example.app"
    (app / "pages" / "0" / "screen").mkdir(parents=True)
    (app / "pages" / "1" / "screen").mkdir(parents=True)
    (memory / "tasks.csv").write_text(
        "name,description,parameters,app\n"
        "takePhoto,Take a photo,{},com.example.app\n",
        encoding="utf-8",
    )
    with (app / "tasks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("name", "path"))
        writer.writerow(
            (
                "takePhoto",
                json.dumps(
                    {"0": ["capture"], "1": ["scroll_screen"]},
                    separators=(",", ":"),
                ),
            )
        )
    (app / "pages.csv").write_text(
        "index,available_subtasks,trigger_uis,extra_uis,screen\n"
        '0,"[]","{}","[]",screen-0\n'
        '1,"[]","{}","[]",screen-1\n',
        encoding="utf-8",
    )
    (app / "hierarchy.csv").write_text(
        "index,screen,embedding\n0,screen-0,[0.0]\n1,screen-1,[0.0]\n",
        encoding="utf-8",
    )
    (app / "pages" / "0" / "subtasks.csv").write_text(
        "name,description,parameters\ncapture,Capture photo,{}\n",
        encoding="utf-8",
    )
    (app / "pages" / "0" / "available_subtasks.csv").write_text(
        "name,description,parameters\ncapture,Capture photo,{}\n",
        encoding="utf-8",
    )
    (app / "pages" / "1" / "subtasks.csv").write_text(
        "name,description,parameters\n",
        encoding="utf-8",
    )
    (app / "pages" / "1" / "available_subtasks.csv").write_text(
        "name,description,parameters\n",
        encoding="utf-8",
    )
    for page_index, rows in {
        "0": [("capture", 0, {"name": "click", "parameters": {"index": 5}}, {})],
        "1": [],
    }.items():
        with (app / "pages" / page_index / "actions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(("subtask_name", "step", "action", "example"))
            for subtask_name, step, action, example in rows:
                writer.writerow((subtask_name, step, json.dumps(action), example))
    for page_index in ("0", "1"):
        for name in ("raw.xml", "html.xml", "hierarchy.xml", "parsed.xml", "pretty.xml"):
            (app / "pages" / page_index / "screen" / name).write_text(
                "<hierarchy />\n",
                encoding="utf-8",
            )

    report = validate_mobilegpt_memory(memory)

    assert report["native_memory_complete"] is True
    assert report["subtask_count"] == 1


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
