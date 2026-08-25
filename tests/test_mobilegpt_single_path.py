from __future__ import annotations

import json
import os
from pathlib import Path
import re

from omniflow.core.trajectory import OMNIFLOW_RUN_LOG_SCHEMA_VERSION

from src.experiment.mobilegpt_contract import MOBILEGPT_MEMORY_SCHEMA, MOBILEGPT_SOURCE_METHOD
from src.integrations.mobilegpt import convert_runlog_to_mobilegpt_bundle


MOBILEGPT_ROOT = Path(
    os.environ.get(
        "MOBILEGPT_TEST_ROOT",
        "/Users/wuzewen/Projects/Omni/OmniFlow/runtime/external/mobilegpt-official",
    )
)


def _runlog(path: Path) -> Path:
    forests = [
        '<hierarchy><node text="Camera" clickable="true" bounds="[0,0][100,100]" />'
        '<node text="Shutter" clickable="true" bounds="[100,0][200,100]" /></hierarchy>',
        '<hierarchy><node text="Camera" clickable="true" bounds="[0,0][100,100]" />'
        '<node text="Done" clickable="true" bounds="[100,0][200,100]" /></hierarchy>',
    ]
    payload = {
        "schema_version": OMNIFLOW_RUN_LOG_SCHEMA_VERSION,
        "run_id": "mobilegpt-single-path",
        "task_name": "CameraTakePhoto",
        "goal": "Take a photo.",
        "task_parameters": {},
        "seed": 111,
        "status": "succeeded",
        "success": True,
        "validator": {"official": True, "success": True, "reward": 1.0},
        "provenance": {"kind": "runtime"},
        "steps": [
            {
                "step_index": index,
                "observation": {
                    "pixels": None,
                    "forest": forest,
                    "ui_elements": [],
                    "auxiliaries": {
                        "state_id": f"state-{index}",
                        "package_name": "com.example.app",
                        "activity_name": ".MainActivity",
                        "display": {"width": 1000, "height": 1000},
                    },
                },
                "action": {"action_type": "click", "x": 50 if index == 0 else 150, "y": 50},
                "result": {"success": True},
            }
            for index, forest in enumerate(forests)
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_runlog_uses_only_official_mobilegpt_authoring(tmp_path: Path) -> None:
    source = _runlog(tmp_path / "source.run_log.json")

    def teacher(messages: object) -> dict[str, object] | None:
        text = "\n".join(
            str(item.get("content") or "")
            for item in messages
            if isinstance(item, dict)
        )
        match = re.search(
            r"MOBILEGPT_AUTHORITATIVE_RUNLOG_STEP=(\{.*\})\n</authoritative_runlog_teacher>",
            text,
        )
        return json.loads(match.group(1)) if match else None

    derive_calls = 0

    def query(messages: object, *, agent_name: str = "", is_list: bool = False, **_: object) -> object:
        nonlocal derive_calls
        step = teacher(messages)
        if agent_name == "task":
            return {"found_match": False, "api": {"name": "Task", "description": "Take a photo.", "parameters": {}, "app": "com.example.app"}}
        if agent_name == "explore":
            return ([{"name": "follow_demonstrated_step", "description": "Follow the step.", "parameters": {}, "trigger_UIs": [0]}] if step and not step.get("terminal") else ([] if is_list else {}))
        if agent_name == "select":
            return {"action": {"name": "finish" if step and step.get("terminal") else "follow_demonstrated_step", "parameters": {}}, "completion_rate": 1 if step and step.get("terminal") else 0, "speak": ""}
        if agent_name == "derive":
            derive_calls += 1
            if derive_calls == 1:
                return None
            return {"reasoning": "Preserve the RunLog action.", "action": {"name": "finish", "parameters": {}} if step and step.get("terminal") else step["required_action"], "completion_rate": 1, "plan": "Verify the result."}
        if agent_name == "action_summarize":
            return "Followed the demonstrated successful step."
        raise AssertionError(agent_name)

    result = convert_runlog_to_mobilegpt_bundle(
        source_run_log=source,
        mobilegpt_root=MOBILEGPT_ROOT,
        output_root=tmp_path / "bundle",
        model="GLM-4.6V",
        embedding_model="GLM-Embedding-2",
        embedding_provider=lambda _text: [0.25, 0.75],
        semantic_query_provider=query,
    )
    manifest = json.loads((tmp_path / "bundle" / "mobilegpt_memory_manifest.json").read_text())
    audit = json.loads((tmp_path / "bundle" / "trajectory_audit.json").read_text())
    assert result["method"] == "mobilegpt"
    assert manifest["schema_version"] == MOBILEGPT_MEMORY_SCHEMA
    assert manifest["source_method"] == MOBILEGPT_SOURCE_METHOD
    assert manifest["provenance"]["original_mobilegpt_prompts"] is True
    assert manifest["provenance"]["actions_supplied_to_mobilegpt"] is False
    assert audit["conversion_mode"] == "official_mobilegpt_learning"
    assert audit["teacher_action_alignment_complete"] is True
    assert audit["complete"] is True
