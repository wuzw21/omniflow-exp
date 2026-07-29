from __future__ import annotations

import json
import subprocess
import sys
from types import ModuleType

from src.integrations import mobilegpt_teacher
from src.integrations.mobilegpt_teacher import install_mobilegpt_teacher


def test_exhausted_teacher_finishes_task_before_subtask_reentry(
    tmp_path,
    monkeypatch,
) -> None:
    source_run_log = tmp_path / "source.run_log.json"
    source_run_log.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "action": {
                            "type": "click",
                            "params": {"x": 1, "y": 1},
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    class MobileGPT:
        def init(self, instruction, task, is_new_task):
            return None

        def get_next_action(
            self,
            parsed_xml=None,
            hierarchy_xml=None,
            encoded_xml=None,
        ):
            calls.append("subtask_reentry")
            return {"name": "click", "parameters": {"index": "1"}}

        def _MobileGPT__finish_task(self):
            calls.append("task_finished")
            return None

    class DeriveAgent:
        def derive(self, screen, examples=None):
            raise AssertionError("derive must not run after teacher exhaustion")

    agents_module = ModuleType("agents")
    derive_module = ModuleType("agents.derive_agent")
    derive_module.DeriveAgent = DeriveAgent
    mobilegpt_module = ModuleType("mobilegpt")
    mobilegpt_module.MobileGPT = MobileGPT
    parsing_utils = ModuleType("utils.parsing_utils")
    utils_module = ModuleType("utils")
    utils_module.parsing_utils = parsing_utils
    monkeypatch.setitem(sys.modules, "agents", agents_module)
    monkeypatch.setitem(sys.modules, "agents.derive_agent", derive_module)
    monkeypatch.setitem(sys.modules, "mobilegpt", mobilegpt_module)
    monkeypatch.setitem(sys.modules, "utils", utils_module)
    monkeypatch.setitem(sys.modules, "utils.parsing_utils", parsing_utils)

    teacher = install_mobilegpt_teacher(source_run_log=source_run_log)
    teacher.mark_exhausted()

    assert MobileGPT().get_next_action() is None
    assert calls == ["task_finished"]


def test_teacher_uses_foreground_package_when_parsed_xml_omits_it(
    tmp_path,
    monkeypatch,
) -> None:
    source_run_log = tmp_path / "source.run_log.json"
    source_run_log.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "observation": {
                            "package_name": "com.android.chrome",
                            "xml": (
                                '<hierarchy><node text="Submit" '
                                'resource-id="com.android.chrome:id/submit" '
                                'clickable="true" bounds="[0,0][100,100]" />'
                                "</hierarchy>"
                            ),
                        },
                        "action": {
                            "type": "click",
                            "params": {"x": 50, "y": 50},
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        mobilegpt_teacher,
        "_adb_foreground_package",
        lambda: "com.android.chrome",
    )
    teacher = mobilegpt_teacher.MobileGPTTeacher(source_run_log)
    teacher.reset(task={"app": "com.google.android.documentsui"})

    result = teacher.next_action(
        '<button text="Submit" id="submit" clickable="true" index="7" />'
    )

    assert result.action == {"name": "click", "parameters": {"index": "7"}}
    assert result.consumed_source_action is True


def test_adb_foreground_package_reads_top_resumed_activity(monkeypatch) -> None:
    monkeypatch.setenv("OMNIFLOW_MOBILEGPT_ADB_PATH", "/sdk/adb")
    monkeypatch.setenv("ANDROID_SERIAL", "emulator-5560")

    def fake_run(argv, **kwargs):
        assert argv == [
            "/sdk/adb",
            "-s",
            "emulator-5560",
            "shell",
            "dumpsys",
            "activity",
            "activities",
        ]
        assert kwargs["timeout"] == 10
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=(
                "topResumedActivity=ActivityRecord{abc u0 "
                "com.android.chrome/.Main} t7}\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(mobilegpt_teacher.subprocess, "run", fake_run)

    assert mobilegpt_teacher._adb_foreground_package() == "com.android.chrome"


def test_teacher_handles_chrome_search_provider_prompt_without_consuming_source() -> None:
    result = mobilegpt_teacher._target_preflight_action(
        '<div><p text="Search with Sogou" index="9" />'
        '<button text="Keep Google" index="12" /></div>'
    )

    assert result is not None
    assert result.action == {"name": "click", "parameters": {"index": "12"}}
    assert result.consumed_source_action is False
