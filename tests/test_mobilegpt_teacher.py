from __future__ import annotations

import json
from types import ModuleType
import sys

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
