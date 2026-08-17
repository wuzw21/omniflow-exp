from __future__ import annotations

import json
from pathlib import Path

from omniflow.functions.assets import FunctionStore
from src.experiment.bmoca_replay import build_bmoca_function_registry


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_build_bmoca_function_registry_uses_ai_step_roles_and_exact_actions(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    trace_root = corpus / "traces" / "trace-chrome"
    runlog = {
        "schema_version": "omniflow.canonical_run_log.v1",
        "run_id": "chrome-env100",
        "goal": "Open a new tab in Chrome",
        "status": "succeeded",
        "success": True,
        "steps": [
            {
                "step_index": 0,
                "before_state_id": "welcome",
                "action": {"tool": "click", "args": {"x": 500, "y": 800}},
                "result": {"success": True},
                "after_state_id": "start",
                "metadata": {},
            },
            {
                "step_index": 1,
                "before_state_id": "start",
                "action": {"tool": "click", "args": {"x": 950, "y": 100}},
                "result": {"success": True},
                "after_state_id": "new-tab",
                "metadata": {},
            },
        ],
    }
    states = {
        "schema_version": "omniflow.transfer-state-catalog.v1",
        "run_id": "chrome-env100",
        "states": {
            "welcome": {
                "state_id": "welcome",
                "xml": '<hierarchy><node text="Welcome to Chrome" /></hierarchy>',
            },
            "start": {
                "state_id": "start",
                "xml": '<hierarchy><node content-desc="New tab" /></hierarchy>',
            },
        },
    }
    _write_json(trace_root / "runlog.json", runlog)
    _write_json(trace_root / "transfer_states.json", states)
    _write_json(
        corpus / "manifest.json",
        {
            "schema_version": "omniflow.offline-trace-corpus.v1",
            "traces": [
                {
                    "trace_id": "trace-chrome",
                    "task_id": "chrome/open_new_tab",
                    "environment_id": "100",
                    "success_evidence": {"official_success": True},
                    "runlog": {"path": "traces/trace-chrome/runlog.json"},
                    "state_catalog": {
                        "path": "traces/trace-chrome/transfer_states.json"
                    },
                }
            ],
        },
    )
    prompts: list[str] = []

    def author(prompt: str) -> dict:
        prompts.append(prompt)
        return {
            "functions": [
                {
                    "function_id": "open_new_tab",
                    "name": "Open a new Chrome tab",
                    "description": "Open one new tab from Chrome.",
                    "steps": [
                        {
                            "step": 0,
                            "role": "checker",
                            "reason": "Dismiss optional first-run onboarding.",
                        },
                        {
                            "step": 1,
                            "role": "function",
                            "reason": "Open the requested tab.",
                        },
                    ],
                }
            ]
        }

    report = build_bmoca_function_registry(
        corpus,
        tmp_path / "registry",
        author=author,
        model="GLM-5.1",
    )

    assert report["summary"] == {
        "source_environment": "100",
        "task_count": 1,
        "function_count": 1,
        "checker_step_count": 1,
        "function_step_count": 1,
        "model_calls": 1,
    }
    assert len(prompts) == 1
    assert "one decision object per Step" in prompts[0]
    store = FunctionStore(tmp_path / "registry" / "store.json")
    function = next(iter(store.functions.values()))
    assert [step.role for step in function.steps] == ["checker", "function"]
    assert function.steps[0].action.args == {"x": 500, "y": 800}
    assert function.steps[1].source_state_id == "start"
    assert function.checker_rules == ()
    catalog = json.loads((tmp_path / "registry" / "transfer_states.json").read_text())
    assert set(catalog["states"]) == {"welcome", "start"}
