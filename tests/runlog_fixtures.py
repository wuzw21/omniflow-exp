from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from omniflow.core.model import Function
from omniflow.core.trajectory import OMNIFLOW_RUN_LOG_SCHEMA_VERSION
from omniflow.functions.assets import STORE_VERSION


def write_function_store(
    path: str | Path,
    functions: Sequence[Function],
) -> Path:
    store_path = Path(path)
    store_path.write_text(
        json.dumps(
            {
                "schema_version": STORE_VERSION,
                "functions": {
                    function.id: function.to_dict() for function in functions
                },
                "source_calls": [],
            }
        ),
        encoding="utf-8",
    )
    return store_path


def androidworld_state(
    state_identifier: str,
    *,
    forest: Any = "<hierarchy />",
    package_name: str = "com.example.app",
    activity_name: str = ".MainActivity",
    width: int = 1000,
    height: int = 1000,
    with_pixels: bool = False,
    ui_elements: Sequence[Any] = (),
) -> dict[str, Any]:
    return {
        "pixels": (
            {
                "path": f"/tmp/{state_identifier}.png",
                "sha256": "0" * 64,
                "width": width,
                "height": height,
                "mime_type": "image/png",
            }
            if with_pixels
            else None
        ),
        "forest": forest,
        "ui_elements": list(ui_elements),
        "auxiliaries": {
            "state_id": state_identifier,
            "package_name": package_name,
            "activity_name": activity_name,
            "display": {"width": width, "height": height},
        },
    }


def androidworld_run_log(
    actions: Sequence[dict[str, Any]],
    *,
    observations: Sequence[dict[str, Any]] | None = None,
    task_name: str = "Task",
    goal: str = "Complete the task.",
    run_id: str = "source-run",
    seed: int | None = 111,
    success: bool = True,
    with_pixels: bool = False,
) -> dict[str, Any]:
    states = list(observations or ())
    if not states:
        states = [
            androidworld_state(
                f"state-{index}",
                with_pixels=with_pixels,
            )
            for index in range(len(actions))
        ]
    if len(states) != len(actions):
        raise ValueError("fixture_observation_count_mismatch")
    return {
        "schema_version": OMNIFLOW_RUN_LOG_SCHEMA_VERSION,
        "run_id": run_id,
        "task_name": task_name,
        "goal": goal,
        "task_parameters": {},
        "seed": seed,
        "status": "succeeded" if success else "failed",
        "success": success,
        "validator": {
            "official": True,
            "success": success,
            "reward": 1.0 if success else 0.0,
        },
        "provenance": {"kind": "runtime"},
        "steps": [
            {
                "step_index": index,
                "observation": states[index],
                "action": dict(action),
                "result": {"success": True},
            }
            for index, action in enumerate(actions)
        ],
    }


def mobilegpt_partial_grounding_run_log(
    *,
    task_name: str = "Task",
) -> dict[str, Any]:
    return androidworld_run_log(
        [
            {"action_type": "click", "x": 50, "y": 50},
            {"action_type": "click", "x": 50, "y": 50},
        ],
        observations=[
            androidworld_state(
                "groundable",
                forest=(
                    '<hierarchy><node text="Continue" clickable="true" '
                    'bounds="[0,0][100,100]" /></hierarchy>'
                ),
            ),
            androidworld_state("ungroundable", forest="<hierarchy />"),
        ],
        task_name=task_name,
    )


def mobilegpt_native_fallback_run_log(
    *,
    task_name: str = "Task",
) -> dict[str, Any]:
    return androidworld_run_log(
        [
            {"action_type": "open_app", "app_name": "com.example.app"},
            {"action_type": "answer", "text": "No tasks are due."},
        ],
        task_name=task_name,
    )
