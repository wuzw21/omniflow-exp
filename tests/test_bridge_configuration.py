from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load_gui_max_steps():
    bridge_path = ROOT / "omniflow" / "bridge.py"
    tree = ast.parse(bridge_path.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id in {"_DEFAULT_GUI_MAX_STEPS", "_MAX_GUI_MAX_STEPS"}
                for target in node.targets
            )
        )
        or isinstance(node, ast.FunctionDef)
        and node.name == "_gui_max_steps"
    ]
    namespace: dict[str, Any] = {"Any": Any}
    exec(compile(ast.Module(body=selected, type_ignores=[]), bridge_path, "exec"), namespace)
    return namespace["_gui_max_steps"]


def test_gui_max_steps_default_and_bounds() -> None:
    gui_max_steps = _load_gui_max_steps()
    assert gui_max_steps(None) == 20
    assert gui_max_steps(32) == 32
    assert gui_max_steps(0) == 20
    assert gui_max_steps(-1) == 1
    assert gui_max_steps(100) == 64


def test_bridge_schema_exposes_adjustable_gui_step_budget() -> None:
    schema_path = (
        ROOT / "schemas"
        / "oob"
        / "omniflow_android_bridge.v2.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    max_steps = schema["operations"]["tools/call"]["request"][
        "run_gui_arguments"
    ]["max_steps"]

    assert max_steps == {
        "type": "integer",
        "default": 20,
        "minimum": 1,
        "maximum": 64,
    }
