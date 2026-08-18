from __future__ import annotations

import json
from pathlib import Path

from omniflow.core.model import Action, ActionResult, Observation
from src.experiment.protocol import DROIDRUN_COMMIT, DROIDRUN_VERSION
from src.integrations.skilldroid_replay import (
    compile_droidrun_macro,
    load_official_droidrun_macro_player,
    run_droidrun_macro_replay,
)


class _Host:
    emulator_serial = "emulator-5600"

    def __init__(self, *, width: int = 200, height: int = 400) -> None:
        self.width = width
        self.height = height
        self.actions: list[Action] = []

    def observe(self, **_: object) -> Observation:
        return Observation(
            package_name="com.example",
            activity_name=".MainActivity",
            extra={"display": {"width": self.width, "height": self.height}},
        )

    def act(self, action: Action) -> ActionResult:
        self.actions.append(action)
        return ActionResult(True)


class _MacroPlayer:
    def __init__(
        self,
        *,
        device_serial: str | None,
        delay_between_actions: float,
    ) -> None:
        self.device_serial = device_serial
        self.delay_between_actions = delay_between_actions
        self.driver = None

    @staticmethod
    def load_macro_from_file(path: str) -> dict[str, object]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    async def replay_macro(self, macro: dict[str, object]) -> bool:
        assert self.driver is not None
        for action in macro["actions"]:
            if action["action_type"] == "tap":
                await self.driver.tap(action["x"], action["y"])
            elif action["action_type"] == "swipe":
                await self.driver.swipe(
                    action["start_x"],
                    action["start_y"],
                    action["end_x"],
                    action["end_y"],
                    action["duration_ms"],
                )
            elif action["action_type"] == "button_press":
                await self.driver.press_button(action["button"])
            elif action["action_type"] == "start_app":
                await self.driver.start_app(action["package"], action["activity"])
            else:
                raise AssertionError(action)
        return True


def _state_catalog(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.transfer-state-catalog.v1",
                "run_id": "source-run",
                "states": {
                    "before": {
                        "state_id": "before",
                        "xml": "<hierarchy />",
                        "package_name": "com.example",
                        "activity_name": ".MainActivity",
                        "display": {"width": 100, "height": 200},
                    },
                    "after": {
                        "state_id": "after",
                        "xml": "<hierarchy />",
                        "package_name": "com.example",
                        "activity_name": ".MainActivity",
                        "display": {"width": 100, "height": 200},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _run_log(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.canonical_run_log.v1",
                "run_id": "source-run",
                "goal": "Save the item",
                "status": "succeeded",
                "success": True,
                "finished_at_ms": 0,
                "steps": [
                    {
                        "step_index": 0,
                        "before_state_id": "before",
                        "action": {"tool": "click", "args": {"x": 999, "y": 999}},
                        "result": {"success": True},
                        "after_state_id": "before",
                    },
                    {
                        "step_index": 1,
                        "before_state_id": "before",
                        "action": {"tool": "click", "args": {"x": 200, "y": 300}},
                        "result": {"success": True},
                        "after_state_id": "after",
                    },
                ],
                "diagnostics": {"official_success": True},
                "final_state_id": "after",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_compile_droidrun_macro_uses_official_format(tmp_path: Path) -> None:
    macro_path = tmp_path / "trajectory" / "macro.json"

    report = compile_droidrun_macro(
        source_run_log=_run_log(tmp_path / "runlog.json"),
        source_state_catalog=_state_catalog(tmp_path / "transfer_states.json"),
        output_path=macro_path,
    )

    macro = json.loads(macro_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        macro_path.with_name("droidrun_manifest.json").read_text(encoding="utf-8")
    )
    assert macro == {
        "version": "1.0",
        "description": "Save the item",
        "timestamp": "19700101_000000",
        "total_actions": 1,
        "actions": [{"action_type": "tap", "x": 20, "y": 60}],
    }
    assert report["source_step_indices"] == [1]
    assert manifest["droidrun_version"] == DROIDRUN_VERSION
    assert manifest["droidrun_commit"] == DROIDRUN_COMMIT


def test_compile_droidrun_macro_accepts_qualified_env100_runlog(
    tmp_path: Path,
) -> None:
    run_log = tmp_path / "target.run_log.json"
    run_log.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.run_log.v1",
                "run_id": "qualified-env100",
                "goal": "Input 1+1",
                "status": "succeeded",
                "success": True,
                "steps": [
                    {
                        "step_index": 0,
                        "action": {
                            "action_type": "open_app",
                            "app_name": "com.google.android.calculator",
                        },
                        "observation": {
                            "auxiliaries": {
                                "display": {"width": 1080, "height": 1920}
                            }
                        },
                        "next_observation": {
                            "auxiliaries": {
                                "display": {"width": 1080, "height": 1920}
                            }
                        },
                        "result": {"success": True},
                    },
                    {
                        "step_index": 1,
                        "action": {"action_type": "click", "x": 159, "y": 1483},
                        "observation": {
                            "auxiliaries": {
                                "display": {"width": 1080, "height": 1920}
                            }
                        },
                        "next_observation": {
                            "auxiliaries": {
                                "display": {"width": 1080, "height": 1920}
                            }
                        },
                        "result": {"success": True},
                    },
                ],
                "validator": {"official": True, "success": True},
            }
        ),
        encoding="utf-8",
    )
    macro_path = tmp_path / "macro.json"

    compile_droidrun_macro(
        source_run_log=run_log,
        source_state_catalog=tmp_path / "missing-transfer-states.json",
        output_path=macro_path,
    )

    macro = json.loads(macro_path.read_text(encoding="utf-8"))
    assert macro["actions"] == [
        {
            "action_type": "start_app",
            "package": "com.google.android.calculator",
            "activity": None,
        },
        {"action_type": "tap", "x": 159, "y": 1483},
    ]


def test_pinned_official_droidrun_macro_player_loads() -> None:
    player = load_official_droidrun_macro_player()

    assert player.__name__ == "MacroPlayer"
    assert player.__module__ == "droidrun.macro.replay"


def test_droidrun_macro_replay_preserves_absolute_source_pixels(
    tmp_path: Path,
) -> None:
    macro_path = tmp_path / "macro.json"
    macro_path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "description": "Save the item",
                "timestamp": "19700101_000000",
                "total_actions": 1,
                "actions": [{"action_type": "tap", "x": 20, "y": 60}],
            }
        ),
        encoding="utf-8",
    )
    host = _Host(width=200, height=400)

    result = run_droidrun_macro_replay(
        memory_path=macro_path,
        host=host,
        macro_player_factory=_MacroPlayer,
    )

    assert result.success is True
    assert result.actions_executed == 1
    assert result.model_calls == 0
    assert result.fallback_steps == 0
    assert host.actions == [Action("click", {"x": 100.0, "y": 150.0})]
    assert result.detail["trace"][0]["action_type"] == "tap"
