"""DroidRun v0.5.6 native macro preparation and B-MoCA replay."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import datetime as dt
import importlib
from importlib import metadata
import json
from pathlib import Path
import sys
from typing import Any, ClassVar

from omniflow.core.model import Action, ActionResult, Observation, RunResult
from omniflow.runlog import _androidworld_action_to_omniflow
from omniflow.transfer.runtime import load_transfer_state_catalog
from src.experiment.protocol import (
    DROIDRUN_COMMIT,
    DROIDRUN_PORTAL_COMMIT,
    DROIDRUN_PORTAL_VERSION,
    DROIDRUN_VERSION,
)

DROIDRUN_MACRO_VERSION = "1.0"
DROIDRUN_MANIFEST_SCHEMA = "omniflow.droidrun-macro-manifest.v1"
_COORDINATE_MAXIMUM = 1000.0


def compile_droidrun_macro(
    *,
    source_run_log: str | Path,
    source_state_catalog: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Convert one official-successful env100 RunLog to DroidRun macro.json."""

    run_log_path = Path(source_run_log).expanduser().resolve()
    run_log = _read_object(run_log_path)
    if (
        run_log.get("status") != "succeeded"
        or run_log.get("success") is not True
        or not isinstance(run_log.get("steps"), list)
        or not run_log["steps"]
    ):
        raise ValueError("droidrun_successful_source_run_log_required")
    diagnostics = run_log.get("diagnostics")
    validator = run_log.get("validator")
    official_success = (
        isinstance(diagnostics, dict)
        and diagnostics.get("official_success") is True
    ) or (
        isinstance(validator, dict)
        and validator.get("official") is True
        and validator.get("success") is True
    )
    if not official_success:
        raise ValueError("droidrun_official_source_success_required")

    states = _source_states(run_log, source_state_catalog)
    actions: list[dict[str, Any]] = []
    source_indices: list[int] = []
    for expected_index, raw_step in enumerate(run_log["steps"]):
        if not isinstance(raw_step, dict) or raw_step.get("step_index") != expected_index:
            raise ValueError("droidrun_source_step_invalid")
        result = raw_step.get("result")
        if not isinstance(result, dict) or result.get("success") is not True:
            continue
        before_state_id = str(raw_step.get("before_state_id") or "").strip()
        after_state_id = str(raw_step.get("after_state_id") or "").strip()
        if before_state_id and before_state_id == after_state_id:
            continue
        source_state = (
            states.get(before_state_id)
            if before_state_id
            else _embedded_source_state(raw_step)
        )
        if not isinstance(source_state, dict):
            raise TypeError(
                f"droidrun_source_state_missing:{expected_index}:{before_state_id}"
            )
        actions.append(_to_droidrun_action(_source_action(raw_step), source_state=source_state))
        source_indices.append(expected_index)
    if not actions:
        raise ValueError("droidrun_source_actions_required")

    destination = Path(output_path).expanduser().resolve()
    if destination.name != "macro.json":
        raise ValueError("droidrun_macro_filename_required")
    if destination.exists():
        raise FileExistsError(f"droidrun_macro_already_exists:{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    macro = {
        "version": DROIDRUN_MACRO_VERSION,
        "description": str(run_log.get("goal") or "DroidRun replay"),
        "timestamp": _macro_timestamp(run_log),
        "total_actions": len(actions),
        "actions": actions,
    }
    destination.write_text(
        json.dumps(macro, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_path = destination.with_name("droidrun_manifest.json")
    manifest = {
        "schema_version": DROIDRUN_MANIFEST_SCHEMA,
        "droidrun_version": DROIDRUN_VERSION,
        "droidrun_commit": DROIDRUN_COMMIT,
        "portal_version": DROIDRUN_PORTAL_VERSION,
        "portal_commit": DROIDRUN_PORTAL_COMMIT,
        "source_run_log": str(run_log_path),
        "source_run_id": str(run_log.get("run_id") or ""),
        "source_step_indices": source_indices,
        "macro_path": str(destination),
        "total_actions": len(actions),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        **manifest,
        "manifest_path": str(manifest_path),
        "memory_path": str(destination),
    }


def run_droidrun_macro_replay(
    *,
    memory_path: str | Path,
    host: Any,
    macro_player_factory: Callable[..., Any] | None = None,
) -> RunResult:
    """Replay one oracle-selected DroidRun macro through the B-MoCA driver seam."""

    macro_path = _resolve_macro_path(memory_path)
    player_class = macro_player_factory or load_official_droidrun_macro_player()
    player = player_class(
        device_serial=str(getattr(host, "emulator_serial", "") or "") or None,
        delay_between_actions=1.0,
    )
    macro = player.load_macro_from_file(str(macro_path))
    _validate_macro(macro)
    driver = _BMocaDroidRunDriver(host)
    player.driver = driver
    replay_success = asyncio.run(player.replay_macro(macro))
    final_state = Observation.from_value(
        host.observe(xml=True, screenshot=False, app_info=True)
    )
    error = None if replay_success else driver.error or "droidrun_macro_replay_failed"
    return RunResult(
        bool(replay_success),
        function_id="skilldroid_replay",
        actions_executed=driver.actions_executed,
        model_calls=0,
        fallback_steps=0,
        error=error,
        final_state=final_state,
        detail={
            "trace": list(driver.trace),
            "memory_hit": True,
            "droidrun_version": DROIDRUN_VERSION,
            "droidrun_commit": DROIDRUN_COMMIT,
            "macro_path": str(macro_path),
        },
    )


class _BMocaDroidRunDriver:
    platform = "Android"
    supported: ClassVar[set[str]] = {
        "tap",
        "swipe",
        "press_button",
        "start_app",
        "drag",
    }
    supported_buttons: ClassVar[set[str]] = {"back", "home", "enter"}

    def __init__(self, host: Any) -> None:
        self.host = host
        self.actions_executed = 0
        self.trace: list[dict[str, Any]] = []
        self.error: str | None = None

    async def connect(self) -> None:
        return None

    async def ensure_connected(self) -> None:
        return None

    async def tap(self, x: int, y: int) -> None:
        await self._act_pixels("tap", Action("click", self._point_args(x, y)))

    async def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: float = 1000,
    ) -> None:
        first = self._point_args(x1, y1)
        second = self._point_args(x2, y2)
        await self._act_pixels(
            "swipe",
            Action(
                "swipe",
                {
                    "x1": first["x"],
                    "y1": first["y"],
                    "x2": second["x"],
                    "y2": second["y"],
                    "duration_ms": float(duration_ms),
                },
            ),
        )

    async def press_button(self, button: str) -> None:
        normalized = str(button).strip().lower()
        if normalized not in self.supported_buttons:
            raise ValueError(f"droidrun_button_unsupported:{normalized}")
        await self._act_pixels(
            "button_press",
            Action("press_key", {"key": normalized}),
        )

    async def start_app(self, package: str, activity: str | None = None) -> str:
        del activity
        await self._act_pixels(
            "start_app",
            Action("open_app", {"package_name": str(package)}),
        )
        return f"App started: {package}"

    async def drag(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration: float = 3.0,
    ) -> None:
        await self.swipe(x1, y1, x2, y2, duration_ms=float(duration) * 1000.0)

    async def input_text(self, text: str, clear: bool = False) -> bool:
        del text, clear
        raise ValueError("droidrun_bmoca_input_text_unsupported")

    async def _act_pixels(self, action_type: str, action: Action) -> None:
        result = ActionResult.from_value(self.host.act(action))
        event = {
            "action_index": len(self.trace),
            "action_type": action_type,
            "target_action": action.to_dict(),
            "status": "executed" if result.success else "failed",
        }
        if result.error:
            event["error"] = result.error
        self.trace.append(event)
        if not result.success:
            self.error = str(result.error or "droidrun_host_action_failed")
            raise RuntimeError(self.error)
        self.actions_executed += 1

    def _point_args(self, x: float, y: float) -> dict[str, float]:
        observation = Observation.from_value(
            self.host.observe(xml=False, screenshot=False, app_info=False)
        )
        display = observation.extra.get("display")
        if not isinstance(display, dict):
            raise TypeError("droidrun_target_display_required")
        width = _positive_number(display.get("width"), "target_width")
        height = _positive_number(display.get("height"), "target_height")
        return {
            "x": float(x) * _COORDINATE_MAXIMUM / width,
            "y": float(y) * _COORDINATE_MAXIMUM / height,
        }


def load_official_droidrun_macro_player() -> type[Any]:
    """Load DroidRun's native player after repairing its published SDK alias."""

    installed = metadata.version("droidrun")
    if installed != DROIDRUN_VERSION:
        raise RuntimeError(
            f"droidrun_version_mismatch:expected={DROIDRUN_VERSION}:actual={installed}"
        )
    _install_published_sdk_alias()
    from droidrun.macro import MacroPlayer

    return MacroPlayer


def _install_published_sdk_alias() -> None:
    try:
        importlib.import_module("mobilerun")
        return
    except ModuleNotFoundError as error:
        if error.name != "mobilerun":
            raise
    sdk = importlib.import_module("mobilerun_sdk")
    exceptions = importlib.import_module("mobilerun_sdk._exceptions")
    sys.modules["mobilerun"] = sdk
    sys.modules["mobilerun._exceptions"] = exceptions


def _to_droidrun_action(
    action: Action,
    *,
    source_state: dict[str, Any],
) -> dict[str, Any]:
    width, height = _display(source_state)
    args = dict(action.args)
    if action.tool == "click":
        return {
            "action_type": "tap",
            "x": _source_pixel(args.get("x"), width, "x"),
            "y": _source_pixel(args.get("y"), height, "y"),
        }
    if action.tool == "swipe":
        return {
            "action_type": "swipe",
            "start_x": _source_pixel(args.get("x1"), width, "x1"),
            "start_y": _source_pixel(args.get("y1"), height, "y1"),
            "end_x": _source_pixel(args.get("x2"), width, "x2"),
            "end_y": _source_pixel(args.get("y2"), height, "y2"),
            "duration_ms": float(args.get("duration_ms") or 1000.0),
        }
    if action.tool == "open_app":
        package = str(args.get("package_name") or "").strip()
        if not package:
            raise ValueError("droidrun_source_package_required")
        return {"action_type": "start_app", "package": package, "activity": None}
    if action.tool in {"press_key", "press_back", "press_home"}:
        button = str(args.get("key") or "").strip().lower()
        if action.tool == "press_back":
            button = "back"
        elif action.tool == "press_home":
            button = "home"
        if button not in {"back", "home", "enter"}:
            raise ValueError(f"droidrun_source_button_unsupported:{button}")
        return {"action_type": "button_press", "button": button}
    if action.tool == "wait":
        return {
            "action_type": "wait",
            "duration": float(args.get("seconds") or args.get("duration") or 1.0),
        }
    raise ValueError(f"droidrun_source_action_unsupported:{action.tool}")


def _resolve_macro_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    macro_path = resolved / "macro.json" if resolved.is_dir() else resolved
    if not macro_path.is_file():
        raise FileNotFoundError(f"droidrun_macro_missing:{macro_path}")
    return macro_path


def _validate_macro(value: Any) -> None:
    if (
        not isinstance(value, dict)
        or value.get("version") != DROIDRUN_MACRO_VERSION
        or not isinstance(value.get("actions"), list)
        or not value["actions"]
        or value.get("total_actions") != len(value["actions"])
    ):
        raise ValueError("droidrun_macro_contract_invalid")


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"json_object_required:{path}")
    return value


def _display(state: dict[str, Any]) -> tuple[float, float]:
    display = state.get("display")
    if not isinstance(display, dict):
        raise TypeError("droidrun_source_display_required")
    return (
        _positive_number(display.get("width"), "source_width"),
        _positive_number(display.get("height"), "source_height"),
    )


def _source_states(
    run_log: dict[str, Any],
    source_state_catalog: str | Path,
) -> dict[str, dict[str, Any]]:
    if all(str(step.get("before_state_id") or "").strip() for step in run_log["steps"]):
        return load_transfer_state_catalog(source_state_catalog)
    return {}


def _embedded_source_state(step: dict[str, Any]) -> dict[str, Any]:
    observation = step.get("observation")
    if not isinstance(observation, dict):
        raise TypeError("droidrun_embedded_source_observation_required")
    auxiliaries = observation.get("auxiliaries")
    if not isinstance(auxiliaries, dict):
        raise TypeError("droidrun_embedded_source_auxiliaries_required")
    display = auxiliaries.get("display")
    if not isinstance(display, dict):
        raise TypeError("droidrun_source_display_required")
    return {"display": display}


def _source_action(step: dict[str, Any]) -> Action:
    raw_action = step.get("action")
    if not isinstance(raw_action, dict):
        raise ValueError("droidrun_source_step_invalid")
    if "tool" in raw_action:
        return Action.from_value(raw_action)
    observation = step.get("observation")
    if not isinstance(observation, dict):
        raise TypeError("droidrun_embedded_source_observation_required")
    return Action.from_value(
        _androidworld_action_to_omniflow(raw_action, observation=observation)
    )


def _positive_number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"droidrun_{name}_invalid")
    return float(value)


def _source_pixel(value: Any, extent: float, name: str) -> int:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or value < 0
        or value > _COORDINATE_MAXIMUM
    ):
        raise ValueError(f"droidrun_source_{name}_out_of_range")
    coordinate = float(value)
    return round(coordinate * extent / _COORDINATE_MAXIMUM)


def _macro_timestamp(run_log: dict[str, Any]) -> str:
    finished_at_ms = run_log.get("finished_at_ms")
    if isinstance(finished_at_ms, (int, float)) and not isinstance(
        finished_at_ms, bool
    ):
        instant = dt.datetime.fromtimestamp(
            float(finished_at_ms) / 1000.0,
        tz=dt.UTC,
        )
        return instant.strftime("%Y%m%d_%H%M%S")
    return "19700101_000000"


__all__ = [
    "DROIDRUN_MACRO_VERSION",
    "DROIDRUN_MANIFEST_SCHEMA",
    "compile_droidrun_macro",
    "load_official_droidrun_macro_player",
    "run_droidrun_macro_replay",
]
