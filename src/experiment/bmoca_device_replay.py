"""Official-device B-MoCA Function replay with zero model execution."""

from __future__ import annotations

import base64
from contextlib import suppress
import csv
from dataclasses import dataclass, replace
import importlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Sequence

import numpy as np
from PIL import Image

from omniflow import (
    Action,
    ActionResult,
    Observation,
    OmniFlow,
    OmniFlowConfig,
    PluginSet,
    RuntimeSettings,
    ToolCall,
)
from omniflow.transfer.runtime import load_transfer_state_catalog


_NAVIGATION_GESTURES = {
    "BACK": (252 / 256, 43 / 128, 252 / 256, 43 / 128),
    "HOME": (252 / 256, 64 / 128, 252 / 256, 64 / 128),
    "OVERVIEW": (252 / 256, 85 / 128, 252 / 256, 85 / 128),
}


@dataclass(frozen=True)
class _Episode:
    task_id: str
    task_path: Path
    instruction: str
    max_steps: int
    env_id: str
    snapshot_id: str
    avd_name: str


class _BMocaHost:
    def __init__(self, env: Any, *, snapshot_id: str):
        self.env = env
        self.snapshot_id = snapshot_id
        self.timestep: Any | None = None
        self.environment_error: str | None = None
        self._xml_cache = ""
        self._previous_android_serial: str | None = None
        self._previous_android_serial_present = False

    @property
    def official_success(self) -> bool:
        return bool(
            self.timestep is not None
            and float(getattr(self.timestep, "curr_rew", 0.0) or 0.0) > 0
        )

    @property
    def episode_done(self) -> bool:
        last = getattr(self.timestep, "last", None)
        return bool(last()) if callable(last) else False

    def reset(self) -> None:
        self.timestep = self.env.reset(target_env_id=self.snapshot_id)
        simulator = getattr(self.env, "_simulator", None)
        adb_port = int(getattr(simulator, "_adb_port", 0) or 0)
        if adb_port > 1:
            self._previous_android_serial_present = "ANDROID_SERIAL" in os.environ
            self._previous_android_serial = os.environ.get("ANDROID_SERIAL")
            os.environ["ANDROID_SERIAL"] = f"emulator-{adb_port - 1}"
        self._xml_cache = ""

    def observe(
        self,
        *,
        xml: bool = True,
        screenshot: bool = False,
        app_info: bool = True,
        **_: Any,
    ) -> Observation:
        if self.timestep is None:
            raise RuntimeError("bmoca_episode_not_reset")
        xml_text = self._xml_text() if xml else ""
        package_name, activity_name = self._app_info() if app_info else ("", "")
        height, width = self._screen_size()
        return Observation(
            xml=xml_text or None,
            package_name=package_name or None,
            activity_name=activity_name or None,
            image_base64=self._image_base64() if screenshot else None,
            extra={
                "benchmark": "b-moca",
                "snapshot_id": self.snapshot_id,
                "official_success": self.official_success,
                "episode_done": self.episode_done,
                "display": {"width": width, "height": height},
            },
        )

    def act(self, value: Action | dict[str, Any], **_: Any) -> ActionResult:
        action = Action.from_value(value)
        if self.timestep is None:
            return ActionResult(False, "bmoca_episode_not_reset")
        if self.episode_done:
            return ActionResult(False, "bmoca_episode_ended")
        try:
            gesture = self._gesture(action)
            self.timestep = self.env.step(np.asarray(gesture, dtype=float))
            self._xml_cache = ""
        except Exception as error:  # noqa: BLE001 - environment boundary
            self.environment_error = f"bmoca_step_failed:{error}"
            return ActionResult(False, self.environment_error)
        if self.timestep is None or getattr(self.timestep, "curr_obs", None) is None:
            self.environment_error = "bmoca_simulator_unhealthy"
            return ActionResult(False, self.environment_error)
        return ActionResult(
            True,
            extra={
                "official_success": self.official_success,
                "episode_done": self.episode_done,
            },
        )

    def get_app_label(self, _package_name: str) -> None:
        return None

    def close(self) -> None:
        try:
            close = getattr(self.env, "close", None)
            if callable(close):
                close()
        finally:
            if self._previous_android_serial_present:
                os.environ["ANDROID_SERIAL"] = str(
                    self._previous_android_serial or ""
                )
            else:
                os.environ.pop("ANDROID_SERIAL", None)

    def _gesture(self, action: Action) -> tuple[float, float, float, float]:
        if action.tool == "click":
            x = _relative_coordinate(action.args.get("x"), "x")
            y = _relative_coordinate(action.args.get("y"), "y")
            return y, x, y, x
        if action.tool == "swipe":
            keys = ("x1", "y1", "x2", "y2")
            if all(action.args.get(key) is not None for key in keys):
                x1, y1, x2, y2 = (
                    _relative_coordinate(action.args[key], key) for key in keys
                )
                return y1, x1, y2, x2
        key = str(
            action.args.get("key") or action.args.get("keycode") or ""
        ).upper().removeprefix("KEYCODE_")
        if action.tool == "press_back":
            key = "BACK"
        elif action.tool == "press_home":
            key = "HOME"
        if key in _NAVIGATION_GESTURES:
            return _NAVIGATION_GESTURES[key]
        raise ValueError(f"bmoca_official_action_unsupported:{action.tool}")

    def _screen_size(self) -> tuple[int, int]:
        raw = getattr(getattr(self.env, "_coordinator", None), "_screen_size", None)
        values = tuple(raw) if raw is not None else ()
        if len(values) != 2:
            raise RuntimeError("bmoca_screen_size_unavailable")
        height, width = int(values[0]), int(values[1])
        if min(height, width) <= 0:
            raise RuntimeError("bmoca_screen_size_invalid")
        return height, width

    def _driver(self) -> Any | None:
        return getattr(getattr(self.env, "_coordinator", None), "_driver", None)

    def _xml_text(self) -> str:
        if self._xml_cache:
            return self._xml_cache
        driver = self._driver()
        if driver is not None:
            with suppress(Exception):
                self._xml_cache = str(driver.page_source or "")
        return self._xml_cache

    def _app_info(self) -> tuple[str, str]:
        driver = self._driver()
        if driver is None:
            return "", ""
        with suppress(Exception):
            package_name = str(driver.current_package or "")
        with suppress(Exception):
            activity_name = str(driver.current_activity or "")
        return locals().get("package_name", ""), locals().get("activity_name", "")

    def _image_base64(self) -> str | None:
        observation = dict(getattr(self.timestep, "curr_obs", {}) or {})
        pixels = observation.get("pixel")
        if pixels is None:
            return None
        array = np.asarray(pixels)
        if np.issubdtype(array.dtype, np.floating):
            scale = 255.0 if float(array.max(initial=0.0)) <= 1.0 else 1.0
            array = np.clip(array * scale, 0, 255).astype(np.uint8)
        elif array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        output = io.BytesIO()
        Image.fromarray(array).save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("ascii")


class _StateHost:
    def __init__(self, host: _BMocaHost, states: dict[str, dict[str, Any]]):
        self.host = host
        self.states = states

    def observe(self, **kwargs: Any) -> Observation:
        return self.host.observe(**kwargs)

    def act(self, action: Any) -> ActionResult:
        return self.host.act(action)

    def get_state(self, state_id: str) -> Observation | None:
        state = self.states.get(str(state_id or ""))
        if state is None:
            return None
        observation = Observation.from_value(state)
        expected_package = _launcher_target_package(self.states, state_id=state_id)
        if expected_package is None:
            return observation
        return replace(
            observation,
            extra={**observation.extra, "expected_package_name": expected_package},
        )

    def get_app_label(self, package_name: str) -> None:
        return self.host.get_app_label(package_name)


def evaluate_device_function_replay(
    *,
    bmoca_root: str | Path,
    store_path: str | Path,
    task_id: str,
    environment_ids: Sequence[str] = ("100", "101", "105"),
    android_sdk_root: str | Path,
    android_avd_home: str | Path,
    run_headless: bool = True,
) -> dict[str, Any]:
    """Run one stored Function on official B-MoCA snapshots."""

    root = Path(bmoca_root).expanduser().resolve()
    store = Path(store_path).expanduser().resolve()
    sdk = Path(android_sdk_root).expanduser().resolve()
    avd_home = Path(android_avd_home).expanduser().resolve()
    _configure_runtime(root, sdk=sdk, avd_home=avd_home)
    episodes = _episodes(root, task_id=task_id, environment_ids=environment_ids)
    results = []
    started = time.monotonic()
    for episode in episodes:
        results.append(
            _evaluate_episode(
                episode,
                bmoca_root=root,
                store_path=store,
                sdk=sdk,
                avd_home=avd_home,
                run_headless=run_headless,
            )
        )
    successes = sum(item["official_success"] for item in results)
    return {
        "schema_version": "omniflow.bmoca-device-function-replay.v1",
        "configuration": {
            "task_id": task_id,
            "environment_ids": list(environment_ids),
            "execution": "official_bmoca_device",
            "function_replay": "direct_single_function",
            "dp": "disabled",
            "vlm_model_calls": 0,
            "fallback_steps_allowed": 0,
            "source_coordinate_fallback": "disabled",
            "store_path": str(store),
        },
        "summary": {
            "episode_count": len(results),
            "official_success_count": successes,
            "official_success_rate": successes / len(results) if results else 0.0,
            "environment_failure_count": sum(
                item["classification"] == "environment_failure" for item in results
            ),
            "method_failure_count": sum(
                item["classification"] == "method_failure" for item in results
            ),
            "model_calls": sum(item["model_calls"] for item in results),
            "fallback_steps": sum(item["fallback_steps"] for item in results),
            "wall_seconds": time.monotonic() - started,
        },
        "results": results,
    }


def _evaluate_episode(
    episode: _Episode,
    *,
    bmoca_root: Path,
    store_path: Path,
    sdk: Path,
    avd_home: Path,
    run_headless: bool,
) -> dict[str, Any]:
    started = time.monotonic()
    env = host = None
    try:
        try:
            module = importlib.import_module("bmoca.environment.environment")
            environment_class = getattr(module, "BMocaEnv")
            kwargs = {
                "task_path": str(episode.task_path),
                "avd_name": episode.avd_name,
                "state_type": "text",
                "action_tanh": False,
                "adjusting_freq": 1.0 / 3.0,
                "run_headless": run_headless,
                "android_avd_home": str(avd_home),
                "android_sdk_root": str(sdk),
                "emulator_path": str(sdk / "emulator/emulator"),
                "adb_path": str(sdk / "platform-tools/adb"),
            }
            env = environment_class(**kwargs)
            host = _BMocaHost(env, snapshot_id=episode.snapshot_id)
            host.reset()
        except Exception as error:  # noqa: BLE001 - environment boundary
            return _episode_result(
                episode,
                official_success=False,
                classification="environment_failure",
                error=str(error),
                duration=time.monotonic() - started,
            )
        states = load_transfer_state_catalog(store_path.parent / "transfer_states.json")
        flow = OmniFlow(
            store_path,
            host=_StateHost(host, states),
            planner=None,
            installed_apps={},
            config=OmniFlowConfig(
                runtime=RuntimeSettings(
                    max_steps=episode.max_steps,
                    max_fallback_steps=0,
                ),
                plugins=PluginSet(checker=lambda _context: None),
            ),
        )
        functions = list(flow.store.functions.values())
        if len(functions) != 1:
            raise RuntimeError(f"independent_task_function_required:{len(functions)}")
        function = functions[0]
        if function.input_schema.get("required"):
            raise RuntimeError("function_replay_arguments_required")
        result = flow.call_tool(ToolCall(function.id, {}))
        if result.model_calls:
            raise RuntimeError(
                f"zero_model_execution_violated:{result.model_calls}"
            )
        if result.fallback_steps:
            raise RuntimeError(
                f"zero_fallback_execution_violated:{result.fallback_steps}"
            )
        official_success = host.official_success
        classification = (
            "success"
            if official_success
            else "environment_failure"
            if host.environment_error
            else "method_failure"
        )
        return _episode_result(
            episode,
            official_success=official_success,
            classification=classification,
            error=result.error,
            duration=time.monotonic() - started,
            actions_executed=result.actions_executed,
            model_calls=result.model_calls,
            fallback_steps=result.fallback_steps,
            trace=list(result.detail.get("trace") or []),
        )
    except Exception as error:  # noqa: BLE001 - result boundary
        return _episode_result(
            episode,
            official_success=bool(host and host.official_success),
            classification=(
                "environment_failure"
                if host and host.environment_error
                else "method_failure"
            ),
            error=str(error),
            duration=time.monotonic() - started,
        )
    finally:
        if host is not None:
            with suppress(Exception):
                host.close()
        elif env is not None:
            with suppress(Exception):
                env.close()


def _episode_result(
    episode: _Episode,
    *,
    official_success: bool,
    classification: str,
    error: str | None,
    duration: float,
    actions_executed: int = 0,
    model_calls: int = 0,
    fallback_steps: int = 0,
    trace: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "task_id": episode.task_id,
        "environment_id": episode.env_id,
        "snapshot_id": episode.snapshot_id,
        "avd_name": episode.avd_name,
        "official_success": official_success,
        "classification": classification,
        "error": error,
        "actions_executed": int(actions_executed),
        "model_calls": int(model_calls),
        "fallback_steps": int(fallback_steps),
        "duration_seconds": duration,
        "trace": list(trace or []),
    }


def _configure_runtime(root: Path, *, sdk: Path, avd_home: Path) -> None:
    os.environ.update(
        {
            "BMOCA_HOME": str(root),
            "ANDROID_HOME": str(sdk),
            "ANDROID_SDK_ROOT": str(sdk),
            "ANDROID_AVD_HOME": str(avd_home),
        }
    )
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()


def _episodes(
    root: Path,
    *,
    task_id: str,
    environment_ids: Sequence[str],
) -> tuple[_Episode, ...]:
    task_path = root / "asset/tasks" / f"{task_id}.textproto"
    if not task_path.is_file():
        raise FileNotFoundError(f"bmoca_task_missing:{task_path}")
    task_text = task_path.read_text(encoding="utf-8")
    max_match = re.search(r"^\s*max_episode_steps\s*:\s*(\d+)\s*$", task_text, re.M)
    if max_match is None:
        raise ValueError("bmoca_task_max_steps_missing")
    instruction = f"Goal: {task_path.name.split('.', 1)[0].replace('_', ' ')}"
    catalog_path = root / "asset/environments/config/environments_test.csv"
    with catalog_path.open(newline="", encoding="utf-8") as stream:
        devices = {
            str(row["idx"]): str(row["device_id"])
            for row in csv.DictReader(stream)
        }
    missing = [str(env_id) for env_id in environment_ids if str(env_id) not in devices]
    if missing:
        raise ValueError("bmoca_unknown_environments:" + ",".join(missing))
    return tuple(
        _Episode(
            task_id=task_id,
            task_path=task_path,
            instruction=instruction,
            max_steps=int(max_match.group(1)),
            env_id=str(env_id),
            snapshot_id=f"test_env_{env_id}",
            avd_name=f"{devices[str(env_id)]}_test_00",
        )
        for env_id in environment_ids
    )


def _launcher_target_package(
    states: dict[str, dict[str, Any]],
    *,
    state_id: str,
) -> str | None:
    source = states.get(str(state_id or ""))
    if source is None or source.get("package_name") != (
        "com.google.android.apps.nexuslauncher"
    ):
        return None
    packages = {
        str(state.get("package_name") or "")
        for key, state in states.items()
        if key != state_id
        and state.get("package_name")
        and state.get("package_name") != source.get("package_name")
    }
    return next(iter(packages)) if len(packages) == 1 else None


def _relative_coordinate(value: Any, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"bmoca_{name}_coordinate_invalid") from error
    if not math.isfinite(numeric) or not 0 <= numeric <= 1000:
        raise ValueError(f"bmoca_{name}_coordinate_out_of_range")
    return numeric / 1000.0


__all__ = ["evaluate_device_function_replay"]
