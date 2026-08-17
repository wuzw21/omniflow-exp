"""Official-device B-MoCA Function replay and native OmniFlow E2E evaluation."""

from __future__ import annotations

import base64
from contextlib import suppress
import csv
from dataclasses import dataclass, replace
import importlib
import io
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
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
from omniflow.vlm.planner import VLMPlanner

_NAVIGATION_GESTURES = {
    "BACK": (252 / 256, 43 / 128, 252 / 256, 43 / 128),
    "HOME": (252 / 256, 64 / 128, 252 / 256, 64 / 128),
    "OVERVIEW": (252 / 256, 85 / 128, 252 / 256, 85 / 128),
}
_WRITABLE_AVD_DISKS = (
    "cache.img.qcow2",
    "encryptionkey.img.qcow2",
    "userdata-qemu.img.qcow2",
)


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
        except (TypeError, ValueError) as error:
            return ActionResult(False, str(error))
        try:
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
        maximum = 2000.0 if self._is_tablet() else 1000.0
        if action.tool == "click":
            x = _relative_coordinate(action.args.get("x"), "x", maximum=maximum)
            y = _relative_coordinate(action.args.get("y"), "y", maximum=maximum)
            touch_y, touch_x = self._touch_point(x, y)
            return touch_y, touch_x, touch_y, touch_x
        if action.tool == "swipe":
            keys = ("x1", "y1", "x2", "y2")
            if all(action.args.get(key) is not None for key in keys):
                x1, y1, x2, y2 = (
                    _relative_coordinate(action.args[key], key, maximum=maximum)
                    for key in keys
                )
                touch_y1, touch_x1 = self._touch_point(x1, y1)
                touch_y2, touch_x2 = self._touch_point(x2, y2)
                return touch_y1, touch_x1, touch_y2, touch_x2
        key = str(
            action.args.get("key") or action.args.get("keycode") or ""
        ).upper().removeprefix("KEYCODE_")
        if action.tool == "press_back":
            key = "BACK"
        elif action.tool == "press_home":
            key = "HOME"
        if key in _NAVIGATION_GESTURES:
            gesture = list(_NAVIGATION_GESTURES[key])
            if self._is_tablet():
                gesture[1], gesture[3] = 1 - gesture[1], 1 - gesture[3]
            return tuple(gesture)
        raise ValueError(f"bmoca_official_action_unsupported:{action.tool}")

    def _touch_point(self, x: float, y: float) -> tuple[float, float]:
        return y, x

    def _is_tablet(self) -> bool:
        coordinator = getattr(self.env, "_coordinator", None)
        return bool(getattr(coordinator, "_is_tablet", False))

    def _screen_size(self) -> tuple[int, int]:
        coordinator = getattr(self.env, "_coordinator", None)
        raw = getattr(coordinator, "_screen_size", None)
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
    avd_template_home: str | Path,
    run_headless: bool = True,
) -> dict[str, Any]:
    """Run one stored Function on official B-MoCA snapshots."""

    return _evaluate_device(
        bmoca_root=bmoca_root,
        store_path=store_path,
        task_id=task_id,
        environment_ids=environment_ids,
        android_sdk_root=android_sdk_root,
        android_avd_home=android_avd_home,
        avd_template_home=avd_template_home,
        run_headless=run_headless,
        planner_model=None,
        planner_provider="openai_compatible",
        planner_timeout_seconds=60.0,
    )


def evaluate_device_omniflow_e2e(
    *,
    bmoca_root: str | Path,
    store_path: str | Path,
    task_id: str,
    planner_model: str,
    environment_ids: Sequence[str] = ("100", "101", "105"),
    android_sdk_root: str | Path,
    android_avd_home: str | Path,
    avd_template_home: str | Path,
    planner_provider: str = "openai_compatible",
    planner_timeout_seconds: float = 60.0,
    run_headless: bool = True,
) -> dict[str, Any]:
    """Run the native observe/recall/plan/act OmniFlow loop on B-MoCA."""

    model = str(planner_model or "").strip()
    if not model:
        raise ValueError("bmoca_e2e_planner_model_required")
    return _evaluate_device(
        bmoca_root=bmoca_root,
        store_path=store_path,
        task_id=task_id,
        environment_ids=environment_ids,
        android_sdk_root=android_sdk_root,
        android_avd_home=android_avd_home,
        avd_template_home=avd_template_home,
        run_headless=run_headless,
        planner_model=model,
        planner_provider=planner_provider,
        planner_timeout_seconds=planner_timeout_seconds,
    )


def _evaluate_device(
    *,
    bmoca_root: str | Path,
    store_path: str | Path,
    task_id: str,
    environment_ids: Sequence[str],
    android_sdk_root: str | Path,
    android_avd_home: str | Path,
    avd_template_home: str | Path,
    run_headless: bool,
    planner_model: str | None,
    planner_provider: str,
    planner_timeout_seconds: float,
) -> dict[str, Any]:

    root = Path(bmoca_root).expanduser().resolve()
    store = Path(store_path).expanduser().resolve()
    sdk = Path(android_sdk_root).expanduser().resolve()
    avd_home = Path(android_avd_home).expanduser().resolve()
    template_home = Path(avd_template_home).expanduser().resolve()
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
                template_home=template_home,
                run_headless=run_headless,
                planner_model=planner_model,
                planner_provider=planner_provider,
                planner_timeout_seconds=planner_timeout_seconds,
            )
        )
    successes = sum(item["official_success"] for item in results)
    function_invocations = sum(item["function_invoked"] for item in results)
    function_actions = sum(item["function_actions_executed"] for item in results)
    native_actions = sum(item["native_actions_executed"] for item in results)
    return {
        "schema_version": "omniflow.bmoca-device-function-replay.v1",
        "configuration": {
            "task_id": task_id,
            "environment_ids": list(environment_ids),
            "execution": "official_bmoca_device",
            "function_replay": (
                "native_omniflow_e2e"
                if planner_model is not None
                else "direct_single_function"
            ),
            "planner_model": planner_model,
            "planner_provider": planner_provider if planner_model else None,
            "planner_timeout_seconds": (
                planner_timeout_seconds if planner_model else None
            ),
            "checker": "optional_function_step_via_omnitransfer",
            "dp": "disabled",
            "runtime_vlm_scope": (
                "planner_only" if planner_model else "disabled"
            ),
            "fallback_steps_allowed": 0,
            "source_coordinate_fallback": "disabled",
            "episode_isolation": "restore_writable_avd_disks_from_template",
            "avd_template_home": str(template_home),
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
            "planner_steps": sum(item["planner_steps"] for item in results),
            "prompt_tokens": sum(
                int(item["llm_usage"].get("prompt_tokens") or 0)
                for item in results
            ),
            "completion_tokens": sum(
                int(item["llm_usage"].get("completion_tokens") or 0)
                for item in results
            ),
            "total_tokens": sum(
                int(item["llm_usage"].get("total_tokens") or 0)
                for item in results
            ),
            "fallback_steps": sum(item["fallback_steps"] for item in results),
            "function_invocation_count": function_invocations,
            "function_invocation_rate": (
                function_invocations / len(results) if results else 0.0
            ),
            "function_actions_executed": function_actions,
            "native_actions_executed": native_actions,
            "function_action_reuse_rate": (
                function_actions / (function_actions + native_actions)
                if function_actions + native_actions
                else 0.0
            ),
            "checker_steps_executed": sum(
                decision.get("status") == "executed"
                for item in results
                for decision in item["checker_decisions"]
            ),
            "checker_steps_skipped": sum(
                decision.get("status") == "skipped"
                for item in results
                for decision in item["checker_decisions"]
            ),
            "checker_steps_failed": sum(
                decision.get("status") == "failed"
                for item in results
                for decision in item["checker_decisions"]
            ),
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
    template_home: Path,
    run_headless: bool,
    planner_model: str | None,
    planner_provider: str,
    planner_timeout_seconds: float,
) -> dict[str, Any]:
    started = time.monotonic()
    env = host = None
    try:
        try:
            _restore_writable_avd_disks(
                episode.avd_name,
                avd_home=avd_home,
                template_home=template_home,
            )
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
            _wait_for_emulator_ready(env, adb_path=sdk / "platform-tools/adb")
            _install_snapshot_ready_gate(
                env,
                adb_path=sdk / "platform-tools/adb",
            )
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
        planner = (
            VLMPlanner(
                model=planner_model,
                provider=planner_provider,
                timeout=planner_timeout_seconds,
                max_steps=episode.max_steps,
            )
            if planner_model is not None
            else None
        )
        flow = OmniFlow(
            store_path,
            host=_StateHost(host, states),
            planner=planner,
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
        result = (
            flow.run(episode.instruction)
            if planner is not None
            else flow.call_tool(ToolCall(function.id, {}))
        )
        if planner is None and result.model_calls:
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
        trace = list(result.detail.get("trace") or [])
        function_resolution = dict(
            result.detail.get("function_resolution") or {}
        )
        function_actions = sum(
            bool(item.get("metadata", {}).get("function_id"))
            for item in trace
            if isinstance(item, dict)
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
            trace=trace,
            checker_decisions=list(result.detail.get("checker_decisions") or []),
            planner_steps=int(result.detail.get("planner_steps") or 0),
            llm_usage=dict(result.detail.get("llm_usage") or {}),
            function_resolution=function_resolution,
            done_reason=str(result.detail.get("done_reason") or "") or None,
            function_invoked=bool(
                function_resolution.get("selected_function_id")
            ),
            function_actions_executed=function_actions,
            native_actions_executed=max(
                0, int(result.actions_executed) - function_actions
            ),
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
    checker_decisions: list[dict[str, Any]] | None = None,
    planner_steps: int = 0,
    llm_usage: dict[str, Any] | None = None,
    function_resolution: dict[str, Any] | None = None,
    done_reason: str | None = None,
    function_invoked: bool = False,
    function_actions_executed: int = 0,
    native_actions_executed: int = 0,
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
        "checker_decisions": list(checker_decisions or []),
        "planner_steps": int(planner_steps),
        "llm_usage": dict(llm_usage or {}),
        "function_resolution": dict(function_resolution or {}),
        "done_reason": done_reason,
        "function_invoked": bool(function_invoked),
        "function_actions_executed": int(function_actions_executed),
        "native_actions_executed": int(native_actions_executed),
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


def _wait_for_emulator_ready(
    env: Any,
    *,
    adb_path: Path,
    timeout_seconds: float = 120.0,
) -> str:
    simulator = getattr(env, "_simulator", None)
    adb_port = int(getattr(simulator, "_adb_port", 0) or 0)
    if adb_port <= 1:
        raise RuntimeError("bmoca_emulator_adb_port_unavailable")
    serial = f"emulator-{adb_port - 1}"
    deadline = time.monotonic() + float(timeout_seconds)
    last_state = "unknown"
    while time.monotonic() < deadline:
        try:
            state = subprocess.run(
                [str(adb_path), "-s", serial, "get-state"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            last_state = state.stdout.strip() or state.stderr.strip() or "unknown"
            if state.returncode == 0 and state.stdout.strip() == "device":
                boot = subprocess.run(
                    [
                        str(adb_path),
                        "-s",
                        serial,
                        "shell",
                        "getprop",
                        "sys.boot_completed",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if boot.returncode == 0 and boot.stdout.strip() == "1":
                    return serial
        except (OSError, subprocess.TimeoutExpired) as error:
            last_state = str(error) or type(error).__name__
        time.sleep(1.0)
    raise RuntimeError(f"bmoca_emulator_not_ready:{serial}:{last_state}")


def _install_snapshot_ready_gate(env: Any, *, adb_path: Path) -> None:
    coordinator = getattr(env, "_coordinator", None)
    if coordinator is None or getattr(coordinator, "_omniflow_ready_gate", False):
        return
    task_manager = getattr(coordinator, "_task_manager", None)
    simulator = getattr(coordinator, "_simulator", None)
    if task_manager is None or simulator is None:
        raise RuntimeError("bmoca_snapshot_ready_gate_unavailable")

    def load_snapshot(request: Any) -> Any:
        task_manager.stop()
        response = simulator.load_state(request)
        _wait_for_emulator_ready(coordinator, adb_path=adb_path)
        task_manager.start(
            adb_call_parser_factory=coordinator._create_adb_call_parser,
            log_stream=simulator.create_log_stream(),
        )
        return response

    coordinator.load_snapshot = load_snapshot
    coordinator._omniflow_ready_gate = True


def _restore_writable_avd_disks(
    avd_name: str,
    *,
    avd_home: Path,
    template_home: Path,
) -> None:
    source_root = template_home / f"{avd_name}.avd"
    target_root = avd_home / f"{avd_name}.avd"
    if not target_root.is_dir():
        raise FileNotFoundError(f"bmoca_avd_missing:{target_root}")
    for filename in _WRITABLE_AVD_DISKS:
        source = source_root / filename
        target = target_root / filename
        if not source.is_file():
            raise FileNotFoundError(f"bmoca_avd_template_disk_missing:{source}")
        temporary = target.with_name(f".{target.name}.restore-{os.getpid()}")
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)
        finally:
            with suppress(FileNotFoundError):
                temporary.unlink()


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


def _relative_coordinate(
    value: Any,
    name: str,
    *,
    maximum: float = 1000.0,
) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"bmoca_{name}_coordinate_invalid") from error
    if not math.isfinite(numeric) or not 0 <= numeric <= maximum:
        raise ValueError(f"bmoca_{name}_coordinate_out_of_range")
    return numeric / 1000.0


__all__ = [
    "evaluate_device_function_replay",
    "evaluate_device_omniflow_e2e",
]
