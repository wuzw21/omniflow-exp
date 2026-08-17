"""B-MoCA lifecycle and Host bridge; OmniFlow remains the execution method."""

from __future__ import annotations

import base64
from contextlib import contextmanager, suppress
import csv
from dataclasses import dataclass
import importlib
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Iterator, Sequence

import numpy as np
from PIL import Image

from omniflow import Action, ActionResult, Observation
from omniflow.core.trajectory import state_id
from omniflow.transfer.runtime import capture_transfer_state
from src.experiment.observation_evidence import AndroidWorldEpisodeRecorder

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
class BMocaEpisode:
    task_id: str
    task_path: Path
    goal: str
    max_steps: int
    environment_id: str
    snapshot_id: str
    avd_name: str


@dataclass(frozen=True)
class BMocaEnvironmentConfig:
    bmoca_root: Path
    android_sdk_root: Path
    android_avd_home: Path
    avd_template_home: Path | None = None
    run_headless: bool = True

    @classmethod
    def resolve(
        cls,
        *,
        bmoca_root: str | Path,
        android_sdk_root: str | Path,
        android_avd_home: str | Path,
        avd_template_home: str | Path | None = None,
        run_headless: bool = True,
    ) -> "BMocaEnvironmentConfig":
        return cls(
            bmoca_root=Path(bmoca_root).expanduser().resolve(),
            android_sdk_root=Path(android_sdk_root).expanduser().resolve(),
            android_avd_home=Path(android_avd_home).expanduser().resolve(),
            avd_template_home=(
                Path(avd_template_home).expanduser().resolve()
                if avd_template_home
                else None
            ),
            run_headless=bool(run_headless),
        )


class BMocaHost:
    """Present one official B-MoCA episode through OmniFlow's Host protocol."""

    def __init__(
        self,
        environment: Any,
        *,
        snapshot_id: str,
        source_states: dict[str, dict[str, Any]] | None = None,
        evidence_root: str | Path | None = None,
    ) -> None:
        self.environment = environment
        self.snapshot_id = str(snapshot_id)
        self.source_states = dict(source_states or {})
        self.timestep: Any | None = None
        self.environment_error: str | None = None
        self._xml_cache = ""
        self._captured_transfer_states: dict[str, dict[str, Any]] = {}
        self._recorder = (
            AndroidWorldEpisodeRecorder(
                self._recording_state,
                self._unused_execute_action,
                evidence_root=evidence_root,
            )
            if evidence_root is not None
            else None
        )

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
        self.timestep = self.environment.reset(target_env_id=self.snapshot_id)
        self._xml_cache = ""
        if self._recorder is not None:
            self._recorder.start_episode()

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
        if self._recorder is not None:
            recorded = self._recorder.get_state()
            auxiliaries = dict(recorded.auxiliaries)
            pixels = bytes(recorded.pixels)
            identified_state_id = state_id(
                {
                    "pixels": None,
                    "forest": recorded.forest,
                    "ui_elements": list(recorded.ui_elements),
                    "auxiliaries": auxiliaries,
                }
            )
            observation = Observation(
                xml=str(recorded.forest or "") if xml else None,
                package_name=str(auxiliaries.get("package_name") or "") or None
                if app_info
                else None,
                activity_name=str(auxiliaries.get("activity_name") or "") or None
                if app_info
                else None,
                image_base64=(
                    base64.b64encode(pixels).decode("ascii") if screenshot else None
                ),
                extra={
                    **auxiliaries,
                    "state_id": identified_state_id,
                },
            )
            transfer_state = capture_transfer_state(observation)
            self._captured_transfer_states[identified_state_id] = transfer_state
            return observation
        package_name, activity_name = self._app_info() if app_info else ("", "")
        height, width = self._screen_size()
        return Observation(
            xml=self._xml_text() if xml else None,
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
        if self._recorder is not None:
            return self._recorder.execute_host_action(
                action,
                execute=lambda: self._act(action),
                project=self._recording_action,
            )
        return self._act(action)

    def _act(self, action: Action) -> ActionResult:
        if self.timestep is None:
            return ActionResult(False, "bmoca_episode_not_reset")
        if self.episode_done:
            return ActionResult(False, "bmoca_episode_ended")
        if action.tool == "open_app":
            return self._open_app(action)
        try:
            gesture = self._gesture(action)
            self.timestep = self.environment.step(np.asarray(gesture, dtype=float))
            self._xml_cache = ""
        except (TypeError, ValueError) as error:
            return ActionResult(False, str(error))
        except Exception as error:  # noqa: BLE001 - official environment boundary
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

    def seal_run_log(
        self,
        *,
        task_name: str,
        goal: str,
        diagnostics: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if self._recorder is None:
            return None
        return self._recorder.seal_run_log(
            task_name=task_name,
            goal=goal,
            task_parameters={
                "benchmark": "b-moca",
                "snapshot_id": self.snapshot_id,
            },
            seed=None,
            validator_success=self.official_success,
            validator_reward=float(
                getattr(self.timestep, "curr_rew", 0.0) or 0.0
            ),
            diagnostics=diagnostics,
        )

    def persist_observations(self) -> list[dict[str, Any]] | None:
        return self._recorder.persist_observations() if self._recorder is not None else None

    def get_captured_transfer_states(self) -> dict[str, dict[str, Any]]:
        return {
            state_identifier: dict(self._captured_transfer_states[state_identifier])
            for state_identifier in sorted(self._captured_transfer_states)
        }

    def get_state(self, source_state_id: str) -> Observation | None:
        state = self.source_states.get(str(source_state_id or ""))
        return Observation.from_value(state) if state is not None else None

    def close(self) -> None:
        close = getattr(self.environment, "close", None)
        if callable(close):
            close()

    def _open_app(self, action: Action) -> ActionResult:
        package_name = str(action.args.get("package_name") or "").strip()
        driver = self._driver()
        activate = getattr(driver, "activate_app", None)
        if not package_name:
            return ActionResult(False, "open_app_package_name_required")
        if not callable(activate):
            return ActionResult(False, "bmoca_activate_app_unavailable")
        try:
            activate(package_name)
            self._xml_cache = ""
        except Exception as error:  # noqa: BLE001 - Appium boundary
            self.environment_error = f"bmoca_open_app_failed:{error}"
            return ActionResult(False, self.environment_error)
        return ActionResult(True)

    def _gesture(self, action: Action) -> tuple[float, float, float, float]:
        maximum = 1000.0
        if action.tool == "click":
            x = _relative_coordinate(action.args.get("x"), "x", maximum=maximum)
            y = _relative_coordinate(action.args.get("y"), "y", maximum=maximum)
            x, y = self._ui_to_official_touch(x, y)
            return y, x, y, x
        if action.tool == "swipe":
            keys = ("x1", "y1", "x2", "y2")
            if all(action.args.get(key) is not None for key in keys):
                x1, y1, x2, y2 = (
                    _relative_coordinate(action.args[key], key, maximum=maximum)
                    for key in keys
                )
                x1, y1 = self._ui_to_official_touch(x1, y1)
                x2, y2 = self._ui_to_official_touch(x2, y2)
                return y1, x1, y2, x2
        key = str(
            action.args.get("key") or action.args.get("keycode") or ""
        ).upper().removeprefix("KEYCODE_")
        if action.tool == "press_back":
            key = "BACK"
        elif action.tool == "press_home":
            key = "HOME"
        if action.tool == "press_key" and not key:
            raise ValueError("press_key_key_required")
        if key in _NAVIGATION_GESTURES:
            gesture = list(_NAVIGATION_GESTURES[key])
            if self._is_tablet():
                gesture[1], gesture[3] = 1 - gesture[1], 1 - gesture[3]
            return tuple(gesture)
        raise ValueError(f"bmoca_official_action_unsupported:{action.tool}")

    def _driver(self) -> Any | None:
        return getattr(getattr(self.environment, "_coordinator", None), "_driver", None)

    def _is_tablet(self) -> bool:
        return bool(
            getattr(getattr(self.environment, "_coordinator", None), "_is_tablet", False)
        )

    def _screen_size(self) -> tuple[int, int]:
        return self._official_touch_size()

    def _official_touch_size(self) -> tuple[int, int]:
        raw = getattr(getattr(self.environment, "_coordinator", None), "_screen_size", None)
        values = tuple(raw) if raw is not None else ()
        if len(values) != 2 or min(int(values[0]), int(values[1])) <= 0:
            raise RuntimeError("bmoca_screen_size_unavailable")
        return int(values[0]), int(values[1])

    def _ui_to_official_touch(self, x: float, y: float) -> tuple[float, float]:
        return x, y

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
        package_name = activity_name = ""
        with suppress(Exception):
            package_name = str(driver.current_package or "")
        with suppress(Exception):
            activity_name = str(driver.current_activity or "")
        return package_name, activity_name

    def _image_base64(self) -> str | None:
        payload = self._image_bytes()
        return base64.b64encode(payload).decode("ascii") if payload else None

    def _image_bytes(self) -> bytes | None:
        driver = self._driver()
        screenshot = getattr(driver, "get_screenshot_as_png", None)
        if callable(screenshot):
            with suppress(Exception):
                payload = screenshot()
                if isinstance(payload, (bytes, bytearray)) and payload:
                    return bytes(payload)
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
        return output.getvalue()

    def _recording_state(self) -> SimpleNamespace:
        package_name, activity_name = self._app_info()
        height, width = self._screen_size()
        return SimpleNamespace(
            pixels=self._image_bytes(),
            forest=self._xml_text(),
            ui_elements=[],
            auxiliaries={
                "benchmark": "b-moca",
                "snapshot_id": self.snapshot_id,
                "package_name": package_name,
                "activity_name": activity_name,
                "display": {"width": width, "height": height},
                "official_success": self.official_success,
                "episode_done": self.episode_done,
            },
        )

    @staticmethod
    def _unused_execute_action(*_: Any, **__: Any) -> None:
        raise RuntimeError("bmoca_recorder_uses_host_action_boundary")

    def _recording_action(self, action: Action | dict[str, Any]) -> dict[str, Any]:
        value = Action.from_value(action)
        args = dict(value.args)
        height, width = self._screen_size()
        if value.tool in {"click", "long_press"}:
            return {
                "action_type": value.tool,
                "x": float(args["x"]) / 1000.0 * width,
                "y": float(args["y"]) / 1000.0 * height,
            }
        if value.tool == "swipe":
            direction = str(args.get("direction") or "").strip().lower()
            if not direction:
                direction = _swipe_direction(args)
            return {"action_type": "swipe", "direction": direction}
        if value.tool == "open_app":
            return {
                "action_type": "open_app",
                "app_name": str(args.get("package_name") or ""),
            }
        key = str(args.get("key") or args.get("keycode") or "").upper()
        if value.tool == "press_back" or key.endswith("BACK"):
            return {"action_type": "navigate_back"}
        if value.tool == "press_home" or key.endswith("HOME"):
            return {"action_type": "navigate_home"}
        if key.endswith("ENTER"):
            return {"action_type": "keyboard_enter"}
        if value.tool == "wait":
            return {"action_type": "wait"}
        return {"action_type": "unknown"}


@contextmanager
def open_bmoca_episode(
    episode: BMocaEpisode,
    *,
    config: BMocaEnvironmentConfig,
    source_states: dict[str, dict[str, Any]] | None = None,
    evidence_root: str | Path | None = None,
    appium_port: int = 4723,
    appium_system_port: int = 8200,
) -> Iterator[BMocaHost]:
    """Open one official B-MoCA episode and expose only the Host contract."""

    _configure_runtime(config)
    if config.avd_template_home is not None:
        _restore_writable_avd_disks(
            episode.avd_name,
            avd_home=config.android_avd_home,
            template_home=config.avd_template_home,
        )
    module = importlib.import_module("bmoca.environment.environment")
    environment_class = getattr(module, "BMocaEnv")
    sdk = config.android_sdk_root
    environment = environment_class(
        task_path=str(episode.task_path),
        avd_name=episode.avd_name,
        state_type="text",
        action_tanh=False,
        adjusting_freq=1.0 / 3.0,
        run_headless=config.run_headless,
        android_avd_home=str(config.android_avd_home),
        android_sdk_root=str(sdk),
        emulator_path=str(sdk / "emulator/emulator"),
        adb_path=str(sdk / "platform-tools/adb"),
        appium_port=int(appium_port),
        appium_system_port=int(appium_system_port),
    )
    host: BMocaHost | None = None
    try:
        _wait_for_emulator_ready(environment, adb_path=sdk / "platform-tools/adb")
        _install_snapshot_ready_gate(environment, adb_path=sdk / "platform-tools/adb")
        host = BMocaHost(
            environment,
            snapshot_id=episode.snapshot_id,
            source_states=source_states,
            evidence_root=evidence_root,
        )
        host.reset()
        yield host
    finally:
        if host is not None:
            with suppress(Exception):
                host.close()
        else:
            close = getattr(environment, "close", None)
            if callable(close):
                with suppress(Exception):
                    close()


def discover_bmoca_episodes(
    bmoca_root: str | Path,
    *,
    task_id: str,
    environment_ids: Sequence[str],
) -> tuple[BMocaEpisode, ...]:
    root = Path(bmoca_root).expanduser().resolve()
    task_path = root / "asset/tasks" / f"{task_id}.textproto"
    if not task_path.is_file():
        raise FileNotFoundError(f"bmoca_task_missing:{task_path}")
    task_text = task_path.read_text(encoding="utf-8")
    max_match = re.search(r"^\s*max_episode_steps\s*:\s*(\d+)\s*$", task_text, re.M)
    if max_match is None:
        raise ValueError("bmoca_task_max_steps_missing")
    instruction_match = re.search(
        r'^\s*(?:instruction|goal)\s*:\s*"([^"]+)"\s*$',
        task_text,
        re.M,
    )
    goal = (
        instruction_match.group(1)
        if instruction_match is not None
        else task_id.rsplit("/", 1)[-1].replace("_", " ")
    )
    catalog_path = root / "asset/environments/config/environments_test.csv"
    with catalog_path.open(newline="", encoding="utf-8") as stream:
        devices = {
            str(row["idx"]): str(row["device_id"])
            for row in csv.DictReader(stream)
        }
    requested = tuple(str(environment_id) for environment_id in environment_ids)
    missing = [environment_id for environment_id in requested if environment_id not in devices]
    if missing:
        raise ValueError("bmoca_unknown_environments:" + ",".join(missing))
    return tuple(
        BMocaEpisode(
            task_id=str(task_id),
            task_path=task_path,
            goal=goal,
            max_steps=int(max_match.group(1)),
            environment_id=environment_id,
            snapshot_id=f"test_env_{environment_id}",
            avd_name=f"{devices[environment_id]}_test_00",
        )
        for environment_id in requested
    )


def _configure_runtime(config: BMocaEnvironmentConfig) -> None:
    os.environ.update(
        {
            "BMOCA_HOME": str(config.bmoca_root),
            "ANDROID_HOME": str(config.android_sdk_root),
            "ANDROID_SDK_ROOT": str(config.android_sdk_root),
            "ANDROID_AVD_HOME": str(config.android_avd_home),
        }
    )
    root_text = str(config.bmoca_root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()


def _wait_for_emulator_ready(
    environment: Any,
    *,
    adb_path: Path,
    timeout_seconds: float = 120.0,
) -> str:
    simulator = getattr(environment, "_simulator", None)
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


def _install_snapshot_ready_gate(environment: Any, *, adb_path: Path) -> None:
    coordinator = getattr(environment, "_coordinator", None)
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


def _relative_coordinate(value: Any, name: str, *, maximum: float) -> float:
    try:
        coordinate = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"bmoca_{name}_coordinate_required") from error
    if not 0.0 <= coordinate <= maximum:
        raise ValueError(f"bmoca_{name}_coordinate_out_of_range:{coordinate}")
    return coordinate / maximum


def _swipe_direction(args: dict[str, Any]) -> str:
    try:
        dx = float(args["x2"]) - float(args["x1"])
        dy = float(args["y2"]) - float(args["y1"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("bmoca_swipe_direction_required") from error
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"
