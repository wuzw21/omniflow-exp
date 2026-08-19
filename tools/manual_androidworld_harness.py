"""Small interactive AndroidWorld harness for manual, one-step-at-a-time runs.

This intentionally does not import the experiment launcher or an agent.  The
operator sends one JSON command at a time, reads the native observation, and
chooses the next AndroidWorld JSONAction.  The official task validator remains
the only success signal.

In addition to ``act``, the interactive protocol accepts a native selector
click, for example ``{"cmd":"click","resource_name":"submitButton"}``.
The selector is resolved against a freshly observed UI tree and still executes
through AndroidWorld's official index-based JSONAction click.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
import uuid


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if hasattr(value, "DESCRIPTOR") and hasattr(value, "ListFields"):
        try:
            from google.protobuf.json_format import MessageToDict

            return MessageToDict(value, preserving_proto_field_name=True)
        except Exception:  # pragma: no cover - protobuf versions vary
            return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return str(value)


def _find_ui_element_index(
    ui_elements: list[Any],
    *,
    resource_name: str | None = None,
    text: str | None = None,
    content_description: str | None = None,
) -> int:
    """Resolve one current native UI element without using coordinates."""
    selectors = {
        "resource_name": resource_name,
        "text": text,
        "content_description": content_description,
    }
    selectors = {key: value for key, value in selectors.items() if value is not None}
    if len(selectors) != 1:
        raise ValueError(
            "exactly one of resource_name, text, or content_description is required"
        )

    key, expected = next(iter(selectors.items()))
    matches = []
    for index, element in enumerate(ui_elements):
        actual = (
            element.get(key)
            if isinstance(element, dict)
            else getattr(element, key, None)
        )
        if actual == expected:
            matches.append(index)
    if not matches:
        raise ValueError(f"no current UI element matched {key}={expected!r}")
    if len(matches) > 1:
        raise ValueError(
            f"multiple current UI elements matched {key}={expected!r}: {matches}"
        )
    return matches[0]


class ManualAndroidWorld:
    def __init__(self, args: argparse.Namespace) -> None:
        if args.android_world_root:
            android_world_root = Path(args.android_world_root).resolve()
            # Accept either the release root or its ``android_world`` package
            # directory; Python needs the parent directory on sys.path.
            import_root = (
                android_world_root.parent
                if android_world_root.name == "android_world"
                else android_world_root
            )
            sys.path.insert(0, str(import_root))
        from android_world import registry
        from android_world.env import android_world_controller
        from android_world.env import env_launcher
        from android_world.env.setup_device import setup as android_world_setup
        from src.integrations.android_world.apps import resolve_androidworld_app_name
        from src.integrations.android_world.run_episode import (
            _patch_androidworld_adb_output_sanitizer,
            _rehydrate_task_params,
        )
        task_types = registry.TaskRegistry().get_registry(family="android_world")
        if args.task not in task_types:
            raise ValueError(f"unknown AndroidWorld task: {args.task}")
        task_type = task_types[args.task]

        self._json_action = __import__(
            "android_world.env.json_action", fromlist=["JSONAction"]
        )
        self._env = env_launcher.load_and_setup_env(
            console_port=args.console_port,
            emulator_setup=False,
            adb_path=args.adb_path,
            grpc_port=args.grpc_port or args.console_port + 3000,
            install_a11y_forwarding_app=args.install_a11y_forwarder,
        )
        # Keep AndroidWorld's official controller/action path, but use its
        # native UIAutomator observation backend when the forwarder's optional
        # host gRPC tree is unavailable on a fresh local AVD.
        self._env.controller._a11y_method = (  # pylint: disable=protected-access
            android_world_controller.A11yMethod.UIAUTOMATOR
        )
        if args.emulator_setup:
            _patch_androidworld_adb_output_sanitizer(self._env.controller)
            app_names = {
                str(name).strip().lower()
                for name in getattr(task_type, "app_names", ())
            }
            app_list = tuple(
                app
                for app in android_world_setup._APPS
                if str(app.app_name).strip().lower() in app_names
            )
            android_world_setup.setup_apps(self._env, app_list=app_list or None)
        task_type.set_device_time(self._env)
        params = json.loads(args.task_params_json or "{}")
        params = _rehydrate_task_params(params=params, task_type=task_type)
        params.setdefault("seed", args.seed)
        self._task = task_type(params)
        self._task.initialize_task(self._env)
        self._resolve_androidworld_app_name = resolve_androidworld_app_name
        self._root = Path(args.output).expanduser().resolve()
        self._images = self._root / "observations" / "objects"
        self._images.mkdir(parents=True, exist_ok=True)
        self._run_id = f"manual_{uuid.uuid4().hex}"
        self._started_ms = int(time.time() * 1000)
        self._steps: list[dict[str, Any]] = []
        self._last_observation: dict[str, Any] | None = None
        self._finished = False
        self._write_run_log(status="running", success=False, reward=0.0)

    @property
    def goal(self) -> str:
        return str(self._task.goal)

    def _observation(self, state: Any, index: int) -> dict[str, Any]:
        from PIL import Image

        image = Image.fromarray(state.pixels)
        raw_path = self._root / "observations" / f"{index:04d}.png"
        image.save(raw_path, format="PNG")
        data = raw_path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        object_path = self._images / f"{digest}.png"
        if not object_path.exists():
            object_path.write_bytes(data)
        raw_path.unlink(missing_ok=True)
        return {
            "pixels": {
                "path": str(object_path),
                "sha256": digest,
                "width": int(state.pixels.shape[1]),
                "height": int(state.pixels.shape[0]),
                "mime_type": "image/png",
            },
            "forest": _jsonable(state.forest),
            "ui_elements": _jsonable(state.ui_elements),
            "auxiliaries": _jsonable(state.auxiliaries or {}),
        }

    def observe(self, *, stable: bool = True) -> dict[str, Any]:
        state = self._env.get_state(wait_to_stabilize=stable)
        self._last_observation = self._observation(state, len(self._steps))
        return {
            "ok": True,
            "goal": self.goal,
            "step_index": len(self._steps),
            "observation": self._last_observation,
        }

    def click_target(self, selector: dict[str, Any]) -> dict[str, Any]:
        """Click one element resolved from the latest native observation."""
        # Refresh the native state immediately before resolving the selector so
        # a delayed manual decision cannot click an index from an old screen.
        self.observe()
        index = _find_ui_element_index(
            self._last_observation["ui_elements"],
            resource_name=selector.get("resource_name"),
            text=selector.get("text"),
            content_description=selector.get("content_description"),
        )
        return self.act({"action_type": "click", "index": index})

    def act(self, action_payload: dict[str, Any]) -> dict[str, Any]:
        if self._last_observation is None:
            self.observe()
        action_record: dict[str, Any]
        if action_payload.get("action_type") == "wait" and "duration" in action_payload:
            # JSONAction's public wait action has no duration field.  Keep the
            # native action contract and make the interactive protocol tolerant
            # of a human-friendly duration by sleeping after the official wait.
            duration = float(action_payload["duration"])
            action = self._json_action.JSONAction(action_type="wait")
            action_record = dict(action_payload)
        elif action_payload.get("action_type") == "swipe_xy":
            # Coordinate-bounded swipe through AndroidWorld's own ADB
            # actuation helper.  This is needed for canvas widgets: the
            # release's JSONAction swipe is full-screen and cannot reach the
            # widget without triggering system navigation.
            from android_world.env import adb_utils

            start = action_payload.get("start_xy")
            end = action_payload.get("end_xy")
            if not (isinstance(start, (list, tuple)) and len(start) == 2
                    and isinstance(end, (list, tuple)) and len(end) == 2):
                raise ValueError("swipe_xy requires start_xy and end_xy")
            duration_ms = int(action_payload.get("duration_ms", 500))
            command = adb_utils.generate_swipe_command(
                int(start[0]), int(start[1]), int(end[0]), int(end[1]), duration_ms
            )
            adb_utils.issue_generic_request(command, self._env.controller)
            action = None
            action_record = dict(action_payload)
        elif action_payload.get("action_type") == "drag_and_drop":
            # AndroidWorld's actuation layer already implements drag-and-drop, but
            # this release's JSONAction parser omits that action from its public
            # enum.  Keep the interactive protocol explicit and construct the
            # official action object through the same env.execute_action seam.
            touch = action_payload.get("touch_xy")
            lift = action_payload.get("lift_xy")
            if not (isinstance(touch, (list, tuple)) and len(touch) == 2
                    and isinstance(lift, (list, tuple)) and len(lift) == 2):
                raise ValueError("drag_and_drop requires touch_xy and lift_xy")
            action = self._json_action.JSONAction(action_type="click")
            action.action_type = "drag_and_drop"
            action.touch_xy = tuple(int(v) for v in touch)
            action.lift_xy = tuple(int(v) for v in lift)
        else:
            action = self._json_action.JSONAction(**action_payload)
            action_record = action.as_dict()
        before = self._last_observation
        if action is not None:
            execution_action = action
            if action_payload.get("action_type") == "open_app":
                package_name = str(action_payload.get("app_name") or "").strip()
                resolved_name = self._resolve_androidworld_app_name(
                    package_name,
                    self._env.controller,
                )
                execution_action = self._json_action.JSONAction(
                    action_type="open_app",
                    app_name=resolved_name,
                )
            self._env.execute_action(execution_action)
        if action_payload.get("action_type") == "wait" and "duration" in action_payload:
            time.sleep(max(0.0, duration))
        after = self.observe()["observation"]
        self._steps.append(
            {
                "step_index": len(self._steps),
                "observation": before,
                "action": action_record,
                "result": {"success": True},
                "next_observation": after,
                "metadata": {"decision": "manual_codex", "source": "native_androidworld"},
            }
        )
        self._write_run_log(status="running", success=False, reward=0.0)
        return {"ok": True, "action": action_record, "observation": after}

    def validate(self) -> dict[str, Any]:
        reward = float(self._task.is_successful(self._env))
        success = reward > 0.5
        self._finished = True
        self._write_run_log(status="succeeded" if success else "failed", success=success, reward=reward)
        return {"ok": True, "success": success, "reward": reward, "run_log": str(self._root / "run_log.json")}

    def _write_run_log(self, *, status: str, success: bool, reward: float) -> None:
        payload = {
            "schema_version": "omniflow.run_log.v1",
            "run_id": self._run_id,
            "task_name": self._task.name,
            "goal": self.goal,
            "task_parameters": _jsonable(self._task.params),
            "seed": self._task.params.get("seed"),
            "status": status,
            "success": success,
            "validator": {"official": True, "success": success, "reward": reward},
            "provenance": {"kind": "manual_native_androidworld"},
            "started_at_ms": self._started_ms,
            "finished_at_ms": int(time.time() * 1000),
            "steps": self._steps,
            "final_observation": self._last_observation,
            "diagnostics": {"model_calls": 0, "decision_owner": "codex_manual"},
        }
        self._root.mkdir(parents=True, exist_ok=True)
        (self._root / "run_log.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def close(self) -> None:
        try:
            if not self._finished:
                self._write_run_log(status="aborted", success=False, reward=0.0)
            self._task.tear_down(self._env)
        finally:
            close = getattr(self._env, "close", None)
            if callable(close):
                close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--android-world-root", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-params-json", default="{}")
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--console-port", type=int, default=5560)
    parser.add_argument("--grpc-port", type=int, default=0)
    parser.add_argument("--adb-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--install-a11y-forwarder",
        action="store_true",
        help="Install the official forwarder APK during startup when absent.",
    )
    parser.add_argument(
        "--emulator-setup",
        action="store_true",
        help="Run AndroidWorld's official emulator setup before task initialization.",
    )
    args = parser.parse_args()
    harness = ManualAndroidWorld(args)
    try:
        print(json.dumps({"ready": True, "goal": harness.goal}), flush=True)
        for line in sys.stdin:
            if not line.strip():
                continue
            command = json.loads(line)
            kind = command.get("cmd")
            if kind == "observe":
                result = harness.observe(stable=bool(command.get("stable", True)))
            elif kind == "click":
                result = harness.click_target(dict(command.get("target", command)))
            elif kind == "act":
                result = harness.act(dict(command["action"]))
            elif kind == "validate":
                result = harness.validate()
            elif kind == "quit":
                result = {"ok": True}
                print(json.dumps(result, ensure_ascii=False), flush=True)
                break
            else:
                result = {"ok": False, "error": f"unknown command: {kind}"}
            print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        harness.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
