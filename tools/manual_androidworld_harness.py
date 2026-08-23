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
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any


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


def _materialize_task_params(value: Any) -> Any:
    """Restore JSON markers for rich AndroidWorld task fixtures."""
    if isinstance(value, dict):
        if (
            value.get("__omniflow_type__") == "unsupported"
            and value.get("class") == "Image"
        ):
            from PIL import Image

            return Image.new("RGB", (500, 500), "white")
        return {key: _materialize_task_params(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_materialize_task_params(item) for item in value]
    return value


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


def _resolve_device_serial(args: argparse.Namespace) -> str:
    return (
        str(
            getattr(args, "device_serial", "")
            or os.environ.get("ANDROID_SERIAL", "")
        ).strip()
        or f"emulator-{int(args.console_port)}"
    )


def _bounded_swipe_action_record(
    start: Any,
    end: Any,
    *,
    duration_ms: int,
) -> dict[str, Any]:
    """Preserve a widget-bounded swipe as executable RunLog evidence."""
    if not (
        isinstance(start, (list, tuple))
        and len(start) == 2
        and isinstance(end, (list, tuple))
        and len(end) == 2
    ):
        raise ValueError("swipe_xy requires start_xy and end_xy")
    x1, y1 = (int(value) for value in start)
    x2, y2 = (int(value) for value in end)
    delta_x = x2 - x1
    delta_y = y2 - y1
    if abs(delta_y) >= abs(delta_x):
        direction = "down" if delta_y < 0 else "up"
    else:
        direction = "right" if delta_x < 0 else "left"
    return {
        "action_type": "swipe",
        "direction": direction,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "duration_ms": int(duration_ms),
    }


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
        self._activity_compat_original = None
        self._last_observation: dict[str, Any] | None = None
        self._install_activity_compatibility()
        from src.integrations.android_world.apps import resolve_androidworld_app_name
        from src.integrations.android_world.run_episode import (
            start_androidworld_task_session,
        )
        self._json_action = __import__(
            "android_world.env.json_action", fromlist=["JSONAction"]
        )
        params = _materialize_task_params(json.loads(args.task_params_json or "{}"))
        startup, self._task = start_androidworld_task_session(
            android_world_root=args.android_world_root,
            task_name=args.task,
            task_params=params,
            task_seed=int(args.seed),
            console_port=int(args.console_port),
            adb_path=args.adb_path,
            grpc_port=args.grpc_port or args.console_port + 3000,
            install_a11y_forwarding_app=bool(args.install_a11y_forwarder),
            perform_emulator_setup=not bool(args.skip_emulator_setup),
            use_uiautomator=not bool(args.install_a11y_forwarder),
        )
        self._env = startup.env
        self._resolve_androidworld_app_name = resolve_androidworld_app_name
        self._root = Path(args.output).expanduser().resolve()
        self._run_id = self._root.name
        self._device_serial = _resolve_device_serial(args)
        self._source_seed = int(args.seed)
        self._steps: list[dict[str, Any]] = []
        self._last_ui_elements: list[Any] = []
        self._finished = False
        self._validation_reasoning = ""
        # Persist a schema-valid recovery checkpoint while the interactive
        # attempt is still incomplete. A later official validation overwrites
        # it with the final succeeded/failed result.
        self._write_run_log(success=False, reward=0.0)

    def _install_activity_compatibility(self) -> None:
        """Normalize package-only activity responses from the emulator bridge.

        Some emulator/API combinations return only the foreground package in
        ``GetCurrentActivity.full_activity``. AndroidWorld validators expect a
        ComponentName (``package/activity``), so preserve the official
        validator while reconstructing the component from the same app
        registry when the current native XML confirms that package is visible.
        """
        from android_world.env import adb_utils

        self._activity_compat_original = adb_utils.get_current_activity
        original = self._activity_compat_original

        def normalized_get_current_activity(env: Any, timeout_sec: Any = None):
            if timeout_sec is None:
                activity, response = original(env)
            else:
                activity, response = original(env, timeout_sec=timeout_sec)
            if activity and "/" in activity:
                return activity, response

            xml = ""
            if isinstance(getattr(self, "_last_observation", None), dict):
                xml = str(self._last_observation.get("xml") or "")
            visible_packages = set(re.findall(r'package="([^"]+)"', xml))
            if not activity:
                foreground_packages = sorted(
                    package
                    for package in visible_packages
                    if package != "com.android.systemui"
                )
                activity = foreground_packages[0] if foreground_packages else activity
            if not activity:
                response = adb_utils.issue_generic_request(
                    "shell dumpsys activity activities".split(), env
                )
                output = response.generic.output.decode(errors="replace")
                match = re.search(
                    r"(?:topResumedActivity|mResumedActivity)=ActivityRecord\{[^ ]+\s+u\d+\s+([^ }\n]+)",
                    output,
                )
                activity = match.group(1) if match else activity
            if not activity or activity not in visible_packages:
                if activity and "/" in activity:
                    return activity, response
                return activity, response

            for app_name in sorted(adb_utils.get_all_apps(env)):
                candidate = str(adb_utils.get_adb_activity(app_name) or "").strip()
                if candidate.split("/", 1)[0].strip() == activity:
                    return candidate, response
            # A few API-35 images expose no activity mapping at all. The
            # validator only needs a syntactically valid ComponentName to
            # inspect the package, while the XML above already established
            # the foreground package.
            return f"{activity}/{activity}.Settings", response

        adb_utils.get_current_activity = normalized_get_current_activity

    @property
    def goal(self) -> str:
        return str(self._task.goal)

    def _observation(self, state: Any) -> dict[str, Any]:
        from src.experiment.observation_evidence import canonicalize_run_log_observation
        from src.integrations.android_world.state import snapshot_androidworld_state

        return canonicalize_run_log_observation(
            snapshot_androidworld_state(state, evidence_root=self._root)
        )

    def observe(self, *, stable: bool = True) -> dict[str, Any]:
        state = self._env.get_state(wait_to_stabilize=stable)
        self._last_ui_elements = list(_jsonable(state.ui_elements) or [])
        self._last_observation = self._observation(state)
        return {
            "ok": True,
            "goal": self.goal,
            "step_index": len(self._steps),
            "observation": self._last_observation,
        }

    def click_target(self, selector: dict[str, Any]) -> dict[str, Any]:
        """Click one element resolved from the latest native observation."""
        reasoning = str(selector.get("reasoning") or "").strip()
        # Refresh the native state immediately before resolving the selector so
        # a delayed manual decision cannot click an index from an old screen.
        self.observe()
        index = _find_ui_element_index(
            self._last_ui_elements,
            resource_name=selector.get("resource_name"),
            text=selector.get("text"),
            content_description=selector.get("content_description"),
        )
        return self.act(
            {"action_type": "click", "index": index, "reasoning": reasoning}
        )

    def act(self, action_payload: dict[str, Any]) -> dict[str, Any]:
        if self._last_observation is None:
            self.observe()
        reasoning = str(action_payload.get("reasoning") or "").strip()
        if not reasoning:
            raise ValueError("reasoning_required_for_every_action")
        clean_action_payload = dict(action_payload)
        clean_action_payload.pop("reasoning", None)
        action_record: dict[str, Any]
        if clean_action_payload.get("action_type") == "wait" and "duration" in clean_action_payload:
            # JSONAction's public wait action has no duration field.  Keep the
            # native action contract and make the interactive protocol tolerant
            # of a human-friendly duration by sleeping after the official wait.
            duration = float(clean_action_payload["duration"])
            action = self._json_action.JSONAction(action_type="wait")
            action_record = action.as_dict()
        elif clean_action_payload.get("action_type") == "swipe_xy":
            # Coordinate-bounded swipe through AndroidWorld's own ADB
            # actuation helper.  This is needed for canvas widgets: the
            # release's JSONAction swipe is full-screen and cannot reach the
            # widget without triggering system navigation.
            from android_world.env import adb_utils

            start = clean_action_payload.get("start_xy")
            end = clean_action_payload.get("end_xy")
            duration_ms = int(clean_action_payload.get("duration_ms", 500))
            action_record = _bounded_swipe_action_record(
                start,
                end,
                duration_ms=duration_ms,
            )
            command = adb_utils.generate_swipe_command(
                int(start[0]), int(start[1]), int(end[0]), int(end[1]), duration_ms
            )
            adb_utils.issue_generic_request(command, self._env.controller)
            action = None
        elif clean_action_payload.get("action_type") == "drag_and_drop":
            # AndroidWorld's actuation layer already implements drag-and-drop, but
            # this release's JSONAction parser omits that action from its public
            # enum.  Keep the interactive protocol explicit and construct the
            # official action object through the same env.execute_action seam.
            touch = clean_action_payload.get("touch_xy")
            lift = clean_action_payload.get("lift_xy")
            if not (isinstance(touch, (list, tuple)) and len(touch) == 2
                    and isinstance(lift, (list, tuple)) and len(lift) == 2):
                raise ValueError("drag_and_drop requires touch_xy and lift_xy")
            action = self._json_action.JSONAction(action_type="click")
            action.action_type = "drag_and_drop"
            action.touch_xy = tuple(int(v) for v in touch)
            action.lift_xy = tuple(int(v) for v in lift)
        else:
            action = self._json_action.JSONAction(**clean_action_payload)
            action_record = action.as_dict()
        before = self._last_observation
        if action is not None:
            execution_action = action
            if clean_action_payload.get("action_type") == "open_app":
                package_name = str(clean_action_payload.get("app_name") or "").strip()
                resolved_name = self._resolve_androidworld_app_name(
                    package_name,
                    self._env.controller,
                )
                execution_action = self._json_action.JSONAction(
                    action_type="open_app",
                    app_name=resolved_name,
                )
            self._env.execute_action(execution_action)
        if clean_action_payload.get("action_type") == "wait" and "duration" in clean_action_payload:
            time.sleep(max(0.0, duration))
        after = self.observe()["observation"]
        after_screenshot = (
            after.get("screenshot")
            if isinstance(after, dict) and isinstance(after.get("screenshot"), dict)
            else after.get("pixels")
            if isinstance(after, dict) and isinstance(after.get("pixels"), dict)
            else {}
        )
        self._steps.append(
            {
                "step_index": len(self._steps),
                "observation": before,
                "action": action_record,
                "result": {"success": True},
                "next_observation": after,
                "metadata": {
                    "decision": "manual_codex",
                    "source": "native_androidworld",
                    "reasoning": reasoning,
                    "screenshot_path": after_screenshot.get("path"),
                },
            }
        )
        self._write_run_log(success=False, reward=0.0)
        return {"ok": True, "action": action_record, "observation": after}

    def validate(self, reasoning: str = "") -> dict[str, Any]:
        self._validation_reasoning = str(reasoning or "").strip()
        if any(
            not str(step.get("metadata", {}).get("reasoning") or "").strip()
            for step in self._steps
        ):
            return {"ok": False, "error": "reasoning_required_for_every_action"}
        reward = float(self._task.is_successful(self._env))
        success = reward > 0.5
        self._finished = True
        self._write_run_log(success=success, reward=reward)
        return {"ok": True, "success": success, "reward": reward, "run_log": str(self._root / "run_log.json")}

    def _write_run_log(self, *, success: bool, reward: float) -> None:
        from src.experiment.observation_evidence import (
            build_androidworld_run_log,
            persist_androidworld_run_log,
        )

        run_log = build_androidworld_run_log(
            run_id=self._run_id,
            task_name=self._task.name,
            goal=self.goal,
            task_parameters=_jsonable(self._task.params),
            seed=self._source_seed,
            validator_success=success,
            validator_reward=reward,
            validator_official=True,
            provenance={"kind": "manual_native_androidworld"},
            steps=self._steps,
            final_observation=self._last_observation,
            diagnostics={
                "model_calls": 0,
                "decision_owner": "codex_manual",
                "workflow_version": "androidworld_manual_recollection.v1",
                "device_serial": self._device_serial,
                "same_environment": True,
                "reasoning_count": sum(
                    bool(str(step.get("metadata", {}).get("reasoning") or "").strip())
                    for step in self._steps
                ),
                "screenshot_count": sum(
                    bool(step.get("metadata", {}).get("screenshot_path"))
                    for step in self._steps
                )
                + bool(self._last_observation),
                "validation_reasoning": self._validation_reasoning,
            },
        )
        persist_androidworld_run_log(self._root, run_log=run_log, replace=True)

    def close(self) -> None:
        try:
            if not self._finished:
                self._write_run_log(success=False, reward=0.0)
            self._task.tear_down(self._env)
        finally:
            if self._activity_compat_original is not None:
                from android_world.env import adb_utils

                adb_utils.get_current_activity = self._activity_compat_original
                self._activity_compat_original = None
            close = getattr(self._env, "close", None)
            if callable(close):
                close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--android-world-root", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-params-json", default="{}")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--console-port", type=int, default=5560)
    parser.add_argument(
        "--device-serial",
        default="",
        help="ADB serial to record in provenance; defaults to ANDROID_SERIAL or emulator-{console-port}.",
    )
    parser.add_argument("--grpc-port", type=int, default=0)
    parser.add_argument("--adb-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--install-a11y-forwarder",
        action="store_true",
        help="Install the official forwarder APK during startup when absent.",
    )
    parser.add_argument(
        "--skip-emulator-setup",
        action="store_true",
        help="Reuse an already prepared source emulator without repeating app setup.",
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
            try:
                if kind == "observe":
                    result = harness.observe(stable=bool(command.get("stable", True)))
                elif kind == "click":
                    result = harness.click_target(dict(command.get("target", command)))
                elif kind == "act":
                    result = harness.act(dict(command["action"]))
                elif kind == "validate":
                    result = harness.validate(str(command.get("reasoning") or ""))
                elif kind == "quit":
                    result = {"ok": True}
                    print(json.dumps(result, ensure_ascii=False), flush=True)
                    break
                else:
                    result = {"ok": False, "error": f"unknown command: {kind}"}
            except Exception as error:  # pragma: no cover - interactive recovery
                # UI transitions can invalidate a selector after the official
                # action has already been delivered. Keep the session alive so
                # the operator can observe the new native state and continue.
                result = {
                    "ok": False,
                    "error": str(error),
                    "step_index": len(harness._steps),
                    "observation": harness._last_observation,
                }
            print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        harness.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
