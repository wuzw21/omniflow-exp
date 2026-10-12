"""Small boundary for launching the pinned external baselines.

This module makes the official MobileGPT and AppAgent checkouts look like
their upstream launchers expect.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterator, Sequence

from src.integrations.android_world.oob_control import (
    CONTROL_ACCESSIBILITY_SERVICE as OOB_CONTROL_ACCESSIBILITY_SERVICE,
)
from src.experiment.protocol import (
    APPAGENT_MODEL,
    FORMAL_MODEL_BASE_URL,
    FORMAL_THINKING,
    MAX_STEPS,
    TASK_DEADLINE_SEC,
    TASK_SEED,
    require_formal_model,
)


_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

# Count MobileGPT chat requests across the whole official-forward process.
# Keeping this at module scope is essential: ``query`` may be called once per
# planner step, and a function-local counter would reset on every call and
# silently defeat MOBILEGPT_MAX_CHAT_CALLS.


def _ensure_androidworld_ffmpeg() -> None:
    """Expose the pinned imageio-ffmpeg binary to AndroidWorld/pydub.

    AndroidWorld creates MP3 fixtures during task initialization.  pydub
    resolves its encoder exclusively through ``PATH``; on 9207 no system
    ffmpeg is installed, and the child environment can replace the launcher
    PATH.  Install a process-local shim in /tmp so fixture generation is
    deterministic without modifying the repository or emulator image.
    """

    try:
        import imageio_ffmpeg

        binary = Path(imageio_ffmpeg.get_ffmpeg_exe()).expanduser().resolve()
    except Exception:
        return
    if not binary.is_file():
        return
    shim_dir = Path(tempfile.gettempdir()) / "omniflow-ffmpeg-bin"
    try:
        shim_dir.mkdir(parents=True, exist_ok=True)
        shim = shim_dir / "ffmpeg"
        if shim.is_symlink() or shim.exists():
            if shim.resolve() != binary:
                shim.unlink()
        if not shim.exists():
            shim.symlink_to(binary)
        os.environ["PATH"] = os.pathsep.join(
            (str(shim_dir), str(binary.parent), os.environ.get("PATH", ""))
        )
        os.environ.setdefault("FFMPEG_BINARY", str(binary))
    except OSError:
        return


def _json_safe(value: Any) -> Any:
    """Return a JSON-compatible copy for result/provenance rows.

    AndroidWorld task parameters can contain lightweight dataclass/model
    objects (for example ``Recipe`` instances).  Those must not prevent the
    final raw result row from being written after an otherwise valid episode.
    Preserve object attributes when available and fall back to a string for
    opaque values.
    """

    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                default=lambda obj: vars(obj)
                if hasattr(obj, "__dict__")
                else str(obj),
            )
        )
    except (TypeError, ValueError):
        return str(value)


def _mobilegpt_accessibility_service_bound(
    dumpsys: str,
    service: str,
) -> bool:
    bound_match = re.search(
        r"(?ms)^\s*Bound services:(.*?)"
        r"(?=^\s*(?:Enabled services|Binding services|Crashed services|"
        r"Client list info):|\Z)",
        str(dumpsys or ""),
    )
    if bound_match is None:
        return False
    package, separator, class_name = str(service or "").partition("/")
    if not separator or not package or not class_name:
        return False
    full_class_name = (
        f"{package}{class_name}" if class_name.startswith(".") else class_name
    )
    identifiers = {
        service,
        f"{package}/{full_class_name}",
        full_class_name,
        "MobileGPT Accessibility",
    }
    bound_services = bound_match.group(1)
    return any(identifier in bound_services for identifier in identifiers)


def _count_appagent_actions(log_text: str) -> int:
    """Count official AppAgent device actions, excluding its FINISH marker."""

    lines = _ANSI_ESCAPE_RE.sub("", str(log_text or "")).splitlines()
    actions = 0
    awaiting_action = False
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line == "Action:":
            awaiting_action = True
            continue
        if not awaiting_action:
            continue
        if line != "FINISH":
            actions += 1
        awaiting_action = False
    return actions


def _load_appagent_stats(path: str | Path) -> dict[str, Any]:
    """Aggregate model telemetry emitted by the disposable AppAgent wrapper."""

    stats_path = Path(path).expanduser()
    calls = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    empty_responses = 0
    errors = 0
    if not stats_path.is_file():
        return {
            "path": str(stats_path),
            "model_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "empty_responses": 0,
            "errors": 0,
            "status": "unavailable",
        }
    for line in stats_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("event") == "model_call":
            calls += 1
            prompt_tokens += int(event.get("prompt_tokens") or 0)
            completion_tokens += int(event.get("completion_tokens") or 0)
            total_tokens += int(event.get("total_tokens") or 0)
        elif event.get("event") == "model_empty_response":
            empty_responses += 1
        elif event.get("event") == "model_error":
            errors += 1
    return {
        "path": str(stats_path),
        "model_calls": calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "empty_responses": empty_responses,
        "errors": errors,
        "status": "tracked",
    }


def _mobilegpt_stats_summary(path: str | Path) -> dict[str, Any]:
    """Read the shared MobileGPT telemetry without duplicating its schema."""

    stats_path = Path(path).expanduser()
    if not str(path).strip() or not stats_path.is_file():
        return {
            "model_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "token_usage_status": "unavailable",
            "memory_lookup_count": 0,
            "memory_hit_count": 0,
            "memory_hit_rate": 0.0,
            "fallback_count": 0,
            "action_sent_count": 0,
            "actions_executed": 0,
        }
    from src.integrations.mobilegpt_memory import summarize_mobilegpt_stats

    return summarize_mobilegpt_stats(stats_path)


@contextmanager
def _androidworld_task_startup(
    *,
    android_world_root: str | Path,
    task_name: str,
    task_params_json: str,
    task_seed: int,
    serial: str,
    console_port: int,
    grpc_port: int,
    adb_path: str,
    perform_emulator_setup: bool,
    use_uiautomator: bool = True,
    output_root: str | Path | None = None,
    method: str = "external",
) -> Iterator[tuple[Any, Any]]:
    """Prepare one official task through the canonical AndroidWorld seam."""

    _ensure_androidworld_ffmpeg()
    from android_world.env import adb_utils

    from src.integrations.android_world.run_episode import (
        _patch_androidworld_current_activity,
        start_androidworld_task_session,
    )

    decoded = json.loads(str(task_params_json or "{}"))
    if not isinstance(decoded, dict):
        raise ValueError("androidworld_task_params_must_be_object")
    try:
        startup, task = start_androidworld_task_session(
            android_world_root=android_world_root,
            task_name=task_name,
            task_params=decoded or None,
            task_seed=int(task_seed),
            console_port=int(console_port),
            adb_path=adb_path,
            grpc_port=int(grpc_port),
            perform_emulator_setup=bool(perform_emulator_setup),
            use_uiautomator=bool(use_uiautomator),
        )
    except Exception as exc:
        if output_root is not None:
            _write_external_startup_failure(
                output_root=output_root,
                method=method,
                serial=serial,
                task_name=task_name,
                task_seed=task_seed,
                task_params=decoded,
                error=exc,
            )
        raise
    original_current_activity = _patch_androidworld_current_activity(adb_utils)
    try:
        yield startup.env, task
    finally:
        try:
            task.tear_down(startup.env)
        finally:
            try:
                adb_utils.get_current_activity = original_current_activity
            finally:
                close = getattr(startup.env, "close", None)
                if callable(close):
                    close()


def _select_androidworld_validator_observer(env: Any) -> None:
    """Select the official read-only observer for final validation.

    External baselines own observation/action during the episode.  In OOB
    mode the AndroidWorld environment is deliberately created without its
    Accessibility wrapper, so calling a UI validator while the controller is
    still in ``NONE``/forwarder mode raises ``Must use A11yGrpcWrapper``.
    UIAutomator is used only for the official validator read-back; it is never
    used to choose or execute a method action.
    """

    controller = getattr(env, "controller", None)
    if controller is None:
        controller = env if hasattr(env, "_a11y_method") else None
    if controller is None:
        return
    try:
        from android_world.env import android_world_controller

        controller._a11y_method = (  # pylint: disable=protected-access
            android_world_controller.A11yMethod.UIAUTOMATOR
        )
    except Exception:
        # Keep the original validator exception visible if the pinned
        # AndroidWorld revision does not expose this enum.
        return


def _forest_from_ui_elements(ui_elements: Any) -> Any:
    """Build the smallest official-API forest equivalent to XML UI elements.

    The pinned AndroidWorld Contacts validators still call
    ``forest_to_ui_elements(state.forest)`` even when the selected observer is
    UIAutomator.  UIAutomator intentionally exposes ``ui_elements`` and no
    raw forest.  This adapter is used only during read-only validation; action
    selection and execution never consume it.
    """

    from android_env.proto.a11y import android_accessibility_forest_pb2

    forest = android_accessibility_forest_pb2.AndroidAccessibilityForest()
    window = forest.windows.add()
    window.id = 0
    window.layer = 0
    for index, element in enumerate(ui_elements or ()):
        node = window.tree.nodes.add()
        node.unique_id = index
        bbox = getattr(element, "bbox_pixels", None)
        if bbox is not None:
            node.bounds_in_screen.left = int(bbox.x_min)
            node.bounds_in_screen.right = int(bbox.x_max)
            node.bounds_in_screen.top = int(bbox.y_min)
            node.bounds_in_screen.bottom = int(bbox.y_max)
        for field in (
            "text",
            "content_description",
            "class_name",
            "hint_text",
            "package_name",
            "resource_id",
        ):
            value = getattr(element, field, None)
            if value:
                setattr(
                    node,
                    "view_id_resource_name" if field == "resource_id" else field,
                    str(value),
                )
        for field in (
            "is_checkable",
            "is_checked",
            "is_clickable",
            "is_editable",
            "is_enabled",
            "is_focusable",
            "is_focused",
            "is_long_clickable",
            "is_scrollable",
            "is_selected",
            "is_visible",
        ):
            value = getattr(element, field, None)
            if value is not None:
                setattr(
                    node,
                    "is_visible_to_user" if field == "is_visible" else field,
                    bool(value),
                )
    return forest


def _safe_official_validator(task: Any, env: Any) -> tuple[bool, float, str]:
    """Run the official validator without losing the episode evidence.

    AndroidWorld validators are the authoritative judge, but a malformed or
    unavailable observation must be classified as an environment failure
    rather than aborting the external-method result writer.  A normal
    validator rejection remains a covered method failure.
    """

    previous_backend = os.environ.get("OMNIFLOW_OBSERVE_BACKEND")
    original_instance_get_state = getattr(env, "get_state", None)
    original_class_get_state = getattr(type(env), "get_state", None)
    restore_get_state: Any = None
    os.environ["OMNIFLOW_OBSERVE_BACKEND"] = "androidworld"
    if callable(original_instance_get_state):
        def validator_get_state(*args: Any, **kwargs: Any) -> Any:
            state = original_instance_get_state(*args, **kwargs)
            if getattr(state, "forest", None) is None:
                elements = getattr(state, "ui_elements", None)
                forest = _forest_from_ui_elements(elements)
                try:
                    return dataclasses.replace(state, forest=forest)
                except TypeError:
                    state.forest = forest
            return state

        try:
            env.get_state = validator_get_state
            restore_get_state = lambda: setattr(
                env, "get_state", original_instance_get_state
            )
        except (AttributeError, TypeError):
            if callable(original_class_get_state):
                def validator_class_get_state(
                    current_env: Any, *args: Any, **kwargs: Any
                ) -> Any:
                    state = original_class_get_state(current_env, *args, **kwargs)
                    if getattr(state, "forest", None) is None:
                        elements = getattr(state, "ui_elements", None)
                        forest = _forest_from_ui_elements(elements)
                        try:
                            return dataclasses.replace(state, forest=forest)
                        except TypeError:
                            state.forest = forest
                    return state

                setattr(type(env), "get_state", validator_class_get_state)
                restore_get_state = lambda: setattr(
                    type(env), "get_state", original_class_get_state
                )
    try:
        return True, float(task.is_successful(env)), ""
    except Exception as exc:  # pragma: no cover - concrete AW task varies
        message = str(exc).strip().replace("\n", " ")
        if len(message) > 500:
            message = message[:500] + "..."
        return (
            False,
            0.0,
            f"official_validator_exception:{type(exc).__name__}:{message}",
        )
    finally:
        if restore_get_state is not None:
            try:
                restore_get_state()
            except (AttributeError, TypeError):
                pass
        if previous_backend is None:
            os.environ.pop("OMNIFLOW_OBSERVE_BACKEND", None)
        else:
            os.environ["OMNIFLOW_OBSERVE_BACKEND"] = previous_backend


def _write_external_startup_failure(
    *,
    output_root: str | Path,
    method: str,
    serial: str,
    task_name: str,
    task_seed: int,
    task_params: dict[str, Any],
    error: Exception,
) -> None:
    """Persist a truthful result when official task startup fails before yield."""

    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    reason = str(error).strip().replace("\n", " ")
    if len(reason) > 500:
        reason = reason[:500] + "..."
    row = {
        "schema_version": "omniflow.androidworld.result.v1",
        "task_name": task_name,
        "task": task_name,
        "method": method,
        "device": serial,
        "task_random_seed": int(task_seed),
        "fixed_task_seed": True,
        "fixed_task_params": True,
        "official_validator_used": False,
        "official_validator_success": False,
        "official_validator_coverage_rate": 0.0,
        "androidworld_validator_result": {
            "validator": "androidworld_official",
            "success": False,
            "reward": 0.0,
            "covered": False,
        },
        "classification": "environment_failure",
        "process_returncode": 1,
        "actions_executed": 0,
        "planner_steps": 0,
        "model_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "token_usage_status": "unavailable",
        "fallback_steps": 0,
        "task_params": _json_safe(task_params),
        "runtime_integrity_error": reason,
        "failure_reason": f"official_task_startup_exception:{type(error).__name__}:{reason}",
        "environment_failure": True,
    }
    (output / "task_results.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _link_or_fail(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"official_forward_source_missing:{source}")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"official_forward_target_exists:{target}")
    target.symlink_to(source, target_is_directory=source.is_dir())


_APPAGENT_DOC_LOOKUP_MARKER = "_omniflow_resolution_agnostic_doc_path"


def _stage_appagent_scripts(source: Path, target: Path) -> None:
    """Copy AppAgent scripts and remove source-resolution from doc lookup."""

    if not source.is_dir():
        raise FileNotFoundError(f"official_appagent_scripts_missing:{source}")
    shutil.copytree(source, target, symlinks=True)
    executor = target / "task_executor.py"
    if not executor.is_file():
        raise FileNotFoundError(f"official_appagent_executor_missing:{executor}")
    text = executor.read_text(encoding="utf-8")
    if _APPAGENT_DOC_LOOKUP_MARKER not in text:
        old_lookup = 'doc_path = os.path.join(docs_dir, f"{elem.uid}.txt")'
        if old_lookup not in text:
            raise RuntimeError("official_appagent_doc_lookup_not_found")
        helper = '''

def _omniflow_resolution_agnostic_doc_path(docs_dir, uid):
    """Find a demo doc by stable UID, ignoring recorded screen dimensions."""
    exact = os.path.join(docs_dir, f"{uid}.txt")
    if os.path.exists(exact):
        return exact
    def stable_key(value):
        stem = os.path.basename(str(value))
        if stem.endswith(".txt"):
            stem = stem[:-4]
        stem = re.sub(r"^_?\\d+_\\d+_", "", stem)
        stem = re.sub(r"_\\d+$", "", stem)
        return stem.strip("_")

    stable_uid = stable_key(uid)
    for filename in sorted(os.listdir(docs_dir)):
        if not filename.endswith(".txt"):
            continue
        candidate_key = stable_key(filename)
        if candidate_key and (
            candidate_key == stable_uid
            or stable_uid.endswith("_" + candidate_key)
            or stable_uid.endswith(candidate_key)
        ):
            return os.path.join(docs_dir, filename)
    return ""
'''
        insertion_anchor = 'arg_desc = "AppAgent Executor"'
        insertion_offset = text.find(insertion_anchor)
        if insertion_offset < 0:
            insertion_offset = 0
        text = (
            text[:insertion_offset]
            + helper.lstrip("\n")
            + "\n"
            + text[insertion_offset:]
        )
        lookup = re.compile(
            r"(?m)^(?P<indent>[ \\t]*)"
            + re.escape(old_lookup)
            + r"$"
        )
        match = lookup.search(text)
        if match is None:
            raise RuntimeError("official_appagent_doc_lookup_line_not_found")
        replacement = (
            match.group("indent")
            + "doc_path = _omniflow_resolution_agnostic_doc_path(docs_dir, elem.uid)"
        )
        executor.write_text(
            text[: match.start()] + replacement + text[match.end():],
            encoding="utf-8",
        )

    # AppAgent's planner and prompts remain the staged upstream files.  Its
    # controller is the only boundary replaced: observations and device
    # actions go through the same resident OOB bridge as OmniFlow.
    adapter = Path(__file__).with_name("appagent_oob_controller.py")
    if not adapter.is_file():
        raise FileNotFoundError(f"omniflow_appagent_oob_controller_missing:{adapter}")
    staged_adapter = target / "oob_appagent_controller.py"
    shutil.copy2(adapter, staged_adapter)
    controller = target / "and_controller.py"
    with controller.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n\n"
            "# OmniFlow adapter: keep AppAgent planner/model, use canonical OOB I/O.\n"
            "from oob_appagent_controller import (\n"
            "    AndroidController as _OmniFlowOobAndroidController,\n"
            "    execute_adb as _omniflow_oob_execute_adb,\n"
            "    list_all_devices as _omniflow_oob_list_all_devices,\n"
            "    traverse_tree as _omniflow_oob_traverse_tree,\n"
            ")\n"
            "AndroidController = _OmniFlowOobAndroidController\n"
            "execute_adb = _omniflow_oob_execute_adb\n"
            "list_all_devices = _omniflow_oob_list_all_devices\n"
            "traverse_tree = _omniflow_oob_traverse_tree\n"
        )

    # Qwen-compatible providers may place the useful answer in
    # ``reasoning_content`` or return one transient empty ``content`` frame.
    # The upstream AppAgent model wrapper only reads ``message.content`` and
    # therefore turns that provider response into a parser failure.  Patch
    # only the disposable staging copy; planner prompts and action parsing
    # remain upstream-owned.
    model = target / "model.py"
    if model.is_file():
        model_source = model.read_text(encoding="utf-8")
        marker = "# omniflow_appagent_glm_response_compat"
        if marker not in model_source:
            model_source += r'''

# omniflow_appagent_glm_response_compat
import json as _omniflow_json
import os as _omniflow_os
import time as _omniflow_time

def _omniflow_appagent_text(message):
    if not isinstance(message, dict):
        return ""
    candidates = (
        message.get("content"),
        message.get("reasoning_content"),
        message.get("reasoning"),
        message.get("output_text"),
    )
    for candidate in candidates:
        if isinstance(candidate, list):
            candidate = "".join(
                str(item.get("text") or item.get("content") or "")
                if isinstance(item, dict) else str(item or "")
                for item in candidate
            )
        if str(candidate or "").strip():
            return str(candidate)
    return ""

def _omniflow_appagent_event(event):
    path = _omniflow_os.environ.get("APPAGENT_STATS_JSONL", "").strip()
    if not path:
        return
    payload = dict(event) if isinstance(event, dict) else {"event": str(event)}
    payload.setdefault("ts", _omniflow_time.time())
    parent = _omniflow_os.path.dirname(path)
    if parent:
        _omniflow_os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as output:
        output.write(_omniflow_json.dumps(payload, ensure_ascii=False) + "\n")

def _omniflow_appagent_get_model_response(self, prompt, images):
    content = [{"type": "text", "text": prompt}]
    for img in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img)}"},
        })
    payload = {
        "model": self.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": self.temperature,
        "max_tokens": self.max_tokens,
        "enable_thinking": False,
    }
    thinking_mode = _omniflow_os.environ.get("APPAGENT_THINKING", "disabled").strip()
    if thinking_mode:
        payload["thinking"] = {"type": thinking_mode}
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.api_key}",
    }
    retries = max(1, int(_omniflow_os.environ.get("APPAGENT_EMPTY_RESPONSE_RETRIES", "3")))
    response = None
    for attempt in range(1, retries + 1):
        request_payload = dict(payload)
        if attempt > 1:
            request_payload["max_tokens"] = max(
                int(self.max_tokens),
                int(_omniflow_os.environ.get("APPAGENT_RETRY_MAX_TOKENS", "512")),
            )
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=request_payload,
                timeout=float(_omniflow_os.environ.get("APPAGENT_MODEL_TIMEOUT_SEC", "180")),
            ).json()
        except Exception as error:
            _omniflow_appagent_event({
                "event": "model_error",
                "attempt": attempt,
                "error": type(error).__name__,
            })
            if attempt == retries:
                return False, f"appagent_model_request_failed:{type(error).__name__}"
            continue
        if "error" in response:
            error = response.get("error") or {}
            message = str(error.get("message") or error)
            _omniflow_appagent_event({
                "event": "model_error",
                "attempt": attempt,
                "error": message[:500],
            })
            if attempt == retries:
                return False, message
            continue
        choices = response.get("choices") or []
        message = choices[0].get("message") if choices else {}
        result = _omniflow_appagent_text(message)
        usage = response.get("usage") or {}
        _omniflow_appagent_event({
            "event": "model_call",
            "attempt": attempt,
            "model": self.model,
            "requested_model": self.model,
            "endpoint": self.base_url,
            "thinking": {"type": thinking_mode},
            "message_keys": sorted(message.keys()) if isinstance(message, dict) else [],
            "content_chars": len(str((message or {}).get("content") or "")) if isinstance(message, dict) else 0,
            "reasoning_chars": len(str((message or {}).get("reasoning_content") or "")) if isinstance(message, dict) else 0,
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        })
        if result.strip():
            print_with_color(
                f"Request cost is ${'{0:.2f}'.format(int(usage.get('prompt_tokens') or 0) / 1000 * 0.01 + int(usage.get('completion_tokens') or 0) / 1000 * 0.03)}",
                "yellow",
            )
            return True, result
    _omniflow_appagent_event({"event": "model_empty_response", "attempts": retries})
    return False, "appagent_model_empty_response"

OpenAIModel.get_model_response = _omniflow_appagent_get_model_response

# omniflow_appagent_action_parse_compat
_omniflow_original_parse_explore_rsp = parse_explore_rsp
_omniflow_original_parse_grid_rsp = parse_grid_rsp

def _omniflow_bare_action(rsp, grid_mode=False):
    text = str(rsp or "").strip()
    if "FINISH" in text.upper():
        return ["FINISH"]
    match = re.search(
        r"(?m)^\s*(tap|text|long_press|swipe|grid)\s*\((.*?)\)\s*$",
        text,
    )
    if not match:
        return ["ERROR"]
    action, params = match.groups()
    values = [value.strip() for value in params.split(",")]
    try:
        if action == "grid":
            return ["grid"]
        if action in {"tap", "long_press"}:
            if grid_mode:
                return [f"{action}_grid", int(values[0]), values[1].strip("\"'"), ""]
            return [action, int(values[0]), ""]
        if action == "text":
            return [action, params.strip().strip("\"'"), ""]
        if action == "swipe" and grid_mode:
            return [
                "swipe_grid",
                int(values[0]),
                values[1].strip("\"'"),
                int(values[2]),
                values[3].strip("\"'"),
                "",
            ]
        if action == "swipe":
            return [
                action,
                int(values[0]),
                values[1].strip("\"'"),
                values[2].strip("\"'"),
                "",
            ]
    except (IndexError, ValueError):
        return ["ERROR"]
    return ["ERROR"]

def parse_explore_rsp(rsp):
    parsed = _omniflow_original_parse_explore_rsp(rsp)
    return parsed if parsed != ["ERROR"] else _omniflow_bare_action(rsp)

def parse_grid_rsp(rsp):
    parsed = _omniflow_original_parse_grid_rsp(rsp)
    return parsed if parsed != ["ERROR"] else _omniflow_bare_action(rsp, grid_mode=True)
'''
            model.write_text(model_source, encoding="utf-8")


def write_adb_proxy(
    workspace: str | Path,
    *,
    serial: str,
    adb_path: str = "adb",
) -> Path:
    """Expose exactly one device to an unmodified official subprocess."""

    resolved_workspace = Path(workspace).expanduser().resolve()
    proxy_dir = resolved_workspace / "bin"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    real_adb = shutil.which(adb_path) or adb_path
    proxy = proxy_dir / "adb"
    script = f'''#!/bin/sh
set -eu
real_adb={shlex_quote(real_adb)}
serial={shlex_quote(str(serial))}
if [ "$#" -eq 1 ] && [ "$1" = "devices" ]; then
  printf 'List of devices attached\\n%s\\tdevice\\n' "$serial"
  exit 0
fi
# AppAgent's upstream controller expects one `Physical size:` line. Its
# screenshots/XML use Android's override display size, so expose that logical
# size under the expected label when it is present. This keeps coordinate
# selection in the same space as the official screenshot/XML without changing
# AppAgent itself.
if [ "$#" -ge 4 ] && [ "$1" = "-s" ] && [ "$3" = "shell" ] && [ "$4" = "wm" ] && [ "${5:-}" = "size" ]; then
  wm_output=$("$real_adb" "$@" </dev/null)
  printf '%s\\n' "$wm_output" | awk '/^[[:space:]]*Override size:/{{sub(/^[[:space:]]*Override size:[[:space:]]*/, "Physical size: "); print; found=1; exit}} /^[[:space:]]*Physical size:/{{physical=$0}} END{{if (!found && physical) print physical; if (!found && !physical) exit 1}}' || printf '%s\\n' "$wm_output" | sed -n '1p'
  exit 0
fi
if [ "$#" -ge 3 ] && [ "$1" = "shell" ] && [ "$2" = "wm" ] && [ "${3:-}" = "size" ]; then
  wm_output=$("$real_adb" -s "$serial" "$@" </dev/null)
  printf '%s\\n' "$wm_output" | awk '/^[[:space:]]*Override size:/{{sub(/^[[:space:]]*Override size:[[:space:]]*/, "Physical size: "); print; found=1; exit}} /^[[:space:]]*Physical size:/{{physical=$0}} END{{if (!found && physical) print physical; if (!found && !physical) exit 1}}' || printf '%s\\n' "$wm_output" | sed -n '1p'
  exit 0
fi
has_serial=0
for arg in "$@"; do
  if [ "$arg" = "-s" ]; then has_serial=1; fi
done
if [ "$has_serial" -eq 1 ]; then
  exec "$real_adb" "$@" </dev/null
fi
exec "$real_adb" -s "$serial" "$@" </dev/null
'''
    proxy.write_text(script, encoding="utf-8")
    proxy.chmod(proxy.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return proxy


def shlex_quote(value: str) -> str:
    return shlex.quote(str(value))


def prepare_appagent_workspace(
    *,
    official_root: str | Path,
    docs_root: str | Path,
    workspace: str | Path,
    app_name: str,
    config: dict[str, Any],
    serial: str,
    adb_path: str = "adb",
) -> dict[str, str]:
    """Prepare an AppAgent workspace without importing AppAgent internals."""

    root = Path(official_root).expanduser().resolve()
    docs = Path(docs_root).expanduser().resolve()
    work = Path(workspace).expanduser().resolve()
    if not (root / "run.py").is_file():
        raise FileNotFoundError(f"official_appagent_entry_missing:{root / 'run.py'}")
    if not docs.is_dir():
        raise FileNotFoundError(f"official_appagent_docs_missing:{docs}")
    if not str(app_name).strip():
        raise ValueError("official_appagent_app_name_required")
    work.mkdir(parents=True, exist_ok=False)
    (work / "apps").mkdir()
    (work / "tasks").mkdir()
    _stage_appagent_scripts(root / "scripts", work / "scripts")
    _link_or_fail(docs.parent, work / "apps" / str(app_name).strip())
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("official_appagent_forward_requires_pyyaml") from exc
    (work / "config.yaml").write_text(
        yaml.safe_dump(dict(config), sort_keys=False),
        encoding="utf-8",
    )
    proxy = write_adb_proxy(work, serial=serial, adb_path=adb_path)
    return {
        "workspace": str(work),
        "app_dir": str(work / "apps" / str(app_name).strip()),
        "config": str(work / "config.yaml"),
        "adb_proxy": str(proxy),
    }


def prepare_mobilegpt_server(
    *,
    official_root: str | Path,
    memory_root: str | Path,
    workspace: str | Path,
    embedding_model: str = "",
    chat_model: str = "",
    write_through_memory: bool = False,
    port: int = 12345,
) -> dict[str, str]:
    """Stage upstream code without planner, protocol or response patches."""
    root = Path(official_root).expanduser().resolve()
    source = root / "Server"
    memory = Path(memory_root).expanduser().resolve()
    work = Path(workspace).expanduser().resolve()
    target = work / "Server"
    if not (source / "main.py").is_file():
        raise FileNotFoundError(f"official_mobilegpt_server_missing:{source}")
    if not memory.is_dir():
        raise FileNotFoundError(f"official_mobilegpt_memory_missing:{memory}")
    work.mkdir(parents=True, exist_ok=False)
    if (root / ".git").exists():
        tracked = subprocess.check_output(
            ["git", "ls-files", "-z", "Server"], cwd=root
        ).decode().split("\0")
        for name in filter(None, tracked):
            destination = work / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, destination)
    else:
        shutil.copytree(source, target, symlinks=True)
    # main.py documents deployment configuration; preserve all agent code.
    main = target / "main.py"
    source_main = main.read_text(encoding="utf-8")
    if int(port) != 12345:
        configured, count = re.subn(
            r"(server_port\s*=\s*)12345", rf"\g<1>{int(port)}", source_main
        )
        if count != 1:
            raise ValueError("upstream_mobilegpt_server_port_configuration_missing")
        main.write_text(configured, encoding="utf-8")
    _configure_mobilegpt_chat_model(target, chat_model=chat_model)
    if embedding_model:
        utils = target / "utils/utils.py"
        original = utils.read_text(encoding="utf-8")
        configured, count = re.subn(
            r'(def get_openai_embedding\(text: str, model=)"text-embedding-3-small"',
            lambda match: match.group(1) + json.dumps(embedding_model),
            original,
        )
        if count != 1:
            raise ValueError("upstream_mobilegpt_embedding_configuration_missing")
        utils.write_text(configured, encoding="utf-8")
    staged_memory = target / "memory"
    overlay = memory / "frozen_memory"
    if not overlay.is_dir():
        overlay = memory
    if write_through_memory:
        if any(memory.iterdir()):
            raise ValueError(f"official_mobilegpt_cold_memory_not_empty:{memory}")
        shutil.copytree(staged_memory, memory, symlinks=True, dirs_exist_ok=True)
        shutil.rmtree(staged_memory)
        staged_memory.symlink_to(memory, target_is_directory=True)
    else:
        # Memory includes upstream Python modules as well as learned data.
        # Do not allow a converted asset to replace the executable code.
        for entry in overlay.rglob("*"):
            relative = entry.relative_to(overlay)
            if "__pycache__" in relative.parts or entry.suffix in {".py", ".pyc"}:
                continue
            destination = staged_memory / relative
            if entry.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            elif entry.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(entry, destination)
    return {
        "workspace": str(work),
        "server_root": str(target),
        "memory_root": str(staged_memory),
    }


def _configure_mobilegpt_chat_model(
    server_root: Path,
    *,
    chat_model: str = "",
) -> None:
    """Set MobileGPT's legacy model aliases in the disposable Server copy."""

    normalized_chat = str(chat_model or "").strip()
    if not normalized_chat:
        return
    main_path = server_root / "main.py"
    if not main_path.is_file():
        return
    source = main_path.read_text(encoding="utf-8")
    quoted_model = json.dumps(normalized_chat)
    source = re.sub(
        r'default_chat_model = os\.getenv\("MOBILEGPT_CHAT_MODEL", "[^"]*"\)',
        f'default_chat_model = os.getenv("MOBILEGPT_CHAT_MODEL", {quoted_model})',
        source,
        count=1,
    )
    aliases = (
        "TASK_AGENT_GPT_VERSION",
        "APP_AGENT_GPT_VERSION",
        "SELECT_AGENT_HISTORY_GPT_VERSION",
        "EXPLORE_AGENT_GPT_VERSION",
        "SELECT_AGENT_GPT_VERSION",
        "DERIVE_AGENT_GPT_VERSION",
        "PARAMETER_FILLER_AGENT_GPT_VERSION",
        "ACTION_SUMMARIZE_AGENT_GPT_VERSION",
        "SUBTASK_MERGE_AGENT_GPT_VERSION",
    )
    for name in aliases:
        source = re.sub(
            rf'os\.environ\["{re.escape(name)}"\] = "[^"]*"',
            f'os.environ["{name}"] = os.getenv("MOBILEGPT_CHAT_MODEL", {quoted_model})',
            source,
        )
    source = re.sub(
        r'os\.environ\["gpt_4"\] = "[^"]*"',
        f'os.environ["gpt_4"] = os.getenv("MOBILEGPT_CHAT_MODEL", {quoted_model})',
        source,
    )
    source = re.sub(
        r'os\.environ\["gpt_4_turbo"\] = "[^"]*"',
        f'os.environ["gpt_4_turbo"] = os.getenv("MOBILEGPT_CHAT_MODEL", {quoted_model})',
        source,
    )
    source = re.sub(
        r'os\.environ\["gpt_3_5_turbo"\] = "[^"]*"',
        f'os.environ["gpt_3_5_turbo"] = os.getenv("MOBILEGPT_CHAT_MODEL", {quoted_model})',
        source,
    )
    source = re.sub(
        r'os\.environ\["vision_model"\] = "[^"]*"',
        f'os.environ["vision_model"] = os.getenv("MOBILEGPT_VISION_MODEL", {quoted_model})',
        source,
    )
    main_path.write_text(source, encoding="utf-8")


def _run_adb(
    adb_path: str,
    serial: str,
    args: Sequence[str],
    *,
    check: bool = True,
    timeout_sec: float = 30.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [adb_path, "-s", serial, *args],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=max(1.0, float(timeout_sec)),
    )


MOBILEGPT_HANDSHAKE_RETURN_CODE = 127
MOBILEGPT_SERVER_ERROR_RETURN_CODE = 128
# The official server may need one planner/derive round before its first
# primitive action.  Qwen-backed cold starts can exceed 20 seconds even when
# the client and server are healthy, so the protocol window must cover that
# normal response latency plus the client's screen-refresh grace period.
MOBILEGPT_HANDSHAKE_TIMEOUT_SEC = 60.0


def _mobilegpt_environment_failure(
    *,
    failure_reason: str,
    returncode: int,
) -> bool:
    """Keep transport/setup failures retryable and method limits terminal."""

    reason = str(failure_reason or "").strip()
    # Hitting the explicit planner-call budget is a terminal method outcome,
    # not a device/server environment fault.  Keep it visible as a method
    # failure so schedulers do not spend another episode repeating it.
    if (
        "mobilegpt_model_call_limit_exceeded" in reason
        or "mobilegpt_model_response_invalid" in reason
    ):
        return False
    if int(returncode) in {
        MOBILEGPT_HANDSHAKE_RETURN_CODE,
        MOBILEGPT_SERVER_ERROR_RETURN_CODE,
    }:
        return True
    if reason in {
        "mobilegpt_handshake_failed",
        "mobilegpt_handshake_timeout",
        "mobilegpt_server_handler_failed",
        "mobilegpt_target_app_package_unresolved",
        "mobilegpt_target_app_not_ready",
    }:
        return True
    return reason.startswith("mobilegpt_target_app_not_ready:")


def _mobilegpt_instruction_broadcast_args(instruction: str) -> list[str]:
    """Build the shell-safe official-client instruction broadcast.

    ``adb shell`` joins its argv before handing it to the device shell.  An
    unquoted AndroidWorld goal therefore gets truncated at the first space
    when used as the value of ``am broadcast --es``.  The official service
    registers a dynamic receiver; Android drops that receiver when an
    explicit ``am broadcast -p`` package filter is added, so keep the action
    implicit and quote only this transport argument.
    """

    return [
        "shell",
        "am",
        "broadcast",
        "-a",
        "com.example.MobileGPT.STRING_ACTION",
        "--es",
        "com.example.MobileGPT.INSTRUCTION_EXTRA",
        shlex.quote(str(instruction)),
    ]


def _wait_for_mobilegpt_accessibility_service(
    adb_path: str,
    serial: str,
    service: str,
    *,
    attempts: int,
) -> bool:
    attempt_count = max(1, int(attempts))
    for attempt in range(attempt_count):
        accessibility_state = _run_adb(
            adb_path,
            serial,
            ["shell", "dumpsys", "accessibility"],
            check=False,
        ).stdout
        if _mobilegpt_accessibility_service_bound(accessibility_state, service):
            return True
        if attempt + 1 < attempt_count:
            time.sleep(1.0)
    return False


def _ensure_mobilegpt_accessibility_service_bound(
    adb_path: str,
    serial: str,
    service: str,
    services: Sequence[str],
    *,
    initial_attempts: int = 5,
    retry_attempts: int = 10,
) -> bool:
    if _wait_for_mobilegpt_accessibility_service(
        adb_path,
        serial,
        service,
        attempts=initial_attempts,
    ):
        return True
    # Retry the accessibility manager transition only after observing an
    # unbound service. Unconditionally disabling an already bound official
    # service calls its non-idempotent onDestroy(), which can crash while it
    # unregisters an already-removed broadcast receiver.
    _run_adb(
        adb_path,
        serial,
        ["shell", "settings", "put", "secure", "accessibility_enabled", "0"],
        check=False,
    )
    _run_adb(
        adb_path,
        serial,
        [
            "shell",
            "settings",
            "put",
            "secure",
            "enabled_accessibility_services",
            ":".join(services),
        ],
        check=False,
    )
    _run_adb(
        adb_path,
        serial,
        ["shell", "settings", "put", "secure", "accessibility_enabled", "1"],
        check=False,
    )
    return _wait_for_mobilegpt_accessibility_service(
        adb_path,
        serial,
        service,
        attempts=retry_attempts,
    )


def _mobilegpt_protocol_probe(
    stats_path: Path,
    log_text: str,
    server_log_text: str = "",
) -> dict[str, Any]:
    """Summarize the client/server boundary without changing the official code."""

    events: list[dict[str, Any]] = []
    if stats_path.is_file():
        for line in stats_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                events.append(value)
    lowered = str(log_text or "").lower()
    server_lowered = str(server_log_text or "").lower()
    # The upstream TCP server creates one thread per short-lived client
    # connection. Android reconnects while switching from app discovery to
    # screen execution, and a closed auxiliary socket can therefore print a
    # bare ConnectionResetError without affecting the active episode. Do not
    # let that transport cleanup traceback kill the healthy official client.
    server_diagnostics = re.sub(
        r"exception in thread[^\n]*\ntraceback .*?"
        r"connectionreseterror:[^\n]*(?:\n|$)",
        "",
        server_lowered,
        flags=re.DOTALL,
    )
    client_errors = tuple(
        marker
        for marker in (
            "server offline",
            "socket not connected yet",
            "connection refused",
            "failed to connect",
            "unable to connect",
        )
        if marker in lowered
    )
    task_started = sum(event.get("event") == "task_started" for event in events)
    task_finished = sum(event.get("event") == "task_finished" for event in events)
    action_sent = sum(
        event.get("event") == "mobilegpt_action_sent" for event in events
    )
    server_error_markers = tuple(
        marker
        for marker in (
            "traceback (most recent call last)",
            "openaierror",
            "missing credentials",
            "connectionrefusederror",
            "exception in thread",
            "mobilegpt_model_call_limit_exceeded",
            "keyerror: 'speak'",
        )
        if marker in server_diagnostics
    )
    return {
        "schema_version": "omniflow.mobilegpt_protocol_probe.v1",
        "stats_event_count": len(events),
        "task_started": task_started > 0 or "receive broadcast" in lowered,
        "task_started_count": task_started,
        "task_finished": task_finished > 0 or "task finished" in lowered,
        "task_finished_count": task_finished,
        "action_sent_count": action_sent,
        "client_service_ready": "# of apps" in lowered,
        "client_broadcast_received": "receive broadcast" in lowered,
        "client_error": bool(client_errors),
        "client_error_markers": list(client_errors),
        "server_error": bool(server_error_markers),
        "server_error_markers": list(server_error_markers),
        "phase": (
            "episode"
            if task_started > 0 or action_sent > 0
            else "client_server_handshake"
            if "receive broadcast" in lowered or client_errors
            else "client_startup"
        ),
    }


def _run_mobilegpt_client(
    *,
    official_root: str | Path,
    serial: str,
    adb_path: str,
    host: str,
    instruction: str,
    output_root: str | Path,
    timeout_sec: float,
    max_steps: int = 0,
    server_port: int = 12345,
    handshake_timeout_sec: float = MOBILEGPT_HANDSHAKE_TIMEOUT_SEC,
    server_log_path: str | Path = "",
) -> tuple[int, float]:
    """Run one episode through MobileGPT's official Accessibility client."""

    root = Path(official_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    installed_client = _run_adb(
        adb_path,
        serial,
        ["shell", "pm", "path", "com.example.MobileGPT"],
        check=False,
    )
    reuse_installed_client = (
        installed_client.returncode == 0
        and "package:" in installed_client.stdout
        and os.environ.get("OMNIFLOW_MOBILEGPT_REBUILD_CLIENT", "").strip() != "1"
    )
    client_workspace = tempfile.TemporaryDirectory(
        prefix="omniflow-mobilegpt-client-"
    )
    client_root = Path(client_workspace.name) / "official_client"
    shutil.copytree(root / "App", client_root)
    global_java = (
        client_root
        / "app/src/main/java/com/example/MobileGPT/MobileGPTGlobal.java"
    )
    source = global_java.read_text(encoding="utf-8")
    source = source.replace(
        'HOST_IP = "INPUT_YOUR_SERVER_IP_ADDRESS"',
        f'HOST_IP = "{str(host).replace(chr(34), "")}"',
    )
    source = re.sub(
        r"(HOST_PORT\s*=\s*)\d+",
        rf"\g<1>{int(server_port)}",
        source,
        count=1,
    )
    global_java.write_text(source, encoding="utf-8")
    sdk = str(
        os.environ.get("ANDROID_HOME")
        or os.environ.get("ANDROID_SDK_ROOT")
        or ""
    ).strip()
    if not sdk and Path(adb_path).is_file():
        adb_parent = Path(adb_path).expanduser().resolve().parent
        if adb_parent.name == "platform-tools":
            sdk = str(adb_parent.parent)
    if sdk:
        (client_root / "local.properties").write_text(
            f"sdk.dir={sdk}\n",
            encoding="utf-8",
        )
    configured_apk = str(os.environ.get("OMNIFLOW_MOBILEGPT_APK") or "").strip()
    if reuse_installed_client:
        apk = Path()
    elif configured_apk:
        prebuilt_apk = Path(configured_apk).expanduser()
        if not prebuilt_apk.is_file():
            raise FileNotFoundError(
                f"official_mobilegpt_configured_apk_missing:{prebuilt_apk}"
            )
        apk = client_root / "app/build/outputs/apk/debug/app-debug.apk"
        apk.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(prebuilt_apk, apk)
    else:
        gradle = shutil.which(os.environ.get("OMNIFLOW_GRADLE_BIN", "gradle"))
        if not gradle:
            candidates = sorted(
                Path.home().glob(".gradle/wrapper/dists/*/*/gradle-*/bin/gradle"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            gradle = str(candidates[0]) if candidates else ""
        if not gradle:
            prebuilt_apk = root / "App/app/build/outputs/apk/debug/app-debug.apk"
            if str(host).strip() != "10.0.2.2" or not prebuilt_apk.is_file():
                raise RuntimeError(
                    "official_mobilegpt_client_requires_gradle:"
                    " install Gradle or provide the official App debug APK"
                )
            apk = client_root / "app/build/outputs/apk/debug/app-debug.apk"
            apk.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(prebuilt_apk, apk)
        else:
            subprocess.run(
                [gradle, ":app:assembleDebug"],
                cwd=client_root,
                check=True,
                text=True,
            )
    if reuse_installed_client:
        install_result = subprocess.CompletedProcess([], 0, "")
    else:
        apk = client_root / "app/build/outputs/apk/debug/app-debug.apk"
        if not apk.is_file():
            raise FileNotFoundError(f"official_mobilegpt_apk_missing:{apk}")
        install_result = _run_adb(
            adb_path,
            serial,
            ["install", "-r", str(apk)],
            check=False,
        )
    if (
        install_result.returncode != 0
        and "INSTALL_FAILED_UPDATE_INCOMPATIBLE" in install_result.stdout
    ):
        # The disposable official client is rebuilt on every run and therefore
        # has a debug signing key that may differ from the APK left on the
        # emulator by a previous run.  This package is pipeline-owned, so
        # replace only that client package when Android reports a signature
        # mismatch; the task application and its state are untouched.
        _run_adb(
            adb_path,
            serial,
            ["shell", "pm", "uninstall", "--user", "0", "com.example.MobileGPT"],
            check=False,
        )
        install_result = _run_adb(
            adb_path,
            serial,
            ["install", "-r", str(apk)],
            check=False,
        )
    if install_result.returncode != 0:
        raise subprocess.CalledProcessError(
            install_result.returncode,
            [adb_path, "-s", serial, "install", "-r", str(apk)],
            output=install_result.stdout,
        )
    # A prior setup or a restored emulator snapshot can leave the official
    # APK installed but disabled (pm reports enabled=0).  In that state the
    # accessibility service can never bind, even when its component is
    # correctly listed in enabled_accessibility_services.
    _run_adb(
        adb_path,
        serial,
        ["shell", "pm", "enable", "com.example.MobileGPT"],
        check=False,
    )
    # The official client requests these permissions on first launch. Grant
    # them through the device shell so a headless run cannot stop at the
    # Android permission controller; failures are tolerated for permissions
    # unavailable on a particular API level.
    for permission in (
        "android.permission.ACCESS_FINE_LOCATION",
        "android.permission.ACCESS_COARSE_LOCATION",
        "android.permission.READ_EXTERNAL_STORAGE",
        "android.permission.WRITE_EXTERNAL_STORAGE",
        "android.permission.RECORD_AUDIO",
    ):
        _run_adb(
            adb_path,
            serial,
            ["shell", "pm", "grant", "com.example.MobileGPT", permission],
            check=False,
        )
    _run_adb(
        adb_path,
        serial,
        ["shell", "am", "force-stop", "com.example.MobileGPT"],
        check=False,
    )
    # The installed official APK declares the service under the
    # ``com.example.MobileGPT.MobileGPTAccessibilityService`` class.  Keep
    # the component's package/class spelling exact; omitting the intermediate
    # ``MobileGPT`` namespace silently leaves API-34 services unbound.
    service = (
        "com.example.MobileGPT/"
        "com.example.MobileGPT.MobileGPTAccessibilityService"
    )
    current = _run_adb(
        adb_path,
        serial,
        ["shell", "settings", "get", "secure", "enabled_accessibility_services"],
        check=False,
    ).stdout.strip()
    disabled_non_mobilegpt_services = {
        OOB_CONTROL_ACCESSIBILITY_SERVICE,
        "com.google.androidenv.accessibilityforwarder/"
        "com.google.androidenv.accessibilityforwarder.AccessibilityForwarder",
    }
    services = [
        value
        for value in current.split(":")
        if value and value != "null" and value not in disabled_non_mobilegpt_services
    ]
    if service not in services:
        services.append(service)
    # Android may retain the old accessibility manager state across an APK
    # replacement or a restored emulator snapshot.  Merely writing the new
    # service list then setting accessibility_enabled=1 is not sufficient on
    # those images: the old manager remains active and the new service never
    # receives a bind.  Force the documented off -> list -> on transition so
    # the official client starts from the same state on every target device.
    _run_adb(
        adb_path,
        serial,
        ["shell", "settings", "put", "secure", "accessibility_enabled", "0"],
        check=False,
    )
    _run_adb(
        adb_path,
        serial,
        ["shell", "settings", "put", "secure", "enabled_accessibility_services", ":".join(services)],
        check=False,
    )
    _run_adb(
        adb_path,
        serial,
        ["shell", "settings", "put", "secure", "accessibility_enabled", "1"],
        check=False,
    )
    _run_adb(adb_path, serial, ["shell", "monkey", "-p", "com.example.MobileGPT", "1"])
    service_bound = _ensure_mobilegpt_accessibility_service_bound(
        adb_path,
        serial,
        service,
        services,
    )
    if not service_bound:
        # Do not broadcast a task to an unbound client.  Without this
        # precondition the official server accepts one socket connection,
        # the APK silently misses the broadcast, and the episode waits until
        # its wall-clock timeout while producing no action or telemetry.
        raise RuntimeError("mobilegpt_accessibility_service_not_bound")
    time.sleep(2.0)
    # The official Accessibility service receives the instruction below and
    # launches the target package itself from the Server's package frame.
    # There is deliberately no target-app adb launch or OOB prelaunch here.
    episode_started = time.monotonic()
    _run_adb(adb_path, serial, ["logcat", "-c"])
    _run_adb(
        adb_path,
        serial,
        _mobilegpt_instruction_broadcast_args(instruction),
    )
    stats_path = Path(os.environ.get("MOBILEGPT_STATS_JSONL", "")).expanduser()
    deadline = time.monotonic() + max(1.0, float(timeout_sec))
    last_log = ""
    server_log = Path(server_log_path).expanduser() if str(server_log_path).strip() else None

    def read_server_log() -> str:
        if server_log is None or not server_log.is_file():
            return ""
        return server_log.read_text(encoding="utf-8", errors="replace")[-20000:]

    def persist_raw_logs() -> None:
        """Preserve upstream telemetry before the episode temp tree is removed."""

        if stats_path.is_file():
            shutil.copy2(stats_path, output / "mobilegpt_stats.jsonl")
        if server_log is not None and server_log.is_file():
            shutil.copy2(server_log, output / "official_server.log")

    def finish_with_probe(
        returncode: int,
        log: str,
        reason: str,
    ) -> tuple[int, float]:
        probe = _mobilegpt_protocol_probe(stats_path, log, read_server_log())
        probe.update(
            {
                "failure_reason": reason,
                "returncode": int(returncode),
                "server_host": str(host),
                "server_port": int(server_port),
                "handshake_timeout_sec": float(handshake_timeout_sec),
            }
        )
        (output / "protocol_probe.json").write_text(
            json.dumps(probe, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output / "client_log.txt").write_text(
            log + f"\n[omniflow] {reason}\n",
            encoding="utf-8",
        )
        persist_raw_logs()
        return returncode, episode_started

    while time.monotonic() < deadline:
        log = _run_adb(
            adb_path,
            serial,
            [
                "logcat",
                "-d",
                "-s",
                "MobileGPT_Service:D",
                "MobileGPT_CLIENT:D",
                "AndroidRuntime:E",
                "*:S",
            ],
            check=False,
        ).stdout
        last_log = log
        probe = _mobilegpt_protocol_probe(stats_path, log, read_server_log())
        if "Task finished" in log or "-----------Task finished--------" in log:
            probe.update(
                {
                    "failure_reason": "",
                    "returncode": 0,
                    "server_host": str(host),
                    "server_port": int(server_port),
                    "handshake_timeout_sec": float(handshake_timeout_sec),
                }
            )
            (output / "protocol_probe.json").write_text(
                json.dumps(probe, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (output / "client_log.txt").write_text(log, encoding="utf-8")
            persist_raw_logs()
            return 0, episode_started
        if probe["client_error"] and not probe["task_started"]:
            _run_adb(
                adb_path,
                serial,
                ["shell", "am", "force-stop", "com.example.MobileGPT"],
                check=False,
            )
            return finish_with_probe(
                MOBILEGPT_HANDSHAKE_RETURN_CODE,
                log,
                "mobilegpt_handshake_failed",
            )
        if probe["server_error"] and not probe["task_started"]:
            _run_adb(
                adb_path,
                serial,
                ["shell", "am", "force-stop", "com.example.MobileGPT"],
                check=False,
            )
            return finish_with_probe(
                MOBILEGPT_SERVER_ERROR_RETURN_CODE,
                log,
                (
                    "mobilegpt_model_call_limit_exceeded"
                    "mobilegpt_model_call_limit_exceeded"
                    if "mobilegpt_model_call_limit_exceeded"
                    in probe["server_error_markers"]
                    else "mobilegpt_model_response_invalid"
                    if "keyerror: 'speak'" in probe["server_error_markers"]
                    else "mobilegpt_server_handler_failed"
                ),
            )
        if probe["server_error"] and probe["task_started"]:
            # A server-side model/protocol exception is terminal for this
            # episode.  Do not let the Android client keep waiting until its
            # unrelated per-step timeout hides the actual cause.
            _run_adb(
                adb_path,
                serial,
                ["shell", "am", "force-stop", "com.example.MobileGPT"],
                check=False,
            )
            return finish_with_probe(
                MOBILEGPT_SERVER_ERROR_RETURN_CODE,
                log,
                (
                    "mobilegpt_model_call_limit_exceeded"
                    "mobilegpt_model_call_limit_exceeded"
                    if "mobilegpt_model_call_limit_exceeded"
                    in probe["server_error_markers"]
                    else "mobilegpt_model_response_invalid"
                    if "keyerror: 'speak'" in probe["server_error_markers"]
                    else "mobilegpt_server_handler_failed"
                ),
            )
        # Upstream emits no injected action counters. Only the episode wall
        # deadline bounds a live native client; absent telemetry is not a stall.
        time.sleep(1.0)
    # Keep the return contract identical to every earlier exit path.  The
    # AndroidWorld wrapper unpacks ``(returncode, episode_started)`` to compute
    # episode duration; returning a bare integer here causes a secondary
    # TypeError exactly when a long episode times out and prevents result/raw
    # log persistence.
    return finish_with_probe(124, last_log, "mobilegpt_episode_timeout")


def run_mobilegpt_client(
    *,
    official_root: str | Path,
    serial: str,
    adb_path: str,
    host: str,
    instruction: str,
    output_root: str | Path,
    timeout_sec: float,
    android_world_root: str | Path | None = None,
    task_name: str = "",
    task_params_json: str = "{}",
    task_seed: int = 113,
    console_port: int = 5560,
    grpc_port: int = 8560,
    perform_emulator_setup: bool = True,
    max_steps: int = 0,
    server_port: int = 12345,
    handshake_timeout_sec: float = MOBILEGPT_HANDSHAKE_TIMEOUT_SEC,
    server_log_path: str | Path = "",
) -> int:
    """Run the upstream Android client with official benchmark validation."""

    if not android_world_root or not task_name:
        return _run_mobilegpt_client(
            official_root=official_root,
            serial=serial,
            adb_path=adb_path,
            host=host,
            instruction=instruction,
            output_root=output_root,
            timeout_sec=timeout_sec,
            max_steps=max_steps,
            server_port=server_port,
            handshake_timeout_sec=handshake_timeout_sec,
            server_log_path=server_log_path,
        )[0]

    task_params = json.loads(str(task_params_json or "{}"))
    if not isinstance(task_params, dict):
        raise ValueError("androidworld_task_params_must_be_object")
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with _androidworld_task_startup(
        android_world_root=android_world_root,
        task_name=task_name,
        task_params_json=task_params_json,
        task_seed=task_seed,
        serial=serial,
        console_port=console_port,
        grpc_port=grpc_port,
        adb_path=adb_path,
        perform_emulator_setup=perform_emulator_setup,
        # MobileGPT temporarily owns the Accessibility service during the
        # episode.  The validator uses AndroidWorld's forwarder, which is
        # restored after the client exits below.
        use_uiautomator=False,
        output_root=output,
        method="mobilegpt",
    ) as (env, task):
        task_params = dict(getattr(task, "params", {}) or {})
        official_instruction = str(
            getattr(task, "goal", "") or instruction or task_name
        ).strip()
        returncode, episode_started = _run_mobilegpt_client(
            official_root=official_root,
            serial=serial,
            adb_path=adb_path,
            host=host,
            instruction=official_instruction,
            output_root=output,
            timeout_sec=timeout_sec,
            max_steps=max_steps,
            server_port=server_port,
            handshake_timeout_sec=handshake_timeout_sec,
            server_log_path=server_log_path,
        )
        episode_finished = time.monotonic()
        # OOB startup deliberately disables AndroidWorld's Accessibility
        # wrapper while MobileGPT owns observation/action.  The final
        # validator is read-only and uses AndroidWorld's official XML path;
        # it must not call the absent gRPC wrapper.
        _select_androidworld_validator_observer(env)
        validator_covered_by_call, reward, validator_error = (
            _safe_official_validator(task, env)
        )
        probe_path = output / "protocol_probe.json"
        try:
            probe = json.loads(probe_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            probe = {}
        if not isinstance(probe, dict):
            probe = {}
        validator_covered = bool(
            probe.get("task_started")
            or probe.get("task_finished")
            or probe.get("action_sent_count")
        ) and validator_covered_by_call
        stats_path = Path(
            os.environ.get("MOBILEGPT_STATS_JSONL", "")
        ).expanduser()
        stats = _mobilegpt_stats_summary(stats_path)
        reason = str(probe.get("failure_reason") or "").strip()
        if validator_error:
            reason = "; ".join(value for value in (reason, validator_error) if value)
        environment_failure = bool(validator_error) or (not validator_covered) or (
            reward <= 0.5 and _mobilegpt_environment_failure(
            failure_reason=reason,
            returncode=returncode,
            )
        )
        result_row = {
            "schema_version": "omniflow.androidworld.result.v1",
            "task_name": task_name,
            "task": task_name,
            "goal": official_instruction,
            "requested_instruction": instruction,
            "official_task_instruction": official_instruction,
            "task_params": _json_safe(task_params),
            "task_params_sha256": hashlib.sha256(
                json.dumps(
                    task_params,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            ).hexdigest(),
            "method": "mobilegpt",
            "device": serial,
            "task_random_seed": int(task_seed),
            "fixed_task_seed": True,
            "fixed_task_params": True,
            "official_validator_used": validator_covered,
            "official_validator_success": validator_covered and reward > 0.5,
            "official_validator_coverage_rate": 1.0 if validator_covered else 0.0,
            "androidworld_validator_result": {
                "validator": "androidworld_official",
                "success": validator_covered and reward > 0.5,
                "reward": reward,
                "covered": validator_covered,
            },
            "process_returncode": int(returncode),
            "classification": (
                "success"
                if validator_covered and reward > 0.5
                else "environment_failure"
                if environment_failure
                else "method_failure"
            ),
            "actions_executed": int(probe.get("action_sent_count") or 0),
            "planner_steps": int(probe.get("action_sent_count") or 0),
            **stats,
            "token_usage_status": str(
                stats.get("token_usage_status") or "unavailable"
            ),
            "fallback_steps": int(stats.get("fallback_count") or 0),
            "mobilegpt_native_action_index_protocol": (
                "mobilegpt_official_accessibility_node_id_v1"
            ),
            "mobilegpt_stats_jsonl": str(
                output / "mobilegpt_stats.jsonl"
                if (output / "mobilegpt_stats.jsonl").is_file()
                else stats_path
            ),
            "mobilegpt_protocol": {
                "transport": "official_accessibility",
                "action_index": "mobilegpt_official_accessibility_node_id_v1",
                "server_host": str(host),
                "server_port": int(server_port),
                "task_finished": bool(probe.get("task_finished")),
            },
            "environment_failure": environment_failure,
            "failure_reason": reason,
            "runtime_integrity_error": reason,
            "physical_backend": "mobilegpt_official_accessibility",
            "observe_backend": "mobilegpt_official_accessibility",
            "action_backend": "mobilegpt_official_accessibility",
            "duration_ms": round(
                (episode_finished - episode_started) * 1000.0,
                3,
            ),
            "protocol_probe": str(probe_path),
        }
        (output / "task_results.jsonl").write_text(
            json.dumps(result_row, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return 0 if reward > 0.5 else int(returncode)


def run_appagent_executor(
    *,
    python_executable: str,
    executor: str | Path,
    app_name: str,
    serial: str,
    workspace: str | Path,
    goal: str,
    timeout_sec: float,
    android_world_root: str | Path,
    task_name: str,
    task_params_json: str,
    task_seed: int,
    console_port: int,
    grpc_port: int,
    adb_path: str,
    output_root: str | Path,
    perform_emulator_setup: bool = True,
    max_steps: int = 0,
) -> int:
    """Run official AppAgent after the canonical task initialization."""

    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    task_params = json.loads(str(task_params_json or "{}"))
    if not isinstance(task_params, dict):
        raise ValueError("androidworld_task_params_must_be_object")
    started = time.monotonic()
    with _androidworld_task_startup(
        android_world_root=android_world_root,
        task_name=task_name,
        task_params_json=task_params_json,
        task_seed=task_seed,
        serial=serial,
        console_port=console_port,
        grpc_port=grpc_port,
        adb_path=adb_path,
        perform_emulator_setup=perform_emulator_setup,
        use_uiautomator=False,
        output_root=output,
        method="appagent",
    ) as (env, task):
        process_returncode = 1
        runtime_integrity_error = ""
        try:
            staged_executor = Path(workspace).expanduser().resolve() / "scripts" / "task_executor.py"
            executor_path = staged_executor if staged_executor.is_file() else Path(executor)
            official_log = output / "official_appagent.log"
            child_env = os.environ.copy()
            repo_root = Path(__file__).resolve().parents[2]
            child_env["PYTHONPATH"] = os.pathsep.join(
                value for value in (str(repo_root), child_env.get("PYTHONPATH", "")) if value
            )
            child_env["ADB_PATH"] = str(adb_path)
            child_env["OMNIFLOW_ADB_PATH"] = str(adb_path)
            child_env["OMNIFLOW_APPA_AGENT_SERIAL"] = str(serial)
            child_env["OMNIFLOW_ANDROIDWORLD_CONTROL_BACKEND"] = "oob"
            child_env["APPAGENT_THINKING"] = FORMAL_THINKING
            appagent_stats_path = output / "appagent_stats.jsonl"
            child_env["APPAGENT_STATS_JSONL"] = str(appagent_stats_path)
            with official_log.open("w", encoding="utf-8") as log_file:
                process = subprocess.Popen(
                    [
                        str(python_executable),
                        "-u",
                        str(executor_path),
                        "--app",
                        str(app_name),
                        "--root_dir",
                        str(workspace),
                    ],
                    cwd=str(workspace),
                    stdin=subprocess.PIPE,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=child_env,
                )
                if process.stdin is not None:
                    process.stdin.write(str(goal) + "\n")
                    process.stdin.close()
                deadline = time.monotonic() + max(1.0, float(timeout_sec))
                while process.poll() is None:
                    log_file.flush()
                    action_count = _count_appagent_actions(
                        official_log.read_text(
                            encoding="utf-8", errors="replace"
                        )
                    )
                    if max_steps > 0 and action_count >= max_steps:
                        process.terminate()
                        try:
                            process.wait(timeout=5.0)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5.0)
                        process_returncode = 125
                        runtime_integrity_error = "appagent_step_budget_exhausted"
                        break
                    if time.monotonic() >= deadline:
                        process.terminate()
                        try:
                            process.wait(timeout=5.0)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5.0)
                        process_returncode = 124
                        runtime_integrity_error = "appagent_step_timeout"
                        break
                    time.sleep(0.5)
                else:
                    process_returncode = int(process.returncode or 0)
        except OSError:
            process_returncode = 1
        # AppAgent also runs with OOB ownership of observation/action.  Use
        # the same read-only AndroidWorld XML observer for final validation;
        # otherwise UI validators attempt to consume the disabled gRPC tree.
        _select_androidworld_validator_observer(env)
        validator_covered_by_call, reward, validator_error = (
            _safe_official_validator(task, env)
        )
        # The AndroidWorld validator is the single authoritative judge of task
        # success shared by every formal method; a clean subprocess exit is a
        # separate liveness fact recorded in process_returncode below, not a
        # gate on success (an AppAgent process that crashes after already
        # reaching the goal state must still count as a success, same as
        # every other method).
        validator_success = reward > 0.5
        official_log = output / "official_appagent.log"
        actions_executed = 0
        if official_log.is_file():
            actions_executed = _count_appagent_actions(
                official_log.read_text(encoding="utf-8", errors="replace")
            )
        appagent_stats = _load_appagent_stats(output / "appagent_stats.jsonl")
        validator_covered = bool(
            process_returncode == 0 or actions_executed > 0
        ) and validator_covered_by_call
        validator_environment_failure = bool(validator_error)
        result_row = {
            "schema_version": "omniflow.androidworld.result.v1",
            "task_name": task_name,
            "task": task_name,
            "goal": str(goal),
            "task_params": task_params,
            "task_params_sha256": hashlib.sha256(
                json.dumps(
                    task_params,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            ).hexdigest(),
            "method": "appagent",
            "device": serial,
            "task_random_seed": int(task_seed),
            "fixed_task_seed": True,
            "fixed_task_params": True,
            "official_validator_used": validator_covered,
            "official_validator_success": validator_covered and validator_success,
            "official_validator_coverage_rate": 1.0 if validator_covered else 0.0,
            "androidworld_validator_result": {
                "validator": "androidworld_official",
                "success": validator_covered and reward > 0.5,
                "reward": reward,
                "covered": validator_covered,
            },
            "process_returncode": process_returncode,
            "classification": (
                "success"
                if validator_covered and validator_success
                else "environment_failure"
                if validator_environment_failure
                else "method_failure"
            ),
            "actions_executed": actions_executed,
            "model_calls": appagent_stats["model_calls"],
            "prompt_tokens": appagent_stats["prompt_tokens"],
            "completion_tokens": appagent_stats["completion_tokens"],
            "total_tokens": appagent_stats["total_tokens"],
            "token_usage_status": appagent_stats["status"],
            "fallback_steps": 0,
            "duration_ms": round((time.monotonic() - started) * 1000.0, 3),
            "official_log": str(official_log),
            "appagent_stats_jsonl": str(output / "appagent_stats.jsonl"),
            "appagent_empty_responses": appagent_stats["empty_responses"],
            "appagent_model_errors": appagent_stats["errors"],
            "runtime_integrity_error": "; ".join(
                value
                for value in (runtime_integrity_error, validator_error)
                if value
            ),
            "failure_reason": validator_error,
            "environment_failure": validator_environment_failure,
        }
        (output / "task_results.jsonl").write_text(
            json.dumps(result_row, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        # AppAgent's official executor owns its native trace and historically
        # emitted only task_results.jsonl.  The shared experiment runner
        # promotes sealed evidence through the canonical run_log.json slot,
        # so write a truthful envelope here as well.  Do not fabricate
        # observations or action steps that the official executor did not
        # expose; the native log and usage JSONL remain the detailed evidence.
        run_log = {
            "schema_version": "oob.run_log.canonical.v1",
            "task_name": task_name,
            "task": task_name,
            "goal": str(goal),
            "method": "appagent",
            "device": serial,
            "task_random_seed": int(task_seed),
            "status": "succeeded" if validator_success else "failed",
            "success": bool(validator_success),
            "steps": [],
            "validator": {
                "official": True,
                "success": bool(validator_success),
                "reward": reward,
            },
            "diagnostics": {
                "execution_summary": {
                    "actions_executed": actions_executed,
                    "model_calls": appagent_stats["model_calls"],
                    "prompt_tokens": appagent_stats["prompt_tokens"],
                    "completion_tokens": appagent_stats["completion_tokens"],
                    "total_tokens": appagent_stats["total_tokens"],
                    "token_usage_status": appagent_stats["status"],
                    "duration_ms": result_row["duration_ms"],
                },
                "official_result": result_row,
                "official_log": str(official_log),
                "appagent_stats_jsonl": str(output / "appagent_stats.jsonl"),
                "trace_evidence_status": "native_appagent_log_only",
            },
        }
        (output / "run_log.json").write_text(
            json.dumps(run_log, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        # The official process status and AndroidWorld validator conclusion are
        # separate facts.  Keep a clean executor exit code even when the
        # validator rejects the resulting device state; the scheduler reads
        # official_validator_success for the method outcome.
    return process_returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Forward one task to an official baseline")
    parser.add_argument(
        "--baseline", choices=("mobilegpt", "appagent"), default="mobilegpt"
    )
    parser.add_argument("--root")
    parser.add_argument("--serial", default="")
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--host", default="")
    parser.add_argument("--instruction", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--android-world-root")
    parser.add_argument("--task")
    parser.add_argument("--task-params-json", default="{}")
    parser.add_argument("--task-seed", type=int, default=113)
    parser.add_argument("--console-port", type=int, default=5560)
    parser.add_argument("--grpc-port", type=int, default=8560)
    parser.add_argument("--no-perform-emulator-setup", action="store_true")
    parser.add_argument("--executor")
    parser.add_argument("--app-name")
    parser.add_argument("--workspace")
    parser.add_argument("--goal", default="")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--server-port", type=int, default=12345)
    parser.add_argument(
        "--handshake-timeout-sec",
        type=float,
        default=MOBILEGPT_HANDSHAKE_TIMEOUT_SEC,
    )
    parser.add_argument("--server-log", default="")
    args = parser.parse_args()
    # Keep the external client boundary on the same formal protocol as the
    # AndroidWorld runner; command-line variants are not experiment variants.
    args.task_seed = TASK_SEED
    # Respect the source-derived budget supplied by the shared atomic runner.
    args.max_steps = min(MAX_STEPS, max(1, int(args.max_steps or MAX_STEPS)))
    args.timeout = float(TASK_DEADLINE_SEC)
    # This module is also callable directly for diagnostics.  Keep that
    # boundary identical to the unified launcher instead of inheriting a
    # stale provider endpoint from the caller's shell.
    os.environ["OPENAI_BASE_URL"] = FORMAL_MODEL_BASE_URL
    os.environ["OPENAI_MODEL"] = require_formal_model()
    if not str(os.environ.get("OPENAI_API_KEY") or "").strip() and str(
        os.environ.get("LLMTHU_API_KEY") or ""
    ).strip():
        os.environ["OPENAI_API_KEY"] = str(os.environ["LLMTHU_API_KEY"])
    os.environ["MOBILEGPT_CHAT_MODEL"] = require_formal_model()
    if args.baseline == "mobilegpt":
        required = {
            "root": args.root,
            "serial": args.serial,
            "host": args.host,
            "instruction": args.instruction,
            "output": args.output,
        }
        missing = [name for name, value in required.items() if not str(value or "").strip()]
        if missing:
            parser.error("mobilegpt arguments required: " + ",".join(missing))
        return run_mobilegpt_client(
            official_root=args.root,
            serial=args.serial,
            adb_path=args.adb,
            host=args.host,
            instruction=args.instruction,
            output_root=args.output,
            timeout_sec=args.timeout,
            android_world_root=args.android_world_root,
            task_name=args.task or "",
            task_params_json=args.task_params_json,
            task_seed=args.task_seed,
            console_port=args.console_port,
            grpc_port=args.grpc_port,
            perform_emulator_setup=not args.no_perform_emulator_setup,
            max_steps=args.max_steps,
            server_port=args.server_port,
            handshake_timeout_sec=args.handshake_timeout_sec,
            server_log_path=args.server_log,
        )
    required = {
        "executor": args.executor,
        "app-name": args.app_name,
        "serial": args.serial,
        "workspace": args.workspace,
        "output": args.output,
        "task": args.task,
        "android-world-root": args.android_world_root,
    }
    missing = [name for name, value in required.items() if not str(value or "").strip()]
    if missing:
        parser.error("appagent arguments required: " + ",".join(missing))
    os.environ["OPENAI_MODEL"] = APPAGENT_MODEL
    return run_appagent_executor(
        python_executable=sys.executable,
        executor=args.executor,
        app_name=args.app_name,
        serial=args.serial,
        workspace=args.workspace,
        goal=args.goal,
        timeout_sec=args.timeout,
        android_world_root=args.android_world_root,
        task_name=args.task,
        task_params_json=args.task_params_json,
        task_seed=args.task_seed,
        console_port=args.console_port,
        grpc_port=args.grpc_port,
        adb_path=args.adb,
        output_root=args.output,
        perform_emulator_setup=not args.no_perform_emulator_setup,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    raise SystemExit(main())
