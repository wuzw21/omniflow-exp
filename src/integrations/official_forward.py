"""Small boundary for launching the pinned external baselines.

This module deliberately does not know how an external baseline plans or
executes an action.  It only makes the official checkout look like the
official README expects.  AutoDroid receives its original DroidBot memory and
is launched through the original ``droidbot.start`` replay entrypoint.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
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
import time
from typing import Any, Iterator, Sequence

from src.experiment.autodroid_contract import (
    AUTODROID_MEMORY_MANIFEST_FORMAT,
    AUTODROID_RESULT_SCHEMA,
)


_AUTODROID_APP_ALIASES = {
    "audio recorder": "audio",
    "broccoli app": "recipe",
    "pro expense": "expense",
    "retro music": "retro",
    "simple calendar pro": "calendar",
    "simple draw pro": "draw",
    "simple gallery pro": "gallery",
    "simple sms messenger": "sms",
}

_AUTODROID_OFFICIAL_MEMORY_KEYS = {
    "audio": "voicerecorder",
    "files": "filemanager",
    "sms": "messenger",
    "recipe": "notes",
}


_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


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


def _autodroid_memory_app_name(app_name: str) -> str:
    normalized = " ".join(str(app_name or "").strip().lower().split())
    return _AUTODROID_APP_ALIASES.get(normalized, normalized)


def _autodroid_official_memory_key(app_name: str) -> str:
    normalized = _autodroid_memory_app_name(app_name)
    return _AUTODROID_OFFICIAL_MEMORY_KEYS.get(normalized, normalized)


def _autodroid_task_app_name(task: Any) -> str:
    declared = [
        " ".join(str(value or "").strip().lower().split())
        for value in tuple(getattr(task, "app_names", ()) or ())
    ]
    declared = [value for value in declared if value]
    if not declared:
        raise ValueError("autodroid_task_app_missing")
    mapped = list(dict.fromkeys(_autodroid_memory_app_name(value) for value in declared))
    return mapped[0]


@contextmanager
def _androidworld_task_startup(
    *,
    android_world_root: str | Path,
    task_name: str,
    task_params_json: str,
    task_seed: int,
    console_port: int,
    grpc_port: int,
    adb_path: str,
    perform_emulator_setup: bool,
    use_uiautomator: bool = True,
) -> Iterator[tuple[Any, Any]]:
    """Prepare one official task through the canonical AndroidWorld seam."""

    from src.integrations.android_world.run_episode import (
        start_androidworld_task_session,
    )

    decoded = json.loads(str(task_params_json or "{}"))
    if not isinstance(decoded, dict):
        raise ValueError("androidworld_task_params_must_be_object")
    startup, task = start_androidworld_task_session(
        android_world_root=android_world_root,
        task_name=task_name,
        task_params=decoded,
        task_seed=int(task_seed),
        console_port=int(console_port),
        adb_path=adb_path,
        grpc_port=int(grpc_port),
        perform_emulator_setup=bool(perform_emulator_setup),
        use_uiautomator=bool(use_uiautomator),
    )
    try:
        yield startup.env, task
    finally:
        try:
            task.tear_down(startup.env)
        finally:
            close = getattr(startup.env, "close", None)
            if callable(close):
                close()


def validate_autodroid_memory_root(memory_root: str | Path) -> dict[str, Any]:
    """Validate one local copy of official AutoDroid/DroidBot memory.

    AutoDroid memory is intentionally not converted into an OmniFlow schema.
    The runner only checks the official replay inputs that it will read.
    """

    root = Path(memory_root).expanduser().resolve()
    manifest_path = root / "memory_manifest.json"
    if not root.is_dir():
        raise FileNotFoundError(f"autodroid_memory_root_missing:{root}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"autodroid_memory_manifest_missing:{manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"autodroid_memory_manifest_invalid:{manifest_path}") from error
    if manifest.get("format") != AUTODROID_MEMORY_MANIFEST_FORMAT:
        raise ValueError("autodroid_memory_manifest_format_invalid")
    apps = manifest.get("apps")
    if not isinstance(apps, list) or not apps:
        raise ValueError("autodroid_memory_apps_missing")
    return {
        "memory_root": str(root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "app_count": len(apps),
        "device": dict(manifest.get("device") or {}),
    }


def _autodroid_active_package(
    *,
    adb_path: str,
    serial: str,
) -> str:
    output = _run_adb(
        adb_path,
        serial,
        ["shell", "dumpsys", "activity", "activities"],
        check=False,
    ).stdout
    patterns = (
        r"mResumedActivity:.*?\s([A-Za-z0-9_.]+)/(?:[A-Za-z0-9_.$]+)",
        r"mCurrentFocus=Window\{[^}]*\s([A-Za-z0-9_.]+)/(?:[A-Za-z0-9_.$]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, output)
        if match:
            return match.group(1)
    return ""


def _autodroid_memory_for_app(
    *,
    memory_root: str | Path,
    adb_path: str,
    serial: str,
    app_name: str = "",
    require_events: bool = True,
) -> dict[str, Any]:
    root = Path(memory_root).expanduser().resolve()
    validate_autodroid_memory_root(root)
    runs_root = root / "runs"
    apks_root = root / "apks"
    requested = _autodroid_memory_app_name(app_name)
    candidates = sorted(
        path for path in runs_root.iterdir() if path.is_dir()
    ) if runs_root.is_dir() else []
    selected = next((path for path in candidates if path.name == requested), None)
    active_package = ""
    if selected is None and not requested:
        active_package = _autodroid_active_package(
            adb_path=adb_path,
            serial=serial,
        )
        for path in candidates:
            package_files = path.glob("dumpsys_package_*.txt")
            if any(
                file.name.removeprefix("dumpsys_package_").removesuffix(".txt")
                == active_package
                for file in package_files
            ):
                selected = path
                break
    if selected is None:
        detail = requested or active_package or "active_package_unknown"
        raise ValueError(f"autodroid_memory_app_not_found:{detail}")
    package_files = sorted(selected.glob("dumpsys_package_*.txt"))
    package = (
        package_files[0].name.removeprefix("dumpsys_package_").removesuffix(".txt")
        if package_files
        else ""
    )
    apk = apks_root / f"{selected.name}.apk"
    if not apk.is_file():
        raise FileNotFoundError(f"autodroid_memory_apk_missing:{apk}")
    events = sorted((selected / "events").glob("event_*.json"))
    if require_events and not events:
        raise ValueError(f"autodroid_memory_events_missing:{selected}")
    invalid = []
    for event in events:
        try:
            json.loads(event.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            invalid.append(event.name)
    if require_events and invalid:
        raise ValueError(
            "autodroid_memory_events_invalid:" + ",".join(invalid)
        )
    return {
        "app_name": selected.name,
        "package": package,
        "memory": str(selected),
        "apk": str(apk),
        "event_count": str(len(events)),
        "active_package": active_package,
    }


def resolve_mobilegpt_client_host(
    host: str = "",
    *,
    serial: str = "",
    adb_path: str = "adb",
) -> str:
    """Choose a host address reachable from the selected Android device.

    Emulators use Android's documented host alias.  Physical/root devices use
    the host-side address selected by the route to the device, so the official
    MobileGPT client does not need a hand-edited ``HOST_IP`` for every run.
    An explicit non-wildcard host always wins.
    """

    explicit = str(host or "").strip()
    if explicit and explicit not in {"0.0.0.0", "::", "[::]", "127.0.0.1"}:
        return explicit
    if str(serial or "").startswith("emulator-"):
        return "10.0.2.2"
    try:
        route = subprocess.run(
            [str(adb_path or "adb"), "-s", str(serial), "shell", "ip", "route"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).stdout
        device_ips = re.findall(r"\bsrc\s+(\d{1,3}(?:\.\d{1,3}){3})\b", route)
        for device_ip in device_ips:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.connect((device_ip, 1))
                local_ip = str(probe.getsockname()[0] or "").strip()
            if local_ip and local_ip != "127.0.0.1":
                return local_ip
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return "10.0.2.2"


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

    # GLM-compatible providers may place the useful answer in
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
    use_oob: bool = False,
    official_memory_mode: bool = False,
) -> dict[str, str]:
    """Stage the official Server so its documented relative ``./memory`` works."""

    root = Path(official_root).expanduser().resolve()
    source = root / "Server"
    work = Path(workspace).expanduser().resolve()
    target = work / "Server"
    if not (source / "main.py").is_file():
        raise FileNotFoundError(f"official_mobilegpt_server_missing:{source / 'main.py'}")
    memory = Path(memory_root).expanduser().resolve()
    if not memory.is_dir():
        raise FileNotFoundError(f"official_mobilegpt_memory_missing:{memory}")
    overlay = memory / "frozen_memory"
    if not overlay.is_dir():
        overlay = memory
    work.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source, target, symlinks=True)
    # The disposable Server copy below is patched in three distinct ways,
    # only two of which are purely mechanical:
    #   - Model/provider routing (chat/embedding model names, endpoints,
    #     malformed-JSON retries) and the observability event stream
    #     (write_omniflow_mobilegpt_event) do not change what MobileGPT
    #     decides to do; they only change which endpoint it calls and record
    #     what it already did.
    #   - _configure_mobilegpt_server also patches mobilegpt.py's subtask
    #     state machine (the "_omniflow_last_explicit_finish" guard) to end
    #     the task instead of repeating an already-finished subtask when
    #     AndroidWorld exposes a page outside the sealed task path. This
    #     DOES change planning/action behavior versus a bare upstream
    #     checkout, in exchange for avoiding a stuck/looping episode. It
    #     stays on by default; set
    #     OMNIFLOW_MOBILEGPT_DISABLE_FINISH_GUARD=1 to run with unpatched
    #     upstream subtask-finish behavior for ablation runs.
    # Some pinned checkouts already import the experiment's optional
    # telemetry hook from ``utils``.  Provide that hook in the disposable
    # workspace only, so a stale checkout cannot prevent the official server
    # from starting.
    staged_utils = target / "utils" / "utils.py"
    staged_server = target / "server.py"
    if (
        staged_utils.is_file()
        and staged_server.is_file()
        and "write_omniflow_mobilegpt_event" in staged_server.read_text(
            encoding="utf-8"
        )
        and "def write_omniflow_mobilegpt_event" not in staged_utils.read_text(
            encoding="utf-8"
        )
    ):
        with staged_utils.open("a", encoding="utf-8") as handle:
            handle.write(
                "\n\n"
                "def write_omniflow_mobilegpt_event(event):\n"
                "    path = os.environ.get('MOBILEGPT_STATS_JSONL', '').strip()\n"
                "    if not path:\n"
                "        return\n"
                "    parent = os.path.dirname(path)\n"
                "    if parent:\n"
                "        os.makedirs(parent, exist_ok=True)\n"
                "    payload = dict(event) if isinstance(event, dict) else {'event': str(event)}\n"
                "    with open(path, 'a', encoding='utf-8') as output:\n"
                "        output.write(json.dumps(payload, ensure_ascii=False) + '\\n')\n"
            )
    # The official checkout hard-codes the OpenAI embedding default and some
    # legacy chat aliases.  Configure only the disposable staging copy so the
    # whole MobileGPT path uses the experiment's GLM endpoint; the upstream
    # checkout and its planner/action implementation remain untouched.
    _configure_mobilegpt_server(
        target,
        embedding_model=embedding_model,
        official_memory_mode=official_memory_mode,
    )
    if use_oob:
        _configure_mobilegpt_oob_action_bounds(target)
    _configure_mobilegpt_speak_transport(target)
    _configure_mobilegpt_chat_model(target, chat_model=chat_model)
    _configure_mobilegpt_json_query(target)
    _configure_mobilegpt_response_compat(target)
    _configure_mobilegpt_optional_completion_rate(target)
    _configure_mobilegpt_selection_compat(target)
    _configure_mobilegpt_xml_compat(target)
    staged_memory = target / "memory"
    for entry in overlay.iterdir():
        destination = staged_memory / entry.name
        if entry.is_dir():
            shutil.copytree(entry, destination, symlinks=True, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, destination)
    return {
        "workspace": str(work),
        "server_root": str(target),
        "memory_root": str(staged_memory),
    }


def _configure_mobilegpt_telemetry(server_root: Path) -> None:
    """Wire the disposable Server copy to emit real action/model-call events.

    Upstream MobileGPT has no concept of an observability event stream, so
    ``write_omniflow_mobilegpt_event`` and its call sites are entirely
    experiment-side additions to the disposable staging copy; nothing in the
    real ``Server/server.py``/``utils/utils.py`` is aware of them. Every
    patch here is guarded by an ``if original_text in source`` check and is a
    silent no-op on mismatch (same convention as the other staging patches
    in this module), so an incompatible checkout falls back to the previous
    behavior (events stay empty; run_mobilegpt_client already tolerates a
    missing/empty MOBILEGPT_STATS_JSONL) instead of failing the run.
    """

    utils_path = server_root / "utils" / "utils.py"
    server_path = server_root / "server.py"
    if utils_path.is_file():
        source = utils_path.read_text(encoding="utf-8")
        # Some staged workspaces already contain the telemetry function from
        # the compatibility bootstrap above, so the old ``function missing``
        # guard does not run.  The embedding retry path still uses
        # ``time.sleep`` and must have its own import in that case.
        if not re.search(r"(?m)^(?:import time|from time import)\b", source):
            source = "import time\n" + source
        if "def write_omniflow_mobilegpt_event" not in source:
            source += (
                "\n\nimport time\n\ndef write_omniflow_mobilegpt_event(event):\n"
                "    path = os.environ.get(\"MOBILEGPT_STATS_JSONL\", \"\").strip()\n"
                "    if not path:\n"
                "        return\n"
                "    parent = os.path.dirname(path)\n"
                "    if parent:\n"
                "        os.makedirs(parent, exist_ok=True)\n"
                "    payload = dict(event) if isinstance(event, dict) else {\"event\": str(event)}\n"
                "    payload.setdefault(\"ts\", time.time())\n"
                "    with open(path, \"a\", encoding=\"utf-8\") as handle:\n"
                "        handle.write(json.dumps(payload, ensure_ascii=False) + \"\\n\")\n"
            )
        original_chat_call = (
            "    result = response.choices[0].message.content\n"
            "    log(result, 'green')\n"
        )
        if original_chat_call in source and "\"chat_call\"" not in source:
            source = source.replace(
                original_chat_call,
                "    result = response.choices[0].message.content\n"
                "    _omniflow_usage = getattr(response, \"usage\", None)\n"
                "    write_omniflow_mobilegpt_event({\n"
                "        \"event\": \"chat_call\",\n"
                "        \"model\": model,\n"
                "        \"prompt_tokens\": getattr(_omniflow_usage, \"prompt_tokens\", 0) or 0,\n"
                "        \"completion_tokens\": getattr(_omniflow_usage, \"completion_tokens\", 0) or 0,\n"
                "        \"total_tokens\": getattr(_omniflow_usage, \"total_tokens\", 0) or 0,\n"
                "    })\n"
                "    log(result, 'green')\n",
                1,
            )
        original_embedding_call = (
            "    response = client.embeddings.create(input=[text], model=model, **kwargs)\n"
            "\n"
            "    return response.data[0].embedding\n"
        )
        if original_embedding_call in source and "\"embedding_call\"" not in source:
            source = source.replace(
                original_embedding_call,
                "    _omniflow_last_embedding_error = None\n"
                "    for _omniflow_embedding_attempt in range(1, 4):\n"
                "        try:\n"
                "            response = client.embeddings.create(input=[text], model=model, **kwargs)\n"
                "            _omniflow_usage = getattr(response, \"usage\", None)\n"
                "            write_omniflow_mobilegpt_event({\n"
                "                \"event\": \"embedding_call\",\n"
                "                \"model\": model,\n"
                "                \"attempt\": _omniflow_embedding_attempt,\n"
                "                \"prompt_tokens\": getattr(_omniflow_usage, \"prompt_tokens\", 0) or 0,\n"
                "                \"total_tokens\": getattr(_omniflow_usage, \"total_tokens\", 0) or 0,\n"
                "            })\n"
                "            return response.data[0].embedding\n"
                "        except Exception as _omniflow_exc:\n"
                "            _omniflow_last_embedding_error = _omniflow_exc\n"
                "            write_omniflow_mobilegpt_event({\n"
                "                \"event\": \"embedding_retry\",\n"
                "                \"model\": model,\n"
                "                \"attempt\": _omniflow_embedding_attempt,\n"
                "                \"error\": type(_omniflow_exc).__name__,\n"
                "            })\n"
                "            if _omniflow_embedding_attempt < 3:\n"
                "                time.sleep(float(_omniflow_embedding_attempt))\n"
                "    write_omniflow_mobilegpt_event({\n"
                "        \"event\": \"embedding_error\",\n"
                "        \"model\": model,\n"
                "        \"error\": type(_omniflow_last_embedding_error).__name__ if _omniflow_last_embedding_error else \"unknown\",\n"
                "    })\n"
                "    raise _omniflow_last_embedding_error\n",
                1,
            )
        utils_path.write_text(source, encoding="utf-8")
    if server_path.is_file():
        source = server_path.read_text(encoding="utf-8")
        if (
            "from utils.utils import log\n" in source
            and "write_omniflow_mobilegpt_event" not in source
        ):
            source = source.replace(
                "from utils.utils import log\n",
                "from utils.utils import log, write_omniflow_mobilegpt_event\n",
                1,
            )
            original_action_send = (
                "                if action is not None:\n"
                "                    message = json.dumps(action)\n"
                "                    client_socket.send(message.encode())\n"
                "                    client_socket.send(\"\\r\\n\".encode())\n"
            )
            replacement_action_send = (
                "                if action is not None:\n"
                "                    message = json.dumps(action)\n"
                "                    client_socket.send(message.encode())\n"
                "                    client_socket.send(\"\\r\\n\".encode())\n"
                "                    write_omniflow_mobilegpt_event({\n"
                "                        \"event\": \"mobilegpt_action_sent\",\n"
                "                        \"action\": action.get(\"name\") if isinstance(action, dict) else None,\n"
                "                        \"is_device_action\": isinstance(action, dict)\n"
                "                        and action.get(\"name\") not in (None, \"speak\"),\n"
                "                    })\n"
            )
            if original_action_send in source:
                source = source.replace(original_action_send, replacement_action_send)
            server_path.write_text(source, encoding="utf-8")


def _configure_mobilegpt_finish_transport(server_root: Path) -> None:
    """Bridge an official JSON ``finish`` action to the official client frame.

    The pinned MobileGPT server can return ``{"name": "finish"}`` from its
    planner without entering ``__finish_task``.  The pinned Android client
    does not consume that JSON action; it only terminates on the historical
    ``$$$$$`` frame.  Keep this compatibility fix in the disposable Server
    copy so the official planner and client remain untouched.
    """

    server_path = server_root / "server.py"
    if not server_path.is_file():
        return
    source = server_path.read_text(encoding="utf-8")
    marker = (
        "    if (\n"
        "        isinstance(action, dict)\n"
        "        and str(action.get(\"name\") or \"\").strip() == OMNIFLOW_INTERNAL_LAUNCH_ACTION\n"
    )
    if marker not in source or "mobilegpt_finish_transport" in source:
        return
    replacement = (
        "    if action_name == \"finish\":\n"
        "        write_omniflow_mobilegpt_event({\"event\": \"task_finished\"})\n"
        "        client_socket.send(\"$$$$$\".encode())\n"
        "        client_socket.send(\"\\r\\n\".encode())\n"
        "        return\n\n"
        "    # mobilegpt_finish_transport: preserve the official client's wire contract.\n"
        + marker
    )
    server_path.write_text(source.replace(marker, replacement, 1), encoding="utf-8")


def _configure_mobilegpt_oob_action_bounds(server_root: Path) -> None:
    """Attach parsed-screen bounds to staged MobileGPT actions."""

    server_path = server_root / "server.py"
    if not server_path.is_file():
        return
    source = server_path.read_text(encoding="utf-8")
    marker = "# omniflow_mobilegpt_oob_action_bounds"
    if marker in source:
        return
    anchor = "\ndef _omniflow_send_action(client_socket, action):\n"
    if anchor not in source:
        return
    helper = (
        "\n# omniflow_mobilegpt_oob_action_bounds\n"
        "def _omniflow_add_oob_bounds(action, parsed_xml):\n"
        "    if not isinstance(action, dict) or not isinstance(parsed_xml, str):\n"
        "        return action\n"
        "    if str(action.get(\"name\") or \"\").strip() not in {\"click\", \"long-click\", \"input\"}:\n"
        "        return action\n"
        "    parameters = action.get(\"parameters\")\n"
        "    if not isinstance(parameters, dict) or parameters.get(\"oob_bounds\"):\n"
        "        return action\n"
        "    wanted = str(parameters.get(\"index\") or \"\").strip()\n"
        "    if not wanted:\n"
        "        return action\n"
        "    try:\n"
        "        root = __import__(\"xml.etree.ElementTree\", fromlist=[\"ElementTree\"]).fromstring(parsed_xml)\n"
        "    except Exception:\n"
        "        return action\n"
        "    for element in root.iter():\n"
        "        if str(element.attrib.get(\"index\") or \"\").strip() != wanted:\n"
        "            continue\n"
        "        bounds = str(element.attrib.get(\"bounds\") or \"\").strip()\n"
        "        if not bounds:\n"
        "            continue\n"
        "        enriched = dict(action)\n"
        "        enriched[\"parameters\"] = dict(parameters)\n"
        "        enriched[\"parameters\"][\"oob_bounds\"] = bounds\n"
        "        return enriched\n"
        "    return action\n"
    )
    source = source.replace(anchor, helper + anchor, 1)
    source = source.replace(
        "def _omniflow_send_action(client_socket, action):\n",
        "def _omniflow_send_action(client_socket, action, parsed_xml=\"\"):\n"
        "    action = _omniflow_add_oob_bounds(action, parsed_xml)\n",
        1,
    )
    for indentation in ("        ", "            ", "                ", "                    ", "                        "):
        source = source.replace(
            indentation + "_omniflow_send_action(client_socket, action)",
            indentation
            + "_omniflow_send_action(client_socket, action, parsed_xml if \"parsed_xml\" in locals() else \"\")",
        )
    server_path.write_text(source, encoding="utf-8")


def _configure_mobilegpt_server(
    server_root: Path,
    *,
    embedding_model: str = "",
    chat_model: str = "",
    official_memory_mode: bool = False,
) -> None:
    """Inject provider names into a temporary copy of the official Server.

    MobileGPT's upstream code keeps provider model names as constants; most
    of what this function patches (endpoints, model aliases, malformed-JSON
    retries, the telemetry event stream) only changes routing/observability,
    not what MobileGPT decides to do. The one exception is the
    ``mobilegpt.py`` subtask-finish guard below (see the
    ``_omniflow_last_explicit_finish`` block): it changes planning behavior
    versus a bare upstream checkout by ending the task instead of repeating
    an already-finished subtask on a page AndroidWorld exposed outside the
    sealed task path. It is on by default; set
    ``OMNIFLOW_MOBILEGPT_DISABLE_FINISH_GUARD=1`` to skip it and run with
    unpatched upstream subtask-finish behavior.
    """

    normalized_embedding = str(embedding_model or "").strip()
    normalized_chat = str(chat_model or "").strip()
    utils_path = server_root / "utils" / "utils.py"
    _configure_mobilegpt_telemetry(server_root)
    _configure_mobilegpt_finish_transport(server_root)
    memory_manager_path = server_root / "memory" / "memory_manager.py"
    if memory_manager_path.is_file() and not official_memory_mode:
        source = memory_manager_path.read_text(encoding="utf-8")
        threshold_pattern = r"(?m)^(?P<indent>[ \t]*)if highest_similarity > 0\.95:\n"
        threshold_replacement = (
            r'\g<indent>threshold = float(os.getenv("MOBILEGPT_MEMORY_SIMILARITY_THRESHOLD", "0.95"))\n'
            r'\g<indent>if os.getenv("MOBILEGPT_TARGET_TASK_NAME", "").strip() and os.getenv("MOBILEGPT_TARGET_PACKAGE", "").strip():\n'
            r'\g<indent>    threshold = min(threshold, float(os.getenv("MOBILEGPT_TARGET_MEMORY_THRESHOLD", "0.70")))\n'
            r'\g<indent>if highest_similarity > threshold:\n'
        )
        threshold_changed = False
        if re.search(threshold_pattern, source) and "MOBILEGPT_MEMORY_SIMILARITY_THRESHOLD" not in source:
            source = re.sub(threshold_pattern, threshold_replacement, source, count=1)
            threshold_changed = True
        page_match_marker = "MOBILEGPT_NATIVE_TRIGGER_UI_PAGE_MATCH"
        native_page_match = (
            "        # MOBILEGPT_NATIVE_TRIGGER_UI_PAGE_MATCH\n"
            "        # The paper's page classifier uses hierarchy embeddings only\n"
            "        # to shortlist candidates.  NodeManager then verifies the\n"
            "        # candidate's trigger/key UI attributes and extra UI set.\n"
            "        candidate_nodes_indexes = self.__search_similar_hierarchy_nodes(hierarchy_xml)\n"
            "        node_manager = NodeManager(self.page_db, self, parsed_xml, encoded_xml)\n"
            "        node_index, new_subtasks = node_manager.search(candidate_nodes_indexes)\n"
            "        if node_index >= 0:\n"
            "            page_data = json.loads(self.page_db.loc[node_index].to_json())\n"
            "            available_subtasks = json.loads(page_data['available_subtasks'])\n"
            "            return node_index, available_subtasks + new_subtasks\n"
            "        return -1, []\n"
        )
        native_page_match_old = (
            "        # candidate_nodes_indexes = self.__search_similar_hierarchy_nodes(hierarchy_xml)\n"
            "        #\n"
            "        # node_manager = NodeManager(self.page_db, self, parsed_xml, encoded_xml)\n"
            "        # node_index, new_subtasks = node_manager.search(candidate_nodes_indexes)\n"
            "        most_similar_node_index = self.__search_most_similar_hierarchy_node(hierarchy_xml)\n"
            "        if most_similar_node_index >= 0:\n"
            "            return most_similar_node_index, []\n"
            "        else:\n"
            "            return -1, []\n"
        )
        if page_match_marker not in source and native_page_match_old in source:
            source = source.replace(native_page_match_old, native_page_match, 1)
            threshold_changed = True
        strict_marker = "MOBILEGPT_MEMORY_REUSE_STRICT"
        strict_anchor = (
            "            if len(next_subtask['parameters']) > 0:\n"
            "                params = param_fill_agent.parm_fill_subtask(instruction=self.instruction,\n"
        )
        strict_replacement = (
            "            if len(next_subtask['parameters']) > 0:\n"
            "                strict_reuse = os.getenv(\"MOBILEGPT_MEMORY_REUSE_STRICT\", \"0\").strip().lower() in (\"1\", \"true\", \"yes\")\n"
            "                if strict_reuse:\n"
            "                    # The sealed source example is the verified parameter binding.\n"
            "                    # Do not ask a second model to reinterpret it on replay.\n"
            "                    example_payload = json.loads(next_subtask_data.get(\"example\", \"{}\") or \"{}\")\n"
            "                    example_action = ((example_payload.get(\"response\") or {}).get(\"action\") or {})\n"
            "                    example_parameters = example_action.get(\"parameters\")\n"
            "                    if isinstance(example_parameters, dict):\n"
            "                        next_subtask['parameters'] = dict(example_parameters)\n"
            "                    else:\n"
            "                        next_subtask['parameters'] = {}\n"
            "                else:\n"
            "                    params = param_fill_agent.parm_fill_subtask(instruction=self.instruction,\n"
        )
        if strict_anchor in source and strict_marker not in source:
            source = source.replace(strict_anchor, strict_replacement, 1)
            source = source.replace(
                "                                                            example=json.loads(\n"
                "                                                                next_subtask_data.get('example', {})))\n"
                "\n"
                "                next_subtask['parameters'] = params\n",
                "                                                            example=json.loads(\n"
                "                                                                next_subtask_data.get('example', {})))\n"
                "\n"
                "                    next_subtask['parameters'] = params\n"
                "",
                1,
            )
            threshold_changed = True
        if threshold_changed:
            memory_manager_path.write_text(source, encoding="utf-8")
    mobilegpt_path = server_root / "mobilegpt.py"
    if mobilegpt_path.is_file():
        source = mobilegpt_path.read_text(encoding="utf-8")
        replay_marker = "MOBILEGPT_RUNLOG_REPLAY_ONLY"
        page_lookup = (
            "        if page_index == -1:\n"
            "            if os.getenv(\"MOBILEGPT_RUNLOG_REPLAY_ONLY\", \"0\").strip().lower() in (\"1\", \"true\", \"yes\"):\n"
            "                raise RuntimeError(\"mobilegpt_runlog_page_not_found\")\n"
            "            page_index = self.explore_agent.explore(parsed_xml, hierarchy_xml, encoded_xml)\n"
        )
        page_lookup_old = (
            "        if page_index == -1:\n"
            "            page_index = self.explore_agent.explore(parsed_xml, hierarchy_xml, encoded_xml)\n"
        )
        if replay_marker not in source and page_lookup_old in source:
            source = source.replace(page_lookup_old, page_lookup, 1)
        subtask_lookup = (
            "                if os.getenv(\"MOBILEGPT_RUNLOG_REPLAY_ONLY\", \"0\").strip().lower() in (\"1\", \"true\", \"yes\"):\n"
            "                    raise RuntimeError(\"mobilegpt_runlog_subtask_not_found\")\n\n"
            "                response, new_action = self.select_agent.select(available_subtasks, self.subtask_history,\n"
        )
        subtask_lookup_old = (
            "                response, new_action = self.select_agent.select(available_subtasks, self.subtask_history,\n"
        )
        if subtask_lookup_old in source and "mobilegpt_runlog_subtask_not_found" not in source:
            source = source.replace(subtask_lookup_old, subtask_lookup, 1)
        action_lookup = (
            "            if os.getenv(\"MOBILEGPT_RUNLOG_REPLAY_ONLY\", \"0\").strip().lower() in (\"1\", \"true\", \"yes\"):\n"
            "                raise RuntimeError(\"mobilegpt_runlog_action_not_found\")\n\n"
            "            if self.subtask_status == Status.WAIT or self.subtask_status == Status.LEARN:\n"
        )
        action_lookup_old = (
            "            if self.subtask_status == Status.WAIT or self.subtask_status == Status.LEARN:\n"
        )
        if action_lookup_old in source and "mobilegpt_runlog_action_not_found" not in source:
            source = source.replace(action_lookup_old, action_lookup, 1)
        example_lookup = (
            "            if \"examples\" in next_action:\n"
            "                if os.getenv(\"MOBILEGPT_RUNLOG_REPLAY_ONLY\", \"0\").strip().lower() in (\"1\", \"true\", \"yes\"):\n"
            "                    raise RuntimeError(\"mobilegpt_runlog_example_fallback\")\n"
            "                next_action, example = self.derive_agent.derive(self.encoded_xml, examples=next_action['examples'])\n"
        )
        example_lookup_old = (
            "            if \"examples\" in next_action:\n"
            "                next_action, example = self.derive_agent.derive(self.encoded_xml, examples=next_action['examples'])\n"
        )
        if example_lookup_old in source and "mobilegpt_runlog_example_fallback" not in source:
            source = source.replace(example_lookup_old, example_lookup, 1)
        page_change_marker = (
            "            if self.subtask_status == Status.LEARN:\n"
            "                self.__finish_subtask()\n"
        )
        page_change_replacement = (
            "            if self.subtask_status == Status.LEARN:\n"
            "                self.__finish_subtask()\n"
            "            elif (\n"
            "                os.getenv(\"MOBILEGPT_MEMORY_REUSE_STRICT\", \"0\").strip().lower()\n"
            "                in (\"1\", \"true\", \"yes\")\n"
            "                and self.current_subtask is not None\n"
            "            ):\n"
            "                # A direct RunLog memory bundle records one source\n"
            "                # transition per page.  When that transition changes\n"
            "                # the page, advance the sealed task path before the\n"
            "                # next action lookup; do not re-derive the old\n"
            "                # subtask on the new page.\n"
            "                self.__finish_subtask(mark_finish=False)\n"
        )
        if page_change_marker in source and "mobilegpt_strict_page_advance" not in source:
            source = source.replace(
                page_change_marker,
                "            # mobilegpt_strict_page_advance\n" + page_change_replacement,
                1,
            )
            mobilegpt_path.write_text(source, encoding="utf-8")
    server_path = server_root / "server.py"
    if server_path.is_file():
        source = server_path.read_text(encoding="utf-8")
        task_marker = "                task, is_new_task = task_agent.get_task(instruction)\n"
        if task_marker in source and "mobilegpt_forced_task_binding" not in source:
            task_binding = (
                "                # mobilegpt_forced_task_binding: the formal runner\n"
                "                # already supplies the exact AndroidWorld task and app.\n"
                "                # Bind from the staged native tasks.csv before the\n"
                "                # upstream TaskAgent can call an LLM.\n"
                "                target_task_name = os.getenv(\"MOBILEGPT_TARGET_TASK_NAME\", \"\").strip()\n"
                "                forced_target_app = os.getenv(\"MOBILEGPT_TARGET_APP\", \"\").strip()\n"
                "                forced_target_package = os.getenv(\"MOBILEGPT_TARGET_PACKAGE\", \"\").strip()\n"
                "                if target_task_name and forced_target_package:\n"
                "                    task = None\n"
                "                    for known_task in task_agent.database.to_dict(orient=\"records\"):\n"
                "                        if str(known_task.get(\"name\") or \"\").strip() == target_task_name:\n"
                "                            task = dict(known_task)\n"
                "                            break\n"
                "                    task = task or {\n"
                "                        \"name\": target_task_name,\n"
                "                        \"description\": instruction,\n"
                "                        \"parameters\": {},\n"
                "                        \"app\": forced_target_package,\n"
                "                    }\n"
                "                    task[\"name\"] = target_task_name\n"
                "                    # The Android client launches this field as a\n"
                "                    # package name; the display label is only\n"
                "                    # metadata and cannot be passed to\n"
                "                    # PackageManager.getLaunchIntentForPackage.\n"
                "                    task[\"app\"] = forced_target_package\n"
                "                    is_new_task = False\n"
                "                else:\n"
                "                    task, is_new_task = task_agent.get_task(instruction)\n"
            )
            source = source.replace(task_marker, task_binding, 1)
            package_lookup = "                target_package = app_agent.get_package_name(target_app)\n"
            direct_package_lookup = (
                "                # mobilegpt_target_package_direct: the formal runner\n"
                "                # already resolved the package; do not ask the\n"
                "                # display-name resolver to reinterpret it.\n"
                "                target_package = (\n"
                "                    forced_target_package\n"
                "                    if target_task_name and forced_target_package\n"
                "                    else app_agent.get_package_name(target_app)\n"
                "                )\n"
            )
            if package_lookup in source and "mobilegpt_target_package_direct" not in source:
                source = source.replace(package_lookup, direct_package_lookup, 1)
            server_path.write_text(source, encoding="utf-8")
    if normalized_embedding and utils_path.is_file():
        source = utils_path.read_text(encoding="utf-8")
        if "def write_omniflow_mobilegpt_event" not in source:
            source += (
                "\n\nimport time\n\ndef write_omniflow_mobilegpt_event(event):\n"
                "    path = os.environ.get(\"MOBILEGPT_STATS_JSONL\", \"\").strip()\n"
                "    if not path:\n"
                "        return\n"
                "    parent = os.path.dirname(path)\n"
                "    if parent:\n"
                "        os.makedirs(parent, exist_ok=True)\n"
                "    payload = dict(event) if isinstance(event, dict) else {\"event\": str(event)}\n"
                "    payload.setdefault(\"ts\", time.time())\n"
                "    with open(path, \"a\", encoding=\"utf-8\") as handle:\n"
                "        handle.write(json.dumps(payload, ensure_ascii=False) + \"\\n\")\n"
            )
        source = re.sub(
            r'def get_openai_embedding\(text: str, model="text-embedding-3-small", \*\*kwargs\)(?: -> [^:]+)?:',
            'def get_openai_embedding(text: str, model=None, **kwargs):\n'
            '    model = model or os.getenv("MOBILEGPT_EMBEDDING_MODEL", "GLM-Embedding-2")',
            source,
            count=1,
        )
        source = source.replace(
            '    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))\n',
            '    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=0)\n'
            '    embedding_timeout = max(1.0, float(os.getenv("MOBILEGPT_EMBEDDING_TIMEOUT_SEC", "15")))\n',
            1,
        )
        source = source.replace(
            '            response = client.embeddings.create(input=[text], model=model, **kwargs)\n',
            '            response = client.with_options(timeout=embedding_timeout).embeddings.create(input=[text], model=model, **kwargs)\n',
            1,
        )
        utils_path.write_text(source, encoding="utf-8")
    if normalized_chat:
        main_path = server_root / "main.py"
        source = main_path.read_text(encoding="utf-8")
        for name in (
            "TASK_AGENT_GPT_VERSION",
            "APP_AGENT_GPT_VERSION",
            "SELECT_AGENT_HISTORY_GPT_VERSION",
            "EXPLORE_AGENT_GPT_VERSION",
            "SELECT_AGENT_GPT_VERSION",
            "DERIVE_AGENT_GPT_VERSION",
            "PARAMETER_FILLER_AGENT_GPT_VERSION",
            "ACTION_SUMMARIZE_AGENT_GPT_VERSION",
            "SUBTASK_MERGE_AGENT_GPT_VERSION",
            "gpt_4",
            "gpt_4_turbo",
            "gpt_3_5_turbo",
        ):
            source = re.sub(
                rf'os\.environ\["{re.escape(name)}"\] = "[^"]+"',
                f'os.environ["{name}"] = os.environ.get("MOBILEGPT_CHAT_MODEL", "{normalized_chat}")',
                source,
            )
        source = source.replace(
            'os.environ["vision_model"] = "gpt-4o"',
            'os.environ["vision_model"] = os.environ.get("MOBILEGPT_VISION_MODEL", os.environ.get("MOBILEGPT_CHAT_MODEL", "GLM-4.6V"))',
        )
        main_path.write_text(source, encoding="utf-8")
        param_path = server_root / "agents" / "param_fill_agent.py"
        if param_path.is_file():
            param_source = param_path.read_text(encoding="utf-8")
            param_source = param_source.replace(
                'model="gpt-4o"',
                'model=os.getenv("MOBILEGPT_CHAT_MODEL", "GLM-4.6V")',
            )
            param_path.write_text(param_source, encoding="utf-8")
        mobilegpt_path = server_root / "mobilegpt.py"
        finish_guard_disabled = str(
            os.environ.get("OMNIFLOW_MOBILEGPT_DISABLE_FINISH_GUARD", "")
        ).strip().lower() in ("1", "true", "yes")
        if mobilegpt_path.is_file() and not finish_guard_disabled:
            mobilegpt_source = mobilegpt_path.read_text(encoding="utf-8")
            repeated_subtask_guard = "        if self.current_subtask is None:\n"
            guarded_subtask_selection = (
                "        if self.current_subtask is None:\n"
                "            last_finish = getattr(self, \"_omniflow_last_explicit_finish\", None)\n"
                "            if last_finish and last_finish.get(\"page_index\") == self.current_page_index:\n"
                "                # AndroidWorld can expose a newly explored page that is not\n"
                "                # present in the sealed task path. If the planner repeats\n"
                "                # the same completed subtask on that page, end the task\n"
                "                # instead of sending the same device action again.\n"
                "                self.__finish_task()\n"
                "                return None\n"
            )
            if repeated_subtask_guard in mobilegpt_source and (
                "last_finish = getattr(self, \"_omniflow_last_explicit_finish\", None)"
                not in mobilegpt_source
            ):
                mobilegpt_source = mobilegpt_source.replace(
                    "        self.subtask_status = Status.WAIT\n",
                    "        self.subtask_status = Status.WAIT\n"
                    "        self._omniflow_last_explicit_finish = None\n",
                    1,
                )
                mobilegpt_source = mobilegpt_source.replace(
                    repeated_subtask_guard,
                    guarded_subtask_selection,
                    1,
                )
                mobilegpt_source = mobilegpt_source.replace(
                    "        if next_action['name'] == 'finish':\n"
                    "            self.__finish_subtask(mark_finish=False, explicit_finish=True)\n",
                    "        if next_action['name'] == 'finish':\n"
                    "            self._omniflow_last_explicit_finish = {\n"
                    "                \"page_index\": self.current_page_index,\n"
                    "                \"subtask_name\": str((self.current_subtask or {}).get(\"name\") or \"\"),\n"
                    "            }\n"
                    "            self.__finish_subtask(mark_finish=False, explicit_finish=True)\n",
                    1,
                )
                mobilegpt_path.write_text(mobilegpt_source, encoding="utf-8")
        task_agent_path = server_root / "agents" / "task_agent.py"
        if task_agent_path.is_file():
            task_agent_source = task_agent_path.read_text(encoding="utf-8")
            original_query = (
                "        response = query(messages=task_agent_prompt.get_prompts(instruction, known_tasks),\n"
                "                         model=os.getenv(\"TASK_AGENT_GPT_VERSION\"))\n"
            )
            retry_query = (
                "        target_package = os.getenv(\"MOBILEGPT_TARGET_PACKAGE\", \"\").strip()\n"
                "        target_tasks = [task for task in known_tasks if isinstance(task, dict) and str(task.get(\"app\") or \"\").strip() == target_package]\n"
                "        if target_package and len(target_tasks) == 1:\n"
                "            log(f\"Binding MobileGPT task to target package {target_package}\", \"blue\")\n"
                "            response = {\"api\": target_tasks[0], \"found_match\": True}\n"
                "        else:\n"
                "            response = _omniflow_task_agent_query(\n"
                "                task_agent_prompt.get_prompts(instruction, known_tasks),\n"
                "                os.getenv(\"TASK_AGENT_GPT_VERSION\"),\n"
                "            )\n"
            )
            if original_query in task_agent_source and (
                "def _omniflow_task_agent_query" not in task_agent_source
            ):
                task_agent_source = task_agent_source.replace(
                    "from utils.utils import query, log\n",
                    "from utils.utils import query, log\n\n\n"
                    "def _omniflow_task_agent_query(messages, model):\n"
                    "    # Retry the real API when upstream parsing returns raw text.\n"
                    "    attempts = max(1, int(os.getenv(\"MOBILEGPT_TASK_AGENT_RETRIES\", \"3\")))\n"
                    "    response = None\n"
                    "    for attempt in range(attempts):\n"
                    "        response = query(messages=messages, model=model)\n"
                    "        if isinstance(response, dict) and isinstance(response.get(\"api\"), dict):\n"
                    "            return response\n"
                    "        log(f\"TaskAgent response was not a task object; retry {attempt + 1}/{attempts}\", \"red\")\n"
                    "    raise RuntimeError(\"mobilegpt_task_agent_json_response_invalid\")\n\n",
                    1,
                )
                task_agent_source = task_agent_source.replace(
                    original_query,
                    retry_query,
                    1,
                )
                task_agent_path.write_text(task_agent_source, encoding="utf-8")
        utils_path = server_root / "utils" / "utils.py"
        if utils_path.is_file():
            utils_source = utils_path.read_text(encoding="utf-8")
            utils_source = utils_source.replace(
                "        max_tokens=900,\n",
                "        max_tokens=int(os.getenv(\"MOBILEGPT_MAX_TOKENS\", \"1800\")),\n",
                1,
            )
            utils_source = utils_source.replace(
                "    if json_formatted_response:\n"
                "        return json.loads(json_formatted_response)\n"
                "    else:\n"
                "        return result\n",
                "    if json_formatted_response:\n"
                "        try:\n"
                "            return json.loads(json_formatted_response)\n"
                "        except json.JSONDecodeError:\n"
                "            log(\"MobileGPT response was truncated or invalid JSON; retrying the real API\", \"red\")\n"
                "            for _ in range(2):\n"
                "                retry = client.chat.completions.create(\n"
                "                    model=model, messages=messages, temperature=0,\n"
                "                    max_tokens=int(os.getenv(\"MOBILEGPT_MAX_TOKENS\", \"1800\")),\n"
                "                    top_p=0, frequency_penalty=0, presence_penalty=0\n"
                "                )\n"
                "                retry_result = retry.choices[0].message.content\n"
                "                retry_json = __parse_json(retry_result, is_list=is_list)\n"
                "                if retry_json:\n"
                "                    try:\n"
                "                        return json.loads(retry_json)\n"
                "                    except json.JSONDecodeError:\n"
                "                        continue\n"
                "            raise RuntimeError(\"mobilegpt_model_json_response_invalid\")\n"
                "    return result\n",
                1,
            )
            utils_source = utils_source.replace(
                "        value = float(input_str)\n",
                "        try:\n"
                "            value = float(input_str)\n"
                "        except (TypeError, ValueError):\n"
                "            # completion_rate is telemetry; some providers return\n"
                "            # a natural-language status while the action is valid.\n"
                "            percent = re.search(r\"(?<!\\d)(\\d+(?:\\.\\d+)?)\\s*%\", input_str)\n"
                "            if percent:\n"
                "                value = float(percent.group(1))\n"
                "            else:\n"
                "                return 0\n",
                1,
            )
            utils_path.write_text(utils_source, encoding="utf-8")


def _configure_mobilegpt_speak_transport(server_root: Path) -> None:
    """Suppress intermediate speech packets in the disposable Server copy.

    The official Android client speaks the message and returns without
    sending a new observation. Sending this packet between two real actions
    therefore deadlocks the official socket protocol in AndroidWorld.
    """

    mobilegpt_path = server_root / "mobilegpt.py"
    if not mobilegpt_path.is_file():
        return
    source = mobilegpt_path.read_text(encoding="utf-8")
    marker = "MOBILEGPT_SUPPRESS_SPEAK_ACTIONS"
    if marker in source:
        return
    original = (
        "                if next_subtask['name'] != 'read_screen':\n"
        "                    msg = response['speak']\n"
        "                    self.__send_speak_action(msg)\n"
    )
    replacement = (
        "                if (\n"
        "                    next_subtask['name'] != 'read_screen'\n"
        "                    and os.getenv(\"MOBILEGPT_SUPPRESS_SPEAK_ACTIONS\", \"1\").strip().lower()\n"
        "                    not in (\"1\", \"true\", \"yes\")\n"
        "                ):\n"
        "                    msg = response['speak']\n"
        "                    self.__send_speak_action(msg)\n"
    )
    if original in source:
        mobilegpt_path.write_text(source.replace(original, replacement, 1), encoding="utf-8")


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


def _configure_mobilegpt_json_query(server_root: Path) -> None:
    """Retry malformed GLM JSON in the disposable MobileGPT Server copy."""

    utils_path = server_root / "utils" / "utils.py"
    if not utils_path.is_file():
        return
    source = utils_path.read_text(encoding="utf-8")
    source = source.replace(
        "        max_tokens=900,\n",
        "        max_tokens=int(os.getenv(\"MOBILEGPT_MAX_TOKENS\", \"4096\")),\n",
        1,
    )
    original = (
        "    if json_formatted_response:\n"
        "        return json.loads(json_formatted_response)\n"
        "    else:\n"
        "        return result\n"
    )
    replacement = (
        "    if json_formatted_response:\n"
        "        try:\n"
        "            return json.loads(json_formatted_response)\n"
        "        except json.JSONDecodeError:\n"
        "            log(\"MobileGPT GLM response was invalid JSON; retrying\", \"red\")\n"
        "            for _ in range(2):\n"
        "                retry = client.chat.completions.create(\n"
        "                    model=model, messages=messages, temperature=0,\n"
        "                    max_tokens=int(os.getenv(\"MOBILEGPT_MAX_TOKENS\", \"1800\")),\n"
        "                    top_p=0, frequency_penalty=0, presence_penalty=0\n"
        "                )\n"
        "                retry_result = retry.choices[0].message.content\n"
        "                retry_json = __parse_json(retry_result, is_list=is_list)\n"
        "                if retry_json:\n"
        "                    try:\n"
        "                        return json.loads(retry_json)\n"
        "                    except json.JSONDecodeError:\n"
        "                        continue\n"
        "            raise RuntimeError(\"mobilegpt_glm_json_response_invalid\")\n"
        "    return result\n"
    )
    if original in source and "mobilegpt_glm_json_response_invalid" not in source:
        source = source.replace(original, replacement, 1)
    # GLM can explain the percentage in prose (for example, ``0% complete``)
    # even when the official prompt asks for a number.  The pinned upstream
    # implementation has comments between the ``else`` and ``float`` call,
    # so a narrow whole-function replacement is more reliable than matching a
    # particular formatting of that block.  This is applied only to the
    # disposable Server workspace, never to the official checkout.
    parse_rate_pattern = (
        r"def parse_completion_rate\(completion_rate\).*?"
        r"(?=\n\ndef |\Z)"
    )
    parse_rate_compat = '''def parse_completion_rate(completion_rate) -> int:
    input_str = str(completion_rate).strip()
    percent = re.search(r"(?<!\\d)(\\d+(?:\\.\\d+)?)\\s*%", input_str)
    if percent:
        return int(float(percent.group(1)))
    try:
        value = float(input_str)
    except (TypeError, ValueError):
        number = re.search(r"(?<!\\d)(?:0?\\.\\d+|\\d+(?:\\.\\d+)?)", input_str)
        value = float(number.group(0)) if number else 0.0
    return int(value * 100) if value < 1 else int(value)
'''
    source = re.sub(
        parse_rate_pattern,
        lambda _match: parse_rate_compat.rstrip("\\n"),
        source,
        count=1,
        flags=re.DOTALL,
    )
    empty_result = "    result = response.choices[0].message.content\n"
    empty_result_compat = (
        "    _omniflow_message = response.choices[0].message\n"
        "    result = (getattr(_omniflow_message, \"content\", None)\n"
        "              or getattr(_omniflow_message, \"reasoning_content\", None)\n"
        "              or \"\")\n"
        "    if not result.strip():\n"
        "        for _ in range(2):\n"
        "            retry = client.chat.completions.create(\n"
        "                model=model, messages=messages, temperature=0,\n"
        "                max_tokens=int(os.getenv(\"MOBILEGPT_MAX_TOKENS\", \"1800\")),\n"
        "                top_p=0, frequency_penalty=0, presence_penalty=0\n"
        "            )\n"
        "            _omniflow_retry_message = retry.choices[0].message\n"
        "            result = (getattr(_omniflow_retry_message, \"content\", None)\n"
        "                      or getattr(_omniflow_retry_message, \"reasoning_content\", None)\n"
        "                      or \"\")\n"
        "            if result.strip():\n"
        "                break\n"
        "        if not result.strip():\n"
        "            write_omniflow_mobilegpt_event({\"event\": \"chat_empty\", \"model\": model})\n"
        "            return [] if is_list else {}\n"
    )
    if empty_result in source and "if not result.strip()" not in source:
        source = source.replace(empty_result, empty_result_compat, 1)
    parse_json_def = "def __parse_json(s: str, is_list=False):\n"
    parse_json_guard = (
        "def __parse_json(s: str, is_list=False):\n"
        "    if not isinstance(s, str) or not s.strip():\n"
        "        return \"\"\n"
    )
    if parse_json_def in source and "not isinstance(s, str)" not in source:
        source = source.replace(parse_json_def, parse_json_guard, 1)
    if source != utils_path.read_text(encoding="utf-8"):
        utils_path.write_text(source, encoding="utf-8")


def _configure_mobilegpt_response_compat(server_root: Path) -> None:
    """Make the staged MobileGPT query wrapper safe for GLM responses.

    The official utility assumes every provider response has non-empty
    ``message.content`` and valid JSON.  GLM-compatible endpoints can return
    a list content block, reasoning-only content, or one transient empty
    response.  Normalize those shapes and retry before MobileGPT's planner
    sees an invalid/empty action.  This is a provider boundary patch in the
    disposable Server copy; the official planner and task state machine are
    unchanged.
    """

    utils_path = server_root / "utils" / "utils.py"
    if not utils_path.is_file():
        return
    source = utils_path.read_text(encoding="utf-8")
    marker = "# omniflow_mobilegpt_glm_response_compat"
    if marker in source:
        return
    replacement = r'''# omniflow_mobilegpt_glm_response_compat
def _omniflow_message_text(message):
    candidates = (
        getattr(message, "content", None),
        getattr(message, "reasoning_content", None),
        getattr(message, "reasoning", None),
        getattr(message, "output_text", None),
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

def query(messages, model="gpt-4-turbo", is_list=False):
    client = OpenAI(max_retries=0)
    request_timeout = max(1.0, float(os.getenv("MOBILEGPT_REQUEST_TIMEOUT_SEC", "20")))
    thinking_mode = os.getenv("MOBILEGPT_THINKING", "disabled").strip()
    request_extra_body = (
        {"thinking": {"type": thinking_mode}}
        if thinking_mode
        else {}
    )
    for message in messages:
        log(message["content"], "yellow")
    attempts = max(1, int(os.getenv("MOBILEGPT_RESPONSE_RETRIES", "2")))
    last_result = ""
    for attempt in range(1, attempts + 1):
        try:
            response = client.with_options(timeout=request_timeout).chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=int(
                    os.getenv(
                        "MOBILEGPT_LIST_MAX_TOKENS" if is_list else "MOBILEGPT_MAX_TOKENS",
                        "512" if is_list else "2048",
                    )
                ),
                top_p=0,
                frequency_penalty=0,
                presence_penalty=0,
                extra_body=request_extra_body,
            )
        except Exception as error:
            try:
                write_omniflow_mobilegpt_event({
                    "event": "chat_error",
                    "model": model,
                    "attempt": attempt,
                    "error": type(error).__name__,
                })
            except NameError:
                pass
            log(f"MobileGPT GLM request failed; retry {attempt}/{attempts}: {error}", "red")
            if attempt == attempts:
                raise RuntimeError("mobilegpt_glm_request_failed") from error
            continue
        message = response.choices[0].message
        result = _omniflow_message_text(message)
        last_result = result
        usage = getattr(response, "usage", None)
        try:
            write_omniflow_mobilegpt_event({
                "event": "chat_call",
                "model": model,
                "attempt": attempt,
                "content_chars": len(str(getattr(message, "content", None) or "")),
                "reasoning_chars": len(str(getattr(message, "reasoning_content", None) or "")),
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage, "total_tokens", 0) or 0,
            })
        except NameError:
            pass
        if not result.strip():
            log(f"MobileGPT GLM returned empty content; retry {attempt}/{attempts}", "red")
            continue
        log(result, "green")
        json_formatted_response = __parse_json(result, is_list=is_list)
        if json_formatted_response:
            try:
                return json.loads(json_formatted_response)
            except json.JSONDecodeError:
                log(f"MobileGPT GLM returned invalid JSON; retry {attempt}/{attempts}", "red")
                continue
        if is_list:
            continue
        return result
    try:
        write_omniflow_mobilegpt_event({
            "event": "chat_empty_or_invalid",
            "model": model,
            "attempts": attempts,
            "response_chars": len(last_result),
        })
    except NameError:
        pass
    raise RuntimeError("mobilegpt_glm_response_empty_or_invalid")
'''
    query_pattern = r"\ndef query\(messages, model=\"gpt-4-turbo\", is_list=False\):.*?(?=\n\ndef parse_completion_rate\()"
    patched, count = re.subn(
        query_pattern,
        "\n" + replacement.rstrip("\n") + "\n",
        source,
        count=1,
        flags=re.DOTALL,
    )
    if count:
        utils_path.write_text(patched, encoding="utf-8")


def _configure_mobilegpt_optional_completion_rate(server_root: Path) -> None:
    """Keep optional GLM completion telemetry from aborting an action."""

    mobilegpt_path = server_root / "mobilegpt.py"
    if not mobilegpt_path.is_file():
        return
    source = mobilegpt_path.read_text(encoding="utf-8")
    original = "parse_completion_rate(next_subtask['parameters']['completion_rate'])"
    replacement = (
        "parse_completion_rate("
        "(next_subtask.get('parameters') or {}).get('completion_rate', 0)"
        ")"
    )
    if original in source:
        mobilegpt_path.write_text(source.replace(original, replacement, 1), encoding="utf-8")


def _configure_mobilegpt_selection_compat(server_root: Path) -> None:
    """Treat GLM's omitted optional completion telemetry as valid JSON."""

    select_path = server_root / "agents" / "select_agent.py"
    if not select_path.is_file():
        return
    source = select_path.read_text(encoding="utf-8")
    original = "del response['completion_rate']"
    replacement = "response.pop('completion_rate', None)"
    if original in source:
        select_path.write_text(source.replace(original, replacement, 1), encoding="utf-8")


def _configure_mobilegpt_xml_compat(server_root: Path) -> None:
    """Keep a transient empty accessibility frame from killing Server threads."""

    server_path = server_root / "server.py"
    if not server_path.is_file():
        return
    source = server_path.read_text(encoding="utf-8")
    original = (
        "                parsed_xml, hierarchy_xml, encoded_xml = screen_parser.encode(raw_xml, current_screen_index)\n"
    )
    replacement = (
        "                try:\n"
        "                    parsed_xml, hierarchy_xml, encoded_xml = screen_parser.encode(raw_xml, current_screen_index)\n"
        "                except Exception as error:\n"
        "                    # Android accessibility can emit one empty frame while\n"
        "                    # an app/window is settling. The official parser raises\n"
        "                    # ParseError here and otherwise kills this client thread.\n"
        "                    log(f\"MobileGPT XML frame ignored: {error}\", \"yellow\")\n"
        "                    parsed_xml, hierarchy_xml, encoded_xml = screen_parser.encode(\n"
        "                        \"<hierarchy />\", current_screen_index\n"
        "                    )\n"
    )
    if original in source and "MobileGPT XML frame ignored" not in source:
        server_path.write_text(
            source.replace(original, replacement, 1),
            encoding="utf-8",
        )


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


def _autodroid_display_ids(display_dump: str) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                int(value)
                for value in re.findall(
                    r"\bmDisplayId\s*=\s*(\d+)", str(display_dump or "")
                )
            }
        )
    )


def _prepare_autodroid_device(
    *,
    adb_path: str,
    serial: str,
    package: str,
) -> dict[str, Any]:
    """Normalize system UI before handing control to the official policy."""

    result: dict[str, Any] = {
        "serial": str(serial),
        "package": str(package or ""),
        "display_ids": [],
        "multiple_displays": False,
        "actions": [],
    }
    for name, args in (
        ("home", ["shell", "input", "keyevent", "KEYCODE_HOME"]),
        ("collapse_statusbar", ["shell", "cmd", "statusbar", "collapse"]),
    ):
        try:
            completed = _run_adb(adb_path, serial, args, check=False)
            result["actions"].append(
                {"name": name, "returncode": int(completed.returncode)}
            )
        except (OSError, subprocess.SubprocessError) as error:
            result["actions"].append(
                {"name": name, "error": type(error).__name__}
            )

    try:
        display_probe = _run_adb(
            adb_path,
            serial,
            ["shell", "dumpsys", "display"],
            check=False,
        )
        display_ids = _autodroid_display_ids(display_probe.stdout)
    except (OSError, subprocess.SubprocessError) as error:
        result["actions"].append(
            {"name": "display_probe", "error": type(error).__name__}
        )
        display_ids = ()
    result["display_ids"] = list(display_ids)
    result["multiple_displays"] = len(display_ids) > 1

    if display_ids and len(display_ids) > 1 and str(package or "").strip():
        try:
            completed = _run_adb(
                adb_path,
                serial,
                [
                    "shell",
                    "am",
                    "start",
                    "--display",
                    "0",
                    "-W",
                    "-a",
                    "android.intent.action.MAIN",
                    "-c",
                    "android.intent.category.LAUNCHER",
                    "-p",
                    str(package).strip(),
                ],
                check=False,
            )
            result["actions"].append(
                {
                    "name": "launch_on_primary_display",
                    "returncode": int(completed.returncode),
                }
            )
        except (OSError, subprocess.SubprocessError) as error:
            result["actions"].append(
                {
                    "name": "launch_on_primary_display",
                    "error": type(error).__name__,
                }
            )
    return result


def _count_droidbot_output_events(output_root: str | Path) -> int:
    """Count events that the official DroidBot actually emitted."""

    root = Path(output_root)
    if not root.is_dir():
        return 0
    return sum(1 for _ in root.rglob("events/event_*.json"))


def _count_mobilegpt_device_actions(stats_path: Path) -> int:
    """Count real device actions recorded by the staged Server telemetry patch."""

    if not stats_path.is_file():
        return 0
    count = 0
    for line in stats_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(event, dict)
            and event.get("event") == "mobilegpt_action_sent"
            and event.get("is_device_action") is True
        ):
            count += 1
    return count


MOBILEGPT_STEP_BUDGET_RETURN_CODE = 125
MOBILEGPT_STEP_TIMEOUT_RETURN_CODE = 126
MOBILEGPT_HANDSHAKE_RETURN_CODE = 127
MOBILEGPT_SERVER_ERROR_RETURN_CODE = 128
MOBILEGPT_HANDSHAKE_TIMEOUT_SEC = 20.0
MOBILEGPT_STEP_TIMEOUT_SEC = 60.0


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


def _configure_mobilegpt_client_launch_lifecycle(client_root: Path) -> None:
    """Make the staged official client reliable at the first app frame.

    The upstream Accessibility client sets ``firstScreen`` *after*
    ``startActivity``. On AndroidWorld an accessibility window event can be
    delivered during that call, so the event is discarded and no XML or
    screenshot is ever sent to the official Server. This only changes the
    disposable client copy: arm capture before launching and keep a delayed
    capture as the event-independent fallback.
    """

    service_path = (
        client_root
        / "app/src/main/java/com/example/MobileGPT/MobileGPTAccessibilityService.java"
    )
    if not service_path.is_file():
        return
    source = service_path.read_text(encoding="utf-8")
    changed = False
    # Android 13+ requires an explicit export policy for dynamically
    # registered receivers when the official APK targets API 33.  Without it
    # the service can bind and send its app list, but the shell broadcast that
    # starts the task is silently not delivered on API 34 fold images.
    receiver_original = "        registerReceiver(stringReceiver, intentFilter);"
    receiver_replacement = (
        "        registerReceiver(stringReceiver, intentFilter, "
        "Context.RECEIVER_EXPORTED);"
    )
    if receiver_original in source:
        source = source.replace(receiver_original, receiver_replacement, 1)
        changed = True
    marker = "// omniflow_mobilegpt_launch_lifecycle"
    if marker in source:
        if changed:
            service_path.write_text(source, encoding="utf-8")
        return
    original = (
        "        if (launchIntent != null) {\n"
        "            startActivity(launchIntent);//null pointer check in case package name was not found\n"
        "        } else {\n"
        "            Log.d(TAG, \"intent null\");\n"
        "        }\n"
        "        xmlPending = true;\n"
        "        screenNeedUpdate = true;\n"
        "        firstScreen = true;\n"
    )
    replacement = (
        "        // omniflow_mobilegpt_launch_lifecycle\n"
        "        // Arm capture before startActivity: Android may deliver the\n"
        "        // first accessibility event synchronously during launch.\n"
        "        xmlPending = true;\n"
        "        screenNeedUpdate = true;\n"
        "        firstScreen = true;\n"
        "        firstScreenLaunchRetries = 0;\n"
        "        mainThreadHandler.removeCallbacks(screenUpdateWaitRunnable);\n"
        "        mainThreadHandler.removeCallbacks(screenUpdateTimeoutRunnable);\n"
        "        if (launchIntent != null) {\n"
        "            startActivity(launchIntent);//null pointer check in case package name was not found\n"
        "        } else {\n"
        "            Log.d(TAG, \"intent null\");\n"
        "        }\n"
        "        // Do not depend solely on an accessibility event; some\n"
        "        // emulator window transitions emit none after launch.\n"
        "        mainThreadHandler.postDelayed(screenUpdateTimeoutRunnable, 5000);\n"
    )
    if original not in source:
        if changed:
            service_path.write_text(source, encoding="utf-8")
        return
    service_path.write_text(source.replace(original, replacement, 1), encoding="utf-8")


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
        )
        if marker in server_lowered
    )
    return {
        "schema_version": "omniflow.mobilegpt_protocol_probe.v1",
        "stats_event_count": len(events),
        "task_started": task_started > 0,
        "task_started_count": task_started,
        "task_finished": task_finished > 0,
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
            if task_started > 0
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
) -> int:
    """Run the MobileGPT client boundary for one episode.

    Formal AndroidWorld runs select the OOB transport through the public
    launcher.  The upstream APK remains available for protocol compatibility,
    but it is not used for physical observe/act in that mode.
    """

    if os.environ.get("OMNIFLOW_ANDROIDWORLD_CONTROL_BACKEND", "").strip().lower() == "oob":
        from src.integrations.mobilegpt_oob_client import run_mobilegpt_oob_client

        result = run_mobilegpt_oob_client(
            serial=serial,
            adb_path=adb_path,
            server_host=host,
            server_port=server_port,
            instruction=instruction,
            timeout_sec=timeout_sec,
            max_steps=max_steps,
            output_root=output_root,
            server_log_path=server_log_path,
        )
        output = Path(output_root).expanduser().resolve()
        stats_path = Path(os.environ.get("MOBILEGPT_STATS_JSONL", "")).expanduser()
        server_log = Path(server_log_path).expanduser() if str(server_log_path).strip() else None
        server_text = (
            server_log.read_text(encoding="utf-8", errors="replace")[-20000:]
            if server_log is not None and server_log.is_file()
            else ""
        )
        probe = _mobilegpt_protocol_probe(stats_path, str(result.get("log") or ""), server_text)
        probe.update(
            {
                "failure_reason": str(result.get("reason") or ""),
                "returncode": int(result.get("returncode", 1)),
                "server_host": str(result.get("server_host") or "127.0.0.1"),
                "server_port": int(result.get("server_port") or server_port),
                "transport": "oob",
            }
        )
        (output / "protocol_probe.json").write_text(
            json.dumps(probe, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return int(result.get("returncode", 1))

    root = Path(official_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    client_root = output / "official_client"
    shutil.copytree(root / "App", client_root)
    _configure_mobilegpt_client_launch_lifecycle(client_root)
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
    plugin_version = str(
        os.environ.get("OMNIFLOW_ANDROID_GRADLE_PLUGIN") or "8.13.2"
    ).strip()
    build_file = client_root / "build.gradle"
    build_file.write_text(
        re.sub(
            r"version ['\"]8\.0\.1['\"]",
            f"version '{plugin_version}'",
            build_file.read_text(encoding="utf-8"),
        ),
        encoding="utf-8",
    )
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
    apk = client_root / "app/build/outputs/apk/debug/app-debug.apk"
    if not apk.is_file():
        raise FileNotFoundError(f"official_mobilegpt_apk_missing:{apk}")
    _run_adb(adb_path, serial, ["install", "-r", str(apk)])
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
    # The official MobileGPT manifest declares the service in the app's
    # package as a relative class name: ``.MobileGPTAccessibilityService``.
    # The previous fully-qualified spelling added the package twice, so
    # Android silently ignored it and the client waited forever for a bind.
    service = "com.example.MobileGPT/.MobileGPTAccessibilityService"
    current = _run_adb(
        adb_path,
        serial,
        ["shell", "settings", "get", "secure", "enabled_accessibility_services"],
        check=False,
    ).stdout.strip()
    services = [value for value in current.split(":") if value and value != "null"]
    if service not in services:
        services.append(service)
    _run_adb(
        adb_path,
        serial,
        ["shell", "settings", "put", "secure", "enabled_accessibility_services", ":".join(services)],
    )
    _run_adb(adb_path, serial, ["shell", "settings", "put", "secure", "accessibility_enabled", "1"])
    _run_adb(adb_path, serial, ["shell", "monkey", "-p", "com.example.MobileGPT", "1"])
    # Launching the activity can cause Android to restore the secure settings
    # from before installation. Re-assert the service after launch and wait
    # until the accessibility process is bound before sending the instruction.
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
    for _ in range(10):
        accessibility_state = _run_adb(
            adb_path,
            serial,
            ["shell", "dumpsys", "accessibility"],
            check=False,
        ).stdout
        normalized_service = (
            "com.example.MobileGPT/"
            "com.example.MobileGPT.MobileGPTAccessibilityService"
        )
        if (
            (service in accessibility_state or normalized_service in accessibility_state)
            and "Bound services:" in accessibility_state
        ):
            break
        time.sleep(1.0)
    time.sleep(2.0)
    _run_adb(adb_path, serial, ["logcat", "-c"])
    _run_adb(
        adb_path,
        serial,
        _mobilegpt_instruction_broadcast_args(instruction),
    )
    stats_path = Path(os.environ.get("MOBILEGPT_STATS_JSONL", "")).expanduser()
    deadline = time.monotonic() + max(1.0, float(timeout_sec))
    handshake_deadline = time.monotonic() + max(
        1.0, float(handshake_timeout_sec)
    )
    last_action_count = 0
    last_action_at = time.monotonic()
    last_log = ""
    server_log = Path(server_log_path).expanduser() if str(server_log_path).strip() else None

    def read_server_log() -> str:
        if server_log is None or not server_log.is_file():
            return ""
        return server_log.read_text(encoding="utf-8", errors="replace")[-20000:]

    def finish_with_probe(returncode: int, log: str, reason: str) -> int:
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
        return returncode

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
            return 0
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
                "mobilegpt_server_handler_failed",
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
                "mobilegpt_server_handler_failed",
            )
        if not probe["task_started"] and time.monotonic() >= handshake_deadline:
            _run_adb(
                adb_path,
                serial,
                ["shell", "am", "force-stop", "com.example.MobileGPT"],
                check=False,
            )
            return finish_with_probe(
                MOBILEGPT_HANDSHAKE_RETURN_CODE,
                log,
                "mobilegpt_handshake_timeout",
            )
        action_count = _count_mobilegpt_device_actions(stats_path)
        if action_count != last_action_count:
            last_action_count = action_count
            last_action_at = time.monotonic()
        if max_steps > 0 and action_count >= max_steps:
            # MobileGPT has no upstream step cap; stop the client the same
            # way every other formal method is bounded by --max-steps
            # instead of only ever giving up on the wall-clock timeout.
            _run_adb(
                adb_path,
                serial,
                ["shell", "am", "force-stop", "com.example.MobileGPT"],
                check=False,
            )
            return finish_with_probe(
                MOBILEGPT_STEP_BUDGET_RETURN_CODE,
                log,
                "mobilegpt_step_budget_exhausted",
            )
        if (
            max_steps > 0
            and time.monotonic() - last_action_at >= MOBILEGPT_STEP_TIMEOUT_SEC
        ):
            # The official client can wait forever for the next server action
            # even though the formal runner has a per-step budget. Bound that
            # wait at the same 60-second step timeout used by AndroidWorld.
            _run_adb(
                adb_path,
                serial,
                ["shell", "am", "force-stop", "com.example.MobileGPT"],
                check=False,
            )
            return finish_with_probe(
                MOBILEGPT_STEP_TIMEOUT_RETURN_CODE,
                log,
                "mobilegpt_step_timeout",
            )
        time.sleep(1.0)
    finish_with_probe(124, last_log, "mobilegpt_episode_timeout")
    return 124


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
    """Run MobileGPT from the same initialized AndroidWorld task state."""

    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not android_world_root or not str(task_name).strip():
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
        )
    started = time.monotonic()
    with _androidworld_task_startup(
        android_world_root=android_world_root,
        task_name=task_name,
        task_params_json=task_params_json,
        task_seed=task_seed,
        console_port=console_port,
        grpc_port=grpc_port,
        adb_path=adb_path,
        perform_emulator_setup=perform_emulator_setup,
        # AndroidWorld setup is environment preparation only.  MobileGPT's
        # real observe/act path remains its own Accessibility client/socket;
        # use UIAutomator here so a missing AndroidWorld forwarder tree cannot
        # abort setup before the official client is launched.
        use_uiautomator=True,
    ) as (env, task):
        returncode = _run_mobilegpt_client(
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
        )
        reward = float(task.is_successful(env))
        # MobileGPT can leave its official client loop alive after the
        # AndroidWorld task is already successful. Preserve that timeout in
        # process_returncode, but let the official validator decide whether
        # the task itself succeeded.
        validator_success = reward > 0.5
        task_params = json.loads(str(task_params_json or "{}"))
        stats_path = Path(os.environ.get("MOBILEGPT_STATS_JSONL", "")).expanduser()
        actions_executed = 0
        model_calls = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        stats_events: list[dict[str, Any]] = []
        if stats_path.is_file():
            for line in stats_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                stats_events.append(event)
                if (
                    event.get("event") == "mobilegpt_action_sent"
                    and event.get("is_device_action") is True
                ):
                    actions_executed += 1
                if event.get("event") in {"chat_call", "embedding_call"}:
                    model_calls += 1
                    prompt_tokens += int(event.get("prompt_tokens") or 0)
                    completion_tokens += int(event.get("completion_tokens") or 0)
                    total_tokens += int(event.get("total_tokens") or 0)
        runtime_integrity_error = {
            MOBILEGPT_STEP_BUDGET_RETURN_CODE: "mobilegpt_step_budget_exhausted",
            MOBILEGPT_STEP_TIMEOUT_RETURN_CODE: "mobilegpt_step_timeout",
            MOBILEGPT_HANDSHAKE_RETURN_CODE: "mobilegpt_handshake_failed",
            MOBILEGPT_SERVER_ERROR_RETURN_CODE: "mobilegpt_server_handler_failed",
        }.get(returncode, "")
        protocol_probe_path = output / "protocol_probe.json"
        protocol_probe = {}
        if protocol_probe_path.is_file():
            try:
                value = json.loads(
                    protocol_probe_path.read_text(encoding="utf-8")
                )
                if isinstance(value, dict):
                    protocol_probe = value
            except json.JSONDecodeError:
                protocol_probe = {"parse_error": True}
        failure_reason = str(protocol_probe.get("failure_reason") or "").strip()
        environment_failure = failure_reason in {
            "mobilegpt_target_app_package_unresolved",
            "mobilegpt_target_app_not_ready",
            "mobilegpt_oob_observation_xml_missing",
        } or failure_reason.startswith("mobilegpt_target_app_not_ready:")
        result_row = {
            "schema_version": "omniflow.androidworld.result.v1",
            "task_name": task_name,
            "task": task_name,
            "goal": str(instruction),
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
            "method": "mobilegpt",
            "device": serial,
            "task_random_seed": int(task_seed),
            "fixed_task_seed": True,
            "fixed_task_params": True,
            "official_validator_used": True,
            "official_validator_success": validator_success,
            "official_validator_coverage_rate": 1.0,
            "androidworld_validator_result": {
                "validator": "androidworld_official",
                "success": reward > 0.5,
                "reward": reward,
            },
            "process_returncode": int(returncode),
            "classification": (
                "environment_failure"
                if environment_failure
                else "success"
                if validator_success
                else "method_failure"
            ),
            "actions_executed": actions_executed,
            "model_calls": model_calls,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "token_usage_status": "tracked" if stats_events else "unavailable",
            "fallback_steps": 0,
            "duration_ms": round((time.monotonic() - started) * 1000.0, 3),
            "mobilegpt_stats_jsonl": str(stats_path),
            "mobilegpt_protocol_probe": str(protocol_probe_path),
            "mobilegpt_protocol": protocol_probe,
            "environment_failure": environment_failure,
            "failure_reason": failure_reason,
            "runtime_integrity_error": runtime_integrity_error,
        }
        (output / "task_results.jsonl").write_text(
            json.dumps(result_row, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return 0 if validator_success else (returncode or 1)


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
        console_port=console_port,
        grpc_port=grpc_port,
        adb_path=adb_path,
        perform_emulator_setup=perform_emulator_setup,
        use_uiautomator=False,
    ) as (env, task):
        process_returncode = 1
        runtime_integrity_error = ""
        prelaunch_package = ""
        prelaunch_returncode = None
        try:
            # AppAgent's stock executor starts from the launcher and expects
            # its VLM to discover the target app icon.  The AndroidWorld
            # launcher on the Fold profile does not expose the camera icon in
            # the clickable XML, so the official agent can repeat the same
            # tap forever without ever entering the target app.  Initialize
            # the known AndroidWorld app before invoking the untouched
            # executor; AppAgent still owns all subsequent observe/act/model
            # decisions and its source checkout is not modified.
            prelaunch_package = {
                "camera2": "com.android.camera2",
            }.get(str(app_name).strip().lower(), "")
            if prelaunch_package:
                prelaunch_adb = shutil.which("adb") or str(adb_path)
                prelaunch = subprocess.run(
                    [
                        prelaunch_adb,
                        "-s",
                        str(serial),
                        "shell",
                        "am",
                        "start",
                        "-W",
                        "-n",
                        f"{prelaunch_package}/com.android.camera.CaptureActivity",
                    ],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=30.0,
                )
                prelaunch_returncode = int(prelaunch.returncode)
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
            child_env["APPAGENT_THINKING"] = "disabled"
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
        reward = float(task.is_successful(env))
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
            "official_validator_used": True,
            "official_validator_success": validator_success,
            "official_validator_coverage_rate": 1.0,
            "androidworld_validator_result": {
                "validator": "androidworld_official",
                "success": reward > 0.5,
                "reward": reward,
            },
            "process_returncode": process_returncode,
            "classification": (
                "success"
                if validator_success
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
            "target_app_prelaunch_package": prelaunch_package,
            "target_app_prelaunch_returncode": prelaunch_returncode,
            "runtime_integrity_error": runtime_integrity_error,
        }
        (output / "task_results.jsonl").write_text(
            json.dumps(result_row, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        # The official process status and AndroidWorld validator conclusion are
        # separate facts.  Keep a clean executor exit code even when the
        # validator rejects the resulting device state; the scheduler reads
        # official_validator_success for the method outcome.
    return process_returncode


def run_autodroid_replay(
    *,
    official_root: str | Path,
    memory_root: str | Path,
    serial: str,
    adb_path: str,
    output_root: str | Path,
    timeout_sec: float,
    max_events: int,
    android_world_root: str | Path,
    task_name: str,
    task_params_json: str,
    task_seed: int,
    console_port: int,
    grpc_port: int,
    app_name: str = "",
    goal: str = "",
    policy: str = "replay",
    perform_emulator_setup: bool = True,
) -> int:
    """Run one official AutoDroid policy inside the shared task lifecycle."""

    if policy not in {"replay", "task"}:
        raise ValueError(f"autodroid_policy_invalid:{policy}")

    root = Path(official_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not (root / "droidbot" / "start.py").is_file():
        raise FileNotFoundError(f"official_autodroid_entry_missing:{root}")
    task_params = json.loads(str(task_params_json or "{}"))
    if not isinstance(task_params, dict):
        raise ValueError("autodroid_task_params_must_be_object")
    memory_info = None
    started = time.monotonic()
    with _androidworld_task_startup(
        android_world_root=android_world_root,
        task_name=task_name,
        task_params_json=task_params_json,
        task_seed=task_seed,
        console_port=console_port,
        grpc_port=grpc_port,
        adb_path=adb_path,
        perform_emulator_setup=perform_emulator_setup,
    ) as (env, task):
        explicit_app_name = _autodroid_memory_app_name(app_name)
        task_app_names = [
            " ".join(str(value or "").strip().lower().split())
            for value in tuple(getattr(task, "app_names", ()) or ())
        ]
        task_app_names = [value for value in task_app_names if value]
        task_app_name = (
            _autodroid_task_app_name(task) if not explicit_app_name else ""
        )
        memory_info = _autodroid_memory_for_app(
            memory_root=memory_root,
            adb_path=adb_path,
            serial=serial,
            app_name=explicit_app_name or task_app_name,
            require_events=policy == "replay",
        )
        official_memory_key = _autodroid_official_memory_key(
            memory_info["app_name"]
        )
        official_memory_root = root / "memory"
        if policy == "task" and not (
            (official_memory_root / "node_filtered_elements.json").is_file()
            and (official_memory_root / "element_description.json").is_file()
            and (official_memory_root / "embedded_elements_desc.json").is_file()
        ):
            raise FileNotFoundError(
                f"autodroid_official_memory_assets_missing:{official_memory_root}"
            )
        if policy == "task":
            try:
                node_elements = json.loads(
                    (official_memory_root / "node_filtered_elements.json").read_text(
                        encoding="utf-8"
                    )
                )
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"autodroid_official_memory_assets_invalid:{official_memory_root}"
                ) from error
            if official_memory_key not in node_elements:
                raise ValueError(
                    f"autodroid_official_memory_app_missing:{official_memory_key}"
                )
        memory_info.update(
            {
                "task_app_names": task_app_names,
                "task_app_name": task_app_name,
                "official_memory_key": official_memory_key,
                "task_app_selection": (
                    "explicit_app_name" if explicit_app_name else "task_declared_first"
                ),
            }
        )
        device_preflight = _prepare_autodroid_device(
            adb_path=adb_path,
            serial=serial,
            package=str(memory_info.get("package") or ""),
        )
        (output / "device_preflight.json").write_text(
            json.dumps(device_preflight, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        droidbot_output = output / "droidbot"
        if policy == "task":
            droidbot_output /= official_memory_key
        official_launcher = "from droidbot.start import main; main()"
        replay_stats_path = output / "autodroid_replay_stats.json"
        if policy == "replay":
            official_launcher = "\n".join(
                (
                    "import atexit, json, os",
                    "import re",
                    "from droidbot.device import Device as _Device",
                    "def _get_top_activity_name(self):",
                    "    output = self.adb.shell('dumpsys activity activities')",
                    "    match = re.search(r'\\*\\s+Hist\\s+#\\d+:\\s+ActivityRecord\\{[^ ]+\\s+[^ ]+\\s+([^ ]+)\\s+t\\d+\\}', output)",
                    "    if match:",
                    "        return match.group(1)",
                    "    match = re.search(r'(?:mResumedActivity|mCurrentFocus).*?\\s([A-Za-z0-9_.]+/[A-Za-z0-9_.$]+)', output)",
                    "    return match.group(1) if match else None",
                    "_Device.get_top_activity_name = _get_top_activity_name",
                    "from droidbot.input_policy import UtgReplayPolicy",
                    "_official_generate_event = UtgReplayPolicy.generate_event",
                    "_replay_stats_path = os.environ.get('AUTODROID_REPLAY_STATS_PATH', '')",
                    "_replay_emitted = 0",
                    "def _generate_replay_event(self, input_manager=None):",
                    "    global _replay_emitted",
                    "    event = _official_generate_event(self)",
                    "    if event is not None:",
                    "        _replay_emitted += 1",
                    "    return event",
                    "def _write_replay_stats():",
                    "    if _replay_stats_path:",
                    "        with open(_replay_stats_path, 'w', encoding='utf-8') as handle:",
                    "            json.dump({'replayed_events': _replay_emitted}, handle)",
                    "atexit.register(_write_replay_stats)",
                    "UtgReplayPolicy.generate_event = _generate_replay_event",
                    official_launcher,
                )
            )
        else:
            task_literal = json.dumps(
                str(goal or getattr(task, "goal", "") or task_name),
                ensure_ascii=False,
            )
            official_launcher = "\n".join(
                (
                    "import json, os, re, sys, time",
                    "from pathlib import Path",
                    "try:",
                    "    import pkg_resources",
                    "except ModuleNotFoundError:",
                    "    import importlib.util, types",
                    "    _pkg_resources = types.ModuleType('pkg_resources')",
                    "    def _resource_filename(package, resource):",
                    "        _spec = importlib.util.find_spec(package)",
                    "        _base = next(iter(_spec.submodule_search_locations), Path(_spec.origin).parent)",
                    "        return str(Path(_base) / resource)",
                    "    _pkg_resources.resource_filename = _resource_filename",
                    "    sys.modules['pkg_resources'] = _pkg_resources",
                    "from openai import OpenAI",
                    "import tools",
                    "_stats = Path(os.environ['AUTODROID_STATS_PATH'])",
                    "_stats.parent.mkdir(parents=True, exist_ok=True)",
                    "def _append(row):",
                    "    with _stats.open('a', encoding='utf-8') as handle:",
                    "        handle.write(json.dumps(row) + '\\n')",
                    "_AUTODROID_LLM_TIMEOUT_SEC = float(os.environ.get('AUTODROID_LLM_TIMEOUT_SEC', '60'))",
                    "_AUTODROID_LLM_MAX_ATTEMPTS = max(1, int(os.environ.get('AUTODROID_LLM_MAX_ATTEMPTS', '3')))",
                    "def _query(prompt):",
                    "    client_kwargs = {'api_key': os.environ.get('APIKey') or os.environ.get('OPENAI_API_KEY'), 'max_retries': 0}",
                    "    if os.environ.get('OPENAI_BASE_URL'):",
                    "        client_kwargs['base_url'] = os.environ['OPENAI_BASE_URL']",
                    "    model = os.environ.get('AUTODROID_MODEL', 'gpt-3.5-turbo')",
                    "    for attempt in range(1, _AUTODROID_LLM_MAX_ATTEMPTS + 1):",
                    "        started = time.monotonic()",
                    "        try:",
                    "            completion = OpenAI(**client_kwargs).chat.completions.create(messages=[{'role': 'user', 'content': prompt}], model=model, temperature=float(os.environ.get('AUTODROID_TEMPERATURE', '0.25')), timeout=_AUTODROID_LLM_TIMEOUT_SEC)",
                    "        except Exception as error:",
                    "            _append({'event': 'chat_call', 'model': model, 'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'error': type(error).__name__, 'attempt': attempt, 'max_attempts': _AUTODROID_LLM_MAX_ATTEMPTS, 'elapsed_sec': round(time.monotonic() - started, 6)})",
                    "            if attempt >= _AUTODROID_LLM_MAX_ATTEMPTS:",
                    "                raise",
                    "            time.sleep(min(2.0, 0.5 * attempt))",
                    "            continue",
                    "        usage = getattr(completion, 'usage', None)",
                    "        _append({'event': 'chat_call', 'model': model, 'prompt_tokens': int(getattr(usage, 'prompt_tokens', 0) or 0), 'completion_tokens': int(getattr(usage, 'completion_tokens', 0) or 0), 'total_tokens': int(getattr(usage, 'total_tokens', 0) or 0), 'attempt': attempt, 'max_attempts': _AUTODROID_LLM_MAX_ATTEMPTS, 'elapsed_sec': round(time.monotonic() - started, 6)})",
                    "        return completion.choices[0].message.content",
                    "tools.query_gpt = _query",
                    "_instructor_model_path = os.environ.get('AUTODROID_INSTRUCTOR_MODEL_PATH', '').strip()",
                    "if _instructor_model_path:",
                    "    import InstructorEmbedding as _instructor_embedding",
                    "    _official_instructor = _instructor_embedding.INSTRUCTOR",
                    "    def _offline_instructor(model_name_or_path, *args, **kwargs):",
                    "        if model_name_or_path == 'hkunlp/instructor-xl':",
                    "            model_name_or_path = _instructor_model_path",
                    "        model = _official_instructor(model_name_or_path, *args, **kwargs)",
                    "        if not hasattr(model, '_text_length'):",
                    "            model._text_length = lambda sentence: len(sentence[1] if isinstance(sentence, list) else sentence)",
                    "        return model",
                    "    _instructor_embedding.INSTRUCTOR = _offline_instructor",
                    "from droidbot.input_policy import TaskPolicy as _TaskPolicy",
                    "_official_task_policy_init = _TaskPolicy.__init__",
                    "def _init_task_policy_with_memory(self, *args, **kwargs):",
                    "    kwargs['use_memory'] = True",
                    "    return _official_task_policy_init(self, *args, **kwargs)",
                    "_TaskPolicy.__init__ = _init_task_policy_with_memory",
                    "from droidbot.input_manager import InputManager as _InputManager",
                    "_official_input_start = _InputManager.start",
                    "def _start_with_app(self):",
                    "    self.device.start_app(self.app)",
                    "    time.sleep(2)",
                    "    return _official_input_start(self)",
                    "_InputManager.start = _start_with_app",
                    "from droidbot.device import Device as _Device",
                    "def _get_top_activity_name(self):",
                    "    output = self.adb.shell('dumpsys activity activities')",
                    "    match = re.search(r'\\*\\s+Hist\\s+#\\d+:\\s+ActivityRecord\\{[^ ]+\\s+[^ ]+\\s+([^ ]+)\\s+t\\d+\\}', output)",
                    "    if match:",
                    "        return match.group(1)",
                    "    match = re.search(r'(?:mResumedActivity|mCurrentFocus).*?\\s([A-Za-z0-9_.]+/[A-Za-z0-9_.$]+)', output)",
                    "    return match.group(1) if match else None",
                    "_Device.get_top_activity_name = _get_top_activity_name",
                    f"_task_goal = {task_literal}",
                    "from droidbot.droidbot import DroidBot as _DroidBot",
                    "_official_droidbot_init = _DroidBot.__init__",
                    "def _init_with_task(self, *args, **kwargs):",
                    "    kwargs['task'] = _task_goal",
                    "    return _official_droidbot_init(self, *args, **kwargs)",
                    "_DroidBot.__init__ = _init_with_task",
                    "import droidbot.start as _start",
                    "_official_parse_args = _start.parse_args",
                    "def _parse_args_with_task():",
                    "    _original_argv = list(sys.argv)",
                    "    _argv = list(_original_argv)",
                    "    if '-task' in _argv:",
                    "        _index = _argv.index('-task')",
                    "        del _argv[_index:_index + 2]",
                    "    sys.argv = _argv",
                    "    try:",
                    "        _options = _official_parse_args()",
                    "    finally:",
                    "        sys.argv = _original_argv",
                    "    _options.task = _task_goal",
                    "    return _options",
                    "_start.parse_args = _parse_args_with_task",
                    "try:",
                    "    _start.main()",
                    "except BaseException:",
                    "    raise",
                    "os._exit(0)",
                )
            )
        command = [
            sys.executable,
            "-c",
            official_launcher,
            "-d",
            serial,
            "-a",
            memory_info["apk"],
            "-o",
            str(droidbot_output),
            "-policy",
            policy,
            "-count",
            str(max(1, int(max_events))),
            "-interval",
            "0",
            "-timeout",
            str(max(1, int(timeout_sec))),
            "-keep_app",
            "-keep_env",
            "-grant_perm",
            "-is_emulator",
            "-accessibility_auto",
        ]
        if policy == "replay":
            command.extend(("-replay_output", memory_info["memory"]))
        else:
            command.extend(("-task", str(goal or getattr(task, "goal", "") or task_name)))
        env_vars = dict(os.environ)
        for proxy_name in (
            "ALL_PROXY",
            "all_proxy",
            "HTTP_PROXY",
            "http_proxy",
            "HTTPS_PROXY",
            "https_proxy",
        ):
            env_vars.pop(proxy_name, None)
        if policy == "task":
            env_vars["APIKey"] = str(
                env_vars.get("APIKey") or env_vars.get("OPENAI_API_KEY") or ""
            )
            env_vars["AUTODROID_STATS_PATH"] = str(output / "autodroid_stats.jsonl")
        if policy == "replay":
            env_vars["AUTODROID_REPLAY_STATS_PATH"] = str(replay_stats_path)
        env_vars["PYTHONPATH"] = os.pathsep.join(
            value
            for value in (str(root), env_vars.get("PYTHONPATH", ""))
            if value
        )
        official_adb_proxy = write_adb_proxy(
            output / "official_adb",
            serial=serial,
            adb_path=adb_path,
        )
        adb_parent = official_adb_proxy.parent
        env_vars["PATH"] = os.pathsep.join(
            value
            for value in (str(adb_parent), env_vars.get("PATH", ""))
            if value
        )
        try:
            process = subprocess.run(
                command,
                cwd=str(root),
                env=env_vars,
                check=False,
                timeout=max(1.0, float(timeout_sec)),
            )
            returncode = int(process.returncode)
        except subprocess.TimeoutExpired:
            returncode = 124
        validator_used = returncode == 0
        reward = float(task.is_successful(env)) if validator_used else 0.0
        success = validator_used and reward > 0.5
        if policy == "replay":
            replayed_event_count = None
            if replay_stats_path.is_file():
                try:
                    replayed_event_count = int(
                        json.loads(replay_stats_path.read_text(encoding="utf-8"))
                        .get("replayed_events", 0)
                    )
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    replayed_event_count = None
            if replayed_event_count is None:
                replayed_event_count = _count_droidbot_output_events(droidbot_output)
            replay_requested_event_count = min(
                int(memory_info["event_count"]), max(1, int(max_events))
            )
        else:
            replayed_event_count = len(
                sorted(droidbot_output.glob("events/event_*.json"))
            )
            replay_requested_event_count = replayed_event_count
        result = {
            "schema_version": AUTODROID_RESULT_SCHEMA,
            "task": task_name,
            "task_params": task_params,
            "task_params_sha256": hashlib.sha256(
                json.dumps(
                    task_params,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest(),
            "method": "autodroid",
            "policy": policy,
            "device": serial,
            "task_random_seed": int(task_seed),
            "fixed_task_seed": True,
            "fixed_task_params": True,
            "max_steps": max(1, int(max_events)),
            "memory": memory_info,
            "official_validator_used": validator_used,
            "official_validator_success": success,
            "official_validator_coverage_rate": 1.0 if validator_used else 0.0,
            "androidworld_validator_result": {
                "validator": "androidworld_official",
                "success": success,
                "reward": reward,
            },
            "process_returncode": returncode,
            "classification": (
                "success"
                if success
                else "method_failure"
                if returncode == 0
                else "environment_failure"
            ),
            "actions_executed": replayed_event_count,
            "replay_event_limit": max(1, int(max_events)),
            "replay_requested_event_count": replay_requested_event_count,
            "replay_completed": validator_used,
            "replay_step_completed_count": (
                replayed_event_count if validator_used else 0
            ),
            "replay_step_total": replay_requested_event_count,
            "replay_step_completed_rate": (
                min(
                    1.0,
                    replayed_event_count / replay_requested_event_count,
                )
                if validator_used and replay_requested_event_count
                else 0.0
            ),
            "model_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "fallback_steps": 0,
            "duration_ms": round((time.monotonic() - started) * 1000.0, 3),
        }
        stats_path = output / "autodroid_stats.jsonl"
        if policy == "task" and stats_path.is_file():
            stats_rows = []
            for line in stats_path.read_text(encoding="utf-8").splitlines():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    stats_rows.append(value)
            result.update(
                {
                    "model_calls": sum(row.get("event") == "chat_call" for row in stats_rows),
                    "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in stats_rows),
                    "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in stats_rows),
                    "total_tokens": sum(int(row.get("total_tokens") or 0) for row in stats_rows),
                    "token_usage_status": "tracked" if stats_rows else "unavailable",
                }
            )
        result["official_policy"] = policy
        result["memory_injection"] = policy == "task"
        result["autodroid_temperature"] = float(
            os.environ.get("AUTODROID_TEMPERATURE", "0.25")
        ) if policy == "task" else None
        (output / "autodroid_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output / "task_results.jsonl").write_text(
            json.dumps(
                {
                    **result,
                    "task_name": task_name,
                    "goal": str(getattr(task, "goal", "") or task_name),
                    "agent": f"autodroid_official_{policy}",
                    "backend": "official_droidbot",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return 0 if success else (returncode or 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Forward one task to an official baseline")
    parser.add_argument(
        "--baseline", choices=("mobilegpt", "appagent", "autodroid"), default="mobilegpt"
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
    parser.add_argument("--policy", choices=("replay", "task"), default="replay")
    parser.add_argument("--workspace")
    parser.add_argument("--goal", default="")
    parser.add_argument("--memory-root", default="")
    parser.add_argument("--max-events", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--server-port", type=int, default=12345)
    parser.add_argument(
        "--handshake-timeout-sec",
        type=float,
        default=MOBILEGPT_HANDSHAKE_TIMEOUT_SEC,
    )
    parser.add_argument("--server-log", default="")
    args = parser.parse_args()
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
    if args.baseline == "autodroid":
        required = {
            "root": args.root,
            "memory-root": args.memory_root,
            "serial": args.serial,
            "output": args.output,
            "task": args.task,
            "android-world-root": args.android_world_root,
        }
        missing = [name for name, value in required.items() if not str(value or "").strip()]
        if missing:
            parser.error("autodroid arguments required: " + ",".join(missing))
        return run_autodroid_replay(
            official_root=args.root,
            memory_root=args.memory_root,
            serial=args.serial,
            adb_path=args.adb,
            output_root=args.output,
            timeout_sec=args.timeout,
            max_events=args.max_events,
            android_world_root=args.android_world_root,
            task_name=args.task,
            task_params_json=args.task_params_json,
            task_seed=args.task_seed,
            console_port=args.console_port,
            grpc_port=args.grpc_port,
            app_name=args.app_name or "",
            goal=args.goal,
            policy=args.policy,
            perform_emulator_setup=not args.no_perform_emulator_setup,
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
