"""Small boundary for launching the pinned external baselines.

This module makes the official MobileGPT and AppAgent checkouts look like
their upstream launchers expect.
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
import tempfile
import time
from typing import Any, Iterator, Sequence


_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


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
    console_port: int,
    grpc_port: int,
    adb_path: str,
    perform_emulator_setup: bool,
    use_uiautomator: bool = True,
) -> Iterator[tuple[Any, Any]]:
    """Prepare one official task through the canonical AndroidWorld seam."""

    from android_world.env import adb_utils

    from src.integrations.android_world.run_episode import (
        _patch_androidworld_current_activity,
        start_androidworld_task_session,
    )

    decoded = json.loads(str(task_params_json or "{}"))
    if not isinstance(decoded, dict):
        raise ValueError("androidworld_task_params_must_be_object")
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
    write_through_memory: bool = False,
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
    # Patch only provider routing, wire compatibility and telemetry in the
    # disposable Server copy. MobileGPT's memory lookup, task selection,
    # planner and action state machine remain upstream behavior.
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
    )
    _configure_mobilegpt_chat_model(target, chat_model=chat_model)
    _configure_mobilegpt_json_query(target)
    _configure_mobilegpt_response_compat(target)
    _configure_mobilegpt_optional_completion_rate(target)
    _configure_mobilegpt_selection_compat(target)
    _configure_mobilegpt_system_app_catalog(target)
    _configure_mobilegpt_target_package_fallback(target)
    _configure_mobilegpt_client_error_transport(target)
    staged_memory = target / "memory"
    if write_through_memory:
        if any(memory.iterdir()):
            raise ValueError(
                f"official_mobilegpt_cold_memory_not_empty:{memory}"
            )
        # Upstream keeps both its Python memory package and the learned CSV
        # graph under ``Server/memory``.  Seed the empty persistent directory
        # with that exact staged package before linking it back into Server;
        # otherwise replacing the directory with an empty symlink removes
        # ``memory.memory_manager`` and the official server cannot import.
        shutil.copytree(staged_memory, memory, symlinks=True, dirs_exist_ok=True)
        if staged_memory.is_symlink() or staged_memory.is_file():
            staged_memory.unlink()
        elif staged_memory.is_dir():
            shutil.rmtree(staged_memory)
        staged_memory.symlink_to(memory, target_is_directory=True)
    else:
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


def _configure_mobilegpt_runtime_guards(server_root: Path) -> None:
    """Keep official Explore alive when a model returns a stale UI index.

    The upstream parser assumes every ``trigger_UIs`` index returned by the
    Explore model exists in the current XML.  A cross-device screen can make
    that assumption false; ``get_ui_key_attrib`` then dereferences ``None``
    and kills the Server handler thread.  The staged copy skips only that
    stale trigger and records it, preserving the official planner/executor.
    """

    parsing_path = server_root / "utils" / "parsing_utils.py"
    if not parsing_path.is_file():
        return
    source = parsing_path.read_text(encoding="utf-8")
    marker = "omniflow_mobilegpt_invalid_trigger_ui"
    if marker in source:
        return
    source = source.replace(
        "from utils.utils import log\n",
        "from utils.utils import log, write_omniflow_mobilegpt_event\n",
        1,
    )
    original = (
        "            ui_attributes = get_ui_key_attrib(int(ui_index), screen)\n"
        "\n"
        "            skip = False\n"
    )
    replacement = (
        "            try:\n"
        "                ui_attributes = get_ui_key_attrib(int(ui_index), screen)\n"
        "            except (AttributeError, TypeError, ValueError):\n"
        "                write_omniflow_mobilegpt_event({\n"
        "                    \"event\": \"omniflow_mobilegpt_invalid_trigger_ui\",\n"
        "                    \"ui_index\": str(ui_index),\n"
        "                    \"subtask_name\": str(subtask_name),\n"
        "                })\n"
        "                continue\n"
        "\n"
        "            skip = False\n"
    )
    if original not in source:
        return
    parsing_path.write_text(source.replace(original, replacement, 1), encoding="utf-8")


def _configure_mobilegpt_server_port(server_root: Path) -> None:
    """Make the disposable official Server honor its per-device port.

    The pinned MobileGPT ``main.py`` hard-codes ``12345`` and ignores the
    environment passed by the launcher.  That is harmless for one device but
    prevents the scheduler from starting multiple official Servers.  Patch
    only the staged copy; the upstream Server source remains unchanged.
    """

    main_path = server_root / "main.py"
    if not main_path.is_file():
        return
    source = main_path.read_text(encoding="utf-8")
    marker = "MOBILEGPT_SERVER_PORT"
    if marker in source:
        return
    original = (
        '    server_ip = "0.0.0.0"\n'
        "    server_port = 12345\n"
    )
    replacement = (
        '    server_ip = os.getenv("MOBILEGPT_SERVER_HOST", "0.0.0.0")\n'
        '    server_port = int(os.getenv("MOBILEGPT_SERVER_PORT", "12345"))\n'
    )
    if original not in source:
        return
    main_path.write_text(source.replace(original, replacement, 1), encoding="utf-8")


def _configure_mobilegpt_memory_telemetry(server_root: Path) -> None:
    """Record official Memory recall/Explore decisions in the stats stream."""

    mobilegpt_path = server_root / "mobilegpt.py"
    if not mobilegpt_path.is_file():
        return
    source = mobilegpt_path.read_text(encoding="utf-8")
    marker = "omniflow_mobilegpt_memory_telemetry"
    if marker in source:
        return
    source = source.replace(
        "from utils.utils import log, parse_completion_rate\n",
        "from utils.utils import (\n"
        "    log, parse_completion_rate, write_omniflow_mobilegpt_event,\n"
        ")\n\n"
        "# omniflow_mobilegpt_memory_telemetry\n",
        1,
    )
    original_lookup = (
        "        page_index, new_subtasks = self.memory.search_node(parsed_xml, hierarchy_xml, encoded_xml)\n"
        "\n"
        "        if page_index == -1:\n"
        "            page_index = self.explore_agent.explore(parsed_xml, hierarchy_xml, encoded_xml)\n"
    )
    replacement_lookup = (
        "        page_index, new_subtasks = self.memory.search_node(parsed_xml, hierarchy_xml, encoded_xml)\n"
        "        memory_lookup_result = (\"direct_hit\" if page_index >= 0 else \"explore\")\n"
        "        write_omniflow_mobilegpt_event({\n"
        "            \"event\": \"memory_lookup\",\n"
        "            \"result\": memory_lookup_result,\n"
        "            \"page_index\": int(page_index),\n"
        "        })\n"
        "\n"
        "        if page_index == -1:\n"
        "            page_index = self.explore_agent.explore(parsed_xml, hierarchy_xml, encoded_xml)\n"
    )
    if original_lookup not in source:
        return
    source = source.replace(original_lookup, replacement_lookup, 1)
    original_action = (
        "        next_action = self.memory.get_next_action(self.current_subtask, self.encoded_xml)\n"
        "        current_action_data = {\"page_index\": self.current_page_index, \"action\": next_action, \"screen\": self.encoded_xml,\n"
    )
    replacement_action = (
        "        next_action = self.memory.get_next_action(self.current_subtask, self.encoded_xml)\n"
        "        write_omniflow_mobilegpt_event({\n"
        "            \"event\": (\"memory_action_recalled\" if next_action else \"memory_action_miss\"),\n"
        "            \"action_name\": (next_action or {}).get(\"name\") if isinstance(next_action, dict) else None,\n"
        "            \"page_index\": int(self.current_page_index),\n"
        "        })\n"
        "        current_action_data = {\"page_index\": self.current_page_index, \"action\": next_action, \"screen\": self.encoded_xml,\n"
    )
    if original_action not in source:
        return
    mobilegpt_path.write_text(source.replace(original_action, replacement_action, 1), encoding="utf-8")


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


def _configure_mobilegpt_server(
    server_root: Path,
    *,
    embedding_model: str = "",
) -> None:
    """Inject provider names into a temporary copy of the official Server.

    MobileGPT's upstream code keeps provider model names as constants. This
    function changes routing/observability only unless an experiment
    explicitly sets ``MOBILEGPT_MEMORY_SIMILARITY_THRESHOLD``. In that case
    only the official cosine-similarity cutoff is parameterized; task
    selection, page embeddings, planning and action semantics remain upstream.
    """

    normalized_embedding = str(embedding_model or "").strip()
    utils_path = server_root / "utils" / "utils.py"
    _configure_mobilegpt_telemetry(server_root)
    _configure_mobilegpt_runtime_guards(server_root)
    _configure_mobilegpt_server_port(server_root)
    _configure_mobilegpt_memory_telemetry(server_root)
    _configure_mobilegpt_finish_transport(server_root)
    configured_threshold = str(
        os.environ.get("MOBILEGPT_MEMORY_SIMILARITY_THRESHOLD") or ""
    ).strip()
    if configured_threshold:
        try:
            threshold_value = float(configured_threshold)
        except ValueError as error:
            raise ValueError("mobilegpt_memory_similarity_threshold_invalid") from error
        if not 0.0 < threshold_value <= 1.0:
            raise ValueError("mobilegpt_memory_similarity_threshold_invalid")
        memory_manager_path = server_root / "memory" / "memory_manager.py"
        if not memory_manager_path.is_file():
            raise FileNotFoundError(
                f"official_mobilegpt_memory_manager_missing:{memory_manager_path}"
            )
        memory_manager_source = memory_manager_path.read_text(encoding="utf-8")
        if "MOBILEGPT_MEMORY_SIMILARITY_THRESHOLD" not in memory_manager_source:
            memory_manager_source, replacement_count = re.subn(
                r"(?m)^(?P<indent>[ \t]*)if highest_similarity > 0\.95:[ \t]*$",
                lambda match: (
                    match.group("indent")
                    + "threshold = float(os.getenv(\n"
                    + match.group("indent")
                    + '    "MOBILEGPT_MEMORY_SIMILARITY_THRESHOLD", "0.95"\n'
                    + match.group("indent")
                    + "))\n"
                    + match.group("indent")
                    + "if highest_similarity > threshold:"
                ),
                memory_manager_source,
                count=1,
            )
            if replacement_count != 1:
                raise RuntimeError(
                    "official_mobilegpt_memory_threshold_anchor_missing"
                )
            memory_manager_path.write_text(
                memory_manager_source,
                encoding="utf-8",
            )
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


def _configure_mobilegpt_system_app_catalog(server_root: Path) -> None:
    """Keep the official AppAgent catalog valid for Android system packages.

    Upstream obtains app names and descriptions from Google Play before it
    computes the official app embedding. Android system packages such as
    ``com.android.settings`` do not have a Play listing, so upstream writes an
    empty embedding and later crashes while reading that same CSV row. The
    AndroidWorld adapter already resolves the target package and app name
    during setup; use those values only as the missing catalog metadata, then
    let the official embedding and AppAgent selection run unchanged.
    """

    app_agent_path = server_root / "agents" / "app_agent.py"
    if not app_agent_path.is_file():
        return
    source = app_agent_path.read_text(encoding="utf-8")
    marker = "mobilegpt_system_app_catalog_fallback"
    if marker in source:
        return
    original = (
        "                app_name, description = get_package_info(package_name)\n"
        "                if description:\n"
        "                    embedding = get_openai_embedding(description)\n"
        "                else:\n"
        "                    embedding = \"\"\n"
    )
    replacement = (
        "                app_name, description = get_package_info(package_name)\n"
        "                # mobilegpt_system_app_catalog_fallback: Android system\n"
        "                # packages have no Google Play metadata. Use only the\n"
        "                # setup-resolved identity, then retain MobileGPT's official\n"
        "                # embedding and AppAgent selection path.\n"
        "                configured_package = os.getenv(\n"
        "                    'MOBILEGPT_TARGET_PACKAGE', ''\n"
        "                ).strip()\n"
        "                if not description and package_name == configured_package:\n"
        "                    app_name = (\n"
        "                        os.getenv('MOBILEGPT_TARGET_APP', '').strip()\n"
        "                        or package_name\n"
        "                    )\n"
        "                    description = app_name\n"
        "                if description:\n"
        "                    embedding = get_openai_embedding(description)\n"
        "                else:\n"
        "                    embedding = \"\"\n"
    )
    if original not in source:
        raise RuntimeError("official_mobilegpt_app_catalog_anchor_missing")
    source = source.replace(original, replacement, 1)
    update_anchor = (
        "    def update_app_list(self, new_packages):\n"
        "        known_packages = [row[\"package_name\"] for _, row in self.database.iterrows()]\n"
    )
    update_replacement = (
        "    def update_app_list(self, new_packages):\n"
        "        configured_package = os.getenv('MOBILEGPT_TARGET_PACKAGE', '').strip()\n"
        "        if os.getenv('MOBILEGPT_SKIP_APP_DISCOVERY', '').strip() == '1':\n"
        "            new_packages = [configured_package] if configured_package else []\n"
        "        known_packages = [row[\"package_name\"] for _, row in self.database.iterrows()]\n"
    )
    if update_anchor not in source:
        raise RuntimeError("official_mobilegpt_app_list_anchor_missing")
    app_agent_path.write_text(
        source.replace(update_anchor, update_replacement, 1),
        encoding="utf-8",
    )


def _configure_mobilegpt_target_package_fallback(server_root: Path) -> None:
    """Fill an unresolved official app launch with the setup-resolved package."""

    server_path = server_root / "server.py"
    if not server_path.is_file():
        return
    source = server_path.read_text(encoding="utf-8")
    if "mobilegpt_target_package_fallback" in source:
        return
    original = "                target_package = app_agent.get_package_name(target_app)\n"
    replacement = (
        original
        + "                # mobilegpt_target_package_fallback: the official cold app\n"
        + "                # catalog initially has no description/name for AndroidWorld\n"
        + "                # system packages. Preserve official selection and only fill\n"
        + "                # the empty launch field resolved during environment setup.\n"
        + "                if not str(target_package or '').strip():\n"
        + "                    target_package = os.environ.get(\n"
        + "                        'MOBILEGPT_TARGET_PACKAGE', ''\n"
        + "                    ).strip()\n"
    )
    if original in source:
        server_path.write_text(source.replace(original, replacement, 1), encoding="utf-8")


def _configure_mobilegpt_client_error_transport(server_root: Path) -> None:
    """Complete the upstream APK's ``E`` frame in the pinned Server."""

    server_path = server_root / "server.py"
    if not server_path.is_file():
        return
    source = server_path.read_text(encoding="utf-8")
    if "mobilegpt_client_error_transport" in source:
        return
    anchor = "            elif message_type == 'A':\n"
    branch = (
        "            elif message_type == 'E':\n"
        "                # mobilegpt_client_error_transport: the official APK\n"
        "                # sends E + one line when an action cannot be applied.\n"
        "                # The pinned Server omitted this declared client frame,\n"
        "                # causing bytes inside the error to become a new task.\n"
        "                action_error = b''\n"
        "                while not action_error.endswith(b'\\n'):\n"
        "                    action_error += client_socket.recv(1)\n"
        "                action_error = action_error.decode().strip()\n"
        "                log(f'Action error is received: {action_error}', 'red')\n"
        "                derive_agent = getattr(mobileGPT, 'derive_agent', None)\n"
        "                action_history = getattr(derive_agent, 'action_history', None)\n"
        "                if isinstance(action_history, list) and action_history:\n"
        "                    action_history[-1] = (\n"
        "                        'The previous device action failed: ' + action_error\n"
        "                    )\n"
        "                action = mobileGPT.get_next_action()\n"
        "                if action is not None:\n"
        "                    message = json.dumps(action)\n"
        "                    client_socket.send(message.encode())\n"
        "                    client_socket.send('\\r\\n'.encode())\n"
        "                    write_omniflow_mobilegpt_event({\n"
        "                        'event': 'mobilegpt_action_sent',\n"
        "                        'action': action.get('name')\n"
        "                        if isinstance(action, dict) else None,\n"
        "                        'is_device_action': isinstance(action, dict)\n"
        "                        and action.get('name') not in (None, 'speak'),\n"
        "                        'retry_after_client_error': True,\n"
        "                    })\n"
        "\n"
    )
    if anchor in source:
        server_path.write_text(source.replace(anchor, branch + anchor, 1), encoding="utf-8")


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
# The official server may need one planner/derive round before its first
# primitive action.  GLM-backed cold starts can exceed 20 seconds even when
# the client and server are healthy, so the protocol window must cover that
# normal response latency plus the client's screen-refresh grace period.
MOBILEGPT_HANDSHAKE_TIMEOUT_SEC = 60.0
MOBILEGPT_STEP_TIMEOUT_SEC = 60.0


def _mobilegpt_environment_failure(
    *,
    failure_reason: str,
    returncode: int,
) -> bool:
    """Keep transport/setup failures retryable and method limits terminal."""

    reason = str(failure_reason or "").strip()
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
    # The official client keeps the previous XML value when Accessibility has
    # no root for the target app. On a fresh episode that value is empty, yet
    # sendScreen still transmits it and the official Server parser thread
    # crashes. Clear the value for every capture and retry locally until a
    # real root exists; never fabricate a hierarchy or alter Server semantics.
    if "omniflow_mobilegpt_xml_capture" not in source:
        source, count = re.subn(
            r"(?m)^(?P<indent>[ \t]*)nodeMap = new HashMap<>\(\);\n",
            lambda match: (
                match.group(0)
                + match.group("indent")
                + "// omniflow_mobilegpt_xml_capture\n"
                + match.group("indent")
                + 'currentScreenXML = "";\n'
            ),
            source,
            count=1,
        )
        changed = changed or count > 0
    if "currentScreenXML.trim().isEmpty()" not in source:
        source, count = re.subn(
            r"(?m)^(?P<indent>[ \t]*)private void sendScreen\(\)\{\n",
            lambda match: (
                match.group(0)
                + match.group("indent")
                + "    if (currentScreenXML == null || currentScreenXML.trim().isEmpty()) {\n"
                + match.group("indent")
                + "        Log.d(TAG, \"Target app root unavailable; retrying screen capture\");\n"
                + match.group("indent")
                + "        xmlPending = true;\n"
                + match.group("indent")
                + "        screenNeedUpdate = true;\n"
                + match.group("indent")
                + "        firstScreen = false;\n"
                + match.group("indent")
                + "        mainThreadHandler.postDelayed(screenUpdateTimeoutRunnable, 500);\n"
                + match.group("indent")
                + "        return;\n"
                + match.group("indent")
                + "    }\n"
            ),
            source,
            count=1,
        )
        changed = changed or count > 0
    speak_original = (
        "            if (action.equals(\"speak\")) {\n"
        "                String content = (String) args.get(\"message\");\n"
        "                mSpeech.speak(content, false);\n"
        "                return;\n"
        "            }\n"
    )
    speak_replacement = (
        "            if (action.equals(\"speak\")) {\n"
        "                String content = (String) args.get(\"message\");\n"
        "                mSpeech.speak(content, false);\n"
        "                // omniflow_mobilegpt_speak_lifecycle: the official\n"
        "                // Server needs a new observation after every response.\n"
        "                xmlPending = true;\n"
        "                screenNeedUpdate = true;\n"
        "                firstScreen = false;\n"
        "                mainThreadHandler.postDelayed(screenUpdateTimeoutRunnable, 500);\n"
        "                return;\n"
        "            }\n"
    )
    if speak_original in source and "omniflow_mobilegpt_speak_lifecycle" not in source:
        source = source.replace(speak_original, speak_replacement, 1)
        changed = True
    root_original = (
        "    private AccessibilityNodeInfo getRootForActiveApp(){\n"
        "        List<AccessibilityWindowInfo> windows = getWindows();\n"
        "\n"
        "        for (AccessibilityWindowInfo window : windows) {\n"
        "            AccessibilityNodeInfo root = window.getRoot();\n"
        "            if (root.getPackageName().equals(targetPackageName)) {\n"
        "                return root;\n"
        "            }\n"
        "        }\n"
        "        Log.d(TAG, \"No Appropriate Root found in this screen.\");\n"
        "        return null;\n"
        "    }\n"
    )
    root_replacement = (
        "    private AccessibilityNodeInfo getRootForActiveApp(){\n"
        "        // omniflow_mobilegpt_primary_app_window: AndroidWorld apps can\n"
        "        // expose a same-package tooltip/snackbar as a separate window.\n"
        "        // Keep the official Accessibility observation, but select the\n"
        "        // largest same-package root instead of a transient overlay.\n"
        "        List<AccessibilityWindowInfo> windows = getWindows();\n"
        "        AccessibilityNodeInfo largestRoot = null;\n"
        "        int largestArea = -1;\n"
        "        for (AccessibilityWindowInfo window : windows) {\n"
        "            AccessibilityNodeInfo root = window.getRoot();\n"
        "            if (root == null || root.getPackageName() == null\n"
        "                    || !root.getPackageName().equals(targetPackageName)) {\n"
        "                continue;\n"
        "            }\n"
        "            Rect bounds = new Rect();\n"
        "            root.getBoundsInScreen(bounds);\n"
        "            int area = Math.max(0, bounds.width()) * Math.max(0, bounds.height());\n"
        "            if (area > largestArea) {\n"
        "                largestArea = area;\n"
        "                largestRoot = root;\n"
        "            }\n"
        "        }\n"
        "        if (largestRoot != null) {\n"
        "            return largestRoot;\n"
        "        }\n"
        "        Log.d(TAG, \"No Appropriate Root found in this screen.\");\n"
        "        return null;\n"
        "    }\n"
    )
    if root_original in source and "omniflow_mobilegpt_primary_app_window" not in source:
        source = source.replace(root_original, root_replacement, 1)
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
        )
        if marker in server_diagnostics
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
    disabled_non_mobilegpt_services = {
        "cn.com.omnimind.bot.debug/cn.com.omnimind.accessibility.service.AssistsService",
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
        # Some pinned official MobileGPT Server versions do not emit the
        # optional ``task_started`` telemetry event. They can nevertheless
        # be fully alive: the accessibility client has received the goal and
        # the server has already sent real device actions. Treating that
        # state as a handshake timeout kills simple tasks in the middle of
        # their official onboarding flow. An observed action is stronger
        # evidence of a completed client/server handshake than the optional
        # event, so only time out while no action has been sent at all.
        if (
            not probe["task_started"]
            and _count_mobilegpt_device_actions(stats_path) == 0
            and time.monotonic() >= handshake_deadline
        ):
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
    """Run the pinned official MobileGPT client inside AndroidWorld.

    MobileGPT owns all observation, target-app launch, action selection and
    Accessibility execution.  This wrapper only supplies the canonical
    AndroidWorld task lifecycle and records the official validator result.
    """

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
        console_port=console_port,
        grpc_port=grpc_port,
        adb_path=adb_path,
        perform_emulator_setup=perform_emulator_setup,
        use_uiautomator=False,
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
        reward = float(task.is_successful(env))
        probe_path = output / "protocol_probe.json"
        try:
            probe = json.loads(probe_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            probe = {}
        if not isinstance(probe, dict):
            probe = {}
        stats_path = Path(
            os.environ.get("MOBILEGPT_STATS_JSONL", "")
        ).expanduser()
        stats = _mobilegpt_stats_summary(stats_path)
        reason = str(probe.get("failure_reason") or "").strip()
        environment_failure = reward <= 0.5 and _mobilegpt_environment_failure(
            failure_reason=reason,
            returncode=returncode,
        )
        result_row = {
            "schema_version": "omniflow.androidworld.result.v1",
            "task_name": task_name,
            "task": task_name,
            "goal": official_instruction,
            "requested_instruction": instruction,
            "official_task_instruction": official_instruction,
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
            "official_validator_success": reward > 0.5,
            "official_validator_coverage_rate": 1.0,
            "androidworld_validator_result": {
                "validator": "androidworld_official",
                "success": reward > 0.5,
                "reward": reward,
            },
            "process_returncode": int(returncode),
            "classification": (
                "success"
                if reward > 0.5
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
            "mobilegpt_stats_jsonl": str(stats_path),
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
                (time.monotonic() - episode_started) * 1000.0,
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
        console_port=console_port,
        grpc_port=grpc_port,
        adb_path=adb_path,
        perform_emulator_setup=perform_emulator_setup,
        use_uiautomator=False,
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
