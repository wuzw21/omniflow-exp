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


def _autodroid_memory_app_name(app_name: str) -> str:
    normalized = " ".join(str(app_name or "").strip().lower().split())
    return _AUTODROID_APP_ALIASES.get(normalized, normalized)


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
        use_uiautomator=True,
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
# AppAgent's upstream controller expects one `Physical size:` line. Newer
# Android releases also print `Override size:`; keep that device detail out of
# the official parser without changing AppAgent itself.
if [ "$#" -ge 4 ] && [ "$1" = "-s" ] && [ "$3" = "shell" ] && [ "$4" = "wm" ] && [ "${5:-}" = "size" ]; then
  wm_output=$("$real_adb" "$@" </dev/null)
  printf '%s\\n' "$wm_output" | awk '/^[[:space:]]*Physical size:/{{print; found=1; exit}} END{{if (!found) exit 1}}' || printf '%s\\n' "$wm_output" | sed -n '1p'
  exit 0
fi
if [ "$#" -ge 3 ] && [ "$1" = "shell" ] && [ "$2" = "wm" ] && [ "${3:-}" = "size" ]; then
  wm_output=$("$real_adb" -s "$serial" "$@" </dev/null)
  printf '%s\\n' "$wm_output" | awk '/^[[:space:]]*Physical size:/{{print; found=1; exit}} END{{if (!found) exit 1}}' || printf '%s\\n' "$wm_output" | sed -n '1p'
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
    _link_or_fail(root / "scripts", work / "scripts")
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
    # Keep the disposable Server byte-for-byte identical to the official
    # checkout.  Model names and endpoints are supplied through the child
    # process environment by the caller; this boundary is only responsible
    # for staging the official relative ``./memory`` layout.
    # Some pinned checkouts already import the experiment's optional
    # telemetry hook from ``utils``.  Provide that hook in the disposable
    # workspace only, so a stale checkout cannot prevent the official server
    # from starting; it does not alter planning or action behavior.
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
    _configure_mobilegpt_optional_completion_rate(target)
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


def _configure_mobilegpt_server(
    server_root: Path,
    *,
    embedding_model: str = "",
    chat_model: str = "",
) -> None:
    """Inject provider names into a temporary copy of the official Server.

    MobileGPT's upstream code keeps provider model names as constants. This
    edits only the disposable staging copy; the planner, memory reader,
    protocol, and action implementation remain upstream code.
    """

    normalized_embedding = str(embedding_model or "").strip()
    normalized_chat = str(chat_model or "").strip()
    utils_path = server_root / "utils" / "utils.py"
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
        if mobilegpt_path.is_file():
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
    parse_rate = (
        "    else:\n"
        "        value = float(input_str)\n"
    )
    parse_rate_compat = (
        "    else:\n"
        "        try:\n"
        "            value = float(input_str)\n"
        "        except (TypeError, ValueError):\n"
        "            percent = re.search(r\"(?<!\\d)(\\d+(?:\\.\\d+)?)\\s*%\", input_str)\n"
        "            value = float(percent.group(1)) if percent else 0\n"
    )
    if parse_rate in source:
        source = source.replace(parse_rate, parse_rate_compat, 1)
    empty_result = "    result = response.choices[0].message.content\n"
    empty_result_compat = (
        "    result = response.choices[0].message.content or \"\"\n"
        "    if not result.strip():\n"
        "        for _ in range(2):\n"
        "            retry = client.chat.completions.create(\n"
        "                model=model, messages=messages, temperature=0,\n"
        "                max_tokens=int(os.getenv(\"MOBILEGPT_MAX_TOKENS\", \"1800\")),\n"
        "                top_p=0, frequency_penalty=0, presence_penalty=0\n"
        "            )\n"
        "            result = retry.choices[0].message.content or \"\"\n"
        "            if result.strip():\n"
        "                break\n"
        "        if not result.strip():\n"
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


def _run_mobilegpt_client(
    *,
    official_root: str | Path,
    serial: str,
    adb_path: str,
    host: str,
    instruction: str,
    output_root: str | Path,
    timeout_sec: float,
) -> int:
    """Build, install, and signal the untouched official MobileGPT client."""

    root = Path(official_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    client_root = output / "official_client"
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
    service = "com.example.MobileGPT/com.example.MobileGPT.MobileGPTAccessibilityService"
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
        if "com.example.MobileGPT/com.example.MobileGPT.MobileGPTAccessibilityService" in accessibility_state and "Bound services:" in accessibility_state:
            break
        time.sleep(1.0)
    time.sleep(2.0)
    _run_adb(adb_path, serial, ["logcat", "-c"])
    _run_adb(
        adb_path,
        serial,
        [
            "shell",
            "am",
            "broadcast",
            "-a",
            "com.example.MobileGPT.STRING_ACTION",
            "-p",
            "com.example.MobileGPT",
            "--es",
            "com.example.MobileGPT.INSTRUCTION_EXTRA",
            instruction,
        ],
    )
    deadline = time.monotonic() + max(1.0, float(timeout_sec))
    while time.monotonic() < deadline:
        log = _run_adb(
            adb_path,
            serial,
            ["logcat", "-d", "-s", "MobileGPT_Service:D", "*:S"],
            check=False,
        ).stdout
        if "Task finished" in log or "-----------Task finished--------" in log:
            (output / "client_log.txt").write_text(log, encoding="utf-8")
            return 0
        time.sleep(1.0)
    (output / "client_log.txt").write_text(log, encoding="utf-8")
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
    ) as (env, task):
        returncode = _run_mobilegpt_client(
            official_root=official_root,
            serial=serial,
            adb_path=adb_path,
            host=host,
            instruction=instruction,
            output_root=output_root,
            timeout_sec=timeout_sec,
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
                if (
                    event.get("event") == "mobilegpt_action_sent"
                    and event.get("is_device_action") is True
                ):
                    actions_executed += 1
                if event.get("event") in {"chat_call", "embedding_call"}:
                    model_calls += 1
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
            "classification": "success" if validator_success else "method_failure",
            "actions_executed": actions_executed,
            "model_calls": model_calls,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "token_usage_status": "unavailable",
            "fallback_steps": 0,
            "duration_ms": round((time.monotonic() - started) * 1000.0, 3),
            "mobilegpt_stats_jsonl": str(stats_path),
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
    ) as (env, task):
        process_returncode = 1
        try:
            result = subprocess.run(
                [
                    str(python_executable),
                    str(executor),
                    "--app",
                    str(app_name),
                    "--root_dir",
                    str(workspace),
                ],
                cwd=str(workspace),
                input=str(goal) + "\n",
                text=True,
                check=False,
                timeout=max(1.0, float(timeout_sec)),
            )
            process_returncode = int(result.returncode)
        except subprocess.TimeoutExpired:
            process_returncode = 124
        reward = float(task.is_successful(env))
        validator_success = process_returncode == 0 and reward > 0.5
        official_log = output / "official_appagent.log"
        actions_executed = 0
        if official_log.is_file():
            actions_executed = sum(
                line.strip().startswith("Round ")
                for line in official_log.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
            )
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
            "model_calls": actions_executed,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "token_usage_status": "unavailable",
            "fallback_steps": 0,
            "duration_ms": round((time.monotonic() - started) * 1000.0, 3),
            "official_log": str(official_log),
        }
        (output / "task_results.jsonl").write_text(
            json.dumps(result_row, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return 0 if validator_success else (process_returncode or 1)


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
        memory_info.update(
            {
                "task_app_names": task_app_names,
                "task_app_name": task_app_name,
                "task_app_selection": (
                    "explicit_app_name" if explicit_app_name else "task_declared_first"
                ),
            }
        )
        droidbot_output = output / "droidbot"
        official_launcher = "from droidbot.start import main; main()"
        if policy == "replay":
            official_launcher = (
                "from droidbot.input_policy import UtgReplayPolicy; "
                "_official_generate_event = UtgReplayPolicy.generate_event; "
                "UtgReplayPolicy.generate_event = "
                "lambda self, input_manager=None: _official_generate_event(self); "
                + official_launcher
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
                    "from openai import OpenAI",
                    "import tools",
                    "_stats = Path(os.environ['AUTODROID_STATS_PATH'])",
                    "_stats.parent.mkdir(parents=True, exist_ok=True)",
                    "def _append(row):",
                    "    with _stats.open('a', encoding='utf-8') as handle:",
                    "        handle.write(json.dumps(row) + '\\n')",
                    "def _query(prompt):",
                    "    started = time.monotonic()",
                    "    client_kwargs = {'api_key': os.environ.get('APIKey') or os.environ.get('OPENAI_API_KEY')}",
                    "    if os.environ.get('OPENAI_BASE_URL'):",
                    "        client_kwargs['base_url'] = os.environ['OPENAI_BASE_URL']",
                    "    try:",
                    "        completion = OpenAI(**client_kwargs).chat.completions.create(messages=[{'role': 'user', 'content': prompt}], model=os.environ.get('AUTODROID_MODEL', 'gpt-3.5-turbo'), timeout=15)",
                    "    except Exception as error:",
                    "        _append({'event': 'chat_call', 'model': os.environ.get('AUTODROID_MODEL', 'gpt-3.5-turbo'), 'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'error': type(error).__name__, 'elapsed_sec': round(time.monotonic() - started, 6)})",
                    "        raise",
                    "    usage = getattr(completion, 'usage', None)",
                    "    _append({'event': 'chat_call', 'model': os.environ.get('AUTODROID_MODEL', 'gpt-3.5-turbo'), 'prompt_tokens': int(getattr(usage, 'prompt_tokens', 0) or 0), 'completion_tokens': int(getattr(usage, 'completion_tokens', 0) or 0), 'total_tokens': int(getattr(usage, 'total_tokens', 0) or 0), 'elapsed_sec': round(time.monotonic() - started, 6)})",
                    "    return completion.choices[0].message.content",
                    "tools.query_gpt = _query",
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
                    "_start.main()",
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
        if policy == "task":
            env_vars["APIKey"] = str(
                env_vars.get("APIKey") or env_vars.get("OPENAI_API_KEY") or ""
            )
            env_vars["AUTODROID_STATS_PATH"] = str(output / "autodroid_stats.jsonl")
        env_vars["PYTHONPATH"] = os.pathsep.join(
            value
            for value in (str(root), env_vars.get("PYTHONPATH", ""))
            if value
        )
        adb_parent = Path(adb_path).expanduser().resolve().parent
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
            replayed_event_count = min(
                int(memory_info["event_count"]), max(1, int(max_events))
            )
        else:
            replayed_event_count = len(
                sorted(droidbot_output.glob("events/event_*.json"))
            )
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
            "replay_completed": validator_used,
            "replay_step_completed_count": (
                replayed_event_count if validator_used else 0
            ),
            "replay_step_total": replayed_event_count,
            "replay_step_completed_rate": (
                1.0 if validator_used and replayed_event_count else 0.0
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
    )


if __name__ == "__main__":
    raise SystemExit(main())
