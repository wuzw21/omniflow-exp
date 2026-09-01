"""Build the native AutoDroid app-memory tables from a collected UTG.

AutoDroid's public runtime reads three JSON tables rather than ``utg.js``
directly.  This module keeps that boundary explicit: the UTG is the offline
source of states, transitions, and candidate elements; a source RunLog is
only used to identify the app and audit whether its path is represented by the
UTG.  Source actions are never copied into the memory tables.
"""

from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from threading import Lock
from typing import Any, Iterable
import xml.etree.ElementTree as ET

from openai import OpenAI

from omniflow.core.trajectory import require_complete_source_run_log
from src.experiment.paths import relative_reference, sha256_file
from src.experiment.protocol import (
    FORMAL_MAX_TOKENS,
    FORMAL_MODEL_BASE_URL,
    FORMAL_REQUEST_TIMEOUT_SEC,
    FORMAL_THINKING,
    require_formal_model,
)
from src.integrations.android_world.host import androidworld_observation_package

AUTODROID_MEMORY_SCHEMA = "autodroid.native_app_memory.v1"
AUTODROID_MEMORY_FILES = (
    "node_filtered_elements.json",
    "element_description.json",
    "embedded_elements_desc.json",
    "app_state_summary.json",
)

# The official AutoDroid checkout stores one native table per app under a
# normalized app key.  These keys are part of the shipped memory contract;
# they are not a second app matcher.  The explicit option is only used when
# materializing an already-published official table.
OFFICIAL_MEMORY_APP_KEYS = {
    "audio": "voicerecorder",
    "calendar": "calendar",
    "camera": "camera",
    "clock": "clock",
    "contacts": "contacts",
    "files": "filemanager",
    "gallery": "gallery",
    "joplin": "notes",
    "retro": "musicplayer",
    "sms": "messenger",
}


def _load_native_device_state():
    """Load AutoDroid's own state-to-HTML renderer from the vendored runtime."""

    runtime = Path(__file__).resolve().parents[2] / "vendor" / "autodroid" / "runtime"
    if not runtime.is_dir():
        raise FileNotFoundError(f"autodroid_runtime_missing:{runtime}")
    runtime_string = str(runtime)
    if runtime_string not in sys.path:
        sys.path.insert(0, runtime_string)
    from droidbot.device_state import DeviceState

    return DeviceState


class _StaticDevice:
    """Minimal device surface needed by AutoDroid's native DeviceState."""

    humanoid = None

    def __init__(self, width: int, height: int, output_dir: Path) -> None:
        self.display_info = {"width": int(width), "height": int(height)}
        self.output_dir = str(output_dir)

    def get_width(self, refresh: bool = False) -> int:
        return int(self.display_info["width"])

    def get_height(self, refresh: bool = False) -> int:
        return int(self.display_info["height"])


def _parse_utg(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"autodroid_utg_json_not_found:{path}")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("autodroid_utg_object_required")
    return payload


def _state_files(utg_root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted((utg_root / "states").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        state_id = str(payload.get("state_str") or "").strip()
        if state_id:
            result[state_id] = payload
    if not result:
        raise ValueError(f"autodroid_utg_states_missing:{utg_root}")
    return result


def _shorten(value: Any, limit: int = 15) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    words = text.split()
    return " ".join(words[:limit])


def _clean_model_text(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^```(?:text|plain)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    text = re.sub(r"^(?:task|summary|description|function)\s*:\s*", "", text, flags=re.IGNORECASE)
    return _shorten(text)


def _validated_source_run_log(value: dict[str, Any]) -> dict[str, Any]:
    """Validate a source log, tolerating legacy raw swipe endpoints.

    Some recollected AndroidWorld source logs retain pixel swipe endpoints
    (`x1`/`y1`/`x2`/`y2`) alongside the canonical direction. AutoDroid only
    needs the official observations and completion proof, so remove those
    redundant fields from a private validation copy while preserving the
    original file and its hash as provenance.
    """

    try:
        return require_complete_source_run_log(value)
    except ValueError as error:
        if "additionalProperties:x1" not in str(error):
            raise
        validation_error = error
    sanitized = copy.deepcopy(value)
    changed = False
    for step in sanitized.get("steps") or []:
        if not isinstance(step, dict):
            continue
        action = step.get("action")
        if not isinstance(action, dict) or action.get("action_type") != "swipe":
            continue
        for key in ("x1", "y1", "x2", "y2"):
            changed = action.pop(key, None) is not None or changed
    if not changed:
        raise validation_error
    return require_complete_source_run_log(sanitized)


def _source_package(source: dict[str, Any]) -> str:
    # AndroidWorld's open_app action may carry the fully-qualified package
    # name even when the trajectory subsequently enters a system-owned
    # picker (for example DocumentsUI).  Treat that explicit app identity as
    # stronger than the modal screen package for source/UTG matching.
    for step in source.get("steps") or []:
        if not isinstance(step, dict):
            continue
        action = step.get("action")
        if not isinstance(action, dict) or str(
            action.get("action_type") or action.get("type") or ""
        ) != "open_app":
            continue
        app_name = str(action.get("app_name") or action.get("package") or "").strip()
        if app_name.count(".") >= 1 and " " not in app_name:
            return app_name
    # A Source RunLog can begin/end on the launcher or a permission dialog.
    # Prefer the package that owns the largest share of UI-tree nodes across
    # the complete trajectory, while retaining the official observation
    # helper as a fallback for observations without XML.
    packages: Counter[str] = Counter()
    fallback: list[str] = []
    observations: list[dict[str, Any]] = []
    for step in source.get("steps") or []:
        if not isinstance(step, dict):
            continue
        for key in ("observation", "next_observation"):
            observation = step.get(key)
            if isinstance(observation, dict):
                observations.append(observation)
    final_observation = source.get("final_observation")
    if isinstance(final_observation, dict):
        observations.append(final_observation)
    for observation in observations:
        package = androidworld_observation_package(observation)
        if package:
            fallback.append(str(package))
        xml = str(observation.get("xml") or "")
        try:
            packages.update(
                str(node.attrib.get("package") or "")
                for node in ET.fromstring(xml).iter()
                if node.attrib.get("package")
            )
        except ET.ParseError:
            continue
    ignored = {
        "android",
        "com.android.systemui",
        "com.google.android.inputmethod.latin",
        "com.google.android.apps.nexuslauncher",
        "com.google.android.permissioncontroller",
    }
    app_packages = [
        (package, count)
        for package, count in packages.most_common()
        if package not in ignored
    ]
    if app_packages:
        return app_packages[0][0]
    return next((package for package in reversed(fallback) if package not in ignored), "")


def _native_state_render(
    state: dict[str, Any],
    *,
    temp_output: Path,
) -> dict[str, Any]:
    temp_output.mkdir(parents=True, exist_ok=True)
    DeviceState = _load_native_device_state()
    device = _StaticDevice(
        int(state.get("width") or 0),
        int(state.get("height") or 0),
        temp_output,
    )
    native = DeviceState(
        device=device,
        views=list(state.get("views") or []),
        foreground_activity=str(state.get("foreground_activity") or ""),
        activity_stack=list(state.get("activity_stack") or []),
        background_services=list(state.get("background_services") or []),
        tag=str(state.get("tag") or ""),
    )
    html, actions, elements, _important = native.get_described_actions(
        remove_time_and_ip=False
    )
    action_views: dict[str, str] = {}
    for index, action in enumerate(actions):
        view = getattr(action, "view", None)
        view_str = str(view.get("view_str") or "").strip() if isinstance(view, dict) else ""
        if view_str and index < len(elements):
            action_views[view_str] = str(elements[index])
    return {
        "html": str(html),
        "elements": [str(item) if item else None for item in elements],
        "action_views": action_views,
        # Keep the exact candidate-to-view relation used by the native
        # renderer.  The official AndroidWorld adapter consumes this relation
        # to project the selected native candidate into its OOB Host action;
        # it never reuses source coordinates.
        "action_view_refs": [
            getattr(action, "view", None) for action in actions
        ],
        "action_types": [type(action).__name__ for action in actions],
    }


def _object_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, dict):
            return dict(result)
    attributes = getattr(value, "__dict__", None)
    return dict(attributes) if isinstance(attributes, dict) else {}


def _androidworld_bounds(value: Any) -> list[list[int]]:
    raw = _object_dict(value)
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        values = list(value)
        try:
            return [[int(values[0]), int(values[1])], [int(values[2]), int(values[3])]]
        except (TypeError, ValueError):
            return [[0, 0], [1, 1]]
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return [
                [int(value[0][0]), int(value[0][1])],
                [int(value[1][0]), int(value[1][1])],
            ]
        except (TypeError, IndexError, KeyError, ValueError):
            return [[0, 0], [1, 1]]
    for keys in (
        ("x_min", "y_min", "x_max", "y_max"),
        ("left", "top", "right", "bottom"),
    ):
        if all(key in raw for key in keys):
            try:
                return [
                    [int(raw[keys[0]]), int(raw[keys[1]])],
                    [int(raw[keys[2]]), int(raw[keys[3]])],
                ]
            except (TypeError, ValueError):
                pass
    return [[0, 0], [1, 1]]


def render_androidworld_observation(
    state_payload: dict[str, Any],
    *,
    temp_output: Path,
) -> dict[str, Any]:
    """Render one live AndroidWorld observation with AutoDroid's native grammar.

    AndroidWorld remains the owner of observation and action dispatch.  This
    helper only converts the live accessibility elements to the same
    description grammar used while building native AutoDroid Memory, so the
    Memory method can make a like-for-like candidate decision.
    """

    payload = dict(state_payload or {})
    auxiliaries = payload.get("auxiliaries")
    auxiliaries = auxiliaries if isinstance(auxiliaries, dict) else {}
    display = auxiliaries.get("display")
    display = display if isinstance(display, dict) else {}
    width = int(display.get("width") or payload.get("width") or 1)
    height = int(display.get("height") or payload.get("height") or 1)
    views: list[dict[str, Any]] = []
    for index, raw_element in enumerate(payload.get("ui_elements") or []):
        element = _object_dict(raw_element)
        bounds = _androidworld_bounds(
            element.get("bbox_pixels") or element.get("bbox")
        )
        views.append(
            {
                "package": str(element.get("package_name") or ""),
                "visible": bool(element.get("is_visible", True)),
                "checkable": bool(element.get("is_checkable")),
                "editable": bool(element.get("is_editable")),
                "clickable": bool(element.get("is_clickable")),
                "is_password": bool(element.get("is_password")),
                "focusable": bool(element.get("is_focusable")),
                "enabled": bool(element.get("is_enabled", True)),
                "content_description": element.get("content_description") or "",
                "children": [],
                "focused": bool(element.get("is_focused")),
                "resource_id": element.get("resource_id")
                or element.get("resource_name")
                or "",
                "checked": bool(element.get("is_checked")),
                "text": element.get("text") or "",
                "class": element.get("class_name") or "",
                "scrollable": bool(element.get("is_scrollable")),
                "selected": bool(element.get("is_selected")),
                "long_clickable": bool(element.get("is_long_clickable")),
                "parent": -1,
                "temp_id": index,
                "bounds": bounds,
            }
        )
    return _native_state_render(
        {
            "width": width,
            "height": height,
            "views": views,
            "foreground_activity": str(
                auxiliaries.get("activity_name") or payload.get("activity_name") or ""
            ),
            "activity_stack": [],
            "background_services": [],
        },
        temp_output=temp_output,
    )


def _event_element(event: dict[str, Any], rendered: dict[str, Any]) -> str | None:
    event_type = str(event.get("event_type") or "").strip().lower()
    if event_type == "click":
        match = re.search(r"view=([0-9a-f]{32})", str(event.get("event_str") or ""))
        if match:
            return rendered["action_views"].get(match.group(1))
        return None
    if event_type == "key" and str(event.get("event_str") or "").find("BACK") >= 0:
        elements = rendered["elements"]
        return elements[-1] if elements else "<button>go back</button>"
    if event_type == "intent":
        return "<intent>open app</intent>"
    return None


def _event_label(event: dict[str, Any], rendered: dict[str, Any]) -> str:
    return _event_element(event, rendered) or _shorten(event.get("event_str"), 30)


def _dedupe_edges(utg: dict[str, Any]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for edge in utg.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        for event in edge.get("events") or []:
            if not isinstance(event, dict):
                continue
            item = dict(event)
            item["from"] = str(edge.get("from") or "")
            item["to"] = str(edge.get("to") or "")
            item["label"] = str(edge.get("label") or "")
            edges.append(item)
    return edges


def _shortest_paths(
    start: str,
    edges: Iterable[dict[str, Any]],
    allowed: set[str],
    rendered: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    outgoing: dict[str, list[dict[str, Any]]] = {}
    for edge in edges:
        source = str(edge.get("from") or "")
        target = str(edge.get("to") or "")
        if source in allowed and target in allowed:
            outgoing.setdefault(source, []).append(edge)
    paths = {start: []}
    queue: deque[str] = deque([start])
    while queue:
        source = queue.popleft()
        for edge in outgoing.get(source, []):
            target = str(edge["to"])
            if target in paths:
                continue
            paths[target] = [
                *paths[source],
                _event_label(edge, rendered[source]),
            ]
            queue.append(target)
    return paths


class _NativeMemoryAuthor:
    def __init__(self, *, model: str) -> None:
        self.model = require_formal_model(model)
        api_key = str(
            os.environ.get("LLMTHU_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        ).strip()
        if not api_key:
            raise RuntimeError("autodroid_memory_model_api_key_required")
        import httpx

        self.client = OpenAI(
            api_key=api_key,
            base_url=FORMAL_MODEL_BASE_URL,
            max_retries=0,
            timeout=FORMAL_REQUEST_TIMEOUT_SEC,
            http_client=httpx.Client(trust_env=False),
        )
        self.usage: list[dict[str, Any]] = []
        self._usage_lock = Lock()

    def close(self) -> None:
        self.client.close()

    def complete(self, *, prompt: str, purpose: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Return only the requested short English description.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=FORMAL_MAX_TOKENS,
            extra_body={
                "enable_thinking": False,
                "thinking": {"type": FORMAL_THINKING},
            },
        )
        message = response.choices[0].message.content if response.choices else ""
        usage = response.usage
        with self._usage_lock:
            self.usage.append(
                {
                    "purpose": purpose,
                    "model": str(getattr(response, "model", "") or self.model),
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                    "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                }
            )
        cleaned = _clean_model_text(str(message or ""))
        if not cleaned:
            raise ValueError(f"autodroid_memory_empty_model_output:{purpose}")
        return cleaned

    def complete_many(
        self,
        requests: list[tuple[str, str]],
        *,
        max_workers: int = 8,
    ) -> list[str]:
        """Complete independent native-memory prompts concurrently."""

        if not requests:
            return []
        with ThreadPoolExecutor(max_workers=min(max_workers, len(requests))) as pool:
            futures = [
                pool.submit(self.complete, prompt=prompt, purpose=purpose)
                for purpose, prompt in requests
            ]
            return [future.result() for future in futures]


def _instructor_embeddings(descriptions: list[str]) -> list[list[float]]:
    if not descriptions:
        return []
    # The formal model endpoint may use the workstation's SOCKS proxy, while
    # the Hugging Face weight fetch must use ordinary HTTPS in this runtime.
    # Isolate that choice to the local embedding load and restore the parent
    # process environment immediately afterwards.
    proxy_keys = (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    )
    saved_proxy = {key: os.environ.get(key) for key in proxy_keys}
    for key in proxy_keys:
        os.environ.pop(key, None)
    try:
        from InstructorEmbedding import INSTRUCTOR

        import inspect

        loader = getattr(INSTRUCTOR, "_load_sbert_model", None)
        if loader is not None and "token" not in inspect.signature(loader).parameters:
            original = loader

            def compatible(self: Any, model_path: Any, token: Any = None, **kwargs: Any) -> Any:
                del token, kwargs
                return original(self, model_path)

            INSTRUCTOR._load_sbert_model = compatible

        configured = str(os.environ.get("AUTODROID_INSTRUCTOR_MODEL") or "").strip()
        candidates = [
            Path(configured).expanduser() if configured else None,
            Path.home() / "models" / "instructor-xl",
            Path("/models/instructor-xl"),
        ]
        model_ref = next(
            (candidate for candidate in candidates if candidate is not None and candidate.is_dir()),
            Path("hkunlp/instructor-xl"),
        )
        model = INSTRUCTOR(str(model_ref))
        try:
            # InstructorEmbedding 1.x calls the legacy SentenceTransformer
            # length helper.  Newer sentence-transformers exposes the same
            # behavior as ``_input_length``; keep the official model and
            # encoder path while adapting only this renamed private hook.
            if not hasattr(model, "_text_length") and hasattr(model, "_input_length"):
                model._text_length = model._input_length
            vectors = model.encode(["task: " + item for item in descriptions])
            return [[float(value) for value in vector] for vector in vectors]
        finally:
            del model
    finally:
        for key, value in saved_proxy.items():
            if value is not None:
                os.environ[key] = value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def materialize_official_autodroid_memory(
    *,
    source_run_log: str | Path,
    utg_root: str | Path,
    official_memory_root: str | Path,
    memory_root: str | Path,
    official_app_key: str,
) -> dict[str, Any]:
    """Materialize one app from AutoDroid's published native memory tables.

    The official repository publishes the four runtime tables as dictionaries
    keyed by app.  This operation extracts exactly one app entry, preserving
    the official values and vectors byte-for-byte at the JSON-value level;
    the Source RunLog is recorded only as provenance and is never injected
    into the tables.
    """

    source_path = Path(source_run_log).expanduser().resolve()
    utg_path = Path(utg_root).expanduser().resolve()
    official_root = Path(official_memory_root).expanduser().resolve()
    root = Path(memory_root).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"autodroid_source_run_log_missing:{source_path}")
    if not (utg_path / "utg.js").is_file():
        raise FileNotFoundError(f"autodroid_utg_missing:{utg_path / 'utg.js'}")
    if root.exists():
        raise FileExistsError(f"immutable_autodroid_memory_exists:{root}")
    source = _validated_source_run_log(
        json.loads(source_path.read_text(encoding="utf-8"))
    )
    utg = _parse_utg(utg_path / "utg.js")
    app_package = str(utg.get("app_package") or "").strip()
    source_package = _source_package(source)
    if source_package and source_package != app_package:
        raise ValueError(
            f"autodroid_source_utg_package_mismatch:{source_package}:{app_package}"
        )
    app_key = str(official_app_key or "").strip()
    if not app_key:
        raise ValueError("autodroid_official_memory_app_key_required")

    tables: dict[str, Any] = {}
    for filename in AUTODROID_MEMORY_FILES:
        path = official_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"autodroid_official_memory_file_missing:{filename}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or app_key not in payload:
            raise KeyError(f"autodroid_official_memory_app_missing:{filename}:{app_key}")
        tables[filename] = {app_key: payload[app_key]}

    # The official runtime only consumes the four tables above.  Keep a small
    # derived task list for audit/UI inspection without changing those native
    # values or adding source actions.
    element_table = tables["element_description.json"][app_key]
    simulated_tasks: list[dict[str, Any]] = []
    if isinstance(element_table, dict):
        for state_id, descriptions in element_table.items():
            if not isinstance(descriptions, list):
                continue
            for index, description in enumerate(descriptions):
                if description:
                    simulated_tasks.append(
                        {
                            "simulated_task": str(description),
                            "ui_states": [str(state_id)],
                            "ui_elements": [index],
                            "source_state": str(state_id),
                            "target_state": None,
                            "event": None,
                        }
                    )

    root.mkdir(parents=True, exist_ok=False)
    for filename, payload in tables.items():
        _write_json(root / filename, payload)
    _write_json(root / "simulated_tasks.json", simulated_tasks)
    report = {
        "schema_version": AUTODROID_MEMORY_SCHEMA,
        "status": "completed",
        "conversion": "official_published_native_memory",
        "app_name": str(utg_path.name),
        "app_package": app_package,
        "official_app_key": app_key,
        "source_run_log": relative_reference(source_path, base=root),
        "source_run_log_sha256": sha256_file(source_path),
        "utg_root": relative_reference(utg_path, base=root),
        "utg_sha256": sha256_file(utg_path / "utg.js"),
        "model": require_formal_model(),
        "embedding_model": "hkunlp/instructor-xl",
        "counts": {
            "simulated_tasks": len(simulated_tasks),
            "model_calls": 0,
            "embedding_vectors": sum(
                len(v)
                for v in (tables["embedded_elements_desc.json"][app_key] or {}).values()
                if isinstance(v, list)
            ),
        },
        "source_run_log_audit": {
            "task_name": str(source.get("task_name") or ""),
            "goal": str(source.get("goal") or ""),
            "source_package": source_package,
            "source_actions_injected": False,
            "utg_is_memory_source": True,
            "path_coverage": "not_injected_and_requires_online_alignment",
        },
    }
    _write_json(root / "memory_build_report.json", report)
    _write_json(root / "model_usage.json", [])
    return {**report, "memory_root": str(root)}


def materialize_autodroid_memory_bundle(
    *,
    source_memory_root: str | Path,
    source_run_log: str | Path,
    utg_root: str | Path,
    memory_root: str | Path,
) -> dict[str, Any]:
    """Reuse one completed app Memory for another explicit Source task."""

    bundle = Path(source_memory_root).expanduser().resolve()
    source_path = Path(source_run_log).expanduser().resolve()
    utg_path = Path(utg_root).expanduser().resolve()
    root = Path(memory_root).expanduser().resolve()
    if not bundle.is_dir():
        raise FileNotFoundError(f"autodroid_memory_bundle_missing:{bundle}")
    if not source_path.is_file():
        raise FileNotFoundError(f"autodroid_source_run_log_missing:{source_path}")
    if not (utg_path / "utg.js").is_file():
        raise FileNotFoundError(f"autodroid_utg_missing:{utg_path / 'utg.js'}")
    if root.exists():
        raise FileExistsError(f"immutable_autodroid_memory_exists:{root}")
    source = _validated_source_run_log(
        json.loads(source_path.read_text(encoding="utf-8"))
    )
    previous_report = json.loads(
        (bundle / "memory_build_report.json").read_text(encoding="utf-8")
    )
    if previous_report.get("schema_version") != AUTODROID_MEMORY_SCHEMA:
        raise ValueError("autodroid_memory_bundle_schema_mismatch")
    root.mkdir(parents=True, exist_ok=False)
    for filename in AUTODROID_MEMORY_FILES + (
        "simulated_tasks.json",
        "model_usage.json",
    ):
        source_file = bundle / filename
        if source_file.is_file():
            shutil.copy2(source_file, root / filename)
    report = dict(previous_report)
    report["conversion"] = "app_memory_reuse"
    report["source_run_log"] = relative_reference(source_path, base=root)
    report["source_run_log_sha256"] = sha256_file(source_path)
    report["utg_root"] = relative_reference(utg_path, base=root)
    report["utg_sha256"] = sha256_file(utg_path / "utg.js")
    audit = dict(report.get("source_run_log_audit") or {})
    audit.update(
        {
            "task_name": str(source.get("task_name") or ""),
            "goal": str(source.get("goal") or ""),
            "source_package": _source_package(source),
            "source_actions_injected": False,
            "app_memory_bundle_reused": True,
        }
    )
    report["source_run_log_audit"] = audit
    _write_json(root / "memory_build_report.json", report)
    return {**report, "memory_root": str(root)}


def convert_runlog_to_autodroid_memory(
    *,
    source_run_log: str | Path,
    utg_root: str | Path,
    memory_root: str | Path,
    model: str,
) -> dict[str, Any]:
    """Convert one Source RunLog plus one explicit UTG to native App Memory."""

    source_path = Path(source_run_log).expanduser().resolve()
    utg_path = Path(utg_root).expanduser().resolve()
    root = Path(memory_root).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"autodroid_source_run_log_missing:{source_path}")
    if not (utg_path / "utg.js").is_file():
        raise FileNotFoundError(f"autodroid_utg_missing:{utg_path / 'utg.js'}")
    if root.exists():
        raise FileExistsError(f"immutable_autodroid_memory_exists:{root}")

    source = _validated_source_run_log(
        json.loads(source_path.read_text(encoding="utf-8"))
    )
    utg = _parse_utg(utg_path / "utg.js")
    states = _state_files(utg_path)
    app_package = str(utg.get("app_package") or "").strip()
    source_package = _source_package(source)
    if not app_package:
        raise ValueError("autodroid_utg_app_package_missing")
    if source_package and source_package != app_package:
        raise ValueError(
            f"autodroid_source_utg_package_mismatch:{source_package}:{app_package}"
        )

    nodes = [
        node
        for node in utg.get("nodes") or []
        if isinstance(node, dict) and str(node.get("id") or "") in states
    ]
    app_state_ids = {
        str(node["id"])
        for node in nodes
        if str(node.get("package") or "") == app_package
    }
    if not app_state_ids:
        raise ValueError("autodroid_utg_app_states_missing")
    initial_state = next(
        (str(node["id"]) for node in nodes if str(node.get("package") or "") == app_package),
        "",
    )
    edges = [
        edge
        for edge in _dedupe_edges(utg)
        if edge.get("from") in states and edge.get("to") in states
    ]

    root.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="autodroid-native-render-") as temp:
        temp_output = Path(temp)
        rendered = {
            state_id: _native_state_render(states[state_id], temp_output=temp_output)
            for state_id in app_state_ids
        }

    paths = _shortest_paths(initial_state, edges, app_state_ids, rendered)
    author = _NativeMemoryAuthor(model=model)
    try:
        descriptions: dict[tuple[str, int], str | None] = {}
        description_by_element: dict[str, str] = {}
        ordered_states = sorted(
            app_state_ids, key=lambda item: (len(paths.get(item, [])), item)
        )
        state_prompts = [
            (
                "screen_function",
                "Summarize the function of this Android screen in at most 15 words. "
                "Output only the summary.\n\nSCREEN:\n"
                + rendered[state_id]["html"][:16000],
            )
            for state_id in ordered_states
        ]
        state_functions = dict(
            zip(ordered_states, author.complete_many(state_prompts))
        )
        element_prompts: list[tuple[str, str]] = []
        ordered_elements: list[str] = []
        for state_id in ordered_states:
            html = rendered[state_id]["html"][:16000]
            previous = " -> ".join(paths.get(state_id, [])) or "(initial app screen)"
            for element in rendered[state_id]["elements"]:
                if element and element not in description_by_element:
                    ordered_elements.append(element)
                    element_prompts.append(
                        (
                            "element_task",
                            "Given the Android screen and the previous UI actions, describe one "
                            "user task that can be advanced by selecting the next UI element. "
                            "Use at most 15 words, do not mention coordinates, IDs, or previous "
                            "steps, and output only the task description.\n\n"
                            f"PREVIOUS UI ACTIONS: {previous}\n"
                            f"NEXT UI ELEMENT: {element}\n"
                            f"SCREEN:\n{html}",
                        )
                    )
        description_by_element.update(
            zip(ordered_elements, author.complete_many(element_prompts))
        )
        for state_id in ordered_states:
            for index, element in enumerate(rendered[state_id]["elements"]):
                descriptions[(state_id, index)] = (
                    description_by_element.get(element) if element else None
                )

        unique_descriptions = list(dict.fromkeys(description_by_element.values()))
        vectors = _instructor_embeddings(unique_descriptions)
        vector_by_description = dict(zip(unique_descriptions, vectors))

        node_filtered: dict[str, Any] = {str(utg_path.name): {}}
        element_description: dict[str, Any] = {str(utg_path.name): {}}
        embedded: dict[str, Any] = {str(utg_path.name): {}}
        summaries: dict[str, Any] = {str(utg_path.name): state_functions}
        simulated_tasks: list[dict[str, Any]] = []
        for state_id in sorted(app_state_ids, key=lambda item: (len(paths.get(item, [])), item)):
            elements = rendered[state_id]["elements"]
            element_texts = [
                element if element else None for element in elements
            ]
            element_tasks = [descriptions[(state_id, index)] for index in range(len(elements))]
            element_vectors = [
                vector_by_description.get(task) if task else None
                for task in element_tasks
            ]
            node_filtered[str(utg_path.name)][state_id] = {
                "path": paths.get(state_id, []),
                "elements": element_texts,
                "debug": elements,
                "gpath": paths.get(state_id, []),
            }
            element_description[str(utg_path.name)][state_id] = element_tasks
            embedded[str(utg_path.name)][state_id] = element_vectors

        for edge in edges:
            source_id = str(edge["from"])
            target_id = str(edge["to"])
            if source_id not in app_state_ids or target_id not in app_state_ids:
                continue
            element = _event_element(edge, rendered[source_id])
            if not element or element not in description_by_element:
                continue
            task = description_by_element[element]
            simulated_tasks.append(
                {
                    "simulated_task": task,
                    "ui_states": [source_id, target_id],
                    "ui_elements": [element],
                    "source_state": source_id,
                    "target_state": target_id,
                    "event": str(edge.get("event_str") or ""),
                }
            )

        source_action_labels = [
            str((step.get("action") or {}).get("action_type") or "")
            for step in source.get("steps") or []
            if isinstance(step, dict)
        ]
        source_element_labels = [
            str((step.get("action") or {}).get("action_type") or "")
            for step in source.get("steps") or []
            if isinstance(step, dict)
            and str((step.get("action") or {}).get("action_type") or "") in {"click", "input_text"}
        ]
        report = {
            "schema_version": AUTODROID_MEMORY_SCHEMA,
            "status": "completed",
            "conversion": "official_online_memory_build",
            "app_name": str(utg_path.name),
            "app_package": app_package,
            "source_run_log": relative_reference(source_path, base=root),
            "source_run_log_sha256": sha256_file(source_path),
            "utg_root": relative_reference(utg_path, base=root),
            "utg_sha256": sha256_file(utg_path / "utg.js"),
            "model": author.model,
            "embedding_model": "hkunlp/instructor-xl",
            "counts": {
                "utg_nodes": len(nodes),
                "app_states": len(app_state_ids),
                "app_edges": sum(
                    1
                    for edge in edges
                    if edge.get("from") in app_state_ids
                    and edge.get("to") in app_state_ids
                ),
                "memory_elements": sum(
                    len(rendered[state_id]["elements"]) for state_id in app_state_ids
                ),
                "described_elements": len(description_by_element),
                "simulated_tasks": len(simulated_tasks),
                "model_calls": len(author.usage),
                "embedding_vectors": len(vectors),
            },
            "source_run_log_audit": {
                "task_name": str(source.get("task_name") or ""),
                "goal": str(source.get("goal") or ""),
                "action_types": source_action_labels,
                "click_or_text_steps": source_element_labels,
                "source_package": source_package,
                "source_actions_injected": False,
                "utg_is_memory_source": True,
                "path_coverage": "not_injected_and_requires_online_alignment",
            },
        }
        _write_json(root / "node_filtered_elements.json", node_filtered)
        _write_json(root / "element_description.json", element_description)
        _write_json(root / "embedded_elements_desc.json", embedded)
        _write_json(root / "app_state_summary.json", summaries)
        _write_json(root / "simulated_tasks.json", simulated_tasks)
        _write_json(root / "memory_build_report.json", report)
        _write_json(root / "model_usage.json", author.usage)
        return {**report, "memory_root": str(root)}
    finally:
        author.close()


def validate_autodroid_memory(
    memory_root: str | Path,
    *,
    source_run_log: str | Path,
    task_name: str,
) -> dict[str, Any]:
    """Validate an explicitly supplied native memory without selecting history."""

    root = Path(memory_root).expanduser().resolve()
    report_path = root / "memory_build_report.json"
    if not report_path.is_file():
        raise FileNotFoundError("autodroid_memory_build_report_missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != AUTODROID_MEMORY_SCHEMA:
        raise ValueError("autodroid_memory_schema_mismatch")
    if str(report.get("model") or "") != require_formal_model():
        raise ValueError("autodroid_memory_model_mismatch")
    source_path = Path(source_run_log).expanduser().resolve()
    if str(report.get("source_run_log_sha256") or "") != sha256_file(source_path):
        raise ValueError("autodroid_memory_source_run_log_mismatch")
    if str(report.get("source_run_log_audit", {}).get("task_name") or "") != str(
        task_name
    ):
        raise ValueError("autodroid_memory_task_mismatch")
    for filename in AUTODROID_MEMORY_FILES:
        if not (root / filename).is_file():
            raise FileNotFoundError(f"autodroid_memory_file_missing:{filename}")
    return report
