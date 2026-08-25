"""Prepare one immutable AppAgent demo memory from a canonical RunLog."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import importlib
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import time
from typing import Any

from PIL import Image

from omniflow.core.androidworld_accessibility import androidworld_forest_xml
from omniflow.core.trajectory import require_complete_source_run_log
from src.integrations.android_world.host import (
    androidworld_elements_xml,
    androidworld_observation_package,
    androidworld_observation_xml,
)
from src.integrations.appagent import (
    APPAGENT_ACTION_TYPES,
    APPAGENT_OFFICIAL_REVISION,
    OfficialAppAgentRuntime,
    appagent_elements_from_xml,
    appagent_record_line,
    build_appagent_teacher_source,
    ground_appagent_teacher_action,
    mark_appagent_teacher_target_interactive,
    seal_appagent_memory,
)


def _appagent_observation_xml(observation: dict[str, Any]) -> str:
    forest_xml = androidworld_observation_xml(observation)
    # Canonical RunLogs keep native AndroidWorld accessibility forests as the
    # structured ``{windows: ...}`` payload.  The generic observation helper
    # intentionally only accepts an already-rendered XML string, so convert
    # the native forest at this boundary before AppAgent's XML parser sees it.
    # Without this, valid source traces become an empty XML document and the
    # prep path reports ``no_interactive_elements`` even though the forest has
    # clickable nodes.
    forest = observation.get("forest")
    if not forest_xml and forest:
        pixels = observation.get("pixels")
        if not isinstance(pixels, dict):
            pixels = observation.get("screenshot")
        pixels = pixels if isinstance(pixels, dict) else {}
        width = pixels.get("width") or pixels.get("display_width") or 1000
        height = pixels.get("height") or pixels.get("display_height") or 1000
        forest_xml = androidworld_forest_xml(
            forest,
            screen_size=(int(width), int(height)),
        ).strip()
    elements = observation.get("ui_elements")
    if not forest_xml or not isinstance(elements, list) or not elements:
        return forest_xml
    missing_interactive_identity = any(
        bool(
            str(
                element.get("resource_name")
                or element.get("resource_id")
                or element.get("content_description")
                or element.get("text")
                or ""
            ).strip()
        )
        and bool(
            element.get("is_clickable")
            or element.get("is_focusable")
            or element.get("is_editable")
            or element.get("is_long_clickable")
        )
        and not any(
            str(value or "").strip() in forest_xml
            for value in (
                element.get("resource_name"),
                element.get("resource_id"),
                element.get("content_description"),
                element.get("text"),
            )
        )
        for element in elements
        if isinstance(element, dict)
    )
    if missing_interactive_identity:
        return androidworld_elements_xml(elements).strip()
    return forest_xml


_APPAGENT_AUXILIARY_PACKAGE_PREFIXES = ("com.google.android.inputmethod.",)
_APPAGENT_LAUNCHER_PACKAGE_PREFIXES = (
    "com.google.android.apps.nexuslauncher",
    "com.android.launcher",
    "com.google.android.launcher",
)


def _is_appagent_launcher_package(package: str) -> bool:
    normalized = str(package or "").strip()
    return any(
        normalized == prefix or normalized.startswith(prefix + ".")
        for prefix in _APPAGENT_LAUNCHER_PACKAGE_PREFIXES
    )


def _appagent_demo_package(
    observation: dict[str, Any],
    source_package: str,
) -> str:
    """Treat the active IME as an auxiliary window of the source app."""

    package = androidworld_observation_package(observation)
    if (
        source_package
        and package
        and package != source_package
        and package.startswith(_APPAGENT_AUXILIARY_PACKAGE_PREFIXES)
    ):
        return source_package
    return package


def _chat_completions_url(base_url: str) -> str:
    normalized = str(base_url or "").strip().rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return normalized + "/chat/completions"


def _write_runtime_config(
    path: Path,
    *,
    api_key: str,
    endpoint: str,
    model: str,
    timeout_sec: float,
    max_tokens: int = 512,
) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("AppAgent source generation requires PyYAML") from exc

    payload = {
        "MODEL": "OpenAI",
        "OPENAI_API_BASE": endpoint,
        "OPENAI_API_KEY": api_key,
        "OPENAI_API_MODEL": model,
        # Keep the temporary authoring call bounded; the staged model adapter
        # also disables provider-side reasoning explicitly.
        "MAX_TOKENS": int(max_tokens),
        "THINKING": "disabled",
        "TEMPERATURE": 0.0,
        "REQUEST_INTERVAL": 0.0,
        "DASHSCOPE_API_KEY": "",
        "QWEN_MODEL": model,
        "ANDROID_SCREENSHOT_DIR": "/sdcard",
        "ANDROID_XML_DIR": "/sdcard",
        "DOC_REFINE": False,
        "MAX_ROUNDS": 20,
        "DARK_MODE": False,
        "MIN_DIST": 30,
        "OMNIFLOW_REQUEST_TIMEOUT_SEC": float(timeout_sec),
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def run_official_document_generation(
    *,
    appagent_root: str | Path,
    workspace_root: str | Path,
    app_name: str,
    demo_name: str,
    log_path: str | Path,
    usage_path: str | Path,
    model: str,
    timeout_sec: float = 180.0,
) -> dict[str, Any]:
    """Run the pinned official generator once and record exact API usage."""

    root = Path(appagent_root).expanduser().resolve()
    workspace = Path(workspace_root).expanduser().resolve()
    output_log = Path(log_path).expanduser().resolve()
    output_usage = Path(usage_path).expanduser().resolve()
    normalized_model = str(model or "").strip()
    if not normalized_model:
        raise ValueError("appagent_document_model_required")
    if float(timeout_sec) <= 0:
        raise ValueError("appagent_document_timeout_must_be_positive")

    # This verifies both the exact official revision and its runtime modules.
    OfficialAppAgentRuntime(root)
    api_key = str(
        os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or ""
    ).strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for AppAgent docs")
    endpoint = _chat_completions_url(
        str(
            os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OMNIFLOW_OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
    )
    output_log.parent.mkdir(parents=True, exist_ok=True)
    output_usage.parent.mkdir(parents=True, exist_ok=True)
    if output_log.exists() or output_usage.exists():
        raise FileExistsError("immutable_appagent_doc_generation_output_exists")

    started = time.monotonic()
    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="appagent-doc-config-") as temporary:
        temporary_root = Path(temporary)
        _write_runtime_config(
            temporary_root / "config.yaml",
            api_key=api_key,
            endpoint=endpoint,
            model=normalized_model,
            timeout_sec=float(timeout_sec),
            max_tokens=512,
        )
        previous_cwd = Path.cwd()
        previous_argv = list(sys.argv)
        scripts_dir = root / "scripts"
        sys.path.insert(0, str(scripts_dir))
        official_model = importlib.import_module("model")
        original_post = official_model.requests.post

        def instrumented_post(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("timeout", float(timeout_sec))
            response = original_post(*args, **kwargs)
            payload = response.json()
            if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
                message = str(payload["error"].get("message") or "").strip()
                if message:
                    errors.append(message)
            usage_payload = payload.get("usage") if isinstance(payload, dict) else None
            usage = dict(usage_payload) if isinstance(usage_payload, dict) else {}
            record = {
                "model": (payload.get("model") if isinstance(payload, dict) else None)
                or normalized_model,
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
            }
            if record["total_tokens"] <= 0:
                record["total_tokens"] = (
                    record["prompt_tokens"] + record["completion_tokens"]
                )
            with output_usage.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            class ResponseProxy:
                def json(self) -> Any:
                    return payload

                def __getattr__(self, name: str) -> Any:
                    return getattr(response, name)

            return ResponseProxy()

        try:
            official_model.requests.post = instrumented_post
            os.chdir(temporary_root)
            sys.argv = [
                str(root / "scripts" / "document_generation.py"),
                "--app",
                str(app_name),
                "--demo",
                str(demo_name),
                "--root_dir",
                str(workspace),
            ]
            with output_log.open("x", encoding="utf-8") as log_handle:
                with redirect_stdout(log_handle), redirect_stderr(log_handle):
                    runpy.run_path(
                        str(root / "scripts" / "document_generation.py"),
                        run_name="__main__",
                    )
        finally:
            official_model.requests.post = original_post
            sys.argv = previous_argv
            os.chdir(previous_cwd)
            try:
                sys.path.remove(str(scripts_dir))
            except ValueError:
                pass

    if not output_usage.exists():
        output_usage.touch()
    usage_rows = [
        json.loads(line)
        for line in output_usage.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    models = sorted(
        {
            str(row.get("model") or "").strip()
            for row in usage_rows
            if str(row.get("model") or "").strip()
        }
    )
    docs_root = workspace / "apps" / str(app_name) / "demo_docs"
    return {
        "official_appagent_revision": APPAGENT_OFFICIAL_REVISION,
        "app_name": str(app_name),
        "demo_name": str(demo_name),
        "docs_root": str(docs_root),
        "docs_count": len([path for path in docs_root.glob("*.txt") if path.is_file()]),
        "model_calls": len(usage_rows),
        "models": models,
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in usage_rows),
        "completion_tokens": sum(
            int(row.get("completion_tokens") or 0) for row in usage_rows
        ),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in usage_rows),
        "wall_sec": round(time.monotonic() - started, 6),
        "log_path": str(output_log),
        "usage_path": str(output_usage),
        "retry_count": 0,
        "errors": errors,
    }


def _appagent_source_observation(
    source: dict[str, Any],
    *,
    step_index: int,
    after: bool,
) -> dict[str, Any]:
    steps = list(source["steps"])
    step = steps[step_index]
    if not after:
        return dict(step["observation"])
    if isinstance(step.get("next_observation"), dict):
        return dict(step["next_observation"])
    if step_index + 1 < len(steps):
        return dict(steps[step_index + 1]["observation"])
    if isinstance(source.get("final_observation"), dict):
        return dict(source["final_observation"])
    raise ValueError(f"appagent_source_after_observation_missing:{step_index}")


def _write_appagent_state(
    *,
    observation: dict[str, Any],
    demo_root: Path,
    demo_name: str,
    state_index: int,
    source_step_index: int,
    phase: str,
    runtime: OfficialAppAgentRuntime,
    source_run_log: Path,
) -> tuple[str, list[Any]]:
    pixels = observation.get("screenshot")
    if pixels is None:
        pixels = observation.get("pixels")
    if not isinstance(pixels, dict):
        raise ValueError(
            f"appagent_source_screenshot_missing:{source_step_index}:{phase}"
        )
    screenshot = _resolve_appagent_screenshot(
        pixels,
        source_run_log=source_run_log,
    )
    xml_text = _appagent_observation_xml(observation)
    if not xml_text:
        raise ValueError(f"appagent_source_xml_missing:{source_step_index}:{phase}")
    base_name = f"{demo_name}_{state_index}"
    (demo_root / "xml" / f"{base_name}.xml").write_text(
        xml_text,
        encoding="utf-8",
    )
    raw_path = demo_root / "raw_screenshots" / f"{base_name}.png"
    with Image.open(screenshot) as image:
        image.convert("RGB").save(raw_path)
    elements = appagent_elements_from_xml(xml_text, min_dist=runtime.min_dist)
    runtime.draw_elements(
        raw_path,
        demo_root / "labeled_screenshots" / f"{base_name}.png",
        elements,
        record_mode=True,
    )
    return xml_text, elements


def _require_appagent_observation_evidence(
    observation: dict[str, Any],
    *,
    source_step_index: int,
    phase: str,
    source_run_log: Path,
) -> None:
    pixels = observation.get("screenshot")
    if pixels is None:
        pixels = observation.get("pixels")
    if not isinstance(pixels, dict):
        raise ValueError(
            f"appagent_source_screenshot_missing:{source_step_index}:{phase}"
        )
    screenshot = _resolve_appagent_screenshot(
        pixels,
        source_run_log=source_run_log,
    )
    if not _appagent_observation_xml(observation):
        raise ValueError(f"appagent_source_xml_missing:{source_step_index}:{phase}")


def _resolve_appagent_screenshot(
    pixels: dict[str, Any],
    *,
    source_run_log: Path,
) -> Path:
    screenshot = Path(str(pixels.get("path") or "")).expanduser().resolve()
    expected_sha256 = str(pixels.get("sha256") or "").strip().lower()
    if screenshot.is_file():
        return screenshot
    remapped = _remap_appagent_data_path(
        screenshot,
        source_run_log=source_run_log,
    )
    if remapped is not None:
        return remapped
    source_object = source_run_log.expanduser().resolve()
    sha256_root = source_object.parent.parent
    suffix = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }.get(str(pixels.get("mime_type") or "").strip())
    if suffix is None:
        return screenshot
    candidate = sha256_root / expected_sha256[:2] / f"{expected_sha256}{suffix}"
    return candidate.resolve() if candidate.is_file() else screenshot


def _remap_appagent_data_path(
    screenshot: Path,
    *,
    source_run_log: Path,
) -> Path | None:
    """Resolve an evidence path copied from another checkout's data root.

    RunLogs are portable evidence, but older archives stored absolute paths in
    their screenshot entries.  On 4090 those paths still point at the
    author's local checkout.  The source RunLog itself is authoritative for
    the active data root, so remap only the shared ``data/androidworld``
    suffix and never fall back to a source-device coordinate or synthetic
    image.
    """

    def data_anchor(path: Path) -> int | None:
        parts = path.parts
        for index in range(len(parts) - 2, -1, -1):
            if parts[index] == "data" and parts[index + 1] == "androidworld":
                return index
        return None

    source_index = data_anchor(source_run_log)
    screenshot_index = data_anchor(screenshot)
    if source_index is None or screenshot_index is None:
        return None
    source_root = Path(*source_run_log.parts[: source_index + 1])
    candidate = source_root.joinpath(
        *screenshot.parts[screenshot_index + 1 :]
    ).resolve()
    return candidate if candidate.is_file() else None


def convert_runlog_to_appagent_memory(
    *,
    source_run_log: str | Path,
    appagent_root: str | Path,
    memory_root: str | Path,
    model: str,
    source_method: str = "runlog_direct",
) -> dict[str, Any]:
    """Convert one successful RunLog into AppAgent's official demo layout."""

    normalized_model = str(model or "").strip()
    if not normalized_model:
        raise ValueError("appagent_source_model_required")
    source_path = Path(source_run_log).expanduser().resolve()
    source = require_complete_source_run_log(
        json.loads(source_path.read_text(encoding="utf-8"))
    )
    source_seed = int(source.get("seed") or 0)
    root = Path(memory_root).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"immutable_appagent_memory_exists:{root}")
    teacher_source = build_appagent_teacher_source(
        source_path,
        task_name=str(source["task_name"]),
        source_seed=source_seed,
    )
    all_demo_records = [
        record
        for record in teacher_source["actions"]
        if str(record["action"].get("type") or "") in APPAGENT_ACTION_TYPES
    ]
    if not all_demo_records:
        raise ValueError("appagent_official_demo_actions_required")
    source_package = next(
        (
            str(record["action"].get("params", {}).get("package_name") or "")
            for record in teacher_source["actions"]
            if str(record["action"].get("type") or "") == "open_app"
        ),
        "",
    )
    # AndroidWorld source actions often retain the human-facing app label
    # (for example, ``Audio Recorder``) while observations expose the real
    # package. Resolve the source identity from the first non-auxiliary demo
    # page so the IME does not become a second AppAgent application.
    if "." not in source_package:
        observed_packages: list[str] = []
        for record in all_demo_records:
            observation = _appagent_source_observation(
                source,
                step_index=int(record["source_step_index"]),
                after=False,
            )
            observed_package = androidworld_observation_package(observation)
            if (
                observed_package
                and not observed_package.startswith(
                    _APPAGENT_AUXILIARY_PACKAGE_PREFIXES
                )
                and not _is_appagent_launcher_package(observed_package)
            ):
                observed_packages.append(observed_package)
        if observed_packages:
            source_package = max(
                set(observed_packages),
                key=lambda package: (observed_packages.count(package), package),
            )
    packages = []
    for record in all_demo_records:
        observation = _appagent_source_observation(
            source,
            step_index=int(record["source_step_index"]),
            after=False,
        )
        package_name = _appagent_demo_package(observation, source_package)
        if package_name and package_name not in packages:
            packages.append(package_name)
    package_name = (
        source_package
        if source_package in packages
        else next(
            (
                package
                for package in packages
                if package and not _is_appagent_launcher_package(package)
            ),
            next((value for value in packages if value), ""),
        )
    )
    if not package_name:
        raise ValueError("appagent_source_package_missing")
    demo_records = [
        record
        for record in all_demo_records
        if _appagent_demo_package(
            _appagent_source_observation(
                source,
                step_index=int(record["source_step_index"]),
                after=False,
            ),
            source_package,
        )
        == package_name
    ]
    if not demo_records:
        raise ValueError("appagent_source_package_actions_missing")
    for record in demo_records:
        step_index = int(record["source_step_index"])
        _require_appagent_observation_evidence(
            _appagent_source_observation(
                source,
                step_index=step_index,
                after=False,
            ),
            source_step_index=step_index,
            phase="before",
            source_run_log=source_path,
        )
    final_step_index = int(demo_records[-1]["source_step_index"])
    _require_appagent_observation_evidence(
        _appagent_source_observation(
            source,
            step_index=final_step_index,
            after=True,
        ),
        source_step_index=final_step_index,
        phase="after",
        source_run_log=source_path,
    )
    runtime = OfficialAppAgentRuntime(appagent_root)
    selected_demo_keys = {
        (
            int(record["source_step_index"]),
            int(record["source_action_index"]),
        )
        for record in demo_records
    }
    app_name = package_name.rsplit(".", 1)[-1]
    demo_name = f"demo_{source['task_name']}_seed{source_seed}"
    demo_root = root / "apps" / app_name / "demos" / demo_name
    for directory in (
        demo_root / "raw_screenshots",
        demo_root / "xml",
        demo_root / "labeled_screenshots",
    ):
        directory.mkdir(parents=True, exist_ok=False)
    (demo_root / "task_desc.txt").write_text(str(source["goal"]), encoding="utf-8")
    # The source RunLog may include a launcher click before the target app is
    # opened.  Keep the complete teacher action count for provenance, while
    # sealing only actions belonging to the selected AppAgent app.
    teacher_source["demo_action_count"] = len(demo_records)
    teacher_source_path = root / "teacher_source.json"
    teacher_source_path.write_text(
        json.dumps(teacher_source, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    record_lines: list[str] = []
    trace_rows: list[dict[str, Any]] = []
    for cursor, record in enumerate(teacher_source["actions"], 1):
        action = dict(record["action"])
        action_type = str(action.get("type") or "")
        trace = {
            "teacher_cursor": cursor,
            "source_step_index": int(record["source_step_index"]),
            "source_action_index": int(record["source_action_index"]),
            "action_type": action_type,
            "source_coordinates_used": False,
            "conversion_mode": "canonical_runlog_offline",
        }
        if (
            action_type in APPAGENT_ACTION_TYPES
            and (
                int(record["source_step_index"]),
                int(record["source_action_index"]),
            )
            in selected_demo_keys
        ):
            state_index = len(record_lines) + 1
            xml_text, _ = _write_appagent_state(
                observation=_appagent_source_observation(
                    source,
                    step_index=int(record["source_step_index"]),
                    after=False,
                ),
                demo_root=demo_root,
                demo_name=demo_name,
                state_index=state_index,
                source_step_index=int(record["source_step_index"]),
                phase="before",
                runtime=runtime,
                source_run_log=source_path,
            )
            xml_text = mark_appagent_teacher_target_interactive(xml_text, action)
            state_xml_path = demo_root / "xml" / f"{demo_name}_{state_index}.xml"
            state_xml_path.write_text(xml_text, encoding="utf-8")
            runtime.draw_elements(
                demo_root / "raw_screenshots" / f"{demo_name}_{state_index}.png",
                demo_root / "labeled_screenshots" / f"{demo_name}_{state_index}.png",
                appagent_elements_from_xml(xml_text, min_dist=runtime.min_dist),
                record_mode=True,
            )
            grounded = ground_appagent_teacher_action(
                xml_text,
                action,
                min_dist=runtime.min_dist,
            )
            record_lines.append(appagent_record_line(action, grounded))
            trace.update(
                {
                    "tag": grounded.tag,
                    "uid": grounded.uid,
                    "match_reason": grounded.match_reason,
                }
            )
        trace_rows.append(trace)
    final_record = demo_records[-1]
    _write_appagent_state(
        observation=_appagent_source_observation(
            source,
            step_index=int(final_record["source_step_index"]),
            after=True,
        ),
        demo_root=demo_root,
        demo_name=demo_name,
        state_index=len(record_lines) + 1,
        source_step_index=int(final_record["source_step_index"]),
        phase="after",
        runtime=runtime,
        source_run_log=source_path,
    )
    (demo_root / "record.txt").write_text(
        "\n".join([*record_lines, "stop"]) + "\n",
        encoding="utf-8",
    )
    (demo_root / "teacher_trace.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in trace_rows),
        encoding="utf-8",
    )
    document_root = root / "_document_generation"
    document_root.mkdir()
    document_log = document_root / "document_generation.log"
    document_usage = document_root / "document_generation_usage.jsonl"
    prep_started = time.monotonic()
    document_result = run_official_document_generation(
        appagent_root=appagent_root,
        workspace_root=root,
        app_name=app_name,
        demo_name=demo_name,
        log_path=document_log,
        usage_path=document_usage,
        model=normalized_model,
    )
    if document_result.get("errors"):
        error = str(document_result["errors"][0]).replace("\n", " ").strip()
        raise ValueError(
            "appagent_document_model_request_failed:"
            f"model={normalized_model}:error={error}"
        )
    prep_wall_sec = max(round(time.monotonic() - prep_started, 6), 0.000001)
    manifest = seal_appagent_memory(
        memory_root=root,
        app_name=app_name,
        demo_name=demo_name,
        teacher_source=teacher_source_path,
        source_result=None,
        document_generation_log=document_log,
        document_generation_usage=document_usage,
        task_name=str(source["task_name"]),
        source_episode_wall_sec=0.0,
        document_generation_wall_sec=float(document_result.get("wall_sec") or 0.0),
        prep_wall_sec=prep_wall_sec,
        source_method=source_method,
        document_generation_model=normalized_model,
        conversion_mode="canonical_runlog_offline",
        native_memory_evidence=None,
    )
    return {
        "schema_version": "omniflow.runlog-memory-conversion.v1",
        "method": "appagent",
        "task_name": str(source["task_name"]),
        "source_run_log": str(source_path),
        "memory_root": str(root),
        "manifest": manifest,
    }
