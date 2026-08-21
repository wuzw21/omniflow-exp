from __future__ import annotations

import argparse
import base64
import copy
import dataclasses
import datetime
import hashlib
import importlib
import inspect
import io
import json
import logging
import math
import os
from pathlib import Path
import pickle
import random
import re
import signal
import subprocess
import sys
import tempfile
import time
from time import perf_counter
from typing import Any, Callable, Sequence
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

from omniflow.vlm.model_config import resolve_openai_compatible_config
from omniflow.vlm.usage import token_usage_status
from src.experiment.observation_evidence import (
    persist_target_run_evidence,
    transfer_state_coverage_audit,
)
from src.experiment.performance_metrics import (
    PerformanceMetrics,
    write_performance_metrics,
)
from src.experiment.protocol import (
    FORMAL_MODEL_BASE_URL,
    FORMAL_MODEL_ENDPOINT_PROFILE,
    MAX_STEPS,
)
from src.integrations.android_world.agent import (
    MODE_OMNIFLOW,
    build_agent,
)
from src.integrations.android_world.environment import (
    AndroidWorldEnvironmentConfig,
    AndroidWorldExperimentEnvironment,
)
from src.integrations.android_world.host import make_agent_result
from src.integrations.android_world.methods import (
    MethodAdapterContext,
    default_method_adapter_registry,
    reuse_metrics,
)
from src.integrations.runlog import import_run_log, project_androidworld_step_actions

OMNIFLOW_ROOT = Path(__file__).resolve().parents[3]
logger = logging.getLogger(__name__)
DEFAULT_RAW_REPLAY_ACTION_WAIT_SECONDS = 1.0
ANDROIDWORLD_A11Y_FORWARDER_PACKAGE = (
    "com.google.androidenv.accessibilityforwarder"
)
ANDROIDWORLD_A11Y_FORWARDER_SHA256 = (
    "97a56a544e44d79f9b3181fc7dbdd72cffa908efd3d53c82afad1773061a350a"
)
OOB_CONTROL_PACKAGE = "cn.com.omnimind.bot.debug"
OOB_CONTROL_ACCESSIBILITY_SERVICE = (
    "cn.com.omnimind.bot.debug/"
    "cn.com.omnimind.accessibility.service.AssistsService"
)
ANDROID_PERMISSION_DENY_RESOURCE_IDS = (
    "com.android.permissioncontroller:id/permission_deny_button",
    "com.android.permissioncontroller:id/permission_deny_and_dont_ask_again_button",
)
DEFAULT_ANDROIDWORLD_ADB_FILE_TRANSFER_TIMEOUT_SEC = 300.0
DEFAULT_ANDROIDWORLD_SETUP_TIMEOUT_SEC = 300.0
DEFAULT_ANDROIDWORLD_APK_INSTALL_TIMEOUT_SEC = 180.0
_LLM_USAGE_COUNTER_KEYS = (
    "model_calls",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "responses_with_usage",
    "responses_without_usage",
    "failed_calls",
)


def _model_base_url_for_profile(profile: str | None) -> str | None:
    resolved_profile = str(profile or "auto").strip().lower()
    if resolved_profile == FORMAL_MODEL_ENDPOINT_PROFILE:
        return FORMAL_MODEL_BASE_URL
    return None


def _normalize_androidworld_setup_label(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return unicodedata.normalize("NFKC", value).translate(
        str.maketrans(
            {
                "\u2018": "'",
                "\u2019": "'",
                "\u201a": "'",
                "\u201b": "'",
                "\u201c": '"',
                "\u201d": '"',
                "\u201e": '"',
                "\u201f": '"',
            }
        )
    )


def _app_chooser_clicks(
    elements: Sequence[Any],
    *,
    app_label: str,
) -> tuple[str, ...]:
    labels = {
        _normalize_androidworld_setup_label(str(label or "").strip())
        for element in elements
        for label in (
            getattr(element, "text", None),
            getattr(element, "content_description", None),
        )
        if str(label or "").strip()
    }
    if "Just once" not in labels:
        return ()
    normalized_app_label = _normalize_androidworld_setup_label(app_label.strip())
    if f"Open with {normalized_app_label}" in labels:
        return ("Just once",)
    if "Open with" not in labels or normalized_app_label not in labels:
        return ()
    return (normalized_app_label, "Just once")


def _raw_replay_visible_setup_recovery(
    agent: Any,
    *,
    goal_text: str,
) -> str | None:
    if "chrome" not in goal_text.casefold():
        return None
    environment = getattr(agent, "env", None)
    controller = getattr(environment, "controller", None)
    get_ui_elements = getattr(controller, "get_ui_elements", None)
    if not callable(get_ui_elements):
        return None
    elements = get_ui_elements() or []
    chooser_clicks = _app_chooser_clicks(elements, app_label="Chrome")
    labels = {
        _normalize_androidworld_setup_label(str(label or "").strip())
        for element in elements
        for label in (
            getattr(element, "text", None),
            getattr(element, "content_description", None),
        )
        if str(label or "").strip()
    }
    if not chooser_clicks and not (
        {"Keep Google", "OK", "Search with Sogou"}.issubset(labels)
    ):
        return None
    tools_module = importlib.import_module("android_world.env.tools")
    tool_controller = tools_module.AndroidToolController(controller)
    if chooser_clicks:
        for label in chooser_clicks:
            tool_controller.click_element(label)
        return "android_app_chooser:Chrome"
    tool_controller.click_element("OK")
    return "chrome_first_run:OK"


def _patch_androidworld_optional_setup_click() -> tuple[Any, Any] | None:
    try:
        tools_module = importlib.import_module("android_world.env.tools")
    except ModuleNotFoundError:
        return None
    controller_type = getattr(tools_module, "AndroidToolController", None)
    original = getattr(controller_type, "click_element", None)
    if controller_type is None or not callable(original):
        return None

    def click_element(controller: Any, element_text: str) -> Any:
        try:
            return original(controller, element_text)
        except ValueError as error:
            message = str(error)
            normalized_label = _normalize_androidworld_setup_label(element_text)
            missing_target = "Target text" in message and "not found" in message
            empty_a11y_tree = (
                normalized_label == "NEXT"
                and "Invalid element index" in message
            )
            if normalized_label == "NEXT" and missing_target:
                activity = str(
                    getattr(controller._env, "foreground_activity_name", "") or ""
                ).strip()
                packages = {
                    str(getattr(element, "package_name", "") or "").strip()
                    for element in controller._env.get_ui_elements() or ()
                }
                camera_is_foreground = activity.startswith(
                    "com.android.camera2/"
                )
                camera_is_visible = "com.android.camera2" in packages
                launcher_is_foreground = activity.startswith(
                    "com.google.android.apps.nexuslauncher/"
                )
                if camera_is_foreground or camera_is_visible or launcher_is_foreground:
                    logger.info(
                        "AndroidWorld Camera setup is already settled; skipping "
                        "stale NEXT lookup"
                    )
                    return None
            if normalized_label == "Skip" and missing_target:
                elements = controller._env.get_ui_elements() or []
                chooser_clicks = _app_chooser_clicks(
                    elements,
                    app_label="Contacts",
                )
                if chooser_clicks:
                    logger.info(
                        "AndroidWorld Contacts setup is resolving the system "
                        "app chooser before onboarding"
                    )
                    for label in chooser_clicks:
                        original(controller, label)
                    return original(controller, "Skip")
            if missing_target or empty_a11y_tree:
                # Camera can still be publishing its accessibility tree while
                # AndroidWorld starts the app. Give the official setup a
                # short bounded chance to observe the real NEXT button before
                # applying the existing optional-click fallback.
                for _ in range(6):
                    time.sleep(0.5)
                    try:
                        return original(controller, element_text)
                    except ValueError as retry_error:
                        message = str(retry_error)
                        retry_missing_target = (
                            "Target text" in message and "not found" in message
                        )
                        retry_empty_a11y_tree = (
                            normalized_label == "NEXT"
                            and "Invalid element index" in message
                        )
                        if not retry_missing_target and not retry_empty_a11y_tree:
                            break
            if not missing_target and not empty_a11y_tree:
                raise
            if normalized_label == "OK":
                activity = str(
                    getattr(controller._env, "foreground_activity_name", "") or ""
                ).strip()
                if not activity.startswith("net.gsantner.markor/"):
                    raise
                logger.info(
                    "AndroidWorld Markor setup is already complete; skipping "
                    "absent OK button"
                )
                return None
            if normalized_label == "NEXT":
                activity = str(
                    getattr(controller._env, "foreground_activity_name", "") or ""
                ).strip()
                packages = {
                    str(getattr(element, "package_name", "") or "").strip()
                    for element in controller._env.get_ui_elements() or ()
                }
                camera_is_foreground = activity.startswith("com.android.camera2/")
                camera_is_visible = "com.android.camera2" in packages
                if camera_is_foreground or camera_is_visible:
                    logger.info(
                        "AndroidWorld Camera setup is already complete; "
                        "skipping absent NEXT button"
                    )
                    return None
                launcher_is_foreground = activity.startswith(
                    "com.google.android.apps.nexuslauncher/"
                )
                non_app_packages = {
                    "",
                    "android",
                    "com.android.systemui",
                    "com.google.android.apps.nexuslauncher",
                }
                if launcher_is_foreground or (
                    packages and packages.issubset(non_app_packages)
                ):
                    logger.info(
                        "AndroidWorld optional app setup did not expose NEXT "
                        "and remains on the launcher; continuing"
                    )
                    return None
            if normalized_label in {
                "All files",
                "Allow access to manage all files",
            }:
                activity = str(
                    getattr(controller._env, "foreground_activity_name", "") or ""
                ).strip()
                packages = {
                    str(getattr(element, "package_name", "") or "").strip()
                    for element in controller._env.get_ui_elements() or ()
                }
                gallery_visible = activity.startswith(
                    "com.simplemobiletools.gallery.pro/"
                ) or "com.simplemobiletools.gallery.pro" in packages
                if gallery_visible:
                    logger.info(
                        "AndroidWorld Gallery setup is already complete; "
                        "skipping absent %s button",
                        normalized_label,
                    )
                    return None
            if normalized_label != "Don't allow":
                raise
            elements = controller._env.get_ui_elements() or []
            packages = {
                str(getattr(element, "package_name", "") or "").strip()
                for element in elements
            }
            if any(package.endswith(".permissioncontroller") for package in packages):
                raise
            app_packages = {
                package
                for package in packages
                if package
                and package not in {"android", "com.android.systemui"}
                and not package.endswith(".permissioncontroller")
            }
            if not app_packages:
                raise
            logger.info(
                "AndroidWorld setup notification permission is already settled; "
                "skipping absent %s button on packages=%s",
                element_text,
                ",".join(sorted(app_packages)),
            )
            return None

    controller_type.click_element = click_element
    return controller_type, original


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read_env_text(name: str) -> str | None:
    value = str(os.environ.get(name) or "").strip()
    return value or None


def to_serializable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return to_serializable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_serializable(item) for item in value]
    enum_value = getattr(value, "value", None)
    if enum_value is not None and enum_value is not value:
        return to_serializable(enum_value)
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return {
            str(key): to_serializable(item)
            for key, item in attributes.items()
            if not str(key).startswith("_")
        }
    return str(value)


def build_response_acceptance_detail(
    *,
    success: bool,
    validator: Any,
    error_code: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    error = str(error_code or error_message or "").strip() or None
    validator_success = validator.get("success") if isinstance(validator, dict) else None
    validator_error = (
        str(validator.get("error") or "").strip()
        if isinstance(validator, dict)
        else ""
    )
    generic = success is True
    androidworld = bool(generic and validator_success is True)
    return {
        "generic": {
            "accepted": generic,
            "failure_reason": None if generic else error or "success_false",
            "agent_stop_condition": None if generic else "vlm_task_success_false",
        },
        "androidworld": {
            "accepted": androidworld,
            "official_validator_required": True,
            "failure_reason": (
                None
                if androidworld
                else error
                or validator_error
                or (
                    "androidworld_validator_missing"
                    if not isinstance(validator, dict)
                    else "androidworld_validator_failed"
                )
            ),
            "agent_stop_condition": (
                None if androidworld else "runtime_response_acceptance_failed"
            ),
        },
    }


def _official_hint_text(value: Any, *, max_len: int = 100) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    lowered = text.casefold()
    if not text or any(
        marker in lowered
        for marker in (
            "<hierarchy",
            "<?xml",
            "screenshot",
            ".png",
            ".jpg",
            ".jpeg",
            "coordinate",
            "bounds=",
        )
    ):
        return ""
    if re.search(r"[\[(]\s*\d{1,4}\s*,\s*\d{1,4}(?:\s*,\s*\d{1,4}){0,2}\s*[\])]", text):
        return ""
    if len(text) > max_len:
        text = text[: max(0, max_len - 3)].rstrip() + "..."
    return text


def _official_hint_node(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    allowed_keys = {
        "node_id",
        "class_name",
        "text",
        "content_description",
        "resource_id",
        "package_name",
    }
    if any(str(key) not in allowed_keys for key in value):
        return ""
    labels = (
        ("id", "node_id"),
        ("class", "class_name"),
        ("text", "text"),
        ("content-desc", "content_description"),
        ("resource-id", "resource_id"),
        ("package", "package_name"),
    )
    fields = []
    for label, key in labels:
        text = _official_hint_text(value.get(key), max_len=120)
        if text:
            fields.append(f"{label}={text!r}")
    return ", ".join(fields)


def _render_official_semantic_hint_step(index: int, step: Any) -> str:
    if not isinstance(step, dict):
        return ""
    allowed_keys = {
        "action",
        "target",
        "app",
        "direction",
        "key",
        "purpose",
        "source_node",
    }
    if any(str(key) not in allowed_keys for key in step):
        return ""
    action = str(step.get("action") or "").strip().lower()
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", action):
        return ""
    target = _official_hint_text(step.get("target"))
    app = _official_hint_text(step.get("app"), max_len=80)
    direction = _official_hint_text(step.get("direction"), max_len=24)
    key = _official_hint_text(step.get("key"), max_len=24)
    purpose = _official_hint_text(step.get("purpose"), max_len=320)
    source_node = _official_hint_node(step.get("source_node"))
    prefix = f"{index}. "
    suffix_parts = []
    if source_node:
        suffix_parts.append(f"Source accessibility node: {source_node}.")
    if purpose:
        suffix_parts.append(f"Purpose observed in the successful run: {purpose}")
    suffix = (" " + " ".join(suffix_parts)) if suffix_parts else ""
    if action in {"open_app", "launch_app"}:
        return prefix + (f"Open app {app}." if app else "Open the relevant app.") + suffix
    if action in {"click", "tap", "long_press"}:
        if not target and not source_node:
            return ""
        verb = "Long press" if action == "long_press" else "Click"
        return prefix + (
            f"{verb} the UI target described as '{target}'."
            if target
            else f"{verb} the UI object identified by the source accessibility node."
        ) + suffix
    if action in {"input_text", "type_text", "set_text", "enter_text"}:
        if not target and not source_node:
            return ""
        return prefix + (
            f"Enter the value requested by the current task into '{target}'."
            if target
            else "Enter the value requested by the current task into the text field identified by the source accessibility node."
        ) + suffix
    if action in {"press_key", "key_event"}:
        return prefix + (f"Press key {key}." if key else "Press the relevant key.") + suffix
    if action in {"swipe", "scroll"}:
        if direction and target:
            return prefix + f"Scroll {direction} on '{target}'." + suffix
        if direction:
            return prefix + f"Scroll {direction}." + suffix
        return prefix + "Scroll as needed to reveal the next target." + suffix
    if action in {"wait", "sleep"}:
        return prefix + "Wait for the screen to update." + suffix
    return prefix + (
        f"Use action {action} on '{target}'." if target else f"Use action {action}."
    ) + suffix


def _render_official_reference_prompt(steps: list[Any]) -> tuple[str, int]:
    rendered = [
        line
        for line in (
            _render_official_semantic_hint_step(index + 1, step)
            for index, step in enumerate(steps)
        )
        if line
    ]
    if not rendered:
        return "", 0
    return (
        "\n".join(
            [
                "The following is a successful source-device action sequence that you may use as semantic reference. Each step preserves the operated UI object, source accessibility-node evidence when available, and the observed purpose:",
                *rendered,
                "This sequence is reference guidance, not a replay command. On every current screen, locate the equivalent semantic control; resource IDs, node IDs, classes, text, and layout may differ across devices. Independently choose one next action, never reuse source coordinates, and substitute values from the current task rather than old source values.",
            ]
        ),
        len(rendered),
    )


def _load_official_agent_goal_hint(
    source_action_hint_path: str | Path | None,
    *,
    max_steps: int | None = None,
) -> tuple[str, dict[str, Any] | None]:
    path_text = str(source_action_hint_path or "").strip()
    if not path_text:
        return "", None
    hint_path = Path(path_text).expanduser()
    meta: dict[str, Any] = {
        "path": str(hint_path),
        "exists": hint_path.exists(),
        "steps": 0,
        "rendered_steps": 0,
        "hint_mode": "official_goal_reference_trace",
        "applied_to_official_agent_goal": False,
    }
    if not hint_path.exists():
        meta["error"] = "hint file does not exist"
        return "", meta
    try:
        payload = json.loads(hint_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        meta["error"] = str(exc)
        return "", meta
    if not isinstance(payload, dict) or payload.get("schema_version") not in {
        "omniflow.t3a_semantic_hint.v1",
        "omniflow.t3a_semantic_hint.v2",
    }:
        meta["error"] = "unsupported or unsafe source hint schema"
        return "", meta
    steps = payload.get("steps")
    if not isinstance(steps, list):
        meta["error"] = "source hint steps must be a list"
        return "", meta
    meta["steps"] = len(steps)
    selected_steps = (
        steps
        if max_steps is None
        else steps[: max(1, int(max_steps))]
    )
    hint_text, rendered_steps = _render_official_reference_prompt(selected_steps)
    meta["rendered_steps"] = rendered_steps
    meta["truncated"] = len(selected_steps) < len(steps)
    if not hint_text:
        meta["error"] = "source hint contains no safe semantic steps"
        return "", meta
    meta["applied_to_official_agent_goal"] = True
    meta["rendered_chars"] = len(hint_text)
    return hint_text, meta


class _ExperimentAgentAdapter:
    """Add experiment-only recording and optional goal context to an agent."""

    def __init__(
        self,
        agent: Any,
        *,
        recording_session: Any,
        goal_hint: str = "",
        max_steps: int | None = None,
        prepare_after_reset: Callable[[], None] | None = None,
    ):
        self._agent = agent
        self._recording_session = recording_session
        self._goal_hint = str(goal_hint or "").strip()
        self._max_steps = max(1, int(max_steps)) if max_steps is not None else None
        self._prepare_after_reset = prepare_after_reset
        self._completed_steps = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)

    def reset(self, go_home: bool = False) -> None:
        self._completed_steps = 0
        reset = self._agent.reset
        reset_parameters = inspect.signature(reset).parameters.values()
        supports_go_home = any(
            parameter.name == "go_home"
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in reset_parameters
        )
        if supports_go_home:
            reset(go_home=go_home)
        else:
            reset()
        if self._prepare_after_reset is not None:
            self._prepare_after_reset()

    def step(self, goal: str) -> Any:
        if self._max_steps is not None and self._completed_steps >= self._max_steps:
            from android_world.agents import base_agent

            return base_agent.AgentInteractionResult(
                True,
                {
                    "summary": "Experiment step budget reached before another model call.",
                    "experiment_step_budget_reached": True,
                },
            )
        ensure_ready = getattr(
            self._recording_session.env,
            "ensure_accessibility_forwarder_ready",
            None,
        )
        if callable(ensure_ready) and not _is_oob_control_backend():
            ensure_ready()
        self._recording_session.start_episode()
        effective_goal = str(goal or "")
        if self._goal_hint:
            effective_goal = f"{effective_goal}\n\n{self._goal_hint}"
        result = self._agent.step(effective_goal)
        self._completed_steps += 1
        if self._max_steps is not None and self._completed_steps >= self._max_steps:
            result.done = True
            result.data["experiment_step_budget_reached"] = True
        return result


def _coerce_int(value: Any) -> int:
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    try:
        return int(value or 0)
    except Exception:
        return 0


def _coerce_float(value: Any) -> float:
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _androidworld_episode_value(result: Any, key: str) -> Any:
    if not isinstance(result, dict):
        return None
    episode_data = result.get("episode_data")
    if not isinstance(episode_data, dict):
        return None
    return episode_data.get(key)


class _OpenAICompatibleMultimodalWrapper:
    """AndroidWorld LLM wrapper backed by an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        *,
        model_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_retry: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> None:
        resolved_model = (
            str(model_name or "").strip()
            or str(os.environ.get("OPENAI_MODEL") or "").strip()
            or str(os.environ.get("OMNIFLOW_PLANNER_MODEL") or "").strip()
            or "GLM-4.6V"
        )
        resolved_base_url = (
            str(base_url or "").strip()
            or str(os.environ.get("OPENAI_BASE_URL") or "").strip()
            or "https://api.openai.com/v1"
        )
        resolved_api_key = (
            str(api_key or "").strip()
            or str(os.environ.get("OPENAI_API_KEY") or "").strip()
            or str(os.environ.get("DASHSCOPE_API_KEY") or "").strip()
        )
        if not resolved_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for official AndroidWorld M3A/T3A "
                "OpenAI-compatible execution."
            )
        try:
            retry_count = int(max_retry or 0)
        except Exception:
            retry_count = 3
        if retry_count <= 0:
            retry_count = 3

        self.model = resolved_model
        self.base_url = resolved_base_url
        self.endpoint = self._chat_completions_url(resolved_base_url)
        self.openai_api_key = resolved_api_key
        self.max_retry = min(retry_count, 5)
        self.temperature = float(temperature)
        configured_max_tokens = os.environ.get("OMNIFLOW_ANDROIDWORLD_LLM_MAX_TOKENS")
        try:
            self.max_tokens = max(1, int(configured_max_tokens)) if configured_max_tokens else int(max_tokens)
        except (TypeError, ValueError):
            self.max_tokens = int(max_tokens)
        self.model_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.responses_with_usage = 0
        self.responses_without_usage = 0
        self.failed_calls = 0
        self.last_error: str | None = None
        self.request_records: list[dict[str, Any]] = []

    @staticmethod
    def _chat_completions_url(base_url: str) -> str:
        normalized = str(base_url or "").strip().rstrip("/")
        if not normalized:
            normalized = "https://api.openai.com/v1"
        if normalized.endswith("/chat/completions"):
            return normalized
        return f"{normalized}/chat/completions"

    @staticmethod
    def _encode_image(image: Any) -> bytes:

        from PIL import Image

        if isinstance(image, Image.Image):
            pil_image = image
        else:
            pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        return buffer.getvalue()

    def predict(self, text_prompt: str) -> tuple[str, bool | None, Any]:
        return self.predict_mm(text_prompt, [])

    def predict_mm(self, text_prompt: str, images: list[Any]) -> tuple[str, bool | None, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": str(text_prompt)}]
        encoded_images = [self._encode_image(image) for image in list(images or [])]
        for image_bytes in encoded_images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
                    },
                }
            )
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}",
        }
        wait_seconds = float(os.environ.get("OMNIFLOW_OPENAI_RETRY_WAIT_SECONDS") or 2.0)
        last_response: dict[str, Any] | None = None
        request_record = {
            "request_index": len(self.request_records),
            "kind": (
                "action_consistency"
                if "SkyMark action-consistency review" in str(text_prompt)
                else
                "action"
                if "Your Answer:" in str(text_prompt)
                else "summary"
                if "Summary of this step:" in str(text_prompt)
                else "unknown"
            ),
            "prompt": str(text_prompt),
            "image_payloads": encoded_images,
            "image_sha256": [hashlib.sha256(value).hexdigest() for value in encoded_images],
            "started_at": utc_now_iso(),
            "duration_ms": None,
            "response_text": None,
            "response_metadata": None,
        }
        request_started = perf_counter()
        for attempt_index in range(max(1, self.max_retry)):
            try:
                request = urllib.request.Request(
                    self.endpoint,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                self.model_calls += 1
                with urllib.request.urlopen(
                    request,
                    timeout=float(os.environ.get("OMNIFLOW_OPENAI_TIMEOUT_SEC") or 120.0),
                ) as response:
                    response_text = response.read().decode("utf-8", errors="replace")
                    response_payload = json.loads(response_text)
                last_response = {
                    "ok": True,
                    "model": response_payload.get("model") or self.model,
                    "usage": response_payload.get("usage") or {},
                    "id": response_payload.get("id"),
                    "object": response_payload.get("object"),
                    "created": response_payload.get("created"),
                }
                usage = (
                    response_payload.get("usage")
                    if isinstance(response_payload.get("usage"), dict)
                    else {}
                )
                if usage:
                    self.responses_with_usage += 1
                else:
                    self.responses_without_usage += 1
                prompt_tokens = _coerce_int(usage.get("prompt_tokens"))
                completion_tokens = _coerce_int(usage.get("completion_tokens"))
                total_tokens = _coerce_int(usage.get("total_tokens"))
                if total_tokens <= 0:
                    total_tokens = prompt_tokens + completion_tokens
                self.prompt_tokens += prompt_tokens
                self.completion_tokens += completion_tokens
                self.total_tokens += total_tokens
                choices = response_payload.get("choices")
                if isinstance(choices, list) and choices:
                    message = (
                        choices[0].get("message")
                        if isinstance(choices[0], dict)
                        else {}
                    )
                    if isinstance(message, dict):
                        content_text = message.get("content")
                        if isinstance(content_text, list):
                            content_text = "\n".join(
                                str(item.get("text") or "")
                                for item in content_text
                                if isinstance(item, dict)
                            )
                        response_text = str(content_text or "")
                        request_record["duration_ms"] = max(
                            0.0, (perf_counter() - request_started) * 1000.0
                        )
                        request_record["response_text"] = response_text
                        request_record["response_metadata"] = last_response
                        self.request_records.append(request_record)
                        return response_text, None, last_response
                self.last_error = "OpenAI-compatible response did not include choices"
                break
            except urllib.error.HTTPError as exc:
                self.failed_calls += 1
                body = exc.read().decode("utf-8", errors="replace")
                self.last_error = f"HTTP {exc.code}: {body[:1000]}"
                last_response = {"ok": False, "status": exc.code, "body": body[:4000]}
            except Exception as exc:  # noqa: BLE001
                self.failed_calls += 1
                self.last_error = str(exc)
                last_response = {"ok": False, "error": str(exc)}
            if attempt_index < self.max_retry - 1:
                time.sleep(wait_seconds)
                wait_seconds *= 2.0
        request_record["duration_ms"] = max(
            0.0, (perf_counter() - request_started) * 1000.0
        )
        request_record["response_text"] = "Error calling LLM"
        request_record["response_metadata"] = last_response
        self.request_records.append(request_record)
        return "Error calling LLM", None, last_response

    def get_usage_summary(self) -> dict[str, Any]:
        summary = {
            "model": self.model,
            "base_url": self.base_url,
            "endpoint": self.endpoint,
            "model_calls": int(self.model_calls),
            "prompt_tokens": int(self.prompt_tokens),
            "completion_tokens": int(self.completion_tokens),
            "total_tokens": int(self.total_tokens),
            "responses_with_usage": int(self.responses_with_usage),
            "responses_without_usage": int(self.responses_without_usage),
            "failed_calls": int(self.failed_calls),
            "last_error": self.last_error,
        }
        summary["token_usage_status"] = token_usage_status(summary)
        return summary


def _get_agent_llm_usage(agent: Any) -> dict[str, Any]:
    tracker = getattr(agent, "_omniflow_llm_usage_tracker", None)
    if tracker is None:
        tracker = getattr(agent, "llm", None)
    if tracker is None:
        return {}
    get_summary = getattr(tracker, "get_usage_summary", None)
    if callable(get_summary):
        payload = get_summary()
        return dict(payload or {}) if isinstance(payload, dict) else {}
    payload: dict[str, Any] = {}
    for key in ("model", "base_url", "endpoint", "last_error"):
        value = getattr(tracker, key, None)
        if value not in (None, ""):
            payload[key] = value
    for key in _LLM_USAGE_COUNTER_KEYS:
        payload[key] = _coerce_int(getattr(tracker, key, 0))
    return payload


def _diff_llm_usage(
    after: dict[str, Any],
    before: dict[str, Any] | None = None,
) -> dict[str, Any]:
    before = before or {}
    delta: dict[str, Any] = {}
    for key in ("model", "base_url", "endpoint", "last_error"):
        if after.get(key) not in (None, ""):
            delta[key] = after.get(key)
    for key in _LLM_USAGE_COUNTER_KEYS:
        delta[key] = max(0, _coerce_int(after.get(key)) - _coerce_int(before.get(key)))
    if delta["total_tokens"] <= 0:
        delta["total_tokens"] = delta["prompt_tokens"] + delta["completion_tokens"]
    return delta


def _is_validator_success(row: dict[str, Any]) -> bool:
    validator = row.get("androidworld_validator_result")
    if isinstance(validator, dict):
        return bool(validator.get("success"))
    return bool(row.get("success"))


def _official_validator_used(row: dict[str, Any]) -> bool:
    if "official_validator_used" in row:
        return bool(row.get("official_validator_used"))
    if "uses_androidworld_official_validator" in row:
        return bool(row.get("uses_androidworld_official_validator"))
    if row.get("validator") == "androidworld_official":
        return True
    validator = row.get("androidworld_validator_result")
    if not isinstance(validator, dict):
        return False
    if "uses_androidworld_official_validator" in validator:
        return bool(validator.get("uses_androidworld_official_validator"))
    return validator.get("validator") == "androidworld_official"


def _official_validator_success(row: dict[str, Any]) -> bool:
    return _official_validator_used(row) and _is_validator_success(row)


def _result_has_official_validator_conclusion(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    reward = result.get("is_successful")
    if isinstance(reward, bool):
        return True
    if str(result.get("exception_info") or result.get("error") or "").strip():
        return False
    if reward is None:
        return False
    try:
        float(reward)
    except (TypeError, ValueError):
        return False
    return True


def _compact_relocation_diagnostic(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        key: value.get(key)
        for key in (
            "schema_version",
            "dir",
            "manifest_path",
            "source_xml_path",
            "target_xml_path",
            "target_mapping_xml_path",
            "source_screenshot_path",
            "target_screenshot_path",
            "source_screenshot_available",
            "target_screenshot_available",
            "failure_reason",
            "error",
        )
        if value.get(key) not in (None, "")
    }


def _extract_relocation_diagnostics(value: Any, *, limit: int = 20) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(candidate: Any) -> None:
        compact = _compact_relocation_diagnostic(candidate)
        if not compact:
            return
        identity = str(
            compact.get("manifest_path")
            or compact.get("dir")
            or compact.get("target_xml_path")
            or compact
        )
        if identity in seen:
            return
        seen.add(identity)
        diagnostics.append(compact)

    def _walk(item: Any, depth: int = 0) -> None:
        if len(diagnostics) >= limit or depth > 8:
            return
        if isinstance(item, dict):
            if "relocation_diagnostic" in item:
                _add(item.get("relocation_diagnostic"))
            if str(item.get("schema_version") or "").startswith(
                "omniflow.relocation_failure."
            ):
                _add(item)
            for nested in item.values():
                _walk(nested, depth + 1)
        elif isinstance(item, list):
            for nested in item:
                _walk(nested, depth + 1)

    _walk(value)
    return diagnostics


def _iter_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _canonical_action_stats(row: dict[str, Any]) -> tuple[int, int]:
    canonical_run = row.get("canonical_run")
    if not isinstance(canonical_run, dict):
        return 0, 0
    total = 0
    success = 0
    for step in canonical_run.get("steps") or []:
        if not isinstance(step, dict):
            continue
        act_result = step.get("act_result")
        if not isinstance(act_result, dict):
            continue
        total += 1
        if bool(act_result.get("success")):
            success += 1
    return success, total


def _rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        to_serializable(payload), ensure_ascii=False, sort_keys=True, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _task_params_provenance(task: Any) -> tuple[dict[str, Any], str]:
    """Return the initialized AndroidWorld task params and a stable digest."""
    try:
        raw_params = dict(getattr(task, "params", {}) or {})
    except Exception:
        raw_params = {}
    serialized = to_serializable(raw_params)
    if not isinstance(serialized, dict):
        serialized = {}
    return serialized, _stable_hash(serialized)


def _summarize_task_results(
    *,
    task_results_path: Path,
    checkpoint_dir: str,
    agent: str,
    tasks: Sequence[str],
) -> dict[str, Any]:
    rows = _iter_jsonl_rows(task_results_path)
    total_tasks = len(rows)
    official_validator_tasks = sum(1 for row in rows if _official_validator_used(row))
    successful_tasks = sum(1 for row in rows if _official_validator_success(row))
    total_duration_ms = sum(_coerce_float(row.get("duration_ms")) for row in rows)
    total_actions = sum(_coerce_int(row.get("actions_executed")) for row in rows)
    total_model_calls = sum(_coerce_int(row.get("model_calls")) for row in rows)
    runtime_integrity_errors = [
        str(row.get("runtime_integrity_error") or "").strip()
        for row in rows
        if str(row.get("runtime_integrity_error") or "").strip()
    ]
    total_tokens = sum(_coerce_int(row.get("total_tokens")) for row in rows)
    successful_action_steps = 0
    total_action_steps = 0
    official_validator_success_actions = 0
    per_task: list[dict[str, Any]] = []
    for row in rows:
        action_completed, action_total = _canonical_action_stats(row)
        successful_action_steps += action_completed
        total_action_steps += action_total
        actions_executed = _coerce_int(row.get("actions_executed"))
        official_validator_used = _official_validator_used(row)
        official_validator_success = _official_validator_success(row)
        if official_validator_success:
            official_validator_success_actions += actions_executed
        duration_ms = _coerce_float(row.get("duration_ms"))
        per_task.append(
            {
                "task_name": row.get("task_name"),
                "goal": row.get("goal"),
                "official_validator_used": official_validator_used,
                "official_validator_success": official_validator_success,
                "duration_ms": round(duration_ms, 3),
                "actions_executed": actions_executed,
                "action_completed_rate": _rate(action_completed, action_total),
                "tool_calls": _coerce_int(row.get("model_calls")),
                "model_calls": _coerce_int(row.get("model_calls")),
                "prompt_tokens": _coerce_int(row.get("prompt_tokens")),
                "completion_tokens": _coerce_int(row.get("completion_tokens")),
                "total_tokens": _coerce_int(row.get("total_tokens")),
                "error": row.get("error"),
            }
        )

    summary = {
        "schema_version": "omniflow.androidworld_run_summary.v3",
        "agent": agent,
        "tasks_requested": list(tasks),
        "task_results_path": str(task_results_path),
        "checkpoint_dir": str(checkpoint_dir),
        "task_count": total_tasks,
        "official_validator_task_count": official_validator_tasks,
        "official_validator_success_count": successful_tasks,
        "official_validator_failure_count": max(
            0,
            official_validator_tasks - successful_tasks,
        ),
        "official_validator_success_rate": _rate(
            successful_tasks,
            official_validator_tasks,
        ),
        "runtime_integrity_error_count": len(runtime_integrity_errors),
        "runtime_integrity_errors": runtime_integrity_errors,
        "official_validator_coverage_rate": _rate(
            official_validator_tasks,
            total_tasks,
        ),
        "duration_ms": round(total_duration_ms, 3),
        "avg_duration_ms": round(total_duration_ms / max(1, total_tasks), 3),
        "actions_executed": total_actions,
        "avg_actions_per_task": round(total_actions / max(1, total_tasks), 3),
        "avg_ms_per_action": round(total_duration_ms / max(1, total_actions), 3),
        "single_step_execution_accuracy": _rate(
            successful_action_steps, total_action_steps
        ),
        "single_step_execution_completed_count": successful_action_steps,
        "single_step_execution_total": total_action_steps,
        "validator_weighted_action_accuracy": _rate(
            official_validator_success_actions, total_actions
        ),
        "tool_calls": total_model_calls,
        "tokens": total_tokens,
        "per_task": per_task,
    }

    print(
        "[omniflow] summary: "
        f"official_validator={successful_tasks}/{official_validator_tasks} "
        f"coverage={official_validator_tasks}/{total_tasks} "
        f"duration={total_duration_ms / 1000.0:.1f}s "
        f"actions={total_actions} "
        f"step_acc={summary['single_step_execution_accuracy']} "
        f"tool_calls={total_model_calls} tokens={total_tokens}"
    )
    return summary


def _default_adb_path() -> str:
    candidates = [
        os.path.expanduser("~/Library/Android/sdk/platform-tools/adb"),
        os.path.expanduser("~/Android/Sdk/platform-tools/adb"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return ""


def _add_android_world_path(android_world_root: Path) -> None:
    root = str(android_world_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _rehydrate_task_params(
    *,
    params: dict[str, object],
    task_type: type[object] | None = None,
) -> dict[str, object]:
    """Restore AndroidWorld task params that were serialized through JSON."""

    hydrated = dict(params)
    if task_type is not None and task_type.__name__ == "MarkorTranscribeReceipt":
        if "seed" not in hydrated:
            raise ValueError("MarkorTranscribeReceipt task params require seed")
        random_state = random.getstate()
        try:
            random.seed(int(hydrated["seed"]))
            generated = task_type.generate_random_params()
        finally:
            random.setstate(random_state)
        for key, value in generated.items():
            if key == "img":
                continue
            if key not in hydrated or hydrated[key] != value:
                raise ValueError(
                    "MarkorTranscribeReceipt generated params do not match "
                    f"canonical source: {key}"
                )
        image = generated.get("img")
        if image is None:
            raise ValueError("MarkorTranscribeReceipt generated img is missing")
        hydrated["img"] = image
    serialized_rows = [
        row
        for key in ("row_objects", "noise_row_objects")
        for row in (hydrated.get(key) if isinstance(hydrated.get(key), list) else [])
        if isinstance(row, dict)
    ]
    if not serialized_rows:
        return hydrated
    from android_world.task_evals.utils import sqlite_schema_utils

    def expense_row(row: object) -> object:
        if not isinstance(row, dict):
            return row
        allowed = {
            "name",
            "amount",
            "category",
            "note",
            "created_date",
            "modified_date",
            "expense_id",
        }
        payload = {key: row[key] for key in allowed if key in row}
        return sqlite_schema_utils.Expense(**payload)

    def calendar_event(row: object) -> object:
        if not isinstance(row, dict):
            return row
        allowed = {
            "start_ts",
            "end_ts",
            "title",
            "location",
            "description",
            "repeat_interval",
            "repeat_rule",
            "reminder_1_minutes",
            "reminder_2_minutes",
            "reminder_3_minutes",
            "reminder_1_type",
            "reminder_2_type",
            "reminder_3_type",
            "repeat_limit",
            "repetition_exceptions",
            "attendees",
            "import_id",
            "time_zone",
            "flags",
            "event_type",
            "parent_id",
            "last_updated",
            "source",
            "availability",
            "color",
            "type",
            "id",
        }
        payload = {key: row[key] for key in allowed if key in row}
        return sqlite_schema_utils.CalendarEvent(**payload)

    def recipe_row(row: object) -> object:
        if not isinstance(row, dict):
            return row
        allowed = {
            "title",
            "description",
            "servings",
            "preparationTime",
            "source",
            "ingredients",
            "directions",
            "favorite",
            "imageName",
            "recipeId",
        }
        payload = {key: row[key] for key in allowed if key in row}
        return sqlite_schema_utils.Recipe(**payload)

    def hydrate_row(row: object) -> object:
        if not isinstance(row, dict):
            return row
        keys = set(row)
        if keys & {"start_ts", "end_ts", "repeat_rule", "event_type"}:
            return calendar_event(row)
        if keys & {"ingredients", "directions", "preparationTime", "recipeId"}:
            return recipe_row(row)
        if keys & {"amount", "expense_id", "created_date", "modified_date"}:
            return expense_row(row)
        return row

    for key in ("row_objects", "noise_row_objects"):
        rows = hydrated.get(key)
        if isinstance(rows, list):
            hydrated[key] = [hydrate_row(row) for row in rows]
    return hydrated


def _androidworld_a11y_forwarder_installed(
    *,
    console_port: int,
    adb_path: str,
) -> bool:
    serial = f"emulator-{int(console_port)}"
    adb_bin = os.path.expanduser(str(adb_path or "").strip()) or "adb"
    try:
        result = subprocess.run(
            [
                adb_bin,
                "-s",
                serial,
                "shell",
                "pm",
                "path",
                ANDROIDWORLD_A11Y_FORWARDER_PACKAGE,
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and any(
        line.strip().startswith("package:")
        for line in str(result.stdout or "").splitlines()
    )


def _ensure_androidworld_a11y_forwarder(
    *, console_port: int, adb_path: str, apk_path: str
) -> bool:
    if _androidworld_a11y_forwarder_installed(
        console_port=console_port, adb_path=adb_path
    ):
        return True
    path = Path(str(apk_path or "")).expanduser().resolve()
    if not path.is_file():
        return False
    adb_bin = os.path.expanduser(str(adb_path or "").strip()) or "adb"
    result = subprocess.run(
        [
            adb_bin,
            "-s",
            f"emulator-{int(console_port)}",
            "install",
            "-r",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "AndroidWorld accessibility forwarder local install failed: "
            + str(result.stdout or result.stderr or "unknown error").strip()
        )
    return _androidworld_a11y_forwarder_installed(
        console_port=console_port, adb_path=adb_path
    )


def _ensure_oob_control_app(*, console_port: int, adb_path: str) -> bool:
    """Restore OOB after AndroidWorld snapshot setup resets user packages."""

    adb_bin = os.path.expanduser(str(adb_path or "").strip()) or "adb"
    serial = f"emulator-{int(console_port)}"
    package_path = subprocess.run(
        [adb_bin, "-s", serial, "shell", "pm", "path", OOB_CONTROL_PACKAGE],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    apk_path = Path(str(os.environ.get("OMNIFLOW_OOB_APK") or "")).expanduser()
    if package_path.returncode != 0 or not any(
        line.strip().startswith("package:")
        for line in str(package_path.stdout or "").splitlines()
    ):
        if not apk_path.is_file():
            raise RuntimeError(
                "oob_control_package_missing_after_androidworld_setup:"
                f"{serial}; set OMNIFLOW_OOB_APK"
            )
        installed = subprocess.run(
            [adb_bin, "-s", serial, "install", "-r", "-t", str(apk_path)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if installed.returncode != 0:
            raise RuntimeError(
                "oob_control_package_install_failed:"
                f"{serial}:{str(installed.stdout or installed.stderr or '').strip()}"
            )
    enabled = subprocess.run(
        [adb_bin, "-s", serial, "shell", "settings", "get", "secure", "enabled_accessibility_services"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    services = [
        value
        for value in str(enabled.stdout or "").strip().split(":")
        if value and value != "null"
    ]
    if OOB_CONTROL_ACCESSIBILITY_SERVICE not in services:
        services.append(OOB_CONTROL_ACCESSIBILITY_SERVICE)
        update = subprocess.run(
            [
                adb_bin,
                "-s",
                serial,
                "shell",
                "settings",
                "put",
                "secure",
                "enabled_accessibility_services",
                ":".join(services),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if update.returncode != 0:
            raise RuntimeError(f"oob_control_accessibility_enable_failed:{serial}")
    enabled_flag = subprocess.run(
        [adb_bin, "-s", serial, "shell", "settings", "put", "secure", "accessibility_enabled", "1"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if enabled_flag.returncode != 0:
        raise RuntimeError(f"oob_control_accessibility_flag_failed:{serial}")
    return True


def _is_oob_control_backend() -> bool:
    backend = str(
        os.environ.get("OMNIFLOW_ANDROIDWORLD_CONTROL_BACKEND", "androidworld")
    ).strip().lower()
    return backend in {"oob", "omniflow", "oob_control"}


def _load_androidworld_env(
    *,
    env_launcher: Any,
    console_port: int,
    adb_path: str,
    grpc_port: int,
    install_a11y_forwarding_app: bool,
) -> Any:
    """Load the official environment without native A11y in OOB mode."""
    # The official AndroidEnv wrapper downloads its forwarder APK on every
    # cold environment start.  Transient/incomplete HTTP responses are common
    # on the source-device network and otherwise abort a whole collection
    # attempt before AndroidWorld setup can begin.  Keep the official loader
    # and wrapper, but retry only this download seam.
    from android_env.wrappers import a11y_grpc_wrapper

    original_forwarder_download = (
        a11y_grpc_wrapper._get_accessibility_forwarder_apk
    )

    def download_forwarder_with_retry() -> bytes:
        fd, temporary_name = tempfile.mkstemp(
            prefix="androidworld-a11y-forwarder.", suffix=".apk"
        )
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            subprocess.run(
                [
                    "curl", "-fL", "--retry", "3", "--retry-delay", "1",
                    "--connect-timeout", "10", "--max-time", "180",
                    "-o", str(temporary),
                    "https://storage.googleapis.com/android_env-tasks/2024.05.13-accessibility_forwarder.apk",
                ],
                check=True,
                timeout=240,
            )
            return temporary.read_bytes()
        except Exception as error:  # pragma: no cover - network dependent
            raise RuntimeError("accessibility_forwarder_download_failed") from error
        finally:
            temporary.unlink(missing_ok=True)

    a11y_grpc_wrapper._get_accessibility_forwarder_apk = (
        download_forwarder_with_retry
    )
    if not _is_oob_control_backend():
        try:
            return env_launcher.load_and_setup_env(
                console_port=console_port,
                emulator_setup=False,
                adb_path=adb_path,
                grpc_port=grpc_port,
                install_a11y_forwarding_app=install_a11y_forwarding_app,
            )
        finally:
            a11y_grpc_wrapper._get_accessibility_forwarder_apk = (
                original_forwarder_download
            )

    from android_world.env import android_world_controller

    original_wrapper = android_world_controller.apply_a11y_forwarder_app_wrapper
    android_world_controller.apply_a11y_forwarder_app_wrapper = (
        lambda raw_env, _install: raw_env
    )
    try:
        env = env_launcher.load_and_setup_env(
            console_port=console_port,
            emulator_setup=False,
            adb_path=adb_path,
            grpc_port=grpc_port,
            install_a11y_forwarding_app=False,
        )
    finally:
        android_world_controller.apply_a11y_forwarder_app_wrapper = original_wrapper
        a11y_grpc_wrapper._get_accessibility_forwarder_apk = (
            original_forwarder_download
        )

    # Do not allow an environment lifecycle fallback to silently read native
    # accessibility data after OOB has taken ownership of observe/act.
    controller = getattr(env, "controller", None)
    if controller is not None and hasattr(android_world_controller, "A11yMethod"):
        controller._a11y_method = android_world_controller.A11yMethod.NONE
    return env


@dataclasses.dataclass(frozen=True)
class AndroidWorldEnvironmentStartup:
    """The one AndroidWorld environment-startup result shared by all methods."""

    env: Any
    adb_output_patches: tuple[tuple[type[Any], Any], ...]


def prepare_androidworld_environment(
    *,
    env_launcher: Any,
    setup_module: Any,
    setup_apps: Sequence[Any],
    console_port: int,
    adb_path: str,
    grpc_port: int,
    install_a11y_forwarding_app: bool,
    perform_emulator_setup: bool = True,
    wait_for_a11y: bool = False,
    use_uiautomator: bool = False,
) -> AndroidWorldEnvironmentStartup:
    """Run the canonical AndroidWorld environment startup sequence.

    The action owner may differ (OmniFlow, an official baseline, or a human),
    but environment state must be prepared identically.  This function owns
    only startup; it never chooses or executes a task action.
    """

    env = _load_androidworld_env(
        env_launcher=env_launcher,
        console_port=int(console_port),
        adb_path=str(adb_path or ""),
        grpc_port=int(grpc_port),
        install_a11y_forwarding_app=bool(install_a11y_forwarding_app),
    )
    if use_uiautomator:
        from android_world.env import android_world_controller

        env.controller._a11y_method = (  # pylint: disable=protected-access
            android_world_controller.A11yMethod.UIAUTOMATOR
        )
    adb_output_patches = _patch_androidworld_adb_output_sanitizer(
        env.controller
    )
    if wait_for_a11y and not _is_oob_control_backend():
        _wait_for_androidworld_a11y(env)
    if perform_emulator_setup:
        logger.info(
            "Setting up AndroidWorld snapshots for selected apps: %s",
            ", ".join(
                str(getattr(app, "app_name", "") or "") for app in setup_apps
            )
            or "<none>",
        )
        controller = getattr(env, "controller", None)
        controller_type = None
        if _is_oob_control_backend():
            controller_type = importlib.import_module(
                "android_world.env.android_world_controller"
            )
        previous_a11y_method = getattr(controller, "_a11y_method", None)
        if controller_type is not None and controller is not None:
            controller._a11y_method = controller_type.A11yMethod.UIAUTOMATOR
        setup_apps_module = None
        original_app_download = None
        try:
            from android_world.env.setup_device import apps as setup_apps_module

            original_app_download = setup_apps_module.download_app_data

            def download_app_data_with_retry(file_name: str) -> str:
                """Download AndroidWorld app data with bounded curl retries."""
                cache_dir = Path(
                    setup_apps_module.file_utils.convert_to_posix_path(
                        setup_apps_module.file_utils.get_local_tmp_directory(),
                        "android_world",
                        "app_data",
                    )
                )
                destination = cache_dir / str(file_name)
                if destination.is_file():
                    return str(destination)
                cache_dir.mkdir(parents=True, exist_ok=True)
                fd, temporary_name = tempfile.mkstemp(
                    prefix=f".{destination.name}.",
                    suffix=".partial",
                    dir=cache_dir,
                )
                os.close(fd)
                temporary = Path(temporary_name)
                try:
                    subprocess.run(
                        [
                            "curl", "-fL", "--retry", "3", "--retry-delay", "1",
                            "--connect-timeout", "10", "--max-time", "180",
                            "-o", str(temporary),
                            f"https://storage.googleapis.com/gresearch/android_world/{file_name}",
                        ],
                        check=True,
                        timeout=240,
                    )
                    os.replace(temporary, destination)
                    return str(destination)
                finally:
                    temporary.unlink(missing_ok=True)

            setup_apps_module.download_app_data = download_app_data_with_retry
            _run_androidworld_setup_apps(
                env,
                setup_module=setup_module,
                setup_apps=tuple(setup_apps),
            )
        finally:
            if setup_apps_module is not None and original_app_download is not None:
                setup_apps_module.download_app_data = original_app_download
            if controller_type is not None and controller is not None:
                controller._a11y_method = previous_a11y_method
    if _is_oob_control_backend():
        _ensure_oob_control_app(console_port=int(console_port), adb_path=str(adb_path or ""))
    _prepare_androidworld_snapshot_restore(env, tuple(setup_apps))
    _reset_androidworld_file_picker_state(
        env,
        setup_module=setup_module,
        setup_apps=tuple(setup_apps),
    )
    if wait_for_a11y and not _is_oob_control_backend():
        # AndroidWorld setup can restart or rebind the forwarder after the
        # initial wait. Verify the post-setup state before the episode reset.
        _wait_for_androidworld_a11y(env)
    return AndroidWorldEnvironmentStartup(
        env=env,
        adb_output_patches=adb_output_patches,
    )


def start_androidworld_task_session(
    *,
    android_world_root: str | Path,
    task_name: str,
    task_params: dict[str, Any] | None,
    task_seed: int,
    console_port: int,
    adb_path: str,
    grpc_port: int,
    install_a11y_forwarding_app: bool = False,
    perform_emulator_setup: bool = True,
    use_uiautomator: bool = True,
) -> tuple[AndroidWorldEnvironmentStartup, Any]:
    """Start one official task for a human or an external baseline.

    This is the non-agent counterpart of the formal launcher.  It uses the
    same environment startup seam and the same AndroidWorld task
    ``initialize_task`` contract; the caller owns all subsequent decisions.
    """

    root = Path(android_world_root).expanduser().resolve()
    _add_android_world_path(root)
    from android_world import registry
    from android_world.env import env_launcher
    from android_world.env.setup_device import setup as setup_module

    task_types = registry.TaskRegistry().get_registry(family="android_world")
    task_type = task_types.get(str(task_name))
    if task_type is None:
        raise ValueError(f"unknown AndroidWorld task: {task_name}")
    app_names = {
        str(name).strip().lower()
        for name in getattr(task_type, "app_names", ())
    }
    setup_apps = tuple(
        app
        for app in setup_module._APPS
        if str(app.app_name).strip().lower() in app_names
    )
    startup = prepare_androidworld_environment(
        env_launcher=env_launcher,
        setup_module=setup_module,
        setup_apps=setup_apps,
        console_port=int(console_port),
        adb_path=str(adb_path or ""),
        grpc_port=int(grpc_port),
        install_a11y_forwarding_app=bool(install_a11y_forwarding_app),
        perform_emulator_setup=bool(perform_emulator_setup),
        use_uiautomator=bool(use_uiautomator),
    )
    from android_world.env import adb_utils

    _patch_androidworld_media_scanner_broadcast_compat(adb_utils)
    task_type.set_device_time(startup.env)
    params = dict(task_params or {})
    params = _rehydrate_task_params(params=params, task_type=task_type)
    params.setdefault("seed", int(task_seed))
    task = task_type(params)
    task.initialize_task(startup.env)
    return startup, task


def _androidworld_setup_apps_for_suite(
    suite: Any,
    *,
    get_app_mapping: Callable[[str], Any],
) -> tuple[Any, ...]:
    app_names: list[str] = []
    for task_instances in suite.values():
        for task_instance in task_instances:
            for app_name in getattr(task_instance, "app_names", ()):
                normalized = str(app_name or "").strip()
                if normalized and normalized not in app_names:
                    app_names.append(normalized)
    setup_apps = []
    missing = []
    for app_name in app_names:
        app = get_app_mapping(app_name)
        if app is None:
            missing.append(app_name)
        elif app not in setup_apps:
            setup_apps.append(app)
    if missing:
        raise RuntimeError(
            "AndroidWorld setup app mapping missing: " + ", ".join(missing)
        )
    return tuple(setup_apps)


def _repair_androidworld_chrome_first_run(
    env: Any,
    *,
    setup_module: Any,
    setup_apps: Sequence[Any],
) -> None:
    """Finish Chrome onboarding when the pinned setup IDs are unavailable.

    AndroidWorld's Chrome setup uses resource IDs that vary across the Chrome
    image bundled in the otherwise fixed emulator.  Keep the official setup
    as the first attempt, then use its semantic text controller as a narrow,
    task-independent environment recovery seam.
    """

    chrome_apps = [
        app
        for app in setup_apps
        if str(getattr(app, "app_name", "") or "").strip().casefold() == "chrome"
    ]
    if not chrome_apps:
        return
    tools = importlib.import_module("android_world.env.tools")
    controller = tools.AndroidToolController(env=env.controller)
    adb_utils = setup_module.adb_utils
    adb_utils.launch_app("chrome", env.controller)
    get_ui_elements = getattr(controller, "get_ui_elements", None)
    if callable(get_ui_elements):
        visible_labels = {
            str(value or "").strip().casefold()
            for element in get_ui_elements() or ()
            for value in (
                getattr(element, "text", None),
                getattr(element, "content_description", None),
            )
            if str(value or "").strip()
        }
        onboarding_labels = {
            "accept & continue",
            "no thanks",
            "next",
            "skip",
            "use chrome without an account",
            "use without an account",
        }
        onboarding_deadline = time.monotonic() + 20.0
        while (
            not visible_labels.intersection(onboarding_labels)
            and time.monotonic() < onboarding_deadline
        ):
            # A clean API-33 image can expose only Chrome's loading spinner for
            # several seconds before the first-run controls become accessible.
            # Keep the official setup app open until that state settles.
            time.sleep(0.5)
            visible_labels = {
                str(value or "").strip().casefold()
                for element in get_ui_elements() or ()
                for value in (
                    getattr(element, "text", None),
                    getattr(element, "content_description", None),
                )
                if str(value or "").strip()
            }
        if not visible_labels.intersection(onboarding_labels):
            adb_utils.close_app("chrome", env.controller)
            return
    try:
        try:
            controller.click_resource_id(
                (
                    "com.android.chrome:id/signin_fre_dismiss_button",
                    "com.android.chrome:id/terms_accept",
                )
            )
        except ValueError:
            for label in (
                "Accept & continue",
                "Accept & Continue",
                "No thanks",
                "Use without an account",
                "NEXT",
                "Next",
                "Skip",
            ):
                try:
                    controller.click_element(label)
                except ValueError:
                    continue
        for _ in range(2):
            try:
                controller.click_resource_id(
                    "com.android.chrome:id/negative_button",
                    timeout_sec=2.0,
                )
            except ValueError:
                for label in ("No thanks", "Not now", "Cancel"):
                    try:
                        controller.click_element(label)
                    except ValueError:
                        continue
                    break
            time.sleep(1.0)
    finally:
        adb_utils.close_app("chrome", env.controller)


def _repair_androidworld_setup_postconditions(
    env: Any,
    *,
    setup_module: Any,
    setup_apps: Sequence[Any],
) -> None:
    """Recover narrowly when an app setup snapshot was incomplete.

    Some clean target AVDs can return from the pinned setup helper without an
    installed third-party APK.  Reuse AndroidWorld's own installer/setup API
    for only the affected app, then create the directory required by the
    official AudioRecorder validator.
    """

    for app in setup_apps:
        package_name_getter = getattr(app, "package_name", None)
        if not callable(package_name_getter):
            continue
        try:
            package_name = (
                str(package_name_getter() or "").strip()
            )
        except Exception:
            package_name = ""
        if not package_name:
            installer = getattr(setup_module, "maybe_install_app", None)
            setup_app = getattr(setup_module, "setup_app", None)
            if not callable(installer) or not callable(setup_app):
                raise RuntimeError(
                    "AndroidWorld setup helpers unavailable for incomplete app"
                )
            installer(app, env)
            setup_app(app, env)
            package_name = str(package_name_getter() or "").strip()
        app_name = str(getattr(app, "app_name", "") or "").strip().casefold()
        if app_name == "audio recorder":
            device_constants = importlib.import_module(
                "android_world.env.device_constants"
            )
            response = setup_module.adb_utils.issue_generic_request(
                ["shell", "mkdir", "-p", device_constants.AUDIORECORDER_DATA],
                env.controller,
            )
            setup_module.adb_utils.check_ok(
                response,
                "Failed to prepare AudioRecorder records directory.",
            )


def _run_androidworld_setup_apps(
    env: Any,
    *,
    setup_module: Any,
    setup_apps: Sequence[Any],
) -> None:
    transfer_timeout_sec = _androidworld_adb_file_transfer_timeout_sec()
    setup_timeout_sec = _androidworld_setup_timeout_sec()
    file_utils = (
        importlib.import_module("android_world.utils.file_utils")
        if "android_world" in sys.modules
        else None
    )
    original_copy_file_to_device = (
        file_utils.copy_file_to_device if file_utils is not None else None
    )

    def copy_file_to_device(
        local_file_path: str,
        remote_file_path: str,
        controller: Any,
        timeout_sec: float | None = None,
    ) -> Any:
        if original_copy_file_to_device is None:
            raise RuntimeError("AndroidWorld file transfer module is unavailable")
        return original_copy_file_to_device(
            local_file_path,
            remote_file_path,
            controller,
            timeout_sec=_bounded_androidworld_adb_file_transfer_timeout(
                timeout_sec,
                default_timeout_sec=transfer_timeout_sec,
            ),
        )

    typography = str.maketrans(
        {
            "\u2018": "'",
            "\u2019": "'",
            "\u201a": "'",
            "\u201b": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u201e": '"',
            "\u201f": '"',
        }
    )

    def normalize_label(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        return unicodedata.normalize("NFKC", value).translate(typography)

    class SetupController:
        def __init__(self, controller: Any) -> None:
            self._controller = controller

        def __getattr__(self, name: str) -> Any:
            return getattr(self._controller, name)

        def get_ui_elements(self) -> list[Any]:
            normalized = []
            for element in self._controller.get_ui_elements() or ():
                element_copy = copy.copy(element)
                element_copy.text = normalize_label(getattr(element, "text", None))
                element_copy.content_description = normalize_label(
                    getattr(element, "content_description", None)
                )
                normalized.append(element_copy)
            return normalized

    class SetupEnvironment:
        def __init__(self, raw_env: Any) -> None:
            self._raw_env = raw_env
            self.controller = SetupController(raw_env.controller)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._raw_env, name)

    setup_env = SetupEnvironment(env)
    if file_utils is not None:
        file_utils.copy_file_to_device = copy_file_to_device
    original_adb_controller_install = _patch_androidworld_adb_controller_install_compat()
    original_install_apk = _patch_androidworld_apk_install_compat(setup_module)
    original_issue_generic_request = _patch_androidworld_chcon_compat(setup_module)
    optional_setup_patch = _patch_androidworld_optional_setup_click()
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)
    setup_started_at = time.monotonic()

    def setup_timeout_handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError(
            "AndroidWorld official app setup exceeded "
            f"{setup_timeout_sec:g} seconds"
        )

    try:
        signal.signal(signal.SIGALRM, setup_timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, setup_timeout_sec)
        setup_module.setup_apps(
            setup_env,
            app_list=tuple(setup_apps),
        )
        _repair_androidworld_setup_postconditions(
            setup_env,
            setup_module=setup_module,
            setup_apps=setup_apps,
        )
        _repair_androidworld_chrome_first_run(
            setup_env,
            setup_module=setup_module,
            setup_apps=setup_apps,
        )
        _prepare_androidworld_episode_apps(
            setup_env,
            setup_module=setup_module,
            setup_apps=setup_apps,
            save_snapshots=True,
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        previous_delay, previous_interval = previous_timer
        if previous_delay > 0:
            elapsed = time.monotonic() - setup_started_at
            signal.setitimer(
                signal.ITIMER_REAL,
                max(1e-6, previous_delay - elapsed),
                previous_interval,
            )
        if file_utils is not None:
            file_utils.copy_file_to_device = original_copy_file_to_device
        if original_install_apk is not None:
            setup_module.adb_utils.install_apk = original_install_apk
        if original_adb_controller_install is not None:
            controller_type, original_execute_command = original_adb_controller_install
            controller_type.execute_command = original_execute_command
        if original_issue_generic_request is not None:
            setup_module.adb_utils.issue_generic_request = (
                original_issue_generic_request
            )
        if optional_setup_patch is not None:
            controller_type, original_click_element = optional_setup_patch
            controller_type.click_element = original_click_element


def _patch_androidworld_adb_controller_install_compat() -> tuple[Any, Any] | None:
    """Retry AndroidWorld APK installs without an unsupported adb flag."""

    try:
        controller_module = importlib.import_module(
            "android_env.components.adb_controller"
        )
    except ImportError:
        return None
    controller_type = getattr(controller_module, "AdbController", None)
    original = getattr(controller_type, "execute_command", None)
    if controller_type is None or not callable(original):
        return None

    def execute_command(
        self: Any,
        args: list[str],
        timeout: float | None = None,
        device_specific: bool = True,
    ) -> bytes:
        normalized = [str(value) for value in args]
        resolved_timeout = timeout
        if normalized and normalized[0] == "install":
            resolved_timeout = max(
                float(timeout or 0.0),
                DEFAULT_ANDROIDWORLD_APK_INSTALL_TIMEOUT_SEC,
            )
        if "--bypass-low-target-sdk-block" not in normalized:
            return original(
                self,
                args,
                timeout=resolved_timeout,
                device_specific=device_specific,
            )
        try:
            return original(
                self,
                args,
                timeout=resolved_timeout,
                device_specific=device_specific,
            )
        except Exception as error:
            details = [repr(error), str(error)]
            for attribute in ("stdout", "stderr", "output", "cmd"):
                value = getattr(error, attribute, None)
                if value is not None:
                    details.append(
                        value.decode("utf-8", errors="replace")
                        if isinstance(value, bytes)
                        else str(value)
                    )
            if "Unknown option --bypass-low-target-sdk-block" not in "\n".join(
                details
            ):
                raise
            fallback_args = [
                value
                for value in normalized
                if value != "--bypass-low-target-sdk-block"
            ]
            return original(
                self,
                fallback_args,
                timeout=resolved_timeout,
                device_specific=device_specific,
            )

    controller_type.execute_command = execute_command
    return controller_type, original


def _patch_androidworld_apk_install_compat(setup_module: Any) -> Any | None:
    """Retry legacy APK installation without an unsupported adb flag.

    The pinned AndroidWorld setup always passes ``--bypass-low-target-sdk-block``.
    Some otherwise valid Android emulator images reject that option.  Keep the
    official call as the first attempt and retry only that precise compatibility
    failure through AndroidWorld's own adb request helper.
    """

    adb_utils = getattr(setup_module, "adb_utils", None)
    original = getattr(adb_utils, "install_apk", None)
    issue_generic_request = getattr(adb_utils, "issue_generic_request", None)
    if not callable(original) or not callable(issue_generic_request):
        return None

    def install_apk(apk_location: str, raw_env: Any) -> Any:
        try:
            return original(apk_location, raw_env)
        except Exception as exc:
            error_parts = [str(exc)]
            for attribute in ("stdout", "stderr", "output"):
                value = getattr(exc, attribute, None)
                if isinstance(value, bytes):
                    error_parts.append(value.decode("utf-8", errors="replace"))
                elif value is not None:
                    error_parts.append(str(value))
            if "Unknown option --bypass-low-target-sdk-block" not in "\n".join(
                error_parts
            ):
                raise
            response = issue_generic_request(
                ["install", apk_location],
                raw_env,
                timeout_sec=DEFAULT_ANDROIDWORLD_APK_INSTALL_TIMEOUT_SEC,
            )
            status = getattr(response, "status", None)
            ok_status = getattr(
                getattr(getattr(setup_module, "adb_pb2", None), "AdbResponse", None),
                "Status",
                None,
            )
            ok_value = getattr(ok_status, "OK", 1)
            if status != ok_value:
                raise
            return response

    adb_utils.install_apk = install_apk
    return original


def _patch_androidworld_media_scanner_broadcast_compat(adb_utils: Any) -> Any | None:
    """Bound the legacy media-scan broadcast used by Retro Music tasks.

    On the current API-35 image the ``MEDIA_SCANNER_SCAN_FILE`` broadcast can
    wait indefinitely even though the scan has already been submitted.  The
    scan must still be sent: skipping it leaves Retro Music with an empty
    library after task initialization.  Force a short request timeout and
    preserve every other AndroidWorld intent unchanged.
    """

    original = getattr(adb_utils, "send_android_intent", None)
    if not callable(original):
        return None

    def send_android_intent(*args: Any, **kwargs: Any) -> Any:
        command = kwargs.get("command")
        action = kwargs.get("action")
        if command is None and args:
            command = args[0]
        if action is None and len(args) > 1:
            action = args[1]
        if (
            str(command or "").strip().lower() == "broadcast"
            and str(action or "").strip()
            == "android.intent.action.MEDIA_SCANNER_SCAN_FILE"
        ):
            kwargs["timeout_sec"] = min(int(kwargs.get("timeout_sec", 10)), 2)
            try:
                return original(*args, **kwargs)
            except Exception:
                # The scan request may have been accepted before the shell
                # command timed out.  Retro Music can consume the resulting
                # MediaStore rows once its UI opens.
                return None
        return original(*args, **kwargs)

    adb_utils.send_android_intent = send_android_intent
    return original


def _patch_androidworld_adb_output_sanitizer(
    controller: Any,
) -> tuple[tuple[type[Any], Any], ...]:
    """Remove gRPC fork diagnostics injected into AndroidWorld ADB stdout."""

    patches: list[tuple[type[Any], Any]] = []
    targets = (controller, getattr(controller, "_original_env", None))
    for target in targets:
        if target is None:
            continue
        controller_type = type(target)
        if any(patched_type is controller_type for patched_type, _ in patches):
            continue
        original = getattr(controller_type, "execute_adb_call", None)
        if not callable(original):
            if target is controller:
                raise RuntimeError("androidworld_execute_adb_call_unavailable")
            continue

        def execute_adb_call(
            instance: Any,
            *args: Any,
            _original: Any = original,
            **kwargs: Any,
        ) -> Any:
            response = _original(instance, *args, **kwargs)
            # Pull/push responses carry binary file contents in a separate
            # protobuf field.  Never line-sanitize those requests: treating a
            # SQLite pull as UTF-8 can silently corrupt the database and make
            # the official validator report a missing table.
            request = args[0] if args else kwargs.get("request")
            try:
                request_kind = request.WhichOneof("command")
            except (AttributeError, ValueError):
                request_kind = None
            if request_kind in {"pull", "push"}:
                return response
            generic = getattr(response, "generic", None)
            output = getattr(generic, "output", None)
            if isinstance(output, (bytes, str)):
                binary = isinstance(output, bytes)
                text = output.decode("utf-8", errors="replace") if binary else output
                clean_lines = [
                    line
                    for line in text.splitlines(keepends=True)
                    if not (
                        ("fork_posix.cc:" in line or "ev_poll_posix.cc:" in line)
                        and (
                            "Other threads are currently calling into gRPC" in line
                            or "FD from fork parent still in poll list" in line
                        )
                    )
                ]
                clean = "".join(clean_lines)
                generic.output = clean.encode("utf-8") if binary else clean
            return response

        controller_type.execute_adb_call = execute_adb_call
        patches.append((controller_type, original))
    return tuple(patches)


def _patch_androidworld_app_launch(adb_utils: Any) -> Any:
    """Restart mapped apps before their official AndroidWorld launch."""
def _patch_androidworld_current_activity(adb_utils: Any) -> Any:
    """Recover a canonical activity when Android's stack output is malformed.

    Some API-29 emulator images intermittently return a package-only value from
    ``am stack list`` even though ``dumpsys activity activities`` already
    contains the resumed ``package/class`` component. AndroidWorld's official
    validators require the latter shape and otherwise crash before producing a
    validator result. Keep the official validator and recover only the
    transport value at this integration boundary.
    """

    original = getattr(adb_utils, "get_current_activity", None)
    if not callable(original):
        raise RuntimeError("androidworld_get_current_activity_unavailable")

    def get_current_activity(
        controller: Any,
        timeout_sec: float | None = None,
    ) -> tuple[Any, Any]:
        activity, response = original(controller, timeout_sec=timeout_sec)
        if isinstance(activity, str) and activity.count("/") == 1:
            return activity, response
        logger.warning(
            "AndroidWorld returned malformed current activity %r; "
            "recovering from dumpsys activity activities.",
            activity,
        )
        fallback = adb_utils.issue_generic_request(
            ["shell", "dumpsys", "activity", "activities"],
            controller,
            timeout_sec=timeout_sec,
        )
        output = getattr(getattr(fallback, "generic", None), "output", b"")
        text = (
            output.decode("utf-8", errors="replace")
            if isinstance(output, bytes)
            else str(output or "")
        )
        pattern = re.compile(
            r"(?:mResumedActivity|topResumedActivity|ResumedActivity|mFocusedApp):"
            r".*?\s([A-Za-z0-9_.]+/[A-Za-z0-9_.$]+)"
        )
        match = pattern.search(text)
        if match:
            return match.group(1), response
        return activity, response

    adb_utils.get_current_activity = get_current_activity
    return original



    original = getattr(adb_utils, "launch_app", None)
    if not callable(original):
        raise RuntimeError("androidworld_launch_app_unavailable")

    def launch_camera_capture_intent(controller: Any) -> Any:
        """Open the installed Camera2 app through its public capture intent.

        Some official API 29 tablet images ship Camera2 without the legacy
        ``CameraLauncher`` component used by AndroidWorld's static registry.
        The public ``IMAGE_CAPTURE`` intent resolves to that same installed
        Camera2 package (usually ``CaptureActivity``), so this is an
        environment compatibility fallback rather than a replacement app.
        """

        issue_generic_request = getattr(adb_utils, "issue_generic_request", None)
        check_ok = getattr(adb_utils, "check_ok", None)
        if not callable(issue_generic_request) or not callable(check_ok):
            raise RuntimeError("androidworld_camera_capture_intent_unavailable")
        response = issue_generic_request(
            ["shell", "am", "start", "-a", "android.media.action.IMAGE_CAPTURE"],
            controller,
        )
        check_ok(response, "Failed to launch the AndroidWorld Camera2 capture intent.")
        return response

    def launch_app(app_name: str, controller: Any) -> Any:
        if adb_utils.get_adb_activity(app_name) is not None:
            adb_utils.close_app(app_name, controller)
        try:
            return original(app_name, controller)
        except Exception:
            if str(app_name or "").strip().casefold() not in {"camera", "camera2"}:
                raise
            logger.warning(
                "AndroidWorld Camera2 component launch failed; retrying through "
                "the public IMAGE_CAPTURE intent",
                exc_info=True,
            )
            return launch_camera_capture_intent(controller)

    adb_utils.launch_app = launch_app
    return original


def _patch_androidworld_chcon_compat(setup_module: Any) -> Any | None:
    """Treat an unsupported external-filesystem ``chcon`` as non-fatal.

    AndroidWorld's OsmAnd setup copies its map into the app's external-files
    directory and then changes its SELinux context.  Some managed emulator
    images expose that directory through a transport endpoint where ``chcon``
    is unsupported, even though the copied file remains usable by the app.
    Preserve every other ADB failure and only normalize this exact setup
    compatibility response to AndroidWorld's normal OK status.
    """

    adb_utils = getattr(setup_module, "adb_utils", None)
    original = getattr(adb_utils, "issue_generic_request", None)
    if not callable(original):
        return None

    def is_unsupported_chcon(command: Any, error_text: str) -> bool:
        if isinstance(command, str):
            command_parts = tuple(command.split())
        elif isinstance(command, (list, tuple)):
            command_parts = tuple(str(part) for part in command)
        else:
            command_parts = ()
        return (
            command_parts[:2] == ("shell", "chcon")
            and "Operation not supported on transport endpoint" in error_text
        )

    def ok_response() -> Any:
        response_type = getattr(
            getattr(setup_module, "adb_pb2", None), "AdbResponse", None
        )
        if response_type is None:
            raise RuntimeError("AndroidWorld AdbResponse type is unavailable")
        response = response_type()
        response.status = getattr(response_type.Status, "OK", 1)
        return response

    def issue_generic_request(*call_args: Any, **call_kwargs: Any) -> Any:
        command = call_args[0] if call_args else call_kwargs.get("args")
        try:
            response = original(*call_args, **call_kwargs)
        except Exception as exc:
            if not is_unsupported_chcon(command, str(exc)):
                raise
            logging.warning(
                "AndroidWorld setup skipped unsupported external-filesystem "
                "chcon; the copied file remains available to the app"
            )
            return ok_response()
        generic = getattr(response, "generic", None)
        output = getattr(generic, "output", b"")
        if isinstance(output, bytes):
            output_text = output.decode(errors="replace")
        else:
            output_text = str(output or "")
        if is_unsupported_chcon(command, output_text):
            if hasattr(response, "CopyFrom"):
                try:
                    normalized = type(response)()
                except TypeError:
                    normalized = None
                if normalized is not None:
                    normalized.CopyFrom(response)
                    normalized.status = getattr(
                        getattr(type(response), "Status", None), "OK", 1
                    )
                    response = normalized
                else:
                    response.status = getattr(
                        getattr(type(response), "Status", None), "OK", 1
                    )
            else:
                response.status = 1
            logging.warning(
                "AndroidWorld setup skipped unsupported external-filesystem "
                "chcon; the copied file remains available to the app"
            )
        return response

    adb_utils.issue_generic_request = issue_generic_request
    return original


def _bounded_androidworld_adb_file_transfer_timeout(
    timeout_sec: float | None,
    *,
    default_timeout_sec: float,
) -> float:
    if timeout_sec is None or float(timeout_sec) <= 0:
        return float(default_timeout_sec)
    return float(timeout_sec)


def _androidworld_adb_file_transfer_timeout_sec() -> float:
    raw_value = str(
        os.environ.get(
            "OMNIFLOW_ANDROIDWORLD_ADB_FILE_TRANSFER_TIMEOUT_SEC",
            DEFAULT_ANDROIDWORLD_ADB_FILE_TRANSFER_TIMEOUT_SEC,
        )
    ).strip()
    try:
        timeout_sec = float(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            "OMNIFLOW_ANDROIDWORLD_ADB_FILE_TRANSFER_TIMEOUT_SEC must be positive"
        ) from exc
    if not math.isfinite(timeout_sec) or timeout_sec <= 0:
        raise RuntimeError(
            "OMNIFLOW_ANDROIDWORLD_ADB_FILE_TRANSFER_TIMEOUT_SEC must be positive"
        )
    return timeout_sec


def _androidworld_setup_timeout_sec() -> float:
    raw_value = str(
        os.environ.get(
            "OMNIFLOW_ANDROIDWORLD_SETUP_TIMEOUT_SEC",
            DEFAULT_ANDROIDWORLD_SETUP_TIMEOUT_SEC,
        )
    ).strip()
    try:
        timeout_sec = float(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            "OMNIFLOW_ANDROIDWORLD_SETUP_TIMEOUT_SEC must be positive"
        ) from exc
    if not math.isfinite(timeout_sec) or timeout_sec <= 0:
        raise RuntimeError(
            "OMNIFLOW_ANDROIDWORLD_SETUP_TIMEOUT_SEC must be positive"
        )
    return timeout_sec


def _prepare_androidworld_episode_apps(
    env: Any,
    *,
    setup_module: Any,
    setup_apps: Sequence[Any],
    save_snapshots: bool = False,
) -> None:
    for app in setup_apps:
        app_name = str(getattr(app, "app_name", "") or "").strip()
        package_name_getter = getattr(app, "package_name", None)
        package_name = (
            str(package_name_getter() or "").strip()
            if callable(package_name_getter)
            else ""
        )
        if not app_name or not package_name:
            continue
        if app_name == "markor":
            device_constants = importlib.import_module(
                "android_world.env.device_constants"
            )
            response = setup_module.adb_utils.issue_generic_request(
                ["shell", "mkdir", "-p", device_constants.MARKOR_DATA],
                env.controller,
            )
            setup_module.adb_utils.check_ok(
                response,
                "Failed to prepare Markor data directory.",
            )
        actuation = importlib.import_module("android_world.env.actuation")
        setup_module.adb_utils.launch_app(app_name, env.controller)
        try:
            time.sleep(2.0)
            elements = env.controller.get_ui_elements()
            if app_name == "osmand":
                tool_controller = importlib.import_module(
                    "android_world.env.tools"
                ).AndroidToolController(env.controller)
                labels = {
                    _normalize_androidworld_setup_label(
                        str(value or "").strip()
                    )
                    for element in elements
                    for value in (
                        getattr(element, "text", None),
                        getattr(element, "content_description", None),
                    )
                    if str(value or "").strip()
                }
                if "SKIP DOWNLOAD" in labels:
                    tool_controller.click_element("SKIP DOWNLOAD")
                # OsmAnd creates the map-marker schema asynchronously after
                # onboarding.  Saving a snapshot before that schema exists
                # makes OsmAndMarker fail in initialize_task, before the first
                # observation can be captured.
                time.sleep(7.0)
                file_utils = importlib.import_module(
                    "android_world.utils.file_utils"
                )
                if not file_utils.check_file_exists(
                    "/data/data/net.osmand/databases/map_markers_db",
                    env.controller,
                ):
                    logger.warning(
                        "AndroidWorld OsmAnd setup has not created the "
                        "map_markers database yet; continuing because "
                        "non-marker tasks do not require it"
                    )
            if app_name == "contacts":
                chooser_clicks = _app_chooser_clicks(
                    elements,
                    app_label="Contacts",
                )
                if chooser_clicks:
                    tool_controller = importlib.import_module(
                        "android_world.env.tools"
                    ).AndroidToolController(env.controller)
                    for label in chooser_clicks:
                        tool_controller.click_element(label)
                    logger.info(
                        "AndroidWorld episode setup resolved the Contacts chooser"
                    )
                    elements = env.controller.get_ui_elements()
            permission_dialog = any(
                str(getattr(element, "package_name", "") or "").endswith(
                    ".permissioncontroller"
                )
                for element in elements
            )
            if permission_dialog:
                actuation.find_and_click_element_by_resource_id(
                    ANDROID_PERMISSION_DENY_RESOURCE_IDS,
                    env.controller,
                    timeout_sec=10.0,
                )
                deadline = time.monotonic() + 10.0
                while True:
                    elements = env.controller.get_ui_elements()
                    packages = {
                        str(getattr(element, "package_name", "") or "").strip()
                        for element in elements
                    }
                    permission_dialog = any(
                        package.endswith(".permissioncontroller")
                        for package in packages
                    )
                    if not permission_dialog and package_name in packages:
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            "AndroidWorld app setup permission dialog did not settle: "
                            f"app={app_name}:expected={package_name}:"
                            f"observed={','.join(sorted(packages)) or '<none>'}"
                        )
                    time.sleep(0.25)
        finally:
            setup_module.adb_utils.close_app(app_name, env.controller)
        if save_snapshots:
            setup_module.app_snapshot.save_snapshot(app_name, env.controller)


def _prepare_official_harness_episode(env: Any, *, selected_agent: str) -> None:
    if not str(selected_agent or "").startswith("official:"):
        return
    from android_world.env import adb_utils

    adb_utils.press_home_button(env.controller)


def _patch_androidworld_directory_clear(file_utils: Any, adb_utils: Any) -> Any:
    """Make AndroidWorld directory clearing idempotent on noisy ADB hosts."""

    original = file_utils.clear_directory

    def clear_directory(directory_path: str, controller: Any) -> None:
        response = adb_utils.issue_generic_request(
            ["shell", "rm", "-rf", f"{directory_path}/*"],
            controller,
        )
        adb_utils.check_ok(
            response,
            f"Failed to clear directory {directory_path}.",
        )

    file_utils.clear_directory = clear_directory
    return original


def _prepare_androidworld_snapshot_restore(
    env: Any,
    setup_apps: Sequence[Any],
) -> None:
    from android_world.env import adb_utils

    for app in setup_apps:
        package_name = str(app.package_name() or "").strip()
        if not package_name:
            raise RuntimeError("AndroidWorld setup app package name missing")
        app_data_path = f"/data/data/{package_name}"
        for arguments, message in (
            (
                ["shell", "mkdir", "-p", app_data_path],
                f"Failed to prepare app data directory for {package_name}.",
            ),
            (
                [
                    "shell",
                    "touch",
                    f"{app_data_path}/omniflow_snapshot_restore_placeholder",
                ],
                f"Failed to prepare snapshot restore placeholder for {package_name}.",
            ),
        ):
            adb_utils.check_ok(
                adb_utils.issue_generic_request(arguments, env.controller),
                message,
            )


def _reset_androidworld_file_picker_state(
    env: Any,
    *,
    setup_module: Any,
    setup_apps: Sequence[Any],
) -> None:
    if not any(
        str(getattr(app, "app_name", "") or "").strip().casefold() == "chrome"
        for app in setup_apps
    ):
        return
    setup_module.adb_utils.clear_app_data(
        "com.google.android.documentsui",
        env.controller,
    )


def _wait_for_androidworld_a11y(env: Any, *, attempts: int = 6) -> None:
    refresh_env = getattr(env, "refresh_env", None)
    if callable(refresh_env):
        refresh_env()
    controller = getattr(env, "controller", None)
    restart_forwarder = getattr(
        controller, "restart_accessibility_forwarder", None
    )
    forwarder_restarted = False
    last_error: RuntimeError | None = None
    for attempt in range(max(1, int(attempts))):
        try:
            state = env.get_state(wait_to_stabilize=False)
            forest = getattr(state, "forest", None)
            if forest is None or (
                isinstance(forest, (dict, list, tuple, set)) and not forest
            ):
                raise RuntimeError("AndroidWorld accessibility forest is empty")
            return
        except RuntimeError as error:
            last_error = error
            if not forwarder_restarted and callable(restart_forwarder):
                try:
                    restart_forwarder()
                except RuntimeError as restart_error:
                    last_error = restart_error
                else:
                    forwarder_restarted = True
            if attempt + 1 < attempts:
                time.sleep(1.0)
    raise RuntimeError("AndroidWorld accessibility forest not ready") from last_error


def _is_pickleable(value: object) -> bool:
    try:
        pickle.dumps(value)
    except Exception:
        return False
    return True


def _sanitize_for_checkpoint(value: object) -> object:
    if _is_pickleable(value):
        return value
    if isinstance(value, dict):
        return {key: _sanitize_for_checkpoint(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_checkpoint(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _sanitize_for_checkpoint(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return _sanitize_for_checkpoint(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return {
                key: _sanitize_for_checkpoint(item)
                for key, item in dict(value.__dict__).items()
                if not key.startswith("_")
            }
        except Exception:
            pass
    return repr(value)


class _SanitizingCheckpointer:
    def __init__(self, delegate: object) -> None:
        self._delegate = delegate
        self.directory = getattr(delegate, "directory", "")

    def save_episodes(
        self, task_episodes: list[dict[str, object]], task_name: str
    ) -> None:
        sanitized = [_sanitize_for_checkpoint(episode) for episode in task_episodes]
        self._delegate.save_episodes(sanitized, task_name)

    def load(self, fields: list[str] | None = None):
        return self._delegate.load(fields=fields)


def _build_official_androidworld_agent(
    *,
    env: Any,
    official_agent_name: str,
    model_name: str | None = None,
) -> Any:
    """Build one upstream AndroidWorld agent for direct benchmark execution.

    Args:
        env: AndroidWorld environment already created by `env_launcher`.
        official_agent_name: Upstream AndroidWorld agent name. Empty value falls
            back to the same default as upstream `run.py`.

    Returns:
        One upstream AndroidWorld agent instance bound to the current env.
    """

    resolved_name = str(official_agent_name or "").strip() or "t3a_gpt4"
    if resolved_name != "t3a_gpt4":
        raise ValueError(f"Unknown AndroidWorld official agent: {resolved_name}")
    llm = _OpenAICompatibleMultimodalWrapper(model_name=model_name)
    from android_world.agents import t3a

    agent = t3a.T3A(env, llm)
    agent._omniflow_llm_usage_tracker = llm
    agent.name = resolved_name
    return agent




def _read_raw_replay_run_log(path_text: str) -> dict[str, Any]:
    path = Path(str(path_text or "").strip()).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"raw replay run log not found: {path}")
    decoded = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(decoded, dict):
        raise ValueError(f"raw replay run log must be a JSON object: {path}")
    return decoded


def _raw_replay_step_actions(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Project the accepted RunLog schema to recorded actions."""

    def replay_action(action: dict[str, Any]) -> dict[str, Any]:
        tool = str(action["tool"])
        params = dict(action.get("args") or {})
        projected = {"type": tool, "params": params}
        if (
            tool in {"click", "long_press", "input_text", "swipe"}
            and any(
                key in params
                for key in ("x", "y", "x1", "y1", "x2", "y2")
            )
        ):
            projected["coordinate_space"] = "canonical_0_1000"
        return projected

    run_log = import_run_log(data)
    actions: list[dict[str, Any]] = []
    for step in run_log["steps"]:
        if step["result"]["success"] is not True:
            continue
        action_type = str(step["action"].get("action_type") or "")
        if action_type in {"status", "unknown"}:
            continue
        if action_type == "answer":
            actions.append(
                {
                    "type": "answer",
                    "params": {
                        "text": str(step["action"].get("text") or ""),
                    },
                }
            )
            continue
        if action_type in {"scroll", "swipe"}:
            actions.append(
                {
                    "type": action_type,
                    "params": {
                        "direction": str(step["action"].get("direction") or ""),
                    },
                }
            )
            continue
        actions.extend(
            replay_action(action)
            for action in project_androidworld_step_actions(step)
        )
    return actions


def _raw_replay_source_size(data: dict[str, Any]) -> tuple[int, int] | None:
    """Read the source display from the accepted RunLog observation."""

    run_log = import_run_log(data)
    for step in run_log["steps"]:
        observation = step["observation"]
        pixels = observation.get("screenshot")
        if pixels is None:
            pixels = observation.get("pixels")
        if isinstance(pixels, dict):
            return int(pixels["width"]), int(pixels["height"])
        auxiliaries = observation.get("auxiliaries")
        display = auxiliaries.get("display") if isinstance(auxiliaries, dict) else None
        if isinstance(display, dict):
            width = _coerce_positive_int(display.get("width"))
            height = _coerce_positive_int(display.get("height"))
            if width and height:
                return width, height
    return None


def _fixed_replay_parameter_leaves(
    value: Any,
    *,
    path: str = "$",
) -> list[tuple[str, Any]]:
    leaves: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            leaves.extend(
                _fixed_replay_parameter_leaves(
                    item,
                    path=f"{path}.{key}",
                )
            )
        return leaves
    if isinstance(value, list):
        for index, item in enumerate(value):
            leaves.extend(
                _fixed_replay_parameter_leaves(
                    item,
                    path=f"{path}[{index}]",
                )
            )
        return leaves
    if value is None or isinstance(value, bool):
        return leaves
    leaves.append((path, value))
    return leaves


def _fixed_replay_goal_value_boundary(
    text: str,
    *,
    start: int,
    value: str,
) -> bool:
    end = start + len(value)
    if value[0].isalnum() and start > 0 and text[start - 1].isalnum():
        return False
    if value[-1].isalnum() and end < len(text) and text[end].isalnum():
        return False
    return True


def _fixed_replay_goal_parameter_bindings(
    run_log_data: dict[str, Any],
    *,
    target_goal: str,
) -> dict[str, Any]:
    run_log = import_run_log(run_log_data)
    source_goal = str(run_log.get("goal") or "")
    target_goal_text = str(target_goal or "")
    task_parameters = run_log.get("task_parameters")
    task_parameters = task_parameters if isinstance(task_parameters, dict) else {}
    candidate_paths: dict[str, list[str]] = {}
    for path, value in _fixed_replay_parameter_leaves(task_parameters):
        final_key = path.rsplit(".", maxsplit=1)[-1]
        if final_key in {"seed", "browser_task_seed"}:
            continue
        source_value = str(value)
        if not source_value or source_value not in source_goal:
            continue
        candidate_paths.setdefault(source_value, []).append(path)

    occurrences: list[tuple[int, int, str]] = []
    cursor = 0
    candidates = sorted(candidate_paths, key=lambda item: (-len(item), item))
    while cursor < len(source_goal):
        matches = [
            value
            for value in candidates
            if source_goal.startswith(value, cursor)
            and _fixed_replay_goal_value_boundary(
                source_goal,
                start=cursor,
                value=value,
            )
        ]
        if not matches:
            cursor += 1
            continue
        selected = matches[0]
        occurrences.append((cursor, cursor + len(selected), selected))
        cursor += len(selected)

    report: dict[str, Any] = {
        "source_goal": source_goal,
        "target_goal": target_goal_text,
        "status": "no_goal_parameters",
        "bindings": [],
    }
    if source_goal == target_goal_text:
        report["status"] = "identical_goal"
        return report
    if not occurrences:
        return report

    pattern_parts: list[str] = []
    source_cursor = 0
    for index, (start, end, _source_value) in enumerate(occurrences):
        pattern_parts.append(re.escape(source_goal[source_cursor:start]))
        pattern_parts.append(f"(?P<value_{index}>.*?)")
        source_cursor = end
    pattern_parts.append(re.escape(source_goal[source_cursor:]))
    match = re.fullmatch("".join(pattern_parts), target_goal_text, flags=re.DOTALL)
    if match is None:
        report["status"] = "target_goal_template_mismatch"
        return report

    captured_by_source: dict[str, set[str]] = {}
    for index, (_start, _end, source_value) in enumerate(occurrences):
        captured_by_source.setdefault(source_value, set()).add(
            str(match.group(f"value_{index}"))
        )
    bindings: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for source_value, captured_values in captured_by_source.items():
        if len(captured_values) != 1:
            conflicts.append(
                {
                    "source_parameter_paths": sorted(candidate_paths[source_value]),
                    "source_value": source_value,
                    "target_values": sorted(captured_values),
                }
            )
            continue
        target_value = next(iter(captured_values))
        bindings.append(
            {
                "source_parameter_paths": sorted(candidate_paths[source_value]),
                "source_value": source_value,
                "target_value": target_value,
                "changed": source_value != target_value,
            }
        )
    report["bindings"] = bindings
    if conflicts:
        report["status"] = "goal_parameter_capture_conflict"
        report["conflicts"] = conflicts
    else:
        report["status"] = "matched_goal_template"
    return report


def _fixed_replay_bind_action_parameters(
    source_action: dict[str, Any],
    bindings: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bound_action = copy.deepcopy(source_action)
    changed: list[dict[str, Any]] = []
    replacements = [
        item
        for item in bindings
        if bool(item.get("changed")) and str(item.get("source_value") or "")
    ]
    replacements.sort(
        key=lambda item: (-len(str(item["source_value"])), str(item["source_value"]))
    )

    def bind_value(value: Any, *, path: str) -> Any:
        if isinstance(value, dict):
            return {
                key: bind_value(item, path=f"{path}.{key}")
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                bind_value(item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ]
        if not isinstance(value, str):
            return value
        updated = value
        for binding in replacements:
            source_value = str(binding["source_value"])
            target_value = str(binding.get("target_value") or "")
            match_kind = ""
            if updated == source_value:
                updated = target_value
                match_kind = "exact"
            elif len(source_value) >= 3 and source_value in updated:
                updated = updated.replace(source_value, target_value)
                match_kind = "substring"
            if match_kind:
                changed.append(
                    {
                        "action_path": path,
                        "source_parameter_paths": list(
                            binding.get("source_parameter_paths") or ()
                        ),
                        "source_value": source_value,
                        "target_value": target_value,
                        "match": match_kind,
                    }
                )
        return updated

    bound_action = bind_value(bound_action, path="$")
    assert isinstance(bound_action, dict)
    return bound_action, changed


def _canonical_execution_summary(canonical_run: dict[str, Any]) -> dict[str, Any]:
    diagnostics = canonical_run.get("diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    execution_summary = diagnostics.get("execution_summary")
    if isinstance(execution_summary, dict):
        return dict(execution_summary)
    metadata = canonical_run.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    execution_summary = metadata.get("execution_summary")
    if isinstance(execution_summary, dict):
        return dict(execution_summary)
    summary = canonical_run.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    execution_summary = summary.get("execution_summary")
    return dict(execution_summary) if isinstance(execution_summary, dict) else {}


def _runtime_execution_trace(runtime_result: Any) -> list[dict[str, Any]]:
    detail = getattr(runtime_result, "detail", None)
    trace = detail.get("trace") if isinstance(detail, dict) else None
    if not isinstance(trace, list):
        return []
    return [
        to_serializable(step)
        for step in trace
        if isinstance(step, dict)
    ]


def _coerce_positive_int(value: Any) -> int:
    try:
        number = int(round(float(value)))
    except Exception:
        return 0
    return number if number > 0 else 0


def _raw_replay_number(params: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key not in params:
            continue
        try:
            return float(params.get(key))
        except Exception:
            continue
    return None


def _raw_replay_scale_coord(
    value: Any,
    *,
    source_extent: int,
    target_extent: int,
) -> int | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    if source_extent > 0 and target_extent > 0:
        numeric = numeric * (float(target_extent) / float(source_extent))
    if target_extent > 0:
        numeric = min(max(numeric, 0.0), float(target_extent - 1))
    return int(round(numeric))


def _raw_replay_pair_from_value(
    value: Any,
    *,
    x_keys: tuple[str, ...],
    y_keys: tuple[str, ...],
) -> tuple[float | None, float | None]:
    if isinstance(value, dict):
        return _raw_replay_number(value, *x_keys), _raw_replay_number(value, *y_keys)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except Exception:
            return None, None
    return None, None


def _raw_replay_relative_coord_pair(
    params: dict[str, Any],
    *,
    x_keys: tuple[str, ...],
    y_keys: tuple[str, ...],
    target_width: int,
    target_height: int,
) -> tuple[int | None, int | None]:
    target_evidence = (
        dict(params.get("target_evidence") or {})
        if isinstance(params.get("target_evidence"), dict)
        else {}
    )
    candidates = (
        params.get("canonical_0_1000"),
        params.get("relative_0_1000"),
        target_evidence.get("relative_0_1000"),
    )
    for candidate in candidates:
        raw_x, raw_y = _raw_replay_pair_from_value(
            candidate,
            x_keys=x_keys,
            y_keys=y_keys,
        )
        if raw_x is None or raw_y is None:
            continue
        x = _raw_replay_scale_coord(
            raw_x,
            source_extent=1000,
            target_extent=target_width,
        )
        y = _raw_replay_scale_coord(
            raw_y,
            source_extent=1000,
            target_extent=target_height,
        )
        return x, y
    return None, None


def _raw_replay_direction_from_points(
    x1: float | None,
    y1: float | None,
    x2: float | None,
    y2: float | None,
) -> str:
    if x1 is None or y1 is None or x2 is None or y2 is None:
        return ""
    dx = float(x2) - float(x1)
    dy = float(y2) - float(y1)
    if abs(dx) > abs(dy):
        return "left" if dx < 0 else "right"
    return "up" if dy < 0 else "down"


def _raw_replay_action_to_payload(
    source_action: dict[str, Any],
    *,
    source_size: tuple[int, int] | None,
    target_size: tuple[int, int],
    resolution: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    action_type = str(
        source_action.get("type")
        or source_action.get("tool")
        or source_action.get("action_type")
        or ""
    ).strip().lower()
    if not action_type:
        return None, "missing_action_type"
    params = (
        dict(source_action.get("params") or {})
        if isinstance(source_action.get("params"), dict)
        else {}
    )
    if not params and isinstance(source_action.get("args"), dict):
        params = dict(source_action.get("args") or {})
    if source_action.get("coordinate_space") == "canonical_0_1000":
        source_width, source_height = 1000, 1000
    else:
        source_width, source_height = source_size or (target_size[0], target_size[1])
    target_width, target_height = target_size

    def _record_resolution(parameter_source: str) -> None:
        if resolution is None:
            return
        resolution["parameter_source"] = parameter_source

    def _scaled_xy() -> tuple[int | None, int | None]:
        relative_x, relative_y = _raw_replay_relative_coord_pair(
            params,
            x_keys=("x", "center_x", "touch_x"),
            y_keys=("y", "center_y", "touch_y"),
            target_width=target_width,
            target_height=target_height,
        )
        if relative_x is not None and relative_y is not None:
            return relative_x, relative_y
        raw_x = _raw_replay_number(params, "x", "center_x", "touch_x")
        raw_y = _raw_replay_number(params, "y", "center_y", "touch_y")
        x = _raw_replay_scale_coord(
            raw_x,
            source_extent=source_width,
            target_extent=target_width,
        )
        y = _raw_replay_scale_coord(
            raw_y,
            source_extent=source_height,
            target_extent=target_height,
        )
        return x, y

    if action_type in {"click", "tap", "double_tap", "long_press", "longpress"}:
        x, y = _scaled_xy()
        _record_resolution("recorded_coordinate")
        if x is None or y is None:
            return None, "missing_coordinates"
        return {
            "action_type": "long_press"
            if action_type in {"long_press", "longpress"}
            else "double_tap"
            if action_type == "double_tap"
            else "click",
            "x": x,
            "y": y,
        }, None

    if action_type in {"input_text", "type_text", "set_text", "type"}:
        payload: dict[str, Any] = {
            "action_type": "input_text",
            "text": str(params.get("text") if params.get("text") is not None else ""),
        }
        if "clear_text" in params:
            payload["clear_text"] = bool(params.get("clear_text"))
        elif "clear" in params:
            payload["clear_text"] = bool(params.get("clear"))
        else:
            # Canonical input_text has no clear flag because the AndroidWorld
            # host always replaces the focused field's existing contents.
            payload["clear_text"] = True
        x, y = _scaled_xy()
        if x is not None and y is not None:
            payload["x"] = x
            payload["y"] = y
        return payload, None

    if action_type == "set_clipboard":
        return {
            "action_type": "raw_set_clipboard",
            "text": str(params.get("text") if params.get("text") is not None else ""),
        }, None

    if action_type in {"open_app", "launch_app", "openapp"}:
        package_name = str(
            params.get("package_name")
            or params.get("package")
            or params.get("packageName")
            or ""
        ).strip()
        app_name = str(
            params.get("app_name") or params.get("app") or ""
        ).strip()
        app_identifier = package_name or app_name
        if not app_identifier:
            return None, "missing_app_identifier"
        return {
            "action_type": "raw_open_app",
            "app_identifier": app_identifier,
        }, None

    if action_type in {"press_key", "key_event", "presskey"}:
        key = str(
            params.get("key") or params.get("key_name") or params.get("keycode") or ""
        ).strip().lower()
        if key in {"back", "navigate_back", "press_back"}:
            return {"action_type": "navigate_back"}, None
        if key in {"home", "navigate_home", "press_home"}:
            return {"action_type": "navigate_home"}, None
        if key in {"enter", "keyboard_enter"}:
            return {"action_type": "keyboard_enter"}, None
        return None, "unsupported_androidworld_key"

    if action_type in {"back", "press_back", "navigate_back"}:
        return {"action_type": "navigate_back"}, None
    if action_type in {"home", "press_home", "navigate_home"}:
        return {"action_type": "navigate_home"}, None
    if action_type in {"wait", "sleep"}:
        seconds = _raw_replay_number(params, "time_s", "seconds", "duration_s")
        if seconds is None:
            duration_ms = _raw_replay_number(params, "duration_ms", "time_ms")
            if duration_ms is not None:
                seconds = float(duration_ms) / 1000.0
        return {
            "action_type": "raw_wait",
            "seconds": max(0.0, float(seconds if seconds is not None else 1.0)),
        }, None
    if action_type in {"finished", "finish", "done", "status"}:
        content = str(params.get("content") or "").strip()
        if content:
            return {"action_type": "answer", "text": content}, None
        return {"action_type": "status", "goal_status": "complete"}, None

    if action_type == "answer":
        return {
            "action_type": "answer",
            "text": str(params.get("text") if params.get("text") is not None else ""),
        }, None

    if action_type in {"swipe", "scroll"}:
        x1 = _raw_replay_number(params, "x1", "start_x", "from_x", "touch_x")
        y1 = _raw_replay_number(params, "y1", "start_y", "from_y", "touch_y")
        x2 = _raw_replay_number(params, "x2", "end_x", "to_x", "lift_x")
        y2 = _raw_replay_number(params, "y2", "end_y", "to_y", "lift_y")
        relative_start = _raw_replay_relative_coord_pair(
            params,
            x_keys=("x1", "start_x", "from_x", "touch_x"),
            y_keys=("y1", "start_y", "from_y", "touch_y"),
            target_width=target_width,
            target_height=target_height,
        )
        relative_end = _raw_replay_relative_coord_pair(
            params,
            x_keys=("x2", "end_x", "to_x", "lift_x"),
            y_keys=("y2", "end_y", "to_y", "lift_y"),
            target_width=target_width,
            target_height=target_height,
        )
        if (
            relative_start[0] is not None
            and relative_start[1] is not None
            and relative_end[0] is not None
            and relative_end[1] is not None
        ):
            sx, sy = relative_start
            ex, ey = relative_end
            return {
                "action_type": "raw_swipe",
                "x1": sx,
                "y1": sy,
                "x2": ex,
                "y2": ey,
                "duration_ms": int(params.get("duration_ms") or 500),
                "input_source": str(params.get("input_source") or "")
                .strip()
                .lower(),
                "direction": _raw_replay_direction_from_points(sx, sy, ex, ey),
            }, None
        if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
            sx = _raw_replay_scale_coord(
                x1,
                source_extent=source_width,
                target_extent=target_width,
            )
            sy = _raw_replay_scale_coord(
                y1,
                source_extent=source_height,
                target_extent=target_height,
            )
            ex = _raw_replay_scale_coord(
                x2,
                source_extent=source_width,
                target_extent=target_width,
            )
            ey = _raw_replay_scale_coord(
                y2,
                source_extent=source_height,
                target_extent=target_height,
            )
            if sx is not None and sy is not None and ex is not None and ey is not None:
                return {
                    "action_type": "raw_swipe",
                    "x1": sx,
                    "y1": sy,
                    "x2": ex,
                    "y2": ey,
                    "duration_ms": int(params.get("duration_ms") or 500),
                    "input_source": str(params.get("input_source") or "")
                    .strip()
                    .lower(),
                    "direction": _raw_replay_direction_from_points(x1, y1, x2, y2),
                }, None
        direction = str(params.get("direction") or "").strip().lower()
        if direction in {"up", "down", "left", "right"}:
            return {
                "action_type": "scroll" if action_type == "scroll" else "swipe",
                "direction": direction,
            }, None
        return None, "missing_swipe_coordinates_or_direction"

    return None, f"unsupported_action_type:{action_type}"


def _sanitize_raw_replay_source_action(action: dict[str, Any]) -> dict[str, Any]:
    params = (
        dict(action.get("params") or {})
        if isinstance(action.get("params"), dict)
        else {}
    )
    for key in (
        "source_context",
        "target_description",
        "target_evidence",
        "xml_ref",
        "index",
    ):
        params.pop(key, None)
    return {"type": str(action.get("type") or action.get("tool") or ""), "params": params}


def _raw_replay_action_wait_seconds(
    source_action: dict[str, Any],
    payload: dict[str, Any],
) -> float:
    if payload.get("action_type") in {"raw_wait", "status"}:
        return 0.0
    params = (
        dict(source_action.get("params") or {})
        if isinstance(source_action.get("params"), dict)
        else {}
    )
    if not params and isinstance(source_action.get("args"), dict):
        params = dict(source_action.get("args") or {})
    seconds = _raw_replay_number(
        params,
        "wait_after_s",
        "post_action_wait_s",
        "post_wait_s",
    )
    if seconds is None:
        seconds = DEFAULT_RAW_REPLAY_ACTION_WAIT_SECONDS
    return max(0.0, float(seconds))


def _execute_raw_replay_host_swipe(
    agent: Any,
    payload: dict[str, Any],
    *,
    target_size: tuple[int, int],
) -> None:
    """Execute a replay swipe through the checked AndroidWorld host path."""

    target_width, target_height = target_size
    args = {
        "x1": float(payload["x1"]) / float(target_width) * 1000.0,
        "y1": float(payload["y1"]) / float(target_height) * 1000.0,
        "x2": float(payload["x2"]) / float(target_width) * 1000.0,
        "y2": float(payload["y2"]) / float(target_height) * 1000.0,
        "duration_ms": int(payload.get("duration_ms") or 500),
    }
    input_source = str(payload.get("input_source") or "").strip().lower()
    if input_source:
        args["input_source"] = input_source
    host_result = agent.host.act({"tool": "swipe", "args": args})
    if getattr(host_result, "success", False) is not True:
        raise RuntimeError(
            str(getattr(host_result, "error", "") or "raw replay swipe failed")
        )


def _raw_replay_observation_record(
    observation: Any,
    *,
    fallback_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    xml = str(getattr(observation, "xml", "") or "")
    package_name = str(getattr(observation, "package_name", "") or "")
    activity_name = str(getattr(observation, "activity_name", "") or "")
    # Observation capture is a measurement sidecar for fixed replay.  An
    # empty transient page must not change the replay action sequence; keep an
    # explicit empty sample and let the action use the device target size.
    extra = dict(getattr(observation, "extra", {}) or {})
    androidworld_state = extra.get("androidworld_state")
    if androidworld_state is not None and not isinstance(androidworld_state, dict):
        raise RuntimeError("raw replay AndroidWorld state is not serializable")
    display = (
        dict(extra.get("display") or {})
        if isinstance(extra.get("display"), dict)
        else {}
    )
    width = _coerce_positive_int(display.get("width"))
    height = _coerce_positive_int(display.get("height"))
    if fallback_size:
        width = width or _coerce_positive_int(fallback_size[0])
        height = height or _coerce_positive_int(fallback_size[1])
    record = {
        "provider": str(extra.get("observe_backend") or "unknown"),
        "package_name": package_name,
        "activity_name": activity_name,
        "xml": xml,
        "xml_available": bool(xml),
        "checker_passed": True,
    }
    if isinstance(androidworld_state, dict):
        record["androidworld_state"] = to_serializable(androidworld_state)
    if width and height:
        record.update(
            width=width,
            height=height,
            screenshot={
                "width": width,
                "height": height,
                "original_width": width,
                "original_height": height,
            },
        )
    return record


def _launch_raw_replay_app(
    app_identifier: str,
    host: Any,
) -> None:
    identifier = str(app_identifier or "").strip()
    if not identifier:
        raise ValueError("raw_replay_open_app_identifier_required")
    result = host.act(
        {"tool": "open_app", "args": {"package_name": identifier}}
    )
    if getattr(result, "success", False) is not True:
        raise RuntimeError(
            str(getattr(result, "error", "") or "raw replay open_app failed")
        )


def _apply_fixed_replay(
    agent: Any,
    *,
    run_log_json_path: str,
    adb_path: str = "",
) -> Any:
    """Replay fixed source actions through recorded AndroidWorld parameters."""

    original_set_max_steps = getattr(agent, "set_max_steps", None)
    run_log_data = _read_raw_replay_run_log(run_log_json_path)
    source_actions = _raw_replay_step_actions(run_log_data)
    source_size = _raw_replay_source_size(run_log_data)
    state: dict[str, Any] = {"ran": False, "payload": None}
    replay_host = getattr(agent, "host", None)
    replay_observe = getattr(replay_host, "observe", None)
    capture_native_observations = str(
        os.environ.get("OMNIFLOW_RAW_REPLAY_CAPTURE_OBSERVATIONS") or ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    capture_observations = capture_native_observations
    if capture_observations and not callable(replay_observe):
        raise RuntimeError("fixed replay observation capture requires host.observe")

    def _forced_reset(go_home: bool = False) -> None:
        state["ran"] = False
        state["payload"] = None
        if go_home:
            agent.env.reset(go_home=True)

    def _forced_set_max_steps(step_budget: int) -> None:
        del step_budget
        if callable(original_set_max_steps):
            original_set_max_steps(max(1, len(source_actions) + 1))

    def _execute_payload(
        payload: dict[str, Any],
        *,
        target_size: tuple[int, int],
    ) -> None:
        from android_world.env import json_action

        if payload.get("action_type") == "raw_open_app":
            _launch_raw_replay_app(
                str(payload["app_identifier"]),
                agent.host,
            )
            return
        if payload.get("action_type") == "raw_wait":
            result = agent.host.act(
                {
                    "tool": "wait",
                    "args": {
                        "duration_ms": int(
                            round(float(payload.get("seconds") or 0.0) * 1000.0)
                        )
                    },
                }
            )
            if getattr(result, "success", False) is not True:
                raise RuntimeError(
                    str(getattr(result, "error", "") or "raw replay wait failed")
                )
            return
        if payload.get("action_type") == "raw_swipe":
            _execute_raw_replay_host_swipe(
                agent,
                payload,
                target_size=target_size,
            )
            return
        if payload.get("action_type") == "raw_set_clipboard":
            raise RuntimeError(
                "fixed_replay_private_action_not_androidworld_json_action:set_clipboard"
            )
        if payload.get("action_type") == "status":
            return
        if payload.get("action_type") in {"click", "long_press"}:
            target_width, target_height = target_size
            host_result = agent.host.act(
                {
                    "tool": str(payload["action_type"]),
                    "args": {
                        "x": float(payload["x"]) / float(target_width) * 1000.0,
                        "y": float(payload["y"]) / float(target_height) * 1000.0,
                        **(
                            {"duration_ms": int(payload.get("duration_ms") or 1000)}
                            if payload.get("action_type") == "long_press"
                            else {}
                        ),
                    },
                }
            )
            if getattr(host_result, "success", False) is not True:
                raise RuntimeError(
                    str(getattr(host_result, "error", "") or "raw replay action failed")
                )
            return
        action = json_action.JSONAction(**payload)
        agent.env.execute_action(action)

    def _forced_step(goal: str):
        goal_text = str(goal or "").strip()
        if bool(state.get("ran")):
            payload = dict(state.get("payload") or {})
            summary = payload.get("run_log_summary")
            summary = dict(summary or {}) if isinstance(summary, dict) else {}
            return make_agent_result(
                done=True,
                data={
                    "summary": "fixed replay already completed",
                    "run_id": payload.get("run_id"),
                    "step_index": 0,
                    "source": "recorded_coordinate_replay_v1",
                    "actions_executed": int(summary.get("actions_executed") or 0),
                    "fallback": False,
                    "error": None,
                    "done_reason": "fixed_replay_already_completed",
                },
            )

        started = perf_counter()
        # Replay coordinates and AndroidWorld accessibility bounds share the
        # logical application display.  The physical display can be rotated
        # or have a different size on tablets/folds (for example 1280x800
        # versus the portrait logical 720x1280), so using it first changes the
        # coordinate space before the first action is dispatched.
        target_size = (0, 0)
        for attribute in ("logical_screen_size", "device_screen_size"):
            candidate = tuple(getattr(agent.env, attribute, ()) or ())
            if (
                len(candidate) == 2
                and float(candidate[0]) > 0
                and float(candidate[1]) > 0
            ):
                target_size = candidate
                break
        if len(target_size) != 2 or not target_size[0] or not target_size[1]:
            target_size = source_size or (1080, 2400)
        step_results: list[dict[str, Any]] = []
        completed = True
        error_text: str | None = None
        actions_executed = 0
        recorded_coordinate_actions = 0
        direct_actions = 0
        parameter_bound_actions = 0
        parameter_bindings_applied = 0
        goal_parameter_binding = _fixed_replay_goal_parameter_bindings(
            run_log_data,
            target_goal=goal_text,
        )
        goal_bindings = list(goal_parameter_binding.get("bindings") or ())
        for index, original_source_action in enumerate(source_actions):
            source_action, action_parameter_bindings = (
                _fixed_replay_bind_action_parameters(
                    original_source_action,
                    goal_bindings,
                )
            )
            if action_parameter_bindings:
                parameter_bound_actions += 1
                parameter_bindings_applied += len(action_parameter_bindings)
            observation_record: dict[str, Any] | None = None
            if capture_observations:
                observation_record = _raw_replay_observation_record(
                    replay_observe(xml=True, screenshot=False, app_info=True),
                    fallback_size=(int(target_size[0]), int(target_size[1])),
                )
            action_target_size = (int(target_size[0]), int(target_size[1]))
            if observation_record is not None:
                observed_width = _coerce_positive_int(
                    observation_record.get("width")
                )
                observed_height = _coerce_positive_int(
                    observation_record.get("height")
                )
                if observed_width and observed_height:
                    action_target_size = (observed_width, observed_height)
            action_resolution: dict[str, Any] = {}
            payload, skip_reason = _raw_replay_action_to_payload(
                source_action,
                source_size=source_size,
                target_size=action_target_size,
                resolution=action_resolution,
            )
            parameter_source = str(
                action_resolution.get("parameter_source")
                or "direct_androidworld_action"
            )
            step_record: dict[str, Any] = {
                "index": index,
                "source_action": _sanitize_raw_replay_source_action(
                    original_source_action
                ),
                "androidworld_action": dict(payload or {}),
                "source_screen_size": list(source_size) if source_size else None,
                "target_screen_size": list(action_target_size),
                "completed": False,
                "skipped": False,
                "parameter_source": parameter_source,
            }
            if action_parameter_bindings:
                step_record["bound_source_action"] = (
                    _sanitize_raw_replay_source_action(source_action)
                )
                step_record["task_parameter_bindings"] = action_parameter_bindings
            if observation_record is not None:
                step_record["observation_before_act"] = observation_record
            if skip_reason:
                completed = False
                error_text = skip_reason
                step_record["error"] = skip_reason
                step_results.append(step_record)
                break
            try:
                assert payload is not None
                semantic_recovery = None
                if payload.get("action_type") in {"click", "long_press"}:
                    semantic_recovery = _raw_replay_visible_setup_recovery(
                        agent,
                        goal_text=goal_text,
                    )
                if semantic_recovery:
                    step_record["semantic_recovery"] = semantic_recovery
                    step_record["parameter_source"] = "semantic_visible_text"
                else:
                    _execute_payload(
                        payload,
                        target_size=action_target_size,
                    )
                actions_executed += 1
                if semantic_recovery:
                    direct_actions += 1
                elif parameter_source == "recorded_coordinate":
                    recorded_coordinate_actions += 1
                else:
                    direct_actions += 1
                step_record["completed"] = True
                wait_after_s = _raw_replay_action_wait_seconds(
                    source_action,
                    payload,
                )
                step_record["wait_after_s"] = wait_after_s
                if wait_after_s > 0:
                    time.sleep(wait_after_s)
                if payload.get("action_type") == "status":
                    break
            except Exception as exc:  # noqa: BLE001
                completed = False
                error_text = str(exc)
                step_record["error"] = error_text
                step_results.append(step_record)
                break
            step_results.append(step_record)

        final_observation_record: dict[str, Any] | None = None
        final_observation_error: str | None = None
        if capture_observations:
            try:
                final_observation_record = _raw_replay_observation_record(
                    replay_observe(xml=True, screenshot=False, app_info=True),
                    fallback_size=(int(target_size[0]), int(target_size[1])),
                )
            except Exception as exc:  # noqa: BLE001
                final_observation_error = str(exc) or type(exc).__name__
        elapsed_ms = max(0.0, (perf_counter() - started) * 1000.0)
        run_id = (
            "fixed_replay_"
            + str(run_log_data.get("run_id") or Path(run_log_json_path).stem)
            + "_"
            + utc_now_iso()
            .replace("-", "")
            .replace(":", "")
            .replace(".", "")
            .replace("+", "")
        )
        done_reason = (
            "fixed_replay_completed"
            if completed
            else "fixed_replay_failed"
        )
        execution_summary = {
            "completed": bool(completed),
            "replay_completed": bool(completed),
            "execution_backend": "recorded_coordinate_replay_v1",
            "steps": int(actions_executed),
            "actions_executed": int(actions_executed),
            "recorded_coordinate_actions": int(recorded_coordinate_actions),
            "direct_actions": int(direct_actions),
            "parameter_bound_actions": int(parameter_bound_actions),
            "parameter_bindings_applied": int(parameter_bindings_applied),
            "model_calls": 0,
            "tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "elapsed_ms": elapsed_ms,
        }
        if error_text:
            execution_summary["failure_reason"] = error_text
        execution_trace = {
            "run_id": run_id,
            "goal": goal_text,
            "device_label": str(
                os.environ.get("OMNIFLOW_EVAL_DEVICE_LABEL") or ""
            ).strip(),
            "android_serial": str(os.environ.get("ANDROID_SERIAL") or "").strip(),
            "completed": bool(completed),
            "replay_completed": bool(completed),
            "done_reason": done_reason,
            "step_count": len(step_results),
            "agent_step_count": 1,
            "function_step_count": int(actions_executed),
            "actions_executed": int(actions_executed),
            "summary": {"execution_summary": execution_summary},
            "source_run_log": str(run_log_json_path),
            "source_screen_size": list(source_size) if source_size else None,
            "target_screen_size": [int(target_size[0]), int(target_size[1])],
            "goal_parameter_binding": goal_parameter_binding,
            "steps": [
                {
                    "step_index": 0,
                    "selection_source": "fixed_replay",
                    "execution_source": "recorded_coordinate_replay_v1",
                    "provider_detail": {
                        "raw_replay": {
                            "source_run_log": str(run_log_json_path),
                            "source_action_count": len(source_actions),
                            "actions_executed": int(actions_executed),
                            "recorded_coordinate_actions": int(
                                recorded_coordinate_actions
                            ),
                            "direct_actions": int(direct_actions),
                            "parameter_bound_actions": int(parameter_bound_actions),
                            "parameter_bindings_applied": int(
                                parameter_bindings_applied
                            ),
                            "goal_parameter_binding": goal_parameter_binding,
                            "source_screen_size": list(source_size)
                            if source_size
                            else None,
                            "target_screen_size": [
                                int(target_size[0]),
                                int(target_size[1]),
                            ],
                            "step_results": step_results,
                            "final_observation": final_observation_record,
                            "final_observation_error": final_observation_error,
                        }
                    },
                }
            ],
        }
        state["ran"] = True
        state["payload"] = {
            "run_id": run_id,
            "goal": goal_text,
            "run_log_summary": {
                "run_id": run_id,
                "completed": bool(completed),
                "replay_completed": bool(completed),
                "step_count": len(step_results),
                "agent_step_count": 1,
                "function_step_count": int(actions_executed),
                "actions_executed": int(actions_executed),
                "done_reason": done_reason,
                "duration_ms": elapsed_ms,
            },
            "execution_trace": execution_trace,
        }
        sidecar_path = str(os.environ.get("OMNIFLOW_RAW_REPLAY_RESULT_JSON") or "").strip()
        if sidecar_path:
            try:
                sidecar = Path(sidecar_path).expanduser().resolve()
                sidecar.parent.mkdir(parents=True, exist_ok=True)
                sidecar.write_text(
                    json.dumps(
                        to_serializable(
                            {
                                "completed": bool(completed),
                                "replay_completed": bool(completed),
                                "error": error_text,
                                "run_id": run_id,
                                "step_count": len(step_results),
                                "actions_executed": int(actions_executed),
                                "recorded_coordinate_actions": int(
                                    recorded_coordinate_actions
                                ),
                                "direct_actions": int(direct_actions),
                                "parameter_bound_actions": int(
                                    parameter_bound_actions
                                ),
                                "parameter_bindings_applied": int(
                                    parameter_bindings_applied
                                ),
                                "duration_ms": elapsed_ms,
                                "execution_trace": execution_trace,
                            }
                        ),
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    ),
                    encoding="utf-8",
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] failed to write raw replay sidecar {sidecar_path}: {exc}")
        return make_agent_result(
            done=True,
            data={
                "summary": error_text or "fixed replay executed",
                "run_id": run_id,
                "step_index": 0,
                "source": "recorded_coordinate_replay_v1",
                "actions_executed": int(actions_executed),
                "fallback": False,
                "error": error_text,
                "done_reason": done_reason,
            },
        )

    agent.reset = _forced_reset
    agent.set_max_steps = _forced_set_max_steps
    agent.step = _forced_step
    agent.name = MODE_OMNIFLOW
    return agent


def _build_launch_agent(
    *,
    agent: str,
    env: Any,
    store_path: str,
    adb_serial: str,
    adb_path: str = "",
    planner_provider: str = "",
    planner_model: str = "",
    model_endpoint_profile: str = "",
    planner_timeout_sec: float | None = None,
    max_steps: int = MAX_STEPS,
    raw_replay_run_log: str = "",
    appagent_root: str = "",
    appagent_workspace_root: str = "",
    appagent_docs_root: str = "",
    appagent_teacher_source: str = "",
    appagent_name: str = "",
    appagent_output_root: str = "",
    task_seed: int | None = None,
    evidence_root: str = "",
    performance_metrics: PerformanceMetrics | None = None,
    direct_function_id: str = "",
    direct_function_arguments: dict[str, Any] | None = None,
) -> Any:
    """Build the launcher-facing AndroidWorld agent for one explicit selector.

    Args:
        agent: Explicit launcher selector. `omniflow` keeps the shared
            cache-first OmniFlow adapter, and `official:<name>` runs one
            upstream AndroidWorld agent directly.
        env: AndroidWorld environment already created by `env_launcher`.
        store_path: Function Store path used only by the cache-first agent.
        adb_serial: Device serial forwarded to the canonical AndroidWorld host.
        raw_replay_run_log: Optional source runlog used only by the fixed replay
            baseline agent.

    Returns:
        One ready-to-run agent instance for `suite_utils.run(...)`.
    """

    resolved_agent = str(agent or MODE_OMNIFLOW).strip() or MODE_OMNIFLOW
    agent_instance = default_method_adapter_registry().build(
        MethodAdapterContext(
            selector=resolved_agent,
            env=env,
            store_path=store_path,
            adb_serial=adb_serial,
            adb_path=adb_path,
            planner_provider=planner_provider,
            planner_model=planner_model,
            model_endpoint_profile=model_endpoint_profile,
            planner_timeout_sec=planner_timeout_sec,
            max_steps=max_steps,
            raw_replay_run_log=raw_replay_run_log,
            appagent_root=appagent_root,
            appagent_workspace_root=appagent_workspace_root,
            appagent_docs_root=appagent_docs_root,
            appagent_teacher_source=appagent_teacher_source,
            appagent_name=appagent_name,
            appagent_output_root=appagent_output_root,
            task_seed=task_seed,
            evidence_root=evidence_root,
            performance_metrics=performance_metrics,
            direct_function_id=direct_function_id,
            direct_function_arguments=direct_function_arguments,
            build_omniflow_agent=build_agent,
            apply_fixed_replay=_apply_fixed_replay,
            build_official_agent=_build_official_androidworld_agent,
        )
    )
    return agent_instance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run OmniFlow E2E in one selected official environment. "
            "Use `--agent omniflow` for the shared OmniFlow cache-first path, "
            "or `--agent official:<name>` for one upstream AndroidWorld agent."
        )
    )
    parser.add_argument(
        "--environment",
        choices=("androidworld", "bmoca"),
        default="androidworld",
        help="Official task environment; this does not select an execution method.",
    )
    parser.add_argument("--android-world-root", default="")
    parser.add_argument("--bmoca-root", default="")
    parser.add_argument(
        "--environment-ids",
        default="100,101,102,103,104,105,106,107,108,109",
        help="Comma-separated B-MoCA environment IDs.",
    )
    parser.add_argument(
        "--android-sdk-root",
        default=os.environ.get("ANDROID_SDK_ROOT") or os.environ.get("ANDROID_HOME") or "",
    )
    parser.add_argument(
        "--android-avd-home",
        default=os.environ.get("ANDROID_AVD_HOME") or "",
    )
    parser.add_argument("--bmoca-avd-template-home", default="")
    parser.add_argument("--appium-port", type=int, default=4723)
    parser.add_argument("--appium-system-port", type=int, default=8200)
    parser.add_argument("--emulator-console-port", type=int, default=5554)
    parser.add_argument("--emulator-adb-port", type=int, default=5555)
    parser.add_argument("--emulator-grpc-port", type=int, default=8554)
    parser.add_argument("--show-emulator", action="store_true")
    parser.add_argument("--suite-family", default="android_world")
    parser.add_argument("--tasks", default="ContactsAddContact")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--task-random-seed", type=int, default=30)
    parser.add_argument("--n-task-combinations", type=int, default=1)
    parser.add_argument("--console-port", type=int, default=5554)
    parser.add_argument("--adb-path", default=_default_adb_path())
    parser.add_argument("--perform-emulator-setup", action="store_true")
    parser.add_argument("--fixed-task-seed", action="store_true")
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument(
        "--agent",
            default=MODE_OMNIFLOW,
        help=(
            "Agent selector. `omniflow` keeps the shared cache-first adapter; "
            "`appagent` runs pinned AppAgent deployment; "
            "`official:t3a_gpt4` keeps the paper baseline compatibility path."
        ),
    )
    parser.add_argument(
        "--output-path",
        default=str(
            (OMNIFLOW_ROOT / "data" / "androidworld").resolve()
        ),
    )
    parser.add_argument(
        "--collect-performance",
        action="store_true",
        help=(
            "Opt in to the standalone performance sidecar; it does not modify "
            "task results or batch summaries."
        ),
    )
    parser.add_argument(
        "--store-path",
        dest="store_path",
        default="",
        help="Function Store path.",
    )
    parser.add_argument(
        "--reuse-memory-path",
        default="",
        help="Oracle-selected task-local memory for a reuse-only method.",
    )
    parser.add_argument(
        "--raw-replay-run-log",
        default="",
        help="Source runlog used only by --agent fixed_replay.",
    )
    parser.add_argument("--appagent-root", default="")
    parser.add_argument("--appagent-workspace-root", default="")
    parser.add_argument("--appagent-docs-root", default="")
    parser.add_argument("--appagent-teacher-source", default="")
    parser.add_argument("--appagent-name", default="")
    parser.add_argument(
        "--task-params-json",
        default="",
        help=(
            "Optional JSON object used to instantiate the selected AndroidWorld "
            "task instead of random generated params. These values only initialize "
            "AndroidWorld and are never sent to the Function resolver."
        ),
    )
    parser.add_argument(
        "--source-action-hint-path",
        default="",
        help=(
            "Optional sanitized omniflow.t3a_semantic_hint.v1 artifact. With "
            "--agent official:t3a_gpt4, its semantic action outline is appended "
            "to the task goal; native AndroidWorld observation and action remain in use."
        ),
    )
    parser.add_argument(
        "--planner-provider",
        default=os.environ.get("OMNIFLOW_PLANNER_PROVIDER", ""),
        help=(
            "Optional online planner provider for --agent omniflow. Use `openai` "
            "for OpenAI-compatible endpoints, including Qwen via OPENAI_BASE_URL. "
            "If omitted, provider is auto-detected."
        ),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OMNIFLOW_PLANNER_MODEL", ""),
        help=(
            "Optional online planner model for --agent omniflow, for example "
            "`gpt-4o`, `qwen-vl-plus`, or `qwen-vl-max`."
        ),
    )
    parser.add_argument(
        "--model-endpoint-profile",
        choices=("auto", "openai", "llmthu"),
        default=FORMAL_MODEL_ENDPOINT_PROFILE,
        help="Credential and endpoint profile for the selected model.",
    )
    parser.add_argument(
        "--planner-timeout-sec",
        type=float,
        default=float(os.environ.get("OMNIFLOW_PLANNER_TIMEOUT_SEC") or 60.0),
        help="Per-call timeout in seconds for the online OmniFlow planner.",
    )
    parser.add_argument("--function-id", default="")
    parser.add_argument("--function-arguments-json", default="{}")
    return parser


def _run_bmoca_e2e(args: argparse.Namespace) -> int:
    """Use the normal OmniFlow runtime against the B-MoCA Host adapter."""

    from omniflow.transfer.runtime import load_transfer_state_catalog
    from src.experiment.observation_evidence import persist_target_run_evidence
    from src.integrations.bmoca import (
        BMocaEnvironmentConfig,
        discover_bmoca_episodes,
        open_bmoca_episode,
    )

    method = str(args.agent or "").strip()
    if method not in {"ours_replay", "skilldroid_replay"}:
        raise ValueError(f"bmoca_method_unsupported:{method}")
    selected_tasks = [item.strip() for item in str(args.tasks).split(",") if item.strip()]
    if len(selected_tasks) != 1:
        raise ValueError("bmoca_e2e_requires_exactly_one_task")
    store_path = Path(args.store_path).expanduser().resolve()
    reuse_memory_path = Path(args.reuse_memory_path).expanduser().resolve()
    transfer_state_path = store_path.with_name("transfer_states.json")
    if method == "ours_replay":
        if not store_path.is_file():
            raise FileNotFoundError(f"bmoca_function_store_missing:{store_path}")
        if not transfer_state_path.is_file():
            raise FileNotFoundError(
                f"bmoca_transfer_state_catalog_missing:{transfer_state_path}"
            )
    elif not reuse_memory_path.exists():
        raise FileNotFoundError(
            f"bmoca_reuse_memory_missing:{reuse_memory_path}"
        )
    environment_ids = tuple(
        item.strip() for item in str(args.environment_ids).split(",") if item.strip()
    )
    if len(environment_ids) != 1:
        raise ValueError("bmoca_single_result_requires_exactly_one_environment")
    ports = (
        int(args.appium_port),
        int(args.appium_system_port),
        int(args.emulator_console_port),
        int(args.emulator_adb_port),
        int(args.emulator_grpc_port),
    )
    if any(port <= 0 for port in ports) or len(set(ports)) != len(ports):
        raise ValueError("bmoca_isolated_ports_invalid")
    if int(args.emulator_adb_port) != int(args.emulator_console_port) + 1:
        raise ValueError("bmoca_emulator_port_pair_invalid")
    config = BMocaEnvironmentConfig.resolve(
        bmoca_root=args.bmoca_root,
        android_sdk_root=args.android_sdk_root,
        android_avd_home=args.android_avd_home,
        avd_template_home=args.bmoca_avd_template_home or None,
        run_headless=not bool(args.show_emulator),
        appium_port=int(args.appium_port),
        appium_system_port=int(args.appium_system_port),
        emulator_console_port=int(args.emulator_console_port),
        emulator_adb_port=int(args.emulator_adb_port),
        emulator_grpc_port=int(args.emulator_grpc_port),
    )
    for required_path, error_code in (
        (config.bmoca_root, "bmoca_root_missing"),
        (config.android_sdk_root, "bmoca_android_sdk_missing"),
        (config.android_avd_home, "bmoca_avd_home_missing"),
    ):
        if not required_path.is_dir():
            raise FileNotFoundError(f"{error_code}:{required_path}")
    episodes = discover_bmoca_episodes(
        config.bmoca_root,
        task_id=selected_tasks[0],
        environment_ids=environment_ids,
    )
    source_states = (
        load_transfer_state_catalog(transfer_state_path)
        if method == "ours_replay"
        else {}
    )
    output_dir = Path(args.output_path).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    episodes_path = output_dir / "episodes.jsonl"
    if summary_path.exists() or episodes_path.exists():
        raise FileExistsError(f"bmoca_attempt_already_exists:{output_dir}")
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        started = perf_counter()
        run_evidence: dict[str, Any] = {}
        observation_evidence: list[dict[str, Any]] = []
        emulator_serial = ""
        try:
            episode_root = output_dir / f"env_{episode.environment_id}"
            with open_bmoca_episode(
                episode,
                config=config,
                source_states=source_states,
                evidence_root=episode_root,
            ) as host:
                emulator_serial = host.emulator_serial
                result = None
                run_error: Exception | None = None
                try:
                    if method == "ours_replay":
                        from src.integrations.script_replay import run_script_replay

                        result = run_script_replay(
                            store_path=store_path,
                            host=host,
                        )
                    elif method == "skilldroid_replay":
                        from src.integrations.skilldroid_replay import (
                            run_droidrun_macro_replay,
                        )

                        result = run_droidrun_macro_replay(
                            memory_path=reuse_memory_path,
                            host=host,
                        )
                    else:
                        raise ValueError(
                            "bmoca_method_unsupported_without_provider_adapter:"
                            f"{method}"
                        )
                except Exception as error:  # noqa: BLE001 - seal failed episodes too
                    run_error = error
                run_log = host.seal_run_log(
                    task_name=episode.task_id,
                    goal=episode.goal,
                    diagnostics={
                        "method": method,
                        "emulator_serial": emulator_serial,
                        **(
                            {"runtime_error": str(run_error)}
                            if run_error is not None else {}
                        ),
                        **(
                            {
                                "reuse_replay_trace": list(
                                    result.detail.get("trace") or []
                                )
                            }
                            if result is not None
                            else {}
                        ),
                    },
                )
                observation_evidence = list(host.persist_observations() or [])
                if run_log is not None:
                    run_evidence = persist_target_run_evidence(
                        episode_root,
                        run_log=run_log,
                        captured_transfer_states=host.get_captured_transfer_states(),
                    )
                if run_error is not None:
                    raise run_error
                if result is None:
                    raise RuntimeError("bmoca_result_missing")
                row = {
                    "task_id": episode.task_id,
                    "environment_id": episode.environment_id,
                    "snapshot_id": episode.snapshot_id,
                    "avd_name": episode.avd_name,
                    "official_success": host.official_success,
                    "method": method,
                    "emulator_serial": emulator_serial,
                    "method_success": result.success,
                    "error": result.error,
                    **result.execution_summary,
                    "function_id": result.function_id,
                    "function_resolution": dict(
                        result.detail.get("function_resolution") or {}
                    ),
                    "checker_decisions": list(
                        result.detail.get("checker_decisions") or []
                    ),
                    "embedding_calls": int(
                        result.detail.get("embedding_calls") or 0
                    ),
                    "trace": list(result.detail.get("trace") or []),
                    "run_log_evidence": run_evidence,
                    "observation_evidence": observation_evidence,
                    "wall_seconds": round(perf_counter() - started, 6),
                }
                if row["fallback_steps"] != 0:
                    raise RuntimeError("bmoca_function_fallback_must_be_zero")
        except Exception as error:  # noqa: BLE001 - preserve one environment result
            row = {
                "task_id": episode.task_id,
                "environment_id": episode.environment_id,
                "snapshot_id": episode.snapshot_id,
                "avd_name": episode.avd_name,
                "official_success": False,
                "method": method,
                "emulator_serial": emulator_serial,
                "method_success": False,
                "error": str(error),
                "actions_executed": 0,
                "model_calls": 0,
                "embedding_calls": 0,
                "fallback_steps": 0,
                "run_log_evidence": run_evidence,
                "observation_evidence": observation_evidence,
                "wall_seconds": round(perf_counter() - started, 6),
            }
        rows.append(row)
        with episodes_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    official_successes = sum(row["official_success"] is True for row in rows)
    revision = subprocess.run(
        ["git", "-C", str(config.bmoca_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    summary = {
        "schema_version": "omniflow.environment-e2e.v1",
        "environment": "bmoca",
        "method": method,
        "task_id": selected_tasks[0],
        "bmoca_root": str(config.bmoca_root),
        "bmoca_revision": revision or None,
        "environment_ids": list(environment_ids),
        "episode_count": len(rows),
        "official_success_count": official_successes,
        "official_success_rate": official_successes / len(rows) if rows else 0.0,
        "actions_executed": sum(int(row.get("actions_executed") or 0) for row in rows),
        "model_calls": sum(int(row.get("model_calls") or 0) for row in rows),
        "embedding_calls": sum(
            int(row.get("embedding_calls") or 0) for row in rows
        ),
        "fallback_steps": sum(int(row.get("fallback_steps") or 0) for row in rows),
        "results": rows,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if official_successes == len(rows) else 1


def _decode_task_params(
    value: object,
    *,
    task_random_seed: int,
) -> dict[str, object]:
    text = str(value or "").strip()
    if not text:
        return {}
    decoded = json.loads(text)
    if not isinstance(decoded, dict):
        raise ValueError("--task-params-json must decode to a JSON object")
    if not decoded:
        return {}
    task_params = dict(decoded)
    task_params.setdefault("seed", int(task_random_seed))
    return task_params


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    selected_agent = str(args.agent or MODE_OMNIFLOW).strip() or MODE_OMNIFLOW
    if str(args.planner_provider or "").strip():
        os.environ["OMNIFLOW_PLANNER_PROVIDER"] = str(args.planner_provider).strip()
    if str(args.model or "").strip():
        os.environ["OMNIFLOW_PLANNER_MODEL"] = str(args.model).strip()
    if float(args.planner_timeout_sec or 0) > 0:
        os.environ["OMNIFLOW_PLANNER_TIMEOUT_SEC"] = str(
            float(args.planner_timeout_sec)
        )
    if args.environment == "bmoca":
        return _run_bmoca_e2e(args)
    android_world_root = Path(args.android_world_root).expanduser().resolve()
    run_py = android_world_root / "run.py"
    if not run_py.exists():
        raise FileNotFoundError(f"run.py not found under {android_world_root}")
    task_params = _decode_task_params(
        args.task_params_json,
        task_random_seed=int(args.task_random_seed),
    )
    direct_function_arguments = json.loads(args.function_arguments_json or "{}")
    if not isinstance(direct_function_arguments, dict):
        raise ValueError("--function-arguments-json must decode to an object")
    env = None
    adb_output_patches: tuple[tuple[type[Any], Any], ...] = ()
    original_launch_app: Any | None = None
    try:
        _add_android_world_path(android_world_root)
    original_current_activity: Any | None = None

        from android_world import checkpointer as checkpointer_lib
        from android_world import registry, suite_utils
        from android_world.env import adb_utils, env_launcher
        from android_world.env.setup_device import setup as aw_setup
        task_registry = registry.TaskRegistry()
        selected_task_names = [
            item.strip() for item in str(args.tasks).split(",") if item.strip()
        ]
        task_types = task_registry.get_registry(family=args.suite_family)
        suite = suite_utils.create_suite(
            task_types,
            n_task_combinations=int(args.n_task_combinations),
            seed=int(args.task_random_seed),
            tasks=selected_task_names,
            use_identical_params=bool(args.fixed_task_seed),
        )
        if aw_setup is None:
            raise RuntimeError("AndroidWorld setup_device module is required.")
        setup_app_list = _androidworld_setup_apps_for_suite(
            suite,
            get_app_mapping=aw_setup.get_app_mapping,
        )

        reuse_a11y_forwarder = False
        if not _is_oob_control_backend():
            reuse_a11y_forwarder = _ensure_androidworld_a11y_forwarder(
                console_port=int(args.console_port),
                adb_path=str(args.adb_path or ""),
                apk_path=str(os.environ.get("OMNIFLOW_ANDROIDWORLD_A11Y_APK") or ""),
            )
        logger.info(
            "AndroidWorld accessibility forwarder mode: %s",
            "oob-bypassed"
            if _is_oob_control_backend()
            else ("reuse-installed" if reuse_a11y_forwarder else "install-official"),
        )
        startup = prepare_androidworld_environment(
            env_launcher=env_launcher,
            setup_module=aw_setup,
            setup_apps=setup_app_list,
            console_port=int(args.console_port),
            adb_path=str(args.adb_path or ""),
            grpc_port=int(args.console_port) + 3000,
            install_a11y_forwarding_app=not reuse_a11y_forwarder,
            perform_emulator_setup=bool(args.perform_emulator_setup),
            wait_for_a11y=not _is_oob_control_backend(),
        )
        env = startup.env
        adb_output_patches = startup.adb_output_patches
        if _is_oob_control_backend():
            logger.info(
                "OOB control backend owns the complete observe/act lifecycle; "
                "native AndroidWorld A11y forwarding is disabled."
            )
        original_launch_app = _patch_androidworld_app_launch(adb_utils)
        if task_params:
        original_current_activity = _patch_androidworld_current_activity(adb_utils)
            if len(selected_task_names) != 1:
                raise ValueError("--task-params-json requires exactly one selected task")
            selected_task_name = selected_task_names[0]
            if selected_task_name not in task_types:
                raise ValueError(f"Task {selected_task_name!r} not found")
            task_type = task_types[selected_task_name]
            task_type.set_device_time(env)
            suite[selected_task_name] = [
                task_type(
                    _rehydrate_task_params(
                        params=dict(task_params),
                        task_type=task_type,
                    )
                )
            ]
        suite.suite_family = args.suite_family
        run_output_dir = Path(args.output_path).expanduser().resolve()
        run_output_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(
            "OMNIFLOW_RELOCATION_DIAGNOSTIC_DIR",
            str(run_output_dir / "relocation_failures"),
        )

        collect_performance = bool(getattr(args, "collect_performance", False)) or (
            str(os.environ.get("OMNIFLOW_COLLECT_PERFORMANCE") or "")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
        )
        performance_metrics = (
            PerformanceMetrics(
                adb_path=str(args.adb_path or ""),
                adb_serial=str(
                    os.environ.get("ANDROID_SERIAL")
                    or f"emulator-{int(args.console_port)}"
                ).strip(),
            )
            if collect_performance
            else None
        )
        experiment_environment = AndroidWorldExperimentEnvironment(
            env,
            AndroidWorldEnvironmentConfig(
                evidence_root=run_output_dir,
                performance_metrics=performance_metrics,
                adb_path=str(args.adb_path or ""),
                adb_serial=str(
                    os.environ.get("ANDROID_SERIAL")
                    or f"emulator-{int(args.console_port)}"
                ).strip(),
            ),
        )
        recording_session = experiment_environment.install_episode_recorder()
        agent = _build_launch_agent(
            agent=str(args.agent or MODE_OMNIFLOW),
            env=recording_session.env,
            store_path=str(args.store_path or ""),
            adb_serial=str(
                os.environ.get("ANDROID_SERIAL") or f"emulator-{int(args.console_port)}"
            ).strip(),
            adb_path=str(args.adb_path or ""),
            raw_replay_run_log=str(args.raw_replay_run_log or ""),
            planner_provider=str(args.planner_provider or ""),
            planner_model=str(args.model or ""),
            model_endpoint_profile=str(args.model_endpoint_profile or "auto"),
            planner_timeout_sec=float(args.planner_timeout_sec or 60.0),
            max_steps=max(1, int(args.max_steps)),
            appagent_root=str(args.appagent_root or ""),
            appagent_workspace_root=str(args.appagent_workspace_root or ""),
            appagent_docs_root=str(args.appagent_docs_root or ""),
            appagent_teacher_source=str(args.appagent_teacher_source or ""),
            appagent_name=str(args.appagent_name or ""),
            appagent_output_root=str(run_output_dir / "appagent_runtime"),
            task_seed=int(args.task_random_seed),
            evidence_root=str(run_output_dir),
            performance_metrics=performance_metrics,
            direct_function_id=str(args.function_id or "").strip(),
            direct_function_arguments=direct_function_arguments,
        )
        checkpoint_dir = (
            str(Path(args.checkpoint_dir).expanduser().resolve())
            if args.checkpoint_dir
            else checkpointer_lib.create_run_directory(args.output_path)
        )
        checkpointer = _SanitizingCheckpointer(
            checkpointer_lib.IncrementalCheckpointer(checkpoint_dir)
        )
        task_results_path = (
            run_output_dir / "task_results.jsonl"
        )
        for stale_output_path in (task_results_path,):
            try:
                stale_output_path.unlink()
            except FileNotFoundError:
                pass
        if len(selected_task_names) != 1 or int(args.n_task_combinations) != 1:
            raise ValueError(
                "AndroidWorld launcher requires exactly one task instance; "
                "the unified script owns task-major scheduling."
            )
        task = suite[selected_task_names[0]][0]
        task_name = str(getattr(task, "name", "") or "human_task")
        goal_text = str(getattr(task, "goal", "") or task_name)
        task_context: dict[str, Any] = {}
        update_task_context = getattr(agent, "update_current_task_context", None)
        if callable(update_task_context):
            task_context = dict(update_task_context(task) or {})
        set_current_task = getattr(agent, "set_current_task", None)
        if callable(set_current_task):
            set_current_task(task_name, goal_text, task_context)

        official_goal_hint_text = ""
        official_goal_hint_meta: dict[str, Any] | None = None
        if selected_agent.startswith("official:"):
            official_goal_hint_text, official_goal_hint_meta = (
                _load_official_agent_goal_hint(args.source_action_hint_path)
            )

        episode_recorder = recording_session.recorder
        episode_recorder_error = recording_session.error
        official_llm_usage_before = (
            _get_agent_llm_usage(agent)
            if selected_agent.startswith("official:")
            or selected_agent == "appagent"
            else {}
        )
        started_at = utc_now_iso()
        started_perf = perf_counter()
        if performance_metrics is not None:
            performance_metrics.start()
        result: dict[str, Any] | None = None
        mainline_name = selected_agent
        instrumented_agent = _ExperimentAgentAdapter(
            agent,
            recording_session=recording_session,
            goal_hint=official_goal_hint_text,
            max_steps=max(1, int(args.max_steps)),
            prepare_after_reset=lambda: _prepare_androidworld_episode_apps(
                env,
                setup_module=aw_setup,
                setup_apps=setup_app_list,
            ),
        )
        print(
            "Starting official AndroidWorld runner with "
            f"agent={mainline_name} and writing to {checkpoint_dir}"
        )
        try:
            _prepare_official_harness_episode(
                env,
                selected_agent=selected_agent,
            )
            file_utils = importlib.import_module("android_world.utils.file_utils")
            original_clear_directory = _patch_androidworld_directory_clear(
                file_utils,
                aw_setup.adb_utils,
            )
            try:
                results = suite_utils.run(
                    suite,
                    instrumented_agent,
                    checkpointer=checkpointer,
                    demo_mode=False,
                    return_full_episode_data=True,
                )
            finally:
                file_utils.clear_directory = original_clear_directory
            result = results[0] if results else None
        finally:
            try:
                canonical_run = None
                canonical_run_id = None
                observation_evidence: list[dict[str, Any]] | None = None
                task_success = False
                validator_reward = 0.0
                step_count = 0
                error_text = None
                if isinstance(result, dict):
                    try:
                        validator_reward = float(
                            result.get("is_successful") or 0.0
                        )
                        task_success = validator_reward > 0.5
                    except Exception:
                        validator_reward = 0.0
                        task_success = False
                    try:
                        step_count = int(result.get("episode_length") or 0)
                    except Exception:
                        step_count = 0
                    error_text = (
                        str(
                            result.get("exception_info")
                            or result.get("error")
                            or ""
                        ).strip()
                        or None
                    )
                official_validator_used = (
                    _result_has_official_validator_conclusion(result)
                )
                if (
                    episode_recorder is not None
                    and episode_recorder.episode_started
                ):
                    runtime_result = getattr(
                        getattr(agent, "host", None),
                        "state",
                        {},
                    )
                    runtime_result = (
                        runtime_result.get("last_result")
                        if isinstance(runtime_result, dict)
                        else None
                    )
                    execution_summary = getattr(
                        runtime_result,
                        "execution_summary",
                        None,
                    )
                    diagnostics = {
                        "method": selected_agent,
                        "official_validator_conclusion": bool(
                            official_validator_used
                        ),
                        "done_reason": error_text or (
                            "validator_success"
                            if task_success
                            else "validator_failure"
                        ),
                    }
                    runtime_function_id = str(
                        getattr(runtime_result, "function_id", "") or ""
                    ).strip()
                    if runtime_function_id:
                        diagnostics["function_id"] = runtime_function_id
                    if isinstance(execution_summary, dict):
                        diagnostics["execution_summary"] = dict(
                            execution_summary
                        )
                    execution_trace = _runtime_execution_trace(runtime_result)
                    if execution_trace:
                        diagnostics["execution_trace"] = execution_trace
                    runtime_detail = getattr(runtime_result, "detail", None)
                    function_resume = (
                        runtime_detail.get("function_resume")
                        if isinstance(runtime_detail, dict)
                        else None
                    )
                    if isinstance(function_resume, dict):
                        diagnostics["function_resume"] = dict(
                            function_resume
                        )
                    canonical_run = recording_session.seal_run_log(
                        task_name=task_name,
                        goal=goal_text,
                        task_parameters=(
                            dict(task_context.get("task_parameters") or {})
                            if isinstance(task_context, dict)
                            else {}
                        ),
                        seed=int(args.task_random_seed),
                        validator_official=official_validator_used,
                        validator_success=task_success,
                        validator_reward=validator_reward,
                        diagnostics=diagnostics,
                    )
                    if canonical_run is not None:
                        canonical_run_id = canonical_run["run_id"]
                if episode_recorder is not None:
                    observation_evidence = (
                        recording_session.persist_observations()
                    )
                    episode_recorder_error = recording_session.error
                if performance_metrics is not None:
                    performance_metrics.finish(
                        method_wall_sec=perf_counter() - started_perf,
                    )
                    write_performance_metrics(
                        performance_metrics.to_dict(),
                        run_output_dir / "performance_sidecar.json",
                    )
                if canonical_run is not None:
                    try:
                        canonical_function_step_count = int(
                            canonical_run.get("function_step_count")
                            or canonical_run.get("actions_executed")
                            or 0
                        )
                    except Exception:
                        canonical_function_step_count = 0
                    try:
                        canonical_step_count = int(
                            canonical_run.get("step_count")
                            or canonical_run.get("steps_count")
                            or len(list(canonical_run.get("steps") or []))
                        )
                    except Exception:
                        canonical_step_count = 0
                    canonical_step_count = max(
                        canonical_step_count,
                        canonical_function_step_count,
                    )
                    if canonical_step_count > 0:
                        step_count = canonical_step_count
                actions_executed = 0
                if canonical_run is not None:
                    actions_executed = _coerce_int(
                        canonical_run.get("actions_executed")
                        or canonical_run.get("actions_count")
                        or canonical_run.get("function_step_count")
                        or len(list(canonical_run.get("steps") or []))
                        or 0
                    )
                else:
                    actions_executed = _coerce_int(
                        _androidworld_episode_value(
                            result,
                            "actions_executed",
                        )
                    )
                    if selected_agent == "appagent":
                        actions_executed = _coerce_int(
                            getattr(
                                agent,
                                "actions_executed",
                                getattr(agent, "teacher_actions_consumed", 0),
                            )
                        )
                    if (
                        actions_executed <= 0
                        and selected_agent.startswith("official:")
                    ):
                        actions_executed = step_count
                if selected_agent == "appagent":
                    actions_executed = max(
                        actions_executed,
                        _coerce_int(getattr(agent, "actions_executed", 0)),
                    )
                model_calls = 0
                fallback_steps = 0
                total_tokens = 0
                prompt_tokens = 0
                completion_tokens = 0
                token_usage_state = "not_applicable"
                model_name: str | None = None
                model_base_url: str | None = None
                llm_usage: dict[str, Any] = {}
                if canonical_run is not None:
                    execution_summary = _canonical_execution_summary(
                        canonical_run
                    )
                    for key, target in (
                        ("model_calls", "model_calls"),
                        ("fallback_steps", "fallback_steps"),
                        ("tokens", "total_tokens"),
                        ("total_tokens", "total_tokens"),
                        ("prompt_tokens", "prompt_tokens"),
                        ("completion_tokens", "completion_tokens"),
                    ):
                        try:
                            value = int(execution_summary.get(key) or 0)
                        except Exception:
                            value = 0
                        if not value:
                            continue
                        if target == "model_calls":
                            model_calls = value
                        elif target == "fallback_steps":
                            fallback_steps = value
                        elif target == "total_tokens":
                            total_tokens = value
                        elif target == "prompt_tokens":
                            prompt_tokens = value
                        elif target == "completion_tokens":
                            completion_tokens = value
                    token_usage_state = str(
                        execution_summary.get("token_usage_status")
                        or token_usage_status(
                            {
                                "model_calls": model_calls,
                                "responses_with_usage": (
                                    model_calls if total_tokens > 0 else 0
                                ),
                            }
                        )
                    )
                    if model_calls > 0:
                        model_name = str(
                            args.model
                            or os.environ.get("OMNIFLOW_PLANNER_MODEL")
                            or os.environ.get("OPENAI_MODEL")
                            or ""
                        ).strip() or None
                        _, model_base_url = resolve_openai_compatible_config(
                            profile=str(args.model_endpoint_profile or "auto"),
                            base_url=_model_base_url_for_profile(
                                args.model_endpoint_profile
                            ),
                        )
                if selected_agent.startswith("official:") or selected_agent == "appagent":
                    official_agent_usage = _diff_llm_usage(
                        _get_agent_llm_usage(agent),
                        official_llm_usage_before,
                    )
                    llm_usage = official_agent_usage
                    model_calls = _coerce_int(llm_usage.get("model_calls"))
                    prompt_tokens = _coerce_int(llm_usage.get("prompt_tokens"))
                    completion_tokens = _coerce_int(
                        llm_usage.get("completion_tokens")
                    )
                    total_tokens = _coerce_int(llm_usage.get("total_tokens"))
                    token_usage_state = str(
                        llm_usage.get("token_usage_status")
                        or token_usage_status(llm_usage)
                    )
                    model_name = str(llm_usage.get("model") or "").strip() or None
                    model_base_url = (
                        str(llm_usage.get("base_url") or "").strip() or None
                    )
                artifact_kind = "none"
                artifact_ref = None
                if canonical_run_id:
                    artifact_kind = "canonical_run"
                    artifact_ref = canonical_run_id
                elif selected_agent.startswith("official:") or selected_agent == "appagent":
                    artifact_kind = "checkpoint"
                    artifact_ref = checkpoint_dir
                evaluation_task_params, evaluation_task_params_sha256 = (
                    _task_params_provenance(task)
                )
                task_result_record = {
                    "task_name": task_name,
                    "goal": goal_text,
                    "agent": selected_agent,
                    "task_random_seed": int(args.task_random_seed),
                    "task_params": evaluation_task_params,
                    "task_params_sha256": evaluation_task_params_sha256,
                    "state_backend": "androidworld",
                    "control_backend": (
                        "oob_control" if _is_oob_control_backend() else "androidworld"
                    ),
                    "action_backend": (
                        "oob_control" if _is_oob_control_backend() else "androidworld"
                    ),
                    "native_androidworld_agent_io": not _is_oob_control_backend(),
                    "success": task_success,
                    "official_validator_used": official_validator_used,
                    "androidworld_validator_result": {
                        "success": task_success,
                        "reward": validator_reward,
                        "error": error_text,
                        "uses_androidworld_official_validator": (
                            official_validator_used
                        ),
                        "validator": (
                            "androidworld_official"
                            if official_validator_used
                            else None
                        ),
                    },
                    "response_acceptance": {
                        "generic": task_success,
                        "androidworld": task_success,
                    },
                    "response_acceptance_detail": build_response_acceptance_detail(
                        success=task_success,
                        validator={
                            "success": task_success,
                            "reward": validator_reward,
                            "error": error_text,
                        },
                        error_message=error_text,
                    ),
                    "started_at": started_at,
                    "duration_ms": max(
                        0.0,
                        (perf_counter() - started_perf) * 1000.0,
                    ),
                    "step_count": step_count,
                    "actions_executed": actions_executed,
                    "model_calls": model_calls,
                    "fallback_steps": fallback_steps,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "token_usage_status": token_usage_state,
                    "model": model_name,
                    "model_base_url": model_base_url,
                    "artifact_kind": artifact_kind,
                    "artifact_ref": artifact_ref,
                    "error": error_text,
                }
                appagent_reuse_result: dict[str, Any] = {}
                if selected_agent == "appagent":
                    appagent_reuse_result = {
                        "decision_round_count": _coerce_int(
                            getattr(agent, "round_count", 0)
                        ),
                        "documentation_round_count": _coerce_int(
                            getattr(agent, "documentation_round_count", 0)
                        ),
                        "startup_action_count": _coerce_int(
                            getattr(agent, "_startup_action_count", 0)
                        ),
                    }
                task_result_record["reuse_metrics"] = reuse_metrics(
                    selected_agent,
                    actions_executed=actions_executed,
                    canonical_run=canonical_run,
                    appagent_result=appagent_reuse_result,
                    source_action_hint=official_goal_hint_meta,
                    uses_source_action_hints=bool(official_goal_hint_text),
                )
                if canonical_run is not None:
                    canonical_diagnostics = canonical_run.get("diagnostics")
                    canonical_diagnostics = (
                        canonical_diagnostics
                        if isinstance(canonical_diagnostics, dict)
                        else {}
                    )
                    function_id = str(
                        canonical_diagnostics.get("function_id") or ""
                    ).strip()
                    if function_id:
                        task_result_record["function_id"] = function_id
                    function_resume = canonical_diagnostics.get(
                        "function_resume"
                    )
                    if isinstance(function_resume, dict):
                        task_result_record["function_resume"] = to_serializable(
                            function_resume
                        )
                if llm_usage:
                    task_result_record["llm_usage"] = to_serializable(llm_usage)
                if official_goal_hint_meta is not None:
                    task_result_record["source_action_hint"] = to_serializable(
                        official_goal_hint_meta
                    )
                if task_context:
                    task_result_record["androidworld_task_context"] = (
                        to_serializable(task_context)
                    )
                if canonical_run is not None:
                    task_result_record["canonical_run"] = to_serializable(
                        canonical_run
                    )
                    captured_transfer_states = None
                    transfer_state_audit = None
                    if selected_agent == MODE_OMNIFLOW:
                        get_transfer_states = getattr(
                            agent,
                            "get_captured_transfer_states",
                            None,
                        )
                        if callable(get_transfer_states):
                            captured_transfer_states = get_transfer_states()
                            transfer_state_audit = transfer_state_coverage_audit(
                                canonical_run,
                                captured_transfer_states,
                            )
                    task_result_record.update(
                        persist_target_run_evidence(
                            run_output_dir,
                            run_log=canonical_run,
                            captured_transfer_states=captured_transfer_states,
                            transfer_state_audit=transfer_state_audit,
                        )
                    )
                    relocation_diagnostics = _extract_relocation_diagnostics(
                        canonical_run
                    )
                    if relocation_diagnostics:
                        task_result_record["relocation_diagnostic_count"] = len(
                            relocation_diagnostics
                        )
                        task_result_record["relocation_diagnostics"] = (
                            to_serializable(relocation_diagnostics)
                        )
                if observation_evidence is not None:
                    task_result_record["observation_count"] = len(
                        observation_evidence
                    )
                    task_result_record["observation_evidence"] = (
                        observation_evidence
                    )
                    if not observation_evidence:
                        task_result_record["observation_evidence_error"] = (
                            "no_observations_recorded"
                        )
                elif episode_recorder_error:
                    task_result_record["observation_evidence_error"] = (
                        episode_recorder_error
                    )
                task_results_path.parent.mkdir(parents=True, exist_ok=True)
                with open(task_results_path, "a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            to_serializable(task_result_record),
                            ensure_ascii=False,
                            default=str,
                        )
                    )
                    handle.write("\n")
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[warn] failed to aggregate canonical run log for {task_name}: {exc}"
                )
            finally:
                recording_session.close()

        print(
            "Finished official AndroidWorld runner "
            f"agent={mainline_name} on {args.suite_family} family. "
            f"Wrote to {checkpoint_dir}."
        )
        run_summary = _summarize_task_results(
            task_results_path=task_results_path,
            checkpoint_dir=str(checkpoint_dir),
            agent=mainline_name,
            tasks=selected_task_names,
        )
        if int(run_summary.get("official_validator_task_count") or 0) != len(
            selected_task_names
        ):
            print(
                "[error] AndroidWorld episode ended without complete official "
                "validator coverage.",
                flush=True,
            )
            return 1
        return 0
    finally:
        if original_launch_app is not None:
            adb_utils.launch_app = original_launch_app
        for controller_type, original_execute_adb_call in adb_output_patches:
            controller_type.execute_adb_call = original_execute_adb_call
        if original_current_activity is not None:
            adb_utils.get_current_activity = original_current_activity
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
