from __future__ import annotations

import argparse
import base64
import copy
import dataclasses
import datetime
import hashlib
import importlib
import io
import json
import logging
import os
from pathlib import Path
import pickle
import random
import re
import socket
import subprocess
import sys
import time
from time import perf_counter
import types
from typing import Any, Callable, Sequence
import urllib.error
import urllib.parse
import urllib.request

from omniflow.vlm.model_config import resolve_openai_compatible_config
from omniflow.vlm.usage import token_usage_status
from src.experiment.observation_evidence import (
    persist_target_run_evidence,
    transfer_state_coverage_audit,
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
SOURCE_RUNLOG_POOL_DIR = (
    OMNIFLOW_ROOT
    / "runtime"
    / "evals"
    / "androidworld_validator"
    / "offline_source_runlog_pool"
)
logger = logging.getLogger(__name__)
DEFAULT_RAW_REPLAY_ACTION_WAIT_SECONDS = 1.0
ANDROIDWORLD_A11Y_FORWARDER_PACKAGE = (
    "com.google.androidenv.accessibilityforwarder"
)
ANDROIDWORLD_A11Y_FORWARDER_SHA256 = (
    "97a56a544e44d79f9b3181fc7dbdd72cffa908efd3d53c82afad1773061a350a"
)


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read_env_text(name: str) -> str | None:
    value = str(os.environ.get(name) or "").strip()
    return value or None


def read_env_int(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        value = int(read_env_text(name) or default)
    except ValueError:
        value = default
    return max(minimum, value)


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


def _mobilegpt_runtime_integrity_error(value: Any) -> str | None:
    error = str(value or "").strip()
    if not error or "mobilegpt_step_budget_exhausted" in error:
        return None
    markers = (
        "ConnectionError:",
        "ConnectionRefusedError:",
        "TimeoutError:",
        "mobilegpt_androidworld_state_",
        "mobilegpt_app_ui_not_ready:",
        "mobilegpt_launch_response_invalid:",
        "mobilegpt_native_xml_invalid:",
        "mobilegpt_server_closed_connection",
    )
    return error if any(marker in error for marker in markers) else None


def _mobilegpt_runtime_integrity_exit_code(run_summary: dict[str, Any]) -> int:
    return int(int(run_summary.get("runtime_integrity_error_count") or 0) > 0)


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


def _render_official_semantic_hint_step(index: int, step: Any) -> str:
    if not isinstance(step, dict):
        return ""
    allowed_keys = {"action", "target", "app", "direction", "key"}
    if any(str(key) not in allowed_keys for key in step):
        return ""
    action = str(step.get("action") or "").strip().lower()
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", action):
        return ""
    target = _official_hint_text(step.get("target"))
    app = _official_hint_text(step.get("app"), max_len=80)
    direction = _official_hint_text(step.get("direction"), max_len=24)
    key = _official_hint_text(step.get("key"), max_len=24)
    prefix = f"{index}. "
    if action in {"open_app", "launch_app"}:
        return prefix + (f"Open app {app}." if app else "Open the relevant app.")
    if action in {"click", "tap", "long_press"}:
        verb = "Long press" if action == "long_press" else "Click"
        return prefix + (
            f"{verb} the UI target described as '{target}'."
            if target
            else f"{verb} the relevant visible UI target."
        )
    if action in {"input_text", "type_text", "set_text", "enter_text"}:
        return prefix + (
            f"Enter the value requested by the current task into '{target}'."
            if target
            else "Enter the value requested by the current task into the relevant text field."
        )
    if action in {"press_key", "key_event"}:
        return prefix + (f"Press key {key}." if key else "Press the relevant key.")
    if action in {"swipe", "scroll"}:
        if direction and target:
            return prefix + f"Scroll {direction} on '{target}'."
        if direction:
            return prefix + f"Scroll {direction}."
        return prefix + "Scroll as needed to reveal the next target."
    if action in {"wait", "sleep"}:
        return prefix + "Wait for the screen to update."
    return prefix + (
        f"Use action {action} on '{target}'." if target else f"Use action {action}."
    )


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
                "The following is an action sequence that you may use as a reference:",
                *rendered,
                "This sequence is reference guidance, not a replay command. Inspect every current screen and independently choose each next action for the current task. Never infer coordinates or reuse old input values.",
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
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        "omniflow.t3a_semantic_hint.v1"
    ):
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
    ):
        self._agent = agent
        self._recording_session = recording_session
        self._goal_hint = str(goal_hint or "").strip()
        self._max_steps = max(1, int(max_steps)) if max_steps is not None else None
        self._completed_steps = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)

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
        if callable(ensure_ready):
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
            or "GLM-5.1"
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
        import io

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


def _valid_reason_action_output(output: object) -> bool:
    text = str(output or "").strip()
    if not text:
        return False
    try:
        from android_world.agents import agent_utils, m3a_utils

        reason, action_text = m3a_utils.parse_reason_action_output(text)
        return bool(reason and action_text and agent_utils.extract_json(action_text))
    except Exception:
        return False


def _candidate_action(text: object) -> dict[str, Any] | None:
    match = re.search(r"Action:\s*(\{.*\})", str(text or "").strip(), flags=re.DOTALL)
    if not match:
        return None
    try:
        action = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return action if isinstance(action, dict) else None


def _last_prompt_action(text_prompt: str) -> dict[str, Any] | None:
    matches = re.findall(r"Action selected:\s*(\{[^\n]*\})", str(text_prompt or ""))
    for value in reversed(matches):
        try:
            action = json.loads(value)
        except json.JSONDecodeError:
            continue
        if isinstance(action, dict):
            return action
    return None


def _keyboard_visible_in_prompt(text_prompt: str) -> bool:
    prompt = str(text_prompt or "").lower()
    markers = (
        "switch input method",
        "emoji button",
        "symbol keyboard",
        "voice input",
        "gif keyboard",
    )
    return sum(marker in prompt for marker in markers) >= 2


def _keyboard_obstruction_guard_applies(text_prompt: str, proposed: object) -> bool:
    action = _candidate_action(proposed)
    last_action = _last_prompt_action(text_prompt)
    return bool(
        action
        and action.get("action_type") == "scroll"
        and last_action
        and last_action.get("action_type") == "input_text"
        and _keyboard_visible_in_prompt(text_prompt)
    )


class _ActionConsistencyLlmWrapper:
    """Candidate-only second-pass action review over the same multimodal prefix."""

    def __init__(self, delegate: Any, policy: dict[str, Any]) -> None:
        self.delegate = delegate
        self.policy = dict(policy)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def predict(self, text_prompt: str) -> tuple[str, bool | None, Any]:
        return self.predict_mm(text_prompt, [])

    def predict_mm(
        self,
        text_prompt: str,
        images: list[Any],
    ) -> tuple[str, bool | None, Any]:
        proposed, is_safe, raw_response = self.delegate.predict_mm(text_prompt, images)
        if "Your Answer:" not in str(text_prompt):
            return proposed, is_safe, raw_response
        mode = str(self.policy.get("mode") or "always")
        if mode == "keyboard_obstruction_guard":
            if not _keyboard_obstruction_guard_applies(text_prompt, proposed):
                return proposed, is_safe, raw_response
            return (
                "Reason: The software keyboard is still open after text entry and "
                "obscures the form, so dismiss it before viewport navigation.\n"
                'Action: {"action_type":"navigate_back"}',
                is_safe,
                {
                    "first_pass": raw_response,
                    "action_consistency_applied": True,
                    "action_consistency_mode": mode,
                },
            )
        instruction = str(self.policy.get("instruction") or "").strip()
        review_prompt = (
            f"{text_prompt}\n\n"
            "SkyMark action-consistency review\n"
            "The first-pass candidate action was:\n"
            f"{proposed}\n\n"
            f"{instruction}\n"
            "Return one final answer in exactly the original Reason/Action format. "
            "Do not discuss the review and do not emit multiple actions.\n\n"
            "Final Answer:\n"
        )
        reviewed, reviewed_safe, reviewed_raw = self.delegate.predict_mm(
            review_prompt,
            images,
        )
        if _valid_reason_action_output(reviewed):
            return reviewed, reviewed_safe, {
                "first_pass": raw_response,
                "review_pass": reviewed_raw,
                "action_consistency_applied": True,
            }
        return proposed, is_safe, {
            "first_pass": raw_response,
            "review_pass": reviewed_raw,
            "action_consistency_applied": False,
            "fallback_reason": "review_output_parse_failed",
        }


_LLM_USAGE_COUNTER_KEYS = (
    "model_calls",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "responses_with_usage",
    "responses_without_usage",
    "failed_calls",
)


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
    if str(result.get("exception_info") or result.get("error") or "").strip():
        return False
    reward = result.get("is_successful")
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


def _safe_slug(value: Any, max_len: int = 120) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "")).strip("_").lower()
    return (text[:max_len].strip("_") or "unknown")


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


def _write_jsonl_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(to_serializable(row), ensure_ascii=False, default=str) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _append_unique_source_pool_record(
    *,
    task_name: str,
    goal: str,
    params: dict[str, Any],
    task_random_seed: int,
    canonical_run: dict[str, Any],
    task_result_record: dict[str, Any],
) -> dict[str, Any]:
    run_id = str(canonical_run.get("run_id") or "").strip()
    task_slug = _safe_slug(task_name)
    canonical_name = f"{task_slug}_{run_id or _stable_hash(canonical_run)[:12]}.run_log.json"
    pool_dir = SOURCE_RUNLOG_POOL_DIR
    raw_rel = (
        Path("raw_source_artifacts")
        / "androidworld_launcher"
        / task_slug
        / canonical_name
    )
    local_abs = (pool_dir / raw_rel).resolve()
    local_abs.parent.mkdir(parents=True, exist_ok=True)
    local_abs.write_text(
        json.dumps(to_serializable(canonical_run), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    try:
        local_ref = str(local_abs.relative_to(OMNIFLOW_ROOT))
    except ValueError:
        local_ref = str(local_abs)

    record = {
        "schema_version": "androidworld_offline_source_runlog_pool.v1",
        "source_kind": "androidworld_validator_success_source_runlog",
        "source_run_log": str(local_abs),
        "canonical_run_log": str(local_abs),
        "function_path": None,
        "artifact_dir": None,
        "run_id": run_id or None,
        "task": task_name,
        "goal": goal,
        "params": to_serializable(params) if isinstance(params, dict) else {},
        "collect_seed": task_random_seed,
        "replay_seed": None,
        "task_random_seed": task_random_seed,
        "androidworld_success": True,
        "androidworld_reward": (
            (task_result_record.get("androidworld_validator_result") or {}).get("reward")
            if isinstance(task_result_record.get("androidworld_validator_result"), dict)
            else 1.0
        ),
        "step_count": _coerce_int(task_result_record.get("step_count")),
        "duration_ms": _coerce_float(task_result_record.get("duration_ms")),
        "action_signature_hash": _stable_hash(
            [
                {
                    "source": step.get("source"),
                    "actions": [
                        {
                            "type": action.get("type"),
                            "params": {
                                key: value
                                for key, value in dict(action.get("params") or {}).items()
                                if key != "source_context"
                            },
                        }
                        for action in list(step.get("actions") or [])
                        if isinstance(action, dict)
                    ],
                }
                for step in list(canonical_run.get("steps") or [])
                if isinstance(step, dict)
            ]
        ),
        "params_hash": _stable_hash(params if isinstance(params, dict) else {}),
        "accepted_first30": False,
        "accepted_case_id": None,
        "latest_official_success_source": True,
        "latest_result_refs": [],
        "path_was_excluded_by_default": False,
        "local_source_run_log": local_ref,
        "local_canonical_run_log": local_ref,
        "local_function_path": None,
        "local_artifact_dir": None,
    }

    pool_dir.mkdir(parents=True, exist_ok=True)
    by_task_dir = pool_dir / "by_task" / task_slug
    global_json = pool_dir / "all_success_source_runlogs.json"
    global_jsonl = pool_dir / "all_success_source_runlogs.jsonl"
    task_json = by_task_dir / "source_runlogs.json"
    task_jsonl = by_task_dir / "source_runlogs.jsonl"

    def _load_list(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []

    def _dedupe_key(item: dict[str, Any]) -> tuple[str, str, str]:
        return (
            str(item.get("run_id") or ""),
            str(item.get("local_canonical_run_log") or ""),
            str(item.get("action_signature_hash") or ""),
        )

    def _write_index(json_path: Path, jsonl_path: Path) -> None:
        rows = _load_list(json_path)
        keys = {_dedupe_key(row) for row in rows}
        key = _dedupe_key(record)
        if key not in keys:
            rows.append(record)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(to_serializable(rows), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _write_jsonl_rows(jsonl_path, rows)

    _write_index(global_json, global_jsonl)
    _write_index(task_json, task_jsonl)
    return record


def _write_task_results_summary(
    *,
    task_results_path: Path,
    output_dir: Path,
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

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(to_serializable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md_lines = [
        "# AndroidWorld Run Summary",
        "",
        f"- agent: `{agent}`",
        f"- task_results: `{task_results_path}`",
        f"- checkpoint_dir: `{checkpoint_dir}`",
        f"- official_validator_success: `{successful_tasks}/{official_validator_tasks}`",
        f"- official_validator_coverage: `{official_validator_tasks}/{total_tasks}`",
        f"- runtime integrity errors: `{len(runtime_integrity_errors)}`",
        f"- total duration: `{round(total_duration_ms / 1000.0, 3)}s`",
        f"- actions executed: `{total_actions}`",
        f"- single-step execution accuracy: `{summary['single_step_execution_accuracy']}`",
        f"- validator-weighted action accuracy: `{summary['validator_weighted_action_accuracy']}`",
        f"- tool_calls / tokens: `{total_model_calls}` / `{total_tokens}`",
        "",
        "| task | official_validator | sec | actions | step_acc | tool_calls | tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in per_task:
        md_lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("task_name") or ""),
                    "1" if item.get("official_validator_success") else "0",
                    str(round(_coerce_float(item.get("duration_ms")) / 1000.0, 3)),
                    str(item.get("actions_executed") or 0),
                    str(item.get("action_completed_rate") or 0),
                    str(item.get("model_calls") or 0),
                    str(item.get("total_tokens") or 0),
                ]
            )
            + " |"
        )
    (output_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(
        "[omniflow] summary: "
        f"official_validator={successful_tasks}/{official_validator_tasks} "
        f"coverage={official_validator_tasks}/{total_tasks} "
        f"duration={total_duration_ms / 1000.0:.1f}s "
        f"actions={total_actions} "
        f"step_acc={summary['single_step_execution_accuracy']} "
        f"tool_calls={total_model_calls} tokens={total_tokens} "
        f"summary={summary_path}"
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


def _assert_existing_emulator_ready(
    *,
    console_port: int,
    adb_path: str,
    grpc_port: int,
) -> None:
    """Fail fast when the target AndroidWorld emulator is not attachable.

    Args:
        console_port: Android emulator console port selected by the launcher.
            The matching adb serial must already exist as `emulator-<port>`.
        adb_path: Resolved adb binary path. Empty values fall back to `adb`
            from PATH so local shells keep working.
        grpc_port: Emulator gRPC port expected by AndroidWorld / AndroidEnv.
            The launcher only attaches to an existing endpoint on this port.

    Raises:
        RuntimeError: The target emulator is missing/offline or the expected
            emulator gRPC endpoint is not reachable.
    """

    normalized_console_port = int(console_port)
    normalized_grpc_port = int(grpc_port)
    serial = f"emulator-{normalized_console_port}"
    adb_bin = os.path.expanduser(str(adb_path or "").strip()) or "adb"
    retry_sec = float(os.environ.get("OMNIFLOW_ANDROIDWORLD_ADB_READY_RETRY_SEC", "20"))
    deadline = time.monotonic() + max(0.0, retry_sec)
    adb_result: subprocess.CompletedProcess[str] | None = None
    device_state = ""
    current_devices: list[str] = []
    while True:
        try:
            adb_result = subprocess.run(
                [adb_bin, "devices"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"adb binary not found for AndroidWorld launcher preflight: {adb_bin}"
            ) from exc

        device_state = ""
        current_devices = []
        if adb_result.returncode == 0:
            for raw_line in str(adb_result.stdout or "").splitlines()[1:]:
                parts = raw_line.strip().split()
                if len(parts) < 2:
                    continue
                current_devices.append(f"{parts[0]}={parts[1]}")
                if parts[0] == serial:
                    device_state = parts[1]
        if adb_result.returncode == 0 and device_state == "device":
            break

        retryable = (
            adb_result.returncode != 0
            or not current_devices
            or device_state in {"offline", "unauthorized"}
        )
        if not retryable or time.monotonic() >= deadline:
            break
        time.sleep(0.5)

    if adb_result is None or adb_result.returncode != 0:
        stderr_text = str(
            (adb_result.stderr if adb_result else "")
            or (adb_result.stdout if adb_result else "")
            or ""
        ).strip()
        returncode = adb_result.returncode if adb_result else "missing"
        raise RuntimeError(
            "AndroidWorld launcher failed before env attach because `adb devices` "
            f"returned {returncode}: {stderr_text or 'unknown adb error'}"
        )

    if device_state != "device":
        summarized_devices = ", ".join(current_devices) or "none"
        missing_reason = device_state or "missing"
        raise RuntimeError(
            "AndroidWorld launcher only attaches to an existing emulator. "
            f"Required emulator not ready: {serial} (state={missing_reason}). "
            f"Current adb devices: {summarized_devices}. "
            "Start the target emulator first, then rerun the launcher."
        )

    try:
        with socket.create_connection(
            ("127.0.0.1", normalized_grpc_port),
            timeout=1.0,
        ):
            pass
    except OSError as exc:
        raise RuntimeError(
            "AndroidWorld launcher only attaches to an existing emulator gRPC "
            f"endpoint. Expected {serial} to expose 127.0.0.1:{normalized_grpc_port}. "
            f"Launch the emulator with `-grpc {normalized_grpc_port}` before rerunning."
        ) from exc


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
    if (
        hashlib.sha256(path.read_bytes()).hexdigest()
        != ANDROIDWORLD_A11Y_FORWARDER_SHA256
    ):
        raise RuntimeError(f"AndroidWorld accessibility forwarder hash mismatch: {path}")
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


def _prepare_official_harness_episode(env: Any, *, selected_agent: str) -> None:
    if not str(selected_agent or "").startswith("official:"):
        return
    from android_world.env import adb_utils

    adb_utils.press_home_button(env.controller)


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


def _wait_for_androidworld_a11y(env: Any, *, attempts: int = 6) -> None:
    refresh_env = getattr(env, "refresh_env", None)
    if callable(refresh_env):
        refresh_env()
    last_error: RuntimeError | None = None
    for attempt in range(max(1, int(attempts))):
        try:
            state = env.get_state(wait_to_stabilize=False)
            if getattr(state, "forest", None) is None:
                raise RuntimeError("AndroidWorld state has no accessibility forest")
            return
        except RuntimeError as error:
            last_error = error
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
    if resolved_name not in {"t3a", "t3a_gpt4", "m3a"}:
        raise ValueError(f"Unknown AndroidWorld official agent: {resolved_name}")
    llm = _OpenAICompatibleMultimodalWrapper()
    if resolved_name == "m3a":
        from android_world.agents import m3a

        agent = m3a.M3A(env, llm)
    else:
        from android_world.agents import t3a

        agent = t3a.T3A(env, llm)
    agent._omniflow_llm_usage_tracker = llm
    agent.name = resolved_name
    return agent


def _official_parser_result(step: dict[str, Any]) -> dict[str, Any]:
    parsed = step.get("action_output_json")
    if parsed is not None:
        return {"status": "parsed", "action": to_serializable(parsed)}
    output = str(step.get("action_output") or "")
    if not output:
        return {"status": "missing_output", "action": None}
    try:
        from android_world.agents import agent_utils, m3a_utils

        reason, action_text = m3a_utils.parse_reason_action_output(output)
        if not reason or not action_text:
            return {"status": "parse_failed", "action": None}
        return {
            "status": "parsed",
            "reason": reason,
            "action": to_serializable(agent_utils.extract_json(action_text)),
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "parse_failed", "action": None, "error": str(exc)}


def _persist_official_step_captures(
    *,
    output_dir: Path,
    agent: Any,
    selected_agent: str,
    task_name: str,
    goal: str,
    task_params_sha256: str,
    version_id: str = "stock",
    candidate_proposal: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if selected_agent not in {"official:t3a", "official:t3a_gpt4", "official:m3a"}:
        return None
    history = list(getattr(agent, "history", None) or [])
    tracker = getattr(agent, "_omniflow_llm_usage_tracker", None)
    action_requests = [
        record
        for record in list(getattr(tracker, "request_records", None) or [])
        if record.get("kind") == "action"
    ]
    capture_root = output_dir / "skymark_stock_capture"
    image_root = capture_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    harness_id = "m3a" if selected_agent == "official:m3a" else "t3a"
    rows = []
    for step_index, step in enumerate(history[:7]):
        if not isinstance(step, dict):
            continue
        request_record = action_requests[step_index] if step_index < len(action_requests) else {}
        image_refs = []
        for image_index, payload in enumerate(list(request_record.get("image_payloads") or [])):
            image_sha256 = hashlib.sha256(payload).hexdigest()
            image_path = image_root / f"{image_sha256}.jpg"
            if not image_path.exists():
                image_path.write_bytes(payload)
            image_refs.append(
                {
                    "role": "raw_screenshot" if image_index == 0 else "som_screenshot",
                    "path": str(image_path),
                    "sha256": image_sha256,
                    "mime_type": "image/jpeg",
                    "exact_model_payload": True,
                }
            )
        rows.append(
            {
                "schema_version": "skymark.stock_androidworld_step_capture.v1",
                "request_id": f"{task_name}:{harness_id}:{version_id}:step-{step_index + 1}",
                "logical_test_id": f"{task_name}:{harness_id}:step-{step_index + 1}",
                "campaign_cell_id": f"{task_name}:{harness_id}:{version_id}",
                "task_id": task_name,
                "harness_id": harness_id,
                "version_id": version_id,
                "step_index": step_index + 1,
                "goal": goal,
                "task_params_sha256": task_params_sha256,
                "action_prompt": str(step.get("action_prompt") or request_record.get("prompt") or ""),
                "model_input_images": image_refs,
                "modality": "vision_text" if image_refs else "text_only",
                "raw_response": step.get("action_raw_response"),
                "action_output": step.get("action_output"),
                "parser_result": _official_parser_result(step),
                "request_timing_ms": request_record.get("duration_ms"),
                "prompt_tokens": _coerce_int(
                    (request_record.get("response_metadata") or {}).get("usage", {}).get("prompt_tokens")
                    if isinstance(request_record.get("response_metadata"), dict)
                    else 0
                ),
                "completion_tokens": _coerce_int(
                    (request_record.get("response_metadata") or {}).get("usage", {}).get("completion_tokens")
                    if isinstance(request_record.get("response_metadata"), dict)
                    else 0
                ),
                "reference_action_available_to_runtime": False,
                "source": "stock_androidworld_agent_step_capture",
                "candidate_proposal": to_serializable(candidate_proposal),
            }
        )
    capture_path = capture_root / "steps.json"
    capture_path.write_text(
        json.dumps({"schema_version": "skymark.stock_capture_bundle.v1", "steps": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "path": str(capture_path),
        "step_count": len(rows),
        "harness_id": harness_id,
        "version_id": version_id,
    }


def _apply_candidate_harness_proposal(agent: Any, path_text: str) -> dict[str, Any] | None:
    text = str(path_text or "").strip()
    if not text:
        return None
    path = Path(text).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "skymark.harness_revision.v1":
        raise ValueError("unsupported_candidate_harness_proposal")
    expected_harness = "m3a" if isinstance(getattr(agent, "history", None), list) and agent.__class__.__name__ == "M3A" else "t3a"
    if str(payload.get("harness_id") or "") != expected_harness:
        raise ValueError("candidate_harness_proposal_agent_mismatch")
    guidelines = []
    action_consistency_policy = None
    for patch in payload.get("patches") or []:
        if not isinstance(patch, dict):
            raise ValueError("candidate_harness_patch_invalid")
        if patch.get("seam") not in {
            "history_policy",
            "system_instruction",
            "completion_policy",
            "action_consistency_policy",
        }:
            raise ValueError(f"candidate_harness_patch_seam_not_runtime_safe:{patch.get('seam')}")
        if patch.get("seam") == "action_consistency_policy":
            if patch.get("operation") != "configure":
                raise ValueError("action_consistency_policy_requires_configure")
            value = patch.get("value")
            if not isinstance(value, dict) or str(value.get("mode") or "") not in {
                "always",
                "keyboard_obstruction_guard",
            }:
                raise ValueError("action_consistency_policy_invalid")
            if not str(value.get("instruction") or "").strip():
                raise ValueError("action_consistency_instruction_required")
            action_consistency_policy = dict(value)
            continue
        if patch.get("operation") != "append":
            raise ValueError("candidate_harness_runtime_text_seams_only_support_append")
        value = str(patch.get("value") or "").strip()
        if value:
            guidelines.append(value)
    if not guidelines and action_consistency_policy is None:
        raise ValueError("candidate_harness_patches_required")
    if guidelines:
        setter = getattr(agent, "set_task_guidelines", None)
        if not callable(setter):
            raise ValueError("candidate_harness_guideline_seam_unavailable")
        setter(guidelines)
    if action_consistency_policy is not None:
        delegate = getattr(agent, "llm", None)
        if delegate is None:
            raise ValueError("candidate_harness_llm_seam_unavailable")
        wrapped = _ActionConsistencyLlmWrapper(delegate, action_consistency_policy)
        agent.llm = wrapped
        agent._omniflow_llm_usage_tracker = wrapped
    return {
        "proposal_id": str(payload.get("proposal_id") or ""),
        "harness_version_id": str(payload.get("harness_version_id") or ""),
        "content_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "path": str(path),
        "guidelines": guidelines,
        "action_consistency_policy": to_serializable(action_consistency_policy),
    }


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
        if action_type in {"answer", "status", "unknown"}:
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
    if not xml and not package_name and not activity_name:
        raise RuntimeError("raw replay observe returned no usable state")
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
    from android_world.env import adb_utils

    identifier = str(app_identifier or "").strip()
    if not identifier:
        raise ValueError("raw_replay_open_app_identifier_required")
    env = host.env
    package_name = ""
    if "." in identifier:
        package_name = identifier
        matches = []
        for app_name in adb_utils.get_all_apps(env.controller):
            activity = adb_utils.get_adb_activity(app_name)
            if activity and adb_utils.extract_package_name(activity) == identifier:
                matches.append(app_name)
        if len(matches) != 1:
            raise RuntimeError(
                f"raw_replay_app_package_unresolved:{identifier}:{len(matches)}"
            )
        identifier = matches[0]
    else:
        activity = adb_utils.get_adb_activity(identifier)
        if activity:
            package_name = str(
                adb_utils.extract_package_name(activity) or ""
            ).strip()
    action_args = (
        {"package_name": package_name}
        if package_name
        else {"app_name": identifier}
    )
    result = host.act({"tool": "open_app", "args": action_args})
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
                    "source": "selector_then_scaled_coordinate_fallback_v2",
                    "actions_executed": int(summary.get("actions_executed") or 0),
                    "fallback": False,
                    "error": None,
                    "done_reason": "fixed_replay_already_completed",
                },
            )

        started = perf_counter()
        target_size = tuple(
            getattr(agent.env, "device_screen_size", (0, 0)) or (0, 0)
        )
        if len(target_size) != 2:
            target_size = (0, 0)
        if not target_size[0] or not target_size[1]:
            target_size = tuple(
                getattr(agent.env, "logical_screen_size", (0, 0)) or (0, 0)
            )
        if len(target_size) != 2 or not target_size[0] or not target_size[1]:
            target_size = source_size or (1080, 2400)
        step_results: list[dict[str, Any]] = []
        completed = True
        error_text: str | None = None
        actions_executed = 0
        selector_actions = 0
        scaled_coordinate_actions = 0
        selector_fallback_actions = 0
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
                _execute_payload(
                    payload,
                    target_size=action_target_size,
                )
                actions_executed += 1
                if parameter_source == "recorded_coordinate":
                    scaled_coordinate_actions += 1
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
            "execution_backend": "selector_then_scaled_coordinate_fallback_v2",
            "steps": int(actions_executed),
            "actions_executed": int(actions_executed),
            "selector_actions": int(selector_actions),
            "scaled_coordinate_actions": int(scaled_coordinate_actions),
            "selector_fallback_actions": int(selector_fallback_actions),
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
                    "execution_source": "selector_then_scaled_coordinate_fallback_v2",
                    "provider_detail": {
                        "raw_replay": {
                            "source_run_log": str(run_log_json_path),
                            "source_action_count": len(source_actions),
                            "actions_executed": int(actions_executed),
                            "selector_actions": int(selector_actions),
                            "scaled_coordinate_actions": int(
                                scaled_coordinate_actions
                            ),
                            "selector_fallback_actions": int(
                                selector_fallback_actions
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
                                "selector_actions": int(selector_actions),
                                "scaled_coordinate_actions": int(
                                    scaled_coordinate_actions
                                ),
                                "selector_fallback_actions": int(
                                    selector_fallback_actions
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
                "source": "selector_then_scaled_coordinate_fallback_v2",
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
    step_skill_guidance: str = "",
    max_steps: int = 20,
    raw_replay_run_log: str = "",
    appagent_root: str = "",
    appagent_workspace_root: str = "",
    appagent_docs_root: str = "",
    appagent_teacher_source: str = "",
    appagent_demo_name: str = "",
    appagent_output_root: str = "",
    task_seed: int | None = None,
    evidence_root: str = "",
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
    return default_method_adapter_registry().build(
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
            step_skill_guidance=step_skill_guidance,
            max_steps=max_steps,
            raw_replay_run_log=raw_replay_run_log,
            appagent_root=appagent_root,
            appagent_workspace_root=appagent_workspace_root,
            appagent_docs_root=appagent_docs_root,
            appagent_teacher_source=appagent_teacher_source,
            appagent_demo_name=appagent_demo_name,
            appagent_output_root=appagent_output_root,
            task_seed=task_seed,
            evidence_root=evidence_root,
            build_omniflow_agent=build_agent,
            apply_fixed_replay=_apply_fixed_replay,
            build_official_agent=_build_official_androidworld_agent,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run AndroidWorld as one unified test shell. "
            "Use `--agent omniflow` for the shared OmniFlow cache-first path, "
            "or `--agent official:<name>` for one upstream AndroidWorld agent."
        )
    )
    parser.add_argument("--android-world-root", required=True)
    parser.add_argument("--suite-family", default="android_world")
    parser.add_argument("--tasks", default="ContactsAddContact")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--task-random-seed", type=int, default=30)
    parser.add_argument("--n-task-combinations", type=int, default=1)
    parser.add_argument("--console-port", type=int, default=5554)
    parser.add_argument("--adb-path", default=_default_adb_path())
    parser.add_argument("--perform-emulator-setup", action="store_true")
    parser.add_argument("--fixed-task-seed", action="store_true")
    parser.add_argument(
        "--publish-success-source-runlog",
        action="store_true",
        help=(
            "Publish an official-validator successful canonical run to the global "
            "offline source pool. Disabled by default so target evaluation cannot "
            "contaminate future source memory."
        ),
    )
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument(
        "--candidate-harness-proposal",
        default="",
        help=(
            "Optional SkyMark harness_revision.v1 wrapper applied only to an "
            "unregistered official:t3a or official:m3a capture."
        ),
    )
    parser.add_argument(
        "--agent",
        default=MODE_OMNIFLOW,
        help=(
            "Agent selector. `omniflow` keeps the shared cache-first adapter; "
            "`external:mobilegpt` delegates one official episode to MobileGPT; "
            "`external:appagent` runs pinned AppAgent deployment; "
            "`external:appagent_teacher` captures one source human demo; "
            "`official:t3a` and `official:m3a` run immutable stock AndroidWorld "
            "agents for unregistered SkyMark capture; `official:t3a_gpt4` "
            "keeps the paper baseline compatibility path."
        ),
    )
    parser.add_argument(
        "--output-path",
        default=str(
            (OMNIFLOW_ROOT / "runtime" / "logs" / "androidworld_runs").resolve()
        ),
    )
    parser.add_argument(
        "--store-path",
        dest="store_path",
        default=str((OMNIFLOW_ROOT / "runtime" / "omniflow" / "store.json").resolve()),
        help="Function Store path.",
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
    parser.add_argument("--appagent-demo-name", default="")
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
        default=os.environ.get("OMNIFLOW_MODEL_ENDPOINT_PROFILE") or "auto",
        help="Credential and endpoint profile for the selected model.",
    )
    parser.add_argument(
        "--planner-timeout-sec",
        type=float,
        default=float(os.environ.get("OMNIFLOW_PLANNER_TIMEOUT_SEC") or 60.0),
        help="Per-call timeout in seconds for the online OmniFlow planner.",
    )
    parser.add_argument(
        "--step-skill-guidance-path",
        default="",
        help=(
            "Optional UTF-8 text artifact containing task-independent candidate "
            "Harness guidance for the online OmniFlow planner."
        ),
    )
    return parser


def _read_step_skill_guidance(path_value: object) -> str:
    text = str(path_value or "").strip()
    if not text:
        return ""
    path = Path(text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"step skill guidance artifact not found: {path}")
    guidance = path.read_text(encoding="utf-8").strip()
    if not guidance:
        raise ValueError(f"step skill guidance artifact is empty: {path}")
    return guidance


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
    android_world_root = Path(args.android_world_root).expanduser().resolve()
    run_py = android_world_root / "run.py"
    if not run_py.exists():
        raise FileNotFoundError(f"run.py not found under {android_world_root}")
    task_params = _decode_task_params(
        args.task_params_json,
        task_random_seed=int(args.task_random_seed),
    )
    step_skill_guidance = _read_step_skill_guidance(
        args.step_skill_guidance_path
    )

    env = None
    try:
        _add_android_world_path(android_world_root)

        from android_world import checkpointer as checkpointer_lib
        try:
            from android_world import constants
        except ImportError:
            class _EpisodeConstants:
                GOAL = "goal"
                TASK_TEMPLATE = "task_template"
                EPISODE_DATA = "episode_data"
                IS_SUCCESSFUL = "is_successful"
                RUN_TIME = "run_time"
                FINISH_DTIME = "finish_dtime"
                EPISODE_LENGTH = "episode_length"
                AUX_DATA = "aux_data"
                SCREEN_CONFIG = "screen_config"
                EXCEPTION_INFO = "exception_info"
                SEED = "seed"

            constants = types.SimpleNamespace(
                STEP_NUMBER="step_number",
                EpisodeConstants=_EpisodeConstants,
            )
        from android_world import registry, suite_utils
        from android_world.env import env_launcher
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

        reuse_a11y_forwarder = _ensure_androidworld_a11y_forwarder(
            console_port=int(args.console_port),
            adb_path=str(args.adb_path or ""),
            apk_path=str(os.environ.get("OMNIFLOW_ANDROIDWORLD_A11Y_APK") or ""),
        )
        logger.info(
            "AndroidWorld accessibility forwarder mode: %s",
            "reuse-installed" if reuse_a11y_forwarder else "install-official",
        )
        env = env_launcher.load_and_setup_env(
            console_port=int(args.console_port),
            emulator_setup=False,
            adb_path=str(args.adb_path or ""),
            grpc_port=int(args.console_port) + 3000,
            install_a11y_forwarding_app=not reuse_a11y_forwarder,
        )
        _wait_for_androidworld_a11y(env)
        if bool(args.perform_emulator_setup):
            logger.info(
                "Setting up AndroidWorld snapshots for selected tasks: %s",
                ", ".join(selected_task_names) or "<all>",
            )
            aw_setup.setup_apps(env, app_list=setup_app_list)
        _prepare_androidworld_snapshot_restore(env, setup_app_list or ())
        if task_params:
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

        experiment_environment = AndroidWorldExperimentEnvironment(
            env,
            AndroidWorldEnvironmentConfig(evidence_root=run_output_dir),
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
            step_skill_guidance=step_skill_guidance,
            max_steps=max(1, int(args.max_steps)),
            appagent_root=str(args.appagent_root or ""),
            appagent_workspace_root=str(args.appagent_workspace_root or ""),
            appagent_docs_root=str(args.appagent_docs_root or ""),
            appagent_teacher_source=str(args.appagent_teacher_source or ""),
            appagent_demo_name=str(args.appagent_demo_name or ""),
            appagent_output_root=str(run_output_dir / "appagent_runtime"),
            task_seed=int(args.task_random_seed),
            evidence_root=str(run_output_dir),
        )
        candidate_harness = None
        if str(args.candidate_harness_proposal or "").strip():
            if selected_agent not in {"official:t3a", "official:m3a"}:
                raise ValueError(
                    "candidate_harness_proposal_requires_stock_capture_agent"
                )
            candidate_harness = _apply_candidate_harness_proposal(
                agent,
                args.candidate_harness_proposal,
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
        for stale_output_path in (
            task_results_path,
            run_output_dir / "summary.json",
            run_output_dir / "summary.md",
        ):
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
            or selected_agent == "external:appagent"
            else {}
        )
        started_at = utc_now_iso()
        started_perf = perf_counter()
        result: dict[str, Any] | None = None
        mainline_name = selected_agent
        instrumented_agent = _ExperimentAgentAdapter(
            agent,
            recording_session=recording_session,
            goal_hint=official_goal_hint_text,
            max_steps=max(1, int(args.max_steps)),
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
            results = suite_utils.run(
                suite,
                instrumented_agent,
                checkpointer=checkpointer,
                demo_mode=False,
                return_full_episode_data=True,
            )
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
                mobilegpt_agent_result: dict[str, Any] = {}
                mobilegpt_agent_error = ""
                runtime_integrity_error = None
                if selected_agent == "external:mobilegpt":
                    raw_agent_result = getattr(
                        agent,
                        "last_result_data",
                        None,
                    )
                    if isinstance(raw_agent_result, dict):
                        mobilegpt_agent_result = dict(raw_agent_result)
                        mobilegpt_agent_error = str(
                            mobilegpt_agent_result.get("error") or ""
                        ).strip()
                        runtime_integrity_error = (
                            _mobilegpt_runtime_integrity_error(
                                mobilegpt_agent_error
                            )
                        )
                        if runtime_integrity_error:
                            error_text = runtime_integrity_error
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
                    if selected_agent.startswith("external:appagent"):
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
                if mobilegpt_agent_result:
                    actions_executed = max(
                        actions_executed,
                        _coerce_int(
                            mobilegpt_agent_result.get("actions_executed")
                        ),
                    )
                if selected_agent == "external:appagent":
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
                        )
                if selected_agent.startswith("official:") or selected_agent == (
                    "external:appagent"
                ):
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
                elif selected_agent.startswith("official:") or selected_agent.startswith(
                    "external:appagent"
                ):
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
                    "action_backend": "androidworld",
                    "native_androidworld_agent_io": True,
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
                if selected_agent == "external:appagent":
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
                if mobilegpt_agent_result:
                    task_result_record["mobilegpt_agent_result"] = (
                        to_serializable(mobilegpt_agent_result)
                    )
                    task_result_record["mobilegpt_agent_error"] = (
                        mobilegpt_agent_error or None
                    )
                    task_result_record["runtime_integrity_error"] = (
                        runtime_integrity_error
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
                stock_capture = _persist_official_step_captures(
                    output_dir=run_output_dir,
                    agent=agent,
                    selected_agent=selected_agent,
                    task_name=task_name,
                    goal=goal_text,
                    task_params_sha256=evaluation_task_params_sha256,
                    version_id=(
                        str(candidate_harness.get("harness_version_id") or "candidate")
                        if candidate_harness
                        else "stock"
                    ),
                    candidate_proposal=candidate_harness,
                )
                if stock_capture is not None:
                    task_result_record["skymark_stock_capture"] = stock_capture
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
                if (
                    task_success
                    and canonical_run is not None
                    and bool(args.publish_success_source_runlog)
                ):
                    try:
                        source_pool_record = _append_unique_source_pool_record(
                            task_name=task_name,
                            goal=goal_text,
                            params=evaluation_task_params,
                            task_random_seed=int(args.task_random_seed),
                            canonical_run=canonical_run,
                            task_result_record=task_result_record,
                        )
                        task_result_record["source_pool_record"] = {
                            "local_canonical_run_log": source_pool_record.get(
                                "local_canonical_run_log"
                            ),
                            "run_id": source_pool_record.get("run_id"),
                            "task": source_pool_record.get("task"),
                        }
                    except Exception as exc:  # noqa: BLE001
                        task_result_record["source_pool_record_error"] = str(exc)
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
        run_summary = _write_task_results_summary(
            task_results_path=task_results_path,
            output_dir=Path(args.output_path).expanduser().resolve(),
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
        runtime_integrity_exit_code = _mobilegpt_runtime_integrity_exit_code(
            run_summary
        )
        if runtime_integrity_exit_code:
            print(
                "[error] AndroidWorld episode ended with a MobileGPT runtime "
                "integrity failure.",
                flush=True,
            )
            return runtime_integrity_exit_code
        return 0
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
