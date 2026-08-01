from __future__ import annotations

import argparse
import base64
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
import re
import socket
import subprocess
import sys
import time
from time import perf_counter
import types
from typing import Any, Sequence
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from omniflow.vlm.usage import token_usage_status
from src.experiment.observation_evidence import (
    ObservationArchive,
    persist_target_run_evidence,
)
from src.integrations.android_world.agent import (
    MODE_OMNIFLOW,
    build_agent,
)
from src.integrations.android_world.host import make_agent_result
from src.integrations.android_world.setup_compat import (
    patch_androidworld_setup_click_retry,
    patch_androidworld_setup_fail_closed,
    restore_task_app_snapshots_after_initialize,
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


def normalize_oob_get_state_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize records used only by retired, unreachable OOB helpers."""
    state = payload.get("state")
    if not isinstance(state, dict):
        return payload
    normalized = dict(payload)
    for key, value in state.items():
        if value is not None or key not in normalized:
            normalized[key] = value
    return normalized


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read_env_text(name: str) -> str | None:
    value = str(os.environ.get(name) or "").strip()
    return value or None


def read_env_bool(name: str, default: bool) -> bool:
    value = read_env_text(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


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


class _TaskGoalProxy:
    def __init__(self, task: Any, goal: str) -> None:
        self._task = task
        self.goal = goal

    def __getattr__(self, name: str) -> Any:
        return getattr(self._task, name)


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
            or "qwen3-vl-plus"
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

    @staticmethod
    def _chat_completions_url(base_url: str) -> str:
        normalized = str(base_url or "").strip().rstrip("/")
        if not normalized:
            normalized = "https://api.openai.com/v1"
        if normalized.endswith("/chat/completions"):
            return normalized
        return f"{normalized}/chat/completions"

    @staticmethod
    def _encode_image(image: Any) -> str:
        import io

        from PIL import Image

        if isinstance(image, Image.Image):
            pil_image = image
        else:
            pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def predict(self, text_prompt: str) -> tuple[str, bool | None, Any]:
        return self.predict_mm(text_prompt, [])

    def predict_mm(self, text_prompt: str, images: list[Any]) -> tuple[str, bool | None, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": str(text_prompt)}]
        for image in list(images or []):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{self._encode_image(image)}"
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
                        return str(content_text or ""), None, last_response
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
    prompt_tokens = sum(_coerce_int(row.get("prompt_tokens")) for row in rows)
    completion_tokens = sum(_coerce_int(row.get("completion_tokens")) for row in rows)
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
                "model_calls": _coerce_int(row.get("model_calls")),
                "prompt_tokens": _coerce_int(row.get("prompt_tokens")),
                "completion_tokens": _coerce_int(row.get("completion_tokens")),
                "total_tokens": _coerce_int(row.get("total_tokens")),
                "error": row.get("error"),
            }
        )

    summary = {
        "schema_version": "omniflow.androidworld_run_summary.v2",
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
        "model_calls": total_model_calls,
        "avg_model_calls_per_task": round(
            total_model_calls / max(1, total_tasks), 3
        ),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "avg_tokens_per_task": round(total_tokens / max(1, total_tasks), 3),
        "avg_tokens_per_action": round(total_tokens / max(1, total_actions), 3),
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
        f"- total duration: `{round(total_duration_ms / 1000.0, 3)}s`",
        f"- actions executed: `{total_actions}`",
        f"- single-step execution accuracy: `{summary['single_step_execution_accuracy']}`",
        f"- validator-weighted action accuracy: `{summary['validator_weighted_action_accuracy']}`",
        f"- model calls / tokens: `{total_model_calls}` / `{total_tokens}`",
        "",
        "| task | official_validator | sec | actions | step_acc | calls | tokens |",
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
        f"calls={total_model_calls} tokens={total_tokens} "
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


def _accessibility_bound_to_oob(
    *, dumpsys: str, package_name: str, service: str
) -> tuple[bool, dict[str, Any]]:
    """Check whether OOB is actually present in the bound accessibility section."""

    text = str(dumpsys or "")
    lines = text.splitlines()
    bound_start = next(
        (index for index, line in enumerate(lines) if "Bound services" in line),
        -1,
    )
    if bound_start < 0:
        return False, {"reason": "missing_bound_services_section"}
    bound_end = len(lines)
    for index in range(bound_start + 1, len(lines)):
        stripped = lines[index].strip()
        if (
            stripped.startswith("Enabled services")
            or stripped.startswith("Binding services")
            or stripped.startswith("Crashed services")
            or stripped.startswith("User state")
            or stripped.startswith("Client list info")
        ):
            bound_end = index
            break
    bound_segment = "\n".join(lines[bound_start:bound_end])
    enabled_line = next(
        (line.strip() for line in lines if line.strip().startswith("Enabled services")),
        "",
    )
    package_name = str(package_name or "").strip()
    service = str(service or "").strip()
    class_name = service.split("/", 1)[1] if "/" in service else service
    class_tail = class_name.rsplit(".", 1)[-1].strip()
    matched_markers = [
        marker
        for marker in (service, package_name, class_name, class_tail)
        if marker and marker in bound_segment
    ]
    enabled_exact = bool(service and service in enabled_line)
    bound = (
        package_name in matched_markers
        and (
            service in matched_markers
            or class_name in matched_markers
            or class_tail in matched_markers
        )
    ) or ("Service[" in bound_segment and enabled_exact)
    return bound, {
        "reason": "matched_oob_bound_service" if bound else "oob_not_in_bound_services",
        "matched_markers": matched_markers,
        "enabled_exact": enabled_exact,
        "bound_services_tail": bound_segment[-1000:],
    }


def _oob_state_has_visible_page(state_payload: dict[str, Any]) -> bool:
    """Return whether OOB get_state captured a non-empty foreground page."""

    state_payload = normalize_oob_get_state_payload(state_payload)
    if state_payload.get("success") is not True:
        return False
    package_name = str(state_payload.get("package_name") or "").strip()
    activity_name = str(state_payload.get("activity_name") or "").strip()
    try:
        xml_chars = int(state_payload.get("xml_chars") or 0)
    except (TypeError, ValueError):
        xml_chars = 0
    xml_text = str(state_payload.get("xml") or "").strip()
    return bool(package_name and activity_name and (xml_chars > 0 or xml_text))


def _read_oob_debug_get_state_payload(
    *,
    adb_serial: str,
    adb_path: str = "",
    max_xml_chars: int = 200000,
    include_screenshot: bool = False,
    timeout_sec: float = 30.0,
) -> dict[str, Any]:
    """Read OOB get_state through the debug receiver without requiring HTTP."""

    package_name = str(
        os.environ.get("OMNIFLOW_OOB_PACKAGE") or "cn.com.omnimind.bot.debug"
    ).strip()
    receiver = str(
        os.environ.get("OMNIFLOW_OOB_GET_STATE_RECEIVER") or ".DebugGetStateReceiver"
    ).strip()
    component = (
        f"{package_name}/{receiver}"
        if receiver.startswith(".")
        else receiver
        if "/" in receiver
        else f"{package_name}/.{receiver}"
    )
    result_path = "files/debug-get-state-result.json"
    commands: list[dict[str, Any]] = []
    clear_result = _run_adb_command(
        adb_serial=adb_serial,
        adb_path=adb_path,
        adb_args=["shell", "run-as", package_name, "rm", "-f", result_path],
        timeout_sec=10,
    )
    commands.append(clear_result)
    broadcast_result = _run_adb_command(
        adb_serial=adb_serial,
        adb_path=adb_path,
        adb_args=[
            "shell",
            "am",
            "broadcast",
            "-a",
            f"{package_name}.RUN_GET_STATE",
            "-n",
            component,
            "--ez",
            "includeXml",
            "true",
            "--ez",
            "includeScreenshot",
            "true" if include_screenshot else "false",
            "--ez",
            "includeIndexedContext",
            "false",
            "--ei",
            "maxXmlChars",
            str(max(0, int(max_xml_chars))),
        ],
        timeout_sec=30,
    )
    commands.append(broadcast_result)
    if broadcast_result["returncode"] != 0:
        return {
            "success": False,
            "error": "OOB debug get_state broadcast failed",
            "commands": commands,
        }
    deadline = time.monotonic() + max(1.0, float(timeout_sec))
    last_error = ""
    while time.monotonic() < deadline:
        read_result = _run_adb_command(
            adb_serial=adb_serial,
            adb_path=adb_path,
            adb_args=["shell", "run-as", package_name, "cat", result_path],
            timeout_sec=10,
            capture_stdout=True,
        )
        stdout = str(read_result.get("stdout") or "").strip()
        if read_result["returncode"] == 0 and stdout:
            try:
                decoded = json.loads(stdout)
            except json.JSONDecodeError as exc:
                return {
                    "success": False,
                    "error": f"OOB debug get_state returned invalid JSON: {exc}",
                    "raw_tail": stdout[-1000:],
                    "commands": [*commands, read_result],
                }
            if isinstance(decoded, dict):
                decoded.setdefault("commands", [*commands, read_result])
                return normalize_oob_get_state_payload(decoded)
            return {
                "success": False,
                "error": "OOB debug get_state returned non-object JSON",
                "commands": [*commands, read_result],
            }
        last_error = str(
            read_result.get("stderr_tail") or read_result.get("stdout_tail") or ""
        ).strip()
        time.sleep(0.5)
    return {
        "success": False,
        "error": "OOB debug get_state result was not written: " + last_error[-500:],
        "commands": commands,
    }


def _oob_payload_int(
    payload: dict[str, Any],
    keys: tuple[str, ...],
    *,
    default: int,
) -> int:
    for key in keys:
        try:
            value = int(payload.get(key) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    return max(1, int(default or 1))


def _oob_payload_blank_pixels(payload: dict[str, Any]) -> Any:
    try:
        import numpy as np  # type: ignore

        width = _oob_payload_int(
            payload,
            ("display_width", "xml_display_width", "width"),
            default=1,
        )
        height = _oob_payload_int(
            payload,
            ("display_height", "xml_display_height", "height"),
            default=1,
        )
        return np.zeros((height, width, 3), dtype=np.uint8)
    except Exception:
        return None


def _oob_payload_pixels(payload: dict[str, Any]) -> Any:
    screenshot = payload.get("screenshot")
    if isinstance(screenshot, dict):
        encoded = (
            screenshot.get("data")
            or screenshot.get("data_uri")
            or screenshot.get("dataUri")
            or screenshot.get("image_base64")
        )
    else:
        encoded = screenshot if isinstance(screenshot, str) else ""
    if isinstance(encoded, str) and encoded.strip():
        raw = encoded.strip()
        if raw.startswith("data:image/") and "," in raw:
            raw = raw.split(",", 1)[1]
        try:
            import numpy as np  # type: ignore
            from PIL import Image

            image = Image.open(io.BytesIO(base64.b64decode(raw, validate=False))).convert(
                "RGB"
            )
            return np.asarray(image)
        except Exception as exc:  # noqa: BLE001
            logging.warning("OOB AndroidWorld screenshot decode failed: %s", exc)
    return _oob_payload_blank_pixels(payload)


def _read_oob_androidworld_state(
    *,
    oob_url: str,
    adb_serial: str = "",
    adb_path: str = "",
) -> Any | None:
    """Read one AndroidWorld-like state from OOB /get_state."""

    try:
        from android_world.env import representation_utils

        if oob_url:
            query = urllib.parse.urlencode(
                {
                    "includeXml": "true",
                    "includeScreenshot": "true",
                    "includeIndexedContext": "false",
                    "maxXmlChars": "200000",
                    "filterOverlay": "true",
                }
            )
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(f"{oob_url}/get_state?{query}", timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
        else:
            payload = _read_oob_debug_get_state_payload(
                adb_serial=(
                    str(adb_serial or os.environ.get("ANDROID_SERIAL") or "").strip()
                ),
                adb_path=adb_path,
                max_xml_chars=200000,
                include_screenshot=True,
            )
            if not _oob_state_has_visible_page(payload):
                logging.warning("OOB AndroidWorld state proxy failed: %s", payload)
                return None
        payload = normalize_oob_get_state_payload(payload)
        xml_text = str(payload.get("xml") or "").strip()
        ui_elements = (
            list(representation_utils.xml_dump_to_ui_elements(xml_text) or [])
            if xml_text
            else []
        )
        return types.SimpleNamespace(
            pixels=_oob_payload_pixels(payload),
            forest=None,
            ui_elements=ui_elements,
            auxiliaries={},
            xml=xml_text,
            activity_name=str(payload.get("activity_name") or ""),
            package_name=str(payload.get("package_name") or ""),
            raw_state=payload,
        )
    except Exception as exc:  # noqa: BLE001
        logging.warning("OOB AndroidWorld state proxy failed: %s", exc)
        return None


def _use_oob_observe_backend() -> bool:
    return False


def _agent_uses_oob_observation(agent: Any) -> bool:
    del agent
    return False


def _check_oob_http_state_ready(
    *,
    oob_url: str,
    port: int,
    run_adb: Any,
    diagnostics: dict[str, Any],
    state_attempts: int = 120,
) -> dict[str, Any]:
    """Check the OOB local HTTP host using the same observe path as online runs."""

    if port > 0:
        run_adb("forward", "--remove", f"tcp:{port}")
        run_adb("forward", f"tcp:{port}", f"tcp:{port}")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    last_error = ""
    for _ in range(20):
        try:
            with opener.open(f"{oob_url}/health", timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("success") is True:
                diagnostics["health"] = payload
                break
            last_error = json.dumps(payload, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001
            last_error = f"{exc.__class__.__name__}: {exc}"
        time.sleep(0.5)
    else:
        diagnostics["error"] = last_error or "OOB health check did not succeed"
        return diagnostics
    state_query = urllib.parse.urlencode(
        {
            "includeXml": "true",
            "includeScreenshot": "false",
            "maxXmlChars": "1000",
            "filterOverlay": "true",
        }
    )
    for attempt in range(max(1, int(state_attempts))):
        try:
            with opener.open(f"{oob_url}/get_state?{state_query}", timeout=5) as response:
                state_payload = json.loads(response.read().decode("utf-8"))
            state_payload = normalize_oob_get_state_payload(state_payload)
            if _oob_state_has_visible_page(state_payload):
                diagnostics["success"] = True
                diagnostics["state_ready"] = {
                    "package_name": state_payload.get("package_name"),
                    "activity_name": state_payload.get("activity_name"),
                    "xml_chars": state_payload.get("xml_chars"),
                }
                return diagnostics
            last_error = str(state_payload.get("error") or state_payload)
        except Exception as exc:  # noqa: BLE001
            last_error = f"{exc.__class__.__name__}: {exc}"
        diagnostics["state_ready_attempts"] = attempt + 1
        diagnostics["state_ready_last_error"] = last_error[-1000:]
        time.sleep(0.5)
    diagnostics["error"] = last_error or "OOB get_state readiness check did not succeed"
    return diagnostics


def _ensure_oob_http_state_ready_after_task_init(
    *,
    oob_url: str,
    adb_serial: str,
    console_port: int,
    adb_path: str = "",
    attempts: int = 20,
) -> dict[str, Any]:
    """Wait for OOB /get_state after AndroidWorld task init without changing UI.

    This deliberately does not launch or force-stop the OOB app. It only restores
    adb forward and waits for the current task page to be observable, so the
    AndroidWorld reset/initialize screen is not disturbed.
    """

    raw_url = str(oob_url or "").strip().rstrip("/")
    diagnostics: dict[str, Any] = {
        "success": False,
        "url": raw_url or None,
        "adb_serial": adb_serial or None,
        "mode": "post_task_init_oob_http_ready_check",
    }
    if not raw_url:
        diagnostics["error"] = "missing_oob_device_url"
        return diagnostics
    parsed = urllib.parse.urlparse(raw_url)
    port = int(parsed.port or 0)
    if port <= 0:
        port = 8910
    adb = str(adb_path or _default_adb_path() or "adb").strip() or "adb"

    def run_adb(*parts: str) -> dict[str, Any]:
        command = [adb]
        if adb_serial:
            command.extend(["-s", adb_serial])
        command.extend(parts)
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=15,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-1000:],
            "stderr_tail": completed.stderr[-1000:],
        }

    try:
        diagnostics["forward_remove"] = run_adb("forward", "--remove", f"tcp:{port}")
        diagnostics["forward"] = run_adb("forward", f"tcp:{port}", f"tcp:{port}")
        return _check_oob_http_state_ready(
            oob_url=raw_url,
            port=0,
            run_adb=run_adb,
            diagnostics=diagnostics,
            state_attempts=max(1, int(attempts)),
        )
    except Exception as exc:  # noqa: BLE001
        diagnostics["error"] = f"{exc.__class__.__name__}: {exc}"
        return diagnostics


def _prepare_oob_device_host_for_replay(
    *,
    adb_serial: str,
    adb_path: str = "",
) -> dict[str, Any]:
    """Restore OOB observe/act host immediately before Function replay."""

    oob_url = str(os.environ.get("OMNIFLOW_OOB_DEVICE_URL") or "").strip().rstrip("/")
    parsed = urllib.parse.urlparse(oob_url)
    port = int(parsed.port or 0)
    package_name = str(os.environ.get("OMNIFLOW_OOB_PACKAGE") or "cn.com.omnimind.bot.debug").strip()
    activity = str(
        os.environ.get("OMNIFLOW_OOB_ACTIVITY")
        or "cn.com.omnimind.bot.activity.LauncherActivity"
    ).strip()
    service = str(
        os.environ.get("OMNIFLOW_OOB_ACCESSIBILITY_SERVICE")
        or f"{package_name}/com.google.android.accessibility.selecttospeak.SelectToSpeakService"
    ).strip()
    adb = str(adb_path or _default_adb_path() or "adb").strip() or "adb"
    diagnostics: dict[str, Any] = {
        "enabled": True,
        "url": oob_url or None,
        "adb_serial": adb_serial or None,
        "package_name": package_name,
        "accessibility_service": service,
        "success": False,
        "commands": [],
    }

    def run_adb(
        *parts: str,
        tolerate_timeout: bool = False,
    ) -> dict[str, Any]:
        command = [adb]
        if adb_serial:
            command.extend(["-s", adb_serial])
        command.extend(parts)
        try:
            completed = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=15,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = (
                exc.stdout.decode("utf-8", errors="replace")
                if isinstance(exc.stdout, bytes)
                else str(exc.stdout or "")
            )
            stderr = (
                exc.stderr.decode("utf-8", errors="replace")
                if isinstance(exc.stderr, bytes)
                else str(exc.stderr or "")
            )
            record = {
                "command": command,
                "returncode": 124,
                "stdout_tail": stdout[-1000:],
                "stderr_tail": stderr[-1000:],
                "timed_out": True,
                "timeout_sec": 15,
            }
            diagnostics["commands"].append(record)
            if not tolerate_timeout:
                raise
            return {**record, "stdout": stdout, "stderr": stderr}
        record = {
            "command": command,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-1000:],
            "stderr_tail": completed.stderr[-1000:],
        }
        diagnostics["commands"].append(record)
        return {**record, "stdout": completed.stdout, "stderr": completed.stderr}

    try:
        current_services_result = run_adb(
            "shell",
            "settings",
            "get",
            "secure",
            "enabled_accessibility_services",
        )
        enabled_services = [
            item
            for item in str(current_services_result.get("stdout") or "").strip().split(":")
            if item and item != "null"
        ]
        if service not in enabled_services:
            enabled_services.append(service)
        merged_services = ":".join(enabled_services)
        diagnostics["enabled_accessibility_services"] = enabled_services
        run_adb(
            "shell",
            "settings",
            "put",
            "secure",
            "enabled_accessibility_services",
            merged_services,
        )
        run_adb("shell", "settings", "put", "secure", "accessibility_enabled", "1")
        if oob_url:
            preflight = _check_oob_http_state_ready(
                oob_url=oob_url,
                port=port,
                run_adb=run_adb,
                diagnostics=diagnostics,
                state_attempts=6,
            )
            diagnostics["preflight_state_ready"] = {
                key: value for key, value in preflight.items() if key != "commands"
            }
            if preflight.get("success") is True:
                diagnostics["reused_ready_host"] = True
                diagnostics["success"] = True
                return diagnostics
            diagnostics.pop("error", None)
            if package_name and activity:
                run_adb("shell", "am", "start", "-n", f"{package_name}/{activity}")
                time.sleep(2.0)
                post_start = _check_oob_http_state_ready(
                    oob_url=oob_url,
                    port=port,
                    run_adb=run_adb,
                    diagnostics=diagnostics,
                    state_attempts=12,
                )
                diagnostics["post_start_state_ready"] = {
                    key: value for key, value in post_start.items() if key != "commands"
                }
                if post_start.get("success") is True:
                    diagnostics["started_existing_host"] = True
                    diagnostics["success"] = True
                    return diagnostics
                diagnostics.pop("error", None)
            if package_name:
                diagnostics["restart_after_state_ready_failure"] = True
                run_adb("shell", "am", "force-stop", package_name)
                time.sleep(1.0)
        else:
            run_adb("shell", "am", "force-stop", package_name)
        last_dumpsys = ""
        last_bound_debug: dict[str, Any] = {}
        for bind_attempt in range(3):
            diagnostics["bind_attempt"] = bind_attempt + 1
            if bind_attempt > 0:
                run_adb("shell", "settings", "put", "secure", "accessibility_enabled", "0")
                time.sleep(0.5)
            run_adb(
                "shell",
                "settings",
                "put",
                "secure",
                "enabled_accessibility_services",
                merged_services,
            )
            run_adb("shell", "settings", "put", "secure", "accessibility_enabled", "1")
            if package_name and activity:
                run_adb("shell", "am", "start", "-n", f"{package_name}/{activity}")
                time.sleep(1.0)
            for _ in range(16):
                dumpsys = run_adb("shell", "dumpsys", "accessibility")
                last_dumpsys = str(dumpsys.get("stdout") or "")
                bound, bound_debug = _accessibility_bound_to_oob(
                    dumpsys=last_dumpsys,
                    package_name=package_name,
                    service=service,
                )
                last_bound_debug = bound_debug
                diagnostics["accessibility_bound_debug"] = bound_debug
                if bound:
                    diagnostics["accessibility_bound"] = True
                    break
                time.sleep(0.5)
            if diagnostics.get("accessibility_bound") is True:
                break
            if not oob_url:
                run_adb("shell", "am", "force-stop", package_name)
            time.sleep(0.5)
        else:
            diagnostics["accessibility_bound"] = False
            bind_error = (
                "OOB accessibility service did not bind before replay; "
                f"enabled_service={service}; bound_debug={last_bound_debug}; "
                f"dumpsys_tail={last_dumpsys[-1000:]}"
            )
            diagnostics["accessibility_bound_warning"] = bind_error
            if not oob_url:
                diagnostics["error"] = bind_error
                return diagnostics
        if package_name and activity:
            run_adb(
                "shell",
                "am",
                "start",
                "-a",
                "android.intent.action.MAIN",
                "-c",
                "android.intent.category.HOME",
            )
            time.sleep(0.5)
        if oob_url:
            ready_attempts = int(
                float(os.environ.get("OMNIFLOW_OOB_READY_ATTEMPTS") or "240")
            )
            return _check_oob_http_state_ready(
                oob_url=oob_url,
                port=port,
                run_adb=run_adb,
                diagnostics=diagnostics,
                state_attempts=ready_attempts,
            )
        get_state_receiver = str(
            os.environ.get("OMNIFLOW_OOB_GET_STATE_RECEIVER") or ".DebugGetStateReceiver"
        ).strip()
        get_state_component = (
            f"{package_name}/{get_state_receiver}"
            if get_state_receiver.startswith(".")
            else get_state_receiver
            if "/" in get_state_receiver
            else f"{package_name}/.{get_state_receiver}"
        )
        state_result_path = "files/debug-get-state-result.json"
        run_adb("shell", "run-as", package_name, "rm", "-f", state_result_path)
        def request_debug_state() -> None:
            run_adb(
                "shell",
                "am",
                "broadcast",
                "-a",
                f"{package_name}.RUN_GET_STATE",
                "-n",
                get_state_component,
                "--ez",
                "includeXml",
                "true",
                "--ez",
                "includeScreenshot",
                "false",
                "--ei",
                "maxXmlChars",
                "1000",
                tolerate_timeout=True,
            )
            diagnostics["debug_get_state_broadcasts"] = int(
                diagnostics.get("debug_get_state_broadcasts") or 0
            ) + 1

        request_debug_state()
        last_state_error = ""
        for state_attempt in range(120):
            if state_attempt > 0 and state_attempt % 20 == 0:
                request_debug_state()
            state_file = run_adb(
                "shell",
                "run-as",
                package_name,
                "cat",
                state_result_path,
            )
            stdout = str(state_file.get("stdout") or "").strip()
            if state_file["returncode"] == 0 and stdout:
                try:
                    state_payload = json.loads(stdout)
                except json.JSONDecodeError as exc:
                    last_state_error = f"invalid get_state json: {exc}"
                else:
                    state_payload = normalize_oob_get_state_payload(state_payload)
                    diagnostics["app_state_ready"] = {
                        "success": bool(state_payload.get("success")),
                        "package_name": state_payload.get("package_name"),
                        "activity_name": state_payload.get("activity_name"),
                        "xml_chars": state_payload.get("xml_chars"),
                        "error": state_payload.get("error_message")
                        or state_payload.get("error"),
                    }
                    if _oob_state_has_visible_page(state_payload):
                        break
                    last_state_error = str(
                        state_payload.get("error_message")
                        or state_payload.get("error")
                        or diagnostics["app_state_ready"]
                        or state_payload
                    )
            else:
                last_state_error = str(
                    state_file.get("stderr")
                    or state_file.get("stdout")
                    or state_file.get("stderr_tail")
                    or state_file.get("stdout_tail")
                    or ""
                ).strip()
            time.sleep(0.5)
        else:
            diagnostics["error"] = (
                "OOB app get_state did not become ready before replay; "
                f"receiver={get_state_component}; last_error={last_state_error[-1000:]}"
            )
            return diagnostics
        diagnostics["success"] = True
        return diagnostics
    except Exception as exc:  # noqa: BLE001
        diagnostics["error"] = f"{exc.__class__.__name__}: {exc}"
        return diagnostics


def _add_android_world_path(android_world_root: Path) -> None:
    root = str(android_world_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _rehydrate_task_params(
    *,
    params: dict[str, object],
) -> dict[str, object]:
    """Restore AndroidWorld task params that were serialized through JSON."""

    hydrated = dict(params)
    serialized_rows = [
        row
        for key in ("row_objects", "noise_row_objects")
        for row in (hydrated.get(key) if isinstance(hydrated.get(key), list) else [])
        if isinstance(row, dict)
    ]
    if not serialized_rows:
        return dict(params)
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


def _patch_androidworld_create_file_diagnostics(
    *,
    file_validators: Any,
    file_utils: Any,
    adb_utils: Any,
) -> Any:
    """Log CreateFile validator inputs and observed device files.

    This does not change AndroidWorld scoring. It only makes episode failures
    inspectable when the official validator returns 0 without
    surfacing whether the file was missing or content mismatched.
    """

    create_file_cls = getattr(file_validators, "CreateFile", None)
    original = getattr(create_file_cls, "is_successful", None)
    if create_file_cls is None or not callable(original):
        return None
    if getattr(create_file_cls, "_omniflow_diagnostics_patched", False):
        return original

    def _diagnostic_is_successful(self: Any, env: Any) -> float:
        params = getattr(self, "params", {}) or {}
        data_directory = str(getattr(self, "data_directory", "") or "")
        file_name = str(params.get("file_name") or "")
        expected_text = str(params.get("text") or "")
        diagnostics: dict[str, Any] = {
            "validator": "CreateFile",
            "data_directory": data_directory,
            "file_name": file_name,
            "expected_text": expected_text,
        }
        try:
            full_path = file_utils.convert_to_posix_path(data_directory, file_name)
            diagnostics["full_path"] = full_path
            diagnostics["exists_before_score"] = file_utils.check_file_or_folder_exists(
                file_name,
                data_directory,
                env.controller,
            )
            tree = adb_utils.issue_generic_request(
                ["shell", "find", data_directory, "-maxdepth", "3", "-print"],
                env.controller,
            )
            diagnostics["tree_status"] = bool(getattr(tree, "status", False))
            diagnostics["tree_tail"] = (
                tree.generic.output.decode(errors="replace").replace("\r", "")[-4000:]
            )
            if diagnostics["exists_before_score"]:
                content = adb_utils.issue_generic_request(
                    ["shell", "cat", full_path],
                    env.controller,
                )
                diagnostics["cat_status"] = bool(getattr(content, "status", False))
                diagnostics["content"] = (
                    content.generic.output.decode(errors="replace").replace("\r", "")
                )
        except Exception as exc:  # noqa: BLE001
            diagnostics["diagnostic_error"] = f"{exc.__class__.__name__}: {exc}"
        score = float(original(self, env))
        diagnostics["score"] = score
        print(
            "[omniflow-validator-diagnostic] "
            + json.dumps(to_serializable(diagnostics), ensure_ascii=False)
        )
        return score

    create_file_cls.is_successful = _diagnostic_is_successful
    create_file_cls._omniflow_diagnostics_patched = True
    return original


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


def _native_androidworld_a11y_method(android_world_controller: Any) -> Any:
    return android_world_controller.A11yMethod.A11Y_FORWARDER_APP


def _patch_androidworld_ui_debug_settings(
    android_world_controller: Any,
) -> Any | None:
    loader = getattr(android_world_controller, "loader", None)
    device_settings_module = getattr(loader, "device_settings_lib", None)
    device_settings_class = getattr(
        device_settings_module,
        "DeviceSettings",
        None,
    )
    original_update = getattr(device_settings_class, "update", None)
    if not callable(original_update):
        return None
    if bool(
        getattr(
            device_settings_class,
            "_omniflow_ui_debug_settings_patch",
            False,
        )
    ):
        return None

    def _update_without_ui_debug_overlays(self, config):
        if not hasattr(config, "show_pointer_location") or not hasattr(
            config,
            "show_touches",
        ):
            raise RuntimeError("androidworld_device_settings_config_missing")
        config.show_pointer_location = False
        config.show_touches = False
        return original_update(self, config)

    device_settings_class.update = _update_without_ui_debug_overlays
    device_settings_class._omniflow_ui_debug_settings_patch = True
    return original_update


def _quiesce_androidworld_accessibility_forwarder(
    *,
    adb_serial: str,
    adb_path: str = "",
) -> dict[str, Any]:
    blocked_components = {
        (
            "com.example.MobileGPT/"
            "com.example.MobileGPT.MobileGPTAccessibilityService"
        ),
        (
            "com.google.androidenv.accessibilityforwarder/"
            "com.google.androidenv.accessibilityforwarder.AccessibilityForwarder"
        ),
        (
            "cn.com.omnimind.bot.debug/"
            "com.google.android.accessibility.selecttospeak.SelectToSpeakService"
        ),
    }
    current = _run_adb_command(
        adb_serial=adb_serial,
        adb_path=adb_path,
        adb_args=[
            "shell",
            "settings",
            "get",
            "secure",
            "enabled_accessibility_services",
        ],
        timeout_sec=15,
        capture_stdout=True,
    )
    if current["returncode"] != 0:
        raise RuntimeError("androidworld_accessibility_services_read_failed")
    services = [
        service
        for service in str(current.get("stdout") or "").strip().split(":")
        if service and service != "null" and service not in blocked_components
    ]
    write_args = [
        "shell",
        "settings",
        "put" if services else "delete",
        "secure",
        "enabled_accessibility_services",
    ]
    if services:
        write_args.append(":".join(services))
    updated = _run_adb_command(
        adb_serial=adb_serial,
        adb_path=adb_path,
        adb_args=write_args,
        timeout_sec=15,
        capture_stdout=True,
    )
    if updated["returncode"] != 0:
        raise RuntimeError("androidworld_accessibility_services_write_failed")
    stopped = _run_adb_command(
        adb_serial=adb_serial,
        adb_path=adb_path,
        adb_args=[
            "shell",
            "am",
            "force-stop",
            "com.google.androidenv.accessibilityforwarder",
        ],
        timeout_sec=15,
        capture_stdout=True,
    )
    if stopped["returncode"] != 0:
        raise RuntimeError("androidworld_a11y_forwarder_stop_failed")
    return {
        "removed": not blocked_components.intersection(services),
        "remaining_services": services,
    }


def _prepare_native_androidworld_a11y_runtime(
    env: Any,
    *,
    adb_serial: str,
    adb_path: str = "",
) -> dict[str, Any]:
    forwarder = _quiesce_androidworld_accessibility_forwarder(
        adb_serial=adb_serial,
        adb_path=adb_path,
    )
    for setting_name in ("pointer_location", "show_touches"):
        disabled = _run_adb_command(
            adb_serial=adb_serial,
            adb_path=adb_path,
            adb_args=[
                "shell",
                "settings",
                "put",
                "system",
                setting_name,
                "0",
            ],
            timeout_sec=15,
            capture_stdout=True,
        )
        if disabled["returncode"] != 0:
            raise RuntimeError(f"androidworld_{setting_name}_disable_failed")
        checked = _run_adb_command(
            adb_serial=adb_serial,
            adb_path=adb_path,
            adb_args=[
                "shell",
                "settings",
                "get",
                "system",
                setting_name,
            ],
            timeout_sec=15,
            capture_stdout=True,
        )
        if (
            checked["returncode"] != 0
            or str(checked.get("stdout") or "").strip() != "0"
        ):
            raise RuntimeError(f"androidworld_{setting_name}_still_enabled")
    controller = getattr(env, "controller", None)
    refresh_env = getattr(controller, "refresh_env", None)
    if callable(refresh_env):
        refresh_env()
    close_dialogs = _run_adb_command(
        adb_serial=adb_serial,
        adb_path=adb_path,
        adb_args=[
            "shell",
            "am",
            "broadcast",
            "-a",
            "android.intent.action.CLOSE_SYSTEM_DIALOGS",
        ],
        timeout_sec=15,
        capture_stdout=True,
    )
    if close_dialogs["returncode"] != 0:
        raise RuntimeError("androidworld_close_system_dialogs_failed")
    dialog_deadline = time.monotonic() + 3.0
    while True:
        focused_window = _run_adb_command(
            adb_serial=adb_serial,
            adb_path=adb_path,
            adb_args=["shell", "dumpsys", "window", "windows"],
            timeout_sec=15,
            capture_stdout=True,
        )
        focused_text = str(focused_window.get("stdout") or "")
        crash_dialog_present = (
            "Application Error: com.google.androidenv.accessibilityforwarder"
            in focused_text
        )
        if focused_window["returncode"] == 0 and not crash_dialog_present:
            break
        if time.monotonic() >= dialog_deadline:
            raise RuntimeError("androidworld_a11y_forwarder_crash_dialog_present")
        time.sleep(0.1)
    state = env.get_state()
    ui_elements = list(getattr(state, "ui_elements", ()) or ())
    if not ui_elements:
        raise RuntimeError("androidworld_a11y_forwarder_not_ready")
    return {
        "ready": True,
        "ui_element_count": len(ui_elements),
        "controller_refreshed": callable(refresh_env),
        "forwarder_quiesced": bool(forwarder.get("removed")),
    }


def _wrap_task_initialize_for_observation_runtime(
    task: Any,
    *,
    agent: Any,
    adb_serial: str,
    adb_path: str,
    oob_url: str,
    console_port: int,
    restore_app_snapshot: Any | None = None,
    after_initialized: Any | None = None,
) -> None:
    original_initialize_task = getattr(task, "initialize_task", None)
    if not callable(original_initialize_task) or bool(
        getattr(task, "_omniflow_observation_runtime_wrapped", False)
    ):
        return

    def _initialize_task_with_ready_runtime(init_env, *init_args, **init_kwargs):
        uses_oob_observe = (
            _use_oob_observe_backend() or _agent_uses_oob_observation(agent)
        )
        if uses_oob_observe:
            pre_init_prepare = _prepare_oob_device_host_for_replay(
                adb_serial=adb_serial,
                adb_path=adb_path,
            )
            if not bool(pre_init_prepare.get("success")):
                raise RuntimeError(
                    "OOB /get_state not ready before task init: "
                    + str(pre_init_prepare.get("error") or pre_init_prepare)
                )
        initialized = original_initialize_task(
            init_env,
            *init_args,
            **init_kwargs,
        )
        if callable(restore_app_snapshot):
            restore_task_app_snapshots_after_initialize(
                restore_app_snapshot,
                task,
                init_env,
            )
        if callable(after_initialized):
            after_initialized(task)
        if uses_oob_observe:
            oob_prepare = (
                _ensure_oob_http_state_ready_after_task_init(
                    oob_url=oob_url,
                    adb_serial=adb_serial,
                    adb_path=adb_path,
                    console_port=console_port,
                )
                if oob_url
                else _prepare_oob_device_host_for_replay(
                    adb_serial=adb_serial,
                    adb_path=adb_path,
                )
            )
            if not bool(oob_prepare.get("success")):
                raise RuntimeError(
                    "OOB /get_state not ready after task init: "
                    + str(oob_prepare.get("error") or oob_prepare)
                )
        else:
            _prepare_native_androidworld_a11y_runtime(
                init_env,
                adb_serial=adb_serial,
                adb_path=adb_path,
            )
        return initialized

    task.initialize_task = _initialize_task_with_ready_runtime
    task._omniflow_observation_runtime_wrapped = True


def _patch_androidworld_settings_get_output(adb_utils_module: Any) -> Any | None:
    """Clean noisy adb output before AndroidWorld validators parse it."""

    original_issue_generic_request = getattr(
        adb_utils_module, "issue_generic_request", None
    )
    if not callable(original_issue_generic_request):
        return None
    if getattr(adb_utils_module, "_omniflow_settings_get_output_patch", False):
        return None

    def _is_android_env_noise_line(line: str) -> bool:
        stripped = line.strip()
        return (
            "FD from fork parent still in poll list" in stripped
            and re.match(r"^[IWEF]\d{4}\s+\d+", stripped) is not None
        )

    def _clean_adb_output(raw: bytes) -> bytes:
        text = raw.decode("utf-8", errors="replace")
        normalized = text.replace("\r", "")
        lines = normalized.splitlines()
        if not lines:
            return raw
        cleaned_lines = [
            line
            for line in lines
            if not _is_android_env_noise_line(line)
        ]
        if cleaned_lines == lines:
            return raw
        cleaned = "\n".join(cleaned_lines)
        if text.endswith("\n") and cleaned:
            cleaned += "\n"
        return cleaned.encode("utf-8")

    def _patched_issue_generic_request(args, env, timeout_sec=None):
        response = original_issue_generic_request(args, env, timeout_sec=timeout_sec)
        try:
            output = response.generic.output
            cleaned = _clean_adb_output(bytes(output or b""))
            if cleaned != output:
                response.generic.output = cleaned
        except Exception as exc:  # noqa: BLE001
            logging.warning("Failed to clean AndroidWorld adb output: %s", exc)
        return response

    adb_utils_module.issue_generic_request = _patched_issue_generic_request
    adb_utils_module._omniflow_settings_get_output_patch = True
    return original_issue_generic_request


def _patch_androidworld_empty_clear_directory(file_utils_module: Any) -> Any | None:
    """Ignore AndroidWorld snapshot cleanup failures caused by empty app data dirs.

    Args:
        file_utils_module: Imported `android_world.utils.file_utils` module. The
            helper patches `clear_directory(...)` in this launcher process only.

    Returns:
        The original `clear_directory` callable for later restoration, or
        `None` when the module is already patched or does not expose it.
    """

    original_clear_directory = getattr(file_utils_module, "clear_directory", None)
    if not callable(original_clear_directory):
        return None
    if getattr(file_utils_module, "_omniflow_empty_clear_directory_patch", False):
        return None

    def _safe_clear_directory(directory_path: str, env: object) -> None:
        try:
            original_clear_directory(directory_path, env)
        except Exception as exc:
            message = str(exc or "")
            if "rm -r" in message and "No such file or directory" in message:
                logging.warning(
                    "AndroidWorld clear_directory ignored empty app data glob for %s: %s",
                    directory_path,
                    message,
                )
                return
            raise

    file_utils_module.clear_directory = _safe_clear_directory
    file_utils_module._omniflow_empty_clear_directory_patch = True
    return original_clear_directory


def _maybe_patch_sqlite_backend() -> None:
    backend = (
        str(read_env_text("OMNIFLOW_ANDROIDWORLD_SQLITE_BACKEND") or "auto")
        .strip()
        .lower()
    )
    if backend not in {"auto", "pysqlite3"}:
        return
    try:
        import pysqlite3.dbapi2 as sqlite3_backend
    except Exception:
        if backend == "pysqlite3":
            raise
        return
    sys.modules["sqlite3"] = sqlite3_backend
    print("[info] sqlite_backend=pysqlite3", file=sys.stderr)


def _patch_androidworld_sqlite_writeback(sqlite_utils_module: Any) -> dict[str, Any] | None:
    """Make AndroidWorld SQLite task setup write back the main DB file.

    AndroidWorld pulls only the main SQLite file, mutates it locally, then
    pushes only that file back to the device. Apps that use WAL can keep the
    inserted rows in a local `-wal` file, so the pushed main DB remains empty
    and task initialization later reports "Found 0 in DB". The patch keeps the
    same AndroidWorld API but forces rollback-journal writes and removes stale
    device sidecars before/after pushing the main DB.
    """

    if getattr(sqlite_utils_module, "_omniflow_sqlite_writeback_patch", False):
        return None
    original_delete = getattr(sqlite_utils_module, "delete_all_rows_from_table", None)
    original_insert = getattr(sqlite_utils_module, "insert_rows_to_remote_db", None)
    if not callable(original_delete) or not callable(original_insert):
        return None

    sqlite3_module = sqlite_utils_module.sqlite3
    os_module = sqlite_utils_module.os
    time_module = sqlite_utils_module.time
    file_utils_module = sqlite_utils_module.file_utils
    adb_utils_module = sqlite_utils_module.adb_utils
    sqlite_schema_utils_module = sqlite_utils_module.sqlite_schema_utils

    def _prepare_main_db_connection(local_db_path: str):
        conn = sqlite3_module.connect(local_db_path)
        try:
            conn.execute("PRAGMA wal_checkpoint(FULL)")
        except sqlite3_module.DatabaseError:
            pass
        try:
            conn.execute("PRAGMA journal_mode=DELETE").fetchone()
        except sqlite3_module.DatabaseError:
            pass
        return conn

    def _remove_remote_sidecars(
        remote_db_file_path: str,
        env: object,
        timeout_sec: float | None,
    ) -> None:
        try:
            adb_utils_module.issue_generic_request(
                [
                    "shell",
                    "rm",
                    "-f",
                    f"{remote_db_file_path}-wal",
                    f"{remote_db_file_path}-shm",
                ],
                env.controller,
                timeout_sec=timeout_sec,
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning(
                "Failed to remove remote SQLite sidecars for %s: %s",
                remote_db_file_path,
                exc,
            )

    def _patched_delete_all_rows_from_table(
        table_name: str,
        remote_db_file_path: str,
        env: object,
        app_name: str,
        timeout_sec: float | None = None,
    ) -> None:
        try:
            table_exists = sqlite_utils_module.table_exists(
                table_name,
                remote_db_file_path,
                env,
            )
        except FileNotFoundError:
            table_exists = False
        if not table_exists:
            adb_utils_module.launch_app(app_name, env.controller)
            time_module.sleep(7.0)

        with env.controller.pull_file(remote_db_file_path, timeout_sec) as local_db_directory:
            local_db_path = file_utils_module.convert_to_posix_path(
                local_db_directory, os_module.path.split(remote_db_file_path)[1]
            )
            conn = _prepare_main_db_connection(local_db_path)
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {table_name}")
            conn.commit()
            conn.close()
            _remove_remote_sidecars(remote_db_file_path, env, timeout_sec)
            env.controller.push_file(local_db_path, remote_db_file_path, timeout_sec)
            _remove_remote_sidecars(remote_db_file_path, env, timeout_sec)
            adb_utils_module.close_app(app_name, env.controller)

    def _patched_insert_rows_to_remote_db(
        rows: list[Any],
        exclude_key: str | None,
        table_name: str,
        remote_db_file_path: str,
        app_name: str,
        env: object,
        timeout_sec: float | None = None,
    ) -> None:
        with env.controller.pull_file(remote_db_file_path, timeout_sec) as local_db_directory:
            local_db_path = file_utils_module.convert_to_posix_path(
                local_db_directory, os_module.path.split(remote_db_file_path)[1]
            )
            conn = _prepare_main_db_connection(local_db_path)
            cursor = conn.cursor()
            for row in rows:
                insert_command, values = sqlite_schema_utils_module.insert_into_db(
                    row, table_name, exclude_key
                )
                cursor.execute(insert_command, values)
            conn.commit()
            conn.close()
            _remove_remote_sidecars(remote_db_file_path, env, timeout_sec)
            env.controller.push_file(local_db_path, remote_db_file_path, timeout_sec)
            _remove_remote_sidecars(remote_db_file_path, env, timeout_sec)
            adb_utils_module.close_app(app_name, env.controller)

    sqlite_utils_module.delete_all_rows_from_table = _patched_delete_all_rows_from_table
    sqlite_utils_module.insert_rows_to_remote_db = _patched_insert_rows_to_remote_db
    sqlite_utils_module._omniflow_sqlite_writeback_patch = True
    return {
        "delete_all_rows_from_table": original_delete,
        "insert_rows_to_remote_db": original_insert,
    }


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

    from android_world.agents import t3a

    resolved_name = str(official_agent_name or "").strip() or "t3a_gpt4"
    if resolved_name != "t3a_gpt4":
        raise ValueError(f"Unknown AndroidWorld official agent: {resolved_name}")
    llm = _OpenAICompatibleMultimodalWrapper()
    agent = t3a.T3A(env, llm)
    agent._omniflow_llm_usage_tracker = llm
    agent.name = resolved_name
    return agent


def _run_adb_command(
    *,
    adb_serial: str,
    adb_path: str,
    adb_args: list[str],
    timeout_sec: int = 60,
    capture_stdout: bool = False,
) -> dict[str, Any]:
    command = [
        os.path.expanduser(str(adb_path or "").strip()) or _default_adb_path() or "adb"
    ]
    if adb_serial:
        command.extend(["-s", adb_serial])
    command.extend(adb_args)
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout_sec,
    )
    record = {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": str(completed.stdout or "")[-4000:],
        "stderr_tail": str(completed.stderr or "")[-4000:],
    }
    if capture_stdout:
        record["stdout"] = completed.stdout
    return record


def _read_raw_replay_run_log(path_text: str) -> dict[str, Any]:
    path = Path(str(path_text or "").strip()).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"raw replay run log not found: {path}")
    decoded = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(decoded, dict):
        raise ValueError(f"raw replay run log must be a JSON object: {path}")
    return decoded


def _raw_replay_step_actions(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Project the only accepted RunLog schema to fixed-replay actions."""

    from omniflow.core.trajectory import observation_display
    from src.experiment.source_assets import _identity_at_action_point

    def replay_action(
        step: dict[str, Any],
        action: dict[str, Any],
    ) -> dict[str, Any]:
        tool = str(action["tool"])
        params = dict(action.get("args") or {})
        if tool in {"click", "long_press"}:
            observation = step["observation"]
            display_size = observation_display(observation)
            display = (
                {"width": display_size[0], "height": display_size[1]}
                if display_size is not None
                else {}
            )
            selector = _identity_at_action_point(
                str(observation.get("forest") or ""),
                action_args=params,
                display=display,
            )
            if selector:
                return {"type": tool, "params": {"selector": selector}}
            projected = {"type": tool, "params": params}
            if any(key in params for key in ("x", "y")):
                projected["coordinate_space"] = "canonical_0_1000"
            return projected
        projected = {"type": tool, "params": params}
        if (
            tool in {"input_text", "swipe"}
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
        actions.extend(
            replay_action(step, action)
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


def _fixed_replay_normalize_selector_value(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _fixed_replay_selector_nodes(
    xml_text: str,
    selector: dict[str, Any],
) -> tuple[list[ET.Element], str | None]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return [], "selector_target_xml_invalid"
    return _fixed_replay_selector_nodes_from_root(root, selector)


def _fixed_replay_selector_nodes_from_root(
    root: ET.Element,
    selector: dict[str, Any],
) -> tuple[list[ET.Element], str | None]:
    nodes = list(root.iter())
    relation = str(selector.get("relation") or "").strip()
    if relation == "unique_actionable_descendant":
        anchor = selector.get("container_anchor")
        if not isinstance(anchor, dict) or not anchor:
            return [], "selector_container_anchor_missing"
        anchor_nodes, anchor_error = _fixed_replay_selector_nodes_from_root(
            root,
            anchor,
        )
        if anchor_error is not None:
            return [], anchor_error
        if len(anchor_nodes) != 1:
            return [], "selector_container_anchor_ambiguous"
        parents = {child: parent for parent in root.iter() for child in list(parent)}
        container = parents.get(anchor_nodes[0])
        if container is None:
            return [], "selector_container_missing"
        actionable = [
            node
            for node in container.iter()
            if node is not anchor_nodes[0]
            and any(
                str(node.attrib.get(key) or "").lower() == "true"
                for key in ("clickable", "editable", "long-clickable")
            )
        ]
        return actionable, None
    if str(selector.get("role") or "").strip() == "editable":
        editable = [
            node
            for node in nodes
            if str(node.attrib.get("editable") or "").lower() == "true"
            or str(node.attrib.get("class") or "") == "android.widget.EditText"
        ]
        return editable, None
    aliases = {
        "resource_id": "resource-id",
        "text": "text",
        "content_desc": "content-desc",
    }
    expected = {
        attribute: _fixed_replay_normalize_selector_value(selector.get(key))
        for key, attribute in aliases.items()
        if _fixed_replay_normalize_selector_value(selector.get(key))
    }
    if not expected:
        return [], "selector_identity_missing"
    resource_id = expected.get("resource-id")
    if resource_id:
        resource_matches = [
            node
            for node in nodes
            if _fixed_replay_normalize_selector_value(
                node.attrib.get("resource-id")
            )
            == resource_id
        ]
        if len(resource_matches) == 1:
            return resource_matches, None
        if resource_matches:
            nodes = resource_matches
    matches = [
        node
        for node in nodes
        if all(
            _fixed_replay_normalize_selector_value(node.attrib.get(attribute))
            == value
            for attribute, value in expected.items()
        )
    ]
    return matches, None


def _fixed_replay_selector_center(
    xml_text: str,
    selector: dict[str, Any],
) -> tuple[tuple[int, int] | None, str | None]:
    matches, error = _fixed_replay_selector_nodes(xml_text, selector)
    if error is not None:
        return None, error
    if not matches:
        return None, "selector_target_not_found"
    if len(matches) != 1:
        return None, "selector_target_ambiguous"
    bounds = str(matches[0].attrib.get("bounds") or "").strip()
    match = re.fullmatch(
        r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]",
        bounds,
    )
    if match is None:
        return None, "selector_target_bounds_missing"
    left, top, right, bottom = (int(value) for value in match.groups())
    if right <= left or bottom <= top:
        return None, "selector_target_bounds_invalid"
    return ((left + right) // 2, (top + bottom) // 2), None


def _raw_replay_action_to_payload(
    source_action: dict[str, Any],
    *,
    source_size: tuple[int, int] | None,
    target_size: tuple[int, int],
    target_xml: str = "",
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
        selector = (
            dict(params.get("selector") or {})
            if isinstance(params.get("selector"), dict)
            else {}
        )
        if selector:
            center, selector_error = _fixed_replay_selector_center(
                target_xml,
                selector,
            )
            if selector_error is not None:
                return None, selector_error
            assert center is not None
            x, y = center
        else:
            x, y = _scaled_xy()
        if x is None or y is None:
            return None, "missing_selector_and_coordinates"
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


def _launch_raw_replay_app(app_identifier: str, env: Any) -> None:
    from android_world.env import adb_utils

    identifier = str(app_identifier or "").strip()
    if not identifier:
        raise ValueError("raw_replay_open_app_identifier_required")
    if "." not in identifier:
        adb_utils.launch_app(identifier, env.controller)
        return
    result = adb_utils.issue_generic_request(
        [
            "shell",
            "monkey",
            "-p",
            identifier,
            "-c",
            "android.intent.category.LAUNCHER",
            "1",
        ],
        env.controller,
    )
    adb_utils.check_ok(result, f"Failed to launch Android package {identifier}.")


def _apply_fixed_replay(
    agent: Any,
    *,
    run_log_json_path: str,
    adb_path: str = "",
) -> Any:
    """Replay fixed source actions through selector-first AndroidWorld parameters."""

    original_set_max_steps = getattr(agent, "set_max_steps", None)
    run_log_data = _read_raw_replay_run_log(run_log_json_path)
    source_actions = _raw_replay_step_actions(run_log_data)
    source_size = _raw_replay_source_size(run_log_data)
    state: dict[str, Any] = {"ran": False, "payload": None}
    replay_host = getattr(agent, "host", None)
    replay_observe = getattr(replay_host, "observe", None)
    use_oob_observe = _use_oob_observe_backend()
    capture_native_observations = str(
        os.environ.get("OMNIFLOW_RAW_REPLAY_CAPTURE_OBSERVATIONS") or ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    capture_observations = use_oob_observe or capture_native_observations
    requires_selector_observations = any(
        isinstance(action.get("params"), dict)
        and isinstance(action["params"].get("selector"), dict)
        and bool(action["params"]["selector"])
        for action in source_actions
    )
    if (
        capture_observations or requires_selector_observations
    ) and not callable(replay_observe):
        raise RuntimeError("fixed replay selector resolution requires host.observe")

    def _forced_reset(go_home: bool = False) -> None:
        state["ran"] = False
        state["payload"] = None
        if go_home:
            from android_world.env import adb_utils

            adb_utils.issue_generic_request(
                ["shell", "input", "keyevent", "KEYCODE_HOME"],
                agent.env.controller,
            )

    def _forced_set_max_steps(step_budget: int) -> None:
        del step_budget
        if callable(original_set_max_steps):
            original_set_max_steps(max(1, len(source_actions) + 1))

    def _execute_payload(
        payload: dict[str, Any],
        *,
        target_size: tuple[int, int],
    ) -> None:
        from android_world.env import actuation, adb_utils, json_action

        if payload.get("action_type") == "raw_open_app":
            _launch_raw_replay_app(str(payload["app_identifier"]), agent.env)
            return
        if payload.get("action_type") == "raw_wait":
            time.sleep(float(payload.get("seconds") or 0.0))
            return
        if payload.get("action_type") == "raw_swipe":
            _execute_raw_replay_host_swipe(
                agent,
                payload,
                target_size=target_size,
            )
            return
        if payload.get("action_type") == "raw_set_clipboard":
            adb_utils.set_clipboard_contents(
                str(payload.get("text") or ""), agent.env.controller
            )
            return
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
        if use_oob_observe:
            actuation.execute_adb_action(
                action,
                [],
                tuple(agent.env.logical_screen_size),
                agent.env.controller,
            )
        else:
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
                    "source": "selector_then_scaled_coordinate_replay",
                    "actions_executed": int(summary.get("actions_executed") or 0),
                    "fallback": False,
                    "error": None,
                    "done_reason": "fixed_replay_already_completed",
                },
            )

        started = perf_counter()
        target_size = tuple(getattr(agent.env, "logical_screen_size", (0, 0)) or (0, 0))
        if len(target_size) != 2:
            target_size = (0, 0)
        if not target_size[0] or not target_size[1]:
            target_size = source_size or (1080, 2400)
        oob_prepare: dict[str, Any] | None = None
        if use_oob_observe:
            oob_prepare = _prepare_oob_device_host_for_replay(
                adb_serial=str(os.environ.get("ANDROID_SERIAL") or "").strip(),
                adb_path=adb_path,
            )
            if not bool(oob_prepare.get("success")):
                raise RuntimeError(
                    "raw replay OOB preparation failed: "
                    + str(oob_prepare.get("error") or oob_prepare)
                )
        step_results: list[dict[str, Any]] = []
        completed = True
        error_text: str | None = None
        actions_executed = 0
        selector_actions = 0
        scaled_coordinate_actions = 0
        for index, source_action in enumerate(source_actions):
            observation_record: dict[str, Any] | None = None
            source_params = (
                dict(source_action.get("params") or {})
                if isinstance(source_action.get("params"), dict)
                else {}
            )
            needs_selector_observation = bool(
                isinstance(source_params.get("selector"), dict)
                and source_params["selector"]
            )
            uses_scaled_coordinates = bool(
                not needs_selector_observation
                and any(
                    key in source_params
                    for key in ("x", "y", "x1", "y1", "x2", "y2")
                )
            )
            if capture_observations or needs_selector_observation:
                observation_record = _raw_replay_observation_record(
                    replay_observe(xml=True, screenshot=False, app_info=True),
                    fallback_size=(int(target_size[0]), int(target_size[1])),
                )
            payload, skip_reason = _raw_replay_action_to_payload(
                source_action,
                source_size=source_size,
                target_size=(int(target_size[0]), int(target_size[1])),
                target_xml=str((observation_record or {}).get("xml") or ""),
            )
            step_record: dict[str, Any] = {
                "index": index,
                "source_action": _sanitize_raw_replay_source_action(source_action),
                "androidworld_action": dict(payload or {}),
                "source_screen_size": list(source_size) if source_size else None,
                "target_screen_size": [int(target_size[0]), int(target_size[1])],
                "completed": False,
                "skipped": False,
                "parameter_source": (
                    "selector"
                    if needs_selector_observation
                    else "scaled_coordinate_fallback"
                    if uses_scaled_coordinates
                    else "direct_androidworld_action"
                ),
            }
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
                    target_size=(int(target_size[0]), int(target_size[1])),
                )
                actions_executed += 1
                if needs_selector_observation:
                    selector_actions += 1
                elif uses_scaled_coordinates:
                    scaled_coordinate_actions += 1
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
            "execution_backend": "selector_then_scaled_coordinate_replay",
            "steps": int(actions_executed),
            "actions_executed": int(actions_executed),
            "selector_actions": int(selector_actions),
            "scaled_coordinate_actions": int(scaled_coordinate_actions),
            "model_calls": 0,
            "tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "elapsed_ms": elapsed_ms,
        }
        if error_text:
            execution_summary["failure_reason"] = error_text
        run_log = {
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
            "steps": [
                {
                    "step_index": 0,
                    "selection_source": "fixed_replay",
                    "execution_source": "selector_then_scaled_coordinate_replay",
                    "provider_detail": {
                        "raw_replay": {
                            "source_run_log": str(run_log_json_path),
                            "source_action_count": len(source_actions),
                            "actions_executed": int(actions_executed),
                            "selector_actions": int(selector_actions),
                            "scaled_coordinate_actions": int(
                                scaled_coordinate_actions
                            ),
                            "source_screen_size": list(source_size)
                            if source_size
                            else None,
                            "target_screen_size": [
                                int(target_size[0]),
                                int(target_size[1]),
                            ],
                            "oob_prepare": {
                                "success": bool(oob_prepare.get("success")),
                                "accessibility_bound": bool(
                                    oob_prepare.get("accessibility_bound")
                                ),
                            }
                            if oob_prepare is not None
                            else None,
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
            "run_log": run_log,
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
                                "duration_ms": elapsed_ms,
                                "run_log": run_log,
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
                "source": "selector_then_scaled_coordinate_replay",
                "actions_executed": int(actions_executed),
                "fallback": False,
                "error": error_text,
                "done_reason": done_reason,
            },
        )

    def _forced_save_run_log(
        success: bool = False,
        done_reason: str = "",
    ) -> dict | None:
        del success, done_reason
        payload = state.get("payload")
        return dict(payload) if isinstance(payload, dict) else None

    agent.reset = _forced_reset
    agent.set_max_steps = _forced_set_max_steps
    agent.step = _forced_step
    agent.save_run_log = _forced_save_run_log
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
    planner_timeout_sec: float | None = None,
    raw_replay_run_log: str = "",
    appagent_root: str = "",
    appagent_workspace_root: str = "",
    appagent_docs_root: str = "",
    appagent_action_source: str = "",
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
    if resolved_agent in {MODE_OMNIFLOW, "fixed_replay"}:
        planner = None
        function_router = None
        resolved_planner_model = str(
            planner_model or os.environ.get("OMNIFLOW_PLANNER_MODEL") or ""
        ).strip()
        resolved_planner_provider = str(
            planner_provider or os.environ.get("OMNIFLOW_PLANNER_PROVIDER") or ""
        ).strip()
        resolved_planner_timeout = float(
            planner_timeout_sec
            or os.environ.get("OMNIFLOW_PLANNER_TIMEOUT_SEC")
            or 60.0
        )
        if resolved_planner_model or resolved_planner_provider or read_env_bool(
            "OMNIFLOW_ENABLE_ONLINE_PLANNER",
            False,
        ):
            from omniflow.vlm.planner import VLMPlanner

            planner = VLMPlanner(
                provider=resolved_planner_provider or None,
                model=resolved_planner_model or None,
                timeout=resolved_planner_timeout,
            )
            if resolved_agent == MODE_OMNIFLOW:
                from omniflow.vlm.function_router import VLMFunctionRouter

                function_router = VLMFunctionRouter(
                    provider=resolved_planner_provider or "openai",
                    model=resolved_planner_model,
                    timeout=resolved_planner_timeout,
                )
        build_kwargs: dict[str, Any] = {
            "env": env,
            "store_path": store_path,
            "adb_serial": adb_serial,
            "adb_path": adb_path,
            "task_seed": task_seed,
            "evidence_root": evidence_root or None,
        }
        if planner is not None:
            build_kwargs["planner"] = planner
        if function_router is not None:
            build_kwargs["function_router"] = function_router
        built_agent = build_agent(**build_kwargs)
        if resolved_agent == "fixed_replay":
            run_log_json_path = str(raw_replay_run_log or "").strip()
            if not run_log_json_path:
                raise ValueError("fixed_replay requires --raw-replay-run-log")
            return _apply_fixed_replay(
                built_agent,
                run_log_json_path=run_log_json_path,
                adb_path=adb_path,
            )
        return built_agent
    if resolved_agent == "external:mobilegpt":
        from src.integrations.android_world.mobilegpt_agent import build_mobilegpt_agent

        return build_mobilegpt_agent(
            env=env,
            evidence_root=evidence_root or None,
        )
    if resolved_agent in {"external:appagent", "external:appagent_teacher"}:
        from src.integrations.appagent_adapter import (
            AppAgentAndroidWorldAgent,
            AppAgentTeacherAgent,
            OfficialAppAgentRuntime,
        )

        runtime = OfficialAppAgentRuntime(appagent_root)
        if resolved_agent == "external:appagent_teacher":
            if not str(appagent_teacher_source or "").strip():
                raise ValueError(
                    "external:appagent_teacher requires --appagent-teacher-source"
                )
            if not str(appagent_workspace_root or "").strip():
                raise ValueError(
                    "external:appagent_teacher requires --appagent-workspace-root"
                )
            return AppAgentTeacherAgent(
                env=env,
                official_runtime=runtime,
                teacher_source=appagent_teacher_source,
                workspace_root=appagent_workspace_root,
                demo_name=appagent_demo_name,
            )
        llm = _OpenAICompatibleMultimodalWrapper()
        return AppAgentAndroidWorldAgent(
            env=env,
            official_runtime=runtime,
            llm=llm,
            output_root=appagent_output_root,
            docs_root=(appagent_docs_root or None),
            action_source=(appagent_action_source or None),
        )
    if resolved_agent.startswith("official:"):
        official_agent_name = str(
            resolved_agent.split(":", maxsplit=1)[1] or ""
        ).strip()
        if not official_agent_name:
            raise ValueError(
                "--agent official:<name> requires one upstream AndroidWorld agent name"
            )
        return _build_official_androidworld_agent(
            env=env,
            official_agent_name=official_agent_name,
        )
    raise ValueError(
        "Unsupported AndroidWorld agent selector. Use `omniflow`, `fixed_replay`, "
        "`external:mobilegpt`, `external:appagent`, "
        "`external:appagent_teacher`, or `official:<name>`."
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
        "--agent",
        default=MODE_OMNIFLOW,
        help=(
            "Agent selector. `omniflow` keeps the shared cache-first adapter; "
            "`external:mobilegpt` delegates one official episode to MobileGPT; "
            "`external:appagent` runs pinned AppAgent deployment; "
            "`external:appagent_teacher` captures one source human demo; "
            "`official:t3a_gpt4` runs the paper's upstream T3A agent."
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
    parser.add_argument("--appagent-action-source", default="")
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
        "--planner-timeout-sec",
        type=float,
        default=float(os.environ.get("OMNIFLOW_PLANNER_TIMEOUT_SEC") or 60.0),
        help="Per-call timeout in seconds for the online OmniFlow planner.",
    )
    return parser


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
    native_appagent = selected_agent in {
        "external:appagent",
        "external:appagent_teacher",
    }
    if str(args.planner_provider or "").strip():
        os.environ["OMNIFLOW_PLANNER_PROVIDER"] = str(args.planner_provider).strip()
    if str(args.model or "").strip():
        os.environ["OMNIFLOW_PLANNER_MODEL"] = str(args.model).strip()
    if float(args.planner_timeout_sec or 0) > 0:
        os.environ["OMNIFLOW_PLANNER_TIMEOUT_SEC"] = str(
            float(args.planner_timeout_sec)
        )
    os.environ.pop("OMNIFLOW_OOB_DEVICE_URL", None)
    os.environ["OMNIFLOW_OBSERVE_BACKEND"] = "androidworld"
    os.environ["OMNIFLOW_ACT_BACKEND"] = "androidworld"
    android_world_root = Path(args.android_world_root).expanduser().resolve()
    run_py = android_world_root / "run.py"
    if not run_py.exists():
        raise FileNotFoundError(f"run.py not found under {android_world_root}")
    task_params = _decode_task_params(
        args.task_params_json,
        task_random_seed=int(args.task_random_seed),
    )

    env = None
    original_allocate_step_budget = None
    original_run_task = None
    original_get_controller = None
    original_clear_directory = None
    original_issue_generic_request = None
    original_restore_snapshot = None
    original_sqlite_writeback = None
    try:
        _maybe_patch_sqlite_backend()
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
        from android_world.agents import m3a_utils
        from android_world.env import android_world_controller, env_launcher
        from android_world.env import tools as android_world_tools
        from android_world.utils import datetime_utils
        try:
            from android_world.env.setup_device import setup as aw_setup
        except ImportError as exc:
            if "android_world.env.setup_device" not in str(exc):
                raise
            aw_setup = None

        patch_androidworld_setup_click_retry(android_world_tools)
        if aw_setup is not None:
            patch_androidworld_setup_fail_closed(aw_setup)

        try:
            from android_world.env import adb_utils
        except ImportError:
            adb_utils = None
        if adb_utils is not None:
            original_issue_generic_request = _patch_androidworld_settings_get_output(
                adb_utils
            )
        try:
            file_utils = importlib.import_module("android_world.utils.file_utils")
        except ImportError as exc:
            if "android_world.utils.file_utils" not in str(exc):
                raise
            file_utils = None
        if file_utils is not None:
            original_clear_directory = _patch_androidworld_empty_clear_directory(
                file_utils
            )
        if file_utils is not None and adb_utils is not None:
            file_validators = importlib.import_module(
                "android_world.task_evals.common_validators.file_validators"
            )
            _patch_androidworld_create_file_diagnostics(
                file_validators=file_validators,
                file_utils=file_utils,
                adb_utils=adb_utils,
            )
            sqlite_utils = importlib.import_module(
                "android_world.task_evals.utils.sqlite_utils"
            )
            original_sqlite_writeback = _patch_androidworld_sqlite_writeback(
                sqlite_utils
            )
        try:
            app_snapshot = importlib.import_module("android_world.utils.app_snapshot")
        except ImportError as exc:
            if "android_world.utils.app_snapshot" not in str(exc):
                raise
            app_snapshot = None
        if app_snapshot is not None and adb_utils is not None and not native_appagent:
            original_restore_snapshot = getattr(app_snapshot, "restore_snapshot", None)
            if callable(original_restore_snapshot):

                def _restore_snapshot_or_clear_app_data(app_name: str, env: object) -> None:
                    """Reset app state even when AndroidWorld has no saved snapshot.

                    AndroidWorld's default behavior logs a missing snapshot and
                    continues with whatever app DB happens to be on the device.
                    That makes validator runs depend on prior tasks. Every
                    normal episode needs the same reset contract, so a missing
                    snapshot falls back to `pm clear` for the app under test.
                    """

                    try:
                        original_restore_snapshot(app_name, env)
                        return
                    except RuntimeError as exc:
                        if "Snapshot not found" not in str(exc):
                            raise
                        try:
                            activity = adb_utils.get_adb_activity(app_name)
                            package_name = adb_utils.extract_package_name(activity)
                            result = adb_utils.issue_generic_request(
                                ["shell", "pm", "clear", package_name],
                                env,
                            )
                            check_ok = getattr(adb_utils, "check_ok", None)
                            if callable(check_ok):
                                check_ok(
                                    result,
                                    f"Failed to clear app data for {package_name}.",
                                )
                            logging.warning(
                                "AndroidWorld snapshot missing for %s; cleared app data for %s.",
                                app_name,
                                package_name,
                            )
                        except Exception:
                            logging.warning(
                                "AndroidWorld snapshot missing for %s and app-data reset failed.",
                                app_name,
                                exc_info=True,
                            )
                            raise exc

                app_snapshot.restore_snapshot = _restore_snapshot_or_clear_app_data

        original_get_controller = getattr(
            android_world_controller, "get_controller", None
        )
        if callable(original_get_controller) and not native_appagent:

            def _get_controller_without_reinstall(
                console_port: int = 5554,
                adb_path: str = android_world_controller.DEFAULT_ADB_PATH,
                grpc_port: int = 8554,
                install_a11y_forwarding_app: bool = True,
            ):
                """Construct one AndroidWorld controller with explicit A11Y setup.

                Args:
                    console_port: Emulator console port used by AndroidEnv.
                    adb_path: Absolute or shell-resolved adb binary path.
                    grpc_port: Emulator gRPC port paired with the console port.
                    install_a11y_forwarding_app: Upstream compatibility flag.
                        Formal episodes honor the upstream installation request
                        so the live accessibility tree is available to validators.

                Returns:
                    An `AndroidWorldController` that keeps the upstream a11y
                    forwarder method but skips the re-install step, which can
                    stall emulator initialization on already-provisioned devices.
                """

                effective_adb_path = adb_path
                target_serial = str(os.environ.get("ANDROID_SERIAL") or "").strip()
                if target_serial and not target_serial.startswith("emulator-"):
                    real_adb_path = str(adb_path or _default_adb_path() or "adb")
                    wrapper_path = (
                        Path(os.environ.get("TMPDIR") or "/tmp")
                        / f"omniflow_adb_{os.getpid()}_{target_serial}.sh"
                    )
                    wrapper_path.write_text(
                        "#!/usr/bin/env bash\n"
                        "set -e\n"
                        f"REAL_ADB={json.dumps(real_adb_path)}\n"
                        f"TARGET_SERIAL={json.dumps(target_serial)}\n"
                        "args=()\n"
                        "while [[ $# -gt 0 ]]; do\n"
                        "  if [[ \"$1\" == \"-s\" && $# -gt 1 ]]; then\n"
                        "    shift\n"
                        "    if [[ \"$1\" == emulator-* ]]; then\n"
                        "      args+=(\"-s\" \"$TARGET_SERIAL\")\n"
                        "    else\n"
                        "      args+=(\"-s\" \"$1\")\n"
                        "    fi\n"
                        "  else\n"
                        "    args+=(\"$1\")\n"
                        "  fi\n"
                        "  shift\n"
                        "done\n"
                        "exec \"$REAL_ADB\" \"${args[@]}\"\n",
                        encoding="utf-8",
                    )
                    wrapper_path.chmod(0o755)
                    effective_adb_path = str(wrapper_path)
                    logging.info(
                        "Routing AndroidWorld adb commands to real device %s via %s",
                        target_serial,
                        effective_adb_path,
                    )

                config = android_world_controller.config_classes.AndroidEnvConfig(
                    task=android_world_controller.config_classes.FilesystemTaskConfig(
                        path=android_world_controller._write_default_task_proto()
                    ),
                    simulator=android_world_controller.config_classes.EmulatorConfig(
                        emulator_launcher=android_world_controller.config_classes.EmulatorLauncherConfig(
                            emulator_console_port=console_port,
                            adb_port=console_port + 1,
                            grpc_port=grpc_port,
                        ),
                        adb_controller=android_world_controller.config_classes.AdbControllerConfig(
                            adb_path=effective_adb_path
                        ),
                    ),
                )
                android_env_instance = android_world_controller.loader.load(config)
                logging.info("Setting up AndroidWorldController.")
                controller_kwargs: dict[str, object] = {
                    "install_a11y_forwarding_app": bool(
                        install_a11y_forwarding_app
                    ),
                    "a11y_method": _native_androidworld_a11y_method(
                        android_world_controller
                    ),
                }
                return android_world_controller.AndroidWorldController(
                    android_env_instance,
                    **controller_kwargs,
                )

            android_world_controller.get_controller = _get_controller_without_reinstall

        original_allocate_step_budget = getattr(
            suite_utils, "_allocate_step_budget", None
        )
        fixed_max_steps = max(1, int(args.max_steps))
        if callable(original_allocate_step_budget):
            suite_utils._allocate_step_budget = lambda task_complexity: fixed_max_steps

        original_set_datetime = datetime_utils.set_datetime
        original_parse_reason_action_output = m3a_utils.parse_reason_action_output
        enable_set_datetime = str(
            read_env_text("OMNIFLOW_ANDROIDWORLD_SET_DATETIME") or "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        set_datetime_skip_logged = False

        def _safe_set_datetime(controller: object, dt: object) -> None:
            """Best-effort device time freeze for hosts that cannot set system time.

            Args:
                controller: AndroidWorld device controller.
                dt: Target datetime requested by the task initializer.
            """

            nonlocal set_datetime_skip_logged
            if not enable_set_datetime:
                if not set_datetime_skip_logged:
                    logger.info(
                        "skip AndroidWorld set_datetime in unified test shell because OMNIFLOW_ANDROIDWORLD_SET_DATETIME is disabled."
                    )
                    set_datetime_skip_logged = True
                return
            try:
                original_set_datetime(controller, dt)
            except Exception as exc:  # noqa: BLE001
                message = str(exc or "")
                if "Operation not permitted" in message or "cannot set date" in message:
                    logger.warning(
                        "skip AndroidWorld set_datetime due to host/device permission: %s",
                        message,
                    )
                    return
                raise

        datetime_utils.set_datetime = _safe_set_datetime

        def _safe_parse_reason_action_output(
            raw_reason_action_output: object,
        ) -> tuple[str | None, str | None]:
            """Keep official AndroidWorld agents from crashing on empty LLM actions.

            Args:
                raw_reason_action_output: Raw action-selection output returned by
                    the upstream multimodal model wrapper. Some upstream paths
                    occasionally hand back `None` or a non-string object.

            Returns:
                The usual `(reason, action_json)` tuple. Empty / non-string
                outputs are converted into one explicit infeasible status so the
                official baseline exits cleanly instead of throwing.
            """

            if isinstance(raw_reason_action_output, str):
                return original_parse_reason_action_output(raw_reason_action_output)
            logger.warning(
                "AndroidWorld official agent produced non-string action output; coerce to explicit infeasible status: %r",
                raw_reason_action_output,
            )
            return (
                "Upstream AndroidWorld agent returned empty or non-string action output.",
                '{"action_type": "status", "goal_status": "infeasible"}',
            )

        m3a_utils.parse_reason_action_output = _safe_parse_reason_action_output
        task_registry = registry.TaskRegistry()
        selected_task_names = [
            item.strip() for item in str(args.tasks).split(",") if item.strip()
        ]
        task_types = task_registry.get_registry(family=args.suite_family)
        setup_app_list = None
        if bool(args.perform_emulator_setup):
            if aw_setup is None:
                raise RuntimeError(
                    "AndroidWorld setup_device module is required when "
                    "--perform-emulator-setup is set."
                )
            setup_apps: list[type] = []
            seen_setup_apps: set[type] = set()
            for selected_task_name in selected_task_names:
                task_type = task_types.get(selected_task_name)
                for app_name in tuple(getattr(task_type, "app_names", ()) or ()):
                    app_setup = aw_setup.get_app_mapping(str(app_name))
                    if app_setup is not None and app_setup not in seen_setup_apps:
                        setup_apps.append(app_setup)
                        seen_setup_apps.add(app_setup)
            for app_setup in aw_setup.get_app_list_to_setup(selected_task_names) or ():
                if app_setup not in seen_setup_apps:
                    setup_apps.append(app_setup)
                    seen_setup_apps.add(app_setup)
            setup_app_list = tuple(setup_apps) if setup_apps else None

        target_adb_serial = str(
            os.environ.get("ANDROID_SERIAL") or f"emulator-{int(args.console_port)}"
        ).strip()
        _patch_androidworld_ui_debug_settings(android_world_controller)
        env = env_launcher.load_and_setup_env(
            console_port=int(args.console_port),
            emulator_setup=False,
            adb_path=str(args.adb_path or ""),
            grpc_port=int(args.console_port) + 3000,
        )
        if _use_oob_observe_backend():
            oob_prepare = _prepare_oob_device_host_for_replay(
                adb_serial=target_adb_serial,
                adb_path=str(args.adb_path or ""),
            )
            if not bool(oob_prepare.get("success")):
                raise RuntimeError(
                    "OOB get_state not ready before AndroidWorld setup: "
                    + str(oob_prepare.get("error") or oob_prepare)
                )
        if bool(args.perform_emulator_setup):
            logger.info(
                "Setting up AndroidWorld snapshots for selected tasks: %s",
                ", ".join(selected_task_names) or "<all>",
            )
            aw_setup.setup_apps(env, app_list=setup_app_list)
        if not _use_oob_observe_backend() and not native_appagent:
            a11y_runtime = _prepare_native_androidworld_a11y_runtime(
                env,
                adb_serial=target_adb_serial,
                adb_path=str(args.adb_path or ""),
            )
            logger.info("Native AndroidWorld A11Y runtime ready: %s", a11y_runtime)

        suite = suite_utils.create_suite(
            task_types,
            n_task_combinations=int(args.n_task_combinations),
            seed=int(args.task_random_seed),
            tasks=selected_task_names,
            use_identical_params=bool(args.fixed_task_seed),
        )
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

        agent = _build_launch_agent(
            agent=str(args.agent or MODE_OMNIFLOW),
            env=env,
            store_path=str(args.store_path or ""),
            adb_serial=str(
                os.environ.get("ANDROID_SERIAL") or f"emulator-{int(args.console_port)}"
            ).strip(),
            adb_path=str(args.adb_path or ""),
            raw_replay_run_log=str(args.raw_replay_run_log or ""),
            planner_provider=str(args.planner_provider or ""),
            planner_model=str(args.model or ""),
            planner_timeout_sec=float(args.planner_timeout_sec or 60.0),
            appagent_root=str(args.appagent_root or ""),
            appagent_workspace_root=str(args.appagent_workspace_root or ""),
            appagent_docs_root=str(args.appagent_docs_root or ""),
            appagent_action_source=str(args.appagent_action_source or ""),
            appagent_teacher_source=str(args.appagent_teacher_source or ""),
            appagent_demo_name=str(args.appagent_demo_name or ""),
            appagent_output_root=str(run_output_dir / "appagent_runtime"),
            task_seed=int(args.task_random_seed),
            evidence_root=str(run_output_dir),
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
        original_run_task = getattr(suite_utils, "_run_task", None)
        if callable(original_run_task):
            official_goal_hint_text = ""
            official_goal_hint_meta: dict[str, Any] | None = None
            if selected_agent.startswith("official:"):
                official_goal_hint_text, official_goal_hint_meta = (
                    _load_official_agent_goal_hint(args.source_action_hint_path)
                )

            def _wrapped_run_task(task, run_episode, env, demo_mode):
                task_name = str(getattr(task, "name", "") or "human_task")
                result: dict[str, Any] | None = None
                goal_text = str(getattr(task, "goal", "") or getattr(task, "name", ""))
                task_context: dict[str, Any] = {}
                started_at = utc_now_iso()
                started_perf = perf_counter()
                original_get_state = getattr(env, "get_state", None)
                observation_archive = (
                    ObservationArchive(original_get_state)
                    if callable(original_get_state)
                    else None
                )
                observation_archive_error = (
                    None
                    if observation_archive is not None
                    else "environment_get_state_unavailable"
                )
                if observation_archive is not None:
                    try:
                        env.get_state = observation_archive.get_state
                    except Exception as exc:  # noqa: BLE001
                        observation_archive = None
                        observation_archive_error = (
                            f"observation_archive_install_failed:{exc}"
                        )
                official_llm_usage_before = (
                    _get_agent_llm_usage(agent)
                    if selected_agent.startswith("official:")
                    or selected_agent == "external:appagent"
                    else {}
                )
                try:
                    set_current_task = getattr(agent, "set_current_task", None)
                    if callable(set_current_task):
                        if native_appagent:
                            update_task_context = getattr(
                                agent,
                                "update_current_task_context",
                                None,
                            )
                            if callable(update_task_context):
                                task_context = dict(update_task_context(task) or {})
                            set_current_task(task_name, goal_text, task_context)
                        else:
                            set_current_task(task_name, goal_text)
                    def _update_context_after_initialize(initialized_task):
                        update_task_context = getattr(
                            agent,
                            "update_current_task_context",
                            None,
                        )
                        if not callable(update_task_context):
                            return
                        try:
                            nonlocal task_context
                            task_context = dict(
                                update_task_context(initialized_task) or {}
                            )
                            reset_current_task = getattr(
                                agent,
                                "set_current_task",
                                None,
                            )
                            if callable(reset_current_task):
                                reset_current_task(
                                    task_name,
                                    goal_text,
                                    task_context,
                                )
                        except Exception as exc:  # noqa: BLE001
                            logging.warning(
                                "Failed to update AndroidWorld task context: %s",
                                exc,
                            )

                    if not native_appagent:
                        task_adb_serial = str(
                            os.environ.get("ANDROID_SERIAL")
                            or f"emulator-{int(args.console_port)}"
                        ).strip()
                        _wrap_task_initialize_for_observation_runtime(
                            task,
                            agent=agent,
                            adb_serial=task_adb_serial,
                            adb_path=str(args.adb_path or ""),
                            oob_url=str(
                                os.environ.get("OMNIFLOW_OOB_DEVICE_URL") or ""
                            ).strip().rstrip("/"),
                            console_port=int(args.console_port),
                            restore_app_snapshot=original_restore_snapshot,
                            after_initialized=_update_context_after_initialize,
                        )
                    reference_text = official_goal_hint_text
                    if reference_text:
                        hinted_goal = f"{goal_text}\n\n{reference_text}"

                        def _run_episode_with_goal_hint(episode_task):
                            return run_episode(_TaskGoalProxy(episode_task, hinted_goal))

                        result = original_run_task(
                            task,
                            _run_episode_with_goal_hint,
                            env,
                            demo_mode,
                        )
                    else:
                        result = original_run_task(task, run_episode, env, demo_mode)
                    return result
                finally:
                    try:
                        canonical_run = None
                        canonical_run_id = None
                        run_log_payload: dict[str, Any] | None = None
                        observation_evidence: list[dict[str, Any]] | None = None
                        if observation_archive is not None:
                            try:
                                observation_evidence = observation_archive.persist(
                                    run_output_dir
                                )
                            except (OSError, TypeError, ValueError) as exc:
                                observation_archive_error = str(exc)
                        save_run_log = getattr(agent, "save_run_log", None)
                        if selected_agent == MODE_OMNIFLOW and callable(save_run_log):
                            official_success = bool(
                                isinstance(result, dict)
                                and float(result.get("is_successful") or 0.0) > 0.5
                            )
                            try:
                                payload = save_run_log(
                                    success=official_success,
                                )
                            except TypeError as exc:
                                if "unexpected keyword argument" not in str(exc):
                                    raise
                                payload = save_run_log()
                            if isinstance(payload, dict):
                                run_log_payload = payload
                                canonical_run_id = (
                                    str(payload.get("run_id") or "").strip() or None
                                )
                                run_log = payload.get("run_log")
                                if isinstance(run_log, dict):
                                    canonical_run = dict(run_log)
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
                                model_base_url = str(
                                    os.environ.get("OPENAI_BASE_URL") or ""
                                ).strip() or None
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
                            "official_validator_used": True,
                            "androidworld_validator_result": {
                                "success": task_success,
                                "reward": validator_reward,
                                "error": error_text,
                                "uses_androidworld_official_validator": True,
                                "validator": "androidworld_official",
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
                            if selected_agent == MODE_OMNIFLOW:
                                if not isinstance(run_log_payload, dict):
                                    raise RuntimeError(
                                        "target_run_evidence_payload_missing"
                                    )
                                captured_transfer_states = run_log_payload.get(
                                    "captured_transfer_states"
                                )
                                transfer_state_audit = run_log_payload.get(
                                    "transfer_state_audit"
                                )
                                if not isinstance(captured_transfer_states, dict):
                                    raise RuntimeError(
                                        "target_transfer_states_missing"
                                    )
                                if not isinstance(transfer_state_audit, dict):
                                    raise RuntimeError(
                                        "target_transfer_state_audit_missing"
                                    )
                                task_result_record.update(
                                    persist_target_run_evidence(
                                        run_output_dir,
                                        run_log=canonical_run,
                                        captured_transfer_states=(
                                            captured_transfer_states
                                        ),
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
                        elif observation_archive_error:
                            task_result_record["observation_evidence_error"] = (
                                observation_archive_error
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
                        if callable(original_get_state):
                            try:
                                env.get_state = original_get_state
                            except Exception as exc:  # noqa: BLE001
                                print(
                                    "[warn] failed to restore AndroidWorld get_state "
                                    f"for {task_name}: {exc}"
                                )

            suite_utils._run_task = _wrapped_run_task
        mainline_name = str(args.agent or MODE_OMNIFLOW).strip() or MODE_OMNIFLOW
        print(
            "Starting AndroidWorld test shell with "
            f"agent={mainline_name} max_steps={fixed_max_steps} "
            f"and writing to {checkpoint_dir}"
        )
        suite_utils.run(
            suite,
            agent,
            checkpointer=checkpointer,
            demo_mode=False,
        )
        print(
            f"Finished AndroidWorld test shell agent={mainline_name} "
            f"max_steps={fixed_max_steps} on {args.suite_family} family. Wrote to {checkpoint_dir}."
        )
        try:
            _write_task_results_summary(
                task_results_path=task_results_path,
                output_dir=Path(args.output_path).expanduser().resolve(),
                checkpoint_dir=str(checkpoint_dir),
                agent=mainline_name,
                tasks=selected_task_names,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] failed to write AndroidWorld summary: {exc}")
        return 0
    finally:
        if "android_world_controller" in locals() and callable(original_get_controller):
            android_world_controller.get_controller = original_get_controller
        if "suite_utils" in locals() and callable(original_allocate_step_budget):
            suite_utils._allocate_step_budget = original_allocate_step_budget
        if "suite_utils" in locals() and callable(original_run_task):
            suite_utils._run_task = original_run_task
        if (
            "m3a_utils" in locals()
            and "original_parse_reason_action_output" in locals()
        ):
            m3a_utils.parse_reason_action_output = original_parse_reason_action_output
        if "datetime_utils" in locals() and "original_set_datetime" in locals():
            datetime_utils.set_datetime = original_set_datetime
        if (
            "file_utils" in locals()
            and original_clear_directory is not None
            and callable(original_clear_directory)
        ):
            file_utils.clear_directory = original_clear_directory
            if hasattr(file_utils, "_omniflow_empty_clear_directory_patch"):
                delattr(file_utils, "_omniflow_empty_clear_directory_patch")
        if (
            "adb_utils" in locals()
            and original_issue_generic_request is not None
            and callable(original_issue_generic_request)
        ):
            adb_utils.issue_generic_request = original_issue_generic_request
            if hasattr(adb_utils, "_omniflow_settings_get_output_patch"):
                delattr(adb_utils, "_omniflow_settings_get_output_patch")
        if (
            "app_snapshot" in locals()
            and app_snapshot is not None
            and original_restore_snapshot is not None
            and callable(original_restore_snapshot)
        ):
            app_snapshot.restore_snapshot = original_restore_snapshot
        if (
            "sqlite_utils" in locals()
            and isinstance(original_sqlite_writeback, dict)
        ):
            original_delete = original_sqlite_writeback.get(
                "delete_all_rows_from_table"
            )
            original_insert = original_sqlite_writeback.get(
                "insert_rows_to_remote_db"
            )
            if callable(original_delete):
                sqlite_utils.delete_all_rows_from_table = original_delete
            if callable(original_insert):
                sqlite_utils.insert_rows_to_remote_db = original_insert
            if hasattr(sqlite_utils, "_omniflow_sqlite_writeback_patch"):
                delattr(sqlite_utils, "_omniflow_sqlite_writeback_patch")
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
