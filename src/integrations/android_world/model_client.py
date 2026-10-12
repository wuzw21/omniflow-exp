"""OpenAI-compatible multimodal wrapper used by AndroidWorld baselines (split out of run_episode).

Moved verbatim from ``src.integrations.android_world.run_episode``; that module re-exports these names.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
from time import perf_counter
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
from omniflow.vlm.usage import token_usage_status
from src.experiment.protocol import FORMAL_MODEL, FORMAL_THINKING
from src.integrations.android_world.episode_utils import (
    _coerce_int,
    utc_now_iso,
)


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
        max_tokens: int = 512,
    ) -> None:
        resolved_model = str(model_name or "").strip() or FORMAL_MODEL
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
        self.latency_ms = 0.0
        self.last_error: str | None = None
        self.request_records: list[dict[str, Any]] = []
        self.embedding_calls = 0

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
            # Keep the formal protocol's Thinking decision explicit at the
            # final OpenAI-compatible boundary.  The common protocol is now
            # non-thinking, so this must never be hard-coded on here.
            "enable_thinking": False,
            "thinking": {"type": FORMAL_THINKING},
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
            "model": self.model,
            "endpoint": self.endpoint,
            "thinking": {"type": FORMAL_THINKING},
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
                        self.latency_ms += request_record["duration_ms"]
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
        self.latency_ms += request_record["duration_ms"]
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
            "embedding_calls": int(self.embedding_calls),
            "latency_ms": round(float(self.latency_ms), 6),
            "last_error": self.last_error,
        }
        summary["token_usage_status"] = token_usage_status(summary)
        return summary
