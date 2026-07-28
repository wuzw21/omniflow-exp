from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

MOBILE_AGENT_V3_OFFICIAL_REVISION = (
    "11cea575561fb7800b5fb6b6cafa56f7a91de11f"
)
GUI_OWL_7B_MODEL_REVISION = "7c1644c0288da07435a485701d0fea0ac353f38a"
GUI_OWL_REQUIRED_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    *(f"model-{index:05d}-of-00005.safetensors" for index in range(1, 6)),
)


def _coerce_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def classify_mobile_agent_v3_role(prompt: str) -> str:
    text = str(prompt or "").lower()
    if "track progress and devise high-level plans" in text:
        return "manager"
    if "decide the next action to perform" in text:
        return "executor"
    if "verify whether the last action produced the expected behavior" in text:
        return "action_reflector"
    if "take notes of important content" in text:
        return "notetaker"
    return "unknown"


def _image_record(
    value: str | Path,
    *,
    capture_root: Path,
    call_index: int,
    image_index: int,
) -> dict[str, Any]:
    source = Path(value).expanduser().resolve()
    record: dict[str, Any] = {
        "source_path": str(source),
        "path": str(source),
        "exists": source.is_file(),
    }
    if source.is_file():
        payload = source.read_bytes()
        suffix = source.suffix if source.suffix else ".bin"
        captured = capture_root / (
            f"call_{call_index:06d}_image_{image_index:02d}{suffix}"
        )
        with captured.open("xb") as handle:
            handle.write(payload)
        record["path"] = str(captured)
        record["sha256"] = hashlib.sha256(payload).hexdigest()
        record["size_bytes"] = len(payload)
    return record


def inspect_gui_owl_model(
    model_root: str | Path,
    *,
    revision: str = GUI_OWL_7B_MODEL_REVISION,
) -> dict[str, Any]:
    """Fail closed unless a complete GUI-Owl snapshot proves its HF revision."""

    root = Path(model_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"gui_owl_model_root_missing:{root}")
    metadata_root = root / ".cache" / "huggingface" / "download"
    files: list[dict[str, Any]] = []
    for relative in GUI_OWL_REQUIRED_FILES:
        artifact = root / relative
        metadata = metadata_root / f"{relative}.metadata"
        if not artifact.is_file() or artifact.stat().st_size <= 0:
            raise FileNotFoundError(f"gui_owl_model_file_missing:{artifact}")
        if not metadata.is_file():
            raise FileNotFoundError(f"gui_owl_revision_metadata_missing:{metadata}")
        lines = metadata.read_text(encoding="utf-8").splitlines()
        found_revision = str(lines[0] if lines else "").strip()
        if found_revision != str(revision).strip():
            raise ValueError(
                "gui_owl_revision_mismatch:"
                f"{relative}:expected={revision}:actual={found_revision}"
            )
        files.append(
            {
                "relative_path": relative,
                "size_bytes": artifact.stat().st_size,
                "upstream_etag": str(lines[1] if len(lines) > 1 else "").strip(),
                "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
            }
        )
    incomplete = sorted(
        str(path.relative_to(root)) for path in root.rglob("*.incomplete")
    )
    if incomplete:
        raise ValueError(f"gui_owl_incomplete_downloads:{','.join(incomplete)}")
    return {
        "schema_version": "omniflow.gui-owl-model-audit.v1",
        "model_root": str(root),
        "revision": str(revision),
        "required_file_count": len(files),
        "revision_metadata_complete": True,
        "total_bytes": sum(int(item["size_bytes"]) for item in files),
        "files": files,
    }


class MobileAgentV3UsageLedger:
    """Append-only per-call accounting for the official multi-agent policy."""

    def __init__(self, calls_jsonl: str | Path) -> None:
        self.calls_jsonl = Path(calls_jsonl).expanduser().resolve()
        self.calls_jsonl.parent.mkdir(parents=True, exist_ok=True)
        if self.calls_jsonl.exists():
            raise FileExistsError(f"mobile_agent_v3_calls_exist:{self.calls_jsonl}")
        self.images_root = self.calls_jsonl.parent / (
            f"{self.calls_jsonl.stem}_images"
        )
        try:
            self.images_root.mkdir(parents=False, exist_ok=False)
        except FileExistsError as exc:
            raise FileExistsError(
                f"mobile_agent_v3_call_images_exist:{self.images_root}"
            ) from exc
        self._rows: list[dict[str, Any]] = []

    def record_call(
        self,
        *,
        prompt: str,
        images: Iterable[str | Path],
        response_text: str,
        response_metadata: dict[str, Any] | None,
        usage: dict[str, Any] | None,
        wall_sec: float,
        ok: bool,
        error: str = "",
    ) -> dict[str, Any]:
        metadata = dict(response_metadata or {})
        usage_payload = dict(usage or {})
        prompt_tokens = _coerce_int(usage_payload.get("prompt_tokens"))
        completion_tokens = _coerce_int(usage_payload.get("completion_tokens"))
        total_tokens = _coerce_int(usage_payload.get("total_tokens"))
        if total_tokens <= 0 and (prompt_tokens > 0 or completion_tokens > 0):
            total_tokens = prompt_tokens + completion_tokens
        call_index = len(self._rows) + 1
        row = {
            "schema_version": "omniflow.mobile-agent-v3-call.v2",
            "call_index": call_index,
            "role": classify_mobile_agent_v3_role(prompt),
            "ok": bool(ok),
            "error": str(error or ""),
            "model": str(metadata.get("model") or ""),
            "response_id": str(metadata.get("id") or ""),
            "prompt": str(prompt or ""),
            "prompt_sha256": hashlib.sha256(
                str(prompt or "").encode("utf-8")
            ).hexdigest(),
            "prompt_chars": len(str(prompt or "")),
            "images": [
                _image_record(
                    image,
                    capture_root=self.images_root,
                    call_index=call_index,
                    image_index=image_index,
                )
                for image_index, image in enumerate(images, 1)
            ],
            "response_text": str(response_text or ""),
            "response_metadata": metadata,
            "usage": usage_payload,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "usage_present": bool(usage_payload),
            "wall_sec": round(float(wall_sec), 6),
        }
        self._rows.append(row)
        with self.calls_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return row

    def get_usage_summary(self) -> dict[str, Any]:
        calls_by_role: Counter[str] = Counter()
        tokens_by_role: Counter[str] = Counter()
        models: list[str] = []
        for row in self._rows:
            role = str(row["role"])
            calls_by_role[role] += 1
            tokens_by_role[role] += _coerce_int(row["total_tokens"])
            model = str(row.get("model") or "")
            if model and model not in models:
                models.append(model)
        responses_with_usage = sum(bool(row["usage_present"]) for row in self._rows)
        responses_without_usage = len(self._rows) - responses_with_usage
        failed_calls = sum(not bool(row["ok"]) for row in self._rows)
        total_tokens = sum(_coerce_int(row["total_tokens"]) for row in self._rows)
        if not self._rows:
            token_status = "not_applicable"
        elif failed_calls or responses_without_usage:
            token_status = "partial" if total_tokens > 0 else "unavailable"
        else:
            token_status = "tracked"
        return {
            "model": models[0] if len(models) == 1 else ",".join(models),
            "model_calls": len(self._rows),
            "prompt_tokens": sum(
                _coerce_int(row["prompt_tokens"]) for row in self._rows
            ),
            "completion_tokens": sum(
                _coerce_int(row["completion_tokens"]) for row in self._rows
            ),
            "total_tokens": total_tokens,
            "responses_with_usage": responses_with_usage,
            "responses_without_usage": responses_without_usage,
            "failed_calls": failed_calls,
            "wall_sec": round(
                sum(float(row["wall_sec"]) for row in self._rows), 6
            ),
            "token_usage_status": token_status,
            "calls_by_role": dict(sorted(calls_by_role.items())),
            "tokens_by_role": dict(sorted(tokens_by_role.items())),
            "calls_jsonl": str(self.calls_jsonl),
        }


def audit_mobile_agent_v3_call_evidence(
    calls_jsonl: str | Path,
    *,
    expected_usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute a Mobile-Agent-V3 call ledger and its immutable image hashes."""

    path = Path(calls_jsonl).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"mobile_agent_v3_calls_missing:{path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"mobile_agent_v3_call_json_invalid:{line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(
                f"mobile_agent_v3_call_not_object:{line_number}"
            )
        rows.append(value)
    if not rows:
        raise ValueError("mobile_agent_v3_calls_empty")

    calls_by_role: Counter[str] = Counter()
    tokens_by_role: Counter[str] = Counter()
    captured_paths: set[Path] = set()
    image_count = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    wall_sec = 0.0
    images_root = path.parent / f"{path.stem}_images"
    for expected_index, row in enumerate(rows, 1):
        if row.get("schema_version") != "omniflow.mobile-agent-v3-call.v2":
            raise ValueError(
                f"mobile_agent_v3_call_schema_invalid:{expected_index}"
            )
        if _coerce_int(row.get("call_index")) != expected_index:
            raise ValueError(
                f"mobile_agent_v3_call_index_invalid:{expected_index}"
            )
        if row.get("ok") is not True or str(row.get("error") or ""):
            raise ValueError(f"mobile_agent_v3_call_failed:{expected_index}")
        if row.get("usage_present") is not True:
            raise ValueError(
                f"mobile_agent_v3_call_usage_missing:{expected_index}"
            )
        prompt = str(row.get("prompt") or "")
        if str(row.get("prompt_sha256") or "") != hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest():
            raise ValueError(
                f"mobile_agent_v3_prompt_sha256_mismatch:{expected_index}"
            )
        row_prompt = _coerce_int(row.get("prompt_tokens"))
        row_completion = _coerce_int(row.get("completion_tokens"))
        row_total = _coerce_int(row.get("total_tokens"))
        if row_prompt + row_completion != row_total:
            raise ValueError(
                f"mobile_agent_v3_token_arithmetic_invalid:{expected_index}"
            )
        prompt_tokens += row_prompt
        completion_tokens += row_completion
        total_tokens += row_total
        wall_sec += float(row.get("wall_sec") or 0.0)
        role = str(row.get("role") or "unknown")
        calls_by_role[role] += 1
        tokens_by_role[role] += row_total

        images = row.get("images")
        if not isinstance(images, list):
            raise ValueError(
                f"mobile_agent_v3_call_images_invalid:{expected_index}"
            )
        for image_index, image in enumerate(images, 1):
            if not isinstance(image, dict) or image.get("exists") is not True:
                raise ValueError(
                    "mobile_agent_v3_call_image_missing:"
                    f"{expected_index}:{image_index}"
                )
            captured = Path(str(image.get("path") or "")).expanduser().resolve()
            source = Path(
                str(image.get("source_path") or "")
            ).expanduser().resolve()
            if captured == source or captured.parent != images_root:
                raise ValueError(
                    "mobile_agent_v3_call_image_not_captured:"
                    f"{expected_index}:{image_index}"
                )
            if captured in captured_paths:
                raise ValueError(
                    "mobile_agent_v3_call_image_duplicate:"
                    f"{expected_index}:{image_index}"
                )
            if not captured.is_file():
                raise FileNotFoundError(
                    f"mobile_agent_v3_call_image_missing:{captured}"
                )
            payload = captured.read_bytes()
            if len(payload) != _coerce_int(image.get("size_bytes")):
                raise ValueError(
                    "mobile_agent_v3_image_size_mismatch:"
                    f"{expected_index}:{image_index}"
                )
            if hashlib.sha256(payload).hexdigest() != str(
                image.get("sha256") or ""
            ):
                raise ValueError(
                    "mobile_agent_v3_image_sha256_mismatch:"
                    f"{expected_index}:{image_index}"
                )
            captured_paths.add(captured)
            image_count += 1

    audit = {
        "schema_version": "omniflow.mobile-agent-v3-call-evidence-audit.v1",
        "status": "passed",
        "model_calls": len(rows),
        "image_count": image_count,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "responses_with_usage": len(rows),
        "responses_without_usage": 0,
        "failed_calls": 0,
        "wall_sec": round(wall_sec, 6),
        "token_usage_status": "tracked",
        "calls_by_role": dict(sorted(calls_by_role.items())),
        "tokens_by_role": dict(sorted(tokens_by_role.items())),
        "calls_jsonl": str(path),
        "images_root": str(images_root),
    }
    expected = dict(expected_usage or {})
    for key in (
        "model_calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "responses_with_usage",
        "responses_without_usage",
        "failed_calls",
        "wall_sec",
        "token_usage_status",
        "calls_by_role",
        "tokens_by_role",
    ):
        if key in expected and audit[key] != expected[key]:
            raise ValueError(
                "mobile_agent_v3_usage_audit_mismatch:"
                f"{key}:expected={expected[key]}:actual={audit[key]}"
            )
    return audit


def count_mobile_agent_v3_actions(episode_data: Any) -> int:
    if not isinstance(episode_data, dict):
        return 0
    histories = episode_data.get("action_history")
    if not isinstance(histories, list) or not histories:
        return 0
    final_history = histories[-1]
    if not isinstance(final_history, list):
        return 0
    excluded = {"done", "terminate", "invalid"}
    return sum(
        1
        for action in final_history
        if isinstance(action, dict)
        and str(action.get("action") or "").strip().lower() not in excluded
    )
