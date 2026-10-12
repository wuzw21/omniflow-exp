"""Shared types and RunLog helpers for run_task and its split-out modules.

Moved verbatim from ``src.experiment.run_task``; that module re-exports these names.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence
from src.experiment.paths import safe_component
from src.experiment.source_records import CanonicalRunLog, SourceRunLogProfile


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DeviceTarget:
    label: str
    serial: str
    console_port: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "serial": self.serial,
            "console_port": self.console_port,
        }


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_stem(value: str, *, fallback: str = "task") -> str:
    return safe_component(
        value,
        fallback=fallback,
        max_length=120,
        strip_chars="._",
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"_read_error": f"missing: {path}"}
    except json.JSONDecodeError as exc:
        return {"_read_error": f"invalid JSON: {exc}"}
    return (
        decoded
        if isinstance(decoded, dict)
        else {"_read_error": "JSON root is not an object"}
    )


def _card_tool_name(card: Any) -> str:
    if not isinstance(card, dict):
        return ""
    tool_call = card.get("tool_call")
    if isinstance(tool_call, dict):
        name = tool_call.get("name")
        if name:
            return str(name).strip()
    action = card.get("action")
    if isinstance(action, dict):
        action_type = action.get("type")
        if action_type:
            return str(action_type).strip()
    return str(card.get("tool") or card.get("type") or "").strip()


def _card_args(card: Any) -> dict[str, Any]:
    if not isinstance(card, dict):
        return {}
    tool_call = card.get("tool_call")
    if isinstance(tool_call, dict) and isinstance(tool_call.get("arguments"), dict):
        return dict(tool_call["arguments"])
    if isinstance(card.get("args"), dict):
        return dict(card["args"])
    action = card.get("action")
    if isinstance(action, dict) and isinstance(action.get("params"), dict):
        return dict(action["params"])
    if isinstance(card.get("params"), dict):
        return dict(card["params"])
    return {}


def _is_empty_input_text_card(card: Any) -> bool:
    if _card_tool_name(card) != "input_text":
        return False
    args = _card_args(card)
    return "text" in args and str(args.get("text") or "") == ""


def _drop_empty_input_text_cards(payload: dict[str, Any]) -> int:
    cards = payload.get("cards")
    if not isinstance(cards, list):
        return 0
    filtered = [card for card in cards if not _is_empty_input_text_card(card)]
    removed_count = len(cards) - len(filtered)
    if removed_count <= 0:
        return 0

    payload["cards"] = filtered
    payload["step_count"] = len(filtered)
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics = {}
        payload["diagnostics"] = diagnostics
    normalizations = diagnostics.get("replay_normalizations")
    if not isinstance(normalizations, list):
        normalizations = []
        diagnostics["replay_normalizations"] = normalizations
    normalizations.append(
        {
            "kind": "drop_empty_input_text_cards",
            "removed_count": removed_count,
            "reason": "optional empty text inputs are no-op during replay",
        }
    )
    return removed_count


def profile_source_run_log(item: CanonicalRunLog) -> SourceRunLogProfile:
    data = _read_json(item.source_run_log)
    notes: list[str] = []
    read_error = str(data.get("_read_error") or "").strip()
    if read_error:
        return SourceRunLogProfile(
            task=item.task,
            source_run_log=item.source_run_log,
            replay_format="missing_or_invalid",
            step_count=0,
            card_count=0,
            latest_official_success_source=bool(
                item.meta.get("latest_official_success_source")
            ),
            direct_replay_ready=False,
            notes=(read_error,),
        )

    steps = data.get("steps")
    if isinstance(steps, list) and steps:
        return SourceRunLogProfile(
            task=item.task,
            source_run_log=item.source_run_log,
            replay_format="canonical_steps",
            step_count=len(steps),
            card_count=0,
            latest_official_success_source=bool(
                item.meta.get("latest_official_success_source")
            ),
            direct_replay_ready=True,
            notes=tuple(notes),
        )

    payload = data.get("payload")
    cards = payload.get("cards") if isinstance(payload, dict) else None
    if isinstance(payload, dict) and isinstance(cards, list) and cards:
        notes.append("wrapped payload must be materialized before OOB replay")
        return SourceRunLogProfile(
            task=item.task,
            source_run_log=item.source_run_log,
            replay_format="payload_cards",
            step_count=_coerce_int(payload.get("step_count"), len(cards)),
            card_count=len(cards),
            latest_official_success_source=bool(
                item.meta.get("latest_official_success_source")
            ),
            direct_replay_ready=True,
            notes=tuple(notes),
        )

    return SourceRunLogProfile(
        task=item.task,
        source_run_log=item.source_run_log,
        replay_format="unsupported",
        step_count=0,
        card_count=0,
        latest_official_success_source=bool(
            item.meta.get("latest_official_success_source")
        ),
        direct_replay_ready=False,
        notes=("no canonical steps or payload cards found",),
    )


def _normalize_observation_payload(value: Any) -> dict[str, Any]:
    observation = dict(value or {}) if isinstance(value, dict) else {}
    xml = str(
        observation.get("xml")
        or observation.get("observation_xml")
        or observation.get("page")
        or ""
    )
    if xml and "xml" not in observation:
        observation["xml"] = xml
    return observation


def _normalize_card_source_context(
    card: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    raw_context = card.get("source_context")
    if isinstance(raw_context, dict):
        for key in ("src_ctx", "source_context", "context"):
            nested = raw_context.get(key)
            if isinstance(nested, dict):
                raw_context = nested
                break
        context = dict(raw_context)
    else:
        context = {}

    before = _normalize_observation_payload(card.get("before"))
    if "page" not in context:
        page = str(before.get("xml") or before.get("observation_xml") or "").strip()
        if page:
            context["page"] = page

    for source_key, context_key in (
        ("screenshot_path", "screenshot_path"),
        ("image_path", "screenshot_path"),
        ("image_base64", "image_base64"),
        ("screenshot_base64", "screenshot_base64"),
        ("screenshot", "screenshot"),
    ):
        value = before.get(source_key)
        if value and context_key not in context:
            context[context_key] = value

    target_evidence = params.get("target_evidence")
    if isinstance(target_evidence, dict) and "element" not in context:
        context["element"] = dict(target_evidence)

    return context


def _canonical_steps_from_cards(cards: Sequence[Any]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for fallback_index, raw_card in enumerate(cards):
        if not isinstance(raw_card, dict):
            continue
        tool_name = _card_tool_name(raw_card)
        if not tool_name:
            continue
        params = _card_args(raw_card)
        source_context = _normalize_card_source_context(raw_card, params)
        if source_context and "source_context" not in params:
            params["source_context"] = source_context
        if params.get("target_description") and "clickPrompt" not in params:
            params["clickPrompt"] = str(params.get("target_description") or "")

        title = str(
            raw_card.get("title")
            or raw_card.get("summary")
            or f"{tool_name} from source card"
        ).strip()
        action = {
            "type": tool_name,
            "params": params,
            "success": bool(raw_card.get("success", True)),
            "description": title,
        }
        if source_context:
            action["source_context"] = source_context
        step_index = _coerce_int(raw_card.get("step_index"), fallback_index)
        steps.append(
            {
                "step_index": step_index,
                "status": str(raw_card.get("status") or "success"),
                "success": bool(raw_card.get("success", True)),
                "tool_call": {
                    "name": tool_name,
                    "params": params,
                    "reason": title,
                },
                "actions": [action],
                "executed_actions": [dict(action)],
                "observation_before_act": _normalize_observation_payload(
                    raw_card.get("before")
                ),
                "observation_after_act": _normalize_observation_payload(
                    raw_card.get("after")
                ),
                "source": raw_card.get("source") or "androidworld_source_card",
                "card_id": raw_card.get("card_id") or raw_card.get("tool_call_id"),
            }
        )
    return steps


def canonicalize_source_run_log(
    item: CanonicalRunLog,
) -> tuple[dict[str, Any], str, SourceRunLogProfile, Path | None]:
    """Return one canonical step-based runlog from the archived source asset."""

    profile = profile_source_run_log(item)
    data = _read_json(item.source_run_log)
    if data.get("_read_error"):
        raise ValueError(str(data.get("_read_error")))

    payload = dict(data.get("payload") or data)
    materialization = "original"
    cards = payload.get("cards")
    if isinstance(cards, list) and cards:
        payload = dict(payload)
        _drop_empty_input_text_cards(payload)
        cards = payload.get("cards") if isinstance(payload.get("cards"), list) else []
        steps = _canonical_steps_from_cards(cards)
        materialization = "payload_cards_to_canonical_steps"
    else:
        steps = list(payload.get("steps") or [])

    if not steps:
        raise ValueError(f"source runlog has no canonicalizable steps: {item.task}")

    run_id = str(
        payload.get("run_id")
        or data.get("run_id")
        or item.meta.get("run_id")
        or f"androidworld_canonical_source_{_safe_stem(item.task)}"
    ).strip()
    canonical = dict(payload)
    canonical.update(
        {
            "schema_version": str(
                canonical.get("schema_version") or "oob.run_log.canonical.v1"
            ),
            "run_id": run_id,
            "goal": str(canonical.get("goal") or item.goal),
            "success": bool(canonical.get("success", True)),
            "done_reason": str(canonical.get("done_reason") or "source_success"),
            "steps": steps,
            "step_count": len(steps),
            "source": str(canonical.get("source") or "androidworld_canonical_source"),
            "androidworld_task": str(canonical.get("androidworld_task") or item.task),
            "androidworld_params": dict(
                canonical.get("androidworld_params")
                if isinstance(canonical.get("androidworld_params"), dict)
                else item.params
            ),
        }
    )

    return canonical, materialization, profile, None


def _last_jsonl_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        text = line.strip()
        if not text:
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}
