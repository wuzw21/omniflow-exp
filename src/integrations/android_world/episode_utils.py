"""Small serialization/coercion helpers shared by the AndroidWorld episode modules.

Moved verbatim from ``src.integrations.android_world.run_episode``; that module re-exports these names.
"""

from __future__ import annotations

import dataclasses
import datetime
import logging
from pathlib import Path
from typing import Any, Sequence
import unicodedata


# Keep the historical logger name so existing log filters/handlers still match.
logger = logging.getLogger("src.integrations.android_world.run_episode")


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


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


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


def _coerce_int(value: Any) -> int:
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    try:
        return int(value or 0)
    except Exception:
        return 0


def _raw_replay_number(params: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key not in params:
            continue
        try:
            return float(params.get(key))
        except Exception:
            continue
    return None
