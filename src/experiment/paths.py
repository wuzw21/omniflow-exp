"""Canonical path and artifact-name rules for the experiment repository."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path, *, root: Path = REPO_ROOT) -> Path:
    """Resolve an absolute path or a repository-relative path."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def resolve_reference(
    index_path: str | Path,
    value: Any,
    *,
    root: Path = REPO_ROOT,
) -> Path:
    """Resolve an evidence reference relative to its index file."""

    path = Path(str(value or "")).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (resolve_path(index_path, root=root).parent / path).resolve()


def safe_component(
    value: Any,
    *,
    fallback: str = "item",
    max_length: int | None = None,
    strip_chars: str = "._",
) -> str:
    """Return one stable filesystem component without path traversal."""

    component = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(value or "").strip(),
    ).strip(strip_chars)
    if max_length is not None:
        component = component[:max(1, int(max_length))]
    return component or fallback


def safe_relative_path(value: str, *, fallback: str = "run") -> Path:
    """Sanitize a user-provided nested run suffix into relative components."""

    parts = [
        safe_component(part, fallback="", max_length=120, strip_chars="._")
        for part in re.split(r"[\\/]+", str(value or "").strip())
        if str(part or "").strip()
    ]
    parts = [part for part in parts if part]
    return Path(*parts) if parts else Path(safe_component("", fallback=fallback))
