"""Canonical path and artifact-name rules for the experiment repository."""

from __future__ import annotations

import hashlib
import os
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


def sha256_file(value: str | Path, *, root: Path = REPO_ROOT) -> str:
    """Return the SHA-256 digest of one resolved file."""

    path = resolve_path(value, root=root)
    if not path.is_file():
        raise FileNotFoundError(f"provenance_artifact_missing:{path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def relative_reference(value: str | Path, *, base: str | Path) -> str:
    """Serialize one artifact path relative to an owning manifest directory.

    Artifact manifests must be relocatable.  Runtime code may still resolve the
    returned reference against the manifest directory, but the serialized
    value never contains the workstation's absolute checkout path.
    """

    target = Path(value).expanduser().resolve()
    owner = Path(base).expanduser().resolve()
    return os.path.relpath(target, owner).replace(os.sep, "/")


def resolve_relative_reference(
    value: str | Path,
    *,
    base: str | Path,
    allow_legacy_absolute: bool = True,
) -> Path:
    """Resolve a manifest reference relative to its owning directory.

    Existing experimental memories used absolute references.  They remain
    readable for evidence preservation, while all newly written manifests use
    :func:`relative_reference`.
    """

    path = Path(str(value or "")).expanduser()
    if path.is_absolute():
        if not allow_legacy_absolute:
            raise ValueError("absolute_artifact_reference_forbidden")
        return path.resolve()
    return (Path(base).expanduser().resolve() / path).resolve()


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
