"""Optional reference to provider-owned memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MemoryRef:
    """The only memory value shared with the common execution flow."""

    provider: str
    task_name: str
    root: Path
    schema_version: str
    sha256: str = ""
    manifest: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "task_name": self.task_name,
            "root": str(self.root),
            "schema_version": self.schema_version,
            "sha256": self.sha256,
            "manifest": str(self.manifest or ""),
            "metadata": dict(self.metadata),
        }


__all__ = ["MemoryRef"]
