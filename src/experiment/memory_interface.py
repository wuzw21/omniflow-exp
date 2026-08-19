"""Small interface shared by provider-owned memory adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable


MemoryOperation = Literal["prepare", "check"]


@dataclass(frozen=True)
class MemoryRequest:
    """Inputs passed from the experiment flow to one provider adapter."""

    task_name: str
    source_index: Path | None = None
    source_run_log: Path | None = None
    output_root: Path | None = None
    memory_root: Path | None = None
    model: str = ""
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryPackage:
    """Provider-neutral envelope around a provider-owned memory tree."""

    provider: str
    task_name: str
    root: Path
    bundle_root: Path
    schema_version: str
    sha256: str = ""
    manifest: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "task_name": self.task_name,
            "root": str(self.root),
            "bundle_root": str(self.bundle_root),
            "schema_version": self.schema_version,
            "sha256": self.sha256,
            "manifest": str(self.manifest or ""),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MemoryCheck:
    """Provider-neutral result of checking one prepared memory."""

    provider: str
    task_name: str
    valid: bool
    root: Path
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "task_name": self.task_name,
            "valid": self.valid,
            "root": str(self.root),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "details": dict(self.details),
        }


@runtime_checkable
class MemoryAdapter(Protocol):
    """Interface used by the experiment flow, implemented by each provider."""

    @property
    def name(self) -> str:
        ...

    @property
    def schema_version(self) -> str:
        ...

    def prepare(self, request: MemoryRequest) -> MemoryPackage:
        ...

    def check(self, request: MemoryRequest) -> MemoryCheck:
        ...


def operation_name(operation: MemoryOperation) -> str:
    """Return the stable name used by command/API boundaries."""

    return str(operation)


__all__ = [
    "MemoryAdapter",
    "MemoryCheck",
    "MemoryOperation",
    "MemoryPackage",
    "MemoryRequest",
    "operation_name",
]
