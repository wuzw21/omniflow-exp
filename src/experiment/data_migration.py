"""Stage authoritative local experiment data for an isolated test host."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable


CURRENT_SCHEMA = "omniflow.data-index.v2"


@dataclass(frozen=True)
class MigrationPlan:
    source_data: Path
    target_data: Path
    current_sha256: str
    files: tuple[Path, ...]
    missing_references: tuple[str, ...]

    @property
    def total_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.files)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "omniflow.authoritative-data-migration.v1",
            "source_data": str(self.source_data),
            "target_data": str(self.target_data),
            "current_sha256": self.current_sha256,
            "file_count": len(self.files),
            "total_bytes": self.total_bytes,
            "missing_references": list(self.missing_references),
            "files": [str(path.relative_to(self.source_data)) for path in self.files],
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)
    elif isinstance(value, str):
        yield value


def _rewrite_paths(value: Any, source_data: Path, target_data: Path) -> Any:
    source_text = str(source_data)
    target_text = str(target_data)
    if isinstance(value, dict):
        return {
            key: _rewrite_paths(item, source_data, target_data)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rewrite_paths(item, source_data, target_data) for item in value]
    if isinstance(value, str) and (
        value == source_text or value.startswith(source_text + os.sep)
    ):
        return target_text + value[len(source_text) :]
    return value


def _referenced_path(value: str, source_data: Path) -> Path | None:
    source_text = str(source_data)
    if value != source_text and not value.startswith(source_text + os.sep):
        return None
    return Path(value).expanduser()


def _add_path(
    path: Path,
    *,
    source_data: Path,
    files: set[Path],
    pending_json: list[Path],
    missing: set[str],
    allow_directory: bool = False,
) -> None:
    path = path.resolve()
    try:
        path.relative_to(source_data)
    except ValueError:
        return
    if not path.exists():
        missing.add(str(path))
        return
    if path.is_dir() and not allow_directory:
        return
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file():
                _add_path(
                    child,
                    source_data=source_data,
                    files=files,
                    pending_json=pending_json,
                    missing=missing,
                    allow_directory=False,
                )
        return
    if path in files:
        return
    files.add(path)
    if path.suffix.casefold() in {".json", ".jsonl"}:
        pending_json.append(path)


def build_migration_plan(
    source_data: str | Path,
    target_data: str | Path,
) -> MigrationPlan:
    source_root = Path(source_data).expanduser().resolve()
    target_root = Path(target_data).expanduser().absolute()
    current_path = source_root / "current.json"
    if not current_path.is_file():
        raise FileNotFoundError(f"authoritative_current_missing:{current_path}")
    current = json.loads(current_path.read_text(encoding="utf-8"))
    if not isinstance(current, dict) or current.get("schema_version") != CURRENT_SCHEMA:
        raise ValueError(f"authoritative_current_schema_invalid:{current_path}")

    files: set[Path] = set()
    pending_json = [current_path]
    missing: set[str] = set()
    _add_path(
        current_path,
        source_data=source_root,
        files=files,
        pending_json=pending_json,
        missing=missing,
    )
    for store in (current.get("canonical", {}).get("function_stores", {}) or {}).values():
        if isinstance(store, dict):
            store_path = str(store.get("store_path") or "").strip()
            if store_path:
                _add_path(
                    Path(store_path).parent,
                    source_data=source_root,
                    files=files,
                    pending_json=pending_json,
                    missing=missing,
                    allow_directory=True,
                )
    for memory in (current.get("canonical", {}).get("prepared_memories", {}) or {}).values():
        if isinstance(memory, dict):
            memory_root = str(memory.get("memory_root") or "").strip()
            if memory_root:
                _add_path(
                    Path(memory_root),
                    source_data=source_root,
                    files=files,
                    pending_json=pending_json,
                    missing=missing,
                    allow_directory=True,
                )

    scanned: set[Path] = set()
    while pending_json:
        path = pending_json.pop()
        if path in scanned or not path.is_file():
            continue
        scanned.add(path)
        try:
            if path.suffix.casefold() == ".jsonl":
                payloads = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            else:
                payloads = [json.loads(path.read_text(encoding="utf-8"))]
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        for payload in payloads:
            for value in _walk_strings(payload):
                referenced = _referenced_path(value, source_root)
                if referenced is not None:
                    _add_path(
                        referenced,
                        source_data=source_root,
                        files=files,
                        pending_json=pending_json,
                        missing=missing,
                    )

    return MigrationPlan(
        source_data=source_root,
        target_data=target_root,
        current_sha256=_sha256(current_path),
        files=tuple(sorted(files)),
        missing_references=tuple(sorted(missing)),
    )


def stage_migration(
    plan: MigrationPlan,
    stage_root: str | Path | None = None,
) -> Path:
    if stage_root is None:
        stage = Path(tempfile.mkdtemp(prefix="omniflow-authoritative-migration-"))
    else:
        stage = Path(stage_root).expanduser().resolve()
        if stage.exists() and any(stage.iterdir()):
            raise FileExistsError(f"migration_stage_not_empty:{stage}")
        stage.mkdir(parents=True, exist_ok=True)

    current = json.loads((plan.source_data / "current.json").read_text(encoding="utf-8"))
    (stage / "current.json").write_text(
        json.dumps(
            _rewrite_paths(current, plan.source_data, plan.target_data),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (stage / "files.txt").write_text(
        "\n".join(
            str(path.relative_to(plan.source_data))
            for path in plan.files
            if path != plan.source_data / "current.json"
        )
        + "\n",
        encoding="utf-8",
    )
    (stage / "migration.json").write_text(
        json.dumps(
            {
                **plan.as_dict(),
                "staged_current_sha256": _sha256(stage / "current.json"),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return stage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or stage data/current.json as authoritative migration input."
    )
    parser.add_argument("--source-data", required=True, type=Path)
    parser.add_argument("--target-data", required=True, type=Path)
    parser.add_argument("--stage-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = build_migration_plan(args.source_data, args.target_data)
    if args.stage_root is None:
        print(json.dumps(plan.as_dict(), ensure_ascii=False, indent=2))
        return 0
    stage = stage_migration(plan, args.stage_root)
    print(json.dumps({**plan.as_dict(), "stage_root": str(stage)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
