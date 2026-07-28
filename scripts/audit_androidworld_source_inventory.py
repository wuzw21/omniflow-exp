from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.migrate_androidworld_source_replay_runlog import (
    load_androidworld_app_name_to_package,
    migrate_source_replay_runlog,
)


SCHEMA_VERSION = "omniflow.androidworld-source-inventory-audit.v1"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "item"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _source_path(canonical_repo: Path, row: dict[str, Any]) -> Path:
    source = Path(str(row.get("retained_source_run_log") or "")).expanduser()
    if not source.is_absolute():
        source = canonical_repo / source
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return source


def audit_inventory(
    *,
    inventory_path: Path,
    canonical_repo: Path,
    android_world_root: Path,
    output_root: Path,
    expected_tasks: int,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"immutable_output_already_exists:{output_root}")
    inventory = _read_object(inventory_path)
    if len(inventory) != expected_tasks:
        raise ValueError(
            f"expected_{expected_tasks}_source_tasks:actual={len(inventory)}"
        )
    output_root.mkdir(parents=True)
    app_mapping, app_mapping_source = load_androidworld_app_name_to_package(
        android_world_root
    )
    started = time.monotonic()
    rows: dict[str, Any] = {}
    passed = 0
    for task, raw_row in inventory.items():
        task_started = time.monotonic()
        task_root = output_root / "tasks" / _safe_name(task)
        try:
            if not isinstance(raw_row, dict):
                raise ValueError("source_inventory_row_invalid")
            source = _source_path(canonical_repo, raw_row)
            manifest = migrate_source_replay_runlog(
                source_path=source,
                output_root=task_root / "migration",
                drop_clear_text=True,
                app_name_to_package=app_mapping,
                app_mapping_source=app_mapping_source,
            )
            row = {
                "classification": "passed",
                "task": task,
                "source_run_log": str(source),
                "source_run_log_sha256": _sha256(source),
                "migrated_run_log": manifest["output_run_log"],
                "migrated_run_log_sha256": manifest["output_run_log_sha256"],
                "step_count": manifest["step_count"],
                "migration_wall_sec": round(time.monotonic() - task_started, 6),
            }
            passed += 1
        except Exception as error:
            row = {
                "classification": "failed",
                "task": task,
                "error_type": type(error).__name__,
                "error": str(error),
                "migration_wall_sec": round(time.monotonic() - task_started, 6),
            }
        rows[task] = row
        _write_json(task_root / "audit_result.json", row)
    total_wall_sec = round(time.monotonic() - started, 6)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "inventory_index": str(inventory_path),
        "inventory_sha256": _sha256(inventory_path),
        "canonical_repo": str(canonical_repo),
        "android_world_root": str(android_world_root),
        "app_mapping_source": str(app_mapping_source),
        "app_mapping_source_sha256": _sha256(Path(app_mapping_source)),
        "task_count": len(inventory),
        "passed_task_count": passed,
        "failed_task_count": len(inventory) - passed,
        "migration_wall_sec": total_wall_sec,
        "model_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "token_usage_status": "not_applicable",
        "uses_oob": False,
        "tasks": rows,
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit canonical migration for a complete AndroidWorld source inventory."
    )
    parser.add_argument("--inventory-index", required=True, type=Path)
    parser.add_argument("--canonical-repo", required=True, type=Path)
    parser.add_argument("--android-world-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-tasks", type=int, default=116)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = audit_inventory(
        inventory_path=args.inventory_index.expanduser().resolve(),
        canonical_repo=args.canonical_repo.expanduser().resolve(),
        android_world_root=args.android_world_root.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        expected_tasks=args.expected_tasks,
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "tasks"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if summary["failed_task_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
