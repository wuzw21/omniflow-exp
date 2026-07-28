from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omniflow import canonicalize_run_log
from omniflow.schemas import canonicalize_action


SCHEMA_VERSION = "omniflow.androidworld-corrected-source-inventory.v1"
CORRECTIONS_SCHEMA_VERSION = "omniflow.androidworld-source-ui-corrections.v1"
FORBIDDEN_DIRECT_MUTATIONS = frozenset(
    {"create_picture_file", "move_file", "set_clipboard"}
)


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


def _write_immutable(path: Path, value: object) -> None:
    content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"immutable_output_differs:{path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _contains_oob_provider(value: object) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "provider" and str(item).strip().casefold() == "oob":
                return True
            if _contains_oob_provider(item):
                return True
    elif isinstance(value, list):
        return any(_contains_oob_provider(item) for item in value)
    return False


def _validated_ui_run_log(
    path: Path,
    *,
    expected_sha256: str,
    expected_seed: int,
    expected_goal: str,
) -> dict[str, Any]:
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"corrected_source_sha256_mismatch:{path}:"
            f"actual={actual_sha256}:expected={expected_sha256}"
        )
    raw = _read_object(path)
    if _contains_oob_provider(raw):
        raise ValueError(f"corrected_source_oob_provider_forbidden:{path}")
    run_log = canonicalize_run_log(raw)
    if run_log.get("success") is not True or not run_log.get("steps"):
        raise ValueError(f"corrected_source_successful_run_log_required:{path}")
    if str(run_log.get("goal") or "") != expected_goal:
        raise ValueError(f"corrected_source_goal_mismatch:{path}")
    diagnostics = run_log.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError(f"corrected_source_diagnostics_required:{path}")
    if diagnostics.get("source_seed") != expected_seed:
        raise ValueError(f"corrected_source_seed_mismatch:{path}")
    if diagnostics.get("direct_state_mutation") is not False:
        raise ValueError(f"corrected_source_direct_state_flag_required:{path}")
    for step in run_log["steps"]:
        action = canonicalize_action(step["action"], replayable_only=True)
        if action["tool"] in FORBIDDEN_DIRECT_MUTATIONS:
            raise ValueError(
                f"corrected_source_direct_mutation_forbidden:{action['tool']}"
            )
    return run_log


def build_corrected_inventory(
    *,
    base_index_path: Path,
    corrections_path: Path,
    source_root: Path,
    expected_tasks: int,
    absolute_source_paths: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_index = _read_object(base_index_path)
    if len(base_index) != expected_tasks:
        raise ValueError(
            f"expected_{expected_tasks}_source_tasks:actual={len(base_index)}"
        )
    corrections_payload = _read_object(corrections_path)
    if corrections_payload.get("schema_version") != CORRECTIONS_SCHEMA_VERSION:
        raise ValueError("source_corrections_schema_version_invalid")
    source_seed = corrections_payload.get("source_seed")
    if source_seed != 111:
        raise ValueError(f"source_corrections_seed_invalid:{source_seed}")
    corrections = corrections_payload.get("corrections")
    if not isinstance(corrections, dict) or not corrections:
        raise ValueError("source_corrections_required")

    inventory = json.loads(json.dumps(base_index))
    applied: dict[str, Any] = {}
    for task, raw_correction in corrections.items():
        if task not in inventory:
            raise ValueError(f"corrected_source_unknown_task:{task}")
        if not isinstance(raw_correction, dict):
            raise ValueError(f"corrected_source_record_invalid:{task}")
        source_ref = str(raw_correction.get("retained_source_run_log") or "").strip()
        expected_sha256 = str(raw_correction.get("run_log_sha256") or "").strip()
        if not source_ref or len(expected_sha256) != 64:
            raise ValueError(f"corrected_source_reference_invalid:{task}")
        source_path = Path(source_ref).expanduser()
        if not source_path.is_absolute():
            source_path = source_root / source_path
        source_path = source_path.resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        row = inventory[task]
        if not isinstance(row, dict):
            raise ValueError(f"source_inventory_row_invalid:{task}")
        run_log = _validated_ui_run_log(
            source_path,
            expected_sha256=expected_sha256,
            expected_seed=source_seed,
            expected_goal=str(row.get("goal") or ""),
        )
        previous_ref = str(row.get("retained_source_run_log") or "")
        prevalidated = raw_correction.get("fresh_official_replay_prevalidated") is True
        row.update(
            {
                "run_id": run_log["run_id"],
                "step_count": len(run_log["steps"]),
                "retained_source_run_log": (
                    str(source_path) if absolute_source_paths else source_ref
                ),
                "latest_official_success_source": prevalidated,
                "raw_replay_completed": prevalidated,
                "function_replay_success": False,
                "source_correction": {
                    "kind": "audited_human_teacher_ui_actions",
                    "reason": str(raw_correction.get("reason") or ""),
                    "previous_retained_source_run_log": previous_ref,
                    "run_log_sha256": expected_sha256,
                    "fresh_official_replay_required": True,
                    "current_replay_evidence": raw_correction.get(
                        "current_replay_evidence", {}
                    ),
                    "historical_evidence": raw_correction.get("historical_evidence", {}),
                },
            }
        )
        applied[task] = {
            "previous_retained_source_run_log": previous_ref,
            "corrected_retained_source_run_log": row["retained_source_run_log"],
            "run_log_sha256": expected_sha256,
            "run_id": run_log["run_id"],
            "step_count": len(run_log["steps"]),
            "fresh_official_replay_prevalidated": prevalidated,
        }

    provenance = {
        "schema_version": SCHEMA_VERSION,
        "source_seed": source_seed,
        "base_index": str(base_index_path),
        "base_index_sha256": _sha256(base_index_path),
        "corrections": str(corrections_path),
        "corrections_sha256": _sha256(corrections_path),
        "source_root": str(source_root),
        "absolute_source_paths": absolute_source_paths,
        "task_count": len(inventory),
        "corrected_task_count": len(applied),
        "corrected_tasks": applied,
        "fresh_official_replay_required": True,
        "uses_oob": False,
    }
    return inventory, provenance


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an immutable AndroidWorld source inventory with audited UI corrections."
    )
    parser.add_argument("--base-index", required=True, type=Path)
    parser.add_argument("--corrections", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-index", required=True, type=Path)
    parser.add_argument("--output-provenance", required=True, type=Path)
    parser.add_argument("--expected-tasks", type=int, default=116)
    parser.add_argument("--absolute-source-paths", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    base_index = args.base_index.expanduser().resolve()
    corrections = args.corrections.expanduser().resolve()
    source_root = args.source_root.expanduser().resolve()
    inventory, provenance = build_corrected_inventory(
        base_index_path=base_index,
        corrections_path=corrections,
        source_root=source_root,
        expected_tasks=args.expected_tasks,
        absolute_source_paths=args.absolute_source_paths,
    )
    _write_immutable(args.output_index.expanduser().resolve(), inventory)
    provenance["output_index"] = str(args.output_index.expanduser().resolve())
    provenance["output_index_sha256"] = _sha256(args.output_index.expanduser().resolve())
    _write_immutable(args.output_provenance.expanduser().resolve(), provenance)
    print(json.dumps(provenance, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
