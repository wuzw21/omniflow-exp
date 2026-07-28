from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from omniflow.artifact import parse_function_artifact


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def migrate_store(payload: Any) -> tuple[dict[str, Any], dict[str, int]]:
    if not isinstance(payload, dict) or payload.get("schema_version") != "omniflow.store.v2":
        raise ValueError("unsupported_store_version")
    raw_functions = payload.get("functions")
    if not isinstance(raw_functions, dict):
        raise ValueError("function_store_functions_must_be_object")
    output_functions: dict[str, dict[str, Any]] = {}
    migrated_steps = 0
    removed_wait_arguments = 0
    derived_swipe_directions = 0
    for key, raw_function in sorted(raw_functions.items()):
        if not isinstance(raw_function, dict):
            raise ValueError(f"function_artifact_must_be_object:{key}")
        migrated_function = json.loads(json.dumps(raw_function, ensure_ascii=False))
        steps = migrated_function.get("steps")
        if not isinstance(steps, list):
            raise ValueError(f"function_steps_required:{key}")
        for step in steps:
            if not isinstance(step, dict):
                raise ValueError(f"function_step_contract_invalid:{key}")
            has_legacy = "state_id" in step
            has_current = "source_state_id" in step
            if has_legacy == has_current:
                raise ValueError(f"function_step_state_field_invalid:{key}")
            if has_legacy:
                step["source_state_id"] = step.pop("state_id")
                migrated_steps += 1
            action = step.get("action")
            args = action.get("args") if isinstance(action, dict) else None
            if isinstance(args, dict) and "wait_after_s" in args:
                args.pop("wait_after_s")
                removed_wait_arguments += 1
            if (
                isinstance(action, dict)
                and action.get("tool") == "swipe"
                and isinstance(args, dict)
                and "direction" not in args
            ):
                args["direction"] = _swipe_direction(args)
                derived_swipe_directions += 1
        function = parse_function_artifact(migrated_function)
        if str(key) != function.id:
            raise ValueError(f"function_store_key_mismatch:{key}")
        output_functions[function.id] = function.to_dict()
    return {
        "schema_version": "omniflow.store.v2",
        "functions": output_functions,
    }, {
        "migrated_step_count": migrated_steps,
        "removed_wait_after_s_count": removed_wait_arguments,
        "derived_swipe_direction_count": derived_swipe_directions,
    }


def _swipe_direction(args: dict[str, Any]) -> str:
    values = [args.get(name) for name in ("x1", "y1", "x2", "y2")]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ValueError("legacy_swipe_coordinates_required_for_direction")
    x1, y1, x2, y2 = (float(value) for value in values)
    horizontal = x2 - x1
    vertical = y2 - y1
    if horizontal == 0 and vertical == 0:
        raise ValueError("legacy_swipe_direction_ambiguous")
    if abs(horizontal) > abs(vertical):
        return "right" if horizontal > 0 else "left"
    return "down" if vertical > 0 else "up"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate legacy Function state_id fields and deprecated wait metadata."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"input_store_missing:{source}")
    if source == output:
        raise ValueError("in_place_migration_forbidden")
    migrated, migration_counts = migrate_store(
        json.loads(source.read_text(encoding="utf-8"))
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(migrated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "omniflow.function_store_legacy_migration.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "migrations": [
            "state_id_to_source_state_id",
            "remove_deprecated_wait_after_s",
            "derive_swipe_direction_from_frozen_endpoints",
        ],
        "source_path": str(source),
        "source_sha256": _sha256(source),
        "output_path": str(output),
        "output_sha256": _sha256(output),
        "function_ids": sorted(migrated["functions"]),
        **migration_counts,
    }
    manifest_path = output.parent / "provenance_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
