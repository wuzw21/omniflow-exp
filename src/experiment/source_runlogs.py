from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from omniflow.core.trajectory import require_complete_source_run_log
from omniflow.transfer.runtime import load_transfer_state_catalog
from src.integrations.runlog import adapt_source_run_log

SOURCE_INDEX_SCHEMA = "omniflow.androidworld.source_run_log_index.v1"
CONVERSION_MANIFEST_SCHEMA = "omniflow.androidworld.run_log_conversion.v1"


def convert_source_index(
    *,
    source_index: str | Path,
    output_root: str | Path,
    screenshot_roots: Sequence[str | Path],
    tasks: Sequence[str] = (),
) -> dict[str, Any]:
    """Convert a complete legacy source index into immutable OmniFlow RunLogs."""
    index_path = Path(source_index).expanduser().resolve()
    if not index_path.is_file():
        raise FileNotFoundError(f"source_index_missing:{index_path}")
    destination = Path(output_root).expanduser().resolve()
    if not destination.is_absolute():
        raise ValueError("source_run_log_output_root_must_be_absolute")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"immutable_source_run_log_output_exists:{destination}"
        )
    source_payload = _load_object(index_path)
    selected_tasks = tuple(dict.fromkeys(str(task).strip() for task in tasks))
    if any(not task for task in selected_tasks):
        raise ValueError("source_run_log_task_filter_invalid")
    missing_tasks = sorted(set(selected_tasks) - set(source_payload))
    if missing_tasks:
        raise ValueError(
            "source_run_log_tasks_missing:" + ",".join(missing_tasks)
        )
    items = (
        [(task, source_payload[task]) for task in selected_tasks]
        if selected_tasks
        else sorted(source_payload.items())
    )
    converted: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    source_catalogs: dict[str, tuple[Path, str]] = {}
    failures: dict[str, str] = {}
    for task, raw_item in items:
        try:
            if not isinstance(raw_item, dict):
                raise ValueError("source_index_item_must_be_object")
            source_path = _resolve_source_path(
                index_path,
                raw_item.get("retained_source_run_log")
                or raw_item.get("source_run_log"),
            )
            raw_run_log = _load_object(source_path)
            source_states = None
            source_catalog_value = raw_item.get(
                "source_state_catalog"
            ) or raw_item.get("transfer_state_catalog")
            if source_catalog_value:
                source_catalog_path = _resolve_source_path(
                    index_path,
                    source_catalog_value,
                )
                source_catalog_sha256 = hashlib.sha256(
                    source_catalog_path.read_bytes()
                ).hexdigest()
                expected_catalog_sha256 = str(
                    raw_item.get("source_state_catalog_sha256")
                    or raw_item.get("transfer_state_catalog_sha256")
                    or ""
                ).strip()
                if (
                    expected_catalog_sha256
                    and expected_catalog_sha256 != source_catalog_sha256
                ):
                    raise ValueError(
                        "source_state_catalog_hash_mismatch:"
                        f"{task}:expected={expected_catalog_sha256}:"
                        f"actual={source_catalog_sha256}"
                    )
                source_states = load_transfer_state_catalog(
                    source_catalog_path
                )
                source_catalogs[str(task)] = (
                    source_catalog_path,
                    source_catalog_sha256,
                )
            run_log = adapt_source_run_log(
                raw_run_log,
                task_name=str(task),
                task_parameters=dict(raw_item.get("params") or {}),
                seed=_source_seed(raw_item),
                source_path=source_path,
                source_states=source_states,
                screenshot_roots=screenshot_roots,
                require_screenshots=False,
            )
            require_complete_source_run_log(run_log)
            converted[str(task)] = (run_log, dict(raw_item))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            failures[str(task)] = str(error)
    if failures:
        details = ";".join(
            f"{task}={failures[task]}" for task in sorted(failures)
        )
        raise ValueError(
            f"source_run_log_conversion_failed:{len(failures)}/{len(items)}:{details}"
        )

    object_payloads: dict[str, bytes] = {}
    index: dict[str, dict[str, Any]] = {}
    for task in sorted(converted):
        run_log, original_item = converted[task]
        encoded = _stable_json_bytes(run_log)
        digest = hashlib.sha256(encoded).hexdigest()
        object_path = destination / "objects" / f"{digest}.run_log.json"
        object_payloads.setdefault(digest, encoded)
        index[task] = {
            "task": task,
            "goal": run_log["goal"],
            "params": run_log["task_parameters"],
            "source_seed": run_log["seed"],
            "source_kind": "androidworld_validator_success_source_runlog",
            "latest_official_success_source": True,
            "retained_source_run_log": str(object_path),
            "retained_source_run_log_sha256": digest,
            "step_count": len(run_log["steps"]),
            "legacy_index_metadata": {
                key: original_item[key]
                for key in (
                    "collect_seed",
                    "replay_seed",
                    "task_random_seed",
                    "run_id",
                )
                if key in original_item
            },
        }
        if task in source_catalogs:
            catalog_path, catalog_sha256 = source_catalogs[task]
            index[task]["source_state_catalog"] = str(catalog_path)
            index[task]["source_state_catalog_sha256"] = catalog_sha256
    destination.mkdir(parents=True, exist_ok=True)
    object_dir = destination / "objects"
    object_dir.mkdir()
    for digest, encoded in sorted(object_payloads.items()):
        _write_immutable(object_dir / f"{digest}.run_log.json", encoded)
    index_path_out = destination / "index_by_task.json"
    _write_immutable(index_path_out, _stable_json_bytes(index))
    manifest = {
        "schema_version": CONVERSION_MANIFEST_SCHEMA,
        "source_index": str(index_path),
        "source_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "source_index_schema": SOURCE_INDEX_SCHEMA,
        "output_index": str(index_path_out),
        "output_index_sha256": hashlib.sha256(index_path_out.read_bytes()).hexdigest(),
        "task_count": len(index),
        "unique_object_count": len(object_payloads),
        **_screenshot_coverage(converted),
        "screenshot_roots": [
            str(Path(root).expanduser().resolve()) for root in screenshot_roots
        ],
    }
    manifest_path = destination / "manifest.json"
    _write_immutable(manifest_path, _stable_json_bytes(manifest))
    return {
        **manifest,
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }


def _screenshot_coverage(
    converted: dict[str, tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    observation_count = 0
    screenshot_reference_count = 0
    tasks_with_missing_screenshots: list[str] = []
    for task, (run_log, _) in sorted(converted.items()):
        task_missing = False
        for step in run_log["steps"]:
            observation_count += 1
            if step["observation"].get("pixels") is None:
                task_missing = True
            else:
                screenshot_reference_count += 1
        if task_missing:
            tasks_with_missing_screenshots.append(task)
    return {
        "observation_count": observation_count,
        "screenshot_reference_count": screenshot_reference_count,
        "missing_screenshot_reference_count": (
            observation_count - screenshot_reference_count
        ),
        "tasks_with_missing_screenshot_references": tasks_with_missing_screenshots,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert legacy AndroidWorld evidence to the one OmniFlow RunLog schema."
    )
    parser.add_argument("--source-index", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--screenshot-root", action="append", default=[])
    parser.add_argument("--task", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = convert_source_index(
        source_index=args.source_index,
        output_root=args.output_root,
        screenshot_roots=args.screenshot_root,
        tasks=args.task,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def _resolve_source_path(index_path: Path, value: Any) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError("source_index_run_log_required")
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        if not candidate.is_file():
            raise FileNotFoundError(f"indexed_source_run_log_missing:{candidate}")
        return candidate.resolve()
    matches = {
        (ancestor / candidate).resolve()
        for ancestor in (index_path.parent, *index_path.parents)
        if (ancestor / candidate).is_file()
    }
    if len(matches) != 1:
        raise ValueError(
            f"indexed_source_run_log_resolution_failed:{text}:matches={len(matches)}"
        )
    return next(iter(matches))


def _source_seed(item: dict[str, Any]) -> int | None:
    for key in ("task_random_seed", "collect_seed", "replay_seed"):
        value = item.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _stable_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


if __name__ == "__main__":
    raise SystemExit(main())
