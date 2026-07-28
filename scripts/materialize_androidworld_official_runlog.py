from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.integrations.android_world.official_runlog import (  # noqa: E402
    materialize_m3a_episode_runlog,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task-results", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--android-world-root", required=True)
    args = parser.parse_args(argv)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    task_results_path = Path(args.task_results).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    android_world_root = Path(args.android_world_root).expanduser().resolve()
    if str(android_world_root) not in sys.path:
        sys.path.insert(0, str(android_world_root))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"immutable output directory is not empty: {output_dir}")
    rows = _jsonl_rows(task_results_path)
    matching_rows = [
        row
        for row in rows
        if str(row.get("task_name") or row.get("task") or "") == args.task
    ]
    if len(matching_rows) != 1:
        raise ValueError("official_task_result_not_unique")
    source_row = matching_rows[0]
    if source_row.get("official_validator_used") is not True:
        raise ValueError("official_validator_result_required")
    validator = source_row.get("androidworld_validator_result")
    if not isinstance(validator, dict) or validator.get("success") is not True:
        raise ValueError("successful_official_validator_result_required")

    with gzip.open(checkpoint_path, "rb") as handle:
        episodes = pickle.load(handle)
    matching_episodes = [
        episode
        for episode in episodes
        if isinstance(episode, dict)
        and str(episode.get("task_template") or "") == args.task
        and float(episode.get("is_successful") or 0.0) > 0.5
    ]
    if len(matching_episodes) != 1:
        raise ValueError("successful_official_checkpoint_episode_not_unique")

    run_log, transfer_catalog = materialize_m3a_episode_runlog(
        matching_episodes[0],
        task_name=args.task,
        goal=str(source_row.get("goal") or matching_episodes[0].get("goal") or ""),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = output_dir / "source.run_log.json"
    transfer_path = output_dir / "transfer_states.json"
    derived_result_path = output_dir / "task_results.jsonl"
    _write_json(run_log_path, run_log)
    _write_json(transfer_path, transfer_catalog)
    derived_result = {
        **source_row,
        "artifact_kind": "canonical_run",
        "artifact_ref": run_log["run_id"],
        "canonical_run": run_log,
        "transfer_state_catalog": str(transfer_path),
        "source_checkpoint_ref": str(checkpoint_path),
        "materialization": {
            "schema_version": "omniflow.androidworld_official_runlog_materialization.v1",
            "state_backend": "androidworld",
            "action_backend": "androidworld",
            "native_androidworld_agent_io": True,
            "display_source": "androidworld_raw_screenshot_shape",
        },
    }
    derived_result_path.write_text(
        json.dumps(derived_result, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "omniflow.androidworld_official_runlog_materialization.v1",
        "task_name": args.task,
        "immutable": True,
        "state_backend": "androidworld",
        "action_backend": "androidworld",
        "native_androidworld_agent_io": True,
        "display_source": "androidworld_raw_screenshot_shape",
        "source": {
            "android_world_root": str(android_world_root),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "task_results": str(task_results_path),
            "task_results_sha256": _sha256(task_results_path),
        },
        "outputs": {
            "run_log": str(run_log_path),
            "run_log_sha256": _sha256(run_log_path),
            "transfer_states": str(transfer_path),
            "transfer_states_sha256": _sha256(transfer_path),
            "task_results": str(derived_result_path),
            "task_results_sha256": _sha256(derived_result_path),
        },
    }
    _write_json(output_dir / "provenance_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
