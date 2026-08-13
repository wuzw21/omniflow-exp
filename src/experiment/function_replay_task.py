"""Run one Function task through source, semantic, and target replay gates."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

PROBE = Path.home() / ".codex" / "skills" / "androidworld-runlog-harvester" / "scripts" / "run_offline_function_probe.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _device(value: str) -> tuple[str, str, str]:
    parts = str(value).split(":")
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"device_invalid:{value}")
    return parts[0], parts[1], parts[2]


def _run_probe(
    args: argparse.Namespace,
    *,
    spec: Path,
    version: str,
    device: str,
    attempt_id: str,
    source: bool,
) -> dict[str, Any]:
    label, serial, port = _device(device)
    command = [
        sys.executable,
        str(PROBE),
        "--repo",
        str(args.repo),
        "--android-world-root",
        str(args.android_world_root),
        "--task",
        args.task,
        "--version",
        version,
        "--replay-spec",
        str(spec),
        "--device",
        device,
        "--adb-path",
        str(args.adb_path),
        "--python-bin",
        str(args.python_bin),
        "--omnitransfer-root",
        str(args.omnitransfer_root),
        "--attempt-id",
        attempt_id,
        "--timeout-sec",
        str(args.timeout_sec),
    ]
    if source:
        command.append("--source-qualification")
    result = subprocess.run(
        command,
        cwd=args.repo,
        text=True,
        capture_output=True,
        check=False,
    )
    evidence = (
        args.repo
        / "data"
        / args.task
        / ("source_attempts" if source else "attempts")
        / version
        / attempt_id
        / label
    )
    receipt = evidence / "probe_summary.json"
    payload = _read(receipt) if receipt.is_file() else {
        "classification": "harness_failed",
        "returncode": result.returncode,
        "stderr": result.stderr[-4000:],
    }
    payload["command_returncode"] = result.returncode
    payload["evidence_path"] = str(evidence)
    return payload


def _validate_specs(source_spec: dict[str, Any], target_spec: dict[str, Any], targets: list[str]) -> None:
    if int(source_spec.get("evaluation", {}).get("seed", -1)) != 111:
        raise ValueError("source_seed_must_be_111")
    if int(target_spec.get("evaluation", {}).get("seed", -1)) != 113:
        raise ValueError("target_seed_must_be_113")
    expected = ["small5554:emulator-5554:5554", "fold5564:emulator-5564:5564"]
    if targets != expected:
        raise ValueError("targets_must_be_smallphone_and_fold")
    target_devices = target_spec.get("target_devices") or []
    actual = [f"{item.get('label')}:{item.get('serial')}:{item.get('console_port')}" for item in target_devices]
    if actual != expected:
        raise ValueError("target_spec_devices_mismatch")


def _same_actions(left: dict[str, Any], right: dict[str, Any]) -> bool:
    def actions(value: dict[str, Any]) -> list[dict[str, Any]]:
        function = (value.get("functions") or {}).get(str(value.get("function_id") or ""), {})
        return [step.get("action") for step in function.get("steps") or []]

    return actions(left) == actions(right)


def _semantic_compile(
    args: argparse.Namespace,
    source_run_log: Path,
    source_spec: dict[str, Any],
    target_spec: dict[str, Any],
    target_version: str,
) -> Path:
    from omniflow.functions.compiler import compile_runlog_to_store
    from omniflow.functions.store import FunctionStore

    output_root = args.semantic_output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"semantic_output_exists:{output_root}")
    model = args.model.strip()
    if not model:
        raise ValueError("semantic_model_required_unless_skipped")
    if os.environ.get("LLMTHU_KEY") and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["LLMTHU_KEY"]
    if os.environ.get("LLMTHU_BASE_URL") and not os.environ.get("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = os.environ["LLMTHU_BASE_URL"]
    report = compile_runlog_to_store(
        source_run_log,
        output_root,
        model=model,
        source_states=source_run_log.parent / "target.transfer_states.json",
    )
    store = FunctionStore(report["store_path"])
    function = store.get_function(args.function_id)
    if function is None:
        raise ValueError(f"semantic_function_missing:{args.function_id}")
    original_store = FunctionStore(source_spec["store"]["path"])
    original = original_store.get_function(args.function_id)
    if original is None or [step.action.to_dict() for step in original.steps] != [step.action.to_dict() for step in function.steps]:
        raise ValueError("semantic_action_sequence_changed")
    spec = json.loads(json.dumps(target_spec, ensure_ascii=False))
    spec["version"] = target_version
    spec["function_id"] = function.id
    spec["function_arguments"] = {}
    spec["source_run_log_sha256"] = _sha256(source_run_log)
    spec["store"] = {"path": str(Path(report["store_path"]).resolve()), "sha256": _sha256(Path(report["store_path"]))}
    transfer = Path(report["transfer_state_catalog"]).resolve()
    spec["transfer_states"] = {"path": str(transfer), "sha256": _sha256(transfer)}
    replay_spec = output_root / "replay_spec.json"
    _write(replay_spec, spec)
    return replay_spec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--android-world-root", type=Path, required=True)
    parser.add_argument("--omnitransfer-root", type=Path, required=True)
    parser.add_argument("--adb-path", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--source-version", default="v0001")
    parser.add_argument("--target-version", default="v0001")
    parser.add_argument("--source-spec", type=Path, required=True)
    parser.add_argument("--target-spec", type=Path, required=True)
    parser.add_argument("--function-id", required=True)
    parser.add_argument("--source-device", default="source5560:emulator-5560:5560")
    parser.add_argument("--targets", default="small5554:emulator-5554:5554,fold5564:emulator-5564:5564")
    parser.add_argument("--model", default="")
    parser.add_argument("--semantic-output-root", type=Path)
    parser.add_argument("--skip-semantic", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--attempt-prefix", default="task_pipeline")
    args = parser.parse_args()
    args.repo = args.repo.expanduser().resolve()
    args.android_world_root = args.android_world_root.expanduser().resolve()
    args.omnitransfer_root = args.omnitransfer_root.expanduser().resolve()
    args.adb_path = args.adb_path.expanduser().resolve()
    args.python_bin = args.python_bin.expanduser().resolve()
    source_spec_path = args.source_spec.expanduser().resolve()
    target_spec_path = args.target_spec.expanduser().resolve()
    source_spec = _read(source_spec_path)
    target_spec = _read(target_spec_path)
    targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    _validate_specs(source_spec, target_spec, targets)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    source_attempt = f"{args.attempt_prefix}_source_{stamp}"
    source_result = _run_probe(
        args,
        spec=source_spec_path,
        version=args.source_version,
        device=args.source_device,
        attempt_id=source_attempt,
        source=True,
    )
    summary: dict[str, Any] = {"task": args.task, "source": source_result, "semantic": None, "targets": []}
    if source_result.get("classification") != "mature" or source_result.get("official_validator_success") is not True:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1
    source_run_log = Path(source_result["evidence_path"]) / "target.run_log.json"
    if not source_run_log.is_file():
        raise FileNotFoundError(f"source_run_log_missing:{source_run_log}")
    replay_spec = target_spec_path
    if not args.skip_semantic:
        if args.semantic_output_root is None:
            raise ValueError("semantic_output_root_required")
        replay_spec = _semantic_compile(
            args,
            source_run_log,
            source_spec,
            target_spec,
            args.target_version,
        )
        summary["semantic"] = {"status": "saved", "replay_spec": str(replay_spec), "model": args.model}
    else:
        summary["semantic"] = {"status": "skipped"}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(targets)) as pool:
        futures = [
            pool.submit(
                _run_probe,
                args,
                spec=replay_spec,
                version=args.target_version,
                device=device,
                attempt_id=f"{args.attempt_prefix}_target_{stamp}",
                source=False,
            )
            for device in targets
        ]
        summary["targets"] = [future.result() for future in futures]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(item.get("classification") == "mature" for item in summary["targets"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
