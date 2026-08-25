#!/usr/bin/env python3
"""Validate and convert one successful RunLog into native MobileGPT memory.

This is an offline management tool.  It owns no episode, device, planner, or
memory-writing behavior; all normalization and persistence are delegated to
the canonical MobileGPT integration.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from src.experiment.mobilegpt_contract import MOBILEGPT_EMBEDDING_MODEL
from src.experiment.mobilegpt_source import convert_runlog_to_mobilegpt_bundle
from src.integrations.mobilegpt import preflight_runlog_conversion


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize a successful RunLog into MobileGPT's official action "
            "schema and optionally build a native memory bundle."
        )
    )
    parser.add_argument("--source-run-log", type=Path, required=True)
    parser.add_argument("--mobilegpt-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--target-package", default="")
    parser.add_argument("--target-app", default="")
    parser.add_argument(
        "--model",
        default=os.environ.get("MOBILEGPT_CHAT_MODEL", ""),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get(
            "MOBILEGPT_EMBEDDING_MODEL",
            MOBILEGPT_EMBEDDING_MODEL,
        ),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only emit the normalized source and official required actions.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    preflight = preflight_runlog_conversion(
        args.source_run_log,
        target_package=str(args.target_package or ""),
        target_app=str(args.target_app or ""),
        mobilegpt_root=args.mobilegpt_root,
    )
    if args.report is not None:
        _write_report(args.report, preflight)
    if preflight.get("ready") is not True:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 2
    if args.check_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0
    if args.output_root is None:
        parser.error("--output-root is required unless --check-only is used")
    model = str(args.model or "").strip()
    if not model:
        parser.error("--model or MOBILEGPT_CHAT_MODEL is required")
    result = convert_runlog_to_mobilegpt_bundle(
        source_run_log=args.source_run_log,
        mobilegpt_root=args.mobilegpt_root,
        output_root=args.output_root,
        model=model,
        embedding_model=str(args.embedding_model or MOBILEGPT_EMBEDDING_MODEL),
        target_package=str(args.target_package or ""),
        target_app=str(args.target_app or ""),
        preflight_audit={
            "schema_version": "omniflow.mobilegpt.source-check.v2",
            "grounding_source": "canonical_androidworld_run_log",
            "source_run_log": str(args.source_run_log.expanduser().resolve()),
            "actions_supplied_to_mobilegpt": True,
            "report": preflight,
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
