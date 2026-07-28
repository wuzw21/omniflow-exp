#!/usr/bin/env python3
"""Synchronize the replay-time OmniTransfer module into OmniFlow."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any


CORE_FILES = (
    "__init__.py",
    "learned_matcher.py",
    "mutual_matcher.py",
    "numpy_matcher.py",
    "runtime.py",
    "ui_graph.py",
    "checkpoints/pair_evidence_mutual_no_null_v3_20260723/README.md",
    "checkpoints/pair_evidence_mutual_no_null_v3_20260723/no_null_seed17.npz",
)
MANIFEST_NAME = "_SYNC_MANIFEST.json"
MANIFEST_SCHEMA = "omniflow.embedded_omnitransfer.v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path.home() / "Projects" / "Omni" / "OmniTransfer",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    destination = repo_root / "omnitransfer"
    if args.check:
        manifest = _load_manifest(destination / MANIFEST_NAME)
        _verify_files(destination, manifest["files"])
        source_package = args.source.expanduser().resolve() / "src" / "omnitransfer"
        source_checked = source_package.is_dir()
        if source_checked:
            _verify_files(source_package, manifest["files"])
        print(
            json.dumps(
                {
                    "ready": True,
                    "files": len(manifest["files"]),
                    "source_checked": source_checked,
                }
            )
        )
        return 0

    source_root = args.source.expanduser().resolve()
    source_package = source_root / "src" / "omnitransfer"
    missing = [name for name in CORE_FILES if not (source_package / name).is_file()]
    if missing:
        raise SystemExit("omnitransfer_core_missing:" + ",".join(missing))

    files: dict[str, dict[str, Any]] = {}
    for name in CORE_FILES:
        source = source_package / name
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files[name] = {
            "sha256": _sha256(source),
            "bytes": source.stat().st_size,
        }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "canonical_repository": "~/Projects/Omni/OmniTransfer",
        "source_repository": str(source_root),
        "source_revision": _git_revision(source_root),
        "source_worktree_dirty": _git_dirty(source_root),
        "files": files,
    }
    (destination / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _verify_files(destination, files)
    print(
        json.dumps(
            {
                "destination": str(destination),
                "files": len(files),
                "source_revision": manifest["source_revision"],
            }
        )
    )
    return 0


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"omnitransfer_manifest_missing:{path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != MANIFEST_SCHEMA
        or not isinstance(manifest.get("files"), dict)
    ):
        raise SystemExit("omnitransfer_manifest_invalid")
    return manifest


def _verify_files(root: Path, files: dict[str, Any]) -> None:
    for name, expected in files.items():
        path = root / name
        if not path.is_file():
            raise SystemExit(f"embedded_omnitransfer_file_missing:{name}")
        if _sha256(path) != str(expected.get("sha256") or ""):
            raise SystemExit(f"embedded_omnitransfer_checksum_mismatch:{name}")


def _git_revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_dirty(root: Path) -> bool:
    paths = [str(Path("src/omnitransfer") / name) for name in CORE_FILES]
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *paths],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
