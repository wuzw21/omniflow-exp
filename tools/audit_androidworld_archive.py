#!/usr/bin/env python3
"""Audit the canonical AndroidWorld archive and refresh its status documents."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from src.experiment.data_index import _require_qualified_source_run_log


FORMAL_METHODS = (
    "fixed_replay",
    "omniflow",
    "mobilegpt",
    "appagent",
    "t3a_hint",
)
DEVICE_KINDS = ("small", "fold")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
CELL_HEADERS = (
    ("fixed_replay", "small", "Fixed·Small"),
    ("fixed_replay", "fold", "Fixed·Fold"),
    ("omniflow", "small", "OmniFlow·Small"),
    ("omniflow", "fold", "OmniFlow·Fold"),
    ("mobilegpt", "small", "MobileGPT·Small"),
    ("mobilegpt", "fold", "MobileGPT·Fold"),
    ("appagent", "small", "AppAgent·Small"),
    ("appagent", "fold", "AppAgent·Fold"),
    ("t3a_hint", "small", "T3A·Small"),
    ("t3a_hint", "fold", "T3A·Fold"),
)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_success(payload: dict[str, Any]) -> bool:
    validator = payload.get("validator")
    validator = validator if isinstance(validator, dict) else {}
    androidworld = payload.get("androidworld")
    androidworld = androidworld if isinstance(androidworld, dict) else {}
    nested_validator = androidworld.get("validator")
    nested_validator = (
        nested_validator if isinstance(nested_validator, dict) else {}
    )
    return bool(
        payload.get("success") is True
        or validator.get("success") is True
        or validator.get("official_success") is True
        or nested_validator.get("success") is True
        or nested_validator.get("official_success") is True
    )


def observations(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    final = payload.get("final_observation")
    if isinstance(final, dict):
        yield final
    for step in payload.get("steps") or ():
        if not isinstance(step, dict):
            continue
        for key in (
            "observation",
            "next_observation",
            "observation_before_act",
            "observation_after_act",
        ):
            value = step.get(key)
            if isinstance(value, dict):
                yield value


def has_xml_evidence(payload: dict[str, Any]) -> bool:
    for observation in observations(payload):
        if observation.get("ui_elements"):
            return True
        forest = observation.get("forest")
        if isinstance(forest, dict) and forest.get("windows"):
            return True
        stack: list[Any] = [observation]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                for key, child in value.items():
                    if "xml" in str(key).lower() and child:
                        return True
                    stack.append(child)
            elif isinstance(value, list):
                stack.extend(value)
    return False


def image_references(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for child in value.values():
            yield from image_references(child)
    elif isinstance(value, list):
        for child in value:
            yield from image_references(child)
    elif isinstance(value, str) and Path(value).suffix.lower() in IMAGE_SUFFIXES:
        yield value


def resolved_reference(reference: str, run_log: Path) -> Path:
    path = Path(reference).expanduser()
    return path if path.is_absolute() else run_log.parent / path


def finished_at(payload: dict[str, Any], path: Path) -> str:
    for key in ("finished_at_ms", "started_at_ms"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(
                float(value) / 1000.0, tz=timezone.utc
            ).isoformat().replace("+00:00", "Z")
    for key in ("finished_at", "created_at", "timestamp"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def device_kind(device: str) -> str | None:
    lowered = device.lower()
    if "targetfold" in lowered:
        return "fold"
    if "targetsmall" in lowered:
        return "small"
    return None


def authoritative_rows(inspect_path: Path) -> list[list[str]]:
    with inspect_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith('{"kind":"table"'):
                continue
            value = json.loads(line)
            rows = value.get("values")
            if isinstance(rows, list) and len(rows) == 117:
                return [[str(cell or "") for cell in row] for row in rows]
    raise ValueError(f"authoritative_116x10_table_not_found:{inspect_path}")


def scan_runlogs(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in sorted(root.rglob("run_log.json")):
        relative = path.relative_to(root)
        if ".archive" in relative.parts:
            continue
        if len(relative.parts) < 5:
            errors.append(f"noncanonical_path:{relative}")
            continue
        task, method, device = relative.parts[:3]
        payload = read_json(path)
        if payload is None:
            errors.append(f"invalid_json:{relative}")
            continue
        images = sum(
            child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
            for child in path.parent.rglob("*")
        )
        references = list(image_references(payload))
        existing_references = sum(
            resolved_reference(reference, path).is_file()
            for reference in references
        )
        xml = has_xml_evidence(payload)
        observation_count = sum(1 for _ in observations(payload))
        success = is_success(payload)
        strict_ready = False
        if method == "source":
            try:
                _require_qualified_source_run_log(
                    payload,
                    task=task,
                    source_metadata=None,
                )
                strict_ready = True
            except (TypeError, ValueError, FileNotFoundError):
                pass
        records.append(
            {
                "task": task,
                "method": method,
                "device": device,
                "device_kind": device_kind(device),
                "success": success,
                "images": images,
                "image_references": len(references),
                "existing_image_references": existing_references,
                "images_complete": bool(references)
                and existing_references == len(references),
                "xml": xml,
                "observations": observation_count,
                "evidence_ready": bool(
                    success
                    and references
                    and existing_references == len(references)
                    and xml
                    and observation_count > 0
                ),
                "strict_ready": strict_ready,
                "finished_at": finished_at(payload, path),
                "path": path,
                "sha256": sha256_file(path),
            }
        )
    return records, errors


def status_token(records: list[dict[str, Any]]) -> str:
    if not records:
        return "—"
    passed = sum(bool(record["success"]) for record in records)
    failed = len(records) - passed
    images = sum(bool(record["images_complete"]) for record in records)
    xml = sum(bool(record["xml"]) for record in records)
    return f"P{passed}/F{failed} I{images} X{xml}"


def absolute_paths(value: Any) -> Iterable[Path]:
    if isinstance(value, dict):
        for child in value.values():
            yield from absolute_paths(child)
    elif isinstance(value, list):
        for child in value:
            yield from absolute_paths(child)
    elif isinstance(value, str) and value.startswith("/"):
        yield Path(value).expanduser()


def write_documents(
    *,
    root: Path,
    records: list[dict[str, Any]],
    errors: list[str],
    rows: list[list[str]],
    authority_path: Path,
) -> None:
    workbook_name = authority_path.name.removesuffix(".inspect.ndjson")
    workbook_path = authority_path.with_name(workbook_name)
    result_groups_path = authority_path.parent / "RESULT_GROUPS.md"
    headers = rows[0]
    task_rows = rows[1:]
    tasks = [row[0] for row in task_rows]
    authority_completed = sum(
        "完成：是" in cell for row in task_rows for cell in row[1:11]
    )
    authority_passed = sum(
        "完成：是" in cell and "通过：是" in cell
        for row in task_rows
        for cell in row[1:11]
    )
    authority_unrun = 1160 - authority_completed
    table_by_cell: dict[tuple[str, str, str], str] = {}
    for row in task_rows:
        for column, (method, kind, _) in enumerate(CELL_HEADERS, start=1):
            table_by_cell[(row[0], method, kind)] = row[column]

    formal = [record for record in records if record["method"] in FORMAL_METHODS]
    sources = [record for record in records if record["method"] == "source"]
    evidence_ready_sources = [record for record in sources if record["evidence_ready"]]
    strict_ready_sources = [record for record in sources if record["strict_ready"]]
    by_cell: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in formal:
        kind = record["device_kind"]
        if kind in DEVICE_KINDS:
            by_cell[(record["task"], record["method"], kind)].append(record)
    local_cells = {key for key, value in by_cell.items() if value}
    local_pass_cells = {
        key for key, value in by_cell.items() if any(row["success"] for row in value)
    }
    authority_complete_cells = {
        key for key, value in table_by_cell.items() if "完成：是" in value
    }
    authority_missing_local = authority_complete_cells - local_cells
    data_root = root.parent
    current = read_json(data_root / "current.json") or {}
    current_source_index = current.get("source_index")
    current_source_index = (
        current_source_index if isinstance(current_source_index, dict) else {}
    )
    records_by_path = {str(record["path"].resolve()): record for record in records}
    indexed_source_paths = []
    indexed_source_contract_accepted = 0
    indexed_source_evidence_ready = 0
    for task, metadata in current_source_index.items():
        if not isinstance(metadata, dict):
            continue
        value = str(metadata.get("retained_source_run_log") or "").strip()
        if not value:
            continue
        path = Path(value).expanduser().resolve()
        indexed_source_paths.append(path)
        record = records_by_path.get(str(path))
        payload = read_json(path)
        if payload is None:
            continue
        if record and record["evidence_ready"]:
            indexed_source_evidence_ready += 1
        elif record is None:
            references = list(image_references(payload))
            references_complete = bool(references) and all(
                resolved_reference(reference, path).is_file()
                for reference in references
            )
            if (
                is_success(payload)
                and references_complete
                and has_xml_evidence(payload)
                and any(True for _ in observations(payload))
            ):
                indexed_source_evidence_ready += 1
        try:
            _require_qualified_source_run_log(
                payload,
                task=str(task),
                source_metadata=metadata,
            )
            indexed_source_contract_accepted += 1
        except (TypeError, ValueError, FileNotFoundError):
            pass
    missing_current_paths = sorted(
        str(path)
        for path in absolute_paths(current)
        if str(data_root) in str(path) and not path.exists()
    )
    raw_object_store = root / ".archive" / "object_store"
    raw_object_files = sum(path.is_file() for path in raw_object_store.rglob("*"))
    provenance_sidecars = []
    provenance_source_hashes: set[str] = set()
    missing_provenance_objects: list[str] = []
    for path in root.rglob("provenance.json"):
        if ".archive" in path.relative_to(root).parts:
            continue
        payload = read_json(path) or {}
        source_hash = str(payload.get("source_sha256") or "").strip()
        source_object = Path(str(payload.get("source_object") or "")).expanduser()
        if source_hash:
            provenance_sidecars.append(path)
            provenance_source_hashes.add(source_hash)
            if not source_object.is_file():
                missing_provenance_objects.append(str(source_object))
    visible_images = sum(
        path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        for path in root.rglob("*")
        if ".archive" not in path.relative_to(root).parts
    )
    duplicate_runlog_copies = len(records) - len(
        {str(record["sha256"]) for record in records}
    )
    data_top_level = sorted(path.name for path in data_root.iterdir())
    bmoca_files = sum(path.is_file() for path in (data_root / "bmoca").rglob("*"))

    summary = [
        "# AndroidWorld 10-cell RunLog 完成情况",
        "",
        "此文档以桌面权威 116×10 主表确定任务和 cell 顺序，以当前 `data/androidworld` 中实际可读取的 RunLog 作为可复用证据。主表历史结论与本地 RunLog 证据分开统计。",
        "",
        f"- 权威主表：{len(tasks)} 个任务 × 10 cells = 1160 cells；已跑 {authority_completed}，未跑/不可用 {authority_unrun}，通过 {authority_passed}。",
        f"- 本地正式方法 RunLog：{len(formal)} 份，覆盖 {len(local_cells)} 个 cells；其中 {len(local_pass_cells)} 个 cells 至少有一份成功 RunLog。",
        f"- 主表标记已跑、但当前归档没有本地 RunLog 的 cells：{len(authority_missing_local)}。这些只能保留为历史结果证据，不能冒充可转换 memory 的完整 RunLog。",
        f"- Source RunLog：{len(sources)} 份，其中成功 {sum(bool(row['success']) for row in sources)}；截图引用完整 {sum(bool(row['images_complete']) for row in sources)}；带 XML/UI tree {sum(bool(row['xml']) for row in sources)}。",
        f"- 截图/XML 证据完整的成功 Source RunLog：{len(evidence_ready_sources)} 份，覆盖 {len({row['task'] for row in evidence_ready_sources})}/116 个任务。",
        f"- 无历史兼容豁免即可通过当前严格 source 合同：{len(strict_ready_sources)} 份，覆盖 {len({row['task'] for row in strict_ready_sources})}/116 个任务。",
        f"- `data/current.json` 已选择 source 的任务：{len(indexed_source_paths)}/116；合同可接受 {indexed_source_contract_accepted}，其中截图/XML 证据完整 {indexed_source_evidence_ready}。",
        f"- 全部可见 RunLog：{len(records)} 份；JSON 读取错误/目录违规 {len(errors)}。",
        f"- 去重检查：内容完全相同的可见 RunLog 副本 {duplicate_runlog_copies}；隐藏原对象库 {raw_object_files} 个文件；可见截图 {visible_images} 张。",
        f"- 原对象 RunLog 迁移映射：{len(provenance_sidecars)} 份 provenance / {len(provenance_source_hashes)} 个唯一原始 SHA；丢失原对象 {len(missing_provenance_objects)}。",
        f"- `data/current.json` 指向仓库 `data/` 内但已不存在的路径：{len(missing_current_paths)}。B-MoCA 保留 {bmoca_files} 个文件。",
        "",
        "单元格格式：`P成功份数/F失败份数 I截图引用完整份数 X带XML或UI树份数`；`—` 表示当前归档没有 RunLog。多份成功记录全部保留，以便选择截图/XML 最完整的一组。",
        "",
        "| Task | " + " | ".join(label for _, _, label in CELL_HEADERS) + " |",
        "|---|" + "---:|" * len(CELL_HEADERS),
    ]
    for task in tasks:
        tokens = [status_token(by_cell[(task, method, kind)]) for method, kind, _ in CELL_HEADERS]
        summary.append(f"| {task} | " + " | ".join(tokens) + " |")
    summary.extend(
        [
            "",
            "## 路径和口径",
            "",
            f"- 权威主表：`{workbook_path}`",
            f"- 主表机器审计：`{authority_path}`",
            f"- 历史结果组说明：`{result_groups_path}`",
            "- 详细到每份 RunLog 的路径、时间、截图/XML 情况见 `RUNLOG_INDEX.md`。",
            "- `source` 和补充 `autodroid` 不占 10-cell 主矩阵；B-MoCA 始终位于 `data/bmoca`。",
        ]
    )
    (root / "COMPLETION_STATUS.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    index = [
        "# AndroidWorld RunLog 索引",
        "",
        "所有非重复、可读取的 RunLog 都列在这里；成功版本不会因同一 setting 还有其他版本而被覆盖。",
        "",
        "| Task | Method | Device model + seed | Status | Image files | Image refs | XML/UI | Ready | Observations | Finished | SHA-256 | RunLog |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for record in sorted(
        records,
        key=lambda row: (
            row["task"], row["method"], row["device"], row["finished_at"], str(row["path"])
        ),
    ):
        relative = record["path"].relative_to(root)
        index.append(
            "| {task} | {method} | {device} | {status} | {images} | {image_refs} | {xml} | {ready} | {observations} | {finished} | `{sha}` | `{path}` |".format(
                task=record["task"],
                method=record["method"],
                device=record["device"],
                status="PASS" if record["success"] else "FAIL",
                images=record["images"],
                image_refs=(
                    f"{record['existing_image_references']}/{record['image_references']}"
                ),
                xml="yes" if record["xml"] else "no",
                ready=(
                    "strict"
                    if record["strict_ready"]
                    else "evidence"
                    if record["evidence_ready"]
                    else "no"
                ),
                observations=record["observations"],
                finished=record["finished_at"],
                sha=record["sha256"],
                path=relative,
            )
        )
    if errors:
        index.extend(["", "## 审计错误", "", *[f"- `{error}`" for error in errors]])
    (root / "RUNLOG_INDEX.md").write_text("\n".join(index) + "\n", encoding="utf-8")

    source_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in sources:
        source_by_task[record["task"]].append(record)
    memory_doc = [
        "# AndroidWorld memory-ready Source RunLogs",
        "",
        "`STRICT` 表示无需历史兼容豁免即可通过当前 source 合同；`EVIDENCE` 表示官方成功、全部截图引用可读取、包含 XML/UI tree 和 native observation，但仍需 canonical adapter 补齐旧 schema/reasoning。每个任务展示证据最完整且较新的一个候选；其他成功版本仍保留在 `RUNLOG_INDEX.md`。",
        "",
        "| Task | Status | Successful versions | Image refs | XML/UI | Candidate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for task in tasks:
        candidates = [row for row in source_by_task.get(task, []) if row["success"]]
        candidates.sort(
            key=lambda row: (
                bool(row["strict_ready"]),
                bool(row["evidence_ready"]),
                bool(row["images_complete"]),
                bool(row["xml"]),
                int(row["existing_image_references"]),
                row["finished_at"],
            ),
            reverse=True,
        )
        best = candidates[0] if candidates else None
        if best is None:
            memory_doc.append(f"| {task} | MISSING | 0 | 0/0 | no | — |")
            continue
        relative = best["path"].relative_to(root)
        memory_doc.append(
            f"| {task} | "
            f"{'STRICT' if best['strict_ready'] else 'EVIDENCE' if best['evidence_ready'] else 'INCOMPLETE'} | "
            f"{len(candidates)} | {best['existing_image_references']}/{best['image_references']} | "
            f"{'yes' if best['xml'] else 'no'} | `{relative}` |"
        )
    (root / "MEMORY_READY_SOURCES.md").write_text(
        "\n".join(memory_doc) + "\n", encoding="utf-8"
    )

    audit = {
        "schema_version": "omniflow.androidworld.archive_audit.v1",
        "generated_at": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "authority_workbook": str(workbook_path),
        "authority_workbook_sha256": sha256_file(workbook_path),
        "authority_inspect": str(authority_path),
        "authority_inspect_sha256": sha256_file(authority_path),
        "authority_result_groups": str(result_groups_path),
        "authority_result_groups_sha256": sha256_file(result_groups_path),
        "tasks": len(tasks),
        "cells": 1160,
        "authority_completed_cells": authority_completed,
        "authority_unrun_or_unavailable_cells": authority_unrun,
        "authority_passed_cells": authority_passed,
        "local_runlogs": len(records),
        "local_formal_runlogs": len(formal),
        "local_source_runlogs": len(sources),
        "evidence_ready_source_runlogs": len(evidence_ready_sources),
        "tasks_with_evidence_ready_source": len(
            {record["task"] for record in evidence_ready_sources}
        ),
        "strict_ready_source_runlogs": len(strict_ready_sources),
        "tasks_with_strict_ready_source": len(
            {record["task"] for record in strict_ready_sources}
        ),
        "current_selected_source_tasks": len(indexed_source_paths),
        "current_contract_accepted_source_tasks": indexed_source_contract_accepted,
        "current_evidence_ready_source_tasks": indexed_source_evidence_ready,
        "local_cells_with_runlog": len(local_cells),
        "local_cells_with_successful_runlog": len(local_pass_cells),
        "authority_completed_cells_without_local_runlog": len(authority_missing_local),
        "duplicate_visible_runlog_copies": duplicate_runlog_copies,
        "raw_object_store_files": raw_object_files,
        "migrated_runlog_provenance_sidecars": len(provenance_sidecars),
        "unique_migrated_source_runlog_hashes": len(provenance_source_hashes),
        "missing_migrated_source_objects": missing_provenance_objects,
        "visible_image_files": visible_images,
        "bmoca_files": bmoca_files,
        "data_top_level": data_top_level,
        "missing_current_data_paths": missing_current_paths,
        "invalid_or_noncanonical": errors,
        "method_counts": Counter(record["method"] for record in records),
    }
    (root / "ARCHIVE_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True, default=dict) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/androidworld"))
    parser.add_argument(
        "--authority-inspect",
        type=Path,
        default=Path.home()
        / "Desktop/OmniFlow-AndroidWorld-Experiments/OmniFlow_AndroidWorld_116Tasks_10cell.xlsx.inspect.ndjson",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    authority = args.authority_inspect.resolve()
    rows = authoritative_rows(authority)
    records, errors = scan_runlogs(root)
    write_documents(
        root=root,
        records=records,
        errors=errors,
        rows=rows,
        authority_path=authority,
    )
    print(
        json.dumps(
            {"runlogs": len(records), "errors": len(errors)},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
