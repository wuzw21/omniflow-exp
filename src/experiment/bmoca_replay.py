"""Offline B-MoCA replay comparison for the transfer-DP sidecar.

The existing replay path is not imported or modified here.  Each source/target
trace pair is measured twice: fixed sequence index is the control path and the
action-aware transfer DP is the sidecar.  Global actions remain directly
replayable; gaps only describe cross-trace correspondence and never execute a
source coordinate on the target device.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Callable, Iterable, Sequence
import xml.etree.ElementTree as ET

import numpy as np

from omniflow.core.model import Observation
from omniflow.transfer.embedding import PageEncoder
from omniflow.transfer.runtime import load_omnitransfer
from src.experiment.transfer_replay import (
    ReplayToken,
    TransferMatchScore,
    align_transfer_replay,
    retarget_transfer_score,
    score_transfer_match,
)

_CORPUS_VERSION = "omniflow.offline-trace-corpus.v1"
_LEGACY_RUNLOG_VERSION = "omniflow.canonical_run_log.v1"
_STATE_CATALOG_VERSION = "omniflow.transfer-state-catalog.v1"
_SELECTOR_ACTIONS = frozenset({"click", "input_text", "long_press"})
_DIRECT_ACTIONS = frozenset({"open_app", "press_key", "swipe", "wait"})
_BOUNDS_PATTERN = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)


@dataclass(frozen=True)
class BmocaStep:
    index: int
    action_kind: str
    xml: str = ""
    point: tuple[float, float] | None = None
    bounds: tuple[float, float, float, float] | None = None
    state_id: str = ""


@dataclass(frozen=True)
class BmocaTrace:
    trace_id: str
    task_id: str
    environment_id: str
    steps: tuple[BmocaStep, ...]


PairScorer = Callable[[BmocaStep, BmocaStep], TransferMatchScore]


def evaluate_trace_pair(
    source: BmocaTrace,
    target: BmocaTrace,
    *,
    scorer: PairScorer | None = None,
    candidate_cells: frozenset[tuple[int, int]] | None = None,
    executor: Executor | None = None,
) -> dict[str, Any]:
    """Measure the control and sidecar on one successful trace pair."""

    resolved_scorer = scorer or _score_step_pair
    scoring_started = time.perf_counter()
    evidence: list[list[TransferMatchScore]] = [
        [TransferMatchScore(reason="selector_endpoint_missing") for _ in target.steps]
        for _ in source.steps
    ]
    scored_cell_count = 0
    pending = []
    for source_position, source_step in enumerate(source.steps):
        for target_position, target_step in enumerate(target.steps):
            if source_step.point is None or target_step.bounds is None:
                continue
            elif (
                candidate_cells is not None
                and (source_position, target_position) not in candidate_cells
            ):
                evidence[source_position][target_position] = TransferMatchScore(
                    reason="coarse_candidate_pruned"
                )
            else:
                scored_cell_count += 1
                if executor is None:
                    evidence[source_position][target_position] = resolved_scorer(
                        source_step,
                        target_step,
                    )
                else:
                    pending.append(
                        (
                            source_position,
                            target_position,
                            executor.submit(
                                resolved_scorer,
                                source_step,
                                target_step,
                            ),
                        )
                    )
    for source_position, target_position, future in pending:
        evidence[source_position][target_position] = future.result()

    scoring_seconds = time.perf_counter() - scoring_started
    dp_started = time.perf_counter()
    alignment = align_transfer_replay(
        tuple(ReplayToken(step.index, step.action_kind) for step in source.steps),
        tuple(ReplayToken(step.index, step.action_kind) for step in target.steps),
        tuple(
            tuple(item.probability for item in row)
            for row in evidence
        ),
    )
    dp_seconds = time.perf_counter() - dp_started
    source_positions = {step.index: position for position, step in enumerate(source.steps)}
    target_positions = {step.index: position for position, step in enumerate(target.steps)}
    selector_positions = [
        position
        for position, step in enumerate(source.steps)
        if step.action_kind in _SELECTOR_ACTIONS and step.point is not None
    ]
    direct_action_count = sum(
        step.action_kind in _DIRECT_ACTIONS for step in source.steps
    )

    fixed_matches = []
    for source_position in selector_positions:
        source_step = source.steps[source_position]
        if source_position >= len(target.steps):
            fixed_matches.append(_missing_match(source_step, reason="target_step_missing"))
            continue
        target_step = target.steps[source_position]
        fixed_matches.append(
            _match_record(source_step, target_step, evidence[source_position][source_position])
        )

    dp_by_source = {pair.source_index: pair for pair in alignment.pairs}
    dp_matches = []
    for source_position in selector_positions:
        source_step = source.steps[source_position]
        pair = dp_by_source.get(source_step.index)
        if pair is None:
            dp_matches.append(_missing_match(source_step, reason="dp_correspondence_missing"))
            continue
        target_position = target_positions[pair.target_index]
        target_step = target.steps[target_position]
        dp_matches.append(
            _match_record(
                source_step,
                target_step,
                evidence[source_positions[pair.source_index]][target_position],
            )
        )

    return {
        "task_id": source.task_id,
        "source_trace_id": source.trace_id,
        "target_trace_id": target.trace_id,
        "source_environment_id": source.environment_id,
        "target_environment_id": target.environment_id,
        "source_step_count": len(source.steps),
        "target_step_count": len(target.steps),
        "replayed_action_count": len(source.steps),
        "selector_action_count": len(selector_positions),
        "direct_action_count": direct_action_count,
        "transfer_candidate_cell_count": scored_cell_count,
        "transfer_full_cell_count": len(selector_positions)
        * sum(step.bounds is not None for step in target.steps),
        "timing": {
            "transfer_scoring_seconds": scoring_seconds,
            "dp_decode_seconds": dp_seconds,
        },
        "fixed_index": _method_result(fixed_matches),
        "dp": {
            **_method_result(dp_matches),
            "protocol": alignment.protocol,
            "alignment_score": _finite_or_none(alignment.score),
            "pair_count": len(alignment.pairs),
            "source_gaps": [asdict(gap) for gap in alignment.source_gaps],
            "target_gaps": [asdict(gap) for gap in alignment.target_gaps],
        },
    }


def load_bmoca_traces(corpus_root: str | Path) -> tuple[BmocaTrace, ...]:
    """Load the explicit external B-MoCA corpus into the sidecar contract."""

    root = Path(corpus_root).expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = _read_object(manifest_path)
    if manifest.get("schema_version") != _CORPUS_VERSION:
        raise ValueError("unsupported_bmoca_corpus_version")
    records = manifest.get("traces")
    if not isinstance(records, list):
        raise ValueError("bmoca_corpus_traces_invalid")
    traces = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("bmoca_trace_record_invalid")
        evidence = record.get("success_evidence")
        if not isinstance(evidence, dict) or evidence.get("official_success") is not True:
            continue
        runlog_entry = record.get("runlog")
        state_entry = record.get("state_catalog")
        if not isinstance(runlog_entry, dict) or not isinstance(state_entry, dict):
            raise ValueError("bmoca_trace_assets_missing")
        runlog = _read_object(_asset_path(root, runlog_entry))
        catalog = _read_object(_asset_path(root, state_entry))
        traces.append(_load_trace(record, runlog, catalog))
    return tuple(traces)


def evaluate_bmoca_corpus(
    corpus_root: str | Path,
    *,
    target_environments: Sequence[str] = ("101", "105"),
    limit_tasks: int | None = None,
    workers: int = 1,
    accelerated: bool = True,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Evaluate every available env100-to-target successful trace pair."""

    traces = load_bmoca_traces(corpus_root)
    evaluation_started = time.perf_counter()
    grouped: dict[str, dict[str, list[BmocaTrace]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for trace in traces:
        grouped[trace.task_id][trace.environment_id].append(trace)
    tasks = sorted(grouped)
    if limit_tasks is not None:
        if limit_tasks <= 0:
            raise ValueError("limit_tasks_must_be_positive")
        tasks = tasks[:limit_tasks]
    if workers <= 0:
        raise ValueError("workers_must_be_positive")

    omnitransfer = load_omnitransfer()
    preflight = getattr(omnitransfer, "runtime_preflight", None)
    if callable(preflight):
        preflight()

    results = []
    missing = []
    score_cache: dict[
        tuple[
            str,
            tuple[float, float, float, float] | tuple[float, float] | None,
            str,
        ],
        TransferMatchScore,
    ] = {}
    page_encoder = PageEncoder()
    page_vectors: dict[str, np.ndarray | None] = {}
    cache_hits = 0
    cache_misses = 0
    candidate_search_seconds = 0.0
    cache_lock = threading.Lock()

    def cached_scorer(
        source_step: BmocaStep,
        target_step: BmocaStep,
    ) -> TransferMatchScore:
        nonlocal cache_hits, cache_misses
        key = (
            source_step.state_id,
            source_step.bounds or source_step.point,
            target_step.state_id,
        )
        with cache_lock:
            cached = score_cache.get(key)
        if cached is not None:
            with cache_lock:
                cache_hits += 1
            return retarget_transfer_score(
                cached,
                source_point=source_step.point or (0.0, 0.0),
                target_bounds=target_step.bounds or (0.0, 0.0, 1.0, 1.0),
            )
        scored = _score_step_pair_base(source_step, target_step)
        with cache_lock:
            existing = score_cache.get(key)
            if existing is None:
                score_cache[key] = scored
                cache_misses += 1
            else:
                scored = existing
                cache_hits += 1
        return retarget_transfer_score(
            scored,
            source_point=source_step.point or (0.0, 0.0),
            target_bounds=target_step.bounds or (0.0, 0.0, 1.0, 1.0),
        )

    def page_vector(step: BmocaStep) -> np.ndarray | None:
        cached = page_vectors.get(step.state_id)
        if cached is not None or step.state_id in page_vectors:
            return cached
        try:
            page = page_encoder.embed(Observation(xml=step.xml))
        except Exception:  # noqa: BLE001 - missing page evidence widens search
            page_vectors[step.state_id] = None
            return None
        vector = page.vector if page.elements else None
        page_vectors[step.state_id] = vector
        return vector

    targets = tuple(str(value) for value in target_environments)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for task_id in tasks:
            sources = grouped[task_id].get("100", [])
            if not sources:
                missing.append(
                    {"task_id": task_id, "environment_id": "100", "role": "source"}
                )
                continue
            for environment_id in targets:
                target_traces = grouped[task_id].get(environment_id, [])
                if not target_traces:
                    missing.append(
                        {
                            "task_id": task_id,
                            "environment_id": environment_id,
                            "role": "target",
                        }
                    )
                    continue
                for source in sorted(sources, key=lambda trace: trace.trace_id):
                    for target in sorted(target_traces, key=lambda trace: trace.trace_id):
                        if accelerated:
                            candidate_started = time.perf_counter()
                            candidate_cells = _transfer_candidate_cells(
                                source,
                                target,
                                page_vector=page_vector,
                            )
                            candidate_search_seconds += (
                                time.perf_counter() - candidate_started
                            )
                        else:
                            candidate_cells = None
                        result = evaluate_trace_pair(
                            source,
                            target,
                            scorer=cached_scorer,
                            candidate_cells=candidate_cells,
                            executor=executor if workers > 1 else None,
                        )
                        results.append(result)
                        if progress is not None:
                            progress(result)

    return {
        "schema_version": "omniflow.bmoca-transfer-replay-sidecar.v1",
        "configuration": {
            "source_environment_id": "100",
            "target_environment_ids": list(targets),
            "main_path_unchanged": True,
            "control": "fixed_sequence_index",
            "sidecar": "action_aware_transfer_sequence_dp",
            "identity_constraints": [],
            "score_source": "canonical_omnitransfer_continuous_pair_probability",
            "vlm_fallback": "disabled",
            "workers": workers,
            "search": {
                "accelerated": accelerated,
                "coarse_evidence": "omniflow_native_512d_page_embedding",
                "final_edge_score": "omnitransfer_only",
                "page_top_k": 3,
                "monotonic_band_radius": 2,
            },
            "direct_actions": sorted(_DIRECT_ACTIONS),
        },
        "summary": {
            **_aggregate(results, missing),
            "timing": {
                "wall_seconds": time.perf_counter() - evaluation_started,
                "candidate_search_seconds": candidate_search_seconds,
                "transfer_scoring_seconds": sum(
                    result["timing"]["transfer_scoring_seconds"]
                    for result in results
                ),
                "dp_decode_seconds": sum(
                    result["timing"]["dp_decode_seconds"]
                    for result in results
                ),
            },
            "transfer_score_cache": {
                "entries": len(score_cache),
                "hits": cache_hits,
                "misses": cache_misses,
            },
        },
        "pairs": results,
        "missing": missing,
    }


def _load_trace(
    record: dict[str, Any],
    runlog: dict[str, Any],
    catalog: dict[str, Any],
) -> BmocaTrace:
    if runlog.get("schema_version") != _LEGACY_RUNLOG_VERSION:
        raise ValueError("unsupported_bmoca_runlog_version")
    if catalog.get("schema_version") != _STATE_CATALOG_VERSION:
        raise ValueError("unsupported_bmoca_state_catalog_version")
    states = catalog.get("states")
    raw_steps = runlog.get("steps")
    if not isinstance(states, dict) or not isinstance(raw_steps, list):
        raise ValueError("bmoca_trace_contract_invalid")
    steps = []
    for position, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, dict):
            raise ValueError("bmoca_runlog_step_invalid")
        result = raw_step.get("result")
        if not isinstance(result, dict) or result.get("success") is not True:
            continue
        action = raw_step.get("action")
        if not isinstance(action, dict):
            raise ValueError("bmoca_runlog_action_invalid")
        action_kind = str(action.get("tool") or "").strip()
        args = action.get("args")
        if not isinstance(args, dict):
            raise ValueError("bmoca_runlog_action_args_invalid")
        state_identifier = str(raw_step.get("before_state_id") or "").strip()
        state = states.get(state_identifier)
        if not isinstance(state, dict):
            raise ValueError(f"bmoca_state_missing:{state_identifier}")
        xml = str(state.get("xml") or "")
        point = _action_point(action_kind, args, state, xml)
        steps.append(
            BmocaStep(
                index=int(raw_step.get("step_index", position)),
                action_kind=action_kind,
                xml=xml,
                point=point,
                bounds=_action_bounds(xml, point, action_kind),
                state_id=state_identifier,
            )
        )
    return BmocaTrace(
        trace_id=str(record.get("trace_id") or "").strip(),
        task_id=str(record.get("task_id") or "").strip(),
        environment_id=str(record.get("environment_id") or "").strip(),
        steps=tuple(steps),
    )


def _score_step_pair(source: BmocaStep, target: BmocaStep) -> TransferMatchScore:
    if not source.xml or not target.xml or source.point is None or target.bounds is None:
        return TransferMatchScore(reason="selector_endpoint_missing")
    return retarget_transfer_score(
        _score_step_pair_base(source, target),
        source_point=source.point,
        target_bounds=target.bounds,
    )


def _score_step_pair_base(
    source: BmocaStep,
    target: BmocaStep,
) -> TransferMatchScore:
    if not source.xml or not target.xml or source.point is None:
        return TransferMatchScore(reason="selector_endpoint_missing")
    return score_transfer_match(
        source_xml=source.xml,
        target_xml=target.xml,
        source_point=source.point,
    )


def _transfer_candidate_cells(
    source: BmocaTrace,
    target: BmocaTrace,
    *,
    page_vector: Callable[[BmocaStep], np.ndarray | None],
    page_top_k: int = 3,
    band_radius: int = 2,
) -> frozenset[tuple[int, int]]:
    target_positions = [
        position
        for position, step in enumerate(target.steps)
        if step.bounds is not None
    ]
    if not target_positions:
        return frozenset()
    cells = set()
    source_denominator = max(1, len(source.steps) - 1)
    target_denominator = max(1, len(target.steps) - 1)
    for source_position, source_step in enumerate(source.steps):
        if source_step.point is None:
            continue
        expected = source_position / source_denominator * target_denominator
        fixed_position = source_position
        if fixed_position in target_positions:
            cells.add((source_position, fixed_position))
        for target_position in target_positions:
            if abs(target_position - expected) <= band_radius:
                cells.add((source_position, target_position))
        source_vector = page_vector(source_step)
        if source_vector is None:
            cells.update((source_position, position) for position in target_positions)
            continue
        ranked = []
        for target_position in target_positions:
            target_vector = page_vector(target.steps[target_position])
            similarity = _cosine(source_vector, target_vector)
            ranked.append(
                (similarity, -abs(target_position - expected), -target_position)
            )
        for _, _, negative_position in sorted(ranked, reverse=True)[:page_top_k]:
            cells.add((source_position, -negative_position))
    return frozenset(cells)


def _action_point(
    action_kind: str,
    args: dict[str, Any],
    state: dict[str, Any],
    xml: str,
) -> tuple[float, float] | None:
    if action_kind not in _SELECTOR_ACTIONS:
        return None
    try:
        x = float(args["x"])
        y = float(args["y"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    display = state.get("display")
    if isinstance(display, dict):
        try:
            width = float(display["width"])
            height = float(display["height"])
        except (KeyError, TypeError, ValueError):
            width = height = 0.0
    else:
        width = height = 0.0
    if width <= 0.0 or height <= 0.0:
        width, height = _xml_size(xml)
    if width <= 0.0 or height <= 0.0:
        return None
    return x / 1000.0 * width, y / 1000.0 * height


def _action_bounds(
    xml: str,
    point: tuple[float, float] | None,
    action_kind: str,
) -> tuple[float, float, float, float] | None:
    if point is None or not xml:
        return None
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None
    actionable_candidates = []
    semantic_candidates = []
    for element in root.iter():
        bounds = _parse_bounds(element.attrib.get("bounds"))
        if bounds is None or not _contains(bounds, point):
            continue
        if _actionable(element.attrib, action_kind):
            actionable_candidates.append(bounds)
        elif _semantic_target(element.attrib):
            semantic_candidates.append(bounds)
    if actionable_candidates:
        return min(actionable_candidates, key=_area)
    return min(semantic_candidates, key=_area) if semantic_candidates else None


def _actionable(attributes: dict[str, str], action_kind: str) -> bool:
    if str(attributes.get("enabled") or "true").lower() == "false":
        return False
    if str(attributes.get("displayed") or "true").lower() == "false":
        return False
    def truthy(key: str) -> bool:
        return str(attributes.get(key) or "").lower() == "true"

    class_name = str(attributes.get("class") or "").lower()
    if action_kind == "input_text":
        return truthy("focusable") or "edittext" in class_name
    if action_kind == "long_press":
        return truthy("long-clickable") or truthy("clickable")
    return any(truthy(key) for key in ("clickable", "focusable", "checkable"))


def _semantic_target(attributes: dict[str, str]) -> bool:
    if str(attributes.get("enabled") or "true").lower() == "false":
        return False
    if str(attributes.get("displayed") or "true").lower() == "false":
        return False
    return any(
        str(attributes.get(key) or "").strip()
        for key in ("text", "content-desc", "resource-id")
    )


def _xml_size(xml: str) -> tuple[float, float]:
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return 0.0, 0.0
    widths = []
    heights = []
    for element in root.iter():
        bounds = _parse_bounds(element.attrib.get("bounds"))
        if bounds is not None:
            widths.append(bounds[2])
            heights.append(bounds[3])
    return max(widths, default=0.0), max(heights, default=0.0)


def _parse_bounds(value: Any) -> tuple[float, float, float, float] | None:
    match = _BOUNDS_PATTERN.fullmatch(str(value or "").strip())
    if match is None:
        return None
    bounds = tuple(float(item) for item in match.groups())
    left, top, right, bottom = bounds
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _match_record(
    source: BmocaStep,
    target: BmocaStep,
    evidence: TransferMatchScore,
) -> dict[str, Any]:
    return {
        "source_step_index": source.index,
        "target_step_index": target.index,
        "source_action_kind": source.action_kind,
        "target_action_kind": target.action_kind,
        "probability": evidence.probability,
        "top_probability": evidence.top_probability,
        "candidate_rank": evidence.candidate_rank,
        "mapped": evidence.mapped,
        "exact_hit": evidence.exact_hit,
        "reason": evidence.reason,
    }


def _missing_match(source: BmocaStep, *, reason: str) -> dict[str, Any]:
    return {
        "source_step_index": source.index,
        "target_step_index": None,
        "source_action_kind": source.action_kind,
        "target_action_kind": None,
        "probability": None,
        "top_probability": None,
        "candidate_rank": None,
        "mapped": False,
        "exact_hit": False,
        "reason": reason,
    }


def _method_result(matches: list[dict[str, Any]]) -> dict[str, Any]:
    exact_hits = sum(match["exact_hit"] is True for match in matches)
    action_count = len(matches)
    return {
        "selector_action_count": action_count,
        "exact_hit_count": exact_hits,
        "exact_hit_rate": exact_hits / action_count if action_count else 1.0,
        "complete_hit": exact_hits == action_count,
        "matches": matches,
    }


def _aggregate(results: list[dict[str, Any]], missing: list[dict[str, Any]]) -> dict[str, Any]:
    environments: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        environments[str(result["target_environment_id"])].append(result)
    candidate_cells = sum(
        result["transfer_candidate_cell_count"] for result in results
    )
    full_cells = sum(result["transfer_full_cell_count"] for result in results)
    return {
        "task_count": len({result["task_id"] for result in results}),
        "trace_pair_count": len(results),
        "missing_pair_count": len(missing),
        "selector_action_count": sum(result["selector_action_count"] for result in results),
        "replayed_action_count": sum(result["replayed_action_count"] for result in results),
        "direct_action_count": sum(result["direct_action_count"] for result in results),
        "transfer_candidate_cell_count": candidate_cells,
        "transfer_full_cell_count": full_cells,
        "transfer_cell_reduction": (
            1.0 - candidate_cells / full_cells if full_cells else 0.0
        ),
        "fixed_index": _aggregate_method(results, "fixed_index"),
        "dp": _aggregate_method(results, "dp"),
        "by_environment": {
            environment: {
                "trace_pair_count": len(values),
                "fixed_index": _aggregate_method(values, "fixed_index"),
                "dp": _aggregate_method(values, "dp"),
            }
            for environment, values in sorted(environments.items())
        },
        "missing_by_environment": dict(
            sorted(Counter(item["environment_id"] for item in missing).items())
        ),
    }


def _aggregate_method(results: Iterable[dict[str, Any]], method: str) -> dict[str, Any]:
    values = list(results)
    action_count = sum(item[method]["selector_action_count"] for item in values)
    exact_hits = sum(item[method]["exact_hit_count"] for item in values)
    complete_hits = sum(item[method]["complete_hit"] is True for item in values)
    return {
        "selector_action_count": action_count,
        "exact_hit_count": exact_hits,
        "exact_hit_rate": exact_hits / action_count if action_count else 1.0,
        "complete_trace_count": complete_hits,
        "complete_trace_rate": complete_hits / len(values) if values else 0.0,
    }


def _asset_path(root: Path, entry: dict[str, Any]) -> Path:
    raw = entry.get("path")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("bmoca_asset_path_required")
    path = (root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("bmoca_asset_outside_corpus") from error
    return path


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _contains(
    bounds: tuple[float, float, float, float],
    point: tuple[float, float],
) -> bool:
    return bounds[0] <= point[0] <= bounds[2] and bounds[1] <= point[1] <= bounds[3]


def _area(bounds: tuple[float, float, float, float]) -> float:
    return (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _cosine(left: np.ndarray, right: np.ndarray | None) -> float:
    if right is None:
        return -1.0
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        return -1.0
    return float(np.dot(left, right) / denominator)


def _write_report(path: Path, report: dict[str, Any]) -> None:
    if not path.is_absolute():
        raise ValueError("bmoca_report_path_must_be_absolute")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-env", action="append", dest="target_environments")
    parser.add_argument("--limit-tasks", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--full-matrix", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    environments = tuple(args.target_environments or ("101", "105"))
    completed = 0

    def progress(result: dict[str, Any]) -> None:
        nonlocal completed
        completed += 1
        print(
            json.dumps(
                {
                    "completed_pairs": completed,
                    "task_id": result["task_id"],
                    "target_environment_id": result["target_environment_id"],
                    "fixed_complete": result["fixed_index"]["complete_hit"],
                    "dp_complete": result["dp"]["complete_hit"],
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )

    report = evaluate_bmoca_corpus(
        args.corpus,
        target_environments=environments,
        limit_tasks=args.limit_tasks,
        workers=args.workers,
        accelerated=not args.full_matrix,
        progress=progress,
    )
    _write_report(args.output.expanduser().resolve(), report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BmocaStep",
    "BmocaTrace",
    "evaluate_bmoca_corpus",
    "evaluate_trace_pair",
    "load_bmoca_traces",
    "main",
]
