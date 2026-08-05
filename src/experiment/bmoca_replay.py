"""Offline B-MoCA replay comparisons for transfer-DP and replay baselines.

The existing replay path is not imported or modified here.  Each source/target
trace pair can be evaluated by the action-aware transfer-DP sidecar or by the
model-free selector/coordinate baseline suite.  No suite executes a source
coordinate on a target device.
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
from omniflow.transfer.embedding import ElementEmbedding, PageEncoder, TreeEmbedding
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
_REPLAY_BASELINES = (
    "identity_unique",
    "structured_unique",
    "normalized_coordinate",
)
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


@dataclass(frozen=True)
class _BaselineTrace:
    trace_id: str
    task_id: str
    environment_id: str
    runlog_path: Path
    state_catalog_path: Path


@dataclass(frozen=True)
class _SelectorResult:
    selected: ElementEmbedding | None
    execute: bool
    reason: str
    candidate_count: int
    tied_candidate_count: int


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


def evaluate_replay_baselines(
    corpus_root: str | Path,
    *,
    target_environments: Sequence[str] = ("101", "105"),
    limit_tasks: int | None = None,
) -> dict[str, Any]:
    """Evaluate non-model replay baselines on one shared aligned corpus.

    Page and action correspondence are read from the offline corpus.  Missing
    correspondence remains a strict replay failure.  This suite never calls
    OmniTransfer, a VLM, or a target runtime.
    """

    started = time.perf_counter()
    root = Path(corpus_root).expanduser().resolve()
    manifest = _read_object(root / "manifest.json")
    if manifest.get("schema_version") != _CORPUS_VERSION:
        raise ValueError("unsupported_bmoca_corpus_version")
    traces = _load_baseline_traces(manifest, root=root)
    task_ids = sorted({trace.task_id for trace in traces.values()})
    if limit_tasks is not None:
        if limit_tasks <= 0:
            raise ValueError("limit_tasks_must_be_positive")
        task_ids = task_ids[:limit_tasks]
    selected_tasks = frozenset(task_ids)
    alignment_ref = manifest.get("alignments")
    if not isinstance(alignment_ref, dict):
        raise ValueError("bmoca_alignment_reference_invalid")
    alignments = _read_jsonl(_asset_path(root, alignment_ref))
    targets = frozenset(str(item) for item in target_environments)
    task_environments: dict[str, set[str]] = defaultdict(set)
    for trace in traces.values():
        task_environments[trace.task_id].add(trace.environment_id)
    balanced_tasks = {
        task_id
        for task_id, environments in task_environments.items()
        if {"100", *targets} <= environments
    }

    encoder = PageEncoder()
    artifact_cache: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    page_cache: dict[tuple[str, str], TreeEmbedding] = {}
    episodes: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    for alignment in alignments:
        normalized = _normalize_baseline_alignment(
            alignment,
            traces=traces,
            target_environments=targets,
        )
        if normalized is None:
            continue
        source, target, action_map, page_map = normalized
        if source.task_id not in selected_tasks:
            continue
        source_runlog, source_states = _baseline_trace_artifacts(
            source,
            cache=artifact_cache,
        )
        target_runlog, target_states = _baseline_trace_artifacts(
            target,
            cache=artifact_cache,
        )
        episode_steps = []
        for source_step in _baseline_selector_steps(source_runlog):
            source_index = int(source_step["step_index"])
            target_step_index = action_map.get(source_index)
            target_step = (
                _baseline_step_by_index(target_runlog, target_step_index)
                if target_step_index is not None
                else None
            )
            target_sequence = page_map.get(int(source_step["_sequence_index"]))
            target_page_step = (
                _baseline_step_by_sequence(target_runlog, target_sequence)
                if target_sequence is not None
                else None
            )
            target_state_id = (
                _baseline_state_by_sequence(target_runlog, target_sequence)
                if target_sequence is not None
                else None
            )
            record = _evaluate_baseline_step(
                source=source,
                target=target,
                source_step=source_step,
                target_step=target_step,
                target_page_step=target_page_step,
                target_state_id=target_state_id,
                source_states=source_states,
                target_states=target_states,
                encoder=encoder,
                page_cache=page_cache,
            )
            episode_steps.append(record)
            steps.append(record)
        episodes.append(
            _baseline_episode(
                source=source,
                target=target,
                steps=episode_steps,
                balanced_triplet=source.task_id in balanced_tasks,
            )
        )

    episodes.sort(
        key=lambda item: (
            item["task_id"],
            item["target_environment_id"],
            item["episode_id"],
        )
    )
    steps.sort(
        key=lambda item: (
            item["task_id"],
            item["target_environment_id"],
            item["source_step_index"],
        )
    )
    return {
        "schema_version": "omniflow.bmoca-replay-baselines.v1",
        "configuration": {
            "source_environment_id": "100",
            "target_environment_ids": sorted(targets),
            "methods": list(_REPLAY_BASELINES),
            "target_page_correspondence": "offline_monotonic_page_alignment",
            "action_reference": "offline_monotonic_action_alignment",
            "missing_correspondence_policy": "strict_replay_failure",
            "point_hit_rule": "predicted_point_inside_gold_actionable_bounds",
            "omnitransfer": "disabled",
            "vlm_fallback": "disabled",
            "runtime_execution": "disabled",
        },
        "summary": {
            "task_count": len({episode["task_id"] for episode in episodes}),
            "episode_count": len(episodes),
            "balanced_triplet_episode_count": sum(
                episode["balanced_triplet"] for episode in episodes
            ),
            "source_selector_step_count": len(steps),
            "gold_aligned_step_count": sum(
                step["gold_alignment_available"] for step in steps
            ),
            "methods": {
                method: _aggregate_baseline_method(episodes, method=method)
                for method in _REPLAY_BASELINES
            },
            "wall_seconds": time.perf_counter() - started,
        },
        "episodes": episodes,
        "steps": steps,
    }


def _load_baseline_traces(
    manifest: dict[str, Any],
    *,
    root: Path,
) -> dict[str, _BaselineTrace]:
    raw_traces = manifest.get("traces")
    if not isinstance(raw_traces, list):
        raise ValueError("bmoca_corpus_traces_invalid")
    traces: dict[str, _BaselineTrace] = {}
    for raw in raw_traces:
        if not isinstance(raw, dict):
            raise ValueError("bmoca_trace_record_invalid")
        success = raw.get("success_evidence")
        if not isinstance(success, dict) or success.get("official_success") is not True:
            continue
        trace_id = str(raw.get("trace_id") or "").strip()
        runlog = raw.get("runlog")
        catalog = raw.get("state_catalog")
        if not trace_id or not isinstance(runlog, dict) or not isinstance(catalog, dict):
            raise ValueError("bmoca_trace_assets_missing")
        traces[trace_id] = _BaselineTrace(
            trace_id=trace_id,
            task_id=str(raw.get("task_id") or "").strip(),
            environment_id=str(raw.get("environment_id") or "").strip(),
            runlog_path=_asset_path(root, runlog),
            state_catalog_path=_asset_path(root, catalog),
        )
    return traces


def _normalize_baseline_alignment(
    alignment: dict[str, Any],
    *,
    traces: dict[str, _BaselineTrace],
    target_environments: frozenset[str],
) -> tuple[
    _BaselineTrace,
    _BaselineTrace,
    dict[int, int],
    dict[int, int],
] | None:
    left = traces.get(str(alignment.get("left_trace_id") or ""))
    right = traces.get(str(alignment.get("right_trace_id") or ""))
    if left is None or right is None or left.task_id != right.task_id:
        raise ValueError("bmoca_alignment_trace_invalid")
    if left.environment_id == "100" and right.environment_id in target_environments:
        source, target = left, right
        source_action, target_action = "left_step_index", "right_step_index"
        source_page, target_page = "left_sequence_index", "right_sequence_index"
    elif right.environment_id == "100" and left.environment_id in target_environments:
        source, target = right, left
        source_action, target_action = "right_step_index", "left_step_index"
        source_page, target_page = "right_sequence_index", "left_sequence_index"
    else:
        return None
    action_alignment = alignment.get("action_alignment")
    page_alignment = alignment.get("page_alignment")
    action_pairs = (
        action_alignment.get("pairs") if isinstance(action_alignment, dict) else None
    )
    page_pairs = (
        page_alignment.get("pairs") if isinstance(page_alignment, dict) else None
    )
    if not isinstance(action_pairs, list) or not isinstance(page_pairs, list):
        raise ValueError("bmoca_alignment_pairs_invalid")
    actions = _unique_int_map(action_pairs, source_action, target_action)
    pages = _unique_int_map(page_pairs, source_page, target_page)
    return source, target, actions, pages


def _unique_int_map(
    values: list[Any],
    source_field: str,
    target_field: str,
) -> dict[int, int]:
    result: dict[int, int] = {}
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("bmoca_alignment_pair_invalid")
        source = value.get(source_field)
        target = value.get(target_field)
        if (
            not isinstance(source, int)
            or isinstance(source, bool)
            or not isinstance(target, int)
            or isinstance(target, bool)
        ):
            raise ValueError("bmoca_alignment_index_invalid")
        if source in result:
            raise ValueError("bmoca_alignment_source_duplicate")
        result[source] = target
    return result


def _baseline_trace_artifacts(
    trace: _BaselineTrace,
    *,
    cache: dict[str, tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    cached = cache.get(trace.trace_id)
    if cached is not None:
        return cached
    runlog = _read_object(trace.runlog_path)
    catalog = _read_object(trace.state_catalog_path)
    states = catalog.get("states")
    if not isinstance(states, dict):
        raise ValueError("bmoca_states_invalid")
    result = runlog, states
    cache[trace.trace_id] = result
    return result


def _baseline_selector_steps(runlog: dict[str, Any]) -> list[dict[str, Any]]:
    raw_steps = runlog.get("steps")
    if not isinstance(raw_steps, list):
        raise ValueError("bmoca_runlog_steps_invalid")
    result = []
    for position, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            raise ValueError("bmoca_runlog_step_invalid")
        action = raw.get("action")
        if isinstance(action, dict) and str(action.get("tool") or "") in _SELECTOR_ACTIONS:
            step = dict(raw)
            step["step_index"] = int(raw.get("step_index", position))
            step["_sequence_index"] = position
            result.append(step)
    return result


def _baseline_step_by_index(
    runlog: dict[str, Any],
    step_index: int,
) -> dict[str, Any]:
    steps = runlog.get("steps")
    for position, step in enumerate(steps if isinstance(steps, list) else ()):
        if isinstance(step, dict) and int(step.get("step_index", position)) == step_index:
            return step
    raise ValueError(f"bmoca_target_step_missing:{step_index}")


def _baseline_step_by_sequence(
    runlog: dict[str, Any],
    sequence_index: int,
) -> dict[str, Any] | None:
    steps = runlog.get("steps")
    if not isinstance(steps, list):
        raise ValueError("bmoca_runlog_steps_invalid")
    if 0 <= sequence_index < len(steps):
        step = steps[sequence_index]
        if not isinstance(step, dict):
            raise ValueError("bmoca_runlog_step_invalid")
        return step
    if sequence_index == len(steps):
        return None
    raise ValueError("bmoca_target_sequence_out_of_range")


def _baseline_state_by_sequence(
    runlog: dict[str, Any],
    sequence_index: int,
) -> str:
    step = _baseline_step_by_sequence(runlog, sequence_index)
    if step is not None:
        return str(step.get("before_state_id") or "")
    steps = runlog.get("steps")
    if isinstance(steps, list) and steps:
        final_step = steps[-1]
        if isinstance(final_step, dict):
            return str(final_step.get("after_state_id") or "")
    return str(runlog.get("final_state_id") or "")


def _evaluate_baseline_step(
    *,
    source: _BaselineTrace,
    target: _BaselineTrace,
    source_step: dict[str, Any],
    target_step: dict[str, Any] | None,
    target_page_step: dict[str, Any] | None,
    target_state_id: str | None,
    source_states: dict[str, Any],
    target_states: dict[str, Any],
    encoder: PageEncoder,
    page_cache: dict[tuple[str, str], TreeEmbedding],
) -> dict[str, Any]:
    source_index = int(source_step["step_index"])
    base = {
        "task_id": source.task_id,
        "source_trace_id": source.trace_id,
        "target_trace_id": target.trace_id,
        "source_environment_id": "100",
        "target_environment_id": target.environment_id,
        "source_step_index": source_index,
        "source_sequence_index": int(source_step["_sequence_index"]),
        "target_step_index": (
            int(target_step["step_index"]) if target_step is not None else None
        ),
    }
    source_state_id = str(source_step.get("before_state_id") or "")
    source_state = source_states.get(source_state_id)
    if not isinstance(source_state, dict):
        raise ValueError(f"bmoca_source_state_missing:{source_state_id}")
    if target_step is not None:
        target_state_id = str(target_step.get("before_state_id") or "")
        target_page_step = target_step
    target_state = target_states.get(str(target_state_id or ""))
    if not isinstance(target_state, dict):
        prediction = _unavailable_baseline("target_page_alignment_missing")
        return {
            **base,
            "gold_alignment_available": False,
            "predictions": {
                method: dict(prediction) for method in _REPLAY_BASELINES
            },
        }

    source_page = _baseline_page(
        source.trace_id,
        source_state_id,
        source_state,
        encoder=encoder,
        cache=page_cache,
    )
    target_page = _baseline_page(
        target.trace_id,
        str(target_state_id or ""),
        target_state,
        encoder=encoder,
        cache=page_cache,
    )
    source_point = _baseline_action_point(source_step, source_state, source_page)
    source_node = _baseline_action_element(source_page, source_point)
    target_point = (
        _baseline_action_point(target_step, target_state, target_page)
        if target_step is not None
        else None
    )
    gold_node = _baseline_action_element(target_page, target_point)
    gold_available = target_point is not None and gold_node is not None
    if source_point is None or source_node is None:
        prediction = _unavailable_baseline("source_node_missing")
        return {
            **base,
            "gold_alignment_available": gold_available,
            "predictions": {
                method: dict(prediction) for method in _REPLAY_BASELINES
            },
        }

    predictions: dict[str, dict[str, Any]] = {}
    for method, selector in (
        ("identity_unique", _baseline_identity_selector),
        ("structured_unique", _baseline_structured_selector),
    ):
        selected = selector(source_page, target_page, source_node)
        point = (
            _baseline_project_offset(
                source_point,
                source_node.bounds,
                selected.selected.bounds,
            )
            if selected.execute and selected.selected is not None
            else None
        )
        predictions[method] = _baseline_prediction(
            selected=selected,
            point=point,
            gold_available=gold_available,
            gold_node=gold_node,
        )

    source_width, source_height = _baseline_page_dimensions(
        source_state,
        source_page,
    )
    target_width, target_height = _baseline_page_dimensions(
        target_state,
        target_page,
    )
    normalized_point = (
        source_point[0] / source_width * target_width,
        source_point[1] / source_height * target_height,
    )
    coordinate_selected = _SelectorResult(
        selected=None,
        execute=True,
        reason="ok",
        candidate_count=1,
        tied_candidate_count=1,
    )
    predictions["normalized_coordinate"] = _baseline_prediction(
        selected=coordinate_selected,
        point=normalized_point,
        gold_available=gold_available,
        gold_node=gold_node,
    )
    return {
        **base,
        "gold_alignment_available": gold_available,
        "target_page_step_index": (
            int(target_page_step.get("step_index", 0))
            if isinstance(target_page_step, dict)
            else None
        ),
        "predictions": predictions,
    }


def _baseline_prediction(
    *,
    selected: _SelectorResult,
    point: tuple[float, float] | None,
    gold_available: bool,
    gold_node: ElementEmbedding | None,
) -> dict[str, Any]:
    return {
        "execute": selected.execute,
        "hit": bool(
            gold_available
            and point is not None
            and gold_node is not None
            and _contains(gold_node.bounds, point)
        ),
        "reason": selected.reason,
        "predicted_point": list(point) if point is not None else None,
        "selected_bounds": (
            list(selected.selected.bounds) if selected.selected is not None else None
        ),
        "candidate_count": selected.candidate_count,
        "tied_candidate_count": selected.tied_candidate_count,
    }


def _unavailable_baseline(reason: str) -> dict[str, Any]:
    return {
        "execute": False,
        "hit": False,
        "reason": reason,
        "predicted_point": None,
        "selected_bounds": None,
        "candidate_count": 0,
        "tied_candidate_count": 0,
    }


def _baseline_page(
    trace_id: str,
    state_id: str,
    state: dict[str, Any],
    *,
    encoder: PageEncoder,
    cache: dict[tuple[str, str], TreeEmbedding],
) -> TreeEmbedding:
    key = trace_id, state_id
    cached = cache.get(key)
    if cached is not None:
        return cached
    page = encoder.embed(
        Observation(
            xml=str(state.get("xml") or ""),
            package_name=str(state.get("package_name") or ""),
            activity_name=str(state.get("activity_name") or ""),
        )
    )
    if not page.elements:
        raise ValueError(f"bmoca_state_xml_empty:{trace_id}:{state_id}")
    cache[key] = page
    return page


def _baseline_action_point(
    step: dict[str, Any],
    state: dict[str, Any],
    page: TreeEmbedding,
) -> tuple[float, float] | None:
    action = step.get("action")
    args = action.get("args") if isinstance(action, dict) else None
    if not isinstance(args, dict):
        return None
    try:
        x, y = float(args["x"]), float(args["y"])
    except (KeyError, TypeError, ValueError):
        return None
    width, height = _baseline_page_dimensions(state, page)
    point = x / 1000.0 * width, y / 1000.0 * height
    if not all(math.isfinite(item) for item in point):
        return None
    return point if 0.0 <= point[0] <= width and 0.0 <= point[1] <= height else None


def _baseline_page_dimensions(
    state: dict[str, Any],
    page: TreeEmbedding,
) -> tuple[float, float]:
    display = state.get("display")
    display = display if isinstance(display, dict) else {}
    try:
        width = float(display.get("width") or 0.0)
        height = float(display.get("height") or 0.0)
    except (TypeError, ValueError):
        width = height = 0.0
    if width > 0.0 and height > 0.0:
        return width, height
    left, top, right, bottom = page.root_bounds
    return float(right - min(0, left)), float(bottom - min(0, top))


def _baseline_action_element(
    page: TreeEmbedding,
    point: tuple[float, float] | None,
) -> ElementEmbedding | None:
    if point is None:
        return None
    containing = [item for item in page.elements if _contains(item.bounds, point)]
    actionable = [item for item in containing if _baseline_actionable(item)]
    return min(
        actionable or containing,
        key=lambda item: (
            _area(item.bounds),
            not _baseline_stable_identity(item),
            -item.depth,
            item.id,
        ),
        default=None,
    )


def _baseline_identity_selector(
    source_page: TreeEmbedding,
    target_page: TreeEmbedding,
    source: ElementEmbedding,
) -> _SelectorResult:
    del source_page
    scored = [
        (_baseline_identity_score(source, target), target)
        for target in _baseline_selector_candidates(target_page)
    ]
    scored = [item for item in scored if item[0] > 0.0]
    if not scored:
        return _SelectorResult(None, False, "no_identity_candidate", 0, 0)
    scored.sort(key=lambda item: (-item[0], item[1].id))
    tied = [target for score, target in scored if score == scored[0][0]]
    if len(tied) != 1:
        return _SelectorResult(
            None,
            False,
            "target_identity_not_unique",
            len(scored),
            len(tied),
        )
    return _SelectorResult(tied[0], True, "ok", len(scored), 1)


def _baseline_structured_selector(
    source_page: TreeEmbedding,
    target_page: TreeEmbedding,
    source: ElementEmbedding,
) -> _SelectorResult:
    scored = [
        (
            _baseline_structured_score(source_page, target_page, source, target),
            target,
        )
        for target in _baseline_selector_candidates(target_page)
    ]
    scored = [item for item in scored if any(item[0][:5])]
    if not scored:
        return _SelectorResult(None, False, "no_structured_candidate", 0, 0)
    scored.sort(key=lambda item: (item[0], item[1].id), reverse=True)
    tied = [target for score, target in scored if score == scored[0][0]]
    if len(tied) != 1:
        return _SelectorResult(
            None,
            False,
            "target_structure_not_unique",
            len(scored),
            len(tied),
        )
    return _SelectorResult(tied[0], True, "ok", len(scored), 1)


def _baseline_identity_score(
    source: ElementEmbedding,
    target: ElementEmbedding,
) -> float:
    source_attributes = source.attributes
    target_attributes = target.attributes
    score = 0.0
    stable = False
    source_resource = _baseline_normalized(source_attributes.get("resource_id"))
    target_resource = _baseline_normalized(target_attributes.get("resource_id"))
    if source_resource and source_resource == target_resource:
        score += 8.0
        stable = True
    elif (
        _baseline_tail(source_resource)
        and _baseline_tail(source_resource) == _baseline_tail(target_resource)
    ):
        score += 6.0
        stable = True
    for key in ("text", "content_description"):
        source_value = _baseline_normalized(source_attributes.get(key))
        target_value = _baseline_normalized(target_attributes.get(key))
        if source_value and source_value == target_value:
            score += 4.0
            stable = True
    if not stable:
        return 0.0
    if _baseline_class_tail(source) == _baseline_class_tail(target):
        score += 1.0
    score += 0.5 * sum(
        source_attributes.get(key) == target_attributes.get(key)
        for key in ("clickable", "editable", "scrollable")
    )
    return score


def _baseline_structured_score(
    source_page: TreeEmbedding,
    target_page: TreeEmbedding,
    source: ElementEmbedding,
    target: ElementEmbedding,
) -> tuple[int, ...]:
    source_resource = _baseline_normalized(source.attributes.get("resource_id"))
    target_resource = _baseline_normalized(target.attributes.get("resource_id"))
    direct_text = sum(
        bool(_baseline_normalized(source.attributes.get(key)))
        and _baseline_normalized(source.attributes.get(key))
        == _baseline_normalized(target.attributes.get(key))
        for key in ("text", "content_description")
    )
    source_subtree = _baseline_subtree_tokens(source_page, source)
    target_subtree = _baseline_subtree_tokens(target_page, target)
    source_parent = _baseline_parent_tokens(source_page, source)
    target_parent = _baseline_parent_tokens(target_page, target)
    source_siblings = _baseline_sibling_tokens(source_page, source)
    target_siblings = _baseline_sibling_tokens(target_page, target)
    return (
        int(bool(source_resource) and source_resource == target_resource),
        int(
            bool(_baseline_tail(source_resource))
            and _baseline_tail(source_resource) == _baseline_tail(target_resource)
        ),
        direct_text,
        len(source_subtree & target_subtree),
        _baseline_scaled_jaccard(source_subtree, target_subtree),
        len(source_parent & target_parent),
        len(source_siblings & target_siblings),
        _baseline_class_path_suffix(source_page, source, target_page, target),
        int(_baseline_class_tail(source) == _baseline_class_tail(target)),
        sum(
            source.attributes.get(key) == target.attributes.get(key)
            for key in ("clickable", "editable", "scrollable", "checkable")
        ),
    )


def _baseline_selector_candidates(
    page: TreeEmbedding,
) -> Iterable[ElementEmbedding]:
    return (
        element
        for element in page.elements
        if element.attributes.get("enabled", True)
        and element.attributes.get("visible", True)
        and _area(element.bounds) > 0.0
    )


def _baseline_subtree_tokens(
    page: TreeEmbedding,
    root: ElementEmbedding,
) -> frozenset[str]:
    by_id = {element.id: element for element in page.elements}
    pending = [root.id]
    tokens: set[str] = set()
    while pending:
        node = by_id.get(pending.pop())
        if node is None:
            continue
        tokens.update(_baseline_semantic_tokens(node))
        pending.extend(node.children_ids)
    return frozenset(tokens)


def _baseline_parent_tokens(
    page: TreeEmbedding,
    node: ElementEmbedding,
) -> frozenset[str]:
    by_id = {element.id: element for element in page.elements}
    parent = by_id.get(str(node.parent_id or ""))
    return _baseline_semantic_tokens(parent) if parent is not None else frozenset()


def _baseline_sibling_tokens(
    page: TreeEmbedding,
    node: ElementEmbedding,
) -> frozenset[str]:
    by_id = {element.id: element for element in page.elements}
    parent = by_id.get(str(node.parent_id or ""))
    if parent is None:
        return frozenset()
    return frozenset(
        token
        for child_id in parent.children_ids
        if child_id != node.id and child_id in by_id
        for token in _baseline_semantic_tokens(by_id[child_id])
    )


def _baseline_semantic_tokens(node: ElementEmbedding) -> frozenset[str]:
    return frozenset(
        f"{key}:{value}"
        for key in ("resource_id", "text", "content_description")
        if (value := _baseline_normalized(node.attributes.get(key)))
    )


def _baseline_class_path_suffix(
    source_page: TreeEmbedding,
    source: ElementEmbedding,
    target_page: TreeEmbedding,
    target: ElementEmbedding,
) -> int:
    matched = 0
    for left, right in zip(
        reversed(_baseline_class_path(source_page, source)),
        reversed(_baseline_class_path(target_page, target)),
        strict=False,
    ):
        if left != right:
            break
        matched += 1
    return matched


def _baseline_class_path(
    page: TreeEmbedding,
    node: ElementEmbedding,
) -> tuple[str, ...]:
    by_id = {element.id: element for element in page.elements}
    result = []
    current: ElementEmbedding | None = node
    while current is not None:
        result.append(_baseline_class_tail(current))
        current = by_id.get(str(current.parent_id or ""))
    return tuple(reversed(result))


def _baseline_episode(
    *,
    source: _BaselineTrace,
    target: _BaselineTrace,
    steps: list[dict[str, Any]],
    balanced_triplet: bool,
) -> dict[str, Any]:
    gold_count = sum(step["gold_alignment_available"] for step in steps)
    methods = {}
    for method in _REPLAY_BASELINES:
        predictions = [step["predictions"][method] for step in steps]
        executed = sum(prediction["execute"] for prediction in predictions)
        hits = sum(prediction["hit"] for prediction in predictions)
        methods[method] = {
            "executed_step_count": executed,
            "step_hit_count": hits,
            "gold_aligned_executed_step_count": sum(
                step["gold_alignment_available"] and step["predictions"][method]["execute"]
                for step in steps
            ),
            "wrong_execution_count": sum(
                step["gold_alignment_available"]
                and step["predictions"][method]["execute"]
                and not step["predictions"][method]["hit"]
                for step in steps
            ),
            "complete_resolution": bool(steps and executed == len(steps)),
            "complete_hit": bool(
                steps and gold_count == len(steps) and hits == len(steps)
            ),
            "prediction_reasons": dict(
                sorted(Counter(prediction["reason"] for prediction in predictions).items())
            ),
        }
    return {
        "episode_id": f"{source.task_id}:env100-env{target.environment_id}",
        "task_id": source.task_id,
        "source_trace_id": source.trace_id,
        "target_trace_id": target.trace_id,
        "target_environment_id": target.environment_id,
        "balanced_triplet": balanced_triplet,
        "source_selector_step_count": len(steps),
        "gold_aligned_step_count": gold_count,
        "fully_gold_aligned": bool(steps and gold_count == len(steps)),
        "methods": methods,
    }


def _aggregate_baseline_method(
    episodes: list[dict[str, Any]],
    *,
    method: str,
) -> dict[str, Any]:
    step_count = sum(item["source_selector_step_count"] for item in episodes)
    gold_count = sum(item["gold_aligned_step_count"] for item in episodes)
    executed = sum(item["methods"][method]["executed_step_count"] for item in episodes)
    hits = sum(item["methods"][method]["step_hit_count"] for item in episodes)
    gold_executed = sum(
        item["methods"][method]["gold_aligned_executed_step_count"]
        for item in episodes
    )
    wrong = sum(item["methods"][method]["wrong_execution_count"] for item in episodes)
    complete_resolution = sum(
        item["methods"][method]["complete_resolution"] for item in episodes
    )
    complete_hit = sum(item["methods"][method]["complete_hit"] for item in episodes)
    comparable = [item for item in episodes if item["fully_gold_aligned"]]
    balanced = [item for item in episodes if item["balanced_triplet"]]
    balanced_resolution = sum(
        item["methods"][method]["complete_resolution"] for item in balanced
    )
    balanced_complete = sum(
        item["methods"][method]["complete_hit"] for item in balanced
    )
    comparable_complete = sum(
        item["methods"][method]["complete_hit"] for item in comparable
    )
    reasons: Counter[str] = Counter()
    for item in episodes:
        reasons.update(item["methods"][method]["prediction_reasons"])
    result = {
        "episode_count": len(episodes),
        "source_selector_step_count": step_count,
        "gold_aligned_step_count": gold_count,
        "gold_alignment_coverage": _baseline_rate(gold_count, step_count),
        "executed_step_count": executed,
        "selector_coverage": _baseline_rate(executed, step_count),
        "resolved_step_count": executed,
        "resolution_rate": _baseline_rate(executed, step_count),
        "step_hit_count": hits,
        "step_hit_rate": _baseline_rate(hits, step_count),
        "executed_step_precision": _baseline_rate(hits, gold_executed),
        "gold_aligned_executed_step_count": gold_executed,
        "wrong_execution_count": wrong,
        "abstained_step_count": step_count - executed,
        "complete_resolution_episode_count": complete_resolution,
        "complete_resolution_rate": _baseline_rate(complete_resolution, len(episodes)),
        "complete_hit_episode_count": complete_hit,
        "complete_hit_rate": _baseline_rate(complete_hit, len(episodes)),
        "fully_gold_aligned_episode_count": len(comparable),
        "fully_gold_aligned_complete_hit_count": comparable_complete,
        "fully_gold_aligned_complete_hit_rate": _baseline_rate(
            comparable_complete,
            len(comparable),
        ),
        "strict_complete_hit_failure_breakdown": {
            "missing_action_reference": sum(
                not item["fully_gold_aligned"]
                and not item["methods"][method]["complete_hit"]
                for item in episodes
            ),
            "replay_disagreement_or_abstention": sum(
                item["fully_gold_aligned"]
                and not item["methods"][method]["complete_hit"]
                for item in episodes
            ),
        },
        "balanced_triplet_complete_resolution_episode_count": balanced_resolution,
        "balanced_triplet_complete_resolution_rate": _baseline_rate(
            balanced_resolution,
            len(balanced),
        ),
        "balanced_triplet_complete_hit_episode_count": balanced_complete,
        "balanced_triplet_complete_hit_rate": _baseline_rate(
            balanced_complete,
            len(balanced),
        ),
        "prediction_reasons": dict(sorted(reasons.items())),
        "by_target_environment": {},
    }
    result["by_target_environment"] = {
        environment: {
            key: value
            for key, value in _aggregate_baseline_method(
                [
                    item
                    for item in episodes
                    if item["target_environment_id"] == environment
                ],
                method=method,
            ).items()
            if key != "by_target_environment"
        }
        for environment in sorted(
            {item["target_environment_id"] for item in episodes}
        )
    } if len({item["target_environment_id"] for item in episodes}) > 1 else {}
    return result


def _baseline_project_offset(
    point: tuple[float, float],
    source_bounds: tuple[int, int, int, int],
    target_bounds: tuple[int, int, int, int],
) -> tuple[float, float]:
    source_width = max(1.0, float(source_bounds[2] - source_bounds[0]))
    source_height = max(1.0, float(source_bounds[3] - source_bounds[1]))
    offset_x = min(1.0, max(0.0, (point[0] - source_bounds[0]) / source_width))
    offset_y = min(1.0, max(0.0, (point[1] - source_bounds[1]) / source_height))
    return (
        target_bounds[0] + offset_x * (target_bounds[2] - target_bounds[0]),
        target_bounds[1] + offset_y * (target_bounds[3] - target_bounds[1]),
    )


def _baseline_actionable(element: ElementEmbedding) -> bool:
    attributes = element.attributes
    return bool(
        attributes.get("enabled", True)
        and attributes.get("visible", True)
        and any(
            attributes.get(key)
            for key in ("clickable", "editable", "focusable", "long_clickable")
        )
    )


def _baseline_stable_identity(element: ElementEmbedding) -> bool:
    return any(
        _baseline_normalized(element.attributes.get(key))
        for key in ("resource_id", "text", "content_description")
    )


def _baseline_class_tail(element: ElementEmbedding) -> str:
    return _baseline_tail(str(element.attributes.get("class") or ""))


def _baseline_tail(value: str) -> str:
    return _baseline_normalized(value).rsplit("/", 1)[-1].rsplit(".", 1)[-1]


def _baseline_normalized(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _baseline_scaled_jaccard(
    left: frozenset[str],
    right: frozenset[str],
) -> int:
    union = left | right
    return round(1000.0 * len(left & right) / len(union)) if union else 0


def _baseline_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"bmoca_jsonl_object_required:{path}:{line_number}")
        values.append(value)
    return values


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
    parser.add_argument(
        "--suite",
        choices=("transfer-dp", "replay-baselines"),
        default="transfer-dp",
    )
    parser.add_argument("--target-env", action="append", dest="target_environments")
    parser.add_argument("--limit-tasks", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--full-matrix", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    environments = tuple(args.target_environments or ("101", "105"))
    if args.suite == "replay-baselines":
        report = evaluate_replay_baselines(
            args.corpus,
            target_environments=environments,
            limit_tasks=args.limit_tasks,
        )
        _write_report(args.output.expanduser().resolve(), report)
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        return 0
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
    "evaluate_replay_baselines",
    "evaluate_trace_pair",
    "load_bmoca_traces",
    "main",
]
