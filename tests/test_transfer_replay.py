from __future__ import annotations

import json

from src.experiment import transfer_replay
from src.experiment.bmoca_replay import (
    BmocaStep,
    BmocaTrace,
    evaluate_trace_pair,
)
from src.experiment.transfer_replay import (
    ReplayToken,
    TransferMatchScore,
    align_transfer_replay,
)


def test_transfer_score_uses_candidate_probability_without_identity_constraints(
    monkeypatch,
) -> None:
    request = {}

    def transfer_action(**kwargs):
        request.update(kwargs)
        return {
            "mapped": True,
            "mapping_mode": "mutual_graph_matcher_no_null_v3",
            "new_x": 900.0,
            "new_y": 900.0,
            "target_bbox": [800.0, 800.0, 1000.0, 1000.0],
            "score": 0.99,
            "top_candidates": [
                {"bbox": [800.0, 800.0, 1000.0, 1000.0], "score": 0.99},
                {"bbox": [20.0, 40.0, 220.0, 240.0], "score": 0.93},
            ],
        }

    monkeypatch.setattr(transfer_replay, "transfer_action", transfer_action)

    evidence = transfer_replay.score_transfer_match(
        source_xml="<hierarchy />",
        target_xml="<hierarchy />",
        source_point=(50.0, 60.0),
        target_bounds=(20.0, 40.0, 220.0, 240.0),
    )

    assert evidence.probability == 0.93
    assert evidence.top_probability == 0.99
    assert evidence.exact_hit is False
    assert set(request) == {"source_xml", "target_xml", "source_point", "top_k"}


def test_cached_node_score_reprojects_the_current_action_offset(monkeypatch) -> None:
    calls = 0

    def transfer_action(**_kwargs):
        nonlocal calls
        calls += 1
        return {
            "mapped": True,
            "new_x": 250.0,
            "new_y": 250.0,
            "score": 0.98,
            "src_element": {"bounds": [0.0, 0.0, 100.0, 100.0]},
            "target_bbox": [200.0, 200.0, 400.0, 400.0],
            "top_candidates": [
                {"bbox": [200.0, 200.0, 400.0, 400.0], "score": 0.97}
            ],
        }

    monkeypatch.setattr(transfer_replay, "transfer_action", transfer_action)
    base = transfer_replay.score_transfer_match(
        source_xml="<hierarchy />",
        target_xml="<hierarchy />",
        source_point=(25.0, 25.0),
    )

    reused = transfer_replay.retarget_transfer_score(
        base,
        source_point=(75.0, 75.0),
        target_bounds=(200.0, 200.0, 400.0, 400.0),
    )

    assert calls == 1
    assert reused.probability == 0.97
    assert reused.mapped_point == (350.0, 350.0)
    assert reused.exact_hit is True


def test_replay_dp_skips_global_action_inside_sequence() -> None:
    source = (
        ReplayToken(0, "click"),
        ReplayToken(1, "open_app"),
        ReplayToken(2, "click"),
    )
    target = (ReplayToken(0, "click"), ReplayToken(1, "input_text"))
    probabilities = (
        (0.97, None),
        (None, None),
        (None, 0.96),
    )

    alignment = align_transfer_replay(source, target, probabilities)

    assert [(pair.source_index, pair.target_index) for pair in alignment.pairs] == [
        (0, 0),
        (2, 1),
    ]
    assert [gap.index for gap in alignment.source_gaps] == [1]
    assert alignment.source_gaps[0].action_kind == "open_app"


def test_replay_dp_does_not_treat_click_gap_like_global_gap() -> None:
    target = (ReplayToken(0, ""),)
    probabilities = ((0.78,), (0.90,))

    click_first = align_transfer_replay(
        (ReplayToken(0, "click"), ReplayToken(1, "click")),
        target,
        probabilities,
        mode="target_prefix",
    )
    global_first = align_transfer_replay(
        (ReplayToken(0, "press_key"), ReplayToken(1, "click")),
        target,
        probabilities,
        mode="target_prefix",
    )

    assert click_first.pairs[-1].source_index == 0
    assert global_first.pairs[-1].source_index == 1


def test_replay_dp_has_no_action_type_hard_constraint() -> None:
    alignment = align_transfer_replay(
        (ReplayToken(4, "click"),),
        (ReplayToken(7, "input_text"),),
        ((0.96,),),
    )

    assert len(alignment.pairs) == 1
    assert alignment.pairs[0].source_index == 4
    assert alignment.pairs[0].target_index == 7


def test_bmoca_dp_beats_fixed_index_when_global_action_is_missing() -> None:
    source = BmocaTrace(
        trace_id="source",
        task_id="task",
        environment_id="100",
        steps=(
            BmocaStep(0, "open_app"),
            BmocaStep(1, "click", point=(10.0, 10.0), bounds=(0.0, 0.0, 20.0, 20.0)),
            BmocaStep(2, "click", point=(50.0, 50.0), bounds=(40.0, 40.0, 60.0, 60.0)),
        ),
    )
    target = BmocaTrace(
        trace_id="target",
        task_id="task",
        environment_id="101",
        steps=(
            BmocaStep(0, "input_text", point=(110.0, 110.0), bounds=(100.0, 100.0, 120.0, 120.0)),
            BmocaStep(1, "long_press", point=(150.0, 150.0), bounds=(140.0, 140.0, 160.0, 160.0)),
        ),
    )

    def scorer(source_step: BmocaStep, target_step: BmocaStep) -> TransferMatchScore:
        probabilities = {(1, 0): 0.97, (2, 1): 0.96}
        probability = probabilities.get((source_step.index, target_step.index))
        return TransferMatchScore(
            probability=probability,
            exact_hit=probability is not None,
        )

    result = evaluate_trace_pair(source, target, scorer=scorer)

    assert result["dp"]["exact_hit_count"] == 2
    assert result["dp"]["complete_hit"] is True
    assert result["fixed_index"]["exact_hit_count"] == 0
    assert result["fixed_index"]["complete_hit"] is False


def test_bmoca_loader_keeps_only_successful_actions_and_virtual_semantic_bounds(
    tmp_path,
) -> None:
    trace_root = tmp_path / "traces" / "trace-one"
    trace_root.mkdir(parents=True)
    manifest = {
        "schema_version": "omniflow.offline-trace-corpus.v1",
        "traces": [
            {
                "trace_id": "trace-one",
                "task_id": "clock/task",
                "environment_id": "100",
                "success_evidence": {"official_success": True},
                "runlog": {"path": "traces/trace-one/runlog.json"},
                "state_catalog": {"path": "traces/trace-one/transfer_states.json"},
            }
        ],
    }
    runlog = {
        "schema_version": "omniflow.canonical_run_log.v1",
        "steps": [
            {
                "step_index": 0,
                "before_state_id": "state-one",
                "action": {"tool": "click", "args": {"x": 500, "y": 500}},
                "result": {"success": False},
            },
            {
                "step_index": 1,
                "before_state_id": "state-one",
                "action": {"tool": "click", "args": {"x": 500, "y": 500}},
                "result": {"success": True},
            },
        ],
    }
    catalog = {
        "schema_version": "omniflow.transfer-state-catalog.v1",
        "states": {
            "state-one": {
                "state_id": "state-one",
                "display": {"width": 100, "height": 100},
                "xml": (
                    '<hierarchy bounds="[0,0][100,100]">'
                    '<node content-desc="30" enabled="true" displayed="true" '
                    'clickable="false" bounds="[40,40][60,60]" />'
                    "</hierarchy>"
                ),
            }
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (trace_root / "runlog.json").write_text(json.dumps(runlog), encoding="utf-8")
    (trace_root / "transfer_states.json").write_text(
        json.dumps(catalog), encoding="utf-8"
    )

    from src.experiment.bmoca_replay import load_bmoca_traces

    traces = load_bmoca_traces(tmp_path)

    assert len(traces) == 1
    assert [step.index for step in traces[0].steps] == [1]
    assert traces[0].steps[0].bounds == (40.0, 40.0, 60.0, 60.0)
