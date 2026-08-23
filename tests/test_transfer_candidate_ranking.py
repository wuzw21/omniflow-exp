import inspect
from types import SimpleNamespace

import omniflow.transfer.runtime as transfer_runtime


def test_transfer_adapter_matches_the_candidate_api() -> None:
    assert tuple(inspect.signature(transfer_runtime.transfer_action).parameters) == (
        "source_xml",
        "target_xml",
        "source_point",
        "source_element_id",
        "source_offset",
        "source_screenshot_path",
        "target_screenshot_path",
        "source_visual_rgb",
        "target_visual_rgb",
        "action_type",
        "top_k",
    )


def _ranking(*, candidates, reason="learned_low_confidence"):
    return {
        "schema_version": "omnitransfer.candidate-ranking.v1",
        "status": "scored" if candidates else "no_candidates",
        "reason": reason,
        "mapping_mode": "omnitransfer_test",
        "score": 0.0001,
        "margin": 0.0,
        "candidates": candidates,
        "top_candidates": candidates[:1],
    }


def test_omniflow_owns_candidate_api_preflight(monkeypatch) -> None:
    module = SimpleNamespace(
        rank_action_candidates=lambda **_kwargs: {
            **_ranking(
                candidates=[
                    {
                        "candidate_id": "search",
                        "bbox": [40.0, 100.0, 120.0, 260.0],
                        "score": 1.0,
                        "new_x": 80.0,
                        "new_y": 180.0,
                    }
                ],
                reason="equivalent_ui_graph",
            ),
            "matcher_backend": "test",
            "matcher_release": "test-release",
            "matcher_checkpoint_sha256": "abc",
            "matcher_feature_schema": "test-schema",
            "matcher_feature_schema_sha256": "def",
        }
    )
    monkeypatch.setattr(transfer_runtime, "load_omnitransfer", lambda: module)

    result = transfer_runtime.preflight_omnitransfer()

    assert result["ready"] is True
    assert result["backend"] == "test"
    assert result["candidate_ranking_schema"] == (
        "omnitransfer.candidate-ranking.v1"
    )


def test_omniflow_selects_rank_one_without_a_confidence_gate(monkeypatch) -> None:
    calls = []
    module = SimpleNamespace(
        rank_action_candidates=lambda **kwargs: calls.append(kwargs)
        or _ranking(
            candidates=[
                {
                    "candidate_id": "first",
                    "score": 0.51,
                    "bbox": [10.0, 20.0, 30.0, 40.0],
                    "new_x": 15.0,
                    "new_y": 25.0,
                },
                {
                    "candidate_id": "second",
                    "score": 0.49,
                    "bbox": [50.0, 60.0, 70.0, 80.0],
                    "new_x": 55.0,
                    "new_y": 65.0,
                },
            ]
        ),
        action_transfer=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy action_transfer must not be used")
        ),
    )
    monkeypatch.setattr(transfer_runtime, "load_omnitransfer", lambda: module)

    result = transfer_runtime.transfer_action(
        source_xml="<hierarchy />",
        target_xml="<hierarchy />",
        source_point=(1.0, 2.0),
    )

    assert len(calls) == 1
    assert result["mapped"] is True
    assert result["selection_policy"] == "omniflow_top_candidate"
    assert result["target_candidate_id"] == "first"
    assert (result["new_x"], result["new_y"]) == (15.0, 25.0)
    assert len(result["candidates"]) == 2


def test_omniflow_does_not_gate_candidates_by_page_identity(monkeypatch) -> None:
    module = SimpleNamespace(
        rank_action_candidates=lambda **_kwargs: _ranking(
            candidates=[
                {
                    "candidate_id": "wrong-page-candidate",
                    "score": 1.0,
                    "bbox": [10.0, 20.0, 30.0, 40.0],
                    "new_x": 15.0,
                    "new_y": 25.0,
                }
            ]
        )
    )
    monkeypatch.setattr(transfer_runtime, "load_omnitransfer", lambda: module)

    result = transfer_runtime.transfer_action(
        source_xml="<hierarchy />",
        target_xml="<hierarchy />",
        source_point=(1.0, 2.0),
    )

    assert result["mapped"] is True
    assert result["selection_policy"] == "omniflow_top_candidate"
    assert result["target_candidate_id"] == "wrong-page-candidate"
    assert (result["new_x"], result["new_y"]) == (15.0, 25.0)


def test_omniflow_accepts_matching_resource_id_anchor(monkeypatch) -> None:
    module = SimpleNamespace(
        rank_action_candidates=lambda **_kwargs: _ranking(
            candidates=[
                {
                    "candidate_id": "stopwatch",
                    "resource_id": "com.google.android.deskclock:id/tab_menu_stopwatch",
                    "content_desc": "Stopwatch",
                    "class": "android.widget.FrameLayout",
                    "score": 0.99,
                    "bbox": [432.0, 1072.0, 576.0, 1232.0],
                    "new_x": 504.0,
                    "new_y": 1152.0,
                }
            ]
        )
    )
    monkeypatch.setattr(transfer_runtime, "load_omnitransfer", lambda: module)

    result = transfer_runtime.transfer_action(
        source_xml=(
            '<hierarchy><node resource-id="com.google.android.deskclock:id/tab_menu_stopwatch" '
            'content-desc="Stopwatch" class="android.widget.FrameLayout" '
            'bounds="[432,1072][576,1232]" /></hierarchy>'
        ),
        target_xml="<hierarchy />",
        source_point=(504.0, 1152.0),
    )

    assert result["mapped"] is True
    assert result["target_candidate_id"] == "stopwatch"


def test_omniflow_owns_empty_candidate_fallback(monkeypatch) -> None:
    module = SimpleNamespace(
        rank_action_candidates=lambda **_kwargs: _ranking(
            candidates=[],
            reason="target_candidates_missing",
        )
    )
    monkeypatch.setattr(transfer_runtime, "load_omnitransfer", lambda: module)

    result = transfer_runtime.transfer_action(
        source_xml="<hierarchy />",
        target_xml="<hierarchy />",
        source_point=(1.0, 2.0),
    )

    assert result["mapped"] is False
    assert result["reason"] == "target_candidates_missing"
    assert result["candidates"] == []
