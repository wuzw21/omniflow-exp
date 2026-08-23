from __future__ import annotations

from types import SimpleNamespace

import omniflow.transfer.runtime as transfer_runtime


def test_transfer_adapter_keeps_policy_fields_out_of_candidate_api(
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    def rank_action_candidates(
        *,
        target_xml: str,
        source_xml: str | None = None,
        source_point: tuple[float, float] | None = None,
        source_element_id: str | None = None,
        source_offset: tuple[float, float] | None = None,
        source_screenshot_path: str | None = None,
        target_screenshot_path: str | None = None,
        source_visual_rgb: dict | None = None,
        target_visual_rgb: dict | None = None,
        action_type: str = "click",
        top_k: int = 1,
    ) -> dict:
        calls.append(
            {
                "target_xml": target_xml,
                "source_xml": source_xml,
                "source_point": source_point,
                "source_element_id": source_element_id,
                "source_offset": source_offset,
                "source_screenshot_path": source_screenshot_path,
                "target_screenshot_path": target_screenshot_path,
                "source_visual_rgb": source_visual_rgb,
                "target_visual_rgb": target_visual_rgb,
                "action_type": action_type,
                "top_k": top_k,
            }
        )
        return {
            "schema_version": "omnitransfer.candidate-ranking.v1",
            "status": "scored",
            "reason": "equivalent_ui_graph",
            "candidates": [
                {
                    "candidate_id": "target",
                    "score": 1.0,
                    "new_x": 10.0,
                    "new_y": 20.0,
                }
            ],
        }

    monkeypatch.setattr(
        transfer_runtime,
        "load_omnitransfer",
        lambda: SimpleNamespace(rank_action_candidates=rank_action_candidates),
    )

    result = transfer_runtime.transfer_action(
        source_xml="<hierarchy />",
        target_xml="<hierarchy />",
        source_point=(1.0, 2.0),
        source_package_name="com.source",
        target_package_name="com.target",
        source_activity_name=".Source",
        target_activity_name=".Target",
        action_type="click",
        top_k=3,
    )

    assert len(calls) == 1
    assert result["mapped"] is False
    assert result["reason"] == "target_page_identity_mismatch"
