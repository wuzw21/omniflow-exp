from __future__ import annotations

from omniflow.transfer.runtime import _select_transfer_candidate


def test_selects_highest_ranked_executable_candidate() -> None:
    result = _select_transfer_candidate(
        {
            "candidates": [
                {
                    "candidate_id": "static-label",
                    "execution_candidate_id": "",
                    "executable": False,
                    "bbox": [0, 0, 100, 20],
                    "execution_bbox": [0, 0, 100, 20],
                    "new_x": 50,
                    "new_y": 10,
                    "score": 0.9,
                },
                {
                    "candidate_id": "lower-ranked-button",
                    "execution_candidate_id": "button",
                    "executable": True,
                    "bbox": [0, 20, 100, 60],
                    "execution_bbox": [0, 20, 100, 60],
                    "new_x": 50,
                    "new_y": 40,
                    "score": 0.1,
                },
            ]
        },
        {
            "source_package_name": "net.gsantner.markor",
            "target_package_name": "net.gsantner.markor",
        },
    )

    assert result["mapped"] is True
    assert result["target_candidate_id"] == "lower-ranked-button"
    assert result["new_x"] == 50
    assert result["new_y"] == 40


def test_rejects_when_all_candidates_are_non_executable() -> None:
    result = _select_transfer_candidate(
        {
            "candidates": [
                {
                    "candidate_id": "static-label",
                    "execution_candidate_id": "",
                    "executable": False,
                    "bbox": [0, 0, 100, 20],
                    "execution_bbox": [0, 0, 100, 20],
                    "new_x": 50,
                    "new_y": 10,
                }
            ]
        },
        {
            "source_package_name": "net.gsantner.markor",
            "target_package_name": "net.gsantner.markor",
        },
    )

    assert result["mapped"] is False
    assert result["reason"] == "target_candidate_not_executable"
