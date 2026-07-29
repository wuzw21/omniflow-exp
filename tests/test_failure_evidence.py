from __future__ import annotations

import base64
import hashlib

from src.experiment.failure_evidence import write_failure_observations


def test_failure_observations_are_content_addressed_and_referenced(tmp_path) -> None:
    image = b"\x89PNG\r\n\x1a\nsame-observation"
    encoded = base64.b64encode(image).decode("ascii")

    records = write_failure_observations(
        tmp_path,
        task_name="SystemWifiTurnOn",
        run_id="run_example",
        observations=[
            {
                "event": "action_failure",
                "step_index": 3,
                "error": "low_confidence",
                "state_id": "state_failed",
                "image_base64": encoded,
                "display": {"width": 2208, "height": 1840},
            },
            {
                "event": "terminal_failure",
                "state_id": "state_failed",
                "image_base64": encoded,
                "display": {"width": 2208, "height": 1840},
            },
        ],
    )

    digest = hashlib.sha256(image).hexdigest()
    expected_path = f"failure_evidence/objects/{digest}.png"
    assert [record["path"] for record in records] == [
        expected_path,
        expected_path,
    ]
    assert records[0] == {
        "event": "action_failure",
        "step_index": 3,
        "error": "low_confidence",
        "state_id": "state_failed",
        "display": {"width": 2208, "height": 1840},
        "path": expected_path,
        "sha256": digest,
    }
    assert len(list((tmp_path / "failure_evidence" / "objects").glob("*.png"))) == 1
    assert (tmp_path / expected_path).read_bytes() == image
