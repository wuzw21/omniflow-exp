from __future__ import annotations

import json
from types import SimpleNamespace

from PIL import Image

from src.experiment.observation_evidence import ObservationArchive


def test_archive_preserves_every_observation_and_deduplicates_images(
    tmp_path,
) -> None:
    states = [
        SimpleNamespace(
            pixels=Image.new("RGB", (4, 3), color="red"),
            package_name="com.android.settings",
            activity_name="com.android.settings/.Settings",
        ),
        SimpleNamespace(
            pixels=Image.new("RGB", (4, 3), color="red"),
            package_name="com.android.settings",
            activity_name="com.android.settings/.Settings",
        ),
    ]
    remaining = iter(states)
    archive = ObservationArchive(lambda: next(remaining))

    assert archive.get_state() is states[0]
    assert archive.get_state() is states[1]
    records = archive.persist(tmp_path)

    assert [record["observation_index"] for record in records] == [0, 1]
    assert records[0]["path"] == records[1]["path"]
    assert records[0]["sha256"] == records[1]["sha256"]
    assert records[0]["display"] == {"width": 4, "height": 3}
    assert records[0]["package_name"] == "com.android.settings"
    assert len(list((tmp_path / "observations" / "objects").glob("*.png"))) == 1
    assert json.loads((tmp_path / "observations" / "index.json").read_text()) == {
        "schema_version": "omniflow.androidworld-observations.v1",
        "observation_count": 2,
        "observations": records,
    }


def test_archive_reports_an_observation_without_pixels(tmp_path) -> None:
    state = SimpleNamespace(
        pixels=None,
        package_name="com.android.settings",
        activity_name="com.android.settings/.Settings",
    )
    archive = ObservationArchive(lambda: state)

    assert archive.get_state() is state

    assert archive.persist(tmp_path) == [
        {
            "observation_index": 0,
            "package_name": "com.android.settings",
            "activity_name": "com.android.settings/.Settings",
            "error": "observation_image_missing",
        }
    ]
