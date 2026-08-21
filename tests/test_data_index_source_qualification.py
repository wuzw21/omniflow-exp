from __future__ import annotations

import hashlib
from pathlib import Path

from src.experiment.data_index import _require_qualified_source_run_log
from runlog_fixtures import androidworld_run_log, androidworld_state


def test_complete_source_run_log_seed_is_provenance_only(tmp_path: Path) -> None:
    before_path = tmp_path / "before.png"
    after_path = tmp_path / "after.png"
    before_path.write_bytes(b"before")
    after_path.write_bytes(b"after")

    def state(name: str, screenshot: Path) -> dict[str, object]:
        value = androidworld_state(name, with_pixels=False)
        value["pixels"] = {
            "path": str(screenshot),
            "sha256": hashlib.sha256(screenshot.read_bytes()).hexdigest(),
            "width": 720,
            "height": 1280,
            "mime_type": "image/png",
        }
        return value

    before = state("before", before_path)
    after = state("after", after_path)
    run_log = androidworld_run_log(
        [{"action_type": "open_app", "app_name": "com.example.app"}],
        observations=[before],
        task_name="Task",
        seed=987654321,
        success=True,
    )
    run_log["steps"][0]["next_observation"] = after
    run_log["steps"][0]["metadata"] = {
        "reasoning": "Open the requested app.",
        "screenshot_path": str(after_path),
    }

    qualified = _require_qualified_source_run_log(
        run_log,
        task="Task",
        source_metadata=None,
    )

    assert qualified["seed"] == 987654321
