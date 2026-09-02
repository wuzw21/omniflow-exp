import json

from src.experiment.run_tasks import _promote_golden_run


def test_promoted_evidence_paths_follow_immutable_destination(tmp_path) -> None:
    candidate = tmp_path / "candidate"
    destination = tmp_path / "golden" / "runlog" / "current"
    screenshot = candidate / "observations" / "objects" / "screen.png"
    screenshot.parent.mkdir(parents=True)
    screenshot.write_bytes(b"png")
    run_log = {
        "status": "succeeded",
        "success": True,
        "validator": {"official": True},
        "steps": [
            {
                "observation": {"screenshot": {"path": str(screenshot)}},
                "action": {"action_type": "click"},
            }
        ],
    }
    (candidate / "run_log.json").write_text(
        json.dumps(run_log), encoding="utf-8"
    )
    (candidate / "target.transfer_states.json").write_text(
        json.dumps({"screenshot_path": str(screenshot)}), encoding="utf-8"
    )
    (candidate / "task_results.jsonl").write_text(
        json.dumps({"path": str(screenshot)}) + "\n", encoding="utf-8"
    )

    assert _promote_golden_run(candidate=candidate, destination=destination)

    promoted = json.loads((destination / "run_log.json").read_text())
    promoted_path = promoted["steps"][0]["observation"]["screenshot"]["path"]
    assert promoted_path == str(destination / "observations" / "objects" / "screen.png")
    assert (destination / "observations" / "objects" / "screen.png").is_file()

    states = json.loads((destination / "target.transfer_states.json").read_text())
    assert states["screenshot_path"] == promoted_path
    result = json.loads((destination / "task_results.jsonl").read_text())
    assert result["path"] == promoted_path
