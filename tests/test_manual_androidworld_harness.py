import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

_HARNESS_PATH = Path(__file__).parents[1] / "tools" / "manual_androidworld_harness.py"
_SPEC = importlib.util.spec_from_file_location("manual_androidworld_harness", _HARNESS_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
ManualAndroidWorld = _MODULE.ManualAndroidWorld
_find_ui_element_index = _MODULE._find_ui_element_index


def test_find_ui_element_index_accepts_native_dicts_and_exact_selector():
    elements = [
        {"resource_name": "com.example:id/other", "text": "Submit"},
        {"resource_name": "submitButton", "text": "Submit"},
    ]

    assert _find_ui_element_index(elements, resource_name="submitButton") == 1


def test_find_ui_element_index_accepts_native_objects():
    elements = [SimpleNamespace(resource_name="clearButton", text="Clear")]

    assert _find_ui_element_index(elements, text="Clear") == 0


def test_find_ui_element_index_rejects_ambiguous_or_missing_targets():
    elements = [
        {"content_description": "More options"},
        {"content_description": "More options"},
    ]

    with pytest.raises(ValueError, match="multiple current UI elements"):
        _find_ui_element_index(elements, content_description="More options")
    with pytest.raises(ValueError, match="no current UI element"):
        _find_ui_element_index(elements, resource_name="submitButton")


def test_find_ui_element_index_requires_one_selector():
    with pytest.raises(ValueError, match="exactly one"):
        _find_ui_element_index([], text="Submit", resource_name="submitButton")


def test_wait_duration_uses_native_json_action_and_sleeps(monkeypatch):
    """The friendly duration must not be forwarded to JSONAction."""

    class FakeJSONAction:
        def __init__(self, *, action_type):
            self.action_type = action_type

        def as_dict(self):
            return {"action_type": self.action_type}

    class FakeEnv:
        def __init__(self):
            self.actions = []

        def execute_action(self, action):
            self.actions.append(action)

    harness = ManualAndroidWorld.__new__(ManualAndroidWorld)
    harness._json_action = type("JsonActionModule", (), {"JSONAction": FakeJSONAction})
    harness._env = FakeEnv()
    harness._last_observation = {"pixels": None, "forest": {}, "ui_elements": [], "auxiliaries": {}}
    harness._steps = []
    harness._finished = False
    harness._write_run_log = lambda **_: None

    observation = {"pixels": None, "forest": {}, "ui_elements": [], "auxiliaries": {}}
    monkeypatch.setattr(harness, "observe", lambda: {"observation": observation})
    slept = []
    monkeypatch.setattr(_MODULE.time, "sleep", slept.append)

    result = harness.act(
        {
            "action_type": "wait",
            "duration": 2.5,
            "reasoning": "The page is transitioning, so I wait before observing again.",
        }
    )

    assert result["action"]["duration"] == 2.5
    assert [action.action_type for action in harness._env.actions] == ["wait"]
    assert slept == [2.5]
    assert harness._steps[0]["action"] == {"action_type": "wait", "duration": 2.5}
    assert harness._steps[0]["metadata"]["reasoning"]


def test_act_requires_reasoning_before_executing_action():
    class FakeJSONAction:
        def __init__(self, **kwargs):
            self.action_type = kwargs["action_type"]

    class FakeEnv:
        def execute_action(self, action):
            raise AssertionError("action must not execute")

    harness = ManualAndroidWorld.__new__(ManualAndroidWorld)
    harness._json_action = type("JsonActionModule", (), {"JSONAction": FakeJSONAction})
    harness._env = FakeEnv()
    harness._last_observation = {"pixels": None, "forest": {}, "ui_elements": [], "auxiliaries": {}}
    harness._steps = []

    with pytest.raises(ValueError, match="reasoning_required"):
        harness.act({"action_type": "wait"})


def test_run_log_records_protocol_source_seed_not_task_parameter_seed(tmp_path):
    harness = ManualAndroidWorld.__new__(ManualAndroidWorld)
    harness._root = tmp_path
    harness._run_id = "manual_test"
    harness._task = SimpleNamespace(
        name="ExampleTask",
        goal="Complete the example",
        params={"seed": 987654321, "value": "kept"},
    )
    harness._source_seed = 111
    harness._started_ms = 1
    harness._steps = []
    harness._last_observation = None
    harness._validation_reasoning = ""
    harness._device_serial = "emulator-5560"

    harness._write_run_log(status="running", success=False, reward=0.0)

    payload = json.loads((tmp_path / "run_log.json").read_text())
    assert payload["seed"] == 111
    assert payload["task_parameters"] == {"seed": 987654321, "value": "kept"}
