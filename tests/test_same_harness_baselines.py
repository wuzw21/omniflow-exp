"""Same-harness MobileGPT-style / AppAgent-style baselines (docs/BASELINES_SAME_HARNESS.md)."""

import pytest

from omniflow.core.model import Action, Observation
from omniflow.transfer.attribute import attribute_transfer
from omniflow.vlm.gui import _turn_text
from omniflow.vlm.planner import planner_state

DISPLAY = {"display": {"width": 1000, "height": 2000}}


def _page(*nodes):
    body = "".join(
        f'<node resource-id="{rid}" text="{text}" content-desc="" class="android.widget.Button" '
        f'bounds="[{l},{t}][{r},{b}]" />'
        for rid, text, (l, t, r, b) in nodes
    )
    return Observation(xml=f'<hierarchy><node bounds="[0,0][1000,2000]">{body}</node></hierarchy>', extra=DISPLAY)


def test_attribute_transfer_relocates_element_by_resource_id():
    source = _page(("com.app:id/save", "Save", (0, 0, 200, 100)))
    target = _page(("com.app:id/other", "Cancel", (0, 0, 200, 100)),
                   ("com.app:id/save", "Save", (600, 1000, 800, 1100)))
    result = attribute_transfer(Action("click", {"x": 50.0, "y": 25.0}), target, source)
    assert result.action is not None
    assert result.reason == "attribute_match_resource_id_text"
    assert result.action.args["x"] == 700.0
    assert result.action.args["y"] == 525.0


def test_attribute_transfer_falls_back_to_text_then_fails_without_match():
    source = _page(("", "Save", (0, 0, 200, 100)))
    moved = _page(("", "Save", (0, 400, 200, 500)))
    assert attribute_transfer(Action("click", {"x": 50.0, "y": 25.0}), moved, source).reason == "attribute_match_text"
    gone = _page(("", "Delete", (0, 0, 200, 100)))
    result = attribute_transfer(Action("click", {"x": 50.0, "y": 25.0}), gone, source)
    assert result.action is None and result.reason == "attribute_no_match"


def test_attribute_transfer_passes_non_pointer_actions_through():
    action = Action("navigate_back", {})
    assert attribute_transfer(action, _page(), None).action == action


def test_element_docs_reach_the_planner_prompt():
    observation = Observation(xml="<hierarchy/>", extra={**DISPLAY, "element_docs": "- Save: tap: stores the note"})
    state = planner_state(observation)
    text = _turn_text(goal="g", state=state, max_steps=5, turn_index=1, target_package_name="", xml_text="")
    assert "Documentation of UI elements on this screen:\n- Save: tap: stores the note" in text


def test_appagent_docs_render_only_for_elements_on_screen(tmp_path):
    from src.integrations.android_world.agent import element_docs_for_xml, load_element_docs

    (tmp_path / "com.app.id_save.txt").write_text(
        "{'tap': 'stores the note', 'text': '', 'v_swipe': '', 'h_swipe': '', 'long_press': ''}",
        encoding="utf-8",
    )
    (tmp_path / "com.app.id_gone.txt").write_text("{'tap': 'never shown'}", encoding="utf-8")
    docs = load_element_docs(tmp_path)
    rendered = element_docs_for_xml(_page(("com.app:id/save", "Save", (0, 0, 200, 100))).xml, docs)
    assert rendered == "- Save [[0,0][200,100]]: tap: stores the note"


def test_same_harness_baseline_requires_builtin_omniflow_with_store(monkeypatch):
    from src.experiment import run_tasks

    monkeypatch.delenv("OMNIFLOW_BASELINE", raising=False)
    with pytest.raises(ValueError, match="same_harness_baseline_requires"):
        run_tasks.main(["run", "--method", "omniflow", "--baseline", "mobilegpt_style", "--dry-run"])
    with pytest.raises(ValueError, match="appagent_style_requires_element_docs"):
        run_tasks.main(["run", "--method", "omniflow", "--baseline", "appagent_style",
                        "--memory", "m", "--dry-run"])


def test_avd_snapshot_uses_emulator_console_and_rejects_ko(monkeypatch):
    import subprocess

    from src.experiment import run_tasks

    calls, waited = [], []
    replies = iter(("OK", "KO: snapshot 'clean' not found"))

    def fake_run(command, **kwargs):
        calls.append(tuple(command[1:]))
        return subprocess.CompletedProcess(command, 0, next(replies), "")

    monkeypatch.setattr(run_tasks, "_sdk_tool", lambda name: name)
    monkeypatch.setattr(run_tasks.subprocess, "run", fake_run)
    monkeypatch.setattr(run_tasks, "_wait_for_device", lambda adb, serial, timeout: waited.append(serial))
    run_tasks._avd_snapshot("emulator-5554", "load", "clean")
    assert calls == [("-s", "emulator-5554", "emu", "avd", "snapshot", "load", "clean")]
    assert waited == ["emulator-5554"]
    with pytest.raises(RuntimeError, match="avd_snapshot_save_failed"):
        run_tasks._avd_snapshot("emulator-5554", "save", "clean")
    with pytest.raises(ValueError, match="avd_snapshot_invalid"):
        run_tasks._avd_snapshot("emulator-5554", "delete", "clean")


def test_demo_hint_baseline_needs_source_run_log_and_disables_functions(monkeypatch):
    from src.experiment import run_tasks
    from src.integrations.android_world import agent

    monkeypatch.delenv("OMNIFLOW_BASELINE", raising=False)
    with pytest.raises(ValueError, match="same_harness_baseline_requires"):
        run_tasks.main(["run", "--method", "omniflow", "--baseline", "demo_hint", "--memory", "m", "--dry-run"])
    monkeypatch.setenv("OMNIFLOW_BASELINE", "demo_hint")
    assert agent.same_harness_baseline() == "demo_hint"
    monkeypatch.setenv("OMNIFLOW_BASELINE", "bogus")
    with pytest.raises(ValueError, match="unknown_same_harness_baseline"):
        agent.same_harness_baseline()

