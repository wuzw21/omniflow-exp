from __future__ import annotations

from omniflow.bridge import _run_result
from omniflow.core.model import Function, RunResult
from omniflow.vlm.planner import (
    DEFAULT_STEP_GUIDANCE,
    build_model_turn_request,
    resolve_step_guidance,
)
from src.experiment import bmoca_device_replay


def _function() -> Function:
    return Function.from_dict(
        {
            "schema_version": "omniflow.function.v2",
            "function_id": "order_beverage",
            "name": "Order a beverage",
            "description": "Order the requested beverage and stop before payment.",
            "input_schema": {
                "type": "object",
                "properties": {"beverage": {"type": "string"}},
                "required": ["beverage"],
                "additionalProperties": False,
            },
            "bindings": [],
            "steps": [],
            "checker_rules": [],
            "agent_visible": True,
        }
    )


def test_planner_guidance_is_explicit_only() -> None:
    assert resolve_step_guidance("find a contact") == DEFAULT_STEP_GUIDANCE
    assert DEFAULT_STEP_GUIDANCE == ""
    assert resolve_step_guidance("order coffee", "custom") == "custom"


def test_bridge_planner_exposes_function_with_native_actions() -> None:
    function = _function()
    request = build_model_turn_request(
        goal="order me a latte",
        model="scene.vlm.operation.primary",
        state={"state_id": "state-1", "display": {"width": 100, "height": 200}},
        functions=(function,),
        max_steps=20,
        turn_index=1,
    )

    tool_names = [tool["function"]["name"] for tool in request["tools"]]
    assert "click" in tool_names
    assert "swipe" in tool_names
    assert function.id in tool_names


def test_core_has_one_planner_implementation() -> None:
    import omniflow.bridge as bridge
    from omniflow.vlm.planner import VLMPlanner

    assert not hasattr(bridge, "_BridgePlanner")
    assert VLMPlanner.__name__ == "VLMPlanner"


def test_successful_online_run_requests_registration_after_run() -> None:
    payload = _run_result(
        RunResult(
            True,
            actions_executed=3,
            detail={
                "trace": [],
                "done_reason": "finished",
                "function_resolution": {
                    "selected_function_id": None,
                },
            },
        ),
        body={"run_id": "run-1", "goal": "order coffee"},
        function=None,
    )

    assert payload["recall_hit"] is False
    assert payload["post_run_actions"] == [
        {
            "name": "save_function",
            "arguments": {
                "run_id": "run-1",
                "agent_visible": True,
            },
        }
    ]


def test_recalled_run_is_not_registered_again() -> None:
    payload = _run_result(
        RunResult(
            True,
            function_id="order_beverage",
            actions_executed=3,
            detail={
                "trace": [],
                "done_reason": "finished",
                "function_resolution": {
                    "selected_function_id": "order_beverage",
                },
            },
        ),
        body={"run_id": "run-2", "goal": "order latte"},
        function=None,
    )

    assert payload["recall_hit"] is True
    assert payload["recalled_function_id"] == "order_beverage"
    assert "post_run_actions" not in payload


def test_bmoca_waits_for_online_booted_emulator(monkeypatch) -> None:
    class Simulator:
        _adb_port = 5555

    class Environment:
        _simulator = Simulator()

    outputs = iter(
        [
            (1, "offline\n", ""),
            (0, "device\n", ""),
            (0, "1\n", ""),
        ]
    )

    def run(*_args, **_kwargs):
        returncode, stdout, stderr = next(outputs)
        return type(
            "Completed",
            (),
            {"returncode": returncode, "stdout": stdout, "stderr": stderr},
        )()

    monkeypatch.setattr(bmoca_device_replay.subprocess, "run", run)
    monkeypatch.setattr(bmoca_device_replay.time, "sleep", lambda _seconds: None)

    serial = bmoca_device_replay._wait_for_emulator_ready(
        Environment(),
        adb_path=bmoca_device_replay.Path("/sdk/adb"),
        timeout_seconds=5,
    )

    assert serial == "emulator-5554"


def test_bmoca_snapshot_gate_waits_before_restarting_task_manager(monkeypatch) -> None:
    events: list[str] = []

    class TaskManager:
        def stop(self):
            events.append("stop")

        def start(self, **_kwargs):
            events.append("start")

    class Simulator:
        _adb_port = 5555

        def load_state(self, _request):
            events.append("load")
            return "loaded"

        def create_log_stream(self):
            events.append("log_stream")
            return object()

    class Coordinator:
        _task_manager = TaskManager()
        _simulator = Simulator()

        def _create_adb_call_parser(self):
            return None

    class Environment:
        _coordinator = Coordinator()

    monkeypatch.setattr(
        bmoca_device_replay,
        "_wait_for_emulator_ready",
        lambda *_args, **_kwargs: events.append("ready"),
    )
    environment = Environment()
    bmoca_device_replay._install_snapshot_ready_gate(
        environment,
        adb_path=bmoca_device_replay.Path("/sdk/adb"),
    )

    assert environment._coordinator.load_snapshot("snapshot") == "loaded"
    assert events == ["stop", "load", "ready", "log_stream", "start"]


def test_bmoca_native_e2e_uses_planner_mode_and_preserves_accounting(
    monkeypatch,
) -> None:
    episode = bmoca_device_replay._Episode(
        task_id="chrome/open_a_new_tab_in_Chrome",
        task_path=bmoca_device_replay.Path("/bmoca/task.csv"),
        instruction="Open a new tab in Chrome.",
        max_steps=5,
        env_id="100",
        snapshot_id="test_env_100",
        avd_name="pixel_3_test_00",
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(bmoca_device_replay, "_configure_runtime", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bmoca_device_replay, "_episodes", lambda *_args, **_kwargs: (episode,))

    def evaluate(_episode, **kwargs):
        captured.update(kwargs)
        return bmoca_device_replay._episode_result(
            episode,
            official_success=True,
            classification="success",
            error=None,
            duration=1.0,
            actions_executed=2,
            model_calls=3,
            planner_steps=3,
            function_invoked=True,
            function_actions_executed=2,
            llm_usage={
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            },
        )

    monkeypatch.setattr(bmoca_device_replay, "_evaluate_episode", evaluate)

    report = bmoca_device_replay.evaluate_device_omniflow_e2e(
        bmoca_root="/bmoca",
        store_path="/store/store.json",
        task_id=episode.task_id,
        planner_model="qwen3-vl-plus",
        environment_ids=("100",),
        android_sdk_root="/sdk",
        android_avd_home="/avd",
        avd_template_home="/templates",
    )

    assert captured["planner_model"] == "qwen3-vl-plus"
    assert report["configuration"]["function_replay"] == "native_omniflow_e2e"
    assert report["summary"]["official_success_count"] == 1
    assert report["summary"]["model_calls"] == 3
    assert report["summary"]["total_tokens"] == 120
    assert report["summary"]["function_invocation_rate"] == 1.0
    assert report["summary"]["function_action_reuse_rate"] == 1.0
