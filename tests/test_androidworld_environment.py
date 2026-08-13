from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

from src.integrations.android_world.environment import (
    AndroidWorldEnvironmentConfig,
    AndroidWorldExperimentEnvironment,
)


def _state(forest: str) -> SimpleNamespace:
    return SimpleNamespace(
        pixels=Image.new("RGB", (4, 3), color="red"),
        forest=forest,
        ui_elements=[],
        auxiliaries={},
    )


def test_shared_environment_records_host_actions_and_restores_every_seam(
    tmp_path,
) -> None:
    states = iter([_state("before"), _state("after")])
    executed = []
    env = SimpleNamespace(
        get_state=lambda: next(states),
        execute_action=lambda action: executed.append(action),
    )
    original_get_state = env.get_state
    original_execute_action = env.execute_action
    raw_host = SimpleNamespace(
        _json_action=lambda action: SimpleNamespace(
            action_type=action["tool"],
            x=action["args"]["x"],
            y=action["args"]["y"],
        )
    )
    host = SimpleNamespace(host=raw_host)

    def act(action):
        return env.execute_action(raw_host._json_action(action))

    host.act = act
    original_host_act = host.act
    agent = SimpleNamespace(host=host)
    experiment_environment = AndroidWorldExperimentEnvironment(
        env,
        AndroidWorldEnvironmentConfig(evidence_root=tmp_path),
    )

    with experiment_environment.install_episode_recorder(agent) as session:
        assert session.recorder is not None
        assert env.get_state is not original_get_state
        assert env.execute_action is not original_execute_action
        assert host.act is not original_host_act
        session.start_episode()
        host.act({"tool": "click", "args": {"x": 1, "y": 2}})
        run_log = session.seal_run_log(
            task_name="Task",
            goal="Goal",
            task_parameters={},
            seed=113,
            validator_success=True,
            validator_reward=1.0,
        )

    assert run_log is not None
    assert len(run_log["steps"]) == 1
    assert len(executed) == 1
    assert env.get_state is original_get_state
    assert env.execute_action is original_execute_action
    assert host.act is original_host_act


def test_shared_environment_does_not_install_duplicate_recorders(tmp_path) -> None:
    env = SimpleNamespace(get_state=lambda: _state("state"), execute_action=lambda _: None)
    experiment_environment = AndroidWorldExperimentEnvironment(
        env,
        AndroidWorldEnvironmentConfig(evidence_root=tmp_path),
    )
    agent = SimpleNamespace()

    first = experiment_environment.install_episode_recorder(agent)
    second = experiment_environment.install_episode_recorder(agent)

    assert first is second
    first.close()
    third = experiment_environment.install_episode_recorder(agent)
    assert third is not first
    third.close()


def test_shared_environment_reports_unavailable_episode_methods(tmp_path) -> None:
    experiment_environment = AndroidWorldExperimentEnvironment(
        SimpleNamespace(),
        AndroidWorldEnvironmentConfig(evidence_root=tmp_path),
    )

    session = experiment_environment.install_episode_recorder(SimpleNamespace())

    assert session.recorder is None
    assert session.error == "environment_episode_methods_unavailable"
    session.close()


def test_shared_environment_restores_partial_install_failure(tmp_path) -> None:
    class RejectExecuteActionReplacement:
        def __init__(self) -> None:
            object.__setattr__(self, "get_state", lambda: _state("state"))
            object.__setattr__(self, "execute_action", lambda _: None)
            object.__setattr__(self, "reject_replacement", True)

        def __setattr__(self, name, value):
            if name == "execute_action" and self.reject_replacement:
                raise RuntimeError("replacement rejected")
            object.__setattr__(self, name, value)

    env = RejectExecuteActionReplacement()
    original_get_state = env.get_state
    original_execute_action = env.execute_action
    experiment_environment = AndroidWorldExperimentEnvironment(
        env,
        AndroidWorldEnvironmentConfig(evidence_root=tmp_path),
    )

    session = experiment_environment.install_episode_recorder(SimpleNamespace())

    assert session.recorder is None
    assert session.error == "episode_recorder_install_failed:replacement rejected"
    assert env.get_state is original_get_state
    assert env.execute_action is original_execute_action
    session.close()
