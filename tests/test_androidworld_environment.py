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


def test_shared_environment_records_calls_without_mutating_official_environment(
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
    experiment_environment = AndroidWorldExperimentEnvironment(
        env,
        AndroidWorldEnvironmentConfig(evidence_root=tmp_path),
    )

    with experiment_environment.install_episode_recorder() as session:
        assert session.recorder is not None
        assert env.get_state is original_get_state
        assert env.execute_action is original_execute_action
        session.start_episode()
        session.env.get_state()
        session.env.execute_action(
            SimpleNamespace(action_type="click", x=1, y=2)
        )
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


def test_shared_environment_does_not_install_duplicate_recorders(tmp_path) -> None:
    env = SimpleNamespace(get_state=lambda: _state("state"), execute_action=lambda _: None)
    experiment_environment = AndroidWorldExperimentEnvironment(
        env,
        AndroidWorldEnvironmentConfig(evidence_root=tmp_path),
    )
    first = experiment_environment.install_episode_recorder()
    second = experiment_environment.install_episode_recorder()

    assert first is second
    first.close()
    third = experiment_environment.install_episode_recorder()
    assert third is not first
    third.close()


def test_shared_environment_reports_unavailable_episode_methods(tmp_path) -> None:
    experiment_environment = AndroidWorldExperimentEnvironment(
        SimpleNamespace(),
        AndroidWorldEnvironmentConfig(evidence_root=tmp_path),
    )

    session = experiment_environment.install_episode_recorder()

    assert session.recorder is None
    assert session.error == "environment_episode_methods_unavailable"
    session.close()


def test_shared_environment_proxy_delegates_other_attributes(tmp_path) -> None:
    env = SimpleNamespace(
        get_state=lambda: _state("state"),
        execute_action=lambda _: None,
        logical_screen_size=(720, 1280),
    )
    experiment_environment = AndroidWorldExperimentEnvironment(
        env,
        AndroidWorldEnvironmentConfig(evidence_root=tmp_path),
    )

    session = experiment_environment.install_episode_recorder()

    assert session.env.logical_screen_size == (720, 1280)
    session.env.interaction_cache = "message"
    assert env.interaction_cache == "message"
    session.close()


def test_shared_environment_recovers_stale_accessibility_state_once(tmp_path) -> None:
    calls: list[str] = []
    states = iter(
        [
            RuntimeError("stale gRPC accessibility tree"),
            _state("recovered"),
        ]
    )

    def get_state():
        value = next(states)
        if isinstance(value, Exception):
            raise value
        return value

    controller = SimpleNamespace(
        restart_accessibility_forwarder=lambda: calls.append("restart"),
    )
    env = SimpleNamespace(
        controller=controller,
        get_state=get_state,
        execute_action=lambda _: None,
    )
    experiment_environment = AndroidWorldExperimentEnvironment(
        env,
        AndroidWorldEnvironmentConfig(evidence_root=tmp_path),
    )

    with experiment_environment.install_episode_recorder() as session:
        state = session.env.get_state()

    assert state.forest == "recovered"
    assert calls == ["restart"]


def test_shared_environment_does_not_recover_unrelated_state_errors(tmp_path) -> None:
    calls: list[str] = []
    controller = SimpleNamespace(
        restart_accessibility_forwarder=lambda: calls.append("restart"),
        refresh_env=lambda: calls.append("refresh"),
    )
    env = SimpleNamespace(
        controller=controller,
        get_state=lambda: (_ for _ in ()).throw(ValueError("bad state schema")),
        execute_action=lambda _: None,
    )
    experiment_environment = AndroidWorldExperimentEnvironment(
        env,
        AndroidWorldEnvironmentConfig(evidence_root=tmp_path),
    )

    with experiment_environment.install_episode_recorder() as session:
        try:
            session.env.get_state()
        except ValueError as error:
            assert str(error) == "bad state schema"
        else:
            raise AssertionError("unrelated state error must propagate")

    assert calls == []
