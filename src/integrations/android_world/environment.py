"""Shared AndroidWorld experiment environment and episode recording lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.experiment.observation_evidence import AndroidWorldEpisodeRecorder


@dataclass(frozen=True)
class AndroidWorldEnvironmentConfig:
    evidence_root: str | Path


class AndroidWorldExperimentEnvironment:
    """Expose the official environment through one accounting-only proxy."""

    def __init__(self, env: Any, config: AndroidWorldEnvironmentConfig) -> None:
        self.env = env
        self.config = config
        self._active_recording: EpisodeRecordingSession | None = None

    def install_episode_recorder(self) -> EpisodeRecordingSession:
        active = self._active_recording
        if active is not None and not active.closed:
            return active
        session = EpisodeRecordingSession(self)
        self._active_recording = session
        return session

    def _release(self, session: EpisodeRecordingSession) -> None:
        if self._active_recording is session:
            self._active_recording = None


class EpisodeRecordingSession:
    """Own one recorder without mutating the official environment."""

    def __init__(
        self,
        owner: AndroidWorldExperimentEnvironment,
    ) -> None:
        self._owner = owner
        self.recorder: AndroidWorldEpisodeRecorder | None = None
        self.error: str | None = None
        self.closed = False
        get_state = getattr(owner.env, "get_state", None)
        execute_action = getattr(owner.env, "execute_action", None)
        if not callable(get_state) or not callable(execute_action):
            self.error = "environment_episode_methods_unavailable"
            self.env = owner.env
            return
        try:
            self.recorder = AndroidWorldEpisodeRecorder(
                get_state,
                execute_action,
                evidence_root=owner.config.evidence_root,
            )
            self.env = _RecordingEnvironmentProxy(owner.env, self.recorder)
        except Exception as exc:  # noqa: BLE001
            self.error = f"episode_recorder_install_failed:{exc}"
            self.env = owner.env

    def __enter__(self) -> EpisodeRecordingSession:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()

    def start_episode(self) -> None:
        if self.recorder is not None:
            self.recorder.start_episode()

    def seal_run_log(self, **kwargs: Any) -> dict[str, Any] | None:
        if self.recorder is None:
            return None
        return self.recorder.seal_run_log(**kwargs)

    def persist_observations(self) -> list[dict[str, Any]] | None:
        if self.recorder is None:
            return None
        try:
            return self.recorder.persist_observations()
        except (OSError, TypeError, ValueError) as exc:
            self.error = str(exc)
            return None

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._owner._release(self)


class _RecordingEnvironmentProxy:
    """Delegate AndroidWorld behavior while observing its public calls."""

    def __init__(self, env: Any, recorder: AndroidWorldEpisodeRecorder) -> None:
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "_recorder", recorder)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._env, name, value)

    def get_state(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return self._recorder.get_state(*args, **kwargs)
        except RuntimeError as error:
            if not _is_stale_accessibility_state_error(error):
                raise
            controller = getattr(self._env, "controller", None)
            restart = getattr(controller, "restart_accessibility_forwarder", None)
            if not callable(restart):
                raise RuntimeError(
                    "androidworld_accessibility_restart_unavailable"
                ) from error
            restart()
            return self._recorder.get_state(*args, **kwargs)

    def execute_action(self, action: Any, *args: Any, **kwargs: Any) -> Any:
        return self._recorder.execute_action(action, *args, **kwargs)


def _is_stale_accessibility_state_error(error: RuntimeError) -> bool:
    message = str(error).casefold()
    return (
        "accessibility" in message
        or "a11y" in message
        or ("grpc" in message and "tree" in message)
    )
