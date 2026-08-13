"""Shared AndroidWorld experiment environment and episode recording lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

from src.experiment.observation_evidence import AndroidWorldEpisodeRecorder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AndroidWorldEnvironmentConfig:
    evidence_root: str | Path


class AndroidWorldExperimentEnvironment:
    """Own the shared official environment lifecycle for every method."""

    def __init__(self, env: Any, config: AndroidWorldEnvironmentConfig) -> None:
        self.env = env
        self.config = config
        self._active_recording: EpisodeRecordingSession | None = None

    def install_episode_recorder(self, agent: Any) -> EpisodeRecordingSession:
        active = self._active_recording
        if active is not None and not active.closed:
            return active
        session = EpisodeRecordingSession(self, agent)
        self._active_recording = session
        return session

    def _release(self, session: EpisodeRecordingSession) -> None:
        if self._active_recording is session:
            self._active_recording = None


class EpisodeRecordingSession:
    """Install, expose, and deterministically restore one episode recorder."""

    def __init__(
        self,
        owner: AndroidWorldExperimentEnvironment,
        agent: Any,
    ) -> None:
        self._owner = owner
        self._env = owner.env
        self._host_action_owner = getattr(agent, "host", None)
        self._original_get_state = getattr(self._env, "get_state", None)
        self._original_execute_action = getattr(self._env, "execute_action", None)
        self._original_host_act = getattr(self._host_action_owner, "act", None)
        self.recorder: AndroidWorldEpisodeRecorder | None = None
        self.error: str | None = None
        self.closed = False
        self._get_state_installed = False
        self._execute_action_installed = False
        self._host_act_installed = False
        self._install()

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
        self._restore_installed_methods()
        self._owner._release(self)

    def _install(self) -> None:
        if not callable(self._original_get_state) or not callable(
            self._original_execute_action
        ):
            self.error = "environment_episode_methods_unavailable"
            return
        try:
            recorder = AndroidWorldEpisodeRecorder(
                self._original_get_state,
                self._original_execute_action,
                evidence_root=self._owner.config.evidence_root,
            )
            self.recorder = recorder
            self._env.get_state = recorder.get_state
            self._get_state_installed = True
            self._env.execute_action = recorder.execute_action
            self._execute_action_installed = True
            raw_androidworld_host = getattr(
                self._host_action_owner,
                "host",
                self._host_action_owner,
            )
            project_host_action = getattr(raw_androidworld_host, "_json_action", None)
            if callable(self._original_host_act) and callable(project_host_action):

                def recorded_host_act(value: Any, **kwargs: Any) -> Any:
                    return recorder.execute_host_action(
                        value,
                        execute=lambda: self._original_host_act(value, **kwargs),
                        project=project_host_action,
                    )

                self._host_action_owner.act = recorded_host_act
                self._host_act_installed = True
        except Exception as exc:  # noqa: BLE001
            self.error = f"episode_recorder_install_failed:{exc}"
            self.recorder = None
            self._restore_installed_methods()

    def _restore_installed_methods(self) -> None:
        self._restore(
            owner=self._env,
            attribute="get_state",
            original=self._original_get_state,
            installed=self._get_state_installed,
        )
        self._get_state_installed = False
        self._restore(
            owner=self._env,
            attribute="execute_action",
            original=self._original_execute_action,
            installed=self._execute_action_installed,
        )
        self._execute_action_installed = False
        self._restore(
            owner=self._host_action_owner,
            attribute="act",
            original=self._original_host_act,
            installed=self._host_act_installed,
        )
        self._host_act_installed = False

    @staticmethod
    def _restore(
        *,
        owner: Any,
        attribute: str,
        original: Any,
        installed: bool,
    ) -> None:
        if not installed:
            return
        try:
            setattr(owner, attribute, original)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to restore AndroidWorld %s: %s",
                attribute,
                exc,
            )
