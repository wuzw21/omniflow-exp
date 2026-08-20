"""Record AndroidWorld episodes and persist immutable execution evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

from omniflow.core.trajectory import (
    OMNIFLOW_RUN_LOG_SCHEMA_VERSION,
    canonicalize_androidworld_action,
    canonicalize_run_log,
    observation_xml,
    state_id,
)
from omniflow.transfer.runtime import TRANSFER_STATE_CATALOG_VERSION
from src.experiment.performance_metrics import PerformanceMetrics
from src.integrations.android_world.state import snapshot_androidworld_state

_ANDROIDWORLD_ACTION_FIELDS = (
    "action_type",
    "index",
    "x",
    "y",
    "text",
    "direction",
    "app_name",
    "goal_status",
    "keycode",
    "clear_text",
)


class AndroidWorldEpisodeRecorder:
    """Harness-owned recorder around official AndroidWorld environment calls."""

    def __init__(
        self,
        get_state: Callable[..., Any],
        execute_action: Callable[..., Any],
        *,
        evidence_root: str | Path,
        performance_metrics: PerformanceMetrics | None = None,
    ):
        if not callable(get_state):
            raise TypeError("episode_recorder_get_state_callable_required")
        if not callable(execute_action):
            raise TypeError("episode_recorder_execute_action_callable_required")
        self._get_state = get_state
        self._execute_action = execute_action
        self._evidence_root = Path(evidence_root).expanduser().resolve()
        self._active = False
        self._latest_observation: dict[str, Any] | None = None
        self._observations: list[dict[str, Any]] = []
        self._steps: list[dict[str, Any]] = []
        self._recording_action = False
        self.performance_metrics = performance_metrics

    @property
    def episode_started(self) -> bool:
        return self._active

    @property
    def action_count(self) -> int:
        return len(self._steps)

    def start_episode(self) -> None:
        if self._active:
            return
        self._active = True

    def get_state(self, *args: Any, **kwargs: Any) -> Any:
        if self.performance_metrics is None:
            state = self._get_state(*args, **kwargs)
            if self._active:
                state = self._capture_state_with_retry(state, args, kwargs)
            return state
        started_ns = time.perf_counter_ns()
        success = False
        try:
            with self.performance_metrics.timed("native_observe_get_state"):
                state = self._get_state(*args, **kwargs)
            if self._active:
                with self.performance_metrics.timed("native_observe_record"):
                    state = self._capture_state_with_retry(state, args, kwargs)
            success = True
            return state
        finally:
            if self._active:
                self.performance_metrics.record(
                    "native_observe",
                    (time.perf_counter_ns() - started_ns) / 1_000_000.0,
                    success=success,
                )

    def record_host_observation(self, state: Any) -> None:
        if self._active:
            self._capture_state(state)

    def _capture_state_with_retry(
        self,
        state: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Recover one transient empty accessibility snapshot.

        AndroidWorld can return a screenshot with an empty accessibility tree
        immediately after an action or app launch.  The run-log contract still
        requires XML, so request one fresh native state and return that state to
        the official agent.  Persistent missing XML remains a hard evidence
        failure; this does not fabricate UI data or replay an old observation.
        """
        candidate = state
        for attempt in range(4):
            try:
                self._capture_state(candidate)
                return candidate
            except ValueError as error:
                if str(error) != "androidworld_run_log_observation_xml_required":
                    raise
                if attempt >= 3:
                    raise
                # The official accessibility forwarder can lag one or two
                # frames behind an app transition.  Re-read the official
                # state instead of persisting an incomplete observation.
                time.sleep(0.25)
                candidate = self._get_state(*args, **kwargs)
        raise AssertionError("unreachable_androidworld_observation_retry")

    def execute_action(self, action: Any, *args: Any, **kwargs: Any) -> Any:
        if self.performance_metrics is None:
            if not self._active or self._recording_action:
                return self._execute_action(action, *args, **kwargs)
            return self._record_action(
                androidworld_json_action_dict(action),
                lambda: self._execute_action(action, *args, **kwargs),
            )
        started_ns = time.perf_counter_ns()
        success = False
        try:
            if not self._active or self._recording_action:
                result = self._execute_action(action, *args, **kwargs)
            else:
                result = self._record_action(
                    androidworld_json_action_dict(action),
                    lambda: self._execute_action(action, *args, **kwargs),
                )
            success = (
                result.get("success") is not False
                if isinstance(result, dict)
                else getattr(result, "success", True) is not False
            )
            return result
        finally:
            if self._active:
                self.performance_metrics.record(
                    "native_act",
                    (time.perf_counter_ns() - started_ns) / 1_000_000.0,
                    success=success,
                )

    def execute_host_action(
        self,
        action: Any,
        *,
        execute: Callable[[], Any],
        project: Callable[[Any], Any],
        after_observation: Callable[[], Any] | None = None,
    ) -> Any:
        if not self._active or self._recording_action:
            return execute()
        return self._record_action(
            androidworld_json_action_dict(project(action)),
            execute,
            after_observation=after_observation,
        )

    def _record_action(
        self,
        canonical_action: dict[str, Any],
        execute: Callable[[], Any],
        after_observation: Callable[[], Any] | None = None,
    ) -> Any:
        metrics = self.performance_metrics
        before = self._latest_observation
        if before is None:
            if metrics is None:
                before = self._capture_state(self._get_state())
            else:
                with metrics.timed("native_act_before_observe"):
                    before = self._capture_state(self._get_state())
        step: dict[str, Any] = {
            "step_index": len(self._steps),
            "observation": _json_copy(before),
            "action": canonical_action,
            "result": {"success": False},
        }
        self._steps.append(step)
        self._recording_action = True
        try:
            if metrics is None:
                result = execute()
            else:
                with metrics.timed("native_act_dispatch"):
                    result = execute()
        except Exception as error:
            step["result"]["error"] = str(error) or type(error).__name__
            raise
        finally:
            self._recording_action = False
        explicit_success = (
            result.get("success")
            if isinstance(result, dict)
            else getattr(result, "success", None)
        )
        step["result"] = {"success": explicit_success is not False}
        explicit_error = (
            result.get("error")
            if isinstance(result, dict)
            else getattr(result, "error", None)
        )
        if explicit_success is False and str(explicit_error or "").strip():
            step["result"]["error"] = str(explicit_error)
        try:
            if metrics is None:
                next_observation = self._capture_state(
                    after_observation()
                    if after_observation is not None
                    else self._get_state()
                )
            else:
                with metrics.timed("native_act_after_observe"):
                    next_observation = self._capture_state(
                        after_observation()
                        if after_observation is not None
                        else self._get_state()
                    )
        except Exception as error:
            step["metadata"] = {
                "next_observation_error": str(error) or type(error).__name__
            }
        else:
            step["next_observation"] = _json_copy(next_observation)
        return result

    def seal_run_log(
        self,
        *,
        task_name: str,
        goal: str,
        task_parameters: dict[str, Any],
        seed: int | None,
        validator_success: bool,
        validator_reward: float,
        validator_official: bool = True,
        diagnostics: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not self._active:
            return None
        return build_androidworld_run_log(
            run_id=self._evidence_root.name,
            task_name=task_name,
            goal=goal,
            task_parameters=task_parameters,
            seed=seed,
            validator_success=validator_success,
            validator_reward=validator_reward,
            validator_official=validator_official,
            provenance={"kind": "runtime"},
            steps=self._steps,
            final_observation=self._latest_observation,
            diagnostics=diagnostics,
        )

    def persist_observations(self) -> list[dict[str, Any]]:
        # RunLog observations already contain the screenshot path and XML.
        # Return the compact summary for result reporting without writing a
        # second observation index beside run_log.json.
        return [
            _observation_index_record(
                item,
                observation_index=index,
                evidence_root=self._evidence_root,
            )
            for index, item in enumerate(self._observations)
        ]

    def _capture_state(self, state: Any) -> dict[str, Any]:
        observation = snapshot_androidworld_state(
            state,
            evidence_root=self._evidence_root,
        )
        observation = canonicalize_run_log_observation(observation)
        self._latest_observation = observation
        self._observations.append(_json_copy(observation))
        return observation


def androidworld_json_action_dict(value: Any) -> dict[str, Any]:
    """Project one official ``JSONAction`` to the shared serializable contract."""
    if isinstance(value, dict):
        raw = dict(value)
    else:
        raw = {
            field: getattr(value, field)
            for field in _ANDROIDWORLD_ACTION_FIELDS
            if hasattr(value, field)
        }
    action = {
        key: _enum_value(item) if key in {"action_type", "direction"} else item
        for key, item in raw.items()
        if key in _ANDROIDWORLD_ACTION_FIELDS and item is not None
    }
    return canonicalize_androidworld_action(action)


def canonicalize_run_log_observation(value: dict[str, Any]) -> dict[str, Any]:
    """Return the compact persisted observation: screenshot reference plus XML.

    The four-field AndroidWorld snapshot remains accepted as a read/import
    compatibility format, but is never emitted by the recorder.
    """
    if set(value) == {"screenshot", "xml"}:
        if not isinstance(value.get("xml"), str) or not value["xml"].strip():
            raise ValueError("androidworld_run_log_observation_xml_required")
        observation = _json_copy(value)
        if isinstance(observation.get("screenshot"), dict):
            observation["screenshot"].pop("sha256", None)
        return observation
    if set(value) != {"pixels", "forest", "ui_elements", "auxiliaries"}:
        raise ValueError("androidworld_run_log_observation_fields_invalid")
    xml = observation_xml(value)
    if not xml and value.get("ui_elements"):
        # Some valid historical native AndroidWorld states have a populated
        # element list but no forest.  Preserve the same UI information as XML
        # instead of retaining the much larger compatibility representation.
        from src.integrations.android_world.host import androidworld_elements_xml

        xml = androidworld_elements_xml(list(value["ui_elements"])).strip()
    if not xml:
        raise ValueError("androidworld_run_log_observation_xml_required")
    screenshot = _json_copy(value.get("pixels"))
    if isinstance(screenshot, dict):
        screenshot.pop("sha256", None)
    return {"screenshot": screenshot, "xml": xml}


def build_androidworld_run_log(
    *,
    run_id: str,
    task_name: str,
    goal: str,
    task_parameters: dict[str, Any],
    seed: int | None,
    validator_success: bool,
    validator_reward: float,
    validator_official: bool,
    provenance: dict[str, Any],
    steps: list[dict[str, Any]],
    final_observation: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the one RunLog representation used by every collector."""

    payload: dict[str, Any] = {
        "schema_version": OMNIFLOW_RUN_LOG_SCHEMA_VERSION,
        "run_id": str(run_id or "").strip(),
        "task_name": str(task_name or "").strip(),
        "goal": str(goal or ""),
        "task_parameters": _json_copy(task_parameters),
        "seed": seed,
        "status": "succeeded" if validator_success else "failed",
        "success": bool(validator_success),
        "validator": {
            "official": bool(validator_official),
            "success": bool(validator_success),
            "reward": max(0.0, float(validator_reward)),
        },
        "provenance": _json_copy(provenance),
        "steps": _json_copy(steps),
    }
    if isinstance(final_observation, dict):
        payload["final_observation"] = _json_copy(final_observation)
    if diagnostics:
        payload["diagnostics"] = _json_copy(diagnostics)
    return canonicalize_run_log(payload)


def persist_androidworld_run_log(
    output_dir: str | Path,
    *,
    run_log: dict[str, Any],
    replace: bool = False,
) -> Path:
    """Write the sole run_log.json artifact for one attempt."""

    root = Path(output_dir).expanduser().resolve()
    canonical_run = canonicalize_run_log(run_log)
    run_log_path = root / "run_log.json"
    content = _stable_json_bytes(canonical_run)
    if not replace:
        _write_immutable(run_log_path, content)
        return run_log_path
    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = run_log_path.with_name(".run_log.json.tmp")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, run_log_path)
    finally:
        temporary.unlink(missing_ok=True)
    return run_log_path


def persist_target_run_evidence(
    output_dir: str | Path,
    *,
    run_log: dict[str, Any],
    captured_transfer_states: dict[str, dict[str, Any]] | None = None,
    transfer_state_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a canonical target RunLog and optional OmniFlow transfer states."""
    root = Path(output_dir).expanduser().resolve()
    canonical_run = canonicalize_run_log(run_log)
    run_log_path = persist_androidworld_run_log(root, run_log=canonical_run)
    evidence = {
        "run_log_path": str(run_log_path),
    }
    if captured_transfer_states is None and transfer_state_audit is None:
        return evidence
    if not isinstance(captured_transfer_states, dict):
        raise ValueError("target_transfer_states_required")
    states = {
        str(state_identifier): dict(state)
        for state_identifier, state in sorted(captured_transfer_states.items())
    }
    for state_identifier, state in states.items():
        if str(state.get("state_id") or "").strip() != state_identifier:
            raise ValueError(f"target_transfer_state_key_mismatch:{state_identifier}")
    expected_audit = transfer_state_coverage_audit(canonical_run, states)
    if transfer_state_audit is not None and transfer_state_audit != expected_audit:
        raise ValueError("target_transfer_state_audit_mismatch")
    transfer_states_path = root / "target.transfer_states.json"
    transfer_states_bytes = _stable_json_bytes(
        {
            "schema_version": TRANSFER_STATE_CATALOG_VERSION,
            "run_id": str(canonical_run["run_id"]),
            "states": states,
        }
    )
    _write_immutable(transfer_states_path, transfer_states_bytes)
    evidence.update(
        {
            "target_transfer_states_path": str(transfer_states_path),
            "target_transfer_states_sha256": hashlib.sha256(
                transfer_states_bytes
            ).hexdigest(),
            "target_transfer_state_audit": expected_audit,
        }
    )
    return evidence


def transfer_state_coverage_audit(
    run_log: dict[str, Any],
    captured_transfer_states: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    canonical_run = canonicalize_run_log(run_log)
    referenced_state_ids = sorted(
        {
            state_id(observation)
            for step in canonical_run["steps"]
            for observation in (step["observation"], step.get("next_observation"))
            if isinstance(observation, dict)
        }
    )
    captured_state_ids = sorted(captured_transfer_states)
    missing_state_ids = sorted(set(referenced_state_ids) - set(captured_state_ids))
    return {
        "referenced_state_ids": referenced_state_ids,
        "captured_state_ids": captured_state_ids,
        "missing_state_ids": missing_state_ids,
        "referenced_state_count": len(referenced_state_ids),
        "captured_state_count": len(captured_state_ids),
        "missing_state_count": len(missing_state_ids),
        "complete": not missing_state_ids,
    }


def _observation_index_record(
    observation: dict[str, Any],
    *,
    observation_index: int,
    evidence_root: Path,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "observation_index": int(observation_index),
        "state_id": state_id(observation),
    }
    auxiliaries = observation.get("auxiliaries")
    if isinstance(auxiliaries, dict):
        for key in ("package_name", "activity_name"):
            text = str(auxiliaries.get(key) or "").strip()
            if text:
                record[key] = text
    pixels = observation.get("screenshot")
    if pixels is None:
        pixels = observation.get("pixels")
    if isinstance(pixels, dict):
        record.update(
            {
                "display": {
                    "width": int(pixels["width"]),
                    "height": int(pixels["height"]),
                },
                "path": Path(pixels["path"]).relative_to(evidence_root).as_posix(),
            }
        )
    return record


def _enum_value(value: Any) -> Any:
    enum_value = getattr(value, "value", value)
    if isinstance(enum_value, str):
        return enum_value.lower()
    return enum_value


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _stable_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise ValueError(f"observation_evidence_content_conflict:{path}")


__all__ = [
    "AndroidWorldEpisodeRecorder",
    "androidworld_json_action_dict",
    "build_androidworld_run_log",
    "canonicalize_run_log_observation",
    "persist_androidworld_run_log",
    "persist_target_run_evidence",
    "transfer_state_coverage_audit",
]
