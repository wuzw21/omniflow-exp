"""Canonical Host adapter for standalone GUI-agent experiments over OOB."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omniflow.core.model import Action, ActionResult, Observation
from src.integrations.android_world.state import store_screenshot


class OobGuiAgentHost:
    """Adapt the resident OOB control client to OmniFlow's Host contract.

    AndroidWorld uses :class:`AndroidWorldHost` as its lifecycle owner.  This
    smaller adapter is for supplemental real-device agent checks where no
    AndroidWorld environment exists; it still keeps all physical observation
    and execution inside the existing OOB client.
    """

    def __init__(self, control_client: Any, *, evidence_root: str | Path | None = None,
                 source_states: dict[str, dict[str, Any]] | None = None) -> None:
        if not callable(getattr(control_client, "observe", None)):
            raise TypeError("gui_agent_oob_observe_required")
        if not callable(getattr(control_client, "act", None)):
            raise TypeError("gui_agent_oob_act_required")
        self.control_client = control_client
        self.evidence_root = Path(evidence_root) if evidence_root is not None else None
        self.source_states = source_states or {}
        self._step_index = 0

    def observe(
        self,
        *,
        xml: bool = True,
        screenshot: bool = True,
        app_info: bool = True,
    ) -> Observation:
        payload = self.control_client.observe(wait_to_stabilize=True)
        if not isinstance(payload, dict):
            raise TypeError("gui_agent_oob_observation_invalid")
        extra: dict[str, Any] = {"observe_backend": "oob_control"}
        for key in ("display", "state_id", "stabilization"):
            if payload.get(key) is not None:
                value = payload[key]
                extra[key] = dict(value) if isinstance(value, dict) else value
        if self.evidence_root is not None and payload.get("image_base64"):
            frame = store_screenshot(payload["image_base64"], evidence_root=self.evidence_root)
            extra["screenshot_path"] = frame["path"]
        return Observation(
            xml=(str(payload.get("xml") or "") or None) if xml else None,
            package_name=(str(payload.get("package_name") or "") or None)
            if app_info
            else None,
            activity_name=(str(payload.get("activity_name") or "") or None)
            if app_info
            else None,
            image_base64=(str(payload.get("image_base64") or "") or None)
            if screenshot
            else None,
            extra=extra,
        )

    def installed_apps(self) -> dict[str, str] | None:
        inventory = getattr(self.control_client, "installed_apps", None)
        return inventory() if callable(inventory) else None

    def get_state(self, state_id: str) -> Observation | None:
        value = self.source_states.get(state_id)
        return Observation.from_value(value) if value is not None else None

    def record_step(self, fact: dict[str, Any]) -> dict[str, Any]:
        step = {"step_index": self._step_index, **fact}
        if self.evidence_root is not None:
            self.evidence_root.mkdir(parents=True, exist_ok=True)
            with (self.evidence_root / "events.ndjson").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(step, ensure_ascii=False) + "\n")
        self._step_index += 1
        return {"step": step}

    def act(self, value: Action | dict[str, Any]) -> ActionResult:
        action = Action.from_value(value)
        if (
            action.tool == "input_text"
            and action.args.get("x") is not None
            and action.args.get("y") is not None
        ):
            focus_result = ActionResult.from_value(
                self.control_client.act(
                    {
                        "tool": "click",
                        "args": {"x": action.args["x"], "y": action.args["y"]},
                    }
                )
            )
            if not focus_result.success:
                return focus_result
            self.control_client.observe(wait_to_stabilize=True)
        return ActionResult.from_value(self.control_client.act(action.to_dict()))

    def reset(self) -> None:
        reset = getattr(self.control_client, "reset", None)
        if callable(reset):
            reset()


__all__ = ["OobGuiAgentHost"]
