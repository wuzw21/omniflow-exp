"""Canonical Host adapter for standalone GUI-agent experiments over OOB."""

from __future__ import annotations

from typing import Any

from omniflow.core.model import Action, ActionResult, Observation


class OobGuiAgentHost:
    """Adapt the resident OOB control client to OmniFlow's Host contract.

    AndroidWorld uses :class:`AndroidWorldHost` as its lifecycle owner.  This
    smaller adapter is for supplemental real-device agent checks where no
    AndroidWorld environment exists; it still keeps all physical observation
    and execution inside the existing OOB client.
    """

    def __init__(self, control_client: Any) -> None:
        if not callable(getattr(control_client, "observe", None)):
            raise TypeError("gui_agent_oob_observe_required")
        if not callable(getattr(control_client, "act", None)):
            raise TypeError("gui_agent_oob_act_required")
        self.control_client = control_client

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
