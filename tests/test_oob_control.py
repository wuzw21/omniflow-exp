from __future__ import annotations

import base64
import io
import json
import sys
from types import ModuleType, SimpleNamespace

from PIL import Image

from omniflow import Action
from src.integrations.android_world import host as host_module
from src.integrations.android_world.oob_control import (
    CONTROL_ACTION,
    OBSERVE_ACTION,
    OBSERVE_RESULT_PATH,
    OobControlClient,
    oob_state_from_payload,
)


def test_oob_control_uses_one_request_and_canonical_action() -> None:
    commands: list[list[str]] = []
    response: dict[str, object] = {}

    def run(command, **kwargs):
        commands.append(command)
        if "rm" in command and "cat" not in command:
            return _completed()
        if "broadcast" in command:
            request_id = command[command.index("requestId") + 1]
            response.update(
                request_id=request_id,
                success=True,
                result={"success": True, "extra": {"message": "ok"}},
            )
            return _completed()
        return _completed(stdout=json.dumps(response))

    client = OobControlClient(
        SimpleNamespace(),
        adb_serial="emulator-5564",
        run=run,
    )
    result = client.act(Action("click", {"x": 500, "y": 250}).to_dict())

    assert result["success"] is True
    assert any(CONTROL_ACTION in command for command in commands)
    broadcast = next(command for command in commands if "broadcast" in command)
    encoded = broadcast[broadcast.index("requestBase64") + 1]
    request = json.loads(base64.b64decode(encoded).decode("utf-8"))
    assert request == {
        "action": {"tool": "click", "args": {"x": 500, "y": 250}},
        "await_stabilization": False,
    }


def test_oob_observe_uses_the_resident_observe_receiver() -> None:
    commands: list[list[str]] = []

    def run(command, **kwargs):
        commands.append(command)
        if "broadcast" in command:
            return _completed()
        if command[-2:] == ["cat", OBSERVE_RESULT_PATH]:
            return _completed(
                stdout=json.dumps(
                    {
                        "schema_version": "oob.observe.v1",
                        "success": True,
                        "state": {"xml": "<hierarchy />"},
                    }
                )
            )
        return _completed()

    client = OobControlClient(SimpleNamespace(), adb_serial="emulator-5564", run=run)
    result = client.observe(wait_to_stabilize=True)

    assert result["xml"] == "<hierarchy />"
    broadcast = next(command for command in commands if "broadcast" in command)
    assert OBSERVE_ACTION in broadcast
    assert "DebugOmniFlowObserveReceiver" in broadcast[broadcast.index("-n") + 1]
    assert broadcast[broadcast.index("waitToStabilize") + 1] == "true"


def test_oob_xml_produces_androidworld_state_shape(monkeypatch) -> None:
    image_buffer = io.BytesIO()
    Image.new("RGB", (2, 3), color="red").save(image_buffer, format="PNG")
    xml = (
        '<hierarchy><node package="com.example" bounds="[0,0][2,3]">'
        '<node text="Save" class="android.widget.Button" '
        'package="com.example" bounds="[0,0][2,3]" clickable="true" />'
        "</node></hierarchy>"
    )
    representation_utils = ModuleType("android_world.env.representation_utils")
    representation_utils.xml_dump_to_ui_elements = lambda value: [
        {"text": "Save", "xml": value}
    ]
    monkeypatch.setitem(
        sys.modules,
        "android_world.env.representation_utils",
        representation_utils,
    )

    state = oob_state_from_payload(
        {
            "package_name": "com.example",
            "activity_name": "com.example/.MainActivity",
            "display": {"width": 2, "height": 3},
            "xml": xml,
            "image_base64": base64.b64encode(image_buffer.getvalue()).decode(),
        }
    )

    assert state.forest == xml
    assert state.ui_elements == [{"text": "Save", "xml": xml}]
    assert state.auxiliaries["package_name"] == "com.example"
    assert tuple(state.pixels.shape) == (3, 2, 3)


def test_oob_host_replaces_androidworld_observe_and_act(monkeypatch, tmp_path) -> None:
    xml = '<hierarchy><node package="com.example" bounds="[0,0][4,3]" /></hierarchy>'
    state = SimpleNamespace(
        pixels=Image.new("RGB", (4, 3), color="blue"),
        forest=xml,
        ui_elements=[],
        auxiliaries={
            "package_name": "com.example",
            "activity_name": "com.example/.MainActivity",
        },
    )

    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            self.actions = []

        def observe(self, *, wait_to_stabilize=False):
            assert wait_to_stabilize is True
            return {
                "package_name": "com.example",
                "activity_name": "com.example/.MainActivity",
                "display": {"width": 4, "height": 3},
                "xml": xml,
                "image_base64": "",
            }

        def act(self, action):
            self.actions.append(action)
            return {"success": True}

        def reset(self):
            return None

    monkeypatch.setattr(host_module, "OobControlClient", FakeClient)
    monkeypatch.setattr(host_module, "oob_state_from_payload", lambda *_args, **_kwargs: state)

    class Env:
        device_screen_size = (4, 3)

        def execute_action(self, _action):
            raise AssertionError("AndroidWorld action path was used")

    host = host_module.AndroidWorldHost(
        Env(),
        evidence_root=tmp_path,
        control_backend="oob",
    )
    observation = host.observe(xml=True, screenshot=False)
    result = host.act(Action("click", {"x": 500, "y": 500}))

    assert result.success is True
    assert host.observe_backend == "oob_control"
    assert host.act_backend == "oob_control"
    assert observation.extra["observe_backend"] == "oob_control"
    assert observation.extra["androidworld_state"]["xml"] == xml


def _completed(*, stdout: str = ""):
    return SimpleNamespace(returncode=0, stdout=stdout, stderr="")
