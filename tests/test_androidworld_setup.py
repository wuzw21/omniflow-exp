from __future__ import annotations

import sys
from types import SimpleNamespace

from src.integrations.android_world.run_episode import (
    _patch_androidworld_expense_setup_timeout,
)


def test_resource_setup_dismisses_legacy_dialog_before_vlc_onboarding(
    monkeypatch,
) -> None:
    class Controller:
        dismissed = False

        def __init__(self) -> None:
            self._env = SimpleNamespace(
                foreground_activity_name="org.videolan.vlc/.StartActivity",
                get_ui_elements=lambda: [
                    SimpleNamespace(
                        text=(
                            "SKIP"
                            if self.dismissed
                            else "This app was built for an older version of Android "
                            "and may not work properly."
                        ),
                        content_description=None,
                        package_name="org.videolan.vlc",
                    ),
                    SimpleNamespace(
                        text=None if self.dismissed else "OK",
                        content_description=None,
                        package_name="android",
                    ),
                ],
            )

        def click_element(self, label: str) -> None:
            assert label == "OK"
            self.dismissed = True

        def click_resource_id(
            self,
            resource_ids: str | tuple[str, ...],
            timeout_sec: float = 10.0,
        ) -> str:
            del timeout_sec
            if not self.dismissed:
                raise ValueError(f"Target resource ID not found: {resource_ids}.")
            return "clicked"

    fake_tools = SimpleNamespace(AndroidToolController=Controller)
    monkeypatch.setitem(sys.modules, "android_world.env.tools", fake_tools)

    patched = _patch_androidworld_expense_setup_timeout()
    assert patched is not None
    controller = Controller()
    try:
        assert (
            controller.click_resource_id("org.videolan.vlc:id/skip_button")
            == "clicked"
        )
        assert controller.dismissed is True
    finally:
        controller_type, original = patched
        controller_type.click_resource_id = original


def test_mobilegpt_speech_keeps_pending_action_on_original_observation(monkeypatch, tmp_path):
    import json
    from src.integrations import mobilegpt_oob_client as client

    # Official Server sends speech and a physical action for one X frame.
    # A second speech notification must not produce another observation either.
    frames = ["##$$##com.android.settings", *[
        json.dumps({"name": "speak", "parameters": {"message": "Opening settings"}})
        for _ in range(2)
    ], json.dumps({"name": "click", "parameters": {"index": 26}}), "$$$$$"]

    class Socket:
        def __init__(self):
            self.pending = bytearray(("\n".join(frames) + "\n").encode())
            self.sent = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def settimeout(self, value):
            pass

        def sendall(self, payload):
            self.sent.append(payload)

        def recv(self, size):
            result = bytes(self.pending[:size])
            del self.pending[:size]
            return result

    class Oob:
        def __init__(self):
            self.observations = 0
            self.actions = []

        def observe(self, **kwargs):
            self.observations += 1
            index = 26 if self.observations == 1 else 30
            return {"xml": f'<hierarchy><node id="{index}" bounds="[10,20][30,40]"/></hierarchy>',
                    "display": {"width": 100, "height": 100}}

        def act(self, action):
            self.actions.append(action)

    sock, oob = Socket(), Oob()
    monkeypatch.setattr(client.socket, "create_connection", lambda *a, **k: sock)
    monkeypatch.setattr(client, "OobControlClient", lambda *a, **k: oob)
    monkeypatch.setattr(client, "_prelaunch_target_package", lambda *a, **k: "com.android.settings")
    monkeypatch.setattr(client, "_wire_packages", lambda *a: ["com.android.settings"])
    result = client._run_mobilegpt_oob_transport(
        serial="regression-device", adb_path="adb", server_host="127.0.0.1",
        server_port=12345, instruction="Open settings", timeout_sec=10,
        max_steps=10, output_root=tmp_path)
    assert result["task_finished"] and result["actions"] == 1
    assert len(oob.actions) == 1 and oob.actions[0]["tool"] == "click"
    assert oob.observations == 2  # initial page, then the physical action result
    assert sum(payload.startswith(b"X") for payload in sock.sent) == 2


def test_source_based_methods_use_explicit_memory_before_task_selection(monkeypatch, tmp_path):
    import pytest
    from src.experiment import run_task as runner

    memory = tmp_path / "source.json"
    memory.write_text('{}')

    class SelectionReached(Exception):
        pass

    def select(args):
        assert args.source_run_log == str(memory)
        raise SelectionReached

    monkeypatch.setattr(runner, "_select_from_args", select)
    for method in ("fixed_replay", "t3a_hint"):
        args = runner.build_parser().parse_args([
            "result", "--task", "SystemBluetoothTurnOn", "--method", method,
            "--memory", str(memory)])
        with pytest.raises(SelectionReached):
            runner.run_task(args)
