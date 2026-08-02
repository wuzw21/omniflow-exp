from __future__ import annotations

from dataclasses import dataclass
import io
import json
from pathlib import Path
import socket
import threading
from types import SimpleNamespace

from PIL import Image

from src.experiment import androidworld as pipeline
from src.integrations.android_world.mobilegpt_agent import (
    _socket_timeout,
    build_mobilegpt_agent,
)
from src.integrations.mobilegpt_runtime import install_mobilegpt_androidworld_observe


@dataclass
class _Bounds:
    x_min: int
    y_min: int
    x_max: int
    y_max: int


def _read_payload(stream: object, prefix: bytes) -> bytes:
    header = stream.readline()
    assert header.startswith(prefix)
    size = int(header[len(prefix) :].strip())
    return stream.read(size)


def test_mobilegpt_negative_timeout_means_unbounded_socket_wait() -> None:
    assert _socket_timeout(-1.0) is None
    assert _socket_timeout(0.0) == 0.1
    assert _socket_timeout(3.5) == 3.5


def test_mobilegpt_executes_server_click_through_androidworld_state_and_action(
    tmp_path: Path,
    monkeypatch,
) -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(5.0)
    port = listener.getsockname()[1]
    transcript: dict[str, object] = {}
    server_errors: list[BaseException] = []

    def serve() -> None:
        try:
            connection, _ = listener.accept()
            with connection, connection.makefile("rwb", buffering=0) as stream:
                transcript["packages"] = stream.readline().decode().strip()
                transcript["instruction"] = stream.readline().decode().strip()
                stream.write(b"##$$##com.example.target\r\n")
                transcript["first_screenshot"] = _read_payload(stream, b"S")
                transcript["first_xml"] = _read_payload(stream, b"X")
                stream.write(b"##$$##com.example.other\r\n")
                transcript["second_screenshot"] = _read_payload(stream, b"S")
                transcript["second_xml"] = _read_payload(stream, b"X")
                stream.write(
                    json.dumps(
                        {"name": "click", "parameters": {"index": 0}}
                    ).encode()
                    + b"\r\n"
                )
                transcript["third_screenshot"] = _read_payload(stream, b"S")
                transcript["third_xml"] = _read_payload(stream, b"X")
                stream.write(b"$$$$$\r\n")
        except BaseException as error:
            server_errors.append(error)
        finally:
            listener.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    monkeypatch.setenv("MOBILEGPT_SERVER_HOST", "127.0.0.1")
    monkeypatch.setenv("MOBILEGPT_SERVER_PORT", str(port))
    monkeypatch.setenv("MOBILEGPT_TARGET_PACKAGE", "com.example.target")

    class FakeEnv:
        logical_screen_size = (100, 200)
        foreground_activity_name = "com.example.target/.MainActivity"

        def __init__(self) -> None:
            self.actions: list[SimpleNamespace] = []
            self.current_package = "com.example.target"

        def get_state(self) -> SimpleNamespace:
            element = SimpleNamespace(
                bbox_pixels=_Bounds(10, 20, 30, 60),
                package_name=self.current_package,
                class_name="android.widget.Button",
                text="Record",
                content_description="",
                resource_name=f"{self.current_package}:id/record",
                is_clickable=True,
                is_editable=False,
                is_scrollable=False,
            )
            return SimpleNamespace(
                pixels=Image.new("RGB", (100, 200), color="blue"),
                forest=None,
                ui_elements=[element],
                auxiliaries={
                    "package_name": self.current_package,
                    "activity_name": f"{self.current_package}/.MainActivity",
                },
            )

        def execute_action(self, action: SimpleNamespace) -> None:
            self.actions.append(action)
            if action.action_type == "open_app":
                self.current_package = action.app_name

        def reset(self, go_home: bool = False) -> None:
            del go_home

    env = FakeEnv()
    agent = build_mobilegpt_agent(
        env=env,
        evidence_root=tmp_path,
        action_factory=lambda **payload: SimpleNamespace(**payload),
    )

    result = agent.step("Record and save audio")
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert server_errors == []
    assert transcript["packages"] == "Lcom.example.target"
    assert transcript["instruction"] == "IRecord and save audio"
    for key in ("first_screenshot", "second_screenshot", "third_screenshot"):
        image = Image.open(io.BytesIO(transcript[key]))
        assert image.format == "JPEG"
        assert image.size == (100, 200)
    first_xml = transcript["first_xml"].decode()
    assert 'text="Record"' in first_xml
    assert 'index="0"' in first_xml
    assert [action.action_type for action in env.actions] == [
        "open_app",
        "open_app",
        "click",
    ]
    assert env.actions[0].app_name == "com.example.target"
    assert env.actions[1].app_name == "com.example.other"
    assert (env.actions[2].x, env.actions[2].y) == (20, 40)
    assert result.done is True
    assert result.data["actions_executed"] == 1
    assert result.data["state_backend"] == "androidworld"
    assert result.data["action_backend"] == "androidworld"


def test_mobilegpt_native_speak_reaches_androidworld_answer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(5.0)
    port = listener.getsockname()[1]
    server_errors: list[BaseException] = []

    def serve() -> None:
        try:
            connection, _ = listener.accept()
            with connection, connection.makefile("rwb", buffering=0) as stream:
                stream.readline()
                stream.readline()
                stream.write(b"##$$##com.example.target\r\n")
                _read_payload(stream, b"S")
                _read_payload(stream, b"X")
                stream.write(
                    json.dumps(
                        {"name": "speak", "parameters": {"message": "20"}}
                    ).encode()
                    + b"\r\n"
                )
                stream.write(b"$$$$$\r\n")
        except BaseException as error:
            server_errors.append(error)
        finally:
            listener.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    monkeypatch.setenv("MOBILEGPT_SERVER_HOST", "127.0.0.1")
    monkeypatch.setenv("MOBILEGPT_SERVER_PORT", str(port))
    monkeypatch.setenv("MOBILEGPT_TARGET_PACKAGE", "com.example.target")
    monkeypatch.setenv("MOBILEGPT_POST_ACTION_WAIT_SEC", "0")

    class FakeEnv:
        logical_screen_size = (100, 200)
        foreground_activity_name = "com.example.target/.MainActivity"

        def __init__(self) -> None:
            self.actions: list[SimpleNamespace] = []

        def get_state(self) -> SimpleNamespace:
            return SimpleNamespace(
                pixels=Image.new("RGB", (100, 200), color="blue"),
                forest=None,
                ui_elements=[
                    SimpleNamespace(
                        bbox_pixels=_Bounds(10, 20, 30, 60),
                        package_name="com.example.target",
                        class_name="android.widget.TextView",
                        text="Attendees: 20",
                        content_description="",
                        resource_name="com.example.target:id/body",
                        is_clickable=False,
                        is_editable=False,
                        is_scrollable=False,
                    )
                ],
                auxiliaries={
                    "package_name": "com.example.target",
                    "activity_name": self.foreground_activity_name,
                },
            )

        def execute_action(self, action: SimpleNamespace) -> None:
            self.actions.append(action)

        def reset(self, go_home: bool = False) -> None:
            del go_home

    env = FakeEnv()
    agent = build_mobilegpt_agent(
        env=env,
        evidence_root=tmp_path,
        action_factory=lambda **payload: SimpleNamespace(**payload),
    )

    result = agent.step("How many attendees were present?")
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert server_errors == []
    assert [action.action_type for action in env.actions] == ["open_app", "answer"]
    assert env.actions[-1].text == "20"
    assert result.done is True
    assert result.data["answer"] == "20"
    assert result.data["actions_executed"] == 0
    assert result.data["error"] is None


def test_mobilegpt_waits_for_real_app_ui_before_sending_first_observation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(5.0)
    port = listener.getsockname()[1]
    transcript: dict[str, object] = {}
    server_errors: list[BaseException] = []

    def serve() -> None:
        try:
            connection, _ = listener.accept()
            with connection, connection.makefile("rwb", buffering=0) as stream:
                stream.readline()
                stream.readline()
                stream.write(b"##$$##com.example.target\r\n")
                transcript["screenshot"] = _read_payload(stream, b"S")
                transcript["xml"] = _read_payload(stream, b"X")
                stream.write(b"$$$$$\r\n")
        except BaseException as error:
            server_errors.append(error)
        finally:
            listener.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    monkeypatch.setenv("MOBILEGPT_SERVER_HOST", "127.0.0.1")
    monkeypatch.setenv("MOBILEGPT_SERVER_PORT", str(port))
    monkeypatch.setenv("MOBILEGPT_TARGET_PACKAGE", "com.example.target")
    monkeypatch.setenv("MOBILEGPT_POST_ACTION_WAIT_SEC", "0")
    monkeypatch.setenv("MOBILEGPT_APP_READY_POLL_SEC", "0")

    class FakeEnv:
        logical_screen_size = (100, 200)
        foreground_activity_name = "com.example.target/.MainActivity"

        def __init__(self) -> None:
            self.actions: list[SimpleNamespace] = []
            self.state_reads = 0

        def get_state(self) -> SimpleNamespace:
            self.state_reads += 1
            if self.state_reads == 1:
                elements = [
                    SimpleNamespace(
                        bbox_pixels=_Bounds(0, 20, 100, 200),
                        package_name="com.example.target",
                        class_name="android.widget.FrameLayout",
                        text="",
                        content_description="",
                        resource_name="",
                        is_clickable=False,
                        is_editable=False,
                        is_scrollable=False,
                    ),
                    SimpleNamespace(
                        bbox_pixels=_Bounds(0, 0, 100, 20),
                        package_name="com.android.systemui",
                        class_name="android.widget.TextView",
                        text="12:00",
                        content_description="",
                        resource_name="com.android.systemui:id/clock",
                        is_clickable=False,
                        is_editable=False,
                        is_scrollable=False,
                    )
                ]
                color = "white"
            else:
                elements = [
                    SimpleNamespace(
                        bbox_pixels=_Bounds(10, 20, 30, 60),
                        package_name="com.example.target",
                        class_name="android.widget.Button",
                        text="New note",
                        content_description="",
                        resource_name="com.example.target:id/new_note",
                        is_clickable=True,
                        is_editable=False,
                        is_scrollable=False,
                    )
                ]
                color = "blue"
            return SimpleNamespace(
                pixels=Image.new("RGB", (100, 200), color=color),
                forest=None,
                ui_elements=elements,
                auxiliaries={
                    "package_name": "com.example.target",
                    "activity_name": self.foreground_activity_name,
                },
            )

        def execute_action(self, action: SimpleNamespace) -> None:
            self.actions.append(action)

        def reset(self, go_home: bool = False) -> None:
            del go_home

    env = FakeEnv()
    agent = build_mobilegpt_agent(
        env=env,
        evidence_root=tmp_path,
        action_factory=lambda **payload: SimpleNamespace(**payload),
    )

    result = agent.step("Create a note")
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert server_errors == []
    assert env.state_reads == 2
    assert transcript["xml"] is not None
    xml_text = transcript["xml"].decode()
    assert 'package="com.example.target"' in xml_text
    assert 'text="New note"' in xml_text
    image = Image.open(io.BytesIO(transcript["screenshot"]))
    assert image.getpixel((50, 100)) == (0, 0, 254)
    assert result.done is True
    assert result.data["error"] is None


def test_mobilegpt_server_trusts_only_androidworld_client_xml(
    tmp_path: Path,
    monkeypatch,
) -> None:
    stats_path = tmp_path / "stats.jsonl"
    monkeypatch.setenv("MOBILEGPT_RUNTIME_OBSERVE_BACKEND", "androidworld")
    monkeypatch.setenv("MOBILEGPT_STATS_JSONL", str(stats_path))

    class FakeServer:
        def __recv_xml(
            self,
            client_socket: object,
            screen_count: int,
            log_directory: str,
        ) -> str:
            del client_socket, screen_count, log_directory
            return (
                '<hierarchy><node id="forest:7" text="Record" '
                'bounds="[10,20][30,60]" clickable="true" /></hierarchy>'
            )

    install_mobilegpt_androidworld_observe(FakeServer)
    receive = getattr(FakeServer(), "_FakeServer__recv_xml")
    xml_text = receive(object(), 0, str(tmp_path / "log"))

    assert 'index="0"' in xml_text
    assert (tmp_path / "log" / "xmls" / "0.xml").read_text() == xml_text
    stats = [json.loads(line) for line in stats_path.read_text().splitlines()]
    assert stats[-1]["backend"] == "androidworld"
    assert "oob_xml_chars" not in stats[-1]


def test_mobilegpt_server_rejects_non_androidworld_observation_backend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MOBILEGPT_RUNTIME_OBSERVE_BACKEND", "client")
    monkeypatch.delenv("MOBILEGPT_STATS_JSONL", raising=False)

    class FakeServer:
        def __recv_xml(
            self,
            client_socket: object,
            screen_count: int,
            log_directory: str,
        ) -> str:
            del client_socket, screen_count, log_directory
            return '<hierarchy><node bounds="[0,0][1,1]" /></hierarchy>'

    install_mobilegpt_androidworld_observe(FakeServer)
    receive = getattr(FakeServer(), "_FakeServer__recv_xml")

    try:
        receive(object(), 0, str(tmp_path / "log"))
    except RuntimeError as error:
        assert str(error) == "mobilegpt_native_observe_backend_required:client"
    else:
        raise AssertionError("non-AndroidWorld MobileGPT backend was accepted")


def test_mobilegpt_episode_command_declares_native_androidworld_io(
    tmp_path: Path,
) -> None:
    source_run_log = tmp_path / "source.run_log.json"
    source_run_log.write_text("{}", encoding="utf-8")
    item = pipeline.ArchivedRunLog(
        task="AudioRecorderRecordAudio",
        goal="Record and save audio",
        params={},
        source_run_log=source_run_log,
        replay_seed=111,
        step_count=1,
        meta={},
    )

    spec = pipeline.build_mobilegpt_androidworld_command(
        item,
        method_name="mobilegpt_offline_retrieval",
        target=pipeline.DeviceTarget("fold5564", "emulator-5564", 5564),
        android_world_root=tmp_path / "androidworld",
        output_root=tmp_path / "results",
        stats_jsonl=tmp_path / "stats.jsonl",
        server_host="0.0.0.0",
        server_port=12345,
        target_package="com.dimowner.audiorecorder",
        max_steps=20,
        task_random_seed=113,
        fixed_task_seed=True,
        fixed_task_params=True,
        task_params_override={},
        perform_emulator_setup=False,
        adb_path="adb",
        start_timeout_sec=60.0,
        finish_timeout_sec=120.0,
        timeout_sec=600.0,
    )

    assert spec.env["MOBILEGPT_SERVER_HOST"] == "127.0.0.1"
    assert spec.env["MOBILEGPT_SERVER_PORT"] == "12345"
    assert spec.env["MOBILEGPT_TARGET_PACKAGE"] == "com.dimowner.audiorecorder"
    assert spec.env["MOBILEGPT_APP_READY_TIMEOUT_SEC"] == "15.0"
    assert spec.env["MOBILEGPT_APP_READY_POLL_SEC"] == "0.25"
    assert "MOBILEGPT_OOB_SERIAL_FILE" not in spec.env
    assert spec.metadata["state_backend"] == "androidworld"
    assert spec.metadata["action_backend"] == "androidworld"
    assert spec.metadata["native_androidworld_agent_io"] is True
    assert spec.metadata["mobilegpt_app_ready_timeout_sec"] == 15.0
    assert spec.metadata["mobilegpt_app_ready_poll_sec"] == 0.25
    assert spec.timeout_sec == 600.0
