from __future__ import annotations

from dataclasses import dataclass
import io
import json
from pathlib import Path
import socket
import sys
import threading
from types import ModuleType, SimpleNamespace
import xml.etree.ElementTree as ET

from PIL import Image

from src.experiment import androidworld as pipeline
from src.experiment.source_records import CanonicalRunLog
from src.integrations import mobilegpt_memory
from src.integrations.android_world.launch import (
    _mobilegpt_runtime_integrity_error,
    _mobilegpt_runtime_integrity_exit_code,
)
from src.integrations.android_world.mobilegpt_agent import (
    _socket_timeout,
    build_mobilegpt_agent,
)
from src.integrations.mobilegpt_runtime import (
    _parse_mobilegpt_model_response,
    install_mobilegpt_androidworld_observe,
    install_mobilegpt_memory_only_guard,
    install_mobilegpt_select_schema_repair,
    install_mobilegpt_upstream_accounting,
    mobilegpt_compatible_xml,
    mobilegpt_embedding_contract,
    preflight_mobilegpt_endpoints,
)


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


def test_mobilegpt_server_command_selects_strict_upstream_runtime(tmp_path: Path) -> None:
    spec = pipeline.build_mobilegpt_command(
        "server",
        mobilegpt_root=tmp_path / "mobilegpt",
        stats_jsonl=tmp_path / "stats.jsonl",
        repo_root=tmp_path,
    )

    assert spec.env["MOBILEGPT_UPSTREAM_MODE"] == "1"
    assert spec.argv[-1] == "--upstream"


def test_mobilegpt_upstream_accounting_preserves_original_query(monkeypatch) -> None:
    utils_package = ModuleType("utils")
    utils_module = ModuleType("utils.utils")
    utils_module.OpenAI = object
    utils_module.query = lambda messages: messages
    monkeypatch.setitem(sys.modules, "utils", utils_package)
    monkeypatch.setitem(sys.modules, "utils.utils", utils_module)

    original_query = utils_module.query
    install_mobilegpt_upstream_accounting()

    assert utils_module.query is original_query


def test_mobilegpt_upstream_accounting_routes_chat_and_embeddings_separately(
    monkeypatch,
) -> None:
    created_clients: list[tuple[str, str]] = []

    class NativeOpenAI:
        def __init__(self, *args, **kwargs) -> None:
            del args
            api_key = str(kwargs.get("api_key") or "")
            base_url = str(kwargs.get("base_url") or "")
            created_clients.append((api_key, base_url))
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_: "chat-response")
            )
            self.embeddings = SimpleNamespace(
                create=lambda **_: "embedding-response"
            )

    utils_package = ModuleType("utils")
    utils_module = ModuleType("utils.utils")
    utils_module.OpenAI = NativeOpenAI
    utils_module.query = lambda messages: messages
    monkeypatch.setitem(sys.modules, "utils", utils_package)
    monkeypatch.setitem(sys.modules, "utils.utils", utils_module)
    monkeypatch.setenv("MOBILEGPT_CHAT_API_KEY", "chat-key")
    monkeypatch.setenv("MOBILEGPT_CHAT_BASE_URL", "https://chat.example/v1")
    monkeypatch.setenv("MOBILEGPT_EMBEDDING_API_KEY", "embedding-key")
    monkeypatch.setenv(
        "MOBILEGPT_EMBEDDING_BASE_URL", "https://embedding.example/v1"
    )

    install_mobilegpt_upstream_accounting()
    client = utils_module.OpenAI(api_key="upstream-key")

    assert client.chat.completions.create(model="GLM-5.1") == "chat-response"
    assert client.embeddings.create(model="text-embedding-v4") == (
        "embedding-response"
    )
    assert created_clients == [
        ("chat-key", "https://chat.example/v1"),
        ("embedding-key", "https://embedding.example/v1"),
    ]


def test_mobilegpt_upstream_accounting_does_not_duplicate_native_events(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class NativeOpenAI:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_: SimpleNamespace(
                        usage=SimpleNamespace(
                            prompt_tokens=3,
                            completion_tokens=1,
                            total_tokens=4,
                        ),
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(content="{}")
                            )
                        ],
                    )
                )
            )
            self.embeddings = SimpleNamespace(
                create=lambda **_: SimpleNamespace(
                    usage=SimpleNamespace(prompt_tokens=2, total_tokens=2),
                    data=[SimpleNamespace(embedding=[0.1, 0.2])],
                )
            )

    utils_package = ModuleType("utils")
    utils_module = ModuleType("utils.utils")
    utils_module.OpenAI = NativeOpenAI
    utils_module.query = lambda messages: messages
    utils_module.write_omniflow_mobilegpt_event = lambda event: event
    monkeypatch.setitem(sys.modules, "utils", utils_package)
    monkeypatch.setitem(sys.modules, "utils.utils", utils_module)
    stats_path = tmp_path / "stats.jsonl"
    monkeypatch.setenv("MOBILEGPT_STATS_JSONL", str(stats_path))

    install_mobilegpt_upstream_accounting()
    client = utils_module.OpenAI()
    client.chat.completions.create(model="GLM-5.1")
    client.embeddings.create(model="text-embedding-v4")

    assert not stats_path.exists()


def test_mobilegpt_endpoint_preflight_uses_sealed_model_and_dimension(
    tmp_path: Path,
    monkeypatch,
) -> None:
    memory = tmp_path / "memory"
    app_memory = memory / "example"
    app_memory.mkdir(parents=True)
    (app_memory / "hierarchy.csv").write_text(
        'name,embedding\npage,"[0.1, 0.2, 0.3]"\n',
        encoding="utf-8",
    )
    manifest = tmp_path / "mobilegpt_memory_manifest.json"
    manifest.write_text(
        json.dumps(
            {"source_stats": {"embedding_models": ["text-embedding-v4"]}}
        ),
        encoding="utf-8",
    )
    clients: list[tuple[str, str]] = []

    class Client:
        def __init__(self, **kwargs) -> None:
            clients.append((kwargs["api_key"], kwargs["base_url"]))
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_: object())
            )
            self.embeddings = SimpleNamespace(
                create=lambda **_: SimpleNamespace(
                    data=[SimpleNamespace(embedding=[0.4, 0.5, 0.6])]
                )
            )

    monkeypatch.setenv("MOBILEGPT_CHAT_API_KEY", "chat-key")
    monkeypatch.setenv("MOBILEGPT_CHAT_BASE_URL", "https://chat.example/v1")
    monkeypatch.setenv("MOBILEGPT_EMBEDDING_API_KEY", "embedding-key")
    monkeypatch.setenv(
        "MOBILEGPT_EMBEDDING_BASE_URL", "https://embedding.example/v1"
    )

    assert mobilegpt_embedding_contract(manifest, memory) == (
        "text-embedding-v4",
        3,
    )
    assert preflight_mobilegpt_endpoints(
        manifest_path=manifest,
        memory_root=memory,
        chat_model="GLM-5.1",
        client_factory=Client,
    ) == ("text-embedding-v4", 3)
    assert clients == [
        ("chat-key", "https://chat.example/v1"),
        ("embedding-key", "https://embedding.example/v1"),
    ]


def test_mobilegpt_runtime_integrity_errors_exclude_method_terminals() -> None:
    assert _mobilegpt_runtime_integrity_error("TimeoutError: timed out") == (
        "TimeoutError: timed out"
    )
    assert _mobilegpt_runtime_integrity_error(
        "RuntimeError: mobilegpt_app_ui_not_ready:expected=com.example.target"
    )
    assert _mobilegpt_runtime_integrity_error(
        "mobilegpt_step_budget_exhausted:20"
    ) is None


def test_mobilegpt_runtime_integrity_summary_returns_nonzero() -> None:
    assert _mobilegpt_runtime_integrity_exit_code(
        {"runtime_integrity_error_count": 1}
    ) == 1
    assert _mobilegpt_runtime_integrity_exit_code(
        {"runtime_integrity_error_count": 0}
    ) == 0
    assert _mobilegpt_runtime_integrity_error(
        "ValueError: mobilegpt_action_unsupported:unknown"
    ) is None


def test_mobilegpt_select_preserves_native_semantic_subtask_names() -> None:
    def native_check(self, response, available_subtasks):
        del self
        action_name = response["action"]["name"]
        return any(
            subtask["name"] == action_name for subtask in available_subtasks
        )

    select_agent_class = type("SelectAgent", (), {})
    setattr(
        select_agent_class,
        "_SelectAgent__check_response_validity",
        native_check,
    )
    install_mobilegpt_select_schema_repair(select_agent_class)
    response = {
        "action": {
            "name": "type_text",
            "parameters": {"text_to_type": "hello"},
        }
    }

    accepted = select_agent_class()._SelectAgent__check_response_validity(
        response,
        [
            {
                "name": "type_text",
                "description": "Enter text into the current field.",
                "parameters": {"text_to_type": "Text to enter."},
            }
        ],
    )

    assert accepted is True
    assert response["action"]["name"] == "type_text"


def test_mobilegpt_select_does_not_synthesize_unknown_subtasks() -> None:
    def native_check(self, response, available_subtasks):
        del self
        action_name = response["action"]["name"]
        return any(
            subtask["name"] == action_name for subtask in available_subtasks
        )

    select_agent_class = type("SelectAgent", (), {})
    setattr(
        select_agent_class,
        "_SelectAgent__check_response_validity",
        native_check,
    )
    install_mobilegpt_select_schema_repair(select_agent_class)
    response = {
        "action": {
            "name": "invented_subtask",
            "parameters": {"value": "example"},
        }
    }
    available_subtasks = [
        {
            "name": "known_subtask",
            "description": "A native semantic subtask.",
            "parameters": {},
        }
    ]

    accepted = select_agent_class()._SelectAgent__check_response_validity(
        response,
        available_subtasks,
    )

    assert accepted is False
    assert "new_action" not in response
    assert [subtask["name"] for subtask in available_subtasks] == [
        "known_subtask"
    ]


def test_mobilegpt_memory_only_guard_stops_instead_of_calling_fallback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class ExploreAgent:
        def explore(self):
            return {"name": "explore"}

    class SelectAgent:
        def select(self):
            return {"name": "select"}

    class DeriveAgent:
        def derive(self):
            return {"name": "derive"}

    class MobileGPT:
        def __init__(self, stage: str) -> None:
            self.stage = stage
            self.instruction = "Use only recalled memory."
            self.current_page_index = 3
            self.current_subtask = {"name": "known_subtask"}
            self.finished = False

        def __finish_task(self):
            self.finished = True

        def get_next_action(self):
            if self.stage == "explore":
                return ExploreAgent().explore()
            if self.stage == "select":
                return SelectAgent().select()
            return DeriveAgent().derive()

    stats_path = tmp_path / "stats.jsonl"
    monkeypatch.setenv("MOBILEGPT_MEMORY_ONLY", "1")
    monkeypatch.setenv("MOBILEGPT_STATS_JSONL", str(stats_path))
    install_mobilegpt_memory_only_guard(
        MobileGPT,
        explore_agent_class=ExploreAgent,
        select_agent_class=SelectAgent,
        derive_agent_class=DeriveAgent,
    )

    for stage in ("explore", "select", "derive"):
        agent = MobileGPT(stage)
        assert agent.get_next_action() is None
        assert agent.finished is True

    events = [json.loads(line) for line in stats_path.read_text().splitlines()]
    assert [event["stage"] for event in events] == [
        "explore",
        "select",
        "derive",
    ]
    assert all(event["event"] == "mobilegpt_memory_only_miss" for event in events)
    summary = mobilegpt_memory.summarize_mobilegpt_stats(stats_path)
    assert summary["memory_only_miss_count"] == 3
    assert summary["memory_only_stage_counts"] == {
        "derive": 1,
        "explore": 1,
        "select": 1,
    }


def test_mobilegpt_memory_only_guard_is_inert_when_disabled(monkeypatch) -> None:
    class ExploreAgent:
        def explore(self):
            return {"name": "click", "parameters": {"index": 7}}

    class SelectAgent:
        def select(self):
            raise AssertionError("not used")

    class DeriveAgent:
        def derive(self):
            raise AssertionError("not used")

    class MobileGPT:
        def get_next_action(self):
            return ExploreAgent().explore()

    monkeypatch.delenv("MOBILEGPT_MEMORY_ONLY", raising=False)
    install_mobilegpt_memory_only_guard(
        MobileGPT,
        explore_agent_class=ExploreAgent,
        select_agent_class=SelectAgent,
        derive_agent_class=DeriveAgent,
    )

    assert MobileGPT().get_next_action() == {
        "name": "click",
        "parameters": {"index": 7},
    }


def test_mobilegpt_xml_preserves_action_indices_and_indexes_structural_children() -> None:
    xml_text = mobilegpt_compatible_xml(
        '<hierarchy bounds="[0,0][100,200]">'
        '<window id="window-target">'
        '<node id="target-root" package="com.example.target" '
        'bounds="[0,0][100,200]" />'
        '<node id="target-button" package="com.example.target" text="Open" '
        'clickable="true" bounds="[10,20][30,60]" />'
        '</window>'
        '</hierarchy>'
    )

    root = ET.fromstring(xml_text)
    indexed_children = [child for parent in root.iter() for child in parent]
    assert all(child.attrib.get("index") is not None for child in indexed_children)
    assert all(
        isinstance(int(child.attrib["index"]), int) for child in indexed_children
    )
    assert root.find(".//*[@id='target-root']").attrib["index"] == "0"
    assert root.find(".//*[@id='target-button']").attrib["index"] == "1"
    assert int(root.find("./window").attrib["index"]) < 0


def test_mobilegpt_repairs_native_task_app_nested_in_parameters(monkeypatch) -> None:
    monkeypatch.setenv("MOBILEGPT_TARGET_APP", "net.cozic.joplin")
    valid, parsed, error = _parse_mobilegpt_model_response(
        json.dumps(
            {
                "reasoning": "Create a task-local API for the requested note.",
                "found_match": False,
                "api": {
                    "name": "getMeetingAttendeeCountByTitle",
                    "description": "Retrieve attendee count by meeting title.",
                    "parameters": {
                        "title": "The exact meeting title.",
                        "app": "Joplin",
                    },
                },
            }
        ),
        messages=[
            {
                "role": "user",
                "content": 'List of known APIs and return an "api" object.',
            }
        ],
        is_list=False,
    )

    assert valid is True
    assert error == ""
    assert parsed["api"]["app"] == "net.cozic.joplin"
    assert parsed["api"]["parameters"] == {
        "title": "The exact meeting title."
    }


def test_mobilegpt_executes_repeat_click_as_multiple_androidworld_clicks(
    tmp_path: Path,
) -> None:
    class FakeEnv:
        def __init__(self) -> None:
            self.actions: list[SimpleNamespace] = []

        def execute_action(self, action: SimpleNamespace) -> None:
            self.actions.append(action)

    env = FakeEnv()
    agent = build_mobilegpt_agent(
        env=env,
        evidence_root=tmp_path,
        action_factory=lambda **payload: SimpleNamespace(**payload),
    )

    should_continue, answer = agent._execute_server_action(
        {"name": "repeat-click", "parameters": {"index": 0, "number": 2}},
        xml_text=(
            '<hierarchy><node index="0" clickable="true" '
            'bounds="[10,20][30,60]" /></hierarchy>'
        ),
    )

    assert should_continue is True
    assert answer == ""
    assert [action.action_type for action in env.actions] == ["click", "click"]
    assert [(action.x, action.y) for action in env.actions] == [(20, 40), (20, 40)]


def test_mobilegpt_can_execute_native_hit_through_canonical_host() -> None:
    from omniflow.core.model import Action, ActionResult, Observation

    class CanonicalHost:
        def __init__(self) -> None:
            self.actions: list[Action] = []

        def observe(self, **_: object) -> Observation:
            return Observation()

        def act(self, action: Action) -> ActionResult:
            self.actions.append(action)
            return ActionResult(True)

    host = CanonicalHost()
    agent = build_mobilegpt_agent(host=host)

    should_continue, answer = agent._execute_server_action(
        {"name": "click", "parameters": {"index": 0}},
        xml_text=(
            '<hierarchy bounds="[0,0][200,400]">'
            '<node index="0" clickable="true" '
            'bounds="[80,80][120,120]" /></hierarchy>'
        ),
    )

    assert should_continue is True
    assert answer == ""
    assert host.actions == [Action("click", {"x": 500, "y": 250})]


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

        def get_state(self, *, wait_to_stabilize: bool = False) -> SimpleNamespace:
            assert wait_to_stabilize is True
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
            self.interaction_cache = ""

        def get_state(self, *, wait_to_stabilize: bool = False) -> SimpleNamespace:
            assert wait_to_stabilize is True
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
    assert env.interaction_cache == "20"
    assert result.done is True
    assert result.data["answer"] == "20"
    assert result.data["actions_executed"] == 0
    assert result.data["error"] is None
    assert agent.last_result_data == result.data


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

        def get_state(self, *, wait_to_stabilize: bool = False) -> SimpleNamespace:
            assert wait_to_stabilize is True
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
                        bbox_pixels=_Bounds(10, 20, 90, 80),
                        package_name="com.example.launcher",
                        class_name="android.widget.Button",
                        text="Target",
                        content_description="",
                        resource_name="com.example.launcher:id/target",
                        is_clickable=True,
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
    item = CanonicalRunLog(
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
        method_name="mobilegpt",
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
