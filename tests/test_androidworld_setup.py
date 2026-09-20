from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


def test_mobilegpt_episode_command_launches_native_client(tmp_path, monkeypatch):
    from src.experiment import run_task

    monkeypatch.setattr(run_task, "task_goal_for_params", lambda *args, **kwargs: "full\ngoal")
    monkeypatch.setattr(run_task, "_subprocess_env", lambda *args, **kwargs: {})
    item = run_task.CanonicalRunLog("Task", "full\ngoal", {}, tmp_path / "source.json", 113, 1, {})
    spec = run_task.build_mobilegpt_command(
        item, method_name="mobilegpt",
        target=run_task.DeviceTarget("standard45562", "emulator-45562", 45562),
        android_world_root=tmp_path, output_root=tmp_path / "output",
        stats_jsonl=tmp_path / "stats.jsonl", mobilegpt_root=tmp_path / "upstream",
        server_host="0.0.0.0", server_port=17562, target_package="app.example",
        max_steps=30, task_random_seed=113, fixed_task_seed=True,
        fixed_task_params=True, task_params_override=None, perform_emulator_setup=False,
        adb_path="adb", start_timeout_sec=30, finish_timeout_sec=600, repo_root=tmp_path,
    )
    assert spec.argv[2] == "src.integrations.official_forward"
    assert spec.argv[spec.argv.index("--host") + 1] == "10.0.2.2"
    assert spec.argv[spec.argv.index("--instruction") + 1] == "full\ngoal"
    assert spec.metadata["action_backend"] == "mobilegpt_accessibility"
    assert "src.integrations.mobilegpt_oob_client" not in spec.argv


def test_mobilegpt_server_uses_upstream_and_memory_cannot_replace_code(tmp_path, monkeypatch):
    from pathlib import Path
    from src.integrations.official_forward import prepare_mobilegpt_server

    root = tmp_path / "upstream"
    server = root / "Server"
    (server / "memory").mkdir(parents=True)
    (server / "main.py").write_text("server_port = 12345\n")
    (server / "server.py").write_text("upstream protocol")
    (server / "memory/memory_manager.py").write_text("upstream memory")
    stale = tmp_path / "stale/server"
    stale.mkdir(parents=True)
    (stale / "main.py").write_text("patched server")
    monkeypatch.setenv("OMNIFLOW_MOBILEGPT_RUNTIME_ROOT", str(stale.parent))
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "memory_manager.py").write_text("old adapter")
    (memory / "tasks.csv").write_text("learned task")
    result = prepare_mobilegpt_server(
        official_root=root, memory_root=memory, workspace=tmp_path / "run", port=17562
    )
    staged = Path(result["server_root"])
    assert (staged / "main.py").read_text() == "server_port = 17562\n"
    assert (staged / "server.py").read_text() == "upstream protocol"
    assert (staged / "memory/memory_manager.py").read_text() == "upstream memory"
    assert (staged / "memory/tasks.csv").read_text() == "learned task"


def test_mobilegpt_native_environment_does_not_alias_provider_credentials(monkeypatch):
    from src.experiment import run_task

    monkeypatch.setattr(run_task, "_local_dotenv_env", lambda **kwargs: {})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLMTHU_API_KEY", "test-other-provider")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    env = run_task._subprocess_env({"MOBILEGPT_CLIENT_MODE": "upstream_accessibility"})
    assert "OPENAI_API_KEY" not in env
    assert env["OPENAI_BASE_URL"] == "https://example.invalid/v1"


def test_mobilegpt_apk_preserves_upstream_client_except_server_address(tmp_path, monkeypatch):
    from pathlib import Path
    from src.integrations import official_forward

    root = tmp_path / "upstream"
    app = root / "App"
    java = "app/src/main/java/com/example/MobileGPT/MobileGPTGlobal.java"
    originals = {
        java: 'HOST_IP = "INPUT_YOUR_SERVER_IP_ADDRESS"; HOST_PORT = 12345;',
        "app/build.gradle": "    compileSdk 33\n",
        "app/src/main/java/com/example/MobileGPT/MainActivity.java": "upstream activity",
        "app/src/main/java/com/example/MobileGPT/MobileGPTAccessibilityService.java": "upstream service",
    }
    for name, content in originals.items():
        path = app / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def build(argv, *, cwd, **kwargs):
        cwd = Path(cwd)
        for name, content in originals.items():
            expected = content.replace("INPUT_YOUR_SERVER_IP_ADDRESS", "10.0.2.2")
            assert (cwd / name).read_text() == expected
        apk = cwd / "app/build/outputs/apk/debug/app-debug.apk"
        apk.parent.mkdir(parents=True)
        apk.write_bytes(b"test build artifact")

    monkeypatch.setattr(official_forward.subprocess, "run", build)
    output = tmp_path / "client.apk"
    official_forward.prepare_mobilegpt_client_apk(
        official_root=root, output_apk=output, gradle_bin="gradle"
    )
    assert output.read_bytes() == b"test build artifact"

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


@pytest.mark.parametrize("instruction", [
    "Open settings",
    "Add the following recipes:\nRecipe: Avocado Toast\nAn easy meal\n ingredients: avocado",
    "Add the following recipes:\r\nRecipe: Avocado Toast\r directions: toast bread",
])
def test_mobilegpt_speech_keeps_pending_action_on_original_observation(monkeypatch, tmp_path, instruction):
    import json
    from src.integrations import mobilegpt_oob_client as client

    # Official Server sends speech and a physical action for one X frame.
    # A second speech notification must not produce another observation either.
    frames = ["##$$##com.android.settings", *[
        json.dumps({"name": "speak", "parameters": {"message": "Opening settings"}})
        for _ in range(2)
    ], json.dumps({"name": "click", "parameters": {"index": 26}}), "$$$$$"]
    clock = [0.0]

    class Socket:
        def __init__(self):
            self.pending = bytearray(("\n".join(frames) + "\n").encode())
            self.sent = []
            self.timeouts = 4

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def settimeout(self, value):
            pass

        def sendall(self, payload):
            self.sent.append(payload)

        def recv(self, size):
            if self.pending.startswith(b'{"name": "click"') and self.timeouts:
                self.timeouts -= 1
                clock[0] += 3
                raise client.socket.timeout()
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
    server_log = tmp_path / "disposable-server.log"
    server_log.write_text("finish subtask!!\nNext task step is still being planned.\n")
    monkeypatch.setattr(client.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(client.socket, "create_connection", lambda *a, **k: sock)
    monkeypatch.setattr(client, "OobControlClient", lambda *a, **k: oob)
    monkeypatch.setattr(client, "_prelaunch_target_package", lambda *a, **k: "com.android.settings")
    monkeypatch.setattr(client, "_wire_packages", lambda *a: ["com.android.settings"])
    result = client._run_mobilegpt_oob_transport(
        serial="regression-device", adb_path="adb", server_host="127.0.0.1",
        server_port=12345, instruction=instruction, timeout_sec=60,
        max_steps=10, output_root=tmp_path, server_log_path=str(server_log))
    assert result["task_finished"] and result["actions"] == 1
    assert len(oob.actions) == 1 and oob.actions[0]["tool"] == "click"
    assert oob.observations == 2  # initial page, then the physical action result
    assert sum(payload.startswith(b"X") for payload in sock.sent) == 2
    # Upstream reads I through the first LF, then interprets every remaining
    # byte as a new message type (including A for QA). No task body may escape.
    instruction_frame = next(payload for payload in sock.sent if payload.startswith(b"I"))
    received_instruction, remaining = instruction_frame[1:].split(b"\n", 1)
    assert remaining == b""
    assert received_instruction.decode().split() == instruction.split()
    assert b"\r" not in received_instruction
    assert (tmp_path / "official_server.log").read_bytes() == server_log.read_bytes()


@pytest.mark.parametrize("skip,target,expected_calls", [
    ("1", "com.example.app", []),
    ("0", "com.example.app", ["com.example.app"]),
    ("1", "com.other.app", ["com.example.app"]),
])
def test_mobilegpt_explicit_app_skips_external_discovery(monkeypatch, tmp_path, skip, target, expected_calls):
    import os
    from src.integrations.official_forward import _configure_mobilegpt_app_discovery_transport

    path = tmp_path / "agents" / "app_agent.py"
    path.parent.mkdir()
    # Upstream app-list boundary, with network and embedding left injectable.
    path.write_text(
        "def update(new_packages):\n"
        "    result = []\n"
        "    for package_name in new_packages:\n"
        "        if package_name:\n"
        "                app_name, description = get_package_info(package_name)\n"
        "                result.append((app_name, get_openai_embedding(description)))\n"
        "    return result\n"
    )
    monkeypatch.setenv("MOBILEGPT_SKIP_APP_DISCOVERY", skip)
    monkeypatch.setenv("MOBILEGPT_TARGET_PACKAGE", target)
    monkeypatch.setenv("MOBILEGPT_TARGET_APP", "Official app")
    calls, embeddings = [], []
    def lookup(package):
        calls.append(package)
        return "Discovered app", "Discovered description"
    def embed(description):
        embeddings.append(description)
        return [1.0]
    _configure_mobilegpt_app_discovery_transport(tmp_path)
    first = path.read_text()
    _configure_mobilegpt_app_discovery_transport(tmp_path)
    assert path.read_text() == first
    namespace = dict(os=os, get_package_info=lookup, get_openai_embedding=embed)
    exec(compile(first, str(path), "exec"), namespace)
    assert namespace["update"](["com.example.app"])[0][1] == [1.0]
    assert calls == expected_calls
    assert embeddings == (["Discovered description"] if expected_calls else ["Official app"])


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("content,is_list,expected", [
    ("Clicked New Recipe button. Suggested Next plan: Fill in recipe details.", False,
     "Clicked New Recipe button. Suggested Next plan: Fill in recipe details."),
    ('{"action":{"name":"read_screen","parameters":{}}}', False,
     {"action": {"name": "read_screen", "parameters": {}}}),
    ('[{"name":"save_recipe"}]', True, [{"name": "save_recipe"}]),
])
def test_mobilegpt_query_preserves_upstream_text_and_json(tmp_path, prepared, content, is_list, expected):
    import json
    import os
    from src.integrations.official_forward import _configure_mobilegpt_response_compat

    path = tmp_path / "utils" / "utils.py"
    path.parent.mkdir()
    path.write_text('def query(messages, model="gpt-4-turbo", is_list=False):\n    pass\n\n'
                    'def parse_completion_rate(value):\n    return value\n')
    _configure_mobilegpt_response_compat(tmp_path)
    if prepared:
        source = path.read_text().replace(
            "        # Upstream also uses query for plain-text action summaries.\n"
            "        # Preserve its return contract when there is no JSON payload.\n"
            "        return result\n",
            "        # Non-list planner calls require an object with the official action\n"
            "        # schema.  Never pass a prose/string payload into the upstream agent;\n"
            "        # it would fail later with ``string indices must be integers``.\n"
            "        continue\n")
        path.write_text(source)
        _configure_mobilegpt_response_compat(tmp_path)
    events = []
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=None)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))
    client.with_options = lambda **kwargs: client
    namespace = dict(os=os, json=json, OpenAI=lambda **kwargs: client,
                     log=lambda *args: None, write_omniflow_mobilegpt_event=events.append,
                     __parse_json=lambda text, is_list=False: text if text.startswith(('{', '[')) else None)
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    assert namespace["query"]([{"role": "user", "content": "Upstream request"}], is_list=is_list) == expected
    assert [event["event"] for event in events] == ["chat_call"]


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


@pytest.mark.parametrize('reward', [0.0, 1.0])
@pytest.mark.parametrize('finished', [False, True])
def test_mobilegpt_official_result_is_independent_of_method_exit(monkeypatch, tmp_path, reward, finished):
    import json
    from contextlib import nullcontext
    from src.integrations import mobilegpt_oob_client as client, official_forward

    task = SimpleNamespace(goal='Turn bluetooth on.', is_successful=lambda env: reward)
    monkeypatch.setattr(official_forward, '_androidworld_task_startup',
                        lambda **kwargs: nullcontext((object(), task)))
    method_result = {'task_finished': finished, 'returncode': 0 if finished else 1,
                     'reason': '' if finished else 'mobilegpt_oob_action_target_missing',
                     'actions': 3, 'planner_steps': 5}
    monkeypatch.setattr(client, '_run_mobilegpt_oob_transport', lambda **kwargs: dict(method_result))
    monkeypatch.delenv('MOBILEGPT_STATS_JSONL', raising=False)
    result = client.run_mobilegpt_oob_client(
        serial='regression-device', adb_path='adb', server_host='127.0.0.1',
        server_port=12345, instruction=task.goal, timeout_sec=10, max_steps=10,
        output_root=tmp_path, android_world_root=str(tmp_path), task_name='SystemBluetoothTurnOn')
    row = json.loads((tmp_path / 'task_results.jsonl').read_text())
    assert result['validator_success'] == (reward > 0.5)
    assert row['official_validator_success'] == (reward > 0.5)
    assert row['androidworld_validator_result']['success'] == (reward > 0.5)
    assert row['androidworld_validator_result']['reward'] == reward
    assert row['mobilegpt_protocol']['task_finished'] is finished
    assert row['process_returncode'] == method_result['returncode']
    assert row['environment_failure'] is (not finished)
    from src.experiment.run_task import _materialize_mobilegpt_canonical_run_log
    runlog_path = _materialize_mobilegpt_canonical_run_log(
        output_path=tmp_path,
        item=SimpleNamespace(task='SystemBluetoothTurnOn', goal=task.goal, params={}, replay_seed=111),
        target=SimpleNamespace(serial='regression-device'), task_seed=113)
    runlog = json.loads(runlog_path.read_text())
    assert runlog['success'] == (reward > 0.5)
    assert runlog['validator']['success'] == (reward > 0.5)
    assert runlog['diagnostics']['mobilegpt_result']['process_returncode'] == method_result['returncode']
    assert runlog['diagnostics']['mobilegpt_result']['failure_reason'] == method_result['reason']
    monkeypatch.setattr(client, 'run_mobilegpt_oob_client', lambda **kwargs: result)
    assert client.main(['--serial', 'regression-device', '--adb', 'adb', '--server-port', '12345',
                        '--instruction', task.goal, '--timeout', '10', '--output', str(tmp_path)]) == method_result['returncode']
