from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import re
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import pytest

import src.integrations.official_forward as official_forward
from src.integrations.official_forward import (
    MOBILEGPT_HANDSHAKE_RETURN_CODE,
    _mobilegpt_instruction_broadcast_args,
    _mobilegpt_protocol_probe,
    _count_appagent_actions,
    _autodroid_task_app_name,
    _autodroid_official_memory_key,
    _autodroid_display_ids,
    _count_droidbot_output_events,
    _configure_mobilegpt_client_launch_lifecycle,
    _configure_mobilegpt_telemetry,
    _prepare_autodroid_device,
    prepare_appagent_workspace,
    prepare_mobilegpt_server,
    resolve_mobilegpt_client_host,
    run_appagent_executor,
    validate_autodroid_memory_root,
    write_adb_proxy,
)


def test_mobilegpt_accessibility_binding_requires_the_mobilegpt_service_in_bound_block() -> None:
    dumpsys = """
User state[
     Bound services:{Service[label=AndroidWorld Accessibility Forwarder,
       id=com.google.androidenv.accessibilityforwarder/.AccessibilityForwarder]}
     Enabled services:{{com.google.androidenv.accessibilityforwarder/.AccessibilityForwarder,
       com.example.MobileGPT/.MobileGPTAccessibilityService}}
]
"""

    assert official_forward._mobilegpt_accessibility_service_bound(
        dumpsys,
        "com.example.MobileGPT/.MobileGPTAccessibilityService",
    ) is False


def test_mobilegpt_accessibility_binding_accepts_the_official_service_label() -> None:
    dumpsys = """
User state[
     Bound services:{Service[label=com.example.MobileGPT.MobileGPTAccessibilityService,
       feedbackType[FEEDBACK_GENERIC], eventTypes=TYPES_ALL_MASK]}
     Enabled services:{{com.example.MobileGPT/.MobileGPTAccessibilityService}}
]
"""

    assert official_forward._mobilegpt_accessibility_service_bound(
        dumpsys,
        "com.example.MobileGPT/.MobileGPTAccessibilityService",
    ) is True


def test_mobilegpt_accessibility_binding_does_not_toggle_an_already_bound_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_adb(
        _adb_path: str,
        _serial: str,
        args: list[str],
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        del check
        calls.append(args)
        return subprocess.CompletedProcess(
            args,
            0,
            stdout=(
                "Bound services:{Service[label=MobileGPT Accessibility]}\n"
                "Enabled services:{{com.example.MobileGPT/"
                "com.example.MobileGPT.MobileGPTAccessibilityService}}\n"
                "Binding services:{}\n"
            ),
        )

    monkeypatch.setattr(official_forward, "_run_adb", fake_adb)

    assert official_forward._ensure_mobilegpt_accessibility_service_bound(
        "adb",
        "emulator-5562",
        "com.example.MobileGPT/.MobileGPTAccessibilityService",
        ["com.example.MobileGPT/.MobileGPTAccessibilityService"],
        initial_attempts=1,
        retry_attempts=1,
    ) is True
    assert calls == [["shell", "dumpsys", "accessibility"]]


def test_official_forward_accepts_appagent_step_budget() -> None:
    source = Path(__file__).parents[1] / "src" / "integrations" / "official_forward.py"

    assert 'parser.add_argument("--max-steps", type=int, default=0)' in source.read_text(
        encoding="utf-8"
    )


def test_mobilegpt_staged_client_arms_first_screen_before_launch(tmp_path: Path) -> None:
    client = tmp_path / "client"
    service = client / "app/src/main/java/com/example/MobileGPT"
    service.mkdir(parents=True)
    path = service / "MobileGPTAccessibilityService.java"
    path.write_text(
        """
        if (launchIntent != null) {
            startActivity(launchIntent);//null pointer check in case package name was not found
        } else {
            Log.d(TAG, "intent null");
        }
        xmlPending = true;
        screenNeedUpdate = true;
        firstScreen = true;
        """,
        encoding="utf-8",
    )

    _configure_mobilegpt_client_launch_lifecycle(client)
    staged = path.read_text(encoding="utf-8")
    assert staged.index("xmlPending = true;") < staged.index("startActivity(launchIntent)")
    assert "postDelayed(screenUpdateTimeoutRunnable, 5000)" in staged


def test_mobilegpt_server_skips_stale_explore_trigger_ui(tmp_path: Path) -> None:
    server = tmp_path / "Server"
    parsing = server / "utils"
    parsing.mkdir(parents=True)
    path = parsing / "parsing_utils.py"
    path.write_text(
        "from utils.utils import log\n\n"
        "def get_trigger_ui_attributes(trigger_ui_indexes, screen):\n"
        "    trigger_ui_data = {}\n"
        "    for subtask_name, ui_indexes in trigger_ui_indexes.items():\n"
        "        trigger_uis_attributes = []\n"
        "        for ui_index in ui_indexes:\n"
        "            ui_attributes = get_ui_key_attrib(int(ui_index), screen)\n\n"
        "            skip = False\n"
        "            trigger_uis_attributes.append(ui_attributes)\n",
        encoding="utf-8",
    )

    official_forward._configure_mobilegpt_runtime_guards(server)
    staged = path.read_text(encoding="utf-8")

    assert "omniflow_mobilegpt_invalid_trigger_ui" in staged
    assert "except (AttributeError, TypeError, ValueError):" in staged
    assert "continue" in staged


def test_mobilegpt_staged_server_honors_runtime_port(tmp_path: Path) -> None:
    server = tmp_path / "Server"
    server.mkdir(parents=True)
    path = server / "main.py"
    path.write_text(
        'import os\n'
        'def main():\n'
        '    server_ip = "0.0.0.0"\n'
        '    server_port = 12345\n'
        '    return server_ip, server_port\n',
        encoding="utf-8",
    )

    official_forward._configure_mobilegpt_server_port(server)
    staged = path.read_text(encoding="utf-8")

    assert 'os.getenv("MOBILEGPT_SERVER_HOST", "0.0.0.0")' in staged
    assert 'os.getenv("MOBILEGPT_SERVER_PORT", "12345")' in staged


def test_mobilegpt_server_records_memory_recall_and_explore(tmp_path: Path) -> None:
    server = tmp_path / "Server"
    server.mkdir(parents=True)
    path = server / "mobilegpt.py"
    path.write_text(
        "from utils.utils import log, parse_completion_rate\n\n"
        "class MobileGPT:\n"
        "    def get_next_action(self, parsed_xml=None, hierarchy_xml=None, encoded_xml=None):\n"
        "        page_index, new_subtasks = self.memory.search_node(parsed_xml, hierarchy_xml, encoded_xml)\n\n"
        "        if page_index == -1:\n"
        "            page_index = self.explore_agent.explore(parsed_xml, hierarchy_xml, encoded_xml)\n"
        "        next_action = self.memory.get_next_action(self.current_subtask, self.encoded_xml)\n"
        "        current_action_data = {\"page_index\": self.current_page_index, \"action\": next_action, \"screen\": self.encoded_xml,\n",
        encoding="utf-8",
    )

    official_forward._configure_mobilegpt_memory_telemetry(server)
    staged = path.read_text(encoding="utf-8")

    assert '"event": "memory_lookup"' in staged
    assert '"event": ("memory_action_recalled" if next_action else "memory_action_miss")' in staged


def test_mobilegpt_staged_client_retries_instead_of_sending_empty_xml(
    tmp_path: Path,
) -> None:
    client = tmp_path / "client"
    service = client / "app/src/main/java/com/example/MobileGPT"
    service.mkdir(parents=True)
    path = service / "MobileGPTAccessibilityService.java"
    path.write_text(
        """
        private void saveCurrScreenXML() {
            nodeMap = new HashMap<>();
            AccessibilityNodeInfo rootNode = getRootForActiveApp();
            if (rootNode != null) {
                currentScreenXML = AccessibilityNodeInfoDumper.dumpWindow(rootNode, nodeMap, fileDirectory);
            }
        }

        private void sendScreen(){
            mExecutorService.execute(()->mClient.sendScreenshot(currentScreenShot));
            mExecutorService.execute(()-> mClient.sendXML(currentScreenXML));
        }
        """,
        encoding="utf-8",
    )

    _configure_mobilegpt_client_launch_lifecycle(client)
    staged = path.read_text(encoding="utf-8")

    assert 'currentScreenXML = "";' in staged
    assert "currentScreenXML.trim().isEmpty()" in staged
    assert "postDelayed(screenUpdateTimeoutRunnable, 500)" in staged
    assert staged.index("currentScreenXML.trim().isEmpty()") < staged.index(
        "mClient.sendXML(currentScreenXML)"
    )


def test_mobilegpt_staged_client_observes_again_after_official_speak(
    tmp_path: Path,
) -> None:
    client = tmp_path / "client"
    service = client / "app/src/main/java/com/example/MobileGPT"
    service.mkdir(parents=True)
    path = service / "MobileGPTAccessibilityService.java"
    path.write_text(
        """
            if (action.equals("speak")) {
                String content = (String) args.get("message");
                mSpeech.speak(content, false);
                return;
            }
        """,
        encoding="utf-8",
    )

    _configure_mobilegpt_client_launch_lifecycle(client)
    staged = path.read_text(encoding="utf-8")

    assert "omniflow_mobilegpt_speak_lifecycle" in staged
    assert staged.index("mSpeech.speak(content, false)") < staged.index(
        "postDelayed(screenUpdateTimeoutRunnable, 500)"
    )


def test_mobilegpt_staged_client_selects_primary_same_package_window(
    tmp_path: Path,
) -> None:
    client = tmp_path / "client"
    service = client / "app/src/main/java/com/example/MobileGPT"
    service.mkdir(parents=True)
    path = service / "MobileGPTAccessibilityService.java"
    path.write_text(
        """
    private AccessibilityNodeInfo getRootForActiveApp(){
        List<AccessibilityWindowInfo> windows = getWindows();

        for (AccessibilityWindowInfo window : windows) {
            AccessibilityNodeInfo root = window.getRoot();
            if (root.getPackageName().equals(targetPackageName)) {
                return root;
            }
        }
        Log.d(TAG, "No Appropriate Root found in this screen.");
        return null;
    }
""",
        encoding="utf-8",
    )

    _configure_mobilegpt_client_launch_lifecycle(client)
    staged = path.read_text(encoding="utf-8")

    assert "omniflow_mobilegpt_primary_app_window" in staged
    assert "largestArea" in staged
    assert "root.getBoundsInScreen(bounds)" in staged


def test_mobilegpt_protocol_probe_distinguishes_handshake_failure(
    tmp_path: Path,
) -> None:
    stats = tmp_path / "mobilegpt_stats.jsonl"
    stats.write_text("", encoding="utf-8")

    probe = _mobilegpt_protocol_probe(
        stats,
        "MobileGPT_Service: # of Apps : 21\n"
        "MobileGPT_CLIENT: server offline\n",
    )

    assert probe["client_service_ready"] is True
    assert probe["client_error"] is True
    assert probe["task_started"] is False
    assert probe["phase"] == "client_server_handshake"
    assert MOBILEGPT_HANDSHAKE_RETURN_CODE == 127


def test_mobilegpt_instruction_broadcast_quotes_androidworld_goal() -> None:
    args = _mobilegpt_instruction_broadcast_args(
        "Create a new folder named folder_20260822_131011."
    )

    assert args[-1] == "'Create a new folder named folder_20260822_131011.'"


def test_mobilegpt_protocol_probe_marks_episode_after_server_task_start(
    tmp_path: Path,
) -> None:
    stats = tmp_path / "mobilegpt_stats.jsonl"
    stats.write_text(
        '{"event":"task_started"}\n{"event":"mobilegpt_action_sent"}\n',
        encoding="utf-8",
    )

    probe = _mobilegpt_protocol_probe(stats, "MobileGPT_Service: receive broadcast")

    assert probe["task_started"] is True
    assert probe["action_sent_count"] == 1
    assert probe["phase"] == "episode"


def test_mobilegpt_protocol_probe_marks_episode_after_action_without_task_start(
    tmp_path: Path,
) -> None:
    stats = tmp_path / "mobilegpt_stats.jsonl"
    stats.write_text(
        '{"event":"mobilegpt_action_sent","action":"click"}\n',
        encoding="utf-8",
    )

    probe = _mobilegpt_protocol_probe(stats, "MobileGPT_Service: receive broadcast")

    assert probe["task_started"] is False
    assert probe["action_sent_count"] == 1
    assert probe["phase"] == "episode"


def test_mobilegpt_protocol_probe_detects_server_handler_crash(
    tmp_path: Path,
) -> None:
    stats = tmp_path / "mobilegpt_stats.jsonl"
    stats.write_text("", encoding="utf-8")

    probe = _mobilegpt_protocol_probe(
        stats,
        "MobileGPT_Service: receive broadcast\n",
        "Exception in thread Thread-1\nopenai.OpenAIError: Missing credentials\n",
    )

    assert probe["server_error"] is True
    assert "missing credentials" in probe["server_error_markers"]
    assert probe["task_started"] is False


def test_mobilegpt_protocol_probe_ignores_auxiliary_connection_reset(
    tmp_path: Path,
) -> None:
    stats = tmp_path / "mobilegpt_stats.jsonl"
    stats.write_text("", encoding="utf-8")

    probe = _mobilegpt_protocol_probe(
        stats,
        "MobileGPT_Service: receive broadcast\n",
        "Exception in thread Thread-5 (handle_client):\n"
        "Traceback (most recent call last):\n"
        "  File \"server.py\", line 59, in handle_client\n"
        "    raw_message_type = client_socket.recv(1)\n"
        "ConnectionResetError: [Errno 104] Connection reset by peer\n",
    )

    assert probe["server_error"] is False
    assert probe["server_error_markers"] == []


@pytest.mark.parametrize(
    ("failure_reason", "returncode"),
    [
        ("mobilegpt_handshake_timeout", 127),
        ("mobilegpt_handshake_failed", 127),
        ("mobilegpt_server_handler_failed", 128),
        ("", 127),
        ("", 128),
    ],
)
def test_mobilegpt_transport_failures_are_retryable_environment_failures(
    failure_reason: str,
    returncode: int,
) -> None:
    assert official_forward._mobilegpt_environment_failure(
        failure_reason=failure_reason,
        returncode=returncode,
    ) is True


@pytest.mark.parametrize(
    ("failure_reason", "returncode"),
    [
        ("mobilegpt_step_timeout", 126),
        ("mobilegpt_step_budget_exhausted", 125),
        ("", 0),
    ],
)
def test_mobilegpt_method_failures_remain_terminal_conclusions(
    failure_reason: str,
    returncode: int,
) -> None:
    assert official_forward._mobilegpt_environment_failure(
        failure_reason=failure_reason,
        returncode=returncode,
    ) is False


def test_autodroid_official_memory_key_uses_paper_app_names() -> None:
    assert _autodroid_official_memory_key("audio") == "voicerecorder"
    assert _autodroid_official_memory_key("files") == "filemanager"
    assert _autodroid_official_memory_key("camera") == "camera"


def test_autodroid_display_ids_detect_fold_secondary_display() -> None:
    dump = "mDisplayId=0\nmDisplayId=3\nmDisplayId=0\n"

    assert _autodroid_display_ids(dump) == (0, 3)


def test_prepare_autodroid_device_normalizes_fold_display(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(_adb_path, _serial, args, **_kwargs):
        calls.append(list(args))
        output = "mDisplayId=0\nmDisplayId=3\n" if args[-1] == "display" else ""
        return SimpleNamespace(returncode=0, stdout=output)

    monkeypatch.setattr("src.integrations.official_forward._run_adb", fake_run)

    result = _prepare_autodroid_device(
        adb_path="adb",
        serial="emulator-5564",
        package="com.android.camera2",
    )

    assert result["multiple_displays"] is True
    assert result["display_ids"] == [0, 3]
    assert calls[:2] == [
        ["shell", "input", "keyevent", "KEYCODE_HOME"],
        ["shell", "cmd", "statusbar", "collapse"],
    ]
    assert calls[-1][-7:] == [
        "-W",
        "-a",
        "android.intent.action.MAIN",
        "-c",
        "android.intent.category.LAUNCHER",
        "-p",
        "com.android.camera2",
    ]


def test_count_droidbot_output_events_counts_only_emitted_events(
    tmp_path: Path,
) -> None:
    output = tmp_path / "droidbot"
    (output / "events").mkdir(parents=True)
    (output / "events" / "event_001.json").write_text("{}\n", encoding="utf-8")
    (output / "camera" / "events").mkdir(parents=True)
    (output / "camera" / "events" / "event_002.json").write_text(
        "{}\n", encoding="utf-8"
    )
    (output / "events" / "not-an-event.txt").write_text("\n", encoding="utf-8")

    assert _count_droidbot_output_events(output) == 2


@contextmanager
def _fake_androidworld_session(task: object):
    yield object(), task


def test_legacy_mobilegpt_client_entry_is_disabled(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        RuntimeError,
        match="mobilegpt_legacy_client_disabled_use_oob",
    ):
        official_forward.run_mobilegpt_client(
            official_root=tmp_path / "MobileGPT",
            serial="emulator-5560",
            adb_path="adb",
            host="127.0.0.1",
            instruction="Run the stopwatch.",
            output_root=tmp_path / "result",
            timeout_sec=10,
        )


def test_appagent_forwarder_only_mounts_official_inputs(tmp_path: Path) -> None:
    official = tmp_path / "AppAgent"
    (official / "scripts").mkdir(parents=True)
    (official / "run.py").write_text("print('official')\n", encoding="utf-8")
    official_executor = (
        'import os\nimport re\n'
        'doc_path = os.path.join(docs_dir, f"{elem.uid}.txt")\n'
    )
    (official / "scripts" / "task_executor.py").write_text(
        official_executor, encoding="utf-8"
    )
    docs = tmp_path / "memory" / "calculator" / "demo_docs"
    docs.mkdir(parents=True)
    (docs / "button.txt").write_text("{}\n", encoding="utf-8")

    result = prepare_appagent_workspace(
        official_root=official,
        docs_root=docs,
        workspace=tmp_path / "workspace",
        app_name="calculator",
        serial="emulator-5554",
        adb_path="adb",
        config={"MODEL": "OpenAI"},
    )

    assert Path(result["workspace"]).is_dir()
    staged_scripts = Path(result["workspace"]) / "scripts"
    assert staged_scripts.is_dir()
    assert not staged_scripts.is_symlink()
    assert (staged_scripts / "oob_appagent_controller.py").is_file()
    assert "oob_appagent_controller" in (
        staged_scripts / "and_controller.py"
    ).read_text(encoding="utf-8")
    staged_model = staged_scripts / "model.py"
    if staged_model.is_file():
        staged_model_source = staged_model.read_text(encoding="utf-8")
        assert "omniflow_appagent_glm_response_compat" in staged_model_source
        assert "omniflow_appagent_action_parse_compat" in staged_model_source
        assert 'payload["thinking"] = {"type": thinking_mode}' in staged_model_source
        assert 'os.environ.get("APPAGENT_THINKING", "disabled")' in staged_model_source
    assert "_omniflow_resolution_agnostic_doc_path" in (
        staged_scripts / "task_executor.py"
    ).read_text(encoding="utf-8")
    compile(
        (staged_scripts / "task_executor.py").read_text(encoding="utf-8"),
        str(staged_scripts / "task_executor.py"),
        "exec",
    )
    assert (official / "scripts" / "task_executor.py").read_text(
        encoding="utf-8"
    ) == official_executor
    assert Path(result["app_dir"]).is_symlink()
    assert Path(result["app_dir"]).resolve() == docs.parent.resolve()
    proxy_text = Path(result["adb_proxy"]).read_text(encoding="utf-8")
    assert "emulator-5554" in proxy_text
    assert "Physical size:" in proxy_text


def test_appagent_adb_proxy_exposes_override_size_for_xml_coordinates(
    tmp_path: Path,
) -> None:
    real_adb = tmp_path / "adb"
    real_adb.write_text(
        "#!/bin/sh\n"
        "printf 'Physical size: 1768x2208\\nOverride size: 2208x1840\\n'\n",
        encoding="utf-8",
    )
    real_adb.chmod(0o755)
    proxy = write_adb_proxy(
        tmp_path / "workspace",
        serial="emulator-5554",
        adb_path=str(real_adb),
    )
    result = subprocess.run(
        [str(proxy), "-s", "emulator-5554", "shell", "wm", "size"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "Physical size: 2208x1840"


def test_adb_proxy_forwards_implicit_official_calls_to_selected_device(
    tmp_path: Path,
) -> None:
    real_adb = tmp_path / "adb"
    args_file = tmp_path / "args"
    real_adb.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$@\" > {args_file}\n"
        "printf 'device\\n'\n",
        encoding="utf-8",
    )
    real_adb.chmod(0o755)

    proxy = write_adb_proxy(
        tmp_path / "workspace",
        serial="emulator-5590",
        adb_path=str(real_adb),
    )

    result = subprocess.run(
        [str(proxy), "wait-for-device"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert args_file.read_text(encoding="utf-8").splitlines() == [
        "-s",
        "emulator-5590",
        "wait-for-device",
    ]


def test_androidworld_task_startup_reuses_current_activity_normalizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.integrations.android_world import run_episode

    events: list[str] = []
    adb_utils = SimpleNamespace()
    android_world_module = ModuleType("android_world")
    env_module = ModuleType("android_world.env")
    env_module.adb_utils = adb_utils
    android_world_module.env = env_module
    monkeypatch.setitem(sys.modules, "android_world", android_world_module)
    monkeypatch.setitem(sys.modules, "android_world.env", env_module)

    def original_current_activity(*_args, **_kwargs):
        return "package-only", object()

    def normalized_current_activity(*_args, **_kwargs):
        return "com.example/com.example.Main", object()

    adb_utils.get_current_activity = original_current_activity

    def patch_current_activity(module):
        assert module is adb_utils
        events.append("patched")
        module.get_current_activity = normalized_current_activity
        return original_current_activity

    monkeypatch.setattr(
        run_episode,
        "_patch_androidworld_current_activity",
        patch_current_activity,
    )

    class Task:
        def tear_down(self, _env):
            events.append("tear_down")

    class Env:
        def close(self):
            events.append("close")

    env = Env()
    task = Task()
    monkeypatch.setattr(
        run_episode,
        "start_androidworld_task_session",
        lambda **_kwargs: (SimpleNamespace(env=env), task),
    )

    with official_forward._androidworld_task_startup(
        android_world_root="android-world",
        task_name="OpenAppTaskEval",
        task_params_json='{"app_name":"clock"}',
        task_seed=111,
        console_port=5560,
        grpc_port=8560,
        adb_path="adb",
        perform_emulator_setup=False,
    ):
        assert adb_utils.get_current_activity is normalized_current_activity

    assert adb_utils.get_current_activity is original_current_activity
    assert events == ["patched", "tear_down", "close"]


def test_appagent_forwarder_writes_validator_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Task:
        def is_successful(self, _env: object) -> float:
            return 1.0

    monkeypatch.setattr(
        "src.integrations.official_forward._androidworld_task_startup",
        lambda **_kwargs: _fake_androidworld_session(Task()),
    )
    monkeypatch.setattr(
        "src.integrations.official_forward.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    executor = tmp_path / "task_executor.py"
    executor.write_text("print('official')\n", encoding="utf-8")
    output = tmp_path / "result"
    assert run_appagent_executor(
        python_executable=sys.executable,
        executor=executor,
        app_name="settings",
        serial="emulator-5554",
        workspace=workspace,
        goal="Turn on Bluetooth",
        timeout_sec=10,
        android_world_root=tmp_path / "android-world",
        task_name="TurnOffWifiAndTurnOnBluetooth",
        task_params_json="{}",
        task_seed=113,
        console_port=5554,
        grpc_port=8554,
        adb_path="adb",
        output_root=output,
        perform_emulator_setup=False,
    ) == 0

    row = json.loads((output / "task_results.jsonl").read_text())
    assert row["official_validator_used"] is True
    assert row["official_validator_success"] is True
    assert row["androidworld_validator_result"]["validator"] == (
        "androidworld_official"
    )


def test_appagent_camera_prelaunch_uses_oob_instead_of_adb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Task:
        def is_successful(self, _env: object) -> float:
            return 1.0

    class FakeOob:
        instances: list["FakeOob"] = []

        def __init__(self, *_args, **_kwargs) -> None:
            self.actions: list[dict] = []
            self.instances.append(self)

        def observe(self, *, wait_to_stabilize: bool = False) -> dict:
            del wait_to_stabilize
            return {"package_name": "com.android.camera2"}

        def act(self, action: dict) -> dict:
            self.actions.append(action)
            return {"success": True}

    monkeypatch.setattr(
        "src.integrations.official_forward._androidworld_task_startup",
        lambda **_kwargs: _fake_androidworld_session(Task()),
    )
    monkeypatch.setattr(
        "src.integrations.official_forward.OobControlClient",
        FakeOob,
        raising=False,
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    executor = tmp_path / "task_executor.py"
    executor.write_text("print('official')\n", encoding="utf-8")
    output = tmp_path / "result"

    assert run_appagent_executor(
        python_executable=sys.executable,
        executor=executor,
        app_name="camera2",
        serial="emulator-5554",
        workspace=workspace,
        goal="Take a photo",
        timeout_sec=10,
        android_world_root=tmp_path / "android-world",
        task_name="CameraTakePhoto",
        task_params_json="{}",
        task_seed=113,
        console_port=5554,
        grpc_port=8554,
        adb_path="adb",
        output_root=output,
        perform_emulator_setup=False,
    ) == 0

    assert len(FakeOob.instances) == 1
    assert FakeOob.instances[0].actions == [
        {"tool": "open_app", "args": {"package_name": "com.android.camera2"}}
    ]
    row = json.loads((output / "task_results.jsonl").read_text())
    assert row["target_app_prelaunch_package"] == "com.android.camera2"
    assert row["target_app_prelaunch_returncode"] == 0


def test_appagent_action_count_strips_ansi_and_excludes_finish() -> None:
    log = (
        "\x1b[33mRound 1\n"
        "\x1b[33mAction:\n"
        "\x1b[35mtap(6)\n"
        "\x1b[33mRound 2\n"
        "\x1b[33mAction:\n"
        "\x1b[35mtap(3)\n"
        "\x1b[33mRound 3\n"
        "\x1b[33mAction:\n"
        "\x1b[35mFINISH\n"
    )

    assert _count_appagent_actions(log) == 2


def test_appagent_validator_failure_keeps_clean_process_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Task:
        def is_successful(self, _env: object) -> float:
            return 0.0

    monkeypatch.setattr(
        "src.integrations.official_forward._androidworld_task_startup",
        lambda **_kwargs: _fake_androidworld_session(Task()),
    )
    output = tmp_path / "result"
    log_path = output / "official_appagent.log"
    official_output = (
        "\x1b[33mRound 1\n\x1b[33mAction:\n\x1b[35mtap(6)\n"
        "\x1b[33mRound 2\n\x1b[33mAction:\n\x1b[35mFINISH\n"
    )

    def fake_run(*_args, **_kwargs):
        output.mkdir(parents=True, exist_ok=True)
        log_path.write_text(official_output, encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "src.integrations.official_forward.subprocess.run", fake_run
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    executor = tmp_path / "task_executor.py"
    executor.write_text(
        "print(" + repr(official_output) + ")\n", encoding="utf-8"
    )
    assert run_appagent_executor(
        python_executable=sys.executable,
        executor=executor,
        app_name="settings",
        serial="emulator-5554",
        workspace=workspace,
        goal="Turn on Bluetooth",
        timeout_sec=10,
        android_world_root=tmp_path / "android-world",
        task_name="TurnOffWifiAndTurnOnBluetooth",
        task_params_json="{}",
        task_seed=113,
        console_port=5554,
        grpc_port=8554,
        adb_path="adb",
        output_root=output,
        perform_emulator_setup=False,
    ) == 0

    row = json.loads((output / "task_results.jsonl").read_text())
    assert row["process_returncode"] == 0
    assert row["classification"] == "method_failure"
    assert row["official_validator_success"] is False
    assert row["actions_executed"] == 1
    assert row["model_calls"] == 0
    assert row["target_app_prelaunch_package"] == ""
    assert row["target_app_prelaunch_returncode"] is None


def test_appagent_step_budget_stops_official_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Task:
        def is_successful(self, _env: object) -> float:
            return 0.0

    monkeypatch.setattr(
        "src.integrations.official_forward._androidworld_task_startup",
        lambda **_kwargs: _fake_androidworld_session(Task()),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    executor = tmp_path / "task_executor.py"
    executor.write_text(
        "import time\n"
        "for index in range(10):\n"
        "    print(f'Round {index + 1}', flush=True)\n"
        "    print('Action:', flush=True)\n"
        "    print('tap(1)', flush=True)\n"
        "    time.sleep(0.1)\n",
        encoding="utf-8",
    )

    output = tmp_path / "result"
    assert run_appagent_executor(
        python_executable=sys.executable,
        executor=executor,
        app_name="settings",
        serial="emulator-5554",
        workspace=workspace,
        goal="Turn on Bluetooth",
        timeout_sec=5,
        android_world_root=tmp_path / "android-world",
        task_name="TurnOffWifiAndTurnOnBluetooth",
        task_params_json="{}",
        task_seed=113,
        console_port=5554,
        grpc_port=8554,
        adb_path="adb",
        output_root=output,
        perform_emulator_setup=False,
        max_steps=2,
    ) == 125

    row = json.loads((output / "task_results.jsonl").read_text())
    assert row["runtime_integrity_error"] == "appagent_step_budget_exhausted"
    assert row["actions_executed"] >= 2


def test_mobilegpt_forwarder_configures_staged_glm_models_and_overlays_memory(
    tmp_path: Path,
) -> None:
    official = tmp_path / "MobileGPT"
    server = official / "Server"
    (server / "memory").mkdir(parents=True)
    (server / "main.py").write_text("print('official server')\n", encoding="utf-8")
    (server / "memory" / "memory_manager.py").write_text(
        "# official memory package\n", encoding="utf-8"
    )
    memory = tmp_path / "prepared"
    (memory / "frozen_memory" / "com.example.app").mkdir(parents=True)
    (memory / "frozen_memory" / "com.example.app" / "pages.csv").write_text(
        "index\n", encoding="utf-8"
    )

    result = prepare_mobilegpt_server(
        official_root=official,
        memory_root=memory,
        workspace=tmp_path / "workspace",
    )

    staged = Path(result["server_root"])
    assert (staged / "main.py").read_text(encoding="utf-8") == (
        "print('official server')\n"
    )
    assert (staged / "memory" / "memory_manager.py").is_file()
    assert (staged / "memory" / "com.example.app" / "pages.csv").is_file()


def test_mobilegpt_source_cold_build_writes_through_staged_memory(
    tmp_path: Path,
) -> None:
    official = tmp_path / "MobileGPT"
    server = official / "Server"
    (server / "memory").mkdir(parents=True)
    (server / "main.py").write_text("print('official server')\n", encoding="utf-8")
    (server / "memory" / "memory_manager.py").write_text(
        "# official memory implementation\n",
        encoding="utf-8",
    )
    memory = tmp_path / "cold-memory"
    memory.mkdir()

    result = prepare_mobilegpt_server(
        official_root=official,
        memory_root=memory,
        workspace=tmp_path / "workspace",
        write_through_memory=True,
    )

    staged_memory = Path(result["server_root"]) / "memory"
    assert staged_memory.is_symlink()
    assert (staged_memory / "memory_manager.py").read_text(encoding="utf-8") == (
        "# official memory implementation\n"
    )
    (staged_memory / "tasks.csv").write_text("name\n", encoding="utf-8")
    assert (memory / "tasks.csv").read_text(encoding="utf-8") == "name\n"


def test_mobilegpt_forwarder_keeps_system_app_catalog_embeddings_valid(
    tmp_path: Path,
) -> None:
    official = tmp_path / "MobileGPT"
    server = official / "Server"
    (server / "memory").mkdir(parents=True)
    (server / "agents").mkdir()
    (server / "main.py").write_text("# official\n", encoding="utf-8")
    original_app_agent = (
        "import os\n\n"
        "def update(package_name):\n"
        "                app_name, description = get_package_info(package_name)\n"
        "                if description:\n"
        "                    embedding = get_openai_embedding(description)\n"
        "                else:\n"
        "                    embedding = \"\"\n"
    )
    (server / "agents" / "app_agent.py").write_text(
        original_app_agent,
        encoding="utf-8",
    )
    memory = tmp_path / "cold-memory"
    memory.mkdir()

    result = prepare_mobilegpt_server(
        official_root=official,
        memory_root=memory,
        workspace=tmp_path / "workspace",
        write_through_memory=True,
    )

    staged = (
        Path(result["server_root"]) / "agents" / "app_agent.py"
    ).read_text(encoding="utf-8")
    assert "mobilegpt_system_app_catalog_fallback" in staged
    assert "MOBILEGPT_TARGET_PACKAGE" in staged
    assert "MOBILEGPT_TARGET_APP" in staged
    assert "embedding = get_openai_embedding(description)" in staged
    assert (server / "agents" / "app_agent.py").read_text(
        encoding="utf-8"
    ) == original_app_agent


def test_mobilegpt_forwarder_keeps_official_server_source_unchanged(
    tmp_path: Path,
) -> None:
    official = tmp_path / "MobileGPT"
    server = official / "Server"
    (server / "memory").mkdir(parents=True)
    (server / "utils").mkdir()
    (server / "agents").mkdir()
    (server / "memory" / "memory_manager.py").write_text(
        'import os\n'
        'def select(candidates):\n'
        '    highest_similarity = candidates[0]\n'
        '    if highest_similarity > 0.95:\n'
        '        return 0\n',
        encoding="utf-8",
    )
    (server / "main.py").write_text(
        'os.environ["TASK_AGENT_GPT_VERSION"] = "gpt-4o"\n'
        'os.environ["vision_model"] = "gpt-4o"\n',
        encoding="utf-8",
    )
    (server / "mobilegpt.py").write_text(
        "class MobileGPT:\n"
        "    def get_next_action(self):\n"
        "        if page_index != self.current_page_index:\n"
        "            self.memory.init_page_manager(page_index)\n"
        "            self.current_page_index = page_index\n"
        "            if self.subtask_status == Status.LEARN:\n"
        "                self.__finish_subtask()\n",
        encoding="utf-8",
    )
    (server / "utils" / "utils.py").write_text(
        'import os\n'
        'import re\n'
        'from openai import OpenAI\n'
        'def get_openai_embedding(text: str, model="text-embedding-3-small", **kwargs):\n'
        '    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))\n'
        '    text = text.replace("\\n", " ")\n'
        '    response = client.embeddings.create(input=[text], model=model, **kwargs)\n'
        '\n'
        '    return response.data[0].embedding\n'
        'def parse_completion_rate(completion_rate):\n'
        '    input_str = str(completion_rate).strip()\n'
        '    if input_str.endswith("%"):\n'
        '        return int(float(input_str[:-1]))\n'
        '    else:\n'
        '        value = float(input_str)\n'
        '    return int(value)\n',
        encoding="utf-8",
    )
    (server / "agents" / "param_fill_agent.py").write_text(
        'def call(query):\n'
        '    return query(model="gpt-4o")\n',
        encoding="utf-8",
    )
    memory = tmp_path / "prepared"
    (memory / "frozen_memory").mkdir(parents=True)

    result = prepare_mobilegpt_server(
        official_root=official,
        memory_root=memory,
        workspace=tmp_path / "workspace",
        embedding_model="GLM-Embedding-2",
        chat_model="GLM-5.1",
    )

    staged = Path(result["server_root"])
    assert (staged / "main.py").read_bytes() != (server / "main.py").read_bytes()
    assert "GLM-5.1" in (staged / "main.py").read_text(encoding="utf-8")
    assert (staged / "utils" / "utils.py").read_bytes() != (
        server / "utils" / "utils.py"
    ).read_bytes()
    assert "GLM-Embedding-2" in (
        staged / "utils" / "utils.py"
    ).read_text(encoding="utf-8")
    assert (staged / "memory" / "memory_manager.py").read_bytes() == (
        server / "memory" / "memory_manager.py"
    ).read_bytes()
    assert (staged / "mobilegpt.py").read_bytes() == (
        server / "mobilegpt.py"
    ).read_bytes()
    utils_source = (staged / "utils" / "utils.py").read_text(encoding="utf-8")
    assert "embedding_retry" in utils_source
    assert "range(1, 4)" in utils_source
    if "def query(" in (server / "utils" / "utils.py").read_text(
        encoding="utf-8"
    ):
        assert "omniflow_mobilegpt_glm_response_compat" in utils_source
        assert '"thinking": {"type": thinking_mode}' in utils_source
        assert "extra_body=request_extra_body" in utils_source
        assert '"512" if is_list else "2048"' in utils_source
    match = re.search(
        r"def parse_completion_rate\(completion_rate\).*?(?=\n\ndef |\Z)",
        utils_source,
        flags=re.DOTALL,
    )
    assert match is not None
    namespace: dict[str, object] = {"re": re}
    exec(match.group(0), namespace)
    assert namespace["parse_completion_rate"](
        "Starting the capture process, so 0% complete"
    ) == 0
    assert (staged / "agents" / "param_fill_agent.py").read_bytes() == (
        server / "agents" / "param_fill_agent.py"
    ).read_bytes()
    assert "gpt-4o" in (server / "main.py").read_text(encoding="utf-8")


def test_mobilegpt_forwarder_parameterizes_only_explicit_memory_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    official = tmp_path / "MobileGPT"
    server = official / "Server"
    (server / "memory").mkdir(parents=True)
    (server / "utils").mkdir()
    (server / "main.py").write_text("# official\n", encoding="utf-8")
    (server / "memory" / "memory_manager.py").write_text(
        "import os\n"
        "def select(candidates):\n"
        "    highest_similarity = candidates[0]\n"
        "    if highest_similarity > 0.95:\n"
        "        return 0\n",
        encoding="utf-8",
    )
    (server / "utils" / "utils.py").write_text(
        "import os\nimport json\n\n"
        "def write_omniflow_mobilegpt_event(event):\n"
        "    return None\n",
        encoding="utf-8",
    )
    memory = tmp_path / "prepared"
    (memory / "frozen_memory").mkdir(parents=True)
    monkeypatch.setenv("MOBILEGPT_MEMORY_SIMILARITY_THRESHOLD", "0.70")

    result = prepare_mobilegpt_server(
        official_root=official,
        memory_root=memory,
        workspace=tmp_path / "workspace",
    )

    staged = Path(result["server_root"])
    staged_memory_manager = (
        staged / "memory" / "memory_manager.py"
    ).read_text(encoding="utf-8")
    assert "MOBILEGPT_MEMORY_SIMILARITY_THRESHOLD" in staged_memory_manager
    assert "if highest_similarity > threshold:" in staged_memory_manager
    assert "if highest_similarity > 0.95:" in (
        server / "memory" / "memory_manager.py"
    ).read_text(encoding="utf-8")


def test_mobilegpt_forwarder_bridges_finish_to_official_client_frame(
    tmp_path: Path,
) -> None:
    official = tmp_path / "MobileGPT"
    server = official / "Server"
    (server / "memory").mkdir(parents=True)
    (server / "utils").mkdir()
    (server / "main.py").write_text("# official\n", encoding="utf-8")
    server_source = (
        "from utils.utils import log\n"
        "OMNIFLOW_INTERNAL_LAUNCH_ACTION = \"__omniflow_launch_package\"\n\n"
        "def _omniflow_send_action(client_socket, action):\n"
        "    action_name = (str(action.get(\"name\") or \"\").strip()\n"
        "                   if isinstance(action, dict) else \"\")\n"
        "    if (\n"
        "        isinstance(action, dict)\n"
        "        and str(action.get(\"name\") or \"\").strip() == OMNIFLOW_INTERNAL_LAUNCH_ACTION\n"
        "    ):\n"
        "        client_socket.send(\"launch\".encode())\n"
        "        return\n"
        "                target_package = app_agent.get_package_name(target_app)\n"
        "                task, is_new_task = task_agent.get_task(instruction)\n"
        "            elif message_type == 'A':\n"
    )
    (server / "server.py").write_text(server_source, encoding="utf-8")
    (server / "utils" / "utils.py").write_text(
        "import os\nimport json\n\ndef log(*args):\n    pass\n",
        encoding="utf-8",
    )
    memory = tmp_path / "prepared"
    (memory / "frozen_memory").mkdir(parents=True)

    result = prepare_mobilegpt_server(
        official_root=official,
        memory_root=memory,
        workspace=tmp_path / "workspace",
    )

    staged_source = (Path(result["server_root"]) / "server.py").read_text(
        encoding="utf-8"
    )
    assert 'action_name == "finish"' in staged_source
    assert 'client_socket.send("$$$$$".encode())' in staged_source
    assert "mobilegpt_forced_task_binding" not in staged_source
    assert "mobilegpt_target_package_direct" not in staged_source
    assert "mobilegpt_target_package_fallback" in staged_source
    assert "target_package = app_agent.get_package_name(target_app)" in staged_source
    assert "'MOBILEGPT_TARGET_PACKAGE', ''" in staged_source
    assert "mobilegpt_client_error_transport" in staged_source
    assert "elif message_type == 'E':" in staged_source
    assert "action_error += client_socket.recv(1)" in staged_source
    assert "task_agent.get_task(instruction)" in staged_source
    assert (server / "server.py").read_text(encoding="utf-8") == server_source


def test_mobilegpt_telemetry_imports_time_for_existing_hook(tmp_path: Path) -> None:
    utils_dir = tmp_path / "utils"
    utils_dir.mkdir()
    utils = utils_dir / "utils.py"
    utils.write_text(
        "import os\nimport json\n\n"
        "def write_omniflow_mobilegpt_event(event):\n"
        "    return None\n\n"
        "def get_openai_embedding(text, model=None, **kwargs):\n"
        "    time.sleep(1)\n",
        encoding="utf-8",
    )
    (tmp_path / "server.py").write_text("", encoding="utf-8")

    _configure_mobilegpt_telemetry(tmp_path)
    source = utils.read_text(encoding="utf-8")

    assert source.startswith("import time\n")
    compile(source, str(utils), "exec")


def test_mobilegpt_forwarder_does_not_patch_task_agent(
    tmp_path: Path,
) -> None:
    official = tmp_path / "MobileGPT"
    server = official / "Server"
    (server / "memory").mkdir(parents=True)
    (server / "agents").mkdir()
    (server / "main.py").write_text("# official\n", encoding="utf-8")
    (server / "agents" / "task_agent.py").write_text(
        "import os\n"
        "from agents.prompts import task_agent_prompt\n"
        "from utils.utils import query, log\n\n"
        "class TaskAgent:\n"
        "    def get_task(self, instruction):\n"
        "        known_tasks = []\n"
        "        response = query(messages=task_agent_prompt.get_prompts(instruction, known_tasks),\n"
        "                         model=os.getenv(\"TASK_AGENT_GPT_VERSION\"))\n"
        "        task = response[\"api\"]\n"
        "        return task, True\n",
        encoding="utf-8",
    )
    memory = tmp_path / "prepared"
    (memory / "frozen_memory").mkdir(parents=True)

    result = prepare_mobilegpt_server(
        official_root=official,
        memory_root=memory,
        workspace=tmp_path / "workspace",
        chat_model="GLM-4.6V",
    )

    staged_source = (
        Path(result["server_root"]) / "agents" / "task_agent.py"
    ).read_bytes()
    assert staged_source == (server / "agents" / "task_agent.py").read_bytes()


def test_mobilegpt_host_defaults_to_emulator_alias() -> None:
    assert resolve_mobilegpt_client_host(
        "",
        serial="emulator-5560",
        adb_path="adb",
    ) == "10.0.2.2"


def test_mobilegpt_explicit_host_wins_over_device_detection() -> None:
    assert resolve_mobilegpt_client_host(
        "192.168.1.155",
        serial="physical-device",
        adb_path="missing-adb",
    ) == "192.168.1.155"


def test_autodroid_memory_manifest_is_validated_without_conversion(
    tmp_path: Path,
) -> None:
    root = tmp_path / "androidworld_apps"
    root.mkdir()
    (root / "memory_manifest.json").write_text(
        '{"format":"autodroid-droidbot-memory-manifest-v1",'
        '"apps":[{"name":"audio"}],"device":{}}\n',
        encoding="utf-8",
    )

    result = validate_autodroid_memory_root(root)

    assert result["app_count"] == 1
    assert result["memory_root"] == str(root.resolve())
    assert len(result["manifest_sha256"]) == 64


def test_autodroid_task_app_name_maps_androidworld_alias() -> None:
    assert _autodroid_task_app_name(
        type("Task", (), {"app_names": ("simple sms messenger",)})()
    ) == "sms"


def test_autodroid_task_app_name_uses_first_declared_app_for_composite_task() -> None:
    assert _autodroid_task_app_name(
        type(
            "Task",
            (),
            {"app_names": ("pro expense", "simple gallery pro")},
        )()
    ) == "expense"
