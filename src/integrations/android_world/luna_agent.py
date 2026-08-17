"""Direct Luna observe/act harness for the official AndroidWorld environment.

This adapter deliberately bypasses OmniFlow Functions and the upstream T3A/M3A
agents.  It exposes only AndroidWorld's native state and JSONAction boundary to
the Luna planner, while the launcher/recorder remains responsible for official
setup, screenshots, and validator accounting.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import os
import re
import socket
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from omniflow import Action, Observation
from omniflow.vlm.planner import VLMPlanner
from src.integrations.android_world.host import AndroidWorldHost, make_agent_result


@dataclass
class _LunaRuntimeResult:
    """Small launcher-compatible result carrying the complete decision trace."""

    detail: dict[str, Any]
    error: str | None = None
    actions_executed: int = 0
    model_calls: int = 0
    fallback_steps: int = 0
    function_id: str = ""

    @property
    def execution_summary(self) -> dict[str, Any]:
        return dict(self.detail.get("execution_summary") or {})


class _UsageSummaryProxy:
    """Expose cumulative usage after the planner drains its per-step tracker."""

    def __init__(self, owner: "LunaAndroidWorldHarness") -> None:
        self.owner = owner

    def get_usage_summary(self) -> dict[str, Any]:
        return self.owner._usage_summary()


class _AndroidWorldBridgeServer:
    """Forward MCP observe/act calls into the live AndroidWorld host."""

    def __init__(self, owner: "LunaAndroidWorldHarness", socket_path: Path) -> None:
        self.owner = owner
        self.socket_path = socket_path
        self._server: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stopped = threading.Event()

    def start(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(self.socket_path))
        self._server.listen(1)
        self._server.settimeout(0.5)
        self._thread = threading.Thread(target=self._serve, name="androidworld-bridge", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass

    def _serve(self) -> None:
        assert self._server is not None
        while not self._stopped.is_set():
            try:
                connection, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with connection:
                try:
                    raw = b""
                    while b"\n" not in raw:
                        chunk = connection.recv(65536)
                        if not chunk:
                            break
                        raw += chunk
                    request = json.loads(raw.split(b"\n", 1)[0].decode("utf-8"))
                    response = self._handle(request if isinstance(request, dict) else {})
                except Exception as error:  # noqa: BLE001
                    response = {"error": f"androidworld_bridge:{error}"}
                connection.sendall((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))

    def _handle(self, request: dict[str, Any]) -> dict[str, Any]:
        operation = str(request.get("op") or "")
        if operation == "observe":
            observation = self.owner.host.observe(xml=True, screenshot=True, app_info=True)
            value = {
                "xml": str(observation.xml or ""),
                "package_name": observation.package_name,
                "activity_name": observation.activity_name,
                "state": _json_copy(observation.extra.get("androidworld_state")),
                "display": _json_copy(observation.extra.get("display")),
            }
            self.owner._record_bridge_observation(value)
            return value
        if operation == "act":
            action = request.get("action")
            if not isinstance(action, dict):
                raise ValueError("action_object_required")
            tool = str(action.get("tool") or "")
            args = action.get("args") if isinstance(action.get("args"), dict) else {}
            if tool in {"click", "long_press", "input_text"}:
                invalid = []
                for key in ("x", "y"):
                    if args.get(key) is None:
                        continue
                    try:
                        value = float(args[key])
                    except (TypeError, ValueError):
                        invalid.append(f"{key}=not_numeric")
                        continue
                    if not 0.0 <= value <= 1000.0:
                        invalid.append(f"{key}={value}")
                if invalid:
                    error = (
                        "coordinate_contract_violation: x/y must be canonical 0..1000, "
                        "not screenshot pixels; convert XML bounds using display width/height. "
                        + ", ".join(invalid)
                    )
                    value = {
                        "action": _json_copy(action),
                        "action_result": {"success": False, "error": error},
                        "error": error,
                    }
                    self.owner._record_bridge_action(value)
                    return value
            result = self.owner.host.act(Action.from_value(action))
            value = {"action": _json_copy(action), "action_result": _json_copy(result.to_dict())}
            self.owner.actions_executed += 1
            self.owner._record_bridge_action(value)
            if str(action.get("tool") or "") == "finished":
                self.owner.done = True
                self.owner._whole_task_finished_event.set()
            return value
        raise ValueError(f"unknown_operation:{operation}")


class LunaAndroidWorldHarness:
    """One-step-per-call direct Luna harness over AndroidWorld get_state/act."""

    def __init__(
        self,
        *,
        env: Any,
        model: str,
        provider: str = "openai",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_steps: int = 20,
        hint: str = "",
        evidence_root: str | Path | None = None,
        adb_serial: str = "",
        adb_path: str = "",
    ) -> None:
        self.env = env
        self.name = "luna"
        self.host = AndroidWorldHost(
            env,
            adb_serial=adb_serial,
            adb_path=adb_path,
            evidence_root=evidence_root,
        )
        # Launcher diagnostics use the same transparent host/state seam as the
        # shared adapter; no task completion or validator logic is stored here.
        self.state: dict[str, Any] = {"last_result": None}
        self.host.state = self.state
        self.max_steps = max(1, int(max_steps))
        self.hint = str(hint or "").strip()
        self.task_name = ""
        self.goal = ""
        self.task_parameters: dict[str, Any] = {}
        self.source_runlog_path = str(
            os.environ.get("OMNIFLOW_LUNA_SOURCE_RUNLOG_PATH") or ""
        ).strip()
        self.source_index_path = str(
            os.environ.get("OMNIFLOW_LUNA_SOURCE_INDEX_PATH") or ""
        ).strip()
        self.source_reference = ""
        self.source_reference_steps = 0
        self.step_index = 0
        self.actions_executed = 0
        self.done = False
        self.trace: list[dict[str, Any]] = []
        self._usage_total: dict[str, int] = {
            key: 0
            for key in (
                "model_calls",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "responses_with_usage",
                "responses_without_usage",
                "failed_calls",
            )
        }
        self._last_result: _LunaRuntimeResult | None = None
        # One persistent Codex conversation owns the complete AndroidWorld
        # task. A fresh CLI invocation per observe/act turn made Luna
        # stateless and caused repeated actions without recovery.
        self._cli_temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._codex_session_id: str | None = None
        self._whole_task_started = False
        self._whole_task_finished_event = threading.Event()
        self._bridge: _AndroidWorldBridgeServer | None = None
        self._planner = VLMPlanner(
            model=str(model).strip() or "gpt-5.6-luna",
            provider=provider or "openai",
            api_key=api_key,
            base_url=base_url,
            timeout=float(timeout),
            max_steps=self.max_steps,
            step_skill_guidance=self.hint,
        )
        self._planner_api_key = api_key
        self._planner_base_url = base_url
        self._planner_provider = provider or "openai"
        self._omniflow_llm_usage_tracker = _UsageSummaryProxy(self)

    def reset(self, go_home: bool = False) -> None:
        self._close_cli_session()
        self.host.reset(go_home=go_home)
        self.step_index = 0
        self.actions_executed = 0
        self.done = False
        self._whole_task_started = False
        self._whole_task_finished_event.clear()
        self.trace = []
        self._last_result = None
        self.state["last_result"] = None
        for key in self._usage_total:
            self._usage_total[key] = 0
        self._planner = type(self._planner)(
            model=self._planner.model,
            provider=self._planner_provider,
            api_key=self._planner_api_key,
            base_url=self._planner_base_url,
            timeout=self._planner.timeout,
            max_steps=self.max_steps,
            step_skill_guidance=self.hint,
        )
        self._omniflow_llm_usage_tracker = _UsageSummaryProxy(self)

    def set_current_task(
        self,
        task_name: str,
        goal: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        if self.task_name and str(task_name or "").strip() != self.task_name:
            self._close_cli_session()
        self.task_name = str(task_name or "").strip()
        self.goal = str(goal or "").strip()
        values = dict(context or {}).get("task_parameters")
        self.task_parameters = dict(values) if isinstance(values, dict) else {}
        self._load_source_reference()

    def update_current_task_context(self, task: Any) -> dict[str, Any]:
        params = getattr(task, "params", {})
        return {"task_parameters": dict(params) if isinstance(params, dict) else {}}

    def set_max_steps(self, step_budget: int) -> None:
        self.max_steps = max(1, int(step_budget))
        self._planner.max_steps = self.max_steps

    def luna_diagnostics(self) -> dict[str, Any]:
        if self._last_result is None:
            return {
                "schema_version": "omniflow.androidworld.luna-harness.v1",
                "trace": _json_copy(self.trace),
            }
        return _json_copy(self._last_result.detail)

    def step(self, goal: str):
        self.goal = str(goal or self.goal or self.task_name).strip()
        if self.done or self._whole_task_started:
            return make_agent_result(
                True,
                {"summary": "luna_whole_task_already_finished", "step_index": self.step_index},
            )
        self._whole_task_started = True
        try:
            usage, metadata = self._run_complete_codex()
            self._merge_usage(usage)
            self.done = True
        except Exception as error:  # noqa: BLE001
            self._last_result = _LunaRuntimeResult(
                detail=self._detail("planner_failed"),
                error=f"luna_harness_failed:{error}",
                actions_executed=self.actions_executed,
                model_calls=1,
            )
            self.state["last_result"] = self._last_result
            self.step_index += 1
            return make_agent_result(False, {"summary": self._last_result.error, "step_index": self.step_index})

        self.step_index += 1
        reason = "whole_task_codex_finished"
        self._last_result = _LunaRuntimeResult(
            detail=self._detail(reason),
            actions_executed=self.actions_executed,
            model_calls=int(self._usage_total.get("model_calls") or 1),
        )
        self.state["last_result"] = self._last_result
        return make_agent_result(
            self.done or self.step_index >= self.max_steps,
            {
                "summary": reason,
                "step_index": self.step_index,
                "actions_executed": self.actions_executed,
                "done_reason": reason,
                "transport": metadata.get("transport"),
            },
        )

    def _record_bridge_observation(self, value: dict[str, Any]) -> None:
        self.trace.append({
            "event": "observe",
            "step_index": len(self.trace),
            "task_name": self.task_name,
            "goal": self.goal,
            "observation": value,
        })

    def _record_bridge_action(self, value: dict[str, Any]) -> None:
        self.trace.append({
            "event": "act",
            "step_index": len(self.trace),
            "task_name": self.task_name,
            "action": value.get("action"),
            "action_result": value.get("action_result"),
        })

    def _run_complete_codex(self) -> tuple[dict[str, int], dict[str, Any]]:
        temp_dir = self._ensure_cli_session()
        bridge = _AndroidWorldBridgeServer(self, temp_dir / "androidworld-bridge.sock")
        self._bridge = bridge
        bridge.start()
        output_path = temp_dir / "whole_task_message.txt"
        codex_home = temp_dir / "codex-home"
        codex_home.mkdir(parents=True, exist_ok=True)
        repo_root = Path(__file__).resolve().parents[3]
        codex_home.joinpath("config.toml").write_text(
            "\n".join((
                f'model = "{self._planner.model}"',
                'model_provider = "omnimind"',
                '[features]', 'plugins = false',
                '[model_providers.omnimind]', 'name = "omnimind"',
                f'base_url = "{os.environ.get("OMNIFLOW_LUNA_CODEX_BASE_URL", "http://cloud.omnimind.com.cn/v1")}"',
                'wire_api = "responses"', 'requires_openai_auth = true', 'env_key = "OMNIMIND_API_KEY"',
                '[mcp_servers.androidworld]',
                f'command = "{sys.executable}"',
                'args = ["-m", "src.integrations.android_world.luna_mcp_bridge"]',
                '[mcp_servers.androidworld.env]',
                f'OMNIFLOW_ANDROIDWORLD_BRIDGE_SOCKET = "{bridge.socket_path}"',
            )) + "\n", encoding="utf-8"
        )
        command = [
            "codex", "exec", "--skip-git-repo-check", "--sandbox", "danger-full-access",
            "--color", "never", "--model", self._planner.model, "-C", str(repo_root),
            "-o", str(output_path), "--json",
        ]
        try:
            timeout_seconds = max(
                float(self._planner.timeout),
                float(os.environ.get("OMNIFLOW_LUNA_TASK_TIMEOUT", "900")),
            )
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ, "HOME": str(codex_home), "CODEX_HOME": str(codex_home)},
                cwd=str(repo_root),
            )
            assert process.stdin is not None
            process.stdin.write(self._whole_task_prompt())
            process.stdin.close()
            deadline = time.monotonic() + timeout_seconds
            stopped_after_finished = False
            while process.poll() is None:
                if self._whole_task_finished_event.wait(timeout=0.5):
                    stopped_after_finished = True
                    process.terminate()
                    break
                if time.monotonic() >= deadline:
                    process.terminate()
                    raise TimeoutError(f"luna_codex_whole_task_timeout:{int(timeout_seconds)}s")
            try:
                stdout, stderr = process.communicate(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
            returncode = int(process.returncode or 0)
            raw_response = output_path.read_text(encoding="utf-8", errors="replace") if output_path.is_file() else ""
            usage = self._cli_usage(stdout)
            if (returncode != 0 and not stopped_after_finished) or (not raw_response.strip() and not stopped_after_finished):
                detail = (stderr or stdout or "codex_cli_failed").strip()
                raise RuntimeError(f"luna_codex_whole_task_failed:{detail[-4000:]}")
            return usage, {
                "transport": "codex_cli_whole_task_mcp",
                "raw_response": raw_response,
                "cli_returncode": returncode,
                "stopped_after_finished": stopped_after_finished,
                "bridge_tools": ["androidworld_observe", "androidworld_act"],
            }
        finally:
            bridge.stop()
            self._bridge = None

    def _whole_task_prompt(self) -> str:
        params = json.dumps(self.task_parameters, ensure_ascii=False, sort_keys=True, default=str)
        return (
            "You are the sole autonomous agent for one complete AndroidWorld task. "
            "Execute the task all the way to its requested end state yourself. "
            "Use androidworld_observe and androidworld_act continuously: observe "
            "first, reason over the screenshot and native accessibility state, act "
            "once, observe again, and recover whenever the screen differs. Do not "
            "merely describe actions to the parent and do not stop after a partial "
            "plan. There is no outer per-step decision loop or fixed action-count "
            "limit; continue until the task is actually complete. Use only the current "
            "device state and current task parameters; never replay source coordinates, "
            "stale node IDs, or old input values. Call the tools rather than using ADB "
            "or a fixed script. For click, long_press, and input_text actions, x/y "
            "must be canonical coordinates from 0 to 1000. The accessibility XML "
            "bounds and screenshot pixels use the physical display size; convert "
            "pixel_x to 1000*pixel_x/display_width and pixel_y to "
            "1000*pixel_y/display_height before calling androidworld_act. Never send "
            "physical pixel coordinates or values above 1000. If the tool reports a "
            "coordinate contract error, observe again and correct the conversion. "
            "Only finish after the final requested state is visible.\n\n"
            f"Task: {self.goal}\nTask parameters: {params}\n\n{self.source_reference}\n"
            "Begin by calling androidworld_observe now, then continue until completion."
        )

    def _decide_with_codex(
        self,
        observation: Observation,
    ) -> tuple[Any, dict[str, Any], dict[str, int]]:
        """Ask one persistent Codex/Luna session for the next native action."""
        pixels = observation.extra.get("androidworld_state", {}).get("pixels", {})
        screenshot = str(pixels.get("path") or "") if isinstance(pixels, dict) else ""
        prompt = self._cli_prompt(observation)
        temp_dir = self._ensure_cli_session()
        output_path = temp_dir / f"last_message_{self.step_index:04d}.txt"
        codex_home = temp_dir / "codex-home"
        try:
            codex_home.mkdir(parents=True, exist_ok=True)
            (codex_home / "config.toml").write_text(
                "\n".join(
                    (
                        f'model = "{self._planner.model}"',
                        'model_provider = "omnimind"',
                        "[features]",
                        "plugins = false",
                        "[model_providers.omnimind]",
                        'name = "omnimind"',
                        f'base_url = "{os.environ.get("OMNIFLOW_LUNA_CODEX_BASE_URL", "http://cloud.omnimind.com.cn/v1")}"',
                        'wire_api = "responses"',
                        "requires_openai_auth = true",
                        'env_key = "OMNIMIND_API_KEY"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            if self._codex_session_id:
                command = [
                    "codex", "exec", "resume", self._codex_session_id,
                    "--skip-git-repo-check", "--model", self._planner.model,
                    "-o", str(output_path), "--json",
                ]
            else:
                command = [
                    "codex", "exec", "--skip-git-repo-check", "--sandbox",
                    "read-only", "--color", "never", "--model",
                    self._planner.model, "-o", str(output_path), "--json",
                ]
            if screenshot and Path(screenshot).is_file():
                command.extend(("-i", screenshot))
            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=float(self._planner.timeout),
                check=False,
                env={
                    **os.environ,
                    "HOME": str(codex_home),
                    "CODEX_HOME": str(codex_home),
                },
            )
            raw_response = (
                output_path.read_text(encoding="utf-8", errors="replace")
                if output_path.is_file() else ""
            )
            usage = self._cli_usage(completed.stdout)
            if completed.returncode != 0 or not raw_response.strip():
                detail = (completed.stderr or completed.stdout or "codex_cli_failed").strip()
                raise RuntimeError(f"luna_codex_cli_failed:{detail[-2000:]}")
            session_id = self._cli_session_id_from_events(completed.stdout)
            if session_id:
                self._codex_session_id = session_id
            payload = _parse_cli_action(raw_response)
            from omniflow.core.model import ToolCall

            call = ToolCall(str(payload["action"]), dict(payload.get("args") or {}))
            metadata = {
                "reasoning": str(payload.get("reasoning") or "").strip(),
                "raw_response": raw_response,
                "transport": "codex_cli_persistent_session",
                "cli_returncode": completed.returncode,
                "codex_session_id": self._codex_session_id,
            }
            return call, metadata, usage
        except Exception:
            self._close_cli_session()
            raise

    def _ensure_cli_session(self) -> Path:
        if self._cli_temp_dir is None:
            self._cli_temp_dir = tempfile.TemporaryDirectory(prefix="luna-cli-session-")
        return Path(self._cli_temp_dir.name)

    def _close_cli_session(self) -> None:
        self._codex_session_id = None
        if self._cli_temp_dir is not None:
            try:
                self._cli_temp_dir.cleanup()
            finally:
                self._cli_temp_dir = None

    @staticmethod
    def _cli_session_id_from_events(events: str) -> str | None:
        for line in str(events or "").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "thread.started":
                continue
            value = event.get("thread_id") or event.get("id")
            if value:
                return str(value)
        return None

    def _load_source_reference(self) -> None:
        path = Path(self.source_runlog_path).expanduser() if self.source_runlog_path else None
        if (path is None or not path.is_file()) and self.source_index_path:
            try:
                index = json.loads(Path(self.source_index_path).expanduser().read_text(encoding="utf-8"))
                row = index.get(self.task_name) if isinstance(index, dict) else None
                candidate = row.get("retained_source_run_log") if isinstance(row, dict) else ""
                if candidate:
                    path = Path(str(candidate)).expanduser()
                    self.source_runlog_path = str(path)
            except Exception:
                path = None
        self.source_reference = ""
        self.source_reference_steps = 0
        if path is None or not path.is_file():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, dict) or payload.get("success") is not True:
            return
        steps = payload.get("steps")
        if not isinstance(steps, list):
            return
        rendered = []
        for index, step in enumerate(steps[:32], start=1):
            line = self._render_source_step(index, step)
            if line:
                rendered.append(line)
        if rendered:
            self.source_reference_steps = len(rendered)
            self.source_reference = (
                "Successful source RunLog semantic reference (guidance only):\n"
                + "\n".join(rendered)
                + "\nUse this as a plan template, not replay. Never reuse source "
                  "coordinates, node IDs, or old input values; match the current "
                  "screenshot/accessibility tree and current task parameters."
            )

    @classmethod
    def _render_source_step(cls, index: int, step: Any) -> str:
        action = step.get("action") if isinstance(step, dict) else None
        if not isinstance(action, dict):
            return ""
        kind = str(action.get("action_type") or "").strip().lower()
        if kind == "open_app":
            app = str(action.get("app_name") or action.get("package_name") or "").strip()
            return f"{index}. Open the matching app{(' ' + app) if app else ''}."
        if kind in {"swipe", "scroll"}:
            return f"{index}. Swipe/scroll {str(action.get('direction') or 'as needed')}."
        if kind in {"press_key", "key_event"}:
            return f"{index}. Press {str(action.get('key') or 'the relevant key')}."
        if kind in {"input_text", "type_text", "set_text"}:
            label = cls._source_node_labels(step)
            return f"{index}. Enter the current task value into {label or 'the matching text field'}."
        if kind in {"click", "tap", "long_press"}:
            label = cls._source_node_labels(step)
            verb = "Long-press" if kind == "long_press" else "Click"
            return f"{index}. {verb} {label or 'the matching semantic UI control'}."
        if kind in {"answer", "finished"}:
            return f"{index}. Finish only after the requested end state is visible."
        return f"{index}. Perform the matching {kind} action if exposed by the current UI."

    @staticmethod
    def _source_node_labels(step: dict[str, Any]) -> str:
        action = step.get("action") or {}
        try:
            x, y = float(action.get("x")), float(action.get("y"))
        except (TypeError, ValueError):
            return ""
        observation = step.get("observation") or {}
        forest = observation.get("forest") if isinstance(observation, dict) else ""
        if not isinstance(forest, str) or not forest.strip():
            return ""
        try:
            root = ET.fromstring(forest)
        except ET.ParseError:
            return ""
        matches = []
        for node in root.iter():
            bounds = str(node.attrib.get("bounds") or "")
            nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", bounds)]
            if len(nums) != 4:
                continue
            left, top, right, bottom = nums
            if left <= x <= right and top <= y <= bottom:
                matches.append(((right-left) * (bottom-top), node))
        if not matches:
            return ""
        node = min(matches, key=lambda item: item[0])[1]
        labels = []
        for key in ("content-desc", "text", "resource-id", "class"):
            value = str(node.attrib.get(key) or "").strip()
            if value and value not in labels:
                labels.append(value)
        return "UI control " + " / ".join(labels[:3]) if labels else ""

    def _cli_prompt(self, observation: Observation) -> str:
        xml = str(observation.xml or "")
        if len(xml) > 30000:
            xml = xml[:30000] + "\n[xml truncated]"
        hint = f"\nGuidance:\n{self.hint}" if self.hint else ""
        task_parameters = json.dumps(
            self.task_parameters, ensure_ascii=False, sort_keys=True, default=str
        )
        if self.trace:
            history_lines = []
            for item in self.trace:
                decision = item.get("decision") or {}
                metadata = decision.get("metadata") or {}
                history_lines.append(
                    json.dumps(
                        {
                            "step": item.get("step_index"),
                            "screen": {
                                "package": (item.get("observation") or {}).get("package_name"),
                                "activity": (item.get("observation") or {}).get("activity_name"),
                            },
                            "action": item.get("action") or {
                                "name": decision.get("tool"),
                                "arguments": decision.get("arguments"),
                            },
                            "action_result": item.get("action_result"),
                            "reasoning": metadata.get("reasoning"),
                            "error": item.get("error"),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            history = "\n".join(history_lines)
        else:
            history = "(no action has been executed yet)"
        return (
            "You are Luna, the decision model executing one complete AndroidWorld task. "
            "This is a persistent conversation: previous turns, screenshots, actions, "
            "and action results remain available. Re-plan from the global goal after "
            "every result; do not blindly repeat an action that did not change the "
            "screen. If an action failed or the UI differs, recover using the current "
            "screenshot/XML. Do not call tools, run shell commands, or modify files. "
            "Inspect the attached current screenshot and accessibility XML, then choose "
            "exactly one next AndroidWorld action. Return ONLY one JSON object with keys "
            "action, args, reasoning. Allowed action values and argument shapes: "
            "click(target_description,x,y), input_text(target_description,text,x,y), "
            "swipe(direction), open_app(package_name), press_key(key), "
            "finished(content). Coordinates x/y are canonical 0-1000 values (not "
            "pixels). Only return finished when the requested end state has actually "
            "been achieved.\n\n"
            f"Task: {self.goal}\nTask parameters: {task_parameters}\n"
            f"Complete action history:\n{history}\n\n"
            f"{self.source_reference}\n"
            f"Current accessibility XML:\n{xml}\n{hint}"
        )

    @staticmethod
    def _cli_usage(events: str) -> dict[str, int]:
        usage = {key: 0 for key in (
            "model_calls", "prompt_tokens", "completion_tokens", "total_tokens",
            "responses_with_usage", "responses_without_usage", "failed_calls",
        )}
        usage["model_calls"] = 1
        for line in str(events or "").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "turn.completed":
                continue
            values = event.get("usage") if isinstance(event.get("usage"), dict) else {}
            usage["prompt_tokens"] = int(values.get("input_tokens") or 0)
            usage["completion_tokens"] = int(values.get("output_tokens") or 0)
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            usage["responses_with_usage"] = 1
        if usage["responses_with_usage"] == 0:
            usage["responses_without_usage"] = 1
        usage["failed_calls"] = 0
        return usage

    def _execute_answer(self, content: str) -> None:
        module = importlib.import_module("android_world.env.json_action")
        answer = getattr(module, "ANSWER", "answer")
        action_class = getattr(module, "JSONAction")
        self.env.execute_action(action_class(action_type=answer, text=content))

    def _detail(self, reason: str) -> dict[str, Any]:
        usage = self._usage_summary()
        return {
            "done_reason": reason,
            "trace": _json_copy(self.trace),
            "luna_harness": {
                "schema_version": "omniflow.androidworld.luna-harness.v1",
                "model": self._planner.model,
                "task_name": self.task_name,
                "steps": len(self.trace),
                "screenshots_per_step": True,
                "source_reference_steps": self.source_reference_steps,
                "source_runlog_path": self.source_runlog_path,
            },
            "llm_usage": _json_copy(usage),
            "execution_summary": {
                "model_calls": int(usage.get("model_calls") or 0),
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
                "token_usage_status": usage.get("token_usage_status"),
            },
        }

    def _merge_usage(self, usage: dict[str, Any]) -> None:
        for key in self._usage_total:
            self._usage_total[key] += int(usage.get(key) or 0)

    def _usage_summary(self) -> dict[str, Any]:
        usage = {
            "component": "planner",
            "model": self._planner.model,
            **self._usage_total,
        }
        calls = usage["model_calls"]
        responses = usage["responses_with_usage"]
        usage["token_usage_status"] = (
            "not_applicable" if calls == 0 else
            "tracked" if responses == calls else
            "partial" if responses > 0 else "unavailable"
        )
        return usage


def build_luna_agent(context: Any) -> LunaAndroidWorldHarness:
    from omniflow.vlm.model_config import resolve_openai_compatible_config

    model = str(
        getattr(context, "planner_model", "")
        or os.environ.get("OMNIFLOW_LUNA_MODEL")
        or "gpt-5.6-luna"
    ).strip()
    profile = str(
        getattr(context, "model_endpoint_profile", "")
        or os.environ.get("OMNIFLOW_LUNA_MODEL_ENDPOINT_PROFILE")
        or "openai"
    ).strip()
    api_key, base_url = resolve_openai_compatible_config(profile=profile)
    return LunaAndroidWorldHarness(
        env=context.env,
        model=model,
        provider="openai",
        api_key=api_key,
        base_url=base_url,
        timeout=float(getattr(context, "planner_timeout_sec", None) or 120.0),
        max_steps=int(getattr(context, "max_steps", 20) or 20),
        hint=str(getattr(context, "step_skill_guidance", "") or ""),
        evidence_root=getattr(context, "evidence_root", "") or None,
        adb_serial=str(getattr(context, "adb_serial", "") or ""),
        adb_path=str(getattr(context, "adb_path", "") or ""),
    )


def _run_async(awaitable: Any) -> Any:
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(awaitable)
    finally:
        loop.close()


def _parse_cli_action(text: str) -> dict[str, Any]:
    candidate = str(text or "").strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("luna_action_json_missing")
        value = json.loads(candidate[start : end + 1])
    if not isinstance(value, dict) or str(value.get("action") or "") not in {
        "click", "input_text", "swipe", "open_app", "press_key", "finished",
    }:
        raise ValueError("luna_action_schema_invalid")
    args = value.get("args")
    if not isinstance(args, dict):
        raise ValueError("luna_action_args_invalid")
    return value


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


__all__ = ["LunaAndroidWorldHarness", "build_luna_agent"]
