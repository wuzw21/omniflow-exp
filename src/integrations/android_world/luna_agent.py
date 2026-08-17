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
        if self.done or self.step_index >= self.max_steps:
            return make_agent_result(
                True,
                {"summary": "luna_step_budget_reached", "step_index": self.step_index},
            )
        observation = self.host.observe(xml=True, screenshot=True, app_info=True)
        started = self.step_index
        record: dict[str, Any] = {
            "step_index": started,
            "task_name": self.task_name,
            "goal": self.goal,
            "observation": {
                "package_name": observation.package_name,
                "activity_name": observation.activity_name,
                "extra": _json_copy(observation.extra),
            },
            "observation_state": _json_copy(observation.extra.get("androidworld_state")),
        }
        try:
            call, metadata, step_usage = self._decide_with_codex(observation)
            self._merge_usage(step_usage)
            metadata["token_usage"] = _json_copy(step_usage)
            record["decision"] = {
                "tool": call.name,
                "arguments": _json_copy(call.arguments),
                "metadata": _json_copy(metadata),
            }
            action = Action(call.name, dict(call.arguments))
            if call.name == "finished":
                self._execute_answer(str(call.arguments.get("content") or ""))
                action_result = {"success": True}
                self.done = True
            else:
                result = self.host.act(action)
                action_result = _json_copy(result.to_dict())
                self.actions_executed += 1
            self._planner.record_action_result(action_result)
            record["action"] = action.to_dict()
            record["action_result"] = action_result
        except Exception as error:  # noqa: BLE001
            record["error"] = str(error) or type(error).__name__
            self.trace.append(record)
            self.step_index += 1
            self._last_result = _LunaRuntimeResult(
                detail=self._detail("planner_failed"),
                error=f"luna_harness_failed:{record['error']}",
                actions_executed=self.actions_executed,
                model_calls=1,
            )
            self.state["last_result"] = self._last_result
            return make_agent_result(False, {"summary": self._last_result.error, "step_index": self.step_index})

        self.trace.append(record)
        self.step_index += 1
        reason = "finished" if self.done else "step_completed"
        self._last_result = _LunaRuntimeResult(
            detail=self._detail(reason),
            actions_executed=self.actions_executed,
            model_calls=self.step_index,
        )
        self.state["last_result"] = self._last_result
        return make_agent_result(
            self.done or self.step_index >= self.max_steps,
            {
                "summary": reason,
                "step_index": self.step_index,
                "actions_executed": self.actions_executed,
                "done_reason": reason,
            },
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
