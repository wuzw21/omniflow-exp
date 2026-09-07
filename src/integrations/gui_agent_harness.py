"""Task harnesses share one runtime; adapters only launch and read their host."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Protocol

from omniflow.core.config import Experiment
from omniflow.core.model import RunResult
from omniflow.runtime.protocol import review_completion
from src.integrations.gui_agent_mcp import runtime_socket
from src.integrations.gui_agent_tools import GuiAgentToolRuntime


@dataclass(frozen=True)
class HarnessContext:
    flow: Any
    goal: str
    max_steps: int
    evidence_root: Path
    timeout_seconds: float = 600


class TaskHarness(Protocol):
    async def arun(self, context: HarnessContext) -> RunResult: ...


class BuiltinHarness:
    async def arun(self, context: HarnessContext) -> RunResult:
        return await context.flow.arun(context.goal, experiment=Experiment(name="androidworld"))


def build_harness(name: str = "builtin") -> TaskHarness:
    """Resolve explicit deployment configuration, never model-provided code."""
    if name == "builtin":
        return BuiltinHarness()
    if name in {"codex", "claude"}:
        return CliHarness(name)
    module, separator, attribute = name.partition(":")
    if not separator or not module or not attribute:
        raise ValueError("unknown_harness:use_builtin_codex_claude_or_module:factory")
    harness = getattr(importlib.import_module(module), attribute)()
    if not callable(getattr(harness, "arun", None)):
        raise TypeError("harness_factory_must_return_arun")
    return harness


def _command(name: str, socket: str, workspace: Path) -> list[str]:
    server = {"command": sys.executable,
              "args": ["-m", "src.integrations.gui_agent_mcp", "--connect", socket],
              "env": {"PYTHONPATH": str(Path(__file__).resolve().parents[2])}}
    if name == "codex":
        config = ("mcp_servers.omniflow={command=" + json.dumps(server["command"])
                  + ",args=" + json.dumps(server["args"])
                  + ",env={PYTHONPATH=" + json.dumps(server["env"]["PYTHONPATH"])
                  + "},startup_timeout_sec=60,tool_timeout_sec=600}")
        return ["codex", "exec", "--ephemeral", "--ignore-user-config", "--skip-git-repo-check",
                "--sandbox", "read-only", "--json", "-C", str(workspace),
                "-c", 'approval_policy="on-request"', "-c", 'approvals_reviewer="auto_review"',
                "-c", config, "-"]
    config_path = workspace / "mcp.json"
    config_path.write_text(json.dumps({"mcpServers": {"omniflow": server}}))
    # Keep the host's configured authentication/model; do not bypass permissions.
    return ["claude", "-p", "--no-session-persistence", "--strict-mcp-config", "--mcp-config",
            str(config_path), "--tools", "", "--allowedTools",
            "mcp__omniflow__omniflow_recall,mcp__omniflow__omniflow_execute",
            "--disable-slash-commands", "--output-format", "stream-json", "--verbose"]


def _host_report(name: str, path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"name": name, "model": None, "model_calls": None,
                              "prompt_tokens": 0, "completion_tokens": 0, "error": None}
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if name == "claude":
            if event.get("type") == "system" and event.get("subtype") == "init":
                report["model"] = event.get("model")
            if event.get("type") == "result":
                usage = event.get("usage") or {}
                report["prompt_tokens"] = sum(int(usage.get(k) or 0) for k in
                    ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
                report["completion_tokens"] = int(usage.get("output_tokens") or 0)
                report["host_turns"] = event.get("num_turns")
                if event.get("is_error"):
                    report["error"] = "host_result_error:" + str(event.get("subtype") or "unknown")
        elif event.get("type") == "turn.completed":
            usage = event.get("usage") or {}
            report["prompt_tokens"] += int(usage.get("input_tokens") or 0)
            report["completion_tokens"] += int(usage.get("output_tokens") or 0)
        elif event.get("type") == "turn.failed":
            report["error"] = "host_turn_failed"
    # A turn or tool count is not an API request count; never infer model calls.
    report["total_tokens"] = report["prompt_tokens"] + report["completion_tokens"]
    report["token_usage_status"] = "reported" if report["total_tokens"] else "unavailable"
    return report


@dataclass(frozen=True)
class CliHarness:
    name: str

    async def arun(self, context: HarnessContext) -> RunResult:
        root = context.evidence_root / "harness"
        root.mkdir(parents=True, exist_ok=True)
        runtime = GuiAgentToolRuntime(host=context.flow.host, flow=context.flow,
            timeout_seconds=context.timeout_seconds, max_service_calls=context.max_steps,
            allow_task_switch=False)
        initial_gate = await review_completion(context.flow.completion_checker)
        skill = Path(__file__).resolve().parents[2] / "skills/omniflow-gui/SKILL.md"
        if not skill.is_file():
            for prefix in (Path(__file__).resolve().parents[2], Path(sys.prefix)):
                candidate = prefix / "share/omniflow/skills/omniflow-gui/SKILL.md"
                if candidate.is_file():
                    skill = candidate
                    break
        prompt = ("Perform this Android task through the configured OmniFlow runtime: " + context.goal
            + "\nUse only omniflow_recall and omniflow_execute. The Android environment, Memory, "
              "device, recording and official completion checker are already owned by this episode. "
              "Do not use shell, other device tools, file edits, subagents, or launch another runtime. "
              "Use the same task_id throughout this episode. "
              "Stop when feedback.control.next is stop. Verified task success is authoritative. "
              "If no Function fits or execution fails and you cannot recover with the available tools, "
              "report the limitation and stop. Never invent success or replay an unknown effect.\n"
            + "\nCurrent official completion check after task initialization: "
            + json.dumps(initial_gate) + "\n" + skill.read_text())
        (root / "prompt.txt").write_text(prompt)
        process = None
        error = None
        environment = dict(os.environ)
        # These credentials belong to the benchmark provider, not to either CLI.
        # Each host uses its own login/configuration (Claude's own API key stays).
        for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "LLMTHU_API_KEY"):
            environment.pop(key, None)
        with tempfile.TemporaryDirectory(prefix="of-host-") as directory:
            async with runtime_socket(runtime) as socket:
                try:
                    with (root / "events.jsonl").open("wb") as output, (root / "stderr.log").open("wb") as stderr:
                        process = await asyncio.create_subprocess_exec(
                            *_command(self.name, socket, Path(directory)), cwd=directory,
                            env=environment, stdin=asyncio.subprocess.PIPE, stdout=output, stderr=stderr)
                        await asyncio.wait_for(process.communicate(prompt.encode()), context.timeout_seconds)
                    if process.returncode:
                        error = f"host_exit:{process.returncode}"
                except asyncio.TimeoutError:
                    error = "deadline_exceeded"
                except asyncio.CancelledError:
                    error = "cancelled"
                except OSError as exc:
                    error = f"host_launch_failed:{exc}"
                finally:
                    if process is not None and process.returncode is None:
                        runtime.cancel()
                        process.terminate()
                        try:
                            await asyncio.wait_for(process.wait(), 5)
                        except asyncio.TimeoutError:
                            process.kill()
                            await process.wait()
            result = runtime.execution_result()
            if not error and not result.detail.get("completion_gate"):
                gate = await review_completion(context.flow.completion_checker)
                if gate:
                    result = replace(result, detail={**result.detail, "completion_gate": gate})
        report = _host_report(self.name, root / "events.jsonl")
        error = error or report["error"] or result.error
        gate = result.detail.get("completion_gate") or {}
        verified = gate.get("status") == "verified"
        if not verified and not error:
            error = "completion_verifier_error" if gate.get("status") == "error" else "host_task_incomplete"
        reason = "function_completed_verified" if verified and not error else error
        detail = {**result.detail, "done_reason": reason, "harness": report,
                  "llm_usage": report, "planner_steps": result.detail.get("service_calls", 0)}
        (root / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        return replace(result, success=verified and not error, error=error, detail=detail)


__all__ = ["TaskHarness", "HarnessContext", "build_harness"]
