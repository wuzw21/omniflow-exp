"""Google ARTEMIS as an external online baseline for AndroidWorld episodes.

ARTEMIS (github.com/google/artemis, Apache-2.0) is a complete agent application:
its own Planner / Operator / Explorer graph, its own adb driver, its own
accessibility helper APK, and its own completion logic.  It does not call any
OmniFlow API, so it is an *external harness*, not a same-harness reuse variant.

It is wired through the existing ``--harness module:factory`` seam so the main
table's code path stays untouched:

    bash scripts/exp/run_androidworld.sh run \\
      --method omniflow --harness src.integrations.artemis_harness:factory ...

What this adapter does and does not claim:

* It runs ARTEMIS's own loop and reports ARTEMIS's own success/failure facts.
* It never consults the AndroidWorld validator in the loop, exactly like the
  builtin harness with ``--completion-oracle off``.  The official validator runs
  once after the episode and remains the only definition of task success, so an
  ARTEMIS row is comparable to an OmniFlow row on *task success* only.  It is
  not comparable on reuse rate, Function calls, or Transfer quality, because
  ARTEMIS has no such artifacts.
* ARTEMIS's published ``99%+`` figure comes from its own Checker exit review, not
  from the official validator.  Do not copy that number into a main-table row.

The adapter has been smoke-validated on the 4090 AndroidWorld AVD.  Physical
device acceptance remains 待真机验证.  DashScope Pro additionally needs the
external ARTEMIS checkout's non-Google structured-output method set to
``function_calling``.  That third-party patch is kept outside this repository.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

from omniflow.core.model import RunResult
from src.integrations.gui_agent_harness import HarnessContext

# ARTEMIS installs its own accessibility helper and drives the device over adb.
# It therefore needs to own the device for the whole episode; the shared OOB
# host must not dispatch actions concurrently.  ``requires_builtin_planner``
# stays True so the launcher never builds an OmniFlow Planner it would not use.
ARTEMIS_ROOT_ENV = "OMNIFLOW_ARTEMIS_ROOT"
ARTEMIS_PROFILE_ENV = "OMNIFLOW_ARTEMIS_PROFILE"
DEFAULT_ARTEMIS_PROFILE = "flash"

# ARTEMIS renders one markdown/jsonc config for every sub-agent.  Pin every node
# to the formal OpenAI-compatible endpoint so a baseline row is model-matched to
# the OmniFlow row it is compared against.  Nodes that can only run on Gemini
# (the embodied-reasoning object detector and the Gemini Files video analyzer)
# are disabled rather than silently falling back to a different provider.
_ARTEMIS_CONFIG_TEMPLATE = {
    "default": {
        "provider": "openai",
        "model": "{model}",
    },
    "nodes": {
        "object_detector": {"provider": "openai", "model": "{model}"},
        "hopper": {"provider": "openai", "model": "{model}"},
    },
    "video_analyzer": {"enabled": False},
}


@dataclass(frozen=True)
class ArtemisHarness:
    """One ARTEMIS CLI process per episode, sealed to the episode's device."""

    requires_builtin_planner = True

    profile: str = DEFAULT_ARTEMIS_PROFILE

    async def arun(self, context: HarnessContext) -> RunResult:
        try:
            root = _artemis_root()
        except ValueError as error:
            # A misconfigured external root is a deployment fact, not a task
            # result.  Report it as this episode's failure instead of aborting
            # a batch run of the rest of the rows.
            return _failure(str(error))
        serial = _device_serial(context.flow)
        if not serial:
            return _failure("artemis_device_serial_required")
        evidence = context.evidence_root / "artemis"
        evidence.mkdir(parents=True, exist_ok=True)
        config_path = _write_config(evidence, root)
        command = [
            sys.executable,
            "-m",
            "artemis",
            "run",
            context.goal,
            "--profile",
            self.profile,
            "--device-serial",
            serial,
            "--test-name",
            "omniflow_episode",
            "--traces-path",
            str(evidence),
            # Without --standalone ARTEMIS submits the task to a shared resident
            # daemon, which would queue behind other work and take the device out
            # of this episode's control.  One episode must own one process.
            "--standalone",
        ]
        environment = _child_environment(root, config_path)
        stdout_path = evidence / "stdout.log"
        stderr_path = evidence / "stderr.log"
        error = None
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(root),
                env=environment,
                stdout=stdout,
                stderr=stderr,
            )
            try:
                await asyncio.wait_for(
                    process.wait(), float(context.timeout_seconds or 600)
                )
            except asyncio.TimeoutError:
                error = "deadline_exceeded"
            except asyncio.CancelledError:
                error = "cancelled"
            finally:
                if process.returncode is None:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), 10)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
        if error is None and process.returncode:
            error = f"artemis_exit:{process.returncode}"
        outcome = _read_outcome(evidence)
        detail = {
            "harness": "artemis",
            "artemis_profile": self.profile,
            "artemis_device_serial": serial,
            "artemis_outcome": outcome,
            "artemis_stdout": str(stdout_path),
            "artemis_stderr": str(stderr_path),
            # ARTEMIS reports no per-request API count.  Never infer one from
            # turn or tool counts, and never report 0 as if it were measured.
            "llm_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "token_usage_status": "unavailable",
                "model": outcome.get("model"),
            },
            "artemis_task_status": outcome.get("task_status"),
            # The episode wrapper only emits the official completion signal for
            # this exact reason, and the validator needs that signal.  ARTEMIS's
            # own task_status therefore stays in its own field and never becomes
            # done_reason.
            "done_reason": error or "finished",
            "planner_steps": int(outcome.get("step_count") or 0),
        }
        if error is not None:
            return RunResult(
                False,
                model_calls=0,
                error=error,
                final_state=None,
                detail=detail,
            )
        # ARTEMIS's own status is a host claim, not the benchmark verdict.  The
        # official validator decides success after the episode, exactly as it
        # does for the builtin harness; a clean ARTEMIS run is therefore
        # "not failed", not "succeeded".
        return RunResult(
            True,
            actions_executed=int(outcome.get("step_count") or 0),
            model_calls=0,
            detail=detail,
        )


def factory() -> ArtemisHarness:
    """Entry point for ``--harness src.integrations.artemis_harness:factory``."""

    return ArtemisHarness(profile=os.environ.get(ARTEMIS_PROFILE_ENV, DEFAULT_ARTEMIS_PROFILE))


def _artemis_root() -> Path:
    configured = str(os.environ.get(ARTEMIS_ROOT_ENV) or "").strip()
    if not configured:
        raise ValueError(f"artemis_root_required:set {ARTEMIS_ROOT_ENV}")
    root = Path(configured).expanduser().resolve()
    if not (root / "artemis").is_dir():
        raise ValueError(f"artemis_root_invalid:{root}")
    return root


def _device_serial(flow: Any) -> str:
    for candidate in (
        getattr(getattr(flow, "host", None), "adb_serial", ""),
        os.environ.get("ANDROID_SERIAL", ""),
    ):
        value = str(candidate or "").strip()
        if value:
            return value
    return ""


def _write_config(evidence: Path, root: Path) -> Path:
    """Render the per-episode ARTEMIS config with the formal model pinned."""

    from src.experiment.protocol import FORMAL_MODEL

    payload = json.loads(
        json.dumps(_ARTEMIS_CONFIG_TEMPLATE).replace("{model}", FORMAL_MODEL)
    )
    config_path = evidence / "artemis.jsonc"
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return config_path


def _child_environment(root: Path, config_path: Path) -> dict[str, str]:
    from src.experiment.protocol import FORMAL_MODEL, FORMAL_MODEL_BASE_URL

    environment = dict(os.environ)
    endpoint = FORMAL_MODEL_BASE_URL.rstrip("/")
    api_key = str(
        environment.get("LLMTHU_API_KEY") or environment.get("OPENAI_API_KEY") or ""
    )
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(
                value for value in (str(root), environment.get("PYTHONPATH", "")) if value
            ),
            # ARTEMIS resolves its agent config through ARTEMIS_<FILENAME>.
            "ARTEMIS_ARTEMIS_JSONC": str(config_path),
            # ARTEMIS resolves its persistent DataEngine and trace store from
            # this setting during process startup.  Keep it beside the
            # episode evidence instead of the external checkout's default.
            "ARTEMIS_TRACES_DIR": str(config_path.parent),
            "OPENAI_BASE_URL": endpoint,
            "OPENAI_API_KEY": api_key,
            "OPENAI_MODEL": FORMAL_MODEL,
            "ARTEMIS_HELPER_AUTO_INSTALL": "true",
            "ARTEMIS_DEFAULT_MODEL": FORMAL_MODEL,
            # Keep ARTEMIS from silently borrowing a Gemini key from the host
            # shell: this row is defined as model-matched, not native-config.
            "GOOGLE_API_KEY": "",
            "GEMINI_API_KEY": "",
        }
    )
    return environment


def _read_outcome(evidence: Path) -> dict[str, Any]:
    """Read the run outcome ARTEMIS writes, without inventing one."""

    for name in ("run_outcome.json", "outcome.json", "summary.json"):
        candidate = evidence / name
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _failure(error: str) -> RunResult:
    return RunResult(False, error=error, detail={"harness": "artemis", "done_reason": error})


def which_artemis() -> str:
    """Diagnostic helper: report how ARTEMIS would be launched, if at all."""

    try:
        root = _artemis_root()
    except ValueError as error:
        return str(error)
    return shutil.which("artemis") or str(root / ".venv" / "bin" / "artemis")


__all__ = ["ArtemisHarness", "factory"]
