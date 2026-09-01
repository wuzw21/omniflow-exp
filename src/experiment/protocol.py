"""Canonical AndroidWorld experiment protocol values."""

from __future__ import annotations

import os

from omniflow.core.config import ANDROIDWORLD_PROTOCOL, DEFAULT_MAX_STEPS

METHODS = tuple(str(value) for value in ANDROIDWORLD_PROTOCOL["methods"])
ENABLED_METHODS = tuple(str(value) for value in ANDROIDWORLD_PROTOCOL["enabled_methods"])
SOURCE_METHOD = "source"
AUTODROID_MEMORY_METHOD = "autodroid"
DEFAULT_TASK = str(ANDROIDWORLD_PROTOCOL["task"])
DEVICES = tuple(
    (
        str(device["label"]),
        str(device["serial"]),
        int(device["console_port"]),
    )
    for device in ANDROIDWORLD_PROTOCOL["devices"]
)
DEFAULT_METHOD = METHODS[0]
DEFAULT_DEVICE = ":".join(str(value) for value in DEVICES[0])
_SOURCE_DEVICE = ANDROIDWORLD_PROTOCOL["source_device"]
SOURCE_DEVICE = (
    str(_SOURCE_DEVICE["label"]),
    str(_SOURCE_DEVICE["serial"]),
    int(_SOURCE_DEVICE["console_port"]),
)
SOURCE_AVD = str(_SOURCE_DEVICE["avd"])
DEVICE_AVDS = tuple(
    (str(device["serial"]), str(device["avd"]))
    for device in (*ANDROIDWORLD_PROTOCOL["devices"], _SOURCE_DEVICE)
)
EMULATOR_AVD_SPECS = tuple(
    (
        str(device["avd"]),
        int(device["api_level"]),
        str(device["profile"]),
    )
    for device in (*ANDROIDWORLD_PROTOCOL["devices"], _SOURCE_DEVICE)
)
SOURCE_SEED = int(ANDROIDWORLD_PROTOCOL["source_seed"])
TASK_SEED = int(ANDROIDWORLD_PROTOCOL["evaluation_seed"])
SOURCE_MAX_STEPS = int(ANDROIDWORLD_PROTOCOL["source_max_steps"])
MAX_STEPS = DEFAULT_MAX_STEPS
MAX_FALLBACK_STEPS = int(ANDROIDWORLD_PROTOCOL["max_fallback_steps"])
FUNCTION_ENHANCEMENT_TIMEOUT_SEC = int(
    ANDROIDWORLD_PROTOCOL["function_enhancement_timeout_sec"]
)
BMOCA_RESULT_TIMEOUT_SEC = int(ANDROIDWORLD_PROTOCOL["bmoca_result_timeout_sec"])
TASK_DEADLINE_SEC = int(ANDROIDWORLD_PROTOCOL["task_deadline_sec"])
STEP_TIMEOUT_SEC = int(ANDROIDWORLD_PROTOCOL["step_timeout_sec"])
# Formal AndroidWorld runs do not accept a per-run planner timeout.  Keep the
# one request budget here with the rest of the protocol constants.
PLANNER_TIMEOUT_SEC = 30.0
VALIDATOR_FLUSH_GRACE_SEC = int(
    ANDROIDWORLD_PROTOCOL["validator_flush_grace_sec"]
)
EPISODE_TIMEOUT_SEC = MAX_STEPS * STEP_TIMEOUT_SEC + VALIDATOR_FLUSH_GRACE_SEC
FIXED_TASK_SEED = bool(ANDROIDWORLD_PROTOCOL["fixed_task_seed"])
FIXED_TASK_PARAMS = bool(ANDROIDWORLD_PROTOCOL["fixed_task_params"])
FOLD_STATE = int(ANDROIDWORLD_PROTOCOL["fold_state"])
FOLD_SIZE = str(ANDROIDWORLD_PROTOCOL["fold_size"])
FORMAL_MODEL = "Qwen3.6-Plus"
OMNIFLOW_PLANNER_MODEL = FORMAL_MODEL
APPAGENT_MODEL = FORMAL_MODEL
# All online and authoring calls use the same non-thinking Qwen mode.
# The value is part of the protocol, not a per-method runtime override.
FORMAL_THINKING = "disabled"
# Request controls are protocol constants as well.  Keeping these here makes
# the launcher, the native planner, and the official baseline adapters agree
# even when the parent shell contains stale experiment variables.
FORMAL_MAX_TOKENS = 512
FORMAL_REQUEST_TIMEOUT_SEC = 120.0
FORMAL_RETRY_WAIT_SEC = 2.0
FORMAL_APPAGENT_TIMEOUT_SEC = 180.0
FORMAL_APPAGENT_EMPTY_RESPONSE_RETRIES = 3
FORMAL_APPAGENT_RETRY_MAX_TOKENS = 512
FORMAL_MODEL_ENDPOINT_PROFILE = str(
    ANDROIDWORLD_PROTOCOL["model_endpoint_profile"]
)
FORMAL_MODEL_BASE_URL = str(ANDROIDWORLD_PROTOCOL["model_base_url"])
# The paper protocol remains pinned to Qwen.  An explicit experimental model
# is allowed only for the OmniFlow adapter, so OmniMind/GPT-5.5 comparisons can
# use the same public launcher and Function/Planner owners without changing
# the formal five-method matrix implicitly.
EXPERIMENTAL_OMNIFLOW_MODEL_ENV = "OMNIFLOW_EXPERIMENTAL_MODEL"


def _experimental_omniflow_model() -> str:
    return str(os.environ.get(EXPERIMENTAL_OMNIFLOW_MODEL_ENV) or "").strip()


def omniflow_model() -> str:
    return _experimental_omniflow_model() or FORMAL_MODEL


def omniflow_endpoint_profile() -> str:
    return (
        str(os.environ.get("OMNIFLOW_EXPERIMENTAL_ENDPOINT_PROFILE") or "").strip()
        or FORMAL_MODEL_ENDPOINT_PROFILE
    )


def omniflow_base_url() -> str:
    return (
        str(os.environ.get("OMNIFLOW_EXPERIMENTAL_BASE_URL") or "").strip()
        or FORMAL_MODEL_BASE_URL
    )


def require_runtime_model(method: str, value: str | None = None) -> str:
    """Validate the model at the public runtime boundary.

    Formal methods retain the fixed paper model.  OmniFlow may opt into one
    explicitly named experimental model through the environment; the value
    is still checked against the same resolved model before execution.
    """

    normalized_method = str(method or "").strip()
    selected = str(value or "").strip()
    if normalized_method == "omniflow" and _experimental_omniflow_model():
        expected = omniflow_model()
        if selected and selected != expected:
            raise ValueError(
                f"omniflow_experimental_model_mismatch:expected={expected}:received={selected}"
            )
        return expected
    return require_formal_model(selected)
ANDROIDWORLD_REVISION = str(
    os.environ.get("OMNIFLOW_ANDROIDWORLD_REVISION")
    or ANDROIDWORLD_PROTOCOL["androidworld_revision"]
)
_DROIDRUN = ANDROIDWORLD_PROTOCOL["droidrun"]
DROIDRUN_VERSION = str(_DROIDRUN["version"])
DROIDRUN_COMMIT = str(_DROIDRUN["commit"])
DROIDRUN_PORTAL_VERSION = str(_DROIDRUN["portal_version"])
DROIDRUN_PORTAL_COMMIT = str(_DROIDRUN["portal_commit"])


def require_formal_model(value: str | None = None) -> str:
    resolved = str(value or "").strip()
    configured = str(ANDROIDWORLD_PROTOCOL.get("model") or "").strip()
    configured_planner = str(
        ANDROIDWORLD_PROTOCOL.get("omniflow_planner_model") or ""
    ).strip()
    if configured != FORMAL_MODEL:
        raise ValueError(
            f"formal_model_config_invalid:expected={FORMAL_MODEL}:configured={configured}"
        )
    if configured_planner != FORMAL_MODEL:
        raise ValueError(
            "formal_planner_model_config_invalid:"
            f"expected={FORMAL_MODEL}:configured={configured_planner}"
        )
    if resolved and resolved != FORMAL_MODEL:
        raise ValueError(
            f"formal_model_mismatch:expected={FORMAL_MODEL}:received={resolved}"
        )
    return FORMAL_MODEL

# Active result vocabulary for immutable historical attempts.
RESULT_SUMMARY_FILE = "result_summary.json"
RESULT_COMMANDS_FILE = "result_commands.jsonl"
RESULT_MARKDOWN_FILE = "result_summary.md"
RESULT_SCHEMA = "omniflow.androidworld.result.v1"
