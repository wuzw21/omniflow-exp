"""Canonical AndroidWorld experiment protocol values."""

from __future__ import annotations

import os

from omniflow.core.config import ANDROIDWORLD_PROTOCOL, DEFAULT_MAX_STEPS

METHODS = tuple(str(value) for value in ANDROIDWORLD_PROTOCOL["methods"])
SOURCE_METHOD = "source"
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
# All online and authoring calls use the same bounded Qwen reasoning mode.
# The value is part of the protocol, not a per-method runtime override.
FORMAL_THINKING = "enabled"
FORMAL_MODEL_ENDPOINT_PROFILE = str(
    ANDROIDWORLD_PROTOCOL["model_endpoint_profile"]
)
FORMAL_MODEL_BASE_URL = str(ANDROIDWORLD_PROTOCOL["model_base_url"])
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

# Active result vocabulary. The registry reads the old one_task names only for
# immutable historical attempts and never writes them again.
RESULT_SUMMARY_FILE = "result_summary.json"
RESULT_COMMANDS_FILE = "result_commands.jsonl"
RESULT_MARKDOWN_FILE = "result_summary.md"
RESULT_SCHEMA = "omniflow.androidworld.result.v1"
