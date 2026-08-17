"""Canonical AndroidWorld experiment protocol values."""

from __future__ import annotations

from omniflow.core.config import ANDROIDWORLD_PROTOCOL, DEFAULT_MAX_STEPS

METHODS = tuple(str(value) for value in ANDROIDWORLD_PROTOCOL["methods"])

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
SOURCE_SEED = int(ANDROIDWORLD_PROTOCOL["source_seed"])
TASK_SEED = int(ANDROIDWORLD_PROTOCOL["evaluation_seed"])
SOURCE_MAX_STEPS = int(ANDROIDWORLD_PROTOCOL["source_max_steps"])
MAX_STEPS = DEFAULT_MAX_STEPS
MAX_FALLBACK_STEPS = int(ANDROIDWORLD_PROTOCOL["max_fallback_steps"])
TASK_DEADLINE_SEC = int(ANDROIDWORLD_PROTOCOL["task_deadline_sec"])
STEP_TIMEOUT_SEC = int(ANDROIDWORLD_PROTOCOL["step_timeout_sec"])
VALIDATOR_FLUSH_GRACE_SEC = int(
    ANDROIDWORLD_PROTOCOL["validator_flush_grace_sec"]
)
EPISODE_TIMEOUT_SEC = MAX_STEPS * STEP_TIMEOUT_SEC + VALIDATOR_FLUSH_GRACE_SEC
FIXED_TASK_SEED = bool(ANDROIDWORLD_PROTOCOL["fixed_task_seed"])
FIXED_TASK_PARAMS = bool(ANDROIDWORLD_PROTOCOL["fixed_task_params"])
FOLD_STATE = int(ANDROIDWORLD_PROTOCOL["fold_state"])
FOLD_SIZE = str(ANDROIDWORLD_PROTOCOL["fold_size"])
FORMAL_MODEL = str(ANDROIDWORLD_PROTOCOL["model"])
FORMAL_MODEL_ENDPOINT_PROFILE = str(
    ANDROIDWORLD_PROTOCOL["model_endpoint_profile"]
)
FORMAL_MODEL_BASE_URL = str(ANDROIDWORLD_PROTOCOL["model_base_url"])
ANDROIDWORLD_REVISION = str(ANDROIDWORLD_PROTOCOL["androidworld_revision"])

# Active result vocabulary. The registry reads the old one_task names only for
# immutable historical attempts and never writes them again.
RESULT_SUMMARY_FILE = "result_summary.json"
RESULT_COMMANDS_FILE = "result_commands.jsonl"
RESULT_MARKDOWN_FILE = "result_summary.md"
RESULT_SCHEMA = "omniflow.androidworld.result.v1"
RESULT_FIELDS = (
    "task",
    "method",
    "device",
    "source_seed",
    "evaluation_seed",
    "status",
    "validator_success",
    "model_calls",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "actions_executed",
    "episode_duration_sec",
    "outer_wall_sec",
    "error",
    "evidence_paths",
)
