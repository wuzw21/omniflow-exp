"""Canonical AndroidWorld experiment protocol values."""

from __future__ import annotations

from omniflow.core.config import DEFAULT_MAX_STEPS

METHODS = (
    "fixed_replay",
    "ours",
    "mobilegpt_offline_retrieval",
    "appagent_demo",
    "t3a_hint",
)

FORMAL_METHODS = METHODS

DEVICES = (
    ("small5554", "emulator-5554", 5554),
    ("fold5564", "emulator-5564", 5564),
)

DEFAULT_METHOD = METHODS[0]
DEFAULT_DEVICE = ":".join(str(value) for value in DEVICES[0])
SOURCE_DEVICE = ("source5560", "emulator-5560", 5560)
SOURCE_SEED = 111
TASK_SEED = 113
SOURCE_MAX_STEPS = 30
MAX_STEPS = DEFAULT_MAX_STEPS
MAX_FALLBACK_STEPS = 5
TASK_DEADLINE_SEC = 1800
STEP_TIMEOUT_SEC = 60
VALIDATOR_FLUSH_GRACE_SEC = 300
EPISODE_TIMEOUT_SEC = MAX_STEPS * STEP_TIMEOUT_SEC + VALIDATOR_FLUSH_GRACE_SEC
FORMAL_MODEL = "GLM-5.1"
FORMAL_MODEL_ENDPOINT_PROFILE = "llmthu"
