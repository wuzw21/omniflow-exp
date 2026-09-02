from __future__ import annotations

import hashlib

from omniflow.core.schemas import (
    CANONICAL_ACTION_SCHEMA_SHA256,
    CANONICAL_ACTION_SCHEMA_VERSION,
    canonical_action_schema_path,
    load_canonical_action_schema,
)


def test_canonical_action_schema_is_pinned() -> None:
    schema = load_canonical_action_schema()

    assert schema["schema_version"] == CANONICAL_ACTION_SCHEMA_VERSION
    digest = hashlib.sha256(canonical_action_schema_path().read_bytes()).hexdigest()
    assert digest == CANONICAL_ACTION_SCHEMA_SHA256
