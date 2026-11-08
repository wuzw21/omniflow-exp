"""Function recall regression cases discovered during device execution."""

from omniflow.core.model import Observation
from omniflow.functions.recall import (
    _observation_package,
    _recoverable_package_mismatch,
)


def test_missing_source_observation_is_treated_as_unknown_package() -> None:
    # A Function can reference a source_state_id whose sibling observation is
    # unavailable on the target device.  Recall must fail closed and let the
    # Planner continue; it must not dereference ``None.package_name``.
    assert _observation_package(None) == ""


def test_missing_source_state_does_not_crash_package_mismatch_preflight() -> None:
    current_observation = Observation(
        xml='<hierarchy><node package="com.android.settings" /></hierarchy>',
        package_name="com.android.settings",
    )

    assert not _recoverable_package_mismatch(
        current_observation=current_observation,
        source_observation=None,
    )
