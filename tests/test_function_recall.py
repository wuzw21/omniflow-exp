from omniflow.core.model import Observation
from omniflow.functions.recall import (
    _observation_package,
    _recoverable_package_mismatch,
)


def test_missing_source_state_does_not_crash_package_mismatch_preflight() -> None:
    current = Observation(
        xml='<hierarchy><node package="com.android.settings" /></hierarchy>',
        package_name="com.android.settings",
    )

    assert _observation_package(None) == ""
    assert not _recoverable_package_mismatch(
        current_observation=current,
        source_observation=None,
    )
