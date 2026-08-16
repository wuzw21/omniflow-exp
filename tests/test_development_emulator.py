from __future__ import annotations

import pytest

from src.experiment.development_emulator import _console_port


def test_console_port_requires_exact_emulator_serial() -> None:
    assert _console_port("emulator-5554") == 5554
    with pytest.raises(ValueError, match="invalid emulator serial"):
        _console_port("device-5554")
