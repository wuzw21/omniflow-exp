from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.integrations.android_world.methods import (
    MethodAdapter,
    MethodAdapterContext,
    MethodAdapterRegistry,
    default_method_adapter_registry,
)


def _context(selector: str) -> MethodAdapterContext:
    return MethodAdapterContext(
        selector=selector,
        env=SimpleNamespace(),
        store_path="store.json",
        adb_serial="emulator-5554",
    )


def test_method_registry_resolves_exactly_one_adapter() -> None:
    registry = MethodAdapterRegistry(
        (
            MethodAdapter("first", lambda selector: selector == "first", lambda _: 1),
            MethodAdapter("second", lambda selector: selector == "second", lambda _: 2),
        )
    )

    assert registry.build(_context("second")) == 2


def test_method_registry_rejects_overlapping_adapters() -> None:
    registry = MethodAdapterRegistry(
        (
            MethodAdapter("first", lambda _: True, lambda _: 1),
            MethodAdapter("second", lambda _: True, lambda _: 2),
        )
    )

    with pytest.raises(
        RuntimeError,
        match="androidworld_method_adapter_ambiguous:omniflow:first,second",
    ):
        registry.build(_context("omniflow"))


def test_default_registry_preserves_unknown_selector_error() -> None:
    with pytest.raises(ValueError, match="Unsupported AndroidWorld agent selector"):
        default_method_adapter_registry().build(_context("unknown"))
