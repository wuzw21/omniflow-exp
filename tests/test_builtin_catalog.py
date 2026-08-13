from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from omniflow.catalog import load_catalog, load_default_catalog
from omniflow.functions.assets import FunctionStore


def test_default_catalog_contains_both_verified_beverage_functions() -> None:
    catalog = load_default_catalog()

    assert catalog.release_id == "2026.08.06.2"
    assert set(catalog.functions) == {
        "manual_americano_checkout_20260806",
        "order_beverage_meituan",
    }
    function = catalog.functions["order_beverage_meituan"]
    assert len(function.steps) == 8
    assert len(function.bindings) == 2
    assert len(function.checker_rules) >= 10
    assert "never submits or pays" in function.description
    assert all(catalog.get_state(step.source_state_id) for step in function.steps)

    manual = catalog.functions["manual_americano_checkout_20260806"]
    assert len(manual.steps) == 14
    assert manual.bindings == ()
    assert manual.checker_rules == ()
    assert [step.action.tool for step in manual.steps].count("input_text") == 3
    assert "不提交、不支付" in manual.description
    assert all(catalog.get_state(step.source_state_id) for step in manual.steps)


def test_catalog_rejects_modified_release_file(tmp_path: Path) -> None:
    source = load_default_catalog().root
    release = tmp_path / source.name
    release.mkdir()
    for path in source.iterdir():
        (release / path.name).write_bytes(path.read_bytes())
    store_path = release / "function_store.json"
    store_path.write_text(store_path.read_text() + " ", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="catalog_file_checksum_mismatch:function_store.json",
    ):
        load_catalog(release)


def test_catalog_manifest_hashes_are_lowercase_sha256() -> None:
    catalog = load_default_catalog()
    manifest = json.loads((catalog.root / "manifest.json").read_text())

    for name, expected in manifest["files"].items():
        assert len(expected) == 64
        assert expected == expected.lower()
        assert hashlib.sha256((catalog.root / name).read_bytes()).hexdigest() == expected


def test_function_store_seeds_catalog_without_overwriting_user_copy(
    tmp_path: Path,
) -> None:
    catalog = load_default_catalog()
    store_path = tmp_path / "store.json"
    store = FunctionStore(
        store_path,
        seed_functions=catalog.functions.values(),
    )
    assert set(store.functions) == {
        "manual_americano_checkout_20260806",
        "order_beverage_meituan",
    }

    payload = json.loads(store_path.read_text())
    payload["functions"]["order_beverage_meituan"]["description"] = "user copy"
    store_path.write_text(json.dumps(payload), encoding="utf-8")

    reloaded = FunctionStore(
        store_path,
        seed_functions=catalog.functions.values(),
    )
    assert reloaded.functions["order_beverage_meituan"].description == "user copy"

    upgraded = FunctionStore(
        store_path,
        seed_functions=catalog.functions.values(),
        replace_seeded=True,
    )
    assert upgraded.functions["order_beverage_meituan"] == catalog.functions[
        "order_beverage_meituan"
    ]
