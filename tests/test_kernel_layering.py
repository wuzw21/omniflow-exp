"""The omniflow kernel must not depend on benchmark/host integrations in src/."""

import ast
from pathlib import Path

KERNEL = Path(__file__).resolve().parents[1] / "omniflow"


def _imported_modules(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.lineno, node.module


def test_kernel_never_imports_integrations():
    violations = [
        f"{path.relative_to(KERNEL.parent)}:{line}:{module}"
        for path in sorted(KERNEL.rglob("*.py"))
        for line, module in _imported_modules(path)
        if module == "src" or module.startswith("src.") or module.startswith("android_world")
    ]
    assert violations == []


def test_open_app_canonicalization_uses_injected_resolver_only():
    from omniflow.functions.compiler import _canonicalize_open_app_action

    action = {"tool": "open_app", "args": {"package_name": "Contacts"}}
    resolved = _canonicalize_open_app_action(
        action, next_observation=None,
        package_resolver=lambda name: "com.google.android.contacts",
    )
    assert resolved["args"]["package_name"] == "com.google.android.contacts"
    assert action["args"]["package_name"] == "Contacts"

    def broken(name):
        raise RuntimeError("adb unavailable")

    unresolved = _canonicalize_open_app_action(
        action, next_observation=None, package_resolver=broken,
    )
    assert unresolved["args"]["package_name"] == "Contacts"
