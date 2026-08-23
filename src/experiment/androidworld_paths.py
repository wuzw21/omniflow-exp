"""Canonical AndroidWorld archive names.

The experiment CLI still accepts historical device labels because they are
part of the launcher/configuration contract.  Archive paths deliberately use
the physical AVD/model identity instead of those labels.
"""

from __future__ import annotations

from pathlib import Path
import re


METHOD_ALIASES = {
    "fixed": "fixed_replay",
    "fixed-replay": "fixed_replay",
    "fixed replay": "fixed_replay",
    "ours": "omniflow",
    "omni-flow": "omniflow",
    "t3a": "t3a_hint",
    "t3a+hint": "t3a_hint",
    "t3a-hint": "t3a_hint",
    "mobilegpt-offline-retrieval": "mobilegpt",
    "mobilegpt_runlog_direct_memory": "mobilegpt",
}


def canonical_method_name(value: str, *, fallback: str = "method") -> str:
    """Return the one method spelling used by AndroidWorld archive paths."""

    raw = str(value or "").strip()
    key = re.sub(r"\s+", " ", raw.lower())
    normalized = METHOD_ALIASES.get(key, raw)
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", normalized).strip("._-")
    return normalized or fallback


def canonical_device_model(
    *,
    label: str = "",
    serial: str = "",
    console_port: int | None = None,
) -> str:
    """Resolve an accepted CLI alias to the configured physical AVD model."""

    label_key = str(label or "").strip().lower()
    serial_key = str(serial or "").strip().lower()
    port = int(console_port or 0)
    if label_key in {"source5554", "source5560", "source5556"}:
        return "OmniFlowSourceSmall"
    if label_key in {"small5554", "tablet45554"}:
        return "WXGA_Tablet_test_00"
    if label_key in {
        "small5562",
        "target5554",
        "target5562",
        "standard45562",
    }:
        return "OmniFlowTargetSmall"
    if label_key in {"pixel5576", "target5576"}:
        return "AndroidWorldAvd4090"
    if label_key in {"fold5564", "target5564", "fold45564"}:
        return "OmniFlowTargetFold"
    if serial_key.endswith("45564") or port == 45564:
        return "OmniFlowTargetFold"
    if serial_key.endswith("45554") or port == 45554:
        return "WXGA_Tablet_test_00"
    if serial_key.endswith("45562") or port == 45562:
        return "OmniFlowTargetSmall"
    if serial_key.endswith("5564") or port == 5564:
        return "OmniFlowTargetFold"
    if serial_key.endswith("5554") or port == 5554:
        return "WXGA_Tablet_test_00"
    if serial_key.endswith("5562") or port == 5562:
        return "OmniFlowTargetSmall"
    if serial_key.endswith("5576") or port == 5576:
        return "AndroidWorldAvd4090"
    if serial_key.endswith(("5556", "5560")) or port in {5556, 5560}:
        return "OmniFlowSourceSmall"
    return str(label or serial or f"device{port or 0}").strip() or "device"


def canonical_device_seed_name(
    *,
    label: str = "",
    serial: str = "",
    console_port: int | None = None,
    source_seed: int = 111,
    evaluation_seed: int = 113,
) -> str:
    """Return the device directory name, including the reproducibility seeds."""

    model = canonical_device_model(
        label=label,
        serial=serial,
        console_port=console_port,
    )
    name = f"{model}_seed{int(source_seed)}"
    if model != "OmniFlowSourceSmall":
        name += f"_eval{int(evaluation_seed)}"
    return name


def canonical_device_metadata(
    *,
    label: str = "",
    serial: str = "",
    console_port: int | None = None,
    source_seed: int = 111,
    evaluation_seed: int = 113,
) -> dict[str, object]:
    """Metadata suitable for ``device.json`` in an archive cell."""

    model = canonical_device_model(
        label=label,
        serial=serial,
        console_port=console_port,
    )
    if model == "WXGA_Tablet_test_00":
        profile = "tablet"
    elif model == "OmniFlowTargetFold":
        profile = "pixel_fold"
    elif model == "AndroidWorldAvd4090":
        profile = "pixel_phone"
    elif model == "OmniFlowTargetSmall":
        profile = "small_phone"
    else:
        profile = "small_phone"
    return {
        "device_model": model,
        "avd": model,
        "profile": profile,
        "archive_name": canonical_device_seed_name(
            label=label,
            serial=serial,
            console_port=console_port,
            source_seed=source_seed,
            evaluation_seed=evaluation_seed,
        ),
        "aliases": [value for value in (str(label or "").strip(),) if value],
        "serial": str(serial or "").strip(),
        "console_port": int(console_port or 0),
        "source_seed": int(source_seed),
        "evaluation_seed": (
            None if model == "OmniFlowSourceSmall" else int(evaluation_seed)
        ),
    }


def next_attempt_name(runlog_root: str | Path) -> str:
    """Return the next stable ``attempt_NNN`` name below one RunLog root."""

    root = Path(runlog_root).expanduser().resolve()
    numbers = []
    if root.is_dir():
        for candidate in root.iterdir():
            suffix = candidate.name.removeprefix("attempt_")
            if candidate.is_dir() and suffix.isdigit():
                numbers.append(int(suffix))
    return f"attempt_{max(numbers, default=0) + 1:03d}"
