"""Canonical AndroidWorld archive names for the fixed experiment devices."""

from __future__ import annotations

from pathlib import Path
import re

from omniflow.core.config import ANDROIDWORLD_PROTOCOL


_CONFIGURED_DEVICES = tuple(ANDROIDWORLD_PROTOCOL["devices"]) + (
    ANDROIDWORLD_PROTOCOL["source_device"],
)


def _configured_device(
    *,
    label: str = "",
    serial: str = "",
    console_port: int | None = None,
) -> dict[str, object] | None:
    label_key = str(label or "").strip().lower()
    serial_key = str(serial or "").strip().lower()
    port = int(console_port or 0)
    for device in _CONFIGURED_DEVICES:
        if (
            (label_key and label_key == str(device["label"]).lower())
            or (serial_key and serial_key == str(device["serial"]).lower())
            or (port and port == int(device["console_port"]))
        ):
            return dict(device)
    return None


METHOD_ALIASES = {
    "fixed": "fixed_replay",
    "fixed-replay": "fixed_replay",
    "fixed replay": "fixed_replay",
    "ours": "omniflow",
    "omni-flow": "omniflow",
    "t3a": "t3a_hint",
    "t3a+hint": "t3a_hint",
    "t3a-hint": "t3a_hint",
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

    configured = _configured_device(
        label=label,
        serial=serial,
        console_port=console_port,
    )
    if configured is not None:
        return str(configured["avd"])

    port = int(console_port or 0)
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
    configured = _configured_device(
        label=label,
        serial=serial,
        console_port=console_port,
    )
    if configured is not None:
        profile = str(configured["profile"])
    elif model == "WXGA_Tablet_test_00":
        profile = "tablet"
    elif model == "OmniFlowTargetFold":
        profile = "pixel_fold"
    elif model == "AndroidWorldAvd4090":
        profile = "pixel_phone"
    elif model == "OmniFlowTargetPixel6Pro":
        profile = "pixel_6_pro"
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
