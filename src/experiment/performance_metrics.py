"""Latency and best-effort energy diagnostics for one experiment episode."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Iterable

PERFORMANCE_METRICS_SCHEMA = "omniflow.androidworld.performance-metrics.v1"


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return round(ordered[0], 6)
    position = (len(ordered) - 1) * float(probability)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        result = ordered[lower]
    else:
        result = ordered[lower] + (ordered[upper] - ordered[lower]) * (
            position - lower
        )
    return round(result, 6)


def summarize_durations(
    durations_ms: Iterable[float],
    *,
    failed_count: int = 0,
    include_samples: bool = True,
) -> dict[str, Any]:
    samples = [round(max(0.0, float(value)), 6) for value in durations_ms]
    total = sum(samples)
    result: dict[str, Any] = {
        "count": len(samples),
        "total_ms": round(total, 6),
        "mean_ms": round(total / len(samples), 6) if samples else 0.0,
        "p50_ms": _percentile(samples, 0.50),
        "p95_ms": _percentile(samples, 0.95),
        "min_ms": round(min(samples), 6) if samples else 0.0,
        "max_ms": round(max(samples), 6) if samples else 0.0,
        "failed_count": max(0, int(failed_count)),
    }
    if include_samples:
        result["samples_ms"] = samples
    return result


class AdbEnergySampler:
    """Capture battery telemetry without treating instantaneous current as energy."""

    def __init__(self, *, adb_path: str = "", serial: str = "") -> None:
        self.adb_path = str(adb_path or "").strip() or "adb"
        self.serial = str(serial or "").strip()

    def snapshot(self) -> dict[str, Any]:
        if not self.serial or shutil.which(self.adb_path) is None:
            return {
                "available": False,
                "error": "adb_unavailable_or_serial_missing",
            }
        battery = self._run_shell("dumpsys", "battery")
        charge_counter = self._run_shell(
            "cat", "/sys/class/power_supply/battery/charge_counter"
        )
        if battery is None and charge_counter is None:
            return {"available": False, "error": "adb_battery_snapshot_failed"}
        parsed = self._parse_battery(battery or "")
        charge_value = _first_number(charge_counter or "")
        if charge_value is not None:
            parsed["charge_counter_uah"] = charge_value
        parsed["available"] = bool(parsed or charge_value is not None)
        if battery is not None:
            parsed["raw_dumpsys_battery"] = battery[-4000:]
        return parsed

    def _run_shell(self, *arguments: str) -> str | None:
        command = [self.adb_path, "-s", self.serial, "shell", *arguments]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        return str(result.stdout or "").strip()

    @staticmethod
    def _parse_battery(text: str) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for line in str(text or "").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            normalized = key.strip().lower().replace(" ", "_")
            value = value.strip()
            if normalized == "level":
                parsed["level_percent"] = _first_number(value)
            elif normalized == "voltage":
                parsed["voltage_mv"] = _first_number(value)
            elif normalized.endswith("powered"):
                parsed[normalized] = value.casefold() == "true"
            elif normalized in {"charge_counter", "charge_counter_uah"}:
                counter = _first_number(value)
                if counter is not None:
                    parsed["charge_counter_uah"] = counter
        powered_keys = (
            "ac_powered",
            "usb_powered",
            "wireless_powered",
            "dock_powered",
        )
        if any(key in parsed for key in powered_keys):
            parsed["plugged"] = any(bool(parsed.get(key)) for key in powered_keys)
        return {key: value for key, value in parsed.items() if value is not None}


class PerformanceMetrics:
    """Collect one episode's timing samples and optional ADB energy evidence."""

    def __init__(
        self,
        *,
        adb_path: str = "",
        adb_serial: str = "",
        energy_enabled: bool = True,
    ) -> None:
        self._durations: dict[str, list[float]] = defaultdict(list)
        self._failures: dict[str, int] = defaultdict(int)
        self._started_ns: int | None = None
        self._finished_ns: int | None = None
        self._energy_sampler = (
            AdbEnergySampler(adb_path=adb_path, serial=adb_serial)
            if energy_enabled
            else None
        )
        self._energy_start: dict[str, Any] | None = None
        self._energy_end: dict[str, Any] | None = None

    def start(self) -> None:
        if self._started_ns is None:
            self._started_ns = time.perf_counter_ns()
            if self._energy_sampler is not None:
                self._energy_start = self._energy_sampler.snapshot()

    def finish(self, *, method_wall_sec: float | None = None) -> None:
        if self._started_ns is None:
            self.start()
        self._finished_ns = time.perf_counter_ns()
        if self._energy_sampler is not None:
            self._energy_end = self._energy_sampler.snapshot()
        if method_wall_sec is not None:
            self._method_wall_sec = max(0.0, float(method_wall_sec))

    def record(self, operation: str, duration_ms: float, *, success: bool = True) -> None:
        name = str(operation or "unknown").strip() or "unknown"
        self._durations[name].append(max(0.0, float(duration_ms)))
        if not success:
            self._failures[name] += 1

    def timed(self, operation: str):
        return _TimingContext(self, operation)

    def to_dict(self) -> dict[str, Any]:
        started_ns = self._started_ns
        finished_ns = self._finished_ns
        wall_sec = getattr(self, "_method_wall_sec", None)
        if wall_sec is None and started_ns is not None and finished_ns is not None:
            wall_sec = (finished_ns - started_ns) / 1_000_000_000.0
        timing = {
            operation: summarize_durations(
                values,
                failed_count=self._failures.get(operation, 0),
            )
            for operation, values in sorted(self._durations.items())
        }
        return {
            "schema_version": PERFORMANCE_METRICS_SCHEMA,
            "method_wall_sec": round(float(wall_sec or 0.0), 6),
            "timing": timing,
            "energy": self._energy_summary(wall_sec),
        }

    def _energy_summary(self, wall_sec: float | None) -> dict[str, Any]:
        start = dict(self._energy_start or {})
        end = dict(self._energy_end or {})
        result: dict[str, Any] = {
            "measurement_available": False,
            "source": "adb_battery_telemetry",
            "start": start,
            "end": end,
            "estimated_mwh": None,
            "estimated_average_power_mw": None,
            "caveat": (
                "ADB charge-counter delta is a diagnostic estimate; hardware power "
                "measurement is required for authoritative energy results."
            ),
        }
        start_counter = _number(start.get("charge_counter_uah"), -1.0)
        end_counter = _number(end.get("charge_counter_uah"), -1.0)
        start_voltage = _number(start.get("voltage_mv"), 0.0)
        end_voltage = _number(end.get("voltage_mv"), 0.0)
        if (
            start_counter >= 0
            and end_counter >= 0
            and end_counter <= start_counter
            and start_voltage > 0
            and end_voltage > 0
            and start.get("plugged") is False
            and end.get("plugged") is False
        ):
            delta_uah = start_counter - end_counter
            average_voltage = (start_voltage + end_voltage) / 2.0
            estimated_mwh = delta_uah * average_voltage / 1_000_000.0
            result["measurement_available"] = True
            result["estimated_mwh"] = round(max(0.0, estimated_mwh), 6)
            if float(wall_sec or 0.0) > 0:
                result["estimated_average_power_mw"] = round(
                    max(0.0, estimated_mwh) * 3600.0 / float(wall_sec),
                    6,
                )
        elif start or end:
            result["caveat"] += (
                " Charge counter, voltage, or stable plug state was unavailable."
            )
        else:
            result["caveat"] += " ADB battery telemetry was unavailable."
        return result


class _TimingContext:
    def __init__(self, metrics: PerformanceMetrics, operation: str) -> None:
        self.metrics = metrics
        self.operation = operation
        self.started_ns = 0

    def __enter__(self) -> _TimingContext:
        self.started_ns = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type: Any, _exc: Any, _traceback: Any) -> None:
        duration_ms = (time.perf_counter_ns() - self.started_ns) / 1_000_000.0
        self.metrics.record(
            self.operation,
            duration_ms,
            success=exc_type is None,
        )


def _first_number(value: Any) -> float | None:
    text = str(value or "").strip()
    try:
        return float(text.split()[0])
    except (IndexError, TypeError, ValueError):
        return None


def aggregate_performance_metrics(
    rows: Iterable[dict[str, Any]],
    *,
    _include_groups: bool = True,
) -> dict[str, Any]:
    """Aggregate task-level performance payloads while retaining method groups."""
    materialized_rows = list(rows)
    operation_samples: dict[str, list[float]] = defaultdict(list)
    operation_failures: dict[str, int] = defaultdict(int)
    method_wall: list[float] = []
    energy_rows: list[dict[str, Any]] = []
    for row in materialized_rows:
        payload = row.get("performance_metrics") if isinstance(row, dict) else None
        if not isinstance(payload, dict):
            continue
        method_wall.append(_number(payload.get("method_wall_sec")))
        timing = payload.get("timing")
        if isinstance(timing, dict):
            for operation, summary in timing.items():
                if not isinstance(summary, dict):
                    continue
                samples = summary.get("samples_ms")
                if isinstance(samples, list):
                    operation_samples[str(operation)].extend(
                        _number(value) for value in samples
                    )
                operation_failures[str(operation)] += int(
                    _number(summary.get("failed_count"))
                )
        energy = payload.get("energy")
        if isinstance(energy, dict):
            energy_rows.append(energy)
    result = {
        "schema_version": PERFORMANCE_METRICS_SCHEMA,
        "task_count": len(method_wall),
        "method_wall_sec": _summarize_seconds(method_wall),
        "timing": {
            operation: summarize_durations(
                samples,
                failed_count=operation_failures.get(operation, 0),
                include_samples=False,
            )
            for operation, samples in sorted(operation_samples.items())
        },
        "energy": {
            "measurement_available_count": sum(
                bool(row.get("measurement_available")) for row in energy_rows
            ),
            "task_count": len(energy_rows),
            "estimated_mwh_total": round(
                sum(
                    _number(row.get("estimated_mwh"))
                    for row in energy_rows
                    if row.get("measurement_available")
                ),
                6,
            ),
            "caveat": (
                "ADB estimates are diagnostic only; use a hardware power analyzer "
                "for authoritative power and energy comparisons."
            ),
        },
    }
    if _include_groups:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in materialized_rows:
            if not isinstance(row, dict) or not isinstance(
                row.get("performance_metrics"), dict
            ):
                continue
            method = str(
                row.get("method") or row.get("agent") or "unknown"
            ).strip() or "unknown"
            grouped[method].append(row)
        result["by_method"] = {
            method: aggregate_performance_metrics(group_rows, _include_groups=False)
            for method, group_rows in sorted(grouped.items())
        }
    return result


def _summarize_seconds(values: Iterable[float]) -> dict[str, Any]:
    samples = [round(max(0.0, float(value)), 6) for value in values]
    total = sum(samples)
    return {
        "count": len(samples),
        "total_sec": round(total, 6),
        "mean_sec": round(total / len(samples), 6) if samples else 0.0,
        "p50_sec": _percentile(samples, 0.50),
        "p95_sec": _percentile(samples, 0.95),
        "min_sec": round(min(samples), 6) if samples else 0.0,
        "max_sec": round(max(samples), 6) if samples else 0.0,
    }


def write_performance_metrics(metrics: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


__all__ = [
    "AdbEnergySampler",
    "PERFORMANCE_METRICS_SCHEMA",
    "PerformanceMetrics",
    "aggregate_performance_metrics",
    "summarize_durations",
    "write_performance_metrics",
]
