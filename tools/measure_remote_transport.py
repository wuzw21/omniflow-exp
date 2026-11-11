#!/usr/bin/env python3
"""Measure persistent remote transport latency without retaining payloads."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime
from hashlib import sha256
import http.client
import json
from pathlib import Path
import re
import select
import socket
import statistics
import subprocess
import sys
import time
from typing import Any

DEFAULT_PAYLOADS = (
    "control_1_byte=1",
    "page_vector_4_kib=4096",
    "transfer_representation_fp16=55197",
    "transfer_representation_fp32=106925",
)
REMOTE_SERVER = r"""
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import sys
import threading
import time

port = int(sys.argv[1])

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        if self.path == "/shutdown":
            response = b'{"status":"stopping"}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        if self.path != "/representation":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length < 0 or length > 2 * 1024 * 1024:
            self.send_error(413)
            return
        started = time.perf_counter_ns()
        representation = self.rfile.read(length)
        result = {
            "received_bytes": len(representation),
            "sha256": sha256(representation).hexdigest(),
            "server_processing_us": (time.perf_counter_ns() - started) / 1000.0,
            "request_retained": False,
        }
        response = json.dumps(result, separators=(",", ":")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format, *args):
        return

server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
server.daemon_threads = True
print("READY", flush=True)
server.serve_forever()
server.server_close()
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure binary request latency through one persistent encrypted SSH "
            "tunnel. The remote receiver hashes each request in memory and discards it."
        )
    )
    parser.add_argument("--host", required=True, help="SSH host alias")
    parser.add_argument("--port", type=int, default=48765)
    parser.add_argument("--cycles", type=int, default=30)
    parser.add_argument(
        "--payload",
        action="append",
        metavar="NAME=BYTES",
        help="Payload label and byte count; repeat for multiple sizes",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    return parser.parse_args()


def parse_payloads(rows: list[str] | None) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for row in rows or list(DEFAULT_PAYLOADS):
        try:
            name, raw_size = row.rsplit("=", 1)
            size = int(raw_size)
        except ValueError as error:
            raise ValueError(f"invalid payload specification: {row}") from error
        if (
            not re.fullmatch(r"[a-zA-Z0-9_.-]+", name)
            or not 0 < size <= 2 * 1024 * 1024
        ):
            raise ValueError(f"invalid payload specification: {row}")
        payloads[name] = (
            sha256(name.encode()).digest() * (size // 32)
            + sha256(f"{name}:{size}".encode()).digest()[: size % 32]
        )
    return payloads


def ensure_local_port_available(port: int) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))


def start_remote_server(host: str, port: int) -> subprocess.Popen[str]:
    if not re.fullmatch(r"[a-zA-Z0-9_.@-]+", host):
        raise ValueError("host must be an SSH alias or simple user@host value")
    command = [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ConnectTimeout=10",
        "-L",
        f"{port}:127.0.0.1:{port}",
        host,
        f"timeout 300s python3 - {port}",
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    process.stdin.write(REMOTE_SERVER)
    process.stdin.close()
    readable, _, _ = select.select([process.stdout], [], [], 15.0)
    if not readable or process.stdout.readline().strip() != "READY":
        process.terminate()
        stderr = process.stderr.read() if process.stderr is not None else ""
        raise RuntimeError(f"remote receiver failed to start: {stderr.strip()}")
    return process


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def measure(port: int, payloads: dict[str, bytes], cycles: int) -> list[dict[str, Any]]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
    latencies: dict[str, list[float]] = defaultdict(list)
    server_times: dict[str, list[float]] = defaultdict(list)
    deltas: dict[str, list[float]] = defaultdict(list)
    names = tuple(payloads)
    baseline_name = names[0]
    warm_payload = payloads[baseline_name]
    connection.request("POST", "/representation", body=warm_payload)
    warm_response = connection.getresponse()
    warm_result = json.loads(warm_response.read())
    if (
        warm_response.status != 200
        or warm_result["sha256"] != sha256(warm_payload).hexdigest()
    ):
        raise RuntimeError("remote warm-up verification failed")
    for cycle in range(cycles):
        order = names if cycle % 2 == 0 else tuple(reversed(names))
        cycle_latency: dict[str, float] = {}
        for name in order:
            payload = payloads[name]
            started = time.perf_counter()
            connection.request("POST", "/representation", body=payload)
            response = connection.getresponse()
            result = json.loads(response.read())
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if (
                response.status != 200
                or result["sha256"] != sha256(payload).hexdigest()
            ):
                raise RuntimeError("remote payload verification failed")
            latencies[name].append(elapsed_ms)
            server_times[name].append(float(result["server_processing_us"]) / 1000.0)
            cycle_latency[name] = elapsed_ms
        for name in names[1:]:
            deltas[name].append(cycle_latency[name] - cycle_latency[baseline_name])
    connection.request("POST", "/shutdown", body=b"")
    connection.getresponse().read()
    connection.close()

    rows = []
    for name in names:
        values = latencies[name]
        row: dict[str, Any] = {
            "name": name,
            "bytes": len(payloads[name]),
            "requests": cycles,
            "rtt_p50_ms": round(statistics.median(values), 2),
            "rtt_p95_ms": round(percentile(values, 0.95), 2),
            "rtt_mean_ms": round(statistics.mean(values), 2),
            "server_read_hash_p50_ms": round(statistics.median(server_times[name]), 3),
            "request_retained": False,
        }
        if name != baseline_name:
            row["paired_increment_over_control_p50_ms"] = round(
                statistics.median(deltas[name]), 2
            )
        rows.append(row)
    return rows


def stop_remote_server(process: subprocess.Popen[str]) -> None:
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def main() -> int:
    args = parse_args()
    if not 0 < args.port < 65536 or not 1 <= args.cycles <= 1000:
        raise ValueError("port or cycle count is out of range")
    payloads = parse_payloads(args.payload)
    ensure_local_port_available(args.port)
    process = start_remote_server(args.host, args.port)
    try:
        rows = measure(args.port, payloads, args.cycles)
    finally:
        stop_remote_server(process)
    result = {
        "schema_version": "omniflow.remote-transport.v1",
        "measured_at": datetime.now(UTC).isoformat(),
        "remote": args.host,
        "transport": "persistent encrypted SSH tunnel",
        "cycles": args.cycles,
        "measurements": rows,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
