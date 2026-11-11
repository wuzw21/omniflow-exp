"""Fetch only routing IDs and boundaries, never weights or hidden-state archives."""
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

REPO = "aswinkumar99/qwen3.8-flash-next-expert-traces"
REVISION = "13969e686ba033c9222e2696c59b0cf091297ad9"
PART = "batch-002a"
CACHE = Path.home() / ".agent-reach" / "edgecloud-traces" / REVISION / PART
FILES = ("experts.npy", "req_index.npy", "req_pos.npy", "session_index.npy")


def fetch():
    CACHE.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/compact/{PART}/"
    for name in ("manifest.json", "sessions.json"):
        with urllib.request.urlopen(base + name, timeout=45) as response:
            payload = response.read(1_000_001)
        if len(payload) > 1_000_000:
            raise ValueError("Unexpected metadata size")
        json.loads(payload)
        (CACHE / name).write_bytes(payload)
    manifest = json.loads((CACHE / "manifest.json").read_text())
    for name in FILES:
        expected = manifest["arrays"][name]
        target = CACHE / name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected["sha256"]:
            print(f"verified cached {name}", flush=True)
            continue
        if expected["bytes"] > 110_000_000:
            raise ValueError("File exceeds this experiment's download bound")
        temporary = target.with_suffix(".download")
        # curl follows HF's object-store redirects using the host's TLS setup.
        subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                        "--retry", "2", "--connect-timeout", "15", "--max-time", "180",
                        "--max-filesize", str(expected["bytes"]),
                        "--output", str(temporary), base + name + "?download=true"], check=True)
        total = temporary.stat().st_size
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if total != expected["bytes"] or digest != expected["sha256"]:
            raise ValueError(f"Integrity check failed: {name}")
        temporary.replace(target)
        print(f"verified {name}: {total} bytes", flush=True)
    return CACHE


if __name__ == "__main__":
    print(fetch())
