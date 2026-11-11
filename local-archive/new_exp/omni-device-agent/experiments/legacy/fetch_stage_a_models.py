"""Fetch a pinned public 0.5B pair for implementation checks, not the 7B experiment."""
import hashlib
import json
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor

CACHE = Path.home() / ".agent-reach" / "edgecloud-stage-a"
MODELS = {
    "base": ("Qwen/Qwen2.5-0.5B", "060db6499f32faf8b98477b0a26969ef7d8b9987",
             "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"),
    "instruct": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775",
                 "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"),
}
FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "model.safetensors")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_one(key):
    repo, revision, weight_sha = MODELS[key]
    dest = CACHE / "models" / revision
    dest.mkdir(parents=True, exist_ok=True)
    records = []
    for name in FILES:
        path = dest / name
        expected = weight_sha if name == "model.safetensors" else None
        if not path.exists() or (expected and sha256(path) != expected):
            tmp = path.with_suffix(path.suffix + ".download")
            url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}?download=true"
            print(f"downloading {repo}/{name}", flush=True)
            subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                            "--retry", "2", "--connect-timeout", "15", "--max-time", "900",
                            "--max-filesize", "1000000000", "--output", str(tmp), url], check=True)
            if expected and sha256(tmp) != expected:
                raise ValueError(f"Weight SHA mismatch: {repo}")
            if name.endswith(".json"):
                json.loads(tmp.read_text())
            tmp.replace(path)
        records.append({"file": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {"repo": repo, "revision": revision, "license": "Apache-2.0",
                "purpose": "0.5B implementation validation only", "files": records}
    (dest / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"verified {repo}: {dest}", flush=True)
    return str(dest)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=2) as pool:
        print(json.dumps(dict(zip(MODELS, pool.map(fetch_one, MODELS))), indent=2))
