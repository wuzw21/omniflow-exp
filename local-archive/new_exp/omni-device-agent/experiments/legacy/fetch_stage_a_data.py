"""Pinned HotpotQA mirror and audited, unmodified upstream answer evaluator."""
import json
import subprocess
from fetch_stage_a_models import CACHE, sha256

DATA_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
DATA_SHA = "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6"
EVAL_REVISION = "3635853403a8735609ee997664e1528f4480762a"
EVAL_SHA = "d35fc91a6db21d791dbdda11daf3856e9359f5701d54e3eefba20d88fecc02c0"


def fetch():
    folder = CACHE / "data" / DATA_REVISION
    folder.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/{DATA_REVISION}/distractor/validation-00000-of-00001.parquet"
    path = folder / "validation.parquet"
    if not path.exists() or sha256(path) != DATA_SHA:
        tmp = path.with_suffix(".download")
        subprocess.run(["curl", "-fLsS", "--retry", "2", "--connect-timeout", "15",
                        "--max-time", "180", "--max-filesize", "30000000",
                        "-o", str(tmp), url], check=True)
        if sha256(tmp) != DATA_SHA:
            raise ValueError("HotpotQA mirror checksum mismatch")
        tmp.replace(path)
    evaluator = folder / "hotpot_evaluate_v1.py"
    eval_url = f"https://raw.githubusercontent.com/hotpotqa/hotpot/{EVAL_REVISION}/hotpot_evaluate_v1.py"
    if not evaluator.exists() or sha256(evaluator) != EVAL_SHA:
        subprocess.run(["curl", "-fLsS", "--connect-timeout", "15", "--max-time", "60",
                        "--max-filesize", "10000", "-o", str(evaluator), eval_url], check=True)
    if sha256(evaluator) != EVAL_SHA:
        raise ValueError("Upstream evaluator checksum mismatch")
    manifest = {"dataset": "HotpotQA distractor validation", "license": "CC-BY-SA-4.0",
                "mirror_url": url, "revision": DATA_REVISION, "sha256": DATA_SHA,
                "evaluator_url": eval_url, "evaluator_sha256": sha256(evaluator),
                "note": "Original CMU HTTPS failed TLS; HTTP returned 502. Using pinned HF mirror."}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(folder)
    return folder


if __name__ == "__main__":
    fetch()
