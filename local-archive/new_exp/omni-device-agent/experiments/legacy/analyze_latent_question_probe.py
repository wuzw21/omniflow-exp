"""Describe the tiny paired diagnostic; never infer population quality from it."""
import argparse
import hashlib
import json
from pathlib import Path


def analyze(path):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if data.get("status") != "completed":
        raise ValueError("A completed real-model result is required")
    records = data["records"]
    expected = {(i, turn) for i in range(len(data["pairs"])) for turn in (0, 1)}
    actual = [(r["pair_index"], r["turn"]) for r in records]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Missing or duplicate paired conditions")
    if not all(r["split_residual_bit_identical"] and r["receiver_unchanged"] for r in records):
        raise ValueError("Failed split or no-mutation check")
    conditions = ("receiver_only", "fresh_latent", "reused_old_question_packet", "all_evidence_text")
    for row in records:
        if set(row["conditions"]) != set(conditions):
            raise ValueError("Missing fair comparison")
    changed = [r for r in records if r["turn"] == 1]
    comparison = []
    for row in changed:
        current = data["pairs"][row["pair_index"]]["current"]
        fresh = row["conditions"]["fresh_latent"]
        reused = row["conditions"]["reused_old_question_packet"]
        comparison.append({
            "id": row["id"],
            "same_generated_token_ids": fresh["token_ids"] == reused["token_ids"],
            "fresh_minus_reused_em": fresh["exact_match"] - reused["exact_match"],
            "fresh_minus_reused_f1": fresh["f1"] - reused["f1"],
            "source_context_utf8_bytes": len(current["sharer_context"].encode("utf-8")),
            "question_utf8_bytes": len(current["question"].encode("utf-8")),
            "latent_tensor_bytes": row["packet_bytes"],
            "packet_relative_l2_change": row["packet_relative_l2_change"],
        })
    return {
        "status": "descriptive_diagnostic_only", "input_sha256": hashlib.sha256(raw).hexdigest(),
        "question_count": len(records), "changed_question_pair_count": len(changed),
        "real_model_split_identity_checks_passed": len(records),
        "descriptive_scores_not_significance_tests": {
            mode: {"exact_count": sum(r["conditions"][mode]["exact_match"] for r in records),
                   "mean_f1": sum(r["conditions"][mode]["f1"] for r in records) / len(records)}
            for mode in conditions
        },
        "changed_question_comparisons": comparison,
        "claim_boundary": [
            "No heterogeneous, growing-history, network or aggregate quality conclusion.",
            "An unchanged answer does not imply equal logits, KV, or all future continuations.",
            "Gold-string-based pair selection is a diagnostic condition, not a representative sample.",
            "Prefix caching plus raw text remains a required systems baseline.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyze(args.input), ensure_ascii=False, indent=2) + "\n")
