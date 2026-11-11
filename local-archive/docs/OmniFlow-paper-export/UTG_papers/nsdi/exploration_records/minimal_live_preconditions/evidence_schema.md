# Contacts Minimal Live Preconditions Evidence Schema

Status: Draft
Last Updated: 2026-05-24
Scope: Schema contract for `runtime/evals/preconditions/contacts_minimal_live_preconditions/`.

## Directory

```text
runtime/evals/preconditions/contacts_minimal_live_preconditions/
```

This directory is runtime evidence. It is not a paper source of truth by itself.

## `manifest.json`

Required fields:

```json
{
  "schema_version": "omniflow.preconditions.contacts.v1",
  "status": "planned|planned_skeleton_available|partial|complete|negative_result",
  "created_at": "YYYY-MM-DDTHH:MM:SSZ",
  "protocol_path": "UTG_papers/nsdi/exploration/minimal_live_preconditions/contacts_protocol.md",
  "seed_function": {
    "utg_path": "runtime/utg/contacts_official_success_5554_latest.json",
    "function_id": "aw_function_515be7bf7e"
  },
  "output_files": {
    "cases": "cases.jsonl",
    "predicate_matrix": "predicate_matrix.json",
    "outcomes": "outcomes.json",
    "baseline_report": "baseline_report.json",
    "learned_contract": "learned_contract.json",
    "summary": "summary.md"
  },
  "planned_candidates": [
    {
      "candidate_id": "contacts_home_create_contact_candidate_20260524",
      "status": "planned_only_not_evidence",
      "candidate_type": "derived_function_hypothesis",
      "source_function_id": "aw_function_515be7bf7e",
      "derived_start_node_id": "aw_node_51b4969e57",
      "candidate_blueprint": {
        "schema_version": "omniflow.preconditions.contacts.candidate_blueprint.v1",
        "status": "canonical_digest_recorded_not_asset",
        "canonicalization": "json_stable_sort_keys_no_whitespace",
        "digest_algorithm": "sha256",
        "digest": "ed993c650c5cfb214e7b20b8a41a202e4b5b61537cdc0773b802259d5f3c93c2",
        "digest_scope": "candidate_blueprint.blueprint only",
        "candidate_asset_written": false,
        "promotion_allowed": false
      },
      "materialization_preflight": {
        "schema_version": "omniflow.preconditions.contacts.materialization_preflight.v1",
        "status": "blocked_pending_explicit_asset_creation_confirmation",
        "current_checks": {
          "candidate_blueprint_digest_recomputed": true,
          "reserved_candidate_asset_absent": true,
          "seed_file_unchanged": true
        },
        "promotion_allowed": false
      },
      "promotion_requirements": [],
      "dry_run_materialization_audit": {
        "status": "dry_run_materializable_not_written",
        "seed_file_written": false,
        "candidate_asset_written": false,
        "promotion_allowed": false
      },
      "materialization_readiness": {
        "schema_version": "omniflow.preconditions.contacts.materialization_readiness.v1",
        "status": "ready_for_explicit_asset_materialization_not_promotion",
        "seed_asset_immutable": true,
        "candidate_asset_path_reserved": "runtime/evals/preconditions/contacts_minimal_live_preconditions/derived_candidates/contacts_home_create_contact_candidate_20260524.utg.json",
        "candidate_asset_written": false,
        "asset_write_allowed_without_explicit_confirmation": false,
        "promotion_allowed": false
      },
      "candidate_materialization_protocol": {
        "schema_version": "omniflow.preconditions.contacts.candidate_materialization_protocol.v1",
        "status": "planned_protocol_only",
        "requires_explicit_new_file_confirmation": true,
        "materialization_phase": {
          "allowed_action": "write only the reserved separate candidate asset",
          "disallowed_actions": [
            "mutate seed UTG",
            "update outcomes/baselines/learned contract from materialization alone"
          ]
        },
        "live_probe_phase": {
          "allowed_only_after_asset_write": true,
          "required_start_state": "live Contacts home observation matching aw_node_51b4969e57"
        }
      }
    }
  ],
  "claim_status": "exploration_only",
  "promotion_gate": {
    "positive_case_count": 0,
    "negative_or_repair_case_count": 0,
    "terminal_evidence_for_success": false,
    "baselines_scored": false,
    "checking_cost_measured": false,
    "promotion_candidate": false
  }
}
```

`planned_candidates` is optional and must not be treated as evidence. It is for materialization hypotheses only. A planned
candidate may list derivation rules, retained actions, future command skeletons, and promotion requirements, but it must not
change case labels, baseline denominators, learned-contract status, or promotion-gate counts until live or oracle evidence
satisfies the protocol.

`candidate_blueprint` may be attached to a planned candidate to freeze the deterministic identity of a future materialization
step. The digest must be computed over `candidate_blueprint.blueprint` only, with stable JSON object-key sorting and no
whitespace. It can be used to detect drift in the source Function id, removed prefix action, retained action sequence,
argument binding, derived start/end nodes, or reserved asset path. It must remain `canonical_digest_recorded_not_asset` until
the candidate asset exists, and it cannot promote labels, score baselines, or update the learned contract.

`materialization_preflight` may be attached to a planned candidate to record the last no-write checks before a future asset
materialization. It should confirm blueprint digest recomputation, seed Function shape, retained action sequence, reserved
asset absence, and seed-file immutability. If explicit new-file confirmation is still missing, it must keep
`status=blocked_pending_explicit_asset_creation_confirmation`. A preflight block cannot prove live execution, cannot replace
post-write function-id uniqueness checks, and cannot update outcomes, baselines, or the learned contract.

`dry_run_materialization_audit` may be attached to a planned candidate when the candidate is derived in memory from an existing
asset. This audit can prove schema-level materializability only. It must explicitly state whether the seed file or candidate
asset was written and whether promotion is allowed. A dry-run audit cannot prove live success, terminal/business-effect
verification, start-node match on a live observation, or cost superiority over replanning.

`materialization_readiness` may be attached after a dry-run audit to record whether a candidate is ready for a future explicit
asset-writing step. It must distinguish "ready to write a separate candidate asset after confirmation" from "ready to promote".
If the candidate file does not exist yet, it must keep `candidate_asset_written=false`, `promotion_allowed=false`, and
`asset_write_allowed_without_explicit_confirmation=false`. It must also list any remaining blockers such as function-id
uniqueness checks, live execution, terminal verification, and row-level cost fields.

`candidate_materialization_protocol` may be attached to a planned candidate to define the future asset-writing and live-probe
procedure. It must remain `planned_protocol_only` until a candidate asset is actually written. The protocol must separate
materialization checks from live execution checks and must state that materialization alone cannot update `outcomes.json`,
`baseline_report.json`, or `learned_contract.json`.

## `cases.jsonl`

One JSON object per case:

```json
{
  "case_id": "c1_launcher_wrong_start",
  "case_family": "wrong_start|positive|repair|missing_target|overlay|argument|verifier|cross_app",
  "state_source": {
    "kind": "live|offline_fixture|synthetic",
    "path": "optional/source/path",
    "device": "emulator-5554"
  },
  "perturbation": {
    "description": "start from launcher with no route-only bridge",
    "predicate_interventions": {
      "p_start_node_match": false
    }
  },
  "expected_outcome": "precondition_failure",
  "label_status": "unlabeled|oracle_labeled|execution_labeled",
  "notes": "why this case is included"
}
```

## `predicate_matrix.json`

Required shape:

```json
{
  "schema_version": "omniflow.preconditions.predicate_matrix.v1",
  "predicates": [
    {
      "id": "p_start_node_match",
      "type": "boolean",
      "source": "UTG node match / route graph",
      "check_cost": {
        "unit": "observation_or_static_check",
        "estimated_count": 1
      }
    }
  ],
  "rows": [
    {
      "case_id": "c1_launcher_wrong_start",
      "values": {
        "p_start_node_match": false
      },
      "unknown_values": [],
      "evidence_refs": []
    }
  ]
}
```

Unknown predicate values must be explicit. Do not silently coerce unknown to false.

## `outcomes.json`

Required shape:

```json
{
  "schema_version": "omniflow.preconditions.outcomes.v1",
  "outcome_labels": [
    "success",
    "precondition_failure",
    "unsafe_or_wrong_effect",
    "reject",
    "reobserve_or_repair",
    "abstain"
  ],
  "rows": [
    {
      "case_id": "c1_launcher_wrong_start",
      "label": "precondition_failure",
      "label_source": "execution|offline_oracle|planned_expected_only",
      "execution": {
        "ran": false,
        "command": null,
        "error_code": "EXEC_START_NODE_PRECONDITION_FAIL"
      },
      "terminal_evidence": {
        "available": false,
        "path": null,
        "summary": null
      },
      "cost_measurement": {
        "observation_capture_ms": null,
        "predicate_check_ms": null,
        "node_match_ms": null,
        "argument_validation_ms": null,
        "route_bridge_check_ms": null,
        "route_bridge_execution_ms": null,
        "semantic_replay_ms": null,
        "terminal_verify_ms": null,
        "total_admission_check_ms": null,
        "total_reuse_path_ms": null,
        "replan_wall_ms": null,
        "replan_model_tokens": null,
        "replan_model_cost_proxy": null,
        "timing_source": null,
        "unresolved_reason": "not_measured"
      }
    }
  ]
}
```

Expected labels from the protocol must use `label_source=planned_expected_only` until verified.

Cost fields may be `null` while a row is planned or oracle-only, but the row must then keep the cost gate unresolved. Execution
rows used for a learned or selected contract must populate `total_admission_check_ms`, `total_reuse_path_ms`, and a measured
or explicitly proxied replan/rerun cost. Do not infer cost from stdout success or final labels alone.

## `baseline_report.json`

Required shape:

```json
{
  "schema_version": "omniflow.preconditions.baseline_report.v1",
  "baselines": [
    {
      "name": "pagevectorset_threshold",
      "policy": "allow if page/family similarity exceeds threshold",
      "thresholds": {
        "page_similarity": 0.0
      },
      "scoring_status": "scored|blocked_missing_evidence|blocked_insufficient_data",
      "blocked_reason": "required when scoring_status starts with blocked",
      "required_evidence": [
        "evidence required before this baseline can be scored"
      ],
      "next_unblock_action": "concrete next action required before scoring",
      "must_not_infer": [
        "claims or labels that must not be inferred from current artifacts"
      ],
      "diagnostic_status": "optional partial_diagnostic_only_not_scored marker for blocker audits",
      "diagnostic_partial_scores": [
        {
          "case_id": "optional case id",
          "score_kind": "existing_utg_match_top1|xml_similarity_best_over_seed_function_nodes|other",
          "source": "path to score evidence or diagnostic command source",
          "best_node_id": "optional matched node id",
          "best_score": 0.0,
          "label": "grounded outcome label when available",
          "diagnostic_implication": "why this is or is not enough to score the baseline"
        }
      ],
      "missing_for_scoring": [
        "remaining evidence required before this baseline can move to scored"
      ],
      "decisions": [
        {
          "case_id": "c1_launcher_wrong_start",
          "decision": "allow|reject|reobserve_or_repair|abstain",
          "reason": "short explanation"
        }
      ],
      "metrics": {
        "false_accept_rate": null,
        "false_reject_rate": null,
        "allowed_success_rate": null,
        "unsafe_or_wrong_effect_count": null
      }
    }
  ]
}
```

Metrics must remain null until outcomes are labeled.
If a baseline remains unscored, it must carry `scoring_status`, `blocked_reason`, `required_evidence`, and
`next_unblock_action` so future agents can distinguish missing evidence from a measured baseline failure.
Optional diagnostic score fields may be recorded for blocker audits, but they must use
`diagnostic_status=partial_diagnostic_only_not_scored` and leave metrics null until every scored row has score provenance, a
frozen threshold rule, and grounded outcome labels.

## `learned_contract.json`

Required shape:

```json
{
  "schema_version": "omniflow.preconditions.learned_contract.v1",
  "status": "planned|learned|not_learned_insufficient_data|negative_result",
  "method": "decision_tree|sparse_logistic|mdl_selector|manual_hypothesis",
  "contract": {
    "expression": "p_args_valid and p_terminal_verifier_available",
    "predicate_ids": ["p_args_valid", "p_terminal_verifier_available"],
    "predicate_count": 2,
    "estimated_check_cost": {
      "observation_count": 1,
      "static_check_count": 1
    }
  },
  "training_cases": [],
  "metrics": {
    "false_accept_rate": null,
    "false_reject_rate": null,
    "allowed_success_rate": null
  },
  "promotion_gate": {
    "beats_pagevectorset_threshold": false,
    "beats_llm_judge": false,
    "checking_cost_below_replan": false,
    "smaller_than_exhaustive_guard": false,
    "cost_gate_status": "unresolved|passed|failed"
  }
}
```

If the contract is a hypothesis, use `method=manual_hypothesis` and `status=planned`.

## `summary.md`

Required sections:

- Status
- Dataset
- Predicate Coverage
- Outcome Coverage
- Baseline Results
- Learned Contract
- Cost Comparison
- Failure Analysis
- Promotion Decision

The promotion decision must be one of:

- `stay_exploration`
- `collect_more_cases`
- `negative_result`
- `promotion_candidate`
