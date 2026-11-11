# Reasoning Sub-Gate Evidence Schema

Status: Draft
Last Updated: 2026-05-24
Scope: Planned evidence contract for the first CMLP-002 / PABench sub-gate row. This file is not runtime evidence and does not
promote CMLP-002 beyond `planned_skeleton_available`.

## Directory

Future row-level evidence, if explicitly created later, should live under:

```text
runtime/evals/preconditions/reasoning_minimal_validity_conditions/
```

The first allowed row id is:

```text
pabench_subgate_parser_missing_guard_original
```

Do not create fixture files, patch artifacts, or verifier outputs from this schema alone. The parser fixture still requires
explicit confirmation before it exists.

## Required Row Object

Each future row must be a JSON object with this top-level shape:

```json
{
  "schema_version": "omniflow.preconditions.reasoning_subgate_row.v1",
  "row_id": "pabench_subgate_parser_missing_guard_original",
  "status": "planned_only|evidence_available|not_learned_insufficient_data|negative_result",
  "gate_role": "cmlp_reuse_admission_subgate",
  "not_project_controller": true,
  "project_graph_update_owner": "pabench_project_control_after_verifier_pass",
  "project_state": {},
  "verifier_observation": {},
  "cached_skeleton": {},
  "predicate_values": {},
  "admission_decision": {},
  "patch_application": {},
  "verifier_results": [],
  "baseline_decisions": [],
  "cost_measurement": {},
  "risk_metrics": {},
  "promotion_gate": {}
}
```

`status=planned_only` means the row is a plan, not evidence. A row may move out of `planned_only` only after the fixture,
predicate, patch, verifier, baseline, and cost fields are backed by executable artifacts.

## `project_state`

Required fields:

```json
{
  "failure_class": "missing_guard_or_parser_empty_input",
  "affected_artifact_refs": ["parser.py", "test_parser.py"],
  "allowed_write_set": ["parser.py"],
  "rollback_available": true,
  "budget": {
    "max_checker_ms": null,
    "max_model_tokens": null
  },
  "evidence_refs": []
}
```

`ProjectState` is controller-owned state. It must not be rewritten as prompt context or used as proof that CMLP owns the outer
`delegate / verify / rollback / stop / escalate` policy.

## `verifier_observation`

Required fields:

```json
{
  "failing_verifier": "test_rejects_empty_record",
  "failure_signature": "empty input parsed instead of rejected",
  "regression_verifier": "test_parse_valid_record",
  "observation_trust": "executable_checker",
  "failing_before_patch": {
    "available": false,
    "command": null,
    "exit_code": null,
    "stdout_digest": null,
    "stderr_digest": null,
    "wall_ms": null
  }
}
```

`observation_trust` must be `executable_checker` before a row can promote. Model agreement or a self-written acceptance
condition is not verifier observation.

## `cached_skeleton`

Required fields:

```json
{
  "skeleton_id": "code_fix_parser_missing_guard_v0",
  "source_anchor": "parser.py::parse_record",
  "patch_slot": "before first split/parse operation",
  "patch_action": "insert empty-input guard",
  "verifier_obligations": [
    "test_rejects_empty_record",
    "test_parse_valid_record"
  ],
  "public_skeleton_path": null,
  "contains_hidden_chain_of_thought": false
}
```

The reusable object is the public skeleton, not the old model's private rationale.

## `predicate_values`

Required predicate ids:

```json
{
  "failure_signature_matches": null,
  "source_anchor_exists": null,
  "source_anchor_semantics_compatible": null,
  "patch_slot_unique": null,
  "allowed_write_set_contains_patch": null,
  "targeted_verifier_available": null,
  "regression_verifier_available": null,
  "no_conflicting_local_diff": null
}
```

Each value must be `true`, `false`, or `null`. `null` means unknown and cannot be silently coerced to `false`.

Each non-null predicate must include an evidence reference:

```json
{
  "predicate_id": "source_anchor_exists",
  "value": true,
  "source": "static_source_inspection|test_inventory|git_diff|tool_metadata|verifier_log",
  "evidence_ref": "path or command output digest"
}
```

## `admission_decision`

Required fields:

```json
{
  "decision": "allow_reuse|reject|reobserve|repair_precondition|abstain",
  "selected_predicates": [],
  "missing_or_failed_predicates": [],
  "decision_source": "planned|manual_guard|learned_selector|baseline",
  "may_apply_cached_skeleton": false,
  "must_run_verifier_after_apply": true,
  "must_not_update_project_graph_without_verifier_pass": true
}
```

`may_apply_cached_skeleton=true` is forbidden while `status=planned_only` or while any required verifier/cost field is missing.

## `patch_application`

Required fields:

```json
{
  "applied": false,
  "patch_file": null,
  "affected_files": [],
  "within_allowed_write_set": null,
  "apply_command": null,
  "apply_exit_code": null,
  "apply_wall_ms": null
}
```

If `within_allowed_write_set=false`, the row must not be `allow_reuse`.

## `verifier_results`

Required rows:

```json
[
  {
    "verifier": "test_rejects_empty_record",
    "phase": "before_patch|after_patch",
    "command": null,
    "exit_code": null,
    "passed": null,
    "stdout_digest": null,
    "stderr_digest": null,
    "wall_ms": null
  },
  {
    "verifier": "test_parse_valid_record",
    "phase": "after_patch",
    "command": null,
    "exit_code": null,
    "passed": null,
    "stdout_digest": null,
    "stderr_digest": null,
    "wall_ms": null
  }
]
```

A promoted positive row requires failing-before evidence for the targeted verifier, patch-applied evidence, targeted verifier
pass after patch, and regression verifier pass after patch.

## `baseline_decisions`

Required baselines:

```json
[
  {"name": "always_reuse", "decision": null, "false_accept": null, "false_reject": null, "cost_proxy": null},
  {"name": "retrieval_score_reuse", "decision": null, "false_accept": null, "false_reject": null, "cost_proxy": null},
  {"name": "llm_judge", "decision": null, "false_accept": null, "false_reject": null, "cost_proxy": null},
  {"name": "always_rerun_code_agent", "decision": null, "false_accept": null, "false_reject": null, "cost_proxy": null},
  {"name": "hand_written_guard", "decision": null, "false_accept": null, "false_reject": null, "cost_proxy": null}
]
```

The LLM judge is a baseline only. It cannot provide labels or verifier observations.

## `cost_measurement`

Required fields:

```json
{
  "predicate_static_check_ms": null,
  "verifier_probe_ms": null,
  "patch_apply_ms": null,
  "post_apply_verifier_ms": null,
  "total_admission_cost_ms": null,
  "rerun_code_agent_wall_ms": null,
  "rerun_code_agent_model_calls": null,
  "rerun_code_agent_tokens": null,
  "rerun_code_agent_cost_proxy": null,
  "timing_source": null
}
```

If `total_admission_cost_ms >= rerun_code_agent_wall_ms` under a fair proxy, the row must be recorded as
`negative_result` or remain unpromoted.

## `risk_metrics`

Required fields:

```json
{
  "false_accept": null,
  "false_reject": null,
  "abstain": null,
  "unsafe_or_wrong_effect": null,
  "verifier_pass_after_apply": null
}
```

These values must be computed from verifier-backed labels, not from planned expectations.

## `promotion_gate`

Required fields:

```json
{
  "labels_verifier_backed": false,
  "predicate_values_executable": false,
  "patch_application_evidence_available": false,
  "targeted_verifier_pass_after_apply": false,
  "regression_verifier_pass_after_apply": false,
  "beats_retrieval_or_llm_judge_on_false_accept": false,
  "admission_cost_below_rerun": false,
  "project_graph_update_not_claimed": true,
  "promotion_allowed": false
}
```

`promotion_allowed=true` requires all verifier, predicate, patch, baseline, and cost gates above. Until then, the row is
planning metadata or diagnostic evidence only.
