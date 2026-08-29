from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET

from omniflow.core.trajectory import require_complete_source_run_log, state_id
from omniflow.functions.management import (
    parameter_candidates,
    semantic_parameter_evidence,
)
from omniflow.runlog import project_run_log_step_actions
from omniflow.runtime.checker import (
    DEFAULT_CHECKER_LIBRARY_PATH,
    CheckerLibrary,
    validate_checker_rule,
)


def compile_runlog_to_store(
    run_log: str | Path | dict[str, Any],
    output_root: str | Path,
    *,
    function_bundle: dict[str, Any] | None = None,
    model: str | None = None,
    client: Any | None = None,
    prompt: str | None = None,
    timeout: float = 120.0,
    source_states: str | Path | dict[str, Any] | None = None,
    state_loader: Any | None = None,
) -> dict[str, Any]:
    """Register strict v2 Functions and their referenced source states."""
    from omniflow.functions.artifact import bind_function, parse_function_artifact
    from omniflow.functions.store import FunctionStore
    from omniflow.transfer.runtime import (
        TRANSFER_STATE_CATALOG_FILENAME,
        TRANSFER_STATE_CATALOG_VERSION,
        load_transfer_state_catalog,
    )

    root = Path(output_root).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"immutable output version already exists: {root}")

    if isinstance(run_log, dict):
        raw = dict(run_log)
    else:
        value = json.loads(Path(run_log).expanduser().resolve().read_text())
        if not isinstance(value, dict):
            raise ValueError("source_runlog_must_be_object")
        raw = value
    payload = require_complete_source_run_log(raw)
    goal = str(payload.get("goal") or "").strip()
    if not goal:
        raise ValueError("successful_source_goal_required")

    steps: list[dict[str, Any]] = []
    parameter_evidence: list[dict[str, Any]] = []
    omitted_action_types: set[str] = set()
    optional_checker_actions: list[dict[str, Any]] = []
    previous_successful_step: dict[str, Any] | None = None
    source_steps = payload["steps"]
    for source_step_index, step in enumerate(source_steps):
        if not isinstance(step, dict):
            continue
        metadata = step.get("metadata") if isinstance(step.get("metadata"), dict) else {}
        result = step.get("result") if isinstance(step.get("result"), dict) else {}
        if result.get("success") is not True:
            continue
        observation = step["observation"]
        before_state_id = state_id(observation)
        next_observation = step.get("next_observation")
        if (
            isinstance(next_observation, dict)
            and state_id(next_observation) == before_state_id
            and source_step_index + 1 < len(source_steps)
        ):
            following_observation = source_steps[source_step_index + 1].get(
                "observation"
            )
            if (
                isinstance(following_observation, dict)
                and state_id(following_observation) != before_state_id
            ):
                next_observation = following_observation
        after_state_id = state_id(
            next_observation
            if isinstance(next_observation, dict)
            else observation
        )
        action_type = str(step.get("action", {}).get("action_type") or "")
        if action_type in {"answer", "status", "unknown"}:
            omitted_action_types.add(action_type)
            continue
        if _is_transient_system_action(step):
            # Permission prompts are optional setup, not task progress. Keep
            # them in the shared dismiss_permission_dialog checker instead of
            # replaying a stale dialog click in the main Function.
            omitted_action_types.add("checker")
            optional_checker_actions.append(
                {
                    "source_step_index": source_step_index,
                    "checker_id": "dismiss_permission_dialog",
                    "reason": "permission_dialog_setup_is_optional",
                }
            )
            continue
        projected_actions = project_run_log_step_actions(
            payload,
            source_step_index,
            previous_step=previous_successful_step,
            next_observation=next_observation,
        )
        promoted_launcher_entry = False
        if not steps and projected_actions:
            original_entry_action = projected_actions[0]
            projected_actions[0] = _promote_launcher_app_entry(
                projected_actions[0],
                observation=observation,
                next_observation=next_observation,
            )
            promoted_launcher_entry = projected_actions[0] != original_entry_action
        previous_successful_step = step
        action_metadata = {
            key: metadata[key]
            for key in ("summary", "action_description")
            if str(metadata.get(key) or "").strip()
        }
        purpose = str(
            metadata.get("action_description")
            or metadata.get("summary")
            or metadata.get("reasoning")
            or ""
        ).strip()
        if purpose:
            action_metadata["purpose"] = purpose
        if promoted_launcher_entry:
            action_metadata["fixed_open_app_package"] = True
        for action in projected_actions:
            action = _canonicalize_open_app_action(
                action,
                next_observation=next_observation,
            )
            parameter_evidence.extend(
                _source_action_parameter_evidence(
                    source_step=step,
                    action=action,
                    source_step_index=source_step_index,
                    task_parameters=payload.get("task_parameters"),
                )
            )
            steps.append(
                {
                    "source_step_index": source_step_index,
                    "before_state_id": before_state_id,
                    "action": action,
                    "result": {"success": True},
                    "after_state_id": after_state_id,
                    "metadata": action_metadata,
                }
            )
    if not steps:
        raise ValueError("successful_source_actions_required")
    facts = {
        "schema_version": "omniflow.function-compilation-facts.v2",
        "run_id": str(payload.get("run_id") or "successful-source"),
        "goal": goal,
        "task_parameters": json.loads(
            json.dumps(payload.get("task_parameters") or {}, ensure_ascii=False)
        ),
        "status": "succeeded",
        "success": True,
        "steps": steps,
        "parameter_evidence": parameter_evidence,
        "omitted_action_types": sorted(omitted_action_types),
        "optional_checker_actions": optional_checker_actions,
    }
    source_parameter_candidates = _source_parameter_candidates(facts)
    authoring_prompt = prompt or """Extract contiguous reusable Function segments from a successful GUI source flow.
Return exactly one object with this shape:
{"reason":"account for every source step and explain the composition","plan":{"functions":[{"function_id":"enter_requested_name","name":"Enter requested name","description":"Fill the requested name and submit it so the form reaches its completed state.","source_step_indices":[6,7],"parameters":[{"name":"name","description":"Name requested by the user","source_step_index":6,"arg_name":"text"}]}],"complete_function":{"function_id":"complete_form","name":"Complete form","description":"Complete the form by entering the requested name and submitting it; the final submit action produces the requested completed form.","source_step_indices":[6,7],"parameters":[{"name":"name","description":"Name requested by the user","source_step_index":6,"arg_name":"text"}]}}}

Do not output input_schema, bindings, steps, actions, coordinates, checker rules,
agent_visible, schema_version, arguments, or source_state_id. The compiler owns
all of them and materializes canonical omniflow.function.v2 artifacts from the
selected immutable source actions.

Inspect source_run in source_step_index order. In functions, return zero or more
strictly contiguous local segments. Never delete, reorder, or skip an action inside
a selected segment. Then return exactly one complete_function whose
source_step_indices exactly equal the entire successful source action sequence in
order after optional checker actions have been removed. The complete Function cannot
omit, split, truncate, or rewrite any remaining main-flow action. Actions already
extracted as checker recovery are not part of source_run and must not be reconstructed
in the main flow.

The source_run includes optional_checker_actions when the compiler has identified
setup such as a permission dialog. Treat those source steps as checker-owned setup:
do not select them in any Function. The shared checker library already provides the
corresponding recovery, and the main workflow must remain valid when those UI states
are absent.

Each source step's metadata.purpose explains what the action accomplishes. Use it
only to name and describe the selected segment. The compiler copies every action,
source_state_id, and source point exactly; the model never emits or edits them.
Function matching uses the recorded source state and source point with canonical
OmniTransfer. Do not invent semantic anchors, target descriptions, selectors, target
coordinates, direct coordinate replay, observation boundaries, or handoff rules.

Parameterize only entries copied exactly from parameter_candidates and selected by
the same Function. Only goal-dependent app package names and input text values are
eligible. Coordinates, labels, repetition counts, target descriptions, and derived
target-side values are never Function parameters. The complete_function must repeat
every parameter target selected by a local Function. Use parameters=[] for fixed
recorded values. Keep reason under 40 words, each description under 120 words, and
return no prose outside the JSON object.
Treat every outcome-affecting recorded choice as a hard applicability constraint.
State fixed modes, types, formats, categories, destinations, and similar choices in
the Function description whenever changing them could change the task result. If a
parameter_candidate has different recorded_value and task_parameter_value, describe
the exact value accepted by the recorded input action and preserve any fixed prefix,
suffix, type, or format in both the parameter and complete Function descriptions.
Never claim that a Function covers goals outside those recorded fixed constraints.
For a short linear workflow whose successful actions already form one stable
end-to-end sequence, use functions=[] and author only the complete_function. Do not
duplicate the complete workflow into local Functions by default. Author a local
Function only when the evidence shows a separately reusable contiguous capability or
an observation-dependent breakpoint that cannot safely stay in the complete flow.
When a value is selected as a parameter, remove its recorded instance literal from
the Function name and description. Describe the requested semantic value and let the
generated input schema carry the concrete value at call time.
Use suggested_name exactly when a candidate provides it. Give distinct semantic
values distinct parameter names. Reuse one parameter name across multiple targets
only when those targets intentionally consume the same recorded semantic value.

For a global Function whose first action is open_app, keep the canonical recorded
package fixed when the goal identifies a concrete app such as Joplin or Settings.
Only expose package_name when its parameter_candidate includes the compiler-backed
task_parameter_name="app_name" and value_contract="android_package_name" evidence.
In that case name the parameter package_name, describe it as the installed Android
package for the requested app, and make the Function name and description generic to
the requested app rather than the recorded source app. A model must never put a
friendly app label such as "clock" into package_name. Once package_name is selected,
the function_id, Function name, Function description, and parameter description must
not contain the recorded app label or package; use only "requested app" wording.
"""
    selected_model = str(model or "").strip() or None
    usage = {
        "model_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    if function_bundle is not None:
        if selected_model is not None or client is not None or prompt is not None:
            raise ValueError("function_bundle_cannot_use_author_model_options")
        authored = {
            "reason": "Registered the supplied Function Schema.",
            "bundle": json.loads(json.dumps(function_bundle, ensure_ascii=False)),
        }
    elif selected_model is None:
        if client is not None or prompt is not None:
            raise ValueError("author_model_required_for_author_options")
        raise ValueError("function_author_model_or_bundle_required")
    else:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install omniflow[llm] to compile RunLogs") from exc
            options: dict[str, Any] = {
                "api_key": os.getenv("OPENAI_API_KEY") or "not-required",
                "max_retries": 0,
            }
            if os.getenv("OPENAI_BASE_URL"):
                options["base_url"] = os.environ["OPENAI_BASE_URL"]
            client = OpenAI(**options)
        response = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": authoring_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "source_run": facts,
                            "parameter_candidates": source_parameter_candidates,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=8192,
            temperature=0,
            stream=False,
            reasoning_effort="none",
            extra_body={
                "enable_thinking": False,
                "thinking": {"type": "disabled"},
            },
            timeout=float(timeout),
        )
        response_usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(response_usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(response_usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(response_usage, "total_tokens", 0) or 0)
        usage = {
            "model_calls": 1,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens or prompt_tokens + completion_tokens,
        }
        raw_author_response = str(response.choices[0].message.content or "")
        try:
            proposal = json.loads(raw_author_response)
            try:
                authored = _materialize_authoring_plan(proposal, facts)
            except ValueError as error:
                # The authoring model may describe a valid local segment but
                # omit or truncate the required complete source sequence. A
                # successful source RunLog is already the immutable evidence;
                # preserve it verbatim through the compiler's strict fallback
                # instead of making collection depend on a second model retry.
                if str(error) != "function_author_plan_complete_sequence_required":
                    raise
                authored = _materialize_authoring_plan(
                    _complete_source_authoring_plan(facts), facts
                )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            _write_authoring_failure(
                root,
                error=error,
                model=selected_model,
                prompt=authoring_prompt,
                response=raw_author_response,
                usage=usage,
            )
            raise
    if not isinstance(authored, dict) or set(authored) != {"reason", "bundle"}:
        raise ValueError("function_author_response_contract_invalid")
    if not isinstance(authored["reason"], str):
        raise ValueError("function_author_reason_must_be_string")

    bundle = authored["bundle"]
    if bundle is None:
        raise ValueError("functions_required")
    if not isinstance(bundle, dict):
        raise ValueError("function_author_bundle_must_be_object_or_null")
    if set(bundle) != {
        "schema_version",
        "run_id",
        "arguments",
        "functions",
        "checker_rules",
    }:
        raise ValueError("function_bundle_contract_invalid")
    if bundle.get("schema_version") != "omniflow.function-bundle.v2":
        raise ValueError("unsupported_function_bundle_version")
    if str(bundle.get("run_id") or "") != facts["run_id"]:
        raise ValueError("function_bundle_run_id_mismatch")
    raw_functions = bundle.get("functions")
    arguments_by_function = bundle.get("arguments")
    raw_checker_rules = bundle.get("checker_rules")
    if not isinstance(raw_functions, list) or not raw_functions:
        raise ValueError("function_bundle_functions_required")
    if not isinstance(arguments_by_function, dict):
        raise ValueError("function_bundle_source_arguments_invalid")
    if not isinstance(raw_checker_rules, list):
        raise ValueError("function_bundle_checker_rules_invalid")
    functions = [parse_function_artifact(value) for value in raw_functions]
    checker_rules = [validate_checker_rule(rule) for rule in raw_checker_rules]
    checker_ids = [rule["id"] for rule in checker_rules]
    if len(checker_ids) != len(set(checker_ids)):
        raise ValueError("function_bundle_duplicate_checker_id")
    shared_checker_rules = CheckerLibrary.load().rules
    shared_checker_ids = {rule["id"] for rule in shared_checker_rules}
    unknown_checker_ids = sorted(set(checker_ids) - shared_checker_ids)
    if unknown_checker_ids:
        raise ValueError(
            "function_bundle_checker_not_in_shared_library:"
            + ",".join(unknown_checker_ids)
        )
    authored_function_ids = [function.id for function in functions]
    if len(authored_function_ids) != len(set(authored_function_ids)):
        raise ValueError("function_bundle_duplicate_function_id")
    if set(arguments_by_function) - set(authored_function_ids):
        raise ValueError("function_bundle_source_arguments_unknown_function")
    normalized_arguments: dict[str, dict[str, Any]] = {}
    for function in functions:
        arguments = arguments_by_function.get(function.id, {})
        if not isinstance(arguments, dict):
            raise ValueError("function_bundle_source_arguments_invalid")
        bind_function(function, arguments)
        normalized_arguments[function.id] = dict(arguments)
    arguments_by_function = normalized_arguments

    function_ids = [function.id for function in functions]

    if source_states is not None and state_loader is not None:
        raise ValueError("function_source_state_provider_ambiguous")
    referenced_state_ids = _referenced_source_state_ids(functions)
    states: dict[str, dict[str, Any]]
    source_catalog_run_id = facts["run_id"]
    if source_states is not None:
        if isinstance(source_states, (str, Path)):
            source_catalog_path = Path(source_states).expanduser().resolve()
            raw_catalog = json.loads(source_catalog_path.read_text(encoding="utf-8"))
            if not isinstance(raw_catalog, dict):
                raise ValueError("function_source_state_catalog_invalid")
            source_catalog_run_id = str(raw_catalog.get("run_id") or "").strip()
            states = load_transfer_state_catalog(source_catalog_path)
        elif isinstance(source_states, dict):
            raw_states = source_states.get("states")
            if raw_states is None:
                raw_states = source_states
            elif source_states.get("schema_version") != (
                TRANSFER_STATE_CATALOG_VERSION
            ):
                raise ValueError("function_source_state_catalog_invalid")
            if not isinstance(raw_states, dict):
                raise ValueError("function_source_state_catalog_invalid")
            source_catalog_run_id = str(
                source_states.get("run_id") or facts["run_id"]
            ).strip()
            states = {
                str(state_id): _normalize_source_state(value, str(state_id))
                for state_id, value in raw_states.items()
            }
        else:
            raise ValueError("function_source_state_catalog_invalid")
    elif callable(state_loader):
        states = {
            state_id: _normalize_source_state(state_loader(state_id), state_id)
            for state_id in referenced_state_ids
        }
    else:
        raise ValueError("function_source_states_required")
    if source_catalog_run_id != facts["run_id"]:
        raise ValueError("function_source_state_run_id_mismatch")
    missing_state_ids = [
        state_id for state_id in referenced_state_ids if state_id not in states
    ]
    if missing_state_ids:
        raise ValueError(
            "function_source_states_missing:" + ",".join(missing_state_ids)
        )
    frozen_states = {
        state_id: states[state_id] for state_id in referenced_state_ids
    }

    root.mkdir(parents=True, exist_ok=True)
    store_path = root / "store.json"
    store = FunctionStore(store_path)
    for function in functions:
        store.put_function(function)
    transfer_state_catalog_path = root / TRANSFER_STATE_CATALOG_FILENAME
    transfer_state_catalog_path.write_text(
        json.dumps(
            {
                "schema_version": TRANSFER_STATE_CATALOG_VERSION,
                "run_id": facts["run_id"],
                "states": frozen_states,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": "omniflow.androidworld.function-gate.v2",
        "success": True,
        "live_probe_allowed": True,
        "classification": "ready_for_live_probe",
        "reason": authored["reason"],
        "model": selected_model,
        "prompt_sha256": (
            hashlib.sha256(authoring_prompt.encode()).hexdigest()
            if selected_model is not None
            else None
        ),
        "store_path": str(store_path),
        "checker_store_path": str(DEFAULT_CHECKER_LIBRARY_PATH),
        "checker_count": len(shared_checker_rules),
        "optional_checker_actions": optional_checker_actions,
        "transfer_state_catalog": str(transfer_state_catalog_path),
        "transfer_state_count": len(frozen_states),
        "function_ids": function_ids,
        "function_count": len(function_ids),
        "source_calls": [
            {
                "function_id": function_id,
                "arguments": json.loads(
                    json.dumps(arguments_by_function[function_id], ensure_ascii=False)
                ),
            }
            for function_id in function_ids
        ],
        "source_arguments": json.loads(
            json.dumps(arguments_by_function, ensure_ascii=False)
        ),
        **usage,
    }
    (root / "compile_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    return report


def _source_parameter_candidates(facts: dict[str, Any]) -> list[dict[str, Any]]:
    fact_steps = [step for step in facts.get("steps") or () if isinstance(step, dict)]
    function_view = {
        "bindings": [],
        "steps": [
            {
                "step_index": index,
                "source_step_index": step.get("source_step_index", index),
                "action": step["action"],
                "metadata": step.get("metadata") or {},
            }
            for index, step in enumerate(fact_steps)
        ],
    }
    evidence_by_target = {
        (
            evidence.get("source_step_index"),
            str(evidence.get("arg_name") or "").strip(),
        ): evidence
        for evidence in facts.get("parameter_evidence") or ()
        if isinstance(evidence, dict)
    }
    candidates = []
    for candidate in parameter_candidates(function_view):
        local_step_index = int(candidate["step_index"])
        source_step_index = int(
            fact_steps[local_step_index].get(
                "source_step_index", candidate["step_index"]
            )
        )
        semantic_evidence = semantic_parameter_evidence(
            function_view["steps"][local_step_index],
            str(candidate["arg_name"]),
            candidate["recorded_value"],
            facts,
        )
        if semantic_evidence is None:
            continue
        value = {
            "source_step_index": source_step_index,
            "tool": candidate["tool"],
            "arg_name": candidate["arg_name"],
            "recorded_value": candidate["recorded_value"],
            **semantic_evidence,
        }
        evidence = evidence_by_target.get(
            (source_step_index, candidate["arg_name"])
        )
        if evidence is not None:
            value.update(
                {
                    "task_parameter_name": evidence["task_parameter_name"],
                    "task_parameter_value": evidence["task_parameter_value"],
                    "value_contract": evidence["value_contract"],
                }
            )
        candidates.append(value)
    return candidates


def _complete_source_authoring_plan(facts: dict[str, Any]) -> dict[str, Any]:
    """Build the minimal valid plan for the complete successful source flow."""

    source_indices = [
        int(step["source_step_index"])
        for step in facts.get("steps") or ()
        if isinstance(step, dict) and isinstance(step.get("source_step_index"), int)
    ]
    if not source_indices:
        raise ValueError("successful_source_actions_required")
    goal_summary = " ".join(str(facts.get("goal") or "").split())[:240]
    name = f"Complete task: {goal_summary}" if goal_summary else "Complete recorded workflow"
    description = (
        f"Complete the requested task: {goal_summary}. Execute the successful "
        "source workflow in order."
        if goal_summary
        else "Execute the successful source workflow in order."
    )
    return {
        "reason": "Compiler fallback for a complete successful source workflow.",
        "plan": {
            "functions": [],
            "complete_function": {
                "function_id": "complete_source_workflow",
                "name": name,
                "description": description,
                "source_step_indices": source_indices,
                "parameters": [],
            },
        },
    }


def _direct_source_authoring_plan(facts: dict[str, Any]) -> dict[str, Any]:
    source_steps = [
        step for step in facts.get("steps") or () if isinstance(step, dict)
    ]
    if not source_steps:
        raise ValueError("successful_source_actions_required")
    goal = " ".join(str(facts.get("goal") or "").split())
    function_id = "complete_source_workflow"
    name = f"Complete task: {goal}" if goal else "Complete source workflow"
    description = (
        f"Complete the requested task: {goal}. Execute every converted source "
        "action in order."
        if goal
        else "Execute every converted source action in order."
    )
    function = {
        "schema_version": "omniflow.function.v2",
        "function_id": function_id,
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "bindings": [],
        "steps": [
            {
                "step_index": index,
                "source_state_id": str(step["before_state_id"]),
                "action": json.loads(json.dumps(step["action"], ensure_ascii=False)),
            }
            for index, step in enumerate(source_steps)
        ],
        "agent_visible": True,
    }
    return {
        "reason": "Registered the complete converted source workflow.",
        "bundle": {
            "schema_version": "omniflow.function-bundle.v2",
            "run_id": str(facts["run_id"]),
            "arguments": {function_id: {}},
            "checker_rules": [],
            "functions": [function],
        },
    }


def _materialize_authoring_plan(
    value: Any,
    facts: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"reason", "plan"}:
        raise ValueError("function_author_plan_response_contract_invalid")
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("function_author_reason_must_be_string")
    plan = value.get("plan")
    if not isinstance(plan, dict) or set(plan) != {
        "functions",
        "complete_function",
    }:
        raise ValueError("function_author_plan_contract_invalid")
    raw_functions = plan.get("functions")
    raw_complete_function = plan.get("complete_function")
    if not isinstance(raw_functions, list):
        raise ValueError("function_author_plan_functions_required")
    fallback_complete_function = raw_complete_function is None or (
        not raw_functions
        and isinstance(raw_complete_function, dict)
        and raw_complete_function.get("source_step_indices") == []
    )
    if fallback_complete_function:
        source_steps = [
            step
            for step in facts.get("steps") or ()
            if isinstance(step, dict)
            and isinstance(step.get("source_step_index"), int)
        ]
        if not source_steps:
            raise ValueError("function_author_plan_complete_function_required")
        goal_summary = " ".join(str(facts.get("goal") or "").split())[:240]
        raw_complete_function = {
            "function_id": "complete_source_workflow",
            "name": f"Complete task: {goal_summary}" if goal_summary else "Complete recorded workflow",
            "description": (
                f"Complete the requested task: {goal_summary}. "
                "Execute the successful source workflow in order."
                if goal_summary
                else "Execute the successful source workflow in order."
            ),
            "source_step_indices": [
                int(step["source_step_index"]) for step in source_steps
            ],
            "parameters": [],
        }
    elif not isinstance(raw_complete_function, dict):
        raise ValueError("function_author_plan_complete_function_required")

    source_steps = list(facts.get("steps") or ())
    source_step_positions = {
        int(step.get("source_step_index", position)): position
        for position, step in enumerate(source_steps)
        if isinstance(step, dict)
    }
    candidates = {
        (candidate["source_step_index"], candidate["arg_name"]): candidate
        for candidate in _source_parameter_candidates(facts)
    }
    functions: list[dict[str, Any]] = []
    materialized_function_ids: set[str] = set()
    arguments: dict[str, dict[str, Any]] = {}
    materialization_notes: list[str] = []
    from omniflow.core.schemas import load_canonical_action_schema

    builtin_tool_names = {
        str(tool.get("name") or "").strip()
        for tool in load_canonical_action_schema().get("tools") or ()
        if isinstance(tool, dict)
    }
    if fallback_complete_function:
        materialization_notes.append(
            "Compiler supplied a complete Function from the successful source action sequence because authoring omitted complete_function."
        )
    function_fields = {
        "function_id",
        "name",
        "description",
        "source_step_indices",
        "parameters",
    }
    parameter_fields = {
        "name",
        "description",
        "source_step_index",
        "arg_name",
    }
    complete_function_id = str(raw_complete_function.get("function_id") or "").strip()
    complete_source_indices = raw_complete_function.get("source_step_indices")
    deduplicated_functions: list[Any] = []
    for raw_function in raw_functions:
        if isinstance(raw_function, dict) and (
            str(raw_function.get("function_id") or "").strip()
            == complete_function_id
            or raw_function.get("source_step_indices") == complete_source_indices
        ):
            materialization_notes.append(
                "Compiler omitted a redundant local Function that duplicated "
                "the complete Function."
            )
            continue
        deduplicated_functions.append(raw_function)
    planned_functions = [
        *((raw_function, False) for raw_function in deduplicated_functions),
        (raw_complete_function, True),
    ]
    for raw_function, is_complete in planned_functions:
        if not isinstance(raw_function, dict) or set(raw_function) != function_fields:
            raise ValueError("function_author_plan_function_contract_invalid")
        requested_function_id = str(raw_function.get("function_id") or "").strip()
        function_id = requested_function_id
        if function_id in builtin_tool_names:
            base_function_id = (
                "complete_source_workflow"
                if is_complete
                else "reusable_source_segment"
            )
            function_id = base_function_id
            suffix = 2
            while function_id in materialized_function_ids:
                function_id = f"{base_function_id}_{suffix}"
                suffix += 1
            materialization_notes.append(
                "Compiler renamed the reserved Function id "
                f"{requested_function_id} to {function_id}; the source Action "
                "tool and arguments remain unchanged."
            )
        name = str(raw_function.get("name") or "").strip()
        description = str(raw_function.get("description") or "").strip()
        raw_indices = raw_function.get("source_step_indices")
        if (
            not function_id
            or not name
            or not description
            or not isinstance(raw_indices, list)
            or not raw_indices
            or any(isinstance(index, bool) or not isinstance(index, int) for index in raw_indices)
        ):
            raise ValueError("function_author_plan_function_invalid")
        indices = list(raw_indices)
        if indices != sorted(set(indices)) or any(
            index not in source_step_positions for index in indices
        ):
            raise ValueError("function_author_plan_source_steps_invalid")
        source_positions = [source_step_positions[index] for index in indices]
        if source_positions != list(
            range(source_positions[0], source_positions[-1] + 1)
        ):
            raise ValueError("function_author_plan_steps_not_contiguous")
        complete_indices = list(source_step_positions)
        if is_complete and indices != complete_indices:
            raise ValueError("function_author_plan_complete_sequence_required")
        if is_complete:
            materialization_notes.append(
                "Compiler materialized the complete successful source sequence."
            )
        function = {
            "schema_version": "omniflow.function.v2",
            "function_id": function_id,
            "name": name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "bindings": [],
            "steps": [
                {
                    "step_index": local_index,
                    "source_state_id": str(
                        source_steps[source_step_positions[source_index]][
                            "before_state_id"
                        ]
                    ),
                    "action": json.loads(
                        json.dumps(
                            source_steps[source_step_positions[source_index]][
                                "action"
                            ],
                            ensure_ascii=False,
                        )
                    ),
                }
                for local_index, source_index in enumerate(indices)
            ],
            "agent_visible": True,
        }
        raw_parameters = raw_function.get("parameters")
        if not isinstance(raw_parameters, list):
            raise ValueError("function_author_plan_parameters_invalid")
        parameter_proposals: list[dict[str, Any]] = []
        source_arguments: dict[str, Any] = {}
        selected_parameter_targets: set[tuple[int, str]] = set()
        for parameter in raw_parameters:
            if not isinstance(parameter, dict) or set(parameter) != parameter_fields:
                raise ValueError("function_author_plan_parameter_contract_invalid")
            source_index = parameter.get("source_step_index")
            arg_name = str(parameter.get("arg_name") or "").strip()
            candidate = candidates.get((source_index, arg_name))
            if candidate is None:
                raise ValueError("function_author_plan_parameter_target_invalid")
            if source_index not in indices:
                raise ValueError("function_author_plan_parameter_step_not_selected")
            target_key = (int(source_index), arg_name)
            if target_key in selected_parameter_targets:
                continue
            parameter_name = str(parameter.get("name") or "").strip()
            parameter_proposals.append(
                {
                    "name": parameter_name,
                    "description": str(parameter.get("description") or "").strip(),
                    "step_index": indices.index(source_index),
                    "arg_name": arg_name,
                }
            )
            source_arguments[parameter_name] = candidate["recorded_value"]
            selected_parameter_targets.add(target_key)
        selected_targets = {
            (
                indices[int(item["step_index"])]
                if isinstance(item.get("step_index"), int)
                and 0 <= int(item["step_index"]) < len(indices)
                else None,
                str(item.get("arg_name") or ""),
            )
            for item in parameter_proposals
        }
        input_value_names: dict[str, str] = {}
        used_parameter_names = {
            str(item.get("name") or "").strip()
            for item in parameter_proposals
            if str(item.get("name") or "").strip()
        }
        for source_index in indices:
            candidate = next(
                (
                    item
                    for (candidate_source_index, candidate_arg_name), item in candidates.items()
                    if candidate_source_index == source_index
                    and candidate_arg_name == "text"
                    and item.get("tool") == "input_text"
                ),
                None,
            )
            target_key = (source_index, "text")
            if candidate is None or target_key in selected_targets:
                continue
            normalized_value = " ".join(
                str(candidate.get("recorded_value") or "").casefold().split()
            )
            parameter_name = input_value_names.get(normalized_value)
            if parameter_name is None:
                parameter_name = str(candidate.get("suggested_name") or "").strip()
                if not parameter_name or parameter_name in used_parameter_names:
                    suffix = 1
                    parameter_name = "input_text"
                    while parameter_name in used_parameter_names:
                        suffix += 1
                        parameter_name = f"input_text_{suffix}"
                input_value_names[normalized_value] = parameter_name
                used_parameter_names.add(parameter_name)
            parameter_proposals.append(
                {
                    "name": parameter_name,
                    "description": "The requested value for this text field",
                    "step_index": indices.index(source_index),
                    "arg_name": "text",
                }
            )
            source_arguments[parameter_name] = candidate["recorded_value"]
        preferred_value_names = {
            " ".join(str(item.get("recorded_value") or "").casefold().split()): str(
                item["suggested_name"]
            ).strip()
            for item in candidates.values()
            if item.get("evidence") == "task_parameter_exact_value"
            and str(item.get("suggested_name") or "").strip()
        }
        if preferred_value_names:
            for proposal in parameter_proposals:
                proposal_step = proposal.get("step_index")
                proposal_arg = str(proposal.get("arg_name") or "")
                if not isinstance(proposal_step, int) or not proposal_arg:
                    continue
                source_index = indices[proposal_step]
                candidate = candidates.get((source_index, proposal_arg))
                if candidate is None:
                    continue
                preferred_name = preferred_value_names.get(
                    " ".join(
                        str(candidate.get("recorded_value") or "").casefold().split()
                    )
                )
                if preferred_name:
                    proposal["name"] = preferred_name
            source_arguments = {}
            for proposal in parameter_proposals:
                proposal_step = proposal.get("step_index")
                proposal_arg = str(proposal.get("arg_name") or "")
                if not isinstance(proposal_step, int) or not proposal_arg:
                    continue
                candidate = candidates.get((indices[proposal_step], proposal_arg))
                if candidate is not None:
                    source_arguments[str(proposal["name"])] = candidate[
                        "recorded_value"
                    ]
        _materialize_agent_parameters(function, parameter_proposals)
        function["description"] = _description_with_action_plan(
            description,
            function=function,
            source_steps=source_steps,
            source_indices=indices,
            source_step_positions=source_step_positions,
        )
        if function_id in materialized_function_ids:
            if not is_complete:
                raise ValueError("function_author_plan_duplicate_function_id")
            functions = [
                item
                for item in functions
                if item.get("function_id") != function_id
            ]
            arguments.pop(function_id, None)
            materialized_function_ids.remove(function_id)
            materialization_notes.append(
                "Compiler kept the complete Function envelope when a local Function reused its id."
            )
        materialized_function_ids.add(function_id)
        functions.append(function)
        arguments[function_id] = source_arguments

    return {
        "reason": " ".join([reason.strip(), *materialization_notes]),
        "bundle": {
            "schema_version": "omniflow.function-bundle.v2",
            "run_id": str(facts["run_id"]),
            "arguments": arguments,
            "checker_rules": [],
            "functions": functions,
        },
    }


def _is_transient_system_action(step: dict[str, Any]) -> bool:
    action = step.get("action") if isinstance(step.get("action"), dict) else {}
    if str(action.get("action_type") or "").strip() != "click":
        return False
    observation = step.get("observation")
    if not isinstance(observation, dict):
        return False
    xml = str(observation.get("xml") or observation.get("forest") or "").strip()
    if not xml:
        return False
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return False
    packages = {
        str(node.get("package") or "").casefold()
        for node in root.iter()
        if str(node.get("package") or "").strip()
    }
    permission_page = any(
        "permissioncontroller" in package or package == "android.permission"
        for package in packages
    )
    if not permission_page:
        # Android's intent resolver is also one-shot setup.  A source episode
        # can capture an "Open with ... / Just once" chooser before the real
        # app page; that click must be supplied by the shared checker when the
        # chooser exists, not baked into every Function replay.
        normalized_xml = " ".join(xml.casefold().split())
        chooser_markers = (
            "open with",
            "just once",
            "always",
            "use a different app",
        )
        if not (
            "com.android.systemui" in packages
            and any(marker in normalized_xml for marker in chooser_markers)
        ):
            return False
        try:
            float(action.get("x"))
            float(action.get("y"))
        except (TypeError, ValueError):
            return False
        return True
    try:
        x = float(action.get("x"))
        y = float(action.get("y"))
    except (TypeError, ValueError):
        return False
    # Some AndroidWorld captures have a stale/misaligned click point for the
    # permission dialog.  The dialog itself is still unambiguously identified
    # by its package/resource, so any click inside that modal is setup-owned.
    if any(
        "grant_dialog" in str(node.get("resource-id") or "").casefold()
        and (bounds := _parse_bounds(node.get("bounds"))) is not None
        and bounds[0] <= x <= bounds[2]
        and bounds[1] <= y <= bounds[3]
        for node in root.iter("node")
    ):
        return True
    labels = {
        "allow",
        "allow all the time",
        "allow only while using the app",
        "only this time",
        "don't allow",
        "close app",
        "app info",
        "wait",
    }
    for node in root.iter("node"):
        bounds = _parse_bounds(node.get("bounds"))
        if bounds is None or not (bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]):
            continue
        node_labels = {
            " ".join(str(node.get(attribute) or "").casefold().split())
            for attribute in ("text", "content-desc")
        }
        if node_labels.intersection(labels):
            return True
    return False


def _parse_bounds(value: Any) -> tuple[float, float, float, float] | None:
    match = re.fullmatch(
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
        r"\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
        str(value or "").strip(),
    )
    if match is None:
        return None
    return tuple(float(item) for item in match.groups())


def _description_with_action_plan(
    description: str,
    *,
    function: dict[str, Any],
    source_steps: list[dict[str, Any]],
    source_indices: list[int],
    source_step_positions: dict[int, int],
) -> str:
    parameter_by_target = {
        str(binding["target"]): str(binding["source"]).removeprefix("$.arguments.")
        for binding in function.get("bindings") or ()
        if isinstance(binding, dict)
    }
    plan: list[str] = []
    for local_index, step in enumerate(function.get("steps") or ()):
        action = step.get("action") if isinstance(step, dict) else None
        if not isinstance(action, dict):
            continue
        args = action.get("args")
        semantic_args: dict[str, Any] = {}
        if isinstance(args, dict):
            for arg_name, arg_value in args.items():
                if arg_name in {"x", "y", "x1", "y1", "x2", "y2"}:
                    continue
                target = f"$.steps[{local_index}].action.args.{arg_name}"
                parameter_name = parameter_by_target.get(target)
                semantic_args[arg_name] = (
                    f"<{parameter_name}>" if parameter_name else arg_value
                )
        tool = str(action.get("tool") or "")
        encoded_args = ",".join(
            f"{name}={json.dumps(value, ensure_ascii=False, separators=(',', ':'))}"
            for name, value in semantic_args.items()
        )
        entry = f"{local_index + 1}:{tool}"
        if encoded_args:
            entry += f"({encoded_args})"
        source_position = source_step_positions[source_indices[local_index]]
        metadata = source_steps[source_position].get("metadata")
        purpose = ""
        if isinstance(metadata, dict):
            purpose = str(
                metadata.get("purpose")
                or metadata.get("action_description")
                or metadata.get("summary")
                or ""
            ).strip()
        if (
            purpose
            and not purpose.startswith("Manually reviewed recorded ")
            and (tool == "swipe" or not semantic_args)
        ):
            entry += f"[{purpose}]"
        plan.append(entry)
    semantic_description = str(description or "").split(" Action plan:", 1)[0].strip()
    encoded_plan = ";".join(plan)
    return f"{semantic_description} Action plan: {encoded_plan}"


def _materialize_agent_parameters(
    function: dict[str, Any],
    parameters: list[dict[str, Any]],
) -> None:
    candidates = {
        (candidate["step_index"], candidate["arg_name"]): candidate
        for candidate in parameter_candidates(function)
    }
    values_by_name: dict[str, str] = {}
    for parameter in parameters:
        name = str(parameter.get("name") or "").strip()
        description = str(parameter.get("description") or "").strip()[:240]
        step_index = parameter.get("step_index")
        arg_name = str(parameter.get("arg_name") or "").strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name) is None:
            raise ValueError("function_author_plan_parameter_name_invalid")
        candidate = candidates.get((step_index, arg_name))
        if candidate is None:
            raise ValueError("function_author_plan_parameter_target_invalid")
        recorded_value = str(candidate["recorded_value"])
        if name in values_by_name and values_by_name[name] != recorded_value:
            raise ValueError("function_author_plan_parameter_name_ambiguous")
        if name not in function["input_schema"]["properties"]:
            definition = {"type": "string"}
            if description:
                definition["description"] = description
            function["input_schema"]["properties"][name] = definition
            function["input_schema"]["required"].append(name)
        binding_steps = [step_index]
        if (
            arg_name == "text"
            and function["steps"][step_index]["action"].get("tool")
            == "input_text"
        ):
            normalized_recorded_value = " ".join(recorded_value.casefold().split())
            bound_targets = {
                str(binding.get("target") or "")
                for binding in function.get("bindings") or ()
                if isinstance(binding, dict)
            }
            for repeated_step_index, repeated_step in enumerate(function["steps"]):
                if repeated_step_index == step_index:
                    continue
                repeated_action = repeated_step.get("action")
                if not isinstance(repeated_action, dict):
                    continue
                if repeated_action.get("tool") != "input_text":
                    continue
                repeated_target = (
                    f"$.steps[{repeated_step_index}].action.args.text"
                )
                if repeated_target in bound_targets:
                    continue
                repeated_value = repeated_action.get("args", {}).get("text")
                if not isinstance(repeated_value, str):
                    continue
                if " ".join(repeated_value.casefold().split()) != normalized_recorded_value:
                    continue
                binding_steps.append(repeated_step_index)
        for binding_step_index in binding_steps:
            target = (
                f"$.steps[{binding_step_index}].action.args.{arg_name}"
            )
            if any(
                str(binding.get("target") or "") == target
                for binding in function.get("bindings") or ()
                if isinstance(binding, dict)
            ):
                continue
            function["bindings"].append(
                {
                    "source": f"$.arguments.{name}",
                    "target": target,
                }
            )
            function["steps"][binding_step_index]["action"]["args"][arg_name] = ""
        values_by_name[name] = recorded_value


def _source_action_parameter_evidence(
    *,
    source_step: dict[str, Any],
    action: dict[str, Any],
    source_step_index: int,
    task_parameters: Any,
) -> list[dict[str, Any]]:
    raw_action = source_step.get("action")
    if not isinstance(raw_action, dict) or not isinstance(task_parameters, dict):
        return []
    if str(action.get("tool") or "") != "open_app":
        return []
    if str(raw_action.get("action_type") or "") != "open_app":
        return []
    raw_app_name = " ".join(str(raw_action.get("app_name") or "").casefold().split())
    task_app_name = " ".join(
        str(task_parameters.get("app_name") or "").casefold().split()
    )
    package_name = str(action.get("args", {}).get("package_name") or "").strip()
    if not raw_app_name or raw_app_name != task_app_name or not package_name:
        return []
    return [
        {
            "source_step_index": int(source_step_index),
            "arg_name": "package_name",
            "task_parameter_name": "app_name",
            "task_parameter_value": str(task_parameters["app_name"]),
            "recorded_value": package_name,
            "value_contract": "android_package_name",
        }
    ]


def _promote_launcher_app_entry(
    action: dict[str, Any],
    *,
    observation: dict[str, Any],
    next_observation: Any,
) -> dict[str, Any]:
    """Turn a recorded launcher app-icon click into the global open_app entry."""

    if str(action.get("tool") or "") != "click":
        return action
    before_package = _primary_observation_package(observation)
    after_package = _primary_observation_package(next_observation)
    if not before_package or not after_package:
        return action
    if "launcher" not in before_package.casefold():
        return action
    if after_package == before_package or "systemui" in after_package.casefold():
        return action
    return {
        "tool": "open_app",
        "args": {"package_name": after_package},
    }


def _canonicalize_open_app_action(
    action: dict[str, Any],
    *,
    next_observation: Any,
) -> dict[str, Any]:
    """Persist the native package observed after a recorded app launch."""

    if str(action.get("tool") or "") != "open_app":
        return action
    package_name = str(action.get("args", {}).get("package_name") or "").strip()
    if "." in package_name and " " not in package_name:
        return action
    try:
        from src.integrations.android_world.apps import resolve_androidworld_package

        resolved_package = resolve_androidworld_package(package_name)
    except Exception:
        resolved_package = ""
    if resolved_package:
        updated = json.loads(json.dumps(action, ensure_ascii=False))
        updated.setdefault("args", {})["package_name"] = resolved_package
        return updated
    after_package = _primary_observation_package(next_observation)
    if not after_package or after_package.casefold() in {
        "com.android.permissioncontroller",
        "com.google.android.permissioncontroller",
    }:
        return action
    updated = json.loads(json.dumps(action, ensure_ascii=False))
    updated.setdefault("args", {})["package_name"] = after_package
    return updated


def _primary_observation_package(observation: Any) -> str:
    if not isinstance(observation, dict):
        return ""
    auxiliaries = observation.get("auxiliaries")
    if isinstance(auxiliaries, dict):
        package_name = str(auxiliaries.get("package_name") or "").strip()
        if package_name:
            return package_name
    xml = str(observation.get("xml") or observation.get("forest") or "").strip()
    if not xml:
        return ""
    try:
        packages = [
            str(node.get("package") or "").strip()
            for node in ET.fromstring(xml).iter()
            if str(node.get("package") or "").strip()
        ]
    except ET.ParseError:
        packages = re.findall(r'\bpackage="([^"]+)"', xml)
    for package in packages:
        if "systemui" not in package.casefold():
            return package
    return ""


def _write_authoring_failure(
    root: Path,
    *,
    error: Exception,
    model: str,
    prompt: str,
    response: str,
    usage: dict[str, int],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "omniflow.function-authoring-failure.v1",
        "success": False,
        "classification": "authoring_rejected",
        "error": str(error) or type(error).__name__,
        "model": model,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "raw_response": response,
        **usage,
    }
    (root / "authoring_failure.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _referenced_source_state_ids(functions: list[Any]) -> list[str]:
    state_ids: list[str] = []
    for function in functions:
        for item in function.steps:
            if isinstance(item, dict):
                state_id = str(item.get("source_state_id") or "").strip()
            else:
                state_id = str(getattr(item, "source_state_id", "") or "").strip()
            if state_id and state_id not in state_ids:
                state_ids.append(state_id)
    if not state_ids:
        raise ValueError("function_source_state_references_required")
    return state_ids


def _normalize_source_state(value: Any, expected_state_id: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"function_source_state_invalid:{expected_state_id}")
    extra = value.get("extra") if isinstance(value.get("extra"), dict) else {}
    state_id = str(
        value.get("state_id") or extra.get("state_id") or expected_state_id
    ).strip()
    if state_id != expected_state_id:
        raise ValueError(f"function_source_state_id_mismatch:{expected_state_id}")
    state: dict[str, Any] = {"state_id": state_id}
    aliases = {
        "xml": ("xml", "page", "observation_xml"),
        "package_name": ("package_name", "packageName"),
        "activity_name": ("activity_name", "activityName"),
        "screenshot_path": ("screenshot_path",),
    }
    for output, names in aliases.items():
        item = next(
            (
                source[name]
                for source in (value, extra)
                for name in names
                if source.get(name) is not None
            ),
            None,
        )
        if item is not None:
            if not isinstance(item, str):
                raise ValueError(
                    f"function_source_state_{output}_invalid:{state_id}"
                )
            state[output] = item
    display = value.get("display") or extra.get("display")
    if not isinstance(display, dict):
        width = value.get("width") or value.get("display_width")
        height = value.get("height") or value.get("display_height")
        if width is None:
            width = extra.get("width") or extra.get("display_width")
        if height is None:
            height = extra.get("height") or extra.get("display_height")
        if width is not None or height is not None:
            display = {"width": width, "height": height}
    if display is not None:
        if not isinstance(display, dict) or set(display) != {"width", "height"}:
            raise ValueError(f"function_source_state_display_invalid:{state_id}")
        try:
            width = int(display["width"])
            height = int(display["height"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"function_source_state_display_invalid:{state_id}"
            ) from error
        if width <= 0 or height <= 0:
            raise ValueError(f"function_source_state_display_invalid:{state_id}")
        state["display"] = {"width": width, "height": height}
    return state


def _compile_runlog_to_store_mechanical(
    run_log: str | Path | dict[str, Any],
    output_root: str | Path,
    *,
    function_bundle: dict[str, Any] | None = None,
    model: str | None = None,
    client: Any | None = None,
    prompt: str | None = None,
    timeout: float = 120.0,
    source_states: str | Path | dict[str, Any] | None = None,
    state_loader: Any | None = None,
) -> dict[str, Any]:
    """Convert a RunLog into one ordered Function Schema.

    Semantic extraction belongs to the offline Agent.  This owner only copies
    successful physical actions, their source state references, and the exact
    action arguments into the canonical Function/Store format.
    """
    del model, client, prompt, timeout, state_loader
    from omniflow.functions.artifact import bind_function, parse_function_artifact
    from omniflow.functions.store import FunctionStore
    from omniflow.transfer.runtime import (
        TRANSFER_STATE_CATALOG_FILENAME,
        TRANSFER_STATE_CATALOG_VERSION,
        load_transfer_state_catalog,
    )

    root = Path(output_root).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"immutable output version already exists: {root}")
    if isinstance(run_log, dict):
        payload = json.loads(json.dumps(run_log, ensure_ascii=False))
    else:
        payload = json.loads(
            Path(run_log).expanduser().resolve().read_text(encoding="utf-8")
        )
    if not isinstance(payload, dict):
        raise ValueError("source_runlog_must_be_object")
    goal = str(payload.get("goal") or "").strip()
    source_steps = payload.get("steps")
    if not goal or not isinstance(source_steps, list):
        raise ValueError("source_runlog_goal_and_steps_required")

    converted_steps: list[dict[str, Any]] = []
    terminal_count = 0
    optional_checker_actions: list[dict[str, Any]] = []
    previous_step: dict[str, Any] | None = None
    for source_index, source_step in enumerate(source_steps):
        if not isinstance(source_step, dict):
            continue
        result = source_step.get("result")
        if isinstance(result, dict) and result.get("success") is not True:
            continue
        action = source_step.get("action")
        action_type = str(action.get("action_type") or "") if isinstance(action, dict) else ""
        if action_type in {"answer", "status", "unknown"}:
            terminal_count += 1
            continue
        if _is_transient_system_action(source_step):
            optional_checker_actions.append(
                {
                    "source_step_index": source_index,
                    "checker_id": "dismiss_permission_dialog",
                    "reason": "permission_dialog_setup_is_optional",
                }
            )
            continue
        observation = source_step.get("observation")
        if not isinstance(observation, dict):
            raise ValueError(f"source_observation_required:{source_index}")
        next_observation = source_step.get("next_observation")
        projected = project_run_log_step_actions(
            payload,
            source_index,
            previous_step=previous_step,
            next_observation=next_observation
            if isinstance(next_observation, dict)
            else None,
        )
        before_state_id = state_id(observation)
        after_state = next_observation if isinstance(next_observation, dict) else observation
        after_state_id = state_id(after_state)
        for projected_action in projected:
            converted_steps.append(
                {
                    "source_step_index": source_index,
                    "before_state_id": before_state_id,
                    "after_state_id": after_state_id,
                    "action": projected_action,
                }
            )
        previous_step = source_step
    if not converted_steps:
        raise ValueError("successful_source_actions_required")

    function_id = "complete_source_workflow"
    function = {
        "schema_version": "omniflow.function.v2",
        "function_id": function_id,
        "name": f"Complete task: {goal[:240]}",
        "description": f"Execute the converted source workflow for: {goal[:240]}",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "bindings": [],
        "steps": [
            {
                "step_index": index,
                "source_state_id": step["before_state_id"],
                "action": step["action"],
            }
            for index, step in enumerate(converted_steps)
        ],
        "agent_visible": True,
    }
    if function_bundle is not None:
        bundle = json.loads(json.dumps(function_bundle, ensure_ascii=False))
    else:
        bundle = {
            "schema_version": "omniflow.function-bundle.v2",
            "run_id": str(payload.get("run_id") or "successful-source"),
            "arguments": {function_id: {}},
            "checker_rules": [],
            "functions": [function],
        }
    if not isinstance(bundle, dict):
        raise ValueError("function_bundle_must_be_object")
    raw_functions = bundle.get("functions")
    if not isinstance(raw_functions, list) or not raw_functions:
        raise ValueError("function_bundle_functions_required")
    functions = [parse_function_artifact(value) for value in raw_functions]
    arguments_by_function = bundle.get("arguments")
    if not isinstance(arguments_by_function, dict):
        raise ValueError("function_bundle_arguments_required")
    normalized_arguments: dict[str, dict[str, Any]] = {}
    for item in functions:
        arguments = arguments_by_function.get(item.id, {})
        if not isinstance(arguments, dict):
            raise ValueError("function_bundle_arguments_invalid")
        bind_function(item, arguments)
        normalized_arguments[item.id] = dict(arguments)

    referenced_state_ids = _referenced_source_state_ids(functions)
    if source_states is None:
        raise ValueError("function_source_states_required")
    if isinstance(source_states, (str, Path)):
        catalog = json.loads(Path(source_states).expanduser().resolve().read_text(encoding="utf-8"))
        states = load_transfer_state_catalog(Path(source_states).expanduser().resolve())
        catalog_run_id = str(catalog.get("run_id") or "") if isinstance(catalog, dict) else ""
    elif isinstance(source_states, dict):
        raw_states = source_states.get("states", source_states)
        states = {
            str(key): _normalize_source_state(value, str(key))
            for key, value in raw_states.items()
        }
        catalog_run_id = str(source_states.get("run_id") or payload.get("run_id") or "")
    else:
        raise ValueError("function_source_states_invalid")
    run_id = str(payload.get("run_id") or "successful-source")
    if catalog_run_id and catalog_run_id != run_id:
        raise ValueError("function_source_state_run_id_mismatch")
    missing = [item for item in referenced_state_ids if item not in states]
    if missing:
        raise ValueError("function_source_states_missing:" + ",".join(missing))

    root.mkdir(parents=True, exist_ok=True)
    store_path = root / "store.json"
    store = FunctionStore(store_path)
    for item in functions:
        store.put_function(item)
    transfer_state_path = root / TRANSFER_STATE_CATALOG_FILENAME
    transfer_state_path.write_text(
        json.dumps(
            {
                "schema_version": TRANSFER_STATE_CATALOG_VERSION,
                "run_id": run_id,
                "states": {key: states[key] for key in referenced_state_ids},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": "omniflow.androidworld.function-gate.v2",
        "success": True,
        "live_probe_allowed": True,
        "classification": "ready_for_live_probe",
        "reason": "Mechanical RunLog to Function Schema conversion.",
        "model": None,
        "prompt_sha256": None,
        "store_path": str(store_path),
        "checker_store_path": str(DEFAULT_CHECKER_LIBRARY_PATH),
        "checker_count": len(CheckerLibrary.load().rules),
        "transfer_state_catalog": str(transfer_state_path),
        "transfer_state_count": len(referenced_state_ids),
        "function_ids": [item.id for item in functions],
        "function_count": len(functions),
        "function_step_count": len(converted_steps),
        "source_successful_action_count": len(converted_steps),
        "source_terminal_output_count": terminal_count,
        "optional_checker_actions": optional_checker_actions,
        "source_calls": [
            {"function_id": item.id, "arguments": normalized_arguments[item.id]}
            for item in functions
        ],
        "source_arguments": normalized_arguments,
        "model_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    (root / "compile_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
