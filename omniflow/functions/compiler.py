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
    iter_task_parameter_values,
    parameter_candidates,
    semantic_parameter_evidence,
)
from omniflow.runlog import project_run_log_step_actions
from omniflow.runtime.checker import (
    DEFAULT_CHECKER_LIBRARY_PATH,
    CheckerLibrary,
    validate_checker_rule,
)


def _default_authoring_workflow_prompt() -> str:
    """Return the A-side authoring contract consumed by the converter."""

    return """Turn one successful GUI RunLog into reusable Function Memory.
Return exactly one JSON object in this shape; return no prose:
{
  "binding_owner": "agent",
  "reason": "short rationale",
  "functions": [{
    "function_id": "delete_recipe",
    "name": "Delete requested recipe",
    "description": "Delete one requested recipe from the visible collection.",
    "occurrences": [
      {"source_step_indices": [2, 3]},
      {"source_step_indices": [6, 7]}
    ],
    "parameters": [{
      "name": "recipe_title",
      "description": "Value requested by the goal",
      "bindings": [
        {
          "occurrence_index": 0,
          "source_step_index": 2,
          "binding_kind": "render_node",
          "node_id": "0:24",
          "attribute": "text",
          "recorded_value": "First Recipe"
        },
        {
          "occurrence_index": 1,
          "source_step_index": 6,
          "binding_kind": "render_node",
          "node_id": "0:19",
          "attribute": "content-desc",
          "recorded_value": "Second Recipe"
        }
      ]
    }]
  }],
  "complete_function": {
    "function_id": "complete_task",
    "name": "Complete requested task",
    "description": "Complete the requested task using the successful workflow.",
    "source_step_indices": [0, 1, 2, 3, 4, 5, 6, 7],
    "parameters": []
  }
}

Stage 1 — discover Functions. Find zero or more semantically stable, contiguous
local operations in the successful source steps. If the same operation repeats,
represent it as one Function with multiple occurrences. Also identify one
complete_function spanning the successful source sequence. Returning no local
Function is valid when raw replay is the safest reusable capability.

Stage 2 — author every binding on the A side. The Compiler will not infer,
recommend, repair, or select bindings. Read task_parameters, each source action,
and source_ui.nodes. Decide the semantic parameter name and emit every binding
yourself:
- action_arg has exactly occurrence_index, source_step_index, binding_kind, and
  arg_name. Bind only an argument that the source action actually consumes.
- render_node has exactly occurrence_index, source_step_index, binding_kind,
  node_id, attribute, and recorded_value. Use a node_id and text/content-desc
  substring from that step's source_ui; render_node is for click/long_press only.
  The node text must itself represent this parameter value. Never bind a Save,
  Delete, OK, menu, tab, or other fixed control merely because it is clicked after
  an input action. Do not add render_node unless changing the parameter requires
  changing the semantic identity of the clicked/long-pressed source node.
The source default for action_arg is copied verbatim from that action argument;
the source default for render_node is its recorded_value. All undeclared action
arguments and UI values remain fixed at their recorded defaults. If a safe binding
cannot be determined, simply omit it. Never bind coordinates or repetition counts.

The complete_function must contain exactly function_id, name, description,
source_step_indices, and parameters. Its parameters may be empty: bindings declared
on local Functions do not need to be repeated. Any omitted value keeps the recorded
RunLog default.

Stage 3 — convert and register. The Compiler only maps your source_step_index values
to immutable RunLog actions/source states and serializes your declared parameters,
bindings, default values, calls, and Function Store. It performs schema and index
checks required to make valid JSON artifacts, but makes no semantic binding decision.
Do not emit actions, coordinates, source_state_id, Store JSON, JSONPath expressions,
checker rules, candidate ids, validation, or registration sections. Treat
optional_checker_actions as removed setup and do not reconstruct them.
"""


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
    omitted_action_types: set[str] = set()
    optional_checker_actions: list[dict[str, Any]] = []
    onboarding_active = False
    entry_recovery_active = True
    previous_successful_step: dict[str, Any] | None = None
    source_steps = payload["steps"]
    # Only the first successful open_app is environment recovery. An open_app
    # later in a workflow can be real task progress (for example, enabling
    # Wi-Fi and then opening a requested app) and must remain in the Function.
    source_has_main_action = any(
        isinstance(source_step, dict)
        and str(
            (source_step.get("action") or {}).get("action_type") or ""
        ).strip()
        not in {"answer", "status", "unknown", "open_app"}
        and not _is_transient_system_action(source_step)
        for source_step in source_steps
    )
    first_effective_step = next(
        (
            (source_step_index, source_step)
            for source_step_index, source_step in enumerate(source_steps)
            if isinstance(source_step, dict)
            and isinstance(source_step.get("action"), dict)
            and str(source_step["action"].get("action_type") or "").strip()
            not in {"answer", "status", "unknown"}
            and not _is_transient_system_action(source_step)
        ),
        None,
    )
    entry_open_app_index = (
        first_effective_step[0]
        if source_has_main_action
        and first_effective_step is not None
        and str(first_effective_step[1]["action"].get("action_type") or "").strip()
        == "open_app"
        else None
    )
    launcher_entry_indices = _recorded_launcher_entry_indices(source_steps)
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
        if source_step_index in launcher_entry_indices:
            omitted_action_types.add("open_app")
            optional_checker_actions.append(
                {
                    "source_step_index": source_step_index,
                    "checker_id": "restore_target_app",
                    "reason": "recorded_launcher_app_entry_is_environment_recovery",
                }
            )
            continue
        if action_type == "open_app" and source_has_main_action:
            omitted_action_types.add("open_app")
            optional_checker_actions.append(
                {
                    "source_step_index": source_step_index,
                    "checker_id": "restore_target_app",
                    "reason": "recorded_app_entry_is_environment_recovery",
                }
            )
            continue
        if action_type in {"navigate_back", "navigate_home"} and entry_recovery_active:
            omitted_action_types.add(action_type)
            continue
        entry_recovery_active = False
        onboarding_checker_id = _optional_onboarding_checker_id(
            step, onboarding_active=onboarding_active
        )
        if onboarding_checker_id is not None:
            onboarding_active = True
            omitted_action_types.add("checker")
            optional_checker_actions.append(
                {
                    "source_step_index": source_step_index,
                    "checker_id": onboarding_checker_id,
                    "reason": "first_run_onboarding_setup_is_optional",
                }
            )
            continue
        if onboarding_active:
            onboarding_active = False
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
        for action in projected_actions:
            action = _canonicalize_open_app_action(
                action,
                next_observation=next_observation,
            )
            steps.append(
                {
                    "source_step_index": source_step_index,
                    "before_state_id": before_state_id,
                    "action": action,
                    "source_ui": _source_ui_projection(observation),
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
        "omitted_action_types": sorted(omitted_action_types),
        "optional_checker_actions": optional_checker_actions,
    }
    authoring_prompt = prompt or _default_authoring_workflow_prompt()
    selected_model = str(model or "").strip() or None
    usage = {
        "model_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    authoring_attempt_trace: list[dict[str, Any]] = []
    if function_bundle is not None:
        if client is not None or prompt is not None:
            raise ValueError("function_bundle_cannot_use_author_model_options")
        authored = {
            "reason": "Registered the supplied Function Schema.",
            "bundle": json.loads(json.dumps(function_bundle, ensure_ascii=False)),
        }
    elif selected_model is None:
        if client is not None or prompt is not None:
            raise ValueError("author_model_required_for_author_options")
        # Device RunLog registration must not require a second Kotlin
        # converter or a model round-trip.  The canonical facts above already
        # contain the official projected actions and immutable source-state
        # ids, so use the same Python authoring/materialization path with a
        # deterministic complete Function.
        authored = _raw_source_replay_authoring(facts)
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
        raw_author_response = ""
        authoring_error: Exception | None = None
        for attempt in range(1, 4):
            request_payload: dict[str, Any] = {
                "output_format": "json",
                "source_run": facts,
            }
            if authoring_attempt_trace:
                request_payload["harness_feedback"] = {
                    "previous_attempt": attempt - 1,
                    "error": authoring_attempt_trace[-1]["error"],
                    "instruction": (
                        "Revise the Function proposal JSON and return a complete "
                        "replacement object. Do not patch the prior response."
                    ),
                }
            response = client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": authoring_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(request_payload, ensure_ascii=False),
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
            prompt_tokens = int(
                getattr(response_usage, "prompt_tokens", 0) or 0
            )
            completion_tokens = int(
                getattr(response_usage, "completion_tokens", 0) or 0
            )
            total_tokens = int(getattr(response_usage, "total_tokens", 0) or 0)
            usage["model_calls"] += 1
            usage["prompt_tokens"] += prompt_tokens
            usage["completion_tokens"] += completion_tokens
            usage["total_tokens"] += (
                total_tokens or prompt_tokens + completion_tokens
            )
            raw_author_response = str(response.choices[0].message.content or "")
            try:
                proposal = json.loads(raw_author_response)
                authored = _materialize_authoring_response(
                    proposal,
                    facts,
                    candidate_map={},
                )
                _validate_materialized_function_artifacts(
                    authored,
                    raw_payload=raw,
                )
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                authoring_error = error
                authoring_attempt_trace.append(
                    {
                        "attempt": attempt,
                        "accepted": False,
                        "error": str(error) or type(error).__name__,
                    }
                )
                continue
            authoring_attempt_trace.append(
                {"attempt": attempt, "accepted": True, "error": None}
            )
            authoring_error = None
            break
        if authoring_error is not None:
            _write_authoring_failure(
                root,
                error=authoring_error,
                model=selected_model,
                prompt=authoring_prompt,
                response=raw_author_response,
                usage=usage,
            )
            authored = _raw_source_replay_authoring(
                facts,
                rejected_error=str(authoring_error) or type(authoring_error).__name__,
            )
    if not isinstance(authored, dict) or not {"reason", "bundle"}.issubset(authored):
        raise ValueError("function_author_response_contract_invalid")
    if set(authored) - {
        "reason",
        "bundle",
        "source_calls",
        "authoring_workflow",
    }:
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
    raw_functions = _remove_compiler_entry_actions(
        raw_functions,
        raw_payload=raw,
    )
    raw_functions = _remove_compiler_environment_actions(
        raw_functions,
        raw_payload=raw,
    )
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
    raw_source_calls = authored.get("source_calls")
    if raw_source_calls is None:
        source_calls = [
            {
                "function_id": function_id,
                "arguments": json.loads(
                    json.dumps(arguments_by_function[function_id], ensure_ascii=False)
                ),
            }
            for function_id in function_ids
        ]
    else:
        if not isinstance(raw_source_calls, list) or not raw_source_calls:
            raise ValueError("function_author_source_calls_invalid")
        functions_by_id = {function.id: function for function in functions}
        source_calls = []
        for raw_call in raw_source_calls:
            if not isinstance(raw_call, dict) or set(raw_call) != {
                "function_id",
                "arguments",
            }:
                raise ValueError("function_author_source_calls_invalid")
            function_id = str(raw_call.get("function_id") or "").strip()
            arguments = raw_call.get("arguments")
            if function_id not in functions_by_id or not isinstance(arguments, dict):
                raise ValueError("function_author_source_calls_invalid")
            bind_function(functions_by_id[function_id], arguments)
            source_calls.append(
                {
                    "function_id": function_id,
                    "arguments": json.loads(
                        json.dumps(arguments, ensure_ascii=False)
                    ),
                }
            )

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
        "source_calls": source_calls,
        "source_arguments": json.loads(
            json.dumps(arguments_by_function, ensure_ascii=False)
        ),
        "authoring_attempts": authoring_attempt_trace,
        **(
            {"authoring_workflow": authored["authoring_workflow"]}
            if "authoring_workflow" in authored
            else {}
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
        "render_bindings": [],
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


def _authoring_candidate_catalog(
    facts: dict[str, Any],
    *,
    action_candidates: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Expose binding evidence while retaining private materialization ids."""

    public: list[dict[str, Any]] = []
    catalog: dict[str, dict[str, Any]] = {}
    actions = (
        list(action_candidates)
        if action_candidates is not None
        else _source_parameter_candidates(facts)
    )
    for index, raw in enumerate(actions):
        candidate_id = f"action_parameter_{index:03d}"
        candidate = {"kind": "action_arg", **dict(raw)}
        catalog[candidate_id] = candidate
        public.append(
            {
                key: candidate[key]
                for key in (
                    "source_step_index",
                    "tool",
                    "arg_name",
                    "recorded_value",
                    "suggested_name",
                    "task_parameter_value",
                    "fixed_suffix",
                )
                if key in candidate
            }
            | {"binding_kind": candidate["kind"]}
        )
    for index, raw in enumerate(facts.get("node_parameter_evidence") or ()):
        if not isinstance(raw, dict):
            continue
        candidate_id = f"render_parameter_{index:03d}"
        candidate = {"kind": "render_node", **dict(raw)}
        catalog[candidate_id] = candidate
        public.append(
            {
                key: candidate[key]
                for key in (
                    "source_step_index",
                    "tool",
                    "attribute",
                    "node_label",
                    "recorded_value",
                    "suggested_name",
                    "task_parameter_value",
                )
                if key in candidate
            }
            | {"binding_kind": candidate["kind"]}
        )
    return public, catalog


def _materialize_authoring_response(
    value: Any,
    facts: dict[str, Any],
    *,
    candidate_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Accept the compact Agent proposal and retain legacy plans for callers."""

    if isinstance(value, dict) and value.get("binding_owner") == "agent":
        return _materialize_agent_owned_workflow(value, facts)
    if isinstance(value, dict) and (
        "functions" in value or "complete_function" in value
    ):
        return _materialize_authoring_workflow(
            value,
            facts,
            candidate_map=candidate_map,
        )
    return _materialize_authoring_plan(value, facts)


def _source_ui_projection(observation: Any) -> dict[str, Any]:
    """Expose source UI facts without choosing any parameter or target node."""

    if not isinstance(observation, dict):
        return {"nodes": []}
    xml = observation.get("xml") or observation.get("forest")
    if not isinstance(xml, str) or not xml.strip():
        return {"nodes": []}
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return {"nodes": []}
    nodes: list[dict[str, Any]] = []
    exposed_attributes = (
        "id",
        "text",
        "content-desc",
        "resource-id",
        "class",
        "bounds",
        "clickable",
        "long-clickable",
        "enabled",
    )
    for node in root.iter("node"):
        projected = {
            attribute: str(node.attrib.get(attribute) or "")
            for attribute in exposed_attributes
            if str(node.attrib.get(attribute) or "").strip()
        }
        if projected:
            nodes.append(projected)
    return {"nodes": nodes}


def _raw_source_replay_authoring(
    facts: dict[str, Any],
    *,
    rejected_error: str | None = None,
) -> dict[str, Any]:
    """Convert the successful source sequence verbatim, with no inferred binding."""

    source_steps = [
        step for step in facts.get("steps") or () if isinstance(step, dict)
    ]
    if not source_steps:
        raise ValueError("successful_source_actions_required")
    function_id = "complete_source_workflow"
    goal = " ".join(str(facts.get("goal") or "").split())
    function = _agent_owned_function_artifact(
        function_id=function_id,
        name=f"Complete task: {goal}" if goal else "Complete source replay",
        description=(
            f"Complete the requested task: {goal}. Replay the converted source "
            "actions in order."
            if goal
            else "Replay the converted source actions in order."
        ),
        source_indices=[int(step["source_step_index"]) for step in source_steps],
        facts=facts,
        parameters=[],
        occurrence_index=0,
    )
    result = {
        "reason": "Registered the unparameterized successful source replay.",
        "bundle": {
            "schema_version": "omniflow.function-bundle.v2",
            "run_id": str(facts["run_id"]),
            "arguments": {function_id: {}},
            "checker_rules": [],
            "functions": [function],
        },
        "source_calls": [{"function_id": function_id, "arguments": {}}],
        "authoring_workflow": {
            "schema_version": "omniflow.function-authoring-workflow.v3",
            "binding_owner": "agent",
            "agent_proposal_accepted": rejected_error is None,
            "fallback_mode": "raw_source_replay",
            "complete_function_id": function_id,
        },
    }
    if rejected_error is not None:
        result["authoring_workflow"]["rejected_error"] = rejected_error
    return result


def _materialize_agent_owned_workflow(
    value: Any,
    facts: dict[str, Any],
) -> dict[str, Any]:
    """Map the A-side Agent's complete binding declaration into Store artifacts."""

    if not isinstance(value, dict) or set(value) != {
        "binding_owner",
        "reason",
        "functions",
        "complete_function",
    }:
        raise ValueError("function_author_workflow_response_contract_invalid")
    reason = str(value.get("reason") or "").strip()
    if not reason:
        raise ValueError("function_author_reason_must_be_string")
    raw_functions = value.get("functions")
    raw_complete = value.get("complete_function")
    if not isinstance(raw_functions, list) or not isinstance(raw_complete, dict):
        raise ValueError("function_author_inventory_contract_invalid")

    source_indices = [
        int(step["source_step_index"])
        for step in facts.get("steps") or ()
        if isinstance(step, dict) and isinstance(step.get("source_step_index"), int)
    ]
    complete_fields = {
        "function_id",
        "name",
        "description",
        "source_step_indices",
        "parameters",
    }
    if set(raw_complete) != complete_fields:
        raise ValueError("function_author_inventory_complete_invalid")
    if raw_complete.get("source_step_indices") != source_indices:
        raise ValueError("function_author_plan_complete_sequence_required")

    function_fields = {
        "function_id",
        "name",
        "description",
        "occurrences",
        "parameters",
    }
    materialized_functions: list[dict[str, Any]] = []
    arguments: dict[str, dict[str, Any]] = {}
    source_calls: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for raw_function in raw_functions:
        if not isinstance(raw_function, dict) or set(raw_function) != function_fields:
            raise ValueError("function_author_inventory_definition_invalid")
        occurrences = raw_function.get("occurrences")
        if not isinstance(occurrences, list) or not occurrences:
            raise ValueError("function_author_inventory_occurrence_invalid")
        spans = [
            _agent_owned_occurrence_span(occurrence, source_indices)
            for occurrence in occurrences
        ]
        function_id = _agent_owned_function_id(raw_function, seen_ids)
        parameters = _agent_owned_parameters(
            raw_function.get("parameters"),
            occurrence_count=len(spans),
            facts=facts,
            spans=spans,
        )
        _validate_agent_owned_binding_spans(parameters, spans)
        function = _agent_owned_function_artifact(
            function_id=function_id,
            name=str(raw_function.get("name") or "").strip(),
            description=str(raw_function.get("description") or "").strip(),
            source_indices=spans[0],
            facts=facts,
            parameters=parameters,
            occurrence_index=0,
        )
        materialized_functions.append(function)
        arguments[function_id] = _agent_owned_occurrence_arguments(parameters, 0)
        for occurrence_index in range(len(spans)):
            source_calls.append(
                {
                    "function_id": function_id,
                    "arguments": _agent_owned_occurrence_arguments(
                        parameters, occurrence_index
                    ),
                }
            )

    complete_id = _agent_owned_function_id(raw_complete, seen_ids)
    complete_parameters = _agent_owned_parameters(
        raw_complete.get("parameters"),
        occurrence_count=1,
        facts=facts,
        spans=[source_indices],
    )
    _validate_agent_owned_binding_spans(complete_parameters, [source_indices])
    complete_function = _agent_owned_function_artifact(
        function_id=complete_id,
        name=str(raw_complete.get("name") or "").strip(),
        description=str(raw_complete.get("description") or "").strip(),
        source_indices=source_indices,
        facts=facts,
        parameters=complete_parameters,
        occurrence_index=0,
    )
    materialized_functions.append(complete_function)
    arguments[complete_id] = _agent_owned_occurrence_arguments(
        complete_parameters, 0
    )
    if not source_calls:
        source_calls.append(
            {"function_id": complete_id, "arguments": arguments[complete_id]}
        )

    from omniflow.functions.artifact import bind_function, parse_function_artifact

    for function in materialized_functions:
        parsed = parse_function_artifact(function)
        bind_function(parsed, arguments[function["function_id"]])
    return {
        "reason": reason,
        "bundle": {
            "schema_version": "omniflow.function-bundle.v2",
            "run_id": str(facts["run_id"]),
            "arguments": arguments,
            "checker_rules": [],
            "functions": materialized_functions,
        },
        "source_calls": source_calls,
        "authoring_workflow": {
            "schema_version": "omniflow.function-authoring-workflow.v3",
            "binding_owner": "agent",
            "agent_proposal_accepted": True,
            "fallback_mode": None,
            "definition_count": len(raw_functions),
            "invocation_count": len(source_calls),
            "complete_function_id": complete_id,
        },
    }


def _agent_owned_occurrence_span(value: Any, source_indices: list[int]) -> list[int]:
    if not isinstance(value, dict) or set(value) != {"source_step_indices"}:
        raise ValueError("function_author_inventory_occurrence_invalid")
    span = value.get("source_step_indices")
    if (
        not isinstance(span, list)
        or not span
        or any(isinstance(index, bool) or not isinstance(index, int) for index in span)
        or span != sorted(set(span))
        or any(index not in source_indices for index in span)
    ):
        raise ValueError("function_author_inventory_occurrence_invalid")
    positions = [source_indices.index(index) for index in span]
    if positions != list(range(positions[0], positions[-1] + 1)):
        raise ValueError("function_author_inventory_occurrence_not_contiguous")
    return list(span)


def _agent_owned_function_id(value: dict[str, Any], seen_ids: set[str]) -> str:
    function_id = str(value.get("function_id") or "").strip()
    if (
        re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", function_id) is None
        or function_id in seen_ids
    ):
        raise ValueError("function_author_inventory_function_id_invalid")
    seen_ids.add(function_id)
    return function_id


def _agent_owned_parameters(
    value: Any,
    *,
    occurrence_count: int,
    facts: dict[str, Any],
    spans: list[list[int]],
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("function_author_parameterization_invalid")
    parameter_fields = {
        "name",
        "description",
        "bindings",
    }
    parsed: list[dict[str, Any]] = []
    names: set[str] = set()
    for parameter in value:
        if not isinstance(parameter, dict) or set(parameter) != parameter_fields:
            raise ValueError("function_author_parameter_invalid")
        name = str(parameter.get("name") or "").strip()
        description = str(parameter.get("description") or "").strip()
        if (
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name) is None
            or name in names
            or not description
        ):
            raise ValueError("function_author_parameter_invalid")
        raw_bindings = parameter.get("bindings")
        if not isinstance(raw_bindings, list) or not raw_bindings:
            raise ValueError("function_author_parameter_binding_invalid")
        bindings: list[dict[str, Any]] = []
        covered_occurrences: set[int] = set()
        for binding in raw_bindings:
            parsed_binding = _agent_owned_binding(binding, occurrence_count)
            bindings.append(parsed_binding)
            covered_occurrences.add(parsed_binding["occurrence_index"])
        if covered_occurrences != set(range(occurrence_count)):
            raise ValueError("function_author_parameter_occurrence_incomplete")
        values = _agent_owned_default_values(
            bindings,
            occurrence_count=occurrence_count,
            facts=facts,
            spans=spans,
        )
        names.add(name)
        parsed.append(
            {
                "name": name,
                "description": description,
                "values": values,
                "bindings": bindings,
            }
        )
    return parsed


def _agent_owned_default_values(
    bindings: list[dict[str, Any]],
    *,
    occurrence_count: int,
    facts: dict[str, Any],
    spans: list[list[int]],
) -> dict[int, Any]:
    fact_steps = {
        int(step["source_step_index"]): step
        for step in facts.get("steps") or ()
        if isinstance(step, dict) and isinstance(step.get("source_step_index"), int)
    }
    values: dict[int, Any] = {}
    for occurrence_index in range(occurrence_count):
        occurrence_values: list[Any] = []
        for binding in bindings:
            if binding["occurrence_index"] != occurrence_index:
                continue
            source_index = int(binding["source_step_index"])
            if source_index not in spans[occurrence_index]:
                raise ValueError("function_author_parameter_binding_outside_occurrence")
            if binding["binding_kind"] == "action_arg":
                step = fact_steps.get(source_index)
                action = step.get("action") if isinstance(step, dict) else None
                args = action.get("args") if isinstance(action, dict) else None
                arg_name = str(binding.get("arg_name") or "")
                if not isinstance(args, dict) or arg_name not in args:
                    raise ValueError("function_author_plan_parameter_target_invalid")
                occurrence_values.append(args[arg_name])
            else:
                occurrence_values.append(binding.get("recorded_value"))
        if not occurrence_values:
            raise ValueError("function_author_parameter_occurrence_incomplete")
        default_value = occurrence_values[0]
        if any(item != default_value for item in occurrence_values[1:]):
            raise ValueError("function_author_parameter_default_conflict")
        values[occurrence_index] = json.loads(
            json.dumps(default_value, ensure_ascii=False)
        )
    return values


def _agent_owned_binding(value: Any, occurrence_count: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("function_author_parameter_binding_invalid")
    binding_kind = str(value.get("binding_kind") or "").strip()
    common = {"occurrence_index", "source_step_index", "binding_kind"}
    if binding_kind == "action_arg":
        expected = common | {"arg_name"}
        allowed_shapes = {frozenset(expected), frozenset(expected | {"recorded_value"})}
    elif binding_kind == "render_node":
        expected = common | {"node_id", "attribute", "recorded_value"}
        allowed_shapes = {frozenset(expected)}
    else:
        raise ValueError(
            "function_author_parameter_binding_kind_invalid:"
            f"{binding_kind or '<missing>'}"
        )
    actual = set(value)
    if frozenset(actual) not in allowed_shapes:
        raise ValueError(
            f"function_author_{binding_kind}_fields_invalid:"
            f"expected={','.join(sorted(expected))}:"
            f"got={','.join(sorted(actual))}"
        )
    occurrence_index = value.get("occurrence_index")
    source_step_index = value.get("source_step_index")
    if (
        isinstance(occurrence_index, bool)
        or not isinstance(occurrence_index, int)
        or occurrence_index not in range(occurrence_count)
        or isinstance(source_step_index, bool)
        or not isinstance(source_step_index, int)
    ):
        raise ValueError("function_author_parameter_binding_invalid")
    return json.loads(json.dumps(value, ensure_ascii=False))


def _agent_owned_occurrence_arguments(
    parameters: list[dict[str, Any]], occurrence_index: int
) -> dict[str, Any]:
    return {
        str(parameter["name"]): json.loads(
            json.dumps(parameter["values"][occurrence_index], ensure_ascii=False)
        )
        for parameter in parameters
    }


def _validate_agent_owned_binding_spans(
    parameters: list[dict[str, Any]], spans: list[list[int]]
) -> None:
    for parameter in parameters:
        for binding in parameter["bindings"]:
            occurrence_index = int(binding["occurrence_index"])
            if int(binding["source_step_index"]) not in spans[occurrence_index]:
                raise ValueError("function_author_parameter_binding_outside_occurrence")


def _agent_owned_function_artifact(
    *,
    function_id: str,
    name: str,
    description: str,
    source_indices: list[int],
    facts: dict[str, Any],
    parameters: list[dict[str, Any]],
    occurrence_index: int,
) -> dict[str, Any]:
    if not name or not description:
        raise ValueError("function_author_inventory_definition_invalid")
    fact_steps = {
        int(step["source_step_index"]): step
        for step in facts.get("steps") or ()
        if isinstance(step, dict) and isinstance(step.get("source_step_index"), int)
    }
    if any(index not in fact_steps for index in source_indices):
        raise ValueError("function_author_plan_source_steps_invalid")
    local_by_source = {
        source_index: local_index
        for local_index, source_index in enumerate(source_indices)
    }
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    bindings: list[dict[str, Any]] = []
    render_bindings: list[dict[str, Any]] = []
    for parameter in parameters:
        parameter_name = str(parameter["name"])
        value = parameter["values"][occurrence_index]
        properties[parameter_name] = {
            "type": _json_schema_type(value),
            "description": str(parameter["description"])[:240],
        }
        required.append(parameter_name)
        for binding in parameter["bindings"]:
            if binding["occurrence_index"] != occurrence_index:
                continue
            source_index = int(binding["source_step_index"])
            if source_index not in local_by_source:
                raise ValueError("function_author_parameter_binding_outside_occurrence")
            local_index = local_by_source[source_index]
            if binding["binding_kind"] == "action_arg":
                arg_name = str(binding.get("arg_name") or "").strip()
                action = fact_steps[source_index].get("action")
                args = action.get("args") if isinstance(action, dict) else None
                if not arg_name or not isinstance(args, dict) or arg_name not in args:
                    raise ValueError("function_author_plan_parameter_target_invalid")
                bindings.append(
                    {
                        "source": f"$.arguments.{parameter_name}",
                        "target": f"$.steps[{local_index}].action.args.{arg_name}",
                    }
                )
            else:
                render_bindings.append(
                    {
                        "source": f"$.arguments.{parameter_name}",
                        "step_index": local_index,
                        "node_id": str(binding.get("node_id") or "").strip(),
                        "attribute": str(binding.get("attribute") or "").strip(),
                        "recorded_value": str(binding.get("recorded_value") or ""),
                    }
                )
    return {
        "schema_version": "omniflow.function.v2",
        "function_id": function_id,
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        "bindings": bindings,
        "render_bindings": render_bindings,
        "steps": [
            {
                "step_index": local_index,
                "source_state_id": str(fact_steps[source_index]["before_state_id"]),
                "action": json.loads(
                    json.dumps(fact_steps[source_index]["action"], ensure_ascii=False)
                ),
            }
            for local_index, source_index in enumerate(source_indices)
        ],
        "agent_visible": True,
    }


def _json_schema_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if value is None:
        return "null"
    return "string"


def _materialize_authoring_workflow(
    value: Any,
    facts: dict[str, Any],
    *,
    candidate_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Validate one semantic proposal, then compile and register its artifacts."""

    if not isinstance(value, dict) or set(value) != {
        "reason",
        "functions",
        "complete_function",
    }:
        raise ValueError("function_author_workflow_response_contract_invalid")
    reason = str(value.get("reason") or "").strip()
    if not reason:
        raise ValueError("function_author_reason_must_be_string")

    source_steps = [
        step for step in facts.get("steps") or () if isinstance(step, dict)
    ]
    source_indices = [int(step["source_step_index"]) for step in source_steps]
    source_positions = {
        source_index: position
        for position, source_index in enumerate(source_indices)
    }
    source_tools = {
        int(step["source_step_index"]): str(
            (step.get("action") or {}).get("tool") or ""
        )
        for step in source_steps
    }
    if not source_indices:
        raise ValueError("successful_source_actions_required")

    raw_functions = value.get("functions")
    raw_complete = value.get("complete_function")
    if not isinstance(raw_functions, list):
        raise ValueError("function_author_inventory_contract_invalid")
    raw_definitions: list[dict[str, Any]] = []
    raw_parameterization: list[dict[str, Any]] = []
    proposal_definition_fields = {
        "function_id",
        "name",
        "description",
        "occurrences",
        "parameters",
    }
    for raw_function in raw_functions:
        if (
            not isinstance(raw_function, dict)
            or set(raw_function) != proposal_definition_fields
        ):
            raise ValueError("function_author_inventory_definition_invalid")
        raw_definitions.append(
            {
                key: raw_function[key]
                for key in ("function_id", "name", "description", "occurrences")
            }
        )
        raw_parameterization.append(
            {
                "function_id": raw_function["function_id"],
                "parameters": raw_function["parameters"],
            }
        )
    definition_fields = {
        "function_id",
        "name",
        "description",
        "occurrences",
    }
    occurrence_fields = {"source_step_indices"}
    definitions: dict[str, dict[str, Any]] = {}
    occurrence_spans: dict[str, list[list[int]]] = {}
    representative_occurrence_indices: dict[str, int] = {}
    all_occurrences: list[tuple[int, str, int, list[int]]] = []
    for raw_definition in raw_definitions:
        if not isinstance(raw_definition, dict) or set(raw_definition) != definition_fields:
            raise ValueError("function_author_inventory_definition_invalid")
        function_id = str(raw_definition.get("function_id") or "").strip()
        name = str(raw_definition.get("name") or "").strip()
        description = str(raw_definition.get("description") or "").strip()
        raw_occurrences = raw_definition.get("occurrences")
        if (
            not function_id
            or not name
            or not description
            or function_id in definitions
            or not isinstance(raw_occurrences, list)
            or not raw_occurrences
        ):
            raise ValueError("function_author_inventory_definition_invalid")
        spans: list[list[int]] = []
        tool_shapes: list[tuple[str, ...]] = []
        for occurrence_index, raw_occurrence in enumerate(raw_occurrences):
            if (
                not isinstance(raw_occurrence, dict)
                or set(raw_occurrence) != occurrence_fields
            ):
                raise ValueError("function_author_inventory_occurrence_invalid")
            raw_span = raw_occurrence.get("source_step_indices")
            if (
                not isinstance(raw_span, list)
                or not raw_span
                or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in raw_span
                )
            ):
                raise ValueError("function_author_inventory_occurrence_invalid")
            span = list(raw_span)
            if span != sorted(set(span)) or any(
                item not in source_positions for item in span
            ):
                raise ValueError("function_author_inventory_occurrence_invalid")
            positions = [source_positions[item] for item in span]
            if positions != list(range(positions[0], positions[-1] + 1)):
                raise ValueError("function_author_inventory_occurrence_not_contiguous")
            tool_shapes.append(tuple(source_tools[item] for item in span))
            spans.append(span)
            all_occurrences.append((positions[0], function_id, occurrence_index, span))
        definitions[function_id] = {
            "function_id": function_id,
            "name": name,
            "description": description,
        }
        occurrence_spans[function_id] = spans
        shape_counts = {
            shape: tool_shapes.count(shape)
            for shape in tool_shapes
        }
        representative_occurrence_indices[function_id] = max(
            range(len(tool_shapes)),
            key=lambda index: shape_counts[tool_shapes[index]],
        )

    complete_fields = {
        "function_id",
        "name",
        "description",
        "source_step_indices",
    }
    if isinstance(raw_complete, dict) and "parameters" in raw_complete:
        raise ValueError(
            "function_author_inventory_complete_invalid:"
            "parameters_must_be_declared_on_a_function"
        )
    if not isinstance(raw_complete, dict) or set(raw_complete) != complete_fields:
        raise ValueError("function_author_inventory_complete_invalid")
    complete_id = str(raw_complete.get("function_id") or "").strip()
    complete_indices = raw_complete.get("source_step_indices")
    if (
        not complete_id
        or complete_id in definitions
        or not str(raw_complete.get("name") or "").strip()
        or not str(raw_complete.get("description") or "").strip()
    ):
        raise ValueError("function_author_inventory_complete_invalid")
    complete_indices_normalized = complete_indices != source_indices
    raw_complete = {
        **raw_complete,
        "source_step_indices": source_indices,
    }

    parameterization_fields = {"function_id", "parameters"}
    parameter_fields = {"name", "description", "bindings"}
    legacy_parameter_binding_fields = {"occurrence_index", "candidate_id"}
    semantic_parameter_binding_fields = {
        "occurrence_index",
        "source_step_index",
        "binding_kind",
    }
    parameters_by_function: dict[str, list[dict[str, Any]]] = {}
    selected_candidate_ids: set[str] = set()
    invocation_arguments: dict[tuple[str, int], dict[str, Any]] = {
        (function_id, occurrence_index): {}
        for function_id, spans in occurrence_spans.items()
        for occurrence_index in range(len(spans))
    }
    candidate_parameter_names: dict[str, str] = {}
    for raw_function_parameters in raw_parameterization:
        if (
            not isinstance(raw_function_parameters, dict)
            or set(raw_function_parameters) != parameterization_fields
        ):
            raise ValueError("function_author_parameterization_invalid")
        function_id = str(raw_function_parameters.get("function_id") or "").strip()
        raw_parameters = raw_function_parameters.get("parameters")
        if (
            function_id not in definitions
            or function_id in parameters_by_function
            or not isinstance(raw_parameters, list)
        ):
            raise ValueError("function_author_parameterization_invalid")
        parsed_parameters: list[dict[str, Any]] = []
        parameter_names: set[str] = set()
        for raw_parameter in raw_parameters:
            if not isinstance(raw_parameter, dict) or set(raw_parameter) != parameter_fields:
                raise ValueError("function_author_parameter_invalid")
            name = str(raw_parameter.get("name") or "").strip()
            description = str(raw_parameter.get("description") or "").strip()
            raw_bindings = raw_parameter.get("bindings")
            if (
                re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name) is None
                or name in parameter_names
                or not description
                or not isinstance(raw_bindings, list)
                or not raw_bindings
            ):
                raise ValueError("function_author_parameter_invalid")
            parameter_names.add(name)
            by_occurrence: dict[int, list[tuple[str, dict[str, Any]]]] = {}
            for raw_binding in raw_bindings:
                if not isinstance(raw_binding, dict):
                    raise ValueError("function_author_parameter_binding_invalid")
                raw_binding_fields = set(raw_binding)
                occurrence_index = raw_binding.get("occurrence_index")
                if (
                    isinstance(occurrence_index, bool)
                    or not isinstance(occurrence_index, int)
                    or occurrence_index not in range(len(occurrence_spans[function_id]))
                ):
                    raise ValueError("function_author_parameter_binding_invalid")
                span = occurrence_spans[function_id][occurrence_index]
                resolved_candidates: list[tuple[str, dict[str, Any]]] = []
                if raw_binding_fields == legacy_parameter_binding_fields:
                    candidate_id = str(
                        raw_binding.get("candidate_id") or ""
                    ).strip()
                    if candidate_id not in candidate_map:
                        raise ValueError(
                            "function_author_parameter_binding_evidence_missing"
                        )
                    resolved_candidates = [(candidate_id, candidate_map[candidate_id])]
                elif raw_binding_fields == semantic_parameter_binding_fields:
                    source_step_index = raw_binding.get("source_step_index")
                    binding_kind = str(
                        raw_binding.get("binding_kind") or ""
                    ).strip()
                    if (
                        isinstance(source_step_index, bool)
                        or not isinstance(source_step_index, int)
                        or binding_kind not in {"action_arg", "render_node"}
                    ):
                        raise ValueError(
                            "function_author_parameter_binding_invalid"
                        )
                    if source_step_index not in span:
                        raise ValueError(
                            "function_author_parameter_binding_outside_occurrence:"
                            f"function={function_id}:parameter={name}:"
                            f"occurrence={occurrence_index}:"
                            f"source_step={source_step_index}:span={span}"
                        )
                    for candidate_id, candidate in candidate_map.items():
                        suggested_name = str(
                            candidate.get("suggested_name") or ""
                        ).strip()
                        parameter_name_matches = bool(suggested_name) and (
                            name == suggested_name
                            or re.fullmatch(
                                re.escape(suggested_name) + r"_[2-9][0-9]*",
                                name,
                            )
                            is not None
                        )
                        if (
                            candidate.get("kind") == binding_kind
                            and candidate.get("source_step_index")
                            == source_step_index
                            and parameter_name_matches
                        ):
                            resolved_candidates.append((candidate_id, candidate))
                    if not resolved_candidates:
                        raise ValueError(
                            "function_author_parameter_binding_evidence_missing:"
                            f"function={function_id}:parameter={name}:"
                            f"occurrence={occurrence_index}:"
                            f"source_step={source_step_index}:kind={binding_kind}"
                        )
                else:
                    raise ValueError("function_author_parameter_binding_invalid")

                for candidate_id, candidate in resolved_candidates:
                    if candidate_id in selected_candidate_ids:
                        raise ValueError(
                            "function_author_parameter_binding_duplicate_evidence"
                        )
                    if candidate.get("source_step_index") not in span:
                        raise ValueError(
                            "function_author_parameter_binding_outside_occurrence"
                        )
                    suggested_name = str(
                        candidate.get("suggested_name") or ""
                    ).strip()
                    if suggested_name and not (
                        name == suggested_name
                        or re.fullmatch(
                            re.escape(suggested_name) + r"_[2-9][0-9]*",
                            name,
                        )
                    ):
                        raise ValueError(
                            "function_author_parameter_name_not_suggested:"
                            f"{candidate_id}:expected={suggested_name}:got={name}"
                        )
                    selected_candidate_ids.add(candidate_id)
                    candidate_parameter_names[candidate_id] = name
                    by_occurrence.setdefault(occurrence_index, []).append(
                        (candidate_id, candidate)
                    )
            if set(by_occurrence) != set(range(len(occurrence_spans[function_id]))):
                raise ValueError("function_author_parameter_occurrence_incomplete")
            for occurrence_index, bindings in sorted(by_occurrence.items()):
                values: set[str] = set()
                for _candidate_id, candidate in bindings:
                    if candidate["kind"] == "action_arg":
                        parameter_value = str(candidate.get("recorded_value") or "")
                    else:
                        parameter_value = str(
                            candidate.get("task_parameter_value")
                            or candidate.get("recorded_value")
                            or ""
                        )
                    values.add(parameter_value)
                if len(values) != 1 or not next(iter(values), ""):
                    raise ValueError("function_author_parameter_value_ambiguous")
                invocation_arguments[(function_id, occurrence_index)][name] = next(
                    iter(values)
                )
            parsed_parameters.append(
                {
                    "name": name,
                    "description": description,
                    "bindings_by_occurrence": by_occurrence,
                }
            )
        parameters_by_function[function_id] = parsed_parameters
    if set(parameters_by_function) != set(definitions):
        raise ValueError("function_author_parameterization_definition_mismatch")
    unselected_candidate_ids = sorted(set(candidate_map) - selected_candidate_ids)

    expected_function_ids = [*definitions, complete_id]
    registered_occurrences = sorted(all_occurrences)
    covered_local_indices = {
        index
        for _position, _function_id, _occurrence_index, span in registered_occurrences
        for index in span
    }
    uncovered_local_indices = [
        index for index in source_indices if index not in covered_local_indices
    ]

    selected_node_evidence: list[dict[str, Any]] = []
    for candidate_id, parameter_name in candidate_parameter_names.items():
        candidate = candidate_map[candidate_id]
        if candidate.get("kind") != "render_node":
            continue
        evidence = {
            key: json.loads(json.dumps(item, ensure_ascii=False))
            for key, item in candidate.items()
            if key != "kind"
        }
        evidence["parameter_name"] = parameter_name
        evidence["suggested_name"] = parameter_name
        selected_node_evidence.append(evidence)
    materialization_facts = json.loads(json.dumps(facts, ensure_ascii=False))
    materialization_facts["node_parameter_evidence"] = selected_node_evidence

    legacy_functions: list[dict[str, Any]] = []
    for function_id, definition in definitions.items():
        representative_occurrence_index = representative_occurrence_indices[
            function_id
        ]
        representative_span = occurrence_spans[function_id][
            representative_occurrence_index
        ]
        legacy_parameters: list[dict[str, Any]] = []
        for parameter in parameters_by_function[function_id]:
            for _candidate_id, candidate in parameter["bindings_by_occurrence"][
                representative_occurrence_index
            ]:
                if candidate.get("kind") != "action_arg":
                    continue
                legacy_parameters.append(
                    {
                        "name": parameter["name"],
                        "description": parameter["description"],
                        "source_step_index": int(candidate["source_step_index"]),
                        "arg_name": str(candidate["arg_name"]),
                    }
                )
        legacy_functions.append(
            {
                **definition,
                "source_step_indices": representative_span,
                "parameters": legacy_parameters,
            }
        )

    complete_parameters: list[dict[str, Any]] = []
    complete_names: dict[str, str] = {}
    for _position, function_id, occurrence_index, _span in registered_occurrences:
        for parameter in parameters_by_function[function_id]:
            parameter_name = parameter["name"]
            parameter_value = invocation_arguments[(function_id, occurrence_index)][
                parameter_name
            ]
            complete_name = parameter_name
            suffix = 2
            while (
                complete_name in complete_names
                and complete_names[complete_name] != parameter_value
            ):
                complete_name = f"{parameter_name}_{suffix}"
                suffix += 1
            complete_names[complete_name] = parameter_value
            for _candidate_id, candidate in parameter["bindings_by_occurrence"][
                occurrence_index
            ]:
                if candidate.get("kind") != "action_arg":
                    continue
                complete_parameters.append(
                    {
                        "name": complete_name,
                        "description": parameter["description"],
                        "source_step_index": int(candidate["source_step_index"]),
                        "arg_name": str(candidate["arg_name"]),
                    }
                )
    legacy_plan = {
        "reason": reason,
        "plan": {
            "functions": legacy_functions,
            "complete_function": {
                **raw_complete,
                "parameters": complete_parameters,
            },
        },
    }
    materialized = _materialize_authoring_plan(
        legacy_plan,
        materialization_facts,
        synthesize_missing_locals=False,
    )
    bundle = materialized["bundle"]
    materialized_ids = {
        str(function.get("function_id") or "")
        for function in bundle.get("functions") or ()
        if isinstance(function, dict)
    }
    redundant_definition_ids = {
        function_id
        for function_id in definitions
        if occurrence_spans[function_id] == [source_indices]
    }
    if (
        complete_id not in materialized_ids
        or set(expected_function_ids) - materialized_ids - redundant_definition_ids
    ):
        raise ValueError("function_author_registration_materialization_mismatch")
    source_calls: list[dict[str, Any]] = []
    for _position, function_id, occurrence_index, _span in registered_occurrences:
        call_function_id = function_id
        call_arguments = invocation_arguments[(function_id, occurrence_index)]
        if function_id not in materialized_ids:
            call_function_id = complete_id
            call_arguments = (bundle.get("arguments") or {}).get(complete_id) or {}
        source_calls.append(
            {
                "function_id": call_function_id,
                "arguments": json.loads(
                    json.dumps(call_arguments, ensure_ascii=False)
                ),
            }
        )
    if not source_calls:
        source_calls.append(
            {
                "function_id": complete_id,
                "arguments": json.loads(
                    json.dumps(
                        (bundle.get("arguments") or {}).get(complete_id) or {},
                        ensure_ascii=False,
                    )
                ),
            }
        )
    functions_by_id = {
        str(function.get("function_id") or ""): function
        for function in bundle.get("functions") or ()
        if isinstance(function, dict)
    }
    from omniflow.functions.artifact import bind_function, parse_function_artifact

    for call in source_calls:
        bind_function(
            parse_function_artifact(functions_by_id[call["function_id"]]),
            call["arguments"],
        )
    return {
        "reason": materialized["reason"],
        "bundle": bundle,
        "source_calls": source_calls,
        "authoring_workflow": {
            "schema_version": "omniflow.function-authoring-workflow.v2",
            "definition_count": len(definitions),
            "invocation_count": len(source_calls),
            "definitions": [
                {
                    "function_id": function_id,
                    "occurrence_count": len(occurrence_spans[function_id]),
                    "representative_source_step_indices": occurrence_spans[
                        function_id
                    ][representative_occurrence_indices[function_id]],
                    "registered": function_id in materialized_ids,
                }
                for function_id in definitions
            ],
            "complete_function_id": complete_id,
            "complete_source_indices_normalized": complete_indices_normalized,
            "uncovered_local_source_step_indices": uncovered_local_indices,
            "unselected_candidate_ids": unselected_candidate_ids,
            "validation_notes": (
                "The Agent authored semantic parameter-to-step binding requests. "
                "Compiler Harness resolved every request against immutable source "
                "evidence, validated each selected segment, selected a representative "
                "occurrence shape, and materialized the complete source sequence."
            ),
        },
    }


def _complete_source_authoring_plan(facts: dict[str, Any]) -> dict[str, Any]:
    """Build a valid state-level plan when authoring needs deterministic fallback."""

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
    local_functions = _state_level_local_authoring_functions(
        facts,
        source_indices=source_indices,
    )
    return {
        "reason": "Compiler fallback preserving state-level local Functions and the complete successful source workflow.",
        "plan": {
            "functions": local_functions,
            "complete_function": {
                "function_id": "complete_source_workflow",
                "name": name,
                "description": description,
                "source_step_indices": source_indices,
                "parameters": [],
            },
        },
    }


def _state_level_local_authoring_functions(
    facts: dict[str, Any],
    *,
    source_indices: list[int],
    reserved_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Build executable one-transition Functions for uncovered source states."""

    reserved = set(reserved_ids or ())
    by_index = {
        int(step.get("source_step_index")): step
        for step in facts.get("steps") or ()
        if isinstance(step, dict)
        and isinstance(step.get("source_step_index"), int)
    }
    result: list[dict[str, Any]] = []
    for source_index in source_indices:
        step = by_index.get(int(source_index))
        if step is None:
            continue
        action = step.get("action") if isinstance(step.get("action"), dict) else {}
        tool = str(action.get("tool") or "transition").strip() or "transition"
        function_id = f"state_transition_{int(source_index):03d}"
        suffix = 2
        while function_id in reserved:
            function_id = f"state_transition_{int(source_index):03d}_{suffix}"
            suffix += 1
        reserved.add(function_id)
        readable_tool = tool.replace("_", " ")
        result.append(
            {
                "function_id": function_id,
                "name": f"State transition {int(source_index) + 1}: {readable_tool}",
                "description": (
                    f"Perform the recorded {readable_tool} transition from its "
                    "matching GUI state."
                ),
                "source_step_indices": [int(source_index)],
                # The compiler derives eligible input_text bindings from the
                # immutable facts, so synthesized Functions cannot lose args.
                "parameters": [],
            }
        )
    return result


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
    return _materialize_authoring_plan(
        {
            "reason": "Registered the complete converted source workflow.",
            "plan": {
                "functions": [],
                "complete_function": {
                    "function_id": function_id,
                    "name": name,
                    "description": description,
                    "source_step_indices": [
                        int(step["source_step_index"])
                        for step in source_steps
                    ],
                    "parameters": [],
                },
            },
        },
        facts,
        synthesize_missing_locals=False,
    )


def _materialize_authoring_plan(
    value: Any,
    facts: dict[str, Any],
    *,
    synthesize_missing_locals: bool = True,
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
    materialization_notes: list[str] = []
    candidates = {
        (candidate["source_step_index"], candidate["arg_name"]): candidate
        for candidate in _source_parameter_candidates(facts)
    }
    functions: list[dict[str, Any]] = []
    materialized_function_ids: set[str] = set()
    arguments: dict[str, dict[str, Any]] = {}
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

    # Deduplicate before measuring local coverage.  Otherwise an authoring
    # response that repeats the complete source sequence as a "local" Function
    # makes every transition appear covered; the duplicate is then removed and
    # the Store silently loses all state-level local Functions.
    covered_source_indices = {
        int(index)
        for raw_function in deduplicated_functions
        if isinstance(raw_function, dict)
        for index in (raw_function.get("source_step_indices") or [])
        if isinstance(index, int) and not isinstance(index, bool)
    }
    missing_local_indices = [
        int(step["source_step_index"])
        for step in source_steps
        if isinstance(step, dict)
        and isinstance(step.get("source_step_index"), int)
        and int(step["source_step_index"]) not in covered_source_indices
    ]
    if synthesize_missing_locals and missing_local_indices and len(source_steps) > 1:
        reserved_ids = {
            str(item.get("function_id") or "").strip()
            for item in deduplicated_functions
            if isinstance(item, dict)
        }
        raw_functions = [
            *deduplicated_functions,
            *_state_level_local_authoring_functions(
                facts,
                source_indices=missing_local_indices,
                reserved_ids=reserved_ids,
            ),
        ]
        materialization_notes.append(
            "Compiler synthesized missing state-level local Functions so every source transition remains reusable."
        )
    else:
        raw_functions = deduplicated_functions
    planned_functions = [
        *((raw_function, False) for raw_function in raw_functions),
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
            "render_bindings": [],
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
        for proposal in parameter_proposals:
            proposal_step = proposal.get("step_index")
            proposal_arg = str(proposal.get("arg_name") or "")
            if not isinstance(proposal_step, int) or not proposal_arg:
                continue
            candidate = candidates.get((indices[proposal_step], proposal_arg))
            if candidate is None:
                continue
            if candidate.get("evidence") != "task_parameter_filename_stem":
                continue
            proposal["name"] = str(candidate["suggested_name"])
            suffix = str(candidate.get("fixed_suffix") or "")
            proposal["description"] = (
                "The requested file name without the fixed "
                f"{suffix} extension selected by this Function."
            )
        if any(
            candidate.get("evidence") == "task_parameter_filename_stem"
            for candidate in candidates.values()
        ):
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
        parameter_names_by_value: dict[tuple[str, str], str] = {}
        used_materialized_names: set[str] = set()
        for proposal in parameter_proposals:
            proposal_step = proposal.get("step_index")
            proposal_arg = str(proposal.get("arg_name") or "")
            if not isinstance(proposal_step, int) or not proposal_arg:
                continue
            candidate = candidates.get((indices[proposal_step], proposal_arg))
            if candidate is None:
                continue
            base_name = str(proposal.get("name") or "input_value").strip()
            recorded_value = " ".join(
                str(candidate.get("recorded_value") or "").casefold().split()
            )
            value_key = (base_name, recorded_value)
            materialized_name = parameter_names_by_value.get(value_key)
            if materialized_name is None:
                materialized_name = base_name
                suffix = 2
                while materialized_name in used_materialized_names:
                    materialized_name = f"{base_name}_{suffix}"
                    suffix += 1
                parameter_names_by_value[value_key] = materialized_name
                used_materialized_names.add(materialized_name)
            proposal["name"] = materialized_name
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
        source_arguments = {
            name: value
            for name, value in source_arguments.items()
            if name in function["input_schema"]["properties"]
        }
        node_parameter_literals = _materialize_node_parameters(
            function,
            facts.get("node_parameter_evidence") or (),
            source_indices=indices,
            source_arguments=source_arguments,
        )
        parameter_literals = _parameter_literal_map(
            parameter_proposals,
            candidates=candidates,
            source_indices=indices,
        )
        parameter_literals.update(node_parameter_literals)
        function["name"] = _redact_parameter_literals(
            function["name"],
            function,
            parameter_literals=parameter_literals,
        )
        function["description"] = _description_with_action_plan(
            description,
            function=function,
            source_steps=source_steps,
            source_indices=indices,
            source_step_positions=source_step_positions,
            parameter_literals=parameter_literals,
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


def _optional_onboarding_checker_id(
    step: dict[str, Any], *, onboarding_active: bool
) -> str | None:
    """Return the shared checker for a first-run setup action, if present.

    First-run pages are device state, not task progress.  A source recording
    can contain them while an initialized target does not.  Keep the
    classification deliberately generic and limited to a contiguous prefix
    with an introduction-like container and next control; ordinary task
    actions are never removed from the Function.
    """
    action = step.get("action") if isinstance(step.get("action"), dict) else {}
    if str(action.get("action_type") or "").strip() != "click":
        return None
    observation = step.get("observation")
    if not isinstance(observation, dict):
        return None
    xml = str(observation.get("xml") or observation.get("forest") or "")
    normalized = " ".join(xml.casefold().split())
    if _is_first_run_onboarding_page(xml):
        return "advance_first_run_onboarding"
    if not onboarding_active and 'text="get started"' in normalized:
        return "dismiss_transient_overlay"
    if not onboarding_active:
        return None
    if 'text="setup"' in normalized:
        return "dismiss_initial_setup"
    if 'text="warning!"' in normalized and 'text="ok"' in normalized:
        return "dismiss_informational_dialog"
    return None


def _is_first_run_onboarding_page(xml: str) -> bool:
    """Recognize a generic, contiguous first-run introduction page.

    Some AndroidWorld source recordings expose a ViewPager whose navigation
    control is an icon-only ``next_button``.  It has no ``Get started`` label,
    so label-only onboarding detection would incorrectly bake those clicks
    into every Function.  Require both an introduction-like container and its
    next control to avoid classifying ordinary task navigation as setup.
    """
    if not xml.strip():
        return False
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return False
    has_intro_container = False
    for container in root.iter("node"):
        container_id = str(container.get("resource-id") or "").casefold()
        container_class = str(container.get("class") or "").casefold()
        if not (
            any(marker in container_id for marker in ("introduction", "onboarding", "welcome"))
            or "viewpager" in container_class
        ):
            continue
        has_intro_container = True
    if not has_intro_container:
        return False
    for node in root.iter("node"):
        if node.get("clickable") != "true":
            continue
        node_id = str(node.get("resource-id") or "").casefold()
        label = " ".join(
            " ".join(str(node.get(attribute) or "").casefold().split())
            for attribute in ("text", "content-desc")
        ).strip()
        if "next_button" in node_id or label in {"next", "continue", "get started"}:
            return True
    return False


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
    parameter_literals: dict[str, str] | None = None,
) -> str:
    parameter_by_target = {
        str(binding["target"]): str(binding["source"]).removeprefix("$.arguments.")
        for binding in function.get("bindings") or ()
        if isinstance(binding, dict)
    }
    parameter_by_render_step = {
        int(binding["step_index"]): str(binding["source"]).removeprefix(
            "$.arguments."
        )
        for binding in function.get("render_bindings") or ()
        if isinstance(binding, dict)
        and isinstance(binding.get("step_index"), int)
        and str(binding.get("source") or "").startswith("$.arguments.")
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
        render_parameter = parameter_by_render_step.get(local_index)
        if render_parameter and tool in {"click", "long_press"}:
            semantic_args["target"] = f"<{render_parameter}>"
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
    semantic_description = _redact_parameter_literals(
        semantic_description,
        function,
        parameter_literals=parameter_literals,
    )
    encoded_plan = ";".join(plan)
    return f"{semantic_description} Action plan: {encoded_plan}"


def _redact_parameter_literals(
    text: str,
    function: dict[str, Any],
    *,
    parameter_literals: dict[str, str] | None = None,
) -> str:
    """Remove source-instance values from agent-facing Function metadata.

    The authoring model is instructed to describe semantic values rather than
    copy source-instance literals, but the compiler must enforce that contract:
    source values remain in the private source arguments/bindings while the
    Router sees the generated parameter names.  This is deliberately limited
    to values proven by a Function binding and never touches coordinates or
    unbound UI labels.
    """

    literals: dict[str, str] = {
        str(literal): str(parameter_name)
        for parameter_name, literal in (parameter_literals or {}).items()
        if str(literal).strip() and str(parameter_name).strip()
    }
    steps = function.get("steps") or ()
    for raw_binding in function.get("bindings") or ():
        if not isinstance(raw_binding, dict):
            continue
        parameter_name = str(raw_binding.get("source") or "")
        if not parameter_name.startswith("$.arguments."):
            continue
        parameter_name = parameter_name.removeprefix("$.arguments.").strip()
        target = str(raw_binding.get("target") or "")
        match = re.fullmatch(
            r"\$\.steps\[(\d+)\]\.action\.args\.([A-Za-z_][A-Za-z0-9_]*)",
            target,
        )
        if match is None:
            continue
        step_index = int(match.group(1))
        argument_name = match.group(2)
        if step_index < 0 or step_index >= len(steps):
            continue
        step = steps[step_index]
        action = step.get("action") if isinstance(step, dict) else None
        args = action.get("args") if isinstance(action, dict) else None
        value = args.get(argument_name) if isinstance(args, dict) else None
        if isinstance(value, str) and value.strip():
            literals.setdefault(value, parameter_name)
    for raw_binding in function.get("render_bindings") or ():
        if not isinstance(raw_binding, dict):
            continue
        source = str(raw_binding.get("source") or "")
        if not source.startswith("$.arguments."):
            continue
        literal = str(raw_binding.get("recorded_value") or "").strip()
        parameter_name = source.removeprefix("$.arguments.").strip()
        if literal and parameter_name:
            literals.setdefault(literal, parameter_name)

    redacted = str(text or "")
    for literal, parameter_name in sorted(
        literals.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        redacted = redacted.replace(literal, f"<{parameter_name}>")
    return redacted


def _parameter_literal_map(
    parameters: list[dict[str, Any]],
    *,
    candidates: dict[tuple[int, str], dict[str, Any]],
    source_indices: list[int],
) -> dict[str, str]:
    literals: dict[str, str] = {}
    for parameter in parameters:
        if not isinstance(parameter, dict):
            continue
        local_index = parameter.get("step_index")
        if not isinstance(local_index, int) or not 0 <= local_index < len(source_indices):
            continue
        arg_name = str(parameter.get("arg_name") or "").strip()
        candidate = candidates.get((source_indices[local_index], arg_name))
        name = str(parameter.get("name") or "").strip()
        if candidate is None or not name:
            continue
        literal = str(candidate.get("recorded_value") or "").strip()
        if literal:
            literals[name] = literal
    return literals


def _materialize_node_parameters(
    function: dict[str, Any],
    candidates: Any,
    *,
    source_indices: list[int],
    source_arguments: dict[str, Any],
) -> dict[str, str]:
    """Bind task values into the source node rendered for point actions."""

    literals: dict[str, str] = {}
    selected: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        source_index = candidate.get("source_step_index")
        if (
            not isinstance(source_index, int)
            or source_index not in source_indices
        ):
            continue
        selected.setdefault(int(source_index), []).append(candidate)
    render_bindings = function.setdefault("render_bindings", [])
    if not isinstance(render_bindings, list):
        raise ValueError("function_render_bindings_invalid")
    properties = function["input_schema"]["properties"]
    required = function["input_schema"]["required"]
    for local_index, source_index in enumerate(source_indices):
        for candidate in selected.get(int(source_index), ()):
            parameter_name = str(candidate.get("parameter_name") or "").strip()
            task_value = str(candidate.get("task_parameter_value") or "")
            recorded_value = str(candidate.get("recorded_value") or "")
            if not parameter_name or not task_value or not recorded_value:
                continue
            base_parameter_name = parameter_name
            suffix = 2
            while (
                parameter_name in source_arguments
                and str(source_arguments[parameter_name]) != task_value
            ):
                parameter_name = f"{base_parameter_name}_{suffix}"
                suffix += 1
            if parameter_name not in properties:
                properties[parameter_name] = {
                    "type": "string",
                    "description": (
                        f"The requested {parameter_name.replace('_', ' ')} "
                        "visible on the selected GUI item."
                    ),
                }
                required.append(parameter_name)
            source_arguments[parameter_name] = task_value
            literals[parameter_name] = recorded_value
            binding = {
                "source": f"$.arguments.{parameter_name}",
                "step_index": local_index,
                "node_id": str(candidate.get("node_id") or "").strip(),
                "attribute": str(candidate.get("attribute") or "").strip(),
                "recorded_value": recorded_value,
            }
            if not binding["node_id"] or not binding["attribute"]:
                raise ValueError("function_node_render_binding_evidence_invalid")
            duplicate = any(
                isinstance(existing, dict)
                and all(existing.get(key) == binding[key] for key in binding)
                for existing in render_bindings
            )
            if not duplicate:
                render_bindings.append(binding)
    return literals


def _source_node_parameter_evidence(
    *,
    source_step: dict[str, Any],
    action: dict[str, Any],
    source_step_index: int,
    task_parameters: Any,
) -> list[dict[str, Any]]:
    """Find task values embedded in the source node receiving a point action."""

    tool = str(action.get("tool") or "").strip()
    args = action.get("args") if isinstance(action.get("args"), dict) else {}
    if tool not in {"click", "long_press"} or not all(
        args.get(key) is not None for key in ("x", "y")
    ):
        return []
    observation = source_step.get("observation")
    xml = observation.get("xml") if isinstance(observation, dict) else None
    if not isinstance(xml, str) or not xml.strip() or not isinstance(task_parameters, dict):
        return []
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []
    try:
        width = float(root.attrib.get("width") or 0)
        height = float(root.attrib.get("height") or 0)
        point = (float(args["x"]) / 1000.0 * width, float(args["y"]) / 1000.0 * height)
    except (TypeError, ValueError):
        return []
    if width <= 0 or height <= 0:
        return []
    parent_by_node = {
        child: parent
        for parent in root.iter()
        for child in parent
    }
    bounds_by_node = {
        node: _compiler_parse_bounds(node.attrib.get("bounds"))
        for node in root.iter("node")
    }

    def point_distance_squared(bounds: tuple[float, float, float, float]) -> float:
        dx = max(bounds[0] - point[0], 0.0, point[0] - bounds[2])
        dy = max(bounds[1] - point[1], 0.0, point[1] - bounds[3])
        return dx * dx + dy * dy

    def action_anchor_area(node: ET.Element) -> float | None:
        current: ET.Element | None = node
        while current is not None:
            bounds = bounds_by_node.get(current)
            if (
                bounds is not None
                and bounds[0] <= point[0] <= bounds[2]
                and bounds[1] <= point[1] <= bounds[3]
            ):
                return (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
            current = parent_by_node.get(current)
        return None

    evidence: list[dict[str, Any]] = []
    seen_targets: dict[tuple[str, str, str], dict[str, Any]] = {}
    ranked_matches: list[
        tuple[tuple[float, int, float], str, str, str, ET.Element]
    ] = []
    for parameter_name, task_value in iter_task_parameter_values(task_parameters):
        if parameter_name in {"seed", "source_seed", "evaluation_seed", "task_random_seed", "noise_candidates"}:
            continue
        normalized_task_value = task_value.casefold()
        for node in root.iter("node"):
            node_id = str(node.attrib.get("id") or "").strip()
            bounds = bounds_by_node.get(node)
            if not node_id or bounds is None:
                continue
            for attribute in ("text", "content-desc"):
                label = str(node.attrib.get(attribute) or "")
                start = label.casefold().find(normalized_task_value)
                if start < 0:
                    continue
                recorded_value = label[start : start + len(task_value)]
                anchor_area = action_anchor_area(node)
                if anchor_area is None:
                    continue
                ranked_matches.append(
                    (
                        (
                            anchor_area,
                            0
                            if re.search(
                                r"(?:^|_)(?:title|name|label|identifier)$",
                                str(parameter_name).casefold(),
                            )
                            else 1,
                            point_distance_squared(bounds),
                        ),
                        str(parameter_name),
                        task_value,
                        recorded_value,
                        node,
                    )
                )
                break
    if not ranked_matches:
        return []
    best_score = min(match[0] for match in ranked_matches)
    for score, parameter_name, task_value, recorded_value, node in ranked_matches:
        if score != best_score:
            continue
        attribute = next(
            name
            for name in ("text", "content-desc")
            if recorded_value in str(node.attrib.get(name) or "")
        )
        label = str(node.attrib.get(attribute) or "")
        target = (
            str(node.attrib["id"]),
            attribute,
            recorded_value.casefold(),
        )
        existing = seen_targets.get(target)
        if existing is not None:
            if (
                existing["task_parameter_value"] != task_value
            ):
                raise ValueError("function_node_parameter_target_ambiguous")
            continue
        item = {
            "source_step_index": int(source_step_index),
            "tool": tool,
            "parameter_name": parameter_name,
            "suggested_name": parameter_name,
            "task_parameter_value": task_value,
            "recorded_value": recorded_value,
            "node_id": str(node.attrib["id"]),
            "attribute": attribute,
            "node_label": label,
        }
        seen_targets[target] = item
        evidence.append(item)
    return evidence


def _compiler_parse_bounds(value: Any) -> tuple[float, float, float, float] | None:
    match = re.fullmatch(
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
        r"\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
        str(value or "").strip(),
    )
    if match is None:
        return None
    values = match.groups()
    return (
        float(values[0]),
        float(values[1]),
        float(values[2]),
        float(values[3]),
    )


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
        unbound_binding_steps = []
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
            unbound_binding_steps.append(binding_step_index)
        if not unbound_binding_steps:
            continue
        if name in values_by_name and values_by_name[name] != recorded_value:
            raise ValueError("function_author_plan_parameter_name_ambiguous")
        if name not in function["input_schema"]["properties"]:
            definition = {"type": "string"}
            if description:
                definition["description"] = description
            function["input_schema"]["properties"][name] = definition
            function["input_schema"]["required"].append(name)
        for binding_step_index in unbound_binding_steps:
            target = (
                f"$.steps[{binding_step_index}].action.args.{arg_name}"
            )
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


def _remove_compiler_entry_actions(
    raw_functions: list[Any],
    *,
    raw_payload: dict[str, Any],
) -> list[Any]:
    """Remove only the source app-entry action from pre-authored Functions.

    Older compiler output may already contain an injected ``open_app`` at the
    beginning of a complete Function.  The shared package-mismatch Checker is
    the owner of that environment recovery.  Match the entry by its recorded
    source state, so a later ``open_app`` that is actual task progress remains
    intact.
    """
    source_steps = raw_payload.get("steps")
    if not isinstance(source_steps, list):
        return raw_functions
    source_has_main_action = any(
        isinstance(source_step, dict)
        and str(
            (source_step.get("action") or {}).get("action_type") or ""
        ).strip()
        not in {"answer", "status", "unknown", "open_app"}
        and not _is_transient_system_action(source_step)
        for source_step in source_steps
    )
    effective_steps = [
        (index, step)
        for index, step in enumerate(source_steps)
        if isinstance(step, dict)
        and isinstance(step.get("action"), dict)
        and str(step["action"].get("action_type") or "").strip()
        not in {"answer", "status", "unknown"}
        and not _is_transient_system_action(step)
    ]
    if len(effective_steps) <= 1:
        return raw_functions
    _first_source_index, first_source_step = effective_steps[0]
    entry_state_ids = {
        state_id(step.get("observation"))
        for _index, step in effective_steps[:2]
    }
    if str(first_source_step["action"].get("action_type") or "").strip() == "open_app":
        entry_state_ids = {
            state_id(step.get("observation"))
            for _index, step in effective_steps
            if str(step["action"].get("action_type") or "").strip() == "open_app"
        }
    if not entry_state_ids:
        return raw_functions
    result: list[Any] = []
    for raw_function in raw_functions:
        if not isinstance(raw_function, dict):
            result.append(raw_function)
            continue
        steps = raw_function.get("steps")
        if not isinstance(steps, list) or len(steps) <= 1:
            result.append(raw_function)
            continue
        first = steps[0] if isinstance(steps[0], dict) else {}
        first_action = first.get("action") if isinstance(first, dict) else {}
        first_state_id = str(first.get("source_state_id") or "").strip()
        if (
            not isinstance(first_action, dict)
            or str(first_action.get("tool") or "").strip() != "open_app"
            or first_state_id not in entry_state_ids
        ):
            result.append(raw_function)
            continue
        updated = json.loads(json.dumps(raw_function, ensure_ascii=False))
        updated_steps = []
        removed_count = 0
        for candidate in updated["steps"]:
            action = candidate.get("action") if isinstance(candidate, dict) else None
            source_id = str(candidate.get("source_state_id") or "").strip() if isinstance(candidate, dict) else ""
            if (
                isinstance(action, dict)
                and str(action.get("tool") or "").strip() == "open_app"
                and source_id in entry_state_ids
                and any(
                    isinstance(rest, dict)
                    and str((rest.get("action") or {}).get("tool") or "").strip()
                    != "open_app"
                    for rest in updated["steps"][removed_count + 1 :]
                )
            ):
                removed_count += 1
                continue
            break
        for step_index, step in enumerate(updated["steps"][removed_count:]):
            if not isinstance(step, dict):
                raise ValueError("function_entry_action_step_invalid")
            step["step_index"] = step_index
            updated_steps.append(step)
        if not updated_steps:
            result.append(raw_function)
            continue
        updated["steps"] = updated_steps
        updated_bindings: list[dict[str, Any]] = []
        for binding in updated.get("bindings") or []:
            if not isinstance(binding, dict):
                raise ValueError("function_entry_action_binding_invalid")
            target = str(binding.get("target") or "")
            match = re.match(r"^\$\.steps\[(\d+)](.*)$", target)
            if match is None:
                updated_bindings.append(binding)
                continue
            target_index = int(match.group(1))
            if target_index < removed_count:
                raise ValueError("function_entry_action_binding_invalid")
            binding["target"] = f"$.steps[{target_index - removed_count}]{match.group(2)}"
            updated_bindings.append(binding)
        updated["bindings"] = updated_bindings
        result.append(updated)
    return result


def _remove_compiler_environment_actions(
    raw_functions: list[Any],
    *,
    raw_payload: dict[str, Any],
) -> list[Any]:
    """Remove environment entry actions from every authored Function.

    The authoring model receives only the compiler's business-flow facts, but
    older or imperfect responses can still copy an app launch or leading
    back-navigation into a Function.  Those actions belong to the shared
    expected-app Checker, never to reusable business memory.  Reindex the
    surviving steps and bindings without changing any other action.
    """
    source_steps = raw_payload.get("steps")
    if not isinstance(source_steps, list):
        return raw_functions
    source_has_main_action = any(
        isinstance(source_step, dict)
        and str(
            (source_step.get("action") or {}).get("action_type") or ""
        ).strip()
        not in {"answer", "status", "unknown", "open_app"}
        and not _is_transient_system_action(source_step)
        for source_step in source_steps
    )
    launcher_entry_indices = _recorded_launcher_entry_indices(source_steps)
    launcher_entry_state_ids = {
        state_id(source_steps[index].get("observation"))
        for index in launcher_entry_indices
    }
    entry_state_ids: set[str] = set()
    for step in source_steps:
        if not isinstance(step, dict) or not isinstance(step.get("action"), dict):
            continue
        action_type = str(step["action"].get("action_type") or "").strip()
        if action_type not in {"navigate_back", "navigate_home", "open_app"}:
            break
        entry_state_ids.add(state_id(step.get("observation")))
    result: list[Any] = []
    for raw_function in raw_functions:
        if not isinstance(raw_function, dict):
            result.append(raw_function)
            continue
        steps = raw_function.get("steps")
        if not isinstance(steps, list):
            result.append(raw_function)
            continue
        updated = json.loads(json.dumps(raw_function, ensure_ascii=False))
        kept_steps: list[dict[str, Any]] = []
        removed_indices: set[int] = set()
        for original_index, candidate in enumerate(steps):
            if not isinstance(candidate, dict):
                raise ValueError("function_environment_step_invalid")
            action = candidate.get("action")
            source_id = str(candidate.get("source_state_id") or "").strip()
            tool = str(action.get("tool") or "").strip() if isinstance(action, dict) else ""
            key = str(action.get("args", {}).get("key") or "").strip() if isinstance(action, dict) else ""
            is_entry_navigation = (
                source_id in entry_state_ids
                and tool == "press_key"
                and key in {"back", "home"}
            )
            is_launcher_app_entry = source_id in launcher_entry_state_ids
            if (
                (tool == "open_app" and source_has_main_action)
                or is_entry_navigation
                or is_launcher_app_entry
            ):
                removed_indices.add(original_index)
                continue
            candidate["step_index"] = len(kept_steps)
            kept_steps.append(candidate)
        if not kept_steps:
            continue
        updated["steps"] = kept_steps
        updated_bindings: list[dict[str, Any]] = []
        for binding in updated.get("bindings") or []:
            if not isinstance(binding, dict):
                raise ValueError("function_environment_binding_invalid")
            match = re.match(r"^\$\.steps\[(\d+)\](.*)$", str(binding.get("target") or ""))
            if match is None:
                updated_bindings.append(binding)
                continue
            target_index = int(match.group(1))
            if target_index in removed_indices:
                continue
            shift = sum(index < target_index for index in removed_indices)
            binding["target"] = f"$.steps[{target_index - shift}]{match.group(2)}"
            updated_bindings.append(binding)
        updated["bindings"] = updated_bindings
        result.append(updated)
    return result


def _recorded_launcher_entry_indices(source_steps: list[Any]) -> set[int]:
    """Find a recorded Launcher-to-app prefix owned by runtime recovery.

    OOB source collection can represent opening the task app as physical
    Launcher swipes/clicks instead of a canonical ``open_app`` action.  A
    target AndroidWorld episode already initializes its task app, so replaying
    that prefix attempts to map a Launcher point against the in-app page.  Only
    classify a leading Launcher prefix when it demonstrably transitions into
    a non-system app and a later successful business action remains.  SystemUI
    gestures (for example brightness or Wi-Fi tasks) stay in the Function.
    """

    launcher_prefix: list[int] = []
    for index, step in enumerate(source_steps):
        if not isinstance(step, dict) or not isinstance(step.get("action"), dict):
            return set()
        action_type = str(step["action"].get("action_type") or "").strip()
        if action_type in {"answer", "status", "unknown"}:
            continue
        if (step.get("result") or {}).get("success") is not True:
            return set()
        before_package = _primary_observation_package(step.get("observation"))
        if "launcher" not in before_package.casefold():
            return set()
        launcher_prefix.append(index)
        after_package = _primary_observation_package(step.get("next_observation"))
        if not after_package or "launcher" in after_package.casefold():
            continue
        if any(
            marker in after_package.casefold()
            for marker in ("systemui", "inputmethod", "permissioncontroller")
        ):
            return set()
        has_later_business_action = any(
            isinstance(candidate, dict)
            and isinstance(candidate.get("action"), dict)
            and (candidate.get("result") or {}).get("success") is True
            and str(candidate["action"].get("action_type") or "").strip()
            not in {"answer", "status", "unknown", "open_app"}
            and not _is_transient_system_action(candidate)
            for candidate in source_steps[index + 1 :]
        )
        return set(launcher_prefix) if has_later_business_action else set()
    return set()


def _validate_materialized_function_artifacts(
    authored: dict[str, Any],
    *,
    raw_payload: dict[str, Any],
) -> None:
    """Validate a model-authored Function before accepting its attempt.

    Materialization proves that source steps and parameter candidates are
    grounded, while ``parse_function_artifact`` owns the complete executable
    schema, including render-binding uniqueness.  Running both inside the
    authoring retry loop prevents a late artifact error from bypassing the
    existing three-attempt recovery and deterministic source-workflow
    fallback.
    """

    from omniflow.functions.artifact import parse_function_artifact

    bundle = authored.get("bundle")
    if not isinstance(bundle, dict):
        raise ValueError("function_author_bundle_must_be_object_or_null")
    raw_functions = bundle.get("functions")
    if not isinstance(raw_functions, list) or not raw_functions:
        raise ValueError("function_bundle_functions_required")
    validated_functions = _remove_compiler_entry_actions(
        raw_functions,
        raw_payload=raw_payload,
    )
    validated_functions = _remove_compiler_environment_actions(
        validated_functions,
        raw_payload=raw_payload,
    )
    if not validated_functions:
        raise ValueError("function_bundle_functions_required")
    for value in validated_functions:
        parse_function_artifact(value)


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
    onboarding_active = False
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
        onboarding_checker_id = _optional_onboarding_checker_id(
            source_step, onboarding_active=onboarding_active
        )
        if onboarding_checker_id is not None:
            onboarding_active = True
            optional_checker_actions.append(
                {
                    "source_step_index": source_index,
                    "checker_id": onboarding_checker_id,
                    "reason": "first_run_onboarding_setup_is_optional",
                }
            )
            continue
        if onboarding_active:
            onboarding_active = False
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
        "render_bindings": [],
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
