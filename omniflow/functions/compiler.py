from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable
import xml.etree.ElementTree as ET

from omniflow.core.trajectory import require_complete_source_run_log, state_id, observation_display
from omniflow.functions.management import iter_task_parameter_values
from omniflow.runlog import project_run_log_step_actions
from omniflow.runtime.checker import (
    DEFAULT_CHECKER_LIBRARY_PATH,
    CheckerLibrary,
    validate_checker_rule,
)

# Mechanically extracted modules; names re-exported for existing importers.
from omniflow.functions.compiler_common import (  # noqa: F401
    _agent_owned_function_artifact,
    _compiler_parse_bounds,
    _json_schema_type,
    _source_parameter_candidates,
    _state_level_local_authoring_functions,
)
from omniflow.functions.authoring import (  # noqa: F401
    _PATCH_COORDINATE_ARGS,
    _PATCH_FUNCTION_ID,
    _PATCH_PARAMETER_NAME,
    _agent_owned_binding,
    _agent_owned_default_values,
    _agent_owned_function_id,
    _agent_owned_occurrence_arguments,
    _agent_owned_occurrence_span,
    _agent_owned_parameters,
    _agent_owned_semantic_analysis,
    _apply_authoring_line,
    _author_by_patches,
    _base_authoring_draft,
    _card_hit_node,
    _description_with_action_plan,
    _direct_source_authoring_plan,
    _draft_source_indices,
    _fact_step,
    _judgment_card,
    _materialize_agent_owned_workflow,
    _materialize_agent_parameters,
    _materialize_authoring_plan,
    _materialize_authoring_response,
    _materialize_authoring_workflow,
    _materialize_node_parameters,
    _parameter_literal_map,
    _parameters_for_occurrences,
    _parse_authoring_patches,
    _parse_patch_assignments,
    _proposal_from_draft,
    _recorded_node_value,
    _redact_parameter_literals,
    _span_indices,
    _validate_agent_owned_binding_spans,
    _validate_agent_owned_semantic_bindings,
)


def _default_authoring_workflow_prompt() -> str:
    """Return the patch contract. The compiler owns the Function document."""

    return """Register reusable pieces of one successful GUI run by returning patches.
Return one JSON object and no prose: {"patches":["..."]}.
The compiler starts from a base where every step is online_observation. Those
steps stay with the online Planner. You do not write functions, bindings objects,
execution_mode, or Store JSON.

Each patch is one line:
- span <start> <end> <function_id>
  Inclusive source_step_index range. The compiler marks steps in the range that
  are still online_observation as stable and registers that contiguous range as
  one Function. function_id matches [A-Za-z_][A-Za-z0-9_]{0,63}.
- step <index> task_parameter name=<name> arg=<arg_name>
  <arg_name> must be a key already listed in that step's args. The value stays
  the recorded value.
- step <index> task_parameter name=<name> node=<node_id> attr=<text|content-desc>
  Use this for a click or long_press whose identity is the named node. node_id
  must be the id on that step's hit, or an id you can see is not required: it
  must already belong to that step.
- step <index> online_observation
  Leave this step to the Planner. If it sits inside a span, the compiler splits
  that span around it.

A task_parameter step must lie inside a span. Put the span line before the step
line. Unlisted steps stay online. Do not invent steps, argument names, or node
ids. A fixed control such as Save, Delete, OK, or a tab is stable, not a
parameter. A record chosen by a predicate such as cheapest, duplicate, or newest
is online_observation, not a parameter.
"""


def compile_runlog_to_store(
    run_log: str | Path | dict[str, Any],
    output_root: str | Path,
    *,
    function_bundle: dict[str, Any] | None = None,
    authoring_proposal: dict[str, Any] | None = None,
    prepare_only: bool = False,
    model: str | None = None,
    client: Any | None = None,
    prompt: str | None = None,
    timeout: float = 120.0,
    source_states: str | Path | dict[str, Any] | None = None,
    state_loader: Any | None = None,
    complete_json: Callable[[str], str] | None = None,
    package_resolver: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Register strict v2 Functions and their referenced source states.

    ``package_resolver`` maps a recorded app label to its Android package.  It
    is supplied by the benchmark/device integration (e.g. AndroidWorld); the
    kernel never imports an integration to obtain it.  Without a resolver the
    package observed after the recorded launch is used.
    """
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
                package_resolver=package_resolver,
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
    if prepare_only:
        if authoring_proposal is not None or function_bundle is not None or complete_json is not None or selected_model:
            raise ValueError("authoring_preparation_cannot_compile_or_call_model")
        return {"source_run": facts, "authoring_contract": authoring_prompt}
    if authoring_proposal is not None:
        if function_bundle is not None or client is not None or complete_json is not None or selected_model or prompt is not None:
            raise ValueError("agent_authoring_proposal_cannot_use_model_or_function_bundle")
        # The existing host Agent owns semantic reasoning and correction. This
        # path only validates/materializes its proposal through the one compiler.
        authored = _materialize_authoring_response(authoring_proposal, facts, candidate_map={})
        _validate_materialized_function_artifacts(authored, raw_payload=raw)
        authoring_attempt_trace.append({"attempt": 1, "accepted": True, "error": None})
    elif function_bundle is not None:
        if client is not None or prompt is not None or complete_json is not None:
            raise ValueError("function_bundle_cannot_use_author_model_options")
        authored = {
            "reason": "Registered the supplied Function Schema.",
            "bundle": json.loads(json.dumps(function_bundle, ensure_ascii=False)),
        }
    elif selected_model is None and complete_json is None:
        if client is not None or prompt is not None:
            raise ValueError("author_model_required_for_author_options")
        # Device RunLog registration must not require a second Kotlin
        # converter or a model round-trip.  The canonical facts above already
        # contain the official projected actions and immutable source-state
        # ids, so use the same Python authoring/materialization path with a
        # deterministic complete Function.
        authored = _raw_source_replay_authoring(facts)
    else:
        if complete_json is not None and client is not None:
            raise ValueError("authoring_transport_ambiguous")
        if client is None and complete_json is None:
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
        authored, authoring_error, raw_author_response = _author_by_patches(
            facts,
            complete_json=complete_json,
            client=client,
            selected_model=selected_model,
            timeout=float(timeout),
            authoring_prompt=authoring_prompt,
            usage=usage,
            authoring_attempt_trace=authoring_attempt_trace,
        )
        if authoring_error is not None or authored is None:
            failure = authoring_error or ValueError(
                "authoring_patches_produced_no_executable_function"
            )
            _write_authoring_failure(
                root,
                error=failure,
                model=selected_model,
                prompt=authoring_prompt,
                response=raw_author_response,
                usage=usage,
            )
            authored = _raw_source_replay_authoring(
                facts,
                rejected_error=str(failure) or type(failure).__name__,
            )
        else:
            _validate_materialized_function_artifacts(
                authored,
                raw_payload=raw,
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
    projection: dict[str, Any] = {
        "nodes": nodes,
        "action_coordinates": "normalized_0_1000",
        "node_bounds_coordinates": "pixels",
    }
    display = observation_display(observation)
    if display is not None:
        projection["display"] = {"width": display[0], "height": display[1]}
    return projection


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
        # A rejected semantic proposal must never turn into an executable
        # static replay. Keep the source sequence only as auditable evidence.
        agent_visible=rejected_error is None,
    )
    result = {
        "reason": (
            "Registered the unparameterized successful source replay."
            if rejected_error is None
            else "Preserved the rejected source replay as hidden evidence."
        ),
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
            "fallback_mode": (
                "raw_source_replay"
                if rejected_error is None
                else "hidden_raw_source_evidence"
            ),
            "complete_function_id": function_id,
        },
    }
    if rejected_error is not None:
        result["authoring_workflow"]["rejected_error"] = rejected_error
    return result


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
    package_resolver: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Persist the native package observed after a recorded app launch."""

    if str(action.get("tool") or "") != "open_app":
        return action
    package_name = str(action.get("args", {}).get("package_name") or "").strip()
    if "." in package_name and " " not in package_name:
        return action
    resolved_package = ""
    if package_resolver is not None:
        try:
            resolved_package = str(package_resolver(package_name) or "").strip()
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
