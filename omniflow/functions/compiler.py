from __future__ import annotations

import hashlib
from itertools import pairwise
import json
import os
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET

from omniflow.core.trajectory import canonicalize_run_log, state_id
from omniflow.functions.management import apply_parameters, parameter_candidates
from omniflow.runlog import project_androidworld_step_actions
from omniflow.runtime.checker import CheckerLibrary, validate_checker_rule


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
    payload = canonicalize_run_log(raw)
    goal = str(payload.get("goal") or "").strip()
    if not goal:
        raise ValueError("successful_source_goal_required")

    steps: list[dict[str, Any]] = []
    recovery_examples: list[dict[str, Any]] = []
    omitted_action_types: set[str] = set()
    previous_successful_step: dict[str, Any] | None = None
    for step in payload["steps"]:
        if not isinstance(step, dict):
            continue
        metadata = step.get("metadata") if isinstance(step.get("metadata"), dict) else {}
        result = step.get("result") if isinstance(step.get("result"), dict) else {}
        if result.get("success") is not True:
            continue
        if _is_transient_system_action(step):
            # Source collection can legitimately dismiss an incidental Android
            # crash/permission dialog before continuing.  That click is part
            # of collection recovery, not of the task's reusable semantics;
            # retaining it would make a clean target page fail transfer before
            # the Planner gets a chance to act.
            omitted_action_types.add("transient_system_dialog")
            previous_successful_step = step
            continue
        observation = step["observation"]
        before_state_id = state_id(observation)
        next_observation = step.get("next_observation")
        after_state_id = state_id(
            next_observation
            if isinstance(next_observation, dict)
            else observation
        )
        action_type = str(step.get("action", {}).get("action_type") or "")
        if action_type in {"answer", "status", "unknown"}:
            omitted_action_types.add(action_type)
            continue
        if (
            action_type in {"click", "double_tap", "long_press", "swipe"}
            and isinstance(next_observation, dict)
            and before_state_id == after_state_id
        ):
            # A successful gesture that leaves the native observation exactly
            # unchanged is not a reusable semantic capability.  Keeping it in
            # a complete Function can make OmniTransfer select a non-clickable
            # label or an inert container and abort an otherwise valid flow.
            # Preserve the omission in the facts so authoring can account for
            # it without turning the no-op into a recorded action.
            omitted_action_types.add(f"noop_{action_type}")
            previous_successful_step = step
            continue
        projected_actions = project_androidworld_step_actions(
            step,
            previous_step=previous_successful_step,
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
        if metadata.get("origin") == "checker":
            for action in projected_actions:
                example = {
                    "source_state_id": before_state_id,
                    "action": action,
                    "metadata": {
                        key: metadata[key]
                        for key in ("thinking", "summary")
                        if str(metadata.get(key) or "").strip()
                    },
                }
                trigger = str(metadata.get("checker_trigger") or "").strip()
                if trigger:
                    example["trigger"] = trigger
                recovery_examples.append(example)
            continue
        action_metadata = {
            key: metadata[key]
            for key in ("summary", "thinking", "action_description")
            if str(metadata.get(key) or "").strip()
        }
        if promoted_launcher_entry:
            action_metadata["fixed_open_app_package"] = True
        for action in projected_actions:
            action = _canonicalize_open_app_action(
                action,
                next_observation=next_observation,
            )
            steps.append(
                {
                    "source_step_index": len(steps),
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
        "status": "succeeded",
        "success": True,
        "steps": steps,
        "omitted_action_types": sorted(omitted_action_types),
    }
    facts["observation_dependent_handoff_indices"] = sorted(
        _generic_coordinate_surface_indices(facts)
    )
    default_bundle = _default_bundle(facts, recovery_examples)
    source_parameter_candidates = _source_parameter_candidates(facts)
    authoring_prompt = prompt or """Convert successful GUI source facts into a reusable Function plan.
Return exactly one object with this shape:
{"reason":"account for every source step and explain the composition","plan":{"functions":[{"function_id":"enter_requested_name","name":"Enter requested name","description":"Enter the name requested by the user.","source_step_indices":[6,7],"parameters":[{"name":"name","description":"Name requested by the user","source_step_index":6,"arg_name":"text"}]}],"complete_function":{"function_id":"complete_form","name":"Complete form","description":"Enter the requested name and submit.","source_step_indices":[6,7],"parameters":[{"name":"name","description":"Name requested by the user","source_step_index":6,"arg_name":"text"}]}}}

Do not output input_schema, bindings, steps, actions, coordinates, checker rules,
agent_visible, schema_version, arguments, or source_state_id. The compiler owns
all of them and materializes canonical omniflow.function.v2 artifacts from the
selected immutable source actions.

Inspect source_run in source_step_index order. The reason must account for every
source index. Actions already marked origin=checker were removed before this plan;
do not reconstruct them in the main flow. Within one Function, source_step_indices must be strictly increasing
and contiguous. Never omit a click immediately following input_text when that click
commits, submits, confirms, or advances the form; keep both in one Function.

If source_run.omitted_action_types contains answer or status, those terminal
outputs are intentionally not Function actions. The complete Function is only a
reusable prefix; its name and description must say that the Planner must observe
the returned page and provide the answer/status afterward. Do not claim that the
Function itself answered the task.
Actions listed as noop_* were successful gestures whose before and after native
observations were identical; keep them omitted and explain the omission rather
than reconstructing them as Function actions.

In functions, return zero or more reusable semantic actions or tightly coupled
contiguous groups. Then author exactly one complete_function as an ordinary Function.
When the successful RunLog starts with open_app, complete_function must start with
the first open_app and end with the terminal successful task action. It may omit
unsafe internal retries, setup noise, checker actions, and observation-dependent
repetitions while preserving that complete task envelope. A Function is atomic: if
an action result must be observed to calculate a later value, stop the reusable
Function before that action. Do not expose that later value as a Function
parameter. The Planner must observe the returned page and continue with ordinary
native tools or a separately recalled Function.
The complete Function must lift its goal-dependent values into parameters, merge the
selected meanings into one coherent name and description, and never merely hard-code
the successful instance values. Do not invent a nesting or parent/child schema.
A Function call is atomic: the Planner observes only after its last step. Never
encode repetition count when the task requires reading changing UI after each
repeat. Keep one representative action as a one-step Function and let the Planner
call it repeatedly.

Do not put a coordinate-only action on a generic canvas, background, grid cell,
map surface, drawing surface, or other reusable container into a cross-instance
Function when the coordinate is what identifies the date, item, or value. A
resource id or label such as `month_view_background` is not enough semantic
grounding for the selected cell. Treat that action as an observation-dependent
handoff boundary: stop the Function before it, and let the Planner inspect the
current page and choose the visible target with native tools. Never preserve the
source coordinate merely because the source RunLog succeeded.

Do not reinterpret onboarding, installers, permissions, ads, errors, waits, or
navigation accidents as standalone capabilities. Omit unsafe or unclear actions
and explain each omission. Preserve the successful source order.

Parameterize only entries copied exactly from parameter_candidates, and only when
the same Function selects that source_step_index. Never parameterize an
observation-dependent input: such a step is a runtime handoff boundary and any
Function containing it is omitted by the compiler. Choose a stable identifier name
and concise description. Every (source_step_index, arg_name) pair must occur
literally in parameter_candidates; never invent file, folder, click-count, integer,
coordinate, or other parameters absent from that list. Usually target_description
is a stable UI label, not a goal-dependent value. The complete_function must repeat
every parameter target selected by a semantic Function. Use parameters=[] for fixed
recorded values. Coordinates never appear in candidates and can never become
Function inputs. Keep reason under 40 words, each description under 20 words, and
return no prose outside the JSON object.

For a global Function whose first action is open_app, keep the canonical recorded
package fixed when the goal identifies a concrete app such as Joplin or Settings.
Only expose package_name when the goal explicitly asks the caller to choose an
app or package; a model must never invent an app package from a friendly app name.
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
            "reason": "Registered offline Codex-authored Functions.",
            "bundle": json.loads(json.dumps(function_bundle, ensure_ascii=False)),
        }
    elif selected_model is None:
        if client is not None or prompt is not None:
            raise ValueError("author_model_required_for_author_options")
        if default_bundle is None:
            raise ValueError("default_bundle_actions_required")
        authored = {
            "reason": (
                "Registered the safe reusable prefix; the Planner resumes at the "
                "observation-dependent handoff."
                if (
                    _observation_dependent_input_indices(facts)
                    or _observation_dependent_handoff_indices(facts)
                )
                else "Registered one complete recorded Function."
            ),
            "bundle": default_bundle,
        }
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
            max_tokens=512,
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
        completion_tokens = int(
            getattr(response_usage, "completion_tokens", 0) or 0
        )
        total_tokens = int(getattr(response_usage, "total_tokens", 0) or 0)
        usage = {
            "model_calls": 1,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
            or prompt_tokens + completion_tokens,
        }
        raw_author_response = str(response.choices[0].message.content or "")
        try:
            proposal = json.loads(raw_author_response)
            authored = _materialize_authoring_plan(proposal, facts)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            _write_authoring_failure(
                root,
                error=error,
                model=selected_model,
                prompt=authoring_prompt,
                response=raw_author_response,
                usage=usage,
            )
            if default_bundle is None:
                raise
            authored = {
                "reason": (
                    "Authoring proposal was unusable; registered the complete "
                    f"schema-valid recorded Function instead ({type(error).__name__})."
                ),
                "bundle": default_bundle,
            }
    if not isinstance(authored, dict) or set(authored) != {"reason", "bundle"}:
        raise ValueError("function_author_response_contract_invalid")
    if not isinstance(authored["reason"], str):
        raise ValueError("function_author_reason_must_be_string")

    bundle = authored["bundle"]
    if bundle is None:
        raise ValueError("functions_required")
    if not isinstance(bundle, dict):
        raise ValueError("function_author_bundle_must_be_object_or_null")
    if (
        selected_model is not None
        and function_bundle is None
        and default_bundle is not None
        and isinstance(bundle.get("functions"), list)
        and not bundle["functions"]
    ):
        authored = {
            "reason": (
                "The authored plan contained no safe tool Function at the runtime "
                "handoff; registered the safe reusable prefix instead."
            ),
            "bundle": default_bundle,
        }
        bundle = default_bundle
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
    restored_commit_steps = _restore_post_input_commit_steps(
        raw_functions,
        facts["steps"],
    )
    if restored_commit_steps:
        authored["reason"] = (
            f"{authored['reason']} Compiler restored {restored_commit_steps} "
            "successful post-input commit action(s)."
        )
    try:
        functions = [parse_function_artifact(value) for value in raw_functions]
    except ValueError as error:
        if selected_model is None or function_bundle is not None or default_bundle is None:
            raise
        authored = {
            "reason": (
                "Authored Functions did not satisfy the runtime schema; registered "
                f"the complete schema-valid recorded Function instead ({type(error).__name__})."
            ),
            "bundle": default_bundle,
        }
        bundle = default_bundle
        raw_functions = list(bundle["functions"])
        arguments_by_function = dict(bundle["arguments"])
        raw_checker_rules = list(bundle["checker_rules"])
        functions = [parse_function_artifact(value) for value in raw_functions]
    checker_rules = [validate_checker_rule(rule) for rule in raw_checker_rules]
    checker_ids = [rule["id"] for rule in checker_rules]
    if len(checker_ids) != len(set(checker_ids)):
        raise ValueError("function_bundle_duplicate_checker_id")
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
    checker_store_path = root / "checker_store.json"
    CheckerLibrary(tuple(checker_rules)).save(checker_store_path)
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
        "checker_store_path": str(checker_store_path),
        "checker_count": len(checker_rules),
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
    function_view = {
        "bindings": [],
        "steps": [
            {
                "step_index": index,
                "action": step["action"],
                "metadata": step.get("metadata") or {},
            }
            for index, step in enumerate(facts.get("steps") or ())
        ],
    }
    candidates = [
        {
            "source_step_index": candidate["step_index"],
            "tool": candidate["tool"],
            "arg_name": candidate["arg_name"],
            "recorded_value": candidate["recorded_value"],
        }
        for candidate in parameter_candidates(function_view)
    ]
    fixed_open_app_steps = {
        int(step["step_index"])
        for step in function_view["steps"]
        if isinstance(step.get("metadata"), dict)
        and step["metadata"].get("fixed_open_app_package") is True
    }
    return [
        candidate
        for candidate in candidates
        if not (
            candidate["tool"] == "open_app"
            and candidate["arg_name"] == "package_name"
            and candidate["source_step_index"] in fixed_open_app_steps
        )
    ]


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
    if not isinstance(raw_complete_function, dict):
        raise ValueError("function_author_plan_complete_function_required")

    source_steps = list(facts.get("steps") or ())
    candidates = {
        (candidate["source_step_index"], candidate["arg_name"]): candidate
        for candidate in _source_parameter_candidates(facts)
    }
    observation_dependent_input_indices = _observation_dependent_input_indices(facts)
    functions: list[dict[str, Any]] = []
    arguments: dict[str, dict[str, Any]] = {}
    selected_source_indices: set[int] = set()
    semantic_parameter_targets: set[tuple[int, str]] = set()
    semantic_parameter_specs: dict[tuple[int, str], dict[str, Any]] = {}
    complete_parameter_targets: set[tuple[int, str]] = set()
    complete_source_indices: set[int] = set()
    materialization_notes: list[str] = []
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
    planned_functions = [
        *((raw_function, False) for raw_function in raw_functions),
        (raw_complete_function, True),
    ]
    for raw_function, is_complete in planned_functions:
        if not isinstance(raw_function, dict) or set(raw_function) != function_fields:
            raise ValueError("function_author_plan_function_contract_invalid")
        function_id = str(raw_function.get("function_id") or "").strip()
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
        skip_function = False
        if (
            indices != sorted(set(indices))
            or indices[0] < 0
            or indices[-1] >= len(source_steps)
            or (
                not is_complete
                and indices != list(range(indices[0], indices[-1] + 1))
            )
        ):
            raise ValueError("function_author_plan_source_steps_invalid")
        source_starts_with_open_app = bool(source_steps) and (
            source_steps[0].get("action", {}).get("tool") == "open_app"
        )
        handoff_indices = _observation_dependent_handoff_indices(facts)
        if is_complete and observation_dependent_input_indices:
            dynamic_indices = [
                index
                for index in indices
                if index in observation_dependent_input_indices
            ]
            if dynamic_indices:
                boundary = dynamic_indices[0]
                safe_indices = [index for index in indices if index < boundary]
                if safe_indices:
                    indices = safe_indices
                    description = (
                        f"{description} Stop before the value-dependent input "
                        "so the Planner can observe and compute its value."
                    )
                    materialization_notes.append(
                        "Compiler split the global Function at an "
                        f"observation-dependent input (source step {boundary})."
                    )
                else:
                    skip_function = True
                    materialization_notes.append(
                        "Compiler omitted a global Function with no safe prefix; "
                        "the Planner owns the runtime handoff."
                    )
        elif (
            not is_complete
            and observation_dependent_input_indices
            and any(
                index in observation_dependent_input_indices for index in indices
            )
        ):
            skip_function = True
            materialization_notes.append(
                "Compiler omitted a semantic Function containing an "
                "observation-dependent input; the Planner owns that runtime handoff."
            )
        if skip_function:
            continue
        if handoff_indices:
            generic_indices = [
                index for index in indices if index in handoff_indices
            ]
            if generic_indices:
                if is_complete:
                    boundary = min(generic_indices)
                    safe_indices = [index for index in indices if index < boundary]
                    if safe_indices:
                        indices = safe_indices
                        description = (
                            f"{description} Stop before the generic surface so "
                            "the Planner can inspect the current page and choose "
                            "the target."
                        )
                        materialization_notes.append(
                            "Compiler split the global Function at a generic "
                            f"coordinate surface (source step {boundary})."
                        )
                    else:
                        skip_function = True
                        materialization_notes.append(
                            "Compiler omitted a global Function with no safe "
                            "prefix before a generic coordinate surface."
                        )
                else:
                    skip_function = True
                    materialization_notes.append(
                        "Compiler omitted a semantic Function containing a "
                        "generic coordinate surface; the Planner owns the "
                        "runtime handoff."
                    )
            if skip_function:
                continue
        if is_complete:
            restored_indices = _restore_omitted_complete_actions(
                indices,
                source_steps,
                observation_dependent_input_indices,
                excluded_indices=handoff_indices,
            )
            if restored_indices:
                indices = sorted(set(indices).union(restored_indices))
                materialization_notes.append(
                    "Compiler restored omitted executable source steps "
                    f"{restored_indices} in the complete Function."
                )
        atomicized_count = 0
        if not (is_complete and source_starts_with_open_app):
            (
                indices,
                function_id,
                name,
                description,
                atomicized_count,
            ) = _atomicize_repeated_click_function(
                indices,
                source_steps,
                function_id=function_id,
                name=name,
                description=description,
            )
        if is_complete and source_starts_with_open_app:
            description = (
                f"{description} This is the global workflow entry; call it "
                "directly from the launcher because it opens the app and owns "
                "the startup navigation prefix."
            )
        if atomicized_count:
            materialization_notes.append(
                f"Compiler reduced {atomicized_count} identical clicks in "
                f"{function_id} to one atomic step so the Planner observes "
                "after every click."
            )
        if not is_complete:
            if selected_source_indices.intersection(indices):
                raise ValueError("function_author_plan_source_step_reused")
            selected_source_indices.update(indices)
        else:
            complete_source_indices.update(indices)

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
                        source_steps[source_index]["before_state_id"]
                    ),
                    "action": json.loads(
                        json.dumps(
                            source_steps[source_index]["action"],
                            ensure_ascii=False,
                        )
                    ),
                }
                for local_index, source_index in enumerate(indices)
            ],
            "agent_visible": True,
        }
        if is_complete:
            function["description"] = _append_terminal_handoff_description(
                function["description"],
                facts,
            )
        raw_parameters = raw_function.get("parameters")
        if not isinstance(raw_parameters, list):
            raise ValueError("function_author_plan_parameters_invalid")
        if is_complete:
            # The complete Function is the public API envelope.  Authoring
            # models sometimes select a parameter on a semantic sub-function
            # but forget to repeat it on complete_function.  Repair that
            # omission from the already validated semantic proposal instead of
            # falling back to a hard-coded recorded Function.
            explicit_targets = {
                (
                    parameter.get("source_step_index"),
                    str(parameter.get("arg_name") or "").strip(),
                )
                for parameter in raw_parameters
                if isinstance(parameter, dict)
            }
            raw_parameters = [*raw_parameters]
            for target, parameter in semantic_parameter_specs.items():
                if target[0] in indices and target not in explicit_targets:
                    raw_parameters.append(dict(parameter))
                    materialization_notes.append(
                        "Compiler copied a validated semantic parameter onto "
                        "the complete Function API."
                    )
            for automatic_parameter in _goal_semantic_parameters(
                facts,
                indices=indices,
                candidates=candidates,
                existing_targets=explicit_targets
                | set(semantic_parameter_specs),
                existing_names={
                    str(parameter.get("name") or "").strip()
                    for parameter in raw_parameters
                    if isinstance(parameter, dict)
                },
            ):
                raw_parameters.append(automatic_parameter)
                materialization_notes.append(
                    "Compiler exposed a source label explicitly named by the "
                    "goal as a public Function API parameter."
                )
        parameter_proposals: list[dict[str, Any]] = []
        source_arguments: dict[str, Any] = {}
        for parameter in raw_parameters:
            if not isinstance(parameter, dict) or set(parameter) != parameter_fields:
                raise ValueError("function_author_plan_parameter_contract_invalid")
            source_index = parameter.get("source_step_index")
            arg_name = str(parameter.get("arg_name") or "").strip()
            candidate = candidates.get((source_index, arg_name))
            if candidate is None:
                raise ValueError("function_author_plan_parameter_target_invalid")
            if arg_name == "package_name" and not _goal_allows_dynamic_app_package(
                facts.get("goal")
            ):
                materialization_notes.append(
                    "Compiler kept the concrete app package fixed instead of "
                    "exposing package_name as a public Function input."
                )
                continue
            if source_index not in indices:
                # A complete Function may be truncated at an observation
                # boundary. Its parameter remains owned by the later
                # semantic Function selected after the observation.
                if is_complete:
                    continue
                # Authoring models occasionally copy a public parameter onto
                # a semantic sub-function without including the source step
                # in that sub-function.  The complete Function still owns
                # the public API, so discard only this misplaced sub-function
                # proposal instead of rejecting the entire authoring plan.
                materialization_notes.append(
                    "Compiler dropped a parameter target not selected by its "
                    f"semantic Function (source step {source_index})."
                )
                continue
            parameter_name = str(parameter.get("name") or "").strip()
            parameter_target = (int(source_index), arg_name)
            if is_complete:
                complete_parameter_targets.add(parameter_target)
            else:
                semantic_parameter_targets.add(parameter_target)
                semantic_parameter_specs.setdefault(
                    parameter_target,
                    {
                        "name": parameter_name,
                        "description": str(parameter.get("description") or "").strip(),
                        "source_step_index": int(source_index),
                        "arg_name": arg_name,
                    },
                )
            parameter_proposals.append(
                {
                    "name": parameter_name,
                    "description": str(parameter.get("description") or "").strip(),
                    "step_index": indices.index(source_index),
                    "arg_name": arg_name,
                }
            )
            source_arguments[parameter_name] = candidate["recorded_value"]
        apply_parameters(function, parameter_proposals, facts)
        functions.append(function)
        arguments[function_id] = source_arguments

    required_complete_parameters = {
        target
        for target in semantic_parameter_targets
        if target[0] in complete_source_indices
    }
    if required_complete_parameters - complete_parameter_targets:
        raise ValueError("function_author_plan_complete_parameters_missing")

    normalized_reason = reason.strip()
    if materialization_notes:
        normalized_reason = f"{normalized_reason} {' '.join(materialization_notes)}"
    return {
        "reason": normalized_reason,
        "bundle": {
            "schema_version": "omniflow.function-bundle.v2",
            "run_id": str(facts["run_id"]),
            "arguments": arguments,
            "checker_rules": [],
            "functions": functions,
        },
    }


def _observation_dependent_input_indices(facts: dict[str, Any]) -> frozenset[int]:
    """Find input values that cannot be supplied before the task is observed."""

    goal = " ".join(str(facts.get("goal") or "").casefold().split())
    indices: set[int] = set()
    for index, step in enumerate(facts.get("steps") or ()):
        action = step.get("action") if isinstance(step, dict) else None
        if not isinstance(action, dict) or action.get("tool") != "input_text":
            continue
        value = " ".join(str(action.get("args", {}).get("text") or "").casefold().split())
        if value and value not in goal:
            indices.add(index)
    return frozenset(indices)


def _goal_semantic_parameters(
    facts: dict[str, Any],
    *,
    indices: list[int],
    candidates: dict[tuple[int, str], dict[str, Any]],
    existing_targets: set[tuple[int, str]],
    existing_names: set[str],
) -> list[dict[str, Any]]:
    """Expose goal-named labels that are part of the public task API.

    Authoring should normally choose these bindings.  The deterministic rule
    covers the important failure mode where a model describes a selected
    folder/item correctly but leaves the literal source label hard-coded.
    It only applies when the exact label appears in the goal and the goal has
    an unambiguous semantic slot (folder, file, contact, etc.); navigation
    labels such as Sidebar are therefore left fixed.
    """
    goal = " ".join(str(facts.get("goal") or "").casefold().split())
    result: list[dict[str, Any]] = []
    for source_index in indices:
        candidate = candidates.get((source_index, "target_description"))
        if candidate is None or (source_index, "target_description") in existing_targets:
            continue
        value = " ".join(str(candidate.get("recorded_value") or "").casefold().split())
        if not value or len(value) < 2 or value not in goal:
            continue
        parameter_name, description = _goal_parameter_slot(goal)
        if not parameter_name or parameter_name in existing_names:
            continue
        result.append(
            {
                "name": parameter_name,
                "description": description,
                "source_step_index": source_index,
                "arg_name": "target_description",
            }
        )
        existing_targets.add((source_index, "target_description"))
        existing_names.add(parameter_name)
    return result


def _goal_parameter_slot(goal: str) -> tuple[str, str]:
    slots = (
        ("folder", "Folder or notebook named by the user."),
        ("directory", "Directory named by the user."),
        ("file", "File named by the user."),
        ("contact", "Contact named by the user."),
        ("person", "Person named by the user."),
        ("recipe", "Recipe named by the user."),
        ("event", "Event named by the user."),
        ("task", "Task named by the user."),
        ("note", "Note named by the user."),
        ("app", "App named by the user."),
    )
    for marker, description in slots:
        if re.search(rf"\b{re.escape(marker)}\b", goal):
            return marker, description
    return "", ""


_GENERIC_COORDINATE_SURFACE_MARKERS = frozenset(
    {
        "background",
        "canvas",
        "calendar background",
        "calendar canvas",
        "calendar grid",
        "drawing canvas",
        "drawing surface",
        "grid cell",
        "map surface",
        "map view",
        "month view background",
        "month view grid",
        "webview",
        "web view",
    }
)


def _generic_coordinate_surface_indices(
    facts: dict[str, Any],
) -> frozenset[int]:
    """Find coordinate actions whose meaning is only the current surface."""

    indices: set[int] = set()
    for index, step in enumerate(facts.get("steps") or ()):
        action = step.get("action") if isinstance(step, dict) else None
        if not isinstance(action, dict) or action.get("tool") not in {
            "click",
            "double_click",
            "long_press",
        }:
            continue
        args = action.get("args")
        if not isinstance(args, dict) or args.get("x") is None or args.get("y") is None:
            continue
        target = " ".join(
            str(args.get("target_description") or "")
            .replace("_", " ")
            .casefold()
            .split()
        )
        if target in _GENERIC_COORDINATE_SURFACE_MARKERS:
            indices.add(index)
    return frozenset(indices)


def _observation_dependent_handoff_indices(
    facts: dict[str, Any],
) -> frozenset[int]:
    raw = facts.get("observation_dependent_handoff_indices")
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return _generic_coordinate_surface_indices(facts)
    return frozenset(
        int(index)
        for index in raw
        if isinstance(index, int) and not isinstance(index, bool)
    )


def _goal_allows_dynamic_app_package(goal: Any) -> bool:
    normalized = " ".join(str(goal or "").casefold().split())
    return any(
        marker in normalized
        for marker in (
            "requested app",
            "choose an app",
            "select an app",
            "app package",
            "package name",
            "package_name",
        )
    )


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
    after_package = _primary_observation_package(next_observation)
    if not after_package:
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


def _restore_omitted_complete_actions(
    indices: list[int],
    source_steps: list[dict[str, Any]],
    observation_dependent_input_indices: frozenset[int],
    *,
    excluded_indices: frozenset[int] = frozenset(),
) -> list[int]:
    """Restore essential recorded actions accidentally omitted by authoring."""

    boundary = min(observation_dependent_input_indices, default=len(source_steps))
    selected = set(indices)
    essential_tools = {
        "open_app",
        "click",
        "double_click",
        "long_press",
        "input_text",
        "swipe",
        "press_key",
    }
    return [
        index
        for index, step in enumerate(source_steps)
        if index < boundary
        and index not in selected
        and index not in excluded_indices
        and str((step.get("action") or {}).get("tool") or "") in essential_tools
    ]


def _atomicize_repeated_click_function(
    indices: list[int],
    source_steps: list[dict[str, Any]],
    *,
    function_id: str,
    name: str,
    description: str,
) -> tuple[list[int], str, str, str, int]:
    actions = [source_steps[index].get("action") for index in indices]
    observation_dependent_repeat = False
    run_start = 0
    while run_start < len(actions):
        run_end = run_start + 1
        while run_end < len(actions) and actions[run_end] == actions[run_start]:
            run_end += 1
        action = actions[run_start]
        if (
            run_end - run_start > 1
            and isinstance(action, dict)
            and action.get("tool") == "click"
        ):
            state_ids = {
                str(source_steps[indices[position]].get("before_state_id") or "")
                for position in range(run_start, run_end)
            }
            if len(state_ids) > 1:
                observation_dependent_repeat = True
                break
        run_start = run_end
    if not observation_dependent_repeat:
        return indices, function_id, name, description, 0
    if not actions or any(action != actions[0] for action in actions[1:]):
        raise ValueError("function_author_plan_repeated_click_must_be_atomic")

    atomic_id = re.sub(r"_(?:\d+_)?times?$", "", function_id, flags=re.IGNORECASE)
    if atomic_id == "click":
        atomic_id = "click_recorded_button"
    atomic_name = re.sub(
        r"\s+\d+\s+times?\b",
        "",
        name,
        flags=re.IGNORECASE,
    ).strip()
    atomic_description = re.sub(
        r"\s+\d+\s+times?\b",
        "",
        description,
        flags=re.IGNORECASE,
    ).strip()
    return (
        [indices[0]],
        atomic_id or "click_recorded_button",
        atomic_name or "Click recorded button",
        atomic_description
        or "Click the recorded button once, then return to the Planner.",
        len(indices),
    )


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


def _restore_post_input_commit_steps(
    raw_functions: list[Any],
    source_steps: list[dict[str, Any]],
) -> int:
    """Restore an authored-away source click that immediately follows input."""
    restored = 0
    for source_step, commit_step in pairwise(source_steps):
        source_action = source_step.get("action")
        commit_action = commit_step.get("action")
        if not (
            isinstance(source_action, dict)
            and source_action.get("tool") == "input_text"
            and isinstance(commit_action, dict)
            and commit_action.get("tool") == "click"
        ):
            continue
        commit_state_id = str(commit_step.get("before_state_id") or "").strip()
        if not commit_state_id:
            continue
        input_state_id = str(source_step.get("before_state_id") or "").strip()
        for function in raw_functions:
            if not isinstance(function, dict):
                continue
            candidate_steps = function.get("steps")
            if not isinstance(candidate_steps, list) or not candidate_steps:
                continue
            last_step = candidate_steps[-1]
            if (
                isinstance(last_step, dict)
                and last_step.get("source_state_id") == input_state_id
                and isinstance(last_step.get("action"), dict)
                and last_step["action"].get("tool") == "input_text"
            ):
                candidate_steps.append(
                    {
                        "step_index": len(candidate_steps),
                        "source_state_id": commit_state_id,
                        "action": json.loads(
                            json.dumps(commit_action, ensure_ascii=False)
                        ),
                    }
                )
                restored += 1
    return restored


def _default_bundle(
    facts: dict[str, Any],
    recovery_examples: list[dict[str, Any]],
) -> dict[str, Any] | None:
    source_steps = list(facts.get("steps") or ())
    if not source_steps:
        return None
    dynamic_indices = _observation_dependent_input_indices(facts)
    handoff_indices = _observation_dependent_handoff_indices(facts)
    boundary_indices = dynamic_indices.union(handoff_indices)
    if boundary_indices:
        boundary = min(boundary_indices)
        safe_prefix = source_steps[:boundary]
        if not safe_prefix:
            safe_prefix = [
                step
                for index, step in enumerate(source_steps)
                if index not in boundary_indices
            ][:1]
        if not safe_prefix:
            return None
        prefix_facts = dict(facts)
        prefix_facts["steps"] = safe_prefix
        function = _complete_function_artifact(prefix_facts)
        if boundary in dynamic_indices:
            handoff_description = (
                "Stop before the value-dependent input and let the Planner "
                "continue from the observed page."
            )
        else:
            handoff_description = (
                "Stop before the generic surface and let the Planner inspect "
                "the current page and choose the target."
            )
        function["description"] = f"{function['description']} {handoff_description}"
    else:
        function = _complete_function_artifact(facts)
    function["description"] = _append_terminal_handoff_description(
        function["description"],
        facts,
    )
    function_id = function["function_id"]
    return {
        "schema_version": "omniflow.function-bundle.v2",
        "run_id": facts["run_id"],
        "arguments": {function_id: {}},
        "checker_rules": [],
        "functions": [function],
    }


def _complete_function_artifact(
    facts: dict[str, Any],
    *,
    existing_function_ids: set[str] | None = None,
) -> dict[str, Any]:
    source_steps = list(facts.get("steps") or ())
    if not source_steps:
        raise ValueError("complete_function_source_actions_required")
    steps = [
        {
            "step_index": index,
            "source_state_id": str(step["before_state_id"]),
            "action": json.loads(json.dumps(step["action"], ensure_ascii=False)),
        }
        for index, step in enumerate(source_steps)
    ]
    digest = hashlib.sha256(
        json.dumps(
            {"goal": facts["goal"], "steps": steps},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:12]
    base_function_id = f"complete_recorded_{digest}"
    used_ids = existing_function_ids or set()
    function_id = base_function_id
    suffix = 2
    while function_id in used_ids:
        function_id = f"{base_function_id}_{suffix}"
        suffix += 1
    goal = str(facts["goal"])
    return {
        "schema_version": "omniflow.function.v2",
        "function_id": function_id,
        "name": goal[:120],
        "description": f"Execute the complete recorded workflow: {goal}",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "bindings": [],
        "steps": steps,
        "agent_visible": True,
    }


def _append_terminal_handoff_description(
    description: str,
    facts: dict[str, Any],
) -> str:
    omitted = {
        str(value).strip().lower()
        for value in facts.get("omitted_action_types") or ()
    }
    if not omitted.intersection({"answer", "status"}):
        return description
    suffix = (
        " This Function only reaches the observed page; the Planner must "
        "inspect it and provide the task answer or status afterward."
    )
    return description if description.endswith(suffix) else f"{description}{suffix}"


def _is_transient_system_action(step: dict[str, Any]) -> bool:
    """Return whether a successful source click only dismisses system noise."""
    action = step.get("action") if isinstance(step.get("action"), dict) else {}
    if str(action.get("action_type") or "") != "click":
        return False
    index = action.get("index")
    if isinstance(index, bool) or not isinstance(index, int):
        return False
    observation = step.get("observation")
    if not isinstance(observation, dict):
        return False
    xml = observation.get("xml")
    if not isinstance(xml, str) or not xml.strip():
        return False
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return False
    nodes = root.iter("node")
    for node in nodes:
        if str(node.get("id") or "") != str(index):
            continue
        resource_id = str(node.get("resource-id") or "")
        package = str(node.get("package") or "")
        text = str(node.get("text") or "").strip().lower()
        if package == "android" and (
            resource_id.startswith("android:id/aerr_")
            or text in {"close app", "app info", "wait"}
        ):
            return True
        break
    return False
